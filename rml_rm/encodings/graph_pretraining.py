"""Graph dynamics pretraining for frozen RML monitor encoders."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

from rml_rm.encodings.graph_models import (
    TRANSITION_LABELS,
    GraphEncoderConfig,
    RMLGraphBatch,
    RMLGraphDynamicsPredictor,
)
from rml_rm.encodings.rml_graph import (
    RMLGraphData,
    build_edge_type_vocab,
    build_node_kind_vocab,
    build_node_value_vocab,
    normalize_generated_variables,
    rml_to_graph,
)
from rml_rm.experiments.runtime import (
    configure_torch_seed,
    json_ready,
    resolve_torch_device,
    write_json,
    write_jsonl,
)

GRAPH_STRUCTURAL_FEATURE_DIM = 15


@dataclass(frozen=True)
class GNNDynamicsConfig:
    dataset_path: Path
    output_dir: Path
    experiment_name: str = "rml_graph_dynamics"
    seed: int = 0
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    max_grad_norm: float = 5.0
    node_embedding_dim: int = 32
    node_value_embedding_dim: int = 0
    node_value_dropout: float = 0.0
    hidden_dim: int = 64
    output_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    output_layer_norm: bool = False
    output_l2_normalize: bool = False
    event_embedding_dim: int = 16
    phase_loss_weight: float = 1.0
    phase_class_weighting: bool = False
    balanced_phase_sampling: bool = False
    use_graph_structural_features: bool = False
    prefer_normalized_monitor_state: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class DynamicsExample:
    previous_state: str
    event: str
    next_state: str
    graph: RMLGraphData
    event_id: int
    transition_label: int
    phase_label: int
    structural_features: tuple[float, ...] = ()


class DynamicsDataset(Dataset):
    def __init__(self, examples: list[DynamicsExample]) -> None:
        if not examples:
            raise ValueError("DynamicsDataset requires at least one example.")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DynamicsExample:
        return self.examples[index]


def train_gnn_dynamics(config: GNNDynamicsConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"experiment": config.experiment_name, "config": asdict(config)})
    configure_torch_seed(config.seed)
    device = resolve_torch_device(config.device)

    examples, event_vocab, phase_vocab = load_monitor_corpus_examples(
        config.dataset_path,
        prefer_normalized_monitor_state=config.prefer_normalized_monitor_state,
        use_graph_structural_features=config.use_graph_structural_features,
    )
    dataset = DynamicsDataset(examples)
    graphs = [example.graph for example in examples]
    node_kind_vocab = build_node_kind_vocab(graphs)
    node_value_vocab = build_node_value_vocab(graphs)
    edge_type_vocab = build_edge_type_vocab(graphs)

    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Not enough graph examples for a train/validation split.")
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    collate = _DynamicsCollator(
        node_kind_vocab=node_kind_vocab,
        node_value_vocab=node_value_vocab,
        edge_type_vocab=edge_type_vocab,
        device=device,
        include_structural_features=config.use_graph_structural_features,
    )
    sampler = _balanced_phase_sampler(train_dataset, seed=config.seed) if config.balanced_phase_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    graph_config = GraphEncoderConfig(
        node_embedding_dim=config.node_embedding_dim,
        node_value_embedding_dim=config.node_value_embedding_dim,
        node_value_dropout=config.node_value_dropout,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_layer_norm=config.output_layer_norm,
        output_l2_normalize=config.output_l2_normalize,
    )
    model = RMLGraphDynamicsPredictor(
        num_node_kinds=len(node_kind_vocab),
        num_node_values=len(node_value_vocab),
        num_edge_types=len(edge_type_vocab),
        num_events=len(event_vocab),
        graph_config=graph_config,
        event_embedding_dim=config.event_embedding_dim,
        num_phase_labels=len(phase_vocab),
        structural_feature_dim=GRAPH_STRUCTURAL_FEATURE_DIM if config.use_graph_structural_features else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    phase_loss = nn.CrossEntropyLoss(
        weight=_phase_class_weights(train_dataset, len(phase_vocab), device=device)
        if config.phase_class_weighting
        else None
    )

    best_val_loss = math.inf
    best_epoch = -1
    best_path = output_dir / "best_dynamics_encoder.pt"
    metrics: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_fn=ce_loss,
            phase_loss_fn=phase_loss,
            phase_loss_weight=config.phase_loss_weight,
            max_grad_norm=config.max_grad_norm,
            num_phase_labels=len(phase_vocab),
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            loss_fn=ce_loss,
            phase_loss_fn=phase_loss,
            phase_loss_weight=config.phase_loss_weight,
            max_grad_norm=config.max_grad_norm,
            num_phase_labels=len(phase_vocab),
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        metrics.append(record)
        write_jsonl(output_dir / "metrics.jsonl", metrics, sort_keys=True)
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            _save_checkpoint(
                best_path,
                model=model,
                config=config,
                graph_config=graph_config,
                node_kind_vocab=node_kind_vocab,
                node_value_vocab=node_value_vocab,
                edge_type_vocab=edge_type_vocab,
                event_vocab=event_vocab,
                phase_vocab=phase_vocab,
                epoch=epoch,
                metrics=record,
            )

    final_path = output_dir / "dynamics_encoder_final.pt"
    _save_checkpoint(
        final_path,
        model=model,
        config=config,
        graph_config=graph_config,
        node_kind_vocab=node_kind_vocab,
        node_value_vocab=node_value_vocab,
        edge_type_vocab=edge_type_vocab,
        event_vocab=event_vocab,
        phase_vocab=phase_vocab,
        epoch=config.epochs,
        metrics=metrics[-1],
    )
    summary = {
        "config": json_ready(asdict(config)),
        "dataset": {
            "path": str(config.dataset_path),
            "examples": len(examples),
            "train_examples": train_size,
            "validation_examples": validation_size,
        },
        "graph": {
            "node_value_embedding_dim": config.node_value_embedding_dim,
            "structural_feature_dim": GRAPH_STRUCTURAL_FEATURE_DIM if config.use_graph_structural_features else 0,
            "num_node_kinds": len(node_kind_vocab),
            "num_node_values": len(node_value_vocab),
            "num_edge_types": len(edge_type_vocab),
            "event_vocab": event_vocab,
            "phase_vocab": phase_vocab,
            "transition_label_counts": _label_counts(examples),
            "phase_label_counts": _phase_counts(examples, phase_vocab),
        },
        "best": {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "encoder_path": str(best_path),
        },
        "final_metrics": metrics[-1],
        "artifacts": {
            "config": str(output_dir / "config.json"),
            "best_encoder": str(best_path),
            "final_encoder": str(final_path),
            "metrics": str(output_dir / "metrics.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def load_monitor_corpus_examples(
    dataset_path: Path,
    *,
    prefer_normalized_monitor_state: bool = True,
    use_graph_structural_features: bool = False,
) -> tuple[list[DynamicsExample], dict[str, int], dict[str, int]]:
    rows = _read_jsonl(dataset_path)
    event_vocab = {"<UNK>": 0}
    phase_vocab = {"<UNK>": 0}
    examples: list[DynamicsExample] = []
    previous_by_trace: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in sorted(rows, key=_row_sort_key):
        trace_key = _trace_key(row)
        previous_row = previous_by_trace.get(trace_key)
        previous_by_trace[trace_key] = row
        if previous_row is None:
            continue

        previous_state = _monitor_state_for_graph(
            previous_row,
            prefer_normalized_monitor_state=prefer_normalized_monitor_state,
        )
        next_state = _monitor_state_for_graph(
            row,
            prefer_normalized_monitor_state=prefer_normalized_monitor_state,
        )
        event = _normalize_event(row.get("event", "_"))
        if event not in event_vocab:
            event_vocab[event] = len(event_vocab)
        phase = _phase_label(previous_row, previous_state)
        if phase not in phase_vocab:
            phase_vocab[phase] = len(phase_vocab)
        graph = rml_to_graph(previous_state)
        examples.append(
            DynamicsExample(
                previous_state=previous_state,
                event=event,
                next_state=next_state,
                graph=graph,
                event_id=event_vocab[event],
                transition_label=_transition_label(previous_state, next_state, row),
                phase_label=phase_vocab[phase],
                structural_features=graph_structural_features(graph) if use_graph_structural_features else (),
            )
        )
    if not examples:
        raise ValueError(f"No graph dynamics examples found in {dataset_path}")
    return examples, event_vocab, phase_vocab


class _DynamicsCollator:
    def __init__(
        self,
        *,
        node_kind_vocab: dict[str, int],
        node_value_vocab: dict[str, int],
        edge_type_vocab: dict[str, int],
        device: torch.device,
        include_structural_features: bool = False,
    ) -> None:
        self.node_kind_vocab = node_kind_vocab
        self.node_value_vocab = node_value_vocab
        self.edge_type_vocab = edge_type_vocab
        self.device = device
        self.include_structural_features = bool(include_structural_features)

    def __call__(self, examples: list[DynamicsExample]) -> dict[str, torch.Tensor | RMLGraphBatch]:
        batch: dict[str, torch.Tensor | RMLGraphBatch] = {
            "graph_batch": RMLGraphBatch.from_graphs(
                [example.graph for example in examples],
                node_kind_vocab=self.node_kind_vocab,
                node_value_vocab=self.node_value_vocab,
                edge_type_vocab=self.edge_type_vocab,
                device=self.device,
            ),
            "event_ids": torch.tensor([example.event_id for example in examples], dtype=torch.long, device=self.device),
            "transition_labels": torch.tensor(
                [example.transition_label for example in examples],
                dtype=torch.long,
                device=self.device,
            ),
            "phase_labels": torch.tensor([example.phase_label for example in examples], dtype=torch.long, device=self.device),
        }
        if self.include_structural_features:
            batch["structural_features"] = torch.tensor(
                [
                    example.structural_features
                    if example.structural_features
                    else (0.0,) * GRAPH_STRUCTURAL_FEATURE_DIM
                    for example in examples
                ],
                dtype=torch.float32,
                device=self.device,
            )
        return batch


def _run_epoch(
    model: RMLGraphDynamicsPredictor,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.CrossEntropyLoss,
    phase_loss_fn: nn.CrossEntropyLoss,
    phase_loss_weight: float,
    max_grad_norm: float,
    num_phase_labels: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    transition_correct = 0
    phase_correct = 0
    total = 0
    phase_correct_by_label = torch.zeros(num_phase_labels, dtype=torch.long)
    phase_count_by_label = torch.zeros(num_phase_labels, dtype=torch.long)
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["graph_batch"], batch["event_ids"], batch.get("structural_features"))
        loss = loss_fn(outputs["transition_logits"], batch["transition_labels"])
        loss = loss + float(phase_loss_weight) * phase_loss_fn(outputs["phase_logits"], batch["phase_labels"])
        if training:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        batch_size = int(batch["transition_labels"].shape[0])
        total += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        transition_correct += int((outputs["transition_logits"].argmax(dim=1) == batch["transition_labels"]).sum().detach().cpu())
        phase_predictions = outputs["phase_logits"].argmax(dim=1)
        phase_labels_cpu = batch["phase_labels"].detach().cpu()
        phase_predictions_cpu = phase_predictions.detach().cpu()
        phase_matches = phase_predictions_cpu == phase_labels_cpu
        phase_correct += int(phase_matches.sum())
        phase_count_by_label += torch.bincount(phase_labels_cpu, minlength=num_phase_labels)
        phase_correct_by_label += torch.bincount(phase_labels_cpu[phase_matches], minlength=num_phase_labels)
    denominator = float(max(total, 1))
    result = {
        "loss": total_loss / denominator,
        "transition_accuracy": transition_correct / denominator,
        "phase_accuracy": phase_correct / denominator,
    }
    seen = phase_count_by_label > 0
    if bool(seen.any()):
        per_label = phase_correct_by_label[seen].float() / phase_count_by_label[seen].float()
        result["phase_macro_accuracy"] = float(per_label.mean())
    return result


def _monitor_state_for_graph(row: dict[str, Any], *, prefer_normalized_monitor_state: bool) -> str:
    raw_state = str(row.get("monitor_state", ""))
    if prefer_normalized_monitor_state:
        state = str(row.get("normalized_monitor_state") or raw_state)
    else:
        state = raw_state
    return normalize_generated_variables(state)


def _balanced_phase_sampler(dataset: Dataset, *, seed: int) -> WeightedRandomSampler:
    phase_labels = [_dataset_phase_label(dataset, index) for index in range(len(dataset))]
    counts = torch.bincount(torch.tensor(phase_labels, dtype=torch.long))
    weights = [1.0 / float(max(int(counts[label]), 1)) for label in phase_labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _phase_class_weights(dataset: Dataset, num_phase_labels: int, *, device: torch.device) -> torch.Tensor:
    phase_labels = [_dataset_phase_label(dataset, index) for index in range(len(dataset))]
    counts = torch.bincount(torch.tensor(phase_labels, dtype=torch.long), minlength=num_phase_labels).float()
    weights = torch.zeros(num_phase_labels, dtype=torch.float32)
    seen = counts > 0
    weights[seen] = counts[seen].sum() / (counts[seen] * float(seen.sum()))
    return weights.to(device)


def _dataset_phase_label(dataset: Dataset, index: int) -> int:
    example = dataset[index]
    return int(example.phase_label)


def graph_structural_features(graph: RMLGraphData) -> tuple[float, ...]:
    num_nodes = max(int(graph.num_nodes), 1)
    num_edges = int(graph.num_edges)
    child_edges = [
        (int(graph.edge_index[0, index]), int(graph.edge_index[1, index]))
        for index, edge_type in enumerate(graph.edge_types)
        if edge_type == "parent_to_child"
    ]
    children: dict[int, list[int]] = {}
    has_parent: set[int] = set()
    for source, target in child_edges:
        children.setdefault(source, []).append(target)
        has_parent.add(target)
    roots = [node for node in range(num_nodes) if node not in has_parent] or [0]
    depths = [0] * num_nodes
    stack = [(root, 0) for root in roots]
    seen: set[int] = set()
    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        depths[node] = depth
        for child in children.get(node, []):
            stack.append((child, depth + 1))
    max_depth = max(depths) if depths else 0
    mean_depth = float(sum(depths)) / float(num_nodes)
    leaves = sum(1 for node in range(num_nodes) if not children.get(node))
    branch_nodes = sum(1 for node in range(num_nodes) if len(children.get(node, [])) > 1)
    values = list(graph.node_values)
    in_soak_count = values.count("in_soak")
    stable_count = values.count("stable")
    safe_count = values.count("safe")
    gen_count = values.count("gen")
    app_count = values.count("app")
    eps_count = values.count("eps")
    zero_count = values.count("0")
    identifier_count = sum(1 for kind in graph.node_kinds if kind == "IDENTIFIER")
    soak_chain_proxy = max(0.0, min(1.0, (float(in_soak_count) - 13.0) / 10.0))
    return (
        float(np.log1p(num_nodes) / 8.0),
        float(np.log1p(num_edges) / 9.0),
        float(max_depth / 64.0),
        float(mean_depth / 64.0),
        float(leaves / num_nodes),
        float(branch_nodes / num_nodes),
        float(in_soak_count / 25.0),
        float(stable_count / 25.0),
        float(safe_count / 25.0),
        float(gen_count / 25.0),
        float(app_count / 60.0),
        float(eps_count / 60.0),
        float(zero_count / 200.0),
        float(identifier_count / 400.0),
        soak_chain_proxy,
    )


def _label_counts(examples: list[DynamicsExample]) -> dict[str, int]:
    inverse = {value: key for key, value in TRANSITION_LABELS.items()}
    counts = {label: 0 for label in TRANSITION_LABELS}
    for example in examples:
        counts[inverse[example.transition_label]] += 1
    return counts


def _phase_counts(examples: list[DynamicsExample], phase_vocab: dict[str, int]) -> dict[str, int]:
    inverse = {value: key for key, value in phase_vocab.items()}
    counts = {label: 0 for label in phase_vocab}
    for example in examples:
        counts[inverse.get(example.phase_label, "<UNK>")] += 1
    return counts


def _save_checkpoint(
    path: Path,
    *,
    model: RMLGraphDynamicsPredictor,
    config: GNNDynamicsConfig,
    graph_config: GraphEncoderConfig,
    node_kind_vocab: dict[str, int],
    node_value_vocab: dict[str, int],
    edge_type_vocab: dict[str, int],
    event_vocab: dict[str, int],
    phase_vocab: dict[str, int],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": json_ready(asdict(config)),
            "graph_config": asdict(graph_config),
            "node_kind_vocab": node_kind_vocab,
            "node_value_vocab": node_value_vocab,
            "edge_type_vocab": edge_type_vocab,
            "event_vocab": event_vocab,
            "phase_vocab": phase_vocab,
            "structural_feature_dim": GRAPH_STRUCTURAL_FEATURE_DIM if config.use_graph_structural_features else 0,
            "transition_labels": TRANSITION_LABELS,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def _trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    task_id = row.get("task_id", row.get("task_key", ""))
    return (
        row.get("corpus_source", ""),
        row.get("task_index", row.get("template_index", task_id)),
        task_id,
        row.get("trace_index", ""),
        row.get("trace_label", row.get("trace_type", "")),
        row.get("n", ""),
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*_trace_key(row), int(row.get("step_index", 0)))


def _normalize_event(event: Any) -> str:
    value = str(event)
    if value == "_":
        return "_"
    return value.split(":", maxsplit=1)[0].upper()


def _phase_label(row: dict[str, Any], monitor_state: str) -> str:
    if "phase_label" in row:
        return str(row["phase_label"])
    if "progress_index" in row:
        return f"{row.get('task_key', row.get('task_id', 'task'))}:{int(row['progress_index'])}"
    if monitor_state == "1":
        return "success"
    if monitor_state == "false_verdict":
        return "failure"
    return "active"


def _transition_label(previous_state: str, next_state: str, row: dict[str, Any]) -> int:
    verdict = str(row.get("verdict", "")).lower()
    if next_state in {"false_verdict", "0"} or verdict in {"false", "0"} or bool(row.get("failure", False)):
        return TRANSITION_LABELS["failure"]
    if next_state == "1" or verdict in {"true", "currently_true"} or bool(row.get("success", False)):
        return TRANSITION_LABELS["success"]
    if previous_state == next_state:
        return TRANSITION_LABELS["neutral"]
    return TRANSITION_LABELS["progress"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                row.setdefault("corpus_source", str(path))
                rows.append(row)
    return rows
