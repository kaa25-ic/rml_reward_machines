"""LetterEnv basic GNN monitor-encoder pretraining."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

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
from rml_rm.experiments.runtime import configure_torch_seed, json_ready, resolve_torch_device, write_json, write_jsonl


@dataclass(frozen=True)
class GNNDynamicsConfig:
    """Configuration for basic LetterEnv GNN dynamics pretraining."""

    dataset_path: Path
    output_dir: Path
    seed: int = 0
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    max_grad_norm: float = 5.0
    node_embedding_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    event_embedding_dim: int = 16
    phase_loss_weight: float = 1.0
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
    """Train the basic graph dynamics encoder from LetterEnv monitor transitions."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"experiment": "letter_env_gnn_dynamics", "config": asdict(config)})
    configure_torch_seed(config.seed)
    device = resolve_torch_device(config.device)
    examples, event_vocab, phase_vocab, source_format = load_dynamics_examples(config.dataset_path)
    dataset = DynamicsDataset(examples)
    graphs = [example.graph for example in examples]
    node_kind_vocab = build_node_kind_vocab(graphs)
    node_value_vocab = build_node_value_vocab(graphs)
    edge_type_vocab = build_edge_type_vocab(graphs)

    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    collate = _DynamicsCollator(
        node_kind_vocab=node_kind_vocab,
        node_value_vocab=node_value_vocab,
        edge_type_vocab=edge_type_vocab,
        device=device,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    graph_config = GraphEncoderConfig(
        node_embedding_dim=config.node_embedding_dim,
        node_value_embedding_dim=0,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model = RMLGraphDynamicsPredictor(
        num_node_kinds=len(node_kind_vocab),
        num_node_values=len(node_value_vocab),
        num_edge_types=len(edge_type_vocab),
        num_events=len(event_vocab),
        graph_config=graph_config,
        event_embedding_dim=config.event_embedding_dim,
        num_phase_labels=len(phase_vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
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
            phase_loss_weight=config.phase_loss_weight,
            max_grad_norm=config.max_grad_norm,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            loss_fn=ce_loss,
            phase_loss_weight=config.phase_loss_weight,
            max_grad_norm=config.max_grad_norm,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        metrics.append(record)
        write_jsonl(output_dir / "metrics.jsonl", metrics, sort_keys=True)
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
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
            "source_format": source_format,
            "examples": len(examples),
            "train_examples": train_size,
            "validation_examples": validation_size,
        },
        "graph": {
            "node_value_embedding_dim": 0,
            "num_node_kinds": len(node_kind_vocab),
            "num_node_values": len(node_value_vocab),
            "num_edge_types": len(edge_type_vocab),
            "event_vocab": event_vocab,
            "phase_vocab": phase_vocab,
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


def load_dynamics_examples(
    dataset_path: Path,
) -> tuple[list[DynamicsExample], dict[str, int], dict[str, int], str]:
    rows = _read_jsonl(dataset_path)
    if rows and {"monitor_state", "event"}.issubset(rows[0]):
        examples, event_vocab, phase_vocab = _load_dynamics_examples_from_monitor_corpus(rows, dataset_path)
        return examples, event_vocab, phase_vocab, "monitor_corpus"
    examples, event_vocab, phase_vocab = _load_dynamics_examples_from_teacher_dataset(rows, dataset_path)
    return examples, event_vocab, phase_vocab, "teacher_transitions"


def _load_dynamics_examples_from_teacher_dataset(
    rows: list[dict[str, Any]],
    dataset_path: Path,
) -> tuple[list[DynamicsExample], dict[str, int], dict[str, int]]:
    event_vocab = {"<UNK>": 0}
    phase_vocab = {"<UNK>": 0}
    examples: list[DynamicsExample] = []
    for row in rows:
        previous_state = normalize_generated_variables(str(row["monitor_state_string"]))
        next_state = normalize_generated_variables(str(row["next_monitor_state_string"]))
        event = _event_from_next_observation(row.get("next_env_obs", []))
        if event not in event_vocab:
            event_vocab[event] = len(event_vocab)
        phase = _phase_label(previous_state)
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
            )
        )
    if not examples:
        raise ValueError(f"No dynamics examples found in {dataset_path}")
    return examples, event_vocab, phase_vocab


def _load_dynamics_examples_from_monitor_corpus(
    rows: list[dict[str, Any]],
    dataset_path: Path,
) -> tuple[list[DynamicsExample], dict[str, int], dict[str, int]]:
    event_vocab = {"<UNK>": 0}
    phase_vocab = {"<UNK>": 0}
    examples: list[DynamicsExample] = []
    sorted_rows = sorted(rows, key=_corpus_row_sort_key)
    previous_by_trace: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in sorted_rows:
        trace_key = _corpus_trace_key(row)
        previous_row = previous_by_trace.get(trace_key)
        previous_by_trace[trace_key] = row
        if previous_row is None:
            continue

        previous_state = normalize_generated_variables(str(previous_row["monitor_state"]))
        next_state = normalize_generated_variables(str(row["monitor_state"]))
        event = _normalize_corpus_event(row.get("event", "_"))
        if event not in event_vocab:
            event_vocab[event] = len(event_vocab)
        phase = _phase_label(previous_state)
        if phase not in phase_vocab:
            phase_vocab[phase] = len(phase_vocab)
        examples.append(
            DynamicsExample(
                previous_state=previous_state,
                event=event,
                next_state=next_state,
                graph=rml_to_graph(previous_state),
                event_id=event_vocab[event],
                transition_label=_corpus_transition_label(previous_state, next_state, row),
                phase_label=phase_vocab[phase],
            )
        )

    if not examples:
        raise ValueError(f"No dynamics examples found in {dataset_path}")
    return examples, event_vocab, phase_vocab


class _DynamicsCollator:
    def __init__(
        self,
        *,
        node_kind_vocab: dict[str, int],
        node_value_vocab: dict[str, int],
        edge_type_vocab: dict[str, int],
        device: torch.device,
    ) -> None:
        self.node_kind_vocab = node_kind_vocab
        self.node_value_vocab = node_value_vocab
        self.edge_type_vocab = edge_type_vocab
        self.device = device

    def __call__(self, examples: list[DynamicsExample]) -> dict[str, torch.Tensor | RMLGraphBatch]:
        return {
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


def _run_epoch(
    model: RMLGraphDynamicsPredictor,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.CrossEntropyLoss,
    phase_loss_weight: float,
    max_grad_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    transition_correct = 0
    phase_correct = 0
    total = 0
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["graph_batch"], batch["event_ids"])
        loss = loss_fn(outputs["transition_logits"], batch["transition_labels"])
        loss = loss + float(phase_loss_weight) * loss_fn(outputs["phase_logits"], batch["phase_labels"])
        if training:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        batch_size = int(batch["transition_labels"].shape[0])
        total += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        transition_correct += int((outputs["transition_logits"].argmax(dim=1) == batch["transition_labels"]).sum().detach().cpu())
        phase_correct += int((outputs["phase_logits"].argmax(dim=1) == batch["phase_labels"]).sum().detach().cpu())
    denominator = float(max(total, 1))
    return {
        "loss": total_loss / denominator,
        "transition_accuracy": transition_correct / denominator,
        "phase_accuracy": phase_correct / denominator,
    }


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
            "transition_labels": TRANSITION_LABELS,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def _event_from_next_observation(next_env_obs: Any) -> str:
    values = np.asarray(next_env_obs, dtype=np.float32).reshape(-1)
    if values.shape[0] < 7:
        return "_"
    labels = ("A", "B", "C", "D", "_")
    index = int(np.argmax(values[2:7]))
    return labels[index]


def _corpus_trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("template_index"),
        row.get("task_id"),
        row.get("trace_index"),
        row.get("trace_label"),
        row.get("n"),
    )


def _corpus_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (*_corpus_trace_key(row), int(row.get("step_index", 0)))


def _normalize_corpus_event(event: Any) -> str:
    value = str(event)
    if value == "_":
        return "_"
    return value.split(":", maxsplit=1)[0].upper()


def _corpus_transition_label(previous_state: str, next_state: str, row: dict[str, Any]) -> int:
    verdict = str(row.get("verdict", "")).lower()
    if next_state == "false_verdict" or verdict == "false":
        return TRANSITION_LABELS["failure"]
    if next_state == "1" or verdict in {"true", "currently_true"}:
        return TRANSITION_LABELS["success"]
    if previous_state == next_state:
        return TRANSITION_LABELS["neutral"]
    return TRANSITION_LABELS["progress"]


def _transition_label(previous_state: str, next_state: str, row: dict[str, Any]) -> int:
    if bool(row.get("task_failed", False)) or next_state == "false_verdict":
        return TRANSITION_LABELS["failure"]
    if next_state == "1" or bool(row.get("terminated", False)) and float(row.get("base_reward", 0.0)) >= 1.0:
        return TRANSITION_LABELS["success"]
    if previous_state == next_state:
        return TRANSITION_LABELS["neutral"]
    return TRANSITION_LABELS["progress"]


def _phase_label(monitor_state: str) -> str:
    state = str(monitor_state)
    if state == "1":
        return "success"
    if state == "false_verdict":
        return "failure"
    if "(d_match" in state and state.startswith("@(app(gen([n],),"):
        return "D"
    if "(c_match" in state and state.startswith("@(app(gen([n],star(not_abcd:eps)*((c_match"):
        return "C"
    if "(b_match" in state:
        return "B"
    return "A"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows

