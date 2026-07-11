"""Supervised GRU pretraining for multi-task LetterEnv monitor states."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from rml_rm.encodings.rml_sequence import MonitorVocab, RMLSequenceEncoder
from rml_rm.experiments.runtime import (
    configure_torch_seed,
    json_ready,
    resolve_torch_device,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class MultitaskGRUPretrainingConfig:
    """Configuration for multi-task LetterEnv GRU monitor pretraining."""

    dataset_path: Path
    output_dir: Path
    seed: int = 0
    max_len: int = 256
    token_dim: int = 32
    hidden_dim: int = 64
    monitor_embedding_dim: int = 32
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    normalize_numbers: bool = False
    projection_activation: str = "tanh"
    max_grad_norm: float = 10.0
    progress_loss_weight: float = 1.0
    next_event_loss_weight: float = 0.5
    terminal_loss_weight: float = 0.5
    task_loss_weight: float = 0.25
    device: str = "auto"


class MultitaskMonitorDataset(Dataset):
    """Labelled RML monitor strings for supervised encoder pretraining."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab: MonitorVocab,
        *,
        max_len: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            monitor_state = str(row.get("normalized_monitor_state") or row["monitor_state"])
            token_ids, length = vocab.encode(monitor_state, max_len=max_len)
            self.examples.append(
                {
                    "token_ids": torch.tensor(token_ids, dtype=torch.long),
                    "length": torch.tensor(length, dtype=torch.long),
                    "progress_label": torch.tensor(int(row["progress_index"]), dtype=torch.long),
                    "next_event_label": torch.tensor(
                        int(row["next_expected_event_id"]), dtype=torch.long
                    ),
                    "terminal_label": torch.tensor(int(row["terminal_type_id"]), dtype=torch.long),
                    "task_label": torch.tensor(int(row["task_id"]), dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class MultitaskGRUPretrainingModel(nn.Module):
    """GRU monitor encoder trained with task-progress classification heads."""

    def __init__(
        self,
        *,
        vocab_size: int,
        token_dim: int,
        hidden_dim: int,
        monitor_embedding_dim: int,
        num_progress_states: int,
        num_tasks: int,
        projection_activation: str,
    ) -> None:
        super().__init__()
        self.monitor_encoder = RMLSequenceEncoder(
            vocab_size=vocab_size,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            output_dim=monitor_embedding_dim,
            projection_activation=projection_activation,
        )
        self.progress_head = nn.Linear(monitor_embedding_dim, num_progress_states)
        self.next_event_head = nn.Linear(monitor_embedding_dim, 5)
        self.terminal_head = nn.Linear(monitor_embedding_dim, 3)
        self.task_head = nn.Linear(monitor_embedding_dim, num_tasks)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.monitor_encoder(token_ids, lengths)
        return {
            "embedding": embedding,
            "progress_logits": self.progress_head(embedding),
            "next_event_logits": self.next_event_head(embedding),
            "terminal_logits": self.terminal_head(embedding),
            "task_logits": self.task_head(embedding),
        }


def train_multitask_gru_encoder(config: MultitaskGRUPretrainingConfig) -> dict[str, Any]:
    """Train and save a frozen GRU monitor encoder for multi-task LetterEnv."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {"experiment": "multitask_letter_env_gru_pretraining", "config": asdict(config)},
    )
    configure_torch_seed(config.seed)
    device = resolve_torch_device(config.device)

    rows = load_jsonl_rows(config.dataset_path)
    if not rows:
        raise ValueError(f"No rows found in {config.dataset_path}.")

    monitor_strings = [str(row.get("normalized_monitor_state") or row["monitor_state"]) for row in rows]
    vocab = MonitorVocab(normalize_numbers=config.normalize_numbers)
    vocab.build(monitor_strings)
    num_progress_states = max(int(row["progress_index"]) for row in rows) + 1
    num_tasks = max(int(row["task_id"]) for row in rows) + 1

    dataset = MultitaskMonitorDataset(rows, vocab, max_len=config.max_len)
    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Not enough rows for a train/validation split.")
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    model = MultitaskGRUPretrainingModel(
        vocab_size=len(vocab),
        token_dim=config.token_dim,
        hidden_dim=config.hidden_dim,
        monitor_embedding_dim=config.monitor_embedding_dim,
        num_progress_states=num_progress_states,
        num_tasks=num_tasks,
        projection_activation=config.projection_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = math.inf
    best_epoch = -1
    metrics: list[dict[str, float | int]] = []
    best_path = output_dir / "best_student.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
            ce_loss=ce_loss,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            config=config,
            device=device,
            optimizer=None,
            ce_loss=ce_loss,
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
            _save_checkpoint(best_path, model, config, vocab, num_progress_states, num_tasks)

    final_path = output_dir / "student_final.pt"
    _save_checkpoint(final_path, model, config, vocab, num_progress_states, num_tasks)
    summary = {
        "config": json_ready(asdict(config)),
        "dataset": {
            "path": str(config.dataset_path),
            "rows": len(rows),
            "train_rows": train_size,
            "validation_rows": validation_size,
            "unique_monitor_strings": len(set(monitor_strings)),
            "num_progress_states": num_progress_states,
            "num_tasks": num_tasks,
        },
        "vocab": {
            "size": len(vocab),
            "normalize_numbers": vocab.normalize_numbers,
        },
        "best": {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "student_path": str(best_path),
        },
        "artifacts": {
            "config": str(output_dir / "config.json"),
            "best_student": str(best_path),
            "final_student": str(final_path),
            "metrics": str(output_dir / "metrics.jsonl"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load labelled monitor-state rows from JSONL."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _run_epoch(
    model: MultitaskGRUPretrainingModel,
    loader: DataLoader,
    *,
    config: MultitaskGRUPretrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ce_loss: nn.Module,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = _Totals()
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            lengths = batch["length"].to(device)
            progress = batch["progress_label"].to(device)
            next_event = batch["next_event_label"].to(device)
            terminal = batch["terminal_label"].to(device)
            task = batch["task_label"].to(device)

            outputs = model(token_ids, lengths)
            progress_loss = ce_loss(outputs["progress_logits"], progress)
            next_event_loss = ce_loss(outputs["next_event_logits"], next_event)
            terminal_loss = ce_loss(outputs["terminal_logits"], terminal)
            task_loss = ce_loss(outputs["task_logits"], task)
            loss = (
                config.progress_loss_weight * progress_loss
                + config.next_event_loss_weight * next_event_loss
                + config.terminal_loss_weight * terminal_loss
                + config.task_loss_weight * task_loss
            )
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

            batch_size = int(token_ids.shape[0])
            totals.add("loss", float(loss.item()), batch_size)
            totals.add("progress_loss", float(progress_loss.item()), batch_size)
            totals.add("next_event_loss", float(next_event_loss.item()), batch_size)
            totals.add("terminal_loss", float(terminal_loss.item()), batch_size)
            totals.add("task_loss", float(task_loss.item()), batch_size)
            totals.add_accuracy("progress_accuracy", outputs["progress_logits"].argmax(dim=1), progress)
            totals.add_accuracy(
                "next_event_accuracy",
                outputs["next_event_logits"].argmax(dim=1),
                next_event,
            )
            totals.add_accuracy("terminal_accuracy", outputs["terminal_logits"].argmax(dim=1), terminal)
            totals.add_accuracy("task_accuracy", outputs["task_logits"].argmax(dim=1), task)
    return totals.as_dict()


def _save_checkpoint(
    path: Path,
    model: MultitaskGRUPretrainingModel,
    config: MultitaskGRUPretrainingConfig,
    vocab: MonitorVocab,
    num_progress_states: int,
    num_tasks: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "monitor_encoder_state_dict": model.monitor_encoder.state_dict(),
            "config": json_ready(asdict(config)),
            "vocab_token_to_id": vocab.token_to_id,
            "vocab_id_to_token": vocab.id_to_token,
            "vocab_normalize_numbers": vocab.normalize_numbers,
            "num_progress_states": int(num_progress_states),
            "num_tasks": int(num_tasks),
            "pretraining_type": "multi_task_rml_supervised_gru",
        },
        path,
    )


class _Totals:
    def __init__(self) -> None:
        self.weighted: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: float, count: int) -> None:
        self.weighted[key] = self.weighted.get(key, 0.0) + value * count
        self.counts[key] = self.counts.get(key, 0) + count

    def add_accuracy(self, key: str, predicted: torch.Tensor, target: torch.Tensor) -> None:
        self.weighted[key] = self.weighted.get(key, 0.0) + float(
            (predicted == target).sum().item()
        )
        self.counts[key] = self.counts.get(key, 0) + int(target.numel())

    def as_dict(self) -> dict[str, float]:
        return {
            key: self.weighted[key] / max(1, self.counts[key])
            for key in sorted(self.weighted)
        }
