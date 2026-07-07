"""LetterEnv GRU monitor-encoder pretraining."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from rml_rm.encodings.rml_sequence import MonitorVocab, RMLSequenceEncoder
from rml_rm.experiments.runtime import configure_torch_seed, json_ready, resolve_torch_device, write_json, write_jsonl
from rml_rm.wrappers.rml_monitor import normalize_monitor_state


@dataclass(frozen=True)
class GRUDistillationConfig:
    """Configuration for LetterEnv GRU monitor-encoder distillation."""

    dataset_path: Path
    output_dir: Path
    seed: int = 0
    max_len: int = 192
    token_dim: int = 32
    hidden_dim: int = 64
    monitor_embedding_dim: int = 16
    batch_size: int = 1024
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    action_loss_weight: float = 0.1
    stage_loss_weight: float = 0.5
    count_loss_weight: float = 0.1
    projection_activation: str = "tanh"
    validation_fraction: float = 0.15
    normalize_numbers: bool = False
    max_grad_norm: float = 10.0
    device: str = "auto"


class TeacherTransitionDataset(Dataset):
    """Teacher transitions used to train a monitor-string GRU encoder."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab: MonitorVocab,
        *,
        max_len: int,
        q_mean: torch.Tensor | None = None,
        q_std: torch.Tensor | None = None,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            token_ids, length = vocab.encode(row["monitor_state_string"], max_len=max_len)
            teacher_q = torch.tensor(row["teacher_q_values"], dtype=torch.float32)
            if q_mean is not None and q_std is not None:
                teacher_q = (teacher_q - q_mean) / q_std
            stage_label, count_targets = infer_letter_env_monitor_targets(
                str(row["monitor_state_string"]),
                max_n=int(row.get("n_value", 5) or 5),
            )
            self.examples.append(
                {
                    "env_obs": torch.tensor(row["env_obs"], dtype=torch.float32),
                    "token_ids": torch.tensor(token_ids, dtype=torch.long),
                    "length": torch.tensor(length, dtype=torch.long),
                    "teacher_q": teacher_q,
                    "teacher_action": torch.tensor(int(row["teacher_action"]), dtype=torch.long),
                    "stage_label": torch.tensor(stage_label, dtype=torch.long),
                    "count_targets": torch.tensor(count_targets, dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class MonitorGRUQStudent(nn.Module):
    """Student Q model whose monitor representation comes from a GRU."""

    def __init__(
        self,
        *,
        env_obs_dim: int,
        num_actions: int,
        vocab_size: int,
        token_dim: int,
        hidden_dim: int,
        monitor_embedding_dim: int,
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
        self.q_head = nn.Sequential(
            nn.Linear(env_obs_dim + monitor_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        self.stage_head = nn.Linear(monitor_embedding_dim, 6)
        self.count_head = nn.Sequential(
            nn.Linear(monitor_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        env_obs: torch.Tensor,
        monitor_token_ids: torch.Tensor,
        monitor_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        monitor_embedding = self.monitor_encoder(monitor_token_ids, monitor_lengths)
        return {
            "q_values": self.q_head(torch.cat([env_obs, monitor_embedding], dim=-1)),
            "stage_logits": self.stage_head(monitor_embedding),
            "count_values": self.count_head(monitor_embedding),
            "monitor_embedding": monitor_embedding,
        }


def train_gru_distillation(config: GRUDistillationConfig) -> dict[str, Any]:
    """Train a frozen GRU monitor encoder from a teacher dataset."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"experiment": "letter_env_gru_distillation", "config": asdict(config)})
    configure_torch_seed(config.seed)
    device = resolve_torch_device(config.device)
    rows = _load_rows(config.dataset_path)
    if not rows:
        raise ValueError(f"No rows found in dataset: {config.dataset_path}")

    monitor_strings = [str(row["monitor_state_string"]) for row in rows]
    vocab = MonitorVocab(normalize_numbers=config.normalize_numbers)
    vocab.build(monitor_strings)

    env_obs_dim = len(rows[0]["env_obs"])
    num_actions = len(rows[0]["teacher_q_values"])
    teacher_q_values = torch.tensor([row["teacher_q_values"] for row in rows], dtype=torch.float32)
    q_mean = teacher_q_values.mean(dim=0)
    q_std = teacher_q_values.std(dim=0).clamp(min=1e-6)
    dataset = TeacherTransitionDataset(rows, vocab, max_len=config.max_len, q_mean=q_mean, q_std=q_std)
    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    model = MonitorGRUQStudent(
        env_obs_dim=env_obs_dim,
        num_actions=num_actions,
        vocab_size=len(vocab),
        token_dim=config.token_dim,
        hidden_dim=config.hidden_dim,
        monitor_embedding_dim=config.monitor_embedding_dim,
        projection_activation=config.projection_activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_mse = math.inf
    best_epoch = -1
    metrics: list[dict[str, float | int]] = []
    best_path = output_dir / "best_student.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            mse_loss=mse_loss,
            ce_loss=ce_loss,
            optimizer=optimizer,
            config=config,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=device,
            mse_loss=mse_loss,
            ce_loss=ce_loss,
            optimizer=None,
            config=config,
        )
        record = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_action_accuracy": train_metrics["action_accuracy"],
            "train_stage_accuracy": train_metrics["stage_accuracy"],
            "train_count_mse": train_metrics["count_mse"],
            "val_mse": validation_metrics["mse"],
            "val_action_accuracy": validation_metrics["action_accuracy"],
            "val_stage_accuracy": validation_metrics["stage_accuracy"],
            "val_count_mse": validation_metrics["count_mse"],
        }
        metrics.append(record)
        write_jsonl(output_dir / "metrics.jsonl", metrics)

        if validation_metrics["mse"] < best_val_mse:
            best_val_mse = validation_metrics["mse"]
            best_epoch = epoch
            _save_student_checkpoint(
                best_path,
                model,
                config,
                vocab,
                env_obs_dim,
                num_actions,
                q_mean,
                q_std,
            )

    final_path = output_dir / "student_final.pt"
    _save_student_checkpoint(
        final_path,
        model,
        config,
        vocab,
        env_obs_dim,
        num_actions,
        q_mean,
        q_std,
    )
    summary = {
        "config": json_ready(asdict(config)),
        "dataset": {
            "path": str(config.dataset_path),
            "rows": len(rows),
            "train_rows": train_size,
            "validation_rows": validation_size,
            "unique_monitor_strings": len(set(monitor_strings)),
            "env_obs_dim": env_obs_dim,
            "num_actions": num_actions,
            "q_targets_standardized": True,
        },
        "vocab": {
            "size": len(vocab),
            "normalize_numbers": vocab.normalize_numbers,
        },
        "best": {
            "epoch": best_epoch,
            "val_mse": best_val_mse,
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


def infer_letter_env_monitor_targets(monitor_state: str, *, max_n: int) -> tuple[int, list[float]]:
    """Return stage label and normalized count targets for a LetterEnv monitor state."""
    max_n = max(1, int(max_n))
    state = normalize_monitor_state(str(monitor_state)).replace(" ", "")
    if state == "false_verdict":
        return 5, [0.0, 0.0]
    if state == "1":
        return 4, [0.0, 0.0]

    stage_name = _infer_letter_env_stage(state)
    values = _extract_letter_env_numeric_values(state)
    n_value = max(values) if values and stage_name != "A" else 0
    if stage_name == "D":
        d_remaining = min(values) if values else n_value
    elif stage_name in {"B", "C"}:
        d_remaining = n_value
    else:
        d_remaining = 0
    if stage_name == "D" and d_remaining <= 0:
        return 4, [float(min(n_value, max_n)) / float(max_n), 0.0]

    stage_id = {"A": 0, "B": 1, "C": 2, "D": 3}[stage_name]
    return stage_id, [
        float(min(max(n_value, 0), max_n)) / float(max_n),
        float(min(max(d_remaining, 0), max_n)) / float(max_n),
    ]


def _run_epoch(
    model: MonitorGRUQStudent,
    loader: DataLoader,
    *,
    device: torch.device,
    mse_loss: nn.Module,
    ce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: GRUDistillationConfig,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_mse = 0.0
    total_count_mse = 0.0
    total_correct = 0
    total_stage_correct = 0
    total_rows = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            env_obs = batch["env_obs"].to(device)
            token_ids = batch["token_ids"].to(device)
            lengths = batch["length"].to(device)
            teacher_q = batch["teacher_q"].to(device)
            teacher_action = batch["teacher_action"].to(device)
            stage_label = batch["stage_label"].to(device)
            count_targets = batch["count_targets"].to(device)

            outputs = model(env_obs, token_ids, lengths)
            q_values = outputs["q_values"]
            mse = mse_loss(q_values, teacher_q)
            action_ce = ce_loss(q_values, teacher_action)
            stage_ce = ce_loss(outputs["stage_logits"], stage_label)
            count_mse = mse_loss(outputs["count_values"], count_targets)
            loss = (
                mse
                + config.action_loss_weight * action_ce
                + config.stage_loss_weight * stage_ce
                + config.count_loss_weight * count_mse
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

            batch_size = int(env_obs.shape[0])
            total_mse += float(mse.item()) * batch_size
            total_count_mse += float(count_mse.item()) * batch_size
            total_correct += int((q_values.argmax(dim=1) == teacher_action).sum().item())
            total_stage_correct += int((outputs["stage_logits"].argmax(dim=1) == stage_label).sum().item())
            total_rows += batch_size

    return {
        "mse": total_mse / max(1, total_rows),
        "count_mse": total_count_mse / max(1, total_rows),
        "action_accuracy": total_correct / max(1, total_rows),
        "stage_accuracy": total_stage_correct / max(1, total_rows),
    }


def _save_student_checkpoint(
    path: Path,
    model: MonitorGRUQStudent,
    config: GRUDistillationConfig,
    vocab: MonitorVocab,
    env_obs_dim: int,
    num_actions: int,
    q_mean: torch.Tensor,
    q_std: torch.Tensor,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "monitor_encoder_state_dict": model.monitor_encoder.state_dict(),
            "config": json_ready(asdict(config)),
            "vocab_token_to_id": vocab.token_to_id,
            "vocab_id_to_token": vocab.id_to_token,
            "vocab_normalize_numbers": vocab.normalize_numbers,
            "env_obs_dim": env_obs_dim,
            "num_actions": num_actions,
            "q_mean": q_mean.tolist(),
            "q_std": q_std.tolist(),
        },
        path,
    )


def _infer_letter_env_stage(state: str) -> str:
    if state.startswith("@(app(gen([n],),") or state.startswith("@(eps*(star(not_abcd:eps)*((d_match:eps)"):
        return "D"
    if state.startswith("@(app(gen([n],star(not_abcd:eps)*((c_match:eps)") or state.startswith(
        "@(eps*(star(not_abcd:eps)*((c_match:eps)"
    ):
        return "C"
    if state.startswith("@(app(gen([n],star(not_abcd:eps)*((b_match:eps)") or state.startswith(
        "@(star(not_abcd:eps)*((b_match:eps)"
    ):
        return "B"
    return "A"


def _extract_letter_env_numeric_values(state: str) -> list[int]:
    values: list[int] = []
    for bracket_content in re.findall(r"\[([0-9]+(?:\.[0-9]+)?(?:[+-][0-9]+(?:\.[0-9]+)?)?)\]", state):
        try:
            value = eval(bracket_content, {"__builtins__": {}}, {})
        except Exception:
            continue
        values.append(int(round(float(value))))
    return values


def _load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

