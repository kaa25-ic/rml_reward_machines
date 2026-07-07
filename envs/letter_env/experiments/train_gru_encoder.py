"""Train a LetterEnv GRU monitor encoder from a teacher dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.letter_env.gru_pretraining import GRUDistillationConfig, train_gru_distillation


REPO_ROOT = Path(__file__).resolve().parents[3]
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_DATASET_PATH = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gru_teacher_dataset_n1to5_seed0"
    / "dataset.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gru_dim16_seed0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--action-loss-weight", type=float, default=0.1)
    parser.add_argument("--stage-loss-weight", type=float, default=0.5)
    parser.add_argument("--count-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--projection-activation",
        choices=["relu", "tanh", "identity", "none"],
        default="tanh",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--normalize-numbers", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_gru_distillation(
        GRUDistillationConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
            max_len=args.max_len,
            token_dim=args.token_dim,
            hidden_dim=args.hidden_dim,
            monitor_embedding_dim=args.monitor_embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            action_loss_weight=args.action_loss_weight,
            stage_loss_weight=args.stage_loss_weight,
            count_loss_weight=args.count_loss_weight,
            projection_activation=args.projection_activation,
            validation_fraction=args.validation_fraction,
            normalize_numbers=args.normalize_numbers,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
