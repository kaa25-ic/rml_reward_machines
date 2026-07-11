"""Train the multitask LetterEnv graph monitor encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rml_rm.encodings.graph_pretraining import GNNDynamicsConfig, train_gnn_dynamics


REPO_ROOT = Path(__file__).resolve().parents[3]
MULTITASK_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"
DEFAULT_DATASET_PATH = (
    MULTITASK_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_corpus_small_v1_seed0"
    / "monitor_states.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    MULTITASK_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_small_v1_seed0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--node-embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--event-embedding-dim", type=int, default=16)
    parser.add_argument("--phase-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_gnn_dynamics(
        GNNDynamicsConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            experiment_name="multitask_letter_env_gnn_dynamics",
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            max_grad_norm=args.max_grad_norm,
            node_embedding_dim=args.node_embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            event_embedding_dim=args.event_embedding_dim,
            phase_loss_weight=args.phase_loss_weight,
            device=args.device,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
