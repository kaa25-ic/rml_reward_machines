"""Train the CSTR graph monitor encoder from monitor-transition data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rml_rm.encodings.graph_pretraining import GNNDynamicsConfig, train_gnn_dynamics


REPO_ROOT = Path(__file__).resolve().parents[3]
CSTR_ROOT = REPO_ROOT / "envs" / "cstr"
DEFAULT_DATASET_PATH = (
    CSTR_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_corpus_seed0"
    / "monitor_states.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    CSTR_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_dynamics_values_phase_count_seed0"
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
    parser.add_argument("--node-value-embedding-dim", type=int, default=0)
    parser.add_argument("--node-value-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-layer-norm", action="store_true")
    parser.add_argument("--output-l2-normalize", action="store_true")
    parser.add_argument("--event-embedding-dim", type=int, default=16)
    parser.add_argument("--phase-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase-class-weighting", action="store_true")
    parser.add_argument("--balanced-phase-sampling", action="store_true")
    parser.add_argument("--use-graph-structural-features", action="store_true")
    parser.add_argument(
        "--prefer-normalized-monitor-state",
        action="store_true",
        help="Use normalized_monitor_state when available instead of raw monitor_state.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_gnn_dynamics(
        GNNDynamicsConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            experiment_name="cstr_gnn_dynamics_values_phase_count",
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            max_grad_norm=args.max_grad_norm,
            node_embedding_dim=args.node_embedding_dim,
            node_value_embedding_dim=args.node_value_embedding_dim,
            node_value_dropout=args.node_value_dropout,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            output_layer_norm=args.output_layer_norm,
            output_l2_normalize=args.output_l2_normalize,
            event_embedding_dim=args.event_embedding_dim,
            phase_loss_weight=args.phase_loss_weight,
            phase_class_weighting=args.phase_class_weighting,
            balanced_phase_sampling=args.balanced_phase_sampling,
            use_graph_structural_features=args.use_graph_structural_features,
            prefer_normalized_monitor_state=args.prefer_normalized_monitor_state,
            device=args.device,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
