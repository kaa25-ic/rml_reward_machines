"""Evaluate trained CSTR PPO policies on longer unseen soak requirements."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from envs.cstr.experiments.evaluate_cstr_ppo import evaluate_checkpoint
from rml_rm.experiments.runtime import write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "envs" / "cstr" / "results_and_evaluation"
PPO_ROOT = RESULTS_ROOT / "ppo"
GENERALIZATION_ROOT = RESULTS_ROOT / "generalization"
GRAPH_ENCODER_CHECKPOINT = (
    RESULTS_ROOT
    / "encoder_pretraining"
    / "gnn_dynamics_phase_count_reference_seed0"
    / "best_dynamics_encoder.pt"
)

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "rml_hidden": {
        "source_dir": "rml_hidden_seed{seed}",
        "output_dir": "rml_hidden_seed{seed}",
        "env_variant": "rml_hidden",
        "reward_mode": "env_rml",
        "monitor_state_limit": 32,
    },
    "rml_semantic_progress": {
        "source_dir": "rml_semantic_progress_seed{seed}",
        "output_dir": "rml_semantic_progress_seed{seed}",
        "env_variant": "semantic_progress",
        "reward_mode": "env_rml",
        "monitor_state_limit": 32,
    },
    "manual_rm_semantic_progress": {
        "source_dir": "manual_rm_semantic_progress_seed{seed}",
        "output_dir": "manual_rm_semantic_progress_seed{seed}",
        "env_variant": "manual_rm_semantic_progress",
        "reward_mode": "env_rml",
        "monitor_state_limit": 32,
    },
    "rml_graph_encoder": {
        "source_dir": "rml_graph_encoder_seed{seed}",
        "output_dir": "rml_graph_encoder_seed{seed}",
        "env_variant": "rml_graph",
        "reward_mode": "env_rml",
        "monitor_state_limit": 16,
        "graph_encoder_checkpoint": GRAPH_ENCODER_CHECKPOINT,
    },
    "baseline": {
        "source_dir": "baseline_seed{seed}",
        "output_dir": "baseline_seed{seed}",
        "env_variant": "baseline",
        "reward_mode": "env",
        "monitor_state_limit": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--train-soak-steps", type=int, default=10)
    parser.add_argument("--eval-soak-steps", type=int, default=15)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--regulation-violation-steps", type=int, default=10)
    parser.add_argument("--deadline-steps", type=int, default=100)
    parser.add_argument("--model-root", type=Path, default=PPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--graph-encoder-checkpoint", type=Path, default=GRAPH_ENCODER_CHECKPOINT)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_SPECS),
        action="append",
        dest="variants",
        help="Variant to evaluate. Repeat to select multiple variants.",
    )
    parser.add_argument("--include-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(args.variants or [
        "rml_hidden",
        "rml_semantic_progress",
        "manual_rm_semantic_progress",
        "rml_graph_encoder",
    ])
    if args.include_baseline and "baseline" not in variants:
        variants.insert(0, "baseline")

    output_root = args.output_root or GENERALIZATION_ROOT / f"soak{args.eval_soak_steps}_seed{args.train_seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []

    for variant in variants:
        results.append(_evaluate_variant(args, output_root, variant))

    summary = {
        "experiment": f"cstr_generalization_soak{args.eval_soak_steps}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "train_soak_steps": args.train_soak_steps,
        "eval_soak_steps": args.eval_soak_steps,
        "episodes": args.episodes,
        "results": results,
        "paths": {
            "output_root": str(output_root),
            "summary": str(output_root / "summary.json"),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _evaluate_variant(args: argparse.Namespace, output_root: Path, variant: str) -> dict[str, Any]:
    spec = VARIANT_SPECS[variant]
    source_dir = args.model_root / spec["source_dir"].format(seed=args.train_seed)
    output_dir = output_root / spec["output_dir"].format(seed=args.train_seed)
    model_path = source_dir / "best_model.zip"
    graph_checkpoint = spec.get("graph_encoder_checkpoint")
    if variant == "rml_graph_encoder":
        graph_checkpoint = args.graph_encoder_checkpoint

    payload = {
        "experiment": f"cstr_generalization_soak{args.eval_soak_steps}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(source_dir),
        "model_path": str(model_path),
        "variant": variant,
        "env_variant": spec["env_variant"],
        "reward_mode": spec["reward_mode"],
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "soak_steps_train": args.train_soak_steps,
        "soak_steps_eval": args.eval_soak_steps,
        "monitor_state_limit": spec["monitor_state_limit"],
    }
    try:
        result = evaluate_checkpoint(
            SimpleNamespace(
                model_path=model_path,
                env_variant=spec["env_variant"],
                reward_mode=spec["reward_mode"],
                episodes=args.episodes,
                seed=args.eval_seed,
                max_episode_steps=args.max_episode_steps,
                regulation_violation_steps=args.regulation_violation_steps,
                soak_steps=args.eval_soak_steps,
                monitor_state_limit=spec["monitor_state_limit"],
                graph_encoder_checkpoint=graph_checkpoint,
                safe_step_bonus=0.10,
                stable_step_bonus=3.0,
                regulation_entry_bonus=5.0,
                success_bonus=50.0,
                failure_penalty=-50.0,
                rml_heating_rate_penalty=0.0,
                preheat_distance_weight=0.08,
                preheat_warming_weight=0.25,
                soak_entry_bonus=5.0,
                soak_progress_bonus=0.75,
                soak_reset_penalty=-3.0,
                soak_lost_step_penalty=0.50,
                approach_distance_weight=1.0,
                approach_progress_bonus=5.0,
                approach_ca_progress_bonus=4.0,
                approach_temp_progress_bonus=4.0,
                approach_warming_weight=0.50,
                production_entry_bonus=10.0,
                regulate_recovery_penalty=-10.0,
                deadline_steps=args.deadline_steps,
                tracking_weight=0.5,
                heating_rate_penalty=0.0,
                critical_penalty=200.0,
                production_temp_low=346.0,
                production_temp_high=354.0,
                concentration_tolerance=0.08,
                ca_overshoot_low=0.44,
                require_soak_concentration_band=True,
                soak_concentration_low=0.58,
                soak_concentration_high=0.74,
                fixed_initial_state=True,
                randomize_initial_state=False,
                randomize_setpoint=False,
                enable_disturbance=False,
                temp_weight=0.015,
                action_weight=0.0002,
                warning_penalty=0.25,
                output_dir=output_dir,
            )
        )
        payload.update(
            {
                "status": "evaluated",
                "summary": result["summary"],
                "paths": result["paths"],
            }
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "paths": {
                    "output_dir": str(output_dir),
                    "summary": str(output_dir / "summary.json"),
                },
            }
        )
        write_json(output_dir / "summary.json", payload)
    return payload


if __name__ == "__main__":
    main()
