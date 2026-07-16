"""Run two-stage PPO training on the RML-based LunarLander protocol."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from envs.lunar_lander.experiments.train_ppo import (
    DEFAULT_MONITOR_CONFIG,
    DEFAULT_MONITOR_SPEC,
    LUNAR_ENV_ROOT,
    LunarLanderPPOTrainingConfig,
    train_lunar_lander_ppo,
)
from rml_rm.agents.common import MLPPolicyConfig
from rml_rm.experiments.runtime import json_ready, utc_now, write_json


DEFAULT_OUTPUT_ROOT = (
    LUNAR_ENV_ROOT / "results_and_evaluation" / "ppo" / "two_stage_training"
)


def run_two_stage_lunar_lander_ppo(args: argparse.Namespace) -> dict[str, Any]:
    """Train a discovery stage, then automatically fine-tune its best checkpoint."""
    run_name = args.run_name or f"semantic_progress_two_stage_seed{args.seed}"
    run_dir = args.output_root / run_name
    stage1_dir = run_dir / "stage1_discovery"
    stage2_dir = run_dir / "stage2_stabilization"

    policy_config = MLPPolicyConfig(
        features_dim=args.features_dim,
        position_hidden_dim=args.position_hidden_dim,
        monitor_hidden_dim=args.monitor_hidden_dim,
        monitor_embedding_dim=args.monitor_embedding_dim,
        max_monitor_states=args.max_monitor_states,
        network_architecture=(args.net_arch_width, args.net_arch_width),
    )

    common_config = {
        "encoding": args.encoding,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "eval_seed_base": args.eval_seed_base,
        "max_episode_steps": args.max_episode_steps,
        "monitor_progress_bonus": args.monitor_progress_bonus,
        "hover_step_bonus": args.hover_step_bonus,
        "hover_complete_bonus": args.hover_complete_bonus,
        "controlled_descent_bonus": args.controlled_descent_bonus,
        "success_bonus": args.success_bonus,
        "failure_penalty": args.failure_penalty,
        "landing_target_bonus": args.landing_target_bonus,
        "landing_angle_bonus": args.landing_angle_bonus,
        "post_descent_landing_bonus": args.post_descent_landing_bonus,
        "post_descent_protocol_miss_penalty": args.post_descent_protocol_miss_penalty,
        "early_stop_protocol_rate": None,
        "early_stop_landing_rate": None,
        "early_stop_patience": 1,
    }

    stage1_config = LunarLanderPPOTrainingConfig(
        **common_config,
        total_timesteps=args.stage1_timesteps,
        learning_rate=args.stage1_learning_rate,
        linear_learning_rate_decay=args.stage1_linear_learning_rate_decay,
        target_kl=args.stage1_target_kl,
        initial_model=args.stage1_initial_model,
        output_dir=stage1_dir,
    )
    stage1_summary = train_lunar_lander_ppo(
        stage1_config,
        policy_config=policy_config,
        monitor_config_template=args.monitor_config,
        monitor_spec_path=args.monitor_spec,
    )

    stage2_initial_model = _stage1_best_checkpoint(stage1_dir)
    stage2_config = LunarLanderPPOTrainingConfig(
        **common_config,
        total_timesteps=args.stage2_timesteps,
        learning_rate=args.stage2_learning_rate,
        linear_learning_rate_decay=args.stage2_linear_learning_rate_decay,
        target_kl=args.stage2_target_kl,
        initial_model=stage2_initial_model,
        output_dir=stage2_dir,
    )
    stage2_summary = train_lunar_lander_ppo(
        stage2_config,
        policy_config=policy_config,
        monitor_config_template=args.monitor_config,
        monitor_spec_path=args.monitor_spec,
    )

    combined_metrics_path = run_dir / "combined_eval_metrics.csv"
    _write_combined_eval_metrics(
        combined_metrics_path,
        stage1_dir / "eval_metrics.csv",
        stage2_dir / "eval_metrics.csv",
        stage1_timesteps=args.stage1_timesteps,
    )

    summary = {
        "completed_at_utc": utc_now(),
        "algorithm": "ppo",
        "experiment": "lunar_lander_protocol_ppo_two_stage",
        "run_name": run_name,
        "seed": args.seed,
        "stage1": {
            "name": "discovery",
            "summary": stage1_summary,
            "output_dir": str(stage1_dir),
        },
        "stage2": {
            "name": "stabilization",
            "initial_model": str(stage2_initial_model),
            "summary": stage2_summary,
            "output_dir": str(stage2_dir),
        },
        "artifacts": {
            "summary": str(run_dir / "summary.json"),
            "combined_eval_metrics": str(combined_metrics_path),
            "stage1_best_model": str(stage1_dir / "best_model.zip"),
            "stage1_final_model": str(stage1_dir / "model_final.zip"),
            "stage2_best_model": str(stage2_dir / "best_model.zip"),
            "stage2_final_model": str(stage2_dir / "model_final.zip"),
        },
        "config": json_ready(
            {
                "stage1": asdict(stage1_config),
                "stage2": asdict(stage2_config),
                "policy": asdict(policy_config),
                "monitor_config": args.monitor_config,
                "monitor_spec": args.monitor_spec,
            }
        ),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _stage1_best_checkpoint(stage1_dir: Path) -> Path:
    best_model = stage1_dir / "best_model.zip"
    if best_model.exists():
        return best_model
    final_model = stage1_dir / "model_final.zip"
    if final_model.exists():
        return final_model
    raise FileNotFoundError(
        f"Stage 1 did not produce a checkpoint in {stage1_dir}."
    )


def _write_combined_eval_metrics(
    output_path: Path,
    stage1_metrics_path: Path,
    stage2_metrics_path: Path,
    *,
    stage1_timesteps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for stage, path, offset in (
        ("stage1_discovery", stage1_metrics_path, 0),
        ("stage2_stabilization", stage2_metrics_path, stage1_timesteps),
    ):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = ["stage", "global_training_steps", *reader.fieldnames]
            for row in reader:
                global_step = offset + int(float(row["training_steps"]))
                rows.append(
                    {
                        "stage": stage,
                        "global_training_steps": str(global_step),
                        **row,
                    }
                )

    if fieldnames is None:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=["semantic_progress"], default="semantic_progress")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--stage1-timesteps", type=int, default=1_000_000)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage1-linear-learning-rate-decay", action="store_true")
    parser.add_argument("--stage1-target-kl", type=float, default=None)
    parser.add_argument(
        "--stage1-initial-model",
        type=Path,
        default=None,
        help="Optional checkpoint for the discovery stage.",
    )

    parser.add_argument("--stage2-timesteps", type=int, default=300_000)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-linear-learning-rate-decay", action="store_true")
    parser.add_argument("--stage2-target-kl", type=float, default=None)

    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)

    parser.add_argument("--monitor-progress-bonus", type=float, default=20.0)
    parser.add_argument("--hover-step-bonus", type=float, default=2.0)
    parser.add_argument("--hover-complete-bonus", type=float, default=30.0)
    parser.add_argument("--controlled-descent-bonus", type=float, default=20.0)
    parser.add_argument("--success-bonus", type=float, default=200.0)
    parser.add_argument("--failure-penalty", type=float, default=-100.0)
    parser.add_argument("--landing-target-bonus", type=float, default=0.0)
    parser.add_argument("--landing-angle-bonus", type=float, default=0.0)
    parser.add_argument("--post-descent-landing-bonus", type=float, default=0.0)
    parser.add_argument("--post-descent-protocol-miss-penalty", type=float, default=0.0)

    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--position-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-embedding-dim", type=int, default=16)
    parser.add_argument("--max-monitor-states", type=int, default=256)
    parser.add_argument("--net-arch-width", type=int, default=128)

    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    parser.add_argument("--monitor-spec", type=Path, default=DEFAULT_MONITOR_SPEC)
    return parser.parse_args()


def main() -> None:
    summary = run_two_stage_lunar_lander_ppo(parse_args())
    stage1_best = (
        summary["stage1"]["summary"].get("best_evaluation") or {}
    ).get("eval_successful_protocol_rate")
    stage2_final = (
        summary["stage2"]["summary"].get("final_evaluation") or {}
    ).get("eval_successful_protocol_rate")
    stage2_best = (
        summary["stage2"]["summary"].get("best_evaluation") or {}
    ).get("eval_successful_protocol_rate")
    print(
        "Completed two-stage LunarLander PPO run: "
        f"stage1_best_protocol={stage1_best}, "
        f"stage2_best_protocol={stage2_best}, "
        f"stage2_final_protocol={stage2_final}, "
        f"output={summary['artifacts']['summary']}"
    )


if __name__ == "__main__":
    main()
