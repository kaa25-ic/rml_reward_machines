"""Evaluate saved PPO checkpoints on native or RML-backed CSTR variants."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from envs.cstr.experiments.train_cstr_ppo import (
    CSTRPPOConfig,
    ENV_VARIANTS,
    EXTERNAL_MONITOR_VARIANTS,
    REWARD_MODES,
    deterministic_model_policy,
    evaluate_cstr_policy,
    make_env,
    write_eval_outputs,
    write_json,
)
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.monitors import RMLMonitorProcess, find_free_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--env-variant", choices=ENV_VARIANTS, required=True)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--regulation-violation-steps", type=int, default=10)
    parser.add_argument("--soak-steps", type=int, default=10)
    parser.add_argument("--monitor-state-limit", type=int, default=16)
    parser.add_argument("--graph-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--safe-step-bonus", type=float, default=0.10)
    parser.add_argument("--stable-step-bonus", type=float, default=1.0)
    parser.add_argument("--regulation-entry-bonus", type=float, default=5.0)
    parser.add_argument("--success-bonus", type=float, default=50.0)
    parser.add_argument("--failure-penalty", type=float, default=-50.0)
    parser.add_argument("--rml-heating-rate-penalty", type=float, default=0.02)
    parser.add_argument("--preheat-distance-weight", type=float, default=0.08)
    parser.add_argument("--preheat-warming-weight", type=float, default=0.25)
    parser.add_argument("--soak-entry-bonus", type=float, default=5.0)
    parser.add_argument("--soak-progress-bonus", type=float, default=0.75)
    parser.add_argument("--soak-reset-penalty", type=float, default=-3.0)
    parser.add_argument("--soak-lost-step-penalty", type=float, default=0.50)
    parser.add_argument("--approach-distance-weight", type=float, default=1.0)
    parser.add_argument("--approach-progress-bonus", type=float, default=5.0)
    parser.add_argument("--approach-ca-progress-bonus", type=float, default=4.0)
    parser.add_argument("--approach-temp-progress-bonus", type=float, default=4.0)
    parser.add_argument("--approach-warming-weight", type=float, default=0.50)
    parser.add_argument("--production-entry-bonus", type=float, default=10.0)
    parser.add_argument("--regulate-recovery-penalty", type=float, default=-10.0)
    parser.add_argument("--deadline-steps", type=int, default=60)
    parser.add_argument("--tracking-weight", type=float, default=0.5)
    parser.add_argument("--heating-rate-penalty", type=float, default=0.02)
    parser.add_argument("--critical-penalty", type=float, default=200.0)
    parser.add_argument("--production-temp-low", type=float, default=348.0)
    parser.add_argument("--production-temp-high", type=float, default=352.0)
    parser.add_argument("--concentration-tolerance", type=float, default=0.04)
    parser.add_argument("--ca-overshoot-low", type=float, default=0.44)
    parser.add_argument("--require-soak-concentration-band", action="store_true")
    parser.add_argument("--soak-concentration-low", type=float, default=0.58)
    parser.add_argument("--soak-concentration-high", type=float, default=0.74)
    parser.add_argument("--fixed-initial-state", action="store_true", default=True)
    parser.add_argument("--randomize-initial-state", action="store_true")
    parser.add_argument("--randomize-setpoint", action="store_true")
    parser.add_argument("--enable-disturbance", action="store_true")
    parser.add_argument("--temp-weight", type=float, default=0.015)
    parser.add_argument("--action-weight", type=float, default=0.0002)
    parser.add_argument("--warning-penalty", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_checkpoint(args)
    print(json.dumps(summary, indent=2))


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    config = CSTRPPOConfig(
        env_variant=args.env_variant,
        reward_mode=args.reward_mode,
        max_episode_steps=args.max_episode_steps,
        regulation_violation_steps=args.regulation_violation_steps,
        soak_steps=args.soak_steps,
        monitor_state_limit=args.monitor_state_limit,
        graph_encoder_checkpoint=args.graph_encoder_checkpoint,
        safe_step_bonus=args.safe_step_bonus,
        stable_step_bonus=args.stable_step_bonus,
        regulation_entry_bonus=args.regulation_entry_bonus,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        rml_heating_rate_penalty=args.rml_heating_rate_penalty,
        preheat_distance_weight=args.preheat_distance_weight,
        preheat_warming_weight=args.preheat_warming_weight,
        soak_entry_bonus=args.soak_entry_bonus,
        soak_progress_bonus=args.soak_progress_bonus,
        soak_reset_penalty=args.soak_reset_penalty,
        soak_lost_step_penalty=args.soak_lost_step_penalty,
        approach_distance_weight=args.approach_distance_weight,
        approach_progress_bonus=args.approach_progress_bonus,
        approach_ca_progress_bonus=args.approach_ca_progress_bonus,
        approach_temp_progress_bonus=args.approach_temp_progress_bonus,
        approach_warming_weight=args.approach_warming_weight,
        production_entry_bonus=args.production_entry_bonus,
        regulate_recovery_penalty=args.regulate_recovery_penalty,
        deadline_steps=args.deadline_steps,
        tracking_weight=args.tracking_weight,
        heating_rate_penalty=args.heating_rate_penalty,
        critical_penalty=args.critical_penalty,
        production_temp_low=args.production_temp_low,
        production_temp_high=args.production_temp_high,
        concentration_tolerance=args.concentration_tolerance,
        ca_overshoot_low=args.ca_overshoot_low,
        require_soak_concentration_band=args.require_soak_concentration_band,
        soak_concentration_low=args.soak_concentration_low,
        soak_concentration_high=args.soak_concentration_high,
        randomize_initial_state=bool(args.randomize_initial_state) or not args.fixed_initial_state,
        randomize_setpoint=args.randomize_setpoint,
        enable_disturbance=args.enable_disturbance,
        temp_weight=args.temp_weight,
        action_weight=args.action_weight,
        warning_penalty=args.warning_penalty,
        output_dir=output_dir,
    )

    monitor_process = None
    monitor_port = None
    config_path = None
    if args.env_variant in EXTERNAL_MONITOR_VARIANTS:
        monitor_port = find_free_port()
        generated = generate_cstr_rml(
            regulation_violation_steps=args.regulation_violation_steps,
            soak_steps=args.soak_steps,
            port=monitor_port,
            max_episode_steps=args.max_episode_steps,
            generated_root=output_dir / "monitor",
        )
        config_path = generated.config_path
        monitor_process = RMLMonitorProcess(
            generated.spec_path,
            port=monitor_port,
            log_path=output_dir / "rml_monitor.log",
        ).start()

    try:
        env = make_env(config, monitor_port=monitor_port, config_path=config_path)
        model = PPO.load(str(model_path))
        records, eval_summary = evaluate_cstr_policy(
            env,
            deterministic_model_policy(model),
            episodes=args.episodes,
            seed=args.seed,
        )
        write_eval_outputs(output_dir, records, eval_summary)
        env.close()

        payload = {
            "experiment": "cstr_rml_ppo_checkpoint_eval",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.monotonic() - started,
            "model_path": str(model_path),
            "env_variant": args.env_variant,
            "reward_mode": args.reward_mode,
            "episodes": args.episodes,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "regulation_violation_steps": args.regulation_violation_steps,
            "soak_steps": args.soak_steps,
            "safe_step_bonus": args.safe_step_bonus,
            "stable_step_bonus": args.stable_step_bonus,
            "regulation_entry_bonus": args.regulation_entry_bonus,
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "rml_heating_rate_penalty": args.rml_heating_rate_penalty,
            "preheat_distance_weight": args.preheat_distance_weight,
            "preheat_warming_weight": args.preheat_warming_weight,
            "soak_entry_bonus": args.soak_entry_bonus,
            "soak_progress_bonus": args.soak_progress_bonus,
            "soak_reset_penalty": args.soak_reset_penalty,
            "soak_lost_step_penalty": args.soak_lost_step_penalty,
            "approach_distance_weight": args.approach_distance_weight,
            "approach_progress_bonus": args.approach_progress_bonus,
            "approach_ca_progress_bonus": args.approach_ca_progress_bonus,
            "approach_temp_progress_bonus": args.approach_temp_progress_bonus,
            "approach_warming_weight": args.approach_warming_weight,
            "production_entry_bonus": args.production_entry_bonus,
            "regulate_recovery_penalty": args.regulate_recovery_penalty,
            "deadline_steps": args.deadline_steps,
            "tracking_weight": args.tracking_weight,
            "heating_rate_penalty": args.heating_rate_penalty,
            "critical_penalty": args.critical_penalty,
            "production_temp_low": args.production_temp_low,
            "production_temp_high": args.production_temp_high,
            "concentration_tolerance": args.concentration_tolerance,
            "ca_overshoot_low": args.ca_overshoot_low,
            "require_soak_concentration_band": args.require_soak_concentration_band,
            "soak_concentration_low": args.soak_concentration_low,
            "soak_concentration_high": args.soak_concentration_high,
            "monitor_port": monitor_port,
            "temp_weight": args.temp_weight,
            "action_weight": args.action_weight,
            "warning_penalty": args.warning_penalty,
            "summary": eval_summary,
            "paths": {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "episode_metrics": str(output_dir / "episode_metrics.csv"),
            },
        }
        write_json(output_dir / "summary.json", payload)
        return payload
    finally:
        if monitor_process is not None:
            monitor_process.stop()


if __name__ == "__main__":
    main()
