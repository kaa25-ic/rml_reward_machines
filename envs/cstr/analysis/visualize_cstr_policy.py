"""Replay a saved CSTR PPO policy and save trajectory visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from envs.cstr.experiments.train_cstr_ppo import (
    CSTRPPOConfig,
    ENV_VARIANTS,
    EXTERNAL_MONITOR_VARIANTS,
    REWARD_MODES,
    make_env,
)
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.monitors import RMLMonitorProcess, find_free_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--env-variant", choices=ENV_VARIANTS, required=True)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="env_rml")
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
    summary = replay_policy(args)
    print(json.dumps(summary, indent=2))


def replay_policy(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

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
        output_dir=args.output_dir,
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
            generated_root=args.output_dir / "monitor",
        )
        config_path = generated.config_path
        monitor_process = RMLMonitorProcess(
            generated.spec_path,
            port=monitor_port,
            log_path=args.output_dir / "rml_monitor.log",
        ).start()

    try:
        env = make_env(config, monitor_port=monitor_port, config_path=config_path)
        model = PPO.load(str(args.model_path))
        rows, episode_summary = _run_episode(env, model, seed=args.seed)
        env.close()
    finally:
        if monitor_process is not None:
            monitor_process.stop()

    _write_trace_csv(args.output_dir / "trajectory.csv", rows)
    _plot_trace(args.output_dir / "trajectory.png", rows, concentration_tolerance=args.concentration_tolerance)

    payload = {
        "experiment": "cstr_rml_policy_visualization",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "model_path": str(args.model_path),
        "env_variant": args.env_variant,
        "reward_mode": args.reward_mode,
        "seed": args.seed,
        "regulation_violation_steps": args.regulation_violation_steps,
        "soak_steps": args.soak_steps,
        "summary": episode_summary,
        "paths": {
            "trajectory_csv": str(args.output_dir / "trajectory.csv"),
            "trajectory_png": str(args.output_dir / "trajectory.png"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _run_episode(env: Any, model: PPO, *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation, info = env.reset(seed=seed)
    rows: list[dict[str, Any]] = [_row_from_info(info, action=np.nan, reward=0.0, terminated=False, truncated=False)]
    terminated = False
    truncated = False
    episode_return = 0.0
    while not (terminated or truncated):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        rows.append(
            _row_from_info(
                info,
                action=float(np.asarray(action).reshape(-1)[0]),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    final_info = rows[-1]
    full_episode_safe = (
        bool(final_info["terminated"] or final_info["truncated"])
        and not bool(final_info["critical_failure"])
        and float(final_info["cumulative_temperature_violation"]) <= 1e-9
    )
    terminal_stable = bool(final_info["stable_step"])
    physical_success = bool(full_episode_safe and terminal_stable)
    summary = {
        "steps": int(final_info["step"]),
        "episode_return": episode_return,
        "physical_success": physical_success,
        "full_episode_safe": bool(full_episode_safe),
        "terminal_stable": terminal_stable,
        "rml_success": bool(final_info["monitor_success"]),
        "monitor_failed": bool(final_info["monitor_failed"]),
        "monitor_soak_steps": int(final_info["monitor_soak_steps"]),
        "monitor_violation_steps": int(final_info["monitor_violation_steps"]),
        "critical_failure": bool(final_info["critical_failure"]),
        "warning_events": int(final_info["warning_events"]),
        "max_stable_steps": int(final_info["max_stable_steps"]),
        "first_stable_step": int(final_info["first_stable_step"]),
        "final_tracking_error": float(final_info["tracking_error"]),
        "final_temperature_violation": float(final_info["temperature_violation"]),
        "terminated": bool(final_info["terminated"]),
        "truncated": bool(final_info["truncated"]),
    }
    return rows, summary


def _row_from_info(
    info: dict[str, Any],
    *,
    action: float,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "step": int(info.get("steps", 0)),
        "reactor_concentration": float(info.get("reactor_concentration", 0.0)),
        "target_concentration": float(info.get("target_concentration", 0.0)),
        "reactor_temperature": float(info.get("reactor_temperature", 0.0)),
        "coolant_action_normalized": float(action),
        "coolant_temperature": float(info.get("action_coolant_temp", np.nan)),
        "reward": float(reward),
        "base_reward": float(info.get("base_reward", reward)),
        "rml_reward": float(info.get("rml_reward", info.get("monitor_reward", 0.0))),
        "tracking_error": float(info.get("tracking_error", 0.0)),
        "temperature_violation": float(info.get("temperature_violation", 0.0)),
        "cumulative_temperature_violation": float(info.get("cumulative_temperature_violation", 0.0)),
        "heating_rate": float(info.get("heating_rate", 0.0)),
        "in_soak_band": bool(info.get("event_in_soak_band", False)),
        "overshoot": bool(info.get("event_overshoot", False)),
        "heating_rate_exceeded": bool(info.get("event_heating_rate_exceeded", False)),
        "past_deadline": bool(info.get("event_past_deadline", False)),
        "deadline_expired": bool(info.get("event_deadline_expired", False)),
        "stable_step": bool(info.get("stable_step", False)),
        "warning_events": int(info.get("warning_events", 0)),
        "critical_failure": bool(info.get("critical_events", 0) > 0),
        "max_stable_steps": int(info.get("max_stable_steps", 0)),
        "first_stable_step": int(info.get("first_stable_step", -1)),
        "monitor_phase": str(info.get("monitor_phase", "none")),
        "rml_monitor_state_normalized": str(info.get("rml_monitor_state_normalized", "")),
        "rml_monitor_state_unencoded_normalized": str(info.get("rml_monitor_state_unencoded_normalized", "")),
        "monitor_soak_steps": int(info.get("monitor_soak_steps", 0)),
        "monitor_violation_steps": int(info.get("monitor_violation_steps", 0)),
        "monitor_success": bool(info.get("monitor_success", False)),
        "monitor_failed": bool(info.get("monitor_failed", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trace(path: Path, rows: list[dict[str, Any]], *, concentration_tolerance: float) -> None:
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    ca = np.asarray([row["reactor_concentration"] for row in rows], dtype=float)
    target = np.asarray([row["target_concentration"] for row in rows], dtype=float)
    temp = np.asarray([row["reactor_temperature"] for row in rows], dtype=float)
    coolant = np.asarray([row["coolant_temperature"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

    axes[0].plot(steps, ca, color="#2b6cb0", label="Concentration")
    axes[0].plot(steps, target, color="#2f855a", linestyle="--", label="Target")
    axes[0].fill_between(
        steps,
        target - float(concentration_tolerance),
        target + float(concentration_tolerance),
        color="#2f855a",
        alpha=0.12,
        label="Target band",
    )
    axes[0].set_ylabel("C_A")
    axes[0].legend(loc="best")

    axes[1].plot(steps, temp, color="#c05621", label="Temperature")
    axes[1].axhspan(315.0, 375.0, color="#38a169", alpha=0.12, label="Safe band")
    axes[1].axhline(382.0, color="#dd6b20", linestyle="--", linewidth=1.2, label="Warning")
    axes[1].axhline(405.0, color="#c53030", linestyle="--", linewidth=1.2, label="Critical high")
    axes[1].set_ylabel("Temp")
    axes[1].legend(loc="best")

    axes[2].plot(steps, coolant, color="#805ad5", label="Coolant temp")
    axes[2].set_ylabel("Coolant")
    axes[2].set_xlabel("Step")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
