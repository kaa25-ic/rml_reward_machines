"""Generate CSTR RML monitor-state corpora for graph encoder pretraining."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shlex
import sys
from typing import Any, Callable

import numpy as np

from envs.cstr import CSTRConfig, RMLCSTRConfig, make_rml_cstr_env
from envs.cstr.reference_automaton import event_label
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.monitors import RMLMonitorProcess, find_free_port


PolicyFn = Callable[[dict[str, Any], np.random.Generator, int], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-policy", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--soak-steps", type=int, default=10)
    parser.add_argument("--deadline-steps", type=int, default=300)
    parser.add_argument("--concentration-tolerance", type=float, default=0.04)
    parser.add_argument("--production-temp-low", type=float, default=348.0)
    parser.add_argument("--production-temp-high", type=float, default=352.0)
    parser.add_argument("--recover-variants", choices=("strict", "recover", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-m", __name__, *sys.argv[1:]])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    variants = {
        "strict": (False,),
        "recover": (True,),
        "both": (False, True),
    }[args.recover_variants]
    for recover in variants:
        variant_rows, summary = _collect_variant(args, recover_from_regulation_failure=recover)
        rows.extend(variant_rows)
        summaries.append(summary)

    corpus_path = args.output_dir / "monitor_states.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_payload = {
        "corpus_path": str(corpus_path),
        "num_rows": len(rows),
        "recover_variants": args.recover_variants,
        "variant_summaries": summaries,
        "config": vars(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True, default=str))


def _collect_variant(
    args: argparse.Namespace,
    *,
    recover_from_regulation_failure: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(args.seed) + (10_000 if recover_from_regulation_failure else 0))
    random.seed(int(args.seed))
    port = find_free_port()
    variant_name = "recover" if recover_from_regulation_failure else "strict"
    generated = generate_cstr_rml(
        soak_steps=args.soak_steps,
        recover_from_regulation_failure=recover_from_regulation_failure,
        port=port,
        max_episode_steps=args.max_episode_steps,
        generated_root=args.output_dir / "monitors" / variant_name,
    )
    monitor = RMLMonitorProcess(
        generated.spec_path,
        port=port,
        log_path=args.output_dir / f"{variant_name}_rml_monitor.log",
    ).start()
    try:
        cstr_config = CSTRConfig(
            max_episode_steps=args.max_episode_steps,
            soak_steps=args.soak_steps,
            deadline_steps=args.deadline_steps,
            concentration_tolerance=args.concentration_tolerance,
            production_temp_low=args.production_temp_low,
            production_temp_high=args.production_temp_high,
            randomize_initial_state=False,
            randomize_setpoint=False,
            enable_disturbance=False,
            heating_rate_penalty=0.0,
        )
        env_config = RMLCSTRConfig(
            cstr_env=cstr_config,
            observation_mode="semantic_progress",
            reward_mode="env_rml",
            config_path=generated.config_path,
            monitor_port=port,
            soak_steps=args.soak_steps,
            recover_from_regulation_failure=recover_from_regulation_failure,
            terminate_on_rml_failure=True,
            monitor_state_limit=max(args.soak_steps + 6, 16),
        )
        env = make_rml_cstr_env(env_config)
        policies: dict[str, PolicyFn] = {
            "staged_success": _staged_policy,
            "reckless": _reckless_policy,
            "cooling": _cooling_policy,
            "oscillating": _oscillating_policy,
            "random": _random_policy,
        }
        rows: list[dict[str, Any]] = []
        phase_counts: dict[str, int] = {}
        verdict_counts: dict[str, int] = {}
        trace_index = 0
        for policy_name, policy in policies.items():
            for episode in range(int(args.episodes_per_policy)):
                seed = int(args.seed) + trace_index
                trace_rows = _rollout(
                    env,
                    policy,
                    rng,
                    trace_index=trace_index,
                    trace_label=f"{variant_name}_{policy_name}",
                    task_id=f"cstr_startup_{variant_name}",
                    seed=seed,
                    recover_from_regulation_failure=recover_from_regulation_failure,
                    cstr_config=cstr_config,
                )
                rows.extend(trace_rows)
                for row in trace_rows:
                    phase_counts[row["canonical_state"]] = phase_counts.get(row["canonical_state"], 0) + 1
                    verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
                trace_index += 1
        env.close()
        return rows, {
            "variant": variant_name,
            "num_rows": len(rows),
            "num_traces": trace_index,
            "phase_counts": phase_counts,
            "verdict_counts": verdict_counts,
            "env_config": asdict(env_config),
        }
    finally:
        monitor.stop()


def _rollout(
    env: Any,
    policy: PolicyFn,
    rng: np.random.Generator,
    *,
    trace_index: int,
    trace_label: str,
    task_id: str,
    seed: int,
    recover_from_regulation_failure: bool,
    cstr_config: CSTRConfig,
) -> list[dict[str, Any]]:
    observation, info = env.reset(seed=seed)
    del observation
    rows = [
        _row_from_info(
            info,
            event="RESET",
            step_index=0,
            trace_index=trace_index,
            trace_label=trace_label,
            task_id=task_id,
            seed=seed,
            recover_from_regulation_failure=recover_from_regulation_failure,
        )
    ]
    terminated = False
    truncated = False
    step_index = 0
    while not (terminated or truncated):
        action = policy(info, rng, step_index)
        _observation, _reward, terminated, truncated, info = env.step(action)
        step_index += 1
        rows.append(
            _row_from_info(
                info,
                event=_event_from_info(info, terminate=bool(truncated)),
                step_index=step_index,
                trace_index=trace_index,
                trace_label=trace_label,
                task_id=task_id,
                seed=seed,
                recover_from_regulation_failure=recover_from_regulation_failure,
            )
        )
        if step_index >= int(cstr_config.max_episode_steps):
            break
    return rows


def _row_from_info(
    info: dict[str, Any],
    *,
    event: str,
    step_index: int,
    trace_index: int,
    trace_label: str,
    task_id: str,
    seed: int,
    recover_from_regulation_failure: bool,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_index": int(recover_from_regulation_failure),
        "trace_index": int(trace_index),
        "trace_label": trace_label,
        "step_index": int(step_index),
        "seed": int(seed),
        "event": str(event),
        "monitor_state": str(info.get("monitor_state_unencoded", "")),
        "normalized_monitor_state": str(info.get("rml_monitor_state_unencoded_normalized", "")),
        "canonical_state": str(info.get("rml_monitor_state_normalized", "")),
        "phase_label": str(info.get("rml_monitor_state_normalized", "")),
        "phase": str(info.get("monitor_phase", "")),
        "verdict": str(info.get("monitor_verdict", "")),
        "recover_from_regulation_failure": bool(recover_from_regulation_failure),
        "ca": float(info.get("reactor_concentration", info.get("ca", 0.0))),
        "temp": float(info.get("reactor_temperature", info.get("temperature", 0.0))),
        "coolant": float(info.get("action_coolant_temp", 0.0)),
        "tracking_error": float(info.get("tracking_error", 0.0)),
        "stable": bool(info.get("event_stable_step", False)),
        "in_soak": bool(info.get("event_in_soak_band", False)),
        "critical": bool(info.get("event_temp_critical", False)),
        "overshoot": bool(info.get("event_overshoot", False)),
    }


def _event_from_info(info: dict[str, Any], *, terminate: bool) -> str:
    payload = {
        "critical": info.get("event_temp_critical", False),
        "temp_safe": info.get("event_temp_safe", False),
        "stable": info.get("event_stable_step", False),
        "in_soak_band": info.get("event_in_soak_band", False),
        "overshoot": info.get("event_overshoot", False),
        "past_deadline": info.get("event_past_deadline", False),
        "terminate": bool(terminate),
    }
    return event_label(payload)


def _to_action(coolant: float, info: dict[str, Any]) -> np.ndarray:
    config = info.get("config", {})
    low = float(config.get("action_low", 250.0)) if isinstance(config, dict) else 250.0
    high = float(config.get("action_high", 350.0)) if isinstance(config, dict) else 350.0
    action = 2.0 * ((float(coolant) - low) / (high - low)) - 1.0
    return np.asarray([np.clip(action, -1.0, 1.0)], dtype=np.float32)


def _staged_policy(info: dict[str, Any], _rng: np.random.Generator, _step: int) -> np.ndarray:
    config = info.get("config", {})
    ca_setpoint = float(config.get("ca_setpoint", 0.5)) if isinstance(config, dict) else 0.5
    temp = float(info.get("reactor_temperature", 331.0))
    ca = float(info.get("reactor_concentration", 0.8))
    phase = str(info.get("monitor_phase", "preheat"))
    if phase in {"preheat", "soak"}:
        coolant = 300.0 + 3.0 * (345.0 - temp)
    else:
        coolant = 300.0 + 5.0 * (350.0 - temp) + 20.0 * (ca - ca_setpoint)
    return _to_action(coolant, info)


def _reckless_policy(info: dict[str, Any], _rng: np.random.Generator, step: int) -> np.ndarray:
    return np.asarray([1.0 if step < 10 else -1.0], dtype=np.float32)


def _cooling_policy(_info: dict[str, Any], _rng: np.random.Generator, _step: int) -> np.ndarray:
    return np.asarray([-1.0], dtype=np.float32)


def _oscillating_policy(_info: dict[str, Any], _rng: np.random.Generator, step: int) -> np.ndarray:
    return np.asarray([0.8 if (step // 4) % 2 == 0 else -0.8], dtype=np.float32)


def _random_policy(_info: dict[str, Any], rng: np.random.Generator, _step: int) -> np.ndarray:
    return np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)


if __name__ == "__main__":
    main()
