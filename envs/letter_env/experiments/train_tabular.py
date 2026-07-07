"""Run tabular Q-learning reproduction experiments on LetterEnv."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from envs.letter_env import LetterAction, LetterEnvConfig, build_letter_env
from rml_rm.agents.tabular import QLearningAgent, QLearningConfig
from rml_rm.experiments.runtime import (
    json_ready,
    managed_monitor,
    utc_now,
    write_json,
)
from rml_rm.wrappers import tabular_state_key


REPO_ROOT = Path(__file__).resolve().parents[3]
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_MONITOR_CONFIG = LETTER_ENV_ROOT / "configs" / "letter_env.yaml"
DEFAULT_MONITOR_SPEC = LETTER_ENV_ROOT / "specs" / "letter_env_monitor.pl"
EXPERIMENT_NAME = "tabular_reproduction_from_previous_thesis"
DEFAULT_SUCCESS_REWARDS = (110.0, 112.0)


@dataclass(frozen=True)
class EpisodeRecord:
    """Metrics for one tabular training episode."""

    encoding: str
    iteration: int
    n_value: int
    episode: int
    seed: int
    total_reward: float
    final_reward: float
    steps: int
    epsilon: float
    q_state_count: int
    success: bool
    converged: bool


@dataclass(frozen=True)
class LetterEnvTabularTrainingConfig:
    """Configuration for the LetterEnv tabular reproduction."""

    encoding: str = "all"
    max_n: int = 10
    n_values: tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 11)))
    iterations: int = 1
    episodes: int = 1_000_000
    success_window: int = 20
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.4
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    state_discovery_bonus: float = 2.0
    seed_base: int = 0
    heartbeat_episodes: int = 100
    max_episode_steps: int = 200
    fixed_n: bool = True
    legacy_transition_bonus: float = 10.0
    output_dir: Path = field(default_factory=Path)


def train_letter_env_tabular(
    config: LetterEnvTabularTrainingConfig,
    *,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Run the tabular reproduction and write train metrics plus summary."""
    if not config.output_dir:
        raise ValueError("output_dir is required.")
    if any(n_value < 1 or n_value > config.max_n for n_value in config.n_values):
        raise ValueError("n_values must all be within 1..max_n.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_config = QLearningConfig(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        min_epsilon=config.min_epsilon,
    )
    encodings = _selected_encodings(config.encoding)

    started = time.monotonic()
    all_records: list[EpisodeRecord] = []
    with managed_monitor(
        output_dir=output_dir,
        monitor_config_template=monitor_config_template,
        monitor_spec_path=monitor_spec_path,
        max_episode_steps=config.max_episode_steps,
    ) as monitor_runtime:
        for encoding_index, encoding in enumerate(encodings):
            for iteration in range(config.iterations):
                for n_value in config.n_values:
                    seed = _derive_seed(
                        config.seed_base + encoding_index * 100_000,
                        iteration,
                        n_value,
                    )
                    print(
                        f"[{utc_now()}] starting encoding={encoding} "
                        f"iteration={iteration} n={n_value} seed={seed}",
                        flush=True,
                    )
                    all_records.extend(
                        _train_condition(
                            iteration=iteration,
                            encoding=encoding,
                            n_value=n_value,
                            config=config,
                            seed=seed,
                            runtime_config_path=monitor_runtime.config_path,
                            agent_config=agent_config,
                        )
                    )

    if not all_records:
        raise RuntimeError("No training records were produced.")

    _write_records(output_dir / "train_metrics.csv", all_records)
    summary = _build_summary(
        records=all_records,
        config=config,
        runtime_seconds=time.monotonic() - started,
        runtime_config_path=monitor_runtime.config_path,
        monitor_spec_path=monitor_spec_path,
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def _train_condition(
    *,
    iteration: int,
    encoding: str,
    n_value: int,
    config: LetterEnvTabularTrainingConfig,
    seed: int,
    runtime_config_path: Path,
    agent_config: QLearningConfig,
) -> list[EpisodeRecord]:
    rng = random.Random(seed)
    np.random.seed(seed)
    env = build_letter_env(
        LetterEnvConfig(
            encoding=encoding,
            n_value=n_value,
            fixed_n=n_value if config.fixed_n else None,
            max_episode_steps=config.max_episode_steps,
            monitor_progress_bonus=0.0,
            monitor_regression_penalty=0.0,
            neutralize_legacy_transition_bonus=False,
            legacy_transition_bonus=config.legacy_transition_bonus,
            step_penalty=0.0,
            no_op_penalty=0.0,
            state_discovery_bonus=0.0,
        ),
        monitor_config_path=runtime_config_path,
    )
    agent = QLearningAgent([action.value for action in LetterAction], agent_config, rng=rng)
    records: list[EpisodeRecord] = []
    recent_successes: list[bool] = []

    try:
        for episode in range(1, config.episodes + 1):
            reset_seed = seed if episode == 1 else None
            observation, _info = env.reset(seed=reset_seed)
            state = tabular_state_key(observation)
            agent.ensure_state(state)
            total_reward = 0.0
            final_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = agent.choose_action(state)
                next_observation, reward, terminated, truncated, _step_info = env.step(action)
                next_state = tabular_state_key(next_observation)
                new_state = agent.ensure_state(next_state)
                shaped_reward = float(reward)
                if new_state:
                    shaped_reward += config.state_discovery_bonus
                agent.update(state, action, shaped_reward, next_state)
                state = next_state
                total_reward += float(reward)
                final_reward = float(reward)
                steps += 1

            success = final_reward in DEFAULT_SUCCESS_REWARDS
            recent_successes.append(success)
            if len(recent_successes) > config.success_window:
                recent_successes.pop(0)
            converged = (
                len(recent_successes) == config.success_window
                and all(recent_successes)
            )
            records.append(
                EpisodeRecord(
                    encoding=encoding,
                    iteration=iteration,
                    n_value=n_value,
                    episode=episode,
                    seed=seed,
                    total_reward=total_reward,
                    final_reward=final_reward,
                    steps=steps,
                    epsilon=agent.epsilon,
                    q_state_count=len(agent.q_table),
                    success=success,
                    converged=converged,
                )
            )
            agent.decay_epsilon()

            if config.heartbeat_episodes > 0 and episode % config.heartbeat_episodes == 0:
                window = records[-config.success_window :]
                success_rate = sum(record.success for record in window) / len(window)
                print(
                    f"[{utc_now()}] encoding={encoding} iteration={iteration} "
                    f"n={n_value} episode={episode}/{config.episodes} "
                    f"steps={steps} reward={total_reward:.1f} epsilon={agent.epsilon:.4f} "
                    f"q_states={len(agent.q_table)} window_success={success_rate:.2f}",
                    flush=True,
                )

            if converged:
                break
    finally:
        env.close()

    return records


def _selected_encodings(encoding: str) -> tuple[str, ...]:
    if encoding == "all":
        return ("simple", "one_hot", "numerical")
    return (encoding,)


def _derive_seed(seed_base: int, iteration: int, n_value: int) -> int:
    return int(seed_base + iteration * 1000 + n_value)


def _write_records(path: Path, records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _build_summary(
    *,
    records: list[EpisodeRecord],
    config: LetterEnvTabularTrainingConfig,
    runtime_seconds: float,
    runtime_config_path: Path,
    monitor_spec_path: Path,
) -> dict[str, Any]:
    final_records_by_condition: dict[tuple[str, int, int], EpisodeRecord] = {}
    for record in records:
        final_records_by_condition[(record.encoding, record.iteration, record.n_value)] = record

    return {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": utc_now(),
        "runtime_seconds": runtime_seconds,
        "config": json_ready(asdict(config)),
        "monitor": {
            "config_path": str(runtime_config_path),
            "spec_path": str(monitor_spec_path),
        },
        "episode_count": len(records),
        "condition_count": len(final_records_by_condition),
        "converged_condition_count": sum(
            record.converged for record in final_records_by_condition.values()
        ),
        "final_by_condition": [
            asdict(record) for _condition, record in sorted(final_records_by_condition.items())
        ],
        "artifacts": {
            "train_metrics": str(config.output_dir / "train_metrics.csv"),
            "summary": str(config.output_dir / "summary.json"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=["simple", "one_hot", "numerical", "all"], default="all")
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--success-window", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--min-epsilon", type=float, default=0.01)
    parser.add_argument("--state-discovery-bonus", type=float, default=2.0)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--heartbeat-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--sample-n", action="store_true")
    parser.add_argument("--legacy-transition-bonus", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LETTER_ENV_ROOT / "results_and_evaluation" / EXPERIMENT_NAME,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = tuple(args.n_values or range(1, args.max_n + 1))
    config = LetterEnvTabularTrainingConfig(
        encoding=args.encoding,
        max_n=args.max_n,
        n_values=n_values,
        iterations=args.iterations,
        episodes=args.episodes,
        success_window=args.success_window,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        state_discovery_bonus=args.state_discovery_bonus,
        seed_base=args.seed_base,
        heartbeat_episodes=args.heartbeat_episodes,
        max_episode_steps=args.max_episode_steps,
        fixed_n=not args.sample_n,
        legacy_transition_bonus=args.legacy_transition_bonus,
        output_dir=args.output_dir,
    )
    summary = train_letter_env_tabular(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
