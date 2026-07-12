"""Train tabular Q-learning on randomized LetterEnv."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from envs.letter_env_core import LetterAction
from envs.randomized_letter_env import RandomizedLetterEnvConfig, build_randomized_letter_env
from envs.randomized_letter_env.experiments.train_ddqn import (
    DEFAULT_MONITOR_CONFIG,
    DEFAULT_MONITOR_SPEC,
)
from rml_rm.agents.tabular import QLearningAgent, QLearningConfig
from rml_rm.experiments.runtime import (
    configure_global_seed,
    json_ready,
    managed_monitor_pair,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOMIZED_ENV_ROOT = REPO_ROOT / "envs" / "randomized_letter_env"
DEFAULT_OUTPUT_ROOT = RANDOMIZED_ENV_ROOT / "results_and_evaluation" / "q_learning"


@dataclass(frozen=True)
class QLearningEvaluationRecord:
    """Aggregate metrics from one deterministic Q-learning policy evaluation."""

    training_episodes: int
    training_steps: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_mean_terminal_base_reward: float
    eval_mean_terminal_task_progress: float
    eval_task_failure_rate: float


@dataclass(frozen=True)
class QLearningTrainingRecord:
    """Training diagnostics recorded at fixed episode intervals."""

    episode: int
    training_steps: int
    epsilon: float
    q_state_count: int
    recent_mean_return: float
    recent_success_rate: float
    recent_task_failure_rate: float


@dataclass(frozen=True)
class RandomizedLetterEnvQLearningConfig:
    """Training configuration for one randomized LetterEnv Q-learning run."""

    encoding: str = "semantic_progress"
    n_value: int = 5
    fixed_n: int | None = None
    placement_mode: str = "regional"
    episodes: int = 50_000
    seed: int = 0
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.4
    epsilon_decay: float = 0.99995
    min_epsilon: float = 0.05
    eval_freq_episodes: int = 1_000
    train_log_freq_episodes: int = 1_000
    n_eval_episodes: int = 25
    eval_seed_base: int = 0
    max_episode_steps: int = 200
    monitor_progress_bonus: float = 10.0
    output_dir: Path = field(default_factory=Path)


def train_randomized_letter_env_q_learning(
    config: RandomizedLetterEnvQLearningConfig,
    *,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Train Q-learning and write metrics plus a compact summary."""
    if config.encoding not in {"one_hot", "numerical", "semantic_progress"}:
        raise ValueError("encoding must be 'one_hot', 'numerical', or 'semantic_progress'.")
    if not config.output_dir:
        raise ValueError("output_dir is required.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_global_seed(config.seed)
    rng = random.Random(config.seed)
    started = time.monotonic()

    agent = QLearningAgent(
        [action.value for action in LetterAction],
        QLearningConfig(
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            min_epsilon=config.min_epsilon,
        ),
        rng=rng,
    )
    training_records: list[QLearningTrainingRecord] = []
    evaluation_records: list[QLearningEvaluationRecord] = []
    recent_returns: list[float] = []
    recent_successes: list[float] = []
    recent_failures: list[float] = []
    training_steps = 0

    with managed_monitor_pair(
        output_dir=output_dir,
        monitor_config_template=monitor_config_template,
        monitor_spec_path=monitor_spec_path,
    ) as monitor_runtime:
        train_env = build_randomized_letter_env(
            _env_config(config, evaluation=False),
            monitor_config_path=monitor_runtime.train_config_path,
        )
        eval_env = build_randomized_letter_env(
            _env_config(config, evaluation=True),
            monitor_config_path=monitor_runtime.eval_config_path,
        )
        try:
            _write_run_config(
                output_dir / "config.json",
                config=config,
                train_config_path=monitor_runtime.train_config_path,
                eval_config_path=monitor_runtime.eval_config_path,
                monitor_spec_path=monitor_spec_path,
            )
            _write_eval_header(output_dir / "eval_metrics.csv")
            _write_train_header(output_dir / "train_metrics.csv")

            for episode in range(1, config.episodes + 1):
                episode_return, episode_length, success, failed = _train_episode(
                    env=train_env,
                    agent=agent,
                    seed=config.seed if episode == 1 else None,
                )
                training_steps += episode_length
                recent_returns.append(episode_return)
                recent_successes.append(1.0 if success else 0.0)
                recent_failures.append(1.0 if failed else 0.0)
                if len(recent_returns) > config.train_log_freq_episodes:
                    recent_returns.pop(0)
                    recent_successes.pop(0)
                    recent_failures.pop(0)

                agent.decay_epsilon()

                if config.train_log_freq_episodes > 0 and episode % config.train_log_freq_episodes == 0:
                    record = QLearningTrainingRecord(
                        episode=episode,
                        training_steps=training_steps,
                        epsilon=float(agent.epsilon),
                        q_state_count=len(agent.q_table),
                        recent_mean_return=float(np.mean(recent_returns)),
                        recent_success_rate=float(np.mean(recent_successes)),
                        recent_task_failure_rate=float(np.mean(recent_failures)),
                    )
                    training_records.append(record)
                    _append_record(output_dir / "train_metrics.csv", record)
                    print(
                        f"[{utc_now()}] episode={episode}/{config.episodes} "
                        f"steps={training_steps} epsilon={agent.epsilon:.4f} "
                        f"q_states={len(agent.q_table)} "
                        f"recent_success={record.recent_success_rate:.3f}",
                        flush=True,
                    )

                if config.eval_freq_episodes > 0 and episode % config.eval_freq_episodes == 0:
                    evaluation = _evaluate_policy(
                        env=eval_env,
                        agent=agent,
                        episode=episode,
                        training_steps=training_steps,
                        config=config,
                    )
                    evaluation_records.append(evaluation)
                    _append_record(output_dir / "eval_metrics.csv", evaluation)
        finally:
            train_env.close()
            eval_env.close()

    summary = _build_summary(
        config=config,
        agent=agent,
        training_records=training_records,
        evaluation_records=evaluation_records,
        runtime_seconds=time.monotonic() - started,
    )
    agent.save_q_table(output_dir / "q_table.pkl")
    write_json(output_dir / "summary.json", summary)
    return summary


def _env_config(
    config: RandomizedLetterEnvQLearningConfig,
    *,
    evaluation: bool,
) -> RandomizedLetterEnvConfig:
    return RandomizedLetterEnvConfig(
        encoding=config.encoding,
        n_value=config.n_value,
        fixed_n=config.fixed_n,
        max_episode_steps=config.max_episode_steps,
        monitor_progress_bonus=config.monitor_progress_bonus,
        placement_mode=config.placement_mode,
    )


def _train_episode(
    *,
    env,
    agent: QLearningAgent,
    seed: int | None,
) -> tuple[float, int, bool, bool]:
    observation, info = env.reset(seed=seed)
    state = _tabular_state_key(observation)
    agent.ensure_state(state)
    total_reward = 0.0
    episode_length = 0
    success = bool(info.get("success", False))
    failed = bool(info.get("task_failed", False))
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.choose_action(state)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = _tabular_state_key(next_observation)
        agent.ensure_state(next_state)
        agent.update(state, action, float(reward), next_state)
        state = next_state
        total_reward += float(reward)
        episode_length += 1
        success = bool(info.get("success", success))
        failed = bool(info.get("task_failed", failed))

    return total_reward, episode_length, success, failed


def _evaluate_policy(
    *,
    env,
    agent: QLearningAgent,
    episode: int,
    training_steps: int,
    config: RandomizedLetterEnvQLearningConfig,
) -> QLearningEvaluationRecord:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    failures: list[float] = []
    final_base_rewards: list[float] = []
    final_task_progress: list[float] = []

    for episode_index, n_value in _evaluation_cases(config):
        observation, info = env.reset(
            seed=config.eval_seed_base + episode_index,
            options={"n": n_value},
        )
        state = _tabular_state_key(observation)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        success = bool(info.get("success", False))
        failed = bool(info.get("task_failed", False))
        final_base_reward = 0.0
        final_task_index = 0.0

        while not terminated and not truncated:
            action = _greedy_action(agent, state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = _tabular_state_key(observation)
            episode_return += float(reward)
            episode_length += 1
            success = bool(info.get("success", success))
            failed = bool(info.get("task_failed", failed))
            final_base_reward = float(info.get("base_reward", reward))
            final_task_index = float(info.get("task_index", final_task_index))

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(1.0 if success else 0.0)
        failures.append(1.0 if failed else 0.0)
        final_base_rewards.append(final_base_reward)
        final_task_progress.append(final_task_index)

    return QLearningEvaluationRecord(
        training_episodes=int(episode),
        training_steps=int(training_steps),
        eval_mean_return=float(np.mean(returns)),
        eval_std_return=float(np.std(returns)),
        eval_mean_episode_length=float(np.mean(lengths)),
        eval_success_rate=float(np.mean(successes)),
        eval_mean_terminal_base_reward=float(np.mean(final_base_rewards)),
        eval_mean_terminal_task_progress=float(np.mean(final_task_progress)),
        eval_task_failure_rate=float(np.mean(failures)),
    )


def _evaluation_cases(config: RandomizedLetterEnvQLearningConfig) -> list[tuple[int, int]]:
    n_values = [config.fixed_n] if config.fixed_n is not None else list(range(1, config.n_value + 1))
    cases: list[tuple[int, int]] = []
    for index in range(config.n_eval_episodes):
        cases.append((index, int(n_values[index % len(n_values)])))
    return cases


def _tabular_state_key(observation: dict[str, np.ndarray]) -> Hashable:
    position = np.asarray(observation["position"], dtype=np.float32).reshape(-1)
    monitor = np.asarray(observation["monitor"], dtype=np.float32).reshape(-1)
    position_tuple = tuple(round(float(value), 6) for value in position)
    monitor_tuple = tuple(round(float(value), 6) for value in monitor)
    return position_tuple, monitor_tuple


def _greedy_action(agent: QLearningAgent, state: Hashable) -> int:
    agent.ensure_state(state)
    action_values = agent.q_table[state]
    max_value = max(action_values.values())
    best_actions = [action for action, value in action_values.items() if value == max_value]
    return int(sorted(best_actions)[0])


def _write_run_config(
    path: Path,
    *,
    config: RandomizedLetterEnvQLearningConfig,
    train_config_path: Path,
    eval_config_path: Path,
    monitor_spec_path: Path,
) -> None:
    write_json(
        path,
        {
            "experiment": "randomized_letter_env_q_learning",
            "started_at_utc": utc_now(),
            "training_config": json_ready(asdict(config)),
            "state_key": "(position_observation_tuple, monitor_encoding_tuple)",
            "monitor": {
                "train_config_path": str(train_config_path),
                "eval_config_path": str(eval_config_path),
                "spec_path": str(monitor_spec_path),
            },
        },
    )


def _build_summary(
    *,
    config: RandomizedLetterEnvQLearningConfig,
    agent: QLearningAgent,
    training_records: list[QLearningTrainingRecord],
    evaluation_records: list[QLearningEvaluationRecord],
    runtime_seconds: float,
) -> dict[str, Any]:
    final_evaluation = asdict(evaluation_records[-1]) if evaluation_records else None
    best_evaluation = None
    if evaluation_records:
        best = max(
            evaluation_records,
            key=lambda record: (record.eval_success_rate, record.eval_mean_return),
        )
        best_evaluation = asdict(best)
    return {
        "completed_at_utc": utc_now(),
        "algorithm": "q_learning",
        "encoding": config.encoding,
        "n_value": config.n_value,
        "fixed_n": config.fixed_n,
        "placement_mode": config.placement_mode,
        "episodes": config.episodes,
        "seed": config.seed,
        "runtime_seconds": runtime_seconds,
        "agent": agent.as_serializable(),
        "training_record_count": len(training_records),
        "evaluation_count": len(evaluation_records),
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "artifacts": {
            "config": str(config.output_dir / "config.json"),
            "summary": str(config.output_dir / "summary.json"),
            "train_metrics": str(config.output_dir / "train_metrics.csv"),
            "eval_metrics": str(config.output_dir / "eval_metrics.csv"),
            "q_table": str(config.output_dir / "q_table.pkl"),
        },
    }


def _write_train_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(_empty_training_record()).keys()))
        writer.writeheader()


def _write_eval_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(_empty_evaluation_record()).keys()))
        writer.writeheader()


def _append_record(path: Path, record: Any) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(record).keys()))
        writer.writerow(asdict(record))


def _empty_training_record() -> QLearningTrainingRecord:
    return QLearningTrainingRecord(
        episode=0,
        training_steps=0,
        epsilon=0.0,
        q_state_count=0,
        recent_mean_return=0.0,
        recent_success_rate=0.0,
        recent_task_failure_rate=0.0,
    )


def _empty_evaluation_record() -> QLearningEvaluationRecord:
    return QLearningEvaluationRecord(
        training_episodes=0,
        training_steps=0,
        eval_mean_return=0.0,
        eval_std_return=0.0,
        eval_mean_episode_length=0.0,
        eval_success_rate=0.0,
        eval_mean_terminal_base_reward=0.0,
        eval_mean_terminal_task_progress=0.0,
        eval_task_failure_rate=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding",
        choices=["one_hot", "numerical", "semantic_progress"],
        default="semantic_progress",
    )
    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    parser.add_argument("--placement-mode", choices=["full_random", "regional"], default="regional")
    parser.add_argument("--n-value", type=int, default=5)
    parser.add_argument("--fixed-n", type=int, default=1)
    parser.add_argument("--sample-n", action="store_true")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--eval-freq-episodes", type=int, default=1_000)
    parser.add_argument("--train-log-freq-episodes", type=int, default=1_000)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-progress-bonus", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_randomized_letter_env_q_learning(
        RandomizedLetterEnvQLearningConfig(
            encoding=args.encoding,
            n_value=args.n_value,
            fixed_n=None if args.sample_n else args.fixed_n,
            placement_mode=args.placement_mode,
            episodes=args.episodes,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            eval_freq_episodes=args.eval_freq_episodes,
            train_log_freq_episodes=args.train_log_freq_episodes,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed_base=args.eval_seed_base,
            max_episode_steps=args.max_episode_steps,
            monitor_progress_bonus=args.monitor_progress_bonus,
            output_dir=args.output_dir,
        ),
        monitor_config_template=args.monitor_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
