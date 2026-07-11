"""Train tabular Q-learning on the RML-based multi-task LetterEnv."""

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
from envs.multitask_letter_env.env import MultiTaskLetterEnv, MultiTaskLetterEnvConfig
from envs.multitask_letter_env.rml_generation import CONFIGS_ROOT, SPECS_ROOT
from envs.multitask_letter_env.tasks import get_task_suite
from rml_rm.agents.tabular import QLearningAgent, QLearningConfig
from rml_rm.experiments.runtime import (
    json_ready,
    managed_monitor_group,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MULTITASK_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"
DEFAULT_OUTPUT_ROOT = MULTITASK_ROOT / "results_and_evaluation" / "tabular"


@dataclass(frozen=True)
class TabularEvaluationRecord:
    """Aggregate metrics from one deterministic tabular policy evaluation."""

    training_episodes: int
    training_steps: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_task_failure_rate: float


@dataclass(frozen=True)
class TabularTrainingRecord:
    """Training diagnostics recorded at fixed episode intervals."""

    episode: int
    training_steps: int
    epsilon: float
    q_state_count: int
    recent_mean_return: float
    recent_success_rate: float
    recent_task_failure_rate: float


@dataclass(frozen=True)
class MultiTaskTabularTrainingConfig:
    """Training configuration for one multi-task tabular Q-learning run."""

    encoding: str = "one_hot"
    task_suite: str = "small_v1"
    max_n: int = 5
    episodes: int = 200_000
    seed: int = 0
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.4
    epsilon_decay: float = 0.99995
    min_epsilon: float = 0.05
    eval_freq_episodes: int = 5_000
    train_log_freq_episodes: int = 1_000
    n_eval_episodes: int = 25
    eval_seed_base: int = 0
    max_episode_steps: int = 200
    monitor_transition_bonus: float = 10.0
    include_monitor_transition_bonus: bool = True
    state_discovery_bonus: float = 0.0
    output_dir: Path = field(default_factory=Path)


def train_multitask_tabular(config: MultiTaskTabularTrainingConfig) -> dict[str, Any]:
    """Train tabular Q-learning and write metrics plus a compact summary."""
    if config.encoding not in {"one_hot", "numerical"}:
        raise ValueError("Tabular multitask runs currently support one_hot and numerical.")
    if not config.output_dir:
        raise ValueError("output_dir is required.")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    started = time.monotonic()

    tasks = get_task_suite(config.task_suite)
    monitor_specs = {task.key: (SPECS_ROOT / f"{task.key}.pl").resolve() for task in tasks}
    monitor_configs = {task.key: (CONFIGS_ROOT / f"{task.key}.yaml").resolve() for task in tasks}
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
    training_records: list[TabularTrainingRecord] = []
    evaluation_records: list[TabularEvaluationRecord] = []
    recent_returns: list[float] = []
    recent_successes: list[float] = []
    recent_failures: list[float] = []
    training_steps = 0

    with managed_monitor_group(
        output_dir=output_dir,
        monitor_specs=monitor_specs,
        monitor_config_templates=monitor_configs,
        config_dir_name="monitor_train_configs",
        log_dir_name="train_rml_monitor_logs",
        max_episode_steps=config.max_episode_steps,
    ) as train_runtime:
        with managed_monitor_group(
            output_dir=output_dir,
            monitor_specs=monitor_specs,
            monitor_config_templates=monitor_configs,
            config_dir_name="monitor_eval_configs",
            log_dir_name="eval_rml_monitor_logs",
            max_episode_steps=config.max_episode_steps,
        ) as eval_runtime:
            train_env = _make_env(config, train_runtime.ports)
            eval_env = _make_env(config, eval_runtime.ports)
            try:
                _write_run_config(
                    output_dir / "config.json",
                    config=config,
                    train_config_paths=train_runtime.config_paths,
                    eval_config_paths=eval_runtime.config_paths,
                    monitor_specs=monitor_specs,
                )
                _write_eval_header(output_dir / "eval_metrics.csv")
                _write_train_header(output_dir / "train_metrics.csv")

                for episode in range(1, config.episodes + 1):
                    episode_return, episode_length, success, failed = _train_episode(
                        env=train_env,
                        agent=agent,
                        seed=config.seed if episode == 1 else None,
                        state_discovery_bonus=config.state_discovery_bonus,
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

                    if (
                        config.train_log_freq_episodes > 0
                        and episode % config.train_log_freq_episodes == 0
                    ):
                        record = TabularTrainingRecord(
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

                    if (
                        config.eval_freq_episodes > 0
                        and episode % config.eval_freq_episodes == 0
                    ):
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


def _make_env(
    config: MultiTaskTabularTrainingConfig,
    ports_by_task_key: dict[str, int],
) -> MultiTaskLetterEnv:
    tasks = get_task_suite(config.task_suite)
    ports_by_task_id = {task.task_id: ports_by_task_key[task.key] for task in tasks}
    return MultiTaskLetterEnv(
        MultiTaskLetterEnvConfig(
            encoding=config.encoding,
            task_suite=config.task_suite,
            max_n=config.max_n,
            max_episode_steps=config.max_episode_steps,
            monitor_ports_by_task_id=ports_by_task_id,
            transition_bonus=config.monitor_transition_bonus,
            include_transition_bonus=config.include_monitor_transition_bonus,
        )
    )


def _train_episode(
    *,
    env: MultiTaskLetterEnv,
    agent: QLearningAgent,
    seed: int | None,
    state_discovery_bonus: float,
) -> tuple[float, int, bool, bool]:
    observation, info = env.reset(seed=seed)
    state = _tabular_state_key(observation, max_n=env.max_n)
    agent.ensure_state(state)
    total_reward = 0.0
    episode_length = 0
    success = bool(info.get("success", False))
    failed = bool(info.get("failed", False))
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.choose_action(state)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = _tabular_state_key(next_observation, max_n=env.max_n)
        new_state = agent.ensure_state(next_state)
        shaped_reward = float(reward)
        if new_state:
            shaped_reward += float(state_discovery_bonus)
        agent.update(state, action, shaped_reward, next_state)
        state = next_state
        total_reward += float(reward)
        episode_length += 1
        success = bool(info.get("success", success))
        failed = bool(info.get("failed", failed))

    return total_reward, episode_length, success, failed


def _evaluate_policy(
    *,
    env: MultiTaskLetterEnv,
    agent: QLearningAgent,
    episode: int,
    training_steps: int,
    config: MultiTaskTabularTrainingConfig,
) -> TabularEvaluationRecord:
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    failures: list[float] = []

    for episode_index, task_id, n_value in _evaluation_cases(config):
        observation, info = env.reset(
            seed=config.eval_seed_base + episode_index,
            options={"task_id": task_id, "n": n_value},
        )
        state = _tabular_state_key(observation, max_n=env.max_n)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        success = bool(info.get("success", False))
        failed = bool(info.get("failed", False))

        while not terminated and not truncated:
            action = _greedy_action(agent, state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = _tabular_state_key(observation, max_n=env.max_n)
            episode_return += float(reward)
            episode_length += 1
            success = bool(info.get("success", success))
            failed = bool(info.get("failed", failed))

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(1.0 if success else 0.0)
        failures.append(1.0 if failed else 0.0)

    return TabularEvaluationRecord(
        training_episodes=int(episode),
        training_steps=int(training_steps),
        eval_mean_return=float(np.mean(returns)),
        eval_std_return=float(np.std(returns)),
        eval_mean_episode_length=float(np.mean(lengths)),
        eval_success_rate=float(np.mean(successes)),
        eval_task_failure_rate=float(np.mean(failures)),
    )


def _evaluation_cases(
    config: MultiTaskTabularTrainingConfig,
) -> list[tuple[int, int, int]]:
    tasks = get_task_suite(config.task_suite)
    full_grid = [(task.task_id, n_value) for n_value in range(1, config.max_n + 1) for task in tasks]
    cases: list[tuple[int, int, int]] = []
    for index in range(config.n_eval_episodes):
        task_id, n_value = full_grid[index % len(full_grid)]
        cases.append((index, task_id, n_value))
    return cases


def _tabular_state_key(observation: dict[str, np.ndarray], *, max_n: int) -> Hashable:
    position = np.asarray(observation["position"], dtype=np.float32).reshape(-1)
    monitor = np.asarray(observation["monitor"], dtype=np.float32).reshape(-1)
    proposition_features = position[2:7]
    task_features = position[8:]
    row = int(round(float(position[0])))
    col = int(round(float(position[1])))
    proposition_id = int(np.argmax(proposition_features))
    n_value = int(round(float(position[7]) * max_n))
    task_id = int(np.argmax(task_features))
    monitor_tuple = tuple(round(float(value), 6) for value in monitor)
    return row, col, proposition_id, n_value, task_id, monitor_tuple


def _greedy_action(agent: QLearningAgent, state: Hashable) -> int:
    agent.ensure_state(state)
    action_values = agent.q_table[state]
    max_value = max(action_values.values())
    best_actions = [action for action, value in action_values.items() if value == max_value]
    return int(sorted(best_actions)[0])


def _write_run_config(
    path: Path,
    *,
    config: MultiTaskTabularTrainingConfig,
    train_config_paths: dict[str, Path],
    eval_config_paths: dict[str, Path],
    monitor_specs: dict[str, Path],
) -> None:
    write_json(
        path,
        {
            "experiment": "multitask_letter_env_tabular_q_learning",
            "started_at_utc": utc_now(),
            "training_config": json_ready(asdict(config)),
            "state_key": "(row, col, proposition_id, n, task_id, monitor_encoding_tuple)",
            "monitors": {
                "train_config_paths": {
                    key: str(value) for key, value in train_config_paths.items()
                },
                "eval_config_paths": {
                    key: str(value) for key, value in eval_config_paths.items()
                },
                "spec_paths": {key: str(value) for key, value in monitor_specs.items()},
            },
        },
    )


def _build_summary(
    *,
    config: MultiTaskTabularTrainingConfig,
    agent: QLearningAgent,
    training_records: list[TabularTrainingRecord],
    evaluation_records: list[TabularEvaluationRecord],
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
        "algorithm": "tabular_q_learning",
        "encoding": config.encoding,
        "task_suite": config.task_suite,
        "max_n": config.max_n,
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


def _empty_training_record() -> TabularTrainingRecord:
    return TabularTrainingRecord(
        episode=0,
        training_steps=0,
        epsilon=0.0,
        q_state_count=0,
        recent_mean_return=0.0,
        recent_success_rate=0.0,
        recent_task_failure_rate=0.0,
    )


def _empty_evaluation_record() -> TabularEvaluationRecord:
    return TabularEvaluationRecord(
        training_episodes=0,
        training_steps=0,
        eval_mean_return=0.0,
        eval_std_return=0.0,
        eval_mean_episode_length=0.0,
        eval_success_rate=0.0,
        eval_task_failure_rate=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding",
        choices=["one_hot", "numerical"],
        default="one_hot",
    )
    parser.add_argument("--task-suite", default="small_v1")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--eval-freq-episodes", type=int, default=5_000)
    parser.add_argument("--train-log-freq-episodes", type=int, default=1_000)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-transition-bonus", type=float, default=10.0)
    parser.add_argument("--disable-monitor-transition-bonus", action="store_true")
    parser.add_argument("--state-discovery-bonus", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_multitask_tabular(
        MultiTaskTabularTrainingConfig(
            encoding=args.encoding,
            task_suite=args.task_suite,
            max_n=args.max_n,
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
            monitor_transition_bonus=args.monitor_transition_bonus,
            include_monitor_transition_bonus=not args.disable_monitor_transition_bonus,
            state_discovery_bonus=args.state_discovery_bonus,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
