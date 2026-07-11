"""Evaluate saved multi-task LetterEnv policies on held-out n values."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from envs.multitask_letter_env.env import MultiTaskLetterEnv, MultiTaskLetterEnvConfig
from envs.multitask_letter_env.experiments.train_tabular import _tabular_state_key
from envs.multitask_letter_env.rml_generation import CONFIGS_ROOT, SPECS_ROOT
from envs.multitask_letter_env.tasks import get_task_suite
from rml_rm.agents.ddqn import DoubleDQN
from rml_rm.agents.tabular import QLearningAgent
from rml_rm.experiments.runtime import json_ready, managed_monitor_group, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
MULTITASK_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Metrics for one zero-shot evaluation episode."""

    algorithm: str
    encoding: str
    train_seed: int
    eval_n: int
    episode_index: int
    eval_seed: int
    task_id: int
    episode_return: float
    episode_length: int
    task_failed: bool
    success: bool
    timed_out: bool


@dataclass(frozen=True)
class AggregateEvaluationRecord:
    """Aggregate metrics for one zero-shot n value."""

    algorithm: str
    encoding: str
    train_seed: int
    eval_n: int
    n_eval_episodes: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_task_failure_rate: float
    eval_timeout_rate: float


def evaluate_zero_shot(
    *,
    algorithm: str,
    encoding: str,
    train_seed: int,
    eval_n: int,
    artifact_path: Path,
    output_dir: Path,
    n_eval_episodes: int,
    eval_seed_base: int,
    max_episode_steps: int,
    monitor_transition_bonus: float,
    include_monitor_transition_bonus: bool,
    task_suite: str = "small_v1",
) -> dict[str, Any]:
    """Evaluate a saved DDQN model or tabular Q-table on one fixed n value."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(train_seed)
    np.random.seed(train_seed)
    started = time.monotonic()

    tasks = get_task_suite(task_suite)
    monitor_specs = {task.key: (SPECS_ROOT / f"{task.key}.pl").resolve() for task in tasks}
    monitor_configs = {task.key: (CONFIGS_ROOT / f"{task.key}.yaml").resolve() for task in tasks}
    with managed_monitor_group(
        output_dir=output_dir,
        monitor_specs=monitor_specs,
        monitor_config_templates=monitor_configs,
        config_dir_name="monitor_eval_configs",
        log_dir_name="eval_rml_monitor_logs",
        max_episode_steps=max_episode_steps,
    ) as runtime:
        env = _make_env(
            encoding=encoding,
            task_suite=task_suite,
            eval_n=eval_n,
            max_episode_steps=max_episode_steps,
            ports_by_task_key=runtime.ports,
            monitor_transition_bonus=monitor_transition_bonus,
            include_monitor_transition_bonus=include_monitor_transition_bonus,
        )
        try:
            policy = _load_policy(algorithm, artifact_path)
            records = _evaluate_policy(
                env=env,
                policy=policy,
                algorithm=algorithm,
                encoding=encoding,
                train_seed=train_seed,
                eval_n=eval_n,
                n_eval_episodes=n_eval_episodes,
                eval_seed_base=eval_seed_base,
            )
        finally:
            env.close()

    aggregate = _aggregate_records(
        algorithm=algorithm,
        encoding=encoding,
        train_seed=train_seed,
        eval_n=eval_n,
        records=records,
    )
    _write_records(output_dir / "episode_metrics.csv", records)
    _write_aggregate(output_dir / "eval_metrics.csv", aggregate)
    summary = {
        "experiment": "multitask_letter_env_zero_shot",
        "created_at_utc": utc_now(),
        "runtime_seconds": time.monotonic() - started,
        "algorithm": algorithm,
        "encoding": encoding,
        "task_suite": task_suite,
        "train_seed": train_seed,
        "eval_n": eval_n,
        "artifact_path": str(artifact_path),
        "n_eval_episodes": n_eval_episodes,
        "eval_seed_base": eval_seed_base,
        "max_episode_steps": max_episode_steps,
        "monitor_transition_bonus": monitor_transition_bonus,
        "include_monitor_transition_bonus": include_monitor_transition_bonus,
        "aggregate": asdict(aggregate),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "eval_metrics": str(output_dir / "eval_metrics.csv"),
            "episode_metrics": str(output_dir / "episode_metrics.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _make_env(
    *,
    encoding: str,
    task_suite: str,
    eval_n: int,
    max_episode_steps: int,
    ports_by_task_key: dict[str, int],
    monitor_transition_bonus: float,
    include_monitor_transition_bonus: bool,
) -> MultiTaskLetterEnv:
    tasks = get_task_suite(task_suite)
    ports_by_task_id = {task.task_id: ports_by_task_key[task.key] for task in tasks}
    return MultiTaskLetterEnv(
        MultiTaskLetterEnvConfig(
            encoding=encoding,
            task_suite=task_suite,
            max_n=eval_n,
            max_episode_steps=max_episode_steps,
            monitor_ports_by_task_id=ports_by_task_id,
            transition_bonus=monitor_transition_bonus,
            include_transition_bonus=include_monitor_transition_bonus,
        )
    )


def _load_policy(algorithm: str, artifact_path: Path):
    if algorithm == "ddqn":
        return DoubleDQN.load(str(artifact_path))
    if algorithm == "tabular":
        return QLearningAgent.load_q_table(artifact_path)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _evaluate_policy(
    *,
    env: MultiTaskLetterEnv,
    policy,
    algorithm: str,
    encoding: str,
    train_seed: int,
    eval_n: int,
    n_eval_episodes: int,
    eval_seed_base: int,
) -> list[EpisodeEvaluationRecord]:
    records: list[EpisodeEvaluationRecord] = []
    for episode_index, task_id, n_value in _evaluation_cases(env, eval_n, n_eval_episodes):
        eval_seed = eval_seed_base + episode_index
        observation, info = env.reset(seed=eval_seed, options={"task_id": task_id, "n": n_value})
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        success = bool(info.get("success", False))
        failed = bool(info.get("failed", False))
        state = _tabular_state_key(observation, max_n=env.max_n)

        while not terminated and not truncated:
            action = _predict_action(policy, algorithm, observation, state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = _tabular_state_key(observation, max_n=env.max_n)
            episode_return += float(reward)
            episode_length += 1
            success = bool(info.get("success", success))
            failed = bool(info.get("failed", failed))

        records.append(
            EpisodeEvaluationRecord(
                algorithm=algorithm,
                encoding=encoding,
                train_seed=train_seed,
                eval_n=eval_n,
                episode_index=episode_index,
                eval_seed=eval_seed,
                task_id=task_id,
                episode_return=episode_return,
                episode_length=episode_length,
                task_failed=failed,
                success=success,
                timed_out=bool(truncated and not success and not failed),
            )
        )
    return records


def _predict_action(policy, algorithm: str, observation: dict[str, np.ndarray], state: Hashable) -> int:
    if algorithm == "ddqn":
        action, _ = policy.predict(observation, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])
    if algorithm == "tabular":
        action_values = policy.q_table.get(state)
        if action_values is None:
            return int(min(policy.actions))
        max_value = max(action_values.values())
        best_actions = [action for action, value in action_values.items() if value == max_value]
        return int(sorted(best_actions)[0])
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _evaluation_cases(
    env: MultiTaskLetterEnv,
    eval_n: int,
    n_eval_episodes: int,
) -> list[tuple[int, int, int]]:
    task_ids = [task.task_id for task in env.tasks]
    full_grid = [(task_id, eval_n) for task_id in task_ids]
    cases: list[tuple[int, int, int]] = []
    for index in range(n_eval_episodes):
        task_id, n_value = full_grid[index % len(full_grid)]
        cases.append((index, task_id, n_value))
    return cases


def _aggregate_records(
    *,
    algorithm: str,
    encoding: str,
    train_seed: int,
    eval_n: int,
    records: list[EpisodeEvaluationRecord],
) -> AggregateEvaluationRecord:
    returns = [record.episode_return for record in records]
    lengths = [record.episode_length for record in records]
    successes = [1.0 if record.success else 0.0 for record in records]
    task_failures = [1.0 if record.task_failed else 0.0 for record in records]
    timeouts = [1.0 if record.timed_out else 0.0 for record in records]
    return AggregateEvaluationRecord(
        algorithm=algorithm,
        encoding=encoding,
        train_seed=train_seed,
        eval_n=eval_n,
        n_eval_episodes=len(records),
        eval_mean_return=float(np.mean(returns)),
        eval_std_return=float(np.std(returns)),
        eval_mean_episode_length=float(np.mean(lengths)),
        eval_success_rate=float(np.mean(successes)),
        eval_task_failure_rate=float(np.mean(task_failures)),
        eval_timeout_rate=float(np.mean(timeouts)),
    )


def _write_records(path: Path, records: list[EpisodeEvaluationRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_aggregate(path: Path, record: AggregateEvaluationRecord) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(record).keys()))
        writer.writeheader()
        writer.writerow(asdict(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ddqn", "tabular"], required=True)
    parser.add_argument(
        "--encoding",
        choices=["one_hot", "numerical", "learned_gru", "learned_graph"],
        required=True,
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-n", type=int, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--monitor-transition-bonus", type=float, default=10.0)
    parser.add_argument("--disable-monitor-transition-bonus", action="store_true")
    parser.add_argument("--task-suite", default="small_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.algorithm == "tabular" and args.encoding not in {"one_hot", "numerical"}:
        raise ValueError("Tabular zero-shot evaluation supports one_hot and numerical only.")
    summary = evaluate_zero_shot(
        algorithm=args.algorithm,
        encoding=args.encoding,
        train_seed=args.train_seed,
        eval_n=args.eval_n,
        artifact_path=args.artifact_path,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_transition_bonus=args.monitor_transition_bonus,
        include_monitor_transition_bonus=not args.disable_monitor_transition_bonus,
        task_suite=args.task_suite,
    )
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
