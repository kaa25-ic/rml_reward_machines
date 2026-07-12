"""Evaluate saved randomized LetterEnv DDQN policies on held-out n values."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs.randomized_letter_env import RandomizedLetterEnvConfig, build_randomized_letter_env
from envs.randomized_letter_env.experiments.train_ddqn import (
    DEFAULT_MONITOR_CONFIG,
    DEFAULT_MONITOR_SPEC,
)
from rml_rm.agents.ddqn import DoubleDQN
from rml_rm.experiments.runtime import (
    configure_global_seed,
    managed_monitor,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Metrics for one zero-shot evaluation episode."""

    algorithm: str
    encoding: str
    train_seed: int
    eval_n: int
    episode_index: int
    eval_seed: int
    episode_return: float
    episode_length: int
    terminal_base_reward: float
    terminal_task_progress: float
    task_failed: bool
    success: bool
    timed_out: bool


@dataclass(frozen=True)
class AggregateEvaluationRecord:
    """Aggregate metrics for one fixed zero-shot n value."""

    algorithm: str
    encoding: str
    train_seed: int
    eval_n: int
    n_eval_episodes: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_mean_terminal_base_reward: float
    eval_mean_terminal_task_progress: float
    eval_task_failure_rate: float
    eval_timeout_rate: float


def evaluate_zero_shot(
    *,
    encoding: str,
    train_seed: int,
    eval_n: int,
    model_path: Path,
    output_dir: Path,
    n_eval_episodes: int,
    eval_seed_base: int,
    max_episode_steps: int,
    monitor_progress_bonus: float,
    placement_mode: str,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Load a saved DDQN policy and evaluate it at one fixed held-out n."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_global_seed(train_seed)
    started = time.monotonic()

    with managed_monitor(
        output_dir=output_dir,
        monitor_config_template=monitor_config_template,
        monitor_spec_path=monitor_spec_path,
        log_name="eval_rml_monitor.log",
        config_name="monitor_eval_config.yaml",
    ) as runtime:
        env = build_randomized_letter_env(
            RandomizedLetterEnvConfig(
                encoding=encoding,
                n_value=eval_n,
                fixed_n=eval_n,
                max_episode_steps=max_episode_steps,
                monitor_progress_bonus=monitor_progress_bonus,
                placement_mode=placement_mode,
            ),
            monitor_config_path=runtime.config_path,
        )
        try:
            model = DoubleDQN.load(str(model_path))
            records = _evaluate_policy(
                model=model,
                env=env,
                encoding=encoding,
                train_seed=train_seed,
                eval_n=eval_n,
                n_eval_episodes=n_eval_episodes,
                eval_seed_base=eval_seed_base,
            )
        finally:
            env.close()

    aggregate = _aggregate_records(
        encoding=encoding,
        train_seed=train_seed,
        eval_n=eval_n,
        records=records,
    )
    _write_records(output_dir / "episode_metrics.csv", records)
    _write_aggregate(output_dir / "eval_metrics.csv", aggregate)
    summary = {
        "experiment": "randomized_letter_env_ddqn_zero_shot",
        "created_at_utc": utc_now(),
        "runtime_seconds": time.monotonic() - started,
        "algorithm": "ddqn",
        "encoding": encoding,
        "train_seed": train_seed,
        "eval_n": eval_n,
        "model_path": str(model_path),
        "n_eval_episodes": n_eval_episodes,
        "eval_seed_base": eval_seed_base,
        "max_episode_steps": max_episode_steps,
        "monitor_progress_bonus": monitor_progress_bonus,
        "placement_mode": placement_mode,
        "aggregate": asdict(aggregate),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "eval_metrics": str(output_dir / "eval_metrics.csv"),
            "episode_metrics": str(output_dir / "episode_metrics.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _evaluate_policy(
    *,
    model: DoubleDQN,
    env,
    encoding: str,
    train_seed: int,
    eval_n: int,
    n_eval_episodes: int,
    eval_seed_base: int,
) -> list[EpisodeEvaluationRecord]:
    records: list[EpisodeEvaluationRecord] = []
    for episode_index in range(n_eval_episodes):
        eval_seed = eval_seed_base + episode_index
        observation, _ = env.reset(seed=eval_seed, options={"n": eval_n})
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        terminal_base_reward = 0.0
        terminal_task_progress = 0.0
        task_failed = False
        success = False

        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(_scalar_action(action))
            episode_return += float(reward)
            episode_length += 1
            terminal_base_reward = float(info.get("base_reward", reward))
            terminal_task_progress = float(info.get("task_index", terminal_task_progress))
            task_failed = bool(info.get("task_failed", task_failed))
            success = bool(info.get("success", success))

        records.append(
            EpisodeEvaluationRecord(
                algorithm="ddqn",
                encoding=encoding,
                train_seed=train_seed,
                eval_n=eval_n,
                episode_index=episode_index,
                eval_seed=eval_seed,
                episode_return=episode_return,
                episode_length=episode_length,
                terminal_base_reward=terminal_base_reward,
                terminal_task_progress=terminal_task_progress,
                task_failed=task_failed,
                success=success,
                timed_out=bool(truncated and not success and not task_failed),
            )
        )
    return records


def _aggregate_records(
    *,
    encoding: str,
    train_seed: int,
    eval_n: int,
    records: list[EpisodeEvaluationRecord],
) -> AggregateEvaluationRecord:
    returns = [record.episode_return for record in records]
    lengths = [record.episode_length for record in records]
    terminal_rewards = [record.terminal_base_reward for record in records]
    terminal_progress = [record.terminal_task_progress for record in records]
    failures = [1.0 if record.task_failed else 0.0 for record in records]
    successes = [1.0 if record.success else 0.0 for record in records]
    timeouts = [1.0 if record.timed_out else 0.0 for record in records]
    return AggregateEvaluationRecord(
        algorithm="ddqn",
        encoding=encoding,
        train_seed=train_seed,
        eval_n=eval_n,
        n_eval_episodes=len(records),
        eval_mean_return=float(np.mean(returns)),
        eval_std_return=float(np.std(returns)),
        eval_mean_episode_length=float(np.mean(lengths)),
        eval_success_rate=float(np.mean(successes)),
        eval_mean_terminal_base_reward=float(np.mean(terminal_rewards)),
        eval_mean_terminal_task_progress=float(np.mean(terminal_progress)),
        eval_task_failure_rate=float(np.mean(failures)),
        eval_timeout_rate=float(np.mean(timeouts)),
    )


def _write_records(path: Path, records: list[EpisodeEvaluationRecord]) -> None:
    if not records:
        return
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


def _scalar_action(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=["one_hot", "numerical", "semantic_progress"], default="semantic_progress")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-n", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-base", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-progress-bonus", type=float, default=10.0)
    parser.add_argument("--placement-mode", choices=["full_random", "regional"], default="regional")
    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_zero_shot(
        encoding=args.encoding,
        train_seed=args.train_seed,
        eval_n=args.eval_n,
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_progress_bonus=args.monitor_progress_bonus,
        placement_mode=args.placement_mode,
        monitor_config_template=args.monitor_config,
    )
    write_json(args.output_dir / "summary.json", summary)
    aggregate = summary["aggregate"]
    print(
        "zero-shot "
        f"encoding={summary['encoding']} "
        f"placement={summary['placement_mode']} "
        f"train_seed={summary['train_seed']} "
        f"eval_n={summary['eval_n']} "
        f"success_rate={aggregate['eval_success_rate']:.3f} "
        f"failure_rate={aggregate['eval_task_failure_rate']:.3f} "
        f"timeout_rate={aggregate['eval_timeout_rate']:.3f} "
        f"mean_length={aggregate['eval_mean_episode_length']:.2f} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
