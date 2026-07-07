"""Evaluate a saved neural LetterEnv policy on fixed zero-shot n values."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO

from envs.letter_env import LetterEnvConfig, build_letter_env
from rml_rm.agents.ddqn import DoubleDQN
from rml_rm.monitors import RMLMonitorProcess, find_free_port


REPO_ROOT = Path(__file__).resolve().parents[3]
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_MONITOR_CONFIG = LETTER_ENV_ROOT / "configs" / "letter_env.yaml"
DEFAULT_MONITOR_SPEC = LETTER_ENV_ROOT / "specs" / "letter_env_monitor.pl"


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


def evaluate_zero_shot(
    *,
    algorithm: str,
    encoding: str,
    train_seed: int,
    eval_n: int,
    model_path: Path,
    learned_gru_checkpoint: Path | None,
    learned_graph_checkpoint: Path | None,
    output_dir: Path,
    n_eval_episodes: int,
    eval_seed_base: int,
    max_episode_steps: int,
    monitor_progress_bonus: float,
    monitor_regression_penalty: float,
    step_penalty: float,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Load a saved policy and evaluate it on one fixed n value."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_global_seed(train_seed)

    port = find_free_port()
    runtime_config_path = _write_runtime_monitor_config(
        output_dir / "monitor_eval_config.yaml",
        template_path=monitor_config_template,
        port=port,
    )
    monitor = RMLMonitorProcess(
        spec_path=monitor_spec_path,
        port=port,
        log_path=output_dir / "eval_rml_monitor.log",
    )
    started = time.monotonic()
    env = None

    try:
        monitor.start()
        env = build_letter_env(
            LetterEnvConfig(
                encoding=encoding,
                learned_gru_checkpoint=learned_gru_checkpoint,
                learned_graph_checkpoint=learned_graph_checkpoint,
                n_value=eval_n,
                fixed_n=eval_n,
                max_episode_steps=max_episode_steps,
                monitor_progress_bonus=monitor_progress_bonus,
                monitor_regression_penalty=monitor_regression_penalty,
                neutralize_legacy_transition_bonus=True,
                step_penalty=step_penalty,
                state_discovery_bonus=0.0,
            ),
            monitor_config_path=runtime_config_path,
        )
        model = _load_model(algorithm, model_path)
        records = _evaluate_policy(
            model=model,
            env=env,
            algorithm=algorithm,
            encoding=encoding,
            train_seed=train_seed,
            eval_n=eval_n,
            n_eval_episodes=n_eval_episodes,
            eval_seed_base=eval_seed_base,
        )
    finally:
        if env is not None:
            env.close()
        monitor.stop()

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
        "experiment": "letter_env_neural_zero_shot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "algorithm": algorithm,
        "encoding": encoding,
        "train_seed": train_seed,
        "eval_n": eval_n,
        "model_path": str(model_path),
        "learned_gru_checkpoint": str(learned_gru_checkpoint) if learned_gru_checkpoint else None,
        "learned_graph_checkpoint": str(learned_graph_checkpoint) if learned_graph_checkpoint else None,
        "n_eval_episodes": n_eval_episodes,
        "eval_seed_base": eval_seed_base,
        "max_episode_steps": max_episode_steps,
        "monitor_progress_bonus": monitor_progress_bonus,
        "monitor_regression_penalty": monitor_regression_penalty,
        "step_penalty": step_penalty,
        "aggregate": asdict(aggregate),
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "eval_metrics": str(output_dir / "eval_metrics.csv"),
            "episode_metrics": str(output_dir / "episode_metrics.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _load_model(algorithm: str, model_path: Path):
    if algorithm == "dqn":
        return DQN.load(str(model_path))
    if algorithm == "ddqn":
        return DoubleDQN.load(str(model_path))
    if algorithm == "ppo":
        return PPO.load(str(model_path))
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _evaluate_policy(
    *,
    model,
    env,
    algorithm: str,
    encoding: str,
    train_seed: int,
    eval_n: int,
    n_eval_episodes: int,
    eval_seed_base: int,
) -> list[EpisodeEvaluationRecord]:
    records: list[EpisodeEvaluationRecord] = []
    for episode_index in range(n_eval_episodes):
        eval_seed = eval_seed_base + episode_index
        observation, _ = env.reset(seed=eval_seed)
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
                algorithm=algorithm,
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
            )
        )
    return records


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
    base_rewards = [record.terminal_base_reward for record in records]
    task_progress = [record.terminal_task_progress for record in records]
    task_failures = [1.0 if record.task_failed else 0.0 for record in records]
    successes = [1.0 if record.success else 0.0 for record in records]
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
        eval_mean_terminal_base_reward=float(np.mean(base_rewards)),
        eval_mean_terminal_task_progress=float(np.mean(task_progress)),
        eval_task_failure_rate=float(np.mean(task_failures)),
    )


def _write_runtime_monitor_config(path: Path, *, template_path: Path, port: int) -> Path:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["host"] = "127.0.0.1"
    config["port"] = int(port)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


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


def _configure_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["dqn", "ddqn", "ppo"], required=True)
    parser.add_argument(
        "--encoding",
        choices=[
            "one_hot",
            "numerical",
            "semantic_progress",
            "learned_gru",
            "learned_graph",
        ],
        required=True,
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-n", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--learned-gru-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-progress-bonus", type=float, default=10.0)
    parser.add_argument("--monitor-regression-penalty", type=float, default=0.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_zero_shot(
        algorithm=args.algorithm,
        encoding=args.encoding,
        train_seed=args.train_seed,
        eval_n=args.eval_n,
        model_path=args.model_path,
        learned_gru_checkpoint=args.learned_gru_checkpoint,
        learned_graph_checkpoint=args.learned_graph_checkpoint,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_progress_bonus=args.monitor_progress_bonus,
        monitor_regression_penalty=args.monitor_regression_penalty,
        step_penalty=args.step_penalty,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
