"""Train DDQN on the RML-based multi-task LetterEnv."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.multitask_letter_env.env import MultiTaskLetterEnv, MultiTaskLetterEnvConfig
from envs.multitask_letter_env.rml_generation import CONFIGS_ROOT, SPECS_ROOT
from envs.multitask_letter_env.tasks import get_task_suite
from rml_rm.agents.common import MLPPolicyConfig, build_monitor_policy_kwargs
from rml_rm.agents.ddqn import DoubleDQN
from rml_rm.experiments.runtime import (
    configure_global_seed,
    json_ready,
    managed_monitor_group,
    read_monitor_csv,
    rename_monitor_csv_columns,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MULTITASK_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"
DEFAULT_OUTPUT_ROOT = (
    MULTITASK_ROOT / "results_and_evaluation" / "ddqn"
)


@dataclass(frozen=True)
class MultiTaskDDQNTrainingConfig:
    """Training configuration for one multi-task DDQN run."""

    encoding: str = "one_hot"
    task_suite: str = "small_v1"
    max_n: int = 5
    learned_gru_checkpoint: Path | None = None
    learned_graph_checkpoint: Path | None = None
    total_timesteps: int = 500_000
    seed: int | None = 0
    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.9
    tau: float = 1.0
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.4
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    eval_freq: int = 20_000
    n_eval_episodes: int = 25
    eval_seed_base: int = 0
    max_episode_steps: int = 200
    monitor_transition_bonus: float = 10.0
    include_monitor_transition_bonus: bool = True
    output_dir: Path = field(default_factory=Path)


@dataclass(frozen=True)
class MultiTaskEvaluationRecord:
    """Aggregate metrics for one multitask policy evaluation."""

    training_steps: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_task_failure_rate: float


class MultiTaskEvaluationCallback(BaseCallback):
    """Evaluate a multitask policy at fixed timestep intervals."""

    def __init__(
        self,
        *,
        evaluation_env: gym.Env,
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        task_suite: str,
        max_n: int,
        eval_seed_base: int = 0,
    ) -> None:
        super().__init__(verbose=0)
        self.evaluation_env = evaluation_env
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.tasks = get_task_suite(task_suite)
        self.max_n = int(max_n)
        self.eval_seed_base = int(eval_seed_base)
        self.records: list[MultiTaskEvaluationRecord] = []
        self.best_record: MultiTaskEvaluationRecord | None = None
        self.metrics_path = self.output_dir / "eval_metrics.csv"
        self.best_model_path = self.output_dir / "best_model"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self._empty_record()).keys()))
            writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        record = self.evaluate_current_policy()
        self.records.append(record)
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(record).keys()))
            writer.writerow(asdict(record))

        if self.best_record is None or (
            record.eval_success_rate,
            record.eval_mean_return,
        ) > (
            self.best_record.eval_success_rate,
            self.best_record.eval_mean_return,
        ):
            self.best_record = record
            self.model.save(str(self.best_model_path))
        return True

    def _on_training_end(self) -> None:
        self.evaluation_env.close()

    def evaluate_current_policy(self) -> MultiTaskEvaluationRecord:
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        failures: list[float] = []

        for episode_index, task_id, n_value in self._evaluation_cases():
            observation, _ = self.evaluation_env.reset(
                seed=self.eval_seed_base + episode_index,
                options={"task_id": task_id, "n": n_value},
            )
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            success = False
            failed = False

            while not terminated and not truncated:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.evaluation_env.step(
                    _scalar_action(action)
                )
                episode_return += float(reward)
                episode_length += 1
                success = bool(info.get("success", success))
                failed = bool(info.get("failed", info.get("task_failed", failed)))

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(1.0 if success else 0.0)
            failures.append(1.0 if failed else 0.0)

        return MultiTaskEvaluationRecord(
            training_steps=int(self.num_timesteps),
            eval_mean_return=float(np.mean(returns)),
            eval_std_return=float(np.std(returns)),
            eval_mean_episode_length=float(np.mean(lengths)),
            eval_success_rate=float(np.mean(successes)),
            eval_task_failure_rate=float(np.mean(failures)),
        )

    @staticmethod
    def _empty_record() -> MultiTaskEvaluationRecord:
        return MultiTaskEvaluationRecord(
            training_steps=0,
            eval_mean_return=0.0,
            eval_std_return=0.0,
            eval_mean_episode_length=0.0,
            eval_success_rate=0.0,
            eval_task_failure_rate=0.0,
        )

    def _evaluation_cases(self) -> list[tuple[int, int, int]]:
        cases: list[tuple[int, int, int]] = []
        task_ids = [task.task_id for task in self.tasks]
        full_grid = [(task_id, n) for n in range(1, self.max_n + 1) for task_id in task_ids]
        for index in range(self.n_eval_episodes):
            task_id, n_value = full_grid[index % len(full_grid)]
            cases.append((index, task_id, n_value))
        return cases


def train_multitask_ddqn(
    config: MultiTaskDDQNTrainingConfig,
    *,
    policy_config: MLPPolicyConfig | None = None,
) -> dict[str, Any]:
    """Train one multitask DDQN run and write its artifacts."""
    if not config.output_dir:
        raise ValueError("output_dir is required.")

    policy_config = policy_config or MLPPolicyConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_global_seed(config.seed)
    started = time.monotonic()

    tasks = get_task_suite(config.task_suite)
    monitor_specs = {task.key: (SPECS_ROOT / f"{task.key}.pl").resolve() for task in tasks}
    monitor_configs = {task.key: (CONFIGS_ROOT / f"{task.key}.yaml").resolve() for task in tasks}

    train_env = None
    eval_env = None
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
            try:
                train_env = Monitor(
                    _make_env(config, train_runtime.ports),
                    filename=str(output_dir / "train_monitor.csv"),
                    info_keywords=("success", "failed", "task_id", "n"),
                )
                eval_env = Monitor(
                    _make_env(config, eval_runtime.ports),
                    filename=str(output_dir / "eval_monitor.csv"),
                    info_keywords=("success", "failed", "task_id", "n"),
                )
                if config.seed is not None:
                    train_env.reset(seed=config.seed)
                    eval_env.reset(seed=config.seed + 10_000)

                callback = MultiTaskEvaluationCallback(
                    evaluation_env=eval_env,
                    output_dir=output_dir,
                    eval_freq=config.eval_freq,
                    n_eval_episodes=config.n_eval_episodes,
                    task_suite=config.task_suite,
                    max_n=config.max_n,
                    eval_seed_base=config.eval_seed_base,
                )
                model = DoubleDQN(
                    policy="MultiInputPolicy",
                    env=train_env,
                    learning_rate=config.learning_rate,
                    buffer_size=config.buffer_size,
                    learning_starts=config.learning_starts,
                    batch_size=config.batch_size,
                    gamma=config.gamma,
                    tau=config.tau,
                    train_freq=(config.train_freq, "step"),
                    gradient_steps=config.gradient_steps,
                    target_update_interval=config.target_update_interval,
                    exploration_fraction=config.exploration_fraction,
                    exploration_initial_eps=config.exploration_initial_eps,
                    exploration_final_eps=config.exploration_final_eps,
                    policy_kwargs=build_monitor_policy_kwargs(config.encoding, policy_config),
                    seed=config.seed,
                    tensorboard_log=None,
                    verbose=1,
                )

                _write_run_config(
                    output_dir / "config.json",
                    config=config,
                    policy_config=policy_config,
                    train_config_paths=train_runtime.config_paths,
                    eval_config_paths=eval_runtime.config_paths,
                    monitor_specs=monitor_specs,
                )
                model.learn(
                    total_timesteps=config.total_timesteps,
                    callback=callback,
                    log_interval=1,
                    progress_bar=False,
                )
                model.save(str(output_dir / "model_final"))

                train_monitor_df = read_monitor_csv(output_dir / "train_monitor.csv")
                rename_monitor_csv_columns(output_dir / "train_monitor.csv")
                rename_monitor_csv_columns(output_dir / "eval_monitor.csv")
                eval_records = [asdict(record) for record in callback.records]
                final_eval = eval_records[-1] if eval_records else None
                best_eval = (
                    asdict(callback.best_record) if callback.best_record is not None else None
                )
                runtime_seconds = time.monotonic() - started
                summary = {
                    "completed_at_utc": utc_now(),
                    "algorithm": "ddqn",
                    "encoding": config.encoding,
                    "task_suite": config.task_suite,
                    "max_n": config.max_n,
                    "total_timesteps": config.total_timesteps,
                    "seed": config.seed,
                    "runtime_seconds": runtime_seconds,
                    "timesteps_per_second": (
                        float(config.total_timesteps) / runtime_seconds
                        if runtime_seconds > 0
                        else None
                    ),
                    "train_episodes_completed": int(len(train_monitor_df)),
                    "train_mean_return": (
                        float(train_monitor_df["episode_return"].mean())
                        if not train_monitor_df.empty
                        else None
                    ),
                    "train_mean_length": (
                        float(train_monitor_df["episode_length"].mean())
                        if not train_monitor_df.empty
                        else None
                    ),
                    "evaluation_count": len(eval_records),
                    "best_evaluation": best_eval,
                    "final_evaluation": final_eval,
                    "artifacts": {
                        "config": str(output_dir / "config.json"),
                        "summary": str(output_dir / "summary.json"),
                        "final_model": str(output_dir / "model_final.zip"),
                        "best_model": str(output_dir / "best_model.zip"),
                        "train_monitor": str(output_dir / "train_monitor.csv"),
                        "eval_monitor": str(output_dir / "eval_monitor.csv"),
                        "eval_metrics": str(output_dir / "eval_metrics.csv"),
                    },
                }
                write_json(output_dir / "summary.json", summary)
                return summary
            finally:
                if train_env is not None:
                    train_env.close()
                if eval_env is not None:
                    eval_env.close()


def _make_env(config: MultiTaskDDQNTrainingConfig, ports_by_task_key: dict[str, int]) -> gym.Env:
    tasks = get_task_suite(config.task_suite)
    ports_by_task_id = {task.task_id: ports_by_task_key[task.key] for task in tasks}
    return MultiTaskLetterEnv(
        MultiTaskLetterEnvConfig(
            encoding=config.encoding,
            task_suite=config.task_suite,
            max_n=config.max_n,
            max_episode_steps=config.max_episode_steps,
            learned_gru_checkpoint=config.learned_gru_checkpoint,
            learned_graph_checkpoint=config.learned_graph_checkpoint,
            monitor_ports_by_task_id=ports_by_task_id,
            transition_bonus=config.monitor_transition_bonus,
            include_transition_bonus=config.include_monitor_transition_bonus,
        )
    )


def _write_run_config(
    path: Path,
    *,
    config: MultiTaskDDQNTrainingConfig,
    policy_config: MLPPolicyConfig,
    train_config_paths: dict[str, Path],
    eval_config_paths: dict[str, Path],
    monitor_specs: dict[str, Path],
) -> None:
    payload = {
        "experiment": "multitask_letter_env_ddqn",
        "started_at_utc": utc_now(),
        "training_config": json_ready(asdict(config)),
        "policy_config": asdict(policy_config),
        "monitors": {
            "train_config_paths": {key: str(value) for key, value in train_config_paths.items()},
            "eval_config_paths": {key: str(value) for key, value in eval_config_paths.items()},
            "spec_paths": {key: str(value) for key, value in monitor_specs.items()},
        },
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding",
        choices=[
            "one_hot",
            "numerical",
            "learned_gru",
            "learned_graph",
        ],
        default="one_hot",
    )
    parser.add_argument("--task-suite", default="small_v1")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--learned-gru-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-transition-bonus", type=float, default=10.0)
    parser.add_argument("--disable-monitor-transition-bonus", action="store_true")
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--position-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-hidden-dim", type=int, default=64)
    parser.add_argument("--network-architecture", type=int, nargs="+", default=[128, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiTaskDDQNTrainingConfig(
        encoding=args.encoding,
        task_suite=args.task_suite,
        max_n=args.max_n,
        learned_gru_checkpoint=args.learned_gru_checkpoint,
        learned_graph_checkpoint=args.learned_graph_checkpoint,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_transition_bonus=args.monitor_transition_bonus,
        include_monitor_transition_bonus=not args.disable_monitor_transition_bonus,
        output_dir=args.output_dir,
    )
    policy_config = MLPPolicyConfig(
        features_dim=args.features_dim,
        position_hidden_dim=args.position_hidden_dim,
        monitor_hidden_dim=args.monitor_hidden_dim,
        network_architecture=tuple(args.network_architecture),
    )
    summary = train_multitask_ddqn(config, policy_config=policy_config)
    print(json.dumps(summary, indent=2))


def _scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])


if __name__ == "__main__":
    main()
