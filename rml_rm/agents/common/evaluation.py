"""Evaluation callbacks shared by SB3 experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class EvaluationRecord:
    """Aggregate metrics for one policy evaluation."""

    training_steps: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_success_rate: float
    eval_mean_terminal_base_reward: float
    eval_mean_terminal_task_progress: float
    eval_task_failure_rate: float


class PeriodicEvaluationCallback(BaseCallback):
    """Evaluate a policy at fixed timestep intervals and save the best checkpoint."""

    def __init__(
        self,
        *,
        evaluation_env: gym.Env,
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed_base: int = 0,
    ) -> None:
        super().__init__(verbose=0)
        self.evaluation_env = evaluation_env
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.records: list[EvaluationRecord] = []
        self.best_record: EvaluationRecord | None = None
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

    def evaluate_current_policy(self) -> EvaluationRecord:
        returns: list[float] = []
        lengths: list[int] = []
        final_base_rewards: list[float] = []
        final_task_indices: list[float] = []
        task_failures: list[float] = []
        successes: list[float] = []

        for episode_index in range(self.n_eval_episodes):
            observation, _ = self.evaluation_env.reset(seed=self.eval_seed_base + episode_index)
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            final_base_reward = 0.0
            final_task_index = 0.0
            task_failed = False
            success = False

            while not terminated and not truncated:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.evaluation_env.step(
                    _scalar_action(action)
                )
                episode_return += float(reward)
                episode_length += 1
                final_base_reward = float(info.get("base_reward", reward))
                final_task_index = float(info.get("task_index", final_task_index))
                task_failed = bool(info.get("task_failed", task_failed))
                success = bool(info.get("success", success))

            returns.append(episode_return)
            lengths.append(episode_length)
            final_base_rewards.append(final_base_reward)
            final_task_indices.append(final_task_index)
            task_failures.append(1.0 if task_failed else 0.0)
            successes.append(1.0 if success else 0.0)

        return EvaluationRecord(
            training_steps=int(self.num_timesteps),
            eval_mean_return=float(np.mean(returns)),
            eval_std_return=float(np.std(returns)),
            eval_mean_episode_length=float(np.mean(lengths)),
            eval_success_rate=float(np.mean(successes)),
            eval_mean_terminal_base_reward=float(np.mean(final_base_rewards)),
            eval_mean_terminal_task_progress=float(np.mean(final_task_indices)),
            eval_task_failure_rate=float(np.mean(task_failures)),
        )

    @staticmethod
    def _empty_record() -> EvaluationRecord:
        return EvaluationRecord(
            training_steps=0,
            eval_mean_return=0.0,
            eval_std_return=0.0,
            eval_mean_episode_length=0.0,
            eval_success_rate=0.0,
            eval_mean_terminal_base_reward=0.0,
            eval_mean_terminal_task_progress=0.0,
            eval_task_failure_rate=0.0,
        )


def _scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])
