"""Train PPO on the RML-based LunarLander protocol."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.lunar_lander import LunarLanderProtocolConfig, build_lunar_lander_protocol_env
from envs.lunar_lander.builder import _lunar_monitor_progress
from rml_rm.agents.common import (
    MLPPolicyConfig,
    build_monitor_policy_kwargs,
)
from rml_rm.experiments.runtime import (
    configure_global_seed,
    json_ready,
    managed_monitor_pair,
    read_monitor_csv,
    rename_monitor_csv_columns,
    utc_now,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LUNAR_ENV_ROOT = REPO_ROOT / "envs" / "lunar_lander"
DEFAULT_MONITOR_CONFIG = LUNAR_ENV_ROOT / "configs" / "lunar_lander_protocol.yaml"
DEFAULT_MONITOR_SPEC = LUNAR_ENV_ROOT / "specs" / "lunar_lander_protocol.pl"


@dataclass(frozen=True)
class LunarLanderPPOTrainingConfig:
    """Training configuration for one LunarLander PPO run."""

    encoding: str = "semantic_progress"
    total_timesteps: int = 500_000
    seed: int | None = 0
    learning_rate: float = 3e-4
    linear_learning_rate_decay: bool = False
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    eval_freq: int = 20_000
    n_eval_episodes: int = 20
    eval_seed_base: int = 10_000
    max_episode_steps: int = 1000
    monitor_progress_bonus: float = 20.0
    hover_step_bonus: float = 2.0
    hover_complete_bonus: float = 30.0
    controlled_descent_bonus: float = 20.0
    success_bonus: float = 100.0
    failure_penalty: float = -25.0
    landing_target_bonus: float = 10.0
    landing_angle_bonus: float = 10.0
    post_descent_landing_bonus: float = 40.0
    post_descent_protocol_miss_penalty: float = -60.0
    initial_model: Path | None = None
    early_stop_protocol_rate: float | None = None
    early_stop_landing_rate: float | None = None
    early_stop_patience: int = 1
    output_dir: Path = field(default_factory=Path)


@dataclass(frozen=True)
class LunarLanderEvaluationRecord:
    """Aggregate metrics for one LunarLander policy evaluation."""

    training_steps: int
    eval_mean_return: float
    eval_std_return: float
    eval_mean_episode_length: float
    eval_successful_landing_rate: float
    eval_successful_protocol_rate: float
    eval_hover_complete_rate: float
    eval_controlled_descent_rate: float
    eval_mean_terminal_lunar_base_reward: float
    eval_mean_terminal_monitor_reward: float
    eval_task_failure_rate: float


class LunarLanderEvaluationCallback(BaseCallback):
    """Evaluate LunarLander policies with separate landing and protocol metrics."""

    def __init__(
        self,
        *,
        evaluation_env: gym.Env,
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed_base: int = 0,
        early_stop_protocol_rate: float | None = None,
        early_stop_landing_rate: float | None = None,
        early_stop_patience: int = 1,
    ) -> None:
        super().__init__(verbose=0)
        self.evaluation_env = evaluation_env
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.records: list[LunarLanderEvaluationRecord] = []
        self.best_record: LunarLanderEvaluationRecord | None = None
        self.metrics_path = self.output_dir / "eval_metrics.csv"
        self.best_model_path = self.output_dir / "best_model"
        self.early_stop_protocol_rate = early_stop_protocol_rate
        self.early_stop_landing_rate = early_stop_landing_rate
        self.early_stop_patience = max(int(early_stop_patience), 1)
        self.early_stop_hits = 0

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

        if self.best_record is None or _lunar_eval_key(record) > _lunar_eval_key(
            self.best_record
        ):
            self.best_record = record
            self.model.save(str(self.best_model_path))

        if self._early_stop_reached(record):
            self.early_stop_hits += 1
            if self.early_stop_hits >= self.early_stop_patience:
                return False
        else:
            self.early_stop_hits = 0

        return True

    def _on_training_end(self) -> None:
        self.evaluation_env.close()

    def _early_stop_reached(self, record: LunarLanderEvaluationRecord) -> bool:
        if self.early_stop_protocol_rate is None and self.early_stop_landing_rate is None:
            return False
        if (
            self.early_stop_protocol_rate is not None
            and record.eval_successful_protocol_rate < self.early_stop_protocol_rate
        ):
            return False
        if (
            self.early_stop_landing_rate is not None
            and record.eval_successful_landing_rate < self.early_stop_landing_rate
        ):
            return False
        return True

    def evaluate_current_policy(self) -> LunarLanderEvaluationRecord:
        returns: list[float] = []
        lengths: list[int] = []
        final_lunar_rewards: list[float] = []
        final_monitor_rewards: list[float] = []
        task_failures: list[float] = []
        successful_landings: list[float] = []
        successful_protocols: list[float] = []
        hover_completions: list[float] = []
        controlled_descents: list[float] = []

        for episode_index in range(self.n_eval_episodes):
            observation, _ = self.evaluation_env.reset(seed=self.eval_seed_base + episode_index)
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            final_lunar_reward = 0.0
            final_monitor_reward = 0.0
            task_failed = False
            successful_landing = False
            successful_protocol = False
            max_monitor_progress = 0.0

            while not terminated and not truncated:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.evaluation_env.step(
                    _scalar_action(action)
                )
                episode_return += float(reward)
                episode_length += 1
                final_lunar_reward = float(info.get("lunar_base_reward", final_lunar_reward))
                final_monitor_reward = float(
                    info.get("monitor_terminal_reward", final_monitor_reward)
                )
                task_failed = bool(info.get("task_failed", task_failed))
                successful_landing = bool(info.get("successful_landing", successful_landing))
                successful_protocol = bool(
                    info.get("successful_protocol", successful_protocol)
                )
                max_monitor_progress = max(
                    max_monitor_progress,
                    _lunar_monitor_progress(info.get("monitor_state_unencoded")),
                )

            returns.append(episode_return)
            lengths.append(episode_length)
            final_lunar_rewards.append(final_lunar_reward)
            final_monitor_rewards.append(final_monitor_reward)
            task_failures.append(1.0 if task_failed else 0.0)
            successful_landings.append(1.0 if successful_landing else 0.0)
            successful_protocols.append(1.0 if successful_protocol else 0.0)
            hover_completions.append(1.0 if max_monitor_progress >= 3.0 else 0.0)
            controlled_descents.append(1.0 if max_monitor_progress >= 4.0 else 0.0)

        return LunarLanderEvaluationRecord(
            training_steps=int(self.num_timesteps),
            eval_mean_return=float(np.mean(returns)),
            eval_std_return=float(np.std(returns)),
            eval_mean_episode_length=float(np.mean(lengths)),
            eval_successful_landing_rate=float(np.mean(successful_landings)),
            eval_successful_protocol_rate=float(np.mean(successful_protocols)),
            eval_hover_complete_rate=float(np.mean(hover_completions)),
            eval_controlled_descent_rate=float(np.mean(controlled_descents)),
            eval_mean_terminal_lunar_base_reward=float(np.mean(final_lunar_rewards)),
            eval_mean_terminal_monitor_reward=float(np.mean(final_monitor_rewards)),
            eval_task_failure_rate=float(np.mean(task_failures)),
        )

    @staticmethod
    def _empty_record() -> LunarLanderEvaluationRecord:
        return LunarLanderEvaluationRecord(
            training_steps=0,
            eval_mean_return=0.0,
            eval_std_return=0.0,
            eval_mean_episode_length=0.0,
            eval_successful_landing_rate=0.0,
            eval_successful_protocol_rate=0.0,
            eval_hover_complete_rate=0.0,
            eval_controlled_descent_rate=0.0,
            eval_mean_terminal_lunar_base_reward=0.0,
            eval_mean_terminal_monitor_reward=0.0,
            eval_task_failure_rate=0.0,
        )


def train_lunar_lander_ppo(
    config: LunarLanderPPOTrainingConfig,
    *,
    policy_config: MLPPolicyConfig | None = None,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Train one LunarLander PPO run and write its artifacts."""
    if config.encoding != "semantic_progress":
        raise ValueError("Only semantic_progress is currently supported.")
    if not config.output_dir:
        raise ValueError("output_dir is required.")

    policy_config = policy_config or MLPPolicyConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_global_seed(config.seed)
    started = time.monotonic()

    train_env = None
    eval_env = None
    with managed_monitor_pair(
        output_dir=output_dir,
        monitor_config_template=monitor_config_template,
        monitor_spec_path=monitor_spec_path,
        max_episode_steps=config.max_episode_steps,
    ) as monitor_runtime:
        try:
            train_env = Monitor(
                build_lunar_lander_protocol_env(
                    _env_config(config),
                    monitor_config_path=monitor_runtime.train_config_path,
                ),
                filename=str(output_dir / "train_monitor.csv"),
            )
            eval_env = Monitor(
                build_lunar_lander_protocol_env(
                    _env_config(config),
                    monitor_config_path=monitor_runtime.eval_config_path,
                ),
                filename=str(output_dir / "eval_monitor.csv"),
            )

            if config.seed is not None:
                train_env.reset(seed=config.seed)
                eval_env.reset(seed=config.seed + 10_000)

            callback = LunarLanderEvaluationCallback(
                evaluation_env=eval_env,
                output_dir=output_dir,
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                eval_seed_base=config.eval_seed_base,
                early_stop_protocol_rate=config.early_stop_protocol_rate,
                early_stop_landing_rate=config.early_stop_landing_rate,
                early_stop_patience=config.early_stop_patience,
            )
            if config.initial_model is not None:
                if not config.initial_model.exists():
                    raise FileNotFoundError(f"Initial model not found: {config.initial_model}")
                model = PPO.load(
                    str(config.initial_model),
                    env=train_env,
                    seed=config.seed,
                    print_system_info=False,
                )
                model.verbose = 1
                model.learning_rate = _ppo_learning_rate(config)
                model.lr_schedule = get_schedule_fn(model.learning_rate)
                model.target_kl = config.target_kl
            else:
                model = PPO(
                    policy="MultiInputPolicy",
                    env=train_env,
                    learning_rate=_ppo_learning_rate(config),
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    clip_range=config.clip_range,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    max_grad_norm=config.max_grad_norm,
                    target_kl=config.target_kl,
                    policy_kwargs=build_monitor_policy_kwargs(config.encoding, policy_config),
                    seed=config.seed,
                    verbose=1,
                )

            _write_run_config(
                output_dir / "config.json",
                config=config,
                policy_config=policy_config,
                train_config_path=monitor_runtime.train_config_path,
                eval_config_path=monitor_runtime.eval_config_path,
                monitor_spec_path=monitor_spec_path,
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
            best_eval = asdict(callback.best_record) if callback.best_record is not None else None
            runtime_seconds = time.monotonic() - started
            summary = {
                "completed_at_utc": utc_now(),
                "algorithm": "ppo",
                "encoding": config.encoding,
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


def _env_config(config: LunarLanderPPOTrainingConfig) -> LunarLanderProtocolConfig:
    return LunarLanderProtocolConfig(
        encoding=config.encoding,
        max_episode_steps=config.max_episode_steps,
        monitor_progress_bonus=config.monitor_progress_bonus,
        hover_step_bonus=config.hover_step_bonus,
        hover_complete_bonus=config.hover_complete_bonus,
        controlled_descent_bonus=config.controlled_descent_bonus,
        success_bonus=config.success_bonus,
        failure_penalty=config.failure_penalty,
        landing_target_bonus=config.landing_target_bonus,
        landing_angle_bonus=config.landing_angle_bonus,
        post_descent_landing_bonus=config.post_descent_landing_bonus,
        post_descent_protocol_miss_penalty=config.post_descent_protocol_miss_penalty,
    )


def _lunar_eval_key(record: LunarLanderEvaluationRecord) -> tuple[float, float, float]:
    return (
        record.eval_successful_protocol_rate,
        record.eval_successful_landing_rate,
        record.eval_mean_return,
    )


def _ppo_learning_rate(config: LunarLanderPPOTrainingConfig):
    if config.linear_learning_rate_decay:
        return lambda progress_remaining: progress_remaining * config.learning_rate
    return config.learning_rate


def _scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _write_run_config(
    path: Path,
    *,
    config: LunarLanderPPOTrainingConfig,
    policy_config: MLPPolicyConfig,
    train_config_path: Path,
    eval_config_path: Path,
    monitor_spec_path: Path,
) -> None:
    payload = {
        "experiment": "lunar_lander_protocol_ppo",
        "started_at_utc": utc_now(),
        "training_config": json_ready(asdict(config)),
        "policy_config": asdict(policy_config),
        "monitor": {
            "train_config_path": str(train_config_path),
            "eval_config_path": str(eval_config_path),
            "spec_path": str(monitor_spec_path),
        },
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", choices=["semantic_progress"], default="semantic_progress")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--linear-learning-rate-decay", action="store_true")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--monitor-progress-bonus", type=float, default=20.0)
    parser.add_argument("--hover-step-bonus", type=float, default=2.0)
    parser.add_argument("--hover-complete-bonus", type=float, default=30.0)
    parser.add_argument("--controlled-descent-bonus", type=float, default=20.0)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--failure-penalty", type=float, default=-25.0)
    parser.add_argument("--landing-target-bonus", type=float, default=10.0)
    parser.add_argument("--landing-angle-bonus", type=float, default=10.0)
    parser.add_argument("--post-descent-landing-bonus", type=float, default=40.0)
    parser.add_argument("--post-descent-protocol-miss-penalty", type=float, default=-60.0)
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="Optional PPO .zip checkpoint to fine-tune under the current RML spec.",
    )
    parser.add_argument("--early-stop-protocol-rate", type=float, default=None)
    parser.add_argument("--early-stop-landing-rate", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    parser.add_argument("--monitor-spec", type=Path, default=DEFAULT_MONITOR_SPEC)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LunarLanderPPOTrainingConfig:
    return LunarLanderPPOTrainingConfig(
        encoding=args.encoding,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        linear_learning_rate_decay=args.linear_learning_rate_decay,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_progress_bonus=args.monitor_progress_bonus,
        hover_step_bonus=args.hover_step_bonus,
        hover_complete_bonus=args.hover_complete_bonus,
        controlled_descent_bonus=args.controlled_descent_bonus,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        landing_target_bonus=args.landing_target_bonus,
        landing_angle_bonus=args.landing_angle_bonus,
        post_descent_landing_bonus=args.post_descent_landing_bonus,
        post_descent_protocol_miss_penalty=args.post_descent_protocol_miss_penalty,
        initial_model=args.initial_model,
        early_stop_protocol_rate=args.early_stop_protocol_rate,
        early_stop_landing_rate=args.early_stop_landing_rate,
        early_stop_patience=args.early_stop_patience,
        output_dir=args.output_dir,
    )


def main() -> None:
    args = parse_args()
    summary = train_lunar_lander_ppo(
        config_from_args(args),
        monitor_config_template=args.monitor_config,
        monitor_spec_path=args.monitor_spec,
    )
    final_eval = summary.get("final_evaluation") or {}
    best_eval = summary.get("best_evaluation") or {}
    print(
        "Completed LunarLander PPO run: "
        f"best_protocol={best_eval.get('eval_successful_protocol_rate')}, "
        f"final_protocol={final_eval.get('eval_successful_protocol_rate')}, "
        f"final_landing={final_eval.get('eval_successful_landing_rate')}, "
        f"output={summary['artifacts']['summary']}"
    )


if __name__ == "__main__":
    main()
