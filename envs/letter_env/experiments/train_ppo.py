"""Train PPO on LetterEnv."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.letter_env import LetterEnvConfig, build_letter_env
from rml_rm.agents.common import (
    MLPPolicyConfig,
    PeriodicEvaluationCallback,
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
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_MONITOR_CONFIG = LETTER_ENV_ROOT / "configs" / "letter_env.yaml"
DEFAULT_MONITOR_SPEC = LETTER_ENV_ROOT / "specs" / "letter_env_monitor.pl"


@dataclass(frozen=True)
class LetterEnvPPOTrainingConfig:
    """Training configuration for one LetterEnv PPO run."""

    encoding: str = "numerical"
    n_value: int = 5
    fixed_n: int | None = None
    total_timesteps: int = 500_000
    seed: int | None = 0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 16_384
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 20_000
    n_eval_episodes: int = 20
    eval_seed_base: int = 0
    max_episode_steps: int = 200
    monitor_progress_bonus: float = 10.0
    monitor_regression_penalty: float = 0.0
    neutralize_legacy_transition_bonus: bool = True
    legacy_transition_bonus: float = 10.0
    step_penalty: float = 0.05
    no_op_penalty: float = 0.0
    state_discovery_bonus: float = 0.0
    output_dir: Path = field(default_factory=Path)


def train_letter_env_ppo(
    config: LetterEnvPPOTrainingConfig,
    *,
    policy_config: MLPPolicyConfig | None = None,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Train one PPO LetterEnv run and write its artifacts."""
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
    ) as monitor_runtime:
        try:
            train_env = Monitor(
                build_letter_env(
                    _env_config(config, evaluation=False),
                    monitor_config_path=monitor_runtime.train_config_path,
                ),
                filename=str(output_dir / "train_monitor.csv"),
            )
            eval_env = Monitor(
                build_letter_env(
                    _env_config(config, evaluation=True),
                    monitor_config_path=monitor_runtime.eval_config_path,
                ),
                filename=str(output_dir / "eval_monitor.csv"),
            )

            if config.seed is not None:
                train_env.reset(seed=config.seed)
                eval_env.reset(seed=config.seed + 10_000)

            callback = PeriodicEvaluationCallback(
                evaluation_env=eval_env,
                output_dir=output_dir,
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                eval_seed_base=config.eval_seed_base,
            )
            model = PPO(
                policy="MultiInputPolicy",
                env=train_env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                policy_kwargs=build_monitor_policy_kwargs(config.encoding, policy_config),
                seed=config.seed,
                tensorboard_log=None,
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
                "n_value": config.n_value,
                "fixed_n": config.fixed_n,
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


def _env_config(config: LetterEnvPPOTrainingConfig, *, evaluation: bool) -> LetterEnvConfig:
    return LetterEnvConfig(
        encoding=config.encoding,
        n_value=config.n_value,
        fixed_n=config.fixed_n,
        max_episode_steps=config.max_episode_steps,
        monitor_progress_bonus=config.monitor_progress_bonus,
        monitor_regression_penalty=config.monitor_regression_penalty,
        neutralize_legacy_transition_bonus=config.neutralize_legacy_transition_bonus,
        legacy_transition_bonus=config.legacy_transition_bonus,
        step_penalty=config.step_penalty,
        no_op_penalty=config.no_op_penalty,
        state_discovery_bonus=0.0 if evaluation else config.state_discovery_bonus,
    )


def _write_run_config(
    path: Path,
    *,
    config: LetterEnvPPOTrainingConfig,
    policy_config: MLPPolicyConfig,
    train_config_path: Path,
    eval_config_path: Path,
    monitor_spec_path: Path,
) -> None:
    payload = {
        "experiment": "letter_env_ppo",
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
    parser.add_argument(
        "--encoding",
        choices=[
            "one_hot",
            "numerical",
            "semantic_progress",
            "simple",
        ],
        default="numerical",
    )
    parser.add_argument("--n-value", type=int, default=5)
    parser.add_argument("--fixed-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-progress-bonus", type=float, default=10.0)
    parser.add_argument("--monitor-regression-penalty", type=float, default=0.0)
    parser.add_argument("--legacy-transition-bonus", type=float, default=10.0)
    parser.add_argument("--include-legacy-transition-bonus", action="store_true")
    parser.add_argument("--step-penalty", type=float, default=0.05)
    parser.add_argument("--no-op-penalty", type=float, default=0.0)
    parser.add_argument("--state-discovery-bonus", type=float, default=0.0)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument("--position-hidden-dim", type=int, default=64)
    parser.add_argument("--monitor-hidden-dim", type=int, default=64)
    parser.add_argument("--network-architecture", type=int, nargs="+", default=[128, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LetterEnvPPOTrainingConfig(
        encoding=args.encoding,
        n_value=args.n_value,
        fixed_n=args.fixed_n,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        max_episode_steps=args.max_episode_steps,
        monitor_progress_bonus=args.monitor_progress_bonus,
        monitor_regression_penalty=args.monitor_regression_penalty,
        neutralize_legacy_transition_bonus=not args.include_legacy_transition_bonus,
        legacy_transition_bonus=args.legacy_transition_bonus,
        step_penalty=args.step_penalty,
        no_op_penalty=args.no_op_penalty,
        state_discovery_bonus=args.state_discovery_bonus,
        output_dir=args.output_dir,
    )
    policy_config = MLPPolicyConfig(
        features_dim=args.features_dim,
        position_hidden_dim=args.position_hidden_dim,
        monitor_hidden_dim=args.monitor_hidden_dim,
        network_architecture=tuple(args.network_architecture),
    )
    summary = train_letter_env_ppo(config, policy_config=policy_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
