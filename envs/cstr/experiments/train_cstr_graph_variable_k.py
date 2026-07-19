"""Train a graph-encoded CSTR RML policy across multiple soak lengths."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cstr.experiments.train_cstr_ppo import (
    CSTRPPOConfig,
    deterministic_model_policy,
    evaluate_cstr_policy,
    make_env,
    serialized_config,
    write_eval_outputs,
    write_json,
)
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.monitors import RMLMonitorProcess, find_free_port


DEFAULT_TRAIN_KS = (5, 8, 10, 12)
DEFAULT_EVAL_KS = DEFAULT_TRAIN_KS


class VariableKEvalCallback(BaseCallback):
    """Evaluate one policy against several fixed-K monitor specs."""

    def __init__(
        self,
        *,
        eval_envs: dict[int, Any],
        train_soak_steps: set[int],
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed_base: int,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_envs = eval_envs
        self.train_soak_steps = set(train_soak_steps)
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.records: list[dict[str, Any]] = []
        self.metrics_path = self.output_dir / "eval_metrics_by_k.csv"
        self.best_score = (float("-inf"), float("-inf"))
        self.best_model_path = self.output_dir / "best_model"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames())
            writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        records = self._evaluate_all()
        self.records.extend(records)
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames())
            writer.writerows(records)
        score = self._selection_score(records)
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.best_model_path))
        return True

    def _evaluate_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for soak_steps, env in sorted(self.eval_envs.items()):
            _episodes, summary = evaluate_cstr_policy(
                env,
                deterministic_model_policy(self.model),
                episodes=self.n_eval_episodes,
                seed=self.eval_seed_base,
            )
            rows.append({"timesteps": int(self.num_timesteps), "soak_steps": int(soak_steps), **summary})
        return rows

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "timesteps",
            "soak_steps",
            "mean_return",
            "mean_base_return",
            "mean_rml_reward",
            "mean_steps",
            "success_rate",
            "rml_success_rate",
            "critical_failure_rate",
            "full_episode_safe_rate",
            "terminal_stable_rate",
            "monitor_failure_rate",
            "warning_rate",
            "mean_warning_events",
            "mean_tracking_error",
            "mean_temperature_violation",
            "mean_max_stable_steps",
            "mean_first_stable_step",
            "mean_regulated_steps_before_failure",
            "mean_monitor_violation_steps",
            "startup_success_rate",
            "deadline_miss_rate",
            "overshoot_rate",
            "soak_completed_rate",
            "mean_steps_to_regulate",
            "mean_rate_violation_count",
            "soak_exact_compliance_rate",
            "soak_under_count_rate",
            "soak_over_count_rate",
            "mean_soak_exit_error",
            "mean_soak_extra_steps",
            "mean_deadline_margin",
            "mean_time_to_success",
            "mean_hidden_soak_exit_error",
        ]

    def _selection_score(self, records: list[dict[str, Any]]) -> tuple[float, float]:
        if not records:
            return (float("-inf"), float("-inf"))
        train_rows = [row for row in records if int(row["soak_steps"]) in self.train_soak_steps]
        return (
            _mean(train_rows, "startup_success_rate"),
            _mean(train_rows, "mean_return"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-soak-steps", type=int, nargs="+", default=list(DEFAULT_TRAIN_KS))
    parser.add_argument("--eval-soak-steps", type=int, nargs="+", default=list(DEFAULT_EVAL_KS))
    parser.add_argument("--graph-encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--deadline-steps", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=-1.5)
    parser.add_argument("--training-failure-penalty", type=float, default=-25.0)
    parser.add_argument(
        "--disable-regulate-recovery-during-training",
        action="store_true",
        help="Use strict regulation failure transitions during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_variable_k(args)
    print(json.dumps(summary, indent=2, default=str))


def train_variable_k(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_processes: list[RMLMonitorProcess] = []
    train_specs: dict[int, tuple[int, Path]] = {}
    eval_specs: dict[int, tuple[int, Path]] = {}

    try:
        for soak_steps in args.train_soak_steps:
            port = find_free_port()
            generated = generate_cstr_rml(
                soak_steps=int(soak_steps),
                recover_from_regulation_failure=True,
                port=port,
                max_episode_steps=args.max_episode_steps,
                generated_root=output_dir / "monitors" / "train" / f"k{soak_steps}",
            )
            monitor_processes.append(
                RMLMonitorProcess(
                    generated.spec_path,
                    port=port,
                    log_path=output_dir / f"train_k{soak_steps}_rml_monitor.log",
                ).start()
            )
            train_specs[int(soak_steps)] = (port, generated.config_path)

        for soak_steps in args.eval_soak_steps:
            port = find_free_port()
            generated = generate_cstr_rml(
                soak_steps=int(soak_steps),
                recover_from_regulation_failure=False,
                port=port,
                max_episode_steps=args.max_episode_steps,
                generated_root=output_dir / "monitors" / "eval" / f"k{soak_steps}",
            )
            monitor_processes.append(
                RMLMonitorProcess(
                    generated.spec_path,
                    port=port,
                    log_path=output_dir / f"eval_k{soak_steps}_rml_monitor.log",
                ).start()
            )
            eval_specs[int(soak_steps)] = (port, generated.config_path)

        train_env = DummyVecEnv(
            [
                _make_training_env_fn(args, soak_steps, port, config_path)
                for soak_steps, (port, config_path) in sorted(train_specs.items())
            ]
        )
        eval_envs = {
            soak_steps: make_env(
                _config_for_k(args, soak_steps, training=False),
                monitor_port=port,
                config_path=config_path,
                training=False,
            )
            for soak_steps, (port, config_path) in sorted(eval_specs.items())
        }

        eval_callback = VariableKEvalCallback(
            eval_envs=eval_envs,
            train_soak_steps=set(map(int, args.train_soak_steps)),
            output_dir=output_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed_base=10_000,
        )
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=float(args.learning_rate),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=float(args.ent_coef),
            policy_kwargs={"log_std_init": float(args.log_std_init)},
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=int(args.seed),
            verbose=1,
        )
        _write_run_config(output_dir / "config.json", args)
        model.learn(total_timesteps=int(args.total_timesteps), callback=CallbackList([eval_callback]), progress_bar=False)
        model.save(str(output_dir / "model_final"))
        train_env.close()

        final_eval: dict[int, dict[str, float]] = {}
        final_records: dict[int, list[dict[str, Any]]] = {}
        for soak_steps, env in eval_envs.items():
            records, summary = evaluate_cstr_policy(
                env,
                deterministic_model_policy(model),
                episodes=int(args.n_eval_episodes),
                seed=10_000,
            )
            final_records[soak_steps] = records
            final_eval[soak_steps] = summary
            write_eval_outputs(output_dir / "final_eval" / f"k{soak_steps}", records, summary)
            env.close()

        summary = {
            "experiment": "cstr_graph_variable_k_ppo",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.monotonic() - started,
            "train_soak_steps": list(map(int, args.train_soak_steps)),
            "eval_soak_steps": list(map(int, args.eval_soak_steps)),
            "training_config": _serialized_args(args),
            "policy": "MultiInputPolicy",
            "eval_records": eval_callback.records,
            "best_eval": _best_records(eval_callback.records, train_soak_steps=set(map(int, args.train_soak_steps))),
            "final_eval": {str(k): v for k, v in sorted(final_eval.items())},
            "paths": {
                "output_dir": str(output_dir),
                "config": str(output_dir / "config.json"),
                "eval_metrics_by_k": str(output_dir / "eval_metrics_by_k.csv"),
                "model_final": str(output_dir / "model_final.zip"),
                "best_model": str(output_dir / "best_model.zip"),
            },
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        for process in monitor_processes:
            process.stop()


def _make_training_env_fn(args: argparse.Namespace, soak_steps: int, monitor_port: int, config_path: Path):
    def _factory():
        return Monitor(
            make_env(
                _config_for_k(args, soak_steps, training=True),
                monitor_port=monitor_port,
                config_path=config_path,
                training=True,
            ),
            filename=str(args.output_dir / f"train_k{soak_steps}_monitor.csv"),
        )

    return _factory


def _config_for_k(args: argparse.Namespace, soak_steps: int, *, training: bool) -> CSTRPPOConfig:
    return CSTRPPOConfig(
        env_variant="rml_graph",
        reward_mode="env_rml",
        total_timesteps=int(args.total_timesteps),
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        ent_coef=float(args.ent_coef),
        log_std_init=float(args.log_std_init),
        eval_freq=int(args.eval_freq),
        n_eval_episodes=int(args.n_eval_episodes),
        max_episode_steps=int(args.max_episode_steps),
        soak_steps=int(soak_steps),
        monitor_state_limit=32,
        graph_encoder_checkpoint=args.graph_encoder_checkpoint,
        recover_from_regulation_failure_during_training=not bool(args.disable_regulate_recovery_during_training),
        stable_step_bonus=3.0,
        training_failure_penalty=float(args.training_failure_penalty),
        rml_heating_rate_penalty=0.0,
        heating_rate_penalty=0.0,
        critical_penalty=200.0,
        concentration_tolerance=0.08,
        production_temp_low=346.0,
        production_temp_high=354.0,
        require_soak_concentration_band=True,
        soak_concentration_low=0.58,
        soak_concentration_high=0.74,
        approach_distance_weight=1.0,
        approach_progress_bonus=5.0,
        approach_ca_progress_bonus=4.0,
        approach_temp_progress_bonus=4.0,
        approach_warming_weight=0.5,
        production_entry_bonus=10.0,
        regulate_recovery_penalty=-10.0,
        terminate_on_rml_failure_during_training=False if training else True,
        deadline_steps=int(args.deadline_steps),
        output_dir=args.output_dir,
    )


def _write_run_config(path: Path, args: argparse.Namespace) -> None:
    write_json(path, {"experiment": "cstr_graph_variable_k_ppo", "args": _serialized_args(args)})


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _best_records(records: list[dict[str, Any]], *, train_soak_steps: set[int]) -> list[dict[str, Any]]:
    if not records:
        return []
    by_timestep: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_timestep.setdefault(int(record["timesteps"]), []).append(record)
    return max(
        by_timestep.values(),
        key=lambda rows: (
            _mean([row for row in rows if int(row["soak_steps"]) in train_soak_steps], "startup_success_rate"),
            _mean([row for row in rows if int(row["soak_steps"]) in train_soak_steps], "mean_return"),
        ),
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("-inf")
    return sum(float(row[key]) for row in rows) / len(rows)


if __name__ == "__main__":
    main()
