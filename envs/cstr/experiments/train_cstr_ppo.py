"""Train PPO on native and RML-backed CSTR control variants."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.cstr import (
    CSTRConfig,
    RMLCSTRConfig,
    make_cstr_env,
    make_manual_rm_cstr_env,
    make_rml_cstr_env,
)
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.experiments.runtime import (
    configure_global_seed,
    json_ready,
    managed_monitor_group,
    utc_now,
    write_json,
)


ENV_VARIANTS = (
    "baseline",
    "rml_hidden",
    "semantic_progress",
    "rml_graph",
    "manual_hidden",
    "manual_rm_semantic_progress",
)
EXTERNAL_MONITOR_VARIANTS = {"rml_hidden", "semantic_progress", "rml_graph"}
REWARD_MODES = ("env", "rml", "env_rml")


@dataclass(frozen=True)
class CSTRPPOConfig:
    """Configuration for one CSTR PPO run."""

    env_variant: str = "semantic_progress"
    reward_mode: str = "env_rml"
    total_timesteps: int = 100_000
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 1e-3
    log_std_init: float = -1.5
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    eval_seed_base: int = 10_000
    checkpoint_freq: int = 0
    max_episode_steps: int = 300
    regulation_violation_steps: int = 10
    soak_steps: int = 10
    monitor_state_limit: int = 16
    graph_encoder_checkpoint: Path | None = None
    recover_from_regulation_failure_during_training: bool = True
    safe_step_bonus: float = 0.10
    stable_step_bonus: float = 1.0
    regulation_entry_bonus: float = 5.0
    success_bonus: float = 50.0
    failure_penalty: float = -50.0
    training_failure_penalty: float = 0.0
    rml_heating_rate_penalty: float = 0.02
    preheat_distance_weight: float = 0.08
    preheat_warming_weight: float = 0.25
    soak_entry_bonus: float = 5.0
    soak_progress_bonus: float = 0.75
    soak_reset_penalty: float = -3.0
    soak_lost_step_penalty: float = 0.50
    approach_distance_weight: float = 1.0
    approach_progress_bonus: float = 5.0
    approach_ca_progress_bonus: float = 4.0
    approach_temp_progress_bonus: float = 4.0
    approach_warming_weight: float = 0.50
    production_entry_bonus: float = 10.0
    regulate_recovery_penalty: float = -10.0
    terminate_on_rml_failure_during_training: bool = False
    randomize_initial_state: bool = False
    randomize_setpoint: bool = False
    enable_disturbance: bool = False
    ca_initial: float = 0.80
    temp_initial: float = 331.0
    initial_coolant_temp: float = 302.5
    ca_setpoint: float = 0.50
    target_temp: float = 350.0
    slew_limit: float = 8.0
    ramp_limit: float = 1.0
    soak_band_low: float = 343.0
    soak_band_high: float = 347.0
    require_soak_concentration_band: bool = False
    soak_concentration_low: float = 0.58
    soak_concentration_high: float = 0.74
    production_temp_low: float = 348.0
    production_temp_high: float = 352.0
    concentration_tolerance: float = 0.04
    ca_overshoot_low: float = 0.44
    deadline_steps: int = 60
    tracking_weight: float = 0.5
    temp_weight: float = 0.015
    action_weight: float = 0.0002
    warning_penalty: float = 0.25
    heating_rate_penalty: float = 0.02
    critical_penalty: float = 200.0
    output_dir: Path = field(default_factory=Path)


@dataclass(frozen=True)
class CSTREvalRecord:
    """Aggregate CSTR evaluation metrics."""

    timesteps: int
    mean_return: float
    mean_base_return: float
    mean_rml_reward: float
    mean_steps: float
    success_rate: float
    rml_success_rate: float
    critical_failure_rate: float
    full_episode_safe_rate: float
    terminal_stable_rate: float
    monitor_failure_rate: float
    warning_rate: float
    mean_warning_events: float
    mean_tracking_error: float
    mean_temperature_violation: float
    mean_max_stable_steps: float
    mean_first_stable_step: float
    mean_regulated_steps_before_failure: float
    mean_monitor_violation_steps: float
    startup_success_rate: float
    deadline_miss_rate: float
    overshoot_rate: float
    soak_completed_rate: float
    mean_steps_to_regulate: float
    mean_rate_violation_count: float
    soak_exact_compliance_rate: float
    soak_under_count_rate: float
    soak_over_count_rate: float
    mean_soak_exit_error: float
    mean_soak_extra_steps: float
    mean_deadline_margin: float
    mean_time_to_success: float
    mean_hidden_soak_exit_error: float


class CSTREvalCallback(BaseCallback):
    """Evaluate PPO checkpoints and append CSTR metrics to CSV."""

    def __init__(
        self,
        *,
        eval_env: Any,
        output_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed_base: int,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.output_dir = output_dir
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.records: list[CSTREvalRecord] = []
        self.best_score: tuple[float, ...] = (float("-inf"),) * 10
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
        record = self._evaluate()
        self.records.append(record)
        with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(record).keys()))
            writer.writerow(asdict(record))
        score = eval_selection_score(record)
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.best_model_path))
        return True

    def _on_training_end(self) -> None:
        return None

    def _evaluate(self) -> CSTREvalRecord:
        _episodes, summary = evaluate_cstr_policy(
            self.eval_env,
            deterministic_model_policy(self.model),
            episodes=self.n_eval_episodes,
            seed=self.eval_seed_base,
        )
        return CSTREvalRecord(timesteps=int(self.num_timesteps), **summary)

    @staticmethod
    def _empty_record() -> CSTREvalRecord:
        return CSTREvalRecord(
            timesteps=0,
            mean_return=0.0,
            mean_base_return=0.0,
            mean_rml_reward=0.0,
            mean_steps=0.0,
            success_rate=0.0,
            rml_success_rate=0.0,
            critical_failure_rate=0.0,
            full_episode_safe_rate=0.0,
            terminal_stable_rate=0.0,
            monitor_failure_rate=0.0,
            warning_rate=0.0,
            mean_warning_events=0.0,
            mean_tracking_error=0.0,
            mean_temperature_violation=0.0,
            mean_max_stable_steps=0.0,
            mean_first_stable_step=0.0,
            mean_regulated_steps_before_failure=0.0,
            mean_monitor_violation_steps=0.0,
            startup_success_rate=0.0,
            deadline_miss_rate=0.0,
            overshoot_rate=0.0,
            soak_completed_rate=0.0,
            mean_steps_to_regulate=0.0,
            mean_rate_violation_count=0.0,
            soak_exact_compliance_rate=0.0,
            soak_under_count_rate=0.0,
            soak_over_count_rate=0.0,
            mean_soak_exit_error=0.0,
            mean_soak_extra_steps=0.0,
            mean_deadline_margin=0.0,
            mean_time_to_success=0.0,
            mean_hidden_soak_exit_error=0.0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-variant", choices=ENV_VARIANTS, default="semantic_progress")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="env_rml")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=-1.5)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--regulation-violation-steps", type=int, default=10)
    parser.add_argument("--soak-steps", type=int, default=10)
    parser.add_argument("--monitor-state-limit", type=int, default=16)
    parser.add_argument(
        "--graph-encoder-checkpoint",
        type=Path,
        default=None,
        help="Frozen CSTR graph encoder checkpoint for --env-variant rml_graph.",
    )
    parser.add_argument(
        "--disable-regulate-recovery-during-training",
        action="store_true",
        help="Use the strict Regulate out-of-band failure rule in training as well as evaluation.",
    )
    parser.add_argument("--safe-step-bonus", type=float, default=0.10)
    parser.add_argument("--stable-step-bonus", type=float, default=1.0)
    parser.add_argument("--regulation-entry-bonus", type=float, default=5.0)
    parser.add_argument("--success-bonus", type=float, default=50.0)
    parser.add_argument("--failure-penalty", type=float, default=-50.0)
    parser.add_argument("--training-failure-penalty", type=float, default=0.0)
    parser.add_argument("--rml-heating-rate-penalty", type=float, default=0.02)
    parser.add_argument("--preheat-distance-weight", type=float, default=0.08)
    parser.add_argument("--preheat-warming-weight", type=float, default=0.25)
    parser.add_argument("--soak-entry-bonus", type=float, default=5.0)
    parser.add_argument("--soak-progress-bonus", type=float, default=0.75)
    parser.add_argument("--soak-reset-penalty", type=float, default=-3.0)
    parser.add_argument("--soak-lost-step-penalty", type=float, default=0.50)
    parser.add_argument("--approach-distance-weight", type=float, default=1.0)
    parser.add_argument("--approach-progress-bonus", type=float, default=5.0)
    parser.add_argument("--approach-ca-progress-bonus", type=float, default=4.0)
    parser.add_argument("--approach-temp-progress-bonus", type=float, default=4.0)
    parser.add_argument("--approach-warming-weight", type=float, default=0.50)
    parser.add_argument("--production-entry-bonus", type=float, default=10.0)
    parser.add_argument("--regulate-recovery-penalty", type=float, default=-10.0)
    parser.add_argument("--terminate-on-rml-failure-during-training", action="store_true")
    parser.add_argument("--fixed-initial-state", action="store_true", default=True)
    parser.add_argument("--randomize-initial-state", action="store_true")
    parser.add_argument("--randomize-setpoint", action="store_true")
    parser.add_argument("--enable-disturbance", action="store_true")
    parser.add_argument("--ca-initial", type=float, default=0.80)
    parser.add_argument("--temp-initial", type=float, default=331.0)
    parser.add_argument("--initial-coolant-temp", type=float, default=302.5)
    parser.add_argument("--ca-setpoint", type=float, default=0.50)
    parser.add_argument("--target-temp", type=float, default=350.0)
    parser.add_argument("--slew-limit", type=float, default=8.0)
    parser.add_argument("--ramp-limit", type=float, default=1.0)
    parser.add_argument("--soak-band-low", type=float, default=343.0)
    parser.add_argument("--soak-band-high", type=float, default=347.0)
    parser.add_argument("--require-soak-concentration-band", action="store_true")
    parser.add_argument("--soak-concentration-low", type=float, default=0.58)
    parser.add_argument("--soak-concentration-high", type=float, default=0.74)
    parser.add_argument("--production-temp-low", type=float, default=348.0)
    parser.add_argument("--production-temp-high", type=float, default=352.0)
    parser.add_argument("--concentration-tolerance", type=float, default=0.04)
    parser.add_argument("--ca-overshoot-low", type=float, default=0.44)
    parser.add_argument("--deadline-steps", type=int, default=60)
    parser.add_argument("--tracking-weight", type=float, default=0.5)
    parser.add_argument("--temp-weight", type=float, default=0.015)
    parser.add_argument("--action-weight", type=float, default=0.0002)
    parser.add_argument("--warning-penalty", type=float, default=0.25)
    parser.add_argument("--heating-rate-penalty", type=float, default=0.02)
    parser.add_argument("--critical-penalty", type=float, default=200.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CSTRPPOConfig:
    output_dir = args.output_dir or default_output_dir(
        env_variant=args.env_variant,
        reward_mode=args.reward_mode,
        seed=args.seed,
    )
    return CSTRPPOConfig(
        env_variant=args.env_variant,
        reward_mode=args.reward_mode,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        log_std_init=args.log_std_init,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed_base=args.eval_seed_base,
        checkpoint_freq=args.checkpoint_freq,
        max_episode_steps=args.max_episode_steps,
        regulation_violation_steps=args.regulation_violation_steps,
        soak_steps=args.soak_steps,
        monitor_state_limit=args.monitor_state_limit,
        graph_encoder_checkpoint=args.graph_encoder_checkpoint,
        recover_from_regulation_failure_during_training=not args.disable_regulate_recovery_during_training,
        safe_step_bonus=args.safe_step_bonus,
        stable_step_bonus=args.stable_step_bonus,
        regulation_entry_bonus=args.regulation_entry_bonus,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        training_failure_penalty=args.training_failure_penalty,
        rml_heating_rate_penalty=args.rml_heating_rate_penalty,
        preheat_distance_weight=args.preheat_distance_weight,
        preheat_warming_weight=args.preheat_warming_weight,
        soak_entry_bonus=args.soak_entry_bonus,
        soak_progress_bonus=args.soak_progress_bonus,
        soak_reset_penalty=args.soak_reset_penalty,
        soak_lost_step_penalty=args.soak_lost_step_penalty,
        approach_distance_weight=args.approach_distance_weight,
        approach_progress_bonus=args.approach_progress_bonus,
        approach_ca_progress_bonus=args.approach_ca_progress_bonus,
        approach_temp_progress_bonus=args.approach_temp_progress_bonus,
        approach_warming_weight=args.approach_warming_weight,
        production_entry_bonus=args.production_entry_bonus,
        regulate_recovery_penalty=args.regulate_recovery_penalty,
        terminate_on_rml_failure_during_training=args.terminate_on_rml_failure_during_training,
        randomize_initial_state=bool(args.randomize_initial_state) or not args.fixed_initial_state,
        randomize_setpoint=args.randomize_setpoint,
        enable_disturbance=args.enable_disturbance,
        ca_initial=args.ca_initial,
        temp_initial=args.temp_initial,
        initial_coolant_temp=args.initial_coolant_temp,
        ca_setpoint=args.ca_setpoint,
        target_temp=args.target_temp,
        slew_limit=args.slew_limit,
        ramp_limit=args.ramp_limit,
        soak_band_low=args.soak_band_low,
        soak_band_high=args.soak_band_high,
        require_soak_concentration_band=args.require_soak_concentration_band,
        soak_concentration_low=args.soak_concentration_low,
        soak_concentration_high=args.soak_concentration_high,
        production_temp_low=args.production_temp_low,
        production_temp_high=args.production_temp_high,
        concentration_tolerance=args.concentration_tolerance,
        ca_overshoot_low=args.ca_overshoot_low,
        deadline_steps=args.deadline_steps,
        tracking_weight=args.tracking_weight,
        temp_weight=args.temp_weight,
        action_weight=args.action_weight,
        warning_penalty=args.warning_penalty,
        heating_rate_penalty=args.heating_rate_penalty,
        critical_penalty=args.critical_penalty,
        output_dir=output_dir,
    )


def default_output_dir(*, env_variant: str, reward_mode: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("envs/cstr/results_and_evaluation/ppo")
        / env_variant
        / f"{reward_mode}_seed{seed}_{timestamp}"
    )


def cstr_config(config: CSTRPPOConfig) -> CSTRConfig:
    return CSTRConfig(
        max_episode_steps=config.max_episode_steps,
        randomize_initial_state=config.randomize_initial_state,
        randomize_setpoint=config.randomize_setpoint,
        enable_disturbance=config.enable_disturbance,
        ca_initial=config.ca_initial,
        temp_initial=config.temp_initial,
        initial_coolant_temp=config.initial_coolant_temp,
        ca_setpoint=config.ca_setpoint,
        target_temp=config.target_temp,
        slew_limit=config.slew_limit,
        ramp_limit=config.ramp_limit,
        soak_steps=config.soak_steps,
        soak_band_low=config.soak_band_low,
        soak_band_high=config.soak_band_high,
        require_soak_concentration_band=config.require_soak_concentration_band,
        soak_concentration_low=config.soak_concentration_low,
        soak_concentration_high=config.soak_concentration_high,
        production_temp_low=config.production_temp_low,
        production_temp_high=config.production_temp_high,
        concentration_tolerance=config.concentration_tolerance,
        ca_overshoot_low=config.ca_overshoot_low,
        deadline_steps=config.deadline_steps,
        tracking_weight=config.tracking_weight,
        temp_weight=config.temp_weight,
        action_weight=config.action_weight,
        warning_penalty=config.warning_penalty,
        heating_rate_penalty=config.heating_rate_penalty,
        critical_penalty=config.critical_penalty,
    )


def make_env(
    config: CSTRPPOConfig,
    *,
    monitor_port: int | None = None,
    config_path: Path | None = None,
    training: bool = False,
):
    native_config = cstr_config(config)
    if config.env_variant == "baseline":
        return make_cstr_env(native_config)
    if config.env_variant in {"manual_hidden", "manual_rm_semantic_progress"}:
        observation_mode = "none" if config.env_variant == "manual_hidden" else "semantic_progress"
        return make_manual_rm_cstr_env(
            RMLCSTRConfig(
                cstr_env=native_config,
                observation_mode=observation_mode,
                reward_mode=config.reward_mode,
                regulation_violation_steps=config.regulation_violation_steps,
                soak_steps=config.soak_steps,
                monitor_state_limit=config.monitor_state_limit,
                terminate_on_rml_failure=bool(config.terminate_on_rml_failure_during_training) if training else True,
                recover_from_regulation_failure=bool(
                    config.recover_from_regulation_failure_during_training
                ) if training else False,
                safe_step_bonus=config.safe_step_bonus,
                stable_step_bonus=config.stable_step_bonus,
                regulation_entry_bonus=config.regulation_entry_bonus,
                success_bonus=config.success_bonus,
                failure_penalty=config.training_failure_penalty if training else config.failure_penalty,
                heating_rate_penalty=config.rml_heating_rate_penalty,
                preheat_distance_weight=config.preheat_distance_weight,
                preheat_warming_weight=config.preheat_warming_weight,
                soak_entry_bonus=config.soak_entry_bonus,
                soak_progress_bonus=config.soak_progress_bonus,
                soak_reset_penalty=config.soak_reset_penalty,
                soak_lost_step_penalty=config.soak_lost_step_penalty,
                approach_distance_weight=config.approach_distance_weight,
                approach_progress_bonus=config.approach_progress_bonus,
                approach_ca_progress_bonus=config.approach_ca_progress_bonus,
                approach_temp_progress_bonus=config.approach_temp_progress_bonus,
                approach_warming_weight=config.approach_warming_weight,
                production_entry_bonus=config.production_entry_bonus,
                regulate_recovery_penalty=config.regulate_recovery_penalty,
            )
        )
    if monitor_port is None or config_path is None:
        raise ValueError("monitor_port and config_path are required for RML env variants.")
    observation_mode = "none" if config.env_variant == "rml_hidden" else config.env_variant
    return make_rml_cstr_env(
        RMLCSTRConfig(
            cstr_env=native_config,
            observation_mode=observation_mode,
            reward_mode=config.reward_mode,
            config_path=config_path,
            monitor_port=monitor_port,
            regulation_violation_steps=config.regulation_violation_steps,
            soak_steps=config.soak_steps,
            monitor_state_limit=config.monitor_state_limit,
            graph_encoder_checkpoint=config.graph_encoder_checkpoint,
            terminate_on_rml_failure=bool(config.terminate_on_rml_failure_during_training) if training else True,
            recover_from_regulation_failure=bool(
                config.recover_from_regulation_failure_during_training
            ) if training else False,
            safe_step_bonus=config.safe_step_bonus,
            stable_step_bonus=config.stable_step_bonus,
            regulation_entry_bonus=config.regulation_entry_bonus,
            success_bonus=config.success_bonus,
            failure_penalty=config.training_failure_penalty if training else config.failure_penalty,
            heating_rate_penalty=config.rml_heating_rate_penalty,
            preheat_distance_weight=config.preheat_distance_weight,
            preheat_warming_weight=config.preheat_warming_weight,
            soak_entry_bonus=config.soak_entry_bonus,
            soak_progress_bonus=config.soak_progress_bonus,
            soak_reset_penalty=config.soak_reset_penalty,
            soak_lost_step_penalty=config.soak_lost_step_penalty,
            approach_distance_weight=config.approach_distance_weight,
            approach_progress_bonus=config.approach_progress_bonus,
            approach_ca_progress_bonus=config.approach_ca_progress_bonus,
            approach_temp_progress_bonus=config.approach_temp_progress_bonus,
            approach_warming_weight=config.approach_warming_weight,
            production_entry_bonus=config.production_entry_bonus,
            regulate_recovery_penalty=config.regulate_recovery_penalty,
        )
    )


def train_ppo(config: CSTRPPOConfig) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_global_seed(config.seed)

    uses_external_monitor = config.env_variant in EXTERNAL_MONITOR_VARIANTS
    if not uses_external_monitor:
        return _train_ppo_with_monitors(
            config,
            started=started,
            train_port=None,
            eval_port=None,
            train_config_path=None,
            eval_config_path=None,
        )

    template_root = output_dir / "monitor_templates"
    train_generated = generate_cstr_rml(
        regulation_violation_steps=config.regulation_violation_steps,
        soak_steps=config.soak_steps,
        recover_from_regulation_failure=config.recover_from_regulation_failure_during_training,
        port=0,
        max_episode_steps=config.max_episode_steps,
        generated_root=template_root / "train",
    )
    eval_generated = generate_cstr_rml(
        regulation_violation_steps=config.regulation_violation_steps,
        soak_steps=config.soak_steps,
        recover_from_regulation_failure=False,
        port=0,
        max_episode_steps=config.max_episode_steps,
        generated_root=template_root / "eval",
    )
    with managed_monitor_group(
        output_dir=output_dir,
        monitor_specs={
            "train": train_generated.spec_path,
            "eval": eval_generated.spec_path,
        },
        monitor_config_templates={
            "train": train_generated.config_path,
            "eval": eval_generated.config_path,
        },
        config_dir_name="monitor_configs",
        log_dir_name="monitor_logs",
        max_episode_steps=config.max_episode_steps,
    ) as runtime:
        return _train_ppo_with_monitors(
            config,
            started=started,
            train_port=runtime.ports["train"],
            eval_port=runtime.ports["eval"],
            train_config_path=runtime.config_paths["train"],
            eval_config_path=runtime.config_paths["eval"],
        )


def _train_ppo_with_monitors(
    config: CSTRPPOConfig,
    *,
    started: float,
    train_port: int | None,
    eval_port: int | None,
    train_config_path: Path | None,
    eval_config_path: Path | None,
) -> dict[str, Any]:
    output_dir = config.output_dir
    train_env = Monitor(
        make_env(config, monitor_port=train_port, config_path=train_config_path, training=True),
        filename=str(output_dir / "train_monitor.csv"),
    )
    eval_env = make_env(config, monitor_port=eval_port, config_path=eval_config_path, training=False)
    try:
        train_env.reset(seed=config.seed)
        eval_env.reset(seed=config.eval_seed_base)

        policy = (
            "MlpPolicy"
            if config.env_variant in {"baseline", "rml_hidden", "manual_hidden"}
            else "MultiInputPolicy"
        )
        _write_config(output_dir / "config.json", config, policy, train_port=train_port, eval_port=eval_port)

        eval_callback = CSTREvalCallback(
            eval_env=eval_env,
            output_dir=output_dir,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            eval_seed_base=config.eval_seed_base,
        )
        callbacks: list[BaseCallback] = [eval_callback]
        if config.checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=config.checkpoint_freq,
                    save_path=str(output_dir / "checkpoints"),
                    name_prefix="checkpoint",
                    verbose=0,
                )
            )

        model = PPO(
            policy=policy,
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            policy_kwargs={"log_std_init": config.log_std_init},
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            verbose=1,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=CallbackList(callbacks), progress_bar=False)
        model.save(str(output_dir / "model_final"))
        train_env.close()

        final_records, final_summary = evaluate_cstr_policy(
            eval_env,
            deterministic_model_policy(model),
            episodes=config.n_eval_episodes,
            seed=config.eval_seed_base,
        )
        write_eval_outputs(output_dir / "final_eval", final_records, final_summary)
        eval_env.close()

        eval_records = [asdict(record) for record in eval_callback.records]
        summary = {
            "experiment": "cstr_rml_ppo",
            "completed_at_utc": utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "training_config": serialized_config(config),
            "policy": policy,
            "monitor_ports": {"train": train_port, "eval": eval_port},
            "eval_records": eval_records,
            "best_eval": max(eval_records, key=eval_selection_score_from_dict, default=None),
            "final_eval": eval_records[-1] if eval_records else None,
            "final_eval_summary": final_summary,
            "paths": {
                "output_dir": str(output_dir),
                "config": str(output_dir / "config.json"),
                "eval_metrics": str(output_dir / "eval_metrics.csv"),
                "train_monitor": str(output_dir / "train_monitor.csv"),
                "model_final": str(output_dir / "model_final.zip"),
                "best_model": str(output_dir / "best_model.zip"),
            },
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        train_env.close()
        eval_env.close()


def evaluate_cstr_policy(
    env: Any,
    policy: Callable[[Any, dict[str, Any]], Any],
    *,
    episodes: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env.action_space.seed(seed)
    records: list[dict[str, Any]] = []
    totals = {
        "return": 0.0,
        "base_return": 0.0,
        "rml_reward": 0.0,
        "steps": 0.0,
        "success": 0.0,
        "rml_success": 0.0,
        "critical_failure": 0.0,
        "full_episode_safe": 0.0,
        "terminal_stable": 0.0,
        "monitor_failure": 0.0,
        "warning": 0.0,
        "warning_events": 0.0,
        "tracking_error": 0.0,
        "temperature_violation": 0.0,
        "max_stable_steps": 0.0,
        "first_stable_step": 0.0,
        "regulated_steps_before_failure": 0.0,
        "monitor_violation_steps": 0.0,
        "startup_success": 0.0,
        "deadline_miss": 0.0,
        "overshoot": 0.0,
        "soak_completed": 0.0,
        "steps_to_regulate": 0.0,
        "rate_violation_count": 0.0,
        "soak_exact_compliance": 0.0,
        "soak_under_count": 0.0,
        "soak_over_count": 0.0,
        "soak_exit_error": 0.0,
        "soak_extra_steps": 0.0,
        "deadline_margin": 0.0,
        "time_to_success": 0.0,
        "hidden_soak_exit_error": 0.0,
    }

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_base_return = 0.0
        episode_rml_reward = 0.0
        startup_config = info.get("config", {})
        required_soak_steps = int(startup_config.get("soak_steps", 10))
        deadline_steps = int(startup_config.get("deadline_steps", 60))
        soak_streak = 0
        soak_completed = False
        steps_to_regulate: int | None = None
        deadline_miss = False
        overshoot = False
        rate_violation_count = 0
        current_soak_dwell = 0
        final_soak_dwell = 0
        monitor_soak_dwell = 0
        final_monitor_soak_dwell = 0
        exited_monitor_soak = False
        entered_regulate = False
        time_to_success: int | None = None
        monitor_encoding = str(info.get("monitor_encoding", "none"))
        while not (terminated or truncated):
            action = policy(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_base_return += float(info.get("base_reward", reward))
            episode_rml_reward += float(info.get("rml_reward", info.get("monitor_reward", 0.0)))
            monitor_encoding = str(info.get("monitor_encoding", monitor_encoding))
            in_soak_now = bool(info.get("event_in_soak_band", False))
            if not entered_regulate:
                if in_soak_now:
                    soak_streak += 1
                    current_soak_dwell += 1
                    if soak_streak >= required_soak_steps:
                        soak_completed = True
                else:
                    if current_soak_dwell > 0:
                        final_soak_dwell = current_soak_dwell
                    if not soak_completed:
                        soak_streak = 0
                    current_soak_dwell = 0
            overshoot = overshoot or bool(info.get("event_overshoot", False))
            rate_violation_count += int(bool(info.get("event_heating_rate_exceeded", False)))
            if bool(info.get("event_past_deadline", False)) and steps_to_regulate is None:
                deadline_miss = True
            if steps_to_regulate is None:
                monitor_phase = str(info.get("monitor_phase", "none"))
                if monitor_phase == "soak" and not exited_monitor_soak:
                    monitor_soak_dwell += 1
                    final_monitor_soak_dwell = monitor_soak_dwell
                elif monitor_phase in {"approach", "regulate", "success", "failure"} and monitor_soak_dwell > 0:
                    exited_monitor_soak = True
                just_entered_regulate = monitor_phase == "regulate" or (soak_completed and bool(info.get("stable_step", False)))
                if just_entered_regulate:
                    steps_to_regulate = int(info.get("steps", 0))
                    entered_regulate = True
                    if current_soak_dwell > 0:
                        final_soak_dwell = current_soak_dwell
            if bool(info.get("monitor_success", False)) and time_to_success is None:
                time_to_success = int(info.get("steps", 0))

        max_stable_steps = int(info.get("max_stable_steps", 0))
        first_stable_step = int(info.get("first_stable_step", -1))
        critical_failure = bool(info.get("critical_events", 0) > 0)
        cumulative_temperature_violation = float(info.get("cumulative_temperature_violation", 0.0))
        full_episode_safe = bool((terminated or truncated) and not critical_failure and cumulative_temperature_violation <= 1e-9)
        terminal_stable = bool(info.get("stable_step", False))
        physical_success = full_episode_safe and terminal_stable
        rml_success = bool(info.get("monitor_success", False))
        monitor_failed = bool(info.get("monitor_failed", False))
        final_steps = int(info.get("steps", 0))
        startup_success = bool(
            terminal_stable
            and full_episode_safe
            and soak_completed
            and not overshoot
            and not deadline_miss
            and steps_to_regulate is not None
            and steps_to_regulate <= deadline_steps
            and not monitor_failed
        )
        regulated_steps_before_failure = max_stable_steps if monitor_failed else final_steps
        missing_regulate_step = config_missing_step_value(final_steps)
        if steps_to_regulate is None and current_soak_dwell > 0:
            final_soak_dwell = current_soak_dwell
        # The RML soak counter is represented by monitor states Soak_1..Soak_K.
        # The physical in_soak event that advances Soak_K -> Approach is the
        # K+1-th consecutive in-soak event, so compliance must be measured from
        # monitor-phase soak dwell, not raw in_soak predicate dwell.
        soak_exit_error = int(final_monitor_soak_dwell - required_soak_steps)
        soak_exact_compliance = bool(steps_to_regulate is not None and final_monitor_soak_dwell == required_soak_steps)
        soak_under_count = bool(steps_to_regulate is None or final_monitor_soak_dwell < required_soak_steps)
        soak_over_count = bool(steps_to_regulate is not None and final_monitor_soak_dwell > required_soak_steps)
        soak_extra_steps = max(0, soak_exit_error)
        deadline_margin = (
            int(deadline_steps - steps_to_regulate)
            if steps_to_regulate is not None
            else int(deadline_steps - final_steps)
        )
        success_time_value = int(time_to_success) if time_to_success is not None else config_missing_step_value(final_steps)
        hidden_soak_exit_error = soak_exit_error if monitor_encoding == "none" else 0
        record = {
            "episode": episode,
            "seed": seed + episode,
            "steps": final_steps,
            "episode_return": episode_return,
            "episode_base_return": episode_base_return,
            "episode_rml_reward": episode_rml_reward,
            "physical_success": bool(physical_success),
            "rml_success": bool(rml_success),
            "monitor_failed": bool(monitor_failed),
            "monitor_phase": str(info.get("monitor_phase", "none")),
            "critical_failure": critical_failure,
            "full_episode_safe": bool(full_episode_safe),
            "terminal_stable": bool(terminal_stable),
            "warning_events": int(info.get("warning_events", 0)),
            "tracking_error": float(info.get("tracking_error", 0.0)),
            "temperature_violation": float(info.get("temperature_violation", 0.0)),
            "cumulative_tracking_error": float(info.get("cumulative_tracking_error", 0.0)),
            "cumulative_temperature_violation": cumulative_temperature_violation,
            "max_stable_steps": max_stable_steps,
            "first_stable_step": first_stable_step,
            "regulated_steps_before_failure": int(regulated_steps_before_failure),
            "monitor_violation_steps": int(info.get("monitor_violation_steps", 0)),
            "startup_success": bool(startup_success),
            "deadline_miss": bool(deadline_miss or (steps_to_regulate is None and final_steps > deadline_steps)),
            "overshoot": bool(overshoot),
            "soak_completed": bool(soak_completed),
            "steps_to_regulate": int(steps_to_regulate) if steps_to_regulate is not None else missing_regulate_step,
            "rate_violation_count": int(rate_violation_count),
            "soak_dwell_before_regulate": int(final_soak_dwell),
            "monitor_soak_dwell_before_approach": int(final_monitor_soak_dwell),
            "soak_exit_error": int(soak_exit_error),
            "soak_exact_compliance": bool(soak_exact_compliance),
            "soak_under_count": bool(soak_under_count),
            "soak_over_count": bool(soak_over_count),
            "soak_extra_steps": int(soak_extra_steps),
            "deadline_margin": int(deadline_margin),
            "time_to_success": int(success_time_value),
            "hidden_soak_exit_error": int(hidden_soak_exit_error),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        records.append(record)
        totals["return"] += episode_return
        totals["base_return"] += episode_base_return
        totals["rml_reward"] += episode_rml_reward
        totals["steps"] += record["steps"]
        totals["success"] += float(physical_success or rml_success)
        totals["rml_success"] += float(rml_success)
        totals["critical_failure"] += float(record["critical_failure"])
        totals["full_episode_safe"] += float(full_episode_safe)
        totals["terminal_stable"] += float(terminal_stable)
        totals["monitor_failure"] += float(monitor_failed)
        totals["warning"] += float(record["warning_events"] > 0)
        totals["warning_events"] += record["warning_events"]
        totals["tracking_error"] += record["tracking_error"]
        totals["temperature_violation"] += record["temperature_violation"]
        totals["max_stable_steps"] += record["max_stable_steps"]
        totals["first_stable_step"] += first_stable_step if first_stable_step >= 0 else config_missing_step_value(record["steps"])
        totals["regulated_steps_before_failure"] += record["regulated_steps_before_failure"]
        totals["monitor_violation_steps"] += record["monitor_violation_steps"]
        totals["startup_success"] += float(record["startup_success"])
        totals["deadline_miss"] += float(record["deadline_miss"])
        totals["overshoot"] += float(record["overshoot"])
        totals["soak_completed"] += float(record["soak_completed"])
        totals["steps_to_regulate"] += record["steps_to_regulate"]
        totals["rate_violation_count"] += record["rate_violation_count"]
        totals["soak_exact_compliance"] += float(record["soak_exact_compliance"])
        totals["soak_under_count"] += float(record["soak_under_count"])
        totals["soak_over_count"] += float(record["soak_over_count"])
        totals["soak_exit_error"] += record["soak_exit_error"]
        totals["soak_extra_steps"] += record["soak_extra_steps"]
        totals["deadline_margin"] += record["deadline_margin"]
        totals["time_to_success"] += record["time_to_success"]
        totals["hidden_soak_exit_error"] += record["hidden_soak_exit_error"]

    n = max(episodes, 1)
    summary = {
        "mean_return": totals["return"] / n,
        "mean_base_return": totals["base_return"] / n,
        "mean_rml_reward": totals["rml_reward"] / n,
        "mean_steps": totals["steps"] / n,
        "success_rate": totals["success"] / n,
        "rml_success_rate": totals["rml_success"] / n,
        "critical_failure_rate": totals["critical_failure"] / n,
        "full_episode_safe_rate": totals["full_episode_safe"] / n,
        "terminal_stable_rate": totals["terminal_stable"] / n,
        "monitor_failure_rate": totals["monitor_failure"] / n,
        "warning_rate": totals["warning"] / n,
        "mean_warning_events": totals["warning_events"] / n,
        "mean_tracking_error": totals["tracking_error"] / n,
        "mean_temperature_violation": totals["temperature_violation"] / n,
        "mean_max_stable_steps": totals["max_stable_steps"] / n,
        "mean_first_stable_step": totals["first_stable_step"] / n,
        "mean_regulated_steps_before_failure": totals["regulated_steps_before_failure"] / n,
        "mean_monitor_violation_steps": totals["monitor_violation_steps"] / n,
        "startup_success_rate": totals["startup_success"] / n,
        "deadline_miss_rate": totals["deadline_miss"] / n,
        "overshoot_rate": totals["overshoot"] / n,
        "soak_completed_rate": totals["soak_completed"] / n,
        "mean_steps_to_regulate": totals["steps_to_regulate"] / n,
        "mean_rate_violation_count": totals["rate_violation_count"] / n,
        "soak_exact_compliance_rate": totals["soak_exact_compliance"] / n,
        "soak_under_count_rate": totals["soak_under_count"] / n,
        "soak_over_count_rate": totals["soak_over_count"] / n,
        "mean_soak_exit_error": totals["soak_exit_error"] / n,
        "mean_soak_extra_steps": totals["soak_extra_steps"] / n,
        "mean_deadline_margin": totals["deadline_margin"] / n,
        "mean_time_to_success": totals["time_to_success"] / n,
        "mean_hidden_soak_exit_error": totals["hidden_soak_exit_error"] / n,
    }
    return records, summary


def config_missing_step_value(steps: int) -> int:
    return int(max(steps, 0) + 1)


def eval_selection_score(record: CSTREvalRecord) -> tuple[float, ...]:
    return (
        float(record.startup_success_rate),
        float(record.rml_success_rate),
        float(record.soak_completed_rate),
        float(record.mean_regulated_steps_before_failure),
        float(record.mean_max_stable_steps),
        -float(record.mean_steps_to_regulate),
        -float(record.mean_tracking_error),
        -float(record.critical_failure_rate),
        -float(record.deadline_miss_rate),
        -float(record.overshoot_rate),
    )


def eval_selection_score_from_dict(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record["startup_success_rate"]),
        float(record["rml_success_rate"]),
        float(record["soak_completed_rate"]),
        float(record["mean_regulated_steps_before_failure"]),
        float(record["mean_max_stable_steps"]),
        -float(record["mean_steps_to_regulate"]),
        -float(record["mean_tracking_error"]),
        -float(record["critical_failure_rate"]),
        -float(record["deadline_miss_rate"]),
        -float(record["overshoot_rate"]),
    )


def deterministic_model_policy(model: Any) -> Callable[[Any, dict[str, Any]], Any]:
    def _policy(observation: Any, _info: dict[str, Any]) -> Any:
        action, _state = model.predict(observation, deterministic=True)
        return np.asarray(action)

    return _policy


def write_eval_outputs(output_dir: str | Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if records:
        with (path / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    write_json(path / "summary.json", summary)


def serialized_config(config: CSTRPPOConfig) -> dict[str, Any]:
    return json_ready(asdict(config))


def _write_config(
    path: Path,
    config: CSTRPPOConfig,
    policy: str,
    *,
    train_port: int | None,
    eval_port: int | None,
) -> None:
    payload = {
        "experiment": "cstr_rml_ppo",
        "started_at_utc": utc_now(),
        "training_config": serialized_config(config),
        "policy": policy,
        "monitor_ports": {"train": train_port, "eval": eval_port},
    }
    write_json(path, payload)


def main() -> None:
    summary = train_ppo(config_from_args(parse_args()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
