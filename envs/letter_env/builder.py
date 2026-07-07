"""Shared LetterEnv construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from envs.letter_env.encodings import build_letter_env_monitor_encoding
from envs.letter_env.env import LetterEnv
from rml_rm.encodings.monitor_state import (
    extract_numerical_values,
    normalize_monitor_state,
    replace_numerical_parts,
    split_top_level_factors,
)
from rml_rm.wrappers import PropositionVectorObservation, RMLMonitorWrapper


@dataclass(frozen=True)
class LetterEnvConfig:
    """Configuration for the cleaned LetterEnv environment stack."""

    encoding: str = "one_hot"
    learned_gru_checkpoint: str | Path | None = None
    learned_graph_checkpoint: str | Path | None = None
    n_value: int = 1
    fixed_n: int | None = None
    task_prefix: str = "ABC"
    counted_suffix: str = "D"
    max_episode_steps: int = 200
    monitor_progress_bonus: float = 0.0
    monitor_regression_penalty: float = 0.0
    neutralize_legacy_transition_bonus: bool = True
    legacy_transition_bonus: float = 10.0
    step_penalty: float = 0.0
    no_op_penalty: float = 0.0
    state_discovery_bonus: float = 0.0


class FixedLetterNWrapper(gym.Wrapper):
    """Force LetterEnv to sample the same count on every reset."""

    def __init__(self, env: gym.Env, *, fixed_n: int) -> None:
        super().__init__(env)
        self.fixed_n = int(fixed_n)

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        options["n"] = self.fixed_n
        return self.env.reset(options=options, **kwargs)


class LetterEnvRewardShapingWrapper(gym.Wrapper):
    """Apply monitor-progress and step-based shaping for neural experiments."""

    def __init__(
        self,
        env: gym.Env,
        *,
        monitor_progress_bonus: float,
        monitor_regression_penalty: float,
        step_penalty: float,
        no_op_penalty: float,
    ) -> None:
        super().__init__(env)
        self.monitor_progress_bonus = float(monitor_progress_bonus)
        self.monitor_regression_penalty = float(monitor_regression_penalty)
        self.step_penalty = float(step_penalty)
        self.no_op_penalty = float(no_op_penalty)
        self.previous_monitor_progress = 0.0
        self.previous_position: tuple[float, ...] | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_monitor_progress = 0.0
        self.previous_position = _position_key(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)
        monitor_progress = _monitor_progress_potential(
            info.get("monitor_state_unencoded"),
            terminated=bool(terminated),
            reward_before_wrapper=shaped_reward,
        )

        if monitor_progress > self.previous_monitor_progress:
            shaped_reward += self.monitor_progress_bonus
        elif monitor_progress < self.previous_monitor_progress:
            shaped_reward += self.monitor_regression_penalty

        if self.step_penalty:
            shaped_reward -= self.step_penalty

        position = _position_key(observation)
        if self.no_op_penalty and position == self.previous_position:
            shaped_reward -= self.no_op_penalty

        self.previous_monitor_progress = monitor_progress
        self.previous_position = position
        return observation, shaped_reward, terminated, truncated, info


class StateDiscoveryRewardWrapper(gym.Wrapper):
    """Reward the first visit to a previously unseen encoded state."""

    def __init__(self, env: gym.Env, *, bonus: float) -> None:
        super().__init__(env)
        self.bonus = float(bonus)
        self.seen_states: set[tuple[Any, ...]] = set()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.seen_states = {_observation_key(observation)}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        key = _observation_key(observation)
        if key not in self.seen_states:
            self.seen_states.add(key)
            reward = float(reward) + self.bonus
        return observation, reward, terminated, truncated, info


def build_letter_env(config: LetterEnvConfig, *, monitor_config_path: str | Path) -> gym.Env:
    """Build the wrapped LetterEnv stack used by experiments."""
    if config.n_value < 1:
        raise ValueError("n_value must be at least 1.")
    if config.fixed_n is not None and not 1 <= config.fixed_n <= config.n_value:
        raise ValueError("fixed_n must be in 1..n_value.")

    raw_env: gym.Env = LetterEnv(
        max_n=config.n_value,
        task_prefix=config.task_prefix,
        counted_suffix=config.counted_suffix,
        max_episode_steps=config.max_episode_steps,
    )
    if config.fixed_n is not None:
        raw_env = FixedLetterNWrapper(raw_env, fixed_n=config.fixed_n)

    monitor_encoder, initial_monitor_state, monitor_space = build_letter_env_monitor_encoding(
        config.encoding,
        learned_gru_checkpoint=config.learned_gru_checkpoint,
        learned_graph_checkpoint=config.learned_graph_checkpoint,
    )
    env: gym.Env = RMLMonitorWrapper(
        PropositionVectorObservation(raw_env),
        config_path=monitor_config_path,
        monitor_encoder=monitor_encoder,
        initial_monitor_state=initial_monitor_state,
        monitor_space=monitor_space,
        transition_bonus=config.legacy_transition_bonus,
        include_transition_bonus=not config.neutralize_legacy_transition_bonus,
    )
    env = LetterEnvRewardShapingWrapper(
        env,
        monitor_progress_bonus=config.monitor_progress_bonus,
        monitor_regression_penalty=config.monitor_regression_penalty,
        step_penalty=config.step_penalty,
        no_op_penalty=config.no_op_penalty,
    )
    if config.state_discovery_bonus:
        env = StateDiscoveryRewardWrapper(env, bonus=config.state_discovery_bonus)
    return env


def _position_key(observation: dict[str, Any]) -> tuple[float, ...]:
    return tuple(np.asarray(observation["position"], dtype=np.float32).round(6).tolist())


def _observation_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    position = _position_key(observation)
    monitor_value = observation.get("monitor")
    if isinstance(monitor_value, (int, np.integer)):
        monitor = (int(monitor_value),)
    else:
        monitor = tuple(np.asarray(monitor_value, dtype=np.float32).round(6).tolist())
    return position + monitor


def _monitor_progress_potential(
    raw_monitor_state: Any,
    *,
    terminated: bool,
    reward_before_wrapper: float,
) -> float:
    if terminated and reward_before_wrapper > 0.0:
        return 1000.0
    if terminated and reward_before_wrapper < 0.0:
        return -1000.0
    if raw_monitor_state is None:
        return 0.0

    normalized_state = normalize_monitor_state(str(raw_monitor_state))
    if normalized_state == "false_verdict":
        return -1000.0
    if normalized_state == "1":
        return 1000.0

    factors = [replace_numerical_parts(factor) for factor in split_top_level_factors(normalized_state)]
    values: list[float] = []
    for factor in split_top_level_factors(normalized_state):
        factor_values = extract_numerical_values(factor)
        if factor_values:
            values.extend(factor_values)
    primary_value = values[0] if values else 0.0

    if any("star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[{num}]))" in factor for factor in factors):
        return 400.0 - primary_value
    if any("(app(gen([n],),[{num}]),[=guarded(var(n)>0" in factor for factor in factors):
        return 350.0 - primary_value
    if any("star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[{num}]))" in factor for factor in factors):
        return 250.0 + primary_value
    if any(
        "app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[{num}])" in factor
        for factor in factors
    ):
        return 150.0 + primary_value
    if any("(star(not_abcd:eps)*app(,[{num}]))" in factor for factor in factors):
        return primary_value
    if any(
        "star(not_abcd:eps)*(app(gen([n],),[{num}])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[{num}]))"
        in factor
        for factor in factors
    ):
        return 50.0 + primary_value
    return 0.0
