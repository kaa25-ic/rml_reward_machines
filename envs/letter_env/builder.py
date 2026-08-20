"""Shared LetterEnv construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    finite_state_rm_max_n: int | None = None


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
    if config.encoding == "finite_state_rm" and config.task_prefix != "ABC":
        raise ValueError("finite_state_rm currently supports the standard ABCD^n task only.")
    finite_state_rm_max_n = config.finite_state_rm_max_n or config.n_value
    if config.encoding == "finite_state_rm" and finite_state_rm_max_n < config.n_value:
        raise ValueError("finite_state_rm_max_n must be at least n_value.")

    raw_env: gym.Env = LetterEnv(
        max_n=config.n_value,
        task_prefix=config.task_prefix,
        counted_suffix=config.counted_suffix,
        max_episode_steps=config.max_episode_steps,
    )
    if config.fixed_n is not None:
        raw_env = FixedLetterNWrapper(raw_env, fixed_n=config.fixed_n)

    if config.encoding == "finite_state_rm":
        env: gym.Env = FiniteStateRewardMachineWrapper(
            PropositionVectorObservation(raw_env),
            max_n=finite_state_rm_max_n,
            transition_bonus=config.legacy_transition_bonus,
            include_transition_bonus=not config.neutralize_legacy_transition_bonus,
        )
    else:
        monitor_encoder, initial_monitor_state, monitor_space = build_letter_env_monitor_encoding(
            config.encoding,
            learned_gru_checkpoint=config.learned_gru_checkpoint,
            learned_graph_checkpoint=config.learned_graph_checkpoint,
        )
        env = RMLMonitorWrapper(
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


class FiniteStateRewardMachineWrapper(gym.Wrapper):
    """Hand-coded finite-state reward machine for the standard LetterEnv task.

    This is a non-RML baseline. It observes the same proposition stream as the
    RML monitor and exposes a one-hot automaton state to the policy.
    """

    success_reward = 100.0
    failure_reward = -40.0

    def __init__(
        self,
        env: gym.Env,
        *,
        max_n: int,
        transition_bonus: float,
        include_transition_bonus: bool,
    ) -> None:
        super().__init__(env)
        self.max_n = int(max_n)
        self.transition_bonus = float(transition_bonus)
        self.include_transition_bonus = bool(include_transition_bonus)
        self.sampled_n = 1
        self.progress = 0
        self.previous_progress = 0
        self.terminal_state: str | None = None
        self.monitor_state = self._encode_progress()
        spaces_dict = dict(self.env.observation_space.spaces)
        spaces_dict["monitor"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_count,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(spaces_dict)

    @property
    def state_count(self) -> int:
        # Non-terminal progress states 0..max_n+2, plus success and failure.
        return self.max_n + 5

    @property
    def success_index(self) -> int:
        return self.state_count - 2

    @property
    def failure_index(self) -> int:
        return self.state_count - 1

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.sampled_n = int(info.get("sampled_n", 1))
        self.progress = 0
        self.previous_progress = 0
        self.terminal_state = None
        self.monitor_state = self._encode_progress()
        info = dict(info)
        info.update(self._monitor_info(monitor_reward=0.0, transition_bonus=0.0))
        return self._with_monitor(observation), info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self.previous_progress = self.progress
        label = self._label_from_observation(observation)
        self._advance(label)

        monitor_reward = 0.0
        if self.terminal_state == "success":
            monitor_reward = self.success_reward
        elif self.terminal_state == "failure":
            monitor_reward = self.failure_reward

        transition_bonus = 0.0
        if (
            self.include_transition_bonus
            and self.terminal_state != "failure"
            and self.progress != self.previous_progress
        ):
            transition_bonus = self.transition_bonus

        self.monitor_state = self._encode_progress()
        terminated = bool(terminated or self.terminal_state is not None)
        info["base_reward"] = float(base_reward)
        info.update(
            self._monitor_info(
                monitor_reward=monitor_reward,
                transition_bonus=transition_bonus,
            )
        )
        return (
            self._with_monitor(observation),
            monitor_reward + transition_bonus,
            terminated,
            truncated,
            info,
        )

    def _advance(self, label: str) -> None:
        if self.terminal_state is not None or label == "_":
            return
        expected = self._expected_label()
        if label != expected:
            self.terminal_state = "failure"
            return
        self.progress += 1
        if self.progress >= self._target_length():
            self.terminal_state = "success"

    def _expected_label(self) -> str:
        if self.progress == 0:
            return "A"
        if self.progress == 1:
            return "B"
        if self.progress == 2:
            return "C"
        return "D"

    def _target_length(self) -> int:
        return 3 + self.sampled_n

    def _label_from_observation(self, observation: dict[str, Any]) -> str:
        vector = np.asarray(observation["position"], dtype=np.float32).reshape(-1)
        proposition_features = vector[2:]
        if proposition_features.size == 0:
            return "_"
        proposition_index = int(np.argmax(proposition_features))
        if float(proposition_features[proposition_index]) <= 0.0:
            return "_"
        index_to_proposition = getattr(self.env.unwrapped, "index_to_proposition", {})
        return str(index_to_proposition.get(proposition_index, "_"))

    def _encode_progress(self) -> np.ndarray:
        vector = np.zeros(self.state_count, dtype=np.float32)
        if self.terminal_state == "success":
            vector[self.success_index] = 1.0
        elif self.terminal_state == "failure":
            vector[self.failure_index] = 1.0
        else:
            vector[min(self.progress, self.max_n + 2)] = 1.0
        return vector

    def _raw_monitor_state(self) -> str:
        if self.terminal_state == "success":
            return "finite_state_rm:success"
        if self.terminal_state == "failure":
            return "finite_state_rm:failure"
        return f"finite_state_rm:progress_{self.progress}"

    def _monitor_info(self, *, monitor_reward: float, transition_bonus: float) -> dict[str, Any]:
        return {
            "monitor_verdict": "true" if self.terminal_state == "success" else "currently_false",
            "monitor_state_unencoded": self._raw_monitor_state(),
            "monitor_transition_bonus": float(transition_bonus),
            "monitor_reward": float(monitor_reward),
            "finite_state_reward_machine": True,
            "finite_state_rm_progress": int(self.progress),
            "finite_state_rm_target_length": int(self._target_length()),
        }

    def _with_monitor(self, observation: dict[str, Any]) -> dict[str, Any]:
        wrapped = dict(observation)
        wrapped["monitor"] = self.monitor_state.copy()
        return wrapped


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

    raw_state = str(raw_monitor_state)
    if raw_state.startswith("finite_state_rm:progress_"):
        return float(raw_state.rsplit("_", maxsplit=1)[-1])
    if raw_state == "finite_state_rm:success":
        return 1000.0
    if raw_state == "finite_state_rm:failure":
        return -1000.0
    normalized_state = normalize_monitor_state(raw_state)
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
