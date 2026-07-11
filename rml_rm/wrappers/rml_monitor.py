"""Gym wrapper for RML monitor integration."""

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rml_rm.monitors.transaction import (
    MonitorClient,
    WebSocketMonitorClient,
    empty_monitor_payload,
    load_monitor_config,
    monitor_payload_from_observation,
    normalize_monitor_state,
    reset_monitor,
    rewards_from_config,
    step_monitor,
)


MonitorEncoder = Callable[[str], int | np.ndarray]


class SimpleMonitorStateEncoder:
    """Assign integer IDs to monitor-state strings as they appear."""

    def __init__(self) -> None:
        self.monitor_states: dict[int, str] = {}

    def __call__(self, monitor_state: str) -> int:
        normalized_state = normalize_monitor_state(monitor_state)
        for state_id, known_state in self.monitor_states.items():
            if known_state == normalized_state:
                return state_id

        state_id = max(self.monitor_states.keys(), default=-1) + 1
        self.monitor_states[state_id] = normalized_state
        return state_id


class RMLMonitorWrapper(gym.Wrapper):
    """Attach RML monitor state and reward to encoded environment observations."""

    terminal_monitor_states = {"1", "false_verdict"}

    def __init__(
        self,
        env: gym.Env,
        *,
        config_path: str | Path,
        monitor_encoder: MonitorEncoder | None = None,
        initial_monitor_state: int | np.ndarray = 0,
        monitor_space: spaces.Space | None = None,
        client: MonitorClient | None = None,
        transition_bonus: float = 10.0,
        include_transition_bonus: bool = True,
    ) -> None:
        super().__init__(env)
        self.config_path = Path(config_path)
        self.config = load_monitor_config(self.config_path)
        self.rml_variables = list(self.config["variables"])
        self.rewards = rewards_from_config(self.config)
        self.max_steps = int(self.config.get("max_episode_steps", 200))
        self.client = client or WebSocketMonitorClient(
            host=str(self.config["host"]),
            port=int(self.config["port"]),
        )
        self.monitor_encoder = monitor_encoder or SimpleMonitorStateEncoder()
        self.initial_monitor_state = encode_monitor_value(initial_monitor_state)
        self.previous_monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state_unencoded = ""
        self.transition_bonus = float(transition_bonus)
        self.include_transition_bonus = bool(include_transition_bonus)
        self.step_num = 0
        self.total_timesteps = 0
        self.data = self._empty_data()

        self.observation_space = self._build_observation_space(monitor_space)

    def reset(self, **kwargs):
        self.step_num = 0
        self.previous_monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state_unencoded = ""
        self.data = self._empty_data()
        self.reset_monitor()

        observation, info = self.env.reset(**kwargs)
        return self._with_monitor(observation), info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_num += 1
        self.total_timesteps += 1
        if self.step_num >= self.max_steps and not terminated:
            truncated = True

        self.data = self._payload_from_observation(observation, info)
        self.data["terminate"] = bool(truncated)
        monitor_reward, monitor_info = self.monitor_reward()

        if normalize_monitor_state(self.monitor_state_unencoded) in self.terminal_monitor_states:
            terminated = True

        info = dict(info)
        info["base_reward"] = base_reward
        info.update(monitor_info)
        return self._with_monitor(observation), monitor_reward, terminated, truncated, info

    def reset_monitor(self) -> None:
        reset_monitor(self.client, self.rml_variables)

    def monitor_reward(self) -> tuple[float, dict[str, Any]]:
        result = step_monitor(self.client, self.data, self.rewards)
        verdict = result.verdict
        monitor_state_unencoded = result.monitor_state
        monitor_reward = result.base_reward

        encoded_monitor_state = encode_monitor_value(self.monitor_encoder(monitor_state_unencoded))
        transition_bonus = 0.0
        if (
            self.include_transition_bonus
            and normalize_monitor_state(monitor_state_unencoded) != "false_verdict"
            and not monitor_values_equal(encoded_monitor_state, self.previous_monitor_state)
        ):
            transition_bonus = self.transition_bonus

        self.monitor_state_unencoded = monitor_state_unencoded
        self.monitor_state = encoded_monitor_state
        self.previous_monitor_state = copy.deepcopy(encoded_monitor_state)

        return (
            monitor_reward + transition_bonus,
            {
                "monitor_verdict": verdict,
                "monitor_state_unencoded": monitor_state_unencoded,
                "monitor_transition_bonus": transition_bonus,
                "monitor_reward": monitor_reward,
            },
        )

    def _payload_from_observation(
        self,
        observation: dict[str, Any],
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        return monitor_payload_from_observation(
            variables=self.rml_variables,
            observation=observation,
            info=info,
            state_owner=self,
        )

    def _with_monitor(self, observation: dict[str, Any]) -> dict[str, Any]:
        wrapped = dict(observation)
        wrapped["monitor"] = self.monitor_state
        return wrapped

    def _build_observation_space(self, monitor_space: spaces.Space | None) -> spaces.Dict:
        if not isinstance(self.env.observation_space, spaces.Dict):
            raise TypeError("RMLMonitorWrapper expects a Dict observation space.")
        spaces_dict = dict(self.env.observation_space.spaces)
        spaces_dict["monitor"] = monitor_space or infer_monitor_space(self.initial_monitor_state)
        return spaces.Dict(spaces_dict)

    def _empty_data(self) -> dict[str, Any]:
        return empty_monitor_payload(self.rml_variables)


def encode_monitor_value(value: int | np.ndarray) -> np.ndarray:
    """Normalize monitor encodings for observations."""
    if isinstance(value, (int, np.integer)):
        return np.asarray([int(value)], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def infer_monitor_space(value: int | np.ndarray) -> spaces.Box:
    encoded = encode_monitor_value(value)
    return spaces.Box(low=-np.inf, high=np.inf, shape=encoded.shape, dtype=np.float32)


def monitor_values_equal(left: int | np.ndarray, right: int | np.ndarray) -> bool:
    return np.array_equal(
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
    )
