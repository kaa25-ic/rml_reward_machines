"""Gym wrapper for RML monitor integration."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import websocket
import yaml
from gymnasium import spaces


MonitorEncoder = Callable[[str], int | np.ndarray]


class MonitorClient(Protocol):
    """Protocol for monitor transports."""

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one payload and return the decoded monitor response."""


@dataclass(frozen=True)
class WebSocketMonitorClient:
    """WebSocket client for the RML online monitor."""

    host: str
    port: int
    timeout: float | None = None

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        socket = websocket.WebSocket(timeout=self.timeout)
        try:
            socket.connect(f"ws://{self.host}:{self.port}")
            socket.send(json.dumps(dict(payload)))
            response = socket.recv()
        finally:
            socket.close()
        if not str(response).strip():
            return {}
        return json.loads(response)


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
        self.config = self._load_config(self.config_path)
        self.rml_variables = list(self.config["variables"])
        self.rewards = {
            normalize_verdict(str(key)): value for key, value in dict(self.config["reward"]).items()
        }
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
        payload = self._empty_data()
        payload["terminate"] = True
        self.client.send(payload)

    def monitor_reward(self) -> tuple[float, dict[str, Any]]:
        response = self.client.send(self.data)
        verdict = normalize_verdict(str(response["verdict"]))
        monitor_state_unencoded = str(response["monitor_state"])
        monitor_reward = float(self.rewards[verdict])

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
        payload: dict[str, Any] = {}
        for variable in self.rml_variables:
            location = variable["location"]
            name = variable["name"]
            identifier = variable["identifier"]
            if location == "obs":
                payload["location"] = location
                payload[name] = float(np.asarray(observation["position"])[int(identifier)])
            elif location == "info":
                payload[name] = float(info[identifier])
            elif location == "state":
                payload[name] = float(getattr(self, identifier))
            else:
                raise ValueError(f"Unsupported RML variable location: {location!r}.")
        return payload

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

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"RML config {config_path} did not contain a mapping.")
        for key in ("variables", "reward", "host", "port"):
            if key not in config:
                raise ValueError(f"RML config {config_path} is missing {key!r}.")
        return config

    def _empty_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"time": [], "action": []}
        for variable in self.rml_variables:
            data[variable["name"]] = []
        return data


def normalize_monitor_state(monitor_state: str) -> str:
    """Normalize monitor states by removing generated variable suffixes."""
    return re.sub(r"_[0-9]+", "", monitor_state)


def normalize_verdict(verdict: str) -> str:
    """Normalize monitor verdict labels to YAML reward keys."""
    if verdict in {"True", "False"}:
        return verdict.lower()
    return verdict


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
