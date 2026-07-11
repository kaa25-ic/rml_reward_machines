"""Shared request/response helpers for RML monitor clients."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import websocket
import yaml


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


@dataclass(frozen=True)
class MonitorStepResult:
    """Normalized response from one RML monitor step."""

    verdict: str
    monitor_state: str
    base_reward: float


def load_monitor_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate an RML monitor YAML configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"RML config {path} did not contain a mapping.")
    for key in ("variables", "reward", "host", "port"):
        if key not in config:
            raise ValueError(f"RML config {path} is missing {key!r}.")
    return config


def normalize_monitor_state(monitor_state: str) -> str:
    """Normalize monitor states by removing generated variable suffixes."""
    return re.sub(r"_[0-9]+", "", monitor_state)


def normalize_verdict(verdict: str) -> str:
    """Normalize monitor verdict labels to YAML reward keys."""
    if verdict in {"True", "False"}:
        return verdict.lower()
    return verdict


def rewards_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    """Return normalized verdict-to-reward mapping from a monitor config."""
    rewards = dict(config["reward"])
    rewards.pop("name", None)
    return {normalize_verdict(str(key)): float(value) for key, value in rewards.items()}


def empty_monitor_payload(variables: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an empty payload with the variables expected by a monitor."""
    payload: dict[str, Any] = {"time": [], "action": []}
    for variable in variables:
        payload[str(variable["name"])] = []
    return payload


def monitor_payload_from_observation(
    *,
    variables: list[Mapping[str, Any]],
    observation: Mapping[str, Any],
    info: Mapping[str, Any] | None = None,
    state_owner: Any | None = None,
) -> dict[str, Any]:
    """Build a monitor payload from observation/info/state variable specs."""
    info = info or {}
    payload: dict[str, Any] = {}
    for variable in variables:
        location = str(variable["location"])
        name = str(variable["name"])
        identifier = variable["identifier"]
        if location == "obs":
            payload["location"] = location
            payload[name] = float(np.asarray(observation["position"])[int(identifier)])
        elif location == "info":
            payload[name] = float(info[identifier])
        elif location == "state":
            if state_owner is None:
                raise ValueError("A state_owner is required for state monitor variables.")
            payload[name] = float(getattr(state_owner, str(identifier)))
        else:
            raise ValueError(f"Unsupported RML variable location: {location!r}.")
    return payload


def reset_monitor(client: MonitorClient, variables: list[Mapping[str, Any]]) -> None:
    """Reset an online RML monitor."""
    payload = empty_monitor_payload(variables)
    payload["terminate"] = True
    client.send(payload)


def step_monitor(
    client: MonitorClient,
    payload: Mapping[str, Any],
    rewards: Mapping[str, float],
) -> MonitorStepResult:
    """Send one monitor payload and normalize the verdict, state, and reward."""
    response = client.send(payload)
    verdict = normalize_verdict(str(response["verdict"]))
    monitor_state = str(response["monitor_state"])
    return MonitorStepResult(
        verdict=verdict,
        monitor_state=monitor_state,
        base_reward=float(rewards[verdict]),
    )
