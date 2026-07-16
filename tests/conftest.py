"""Shared pytest fixtures and small test doubles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


class FakeMonitorClient:
    """Scripted monitor client for wrapper contract tests."""

    def __init__(self, responses: Iterable[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        payload_dict = dict(payload)
        self.payloads.append(payload_dict)
        if payload_dict.get("terminate") is True and not payload_dict.get("action"):
            return {}
        if not self.responses:
            raise AssertionError("FakeMonitorClient received more step calls than expected.")
        return dict(self.responses.pop(0))


class ScriptedMonitorEnv(gym.Env):
    """Minimal env that emits scripted rewards and info dictionaries."""

    metadata: dict[str, Any] = {}

    def __init__(self, steps: Iterable[Mapping[str, Any]]) -> None:
        self.steps = list(steps)
        self.step_index = 0
        self.observation_space = spaces.Dict(
            {"position": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)}
        )
        self.action_space = spaces.Discrete(1)

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get("seed"))
        self.step_index = 0
        return {"position": np.zeros(1, dtype=np.float32)}, {}

    def step(self, action):
        if self.step_index >= len(self.steps):
            raise AssertionError("ScriptedMonitorEnv received more steps than expected.")
        step = dict(self.steps[self.step_index])
        self.step_index += 1
        return (
            step.get("observation", {"position": np.zeros(1, dtype=np.float32)}),
            float(step.get("reward", 0.0)),
            bool(step.get("terminated", False)),
            bool(step.get("truncated", False)),
            dict(step.get("info", {})),
        )


@pytest.fixture
def fake_monitor_client_factory():
    return FakeMonitorClient


@pytest.fixture
def scripted_monitor_env_factory():
    return ScriptedMonitorEnv
