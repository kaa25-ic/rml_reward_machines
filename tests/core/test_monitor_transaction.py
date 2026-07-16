"""Tests for monitor transaction helpers and runtime config rewriting."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from rml_rm.experiments.runtime import write_runtime_monitor_config
from rml_rm.monitors.transaction import (
    monitor_payload_from_observation,
    normalize_verdict,
    rewards_from_config,
    step_monitor,
)


class _StateOwner:
    elapsed_steps = 7


def test_monitor_payload_from_observation_supports_obs_info_and_state() -> None:
    payload = monitor_payload_from_observation(
        variables=[
            {"name": "x", "location": "obs", "identifier": 0},
            {"name": "event", "location": "info", "identifier": "event"},
            {"name": "elapsed", "location": "state", "identifier": "elapsed_steps"},
        ],
        observation={"position": np.asarray([1.5, -2.0], dtype=np.float32)},
        info={"event": 3.0},
        state_owner=_StateOwner(),
    )

    assert payload == {
        "location": "obs",
        "x": 1.5,
        "event": 3.0,
        "elapsed": 7.0,
    }


def test_monitor_payload_requires_state_owner_for_state_variables() -> None:
    with pytest.raises(ValueError, match="state_owner"):
        monitor_payload_from_observation(
            variables=[{"name": "elapsed", "location": "state", "identifier": "elapsed_steps"}],
            observation={"position": np.asarray([0.0], dtype=np.float32)},
        )


def test_rewards_and_step_monitor_normalize_verdicts(fake_monitor_client_factory) -> None:
    rewards = rewards_from_config(
        {
            "reward": {
                "name": "reward",
                "True": 5,
                "False": -5,
                "currently_true": 1,
            }
        }
    )
    client = fake_monitor_client_factory(
        [{"verdict": "True", "monitor_state": "state_123"}]
    )

    result = step_monitor(client, {"event": 1.0}, rewards)

    assert normalize_verdict("False") == "false"
    assert rewards == {"true": 5.0, "false": -5.0, "currently_true": 1.0}
    assert result.verdict == "true"
    assert result.monitor_state == "state_123"
    assert result.base_reward == 5.0


def test_write_runtime_monitor_config_sets_port_host_and_max_steps(tmp_path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "variables": [{"name": "event", "location": "info", "identifier": "event"}],
                "reward": {"name": "reward", "true": 1, "false": -1},
                "host": "0.0.0.0",
                "port": 1234,
                "max_episode_steps": 10,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output = write_runtime_monitor_config(
        tmp_path / "runtime.yaml",
        template_path=template,
        port=9876,
        max_episode_steps=42,
    )

    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 9876
    assert config["max_episode_steps"] == 42
    assert config["variables"] == [
        {"name": "event", "location": "info", "identifier": "event"}
    ]
