"""Shared-core RMLMonitorWrapper contract tests using a fake monitor client."""

from __future__ import annotations

import numpy as np

from rml_rm.wrappers.rml_monitor import RMLMonitorWrapper


def test_rml_monitor_wrapper_adds_monitor_observation_and_terminal_reward(
    fake_monitor_client_factory,
    scripted_monitor_env_factory,
    tmp_path,
) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        """
variables:
  - name: event
    location: info
    identifier: event
reward:
  name: reward
  partial: 0
  true: 1
  false: -1
host: 127.0.0.1
port: 0
max_episode_steps: 5
""".strip(),
        encoding="utf-8",
    )
    client = fake_monitor_client_factory(
        [
            {"verdict": "true", "monitor_state": "1"},
        ]
    )
    env = RMLMonitorWrapper(
        scripted_monitor_env_factory([{"reward": 7.0, "info": {"event": 1.0}}]),
        config_path=config_path,
        client=client,
        include_transition_bonus=False,
    )

    observation, _ = env.reset()
    assert "monitor" in observation

    observation, reward, terminated, truncated, info = env.step(0)

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert np.asarray(observation["monitor"]).shape == (1,)
    assert info["base_reward"] == 7.0
    assert info["monitor_verdict"] == "true"
    assert info["monitor_state_unencoded"] == "1"
    assert info["monitor_reward"] == 1.0
    assert client.payloads[-1]["event"] == 1.0
