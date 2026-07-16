"""Runtime-surface tests for multi-task LetterEnv using fake monitor clients."""

from __future__ import annotations

import pytest

from envs.letter_env_core import LetterAction
from envs.multitask_letter_env.env import MultiTaskLetterEnv, MultiTaskLetterEnvConfig
from envs.multitask_letter_env.tasks import get_task_suite


def _fake_clients(fake_monitor_client_factory, selected_task_id: int):
    clients = {}
    for task in get_task_suite():
        responses = (
            [{"verdict": "true", "monitor_state": "1"}]
            if task.task_id == selected_task_id
            else []
        )
        clients[task.task_id] = fake_monitor_client_factory(responses)
    return clients


def test_task_conditioned_observation_encodes_selected_task_and_n(
    fake_monitor_client_factory,
) -> None:
    selected_task_id = 3
    config = MultiTaskLetterEnvConfig(max_n=5)
    env = MultiTaskLetterEnv(
        config=config,
        clients=_fake_clients(fake_monitor_client_factory, selected_task_id),
    )

    observation, info = env.reset(seed=0, options={"task_id": selected_task_id, "n": 4})

    task_feature_size = len(env.tasks)
    n_feature = observation["position"][-task_feature_size - 1]
    task_features = observation["position"][-task_feature_size:]

    assert info["task_id"] == selected_task_id
    assert info["n"] == 4
    assert n_feature == pytest.approx(4 / config.max_n)
    assert task_features.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_step_uses_only_the_selected_task_monitor_client(
    fake_monitor_client_factory,
) -> None:
    selected_task_id = 2
    clients = _fake_clients(fake_monitor_client_factory, selected_task_id)
    env = MultiTaskLetterEnv(clients=clients)
    env.reset(seed=0, options={"task_id": selected_task_id, "n": 1})

    _, reward, terminated, truncated, info = env.step(LetterAction.UP)

    assert reward == 100.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["task_id"] == selected_task_id
    assert info["monitor_state_normalized"] == "1"
    assert len(clients[selected_task_id].payloads) == 2
    assert clients[selected_task_id].payloads[0]["terminate"] is True
    assert clients[selected_task_id].payloads[1]["terminate"] is False

    for task_id, client in clients.items():
        if task_id != selected_task_id:
            assert client.payloads == []
