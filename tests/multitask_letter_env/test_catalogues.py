"""Catalogue consistency tests for multi-task LetterEnv."""

from __future__ import annotations

import numpy as np
import pytest

from envs.multitask_letter_env.encodings import (
    build_multitask_monitor_encoding,
    load_monitor_progress_catalogue,
    load_monitor_state_catalogue,
)
from envs.multitask_letter_env.tasks import get_task_suite


@pytest.mark.parametrize("encoding", ("one_hot", "numerical"))
def test_all_multitask_catalogue_states_encode(encoding: str) -> None:
    states_by_task = load_monitor_state_catalogue()
    encoder, reset_vector, vector_size = build_multitask_monitor_encoding(encoding)

    reset_vector = np.asarray(reset_vector, dtype=np.float32)
    assert reset_vector.shape == (vector_size,)

    for task_key, states in states_by_task.items():
        assert states, task_key
        for monitor_state in states:
            vector = encoder(monitor_state)
            assert vector.shape == (vector_size,), (task_key, monitor_state)
            assert np.isfinite(vector).all(), (task_key, monitor_state)
            assert vector.sum() > 0, (task_key, monitor_state)


def test_progress_catalogue_matches_task_suite_and_state_catalogue() -> None:
    task_keys = {task.key for task in get_task_suite()}
    states_by_task = load_monitor_state_catalogue()
    progress_by_task = load_monitor_progress_catalogue()

    assert set(states_by_task) == task_keys
    assert set(progress_by_task) == task_keys

    for task_key, progress_by_n in progress_by_task.items():
        known_states = set(states_by_task[task_key])
        assert set(progress_by_n) == set(range(1, 6))
        for n_value, progress_by_state in progress_by_n.items():
            assert progress_by_state, (task_key, n_value)
            assert set(progress_by_state).issubset(known_states), (task_key, n_value)
            assert progress_by_state["false_verdict"] == -1
            assert all(
                progress >= 0
                for state, progress in progress_by_state.items()
                if state != "false_verdict"
            )
