"""Catalogue and semantic encoder tests for single-task LetterEnv."""

from __future__ import annotations

import numpy as np
import pytest

from envs.letter_env.encodings import (
    build_letter_env_monitor_encoding,
    build_letter_env_semantic_progress_encoder,
    load_letter_env_monitor_state_catalogue,
)


@pytest.mark.parametrize("encoding", ("one_hot", "numerical"))
def test_catalogue_states_encode_without_fallback_or_error(encoding: str) -> None:
    states_by_id = load_letter_env_monitor_state_catalogue()
    encoder, initial_state, _ = build_letter_env_monitor_encoding(encoding)

    initial_vector = np.asarray(initial_state, dtype=np.float32)
    assert initial_vector.ndim == 1
    assert initial_vector.size > 0

    for state_id, monitor_state in states_by_id.items():
        vector = encoder(monitor_state)
        assert vector.shape == initial_vector.shape, state_id
        assert np.isfinite(vector).all(), state_id
        assert vector.sum() > 0, state_id


def test_semantic_progress_encoder_covers_every_catalogue_state_once() -> None:
    states_by_id = load_letter_env_monitor_state_catalogue()
    encoder = build_letter_env_semantic_progress_encoder(states_by_id)

    for state_id, monitor_state in states_by_id.items():
        vector = encoder(monitor_state)
        assert vector.shape == (encoder.vector_length,), state_id
        assert vector.sum() == pytest.approx(1.0), state_id
        assert set(vector).issubset({0.0, 1.0}), state_id
