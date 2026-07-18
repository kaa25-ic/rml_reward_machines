"""CSTR semantic encoder and generated RML specification tests."""

from __future__ import annotations

import numpy as np
import pytest

from envs.cstr.encodings import CSTRSemanticProgressEncoder
from envs.cstr.rml_generation import render_cstr_config, render_cstr_spec


def test_semantic_progress_encoder_state_order_and_one_hot_vectors() -> None:
    encoder = CSTRSemanticProgressEncoder(max_states=9, soak_steps=3)
    expected_states = [
        "<initial>",
        "preheat",
        "soak_1",
        "soak_2",
        "soak_3",
        "approach",
        "regulate",
        "success",
        "failure",
    ]

    assert encoder.state_count == len(expected_states)
    assert list(encoder.monitor_states.values()) == expected_states

    for state in expected_states:
        vector = encoder.encode(state)
        assert vector.shape == (encoder.state_count,)
        assert vector.dtype == np.float32
        assert vector.sum() == pytest.approx(1.0)
        assert set(vector.tolist()).issubset({0.0, 1.0})
        assert encoder.current_state_name == state


def test_semantic_progress_encoder_rejects_unknown_and_excessive_state_count() -> None:
    encoder = CSTRSemanticProgressEncoder(max_states=9, soak_steps=3)

    with pytest.raises(ValueError, match="Unknown CSTR canonical monitor state"):
        encoder.encode("soak_4")

    with pytest.raises(ValueError, match="monitor_state_limit"):
        CSTRSemanticProgressEncoder(max_states=7, soak_steps=3)


def test_semantic_progress_encoder_covers_classifier_phase_vocabulary() -> None:
    encoder = CSTRSemanticProgressEncoder(max_states=9, soak_steps=3)
    classifier_states = {"preheat", "soak_1", "soak_2", "soak_3", "approach", "regulate", "success", "failure"}

    assert classifier_states.issubset(set(encoder.state_ids))


def test_render_cstr_spec_contains_ordered_soak_chain_and_terminal_verdicts() -> None:
    spec = render_cstr_spec(soak_steps=3, recover_from_regulation_failure=False)

    assert "Main = app(Preheat, [])" in spec
    assert "(in_soak:eps) * app(Soak_1, [])" in spec
    assert "Soak_1 = gen([], (" in spec
    assert "(in_soak:eps) * app(Soak_2, [])" in spec
    assert "Soak_2 = gen([], (" in spec
    assert "(in_soak:eps) * app(Soak_3, [])" in spec
    assert "Soak_3 = gen([], (" in spec
    assert "(in_soak:eps) * app(Approach, [])" in spec
    assert "(stable:eps) * app(Regulate, [])" in spec
    assert "(done_regulated:1)" in spec
    assert "(done_unregulated:0)" in spec
    assert "Soak_4" not in spec


def test_render_cstr_spec_recovery_flag_changes_recovery_branches() -> None:
    strict_spec = render_cstr_spec(soak_steps=2, recover_from_regulation_failure=False)
    recovery_spec = render_cstr_spec(soak_steps=2, recover_from_regulation_failure=True)

    assert "(safe:0)" in strict_spec
    assert "(unsafe:0)" in strict_spec
    assert "(safe:eps) * app(Approach, [])" in recovery_spec
    assert "(unsafe:eps) * app(Approach, [])" in recovery_spec
    assert "(overshoot:eps) * app(Approach, [])" in recovery_spec


def test_render_cstr_config_exposes_expected_info_variables() -> None:
    config = render_cstr_config(env_name="cstr-test", host="127.0.0.1", port=18401, max_episode_steps=300)

    assert "env_name: cstr-test" in config
    assert "host: 127.0.0.1" in config
    assert "port: 18401" in config
    for identifier in (
        "event_temp_critical",
        "event_stable_step",
        "event_temp_safe",
        "event_in_soak_band",
        "event_overshoot",
        "event_past_deadline",
        "event_heating_rate_exceeded",
    ):
        assert f"identifier: {identifier}" in config
