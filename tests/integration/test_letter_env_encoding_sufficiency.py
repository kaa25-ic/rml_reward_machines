"""Opt-in integration checks for the LetterEnv monitor-driven sufficiency audit."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from rml_rm.verification.encoding_sufficiency import audit_encoding, encoding_key


RUN_INTEGRATION = os.environ.get("RUN_RML_INTEGRATION") == "1"


def _has_swipl() -> bool:
    if shutil.which("swipl") is not None:
        return True
    bundled_candidates = [
        Path("legacy/SWI-Prolog.app/Contents/MacOS/swipl"),
        Path("SWI-Prolog.app/Contents/MacOS/swipl"),
    ]
    return any(path.exists() and os.access(path, os.X_OK) for path in bundled_candidates)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_RML_INTEGRATION=1 to run"),
    pytest.mark.skipif(not _has_swipl(), reason="SWI-Prolog executable not available"),
]


@pytest.fixture(scope="module")
def oracle():
    from envs.letter_env.experiments.monitor_oracle import build_letter_env_monitor_oracle

    return build_letter_env_monitor_oracle(max_n=5)


def _audit(oracle, encoding):
    from envs.letter_env.encodings import build_letter_env_monitor_encoding

    base_encoder, _, _ = build_letter_env_monitor_encoding(encoding)
    return audit_encoding(
        list(oracle.states),
        lambda state: base_encoder(oracle.encoder_input(state)),
        events=oracle.events,
        transition=oracle.transition,
        transition_reward=oracle.transition_reward,
    )


def test_oracle_enumerates_rewarding_reachable_states(oracle) -> None:
    assert len(oracle.states) > 1
    assert {oracle.reward(state) for state in oracle.states} == {-40.0, 0.0, 100.0}


def test_injective_encoding_is_certified_sufficient(oracle) -> None:
    audit = _audit(oracle, "learned_gru")
    assert audit.injective
    assert audit.sufficient is True
    assert audit.witnesses == ()


def test_semantic_progress_is_not_sufficient(oracle) -> None:
    audit = _audit(oracle, "semantic_progress")
    assert audit.sufficient is False
    assert any(witness.kind == "transition" for witness in audit.witnesses)


def test_reported_witnesses_are_genuine(oracle) -> None:
    from envs.letter_env.encodings import build_letter_env_monitor_encoding

    base_encoder, _, _ = build_letter_env_monitor_encoding("semantic_progress")
    audit = _audit(oracle, "semantic_progress")

    def encode(state):
        return encoding_key(base_encoder(oracle.encoder_input(state)))

    for witness in audit.witnesses:
        left, right = witness.states
        assert encode(left) == encode(right)
        if witness.kind == "reward":
            reward_left = oracle.transition_reward(left, witness.event)
            reward_right = oracle.transition_reward(right, witness.event)
            assert reward_left != reward_right
        else:
            successor_left = oracle.transition(left, witness.event)
            successor_right = oracle.transition(right, witness.event)
            assert encode(successor_left) != encode(successor_right)
