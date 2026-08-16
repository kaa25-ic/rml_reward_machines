"""Ground-truth tests for the encoding-sufficiency audit engine."""

from __future__ import annotations

import numpy as np

from rml_rm.verification.encoding_sufficiency import Witness, audit_encoding


STATES = ("p", "q", "r", "t")


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _encoder(mapping: dict[str, np.ndarray]):
    return mapping.__getitem__


def test_injective_encoding_is_sufficient() -> None:
    encoder = _encoder({"p": _vec(0), "q": _vec(1), "r": _vec(2), "t": _vec(3)})
    audit = audit_encoding(STATES, encoder)
    assert audit.injective
    assert audit.sufficient is True
    assert audit.witnesses == ()


def test_collision_is_inconclusive_without_oracles() -> None:
    encoder = _encoder({"p": _vec(0), "q": _vec(0), "r": _vec(1), "t": _vec(2)})
    audit = audit_encoding(STATES, encoder)
    assert not audit.injective
    assert audit.sufficient is None
    assert audit.checked is False


def test_sound_merge_is_certified_with_oracles() -> None:
    encoder = _encoder({"p": _vec(0), "q": _vec(0), "r": _vec(1), "t": _vec(2)})
    transitions = {("p", "e"): "r", ("q", "e"): "r"}
    transition_rewards = {("p", "e"): 1.0, ("q", "e"): 1.0}
    audit = audit_encoding(
        STATES,
        encoder,
        events=("e",),
        transition=lambda state, event: transitions.get((state, event)),
        transition_reward=lambda state, event: transition_rewards.get((state, event)),
    )
    assert not audit.injective
    assert audit.sufficient is True
    assert audit.witnesses == ()


def test_unsound_transition_merge_reports_witness() -> None:
    encoder = _encoder({"p": _vec(0), "q": _vec(0), "r": _vec(1), "t": _vec(2)})
    transitions = {("p", "e"): "r", ("q", "e"): "t"}
    transition_rewards = {("p", "e"): 1.0, ("q", "e"): 1.0}
    audit = audit_encoding(
        STATES,
        encoder,
        events=("e",),
        transition=lambda state, event: transitions.get((state, event)),
        transition_reward=lambda state, event: transition_rewards.get((state, event)),
    )
    assert audit.sufficient is False
    assert Witness("transition", ("p", "q"), "e") in audit.witnesses


def test_reward_violating_merge_reports_witness() -> None:
    # p and q are merged and reach the same-encoded successor under event e, but
    # firing e emits different rewards from p and from q.
    encoder = _encoder({"p": _vec(0), "q": _vec(0), "r": _vec(1), "t": _vec(1)})
    transitions = {("p", "e"): "r", ("q", "e"): "t"}
    transition_rewards = {("p", "e"): 0.0, ("q", "e"): 100.0}
    audit = audit_encoding(
        STATES,
        encoder,
        events=("e",),
        transition=lambda state, event: transitions.get((state, event)),
        transition_reward=lambda state, event: transition_rewards.get((state, event)),
    )
    assert audit.sufficient is False
    assert Witness("reward", ("p", "q"), "e") in audit.witnesses
    assert not any(witness.kind == "transition" for witness in audit.witnesses)
