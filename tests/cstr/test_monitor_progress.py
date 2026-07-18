"""CSTR monitor phase decoding and soak-counting contracts."""

from __future__ import annotations

import pytest

from envs.cstr.builder import (
    _canonical_monitor_state,
    _classify_cstr_rml_state,
    _next_soak_steps,
)


def payload(
    *,
    stable: bool = False,
    in_soak: bool = False,
    temp_safe: bool = True,
    critical: bool = False,
    overshoot: bool = False,
) -> dict[str, float]:
    return {
        "stable": float(stable),
        "in_soak_band": float(in_soak),
        "temp_safe": float(temp_safe),
        "critical": float(critical),
        "overshoot": float(overshoot),
    }


@pytest.mark.parametrize(
    ("monitor_state", "verdict", "expected"),
    [
        ("1", "currently_true", "success"),
        ("anything", "true", "success"),
        ("0", "currently_false", "failure"),
        ("false_verdict", "currently_false", "failure"),
        ("anything", "false", "failure"),
    ],
)
def test_classifier_terminal_monitor_tokens(monitor_state: str, verdict: str, expected: str) -> None:
    assert (
        _classify_cstr_rml_state(
            monitor_state,
            verdict=verdict,
            payload=payload(),
            previous_phase="preheat",
            previous_soak_steps=0,
            soak_steps=3,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("previous_phase", "previous_soak_steps", "event_payload", "expected"),
    [
        ("preheat", 0, payload(in_soak=False), "preheat"),
        ("preheat", 0, payload(in_soak=True), "soak"),
        ("soak", 1, payload(in_soak=True), "soak"),
        ("soak", 3, payload(in_soak=True), "approach"),
        ("soak", 2, payload(in_soak=False, temp_safe=True), "preheat"),
        ("soak", 2, payload(in_soak=False, temp_safe=False), "failure"),
        ("approach", 0, payload(stable=False), "approach"),
        ("approach", 0, payload(stable=True), "regulate"),
        ("regulate", 0, payload(stable=True), "regulate"),
        ("regulate", 0, payload(stable=False, temp_safe=False), "regulate"),
    ],
)
def test_classifier_phase_transitions(
    previous_phase: str,
    previous_soak_steps: int,
    event_payload: dict[str, float],
    expected: str,
) -> None:
    assert (
        _classify_cstr_rml_state(
            "@(app(...), [])",
            verdict="currently_false",
            payload=event_payload,
            previous_phase=previous_phase,
            previous_soak_steps=previous_soak_steps,
            soak_steps=3,
        )
        == expected
    )


def test_classifier_recovery_from_regulation_failure_returns_to_approach() -> None:
    assert (
        _classify_cstr_rml_state(
            "@(app(...), [])",
            verdict="currently_false",
            payload=payload(stable=False, temp_safe=True),
            previous_phase="regulate",
            previous_soak_steps=0,
            soak_steps=3,
            recover_from_regulation_failure=True,
        )
        == "approach"
    )


def test_next_soak_steps_counts_until_k_then_resets_outside_soak() -> None:
    assert _next_soak_steps(
        phase="soak",
        payload=payload(in_soak=True),
        previous_phase="preheat",
        previous_soak_steps=0,
        soak_steps=3,
    ) == 1
    assert _next_soak_steps(
        phase="soak",
        payload=payload(in_soak=True),
        previous_phase="soak",
        previous_soak_steps=1,
        soak_steps=3,
    ) == 2
    assert _next_soak_steps(
        phase="soak",
        payload=payload(in_soak=True),
        previous_phase="soak",
        previous_soak_steps=3,
        soak_steps=3,
    ) == 3
    for phase in ("preheat", "approach", "regulate", "success", "failure"):
        assert _next_soak_steps(
            phase=phase,
            payload=payload(),
            previous_phase="soak",
            previous_soak_steps=2,
            soak_steps=3,
        ) == 0


def test_canonical_monitor_state_uses_soak_index_only_for_soak_phase() -> None:
    assert _canonical_monitor_state(phase="soak", soak_steps=2) == "soak_2"
    assert _canonical_monitor_state(phase="soak", soak_steps=0) == "soak_1"
    assert _canonical_monitor_state(phase="preheat", soak_steps=2) == "preheat"
    assert _canonical_monitor_state(phase="approach", soak_steps=2) == "approach"
