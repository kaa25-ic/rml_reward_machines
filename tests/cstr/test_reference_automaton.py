"""Reference and manual reward-machine protocol tests for CSTR."""

from __future__ import annotations

from envs.cstr.manual_rm import _phase_and_soak_steps, _verdict_to_monitor_label
from envs.cstr.reference_automaton import ReferenceStartupAutomaton, event_label, verdict_matches_monitor


def event(
    *,
    temp_safe: bool = True,
    in_soak: bool = False,
    stable: bool = False,
    terminate: bool = False,
    critical: bool = False,
    overshoot: bool = False,
    past_deadline: bool = False,
) -> dict[str, float | bool]:
    return {
        "temp_safe": float(temp_safe),
        "in_soak_band": float(in_soak),
        "stable": float(stable),
        "terminate": bool(terminate),
        "critical": float(critical),
        "overshoot": float(overshoot),
        "past_deadline": float(past_deadline),
    }


def test_event_label_priority_order() -> None:
    assert event_label(event(critical=True, stable=True, terminate=True)) == "critical"
    assert event_label(event(temp_safe=False, stable=True, terminate=True)) == "unsafe"
    assert event_label(event(stable=True, terminate=True)) == "done_regulated"
    assert event_label(event(terminate=True)) == "done_unregulated"
    assert event_label(event(past_deadline=True, in_soak=True)) == "deadline"
    assert event_label(event(overshoot=True, in_soak=True)) == "overshoot"
    assert event_label(event(stable=True, in_soak=True)) == "stable"
    assert event_label(event(in_soak=True)) == "in_soak"
    assert event_label(event()) == "safe"
    assert event_label(event(temp_safe=False, critical=False)) == "unsafe"


def test_reference_automaton_counts_soak_then_accepts_regulated_episode() -> None:
    automaton = ReferenceStartupAutomaton(soak_steps=3, recover_from_regulation_failure=False)

    assert automaton.canonical_state == "preheat"
    assert automaton.step(event(in_soak=True)) == "run"
    assert automaton.canonical_state == "soak_1"
    assert automaton.step(event(in_soak=True)) == "run"
    assert automaton.canonical_state == "soak_2"
    assert automaton.step(event(in_soak=True)) == "run"
    assert automaton.canonical_state == "soak_3"
    assert automaton.step(event(in_soak=True)) == "run"
    assert automaton.canonical_state == "approach"
    assert automaton.step(event(stable=True)) == "run"
    assert automaton.canonical_state == "regulate"
    assert automaton.step(event(stable=True, terminate=True)) == "accept"
    assert automaton.canonical_state == "success"


def test_reference_automaton_rejects_early_stability_before_soak_count() -> None:
    automaton = ReferenceStartupAutomaton(soak_steps=3, recover_from_regulation_failure=False)

    assert automaton.step(event(in_soak=True)) == "run"
    assert automaton.canonical_state == "soak_1"
    assert automaton.step(event(stable=True)) == "reject"
    assert automaton.canonical_state == "failure"


def test_reference_automaton_recovery_flag_controls_regulation_failure_behavior() -> None:
    strict = ReferenceStartupAutomaton(soak_steps=1, recover_from_regulation_failure=False)
    recover = ReferenceStartupAutomaton(soak_steps=1, recover_from_regulation_failure=True)

    for automaton in (strict, recover):
        automaton.step(event(in_soak=True))
        automaton.step(event(in_soak=True))
        automaton.step(event(stable=True))
        assert automaton.canonical_state == "regulate"

    assert strict.step(event(temp_safe=True, stable=False)) == "reject"
    assert strict.canonical_state == "failure"
    assert recover.step(event(temp_safe=True, stable=False)) == "run"
    assert recover.canonical_state == "approach"


def test_manual_phase_and_verdict_labels_match_reference_states() -> None:
    automaton = ReferenceStartupAutomaton(soak_steps=2, recover_from_regulation_failure=False)
    expected = [
        ("soak_1", "soak", 1, "currently_false"),
        ("soak_2", "soak", 2, "currently_false"),
        ("approach", "approach", 0, "currently_false"),
        ("regulate", "regulate", 0, "currently_false"),
        ("success", "success", 0, "true"),
    ]
    sequence = [
        event(in_soak=True),
        event(in_soak=True),
        event(in_soak=True),
        event(stable=True),
        event(stable=True, terminate=True),
    ]

    for payload, (canonical, phase, soak_steps, monitor_label) in zip(sequence, expected, strict=True):
        verdict = automaton.step(payload)
        assert automaton.canonical_state == canonical
        assert _phase_and_soak_steps(canonical) == (phase, soak_steps)
        assert _verdict_to_monitor_label(verdict) == monitor_label


def test_verdict_matches_monitor_terminal_and_running_cases() -> None:
    assert verdict_matches_monitor("accept", "currently_true", "1")
    assert verdict_matches_monitor("reject", "currently_false", "0")
    assert verdict_matches_monitor("reject", "false", "@(...)")
    assert verdict_matches_monitor("run", "currently_false", "@(...)")
    assert not verdict_matches_monitor("run", "currently_false", "1")
