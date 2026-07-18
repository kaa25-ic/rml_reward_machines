"""Pure-Python reference automaton for the CSTR startup RML specification.

This is an *independent* re-implementation of the trace expression produced by
``render_cstr_spec`` (see ``rml_generation.py``). It is parameterised by the same
``soak_steps`` and ``recover_from_regulation_failure`` flags, computes its
accept/reject verdict **only from the event payload** (it never reads the RML
monitor's verdict), and is therefore usable to *validate* the external RML
monitor: if this automaton's accept/reject decision matches the monitor's verdict
at every step over many episodes, the Python decoding of the task is provably
equivalent to the RML specification it claims to implement.

It is deliberately tiny and side-effect free so it can run in-process next to the
monitor with negligible cost.
"""

from __future__ import annotations

from typing import Any, Mapping

# match/2 priority order in the generated spec (highest first).
EVENT_PRIORITY = (
    "critical", "unsafe", "done_regulated", "done_unregulated",
    "deadline", "overshoot", "stable", "in_soak", "safe", "other",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def event_label(payload: Mapping[str, Any]) -> str:
    """Reduce one event payload to a single spec event label (priority order)."""
    if _truthy(payload.get("critical", 0)):
        return "critical"
    temp_safe = _truthy(payload.get("temp_safe", 0))
    if not temp_safe:
        return "unsafe"
    terminate = _truthy(payload.get("terminate", False))
    stable = _truthy(payload.get("stable", 0))
    if terminate and stable:
        return "done_regulated"
    if terminate:
        return "done_unregulated"
    if _truthy(payload.get("past_deadline", 0)):
        return "deadline"
    if _truthy(payload.get("overshoot", 0)):
        return "overshoot"
    if stable:
        return "stable"
    if _truthy(payload.get("in_soak_band", 0)):
        return "in_soak"
    if temp_safe:
        return "safe"
    return "other"


_ACCEPT = ("__accept__", "accept")
_REJECT = ("__reject__", "reject")


class ReferenceStartupAutomaton:
    """Reference implementation of the CSTR startup trace expression."""

    def __init__(self, *, soak_steps: int, recover_from_regulation_failure: bool) -> None:
        self.soak_steps = int(soak_steps)
        self.recover = bool(recover_from_regulation_failure)
        self.reset()

    def reset(self) -> None:
        self.state = "Preheat"
        self.verdict = "run"  # one of {"run", "accept", "reject"}

    def step(self, payload: Mapping[str, Any]) -> str:
        """Advance on one event payload; return the verdict {run,accept,reject}."""
        if self.verdict in ("accept", "reject"):
            return self.verdict
        self.state, self.verdict = self._transition(self.state, event_label(payload))
        return self.verdict

    @property
    def canonical_state(self) -> str:
        if self.verdict == "accept":
            return "success"
        if self.verdict == "reject":
            return "failure"
        return self.state.lower()  # preheat / soak_<i> / approach / regulate

    def _recover_or_reject(self) -> tuple[str, str]:
        return ("Approach", "run") if self.recover else _REJECT

    def _transition(self, state: str, ev: str) -> tuple[str, str]:
        K = self.soak_steps
        if state == "Preheat":
            if ev == "in_soak":
                return ("Soak_1", "run")
            if ev == "safe":
                return ("Preheat", "run")
            return _REJECT
        if state.startswith("Soak_"):
            i = int(state.split("_")[1])
            if ev == "in_soak":
                return ("Approach", "run") if i >= K else (f"Soak_{i + 1}", "run")
            if ev == "stable":
                return ("Regulate", "run") if i >= K else _REJECT
            if ev == "safe":
                return ("Preheat", "run")
            return _REJECT
        if state == "Approach":
            if ev == "stable":
                return ("Regulate", "run")
            if ev in ("in_soak", "safe"):
                return ("Approach", "run")
            if ev in ("unsafe", "overshoot"):
                return self._recover_or_reject()
            return _REJECT
        if state == "Regulate":
            if ev == "done_regulated":
                return _ACCEPT
            if ev in ("deadline", "stable", "in_soak"):
                return ("Regulate", "run")
            if ev in ("unsafe", "overshoot", "safe"):
                return self._recover_or_reject()
            return _REJECT
        return _REJECT


def verdict_matches_monitor(
    reference_verdict: str, monitor_verdict: str, monitor_state: str = ""
) -> bool:
    """True iff the reference {run,accept,reject} agrees with the monitor.

    The RML monitor signals a *hard* accept/reject through ``monitor_state`` going
    to ``'1'``/``'0'`` while its ``verdict`` field may stay ``currently_true`` /
    ``currently_false``. We therefore treat ``monitor_state`` as authoritative when
    it has collapsed to a terminal token, and fall back to the verdict otherwise.
    """
    monitor = str(monitor_verdict).strip().lower()
    state = str(monitor_state).strip()
    accepted = monitor in {"true", "1"} or state == "1"
    rejected = monitor in {"false", "0", "false_verdict"} or state == "0"
    if reference_verdict == "accept":
        return accepted
    if reference_verdict == "reject":
        return rejected
    # "run" -> the monitor must still be undecided (neither terminal token set)
    return not accepted and not rejected
