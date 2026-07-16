"""Monitor-state encodings for the LunarLander protocol."""

from __future__ import annotations

import numpy as np

from rml_rm.encodings.monitor_state import normalize_monitor_state
from rml_rm.encodings.semantic_progress import SemanticPhase, SemanticProgressEncoder
from rml_rm.encodings.vector import VectorMonitorStateEncoder


def build_lunar_lander_monitor_encoding(encoding: str):
    """Return encoder, initial monitor state, and monitor space for LunarLander."""
    if encoding != "semantic_progress":
        raise ValueError("Only semantic_progress is currently supported for LunarLander.")

    encoder = build_lunar_lander_semantic_progress_encoder()
    return (
        VectorMonitorStateEncoder(encode_lunar_lander_semantic_progress),
        _with_hover_count(encoder.encode_phase("approach_corridor"), 0),
        None,
    )


def build_lunar_lander_semantic_progress_encoder() -> SemanticProgressEncoder:
    """Build semantic protocol phases for LunarLander monitor states."""
    return SemanticProgressEncoder(
        (
            SemanticPhase("approach_corridor", _matches_approach_corridor),
            SemanticPhase("hovering", _matches_hovering),
            SemanticPhase("hover_complete", _matches_hover_complete),
            SemanticPhase("controlled_descent", _matches_controlled_descent),
            SemanticPhase("terminal_or_failure", _matches_terminal_or_failure),
        )
    )


def encode_lunar_lander_semantic_progress(monitor_state: str) -> np.ndarray:
    """Encode RML phase plus hover-count progress as a fixed monitor vector."""
    encoder = build_lunar_lander_semantic_progress_encoder()
    return _with_hover_count(encoder(monitor_state), _hover_count_from_state(monitor_state))


def _matches_approach_corridor(monitor_state: str) -> bool:
    state = _normalized_state(monitor_state)
    return state.startswith("star(waiting_for_corridor")


def _matches_hovering(monitor_state: str) -> bool:
    state = _normalized_state(monitor_state)
    return not _matches_approach_corridor(state) and "waiting_for_hover" in state


def _matches_hover_complete(monitor_state: str) -> bool:
    state = _normalized_state(monitor_state)
    return state.startswith("app(gen([],star(waiting_for_descent") or state.startswith(
        "star(waiting_for_descent"
    )


def _matches_controlled_descent(monitor_state: str) -> bool:
    state = _normalized_state(monitor_state)
    return state.startswith("star(waiting_for_landing")


def _matches_terminal_or_failure(monitor_state: str) -> bool:
    state = _normalized_state(monitor_state)
    return state in {"1", "false_verdict"}


def semantic_progress_initial_vector() -> np.ndarray:
    """Initial semantic-progress vector for tests and metadata."""
    return _with_hover_count(
        build_lunar_lander_semantic_progress_encoder().encode_phase("approach_corridor"),
        0,
    )


def _normalized_state(monitor_state: str) -> str:
    return normalize_monitor_state(str(monitor_state)).replace("\\_", "_")


def _with_hover_count(phase_vector: np.ndarray, hover_count: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(phase_vector, dtype=np.float32),
            np.asarray([float(np.clip(hover_count / 3.0, 0.0, 1.0))], dtype=np.float32),
        ]
    )


def _hover_count_from_state(monitor_state: str) -> int:
    state = _normalized_state(monitor_state)
    if state in {"1", "false_verdict"}:
        return 0
    if (
        state.startswith("app(gen([],star(waiting_for_descent")
        or state.startswith("star(waiting_for_descent")
        or state.startswith("star(waiting_for_landing")
    ):
        return 3
    if "(1+1)" in state:
        return 2
    if "(0+1)" in state:
        return 1
    return 0
