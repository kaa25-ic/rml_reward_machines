"""LetterEnv monitor-state encodings."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from rml_rm.encodings.frozen import FrozenGRUMonitorStateEncoder, FrozenGraphMonitorStateEncoder
from rml_rm.encodings.monitor_state import (
    extract_events,
    extract_numerical_values,
    normalize_monitor_state,
    replace_numerical_parts,
    split_top_level_factors,
)
from rml_rm.encodings.semantic_progress import SemanticPhase, SemanticProgressEncoder


LETTER_ENV_ROOT = Path(__file__).resolve().parent
DEFAULT_GRU_CHECKPOINT = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gru_dim16_seed0"
    / "best_student.pt"
)
DEFAULT_GRAPH_CHECKPOINT = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_basic_seed0"
    / "best_dynamics_encoder.pt"
)

RUNTIME_COMPATIBLE_INITIAL_SIGNATURE = (
    "(star(not_abcd:eps)*var(n,(a_match(var(n)):eps)*app(gen([n],star(not_abcd:eps)*"
    "((b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),"
    "[var(n)]))),[var(n)])),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*"
    "((d_match:eps)*app(,[var(n)-1])),1))])"
)
RUNTIME_COMPATIBLE_B_APP_SIGNATURE = (
    "(app(gen([n],star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*"
    "((c_match:eps)*app(,[var(n)]))),[var(n)]))),[{num}]),[=gen([n],guarded(var(n)>0,"
    "star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])"
)
RUNTIME_COMPATIBLE_B_STAR_SIGNATURE = (
    "(star(not_abcd:eps)*((b_match:eps)*app(gen([n],star(not_abcd:eps)*"
    "((c_match:eps)*app(,[var(n)]))),[{num}])),[=gen([n],guarded(var(n)>0,"
    "star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])"
)
RUNTIME_COMPATIBLE_C_SIGNATURE = (
    "(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[{num}])),"
    "[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])"
)
RUNTIME_COMPATIBLE_D_SIGNATURE = (
    "(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[{num}])),"
    "[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])"
)


class VectorMonitorStateEncoder:
    """Callable adapter that returns monitor-state vectors."""

    def __init__(self, encode) -> None:
        self.encode = encode

    def __call__(self, monitor_state: str) -> np.ndarray:
        return np.asarray(self.encode(monitor_state), dtype=np.float32)


def load_letter_env_monitor_state_catalogue() -> dict[int, str]:
    """Load the tracked LetterEnv monitor-state catalogue."""
    catalogue_path = resources.files("envs.letter_env").joinpath(
        "configs/monitor_state_catalogue.json"
    )
    payload = json.loads(catalogue_path.read_text(encoding="utf-8"))
    states = payload["states"]
    return {int(key): str(value) for key, value in states.items()}


def build_letter_env_monitor_encoding(
    encoding: str,
    *,
    learned_gru_checkpoint: str | Path | None = None,
    learned_graph_checkpoint: str | Path | None = None,
):
    """Return encoder, initial state, and monitor space for LetterEnv."""
    if encoding == "simple":
        return None, 0, None

    if encoding == "hidden_monitor_state":
        hidden_state = np.zeros(1, dtype=np.float32)
        return VectorMonitorStateEncoder(lambda state: hidden_state.copy()), hidden_state, None

    states_by_id = load_letter_env_monitor_state_catalogue()
    states = list(states_by_id.values())
    initial_state = states_by_id[0]

    if encoding == "one_hot":
        event_index = _build_legacy_one_hot_event_index(states)
        initial_encoding = _legacy_one_hot_encoding(initial_state, event_index)
        signature_to_vector: dict[tuple[str, ...], np.ndarray] = {}
        for state in states_by_id.values():
            signature = tuple(extract_events(state))
            signature_to_vector.setdefault(
                signature,
                _legacy_one_hot_encoding(state, event_index).astype(np.float32),
            )
        _add_runtime_compatible_one_hot_aliases(signature_to_vector, states_by_id)
        return (
            VectorMonitorStateEncoder(
                lambda state: _encode_runtime_compatible_one_hot_state(
                    state,
                    event_index,
                    signature_to_vector,
                )
            ),
            initial_encoding,
            None,
        )

    if encoding == "numerical":
        event_index = _build_legacy_numerical_event_index(states)
        initial_encoding = _legacy_numerical_encoding(initial_state, event_index)
        state_lookup = {
            normalize_monitor_state(state): _legacy_numerical_encoding(state, event_index).astype(
                np.float32
            )
            for state in states_by_id.values()
        }
        runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray] = {}
        template_by_signature = {
            RUNTIME_COMPATIBLE_B_APP_SIGNATURE: (states_by_id[1], "[0+1]"),
            RUNTIME_COMPATIBLE_B_STAR_SIGNATURE: (states_by_id[1], "[0+1]"),
            tuple(extract_events(states_by_id[10])): (states_by_id[10], "[1]"),
            tuple(extract_events(states_by_id[18])): (states_by_id[18], "[1]"),
        }
        _add_runtime_compatible_numerical_aliases(
            runtime_signature_to_vector,
            states_by_id,
            event_index,
        )
        return (
            VectorMonitorStateEncoder(
                lambda state: _encode_runtime_compatible_numerical_state(
                    state,
                    event_index,
                    state_lookup,
                    runtime_signature_to_vector,
                    template_by_signature,
                )
            ),
            initial_encoding,
            None,
        )

    if encoding == "semantic_progress":
        encoder = build_letter_env_semantic_progress_encoder(states_by_id)
        return (
            VectorMonitorStateEncoder(encoder),
            encoder.encode_phase("waiting_for_a"),
            None,
        )

    if encoding == "learned_gru":
        encoder = FrozenGRUMonitorStateEncoder(learned_gru_checkpoint or DEFAULT_GRU_CHECKPOINT)
        return (
            VectorMonitorStateEncoder(encoder),
            encoder(initial_state),
            None,
        )

    if encoding == "learned_graph":
        encoder = FrozenGraphMonitorStateEncoder(learned_graph_checkpoint or DEFAULT_GRAPH_CHECKPOINT)
        return (
            VectorMonitorStateEncoder(encoder),
            encoder(initial_state),
            None,
        )

    raise ValueError(f"Unsupported LetterEnv monitor encoding: {encoding}")


def build_letter_env_semantic_progress_encoder(
    states_by_id: dict[int, str] | None = None,
) -> SemanticProgressEncoder:
    """Build the five-phase semantic progress encoder for LetterEnv."""
    states_by_id = states_by_id or load_letter_env_monitor_state_catalogue()
    signatures_by_phase = _letter_env_semantic_phase_signatures(states_by_id)
    return SemanticProgressEncoder(
        (
            SemanticPhase(
                "waiting_for_a",
                _matches_signature(signatures_by_phase["waiting_for_a"]),
            ),
            SemanticPhase(
                "waiting_for_b",
                _matches_signature(signatures_by_phase["waiting_for_b"]),
            ),
            SemanticPhase(
                "waiting_for_c",
                _matches_signature(signatures_by_phase["waiting_for_c"]),
            ),
            SemanticPhase(
                "waiting_for_d",
                _matches_signature(signatures_by_phase["waiting_for_d"]),
            ),
            SemanticPhase("terminal_or_failure", _matches_terminal_state),
        )
    )


def _letter_env_semantic_phase_signatures(
    states_by_id: dict[int, str],
) -> dict[str, set[tuple[str, ...]]]:
    phase_state_ids = {
        "waiting_for_a": (0,),
        "waiting_for_b": (1, 2, 3, 4, 5, 6, 7, 8, 22, 23),
        "waiting_for_c": (10, 11, 12, 13, 16, 17, 20, 21, 24, 25),
        "waiting_for_d": (
            14,
            15,
            18,
            19,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            35,
            36,
            37,
        ),
    }
    signatures_by_phase = {
        phase: {tuple(extract_events(states_by_id[state_id])) for state_id in state_ids}
        for phase, state_ids in phase_state_ids.items()
    }
    signatures_by_phase["waiting_for_a"].add(
        tuple(extract_events(RUNTIME_COMPATIBLE_INITIAL_SIGNATURE))
    )
    signatures_by_phase["waiting_for_b"].update(
        {
            tuple(extract_events(RUNTIME_COMPATIBLE_B_APP_SIGNATURE)),
            tuple(extract_events(RUNTIME_COMPATIBLE_B_STAR_SIGNATURE)),
        }
    )
    signatures_by_phase["waiting_for_c"].add(
        tuple(extract_events(RUNTIME_COMPATIBLE_C_SIGNATURE))
    )
    signatures_by_phase["waiting_for_d"].add(
        tuple(extract_events(RUNTIME_COMPATIBLE_D_SIGNATURE))
    )
    return signatures_by_phase


def _matches_signature(signatures: set[tuple[str, ...]]):
    def matches(monitor_state: str) -> bool:
        return tuple(extract_events(monitor_state)) in signatures

    return matches


def _matches_terminal_state(monitor_state: str) -> bool:
    return normalize_monitor_state(monitor_state) in {"1", "false_verdict"}


def _add_runtime_compatible_one_hot_aliases(
    signature_to_vector: dict[tuple[str, ...], np.ndarray],
    states_by_id: dict[int, str],
) -> None:
    target_signatures = {
        "initial": tuple(extract_events(states_by_id[0])),
        "a_or_b": tuple(extract_events(states_by_id[1])),
        "c_pending": tuple(extract_events(states_by_id[11])),
        "d_pending": tuple(extract_events(states_by_id[15])),
    }
    runtime_aliases = {
        RUNTIME_COMPATIBLE_INITIAL_SIGNATURE: target_signatures["initial"],
        RUNTIME_COMPATIBLE_B_APP_SIGNATURE: target_signatures["a_or_b"],
        RUNTIME_COMPATIBLE_B_STAR_SIGNATURE: target_signatures["a_or_b"],
        RUNTIME_COMPATIBLE_C_SIGNATURE: target_signatures["c_pending"],
        RUNTIME_COMPATIBLE_D_SIGNATURE: target_signatures["d_pending"],
    }
    for runtime_signature, target_signature in runtime_aliases.items():
        signature_to_vector[tuple(extract_events(runtime_signature))] = signature_to_vector[
            target_signature
        ].copy()


def _encode_runtime_compatible_one_hot_state(
    monitor_state: str,
    event_index: dict[str, int],
    signature_to_vector: dict[tuple[str, ...], np.ndarray],
) -> np.ndarray:
    signature = tuple(extract_events(monitor_state))
    vector = signature_to_vector.get(signature)
    if vector is not None:
        return vector.copy()
    return _legacy_one_hot_encoding(monitor_state, event_index).astype(np.float32)


def _add_runtime_compatible_numerical_aliases(
    runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray],
    states_by_id: dict[int, str],
    event_index: dict[str, int],
) -> None:
    target_vectors = {
        "initial": _legacy_numerical_encoding(states_by_id[0], event_index).astype(np.float32),
        "c_pending": _legacy_numerical_encoding(states_by_id[11], event_index).astype(np.float32),
        "d_pending": _legacy_numerical_encoding(states_by_id[15], event_index).astype(np.float32),
    }
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_INITIAL_SIGNATURE] = target_vectors["initial"]
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_C_SIGNATURE] = target_vectors["c_pending"]
    runtime_signature_to_vector[RUNTIME_COMPATIBLE_D_SIGNATURE] = target_vectors["d_pending"]


def _encode_runtime_compatible_numerical_state(
    monitor_state: str,
    event_index: dict[str, int],
    state_lookup: dict[str, np.ndarray],
    runtime_signature_to_vector: dict[tuple[str, ...], np.ndarray],
    template_by_signature: dict[tuple[str, ...], tuple[str, str]],
) -> np.ndarray:
    normalized_state = normalize_monitor_state(monitor_state)
    if normalized_state in state_lookup:
        return state_lookup[normalized_state].copy()

    signature = tuple(extract_events(monitor_state))
    values = extract_numerical_values(monitor_state) or []
    template = template_by_signature.get(signature)
    if template is not None and values:
        template_state, placeholder = template
        template_state = template_state.replace(placeholder, f"[{values[0]}]")
        return _legacy_numerical_encoding(template_state, event_index).astype(np.float32)

    vector = runtime_signature_to_vector.get(signature)
    if vector is not None:
        return vector.copy()
    return _legacy_numerical_encoding(monitor_state, event_index).astype(np.float32)


def _legacy_event_parts(state: str) -> list[str]:
    normalized = normalize_monitor_state(state).replace("@", "")
    if normalized.startswith("(eps"):
        normalized = normalized[len("(eps*") :]
    return normalized.split("*")


def _build_legacy_one_hot_event_index(states: list[str]) -> dict[str, int]:
    event_index: dict[str, int] = {}
    for state in states:
        for event in _legacy_extract_events(state):
            if event not in event_index:
                event_index[event] = len(event_index)
    return event_index


def _build_legacy_numerical_event_index(states: list[str]) -> dict[str, int]:
    event_index: dict[str, int] = {}
    next_index = 0
    for state in states:
        for event in _legacy_extract_events(state):
            if event in event_index:
                continue
            event_index[event] = next_index
            next_index += 1
            for extra_index in range(1, event.count("{num}")):
                event_index[event + "£ADDITIONAL£" * extra_index] = next_index
                next_index += 1
    return event_index


def _legacy_extract_events(state: str) -> list[str]:
    return [replace_numerical_parts(part.strip()) for part in _legacy_event_parts(state)]


def _legacy_one_hot_encoding(state: str, event_index: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(event_index), dtype=np.float32)
    parts = _legacy_event_parts(state)
    _set_one_hot_part(vector, parts[0], event_index)
    if "star" in replace_numerical_parts(parts[0]) and len(parts) > 1:
        _set_one_hot_part(vector, parts[1], event_index)
    return vector


def _legacy_numerical_encoding(state: str, event_index: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(event_index), dtype=np.float32)
    parts = _legacy_event_parts(state)
    normalized_first_part = replace_numerical_parts(parts[0])
    _set_numerical_part(vector, parts[0], event_index)
    if "star" in normalized_first_part and len(parts) > 1:
        _set_numerical_part(vector, parts[1], event_index)
    return vector


def _set_one_hot_part(vector: np.ndarray, part: str, event_index: dict[str, int]) -> None:
    normalized_part = replace_numerical_parts(part.strip())
    index = event_index.get(normalized_part)
    if index is not None:
        vector[index] = 1.0


def _set_numerical_part(vector: np.ndarray, part: str, event_index: dict[str, int]) -> None:
    normalized_part = replace_numerical_parts(part.strip())
    values = _legacy_numerical_values(part)
    index = event_index.get(normalized_part)
    if index is None:
        return
    if values is None:
        vector[index] = 1.0
        return
    for value_index, value in enumerate(values):
        key = normalized_part + "£ADDITIONAL£" * value_index
        if key in event_index:
            vector[event_index[key]] = value


def _legacy_numerical_values(part: str) -> list[float] | None:
    values: list[float] = []
    for factor in split_top_level_factors(part):
        extracted_values = extract_numerical_values(factor)
        if extracted_values:
            values.extend(extracted_values)
    return values or None
