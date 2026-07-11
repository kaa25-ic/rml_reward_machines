"""RML monitor-state encodings for multi-task LetterEnv."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from rml_rm.encodings.frozen import FrozenGRUMonitorStateEncoder, FrozenGraphMonitorStateEncoder
from rml_rm.encodings.monitor_state import (
    extract_numerical_values,
    normalize_monitor_state,
    replace_numerical_parts,
    split_top_level_factors,
)
from rml_rm.encodings.vector import VectorMonitorStateEncoder


MULTITASK_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOGUE_PATH = "configs/monitor_state_catalogue.json"
DEFAULT_PROGRESS_CATALOGUE_PATH = "configs/monitor_progress_catalogue.json"
DEFAULT_GRU_CHECKPOINT = (
    MULTITASK_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gru_dim32_seed0"
    / "best_student.pt"
)
DEFAULT_GRAPH_CHECKPOINT = (
    MULTITASK_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_basic_seed0"
    / "best_dynamics_encoder.pt"
)


def build_multitask_monitor_encoding(
    encoding: str,
    *,
    catalogue_path: str | Path | None = None,
    learned_gru_checkpoint: str | Path | None = None,
    learned_graph_checkpoint: str | Path | None = None,
):
    """Return encoder, reset vector, and vector length for a monitor encoding."""
    if encoding == "one_hot":
        catalogue = load_monitor_state_catalogue(catalogue_path)
        templates = _factor_templates(catalogue)
        reset_vector = np.zeros(len(templates) + 3, dtype=np.float32)
        reset_vector[0] = 1.0
        return (
            VectorMonitorStateEncoder(lambda state: _encode_one_hot(state, templates)),
            reset_vector,
            int(reset_vector.shape[0]),
        )

    if encoding == "numerical":
        catalogue = load_monitor_state_catalogue(catalogue_path)
        templates = _factor_templates(catalogue)
        reset_vector = np.zeros(len(templates) + 3, dtype=np.float32)
        reset_vector[0] = 1.0
        return (
            VectorMonitorStateEncoder(lambda state: _encode_numerical(state, templates)),
            reset_vector,
            int(reset_vector.shape[0]),
        )

    if encoding == "learned_gru":
        encoder = FrozenGRUMonitorStateEncoder(learned_gru_checkpoint or DEFAULT_GRU_CHECKPOINT)
        reset_vector = np.zeros_like(encoder(""), dtype=np.float32)
        return VectorMonitorStateEncoder(encoder), reset_vector, int(reset_vector.shape[0])

    if encoding == "learned_graph":
        encoder = FrozenGraphMonitorStateEncoder(
            learned_graph_checkpoint or DEFAULT_GRAPH_CHECKPOINT
        )
        reset_vector = np.zeros_like(encoder("1"), dtype=np.float32)
        return VectorMonitorStateEncoder(encoder), reset_vector, int(reset_vector.shape[0])

    raise ValueError(f"Unsupported multi-task LetterEnv encoding: {encoding!r}")


def load_monitor_state_catalogue(
    catalogue_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Load a task-keyed monitor-state catalogue."""
    if catalogue_path is None:
        path = resources.files("envs.multitask_letter_env").joinpath(DEFAULT_CATALOGUE_PATH)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))

    raw_states = payload.get("states_by_task", {})
    if not isinstance(raw_states, dict) or not raw_states:
        raise ValueError("Monitor-state catalogue must contain a non-empty states_by_task mapping.")
    return {
        str(task_key): [normalize_monitor_state(str(state)) for state in states]
        for task_key, states in raw_states.items()
    }


def load_monitor_progress_catalogue(
    progress_catalogue_path: str | Path | None = None,
) -> dict[str, dict[int, dict[str, int]]]:
    """Load task/count-specific progress values for normalized RML states."""
    if progress_catalogue_path is None:
        path = resources.files("envs.multitask_letter_env").joinpath(
            DEFAULT_PROGRESS_CATALOGUE_PATH
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(progress_catalogue_path).read_text(encoding="utf-8"))

    raw_progress = payload.get("progress_by_task_and_n", {})
    if not isinstance(raw_progress, dict) or not raw_progress:
        raise ValueError(
            "Monitor-progress catalogue must contain a non-empty progress_by_task_and_n mapping."
        )
    return {
        str(task_key): {
            int(n_value): {
                normalize_monitor_state(str(state)): int(progress)
                for state, progress in states.items()
            }
            for n_value, states in by_n.items()
        }
        for task_key, by_n in raw_progress.items()
    }


def _factor_templates(catalogue: dict[str, list[str]]) -> tuple[str, ...]:
    templates: set[str] = set()
    for states in catalogue.values():
        for state in states:
            if state in {"1", "false_verdict"}:
                continue
            templates.update(
                replace_numerical_parts(factor) for factor in split_top_level_factors(state)
            )
    return tuple(sorted(templates))


def _encode_numerical(monitor_state: str, templates: tuple[str, ...]) -> np.ndarray:
    normalized_state = normalize_monitor_state(monitor_state)
    vector = np.zeros(len(templates) + 3, dtype=np.float32)
    if normalized_state == "1":
        vector[1] = 1.0
        return vector
    if normalized_state == "false_verdict":
        vector[2] = 1.0
        return vector

    template_index = {template: index + 3 for index, template in enumerate(templates)}
    for factor in split_top_level_factors(normalized_state):
        template = replace_numerical_parts(factor)
        if template not in template_index:
            raise ValueError(f"Monitor factor is not in the numerical catalogue: {template}")
        values = extract_numerical_values(factor)
        vector[template_index[template]] = float(values[0]) if values else 1.0
    return vector


def _encode_one_hot(monitor_state: str, templates: tuple[str, ...]) -> np.ndarray:
    normalized_state = normalize_monitor_state(monitor_state)
    vector = np.zeros(len(templates) + 3, dtype=np.float32)
    if normalized_state == "1":
        vector[1] = 1.0
        return vector
    if normalized_state == "false_verdict":
        vector[2] = 1.0
        return vector

    template_index = {template: index + 3 for index, template in enumerate(templates)}
    for factor in split_top_level_factors(normalized_state):
        template = replace_numerical_parts(factor)
        if template not in template_index:
            raise ValueError(f"Monitor factor is not in the one-hot catalogue: {template}")
        vector[template_index[template]] = 1.0
    return vector


def catalogue_to_jsonable(states_by_task: dict[str, list[str]]) -> dict[str, Any]:
    """Return a stable JSON payload for a monitor-state catalogue."""
    normalized = {
        task_key: sorted({normalize_monitor_state(state) for state in states})
        for task_key, states in sorted(states_by_task.items())
    }
    return {
        "description": "Reachable RML monitor states for multi-task LetterEnv small_v1.",
        "states_by_task": normalized,
    }
