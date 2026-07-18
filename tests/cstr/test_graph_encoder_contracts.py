"""CSTR graph-monitor normalization and frozen-adapter contracts."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from envs.cstr.builder import _classify_cstr_rml_state
import envs.cstr.encodings as cstr_encodings
from rml_rm.encodings.rml_graph import normalize_generated_variables, rml_to_graph


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "golden_monitor_states.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _graph_signature(monitor_state: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, ...], ...], tuple[str, ...]]:
    graph = rml_to_graph(monitor_state)
    return (
        graph.node_kinds,
        graph.node_values,
        tuple(tuple(int(value) for value in row) for row in graph.edge_index.tolist()),
        graph.edge_types,
    )


def test_generated_variable_alpha_equivalent_states_have_identical_graphs() -> None:
    left, right = _fixture()["alpha_equivalent_states"]

    left_graph = _graph_signature(normalize_generated_variables(str(left)))
    right_graph = _graph_signature(normalize_generated_variables(str(right)))

    assert left_graph == right_graph


def test_destructive_suffix_stripping_degenerates_the_graph() -> None:
    monitor_state = str(_fixture()["alpha_equivalent_states"][0])

    alias_preserving_graph = rml_to_graph(normalize_generated_variables(monitor_state))
    stripped_graph = rml_to_graph(re.sub(r"_[0-9]+", "", monitor_state))

    assert _graph_signature(normalize_generated_variables(monitor_state)) != _graph_signature(
        re.sub(r"_[0-9]+", "", monitor_state)
    )
    assert {"_v0", "_v1"}.issubset(set(alias_preserving_graph.node_values))
    assert "_v0" not in stripped_graph.node_values
    assert "_v1" not in stripped_graph.node_values


def test_golden_monitor_states_classify_to_expected_phases() -> None:
    for case in _fixture()["classifier_states"]:
        assert (
            _classify_cstr_rml_state(
                str(case["monitor_state"]),
                verdict=str(case["verdict"]),
                payload=case["payload"],
                previous_phase=str(case["previous_phase"]),
                previous_soak_steps=int(case["previous_soak_steps"]),
                soak_steps=int(case["soak_steps"]),
            )
            == case["expected_phase"]
        )


def test_cstr_frozen_graph_adapter_initial_state_cache_and_output_shape(monkeypatch) -> None:
    class FakeFrozenGraphMonitorStateEncoder:
        def __init__(self, checkpoint_path: str) -> None:
            self.checkpoint_path = checkpoint_path
            self.calls: list[str] = []

        def __call__(self, monitor_state: str) -> np.ndarray:
            state = str(monitor_state)
            self.calls.append(state)
            if state == "":
                return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            return np.asarray([float(len(state)), 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(
        cstr_encodings,
        "FrozenGraphMonitorStateEncoder",
        FakeFrozenGraphMonitorStateEncoder,
    )
    encoder = cstr_encodings.CSTRFrozenGraphMonitorStateEncoder("unused.pt")

    initial = encoder.encode("")
    initial[0] = 99.0
    np.testing.assert_array_equal(encoder.encode(""), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    first = encoder.encode("state_1")
    first[0] = -1.0
    second = encoder.encode("state_1")

    np.testing.assert_array_equal(second, np.asarray([7.0, 2.0, 3.0], dtype=np.float32))
    assert encoder.state_count == 3
    assert encoder.encoder.calls.count("state_1") == 1
