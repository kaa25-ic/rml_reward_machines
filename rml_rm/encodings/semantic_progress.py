"""Generic semantic progress encoders for monitor states."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


MonitorStatePredicate = Callable[[str], bool]


@dataclass(frozen=True)
class SemanticPhase:
    """One named task phase with a monitor-state predicate."""

    name: str
    matches: MonitorStatePredicate


class SemanticProgressEncoder:
    """Encode monitor states as one-hot task-phase vectors."""

    def __init__(self, phases: tuple[SemanticPhase, ...]) -> None:
        if not phases:
            raise ValueError("At least one semantic phase is required.")
        phase_names = [phase.name for phase in phases]
        if len(set(phase_names)) != len(phase_names):
            raise ValueError("Semantic phase names must be unique.")
        self.phases = phases
        self.phase_to_index = {phase.name: index for index, phase in enumerate(phases)}

    @property
    def vector_length(self) -> int:
        return len(self.phases)

    def __call__(self, monitor_state: str) -> np.ndarray:
        for phase in self.phases:
            if phase.matches(monitor_state):
                return self.encode_phase(phase.name)
        raise ValueError(f"Monitor state did not match any semantic phase: {monitor_state}")

    def encode_phase(self, phase_name: str) -> np.ndarray:
        if phase_name not in self.phase_to_index:
            raise ValueError(f"Unknown semantic phase: {phase_name}")
        vector = np.zeros(self.vector_length, dtype=np.float32)
        vector[self.phase_to_index[phase_name]] = 1.0
        return vector
