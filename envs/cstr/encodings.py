"""Monitor-state encodings for the CSTR startup task."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rml_rm.encodings.frozen import FrozenGraphMonitorStateEncoder


class CSTRSemanticProgressEncoder:
    """Encode canonical CSTR startup phases with fixed progress features."""

    initial_state = "<initial>"

    def __init__(self, *, max_states: int, soak_steps: int) -> None:
        self.max_states = int(max_states)
        self.soak_steps = int(soak_steps)
        states = [
            self.initial_state,
            "preheat",
            *[f"soak_{index}" for index in range(1, self.soak_steps + 1)],
            "approach",
            "regulate",
            "success",
            "failure",
        ]
        if len(states) > self.max_states:
            raise ValueError(
                f"CSTR semantic-progress encoding needs {len(states)} states, "
                f"but monitor_state_limit={self.max_states}."
            )
        self.monitor_states: dict[int, str] = dict(enumerate(states))
        self.state_ids: dict[str, int] = {state: index for index, state in self.monitor_states.items()}
        self.current_state = self.initial_state

    @property
    def state_count(self) -> int:
        return len(self.monitor_states)

    @property
    def current_state_name(self) -> str:
        return self.current_state

    @property
    def current_state_id(self) -> int:
        return self.state_ids[self.current_state]

    def reset(self) -> None:
        self.current_state = self.initial_state

    def encode(self, canonical_state: str) -> np.ndarray:
        state = str(canonical_state).strip() or self.initial_state
        if state not in self.state_ids:
            raise ValueError(f"Unknown CSTR canonical monitor state: {state!r}.")
        state_id = self.state_ids[state]
        self.current_state = state
        vector = np.zeros(self.state_count, dtype=np.float32)
        vector[state_id] = 1.0
        return vector


class CSTRFrozenGraphMonitorStateEncoder:
    """Encode raw CSTR monitor strings with a shared frozen graph encoder."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.encoder = FrozenGraphMonitorStateEncoder(checkpoint_path)
        self._initial = np.asarray(self.encoder(""), dtype=np.float32)
        self._cache: dict[str, np.ndarray] = {}
        self._state_count: int | None = None

    @property
    def state_count(self) -> int:
        if self._state_count is None:
            return int(self._initial.shape[0])
        return self._state_count

    @property
    def current_state_name(self) -> str:
        return "raw_monitor_graph"

    @property
    def current_state_id(self) -> int:
        return -1

    def reset(self) -> None:
        return None

    def encode(self, monitor_state: str) -> np.ndarray:
        raw_state = str(monitor_state)
        if raw_state.strip() == "":
            return self._initial.copy()
        cached = self._cache.get(raw_state)
        if cached is not None:
            return cached.copy()
        encoded = np.asarray(self.encoder(raw_state), dtype=np.float32)
        self._state_count = int(encoded.shape[0])
        if self._initial.shape != encoded.shape:
            self._initial = np.zeros(encoded.shape, dtype=np.float32)
        self._cache[raw_state] = encoded.copy()
        return encoded
