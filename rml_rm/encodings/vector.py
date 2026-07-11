"""Small adapters for vector-valued monitor-state encoders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class VectorMonitorStateEncoder:
    """Callable adapter that returns monitor-state vectors."""

    def __init__(self, encode: Callable[[str], np.ndarray]) -> None:
        self.encode = encode

    def __call__(self, monitor_state: str) -> np.ndarray:
        return np.asarray(self.encode(monitor_state), dtype=np.float32)
