"""Observation adapters for raw environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


RawObservation = dict[str, Any]
EncodedObservation = dict[str, np.ndarray]
TabularStateKey = tuple[tuple[float, ...], tuple[float, ...] | int | None]


@dataclass(frozen=True)
class PropositionEncodingSpec:
    """Metadata needed to encode proposition observations."""

    proposition_count: int
    no_proposition_index: int
    position_low: np.ndarray
    position_high: np.ndarray
    max_value: int

    @property
    def encoded_position_shape(self) -> tuple[int, ...]:
        return (int(self.position_low.shape[0]) + self.proposition_count,)


class PropositionVectorObservation(gym.ObservationWrapper):
    """Encode proposition observations into a numeric feature vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.encoding_spec = self._build_spec(env)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.concatenate(
                        [
                            self.encoding_spec.position_low.astype(np.float32),
                            np.zeros(self.encoding_spec.proposition_count, dtype=np.float32),
                        ]
                    ),
                    high=np.concatenate(
                        [
                            self.encoding_spec.position_high.astype(np.float32),
                            np.full(
                                self.encoding_spec.proposition_count,
                                max(1, self.encoding_spec.max_value),
                                dtype=np.float32,
                            ),
                        ]
                    ),
                    shape=self.encoding_spec.encoded_position_shape,
                    dtype=np.float32,
                )
            }
        )

    def observation(self, observation: RawObservation) -> EncodedObservation:
        return {"position": encode_proposition_vector(observation, self.encoding_spec)}

    @staticmethod
    def _build_spec(env: gym.Env) -> PropositionEncodingSpec:
        proposition_to_index = getattr(env, "proposition_to_index", None)
        if proposition_to_index is None:
            raise AttributeError("Environment must expose proposition_to_index.")

        no_proposition_index = proposition_to_index.get("_")
        if no_proposition_index is None:
            raise ValueError("proposition_to_index must include '_' for empty cells.")

        raw_position_space = env.observation_space["position"]
        raw_value_space = env.observation_space["value"]
        return PropositionEncodingSpec(
            proposition_count=len(proposition_to_index),
            no_proposition_index=int(no_proposition_index),
            position_low=np.asarray(raw_position_space.low, dtype=np.float32),
            position_high=np.asarray(raw_position_space.high, dtype=np.float32),
            max_value=int(raw_value_space.n) - 1,
        )


def encode_proposition_vector(observation: RawObservation, spec: PropositionEncodingSpec) -> np.ndarray:
    """Encode one raw observation as coordinates plus proposition features."""
    position = np.asarray(observation["position"], dtype=np.float32).reshape(-1)
    proposition_index = int(observation["proposition"])
    value = float(observation.get("value", 0))

    if not 0 <= proposition_index < spec.proposition_count:
        raise ValueError(
            f"Proposition index {proposition_index} is outside 0..{spec.proposition_count - 1}."
        )

    proposition_features = np.zeros(spec.proposition_count, dtype=np.float32)
    if proposition_index == spec.no_proposition_index:
        proposition_features[proposition_index] = 1.0
    else:
        proposition_features[proposition_index] = value if value > 0 else 1.0

    return np.concatenate([position, proposition_features]).astype(np.float32)


def tabular_state_key(
    observation: dict[str, Any],
    *,
    monitor: np.ndarray | int | None = None,
) -> TabularStateKey:
    """Return a stable hashable state key for tabular Q-learning."""
    position_key = tuple(np.asarray(observation["position"], dtype=np.float32).round(6).tolist())
    monitor_value = observation.get("monitor", monitor)

    if monitor_value is None:
        monitor_key: tuple[float, ...] | int | None = None
    elif isinstance(monitor_value, (int, np.integer)):
        monitor_key = int(monitor_value)
    else:
        monitor_key = tuple(np.asarray(monitor_value, dtype=np.float32).round(6).tolist())

    return position_key, monitor_key
