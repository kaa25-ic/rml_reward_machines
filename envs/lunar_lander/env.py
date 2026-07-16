"""LunarLander environment adapter for RML protocol monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PREFERRED_LUNAR_LANDER_ENV_ID = "LunarLander-v3"
FALLBACK_LUNAR_LANDER_ENV_ID = "LunarLander-v2"
DEFAULT_LUNAR_LANDER_ENV_ID = (
    PREFERRED_LUNAR_LANDER_ENV_ID
    if PREFERRED_LUNAR_LANDER_ENV_ID in gym.registry
    else FALLBACK_LUNAR_LANDER_ENV_ID
)


EVENT_NAMES = (
    "corridor",
    "hover",
    "controlled_descent",
    "target_zone",
    "safe_landing_angle",
    "both_contact",
    "env_terminated",
    "env_successful_landing",
    "env_truncated",
)


THRESHOLD_TOLERANCE = 1e-6


@dataclass(frozen=True)
class LunarProtocolThresholds:
    """Thresholds used to derive monitor propositions from LunarLander states."""

    x_corridor: float = 0.7
    y_corridor_low: float = 0.7
    y_corridor_high: float = 1.4
    y_hover_low: float = 0.6
    y_hover_high: float = 1.0
    vy_hover_max: float = 0.25
    angle_hover_max: float = 0.35
    vy_descent_max: float = 0.6
    angle_descent_max: float = 0.45
    x_target: float = 0.25
    angle_landing_max: float = 0.30
    contact_threshold: float = 0.5


class LunarLanderProtocolEnv(gym.Wrapper):
    """Expose raw LunarLander state and protocol propositions for the monitor."""

    def __init__(
        self,
        env: gym.Env,
        *,
        thresholds: LunarProtocolThresholds | None = None,
    ) -> None:
        super().__init__(env)
        self.thresholds = thresholds or LunarProtocolThresholds()
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                )
            }
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        encoded, events = self._encode_observation(
            observation,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        return {"position": encoded}, _with_event_info(info, events)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        encoded, events = self._encode_observation(
            observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return {"position": encoded}, reward, terminated, truncated, _with_event_info(info, events)

    def _encode_observation(
        self,
        observation: Any,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        lander_state = np.asarray(observation, dtype=np.float32)
        if lander_state.shape != (8,):
            raise ValueError(f"Expected LunarLander observation shape (8,), got {lander_state.shape}.")
        events = lunar_protocol_events(
            lander_state,
            thresholds=self.thresholds,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        return lander_state.astype(np.float32), events


def make_lunar_lander_base_env(
    *,
    env_id: str = DEFAULT_LUNAR_LANDER_ENV_ID,
    render_mode: str | None = None,
    thresholds: LunarProtocolThresholds | None = None,
) -> LunarLanderProtocolEnv:
    """Create LunarLander with RML proposition features but no task tracker."""
    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return LunarLanderProtocolEnv(
        gym.make(env_id, **kwargs),
        thresholds=thresholds,
    )


def lunar_protocol_events(
    observation: np.ndarray,
    *,
    thresholds: LunarProtocolThresholds,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> np.ndarray:
    """Return event propositions used by the RML protocol monitor."""
    x, y, _vx, vy, angle, _angular_velocity, left_contact, right_contact = [
        float(value) for value in observation
    ]
    left = left_contact > thresholds.contact_threshold
    right = right_contact > thresholds.contact_threshold

    corridor = (
        abs(x) <= thresholds.x_corridor + THRESHOLD_TOLERANCE
        and thresholds.y_corridor_low - THRESHOLD_TOLERANCE
        <= y
        <= thresholds.y_corridor_high + THRESHOLD_TOLERANCE
    )
    hover = (
        thresholds.y_hover_low - THRESHOLD_TOLERANCE
        <= y
        <= thresholds.y_hover_high + THRESHOLD_TOLERANCE
        and abs(vy) <= thresholds.vy_hover_max + THRESHOLD_TOLERANCE
        and abs(angle) <= thresholds.angle_hover_max + THRESHOLD_TOLERANCE
    )
    controlled_descent = (
        y < thresholds.y_hover_low
        and abs(vy) <= thresholds.vy_descent_max + THRESHOLD_TOLERANCE
        and abs(angle) <= thresholds.angle_descent_max + THRESHOLD_TOLERANCE
    )
    target_zone = abs(x) <= thresholds.x_target + THRESHOLD_TOLERANCE
    safe_landing_angle = abs(angle) <= thresholds.angle_landing_max + THRESHOLD_TOLERANCE
    env_successful_landing = terminated and reward > 0.0

    return np.asarray(
        [
            float(corridor),
            float(hover),
            float(controlled_descent),
            float(target_zone),
            float(safe_landing_angle),
            float(left and right),
            float(terminated),
            float(env_successful_landing),
            float(truncated),
        ],
        dtype=np.float32,
    )


def _with_event_info(info: dict[str, Any], events: np.ndarray) -> dict[str, Any]:
    wrapped = dict(info)
    for name, value in zip(EVENT_NAMES, events):
        wrapped[name] = float(value)
    return wrapped
