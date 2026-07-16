"""Boundary tests for LunarLander monitor proposition extraction."""

from __future__ import annotations

import numpy as np

from envs.lunar_lander.env import EVENT_NAMES, LunarProtocolThresholds, lunar_protocol_events


THRESHOLDS = LunarProtocolThresholds()
EPSILON = 1e-4


def _event_dict(
    state: list[float],
    *,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, float]:
    events = lunar_protocol_events(
        np.asarray(state, dtype=np.float32),
        thresholds=THRESHOLDS,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )
    return dict(zip(EVENT_NAMES, [float(value) for value in events]))


def _state(
    *,
    x: float = 0.0,
    y: float = 0.8,
    vx: float = 0.0,
    vy: float = 0.0,
    angle: float = 0.0,
    angular_velocity: float = 0.0,
    left_contact: float = 0.0,
    right_contact: float = 0.0,
) -> list[float]:
    return [x, y, vx, vy, angle, angular_velocity, left_contact, right_contact]


def test_corridor_boundaries() -> None:
    assert _event_dict(_state(x=THRESHOLDS.x_corridor, y=THRESHOLDS.y_corridor_low))[
        "corridor"
    ] == 1.0
    assert _event_dict(
        _state(x=THRESHOLDS.x_corridor + EPSILON, y=THRESHOLDS.y_corridor_low)
    )["corridor"] == 0.0
    assert _event_dict(_state(x=0.0, y=THRESHOLDS.y_corridor_high))[
        "corridor"
    ] == 1.0
    assert _event_dict(_state(x=0.0, y=THRESHOLDS.y_corridor_high + EPSILON))[
        "corridor"
    ] == 0.0


def test_hover_boundaries() -> None:
    assert _event_dict(
        _state(
            y=THRESHOLDS.y_hover_low,
            vy=THRESHOLDS.vy_hover_max,
            angle=THRESHOLDS.angle_hover_max,
        )
    )["hover"] == 1.0
    assert _event_dict(_state(y=THRESHOLDS.y_hover_low - EPSILON))["hover"] == 0.0
    assert _event_dict(_state(y=THRESHOLDS.y_hover_high + EPSILON))["hover"] == 0.0
    assert _event_dict(_state(vy=THRESHOLDS.vy_hover_max + EPSILON))["hover"] == 0.0
    assert _event_dict(_state(angle=THRESHOLDS.angle_hover_max + EPSILON))[
        "hover"
    ] == 0.0


def test_controlled_descent_boundaries() -> None:
    assert _event_dict(
        _state(
            y=THRESHOLDS.y_hover_low - EPSILON,
            vy=THRESHOLDS.vy_descent_max,
            angle=THRESHOLDS.angle_descent_max,
        )
    )["controlled_descent"] == 1.0
    assert _event_dict(_state(y=THRESHOLDS.y_hover_low))["controlled_descent"] == 0.0
    assert _event_dict(_state(y=THRESHOLDS.y_hover_low - EPSILON, vy=0.7))[
        "controlled_descent"
    ] == 0.0
    assert _event_dict(
        _state(y=THRESHOLDS.y_hover_low - EPSILON, angle=THRESHOLDS.angle_descent_max + EPSILON)
    )["controlled_descent"] == 0.0


def test_landing_zone_angle_contact_and_terminal_events() -> None:
    assert _event_dict(_state(x=THRESHOLDS.x_target))["target_zone"] == 1.0
    assert _event_dict(_state(x=THRESHOLDS.x_target + EPSILON))["target_zone"] == 0.0

    assert _event_dict(_state(angle=THRESHOLDS.angle_landing_max))[
        "safe_landing_angle"
    ] == 1.0
    assert _event_dict(_state(angle=THRESHOLDS.angle_landing_max + EPSILON))[
        "safe_landing_angle"
    ] == 0.0

    assert _event_dict(
        _state(
            left_contact=THRESHOLDS.contact_threshold + EPSILON,
            right_contact=THRESHOLDS.contact_threshold + EPSILON,
        )
    )["both_contact"] == 1.0
    assert _event_dict(
        _state(left_contact=THRESHOLDS.contact_threshold, right_contact=1.0)
    )["both_contact"] == 0.0

    assert _event_dict(_state(), terminated=True, reward=1.0)[
        "env_successful_landing"
    ] == 1.0
    assert _event_dict(_state(), terminated=True, reward=-1.0)[
        "env_successful_landing"
    ] == 0.0
    assert _event_dict(_state(), terminated=False, truncated=True)["env_truncated"] == 1.0
