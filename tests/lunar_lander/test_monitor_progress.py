"""Golden tests for LunarLander protocol monitor-state interpretation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from envs.lunar_lander.builder import (
    LunarLanderProtocolConfig,
    _lunar_hover_count,
    _lunar_monitor_progress,
)
from envs.lunar_lander.experiments.render_policy import env_config_from_training_config
from envs.lunar_lander.experiments.train_ppo import LunarLanderPPOTrainingConfig


GOLDEN_STATES_PATH = Path(__file__).parent / "fixtures" / "golden_monitor_states.json"


def _load_golden_monitor_states() -> list[dict[str, object]]:
    return json.loads(GOLDEN_STATES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("raw_state", "expected_progress", "expected_hover_count"),
    [
        (None, 0.0, 0),
        ("false_verdict", -1000.0, 0),
        ("1", 5.0, 0),
        ("star(waiting_for_corridor:eps)", 0.0, 0),
        ("star(waiting\\_for\\_corridor:eps)", 0.0, 0),
        ("star(waiting_for_hover:eps)", 1.0, 0),
        ("star(waiting_for_hover:eps)*(0+1)", 2.0, 1),
        ("star(waiting_for_hover:eps)*(1+1)", 2.0, 2),
        ("star(waiting_for_hover:eps)*[0+1]", 2.0, 1),
        ("star(waiting_for_hover:eps)*[1+1]", 2.0, 2),
        ("star(waiting_for_descent:eps)", 3.0, 3),
        ("app(gen([],star(waiting_for_descent:eps)),[])", 3.0, 3),
        ("star(waiting_for_landing:eps)", 4.0, 3),
        ("star(waiting_for_landing_42:eps)", 4.0, 3),
        ("star(waiting\\_for\\_landing:eps)", 4.0, 3),
    ],
)
def test_lunar_monitor_progress_and_hover_count_are_stable(
    raw_state: str | None,
    expected_progress: float,
    expected_hover_count: int,
) -> None:
    assert _lunar_monitor_progress(raw_state) == expected_progress
    assert _lunar_hover_count(raw_state) == expected_hover_count


@pytest.mark.parametrize("case", _load_golden_monitor_states(), ids=lambda case: case["label"])
def test_real_monitor_states_match_golden_progress(case: dict[str, object]) -> None:
    state = str(case["state"])

    assert _lunar_monitor_progress(state) == float(case["progress"])
    assert _lunar_hover_count(state) == int(case["hover_count"])


@pytest.mark.parametrize("case", _load_golden_monitor_states(), ids=lambda case: case["label"])
def test_real_monitor_states_do_not_fall_through_unexpectedly(
    case: dict[str, object],
) -> None:
    state = str(case["state"])
    progress = _lunar_monitor_progress(state)

    assert progress != 0.0 or str(case["label"]).startswith("corridor")


@pytest.mark.parametrize(
    "config",
    [
        LunarLanderProtocolConfig(),
        LunarLanderPPOTrainingConfig(),
        env_config_from_training_config(
            {},
            render_mode="rgb_array",
            max_episode_steps=None,
        ),
    ],
)
def test_lunar_defaults_match_final_success_aligned_recipe(config: object) -> None:
    assert config.success_bonus == 200.0
    assert config.failure_penalty == -100.0
    assert config.landing_target_bonus == 0.0
    assert config.landing_angle_bonus == 0.0
    assert config.post_descent_landing_bonus == 0.0
    assert config.post_descent_protocol_miss_penalty == 0.0
