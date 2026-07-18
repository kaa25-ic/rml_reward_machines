"""CSTR RML reward-shaping transition contracts."""

from __future__ import annotations

import pytest

from envs.cstr.builder import RMLCSTRConfig, RMLCSTREnv
from envs.cstr.env import CSTRConfig


def _reward_env() -> RMLCSTREnv:
    env = object.__new__(RMLCSTREnv)
    env.config = RMLCSTRConfig(
        cstr_env=CSTRConfig(),
        soak_steps=3,
        safe_step_bonus=0.1,
        stable_step_bonus=1.0,
        regulation_entry_bonus=5.0,
        production_entry_bonus=10.0,
        success_bonus=50.0,
        failure_penalty=-50.0,
        heating_rate_penalty=0.02,
        preheat_distance_weight=0.0,
        preheat_warming_weight=0.0,
        soak_entry_bonus=5.0,
        soak_progress_bonus=0.75,
        soak_reset_penalty=-3.0,
        soak_lost_step_penalty=0.5,
        approach_distance_weight=0.0,
        approach_progress_bonus=0.0,
        approach_ca_progress_bonus=0.0,
        approach_temp_progress_bonus=0.0,
        approach_warming_weight=0.0,
        regulate_recovery_penalty=-10.0,
    )
    env.monitor_phase = "preheat"
    env.previous_monitor_phase = "preheat"
    env.monitor_soak_steps = 0
    env.previous_monitor_soak_steps = 0
    env.max_rewarded_soak_steps = 0
    env.has_entered_soak = False
    env.has_entered_regulate = False
    env.previous_approach_distance = None
    env.previous_approach_ca_error = None
    env.previous_approach_temp_error = None
    return env


def _info(**overrides: object) -> dict[str, object]:
    info = {
        "event_heating_rate_exceeded": False,
        "reactor_temperature": 345.0,
        "reactor_concentration": 0.5,
        "target_concentration": 0.5,
        "heating_rate": 0.0,
    }
    info.update(overrides)
    return info


def test_soak_entry_bonus_fires_once_and_soak_progress_is_counted() -> None:
    env = _reward_env()
    env.previous_monitor_phase = "preheat"
    env.monitor_phase = "soak"
    env.monitor_soak_steps = 1

    first_soak_reward = env._rml_reward(_info())
    second_soak_reward = env._rml_reward(_info())

    assert first_soak_reward == pytest.approx(0.1 + 0.5 + 0.75 / 3.0 + 5.0)
    assert second_soak_reward == pytest.approx(0.1 + 0.5 + 0.75 / 3.0)
    assert env.has_entered_soak
    assert env.max_rewarded_soak_steps == 1


def test_losing_soak_progress_applies_reset_and_lost_step_penalties() -> None:
    env = _reward_env()
    env.previous_monitor_phase = "soak"
    env.previous_monitor_soak_steps = 2
    env.monitor_phase = "preheat"

    assert env._rml_reward(_info()) == pytest.approx(-3.0 - 0.5 * 2)


def test_soak_to_approach_adds_safe_step_and_regulation_entry_reward() -> None:
    env = _reward_env()
    env.previous_monitor_phase = "soak"
    env.monitor_phase = "approach"

    assert env._rml_reward(_info()) == pytest.approx(0.1 + 0.5 * 5.0)


def test_approach_to_regulate_adds_stability_and_entry_rewards_once() -> None:
    env = _reward_env()
    env.previous_monitor_phase = "approach"
    env.monitor_phase = "regulate"

    first_regulate_reward = env._rml_reward(_info())
    second_regulate_reward = env._rml_reward(_info())

    assert first_regulate_reward == pytest.approx(0.1 + 1.0 + 5.0 + 10.0)
    assert second_regulate_reward == pytest.approx(0.1 + 1.0)
    assert env.has_entered_regulate


def test_regulate_recovery_to_approach_is_penalized_but_not_terminal_reward() -> None:
    env = _reward_env()
    env.previous_monitor_phase = "regulate"
    env.monitor_phase = "approach"

    assert env._rml_reward(_info()) == pytest.approx(0.1 - 10.0)


def test_terminal_success_and_failure_rewards_fire_on_entry_only() -> None:
    success_env = _reward_env()
    success_env.previous_monitor_phase = "regulate"
    success_env.monitor_phase = "success"
    assert success_env._rml_reward(_info()) == pytest.approx(50.0)
    success_env.previous_monitor_phase = "success"
    assert success_env._rml_reward(_info()) == pytest.approx(0.0)

    failure_env = _reward_env()
    failure_env.previous_monitor_phase = "approach"
    failure_env.monitor_phase = "failure"
    assert failure_env._rml_reward(_info()) == pytest.approx(-50.0)
    failure_env.previous_monitor_phase = "failure"
    assert failure_env._rml_reward(_info()) == pytest.approx(0.0)
