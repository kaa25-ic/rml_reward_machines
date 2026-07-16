"""Reward wrapper contract tests for the LunarLander protocol stack."""

from __future__ import annotations

from envs.lunar_lander.builder import LunarBaseRewardWrapper, LunarProtocolRewardShapingWrapper


def _info(state: str, **overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "monitor_state_unencoded": state,
        "target_zone": 0.0,
        "safe_landing_angle": 0.0,
        "successful_landing": False,
    }
    info.update(overrides)
    return info


def _shaping_env(scripted_env):
    return LunarProtocolRewardShapingWrapper(
        scripted_env,
        monitor_progress_bonus=10.0,
        hover_step_bonus=2.0,
        hover_complete_bonus=30.0,
        controlled_descent_bonus=40.0,
        success_bonus=100.0,
        failure_penalty=-50.0,
        landing_target_bonus=7.0,
        landing_angle_bonus=9.0,
        post_descent_landing_bonus=11.0,
        post_descent_protocol_miss_penalty=-13.0,
    )


def test_success_shaping_components_fire_on_expected_transitions(
    scripted_monitor_env_factory,
) -> None:
    env = _shaping_env(
        scripted_monitor_env_factory(
            [
                {"info": _info("star(waiting_for_corridor:eps)")},
                {"info": _info("star(waiting_for_hover:eps)")},
                {"info": _info("star(waiting_for_hover:eps)*[0+1]")},
                {"info": _info("star(waiting_for_hover:eps)*[1+1]")},
                {"info": _info("star(waiting_for_descent:eps)")},
                {
                    "info": _info(
                        "star(waiting_for_landing:eps)",
                        target_zone=1.0,
                        safe_landing_angle=1.0,
                    )
                },
                {"info": _info("1", target_zone=1.0, safe_landing_angle=1.0)},
                {"info": _info("1", target_zone=1.0, safe_landing_angle=1.0)},
            ]
        )
    )
    env.reset()

    _, _, _, _, corridor = env.step(0)
    assert corridor["protocol_shaping_reward"] == 0.0

    _, _, _, _, hover = env.step(0)
    assert hover["protocol_reward_progress"] == 10.0

    _, _, _, _, hover_one = env.step(0)
    assert hover_one["protocol_reward_progress"] == 10.0
    assert hover_one["protocol_reward_hover_step"] == 2.0

    _, _, _, _, hover_two = env.step(0)
    assert hover_two["protocol_reward_progress"] == 0.0
    assert hover_two["protocol_reward_hover_step"] == 2.0

    _, _, _, _, descent = env.step(0)
    assert descent["protocol_reward_progress"] == 10.0
    assert descent["protocol_reward_hover_step"] == 2.0
    assert descent["protocol_reward_hover_complete"] == 30.0

    _, _, _, _, landing = env.step(0)
    assert landing["protocol_reward_progress"] == 10.0
    assert landing["protocol_reward_controlled_descent"] == 40.0
    assert landing["protocol_reward_landing_target"] == 7.0
    assert landing["protocol_reward_landing_angle"] == 9.0

    _, _, _, _, success = env.step(0)
    assert success["protocol_reward_progress"] == 10.0
    assert success["protocol_reward_success"] == 100.0
    assert success["protocol_reward_landing_target"] == 0.0
    assert success["protocol_reward_landing_angle"] == 0.0

    _, _, _, _, repeated_success = env.step(0)
    assert repeated_success["protocol_reward_success"] == 0.0
    assert repeated_success["protocol_shaping_reward"] == 0.0


def test_failure_and_post_descent_components_fire_once(scripted_monitor_env_factory) -> None:
    env = _shaping_env(
        scripted_monitor_env_factory(
            [
                {"info": _info("star(waiting_for_landing:eps)")},
                {"info": _info("false_verdict", successful_landing=True)},
                {"info": _info("false_verdict", successful_landing=True)},
            ]
        )
    )
    env.reset()

    env.step(0)
    _, _, _, _, failure = env.step(0)
    assert failure["protocol_reward_failure"] == -50.0
    assert failure["protocol_reward_post_descent_landing"] == 11.0
    assert failure["protocol_reward_post_descent_protocol_miss"] == -13.0

    _, _, _, _, repeated_failure = env.step(0)
    assert repeated_failure["protocol_reward_failure"] == 0.0
    assert repeated_failure["protocol_reward_post_descent_landing"] == 0.0
    assert repeated_failure["protocol_reward_post_descent_protocol_miss"] == 0.0


def test_lunar_base_reward_wrapper_adds_simulator_reward(scripted_monitor_env_factory) -> None:
    env = LunarBaseRewardWrapper(
        scripted_monitor_env_factory(
            [
                {
                    "reward": 3.0,
                    "info": {
                        "base_reward": 4.0,
                        "env_terminated": 1.0,
                    },
                }
            ]
        )
    )
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == 7.0
    assert info["lunar_base_reward"] == 4.0
    assert info["successful_landing"] is True
