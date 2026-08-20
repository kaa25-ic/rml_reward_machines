"""Finite-state reward-machine baseline tests for LetterEnv."""

from __future__ import annotations

import numpy as np
import pytest

from envs.letter_env import LetterAction, LetterEnvConfig, build_letter_env


MONITOR_CONFIG = "envs/letter_env/configs/letter_env.yaml"


def _make_env(**overrides: object):
    config = LetterEnvConfig(
        encoding="finite_state_rm",
        n_value=3,
        fixed_n=2,
        max_episode_steps=100,
        **overrides,
    )
    return build_letter_env(config, monitor_config_path=MONITOR_CONFIG)


def _step_path(env, actions: list[LetterAction]):
    transition = None
    for action in actions:
        transition = env.step(action.value)
    assert transition is not None
    return transition


def _path_to_a() -> list[LetterAction]:
    return [
        LetterAction.UP,
        LetterAction.UP,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.UP,
    ]


def _successful_n2_path() -> list[LetterAction]:
    return [
        *_path_to_a(),
        LetterAction.RIGHT,
        LetterAction.LEFT,
        LetterAction.RIGHT,
        LetterAction.RIGHT,
        LetterAction.RIGHT,
        LetterAction.DOWN,
        LetterAction.DOWN,
        LetterAction.DOWN,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.RIGHT,
        LetterAction.LEFT,
    ]


def test_finite_state_rm_reset_exposes_one_hot_initial_state() -> None:
    env = _make_env()

    try:
        observation, info = env.reset(seed=0)

        assert observation["monitor"].shape == (8,)
        np.testing.assert_array_equal(
            observation["monitor"],
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        assert info["monitor_state_unencoded"] == "finite_state_rm:progress_0"
        assert info["finite_state_reward_machine"] is True
        assert info["finite_state_rm_target_length"] == len("ABCDD")
    finally:
        env.close()


def test_finite_state_rm_tracks_progress_and_rewards_success_without_rml_monitor() -> None:
    env = _make_env()

    try:
        env.reset(seed=1)
        _observation, reward, terminated, truncated, info = _step_path(env, _successful_n2_path())

        assert reward == pytest.approx(100.0)
        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        assert info["monitor_reward"] == pytest.approx(100.0)
        assert info["monitor_transition_bonus"] == pytest.approx(0.0)
        assert info["monitor_state_unencoded"] == "finite_state_rm:success"
        assert info["finite_state_rm_progress"] == len("ABCDD")
    finally:
        env.close()


def test_finite_state_rm_rejects_wrong_letter_with_monitor_failure_reward() -> None:
    env = _make_env()

    try:
        env.reset(seed=2)
        _step_path(env, _path_to_a())
        _observation, reward, terminated, _truncated, info = _step_path(
            env,
            [LetterAction.RIGHT, LetterAction.RIGHT, LetterAction.RIGHT],
        )

        assert reward == pytest.approx(-40.0)
        assert terminated is True
        assert info["task_failed"] is True
        assert info["monitor_reward"] == pytest.approx(-40.0)
        assert info["monitor_state_unencoded"] == "finite_state_rm:failure"
    finally:
        env.close()


def test_finite_state_rm_supports_transition_and_progress_shaping() -> None:
    env = _make_env(
        neutralize_legacy_transition_bonus=False,
        legacy_transition_bonus=10.0,
        monitor_progress_bonus=5.0,
    )

    try:
        env.reset(seed=3)
        _observation, reward, terminated, _truncated, info = _step_path(env, _path_to_a())

        assert reward == pytest.approx(15.0)
        assert terminated is False
        assert info["monitor_reward"] == pytest.approx(0.0)
        assert info["monitor_transition_bonus"] == pytest.approx(10.0)
        assert info["monitor_state_unencoded"] == "finite_state_rm:progress_1"
    finally:
        env.close()
