"""Pure environment tests for the single-task LetterEnv."""

from __future__ import annotations

from envs.letter_env.env import LetterEnv
from envs.letter_env_core import LetterAction


def _step_path(env: LetterEnv, actions: list[LetterAction]):
    transition = None
    for action in actions:
        transition = env.step(action)
    assert transition is not None
    return transition


def test_task_string_and_correct_wrong_letter_logic() -> None:
    env = LetterEnv(max_n=3)
    _, info = env.reset(seed=0, options={"n": 2})

    assert info["task_string"] == "ABCDD"

    _, reward, terminated, truncated, info = _step_path(
        env,
        [
            LetterAction.UP,
            LetterAction.UP,
            LetterAction.LEFT,
            LetterAction.LEFT,
            LetterAction.LEFT,
            LetterAction.UP,
        ],
    )
    assert reward == 0
    assert terminated is False
    assert truncated is False
    assert info["task_index"] == 1
    assert info["task_failed"] is False

    _, reward, terminated, _, info = _step_path(
        env,
        [LetterAction.RIGHT, LetterAction.RIGHT, LetterAction.RIGHT],
    )
    assert reward == 0
    assert terminated is True
    assert info["task_failed"] is True
    assert info["success"] is False


def test_scripted_oracle_completes_abc_d_n_task() -> None:
    env = LetterEnv(max_n=2, max_episode_steps=100)
    env.reset(seed=1, options={"n": 2})

    actions = [
        LetterAction.UP,
        LetterAction.UP,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.LEFT,
        LetterAction.UP,
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

    _, reward, terminated, truncated, info = _step_path(env, actions)

    assert reward == 1
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["task_failed"] is False
    assert info["task_index"] == len("ABCDD")
