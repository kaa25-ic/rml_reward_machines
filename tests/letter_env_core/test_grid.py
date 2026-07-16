"""Shared LetterGridWorld behavior tests."""

from __future__ import annotations

import pytest

from envs.letter_env_core import LetterAction, LetterGridWorld, NO_PROPOSITION


def test_movement_actions_and_extra_forbidden_transition() -> None:
    env = LetterGridWorld(
        agent_start_location=(2, 2),
        forbidden_transitions={(2, 2, LetterAction.RIGHT)},
    )
    env.reset_grid(sampled_n=2)

    assert env._next_position((2, 2), LetterAction.RIGHT) == (2, 2)
    assert env._next_position((2, 2), LetterAction.LEFT) == (2, 1)
    assert env._next_position((2, 2), LetterAction.UP) == (1, 2)
    assert env._next_position((2, 2), LetterAction.DOWN) == (3, 2)


@pytest.mark.parametrize(
    ("position", "action"),
    [
        ((0, 0), LetterAction.UP),
        ((0, 5), LetterAction.UP),
        ((5, 0), LetterAction.DOWN),
        ((5, 5), LetterAction.DOWN),
        ((0, 0), LetterAction.LEFT),
        ((5, 0), LetterAction.LEFT),
        ((0, 5), LetterAction.RIGHT),
        ((5, 5), LetterAction.RIGHT),
    ],
)
def test_border_walls_block_off_grid_moves(
    position: tuple[int, int],
    action: LetterAction,
) -> None:
    env = LetterGridWorld()
    env.reset_grid(sampled_n=1)

    assert env._next_position(position, action) == position


def test_proposition_replacement_and_current_proposition() -> None:
    env = LetterGridWorld(locations={"A": (1, 1), "C": (1, 4), "D": (4, 1)})
    env.reset_grid(sampled_n=3)
    env.agent_position = (1, 1)

    assert env.current_proposition == "A"
    env.record_observation("A")
    env.apply_replacement_if_needed("A")

    assert env.current_proposition == "B"
    assert env.active_propositions[(1, 1)] == "B"


def test_make_observation_exposes_position_proposition_and_a_value_only() -> None:
    env = LetterGridWorld(max_n=5)
    env.reset_grid(sampled_n=4)
    env.agent_position = (1, 1)

    a_observation = env.make_observation()
    assert a_observation["position"].tolist() == [1, 1]
    assert a_observation["proposition"] == env.proposition_to_index["A"]
    assert a_observation["value"] == 4

    env.agent_position = (4, 1)
    d_observation = env.make_observation()
    assert d_observation["proposition"] == env.proposition_to_index["D"]
    assert d_observation["value"] == 0

    env.agent_position = (0, 0)
    empty_observation = env.make_observation()
    assert empty_observation["proposition"] == env.proposition_to_index[NO_PROPOSITION]
    assert empty_observation["value"] == 0


def test_reset_seed_and_fixed_actions_are_deterministic() -> None:
    actions = [
        LetterAction.UP,
        LetterAction.LEFT,
        LetterAction.DOWN,
        LetterAction.RIGHT,
        LetterAction.RIGHT,
    ]

    trajectories = []
    for _ in range(2):
        env = LetterGridWorld(max_n=5)
        env.reset(seed=123)
        env.reset_grid(sampled_n=3)
        trajectory = []
        for action in actions:
            label = env.move_agent(action)
            trajectory.append((env.agent_position, label))
        trajectories.append(trajectory)

    assert trajectories[0] == trajectories[1]
