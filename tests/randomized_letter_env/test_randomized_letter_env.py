"""Pure randomized LetterEnv invariant tests."""

from __future__ import annotations

import numpy as np

from envs.letter_env_core import LetterAction
from envs.randomized_letter_env.builder import RandomizedLetterObservation
from envs.randomized_letter_env.env import RandomizedLetterEnv


def test_regional_placement_invariants_over_many_seeded_resets() -> None:
    env = RandomizedLetterEnv(placement_mode="regional")

    for seed in range(100):
        _, info = env.reset(seed=seed)
        targets = info["target_positions"]
        assert 0 <= targets["A"][0] <= 2 and 0 <= targets["A"][1] <= 2
        assert 0 <= targets["C"][0] <= 2 and 3 <= targets["C"][1] <= 5
        assert 3 <= targets["D"][0] <= 5 and 0 <= targets["D"][1] <= 2


def test_full_random_letters_do_not_overlap_or_start_on_agent() -> None:
    env = RandomizedLetterEnv(placement_mode="full_random")

    for seed in range(100):
        _, info = env.reset(seed=seed)
        targets = info["target_positions"]
        unique_letters = {targets["A"], targets["C"], targets["D"]}
        assert len(unique_letters) == 3
        assert info["agent_start_location"] not in unique_letters


def test_randomized_observation_encodes_normalized_target_coordinates() -> None:
    env = RandomizedLetterObservation(RandomizedLetterEnv(placement_mode="regional"))
    observation, info = env.reset(seed=7, options={"n": 1})
    raw_env = env.unwrapped

    encoded = np.asarray(observation["position"], dtype=np.float32)
    row_scale = raw_env.n_rows - 1
    col_scale = raw_env.n_cols - 1
    expected_targets = []
    for symbol in ("A", "B", "C", "D"):
        location_symbol = "A" if symbol == "B" else symbol
        row, col = info["target_positions"][location_symbol]
        expected_targets.extend([row / row_scale, col / col_scale])

    np.testing.assert_allclose(encoded[-8:], np.asarray(expected_targets, dtype=np.float32))


def _move_toward(current: tuple[int, int], target: tuple[int, int]) -> LetterAction:
    row, col = current
    target_row, target_col = target
    if row < target_row:
        return LetterAction.DOWN
    if row > target_row:
        return LetterAction.UP
    if col < target_col:
        return LetterAction.RIGHT
    if col > target_col:
        return LetterAction.LEFT
    raise AssertionError("Already at target.")


def test_scripted_oracle_can_visit_targets_reported_in_info() -> None:
    env = RandomizedLetterEnv(placement_mode="full_random", max_episode_steps=100)
    _, info = env.reset(seed=11, options={"n": 1})
    targets = info["target_positions"]

    observed = []
    for target_key in ("A", "B", "C", "D"):
        target = targets[target_key]
        if env.agent_position == target and target_key == "B":
            env.step(LetterAction.RIGHT if target[1] < env.n_cols - 1 else LetterAction.LEFT)
        while env.agent_position != target:
            _, _, _, _, info = env.step(_move_toward(env.agent_position, target))
        observed.append(info["proposition_label"])

    assert observed == ["A", "B", "C", "D"]
