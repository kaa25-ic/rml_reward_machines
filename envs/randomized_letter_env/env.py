"""Randomized single-task LetterEnv gridworld."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.letter_env_core import NO_PROPOSITION, ForbiddenTransition, LetterGridWorld


class RandomizedLetterEnv(LetterGridWorld):
    """LetterEnv variant with randomized letter locations.

    The symbolic task is A, B, C, then D repeated ``n`` times. The agent starts
    from a fixed location by default, while A, C, and D are sampled without
    replacement on every episode. B is revealed at the A location after A has
    been observed once.
    """

    def __init__(
        self,
        *,
        max_n: int = 1,
        n_rows: int = 6,
        n_cols: int = 6,
        propositions: tuple[str, ...] = ("A", "B", "C", "D"),
        agent_start_location: tuple[int, int] = (4, 4),
        max_observation_counts: dict[str, int | None] | None = None,
        replacement_mapping: dict[str, str] | None = None,
        max_episode_steps: int = 200,
        forbidden_transitions: set[ForbiddenTransition] | None = None,
        placement_mode: str = "full_random",
    ) -> None:
        if placement_mode not in {"full_random", "regional"}:
            raise ValueError("placement_mode must be 'full_random' or 'regional'.")
        self.placement_mode = placement_mode
        super().__init__(
            max_n=max_n,
            n_rows=n_rows,
            n_cols=n_cols,
            propositions=propositions,
            locations={"A": (1, 1), "C": (1, 4), "D": (4, 1)},
            agent_start_location=agent_start_location,
            max_observation_counts=max_observation_counts,
            replacement_mapping=replacement_mapping,
            max_episode_steps=max_episode_steps,
            forbidden_transitions=forbidden_transitions,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        sampled_n = int(options.get("n", self.np_random.integers(1, self.max_n + 1)))
        self.locations = self._sample_letter_locations()
        self.reset_grid(sampled_n=sampled_n)
        return self.make_observation(), self._info(observed_label=NO_PROPOSITION)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray | int], float, bool, bool, dict[str, Any]]:
        label = self.move_agent(action)
        if label != NO_PROPOSITION:
            self.apply_replacement_if_needed(label)

        truncated = self.n_steps >= self.max_episode_steps
        return self.make_observation(label), 0.0, False, truncated, self._info(observed_label=label)

    def _sample_letter_locations(self) -> dict[str, tuple[int, int]]:
        if self.placement_mode == "regional":
            return self._sample_regional_letter_locations()

        candidates = [
            (row, col)
            for row in range(self.n_rows)
            for col in range(self.n_cols)
            if (row, col) != self.agent_start_location
        ]
        selected = self.np_random.choice(len(candidates), size=3, replace=False)
        a_pos, c_pos, d_pos = (candidates[int(index)] for index in selected)
        return {"A": a_pos, "C": c_pos, "D": d_pos}

    def _sample_regional_letter_locations(self) -> dict[str, tuple[int, int]]:
        regions = {
            "A": [(row, col) for row in range(0, 3) for col in range(0, 3)],
            "C": [(row, col) for row in range(0, 3) for col in range(3, 6)],
            "D": [(row, col) for row in range(3, 6) for col in range(0, 3)],
        }
        return {
            symbol: region[int(self.np_random.integers(0, len(region)))]
            for symbol, region in regions.items()
        }

    def _info(self, *, observed_label: str) -> dict[str, Any]:
        return {
            "proposition_label": observed_label,
            "sampled_n": self.sampled_n,
            "target_positions": {
                "A": self.locations["A"],
                "B": self.locations["A"],
                "C": self.locations["C"],
                "D": self.locations["D"],
            },
            "agent_start_location": self.agent_start_location,
            "placement_mode": self.placement_mode,
        }
