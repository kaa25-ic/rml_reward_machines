"""Base LetterEnv gridworld."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


NO_PROPOSITION = "_"
ForbiddenTransition = tuple[int, int, "LetterAction"]


class LetterAction(IntEnum):
    """Actions used by LetterEnv."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class LetterEnv(gym.Env):
    """Counting LetterEnv used by the RML experiments.

    The task is A once, B once, C once, then D repeated ``n`` times. The first
    visit to A reveals the sampled value of ``n`` through the observation's
    ``value`` field. After A is observed once, that cell becomes B.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        max_n: int = 4,
        n_rows: int = 6,
        n_cols: int = 6,
        propositions: tuple[str, ...] = ("A", "B", "C", "D"),
        locations: dict[str, tuple[int, int]] | None = None,
        agent_start_location: tuple[int, int] = (4, 4),
        max_observation_counts: dict[str, int | None] | None = None,
        replacement_mapping: dict[str, str] | None = None,
        task_prefix: str = "ABC",
        counted_suffix: str = "D",
        max_episode_steps: int = 200,
        forbidden_transitions: set[ForbiddenTransition] | None = None,
    ) -> None:
        super().__init__()
        if max_n < 1:
            raise ValueError("max_n must be at least 1.")

        self.max_n = int(max_n)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.propositions = tuple(propositions)
        self.proposition_to_index = {
            proposition: index
            for index, proposition in enumerate((*self.propositions, NO_PROPOSITION))
        }
        self.index_to_proposition = {
            index: proposition for proposition, index in self.proposition_to_index.items()
        }
        self.locations = dict(
            locations
            or {
                "A": (1, 1),
                "C": (1, 4),
                "D": (4, 1),
            }
        )
        self.agent_start_location = tuple(agent_start_location)
        self.max_observation_counts = dict(
            max_observation_counts
            or {
                "A": 1,
                "B": None,
                "C": None,
                "D": None,
            }
        )
        self.replacement_mapping = dict(replacement_mapping or {"A": "B"})
        self.task_prefix = str(task_prefix)
        self.counted_suffix = str(counted_suffix)
        self._validate_task_symbols(self.task_prefix + self.counted_suffix)
        self.max_episode_steps = int(max_episode_steps)
        self.extra_forbidden_transitions = set(forbidden_transitions or set())

        self.action_space = spaces.Discrete(len(LetterAction))
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([self.n_rows - 1, self.n_cols - 1], dtype=np.int32),
                    dtype=np.int32,
                ),
                "proposition": spaces.Discrete(len(self.proposition_to_index)),
                "value": spaces.Discrete(self.max_n + 1),
            }
        )
        self.reward_range = (0, 1)
        self.forbidden_transitions = self._build_forbidden_transitions()

        self.sampled_n = 1
        self.task_string = ""
        self.task_string_idx = 0
        self.task_failed = False
        self.n_steps = 0
        self.active_propositions: dict[tuple[int, int], str] = {}
        self.proposition_observation_counts: dict[str, int] = {}
        self.agent_position = self.agent_start_location

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        fixed_n = None if options is None else options.get("n")
        self.sampled_n = (
            int(fixed_n)
            if fixed_n is not None
            else int(self.np_random.integers(1, self.max_n + 1))
        )
        if not 1 <= self.sampled_n <= self.max_n:
            raise ValueError(f"Episode n must be in 1..{self.max_n}; got {self.sampled_n}.")

        self.task_string = self.task_prefix + self.counted_suffix * self.sampled_n
        self.task_string_idx = 0
        self.task_failed = False
        self.n_steps = 0
        self.active_propositions = {
            tuple(position): proposition for proposition, position in self.locations.items()
        }
        self.proposition_observation_counts = {proposition: 0 for proposition in self.propositions}
        self.agent_position = self.agent_start_location

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray | int], int, bool, bool, dict[str, Any]]:
        action = int(action)
        if action not in [item.value for item in LetterAction]:
            raise ValueError(f"Invalid LetterEnv action: {action}.")

        self.n_steps += 1
        self.agent_position = self._next_position(self.agent_position, LetterAction(action))
        label = self.current_proposition

        if label != NO_PROPOSITION:
            self._record_observation(label)
            self._advance_task(label)
            self._apply_replacement_if_needed(label)

        terminated = self.task_failed or self.task_string_idx == len(self.task_string)
        truncated = not terminated and self.n_steps >= self.max_episode_steps
        reward = int(self.task_string_idx == len(self.task_string) and not self.task_failed)
        return self._observation(label), reward, terminated, truncated, self._info()

    @property
    def current_proposition(self) -> str:
        return self.active_propositions.get(self.agent_position, NO_PROPOSITION)

    def get_true_propositions(self) -> str:
        """Return the proposition currently true at the agent position."""
        return self.current_proposition

    def render(self) -> str:
        rows: list[str] = []
        for row in range(self.n_rows):
            cells: list[str] = []
            for col in range(self.n_cols):
                position = (row, col)
                if position == self.agent_position:
                    cells.append("x")
                else:
                    cells.append(self.active_propositions.get(position, "."))
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _observation(self, observed_label: str | None = None) -> dict[str, np.ndarray | int]:
        label = self.current_proposition if observed_label is None else observed_label
        value = self.sampled_n if label == "A" else 0
        return {
            "position": np.asarray(self.agent_position, dtype=np.int32),
            "proposition": self.proposition_to_index[label],
            "value": value,
        }

    def _info(self) -> dict[str, Any]:
        success = self.task_string_idx == len(self.task_string) and not self.task_failed
        return {
            "proposition_label": self.current_proposition,
            "sampled_n": self.sampled_n,
            "task_string": self.task_string,
            "task_index": self.task_string_idx,
            "task_failed": self.task_failed,
            "success": success,
        }

    def _record_observation(self, label: str) -> None:
        self.proposition_observation_counts[label] = (
            self.proposition_observation_counts.get(label, 0) + 1
        )

    def _advance_task(self, label: str) -> None:
        if self.task_string_idx >= len(self.task_string):
            return
        if label == self.task_string[self.task_string_idx]:
            self.task_string_idx += 1
        else:
            self.task_failed = True

    def _apply_replacement_if_needed(self, label: str) -> None:
        max_count = self.max_observation_counts.get(label)
        if max_count is None:
            return
        if self.proposition_observation_counts[label] == max_count:
            replacement = self.replacement_mapping.get(label)
            if replacement is not None:
                self.active_propositions[self.agent_position] = replacement

    def _next_position(self, position: tuple[int, int], action: LetterAction) -> tuple[int, int]:
        row, col = position
        if (row, col, action) in self.forbidden_transitions:
            return position
        if action == LetterAction.RIGHT:
            return row, col + 1
        if action == LetterAction.LEFT:
            return row, col - 1
        if action == LetterAction.UP:
            return row - 1, col
        if action == LetterAction.DOWN:
            return row + 1, col
        raise ValueError(f"Unsupported action: {action}.")

    def _build_forbidden_transitions(self) -> set[tuple[int, int, LetterAction]]:
        forbidden: set[tuple[int, int, LetterAction]] = set(self.extra_forbidden_transitions)
        for col in range(self.n_cols):
            forbidden.add((0, col, LetterAction.UP))
            forbidden.add((self.n_rows - 1, col, LetterAction.DOWN))
        for row in range(self.n_rows):
            forbidden.add((row, 0, LetterAction.LEFT))
            forbidden.add((row, self.n_cols - 1, LetterAction.RIGHT))
        return forbidden

    def _validate_task_symbols(self, task_symbols: str) -> None:
        unknown_symbols = sorted(set(task_symbols) - set(self.propositions))
        if unknown_symbols:
            raise ValueError(
                "LetterEnv task contains symbols that are not propositions: "
                + ", ".join(unknown_symbols)
            )
