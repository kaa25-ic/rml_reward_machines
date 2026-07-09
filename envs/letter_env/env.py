"""Single-task LetterEnv gridworld."""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.letter_env_core import NO_PROPOSITION, ForbiddenTransition, LetterAction, LetterGridWorld


class LetterEnv(LetterGridWorld):
    """Counting LetterEnv used by the RML experiments.

    The task is A once, B once, C once, then D repeated ``n`` times. The first
    visit to A reveals the sampled value of ``n`` through the observation's
    ``value`` field. After A is observed once, that cell becomes B.
    """

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
        super().__init__(
            max_n=max_n,
            n_rows=n_rows,
            n_cols=n_cols,
            propositions=propositions,
            locations=locations,
            agent_start_location=agent_start_location,
            max_observation_counts=max_observation_counts,
            replacement_mapping=replacement_mapping,
            max_episode_steps=max_episode_steps,
            forbidden_transitions=forbidden_transitions,
        )
        self.task_prefix = str(task_prefix)
        self.counted_suffix = str(counted_suffix)
        self._validate_task_symbols(self.task_prefix + self.counted_suffix)
        self.task_string = ""
        self.task_string_idx = 0
        self.task_failed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | int], dict[str, Any]]:
        super().reset(seed=seed)
        fixed_n = None if options is None else options.get("n")
        sampled_n = (
            int(fixed_n)
            if fixed_n is not None
            else int(self.np_random.integers(1, self.max_n + 1))
        )
        self.reset_grid(sampled_n=sampled_n)

        self.task_string = self.task_prefix + self.counted_suffix * self.sampled_n
        self.task_string_idx = 0
        self.task_failed = False

        return self.make_observation(), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray | int], int, bool, bool, dict[str, Any]]:
        label = self.move_agent(action)

        if label != NO_PROPOSITION:
            self._advance_task(label)
            self.apply_replacement_if_needed(label)

        terminated = self.task_failed or self.task_string_idx == len(self.task_string)
        truncated = not terminated and self.n_steps >= self.max_episode_steps
        reward = int(self.task_string_idx == len(self.task_string) and not self.task_failed)
        return self.make_observation(label), reward, terminated, truncated, self._info()

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

    def _advance_task(self, label: str) -> None:
        if self.task_string_idx >= len(self.task_string):
            return
        if label == self.task_string[self.task_string_idx]:
            self.task_string_idx += 1
        else:
            self.task_failed = True

    def _validate_task_symbols(self, task_symbols: str) -> None:
        unknown_symbols = sorted(set(task_symbols) - set(self.propositions))
        if unknown_symbols:
            raise ValueError(
                "LetterEnv task contains symbols that are not propositions: "
                + ", ".join(unknown_symbols)
            )
