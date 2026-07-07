"""Tabular Q-learning utilities."""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for dictionary-backed Q-learning."""

    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.4
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01


class QLearningAgent:
    """Small tabular Q-learning agent for discrete action spaces."""

    def __init__(
        self,
        actions: Sequence[int],
        config: QLearningConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        if not actions:
            raise ValueError("QLearningAgent requires at least one action.")
        self.actions = tuple(int(action) for action in actions)
        self.config = config or QLearningConfig()
        self.epsilon = float(self.config.epsilon)
        self.rng = rng or random.Random()
        self.q_table: dict[Hashable, dict[int, float]] = {}

    def ensure_state(self, state: Hashable) -> bool:
        """Add a state to the Q-table and return whether it was new."""
        if state in self.q_table:
            return False
        self.q_table[state] = {action: 0.0 for action in self.actions}
        return True

    def choose_action(self, state: Hashable) -> int:
        self.ensure_state(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self._best_action(state)

    def update(self, state: Hashable, action: int, reward: float, next_state: Hashable) -> None:
        self.ensure_state(state)
        self.ensure_state(next_state)
        old_value = self.q_table[state][int(action)]
        next_max = max(self.q_table[next_state].values())
        self.q_table[state][int(action)] = old_value + self.config.alpha * (
            float(reward) + self.config.gamma * next_max - old_value
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def as_serializable(self) -> dict[str, Any]:
        """Return compact agent metadata without serializing the full Q-table."""
        return {
            "state_count": len(self.q_table),
            "epsilon": self.epsilon,
            "actions": list(self.actions),
            "config": {
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "epsilon": self.config.epsilon,
                "epsilon_decay": self.config.epsilon_decay,
                "min_epsilon": self.config.min_epsilon,
            },
        }

    def _best_action(self, state: Hashable) -> int:
        action_values = self.q_table[state]
        max_value = max(action_values.values())
        best_actions = [action for action, value in action_values.items() if value == max_value]
        return self.rng.choice(best_actions)
