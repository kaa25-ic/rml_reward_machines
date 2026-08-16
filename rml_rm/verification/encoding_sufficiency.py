"""Audit of monitor-state encoding sufficiency.

An encoding is sufficient on a set of reachable monitor states when, for every
pair of states that receive the same encoded value and every monitor event, the
two states emit the same reward and reach states with the same encoded value.
Both conditions are checked over transitions: the reward emitted for acting in a
state is the reward of the successor reached under the event. An encoding that is
injective on the reachable states is sufficient by construction.

The reward condition is stated over ``R((s, u), a)`` but is verified over monitor
states alone. This is valid whenever the reward is separable into an
environment-only term and a monitor-only term, since the environment term is
compared at a fixed ``(s, a)`` and cancels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


Encoder = Callable[[str], np.ndarray]
TransitionOracle = Callable[[str, str], "str | None"]
TransitionRewardOracle = Callable[[str, str], "float | None"]

_TERMINAL_KEY: tuple[str, ...] = ("__terminal__",)


def encoding_key(vector: np.ndarray, *, decimals: int = 9) -> tuple[float, ...]:
    """Return a hashable, tolerance-rounded key for an encoded vector."""
    array = np.asarray(vector, dtype=np.float64).ravel()
    return tuple(np.round(array, decimals).tolist())


@dataclass(frozen=True)
class Witness:
    """A pair of merged states that violates a sufficiency assumption."""

    kind: str
    states: tuple[str, str]
    event: str | None = None


@dataclass(frozen=True)
class EncodingAudit:
    """Outcome of auditing one encoding on a set of reachable states."""

    injective: bool
    sufficient: bool | None
    groups: tuple[tuple[str, ...], ...]
    witnesses: tuple[Witness, ...] = ()

    @property
    def checked(self) -> bool:
        return self.sufficient is not None


def partition_by_encoding(
    states: Sequence[str], encoder: Encoder, *, decimals: int = 9
) -> list[list[str]]:
    """Group monitor states that share an encoded value."""
    groups: dict[tuple[float, ...], list[str]] = {}
    for state in states:
        key = encoding_key(encoder(state), decimals=decimals)
        groups.setdefault(key, []).append(state)
    return list(groups.values())


def audit_encoding(
    states: Sequence[str],
    encoder: Encoder,
    *,
    events: Sequence[str] | None = None,
    transition: TransitionOracle | None = None,
    transition_reward: TransitionRewardOracle | None = None,
    decimals: int = 9,
) -> EncodingAudit:
    """Audit an encoding for reward and transition-closure sufficiency.

    Without a transition oracle a non-injective encoding cannot be verified and
    the audit is inconclusive (``sufficient`` is ``None``). Given the transition
    oracle and events, every non-singleton group is checked over transitions:
    transition closure always, and reward closure when a transition-reward
    oracle is supplied. ``transition_reward(u, e)`` is the reward emitted for
    firing event ``e`` in state ``u`` (``None`` when no transition applies). The
    encoding is sufficient exactly when no witness is produced.
    """
    groups = partition_by_encoding(states, encoder, decimals=decimals)
    frozen_groups = tuple(tuple(group) for group in groups)
    injective = all(len(group) == 1 for group in groups)

    if injective:
        return EncodingAudit(True, True, frozen_groups, ())

    if transition is None or not events:
        return EncodingAudit(False, None, frozen_groups, ())

    witnesses: list[Witness] = []
    for group in groups:
        if len(group) == 1:
            continue
        witnesses.extend(
            _closure_witnesses(group, encoder, events, transition, transition_reward, decimals)
        )

    if transition_reward is None:
        sufficient = False if witnesses else None
    else:
        sufficient = len(witnesses) == 0
    return EncodingAudit(False, sufficient, frozen_groups, tuple(witnesses))


def _closure_witnesses(
    group: Sequence[str],
    encoder: Encoder,
    events: Sequence[str],
    transition: TransitionOracle,
    transition_reward: TransitionRewardOracle | None,
    decimals: int,
) -> list[Witness]:
    witnesses: list[Witness] = []
    for event in events:
        keys = [_successor_key(transition(state, event), encoder, decimals) for state in group]
        rewards = (
            [transition_reward(state, event) for state in group]
            if transition_reward is not None
            else None
        )
        for index in range(1, len(group)):
            if keys[index] != keys[0]:
                witnesses.append(Witness("transition", (group[0], group[index]), event))
            if (
                rewards is not None
                and rewards[0] is not None
                and rewards[index] is not None
                and rewards[0] != rewards[index]
            ):
                witnesses.append(Witness("reward", (group[0], group[index]), event))
    return witnesses


def _successor_key(
    successor: str | None, encoder: Encoder, decimals: int
) -> tuple[float, ...] | tuple[str, ...]:
    if successor is None:
        return _TERMINAL_KEY
    return encoding_key(encoder(successor), decimals=decimals)
