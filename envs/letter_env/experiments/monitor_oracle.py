"""Reward and transition oracle over LetterEnv's reachable monitor states.

The oracle drives the live RML monitor to enumerate the reachable monitor
states and record, for every state and event, the successor state and the
reward emitted by the monitor. It supplies the transition and reward oracles
required to audit an encoding for sufficiency beyond injectivity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rml_rm.encodings.rml_graph import normalize_generated_variables
from rml_rm.monitors.process import RMLMonitorProcess, find_free_port
from rml_rm.monitors.transaction import (
    WebSocketMonitorClient,
    load_monitor_config,
    reset_monitor,
    rewards_from_config,
    step_monitor,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = PACKAGE_ROOT / "specs" / "letter_env_monitor.pl"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "letter_env.yaml"

TERMINAL_STATES = frozenset({"1", "false_verdict"})


@dataclass(frozen=True)
class LetterEnvMonitorOracle:
    """Static reward and transition tables over reachable monitor states."""

    states: tuple[str, ...]
    events: tuple[str, ...]
    representatives: dict[str, str] = field(repr=False)
    transitions: dict[tuple[str, str], str] = field(repr=False)
    rewards: dict[str, float] = field(repr=False)
    transition_rewards: dict[tuple[str, str], float] = field(repr=False)

    def encoder_input(self, state: str) -> str:
        """Return the raw monitor string an encoder receives for ``state``."""
        return self.representatives[state]

    def transition(self, state: str, event: str) -> str | None:
        return self.transitions.get((state, event))

    def reward(self, state: str) -> float:
        return self.rewards[state]

    def transition_reward(self, state: str, event: str) -> float | None:
        """Return the reward emitted for firing ``event`` in ``state``."""
        return self.transition_rewards.get((state, event))


def build_letter_env_monitor_oracle(
    *,
    max_n: int = 5,
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> LetterEnvMonitorOracle:
    """Enumerate the reachable monitor states and build their oracle tables."""
    config = load_monitor_config(config_path)
    variables = list(config["variables"])
    rewards = rewards_from_config(config)
    events = tuple([f"A{count}" for count in range(1, max_n + 1)] + ["B", "C", "D", "O"])

    port = find_free_port()
    with RMLMonitorProcess(spec_path=Path(spec_path), port=port) as monitor:
        client = WebSocketMonitorClient(host=monitor.host, port=port)
        representatives, transitions, state_rewards, transition_rewards = _explore(
            client, variables, rewards, events
        )

    return LetterEnvMonitorOracle(
        states=tuple(representatives),
        events=events,
        representatives=representatives,
        transitions=transitions,
        rewards=state_rewards,
        transition_rewards=transition_rewards,
    )


def _event_payload(variables, event: str) -> dict[str, object]:
    indicators = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    if event.startswith("A"):
        indicators["a"] = float(event[1:])
    elif event in ("B", "C", "D"):
        indicators[event.lower()] = 1.0
    payload: dict[str, object] = {"location": "obs"}
    for variable in variables:
        name = str(variable["name"])
        payload[name] = indicators.get(name, 0.0)
    return payload


def _replay(client, variables, rewards, path):
    reset_monitor(client, variables)
    result = None
    for event in path:
        result = step_monitor(client, _event_payload(variables, event), rewards)
    return result


def _explore(client, variables, rewards, events):
    seed = _replay(client, variables, rewards, ["O"])
    start = normalize_generated_variables(seed.monitor_state)

    representatives = {start: seed.monitor_state}
    state_rewards = {start: seed.base_reward}
    path_to = {start: ["O"]}
    transitions: dict[tuple[str, str], str] = {}
    transition_rewards: dict[tuple[str, str], float] = {}
    frontier = deque([start])

    while frontier:
        state = frontier.popleft()
        if state in TERMINAL_STATES:
            continue
        for event in events:
            result = _replay(client, variables, rewards, path_to[state] + [event])
            successor = normalize_generated_variables(result.monitor_state)
            transitions[(state, event)] = successor
            transition_rewards[(state, event)] = result.base_reward
            if successor not in representatives:
                representatives[successor] = result.monitor_state
                state_rewards[successor] = result.base_reward
                path_to[successor] = path_to[state] + [event]
                frontier.append(successor)

    return representatives, transitions, state_rewards, transition_rewards
