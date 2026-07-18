"""Hand-coded reward-machine wrapper for the CSTR startup task."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from envs.cstr.env import make_cstr_env
from envs.cstr.builder import (
    CSTRSemanticProgressEncoder,
    RMLCSTRConfig,
    RMLCSTREnv,
    _truthy,
)
from envs.cstr.reference_automaton import ReferenceStartupAutomaton


class ManualRMCSTREnv(RMLCSTREnv):
    """Pure-Python startup reward machine with the same rewards as RMLCSTREnv.

    This wrapper intentionally avoids the external RML monitor. It uses the
    reference automaton directly, which makes it a strong non-strawman baseline:
    the agent receives the same procedural reward signal, but the task logic is
    hand-coded in Python rather than represented by an RML specification.
    """

    def __init__(self, config: RMLCSTRConfig | None = None) -> None:
        self.config = config or RMLCSTRConfig()
        if self.config.observation_mode not in {"none", "semantic_progress"}:
            raise ValueError(
                "ManualRMCSTREnv supports observation_mode 'none' or 'semantic_progress', "
                f"got {self.config.observation_mode!r}."
            )

        env = make_cstr_env(self.config.cstr_env)
        super(RMLCSTREnv, self).__init__(env)

        self.automaton = ReferenceStartupAutomaton(
            soak_steps=self.config.soak_steps,
            recover_from_regulation_failure=self.config.recover_from_regulation_failure,
        )
        self.monitor_state_unencoded = "Preheat"
        self.last_verdict = "currently_false"
        self.monitor_phase = "preheat"
        self.previous_monitor_phase = "preheat"
        self.monitor_soak_steps = 0
        self.previous_monitor_soak_steps = 0
        self.max_rewarded_soak_steps = 0
        self.has_entered_soak = False
        self.has_entered_regulate = False
        self.previous_approach_distance: float | None = None
        self.previous_approach_ca_error: float | None = None
        self.previous_approach_temp_error: float | None = None
        self.rml_state_encoder = CSTRSemanticProgressEncoder(
            max_states=self.config.monitor_state_limit,
            soak_steps=self.config.soak_steps,
        )
        self.monitor_state = self._initial_monitor_state()
        self.previous_monitor_state = self.monitor_state.copy()

        if self.config.observation_mode == "none":
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent": self.env.observation_space,
                    "monitor": self._monitor_observation_space(),
                }
            )

    def reset(self, **kwargs: Any):
        self.automaton.reset()
        self.monitor_state_unencoded = "Preheat"
        self.last_verdict = "currently_false"
        self.monitor_phase = "preheat"
        self.previous_monitor_phase = "preheat"
        self.monitor_soak_steps = 0
        self.previous_monitor_soak_steps = 0
        self.max_rewarded_soak_steps = 0
        self.has_entered_soak = False
        self.has_entered_regulate = False
        self.previous_approach_distance = None
        self.previous_approach_ca_error = None
        self.previous_approach_temp_error = None
        self.rml_state_encoder.reset()
        self.monitor_state = self._initial_monitor_state()
        self.previous_monitor_state = self.monitor_state.copy()
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._monitor_info(rml_reward=0.0))
        return self._with_monitor(observation), info

    def step(self, action: Any):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        payload = self._payload_from_info(info, terminate=bool(truncated))
        monitor_info = self._monitor_step(payload)
        rml_reward = self._rml_reward(info)
        reward = self._combined_reward(base_reward=float(base_reward), rml_reward=rml_reward)
        terminated = bool(terminated or self._rml_termination())
        monitor_info.update(self._monitor_info(rml_reward=rml_reward))
        info.update(monitor_info)
        return self._with_monitor(observation), reward, terminated, truncated, info

    def _monitor_step(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.previous_monitor_phase = self.monitor_phase
        previous_soak_steps = self.monitor_soak_steps
        self.previous_monitor_soak_steps = previous_soak_steps

        reference_verdict = self.automaton.step(payload)
        reference_state = self.automaton.canonical_state
        self.last_verdict = _verdict_to_monitor_label(reference_verdict)
        self.monitor_state_unencoded = reference_state
        self.monitor_phase, self.monitor_soak_steps = _phase_and_soak_steps(reference_state)

        encoded_monitor_state = self._encode_monitor_state(
            canonical_state=reference_state,
            raw_state=self.monitor_state_unencoded,
        )
        monitor_changed = not np.array_equal(encoded_monitor_state, self.previous_monitor_state)
        self.monitor_state = encoded_monitor_state
        self.previous_monitor_state = copy.deepcopy(encoded_monitor_state)
        return {
            "monitor_verdict": self.last_verdict,
            "monitor_state_unencoded": self.monitor_state_unencoded,
            "monitor_changed": bool(monitor_changed),
            "reference_verdict": reference_verdict,
            "reference_state": reference_state,
            "monitor_consistent": True,
            "monitor_consistency_mismatches": 0,
            "manual_reward_machine": True,
        }

    def _payload_from_info(self, info: Mapping[str, Any], *, terminate: bool) -> dict[str, Any]:
        return {
            "time": [],
            "action": [],
            "terminate": bool(terminate),
            "critical": float(_truthy(info.get("event_temp_critical", False))),
            "temp_safe": float(_truthy(info.get("event_temp_safe", False))),
            "stable": float(_truthy(info.get("event_stable_step", False))),
            "in_soak_band": float(_truthy(info.get("event_in_soak_band", False))),
            "overshoot": float(_truthy(info.get("event_overshoot", False))),
            "past_deadline": float(_truthy(info.get("event_past_deadline", False))),
            "heating_rate_exceeded": float(_truthy(info.get("event_heating_rate_exceeded", False))),
        }

    def _monitor_info(self, *, rml_reward: float) -> dict[str, Any]:
        info = super()._monitor_info(rml_reward=rml_reward)
        info["monitor_encoding_description"] = (
            ()
            if self.config.observation_mode == "none"
            else ("manual_reward_machine_semantic_progress",)
        )
        info["manual_reward_machine"] = True
        return info


def make_manual_rm_cstr_env(config: RMLCSTRConfig | None = None) -> ManualRMCSTREnv:
    """Create the hand-coded reward-machine CSTR wrapper."""

    return ManualRMCSTREnv(config=config)


def _phase_and_soak_steps(canonical_state: str) -> tuple[str, int]:
    state = str(canonical_state).strip().lower()
    if state.startswith("soak_"):
        return "soak", int(state.split("_", maxsplit=1)[1])
    return state, 0


def _verdict_to_monitor_label(reference_verdict: str) -> str:
    if reference_verdict == "accept":
        return "true"
    if reference_verdict == "reject":
        return "false"
    return "currently_false"
