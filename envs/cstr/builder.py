"""RML-backed wrapper for the CSTR environment."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from envs.cstr.encodings import CSTRFrozenGraphMonitorStateEncoder, CSTRSemanticProgressEncoder
from envs.cstr.env import CSTRConfig, make_cstr_env
from envs.cstr.rml_generation import CONFIGS_ROOT, generate_cstr_rml
from envs.cstr.reference_automaton import (
    ReferenceStartupAutomaton,
    verdict_matches_monitor,
)
from rml_rm.monitors import MonitorClient, WebSocketMonitorClient
from rml_rm.wrappers import RMLMonitorWrapper


CSTRObservationMode = Literal["none", "semantic_progress", "rml_graph"]
CSTRRewardMode = Literal["env", "rml", "env_rml"]
SUPPORTED_OBSERVATION_MODES = ("none", "semantic_progress", "rml_graph")
SUPPORTED_REWARD_MODES = ("env", "rml", "env_rml")
STREAM_ENCODING_DESCRIPTIONS = {
    "none": (),
    "semantic_progress": ("rml_monitor_state_string_semantic_progress",),
    "rml_graph": ("frozen_cstr_rml_graph_embedding",),
}


class CSTRAgentObservationWrapper(gym.Wrapper):
    """Expose native CSTR observations in the dict form used by monitor wrappers."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Dict({"agent": env.observation_space})

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        return {"agent": np.asarray(observation, dtype=np.float32)}, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            {"agent": np.asarray(observation, dtype=np.float32)},
            reward,
            terminated,
            truncated,
            info,
        )


@dataclass(frozen=True)
class RMLCSTRConfig:
    """Configuration for the RML-backed CSTR wrapper."""

    cstr_env: CSTRConfig = CSTRConfig()
    observation_mode: CSTRObservationMode = "semantic_progress"
    reward_mode: CSTRRewardMode = "env_rml"
    config_path: Path = CONFIGS_ROOT / "cstr_startup_procedure.yaml"
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 18_401
    regulation_violation_steps: int = 10
    soak_steps: int = 10
    monitor_state_limit: int = 16
    graph_encoder_checkpoint: Path | None = None
    terminate_on_rml_success: bool = True
    terminate_on_rml_failure: bool = True
    recover_from_regulation_failure: bool = False
    assert_monitor_consistency: bool = False
    safe_step_bonus: float = 0.10
    stable_step_bonus: float = 1.0
    regulation_entry_bonus: float = 5.0
    success_bonus: float = 50.0
    failure_penalty: float = -50.0
    heating_rate_penalty: float = 0.02
    preheat_distance_weight: float = 0.08
    preheat_warming_weight: float = 0.25
    soak_entry_bonus: float = 5.0
    soak_progress_bonus: float = 0.75
    soak_reset_penalty: float = -3.0
    soak_lost_step_penalty: float = 0.50
    approach_distance_weight: float = 1.0
    approach_progress_bonus: float = 5.0
    approach_ca_progress_bonus: float = 4.0
    approach_temp_progress_bonus: float = 4.0
    approach_warming_weight: float = 0.50
    production_entry_bonus: float = 10.0
    regulate_recovery_penalty: float = -10.0


class RMLCSTREnv(gym.Wrapper):
    """Attach an RML temporal-task monitor to the native CSTR environment."""

    def __init__(
        self,
        config: RMLCSTRConfig | None = None,
        *,
        client: MonitorClient | None = None,
    ) -> None:
        self.config = config or RMLCSTRConfig()
        if self.config.observation_mode not in SUPPORTED_OBSERVATION_MODES:
            raise ValueError(f"Unsupported observation mode: {self.config.observation_mode!r}.")
        if self.config.reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {self.config.reward_mode!r}.")

        default_config_path = CONFIGS_ROOT / "cstr_startup_procedure.yaml"
        if self.config.config_path == default_config_path:
            generated = generate_cstr_rml(
                regulation_violation_steps=self.config.regulation_violation_steps,
                soak_steps=self.config.soak_steps,
                recover_from_regulation_failure=self.config.recover_from_regulation_failure,
                host=self.config.monitor_host,
                port=self.config.monitor_port,
                max_episode_steps=self.config.cstr_env.max_episode_steps,
            )
            config_path = generated.config_path
        else:
            config_path = self.config.config_path
        self.rml_config = _load_rml_config(config_path)
        self.rml_variables = list(self.rml_config["variables"])
        monitor_client = client or WebSocketMonitorClient(
            host=str(self.rml_config.get("host", self.config.monitor_host)),
            port=int(self.rml_config.get("port", self.config.monitor_port)),
        )
        raw_env = CSTRAgentObservationWrapper(make_cstr_env(self.config.cstr_env))
        env = RMLMonitorWrapper(
            raw_env,
            config_path=config_path,
            monitor_encoder=lambda _state: np.zeros(0, dtype=np.float32),
            initial_monitor_state=np.zeros(0, dtype=np.float32),
            monitor_space=spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32),
            client=monitor_client,
            include_transition_bonus=False,
            terminal_monitor_states=set(),
        )
        super().__init__(env)

        self.monitor_state_unencoded = ""
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
        if self.config.observation_mode == "rml_graph":
            if self.config.graph_encoder_checkpoint is None:
                raise ValueError("graph_encoder_checkpoint is required when observation_mode='rml_graph'.")
            self.rml_state_encoder = CSTRFrozenGraphMonitorStateEncoder(self.config.graph_encoder_checkpoint)
        else:
            self.rml_state_encoder = CSTRSemanticProgressEncoder(
                max_states=self.config.monitor_state_limit,
                soak_steps=self.config.soak_steps,
            )
        self.reference_automaton = (
            ReferenceStartupAutomaton(
                soak_steps=self.config.soak_steps,
                recover_from_regulation_failure=self.config.recover_from_regulation_failure,
            )
            if self.config.assert_monitor_consistency
            else None
        )
        self.monitor_consistency_checks = 0
        self.monitor_consistency_mismatches = 0
        self.monitor_state = self._initial_monitor_state()
        self.previous_monitor_state = self.monitor_state.copy()

        if self.config.observation_mode == "none":
            self.observation_space = self.env.env.observation_space.spaces["agent"]
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent": self.env.env.observation_space.spaces["agent"],
                    "monitor": self._monitor_observation_space(),
                }
            )

    def reset(self, **kwargs: Any):
        self.monitor_state_unencoded = ""
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
        if self.reference_automaton is not None:
            self.reference_automaton.reset()
        self.rml_state_encoder.reset()
        self.monitor_state = self._initial_monitor_state()
        self.previous_monitor_state = self.monitor_state.copy()
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._monitor_info(rml_reward=0.0))
        return self._with_monitor(observation), info

    def step(self, action: Any):
        observation, _monitor_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        payload = dict(getattr(self.env, "data", {}))
        monitor_info = self._monitor_step(payload, info)
        rml_reward = self._rml_reward(info)
        reward = self._combined_reward(base_reward=float(info.get("base_reward", 0.0)), rml_reward=rml_reward)
        terminated = bool(terminated or self._rml_termination())
        monitor_info.update(self._monitor_info(rml_reward=rml_reward))
        info.update(monitor_info)
        return self._with_monitor(observation), reward, terminated, truncated, info

    def _monitor_step(self, payload: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
        self.last_verdict = normalize_verdict(str(info.get("monitor_verdict", "currently_false")))
        self.monitor_state_unencoded = str(info.get("monitor_state_unencoded", ""))
        self.previous_monitor_phase = self.monitor_phase
        previous_soak_steps = self.monitor_soak_steps
        self.previous_monitor_soak_steps = previous_soak_steps
        self.monitor_phase = _classify_cstr_rml_state(
            self.monitor_state_unencoded,
            verdict=self.last_verdict,
            payload=payload,
            previous_phase=self.previous_monitor_phase,
            previous_soak_steps=previous_soak_steps,
            soak_steps=self.config.soak_steps,
            recover_from_regulation_failure=self.config.recover_from_regulation_failure,
        )
        self.monitor_soak_steps = _next_soak_steps(
            phase=self.monitor_phase,
            payload=payload,
            previous_phase=self.previous_monitor_phase,
            previous_soak_steps=previous_soak_steps,
            soak_steps=self.config.soak_steps,
        )
        canonical_state = _canonical_monitor_state(
            phase=self.monitor_phase,
            soak_steps=self.monitor_soak_steps,
        )
        encoded_monitor_state = self._encode_monitor_state(
            canonical_state=canonical_state,
            raw_state=self.monitor_state_unencoded,
        )
        monitor_changed = not np.array_equal(encoded_monitor_state, self.previous_monitor_state)
        self.monitor_state = encoded_monitor_state
        self.previous_monitor_state = copy.deepcopy(encoded_monitor_state)
        step_info = {
            "monitor_verdict": self.last_verdict,
            "monitor_state_unencoded": self.monitor_state_unencoded,
            "monitor_changed": bool(monitor_changed),
        }
        if self.reference_automaton is not None:
            reference_verdict = self.reference_automaton.step(payload)
            consistent = verdict_matches_monitor(
                reference_verdict,
                self.last_verdict,
                normalize_monitor_state(self.monitor_state_unencoded),
            )
            self.monitor_consistency_checks += 1
            self.monitor_consistency_mismatches += int(not consistent)
            step_info.update(
                {
                    "reference_verdict": reference_verdict,
                    "reference_state": self.reference_automaton.canonical_state,
                    "monitor_consistent": bool(consistent),
                    "monitor_consistency_mismatches": int(self.monitor_consistency_mismatches),
                }
            )
        return step_info

    def _rml_reward(self, info: Mapping[str, Any]) -> float:
        reward = 0.0
        if self.monitor_phase == "failure" and self.previous_monitor_phase != "failure":
            reward += float(self.config.failure_penalty)
        if self.monitor_phase == "success" and self.previous_monitor_phase != "success":
            reward += float(self.config.success_bonus)
        if _truthy(info.get("event_heating_rate_exceeded", False)):
            reward -= float(self.config.heating_rate_penalty)
        if self.monitor_phase in {"soak", "approach", "regulate"}:
            reward += float(self.config.safe_step_bonus)
        if self.monitor_phase == "preheat":
            reward += self._preheat_reward(info)
        if self.monitor_phase == "soak":
            reward += 0.5 * float(self.config.stable_step_bonus)
            reward += float(self.config.soak_progress_bonus) * (
                float(self.monitor_soak_steps) / max(float(self.config.soak_steps), 1.0)
            )
            self.max_rewarded_soak_steps = max(self.max_rewarded_soak_steps, self.monitor_soak_steps)
        if self.previous_monitor_phase == "soak" and self.monitor_phase == "preheat":
            reward += float(self.config.soak_reset_penalty)
            reward -= float(self.config.soak_lost_step_penalty) * float(self.previous_monitor_soak_steps)
            self._reset_approach_progress()
        if self.monitor_phase == "approach":
            reward += self._approach_reward(info)
        if self.monitor_phase == "regulate":
            reward += float(self.config.stable_step_bonus)
            self._reset_approach_progress()
        if self.previous_monitor_phase == "regulate" and self.monitor_phase == "approach":
            reward += float(self.config.regulate_recovery_penalty)
        if (
            self.monitor_phase == "soak"
            and self.previous_monitor_phase == "preheat"
            and not self.has_entered_soak
        ):
            reward += float(self.config.soak_entry_bonus)
            self.has_entered_soak = True
        if self.monitor_phase == "approach" and self.previous_monitor_phase == "soak":
            reward += 0.5 * float(self.config.regulation_entry_bonus)
        if (
            self.monitor_phase == "regulate"
            and self.previous_monitor_phase == "approach"
            and not self.has_entered_regulate
        ):
            reward += float(self.config.regulation_entry_bonus)
            reward += float(self.config.production_entry_bonus)
            self.has_entered_regulate = True
        return float(reward)

    def _preheat_reward(self, info: Mapping[str, Any]) -> float:
        temp = float(info.get("reactor_temperature", 0.0))
        heat_rate = float(info.get("heating_rate", 0.0))
        target = 0.5 * (self.config.cstr_env.soak_band_low + self.config.cstr_env.soak_band_high)
        distance = max(0.0, target - temp)
        reward = -float(self.config.preheat_distance_weight) * distance
        reward += float(self.config.preheat_warming_weight) * max(0.0, heat_rate)
        return float(reward)

    def _approach_reward(self, info: Mapping[str, Any]) -> float:
        ca = float(info.get("reactor_concentration", 0.0))
        temp = float(info.get("reactor_temperature", 0.0))
        ca_setpoint = float(info.get("target_concentration", self.config.cstr_env.ca_setpoint))
        ca_error = abs(ca - ca_setpoint) / max(self.config.cstr_env.concentration_error_scale, 1e-9)
        temp_error = abs(temp - self.config.cstr_env.target_temp) / max(
            self.config.cstr_env.temperature_error_scale,
            1e-9,
        )
        distance = float(ca_error + temp_error)
        reward = -float(self.config.approach_distance_weight) * distance
        if self.previous_approach_distance is not None:
            reward += float(self.config.approach_progress_bonus) * max(
                0.0,
                self.previous_approach_distance - distance,
            )
        if self.previous_approach_ca_error is not None:
            reward += float(self.config.approach_ca_progress_bonus) * max(
                0.0,
                self.previous_approach_ca_error - ca_error,
            )
        if self.previous_approach_temp_error is not None:
            reward += float(self.config.approach_temp_progress_bonus) * max(
                0.0,
                self.previous_approach_temp_error - temp_error,
            )
        if temp < self.config.cstr_env.target_temp:
            reward += float(self.config.approach_warming_weight) * max(
                0.0,
                float(info.get("heating_rate", 0.0)),
            )
        self.previous_approach_distance = distance
        self.previous_approach_ca_error = ca_error
        self.previous_approach_temp_error = temp_error
        return float(reward)

    def _reset_approach_progress(self) -> None:
        self.previous_approach_distance = None
        self.previous_approach_ca_error = None
        self.previous_approach_temp_error = None

    def _combined_reward(self, *, base_reward: float, rml_reward: float) -> float:
        if self.config.reward_mode == "env":
            return base_reward
        if self.config.reward_mode == "rml":
            return rml_reward
        if self.config.reward_mode == "env_rml":
            return base_reward + rml_reward
        raise ValueError(f"Unsupported reward mode: {self.config.reward_mode!r}.")

    def _rml_termination(self) -> bool:
        if self.config.terminate_on_rml_success and self.monitor_phase == "success":
            return True
        if self.config.terminate_on_rml_failure and self.monitor_phase == "failure":
            return True
        return False

    def _payload_from_info(self, info: Mapping[str, Any], *, terminate: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"time": [], "action": [], "terminate": bool(terminate)}
        for variable in self.rml_variables:
            if variable["location"] != "info":
                raise ValueError(f"CSTR RML only supports info variables, got {variable!r}.")
            payload[variable["name"]] = float(_truthy(info[variable["identifier"]]))
        return payload

    def _with_monitor(self, observation: Any) -> Any:
        agent_observation = (
            observation["agent"]
            if isinstance(observation, Mapping) and "agent" in observation
            else observation
        )
        if self.config.observation_mode == "none":
            return np.asarray(agent_observation, dtype=np.float32)
        return {
            "agent": np.asarray(agent_observation, dtype=np.float32),
            "monitor": self.monitor_state.astype(np.float32),
        }

    def _initial_monitor_state(self) -> np.ndarray:
        if self.config.observation_mode == "none":
            return np.zeros(0, dtype=np.float32)
        if self.config.observation_mode == "semantic_progress":
            return self.rml_state_encoder.encode(CSTRSemanticProgressEncoder.initial_state)
        if self.config.observation_mode == "rml_graph":
            return self.rml_state_encoder.encode("")
        raise ValueError(f"Unsupported observation mode: {self.config.observation_mode!r}.")

    def _encode_monitor_state(self, *, canonical_state: str, raw_state: str) -> np.ndarray:
        if self.config.observation_mode == "none":
            return np.zeros(0, dtype=np.float32)
        if self.config.observation_mode == "semantic_progress":
            return self.rml_state_encoder.encode(canonical_state)
        if self.config.observation_mode == "rml_graph":
            return self.rml_state_encoder.encode(raw_state)
        raise ValueError(f"Unsupported observation mode: {self.config.observation_mode!r}.")

    def _monitor_observation_space(self) -> spaces.Box:
        if self.config.observation_mode == "semantic_progress":
            low = np.zeros(self.monitor_state.shape, dtype=np.float32)
            high = np.ones(self.monitor_state.shape, dtype=np.float32)
        else:
            low = np.full(self.monitor_state.shape, -np.inf, dtype=np.float32)
            high = np.full(self.monitor_state.shape, np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=self.monitor_state.shape, dtype=np.float32)

    def _monitor_info(self, *, rml_reward: float) -> dict[str, Any]:
        return {
            "monitor_reward": float(rml_reward),
            "rml_reward": float(rml_reward),
            "monitor_phase": self.monitor_phase,
            "monitor_success": self.monitor_phase == "success",
            "monitor_failed": self.monitor_phase == "failure",
            "monitor_encoding": self.config.observation_mode,
            "monitor_encoding_description": STREAM_ENCODING_DESCRIPTIONS[self.config.observation_mode],
            "monitor_encoding_size": int(self.monitor_state.shape[0]),
            "rml_monitor_state_count": int(getattr(self.rml_state_encoder, "state_count", 0)),
            "rml_monitor_state_id": int(self.rml_state_encoder.current_state_id),
            "rml_monitor_state_normalized": self.rml_state_encoder.current_state_name,
            "rml_monitor_state_unencoded_normalized": normalize_monitor_state(self.monitor_state_unencoded),
            "monitor_soak_steps": int(self.monitor_soak_steps),
            "monitor_violation_steps": 0,
        }


def make_rml_cstr_env(
    config: RMLCSTRConfig | None = None,
    *,
    client: MonitorClient | None = None,
) -> RMLCSTREnv:
    """Create the RML-backed CSTR wrapper."""

    return RMLCSTREnv(config=config, client=client)


def _scale_graph_monitor_embedding(encoded: np.ndarray, *, structural_feature_dim: int) -> np.ndarray:
    """Balance frozen graph monitor features against the physical observation.

    The learned graph embedding can have an L2 norm around 6-9, which overwhelms
    the four normalized CSTR state features in SB3's concatenated MultiInput
    representation. Normalize only the learned part and preserve structural
    count/depth features in their explicit 0-1 scale.
    """
    vector = np.asarray(encoded, dtype=np.float32).reshape(-1)
    structural_dim = max(0, min(int(structural_feature_dim), int(vector.shape[0])))
    structural_scale = 0.5
    if structural_dim == 0:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else vector
    learned_dim = int(vector.shape[0]) - structural_dim
    learned = vector[:learned_dim]
    structural = vector[learned_dim:]
    norm = float(np.linalg.norm(learned))
    if norm > 1e-8:
        learned = learned / norm
    return np.concatenate([learned, structural_scale * structural], dtype=np.float32)


def _load_rml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"RML config {config_path} did not contain a mapping.")
    for key in ("variables", "reward", "host", "port"):
        if key not in config:
            raise ValueError(f"RML config {config_path} is missing {key!r}.")
    return config


def _classify_cstr_rml_state(
    monitor_state: str,
    *,
    verdict: str,
    payload: Mapping[str, Any],
    previous_phase: str,
    previous_soak_steps: int,
    soak_steps: int,
    recover_from_regulation_failure: bool = False,
) -> str:
    normalized = normalize_monitor_state(str(monitor_state))
    normalized_verdict = normalize_verdict(str(verdict))
    if normalized_verdict in {"true"} or normalized == "1":
        return "success"
    if normalized_verdict in {"false"} or normalized in {"0", "false_verdict"}:
        return "failure"
    stable = _truthy(payload.get("stable", False))
    in_soak = _truthy(payload.get("in_soak_band", False))
    safe = _truthy(payload.get("temp_safe", False))
    if _truthy(payload.get("stable", False)):
        return "regulate"
    critical = _truthy(payload.get("critical", False))
    overshoot = _truthy(payload.get("overshoot", False))
    if previous_phase == "regulate":
        if stable:
            return "regulate"
        if recover_from_regulation_failure and not critical and (safe or overshoot):
            return "approach"
        return "regulate"
    if previous_phase == "soak":
        if in_soak and previous_soak_steps >= soak_steps:
            return "approach"
        if in_soak:
            return "soak"
        if safe:
            return "preheat"
        return "failure"
    if previous_phase == "preheat":
        if in_soak:
            return "soak"
        return "preheat"
    if previous_phase == "approach":
        return "regulate" if stable else "approach"
    return "approach"


def _next_soak_steps(
    *,
    phase: str,
    payload: Mapping[str, Any],
    previous_phase: str,
    previous_soak_steps: int,
    soak_steps: int,
) -> int:
    if phase in {"success", "failure", "preheat", "approach", "regulate"}:
        return 0
    if phase != "soak":
        raise ValueError(f"Unsupported CSTR monitor phase: {phase!r}.")
    if previous_phase == "soak":
        return min(previous_soak_steps + 1, soak_steps)
    return 1


def _canonical_monitor_state(*, phase: str, soak_steps: int) -> str:
    if phase == "soak":
        return f"soak_{max(1, int(soak_steps))}"
    return phase


def normalize_monitor_state(monitor_state: str) -> str:
    """Normalize monitor states by removing generated unbound variable suffixes."""

    return re.sub(r"_[0-9]+", "", monitor_state)


def normalize_verdict(verdict: str) -> str:
    """Normalize monitor verdict labels to YAML reward keys."""

    if verdict in {"True", "False"}:
        return verdict.lower()
    return verdict


def _truthy(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.any())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value) != 0
    if isinstance(value, (np.floating, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
