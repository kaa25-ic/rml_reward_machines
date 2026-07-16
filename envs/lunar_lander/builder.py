"""Environment construction utilities for RML-based LunarLander."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

from envs.lunar_lander.encodings import build_lunar_lander_monitor_encoding
from envs.lunar_lander.env import (
    DEFAULT_LUNAR_LANDER_ENV_ID,
    LunarProtocolThresholds,
    make_lunar_lander_base_env,
)
from rml_rm.monitors.transaction import normalize_monitor_state
from rml_rm.wrappers import RMLMonitorWrapper


@dataclass(frozen=True)
class LunarLanderProtocolConfig:
    """Configuration for the LunarLander protocol experiment stack."""

    encoding: str = "semantic_progress"
    env_id: str = DEFAULT_LUNAR_LANDER_ENV_ID
    max_episode_steps: int = 1000
    monitor_progress_bonus: float = 20.0
    hover_step_bonus: float = 2.0
    hover_complete_bonus: float = 30.0
    controlled_descent_bonus: float = 20.0
    success_bonus: float = 200.0
    failure_penalty: float = -100.0
    landing_target_bonus: float = 0.0
    landing_angle_bonus: float = 0.0
    post_descent_landing_bonus: float = 0.0
    post_descent_protocol_miss_penalty: float = 0.0
    render_mode: str | None = None
    thresholds: LunarProtocolThresholds | None = None


class LunarProtocolOutcomeWrapper(gym.Wrapper):
    """Expose success and failure fields from the RML monitor state."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        monitor_state = normalize_monitor_state(str(info.get("monitor_state_unencoded", "")))
        monitor_reward = float(info.get("monitor_reward", 0.0))
        info["successful_protocol"] = False
        if monitor_state == "1":
            terminated = True
            info["success"] = True
            info["successful_protocol"] = True
            info["task_failed"] = False
            info["task_index"] = 1
            info["monitor_terminal_reward"] = monitor_reward
        elif monitor_state == "false_verdict":
            terminated = True
            info["success"] = False
            info["task_failed"] = True
            info["monitor_terminal_reward"] = monitor_reward
        else:
            info.setdefault("success", False)
            info.setdefault("task_failed", False)
            info["monitor_terminal_reward"] = monitor_reward
        return observation, reward, terminated, truncated, info


class LunarBaseRewardWrapper(gym.Wrapper):
    """Add the original LunarLander reward back to the RML monitor reward."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        lunar_base_reward = float(info.get("base_reward", 0.0))
        info["lunar_base_reward"] = lunar_base_reward
        info["successful_landing"] = bool(info.get("env_terminated", 0.0)) and lunar_base_reward > 0.0
        return observation, float(reward) + lunar_base_reward, terminated, truncated, info


class LunarProtocolRewardShapingWrapper(gym.Wrapper):
    """Apply protocol shaping components using RML monitor progress."""

    def __init__(
        self,
        env: gym.Env,
        *,
        monitor_progress_bonus: float,
        hover_step_bonus: float,
        hover_complete_bonus: float,
        controlled_descent_bonus: float,
        success_bonus: float,
        failure_penalty: float,
        landing_target_bonus: float,
        landing_angle_bonus: float,
        post_descent_landing_bonus: float,
        post_descent_protocol_miss_penalty: float,
    ) -> None:
        super().__init__(env)
        self.monitor_progress_bonus = float(monitor_progress_bonus)
        self.hover_step_bonus = float(hover_step_bonus)
        self.hover_complete_bonus = float(hover_complete_bonus)
        self.controlled_descent_bonus = float(controlled_descent_bonus)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.landing_target_bonus = float(landing_target_bonus)
        self.landing_angle_bonus = float(landing_angle_bonus)
        self.post_descent_landing_bonus = float(post_descent_landing_bonus)
        self.post_descent_protocol_miss_penalty = float(post_descent_protocol_miss_penalty)
        self.previous_progress = 0.0
        self.previous_hover_count = 0
        self.terminal_shaping_applied = False
        self.landing_target_bonus_applied = False
        self.landing_angle_bonus_applied = False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_progress = 0.0
        self.previous_hover_count = 0
        self.terminal_shaping_applied = False
        self.landing_target_bonus_applied = False
        self.landing_angle_bonus_applied = False
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        shaped_reward = float(reward)
        monitor_state = info.get("monitor_state_unencoded")
        progress = _lunar_monitor_progress(monitor_state)
        hover_count = _lunar_hover_count(monitor_state)
        shaping_components = {
            "progress": 0.0,
            "hover_step": 0.0,
            "hover_complete": 0.0,
            "controlled_descent": 0.0,
            "success": 0.0,
            "failure": 0.0,
            "landing_target": 0.0,
            "landing_angle": 0.0,
            "post_descent_landing": 0.0,
            "post_descent_protocol_miss": 0.0,
        }
        if progress > self.previous_progress:
            shaping_components["progress"] = self.monitor_progress_bonus
        if hover_count > self.previous_hover_count:
            shaping_components["hover_step"] = (
                hover_count - self.previous_hover_count
            ) * self.hover_step_bonus
        if progress >= 3.0 > self.previous_progress:
            shaping_components["hover_complete"] = self.hover_complete_bonus
        if progress >= 4.0 > self.previous_progress:
            shaping_components["controlled_descent"] = self.controlled_descent_bonus
        normalized_state = normalize_monitor_state(str(monitor_state)).replace("\\_", "_")
        in_landing_phase = progress >= 4.0 or self.previous_progress >= 4.0
        if in_landing_phase:
            if (
                not self.landing_target_bonus_applied
                and float(info.get("target_zone", 0.0)) == 1.0
            ):
                shaping_components["landing_target"] = self.landing_target_bonus
                self.landing_target_bonus_applied = True
            if (
                not self.landing_angle_bonus_applied
                and float(info.get("safe_landing_angle", 0.0)) == 1.0
            ):
                shaping_components["landing_angle"] = self.landing_angle_bonus
                self.landing_angle_bonus_applied = True
        if not self.terminal_shaping_applied:
            if normalized_state == "1":
                shaping_components["success"] = self.success_bonus
                self.terminal_shaping_applied = True
            elif normalized_state == "false_verdict":
                shaping_components["failure"] = self.failure_penalty
                if in_landing_phase and bool(info.get("successful_landing", False)):
                    shaping_components["post_descent_landing"] = self.post_descent_landing_bonus
                    shaping_components[
                        "post_descent_protocol_miss"
                    ] = self.post_descent_protocol_miss_penalty
                self.terminal_shaping_applied = True

        shaping_reward = float(sum(shaping_components.values()))
        shaped_reward += shaping_reward
        info["protocol_shaping_reward"] = shaping_reward
        for name, value in shaping_components.items():
            info[f"protocol_reward_{name}"] = value
        self.previous_progress = progress
        self.previous_hover_count = hover_count
        return observation, shaped_reward, terminated, truncated, info


def build_lunar_lander_protocol_env(
    config: LunarLanderProtocolConfig,
    *,
    monitor_config_path: str | Path,
) -> gym.Env:
    """Build the wrapped LunarLander stack used by experiments."""
    if config.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive.")

    raw_env = make_lunar_lander_base_env(
        env_id=config.env_id,
        render_mode=config.render_mode,
        thresholds=config.thresholds,
    )
    monitor_encoder, initial_monitor_state, monitor_space = build_lunar_lander_monitor_encoding(
        config.encoding
    )
    env: gym.Env = RMLMonitorWrapper(
        raw_env,
        config_path=monitor_config_path,
        monitor_encoder=monitor_encoder,
        initial_monitor_state=initial_monitor_state,
        monitor_space=monitor_space,
        include_transition_bonus=False,
    )
    env = LunarBaseRewardWrapper(env)
    env = LunarProtocolOutcomeWrapper(env)
    env = LunarProtocolRewardShapingWrapper(
        env,
        monitor_progress_bonus=config.monitor_progress_bonus,
        hover_step_bonus=config.hover_step_bonus,
        hover_complete_bonus=config.hover_complete_bonus,
        controlled_descent_bonus=config.controlled_descent_bonus,
        success_bonus=config.success_bonus,
        failure_penalty=config.failure_penalty,
        landing_target_bonus=config.landing_target_bonus,
        landing_angle_bonus=config.landing_angle_bonus,
        post_descent_landing_bonus=config.post_descent_landing_bonus,
        post_descent_protocol_miss_penalty=config.post_descent_protocol_miss_penalty,
    )
    return env


def _lunar_monitor_progress(raw_monitor_state: Any) -> float:
    if raw_monitor_state is None:
        return 0.0
    state = normalize_monitor_state(str(raw_monitor_state)).replace("\\_", "_")
    if state == "false_verdict":
        return -1000.0
    if state == "1":
        return 5.0
    if state.startswith("star(waiting_for_corridor"):
        return 0.0
    if state.startswith("star(waiting_for_landing"):
        return 4.0
    if state.startswith("app(gen([],star(waiting_for_descent") or state.startswith(
        "star(waiting_for_descent"
    ):
        return 3.0
    if "waiting_for_hover" in state:
        return 2.0 if _lunar_hover_count(state) > 0 else 1.0
    return 0.0


def _lunar_hover_count(raw_monitor_state: Any) -> int:
    if raw_monitor_state is None:
        return 0
    state = normalize_monitor_state(str(raw_monitor_state)).replace("\\_", "_")
    if state in {"1", "false_verdict"}:
        return 0
    if (
        state.startswith("app(gen([],star(waiting_for_descent")
        or state.startswith("star(waiting_for_descent")
        or state.startswith("star(waiting_for_landing")
    ):
        return 3
    if "(1+1)" in state or "[1+1]" in state:
        return 2
    if "(0+1)" in state or "[0+1]" in state:
        return 1
    return 0
