"""Native CSTR Gymnasium environment with physical event predicates.

The environment models an exothermic continuous stirred-tank reactor (CSTR)
with a single coolant-temperature control action. It intentionally does not
implement temporal task memory. Instead it exposes instantaneous physical
events in ``info`` so an RML monitor can own the task logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class CSTRConfig:
    """Configuration for the native CSTR control task."""

    max_episode_steps: int = 300
    integration_steps: int = 50
    dt: float = 0.001
    ca_initial: float = 0.80
    temp_initial: float = 331.0
    ca_setpoint: float = 0.50
    randomize_initial_state: bool = False
    ca_initial_noise: float = 0.03
    temp_initial_noise: float = 3.0
    randomize_setpoint: bool = False
    ca_setpoint_low: float = 0.45
    ca_setpoint_high: float = 0.55
    enable_disturbance: bool = False
    disturbance_probability: float = 0.02
    disturbance_duration_steps: int = 25
    disturbance_feed_temp_delta: float = 8.0
    disturbance_feed_conc_delta: float = 0.08
    action_low: float = 250.0
    action_high: float = 350.0
    default_coolant_temp: float = 300.0
    initial_coolant_temp: float = 302.5
    slew_limit: float = 8.0
    ramp_limit: float = 1.0
    soak_steps: int = 10
    safe_temp_low: float = 315.0
    safe_temp_high: float = 375.0
    soak_band_low: float = 343.0
    soak_band_high: float = 347.0
    require_soak_concentration_band: bool = False
    soak_concentration_low: float = 0.58
    soak_concentration_high: float = 0.74
    production_temp_low: float = 348.0
    production_temp_high: float = 352.0
    ca_overshoot_low: float = 0.44
    deadline_steps: int = 60
    warning_temp_high: float = 382.0
    recovery_temp_threshold: float = 365.0
    critical_temp_high: float = 405.0
    critical_temp_low: float = 280.0
    concentration_tolerance: float = 0.04
    temperature_tolerance: float = 8.0
    target_temp: float = 350.0
    tracking_weight: float = 0.5
    temp_weight: float = 0.015
    action_weight: float = 0.0002
    warning_penalty: float = 0.25
    heating_rate_penalty: float = 0.02
    critical_penalty: float = 200.0
    stable_bonus: float = 0.15
    normalize_observation: bool = True
    concentration_error_scale: float = 0.15
    temperature_error_scale: float = 25.0
    setpoint_center: float = 0.5
    setpoint_scale: float = 0.25
    # CSTR model parameters, close to the classic exothermic reactor example.
    q: float = 100.0
    volume: float = 100.0
    rho: float = 1000.0
    cp: float = 0.239
    delta_h: float = -50_000.0
    e_over_r: float = 8750.0
    k0: float = 7.2e10
    ua: float = 50_000.0
    ca_feed: float = 1.0
    feed_temp: float = 350.0


class CSTREnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous CSTR setpoint-control environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: CSTRConfig | None = None) -> None:
        super().__init__()
        self.config = config or CSTRConfig()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self._build_observation_space()
        self.steps = 0
        self.ca = float(self.config.ca_initial)
        self.temp = float(self.config.temp_initial)
        self.ca_setpoint = float(self.config.ca_setpoint)
        self.previous_action = 0.0
        self.previous_coolant = float(self.config.initial_coolant_temp)
        self.heating_rate = 0.0
        self.previous_warning_active = False
        self.disturbance_steps_remaining = 0
        self.warning_events = 0
        self.critical_events = 0
        self.stable_steps = 0
        self.max_stable_steps = 0
        self.first_stable_step: int | None = None
        self.cumulative_tracking_error = 0.0
        self.cumulative_temp_violation = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.steps = 0
        self.ca_setpoint = float(options.get("ca_setpoint", self._sample_setpoint()))
        self.ca = float(options.get("ca_initial", self._sample_initial_ca()))
        self.temp = float(options.get("temp_initial", self._sample_initial_temp()))
        self.previous_action = 0.0
        self.previous_coolant = float(options.get("initial_coolant_temp", self.config.initial_coolant_temp))
        self.heating_rate = 0.0
        self.previous_warning_active = False
        self.disturbance_steps_remaining = 0
        self.warning_events = 0
        self.critical_events = 0
        self.stable_steps = 0
        self.max_stable_steps = 0
        self.first_stable_step = None
        self.cumulative_tracking_error = 0.0
        self.cumulative_temp_violation = 0.0
        info = self._info(base_reward=0.0, action_coolant_temp=self.previous_coolant)
        return self._observation(), info

    def step(self, action: np.ndarray | list[float] | float):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        action_value = float(np.clip(action_value, -1.0, 1.0))
        cmd_coolant_temp = self._scale_action(action_value)
        slew = float(max(self.config.slew_limit, 0.0))
        coolant_temp = self.previous_coolant + float(np.clip(cmd_coolant_temp - self.previous_coolant, -slew, slew))
        coolant_temp = float(np.clip(coolant_temp, self.config.action_low, self.config.action_high))
        self._maybe_update_disturbance()
        prev_temp = float(self.temp)
        for _ in range(self.config.integration_steps):
            self._integrate(coolant_temp)
        self.ca = float(np.clip(self.ca, 0.0, 1.5))
        self.temp = float(np.clip(self.temp, 200.0, 500.0))
        self.heating_rate = float(self.temp - prev_temp)
        self.steps += 1

        tracking_error = abs(self.ca - self.ca_setpoint)
        temp_error = abs(self.temp - self.config.target_temp)
        warning_active = self._temp_warning()
        warning_event = warning_active and not self.previous_warning_active
        critical_event = self._temp_critical()
        heating_rate_exceeded = self._heating_rate_exceeded()
        stable_step = self._stable_step()
        self.warning_events += int(warning_event)
        self.critical_events += int(critical_event)
        self.stable_steps = self.stable_steps + 1 if stable_step else 0
        self.max_stable_steps = max(self.max_stable_steps, self.stable_steps)
        if stable_step and self.first_stable_step is None:
            self.first_stable_step = self.steps
        self.cumulative_tracking_error += tracking_error
        self.cumulative_temp_violation += self._temperature_violation()

        reward = -self.config.tracking_weight * tracking_error
        reward -= self.config.temp_weight * temp_error
        reward -= self.config.action_weight * (coolant_temp - self.config.default_coolant_temp) ** 2
        reward -= self.config.warning_penalty * float(warning_active)
        reward -= self.config.heating_rate_penalty * float(heating_rate_exceeded)
        reward -= self.config.critical_penalty * float(critical_event)
        reward += self.config.stable_bonus * float(stable_step)

        terminated = bool(critical_event)
        truncated = bool(self.steps >= self.config.max_episode_steps)
        self.previous_action = action_value
        self.previous_coolant = coolant_temp
        self.previous_warning_active = warning_active
        info = self._info(base_reward=reward, action_coolant_temp=coolant_temp)
        return self._observation(), float(reward), terminated, truncated, info

    def _integrate(self, coolant_temp: float) -> None:
        cfg = self.config
        ca_feed, feed_temp = self._feed_conditions()
        reaction_rate = cfg.k0 * np.exp(-cfg.e_over_r / max(self.temp, 1.0)) * self.ca
        dca_dt = (cfg.q / cfg.volume) * (ca_feed - self.ca) - reaction_rate
        heat_generation = (-cfg.delta_h / (cfg.rho * cfg.cp)) * reaction_rate
        heat_removal = (cfg.ua / (cfg.rho * cfg.cp * cfg.volume)) * (coolant_temp - self.temp)
        dtemp_dt = (cfg.q / cfg.volume) * (feed_temp - self.temp) + heat_generation + heat_removal
        self.ca += cfg.dt * dca_dt
        self.temp += cfg.dt * dtemp_dt

    def _maybe_update_disturbance(self) -> None:
        if not self.config.enable_disturbance:
            self.disturbance_steps_remaining = 0
            return
        if self.disturbance_steps_remaining > 0:
            self.disturbance_steps_remaining -= 1
            return
        if float(self.np_random.random()) < self.config.disturbance_probability:
            self.disturbance_steps_remaining = int(self.config.disturbance_duration_steps)

    def _feed_conditions(self) -> tuple[float, float]:
        if self.disturbance_steps_remaining <= 0:
            return self.config.ca_feed, self.config.feed_temp
        return (
            self.config.ca_feed + self.config.disturbance_feed_conc_delta,
            self.config.feed_temp + self.config.disturbance_feed_temp_delta,
        )

    def _sample_initial_ca(self) -> float:
        if not self.config.randomize_initial_state:
            return self.config.ca_initial
        return float(self.config.ca_initial + self.np_random.uniform(-self.config.ca_initial_noise, self.config.ca_initial_noise))

    def _sample_initial_temp(self) -> float:
        if not self.config.randomize_initial_state:
            return self.config.temp_initial
        return float(self.config.temp_initial + self.np_random.uniform(-self.config.temp_initial_noise, self.config.temp_initial_noise))

    def _sample_setpoint(self) -> float:
        if not self.config.randomize_setpoint:
            return self.config.ca_setpoint
        return float(self.np_random.uniform(self.config.ca_setpoint_low, self.config.ca_setpoint_high))

    def _scale_action(self, action_value: float) -> float:
        fraction = 0.5 * (action_value + 1.0)
        return float(self.config.action_low + fraction * (self.config.action_high - self.config.action_low))

    def _observation(self) -> np.ndarray:
        if not self.config.normalize_observation:
            return np.asarray([self.ca, self.temp, self.ca_setpoint, self.previous_action], dtype=np.float32)
        return np.asarray(
            [
                (self.ca - self.ca_setpoint) / self.config.concentration_error_scale,
                (self.temp - self.config.target_temp) / self.config.temperature_error_scale,
                (self.ca_setpoint - self.config.setpoint_center) / self.config.setpoint_scale,
                (self.previous_coolant - self.config.default_coolant_temp) / 50.0,
            ],
            dtype=np.float32,
        )

    def _build_observation_space(self) -> spaces.Box:
        if not self.config.normalize_observation:
            return spaces.Box(
                low=np.asarray([0.0, 250.0, 0.0, -1.0], dtype=np.float32),
                high=np.asarray([1.5, 500.0, 1.5, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=np.asarray([-10.0, -12.0, -4.0, -1.0], dtype=np.float32),
            high=np.asarray([10.0, 12.0, 4.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _temp_safe(self) -> bool:
        return self.config.safe_temp_low <= self.temp <= self.config.safe_temp_high

    def _temp_warning(self) -> bool:
        return self.temp >= self.config.warning_temp_high

    def _temp_recovered(self) -> bool:
        return self.temp <= self.config.recovery_temp_threshold

    def _temp_critical(self) -> bool:
        return self.temp >= self.config.critical_temp_high or self.temp <= self.config.critical_temp_low

    def _concentration_near_setpoint(self) -> bool:
        return abs(self.ca - self.ca_setpoint) <= self.config.concentration_tolerance

    def _stable_step(self) -> bool:
        return self._production_stable()

    def _in_soak_band(self) -> bool:
        temp_in_band = self.config.soak_band_low <= self.temp <= self.config.soak_band_high
        if not self.config.require_soak_concentration_band:
            return temp_in_band
        ca_in_band = self.config.soak_concentration_low <= self.ca <= self.config.soak_concentration_high
        return temp_in_band and ca_in_band

    def _in_soak_temp_band(self) -> bool:
        return self.config.soak_band_low <= self.temp <= self.config.soak_band_high

    def _in_soak_concentration_band(self) -> bool:
        return self.config.soak_concentration_low <= self.ca <= self.config.soak_concentration_high

    def _production_temp_near(self) -> bool:
        return self.config.production_temp_low <= self.temp <= self.config.production_temp_high

    def _production_stable(self) -> bool:
        return self._concentration_near_setpoint() and self._production_temp_near() and self._temp_safe()

    def _overshoot(self) -> bool:
        return self.ca < self.config.ca_overshoot_low

    def _heating_rate_exceeded(self) -> bool:
        return self.heating_rate > self.config.ramp_limit

    def _past_deadline(self) -> bool:
        return self.steps == self.config.deadline_steps

    def _deadline_expired(self) -> bool:
        return self.steps > self.config.deadline_steps

    def _temperature_violation(self) -> float:
        if self.temp < self.config.safe_temp_low:
            return float(self.config.safe_temp_low - self.temp)
        if self.temp > self.config.safe_temp_high:
            return float(self.temp - self.config.safe_temp_high)
        return 0.0

    def _info(self, *, base_reward: float, action_coolant_temp: float) -> dict[str, Any]:
        tracking_error = abs(self.ca - self.ca_setpoint)
        temp_error = abs(self.temp - self.config.target_temp)
        temp_safe = self._temp_safe()
        temp_warning = self._temp_warning()
        temp_recovered = self._temp_recovered()
        temp_critical = self._temp_critical()
        concentration_near = self._concentration_near_setpoint()
        in_soak_band = self._in_soak_band()
        in_soak_temp_band = self._in_soak_temp_band()
        in_soak_concentration_band = self._in_soak_concentration_band()
        production_temp_near = self._production_temp_near()
        production_stable = self._production_stable()
        overshoot = self._overshoot()
        heating_rate_exceeded = self._heating_rate_exceeded()
        past_deadline = self._past_deadline()
        deadline_expired = self._deadline_expired()
        stable_step = production_stable
        warning_event = temp_warning and not self.previous_warning_active
        ca_feed, feed_temp = self._feed_conditions()
        return {
            "base_reward": float(base_reward),
            "steps": int(self.steps),
            "ca": float(self.ca),
            "reactor_concentration": float(self.ca),
            "temperature": float(self.temp),
            "reactor_temperature": float(self.temp),
            "ca_setpoint": float(self.ca_setpoint),
            "target_concentration": float(self.ca_setpoint),
            "tracking_error": float(tracking_error),
            "temperature_error": float(temp_error),
            "temperature_violation": float(self._temperature_violation()),
            "cumulative_tracking_error": float(self.cumulative_tracking_error),
            "cumulative_temperature_violation": float(self.cumulative_temp_violation),
            "action_normalized": float(self.previous_action),
            "action_coolant_temp": float(action_coolant_temp),
            "previous_coolant_temp": float(self.previous_coolant),
            "heating_rate": float(self.heating_rate),
            "ca_feed": float(ca_feed),
            "feed_temp": float(feed_temp),
            "disturbance_active": bool(self.disturbance_steps_remaining > 0),
            "disturbance_steps_remaining": int(self.disturbance_steps_remaining),
            "temp_safe": bool(temp_safe),
            "temp_warning": bool(temp_warning),
            "temp_recovered": bool(temp_recovered),
            "temp_critical": bool(temp_critical),
            "concentration_near_setpoint": bool(concentration_near),
            "in_soak_band": bool(in_soak_band),
            "in_soak_temp_band": bool(in_soak_temp_band),
            "in_soak_concentration_band": bool(in_soak_concentration_band),
            "production_temp_near": bool(production_temp_near),
            "production_stable": bool(production_stable),
            "overshoot": bool(overshoot),
            "heating_rate_exceeded": bool(heating_rate_exceeded),
            "past_deadline": bool(past_deadline),
            "deadline_expired": bool(deadline_expired),
            "stable_step": bool(stable_step),
            "warning_events": int(self.warning_events),
            "critical_events": int(self.critical_events),
            "stable_steps": int(self.stable_steps),
            "max_stable_steps": int(self.max_stable_steps),
            "first_stable_step": -1 if self.first_stable_step is None else int(self.first_stable_step),
            "event_temp_safe": bool(temp_safe),
            "event_temp_warning": bool(temp_warning),
            "event_temp_warning_entry": bool(warning_event),
            "event_temp_recovered": bool(temp_recovered),
            "event_temp_critical": bool(temp_critical),
            "event_concentration_near": bool(concentration_near),
            "event_in_soak_band": bool(in_soak_band),
            "event_in_soak_temp_band": bool(in_soak_temp_band),
            "event_in_soak_concentration_band": bool(in_soak_concentration_band),
            "event_production_temp_near": bool(production_temp_near),
            "event_production_stable": bool(production_stable),
            "event_overshoot": bool(overshoot),
            "event_heating_rate_exceeded": bool(heating_rate_exceeded),
            "event_past_deadline": bool(past_deadline),
            "event_deadline_expired": bool(deadline_expired),
            "event_stable_step": bool(stable_step),
            "event_clean_step": bool(not temp_warning and not temp_critical),
            "config": asdict(self.config),
        }


def make_cstr_env(config: CSTRConfig | None = None) -> CSTREnv:
    """Create a native CSTR Gymnasium environment."""

    return CSTREnv(config=config)
