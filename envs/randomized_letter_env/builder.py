"""Environment construction utilities for randomized LetterEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.letter_env.builder import FixedLetterNWrapper
from envs.letter_env.encodings import build_letter_env_semantic_progress_encoder
from envs.randomized_letter_env.encodings import build_randomized_letter_env_monitor_encoding
from envs.randomized_letter_env.env import RandomizedLetterEnv
from rml_rm.monitors.transaction import normalize_monitor_state
from rml_rm.wrappers import RMLMonitorWrapper


@dataclass(frozen=True)
class RandomizedLetterEnvConfig:
    """Configuration for the randomized LetterEnv experiment stack."""

    encoding: str = "one_hot"
    learned_gru_checkpoint: str | Path | None = None
    learned_graph_checkpoint: str | Path | None = None
    n_value: int = 1
    fixed_n: int | None = 1
    max_episode_steps: int = 200
    monitor_progress_bonus: float = 0.0
    placement_mode: str = "full_random"


class RandomizedLetterObservation(gym.ObservationWrapper):
    """Expose normalized agent and target locations plus proposition features."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        raw_env = self.unwrapped
        self.proposition_count = len(raw_env.proposition_to_index)
        self.no_proposition_index = int(raw_env.proposition_to_index["_"])
        feature_dim = 2 + self.proposition_count + 8
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.zeros(feature_dim, dtype=np.float32),
                    high=np.ones(feature_dim, dtype=np.float32),
                    shape=(feature_dim,),
                    dtype=np.float32,
                )
            }
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        raw_env = self.unwrapped
        row_scale = max(1, raw_env.n_rows - 1)
        col_scale = max(1, raw_env.n_cols - 1)

        position = np.asarray(observation["position"], dtype=np.float32)
        coordinates = np.asarray(
            [position[0] / row_scale, position[1] / col_scale],
            dtype=np.float32,
        )

        proposition_index = int(observation["proposition"])
        proposition_features = np.zeros(self.proposition_count, dtype=np.float32)
        value = float(observation.get("value", 0))
        if proposition_index == self.no_proposition_index:
            proposition_features[proposition_index] = 1.0
        else:
            proposition_features[proposition_index] = value if value > 0 else 1.0

        target_features: list[float] = []
        for symbol in ("A", "B", "C", "D"):
            location_symbol = "A" if symbol == "B" else symbol
            target_row, target_col = raw_env.locations[location_symbol]
            target_features.extend([target_row / row_scale, target_col / col_scale])

        encoded = np.concatenate(
            [
                coordinates,
                proposition_features,
                np.asarray(target_features, dtype=np.float32),
            ]
        ).astype(np.float32)
        return {"position": encoded}


class RMLTaskOutcomeWrapper(gym.Wrapper):
    """Expose task outcome fields using the RML monitor reward."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        monitor_verdict = str(info.get("monitor_verdict", ""))
        monitor_reward = float(info.get("monitor_reward", 0.0))
        monitor_state = normalize_monitor_state(str(info.get("monitor_state_unencoded", "")))
        if monitor_state == "1" or (
            monitor_verdict in {"true", "currently_true"} and monitor_reward > 0.0
        ):
            terminated = True
            info["success"] = True
            info["task_failed"] = False
            info["task_index"] = 1
            info["base_reward"] = monitor_reward
        elif monitor_state == "false_verdict":
            terminated = True
            info["success"] = False
            info["task_failed"] = True
            info["base_reward"] = monitor_reward
        else:
            info.setdefault("success", False)
            info.setdefault("task_failed", False)
        return observation, reward, terminated, truncated, info


class RandomizedLetterRewardShapingWrapper(gym.Wrapper):
    """Apply phase-based RML progress shaping for randomized LetterEnv."""

    def __init__(
        self,
        env: gym.Env,
        *,
        monitor_progress_bonus: float,
    ) -> None:
        super().__init__(env)
        self.monitor_progress_bonus = float(monitor_progress_bonus)
        self.progress_encoder = build_letter_env_semantic_progress_encoder()
        self.previous_monitor_progress = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_monitor_progress = 0.0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)
        monitor_progress = self._monitor_progress(
            info.get("monitor_state_unencoded"),
            terminated=bool(terminated),
            reward_before_wrapper=shaped_reward,
        )

        if monitor_progress > self.previous_monitor_progress:
            shaped_reward += self.monitor_progress_bonus

        self.previous_monitor_progress = monitor_progress
        return observation, shaped_reward, terminated, truncated, info

    def _monitor_progress(
        self,
        monitor_state: Any,
        *,
        terminated: bool,
        reward_before_wrapper: float,
    ) -> float:
        if terminated and reward_before_wrapper > 0.0:
            return 1000.0
        if terminated and reward_before_wrapper < 0.0:
            return -1000.0
        if monitor_state is None:
            return 0.0

        normalized_state = normalize_monitor_state(str(monitor_state))
        if normalized_state == "false_verdict":
            return -1000.0
        if normalized_state == "1":
            return 1000.0

        phase_vector = self.progress_encoder(str(monitor_state))
        return float(np.argmax(phase_vector))


def build_randomized_letter_env(
    config: RandomizedLetterEnvConfig,
    *,
    monitor_config_path: str | Path,
) -> gym.Env:
    """Build the wrapped randomized LetterEnv stack used by experiments."""
    if config.n_value < 1:
        raise ValueError("n_value must be at least 1.")
    if config.fixed_n is not None and not 1 <= config.fixed_n <= config.n_value:
        raise ValueError("fixed_n must be in 1..n_value.")

    raw_env: gym.Env = RandomizedLetterEnv(
        max_n=config.n_value,
        max_episode_steps=config.max_episode_steps,
        placement_mode=config.placement_mode,
    )
    if config.fixed_n is not None:
        raw_env = FixedLetterNWrapper(raw_env, fixed_n=config.fixed_n)

    monitor_encoder, initial_monitor_state, monitor_space = build_randomized_letter_env_monitor_encoding(
        config.encoding,
        learned_gru_checkpoint=config.learned_gru_checkpoint,
        learned_graph_checkpoint=config.learned_graph_checkpoint,
    )
    env: gym.Env = RMLMonitorWrapper(
        RandomizedLetterObservation(raw_env),
        config_path=monitor_config_path,
        monitor_encoder=monitor_encoder,
        initial_monitor_state=initial_monitor_state,
        monitor_space=monitor_space,
        include_transition_bonus=False,
    )
    env = RMLTaskOutcomeWrapper(env)
    env = RandomizedLetterRewardShapingWrapper(
        env,
        monitor_progress_bonus=config.monitor_progress_bonus,
    )
    return env
