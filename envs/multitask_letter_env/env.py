"""Pure RML-based multi-task LetterEnv."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.letter_env_core import NO_PROPOSITION, LetterGridWorld
from envs.multitask_letter_env.encodings import (
    build_multitask_monitor_encoding,
    load_monitor_progress_catalogue,
)
from envs.multitask_letter_env.rml_generation import CONFIGS_ROOT
from envs.multitask_letter_env.tasks import get_task_suite
from rml_rm.monitors.transaction import (
    load_monitor_config,
    monitor_payload_from_observation,
    normalize_monitor_state,
    reset_monitor,
    rewards_from_config,
    step_monitor,
)
from rml_rm.wrappers.observation import PropositionEncodingSpec, encode_proposition_vector
from rml_rm.wrappers.rml_monitor import (
    MonitorClient,
    WebSocketMonitorClient,
)


SUPPORTED_ENCODINGS = (
    "one_hot",
    "numerical",
    "learned_gru",
    "learned_graph",
)


@dataclass(frozen=True)
class MultiTaskLetterEnvConfig:
    """Configuration for the RML-based multi-task LetterEnv."""

    encoding: str = "one_hot"
    task_suite: str = "small_v1"
    max_n: int = 5
    max_episode_steps: int = 200
    monitor_host: str = "127.0.0.1"
    monitor_base_port: int = 18_201
    catalogue_path: str | Path | None = None
    progress_catalogue_path: str | Path | None = None
    learned_gru_checkpoint: str | Path | None = None
    learned_graph_checkpoint: str | Path | None = None
    config_root: str | Path = CONFIGS_ROOT
    monitor_ports_by_task_id: dict[int, int] | None = None
    transition_bonus: float = 0.0
    include_transition_bonus: bool = False


class MultiTaskLetterEnv(LetterGridWorld):
    """LetterEnv family where task progress is evaluated by RML monitors."""

    terminal_monitor_states = {"1", "false_verdict"}

    def __init__(
        self,
        config: MultiTaskLetterEnvConfig | None = None,
        *,
        clients: Mapping[int, MonitorClient] | None = None,
    ) -> None:
        self.config = config or MultiTaskLetterEnvConfig()
        if self.config.encoding not in SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported encoding: {self.config.encoding!r}")

        super().__init__(
            max_n=self.config.max_n,
            max_episode_steps=self.config.max_episode_steps,
        )
        self.tasks = get_task_suite(self.config.task_suite)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        self.config_root = Path(self.config.config_root)
        self.monitor_configs_by_task = {
            task.task_id: load_monitor_config(self.config_root / f"{task.key}.yaml")
            for task in self.tasks
        }
        self.monitor_variables_by_task = {
            task_id: list(config["variables"])
            for task_id, config in self.monitor_configs_by_task.items()
        }
        self.rewards_by_task = {
            task_id: rewards_from_config(config)
            for task_id, config in self.monitor_configs_by_task.items()
        }
        self.clients = dict(clients or self._build_clients())
        self.monitor_encoder, self.initial_monitor_state, self.monitor_state_size = (
            build_multitask_monitor_encoding(
                self.config.encoding,
                catalogue_path=self.config.catalogue_path,
                learned_gru_checkpoint=self.config.learned_gru_checkpoint,
                learned_graph_checkpoint=self.config.learned_graph_checkpoint,
            )
        )
        self.monitor_progress_catalogue = load_monitor_progress_catalogue(
            self.config.progress_catalogue_path
        )
        self.previous_monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state_unencoded = ""
        self.previous_monitor_progress = 0
        self.last_verdict = "currently_false"
        self.selected_task = self.tasks[0]

        self.task_feature_size = len(self.tasks)
        self.proposition_encoding_spec = _build_proposition_encoding_spec(self)
        base_size = 2 + len(self.proposition_to_index) + 1 + self.task_feature_size
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(base_size,),
                    dtype=np.float32,
                ),
                "monitor": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.monitor_state_size,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        task_id = options.get("task_id")
        n_value = options.get("n")
        self.selected_task = (
            self.task_by_id[int(task_id)]
            if task_id is not None
            else self.tasks[int(self.np_random.integers(0, len(self.tasks)))]
        )
        sampled_n = (
            int(n_value)
            if n_value is not None
            else int(self.np_random.integers(1, self.max_n + 1))
        )
        self.reset_grid(sampled_n=sampled_n)
        self.previous_monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state = copy.deepcopy(self.initial_monitor_state)
        self.monitor_state_unencoded = ""
        self.previous_monitor_progress = 0
        self.last_verdict = "currently_false"
        self._reset_selected_monitor()
        return self._observation(), self._info(NO_PROPOSITION, 0.0, 0.0)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        label = self.move_agent(action)
        if label != NO_PROPOSITION:
            self.apply_replacement_if_needed(label)

        monitor_reward, monitor_info = self._monitor_step(label)
        normalized_state = normalize_monitor_state(self.monitor_state_unencoded)
        terminated = normalized_state in self.terminal_monitor_states or self.last_verdict in {
            "true",
            "false",
        }
        truncated = not terminated and self.n_steps >= self.max_episode_steps
        return self._observation(label), monitor_reward, terminated, truncated, monitor_info

    def _monitor_step(self, observed_label: str) -> tuple[float, dict[str, Any]]:
        payload = self._payload(observed_label)
        payload["terminate"] = False
        result = step_monitor(
            self.clients[self.selected_task.task_id],
            payload,
            self.rewards_by_task[self.selected_task.task_id],
        )
        verdict = result.verdict
        raw_state = result.monitor_state
        base_reward = result.base_reward
        encoded_state = self.monitor_encoder(raw_state)
        transition_bonus = 0.0
        monitor_progress = self._monitor_progress(raw_state)
        if self.config.include_transition_bonus and monitor_progress > self.previous_monitor_progress:
            transition_bonus = float(self.config.transition_bonus)

        self.last_verdict = verdict
        self.monitor_state_unencoded = raw_state
        self.monitor_state = np.asarray(encoded_state, dtype=np.float32)
        self.previous_monitor_state = copy.deepcopy(self.monitor_state)
        self.previous_monitor_progress = monitor_progress
        return (
            base_reward + transition_bonus,
            self._info(observed_label, base_reward, transition_bonus),
        )

    def _reset_selected_monitor(self) -> None:
        reset_monitor(
            self.clients[self.selected_task.task_id],
            self.monitor_variables_by_task[self.selected_task.task_id],
        )

    def _payload(self, observed_label: str | None = None) -> dict[str, Any]:
        encoded_position = encode_proposition_vector(
            self.make_observation(observed_label),
            self.proposition_encoding_spec,
        )
        return monitor_payload_from_observation(
            variables=self.monitor_variables_by_task[self.selected_task.task_id],
            observation={"position": encoded_position},
            state_owner=self,
        )

    def _observation(self, observed_label: str | None = None) -> dict[str, np.ndarray]:
        raw_observation = self.make_observation(observed_label)
        position_features = encode_proposition_vector(
            raw_observation,
            self.proposition_encoding_spec,
        )
        n_feature = np.asarray([self.sampled_n / self.max_n], dtype=np.float32)
        task_features = np.zeros(self.task_feature_size, dtype=np.float32)
        task_features[self.selected_task.task_id] = 1.0
        return {
            "position": np.concatenate([position_features, n_feature, task_features]).astype(
                np.float32
            ),
            "monitor": np.asarray(self.monitor_state, dtype=np.float32),
        }

    def _info(
        self,
        observed_label: str,
        monitor_reward: float,
        transition_bonus: float,
    ) -> dict[str, Any]:
        normalized_state = normalize_monitor_state(self.monitor_state_unencoded)
        success = normalized_state == "1" or self.last_verdict == "true"
        failed = normalized_state == "false_verdict" or self.last_verdict == "false"
        return {
            "proposition_label": observed_label,
            "task_id": self.selected_task.task_id,
            "task_key": self.selected_task.key,
            "task_expression": self.selected_task.expression,
            "n": self.sampled_n,
            "success": success,
            "failed": failed,
            "monitor_verdict": self.last_verdict,
            "monitor_state_unencoded": self.monitor_state_unencoded,
            "monitor_state_normalized": normalized_state,
            "monitor_reward": float(monitor_reward),
            "base_reward": float(monitor_reward),
            "monitor_transition_bonus": float(transition_bonus),
            "monitor_progress": self.previous_monitor_progress,
            "task_failed": failed,
        }

    def _build_clients(self) -> dict[int, MonitorClient]:
        return {
            task.task_id: WebSocketMonitorClient(
                host=self.config.monitor_host,
                port=(
                    self.config.monitor_ports_by_task_id[task.task_id]
                    if self.config.monitor_ports_by_task_id is not None
                    else self.config.monitor_base_port + task.task_id
                ),
            )
            for task in self.tasks
        }

    def _monitor_progress(self, monitor_state: str) -> int:
        normalized_state = normalize_monitor_state(monitor_state)
        if normalized_state == "false_verdict":
            return -1
        if normalized_state == "1":
            return len(self.selected_task.successful_events(n=self.sampled_n))
        task_progress = self.monitor_progress_catalogue.get(self.selected_task.key, {})
        n_progress = task_progress.get(self.sampled_n, {})
        return int(n_progress.get(normalized_state, self.previous_monitor_progress))


def _build_proposition_encoding_spec(env: LetterGridWorld) -> PropositionEncodingSpec:
    raw_position_space = env.observation_space["position"]
    raw_value_space = env.observation_space["value"]
    return PropositionEncodingSpec(
        proposition_count=len(env.proposition_to_index),
        no_proposition_index=int(env.proposition_to_index[NO_PROPOSITION]),
        position_low=np.asarray(raw_position_space.low, dtype=np.float32),
        position_high=np.asarray(raw_position_space.high, dtype=np.float32),
        max_value=int(raw_value_space.n) - 1,
    )
