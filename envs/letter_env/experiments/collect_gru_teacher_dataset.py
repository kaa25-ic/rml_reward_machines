"""Collect LetterEnv teacher rollouts for GRU monitor-encoder distillation."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.letter_env import LetterEnvConfig, build_letter_env
from envs.letter_env.encodings import load_letter_env_monitor_state_catalogue
from rml_rm.agents.ddqn import DoubleDQN
from rml_rm.experiments.runtime import json_ready, managed_monitor, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_MONITOR_CONFIG = LETTER_ENV_ROOT / "configs" / "letter_env.yaml"
DEFAULT_MONITOR_SPEC = LETTER_ENV_ROOT / "specs" / "letter_env_spec_numerical_runtime_compatible.pl"
DEFAULT_OUTPUT_DIR = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gru_teacher_dataset_n1to5_seed0"
)


@dataclass(frozen=True)
class TeacherDatasetConfig:
    """Configuration for collecting a LetterEnv GRU teacher dataset."""

    teacher_model_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    n_value: int = 5
    fixed_n: int | None = None
    greedy_episodes: int = 200
    epsilon_episodes: int = 200
    epsilon: float = 0.05
    seed: int = 0
    max_episode_steps: int = 200
    monitor_progress_bonus: float = 10.0
    monitor_regression_penalty: float = 0.0


def collect_teacher_dataset(
    config: TeacherDatasetConfig,
    *,
    monitor_config_template: Path = DEFAULT_MONITOR_CONFIG,
    monitor_spec_path: Path = DEFAULT_MONITOR_SPEC,
) -> dict[str, Any]:
    """Collect a JSONL teacher dataset from a trained numerical DDQN policy."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "experiment": "letter_env_gru_teacher_dataset",
            "config": asdict(config),
        },
    )
    dataset_path = output_dir / "dataset.jsonl"
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = DoubleDQN.load(str(config.teacher_model_path), device="auto")
    initial_monitor_string = load_letter_env_monitor_state_catalogue()[0]
    started = time.monotonic()
    rows_written = 0
    split_summaries: list[dict[str, Any]] = []

    with managed_monitor(
        output_dir=output_dir,
        monitor_config_template=monitor_config_template,
        monitor_spec_path=monitor_spec_path,
        log_name="dataset_rml_monitor.log",
        config_name="monitor_dataset_config.yaml",
        max_episode_steps=config.max_episode_steps,
    ) as runtime:
        env = build_letter_env(
            LetterEnvConfig(
                encoding="numerical",
                n_value=config.n_value,
                fixed_n=config.fixed_n,
                max_episode_steps=config.max_episode_steps,
                monitor_progress_bonus=config.monitor_progress_bonus,
                monitor_regression_penalty=config.monitor_regression_penalty,
                neutralize_legacy_transition_bonus=True,
                state_discovery_bonus=0.0,
            ),
            monitor_config_path=runtime.config_path,
        )
        try:
            with dataset_path.open("w", encoding="utf-8") as handle:
                for split, episodes, epsilon, seed_offset in (
                    ("greedy", config.greedy_episodes, 0.0, 0),
                    ("epsilon", config.epsilon_episodes, config.epsilon, 50_000),
                ):
                    summary = _collect_episodes(
                        env=env,
                        model=model,
                        handle=handle,
                        split=split,
                        num_episodes=episodes,
                        epsilon=epsilon,
                        seed=config.seed + seed_offset,
                        rng=rng,
                        initial_monitor_string=initial_monitor_string,
                    )
                    rows_written += int(summary["rows"])
                    split_summaries.append(summary)
        finally:
            env.close()

    summary = {
        "created_at_utc": utc_now(),
        "runtime_seconds": time.monotonic() - started,
        "dataset_path": str(dataset_path),
        "rows": rows_written,
        "config": json_ready(asdict(config)),
        "splits": split_summaries,
        "artifacts": {
            "config": str(output_dir / "config.json"),
            "dataset": str(dataset_path),
            "summary": str(output_dir / "summary.json"),
            "monitor_config": str(output_dir / "monitor_dataset_config.yaml"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _collect_episodes(
    *,
    env,
    model: DoubleDQN,
    handle,
    split: str,
    num_episodes: int,
    epsilon: float,
    seed: int,
    rng: random.Random,
    initial_monitor_string: str,
) -> dict[str, Any]:
    rows = 0
    successes = 0
    for episode_index in range(num_episodes):
        observation, info = env.reset(seed=seed + episode_index)
        monitor_string = initial_monitor_string
        terminated = False
        truncated = False
        step_id = 0
        final_base_reward = 0.0
        sampled_n = int(info.get("sampled_n", 0))

        while not terminated and not truncated:
            q_values = _teacher_q_values(model, observation)
            teacher_action = int(np.argmax(q_values))
            behavior_action = teacher_action
            if epsilon > 0.0 and rng.random() < epsilon:
                behavior_action = int(env.action_space.sample())

            next_observation, reward, terminated, truncated, info = env.step(behavior_action)
            next_monitor_string = str(info.get("monitor_state_unencoded", monitor_string))
            final_base_reward = float(info.get("base_reward", reward))
            row = {
                "split": split,
                "episode_id": episode_index,
                "step_id": step_id,
                "n_value": sampled_n,
                "env_obs": _array_list(observation["position"]),
                "monitor_state_string": monitor_string,
                "teacher_q_values": q_values.tolist(),
                "teacher_action": teacher_action,
                "behavior_action": behavior_action,
                "reward": float(reward),
                "base_reward": final_base_reward,
                "done": bool(terminated or truncated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "next_env_obs": _array_list(next_observation["position"]),
                "next_monitor_state_string": next_monitor_string,
                "task_index": int(info.get("task_index", -1)),
                "task_string": str(info.get("task_string", "")),
                "task_failed": bool(info.get("task_failed", False)),
            }
            handle.write(json.dumps(row) + "\n")
            rows += 1
            step_id += 1
            observation = next_observation
            monitor_string = next_monitor_string

        successes += int(final_base_reward >= 1.0)

    return {
        "split": split,
        "episodes": num_episodes,
        "rows": rows,
        "successes": successes,
        "success_rate": float(successes / num_episodes) if num_episodes else 0.0,
    }


def _teacher_q_values(model: DoubleDQN, observation: dict[str, Any]) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    return q_values.detach().cpu().numpy().reshape(-1).astype(float)


def _array_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-value", type=int, default=5)
    parser.add_argument("--fixed-n", type=int, default=None)
    parser.add_argument("--greedy-episodes", type=int, default=200)
    parser.add_argument("--epsilon-episodes", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--monitor-progress-bonus", type=float, default=10.0)
    parser.add_argument("--monitor-regression-penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_teacher_dataset(
        TeacherDatasetConfig(
            teacher_model_path=args.teacher_model_path,
            output_dir=args.output_dir,
            n_value=args.n_value,
            fixed_n=args.fixed_n,
            greedy_episodes=args.greedy_episodes,
            epsilon_episodes=args.epsilon_episodes,
            epsilon=args.epsilon,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            monitor_progress_bonus=args.monitor_progress_bonus,
            monitor_regression_penalty=args.monitor_regression_penalty,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
