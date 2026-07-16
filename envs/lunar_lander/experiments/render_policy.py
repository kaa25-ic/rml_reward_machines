"""Render trained RML LunarLander PPO policies.

Examples from the repository root:

    python envs/lunar_lander/experiments/render_policy.py \\
        --run-dir envs/lunar_lander/results_and_evaluation/ppo/two_stage_training/semantic_progress_two_stage_seed0/stage2_stabilization \\
        --model model_final \\
        --record-video

    python envs/lunar_lander/experiments/render_policy.py \\
        --runs-root envs/lunar_lander/results_and_evaluation/ppo/two_stage_training \\
        --stage stage2_stabilization \\
        --model model_final \\
        --record-gif
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from envs.lunar_lander import LunarLanderProtocolConfig, build_lunar_lander_protocol_env
from envs.lunar_lander.builder import _lunar_monitor_progress
from envs.lunar_lander.experiments.train_ppo import (
    DEFAULT_MONITOR_CONFIG,
    DEFAULT_MONITOR_SPEC,
    LUNAR_ENV_ROOT,
)
from rml_rm.experiments.runtime import json_ready, managed_monitor, write_json


DEFAULT_RUNS_ROOT = (
    LUNAR_ENV_ROOT / "results_and_evaluation" / "ppo" / "two_stage_training"
)
DEFAULT_STAGE = "stage2_stabilization"


@dataclass(frozen=True)
class RenderEpisodeSummary:
    """Summary for one rendered policy episode."""

    run_name: str
    run_dir: str
    model_path: str
    episode_index: int
    seed: int
    episode_return: float
    episode_length: int
    successful_landing: bool
    successful_protocol: bool
    task_failed: bool
    max_monitor_progress: float
    final_lunar_base_reward: float
    final_monitor_reward: float
    terminated: bool
    truncated: bool
    video_dir: str | None
    gif_path: str | None
    trajectory_path: str


def render_runs(args: argparse.Namespace) -> list[RenderEpisodeSummary]:
    """Render one or more training runs and save per-episode artifacts."""
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        raise FileNotFoundError("No run directories found to render.")
    if args.n_episodes < 1:
        raise ValueError("n_episodes must be at least 1.")
    if args.live and (args.record_video or args.record_gif):
        raise ValueError("--live cannot be combined with --record-video or --record-gif.")

    output_root = resolve_output_root(args, run_dirs)
    summaries: list[RenderEpisodeSummary] = []
    for run_dir in run_dirs:
        run_output_dir = output_root / render_run_label(run_dir)
        run_summaries = render_run(
            run_dir=run_dir,
            model_path=resolve_model_path(args, run_dir),
            output_dir=run_output_dir,
            args=args,
        )
        summaries.extend(run_summaries)

    write_render_index(output_root, summaries)
    return summaries


def render_run(
    *,
    run_dir: Path,
    model_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[RenderEpisodeSummary]:
    """Render episodes for one run directory."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    training_config = load_training_config(run_dir)
    monitor_config = args.monitor_config or load_monitor_config(run_dir)
    monitor_spec = args.monitor_spec or load_monitor_spec(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_mode = "human" if args.live else "rgb_array"
    env_config = env_config_from_training_config(
        training_config,
        render_mode=render_mode,
        max_episode_steps=args.max_episode_steps,
    )

    summaries: list[RenderEpisodeSummary] = []
    with managed_monitor(
        output_dir=output_dir,
        monitor_config_template=monitor_config,
        monitor_spec_path=monitor_spec,
        log_name="render_rml_monitor.log",
        config_name="render_monitor_config.yaml",
        max_episode_steps=env_config.max_episode_steps,
    ) as monitor_runtime:
        env = build_lunar_lander_protocol_env(
            env_config,
            monitor_config_path=monitor_runtime.config_path,
        )
        env = maybe_record_video(
            env,
            record_video=args.record_video,
            output_dir=output_dir,
            run_name=render_run_label(run_dir),
        )
        try:
            model = PPO.load(str(model_path), env=env, print_system_info=False)
            for episode_index in range(args.n_episodes):
                seed = args.seed + episode_index
                summary = run_episode(
                    env=env,
                    model=model,
                    run_dir=run_dir,
                    model_path=model_path,
                    output_dir=output_dir,
                    episode_index=episode_index,
                    seed=seed,
                    delay=args.delay,
                    render_live=args.live,
                    record_gif=args.record_gif,
                    fps=args.fps,
                )
                summaries.append(summary)
        finally:
            env.close()

    write_run_summary(output_dir, summaries, training_config)
    return summaries


def run_episode(
    *,
    env: gym.Env,
    model: PPO,
    run_dir: Path,
    model_path: Path,
    output_dir: Path,
    episode_index: int,
    seed: int,
    delay: float,
    render_live: bool,
    record_gif: bool,
    fps: int,
) -> RenderEpisodeSummary:
    """Run one deterministic policy episode and write its step trace."""
    observation, _ = env.reset(seed=seed)
    trajectory_path = output_dir / f"episode_{episode_index:03d}_trajectory.csv"
    frames: list[Any] = []
    if record_gif:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    fieldnames = [
        "step",
        "action",
        "reward",
        "episode_return",
        "x",
        "y",
        "vx",
        "vy",
        "angle",
        "angular_velocity",
        "left_contact",
        "right_contact",
        "corridor",
        "hover",
        "controlled_descent",
        "target_zone",
        "safe_landing_angle",
        "both_contact",
        "successful_landing",
        "successful_protocol",
        "task_failed",
        "monitor_progress",
        "monitor_state",
    ]

    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    max_monitor_progress = 0.0

    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while not terminated and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            scalar_action = int(np.asarray(action).reshape(-1)[0])
            observation, reward, terminated, truncated, info = env.step(scalar_action)
            final_info = dict(info)
            episode_return += float(reward)
            episode_length += 1
            progress = _lunar_monitor_progress(info.get("monitor_state_unencoded"))
            max_monitor_progress = max(max_monitor_progress, progress)
            state = np.asarray(observation["position"], dtype=float)
            writer.writerow(
                {
                    "step": episode_length,
                    "action": scalar_action,
                    "reward": float(reward),
                    "episode_return": episode_return,
                    "x": state[0],
                    "y": state[1],
                    "vx": state[2],
                    "vy": state[3],
                    "angle": state[4],
                    "angular_velocity": state[5],
                    "left_contact": state[6],
                    "right_contact": state[7],
                    "corridor": info.get("corridor", 0.0),
                    "hover": info.get("hover", 0.0),
                    "controlled_descent": info.get("controlled_descent", 0.0),
                    "target_zone": info.get("target_zone", 0.0),
                    "safe_landing_angle": info.get("safe_landing_angle", 0.0),
                    "both_contact": info.get("both_contact", 0.0),
                    "successful_landing": bool(info.get("successful_landing", False)),
                    "successful_protocol": bool(info.get("successful_protocol", False)),
                    "task_failed": bool(info.get("task_failed", False)),
                    "monitor_progress": progress,
                    "monitor_state": info.get("monitor_state_unencoded", ""),
                }
            )

            if render_live:
                env.render()
            elif record_gif:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if delay > 0.0:
                time.sleep(delay)

    gif_path = None
    if record_gif:
        gif_path = output_dir / f"episode_{episode_index:03d}.gif"
        save_gif(frames, gif_path, fps=fps)

    summary = RenderEpisodeSummary(
        run_name=render_run_label(run_dir),
        run_dir=str(run_dir),
        model_path=str(model_path),
        episode_index=episode_index,
        seed=seed,
        episode_return=float(episode_return),
        episode_length=int(episode_length),
        successful_landing=bool(final_info.get("successful_landing", False)),
        successful_protocol=bool(final_info.get("successful_protocol", False)),
        task_failed=bool(final_info.get("task_failed", False)),
        max_monitor_progress=float(max_monitor_progress),
        final_lunar_base_reward=float(final_info.get("lunar_base_reward", 0.0)),
        final_monitor_reward=float(final_info.get("monitor_terminal_reward", 0.0)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        video_dir=str(output_dir / "videos") if (output_dir / "videos").exists() else None,
        gif_path=str(gif_path) if gif_path is not None else None,
        trajectory_path=str(trajectory_path),
    )
    write_json(
        output_dir / f"episode_{episode_index:03d}_summary.json",
        asdict(summary),
    )
    print_episode_summary(summary)
    return summary


def maybe_record_video(
    env: gym.Env,
    *,
    record_video: bool,
    output_dir: Path,
    run_name: str,
) -> gym.Env:
    """Wrap the env with RecordVideo when video output is requested."""
    if not record_video:
        return env
    require_video_dependencies()
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    return RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=run_name,
        disable_logger=True,
    )


def require_video_dependencies() -> None:
    """Fail early with a clear message if video recording dependencies are absent."""
    if importlib.util.find_spec("moviepy") is None:
        raise ModuleNotFoundError(
            "Recording videos requires moviepy in the active environment."
        )


def save_gif(frames: list[Any], path: Path, *, fps: int) -> None:
    """Save RGB frames as a GIF using Pillow."""
    if not frames:
        raise RuntimeError("No frames were captured for GIF output.")
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Recording GIFs requires Pillow in the active environment."
        ) from exc

    duration_ms = max(int(1000 / max(fps, 1)), 1)
    images = [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir is not None:
        return [args.run_dir]
    if args.runs_root is None:
        return [DEFAULT_RUNS_ROOT / "semantic_progress_two_stage_seed0" / DEFAULT_STAGE]
    return sorted(
        path
        for path in args.runs_root.glob(f"*/{args.stage}")
        if (path / "config.json").exists()
    )


def resolve_model_path(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.model_path is not None:
        return args.model_path
    return run_dir / f"{args.model}.zip"


def resolve_output_root(args: argparse.Namespace, run_dirs: list[Path]) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.run_dir is not None and len(run_dirs) == 1:
        return run_dirs[0] / "rendering" / args.model
    root = args.runs_root or DEFAULT_RUNS_ROOT
    return root / "rendering" / f"{args.stage}_{args.model}"


def load_training_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return dict(config.get("training_config", {}))


def load_monitor_config(run_dir: Path) -> Path:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        monitor_config = config.get("monitor", {}).get("eval_config_path")
        if monitor_config:
            path = Path(monitor_config)
            if path.exists():
                return path
    return DEFAULT_MONITOR_CONFIG


def load_monitor_spec(run_dir: Path) -> Path:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        monitor_spec = config.get("monitor", {}).get("spec_path")
        if monitor_spec:
            path = Path(monitor_spec)
            if path.exists():
                return path
    return DEFAULT_MONITOR_SPEC


def env_config_from_training_config(
    training_config: dict[str, Any],
    *,
    render_mode: str,
    max_episode_steps: int | None,
) -> LunarLanderProtocolConfig:
    """Rebuild the monitored environment from a saved training config."""
    return LunarLanderProtocolConfig(
        encoding=str(training_config.get("encoding", "semantic_progress")),
        max_episode_steps=int(
            max_episode_steps
            if max_episode_steps is not None
            else training_config.get("max_episode_steps", 1000)
        ),
        monitor_progress_bonus=float(training_config.get("monitor_progress_bonus", 20.0)),
        hover_step_bonus=float(training_config.get("hover_step_bonus", 2.0)),
        hover_complete_bonus=float(training_config.get("hover_complete_bonus", 30.0)),
        controlled_descent_bonus=float(training_config.get("controlled_descent_bonus", 20.0)),
        success_bonus=float(training_config.get("success_bonus", 200.0)),
        failure_penalty=float(training_config.get("failure_penalty", -100.0)),
        landing_target_bonus=float(training_config.get("landing_target_bonus", 0.0)),
        landing_angle_bonus=float(training_config.get("landing_angle_bonus", 0.0)),
        post_descent_landing_bonus=float(
            training_config.get("post_descent_landing_bonus", 0.0)
        ),
        post_descent_protocol_miss_penalty=float(
            training_config.get("post_descent_protocol_miss_penalty", 0.0)
        ),
        render_mode=render_mode,
    )


def render_run_label(run_dir: Path) -> str:
    if run_dir.name in {"stage1_discovery", "stage2_stabilization"}:
        return f"{run_dir.parent.name}_{run_dir.name}"
    return run_dir.name


def write_run_summary(
    output_dir: Path,
    summaries: list[RenderEpisodeSummary],
    training_config: dict[str, Any],
) -> None:
    write_json(
        output_dir / "render_summary.json",
        {
            "training_config": json_ready(training_config),
            "episodes": [asdict(summary) for summary in summaries],
        },
    )


def write_render_index(
    output_root: Path,
    summaries: list[RenderEpisodeSummary],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "render_index.json",
        {
            "episode_count": len(summaries),
            "successful_protocol_count": sum(
                1 for summary in summaries if summary.successful_protocol
            ),
            "successful_landing_count": sum(
                1 for summary in summaries if summary.successful_landing
            ),
            "episodes": [asdict(summary) for summary in summaries],
        },
    )
    csv_path = output_root / "render_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def print_episode_summary(summary: RenderEpisodeSummary) -> None:
    status = "SUCCESS" if summary.successful_protocol else "FAIL"
    print(
        f"{summary.run_name} episode={summary.episode_index} seed={summary.seed} "
        f"{status} return={summary.episode_return:.1f} "
        f"length={summary.episode_length} landing={summary.successful_landing} "
        f"protocol={summary.successful_protocol} progress={summary.max_monitor_progress:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, default=None)
    source.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--model", choices=("best_model", "model_final"), default="model_final")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--monitor-config", type=Path, default=None)
    parser.add_argument("--monitor-spec", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    summaries = render_runs(parse_args())
    protocol_successes = sum(1 for summary in summaries if summary.successful_protocol)
    print(
        f"Rendered {len(summaries)} episode(s): "
        f"protocol_successes={protocol_successes}/{len(summaries)}"
    )


if __name__ == "__main__":
    main()
