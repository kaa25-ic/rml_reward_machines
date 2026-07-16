"""Generate report figures for LunarLander protocol experiments."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from envs.lunar_lander.builder import _lunar_hover_count


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "envs" / "lunar_lander"
DEFAULT_RESULTS_DIR = ENV_ROOT / "results_and_evaluation"
DEFAULT_TWO_STAGE_DIR = DEFAULT_RESULTS_DIR / "ppo" / "two_stage_training"
DEFAULT_RENDERING_DIR = (
    DEFAULT_TWO_STAGE_DIR / "rendering" / "stage2_stabilization_model_final"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"

FIGURE_NAMES = (
    "learning_curves",
    "learning_curve_seed",
    "landing_protocol_gap",
    "phase_trajectory",
)
SUCCESSFUL_SEED_PROTOCOL_THRESHOLD = 0.5
PHASE_LABELS = {
    -1000: "Failure",
    0: "Corridor",
    1: "Hover entry",
    2: "Hover count",
    3: "Controlled descent",
    4: "Landing",
    5: "Success",
}
PHASE_COLORS = {
    -1000: "#9CA3AF",
    0: "#6B7280",
    1: "#4C78A8",
    2: "#54A24B",
    3: "#F58518",
    4: "#B279A2",
    5: "#E45756",
}


@dataclass(frozen=True)
class FigureConfig:
    two_stage_dir: Path
    rendering_dir: Path
    output_dir: Path
    formats: tuple[str, ...]
    figures: tuple[str, ...]
    run_prefix: str
    seed_figure_seed: int


def main() -> None:
    config = parse_args()
    configure_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_two_stage_metrics(config.two_stage_dir, config.run_prefix)
    if metrics.empty:
        raise FileNotFoundError(
            f"No two-stage eval_metrics.csv files found under {config.two_stage_dir}"
        )

    if "learning_curves" in config.figures:
        plot_learning_curves(metrics, config)
    if "learning_curve_seed" in config.figures:
        plot_seed_learning_curve(metrics, config)
    if "landing_protocol_gap" in config.figures:
        plot_landing_protocol_gap(metrics, config)
    if "phase_trajectory" in config.figures:
        trajectory, trajectory_source = load_successful_trajectory(config.rendering_dir)
        plot_phase_trajectory(trajectory, trajectory_source, config)

    print(f"Wrote LunarLander figures to {config.output_dir}")


def parse_args() -> FigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-stage-dir", type=Path, default=DEFAULT_TWO_STAGE_DIR)
    parser.add_argument("--rendering-dir", type=Path, default=DEFAULT_RENDERING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "png"),
        choices=("pdf", "png", "svg"),
    )
    parser.add_argument("--figures", nargs="+", default=FIGURE_NAMES, choices=FIGURE_NAMES)
    parser.add_argument(
        "--seed-figure-seed",
        type=int,
        default=0,
        help="Seed used for the single-seed learning-curve figure.",
    )
    parser.add_argument(
        "--run-prefix",
        default="semantic_progress_two_stage_seed",
        help="Two-stage run directory prefix before the seed number.",
    )
    args = parser.parse_args()
    return FigureConfig(
        two_stage_dir=args.two_stage_dir,
        rendering_dir=args.rendering_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        figures=tuple(args.figures),
        run_prefix=str(args.run_prefix),
        seed_figure_seed=int(args.seed_figure_seed),
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.8, 4.2),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_two_stage_metrics(two_stage_dir: Path, run_prefix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in sorted(two_stage_dir.glob(f"{run_prefix}*")):
        if not run_dir.is_dir():
            continue
        seed_text = run_dir.name.removeprefix(run_prefix)
        if not seed_text.isdigit():
            continue
        seed = int(seed_text)
        stage1 = _read_stage_metrics(run_dir / "stage1_discovery", seed, "stage1_discovery")
        stage2 = _read_stage_metrics(run_dir / "stage2_stabilization", seed, "stage2_stabilization")
        if stage1.empty or stage2.empty:
            continue
        stage1_total = int(stage1["training_steps"].max())
        stage1["global_training_steps"] = stage1["training_steps"]
        stage1["stage_boundary_steps"] = stage1_total
        stage2["global_training_steps"] = stage1_total + stage2["training_steps"]
        stage2["stage_boundary_steps"] = stage1_total
        rows.extend([stage1, stage2])
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    for column in [
        "training_steps",
        "global_training_steps",
        "stage_boundary_steps",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "eval_successful_landing_rate",
        "eval_successful_protocol_rate",
        "eval_hover_complete_rate",
        "eval_controlled_descent_rate",
        "eval_task_failure_rate",
        "seed",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["landing_protocol_gap"] = (
        frame["eval_successful_landing_rate"] - frame["eval_successful_protocol_rate"]
    )
    return frame.sort_values(["seed", "global_training_steps"]).reset_index(drop=True)


def _read_stage_metrics(stage_dir: Path, seed: int, stage: str) -> pd.DataFrame:
    metrics_path = stage_dir / "eval_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(metrics_path)
    frame["seed"] = seed
    frame["stage"] = stage
    frame["run_dir"] = str(stage_dir)
    return frame


def load_successful_trajectory(rendering_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    index_path = rendering_dir / "render_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No render_index.csv found at {index_path}")
    index = pd.read_csv(index_path)
    successful = index[index["successful_protocol"].astype(str).str.lower() == "true"]
    if successful.empty:
        raise FileNotFoundError(f"No successful rendered protocol episodes found in {index_path}")
    source = successful.sort_values(["episode_return", "episode_length"], ascending=[False, True]).iloc[0]
    trajectory_path = REPO_ROOT / str(source["trajectory_path"])
    trajectory = pd.read_csv(trajectory_path)
    for column in [
        "step",
        "x",
        "y",
        "monitor_progress",
        "corridor",
        "hover",
        "controlled_descent",
        "target_zone",
        "safe_landing_angle",
        "both_contact",
    ]:
        trajectory[column] = pd.to_numeric(trajectory[column], errors="coerce")
    trajectory["phase"] = trajectory["monitor_progress"].apply(_phase_from_progress)
    trajectory["hover_count"] = trajectory["monitor_state"].apply(_lunar_hover_count)
    trajectory["hover_count_display"] = trajectory["hover_count"].clip(upper=2)
    trajectory.loc[
        (trajectory["phase"] == 1) & (trajectory["hover_count_display"] > 0),
        "phase",
    ] = 2
    trajectory["phase_label"] = trajectory["phase"].map(PHASE_LABELS)
    return trajectory, source


def plot_learning_curves(metrics: pd.DataFrame, config: FigureConfig) -> None:
    successful_seeds = successful_stage2_final_seeds(metrics)
    curve_metrics = metrics[metrics["seed"].isin(successful_seeds)].copy()
    if curve_metrics.empty:
        raise FileNotFoundError("No successful seeds found for the learning-curve figure.")
    summary = summarize_curve(
        curve_metrics,
        ["global_training_steps"],
        ["eval_successful_landing_rate", "eval_successful_protocol_rate"],
    )
    summary["included_seeds"] = ",".join(str(seed) for seed in successful_seeds)
    summary["excluded_seeds"] = ",".join(
        str(seed) for seed in sorted(set(metrics["seed"]) - set(successful_seeds))
    )
    summary.to_csv(config.output_dir / "lunar_two_stage_learning_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    draw_mean_std_curve(
        ax,
        summary,
        "eval_successful_landing_rate",
        "Successful landing",
        "#4C78A8",
    )
    draw_mean_std_curve(
        ax,
        summary,
        "eval_successful_protocol_rate",
        "RML protocol success",
        "#E45756",
    )
    draw_stage_boundary(ax, curve_metrics)
    ax.set_title("LunarLander: landing vs RML protocol success")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Evaluation success rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, config, "lunar_learning_landing_vs_protocol")


def plot_seed_learning_curve(metrics: pd.DataFrame, config: FigureConfig) -> None:
    subset = metrics[metrics["seed"] == config.seed_figure_seed].sort_values(
        "global_training_steps"
    )
    if subset.empty:
        raise FileNotFoundError(f"No two-stage metrics found for seed {config.seed_figure_seed}")
    output = subset[
        [
            "seed",
            "stage",
            "training_steps",
            "global_training_steps",
            "eval_successful_landing_rate",
            "eval_successful_protocol_rate",
            "landing_protocol_gap",
        ]
    ].copy()
    output.to_csv(
        config.output_dir
        / f"lunar_learning_landing_vs_protocol_seed{config.seed_figure_seed}.csv",
        index=False,
    )

    x = subset["global_training_steps"].to_numpy(dtype=float) / 1_000_000.0
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        x,
        subset["eval_successful_landing_rate"],
        color="#4C78A8",
        marker="o",
        linewidth=2.0,
        label="Successful landing",
    )
    ax.plot(
        x,
        subset["eval_successful_protocol_rate"],
        color="#E45756",
        marker="s",
        linewidth=2.0,
        label="RML protocol success",
    )
    draw_stage_boundary(ax, subset)
    ax.set_title(f"LunarLander seed {config.seed_figure_seed}: landing vs RML protocol")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Evaluation success rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(
        fig,
        config,
        f"lunar_learning_landing_vs_protocol_seed{config.seed_figure_seed}",
    )


def plot_landing_protocol_gap(metrics: pd.DataFrame, config: FigureConfig) -> None:
    summary = summarize_curve(metrics, ["global_training_steps"], ["landing_protocol_gap"])
    summary.to_csv(config.output_dir / "lunar_landing_protocol_gap.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    draw_mean_std_curve(
        ax,
        summary,
        "landing_protocol_gap",
        "Landing success - protocol success",
        "#F58518",
    )
    draw_stage_boundary(ax, metrics)
    ax.axhline(0.0, color="#374151", linewidth=0.9, alpha=0.8)
    ax.set_title("RML protocol strictness beyond landing")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Success-rate gap")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, config, "lunar_landing_protocol_gap")


def plot_phase_trajectory(
    trajectory: pd.DataFrame,
    source: pd.Series,
    config: FigureConfig,
) -> None:
    output = trajectory[
        [
            "step",
            "x",
            "y",
            "monitor_progress",
            "phase",
            "phase_label",
            "hover_count_display",
        ]
    ].copy()
    output.to_csv(config.output_dir / "lunar_phase_trajectory.csv", index=False)

    points = trajectory[["x", "y"]].to_numpy(dtype=float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_phases = trajectory["phase"].iloc[:-1].to_numpy(dtype=int)
    colors = [PHASE_COLORS[int(phase)] for phase in segment_phases]

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2.4, alpha=0.95))
    ax.scatter(
        points[0, 0],
        points[0, 1],
        facecolors="white",
        edgecolors=PHASE_COLORS[1],
        linewidths=2.3,
        marker="o",
        s=120,
        label="Hover entry point",
        zorder=4,
    )
    ax.scatter(points[0, 0], points[0, 1], color="#111827", marker="o", s=28, label="Start", zorder=5)
    ax.scatter(points[-1, 0], points[-1, 1], color="#111827", marker="*", s=90, label="End", zorder=3)
    ax.plot([-0.2, 0.2], [0.0, 0.0], color="#111827", linewidth=2.0, alpha=0.85)

    for phase in sorted(set(segment_phases)):
        ax.plot([], [], color=PHASE_COLORS[int(phase)], linewidth=3, label=PHASE_LABELS[int(phase)])
    ax.set_title(f"Phase-colored RML landing trajectory ({short_run_label(source)})")
    ax.set_xlabel("Lander x position")
    ax.set_ylabel("Lander y position")
    ax.set_xlim(min(trajectory["x"].min() - 0.08, -0.3), max(trajectory["x"].max() + 0.08, 0.3))
    ax.set_ylim(min(-0.04, trajectory["y"].min() - 0.05), trajectory["y"].max() + 0.08)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    save_figure(fig, config, "lunar_phase_colored_trajectory")


def summarize_curve(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    pieces = []
    for value_column in value_columns:
        summary = (
            frame.groupby(group_columns)[value_column]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{value_column}_mean",
                    "std": f"{value_column}_std",
                    "count": f"{value_column}_count",
                }
            )
        )
        pieces.append(summary)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=group_columns, how="outer")
    return result.sort_values(group_columns).reset_index(drop=True)


def draw_mean_std_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    value_column: str,
    label: str,
    color: str,
) -> None:
    x = summary["global_training_steps"].to_numpy(dtype=float) / 1_000_000.0
    mean = summary[f"{value_column}_mean"].to_numpy(dtype=float)
    std = summary[f"{value_column}_std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, mean, color=color, linewidth=2.2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


def draw_stage_boundary(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    boundary = float(metrics["stage_boundary_steps"].median()) / 1_000_000.0
    ax.axvline(boundary, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        boundary,
        1.02,
        "Stage 2 starts",
        ha="right",
        va="top",
        rotation=90,
        fontsize=8.5,
        color="#111827",
    )


def _phase_from_progress(progress: float) -> int:
    if progress <= -999:
        return -1000
    return int(max(0, min(5, round(float(progress)))))


def successful_stage2_final_seeds(metrics: pd.DataFrame) -> list[int]:
    successful: list[int] = []
    for seed, group in metrics.groupby("seed"):
        stage2 = group[group["stage"] == "stage2_stabilization"].sort_values("training_steps")
        if stage2.empty:
            continue
        final_protocol_rate = float(stage2["eval_successful_protocol_rate"].iloc[-1])
        if final_protocol_rate >= SUCCESSFUL_SEED_PROTOCOL_THRESHOLD:
            successful.append(int(seed))
    return sorted(successful)


def short_run_label(source: pd.Series) -> str:
    run_name = str(source["run_name"])
    parts = run_name.split("_")
    seed = next((part.removeprefix("seed") for part in parts if part.startswith("seed")), "?")
    return f"seed {seed} stage-2 final policy"


def save_figure(fig: plt.Figure, config: FigureConfig, name: str) -> None:
    for fmt in config.formats:
        fig.savefig(config.output_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
