"""Generate report figures for randomized LetterEnv experiments."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = REPO_ROOT / "envs" / "randomized_letter_env"
DEFAULT_RESULTS_DIR = ENV_ROOT / "results_and_evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"

FIGURE_NAMES = (
    "regional_method_summary",
    "regional_learning_curves",
    "full_random_learning_curve",
    "regional_zero_shot_success",
)
ZERO_SHOT_N_VALUES = [10, 15, 20]

METHOD_LABELS = {
    "ddqn": "DDQN",
    "q_learning": "Q-learning",
}
VARIANT_LABELS = {
    "full_random_n1": "Full random\nn=1",
    "regional_randomness_n1to5": "Regional random\nn=1..5",
}
COLORS = {
    "ddqn": "#4C78A8",
    "q_learning": "#E45756",
    "full_random_n1": "#F58518",
    "regional_randomness_n1to5": "#54A24B",
    "zero_shot": "#4C78A8",
}


@dataclass(frozen=True)
class FigureConfig:
    results_dir: Path
    output_dir: Path
    formats: tuple[str, ...]
    figures: tuple[str, ...]


def main() -> None:
    config = parse_args()
    configure_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ddqn_regional = load_ddqn_regional(config.results_dir)
    q_learning = load_q_learning(config.results_dir)
    ddqn_full_random = load_ddqn_full_random(config.results_dir)
    zero_shot = load_zero_shot(config.results_dir)

    if ddqn_regional.empty:
        raise FileNotFoundError("No regional DDQN eval_metrics.csv files found.")
    if q_learning.empty:
        raise FileNotFoundError("No Q-learning eval_metrics.csv files found.")

    if "regional_method_summary" in config.figures:
        plot_regional_method_summary(ddqn_regional, q_learning, config)
    if "regional_learning_curves" in config.figures:
        plot_regional_learning_curves(ddqn_regional, q_learning, config)
    if "full_random_learning_curve" in config.figures:
        if ddqn_full_random.empty:
            raise FileNotFoundError("No full-random DDQN eval_metrics.csv files found.")
        plot_full_random_learning_curve(ddqn_full_random, config)
    if "regional_zero_shot_success" in config.figures:
        if zero_shot.empty:
            raise FileNotFoundError("No zero-shot eval_metrics.csv files found.")
        plot_regional_zero_shot_success(zero_shot, config)

    print(f"Wrote figures to {config.output_dir}")


def parse_args() -> FigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"), choices=("pdf", "png", "svg"))
    parser.add_argument("--figures", nargs="+", default=FIGURE_NAMES, choices=FIGURE_NAMES)
    args = parser.parse_args()
    return FigureConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        figures=tuple(args.figures),
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.7, 4.2),
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


def load_ddqn_regional(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "ddqn" / "regional_randomness_n1to5"
    pattern = re.compile(r"semantic_progress_n1to5_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/eval_metrics.csv")):
        match = pattern.match(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = "ddqn"
        frame["variant"] = "regional_randomness_n1to5"
        frame["seed"] = int(match.group("seed"))
        rows.append(frame)
    return _normalize_numeric(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


def load_q_learning(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "q_learning"
    pattern = re.compile(r"semantic_progress_n1to5_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/eval_metrics.csv")):
        match = pattern.match(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = "q_learning"
        frame["variant"] = "regional_randomness_n1to5"
        frame["seed"] = int(match.group("seed"))
        rows.append(frame)
    return _normalize_numeric(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


def load_ddqn_full_random(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "ddqn" / "full_random_n1"
    pattern = re.compile(r"semantic_progress_terminal_30_expfra02_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/eval_metrics.csv")):
        match = pattern.match(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = "ddqn"
        frame["variant"] = "full_random_n1"
        frame["seed"] = int(match.group("seed"))
        rows.append(frame)
    return _normalize_numeric(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


def load_zero_shot(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "generalization"
    pattern = re.compile(r"semantic_progress_regional_zeroshot_n(?P<n>[0-9]+)_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/eval_metrics.csv")):
        match = pattern.match(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["variant"] = "regional_randomness_n1to5"
        frame["train_seed"] = int(match.group("seed"))
        frame["eval_n"] = int(match.group("n"))
        rows.append(frame)
    return _normalize_numeric(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()


def _normalize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "training_steps",
        "training_episodes",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "eval_success_rate",
        "eval_task_failure_rate",
        "eval_timeout_rate",
        "seed",
        "train_seed",
        "eval_n",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def plot_regional_method_summary(
    ddqn_regional: pd.DataFrame,
    q_learning: pd.DataFrame,
    config: FigureConfig,
) -> None:
    final_rows = pd.concat(
        [
            final_by_seed(ddqn_regional, "training_steps"),
            final_by_seed(q_learning, "training_episodes"),
        ],
        ignore_index=True,
    )
    summary = summarize(final_rows, ["algorithm"], "eval_success_rate")
    summary.to_csv(config.output_dir / "regional_method_summary.csv", index=False)

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    bars = ax.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        color=[COLORS[item] for item in summary["algorithm"]],
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )
    ax.set_title("Regional randomized LetterEnv: final success")
    ax.set_ylabel("Final evaluation success rate")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[item] for item in summary["algorithm"]])
    ax.grid(axis="x", visible=False)
    for bar, row in zip(bars, summary.itertuples(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.mean + row.std + 0.035,
            f"{row.mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save_figure(fig, config, "regional_method_summary")


def plot_regional_learning_curves(
    ddqn_regional: pd.DataFrame,
    q_learning: pd.DataFrame,
    config: FigureConfig,
) -> None:
    ddqn_summary = summarize(ddqn_regional, ["training_steps"], "eval_success_rate")
    q_summary = summarize(q_learning, ["training_episodes"], "eval_success_rate")
    ddqn_summary.to_csv(config.output_dir / "regional_ddqn_learning_curve.csv", index=False)
    q_summary.to_csv(config.output_dir / "regional_q_learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), sharey=True)
    draw_mean_std_curve(
        axes[0],
        ddqn_summary,
        x_column="training_steps",
        x_scale=1000.0,
        color=COLORS["ddqn"],
        label="DDQN",
    )
    axes[0].set_title("DDQN")
    axes[0].set_xlabel("Training steps (thousands)")
    axes[0].set_ylabel("Evaluation success rate")

    draw_mean_std_curve(
        axes[1],
        q_summary,
        x_column="training_episodes",
        x_scale=1000.0,
        color=COLORS["q_learning"],
        label="Q-learning",
    )
    axes[1].set_title("Q-learning")
    axes[1].set_xlabel("Training episodes (thousands)")

    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlim(left=0)
    fig.suptitle("Regional randomized LetterEnv learning curves", y=1.02)
    fig.tight_layout()
    save_figure(fig, config, "regional_learning_curves")


def plot_full_random_learning_curve(ddqn_full_random: pd.DataFrame, config: FigureConfig) -> None:
    summary = summarize(ddqn_full_random, ["training_steps"], "eval_success_rate")
    summary.to_csv(config.output_dir / "full_random_ddqn_learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    draw_mean_std_curve(
        ax,
        summary,
        x_column="training_steps",
        x_scale=1000.0,
        color=COLORS["full_random_n1"],
        label="DDQN full random n=1",
    )
    ax.set_title("Full-random LetterEnv learning curve")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Evaluation success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlim(left=0)
    fig.tight_layout()
    save_figure(fig, config, "full_random_ddqn_learning_curve")


def plot_regional_zero_shot_success(zero_shot: pd.DataFrame, config: FigureConfig) -> None:
    subset = zero_shot[zero_shot["eval_n"].isin(ZERO_SHOT_N_VALUES)].sort_values("eval_n")
    subset.to_csv(config.output_dir / "regional_zero_shot_success.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    draw_train_range(ax)
    ax.plot(
        subset["eval_n"],
        subset["eval_success_rate"],
        color=COLORS["zero_shot"],
        marker="o",
        linewidth=2.2,
        markersize=6.0,
    )
    for row in subset.itertuples():
        ax.text(row.eval_n, row.eval_success_rate + 0.035, f"{row.eval_success_rate:.2f}", ha="center", fontsize=8)
    ax.set_title("Regional zero-shot generalization")
    ax.set_xlabel("Evaluation count n")
    ax.set_ylabel("Evaluation success rate")
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(-0.03, 1.08)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    fig.tight_layout()
    save_figure(fig, config, "regional_zero_shot_success")


def final_by_seed(frame: pd.DataFrame, step_column: str) -> pd.DataFrame:
    return frame.sort_values(step_column).groupby("seed", as_index=False).tail(1)


def summarize(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(mean=(value_column, "mean"), std=(value_column, "std"), count=(value_column, "count"))
        .sort_values(group_columns)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def draw_mean_std_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    x_column: str,
    x_scale: float,
    color: str,
    label: str,
) -> None:
    summary = summary.sort_values(x_column)
    x = summary[x_column].to_numpy(dtype=float) / x_scale
    y = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, linewidth=2.1, label=label)
    ax.fill_between(x, np.clip(y - std, 0.0, 1.0), np.clip(y + std, 0.0, 1.0), color=color, alpha=0.16, linewidth=0)


def draw_train_range(ax: plt.Axes) -> None:
    ax.axvspan(1, 5, color="#D8D8D8", alpha=0.36, linewidth=0)
    ax.text(3.0, 0.93, "train n=1..5", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#555555")


def save_figure(fig: plt.Figure, config: FigureConfig, name: str) -> None:
    for fmt in config.formats:
        fig.savefig(config.output_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
