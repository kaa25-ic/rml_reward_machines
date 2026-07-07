"""Generate report figures for LetterEnv experiments."""

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
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_RESULTS_DIR = LETTER_ENV_ROOT / "results_and_evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"

ALGORITHM_ORDER = ["dqn", "ddqn", "ppo"]
ENCODING_ORDER = ["numerical", "one_hot", "semantic_progress", "learned_gru", "learned_graph"]
ZERO_SHOT_N_VALUES = [10, 15, 20]

ALGORITHM_LABELS = {
    "dqn": "DQN",
    "ddqn": "Double DQN",
    "ppo": "PPO",
}
ENCODING_LABELS = {
    "numerical": "Numerical",
    "one_hot": "One-hot",
    "semantic_progress": "Semantic progress",
    "learned_gru": "Learned GRU",
    "learned_graph": "Learned GNN",
}
COLORS = {
    "dqn": "#4C78A8",
    "ddqn": "#F58518",
    "ppo": "#54A24B",
    "numerical": "#4C78A8",
    "one_hot": "#F58518",
    "semantic_progress": "#54A24B",
    "learned_gru": "#B279A2",
    "learned_graph": "#E45756",
}
MARKERS = {
    "dqn": "o",
    "ddqn": "s",
    "ppo": "^",
    "numerical": "o",
    "one_hot": "s",
    "semantic_progress": "^",
    "learned_gru": "D",
    "learned_graph": "P",
}


@dataclass(frozen=True)
class FigureConfig:
    results_dir: Path
    output_dir: Path
    formats: tuple[str, ...]
    success_threshold: float
    max_learning_steps: int


def main() -> None:
    config = parse_args()
    configure_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    learning = load_learning_metrics(config.results_dir)
    zero_shot = load_zero_shot_metrics(config.results_dir)

    if learning.empty:
        raise FileNotFoundError(f"No learning eval_metrics.csv files found under {config.results_dir}")
    if zero_shot.empty:
        raise FileNotFoundError(f"No zero-shot eval_metrics.csv files found under {config.results_dir}")

    plot_learning_by_algorithm(learning, config)
    plot_learning_by_encoding(learning, config)
    plot_sample_efficiency(learning, config)
    plot_zero_shot_success(zero_shot, config)
    plot_zero_shot_episode_length(zero_shot, config)

    print(f"Wrote figures to {config.output_dir}")


def parse_args() -> FigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"), choices=("pdf", "png", "svg"))
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--max-learning-steps", type=int, default=250000)
    args = parser.parse_args()
    return FigureConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        success_threshold=float(args.success_threshold),
        max_learning_steps=int(args.max_learning_steps),
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.6, 4.2),
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


def load_learning_metrics(results_dir: Path) -> pd.DataFrame:
    base = results_dir / "experiments_with_variable_n"
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"(?P<encoding>.+)_n_1to5_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/*/eval_metrics.csv")):
        algorithm = path.parents[1].name
        run_name = path.parent.name
        match = pattern.match(run_name)
        if algorithm not in ALGORITHM_ORDER or match is None:
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = algorithm
        frame["encoding"] = match.group("encoding")
        frame["seed"] = int(match.group("seed"))
        frame["run_dir"] = str(path.parent)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    numeric_columns = [
        "training_steps",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "eval_success_rate",
        "eval_mean_terminal_base_reward",
        "eval_mean_terminal_task_progress",
        "eval_task_failure_rate",
        "seed",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_zero_shot_metrics(results_dir: Path) -> pd.DataFrame:
    base = results_dir / "generalization_experiments_with_zero_shot_on_larger_n"
    rows: list[pd.DataFrame] = []
    for path in sorted(base.glob("*/*/eval_metrics.csv")):
        frame = pd.read_csv(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    for column in [
        "train_seed",
        "eval_n",
        "n_eval_episodes",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "eval_success_rate",
        "eval_mean_terminal_base_reward",
        "eval_mean_terminal_task_progress",
        "eval_task_failure_rate",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def plot_learning_by_algorithm(learning: pd.DataFrame, config: FigureConfig) -> None:
    subset = learning[
        (learning["encoding"] == "numerical")
        & learning["algorithm"].isin(ALGORITHM_ORDER)
        & (learning["training_steps"] <= config.max_learning_steps)
    ]
    summary = summarize_curve(subset, ["algorithm", "training_steps"], "eval_success_rate")
    summary.to_csv(config.output_dir / "learning_by_algorithm_success_rate.csv", index=False)

    fig, ax = plt.subplots()
    for algorithm in ALGORITHM_ORDER:
        group = summary[summary["algorithm"] == algorithm]
        if group.empty:
            continue
        draw_curve(
            ax,
            group,
            label=ALGORITHM_LABELS[algorithm],
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
        )
    format_learning_axis(ax, title="Learning curves by algorithm", ylabel="Evaluation success rate")
    ax.legend(loc="lower right")
    save_figure(fig, config, "learning_by_algorithm_success_rate")


def plot_learning_by_encoding(learning: pd.DataFrame, config: FigureConfig) -> None:
    subset = learning[
        (learning["algorithm"] == "ddqn")
        & learning["encoding"].isin(ENCODING_ORDER)
        & (learning["training_steps"] <= config.max_learning_steps)
    ]
    summary = summarize_curve(subset, ["encoding", "training_steps"], "eval_success_rate")
    summary.to_csv(config.output_dir / "learning_by_encoding_success_rate.csv", index=False)

    fig, ax = plt.subplots()
    for encoding in ENCODING_ORDER:
        group = summary[summary["encoding"] == encoding]
        if group.empty:
            continue
        draw_curve(
            ax,
            group,
            label=ENCODING_LABELS[encoding],
            color=COLORS[encoding],
            marker=MARKERS[encoding],
        )
    format_learning_axis(ax, title="DDQN learning curves by encoding", ylabel="Evaluation success rate")
    ax.legend(loc="lower right", ncol=1)
    save_figure(fig, config, "learning_by_encoding_success_rate")


def plot_sample_efficiency(learning: pd.DataFrame, config: FigureConfig) -> None:
    efficiency = compute_sample_efficiency(learning, threshold=config.success_threshold)
    efficiency.to_csv(config.output_dir / "sample_efficiency_first_success.csv", index=False)

    labels = [
        f"{ALGORITHM_LABELS[row.algorithm]}\n{ENCODING_LABELS[row.encoding]}"
        for row in efficiency.itertuples()
    ]
    y = np.arange(len(efficiency))
    means = efficiency["mean_first_success_steps"].to_numpy(dtype=float) / 1000.0
    stds = efficiency["std_first_success_steps"].fillna(0.0).to_numpy(dtype=float) / 1000.0
    colors = [COLORS.get(row.encoding, "#555555") for row in efficiency.itertuples()]

    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.88, capsize=3, edgecolor="white", linewidth=0.7)
    ax.set_title(f"Sample efficiency to success rate >= {config.success_threshold:.1f}")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for index, row in enumerate(efficiency.itertuples()):
        if row.unsolved_seeds > 0:
            ax.text(means[index] + stds[index] + 4, index, f"{row.unsolved_seeds} unsolved", ha="left", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, config, "sample_efficiency_first_success")


def plot_zero_shot_success(zero_shot: pd.DataFrame, config: FigureConfig) -> None:
    subset = zero_shot[
        (zero_shot["algorithm"] == "ddqn")
        & (zero_shot["encoding"].isin(ENCODING_ORDER))
        & (zero_shot["eval_n"].isin(ZERO_SHOT_N_VALUES))
    ]
    summary = summarize_zero_shot(subset, "eval_success_rate")
    summary.to_csv(config.output_dir / "zero_shot_generalization_success_rate.csv", index=False)

    fig, ax = plt.subplots()
    draw_train_range(ax)
    for encoding in ENCODING_ORDER:
        group = summary[summary["encoding"] == encoding]
        if group.empty:
            continue
        draw_zero_shot_curve(ax, group, encoding=encoding, value_column="mean")
    format_zero_shot_axis(ax, title="DDQN zero-shot generalization", ylabel="Evaluation success rate")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=0.92)
    save_figure(fig, config, "zero_shot_generalization_success_rate")


def plot_zero_shot_episode_length(zero_shot: pd.DataFrame, config: FigureConfig) -> None:
    subset = zero_shot[
        (zero_shot["algorithm"] == "ddqn")
        & (zero_shot["encoding"].isin(ENCODING_ORDER))
        & (zero_shot["eval_n"].isin(ZERO_SHOT_N_VALUES))
    ]
    summary = summarize_zero_shot(subset, "eval_mean_episode_length")
    summary.to_csv(config.output_dir / "zero_shot_generalization_episode_length.csv", index=False)

    fig, ax = plt.subplots()
    draw_train_range(ax)
    for encoding in ENCODING_ORDER:
        group = summary[summary["encoding"] == encoding]
        if group.empty:
            continue
        draw_zero_shot_curve(ax, group, encoding=encoding, value_column="mean")
    format_zero_shot_axis(ax, title="DDQN zero-shot trajectory length", ylabel="Mean episode length")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="white", framealpha=0.92)
    save_figure(fig, config, "zero_shot_generalization_episode_length")


def summarize_curve(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            mean=(value_column, "mean"),
            std=(value_column, "std"),
            count=(value_column, "count"),
        )
        .sort_values(group_columns)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def summarize_zero_shot(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    summary = summarize_curve(frame, ["algorithm", "encoding", "eval_n"], value_column)
    return summary.sort_values(["encoding", "eval_n"])


def compute_sample_efficiency(learning: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    combinations = [
        ("dqn", "numerical"),
        ("dqn", "one_hot"),
        ("ddqn", "numerical"),
        ("ddqn", "one_hot"),
        ("ddqn", "semantic_progress"),
        ("ddqn", "learned_gru"),
        ("ddqn", "learned_graph"),
        ("ppo", "numerical"),
        ("ppo", "one_hot"),
    ]
    for algorithm, encoding in combinations:
        subset = learning[(learning["algorithm"] == algorithm) & (learning["encoding"] == encoding)]
        if subset.empty:
            continue
        first_steps: list[float] = []
        unsolved = 0
        for _seed, seed_frame in subset.groupby("seed"):
            solved = seed_frame[seed_frame["eval_success_rate"] >= threshold].sort_values("training_steps")
            if solved.empty:
                unsolved += 1
            else:
                first_steps.append(float(solved.iloc[0]["training_steps"]))
        rows.append(
            {
                "algorithm": algorithm,
                "encoding": encoding,
                "mean_first_success_steps": float(np.mean(first_steps)) if first_steps else np.nan,
                "std_first_success_steps": float(np.std(first_steps, ddof=1)) if len(first_steps) > 1 else 0.0,
                "solved_seeds": len(first_steps),
                "unsolved_seeds": unsolved,
                "total_seeds": int(subset["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def draw_curve(
    ax: plt.Axes,
    group: pd.DataFrame,
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    group = group.sort_values("training_steps")
    x = group["training_steps"].to_numpy(dtype=float) / 1000.0
    y = group["mean"].to_numpy(dtype=float)
    std = group["std"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, marker=marker, markevery=max(1, len(group) // 8), linewidth=2.0, markersize=4.5, label=label)
    ax.fill_between(x, np.clip(y - std, 0.0, 1.0), np.clip(y + std, 0.0, 1.0), color=color, alpha=0.16, linewidth=0)


def draw_zero_shot_curve(ax: plt.Axes, group: pd.DataFrame, *, encoding: str, value_column: str) -> None:
    group = group.sort_values("eval_n")
    ax.plot(
        group["eval_n"],
        group[value_column],
        color=COLORS[encoding],
        marker=MARKERS[encoding],
        linewidth=2.0,
        markersize=5.5,
        label=ENCODING_LABELS[encoding],
    )
    if "std" in group.columns and group["std"].notna().any():
        y = group[value_column].to_numpy(dtype=float)
        std = group["std"].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(group["eval_n"], y - std, y + std, color=COLORS[encoding], alpha=0.12, linewidth=0)


def draw_train_range(ax: plt.Axes) -> None:
    ax.axvspan(1, 5, color="#D8D8D8", alpha=0.35, linewidth=0)
    ax.text(3.0, 0.03, "train n=1..5", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#555555")


def format_learning_axis(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlim(left=0)


def format_zero_shot_axis(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Evaluation sequence length n")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(1, 20)
    if "success" in ylabel.lower():
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))


def save_figure(fig: plt.Figure, config: FigureConfig, name: str) -> None:
    for fmt in config.formats:
        path = config.output_dir / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
