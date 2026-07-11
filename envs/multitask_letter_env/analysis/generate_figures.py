"""Generate report figures for multitask LetterEnv experiments."""

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
ENV_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"
DEFAULT_RESULTS_DIR = ENV_ROOT / "results_and_evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"

ENCODING_ORDER = ["numerical", "one_hot", "learned_gru", "learned_graph"]
TABULAR_ENCODINGS = ["numerical", "one_hot"]
ZERO_SHOT_N_VALUES = [10, 15, 20]
FIGURE_NAMES = (
    "learning_by_encoding",
    "zero_shot_success_with_tabular",
    "sample_efficiency",
    "tabular_neural_summary",
)

ENCODING_LABELS = {
    "numerical": "Numerical",
    "one_hot": "One-hot",
    "learned_gru": "Learned GRU",
    "learned_graph": "Learned GNN",
}
ALGORITHM_LABELS = {
    "ddqn": "DDQN",
    "tabular": "Tabular Q-learning",
}
COLORS = {
    "numerical": "#4C78A8",
    "one_hot": "#F58518",
    "learned_gru": "#B279A2",
    "learned_graph": "#E45756",
    "tabular_numerical": "#6B7280",
    "tabular_one_hot": "#9CA3AF",
}
MARKERS = {
    "numerical": "o",
    "one_hot": "s",
    "learned_gru": "D",
    "learned_graph": "P",
}
ZERO_SHOT_X_OFFSETS = {
    "numerical": -0.27,
    "one_hot": -0.09,
    "learned_gru": 0.09,
    "learned_graph": 0.27,
    "tabular_numerical": -0.12,
    "tabular_one_hot": 0.12,
}


@dataclass(frozen=True)
class FigureConfig:
    results_dir: Path
    output_dir: Path
    formats: tuple[str, ...]
    success_threshold: float
    max_learning_steps: int
    figures: tuple[str, ...]


def main() -> None:
    config = parse_args()
    configure_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ddqn_learning = load_learning_metrics(config.results_dir, algorithm="ddqn")
    tabular_learning = load_learning_metrics(config.results_dir, algorithm="tabular")
    zero_shot = load_zero_shot_metrics(config.results_dir)

    if ddqn_learning.empty:
        raise FileNotFoundError(f"No DDQN eval_metrics.csv files found under {config.results_dir}")
    if "zero_shot_success_with_tabular" in config.figures and zero_shot.empty:
        raise FileNotFoundError(f"No zero-shot eval_metrics.csv files found under {config.results_dir}")

    if "learning_by_encoding" in config.figures:
        plot_learning_by_encoding(ddqn_learning, config)
    if "zero_shot_success_with_tabular" in config.figures:
        plot_zero_shot_success_with_tabular(zero_shot, config)
    if "sample_efficiency" in config.figures:
        plot_sample_efficiency(ddqn_learning, config)
    if "tabular_neural_summary" in config.figures:
        plot_tabular_neural_summary(ddqn_learning, tabular_learning, zero_shot, config)

    print(f"Wrote figures to {config.output_dir}")


def parse_args() -> FigureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"), choices=("pdf", "png", "svg"))
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--max-learning-steps", type=int, default=250000)
    parser.add_argument("--figures", nargs="+", default=FIGURE_NAMES, choices=FIGURE_NAMES)
    args = parser.parse_args()
    return FigureConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        success_threshold=float(args.success_threshold),
        max_learning_steps=int(args.max_learning_steps),
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


def load_learning_metrics(results_dir: Path, *, algorithm: str) -> pd.DataFrame:
    base = results_dir / algorithm
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"(?P<encoding>.+)_n_1to5_seed(?P<seed>[0-9]+)$")
    for path in sorted(base.glob("*/eval_metrics.csv")):
        run_name = path.parent.name
        if "diagnostic" in run_name or "rerun" in run_name:
            continue
        match = pattern.match(run_name)
        if match is None:
            continue
        encoding = match.group("encoding")
        if algorithm == "ddqn" and encoding not in ENCODING_ORDER:
            continue
        if algorithm == "tabular" and encoding not in TABULAR_ENCODINGS:
            continue
        frame = pd.read_csv(path)
        frame["algorithm"] = algorithm
        frame["encoding"] = encoding
        frame["seed"] = int(match.group("seed"))
        frame["run_dir"] = str(path.parent)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    for column in [
        "training_steps",
        "training_episodes",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "eval_success_rate",
        "eval_task_failure_rate",
        "seed",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_zero_shot_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((results_dir / "generalization").glob("*/*/eval_metrics.csv")):
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
        "eval_task_failure_rate",
        "eval_timeout_rate",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def plot_learning_by_encoding(learning: pd.DataFrame, config: FigureConfig) -> None:
    subset = learning[
        learning["encoding"].isin(ENCODING_ORDER)
        & (learning["training_steps"] <= config.max_learning_steps)
    ]
    summary = summarize_curve(subset, ["encoding", "training_steps"], "eval_success_rate")
    summary.to_csv(config.output_dir / "learning_by_encoding_success_rate.csv", index=False)

    fig, ax = plt.subplots()
    for encoding in ENCODING_ORDER:
        group = summary[summary["encoding"] == encoding]
        if group.empty:
            continue
        draw_curve(ax, group, encoding=encoding, label=ENCODING_LABELS[encoding])
    format_learning_axis(ax, title="DDQN learning curves by encoding", ylabel="Evaluation success rate")
    ax.legend(loc="lower right")
    save_figure(fig, config, "learning_by_encoding_success_rate")


def plot_zero_shot_success_with_tabular(zero_shot: pd.DataFrame, config: FigureConfig) -> None:
    subset = zero_shot[zero_shot["eval_n"].isin(ZERO_SHOT_N_VALUES)]
    summary = summarize_curve(subset, ["algorithm", "encoding", "eval_n"], "eval_success_rate")
    summary.to_csv(config.output_dir / "zero_shot_success_with_tabular.csv", index=False)

    fig, ax = plt.subplots()
    draw_train_range(ax)
    for encoding in ENCODING_ORDER:
        group = summary[(summary["algorithm"] == "ddqn") & (summary["encoding"] == encoding)]
        if group.empty:
            continue
        draw_zero_shot_curve(
            ax,
            group,
            encoding=encoding,
            label=f"DDQN {ENCODING_LABELS[encoding]}",
            x_offset=ZERO_SHOT_X_OFFSETS[encoding],
        )
    for encoding in TABULAR_ENCODINGS:
        group = summary[(summary["algorithm"] == "tabular") & (summary["encoding"] == encoding)]
        if group.empty:
            continue
        draw_zero_shot_curve(
            ax,
            group,
            encoding=f"tabular_{encoding}",
            label=f"Tabular {ENCODING_LABELS[encoding]}",
            linestyle="--",
            marker="x",
            x_offset=ZERO_SHOT_X_OFFSETS[f"tabular_{encoding}"],
        )
    format_zero_shot_axis(ax, title="Zero-shot generalization with tabular contrast", ylabel="Evaluation success rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    save_figure(fig, config, "zero_shot_success_with_tabular")


def plot_sample_efficiency(learning: pd.DataFrame, config: FigureConfig) -> None:
    efficiency = compute_sample_efficiency(learning, threshold=config.success_threshold)
    efficiency.to_csv(config.output_dir / "sample_efficiency_first_success.csv", index=False)

    labels = [ENCODING_LABELS[row.encoding] for row in efficiency.itertuples()]
    y = np.arange(len(efficiency))
    means = efficiency["mean_first_success_steps"].to_numpy(dtype=float) / 1000.0
    stds = efficiency["std_first_success_steps"].fillna(0.0).to_numpy(dtype=float) / 1000.0
    colors = [COLORS[row.encoding] for row in efficiency.itertuples()]

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.9, capsize=3, edgecolor="white", linewidth=0.7)
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


def plot_tabular_neural_summary(
    ddqn_learning: pd.DataFrame,
    tabular_learning: pd.DataFrame,
    zero_shot: pd.DataFrame,
    config: FigureConfig,
) -> None:
    summary = build_tabular_neural_summary(ddqn_learning, tabular_learning, zero_shot)
    summary.to_csv(config.output_dir / "tabular_neural_summary.csv", index=False)

    display = summary.copy()
    display["method"] = display["algorithm"].map(ALGORITHM_LABELS) + "\n" + display["encoding"].map(ENCODING_LABELS)
    values = display[["in_distribution_success_rate", "zero_shot_success_rate"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(display))
    width = 0.36
    ax.bar(x - width / 2, values[:, 0], width, label="n=1..5 final eval", color="#4C78A8", alpha=0.9)
    ax.bar(x + width / 2, values[:, 1], width, label="n=10/15/20 zero-shot", color="#E45756", alpha=0.9)
    ax.set_title("In-distribution success and zero-shot transfer")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(display["method"], rotation=25, ha="right")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    for index, (in_dist, zero) in enumerate(values):
        ax.text(index - width / 2, in_dist + 0.025, f"{in_dist:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, zero + 0.025, f"{zero:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_figure(fig, config, "tabular_neural_summary")


def summarize_curve(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(mean=(value_column, "mean"), std=(value_column, "std"), count=(value_column, "count"))
        .sort_values(group_columns)
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def compute_sample_efficiency(learning: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for encoding in ENCODING_ORDER:
        subset = learning[learning["encoding"] == encoding]
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
                "algorithm": "ddqn",
                "encoding": encoding,
                "mean_first_success_steps": float(np.mean(first_steps)) if first_steps else np.nan,
                "std_first_success_steps": float(np.std(first_steps, ddof=1)) if len(first_steps) > 1 else 0.0,
                "solved_seeds": len(first_steps),
                "unsolved_seeds": unsolved,
                "total_seeds": int(subset["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_tabular_neural_summary(
    ddqn_learning: pd.DataFrame,
    tabular_learning: pd.DataFrame,
    zero_shot: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    combinations = [("ddqn", encoding) for encoding in ENCODING_ORDER]
    combinations.extend(("tabular", encoding) for encoding in TABULAR_ENCODINGS)
    learning_by_algorithm = {"ddqn": ddqn_learning, "tabular": tabular_learning}
    for algorithm, encoding in combinations:
        learning = learning_by_algorithm[algorithm]
        subset = learning[learning["encoding"] == encoding]
        if subset.empty:
            continue
        final_by_seed = subset.sort_values("training_steps").groupby("seed", as_index=False).tail(1)
        zero_subset = zero_shot[(zero_shot["algorithm"] == algorithm) & (zero_shot["encoding"] == encoding)]
        rows.append(
            {
                "algorithm": algorithm,
                "encoding": encoding,
                "in_distribution_success_rate": float(final_by_seed["eval_success_rate"].mean()),
                "in_distribution_success_std": float(final_by_seed["eval_success_rate"].std(ddof=1) or 0.0),
                "in_distribution_seeds": int(final_by_seed["seed"].nunique()),
                "zero_shot_success_rate": float(zero_subset["eval_success_rate"].mean()) if not zero_subset.empty else np.nan,
                "zero_shot_success_std": float(zero_subset["eval_success_rate"].std(ddof=1) or 0.0) if not zero_subset.empty else np.nan,
                "zero_shot_points": int(len(zero_subset)),
            }
        )
    return pd.DataFrame(rows)


def draw_curve(ax: plt.Axes, group: pd.DataFrame, *, encoding: str, label: str) -> None:
    group = group.sort_values("training_steps")
    x = group["training_steps"].to_numpy(dtype=float) / 1000.0
    y = group["mean"].to_numpy(dtype=float)
    std = group["std"].to_numpy(dtype=float)
    ax.plot(
        x,
        y,
        color=COLORS[encoding],
        marker=MARKERS[encoding],
        markevery=max(1, len(group) // 8),
        linewidth=2.0,
        markersize=4.5,
        label=label,
    )
    ax.fill_between(x, np.clip(y - std, 0.0, 1.0), np.clip(y + std, 0.0, 1.0), color=COLORS[encoding], alpha=0.16, linewidth=0)


def draw_zero_shot_curve(
    ax: plt.Axes,
    group: pd.DataFrame,
    *,
    encoding: str,
    label: str,
    linestyle: str = "-",
    marker: str | None = None,
    x_offset: float = 0.0,
) -> None:
    group = group.sort_values("eval_n")
    color = COLORS[encoding]
    marker = marker or MARKERS.get(encoding, "o")
    x = group["eval_n"].to_numpy(dtype=float) + float(x_offset)
    ax.plot(
        x,
        group["mean"],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.0,
        markersize=5.5,
        label=label,
    )
    if "std" in group.columns and group["std"].notna().any():
        y = group["mean"].to_numpy(dtype=float)
        std = group["std"].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(x, np.clip(y - std, 0.0, 1.0), np.clip(y + std, 0.0, 1.0), color=color, alpha=0.10, linewidth=0)


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
    ax.set_xlabel("Evaluation count n")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(1, 20)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))


def save_figure(fig: plt.Figure, config: FigureConfig, name: str) -> None:
    for fmt in config.formats:
        fig.savefig(config.output_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
