"""Generate CSTR PPO result figures from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.cstr.analysis.visualize_cstr_policy import replay_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "envs" / "cstr" / "results_and_evaluation"
PPO_ROOT = RESULTS_ROOT / "ppo"
GENERALIZATION_ROOT = RESULTS_ROOT / "generalization"
FIGURES_ROOT = RESULTS_ROOT / "figures"
GRAPH_ENCODER_CHECKPOINT = (
    RESULTS_ROOT
    / "encoder_pretraining"
    / "gnn_dynamics_phase_count_reference_seed0"
    / "best_dynamics_encoder.pt"
)

VARIANTS = (
    {
        "key": "baseline",
        "label": "Baseline",
        "folder": "baseline_seed{seed}",
        "env_variant": "baseline",
        "reward_mode": "env",
        "monitor_state_limit": 16,
    },
    {
        "key": "rml_hidden",
        "label": "RML hidden",
        "folder": "rml_hidden_seed{seed}",
        "env_variant": "rml_hidden",
        "reward_mode": "env_rml",
        "monitor_state_limit": 16,
    },
    {
        "key": "rml_semantic_progress",
        "label": "RML semantic progress",
        "folder": "rml_semantic_progress_seed{seed}",
        "env_variant": "semantic_progress",
        "reward_mode": "env_rml",
        "monitor_state_limit": 16,
    },
    {
        "key": "manual_rm_semantic_progress",
        "label": "Manual RM semantic progress",
        "folder": "manual_rm_semantic_progress_seed{seed}",
        "env_variant": "manual_rm_semantic_progress",
        "reward_mode": "env_rml",
        "monitor_state_limit": 16,
    },
    {
        "key": "rml_graph_encoder",
        "label": "RML graph encoder",
        "folder": "rml_graph_encoder_seed{seed}",
        "env_variant": "rml_graph",
        "reward_mode": "env_rml",
        "monitor_state_limit": 16,
        "graph_encoder_checkpoint": GRAPH_ENCODER_CHECKPOINT,
    },
)

BAR_METRICS = (
    ("rml_success_rate", "RML success rate"),
    ("mean_tracking_error", "Mean tracking error"),
    ("mean_return", "Mean return"),
)

COLORS = {
    "baseline": "#4b5563",
    "rml_hidden": "#2563eb",
    "rml_semantic_progress": "#059669",
    "manual_rm_semantic_progress": "#d97706",
    "rml_graph_encoder": "#7c3aed",
}

PHASE_COLORS = {
    "preheat": "#ef4444",
    "soak": "#f59e0b",
    "approach": "#2563eb",
    "regulate": "#059669",
    "success": "#16a34a",
    "failure": "#991b1b",
    "none": "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-root", type=Path, default=PPO_ROOT)
    parser.add_argument("--generalization-root", type=Path, default=GENERALIZATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--rml-hidden-trajectory-seed", type=int, default=2)
    parser.add_argument("--manual-rm-trajectory-seed", type=int, default=1)
    parser.add_argument("--trajectory-seed", type=int, default=10_000)
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--graph-encoder-checkpoint", type=Path, default=GRAPH_ENCODER_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_eval_metrics(args.ppo_root, args.seeds)
    final_rows = load_final_metrics(args.ppo_root, args.seeds)
    if eval_rows.empty or final_rows.empty:
        raise FileNotFoundError(f"No CSTR PPO metrics found under {args.ppo_root}")

    write_table(csv_dir / "ppo_eval_metrics_all_seeds.csv", eval_rows)
    write_table(csv_dir / "ppo_final_metrics_all_seeds.csv", final_rows)

    final_summary = aggregate_final(final_rows)
    write_table(csv_dir / "ppo_final_metric_summary_multiseed.csv", final_summary)

    plot_best_metrics(
        final_summary,
        args.output_dir / "all_variants_best_metrics_multiseed",
        title="CSTR PPO final metrics, seeds 0-4",
        show_std=True,
    )

    generalization_rows = load_generalization_metrics(args.generalization_root, final_rows, args.seed0)
    if not generalization_rows.empty:
        write_table(csv_dir / "generalization_success_by_soak_steps.csv", generalization_rows)
        plot_generalization_success(
            generalization_rows,
            args.output_dir / "generalization_success_by_soak_steps_seed0",
        )

    trajectory_rows = []
    if not args.skip_trajectories:
        trajectory_rows = generate_all_seed_trajectories(args, final_rows)
        write_dicts(csv_dir / "trajectory_summary_all_seeds.csv", trajectory_rows)
        trajectory_suffix = trajectory_suffix_from_args(args)
        plot_trajectory_comparison(
            args,
            args.output_dir / "cstr_trajectories",
            args.output_dir / f"all_variants_trajectory_comparison_{trajectory_suffix}",
            title=(
                f"CSTR trajectory comparison, seed {args.seed0} "
                f"with RML hidden seed {args.rml_hidden_trajectory_seed} "
                f"and manual RM seed {args.manual_rm_trajectory_seed}"
            ),
        )
        plot_headline_trajectory(
            args.output_dir / "cstr_trajectories",
            args.output_dir / f"baseline_vs_rml_graph_trajectory_seed{args.seed0}",
            baseline_seed=args.seed0,
            rml_seed=args.seed0,
        )
        plot_phase_trajectory(
            args.output_dir / "cstr_trajectories",
            args.output_dir / f"rml_graph_phase_trajectory_seed{args.seed0}",
            seed=args.seed0,
        )
        relocate_trajectory_csvs(args.output_dir / "cstr_trajectories", csv_dir / "cstr_trajectories")

    figure_names = [
        "all_variants_best_metrics_multiseed.png",
        "generalization_success_by_soak_steps_seed0.png",
    ]
    if not args.skip_trajectories:
        figure_names.extend(
            [
                f"all_variants_trajectory_comparison_{trajectory_suffix_from_args(args)}.png",
                f"baseline_vs_rml_graph_trajectory_seed{args.seed0}.png",
                f"rml_graph_phase_trajectory_seed{args.seed0}.png",
            ]
        )

    summary = {
        "figures": figure_names,
        "trajectory_runs": trajectory_rows,
        "csv_dir": str(csv_dir),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_eval_metrics(ppo_root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            run_dir = ppo_root / variant["folder"].format(seed=seed)
            metrics_path = run_dir / "eval_metrics.csv"
            if not metrics_path.exists():
                continue
            frame = pd.read_csv(metrics_path)
            frame["variant"] = variant["key"]
            frame["label"] = variant["label"]
            frame["seed"] = seed
            frame["run_dir"] = str(run_dir)
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_final_metrics(ppo_root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            run_dir = ppo_root / variant["folder"].format(seed=seed)
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            metrics = dict(summary.get("final_eval") or summary.get("final_eval_summary") or {})
            if not metrics:
                continue
            metrics.update(
                {
                    "variant": variant["key"],
                    "label": variant["label"],
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def variant_by_key(key: str) -> dict[str, Any] | None:
    for variant in VARIANTS:
        if variant["key"] == key:
            return variant
    return None


def aggregate_final(rows: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [metric for metric, _label in BAR_METRICS]
    grouped = rows.groupby(["variant", "label"])[metric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    return grouped


def plot_best_metrics(rows: pd.DataFrame, output_prefix: Path, *, title: str, show_std: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    positions = np.arange(len(VARIANTS))
    labels = [variant["label"] for variant in VARIANTS]
    for axis, (metric, metric_label) in zip(np.ravel(axes), BAR_METRICS, strict=True):
        means = []
        stds = []
        colors = []
        for variant in VARIANTS:
            subset = rows.loc[rows["variant"] == variant["key"]]
            means.append(float(subset[f"{metric}_mean"].iloc[0]) if not subset.empty else np.nan)
            stds.append(float(subset[f"{metric}_std"].iloc[0]) if show_std and not subset.empty else 0.0)
            colors.append(COLORS[variant["key"]])
        axis.bar(positions, means, yerr=stds if show_std else None, color=colors, alpha=0.88, capsize=4)
        axis.set_title(metric_label)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        if metric.endswith("_rate"):
            axis.set_ylim(0.0, 1.05)
    fig.suptitle(title)
    save_figure(fig, output_prefix)


def load_generalization_metrics(generalization_root: Path, final_rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    records = []
    for variant in VARIANTS:
        if variant["key"] == "baseline":
            continue
        subset = final_rows.loc[(final_rows["variant"] == variant["key"]) & (final_rows["seed"] == seed)]
        if not subset.empty:
            row = subset.iloc[0]
            records.append(
                {
                    "variant": variant["key"],
                    "label": variant["label"],
                    "train_seed": seed,
                    "soak_steps": 10,
                    "split": "train",
                    "status": "evaluated",
                    "rml_success_rate": float(row.get("rml_success_rate", np.nan)),
                }
            )
    if generalization_root.exists():
        for summary_path in sorted(generalization_root.glob("soak*_seed*/summary.json")):
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            soak_steps = int(summary.get("eval_soak_steps", 0))
            for result in summary.get("results", []):
                variant_key = str(result.get("variant", ""))
                variant = variant_by_key(variant_key)
                if variant is None:
                    continue
                result_seed = int(result.get("train_seed", seed))
                if result_seed != seed:
                    continue
                result_summary = result.get("summary") or {}
                records.append(
                    {
                        "variant": variant_key,
                        "label": variant["label"],
                        "train_seed": result_seed,
                        "soak_steps": soak_steps,
                        "split": "heldout",
                        "status": str(result.get("status", "")),
                        "rml_success_rate": float(result_summary.get("rml_success_rate", np.nan)),
                    }
                )
    return pd.DataFrame(records)


def plot_generalization_success(rows: pd.DataFrame, output_prefix: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.axvspan(9.5, 10.5, color="#111827", alpha=0.08, label="Trained k")
    for variant in VARIANTS:
        subset = rows.loc[rows["variant"] == variant["key"]].sort_values("soak_steps")
        if subset.empty:
            continue
        evaluated = subset.loc[subset["status"] == "evaluated"]
        if not evaluated.empty:
            axis.plot(
                evaluated["soak_steps"].to_numpy(dtype=float),
                evaluated["rml_success_rate"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                color=COLORS[variant["key"]],
                label=variant["label"],
            )
        failed = subset.loc[subset["status"] != "evaluated"]
        if not failed.empty:
            axis.scatter(
                failed["soak_steps"].to_numpy(dtype=float),
                np.zeros(len(failed)),
                marker="x",
                s=70,
                color=COLORS[variant["key"]],
            )
    axis.set_xlabel("Required soak duration k")
    axis.set_ylabel("RML success rate")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks(sorted(rows["soak_steps"].dropna().unique()))
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axis.set_title("CSTR zero-shot counting generalization")
    save_figure(fig, output_prefix)


def generate_all_seed_trajectories(args: argparse.Namespace, final_rows: pd.DataFrame) -> list[dict[str, Any]]:
    trajectory_root = args.output_dir / "cstr_trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        variant_rows = final_rows.loc[final_rows["variant"] == variant["key"]]
        failed_seen = False
        for seed in args.seeds:
            successful = is_successful_final_run(variant["key"], variant_rows, seed)
            split = "successful_seeds" if successful else "failed_seeds"
            failed_seen = failed_seen or not successful
            seed_dir = trajectory_root / variant["key"] / split / f"seed{seed}"
            payload = replay_trajectory(args, variant, seed, seed_dir)
            rows.append(
                {
                    "variant": variant["key"],
                    "label": variant["label"],
                    "train_seed": seed,
                    "success_group": split,
                    **payload["summary"],
                }
            )
        failed_dir = trajectory_root / variant["key"] / "failed_seeds"
        if not failed_seen and failed_dir.exists():
            failed_dir.rmdir()
    return rows


def trajectory_suffix_from_args(args: argparse.Namespace) -> str:
    return (
        f"seed{args.seed0}"
        f"_rml_hidden_seed{args.rml_hidden_trajectory_seed}"
        f"_manual_rm_seed{args.manual_rm_trajectory_seed}"
    )


def trajectory_train_seed(args: argparse.Namespace, variant: str) -> int:
    if variant == "rml_hidden":
        return int(args.rml_hidden_trajectory_seed)
    if variant == "manual_rm_semantic_progress":
        return int(args.manual_rm_trajectory_seed)
    return int(args.seed0)


def is_successful_final_run(variant: str, rows: pd.DataFrame, seed: int) -> bool:
    subset = rows.loc[rows["seed"] == seed]
    if subset.empty:
        return False
    row = subset.iloc[0]
    metric = "success_rate" if variant == "baseline" else "rml_success_rate"
    return float(row.get(metric, 0.0)) >= 1.0


def replay_trajectory(args: argparse.Namespace, variant: dict[str, Any], seed: int, output_dir: Path) -> dict[str, Any]:
    run_dir = args.ppo_root / variant["folder"].format(seed=seed)
    model_path = run_dir / "best_model.zip"
    graph_checkpoint = args.graph_encoder_checkpoint if variant["key"] == "rml_graph_encoder" else None
    return replay_policy(
        SimpleNamespace(
            model_path=model_path,
            env_variant=variant["env_variant"],
            reward_mode=variant["reward_mode"],
            seed=args.trajectory_seed,
            max_episode_steps=300,
            regulation_violation_steps=10,
            soak_steps=10,
            monitor_state_limit=variant["monitor_state_limit"],
            graph_encoder_checkpoint=graph_checkpoint,
            safe_step_bonus=0.10,
            stable_step_bonus=3.0,
            regulation_entry_bonus=5.0,
            success_bonus=50.0,
            failure_penalty=-50.0,
            rml_heating_rate_penalty=0.0,
            preheat_distance_weight=0.08,
            preheat_warming_weight=0.25,
            soak_entry_bonus=5.0,
            soak_progress_bonus=0.75,
            soak_reset_penalty=-3.0,
            soak_lost_step_penalty=0.50,
            approach_distance_weight=1.0,
            approach_progress_bonus=5.0,
            approach_ca_progress_bonus=4.0,
            approach_temp_progress_bonus=4.0,
            approach_warming_weight=0.50,
            production_entry_bonus=10.0,
            regulate_recovery_penalty=-10.0,
            deadline_steps=100,
            tracking_weight=0.5,
            heating_rate_penalty=0.0,
            critical_penalty=200.0,
            production_temp_low=346.0,
            production_temp_high=354.0,
            concentration_tolerance=0.08,
            ca_overshoot_low=0.44,
            require_soak_concentration_band=True,
            soak_concentration_low=0.58,
            soak_concentration_high=0.74,
            fixed_initial_state=True,
            randomize_initial_state=False,
            randomize_setpoint=False,
            enable_disturbance=False,
            temp_weight=0.015,
            action_weight=0.0002,
            warning_penalty=0.25,
            output_dir=output_dir,
        )
    )


def plot_trajectory_comparison(
    args: argparse.Namespace,
    trajectory_root: Path,
    output_prefix: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for variant in VARIANTS:
        train_seed = trajectory_train_seed(args, variant["key"])
        path = trajectory_root / variant["key"] / "successful_seeds" / f"seed{train_seed}" / "trajectory.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        steps = frame["step"].to_numpy(dtype=float)
        axes[0].plot(
            steps,
            frame["reactor_concentration"].to_numpy(dtype=float),
            color=COLORS[variant["key"]],
            label=variant["label"],
            linewidth=2.0,
        )
        axes[1].plot(
            steps,
            frame["reactor_temperature"].to_numpy(dtype=float),
            color=COLORS[variant["key"]],
            label=variant["label"],
            linewidth=2.0,
        )
    axes[0].axhline(0.5, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].fill_between([0, 300], [0.42, 0.42], [0.58, 0.58], color="#6b7280", alpha=0.10)
    axes[0].set_ylabel("Concentration")
    axes[1].axhspan(346.0, 354.0, color="#059669", alpha=0.10)
    axes[1].axhline(350.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("Temperature")
    axes[1].set_xlabel("Step")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(title)
    save_figure(fig, output_prefix)


def plot_headline_trajectory(
    trajectory_root: Path,
    output_prefix: Path,
    *,
    baseline_seed: int,
    rml_seed: int,
) -> None:
    baseline_path = trajectory_root / "baseline" / "successful_seeds" / f"seed{baseline_seed}" / "trajectory.csv"
    rml_path = trajectory_root / "rml_graph_encoder" / "successful_seeds" / f"seed{rml_seed}" / "trajectory.csv"
    if not baseline_path.exists() or not rml_path.exists():
        return

    baseline = pd.read_csv(baseline_path)
    rml = pd.read_csv(rml_path)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    for frame, label, color in (
        (baseline, "Baseline", COLORS["baseline"]),
        (rml, "RML graph encoder", COLORS["rml_graph_encoder"]),
    ):
        steps = frame["step"].to_numpy(dtype=float)
        axes[0].plot(steps, frame["reactor_concentration"].to_numpy(dtype=float), color=color, label=label, linewidth=2.1)
        axes[1].plot(steps, frame["reactor_temperature"].to_numpy(dtype=float), color=color, label=label, linewidth=2.1)

    axes[0].fill_between([0, 300], [0.58, 0.58], [0.74, 0.74], color="#f59e0b", alpha=0.14)
    axes[0].fill_between([0, 300], [0.42, 0.42], [0.58, 0.58], color="#059669", alpha=0.08)
    axes[0].axhline(0.5, color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    axes[0].set_ylabel("Concentration")
    axes[1].axhspan(343.0, 347.0, color="#f59e0b", alpha=0.14)
    axes[1].axhspan(346.0, 354.0, color="#059669", alpha=0.08)
    axes[1].axhline(350.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    axes[1].set_ylabel("Temperature")
    axes[1].set_xlabel("Step")
    annotate_phase_spans(axes[0], rml)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("CSTR procedure contrast: baseline vs RML")
    save_figure(fig, output_prefix)


def plot_phase_trajectory(trajectory_root: Path, output_prefix: Path, *, seed: int) -> None:
    path = trajectory_root / "rml_graph_encoder" / "successful_seeds" / f"seed{seed}" / "trajectory.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    for axis, column, ylabel in (
        (axes[0], "reactor_concentration", "Concentration"),
        (axes[1], "reactor_temperature", "Temperature"),
    ):
        plot_phase_segments(axis, frame, column)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].axhline(0.5, color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    axes[1].axhline(350.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    axes[1].set_xlabel("Step")
    phase_handles = [
        plt.Line2D([0], [0], color=PHASE_COLORS[phase], linewidth=3.0, label=phase.title())
        for phase in ("preheat", "soak", "approach", "regulate", "success")
    ]
    fig.legend(handles=phase_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"CSTR RML graph trajectory by protocol phase, seed {seed}")
    save_figure(fig, output_prefix)


def plot_phase_segments(axis: plt.Axes, frame: pd.DataFrame, value_column: str) -> None:
    for start, stop, phase in contiguous_phase_spans(frame):
        segment = frame.iloc[start : stop + 1]
        axis.plot(
            segment["step"].to_numpy(dtype=float),
            segment[value_column].to_numpy(dtype=float),
            color=PHASE_COLORS.get(phase, PHASE_COLORS["none"]),
            linewidth=2.2,
        )


def annotate_phase_spans(axis: plt.Axes, frame: pd.DataFrame) -> None:
    ymin, ymax = axis.get_ylim()
    y = ymax - 0.08 * (ymax - ymin)
    for start, stop, phase in contiguous_phase_spans(frame):
        if phase in {"none", "success", "failure"}:
            continue
        x0 = float(frame["step"].iloc[start])
        x1 = float(frame["step"].iloc[stop])
        axis.axvspan(x0, x1, color=PHASE_COLORS.get(phase, "#6b7280"), alpha=0.06)
        if x1 - x0 >= 18:
            axis.text((x0 + x1) / 2, y, phase.title(), ha="center", va="top", fontsize=8)
        elif x1 - x0 >= 2:
            axis.text(
                (x0 + x1) / 2,
                y,
                phase.title(),
                ha="center",
                va="top",
                rotation=90,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
            )


def contiguous_phase_spans(frame: pd.DataFrame) -> list[tuple[int, int, str]]:
    if "monitor_phase" not in frame or frame.empty:
        return [(0, len(frame) - 1, "none")] if not frame.empty else []
    phases = frame["monitor_phase"].fillna("none").astype(str).str.lower().tolist()
    spans = []
    start = 0
    current = phases[0]
    for index, phase in enumerate(phases[1:], start=1):
        if phase != current:
            spans.append((start, index - 1, current))
            start = index
            current = phase
    spans.append((start, len(phases) - 1, current))
    return spans


def relocate_trajectory_csvs(trajectory_root: Path, csv_root: Path) -> None:
    csv_root.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(trajectory_root.rglob("trajectory.csv")):
        relative = csv_path.relative_to(trajectory_root)
        method = relative.parts[0]
        group = relative.parts[1]
        seed = relative.parts[2]
        destination = csv_root / f"{method}_{group}_{seed}_trajectory.csv"
        if destination.exists():
            destination.unlink()
        shutil.move(str(csv_path), str(destination))


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
