#!/usr/bin/env python3
"""Plot the oracle-teacher training mechanism for ts_transformer control output.

No plotting code exists anywhere in ``4dTrajectory/ts_transformer`` (the package, its
``oracle_teacher/`` sub-package, or any root-level ``run_ts_oracle_teacher_*.py`` /
``run_ts_simple_teacher_paired_cv.py`` driver script) -- verified 2026-08-17 by grepping
for matplotlib/plotly/savefig across all of them. Every stage DOES record numeric loss
history to JSON; nothing turns it into a chart. This script fills that gap by reading the
artifacts documented in
``4dTrajectory/ts_transformer/docs/control_parameter_prediction.zh.md`` Sec 4, and produces
four figures:

1. offline_optimization.{png,svg}   -- the direct-shooting teacher-schedule fit
   (``teacher_optimization.json``: per-step loss by stage, plus per-source median ADE
   initial vs optimized).
2. imitation_warmstart.{png,svg}    -- the online behaviour-cloning warm-start
   (``model_pretraining.history`` inside a ``result.json`` or ``checkpoint_metadata.json``:
   total loss and its three components, log scale).
3. paired_downstream_training.{png,svg} -- teacher vs no-teacher per-epoch train/val loss
   for each fold of the current ``run_ts_simple_teacher_paired_cv.py`` ablation.
4. paired_summary.{png,svg}         -- the headline teacher-vs-no-teacher metric comparison
   from ``paired_cv_results.json`` (the same numbers as ``paired_cv_report.zh.md``).

Usage::

    python plot_teacher_training.py
    python plot_teacher_training.py --output-dir <dir>

All input paths default to the artifacts synced from the compute box
(supercomputing@192.168.1.103) on 2026-08-17; override individual ``--*`` paths to point at
a different run. Every input file is read as-is; nothing here re-derives or approximates the
underlying numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_PAIRED_CV_DIR = (
    REPO_ROOT
    / "4dTrajectory/outputs/KSJC/experiments/control_simple_v1_20260816"
    / "paired_teacher_cv_outer_train_seed1337"
)
DEFAULT_POOLED_SCHEDULE = (
    REPO_ROOT
    / "4dTrajectory/outputs/POOLED/oracle_teacher_simple_v1_20260816"
    / "optimized_balanced_32/teacher_optimization.json"
)
DEFAULT_POOLED_RUN_METADATA = (
    REPO_ROOT
    / "4dTrajectory/outputs/POOLED/ts_itr_control_simple_v1_teacher_all_airports_seed1337"
    / "checkpoint_metadata.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "4dTrajectory/outputs/POOLED/oracle_teacher_training_plots"
)

FOLD_COLORS = {"fold_0": "#1f77b4", "fold_1": "#ff7f0e", "fold_2": "#2ca02c"}
ARM_STYLE = {"teacher": dict(linestyle="-"), "no_teacher": dict(linestyle="--")}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"expected artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pyplot(output_dir: Path):
    cache = output_dir / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )
    return plt


def _save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def _stage_boundaries(recipe: dict[str, Any]) -> list[tuple[str, int]]:
    """Cumulative global-step index at the END of each direct-shooting stage."""
    boundary = 0
    marks = []
    for stage in recipe["stages"]:
        boundary += int(stage["steps"])
        marks.append((stage["label"], boundary))
    return marks


# --------------------------------------------------------------------------- 1. offline


def plot_offline_optimization(
    sources: list[tuple[str, Path]], output_dir: Path, plt
) -> list[Path]:
    """Direct-shooting teacher-schedule fit: loss-by-stage and ADE improvement."""

    fig, (ax_loss, ax_ade) = plt.subplots(1, 2, figsize=(13, 4.5))

    bar_labels: list[str] = []
    initial_medians: list[float] = []
    optimized_medians: list[float] = []

    for idx, (label, path) in enumerate(sources):
        doc = _load_json(path)
        history = doc["history"]
        steps = [row["step"] for row in history]
        losses = [row["loss"] for row in history]
        color = plt.cm.tab10(idx % 10)
        ax_loss.plot(steps, losses, color=color, label=label, linewidth=1.6)

        if idx == 0:
            for stage_label, boundary in _stage_boundaries(doc["recipe"]):
                ax_loss.axvline(boundary, color="0.7", linestyle=":", linewidth=0.9)
                ax_loss.annotate(
                    stage_label,
                    xy=(boundary, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(-2, -2),
                    textcoords="offset points",
                    rotation=90,
                    fontsize=7,
                    color="0.4",
                    ha="right",
                    va="top",
                )

        bar_labels.append(label)
        initial_medians.append(doc["median_metrics"]["initial"]["ade_m"])
        optimized_medians.append(doc["median_metrics"]["optimized"]["ade_m"])

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("global step (direct-shooting Adam, staged 60s -> 120s -> 240s -> full)")
    ax_loss.set_ylabel("total loss (log scale)")
    ax_loss.set_title("Offline teacher-schedule optimization\n(oracle_teacher/optimization.py)")
    ax_loss.legend(fontsize=8, loc="upper right")

    x = range(len(bar_labels))
    width = 0.35
    ax_ade.bar(
        [i - width / 2 for i in x], initial_medians, width, label="initial (inverse-dynamics)", color="#c44e52"
    )
    ax_ade.bar(
        [i + width / 2 for i in x], optimized_medians, width, label="optimized", color="#55a868"
    )
    for i, (init, opt) in enumerate(zip(initial_medians, optimized_medians)):
        pct = 100.0 * (init - opt) / init
        ax_ade.annotate(
            f"-{pct:.0f}%",
            xy=(i, max(init, opt)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_ade.set_xticks(list(x))
    ax_ade.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=8)
    ax_ade.set_ylabel("median ADE across the cohort (m)")
    ax_ade.set_title("Teacher-cohort fit quality:\ninverse-dynamics initial guess vs. optimized")
    ax_ade.legend(fontsize=8)

    fig.suptitle(
        "Teacher training mechanism -- stage 1: offline schedule generation "
        "(run_ts_oracle_teacher_optimize.py)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = _save_figure(fig, output_dir, "offline_optimization")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- 2. imitation warm-start


def plot_imitation_warmstart(
    sources: list[tuple[str, dict[str, Any]]], output_dir: Path, plt
) -> list[Path]:
    """Online imitation (behaviour cloning) of the cached teacher schedule."""

    components = ["loss", "control", "duration_fraction", "final_time"]
    titles = {
        "loss": "total loss",
        "control": "control component (unit-box MSE)",
        "duration_fraction": "duration-fraction component\n(uniform duration -> no head -> always 0)",
        "final_time": "final-time component (/600 s scale)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, component in zip(axes.flat, components):
        for idx, (label, mp) in enumerate(sources):
            history = mp["history"]
            steps = [row["step"] for row in history]
            values = [row[component] for row in history]
            color = plt.cm.tab10(idx % 10)
            ax.plot(steps, values, marker="o", markersize=3, color=color, label=label)
        ax.set_title(titles[component], fontsize=9)
        ax.set_xlabel("imitation step (of 1000)")
        if component != "duration_fraction":
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Teacher training mechanism -- stage 2: online imitation warm-start\n"
        "(oracle_teacher/pretraining.py::CachedSchedulePretrainer, before the ordinary "
        "rollout-loss training loop begins)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    paths = _save_figure(fig, output_dir, "imitation_warmstart")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- 3. paired downstream


def plot_paired_downstream_training(
    fold_dirs: list[tuple[str, Path]], output_dir: Path, plt
) -> list[Path]:
    fig, axes = plt.subplots(1, len(fold_dirs), figsize=(5.2 * len(fold_dirs), 4.5), sharey=True)
    if len(fold_dirs) == 1:
        axes = [axes]

    for ax, (fold_label, fold_dir) in zip(axes, fold_dirs):
        for arm in ("teacher", "no_teacher"):
            result = _load_json(fold_dir / arm / "result.json")
            history = result["history"]
            epochs = [row["epoch"] for row in history]
            train_loss = [row["train_loss"] for row in history]
            val_loss = [row["val_loss"] for row in history]
            style = ARM_STYLE[arm]
            color = "#1f77b4" if arm == "teacher" else "#d62728"
            ax.plot(epochs, train_loss, color=color, alpha=0.55, linewidth=1.1, **style)
            ax.plot(epochs, val_loss, color=color, linewidth=1.8, label=f"{arm} val", **style)
            best_epoch = result["best_epoch"]
            best_ade = result["validation_metrics"]["ade_m"]
            best_row = next(r for r in history if r["epoch"] == best_epoch)
            ax.scatter(
                [best_epoch],
                [best_row["val_loss"]],
                color=color,
                zorder=5,
                s=45,
                edgecolor="white",
            )
            ax.annotate(
                f"{arm}\nbest ep {best_epoch}\nADE {best_ade:.0f} m",
                xy=(best_epoch, best_row["val_loss"]),
                xytext=(6, 6 if arm == "teacher" else -28),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
        ax.set_title(fold_label, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("loss (log scale); solid=val, faint=train")
    axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Teacher training mechanism -- downstream simple-v1 training, "
        "teacher vs. no-teacher (run_ts_simple_teacher_paired_cv.py)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = _save_figure(fig, output_dir, "paired_downstream_training")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------- 4. paired summary


def plot_paired_summary(results: dict[str, Any], output_dir: Path, plt) -> list[Path]:
    summary = results["paired_summary"]
    metrics = [
        ("ade_m", "ADE (m)"),
        ("fde_m", "FDE (m)"),
        ("arrival_endpoint_error_mean_m", "Arrival endpoint\nerror mean (m)"),
        ("cross_track_p95_abs_m", "Cross-track p95 (m)"),
        ("vertical_p95_abs_m", "Vertical p95 (m)"),
        ("final_time_mae_s", "Final-time MAE (s)"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(metrics))
    width = 0.32
    teacher_means = [summary[key]["teacher_mean"] for key, _ in metrics]
    no_teacher_means = [summary[key]["no_teacher_mean"] for key, _ in metrics]

    # Normalize each metric to its own no-teacher mean so six different units share one axis.
    teacher_norm = [t / n for t, n in zip(teacher_means, no_teacher_means)]
    no_teacher_norm = [1.0 for _ in metrics]

    ax.bar([i - width / 2 for i in x], teacher_norm, width, label="teacher", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], no_teacher_norm, width, label="no-teacher (=1.0 baseline)", color="#d62728")

    for i, (key, _) in enumerate(metrics):
        m = summary[key]
        for fold_idx, value in enumerate(m["teacher_by_fold"]):
            ax.scatter(
                i - width / 2, value / m["no_teacher_mean"], color="black", s=14, zorder=5
            )
        for fold_idx, value in enumerate(m["no_teacher_by_fold"]):
            ax.scatter(
                i + width / 2, value / m["no_teacher_mean"], color="black", s=14, zorder=5
            )
        pct = m["paired_percent_improvement_mean"]
        wins = m["teacher_wins"]
        ax.annotate(
            f"-{pct:.1f}%\n({wins}/{m['folds']} folds)",
            xy=(i, max(teacher_norm[i], 1.0)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel("value / no-teacher mean (lower is better; dots = individual folds)")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.18), ncol=2, frameon=False)
    ax.set_title(
        "simple-v1: teacher warm-start vs. no-teacher, 3-fold paired outer-train CV\n"
        f"(run contract {results['run_contract_sha256'][:12]}...)",
        fontsize=11,
    )
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "paired_summary")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------------- index


def _write_index(output_dir: Path, figures: dict[str, list[Path]]) -> Path:
    lines = [
        "# Oracle-teacher training mechanism -- plots",
        "",
        "Generated by `plot_teacher_training.py` from artifacts synced from",
        "`supercomputing@192.168.1.103` on 2026-08-17. Source data is the current",
        "(2026-08-16) `control_recipe_name=simple-v1` teacher-training generation --",
        "see `4dTrajectory/ts_transformer/docs/control_parameter_prediction.zh.md` Sec 4.",
        "",
    ]
    captions = {
        "offline_optimization": (
            "Stage 1 -- offline direct-shooting fit of the cached teacher control "
            "schedule (`oracle_teacher/optimization.py`), for the 3 paired-CV fold "
            "cohorts plus the pooled 32-flight balanced cohort that feeds the final "
            "all-airports run."
        ),
        "imitation_warmstart": (
            "Stage 2 -- online behaviour-cloning warm-start of the model "
            "(`oracle_teacher/pretraining.py::CachedSchedulePretrainer`), for the same "
            "3 folds plus the KSJC development run and the final pooled all-airports run."
        ),
        "paired_downstream_training": (
            "Downstream: per-epoch train/val loss of the ordinary simple-v1 training "
            "loop that follows warm-start, teacher vs. no-teacher, one panel per fold."
        ),
        "paired_summary": (
            "Headline result: the 3-fold paired teacher-vs-no-teacher comparison across "
            "6 validation metrics, normalized to the no-teacher mean per metric. Teacher "
            "wins on ADE in 3/3 folds, mean -15.0%."
        ),
    }
    for stem, paths in figures.items():
        png = paths[0].name
        lines.append(f"## {stem}")
        lines.append("")
        lines.append(captions.get(stem, ""))
        lines.append("")
        lines.append(f"![{stem}]({png})")
        lines.append("")
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-cv-dir", type=Path, default=DEFAULT_PAIRED_CV_DIR)
    parser.add_argument(
        "--pooled-schedule", type=Path, default=DEFAULT_POOLED_SCHEDULE
    )
    parser.add_argument(
        "--pooled-run-metadata", type=Path, default=DEFAULT_POOLED_RUN_METADATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    plt = _pyplot(args.output_dir)

    fold_labels = ["fold_0", "fold_1", "fold_2"]
    fold_dirs = [(label, args.paired_cv_dir / label) for label in fold_labels]
    for label, path in fold_dirs:
        if not path.exists():
            parser.error(f"missing {label} directory under --paired-cv-dir: {path}")

    figures: dict[str, list[Path]] = {}

    offline_sources = [
        (label, fold_dir / "teacher_schedule" / "teacher_optimization.json")
        for label, fold_dir in fold_dirs
    ]
    offline_sources.append(("pooled_balanced_32", args.pooled_schedule))
    figures["offline_optimization"] = plot_offline_optimization(
        offline_sources, args.output_dir, plt
    )

    imitation_sources = []
    for label, fold_dir in fold_dirs:
        result = _load_json(fold_dir / "teacher" / "result.json")
        imitation_sources.append((label, result["model_pretraining"]))
    dev_teacher = _load_json(
        args.paired_cv_dir.parent / "development_teacher_current_manifest_seed1337"
        / "checkpoint_metadata.json"
    )
    imitation_sources.append(("ksjc_development", dev_teacher["model_pretraining"]))
    pooled_run = _load_json(args.pooled_run_metadata)
    imitation_sources.append(("pooled_all_airports_final", pooled_run["model_pretraining"]))
    figures["imitation_warmstart"] = plot_imitation_warmstart(
        imitation_sources, args.output_dir, plt
    )

    figures["paired_downstream_training"] = plot_paired_downstream_training(
        fold_dirs, args.output_dir, plt
    )

    results = _load_json(args.paired_cv_dir / "paired_cv_results.json")
    figures["paired_summary"] = plot_paired_summary(results, args.output_dir, plt)

    index_path = _write_index(args.output_dir, figures)
    print(f"wrote {sum(len(v) for v in figures.values())} figures + {index_path}")
    for stem, paths in figures.items():
        for p in paths:
            print(f"  {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
