#!/usr/bin/env python
"""Plot TS cross-validation and final-training results from one run directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _pyplot(output_dir: Path):
    cache = output_dir / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_cv(cv_path: Path, output_dir: Path, plt) -> tuple[list[Path], dict[str, Any]]:
    result = _load_json(cv_path)
    candidates = result.get("candidates")
    tuned_parameters = result.get("tuned_parameters")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{cv_path} has no exhaustive-grid candidates")
    if not isinstance(tuned_parameters, list) or not tuned_parameters:
        raise ValueError(f"{cv_path} has no tuned_parameters")

    generated: list[Path] = []
    override_fields = sorted({
        field
        for candidate in candidates
        for field in candidate.get("overrides", {})
    })
    candidate_rows: list[dict[str, Any]] = []
    airport_scores: dict[int, dict[str, float]] = {}
    for candidate in candidates:
        candidate_id = int(candidate["candidate"])
        folds = candidate.get("folds", [])
        row = {
            "candidate": candidate_id,
            "selected": candidate_id == result.get("best_candidate"),
            "mean_val_macro_loss": candidate["mean_val_macro_loss"],
            "std_val_macro_loss": candidate["std_val_macro_loss"],
            "mean_best_epoch": fmean(float(fold["best_epoch"]) for fold in folds),
        }
        row.update({
            field: candidate.get("overrides", {}).get(field)
            for field in override_fields
        })
        candidate_rows.append(row)

        by_airport: dict[str, list[float]] = defaultdict(list)
        for fold in folds:
            for airport, score in fold.get("val_by_airport", {}).items():
                by_airport[airport].append(float(score))
        airport_scores[candidate_id] = {
            airport: fmean(scores) for airport, scores in sorted(by_airport.items())
        }

    candidate_csv = output_dir / "cv_candidates.csv"
    _write_csv(
        candidate_csv,
        candidate_rows,
        [
            "candidate", "selected", "mean_val_macro_loss",
            "std_val_macro_loss", "mean_best_epoch", *override_fields,
        ],
    )
    generated.append(candidate_csv)

    ordered = sorted(candidate_rows, key=lambda row: row["mean_val_macro_loss"])
    figure, axis = plt.subplots(figsize=(10, 5))
    ranks = list(range(1, len(ordered) + 1))
    means = [float(row["mean_val_macro_loss"]) for row in ordered]
    stds = [float(row["std_val_macro_loss"]) for row in ordered]
    colors = ["#d62728" if row["selected"] else "#4c78a8" for row in ordered]
    axis.errorbar(ranks, means, yerr=stds, fmt="none", ecolor="#9aa0a6", capsize=2)
    axis.scatter(ranks, means, c=colors, s=35)
    axis.set(title="CV candidate ranking", xlabel="Rank", ylabel="Mean validation loss")
    axis.grid(alpha=0.25)
    generated.extend(_save_figure(figure, output_dir, "cv_candidate_scores"))
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(tuned_parameters),
        figsize=(5 * len(tuned_parameters), 4),
        squeeze=False,
    )
    parameter_grid = result.get("parameter_grid", {})
    for axis, parameter in zip(axes[0], tuned_parameters):
        values = parameter_grid.get(parameter, [])
        grouped = [
            [
                float(row["mean_val_macro_loss"])
                for row in candidate_rows
                if row.get(parameter) == value
            ]
            for value in values
        ]
        means = [fmean(scores) for scores in grouped]
        stds = [pstdev(scores) for scores in grouped]
        axis.errorbar(
            range(len(values)), means, yerr=stds,
            marker="o", color="#4c78a8", capsize=4,
        )
        axis.set_xticks(range(len(values)), [str(value) for value in values])
        axis.set(title=parameter, xlabel="Value", ylabel="Marginal mean CV loss")
        axis.grid(alpha=0.25)
    figure.suptitle("CV hyperparameter effects")
    generated.extend(_save_figure(figure, output_dir, "cv_hyperparameter_effects"))
    plt.close(figure)

    airports = sorted({
        airport for scores in airport_scores.values() for airport in scores
    })
    airport_rows: list[dict[str, Any]] = []
    if airports:
        heatmap_candidates = [int(row["candidate"]) for row in ordered]
        matrix = [
            [airport_scores[candidate].get(airport, float("nan")) for airport in airports]
            for candidate in heatmap_candidates
        ]
        for candidate in heatmap_candidates:
            for airport in airports:
                airport_rows.append({
                    "candidate": candidate,
                    "airport": airport,
                    "mean_val_loss": airport_scores[candidate].get(airport),
                })
        airport_csv = output_dir / "cv_airport_scores.csv"
        _write_csv(
            airport_csv, airport_rows,
            ["candidate", "airport", "mean_val_loss"],
        )
        generated.append(airport_csv)

        figure, axis = plt.subplots(
            figsize=(max(6, 1.1 * len(airports)), max(5, 0.28 * len(matrix)))
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(airports)), airports, rotation=45, ha="right")
        axis.set_yticks(
            range(len(heatmap_candidates)),
            [f"#{candidate}" for candidate in heatmap_candidates],
        )
        axis.set(title="CV loss by airport", xlabel="Airport", ylabel="Candidate (ranked)")
        figure.colorbar(image, ax=axis, label="Mean validation loss")
        generated.extend(_save_figure(figure, output_dir, "cv_airport_heatmap"))
        plt.close(figure)

    return generated, {
        "candidate_count": len(candidates),
        "best_candidate": result.get("best_candidate"),
        "best_mean_val_macro_loss": result.get("best_mean_val_macro_loss"),
        "best_overrides": result.get("best_overrides"),
    }


def _plot_training(
    history_path: Path, output_dir: Path, plt
) -> tuple[list[Path], dict[str, Any]]:
    result = _load_json(history_path)
    history = result.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{history_path} has no epoch history")

    airports = sorted({
        airport
        for epoch in history
        for airport in epoch.get("val_by_airport", {})
    })
    rows: list[dict[str, Any]] = []
    for epoch in history:
        row = {
            "epoch": epoch["epoch"],
            "train_loss": epoch["train_loss"],
            "val_loss": epoch["val_loss"],
            "seconds": epoch["seconds"],
        }
        row.update({
            f"val_{airport}": epoch.get("val_by_airport", {}).get(airport)
            for airport in airports
        })
        rows.append(row)

    epoch_csv = output_dir / "training_epochs.csv"
    _write_csv(
        epoch_csv, rows,
        ["epoch", "train_loss", "val_loss", "seconds", *[f"val_{a}" for a in airports]],
    )
    generated = [epoch_csv]

    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Joint normalized loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["seconds"] for row in rows], color="#f58518")
    axes[1].set(title="Epoch duration", xlabel="Epoch", ylabel="Seconds")
    axes[1].grid(alpha=0.25)
    generated.extend(_save_figure(figure, output_dir, "training_curves"))
    plt.close(figure)

    if airports:
        figure, axis = plt.subplots(figsize=(10, 5))
        for airport in airports:
            axis.plot(
                epochs,
                [row[f"val_{airport}"] for row in rows],
                label=airport,
            )
        axis.set(
            title="Validation loss by airport",
            xlabel="Epoch",
            ylabel="Joint normalized loss",
        )
        axis.legend(ncol=min(4, len(airports)))
        axis.grid(alpha=0.25)
        generated.extend(_save_figure(figure, output_dir, "training_airport_loss"))
        plt.close(figure)

    return generated, {
        "epochs_run": result.get("epochs_run"),
        "best_val_loss": result.get("best_val_loss"),
        "device": result.get("device"),
        "flights": result.get("flights"),
    }


def plot_run(run_dir: Path, output_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot(output_dir)

    cv_path = run_dir / "cross_validation" / "cv_results.json"
    history_path = run_dir / "history.json"
    if not cv_path.is_file() and not history_path.is_file():
        raise ValueError(f"no cv_results.json or history.json found under {run_dir}")

    generated: list[Path] = []
    summaries: dict[str, Any] = {}
    sections: list[tuple[str, list[str]]] = []
    sources: dict[str, str] = {}
    if cv_path.is_file():
        files, summaries["cv"] = _plot_cv(cv_path, output_dir, plt)
        generated.extend(files)
        sources["cv"] = str(cv_path)
        sections.append(("Cross-validation", [
            "cv_candidate_scores.png",
            "cv_hyperparameter_effects.png",
            "cv_airport_heatmap.png",
        ]))
    if history_path.is_file():
        files, summaries["training"] = _plot_training(history_path, output_dir, plt)
        generated.extend(files)
        sources["training"] = str(history_path)
        sections.append(("Final training", [
            "training_curves.png",
            "training_airport_loss.png",
        ]))

    index_path = output_dir / "index.md"
    lines = [
        "# TS experiment plots",
        "",
        f"Run directory: `{run_dir}`",
        "",
    ]
    for title, images in sections:
        existing = [name for name in images if (output_dir / name).is_file()]
        if not existing:
            continue
        lines.extend([f"## {title}", ""])
        for name in existing:
            lines.extend([f"![{Path(name).stem}]({name})", ""])
    index_path.write_text("\n".join(lines), encoding="utf-8")
    generated.append(index_path)

    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "ts-plot-manifest-v1",
        "run_dir": str(run_dir),
        "sources": sources,
        "summaries": summaries,
        "generated_files": sorted(path.name for path in generated),
    }, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        manifest = plot_run(args.run_dir, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"plots: {manifest.parent}")
    print(f"index: {manifest.parent / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
