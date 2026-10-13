"""Top-level TS orchestration: CV/train scope, pooled reuse, and per-airport tails."""

from __future__ import annotations

import hashlib

import json

import math

import pytest

import ts_transformer.experiments.plot_results as result_plots

import ts_transformer.experiments.coordinate_ablation as ablation

import ts_transformer.experiments.cv as cv_runner

import ts_transformer.experiments.pipeline as pipeline

from ts_transformer.data.lateral_eligibility import (
    LATERAL_PASS_POLICY,
    LATERAL_PASS_ROSTER_SCHEMA,
    default_lateral_pass_roster_path,
)

def _manifest(root, airport: str, generation: int = 1):
    """A fixture harvest for one airport: the arrivals manifest AND the lateral-pass roster
    beside it — the runner refuses to reuse a checkpoint or a CV result without the roster
    (2026-09-08), which is what left twelve of these tests red until 2026-09-10."""
    path = root / airport / "arrivals" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"airport": airport, "generation": generation}), encoding="utf-8"
    )
    default_lateral_pass_roster_path(path).write_text(json.dumps({
        "schema_version": LATERAL_PASS_ROSTER_SCHEMA,
        "airport": airport,
        "policy": LATERAL_PASS_POLICY,
        "sources": {
            "arrival_manifest": str(path),
            "arrival_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evaluation_report": str(path.parent.parent / "approach" / "evaluation_report.json"),
            "evaluation_report_sha256": "0" * 64,
            "evaluation_report_schema": "fixture",
        },
        "counts": {
            "arrival_candidates": 0, "eligible_lateral_pass": 0, "excluded_lateral_fail": 0,
            "excluded_lateral_indeterminate": 0, "evaluation_only": 0,
        },
        "eligible_flight_keys": [],
    }), encoding="utf-8")
    return path

def test_simple_cv_runner_uses_the_fixed_default_grid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    krdu = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    kstl = _manifest(pipeline.HARVEST_ROOT, "KSTL")

    assert cv_runner.main(["--dry-run"]) == 0
    output = capsys.readouterr().out

    assert str(krdu) in output and str(kstl) in output
    assert "--cv-parameters n_segments,learning_rate,d_model" in output
    assert f"--cv-epochs {pipeline.DEFAULT_CV_EPOCHS}" in output
    assert "--cv-patience 6" in output
    assert "--batch-size 2048" in output
    # The default grid: n_segments (5) × learning_rate (3) × d_model (3).
    assert "(45 candidates)" in output
    assert "--trials" not in output
    assert "after CV:" in output and "plot_results" in output

def test_simple_cv_runner_forwards_explicit_batch_size(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")

    assert cv_runner.main(["--batch-size", "2048", "--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "--batch-size 2048" in output

def test_result_plotter_writes_viewable_charts_and_csv_tables(tmp_path):
    run_dir = tmp_path / "run"
    cv_dir = run_dir / "cross_validation"
    cv_dir.mkdir(parents=True)
    folds = [
        {
            "fold": fold,
            "best_epoch": fold + 2,
            "val_by_airport": {"KRDU": 0.4 + fold * 0.1, "KSTL": 0.6 + fold * 0.1},
        }
        for fold in range(2)
    ]
    (cv_dir / "cv_results.json").write_text(json.dumps({
        "tuned_parameters": ["n_segments"],
        "parameter_grid": {"n_segments": [64, 128]},
        "best_candidate": 1,
        "best_mean_val_macro_loss": 0.45,
        "best_overrides": {"n_segments": 128},
        "candidates": [
            {
                "candidate": 0,
                "overrides": {"n_segments": 64},
                "mean_val_macro_loss": 0.6,
                "std_val_macro_loss": 0.1,
                "folds": folds,
            },
            {
                "candidate": 1,
                "overrides": {"n_segments": 128},
                "mean_val_macro_loss": 0.45,
                "std_val_macro_loss": 0.05,
                "folds": folds,
            },
        ],
    }), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps({
        "epochs_run": 2,
        "best_val_loss": 0.4,
        "device": "cpu",
        "flights": {"train": 10, "val": 2, "test": 2},
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "seconds": 1.2,
                "val_by_airport": {"KRDU": 0.6, "KSTL": 0.8},
            },
            {
                "epoch": 2,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "seconds": 1.1,
                "val_by_airport": {"KRDU": 0.3, "KSTL": 0.5},
            },
        ],
    }), encoding="utf-8")

    manifest_path = result_plots.plot_run(run_dir)
    plots = run_dir / "plots"

    assert manifest_path == plots / "plot_manifest.json"
    for name in (
        "index.md",
        "cv_candidate_scores.png",
        "cv_hyperparameter_effects.svg",
        "cv_airport_heatmap.png",
        "cv_candidates.csv",
        "cv_airport_scores.csv",
        "training_curves.png",
        "training_airport_loss.svg",
        "training_epochs.csv",
    ):
        assert (plots / name).is_file(), name

def _ablation_cv_result(plan, manifest_digest, *, score):
    return {
        "schema_version": pipeline.CV_RESULTS_SCHEMA,
        "selection_metric": (
            "mean outer-train-fold airport-macro weighted sum of normalized state MSE, "
            "scaled final-time MSE, position/velocity displacement-consistency MSE, and "
            "terminal-position MSE"
        ),
        "leakage_guard": {
            "search_population": "outer_train_only",
            "outer_validation_used": False,
            "outer_test_used": False,
        },
        "outer_split": {
            "train_flights": 70,
            "validation_flights": 15,
            "test_flights": 15,
            "train_sha256": "1" * 64,
            "validation_sha256": "2" * 64,
            "test_sha256": "3" * 64,
        },
        "n_splits": plan.cv_folds,
        "search_strategy": "exhaustive_grid",
        "tuned_parameters": list(plan.cv_parameters),
        "parameter_grid": pipeline.parameter_grid(plan.cv_parameters),
        "candidate_count": math.prod(
            len(values) for values in pipeline.parameter_grid(plan.cv_parameters).values()
        ),
        "cv_epochs": plan.cv_epochs,
        "cv_patience": plan.cv_patience,
        "auto_batch_size": plan.batch_size == "auto",
        "base_config": plan._expected_cv_base_config(),
        "eligible_sets": pipeline._eligible_set_digests(plan.airports),
        "arrival_manifests": {"KRDU": manifest_digest},
        "candidates": [{
            "candidate": 0,
            "overrides": {"learning_rate": 0.0001},
            "folds": [{
                "fold": fold,
                "train_flights": 46 if fold == 0 else 47,
                "validation_flights": 24 if fold == 0 else 23,
                "validation_by_airport": {"KRDU": 24 if fold == 0 else 23},
                "validation_split_sha256": str(fold + 4) * 64,
                "batch_size": 256,
            } for fold in range(plan.cv_folds)],
        }],
        "best_mean_val_macro_loss": score,
        "best_overrides": {"learning_rate": 0.0001},
    }

def test_coordinate_ablation_accepts_only_paired_cv_and_selects_by_cv(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plans = {
        frame: pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled",
            seed=29, cv_folds=3, output_dir=tmp_path / frame,
        )
        for frame in ablation.FRAMES
    }
    results = {
        "enu": _ablation_cv_result(plans["enu"], digest, score=0.8),
        "runway-aligned": _ablation_cv_result(
            plans["runway-aligned"], digest, score=0.7
        ),
    }

    ablation.assert_comparable(plans, results)
    assert ablation.select_winner(results) == "runway-aligned"

    results["runway-aligned"]["outer_split"]["test_sha256"] = "9" * 64
    with pytest.raises(ablation.AblationContractError, match="outer_split differs"):
        ablation.assert_comparable(plans, results)

def test_coordinate_ablation_rejects_reused_cv_with_a_different_default_n(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plans = {
        frame: pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled",
            seed=29, cv_folds=3, output_dir=tmp_path / frame,
        )
        for frame in ablation.FRAMES
    }
    results = {
        frame: _ablation_cv_result(plan, digest, score=0.7)
        for frame, plan in plans.items()
    }
    for result in results.values():
        result["base_config"]["n_segments"] = 128

    with pytest.raises(ablation.AblationContractError, match="base configuration"):
        ablation.assert_comparable(plans, results)

def test_coordinate_ablation_exact_tie_keeps_enu_baseline():
    results = {
        frame: {"best_mean_val_macro_loss": 0.5} for frame in ablation.FRAMES
    }
    assert ablation.select_winner(results) == "enu"

def test_coordinate_ablation_verifies_final_split_before_test(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        seed=29, output_dir=tmp_path / "run",
    )
    result = _ablation_cv_result(plan, digest, score=0.7)
    plan.train_dir.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"selected checkpoint")
    split_sha256 = {
        "train": result["outer_split"]["train_sha256"],
        "val": result["outer_split"]["validation_sha256"],
        "test": result["outer_split"]["test_sha256"],
    }
    plan.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(plan.checkpoint.read_bytes()).hexdigest(),
        "eligible_sets": result["eligible_sets"],
        "arrival_manifests": result["arrival_manifests"],
        "random_train_anchor": False,
        "split_sha256": split_sha256,
    }), encoding="utf-8")

    assert ablation.verify_final_checkpoint(plan, result)["split_sha256"] == split_sha256
    result["outer_split"]["test_sha256"] = "9" * 64
    with pytest.raises(ablation.AblationContractError, match="split differs"):
        ablation.verify_final_checkpoint(plan, result)

def test_coordinate_ablation_refuses_repeated_or_partial_test(tmp_path):
    result_path = tmp_path / ablation.RESULT_NAME
    result_path.write_text(json.dumps({
        "schema_version": ablation.RESULT_SCHEMA,
        "leakage_guard": {
            "outer_test_evaluation_started": True,
            "outer_test_evaluated": False,
        },
    }), encoding="utf-8")

    with pytest.raises(ablation.AblationContractError, match="already started"):
        ablation.refuse_repeated_test(result_path)
