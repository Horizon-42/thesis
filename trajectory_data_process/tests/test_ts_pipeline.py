"""Top-level TS orchestration: CV/train scope, pooled reuse, and per-airport tails."""

from __future__ import annotations

import hashlib
import json

import pytest

import run_ts_coordinate_ablation as ablation
import run_ts_pipeline as pipeline


def _manifest(root, airport: str, generation: int = 1):
    path = root / airport / "arrivals" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"airport": airport, "generation": generation}), encoding="utf-8"
    )
    return path


def test_per_airport_plan_runs_cv_then_final_train(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")

    assert pipeline.discover_k_airports() == ["KRDU"]
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "window", training_mode="per-airport"
    )
    commands = [command for _label, command in plan.steps(skip_cv=False, reuse_checkpoint=False)]
    assert commands[0][2] == "cross-validate"
    assert commands[1][2] == "train"
    assert all(str(manifest) in command for command in commands)
    assert pipeline.run_training(
        plan, dry_run=True, skip_cv=False, skip_train=False
    ) is True


def test_training_plan_can_isolate_ablation_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    output_dir = tmp_path / "ablation" / "enu"
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "full", training_mode="pooled",
        output_dir=output_dir,
    )

    assert plan.train_dir == output_dir
    assert str(manifest) in plan.cv_step()[1]
    assert str(plan.best_config) in plan.train_step(use_best_config=True)[1]
    assert "--config-overrides" not in plan.train_step(use_best_config=False)[1]


def test_pooled_plan_trains_once_and_publishes_each_airport(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    krdu = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    kstl = _manifest(pipeline.HARVEST_ROOT, "KSTL")

    training = pipeline.TrainingPlan(
        ("KRDU", "KSTL"), "itransformer", "full", training_mode="pooled"
    )
    commands = [command for _label, command in training.steps(
        skip_cv=False, reuse_checkpoint=False
    )]
    assert all(str(krdu) in command and str(kstl) in command for command in commands)
    assert commands[0].count("--data") == 2
    assert "airport-flight-balanced" in commands[0]

    krdu_prediction = pipeline.PredictionPlan(training, "KRDU", ("eval",))
    kstl_prediction = pipeline.PredictionPlan(training, "KSTL", ("eval",))
    assert krdu_prediction.training.checkpoint == kstl_prediction.training.checkpoint
    assert krdu_prediction.pred_dir != kstl_prediction.pred_dir
    assert "pooled" in krdu_prediction.category


def test_prediction_outputs_and_categories_are_split_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    training = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "window", training_mode="per-airport"
    )
    test = pipeline.PredictionPlan(training, "KRDU", ("czml",), split="test")
    train = pipeline.PredictionPlan(training, "KRDU", ("czml",), split="train")
    assert test.pred_dir != train.pred_dir
    assert test.category != train.category
    assert test.pred_dir.name.endswith("_test")
    assert train.category.endswith("_train")


def test_prediction_labels_distinguish_coordinate_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    enu = pipeline.PredictionPlan(
        pipeline.TrainingPlan(
            ("KRDU",), "itransformer", "full", training_mode="pooled",
            coordinate_frame="enu",
        ),
        "KRDU",
        ("czml",),
    )
    aligned = pipeline.PredictionPlan(
        pipeline.TrainingPlan(
            ("KRDU",), "itransformer", "full", training_mode="pooled",
            coordinate_frame="runway-aligned",
        ),
        "KRDU",
        ("czml",),
    )

    assert enu.label != aligned.label
    assert "ENU" in enu.label
    assert "runway-aligned" in aligned.label


def test_skip_train_rejects_checkpoint_for_different_manifest_set(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU", generation=2)

    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "window", training_mode="per-airport"
    )
    plan.train_dir.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"checkpoint")
    plan.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(plan.checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {"KRDU": hashlib.sha256(
            json.dumps({"airport": "KRDU", "generation": 1}).encode()
        ).hexdigest()},
    }), encoding="utf-8")

    assert "arrival manifests" in (plan.checkpoint_reuse_error() or "")


def _ablation_cv_result(plan, manifest_digest, *, score):
    return {
        "schema_version": "ts-cross-validation-v1",
        "selection_metric": "mean outer-train-fold airport-macro normalized MSE",
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
        "max_trials": plan.cv_trials,
        "cv_epochs": plan.cv_epochs,
        "cv_patience": plan.cv_patience,
        "auto_batch_size": plan.batch_size == "auto",
        "base_config": plan._expected_cv_base_config(),
        "arrival_manifests": {"KRDU": manifest_digest},
        "trials": [{
            "trial": 0,
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


def test_cv_reuse_rejects_changed_split_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "full", training_mode="pooled",
        seed=29, cv_folds=3, cv_trials=4, cv_epochs=12, cv_patience=4,
        output_dir=tmp_path / "run",
    )
    stale = _ablation_cv_result(plan, digest, score=0.7)
    stale["base_config"]["seed"] = 1337
    plan.cv_dir.mkdir(parents=True)
    plan.cv_results.write_text(json.dumps(stale), encoding="utf-8")
    plan.best_config.write_text(json.dumps(stale["best_overrides"]), encoding="utf-8")

    assert "seed" in (plan.cv_reuse_error() or "")
    commands = plan.steps(skip_cv=True, reuse_checkpoint=False)
    assert "--config-overrides" not in commands[0][1]


def test_cv_reuse_accepts_the_exact_current_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", "full", training_mode="pooled",
        seed=29, cv_folds=3, cv_trials=4, cv_epochs=12, cv_patience=4,
        output_dir=tmp_path / "run",
    )
    current = _ablation_cv_result(plan, digest, score=0.7)
    plan.cv_dir.mkdir(parents=True)
    plan.cv_results.write_text(json.dumps(current), encoding="utf-8")
    plan.best_config.write_text(json.dumps(current["best_overrides"]), encoding="utf-8")

    assert plan.cv_reuse_error() is None
    commands = plan.steps(skip_cv=True, reuse_checkpoint=False)
    assert str(plan.best_config) in commands[0][1]


def test_coordinate_ablation_accepts_only_paired_cv_and_selects_by_cv(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plans = {
        frame: pipeline.TrainingPlan(
            ("KRDU",), "itransformer", "full", training_mode="pooled",
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
        ("KRDU",), "itransformer", "full", training_mode="pooled",
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
        "arrival_manifests": result["arrival_manifests"],
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
