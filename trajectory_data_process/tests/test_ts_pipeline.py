"""Top-level TS orchestration: CV/train scope, pooled reuse, and per-airport tails."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

import plot_ts_results as result_plots
import run_ts_coordinate_ablation as ablation
import run_ts_cv as cv_runner
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
        ("KRDU",), "itransformer", training_mode="per-airport"
    )
    commands = [command for _label, command in plan.steps(skip_cv=False, reuse_checkpoint=False)]
    assert commands[0][2] == "cross-validate"
    assert commands[1][2] == "train"
    assert all(str(manifest) in command for command in commands)
    assert pipeline.run_training(
        plan, dry_run=True, skip_cv=False, skip_train=False
    ) is True


def test_final_training_prints_the_resolved_config_before_the_command(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        n_segments=32, epochs=7, seed=29, batch_size="2048",
    )

    assert pipeline.run_training(
        plan, dry_run=True, skip_cv=True, skip_train=False
    ) is True
    output = capsys.readouterr().out

    assert "config    : TSConfig defaults" in output
    assert (
        "trajectory: dt=2s, L=60, prediction_output=state, mode=normalized, "
        "output=32, N=32, "
        "H_full=300, H_window=30, "
        "frame=enu, anchor=fixed L-1"
    ) in output
    assert "network   : d_model=256, d_ff=512, heads=8, layers=3" in output
    assert "optimizer : lr=0.0005" in output
    assert "loss      : final_time=1, kinematic=3, terminal=0.02" in output
    assert (
        "runtime   : batch=2048, device=auto, seed=29, aircraft=A320, "
        "aircraft_filter=all"
    ) in output
    assert output.index("config    :") < output.index("[1/1 final train")


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
    assert "(27 candidates)" in output
    assert "--trials" not in output
    assert "after CV:" in output and "plot_ts_results.py" in output


def test_pipeline_defaults_to_both_pooled_models_with_batch_2048(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    _manifest(pipeline.HARVEST_ROOT, "KSTL")
    monkeypatch.setattr(
        "sys.argv",
        ["run_ts_pipeline.py", "--dry-run", "--outputs", "eval"],
    )

    pipeline.main()
    output = capsys.readouterr().out

    assert "2 training cell(s), mode=pooled, airports=KRDU,KSTL" in output
    assert "--model itransformer" in output
    assert "--model patchtst" in output
    assert "--batch-size 2048" in output
    assert "--split train" in output
    assert "--split val" in output
    assert "--split test" not in output


def test_pipeline_requires_explicit_release_before_test(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ts_pipeline.py", "--dry-run", "--models", "itransformer",
            "--split", "test",
        ],
    )

    with pytest.raises(SystemExit):
        pipeline.main()


def test_pipeline_test_release_is_an_explicit_audited_step(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ts_pipeline.py", "--dry-run", "--models", "itransformer",
            "--outputs", "eval", "--split", "test", "--release-test",
        ],
    )

    pipeline.main()
    output = capsys.readouterr().out

    assert " freeze-test " in output
    assert "--test-release" in output


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


def test_training_plan_can_isolate_ablation_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    output_dir = tmp_path / "ablation" / "enu"
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        output_dir=output_dir,
    )

    assert plan.train_dir == output_dir
    assert str(manifest) in plan.cv_step()[1]
    assert str(plan.best_config) in plan.train_step(use_best_config=True)[1]
    assert "--config-overrides" not in plan.train_step(use_best_config=False)[1]


def test_final_train_leaves_n_segments_to_the_cv_override(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        n_segments=64, output_dir=tmp_path / "run",
    )

    assert "--n-segments" in plan.cv_step()[1]
    assert "--n-segments" in plan.train_step(use_best_config=False)[1]
    assert "--n-segments" not in plan.train_step(use_best_config=True)[1]


def test_final_train_keeps_fixed_n_when_cv_does_not_tune_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        n_segments=64,
        cv_parameters=("learning_rate", "d_model"),
        output_dir=tmp_path / "run",
    )

    command = plan.train_step(use_best_config=True)[1]
    assert command[command.index("--n-segments") + 1] == "64"


def test_pooled_plan_trains_once_and_publishes_each_airport(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    krdu = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    kstl = _manifest(pipeline.HARVEST_ROOT, "KSTL")

    training = pipeline.TrainingPlan(
        ("KRDU", "KSTL"), "itransformer", training_mode="pooled"
    )
    commands = [command for _label, command in training.steps(
        skip_cv=False, reuse_checkpoint=False
    )]
    assert all(str(krdu) in command and str(kstl) in command for command in commands)
    assert commands[0].count("--data") == 2
    assert "--sampling-strategy" not in commands[0]
    assert "--samples-per-epoch" not in commands[0]
    assert "--random-train-anchor" not in commands[0]

    rolling = pipeline.TrainingPlan(
        ("KRDU", "KSTL"), "itransformer", training_mode="pooled",
        random_train_anchor=True,
    )
    assert "--random-train-anchor" in rolling.cv_step()[1]

    krdu_prediction = pipeline.PredictionPlan(training, "KRDU", ("eval",))
    kstl_prediction = pipeline.PredictionPlan(training, "KSTL", ("eval",))
    assert krdu_prediction.training.checkpoint == kstl_prediction.training.checkpoint
    assert krdu_prediction.pred_dir != kstl_prediction.pred_dir
    assert "pooled" in krdu_prediction.category


def test_fixed_and_random_anchor_modes_use_distinct_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")

    fixed = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled"
    )
    random = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        random_train_anchor=True,
    )
    common_grid = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
    )
    fixed_prediction = pipeline.PredictionPlan(fixed, "KRDU", ("eval",))
    random_prediction = pipeline.PredictionPlan(random, "KRDU", ("eval",))
    common_grid_prediction = pipeline.PredictionPlan(
        common_grid, "KRDU", ("eval",)
    )

    assert fixed.train_dir != random.train_dir
    assert fixed_prediction.pred_dir != random_prediction.pred_dir
    assert fixed_prediction.category != random_prediction.category
    assert fixed.train_dir != common_grid.train_dir
    assert fixed_prediction.pred_dir != common_grid_prediction.pred_dir
    assert "--checkpoint-selection-metric" in common_grid.train_step(
        use_best_config=False
    )[1]
    assert "--random-train-anchor-min-future-s" in random.train_step(
        use_best_config=False
    )[1]


def test_three_horizon_modes_use_distinct_commands_and_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")
    plans = {
        mode: pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled", horizon_mode=mode
        )
        for mode in ("normalized", "full", "window")
    }

    assert len({plan.train_dir for plan in plans.values()}) == 3
    for mode, plan in plans.items():
        command = plan.train_step(use_best_config=False)[1]
        assert command[command.index("--horizon-mode") + 1] == mode


def test_prediction_outputs_and_categories_are_split_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    training = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="per-airport"
    )
    test = pipeline.PredictionPlan(training, "KRDU", ("czml",), split="test")
    train = pipeline.PredictionPlan(training, "KRDU", ("czml",), split="train")
    assert test.pred_dir != train.pred_dir
    assert test.category != train.category
    assert test.pred_dir.name.endswith("_test")
    assert train.category.endswith("_train")
    assert test.label.startswith("Test split (held-out)")
    assert train.label.startswith("Training split (in-sample)")
    test_czml = test.steps()[-1][1]
    train_czml = train.steps()[-1][1]
    test_predict = test.steps()[0][1]
    assert "--test-release" in test_predict
    assert test_czml[test_czml.index("--dataset-split") + 1] == "test"
    assert train_czml[train_czml.index("--dataset-split") + 1] == "train"


def test_prediction_labels_distinguish_coordinate_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    enu = pipeline.PredictionPlan(
        pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled",
            coordinate_frame="enu",
        ),
        "KRDU",
        ("czml",),
    )
    aligned = pipeline.PredictionPlan(
        pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled",
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
        ("KRDU",), "itransformer", training_mode="per-airport"
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


@pytest.mark.parametrize(
    ("trained_policy", "requested_policy"),
    ((False, True), (True, False)),
)
def test_skip_train_rejects_checkpoint_from_opposite_anchor_policy(
    tmp_path, monkeypatch, trained_policy, requested_policy
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    shared_output = tmp_path / "shared-run"
    trained = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        random_train_anchor=trained_policy, output_dir=shared_output,
    )
    requested = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        random_train_anchor=requested_policy, output_dir=shared_output,
    )
    trained.train_dir.mkdir(parents=True)
    trained.checkpoint.write_bytes(b"checkpoint")
    trained_config, _source = trained.resolved_train_config(use_best_config=False)
    trained.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(trained.checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {
            "KRDU": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
            "random_train_anchor": trained_policy,
            "random_train_anchor_min_future_s": trained_config.random_train_anchor_min_future_s,
            "checkpoint_selection_metric": trained_config.checkpoint_selection_metric,
            "validation_common_grid_points": trained_config.validation_common_grid_points,
        "horizon_mode": trained.horizon_mode,
        "prediction_output": trained_config.prediction_output,
        "aircraft_filter": trained_config.aircraft_filter,
        "pred_len": trained_config.pred_len,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": trained_config.lr_plateau_factor,
            "patience": trained_config.lr_plateau_patience,
        },
    }), encoding="utf-8")

    assert trained.checkpoint_reuse_error() is None
    assert "random_train_anchor" in (requested.checkpoint_reuse_error() or "")


def test_skip_train_rejects_checkpoint_from_opposite_horizon_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    shared_output = tmp_path / "shared-run"
    trained = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        horizon_mode="normalized", output_dir=shared_output,
    )
    requested = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        horizon_mode="full", full_horizon_steps=300,
        output_dir=shared_output,
    )
    trained.train_dir.mkdir(parents=True)
    trained.checkpoint.write_bytes(b"checkpoint")
    trained_config, _source = trained.resolved_train_config(use_best_config=False)
    trained.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(trained.checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {
            "KRDU": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
            "random_train_anchor": False,
            "random_train_anchor_min_future_s": trained_config.random_train_anchor_min_future_s,
            "checkpoint_selection_metric": trained_config.checkpoint_selection_metric,
            "validation_common_grid_points": trained_config.validation_common_grid_points,
        "horizon_mode": trained_config.horizon_mode,
        "prediction_output": trained_config.prediction_output,
        "aircraft_filter": trained_config.aircraft_filter,
        "pred_len": trained_config.pred_len,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": trained_config.lr_plateau_factor,
            "patience": trained_config.lr_plateau_patience,
        },
    }), encoding="utf-8")

    assert trained.checkpoint_reuse_error() is None
    assert "horizon mode" in (requested.checkpoint_reuse_error() or "")


def test_skip_train_rejects_window_checkpoint_with_different_rollout_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    shared_output = tmp_path / "shared-window-run"
    trained = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        horizon_mode="window", full_horizon_steps=300, window_horizon_steps=30,
        output_dir=shared_output,
    )
    requested = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        horizon_mode="window", full_horizon_steps=600, window_horizon_steps=30,
        output_dir=shared_output,
    )
    trained.train_dir.mkdir(parents=True)
    trained.checkpoint.write_bytes(b"window checkpoint")
    trained_config, _source = trained.resolved_train_config(use_best_config=False)
    trained.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(trained.checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {
            "KRDU": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
            "random_train_anchor": False,
            "random_train_anchor_min_future_s": trained_config.random_train_anchor_min_future_s,
            "checkpoint_selection_metric": trained_config.checkpoint_selection_metric,
            "validation_common_grid_points": trained_config.validation_common_grid_points,
        "horizon_mode": trained_config.horizon_mode,
        "prediction_output": trained_config.prediction_output,
        "aircraft_filter": trained_config.aircraft_filter,
        "pred_len": trained_config.pred_len,
        "full_horizon_steps": trained_config.full_horizon_steps,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": trained_config.lr_plateau_factor,
            "patience": trained_config.lr_plateau_patience,
        },
    }), encoding="utf-8")

    assert trained.checkpoint_reuse_error() is None
    assert "rollout cap" in (requested.checkpoint_reuse_error() or "")


def test_fixed_horizon_plan_drops_normalized_only_n_from_cv(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    _manifest(pipeline.HARVEST_ROOT, "KRDU")

    for mode in ("full", "window"):
        plan = pipeline.TrainingPlan(
            ("KRDU",), "itransformer", training_mode="pooled",
            horizon_mode=mode, output_dir=tmp_path / mode,
        )
        assert plan.cv_parameters == ("learning_rate", "d_model")
        command = plan.cv_step()[1]
        assert command[command.index("--cv-parameters") + 1] == "learning_rate,d_model"


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


def test_cv_reuse_rejects_changed_split_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        seed=29, cv_folds=3, cv_epochs=12, cv_patience=4,
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
        ("KRDU",), "itransformer", training_mode="pooled",
        seed=29, cv_folds=3, cv_epochs=12, cv_patience=4,
        output_dir=tmp_path / "run",
    )
    current = _ablation_cv_result(plan, digest, score=0.7)
    plan.cv_dir.mkdir(parents=True)
    plan.cv_results.write_text(json.dumps(current), encoding="utf-8")
    plan.best_config.write_text(json.dumps(current["best_overrides"]), encoding="utf-8")

    assert plan.cv_reuse_error() is None
    commands = plan.steps(skip_cv=True, reuse_checkpoint=False)
    assert str(plan.best_config) in commands[0][1]


def test_cv_reuse_rejects_a_different_parameter_set(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    manifest = _manifest(pipeline.HARVEST_ROOT, "KRDU")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        seed=29, output_dir=tmp_path / "run",
    )
    result = _ablation_cv_result(original, digest, score=0.7)
    original.cv_dir.mkdir(parents=True)
    original.cv_results.write_text(json.dumps(result), encoding="utf-8")
    original.best_config.write_text(json.dumps(result["best_overrides"]), encoding="utf-8")

    changed = pipeline.TrainingPlan(
        ("KRDU",), "itransformer", training_mode="pooled",
        seed=29,
        cv_parameters=("learning_rate", "d_model"),
        output_dir=tmp_path / "run",
    )
    assert "tuned_parameters" in (changed.cv_reuse_error() or "")


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
