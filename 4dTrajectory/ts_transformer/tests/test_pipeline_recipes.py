"""The pipeline runner carries and names the control recipe.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import hashlib
import json

import pytest

import ts_transformer.experiments.pipeline as pipeline_module
from ts_transformer.config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    control_recipe,
)
from ts_transformer.training.train import CHECKPOINT_METADATA_SCHEMA

AIRPORT, RUNWAY = "KRDU", "05L"


def test_pipeline_carries_and_names_complete_control_recipe(tmp_path):
    plan = pipeline_module.TrainingPlan(
        ("KMSY", "KRDU"),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        n_segments=32,
        seed=2027,
        split_seed=1337,
        aircraft_type="A320",
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        batch_size="16",
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        control_rollout_dt=0.5,
        output_dir=tmp_path,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(plan, "KMSY", ("eval",), split="val")

    assert recipe[recipe.index("--prediction-output") + 1] == PREDICTION_CONTROL
    assert recipe[recipe.index("--split-seed") + 1] == "1337"
    assert recipe[recipe.index("--control-duration-parameterization") + 1] == "uniform"
    assert recipe[recipe.index("--control-state-supervision-clock") + 1] == "observed"
    assert (
        recipe[recipe.index("--control-state-loss-grid") + 1]
        == "native-segment-endpoints"
    )
    assert (
        recipe[recipe.index("--control-state-objective") + 1] == "true-time-position"
    )
    assert "--no-control-state-duration-gradient" in recipe
    assert recipe[recipe.index("--control-rollout-integrator-dt-s") + 1] == "0.5"
    assert recipe[recipe.index("--aircraft-filter") + 1] == "openap-direct"
    assert config.prediction_output == PREDICTION_CONTROL
    assert config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM
    assert config.control_state_supervision_clock == CONTROL_STATE_CLOCK_OBSERVED
    assert config.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_NATIVE
    assert (
        config.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
    )
    assert config.control_state_duration_gradient is False
    assert "control" in prediction.pred_dir.name
    assert "uniform_duration" in prediction.pred_dir.name
    assert "observed_clock" in prediction.pred_dir.name
    assert "true_time_position" in prediction.pred_dir.name
    assert "detached_duration_gradient" in prediction.pred_dir.name
    assert "control" in prediction.category
    assert "openap_direct" in prediction.category
    assert "duration=uniform" in prediction.label
    # This IS the simple-v1 loss design (true-time-position on the native grid, detached
    # duration gradients, every auxiliary weight at its default zero), so the grammar names
    # it by that recipe rather than spelling the fields out; the dynamics field stays
    # `point-mass`, which is what simple-v1 (not simple-v1-lag) pairs with.
    assert "simple-v1" in prediction.label
    assert "simple-v1-lag" not in prediction.label


def test_pipeline_carries_and_names_control_gradient_clip():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        epochs=4,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_duration_gradient=False,
        control_gradient_clip_norm=20.0,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert config.control_gradient_clip_norm == pytest.approx(20.0)
    assert recipe[recipe.index("--control-gradient-clip-norm") + 1] == "20"
    assert len(plan.train_dir.name.encode("utf-8")) <= (
        pipeline_module.MAX_PATH_COMPONENT_BYTES
    )
    assert "gradient_clip20" in prediction.category
    assert "grad-clip=20" in prediction.label


def test_pipeline_rejects_control_checkpoint_metadata_without_duration_recipe(
    tmp_path, monkeypatch
):
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="per-airport",
        prediction_output=PREDICTION_CONTROL,
        output_dir=tmp_path / "run",
    )
    plan.train_dir.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "arrivals.json"
    manifest.write_text("{}", encoding="utf-8")
    plan.data_manifests = (manifest,)
    monkeypatch.setattr(pipeline_module, "_manifest_digests", lambda _airports: [])
    roster = tmp_path / "lateral_pass_eligibility.json"
    roster.write_text("{}", encoding="utf-8")
    plan.eligibility_rosters = (roster,)
    eligible_set_digest = hashlib.sha256(b"A").hexdigest()
    monkeypatch.setattr(
        pipeline_module,
        "_eligible_set_digests",
        lambda _airports: {AIRPORT: eligible_set_digest},
    )

    config, _source = plan.resolved_train_config(use_best_config=False)
    legacy_recipe = control_recipe(config)
    legacy_recipe.pop("duration_parameterization")
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "arrival_manifests": [],
        "eligible_sets": {AIRPORT: eligible_set_digest},
        "random_train_anchor": plan.random_train_anchor,
        "training_cohort_min_future_s": plan.training_cohort_min_future_s,
        "random_train_anchor_min_future_s": plan.random_train_anchor_min_future_s,
        "checkpoint_selection_metric": plan.checkpoint_selection_metric,
        "validation_common_grid_points": plan.validation_common_grid_points,
        "prediction_output": config.prediction_output,
        "aircraft_filter": config.aircraft_filter,
        "horizon_mode": config.horizon_mode,
        "pred_len": config.pred_len,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "control_recipe": legacy_recipe,
    }
    plan.checkpoint_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    assert plan.checkpoint_reuse_error() == (
        "checkpoint control recipe does not match the requested recipe"
    )
