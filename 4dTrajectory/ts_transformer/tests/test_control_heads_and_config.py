"""The control heads and the control config contract: bounds, partitions, recipes, clocks, legacy configs.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import argparse
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ts_transformer.data.channels as ch
import ts_transformer.cli.common as cli_common
import ts_transformer.outputs.control.dynamics.rollout as control_rollout_module
import ts_transformer.outputs.control.loss.fixed_dt as fixed_dt_loss_module
import ts_transformer.training.objective as objective
import ts_transformer.experiments.pipeline as pipeline_module
from ts_transformer.config import (
    HORIZON_FULL,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CONTROL_RECIPE_SIMPLE_V1,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    control_simple_v1_overrides,
)
from ts_transformer.outputs.control.training.diagnostics import (
    clip_gradients_by_global_norm,
    gradient_norms,
)
from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import (
    ControlBounds,
    ControlOutputHead,
    ControlPrediction,
)
from ts_transformer.outputs.control.heads import UniformDurationControlHead

AIRPORT, RUNWAY = "KRDU", "05L"


# ── Models ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_both_models_return_structured_states_and_final_time(model_name):
    config = TSConfig(model=model_name, seq_len=32, n_segments=16, d_model=32, n_heads=4,
                      d_ff=64, e_layers=1)
    model = build_model(config)
    x = torch.randn(3, config.seq_len, config.enc_in)
    prediction = model(x)
    assert prediction.states.shape == (3, config.n_segments, config.enc_in)
    assert prediction.final_time_s.shape == (3,)
    assert torch.all(prediction.final_time_s > 0.0)


def test_control_head_bounds_controls_and_partitions_time_non_uniformly():
    bounds = ControlBounds(
        lower=(0.0, -math.pi / 4, 0.5),
        upper=(120_000.0, math.pi / 4, 2.0),
    )
    head = ControlOutputHead(input_dim=8, n_segments=5, bounds=bounds)
    result = head(torch.randn(3, 8), torch.tensor([100.0, 240.0, 60.0]))

    assert result.controls.shape == (3, 5, 3)
    assert result.segment_durations.shape == (3, 5)
    assert torch.all(result.segment_durations > 0.0)
    assert torch.allclose(result.segment_durations.sum(dim=-1), result.final_time_s)
    assert torch.all(result.controls >= head.lower)
    assert torch.all(result.controls <= head.upper)


def test_control_head_reserves_uniform_duration_mass_to_prevent_partition_collapse():
    head = ControlOutputHead(
        input_dim=1,
        n_segments=5,
        bounds=ControlBounds(
            lower=(0.0, -math.pi / 4, 0.5),
            upper=(120_000.0, math.pi / 4, 2.0),
        ),
        duration_uniform_floor=0.8,
    )
    with torch.no_grad():
        head.duration_projection.weight.zero_()
        head.duration_projection.bias.copy_(
            torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])
        )

    result = head(torch.zeros(1, 1), torch.tensor([100.0]))
    fractions = result.segment_durations[0] / result.final_time_s[0]

    # 80% of the duration is reserved uniformly. The remaining 20% stays learnable,
    # so even adversarial logits cannot recreate the historical ~95% single segment.
    fractions = fractions.detach()
    assert fractions.min().item() >= 0.8 / 5.0 - 1e-6
    assert fractions.max().item() <= 0.2 + 0.8 / 5.0 + 1e-6
    assert fractions.sum().item() == pytest.approx(1.0)


def test_uniform_duration_control_head_has_no_duration_parameters():
    head = UniformDurationControlHead(input_dim=8, n_segments=4)
    lower = torch.tensor([[0.0, -0.7, 0.5], [0.0, -0.5, 0.6]])
    upper = torch.tensor([[200_000.0, 0.7, 2.0], [150_000.0, 0.5, 1.8]])
    final_time = torch.tensor([80.0, 100.0], requires_grad=True)

    prediction = head(
        torch.randn(2, 8), final_time, lower=lower, upper=upper
    )

    assert not any("duration_projection" in name for name, _ in head.named_parameters())
    torch.testing.assert_close(
        prediction.segment_durations,
        torch.tensor([[20.0] * 4, [25.0] * 4]),
    )
    torch.testing.assert_close(
        prediction.segment_durations.sum(dim=1), prediction.final_time_s
    )
    assert torch.all(prediction.controls >= lower[:, None, :])
    assert torch.all(prediction.controls <= upper[:, None, :])


def test_control_simple_v1_is_a_frozen_serialized_recipe():
    config = TSConfig(
        control_recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        **control_simple_v1_overrides(),
    )

    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM
    assert config.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
    assert config.checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE
    assert control_recipe(config)["name"] == CONTROL_RECIPE_SIMPLE_V1
    assert TSConfig.from_dict(config.to_dict()) == config
    assert "bounded-control-uniform-duration" in objective.target_contract(config)
    with pytest.raises(ValueError, match="simple-v1 recipe fields are frozen"):
        replace(config, n_segments=32)


def test_control_simple_v1_cli_applies_defaults_and_rejects_conflicts(capsys):
    parser = argparse.ArgumentParser()
    cli_common.add_data_args(parser)
    cli_common.add_training_args(parser)
    args = parser.parse_args(
        [
            "--data", "unused.json",
            "--output-dir", "unused-output",
            "--control-recipe-name", CONTROL_RECIPE_SIMPLE_V1,
            "--seed", "2027",
        ]
    )

    config, batch_auto = cli_common.config_from_args(args, parser)

    assert not batch_auto
    assert config.control_recipe_name == CONTROL_RECIPE_SIMPLE_V1
    assert config.seed == 2027
    assert config.n_segments == 64
    assert config.batch_size == 512
    assert config.control_duration_parameterization == CONTROL_DURATION_UNIFORM

    conflicting = parser.parse_args(
        [
            "--data", "unused.json",
            "--output-dir", "unused-output",
            "--control-recipe-name", CONTROL_RECIPE_SIMPLE_V1,
            "--n-segments", "32",
        ]
    )
    with pytest.raises(SystemExit) as info:
        cli_common.config_from_args(conflicting, parser)
    assert info.value.code == 2 and "recipe fields are frozen: n_segments=32" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((0.0,), (1.0,)),
        ((0.0, -1.0, 0.5), (1.0, 1.0)),
        ((0.0, -1.0, 0.5, 2.0), (1.0, 1.0, 2.0, 3.0)),
    ],
)
def test_control_bounds_require_one_bound_per_control(lower, upper):
    with pytest.raises(ValueError, match="exactly 3"):
        ControlBounds(lower=lower, upper=upper)


def test_control_output_is_parallel_and_requires_normalized_horizon():
    with pytest.raises(ValueError, match="requires horizon_mode='normalized'"):
        TSConfig(prediction_output=PREDICTION_CONTROL, horizon_mode=HORIZON_FULL)


def test_transport_chart_dynamics_is_an_explicit_control_only_contract():
    backend = CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=backend,
    )

    assert control_recipe(config)["dynamics_backend"] == backend
    assert f"+dynamics={backend}-v1" in objective.target_contract(config)
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(control_dynamics_backend=backend)


def test_pipeline_carries_and_names_common_training_cohort():
    fixed = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        training_cohort_min_future_s=60.0,
    )
    random = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        random_train_anchor=True,
        training_cohort_min_future_s=60.0,
    )
    recipe = fixed._recipe_args()
    config, _source = fixed.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        fixed, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--training-cohort-min-future-s") + 1] == "60.0"
    assert config.training_cohort_min_future_s == pytest.approx(60.0)
    assert "cohort_min60" in fixed.train_dir.name
    assert "cohort_min60" in random.train_dir.name
    assert fixed.train_dir != random.train_dir
    assert "cohort_min60" in prediction.pred_dir.name
    assert "cohort_min60" in prediction.category


def test_control_state_supervision_clock_rejects_unknown_value():
    with pytest.raises(ValueError, match="control_state_supervision_clock"):
        TSConfig(control_state_supervision_clock="future")


def test_fixed_dt_control_state_loss_requires_observed_single_control_clock():
    with pytest.raises(ValueError, match="requires.*observed"):
        TSConfig(
            prediction_output=PREDICTION_CONTROL,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        )
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(
            prediction_output=PREDICTION_STATE,
            control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
            control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        )


def test_uniform_control_durations_reject_non_control_outputs():
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(
            prediction_output=PREDICTION_STATE,
            control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        )


def test_legacy_control_config_without_duration_parameterization_is_rejected():
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop("control_duration_parameterization")

    with pytest.raises(
        ValueError,
        match="missing control_duration_parameterization.*regenerate",
    ):
        TSConfig.from_dict(serialized)


def test_legacy_control_config_without_state_loss_grid_is_rejected():
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop("control_state_loss_grid")

    with pytest.raises(ValueError, match="missing control_state_loss_grid.*regenerate"):
        TSConfig.from_dict(serialized)


@pytest.mark.parametrize(
    "field",
    [
        "control_state_objective",
        "control_state_duration_gradient",
        "control_gradient_clip_norm",
        "control_dynamics_backend",
        "control_duration_uniform_floor",
    ],
)
def test_legacy_control_config_without_physical_criteria_recipe_is_rejected(field):
    serialized = TSConfig(prediction_output=PREDICTION_CONTROL).to_dict()
    serialized.pop(field)

    with pytest.raises(ValueError, match=f"missing {field}.*regenerate"):
        TSConfig.from_dict(serialized)


def test_control_gradient_clip_is_explicit_and_control_only():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_gradient_clip_norm=20.0,
    )

    assert control_recipe(config)["gradient_clip_norm"] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(config, control_gradient_clip_norm=-1.0)
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(control_gradient_clip_norm=20.0)


def test_control_gradient_clip_records_preclip_module_norms_and_caps_global_norm():
    class GradientGroups(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = torch.nn.Linear(1, 1, bias=False)
            self.control_head = torch.nn.Linear(1, 1, bias=False)
            self.final_time_head = torch.nn.Linear(1, 1, bias=False)

    model = GradientGroups()
    x = torch.ones(1, 1)
    loss = (
        3.0 * model.feature_encoder(x)
        + 4.0 * model.control_head(x)
        + 0.0 * model.final_time_head(x)
    ).sum()
    loss.backward()

    preclip, coefficient = clip_gradients_by_global_norm(model, 1.0)
    postclip = gradient_norms(model)

    assert coefficient == pytest.approx(0.2)
    assert preclip == pytest.approx(
        {
            "backbone": 3.0,
            "control_head": 4.0,
            "final_time_head": 0.0,
            "total": 5.0,
        }
    )
    assert postclip["total"] == pytest.approx(1.0)


def test_fixed_dt_rollout_closes_the_float32_duration_clock(monkeypatch):
    """The dense seam must hand the rollout a clock that closes on the float64 total."""
    torch.manual_seed(0)
    durations = torch.softmax(torch.randn(1, 64), dim=1) * 580.0
    prediction = ControlPrediction(
        controls=torch.zeros(1, 64, 3),
        segment_durations=durations,
        final_time_s=torch.tensor([580.0]),
    )
    states = torch.zeros(1, 1, len(ch.CHANNELS))
    supervision = FixedDTControlSupervision(
        query_offsets_s=torch.tensor([[580.0]], dtype=torch.float64),
        states=states,
        weights=torch.ones_like(states),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    class CapturingBackend:
        def dense_rollout(
            self,
            inputs,
            query_offsets_s,
            query_valid,
            config,
            *,
            command_hook=None,
        ):
            del command_hook
            total = inputs.segment_durations_s.cumsum(dim=1)[0, -1]
            torch.testing.assert_close(
                total,
                torch.tensor(580.0, dtype=torch.float64),
                rtol=0.0,
                atol=1e-12,
            )
            assert query_offsets_s[0, -1] <= total
            return SimpleNamespace(
                query_channels=torch.zeros(1, 1, len(ch.CHANNELS)),
                segment_end_channels=torch.zeros(1, 64, len(ch.CHANNELS)),
            )

    monkeypatch.setattr(
        control_rollout_module,
        "control_dynamics_backend",
        lambda config: CapturingBackend(),
    )
    dynamics = {
        "initial_state": torch.zeros(1, 7),
        "initial_controls": torch.zeros(1, 3),
        "aero_params": torch.zeros(1, 1),
        "frame_params": torch.zeros(1, 1),
        "max_thrust_n": torch.ones(1),
    }

    _queries, _endpoints, closed_durations = (
        fixed_dt_loss_module.fixed_dt_rollout_channels(
        prediction,
        supervision,
        dynamics,
        TSConfig(),
        )
    )
    torch.testing.assert_close(
        closed_durations.sum(dim=1),
        prediction.final_time_s.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
