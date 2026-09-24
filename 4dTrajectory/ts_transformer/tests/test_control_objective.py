"""The control objective: clock alignment, the true-time loss, one differentiable training step, the fixed-dt targets.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
from ts_transformer.outputs.conditioning import CONDITION_WIDTH
import ts_transformer.outputs.dynamics.rollout as control_rollout_module
import ts_transformer.training.objective as objective
import ts_transformer.outputs.control.loss.objective as control_objective
import ts_transformer.experiments.pipeline as pipeline_module
from ts_transformer.config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER, THRUST_FRACTION_CONTRACT
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.fixed_dt_supervision import build_fixed_dt_supervision
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.training.objective import prediction_loss

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_state_config_without_duration_parameterization_remains_loadable():
    serialized = TSConfig().to_dict()
    serialized.pop("control_duration_parameterization")

    restored = TSConfig.from_dict(serialized)

    assert restored.control_duration_parameterization == CONTROL_DURATION_FACTORIZED


def test_observed_control_state_clock_preserves_partition_and_uses_true_total():
    prediction = ControlPrediction(
        controls=torch.randn(2, 2, 3),
        segment_durations=torch.tensor([[1.0, 3.0], [3.0, 2.0]]),
        final_time_s=torch.tensor([4.0, 5.0]),
    )
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
    )

    supervised = control_objective.control_state_supervision_prediction(
        prediction, torch.tensor([8.0, 10.0]), config
    )

    assert supervised.controls is prediction.controls
    torch.testing.assert_close(
        supervised.segment_durations,
        torch.tensor([[2.0, 6.0], [6.0, 4.0]]),
    )
    torch.testing.assert_close(supervised.final_time_s, torch.tensor([8.0, 10.0]))
    torch.testing.assert_close(prediction.final_time_s, torch.tensor([4.0, 5.0]))
    assert objective.target_contract(config) == (
        "bounded-control-nonuniform-duration-casadi-rollout-observed-clock-aligned-v3"
        "+duration-uniform-floor=0.8-v1"
    )


def test_predicted_control_state_clock_preserves_original_training_behavior():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[1.0, 3.0]]),
        final_time_s=torch.tensor([4.0]),
    )
    config = TSConfig(prediction_output=PREDICTION_CONTROL)

    assert control_objective.control_state_supervision_prediction(
        prediction, torch.tensor([8.0]), config
    ) is prediction


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
@pytest.mark.parametrize(
    "duration_parameterization",
    [CONTROL_DURATION_FACTORIZED, CONTROL_DURATION_UNIFORM],
)
def test_control_models_use_per_sample_bounds_and_aircraft_condition(
    model_name, duration_parameterization
):
    config = TSConfig(
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=duration_parameterization,
        seq_len=32,
        n_segments=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
    )
    model = build_model(config)
    lower = torch.tensor([[0.0, -0.5, 0.5], [0.0, -0.7, 0.6]])
    upper = torch.tensor([[10_000.0, 0.5, 1.8], [250_000.0, 0.7, 2.0]])
    dynamics = {
        "condition": torch.rand(2, CONDITION_WIDTH),
        "control_lower": lower,
        "control_upper": upper,
    }
    result = model(torch.randn(2, config.seq_len, config.enc_in), dynamics)

    assert isinstance(result, ControlPrediction)
    assert result.controls.shape == (2, 4, 3)
    assert torch.all(result.controls >= lower[:, None, :])
    assert torch.all(result.controls <= upper[:, None, :])
    assert torch.allclose(result.segment_durations.sum(dim=-1), result.final_time_s)


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_control_models_preserve_ordered_channel_identity(model_name):
    torch.manual_seed(7)
    config = TSConfig(
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        seq_len=32,
        n_segments=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=1,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
    )
    model = build_model(config).eval()
    history = torch.randn(1, config.seq_len, config.enc_in)
    mirrored = history.clone()
    mirrored[:, :, [0, 1]] = history[:, :, [1, 0]]
    mirrored[:, :, [3, 4]] = history[:, :, [4, 3]]
    original_features = model.feature_encoder.encode_features(history)
    swapped_features = model.feature_encoder.encode_features(mirrored)

    assert original_features.shape == (1, config.enc_in * config.d_model)
    assert torch.max(torch.abs(original_features - swapped_features)) > 1e-4


def test_control_loss_aligns_truth_to_predicted_cumulative_clock(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=2,
        n_segments=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        e_layers=1,
        terminal_loss_weight=0.0,
    )
    channel_count = config.enc_in
    # Truth is linear in physical time and was sampled on the uniform true clock [5, 10].
    target = torch.tensor(
        [[[5.0] * channel_count, [10.0] * channel_count]], dtype=torch.float32
    )
    weights = torch.full_like(target, 1.0 / channel_count)
    # The learned partition asks for endpoints at [1, 10]. A clock-aligned target is [1, 10],
    # so this exact rollout must have zero state loss.
    physical_rollout = torch.tensor(
        [[[1.0] * channel_count, [10.0] * channel_count]], dtype=torch.float64
    )
    monkeypatch.setattr(
        control_rollout_module,
        "rollout_control_endpoints",
        lambda _controls, _durations, _dynamics, _config, command_hook=None: SimpleNamespace(
            channels=physical_rollout,
            geodetic_states=torch.zeros(1, 2, 7, dtype=torch.float64),
        ),
    )
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[1.0, 9.0]]),
        final_time_s=torch.tensor([10.0]),
    )
    normalizer = Normalizer(
        mean=np.zeros(channel_count, dtype=np.float64),
        std=np.ones(channel_count, dtype=np.float64),
    )
    dynamics = {
        "control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
        "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32),
    }

    components = objective.prediction_loss_components(
        prediction,
        torch.zeros(1, channel_count),
        target,
        weights,
        torch.tensor([10.0]),
        torch.ones(1),
        config,
        normalizer,
        dynamics,
    )

    assert components.state.item() == pytest.approx(0.0, abs=1e-12)


def test_true_time_control_loss_is_physical_position_endpoint_and_time_only(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        n_segments=2,
        position_loss_scale_m=10_000.0,
        final_time_scale_s=600.0,
        state_endpoint_loss_weight=0.25,
    )
    normalizer = Normalizer(
        mean=np.zeros(config.enc_in),
        std=np.array([100.0, 200.0, 10.0, 2.0, 3.0, 4.0]),
    )
    # Physical position errors are [100, 200, 10] m and [0, 0, 20] m. Deliberately huge
    # future velocity values prove that the minimal objective does not supervise velocity.
    physical_rollout = torch.zeros(1, 2, config.enc_in, dtype=torch.float64)
    physical_rollout[0, 0, list(ch.POSITION_IDX)] = torch.tensor(
        [100.0, 200.0, 10.0], dtype=torch.float64
    )
    physical_rollout[0, 1, ch.IDX["u"]] = 20.0
    physical_rollout[..., list(ch.VELOCITY_IDX)] = 1_000_000.0
    monkeypatch.setattr(
        control_rollout_module,
        "rollout_control_endpoints",
        lambda _controls, _durations, _dynamics, _config, command_hook=None: SimpleNamespace(
            channels=physical_rollout,
            geodetic_states=torch.zeros(1, 2, 7, dtype=torch.float64),
        ),
    )
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[5.0, 5.0]]),
        final_time_s=torch.tensor([10.0]),
    )
    target = torch.zeros(1, 2, config.enc_in)
    weights = torch.full_like(target, 1.0 / config.enc_in)
    dynamics = {
        "control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
        "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32),
    }

    components = control_objective.control_prediction_loss_components(
        prediction,
        torch.zeros(1, config.enc_in),
        target,
        weights,
        torch.tensor([8.0]),
        torch.ones(1),
        config,
        normalizer,
        dynamics,
    )

    expected_path = ((100.0**2 + 200.0**2 + 10.0**2) + 20.0**2) / 2.0
    assert float(components.state) == pytest.approx(expected_path / 10_000.0**2)
    assert float(components.terminal) == pytest.approx(
        0.25 * 20.0**2 / 10_000.0**2
    )
    assert float(components.final_time) == pytest.approx((2.0 / 600.0) ** 2)
    assert float(components.kinematic) == pytest.approx(0.0)


def test_control_model_starts_from_neutral_uniform_rollout():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=8,
        n_segments=4,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=600.0,
    )
    model = build_model(config).eval()
    history = torch.randn(2, config.seq_len, config.enc_in)
    lower = torch.tensor([CONTROL_LOWER, CONTROL_LOWER], dtype=torch.float32)
    upper = torch.tensor([CONTROL_UPPER, CONTROL_UPPER], dtype=torch.float32)
    dynamics = {
        "condition": torch.randn(2, CONDITION_WIDTH),
        "control_lower": lower,
        "control_upper": upper,
    }

    prediction = model(history, dynamics)
    # The untrained head emits the neutral physical control, not a fixed fraction of
    # whatever the bounds happen to be: 20% thrust, wings level, load factor one.
    expected = torch.tensor(THRUST_FRACTION_CONTRACT.neutral).view(1, 1, 3)
    expected_time = math.log(2.0) * config.final_time_scale_s

    torch.testing.assert_close(
        prediction.controls, expected.expand_as(prediction.controls)
    )
    torch.testing.assert_close(
        prediction.final_time_s,
        torch.full_like(prediction.final_time_s, expected_time),
    )
    torch.testing.assert_close(
        prediction.segment_durations,
        prediction.final_time_s[:, None].expand(-1, config.n_segments)
        / config.n_segments,
    )


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_fixed_dt_objective_trains_both_backbones_without_duration_state_gradient(
    model_name,
):
    series, config = _series(
        n_flights=2,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_duration_gradient=False,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0, 1])
    )
    model = build_model(config)
    loss = prediction_loss(
        model(x, dynamics),
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.control_head.duration_projection.weight.grad is None
    assert model.control_head.control_projection.weight.grad is not None
    assert model.final_time_head.network[-1].bias.grad is not None


@pytest.mark.parametrize(
    "dynamics_backend",
    [
        CONTROL_DYNAMICS_REANCHORED_RK4,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ],
)
@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_control_dataset_and_rollout_loss_form_one_differentiable_training_step(
    dynamics_backend, model_name,
):
    series, config = _series(
        n_flights=2,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=dynamics_backend,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    batch = dataset.batch(np.array([0, 1]))
    x, target, weights, final_time, flight_weights, dynamics = batch
    model = build_model(config)
    prediction = model(x, dynamics)
    loss = prediction_loss(
        prediction,
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(dynamics) == {
        "condition", "initial_state", "initial_controls", "aero_params",
        "control_lower", "control_upper", "max_thrust_n", "frame_params",
        "runway_heading_rad", "glidepath_tan",
    }
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


def test_control_simple_loss_forms_one_real_dynamics_training_step():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics = dataset.batch(
        np.array([0, 1])
    )
    model = build_model(config)

    prediction = model(x, dynamics)
    loss = prediction_loss(
        prediction,
        x[:, -1],
        target,
        weights,
        final_time,
        flight_weights,
        config,
        normalizer,
        dynamics,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert not any(
        "duration_projection" in name for name, _ in model.named_parameters()
    )
    torch.testing.assert_close(
        prediction.segment_durations,
        prediction.final_time_s.unsqueeze(1).expand_as(
            prediction.segment_durations
        ) / config.n_segments,
    )
    assert model.control_head.control_projection.weight.grad is not None
    assert model.final_time_head.network[-1].weight.grad is not None


def test_pipeline_carries_and_names_scaled_transport_chart_dynamics():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=(
            CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
        ),
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert recipe[recipe.index("--control-dynamics-backend") + 1] == (
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
    )
    assert config.control_dynamics_backend == (
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY
    )
    assert "stcv" in plan.train_dir.name.split("_")
    assert "scaled_transport_chart_velocity" in prediction.category
    assert "@scaled-transport-chart-velocity" in prediction.label


def test_transport_chart_prediction_directory_stays_within_component_limit():
    plan = pipeline_module.TrainingPlan(
        (AIRPORT,),
        "itransformer",
        training_mode="pooled",
        prediction_output=PREDICTION_CONTROL,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_duration_gradient=False,
        control_gradient_clip_norm=20.0,
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        coordinate_frame="runway-aligned",
    )
    prediction = pipeline_module.PredictionPlan(
        plan, AIRPORT, ("eval",), split="val"
    )

    assert len(plan.train_dir.name.encode("utf-8")) <= 255
    assert len(prediction.pred_dir.name.encode("utf-8")) <= 255
    assert "scaled_transport_chart_velocity" in prediction.category
    assert "@scaled-transport-chart-velocity" in prediction.label


def test_fixed_dt_control_targets_gather_existing_two_second_reference_rows():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    batch = dataset.batch(np.array([0, 1]))

    assert len(batch) == 7
    dense = batch[-1]
    for row, item in enumerate(series):
        valid = dense.valid[row]
        offsets = dense.query_offsets_s[row, valid].numpy()
        np.testing.assert_allclose(
            offsets,
            np.arange(1, len(offsets) + 1, dtype=np.float64) * config.dt_s,
        )
        query_times = item.times[config.seq_len - 1] + offsets
        source = np.searchsorted(item.supervision_times, query_times)
        expected = normalizer.encode(item.supervision_values[source]).astype(np.float32)
        np.testing.assert_allclose(dense.states[row, valid].numpy(), expected)


def test_fixed_dt_control_targets_choose_nearest_row_across_float_ulp():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    series = SimpleNamespace(
        times=times,
        supervision_times=times,
        supervision_weights=np.ones_like(values),
    )

    dense = build_fixed_dt_supervision(
        [series], [values], [(0, 2)], dt_s=0.1
    )

    assert dense.query_offsets_s[0].tolist() == pytest.approx([0.1, 0.2])
    np.testing.assert_array_equal(dense.states[0].numpy(), values[[3, 4]])
