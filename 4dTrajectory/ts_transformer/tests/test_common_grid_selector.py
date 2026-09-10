"""The formal common-grid checkpoint selector.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""


import numpy as np
import pytest
import torch

import ts_transformer.channels as ch
import ts_transformer.validation as validation
from ts_transformer.config import (
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.fixed_anchor_validation import (
    fixed_anchor_common_grid_ade_metrics,
    fixed_anchor_common_truth,
)
from ts_transformer.models import build_model
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.objective import prediction_loss

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_formal_common_grid_selector_returns_only_lean_position_time_metrics():
    series, config = _series(n_flights=1, seq_len=8, n_segments=2)
    item = series[0]
    anchor = config.seq_len - 1
    duration = float(item.supervision_times[-1] - item.times[anchor])
    offsets = np.array([duration / 2.0, duration])
    query_times = item.times[anchor] + offsets
    predicted = np.column_stack([
        np.interp(query_times, item.supervision_times, item.supervision_values[:, channel])
        for channel in range(len(ch.CHANNELS))
    ])[None, ...]

    metrics = fixed_anchor_common_grid_ade_metrics(
        series,
        config,
        item.values[anchor][None, :],
        predicted,
        np.array([duration]),
        np.diff(np.concatenate(([0.0], offsets)))[None, :],
        points=2,
        anchor=anchor,
    )

    assert metrics["ade_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["fde_m"] == pytest.approx(0.0, abs=1e-7)
    assert "dense_state_loss" not in metrics
    assert not any(key.startswith("arc_length_") for key in metrics)


def test_formal_common_grid_selector_reuses_identical_precomputed_truth():
    series, config = _series(n_flights=3, seq_len=8, n_segments=4)
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    model = build_model(config).eval()
    replay = validation.predict_split(
        model,
        dataset,
        normalizer,
        torch.device("cpu"),
        config.batch_size,
    )
    arguments = (
        series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
    )
    baseline = fixed_anchor_common_grid_ade_metrics(
        *arguments, points=7, anchor=config.seq_len - 1
    )
    cached = fixed_anchor_common_grid_ade_metrics(
        *arguments,
        points=7,
        common_truth=fixed_anchor_common_truth(series, config, 7, anchor=config.seq_len - 1),
    )

    assert baseline.keys() == cached.keys()
    for key in baseline:
        if isinstance(baseline[key], np.ndarray):
            np.testing.assert_array_equal(cached[key], baseline[key])
        else:
            assert cached[key] == baseline[key]


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
@pytest.mark.parametrize("integrator_dt_s", [2.0, 0.3])
def test_fixed_dt_control_loss_forms_one_differentiable_training_step(
    model_name, integrator_dt_s
):
    series, config = _series(
        n_flights=1,
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        patch_len=4,
        stride=2,
        final_time_scale_s=600.0,
        control_rollout_integrator_dt_s=integrator_dt_s,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0])
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
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )
