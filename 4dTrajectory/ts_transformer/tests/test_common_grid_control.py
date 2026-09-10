"""The common-grid replay of a control model: the non-uniform clock, the report, the control statistics.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import ts_transformer.experiments.predictability_report as predictability_report
from ts_transformer.config import PREDICTION_CONTROL, TSConfig
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.backbone.adapters import build_model
from ts_transformer.data.synthetic import synthetic_arrivals

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_common_grid_resampling_uses_explicit_nonuniform_control_clock():
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        seq_len=2,
        n_segments=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        e_layers=1,
    )
    anchor = np.zeros(config.enc_in)
    values = np.zeros((2, config.enc_in))
    values[:, 0] = [1.0, 3.0]

    sampled, capped = predictability_report.resample_prediction(
        anchor,
        values,
        3.0,
        config,
        np.array([1.0, 2.0, 3.0]),
        segment_durations_s=np.array([1.0, 2.0]),
        target_chart=np.zeros(3),
    )

    assert not capped
    np.testing.assert_allclose(sampled[:, 0], [1.0, 2.0, 3.0])


def test_common_grid_report_executes_control_model_with_flight_dynamics():
    series, config = _series(
        n_flights=2,
        prediction_output=PREDICTION_CONTROL,
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
    run = predictability_report.LoadedRun(
        "control", Path("checkpoint.pt"), build_model(config), config, normalizer, {}
    )
    histories = torch.from_numpy(
        predictability_report.history_tensor(series, config, normalizer)
    )

    with torch.no_grad():
        values, final_time, durations, controls, control_durations = (
            predictability_report.predict_batch_nodes(
                run, histories, series, torch.device("cpu")
            )
        )

    assert values.shape == (2, config.validation_common_grid_points, config.enc_in)
    assert durations.shape == (2, config.validation_common_grid_points)
    assert controls is not None and controls.shape == (2, config.n_segments, 3)
    assert control_durations is not None
    assert control_durations.shape == (2, config.n_segments)
    np.testing.assert_allclose(durations.sum(axis=1), final_time, rtol=1e-6)
    np.testing.assert_allclose(control_durations.sum(axis=1), final_time, rtol=1e-6)


def test_control_distribution_statistics_reports_bounds_changes_and_duration_tails():
    controls = np.array([
        [[0.0, -0.5, 0.5], [50.0, 0.0, 1.0], [100.0, 0.5, 2.0]],
        [[0.0, -1.0, 0.5], [100.0, 0.0, 1.0], [200.0, 1.0, 2.0]],
    ])
    durations = np.array([[1.0, 2.0, 3.0], [0.5, 4.0, 8.0]])
    lower = np.array([[0.0, -0.5, 0.5], [0.0, -1.0, 0.5]])
    upper = np.array([[100.0, 0.5, 2.0], [200.0, 1.0, 2.0]])

    result = predictability_report.control_distribution_statistics(
        controls, durations, lower, upper
    )

    thrust = result["channels"]["thrust_N"]
    assert thrust["median"] == pytest.approx(75.0)
    assert thrust["near_lower_fraction"] == pytest.approx(2.0 / 6.0)
    assert thrust["near_upper_fraction"] == pytest.approx(2.0 / 6.0)
    assert thrust["adjacent_abs_change_median"] == pytest.approx(75.0)
    assert result["durations_s"]["min"] == pytest.approx(0.5)
    assert result["durations_s"]["max"] == pytest.approx(8.0)
