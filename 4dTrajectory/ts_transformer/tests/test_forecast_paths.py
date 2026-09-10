"""The forecast paths: batched control rollouts, fixed-time postprocessors, the late anchor.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""


import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
import ts_transformer.outputs.control.dynamics.rollout as control_rollout_module
from ts_transformer.outputs import ForecastOptions
from ts_transformer.config import (
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.inference.forecast import forecast_approach, forecast_approaches
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.state.model import StatePrediction
from ts_transformer.data.synthetic import synthetic_arrivals

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


# ── Forecast ─────────────────────────────────────────────────────────────────

def test_batched_control_forecasts_match_independent_dense_rollouts(monkeypatch):
    series, config = _series(
        n_flights=4,
        prediction_output=PREDICTION_CONTROL,
        n_segments=2,
        seq_len=20,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dense_batch_sizes: list[int] = []
    original_dense_rollout = control_rollout_module.rollout_control_dense

    def capture_dense_batch(controls, *args, **kwargs):
        dense_batch_sizes.append(len(controls))
        return original_dense_rollout(controls, *args, **kwargs)

    monkeypatch.setattr(
        control_rollout_module, "rollout_control_dense", capture_dense_batch
    )

    class HeterogeneousControlModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, history, dynamics):
            self.batch_sizes.append(len(history))
            total = 2.0 + torch.sigmoid(history[:, -1, ch.IDX["e"]])
            durations = total[:, None] * history.new_tensor([[0.375, 0.625]])
            controls = 0.5 * (
                dynamics["control_lower"] + dynamics["control_upper"]
            )
            return ControlPrediction(
                controls=controls[:, None, :].expand(-1, 2, -1).contiguous(),
                segment_durations=durations,
                final_time_s=total,
            )

    batched_model = HeterogeneousControlModel()
    batched = forecast_approaches(
        batched_model,
        series,
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    assert batched_model.batch_sizes == [1] * len(series)
    assert dense_batch_sizes == [len(series)]

    independent_model = HeterogeneousControlModel()
    independent = [
        forecast_approach(
            independent_model,
            item,
            config,
            normalizer,
            device=torch.device("cpu"),
        )
        for item in series
    ]
    assert independent_model.batch_sizes == [1] * len(series)
    assert dense_batch_sizes == [len(series)] + [1] * len(series)

    for actual, expected in zip(batched, independent, strict=True):
        assert actual.anchor == expected.anchor
        assert actual.final_time_s == pytest.approx(expected.final_time_s, abs=1e-12)
        assert actual.predicted_final_time_s == pytest.approx(
            expected.predicted_final_time_s, abs=1e-12
        )
        np.testing.assert_allclose(actual.times, expected.times, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            actual.geodetic_values,
            expected.geodetic_values,
            rtol=1e-12,
            atol=1e-10,
        )
        np.testing.assert_allclose(actual.controls, expected.controls, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual.sample_durations_s,
            expected.sample_durations_s,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.segment_durations_s,
            expected.segment_durations_s,
            rtol=0.0,
            atol=0.0,
        )


def test_forecast_uses_n_normalized_points_and_the_predicted_final_time():
    series, config = _series(n_flights=2, n_segments=10, seq_len=20)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))

    anchor_time = series[0].times[forecast.anchor]
    assert forecast.n_steps == config.n_segments
    assert forecast.normalized_progress == pytest.approx(np.arange(1, 11) / 10)
    assert forecast.times[-1] - anchor_time == pytest.approx(forecast.final_time_s)
    assert np.diff(np.concatenate([[anchor_time], forecast.times])) == pytest.approx(
        np.full(config.n_segments, forecast.final_time_s / config.n_segments)
    )


def test_full_forecast_uses_one_fixed_dt_pass_and_threshold_truncation():
    series, config = _series(
        n_flights=2,
        seq_len=20,
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=5,
    )
    normalizer = _identity_normalizer()

    class FixedPrediction(torch.nn.Module):
        def forward(self, history):
            batch = len(history)
            states = torch.zeros(
                batch, config.pred_len, len(config.channels), device=history.device
            )
            states[:, :, ch.IDX["e"]] = torch.tensor(
                [5.0, 2.0, 0.0, 3.0, 6.0], device=history.device
            )
            return StatePrediction(
                states=states,
                final_time_s=torch.full((batch,), 5.0, device=history.device),
            )

    forecast = forecast_approach(
        FixedPrediction(), series[0], config, normalizer, device=torch.device("cpu")
    )
    anchor_time = series[0].times[forecast.anchor]

    assert forecast.horizon_mode == HORIZON_FULL
    assert forecast.n_steps == 3
    assert forecast.times - anchor_time == pytest.approx([2.0, 4.0, 6.0])
    assert forecast.final_time_s == pytest.approx(6.0)
    assert forecast.passes == 1
    assert forecast.truncated_at_threshold
    assert not forecast.horizon_capped


def test_window_forecast_recurses_to_the_full_horizon():
    series, config = _series(
        n_flights=2,
        seq_len=20,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=2,
        full_horizon_steps=6,
    )
    normalizer = Normalizer.fit(series)

    class FixedPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, history):
            self.calls += 1
            states = torch.zeros(
                len(history), config.pred_len, len(config.channels), device=history.device
            )
            states[:, :, ch.IDX["e"]] = 100.0 - self.calls
            return StatePrediction(
                states=states,
                final_time_s=torch.full((len(history),), 12.0, device=history.device),
            )

    model = FixedPrediction()
    forecast = forecast_approach(
        model, series[0], config, normalizer, device=torch.device("cpu"),
        options=ForecastOptions(truncate=False),
    )

    assert forecast.horizon_mode == HORIZON_WINDOW
    assert forecast.n_steps == config.full_horizon_steps
    assert forecast.passes == 3
    assert model.calls == 3


def test_config_keeps_three_horizon_output_lengths_separate():
    normalized = TSConfig(model="itransformer")
    full = TSConfig(
        model="itransformer",
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=300,
    )
    window = TSConfig(
        model="itransformer",
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=30,
    )

    assert normalized.horizon_mode == HORIZON_NORMALIZED
    assert normalized.pred_len == normalized.n_segments == 16
    assert full.pred_len == 300
    assert window.pred_len == 30
