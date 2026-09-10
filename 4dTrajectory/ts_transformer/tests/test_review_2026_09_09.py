"""The 2026-09-09 package review's regression pins.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math

import numpy as np
import pytest
import torch

import ts_transformer.channels as ch
import ts_transformer.coordinate_frames as frames
import ts_transformer.outputs.control.strategy as control_strategy
from ts_transformer.config import TSConfig
from ts_transformer.dataset import Normalizer, build_series
from ts_transformer.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.forecast import forecast_approach
from ts_transformer.models import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.synthetic import synthetic_arrivals

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


# ── review 2026-09-09 ────────────────────────────────────────────────────────

def test_the_heterogeneous_probe_keeps_every_field_of_the_prediction():
    """Review B-2: the probe rebuilt `ControlPrediction` from three fields, so a quantile
    head's `duration_quantiles_s` arrived at the pinball loss as None."""
    batch_size, n_segments = 4, 8
    quantiles = torch.ones(batch_size, 5)
    prediction = ControlPrediction(
        controls=torch.zeros(batch_size, n_segments, 3),
        segment_durations=torch.full((batch_size, n_segments), 10.0),
        final_time_s=torch.full((batch_size,), 80.0),
        duration_quantiles_s=quantiles,
    )
    probed = control_strategy.heterogeneous_control_probe_prediction(prediction)
    assert probed.duration_quantiles_s is quantiles
    assert not torch.equal(probed.segment_durations, prediction.segment_durations)


def test_write_batch_refuses_a_non_finite_ade_before_writing_any_record(tmp_path):
    """Review B-3: the accuracy block's refusal used to come AFTER every record file was
    written, leaving a record directory with no summary.json."""
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(
        model, series[0], config, normalizer, device=torch.device("cpu")
    )
    record = build_prediction_record(
        series[0], forecast, index=0,
        model_name=config.model, horizon_mode=config.horizon_mode,
    )
    metrics = [observed_series_metrics(series[0], forecast)]
    metrics[0]["ade_m"] = float("nan")
    out = tmp_path / "records"
    with pytest.raises(ValueError, match="finite"):
        write_batch(
            [record], output_dir=out, config_dict=config.to_dict(), flight_metrics=metrics,
        )
    assert not out.exists()


def test_a_vertical_only_velocity_survives_the_channel_round_trip():
    """Review C-20: with zero ground speed the old fallback set gamma to 0, so the state's
    V·sin(gamma) was 0 while the channel said udot."""
    frame = frames.ENUFrame(lat0=35.9, lon0=-78.8, alt0=100.0)
    values = np.zeros((1, len(ch.CHANNELS)))
    values[0, ch.IDX["u"]] = 500.0
    values[0, ch.IDX["udot"]] = -4.0
    (_t, state), = ch.states_from_channels(np.array([0.0]), values, frame, mass_kg=60_000.0)
    assert state.V * math.sin(state.gamma) == pytest.approx(-4.0)
