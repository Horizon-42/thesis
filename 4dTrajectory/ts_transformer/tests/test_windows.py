"""Windowing and masking: the anchors a window may take, the target grids, the fitted tail, the normalizer.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
import ts_transformer.data.dataset as dataset_module
from ts_transformer.config import HORIZON_FULL, HORIZON_WINDOW, TSConfig
from ts_transformer.data.dataset import (
    FixedAnchorTrajectoryWindows,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    build_series,
    window_anchors,
)
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.training.objective import masked_mse

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def _fitted_tail_flight():
    """A 100 m/s northbound final ending 500 m short of its published threshold."""
    from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

    lat0, lon0, elevation = 35.0, -78.0, 100.0
    cross_m, crossing_height_m = 25.0, 17.5
    waypoints = []
    for along_m in range(-5000, 0, 500):
        time_s = (along_m + 5000) / 100.0
        height = crossing_height_m - math.tan(math.radians(3.0)) * along_m
        waypoints.append([
            time_s,
            lon0 + cross_m / metres_per_deg_lon(lat0),
            lat0 + along_m / METRES_PER_DEG_LAT,
            elevation + height,
        ])
    return {
        "id": "FIT001", "callsign": "FIT001", "type": "A320", "icao24": "abc001",
        "arr_airport": "KFIT", "runway": "36",
        "landing_time_utc": "2026-01-01T00:00:00Z",
        "altitude_source": "opensky_history_geoaltitude_m",
        "runway_target": {
            "lat": lat0, "lon": lon0, "elevation_msl_m": elevation,
            "elevation_hae_m": elevation, "hae_minus_msl_m": 0.0, "course_deg": 0.0,
            "threshold_crossing_height_m": 15.0, "published_glidepath_deg": 3.0,
            "position_source": "faa_cifp_path_point",
            "vertical_source": "faa_cifp_path_point",
        },
        "waypoints": waypoints,
    }


def _post_threshold_flight():
    """The fitted-tail fixture continued 500 m beyond the threshold."""
    from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

    flight = _fitted_tail_flight()
    lat0, lon0, elevation = 35.0, -78.0, 100.0
    cross_m, crossing_height_m = 25.0, 17.5
    along_m = 500.0
    time_s = 55.0
    height = crossing_height_m - math.tan(math.radians(3.0)) * along_m
    flight["waypoints"].append([
        time_s,
        lon0 + cross_m / metres_per_deg_lon(lat0),
        lat0 + along_m / METRES_PER_DEG_LAT,
        elevation + height,
    ])
    return flight


# ── Windowing + masking ──────────────────────────────────────────────────────

def test_normalized_windows_use_every_anchor_with_a_future_remainder():
    series, config = _series(n_flights=2, n_segments=16)
    s = series[0]
    anchors = window_anchors(s, config)
    assert anchors.start == config.seq_len - 1
    assert anchors.stop - 1 == min(s.n_samples - 1, s.n_supervision_samples - 2)

    dataset = RandomAnchorTrajectoryWindows([s], config, Normalizer.fit([s]))
    x, y, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert y.shape == (config.n_segments, len(config.channels))
    assert weights.shape == y.shape
    assert float(final_time_s) > 0.0


def test_normalized_windows_interpolate_the_endpoint_without_padding():
    series, config = _series(n_flights=2, n_segments=12)
    s = series[0]
    normalizer = Normalizer.fit([s])
    dataset = FixedAnchorTrajectoryWindows([s], config, normalizer)

    x, y, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert y.shape == (config.n_segments, len(config.channels))
    assert torch.all(weights > 0.0)
    expected_endpoint = normalizer.encode(s.supervision_values[-1:])[0]
    assert y[-1].numpy() == pytest.approx(expected_endpoint)
    assert float(final_time_s) == pytest.approx(
        s.supervision_times[-1] - s.times[dataset.index[-1][1]]
    )


def test_full_windows_use_physical_dt_and_mask_after_the_endpoint():
    config = TSConfig(
        seq_len=3,
        n_segments=4,
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=4,
        dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    _x, target, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert config.pred_len == 4
    assert target.shape == (4, len(config.channels))
    assert float(final_time_s) == pytest.approx(6.0)
    # Queries are +2, +4, +6 seconds, followed by one padded row.
    assert torch.all(weights[:3, :3].sum(dim=-1) > 0.0)
    assert torch.all(weights[3] == 0.0)
    endpoint = dataset.normalizer.decode(target[2:3].numpy())[0]
    assert endpoint[ch.IDX["e"]] == pytest.approx(25.0, abs=1e-4)


def test_window_mode_requires_a_complete_short_horizon():
    config = TSConfig(
        seq_len=3,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=4,
        dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1

    anchors = window_anchors(series[0], config)
    assert anchors.stop - 1 == series[0].n_supervision_samples - config.pred_len - 1
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    _x, _target, weights, _final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert torch.all(weights.sum(dim=-1) > 0.0)


def test_window_anchor_requires_complete_physical_duration_for_fractional_endpoint():
    config = TSConfig(
        seq_len=3,
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=4,
        dt_s=2.0,
    )
    [source], _ = _series(n_flights=1, seq_len=3)
    observed_times = np.arange(0.0, 12.0, 2.0)
    observed_values = np.zeros((len(observed_times), len(ch.CHANNELS)))
    series = dataset_module.FlightSeries(
        flight_id=source.flight_id,
        scenario=source.scenario,
        frame=source.frame,
        times=observed_times,
        values=observed_values,
        supervision_times=np.append(observed_times, 10.5),
        supervision_values=np.zeros((len(observed_times) + 1, len(ch.CHANNELS))),
        supervision_weights=np.ones(
            (len(observed_times) + 1, len(ch.CHANNELS))
        ) / len(ch.CHANNELS),
    )

    # Counting the four rows after index 2 would admit it, but 10.5 - 4.0 is only
    # 6.5 seconds. Recursive inference always advances 4 * 2 = 8 seconds per pass.
    assert list(window_anchors(series, config)) == []


def test_vectorized_interpolation_matches_the_scalar_reference():
    series, config = _series(n_flights=2, n_segments=17)
    s = series[0]
    normalizer = Normalizer.fit([s])
    dataset = RandomAnchorTrajectoryWindows([s], config, normalizer)
    sample_index = len(dataset) // 2
    _x, target, weights, final_time_s, _flight_weight = dataset[sample_index]
    _series_index, anchor = dataset.index[sample_index]
    query_times = s.times[anchor] + dataset.progress * float(final_time_s)
    encoded = normalizer.encode(s.supervision_values).astype(np.float32)

    expected_target = np.column_stack([
        np.interp(query_times, s.supervision_times, encoded[:, channel])
        for channel in range(len(config.channels))
    ]).astype(np.float32)
    expected_weights = np.column_stack([
        np.interp(query_times, s.supervision_times, s.supervision_weights[:, channel])
        for channel in range(len(config.channels))
    ]).astype(np.float32)

    assert target.numpy() == pytest.approx(expected_target)
    assert weights.numpy() == pytest.approx(expected_weights)


def test_masked_mse_ignores_padded_steps():
    # Two horizon steps, one of them padding. A huge error hidden in the padded step must
    # not move the loss at all — otherwise every short approach trains the model to
    # reproduce its own zero padding and forecast tails collapse toward the threshold.
    predicted = torch.tensor([[[1.0], [999.0]]])
    target = torch.tensor([[[0.0], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    assert float(masked_mse(predicted, target, mask)) == pytest.approx(1.0)

    all_valid = torch.tensor([[[1.0], [1.0]]])
    assert float(masked_mse(predicted, target, all_valid)) > 1.0


def test_fitted_tail_supervises_position_only_and_keeps_observed_inputs_separate():
    config = TSConfig(
        seq_len=3, n_segments=3, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]

    # Observations stop at t=44 (the raw t=45 endpoint is off-grid); fitted labels continue
    # at t=46/48/50, but series.values — the forecast input — remains measured-only.
    assert s.times[-1] == pytest.approx(44.0)
    assert s.supervision_times[-3:] == pytest.approx([46.0, 48.0, 50.0])
    assert s.n_supervision_samples == s.n_samples + 3

    tail_weights = s.supervision_weights[s.n_samples:]
    assert np.all(tail_weights[:, 3:] == 0.0)
    assert tail_weights.sum(axis=1) == pytest.approx([0.25, 0.25, 1.25])
    assert s.supervision_values[-1, ch.IDX["e"]] == pytest.approx(25.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["u"]] == pytest.approx(2.5, abs=1e-6)

    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    assert dataset.index[-1] == (0, s.n_samples - 1)
    x, _, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert weights.sum() == pytest.approx(1.75)
    assert torch.all(weights[:, 3:] == 0.0)
    assert float(final_time_s) == pytest.approx(6.0)


def test_normalized_interpolation_never_supervises_fitted_velocity_placeholders():
    config = TSConfig(
        seq_len=3, n_segments=4, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_fitted_tail_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    _, _, weights, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    first_query_time = s.times[-1] + float(final_time_s) / config.n_segments
    assert s.times[-1] < first_query_time < s.supervision_times[s.n_samples]
    assert torch.all(weights[:, 3:] == 0.0)
    assert torch.all(weights[:, :3].sum(dim=-1) > 0.0)


def test_normalized_target_interpolates_and_stops_at_observed_threshold_crossing():
    config = TSConfig(
        seq_len=3, n_segments=4, dt_s=2.0,
        random_train_anchor_min_future_s=0.0,
    )
    series, report = build_series([_post_threshold_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]

    assert s.times[-1] == pytest.approx(48.0)
    assert s.supervision_times[-1] == pytest.approx(50.0)
    assert s.supervision_values[-1, ch.IDX["e"]] == pytest.approx(25.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)
    # The crossing is a measured row (1/C on all six channels) AND the terminal row (the
    # terminal emphasis on its position channels) — the same contract a fitted crossing gets.
    measured = 1.0 / len(ch.CHANNELS)
    terminal = config.fitted_terminal_position_weight / len(ch.POSITION_IDX)
    assert s.supervision_weights[-1, :3] == pytest.approx(measured + terminal)
    assert s.supervision_weights[-1, 3:] == pytest.approx(measured)

    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    _, target, _, final_time_s, _flight_weight = dataset[len(dataset) - 1]
    assert float(final_time_s) == pytest.approx(2.0)
    endpoint = dataset.normalizer.decode(target[-1:].numpy())[0]
    assert endpoint[ch.IDX["n"]] == pytest.approx(0.0, abs=1e-4)


def test_observed_threshold_crossing_does_not_depend_on_a_fitted_tail(monkeypatch):
    monkeypatch.setattr(dataset_module, "fit_flight_final_approach", lambda _flight: None)
    config = TSConfig(seq_len=3, n_segments=4, dt_s=2.0)
    series, report = build_series([_post_threshold_flight()], config, airport="KFIT")

    assert report.built == 1
    assert series[0].times[-1] == pytest.approx(48.0)
    assert series[0].supervision_times[-1] == pytest.approx(50.0)
    assert series[0].supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)


def test_observed_and_fitted_threshold_crossings_share_the_terminal_weight_contract():
    """Review 2026-09-09 A-3: the crossing used to be supervised under two contracts chosen
    by ADS-B coverage — an observed one flat at 1/C with no terminal emphasis, a fitted one
    with the terminal emphasis on position and nothing on velocity. Now the terminal emphasis
    is a property of the terminal ROW, whichever way it was obtained, and the rest of the row
    keeps its origin's own weights (measured: all six channels; extrapolated: position only).
    """
    config = TSConfig(
        seq_len=3, n_segments=3, dt_s=2.0, random_train_anchor_min_future_s=0.0,
        fitted_tail_position_weight=0.25, fitted_terminal_position_weight=1.0,
    )
    measured = 1.0 / len(ch.CHANNELS)
    n_position = len(ch.POSITION_IDX)

    observed, _ = build_series([_post_threshold_flight()], config, airport="KFIT")
    observed_terminal = observed[0].supervision_weights[-1]
    fitted, _ = build_series([_fitted_tail_flight()], config, airport="KFIT")
    fitted_terminal = fitted[0].supervision_weights[-1]

    # Position: each origin's own weight plus the SAME terminal emphasis.
    assert observed_terminal[:3] - measured == pytest.approx(
        config.fitted_terminal_position_weight / n_position)
    assert fitted_terminal[:3] - config.fitted_tail_position_weight / n_position == pytest.approx(
        config.fitted_terminal_position_weight / n_position)
    # Velocity: real (interpolated between two measured samples) on the observed crossing,
    # a placeholder on the fitted one.
    assert observed_terminal[3:] == pytest.approx(measured)
    assert np.all(fitted_terminal[3:] == 0.0)
    # Every other measured row is untouched by the emphasis.
    assert np.all(observed[0].supervision_weights[:-1] == pytest.approx(measured))

    # No terminal emphasis asked for: the observed crossing is a plain measured row again.
    plain = replace(config, fitted_terminal_position_weight=0.0)
    observed, _ = build_series([_post_threshold_flight()], plain, airport="KFIT")
    assert np.all(observed[0].supervision_weights[-1] == pytest.approx(measured))


def test_normalizer_round_trips_and_survives_a_constant_channel():
    series, _ = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    values = series[0].values
    assert np.allclose(normalizer.decode(normalizer.encode(values)), values)

    # A channel with zero variance must not divide by zero.
    flat = np.zeros((10, len(ch.CHANNELS)))
    constant = Normalizer.fit([type(series[0])(
        flight_id="x", scenario=series[0].scenario, frame=series[0].frame,
        times=np.arange(10.0), values=flat)])
    assert np.all(np.isfinite(constant.encode(flat)))


def test_balanced_normalizer_weights_airports_then_flights_equally():
    series, _ = _series(n_flights=3)
    series[0].scenario.source["arr_airport"] = "KAAA"
    series[0].values[:] = 0.0
    for value, item in zip((10.0, 20.0), series[1:]):
        item.scenario.source["arr_airport"] = "KBBB"
        item.values[:] = value
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    assert np.allclose(normalizer.mean, 7.5)
