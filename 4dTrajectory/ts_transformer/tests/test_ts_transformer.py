"""The ts_transformer contracts: channel round-trip, windowing/masking, and the export seam.

What is deliberately NOT covered: whether the models predict *well*. Training quality needs
real harvested tracks and a GPU budget, and asserting a loss threshold against synthetic
straight-line approaches would only test that a transformer can extend a straight line.
The tests here pin the things that break silently instead — the heading convention, the
padding mask, the split, and whether an exported record actually satisfies the validator in
``evaluation.records``.

The end-to-end test trains for two epochs on synthetic data. That is a plumbing check
(does a checkpoint round-trip into a gradeable batch), not a quality check.
"""

import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_test", _TS_DIR / "__main__.py"
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

import channels as ch  # noqa: E402
import batching  # noqa: E402
import coordinate_frames as frames  # noqa: E402
import cross_validation as cv  # noqa: E402
import dataset as dataset_module  # noqa: E402
import detect_ts_best_batch as batch_probe  # noqa: E402
import evaluation_protocol  # noqa: E402
import experiment_index  # noqa: E402
import run_ts_history_ablation as history_ablation  # noqa: E402
import run_ts_pipeline as pipeline_module  # noqa: E402
import run_ts_predictability_report as predictability_report  # noqa: E402
import train as train_module  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from batching import resolve_batch_size  # noqa: E402
from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT, HORIZON_FULL, HORIZON_NORMALIZED, HORIZON_WINDOW,
    CONTROL_STATE_CLOCK_OBSERVED, PREDICTION_CONTROL, TSConfig,
)
from dataset import (  # noqa: E402
    ARRIVAL_DATA_PROVENANCE_SCHEMA, FixedAnchorTrajectoryWindows, FlightEpochSampler,
    Normalizer, RandomAnchorTrajectoryWindows, arrival_data_provenance, build_series,
    cross_validation_folds, require_matching_data_provenance, split_by_flight,
    split_name_for_dataset_id, window_anchors,
)
from evaluation.metrics import evaluate_batch  # noqa: E402
from evaluation.records import load_records, record_from_dict  # noqa: E402
from export import (  # noqa: E402
    accuracy_block, build_prediction_record, observed_series_metrics, record_stem, write_batch,
)
from forecast import Forecast, forecast_approach  # noqa: E402
from metrics import (  # noqa: E402
    RAW_KINEMATIC_METRIC_KEYS, error_components, raw_kinematic_metrics,
    trajectory_metrics,
)
from models import build_model  # noqa: E402
from prediction_outputs import (  # noqa: E402
    ControlBounds, ControlOutputHead, ControlPrediction, StatePrediction,
)
from synthetic import synthetic_arrivals  # noqa: E402
from train import (  # noqa: E402
    FIT_EVALUATION_NAME, FIT_EVALUATION_SCHEMA, evaluate_fit_splits,
    load_checkpoint, masked_mse, position_velocity_consistency_loss,
    prediction_loss, train,
)

AIRPORT, RUNWAY = "KRDU", "05L"


def _fake_data_provenance(airport: str = AIRPORT):
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": airport,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


def test_test_release_is_checkpoint_bound_and_one_shot_per_flight(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1", "KRDU:flight-2"]},
    }

    release = evaluation_protocol.create_test_release(
        checkpoint, payload, provenance
    )
    claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-1"],
        output_dir=tmp_path / "prediction",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, claim)

    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "partially_evaluated"
    assert recorded["claims"][0]["status"] == "complete"
    with pytest.raises(evaluation_protocol.TestReleaseError, match="already exposed"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "repeat",
        )

    second_claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-2"],
        output_dir=tmp_path / "second",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, second_claim)
    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "complete"

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(evaluation_protocol.TestReleaseError, match="checkpoint"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-2"],
            output_dir=tmp_path / "changed",
        )


def test_test_evaluation_requires_a_frozen_release(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="freeze-test"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "prediction",
        )


def test_legacy_checkpoint_cannot_be_released_as_a_fresh_blind_test(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"legacy checkpoint")
    provenance = _fake_data_provenance()
    payload = {
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="predates"):
        evaluation_protocol.create_test_release(checkpoint, payload, provenance)


def test_predict_cli_refuses_test_without_explicit_release(tmp_path):
    with pytest.raises(SystemExit):
        ts_cli.main([
            "predict",
            "--checkpoint", str(tmp_path / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "prediction"),
            "--split", "test",
        ])


def _frame() -> frames.ENUFrame:
    return frames.ENUFrame(lat0=35.8745, lon0=-78.8020, alt0=133.0)


def _state(*, lat=35.90, lon=-78.85, alt=900.0, V=90.0, psi=0.5, gamma=-0.05, m=60_000.0):
    return GeodeticState(latitude=lat, longitude=lon, altitude=alt, V=V, psi=psi,
                         gamma=gamma, m=m)


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_openap_direct_filter_rejects_synonym_before_scenario_fallback():
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3)
    flights[1] = {**flights[1], "type": "A306"}
    config = TSConfig(aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT)

    series, report = build_series(flights, config, airport=AIRPORT)

    assert [item.scenario.aircraft.code for item in series] == ["A320"]
    assert series[0].scenario.source["dynamics_source"].startswith("openap-")
    assert series[0].scenario.source["dynamics_surrogate_typecode"] == "A320"
    assert report.selected_typecodes == {"A320": 1}
    assert report.skipped["aircraft filter rejected"] == 1
    assert report.rejected_aircraft == {"OpenAP synonym model (A306)": 1}


def _identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


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


# ── Channel contract ─────────────────────────────────────────────────────────

def test_channels_round_trip_reproduces_the_original_states():
    # The forward map goes through ground speed (V*cos gamma) and the inverse recomposes
    # V from three components; the two must compose to the identity or every exported
    # trajectory carries a systematic speed/angle bias that no metric would attribute here.
    frame = _frame()
    samples = [
        (0.0, _state(psi=0.5, gamma=-0.05)),
        (2.0, _state(lat=35.91, psi=-2.9, gamma=0.02, V=110.0)),
        (4.0, _state(lat=35.92, psi=3.0, gamma=-0.10, V=70.0)),
    ]
    times, values = ch.channels_from_states(samples, frame)
    recovered = ch.states_from_channels(times, values, frame, mass_kg=60_000.0)

    for (t_in, s_in), (t_out, s_out) in zip(samples, recovered):
        assert t_out == pytest.approx(t_in)
        assert s_out.latitude == pytest.approx(s_in.latitude, abs=1e-9)
        assert s_out.longitude == pytest.approx(s_in.longitude, abs=1e-9)
        assert s_out.altitude == pytest.approx(s_in.altitude, abs=1e-9)
        assert s_out.V == pytest.approx(s_in.V, rel=1e-12)
        assert s_out.psi == pytest.approx(s_in.psi, abs=1e-12)
        assert s_out.gamma == pytest.approx(s_in.gamma, abs=1e-12)


def test_heading_uses_math_enu_not_compass_bearing():
    # psi = 0 must mean due EAST (math-ENU), not due north. Getting this backwards is a
    # reflection about the 45-degree line: it still produces plausible-looking tracks and
    # plausible-looking metrics, and it silently reads an aligned aircraft as a 90-degree
    # intercept. Assert the velocity channels directly, both directions. The 0.5%
    # tolerance absorbs the transport factor (a chart derivative is the physical
    # component ± ~0.3%); a swapped convention is a 100-vs-0 error, not a 0.5% one.
    frame = _frame()
    eastbound = [(0.0, _state(psi=0.0, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(eastbound, frame)
    assert values[0, ch.IDX["edot"]] == pytest.approx(100.0, rel=5e-3)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(0.0, abs=1e-9)

    northbound = [(0.0, _state(psi=math.pi / 2, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(northbound, frame)
    assert values[0, ch.IDX["edot"]] == pytest.approx(0.0, abs=1e-9)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(100.0, rel=5e-3)

    # ...and the inverse agrees EXACTLY: pure-north velocity reads back as psi = +90 deg
    # (the transport factors cancel through the round trip).
    back = ch.states_from_channels(np.array([0.0]), values, frame, mass_kg=1.0)
    assert back[0][1].psi == pytest.approx(math.pi / 2)


def test_transport_factors_are_pinned_at_the_closed_form():
    # The full-transport Jacobian from a physical ENU velocity to this chart's
    # derivatives: f_n = A/(R_M+h), f_e = A·cos(lat0)/((R_N+h)·cos(lat)). Pinned
    # against an independent evaluation of the closed form at lat=35.90, h=900 m with
    # the _frame() anchor — a regression in either radius, either cosine, or the h
    # term moves these in the 4th–6th decimal.
    frame = _frame()
    f_e, f_n = frame.chart_velocity_factors(35.90, 900.0)
    assert f_e == pytest.approx(0.9990293554373, abs=1e-10)
    assert f_n == pytest.approx(1.0031236001934, abs=1e-10)

    # Wiring: the factors land on the right axes with the state's own position.
    northbound = [(0.0, _state(psi=math.pi / 2, gamma=0.0, V=100.0))]
    _, values = ch.channels_from_states(northbound, frame)
    assert values[0, ch.IDX["ndot"]] == pytest.approx(100.0 * f_n, rel=1e-12)


def test_integrating_the_velocity_channels_reproduces_the_position_channels():
    # THE transport-consistency property (2026-07-20 finding A7): for a state sequence
    # whose (V, psi, gamma) is the true physical velocity of its own positions, the
    # velocity channels are the exact time derivatives of the position channels.
    # Generate such a sequence with explicit-Euler steps of the geodetic position
    # kinematics (lat_dot = V_north/(R_M+h) etc. — the optimizer's full-transport RHS)
    # at constant physical velocity. The chart coordinates are linear in (lat, lon,
    # alt), so each Euler step's chart displacement is EXACTLY dt times the chart
    # derivative at the step start — a forward difference against edot/ndot/udot at the
    # step's left endpoint is an identity up to float rounding. Before the fix the
    # velocity channels held the raw physical components, off by the transport factors
    # (~0.3%, i.e. ~0.3 m/s here); the tolerance is ~3000x tighter than that regression.
    from geokit import wgs84_curvature_radii

    frame = _frame()
    V, psi, gamma = 100.0, 0.9, -0.05
    ground = V * math.cos(gamma)
    v_east, v_north, v_up = ground * math.cos(psi), ground * math.sin(psi), V * math.sin(gamma)

    lat, lon, alt = 35.95, -78.90, 1200.0
    dt = 0.05
    samples = []
    for k in range(3):
        samples.append((k * dt, _state(lat=lat, lon=lon, alt=alt, V=V, psi=psi, gamma=gamma)))
        r_m, r_n = wgs84_curvature_radii(lat)
        lat_rate = math.degrees(v_north / (r_m + alt))
        lon_rate = math.degrees(v_east / ((r_n + alt) * math.cos(math.radians(lat))))
        lat, lon, alt = lat + lat_rate * dt, lon + lon_rate * dt, alt + v_up * dt

    times, values = ch.channels_from_states(samples, frame)
    for step in (0, 1):
        for position, derivative in (("e", "edot"), ("n", "ndot"), ("u", "udot")):
            forward = (values[step + 1, ch.IDX[position]] - values[step, ch.IDX[position]]) / dt
            assert forward == pytest.approx(values[step, ch.IDX[derivative]], rel=1e-6), (
                f"d({position})/dt != {derivative} at step {step}"
            )


def test_channels_place_the_frame_origin_at_the_threshold():
    # u is height above the THRESHOLD, not above the ellipsoid — so a state sitting exactly
    # at the frame anchor has all-zero position channels.
    frame = _frame()
    at_origin = [(0.0, _state(lat=frame.lat0, lon=frame.lon0, alt=frame.alt0))]
    _, values = ch.channels_from_states(at_origin, frame)
    assert values[0, ch.IDX["e"]] == pytest.approx(0.0)
    assert values[0, ch.IDX["n"]] == pytest.approx(0.0)
    assert values[0, ch.IDX["u"]] == pytest.approx(0.0)


def test_runway_aligned_frame_rotates_and_round_trips_horizontal_channels():
    target = _state(psi=0.73)
    frame = frames.frame_for_state(target, "runway-aligned")
    assert type(frame) is frames.RunwayAlignedFrame
    times, values = ch.channels_from_states([(0.0, target)], frame)
    assert values[0, ch.IDX["edot"]] > 0.0
    assert abs(values[0, ch.IDX["ndot"]]) < values[0, ch.IDX["edot"]] * 0.01
    restored = ch.states_from_channels(times, values, frame, mass_kg=target.m)[0][1]
    assert restored.latitude == pytest.approx(target.latitude)
    assert restored.longitude == pytest.approx(target.longitude)
    assert restored.psi == pytest.approx(target.psi)


def test_coordinate_frame_setting_selects_a_concrete_implementation():
    target = _state(psi=0.73)

    assert type(frames.frame_for_state(target, "enu")) is frames.ENUFrame
    assert type(frames.frame_for_state(target, "runway-aligned")) is frames.RunwayAlignedFrame
    assert not hasattr(frames.frame_for_state(target, "enu"), "coordinate_mode")
    with pytest.raises(ValueError, match="unknown coordinate frame"):
        frames.frame_for_state(target, "other")

    enu_series, _ = _series(n_flights=1, coordinate_frame="enu")
    aligned_series, _ = _series(n_flights=1, coordinate_frame="runway-aligned")
    assert type(enu_series[0].frame) is frames.ENUFrame
    assert type(aligned_series[0].frame) is frames.RunwayAlignedFrame


def test_resample_lands_on_a_regular_grid_without_extrapolating():
    times = np.array([0.0, 1.0, 3.0, 7.5])
    values = np.tile(np.arange(len(times), dtype=float)[:, None], (1, len(ch.CHANNELS)))
    grid, resampled = ch.resample_uniform(times, values, 2.0)

    assert np.allclose(np.diff(grid), 2.0)
    assert grid[0] == 0.0
    # Never past the last real sample: 7.5s of track on a 2s grid stops at 6.0.
    assert grid[-1] == pytest.approx(6.0)
    assert len(resampled) == len(grid)


def test_resample_rejects_a_track_shorter_than_one_step():
    with pytest.raises(ValueError, match="too short"):
        ch.resample_uniform(np.array([0.0, 0.5]), np.zeros((2, len(ch.CHANNELS))), 4.0)


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
    config = TSConfig(seq_len=3, n_segments=3, dt_s=2.0)
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
    config = TSConfig(seq_len=3, n_segments=4, dt_s=2.0)
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
    config = TSConfig(seq_len=3, n_segments=4, dt_s=2.0)
    series, report = build_series([_post_threshold_flight()], config, airport="KFIT")
    assert report.built == 1
    s = series[0]

    assert s.times[-1] == pytest.approx(48.0)
    assert s.supervision_times[-1] == pytest.approx(50.0)
    assert s.supervision_values[-1, ch.IDX["e"]] == pytest.approx(25.0, abs=1e-6)
    assert s.supervision_values[-1, ch.IDX["n"]] == pytest.approx(0.0, abs=1e-6)
    assert np.all(s.supervision_weights[-1] == pytest.approx(1.0 / len(ch.CHANNELS)))

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


def test_channel_weighted_mse_ignores_fitted_velocity_placeholders():
    predicted = torch.zeros((1, 1, len(ch.CHANNELS)))
    target = torch.tensor([[[1.0, 1.0, 1.0, 999.0, 999.0, 999.0]]])
    weights = torch.tensor([[[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]]])
    assert float(masked_mse(predicted, target, weights)) == pytest.approx(1.0)


def test_prediction_loss_adds_scaled_final_time_error():
    config = TSConfig(
        final_time_scale_s=600.0,
        final_time_loss_weight=2.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    states = torch.zeros((1, 2, len(ch.CHANNELS)))
    prediction = StatePrediction(states=states, final_time_s=torch.tensor([900.0]))
    loss = prediction_loss(
        prediction,
        torch.zeros((1, len(ch.CHANNELS))),
        states,
        torch.ones_like(states),
        torch.tensor([600.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )
    assert float(loss) == pytest.approx(0.5)


def test_prediction_loss_applies_per_flight_airport_weights():
    config = TSConfig(
        final_time_loss_weight=0.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=0.0,
    )
    target = torch.zeros((2, 1, len(ch.CHANNELS)))
    prediction = StatePrediction(
        states=torch.stack((torch.ones_like(target[0]), 3.0 * torch.ones_like(target[0]))),
        final_time_s=torch.zeros(2),
    )
    loss = prediction_loss(
        prediction,
        torch.zeros((2, len(ch.CHANNELS))),
        target,
        torch.ones_like(target),
        torch.zeros(2),
        torch.tensor([1.5, 0.5]),
        config,
        _identity_normalizer(),
    )

    assert float(loss) == pytest.approx((1.0 * 1.5 + 9.0 * 0.5) / 2.0)


def test_position_velocity_consistency_loss_is_zero_for_integrated_motion():
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    anchor[0, ch.IDX["edot"]] = 1.0
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    states[0, :, ch.IDX["e"]] = torch.tensor([1.0, 2.0, 3.0])
    states[0, :, ch.IDX["edot"]] = 1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([3.0]),
        _identity_normalizer(),
    )

    assert loss.tolist() == pytest.approx([0.0])


def test_position_velocity_consistency_loss_uses_physical_channel_scales():
    normalizer = Normalizer(
        mean=np.array([10.0, 20.0, 30.0, 5.0, 6.0, 7.0]),
        std=np.array([2.0, 3.0, 4.0, 8.0, 9.0, 10.0]),
    )
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    # One normalized e increment is 2 m. With dt=0.25 s it equals the decoded
    # edot=8 m/s represented by (8 - mean 5) / std 8.
    states[0, :, ch.IDX["e"]] = torch.tensor([0.0, 1.0, 2.0])
    states[0, :, ch.IDX["edot"]] = (8.0 - 5.0) / 8.0
    states[0, :, ch.IDX["ndot"]] = (0.0 - 6.0) / 9.0
    states[0, :, ch.IDX["udot"]] = (0.0 - 7.0) / 10.0
    anchor = states[:, 0].clone()
    anchor[0, ch.IDX["e"]] = -1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([0.75]),
        normalizer,
    )

    assert loss.tolist() == pytest.approx([0.0], abs=1e-12)


def test_position_velocity_consistency_normalizes_displacement_by_position_scale():
    normalizer = Normalizer(
        mean=np.zeros(len(ch.CHANNELS)),
        std=np.array([100.0, 200.0, 300.0, 2.0, 4.0, 5.0]),
    )
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    states = torch.zeros((1, 1, len(ch.CHANNELS)))
    states[0, 0, ch.IDX["e"]] = 0.1  # 10 m displacement, zero predicted velocity.

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([1.0]),
        normalizer,
    )

    assert loss.tolist() == pytest.approx([(10.0 / 100.0) ** 2 / 3.0])


def test_full_kinematic_loss_uses_dt_and_short_final_segment():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        dt_s=2.0,
    )
    normalizer = _identity_normalizer()
    anchor = torch.zeros((1, len(ch.CHANNELS)))
    states = torch.zeros((1, 3, len(ch.CHANNELS)))
    states[0, :, ch.IDX["e"]] = torch.tensor([2.0, 3.0, 999.0])
    states[0, :2, ch.IDX["edot"]] = 1.0
    anchor[0, ch.IDX["edot"]] = 1.0
    weights = torch.zeros_like(states)
    weights[0, :2, list(ch.POSITION_IDX)] = 1.0

    loss = position_velocity_consistency_loss(
        anchor,
        states,
        torch.tensor([3.0]),
        normalizer,
        config=config,
        state_weights=weights,
    )

    assert loss.tolist() == pytest.approx([0.0], abs=1e-12)


def test_prediction_loss_adds_explicit_terminal_position_error():
    config = TSConfig(
        final_time_loss_weight=0.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=3.0,
    )
    target = torch.zeros((1, 2, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, -1, ch.IDX["e"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([2.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        torch.zeros_like(target),
        torch.tensor([2.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    assert float(loss) == pytest.approx(1.0)


def test_full_terminal_loss_uses_last_unpadded_target():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=3,
        final_time_loss_weight=0.0,
        kinematic_consistency_loss_weight=0.0,
        terminal_loss_weight=3.0,
    )
    target = torch.zeros((1, 3, len(ch.CHANNELS)))
    predicted = target.clone()
    predicted[0, 1, ch.IDX["e"]] = 1.0
    predicted[0, 2, ch.IDX["e"]] = 1000.0
    weights = torch.zeros_like(target)
    # Keep the rows identifiable as valid without also charging the e error to state MSE;
    # this isolates the explicit terminal component in the total loss.
    weights[0, :2, ch.IDX["n"]] = 1.0

    loss = prediction_loss(
        StatePrediction(states=predicted, final_time_s=torch.tensor([3.0])),
        torch.zeros((1, len(ch.CHANNELS))),
        target,
        weights,
        torch.tensor([3.0]),
        torch.ones(1),
        config,
        _identity_normalizer(),
    )

    assert float(loss) == pytest.approx(1.0)


def test_split_by_flight_is_disjoint_and_reproducible():
    series, config = _series(n_flights=10)
    train_a, val_a, test_a = split_by_flight(series, config)
    train_b, val_b, test_b = split_by_flight(series, config)

    ids = lambda group: [s.flight_id for s in group]  # noqa: E731
    assert ids(train_a) == ids(train_b) and ids(val_a) == ids(val_b)  # seeded -> stable

    everything = ids(train_a) + ids(val_a) + ids(test_a)
    assert len(everything) == len(set(everything)) == len(series)  # partition, no overlap
    for split_name, group in (("train", train_a), ("val", val_a), ("test", test_a)):
        assert all(split_name_for_dataset_id(item.dataset_id, config) == split_name
                   for item in group)


def test_split_seed_locks_outer_split_across_training_seeds():
    series, _config = _series(n_flights=100)
    first = TSConfig(seed=1337, split_seed=1337)
    repeated = TSConfig(seed=2027, split_seed=1337)

    def split_ids(config):
        return tuple(
            tuple(item.dataset_id for item in group)
            for group in split_by_flight(series, config)
        )

    assert split_ids(first) == split_ids(repeated)


def test_pooled_prediction_filters_checkpoint_split_to_current_airport_subset():
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": "KAAA"}],
    }
    assert ts_cli.split_keys_for_current_data(
        ["KAAA:flight-a", "KBBB:flight-b", "KAAA:flight-c"], provenance
    ) == ["KAAA:flight-a", "KAAA:flight-c"]


def test_windows_never_straddle_two_flights():
    # The index is (series, anchor) pairs, so a window cannot span flights. Guard it
    # explicitly: concatenating series into one array would be an easy "optimisation" that
    # silently teaches the model to fly from one aircraft's track into another's.
    series, config = _series(n_flights=3)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    for s_idx, anchor in dataset.index:
        assert anchor - config.seq_len + 1 >= 0
        assert anchor < series[s_idx].n_samples


def _write_arrival_manifest(root: Path, ids: list[str], *, airport: str = "KRDU") -> Path:
    from dataset import flight_key

    arrivals = root / "arrivals"
    tracks = root / "tracks"
    records = tracks / "assigned" / "05L"
    records.mkdir(parents=True)
    roster = []
    source_roster = []
    runway_target = {
        "lat": 35.8, "lon": -78.8, "elevation_hae_m": 100.0,
        "elevation_msl_m": 133.5, "hae_minus_msl_m": -33.5,
        "course_deg": 50.0, "threshold_crossing_height_m": 15.0,
        "published_glidepath_deg": 3.0,
        "position_source": "faa_cifp_path_point",
        "vertical_source": "faa_cifp_path_point",
    }
    for index, ident in enumerate(ids):
        flight = {
            "flight_key": None,
            "callsign": ident,
            "icao24": f"abc{index:03d}",
            "runway": "05L",
            "landing_time_utc": f"2026-01-01T00:00:{index:02d}Z",
            "altitude_source": "opensky_history_geoaltitude_m",
            "samples": [[0.0, -78.8, 35.8, 500.0]],
        }
        key = flight_key(
            {
                "id": ident,
                "runway": "05L",
                "icao24": flight["icao24"],
                "landing_time_utc": flight["landing_time_utc"],
            },
            index,
        )
        flight["flight_key"] = key
        relative = f"assigned/05L/{key}.json"
        source_text = json.dumps(flight)
        (tracks / relative).write_text(source_text, encoding="utf-8")
        source_roster.append({"flight_key": key, "file": relative})
        roster.append({
            "flight_key": key,
            "source_file": relative,
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "first_sample_index": 0,
            "last_sample_index": 0,
            "runway": "05L",
            "arrival_truncated": False,
            "cut_samples": 0,
            "arrival_duration_s": 0.0,
            "entry_time_utc": flight["landing_time_utc"],
        })
    (tracks / "manifest.json").write_text(
        json.dumps({"records": source_roster}), encoding="utf-8"
    )
    manifest = arrivals / "manifest.json"
    arrivals.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "harvest-arrivals-v3-track-slices",
                "airport": airport,
                "source_manifest": "../tracks/manifest.json",
                "altitude_source": "opensky_history_geoaltitude_m",
                "runway_targets": {"05L": runway_target},
                "records": roster,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_ts_load_uses_only_the_arrival_manifest_roster(tmp_path):
    # An orphan beside the roster is deliberately ignored: no glob can leak rejected or
    # stale flights into the train/validation/test split.
    from dataset import load_flight_dicts

    _write_arrival_manifest(tmp_path, ["A", "B", "C"])
    (tmp_path / "tracks" / "assigned" / "05L" / "orphan.json").write_text(
        json.dumps({"id": "ORPHAN"}), encoding="utf-8"
    )

    assert [f["id"] for f in load_flight_dicts(tmp_path, verbose=False)] == ["A", "B", "C"]


def test_ts_load_aggregates_multiple_airport_manifests(tmp_path):
    from dataset import load_flight_dicts

    first = _write_arrival_manifest(tmp_path / "first", ["A"], airport="KAAA")
    second = _write_arrival_manifest(tmp_path / "second", ["B"], airport="KBBB")
    flights = load_flight_dicts([second, first], verbose=False)
    assert [(flight["arr_airport"], flight["id"]) for flight in flights] == [
        ("KAAA", "A"), ("KBBB", "B")
    ]


def test_ts_load_filters_qualified_keys_before_opening_source_tracks(tmp_path):
    manifest_path = _write_arrival_manifest(tmp_path, ["A", "B"], airport="KRDU")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_key = manifest["records"][0]["flight_key"]
    excluded = manifest["records"][1]
    (tmp_path / "tracks" / excluded["source_file"]).write_text(
        "excluded source track must stay closed", encoding="utf-8"
    )

    flights = dataset_module.load_flight_dicts(
        manifest_path,
        include_flight_keys={f"KRDU:{selected_key}"},
        verbose=False,
    )

    assert [flight["id"] for flight in flights] == ["A"]


def test_manifest_split_keys_are_resolved_without_loading_trajectory_values():
    config = TSConfig(seed=1337)
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": "KRDU",
            "source_records": [
                {"flight_key": f"flight-{index}", "source_sha256": f"{index:064x}"}
                for index in range(20)
            ],
        }],
    }

    resolved = dataset_module.flight_keys_by_split(provenance, config)

    assert set(resolved) == {"train", "val", "test"}
    assert sum(map(len, resolved.values())) == 20
    assert all(
        split_name_for_dataset_id(key, config) == split
        for split, keys in resolved.items()
        for key in keys
    )


def test_batch_probe_opens_outer_train_track_files_only(tmp_path):
    manifest_path = _write_arrival_manifest(
        tmp_path, [f"F{index:02d}" for index in range(40)]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = TSConfig(seed=1337)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in manifest["records"]:
        split = split_name_for_dataset_id(f"KRDU:{row['flight_key']}", config)
        counts[split] += 1
        if split != "train":
            # A filtered loader would fail JSON/SHA validation if it opened this test/val file.
            source_path = tmp_path / "tracks" / row["source_file"]
            source_path.write_text("not opened by the batch probe", encoding="utf-8")

    flights, audit = batch_probe.load_outer_train_flights([manifest_path], config)
    assert len(flights) == counts["train"]
    assert audit["split_counts_from_manifest_rosters"] == counts
    assert audit["loaded_source_tracks"] == {
        "train": counts["train"], "validation": 0, "test": 0,
    }


def test_batch_probe_candidate_grid_and_throughput_selection():
    assert batch_probe.candidate_batch_sizes(32, 256) == [32, 64, 128, 256]
    best = batch_probe.select_best_batch([
        {"batch_size": 64, "status": "ok", "median_samples_per_second": 1000.0},
        {"batch_size": 128, "status": "ok", "median_samples_per_second": 1400.0},
        {"batch_size": 256, "status": "oom"},
    ])
    assert best["batch_size"] == 128


def test_batch_benchmark_executes_current_training_loss(monkeypatch):
    series, config = _series(
        n_flights=4,
        device="cpu",
        seq_len=20,
        n_segments=8,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    normalizer = _identity_normalizer()
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)

    result = batch_probe.benchmark_candidate(
        windows,
        config,
        torch.device("cpu"),
        batch_size=2,
        warmup_steps=1,
        measure_steps=1,
        repeats=1,
    )

    assert result["status"] == "ok"


def test_ts_load_rejects_duplicate_manifest_identity(tmp_path):
    from dataset import load_flight_dicts

    manifest_path = _write_arrival_manifest(tmp_path, ["A"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append(dict(manifest["records"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate flight_key"):
        load_flight_dicts(manifest_path, verbose=False)


def test_ts_load_rejects_legacy_json_input(tmp_path):
    from dataset import load_flight_dicts

    legacy = tmp_path / "KRDU_05L_landings.json"
    legacy.write_text(json.dumps([{"id": "OLD"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="arrival manifest"):
        load_flight_dicts(legacy, verbose=False)


def test_flight_identity_separates_the_same_callsign_on_different_runways():
    # Regression: FlightSeries.flight_id was flight["id"] (the callsign). Across a
    # multi-runway harvest that collides, and the collision is invisible until you notice
    # `predict --split test` returning 48 flights for an 18-flight split — because every
    # namesake on every runway matched. It also leaks the split: three copies of one id
    # land in train, val and test at once.
    from dataset import flight_key

    series, _ = _series(n_flights=6)
    ids = [s.flight_id for s in series]
    assert len(ids) == len(set(ids))

    same_callsign_other_runway = flight_key(
        {"id": "AAL1", "runway": "23R", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)
    assert same_callsign_other_runway != flight_key(
        {"id": "AAL1", "runway": "05L", "icao24": "a1b2c3",
         "landing_time_utc": "2026-06-18T21:37:36Z"}, 0)


def test_split_membership_selects_exactly_the_flights_it_names():
    # The consumer of the leak above: predict --split test filters by flight_id, so the
    # filter must return exactly as many series as the split names.
    series, config = _series(n_flights=12)
    train_group, val_group, test_group = split_by_flight(series, config)
    wanted = {s.dataset_id for s in test_group}
    assert len([s for s in series if s.dataset_id in wanted]) == len(test_group)


def test_dataset_identity_qualifies_flight_key_with_airport():
    series, _ = _series(n_flights=1)
    assert series[0].dataset_id == f"{AIRPORT}:{series[0].flight_id}"


def test_cross_validation_folds_are_disjoint_and_cover_outer_train():
    series, config = _series(n_flights=30)
    outer_train, _outer_val, _outer_test = split_by_flight(series, config)
    folds = cross_validation_folds(outer_train, 3, seed=config.seed)
    identities = [item.dataset_id for fold in folds for item in fold]
    assert len(identities) == len(set(identities)) == len(outer_train)
    assert set(identities) == {item.dataset_id for item in outer_train}


def test_fixed_anchor_dataset_keeps_one_window_per_flight():
    series, config = _series(n_flights=4)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    assert len(dataset) == len(series)
    assert all(anchor == config.seq_len - 1 for _series_index, anchor in dataset.index)


def test_vectorized_batch_matches_individual_random_anchor_samples():
    series, config = _series(n_flights=3, n_segments=16)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    indices = np.array([0, len(dataset) // 2, len(dataset) - 1])

    batch = dataset.batch(indices)
    individual = [dataset[int(index)] for index in indices]
    for field, values in enumerate(batch):
        assert torch.equal(values, torch.stack([sample[field] for sample in individual]))


def test_common_anchor_is_independent_of_history_length():
    series, base = _series(n_flights=2)
    normalizer = Normalizer.fit(series)
    common_anchor = 89
    datasets = [
        FixedAnchorTrajectoryWindows(
            series,
            replace(base, seq_len=seq_len),
            normalizer,
            minimum_anchor_index=common_anchor,
        )
        for seq_len in (30, 60, 90)
    ]

    assert all(dataset.index[0] == (0, common_anchor) for dataset in datasets)
    samples = [dataset[0] for dataset in datasets]
    assert [sample[0].shape[0] for sample in samples] == [30, 60, 90]
    assert all(torch.equal(samples[0][1], sample[1]) for sample in samples[1:])
    assert all(float(samples[0][3]) == pytest.approx(float(sample[3])) for sample in samples[1:])


def test_history_ablation_runs_on_common_outer_train_anchors(tmp_path):
    series, config = _series(
        n_flights=20,
        device="cpu",
        seq_len=20,
        n_segments=8,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
        epochs=1,
        patience=1,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }

    result = history_ablation.run_history_ablation(
        series,
        config,
        seq_lens=(10, 20),
        data_provenance=provenance,
        output_dir=tmp_path,
        n_splits=2,
        epochs=1,
        patience=1,
        auto_batch_size=False,
        verbose=False,
    )

    assert result["selected_seq_len"] in (10, 20)
    assert result["common_anchor"]["index"] == 19
    assert result["leakage_guard"]["outer_test_used"] is False
    signatures = [
        history_ablation.anchor_signature(
            split_by_flight(series, replace(config, seq_len=20))[0],
            replace(config, seq_len=seq_len),
            19,
        )
        for seq_len in (10, 20)
    ]
    assert signatures[0] == signatures[1]
    for name in (
        "history_length_ablation.json",
        "best_history_length.json",
        "history_length_candidates.csv",
        "history_length_folds.csv",
        "plots/history_length_cv_loss.png",
        "plots/history_length_metrics.svg",
        "plots/index.md",
    ):
        assert (tmp_path / name).is_file(), name


def test_fixed_epoch_sampler_uses_every_flight_once_and_reshuffles():
    series, config = _series(n_flights=8)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    repeated = list(FlightEpochSampler(dataset, seed=7))
    reshuffled = list(FlightEpochSampler(dataset, seed=8))

    assert first == repeated
    assert first != reshuffled
    assert len(first) == len(series)
    assert {dataset.index[index][0] for index in first} == set(range(len(series)))


def test_random_epoch_sampler_selects_one_valid_anchor_per_flight():
    series, config = _series(n_flights=8)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    second = list(FlightEpochSampler(dataset, seed=8))

    for epoch in (first, second):
        assert len(epoch) == len(series)
        assert {dataset.index[index][0] for index in epoch} == set(range(len(series)))
    assert {dataset.index[index] for index in first} != {
        dataset.index[index] for index in second
    }


def test_flight_loss_weights_give_every_airport_equal_epoch_weight():
    series, config = _series(n_flights=8)
    series[0].scenario.source["arr_airport"] = "KAAA"
    for item in series[1:]:
        item.scenario.source["arr_airport"] = "KBBB"
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    totals = {"KAAA": 0.0, "KBBB": 0.0}
    for index in FlightEpochSampler(dataset, seed=7):
        series_index, _anchor = dataset.index[index]
        totals[series[series_index].airport] += float(dataset[index][4])

    assert totals["KAAA"] == pytest.approx(totals["KBBB"])
    assert sum(totals.values()) == pytest.approx(len(series))


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


def test_control_state_supervision_clock_rejects_unknown_value():
    with pytest.raises(ValueError, match="control_state_supervision_clock"):
        TSConfig(control_state_supervision_clock="future")


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

    supervised = train_module.control_state_supervision_prediction(
        prediction, torch.tensor([8.0, 10.0]), config
    )

    assert supervised.controls is prediction.controls
    torch.testing.assert_close(
        supervised.segment_durations,
        torch.tensor([[2.0, 6.0], [6.0, 4.0]]),
    )
    torch.testing.assert_close(supervised.final_time_s, torch.tensor([8.0, 10.0]))
    torch.testing.assert_close(prediction.final_time_s, torch.tensor([4.0, 5.0]))
    assert train_module.target_contract(config) == (
        "bounded-control-nonuniform-duration-casadi-rollout-observed-clock-aligned-v3"
    )


def test_predicted_control_state_clock_preserves_original_training_behavior():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[1.0, 3.0]]),
        final_time_s=torch.tensor([4.0]),
    )
    config = TSConfig(prediction_output=PREDICTION_CONTROL)

    assert train_module.control_state_supervision_prediction(
        prediction, torch.tensor([8.0]), config
    ) is prediction


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_control_models_use_per_sample_bounds_and_aircraft_condition(model_name):
    config = TSConfig(
        model=model_name,
        prediction_output=PREDICTION_CONTROL,
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
        "condition": torch.rand(2, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
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
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
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
        train_module,
        "control_rollout_channels",
        lambda _prediction, _dynamics, _config: (
            physical_rollout,
            torch.zeros(1, 2, 7, dtype=torch.float64),
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
        "control_lower": torch.tensor([[0.0, -0.7, 0.5]]),
        "control_upper": torch.tensor([[240_000.0, 0.7, 2.0]]),
    }

    components = train_module.prediction_loss_components(
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
    lower = torch.tensor([[0.0, -math.pi / 4.0, 0.5], [0.0, -0.5, 0.5]])
    upper = torch.tensor([[240_000.0, math.pi / 4.0, 2.0], [200_000.0, 0.5, 2.0]])
    dynamics = {
        "condition": torch.randn(2, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
        "control_lower": lower,
        "control_upper": upper,
    }

    prediction = model(history, dynamics)
    unit_controls = (prediction.controls - lower[:, None]) / (upper - lower)[:, None]
    expected_units = torch.tensor([0.2, 0.5, 1.0 / 3.0]).view(1, 1, 3)
    expected_time = math.log(2.0) * config.final_time_scale_s

    torch.testing.assert_close(unit_controls, expected_units.expand_as(unit_controls))
    torch.testing.assert_close(
        prediction.final_time_s,
        torch.full_like(prediction.final_time_s, expected_time),
    )
    torch.testing.assert_close(
        prediction.segment_durations,
        prediction.final_time_s[:, None].expand(-1, config.n_segments)
        / config.n_segments,
    )


def test_control_dataset_and_rollout_loss_form_one_differentiable_training_step():
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
        "condition", "initial_state", "aero_params", "control_lower",
        "control_upper", "frame_params",
    }
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.parameters()
    )


def test_config_rejects_a_head_count_that_does_not_divide_d_model():
    with pytest.raises(ValueError, match="n_heads"):
        TSConfig(d_model=100, n_heads=8)


def test_default_config_uses_selected_normalized_output_and_physics_losses():
    config = TSConfig()
    assert config.n_segments == 64
    assert TSConfig(model="patchtst").n_segments == 256
    assert config.epochs == 180
    assert config.patience == 20
    assert config.lr_plateau_patience == 3
    assert config.lr_plateau_factor == 0.5
    assert config.kinematic_consistency_loss_weight == 3.0
    assert config.terminal_loss_weight == 0.02


@pytest.mark.parametrize(
    "field",
    ["kinematic_consistency_loss_weight", "terminal_loss_weight"],
)
def test_negative_physics_loss_weights_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: -0.1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq_len", 0),
        ("n_segments", 0),
        ("dt_s", 0.0),
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0.0),
        ("final_time_scale_s", 0.0),
        ("patience", 0),
        ("d_model", 0),
        ("n_heads", 0),
        ("e_layers", 0),
    ],
)
def test_non_positive_training_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        TSConfig(**{field: value})


def test_vendor_pred_len_contract_tracks_n_segments():
    assert TSConfig(n_segments=99).pred_len == 99


def test_arrival_data_provenance_binds_manifest_and_per_flight_sources(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "airport": "KRDU",
        "records": [
            {"flight_key": "B", "source_sha256": "b" * 64},
            {"flight_key": "A", "source_sha256": "a" * 64},
        ],
    }), encoding="utf-8")

    provenance = arrival_data_provenance(manifest)
    entry = provenance["manifests"][0]
    assert provenance["schema_version"] == ARRIVAL_DATA_PROVENANCE_SCHEMA
    assert entry["airport"] == "KRDU"
    assert entry["arrival_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert entry["source_records"] == [
        {"flight_key": "A", "source_sha256": "a" * 64},
        {"flight_key": "B", "source_sha256": "b" * 64},
    ]
    require_matching_data_provenance({"data_provenance": provenance}, provenance)
    with pytest.raises(ValueError, match="does not match"):
        changed = json.loads(json.dumps(provenance))
        changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
        require_matching_data_provenance(
            {"data_provenance": provenance},
            changed,
        )


def test_prediction_provenance_accepts_only_exact_training_airport_subset():
    stored = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": "KAAA", "arrival_manifest_sha256": "a" * 64, "source_records": []},
            {"airport": "KBBB", "arrival_manifest_sha256": "b" * 64, "source_records": []},
        ],
    }
    subset = {**stored, "manifests": [stored["manifests"][1]]}
    require_matching_data_provenance(
        {"data_provenance": stored}, subset, allow_subset=True
    )
    changed = json.loads(json.dumps(subset))
    changed["manifests"][0]["arrival_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="subset"):
        require_matching_data_provenance(
            {"data_provenance": stored}, changed, allow_subset=True
        )


def test_auto_batch_uses_config_default_without_cuda():
    config = TSConfig(batch_size=37)
    assert resolve_batch_size(
        config, torch.device("cpu"), auto=True, verbose=False
    ) == 37


def test_auto_batch_selects_2048_when_2048_probe_succeeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(), torch.device("cuda"), auto=True, verbose=False
    ) == 2048


def test_control_auto_batch_probe_uses_heterogeneous_duration_partitions():
    batch_size, n_segments = 8, 8
    final_time = torch.full((batch_size,), 80.0)
    prediction = ControlPrediction(
        controls=torch.zeros(batch_size, n_segments, 3),
        segment_durations=torch.full((batch_size, n_segments), 10.0),
        final_time_s=final_time,
    )

    probed = batching._heterogeneous_control_probe_prediction(prediction)
    fractions = probed.segment_durations / probed.final_time_s[:, None]
    baseline_steps = int(torch.ceil(
        prediction.segment_durations.max(dim=0).values / 0.5
    ).sum())
    heterogeneous_steps = int(torch.ceil(
        probed.segment_durations.max(dim=0).values / 0.5
    ).sum())

    assert torch.allclose(probed.segment_durations.sum(dim=-1), final_time)
    assert torch.unique(fractions, dim=0).shape[0] == batch_size
    assert heterogeneous_steps >= 2 * baseline_steps


def test_control_auto_batch_retains_one_power_of_two_safety_margin(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(prediction_output=PREDICTION_CONTROL),
        torch.device("cuda"),
        auto=True,
        verbose=False,
    ) == 1024


def test_control_auto_batch_training_probe_applies_heterogeneous_partition(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=4,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    original = batching._heterogeneous_control_probe_prediction
    calls: list[torch.Size] = []

    def tracked(prediction):
        calls.append(prediction.segment_durations.shape)
        return original(prediction)

    monkeypatch.setattr(batching, "_heterogeneous_control_probe_prediction", tracked)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [torch.Size([2, 2])]


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
        control_effort_weight=1e-4,
        control_smoothness_weight=1e-2,
        control_state_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_rollout_dt=0.5,
        output_dir=tmp_path,
    )
    recipe = plan._recipe_args()
    config, _source = plan.resolved_train_config(use_best_config=False)
    prediction = pipeline_module.PredictionPlan(plan, "KMSY", ("eval",), split="val")

    assert recipe[recipe.index("--prediction-output") + 1] == PREDICTION_CONTROL
    assert recipe[recipe.index("--split-seed") + 1] == "1337"
    assert recipe[recipe.index("--control-effort-weight") + 1] == "0.0001"
    assert recipe[recipe.index("--control-smoothness-weight") + 1] == "0.01"
    assert recipe[recipe.index("--control-state-clock") + 1] == "observed"
    assert recipe[recipe.index("--control-rollout-dt") + 1] == "0.5"
    assert recipe[recipe.index("--aircraft-filter") + 1] == "openap-direct"
    assert config.prediction_output == PREDICTION_CONTROL
    assert config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
    assert config.control_effort_loss_weight == pytest.approx(1e-4)
    assert config.control_state_supervision_clock == CONTROL_STATE_CLOCK_OBSERVED
    assert "control" in prediction.pred_dir.name
    assert "observed_clock" in prediction.pred_dir.name
    assert "control" in prediction.category
    assert "openap_direct" in prediction.category


def test_experiment_index_keeps_legacy_incomplete_and_formal_runs_distinct(tmp_path):
    legacy_complete = tmp_path / "legacy" / "run_complete"
    legacy_complete.mkdir(parents=True)
    (legacy_complete / "checkpoint.pt").write_bytes(b"checkpoint")
    (legacy_complete / "history.json").write_text(
        json.dumps({"config": {"prediction_output": "state", "seed": 1337}}),
        encoding="utf-8",
    )
    legacy_incomplete = tmp_path / "legacy" / "run_aborted"
    legacy_incomplete.mkdir(parents=True)
    (legacy_incomplete / "experiment_notes.md").write_text("aborted", encoding="utf-8")
    formal = tmp_path / "openap_direct" / "control"
    formal.mkdir(parents=True)
    (formal / experiment_index.RUN_MANIFEST_NAME).write_text(
        json.dumps({
            "schema_version": experiment_index.RUN_MANIFEST_SCHEMA,
            "campaign_id": "openap-direct-20260729",
            "run_id": "control",
            "status": "completed",
            "config": {
                "prediction_output": "control",
                "aircraft_filter": "openap-direct",
                "seed": 1337,
            },
            "artifacts": {"checkpoint.pt": {}},
        }),
        encoding="utf-8",
    )

    document = experiment_index.rebuild_index(tmp_path)
    by_run = {entry["run_id"]: entry for entry in document["entries"]}

    assert by_run["run_complete"]["status"] == "completed"
    assert by_run["run_aborted"]["status"] == "incomplete"
    assert by_run["control"]["campaign_id"] == "openap-direct-20260729"
    assert by_run["control"]["aircraft_filter"] == "openap-direct"
    assert (tmp_path / "index.json").is_file()
    assert (tmp_path / "INDEX.md").is_file()


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
        values, final_time, durations, controls = predictability_report.predict_batch_nodes(
            run, histories, series, torch.device("cpu")
        )

    assert values.shape == (2, config.n_segments, config.enc_in)
    assert durations.shape == (2, config.n_segments)
    assert controls is not None and controls.shape == (2, config.n_segments, 3)
    np.testing.assert_allclose(durations.sum(axis=1), final_time, rtol=1e-6)


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


def test_auto_batch_probe_executes_the_shared_physics_loss(monkeypatch):
    config = TSConfig(
        device="cpu",
        seq_len=4,
        n_segments=4,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    original_loss = train_module.prediction_loss
    calls: list[tuple[torch.Size, torch.Size]] = []

    def tracked_loss(prediction, anchor, target, *args):
        calls.append((anchor.shape, target.shape))
        return original_loss(prediction, anchor, target, *args)

    monkeypatch.setattr(train_module, "prediction_loss", tracked_loss)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [
        (torch.Size([2, len(ch.CHANNELS)]), torch.Size([2, 4, len(ch.CHANNELS)]))
    ]


def test_cross_validation_never_passes_outer_val_or_test_to_fit(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=36, device="cpu", epochs=1, patience=1,
        d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    outer_train, outer_val, outer_test = split_by_flight(series, config)
    forbidden = {item.dataset_id for item in [*outer_val, *outer_test]}
    observed: list[set[str]] = []

    def fake_fit(train_series, val_series, fold_config, **_kwargs):
        identities = {item.dataset_id for item in [*train_series, *val_series]}
        observed.append(identities)
        score = float(fold_config.learning_rate + fold_config.d_model * 1e-8)
        row = SimpleNamespace(
            epoch=1, val_loss=score, val_by_airport={AIRPORT: score}
        )
        return SimpleNamespace(
            history=[row], best_val_loss=score, config=fold_config,
            model=object(), normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", fake_fit)
    monkeypatch.setattr(
        cv, "evaluate_split",
        lambda *_args, **_kwargs: {
            "ade_m": 1.0,
            "raw_kinematics": {key: 1.0 for key in RAW_KINEMATIC_METRIC_KEYS},
        },
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=3,
        cv_parameters=("learning_rate",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert observed
    assert all(not (identities & forbidden) for identities in observed)
    assert set.union(*observed) == {item.dataset_id for item in outer_train}
    assert result["leakage_guard"]["outer_test_used"] is False
    assert result["schema_version"] == (
        "ts-cross-validation-v10-raw-kinematic-metrics"
    )
    assert result["selection_metric"] == (
        "mean outer-train-fold airport-macro weighted sum of normalized state MSE, "
        "scaled final-time MSE, position/velocity displacement-consistency MSE, and "
        "terminal-position MSE"
    )
    assert (tmp_path / "best_config.json").is_file()


def test_cross_validation_runs_real_two_fold_search(tmp_path):
    series, config = _series(
        n_flights=20, device="cpu", epochs=1, patience=1,
        d_model=16, d_ff=32, n_heads=4, e_layers=1,
        seq_len=20, n_segments=10, batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert len(result["candidates"]) == 2
    assert all(len(candidate["folds"]) == 2 for candidate in result["candidates"])
    assert all(
        "raw_kinematics" in fold["validation_metrics"]
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
    assert result["base_config"]["n_segments"] == config.n_segments
    assert "n_segments" not in result["best_overrides"]
    assert json.loads((tmp_path / "best_config.json").read_text()) == result["best_overrides"]


def test_cross_validation_exhausts_the_default_three_parameter_grid():
    config = TSConfig(n_segments=128)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 27
    assert {
        (candidate["n_segments"], candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (64, 128, 256),
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))
    assert all(candidate["d_ff"] == 2 * candidate["d_model"] for candidate in candidates)


@pytest.mark.parametrize("horizon_mode", [HORIZON_FULL, HORIZON_WINDOW])
def test_fixed_horizon_cv_does_not_repeat_inert_n_segment_candidates(horizon_mode):
    config = TSConfig(horizon_mode=horizon_mode)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 9
    assert all("n_segments" not in candidate for candidate in candidates)
    assert {
        (candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_error_decomposes_into_along_and_cross_track_in_the_truth_frame():
    # Truth heading due east; predicted position offset 100 m north of it. That is a pure
    # CROSS-track error — the aircraft is on schedule but off the path. If the two came out
    # swapped, a lateral containment failure would read as a harmless timing error.
    truth = np.zeros((1, 1, len(ch.CHANNELS)))
    truth[0, 0, ch.IDX["edot"]] = 100.0
    predicted = truth.copy()
    predicted[0, 0, ch.IDX["n"]] = 100.0
    mask = np.ones((1, 1))

    components = error_components(predicted, truth, mask)
    assert components["along"][0] == pytest.approx(0.0, abs=1e-9)
    assert components["cross"][0] == pytest.approx(100.0)   # + = left of the true course

    # ...and an offset straight ahead is pure along-track.
    ahead = truth.copy()
    ahead[0, 0, ch.IDX["e"]] = 100.0
    components = error_components(ahead, truth, mask)
    assert components["along"][0] == pytest.approx(100.0)
    assert components["cross"][0] == pytest.approx(0.0, abs=1e-9)


def test_spread_matches_the_gate_side_signed_spread():
    # metrics._spread is the VECTORISED twin of evaluation/stats.signed_spread (that one
    # is stdlib-only by design and would sort millions of boxed floats here). This seam
    # test is what makes "same statistic" a checked property instead of a mirror comment:
    # if either side changes its percentile method or keys, this fails.
    from evaluation.stats import signed_spread
    from metrics import _spread

    values = np.array([3.0, -1.5, 0.25, -7.0, 4.0, 2.5, -0.75])
    ours, theirs = _spread(values), signed_spread(values.tolist())
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key])


def test_fde_reads_each_sample_own_last_valid_step():
    # Two samples with different valid lengths. FDE must use each one's real endpoint, not
    # a fixed column — otherwise a padded short approach reports the error of its padding.
    truth = np.zeros((2, 3, len(ch.CHANNELS)))
    predicted = np.zeros((2, 3, len(ch.CHANNELS)))
    predicted[0, 1, ch.IDX["e"]] = 10.0   # sample 0 ends at step 1
    predicted[1, 2, ch.IDX["e"]] = 20.0   # sample 1 ends at step 2
    mask = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    block = trajectory_metrics(predicted, truth, mask)
    assert block["fde_m"] == pytest.approx((10.0 + 20.0) / 2)


def test_raw_kinematic_metrics_use_nonuniform_segment_durations():
    # Constant 1 m/s² eastward acceleration sampled at deliberately nonuniform node times.
    # Trapezoidal velocity integration is exact here, acceleration is constant, and jerk is
    # zero. A metric that silently assumes uniform N spacing fails this test.
    node_times = np.array([0.0, 1.0, 3.0, 6.0])
    nodes = np.zeros((1, len(node_times), len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = 0.5 * node_times**2
    nodes[0, :, ch.IDX["edot"]] = node_times

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.diff(node_times)[None, :]
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(0.0, abs=1e-12)
    assert block["acceleration_p95_mps2"] == pytest.approx(1.0)
    assert block["jerk_p95_mps3"] == pytest.approx(0.0, abs=1e-12)


def test_raw_kinematic_metrics_measure_geometric_turn_acceleration_and_jerk():
    # Three one-second position segments turn east -> north -> west. Node velocities are
    # chosen so their trapezoidal midpoint exactly matches each geometric segment velocity;
    # consistency and heading error must therefore be zero while the path still has a
    # 90 deg/s turn rate, sqrt(200) m/s² acceleration and 20 m/s³ jerk.
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0][:, [ch.IDX["e"], ch.IDX["n"]]] = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
    ])
    nodes[0][:, [ch.IDX["edot"], ch.IDX["ndot"]]] = np.array([
        [10.0, 0.0], [10.0, 0.0], [-10.0, 20.0], [-10.0, -20.0],
    ])

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(90.0)
    assert block["acceleration_p95_mps2"] == pytest.approx(math.sqrt(200.0))
    assert block["jerk_p95_mps3"] == pytest.approx(20.0)


def test_raw_kinematic_metrics_detect_velocity_heading_disagreement():
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = np.arange(4) * 10.0
    nodes[0, :, ch.IDX["ndot"]] = 10.0

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(math.sqrt(200.0 / 3.0))
    assert block["heading_consistency_p95_deg"] == pytest.approx(90.0)


# ── Forecast ─────────────────────────────────────────────────────────────────

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
        model, series[0], config, normalizer, device=torch.device("cpu"), truncate=False
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
    assert normalized.pred_len == normalized.n_segments == 64
    assert full.pred_len == 300
    assert window.pred_len == 30


# ── Export seam ──────────────────────────────────────────────────────────────

def test_record_stem_disambiguates_flights_sharing_a_callsign():
    # The same callsign flies the same approach daily; id alone collides and one day's
    # result silently overwrites another's.
    monday = {"id": "AAL1", "runway": "05L", "icao24": "a1b2c3",
              "landing_time_utc": "2026-06-18T21:37:36Z"}
    tuesday = dict(monday, landing_time_utc="2026-06-19T21:31:02Z")
    assert record_stem(monday, 0) != record_stem(tuesday, 0)
    assert record_stem(monday, 0) == "AAL1_05L_a1b2c3_20260618T213736Z"


def test_exported_record_satisfies_the_evaluation_contract():
    # The real validator, not a copy of it: record_from_dict enforces the state keys, the
    # target, and final_time_s == states[-1]["t"] within 1e-6. Here final_time_s is learned,
    # so deriving it from N or dt would violate the record contract.
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))
    record = build_prediction_record(series[0], forecast, index=0,
                                     model_name=config.model, horizon_mode=config.horizon_mode)

    parsed = record_from_dict(record.eval_record)
    assert parsed.solved
    assert parsed.controls == []            # a predictor emits no control schedule
    assert parsed.final_time_s == pytest.approx(parsed.states[-1]["t"], abs=1e-6)
    assert set(parsed.states[0]) == {"t", "lat", "lon", "alt", "V", "psi", "gamma", "m"}

    # t=0 is the anchor, and states[0] IS initial_state — same convention as an optimizer
    # record, so the two are readable side by side.
    assert parsed.states[0]["t"] == pytest.approx(0.0)
    for key, value in parsed.initial_state.items():
        assert parsed.states[0][key] == pytest.approx(value)


def test_normalized_state_export_preserves_a_tiny_relative_clock():
    """Relative offsets must not collapse when added to and subtracted from the anchor."""
    series, config = _series(n_flights=1)
    normalizer = Normalizer.fit(series)
    tiny_final_time_s = 3.7044681016305814e-13

    class TinyDurationStateModel(torch.nn.Module):
        def forward(self, history):
            batch = len(history)
            return StatePrediction(
                states=torch.zeros(
                    (batch, config.pred_len, config.enc_in),
                    dtype=history.dtype,
                    device=history.device,
                ),
                final_time_s=torch.full(
                    (batch,),
                    tiny_final_time_s,
                    dtype=history.dtype,
                    device=history.device,
                ),
            )

    forecast = forecast_approach(
        TinyDurationStateModel(),
        series[0],
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    metrics = observed_series_metrics(series[0], forecast)
    record = build_prediction_record(
        series[0],
        forecast,
        index=0,
        model_name=config.model,
        horizon_mode=config.horizon_mode,
    )
    parsed = record_from_dict(record.eval_record)
    exported_times = np.array([state["t"] for state in parsed.states])

    assert metrics["raw_kinematics"]["predicted"]["segments"] == config.pred_len
    assert np.all(np.diff(exported_times) > 0.0)
    assert parsed.final_time_s == pytest.approx(tiny_final_time_s, rel=1e-6)


def test_control_forecast_exports_optimizer_shaped_states_and_aligned_controls():
    series, config = _series(
        n_flights=1,
        prediction_output=PREDICTION_CONTROL,
        n_segments=2,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)

    class FixedControlModel(torch.nn.Module):
        def forward(self, history, dynamics):
            batch = len(history)
            controls = torch.tensor(
                [[[40_000.0, 0.04, 1.01], [32_000.0, -0.02, 0.99]]],
                dtype=history.dtype,
                device=history.device,
            ).expand(batch, -1, -1)
            durations = torch.tensor(
                [[0.75, 1.25]], dtype=history.dtype, device=history.device
            ).expand(batch, -1)
            return ControlPrediction(
                controls=controls,
                segment_durations=durations,
                final_time_s=durations.sum(dim=-1),
            )

    forecast = forecast_approach(
        FixedControlModel(),
        series[0],
        config,
        normalizer,
        device=torch.device("cpu"),
    )
    record = build_prediction_record(
        series[0], forecast, index=0, model_name=config.model,
        horizon_mode=config.horizon_mode,
    )
    parsed = record_from_dict(record.eval_record)

    assert parsed.solved
    assert len(parsed.states) == len(parsed.controls) == 3
    assert record.source["predictionOutput"] == "control"
    assert [state["t"] for state in parsed.states] == pytest.approx([0.0, 0.75, 2.0])
    assert parsed.controls[0]["thrust"] == pytest.approx(40_000.0)
    assert parsed.controls[-1]["thrust"] == pytest.approx(32_000.0)
    assert parsed.final_time_s == pytest.approx(2.0)


def test_control_training_checkpoint_round_trip_keeps_output_identity(tmp_path):
    series, config = _series(
        n_flights=12,
        prediction_output=PREDICTION_CONTROL,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=2,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
        device="cpu",
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "b" * 64,
            "source_records": [
                {"flight_key": item.flight_id, "source_sha256": f"{index + 100:064x}"}
                for index, item in enumerate(series)
            ],
        }],
    }
    train(series, config, output_dir=tmp_path, data_provenance=provenance, verbose=False)
    model, loaded_config, _normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    metadata = json.loads((tmp_path / "checkpoint_metadata.json").read_text())

    assert loaded_config == config
    assert loaded_config.prediction_output == PREDICTION_CONTROL
    assert isinstance(model(torch.zeros(1, config.seq_len, config.enc_in), {
        "condition": torch.ones(1, len(dataset_module.DYNAMICS_CONDITION_NAMES)),
        "control_lower": torch.tensor([[0.0, -0.5, 0.5]]),
        "control_upper": torch.tensor([[100_000.0, 0.5, 2.0]]),
    }), ControlPrediction)
    assert payload["target_contract"] == (
        "bounded-control-nonuniform-duration-casadi-rollout-clock-aligned-v2"
    )
    assert metadata["prediction_output"] == PREDICTION_CONTROL
    assert metadata["control_recipe"] == {
        "effort_loss_weight": config.control_effort_loss_weight,
        "smoothness_loss_weight": config.control_smoothness_loss_weight,
        "rollout_integrator_dt_s": config.control_rollout_integrator_dt_s,
        "state_supervision_clock": config.control_state_supervision_clock,
    }


def test_reference_covers_the_same_span_as_the_prediction():
    # evaluation.reference resamples both paths at 101 fractions of THEIR OWN arc length,
    # so a whole-track reference against an anchor-to-threshold prediction compares
    # mismatched segments — it reported kilometres of "path deviation" that were pure span
    # mismatch. Both must start at the anchor.
    from evaluation.reference import compare_to_reference

    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"))
    record = build_prediction_record(series[0], forecast, index=0,
                                     model_name=config.model, horizon_mode=config.horizon_mode)

    predicted = record_from_dict(record.eval_record)
    reference = record_from_dict(record.reference_record)
    assert reference.states[0]["t"] == pytest.approx(0.0)
    # Same starting point, to the metre — they are the same observed anchor sample.
    assert reference.states[0]["lat"] == pytest.approx(predicted.states[0]["lat"], abs=1e-9)
    assert reference.states[0]["lon"] == pytest.approx(predicted.states[0]["lon"], abs=1e-9)

    comparison = compare_to_reference(predicted, reference)
    assert comparison.path_lateral_m is not None


def test_batch_writes_a_manifest_that_evaluation_can_load_and_grade(tmp_path):
    series, config = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    records, overlap = [], []
    for index, s in enumerate(series):
        forecast = forecast_approach(model, s, config, normalizer, device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=config.model,
                                               horizon_mode=config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    write_batch(records, output_dir=tmp_path, config_dict=config.to_dict(), overlap=overlap)

    # load_records is manifest-ONLY (no glob fallback), so this also proves summary.json
    # carries a results[] roster with resolvable eval_file entries.
    loaded = load_records(tmp_path)
    assert len(loaded) == len(series)
    report = evaluate_batch(loaded)
    assert report["total"] == len(series) and report["solved"] == len(series)

    # Every record points at a reference that exists, so compare_to_reference works.
    for record in loaded:
        assert record.reference_file is not None
        assert (Path(tmp_path) / record.reference_file).is_file()


def test_manifest_carries_the_accuracy_the_run_printed(tmp_path):
    # The batch's error against the observed tracks is its headline result; it used to exist
    # only in terminal scrollback, which made any cross-batch comparison (the instance-norm
    # ablation) a stdout-scraping exercise.
    series, config = _series(n_flights=4)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    records, overlap = [], []
    for index, s in enumerate(series):
        forecast = forecast_approach(model, s, config, normalizer, device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=config.model,
                                               horizon_mode=config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    write_batch(
        records,
        output_dir=tmp_path,
        config_dict=config.to_dict(),
        overlap=overlap,
        split="val",
    )

    summary = json.loads((Path(tmp_path) / "summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "val"
    assert all(row["split"] == "val" for row in summary["results"])
    states = json.loads(next(Path(tmp_path).glob("*_states.json")).read_text(encoding="utf-8"))
    assert states["source"]["predictionSplit"] == "val"
    assert summary["accuracy"] == accuracy_block(overlap)
    assert summary["accuracy"]["flights"] == len(series)
    assert summary["accuracy"]["ade_m"]["mean"] > 0.0
    assert summary["accuracy"]["final_time_s"]["mae"] >= 0.0
    assert set(summary["accuracy"]["raw_kinematics"]) == {
        "predicted", "observed_baseline", "delta"
    }
    # Per-flight too, so a batch can be re-aggregated (per runway, per capped/uncapped)
    # without re-running the forecast.
    for row, metrics in zip(summary["results"], overlap):
        assert row["ade_m"] == pytest.approx(metrics["ade_m"])
        assert row["overlap_steps"] == metrics["n_steps"]
        assert row["final_time_error_s"] == pytest.approx(metrics["final_time_error_s"])
        assert row["raw_kinematics"] == metrics["raw_kinematics"]


def test_accuracy_block_excludes_and_counts_flights_with_no_overlap():
    # A forecast that shares no samples with its observed track carries NaN errors. Averaging
    # those in would poison the batch mean; silently dropping them would overstate coverage.
    raw = {
        "predicted": {key: 2.0 for key in RAW_KINEMATIC_METRIC_KEYS},
        "observed_baseline": {key: 1.0 for key in RAW_KINEMATIC_METRIC_KEYS},
    }
    overlap = [{"ade_m": 100.0, "fde_m": 200.0, "cross_track_p95_m": 50.0,
                "altitude_p95_m": 10.0, "n_steps": 30,
                "true_final_time_s": 300.0, "final_time_error_s": 20.0,
                "raw_kinematics": raw},
               {"ade_m": float("nan"), "fde_m": float("nan"), "cross_track_p95_m": float("nan"),
                "altitude_p95_m": float("nan"), "n_steps": 0,
                "true_final_time_s": 400.0, "final_time_error_s": -10.0,
                "raw_kinematics": raw}]
    block = accuracy_block(overlap)
    assert block["flights"] == 1 and block["flights_without_overlap"] == 1
    assert block["ade_m"]["mean"] == pytest.approx(100.0)
    assert block["ade_m"]["median"] == pytest.approx(100.0)
    assert block["final_time_s"]["mae"] == pytest.approx(15.0)
    raw_summary = block["raw_kinematics"]
    assert raw_summary["predicted"][RAW_KINEMATIC_METRIC_KEYS[0]]["median"] == 2.0
    assert raw_summary["delta"][RAW_KINEMATIC_METRIC_KEYS[0]] == 1.0

    empty = accuracy_block([overlap[1]])
    assert empty["flights"] == 0 and "ade_m" not in empty


def test_accuracy_block_emits_empty_raw_metric_stats_when_all_values_are_nan():
    raw = {
        "predicted": {key: float("nan") for key in RAW_KINEMATIC_METRIC_KEYS},
        "observed_baseline": {key: float("nan") for key in RAW_KINEMATIC_METRIC_KEYS},
    }
    row = {
        "ade_m": 1.0,
        "fde_m": 2.0,
        "cross_track_p95_m": 1.0,
        "altitude_p95_m": 1.0,
        "n_steps": 1,
        "true_final_time_s": 1.0,
        "final_time_error_s": 0.0,
        "raw_kinematics": raw,
    }

    block = accuracy_block([row])

    for role in ("predicted", "observed_baseline"):
        for key in RAW_KINEMATIC_METRIC_KEYS:
            stats = block["raw_kinematics"][role][key]
            assert stats["count"] == 0
            assert all(math.isnan(stats[name]) for name in ("median", "mean", "p95", "max"))
    assert all(
        math.isnan(value) for value in block["raw_kinematics"]["delta"].values()
    )


def test_write_batch_rejects_overlap_that_does_not_line_up_with_the_records(tmp_path):
    # Positional alignment is the whole contract — a short list would silently zip away the
    # tail of the batch, attributing metrics to the wrong flights.
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    records = [
        build_prediction_record(
            s, forecast_approach(model, s, config, normalizer, device=torch.device("cpu")),
            index=index, model_name=config.model, horizon_mode=config.horizon_mode)
        for index, s in enumerate(series)
    ]
    with pytest.raises(ValueError, match="once per record"):
        write_batch(records, output_dir=tmp_path, config_dict=config.to_dict(), overlap=[])


def test_stale_records_are_cleared_before_a_rerun(tmp_path):
    series, config = _series(n_flights=3)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    def batch(count):
        records, overlap = [], []
        for index, s in enumerate(series[:count]):
            forecast = forecast_approach(model, s, config, normalizer,
                                         device=torch.device("cpu"))
            records.append(build_prediction_record(s, forecast, index=index,
                                                   model_name=config.model,
                                                   horizon_mode=config.horizon_mode))
            overlap.append(observed_series_metrics(s, forecast))
        write_batch(records, output_dir=tmp_path, config_dict=config.to_dict(), overlap=overlap)

    batch(3)
    batch(1)   # a shrinking flight set must not leave orphans behind
    assert len(list(Path(tmp_path).glob("*_eval.json"))) == 1
    assert len(list((Path(tmp_path) / "references").glob("*_reference_eval.json"))) == 1


# ── End to end ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("horizon_mode", "contract", "expected_passes"),
    (
        (HORIZON_FULL, "full-horizon-fixed-time-displacement-kinematic-v2", 1),
        (HORIZON_WINDOW, "recursive-window-fixed-time-displacement-kinematic-v2", 3),
    ),
)
def test_fixed_time_modes_train_checkpoint_and_forecast(
    tmp_path, horizon_mode, contract, expected_passes
):
    series, config = _series(
        n_flights=12,
        model="itransformer",
        horizon_mode=horizon_mode,
        full_horizon_steps=12,
        window_horizon_steps=4,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        device="cpu",
    )
    train(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=_fake_data_provenance(),
        verbose=False,
    )

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    forecast = forecast_approach(
        model,
        series[0],
        loaded_config,
        normalizer,
        device=torch.device("cpu"),
        truncate=False,
    )

    assert payload["target_contract"] == contract
    assert loaded_config.horizon_mode == horizon_mode
    assert forecast.passes == expected_passes
    assert forecast.n_steps == loaded_config.full_horizon_steps


def test_fit_evaluation_is_fixed_anchor_eval_mode_and_repeatable():
    series, config = _series(
        n_flights=12,
        random_train_anchor=True,
        dropout=0.5,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    model = build_model(config).train()

    first = evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )
    second = evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )

    assert not model.training
    assert first == second
    assert first["schema_version"] == FIT_EVALUATION_SCHEMA
    assert first["evaluation_contract"] == {
        "model_mode": "eval",
        "dropout": "disabled",
        "anchor": "fixed L-1",
        "batch_order": "sequential (shuffle disabled)",
        "splits": ["train", "val"],
        "metric_grid": "native target grid; measured-data mask",
    }
    assert first["splits"]["train"]["flights"] == len(train_series)
    assert first["splits"]["val"]["flights"] == len(val_series)
    assert first["diagnostics"]["native_generalization"]["ade_m"]["ratio"] > 0.0


def test_evaluate_fit_cli_runs_train_and_validation_together(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=12,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    provenance = _fake_data_provenance()
    payload = {
        "split": {
            "train": [item.dataset_id for item in train_series],
            "val": [item.dataset_id for item in val_series],
            "test": [item.dataset_id for item in test_series],
        },
        "data_provenance": provenance,
    }
    flights = [{"key": item.dataset_id} for item in (*train_series, *val_series)]
    report = SimpleNamespace(format=lambda: "built synthetic fit replay")

    monkeypatch.setattr(
        ts_cli, "load_checkpoint",
        lambda _path: (build_model(config), config, normalizer, payload),
    )
    monkeypatch.setattr(ts_cli, "arrival_data_provenance", lambda _data: provenance)
    monkeypatch.setattr(ts_cli, "require_matching_data_provenance", lambda *_args: None)
    loaded_keys = None

    def load_selected(_data, *, include_flight_keys=None):
        nonlocal loaded_keys
        loaded_keys = include_flight_keys
        return flights

    monkeypatch.setattr(ts_cli, "load_flight_dicts", load_selected)
    monkeypatch.setattr(ts_cli, "dataset_flight_key", lambda flight, _index: flight["key"])
    monkeypatch.setattr(ts_cli, "build_series", lambda *_args, **_kwargs: (series, report))
    monkeypatch.setattr(ts_cli, "resolve_device", lambda _device: torch.device("cpu"))

    assert ts_cli.main([
        "evaluate-fit",
        "--checkpoint", str(checkpoint),
        "--data", str(tmp_path / "manifest.json"),
        "--output-dir", str(tmp_path / "fit"),
        "--device", "cpu",
    ]) == 0
    replay = json.loads(
        (tmp_path / "fit" / FIT_EVALUATION_NAME).read_text(encoding="utf-8")
    )
    assert set(replay["splits"]) == {"train", "val"}
    assert loaded_keys == set(payload["split"]["train"] + payload["split"]["val"])


def test_train_refuses_to_replace_a_checkpoint_with_a_test_release(tmp_path):
    series, config = _series(
        n_flights=12,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        device="cpu",
    )
    output = tmp_path / "released"
    output.mkdir()
    checkpoint = output / "checkpoint.pt"
    checkpoint.write_bytes(b"released checkpoint")
    (output / evaluation_protocol.TEST_RELEASE_NAME).write_text(
        json.dumps({"schema_version": evaluation_protocol.TEST_RELEASE_SCHEMA}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="test release"):
        train(
            series,
            config,
            output_dir=output,
            data_provenance=_fake_data_provenance(),
            verbose=False,
        )

    assert checkpoint.read_bytes() == b"released checkpoint"


@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_train_then_predict_produces_a_gradeable_batch(tmp_path, model_name):
    # Plumbing, not quality: two epochs on synthetic straight-ins. Proves a checkpoint
    # round-trips (config + normalizer + weights) into records `evaluation` can grade.
    series, config = _series(
        n_flights=12, model=model_name, epochs=2, patience=2,
        batch_size=32, d_model=32, n_heads=4, d_ff=64, e_layers=1, seq_len=20,
        n_segments=10,
        device="cpu",
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [
                {"flight_key": item.flight_id, "source_sha256": f"{index:064x}"}
                for index, item in enumerate(series)
            ],
        }],
    }
    summary = train(
        series,
        config,
        output_dir=tmp_path / "run",
        data_provenance=provenance,
        verbose=False,
    )
    assert summary["epochs_run"] == 2
    assert set(summary["metrics"]) == {"train", "val"}
    assert summary["metrics"]["train"]["ade_m"] > 0.0
    assert summary["metrics"]["val"]["ade_m"] > 0.0
    assert "raw_kinematics" in summary["metrics"]["train"]
    assert "raw_kinematics" in summary["metrics"]["val"]
    fit_evaluation = json.loads(
        (tmp_path / "run" / FIT_EVALUATION_NAME).read_text(encoding="utf-8")
    )
    assert fit_evaluation["schema_version"] == FIT_EVALUATION_SCHEMA
    assert fit_evaluation["checkpoint"]["sha256"] == hashlib.sha256(
        (tmp_path / "run" / "checkpoint.pt").read_bytes()
    ).hexdigest()
    assert set(fit_evaluation["splits"]) == {"train", "val"}

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded_config == config          # the config survives the round-trip verbatim
    assert payload["target_contract"] == (
        "normalized-time-runway-crossing-displacement-kinematic-v3"
    )
    assert payload[evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD] == (
        evaluation_protocol.TEST_RELEASE_SCHEMA
    )
    assert set(payload["split"]) == {"train", "val", "test"}
    assert payload["data_provenance"] == provenance
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    metadata = json.loads(
        (tmp_path / "run" / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "schema_version": "ts-checkpoint-metadata-v14-aircraft-filter-audit",
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "arrival_manifests": {AIRPORT: "a" * 64},
        "random_train_anchor": False,
        "horizon_mode": HORIZON_NORMALIZED,
        "prediction_output": "state",
        "aircraft_filter": "all",
        "pred_len": config.pred_len,
        "full_horizon_steps": config.full_horizon_steps,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "split_sha256": {
            split: hashlib.sha256("\n".join(sorted(payload["split"][split])).encode()).hexdigest()
            for split in ("train", "val", "test")
        },
    }
    assert set(summary["history"][0]["train_components"]) == {
        "state", "final_time", "kinematic", "terminal"
    }
    assert set(summary["history"][0]["val_components"]) == {
        "state", "final_time", "kinematic", "terminal"
    }
    assert summary["history"][0]["learning_rate"] == config.learning_rate
    assert summary["history"][0]["optimizer_updates"] > 0
    objective = fit_evaluation["diagnostics"]["training_objective"]
    assert objective["total_optimizer_updates"] == summary["history"][-1]["optimizer_updates"]
    assert objective["final_learning_rate"] == summary["history"][-1]["learning_rate"]

    records, overlap = [], []
    for index, s in enumerate(series[:4]):
        forecast = forecast_approach(model, s, loaded_config, normalizer,
                                     device=torch.device("cpu"))
        records.append(build_prediction_record(s, forecast, index=index,
                                               model_name=loaded_config.model,
                                               horizon_mode=loaded_config.horizon_mode))
        overlap.append(observed_series_metrics(s, forecast))
    out = tmp_path / "pred"
    write_batch(records, output_dir=out, config_dict=loaded_config.to_dict(), overlap=overlap)

    report = evaluate_batch(load_records(out))
    assert report["total"] == 4
    # The gate outcome is not asserted — an undertrained model on synthetic data may or may
    # not land inside 106.75 m, and pinning that would make this a flaky quality test.
    assert "lateral_m" in report and "success_rate" in report

    history = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    assert history["config"]["model"] == model_name
    assert len(history["history"]) == 2
    assert history["data_provenance"]["source_record_count"] == len(series)
