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
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import channels as ch  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from config import HORIZON_FULL, HORIZON_WINDOW, TSConfig, config_for_mode  # noqa: E402
from dataset import (  # noqa: E402
    Normalizer, TrajectoryWindows, build_series, split_by_flight, window_anchors,
)
from evaluation.metrics import evaluate_batch  # noqa: E402
from evaluation.records import load_records, record_from_dict  # noqa: E402
from export import (  # noqa: E402
    accuracy_block, build_prediction_record, observed_series_metrics, record_stem, write_batch,
)
from forecast import Forecast, forecast_approach, recursive_forecast, truncate_at_threshold  # noqa: E402
from metrics import error_components, trajectory_metrics  # noqa: E402
from models import build_model  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from train import load_checkpoint, masked_mse, train  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"


def _frame() -> ch.Frame:
    return ch.Frame(lat0=35.8745, lon0=-78.8020, alt0=133.0)


def _state(*, lat=35.90, lon=-78.85, alt=900.0, V=90.0, psi=0.5, gamma=-0.05, m=60_000.0):
    return GeodeticState(latitude=lat, longitude=lon, altitude=alt, V=V, psi=psi,
                         gamma=gamma, m=m)


def _series(n_flights=8, mode=HORIZON_WINDOW, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = config_for_mode(mode, **config_overrides)
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

def test_window_mode_anchors_require_a_full_unpadded_horizon():
    series, config = _series(n_flights=2, mode=HORIZON_WINDOW)
    s = series[0]
    anchors = window_anchors(s, config)
    assert anchors.start == config.seq_len - 1
    # Last anchor must leave pred_len real samples after it.
    assert anchors.stop - 1 + config.pred_len <= s.n_samples - 1

    dataset = TrajectoryWindows([s], config, Normalizer.fit([s]))
    _, _, mask = dataset[len(dataset) - 1]
    assert float(mask.sum()) == pytest.approx(config.pred_len)  # nothing padded in window mode


def test_full_mode_pads_the_tail_and_masks_it_out():
    series, config = _series(n_flights=2, mode=HORIZON_FULL)
    s = series[0]
    dataset = TrajectoryWindows([s], config, Normalizer.fit([s]))

    # The very last anchor has exactly one real future sample; the rest is padding.
    x, y, mask = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert y.shape == (config.pred_len, len(config.channels))
    assert mask.sum() == pytest.approx(1.0)
    assert torch.all(y[1:] == 0.0)          # padded rows are zeros...
    assert torch.all(mask[1:] == 0.0)       # ...and the mask says so


def test_masked_mse_ignores_padded_steps():
    # Two horizon steps, one of them padding. A huge error hidden in the padded step must
    # not move the loss at all — otherwise every short approach trains the model to
    # reproduce its own zero padding and forecast tails collapse toward the threshold.
    predicted = torch.tensor([[[1.0], [999.0]]])
    target = torch.tensor([[[0.0], [0.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    assert float(masked_mse(predicted, target, mask)) == pytest.approx(1.0)

    all_valid = torch.tensor([[1.0, 1.0]])
    assert float(masked_mse(predicted, target, all_valid)) > 1.0


def test_fitted_tail_supervises_position_only_and_keeps_observed_inputs_separate():
    config = config_for_mode(HORIZON_FULL, seq_len=3, pred_len=3, dt_s=2.0)
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

    dataset = TrajectoryWindows(series, config, Normalizer.fit(series))
    assert dataset.index[-1] == (0, s.n_samples - 1)
    x, _, weights = dataset[len(dataset) - 1]
    assert x.shape == (config.seq_len, len(config.channels))
    assert weights.sum() == pytest.approx(1.75)
    assert torch.all(weights[:, 3:] == 0.0)


def test_channel_weighted_mse_ignores_fitted_velocity_placeholders():
    predicted = torch.zeros((1, 1, len(ch.CHANNELS)))
    target = torch.tensor([[[1.0, 1.0, 1.0, 999.0, 999.0, 999.0]]])
    weights = torch.tensor([[[1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]]])
    assert float(masked_mse(predicted, target, weights)) == pytest.approx(1.0)


def test_split_by_flight_is_disjoint_and_reproducible():
    series, config = _series(n_flights=10)
    train_a, val_a, test_a = split_by_flight(series, config)
    train_b, val_b, test_b = split_by_flight(series, config)

    ids = lambda group: [s.flight_id for s in group]  # noqa: E731
    assert ids(train_a) == ids(train_b) and ids(val_a) == ids(val_b)  # seeded -> stable

    everything = ids(train_a) + ids(val_a) + ids(test_a)
    assert len(everything) == len(set(everything)) == len(series)  # partition, no overlap


def test_windows_never_straddle_two_flights():
    # The index is (series, anchor) pairs, so a window cannot span flights. Guard it
    # explicitly: concatenating series into one array would be an easy "optimisation" that
    # silently teaches the model to fly from one aircraft's track into another's.
    series, config = _series(n_flights=3)
    dataset = TrajectoryWindows(series, config, Normalizer.fit(series))
    for s_idx, anchor in dataset.index:
        assert anchor - config.seq_len + 1 >= 0
        assert anchor < series[s_idx].n_samples


def _write_arrival_manifest(root: Path, ids: list[str]) -> Path:
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
                "airport": "KRDU",
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
    wanted = {s.flight_id for s in test_group}
    assert len([s for s in series if s.flight_id in wanted]) == len(test_group)


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


# ── Models ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_both_models_map_lookback_to_horizon(model_name):
    config = TSConfig(model=model_name, seq_len=32, pred_len=16, d_model=32, n_heads=4,
                      d_ff=64, e_layers=1)
    model = build_model(config)
    x = torch.randn(3, config.seq_len, config.enc_in)
    assert model(x).shape == (3, config.pred_len, config.enc_in)


def test_config_rejects_a_head_count_that_does_not_divide_d_model():
    with pytest.raises(ValueError, match="n_heads"):
        TSConfig(d_model=100, n_heads=8)


def test_config_for_mode_picks_the_horizon_default_but_yields_to_an_override():
    from config import DEFAULT_PRED_LEN_FULL

    # The bug this guards: constructing TSConfig(horizon_mode="full") directly keeps the
    # WINDOW pred_len default, silently giving a 60-second "whole approach".
    assert config_for_mode(HORIZON_FULL).pred_len == DEFAULT_PRED_LEN_FULL
    assert config_for_mode(HORIZON_FULL, pred_len=99).pred_len == 99


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


# ── Forecast ─────────────────────────────────────────────────────────────────

def test_truncate_cuts_the_forecast_at_closest_approach_to_the_threshold():
    # A forecast that flies past the runway and out the other side must be cut at the
    # runway, not at the horizon's end — the evaluation gates read the FINAL state, so an
    # untruncated forecast would be judged somewhere beyond the airport.
    values = np.zeros((5, len(ch.CHANNELS)))
    values[:, ch.IDX["e"]] = [4000.0, 2000.0, 50.0, -2000.0, -4000.0]
    forecast = Forecast(times=np.arange(5.0), values=values, anchor=0, passes=1,
                        truncated_at_threshold=False)

    cut = truncate_at_threshold(forecast)
    assert cut.n_steps == 3 and cut.truncated_at_threshold
    assert cut.values[-1, ch.IDX["e"]] == pytest.approx(50.0)


def test_truncate_leaves_a_forecast_that_never_reaches_the_runway_alone():
    values = np.zeros((4, len(ch.CHANNELS)))
    values[:, ch.IDX["e"]] = [8000.0, 6000.0, 4000.0, 2000.0]   # still inbound at the end
    forecast = Forecast(times=np.arange(4.0), values=values, anchor=0, passes=1,
                        truncated_at_threshold=False)
    assert truncate_at_threshold(forecast).n_steps == 4


def test_recursive_forecast_chains_enough_passes_to_cover_the_request():
    series, config = _series(n_flights=2, mode=HORIZON_WINDOW, pred_len=10, seq_len=20)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()

    forecast = recursive_forecast(model, series[0], config, normalizer, max_steps=35,
                                  device=torch.device("cpu"))
    assert forecast.n_steps == 35
    assert forecast.passes == 4              # ceil(35 / 10)
    # Times continue the series' own grid, one dt past the anchor.
    assert forecast.times[0] == pytest.approx(series[0].times[forecast.anchor] + config.dt_s)


def test_full_mode_forecasts_in_a_single_pass():
    series, config = _series(n_flights=2, mode=HORIZON_FULL)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecast = forecast_approach(model, series[0], config, normalizer,
                                 device=torch.device("cpu"), truncate=False)
    assert forecast.passes == 1 and forecast.n_steps == config.pred_len


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
    # target, and final_time_s == states[-1]["t"] within 1e-6. Truncation at the threshold
    # makes final_time_s differ from pred_len * dt, which is exactly how that check gets
    # violated if final_time_s is ever derived from the horizon length instead.
    series, config = _series(n_flights=3, mode=HORIZON_FULL)
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


def test_reference_covers_the_same_span_as_the_prediction():
    # evaluation.reference resamples both paths at 101 fractions of THEIR OWN arc length,
    # so a whole-track reference against an anchor-to-threshold prediction compares
    # mismatched segments — it reported kilometres of "path deviation" that were pure span
    # mismatch. Both must start at the anchor.
    from evaluation.reference import compare_to_reference

    series, config = _series(n_flights=3, mode=HORIZON_FULL)
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
    series, config = _series(n_flights=4, mode=HORIZON_FULL)
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
    series, config = _series(n_flights=4, mode=HORIZON_FULL)
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

    summary = json.loads((Path(tmp_path) / "summary.json").read_text(encoding="utf-8"))
    assert summary["accuracy"] == accuracy_block(overlap)
    assert summary["accuracy"]["flights"] == len(series)
    assert summary["accuracy"]["ade_m"]["mean"] > 0.0
    # Per-flight too, so a batch can be re-aggregated (per runway, per capped/uncapped)
    # without re-running the forecast.
    for row, metrics in zip(summary["results"], overlap):
        assert row["ade_m"] == pytest.approx(metrics["ade_m"])
        assert row["overlap_steps"] == metrics["n_steps"]


def test_accuracy_block_excludes_and_counts_flights_with_no_overlap():
    # A forecast that shares no samples with its observed track carries NaN errors. Averaging
    # those in would poison the batch mean; silently dropping them would overstate coverage.
    overlap = [{"ade_m": 100.0, "fde_m": 200.0, "cross_track_p95_m": 50.0,
                "altitude_p95_m": 10.0, "n_steps": 30},
               {"ade_m": float("nan"), "fde_m": float("nan"), "cross_track_p95_m": float("nan"),
                "altitude_p95_m": float("nan"), "n_steps": 0}]
    block = accuracy_block(overlap)
    assert block["flights"] == 1 and block["flights_without_overlap"] == 1
    assert block["ade_m"]["mean"] == pytest.approx(100.0)

    empty = accuracy_block([overlap[1]])
    assert empty["flights"] == 0 and "ade_m" not in empty


def test_write_batch_rejects_overlap_that_does_not_line_up_with_the_records(tmp_path):
    # Positional alignment is the whole contract — a short list would silently zip away the
    # tail of the batch, attributing metrics to the wrong flights.
    series, config = _series(n_flights=3, mode=HORIZON_FULL)
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
    series, config = _series(n_flights=3, mode=HORIZON_FULL)
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

@pytest.mark.parametrize("model_name", ["itransformer", "patchtst"])
def test_train_then_predict_produces_a_gradeable_batch(tmp_path, model_name):
    # Plumbing, not quality: two epochs on synthetic straight-ins. Proves a checkpoint
    # round-trips (config + normalizer + weights) into records `evaluation` can grade.
    series, config = _series(
        n_flights=12, mode=HORIZON_WINDOW, model=model_name, epochs=2, patience=2,
        batch_size=32, d_model=32, n_heads=4, d_ff=64, e_layers=1, seq_len=20, pred_len=10,
        device="cpu",
    )
    summary = train(series, config, output_dir=tmp_path / "run", verbose=False)
    assert summary["epochs_run"] == 2
    assert summary["metrics"]["val"]["ade_m"] > 0.0

    model, loaded_config, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded_config == config          # the config survives the round-trip verbatim
    assert set(payload["split"]) == {"train", "val", "test"}

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
