"""The channel contract: the round trip through the chart, the velocity seam, the frames, the conditioning columns.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math

import numpy as np
import pytest
import torch

import ts_transformer.data.channels as ch
import ts_transformer.data.coordinate_frames as frames
import ts_transformer.data.dataset as dataset_module
import ts_transformer.outputs.dynamics.context as supervision_module
from aerodynamic_model.common import GeodeticState
from ts_transformer.config import CONTROL_THRUST_FRACTION, COORDINATE_FRAMES, TSConfig
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.inference.forecast import forecast_approach
from ts_transformer.backbone.adapters import build_model
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.tests.support import fake_data_provenance
from ts_transformer.training.train import load_checkpoint, train

AIRPORT, RUNWAY = "KRDU", "05L"


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

    # Any anchor the pipeline builds has the whole lookback behind it; the anchor-state
    # control inversion differentiates that window, so it needs a real anchor, not 0.
    anchor = supervision_module.ANCHOR_CONTROL_SAMPLES
    enu_dynamics = supervision_module.dynamics_arrays(
        enu_series[0], anchor, parameterization=CONTROL_THRUST_FRACTION
    )
    aligned_dynamics = supervision_module.dynamics_arrays(
        aligned_series[0], anchor, parameterization=CONTROL_THRUST_FRACTION
    )
    runway_heading = enu_series[0].scenario.target.psi
    assert enu_dynamics["frame_params"][3] == pytest.approx(0.0)
    assert aligned_dynamics["frame_params"][3] == pytest.approx(runway_heading)
    assert enu_dynamics["runway_heading_rad"] == pytest.approx(runway_heading)
    assert aligned_dynamics["runway_heading_rad"] == pytest.approx(runway_heading)


def test_airport_enu_frame_is_anchored_at_the_airport_reference_point():
    target = _state(psi=0.73)
    with pytest.raises(ValueError, match="requires an airport reference"):
        frames.frame_for_state(target, "airport-enu")
    reference = frames.AirportReference(
        code="KRDU", lat=35.878659, lon=-78.7873, elevation_msl_m=132.59
    )
    frame = frames.frame_for_state(target, "airport-enu", airport_ref=reference)
    assert type(frame) is frames.AirportENUFrame
    assert (frame.lat0, frame.lon0, frame.alt0, frame.code) == (
        reference.lat, reference.lon, reference.elevation_msl_m, "KRDU"
    )
    # The threshold-anchored modes ignore the reference; the airport mode never anchors
    # at a state.
    assert frames.frame_for_state(target, "enu", airport_ref=reference).lat0 == target.latitude
    with pytest.raises(TypeError, match="for_airport"):
        frames.AirportENUFrame.for_state(target)
    assert "airport-enu" in COORDINATE_FRAMES


def test_airport_enu_series_differ_from_threshold_enu_only_by_the_anchor():
    enu_series, _ = _series(n_flights=2, coordinate_frame="enu")
    airport_series, _ = _series(n_flights=2, coordinate_frame="airport-enu")
    for enu, apt in zip(enu_series, airport_series):
        assert type(apt.frame) is frames.AirportENUFrame and apt.frame.code == AIRPORT
        assert np.array_equal(enu.times, apt.times)
        assert np.array_equal(enu.target_chart, np.zeros(3))
        # KRDU 05L's threshold sits kilometres from the reference point: the target is a
        # real point in this chart, not the origin.
        assert np.linalg.norm(apt.target_chart[:2]) > 1_000.0
        # Positions differ by ONE constant vector — the threshold's chart position — up to
        # the east-scale change from cos(lat0) (~1e-4 relative, metres over a 25 km ring).
        offset = apt.values[:, list(ch.POSITION_IDX)] - enu.values[:, list(ch.POSITION_IDX)]
        assert np.allclose(offset, apt.target_chart, atol=5.0)
        assert np.allclose(offset[:, 1:], apt.target_chart[1:], atol=1e-6)
        # Velocities carry the same scale factor and nothing else.
        assert np.allclose(
            apt.values[:, list(ch.VELOCITY_IDX)],
            enu.values[:, list(ch.VELOCITY_IDX)],
            rtol=1e-3, atol=1e-6,
        )
        # The observed threshold crossing is found at the SAME time: the plane goes through
        # the target, not the origin, so the supervision span does not move.
        assert apt.n_samples == enu.n_samples
        assert apt.supervision_times[-1] == pytest.approx(enu.supervision_times[-1], abs=1e-3)
        assert np.allclose(apt.supervision_weights, enu.supervision_weights)
        anchor = supervision_module.ANCHOR_CONTROL_SAMPLES
        dynamics = supervision_module.dynamics_arrays(
            apt, anchor, parameterization=CONTROL_THRUST_FRACTION
        )
        assert dynamics["frame_params"][:3] == pytest.approx(
            [apt.frame.lat0, apt.frame.lon0, apt.frame.alt0]
        )
        assert dynamics["frame_params"][3] == pytest.approx(0.0)
        assert dynamics["runway_heading_rad"] == pytest.approx(apt.scenario.target.psi)


def test_airport_enu_needs_the_arrival_airport():
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=3)
    flights[0]["arr_airport"] = None
    with pytest.raises(ValueError, match="needs the arrival airport"):
        build_series(flights, TSConfig(coordinate_frame="airport-enu"))
    series, report = build_series(
        flights, TSConfig(coordinate_frame="airport-enu"), airport=AIRPORT
    )
    assert report.built == 1 and type(series[0].frame) is frames.AirportENUFrame


def test_target_conditioning_appends_input_only_channels():
    from ts_transformer.data.target_conditioning import CONDITIONING_CHANNELS

    config = TSConfig(target_conditioning="channels")
    assert config.input_channels == ch.CHANNELS + CONDITIONING_CHANNELS
    assert config.enc_in == len(ch.CHANNELS) + len(CONDITIONING_CHANNELS)
    assert config.channels == ch.CHANNELS  # the OUTPUT contract does not grow
    assert TSConfig().input_channels == ch.CHANNELS
    with pytest.raises(ValueError, match="requires the itransformer backbone"):
        TSConfig(model="patchtst", target_conditioning="channels")
    with pytest.raises(ValueError, match="unknown target_conditioning"):
        TSConfig(target_conditioning="tokens")


def test_conditioned_windows_carry_the_target_and_the_model_still_predicts_six_channels():
    from ts_transformer.inference.forecast import history_at_anchor

    series, config = _series(
        n_flights=3, seq_len=20, n_segments=4, coordinate_frame="airport-enu",
        target_conditioning="channels", d_model=16, n_heads=4, d_ff=32, e_layers=1,
    )
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    extra = len(config.input_channels) - len(ch.CHANNELS)
    x, y, _weights, _final_time_s, _flight_weight = (t[0] for t in windows.batch([0])[:5])
    assert x.shape == (config.seq_len, len(ch.CHANNELS) + extra)
    assert y.shape == (config.pred_len, len(ch.CHANNELS))
    # Constant over the history, and exactly the flight's normalized target + course.
    row = windows.conditioning[0]
    assert np.allclose(x[:, len(ch.CHANNELS):].numpy(), row[None, :])
    position = list(ch.POSITION_IDX)
    assert np.allclose(
        row[:3],
        (series[0].target_chart - normalizer.mean[position]) / normalizer.std[position],
    )
    psi = series[0].scenario.target.psi
    assert row[3:] == pytest.approx([math.cos(psi), math.sin(psi)])
    batch = windows.batch(np.arange(len(windows)))
    assert batch[0].shape == (len(windows), config.seq_len, len(ch.CHANNELS) + extra)
    prediction = build_model(config)(batch[0])
    assert prediction.states.shape == (len(windows), config.pred_len, len(ch.CHANNELS))
    assert prediction.final_time_s.shape == (len(windows),)
    # Inference builds the SAME augmented history the training windows carried.
    history = history_at_anchor(series[0], config, normalizer, config.seq_len - 1)
    assert np.allclose(history, x.numpy())


def test_conditioning_is_one_constant_row_under_a_threshold_frame():
    series, config = _series(n_flights=2, target_conditioning="channels")
    normalizer = Normalizer.fit(series)
    anchor = config.seq_len - 1
    rows = [
        dataset_module.series_conditioning(s, config, normalizer, anchor=anchor)
        for s in series
    ]
    position = list(ch.POSITION_IDX)
    for row in rows:
        assert np.allclose(row[:3], -normalizer.mean[position] / normalizer.std[position])
    # Same runway, threshold at the origin: the rows are identical — the mechanism carries
    # no per-flight information here, which is what makes it a free control under arm A.
    assert np.allclose(rows[0], rows[1])
    assert dataset_module.series_conditioning(
        series[0], TSConfig(), normalizer, anchor=anchor
    ) is None


def test_conditioned_checkpoint_round_trips_and_refuses_a_different_input_contract(tmp_path):
    series, config = _series(
        n_flights=12, epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4,
        d_ff=32, e_layers=1, seq_len=20, n_segments=8, device="cpu",
        coordinate_frame="airport-enu", target_conditioning="channels",
    )
    train(
        series, config, output_dir=tmp_path,
        data_provenance=fake_data_provenance(), verbose=False,
    )
    model, loaded, normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert loaded.target_conditioning == "channels"
    assert payload["input_channels"] == list(config.input_channels)
    forecast = forecast_approach(
        model, series[0], loaded, normalizer, device=torch.device("cpu")
    )
    assert forecast.values.shape[1] == len(ch.CHANNELS)
    payload["input_channels"] = list(ch.CHANNELS)
    torch.save(payload, tmp_path / "stale.pt")
    with pytest.raises(ValueError, match="input channel contract"):
        load_checkpoint(tmp_path / "stale.pt")


def test_anchor_relative_state_output_starts_where_the_aircraft_is():
    from ts_transformer.outputs.state.model import StateOutputLayer

    config = TSConfig(state_position_reference="anchor-relative", seq_len=4, n_segments=3)

    class Zero(torch.nn.Module):
        def forward(self, history):
            return torch.zeros(len(history), config.pred_len, len(ch.CHANNELS))

    history = torch.randn(2, config.seq_len, len(ch.CHANNELS))
    states = StateOutputLayer(Zero(), config)(history).states
    position = list(ch.POSITION_IDX)
    # A zero network output means "stay at the anchor": every predicted row's position is
    # the history's last observed position; velocities pass through untouched.
    assert torch.allclose(
        states[:, :, position],
        history[:, -1:, position].expand(-1, config.pred_len, -1),
    )
    assert torch.all(states[:, :, list(ch.VELOCITY_IDX)] == 0.0)
    # ...and with conditioning columns appended, only the state part of the anchor is used.
    conditioned = TSConfig(
        state_position_reference="anchor-relative", target_conditioning="channels",
        seq_len=4, n_segments=3,
    )
    augmented = torch.cat([history, torch.ones(2, config.seq_len, 5)], dim=2)
    assert torch.allclose(
        StateOutputLayer(Zero(), conditioned)(augmented).states[:, :, position],
        history[:, -1:, position].expand(-1, config.pred_len, -1),
    )
    # The default is the absolute contract: the raw output is the prediction.
    plain = StateOutputLayer(Zero(), TSConfig(seq_len=4, n_segments=3))(history).states
    assert torch.all(plain == 0.0)
    with pytest.raises(ValueError, match="unknown state_position_reference"):
        TSConfig(state_position_reference="origin")


def test_anchor_relative_checkpoint_round_trips(tmp_path):
    series, config = _series(
        n_flights=12, epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4,
        d_ff=32, e_layers=1, seq_len=20, n_segments=8, device="cpu",
        state_position_reference="anchor-relative",
    )
    train(
        series, config, output_dir=tmp_path,
        data_provenance=fake_data_provenance(), verbose=False,
    )
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert loaded.state_position_reference == "anchor-relative"
    forecast = forecast_approach(
        model, series[0], loaded, normalizer, device=torch.device("cpu")
    )
    assert forecast.values.shape[1] == len(ch.CHANNELS)


def test_target_chart_is_exactly_the_origin_under_threshold_anchored_frames():
    # Every consumer that used to say "the origin" now measures from target_chart. Under the
    # threshold-anchored frames that must be EXACTLY (0, 0, 0) — not approximately — so the
    # refactor is bit-identical for them (x - 0.0 is x).
    for frame_name in ("enu", "runway-aligned"):
        series, _ = _series(n_flights=1, coordinate_frame=frame_name)
        assert np.array_equal(series[0].target_chart, np.zeros(3)), frame_name
    values = np.zeros((2, len(ch.CHANNELS)))
    values[1, ch.IDX["e"]], values[1, ch.IDX["n"]] = 3.0, 4.0
    assert ch.horizontal_distance_m(values, np.zeros(3)) == pytest.approx([0.0, 5.0])
    # ...and from a displaced target it is the distance to THAT point, not to the origin.
    assert ch.horizontal_distance_m(values, np.array([3.0, 4.0, 0.0])) == pytest.approx([5.0, 0.0])


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
