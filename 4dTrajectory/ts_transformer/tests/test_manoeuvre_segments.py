"""Segments (`manoeuvre/segments.py`, plan §2.3): the start frame reads the same manoeuvre the
same way wherever it is flown, a flown segment and the observed segment of one path are the
same rows, and the cut from an anchor covers only full segments."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from ts_transformer.config import TSConfig
from ts_transformer.data.channels import CHANNELS, POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FlightSeries, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import Forecast
from ts_transformer.manoeuvre import segments as seg

AIRPORT, RUNWAY = "KRDU", "05L"
DT_S = 2.0


def _turning_polyline(n_rows: int = 91, *, speed: float = 70.0, rate_dps: float = 1.5,
                      descent: float = -3.0, start=(-12000.0, 8000.0, 900.0), course_deg: float = 20.0):
    """A constant-rate turn with a constant descent: a manoeuvre with every channel moving."""
    times = np.arange(n_rows, dtype=np.float64) * DT_S
    course = np.radians(course_deg) + np.radians(rate_dps) * times
    e = start[0] + np.concatenate(([0.0], np.cumsum(speed * np.cos(course[:-1]) * DT_S)))
    n = start[1] + np.concatenate(([0.0], np.cumsum(speed * np.sin(course[:-1]) * DT_S)))
    u = start[2] + descent * times
    values = np.column_stack([e, n, u, speed * np.cos(course), speed * np.sin(course), np.full(n_rows, descent)])
    return times, values


def _rotate_translate(values: np.ndarray, angle_rad: float, shift) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array([[c, -s], [s, c]])
    out = values.copy()
    out[:, :2] = values[:, :2] @ rotation.T + np.asarray(shift[:2])
    out[:, 2] += shift[2]
    out[:, 3:5] = values[:, 3:5] @ rotation.T
    return out


def _bare_series(times, values, *, flight_id="BARE1_05L_abc123_20260101T000000Z") -> FlightSeries:
    return FlightSeries(
        flight_id=flight_id, scenario=SimpleNamespace(target=None), frame=None,
        times=np.asarray(times, dtype=np.float64), values=np.asarray(values, dtype=np.float64),
    )


def _forecast(series: FlightSeries, anchor: int, times, values) -> Forecast:
    """A forecast whose rows are the given polyline after the anchor (the fields a Forecast
    must carry are filled with what they would be for a control rollout)."""
    times = np.asarray(times, dtype=np.float64)
    offsets = times - float(series.times[anchor])
    return Forecast(
        times=times, values=np.asarray(values, dtype=np.float64),
        normalized_progress=offsets / offsets[-1], anchor=anchor,
        final_time_s=float(offsets[-1]), predicted_final_time_s=float(offsets[-1]),
        horizon_mode="normalized", passes=1, truncated_at_threshold=False, horizon_capped=False,
        sample_durations_s=np.diff(np.concatenate(([0.0], offsets))), segment_durations_s=np.array([offsets[-1]]),
    )


# ── the grid ─────────────────────────────────────────────────────────────────

def test_segment_offsets_are_the_whole_steps_and_a_partial_step_is_refused():
    assert seg.segment_offsets_s(60.0, DT_S).tolist() == [2.0 * k for k in range(31)]
    assert seg.segment_row_count(20.0, DT_S) == 11
    assert seg.segment_row_count(120.0, DT_S) == 61
    with pytest.raises(ValueError, match="whole number"):
        seg.segment_offsets_s(45.0, DT_S)
    with pytest.raises(ValueError, match="positive"):
        seg.segment_offsets_s(0.0, DT_S)


# ── the start frame ──────────────────────────────────────────────────────────

def test_the_first_row_of_a_segment_is_the_origin_flying_along_x():
    times, values = _turning_polyline()
    rows = seg.segment_rows(times, values, 10.0, 60.0, DT_S)
    assert rows.shape == (31, len(CHANNELS))
    ground_speed = math.hypot(*values[5, list(VELOCITY_IDX[:2])])
    np.testing.assert_allclose(rows[0], [0.0, 0.0, 0.0, ground_speed, 0.0, values[5, VELOCITY_IDX[2]]], atol=1e-9)


def test_a_manoeuvre_reads_the_same_rows_wherever_and_however_it_is_oriented():
    """Plan §2.3: the same manoeuvre on any runway at any airport reads the same rows — the
    frame removes the chart position and the course, and rotates the velocities with it."""
    times, values = _turning_polyline()
    reference = seg.segment_rows(times, values, 20.0, 60.0, DT_S)
    for angle_deg, shift in ((37.0, (5000.0, -22000.0, 150.0)), (-140.0, (-800.0, 300.0, -40.0)), (180.0, (0.0, 0.0, 0.0))):
        moved = _rotate_translate(values, math.radians(angle_deg), shift)
        np.testing.assert_allclose(seg.segment_rows(times, moved, 20.0, 60.0, DT_S), reference, atol=1e-7)


def test_a_left_turn_reads_as_positive_y_and_a_climb_as_positive_z():
    """y is to the LEFT of the course and z is up: a turn toward +n from an eastbound start
    bends the rows to positive y; a descent reads as negative z."""
    times, values = _turning_polyline(course_deg=0.0, rate_dps=1.5, descent=-3.0)
    rows = seg.segment_rows(times, values, 0.0, 60.0, DT_S)
    assert rows[-1, POSITION_IDX[1]] > 0.0          # left turn
    assert rows[-1, POSITION_IDX[0]] > 0.0          # still mostly ahead
    assert rows[-1, POSITION_IDX[2]] == pytest.approx(-3.0 * 60.0)
    times, values = _turning_polyline(course_deg=0.0, rate_dps=-1.5)
    assert seg.segment_rows(times, values, 0.0, 60.0, DT_S)[-1, POSITION_IDX[1]] < 0.0   # right turn


def test_the_frame_round_trips_exactly():
    times, values = _turning_polyline()
    frame = seg.SegmentFrame.at_row(values[7])
    np.testing.assert_allclose(frame.chart_rows(frame.rows(values)), values, atol=1e-9)
    assert frame.course_rad == pytest.approx(math.atan2(values[7, VELOCITY_IDX[1]], values[7, VELOCITY_IDX[0]]))


def test_a_start_row_without_a_course_is_refused_by_name():
    row = np.zeros(len(CHANNELS))
    with pytest.raises(ValueError, match="ground speed"):
        seg.SegmentFrame.at_row(row)


# ── the polyline ─────────────────────────────────────────────────────────────

def test_a_segment_the_polyline_does_not_cover_is_refused_never_held():
    times, values = _turning_polyline(n_rows=21)   # 40 s of polyline
    with pytest.raises(ValueError, match="span"):
        seg.segment_rows(times, values, 0.0, 60.0, DT_S)
    with pytest.raises(ValueError, match="span"):
        seg.segment_rows(times, values, -2.0, 20.0, DT_S)
    seg.segment_rows(times, values, 20.0, 20.0, DT_S)   # exactly to the end: fine


def test_rows_are_read_on_the_polyline_between_its_samples():
    """The rows are interpolated on the polyline, not snapped to its nearest sample: a start
    between two samples reads the point between them."""
    times, values = _turning_polyline()
    rows = seg.segment_rows(times, values, 11.0, 20.0, DT_S)   # starts half a step in
    chart = seg.interpolate_rows(times, values, np.array([11.0]))[0]
    np.testing.assert_allclose(chart[:3], 0.5 * (values[5, :3] + values[6, :3]))
    np.testing.assert_allclose(seg.SegmentFrame.at_row(chart).origin, chart[:3])
    assert rows.shape == (11, len(CHANNELS))


def test_a_flown_segment_and_the_observed_segment_of_one_path_are_the_same_rows():
    """`flown_segment_rows` and `truth_segment_rows` go through ONE function: a forecast whose
    rows are the truth's own rows after the anchor reads back as the truth's segment, and a
    forecast on a denser clock (the rollout's half-second grid) reads the same rows too."""
    times, values = _turning_polyline()
    series = _bare_series(times, values)
    anchor = 10
    truth = seg.truth_segment_rows(series, float(series.times[anchor]), 60.0, DT_S)
    flown = _forecast(series, anchor, times[anchor + 1 :], values[anchor + 1 :])
    np.testing.assert_allclose(
        seg.flown_segment_rows(series, anchor, flown, float(series.times[anchor]), 60.0, DT_S), truth, atol=1e-9,
    )
    dense_times = np.arange(times[anchor] + 0.5, times[-1] + 1e-9, 0.5)
    dense_values = seg.interpolate_rows(times, values, dense_times)
    dense = _forecast(series, anchor, dense_times, dense_values)
    np.testing.assert_allclose(
        seg.flown_segment_rows(series, anchor, dense, float(series.times[anchor]), 60.0, DT_S), truth, atol=1e-9,
    )
    # a later segment of the flown history starts on the FLOWN row, not on an observed one
    later = seg.flown_segment_rows(series, anchor, flown, float(series.times[anchor]) + 60.0, 60.0, DT_S)
    np.testing.assert_allclose(later, seg.truth_segment_rows(series, float(series.times[anchor]) + 60.0, 60.0, DT_S), atol=1e-9)


def test_a_forecast_from_another_anchor_is_refused():
    times, values = _turning_polyline()
    series = _bare_series(times, values)
    flown = _forecast(series, 12, times[13:], values[13:])
    with pytest.raises(ValueError, match="anchor"):
        seg.flown_segment_rows(series, 10, flown, float(series.times[10]), 60.0, DT_S)


# ── the cut from an anchor ───────────────────────────────────────────────────

def test_the_cut_from_an_anchor_holds_only_full_segments():
    times, values = _turning_polyline(n_rows=91)   # 180 s of truth
    series = _bare_series(times, values)
    starts = seg.segment_start_times(series, 5, 60.0)      # 170 s remain → 2 full segments
    np.testing.assert_allclose(starts, [10.0, 70.0])
    np.testing.assert_allclose(seg.segment_start_times(series, 0, 60.0), [0.0, 60.0, 120.0])
    np.testing.assert_allclose(seg.segment_start_times(series, 0, 90.0), [0.0, 90.0])
    assert seg.segment_start_times(series, 85, 60.0).size == 0
    for start in seg.segment_start_times(series, 5, 60.0):
        assert seg.truth_segment_rows(series, start, 60.0, DT_S).shape == (31, len(CHANNELS))


def test_state_row_is_the_polyline_row_at_that_time():
    times, values = _turning_polyline()
    series = _bare_series(times, values)
    np.testing.assert_allclose(seg.state_row(series, 8.0), values[4])
    np.testing.assert_allclose(seg.state_row(series, 9.0), 0.5 * (values[4] + values[5]))


# ── a real cohort ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cohort() -> list[FlightSeries]:
    config = TSConfig(
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8,
        val_fraction=0.25, test_fraction=0.25,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3), config, airport=AIRPORT
    )
    return series


def test_on_a_built_cohort_the_truth_segment_starts_on_the_anchors_observed_row(cohort):
    for item in cohort:
        anchor = 30
        rows = seg.truth_segment_rows(item, float(item.times[anchor]), 60.0, DT_S)
        assert rows.shape == (31, len(CHANNELS))
        frame = seg.SegmentFrame.at_row(item.values[anchor])
        np.testing.assert_allclose(frame.origin, item.values[anchor, list(POSITION_IDX)])
        np.testing.assert_allclose(seg.state_row(item, float(item.times[anchor])), item.values[anchor], atol=1e-9)
        # the rows are the truth's own rows after the anchor, brought into the start frame
        np.testing.assert_allclose(
            frame.chart_rows(rows)[:, list(POSITION_IDX)],
            item.supervision_values[anchor : anchor + 31, list(POSITION_IDX)], atol=1e-6,
        )
        # every full segment from the fixed anchor exists on the truth
        for start in seg.segment_start_times(item, anchor, 30.0):
            assert seg.truth_segment_rows(item, start, 30.0, DT_S).shape == (16, len(CHANNELS))
