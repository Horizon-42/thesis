"""Segments (plan §2.3): how an approach is cut into `segment_s` pieces from an anchor, the
segment-start frame every piece is read in, and the rows the tokenizer's encoder sees.

A segment is `segment_s` seconds of a six-channel chart polyline (`data/channels.py`: e, n, u
and their exact chart derivatives) sampled every `dt_s` from its first row, so it has
``segment_s / dt_s + 1`` rows and the first row is the segment's own start. Every segment is
read in its **start frame** (`SegmentFrame`): origin = the first row's position, x-axis = the
first row's ground-track direction, y-axis to the left, z-axis up, the velocities rotated with
it — so the first row is always ``(0, 0, 0, V_ground, 0, udot)`` and the same manoeuvre reads
the same rows on any runway at any airport. The runway heading does not enter (the two-tier
v2 layer rotated its features by the LANDED runway and wrote the T1 future information into
them; plan §2.3).

**One polyline function feeds both sources.** `segment_rows` reads a ``(times, values)``
polyline; `truth_segment_rows` hands it the flight's supervision arrays (the observed rows
plus the fitted tail, what the executor is scored against) and `flown_segment_rows` hands it
a forecast with the anchor's observed row standing in before the first flown row (the way
`inference/receding.py` reads a flown history). A flown segment and the observed segment of
the same path therefore produce the same rows — pinned by `tests/test_manoeuvre_segments.py`.

The fitted tail's velocity columns are the terminal measured state's, held (`dataset.
_build_supervision`); a segment reaching into it reads them as they are. The tail is short
(KRDU: a median 380 m / 6 s), and the rows are an encoder INPUT, never a loss target.

No handcrafted descriptors, no resampling, no residual against a straight-line extrapolation
(plan §2.3): the rows are what the encoder sees, the representation is learned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.data.channels import CHANNELS, POSITION_IDX, VELOCITY_IDX

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries
    from ts_transformer.inference.forecast import Forecast

#: The segment rows' channels, in the start frame, in the chart channels' order.
SEGMENT_CHANNELS: tuple[str, ...] = ("x", "y", "z", "xdot", "ydot", "zdot")
#: A segment must be a whole number of `dt_s` steps; this is the tolerance that is judged at.
STEP_TOLERANCE_S = 1e-6
#: A start row must carry a ground-track direction: below this ground speed (m/s) the course
#: is undefined and the frame is refused by name. An approach flies at 60–80 m/s; only a
#: corrupt row or a stationary synthetic one can bind it.
MINIMUM_GROUND_SPEED_MPS = 1.0


def segment_offsets_s(segment_s: float, dt_s: float) -> np.ndarray:
    """``0, dt_s, 2·dt_s, …, segment_s`` — the segment's row times from its start; refuses a
    segment that is not a whole number of steps (the rows would not land on the grid)."""
    if segment_s <= 0.0 or dt_s <= 0.0:
        raise ValueError(f"segment_s and dt_s are positive seconds, got {segment_s!r}, {dt_s!r}")
    steps = segment_s / dt_s
    if abs(steps - round(steps)) > STEP_TOLERANCE_S:
        raise ValueError(
            f"segment_s={segment_s:g} is not a whole number of dt_s={dt_s:g} steps"
        )
    return np.arange(int(round(steps)) + 1, dtype=np.float64) * dt_s


def segment_row_count(segment_s: float, dt_s: float) -> int:
    """How many rows a segment has (``segment_s / dt_s + 1``)."""
    return len(segment_offsets_s(segment_s, dt_s))


@dataclass(frozen=True)
class SegmentFrame:
    """The start frame of one segment: ``origin`` = the first row's chart position ``(e, n,
    u)``, ``course_rad`` = its ground-track direction in the chart (math-ENU: 0 = +e, counter-
    clockwise toward +n, the convention `channels.states_from_channels` reads ψ in).

    ``rows`` takes chart rows to the frame (x along the course, y to its left, z up; the
    velocities rotated with the axes) and ``chart_rows`` is its exact inverse.
    """

    origin: np.ndarray
    course_rad: float

    @classmethod
    def at_row(cls, row: np.ndarray) -> SegmentFrame:
        """The frame whose origin and course are this chart row's."""
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (len(CHANNELS),):
            raise ValueError(f"a chart row has {len(CHANNELS)} channels, got shape {row.shape}")
        edot, ndot = row[VELOCITY_IDX[0]], row[VELOCITY_IDX[1]]
        ground_speed = math.hypot(edot, ndot)
        if ground_speed < MINIMUM_GROUND_SPEED_MPS:
            raise ValueError(
                f"the start row's ground speed {ground_speed:.3f} m/s is below "
                f"{MINIMUM_GROUND_SPEED_MPS:g} m/s: no ground-track direction to build the "
                "segment frame from"
            )
        return cls(origin=row[list(POSITION_IDX)].copy(), course_rad=math.atan2(ndot, edot))

    def _rotation(self) -> np.ndarray:
        """The 2×2 matrix taking chart (e, n) components to frame (x, y) components."""
        c, s = math.cos(self.course_rad), math.sin(self.course_rad)
        return np.array([[c, s], [-s, c]], dtype=np.float64)

    def rows(self, chart_rows: np.ndarray) -> np.ndarray:
        """``[R, 6]`` chart rows → ``[R, 6]`` rows in this frame."""
        chart_rows = np.asarray(chart_rows, dtype=np.float64)
        if chart_rows.ndim != 2 or chart_rows.shape[1] != len(CHANNELS):
            raise ValueError(f"chart rows are [R, {len(CHANNELS)}], got shape {chart_rows.shape}")
        rotation = self._rotation()
        out = np.empty_like(chart_rows)
        horizontal = list(POSITION_IDX[:2])
        out[:, horizontal] = (chart_rows[:, horizontal] - self.origin[:2]) @ rotation.T
        out[:, POSITION_IDX[2]] = chart_rows[:, POSITION_IDX[2]] - self.origin[2]
        horizontal_dot = list(VELOCITY_IDX[:2])
        out[:, horizontal_dot] = chart_rows[:, horizontal_dot] @ rotation.T
        out[:, VELOCITY_IDX[2]] = chart_rows[:, VELOCITY_IDX[2]]
        return out

    def chart_rows(self, rows: np.ndarray) -> np.ndarray:
        """``[R, 6]`` rows in this frame → ``[R, 6]`` chart rows (the inverse of `rows`)."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(CHANNELS):
            raise ValueError(f"frame rows are [R, {len(CHANNELS)}], got shape {rows.shape}")
        rotation = self._rotation()
        out = np.empty_like(rows)
        horizontal = list(POSITION_IDX[:2])
        out[:, horizontal] = rows[:, horizontal] @ rotation + self.origin[:2]
        out[:, POSITION_IDX[2]] = rows[:, POSITION_IDX[2]] + self.origin[2]
        horizontal_dot = list(VELOCITY_IDX[:2])
        out[:, horizontal_dot] = rows[:, horizontal_dot] @ rotation
        out[:, VELOCITY_IDX[2]] = rows[:, VELOCITY_IDX[2]]
        return out


def interpolate_rows(times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """The polyline ``(times, values)`` read at ``query_times``, every channel linearly; refuses
    a query outside the polyline's span (nothing is extrapolated or held)."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1 or values.shape != (len(times), len(CHANNELS)):
        raise ValueError(
            f"a polyline is [T] times with [T, {len(CHANNELS)}] values, got {times.shape}, {values.shape}"
        )
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("polyline times must be strictly increasing, at least two of them")
    if query_times.min() < times[0] - STEP_TOLERANCE_S or query_times.max() > times[-1] + STEP_TOLERANCE_S:
        raise ValueError(
            f"query times [{query_times.min():.3f}, {query_times.max():.3f}] s leave the polyline's "
            f"span [{times[0]:.3f}, {times[-1]:.3f}] s"
        )
    return np.column_stack([np.interp(query_times, times, values[:, c]) for c in range(len(CHANNELS))])


def segment_rows(
    times: np.ndarray, values: np.ndarray, start_time_s: float, segment_s: float, dt_s: float
) -> np.ndarray:
    """The segment of the polyline that starts at ``start_time_s``: ``[R, 6]`` rows every
    ``dt_s`` up to ``segment_s`` after the start, in the frame of the start row."""
    chart = interpolate_rows(times, values, float(start_time_s) + segment_offsets_s(segment_s, dt_s))
    return SegmentFrame.at_row(chart[0]).rows(chart)


def truth_segment_rows(series: FlightSeries, start_time_s: float, segment_s: float, dt_s: float) -> np.ndarray:
    """The TRUTH's segment starting at ``start_time_s`` (the flight's own clock): read from the
    supervision arrays, so it reaches into the fitted tail exactly as far as the executor's
    targets do. At an observed anchor the start row IS ``series.values[anchor]``."""
    return segment_rows(series.supervision_times, series.supervision_values, start_time_s, segment_s, dt_s)


def state_row(series: FlightSeries, time_s: float) -> np.ndarray:
    """The truth's chart row at ``time_s`` — the encoder's current-state input (plan §2.4), the
    same polyline `truth_segment_rows` reads."""
    return interpolate_rows(series.supervision_times, series.supervision_values, np.array([float(time_s)]))[0]


def flown_polyline(series: FlightSeries, anchor: int, forecast: Forecast) -> tuple[np.ndarray, np.ndarray]:
    """A forecast as a polyline on the flight's clock: the anchor's observed row, then the
    flown rows (`inference/receding.py` reads a flown history the same way)."""
    if int(forecast.anchor) != int(anchor):
        raise ValueError(f"the forecast was made at anchor {forecast.anchor}, not {anchor}")
    times = np.concatenate(([float(series.times[anchor])], np.asarray(forecast.times, dtype=np.float64)))
    values = np.concatenate((
        np.asarray(series.values[anchor : anchor + 1], dtype=np.float64),
        np.asarray(forecast.values, dtype=np.float64),
    ))
    return times, values


def flown_segment_rows(
    series: FlightSeries, anchor: int, forecast: Forecast, start_time_s: float, segment_s: float, dt_s: float
) -> np.ndarray:
    """The segment of a FLOWN forecast starting at ``start_time_s`` — what the encoder reads
    back after the executor flies a round (plan §2.1), through the same function as the truth."""
    times, values = flown_polyline(series, anchor, forecast)
    return segment_rows(times, values, start_time_s, segment_s, dt_s)


def segment_start_times(series: FlightSeries, anchor: int, segment_s: float) -> np.ndarray:
    """The start times of the FULL segments from ``anchor`` forward: ``t_anchor + k·segment_s``
    for every k whose segment ends at or before the truth's end (the supervision rows, the
    track closed to the threshold). What remains after the last full segment is the prior's
    ``LANDED`` business (plan §2.5), not a shorter segment."""
    if segment_s <= 0.0:
        raise ValueError(f"segment_s is positive seconds, got {segment_s!r}")
    start = float(series.times[anchor])
    end = float(series.supervision_times[-1])
    count = int(math.floor((end - start + STEP_TOLERANCE_S) / segment_s))
    return start + segment_s * np.arange(max(count, 0), dtype=np.float64)


__all__ = [
    "MINIMUM_GROUND_SPEED_MPS", "SEGMENT_CHANNELS", "STEP_TOLERANCE_S", "SegmentFrame",
    "flown_polyline", "flown_segment_rows", "interpolate_rows", "segment_offsets_s",
    "segment_row_count", "segment_rows", "segment_start_times", "state_row", "truth_segment_rows",
]
