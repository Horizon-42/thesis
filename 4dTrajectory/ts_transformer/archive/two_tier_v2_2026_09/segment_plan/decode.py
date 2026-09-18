"""The segment-plan head's prediction as a plan in the chart (two-tier v2 §4).

The head emits, per coarse segment, a position in runway axes about the anchor, the
probability of having arrived by the segment's end and the fraction of the segment flown
before arriving. `decode_plan` turns that into the flight's waypoints in the chart and its
ARRIVAL — the first segment whose arrival probability clears `ARRIVAL_PROBABILITY` — and
`plan_rows` lays the rows a forecast carries: the waypoints before the arrival segment,
then the THRESHOLD at the arrival time (a flight that has arrived is at the threshold; the
head's position in that segment is never supervised, `labels.segment_labels`). A plan
that never arrives inside its M segments is horizon-capped at the last waypoint.

Pure numpy over one flight: the lockstep's segment-head source (S3) and the strategy's
forecast and replay read the same two functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_transformer.config import PLAN_WAYPOINT_SEGMENT_S
from ts_transformer.outputs.segment_plan.labels import chart_deltas

#: A segment whose arrival probability reaches this is the arrival segment.
ARRIVAL_PROBABILITY = 0.5
#: The least time the arrival row may sit after the previous row: the plan's clock must be
#: strictly increasing (the common-grid resampler refuses a zero segment).
MIN_ARRIVAL_S = 1.0


@dataclass(frozen=True)
class DecodedPlan:
    """One flight's plan: ``waypoints`` ``[M, 3]`` the chart position (e, n, u) at each
    segment's end, ``times_s`` ``[M]`` seconds after the anchor, the ``arrival_segment``
    (0-based, or None when no segment clears the threshold), the ``arrival_time_s`` inside
    it, and the arrival probability of that segment (the highest of the M without one)."""

    waypoints: np.ndarray
    times_s: np.ndarray
    arrival_segment: int | None
    arrival_time_s: float | None
    arrival_probability: float

    @property
    def arrives(self) -> bool:
        return self.arrival_segment is not None


def decode_plan(
    positions: np.ndarray,
    arrived_probability: np.ndarray,
    arrival_fraction: np.ndarray,
    anchor_position: np.ndarray,
    psi: float,
    *,
    threshold: float = ARRIVAL_PROBABILITY,
) -> DecodedPlan:
    """``positions`` ``[M, 3]`` (along, across, up) about the anchor → the plan in the chart."""
    positions = np.asarray(positions, dtype=np.float64)
    arrived = np.asarray(arrived_probability, dtype=np.float64)
    fraction = np.asarray(arrival_fraction, dtype=np.float64)
    count = len(positions)
    anchor_position = np.asarray(anchor_position, dtype=np.float64)
    de, dn = chart_deltas(positions[:, 0], positions[:, 1], psi)
    waypoints = np.column_stack([de, dn, positions[:, 2]]) + anchor_position
    ends = PLAN_WAYPOINT_SEGMENT_S * np.arange(1, count + 1, dtype=np.float64)
    reached = np.flatnonzero(arrived >= threshold)
    if len(reached) == 0:
        return DecodedPlan(waypoints, ends, None, None, float(arrived.max()))
    k = int(reached[0])
    into = float(np.clip(fraction[k] * PLAN_WAYPOINT_SEGMENT_S, MIN_ARRIVAL_S, PLAN_WAYPOINT_SEGMENT_S))
    return DecodedPlan(waypoints, ends, k, float(ends[k] - PLAN_WAYPOINT_SEGMENT_S + into), float(arrived[k]))


def plan_rows(plan: DecodedPlan, target_chart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows a forecast of ``plan`` carries: ``(offsets_s [R], positions [R, 3])`` — the
    waypoints before the arrival segment, then the threshold at the arrival time; every
    waypoint when the plan never arrives."""
    if not plan.arrives:
        return plan.times_s.copy(), plan.waypoints.copy()
    k = plan.arrival_segment
    offsets = np.concatenate([plan.times_s[:k], [plan.arrival_time_s]])
    positions = np.concatenate([plan.waypoints[:k], np.asarray(target_chart, dtype=np.float64)[None, :3]], axis=0)
    return offsets, positions


def replay_rows(
    plan: DecodedPlan, target_chart: np.ndarray, segments: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """`plan_rows` padded to exactly ``segments`` rows for the batch replay — the threshold
    HELD, one coarse segment apart, after the arrival (a landed flight stays where it
    landed) — as ``(durations_s [M], positions [M, 3], arrival_time_s, rows)``: the arrival
    time is the last waypoint's when the plan never arrives, ``rows`` how many rows are the
    plan's own (the rest are the hold, which no kinematic reading may score)."""
    offsets, positions = plan_rows(plan, target_chart)
    final_s = float(offsets[-1])
    rows = len(offsets)
    missing = segments - rows
    if missing:
        held = final_s + PLAN_WAYPOINT_SEGMENT_S * np.arange(1, missing + 1, dtype=np.float64)
        offsets = np.concatenate([offsets, held])
        positions = np.concatenate([positions, np.repeat(positions[-1:], missing, axis=0)], axis=0)
    return np.diff(np.concatenate([[0.0], offsets])), positions, final_s, rows


__all__ = ["ARRIVAL_PROBABILITY", "MIN_ARRIVAL_S", "DecodedPlan", "decode_plan", "plan_rows", "replay_rows"]
