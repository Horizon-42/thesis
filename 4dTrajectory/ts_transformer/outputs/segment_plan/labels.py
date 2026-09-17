"""The segment-plan labels (two-tier v2 §4): the TRUTH's coarse plan from an anchor.

The plan layer predicts the flight's position every `PLAN_WAYPOINT_SEGMENT_S` (30 s) after
the anchor for M segments, in RUNWAY AXES about the anchor — along the course (positive
toward the threshold), across it, up — and, per segment, whether the flight has ARRIVED
by the segment's end (crossed the threshold on the final: the truth's supervision rows end
at that crossing, observed or fitted) and, in the segment it arrives in, the fraction of
the segment flown before it did. A segment the truth never reaches carries no position;
every segment carries the arrival bit.

Runway axes are what makes a plan flight-independent: the same base turn at KRDU 05L and
23R is the same label. The interface to L1 is in the chart (`chart_deltas` rotates back).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.config import PLAN_WAYPOINT_SEGMENT_S, TSConfig
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.geometry.final_approach_geometry import final_approach_arrays

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The batch context a segment-plan window carries (`SegmentPlanContext.row`).
CONTEXT_TARGETS = "segment_targets"                    # [M, 3] (along, across, up) metres about the anchor
CONTEXT_VALID = "segment_valid"                        # [M] 1 where the truth reaches the segment's end
CONTEXT_ARRIVED = "segment_arrived"                    # [M] 1 where the truth has arrived by the segment's end
CONTEXT_ARRIVAL_FRACTION = "segment_arrival_fraction"  # [M] the fraction flown of the arrival segment, 0 elsewhere
CONTEXT_ARRIVAL_VALID = "segment_arrival_valid"        # [M] 1 at the arrival segment
CONTEXT_RUNWAY_HEADING = "runway_heading_rad"          # [] the course (math-ENU) — the features' and the labels' axis
CONTEXT_TARGET_CHART = "target_chart"                  # [3] the threshold in the chart — where an arrived plan ends
CONTEXT_KEYS = (
    CONTEXT_TARGETS, CONTEXT_VALID, CONTEXT_ARRIVED, CONTEXT_ARRIVAL_FRACTION, CONTEXT_ARRIVAL_VALID,
    CONTEXT_RUNWAY_HEADING, CONTEXT_TARGET_CHART,
)
#: What a stored checkpoint's targets are: the segment length is spelled so a change of it
#: refuses the checkpoint instead of reading its plan at another spacing.
SEGMENT_TARGET_CONTRACT = f"segment-plan-runway-axes-{PLAN_WAYPOINT_SEGMENT_S:g}s-v1"
#: The scales the positions are regressed in (metres): a plan's along-course reach, its
#: cross-track spread, its descent.
ALONG_SCALE_M = 10_000.0
ACROSS_SCALE_M = 10_000.0
UP_SCALE_M = 1_000.0
POSITION_SCALE = np.array([ALONG_SCALE_M, ACROSS_SCALE_M, UP_SCALE_M], dtype=np.float32)


def runway_deltas(de: np.ndarray, dn: np.ndarray, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Chart deltas → (along, across) the runway course ``psi`` (math-ENU): along is positive
    in the direction of flight on the final (toward the threshold), across positive to its
    left. The inverse of `chart_deltas`."""
    c, s = np.cos(psi), np.sin(psi)
    return de * c + dn * s, -de * s + dn * c


def chart_deltas(along: np.ndarray, across: np.ndarray, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """(along, across) → chart (Δe, Δn): the inverse of `runway_deltas`."""
    c, s = np.cos(psi), np.sin(psi)
    return along * c - across * s, along * s + across * c


def runway_heading_rad(series: FlightSeries) -> float:
    """The flight's runway course, the one definition the state path's corridor reads."""
    return float(final_approach_arrays(series)[CONTEXT_RUNWAY_HEADING])


def segment_labels(series: FlightSeries, anchor: int, config: TSConfig) -> dict[str, np.ndarray]:
    """The truth's coarse plan from ``anchor`` for ``config.segment_plan_segments`` segments.

    The truth is the flight's supervision rows (the observed track closed to the threshold at
    its crossing, `dataset._build_supervision`), so the arrival time is `truth_duration_s`.
    """
    count = int(config.segment_plan_segments)
    psi = runway_heading_rad(series)
    anchor_time = float(series.times[anchor])
    anchor_position = np.asarray(series.values[anchor], dtype=np.float64)[list(POSITION_IDX)]
    arrival_s = truth_duration_s(series, anchor)
    ends = PLAN_WAYPOINT_SEGMENT_S * np.arange(1, count + 1, dtype=np.float64)
    valid = ends <= arrival_s + 1e-9
    positions = np.column_stack([
        np.interp(anchor_time + ends, series.supervision_times, series.supervision_values[:, channel])
        for channel in POSITION_IDX
    ]) - anchor_position
    along, across = runway_deltas(positions[:, 0], positions[:, 1], psi)
    targets = np.column_stack([along, across, positions[:, 2]]) * valid[:, None]
    arrived = ends >= arrival_s - 1e-9
    # the segment the truth arrives in: the first whose end the arrival does not pass
    arrival_valid = np.zeros(count, dtype=np.float64)
    fraction = np.zeros(count, dtype=np.float64)
    if arrival_s <= ends[-1] + 1e-9:
        k = int(np.argmax(arrived))
        arrival_valid[k] = 1.0
        fraction[k] = min(max((arrival_s - (ends[k] - PLAN_WAYPOINT_SEGMENT_S)) / PLAN_WAYPOINT_SEGMENT_S, 0.0), 1.0)
    return {
        CONTEXT_TARGETS: targets.astype(np.float32),
        CONTEXT_VALID: valid.astype(np.float32),
        CONTEXT_ARRIVED: arrived.astype(np.float32),
        CONTEXT_ARRIVAL_FRACTION: fraction.astype(np.float32),
        CONTEXT_ARRIVAL_VALID: arrival_valid.astype(np.float32),
        CONTEXT_RUNWAY_HEADING: np.array(psi, dtype=np.float32),
        CONTEXT_TARGET_CHART: np.asarray(series.target_chart, dtype=np.float32),
    }


__all__ = [
    "ACROSS_SCALE_M", "ALONG_SCALE_M", "CONTEXT_ARRIVAL_FRACTION", "CONTEXT_ARRIVAL_VALID", "CONTEXT_ARRIVED",
    "CONTEXT_KEYS", "CONTEXT_RUNWAY_HEADING", "CONTEXT_TARGET_CHART", "CONTEXT_TARGETS", "CONTEXT_VALID", "POSITION_SCALE",
    "SEGMENT_TARGET_CONTRACT", "UP_SCALE_M", "chart_deltas", "runway_deltas", "runway_heading_rad", "segment_labels",
]
