"""The segment-plan head's own reading of a cohort (two-tier v2 §4, the per-epoch
`segment_plan_validation` block): what the decoded plans say against the truth INSIDE the
plan's span. The common-grid report scores the whole remaining approach with the last
waypoint held flat past M × 30 s and cannot say it did; this block is the number to read.

Per flight, from the fixed anchor:

* ``covered_ade_m`` — the mean 3D displacement between the plan's rows (the anchor row
  first, the decoded rows after, the last held) and the truth's supervision rows on a 1 s
  grid over [0, min(T_truth, span)]: the plan judged over what it claims to cover;
* ``segment_error_m[k]`` — the waypoint error at the end of segment k over the flights
  whose truth reaches it (the plan's e(30), e(60), …);
* the ARRIVAL confusion — the truth arrives within the span / the plan says it does —
  and the signed arrival-time error where both do.

Airports are pooled with equal weight, as the objective is read.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.config import PLAN_WAYPOINT_SEGMENT_S, TSConfig
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FlightSeries, truth_duration_s
from ts_transformer.outputs.segment_plan.decode import DecodedPlan, plan_rows

READOUT_GRID_S = 1.0


def _truth_positions(series: FlightSeries, times: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(times, series.supervision_times, series.supervision_values[:, channel]) for channel in POSITION_IDX
    ])


def plan_reading(plan: DecodedPlan, series: FlightSeries, anchor: int, span_s: float) -> dict[str, object]:
    """One flight's plan against its truth (`readout` module docstring)."""
    anchor_time = float(series.times[anchor])
    truth_s = truth_duration_s(series, anchor)
    covered_s = min(truth_s, span_s)
    offsets, positions = plan_rows(plan, series.target_chart)
    rows_t = np.concatenate([[0.0], offsets])
    rows = np.concatenate([np.asarray(series.values[anchor], dtype=np.float64)[list(POSITION_IDX)][None, :], positions], axis=0)
    grid = np.arange(0.0, covered_s + 1e-9, READOUT_GRID_S)
    predicted = np.column_stack([np.interp(grid, rows_t, rows[:, c]) for c in range(3)])
    observed = _truth_positions(series, anchor_time + grid)
    ends = PLAN_WAYPOINT_SEGMENT_S * np.arange(1, len(plan.waypoints) + 1)
    reached = ends <= truth_s + 1e-9
    waypoint_error = np.full(len(ends), np.nan)
    if reached.any():
        waypoint_error[reached] = np.linalg.norm(
            plan.waypoints[reached] - _truth_positions(series, anchor_time + ends[reached]), axis=1,
        )
    truth_arrives = truth_s <= span_s + 1e-9
    return {
        "covered_ade_m": float(np.linalg.norm(predicted - observed, axis=1).mean()),
        "covered_s": float(covered_s),
        "waypoint_error_m": waypoint_error,
        "truth_arrives": truth_arrives,
        "plan_arrives": plan.arrives,
        "arrival_time_error_s": (
            float(plan.arrival_time_s - truth_s) if plan.arrives and truth_arrives else None
        ),
    }


def summarize_readings(readings: Sequence[dict[str, object]], config: TSConfig) -> dict[str, object]:
    """The cohort block from per-flight readings (`plan_reading`)."""
    if not readings:
        raise ValueError("a segment-plan readout needs at least one flight")
    segments = int(config.segment_plan_segments)
    errors = np.stack([np.asarray(r["waypoint_error_m"], dtype=np.float64) for r in readings])   # [N, M]
    truth_arrives = np.array([bool(r["truth_arrives"]) for r in readings])
    plan_arrives = np.array([bool(r["plan_arrives"]) for r in readings])
    timing = np.array([r["arrival_time_error_s"] for r in readings if r["arrival_time_error_s"] is not None], dtype=np.float64)
    return {
        "flights": len(readings),
        "span_s": float(segments * PLAN_WAYPOINT_SEGMENT_S),
        "covered_ade_m": float(np.mean([r["covered_ade_m"] for r in readings])),
        "covered_s_mean": float(np.mean([r["covered_s"] for r in readings])),
        "segment_error_m": [
            {
                "end_s": float(PLAN_WAYPOINT_SEGMENT_S * (k + 1)),
                "mean": float(np.nanmean(errors[:, k])) if np.isfinite(errors[:, k]).any() else None,
                "n": int(np.isfinite(errors[:, k]).sum()),
            }
            for k in range(segments)
        ],
        "truth_arrives_share": float(truth_arrives.mean()),
        "plan_arrives_share": float(plan_arrives.mean()),
        "arrival_confusion": {
            "both": int((truth_arrives & plan_arrives).sum()),
            "plan_only": int((~truth_arrives & plan_arrives).sum()),
            "truth_only": int((truth_arrives & ~plan_arrives).sum()),
            "neither": int((~truth_arrives & ~plan_arrives).sum()),
        },
        "arrival_time_error_s": {
            "mean_abs": float(np.abs(timing).mean()) if len(timing) else None,
            "mean_signed": float(timing.mean()) if len(timing) else None,
            "n": int(len(timing)),
        },
    }


def pool_airports(blocks: dict[str, dict[str, object]]) -> dict[str, object]:
    """Equal airport weight on the means and shares, summed counts, the per-airport blocks kept."""
    values = list(blocks.values())
    first = values[0]

    def mean_of(key: str) -> float:
        return float(np.mean([block[key] for block in values]))

    segments = len(first["segment_error_m"])
    segment_error = []
    for k in range(segments):
        cells = [block["segment_error_m"][k] for block in values if block["segment_error_m"][k]["mean"] is not None]
        segment_error.append({
            "end_s": first["segment_error_m"][k]["end_s"],
            "mean": float(np.mean([cell["mean"] for cell in cells])) if cells else None,
            "n": int(sum(block["segment_error_m"][k]["n"] for block in values)),
        })
    timings = [block["arrival_time_error_s"] for block in values if block["arrival_time_error_s"]["n"]]
    return {
        "flights": int(sum(block["flights"] for block in values)),
        "span_s": first["span_s"],
        "covered_ade_m": mean_of("covered_ade_m"),
        "covered_s_mean": mean_of("covered_s_mean"),
        "segment_error_m": segment_error,
        "truth_arrives_share": mean_of("truth_arrives_share"),
        "plan_arrives_share": mean_of("plan_arrives_share"),
        "arrival_confusion": {
            key: int(sum(block["arrival_confusion"][key] for block in values)) for key in first["arrival_confusion"]
        },
        "arrival_time_error_s": {
            "mean_abs": float(np.mean([t["mean_abs"] for t in timings])) if timings else None,
            "mean_signed": float(np.mean([t["mean_signed"] for t in timings])) if timings else None,
            "n": int(sum(t["n"] for t in timings)),
        },
        "by_airport": blocks,
    }


__all__ = ["READOUT_GRID_S", "plan_reading", "pool_airports", "summarize_readings"]
