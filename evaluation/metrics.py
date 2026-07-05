"""Final-state matching + batch metrics.

The core judgement: does the trajectory's LAST state match the target state?
Lateral deviation = great-circle distance (``geokit.haversine_m``); vertical =
signed altitude difference. Speed / heading deltas are reported for context but
NOT gated — the regulation gates (``thresholds.py``) are positional.

Batch metrics over a set of records:

  * solve rate    — records with a non-empty solution / all records
  * success rate  — records whose final state passes BOTH gates / all records
                    (also reported among solved only)
  * lateral / vertical deviation spreads over the solved records — mean, p95
    and max. p95 is reported because RNP containment is itself a 95 % statistic
    (8260.58D: position within the leg's RNP radius 95 % of the time), so the
    p95 lateral deviation compares directly against RNP-style limits.
  * mean / min / max flight time over the solved records
  * when records carry a ``reference_file`` pointer, the observed-track
    comparison (``reference.py``): per-trajectory flight-time delta + path
    deviation, aggregated over the solved records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from geokit import haversine_m

from evaluation.records import TrajectoryRecord
from evaluation.reference import (
    ReferenceComparison,
    compare_to_reference,
    horizontal_arc_length_m,
    load_reference,
)
from evaluation.stats import magnitude_spread, mean, signed_spread
from evaluation.thresholds import DeviationThresholds


@dataclass(frozen=True)
class FinalStateDeviation:
    """The last state vs the target state (signed where a sign is meaningful)."""

    lateral_m: float
    vertical_m: float   # final − target (positive = high)
    speed_ms: float     # final − target
    heading_rad: float  # smallest signed angular difference
    flight_time_s: float


def final_state_deviation(record: TrajectoryRecord) -> FinalStateDeviation:
    """Measure the record's final state against its target (solved records only)."""
    final = record.states[-1]
    target = record.target_state
    return FinalStateDeviation(
        lateral_m=haversine_m(final["lat"], final["lon"], target["lat"], target["lon"]),
        vertical_m=final["alt"] - target["alt"],
        speed_ms=final["V"] - target["V"],
        heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
        flight_time_s=final["t"],
    )


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """One record's verdict: solved? within the regulation gates?

    ``violations`` holds one human-readable string per failed check (empty =
    success; exactly ``("unsolved",)`` for unsolved records). ``deviation`` is
    None iff unsolved; ``reason`` is the solver failure (unsolved only).
    """

    record_id: str
    file: str | None
    solved: bool
    success: bool
    deviation: FinalStateDeviation | None
    violations: tuple[str, ...]
    reason: str | None


def evaluate_record(
    record: TrajectoryRecord,
    thresholds: DeviationThresholds = DeviationThresholds(),
) -> TrajectoryEvaluation:
    """Judge one trajectory: unsolved fails outright; solved must pass both gates."""
    record_id = str(
        record.source.get("id")
        or (record.path.stem if record.path is not None else "trajectory")
    )
    file = record.path.name if record.path is not None else None
    if not record.solved:
        return TrajectoryEvaluation(
            record_id, file, solved=False, success=False,
            deviation=None, violations=("unsolved",), reason=record.reason,
        )
    deviation = final_state_deviation(record)
    violations: list[str] = []
    if deviation.lateral_m > thresholds.lateral_max_m:
        violations.append(
            f"lateral {deviation.lateral_m:.1f} m > {thresholds.lateral_max_m:.2f} m"
        )
    if deviation.vertical_m < -thresholds.vertical_below_max_m:
        violations.append(
            f"vertical {deviation.vertical_m:.1f} m below the "
            f"-{thresholds.vertical_below_max_m:.2f} m window"
        )
    if deviation.vertical_m > thresholds.vertical_above_max_m:
        violations.append(
            f"vertical {deviation.vertical_m:.1f} m above the "
            f"+{thresholds.vertical_above_max_m:.2f} m window"
        )
    return TrajectoryEvaluation(
        record_id, file, solved=True, success=not violations,
        deviation=deviation, violations=tuple(violations), reason=None,
    )


def evaluate_batch(
    records: Sequence[TrajectoryRecord],
    thresholds: DeviationThresholds = DeviationThresholds(),
) -> dict[str, Any]:
    """Evaluate every record and aggregate the batch metrics.

    Returns the report dict (= the ``evaluation_report.json`` schema — defined
    HERE; other modules reference it):

        {
          "thresholds": asdict(DeviationThresholds),
          "total": int,                       # all records
          "solved": int,                      # records with a non-empty solution
          "solve_rate": float,                # solved / total (0.0 on an empty batch)
          "successful": int,                  # solved AND inside both gates
          "success_rate": float,              # successful / total
          "success_rate_among_solved": float | None,   # None when nothing solved
          "lateral_m": magnitude spread | None,        # final lateral dev over solved
          "vertical_m": signed spread | None,          # final vertical dev (+ = high)
          "final_time_s": {"mean","min","max"} | None, # flight times over solved
          "reference": _reference_aggregate() result,
          "trajectories": [_row() result per record, input order],
        }

    "magnitude/signed spread" are the ``stats.magnitude_spread`` /
    ``stats.signed_spread`` dicts; ``None`` aggregates mean "no solved records".
    Rows of records that carry ``reference_file`` gain a ``"reference"`` block —
    always ``{"file", "flight_time_s"}`` (the observed baseline); solved records
    with arc-matchable paths add ``{"flight_time_delta_s", "path_lateral_m",
    "path_vertical_m"}`` (the :class:`ReferenceComparison` fields); a solved
    record whose path (or reference) has zero horizontal extent adds a ``"note"``
    saying the comparison was skipped instead.
    """
    evaluations = [evaluate_record(record, thresholds) for record in records]
    solved = [e for e in evaluations if e.solved]
    succeeded = [e for e in evaluations if e.success]
    lateral = [e.deviation.lateral_m for e in solved]
    vertical = [e.deviation.vertical_m for e in solved]
    times = [e.deviation.flight_time_s for e in solved]
    total = len(evaluations)

    rows = []
    comparisons: list[ReferenceComparison] = []
    for record, evaluation in zip(records, evaluations):
        row = _row(evaluation)
        if record.reference_file is not None:
            reference = load_reference(record)
            block: dict[str, Any] = {
                "file": record.reference_file,
                "flight_time_s": float(reference.final_time_s),
            }
            if record.solved and horizontal_arc_length_m(record.states) > 0.0 \
                    and horizontal_arc_length_m(reference.states) > 0.0:
                comparison = compare_to_reference(record, reference)
                comparisons.append(comparison)
                block.update(
                    flight_time_delta_s=comparison.flight_time_delta_s,
                    path_lateral_m=comparison.path_lateral_m,
                    path_vertical_m=comparison.path_vertical_m,
                )
            elif record.solved:
                # A zero-horizontal-extent path (e.g. a rollout truncated at its very
                # first step) cannot be arc-length matched — keep the observed baseline
                # and say why, instead of one degenerate record aborting the batch.
                block["note"] = "zero-horizontal-extent path; comparison skipped"
            rows.append({**row, "reference": block})
        else:
            rows.append(row)

    return {
        "thresholds": asdict(thresholds),
        "total": total,
        "solved": len(solved),
        "solve_rate": len(solved) / total if total else 0.0,
        "successful": len(succeeded),
        "success_rate": len(succeeded) / total if total else 0.0,
        "success_rate_among_solved": (len(succeeded) / len(solved)) if solved else None,
        "lateral_m": magnitude_spread(lateral),
        "vertical_m": signed_spread(vertical),
        "final_time_s": (
            {"mean": mean(times), "min": min(times), "max": max(times)} if times else None
        ),
        "reference": _reference_aggregate(comparisons),
        "trajectories": rows,
    }


def _reference_aggregate(comparisons: list[ReferenceComparison]) -> dict[str, Any] | None:
    """Batch view of the observed-track comparisons (solved records only).

    ``None`` when nothing was compared, else:

        {"compared": int,                                # comparisons aggregated
         "flight_time_delta_s": {"mean","min","max"},    # optimized − observed, s
         "path_lateral_m": {"mean","max"},               # mean of per-flight means /
         "path_vertical_m": {"mean_abs","max_abs"}}      #   max of per-flight maxes, m
    """
    if not comparisons:
        return None
    deltas = [c.flight_time_delta_s for c in comparisons]
    return {
        "compared": len(comparisons),
        "flight_time_delta_s": {"mean": mean(deltas), "min": min(deltas), "max": max(deltas)},
        "path_lateral_m": {
            "mean": mean([c.path_lateral_m["mean"] for c in comparisons]),
            "max": max(c.path_lateral_m["max"] for c in comparisons),
        },
        "path_vertical_m": {
            "mean_abs": mean([c.path_vertical_m["mean_abs"] for c in comparisons]),
            "max_abs": max(c.path_vertical_m["max_abs"] for c in comparisons),
        },
    }


def _row(evaluation: TrajectoryEvaluation) -> dict[str, Any]:
    """One report row (flattened :class:`TrajectoryEvaluation`).

    Always ``{"id", "file", "solved", "success", "violations": [str, …]}``;
    solved rows add the :class:`FinalStateDeviation` fields ``{"lateral_m",
    "vertical_m", "speed_ms", "heading_rad", "final_time_s"}``, unsolved rows
    add ``"reason"``. (``evaluate_batch`` may attach a ``"reference"`` block on
    top — documented there.)
    """
    row: dict[str, Any] = {
        "id": evaluation.record_id,
        "file": evaluation.file,
        "solved": evaluation.solved,
        "success": evaluation.success,
        "violations": list(evaluation.violations),
    }
    if evaluation.deviation is not None:
        deviation = evaluation.deviation
        row.update(
            lateral_m=deviation.lateral_m,
            vertical_m=deviation.vertical_m,
            speed_ms=deviation.speed_ms,
            heading_rad=deviation.heading_rad,
            final_time_s=deviation.flight_time_s,
        )
    if evaluation.reason:
        row["reason"] = evaluation.reason
    return row
