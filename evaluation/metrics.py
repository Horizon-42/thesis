"""The arrival judgement + batch metrics.

Per record: where did the trajectory arrive, and inside the regulation gates?
The arrival EVENT depends on the record's subject (``arrival.py``): a solve or a
prediction is measured at ``states[-1]`` vs ``target_state``, an observed track
at its fitted final approach extrapolated to the threshold. Lateral deviation =
great-circle distance (``geokit.haversine_m``) for a final state, |cross-track
at threshold| for an observed crossing; vertical = signed altitude difference.
Speed / heading deltas are reported for context but NOT gated — the regulation
gates (``thresholds.py``) are positional.

Batch metrics over a set of records:

  * solve rate    — records with a non-empty solution / all records
  * success rate  — records whose arrival passes BOTH gates / all records
                    (also reported among solved only)
  * for observed batches, an ``observed`` block: established / not-established
    counts + rate (the honest analogue of a solve rate, which is 1.0 by
    construction there) and the marginal count (verdicts the data cannot decide)
  * lateral / vertical deviation spreads over the measured records — mean, p95
    and max. p95 is reported because RNP containment is itself a 95 % statistic
    (8260.58D: position within the leg's RNP radius 95 % of the time), so the
    p95 lateral deviation compares directly against RNP-style limits.
  * mean / min / max flight time over the measured records
  * when records carry a ``reference_file`` pointer, the observed-track
    comparison (``reference.py``): per-trajectory flight-time delta + path
    deviation, aggregated over the solved records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from evaluation.arrival import (
    DEFAULT_SUBJECT,
    ArrivalDeviation,
    EstablishedCriteria,
    arrival_deviation,
    subject_of,
)
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
class TrajectoryEvaluation:
    """One record's verdict: did it arrive, and inside the regulation gates?

    ``violations`` holds one human-readable string per failed check (empty =
    success; exactly ``("unsolved",)`` for unsolved records, ``("not_established",)``
    for an observed track with no measurable arrival). ``deviation`` is None iff there
    is no arrival to measure; ``reason`` says why (solver failure, or which established
    criterion was unmet).

    ``established`` is None for subjects where the notion does not apply (a solve either
    reaches its target or does not exist). ``marginal`` is True when the arrival's 95%
    confidence interval straddles a gate boundary — i.e. the data cannot decide the
    verdict. It is always False for a computed trajectory, which has no measurement
    uncertainty, and on real observed data it is the MAJORITY case: a bare pass rate
    over 25 ft-quantised altitudes and a 9.15 m window claims more than it knows.
    """

    record_id: str
    file: str | None
    solved: bool
    success: bool
    deviation: ArrivalDeviation | None
    violations: tuple[str, ...]
    reason: str | None
    subject: str = DEFAULT_SUBJECT
    established: bool | None = None
    marginal: bool = False
    flight_key: str | None = None


def evaluate_record(
    record: TrajectoryRecord,
    thresholds: DeviationThresholds = DeviationThresholds(),
    *,
    criteria: EstablishedCriteria = EstablishedCriteria(),
) -> TrajectoryEvaluation:
    """Judge one trajectory against the gates, at the arrival its subject defines.

    The arrival event is dispatched by ``arrival.arrival_deviation``: a solve or a
    prediction is measured at ``states[-1]``, an observation at its fitted, extrapolated
    threshold crossing. See ``arrival.py`` for why those are not the same event.
    """
    record_id = str(
        record.source.get("id")
        or (record.path.stem if record.path is not None else "trajectory")
    )
    file = record.path.name if record.path is not None else None
    subject = subject_of(record)
    if not record.solved:
        return TrajectoryEvaluation(
            record_id, file, solved=False, success=False,
            deviation=None, violations=("unsolved",), reason=record.reason,
            subject=subject, flight_key=record.source.get("flight_key"),
        )

    outcome = arrival_deviation(record, criteria=criteria)
    if outcome.deviation is None:
        # An observed track with no measurable arrival. Counted as its own bucket, never
        # dropped and never extrapolated anyway.
        return TrajectoryEvaluation(
            record_id, file, solved=True, success=False,
            deviation=None, violations=("not_established",), reason=outcome.reason,
            subject=subject, established=False,
            flight_key=record.source.get("flight_key"),
        )

    deviation = outcome.deviation
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
        subject=subject, established=outcome.established,
        marginal=_is_marginal(deviation, thresholds),
        flight_key=record.source.get("flight_key"),
    )


def _is_marginal(deviation: ArrivalDeviation, thresholds: DeviationThresholds) -> bool:
    """True when the 95% interval crosses a gate boundary, so the data cannot decide.

    Only ever True for a measured arrival: a computed final state carries no sigma, so
    ``lateral_sigma_m``/``vertical_sigma_m`` are None and this is False by construction.
    """
    if deviation.vertical_sigma_m is None and deviation.lateral_sigma_m is None:
        return False
    half_width = 1.96
    if deviation.vertical_sigma_m is not None:
        margin = half_width * deviation.vertical_sigma_m
        if (
            deviation.vertical_m - margin < -thresholds.vertical_below_max_m
            or deviation.vertical_m + margin > thresholds.vertical_above_max_m
        ) and (
            deviation.vertical_m + margin >= -thresholds.vertical_below_max_m
            and deviation.vertical_m - margin <= thresholds.vertical_above_max_m
        ):
            return True
    if deviation.lateral_sigma_m is not None:
        margin = half_width * deviation.lateral_sigma_m
        # lateral_m is |cross-track|, so the interval on the magnitude is
        # [max(0, lateral − margin), lateral + margin] — the lower bound clamps at 0
        # when the signed CI contains the centreline, it never folds past it.
        low = max(0.0, deviation.lateral_m - margin)
        if low <= thresholds.lateral_max_m <= deviation.lateral_m + margin:
            return True
    return False


def evaluate_batch(
    records: Iterable[TrajectoryRecord],
    thresholds: DeviationThresholds = DeviationThresholds(),
    *,
    criteria: EstablishedCriteria = EstablishedCriteria(),
) -> dict[str, Any]:
    """Evaluate every record and aggregate the batch metrics.

    Returns the report dict (= the ``evaluation_report.json`` schema — defined
    HERE; other modules reference it):

        {
          "thresholds": asdict(DeviationThresholds),
          "subject": "optimized"|"predicted"|"observed"|"mixed",  # of the batch
          "observed": {...},                  # only when observed records exist:
                                              #   established / not_established /
                                              #   established_rate / marginal
          "total": int,                       # all records
          "measured": int,                    # records with an arrival to grade
          "solved": int,                      # records with a non-empty solution
          "solve_rate": float,                # solved / total (0.0 on an empty batch)
          "successful": int,                  # measured AND inside both gates
          "success_rate": float,              # successful / total
          "success_rate_among_solved": float | None,   # None when nothing solved
          "lateral_m": magnitude spread | None,        # lateral dev over measured
          "vertical_m": signed spread | None,          # vertical dev (+ = high)
          "final_time_s": {"mean","min","max"} | None, # flight times over measured
          "reference": _reference_aggregate() result,
          "trajectories": [_row() result per record, input order],
        }

    "magnitude/signed spread" are the ``stats.magnitude_spread`` /
    ``stats.signed_spread`` dicts; ``None`` aggregates mean "nothing measured".
    Rows of records that carry ``reference_file`` gain a ``"reference"`` block —
    always ``{"file", "flight_time_s"}`` (the observed baseline); solved records
    with arc-matchable paths add ``{"flight_time_delta_s", "path_lateral_m",
    "path_vertical_m"}`` (the :class:`ReferenceComparison` fields); a solved
    record whose path (or reference) has zero horizontal extent adds a ``"note"``
    saying the comparison was skipped instead.
    """
    evaluations: list[TrajectoryEvaluation] = []
    rows = []
    comparisons: list[ReferenceComparison] = []
    for record in records:
        evaluation = evaluate_record(record, thresholds, criteria=criteria)
        evaluations.append(evaluation)
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

    # "solved" is structural (the record has states); "measured" is the set with an
    # arrival to grade. They differ only for observed subjects, where a track can have
    # states yet no established final approach to extrapolate from.
    solved = [e for e in evaluations if e.solved]
    measured = [e for e in evaluations if e.deviation is not None]
    succeeded = [e for e in evaluations if e.success]
    lateral = [e.deviation.lateral_m for e in measured]
    vertical = [e.deviation.vertical_m for e in measured]
    times = [e.deviation.flight_time_s for e in measured]
    total = len(evaluations)

    subjects = {e.subject for e in evaluations}
    observed = [e for e in evaluations if e.subject == "observed"]
    subject_block: dict[str, Any] = {}
    if observed:
        established = [e for e in observed if e.established]
        subject_block = {
            # An observed track trivially "has states", so a solve rate over observed
            # data is 1.0 by construction and means nothing. The established rate is
            # the honest analogue; solve_rate stays in the top-level dict only for
            # schema stability, and the renderers suppress it for observed batches.
            "established": len(established),
            "not_established": len(observed) - len(established),
            "established_rate": len(established) / len(observed),
            # How many verdicts the data cannot actually decide (see _is_marginal).
            "marginal": sum(1 for e in observed if e.marginal),
        }

    return {
        "thresholds": asdict(thresholds),
        "subject": sorted(subjects)[0] if len(subjects) == 1 else "mixed",
        **({"observed": subject_block} if subject_block else {}),
        "total": total,
        "measured": len(measured),
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
    measured rows add the deviation fields ``{"lateral_m", "vertical_m",
    "speed_ms", "heading_rad", "final_time_s"}`` (+ the observed extras when
    ``extrapolated``), unsolved / not-established rows add ``"reason"``.
    (``evaluate_batch`` may attach a ``"reference"`` block on top — documented
    there.)
    """
    row: dict[str, Any] = {
        "id": evaluation.record_id,
        "file": evaluation.file,
        "solved": evaluation.solved,
        "success": evaluation.success,
        "violations": list(evaluation.violations),
    }
    if evaluation.flight_key is not None:
        # The join key a consumer needs to match this verdict to a rendered track. `id`
        # is the callsign and is NOT unique (552 distinct callsigns across 996 KRDU
        # arrivals), so a UI that joined on it would swap verdicts between namesakes --
        # which it has done before.
        row["flight_key"] = evaluation.flight_key
    if evaluation.subject != DEFAULT_SUBJECT:
        row["subject"] = evaluation.subject
    if evaluation.established is not None:
        row["established"] = evaluation.established
    if evaluation.deviation is not None:
        deviation = evaluation.deviation
        row.update(
            lateral_m=deviation.lateral_m,
            vertical_m=deviation.vertical_m,
            speed_ms=deviation.speed_ms,
            heading_rad=deviation.heading_rad,
            final_time_s=deviation.flight_time_s,
        )
        if deviation.extrapolated:
            # A measured arrival: say so, and carry what makes the verdict readable.
            row.update(
                extrapolated=True,
                marginal=evaluation.marginal,
                lateral_sigma_m=deviation.lateral_sigma_m,
                vertical_sigma_m=deviation.vertical_sigma_m,
                glidepath_deg=deviation.glidepath_deg,
                extrapolation_m=deviation.extrapolation_m,
            )
    if evaluation.reason:
        row["reason"] = evaluation.reason
    return row
