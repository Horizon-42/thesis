"""Terminal-event verdicts and auditable batch aggregation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from evaluation.arrival import (
    TARGET_CONTEXT_TOLERANCE_M,
    TERMINAL_PLANE_TOLERANCE_M,
    ArrivalDeviation,
    arrival_deviation,
    subject_of,
)
from evaluation.context import ContextKey, resolve_context
from evaluation.records import TrajectoryRecord
from evaluation.reference import (
    ENDPOINT_TOLERANCE_M,
    ReferenceComparison,
    compare_to_reference,
    horizontal_arc_length_m,
    load_reference,
    reference_span,
)
from evaluation.stats import magnitude_spread, mean, signed_spread
from evaluation.thresholds import (
    ICAO_NORMAL_FSD_FRACTION,
    LPV_VERTICAL_BOUND_M,
    LPV_VERTICAL_FSD_MIN_M,
    LPV_VERTICAL_SCALE_MODEL,
    NORMAL_95_MULTIPLIER,
    AssessmentContext,
    ComponentResult,
    Verdict,
)

@dataclass(frozen=True)
class TrajectoryEvaluation:
    record_id: str
    file: str | None
    subject: str
    solved: bool
    success: bool
    verdict: Verdict
    lateral_result: ComponentResult
    vertical_result: ComponentResult
    deviation: ArrivalDeviation | None
    event_status: str
    violations: tuple[str, ...]
    reason: str | None
    benchmark: str
    airport: str
    runway: str
    lateral_bound_m: float | None
    guidance_lateral_bound_m: float | None
    runway_lateral_bound_m: float
    vertical_lower_bound_m: float | None
    vertical_upper_bound_m: float | None
    lateral_interval_m: tuple[float, float] | None = None
    vertical_interval_m: tuple[float, float] | None = None
    flight_key: str | None = None


def _component(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
) -> ComponentResult:
    if estimate is None or lower is None or upper is None:
        return "indeterminate"
    return "pass" if lower <= estimate <= upper else "fail"


def _diagnostic_interval(
    estimate: float | None,
    sigma: float | None,
) -> tuple[float, float] | None:
    """Return an estimator interval for audit, never for verdict classification."""
    if estimate is None:
        return None
    margin = 0.0 if sigma is None else NORMAL_95_MULTIPLIER * sigma
    return estimate - margin, estimate + margin


def _composite(lateral: ComponentResult, vertical: ComponentResult) -> Verdict:
    if "fail" in (lateral, vertical):
        return "fail"
    if lateral == "pass" and vertical == "pass":
        return "pass"
    return "indeterminate"


def _validate_deviation(value: ArrivalDeviation) -> None:
    required = {
        "along_track_m": value.along_track_m,
        "cross_track_m": value.cross_track_m,
        "speed_ms": value.speed_ms,
        "heading_rad": value.heading_rad,
        "flight_time_s": value.flight_time_s,
    }
    optional = {
        "vertical_m": value.vertical_m,
        "lateral_sigma_m": value.lateral_sigma_m,
        "vertical_sigma_m": value.vertical_sigma_m,
        "glidepath_deg": value.glidepath_deg,
        "extrapolation_m": value.extrapolation_m,
    }
    for name, number in {**required, **optional}.items():
        if number is not None and not math.isfinite(float(number)):
            raise ValueError(f"derived deviation {name} must be finite, got {number!r}")
    for name in ("lateral_sigma_m", "vertical_sigma_m", "extrapolation_m"):
        number = optional[name]
        if number is not None and number < 0.0:
            raise ValueError(f"derived deviation {name} must be non-negative")


def evaluate_record(
    record: TrajectoryRecord,
    *,
    context: AssessmentContext,
) -> TrajectoryEvaluation:
    """Evaluate one trajectory at its runway-threshold event."""
    record_id = str(
        record.source.get("id")
        or (record.path.stem if record.path is not None else "trajectory")
    )
    file = record.path.name if record.path is not None else None
    subject = subject_of(record)
    limits = context.limits()
    common = dict(
        record_id=record_id,
        file=file,
        subject=subject,
        benchmark=context.benchmark,
        airport=context.airport,
        runway=context.runway,
        lateral_bound_m=limits.effective_lateral_m,
        guidance_lateral_bound_m=limits.guidance_lateral_m,
        runway_lateral_bound_m=limits.runway_lateral_m,
        vertical_lower_bound_m=limits.vertical_lower_m,
        vertical_upper_bound_m=limits.vertical_upper_m,
        flight_key=record.source.get("flight_key"),
    )
    if not record.solved:
        return TrajectoryEvaluation(
            **common, solved=False, success=False, verdict="fail",
            lateral_result="indeterminate", vertical_result="indeterminate",
            deviation=None, event_status="unsolved", violations=("unsolved",),
            reason=record.reason or "trajectory unsolved",
        )

    outcome = arrival_deviation(record, context=context)
    if outcome.deviation is None:
        computed_failure = subject != "observed"
        return TrajectoryEvaluation(
            **common, solved=True, success=False,
            verdict="fail" if computed_failure else "indeterminate",
            lateral_result="indeterminate", vertical_result="indeterminate",
            deviation=None, event_status=outcome.event_status,
            violations=((outcome.event_status,) if computed_failure else ()),
            reason=outcome.reason,
        )
    deviation = outcome.deviation
    _validate_deviation(deviation)

    lateral_result = _component(
        deviation.cross_track_m,
        None if limits.effective_lateral_m is None else -limits.effective_lateral_m,
        limits.effective_lateral_m,
    )
    vertical_result = _component(
        deviation.vertical_m,
        limits.vertical_lower_m,
        limits.vertical_upper_m,
    )
    lateral_interval = _diagnostic_interval(
        deviation.cross_track_m, deviation.lateral_sigma_m
    )
    vertical_interval = _diagnostic_interval(
        deviation.vertical_m, deviation.vertical_sigma_m
    )
    verdict = _composite(lateral_result, vertical_result)
    violations: list[str] = []
    if lateral_result == "fail":
        violations.append("lateral")
    if vertical_result == "fail":
        violations.append("vertical")
    reason = None
    if verdict == "indeterminate":
        reasons: list[str] = []
        if lateral_result == "indeterminate":
            reasons.append("lateral bound or estimate unavailable")
        if vertical_result == "indeterminate":
            reasons.append(
                limits.vertical_reason
                or "vertical bound or estimate unavailable"
            )
        reason = "; ".join(reasons) or None
    return TrajectoryEvaluation(
        **common, solved=True, success=verdict == "pass", verdict=verdict,
        lateral_result=lateral_result, vertical_result=vertical_result,
        deviation=deviation, event_status=outcome.event_status,
        violations=tuple(violations), reason=reason,
        lateral_interval_m=lateral_interval, vertical_interval_m=vertical_interval,
    )


def evaluate_batch(
    records: Iterable[TrajectoryRecord],
    *,
    contexts: Mapping[ContextKey, AssessmentContext],
    observed_availability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a batch and serialize every verdict-changing parameter."""
    record_list = list(records)
    evaluations: list[TrajectoryEvaluation] = []
    rows: list[dict[str, Any]] = []
    comparisons: list[ReferenceComparison] = []
    used: dict[ContextKey, AssessmentContext] = {}
    for record in record_list:
        context = resolve_context(record, contexts)
        used[(context.airport, context.runway)] = context
        evaluation = evaluate_record(record, context=context)
        evaluations.append(evaluation)
        row = _row(evaluation)
        if evaluation.subject == "observed":
            event = record.source.get("observed_threshold_event")
            if isinstance(event, dict):
                # Audit copy of policy-free producer output; no evaluation
                # limits or verdicts are added to it.
                row["observed_threshold_event"] = event
            source_integrity = record.source.get("source_integrity")
            if isinstance(source_integrity, dict):
                row["source_integrity"] = source_integrity
        if record.reference_file is not None:
            reference = load_reference(record)
            span = reference_span(record, reference)
            block: dict[str, Any] = {
                "file": record.reference_file,
                "comparison_status": "compared" if span.comparable else "skipped",
                "endpoint_tolerance_m": span.tolerance_m,
                "start_gap_m": span.start_gap_m,
                "end_gap_m": span.end_gap_m,
            }
            if (
                record.solved and span.comparable
                and horizontal_arc_length_m(record.states) > 0.0
                and horizontal_arc_length_m(reference.states) > 0.0
            ):
                comparison = compare_to_reference(record, reference)
                comparisons.append(comparison)
                block.update(
                    reference_flight_time_s=comparison.reference_flight_time_s,
                    flight_time_delta_s=comparison.flight_time_delta_s,
                    path_lateral_m=comparison.path_lateral_m,
                    path_vertical_m=comparison.path_vertical_m,
                )
            else:
                block["note"] = span.reason or "zero-horizontal-extent path; comparison skipped"
                block["comparison_status"] = "skipped"
            row["reference"] = block
        rows.append(row)

    measured = [item for item in evaluations if item.deviation is not None]
    solved = [item for item in evaluations if item.solved]
    verdict_counts = {
        key: sum(item.verdict == key for item in evaluations)
        for key in ("pass", "fail", "indeterminate")
    }
    subjects = {item.subject for item in evaluations}
    observed_block = None
    if observed_availability is not None:
        if subjects != {"observed"}:
            raise ValueError(
                "observed_availability can be attached only to an observed-only batch"
            )
        observed_block = _validated_observed_availability(observed_availability)
    total = len(evaluations)
    times = [item.deviation.flight_time_s for item in measured]
    return {
        "schema_version": "terminal-approach-evaluation-v3",
        "methodology": {
            "event": {
                "computed_predicted": "terminal_state_at_threshold_plane",
                "observed": (
                    "serialized_observed_threshold_event_v6: bracket-selected physical "
                    "pass with one producer-side robust 3D fit; no evaluation refit"
                ),
                "terminal_plane_tolerance_m": TERMINAL_PLANE_TOLERANCE_M,
            },
            "uncertainty": {
                "confidence": 0.95,
                "normal_multiplier": NORMAL_95_MULTIPLIER,
                "classification": "diagnostic_only_not_used_by_verdict",
                "verdict_rule": "point_estimate_against_inclusive_component_bounds",
                "observed_sigma_source": (
                    "serialized event-v6 diagnostic 95% margin divided by 1.96"
                ),
                "unmodelled_sources": [
                    "ADS-B geometric-altitude update alignment and measurement error",
                    "runway/FAS survey uncertainty",
                    "geoid/datum uncertainty", "model-form and extrapolation uncertainty",
                ],
            },
            "observed_source_integrity": {
                "required_track_schema": "harvest-tracks-v2-source-timing",
                "position_time_basis": "lastposupdate",
                "freshness": (
                    "state_time-lastcontact <= 15 s and "
                    "state_time-lastposupdate <= 15 s"
                ),
                "held_state_policy": (
                    "one state snapshot nearest each lastposupdate; asynchronous "
                    "geoaltitude changes are audited"
                ),
                "coverage_gap_policy": (
                    "do not bridge position-update gaps greater than 15 s"
                ),
            },
            "terminal_vertical": {
                "reference": "LTP elevation MSL + published FAS TCH",
                "trajectory_altitude_datum": "msl",
                "target_context_tolerance_m": TARGET_CONTEXT_TOLERANCE_M,
                "lpv": {
                    "scale_model": LPV_VERTICAL_SCALE_MODEL,
                    "one_sided_minimum_fsd_m": LPV_VERTICAL_FSD_MIN_M,
                    "normal_fsd_fraction": ICAO_NORMAL_FSD_FRACTION,
                    "effective_threshold_bound_m": LPV_VERTICAL_BOUND_M,
                    "sources": [
                        {
                            "document": "RTCA DO-229D",
                            "location": "§§2.2.4.4.4 and 2.2.5.4.4",
                            "use": "angular LPV scale and 15 m minimum linear FSD",
                        },
                        {
                            "document": "ICAO Doc 9613, Fifth Edition (2023)",
                            "location": (
                                "Volume II, Part C, Chapter 5, Section B, "
                                "§5.3.3.1.1.1(b)"
                            ),
                            "use": "normal-operation one-half vertical FSD",
                        },
                        {
                            "document": (
                                "Garmin AXIS Pilot's Guide for Certified Aircraft, "
                                "190-03123-01 Rev B"
                            ),
                            "location": "Chapter 2, page 2-15, Glidepath - GPS Source",
                            "use": "current certified-avionics confirmation of 15 m LPV lower FSD",
                        },
                    ],
                },
            },
            "reference_comparison": {
                "endpoint_tolerance_m": ENDPOINT_TOLERANCE_M,
                "mismatched_span_policy": "skip_path_and_time_metrics",
                "resampling": "common-endpoint horizontal arc fraction",
            },
        },
        "assessment_contexts": [
            {**context.to_dict(), "resolved_limits": context.limits().to_dict()}
            for _key, context in sorted(used.items())
        ],
        "subject": sorted(subjects)[0] if len(subjects) == 1 else "mixed",
        **({"observed": observed_block} if observed_block is not None else {}),
        "total": total,
        "measured": len(measured),
        "solved": len(solved),
        "solve_rate": len(solved) / total if total else 0.0,
        "verdict_counts": verdict_counts,
        "successful": verdict_counts["pass"],
        "failed": verdict_counts["fail"],
        "indeterminate": verdict_counts["indeterminate"],
        "success_rate": verdict_counts["pass"] / total if total else 0.0,
        "lateral_m": magnitude_spread([item.deviation.lateral_m for item in measured]),
        "vertical_m": signed_spread([
            item.deviation.vertical_m
            for item in measured
            if item.deviation.vertical_m is not None
        ]),
        "final_time_s": (
            {"mean": mean(times), "min": min(times), "max": max(times)} if times else None
        ),
        "reference": _reference_aggregate(comparisons),
        "trajectories": rows,
    }


def _validated_observed_availability(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = "arrival_candidates_excluding_not_landing"
    if value.get("denominator") != expected:
        raise ValueError(f"observed availability denominator must be {expected!r}")

    def count(name: str) -> int:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"observed availability {name} must be a non-negative integer")
        return item

    denominator = count("event_denominator")
    estimated = count("event_estimated")
    unavailable = count("event_unavailable")
    excluded = count("excluded_not_landing")
    integrity_excluded = value.get("source_integrity_excluded_candidates", 0)
    if (
        isinstance(integrity_excluded, bool)
        or not isinstance(integrity_excluded, int)
        or integrity_excluded < 0
    ):
        raise ValueError(
            "observed availability source_integrity_excluded_candidates must be "
            "a non-negative integer"
        )
    if integrity_excluded > unavailable:
        raise ValueError(
            "observed availability source-integrity exclusions exceed unavailable "
            "events"
        )
    if estimated + unavailable != denominator:
        raise ValueError("observed availability counts do not match event_denominator")
    rate = value.get("event_estimated_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
            or not math.isfinite(float(rate)):
        raise ValueError("observed availability rate must be finite")
    expected_rate = estimated / denominator if denominator else 0.0
    if not math.isclose(float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("observed availability rate disagrees with its counts")
    return {
        "denominator": expected,
        "event_denominator": denominator,
        "event_estimated": estimated,
        "event_unavailable": unavailable,
        "event_estimated_rate": expected_rate,
        "excluded_not_landing": excluded,
        "source_integrity_excluded_candidates": integrity_excluded,
    }


def _reference_aggregate(comparisons: list[ReferenceComparison]) -> dict[str, Any] | None:
    if not comparisons:
        return None
    deltas = [item.flight_time_delta_s for item in comparisons]
    return {
        "compared": len(comparisons),
        "flight_time_delta_s": {"mean": mean(deltas), "min": min(deltas), "max": max(deltas)},
        "path_lateral_m": {
            "mean": mean([item.path_lateral_m["mean"] for item in comparisons]),
            "max": max(item.path_lateral_m["max"] for item in comparisons),
        },
        "path_vertical_m": {
            "mean_abs": mean([item.path_vertical_m["mean_abs"] for item in comparisons]),
            "max_abs": max(item.path_vertical_m["max_abs"] for item in comparisons),
        },
    }


def _row(item: TrajectoryEvaluation) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": item.record_id,
        "file": item.file,
        "flight_key": item.flight_key,
        "subject": item.subject,
        "airport": item.airport,
        "runway": item.runway,
        "benchmark": item.benchmark,
        "solved": item.solved,
        "success": item.success,
        "verdict": item.verdict,
        "event_status": item.event_status,
        "lateral_result": item.lateral_result,
        "vertical_result": item.vertical_result,
        "violations": list(item.violations),
        "bounds": {
            "guidance_lateral_m": item.guidance_lateral_bound_m,
            "runway_lateral_m": item.runway_lateral_bound_m,
            "effective_lateral_m": item.lateral_bound_m,
            "vertical_lower_m": item.vertical_lower_bound_m,
            "vertical_upper_m": item.vertical_upper_bound_m,
        },
    }
    if item.deviation is not None:
        deviation = item.deviation
        row["deviation"] = {
            "along_track_m": deviation.along_track_m,
            "cross_track_m": deviation.cross_track_m,
            "vertical_m": deviation.vertical_m,
            "speed_ms": deviation.speed_ms,
            "heading_rad": deviation.heading_rad,
            "final_time_s": deviation.flight_time_s,
            "lateral_sigma_m": deviation.lateral_sigma_m,
            "vertical_sigma_m": deviation.vertical_sigma_m,
            "lateral_interval_m": item.lateral_interval_m,
            "vertical_interval_m": item.vertical_interval_m,
            "extrapolated": deviation.extrapolated,
            "glidepath_deg": deviation.glidepath_deg,
            "extrapolation_m": deviation.extrapolation_m,
        }
        # Keep common descriptive columns flat for simple report consumers.
        row.update(
            lateral_m=deviation.lateral_m,
            cross_track_m=deviation.cross_track_m,
            along_track_m=deviation.along_track_m,
            vertical_m=deviation.vertical_m,
            speed_ms=deviation.speed_ms,
            heading_rad=deviation.heading_rad,
            final_time_s=deviation.flight_time_s,
        )
    if item.reason is not None:
        row["reason"] = item.reason
    return row
