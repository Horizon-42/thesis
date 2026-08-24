"""Terminal-event verdicts and auditable batch aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from evaluation.arrival import (
    TARGET_CONTEXT_TOLERANCE_M,
    TERMINAL_PLANE_TOLERANCE_M,
    ArrivalDeviation,
    arrival_deviation,
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
from evaluation.speed_gate import (
    LANDING_AERO_KEY,
    MISSING_LANDING_AERO_REASON,
    OBSERVED_NO_CROSSING_SPEED_REASON,
    OBSERVED_SPEED_CRITERION_ID,
    OBSERVED_SPEED_POLICY,
    OBSERVED_UNRESOLVED_AIRFRAME_REASON,
    SPEED_CRITERION_ID,
    SPEED_GATE_UPPER_ADDITIVE_MS,
    VREF_STALL_MULTIPLIER,
    SpeedGateBounds,
    speed_gate_bounds,
)
from evaluation.stats import magnitude_spread, signed_spread
from evaluation.thresholds import (
    LATERAL_CRITERION_ID,
    RNAV_TERMINAL_VERTICAL_BOUND_M,
    RNAV_TERMINAL_VERTICAL_STANDARD_ID,
    AssessmentContext,
    ComponentResult,
    Verdict,
)

REPORT_SCHEMA_VERSION = "terminal-approach-evaluation-v6"

# The denominator every observed availability block is counted against. Evaluation
# checks the LABEL, not the counts: the block is producer-owned audit output (the
# harvest computed it from the unfiltered track roster, which evaluation never sees),
# and a report that renamed the population would be claiming a different measurement.
OBSERVED_AVAILABILITY_DENOMINATOR = "arrival_candidates_excluding_not_landing"

# What this report claims and how it was produced -- static prose, serialized with
# every batch so a report can be read years later without this source. It lives at
# module level because none of it varies per batch; the constants interpolated into
# it are the same ones the verdicts use.
METHODOLOGY: dict[str, Any] = {
    "event": {
        "computed_predicted": "terminal_state_at_threshold_plane",
        "observed": (
            "serialized_runway_threshold_event_v1: direct 3D interpolation "
            "inside observed support, otherwise the single winning "
            "assignment fit for a right-censored pass; no evaluation refit"
        ),
        "terminal_plane_tolerance_m": TERMINAL_PLANE_TOLERANCE_M,
    },
    "uncertainty": {
        "verdict_rule": "point_estimate_against_inclusive_component_bounds",
        "observed_status": "uncalibrated",
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
    "terminal_lateral": {
        "criterion": LATERAL_CRITERION_ID,
        "bound_m": "runway_width_m / 2, per runway",
        "reference": "authoritative published LANDING threshold (displaced where published)",
        "claim_boundary": (
            "landing geometry -- whether the crossing lay over the pavement. This is "
            "NOT a navigation-containment result: the LPV course width at threshold "
            "(106.75 m) and the RNP APCH LNAV 0.15 NM cross-track allowance (277.8 m) "
            "are 2.3x-18x wider than the runway at every threshold in this fleet, so "
            "they are reported as procedure provenance and never used as the bound"
        ),
    },
    "terminal_vertical": {
        "reference": "LTP elevation MSL + published FAS TCH",
        "trajectory_altitude_datum": "msl",
        "target_context_tolerance_m": TARGET_CONTEXT_TOLERANCE_M,
        "common_rnav_terminal_acceptance": {
            "standard_id": RNAV_TERMINAL_VERTICAL_STANDARD_ID,
            "lower_m": -RNAV_TERMINAL_VERTICAL_BOUND_M,
            "upper_m": RNAV_TERMINAL_VERTICAL_BOUND_M,
            "source": {
                "document": "ICAO Doc 9613, Fifth Edition (2023)",
                "location": (
                    "Volume II, Part C, Chapter 5, Section A, "
                    "§5.3.4.4.7"
                ),
                "use": (
                    "RNP APCH Baro-VNAV final-approach vertical-deviation "
                    "limit used as the common RNAV/LPV terminal acceptance bound"
                ),
            },
            "claim_boundary": (
                "terminal final-approach geometry; not touchdown or "
                "landing certification"
            ),
        },
    },
    "terminal_speed": {
        "criterion": SPEED_CRITERION_ID,
        "bound_ms": (
            "[1.23 x Vs1g(crossing mass), 1.23 x Vs1g + 20 kt] inclusive, per record"
        ),
        "stall_model": (
            "Vs1g = sqrt(2 m g / (rho0 S Cl_max_landing)); ISA sea-level rho0, the "
            "model's landing Cl_max (aircraft.aero_params, shared with the optimizer's "
            "velocity floor); S and Cl_max from the record's source.landing_aero"
        ),
        "vref_stall_multiplier": VREF_STALL_MULTIPLIER,
        "upper_additive_ms": SPEED_GATE_UPPER_ADDITIVE_MS,
        "sources": [
            {
                "document": "14 CFR 25.125(b)(2)(i)",
                "use": "V_REF may not be less than 1.23 V_SR0 (the lower bound anchor)",
            },
            {
                "document": (
                    "FSF ALAR Briefing Note 7.1 'Stabilized Approach', Table 1 "
                    "element 3 (Flight Safety Digest, Aug-Nov 2000)"
                ),
                "use": (
                    "speed not more than V_REF + 20 kt and not less than V_REF "
                    "(the window)"
                ),
            },
        ],
        "subjects": (
            "all subjects; optimized/predicted are judged on the crossing model "
            "airspeed, and " + OBSERVED_SPEED_POLICY
        ),
        "observed_proxy_criterion": OBSERVED_SPEED_CRITERION_ID,
        "observed_proxy_caveat": (
            "wind is unmodelled: an ordinary 10 kt headwind is half the 20 kt "
            "window, so an observed speed fail can reflect the day's wind rather "
            "than the flight; quote observed speed rates with this caveat"
        ),
        "claim_boundary": (
            "model-consistent threshold-crossing energy, judged in TAS with TAS "
            "treated as CAS (<1% at this fleet's threshold elevations, all below "
            "200 m); not an operational or certification speed check"
        ),
    },
    # Additive within v6 (2026-08-24). Descriptive only — the row/batch fields it
    # describes carry a measured quantity, never a verdict component.
    "observed_crossing_ground_speed": {
        "source": (
            "harvest threshold event `crossing_ground_speed_m_s`: ADS-B reported "
            "ground speed interpolated at a direct bracket, or OLS-extrapolated over "
            "the same kept samples as the position fit (final_approach.fit_line)"
        ),
        "reference": "ground-referenced; wind is unmodelled",
        "use": (
            "audit statistic on observed subjects only; never composed into any "
            "verdict and never an input to the stall-anchored airspeed gate"
        ),
        "availability": (
            "rows whose serialized event carries the field; events written before "
            "2026-08-24, and censored fits without enough speed-bearing samples, "
            "report null"
        ),
    },
    "reference_comparison": {
        "endpoint_tolerance_m": ENDPOINT_TOLERANCE_M,
        "mismatched_span_policy": "skip_path_and_time_metrics",
        "resampling": "common-endpoint horizontal arc fraction",
    },
}


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
    speed_result: ComponentResult
    deviation: ArrivalDeviation | None
    event_status: str
    violations: tuple[str, ...]
    reason: str | None
    benchmark: str
    airport: str
    runway: str
    lateral_bound_m: float
    vertical_lower_bound_m: float | None
    vertical_upper_bound_m: float | None
    # Per-record (mass-anchored), unlike the two context-owned bounds above; None when
    # the record was not speed-gradable (unsolved, observed, or no landing_aero block).
    speed_bounds: SpeedGateBounds | None = None
    flight_key: str | None = None


def _component(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
) -> ComponentResult:
    if estimate is None or lower is None or upper is None:
        return "indeterminate"
    return "pass" if lower <= estimate <= upper else "fail"


def _composite(
    lateral: ComponentResult,
    vertical: ComponentResult,
    speed: ComponentResult | None,
) -> Verdict:
    """Compose the components that are IN SCOPE for this record.

    ``speed`` is ``None`` for observed subjects — the gate is out of scope there (no
    crossing airspeed was measured; see ``speed_gate.OBSERVED_SPEED_POLICY``), which is
    different from an in-scope component that came back ``indeterminate``.
    """
    components = (lateral, vertical) if speed is None else (lateral, vertical, speed)
    if "fail" in components:
        return "fail"
    if all(component == "pass" for component in components):
        return "pass"
    return "indeterminate"


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
    subject = record.source["subject"]
    limits = context.limits()
    common = dict(
        record_id=record_id,
        file=file,
        subject=subject,
        benchmark=context.benchmark,
        airport=context.airport,
        runway=context.runway,
        lateral_bound_m=limits.lateral_m,
        vertical_lower_bound_m=limits.vertical_lower_m,
        vertical_upper_bound_m=limits.vertical_upper_m,
        flight_key=record.source.get("flight_key"),
    )
    if not record.solved:
        return TrajectoryEvaluation(
            **common, solved=False, success=False, verdict="fail",
            lateral_result="indeterminate", vertical_result="indeterminate",
            speed_result="indeterminate",
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
            speed_result="indeterminate",
            deviation=None, event_status=outcome.event_status,
            violations=((outcome.event_status,) if computed_failure else ()),
            reason=outcome.reason,
        )
    deviation = outcome.deviation
    lateral_result = _component(
        deviation.cross_track_m, -limits.lateral_m, limits.lateral_m
    )
    vertical_result = _component(
        deviation.vertical_m,
        limits.vertical_lower_m,
        limits.vertical_upper_m,
    )
    # Every subject is speed-graded against the stall-anchored window; the two
    # branches differ only in WHICH measured quantity is judged. Computed subjects:
    # the crossing state's model airspeed. Observed subjects: the event's fitted
    # crossing GROUND speed as a STATED PROXY (wind unmodelled — declared in the
    # proxy criterion id and METHODOLOGY["terminal_speed"], never silently equated
    # with airspeed). Absent/null landing_aero reads "unspecified" and grades
    # indeterminate; a PRESENT malformed block raises in speed_gate_bounds.
    speed_in_scope: ComponentResult | None = None
    speed_bounds: SpeedGateBounds | None = None
    speed_reason: str | None = None
    landing_aero = record.source.get(LANDING_AERO_KEY)
    if subject != "observed":
        if landing_aero is None:
            speed_in_scope = "indeterminate"
            speed_reason = MISSING_LANDING_AERO_REASON
        else:
            speed_bounds = speed_gate_bounds(deviation.crossing_mass_kg, landing_aero)
            speed_in_scope = _component(
                deviation.crossing_speed_ms, speed_bounds.lower_ms, speed_bounds.upper_ms
            )
    else:
        if landing_aero is None:
            speed_in_scope = "indeterminate"
            speed_reason = OBSERVED_UNRESOLVED_AIRFRAME_REASON
        elif deviation.crossing_ground_speed_ms is None:
            speed_in_scope = "indeterminate"
            speed_reason = OBSERVED_NO_CROSSING_SPEED_REASON
        else:
            speed_bounds = speed_gate_bounds(deviation.crossing_mass_kg, landing_aero)
            speed_in_scope = _component(
                deviation.crossing_ground_speed_ms,
                speed_bounds.lower_ms,
                speed_bounds.upper_ms,
            )
    verdict = _composite(lateral_result, vertical_result, speed_in_scope)
    violations: list[str] = []
    if lateral_result == "fail":
        violations.append("lateral")
    if vertical_result == "fail":
        violations.append("vertical")
    if speed_in_scope == "fail":
        violations.append("speed")
    # Lateral is always decidable once a crossing was measured -- a runway always has
    # a width -- so an indeterminate composite means a missing vertical reference, a
    # missing landing_aero block, or both; name every one that applies.
    reason = None
    if verdict == "indeterminate":
        parts = []
        if vertical_result == "indeterminate":
            parts.append(limits.vertical_reason or "vertical bound or estimate unavailable")
        if speed_in_scope == "indeterminate" and speed_reason is not None:
            parts.append(speed_reason)
        reason = "; ".join(parts) or None
    return TrajectoryEvaluation(
        **common, solved=True, success=verdict == "pass", verdict=verdict,
        lateral_result=lateral_result, vertical_result=vertical_result,
        speed_result=speed_in_scope if speed_in_scope is not None else "indeterminate",
        speed_bounds=speed_bounds,
        deviation=deviation, event_status=outcome.event_status,
        violations=tuple(violations), reason=reason,
    )


def evaluate_batch(
    records: Iterable[TrajectoryRecord],
    *,
    contexts: Mapping[ContextKey, AssessmentContext],
    observed_availability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a batch and serialize every verdict-changing parameter."""
    # Iterated, never materialized: `records` may be a generator over a batch whose
    # resolved states are ~1 MB per flight. Everything retained below (evaluations, rows,
    # comparisons) is per-flight metadata, not trajectory arrays.
    evaluations: list[TrajectoryEvaluation] = []
    rows: list[dict[str, Any]] = []
    comparisons: list[ReferenceComparison] = []
    used: dict[ContextKey, AssessmentContext] = {}
    for record in records:
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
            # ``comparable`` already implies both paths are non-empty; what it does
            # not imply is that either MOVED, and an arc-length resample of a
            # stationary path has nothing to parametrize by.
            if (
                span.comparable
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
    speed_result_counts = {
        key: sum(item.speed_result == key for item in evaluations)
        for key in ("pass", "fail", "indeterminate")
    }
    subjects = {item.subject for item in evaluations}
    if observed_availability is not None and subjects != {"observed"}:
        raise ValueError(
            "observed_availability can be attached only to an observed-only batch"
        )
    total = len(evaluations)
    times = [item.deviation.flight_time_s for item in measured]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "methodology": METHODOLOGY,
        "assessment_contexts": [
            {**context.to_dict(), "resolved_limits": context.limits().to_dict()}
            for _key, context in sorted(used.items())
        ],
        "subject": sorted(subjects)[0] if len(subjects) == 1 else "mixed",
        **(
            {"observed": _observed_availability(observed_availability)}
            if observed_availability is not None
            else {}
        ),
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
        "speed_result_counts": speed_result_counts,
        "crossing_speed_ms": magnitude_spread([
            item.deviation.crossing_speed_ms
            for item in measured
            if item.deviation.crossing_speed_ms is not None
        ]),
        # Additive within v6 (2026-08-24): spread of the events' audit-only ADS-B
        # ground speeds (observed subjects; see METHODOLOGY
        # ["observed_crossing_ground_speed"]). Null when no row carries one.
        "crossing_ground_speed_ms": magnitude_spread([
            item.deviation.crossing_ground_speed_ms
            for item in measured
            if item.deviation.crossing_ground_speed_ms is not None
        ]),
        "final_time_s": (
            {"mean": fmean(times), "min": min(times), "max": max(times)} if times else None
        ),
        "reference": _reference_aggregate(comparisons),
        "trajectories": rows,
    }


def _observed_availability(value: Mapping[str, Any]) -> dict[str, Any]:
    """Carry the harvest's event-availability block into the report.

    Copied verbatim, like ``observed_threshold_event``: the counts are measured
    upstream from the unfiltered track roster (``harvest.observed
    .source_event_availability``, which validates that roster and derives them),
    and re-deriving them here from data evaluation cannot see would only be able to
    restate them. What IS checked is the denominator label, because that names the
    population the rate refers to and the report repeats the claim.
    """
    if value.get("denominator") != OBSERVED_AVAILABILITY_DENOMINATOR:
        raise ValueError(
            "observed availability denominator must be "
            f"{OBSERVED_AVAILABILITY_DENOMINATOR!r}"
        )
    return dict(value)


def _reference_aggregate(comparisons: list[ReferenceComparison]) -> dict[str, Any] | None:
    if not comparisons:
        return None
    deltas = [item.flight_time_delta_s for item in comparisons]
    return {
        "compared": len(comparisons),
        "flight_time_delta_s": {"mean": fmean(deltas), "min": min(deltas), "max": max(deltas)},
        "path_lateral_m": {
            "mean": fmean([item.path_lateral_m["mean"] for item in comparisons]),
            "max": max(item.path_lateral_m["max"] for item in comparisons),
        },
        "path_vertical_m": {
            "mean_abs": fmean([item.path_vertical_m["mean_abs"] for item in comparisons]),
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
        "speed_result": item.speed_result,
        "violations": list(item.violations),
        "bounds": {
            "lateral_criterion": LATERAL_CRITERION_ID,
            "lateral_m": item.lateral_bound_m,
            "vertical_lower_m": item.vertical_lower_bound_m,
            "vertical_upper_m": item.vertical_upper_bound_m,
            **(
                item.speed_bounds.to_dict(
                    OBSERVED_SPEED_CRITERION_ID
                    if item.subject == "observed"
                    else SPEED_CRITERION_ID
                )
                if item.speed_bounds is not None
                else {
                    "speed_criterion": (
                        OBSERVED_SPEED_CRITERION_ID
                        if item.subject == "observed"
                        else SPEED_CRITERION_ID
                    ),
                    "stall_speed_ms": None,
                    "speed_lower_ms": None,
                    "speed_upper_ms": None,
                }
            ),
        },
    }
    if item.deviation is not None:
        deviation = item.deviation
        row["deviation"] = {
            "along_track_m": deviation.along_track_m,
            "cross_track_m": deviation.cross_track_m,
            "vertical_m": deviation.vertical_m,
            "speed_ms": deviation.speed_ms,
            "crossing_speed_ms": deviation.crossing_speed_ms,
            "crossing_mass_kg": deviation.crossing_mass_kg,
            "crossing_ground_speed_ms": deviation.crossing_ground_speed_ms,
            "heading_rad": deviation.heading_rad,
            "final_time_s": deviation.flight_time_s,
            "extrapolated": bool(
                deviation.extrapolation_m is not None
                and deviation.extrapolation_m > 0.0
            ),
            "extrapolation_m": deviation.extrapolation_m,
        }
        # Keep common descriptive columns flat for simple report consumers. The two
        # crossing speeds are flat too (the frontend's verdict table reads them here):
        # ``crossing_speed_ms`` is the gate-graded model airspeed (computed subjects),
        # ``crossing_ground_speed_ms`` the event's audit-only ADS-B ground speed
        # (observed subjects) — different physical quantities, never merged.
        row.update(
            lateral_m=deviation.lateral_m,
            cross_track_m=deviation.cross_track_m,
            along_track_m=deviation.along_track_m,
            vertical_m=deviation.vertical_m,
            speed_ms=deviation.speed_ms,
            crossing_speed_ms=deviation.crossing_speed_ms,
            crossing_ground_speed_ms=deviation.crossing_ground_speed_ms,
            heading_rad=deviation.heading_rad,
            final_time_s=deviation.flight_time_s,
        )
    if item.reason is not None:
        row["reason"] = item.reason
    return row
