"""Terminal-event verdicts and auditable batch aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from aircraft.reference_speeds import (
    REFERENCE_SPEEDS_PATH,
    REFERENCE_SPEEDS_SCHEMA,
    reference_speed,
)
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
    LOAD_FACTOR_ASSUMED_1G,
    LOAD_FACTOR_FROM_ADSB,
    LOAD_FACTOR_FROM_CONTROLS,
    MASS_BASIS_CROSSING,
    MASS_BASIS_TYPE_RANGE,
    MIN_BOUND_LOAD_FACTOR,
    NO_CROSSING_REASON,
    NO_MIN_MASS_REASON,
    NO_REFERENCE_SPEED_REASON,
    NO_TYPECODE_REASON,
    OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID,
    OBSERVED_NO_CROSSING_SPEED_REASON,
    OBSERVED_SPEED_CRITERION_ID,
    OBSERVED_SPEED_POLICY,
    OBSERVED_UNRESOLVED_AIRFRAME_REASON,
    SPEED_CRITERION_ID,
    SPEED_GATE_UPPER_ADDITIVE_MS,
    TYPECODE_KEYS,
    SpeedGateBounds,
    record_typecode,
    speed_gate_bounds,
)
from evaluation.stats import magnitude_spread, percentile, signed_spread
from evaluation.thresholds import (
    LATERAL_CRITERION_ID,
    RNAV_TERMINAL_VERTICAL_BOUND_M,
    RNAV_TERMINAL_VERTICAL_STANDARD_ID,
    AssessmentContext,
    ComponentResult,
    Verdict,
)
from evaluation.wind import (
    WIND_MAX_AGE_S,
    WIND_SOURCE,
    WindTable,
    wind_at_landing,
)

REPORT_SCHEMA_VERSION = "terminal-approach-evaluation-v9"
# Versions a consumer that reads only the fields these SHARE (the component results,
# the geometry deviations) may accept. v7 anchored the speed gate's lower bound on the
# crossing load factor; v8 measures that load factor on observed baselines from their
# own ADS-B kinematics and corrects their ground speed by the METAR headwind. Lateral
# and vertical verdicts are unchanged throughout, which is all the ts
# lateral-eligibility seam reads.
READABLE_REPORT_SCHEMA_VERSIONS = (
    "terminal-approach-evaluation-v6", "terminal-approach-evaluation-v7",
    "terminal-approach-evaluation-v8", REPORT_SCHEMA_VERSION,
)

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
            "[V_ref,lo(m) x sqrt(max(n, 1)), V_ref,hi(m) + 20 kt] inclusive, per record; "
            "V_ref,x(m) = V_published,x x sqrt(m / MALW) from the type's published "
            "approach speed at its Maximum Allowable Landing Weight (lo = the lowest "
            "landing flap-configuration value, hi = the highest); computed records at "
            "their crossing mass, observed records over the type's published mass range"
        ),
        "reference_speeds": {
            "table": str(REFERENCE_SPEEDS_PATH.relative_to(REFERENCE_SPEEDS_PATH.parents[1])),
            "schema": REFERENCE_SPEEDS_SCHEMA,
            "definition": (
                "FAA Office of Airports, Aircraft Characteristics Database (October 2024): "
                "Approach_Speed_knot = indicated airspeed at the Maximum Allowable Landing "
                "Weight, the highest Flight Standardization Board / manufacturer value over "
                "the landing flap configurations; Approach_Speed_minimum/maximum_knot where "
                "the FSB gives dual flap-configuration values; MALW_lb the weight it is "
                "quoted at. Minimum operating masses from the manufacturers' airport "
                "planning documents (minimum flight weight, OEW or BOW), else OpenAP 2.4"
            ),
            "stated_approximation": (
                "both flap-configuration values are scaled from the row's ONE MALW; where "
                "the FSB quotes the lower value at a lower landing weight (A321: 140 kt at "
                "75,500 kg, 142 kt at 77,800 kg) the lower edge is up to ~1.5 % low -- "
                "permissive, noted in the row's approach_speed_note"
            ),
            "provenance": "docs/reference_speeds/README.md (URL, retrieval date, SHA-256, page per number)",
            "mass_scaling": "V_ref(m) = V_ref(MALW) x sqrt(m / MALW) (constant lift coefficient)",
            "record_type_keys": list(TYPECODE_KEYS),
            "mass_bases": {
                MASS_BASIS_CROSSING: "computed records: the crossing state's own mass",
                MASS_BASIS_TYPE_RANGE: (
                    "observed records: lower edge at the type's published minimum "
                    "operating mass, upper edge at its MALW (the flight's mass is not measured)"
                ),
            },
        },
        "upper_additive_ms": SPEED_GATE_UPPER_ADDITIVE_MS,
        "load_factor": {
            "rule": (
                "the LOWER bound is V_ref under the lift the crossing manoeuvre demands "
                "(n m g): V_ref,lo x sqrt(n), n clamped at "
                f"{MIN_BOUND_LOAD_FACTOR:g}; the UPPER bound stays at 1 g because it is "
                "an energy criterion, not a stall margin"
            ),
            "sources": {
                LOAD_FACTOR_FROM_CONTROLS: (
                    "controls[-1].load_factor -- the control active over the final "
                    "rollout step, where the graded crossing lies (records with controls: "
                    "optimizer solves, control-output predictions)"
                ),
                LOAD_FACTOR_FROM_ADSB: (
                    "observed baselines: n inverted from the flight's own kinematics "
                    "(psi and gamma rates fitted over the final 20 s of measured track "
                    "before the crossing, the point-mass rotational equations); the "
                    "window facts are on the row"
                ),
                LOAD_FACTOR_ASSUMED_1G: (
                    "declared 1 g for records without controls or a usable window "
                    "(state-output predictions; observed tracks with fewer than four "
                    "samples in the final 20 s); reported on the row"
                ),
            },
            "row_fields": [
                "crossing_load_factor", "crossing_load_factor_source",
                "crossing_load_factor_window",
            ],
        },
        "sources": [
            {
                "document": (
                    "FAA Office of Airports, Aircraft Characteristics Database, "
                    "'Aircraft Characteristics (October 2024)' (approach speed at MALW "
                    "per ICAO type, FSB-validated; definitions per AC 150/5300-13B)"
                ),
                "use": "the published approach speed and MALW anchoring every window",
            },
            {
                "document": (
                    "manufacturer airport planning documents (Airbus AC 3-5-0 Final "
                    "Approach Speed; Boeing ACAP 2.1 weights; Embraer APM Table 2.1; "
                    "Bombardier CRJ900 APM 00-02-01 / 00-03-03) and the Eurocontrol "
                    "Aircraft Performance Database (Vat)"
                ),
                "use": "corroboration of the FAA speeds; the published minimum operating masses",
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
            "airspeed at their crossing mass and load factor, and " + OBSERVED_SPEED_POLICY
        ),
        "observed_proxy_criterion": OBSERVED_SPEED_CRITERION_ID,
        "observed_proxy_caveat": (
            "wind is unmodelled in the proxy: an ordinary 10 kt headwind is half the "
            "20 kt additive, so a proxy speed fail can reflect the day's wind rather "
            "than the flight; quote proxy-judged rates with this caveat"
        ),
        "observed_wind_correction": {
            "criterion": OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID,
            "source": (
                WIND_SOURCE + ": the field's ASOS/METAR reports (IEM archive; direction "
                "degrees true, speed knots), joined to each flight by landing time"
            ),
            "rule": (
                "airspeed estimate = crossing ground speed + W cos(direction_from - "
                "runway course), using the report nearest the landing time when it is "
                f"within {WIND_MAX_AGE_S:g} s and the direction is not variable; "
                "otherwise the row is judged on the ground-speed proxy and says so"
            ),
            "limits": (
                "stated, not modelled: the tower's 10 m wind is not the threshold wind, "
                "reports are hourly with specials, gusts are not applied"
            ),
        },
        "verdict": (
            "pass/fail against the published window; indeterminate only when nothing "
            "can be judged (no type on the record, no published entry for the type, no "
            "published minimum mass for an observed row, no crossing speed) -- "
            "speed_indeterminate_reasons counts each cause"
        ),
        "claim_boundary": (
            "threshold-crossing speed against the type's published landing speed, judged "
            "in TAS with TAS treated as CAS (<1% at this fleet's threshold elevations, "
            "all below 200 m); not an operational or certification speed check"
        ),
    },
    # Describes the observed rows' ``crossing_ground_speed_ms`` -- the quantity the
    # observed speed gate judges (terminal_speed.subjects above), kept in its own field
    # so it is never pooled with the computed subjects' airspeed.
    "observed_crossing_ground_speed": {
        "source": (
            "harvest threshold event `crossing_ground_speed_m_s`: ADS-B reported "
            "ground speed interpolated at a direct bracket, or OLS-extrapolated over "
            "the same kept samples as the position fit (final_approach.fit_line)"
        ),
        "reference": "ground-referenced; wind is unmodelled",
        "use": (
            "observed subjects only: the input to their speed gate. With a usable METAR "
            "report it is corrected by the headwind into crossing_airspeed_estimate_ms "
            "and THAT is judged (criterion " + OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
            + "); without one the raw ground speed is judged as a stated proxy "
            "(criterion " + OBSERVED_SPEED_CRITERION_ID + "). The batch spread is "
            "reported separately from crossing_speed_ms and the two are never "
            "compared as one quantity"
        ),
        "availability": (
            "rows whose serialized event carries the field; events written before "
            "2026-08-24, and censored fits without enough speed-bearing samples, "
            "report null and grade speed indeterminate"
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
    # Which quantity the speed gate judged (the criterion id serialized with the
    # bounds): the model airspeed, the observed METAR-corrected airspeed estimate, or
    # the observed ground-speed proxy.
    speed_criterion: str
    # Per-record (type-, mass- and load-factor-anchored), unlike the two context-owned
    # bounds above; None when no window could be resolved (unsolved, no crossing, no
    # type or no published entry, or an observed event without a fitted speed) --
    # ``speed_reason`` then says why.
    speed_bounds: SpeedGateBounds | None = None
    speed_reason: str | None = None
    # Observed subjects: ground speed + headwind when a usable METAR report exists
    # (``wind`` says which, or why not); the value judged under the estimate criterion.
    crossing_airspeed_estimate_ms: float | None = None
    wind: dict[str, Any] | None = None
    # The judged value's signed distance to the nearest bound (positive inside the
    # window, negative outside): one number, so a residual can be read off the row.
    speed_margin_ms: float | None = None
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
    speed: ComponentResult,
) -> Verdict:
    """Three components, every subject: one fail fails, all pass passes, else open."""
    components = (lateral, vertical, speed)
    if "fail" in components:
        return "fail"
    if all(component == "pass" for component in components):
        return "pass"
    return "indeterminate"


def evaluate_record(
    record: TrajectoryRecord,
    *,
    context: AssessmentContext,
    wind: WindTable | None = None,
) -> TrajectoryEvaluation:
    """Evaluate one trajectory at its runway-threshold event.

    ``wind`` is the airport's METAR table (``evaluation.wind``); it corrects an OBSERVED
    record's crossing ground speed to an airspeed estimate before the speed gate. Absent
    (no table for the airport), the ground-speed proxy applies and the row says so.
    """
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
    default_criterion = (
        OBSERVED_SPEED_CRITERION_ID if subject == "observed" else SPEED_CRITERION_ID
    )
    if not record.solved:
        return TrajectoryEvaluation(
            **common, speed_criterion=default_criterion, speed_reason=NO_CROSSING_REASON,
            solved=False, success=False, verdict="fail",
            lateral_result="indeterminate", vertical_result="indeterminate",
            speed_result="indeterminate",
            deviation=None, event_status="unsolved", violations=("unsolved",),
            reason=record.reason or "trajectory unsolved",
        )

    outcome = arrival_deviation(record, context=context)
    if outcome.deviation is None:
        computed_failure = subject != "observed"
        return TrajectoryEvaluation(
            **common, speed_criterion=default_criterion, speed_reason=NO_CROSSING_REASON,
            solved=True, success=False,
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
    # Every subject is speed-graded against its type's published window; the two
    # branches differ in WHICH measured quantity is judged and which mass frames the
    # window. Computed subjects: the crossing state's model airspeed at the crossing
    # mass. Observed subjects: the METAR-corrected airspeed estimate, or the event's
    # fitted crossing GROUND speed as a STATED PROXY, over the type's published mass
    # range (declared in the criterion id and METHODOLOGY["terminal_speed"]).
    speed_bounds: SpeedGateBounds | None = None
    speed_reason: str | None = None
    typecode = record_typecode(record.source)
    reference = reference_speed(typecode) if typecode is not None else None
    speed_criterion = default_criterion
    airspeed_estimate: float | None = None
    wind_block: dict[str, Any] | None = None
    if subject == "observed":
        judged_speed = deviation.crossing_ground_speed_ms
        if judged_speed is not None:
            # The proxy corrected by the field's wind, when a usable report exists; the
            # block says which report or why none -- a fallback that is stated per row.
            if wind is None:
                wind_block = {
                    "source": WIND_SOURCE, "status": "unavailable",
                    "reason": "no METAR table for this airport (evaluation --metar-root)",
                }
            else:
                headwind, wind_block = wind_at_landing(
                    wind, record.source["landing_time_utc"], context.runway_course_deg
                )
                if headwind is not None:
                    airspeed_estimate = judged_speed + headwind
                    judged_speed = airspeed_estimate
                    speed_criterion = OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    else:
        judged_speed = deviation.crossing_speed_ms
    if typecode is None:
        speed_result: ComponentResult = "indeterminate"
        speed_reason = (
            OBSERVED_UNRESOLVED_AIRFRAME_REASON if subject == "observed" else NO_TYPECODE_REASON
        )
    elif reference is None:
        speed_result = "indeterminate"
        speed_reason = NO_REFERENCE_SPEED_REASON.format(typecode=typecode)
    elif judged_speed is None:
        # Only an observed event can lack its speed: a computed crossing always has a
        # state V (``arrival._state_deviation``).
        speed_result = "indeterminate"
        speed_reason = OBSERVED_NO_CROSSING_SPEED_REASON
    elif subject == "observed" and reference.min_mass_kg is None:
        speed_result = "indeterminate"
        speed_reason = NO_MIN_MASS_REASON.format(typecode=typecode)
    else:
        speed_bounds = speed_gate_bounds(
            reference,
            load_factor=deviation.crossing_load_factor,
            crossing_mass_kg=None if subject == "observed" else deviation.crossing_mass_kg,
        )
        speed_result = _component(judged_speed, speed_bounds.lower_ms, speed_bounds.upper_ms)
    speed_margin: float | None = None
    if speed_bounds is not None and judged_speed is not None:
        inside = speed_bounds.lower_ms <= judged_speed <= speed_bounds.upper_ms
        distance = min(
            abs(judged_speed - speed_bounds.lower_ms),
            abs(judged_speed - speed_bounds.upper_ms),
        )
        speed_margin = distance if inside else -distance
    verdict = _composite(lateral_result, vertical_result, speed_result)
    violations: list[str] = []
    if lateral_result == "fail":
        violations.append("lateral")
    if vertical_result == "fail":
        violations.append("vertical")
    if speed_result == "fail":
        violations.append("speed")
    # Lateral is always decidable once a crossing was measured -- a runway always has
    # a width -- so an indeterminate composite means a missing vertical reference, a
    # missing speed window, or both; name every one that applies.
    reason = None
    if verdict == "indeterminate":
        parts = []
        if vertical_result == "indeterminate":
            parts.append(limits.vertical_reason or "vertical bound or estimate unavailable")
        if speed_reason is not None:
            parts.append(speed_reason)
        reason = "; ".join(parts) or None
    return TrajectoryEvaluation(
        **common, solved=True, success=verdict == "pass", verdict=verdict,
        lateral_result=lateral_result, vertical_result=vertical_result,
        speed_result=speed_result,
        speed_criterion=speed_criterion,
        speed_bounds=speed_bounds,
        speed_reason=speed_reason,
        crossing_airspeed_estimate_ms=airspeed_estimate,
        wind=wind_block,
        speed_margin_ms=speed_margin,
        deviation=deviation, event_status=outcome.event_status,
        violations=tuple(violations), reason=reason,
    )


def evaluate_batch(
    records: Iterable[TrajectoryRecord],
    *,
    contexts: Mapping[ContextKey, AssessmentContext],
    observed_availability: Mapping[str, Any] | None = None,
    winds: Mapping[str, WindTable] | None = None,
) -> dict[str, Any]:
    """Evaluate a batch and serialize every verdict-changing parameter.

    ``winds`` maps airport code to its METAR table; observed records at an airport
    without one are judged on the ground-speed proxy, and the report counts both.
    """
    # Iterated, never materialized: `records` may be a generator over a batch whose
    # resolved states are ~1 MB per flight. Everything retained below (evaluations, rows,
    # comparisons) is per-flight metadata, not trajectory arrays.
    evaluations: list[TrajectoryEvaluation] = []
    rows: list[dict[str, Any]] = []
    comparisons: list[ReferenceComparison] = []
    used: dict[ContextKey, AssessmentContext] = {}
    # The availability block belongs to an observed-only batch; its label is checked
    # before the first record and its subject claim at the first record that breaks
    # it -- not after a stream of tens of thousands has been consumed.
    observed = (
        _observed_availability(observed_availability)
        if observed_availability is not None
        else None
    )
    for record in records:
        context = resolve_context(record, contexts)
        used[(context.airport, context.runway)] = context
        evaluation = evaluate_record(
            record, context=context,
            wind=winds.get(context.airport) if winds is not None else None,
        )
        if observed is not None and evaluation.subject != "observed":
            raise ValueError(
                "observed_availability can be attached only to an observed-only batch; "
                f"record {evaluation.record_id!r} is {evaluation.subject!r}"
            )
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
            # ``comparable`` already implies both MEASURED paths are non-empty; what it
            # does not imply is that either MOVED, and an arc-length resample of a
            # stationary path has nothing to parametrize by. The guard walks the same
            # measured lists the comparison resamples.
            if (
                span.comparable
                and horizontal_arc_length_m(record.measured_states) > 0.0
                and horizontal_arc_length_m(reference.measured_states) > 0.0
            ):
                comparison = compare_to_reference(record, reference, span=span)
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
    # The same tallies per criterion: an observed batch mixes estimate-judged and
    # proxy-judged rows, and a pooled rate cannot be decomposed after the fact.
    speed_result_counts_by_criterion = {
        criterion: {
            key: sum(
                item.speed_result == key and item.speed_criterion == criterion
                for item in evaluations
            )
            for key in ("pass", "fail", "indeterminate")
        }
        for criterion in sorted({item.speed_criterion for item in evaluations})
    }
    # How the observed rows' speed was judged: the METAR-corrected estimate, or the
    # proxy because no usable report existed (a per-row fact, summed here).
    wind_counts = {
        "estimated": sum(item.crossing_airspeed_estimate_ms is not None for item in measured),
        "unavailable": sum(
            item.wind is not None and item.crossing_airspeed_estimate_ms is None
            for item in measured
        ),
    }
    subjects = {item.subject for item in evaluations}
    total = len(evaluations)
    times = [item.deviation.flight_time_s for item in measured]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "methodology": METHODOLOGY,
        "assessment_contexts": [
            {**context.to_dict(), "resolved_limits": context.limits().to_dict()}
            for _key, context in sorted(used.items())
        ],
        # One subject names itself; several are "mixed"; none is "empty" -- a filtered
        # stream that came up empty must not read as two subjects having been present.
        "subject": (
            sorted(subjects)[0] if len(subjects) == 1
            else "mixed" if subjects else "empty"
        ),
        **({"observed": observed} if observed is not None else {}),
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
        "speed_result_counts_by_criterion": speed_result_counts_by_criterion,
        "wind_counts": wind_counts,
        # Why speed could not be judged, per cause, over the SAME rows as
        # speed_result_counts (unmeasured crossings included): coverage of the
        # published table is a fact the batch must state.
        "speed_indeterminate_reasons": _speed_indeterminate_reasons(evaluations),
        # Observed rows' METAR-corrected crossing airspeed estimates (null when none).
        "crossing_airspeed_estimate_ms": magnitude_spread([
            item.crossing_airspeed_estimate_ms
            for item in measured
            if item.crossing_airspeed_estimate_ms is not None
        ]),
        "crossing_speed_ms": magnitude_spread([
            item.deviation.crossing_speed_ms
            for item in measured
            if item.deviation.crossing_speed_ms is not None
        ]),
        # The observed rows' graded ground-speed proxy (METHODOLOGY
        # ["observed_crossing_ground_speed"]); null when no row carries one. Kept apart
        # from crossing_speed_ms: two quantities, two spreads.
        "crossing_ground_speed_ms": magnitude_spread([
            item.deviation.crossing_ground_speed_ms
            for item in measured
            if item.deviation.crossing_ground_speed_ms is not None
        ]),
        # The load factor the crossings were flown at: the spread, how many sat below
        # 1 g (their lower bound was clamped to the 1-g floor), and how many were a
        # declared 1 g rather than a measured one. Null for an empty batch.
        "crossing_load_factor": _load_factor_aggregate(measured),
        "final_time_s": (
            {"mean": fmean(times), "min": min(times), "max": max(times)} if times else None
        ),
        "reference": _reference_aggregate(comparisons),
        "trajectories": rows,
    }


def _speed_indeterminate_reasons(evaluations: list[TrajectoryEvaluation]) -> dict[str, int]:
    """Every speed-indeterminate row carries a reason; the counts sum to the
    indeterminate tally in ``speed_result_counts``."""
    counts: dict[str, int] = {}
    for item in evaluations:
        if item.speed_result == "indeterminate":
            if item.speed_reason is None:
                raise ValueError(f"{item.record_id}: speed indeterminate without a reason")
            counts[item.speed_reason] = counts.get(item.speed_reason, 0) + 1
    return dict(sorted(counts.items()))


def _load_factor_aggregate(
    measured: list[TrajectoryEvaluation],
) -> dict[str, float | int] | None:
    values = [item.deviation.crossing_load_factor for item in measured]
    if not values:
        return None
    return {
        "mean": fmean(values),
        "min": min(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "below_1g": sum(value < MIN_BOUND_LOAD_FACTOR for value in values),
        "adsb_kinematics": sum(
            item.deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_ADSB
            for item in measured
        ),
        "assumed_1g": sum(
            item.deviation.crossing_load_factor_source == LOAD_FACTOR_ASSUMED_1G
            for item in measured
        ),
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
                item.speed_bounds.to_dict(item.speed_criterion)
                if item.speed_bounds is not None
                else {
                    "speed_criterion": item.speed_criterion,
                    "reference_typecode": None,
                    "reference_sources": None,
                    "mass_basis": None,
                    "vref_low_ms": None,
                    "vref_high_ms": None,
                    "speed_lower_ms": None,
                    "speed_upper_ms": None,
                }
            ),
        },
        # Why the speed component is indeterminate (None when it was graded) -- on
        # the row itself, so it is readable when the composite is a lateral/vertical
        # fail and ``reason`` is not written.
        "speed_reason": item.speed_reason,
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
            "crossing_load_factor": deviation.crossing_load_factor,
            "crossing_load_factor_source": deviation.crossing_load_factor_source,
            "crossing_load_factor_window": deviation.crossing_load_factor_window,
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
        # ``crossing_ground_speed_ms`` the gate-graded ground-speed proxy (observed
        # subjects) — different physical quantities, never merged.
        row.update(
            lateral_m=deviation.lateral_m,
            cross_track_m=deviation.cross_track_m,
            along_track_m=deviation.along_track_m,
            vertical_m=deviation.vertical_m,
            speed_ms=deviation.speed_ms,
            crossing_speed_ms=deviation.crossing_speed_ms,
            crossing_ground_speed_ms=deviation.crossing_ground_speed_ms,
            crossing_airspeed_estimate_ms=item.crossing_airspeed_estimate_ms,
            speed_margin_ms=item.speed_margin_ms,
            crossing_load_factor=deviation.crossing_load_factor,
            crossing_load_factor_source=deviation.crossing_load_factor_source,
            heading_rad=deviation.heading_rad,
            final_time_s=deviation.flight_time_s,
        )
    if item.wind is not None:
        row["wind"] = item.wind
    if item.reason is not None:
        row["reason"] = item.reason
    return row
