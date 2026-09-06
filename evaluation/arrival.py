"""Resolve the single runway-threshold event without fitting trajectories.

Computed and predicted records use their terminal state.  Observed records
consume the policy-free ``observed_threshold_event`` produced by runway
assignment.  This module never selects ADS-B samples and never calls the final
approach fitter.

Every function here takes a SOLVED record (non-empty ``states``, hence a
``target_state``): ``evaluation.records`` enforces that pairing at the file
boundary and ``evaluate_record`` filters the unsolved ones out before calling in,
so the shape is a precondition rather than something re-checked per record. The
event payload's own schema is validated once, by the seam both packages share
(``final_approach.event_contract.validate_event``); what stays here is only the
two bindings evaluation alone can make -- the event's runway and physical frame
against the assessment context, and the record's datum offset and target altitude
against the authoritative runway data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from final_approach.crossing import (
    CROSSING_SPAN_KEY,
    MEASURED_BRACKET_KIND,
    bracket_fraction,
    interpolate_channels,
)
from final_approach.event_contract import validate_event
from final_approach.frame import RunwayFrame, TrackPoint
from geokit import haversine_m

from evaluation.records import STATE_KEYS, TrajectoryRecord
from evaluation.speed_gate import LOAD_FACTOR_ASSUMED_1G, LOAD_FACTOR_FROM_CONTROLS
from evaluation.thresholds import AssessmentContext

# Every state channel a crossing interpolation blends (the record contract's keys plus
# the clock); ψ is the one that wraps.
_TIMED_STATE_KEYS = ("t", *STATE_KEYS)
_ANGULAR_STATE_KEYS = ("psi",)
# The control column the speed gate reads (evaluation_export.CONTROL_DECIMALS names the
# same three: thrust, bank_rad, load_factor).
LOAD_FACTOR_KEY = "load_factor"

TERMINAL_PLANE_TOLERANCE_M = 1.0
TARGET_CONTEXT_TOLERANCE_M = 0.01

# ``source.target_source`` value meaning "this scenario aimed at the published landing
# threshold". The other modes (``fitted_adsb_crossing``, ``track_end``) aim elsewhere
# by design, so only this one can be cross-checked against the runway data.
THRESHOLD_TARGET_SOURCE = "runway_threshold"


@dataclass(frozen=True)
class ArrivalDeviation:
    along_track_m: float
    cross_track_m: float
    vertical_m: float | None
    speed_ms: float
    heading_rad: float
    flight_time_s: float
    # Absolute state AT the graded event, for the speed gate. Mass is the crossing
    # state's own on every subject (the observed record carries its resolved airframe's
    # landing mass). The crossing AIRSPEED is the computed path's state V; it stays
    # None on observed rows, whose V is ground-speed derived — the gate judges those on
    # ``crossing_ground_speed_ms`` below as a stated proxy (speed_gate.OBSERVED_SPEED_POLICY).
    crossing_mass_kg: float
    # The load factor the crossing was flown at: the control active over the final
    # rollout step when the record carries controls, else a declared 1 g — the
    # ``source`` names which (speed_gate.LOAD_FACTOR_*). The gate's lower bound is the
    # stall speed at this n (docs/THRESHOLD_SPEED_GATE.md section 3.5).
    crossing_load_factor: float
    crossing_load_factor_source: str
    extrapolation_m: float | None = None
    crossing_speed_ms: float | None = None
    # The event's estimated GROUND speed at the crossing (ADS-B velocity source;
    # interpolated at a direct bracket, OLS-extrapolated for a censored fit). Observed
    # subjects only; None on events serialized before 2026-08-24, and the quantity the
    # observed speed gate judges — under its own criterion id, because wind is
    # unmodelled and it is not an airspeed.
    crossing_ground_speed_ms: float | None = None

    @property
    def lateral_m(self) -> float:
        return abs(self.cross_track_m)


@dataclass(frozen=True)
class ArrivalOutcome:
    deviation: ArrivalDeviation | None
    event_status: str
    reason: str | None = None


def arrival_deviation(
    record: TrajectoryRecord,
    *,
    context: AssessmentContext,
) -> ArrivalOutcome:
    # ``record_from_dict`` already rejected any subject outside ``SUBJECTS``.
    if record.source["subject"] == "observed":
        return _observed_arrival(record, context)
    return _computed_arrival(record, context)


def _require_target_agrees_with_runway_data(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> None:
    """A record aiming at the published threshold must agree on where it is.

    ``target_source`` names what the scenario aimed at, and only ``runway_threshold``
    claims the published point: the fitted-ADS-B and track-end target modes aim
    somewhere else on purpose and have nothing to cross-check. A record that declares
    no source is held to the strict reading -- an unlabelled target gets MORE
    checking, never a bypass.

    Both coordinates are checked, and until now only one was. The altitude had to sit
    on the published LTP+TCH plane while the POSITION was taken from the artifact
    unexamined -- which is precisely the shape of the displaced-threshold bug: a
    target 775 m from the published landing threshold yields clean near-zero
    deviations and shows no symptom anywhere downstream.
    """
    # ``or``, not ``get(key, default)``: a key PRESENT with a null value returns None
    # from the latter, which would take the early return -- bypassing the check for
    # exactly the record that declared nothing, the opposite of what is promised above.
    if (record.source.get("target_source") or THRESHOLD_TARGET_SOURCE) != THRESHOLD_TARGET_SOURCE:
        return
    target = record.target_state
    offset_m = haversine_m(
        target["lat"], target["lon"], context.threshold_lat, context.threshold_lon
    )
    if offset_m > TARGET_CONTEXT_TOLERANCE_M:
        raise ValueError(
            f"target_state lies {offset_m:.3f} m from the authoritative "
            f"{context.airport} {context.runway} landing threshold; the record was "
            "built against different runway data"
        )
    desired = context.desired_threshold_altitude_msl_m
    if desired is None:
        return
    supplied = float(target["alt"])
    if not math.isclose(
        supplied,
        desired,
        rel_tol=0.0,
        abs_tol=TARGET_CONTEXT_TOLERANCE_M,
    ):
        raise ValueError(
            f"target_state.alt {supplied:.6f} m disagrees with authoritative "
            f"LTP elevation + published TCH {desired:.6f} m"
        )


def _authoritative_frame(
    context: AssessmentContext, target: dict[str, float]
) -> RunwayFrame:
    """The runway frame every deviation is measured in: the CONTEXT's threshold.

    Not one the artifact chose -- a record whose target is the fitted crossing of its
    own flight would otherwise be graded against itself and score a near-zero deviation
    by construction. The plane's elevation is the published LTP+TCH altitude, or the
    target's when the context publishes none (vertical then grades indeterminate).
    """
    desired = context.desired_threshold_altitude_msl_m
    return RunwayFrame(
        ident=context.runway,
        lat=context.threshold_lat,
        lon=context.threshold_lon,
        elevation_m=desired if desired is not None else float(target["alt"]),
        course_deg=context.runway_course_deg,
    )


def _crossing_load_factor(record: TrajectoryRecord) -> tuple[float, str]:
    """``(n, source)`` for the crossing: the control active over the final step.

    The rollout writes on every sample the control that PRODUCED it
    (``aerodynamic_model.rollout``), so ``controls[-1]`` is the one in force over the
    last segment -- where both the terminal-state and the bracket-interpolated crossing
    lie. A record with controls but no load-factor column is malformed (the control
    contract is thrust / bank_rad / load_factor); a record with no controls at all
    (observed baselines, state-output predictions) is judged at a declared 1 g.
    """
    if not record.controls:
        return 1.0, LOAD_FACTOR_ASSUMED_1G
    try:
        return float(record.controls[-1][LOAD_FACTOR_KEY]), LOAD_FACTOR_FROM_CONTROLS
    except KeyError:
        raise ValueError(
            f"record {record.path or record.source.get('id')!r} carries controls "
            f"without a {LOAD_FACTOR_KEY!r} column; the speed gate cannot anchor on "
            "the crossing load factor"
        ) from None


def _computed_arrival(
    record: TrajectoryRecord,
    context: AssessmentContext,
    *,
    plane_tolerance_m: float = TERMINAL_PLANE_TOLERANCE_M,
) -> ArrivalOutcome:
    """Use the final state, or interpolate only the final bracketing segment."""
    target = record.target_state
    _require_target_agrees_with_runway_data(record, context)
    desired_altitude_msl_m = context.desired_threshold_altitude_msl_m
    frame = _authoritative_frame(context, target)
    final = record.states[-1]
    final_projected = frame.project(
        TrackPoint(final["lat"], final["lon"], final["alt"])
    )
    if abs(final_projected.along_m) <= plane_tolerance_m:
        # No interpolation needed; the final state is already within the threshold plane tolerance.
        return ArrivalOutcome(
            _state_deviation(
                final,
                target,
                frame,
                desired_altitude_msl_m=desired_altitude_msl_m,
                *_crossing_load_factor(record),
            ),
            "terminal_state",
        )
    if len(record.states) >= 2:
        # The final segment may bracket the threshold plane, so interpolate to the crossing point.
        previous = record.states[-2]
        previous_projected = frame.project(
            TrackPoint(previous["lat"], previous["lon"], previous["alt"])
        )
        if previous_projected.along_m <= 0.0 <= final_projected.along_m:
            crossing = interpolate_channels(
                previous,
                final,
                bracket_fraction(previous_projected.along_m, final_projected.along_m),
                keys=_TIMED_STATE_KEYS,
                angular_keys=_ANGULAR_STATE_KEYS,
            )
            return ArrivalOutcome(
                _state_deviation(
                    crossing,
                    target,
                    frame,
                    desired_altitude_msl_m=desired_altitude_msl_m,
                    *_crossing_load_factor(record),
                ),
                "interpolated_threshold",
            )
    if final_projected.along_m < 0.0:
        return ArrivalOutcome(
            None, "not_reached",
            f"trajectory ended {abs(final_projected.along_m):.1f} m before the threshold plane",
        )
    return ArrivalOutcome(
        None, "threshold_not_bracketed",
        "trajectory ended beyond the threshold but its final segment does not bracket the plane",
    )


def _state_deviation(
    state: dict[str, float],
    target: dict[str, float],
    frame: RunwayFrame,
    load_factor: float,
    load_factor_source: str,
    *,
    desired_altitude_msl_m: float | None,
) -> ArrivalDeviation:
    projected = frame.project(TrackPoint(state["lat"], state["lon"], state["alt"]))
    return ArrivalDeviation(
        along_track_m=projected.along_m,
        cross_track_m=projected.cross_m,
        vertical_m=(
            None
            if desired_altitude_msl_m is None
            else state["alt"] - desired_altitude_msl_m
        ),
        speed_ms=state["V"] - target["V"],
        heading_rad=math.remainder(state["psi"] - target["psi"], math.tau),
        flight_time_s=state["t"],
        crossing_speed_ms=state["V"],
        crossing_mass_kg=state["m"],
        crossing_load_factor=load_factor,
        crossing_load_factor_source=load_factor_source,
    )


def _observed_arrival(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> ArrivalOutcome:
    """Resolve the observed crossing from the record's own states via its marker.

    The record says WHERE its crossing lives (``source.crossing_span``: the
    instrument-selected measured bracket, or the appended fitted-tail row), and
    grading is the same state interpolation the computed path uses — the datum
    conversion and the crossing estimation both happened at the producing seam,
    never here. The event stays on the record for identity/staleness validation
    and audit; it is no longer the graded payload.

    A record with no event is a legitimate input, not a broken one: optimizer and
    ts_transformer reference tracks share the ``observed`` subject without ever
    passing through runway assignment. They report ``unavailable`` and grade
    indeterminate, which is what "we never measured a crossing for this" means.
    """
    event = record.source.get("observed_threshold_event")
    if event is None:
        return ArrivalOutcome(None, "unavailable", "observed threshold event missing")
    # Identity first, payload second -- same order as the producer's own check, so a
    # stale artifact is reported as stale rather than as a field complaint.
    if event.get("runway") != context.runway:
        raise ValueError(
            "source.observed_threshold_event.runway disagrees with assessment context"
        )
    if context.threshold_frame_fingerprint is not None and (
        event.get("threshold_frame_fingerprint") != context.threshold_frame_fingerprint
    ):
        raise ValueError(
            "source.observed_threshold_event physical frame disagrees with "
            "assessment context; run --reclassify-existing"
        )
    if validate_event(event) == "unavailable":
        return ArrivalOutcome(None, event["observability"], event["unavailable_reason"])

    marker = record.source.get(CROSSING_SPAN_KEY)
    if marker is None:
        raise ValueError(
            "record carries an estimated threshold event but no crossing_span; "
            "rebuild the observed records (--evaluate-only)"
        )
    # The record's states are MSL by the producing seam's conversion; the
    # cross-check below still guards the ~33 m datum class without re-applying it.
    _authoritative_datum_offset(record, context)
    target = record.target_state
    _require_target_agrees_with_runway_data(record, context)
    crossing = _marker_crossing(record.states, marker)
    deviation = _state_deviation(
        crossing,
        target,
        _authoritative_frame(context, target),
        # An observed record carries no controls: the gate judges it at a declared 1 g
        # (a censored crossing is a straight-line fit, and an ADS-B bracket's rates are
        # quantisation noise -- docs/THRESHOLD_SPEED_GATE.md section 3.5).
        1.0,
        LOAD_FACTOR_ASSUMED_1G,
        desired_altitude_msl_m=context.desired_threshold_altitude_msl_m,
    )
    return ArrivalOutcome(
        replace(
            deviation,
            # The flight time of the MEASURED trajectory (the record contract pins
            # final_time_s to the last measured row), not the estimated crossing
            # time an appended tail row carries.
            flight_time_s=float(record.final_time_s),
            extrapolation_m=event["extrapolation_distance_m"],
            # No crossing AIRSPEED was measured (the state's V is ground-speed derived);
            # the gate judges observed subjects on the event's GROUND speed as a stated
            # proxy, anchored on the record's resolved-airframe crossing mass.
            crossing_speed_ms=None,
            crossing_ground_speed_ms=event.get("crossing_ground_speed_m_s"),
        ),
        "estimated",
    )


def _marker_crossing(
    states: list[dict[str, float]], marker: dict[str, Any]
) -> dict[str, float]:
    """The crossing state the record's own marker names — interpolated or appended."""
    if marker["kind"] == MEASURED_BRACKET_KIND:
        left = states[marker["left_index"]]
        return interpolate_channels(
            left,
            states[marker["left_index"] + 1],
            float(marker["fraction"]),
            keys=_TIMED_STATE_KEYS,
            angular_keys=_ANGULAR_STATE_KEYS,
        )
    return states[marker["start_index"]]


def _authoritative_datum_offset(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> float:
    """The HAE-MSL undulation, agreed between the record and the runway data."""
    supplied = record.source["hae_minus_msl_m"]
    authoritative = context.hae_minus_msl_m
    if authoritative is None:
        raise ValueError("assessment context requires authoritative HAE and MSL elevations")
    if not math.isclose(
        float(supplied), authoritative, rel_tol=0.0, abs_tol=TARGET_CONTEXT_TOLERANCE_M
    ):
        raise ValueError(
            f"source.hae_minus_msl_m {float(supplied):.6f} m disagrees with authoritative "
            f"threshold datum offset {authoritative:.6f} m"
        )
    return authoritative
