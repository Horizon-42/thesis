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
from dataclasses import dataclass

from final_approach.event_contract import validate_event
from final_approach.frame import RunwayFrame, TrackPoint
from geokit import haversine_m

from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext

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
    extrapolation_m: float | None = None
    # Absolute state AT the graded event, for the speed gate: the computed path fills
    # both from the crossing state; observed leaves them None — the event estimators
    # extrapolate POSITION only, so no observed crossing speed or mass was measured
    # (the ``speed_ms`` DEVIATION above already quotes the last sample for observed).
    crossing_speed_ms: float | None = None
    crossing_mass_kg: float | None = None

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
    # Measured in the AUTHORITATIVE runway frame, not in one the artifact chose. A
    # record whose target is the fitted crossing of its own flight would otherwise be
    # graded against itself and score a near-zero deviation by construction.
    frame = RunwayFrame(
        ident=context.runway,
        lat=context.threshold_lat,
        lon=context.threshold_lon,
        elevation_m=(
            desired_altitude_msl_m
            if desired_altitude_msl_m is not None
            else float(target["alt"])
        ),
        course_deg=context.runway_course_deg,
    )
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
            span = final_projected.along_m - previous_projected.along_m
            fraction = -previous_projected.along_m / span
            crossing = {
                key: previous[key] + (final[key] - previous[key]) * fraction
                for key in ("t", "lat", "lon", "alt", "V", "psi", "gamma", "m")
            }
            crossing["psi"] = previous["psi"] + math.remainder(
                final["psi"] - previous["psi"], math.tau
            ) * fraction
            return ArrivalOutcome(
                _state_deviation(
                    crossing,
                    target,
                    frame,
                    desired_altitude_msl_m=desired_altitude_msl_m,
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
    )


def _observed_arrival(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> ArrivalOutcome:
    """Read the producer's serialized crossing; never reselect or refit samples.

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

    target, final = record.target_state, record.states[-1]
    _require_target_agrees_with_runway_data(record, context)
    desired_altitude_msl_m = context.desired_threshold_altitude_msl_m
    vertical_m = None
    if desired_altitude_msl_m is not None:
        # The event is HAE as broadcast; the desired crossing is MSL. Both the record
        # and the context carry the undulation, and disagreement between them is a
        # ~33 m error with no other symptom -- so they are cross-checked, not merged.
        geoid = _authoritative_datum_offset(record, context)
        crossing_alt_msl = event["threshold_crossing_altitude_m"] - geoid
        vertical_m = crossing_alt_msl - desired_altitude_msl_m
    return ArrivalOutcome(
        ArrivalDeviation(
            # The event is evaluated at the threshold plane by construction.
            along_track_m=0.0,
            cross_track_m=event["signed_cross_track_m"],
            vertical_m=vertical_m,
            speed_ms=final["V"] - target["V"],
            heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
            flight_time_s=final["t"],
            extrapolation_m=event["extrapolation_distance_m"],
        ),
        "estimated",
    )


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
