"""Resolve the single runway-threshold event without fitting trajectories.

Computed and predicted records use their terminal state.  Observed records
consume the policy-free ``observed_threshold_event`` produced by runway
assignment.  This module never selects ADS-B samples and never calls the final
approach fitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from final_approach.event_contract import (
    CENSORED_EVENT_METHOD,
    DIRECT_EVENT_METHOD,
    ESTIMATED_OBSERVABILITY_BY_METHOD,
    EVENT_SCHEMA_VERSION,
    NO_EVENT_METHOD,
    UNAVAILABLE_OBSERVABILITIES,
)
from final_approach.frame import RunwayFrame, TrackPoint

from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext

Subject = Literal["optimized", "predicted", "observed"]
TERMINAL_PLANE_TOLERANCE_M = 1.0
TARGET_CONTEXT_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class ArrivalDeviation:
    along_track_m: float
    cross_track_m: float
    vertical_m: float | None
    speed_ms: float
    heading_rad: float
    flight_time_s: float
    extrapolation_m: float | None = None

    @property
    def lateral_m(self) -> float:
        return abs(self.cross_track_m)


@dataclass(frozen=True)
class ArrivalOutcome:
    deviation: ArrivalDeviation | None
    event_status: str
    reason: str | None = None


def subject_of(record: TrajectoryRecord) -> Subject:
    subject = record.source.get("subject")
    if subject not in ("optimized", "predicted", "observed"):
        raise ValueError(
            "source.subject must be 'optimized', 'predicted', or 'observed'; "
            f"got {subject!r}"
        )
    return subject


def arrival_deviation(
    record: TrajectoryRecord,
    *,
    context: AssessmentContext,
) -> ArrivalOutcome:
    if subject_of(record) == "observed":
        return _observed_arrival(record, context)
    return _computed_arrival(record, context)


def _authoritative_target_altitude(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> float | None:
    if record.target_state is None:
        raise ValueError("solved trajectory requires target_state")
    desired = context.desired_threshold_altitude_msl_m
    if desired is None:
        return None
    supplied = float(record.target_state["alt"])
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
    return desired


def _computed_arrival(
    record: TrajectoryRecord,
    context: AssessmentContext,
    *,
    plane_tolerance_m: float = TERMINAL_PLANE_TOLERANCE_M,
) -> ArrivalOutcome:
    """Use the final state, or interpolate only the final bracketing segment."""
    if not record.states or record.target_state is None:
        raise ValueError("computed arrival requires a solved record and target_state")
    target = record.target_state
    desired_altitude_msl_m = _authoritative_target_altitude(record, context)
    frame = RunwayFrame(
        ident=context.runway, lat=target["lat"], lon=target["lon"],
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
    )


def _event_number(event: dict, key: str, *, nonnegative: bool = False) -> float:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"source.observed_threshold_event.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"source.observed_threshold_event.{key} must be finite")
    if nonnegative and number < 0.0:
        raise ValueError(f"source.observed_threshold_event.{key} must be non-negative")
    return number


def _observed_arrival(
    record: TrajectoryRecord,
    context: AssessmentContext,
) -> ArrivalOutcome:
    event = record.source.get("observed_threshold_event")
    if not isinstance(event, dict):
        return ArrivalOutcome(None, "unavailable", "observed threshold event missing")
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"observed threshold event must use {EVENT_SCHEMA_VERSION}; "
            "run --reclassify-existing"
        )
    if event.get("runway") != context.runway:
        raise ValueError(
            "source.observed_threshold_event.runway disagrees with assessment context"
        )
    frame_fingerprint = context.threshold_frame_fingerprint
    if frame_fingerprint is not None and (
        event.get("threshold_frame_fingerprint") != frame_fingerprint
    ):
        raise ValueError(
            "source.observed_threshold_event physical frame disagrees with "
            "assessment context; run --reclassify-existing"
        )
    status = event.get("status")
    if status == "unavailable":
        if (
            event.get("method") != NO_EVENT_METHOD
            or event.get("observability") not in UNAVAILABLE_OBSERVABILITIES
        ):
            raise ValueError("unavailable observed threshold event has invalid discriminators")
        reason = event.get("unavailable_reason")
        return ArrivalOutcome(
            None,
            str(event["observability"]),
            str(reason or "observed threshold event unavailable"),
        )
    if status != "estimated":
        raise ValueError(f"observed threshold event has invalid status {status!r}")
    method = event.get("method")
    observability = event.get("observability")
    expected_observability = ESTIMATED_OBSERVABILITY_BY_METHOD.get(method)
    if expected_observability is None or observability != expected_observability:
        raise ValueError(
            "unsupported observed threshold-event method/observability "
            f"{method!r}/{observability!r}"
        )
    source_range = event.get("source_sample_range")
    if (
        not isinstance(source_range, list)
        or len(source_range) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in source_range
        )
        or source_range[0] < 0
        or source_range[1] < source_range[0]
    ):
        raise ValueError("observed threshold event has invalid source_sample_range")
    if event.get("altitude_datum") != "hae":
        raise ValueError("observed threshold event altitude_datum must be 'hae'")
    if record.target_state is None or not record.states:
        raise ValueError("observed threshold event requires a solved record and target_state")

    target, final = record.target_state, record.states[-1]
    desired_altitude_msl_m = _authoritative_target_altitude(record, context)
    vertical_m = None
    if desired_altitude_msl_m is not None:
        geoid = record.source.get("hae_minus_msl_m")
        if isinstance(geoid, bool) or not isinstance(geoid, (int, float)) \
                or not math.isfinite(float(geoid)):
            raise ValueError("observed record requires finite source.hae_minus_msl_m")
        authoritative_geoid = context.hae_minus_msl_m
        if authoritative_geoid is None:
            raise ValueError("assessment context requires authoritative HAE and MSL elevations")
        if not math.isclose(
            float(geoid),
            authoritative_geoid,
            rel_tol=0.0,
            abs_tol=TARGET_CONTEXT_TOLERANCE_M,
        ):
            raise ValueError(
                f"source.hae_minus_msl_m {float(geoid):.6f} m disagrees with authoritative "
                f"threshold datum offset {authoritative_geoid:.6f} m"
            )
        crossing_alt_msl = (
            _event_number(event, "threshold_crossing_altitude_m") - authoritative_geoid
        )
        vertical_m = crossing_alt_msl - desired_altitude_msl_m
    if event.get("uncertainty") != {"status": "uncalibrated"}:
        raise ValueError(
            "source.observed_threshold_event uncertainty must be explicitly uncalibrated"
        )
    extrapolation_m = _event_number(
        event, "extrapolation_distance_m", nonnegative=True
    )
    if method == DIRECT_EVENT_METHOD and extrapolation_m != 0.0:
        raise ValueError("direct observed threshold event cannot be extrapolated")
    if method == CENSORED_EVENT_METHOD and extrapolation_m <= 0.0:
        raise ValueError("censored observed threshold event requires positive extrapolation")
    return ArrivalOutcome(
        ArrivalDeviation(
            # The event is evaluated at the threshold plane by construction.
            along_track_m=0.0,
            cross_track_m=_event_number(event, "signed_cross_track_m"),
            vertical_m=vertical_m,
            speed_ms=final["V"] - target["V"],
            heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
            flight_time_s=final["t"],
            extrapolation_m=extrapolation_m,
        ),
        "estimated",
    )
