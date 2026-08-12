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

from final_approach import RunwayFrame, TrackPoint

from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext

Subject = Literal["optimized", "predicted", "observed"]
TERMINAL_PLANE_TOLERANCE_M = 1.0


@dataclass(frozen=True)
class ArrivalDeviation:
    along_track_m: float
    cross_track_m: float
    vertical_m: float
    speed_ms: float
    heading_rad: float
    flight_time_s: float
    event_status: str
    extrapolated: bool
    lateral_sigma_m: float | None = None
    vertical_sigma_m: float | None = None
    glidepath_deg: float | None = None
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
    frame = RunwayFrame(
        ident=context.runway,
        lat=target["lat"], lon=target["lon"], elevation_m=target["alt"],
        course_deg=context.runway_course_deg,
    )
    final = record.states[-1]
    final_projected = frame.project(
        TrackPoint(final["lat"], final["lon"], final["alt"])
    )
    if abs(final_projected.along_m) <= plane_tolerance_m:
        return ArrivalOutcome(
            _state_deviation(final, target, frame, event_status="terminal_state"),
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
                    crossing, target, frame, event_status="interpolated_threshold"
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


def final_state_deviation(
    record: TrajectoryRecord,
    *,
    context: AssessmentContext,
) -> ArrivalDeviation:
    """Measure a computed terminal state in a runway-aligned frame."""
    if not record.states or record.target_state is None:
        raise ValueError("final-state deviation requires a solved record and target_state")
    final, target = record.states[-1], record.target_state
    frame = RunwayFrame(
        ident=context.runway,
        lat=target["lat"],
        lon=target["lon"],
        elevation_m=target["alt"],
        course_deg=context.runway_course_deg,
    )
    return _state_deviation(final, target, frame, event_status="terminal_state")


def _state_deviation(
    state: dict[str, float],
    target: dict[str, float],
    frame: RunwayFrame,
    *,
    event_status: str,
) -> ArrivalDeviation:
    projected = frame.project(TrackPoint(state["lat"], state["lon"], state["alt"]))
    return ArrivalDeviation(
        along_track_m=projected.along_m,
        cross_track_m=projected.cross_m,
        vertical_m=state["alt"] - target["alt"],
        speed_ms=state["V"] - target["V"],
        heading_rad=math.remainder(state["psi"] - target["psi"], math.tau),
        flight_time_s=state["t"],
        event_status=event_status,
        extrapolated=False,
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
    status = event.get("status")
    if status != "estimated":
        reason = event.get("unavailable_reason")
        return ArrivalOutcome(
            None,
            str(status or "unavailable"),
            str(reason or "observed threshold event unavailable"),
        )
    if event.get("runway") != context.runway:
        raise ValueError(
            "source.observed_threshold_event.runway disagrees with assessment context"
        )
    if event.get("altitude_datum") != "hae":
        raise ValueError("observed threshold event altitude_datum must be 'hae'")
    geoid = record.source.get("hae_minus_msl_m")
    if isinstance(geoid, bool) or not isinstance(geoid, (int, float)) \
            or not math.isfinite(float(geoid)):
        raise ValueError("observed record requires finite source.hae_minus_msl_m")
    if record.target_state is None or not record.states:
        raise ValueError("observed threshold event requires a solved record and target_state")

    target, final = record.target_state, record.states[-1]
    crossing_alt_msl = _event_number(event, "threshold_crossing_altitude_m") - float(geoid)
    return ArrivalOutcome(
        ArrivalDeviation(
            # The event is evaluated at the threshold plane by construction.
            along_track_m=0.0,
            cross_track_m=_event_number(event, "signed_cross_track_m"),
            vertical_m=crossing_alt_msl - target["alt"],
            speed_ms=final["V"] - target["V"],
            heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
            flight_time_s=final["t"],
            event_status="estimated",
            extrapolated=True,
            lateral_sigma_m=_event_number(event, "cross_track_sigma_m", nonnegative=True),
            vertical_sigma_m=_event_number(event, "altitude_sigma_m", nonnegative=True),
            glidepath_deg=(
                _event_number(event, "glidepath_deg")
                if event.get("glidepath_deg") is not None else None
            ),
            extrapolation_m=_event_number(event, "extrapolation_m", nonnegative=True),
        ),
        "estimated",
    )
