"""Resolve one policy-free physical runway-threshold event.

There are exactly two estimators.  A source-valid pair that brackets the LTP plane is
interpolated directly in one runway frame.  When the selected inbound pass ends before
that plane, the event reuses the one robust fit that already won runway assignment.
Evaluation consumes the serialized event and never fits trajectory samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from final_approach import (
    AMBIGUITY_MARGIN_M,
    Assignment,
    Projected,
    SegmentFit,
    TrackPoint,
)
from final_approach.event_contract import (
    CENSORED_EVENT_METHOD,
    DIRECT_EVENT_METHOD,
    ESTIMATED_OBSERVABILITY_BY_METHOD,
    EVENT_SCHEMA_VERSION,
    NO_EVENT_METHOD,
    UNAVAILABLE_OBSERVABILITIES,
)
from geokit import haversine_m

from trajectory_data_process.harvest.airports import (
    Runway,
    require_matching_threshold_frame,
    threshold_frame_fingerprint,
    threshold_frame_snapshot,
)
from trajectory_data_process.harvest.tracks import (
    MAX_POSITION_COVERAGE_GAP_S,
    Track,
)

# Source-integrity checks, never approach-performance gates.  The speed tolerances are
# the development-corpus controls documented in the simplified implementation record.
MAX_REPORTED_GROUND_SPEED_M_S = 200.0
MAX_SPEED_DISAGREEMENT_M_S = 25.0
MIN_POSITION_TO_REPORTED_SPEED_RATIO = 0.5
MAX_POSITION_TO_REPORTED_SPEED_RATIO = 1.5
MAX_PARALLEL_COURSE_DELTA_DEG = 5.0

# Continue from the assignment fit to the closest observed support on that same inbound
# pass. This is the same one-sample jitter allowance used by the final-approach fitter;
# comparing to the running maximum prevents a slow reversal accumulating indefinitely.
_INBOUND_TOLERANCE_M = 100.0


@dataclass(frozen=True)
class ThresholdBracket:
    """One source-validated inbound crossing of one physical LTP plane."""

    runway: Runway
    projected: Projected
    source_sample_range: tuple[int, int]
    fraction: float
    event_time_s: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ThresholdBracketSelection:
    """Relative runway choice made only from source-valid structural brackets."""

    outcome: str
    bracket: ThresholdBracket | None
    scores_m: dict[str, float]
    margin_m: float | None
    reason: str | None
    rejections: tuple[dict[str, Any], ...]


def select_observed_threshold_bracket(
    track: Track,
    runways: tuple[Runway, ...] | list[Runway],
    *,
    max_structural_cross_m: float,
    max_structural_height_m: float,
) -> ThresholdBracketSelection:
    """Find a direct event, distinguishing absent support from invalid support."""
    runway_list = tuple(runways)
    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    winners: dict[str, ThresholdBracket] = {}
    rejections: list[dict[str, Any]] = []
    for runway in runway_list:
        structural_cross_m = _runway_bracket_cross_limit(
            runway,
            runway_list,
            fallback_m=max_structural_cross_m,
        )
        candidates, runway_rejections = _validated_brackets_for_runway(
            track,
            points,
            runway,
            max_structural_cross_m=structural_cross_m,
            max_structural_height_m=max_structural_height_m,
        )
        rejections.extend(
            {"runway": runway.ident, **rejection}
            for rejection in runway_rejections
        )
        if candidates:
            winners[runway.ident] = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate.projected.cross_m),
                    -candidate.source_sample_range[1],
                ),
            )

    if not winners:
        invalid_support = any(
            rejection.get("category") == "source_integrity"
            for rejection in rejections
        )
        return ThresholdBracketSelection(
            "invalid_support" if invalid_support else "unavailable",
            None,
            {},
            None,
            (
                "a plausible threshold bracket failed source-integrity checks"
                if invalid_support
                else "no source-valid threshold bracket lies inside the landing structure"
            ),
            tuple(rejections),
        )

    scores = {
        ident: abs(candidate.projected.cross_m)
        for ident, candidate in winners.items()
    }
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    best_ident, best_score = ranked[0]
    margin = ranked[1][1] - best_score if len(ranked) > 1 else None
    if margin is not None and margin < AMBIGUITY_MARGIN_M:
        runner_ident, runner_score = ranked[1]
        return ThresholdBracketSelection(
            "ambiguous",
            None,
            scores,
            margin,
            f"{best_ident} ({best_score:.0f} m) and {runner_ident} "
            f"({runner_score:.0f} m) differ by {margin:.0f} m < "
            f"{AMBIGUITY_MARGIN_M:.0f} m",
            tuple(rejections),
        )
    return ThresholdBracketSelection(
        "assigned",
        winners[best_ident],
        scores,
        margin,
        None,
        tuple(rejections),
    )


def build_observed_threshold_event(
    track: Track,
    runway: Runway | None,
    assignment: Assignment,
    *,
    bracket: ThresholdBracket | None = None,
    bracket_rejections: tuple[dict[str, Any], ...] = (),
    unavailable_observability: str = "unavailable",
) -> dict[str, Any]:
    """Serialize the direct event or reuse the winning censored assignment fit."""
    if runway is None or assignment.runway is None:
        return _unavailable_event(
            None,
            assignment.reason or assignment.outcome,
            observability=unavailable_observability,
            bracket_rejections=bracket_rejections,
        )
    if runway.ident != assignment.runway:
        raise ValueError("threshold-event runway disagrees with assignment")
    if bracket is not None:
        if bracket.runway.ident != runway.ident:
            raise ValueError("threshold bracket runway disagrees with assignment")
        return _direct_event(track, runway, bracket, bracket_rejections)

    fit = assignment.fit
    if fit is None:
        return _unavailable_event(
            runway,
            assignment.reason or "selected final inbound pass has no fittable segment",
            observability=unavailable_observability,
            bracket_rejections=bracket_rejections,
        )
    return _censored_event(track, runway, fit, bracket_rejections)


def require_current_threshold_event(event: dict[str, Any], runway: Runway) -> None:
    """Strictly validate the current discriminated event contract and frame."""
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"track threshold event is not {EVENT_SCHEMA_VERSION}; "
            "run --reclassify-existing"
        )
    status = event.get("status")
    if status not in ("estimated", "unavailable"):
        raise ValueError(
            f"track threshold event has invalid status {status!r}; "
            "run --reclassify-existing"
        )
    if event.get("runway") != runway.ident:
        raise ValueError(
            f"track threshold event is not for runway {runway.ident}; "
            "run --reclassify-existing"
        )
    require_matching_threshold_frame(event, runway)
    if event.get("threshold_frame_snapshot") != threshold_frame_snapshot(runway):
        raise ValueError(
            "track threshold event physical-frame snapshot is stale or inconsistent; "
            "run --reclassify-existing"
        )

    method = event.get("method")
    observability = event.get("observability")
    if status == "unavailable":
        if method != NO_EVENT_METHOD or observability not in UNAVAILABLE_OBSERVABILITIES:
            raise ValueError("unavailable threshold event has an invalid discriminator")
        if not isinstance(event.get("unavailable_reason"), str):
            raise ValueError("unavailable threshold event requires a reason")
        return

    expected_observability = ESTIMATED_OBSERVABILITY_BY_METHOD.get(method)
    if expected_observability is None or observability != expected_observability:
        raise ValueError(
            "estimated threshold event has an invalid method/observability pair; "
            "run --reclassify-existing"
        )
    if event.get("altitude_datum") != "hae":
        raise ValueError("threshold event altitude_datum must be 'hae'")
    _source_range(event.get("source_sample_range"))
    for key in (
        "threshold_crossing_lat",
        "threshold_crossing_lon",
        "threshold_crossing_altitude_m",
        "signed_cross_track_m",
        "extrapolation_distance_m",
    ):
        _finite_event_number(event, key)
    uncertainty = event.get("uncertainty")
    if uncertainty != {"status": "uncalibrated"}:
        raise ValueError("threshold event uncertainty must be explicitly uncalibrated")
    distance = float(event["extrapolation_distance_m"])
    if method == DIRECT_EVENT_METHOD:
        _finite_event_number(event, "event_time_s")
        fraction = _finite_event_number(event, "interpolation_fraction")
        if not 0.0 < fraction <= 1.0 or distance != 0.0:
            raise ValueError("direct threshold event has invalid interpolation geometry")
    elif (
        event.get("event_time_s") is not None
        or event.get("interpolation_fraction") is not None
        or distance <= 0.0
    ):
        raise ValueError("censored threshold event has invalid extrapolation geometry")


def _validated_brackets_for_runway(
    track: Track,
    points: list[TrackPoint],
    runway: Runway,
    *,
    max_structural_cross_m: float,
    max_structural_height_m: float,
) -> tuple[list[ThresholdBracket], list[dict[str, Any]]]:
    frame = runway.frame("hae")
    projected = frame.project_all(points)
    candidates: list[ThresholdBracket] = []
    rejections: list[dict[str, Any]] = []
    for index in range(len(projected) - 1):
        before = projected[index]
        after = projected[index + 1]
        if not (before.along_m < 0.0 <= after.along_m):
            continue

        fraction = -before.along_m / (after.along_m - before.along_m)
        crossing = Projected(
            0.0,
            before.cross_m + fraction * (after.cross_m - before.cross_m),
            before.height_m + fraction * (after.height_m - before.height_m),
        )
        evidence = {
            "source_sample_range": [index, index + 1],
            "interpolation_fraction": fraction,
            "signed_cross_track_m": crossing.cross_m,
            "height_at_threshold_m": crossing.height_m,
        }
        if (
            abs(crossing.cross_m) > max_structural_cross_m
            or abs(crossing.height_m) > max_structural_height_m
        ):
            rejections.append(
                {
                    "category": "structural",
                    "reason": "threshold bracket is outside landing structure",
                    **evidence,
                    "max_structural_cross_m": max_structural_cross_m,
                    "max_structural_height_m": max_structural_height_m,
                }
            )
            continue

        before_sample = track.samples[index]
        after_sample = track.samples[index + 1]
        dt = after_sample.time_s - before_sample.time_s
        integrity_error = _timing_integrity_error(before_sample, after_sample, dt)
        if integrity_error is not None:
            rejections.append(
                {
                    "category": "source_integrity",
                    "reason": integrity_error,
                    **evidence,
                    "position_update_gap_s": dt,
                    "max_position_coverage_gap_s": MAX_POSITION_COVERAGE_GAP_S,
                }
            )
            continue

        before_speed = before_sample.reported_ground_speed_m_s
        after_speed = after_sample.reported_ground_speed_m_s
        if (
            before_speed is None
            or after_speed is None
            or not math.isfinite(before_speed)
            or not math.isfinite(after_speed)
        ):
            rejections.append(
                {
                    "category": "source_integrity",
                    "reason": "ADS-B reported ground speed is unavailable",
                    **evidence,
                    "position_update_gap_s": dt,
                }
            )
            continue
        if (
            before_speed <= 0.0
            or after_speed <= 0.0
            or before_speed > MAX_REPORTED_GROUND_SPEED_M_S
            or after_speed > MAX_REPORTED_GROUND_SPEED_M_S
        ):
            rejections.append(
                {
                    "category": "source_integrity",
                    "reason": "ADS-B reported ground speed is outside the integrity range",
                    **evidence,
                    "reported_ground_speed_before_m_s": before_speed,
                    "reported_ground_speed_after_m_s": after_speed,
                    "max_reported_ground_speed_m_s": MAX_REPORTED_GROUND_SPEED_M_S,
                }
            )
            continue

        distance_m = haversine_m(
            points[index].lat,
            points[index].lon,
            points[index + 1].lat,
            points[index + 1].lon,
        )
        position_speed = distance_m / dt
        reported_speed = (before_speed + after_speed) / 2.0
        speed_error = position_speed - reported_speed
        speed_ratio = position_speed / reported_speed
        diagnostics = {
            "position_update_gap_s": dt,
            "along_track_span_m": after.along_m - before.along_m,
            "position_distance_m": distance_m,
            "position_derived_speed_m_s": position_speed,
            "reported_ground_speed_before_m_s": before_speed,
            "reported_ground_speed_after_m_s": after_speed,
            "reported_ground_speed_mean_m_s": reported_speed,
            "signed_speed_disagreement_m_s": speed_error,
            "position_to_reported_speed_ratio": speed_ratio,
        }
        if (
            abs(speed_error) > MAX_SPEED_DISAGREEMENT_M_S
            or speed_ratio < MIN_POSITION_TO_REPORTED_SPEED_RATIO
            or speed_ratio > MAX_POSITION_TO_REPORTED_SPEED_RATIO
        ):
            rejections.append(
                {
                    "category": "source_integrity",
                    "reason": (
                        "position displacement disagrees with ADS-B reported ground speed"
                    ),
                    **evidence,
                    **diagnostics,
                }
            )
            continue
        candidates.append(
            ThresholdBracket(
                runway=runway,
                projected=crossing,
                source_sample_range=(index, index + 1),
                fraction=fraction,
                event_time_s=(
                    before_sample.time_s + fraction * dt
                ),
                diagnostics=diagnostics,
            )
        )
    return candidates, rejections


def _timing_integrity_error(before: Any, after: Any, dt: float) -> str | None:
    if not math.isfinite(dt) or dt <= 0.0:
        return "source position time is not strictly increasing"
    if dt > MAX_POSITION_COVERAGE_GAP_S:
        return "source position-update coverage gap exceeds 15 s"
    for sample in (before, after):
        position_time = sample.last_position_update_s
        if (
            position_time is None
            or not math.isfinite(position_time)
            or not math.isclose(position_time, sample.time_s, abs_tol=1e-6)
        ):
            return "track sample time is not its ADS-B lastposupdate"
    return None


def _direct_event(
    track: Track,
    runway: Runway,
    bracket: ThresholdBracket,
    rejections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    crossing = runway.frame("hae").unproject(bracket.projected)
    return {
        **_event_common(runway),
        "status": "estimated",
        "observability": "within_observed_support",
        "method": DIRECT_EVENT_METHOD,
        "event_time_s": bracket.event_time_s,
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": bracket.projected.cross_m,
        "source_sample_range": list(bracket.source_sample_range),
        "interpolation_fraction": bracket.fraction,
        "extrapolation_distance_m": 0.0,
        "uncertainty": {"status": "uncalibrated"},
        "source_integrity": _source_integrity(track, bracket.diagnostics),
        "diagnostics": {
            "bracket": {
                "height_at_threshold_m": bracket.projected.height_m,
                **bracket.diagnostics,
            },
            "bracket_rejections": list(rejections),
        },
    }


def _censored_event(
    track: Track,
    runway: Runway,
    fit: SegmentFit,
    rejections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if fit.runway != runway.ident:
        raise ValueError("assignment fit runway disagrees with threshold event")
    support = _right_censored_support(track, runway, fit)
    if support is None:
        return _unavailable_event(
            runway,
            "selected pass does not remain strictly before the threshold plane",
            observability="invalid_support",
            bracket_rejections=rejections,
        )
    support_index, support_along_m = support
    crossing = runway.frame("hae").unproject(
        Projected(0.0, fit.cross_at_threshold_m, fit.height_at_threshold_m)
    )
    return {
        **_event_common(runway),
        "status": "estimated",
        "observability": "right_censored",
        "method": CENSORED_EVENT_METHOD,
        "event_time_s": None,
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": fit.cross_at_threshold_m,
        "source_sample_range": [fit.first_sample_index, fit.last_sample_index],
        "interpolation_fraction": None,
        "extrapolation_distance_m": -support_along_m,
        "uncertainty": {"status": "uncalibrated"},
        "source_integrity": _source_integrity(track, None),
        "diagnostics": {
            "fit": _fit_audit(fit),
            "closest_support_sample_index": support_index,
            "closest_support_along_m": support_along_m,
            "bracket_rejections": list(rejections),
        },
    }


def _right_censored_support(
    track: Track,
    runway: Runway,
    fit: SegmentFit,
) -> tuple[int, float] | None:
    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    projected = runway.frame("hae").project_all(points)
    if not 0 <= fit.last_sample_index < len(projected):
        raise ValueError("assignment fit ends outside the source track")
    support_index = fit.last_sample_index
    support_along_m = projected[support_index].along_m
    for index in range(fit.last_sample_index + 1, len(projected)):
        along_m = projected[index].along_m
        if along_m < support_along_m - _INBOUND_TOLERANCE_M:
            break
        if along_m > support_along_m:
            support_index = index
            support_along_m = along_m
    if support_along_m >= 0.0:
        return None
    return support_index, support_along_m


def _unavailable_event(
    runway: Runway | None,
    reason: str,
    *,
    observability: str,
    bracket_rejections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if observability not in UNAVAILABLE_OBSERVABILITIES:
        raise ValueError(f"invalid unavailable observability {observability!r}")
    common = (
        {"schema_version": EVENT_SCHEMA_VERSION}
        if runway is None
        else _event_common(runway)
    )
    return {
        **common,
        "status": "unavailable",
        "observability": observability,
        "method": NO_EVENT_METHOD,
        "unavailable_reason": reason,
        "diagnostics": {"bracket_rejections": list(bracket_rejections)},
    }


def _event_common(runway: Runway) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "runway": runway.ident,
        "threshold_frame_snapshot": threshold_frame_snapshot(runway),
        "threshold_frame_fingerprint": threshold_frame_fingerprint(runway),
    }


def _source_integrity(
    track: Track,
    bracket_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "accepted",
        "position_time_basis": "lastposupdate",
        "track_schema": (
            track.source_integrity.schema_version
            if track.source_integrity is not None
            else None
        ),
        "bracket_speed_check": bracket_diagnostics,
        "limits": {
            "max_position_coverage_gap_s": MAX_POSITION_COVERAGE_GAP_S,
            "max_reported_ground_speed_m_s": MAX_REPORTED_GROUND_SPEED_M_S,
            "max_speed_disagreement_m_s": MAX_SPEED_DISAGREEMENT_M_S,
            "min_position_to_reported_speed_ratio": (
                MIN_POSITION_TO_REPORTED_SPEED_RATIO
            ),
            "max_position_to_reported_speed_ratio": (
                MAX_POSITION_TO_REPORTED_SPEED_RATIO
            ),
        },
    }


def _fit_audit(fit: SegmentFit) -> dict[str, Any]:
    return {
        "window_m": list(fit.window_m),
        "source_sample_range": [fit.first_sample_index, fit.last_sample_index],
        "sample_count": fit.n_samples,
        "along_track_span_m": fit.span_m,
        "nearest_fit_sample_along_m": fit.nearest_sample_along_m,
        "signed_cross_track_m": fit.cross_at_threshold_m,
        "height_at_threshold_m": fit.height_at_threshold_m,
        "glidepath_deg": fit.glidepath_deg,
        "median_abs_cross_track_m": fit.median_abs_cross_m,
        "cross_track_fit": _line_audit(fit.cross),
        "altitude_fit": _line_audit(fit.height),
        "rejected_sample_indices": list(fit.rejected_sample_indices),
    }


def _line_audit(line: Any) -> dict[str, float]:
    return {
        "rms_residual_m": line.rms_residual_m,
        "max_abs_residual_m": line.max_abs_residual_m,
    }


def _source_range(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[0] < 0
        or value[1] < value[0]
    ):
        raise ValueError("threshold event has an invalid source_sample_range")
    return value[0], value[1]


def _finite_event_number(event: dict[str, Any], key: str) -> float:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"threshold event {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"threshold event {key} must be finite")
    return number


def _course_difference_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _runway_bracket_cross_limit(
    runway: Runway,
    runways: tuple[Runway, ...],
    *,
    fallback_m: float,
) -> float:
    """Keep each parallel-runway bracket inside its half of the pair."""
    frame = runway.frame("hae")
    half_separations = []
    for other in runways:
        if other.ident == runway.ident or _course_difference_deg(
            runway.course_deg,
            other.course_deg,
        ) > MAX_PARALLEL_COURSE_DELTA_DEG:
            continue
        other_threshold = frame.project(
            TrackPoint(other.lat, other.lon, other.elevation_hae_m)
        )
        separation_m = abs(other_threshold.cross_m)
        if separation_m > 0.0:
            half_separations.append(separation_m / 2.0)
    return min([fallback_m, *half_separations])
