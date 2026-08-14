"""Produce one policy-free observed crossing event during runway classification.

Evaluation consumes this serialized result. It must never import this module to
reconstruct, interpolate, or refit an observed trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from final_approach import (
    AMBIGUITY_MARGIN_M,
    Assignment,
    Projected,
    SegmentFit,
    TrackPoint,
    fit_final_segment,
)
from geokit import haversine_m

from trajectory_data_process.harvest.airports import (
    Runway,
    require_matching_runway_data,
    runway_data_fingerprint,
    runway_data_snapshot,
)
from trajectory_data_process.harvest.adsb_metadata import AdsbStateMetadata
from trajectory_data_process.harvest.tracks import Track

EVENT_SCHEMA_VERSION = "observed-threshold-event-v6"
EVENT_METHOD_VERSION = 6
EVENT_METHOD = "final_segment_robust_fit"

# These are source-integrity gates, not approach-performance limits. The 25 m/s
# tolerance is the rounded-up p99 (22.62 m/s) of |position-derived speed - reported
# ground speed| over 21,873 accepted threshold brackets across the five-airport
# development corpus. The 30 s freshness cap rounds up the maximum (27.25 s) of that
# same control set. Ratio bounds stop a low reported speed from passing on the absolute
# allowance alone. They are intentionally serialized with every accepted event.
MAX_POSITION_UPDATE_GAP_S = 30.0
MAX_REPORTED_GROUND_SPEED_M_S = 200.0
MAX_SPEED_DISAGREEMENT_M_S = 25.0
MIN_POSITION_TO_REPORTED_SPEED_RATIO = 0.5
MAX_POSITION_TO_REPORTED_SPEED_RATIO = 1.5

StateMetadataLookup = Callable[[str, float], AdsbStateMetadata | None]

NORMAL_95_MULTIPLIER = 1.96

EXTRAPOLATION_WINDOWS_M = (
    (-3000.0, -300.0),
    (-4000.0, -300.0),
    (-5000.0, -300.0),
)
LATERAL_FIT_MODEL_FLOOR_95_M = 10.5


@dataclass(frozen=True)
class _FitEnsemble:
    primary: SegmentFit
    candidates: tuple[SegmentFit, ...]
    vertical_statistical_95_m: float
    vertical_window_sensitivity_m: float
    vertical_effective_95_m: float
    lateral_statistical_95_m: float
    lateral_window_sensitivity_m: float


@dataclass(frozen=True)
class ThresholdBracket:
    """One source-validated crossing of one runway threshold plane."""

    runway: Runway
    projected: Projected
    source_sample_range: tuple[int, int]
    fraction: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ThresholdBracketSelection:
    """Relative runway choice made only from structurally valid brackets."""

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
    metadata_lookup: StateMetadataLookup | None = None,
    max_structural_cross_m: float,
    max_structural_height_m: float,
) -> ThresholdBracketSelection:
    """Select a runway from physical threshold brackets, before any fit.

    Every runway sees the same complete, time-ordered track.  A candidate must cross
    the finite landing structure and pass the ADS-B position/speed integrity checks.
    The runway with the smallest absolute crossing offset wins; sample recency is only
    a deterministic tie-break within one runway and never substitutes for geometry.
    """
    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    winners: dict[str, ThresholdBracket] = {}
    rejections: list[dict[str, Any]] = []
    for runway in runways:
        candidates, runway_rejections = _validated_brackets_for_runway(
            track,
            points,
            runway,
            metadata_lookup=metadata_lookup,
            max_structural_cross_m=max_structural_cross_m,
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
        return ThresholdBracketSelection(
            "unavailable",
            None,
            {},
            None,
            "no source-valid threshold bracket lies inside the landing structure",
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
) -> dict[str, Any]:
    """Return one fitted 3D event for the already selected final inbound pass."""
    common: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "unavailable",
        "method": EVENT_METHOD,
        "method_version": EVENT_METHOD_VERSION,
    }
    fit = assignment.fit
    if runway is None or assignment.runway is None:
        return {**common, "unavailable_reason": assignment.reason or assignment.outcome}
    if runway.ident != assignment.runway:
        raise ValueError("threshold-event runway disagrees with assignment")

    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    pass_anchor_index = (
        bracket.source_sample_range[0] if bracket is not None else None
    )
    if bracket is not None:
        fit_points = points
    elif fit is not None:
        # Runway assignment has already selected one physical inbound pass.  Bound
        # every event-window fit to that pass so a later rollout, circuit, or second
        # approach-like stretch cannot replace it merely because a narrower window
        # finds that later data attractive.
        fit_points = points[: fit.last_sample_index + 1]
    else:
        fit_points = points
    if fit is None:
        return {
            **_event_common(runway),
            "unavailable_reason": (
                assignment.reason
                or "selected final inbound pass has no fittable segment"
            ),
            **_bracket_audit(bracket, bracket_rejections),
        }
    ensemble = _fit_ensemble(
        runway,
        fit_points,
        fit,
        pass_anchor_index=pass_anchor_index,
    )
    if ensemble is None:
        return {
            **_event_common(runway),
            "unavailable_reason": "no event fit candidate yielded a final segment",
            **_bracket_audit(bracket, bracket_rejections),
        }
    if bracket is not None and bracket.runway.ident != runway.ident:
        raise ValueError("threshold bracket runway disagrees with assignment")
    return _fitted_event(
        runway,
        fit,
        ensemble=ensemble,
        bracket=bracket,
        bracket_rejections=bracket_rejections,
    )


def require_current_threshold_event(event: dict[str, Any], runway: Runway) -> None:
    """Require a current event contract and runway frame, regardless of availability."""
    if (
        event.get("schema_version") != EVENT_SCHEMA_VERSION
        or event.get("method_version") != EVENT_METHOD_VERSION
        or event.get("method") != EVENT_METHOD
    ):
        raise ValueError(
            f"track threshold event is not {EVENT_SCHEMA_VERSION}; "
            "run --reclassify-existing"
        )
    if event.get("status") not in ("estimated", "unavailable"):
        raise ValueError(
            f"track threshold event has invalid status {event.get('status')!r}; "
            "run --reclassify-existing"
        )
    if event.get("runway") != runway.ident:
        raise ValueError(
            f"track threshold event is not for runway {runway.ident}; "
            "run --reclassify-existing"
        )
    require_matching_runway_data(event, runway)


def _validated_brackets_for_runway(
    track: Track,
    points: list[TrackPoint],
    runway: Runway,
    *,
    metadata_lookup: StateMetadataLookup | None,
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
        # Count an exact threshold sample as the end of its inbound bracket, not
        # again as the beginning of the next pair.  Otherwise one physical crossing
        # appears twice and the recency tie-break selects a post-threshold interval.
        if not (before.along_m < 0.0 <= after.along_m):
            continue
        if after.along_m <= before.along_m:
            rejections.append(
                {
                    "reason": "threshold bracket is not strictly inbound",
                    "source_sample_range": [index, index + 1],
                }
            )
            continue
        dt = track.samples[index + 1].time_s - track.samples[index].time_s
        if dt <= 0.0:
            rejections.append(
                {
                    "reason": "threshold bracket time is not strictly increasing",
                    "source_sample_range": [index, index + 1],
                }
            )
            continue
        before_metadata = _state_metadata(track, index, metadata_lookup)
        after_metadata = _state_metadata(track, index + 1, metadata_lookup)
        if before_metadata is None or after_metadata is None:
            rejections.append(
                {
                    "reason": "ADS-B velocity or position-update metadata unavailable",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                }
            )
            continue
        before_speed = before_metadata.reported_ground_speed_m_s
        after_speed = after_metadata.reported_ground_speed_m_s
        before_position_time = before_metadata.last_position_update_s
        after_position_time = after_metadata.last_position_update_s
        if any(
            value is None or not math.isfinite(value)
            for value in (
                before_speed,
                after_speed,
                before_position_time,
                after_position_time,
            )
        ):
            rejections.append(
                {
                    "reason": "ADS-B velocity or position-update metadata unavailable",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                }
            )
            continue
        assert before_speed is not None and after_speed is not None
        assert before_position_time is not None and after_position_time is not None
        position_update_gap_s = after_position_time - before_position_time
        if position_update_gap_s <= 0.0:
            rejections.append(
                {
                    "reason": "ADS-B position-update time is not strictly increasing",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                    "position_update_gap_s": position_update_gap_s,
                }
            )
            continue
        if position_update_gap_s > MAX_POSITION_UPDATE_GAP_S:
            rejections.append(
                {
                    "reason": "ADS-B position-update gap exceeds 30 s",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                    "position_update_gap_s": position_update_gap_s,
                    "max_position_update_gap_s": MAX_POSITION_UPDATE_GAP_S,
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
                    "reason": "ADS-B reported ground speed is outside the integrity range",
                    "source_sample_range": [index, index + 1],
                    "reported_ground_speed_before_m_s": before_speed,
                    "reported_ground_speed_after_m_s": after_speed,
                    "max_reported_ground_speed_m_s": (
                        MAX_REPORTED_GROUND_SPEED_M_S
                    ),
                }
            )
            continue
        distance_m = haversine_m(
            points[index].lat,
            points[index].lon,
            points[index + 1].lat,
            points[index + 1].lon,
        )
        position_speed = distance_m / position_update_gap_s
        reported_speed = (before_speed + after_speed) / 2.0
        speed_error = position_speed - reported_speed
        speed_ratio = position_speed / reported_speed
        if (
            abs(speed_error) > MAX_SPEED_DISAGREEMENT_M_S
            or speed_ratio < MIN_POSITION_TO_REPORTED_SPEED_RATIO
            or speed_ratio > MAX_POSITION_TO_REPORTED_SPEED_RATIO
        ):
            rejections.append(
                {
                    "reason": (
                        "position displacement disagrees with ADS-B reported ground speed"
                    ),
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                    "position_update_gap_s": position_update_gap_s,
                    "position_distance_m": distance_m,
                    "position_derived_speed_m_s": position_speed,
                    "reported_ground_speed_before_m_s": before_speed,
                    "reported_ground_speed_after_m_s": after_speed,
                    "reported_ground_speed_mean_m_s": reported_speed,
                    "signed_speed_disagreement_m_s": speed_error,
                    "position_to_reported_speed_ratio": speed_ratio,
                }
            )
            continue
        fraction = -before.along_m / (after.along_m - before.along_m)
        crossing = Projected(
            0.0,
            before.cross_m + fraction * (after.cross_m - before.cross_m),
            before.height_m + fraction * (after.height_m - before.height_m),
        )
        if (
            abs(crossing.cross_m) > max_structural_cross_m
            or abs(crossing.height_m) > max_structural_height_m
        ):
            rejections.append(
                {
                    "reason": "threshold bracket is outside landing structure",
                    "source_sample_range": [index, index + 1],
                    "signed_cross_track_m": crossing.cross_m,
                    "height_at_threshold_m": crossing.height_m,
                    "max_structural_cross_m": max_structural_cross_m,
                    "max_structural_height_m": max_structural_height_m,
                }
            )
            continue
        candidates.append(
            ThresholdBracket(
                runway=runway,
                projected=crossing,
                source_sample_range=(index, index + 1),
                fraction=fraction,
                diagnostics={
                    "sample_gap_s": dt,
                    "along_track_span_m": after.along_m - before.along_m,
                    "position_update_gap_s": position_update_gap_s,
                    "position_distance_m": distance_m,
                    "position_derived_speed_m_s": position_speed,
                    "reported_ground_speed_before_m_s": before_speed,
                    "reported_ground_speed_after_m_s": after_speed,
                    "reported_ground_speed_mean_m_s": reported_speed,
                    "signed_speed_disagreement_m_s": speed_error,
                    "position_to_reported_speed_ratio": speed_ratio,
                },
            )
        )
    return candidates, rejections


def _state_metadata(
    track: Track,
    index: int,
    lookup: StateMetadataLookup | None,
) -> AdsbStateMetadata | None:
    sample = track.samples[index]
    if (
        sample.reported_ground_speed_m_s is not None
        and sample.last_position_update_s is not None
    ):
        return AdsbStateMetadata(
            sample.reported_ground_speed_m_s,
            sample.last_position_update_s,
            sample.last_contact_s,
        )
    return lookup(track.icao24, sample.time_s) if lookup is not None else None


def _fitted_event(
    runway: Runway,
    assignment_fit: SegmentFit,
    *,
    ensemble: _FitEnsemble,
    bracket: ThresholdBracket | None,
    bracket_rejections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    frame = runway.frame("hae")
    primary = ensemble.primary
    lateral_effective = max(
        LATERAL_FIT_MODEL_FLOOR_95_M,
        ensemble.lateral_statistical_95_m
        + ensemble.lateral_window_sensitivity_m,
    )
    crossing = frame.unproject(Projected(
        0.0,
        primary.cross_at_threshold_m,
        primary.height_at_threshold_m,
    ))
    return {
        **_event_common(runway),
        "status": "estimated",
        "method": EVENT_METHOD,
        "source_sample_range": [
            primary.first_sample_index,
            primary.last_sample_index,
        ],
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": primary.cross_at_threshold_m,
        "cross_track_sigma_m": lateral_effective / NORMAL_95_MULTIPLIER,
        "altitude_sigma_m": (
            ensemble.vertical_effective_95_m / NORMAL_95_MULTIPLIER
        ),
        "fit_window_m": list(primary.window_m),
        "sample_count": primary.n_samples,
        "along_track_span_m": primary.span_m,
        "cross_track_fit": _diagnostics(primary.cross),
        "altitude_fit": _diagnostics(primary.height),
        "assignment_fit": _fit_audit(assignment_fit),
        "candidate_fits": [_fit_audit(fit) for fit in ensemble.candidates],
        "glidepath_deg": primary.glidepath_deg,
        "median_abs_cross_track_m": primary.median_abs_cross_m,
        "nearest_sample_along_m": primary.nearest_sample_along_m,
        "lateral_extrapolation_m": primary.extrapolation_m,
        "vertical_extrapolation_m": primary.extrapolation_m,
        "extrapolation_m": primary.extrapolation_m,
        **_bracket_audit(bracket, bracket_rejections),
        "uncertainty_95_m": {
            "lateral_statistical": ensemble.lateral_statistical_95_m,
            "lateral_window_sensitivity": (
                ensemble.lateral_window_sensitivity_m
            ),
            "lateral_model_floor": LATERAL_FIT_MODEL_FLOOR_95_M,
            "lateral_effective": lateral_effective,
            "vertical_statistical": ensemble.vertical_statistical_95_m,
            "vertical_window_sensitivity": (
                ensemble.vertical_window_sensitivity_m
            ),
            "vertical_effective": ensemble.vertical_effective_95_m,
        },
        "unmodelled_uncertainty_sources": [
            "ADS-B geometric-altitude update alignment and measurement error",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
        ],
    }


def _fit_ensemble(
    runway: Runway,
    points: list[TrackPoint],
    assignment_fit: SegmentFit,
    *,
    pass_anchor_index: int | None = None,
) -> _FitEnsemble | None:
    frame = runway.frame("hae")
    candidates: list[SegmentFit] = []
    for window in EXTRAPOLATION_WINDOWS_M:
        candidate = (
            assignment_fit
            if tuple(assignment_fit.window_m) == window
            else fit_final_segment(
                points,
                frame,
                window_m=window,
                pass_anchor_index=pass_anchor_index,
            )
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    primary = next(
        (
            fit
            for fit in candidates
            if tuple(fit.window_m) == EXTRAPOLATION_WINDOWS_M[0]
        ),
        candidates[0],
    )
    vertical_statistical = max(
        NORMAL_95_MULTIPLIER * fit.height.sigma_at_zero
        for fit in candidates
    )
    vertical_sensitivity = max(
        abs(fit.height_at_threshold_m - primary.height_at_threshold_m)
        for fit in candidates
    )
    lateral_statistical = max(
        NORMAL_95_MULTIPLIER * fit.cross.sigma_at_zero
        for fit in candidates
    )
    lateral_sensitivity = max(
        abs(fit.cross_at_threshold_m - primary.cross_at_threshold_m)
        for fit in candidates
    )
    return _FitEnsemble(
        primary=primary,
        candidates=tuple(candidates),
        vertical_statistical_95_m=vertical_statistical,
        vertical_window_sensitivity_m=vertical_sensitivity,
        vertical_effective_95_m=vertical_statistical + vertical_sensitivity,
        lateral_statistical_95_m=lateral_statistical,
        lateral_window_sensitivity_m=lateral_sensitivity,
    )


def _bracket_audit(
    bracket: ThresholdBracket | None,
    rejections: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {"bracket_rejections": list(rejections)}
    if bracket is None:
        return output
    output["threshold_bracket"] = {
        "runway": bracket.runway.ident,
        "source_sample_range": list(bracket.source_sample_range),
        "fraction": bracket.fraction,
        "signed_cross_track_m": bracket.projected.cross_m,
        "height_at_threshold_m": bracket.projected.height_m,
        **bracket.diagnostics,
        "integrity_limits": {
            "max_position_update_gap_s": MAX_POSITION_UPDATE_GAP_S,
            "max_reported_ground_speed_m_s": MAX_REPORTED_GROUND_SPEED_M_S,
            "max_speed_disagreement_m_s": MAX_SPEED_DISAGREEMENT_M_S,
            "min_position_to_reported_speed_ratio": (
                MIN_POSITION_TO_REPORTED_SPEED_RATIO
            ),
            "max_position_to_reported_speed_ratio": (
                MAX_POSITION_TO_REPORTED_SPEED_RATIO
            ),
        },
        "role": "runway_and_pass_anchor_not_event_estimator",
    }
    return output


def _event_common(runway: Runway) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "unavailable",
        "method": EVENT_METHOD,
        "method_version": EVENT_METHOD_VERSION,
        "runway": runway.ident,
        "runway_data": runway_data_snapshot(runway),
        "runway_data_fingerprint": runway_data_fingerprint(runway),
    }


def _diagnostics(line: Any) -> dict[str, float]:
    return {
        "rms_residual_m": line.rms_residual_m,
        "max_abs_residual_m": line.max_abs_residual_m,
        "rho": line.rho,
        "n_effective": line.n_effective,
    }


def _fit_audit(fit: SegmentFit) -> dict[str, Any]:
    return {
        "window_m": list(fit.window_m),
        "source_sample_range": [fit.first_sample_index, fit.last_sample_index],
        "sample_count": fit.n_samples,
        "along_track_span_m": fit.span_m,
        "nearest_sample_along_m": fit.nearest_sample_along_m,
        "extrapolation_m": fit.extrapolation_m,
        "signed_cross_track_m": fit.cross_at_threshold_m,
        "height_at_threshold_m": fit.height_at_threshold_m,
        "glidepath_deg": fit.glidepath_deg,
        "cross_track_sigma_m": fit.cross.sigma_at_zero,
        "altitude_sigma_m": fit.height.sigma_at_zero,
        "cross_track_fit": _diagnostics(fit.cross),
        "altitude_fit": _diagnostics(fit.height),
        "rejected_sample_indices": list(fit.rejected_sample_indices),
    }
