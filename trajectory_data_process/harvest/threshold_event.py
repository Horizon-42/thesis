"""Produce one policy-free observed crossing event during runway classification.

Evaluation consumes this serialized result. It must never import this module to
reconstruct, interpolate, or refit an observed trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from final_approach import (
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

EVENT_SCHEMA_VERSION = "observed-threshold-event-v4"
EVENT_METHOD_VERSION = 4

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
LATERAL_DIRECT_FIT_DISAGREEMENT_P95_M = 10.5


@dataclass(frozen=True)
class _FitEnsemble:
    primary: SegmentFit
    candidates: tuple[SegmentFit, ...]
    vertical_statistical_95_m: float
    vertical_window_sensitivity_m: float
    vertical_effective_95_m: float
    lateral_statistical_95_m: float
    lateral_window_sensitivity_m: float


def build_observed_threshold_event(
    track: Track,
    runway: Runway | None,
    assignment: Assignment,
    *,
    metadata_lookup: StateMetadataLookup | None = None,
) -> dict[str, Any]:
    """Return componentwise event-v4 geometry for the winning final inbound pass."""
    common: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "unavailable",
        "method": "threshold_event_estimator",
        "method_version": EVENT_METHOD_VERSION,
    }
    fit = assignment.fit
    if runway is None or assignment.runway is None or fit is None:
        return {**common, "unavailable_reason": assignment.reason or assignment.outcome}
    if runway.ident != assignment.runway:
        raise ValueError("threshold-event runway disagrees with assignment")

    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    ensemble = _fit_ensemble(runway, points, fit)
    if ensemble is None:
        return {
            **_event_common(runway),
            "unavailable_reason": "no event fit candidate yielded a final segment",
        }

    direct, rejections = _direct_crossing(
        track, points, runway, fit, metadata_lookup=metadata_lookup
    )
    if direct is not None:
        return _direct_event(runway, fit, ensemble, direct, rejections)
    return _extrapolated_event(
        runway,
        points,
        fit,
        rejections,
        ensemble=ensemble,
    )


def require_current_threshold_event(event: dict[str, Any], runway: Runway) -> None:
    """Reject obsolete derived events before a downstream consumer uses them."""
    if (
        event.get("schema_version") != EVENT_SCHEMA_VERSION
        or event.get("method_version") != EVENT_METHOD_VERSION
    ):
        raise ValueError(
            f"track threshold event is not {EVENT_SCHEMA_VERSION}; "
            "run --reclassify-existing"
        )
    if event.get("status") != "estimated" or event.get("runway") != runway.ident:
        raise ValueError(
            f"track lacks an estimated current threshold event for runway "
            f"{runway.ident}; run --reclassify-existing"
        )
    require_matching_runway_data(event, runway)


def _direct_crossing(
    track: Track,
    points: list[TrackPoint],
    runway: Runway,
    fit: SegmentFit,
    *,
    metadata_lookup: StateMetadataLookup | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    frame = runway.frame("hae")
    projected = frame.project_all(points)
    rejections: list[dict[str, Any]] = []
    for index in range(fit.last_sample_index, len(projected) - 1):
        before = projected[index]
        after = projected[index + 1]
        if not (before.along_m <= 0.0 <= after.along_m):
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
        return {
            "projected": crossing,
            "source_sample_range": [index, index + 1],
            "fraction": fraction,
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
        }, rejections
    if not rejections:
        rejections.append(
            {"reason": "no threshold bracket after the winning assignment fit"}
        )
    return None, rejections


def _direct_event(
    runway: Runway,
    assignment_fit: SegmentFit,
    ensemble: _FitEnsemble,
    direct: dict[str, Any],
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_projected: Projected = direct["projected"]
    primary = ensemble.primary
    crossing = runway.frame("hae").unproject(Projected(
        0.0,
        direct_projected.cross_m,
        primary.height_at_threshold_m,
    ))
    lateral_statistical = ensemble.lateral_statistical_95_m
    lateral_disagreement = abs(
        direct_projected.cross_m - primary.cross_at_threshold_m
    )
    lateral_effective = max(lateral_statistical, lateral_disagreement)
    vertical_signed_disagreement = (
        direct_projected.height_m - primary.height_at_threshold_m
    )
    return {
        **_event_common(runway),
        "status": "estimated",
        "method": "direct_lateral_fitted_vertical",
        "component_methods": {
            "lateral": "threshold_plane_interpolation",
            "vertical": "final_segment_window_ensemble",
        },
        "component_source_sample_ranges": {
            "lateral": direct["source_sample_range"],
            "vertical": [
                primary.first_sample_index,
                primary.last_sample_index,
            ],
        },
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": direct_projected.cross_m,
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
        "lateral_extrapolation_m": 0.0,
        "vertical_extrapolation_m": primary.extrapolation_m,
        "extrapolation_m": primary.extrapolation_m,
        "interpolation": {
            "fraction": direct["fraction"],
            "sample_gap_s": direct["sample_gap_s"],
            "position_update_gap_s": direct["position_update_gap_s"],
            "position_distance_m": direct["position_distance_m"],
            "position_derived_speed_m_s": direct["position_derived_speed_m_s"],
            "reported_ground_speed_before_m_s": (
                direct["reported_ground_speed_before_m_s"]
            ),
            "reported_ground_speed_after_m_s": (
                direct["reported_ground_speed_after_m_s"]
            ),
            "reported_ground_speed_mean_m_s": (
                direct["reported_ground_speed_mean_m_s"]
            ),
            "signed_speed_disagreement_m_s": (
                direct["signed_speed_disagreement_m_s"]
            ),
            "position_to_reported_speed_ratio": (
                direct["position_to_reported_speed_ratio"]
            ),
            "integrity_limits": {
                "max_position_update_gap_s": MAX_POSITION_UPDATE_GAP_S,
                "max_reported_ground_speed_m_s": (
                    MAX_REPORTED_GROUND_SPEED_M_S
                ),
                "max_speed_disagreement_m_s": MAX_SPEED_DISAGREEMENT_M_S,
                "min_position_to_reported_speed_ratio": (
                    MIN_POSITION_TO_REPORTED_SPEED_RATIO
                ),
                "max_position_to_reported_speed_ratio": (
                    MAX_POSITION_TO_REPORTED_SPEED_RATIO
                ),
            },
        },
        "interpolation_rejections": rejections,
        "direct_vertical_proxy": {
            "height_m": direct_projected.height_m,
            "threshold_crossing_altitude_m": (
                runway.elevation_hae_m + direct_projected.height_m
            ),
            "signed_direct_minus_fit_m": vertical_signed_disagreement,
            "fit_disagreement_m": abs(vertical_signed_disagreement),
            "classification": "diagnostic_only_not_event_point",
        },
        "uncertainty_95_m": {
            "lateral_fit_statistical": lateral_statistical,
            "lateral_direct_fit_disagreement": lateral_disagreement,
            "lateral_effective": lateral_effective,
            "vertical_statistical": ensemble.vertical_statistical_95_m,
            "vertical_window_sensitivity": (
                ensemble.vertical_window_sensitivity_m
            ),
            "vertical_effective": ensemble.vertical_effective_95_m,
        },
        "unmodelled_uncertainty_sources": [
            "ADS-B source integrity",
            "ADS-B geometric-altitude update alignment",
            "systematic horizontal position error",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
        ],
    }


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


def _extrapolated_event(
    runway: Runway,
    points: list[TrackPoint],
    assignment_fit: SegmentFit,
    rejections: list[dict[str, Any]],
    *,
    ensemble: _FitEnsemble | None = None,
) -> dict[str, Any]:
    selected = ensemble or _fit_ensemble(runway, points, assignment_fit)
    if selected is None:
        return {
            **_event_common(runway),
            "unavailable_reason": "no extrapolation candidate yielded a final segment",
            "interpolation_rejections": rejections,
        }

    frame = runway.frame("hae")
    primary = selected.primary
    lateral_effective = max(
        LATERAL_DIRECT_FIT_DISAGREEMENT_P95_M,
        selected.lateral_statistical_95_m
        + selected.lateral_window_sensitivity_m,
    )
    crossing = frame.unproject(Projected(
        0.0,
        primary.cross_at_threshold_m,
        primary.height_at_threshold_m,
    ))
    return {
        **_event_common(runway),
        "status": "estimated",
        "method": "final_segment_window_ensemble",
        "component_methods": {
            "lateral": "final_segment_window_ensemble",
            "vertical": "final_segment_window_ensemble",
        },
        "component_source_sample_ranges": {
            "lateral": [primary.first_sample_index, primary.last_sample_index],
            "vertical": [primary.first_sample_index, primary.last_sample_index],
        },
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": primary.cross_at_threshold_m,
        "cross_track_sigma_m": lateral_effective / NORMAL_95_MULTIPLIER,
        "altitude_sigma_m": (
            selected.vertical_effective_95_m / NORMAL_95_MULTIPLIER
        ),
        "fit_window_m": list(primary.window_m),
        "sample_count": primary.n_samples,
        "along_track_span_m": primary.span_m,
        "cross_track_fit": _diagnostics(primary.cross),
        "altitude_fit": _diagnostics(primary.height),
        "assignment_fit": _fit_audit(assignment_fit),
        "candidate_fits": [_fit_audit(fit) for fit in selected.candidates],
        "glidepath_deg": primary.glidepath_deg,
        "median_abs_cross_track_m": primary.median_abs_cross_m,
        "nearest_sample_along_m": primary.nearest_sample_along_m,
        "lateral_extrapolation_m": primary.extrapolation_m,
        "vertical_extrapolation_m": primary.extrapolation_m,
        "extrapolation_m": primary.extrapolation_m,
        "interpolation_rejections": rejections,
        "uncertainty_95_m": {
            "lateral_statistical": selected.lateral_statistical_95_m,
            "lateral_window_sensitivity": (
                selected.lateral_window_sensitivity_m
            ),
            "lateral_proxy_disagreement_floor": (
                LATERAL_DIRECT_FIT_DISAGREEMENT_P95_M
            ),
            "lateral_effective": lateral_effective,
            "vertical_statistical": selected.vertical_statistical_95_m,
            "vertical_window_sensitivity": (
                selected.vertical_window_sensitivity_m
            ),
            "vertical_effective": selected.vertical_effective_95_m,
        },
        "unmodelled_uncertainty_sources": [
            "ADS-B source integrity",
            "ADS-B geometric-altitude update alignment",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
        ],
    }


def _fit_ensemble(
    runway: Runway,
    points: list[TrackPoint],
    assignment_fit: SegmentFit,
) -> _FitEnsemble | None:
    frame = runway.frame("hae")
    candidates: list[SegmentFit] = []
    for window in EXTRAPOLATION_WINDOWS_M:
        candidate = (
            assignment_fit
            if tuple(assignment_fit.window_m) == window
            else fit_final_segment(points, frame, window_m=window)
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


def _event_common(runway: Runway) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "unavailable",
        "method": "threshold_event_estimator",
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
    }
