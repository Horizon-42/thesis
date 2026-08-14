"""Produce one policy-free observed crossing event during runway classification.

Evaluation consumes this serialized result. It must never import this module to
reconstruct, interpolate, or refit an observed trajectory.
"""

from __future__ import annotations

import math
from typing import Any

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
from trajectory_data_process.harvest.tracks import Track

EVENT_SCHEMA_VERSION = "observed-threshold-event-v2"
EVENT_METHOD_VERSION = 2

MAX_INTERPOLATION_GAP_S = 5.0
MAX_INTERPOLATION_SPEED_M_S = 200.0

NORMAL_95_MULTIPLIER = 1.96
OPEN_SKY_ALTITUDE_QUANTUM_M = 25.0 * 0.3048
DIRECT_VERTICAL_MARGIN_95_M = OPEN_SKY_ALTITUDE_QUANTUM_M / 2.0

EXTRAPOLATION_WINDOWS_M = (
    (-3000.0, -300.0),
    (-4000.0, -300.0),
    (-5000.0, -300.0),
)
EXTRAPOLATION_VERTICAL_ERROR_P95_M = {
    (-3000.0, -300.0): 12.65,
    (-4000.0, -300.0): 13.43,
    (-5000.0, -300.0): 14.39,
}
EXTRAPOLATION_VERTICAL_MARGIN_QUANTUM_M = 0.5
EXTRAPOLATION_LATERAL_MARGIN_95_M = 10.5
CALIBRATION_POPULATION = (
    "21599 valid directly bracketed threshold crossings from KMSY, KRDU, "
    "KSJC, KSMF, and KSTL"
)


def build_observed_threshold_event(
    track: Track,
    runway: Runway | None,
    assignment: Assignment,
) -> dict[str, Any]:
    """Return event-v2 geometry for the winning final inbound pass."""
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
    direct, rejections = _direct_crossing(track, points, runway, fit)
    if direct is not None:
        return _direct_event(runway, fit, direct, rejections)
    return _extrapolated_event(runway, points, fit, rejections)


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
        if dt > MAX_INTERPOLATION_GAP_S:
            rejections.append(
                {
                    "reason": "sample gap exceeds 5 s",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                }
            )
            continue
        speed = haversine_m(
            points[index].lat,
            points[index].lon,
            points[index + 1].lat,
            points[index + 1].lon,
        ) / dt
        if speed > MAX_INTERPOLATION_SPEED_M_S:
            rejections.append(
                {
                    "reason": "implied horizontal speed exceeds 200 m/s",
                    "source_sample_range": [index, index + 1],
                    "sample_gap_s": dt,
                    "implied_horizontal_speed_m_s": speed,
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
            "implied_horizontal_speed_m_s": speed,
        }, rejections
    if not rejections:
        rejections.append(
            {"reason": "no threshold bracket after the winning assignment fit"}
        )
    return None, rejections


def _direct_event(
    runway: Runway,
    fit: SegmentFit,
    direct: dict[str, Any],
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    projected: Projected = direct["projected"]
    crossing = runway.frame("hae").unproject(projected)
    lateral_statistical = NORMAL_95_MULTIPLIER * fit.cross.sigma_at_zero
    lateral_disagreement = abs(projected.cross_m - fit.cross_at_threshold_m)
    lateral_effective = max(lateral_statistical, lateral_disagreement)
    vertical_fit_statistical = NORMAL_95_MULTIPLIER * fit.height.sigma_at_zero
    vertical_effective = max(DIRECT_VERTICAL_MARGIN_95_M, vertical_fit_statistical)
    return {
        **_event_common(runway),
        "status": "estimated",
        "method": "threshold_plane_interpolation",
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": projected.cross_m,
        "cross_track_sigma_m": lateral_effective / NORMAL_95_MULTIPLIER,
        "altitude_sigma_m": vertical_effective / NORMAL_95_MULTIPLIER,
        "source_sample_range": direct["source_sample_range"],
        "sample_count": 2,
        "along_track_span_m": direct["along_track_span_m"],
        "assignment_fit": _fit_audit(fit),
        "glidepath_deg": fit.glidepath_deg,
        "median_abs_cross_track_m": fit.median_abs_cross_m,
        "nearest_sample_along_m": 0.0,
        "extrapolation_m": 0.0,
        "interpolation": {
            "fraction": direct["fraction"],
            "sample_gap_s": direct["sample_gap_s"],
            "implied_horizontal_speed_m_s": direct["implied_horizontal_speed_m_s"],
            "max_sample_gap_s": MAX_INTERPOLATION_GAP_S,
            "max_horizontal_speed_m_s": MAX_INTERPOLATION_SPEED_M_S,
        },
        "interpolation_rejections": rejections,
        "uncertainty_95_m": {
            "lateral_fit_statistical": lateral_statistical,
            "lateral_direct_fit_disagreement": lateral_disagreement,
            "lateral_effective": lateral_effective,
            "vertical_altitude_quantization": DIRECT_VERTICAL_MARGIN_95_M,
            "vertical_fit_statistical": vertical_fit_statistical,
            "vertical_effective": vertical_effective,
        },
        "unmodelled_uncertainty_sources": [
            "ADS-B source integrity",
            "systematic horizontal position error",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
        ],
    }


def _extrapolated_event(
    runway: Runway,
    points: list[TrackPoint],
    assignment_fit: SegmentFit,
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
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
        return {
            **_event_common(runway),
            "unavailable_reason": "no extrapolation candidate yielded a final segment",
            "interpolation_rejections": rejections,
        }

    primary = next(
        (fit for fit in candidates if tuple(fit.window_m) == EXTRAPOLATION_WINDOWS_M[0]),
        candidates[0],
    )
    primary_window = tuple(primary.window_m)
    vertical_calibration_p95 = EXTRAPOLATION_VERTICAL_ERROR_P95_M[primary_window]
    vertical_empirical_floor = _round_up_to_quantum(
        vertical_calibration_p95,
        EXTRAPOLATION_VERTICAL_MARGIN_QUANTUM_M,
    )
    vertical_sensitivity = max(
        abs(fit.height_at_threshold_m - primary.height_at_threshold_m)
        for fit in candidates
    )
    lateral_sensitivity = max(
        abs(fit.cross_at_threshold_m - primary.cross_at_threshold_m)
        for fit in candidates
    )
    vertical_statistical = max(
        NORMAL_95_MULTIPLIER * fit.height.sigma_at_zero for fit in candidates
    )
    lateral_statistical = max(
        NORMAL_95_MULTIPLIER * fit.cross.sigma_at_zero for fit in candidates
    )
    vertical_effective = max(
        vertical_empirical_floor,
        vertical_statistical + vertical_sensitivity,
    )
    lateral_effective = max(
        EXTRAPOLATION_LATERAL_MARGIN_95_M,
        lateral_statistical + lateral_sensitivity,
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
        "threshold_crossing_lat": crossing.lat,
        "threshold_crossing_lon": crossing.lon,
        "threshold_crossing_altitude_m": crossing.alt_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": primary.cross_at_threshold_m,
        "cross_track_sigma_m": lateral_effective / NORMAL_95_MULTIPLIER,
        "altitude_sigma_m": vertical_effective / NORMAL_95_MULTIPLIER,
        "source_sample_range": [primary.first_sample_index, primary.last_sample_index],
        "fit_window_m": list(primary.window_m),
        "sample_count": primary.n_samples,
        "along_track_span_m": primary.span_m,
        "cross_track_fit": _diagnostics(primary.cross),
        "altitude_fit": _diagnostics(primary.height),
        "assignment_fit": _fit_audit(assignment_fit),
        "candidate_fits": [_fit_audit(fit) for fit in candidates],
        "glidepath_deg": primary.glidepath_deg,
        "median_abs_cross_track_m": primary.median_abs_cross_m,
        "nearest_sample_along_m": primary.nearest_sample_along_m,
        "extrapolation_m": primary.extrapolation_m,
        "interpolation_rejections": rejections,
        "uncertainty_95_m": {
            "lateral_statistical": lateral_statistical,
            "lateral_window_sensitivity": lateral_sensitivity,
            "lateral_empirical_floor": EXTRAPOLATION_LATERAL_MARGIN_95_M,
            "lateral_effective": lateral_effective,
            "vertical_statistical": vertical_statistical,
            "vertical_window_sensitivity": vertical_sensitivity,
            "vertical_empirical_floor": vertical_empirical_floor,
            "vertical_effective": vertical_effective,
        },
        "empirical_calibration": {
            "population": CALIBRATION_POPULATION,
            "primary_window_m": list(primary_window),
            "primary_window_vertical_error_p95_m": vertical_calibration_p95,
            "vertical_margin_rounding_quantum_m": (
                EXTRAPOLATION_VERTICAL_MARGIN_QUANTUM_M
            ),
            "applied_vertical_margin_95_m": vertical_empirical_floor,
            "lateral_direct_fit_difference_p95_m": 10.29,
            "applied_lateral_margin_95_m": EXTRAPOLATION_LATERAL_MARGIN_95_M,
        },
        "unmodelled_uncertainty_sources": [
            "airport/runway-specific extrapolation bias",
            "ADS-B source integrity",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
        ],
    }


def _round_up_to_quantum(value: float, quantum: float) -> float:
    """Conservatively round a measured error bound up to an auditable margin."""
    return math.ceil(value / quantum) * quantum


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
