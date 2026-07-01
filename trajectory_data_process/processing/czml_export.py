"""Convert trajectories into the CZML-input schema consumed by generate_czml.py.

Each exported flight is ``{id, callsign, type, icao24, dep_airport, arr_airport,
runway, waypoints}`` where every waypoint is ``[offset_sec, lon, lat, alt_m]`` and
``alt_m`` is **geometric** altitude. No barometric bias correction is applied:
geometric altitude is already referenced to the ellipsoid.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geokit import bearing_rad

from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.geo import haversine_km
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint

# The aircraft's approach direction must line up with the runway's landing
# direction (its true heading). Tight enough to reject misaligned overflights and
# the wrong runway end (ends are 180° apart), loose enough to tolerate a crosswind
# crab, a not-yet-rolled-out base-to-final turn, and ADS-B track noise.
DEFAULT_HEADING_TOLERANCE_DEG = 20.0

# How far back along the track (great-circle metres) the geometric approach course
# is measured from the threshold-crossing anchor.
APPROACH_COURSE_LOOKBACK_M = 1500.0


def trajectory_to_czml_flight(
    traj: Trajectory,
    *,
    airport_lat: float,
    airport_lon: float,
    max_end_distance_km: float = 2.5,
    approach_window_min: int = 20,
    crop_radius_km: float | None = None,
    exclude_ground: bool = False,
    runway_threshold: RunwayThreshold | None = None,
    runway_threshold_radius_m: float = 1000.0,
    landing_only: bool = False,
    descent_margin_m: float = 300.0,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
    landing_max_agl_m: float = 1500.0,
) -> dict[str, Any] | None:
    """Convert one trajectory to a CZML-input flight, or ``None`` if it is not a
    relevant arrival at the airport (or the requested runway threshold).

    With ``landing_only`` (which requires ``runway_threshold``) the trajectory is
    kept only when it lands at that threshold: its closest point passes within
    ``runway_threshold_radius_m`` of the threshold, below ``landing_max_agl_m``
    above it, having descended at least ``descent_margin_m`` from earlier in the
    track. Real ADS-B coverage rarely reaches the ground near a runway, so this
    keys off the aligned descent rather than a touchdown altitude.

    The direction test is kept separate: a landing that passes the geometry above is
    still returned, but tagged ``heading_ok`` (plus the measured errors) by
    :func:`_annotate_heading` — the aircraft's approach direction must line up with
    the runway's landing heading. Callers route ``heading_ok is False`` flights
    aside for false-kill review rather than dropping them silently.

    The exported waypoints run up to (and including) the anchor — the post-anchor
    rollout/taxi is always dropped. With ``crop_radius_km`` the earlier end is cropped
    spatially (points within that many km of the airport centre); otherwise it falls
    back to the ``approach_window_min`` time window.
    """
    points = traj.points
    if len(points) < 2:
        return None

    # Distance from each point to the airport centre — drives anchor selection and,
    # when set, the radius crop. The anchor-validity gate below (threshold / airport
    # end distance) is what rejects trajectories that never reach the airport.
    distances_km = [haversine_km(p.lat, p.lon, airport_lat, airport_lon) for p in points]

    anchor_index = _anchor_index(points, distances_km, runway_threshold)
    if not _anchor_is_valid(
        points[anchor_index],
        distances_km[anchor_index],
        max_end_distance_km=max_end_distance_km,
        runway_threshold=runway_threshold,
        runway_threshold_radius_m=runway_threshold_radius_m,
    ):
        return None

    if landing_only:
        if runway_threshold is None:
            raise ValueError("landing_only requires a runway_threshold")
        if not _is_landing_geometry(
            points, anchor_index, runway_threshold, descent_margin_m, landing_max_agl_m
        ):
            return None

    window = _select_window(points, distances_km, anchor_index, approach_window_min, crop_radius_km)
    waypoints = [
        [p.time - window[0].time, round(p.lon, 6), round(p.lat, 6), round(p.geo_altitude_m, 1)]
        for p in window
        if (not exclude_ground or not p.on_ground) and p.geo_altitude_m is not None
    ]
    if len(waypoints) < 2:
        return None

    callsign = (traj.callsign or traj.icao24).strip()
    flight_id = callsign.replace(" ", "")[:16] or traj.icao24
    flight: dict[str, Any] = {
        "id": flight_id,
        "callsign": callsign,
        "type": "UNK",
        "icao24": traj.icao24,
        "dep_airport": traj.dep_airport,
        "arr_airport": traj.arr_airport,
        "runway": runway_threshold.ident if runway_threshold else None,
        "landing_time_utc": _iso_utc(points[anchor_index].time) if runway_threshold else None,
        "altitude_source": "opensky_history_geoaltitude_m",
        "waypoints": waypoints,
    }
    if landing_only:
        _annotate_heading(flight, points, anchor_index, runway_threshold, heading_tolerance_deg)
    return flight


def trajectories_to_czml_input(
    trajectories: list[Trajectory],
    *,
    airport_lat: float,
    airport_lon: float,
    max_end_distance_km: float = 2.5,
    approach_window_min: int = 20,
    exclude_ground: bool = False,
    runway_threshold: RunwayThreshold | None = None,
    runway_threshold_radius_m: float = 1000.0,
    landing_only: bool = False,
    landing_max_agl_m: float = 1500.0,
    max_flights: int = 80,
) -> list[dict[str, Any]]:
    """Convert trajectories to CZML-input flights with unique ids.

    In ``landing_only`` mode a landing whose approach direction does not line up
    with the runway (``heading_ok is False``) is dropped here; use
    :func:`classify_landing_flights` to keep those aside for review.
    """
    flights: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for traj in trajectories:
        flight = trajectory_to_czml_flight(
            traj,
            landing_only=landing_only,
            landing_max_agl_m=landing_max_agl_m,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            max_end_distance_km=max_end_distance_km,
            approach_window_min=approach_window_min,
            exclude_ground=exclude_ground,
            runway_threshold=runway_threshold,
            runway_threshold_radius_m=runway_threshold_radius_m,
        )
        if flight is None or flight.get("heading_ok") is False:
            continue
        flight["id"] = _unique_id(flight["id"], used_ids)
        used_ids.add(flight["id"])
        flights.append(flight)
        if len(flights) >= max_flights:
            break
    return flights


def classify_landing_flights(
    trajectories: list[Trajectory],
    *,
    airport_lat: float,
    airport_lon: float,
    runway_threshold: RunwayThreshold,
    crop_radius_km: float | None = None,
    exclude_ground: bool = False,
    runway_threshold_radius_m: float = 1000.0,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
    landing_max_agl_m: float = 1500.0,
    max_accepted: int = 80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split landings at ``runway_threshold`` into (accepted, heading_rejected).

    Both share the same landing geometry (near the threshold, low, descending);
    they differ only on the direction test. ``accepted`` flights track the runway
    heading within ``heading_tolerance_deg``; ``heading_rejected`` flights are the
    ones that otherwise look like a landing but whose approach direction disagrees —
    kept (tagged with the measured heading errors) so a run can be audited for
    false kills rather than silently discarding them. Collection stops once
    ``max_accepted`` accepted landings are found.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for traj in trajectories:
        flight = trajectory_to_czml_flight(
            traj,
            landing_only=True,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            crop_radius_km=crop_radius_km,
            exclude_ground=exclude_ground,
            runway_threshold=runway_threshold,
            runway_threshold_radius_m=runway_threshold_radius_m,
            heading_tolerance_deg=heading_tolerance_deg,
            landing_max_agl_m=landing_max_agl_m,
        )
        if flight is None:
            continue
        flight["id"] = _unique_id(flight["id"], used_ids)
        used_ids.add(flight["id"])
        (accepted if flight["heading_ok"] else rejected).append(flight)
        if len(accepted) >= max_accepted:
            break
    return accepted, rejected


def _anchor_index(
    points: list[TrajectoryPoint],
    distances_km: list[float],
    runway_threshold: RunwayThreshold | None,
) -> int:
    """Index of the arrival anchor: closest to the runway threshold if given,
    otherwise closest to the airport center."""
    if runway_threshold is not None:
        return min(
            range(len(points)),
            key=lambda i: haversine_km(
                points[i].lat, points[i].lon, runway_threshold.lat, runway_threshold.lon
            ),
        )
    return min(range(len(distances_km)), key=lambda i: distances_km[i])


def _anchor_is_valid(
    anchor: TrajectoryPoint,
    anchor_distance_km: float,
    *,
    max_end_distance_km: float,
    runway_threshold: RunwayThreshold | None,
    runway_threshold_radius_m: float,
) -> bool:
    if runway_threshold is not None:
        threshold_km = haversine_km(
            anchor.lat, anchor.lon, runway_threshold.lat, runway_threshold.lon
        )
        return threshold_km <= runway_threshold_radius_m / 1000.0
    return anchor_distance_km <= max_end_distance_km


def _select_window(
    points: list[TrajectoryPoint],
    distances_km: list[float],
    anchor_index: int,
    approach_window_min: int,
    crop_radius_km: float | None,
) -> list[TrajectoryPoint]:
    """The exported approach: the anchor and the points before it (rollout dropped).

    ``crop_radius_km`` crops the earlier end by distance from the airport centre
    (reusing the already-computed ``distances_km``); without it the crop is the
    ``approach_window_min`` time window.
    """
    if crop_radius_km is not None:
        return [points[i] for i in range(anchor_index + 1) if distances_km[i] <= crop_radius_km]
    return _approach_window(points, anchor_index, approach_window_min)


def _approach_window(
    points: list[TrajectoryPoint], anchor_index: int, approach_window_min: int
) -> list[TrajectoryPoint]:
    """Keep points from ``approach_window_min`` before the anchor up to the anchor."""
    if approach_window_min <= 0:
        return points[: anchor_index + 1]
    anchor_time = points[anchor_index].time
    window_start = anchor_time - approach_window_min * 60
    window = [p for p in points[: anchor_index + 1] if window_start <= p.time <= anchor_time]
    return window if len(window) >= 2 else points[: anchor_index + 1]


def _is_landing_geometry(
    points: list[TrajectoryPoint],
    anchor_index: int,
    threshold: RunwayThreshold,
    descent_margin_m: float,
    max_agl_m: float,
) -> bool:
    """True if the trajectory descends to near the threshold (direction aside).

    The anchor is the closest point to the threshold (already within lateral range). A
    landing (a) reaches it below ``max_agl_m`` above the threshold — so high overflights
    that merely clip the threshold laterally are rejected — and (b) has descended at
    least ``descent_margin_m`` from earlier in the track (rejecting climbing
    departures). The direction test is applied separately by :func:`_annotate_heading`.

    ``max_agl_m`` is deliberately generous (not a touchdown altitude): low-altitude
    ADS-B coverage near runways is sparse, so a real approach's closest tracked point
    may still be a few hundred to a couple thousand feet up.
    """
    anchor = points[anchor_index]
    anchor_geo = anchor.geo_altitude_m
    if anchor_geo is None or anchor_geo - threshold.elevation_m > max_agl_m:
        return False

    earlier = [points[i].geo_altitude_m for i in range(anchor_index) if points[i].geo_altitude_m is not None]
    return bool(earlier) and max(earlier) - anchor_geo >= descent_margin_m


def _annotate_heading(
    flight: dict[str, Any],
    points: list[TrajectoryPoint],
    anchor_index: int,
    threshold: RunwayThreshold,
    heading_tolerance_deg: float,
) -> None:
    """Tag ``flight`` with the direction test vs the runway's landing heading.

    The aircraft's approach direction is checked two independent ways, and BOTH (when
    available) must fall within ``heading_tolerance_deg`` of the runway heading:

    * ``approach_course_deg`` — the ground course over the final segment into the
      threshold, derived purely from the track geometry (always available, robust to
      a missing ADS-B sample); and
    * ``adsb_track_deg`` — the ADS-B track reported at the threshold-crossing point.

    Sets ``heading_ok`` plus the measured values/errors so heading-rejected landings
    can be reviewed. When the runway has no published heading there is nothing to
    compare against and the landing is left ``heading_ok = True``.
    """
    runway_heading = threshold.heading_deg
    course = _approach_course_deg(points, anchor_index)
    track = points[anchor_index].heading_deg

    course_error = _heading_diff(course, runway_heading) if course is not None and runway_heading is not None else None
    track_error = _heading_diff(track, runway_heading) if track is not None and runway_heading is not None else None

    errors = [e for e in (course_error, track_error) if e is not None]
    heading_ok = runway_heading is None or not errors or all(e <= heading_tolerance_deg for e in errors)

    flight["heading_ok"] = heading_ok
    flight["runway_heading_deg"] = runway_heading
    flight["approach_course_deg"] = round(course, 1) if course is not None else None
    flight["adsb_track_deg"] = round(track, 1) if track is not None else None
    flight["course_error_deg"] = round(course_error, 1) if course_error is not None else None
    flight["track_error_deg"] = round(track_error, 1) if track_error is not None else None
    flight["heading_tolerance_deg"] = heading_tolerance_deg


def _approach_course_deg(
    points: list[TrajectoryPoint],
    anchor_index: int,
    lookback_m: float = APPROACH_COURSE_LOOKBACK_M,
) -> float | None:
    """Ground course (deg true, 0=N) over the final segment into the anchor, or None.

    Walks backward from the anchor until the straight-line distance reaches
    ``lookback_m`` (or the track start), then takes the great-circle bearing from
    that earlier point to the anchor — the direction the aircraft is actually moving
    as it crosses the threshold, independent of any (possibly missing) ADS-B track.
    """
    if anchor_index <= 0:
        return None
    anchor = points[anchor_index]
    start = anchor_index
    while start > 0:
        start -= 1
        if haversine_km(points[start].lat, points[start].lon, anchor.lat, anchor.lon) * 1000.0 >= lookback_m:
            break
    origin = points[start]
    if origin.lat == anchor.lat and origin.lon == anchor.lon:
        return None
    return math.degrees(bearing_rad(origin.lat, origin.lon, anchor.lat, anchor.lon)) % 360.0


def _heading_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two headings in degrees (0–180)."""
    return min((a - b) % 360.0, (b - a) % 360.0)


def _iso_utc(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def _unique_id(base: str, used_ids: set[str]) -> str:
    if base not in used_ids:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used_ids:
        suffix += 1
    return f"{base}_{suffix}"
