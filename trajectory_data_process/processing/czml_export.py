"""Convert trajectories into the CZML-input schema consumed by generate_czml.py.

Each exported flight is ``{id, callsign, type, icao24, dep_airport, arr_airport,
runway, waypoints}`` where every waypoint is ``[offset_sec, lon, lat, alt_m]`` and
``alt_m`` is **geometric** altitude. No barometric bias correction is applied:
geometric altitude is already referenced to the ellipsoid.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.geo import haversine_km
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


def trajectory_to_czml_flight(
    traj: Trajectory,
    *,
    airport_lat: float,
    airport_lon: float,
    match_radius_km: float = 35.0,
    max_end_distance_km: float = 2.5,
    approach_window_min: int = 20,
    exclude_ground: bool = False,
    runway_threshold: RunwayThreshold | None = None,
    runway_threshold_radius_m: float = 600.0,
    landing_only: bool = False,
    landing_agl_m: float = 150.0,
    descent_margin_m: float = 300.0,
) -> dict[str, Any] | None:
    """Convert one trajectory to a CZML-input flight, or ``None`` if it is not a
    relevant arrival at the airport (or the requested runway threshold).

    With ``landing_only`` (which requires ``runway_threshold``) the trajectory is
    kept only when it actually lands at that threshold: its closest point reaches
    low altitude over the threshold (within ``landing_agl_m``) after descending
    from at least ``descent_margin_m`` higher earlier — which excludes departures.
    """
    points = traj.points
    if len(points) < 2:
        return None

    # The trajectory must come close to the airport at least once.
    distances_km = [haversine_km(p.lat, p.lon, airport_lat, airport_lon) for p in points]
    if min(distances_km) > match_radius_km:
        return None

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
        if not _is_landing(points, anchor_index, runway_threshold, landing_agl_m, descent_margin_m):
            return None

    window = _approach_window(points, anchor_index, approach_window_min)
    waypoints = [
        [p.time - window[0].time, round(p.lon, 6), round(p.lat, 6), round(p.geo_altitude_m, 1)]
        for p in window
        if (not exclude_ground or not p.on_ground) and p.geo_altitude_m is not None
    ]
    if len(waypoints) < 2:
        return None

    callsign = (traj.callsign or traj.icao24).strip()
    flight_id = callsign.replace(" ", "")[:16] or traj.icao24
    return {
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


def trajectories_to_czml_input(
    trajectories: list[Trajectory],
    *,
    airport_lat: float,
    airport_lon: float,
    match_radius_km: float = 35.0,
    max_end_distance_km: float = 2.5,
    approach_window_min: int = 20,
    exclude_ground: bool = False,
    runway_threshold: RunwayThreshold | None = None,
    runway_threshold_radius_m: float = 600.0,
    landing_only: bool = False,
    max_flights: int = 80,
) -> list[dict[str, Any]]:
    """Convert trajectories to CZML-input flights with unique ids."""
    flights: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for traj in trajectories:
        flight = trajectory_to_czml_flight(
            traj,
            landing_only=landing_only,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            match_radius_km=match_radius_km,
            max_end_distance_km=max_end_distance_km,
            approach_window_min=approach_window_min,
            exclude_ground=exclude_ground,
            runway_threshold=runway_threshold,
            runway_threshold_radius_m=runway_threshold_radius_m,
        )
        if flight is None:
            continue
        flight["id"] = _unique_id(flight["id"], used_ids)
        used_ids.add(flight["id"])
        flights.append(flight)
        if len(flights) >= max_flights:
            break
    return flights


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


def _is_landing(
    points: list[TrajectoryPoint],
    anchor_index: int,
    threshold: RunwayThreshold,
    landing_agl_m: float,
    descent_margin_m: float,
) -> bool:
    """True if the anchor reaches the runway after descending from higher up.

    Geometric altitude carries a geoid offset versus the threshold's MSL elevation,
    so the thresholds are deliberately generous: ``landing_agl_m`` only needs to be
    near the runway, and the descent check (a clearly higher earlier point) is what
    separates a landing from a departure.
    """
    anchor_geo = points[anchor_index].geo_altitude_m
    if anchor_geo is None or anchor_geo - threshold.elevation_m > landing_agl_m:
        return False
    return any(
        points[i].geo_altitude_m is not None
        and points[i].geo_altitude_m - threshold.elevation_m >= descent_margin_m
        for i in range(anchor_index)
    )


def _iso_utc(unix_seconds: int) -> str:
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def _unique_id(base: str, used_ids: set[str]) -> str:
    if base not in used_ids:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used_ids:
        suffix += 1
    return f"{base}_{suffix}"
