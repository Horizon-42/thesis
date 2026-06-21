"""Extract airport entry/exit episodes from a trajectory.

An event is a complete crossing of the airport radius: the trajectory enters the
circle of ``radius_nm`` around the airport and later leaves it. Each event is
classified (landing / depart / pass / ambiguous) from conservative evidence and
references the index range of points inside the circle, so the training builder
can attach the underlying trajectory points.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.geo import distance_nm
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


def extract_airport_events(
    traj: Trajectory,
    *,
    airport: str,
    airport_lat: float,
    airport_lon: float,
    airport_elev_m: float,
    radius_nm: float = 5.0,
    low_altitude_agl_m: float = 600.0,
) -> list[dict[str, Any]]:
    """Return complete entry-exit airport episodes for one trajectory."""
    points = traj.points
    if len(points) < 2:
        return []
    distances = [distance_nm(p.lat, p.lon, airport_lat, airport_lon) for p in points]

    events: list[dict[str, Any]] = []
    inside_start: int | None = None
    entry: dict[str, Any] | None = None

    for idx in range(1, len(points)):
        before_inside = distances[idx - 1] <= radius_nm
        after_inside = distances[idx] <= radius_nm

        if not before_inside and after_inside:
            entry = _boundary_crossing(points, distances, idx, radius_nm, "entry")
            inside_start = idx
        elif before_inside and not after_inside and entry is not None and inside_start is not None:
            exit_crossing = _boundary_crossing(points, distances, idx, radius_nm, "exit")
            event = _build_event(
                traj=traj,
                airport=airport,
                airport_elev_m=airport_elev_m,
                radius_nm=radius_nm,
                low_altitude_agl_m=low_altitude_agl_m,
                inside_points=points[inside_start:idx],
                inside_distances=distances[inside_start:idx],
                start_index=inside_start,
                end_index=idx - 1,
                entry=entry,
                exit_crossing=exit_crossing,
            )
            if event is not None:
                events.append(event)
            entry = None
            inside_start = None

    return events


def _build_event(
    *,
    traj: Trajectory,
    airport: str,
    airport_elev_m: float,
    radius_nm: float,
    low_altitude_agl_m: float,
    inside_points: list[TrajectoryPoint],
    inside_distances: list[float],
    start_index: int,
    end_index: int,
    entry: dict[str, Any],
    exit_crossing: dict[str, Any],
) -> dict[str, Any] | None:
    if not inside_points:
        return None

    min_distance = min(inside_distances)
    min_agl = min(_agl(p, airport_elev_m) for p in inside_points)
    has_ground = any(p.on_ground for p in inside_points)
    max_gap = max(
        (inside_points[i].time - inside_points[i - 1].time for i in range(1, len(inside_points))),
        default=0,
    )
    closest_point = min(inside_points, key=lambda p: distance_nm(p.lat, p.lon, entry["lat"], entry["lon"]))

    label, evidence = classify_event(
        airport=airport,
        arr_airport=traj.arr_airport,
        dep_airport=traj.dep_airport,
        has_near_airport_on_ground=has_ground,
        min_agl_m=min_agl,
        min_distance_nm=min_distance,
        low_altitude_agl_m=low_altitude_agl_m,
    )
    event_id = f"{airport.upper()}_{_utc_tag(entry['time'])}_{traj.icao24}_{traj.start_time}_{radius_nm:g}nm"
    return {
        "schema_version": "airport-event-v3",
        "event_id": event_id,
        "airport": airport.upper(),
        "radius_nm": radius_nm,
        "label": label,
        "flight": {
            "icao24": traj.icao24,
            "callsign": traj.callsign,
            "dep_airport": traj.dep_airport,
            "arr_airport": traj.arr_airport,
        },
        "event_time": {
            "entry_time": entry["time"],
            "exit_time": exit_crossing["time"],
            "closest_time": closest_point.time,
        },
        "quality": {
            "num_points": len(inside_points),
            "max_gap_s": max_gap,
            "min_distance_nm": round(min_distance, 3),
        },
        "label_evidence": evidence,
        "boundary_crossings": {"entry": entry, "exit": exit_crossing},
        "point_range": {"start_index": start_index, "end_index": end_index},
    }


def classify_event(
    *,
    airport: str,
    arr_airport: str | None,
    dep_airport: str | None,
    has_near_airport_on_ground: bool,
    min_agl_m: float,
    min_distance_nm: float,
    low_altitude_agl_m: float,
) -> tuple[str, dict[str, Any]]:
    """Classify an episode using conservative evidence rules."""
    code = airport.upper()
    arrival_match = (arr_airport or "").upper() == code
    departure_match = (dep_airport or "").upper() == code
    has_low_altitude = min_agl_m <= low_altitude_agl_m
    near_ground = has_near_airport_on_ground or has_low_altitude

    if arrival_match and departure_match:
        label = "ambiguous"
    elif arrival_match:
        label = "landing" if near_ground else "ambiguous"
    elif departure_match:
        label = "depart" if near_ground else "ambiguous"
    elif not near_ground:
        label = "pass"
    else:
        label = "unknown"

    evidence = {
        "arr_airport_matches": arrival_match,
        "dep_airport_matches": departure_match,
        "has_near_airport_on_ground": has_near_airport_on_ground,
        "min_agl_m": round(min_agl_m, 3),
        "min_distance_nm": round(min_distance_nm, 3),
    }
    return label, evidence


def _boundary_crossing(
    points: list[TrajectoryPoint],
    distances: list[float],
    idx: int,
    radius_nm: float,
    kind: str,
) -> dict[str, Any]:
    """Linear estimate of where a segment crosses the airport radius."""
    before, after = points[idx - 1], points[idx]
    denom = distances[idx] - distances[idx - 1]
    alpha = 0.0 if abs(denom) < 1e-12 else (radius_nm - distances[idx - 1]) / denom
    alpha = max(0.0, min(1.0, alpha))
    return {
        "boundary": kind,
        "radius_nm": radius_nm,
        "time": before.time + alpha * (after.time - before.time),
        "lat": before.lat + alpha * (after.lat - before.lat),
        "lon": before.lon + alpha * (after.lon - before.lon),
    }


def _agl(point: TrajectoryPoint, airport_elev_m: float) -> float:
    altitude = point.geo_altitude_m if point.geo_altitude_m is not None else point.baro_altitude_m
    return (altitude if altitude is not None else airport_elev_m) - airport_elev_m


def _utc_tag(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
