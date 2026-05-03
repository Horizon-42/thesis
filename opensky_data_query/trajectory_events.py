"""Derived airport-event extraction for OpenSky tracks.

The functions in this module treat OpenSky tracks as immutable source data.
They build an analysis view with distances and boundary estimates, then return
derived event records that reference raw waypoint indexes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


KM_PER_NM = 1.852
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class AnalysisWaypoint:
    """A validated waypoint in analysis order with a pointer to raw path index."""

    raw_index: int
    time: float
    lat: float
    lon: float
    baro_altitude_m: float
    true_track_deg: float | None
    on_ground: bool
    distance_nm: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    return haversine_km(lat1, lon1, lat2, lon2) / KM_PER_NM


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_track_for_analysis(
    track: dict[str, Any],
    *,
    airport_lat: float,
    airport_lon: float,
) -> list[AnalysisWaypoint]:
    """Parse valid OpenSky path points into an immutable analysis view."""
    parsed: list[AnalysisWaypoint] = []
    for raw_index, wp in enumerate(track.get("path") or []):
        if not wp or len(wp) < 6:
            continue
        t, lat, lon, baro_altitude_m, true_track_deg, on_ground = wp[:6]
        t_value = _finite_float(t)
        lat_value = _finite_float(lat)
        lon_value = _finite_float(lon)
        alt_value = _finite_float(baro_altitude_m)
        if t_value is None or lat_value is None or lon_value is None or alt_value is None:
            continue

        track_value = _finite_float(true_track_deg)
        parsed.append(
            AnalysisWaypoint(
                raw_index=raw_index,
                time=t_value,
                lat=lat_value,
                lon=lon_value,
                baro_altitude_m=alt_value,
                true_track_deg=track_value,
                on_ground=bool(on_ground),
                distance_nm=distance_nm(lat_value, lon_value, airport_lat, airport_lon),
            )
        )
    return sorted(parsed, key=lambda item: item.time)


def _interpolate_angle_deg(a: float | None, b: float | None, alpha: float) -> float | None:
    """Interpolate heading degrees across the 360/0 wrap."""
    if a is None or b is None:
        return None
    delta = ((b - a + 180.0) % 360.0) - 180.0
    return (a + alpha * delta) % 360.0


def estimate_boundary_crossing(
    before: AnalysisWaypoint,
    after: AnalysisWaypoint,
    *,
    radius_nm: float,
    kind: str,
) -> dict[str, Any]:
    """Estimate where a segment crosses the airport radius without changing raw data."""
    denom = after.distance_nm - before.distance_nm
    alpha = 0.0 if abs(denom) < 1e-12 else (radius_nm - before.distance_nm) / denom
    alpha = max(0.0, min(1.0, alpha))

    return {
        "kind": "estimated_boundary_crossing",
        "boundary": kind,
        "radius_nm": radius_nm,
        "time": before.time + alpha * (after.time - before.time),
        "lat": before.lat + alpha * (after.lat - before.lat),
        "lon": before.lon + alpha * (after.lon - before.lon),
        "baro_altitude_m": before.baro_altitude_m + alpha * (after.baro_altitude_m - before.baro_altitude_m),
        "true_track_deg": _interpolate_angle_deg(before.true_track_deg, after.true_track_deg, alpha),
        "source_segment": {
            "before_raw_index": before.raw_index,
            "after_raw_index": after.raw_index,
        },
    }


def _utc_tag(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _track_identity(track: dict[str, Any]) -> tuple[str, int]:
    icao24 = str(track.get("icao24") or "unknown").lower()
    start = int(float(track.get("startTime") or 0))
    return icao24, start


def _source_matches(
    *,
    airport: str,
    candidate_sources: list[str],
    flight_metadata: dict[str, Any],
) -> tuple[bool, bool]:
    airport = airport.upper()
    sources = {item.lower() for item in candidate_sources}
    arrival_match = "arrival" in sources or str(flight_metadata.get("estArrivalAirport") or "").upper() == airport
    departure_match = "departure" in sources or str(flight_metadata.get("estDepartureAirport") or "").upper() == airport
    return arrival_match, departure_match


def classify_event(
    *,
    airport: str,
    candidate_sources: list[str],
    flight_metadata: dict[str, Any],
    has_complete_entry: bool,
    has_complete_exit: bool,
    has_near_airport_on_ground: bool,
    min_agl_m: float | None,
    min_distance_nm: float,
    low_altitude_agl_m: float,
) -> tuple[str, dict[str, Any]]:
    """Classify a derived episode using conservative evidence rules."""
    arrival_match, departure_match = _source_matches(
        airport=airport,
        candidate_sources=candidate_sources,
        flight_metadata=flight_metadata,
    )
    has_low_altitude = min_agl_m is not None and min_agl_m <= low_altitude_agl_m
    label = "unknown"

    if arrival_match and departure_match:
        label = "ambiguous"
    elif arrival_match:
        label = "landing" if has_complete_entry and (has_near_airport_on_ground or has_low_altitude) else "ambiguous"
    elif departure_match:
        label = "depart" if has_complete_exit and (has_near_airport_on_ground or has_low_altitude) else "ambiguous"
    elif has_complete_entry and has_complete_exit and (not has_near_airport_on_ground) and (not has_low_altitude):
        label = "pass"

    evidence = {
        "candidate_sources": candidate_sources,
        "estArrivalAirport_matches": arrival_match,
        "estDepartureAirport_matches": departure_match,
        "has_complete_entry": has_complete_entry,
        "has_complete_exit": has_complete_exit,
        "has_near_airport_on_ground": has_near_airport_on_ground,
        "min_agl_m": None if min_agl_m is None else round(min_agl_m, 3),
        "min_distance_nm": round(min_distance_nm, 3),
    }
    return label, evidence


def extract_complete_airport_events(
    track: dict[str, Any],
    *,
    airport: str,
    airport_lat: float,
    airport_lon: float,
    airport_elev_m: float,
    radius_nm: float = 5.0,
    candidate_sources: list[str] | None = None,
    flight_metadata: dict[str, Any] | None = None,
    low_altitude_agl_m: float = 600.0,
) -> list[dict[str, Any]]:
    """Extract complete entry-exit airport episodes from one raw OpenSky track."""
    candidate_sources = list(candidate_sources or [])
    flight_metadata = dict(flight_metadata or {})
    points = parse_track_for_analysis(track, airport_lat=airport_lat, airport_lon=airport_lon)
    if len(points) < 2:
        return []

    events: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = None
    inside_start_idx: int | None = None

    for idx in range(1, len(points)):
        before = points[idx - 1]
        after = points[idx]
        before_inside = before.distance_nm <= radius_nm
        after_inside = after.distance_nm <= radius_nm

        if (not before_inside) and after_inside:
            entry = estimate_boundary_crossing(before, after, radius_nm=radius_nm, kind="entry")
            inside_start_idx = idx
            continue

        if before_inside and (not after_inside) and entry is not None and inside_start_idx is not None:
            exit_crossing = estimate_boundary_crossing(before, after, radius_nm=radius_nm, kind="exit")
            inside_points = points[inside_start_idx:idx]
            if not inside_points:
                entry = None
                inside_start_idx = None
                continue

            min_distance = min(point.distance_nm for point in inside_points)
            min_agl = min(point.baro_altitude_m - airport_elev_m for point in inside_points)
            has_ground = any(point.on_ground and point.distance_nm <= radius_nm for point in inside_points)
            max_gap = max(
                (inside_points[i].time - inside_points[i - 1].time for i in range(1, len(inside_points))),
                default=0.0,
            )
            label, evidence = classify_event(
                airport=airport,
                candidate_sources=candidate_sources,
                flight_metadata=flight_metadata,
                has_complete_entry=True,
                has_complete_exit=True,
                has_near_airport_on_ground=has_ground,
                min_agl_m=min_agl,
                min_distance_nm=min_distance,
                low_altitude_agl_m=low_altitude_agl_m,
            )
            icao24, track_start = _track_identity(track)
            event_id = f"{airport.upper()}_{_utc_tag(entry['time'])}_{icao24}_{track_start}_{radius_nm:g}nm"
            events.append(
                {
                    "schema_version": "opensky-airport-event-v2",
                    "event_id": event_id,
                    "airport": airport.upper(),
                    "radius_nm": radius_nm,
                    "label": label,
                    "complete_radius_crossing": True,
                    "flight": {
                        "icao24": icao24,
                        "callsign": str(track.get("callsign") or flight_metadata.get("callsign") or "").strip(),
                        "track_start_time": track.get("startTime"),
                        "track_end_time": track.get("endTime"),
                        "estDepartureAirport": flight_metadata.get("estDepartureAirport"),
                        "estArrivalAirport": flight_metadata.get("estArrivalAirport"),
                    },
                    "event_time": {
                        "entry_time": entry["time"],
                        "exit_time": exit_crossing["time"],
                        "closest_time": min(inside_points, key=lambda point: point.distance_nm).time,
                    },
                    "quality": {
                        "num_points": len(inside_points),
                        "max_gap_s": max_gap,
                        "min_distance_nm": min_distance,
                        "complete": True,
                    },
                    "label_evidence": evidence,
                    "boundary_crossings": {
                        "entry": entry,
                        "exit": exit_crossing,
                    },
                    "raw_waypoint_range": {
                        "start_raw_index": inside_points[0].raw_index,
                        "end_raw_index": inside_points[-1].raw_index,
                    },
                }
            )
            entry = None
            inside_start_idx = None

    return events

