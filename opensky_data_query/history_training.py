"""Transform OpenSky history DB rows into training records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    from .training_dataset import attach_training_points_or_quarantine, make_raw_track_record
    from .trajectory_events import extract_complete_airport_events
except ImportError:  # pragma: no cover - supports direct script execution.
    from training_dataset import attach_training_points_or_quarantine, make_raw_track_record
    from trajectory_events import extract_complete_airport_events


def _timestamp_seconds(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp())


def _first_nonempty(values: pd.Series) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def history_group_to_track(group: pd.DataFrame) -> dict[str, Any]:
    """Convert one icao24 history group to a track-like raw payload."""
    ordered = group.sort_values("time", kind="stable")
    path: list[list[Any]] = []
    for row in ordered.to_dict(orient="records"):
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")) or pd.isna(row.get("baroaltitude")):
            continue
        path.append(
            [
                _timestamp_seconds(row["time"]),
                float(row["lat"]),
                float(row["lon"]),
                float(row["baroaltitude"]),
                None if pd.isna(row.get("heading")) else float(row["heading"]),
                bool(row.get("onground")),
            ]
        )

    times = [point[0] for point in path]
    return {
        "icao24": str(ordered["icao24"].iloc[0]).lower().strip(),
        "callsign": _first_nonempty(ordered.get("callsign", pd.Series(dtype=object))),
        "startTime": min(times) if times else None,
        "endTime": max(times) if times else None,
        "path": path,
    }


def history_group_to_dual_altitude_points(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Build derived points that retain both barometric and geometric altitude."""
    points: list[dict[str, Any]] = []
    ordered = group.sort_values("time", kind="stable").reset_index(drop=True)
    for raw_index, row in enumerate(ordered.to_dict(orient="records")):
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        points.append(
            {
                "raw_index": raw_index,
                "time": _timestamp_seconds(row["time"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "baro_altitude_m": None if pd.isna(row.get("baroaltitude")) else float(row["baroaltitude"]),
                "geo_altitude_m": None if pd.isna(row.get("geoaltitude")) else float(row["geoaltitude"]),
                "true_track_deg": None if pd.isna(row.get("heading")) else float(row["heading"]),
                "on_ground": bool(row.get("onground")),
                "altitude_sources": {
                    "baro_altitude_m": "opensky_history_db_baroaltitude_m",
                    "geo_altitude_m": "opensky_history_db_geoaltitude_m",
                },
                "geo_altitude_match": {
                    "method": "same_history_row",
                    "source_time": _timestamp_seconds(row["time"]),
                    "delta_t_sec": 0,
                    "source_response_id": "opensky_history_db",
                },
            }
        )
    return points


def history_group_to_flight_metadata(group: pd.DataFrame) -> dict[str, Any]:
    """Extract stable flight metadata from a history group."""
    ordered = group.sort_values("time", kind="stable")
    return {
        "icao24": str(ordered["icao24"].iloc[0]).lower().strip(),
        "callsign": _first_nonempty(ordered.get("callsign", pd.Series(dtype=object))),
        "estDepartureAirport": _first_nonempty(ordered.get("estdepartureairport", pd.Series(dtype=object))),
        "estArrivalAirport": _first_nonempty(ordered.get("estarrivalairport", pd.Series(dtype=object))),
    }


def build_training_records_from_history(
    df: pd.DataFrame,
    *,
    airport: str,
    airport_lat: float,
    airport_lon: float,
    airport_elev_m: float,
    radius_nm: float,
    low_altitude_agl_m: float,
    require_geo_altitude: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build raw tracks, airport events, and quarantine records from history rows."""
    raw_tracks: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    if df.empty:
        return {"raw_tracks": raw_tracks, "events": events_out, "quarantine": quarantine}

    for _icao24, group in df.groupby("icao24", sort=False):
        track = history_group_to_track(group)
        if not track.get("path"):
            continue

        metadata = history_group_to_flight_metadata(group)
        candidate_sources = _candidate_sources_for_airport(metadata, airport)
        raw_record = make_raw_track_record(
            airport=airport,
            fetch_profile="history_db",
            candidate_sources=candidate_sources,
            source_response_ids=["opensky_history_db"],
            flight_metadata=metadata,
            track=track,
        )
        raw_tracks.append(raw_record)

        events = extract_complete_airport_events(
            track,
            airport=airport,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            airport_elev_m=airport_elev_m,
            radius_nm=radius_nm,
            candidate_sources=candidate_sources,
            flight_metadata=metadata,
            low_altitude_agl_m=low_altitude_agl_m,
        )
        dual_points = history_group_to_dual_altitude_points(group)
        for event in events:
            ready, quarantined = attach_training_points_or_quarantine(
                event,
                raw_track_id=raw_record["raw_track_id"],
                dual_altitude_points=dual_points,
                require_geo_altitude=require_geo_altitude,
            )
            if ready is not None:
                events_out.append(ready)
            if quarantined is not None:
                quarantine.append(quarantined)

    return {"raw_tracks": raw_tracks, "events": events_out, "quarantine": quarantine}


def _candidate_sources_for_airport(metadata: dict[str, Any], airport: str) -> list[str]:
    airport = airport.upper()
    sources: list[str] = []
    if str(metadata.get("estArrivalAirport") or "").upper() == airport:
        sources.append("arrival")
    if str(metadata.get("estDepartureAirport") or "").upper() == airport:
        sources.append("departure")
    if not sources:
        sources.append("area")
    return sources

