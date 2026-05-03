"""Transform OpenSky history DB rows into training records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.acquisition.opensky_history_db import normalize_history_dataframe
from trajectory_data_process.datasets.training_dataset import (
    attach_training_points_or_quarantine,
    make_raw_track_record,
)
from trajectory_data_process.processing.trajectory_events import extract_complete_airport_events


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


def _bool_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _history_rows_for_track(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Return rows that can produce path points, in path/raw-index order."""
    ordered = group.sort_values("time", kind="stable")
    rows: list[dict[str, Any]] = []
    for row in ordered.to_dict(orient="records"):
        if pd.isna(row.get("time")) or pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        if pd.isna(row.get("baroaltitude")):
            continue
        rows.append(row)
    return rows


def history_group_to_track(group: pd.DataFrame) -> dict[str, Any]:
    """Convert one icao24 history group to a track-like raw payload."""
    ordered = group.sort_values("time", kind="stable")
    path: list[list[Any]] = []
    for row in _history_rows_for_track(group):
        path.append(
            [
                _timestamp_seconds(row["time"]),
                float(row["lat"]),
                float(row["lon"]),
                float(row["baroaltitude"]),
                None if pd.isna(row.get("heading")) else float(row["heading"]),
                _bool_value(row.get("onground")),
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
    for raw_index, row in enumerate(_history_rows_for_track(group)):
        points.append(
            {
                "raw_index": raw_index,
                "time": _timestamp_seconds(row["time"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "baro_altitude_m": None if pd.isna(row.get("baroaltitude")) else float(row["baroaltitude"]),
                "geo_altitude_m": None if pd.isna(row.get("geoaltitude")) else float(row["geoaltitude"]),
                "true_track_deg": None if pd.isna(row.get("heading")) else float(row["heading"]),
                "on_ground": _bool_value(row.get("onground")),
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
    fetch_profile: str = "history_db",
    source_response_ids: list[str] | None = None,
    max_tracks: int | None = None,
    max_segment_gap_sec: int = 900,
) -> dict[str, list[dict[str, Any]]]:
    """Build raw tracks, airport events, and quarantine records from history rows."""
    raw_tracks: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    if df.empty:
        return {"raw_tracks": raw_tracks, "events": events_out, "quarantine": quarantine}

    normalized = normalize_history_dataframe(df)
    for group in iter_history_track_segments(normalized, max_gap_sec=max_segment_gap_sec):
        if max_tracks is not None and len(raw_tracks) >= max_tracks:
            break
        track = history_group_to_track(group)
        if not track.get("path"):
            continue

        metadata = history_group_to_flight_metadata(group)
        candidate_sources = _candidate_sources_for_airport(metadata, airport)
        raw_record = make_raw_track_record(
            airport=airport,
            fetch_profile=fetch_profile,
            candidate_sources=candidate_sources,
            source_response_ids=source_response_ids or ["opensky_history_db"],
            flight_metadata=metadata,
            track=track,
            source_api="opensky-history-db",
            source_endpoint="traffic.data.opensky.history",
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


def iter_history_track_segments(df: pd.DataFrame, *, max_gap_sec: int = 900) -> list[pd.DataFrame]:
    """Split history rows into per-aircraft track segments for event extraction."""
    if df.empty:
        return []

    out: list[pd.DataFrame] = []
    for _icao24, group in df.groupby("icao24", sort=False):
        ordered = group.sort_values("time", kind="stable").reset_index(drop=True)
        if ordered.empty:
            continue
        segment_start = 0
        previous_time = pd.Timestamp(ordered.at[0, "time"])
        previous_metadata_key = _metadata_key(ordered.iloc[0])
        for idx in range(1, len(ordered)):
            current_time = pd.Timestamp(ordered.at[idx, "time"])
            current_metadata_key = _metadata_key(ordered.iloc[idx])
            gap_sec = (current_time - previous_time).total_seconds()
            metadata_changed = _has_complete_metadata(previous_metadata_key) and _has_complete_metadata(current_metadata_key) and current_metadata_key != previous_metadata_key
            if gap_sec > max_gap_sec or metadata_changed:
                out.append(ordered.iloc[segment_start:idx].copy())
                segment_start = idx
            previous_time = current_time
            previous_metadata_key = current_metadata_key
        out.append(ordered.iloc[segment_start:].copy())
    return out


def _metadata_key(row: pd.Series) -> tuple[str | None, str | None]:
    dep = _clean_text(row.get("estdepartureairport"))
    arr = _clean_text(row.get("estarrivalairport"))
    return dep, arr


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    return text or None


def _has_complete_metadata(key: tuple[str | None, str | None]) -> bool:
    return bool(key[0] or key[1])


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
