"""Assembly helpers for OpenSky training dataset records.

This layer combines immutable raw tracks, derived airport events, and
dual-altitude points. It decides whether an event is training-ready or should
be quarantined, but it does not fetch network data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.datasets.dataset_store import stable_json_dumps, sha256_text


def stable_record_id(prefix: str, payload: dict[str, Any]) -> str:
    """Create a stable content-derived identifier."""
    return f"{prefix}:sha256:{sha256_text(stable_json_dumps(payload))}"


def make_raw_track_record(
    *,
    airport: str,
    fetch_profile: str,
    candidate_sources: list[str],
    source_response_ids: list[str],
    flight_metadata: dict[str, Any],
    track: dict[str, Any],
    source_api: str = "opensky",
    source_endpoint: str = "/tracks/all",
) -> dict[str, Any]:
    """Build a raw track JSONL record without mutating the OpenSky track."""
    raw_track_id = stable_record_id("raw_track", track)
    return {
        "schema_version": "opensky-raw-track-v2",
        "raw_track_id": raw_track_id,
        "airport": airport.upper(),
        "fetch_profile": fetch_profile,
        "source": {
            "api": source_api,
            "endpoint": source_endpoint,
            "candidate_sources": list(candidate_sources),
            "source_response_ids": list(source_response_ids),
        },
        "flight_metadata": dict(flight_metadata),
        "track": track,
    }


def attach_training_points_or_quarantine(
    event: dict[str, Any],
    *,
    raw_track_id: str,
    dual_altitude_points: list[dict[str, Any]],
    require_geo_altitude: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Attach dual-altitude points to an event or return a quarantine record."""
    point_by_raw_index = {int(point["raw_index"]): point for point in dual_altitude_points}
    raw_range = event.get("raw_waypoint_range") or {}
    start = raw_range.get("start_raw_index")
    end = raw_range.get("end_raw_index")
    if start is None or end is None:
        return None, _quarantine(event, raw_track_id, "missing_raw_waypoint_range")

    selected: list[dict[str, Any]] = []
    missing_geo_indexes: list[int] = []
    for raw_index in range(int(start), int(end) + 1):
        point = point_by_raw_index.get(raw_index)
        if point is None:
            missing_geo_indexes.append(raw_index)
            continue
        if require_geo_altitude and point.get("geo_altitude_m") is None:
            missing_geo_indexes.append(raw_index)
            continue
        selected.append(point)

    if missing_geo_indexes and require_geo_altitude:
        quarantine = _quarantine(
            event,
            raw_track_id,
            "missing_geo_altitude",
            {"missing_raw_indexes": missing_geo_indexes},
        )
        return None, quarantine

    out = dict(event)
    out["source"] = dict(out.get("source") or {})
    out["source"]["raw_track_id"] = raw_track_id
    out["training_points"] = selected
    out["quality"] = dict(out.get("quality") or {})
    out["quality"]["points_total"] = len(selected)
    out["quality"]["points_with_baro_altitude"] = sum(1 for point in selected if point.get("baro_altitude_m") is not None)
    out["quality"]["points_with_geo_altitude"] = sum(1 for point in selected if point.get("geo_altitude_m") is not None)
    out["quality"]["points_with_both_altitudes"] = sum(
        1
        for point in selected
        if point.get("baro_altitude_m") is not None and point.get("geo_altitude_m") is not None
    )
    return out, None


def _quarantine(
    event: dict[str, Any],
    raw_track_id: str,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an auditable quarantine record for a derived event."""
    return {
        "schema_version": "opensky-training-quarantine-v2",
        "event_id": event.get("event_id"),
        "airport": event.get("airport"),
        "label": event.get("label"),
        "raw_track_id": raw_track_id,
        "reason": reason,
        "detail": detail or {},
    }
