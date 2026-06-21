"""Assemble training records from trajectories and derived airport events.

A raw-track record stores the immutable trajectory. A training event attaches the
trajectory points inside the airport radius (each carrying barometric and
geometric altitude). Events whose points lack required geometric altitude are
quarantined for audit rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.datasets.dataset_store import sha256_text, stable_json_dumps
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


SOURCE_API = "opensky-history-db"
SOURCE_ENDPOINT = "traffic.data.opensky.history"


def trajectory_as_dict(traj: Trajectory) -> dict[str, Any]:
    """Serialize a trajectory (with both altitudes per point) to plain JSON."""
    return {
        "icao24": traj.icao24,
        "callsign": traj.callsign,
        "dep_airport": traj.dep_airport,
        "arr_airport": traj.arr_airport,
        "start_time": traj.start_time,
        "end_time": traj.end_time,
        "points": [asdict(p) for p in traj.points],
    }


def make_raw_track_record(
    *,
    airport: str,
    fetch_profile: str,
    source_response_ids: list[str],
    traj: Trajectory,
) -> dict[str, Any]:
    """Build a raw-track JSONL record for one trajectory."""
    trajectory = trajectory_as_dict(traj)
    raw_track_id = f"raw_track:sha256:{sha256_text(stable_json_dumps(trajectory))}"
    return {
        "schema_version": "raw-track-v3",
        "raw_track_id": raw_track_id,
        "airport": airport.upper(),
        "fetch_profile": fetch_profile,
        "source": {
            "api": SOURCE_API,
            "endpoint": SOURCE_ENDPOINT,
            "source_response_ids": list(source_response_ids),
        },
        "trajectory": trajectory,
    }


def build_training_event(
    event: dict[str, Any],
    *,
    raw_track_id: str,
    traj: Trajectory,
    require_geo_altitude: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Attach trajectory points to an event, or quarantine it.

    Returns ``(ready, None)`` for a training-ready event and ``(None, quarantine)``
    when geometric altitude is required but missing on some points.
    """
    point_range = event.get("point_range") or {}
    start, end = point_range.get("start_index"), point_range.get("end_index")
    if start is None or end is None:
        return None, _quarantine(event, raw_track_id, "missing_point_range")

    selected = traj.points[int(start): int(end) + 1]
    missing_geo = [i for i, p in enumerate(selected, int(start)) if p.geo_altitude_m is None]
    if missing_geo and require_geo_altitude:
        return None, _quarantine(event, raw_track_id, "missing_geo_altitude", {"missing_indexes": missing_geo})

    ready = dict(event)
    ready["source"] = {"raw_track_id": raw_track_id}
    ready["training_points"] = [_point_as_dict(p) for p in selected]
    ready["quality"] = dict(ready.get("quality") or {})
    ready["quality"].update(
        {
            "points_total": len(selected),
            "points_with_geo_altitude": sum(1 for p in selected if p.geo_altitude_m is not None),
            "points_with_both_altitudes": sum(
                1 for p in selected if p.geo_altitude_m is not None and p.baro_altitude_m is not None
            ),
        }
    )
    return ready, None


def _point_as_dict(point: TrajectoryPoint) -> dict[str, Any]:
    return asdict(point)


def _quarantine(
    event: dict[str, Any], raw_track_id: str, reason: str, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "training-quarantine-v3",
        "event_id": event.get("event_id"),
        "airport": event.get("airport"),
        "label": event.get("label"),
        "raw_track_id": raw_track_id,
        "reason": reason,
        "detail": detail or {},
    }
