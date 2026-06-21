#!/usr/bin/env python3
"""Download airport trajectories from the OpenSky history database.

Single source: ``traffic.data.opensky.history``. Every trajectory carries
geometric altitude. Two output modes:

- ``--dataset-mode czml`` (default): write a ``*_czml_input_*.json`` file for
  ``aeroviz-4d/python/generate_czml.py``.
- ``--dataset-mode training``: write partitioned raw-track / event / quarantine
  JSONL plus a manifest.

Selection can be narrowed to the exact runway threshold the aircraft arrives at
with ``--runway`` (e.g. ``--runway 23R``).

Example:
    python trajectory_data_process/download_trajectories.py \\
      --airport KRDU --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z \\
      --runway 23R --max-trajectories 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.acquisition.airports import AirportProfile, resolve_airport_profile
from trajectory_data_process.acquisition.opensky_history import (
    AIRPORT_HISTORY_COLUMNS,
    STATE_VECTOR_COLUMNS,
    fetch_history_dataframe,
)
from trajectory_data_process.acquisition.runways import (
    RunwayThreshold,
    resolve_runway_threshold,
    runways_csv_path,
)
from trajectory_data_process.datasets.dataset_store import partition_path, write_jsonl_records
from trajectory_data_process.datasets.training_dataset import build_training_event, make_raw_track_record
from trajectory_data_process.processing.czml_export import trajectories_to_czml_input
from trajectory_data_process.processing.trajectory_events import extract_airport_events
from trajectory_data_process.trajectory import Trajectory, build_trajectories_from_history


def default_outputs_root(script_path: Path) -> Path:
    return script_path.parent / "outputs"


def default_aeroviz_root(script_path: Path) -> Path:
    return script_path.parents[1] / "aeroviz-4d"


def parse_utc(value: str) -> datetime:
    """Parse a Unix-seconds string or ISO datetime as UTC."""
    if value.isdigit():
        return datetime.fromtimestamp(int(value), timezone.utc)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def iter_time_chunks(start: datetime, stop: datetime, *, chunk_hours: float) -> list[tuple[datetime, datetime]]:
    """Split a time range into positive UTC chunks of at most ``chunk_hours``."""
    if stop <= start:
        raise ValueError("--end must be later than --begin")
    if chunk_hours <= 0:
        raise ValueError("--chunk-hours must be positive")
    step = timedelta(hours=chunk_hours)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < stop:
        chunks.append((cursor, min(cursor + step, stop)))
        cursor += step
    return chunks


def traffic_bounds(profile: AirportProfile, *, lat_pad: float, lon_pad: float) -> tuple[float, float, float, float]:
    """Bounds in traffic order: west, south, east, north."""
    return (profile.lon - lon_pad, profile.lat - lat_pad, profile.lon + lon_pad, profile.lat + lat_pad)


def fetch_history(
    *,
    start: datetime,
    stop: datetime,
    airport: str,
    profile: AirportProfile,
    fetch_profile: str,
    chunk_hours: float,
    bbox_lat_pad: float,
    bbox_lon_pad: float,
    cached: bool,
) -> pd.DataFrame:
    """Fetch history rows over the window, one query per time chunk."""
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_stop in iter_time_chunks(start, stop, chunk_hours=chunk_hours):
        print(f"[history] {airport} airport_ops {chunk_start.isoformat()} -> {chunk_stop.isoformat()}", flush=True)
        frames.append(
            fetch_history_dataframe(
                start=chunk_start,
                stop=chunk_stop,
                airport=airport,
                selected_columns=AIRPORT_HISTORY_COLUMNS,
                cached=cached,
            )
        )
        if fetch_profile == "terminal_all":
            bounds = traffic_bounds(profile, lat_pad=bbox_lat_pad, lon_pad=bbox_lon_pad)
            print(f"[history] {airport} terminal_area bounds={bounds}", flush=True)
            frames.append(
                fetch_history_dataframe(
                    start=chunk_start,
                    stop=chunk_stop,
                    bounds=bounds,
                    selected_columns=STATE_VECTOR_COLUMNS,
                    cached=cached,
                )
            )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_czml_input(
    *,
    trajectories: list[Trajectory],
    profile: AirportProfile,
    runway_threshold: RunwayThreshold | None,
    args: argparse.Namespace,
    output_dir: Path,
    timestamp: str,
) -> None:
    flights = trajectories_to_czml_input(
        trajectories,
        airport_lat=profile.lat,
        airport_lon=profile.lon,
        match_radius_km=args.match_radius_km,
        max_end_distance_km=args.max_end_distance_km,
        approach_window_min=args.approach_window_min,
        exclude_ground=args.exclude_ground,
        runway_threshold=runway_threshold,
        runway_threshold_radius_m=args.runway_threshold_radius_m,
        max_flights=args.max_trajectories,
    )
    airport_tag = profile.code.lower()
    czml_path = output_dir / f"{airport_tag}_czml_input_{timestamp}.json"
    czml_path.write_text(json.dumps(flights, indent=2), encoding="utf-8")
    print(f"[czml] trajectories considered: {len(trajectories)}")
    print(f"[czml] flights exported: {len(flights)}")
    print(f"[czml] output: {czml_path}")


def write_training(
    *,
    trajectories: list[Trajectory],
    profile: AirportProfile,
    args: argparse.Namespace,
    output_root: Path,
    start: datetime,
) -> None:
    raw_written = events_written = quarantined = 0
    for traj in trajectories:
        raw_record = make_raw_track_record(
            airport=profile.code,
            fetch_profile=args.fetch_profile,
            source_response_ids=["traffic.data.opensky.history"],
            traj=traj,
        )
        raw_path = partition_path(
            output_root, "raw_tracks", airport=profile.code,
            timestamp=datetime.fromtimestamp(traj.start_time, timezone.utc),
        ) / "tracks.jsonl"
        raw_written += write_jsonl_records(raw_path, [raw_record])

        events = extract_airport_events(
            traj,
            airport=profile.code,
            airport_lat=profile.lat,
            airport_lon=profile.lon,
            airport_elev_m=profile.elevation_m,
            radius_nm=args.airport_event_radius_nm,
            low_altitude_agl_m=args.low_altitude_agl_m,
        )
        for event in events:
            ready, quarantine = build_training_event(
                event, raw_track_id=raw_record["raw_track_id"], traj=traj, require_geo_altitude=True
            )
            entry_time = datetime.fromtimestamp(int(event["event_time"]["entry_time"]), timezone.utc)
            if ready is not None:
                path = partition_path(output_root, "airport_events", airport=profile.code, timestamp=entry_time) / "events.jsonl"
                events_written += write_jsonl_records(path, [ready])
            if quarantine is not None:
                path = partition_path(output_root, "quarantine", airport=profile.code, timestamp=entry_time) / "incomplete_events.jsonl"
                quarantined += write_jsonl_records(path, [quarantine])

    manifest = {
        "schema_version": "history-fetch-manifest-v3",
        "airport": profile.code,
        "fetch_profile": args.fetch_profile,
        "begin": int(start.timestamp()),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "radius_nm": args.airport_event_radius_nm,
        "counts": {
            "trajectories": len(trajectories),
            "raw_tracks_written": raw_written,
            "events_written": events_written,
            "quarantined": quarantined,
        },
    }
    manifest_path = partition_path(
        output_root, "manifests", airport=profile.code, timestamp=start, granularity="day"
    ) / "fetch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[training] trajectories: {len(trajectories)}")
    print(f"[training] raw tracks: {raw_written}  events: {events_written}  quarantined: {quarantined}")
    print(f"[training] manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download airport trajectories from the OpenSky history DB")
    p.add_argument("--airport", required=True, help="ICAO airport code, e.g. KRDU")
    p.add_argument("--begin", required=True, help="UTC begin time, ISO (2026-04-19T10:00:00Z) or Unix seconds")
    p.add_argument("--end", required=True, help="UTC end time, ISO or Unix seconds")
    p.add_argument("--dataset-mode", choices=["czml", "training"], default="czml")
    p.add_argument("--fetch-profile", choices=["airport_ops", "terminal_all"], default="airport_ops")

    p.add_argument("--runway", default=None, help="Keep only trajectories arriving at this runway threshold, e.g. 23R")
    p.add_argument("--runway-threshold-radius-m", type=float, default=600.0)

    p.add_argument("--match-radius-km", type=float, default=35.0, help="Trajectory must approach within this distance of the airport")
    p.add_argument("--max-end-distance-km", type=float, default=2.5, help="Arrival anchor must be within this distance of the airport (ignored when --runway is set)")
    p.add_argument("--approach-window-min", type=int, default=20, help="Keep this many minutes before the arrival anchor")
    p.add_argument("--exclude-ground", action="store_true", help="Drop on-ground points from exported trajectories")
    p.add_argument("--max-trajectories", type=int, default=80)

    p.add_argument("--airport-event-radius-nm", type=float, default=5.0, help="Training event radius (nautical miles)")
    p.add_argument("--low-altitude-agl-m", type=float, default=600.0, help="Low-altitude evidence threshold for training labels")
    p.add_argument("--segment-gap-sec", type=int, default=900, help="Split a trajectory when the time gap exceeds this")

    p.add_argument("--bbox-lat-pad", type=float, default=0.45, help="terminal_all latitude pad (degrees)")
    p.add_argument("--bbox-lon-pad", type=float, default=0.60, help="terminal_all longitude pad (degrees)")
    p.add_argument("--chunk-hours", type=float, default=1.0, help="Hours per history query chunk")
    p.add_argument("--no-traffic-cache", dest="cached", action="store_false", default=True)

    p.add_argument("--output-root", default=None, help="Output root (default: trajectory_data_process/outputs)")
    p.add_argument("--aeroviz-root", default=None, help="Path to aeroviz-4d (for airport/runway reference data)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else default_aeroviz_root(script_path)
    output_root = Path(args.output_root) if args.output_root else default_outputs_root(script_path)

    start, stop = parse_utc(args.begin), parse_utc(args.end)
    profile = resolve_airport_profile(args.airport, aeroviz_root)
    runway_threshold = (
        resolve_runway_threshold(profile.code, args.runway, runways_csv_path(aeroviz_root))
        if args.runway
        else None
    )

    df = fetch_history(
        start=start,
        stop=stop,
        airport=profile.code,
        profile=profile,
        fetch_profile=args.fetch_profile,
        chunk_hours=args.chunk_hours,
        bbox_lat_pad=args.bbox_lat_pad,
        bbox_lon_pad=args.bbox_lon_pad,
        cached=args.cached,
    )
    trajectories = build_trajectories_from_history(df, max_gap_sec=args.segment_gap_sec, require_geo_altitude=True)
    print(f"[history] rows: {len(df)}  trajectories: {len(trajectories)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.dataset_mode == "training":
        write_training(
            trajectories=trajectories, profile=profile, args=args, output_root=output_root, start=start
        )
        return

    output_dir = output_root / profile.code.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_czml_input(
        trajectories=trajectories,
        profile=profile,
        runway_threshold=runway_threshold,
        args=args,
        output_dir=output_dir,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
