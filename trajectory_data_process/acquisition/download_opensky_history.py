#!/usr/bin/env python3
"""Download OpenSky history DB rows through traffic/Trino in bounded chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.acquisition.fetch_cylw_opensky import (
    default_aeroviz_root,
    default_outputs_root,
    parse_time_to_unix,
    resolve_airport_profile,
    traffic_bounds_from_airport_bbox,
)
from trajectory_data_process.acquisition.opensky_history_db import (
    AIRPORT_HISTORY_COLUMNS,
    STATE_VECTOR_COLUMNS,
    fetch_history_dataframe,
    write_history_rows,
)
from trajectory_data_process.datasets.dataset_store import ensure_utc, partition_path


FetchHistory = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class HistoryDownloadTask:
    """One small OpenSky history query planned for download."""

    airport: str
    query: str
    query_name: str
    start: datetime
    stop: datetime
    selected_columns: tuple[str, ...]
    airport_filter: str | None = None
    bounds: tuple[float, float, float, float] | None = None


def parse_utc_datetime(value: str) -> datetime:
    """Parse a Unix timestamp or ISO datetime as UTC."""
    return datetime.fromtimestamp(parse_time_to_unix(value), timezone.utc)


def utc_tag(value: datetime) -> str:
    """Return a compact stable UTC timestamp tag."""
    return ensure_utc(value).strftime("%Y%m%dT%H%M%SZ")


def iter_time_chunks(
    start: datetime,
    stop: datetime,
    *,
    chunk_hours: float,
    max_chunks: int | None = None,
) -> list[tuple[datetime, datetime]]:
    """Split a time range into positive UTC chunks."""
    start_utc = ensure_utc(start)
    stop_utc = ensure_utc(stop)
    if stop_utc <= start_utc:
        raise ValueError("--end must be later than --begin")
    if chunk_hours <= 0:
        raise ValueError("--chunk-hours must be positive")

    step = timedelta(hours=chunk_hours)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start_utc
    while cursor < stop_utc:
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        next_stop = min(cursor + step, stop_utc)
        chunks.append((cursor, next_stop))
        cursor = next_stop
    return chunks


def build_download_tasks(
    *,
    airport: str,
    start: datetime,
    stop: datetime,
    fetch_profile: str,
    chunk_hours: float,
    run_id: str,
    bbox_lat_pad: float,
    bbox_lon_pad: float,
    airport_lat: float | None = None,
    airport_lon: float | None = None,
    max_chunks: int | None = None,
) -> list[HistoryDownloadTask]:
    """Build airport operation and optional terminal-area history tasks."""
    airport_code = airport.upper()
    if fetch_profile not in {"airport_ops", "terminal_all"}:
        raise ValueError(f"Unsupported fetch profile: {fetch_profile}")
    if fetch_profile == "terminal_all" and (airport_lat is None or airport_lon is None):
        raise ValueError("terminal_all requires airport_lat and airport_lon")

    tasks: list[HistoryDownloadTask] = []
    for chunk_start, chunk_stop in iter_time_chunks(
        start,
        stop,
        chunk_hours=chunk_hours,
        max_chunks=max_chunks,
    ):
        start_tag = utc_tag(chunk_start)
        stop_tag = utc_tag(chunk_stop)
        tasks.append(
            HistoryDownloadTask(
                airport=airport_code,
                query="airport_ops",
                query_name=f"airport_ops_{start_tag}_{stop_tag}_{run_id}",
                start=chunk_start,
                stop=chunk_stop,
                selected_columns=AIRPORT_HISTORY_COLUMNS,
                airport_filter=airport_code,
            )
        )
        if fetch_profile == "terminal_all":
            assert airport_lat is not None and airport_lon is not None
            bounds = traffic_bounds_from_airport_bbox(
                airport_lat=airport_lat,
                airport_lon=airport_lon,
                lat_pad=bbox_lat_pad,
                lon_pad=bbox_lon_pad,
            )
            tasks.append(
                HistoryDownloadTask(
                    airport=airport_code,
                    query="terminal_area",
                    query_name=f"terminal_area_{start_tag}_{stop_tag}_{run_id}",
                    start=chunk_start,
                    stop=chunk_stop,
                    selected_columns=STATE_VECTOR_COLUMNS,
                    bounds=bounds,
                )
            )
    return tasks


def task_manifest_record(task: HistoryDownloadTask) -> dict[str, Any]:
    """Serialize task metadata for dry-run output and manifests."""
    params: dict[str, Any] = {
        "start": task.start.isoformat().replace("+00:00", "Z"),
        "stop": task.stop.isoformat().replace("+00:00", "Z"),
        "selected_columns": list(task.selected_columns),
    }
    if task.airport_filter is not None:
        params["airport"] = task.airport_filter
    if task.bounds is not None:
        west, south, east, north = task.bounds
        params["bounds"] = {"west": west, "south": south, "east": east, "north": north}
    return {
        "airport": task.airport,
        "query": task.query,
        "query_name": task.query_name,
        "params": params,
    }


def write_download_manifest(
    output_root: Path,
    *,
    airport: str,
    start: datetime,
    run_id: str,
    manifest: dict[str, Any],
) -> Path:
    """Write one manifest for an airport download run."""
    path = (
        partition_path(output_root, "manifests", airport=airport, timestamp=start, granularity="day")
        / f"opensky_history_download_{run_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_download_for_airport(
    *,
    airport: str,
    start: datetime,
    stop: datetime,
    output_root: Path,
    fetch_profile: str,
    chunk_hours: float,
    bbox_lat_pad: float,
    bbox_lon_pad: float,
    aeroviz_root: Path,
    cached: bool,
    max_chunks: int | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
    fetcher: FetchHistory = fetch_history_dataframe,
) -> dict[str, Any]:
    """Execute a bounded history download for one airport."""
    airport_code = airport.upper()
    created_at_utc = ensure_utc(created_at or datetime.now(timezone.utc))
    run_id_value = run_id or created_at_utc.strftime("%Y%m%dT%H%M%S%fZ")
    airport_lat: float | None = None
    airport_lon: float | None = None
    airport_elev_m: float | None = None
    if fetch_profile == "terminal_all":
        airport_lat, airport_lon, airport_elev_m = resolve_airport_profile(airport_code, aeroviz_root)

    tasks = build_download_tasks(
        airport=airport_code,
        start=start,
        stop=stop,
        fetch_profile=fetch_profile,
        chunk_hours=chunk_hours,
        run_id=run_id_value,
        bbox_lat_pad=bbox_lat_pad,
        bbox_lon_pad=bbox_lon_pad,
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        max_chunks=max_chunks,
    )

    outputs: list[dict[str, Any]] = []
    total_rows = 0
    for index, task in enumerate(tasks, start=1):
        print(
            f"[OpenSky] {airport_code} {index}/{len(tasks)} {task.query} "
            f"{task.start.isoformat()} -> {task.stop.isoformat()}",
            flush=True,
        )
        df = fetcher(
            start=task.start,
            stop=task.stop,
            airport=task.airport_filter,
            bounds=task.bounds,
            selected_columns=task.selected_columns,
            cached=cached,
        )
        rows_path = write_history_rows(
            output_root,
            airport=airport_code,
            df=df,
            fetched_at=created_at_utc,
            query_name=task.query_name,
        )
        count = int(len(df))
        total_rows += count
        record = task_manifest_record(task)
        record.update({"count": count, "path": str(rows_path)})
        outputs.append(record)
        print(f"[OpenSky] wrote {count} rows: {rows_path}", flush=True)

    manifest = {
        "schema_version": "opensky-history-download-manifest-v1",
        "airport": airport_code,
        "begin": ensure_utc(start).isoformat().replace("+00:00", "Z"),
        "end": ensure_utc(stop).isoformat().replace("+00:00", "Z"),
        "created_at_utc": created_at_utc.isoformat().replace("+00:00", "Z"),
        "run_id": run_id_value,
        "fetch_profile": fetch_profile,
        "chunk_hours": chunk_hours,
        "cached": cached,
        "bbox": None
        if airport_lat is None or airport_lon is None
        else {
            "airport_lat": airport_lat,
            "airport_lon": airport_lon,
            "airport_elev_m": airport_elev_m,
            "lat_pad": bbox_lat_pad,
            "lon_pad": bbox_lon_pad,
        },
        "task_count": len(tasks),
        "history_rows": {"count": total_rows, "outputs": outputs},
    }
    manifest_path = write_download_manifest(
        output_root,
        airport=airport_code,
        start=start,
        run_id=run_id_value,
        manifest=manifest,
    )
    manifest["manifest_path"] = str(manifest_path)
    print(f"[OpenSky] manifest: {manifest_path}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenSky history DB rows through traffic/Trino")
    parser.add_argument("--airports", nargs="+", required=True, help="ICAO airport codes, e.g. KRDU KSJC")
    parser.add_argument("--begin", required=True, help="UTC begin time, ISO or Unix seconds")
    parser.add_argument("--end", required=True, help="UTC end time, ISO or Unix seconds")
    parser.add_argument("--fetch-profile", choices=["airport_ops", "terminal_all"], default="airport_ops")
    parser.add_argument("--chunk-hours", type=float, default=1.0, help="Hours per Trino query chunk")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit time chunks per airport for smoke runs")
    parser.add_argument("--bbox-lat-pad", type=float, default=0.45, help="terminal_all latitude pad in degrees")
    parser.add_argument("--bbox-lon-pad", type=float, default=0.60, help="terminal_all longitude pad in degrees")
    parser.add_argument(
        "--output-root",
        default=str(default_outputs_root(Path(__file__).resolve())),
        help="Output root for history_rows and manifests",
    )
    parser.add_argument(
        "--aeroviz-root",
        default=str(default_aeroviz_root(Path(__file__).resolve())),
        help="AeroViz root used to resolve airport centers for terminal_all",
    )
    parser.add_argument("--no-traffic-cache", dest="cached", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks without querying Trino")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc_datetime(args.begin)
    stop = parse_utc_datetime(args.end)
    output_root = Path(args.output_root)
    aeroviz_root = Path(args.aeroviz_root)
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")

    if args.dry_run:
        planned: list[dict[str, Any]] = []
        for airport in args.airports:
            airport_lat: float | None = None
            airport_lon: float | None = None
            if args.fetch_profile == "terminal_all":
                airport_lat, airport_lon, _airport_elev_m = resolve_airport_profile(airport.upper(), aeroviz_root)
            planned.extend(
                task_manifest_record(task)
                for task in build_download_tasks(
                    airport=airport,
                    start=start,
                    stop=stop,
                    fetch_profile=args.fetch_profile,
                    chunk_hours=args.chunk_hours,
                    run_id=run_id,
                    bbox_lat_pad=args.bbox_lat_pad,
                    bbox_lon_pad=args.bbox_lon_pad,
                    airport_lat=airport_lat,
                    airport_lon=airport_lon,
                    max_chunks=args.max_chunks,
                )
            )
        print(json.dumps({"schema_version": "opensky-history-download-plan-v1", "tasks": planned}, indent=2))
        return

    manifests = [
        run_download_for_airport(
            airport=airport,
            start=start,
            stop=stop,
            output_root=output_root,
            fetch_profile=args.fetch_profile,
            chunk_hours=args.chunk_hours,
            bbox_lat_pad=args.bbox_lat_pad,
            bbox_lon_pad=args.bbox_lon_pad,
            aeroviz_root=aeroviz_root,
            cached=args.cached,
            max_chunks=args.max_chunks,
            run_id=run_id,
            created_at=created_at,
        )
        for airport in args.airports
    ]
    print(json.dumps({"schema_version": "opensky-history-download-run-v1", "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
