"""Collect historical landing trajectories per runway threshold.

This is the reusable engine behind ``download_landings.py``. For one airport it
issues a single history query per time chunk and reuses the resulting trajectories
for every threshold, scanning backward in time until each threshold has the
requested number of landings (or a maximum lookback is reached).

Queries use a bounding box around the airport (``bbox_radius_km``) rather than the
full-track airport join, so only terminal-area state vectors are downloaded. Landing
detection keys off runway-heading alignment and descent geometry, not the flight's
arrival-airport metadata, so the bbox query (which omits that metadata) is sufficient.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.acquisition.airports import AirportProfile
from trajectory_data_process.acquisition.opensky_history import STATE_VECTOR_COLUMNS, fetch_history_dataframe
from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.geo import bounds_from_radius_km
from trajectory_data_process.processing.czml_export import trajectories_to_czml_input
from trajectory_data_process.trajectory import build_trajectories_from_history

DEFAULT_BBOX_RADIUS_KM = 30.0

FetchHistory = Callable[..., pd.DataFrame]


def load_runway_config(path: Path) -> dict[str, Any]:
    """Load the runway-threshold mapping JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def iter_airport_entries(config: dict[str, Any]) -> list[tuple[AirportProfile, list[RunwayThreshold]]]:
    """Turn the config JSON into (airport, thresholds) pairs."""
    entries: list[tuple[AirportProfile, list[RunwayThreshold]]] = []
    for code, airport in config["airports"].items():
        profile = AirportProfile(
            code=code, lat=airport["lat"], lon=airport["lon"], elevation_m=airport.get("elevation_m", 0.0)
        )
        thresholds = [
            RunwayThreshold(
                airport=code,
                ident=t["ident"],
                lat=t["lat"],
                lon=t["lon"],
                elevation_m=t.get("elevation_m") or airport.get("elevation_m", 0.0),
                heading_deg=t.get("heading_deg"),
            )
            for runway in airport["runways"]
            for t in runway["thresholds"]
        ]
        entries.append((profile, thresholds))
    return entries


def check_history_access(
    *,
    profile: AirportProfile,
    reference: datetime,
    bbox_radius_km: float = DEFAULT_BBOX_RADIUS_KM,
    fetch_history_fn: FetchHistory = fetch_history_dataframe,
) -> None:
    """Run one tiny probe query so a whole run fails fast on missing access.

    Uses the same bbox query shape as the real download over a 1-minute window two
    days before ``reference``. Raises the wrapped, actionable error if credentials or
    historical-data access are missing; returns on success.
    """
    stop = reference - timedelta(days=2)
    start = stop - timedelta(minutes=1)
    fetch_history_fn(
        start=start,
        stop=stop,
        bounds=bounds_from_radius_km(profile.lat, profile.lon, bbox_radius_km),
        selected_columns=STATE_VECTOR_COLUMNS,
        cached=True,
    )


def download_airport_landings(
    *,
    profile: AirportProfile,
    thresholds: list[RunwayThreshold],
    count: int,
    start: datetime,
    max_lookback_days: float,
    chunk_hours: float = 12.0,
    bbox_radius_km: float = DEFAULT_BBOX_RADIUS_KM,
    runway_threshold_radius_m: float = 1000.0,
    approach_window_min: int = 25,
    segment_gap_sec: int = 900,
    dry_give_up_chunks: int = 8,
    cached: bool = True,
    preloaded: dict[str, list[dict[str, Any]]] | None = None,
    fetch_history_fn: FetchHistory = fetch_history_dataframe,
) -> dict[str, list[dict[str, Any]]]:
    """Collect up to ``count`` landings for each threshold, scanning backward.

    Each chunk is fetched as a ``bbox_radius_km`` box around the airport. A threshold
    that yields no new landing for ``dry_give_up_chunks`` consecutive chunks is given
    up (idle runway ends would otherwise drag the whole airport back to
    ``max_lookback_days``); the scan stops once every threshold is at ``count`` or
    given up. ``preloaded`` seeds already-collected flights per threshold (for resume),
    de-duplicated by ``(icao24, landing_time_utc)``.
    """
    preloaded = preloaded or {}
    bounds = bounds_from_radius_km(profile.lat, profile.lon, bbox_radius_km)
    collected: dict[str, list[dict[str, Any]]] = {
        t.ident: list(preloaded.get(t.ident, []))[:count] for t in thresholds
    }
    seen: dict[str, set[tuple[str, str | None]]] = {
        t.ident: {(f["icao24"], f.get("landing_time_utc")) for f in collected[t.ident]}
        for t in thresholds
    }
    dry_chunks: dict[str, int] = {t.ident: 0 for t in thresholds}
    given_up: set[str] = set()

    def active(ident: str) -> bool:
        return len(collected[ident]) < count and ident not in given_up

    earliest = start - timedelta(days=max_lookback_days)
    cursor = start
    while cursor > earliest and any(active(t.ident) for t in thresholds):
        chunk_start = max(cursor - timedelta(hours=chunk_hours), earliest)
        print(
            f"[landings] {profile.code} {chunk_start.isoformat()} -> {cursor.isoformat()} "
            f"(have {{{_progress(collected, count)}}})",
            flush=True,
        )
        df = fetch_history_fn(
            start=chunk_start,
            stop=cursor,
            bounds=bounds,
            selected_columns=STATE_VECTOR_COLUMNS,
            cached=cached,
        )
        trajectories = build_trajectories_from_history(df, max_gap_sec=segment_gap_sec)

        for threshold in thresholds:
            if not active(threshold.ident):
                continue
            before = len(collected[threshold.ident])
            flights = trajectories_to_czml_input(
                trajectories,
                airport_lat=profile.lat,
                airport_lon=profile.lon,
                runway_threshold=threshold,
                runway_threshold_radius_m=runway_threshold_radius_m,
                landing_only=True,
                approach_window_min=approach_window_min,
                max_flights=count - before + 10,
            )
            for flight in flights:
                key = (flight["icao24"], flight.get("landing_time_utc"))
                if key in seen[threshold.ident]:
                    continue
                seen[threshold.ident].add(key)
                collected[threshold.ident].append(flight)
                if len(collected[threshold.ident]) >= count:
                    break

            if len(collected[threshold.ident]) > before:
                dry_chunks[threshold.ident] = 0
            else:
                dry_chunks[threshold.ident] += 1
                if dry_chunks[threshold.ident] >= dry_give_up_chunks:
                    given_up.add(threshold.ident)
                    print(f"[landings] {profile.code} {threshold.ident}: dry for "
                          f"{dry_give_up_chunks} chunks, giving up at "
                          f"{len(collected[threshold.ident])}/{count}", flush=True)

        cursor = chunk_start

    return collected


def _progress(collected: dict[str, list[dict[str, Any]]], count: int) -> str:
    return ", ".join(f"{ident}:{len(flights)}/{count}" for ident, flights in collected.items())
