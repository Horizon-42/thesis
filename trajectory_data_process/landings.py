"""Collect historical landing trajectories per runway threshold.

This is the reusable engine behind ``download_landings.py``. For one airport it
issues a single history query per time chunk and reuses the resulting trajectories
for every threshold, scanning backward in time until each threshold has the
requested number of landings (or a maximum lookback is reached).
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
from trajectory_data_process.acquisition.opensky_history import AIRPORT_HISTORY_COLUMNS, fetch_history_dataframe
from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.processing.czml_export import trajectories_to_czml_input
from trajectory_data_process.trajectory import build_trajectories_from_history

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


def download_airport_landings(
    *,
    profile: AirportProfile,
    thresholds: list[RunwayThreshold],
    count: int,
    start: datetime,
    max_lookback_days: float,
    chunk_hours: float = 6.0,
    runway_threshold_radius_m: float = 600.0,
    approach_window_min: int = 25,
    segment_gap_sec: int = 900,
    cached: bool = True,
    fetch_history_fn: FetchHistory = fetch_history_dataframe,
) -> dict[str, list[dict[str, Any]]]:
    """Collect up to ``count`` landings for each threshold, scanning backward.

    Returns a mapping of runway-threshold ident to a list of CZML-input flights.
    """
    collected: dict[str, list[dict[str, Any]]] = {t.ident: [] for t in thresholds}
    seen: dict[str, set[tuple[str, str | None]]] = {t.ident: set() for t in thresholds}

    earliest = start - timedelta(days=max_lookback_days)
    cursor = start
    while cursor > earliest and any(len(collected[t.ident]) < count for t in thresholds):
        chunk_start = max(cursor - timedelta(hours=chunk_hours), earliest)
        print(
            f"[landings] {profile.code} {chunk_start.isoformat()} -> {cursor.isoformat()} "
            f"(have {{{_progress(collected, count)}}})",
            flush=True,
        )
        df = fetch_history_fn(
            start=chunk_start,
            stop=cursor,
            airport=profile.code,
            selected_columns=AIRPORT_HISTORY_COLUMNS,
            cached=cached,
        )
        trajectories = build_trajectories_from_history(df, max_gap_sec=segment_gap_sec)

        for threshold in thresholds:
            remaining = count - len(collected[threshold.ident])
            if remaining <= 0:
                continue
            flights = trajectories_to_czml_input(
                trajectories,
                airport_lat=profile.lat,
                airport_lon=profile.lon,
                runway_threshold=threshold,
                runway_threshold_radius_m=runway_threshold_radius_m,
                landing_only=True,
                approach_window_min=approach_window_min,
                max_flights=remaining + 10,
            )
            for flight in flights:
                key = (flight["icao24"], flight.get("landing_time_utc"))
                if key in seen[threshold.ident]:
                    continue
                seen[threshold.ident].add(key)
                collected[threshold.ident].append(flight)
                if len(collected[threshold.ident]) >= count:
                    break

        cursor = chunk_start

    return collected


def _progress(collected: dict[str, list[dict[str, Any]]], count: int) -> str:
    return ", ".join(f"{ident}:{len(flights)}/{count}" for ident, flights in collected.items())
