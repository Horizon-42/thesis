"""Collect historical landing trajectories per runway threshold.

This is the reusable engine behind ``download_landings.py``. For one airport it
issues a single history query per time chunk and reuses the resulting trajectories
for every threshold, scanning backward in time until each threshold has the
requested number of landings (or a maximum lookback is reached).

Queries use a bounding box around the airport (``radius_km``) rather than the
full-track airport join, so only terminal-area state vectors are downloaded. Landing
detection keys off runway-heading alignment and descent geometry, not the flight's
arrival-airport metadata, so the bbox query (which omits that metadata) is sufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from trajectory_data_process.processing.czml_export import (
    DEFAULT_HEADING_TOLERANCE_DEG,
    classify_landing_flights,
)
from trajectory_data_process.trajectory import build_trajectories_from_history

DEFAULT_RADIUS_KM = 30.0

# Internal scan/detection knobs — not CLI-exposed (implementation details, not
# research requirements). Change here if a run ever needs them tuned.
DEFAULT_CHUNK_HOURS = 6.0            # hours per history query (Trino batching)
DEFAULT_DRY_GIVE_UP_DAYS = 4.0       # give up a runway after this long with no new landing
RUNWAY_THRESHOLD_RADIUS_M = 1000.0   # a landing's closest point must fall this near the threshold

FetchHistory = Callable[..., pd.DataFrame]


@dataclass
class LandingHarvest:
    """Per-threshold landings, split by the runway-direction test.

    ``accepted`` are landings whose approach direction lines up with the runway.
    ``rejected`` otherwise look like landings (near the threshold, low, descending)
    but their direction disagrees; they are kept — tagged with the measured heading
    errors — so a run can be audited for false kills instead of dropping them.
    """

    accepted: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rejected: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


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
    radius_km: float = DEFAULT_RADIUS_KM,
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
        bounds=bounds_from_radius_km(profile.lat, profile.lon, radius_km),
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
    chunk_hours: float = DEFAULT_CHUNK_HOURS,
    radius_km: float = DEFAULT_RADIUS_KM,
    runway_threshold_radius_m: float = RUNWAY_THRESHOLD_RADIUS_M,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
    segment_gap_sec: int = 900,
    dry_give_up_days: float = DEFAULT_DRY_GIVE_UP_DAYS,
    cached: bool = True,
    preloaded: dict[str, list[dict[str, Any]]] | None = None,
    fetch_history_fn: FetchHistory = fetch_history_dataframe,
) -> LandingHarvest:
    """Collect up to ``count`` landings for each threshold, scanning backward.

    Each chunk is fetched as a ``radius_km`` box around the airport, and each kept
    landing's waypoints are cropped to that same ``radius_km`` circle (so the query
    footprint and the exported track share one radius). A threshold
    is given up once the scan has gone ``dry_give_up_days`` past its last new landing
    (idle runway ends would otherwise drag the whole airport back to
    ``max_lookback_days``); this depth is a fixed duration, independent of
    ``chunk_hours``. The scan stops once every threshold is at ``count`` or given up.
    ``preloaded`` seeds already-collected flights per threshold (for resume),
    de-duplicated by ``(icao24, landing_time_utc)``.

    Returns a :class:`LandingHarvest`: the accepted landings plus the ones the
    runway-direction test (``heading_tolerance_deg``) set aside, so a run can be
    reviewed for false kills. Only accepted landings count toward ``count`` and drive
    the scan; the rejected list is best-effort review data (not resumed via
    ``preloaded``).
    """
    preloaded = preloaded or {}
    bounds = bounds_from_radius_km(profile.lat, profile.lon, radius_km)
    collected: dict[str, list[dict[str, Any]]] = {
        t.ident: list(preloaded.get(t.ident, []))[:count] for t in thresholds
    }
    rejected: dict[str, list[dict[str, Any]]] = {t.ident: [] for t in thresholds}
    seen: dict[str, set[tuple[str, str | None]]] = {
        t.ident: {(f["icao24"], f.get("landing_time_utc")) for f in collected[t.ident]}
        for t in thresholds
    }
    seen_rejected: dict[str, set[tuple[str, str | None]]] = {t.ident: set() for t in thresholds}
    # The deepest time each threshold last found a landing (init: top of the scan).
    # When the scan goes more than dry_give_up_days past it, the threshold is given up.
    dry_floor: dict[str, datetime] = {t.ident: start for t in thresholds}
    give_up_span = timedelta(days=dry_give_up_days)
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
            accepted, heading_rejected = classify_landing_flights(
                trajectories,
                airport_lat=profile.lat,
                airport_lon=profile.lon,
                runway_threshold=threshold,
                runway_threshold_radius_m=runway_threshold_radius_m,
                heading_tolerance_deg=heading_tolerance_deg,
                crop_radius_km=radius_km,
                max_accepted=count - before + 10,
            )
            for flight in accepted:
                key = (flight["icao24"], flight.get("landing_time_utc"))
                if key in seen[threshold.ident]:
                    continue
                seen[threshold.ident].add(key)
                collected[threshold.ident].append(flight)
                if len(collected[threshold.ident]) >= count:
                    break
            for flight in heading_rejected:
                key = (flight["icao24"], flight.get("landing_time_utc"))
                if key in seen_rejected[threshold.ident]:
                    continue
                seen_rejected[threshold.ident].add(key)
                rejected[threshold.ident].append(flight)

            if len(collected[threshold.ident]) > before:
                dry_floor[threshold.ident] = chunk_start
            elif dry_floor[threshold.ident] - chunk_start >= give_up_span:
                given_up.add(threshold.ident)
                print(f"[landings] {profile.code} {threshold.ident}: no landings in the last "
                      f"{dry_give_up_days:g} days scanned, giving up at "
                      f"{len(collected[threshold.ident])}/{count}", flush=True)

        cursor = chunk_start

    return LandingHarvest(accepted=collected, rejected=rejected)


def _progress(collected: dict[str, list[dict[str, Any]]], count: int) -> str:
    return ", ".join(f"{ident}:{len(flights)}/{count}" for ident, flights in collected.items())
