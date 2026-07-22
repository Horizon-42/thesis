"""The scan loop: fetch backward in time until every runway has enough, then store.

ORDER MATTERS -- FETCH, RECONSTRUCT, THEN ASSIGN
------------------------------------------------
The predecessor classified inside the download loop, once per threshold per chunk. That
ordering caused two of the defects the audit found: a track straddling a chunk boundary
was classified from a partial view, and because each threshold was tested independently
one landing could be written into several runways' files (measured: 72.7% of KSJC's
landings). Here a chunk is only ever ACCUMULATED; reconstruction and assignment run once
at the end over the merged rows, so every track is judged whole and exactly once.

THE QUOTA IS COUNTED ON LANDINGS, NOT ON ESTABLISHED APPROACHES
---------------------------------------------------------------
It is tempting to scan until each runway has N flights that pass the established-on-final
criterion. Two reasons not to:

  * only 21-54% of real arrivals are established by that criterion, so an established
    quota needs 2-5x more history -- and the deeper the scan runs, the more the fleet mix
    drifts away from the period actually under study;
  * more importantly, the established RATE is a headline result of this project. Making
    the stopping rule depend on it invites the selection to leak into the number.

So the loop stops on assigned landings and merely REPORTS how many were established.
That keeps the denominator honest: every fetched track is stored and counted, in one of
the four buckets, whatever it turned out to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from final_approach import LandingScreen

from trajectory_data_process.acquisition.opensky_history import (
    STATE_VECTOR_COLUMNS,
    fetch_history_dataframe,
)
from trajectory_data_process.geo import bounds_from_radius_km
from trajectory_data_process.harvest.airports import Airport
from trajectory_data_process.harvest.classify import ClassifiedTrack, classify_tracks
from trajectory_data_process.harvest.store import HarvestPaths, write_tracks
from trajectory_data_process.harvest.tracks import (
    DEFAULT_CROP_RADIUS_KM,
    reconstruct_tracks,
)

DEFAULT_CHUNK_HOURS = 6.0
DEFAULT_MAX_LOOKBACK_DAYS = 30.0
# Give a runway up once the scan has gone this long past its last new landing: an idle
# runway end would otherwise drag the whole airport to the lookback limit.
DEFAULT_DRY_GIVE_UP_DAYS = 4.0


@dataclass
class HarvestPlan:
    """Everything the loop needs that is not the airport itself."""

    target_per_runway: int = 200
    start: datetime | None = None
    chunk_hours: float = DEFAULT_CHUNK_HOURS
    max_lookback_days: float = DEFAULT_MAX_LOOKBACK_DAYS
    dry_give_up_days: float = DEFAULT_DRY_GIVE_UP_DAYS
    radius_km: float = DEFAULT_CROP_RADIUS_KM
    screen: LandingScreen = field(default_factory=LandingScreen)
    cached: bool = True


@dataclass
class HarvestResult:
    classified: list[ClassifiedTrack]
    manifest: dict[str, Any]
    chunks_fetched: int
    scanned_from: datetime
    scanned_to: datetime


def harvest_airport(
    airport: Airport,
    paths: HarvestPaths,
    plan: HarvestPlan,
    *,
    fetch: Callable[..., Any] = fetch_history_dataframe,
    log: Callable[[str], None] = print,
) -> HarvestResult:
    """Scan backward until each runway reaches its target, then reconstruct and store."""
    stop = plan.start or datetime.now(tz=_utc())
    earliest = stop - timedelta(days=plan.max_lookback_days)
    bounds = bounds_from_radius_km(airport.lat, airport.lon, plan.radius_km)

    rows: list[dict[str, Any]] = []
    per_runway: dict[str, int] = {r.ident: 0 for r in airport.runways}
    last_new: dict[str, datetime] = {r.ident: stop for r in airport.runways}
    given_up: set[str] = set()
    classified: list[ClassifiedTrack] = []
    cursor = stop
    chunks = 0

    def unfinished() -> list[str]:
        return [i for i, n in per_runway.items() if n < plan.target_per_runway and i not in given_up]

    while cursor > earliest and unfinished():
        chunk_start = max(cursor - timedelta(hours=plan.chunk_hours), earliest)
        log(
            f"[harvest] {airport.code} {chunk_start.isoformat()} -> {cursor.isoformat()} "
            f"({_progress(per_runway, plan.target_per_runway)})"
        )
        frame = fetch(
            start=chunk_start,
            stop=cursor,
            bounds=bounds,
            selected_columns=STATE_VECTOR_COLUMNS,
            cached=plan.cached,
        )
        chunks += 1
        rows.extend(_records(frame))

        # Re-run over ALL rows accumulated so far: a track that straddles a chunk
        # boundary is only whole once both chunks are in hand, and assignment must never
        # see a partial approach.
        classified = classify_tracks(
            reconstruct_tracks(
                rows,
                airport_lat=airport.lat,
                airport_lon=airport.lon,
                crop_radius_km=plan.radius_km,
                altitude_units="m",  # fetch_history_dataframe normalises feet -> metres
            ),
            airport,
            screen=plan.screen,
        )
        counted = _count(classified)
        for ident, count in counted.items():
            if count > per_runway.get(ident, 0):
                last_new[ident] = chunk_start
            per_runway[ident] = count
        for ident in list(unfinished()):
            if stop - last_new[ident] > timedelta(days=plan.dry_give_up_days):
                given_up.add(ident)
                log(f"[harvest] {airport.code} {ident}: no new landing in "
                    f"{plan.dry_give_up_days:g} days of scan — giving up at {per_runway[ident]}")
        cursor = chunk_start

    manifest = write_tracks(
        classified,
        paths,
        provenance={
            # This is the CLI --start anchor. Reusing it reproduces the exact chunk
            # boundaries and lets pyopensky match its per-query cache entries.
            "start_utc": stop.isoformat(),
            "scanned_from_utc": cursor.isoformat(),
            "scanned_to_utc": stop.isoformat(),
            "chunks_fetched": chunks,
            "radius_km": plan.radius_km,
            "target_per_runway": plan.target_per_runway,
            "given_up": sorted(given_up),
            "landing_screen": {
                "threshold_radius_m": plan.screen.threshold_radius_m,
                "max_height_m": plan.screen.max_height_m,
                "descent_margin_m": plan.screen.descent_margin_m,
            },
        },
    )
    return HarvestResult(
        classified=classified,
        manifest=manifest,
        chunks_fetched=chunks,
        scanned_from=cursor,
        scanned_to=stop,
    )


def _count(classified: Sequence[ClassifiedTrack]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in classified:
        if item.runway:
            counts[item.runway] = counts.get(item.runway, 0) + 1
    return counts


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None or getattr(frame, "empty", False):
        return []
    return frame.to_dict(orient="records")


def _progress(per_runway: dict[str, int], target: int) -> str:
    return " ".join(f"{k}:{v}/{target}" for k, v in sorted(per_runway.items()))


def _utc():
    from datetime import timezone

    return timezone.utc
