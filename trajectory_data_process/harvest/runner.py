"""The scan loop: fetch backward in time until every runway has enough, then store.

ORDER MATTERS -- FETCH, RECONSTRUCT, THEN ASSIGN
------------------------------------------------
The predecessor classified inside the download loop, once per threshold per chunk. That
ordering caused two of the defects the audit found: a track straddling a chunk boundary
was classified from a partial view, and because each threshold was tested independently
one landing could be written into several runways' files (measured: 72.7% of KSJC's
landings). Here rows are accumulated on disk. Each new chunk reprocesses the complete
history of only the aircraft it touches, and the final store is streamed one aircraft at
a time, so boundary tracks stay whole without holding the airport's history in memory.

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

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Sequence

from final_approach import LandingScreen

from trajectory_data_process.acquisition.opensky_history import (
    STATE_VECTOR_COLUMNS,
    fetch_history_dataframe,
)
from trajectory_data_process.geo import bounds_from_radius_km
from trajectory_data_process.harvest.airports import Airport
from trajectory_data_process.harvest.classify import ClassifiedTrack, classify_tracks
from trajectory_data_process.harvest.history_store import DiskHistoryStore
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
CHECKPOINT_VERSION = 1


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

    per_runway: dict[str, int] = {r.ident: 0 for r in airport.runways}
    per_aircraft: dict[str, dict[str, int]] = {}
    last_new: dict[str, datetime] = {r.ident: stop for r in airport.runways}
    given_up: set[str] = set()
    cursor = stop
    chunks = 0

    def unfinished() -> list[str]:
        return [i for i, n in per_runway.items() if n < plan.target_per_runway and i not in given_up]

    checkpoint = _load_checkpoint(paths, airport, plan, stop)
    if checkpoint is not None:
        cursor = _parse_utc(checkpoint["cursor_utc"])
        chunks = int(checkpoint["chunks_fetched"])
        per_runway = _integer_counts(checkpoint["per_runway"], per_runway)
        per_aircraft = {
            str(icao24): _integer_counts(counts)
            for icao24, counts in checkpoint["per_aircraft"].items()
            if isinstance(counts, dict)
        }
        last_new = {
            ident: _parse_utc(checkpoint["last_new"].get(ident, stop.isoformat()))
            for ident in per_runway
        }
        given_up = {
            str(ident) for ident in checkpoint["given_up"] if ident in per_runway
        }
        log(
            f"[harvest] {airport.code}: resuming checkpoint at {cursor.isoformat()} "
            f"after {chunks} chunks"
        )
    else:
        clear_harvest_checkpoint(paths)
        paths.checkpoint.mkdir(parents=True, exist_ok=True)
        _write_checkpoint(
            paths,
            airport=airport,
            plan=plan,
            stop=stop,
            cursor=cursor,
            chunks=chunks,
            per_runway=per_runway,
            per_aircraft=per_aircraft,
            last_new=last_new,
            given_up=given_up,
        )

    with DiskHistoryStore(paths.checkpoint_db) as history:
        while cursor > earliest and unfinished():
            chunk_start = max(cursor - timedelta(hours=plan.chunk_hours), earliest)
            log(
                f"[harvest] {airport.code} {chunk_start.isoformat()} -> {cursor.isoformat()} "
                f"({_progress(per_runway, plan.target_per_runway, given_up)})"
            )
            frame = fetch(
                start=chunk_start,
                stop=cursor,
                bounds=bounds,
                selected_columns=STATE_VECTOR_COLUMNS,
                cached=plan.cached,
            )
            chunks += 1
            affected = history.add_frame(frame)
            del frame

            previous = dict(per_runway)
            for icao24 in affected:
                for ident, count in per_aircraft.get(icao24, {}).items():
                    per_runway[ident] = per_runway.get(ident, 0) - count
                counted = _count(_classify_aircraft(history, icao24, airport, plan))
                if counted:
                    per_aircraft[icao24] = counted
                else:
                    per_aircraft.pop(icao24, None)
                for ident, count in counted.items():
                    per_runway[ident] = per_runway.get(ident, 0) + count

            for ident, count in per_runway.items():
                if count > previous.get(ident, 0):
                    last_new[ident] = chunk_start
            for ident in list(unfinished()):
                # The scan moves backward. ``last_new`` is therefore later than the
                # current oldest boundary; their difference is the dry scan window.
                if last_new[ident] - chunk_start >= timedelta(
                    days=plan.dry_give_up_days
                ):
                    given_up.add(ident)
                    log(
                        f"[harvest] {airport.code} {ident}: no new landing in "
                        f"{plan.dry_give_up_days:g} days of scan — giving up at "
                        f"{per_runway[ident]}"
                    )
            cursor = chunk_start
            _write_checkpoint(
                paths,
                airport=airport,
                plan=plan,
                stop=stop,
                cursor=cursor,
                chunks=chunks,
                per_runway=per_runway,
                per_aircraft=per_aircraft,
                last_new=last_new,
                given_up=given_up,
            )

        manifest = write_tracks(
            _iter_classified(history, airport, plan),
            paths,
            provenance={
                # This is the CLI --start anchor. Reusing it reproduces the exact
                # chunk boundaries and lets pyopensky match its query cache entries.
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
    clear_harvest_checkpoint(paths)
    return HarvestResult(
        # Classified tracks are intentionally streamed to disk rather than retained.
        classified=[],
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


def _classify_aircraft(
    history: DiskHistoryStore,
    icao24: str,
    airport: Airport,
    plan: HarvestPlan,
) -> list[ClassifiedTrack]:
    tracks = reconstruct_tracks(
        history.rows_for(icao24),
        airport_lat=airport.lat,
        airport_lon=airport.lon,
        crop_radius_km=plan.radius_km,
        altitude_units="m",  # fetch_history_dataframe returns metres
    )
    return classify_tracks(tracks, airport, screen=plan.screen)


def _iter_classified(
    history: DiskHistoryStore,
    airport: Airport,
    plan: HarvestPlan,
) -> Iterator[ClassifiedTrack]:
    for icao24 in history.aircraft():
        yield from _classify_aircraft(history, icao24, airport, plan)


def _progress(per_runway: dict[str, int], target: int, given_up: set[str]) -> str:
    return " ".join(
        f"{ident}:{count}/{target}" + ("[given-up]" if ident in given_up else "")
        for ident, count in sorted(per_runway.items())
    )


def checkpoint_start(paths: HarvestPaths) -> datetime | None:
    """Return an interrupted harvest's stable start anchor, if its state is readable."""
    try:
        state = json.loads(paths.checkpoint_state.read_text(encoding="utf-8"))
        if state.get("version") != CHECKPOINT_VERSION:
            return None
        return _parse_utc(state["start_utc"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def clear_harvest_checkpoint(paths: HarvestPaths) -> None:
    """Remove only this airport's known checkpoint files."""
    for name in (
        "state.json.tmp",
        "state.json",
        "history.sqlite-journal",
        "history.sqlite-shm",
        "history.sqlite-wal",
        "history.sqlite",
    ):
        (paths.checkpoint / name).unlink(missing_ok=True)
    try:
        paths.checkpoint.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _load_checkpoint(
    paths: HarvestPaths,
    airport: Airport,
    plan: HarvestPlan,
    stop: datetime,
) -> dict[str, Any] | None:
    try:
        state = json.loads(paths.checkpoint_state.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not paths.checkpoint_db.exists():
        return None
    try:
        compatible = (
            state["version"] == CHECKPOINT_VERSION
            and state["airport"] == airport.code
            and _parse_utc(state["start_utc"]) == stop
            and float(state["radius_km"]) == plan.radius_km
            and float(state["chunk_hours"]) == plan.chunk_hours
            and isinstance(state["per_runway"], dict)
            and isinstance(state["per_aircraft"], dict)
            and isinstance(state["last_new"], dict)
            and isinstance(state["given_up"], list)
        )
    except (KeyError, TypeError, ValueError):
        return None
    return state if compatible else None


def _write_checkpoint(
    paths: HarvestPaths,
    *,
    airport: Airport,
    plan: HarvestPlan,
    stop: datetime,
    cursor: datetime,
    chunks: int,
    per_runway: dict[str, int],
    per_aircraft: dict[str, dict[str, int]],
    last_new: dict[str, datetime],
    given_up: set[str],
) -> None:
    state = {
        "version": CHECKPOINT_VERSION,
        "airport": airport.code,
        "start_utc": stop.isoformat(),
        "cursor_utc": cursor.isoformat(),
        "chunk_hours": plan.chunk_hours,
        "radius_km": plan.radius_km,
        "chunks_fetched": chunks,
        "per_runway": per_runway,
        "per_aircraft": per_aircraft,
        "last_new": {ident: value.isoformat() for ident, value in last_new.items()},
        "given_up": sorted(given_up),
    }
    paths.checkpoint.mkdir(parents=True, exist_ok=True)
    temporary = paths.checkpoint / "state.json.tmp"
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    temporary.replace(paths.checkpoint_state)


def _integer_counts(
    value: dict[str, Any], defaults: dict[str, int] | None = None
) -> dict[str, int]:
    counts = dict(defaults or {})
    for ident, count in value.items():
        counts[str(ident)] = int(count)
    return counts


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_utc())


def _utc():
    from datetime import timezone

    return timezone.utc
