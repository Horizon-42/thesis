"""The arrivals roster as the interactive multi-aircraft mode reads it (design §10.3, IM5): rows only, never a track.

The backend lists an airport's arrivals of one UTC day from it and checks a job request against it; the job
(``traffic_job.py``) picks from it the arrivals whose tracks it must load. Import-light on purpose (the backend
imports it): the standard library and the harvest's time parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from trajectory_data_process.harvest.utc import parse_iso_utc_s

#: A row's slice end (entry + ``arrival_duration_s``) and the loaded track's last sample are the same instant to
#: float rounding. The exact ``Traffic.airborne`` filter follows the load; this keeps the load a superset of it.
SLICE_SLACK_S = 1.0


@dataclass(frozen=True)
class RosterRow:
    """One arrival of ``arrivals/manifest.json``, as far as selecting it needs."""

    flight_key: str
    callsign: str | None          # None: the roster has none for this flight
    icao24: str
    runway: str
    entry_utc_s: float
    landing_utc_s: float
    duration_s: float

    @property
    def end_utc_s(self) -> float:
        """The last sample of its arrival slice."""
        return self.entry_utc_s + self.duration_s


def read_roster(manifest_path: str | Path) -> list[RosterRow]:
    """Every row of an ``arrivals/manifest.json``, in roster order. Reads the manifest only."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [
        RosterRow(
            flight_key=row["flight_key"], callsign=row["callsign"], icao24=row["icao24"], runway=row["runway"],
            entry_utc_s=parse_iso_utc_s(row["entry_time_utc"]), landing_utc_s=parse_iso_utc_s(row["landing_time_utc"]),
            duration_s=float(row["arrival_duration_s"]))
        for row in manifest["records"]
    ]


def utc_day_bounds_s(day: date) -> tuple[float, float]:
    """``[start, end)`` of a UTC day in epoch seconds."""
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def landing_in(rows: list[RosterRow], start_utc_s: float, end_utc_s: float) -> list[RosterRow]:
    """The rows whose recorded landing is in ``[start, end)``, by landing time then flight key."""
    return sorted((r for r in rows if start_utc_s <= r.landing_utc_s < end_utc_s),
                  key=lambda r: (r.landing_utc_s, r.flight_key))


def airborne_in(rows: list[RosterRow], start_utc_s: float, end_utc_s: float) -> list[RosterRow]:
    """The rows in the air at some time in ``[start, end]`` (``SLICE_SLACK_S`` wider on both sides)."""
    return [r for r in rows
            if r.entry_utc_s <= end_utc_s + SLICE_SLACK_S and r.end_utc_s >= start_utc_s - SLICE_SLACK_S]


def m1_near(rows: list[RosterRow], own: RosterRow, horizon_s: float) -> list[RosterRow]:
    """The arrivals whose tracks M1 reads for ``own``: those that can share its window, which is
    ``[own entry, own entry + horizon]`` (``traffic_optimization.window_scenario``), and ``own`` itself."""
    return airborne_in(rows, own.entry_utc_s, own.entry_utc_s + horizon_s)


def m2_near(rows: list[RosterRow], block: list[RosterRow], horizon_s: float) -> list[RosterRow]:
    """The arrivals whose tracks M2 reads for ``block``: those that can share the block's traffic window
    ``[first entry, last entry + horizon]`` (``traffic_optimization.block_scenarios``) — which the block's own
    aircraft are in. ``block`` is every row that lands in it, a superset of the aircraft that have dynamics."""
    entries = [r.entry_utc_s for r in block]
    return airborne_in(rows, min(entries), max(entries) + horizon_s)
