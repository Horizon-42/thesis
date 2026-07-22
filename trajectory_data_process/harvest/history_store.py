"""Disk-backed state-vector accumulation for memory-bounded harvests.

OpenSky already caches each query as parquet, but the harvest needs a merged view while
it scans backward. Keeping that view as millions of Python dictionaries was the largest
memory consumer. This store keeps only reconstruction-relevant columns in SQLite and
lets the runner reprocess one affected aircraft at a time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

_COLUMNS = (
    "time",
    "icao24",
    "lat",
    "lon",
    "callsign",
    "onground",
    "geoaltitude",
)


class DiskHistoryStore:
    """Accumulate history frames on disk, indexed by aircraft and sample time."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        # Keep SQLite's own cache bounded as well; sorting temporary data must use disk.
        self.connection.execute("PRAGMA cache_size = -16384")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                time REAL NOT NULL,
                icao24 TEXT NOT NULL,
                lat REAL,
                lon REAL,
                callsign TEXT,
                onground INTEGER,
                geoaltitude REAL,
                PRIMARY KEY (icao24, time)
            ) WITHOUT ROWID
            """
        )

    def __enter__(self) -> DiskHistoryStore:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def add_frame(self, frame: Any) -> set[str]:
        """Persist one fetched frame and return the aircraft it can change."""
        if frame is None or getattr(frame, "empty", False):
            return set()

        positions = {
            column: frame.columns.get_loc(column) if column in frame.columns else None
            for column in _COLUMNS
        }
        affected: set[str] = set()

        def values() -> Iterator[tuple[Any, ...]]:
            for source in frame.itertuples(index=False, name=None):
                icao24 = _text(_at(source, positions["icao24"]))
                time_s = _time_s(_at(source, positions["time"]))
                if not icao24 or time_s is None:
                    continue
                affected.add(icao24)
                yield (
                    time_s,
                    icao24,
                    _scalar(_at(source, positions["lat"])),
                    _scalar(_at(source, positions["lon"])),
                    _text(_at(source, positions["callsign"]), strip_only=True),
                    _boolean(_at(source, positions["onground"])),
                    _scalar(_at(source, positions["geoaltitude"])),
                )

        # Adjacent history requests both include their shared timestamp. Replacing on
        # (icao24, time) removes that duplicate and bounds the database to unique samples.
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO samples
                (time, icao24, lat, lon, callsign, onground, geoaltitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values(),
        )
        self.connection.commit()
        return affected

    def rows_for(self, icao24: str) -> list[dict[str, Any]]:
        """Load every accumulated row for one aircraft, in chronological order."""
        cursor = self.connection.execute(
            """
            SELECT time, icao24, lat, lon, callsign, onground, geoaltitude
            FROM samples
            WHERE icao24 = ?
            ORDER BY time
            """,
            (icao24,),
        )
        return [dict(zip(_COLUMNS, row)) for row in cursor]

    def aircraft(self) -> Iterator[str]:
        """Yield stored aircraft identifiers without loading their samples."""
        cursor = self.connection.execute(
            "SELECT DISTINCT icao24 FROM samples ORDER BY icao24"
        )
        for (icao24,) in cursor:
            yield str(icao24)


def _at(row: tuple[Any, ...], position: int | None) -> Any:
    return None if position is None else row[position]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def _scalar(value: Any) -> Any:
    if _missing(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    return value


def _text(value: Any, *, strip_only: bool = False) -> str | None:
    value = _scalar(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if strip_only else text.lower()


def _boolean(value: Any) -> int | None:
    value = _scalar(value)
    return None if value is None else int(bool(value))


def _time_s(value: Any) -> float | None:
    value = _scalar(value)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return float(timestamp.timestamp())
