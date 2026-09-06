"""Surface wind for the observed baseline's speed gate (read-time join, stdlib only).

The observed baseline's crossing speed is a GROUND speed; the gate's window is an
airspeed window. An ordinary 10 kt headwind is half of it, so without the wind an
observed speed fail can be the day's weather. OpenSky carries no airspeed and no true
heading, so the wind comes from the field's own ASOS/METAR reports
(``trajectory_data_process/metar/fetch_iem_asos.py`` fetches them; this module reads
the archive CSV) and is joined to each flight by its landing time:

    headwind = W · cos(direction_from − runway course)      # both degrees TRUE
    airspeed estimate = crossing ground speed + headwind

The estimate is declared with a fixed uncertainty (``AIRSPEED_ESTIMATE_UNCERTAINTY_MS``,
5 kt: the tower's 10 m wind is not the wind at the threshold, reports are hourly and
gusty). A report older than ``WIND_MAX_AGE_S`` or a variable-direction wind yields NO
estimate, and the row says so and falls back to the ground-speed proxy — never
silently.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geokit import kt_to_ms

WIND_SOURCE = "iem_asos_metar"
# The nearest report may be this far from the landing time (routine reports are hourly
# at :51-:56; specials fill in when the weather moves).
WIND_MAX_AGE_S = 1800.0
# Declared, not fitted: the tower wind vs the threshold wind, hourly sampling, gusts.
AIRSPEED_ESTIMATE_UNCERTAINTY_MS = kt_to_ms(5.0)
_IEM_HEADER = ("station", "valid", "drct", "sknt", "gust")
_IEM_MISSING = "M"


@dataclass(frozen=True)
class WindObservation:
    valid_utc: datetime
    direction_deg: float | None   # degrees TRUE the wind blows FROM; None = variable
    speed_ms: float
    gust_ms: float | None

    @property
    def calm(self) -> bool:
        return self.speed_ms == 0.0

    def headwind_ms(self, runway_course_deg: float) -> float | None:
        """Component along the approach course (positive = headwind); None when the
        direction is variable and the speed is not zero."""
        if self.calm:
            return 0.0
        if self.direction_deg is None:
            return None
        return self.speed_ms * math.cos(math.radians(self.direction_deg - runway_course_deg))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at_utc": self.valid_utc.isoformat(timespec="minutes"),
            "direction_deg_true": self.direction_deg,
            "speed_ms": self.speed_ms,
            "gust_ms": self.gust_ms,
        }


class WindTable:
    """One station's observations, time-sorted, with nearest-report lookup."""

    def __init__(self, station: str, observations: list[WindObservation]) -> None:
        if not observations:
            raise ValueError(f"{station}: a wind table needs at least one observation")
        self.station = station
        self.observations = sorted(observations, key=lambda o: o.valid_utc)
        self._times = [o.valid_utc for o in self.observations]

    @classmethod
    def from_iem_csv(cls, paths: list[Path]) -> "WindTable":
        """Read one or more IEM ASOS CSV files (``fetch_iem_asos``) for ONE station."""
        station: str | None = None
        observations: list[WindObservation] = []
        for path in paths:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = tuple(next(reader))
                if header != _IEM_HEADER:
                    raise ValueError(f"{path}: expected columns {_IEM_HEADER}, got {header}")
                for row in reader:
                    if len(row) != len(_IEM_HEADER):
                        raise ValueError(f"{path}: malformed row {row!r}")
                    name, valid, drct, sknt, gust = row
                    if station is None:
                        station = name
                    elif name != station:
                        raise ValueError(f"{path}: mixes stations {station} and {name}")
                    if sknt == _IEM_MISSING:
                        continue  # a report with no wind speed says nothing usable
                    observations.append(WindObservation(
                        valid_utc=datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(
                            tzinfo=timezone.utc
                        ),
                        direction_deg=None if drct == _IEM_MISSING else float(drct),
                        speed_ms=kt_to_ms(float(sknt)),
                        gust_ms=None if gust == _IEM_MISSING else kt_to_ms(float(gust)),
                    ))
        if station is None:
            raise ValueError(f"{[str(p) for p in paths]}: no observations")
        return cls(station, observations)

    def nearest(self, when: datetime, *, max_age_s: float = WIND_MAX_AGE_S) -> WindObservation | None:
        """The report closest to ``when`` if within ``max_age_s``, else None."""
        index = bisect.bisect_left(self._times, when)
        candidates = [
            self.observations[i] for i in (index - 1, index) if 0 <= i < len(self.observations)
        ]
        best = min(candidates, key=lambda o: abs((o.valid_utc - when).total_seconds()))
        return best if abs((best.valid_utc - when).total_seconds()) <= max_age_s else None


def parse_landing_time(value: str) -> datetime:
    """The record's ``landing_time_utc`` (ISO 8601, ``Z`` or offset) as an aware datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"landing_time_utc {value!r} carries no timezone")
    return parsed.astimezone(timezone.utc)


def wind_at_landing(
    table: WindTable,
    landing_time_utc: str,
    runway_course_deg: float,
) -> tuple[float | None, dict[str, Any]]:
    """``(headwind_ms or None, wind block for the row)`` for one flight.

    The block always says what happened: the report used and its age, or why no
    estimate exists (no report within ``WIND_MAX_AGE_S``, or a variable wind).
    """
    when = parse_landing_time(landing_time_utc)
    observation = table.nearest(when)
    if observation is None:
        return None, {
            "source": WIND_SOURCE, "station": table.station, "status": "unavailable",
            "reason": f"no report within {WIND_MAX_AGE_S:g} s of the landing time",
        }
    headwind = observation.headwind_ms(runway_course_deg)
    block = {
        "source": WIND_SOURCE,
        "station": table.station,
        "age_s": abs((observation.valid_utc - when).total_seconds()),
        **observation.to_dict(),
    }
    if headwind is None:
        return None, {**block, "status": "unavailable",
                      "reason": "variable wind direction; no headwind component"}
    return headwind, {**block, "status": "estimated", "headwind_ms": headwind}


def load_wind_tables(root: Path) -> dict[str, WindTable]:
    """``{ICAO: WindTable}`` from ``<root>/<ICAO>/*.csv``; an absent root is no tables."""
    tables: dict[str, WindTable] = {}
    if not root.is_dir():
        return tables
    for airport_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(airport_dir.glob("*.csv"))
        if files:
            tables[airport_dir.name.upper()] = WindTable.from_iem_csv(files)
    return tables
