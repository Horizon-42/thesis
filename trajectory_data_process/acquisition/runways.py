"""Resolve runway-threshold coordinates from the OurAirports runways dataset.

Used to select trajectories by the exact runway threshold an aircraft arrives at,
rather than by arrival airport alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from geokit import FT_M as FT_TO_M


@dataclass(frozen=True)
class RunwayThreshold:
    """One end of a runway: the landing threshold and its alignment."""

    airport: str
    ident: str  # e.g. "23R"
    lat: float
    lon: float
    elevation_m: float
    heading_deg: float | None


def runways_csv_path(aeroviz_root: Path) -> Path:
    """Return the common runways.csv path."""
    return aeroviz_root / "public" / "data" / "common" / "runways.csv"


def resolve_runway_threshold(airport: str, runway_ident: str, csv_path: Path) -> RunwayThreshold:
    """Find the threshold for ``runway_ident`` (e.g. "05L"/"23R") at an airport.

    Each runways.csv row describes both ends with ``le_*`` and ``he_*`` fields;
    either end may match the requested ident.
    """
    code = airport.upper()
    wanted = runway_ident.upper().lstrip("0") or "0"
    if not csv_path.exists():
        raise RuntimeError(f"runways.csv not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("airport_ident") or "").upper() != code:
                continue
            for prefix in ("le", "he"):
                ident = (row.get(f"{prefix}_ident") or "").upper()
                if ident.lstrip("0") != wanted:
                    continue
                threshold = _threshold_from_row(code, ident, row, prefix)
                if threshold is not None:
                    return threshold

    raise RuntimeError(f"Runway {runway_ident} not found for airport {code} in {csv_path}")


def _threshold_from_row(
    airport: str, ident: str, row: dict[str, str], prefix: str
) -> RunwayThreshold | None:
    lat = row.get(f"{prefix}_latitude_deg")
    lon = row.get(f"{prefix}_longitude_deg")
    if not lat or not lon:
        return None
    elev_ft = row.get(f"{prefix}_elevation_ft")
    heading = row.get(f"{prefix}_heading_degT")
    return RunwayThreshold(
        airport=airport,
        ident=ident,
        lat=float(lat),
        lon=float(lon),
        elevation_m=float(elev_ft) * FT_TO_M if elev_ft not in (None, "") else 0.0,
        heading_deg=float(heading) if heading not in (None, "") else None,
    )
