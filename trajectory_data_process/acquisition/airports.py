"""Resolve airport reference points from the AeroViz OurAirports dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


FT_TO_M = 0.3048


@dataclass(frozen=True)
class AirportProfile:
    """Airport center and field elevation."""

    code: str
    lat: float
    lon: float
    elevation_m: float


def airports_csv_path(aeroviz_root: Path) -> Path:
    """Return the airports.csv path, preferring the common/ subfolder."""
    common = aeroviz_root / "public" / "data" / "common" / "airports.csv"
    return common if common.exists() else aeroviz_root / "public" / "data" / "airports.csv"


def resolve_airport_profile(airport: str, aeroviz_root: Path) -> AirportProfile:
    """Look up an airport center and elevation by ICAO/GPS/ident code."""
    code = airport.upper()
    csv_path = airports_csv_path(aeroviz_root)
    if not csv_path.exists():
        raise RuntimeError(f"airports.csv not found under {csv_path.parent}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            idents = {
                (row.get("ident") or "").upper(),
                (row.get("gps_code") or "").upper(),
                (row.get("icao_code") or "").upper(),
            }
            if code not in idents:
                continue
            lat, lon = row.get("latitude_deg"), row.get("longitude_deg")
            if not lat or not lon:
                continue
            elev_ft = row.get("elevation_ft")
            elev_m = float(elev_ft) * FT_TO_M if elev_ft not in (None, "") else 0.0
            return AirportProfile(code=code, lat=float(lat), lon=float(lon), elevation_m=elev_m)

    raise RuntimeError(f"Airport {code} not found in {csv_path}")
