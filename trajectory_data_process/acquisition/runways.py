"""Resolve runway-threshold coordinates from the OurAirports runways dataset.

Used to select trajectories by the exact runway threshold an aircraft arrives at,
rather than by arrival airport alone.

This module is the single source for turning a runways.csv row into landing thresholds
(displaced where the source says so). ``build_runway_config.py`` uses it to generate
``runway_thresholds.json``; harvest then loads that canonical config. The old second
downloader path that resolved thresholds independently has been removed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geokit import FT_M as FT_TO_M
from geokit import haversine_m


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


def _pavement_end(row: dict[str, str], side: str) -> dict[str, Any] | None:
    """One pavement end of a runway row, as OurAirports records it (None without coords)."""
    lat = row.get(f"{side}_latitude_deg")
    lon = row.get(f"{side}_longitude_deg")
    if not lat or not lon:
        return None
    elev_ft = row.get(f"{side}_elevation_ft", "")
    heading = row.get(f"{side}_heading_degT", "")
    return {
        "ident": row[f"{side}_ident"].upper(),
        "lat": float(lat),
        "lon": float(lon),
        "elevation_m": float(elev_ft) * FT_TO_M if elev_ft else None,
        "heading_deg": float(heading) if heading else None,
        "displaced_m": float(row.get(f"{side}_displaced_threshold_ft") or 0.0) * FT_TO_M,
    }


def _landing_threshold(near: dict[str, Any], far: dict[str, Any], length_m: float) -> dict[str, Any]:
    """The LANDING threshold for one runway end: the pavement end moved down the centreline.

    Where a runway has a displaced threshold, the landing surface begins that far past the
    pavement end and the pavement before it is unavailable for touchdown. Aircraft fly the
    glidepath to the DISPLACED point, so that -- not the pavement end -- is the approach
    target, the along-track origin, and the reference for the threshold-crossing height.

    KSJC 30L/30R are displaced 775 m; taking the pavement end instead put the target 775 m
    short, which on a 3 deg glidepath is a 40.6 m altitude error at the threshold.

    A runway is straight, so the displaced point is an exact linear interpolation between the
    two ends -- elevation included, which is what carries the runway's slope.
    """
    if near["displaced_m"] and not length_m:
        raise ValueError(
            f"runway end {near['ident']} declares a {near['displaced_m']:.0f} m displaced "
            "threshold but the row gives no usable runway length to place it along -- "
            "refusing to silently publish the undisplaced pavement end"
        )
    fraction = near["displaced_m"] / length_m if length_m else 0.0
    elevation, source = near["elevation_m"], "runway_end"
    if elevation is not None and far["elevation_m"] is not None:
        elevation += fraction * (far["elevation_m"] - elevation)
    elif elevation is None and far["elevation_m"] is not None:
        # OurAirports publishes an elevation for only one end of this runway (KMSY 02/20
        # is the fleet's one case). The opposite end is the closest available reference --
        # it differs by the runway's slope, metres at most over a runway's length, and is
        # far better than the airport field elevation, let alone the silent 0.0 the
        # previous code substituted. The substitution is RECORDED, not assumed: a
        # threshold elevation is the vertical origin for every gate measured against it.
        elevation, source = far["elevation_m"], "opposite_end"
    return {
        "ident": near["ident"],
        "lat": round(near["lat"] + fraction * (far["lat"] - near["lat"]), 7),
        "lon": round(near["lon"] + fraction * (far["lon"] - near["lon"]), 7),
        "elevation_m": round(elevation, 2) if elevation is not None else None,
        "elevation_source": source if elevation is not None else "missing",
        "heading_deg": near["heading_deg"],
        "displaced_threshold_m": round(near["displaced_m"], 1),
    }


def landing_thresholds_from_row(row: dict[str, str]) -> list[dict[str, Any]]:
    """The landing thresholds for every end of one runways.csv row that has coordinates."""
    le, he = _pavement_end(row, "le"), _pavement_end(row, "he")
    thresholds = []
    for near, far in ((le, he), (he, le)):
        if near is None:
            continue
        # An end without a recorded partner can still be a threshold as long as it is not
        # displaced (fraction 0 needs no interpolation target); _landing_threshold raises
        # on the displaced-but-lengthless combination rather than guessing.
        far_or_self = far if far is not None else near
        # The published ``length_ft`` and the distance between the two ends disagree by
        # ~0.1 %; the coordinates are what the displacement is interpolated against, so
        # measure them.
        length_m = haversine_m(near["lat"], near["lon"], far_or_self["lat"], far_or_self["lon"])
        thresholds.append(_landing_threshold(near, far_or_self, length_m))
    return thresholds


def resolve_runway_threshold(airport: str, runway_ident: str, csv_path: Path) -> RunwayThreshold:
    """Find the LANDING threshold for ``runway_ident`` (e.g. "05L"/"23R") at an airport.

    Each runways.csv row describes both ends with ``le_*`` and ``he_*`` fields; either end
    may match the requested ident. The returned point is the displaced landing threshold
    where one is published — the same point the runway_thresholds.json generator emits.
    """
    code = airport.upper()
    wanted = runway_ident.upper().lstrip("0") or "0"
    if not csv_path.exists():
        raise RuntimeError(f"runways.csv not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("airport_ident") or "").upper() != code:
                continue
            for t in landing_thresholds_from_row(row):
                if t["ident"].lstrip("0") != wanted:
                    continue
                return RunwayThreshold(
                    airport=code,
                    ident=t["ident"],
                    lat=t["lat"],
                    lon=t["lon"],
                    elevation_m=t["elevation_m"] if t["elevation_m"] is not None else 0.0,
                    heading_deg=t["heading_deg"],
                )

    raise RuntimeError(f"Runway {runway_ident} not found for airport {code} in {csv_path}")
