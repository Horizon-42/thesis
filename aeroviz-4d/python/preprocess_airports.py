"""
preprocess_airports.py
======================
One-time preprocessing script: converts OurAirports CSV data into the
GeoJSON files expected by the frontend.

Outputs (written to ../aeroviz-4d/public/data/):
  runway.geojson   — runway polygons, one Feature per runway
  waypoints.geojson — NOT produced here; comes from ARINC 424 / CIFP parsing

Data source:
  Download from https://ourairports.com/data/
  Files needed: airports.csv, runways.csv

📖 Tutorial: see docs/01-data-pipeline.md

How to run:
  cd python
  python preprocess_airports.py --airport CYLW
"""

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

METRES_PER_FOOT = 0.3048
METRES_PER_DEG_LAT = 111_320.0  # approximately constant globally

OUTPUT_DIR = Path(__file__).parent.parent / "public" / "data"


# ── Data types ────────────────────────────────────────────────────────────────

class RunwayEnds(NamedTuple):
    """Parsed row from OurAirports runways.csv"""
    le_ident: str
    he_ident: str
    le_lon: float
    le_lat: float
    he_lon: float
    he_lat: float
    le_elevation_ft: float
    he_elevation_ft: float
    length_ft: float
    width_ft: float
    surface: str
    lighted: int


# ── Geometry helpers ──────────────────────────────────────────────────────────

def metres_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at the given latitude."""
    return METRES_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def runway_bearing_rad(
    le_lon: float, le_lat: float,
    he_lon: float, he_lat: float,
) -> float:
    """
    Compute the bearing from the lower-end threshold to the higher-end threshold.

    Uses the same flat-Earth approximation as the TypeScript ocsGeometry.ts.
    Returns bearing in radians, range (−π, π].
    """
    # TODO ① — Implement this function.
    #
    # Formula:
    #   dx = (he_lon - le_lon) * metres_per_deg_lon(le_lat)
    #   dy = (he_lat - le_lat) * METRES_PER_DEG_LAT
    #   return math.atan2(dx, dy)
    #
    # This is identical to the TypeScript bearingRad() — implement both
    # together so you understand the formula in one place.

    raise NotImplementedError("TODO: implement runway_bearing_rad")


def offset_point_deg(
    lon: float, lat: float,
    bearing_rad: float,
    distance_m: float,
) -> tuple[float, float]:
    """
    Offset a (lon, lat) point by `distance_m` metres in direction `bearing_rad`.

    Returns (new_lon, new_lat).  Altitude is not changed here — caller handles it.

    Formula (flat-Earth, valid for distances < 20 km):
      new_lon = lon + (distance_m × sin(bearing)) / metres_per_deg_lon(lat)
      new_lat = lat + (distance_m × cos(bearing)) / METRES_PER_DEG_LAT
    """
    # TODO ② — Implement this function.
    #
    # Hint: this is the Python equivalent of offsetPoint() in ocsGeometry.ts.
    # Implement them together — the math is identical.

    raise NotImplementedError("TODO: implement offset_point_deg")


def runway_to_polygon(runway: RunwayEnds) -> list[list[float]]:
    """
    Convert a runway (defined by two centreline endpoints) into a 4-corner
    polygon accounting for runway width.

    Returns a closed GeoJSON polygon ring:
      [[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4], [lon1,lat1]]

    Strategy:
      1. Compute the centreline bearing.
      2. Compute the perpendicular bearing (bearing ± π/2).
      3. Offset each centreline end by half the width to get the four corners:
           LE_left, LE_right, HE_right, HE_left

    Visual diagram (looking down):
        LE_left  ───────────────  HE_left
            |   centreline →         |
        LE_right ───────────────  HE_right
    """
    # TODO ③ — Implement this function using runway_bearing_rad() and offset_point_deg().
    #
    # Steps:
    #   bearing    = runway_bearing_rad(le_lon, le_lat, he_lon, he_lat)
    #   perp_left  = bearing - math.pi / 2
    #   perp_right = bearing + math.pi / 2
    #   half_w_m   = (runway.width_ft * METRES_PER_FOOT) / 2
    #
    #   le_left  = offset_point_deg(le_lon, le_lat, perp_left,  half_w_m)
    #   le_right = offset_point_deg(le_lon, le_lat, perp_right, half_w_m)
    #   he_right = offset_point_deg(he_lon, he_lat, perp_right, half_w_m)
    #   he_left  = offset_point_deg(he_lon, he_lat, perp_left,  half_w_m)
    #
    # Build the GeoJSON ring (lon, lat per point, closed = first == last):
    #   return [le_left, le_right, he_right, he_left, le_left]
    # where each element is a list [lon, lat].
    #
    # ⚠ GeoJSON uses [longitude, latitude] order — NOT [lat, lon]!

    raise NotImplementedError("TODO: implement runway_to_polygon")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_runways(csv_path: Path, airport_ident: str) -> list[RunwayEnds]:
    """Load and parse runway rows for a single airport from OurAirports CSV."""
    df = pd.read_csv(csv_path)
    # Filter to the target airport; drop rows with missing coordinate data
    subset = df[df["airport_ident"] == airport_ident].dropna(
        subset=["le_longitude_deg", "le_latitude_deg",
                "he_longitude_deg", "he_latitude_deg"]
    )

    runways = []
    for _, row in subset.iterrows():
        runways.append(RunwayEnds(
            le_ident=str(row.get("le_ident", "")),
            he_ident=str(row.get("he_ident", "")),
            le_lon=float(row["le_longitude_deg"]),
            le_lat=float(row["le_latitude_deg"]),
            he_lon=float(row["he_longitude_deg"]),
            he_lat=float(row["he_latitude_deg"]),
            le_elevation_ft=float(row.get("le_elevation_ft", 0) or 0),
            he_elevation_ft=float(row.get("he_elevation_ft", 0) or 0),
            length_ft=float(row.get("length_ft", 0) or 0),
            width_ft=float(row.get("width_ft", 150) or 150),
            surface=str(row.get("surface", "ASP") or "ASP"),
            lighted=int(row.get("lighted", 0) or 0),
        ))
    return runways


def build_runway_geojson(runways: list[RunwayEnds], airport_ident: str) -> dict:
    """Convert parsed runway data to a GeoJSON FeatureCollection."""
    features = []
    for rwy in runways:
        coords = runway_to_polygon(rwy)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "airport_ident": airport_ident,
                "le_ident": rwy.le_ident,
                "he_ident": rwy.he_ident,
                "length_ft": rwy.length_ft,
                "width_ft": rwy.width_ft,
                "surface": rwy.surface,
                "lighted": rwy.lighted,
                "le_elevation_ft": rwy.le_elevation_ft,
                "he_elevation_ft": rwy.he_elevation_ft,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess OurAirports CSV into GeoJSON for AeroViz-4D"
    )
    parser.add_argument(
        "--airport", default="CYLW",
        help="ICAO airport code to process (default: CYLW)"
    )
    parser.add_argument(
        "--runways-csv", default="runways.csv",
        help="Path to OurAirports runways.csv (default: ./runways.csv)"
    )
    args = parser.parse_args()

    runways_path = Path(args.runways_csv)
    if not runways_path.exists():
        print(f"ERROR: {runways_path} not found.")
        print("Download from: https://ourairports.com/data/runways.csv")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing runways for {args.airport}...")
    runways = load_runways(runways_path, args.airport)
    print(f"  Found {len(runways)} runway(s)")

    geojson = build_runway_geojson(runways, args.airport)
    out_path = OUTPUT_DIR / "runway.geojson"
    out_path.write_text(json.dumps(geojson, indent=2))
    print(f"  ✓ Written: {out_path}")


if __name__ == "__main__":
    main()
