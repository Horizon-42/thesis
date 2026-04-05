"""
preprocess_airports.py
======================
One-time preprocessing script: converts OurAirports CSV data into the
GeoJSON files expected by the frontend.

Outputs (written to ../aeroviz-4d/public/data/):
    runway.geojson   — runway polygons (runway surface + landing zone)
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
    le_heading_degT: float | None = None
    he_heading_degT: float | None = None
    le_displaced_threshold_ft: float = 0.0
    he_displaced_threshold_ft: float = 0.0


# ── Geometry helpers ──────────────────────────────────────────────────────────

def metres_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at the given latitude."""
    return METRES_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def lonlat_to_local_m(
    lon: float,
    lat: float,
    ref_lon: float,
    ref_lat: float,
) -> tuple[float, float]:
    """Convert lon/lat to local East/North metres in a flat tangent plane."""
    east_m = (lon - ref_lon) * metres_per_deg_lon(ref_lat)
    north_m = (lat - ref_lat) * METRES_PER_DEG_LAT
    return east_m, north_m


def local_m_to_lonlat(
    east_m: float,
    north_m: float,
    ref_lon: float,
    ref_lat: float,
) -> tuple[float, float]:
    """Convert local East/North metres back to lon/lat."""
    lon = ref_lon + east_m / metres_per_deg_lon(ref_lat)
    lat = ref_lat + north_m / METRES_PER_DEG_LAT
    return lon, lat


def runway_bearing_rad(
    le_lon: float, le_lat: float,
    he_lon: float, he_lat: float,
) -> float:
    """
    Compute the bearing from the lower-end threshold to the higher-end threshold.

    Uses the same flat-Earth approximation as the TypeScript ocsGeometry.ts.
    Returns bearing in radians, range (−π, π].
    """
    # ① — Implement this function.
    #
    # Formula:
    #   dx = (he_lon - le_lon) * metres_per_deg_lon(le_lat)
    #   dy = (he_lat - le_lat) * METRES_PER_DEG_LAT
    #   return math.atan2(dx, dy)
    #
    # This is identical to the TypeScript bearingRad() — implement both
    # together so you understand the formula in one place.

    dx = (he_lon - le_lon) * metres_per_deg_lon(le_lat)
    dy = (he_lat - le_lat) * METRES_PER_DEG_LAT

    return math.atan2(dx, dy)


def runway_bearing_from_metadata(runway: RunwayEnds) -> float:
    """Prefer declared runway heading; fallback to coordinate-derived bearing."""
    if runway.le_heading_degT is not None:
        return math.radians(runway.le_heading_degT % 360)
    return runway_bearing_rad(runway.le_lon, runway.le_lat, runway.he_lon, runway.he_lat)


def flat_distance_m(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    """Approximate ground distance for short segments (< 20 km)."""
    mid_lat = (lat_a + lat_b) / 2
    dx = (lon_b - lon_a) * metres_per_deg_lon(mid_lat)
    dy = (lat_b - lat_a) * METRES_PER_DEG_LAT
    return math.hypot(dx, dy)


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
    # ② — Implement this function.
    #
    # Hint: this is the Python equivalent of offsetPoint() in ocsGeometry.ts.
    # Implement them together — the math is identical.
    new_lon = lon + (distance_m * math.sin(bearing_rad)) / metres_per_deg_lon(lat)
    new_lat = lat + (distance_m * math.cos(bearing_rad)) / METRES_PER_DEG_LAT
    return new_lon, new_lat

def runway_to_polygon(runway: RunwayEnds, lateral_offset_m: float = 0.0) -> list[list[float]]:
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
    # ③ — Implement this function using runway_bearing_rad() and offset_point_deg().
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

    return build_runway_ring(runway, include_displaced=True, lateral_offset_m=lateral_offset_m)


def landing_zone_polygon(runway: RunwayEnds, lateral_offset_m: float = 0.0) -> list[list[float]]:
    """Polygon for the touchdown-allowed region between displaced thresholds."""
    return build_runway_ring(runway, include_displaced=False, lateral_offset_m=lateral_offset_m)


def build_runway_ring(
    runway: RunwayEnds,
    include_displaced: bool,
    lateral_offset_m: float = 0.0,
) -> list[list[float]]:
    """Build a runway-width polygon ring using threshold centres or physical ends."""
    bearing = runway_bearing_from_metadata(runway)

    le_disp_m = runway.le_displaced_threshold_ft * METRES_PER_FOOT
    he_disp_m = runway.he_displaced_threshold_ft * METRES_PER_FOOT

    # Use declared dimensions when available to avoid endpoint coordinate noise.
    raw_threshold_len_m = flat_distance_m(runway.le_lon, runway.le_lat, runway.he_lon, runway.he_lat)
    declared_len_m = runway.length_ft * METRES_PER_FOOT if runway.length_ft > 0 else 0.0
    declared_landing_len_m = declared_len_m - le_disp_m - he_disp_m
    landing_len_m = declared_landing_len_m if declared_landing_len_m > 0 else raw_threshold_len_m

    ref_lon = (runway.le_lon + runway.he_lon) / 2
    ref_lat = (runway.le_lat + runway.he_lat) / 2

    # Local unit vectors: forward along runway heading, right perpendicular.
    fwd_e = math.sin(bearing)
    fwd_n = math.cos(bearing)
    right_e = fwd_n
    right_n = -fwd_e

    half_landing_m = landing_len_m / 2
    le_center_e = -fwd_e * half_landing_m
    le_center_n = -fwd_n * half_landing_m
    he_center_e = fwd_e * half_landing_m
    he_center_n = fwd_n * half_landing_m

    # Expand from thresholds to pavement ends only for runway_surface.
    if include_displaced:
        le_center_e -= fwd_e * le_disp_m
        le_center_n -= fwd_n * le_disp_m
        he_center_e += fwd_e * he_disp_m
        he_center_n += fwd_n * he_disp_m

    # Optional fine-tuning for imagery alignment.
    # Positive value shifts polygon to the right side of runway bearing.
    if lateral_offset_m != 0:
        le_center_e += right_e * lateral_offset_m
        le_center_n += right_n * lateral_offset_m
        he_center_e += right_e * lateral_offset_m
        he_center_n += right_n * lateral_offset_m

    half_width_m = (runway.width_ft * METRES_PER_FOOT) / 2
    left_e = -right_e
    left_n = -right_n

    le_left = local_m_to_lonlat(
        le_center_e + left_e * half_width_m,
        le_center_n + left_n * half_width_m,
        ref_lon,
        ref_lat,
    )
    le_right = local_m_to_lonlat(
        le_center_e + right_e * half_width_m,
        le_center_n + right_n * half_width_m,
        ref_lon,
        ref_lat,
    )
    he_right = local_m_to_lonlat(
        he_center_e + right_e * half_width_m,
        he_center_n + right_n * half_width_m,
        ref_lon,
        ref_lat,
    )
    he_left = local_m_to_lonlat(
        he_center_e + left_e * half_width_m,
        he_center_n + left_n * half_width_m,
        ref_lon,
        ref_lat,
    )

    return [list(le_left), list(le_right), list(he_right), list(he_left), list(le_left)]

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

    def parse_optional_float(value: object) -> float | None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(v):
            return None
        return v

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
            le_heading_degT=parse_optional_float(row.get("le_heading_degT")),
            he_heading_degT=parse_optional_float(row.get("he_heading_degT")),
            le_displaced_threshold_ft=float(row.get("le_displaced_threshold_ft", 0) or 0),
            he_displaced_threshold_ft=float(row.get("he_displaced_threshold_ft", 0) or 0),
        ))
    return runways


def build_runway_geojson(
    runways: list[RunwayEnds],
    airport_ident: str,
    lateral_offset_m: float = 0.0,
) -> dict:
    """Convert parsed runway data to a GeoJSON FeatureCollection."""
    features = []

    def make_feature(rwy: RunwayEnds, coords: list[list[float]], zone_type: str) -> dict:
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "airport_ident": airport_ident,
                "runway_ident": f"{rwy.le_ident}/{rwy.he_ident}",
                "zone_type": zone_type,
                "le_ident": rwy.le_ident,
                "he_ident": rwy.he_ident,
                "length_ft": rwy.length_ft,
                "width_ft": rwy.width_ft,
                "surface": rwy.surface,
                "lighted": rwy.lighted,
                "le_elevation_ft": rwy.le_elevation_ft,
                "he_elevation_ft": rwy.he_elevation_ft,
                "le_displaced_threshold_ft": rwy.le_displaced_threshold_ft,
                "he_displaced_threshold_ft": rwy.he_displaced_threshold_ft,
                "lateral_offset_m": lateral_offset_m,
            },
        }

    for rwy in runways:
        features.append(
            make_feature(
                rwy,
                runway_to_polygon(rwy, lateral_offset_m=lateral_offset_m),
                "runway_surface",
            )
        )
        features.append(
            make_feature(
                rwy,
                landing_zone_polygon(rwy, lateral_offset_m=lateral_offset_m),
                "landing_zone",
            )
        )

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
    parser.add_argument(
        "--lateral-offset-m",
        type=float,
        default=0.0,
        help="Optional sideways shift (metres) to align polygons with imagery; positive shifts to runway right",
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

    geojson = build_runway_geojson(runways, args.airport, lateral_offset_m=args.lateral_offset_m)
    out_path = OUTPUT_DIR / "runway.geojson"
    out_path.write_text(json.dumps(geojson, indent=2))
    print(f"  ✓ Written: {out_path}")


if __name__ == "__main__":
    main()
