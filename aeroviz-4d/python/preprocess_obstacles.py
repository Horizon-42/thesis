"""
preprocess_obstacles.py
=======================
Parses FAA Digital Obstacle File (DOF) fixed-width .Dat files and outputs
a GeoJSON FeatureCollection of obstacles near a reference airport.

Default behavior:
- Reads a single DOF .Dat file (e.g. 37-NC.Dat for North Carolina)
- Filters to verified obstacles within 20 km of the airport center
- Converts DMS coordinates to decimal degrees, feet to metres
- Writes GeoJSON FeatureCollection with Point features to public/data/

Radius selection (--radius-km, default 20 km / ~10.8 NM):
  The default covers the full instrument approach corridor: initial approach
  (IAF), intermediate, final approach (FAF → threshold), and missed approach.
  ICAO PANS-OPS obstacle assessment surfaces rarely extend beyond 10 NM from
  the aerodrome reference point.  20 km keeps the obstacle count manageable
  for real-time CesiumJS rendering (~100–200 entities for a typical airport).

  Approximate counts for KRDU at different radii:
    5 km  →  ~50      (runway vicinity only)
    10 km →  ~80      (final approach segment)
    20 km → ~185      (full approach area — default)
    30 km → ~300      (includes initial approach fixes)
    60 km → ~900      (full TMA — may cause slow rendering)

Usage:
  # Use explicit center (KRDU default)
  python preprocess_obstacles.py --input ../../data/DOF/DOF_260412/37-NC.Dat

  # Auto-read center from airport.json
  python preprocess_obstacles.py --input ../../data/DOF/DOF_260412/37-NC.Dat --airport

  # Wider radius for full TMA analysis
  python preprocess_obstacles.py --input ../../data/DOF/DOF_260412/37-NC.Dat --airport --radius-km 40

  # Canada file for CYLW
  python preprocess_obstacles.py --input ../../data/DOF/DOF_260412/Canada.Dat --center-lon -119.3775 --center-lat 49.9561
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CENTER_LON = -78.7873
DEFAULT_CENTER_LAT = 35.878659
DEFAULT_RADIUS_KM = 20.0
DEFAULT_OUTPUT = Path(__file__).parent.parent / "public" / "data" / "obstacles.geojson"
AIRPORT_JSON = Path(__file__).parent.parent / "public" / "data" / "airport.json"

HEADER_LINES = 4
FEET_TO_METRES = 0.3048

# ── Coordinate math ──────────────────────────────────────────────────────────


def dof_dms_to_decimal(
    deg_str: str, min_str: str, sec_str: str, hemisphere: str
) -> float:
    """Convert DOF DMS strings to signed decimal degrees."""
    deg = float(deg_str)
    minutes = float(min_str)
    seconds = float(sec_str)
    decimal = deg + minutes / 60.0 + seconds / 3600.0
    if hemisphere in ("S", "W"):
        decimal = -decimal
    return decimal


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── DOF parsing ──────────────────────────────────────────────────────────────


def parse_dof_line(line: str) -> dict | None:
    """Parse one fixed-width DOF record into a dict. Returns None if unparseable."""
    if len(line) < 96:
        return None

    try:
        lat_deg = line[35:37].strip()
        lat_min = line[38:40].strip()
        lat_sec = line[41:46].strip()
        lat_hem = line[46].strip()
        lon_deg = line[48:51].strip()
        lon_min = line[52:54].strip()
        lon_sec = line[55:60].strip()
        lon_hem = line[60].strip()

        if not (lat_deg and lat_min and lat_sec and lat_hem):
            return None
        if not (lon_deg and lon_min and lon_sec and lon_hem):
            return None

        lat = dof_dms_to_decimal(lat_deg, lat_min, lat_sec, lat_hem)
        lon = dof_dms_to_decimal(lon_deg, lon_min, lon_sec, lon_hem)
    except (ValueError, IndexError):
        return None

    verification = line[10].strip() if len(line) > 10 else ""
    agl_str = line[83:88].strip() if len(line) > 88 else "0"
    amsl_str = line[89:94].strip() if len(line) > 94 else "0"

    try:
        agl_ft = int(agl_str) if agl_str else 0
        amsl_ft = int(amsl_str) if amsl_str else 0
    except ValueError:
        agl_ft = 0
        amsl_ft = 0

    quantity_char = line[81].strip() if len(line) > 81 else "1"
    try:
        quantity = int(quantity_char) if quantity_char else 1
    except ValueError:
        quantity = 1

    return {
        "oas_number": line[0:9].strip(),
        "verified": verification == "O",
        "country": line[12:14].strip() if len(line) > 14 else "",
        "state": line[15:17].strip() if len(line) > 17 else "",
        "city": line[18:34].strip() if len(line) > 34 else "",
        "lat": lat,
        "lon": lon,
        "obstacle_type": line[62:80].strip() if len(line) > 80 else "",
        "quantity": quantity,
        "agl_ft": agl_ft,
        "amsl_ft": amsl_ft,
        "agl_m": round(agl_ft * FEET_TO_METRES, 2),
        "amsl_m": round(amsl_ft * FEET_TO_METRES, 2),
        "lighting": line[95].strip() if len(line) > 95 else "",
        "horizontal_accuracy": line[97].strip() if len(line) > 97 else "",
        "vertical_accuracy": line[99].strip() if len(line) > 99 else "",
        "marking": line[101].strip() if len(line) > 101 else "",
    }


# ── Airport center resolution ────────────────────────────────────────────────


def load_airport_center(airport_json_path: Path) -> tuple[float, float]:
    """Read center coordinates from airport.json. Returns (lon, lat)."""
    with open(airport_json_path) as f:
        data = json.load(f)
    return data["lon"], data["lat"]


# ── GeoJSON builder ──────────────────────────────────────────────────────────


def build_obstacle_geojson(obstacles: list[dict]) -> dict:
    """Build a GeoJSON FeatureCollection from parsed obstacle records."""
    features = []
    for obs in obstacles:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [obs["lon"], obs["lat"], obs["amsl_m"]],
            },
            "properties": {
                "oas_number": obs["oas_number"],
                "verified": obs["verified"],
                "country": obs["country"],
                "state": obs["state"],
                "city": obs["city"],
                "obstacle_type": obs["obstacle_type"],
                "quantity": obs["quantity"],
                "agl_ft": obs["agl_ft"],
                "amsl_ft": obs["amsl_ft"],
                "agl_m": obs["agl_m"],
                "amsl_m": obs["amsl_m"],
                "lighting": obs["lighting"],
                "horizontal_accuracy": obs["horizontal_accuracy"],
                "vertical_accuracy": obs["vertical_accuracy"],
                "marking": obs["marking"],
                "source": "FAA-DOF",
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse FAA DOF .Dat file and output obstacles.geojson"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to DOF .Dat file (e.g. 37-NC.Dat or Canada.Dat)",
    )
    parser.add_argument(
        "--center-lon",
        type=float,
        default=DEFAULT_CENTER_LON,
        help=f"Reference longitude (default: {DEFAULT_CENTER_LON})",
    )
    parser.add_argument(
        "--center-lat",
        type=float,
        default=DEFAULT_CENTER_LAT,
        help=f"Reference latitude (default: {DEFAULT_CENTER_LAT})",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=f"Filter radius in km (default: {DEFAULT_RADIUS_KM})",
    )
    parser.add_argument(
        "--airport",
        action="store_true",
        help="Auto-read center from public/data/airport.json",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified (U) obstacles (default: verified only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output GeoJSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # Resolve center
    center_lon = args.center_lon
    center_lat = args.center_lat
    if args.airport:
        center_lon, center_lat = load_airport_center(AIRPORT_JSON)
        print(f"Airport center from airport.json: lon={center_lon}, lat={center_lat}")

    # Parse DOF file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    obstacles: list[dict] = []
    skipped_unverified = 0
    skipped_distance = 0
    parse_errors = 0

    with open(input_path, encoding="ascii", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            # Skip 4-line header
            if line_num <= HEADER_LINES:
                continue

            line = line.rstrip("\n\r")
            record = parse_dof_line(line)

            if record is None:
                parse_errors += 1
                continue

            # Verified filter
            if not args.include_unverified and not record["verified"]:
                skipped_unverified += 1
                continue

            # Distance filter
            dist = haversine_km(center_lon, center_lat, record["lon"], record["lat"])
            if dist > args.radius_km:
                skipped_distance += 1
                continue

            record["distance_km"] = round(dist, 2)
            obstacles.append(record)

    # Sort by distance (nearest first)
    obstacles.sort(key=lambda o: o["distance_km"])

    # Build and write GeoJSON
    geojson = build_obstacle_geojson(obstacles)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nDOF obstacle preprocessing complete:")
    print(f"  Input:              {input_path}")
    print(f"  Center:             ({center_lat:.4f}, {center_lon:.4f})")
    print(f"  Radius:             {args.radius_km} km")
    print(f"  Parse errors:       {parse_errors}")
    print(f"  Skipped unverified: {skipped_unverified}")
    print(f"  Skipped (distance): {skipped_distance}")
    print(f"  Obstacles written:  {len(obstacles)}")
    print(f"  Output:             {output_path}")


if __name__ == "__main__":
    main()
