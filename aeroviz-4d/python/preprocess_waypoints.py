"""
preprocess_waypoints.py
=======================
Builds public/data/airports/<ICAO>/waypoints.geojson from OpenNav waypoint listings.

Default behavior:
- Downloads Canada waypoints from OpenNav
- Parses waypoint ident + latitude + longitude from HTML table rows
- Filters waypoints within a radius around CYLW
- Assigns display-oriented waypoint roles (IAF/IF/FAF/MAPt)
- Writes GeoJSON FeatureCollection for Cesium waypoint rendering

Usage:
  python preprocess_waypoints.py
  python preprocess_waypoints.py --center-lon -119.3775 --center-lat 49.9561 --radius-km 120 --max-waypoints 40
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from data_layout import airport_data_path, find_airport_record, resolve_common_csv

DEFAULT_SOURCE_URL = "https://opennav.com/waypoint/CA"
DEFAULT_AIRPORT = "CYLW"
DEFAULT_RADIUS_KM = 120.0
DEFAULT_MAX_WAYPOINTS = 30
DEFAULT_PROCEDURE = "OpenNav Canada waypoint sample"

ROW_RE = re.compile(
    r'<tr><td><a href="/waypoint/[^"]+">([^<]+)</a></td>'
    r'<td[^>]*>.*?</td><td>([^<]+)</td><td[^>]*>.*?</td><td>([^<]+)</td></tr>',
    re.IGNORECASE,
)
DMS_RE = re.compile(
    r"^(\d{1,3})\s*[°º]\s*(\d{1,2})\s*'\s*(\d{1,2}(?:\.\d+)?)\s*\"?\s*([NSEW])$",
    re.IGNORECASE,
)
DMS_NO_SEC_RE = re.compile(
    r"^(\d{1,3})\s*[°º]\s*(\d{1,2}(?:\.\d+)?)\s*'\s*([NSEW])$",
    re.IGNORECASE,
)
COMPACT_DMS_RE = re.compile(r"^(\d{4,7})([NSEW])$", re.IGNORECASE)


def dms_to_decimal(raw_value: str) -> float | None:
    """Convert a coordinate string (DMS/compact DMS/decimal) to signed decimal degrees."""
    value = html.unescape(raw_value).strip().upper()

    match = DMS_RE.match(value)
    if match:
        deg = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        hemisphere = match.group(4)
        decimal = deg + minutes / 60.0 + seconds / 3600.0
        if hemisphere in {"S", "W"}:
            decimal *= -1.0
        return decimal

    match = DMS_NO_SEC_RE.match(value)
    if match:
        deg = float(match.group(1))
        minutes = float(match.group(2))
        hemisphere = match.group(3)
        decimal = deg + minutes / 60.0
        if hemisphere in {"S", "W"}:
            decimal *= -1.0
        return decimal

    match = COMPACT_DMS_RE.match(value)
    if match:
        digits = match.group(1)
        hemisphere = match.group(2)
        deg_width = 2 if hemisphere in {"N", "S"} else 3

        if len(digits) == deg_width + 4:
            deg = float(digits[:deg_width])
            minutes = float(digits[deg_width : deg_width + 2])
            seconds = float(digits[deg_width + 2 :])
            decimal = deg + minutes / 60.0 + seconds / 3600.0
        elif len(digits) == deg_width + 2:
            deg = float(digits[:deg_width])
            minutes = float(digits[deg_width:])
            decimal = deg + minutes / 60.0
        else:
            return None

        if hemisphere in {"S", "W"}:
            decimal *= -1.0
        return decimal

    try:
        return float(value)
    except ValueError:
        return None


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers."""
    r_earth_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth_km * c


def assign_waypoint_role(rank: int, total: int) -> str:
    """Assign role from far-to-near ordering for display purposes."""
    if total <= 1:
        return "MAPt"

    progress = rank / (total - 1)
    if progress < 0.4:
        return "IAF"
    if progress < 0.75:
        return "IF"
    if progress < 0.95:
        return "FAF"
    return "MAPt"


def parse_opennav_rows(page_html: str) -> list[dict[str, str]]:
    """Extract {ident, lat_raw, lon_raw} rows from OpenNav waypoint table HTML."""
    rows: list[dict[str, str]] = []
    for ident, lat_raw, lon_raw in ROW_RE.findall(page_html):
        rows.append(
            {
                "ident": html.unescape(ident).strip().upper(),
                "lat_raw": html.unescape(lat_raw).strip(),
                "lon_raw": html.unescape(lon_raw).strip(),
            }
        )
    return rows


def build_waypoint_geojson(
    page_html: str,
    center_lon: float,
    center_lat: float,
    radius_km: float,
    max_waypoints: int,
    procedure: str,
) -> dict[str, Any]:
    """Convert parsed OpenNav rows into GeoJSON filtered around a center point."""
    parsed_rows = parse_opennav_rows(page_html)

    selected: list[dict[str, Any]] = []
    for row in parsed_rows:
        lat = dms_to_decimal(row["lat_raw"])
        lon = dms_to_decimal(row["lon_raw"])
        if lat is None or lon is None:
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        distance = haversine_km(center_lon, center_lat, lon, lat)
        if distance <= radius_km:
            selected.append(
                {
                    "ident": row["ident"],
                    "lat": lat,
                    "lon": lon,
                    "distance_km": distance,
                }
            )

    if not selected:
        raise RuntimeError("No waypoints found within filter radius. Try a larger --radius-km.")

    # Sort far-to-near so sequence resembles inbound approach progression.
    selected.sort(key=lambda item: item["distance_km"], reverse=True)
    selected = selected[:max_waypoints]

    features: list[dict[str, Any]] = []
    total = len(selected)
    for idx, item in enumerate(selected, start=1):
        role = assign_waypoint_role(idx - 1, total)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [item["lon"], item["lat"], 0.0],
                },
                "properties": {
                    "name": item["ident"],
                    "type": role,
                    "min_alt_ft": None,
                    "procedure": procedure,
                    "sequence": idx,
                    "source": "opennav",
                    "distance_km": round(item["distance_km"], 2),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def resolve_waypoint_center(
    airport_code: str,
    airports_csv: Path,
    center_lon: float | None,
    center_lat: float | None,
) -> tuple[float, float]:
    airport_record = find_airport_record(resolve_common_csv(airports_csv), airport_code)
    resolved_lon = center_lon if center_lon is not None else airport_record["lon"]
    resolved_lat = center_lat if center_lat is not None else airport_record["lat"]
    return resolved_lon, resolved_lat


def resolve_waypoint_output_path(airport_code: str, output: str | None) -> Path:
    return Path(output) if output else airport_data_path(airport_code, "waypoints.geojson")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate waypoints.geojson from OpenNav waypoints")
    parser.add_argument("--airport", default=DEFAULT_AIRPORT, help="Airport code used for default center/output")
    parser.add_argument("--airports-csv", default="airports.csv", help="OurAirports airports.csv path")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="OpenNav waypoint page URL")
    parser.add_argument("--center-lon", type=float, default=None, help="Reference longitude")
    parser.add_argument("--center-lat", type=float, default=None, help="Reference latitude")
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM, help="Filter radius in kilometers")
    parser.add_argument("--max-waypoints", type=int, default=DEFAULT_MAX_WAYPOINTS, help="Maximum output features")
    parser.add_argument("--procedure", default=DEFAULT_PROCEDURE, help="Procedure label in feature properties")
    parser.add_argument("--output", default=None, help="Output GeoJSON path")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()

    center_lon, center_lat = resolve_waypoint_center(
        args.airport,
        Path(args.airports_csv),
        args.center_lon,
        args.center_lat,
    )

    with urlopen(args.source_url, timeout=args.timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")

    collection = build_waypoint_geojson(
        page_html=page_html,
        center_lon=center_lon,
        center_lat=center_lat,
        radius_km=args.radius_km,
        max_waypoints=args.max_waypoints,
        procedure=args.procedure,
    )

    output_path = resolve_waypoint_output_path(args.airport, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    print(f"Generated {len(collection['features'])} waypoints")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
