"""
generate_czml.py
================
Generates CZML trajectory files from 4D flight data produced by the
scheduling algorithm.

This script is the bridge between your Python optimisation code and the
CesiumJS frontend.  The frontend loads trajectories.czml as-is —
you never need to touch the React code to update the visualisation.

Usage:
  python generate_czml.py                  # uses built-in mock data
  python generate_czml.py --input my_flights.json --output custom.czml

Input JSON format (when using --input):
  [
    { "id": "UAL123", "callsign": "United 123", "type": "B738",
      "waypoints": [[0, -119.38, 49.95, 4500], [120, -119.40, 49.90, 3800]] }
  ]
  Each waypoint is [offset_seconds, longitude, latitude, altitude_metres].

📖 Tutorial: see docs/02-czml-generation.md
"""

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "public" / "data"
DEFAULT_OUTPUT = OUTPUT_DIR / "trajectories.czml"

# Default input: real OpenSky data downloaded by opensky_cylw/fetch_cylw_opensky.py
DEFAULT_INPUT = Path(__file__).parent.parent.parent / "opensky_cylw" / "outputs" / "cylw_czml_input_20260405T223356Z.json"

# Colour palette for trail polylines (RGBA 0-255)
TRAIL_COLORS = [
    (255, 140,   0, 200),   # orange
    (  0, 191, 255, 200),   # deep sky blue
    (124, 252,   0, 200),   # lawn green
    (255,  20, 147, 200),   # deep pink
    (138,  43, 226, 200),   # blue violet
]


# ── Velocity / orientation math ───────────────────────────────────────────────
# All helpers are pure math — no external dependencies beyond the stdlib.

_EARTH_RADIUS_M = 6_371_000  # mean Earth radius in metres


def _great_circle_bearing(
    lon1_deg: float, lat1_deg: float,
    lon2_deg: float, lat2_deg: float,
) -> float:
    """
    Initial bearing from point 1 to point 2 on a sphere.

    Returns radians, 0 = North, positive = clockwise (East).
    """
    lon1, lat1 = math.radians(lon1_deg), math.radians(lat1_deg)
    lon2, lat2 = math.radians(lon2_deg), math.radians(lat2_deg)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return math.atan2(y, x)


def _haversine_distance(
    lon1_deg: float, lat1_deg: float,
    lon2_deg: float, lat2_deg: float,
) -> float:
    """Great-circle distance between two points, in metres."""
    lon1, lat1 = math.radians(lon1_deg), math.radians(lat1_deg)
    lon2, lat2 = math.radians(lon2_deg), math.radians(lat2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def compute_velocity_orientation(
    waypoints: list[tuple[float, float, float, float]],
) -> list[tuple[float, float]]:
    """
    Derive heading and pitch at each waypoint from the 3-D velocity vector.

    Parameters
    ----------
    waypoints : list of (offset_sec, lon_deg, lat_deg, alt_m)

    Returns
    -------
    list of (heading_rad, pitch_rad) per waypoint.
    heading: 0 = North, positive = clockwise (East).
    pitch:   positive = climbing.
    """
    n = len(waypoints)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]

    result: list[tuple[float, float]] = []
    for idx in range(n):
        # Forward difference for all but last; backward for last.
        if idx < n - 1:
            i, j = idx, idx + 1
        else:
            i, j = idx - 1, idx

        _, lon1, lat1, alt1 = waypoints[i]
        _, lon2, lat2, alt2 = waypoints[j]

        heading = _great_circle_bearing(lon1, lat1, lon2, lat2)
        horiz = _haversine_distance(lon1, lat1, lon2, lat2)
        pitch = math.atan2(alt2 - alt1, horiz) if horiz > 1e-6 else 0.0

        result.append((heading, pitch))
    return result


# ── Matrix / quaternion helpers ──────────────────────────────────────────────

def _mat3_multiply(
    a: list[list[float]], b: list[list[float]],
) -> list[list[float]]:
    """Multiply two 3×3 row-major matrices."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _mat3_to_quaternion(m: list[list[float]]) -> tuple[float, float, float, float]:
    """
    Convert a 3×3 rotation matrix to a unit quaternion (x, y, z, w).

    Uses Shepperd's method for numerical stability.
    """
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def _hpr_to_ecef_quaternion(
    heading: float, pitch: float,
    lon_deg: float, lat_deg: float,
) -> tuple[float, float, float, float]:
    """
    Compute an ECEF orientation quaternion for an aircraft at (lon, lat)
    with given heading and pitch (roll = 0, level flight).

    Coordinate conventions
    ----------------------
    ENU local frame: X = East, Y = North, Z = Up.
    Model axes:      +X = forward (nose), +Y = left, +Z = up.
    heading:         0 = North, positive = clockwise (toward East).
    pitch:           positive = climbing.

    Steps
    -----
    1. Build a model-to-ENU rotation from heading and pitch.
    2. Multiply by the ENU-to-ECEF rotation at (lon, lat).
    3. Convert the combined matrix to a unit quaternion.
    """
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    ch, sh = math.cos(heading), math.sin(heading)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Model-to-ENU  (model +X = forward/nose → heading direction in ENU)
    #   Column 0: model +X (forward) → ENU
    #   Column 1: model +Y (left)    → ENU
    #   Column 2: model +Z (up)      → ENU
    m_model = [
        [ sh * cp,  -ch,  -sh * sp],
        [ ch * cp,   sh,  -ch * sp],
        [ sp,        0.0,  cp     ],
    ]

    # ENU-to-ECEF at (lon, lat)
    slon, clon = math.sin(lon), math.cos(lon)
    slat, clat = math.sin(lat), math.cos(lat)
    m_enu = [
        [-slon,  -slat * clon,  clat * clon],
        [ clon,  -slat * slon,  clat * slon],
        [ 0.0,    clat,         slat       ],
    ]

    m_ecef = _mat3_multiply(m_enu, m_model)
    return _mat3_to_quaternion(m_ecef)


# ── CZML packet builders ──────────────────────────────────────────────────────

def build_document_packet(
    start_dt: datetime,
    end_dt: datetime,
    multiplier: int = 60,
) -> dict[str, Any]:
    """
    Build the mandatory CZML "document" packet (always the first element).

    The document packet defines the simulation clock:
      - interval:    the full time range (start/end as ISO 8601)
      - currentTime: where playback begins (usually == start)
      - multiplier:  1 real second = `multiplier` simulation seconds
      - range:       LOOP_STOP = pause at end (vs CLAMPED = hold last value)
    """
    def iso(dt: datetime) -> str:
        """Convert datetime to ISO 8601 UTC string with Z suffix."""
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "id": "document",
        "name": "AeroViz-4D Trajectories",
        "version": "1.0",
        "clock": {
            "interval":    f"{iso(start_dt)}/{iso(end_dt)}",
            "currentTime": iso(start_dt),
            "multiplier":  multiplier,
            "range":       "LOOP_STOP",
            "step":        "SYSTEM_CLOCK_MULTIPLIER",
        },
    }


def build_position_property(
    epoch_dt: datetime,
    waypoints: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    """
    Build the CZML `position` property for one aircraft.

    Parameters
    ----------
    epoch_dt : datetime
        The reference epoch.  All time offsets in `waypoints` are SECONDS
        relative to this moment.

    waypoints : list of (offset_sec, lon, lat, alt_m)
        Sorted ascending by offset_sec.

    Returns
    -------
    A dict with key "cartographicDegrees" containing the flat array:
      [t0, lon0, lat0, alt0,  t1, lon1, lat1, alt1, ...]

    CZML interprets this with linear interpolation between samples to
    preserve the sampled geometry without polynomial overshoot.
    The `epoch` key is the ISO string of epoch_dt.
    """
    flat: list[float] = []
    for offset_sec, lon, lat, alt_m in waypoints:
        flat.extend([offset_sec, lon, lat, alt_m])

    return {
        "epoch": epoch_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cartographicDegrees": flat,
        "interpolationAlgorithm": "LINEAR",
        "forwardExtrapolationType": "HOLD",
    }


def build_orientation_property(
    epoch_dt: datetime,
    waypoints: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    """
    Build the CZML ``orientation`` property with explicit unitQuaternion
    samples derived from the 3-D velocity vector between consecutive
    waypoints.

    Each sample is [offset_sec, x, y, z, w] in CZML's ``unitQuaternion``
    flat array.  Cesium interpolates between samples using SLERP.
    """
    orientations = compute_velocity_orientation(waypoints)

    flat: list[float] = []
    for (offset, lon, lat, _alt), (heading, pitch) in zip(waypoints, orientations):
        x, y, z, w = _hpr_to_ecef_quaternion(heading, pitch, lon, lat)
        flat.extend([offset, x, y, z, w])

    return {
        "epoch": epoch_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unitQuaternion": flat,
        "interpolationAlgorithm": "LINEAR",
    }


def build_flight_packet(
    flight_id: str,
    callsign: str,
    aircraft_type: str,
    waypoints: list[tuple[float, float, float, float]],
    epoch_dt: datetime,
    color_rgba: tuple[int, int, int, int],
) -> dict[str, Any]:
    """
    Build a complete CZML entity packet for one aircraft.

    Parameters
    ----------
    flight_id     : unique string ID, e.g. "UAL123"
    callsign      : human-readable name, e.g. "United 123"
    aircraft_type : ICAO type code, e.g. "B738"
    waypoints     : list of (offset_sec, lon, lat, alt_m)
    epoch_dt      : simulation start time (all offsets relative to this)
    color_rgba    : (R, G, B, A) each 0–255 for the trail polyline
    """
    return {
        "id": flight_id,
        "name": callsign,
        "description": f"<b>{callsign}</b><br/>Type: {aircraft_type}",
        "model": {
            "gltf": "/models/aircraft.glb",
            "scale": 3.0,
            "minimumPixelSize": 32,
            "maximumScale": 20000,
            "runAnimations": True,
        },
        "position": build_position_property(epoch_dt, waypoints),
        "orientation": build_orientation_property(epoch_dt, waypoints),
        "path": {
            "show": True,
            "leadTime": 0,
            "trailTime": 300,
            "width": 2,
            "material": {
                "solidColor": {
                    "color": {"rgba": list(color_rgba)}
                }
            },
        },
        "label": {
            "text": callsign,
            "font": "12px sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -30]},
        },
    }


# ── Top-level assembler ───────────────────────────────────────────────────────

def build_czml(
    flights: list[dict[str, Any]],
    start_dt: datetime,
    multiplier: int = 60,
) -> list[dict[str, Any]]:
    """
    Assemble a complete CZML document from a list of flight dicts.

    Each dict in `flights` must have:
      id        : str
      callsign  : str
      type      : str
      waypoints : list of [offset_sec, lon, lat, alt_m]

    Returns a CZML array: [document_packet, entity_packet, entity_packet, ...]
    """
    max_offset = max(
        (float(wp[0]) for flight in flights for wp in flight.get("waypoints", [])),
        default=0.0,
    )
    end_dt = start_dt + timedelta(seconds=max_offset)
    doc = build_document_packet(start_dt, end_dt, multiplier)

    entity_packets: list[dict[str, Any]] = []
    for i, flight in enumerate(flights):
        color = TRAIL_COLORS[i % len(TRAIL_COLORS)]
        packet = build_flight_packet(
            str(flight["id"]),
            str(flight["callsign"]),
            str(flight["type"]),
            [tuple(wp) for wp in flight["waypoints"]],
            start_dt,
            color,
        )
        entity_packets.append(packet)

    return [doc, *entity_packets]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate CZML trajectory file")
    parser.add_argument("--input",  default=str(DEFAULT_INPUT), help="JSON file with flight data")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CZML path")
    parser.add_argument("--multiplier", type=int, default=60, help="Clock speed multiplier")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        print("  Run opensky_cylw/fetch_cylw_opensky.py first to download real flight data.")
        raise SystemExit(1)

    with open(input_path) as f:
        flights = json.load(f)

    start_dt = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
    czml = build_czml(flights, start_dt, args.multiplier)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(czml, indent=2, ensure_ascii=False))

    print(f"✓ Generated CZML for {len(flights)} flight(s)")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
