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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "public" / "data"
DEFAULT_OUTPUT = OUTPUT_DIR / "trajectories.czml"

# Colour palette for trail polylines (RGBA 0-255)
TRAIL_COLORS = [
    (255, 140,   0, 200),   # orange
    (  0, 191, 255, 200),   # deep sky blue
    (124, 252,   0, 200),   # lawn green
    (255,  20, 147, 200),   # deep pink
    (138,  43, 226, 200),   # blue violet
]


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

    CZML interprets this with Lagrange interpolation between samples.
    The `epoch` key is the ISO string of epoch_dt.
    """
    flat: list[float] = []
    for offset_sec, lon, lat, alt_m in waypoints:
        flat.extend([offset_sec, lon, lat, alt_m])

    return {
        "epoch": epoch_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cartographicDegrees": flat,
        "interpolationAlgorithm": "LAGRANGE",
        "interpolationDegree": 3,
        "forwardExtrapolationType": "HOLD",
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
        "orientation": {"velocityReference": f"#{flight_id}.position"},
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


# ── Mock data (for front-end integration testing) ─────────────────────────────

MOCK_FLIGHTS = [
    {
        "id": "UAL123",
        "callsign": "United 123",
        "type": "B738",
        # Each waypoint: [offset_seconds, longitude, latitude, altitude_metres]
        # This represents a straight inbound approach to CYLW runway 34
        "waypoints": [
            [0,    -119.10, 50.20, 5500],
            [180,  -119.20, 50.10, 4800],
            [360,  -119.30, 50.00, 4000],
            [540,  -119.36, 49.97, 3200],
            [660,  -119.38, 49.96, 2500],
            [780,  -119.385, 49.957, 1800],
            [900,  -119.390, 49.955, 600],
        ],
    },
    {
        "id": "WJA456",
        "callsign": "WestJet 456",
        "type": "B737",
        # This flight is delayed by ~240 s (4 min) to maintain separation
        "waypoints": [
            [0,    -119.05, 50.30, 6000],
            [240,  -119.15, 50.15, 5200],
            [480,  -119.28, 50.02, 4200],
            [720,  -119.35, 49.98, 3300],
            [900,  -119.37, 49.96, 2600],
            [1020, -119.382, 49.958, 1900],
            [1140, -119.390, 49.955, 600],
        ],
    },
]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate CZML trajectory file")
    parser.add_argument("--input",  default=None, help="JSON file with flight data")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CZML path")
    parser.add_argument("--multiplier", type=int, default=60, help="Clock speed multiplier")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            flights = json.load(f)
    else:
        print("No --input provided; using built-in mock data.")
        flights = MOCK_FLIGHTS

    start_dt = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
    czml = build_czml(flights, start_dt, args.multiplier)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(czml, indent=2, ensure_ascii=False))

    print(f"✓ Generated CZML for {len(flights)} flight(s)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
