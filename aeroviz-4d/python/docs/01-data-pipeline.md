# Tutorial — Python Data Pipeline

**Covers:** `preprocess_airports.py`, `generate_czml.py`, and their tests

---

## Overview

The Python side of this project has two jobs:

| Script | Input | Output |
|--------|-------|--------|
| `preprocess_airports.py` | OurAirports `runways.csv` | `runway.geojson` (two polygons per runway) |
| `generate_czml.py` | Scheduling algorithm output | `trajectories.czml` |

Both outputs land in `aeroviz-4d/public/data/` where the Vite dev server
serves them as static files.

---

## Part 1 — preprocess_airports.py

### Data source: OurAirports

OurAirports (https://ourairports.com/data/) publishes free, CC0-licensed
aviation data.  Download `runways.csv` (≈ 5 MB) — it has one row per runway
end, with columns:

| Column | Example | Meaning |
|--------|---------|---------|
| `airport_ident` | CYLW | ICAO airport code |
| `le_ident` | 16 | Lower-end runway identifier, magnetic heading/10 |
| `he_ident` | 34 | Higher-end identifier |
| `le_displaced_threshold_ft` | 1200 | LE threshold displacement (feet) |
| `he_displaced_threshold_ft` | 400 | HE threshold displacement (feet) |
| `le_longitude_deg` | -119.38 | Lower-end threshold longitude |
| `le_latitude_deg` | 49.92 | Lower-end threshold latitude |
| `he_longitude_deg` | -119.38 | Higher-end threshold longitude |
| `he_latitude_deg` | 49.96 | Higher-end threshold latitude |
| `length_ft` | 8200 | Total runway length in feet |
| `width_ft` | 150 | Width in feet |
| `surface` | ASP | Surface type (ASP=asphalt, CON=concrete, GRS=grass) |

### Your TODOs

You need to implement three functions:

#### `runway_bearing_rad(le_lon, le_lat, he_lon, he_lat)`

Compute the bearing from the LE threshold to the HE threshold.
This is the SAME formula as `bearingRad()` in the TypeScript code.

**Pseudocode:**
```
dx = (he_lon - le_lon) × metres_per_deg_lon(le_lat)
dy = (he_lat - le_lat) × 111320
return atan2(dx, dy)
```

**What this means (intuition):**
- `bearing` is the direction from LE to HE, measured clockwise from north.
- `dx` is the east-west component in metres.
- `dy` is the north-south component in metres.
- `atan2(dx, dy)` converts those two components into one heading angle (radians).

Why scale longitude by latitude?
- One degree of longitude is not a constant distance on Earth.
- At latitude `φ`, an approximation is:
    `metres_per_deg_lon(φ) ≈ 111320 × cos(φ)`
- So `dx` must use `metres_per_deg_lon(le_lat)`; otherwise east-west distance is wrong,
    especially at higher latitudes.

#### `offset_point_deg(lon, lat, bearing_rad, distance_m)`

Move a point `distance_m` metres in direction `bearing_rad`.
Same flat-Earth formula as TypeScript `offsetPoint()`.

Returns `(new_lon, new_lat)`.

#### `runway_to_polygon(runway: RunwayEnds)`

Build the four-corner polygon ring for a runway.

```
                       ← half_width →
LE_left  ─────────────────────────────  HE_left
         │       centreline →          │
LE_right ─────────────────────────────  HE_right
```

Steps:
1. Compute centreline bearing with `runway_bearing_rad()`
2. `perp_left  = bearing - π/2`
3. `perp_right = bearing + π/2`
4. `half_w_m   = (width_ft × 0.3048) / 2`
5. Offset each threshold end left and right by `half_w_m`
6. Return the 5-point closed ring: `[LE_left, LE_right, HE_right, HE_left, LE_left]`

What does `perp` mean?
- `perp` is short for **perpendicular** (90° to the centreline direction).
- The runway centreline uses `bearing` (LE -> HE).
- To move sideways from the centreline (to get runway edges), you need a direction
    that is exactly perpendicular to that bearing.

Why `± π/2`?
- `π/2` radians = 90°.
- `bearing - π/2` rotates the heading 90° to one side (left).
- `bearing + π/2` rotates the heading 90° to the other side (right).
- Those two perpendicular directions are then used in `offset_point_deg(...)`
    to compute left/right edge points at each runway end.

Quick intuition:
- Centreline direction tells you "forward".
- Perpendicular directions tell you "sideways left" and "sideways right".
- Runway polygon = forward endpoints shifted sideways by half width.

⚠️ GeoJSON polygon rings use `[longitude, latitude]` order, NOT `[lat, lon]`!

### Dual polygon output (important)

`runway.geojson` now emits **two Features per runway**:

1. `zone_type = "runway_surface"`
    Uses displaced-threshold values to extend from threshold positions to the
    physical pavement ends.

2. `zone_type = "landing_zone"`
    Uses threshold-to-threshold geometry (the touchdown-allowed region).

Frontend rendering can then style these differently, e.g.:
- runway surface: dark grey
- landing zone: light green

### Running the tests

```bash
cd python
pytest tests/test_preprocess_airports.py -v
```

---

## Part 2 — generate_czml.py

### Your TODOs

#### `build_position_property(epoch_dt, waypoints)`

Build the CZML position dict.  The key is constructing the flat
`cartographicDegrees` list correctly:

```python
flat = []
for (offset_sec, lon, lat, alt_m) in waypoints:
    flat.extend([offset_sec, lon, lat, alt_m])

return {
    "epoch": epoch_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cartographicDegrees": flat,
    "interpolationAlgorithm": "LAGRANGE",
    "interpolationDegree": 3,
    "forwardExtrapolationType": "HOLD",
}
```

#### `build_flight_packet(...)`

Assemble the full entity dict.  The most important fields are:
- `"position"` → result of `build_position_property()`
- `"orientation"` → `{"velocityReference": f"#{flight_id}.position"}`
  (Cesium auto-computes heading from velocity — you don't do angle math here)

#### `build_czml(flights, start_dt, multiplier)`

Three steps:
1. Find the max offset to compute `end_dt`
2. Build the document packet with `build_document_packet(start_dt, end_dt, multiplier)`
3. Build one entity packet per flight with `build_flight_packet()`
4. Return `[document, *entities]`

### Running the tests

```bash
pytest tests/test_generate_czml.py -v
```

### Manual integration test

```bash
python generate_czml.py   # uses mock data
# Then in the aeroviz-4d directory:
npm run dev
# Open http://localhost:5173 — two aircraft should appear and move
```

---

## Part 3 — Connecting your scheduling algorithm

When your MILP/genetic algorithm produces an optimised sequence, convert its
output to the `flights` list format and call `build_czml()`:

```python
# In your algorithm module:
from generate_czml import build_czml, build_document_packet
from pathlib import Path
from datetime import datetime, timezone
import json

# Your algorithm returns something like this:
scheduled_flights = milp_solve(arrivals, constraints)

# Convert to the CZML input format:
flights_for_czml = []
for flight in scheduled_flights:
    waypoints = trajectory_predictor.predict_4d(flight)  # your function
    flights_for_czml.append({
        "id": flight.callsign,
        "callsign": flight.callsign,
        "type": flight.aircraft_type,
        "waypoints": [(wp.t, wp.lon, wp.lat, wp.alt_m) for wp in waypoints],
    })

czml = build_czml(flights_for_czml, start_dt=datetime(2026,4,1,8,0,0, tzinfo=timezone.utc))

output = Path("../aeroviz-4d/public/data/trajectories.czml")
output.write_text(json.dumps(czml, indent=2))
print("Visualisation updated — refresh browser")
```

No changes to the React frontend are needed.
