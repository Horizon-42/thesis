# Data formats: from raw Trino rows to CZML

> **Archived format reference:** examples below describe the removed bare-array download
> path. The current roster and record layout is `tracks/manifest.json` →
> `arrivals/manifest.json`, documented in `trajectory_data_process/README.md`.

The pipeline carries flight data through **three JSON shapes**. Each stage has one
owner and one job, and the boundaries between them are deliberately simple so the
data package and the frontend stay decoupled.

```
OpenSky history DB (Trino)
        │  acquisition/opensky_history.py      ── Stage 1: raw state-vector rows
        ▼
  Trajectory model (in memory)                 ── grouping/segmentation only
        │  processing/czml_export.py
        ▼
  CZML-input JSON  (*_czml_input_*.json)        ── Stage 2: neutral interchange
        │  aeroviz-4d/python/generate_czml.py
        ▼
  CZML  (trajectories.czml)                     ── Stage 3: what CesiumJS renders
```

| Stage | Format | Produced by | One unit is… |
|-------|--------|-------------|--------------|
| 1 | raw history rows | `acquisition/opensky_history.py` | one aircraft at **one second** (a state vector) |
| 2 | CZML-input JSON | `processing/czml_export.py` | one **flight** (a full approach track) |
| 3 | CZML | `aeroviz-4d/python/generate_czml.py` | one CZML **packet** (document + one per flight) |

Stage 2 is the **seam**: everything before it is "acquire and process flight data";
everything after it is "render for Cesium". Stage 2 carries no Cesium concepts
(no models, colours, quaternions); Stage 3 is entirely presentation.

---

## Stage 1 — Raw OpenSky history rows

`acquisition/opensky_history.py` asks the OpenSky **Trino history database** (through
the `traffic` package) for state-vector rows. The columns requested are:

```python
STATE_VECTOR_COLUMNS = (
    "time", "icao24", "lat", "lon", "velocity", "heading", "vertrate",
    "callsign", "onground", "baroaltitude", "geoaltitude",
)
# + when querying by airport: FlightsData4.estdepartureairport / estarrivalairport
```

`traffic` hands the result back as a DataFrame **renamed into its own vocabulary**
and in **aviation units** (feet, knots, ft/min). One real row:

```json
{
  "timestamp":          "2026-04-19T10:00:01Z",
  "icao24":             "a046f2",
  "latitude":            35.83884546312235,
  "longitude":          -78.91198990192817,
  "geoaltitude":         5025.0,
  "altitude":            5000.0,
  "groundspeed":         240.0,
  "track":               267.6140559696112,
  "vertical_rate":       2496.0,
  "onground":            false,
  "callsign":           "RPA4619",
  "estdepartureairport": "KRDU",
  "estarrivalairport":   "KORD"
}
```

| Field (traffic name) | Requested as | Meaning |
|---|---|---|
| `timestamp` | `time` | instant of the sample (UTC) |
| `icao24` | `icao24` | unique 24-bit aircraft address (hex) |
| `latitude` / `longitude` | `lat` / `lon` | position, degrees |
| `geoaltitude` | `geoaltitude` | **geometric** altitude (ellipsoid) — the one used downstream |
| `altitude` | `baroaltitude` | barometric altitude — kept for reference only |
| `groundspeed` | `velocity` | speed (knots) — fetched but unused downstream |
| `track` | `heading` | track angle, degrees (0 = N, CW) |
| `vertical_rate` | `vertrate` | climb/descent (ft/min) — fetched but unused downstream |
| `onground` | `onground` | on-ground flag |
| `callsign` | `callsign` | flight callsign |
| `estdepartureairport` / `estarrivalairport` | — | estimated dep/arr airport (airport-join only; `null` for the bbox/landings path) |

**The one transformation at this boundary:** `fetch_history_dataframe` converts the
three altitude columns from feet to **metres** (`× 0.3048`) on the way out. So the row
above becomes `geoaltitude ≈ 1531.6`, `altitude ≈ 1524.0` (m). Everything past this
point is metric. Latitude/longitude/heading/time are untouched.

### Between Stage 1 and Stage 2 — the Trajectory model

Rows are not exported one-by-one. `trajectory.py` groups them into
`Trajectory` objects (in memory only, in CZML mode):

- group all rows by `icao24`;
- **split into separate tracks** on a time gap larger than `--segment-gap-sec`
  (default 900 s) or a change of departure/arrival airport;
- each kept sample becomes a `TrajectoryPoint(time, lat, lon, geo_altitude_m,
  baro_altitude_m, heading_deg, on_ground)`; points with no geometric altitude are
  dropped.

This is a structural step (grouping + segmentation), not a format the pipeline
writes in CZML mode — it is the bridge that lets Stage 2 work on whole flights.

---

## Stage 2 — CZML-input JSON

`processing/czml_export.py` turns each relevant `Trajectory` into one **CZML-input
flight**. This is the neutral interchange format — a plain list of flights, no Cesium
vocabulary. It is what `*_czml_input_*.json` (and the per-runway
`*_landings.json`) files contain. One real flight (waypoints trimmed):

```json
{
  "id":              "AFR074",
  "callsign":        "AFR074",
  "type":            "UNK",
  "icao24":          "3949ea",
  "dep_airport":      null,
  "arr_airport":      null,
  "runway":          "05L",
  "landing_time_utc": "2026-06-18T20:07:51Z",
  "altitude_source": "opensky_history_geoaltitude_m",
  "waypoints": [
    [0, -78.454857, 35.739853, 2537.5],
    [1, -78.455715, 35.740219, 2537.5],
    [2, -78.457168, 35.740806, 2537.5]
    // …742 waypoints total
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | unique id (from callsign, de-duplicated) |
| `callsign` / `type` | display name / ICAO type (`UNK` — not derived here) |
| `icao24` | aircraft address, carried through from Stage 1 |
| `dep_airport` / `arr_airport` | from the airport join; `null` for the bbox/landings path |
| `runway` | runway threshold this flight was selected for (`null` if not runway-filtered) |
| `landing_time_utc` | absolute time the flight reached the threshold (landings only) |
| `altitude_source` | provenance tag — always geometric altitude |
| `waypoints` | the track: a list of `[offset_sec, lon, lat, alt_m]` |

**Each waypoint is `[offset_seconds, longitude, latitude, geometric_altitude_m]`.**
Note the order: time first, then **lon before lat** (GeoJSON convention), then metric
altitude.

What Stage 2 changes from the raw rows:

- **Absolute time → relative offset.** `timestamp` becomes `offset_sec`, counted from
  the **first kept waypoint** (so the first is always `0`).
- **Only geometric altitude survives** into the waypoint; barometric is left behind.
- **Selection + trimming.** Only flights that approach the airport (or land at the
  requested runway threshold) are kept, and only the last `--approach-window-min`
  minutes before the arrival anchor (25 min in the landings flow). See the README for
  the runway/landing selection rules.
- **Rounding.** lon/lat to 6 decimals, altitude to 1 decimal.

---

## Stage 3 — CZML

`aeroviz-4d/python/generate_czml.py` renders the CZML-input into **CZML** — the
native time-dynamic format CesiumJS loads directly (`trajectories.czml`). A CZML
document is a JSON **array of packets**: the first is the `"document"` packet (the
clock), and each following packet is one aircraft entity.

### The document packet (always element 0)

```json
{
  "id": "document",
  "name": "AeroViz-4D Trajectories",
  "version": "1.0",
  "clock": {
    "interval":    "2026-04-01T08:00:00Z/2026-04-01T08:25:00Z",
    "currentTime": "2026-04-01T08:00:00Z",
    "multiplier":  60,
    "range":       "LOOP_STOP",
    "step":        "SYSTEM_CLOCK_MULTIPLIER"
  }
}
```

| Clock field | Meaning |
|---|---|
| `interval` | playback span: `epoch / epoch + longest track` |
| `currentTime` | where playback starts (= epoch) |
| `multiplier` | `60` → 1 real second = 60 simulated seconds |
| `range` | `LOOP_STOP` → pause at the end (vs loop) |

The **epoch is a fixed display epoch** chosen by the generator
(`2026-04-01T08:00:00Z`), not the real landing time — the *offsets* preserve the real
relative timing, so the motion is faithful even though the absolute clock is synthetic
(the true wall-clock time lives in Stage 2's `landing_time_utc`).

### An entity packet (one per flight)

```json
{
  "id": "AFR074",
  "name": "AFR074",
  "description": "<b>AFR074</b><br/>Type: UNK",
  "model": { "gltf": "/models/aircraft.glb", "scale": 3.0, "minimumPixelSize": 32,
             "maximumScale": 20000, "runAnimations": true },
  "position": {
    "epoch": "2026-04-01T08:00:00Z",
    "cartographicDegrees": [0, -78.454857, 35.739853, 2537.5,
                            1, -78.455715, 35.740219, 2537.5,  /* … */],
    "interpolationAlgorithm": "LINEAR",
    "forwardExtrapolationType": "HOLD"
  },
  "orientation": {
    "epoch": "2026-04-01T08:00:00Z",
    "unitQuaternion": [0, 0.15323018, -0.42949685, 0.88111772,
                       1, /* x,y,z,w … */],
    "interpolationAlgorithm": "LINEAR"
  },
  "path":  { "show": true, "leadTime": 0, "trailTime": 300, "width": 2,
             "material": { "solidColor": { "color": { "rgba": [255,140,0,200] } } } },
  "label": { "text": "AFR074", "font": "12px sans-serif", /* … styling … */ }
}
```

| Field | Meaning |
|---|---|
| `model` | the glTF aircraft model + sizing (a **frontend** decision) |
| `position` | the track, as one flat array `[t, lon, lat, alt, t, lon, lat, alt, …]` relative to `epoch`; `LINEAR` interpolation, `HOLD` after the last sample |
| `orientation` | nose attitude as `[t, x, y, z, w, …]` unit quaternions, **derived** (see below) |
| `path` | the trail polyline (colour cycles through a 5-entry palette per flight) |
| `label` | the floating callsign text |

What Stage 3 changes from the CZML-input:

- **Waypoints → flat `cartographicDegrees`.** Each `[offset, lon, lat, alt]` is
  spread inline into one long array. The per-waypoint ordering is **identical** to
  Stage 2 (`offset, lon, lat, alt`) — Stage 3 just flattens it and attaches an
  `epoch`.
- **Orientation is computed, not carried.** Stage 2 has no heading/attitude in the
  waypoints. Stage 3 derives a heading and pitch from the **3-D velocity vector**
  between consecutive waypoints (great-circle bearing + climb angle), converts that to
  an ECEF unit quaternion, and emits one per sample so the model noses along its path.
- **Presentation is added** — model, trail colour, label, clock multiplier — none of
  which exist before this stage.

---

## Orientation in depth — deriving the attitude quaternion

The `orientation` row in the entity-packet table above is marked *"derived (see
below)"*. This is the "below": Stage 3 is the **only** place a flight gets an
**attitude** (which way the nose points), and it has to compute it from scratch.
The code lives in `aeroviz-4d/python/generate_czml.py`
(`compute_velocity_orientation` → `_hpr_to_ecef_quaternion` → `_mat3_to_quaternion`).

> **Interactive companion:** the same derivation with three driveable 3-D
> visualizations (drag the waypoints; orient an aircraft in the ENU box; fly it
> around the globe and watch every matrix + the quaternion update live) is in
> [`07-attitude-quaternion-derivation.en.html`](07-attitude-quaternion-derivation.en.html).

### What the field is, and why a quaternion

`position` answers *where* the aircraft is; `orientation` answers *which way it is
pointing*. CesiumJS renders everything in **ECEF** (Earth-Centered, Earth-Fixed:
origin at the Earth's centre, axes bolted to the rotating planet). So a model's
orientation is the **rotation that turns the model's own body axes into ECEF axes**,
and CZML stores that rotation as a **unit quaternion** `(x, y, z, w)`.

Why a quaternion and not three Euler angles (heading/pitch/roll)?

| Reason | Detail |
|---|---|
| No gimbal lock | Euler angles degenerate when pitch → ±90°; a quaternion never does |
| Smooth interpolation | Between two samples Cesium does **SLERP** (shortest-arc, constant-speed) — clean even through steep turns |
| Native CZML type | CZML's `unitQuaternion` is exactly this 4-number rotation |

Emitted shape — one sample per waypoint, epoch-relative, `LINEAR` (Cesium SLERPs
quaternion samples regardless of the label):

```
"orientation": { "epoch": …, "unitQuaternion": [t0,x0,y0,z0,w0,  t1,x1,y1,z1,w1, …] }
```

### The problem: the data carries no attitude

Stage 2 waypoints are `[offset, lon, lat, alt]` — **position and time only**. The raw
`track` heading from Stage 1 was *deliberately dropped at the seam* (it never entered
the waypoint). So Stage 3 must **reconstruct** the attitude from the path geometry
itself, on the physical assumption that **an aircraft noses along its own velocity
vector**.

That recovers two of the three attitude angles:

| Angle | Recoverable? | How |
|---|---|---|
| **heading** (yaw) | yes | direction of travel between consecutive waypoints |
| **pitch** | yes | climb/descent angle of that same segment |
| **roll** (bank) | **no** | no bank information in position data → **assumed 0** |

(Consequence of roll = 0: the model won't *visibly bank* into turns; it stays
wings-level and simply yaws. This is the one fidelity compromise of the derivation.)

The derivation is two steps: **(1)** velocity vector → heading + pitch, then
**(2)** heading + pitch at `(lon, lat)` → ECEF quaternion.

### Step 1 — heading & pitch from the velocity vector

`compute_velocity_orientation()` walks the waypoints. For each one it forms the
**segment to the next waypoint** (a forward difference). The very last waypoint has no
"next", so it reuses the *previous* segment (a backward difference) — that way the
parked model at the end of playback keeps its final heading instead of snapping to a
default.

For a segment from `(lon1, lat1, alt1)` to `(lon2, lat2, alt2)`:

```
            • next waypoint
           /┊
          / ┊  Δh = alt2 − alt1      (vertical rise, metres)
         /  ┊
        / θ ┊        pitch  θ = atan2(Δh, d)
  ─────•────┘
   this wp   d = haversine(this, next)   (horizontal run, metres)
```

- **heading** = the great-circle **bearing** of the segment, via geokit
  `bearing_rad(lat1, lon1, lat2, lon2)` — `0 = North`, increasing **clockwise**
  (East = 90°). This is the *plan-view* direction of travel.
- **horizontal run** `d` = geokit `haversine_m(...)` — ground distance, ignoring
  altitude.
- **pitch** `θ` = `atan2(alt2 − alt1, d)` — positive when climbing, negative when
  descending; `0` if the two points coincide horizontally.
- **roll** = `0` (see above).

The degenerate cases are explicit: an empty track → no samples; a single waypoint →
`(heading 0, pitch 0)` (nothing to take a difference against).

### Step 2 — heading & pitch → ECEF quaternion

The recovered `(heading, pitch)` are angles in the aircraft's **local horizon** — but
"local horizon" means something different at every point on the globe (the "up"
direction at Raleigh is not the "up" direction at London). So building the ECEF
quaternion is a composition of **two rotations through two nested frames**:

| Frame | Axes | Depends on |
|---|---|---|
| **Model / body** | `+X` = nose (forward), `+Y` = left, `+Z` = up | the model file |
| **ENU** (local tangent) | `X` = East, `Y` = North, `Z` = Up | the position `(lon, lat)` |
| **ECEF** (world) | fixed to the planet | nothing — it's the global frame |

**(a) model → ENU** (`_hpr_to_ecef_quaternion`, matrix `m_model`). Build the rotation
whose **columns are the model's basis vectors written in ENU coordinates**. With
`h = heading`, `θ = pitch`:

```
            nose (+X)      left (+Y)    up (+Z)
   East  [  sin h·cos θ     −cos h     −sin h·sin θ ]
  North  [  cos h·cos θ      sin h     −cos h·sin θ ]
     Up  [     sin θ           0           cos θ    ]
```

- **column 0 (nose)** is just the velocity direction as a unit vector: a vector at
  bearing `h`, tilted up by `θ` → `(sin h·cos θ, cos h·cos θ, sin θ)` in (E, N, Up).
- **column 1 (left wing)** is horizontal and 90° to port of the nose →
  `(−cos h, sin h, 0)`. (No `θ` term: roll = 0 keeps the wings in the horizontal
  plane.)
- **column 2 (body up)** is perpendicular to both, tilted back by the pitch →
  `(−sin h·sin θ, −cos h·sin θ, cos θ)`; at `θ = 0` it is exactly ENU "up".

**(b) ENU → ECEF** (matrix `m_enu`). The local ENU frame is itself tilted relative to
the planet's fixed axes by **where you stand on the globe**. At longitude `λ` and
latitude `φ`, the columns (East, North, Up expressed in ECEF) are the standard
geodetic tangent-frame rotation:

```
          East        North             Up
  X  [  −sin λ     −sin φ·cos λ      cos φ·cos λ ]
  Y  [   cos λ     −sin φ·sin λ      cos φ·sin λ ]
  Z  [     0          cos φ             sin φ    ]
```

**Compose** them — apply model→ENU first, then ENU→ECEF:

```
R_ecef = m_enu · m_model        (model body axes  →  ECEF axes)
```

**(c) matrix → quaternion** (`_mat3_to_quaternion`). The 3×3 rotation `R_ecef` is
converted to the unit quaternion `(x, y, z, w)` CZML wants, using **Shepperd's
method**: it picks the largest of `trace / R₀₀ / R₁₁ / R₂₂` as the pivot so it never
divides by a near-zero. (The naïve `w = ½√(1+trace)` formula blows up near a 180°
rotation; Shepperd's is numerically stable for any orientation.)

That `(x, y, z, w)` is what lands in the `unitQuaternion` array — one per waypoint, so
the model noses along its own reconstructed velocity vector for the whole flight.

### Why this is "computed, not carried"

The crosswalk table summarises it: heading existed as `track` back in **Stage 1** but
was dropped, and pitch/roll were never measured at all. Rather than thread an aviation
heading through the neutral interchange format, the pipeline keeps Stage 2 free of any
attitude and **re-derives the full 3-D nose direction from the geometry** at the only
stage that needs it — keeping the seam simple at the cost of the wings-level
approximation.

---

## The same flight, end to end

Stages 2 and 3 above are the **same real flight** (`AFR074`, landing 05L). Its first
waypoint is byte-identical across the seam — only the surrounding shape changes:

```
Stage 2 (waypoint):          [0, -78.454857, 35.739853, 2537.5]
Stage 3 (cartographicDegrees): … 0, -78.454857, 35.739853, 2537.5, …
```

And the field-by-field crosswalk of a single position sample:

| Concept | Stage 1 (raw row) | Stage 2 (waypoint) | Stage 3 (CZML) |
|---|---|---|---|
| time | `timestamp` (absolute ISO) | `offset_sec` from first point | `offset_sec` from `epoch` |
| longitude | `longitude` (deg) | element 1 of `[t,lon,lat,alt]` | 2nd of each 4-tuple in `cartographicDegrees` |
| latitude | `latitude` (deg) | element 2 | 3rd of each 4-tuple |
| altitude | `geoaltitude` (ft → m) | element 3 (geometric, m) | 4th of each 4-tuple (m) |
| heading | `track` (deg) | *(not stored)* | re-derived into `orientation` quaternion |
| identity | `icao24` / `callsign` | `id` / `icao24` / `callsign` | packet `id` / `name` |

---

## Where each format lives

| Format | File(s) | Reader |
|---|---|---|
| raw rows | (transient; `outputs/raw_tracks/…` JSONL in training mode) | `trajectory.build_trajectories_from_history` |
| CZML-input | `outputs/<airport>/<airport>_czml_input_*.json`, `outputs/landings/<A>/<A>_<RWY>_landings.json` | `generate_czml.py --input …` |
| CZML | `aeroviz-4d/public/data/airports/<ICAO>/trajectories.czml` | the CesiumJS frontend |

For *how* the files are produced and every CLI flag, see the package
[README](../README.md).
