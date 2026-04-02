# Tutorial 04 — CZML Format & 4D Trajectory Playback

**Covers:** `src/hooks/useCzmlLoader.ts`, `src/utils/czmlBuilder.ts`,
`python/generate_czml.py`

---

## What you will implement

- A TypeScript hook that loads a CZML file and synchronises the Cesium clock
- Pure TypeScript utilities that build valid CZML JSON objects
- A Python script that serialises scheduling algorithm output to CZML

---

## Concept 1 — CZML structure

CZML is a **JSON array** where:
- Element `[0]` is always the **document packet** (global clock settings)
- Elements `[1..N]` are **entity packets** (one per aircraft, waypoint, etc.)

```
[
  { "id": "document", "clock": { ... } },   ← always first
  { "id": "UAL123", "position": { ... } },  ← one entity per aircraft
  { "id": "WJA456", "position": { ... } },
]
```

---

## Concept 2 — The `cartographicDegrees` position array

The most important field in an entity packet is `position.cartographicDegrees`.
It is a **flat array** of groups of 4 numbers:

```
[t0, lon0, lat0, alt0,  t1, lon1, lat1, alt1,  ...]
```

- `t`   = seconds since `epoch` (NOT a unix timestamp)
- `lon` = longitude in decimal degrees
- `lat` = latitude in decimal degrees
- `alt` = altitude in **metres** above the WGS84 ellipsoid (MSL)

Example: an aircraft at three positions over 4 minutes:
```json
{
  "epoch": "2026-04-01T08:00:00Z",
  "cartographicDegrees": [
    0,   -119.38, 49.95, 4500,
    120, -119.40, 49.90, 3800,
    240, -119.42, 49.85, 3200
  ]
}
```

Cesium interpolates between samples.  The `interpolationAlgorithm: "LAGRANGE"`
with `interpolationDegree: 3` gives smooth curves through the waypoints.

---

## Concept 3 — The Cesium Clock and JulianDate

CesiumJS uses `JulianDate` (Julian Day Number) as its internal time type.
It is NOT a JavaScript `Date` object.

Key rules:
- Always use `Cesium.JulianDate` methods for time arithmetic
- **Always `.clone()` a JulianDate before assigning it to a new variable**,
  otherwise both variables point to the same mutable object and you get
  confusing bugs where changing `currentTime` silently corrupts `startTime`

```typescript
// ✅ Correct:
viewer.clock.startTime = ds.clock.startTime.clone();
viewer.clock.currentTime = ds.clock.startTime.clone();

// ❌ Wrong (aliasing bug):
viewer.clock.startTime = ds.clock.startTime;
viewer.clock.currentTime = ds.clock.startTime; // same object!
```

---

## Concept 4 — `viewer.trackedEntity`

When you set `viewer.trackedEntity = someEntity`, the camera automatically
follows that entity.  The camera stays behind and above the entity as it moves,
giving a "chase-plane" view.

```typescript
viewer.trackedEntity = ds.entities.getById("UAL123") ?? undefined;
```

Set it to `undefined` to return to free-roam mode.

---

## Concept 5 — `velocityReference` for orientation

If you set orientation to:
```json
{ "velocityReference": "#UAL123.position" }
```

Cesium automatically computes the heading and pitch from the velocity vector
of the entity's position.  This means the aircraft 3D model always points in
the direction it is flying — you get this for free without computing any angles.

---

## Your TODOs in `useCzmlLoader.ts`

Work through the TODOs in order (① → ⑥):

| TODO | What to implement |
|------|------------------|
| ① | `Cesium.CzmlDataSource.load(czmlUrl)` + `.catch()` error handler |
| ② | Store DataSource + add to viewer |
| ③ | Synchronise `viewer.clock` from `ds.clock` (6 lines) |
| ④ | Collect flight IDs (filter out "document") |
| ⑤ | Set `viewer.trackedEntity` to the first flight |
| ⑥ | Call `setState(...)` with results |

---

## Your TODOs in `czmlBuilder.ts`

| TODO | Function | What to implement |
|------|---------|------------------|
| ① | `buildSampledPosition` | Build the flat `cartographicDegrees` array |
| ② | `buildFlightPacket` | Assemble all packet fields |
| ③ | `buildCzml` | Compose document + entity packets |

---

## Testing workflow (recommended order)

### Phase 1 — Test `czmlBuilder.ts` in isolation
```bash
npm test -- czmlBuilder
```
No Cesium, no browser needed.  Fix the TODO assertions and implementations together.

### Phase 2 — Generate mock CZML from Python
```bash
cd python
python generate_czml.py
# Should create aeroviz-4d/public/data/trajectories.czml
```

### Phase 3 — Load in browser
```bash
cd aeroviz-4d
npm run dev
```
Open `http://localhost:5173`.  If the CZML loads correctly, you should see:
- Aircraft models at the starting positions
- The timeline bar spans the trajectory duration
- Pressing ▶ animates the aircraft along their paths

### Phase 4 — Replace mock with your algorithm output
When your scheduling algorithm is ready:
```python
from generate_czml import build_czml
czml = build_czml(your_algorithm_output, start_dt)
Path("../aeroviz-4d/public/data/trajectories.czml").write_text(
    json.dumps(czml, indent=2)
)
```
Then refresh the browser — the 3D visualisation updates automatically.

---

## Checklist

- [ ] `czmlBuilder` unit tests pass
- [ ] `python generate_czml.py` writes a valid JSON file (check with `python -c "import json; json.load(open('public/data/trajectories.czml'))"`)
- [ ] Aircraft appear on the globe after loading
- [ ] Timeline spans the correct duration
- [ ] Pressing ▶ moves the aircraft
- [ ] Clicking a flight in FlightTable tracks it with the camera
