# Tutorial 03 — OCS Geometry (Geodesy Math)

**Covers:** `src/utils/ocsGeometry.ts` and its unit tests

---

## What you will implement

The three pure functions that compute the 3D polygon corners of the PANS-OPS
Obstacle Clearance Surface (OCS):

1. `bearingRad(lonA, latA, lonB, latB)` — compass bearing A→B
2. `offsetPoint(lon, lat, altM, bearing, distance)` — move a point N metres
3. `buildFinalApproachOCS(params)` — assemble the three OCS polygons

**These are the most math-intensive functions in the project.**
Complete the unit tests in `src/utils/__tests__/ocsGeometry.test.ts` alongside
the implementation — verify each function independently before composing them.

---

## Background — PANS-OPS Protection Areas

ICAO Document 8168 (PANS-OPS) defines the airspace protection geometry for
instrument approach procedures.  For the final approach segment:

```
                    FAF
                     │  ← centreline
        ─────────────┼──────────────
        secondary    │ primary  secondary
        (7:1 slope)  │          (7:1 slope)
        ─────────────┼──────────────
                     │
                  Threshold
```

- **Primary protection area**: a rectangle (or slight trapezoid) centred on
  the approach centreline.  Full obstacle clearance is guaranteed inside.
- **Secondary protection area**: flanks outside the primary.  Clearance
  diminishes linearly from 100% at the inner edge to 0% at the outer edge.
  The outer boundary sits at a 7:1 gradient below the primary boundary.

**7:1 means**: for every 7 metres you move outward (horizontally), the
protection surface drops 1 metre in altitude.

---

## Concept 1 — Flat-Earth approximation

For distances under ~20 km (the length of any approach segment), the Earth's
curvature is negligible.  We use a simple Cartesian approximation:

```
Δx (metres) = Δlon (degrees) × metres_per_deg_lon(lat)
Δy (metres) = Δlat (degrees) × 111320

where metres_per_deg_lon(lat) = 111320 × cos(lat_radians)
```

This is accurate to about 0.1% at these scales.

**Why does metres_per_deg_lon depend on latitude?**
Lines of longitude converge toward the poles.  At the equator (lat=0°),
1° of longitude ≈ 111 km.  At 60° N latitude, the same 1° of longitude
spans only ≈ 55 km (cos(60°) = 0.5).

---

## Concept 2 — Forward azimuth (bearing)

The bearing FROM point A TO point B in the flat-Earth model:

```
Δx = (lonB − lonA) × metres_per_deg_lon(latA)   ← east component
Δy = (latB − latA) × 111320                      ← north component
bearing = atan2(Δx, Δy)
```

Note the argument order: `atan2(east, north)`, **not** `atan2(y, x)`.
This gives 0 = north, π/2 = east, matching compass conventions.

```
        North (Δy > 0)
             ↑
West ←───────┼────────→ East (Δx > 0)
             │
        South (Δy < 0)

atan2(0, 1)  = 0       → north
atan2(1, 0)  = π/2     → east
atan2(0, -1) = -π or π → south
atan2(-1, 0) = -π/2    → west
```

---

## Concept 3 — Perpendicular offsets

To build the OCS polygon, you need to move points perpendicular to the
centreline bearing.  If the centreline bearing is `β`:
- Left perpendicular:  `β − π/2`
- Right perpendicular: `β + π/2`

Then use `offsetPoint()` to shift each centreline end by `primaryHalfWidthM`
in each perpendicular direction.

---

## Concept 4 — The 7:1 slope altitude formula

For the outer edge of the secondary protection area at the FAF:

```
outer_alt_at_faf = faf.altM − secondaryWidthM / 7
```

At the threshold end, the outer altitude equals `threshold.altM` (the
OCS slope "lands" exactly at the threshold elevation — it does not dip below).

This creates a slanted surface: the outer edge is lower at the FAF end
and meets the inner (primary) edge at the threshold elevation.

---

## Step-by-step implementation guide

### Step 1 — Implement `bearingRad`

```typescript
export function bearingRad(lonA, latA, lonB, latB): number {
  const dx = (lonB - lonA) * metresPerDegLon(latA);
  const dy = (latB - latA) * METRES_PER_DEG_LAT;
  return Math.atan2(dx, dy);
}
```

Run the tests: `npm test -- ocsGeometry`
Expected: `bearingRad(-119.38, 49.90, -119.38, 49.95)` → ≈ 0 (north)

### Step 2 — Implement `offsetPoint`

```typescript
export function offsetPoint(lon, lat, altM, bearingRad, distanceM): GeoPoint3D {
  return {
    lon: lon + (distanceM * Math.sin(bearingRad)) / metresPerDegLon(lat),
    lat: lat + (distanceM * Math.cos(bearingRad)) / METRES_PER_DEG_LAT,
    altM,  // altitude unchanged — only horizontal movement
  };
}
```

### Step 3 — Implement `buildFinalApproachOCS`

Pseudocode:
```
bearing    ← bearingRad(faf → threshold)
perpLeft   ← bearing - π/2
perpRight  ← bearing + π/2

// PRIMARY corners (all at FAF/threshold altitude)
fafLeft    ← offsetPoint(faf,       perpLeft,  primaryHalfWidthM)
fafRight   ← offsetPoint(faf,       perpRight, primaryHalfWidthM)
thrLeft    ← offsetPoint(threshold, perpLeft,  primaryHalfWidthM)
thrRight   ← offsetPoint(threshold, perpRight, primaryHalfWidthM)

// SECONDARY outer corners (altitude drops by 1/7 of secondary width at FAF)
secFafLeft    ← offsetPoint(faf,       perpLeft,  prim+sec, alt = faf.altM − sec/7)
secFafRight   ← offsetPoint(faf,       perpRight, prim+sec, alt = faf.altM − sec/7)
secThrLeft    ← offsetPoint(threshold, perpLeft,  prim+sec, alt = threshold.altM)
secThrRight   ← offsetPoint(threshold, perpRight, prim+sec, alt = threshold.altM)

return {
  primaryPolygon: [fafLeft, fafRight, thrRight, thrLeft],
  secondaryLeft:  [fafLeft, secFafLeft, secThrLeft, thrLeft],
  secondaryRight: [fafRight, secFafRight, secThrRight, thrRight],
}
```

---

## Testing strategy

1. **Test `bearingRad` first** — it has simple exact answers (0, π/2, −π/2, −π).
2. **Test `offsetPoint` second** — verify that a 1000 m east offset moves
   longitude by exactly `1000 / metresPerDegLon(lat)` degrees.
3. **Test `buildFinalApproachOCS` last** — a north–south approach lets you
   predict all corner longitudes analytically.

---

## Checklist

- [ ] `bearingRad` tests pass
- [ ] `offsetPoint` tests pass
- [ ] `buildFinalApproachOCS` returns non-empty polygons
- [ ] Red semi-transparent surfaces appear over the terrain in the browser
- [ ] The 7:1 slope is visible: the outer edge at FAF is lower than the inner edge
