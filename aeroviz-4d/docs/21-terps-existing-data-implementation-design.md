# TERPS Existing-Data Implementation Design

## Status

Date: 2026-05-04

This document defines the implementable TERPS/PBN model that AeroViz can build
from the data already present in the repository. It is a staged engineering
design, not a statement that every generated object is a certified TERPS
construction.

## Scope

The implementation is organized into three layers:

1. Obstacle clearance assessment.
2. Protected-area geometry.
3. Vertical path and surface construction.

Each generated object must keep an explicit confidence/status value:

- `SOURCE_BACKED`: source geometry or rule parameters are present.
- `TERPS_ESTIMATE`: source path data is available, but the construction is a
  simplified TERPS/PBN rule implementation.
- `DISPLAY_ESTIMATE`: continuity or missing-source geometry generated for
  visualization/assessment context.
- `DEBUG_ESTIMATE`: debugging aid only.
- `MISSING_SOURCE`: expected construction cannot be built from current data.

## Existing Inputs

Procedure data:

- Segment centerlines from IF/TF/DF/RF geometry.
- Leg metadata: `legType`, start/end fixes, course, turn direction, required
  altitude and speed.
- Segment metadata: `xttNm`, `attNm`, `secondaryEnabled`,
  `widthChangeMode`, `transitionRule`, and `verticalRule`.
- Fix roles and elevations for IAF/IF/FAF/MAP/MAHF/RWY.

Obstacle/terrain data:

- `public/data/airports/<ICAO>/obstacles.geojson`, with obstacle top height as
  `amsl_ft` / `amsl_m` and obstacle type.
- Runway threshold elevations in `runway.geojson`.
- DSM/heightmap terrain where available. DSM sampling can be added later; the
  first obstacle-clearance implementation uses known obstacle top elevations.

## Layer 1 - Obstacle Clearance

Goal: determine how an obstacle relates to a protected surface.

For each obstacle and each `ProcedureProtectionSurface`:

1. Project obstacle lon/lat to the surface centerline.
2. Interpolate station along the surface.
3. Interpolate lateral width samples:
   - primary half-width;
   - secondary outer half-width.
4. Classify lateral containment:
   - `PRIMARY`;
   - `SECONDARY`;
   - `OUTSIDE`.
5. If the surface has `vertical.kind === "OCS"`, interpolate the OCS altitude
   at the same station and compute:

```text
clearanceFt = surfaceAltitudeFtMsl - obstacleTopFtMsl
```

6. Mark `clearanceFt < 0` as OCS penetration.

Important limits:

- `FINAL_LNAV_OEA` is lateral OEA only. It can identify included obstacles, but
  it cannot by itself produce vertical clearance.
- Secondary-area reduced-clearance rules are not fully modeled yet. Secondary
  OCS results are reported as raw OCS comparisons with an explicit secondary
  status.
- Display/debug surfaces are excluded from operational obstacle assessment by
  default.

## Layer 2 - Protected-Area Geometry

Goal: make lateral protected geometry rule-driven instead of only constant
width offsets.

Implemented rule modes:

- `NONE`: constant width from current segment tolerance model.
- `ABRUPT`: constant width until explicit transition rules are modeled.
- `LINEAR_TAPER`: variable-width straight envelope.
  - Primary starts at `1 * XTT`.
  - Primary stabilizes at `2 * XTT`.
  - Secondary outer starts at `2 * XTT`.
  - Secondary outer stabilizes at `3 * XTT`.
  - Taper end station uses `transitionRule.afterNm` when present; otherwise it
    uses the segment length.

RF/arc caution:

- RF envelopes remain `RF_PARALLEL_ARC` for now.
- Variable-width RF taper is not substituted with straight offsets because that
  would distort the arc protected area.

Future rule modes:

- `SPLAY_30` should be implemented only when the associated TERPS/PBN segment
  rule and source parameters are modeled.
- Missed-section XTT-derived widths should be replaced by chapter-specific
  missed approach width tables.
- Turning missed TIA/wind spiral debug objects should remain debug until wind,
  speed, bank/turn-radius, and turn-initiation tolerances are modeled.

## Layer 3 - Vertical Path

Goal: separate display altitude profiles from obstacle clearance surfaces.

Supported from current data:

- `GPA` and `TCH` produce an LNAV/VNAV OCS estimate when source values exist.
- `MDA` / `DA` are minima markers; they do not define a sloping OCS.
- Missed climb gradient can produce a missed OCS-like climb surface.
- Missing climb gradient falls back to centerline altitude/profile data and is
  not treated as source-backed vertical clearance.

Vertical kinds:

- `NONE`: lateral-only OEA/protected area.
- `ALTITUDE_PROFILE`: display or continuity profile, not obstacle clearance.
- `OCS`: vertical surface usable for obstacle clearance assessment.

## Implementation Plan

1. Document this three-layer model.
2. Make `widthChangeMode: "LINEAR_TAPER"` affect segment envelopes, not only
   final OEA surfaces.
3. Add obstacle clearance assessment utilities using
   `ProcedureProtectionSurface`.
4. Add tests covering:
   - variable-width segment envelopes;
   - lateral-only OEA obstacle inclusion;
   - OCS clearance and penetration;
   - secondary raw OCS reporting;
   - debug surfaces excluded by default.
5. Keep rendering unchanged until the data-layer assessment is validated.

## Acceptance Criteria

- A final segment with `LINEAR_TAPER` exposes increasing primary/secondary
  width samples in `segmentGeometry`.
- Final OEA/OCS surfaces continue to expose their existing protection-surface
  metadata.
- Obstacles can be assessed against the same `ProcedureProtectionSurface`
  objects used by 3D rendering and runway-profile assessment.
- OEA-only results do not claim vertical clearance.
- OCS penetration is reported when obstacle top altitude is above the surface.
