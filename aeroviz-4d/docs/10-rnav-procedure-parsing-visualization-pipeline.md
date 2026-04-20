# RNAV Procedure Parsing and Visualization Pipeline

## Purpose

This document explains the current AeroViz-4D procedure-layer pipeline end to end:

1. Read local FAA CIFP cycle data.
2. Extract selected KRDU RNAV/GPS procedure branches.
3. Resolve CIFP fixed-width records into route fixes and vertical samples.
4. Write deterministic browser-ready `procedures.geojson`.
5. Load that GeoJSON in the React/Cesium frontend.
6. Render plan-view procedure paths, fix labels, and an approximate 3D/4D tunnel.

The implementation is for research visualization only. It is not certified navigation data, not a TERPS/PANS-OPS containment model, and not an FAA-approved procedure depiction.

## Main Files

- `aeroviz-4d/python/preprocess_procedures.py`
  - Python preprocessing entrypoint.
  - Reads CIFP files and writes `aeroviz-4d/public/data/procedures.geojson`.
- `data/CIFP/CIFP_260319/IN_CIFP.txt`
  - Local CIFP procedure index.
  - Used to verify that an airport/procedure exists before parsing detail records.
- `data/CIFP/CIFP_260319/FAACIFP18`
  - Local fixed-width CIFP detail file.
  - Used for fixes, runway threshold coordinates, procedure legs, and source cycle.
  - Contains profile records that future work should parse for official vertical-path support.
- `aeroviz-4d/src/types/geojson-aviation.d.ts`
  - TypeScript schema for generated procedure GeoJSON.
- `aeroviz-4d/src/utils/procedureGeometry.ts`
  - Pure geometry helpers for tunnel sampling and cross-section generation.
- `aeroviz-4d/src/hooks/useProcedureLayer.ts`
  - Cesium loader and renderer for procedure routes, fixes, and tunnel geometry.
- `aeroviz-4d/src/components/ProcedurePanel.tsx`
  - UI controls for runway/procedure/branch visibility.
- `aeroviz-4d/src/context/AppContext.tsx`
  - Global layer visibility and per-route procedure visibility state.

## Current Generation Command

For the current KRDU multi-runway RNAV/GPS layer:

```bash
python aeroviz-4d/python/preprocess_procedures.py \
  --cifp-root data/CIFP/CIFP_260319 \
  --airport KRDU \
  --procedure-type SIAP \
  --include-all-rnav \
  --include-transitions \
  --output aeroviz-4d/public/data/procedures.geojson
```

The default `--include-all-rnav` set is:

```text
R05LY, R05RY, R23LY, R23RY, R32
```

These map to the current runway groups:

```text
RW05L, RW05R, RW23L, RW23R, RW32
```

## Pipeline Overview

```text
data/CIFP/CIFP_260319/
  IN_CIFP.txt
  FAACIFP18
      |
      v
preprocess_procedures.py
  1. validate procedure exists
  2. build airport fix index
  3. parse available branches
  4. parse procedure leg records
  5. resolve IF/TF legs into 3D route points
  6. attach warnings for unsupported/unresolved data
  7. emit deterministic GeoJSON
      |
      v
aeroviz-4d/public/data/procedures.geojson
      |
      v
Frontend
  ProcedurePanel fetches route metadata for controls
  useProcedureLayer fetches geometry for Cesium rendering
      |
      v
Cesium scene
  plan-view polyline
  fix points and labels
  approximate 3D tunnel
```

## CIFP Input Model

### `IN_CIFP.txt`

`IN_CIFP.txt` is treated as the airport/procedure index. The parser does not infer procedure availability from the detail file alone. It first checks whether the requested airport, procedure type, and procedure ident are listed.

For KRDU RNAV/GPS procedures, relevant index rows include:

```text
KRDU SIAP R05LY
KRDU SIAP R05RY
KRDU SIAP R23LY
KRDU SIAP R23RY
KRDU SIAP R32
```

Implementation:

- Function: `procedure_exists(index_path, airport, procedure_type, procedure)`
- Matching is case-insensitive.
- Rows are split by whitespace.
- The first three fields must match:
  - airport, for example `KRDU`
  - procedure type, for example `SIAP`
  - procedure ident, for example `R05LY`

If a selected procedure is missing from `IN_CIFP.txt`, preprocessing fails early with a clear error.

### `FAACIFP18`

`FAACIFP18` contains fixed-width CIFP records. The current parser uses three kinds of records:

1. Header records
   - Used to infer source cycle, currently `2603`.
2. Airport coordinate records
   - Used to build a local fix/runway coordinate index for one airport.
3. Final/intermediate procedure leg records
   - Used to extract branch, sequence, fix ident, path terminator, role hints, and altitude constraints.

Profile records are present in the file, but the current pipeline only preserves their future role in the design. It does not yet parse `P` profile records into the vertical path.

The fixed-width slices listed below are the fields currently used by the local parser. They should be treated as an implementation contract for this codebase, not as a complete ARINC 424 record specification.

## CIFP Parsing Details

### Source Cycle

The source cycle is parsed from early header lines in `FAACIFP18`.

Implementation:

- Function: `parse_source_cycle(faacifp_path)`
- Looks for:
  - `VOLUME <cycle>`
  - or a `FAACIFP18` header pattern containing the cycle.
- Stops scanning once the file leaves the header section.

The generated GeoJSON stores the result in:

- collection metadata: `metadata.sourceCycle`
- each feature property: `sourceCycle`

### Coordinate Decoding

CIFP compact coordinates are encoded as hemisphere plus degrees, minutes, and seconds digits.

Examples covered by tests:

```text
N35483156       -> 35.8087667
W078525864      -> -78.8829556
N3552280160     -> 35.8744489
W07848070690    -> -78.8019630
```

Implementation:

- Function: `decode_cifp_coordinate(token)`
- Latitude tokens use two degree digits.
- Longitude tokens use three degree digits.
- Remaining digits are parsed as minutes plus seconds with decimal precision inferred from token length.
- South and west hemispheres are negated.

Coordinate pairs are decoded with:

```text
decode_coordinate_pair(lat_token, lon_token) -> (lon, lat)
```

GeoJSON uses standard coordinate order:

```json
[lon, lat, altitudeMeters]
```

### Building the Fix Index

The parser builds a local coordinate lookup before parsing procedure legs.

Implementation:

- Function: `build_fix_index(faacifp_path, airport)`
- Scans lines beginning with the airport prefix:

```text
SUSAP KRDU
```

- Searches each matching line for the first compact latitude/longitude pair using:

```text
([NS]\d{8,10})([EW]\d{9,11})
```

- Extracts fix/runway ident from:

```text
line[13:19]
```

- Extracts elevation or altitude after the coordinate pair when available.
- Stores a `FixRecord`:

```text
ident
lon
lat
altitude_ft
source_line
```

This fix index includes both named terminal fixes and runway-like records such as `RW05L`.

### Procedure Ident Interpretation

Procedure idents are classified with simple display-oriented rules:

- `R...` -> `RNAV_GPS`
- `H...` -> `RNAV_RNP`
- `I...` -> `ILS`
- `L...` -> `LOC`
- otherwise -> `UNKNOWN`

For KRDU RNAV/GPS idents:

- `R05LY`
  - family: `RNAV_GPS`
  - runway inferred from ident: `RW05L`
  - variant: `Y`
- `R32`
  - family: `RNAV_GPS`
  - runway inferred from ident: `RW32`
  - variant: `null`

The main final branch is inferred by procedure family:

- RNAV/GPS: `R`
- RNAV/RNP: `H`
- ILS: `I`
- LOC: `L`

All other branches are currently classified as `transition`.

### Available Branches

When `--include-transitions` is used, the parser reads every branch ident available for the procedure.

Implementation:

- Function: `parse_available_branches(faacifp_path, airport, procedure)`
- Scans airport procedure-leg records.
- Keeps branches in deterministic order:
  - final branch first
  - transition branches after

For example, `KRDU R05LY` currently produces:

```text
R, ACHWDR, AOTTOS
```

The final branch is visible by default. Transition branches are generated but hidden by default.

### Procedure Leg Records

The current parser reads fixed-width final/intermediate procedure records from `FAACIFP18`.

Implementation:

- Function: `parse_procedure_legs(faacifp_path, airport, procedure, branch)`
- The line must:
  - start with `SUSAP <airport>`
  - be long enough to contain procedure-leg fields
  - have `line[12] == "F"`
  - match `line[13:19]` as procedure ident
  - match `line[19:26]` as branch ident

Important fixed-width slices currently used:

```text
line[13:19]  procedure ident, e.g. R05LY
line[19:26]  branch ident, e.g. R or ACHWDR
line[26:29]  sequence number, e.g. 010
line[29:34]  fix ident, e.g. SCHOO
line[40:70]  path terminator search area, e.g. IF or TF
line[42]     waypoint-description role hint
line[70:90]  altitude search area
```

Path terminators are extracted with a conservative regex over `line[40:70]`:

```text
IF, TF, DF, CA, CF, HM, HF, RF, VI, VA, FA
```

The current renderer only supports:

```text
IF, TF
```

Unsupported path terminators are not silently lost. They are listed in route warnings and `legCoverage.skippedLegTypes`.

### Role Classification

Each parsed leg gets a display role. This role controls labels and point styling in Cesium.

Implementation:

- Function: `parse_leg_role(line, leg_type, fix_ident, sequence)`

Current rules:

- runway-like fix ident starting with `RW` -> `MAPt`
- waypoint-description character `F` -> `FAF`
- waypoint-description character `I` -> `IF`
- leg type `IF` -> `IF`
- sequence number less than or equal to 20 -> `IF`
- otherwise -> `Route`

This is a practical display heuristic, not a full ARINC 424 semantic interpretation.

### Altitude Parsing

Altitude constraints are parsed with:

```text
parse_signed_altitude_ft(text)
```

Current behavior:

- Finds the first five-digit altitude in a text slice.
- Supports optional `+`, `-`, or `V` prefix.
- Negative prefix produces a negative altitude.
- Missing altitude returns `None`.

When building route geometry:

1. Use leg altitude if present.
2. If the fix is runway-like and the fix record has elevation, use runway/fix elevation.
3. Otherwise use fix altitude if present.
4. If no altitude is available, use `0 ft` and add a warning.

The route's 3D coordinate altitude is stored as meters:

```text
geometry_altitude_ft * 0.3048
```

## Route Point Generation

Procedure legs are converted into ordered `RoutePoint` samples by:

```text
build_route_points(legs, fixes, nominal_speed_kt)
```

For each leg:

1. Skip unsupported path terminators with a warning.
2. Skip missing or unresolved fix idents with a warning.
3. Resolve the fix ident through the local fix index.
4. Compute great-circle distance from the previous resolved fix.
5. Accumulate distance from route start.
6. Convert nominal speed to meters per second.
7. Accumulate `timeSeconds` for simple 4D gates.
8. Resolve geometry altitude.
9. Append a `RoutePoint`.

The nominal speed default is:

```text
140 kt
```

Current generated sample fields:

```text
sequence
fixIdent
legType
role
altitudeFt
geometryAltitudeFt
distanceFromStartM
timeSeconds
sourceLine
```

If fewer than two supported points are resolved, the branch is marked with a warning. During multi-procedure generation, such a branch is omitted from drawable features but the warning is preserved in collection metadata.

## GeoJSON Output

The generated file is:

```text
aeroviz-4d/public/data/procedures.geojson
```

It is intentionally deterministic:

- stable procedure ordering
- stable branch ordering
- rounded coordinates and distances
- `generatedAt: null`
- JSON emitted with `sort_keys=True`

This makes diffs reviewable and prevents the file from changing just because preprocessing was rerun.

### Collection Metadata

Top-level metadata includes:

```json
{
  "airport": "KRDU",
  "procedureType": "SIAP",
  "procedureFamilies": ["RNAV_GPS"],
  "procedureIdents": ["R05LY", "R05RY", "R23LY", "R23RY", "R32"],
  "runwayIdents": ["RW05L", "RW05R", "RW23L", "RW23R", "RW32"],
  "sourceCycle": "2603",
  "generatedAt": null,
  "researchUseOnly": true,
  "warnings": []
}
```

Warnings are collected at the collection level as human-readable strings.

### Route Features

Each drawable branch creates one `LineString` feature.

Geometry:

```json
{
  "type": "LineString",
  "coordinates": [
    [-78.92647222, 35.77341389, 914.4],
    [-78.88295556, 35.80876667, 670.56],
    [-78.80196389, 35.87445, 243.23]
  ]
}
```

Important properties:

```text
featureType: procedure-route
routeId: KRDU-R05LY-R
airport: KRDU
procedureType: SIAP
procedureIdent: R05LY
procedureName: RNAV(GPS) Y RW05L
procedureFamily: RNAV_GPS
procedureVariant: Y
runwayIdent: RW05L
branchIdent: R
branchType: final
defaultVisible: true
source: FAA-CIFP
sourceCycle: 2603
researchUseOnly: true
nominalSpeedKt: 140
samples: [...]
warnings: [...]
```

The tunnel display settings are embedded in each route:

```json
{
  "lateralHalfWidthNm": 0.3,
  "verticalHalfHeightFt": 300.0,
  "sampleSpacingM": 250.0,
  "mode": "visualApproximation"
}
```

The leg coverage summary is also embedded:

```json
{
  "parsedLegTypes": ["CA", "DF", "HM", "IF", "TF"],
  "renderedLegTypes": ["IF", "TF"],
  "skippedLegTypes": ["CA", "DF", "HM"],
  "simplifiedLegTypes": []
}
```

### Fix Features

Each resolved route point also creates a `Point` feature.

Important properties:

```text
featureType: procedure-fix
routeId: KRDU-R05LY-R
name: SCHOO
sequence: 10
legType: IF
role: IF
altitudeFt: 3000
geometryAltitudeFt: 3000
distanceFromStartM: 0
timeSeconds: 0
sourceLine: 300787
```

Fix features allow Cesium to render labels and point symbols independently from the route polyline.

## Frontend Loading

The frontend reads the generated static file from:

```text
/data/procedures.geojson
```

Two separate frontend paths read the same file:

1. `ProcedurePanel`
   - reads route properties for UI grouping and toggles.
2. `useProcedureLayer`
   - reads geometry for Cesium rendering.

This avoids coupling UI grouping logic to Cesium entity creation.

## Procedure Panel

`ProcedurePanel.tsx` provides user-facing controls for procedure visibility.

Data flow:

```text
fetch /data/procedures.geojson
  -> filter featureType == procedure-route
  -> convert route properties into ProcedureRouteItem
  -> group by runway
  -> group by procedure
  -> list branches under each procedure
```

Grouping order:

```text
RW05L, RW05R, RW23L, RW23R, RW32
```

Branch order:

```text
final branch first, transitions after
```

Visibility behavior:

- `layers.procedures` is the master layer toggle.
- `procedureVisibility[routeId]` stores per-route overrides.
- If no override exists, `defaultVisible` from GeoJSON is used.
- Final branches default to visible.
- Transition branches default to hidden.

This means toggling procedures does not reload the GeoJSON. It only updates Cesium entity `show` flags.

## Cesium Rendering

`useProcedureLayer.ts` is responsible for converting GeoJSON features into Cesium entities.

### Route Polylines

Each `procedure-route` feature creates one Cesium polyline:

```text
id: procedure-<routeId>-line
positions: Cartesian3.fromDegreesArrayHeights(...)
width: 5
material: cyan
```

This is the plan-view route path, but because coordinates include altitude it is rendered in 3D.

### Fix Points and Labels

Each `procedure-fix` feature creates one Cesium point and label:

```text
id: procedure-<routeId>-fix-<sequence>-<index>
position: Cartesian3.fromDegrees(lon, lat, altM)
point: yellow or orange
label: fix name plus role
```

Current styling:

- `FAF` fixes are orange.
- `MAPt` points are slightly larger.
- Labels use monospace text and distance display limiting.

### Tunnel Geometry

For each route feature, the hook reads:

```text
geometry.coordinates
properties.tunnel
properties.nominalSpeedKt
```

It then calls:

```text
buildTunnelSections(routePoints, options)
```

The output is a list of cross-sections along the route. Consecutive sections are connected with four polygon strips:

```text
left wall
right wall
top
bottom
```

Each quad is a Cesium polygon with:

```text
perPositionHeight: true
material: translucent blue
outline: cyan
```

Entity IDs are grouped under the route ID so branch toggles can show/hide the polyline, fixes, and all tunnel quads together.

## Tunnel Geometry Algorithm

`procedureGeometry.ts` is intentionally independent from Cesium. It accepts simple 3D geographic points:

```ts
interface ProcedurePoint3D {
  lon: number;
  lat: number;
  altM: number;
}
```

### Densification

The route is densified before tunnel generation:

```text
sample spacing default: 250 m
```

For each segment between two route fixes:

1. Compute great-circle distance.
2. Insert interior interpolated points at the requested spacing.
3. Preserve every original endpoint.
4. Track cumulative distance from start.

### Local Bearing

Each sample receives a local bearing:

- first sample: bearing to next sample
- last sample: bearing from previous sample
- middle sample: bearing from previous sample to next sample

The bearing is only used to define a local left/right cross-section direction.

### Cross-Section

Default tunnel dimensions:

```text
lateral half-width: 0.3 NM
vertical half-height: 300 ft
nominal speed: 140 kt
```

For each sample:

1. Offset left by half-width.
2. Offset right by half-width.
3. Add vertical top and bottom points around the center altitude.
4. Store cumulative distance.
5. Store simple time gate:

```text
timeSeconds = distanceFromStartM / speedMps
```

The result is:

```ts
interface TunnelSection {
  center: ProcedurePoint3D;
  leftBottom: ProcedurePoint3D;
  leftTop: ProcedurePoint3D;
  rightBottom: ProcedurePoint3D;
  rightTop: ProcedurePoint3D;
  distanceFromStartM: number;
  timeSeconds: number;
}
```

This produces a visually useful corridor, but it is still a simplified display tunnel.

## Layer State and Cleanup

The global context stores:

```text
layers.procedures
procedureVisibility
setProcedureRouteVisible(routeId, visible)
setProcedureRoutesVisible(routeIds, visible)
```

`useProcedureLayer` keeps internal maps:

```text
routeEntityIdsRef: routeId -> entity ids
routeDefaultsRef: routeId -> defaultVisible
```

When visibility changes:

1. Iterate route IDs.
2. Resolve each Cesium entity by ID.
3. Set:

```text
entity.show = layers.procedures && routeVisible
```

When the hook unmounts or reloads:

1. Iterate all entity IDs added by the hook.
2. Remove only those entities from `viewer.entities`.
3. Leave unrelated layers untouched.

## Current KRDU Output Shape

With `--include-all-rnav --include-transitions`, the current generated file contains:

```text
17 drawable route branches
57 fix points
5 runway groups
22 warnings
```

Final branches:

```text
KRDU-R05LY-R
KRDU-R05RY-R
KRDU-R23LY-R
KRDU-R23RY-R
KRDU-R32-R
```

The final branches are visible by default. Transition branches are available in the panel but hidden until enabled.

## Error Handling

The pipeline should prefer warnings over crashes for incomplete procedure geometry.

Current warning cases:

- unsupported leg type, for example `CA`, `DF`, `HM`
- missing fix ident
- unresolved fix coordinate
- missing altitude defaulted to `0 ft`
- branch with fewer than two supported route points

Hard failures are still used for invalid setup:

- missing `IN_CIFP.txt`
- missing `FAACIFP18`
- selected procedure not listed in `IN_CIFP.txt`
- requested branch has no matching procedure legs

Frontend behavior:

- Missing `procedures.geojson` logs a console warning with the preprocessing command.
- Malformed or failed panel fetch is shown as a panel error.
- Route features with fewer than two points are skipped in the Cesium hook.

## Current Limitations

The following items are intentionally not complete yet:

- `CF` course-to-fix geometry
- `DF` direct-to-fix geometry in missed approach
- `CA` and `VA` course/heading-to-altitude legs
- `HM`, `HF`, and `HA` hold geometry
- `RF` radius-to-fix arcs
- `P` profile record parsing
- LPV/LNAV/VNAV vertical path variants
- true procedure containment widths
- certified obstacle protection surfaces

Current vertical geometry uses leg altitudes, fix elevations, or fallback values. It does not yet derive a true glidepath from profile records.

Current tunnel geometry is marked:

```text
mode: visualApproximation
```

Do not describe it as official RNAV containment.

## Testing

Python tests:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python -m pytest \
  aeroviz-4d/python/tests/test_preprocess_procedures.py -q
```

Covered behavior:

- CIFP coordinate decoding
- KRDU `R05LY` index lookup
- single-procedure GeoJSON generation
- multi-runway RNAV generation
- transition branches hidden by default
- unresolved fixes produce warnings instead of exceptions

Frontend tests:

```bash
npm test -- --run \
  src/hooks/__tests__/useProcedureLayer.test.ts \
  src/components/__tests__/ProcedurePanel.test.tsx \
  src/utils/__tests__/procedureGeometry.test.ts
```

Covered behavior:

- tunnel cross-section generation
- empty or malformed procedure data handling
- entity cleanup
- panel grouping and toggles

Build check:

```bash
npm run build
```

## Development Workflow

When updating procedure support:

1. Modify parser behavior in `preprocess_procedures.py`.
2. Add or update Python tests.
3. Regenerate `procedures.geojson`.
4. Update TypeScript schema if properties change.
5. Update Cesium rendering only after schema changes are stable.
6. Update `ProcedurePanel` if the UI needs new grouping/filter controls.
7. Run Python tests, frontend tests, and build.
8. Manually inspect the Cesium scene near KRDU.

Recommended manual checks:

- Procedure final branch aligns with the expected runway end.
- Transition branches start at expected IAF/transition fixes.
- Warnings appear for skipped/simplified legs.
- Toggling a runway hides all its branches.
- Toggling a branch hides its polyline, fixes, and tunnel together.
- Existing trajectory/CZML loading still works.

## Future Work

Highest priority parser improvements:

1. Add `CF` geometry.
2. Add `DF` geometry.
3. Parse `P` profile records.
4. Use profile-derived vertical path for final approach.
5. Add hold geometry for missed approaches.
6. Add `RF` arc support for RNAV/RNP procedures.
7. Add explicit validation notes against FAA chart products for each rendered procedure.

When a leg is not fully supported, the output should continue to preserve a warning. The user should always be able to tell which parts of the displayed procedure are exact, simplified, or skipped.
