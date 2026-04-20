# RNAV Procedure Intermediate Data Layer

## Purpose

The current pipeline goes straight from FAA CIFP parsing to frontend-facing
`procedures.geojson`. That works for Cesium rendering, but it mixes three
concerns into one file:

1. published procedure semantics
2. geometry construction rules
3. renderer-specific output

For the next stage of the thesis work, AeroViz-4D should add a source-neutral
intermediate layer:

```text
ARINC 424 / FAA CIFP
        or
manual entry from IAP RNAV chart
        ->
RNAV procedure intermediate JSON
        ->
GeoJSON / Cesium route / profile / OCS / tunnel
```

This intermediate JSON should be the single procedure-construction contract.
It should be rich enough to build a runway-specific procedure even when the
input came from:

- direct CIFP parsing
- chart-guided manual authoring
- a hybrid workflow where CIFP is primary but some missing items are completed
  manually

## Design Goals

- One JSON document describes one runway-specific procedure, for example
  `KRDU R05LY RW05L`.
- The model is more semantic than `procedures.geojson`.
- The model is less raw than fixed-width CIFP text.
- Branch topology is preserved.
- Final and missed approach are both preserved, even if the frontend does not
  yet render every leg type.
- Published altitude constraints stay separate from geometry altitudes.
- Provenance, warnings, and validation notes stay attached to the data.
- Manual chart entry is allowed, but the record must still be geometry-ready.

## Boundary

This layer should contain everything needed to construct the procedure.

It should contain:

- airport, runway, and procedure identity
- all fixes used by the procedure, with coordinates when geometry is expected
- branch topology, including transitions and the base final branch
- ordered legs with segment type and path-construction rules
- published altitude and speed constraints
- optional vertical-profile data and minima modes
- provenance, warnings, and validation notes
- optional display hints that are useful downstream but not authoritative

It should not contain:

- Cesium entities
- duplicated `Point` features just for labels
- prebuilt tunnel polygons
- frontend-only grouping state

Those belong in derived outputs such as `procedures.geojson`.

## Key Modeling Decision

The intermediate layer should model **source branches** and **procedure
segments** separately.

Why this matters:

- In current KRDU `R05LY`, branch `R` contains both final-approach legs and
  missed-approach legs.
- If branch type alone is used, missed approach semantics get lost or forced
  into an awkward shape.

So the model should use:

- `branchRole`
  - `base_final`
  - `transition`
  - `other`
- `segmentType` on each leg
  - `initial`
  - `intermediate`
  - `final`
  - `missed`

## Proposed JSON Contract

### Top-Level Object

```json
{
  "schemaVersion": "1.0.0",
  "modelType": "rnav-procedure-runway",
  "procedureUid": "KRDU-R05LY-RW05L",
  "provenance": {},
  "airport": {},
  "runway": {},
  "procedure": {},
  "fixes": [],
  "branches": [],
  "verticalProfiles": [],
  "validation": {},
  "displayHints": {}
}
```

### Top-Level Fields

| Field | Required | Purpose |
| --- | --- | --- |
| `schemaVersion` | yes | Version the intermediate contract independently from GeoJSON. |
| `modelType` | yes | Fixed type marker such as `rnav-procedure-runway`. |
| `procedureUid` | yes | Deterministic unique ID, for example `KRDU-R05LY-RW05L`. |
| `provenance` | yes | Source files, chart references, manual edits, warnings, confidence. |
| `airport` | yes | Airport identity and optional airport metadata. |
| `runway` | yes | Landing runway context and threshold reference. |
| `procedure` | yes | Published procedure identity and modes. |
| `fixes` | yes | Deduplicated fix catalog used by all branches. |
| `branches` | yes | Ordered source branches with leg definitions. |
| `verticalProfiles` | recommended | Final approach vertical data, P-record results, or chart-derived fallback. |
| `validation` | recommended | Cross-check notes against charted runway, FAF, MAPt, MAHF, etc. |
| `displayHints` | optional | Nominal speed, tunnel defaults, default-visible branches. |

## Object Definitions

### `provenance`

Use `provenance` to make the data trustworthy and auditable.

Recommended shape:

```json
{
  "assemblyMode": "cifp_primary_chart_crosscheck",
  "researchUseOnly": true,
  "sources": [
    {
      "sourceId": "src:cifp-detail",
      "kind": "FAA_CIFP",
      "cycle": "2603",
      "path": "data/CIFP/CIFP_260319/FAACIFP18"
    },
    {
      "sourceId": "src:chart",
      "kind": "FAA_IAP_CHART",
      "procedureTitle": "RNAV(GPS) Y RW05L",
      "usedFor": ["validation", "manual_completion"]
    }
  ],
  "warnings": []
}
```

Recommended `assemblyMode` values:

- `cifp_only`
- `cifp_primary_chart_crosscheck`
- `chart_manual`
- `hybrid_manual_completion`

### `airport`

Minimal required shape:

```json
{
  "icao": "KRDU",
  "faa": "RDU"
}
```

Optional fields:

- `name`
- `referenceElevationFt`
- `magneticVariationDeg`

### `runway`

This is the landing runway context for the procedure document, not the whole
airport runway system.

Recommended shape:

```json
{
  "ident": "RW05L",
  "landingThresholdFixRef": "fix:RW05L",
  "threshold": {
    "lon": -78.80196389,
    "lat": 35.87445,
    "elevationFt": 798
  }
}
```

Optional fields:

- `trueCourseDeg`
- `magneticCourseDeg`
- `tchFt`
- `runwayLengthFt`

### `procedure`

This is the published identity block.

Recommended shape:

```json
{
  "procedureType": "SIAP",
  "procedureFamily": "RNAV_GPS",
  "procedureIdent": "R05LY",
  "chartName": "RNAV(GPS) Y RW05L",
  "variant": "Y",
  "runwayIdent": "RW05L",
  "baseBranchIdent": "R",
  "approachModes": ["LPV", "LNAV_VNAV", "LNAV"]
}
```

This object should capture chart/CIFP identity once, instead of duplicating it
on every output feature.

### `fixes`

`fixes` is a deduplicated catalog shared by every branch.

Each fix should contain:

- `fixId`
- `ident`
- `kind`
- `position`
- `elevationFt`
- `sourceRefs`
- optional `roleHints`

Recommended `kind` values:

- `named_fix`
- `runway_threshold`
- `mapt`
- `missed_hold_fix`
- `center_fix`
- `virtual_fix`

Recommended shape:

```json
{
  "fixId": "fix:WEPAS",
  "ident": "WEPAS",
  "kind": "named_fix",
  "position": {
    "lon": -78.88295556,
    "lat": 35.80876667
  },
  "elevationFt": null,
  "roleHints": ["FAF"],
  "sourceRefs": ["src:cifp-detail"]
}
```

### `branches`

Each source branch becomes one branch object. That keeps CIFP/chart topology
intact.

Recommended branch fields:

- `branchId`
- `branchIdent`
- `branchRole`
- `sequenceOrder`
- `mergeFixRef`
- `continuesWithBranchId`
- `defaultVisible`
- `legs`
- `warnings`

For the current KRDU data:

- `ACHWDR` is a `transition`
- `AOTTOS` is a `transition`
- `R` is the `base_final` branch

### `legs`

Each leg is the core construction record.

Recommended leg fields:

- `legId`
- `sequence`
- `segmentType`
- `path`
- `termination`
- `constraints`
- `roleAtEnd`
- `sourceRefs`
- `quality`

Recommended leg skeleton:

```json
{
  "legId": "leg:R:020",
  "sequence": 20,
  "segmentType": "final",
  "path": {},
  "termination": {},
  "constraints": {},
  "roleAtEnd": "FAF",
  "sourceRefs": ["src:cifp-detail"],
  "quality": {
    "status": "exact"
  }
}
```

#### `path`

`path` describes how to construct the leg in plan view.

Recommended common fields:

- `pathTerminator`
- `constructionMethod`
- `startFixRef`
- `endFixRef`

Recommended `constructionMethod` values:

- `if_to_fix`
- `track_to_fix`
- `course_to_fix`
- `direct_to_fix`
- `course_to_altitude`
- `radius_to_fix`
- `hold`
- `manual_polyline`

This is the source-neutral bridge between CIFP parsing and manual chart entry.

Examples:

- CIFP `IF` -> `constructionMethod: if_to_fix`
- CIFP `TF` -> `constructionMethod: track_to_fix`
- CIFP `CF` -> `constructionMethod: course_to_fix`
- CIFP `DF` -> `constructionMethod: direct_to_fix`
- CIFP `CA` / `VA` -> `constructionMethod: course_to_altitude`
- CIFP `RF` -> `constructionMethod: radius_to_fix`
- CIFP `HM` / `HF` / `HA` -> `constructionMethod: hold`
- chart-only manual segment -> `constructionMethod: manual_polyline`

If the source is a chart and no ARINC terminator is known, `manual_polyline`
is acceptable, but then the author must provide explicit geometry points or
equivalent arc data. Chart text alone is not geometry-ready.

#### `termination`

`termination` keeps leg-end semantics explicit.

Recommended `kind` values:

- `fix`
- `altitude`
- `hold`
- `manual`

Examples:

- `TF WEPAS` -> `kind: fix`
- `CA 1000 ft` -> `kind: altitude`
- `HM at DUHAM` -> `kind: hold`

#### `constraints`

Keep published constraints separate from geometry.

Recommended shape:

```json
{
  "altitude": {
    "qualifier": "at",
    "valueFt": 2200,
    "rawText": "+ 02200"
  },
  "speedKt": null,
  "geometryAltitudeFt": 2200
}
```

Recommended altitude qualifiers:

- `at`
- `at_or_above`
- `at_or_below`
- `terminate_at`
- `window`
- `advisory`
- `unknown`

`geometryAltitudeFt` is the altitude used for 3D geometry construction. It is
allowed to differ from the published altitude constraint. That is already true
in the current pipeline at `RW05L`, where the published altitude is `424 ft`
while the geometry altitude comes from runway elevation `798 ft`.

### `verticalProfiles`

This section is where future CIFP `P` records or chart-derived glidepath data
should live.

Recommended fields:

- `profileId`
- `appliesToModes`
- `branchId`
- `fromFixRef`
- `toFixRef`
- `basis`
- `glidepathAngleDeg`
- `thresholdCrossingHeightFt`
- `constraintSamples`

Recommended `basis` values:

- `cifp_p_record`
- `chart_manual`
- `constraint_interpolation`
- `unknown`

If `P` records are not yet parsed, the model can still carry a placeholder
profile so the gap is explicit instead of hidden.

### `validation`

This section records cross-checks against the published chart.

Recommended fields:

- `expectedRunwayIdent`
- `expectedIAFs`
- `expectedIF`
- `expectedFAF`
- `expectedMAPt`
- `expectedMissedHoldFix`
- `knownSimplifications`

### `displayHints`

This is optional and non-authoritative. It is useful for downstream derived
products.

Recommended fields:

- `nominalSpeedKt`
- `defaultVisibleBranchIds`
- `tunnelDefaults`

Example:

```json
{
  "nominalSpeedKt": 140,
  "defaultVisibleBranchIds": ["branch:R"],
  "tunnelDefaults": {
    "lateralHalfWidthNm": 0.3,
    "verticalHalfHeightFt": 300,
    "sampleSpacingM": 250,
    "mode": "visualApproximation"
  }
}
```

## Manual Chart Entry Rules

To be considered complete enough for geometry construction, a manually-authored
record should provide all of the following:

1. landing runway threshold position and elevation
2. coordinates for every named fix used in the path, or explicit manual path
   geometry
3. branch topology, including where a transition merges into the base final
   branch
4. ordered leg construction rules
5. published altitude constraints
6. final approach profile information if available from the chart

If the chart alone does not expose enough geometry to reconstruct an arc or a
hold precisely, the author should either:

- resolve the missing fix/arc data from another nav source and mark it in
  `sourceRefs`, or
- mark the leg as `quality.status: partial` and provide a manual polyline
  approximation with an explicit warning

## Mapping From Existing Docs

| Existing source | Intermediate field |
| --- | --- |
| `IN_CIFP.txt` airport/procedure row | `airport`, `procedure.procedureType`, `procedure.procedureIdent` |
| `FAACIFP18` header cycle | `provenance.sources[].cycle` |
| airport/local fix records | `fixes[]` |
| procedure branch ident | `branches[].branchIdent` |
| leg sequence | `branches[].legs[].sequence` |
| path terminator (`IF`, `TF`, `CF`, `DF`, `CA`, `HM`, `RF`) | `branches[].legs[].path.pathTerminator` |
| altitude text | `branches[].legs[].constraints.altitude.rawText` |
| rendered geometry altitude | `branches[].legs[].constraints.geometryAltitudeFt` |
| future `P` profile records | `verticalProfiles[]` |
| chart title/minima box | `procedure.chartName`, `procedure.approachModes`, `verticalProfiles[]` |
| chart missed-approach text | missed-approach legs in `branches[]` plus `validation` notes |

## Example

A concrete KRDU example is provided here:

[`11-rnav-procedure-intermediate-data-layer.example.json`](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/docs/11-rnav-procedure-intermediate-data-layer.example.json)

That example shows:

- a deduplicated fix catalog
- one transition branch (`ACHWDR`)
- the base final branch (`R`)
- final approach and missed-approach legs in the same source branch
- separation of published altitude versus geometry altitude
- optional display hints for the current AeroViz-4D pipeline

## Recommended Next Step In AeroViz-4D

Refactor the pipeline to:

1. parse CIFP or manual chart input into this intermediate JSON
2. validate the intermediate JSON
3. derive `procedures.geojson` from the intermediate JSON
4. later derive profile views, OCS inputs, and tunnel geometry from the same
   source

That keeps the thesis pipeline honest: one semantic procedure model, many
derived visual products.
