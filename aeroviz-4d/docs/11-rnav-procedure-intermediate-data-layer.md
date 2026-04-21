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

**Explanation**:

- `landingThresholdFixRef`
  This is a reference into the shared `fixes[]` catalog. In simple words, it
  says: "the landing threshold for this procedure is that fix over there."
  We keep it as a reference so every branch and leg can point to the same
  runway-threshold object instead of repeating the same coordinates many
  times.

- `threshold`
  This is the physical runway-threshold data itself: longitude, latitude, and
  elevation. Think of `landingThresholdFixRef` as the link, and `threshold` as
  the actual location details. Keeping both is useful:
  - the reference keeps the model normalized and consistent
  - the inline threshold block makes downstream geometry work easier for final
    approach, tunnel generation, OCS generation, and validation

For a beginner, a good mental model is:

- `landingThresholdFixRef` answers: "which fix object is the runway threshold?"
- `threshold` answers: "where exactly is that threshold in the world?"

In a well-formed document, these two should agree with each other.

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

**Explanation**:

- `procedureFamily`
  This is the broad navigation family of the procedure. It tells us what kind
  of procedure we are dealing with before we look at the exact runway or
  variant.

  In this project, the main values are:
  - `RNAV_GPS`
    Standard RNAV approach based on GNSS/GPS-style navigation, such as
    `RNAV(GPS) Y RW05L`.
  - `RNAV_RNP`
    RNP-style RNAV approach, usually with tighter coded path requirements and
    sometimes more advanced leg types such as `RF`.
  - `ILS`
    Included for completeness even though it is not an RNAV procedure.
  - `LOC`
    Included for completeness even though it is not an RNAV procedure.
  - `UNKNOWN`
    Safe fallback when the family cannot yet be classified.

  For the thesis work here, the most important families are `RNAV_GPS` and
  `RNAV_RNP`.

- `procedureIdent`
  This is the compact machine-friendly published identifier used in CIFP. It is
  not the full chart title.

  Example:
  - `R05LY`

  In the current project conventions, we read that as:
  - `R`: RNAV(GPS)-family approach
  - `05L`: runway 05 Left
  - `Y`: one specific published variant for that runway

  The full human-readable chart title for that example is
  `RNAV(GPS) Y RW05L`.

  So a useful beginner rule is:
  - `procedureIdent` is the short coded name used by the database
  - `chartName` is the full chart-facing label a pilot would recognize

- `variant`
  This distinguishes multiple different procedures to the same runway inside
  the same family. Common letters are `Y` and `Z`.

  Important point:
  - `Y` and `Z` do not mean "better" or "worse"
  - they simply distinguish different published procedure designs

  Example:
  - `R05LY` has variant `Y`
  - `H05LZ` would have variant `Z`
  - `R32` has no suffix, so `variant` can be `null`

- `approachModes`
  This lists the published operating/minima modes associated with the
  procedure. These are not separate branches; they are different ways the same
  procedure may be flown, especially in the vertical guidance and minima sense.

  Common values in RNAV approach charts are:
  - `LPV`
    Localizer Performance with Vertical guidance. Practically, this is the
    precision-like vertically guided mode many pilots expect on an RNAV(GPS)
    chart.
  - `LNAV_VNAV`
    Lateral navigation plus vertical navigation. This also gives vertical
    guidance, but it is a different mode from LPV.
  - `LNAV`
    Lateral navigation only. This gives left-right guidance but no coded
    vertical guidance path in the same way as LPV or LNAV/VNAV.

  A helpful beginner summary is:
  - `LPV`, `LNAV_VNAV`, `LNAV` usually share the same lateral path
  - what changes is mostly the vertical guidance and the published minima

  This field exists because one procedure document may need to say:
  "these are the published modes supported by this approach," even if the
  branch geometry is the same.

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

**Explanation**:

The `kind` field tells us what a fix *means* in the procedure, not just that it
has coordinates.

Recommended kinds:

- `named_fix`
  A normal published waypoint such as `WEPAS` or `SCHOO`. This is the most
  common kind.

- `runway_threshold`
  The physical runway threshold point. Use this when the point is literally the
  runway end that the approach is built toward.

- `mapt`
  Missed Approach Point as a semantic object. Sometimes the MAPt is located at
  the runway threshold, but its *role* is different: it marks the point where
  the missed-approach decision/procedure starts. If the project needs to keep
  that meaning explicit, using `mapt` as a separate kind can be useful.

- `missed_hold_fix`
  A fix that anchors the missed-approach hold, such as a MAHF. This kind is
  useful because missed-approach logic is often special and should not be mixed
  up with normal enroute or final-approach fixes.

- `center_fix`
  A geometric center point used for curved path construction, especially for
  `RF` arcs or similar turn geometry. This point may matter for geometry even
  if it is not emphasized on the chart like a normal waypoint.

- `virtual_fix`
  A synthetic point created by our own pipeline, not necessarily a named
  published fix. We use this when the procedure logic needs an explicit point
  for geometry or sequencing, but the source only gave an instruction such as
  "fly course to altitude."

Beginner-friendly rule:

- if the point is published and named, start with `named_fix`
- if it is literally the runway end, use `runway_threshold`
- if it exists mainly to explain missed-approach logic, use `mapt` or
  `missed_hold_fix`
- if we invented it to make geometry explicit, use `virtual_fix`

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

**Explanation**:

- `roleAtEnd`
  This tells us what the endpoint of the leg *means in the procedure*.

  This is very important for beginners because `roleAtEnd` is **not** the same
  thing as the path terminator:
  - `path.pathTerminator` says how the aircraft gets there
  - `roleAtEnd` says what that endpoint means operationally

  Example:
  - a `TF` leg can end at the `FAF`
  - another `TF` leg can end at the `MAPt`

  Same leg type, different role.

  Typical values might be:
  - `IAF`
  - `IF`
  - `FAF`
  - `MAPt`
  - `MAHF`
  - `Route`

  So `roleAtEnd` is mainly about semantics, display, and validation.

- `constraints`
  This groups the published rules attached to this leg or its endpoint. In
  plain language, constraints answer questions such as:
  - how high must the aircraft be here?
  - is there a speed limit here?
  - what altitude should our 3D geometry use if we want to draw this leg?

  This grouping matters because the chart/CIFP may tell us several different
  things at once:
  - a published altitude restriction
  - maybe a speed restriction
  - a geometry altitude we choose for rendering or tunnel construction

  In other words:
  - `roleAtEnd` says what the point is
  - `constraints` says what rules apply there

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

**Explanation**:

`termination` answers a simple but very important question:

"What condition tells us this leg is finished?"

This is different from `path`.

- `path` tells us how we travel
- `termination` tells us when we stop that leg

Examples:

- If the leg ends when we reach `WEPAS`, then termination is `kind: fix`
- If the leg ends when we climb to `1000 ft`, then termination is
  `kind: altitude`
- If the leg ends by entering or establishing a hold, then termination is
  `kind: hold`

This separation keeps the model cleaner. A leg such as `CA` is not really
"to a fix"; it is "fly a course until an altitude condition is met." That is
why `termination` deserves its own object instead of being hidden inside the
path text.

For manual authoring, if the exact coded termination is unknown, `kind: manual`
is acceptable as long as a warning explains the limitation.

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

**Explanation**:

`constraints` is where we preserve the published operational limits from the
source.

For beginners, the most important idea is:

- published constraint altitude and drawn geometry altitude are related, but
  they are not always identical

Why can they differ?

- the chart/CIFP may publish a crossing altitude or minimum altitude
- but the geometry we draw in 3D may need the physical runway elevation or a
  derived altitude so the line/tunnel is visually meaningful

That is exactly why this schema keeps:

- `altitude`
  the published operational rule
- `geometryAltitudeFt`
  the altitude we actually use to place the point/segment in 3D

Example:

- At `RW05L`, the published altitude in the current pipeline is `424 ft`
- but the 3D geometry altitude is `798 ft`, because that matches the runway
  threshold elevation used for drawing

So when reading this object:

- trust `altitude` for procedure semantics
- trust `geometryAltitudeFt` for rendering and downstream geometry generation

If later we add more detail, this section is also the natural place for:

- speed windows
- climb/descent gradient notes
- raw chart text preserved for auditability

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

**Explanation**:

`verticalProfiles` describes the procedure in the vertical dimension: height
versus distance along the approach.

That is different from the lateral path:

- branches and legs tell us where the aircraft goes on the map
- vertical profile tells us how high it should be while going there

For a beginner, think of it this way:

- `branches[].legs[]` describe the route drawn from above
- `verticalProfiles[]` describe the side-view descent/climb story

Why do we need a separate section?

- altitude constraints at fixes are only *discrete checkpoints*
- a vertical profile is a more continuous description of the descent or climb
  between those checkpoints

### Do we have vertical profile data today?

Not really, at least not in a fully parsed form in this repo.

Current situation in AeroViz-4D:

- we do have some altitude information from the leg records we already parse
- we do **not** yet build a full authoritative continuous vertical profile
- we therefore use placeholders such as `constraint_interpolation` when needed

### Does CIFP contain vertical-profile data?

Potentially yes, but the answer is subtle:

- FAA CIFP can contain more vertical information than we currently use
- in the current docs and plan, this is described as future `P` profile record
  support
- our current parser mainly reads a subset of procedure-leg records, so that
  richer vertical data is not yet normalized into this model

So the practical answer is:

- it is not correct to simply say "CIFP has no vertical data"
- it is more accurate to say "our current pipeline does not yet parse and use
  the richer vertical-profile data"

### Why keep this section now if it is not fully populated yet?

Because the schema should describe the *complete target model*, not only the
subset we have already implemented.

Keeping `verticalProfiles[]` now gives us a clean place for future data such as:

- glidepath angle
- threshold crossing height
- mode-specific vertical behavior
- chart-derived descent profile information

Until that parser work exists, the field can:

- be empty
- contain a placeholder profile with `basis: constraint_interpolation`
- or contain manual chart-derived profile data when available

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
