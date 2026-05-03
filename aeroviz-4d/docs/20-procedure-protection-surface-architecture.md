# Procedure Protection Surface Architecture

## Status

Date: 2026-05-03

This document defines the next geometry architecture for AeroViz-4D procedure
protection visualization. It is a design plan, not a claim that the current
implementation is a certified TERPS construction.

Current official source anchors checked for this design:

- FAA Order 8260.3G, United States Standard for Terminal Instrument Procedures
  (TERPS), issued 2024-07-01, status Active:
  https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.current/documentNumber/8260.3
- FAA Order 8260.58D, United States Standard for Performance Based Navigation
  (PBN) Instrument Procedure Design, issued 2025-01-15, status Active:
  https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.information/documentID/1043458

## Problem Statement

The current procedure layer has improved from simple centerline drawing, but it
still mixes three different ideas:

1. Lateral envelopes: plan-view footprints around segment centerlines.
2. Vertical profile aids: blue profile ribbons connecting fix altitudes.
3. Protected surfaces: final OEA, LNAV/VNAV OCS, and missed approach surfaces.

This creates two practical problems:

- Users can mistake an estimated vertical-profile ribbon for lateral protection
  or OCS geometry.
- Missed approach continuity after a CA endpoint, such as CA endpoint to MAHF,
  is currently represented as a line connector rather than a lateral/vertical
  protected surface.

The target architecture must make lateral and vertical protection two aspects
of the same typed surface object, while keeping display-only aids separate and
visibly lower confidence.

## Design Principles

1. The renderer must not invent certified semantics. If the source data or rule
   implementation is incomplete, the object must be marked `DISPLAY_ESTIMATE`
   or `TERPS_ESTIMATE`.
2. A surface object owns both lateral and vertical definitions. Plan footprint
   and 3D OCS rendering are different views of the same object.
3. Connector geometry must be explicit. A CA endpoint to MAHF connector cannot
   be hidden inside a branch-level profile ribbon or merged into the CA surface.
4. Display aids must be separate from protection. The blue fix vertical profile
   should remain a profile aid/debug object, not a protection surface.
5. Rule modules belong in geometry utilities and render-bundle construction,
   not in Cesium rendering hooks.

## Proposed Data Model

Introduce a typed protection-surface model alongside the existing incremental
objects. The first implementation can coexist with current fields, then replace
them gradually.

```ts
type ProtectionSurfaceKind =
  | "FINAL_LNAV_OEA"
  | "FINAL_LNAV_VNAV_OCS"
  | "FINAL_PRECISION_DEBUG"
  | "MISSED_SECTION_1"
  | "MISSED_SECTION_2_STRAIGHT"
  | "MISSED_CONNECTOR"
  | "TURNING_MISSED_DEBUG";

type ProtectionSurfaceStatus =
  | "SOURCE_BACKED"
  | "TERPS_ESTIMATE"
  | "DISPLAY_ESTIMATE"
  | "DEBUG_ESTIMATE"
  | "MISSING_SOURCE";

interface ProcedureProtectionSurface {
  surfaceId: string;
  segmentId: string;
  sourceLegIds: string[];
  kind: ProtectionSurfaceKind;
  status: ProtectionSurfaceStatus;
  centerline: PolylineGeometry3D;
  lateral: {
    primary: VariableWidthRibbonGeometry | LateralEnvelopeGeometry;
    secondaryOuter: VariableWidthRibbonGeometry | LateralEnvelopeGeometry | null;
    widthSamples: Array<{ stationNm: number; primaryHalfWidthNm: number; secondaryOuterHalfWidthNm?: number }>;
    rule: string;
    notes: string[];
  };
  vertical: {
    kind: "NONE" | "ALTITUDE_PROFILE" | "OCS";
    origin: "SOURCE" | "GPA_TCH" | "MISSED_CLIMB" | "CENTERLINE_ALTITUDE_ONLY" | "ESTIMATED_CLIMB";
    samples: Array<{ stationNm: number; altitudeFtMsl: number }>;
    slopeFtPerNm?: number;
    notes: string[];
  };
  diagnostics: BuildDiagnostic[];
}
```

Initial migration rule: keep existing `finalOea`, `lnavVnavOcs`,
`missedSectionSurface`, and connector fields, but add a builder that can produce
`ProcedureProtectionSurface[]` from them. Once the renderer and tests use the
new model, remove duplicate legacy fields in a later pass.

## Rendering Model

The same `ProcedureProtectionSurface` should support multiple views:

- `Footprint`: plan-view lateral footprint, rendered as primary/secondary
  ribbons at a small height offset.
- `OCS/Vertical Surface`: 3D surface using the same lateral boundaries but with
  vertical samples applied per station.
- `Volume`: assessment volume derived from the same surface centerline,
  lateral width samples, and vertical samples. A dedicated Cesium side-wall
  volume mesh remains an optional future view; current rendering still uses
  footprints/OCS polygons plus direct volume assessment.
- `Profile Aid`: separate blue fix profile ribbon, hidden from protection
  semantics and not named as a surface.

Display levels:

- `Protection`: source-backed centerlines, source-backed lateral envelopes,
  source-backed OEA/OCS surfaces.
- `Estimated`: TERPS/display estimates such as CA endpoint surfaces and
  connector surfaces.
- `Debug`: incomplete W/X/Y, turning missed primitives, missing-source markers,
  and pure display aids.

## Problem 1: TERPS Coverage And Lateral/Vertical Unification

Current state:

- Final LNAV OEA exists as a lateral footprint.
- LNAV/VNAV OCS exists when GPA/TCH are available, but it is still an estimate.
- Missed section surfaces exist only for constructible straight geometry or
  estimated CA subsections.
- The blue vertical profile is not a TERPS surface.

Target state:

1. Convert final OEA and OCS builders to emit `ProcedureProtectionSurface`.
2. Convert missed section builders to emit `ProcedureProtectionSurface`.
3. Move the blue vertical profile to `Profile Aid` naming and annotation.
4. In the popup, show `lateral.rule`, width samples, `vertical.origin`, and
   confidence status so users understand why a surface is wide or estimated.
5. Use the same surface model for runway-profile assessment so lateral and
   vertical checks reference the same geometry.

Do not combine lateral and vertical by drawing a single ambiguous patch. Combine
them in data first; render as footprint, vertical surface, or volume depending
on the selected view.

## Problem 2: CA Endpoint To MAHF Connector Surface

Current state:

- `CA_MAHF_CONNECTOR` is a line from the estimated CA endpoint to the later
  MAHF/HOLD fix.
- It is correctly marked as estimated continuity geometry.
- It does not provide a lateral protected area.

Target state:

Add `MISSED_CONNECTOR` as an estimated protection surface:

1. Start point: estimated CA endpoint.
2. End point: MAHF/HOLD fix position, altitude fallback from CA target altitude
   if source elevation is missing.
3. Centerline: sampled great-circle line from CA endpoint to MAHF/HOLD.
4. Lateral width:
   - starts from the CA surface terminal primary/secondary widths;
   - can initially hold constant as `DISPLAY_ESTIMATE`;
   - later should implement rule-backed splay/width transition when source
     chapter logic is implemented.
5. Vertical samples:
   - start at CA target altitude;
   - continue using explicit missed climb gradient when available;
   - otherwise use centerline altitude/fallback and mark `CENTERLINE_ALTITUDE_ONLY`.
6. Rendering:
   - orange estimated connector footprint;
   - separate annotation kind or surface kind so it is not confused with the CA
     subsection surface.

The connector surface must not be merged into the CA surface. It is a separate
estimated surface with its own diagnostics and confidence level.

## Current CA Surface Width

The current CA estimate can look large because of conservative placeholder
parameters:

- missed `xttNm` defaults to `1 NM`;
- primary half-width is `2 * XTT = 2 NM`;
- secondary outer half-width is `3 * XTT = 3 NM`;
- CA endpoint distance is derived from altitude delta and default climb model
  when no explicit gradient is parsed.

This is explainable, but not precise TERPS modeling. The next implementation
must expose this width basis in the annotation and keep the object marked
estimated. Width reductions should come from rule-backed construction, not from
visual tuning.

## Implementation Plan

### Phase 0 - Preserve Current Fixes

Already implemented before this design page:

- Branch-level blue vertical profile no longer crosses missed approach
  discontinuities into CA endpoint/MAHF.
- Mixed `CA + DF` missed section can now generate a CA-only estimated missed
  surface without filling the DF connector as a surface.

### Phase 1 - Connector Surface

1. Add `MissedConnectorSurfaceGeometry`.
2. Build connector surfaces from `MissedCaEndpointGeometry` plus later
   MAHF/HOLD fixes.
3. Keep `MissedCaMahfConnectorGeometry` as the centerline/continuity line, but
   render the surface separately.
4. Add unit tests for KRDU-like `CA -> DF -> HM` data where DF start is the
   estimated CA endpoint.
5. Display in 3D at `Estimated` level with annotation explaining the rule basis.

### Phase 2 - Protection Surface Adapter

Implemented first adapter step:

- `ProcedureProtectionSurface` types now exist in
  `src/data/procedureProtectionSurfaces.ts`.
- Each branch render bundle exposes `protectionSurfaces`.
- The adapter maps:
  - final LNAV OEA;
  - LNAV/VNAV OCS;
  - LPV/GLS W/X/Y debug-estimate surfaces;
  - missed section surfaces;
  - missed connector surfaces.
- Existing legacy fields remain in place so current rendering and assessment
  logic do not change behavior during migration.
- Cesium annotations now read unified surface kind/status, lateral rule,
  primary/secondary width samples, and vertical origin from the adapter when a
  rendered surface has a matching `surfaceId`.
- Cesium surface polygons are rendered from `protectionSurfaces`; legacy
  segment fields are retained only as fallback for older bundle shapes.
- Runway-profile assessment now reads lateral width, LNAV/VNAV OCS, and
  precision debug surfaces from `protectionSurfaces`.

Remaining Phase 2 work:

1. Remove duplicate legacy fields only after rendering and assessment are fully
   migrated.

### Phase 3 - Profile Aid Cleanup

Implemented:

- `SEGMENT_VERTICAL_PROFILE` annotations now display as profile aids.
- The status is `PROFILE_AID`, not estimated protection.
- The display level is `Visual Aid`, so the blue profile ribbon no longer
  appears in `Estimated` protection views.
- Popup wording explicitly says it is not OEA, OCS, TERPS, or a protected
  surface.

### Phase 4 - Rule Refinement

Implemented first rule-structure step:

- Missed section and missed connector surfaces now carry explicit
  `lateralWidthRule` metadata.
- Missed section surfaces expose whether the width is a section 1 terminal
  width or section 2 straight width.
- CA endpoint to MAHF connector surfaces inherit the source missed surface
  terminal width when available; only missing source surfaces fall back to
  segment XTT-derived width.
- The connector transition status is explicit:
  `TERMINAL_WIDTH_HELD_TO_MAHF`.
- Turning missed debug primitives are adapted into
  `ProcedureProtectionSurface` objects with kind `TURNING_MISSED_DEBUG` and
  status `DEBUG_ESTIMATE`.
  - Closed TIA/wind placeholders render as debug areas.
  - Early/late/nominal turn placeholders render as narrow debug ribbons.
  - All annotations explicitly state that these are not certified TIA, wind
    spiral, or TERPS protected-area construction.
- Aircraft/profile assessment now has a direct protection-volume path:
  - `src/utils/procedureProtectionVolumeAssessment.ts` projects a GeoPoint to
    the same `ProcedureProtectionSurface` centerline rendered in 3D.
  - It interpolates lateral primary/secondary widths and vertical samples at
    the station, then returns primary/secondary/outside containment plus
    vertical relation to the surface/profile.
  - Runway profile routes carry the branch `protectionSurfaces`, and live
    aircraft samples prefer this 3D surface assessment before falling back to
    legacy route-band assessment.

Remaining Phase 4 work:

1. Replace the current XTT-derived estimate with chapter-specific TERPS/PBN
   missed-width tables when those source parameters are modeled.
2. Add a selectable rendered side-wall/top-bottom volume mesh if the UI needs a
   literal volume view. The current completed step is direct 3D containment
   assessment against the unified surface object, not a separate solid mesh.

## Acceptance Criteria

- `CA -> DF -> HM` procedures display:
  - CA estimated surface;
  - CA endpoint marker;
  - CA endpoint to MAHF connector line;
  - CA endpoint to MAHF estimated connector surface.
- No branch-level blue profile ribbon creates a large lateral-looking polygon
  between final and missed segments.
- Surface annotations show:
  - surface kind;
  - status;
  - lateral width rule;
  - vertical origin;
  - source diagnostics.
- Default `Protection` view does not imply estimated geometry is source-backed.
- `Estimated` view clearly shows estimated missed surfaces without mixing them
  with source-backed TERPS objects.
- `Debug` view shows turning missed placeholders as `TURNING_MISSED_DEBUG`
  surfaces, not as straight missed connector surfaces.
- Aircraft containment/profile assessment can use the same
  `ProcedureProtectionSurface` objects that the 3D procedure layer renders.
