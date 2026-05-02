# AeroViz-4D 3D Procedure Annotation Layer Design

## 1. Goal

为当前 3D procedure protected-mode 可视化增加一个 annotation layer，使用户能快速理解每个 3D 图形代表什么、来自哪个 procedure/branch/segment/leg、是否是 source-backed geometry、estimate geometry、debug geometry，及其主要用途。

本阶段只设计，不实现。实现前需要 review 本文档。

## 2. Current Context

当前 3D procedure 可视化主要由 `src/hooks/useProcedureSegmentLayer.ts` 负责：

- 从 `loadProcedureRenderBundleData(activeAirportCode)` 读取 procedure detail docs、normalized procedure packages、render bundles。
- 对每个 branch/segment 创建 Cesium entities：
  - centerline polyline
  - primary/secondary envelope polygons
  - LNAV OEA
  - LNAV/VNAV OCS
  - LPV/GLS W/X/Y debug-estimate surfaces
  - aligned connector
  - missed section surface
  - CA course guide / CA estimated centerline / CA endpoint
  - turning missed debug anchor and debug primitives
  - missing-final-surface status point
  - visual turn fill patches
- 当前 entity 主要只有 `id`、`name`、`show`、geometry styling，没有统一的 annotation metadata。
- `ProcedurePanel.tsx` 已有 procedure branch visibility 和 `3D status`，适合作为 annotation mode 的主入口。

## 3. Design Principles

1. Annotation layer must explain, not certify.
   - UI 文案必须区分 `source-backed`、`estimated`、`debug-estimate`、`visual-fill-only`、`missing-source`。
   - 不把 first-pass/debug geometry 说成 FAA compliant/certified geometry。

2. Do not pollute geometry builders.
   - Geometry utilities continue to output geometry and diagnostics.
   - Annotation metadata is assembled at render/entity registration time from existing package, segment, leg, and diagnostic context.

3. Labels should be useful but not overwhelming.
   - Annotation mode off: no extra labels, current clean view.
   - Annotation mode on: show only necessary labels by default.
   - Selected entity: show rich detail in a floating inspector.

4. Click selection should be deterministic.
   - User single-clicks a visible procedure geometry.
   - App picks nearest Cesium entity with annotation metadata.
   - Selected entity gets highlight styling and a floating details panel.

5. Keep implementation modular.
   - Separate annotation metadata, label entity creation, Cesium picking, and React UI panel.

## 4. Proposed UX

### 4.1 Procedure Panel Entry

Add an `Annotate` toggle and a display-level selector in `ProcedurePanel`, near the existing `On` procedure master toggle and `3D status`.

Suggested controls:

- `Annotate` checkbox/toggle:
  - off by default.
  - enabled only when procedures layer is on.
  - when on, label entities become visible.

- Optional later control: annotation density.
  - `Key` / `All`
  - Phase 1 should implement only `Key` to keep scope tight.

- `Display` selector:
  - controls which classes of 3D procedure entities are visible.
  - applies to geometry entities and their annotation labels.
  - does not change branch/procedure visibility; it is an additional global filter.
  - default: `Protection`.

### 4.1.1 Display Levels

Display levels are cumulative. Selecting a higher level includes all lower levels.

| UI label | Included levels | Purpose |
| --- | --- | --- |
| `Core` | Level 1 | Coded procedure skeleton: fixes and nominal source-backed paths. |
| `Protection` | Levels 1-2 | Default research view: source-backed core plus primary protected geometry. |
| `Estimated` | Levels 1-3 | Adds model-derived operational geometry that is useful but should not be treated as direct source data. |
| `Visual Aid` | Levels 1-4 | Adds readability helpers such as turn fill patches. |
| `Debug` | Levels 1-5 | Adds experimental, missing-source, and debug construction artifacts. |

Level definitions:

1. `Source Backed Core`
   - `FIX`
   - `SEGMENT_CENTERLINE` when status is `SOURCE_BACKED`
   - `CA_COURSE_GUIDE`
   - Meaning: source-backed procedure structure and coded navigation intent.

2. `Source Backed Protection`
   - `SEGMENT_ENVELOPE_PRIMARY`
   - `SEGMENT_ENVELOPE_SECONDARY`
   - `FINAL_OEA`
   - `MISSED_SURFACE` when status is `SOURCE_BACKED`
   - Meaning: lateral protection and surface geometry built from source-backed segment paths and current rule assumptions.

3. `Estimated Operational Geometry`
   - `LNAV_VNAV_OCS`
   - `ALIGNED_CONNECTOR`
   - `MISSED_SURFACE` when status is `ESTIMATED`
   - `CA_CENTERLINE`
   - `CA_ENDPOINT`
   - `SEGMENT_CENTERLINE` when status is `ESTIMATED`
   - Meaning: geometry requiring repair, inference, GPA/TCH model construction, or CA climb assumptions.

4. `Visual Fill / Readability`
   - `TURN_FILL`
   - inter-segment turn fill patches
   - Meaning: visual continuity aids that make turns readable but are not compliant turn construction.

5. `Debug / Missing / Experimental`
   - `PRECISION_SURFACE`
   - `TURNING_MISSED_DEBUG`
   - `MISSING_FINAL_SURFACE`
   - Meaning: diagnostic or incomplete construction artifacts for development and validation review.

Implementation rule:

- Every procedure entity with annotation metadata must resolve to one display level from its `kind` and `status`.
- Entities without annotation metadata inside the procedure segment layer should default to Level 5 until explicitly classified, so unclassified helper geometry does not leak into the default view.
- Annotation label visibility is gated by both `Annotate` and the source entity's display level.

### 4.2 Label Behavior

When `Annotate` is on, show stable Cesium labels for:

- Important fixes:
  - IAF
  - IF
  - FAF/PFAF
  - MAPt/runway threshold
  - MAHF / missed hold fix
- Segment labels:
  - `INITIAL`
  - `INTERMEDIATE`
  - `FINAL_LNAV`, `FINAL_LNAV_VNAV`, `FINAL_LPV`, `FINAL_RNP_AR`
  - `MISSED_S1`, `MISSED_S2`
- Special/protected geometry labels:
  - `Primary`
  - `Secondary`
  - `LNAV OEA`
  - `LNAV/VNAV OCS`
  - `W/X/Y estimate`
  - `CA guide`
  - `CA estimated endpoint`
  - `Turning missed debug`
  - `Missing final surface`

Default label text should be short:

- Fix label: `FAF WEPAS`
- Segment label: `FINAL_LNAV`
- Protected geometry label: `LNAV/VNAV OCS`
- Missing-data label: `Missing: TCH`

Use muted color and small scale so labels do not dominate the 3D scene.

### 4.3 Click Detail Inspector

When annotation mode is on, single-clicking a procedure entity opens a floating inspector.

The inspector should show:

- Title:
  - procedure name
  - geometry type
- Context:
  - airport
  - runway
  - procedure id/name
  - branch id/name/role
  - segment id/type
  - leg id/type if applicable
- Meaning:
  - one or two sentences explaining what the clicked shape represents.
- Construction status:
  - `source-backed`
  - `estimated`
  - `debug-estimate`
  - `visual-fill-only`
  - `missing-source marker`
- Key parameters:
  - XTT/ATT if segment envelope
  - GPA if vertical surface uses it
  - TCH if available, otherwise explicitly `missing`
  - course and target altitude if CA
  - turn direction/course/debug type if turning missed
- Diagnostics:
  - top 3 relevant diagnostics from `segmentBundle.diagnostics` / `bundle.diagnostics`
- Source refs:
  - show compact source ids, e.g. `src:cifp-detail`

The panel should include:

- `Close`
- `Open Procedure Details` link/button if `procedureUid` is known
- later: `Focus branch` / `Show only this branch`

### 4.4 Highlight Behavior

On selected entity:

- For polyline:
  - increase width temporarily if feasible.
  - or add a second highlight polyline entity using the same positions.
- For polygon:
  - avoid mutating original material directly if it complicates cleanup.
  - add outline/highlight boundary entity or temporarily increase outline contrast.
- For point:
  - increase pixel size or add ring-like companion point.

Phase 1 can use a simple companion highlight entity because it is easier to clean up and avoids mutating original entity graphics.

### 4.5 Interaction Scope

Phase 1:

- Single-click only.
- No hover tooltip, because hover picking every mouse move can become noisy/performance-sensitive.

Phase 2:

- Optional hover pre-highlight and small tooltip.
- Optional label density controls.

## 5. Information Architecture

### 5.1 New Annotation Metadata Type

Add a frontend-only metadata model, for example:

```ts
export type ProcedureAnnotationKind =
  | "FIX"
  | "SEGMENT_CENTERLINE"
  | "SEGMENT_ENVELOPE_PRIMARY"
  | "SEGMENT_ENVELOPE_SECONDARY"
  | "FINAL_OEA"
  | "LNAV_VNAV_OCS"
  | "PRECISION_SURFACE"
  | "ALIGNED_CONNECTOR"
  | "MISSED_SURFACE"
  | "CA_COURSE_GUIDE"
  | "CA_CENTERLINE"
  | "CA_ENDPOINT"
  | "TURNING_MISSED_DEBUG"
  | "TURN_FILL"
  | "MISSING_FINAL_SURFACE";

export type ProcedureAnnotationStatus =
  | "SOURCE_BACKED"
  | "ESTIMATED"
  | "DEBUG_ESTIMATE"
  | "VISUAL_FILL_ONLY"
  | "MISSING_SOURCE";

export interface ProcedureEntityAnnotation {
  entityId: string;
  label: string;
  title: string;
  kind: ProcedureAnnotationKind;
  status: ProcedureAnnotationStatus;
  airportId: string;
  runwayId: string | null;
  procedureUid: string;
  procedureId: string;
  procedureName: string;
  branchId: string;
  branchName: string;
  branchRole: string;
  segmentId?: string;
  segmentType?: string;
  legId?: string;
  legType?: string;
  meaning: string;
  parameters: Array<{ label: string; value: string }>;
  diagnostics: string[];
  sourceRefs: string[];
}
```

This object should not be serialized into public data. It is assembled from render bundles at runtime.

### 5.2 Entity Registration

Current `addPolyline`, `addPoint`, `addRibbonPolygon` helpers return void and only add Cesium entities.

Proposed change:

- Create a small helper:

```ts
interface AddedProcedureEntity {
  entityId: string;
  annotation?: ProcedureEntityAnnotation;
  labelAnchor?: GeoPoint;
}
```

- Make each add helper return the `Cesium.Entity | null` or the id plus anchor.
- Keep a `Map<string, ProcedureEntityAnnotation>` in a hook/ref.
- Attach a lightweight property directly to entity for quick picking:

```ts
entity.properties = new Cesium.PropertyBag({
  aeroVizEntityType: "procedure-annotation",
  annotationId: entityId,
});
```

Use the ref map as the source of truth. The entity property is only a pick bridge.

### 5.3 Label Entities

Labels should be separate Cesium entities, not labels attached to the primary shape.

Reasons:

- Easy to toggle annotation visibility without touching geometry entities.
- Easy cleanup.
- Label placement can use representative points or dedicated fix points.
- Label entity can be excluded from detail picking if needed.

Label IDs:

```txt
procedure-annotation-label-<sourceEntityId>
```

Label entity properties:

```ts
{
  aeroVizEntityType: "procedure-annotation-label",
  annotationId: sourceEntityId
}
```

Clicking a label should select the source annotation.

### 5.4 React State

Extend `AppContext` with:

```ts
procedureAnnotationEnabled: boolean;
setProcedureAnnotationEnabled(enabled: boolean): void;
selectedProcedureAnnotation: ProcedureEntityAnnotation | null;
setSelectedProcedureAnnotation(annotation: ProcedureEntityAnnotation | null): void;
procedureDisplayLevel: ProcedureDisplayLevel;
setProcedureDisplayLevel(level: ProcedureDisplayLevel): void;
```

Where:

```ts
export type ProcedureDisplayLevel =
  | "CORE"
  | "PROTECTION"
  | "ESTIMATED"
  | "VISUAL_AID"
  | "DEBUG";
```

Optional later:

```ts
procedureAnnotationDensity: "KEY" | "ALL";
```

### 5.5 New Modules

Proposed files:

- `src/data/procedureAnnotations.ts`
  - annotation types
  - meaning text helpers
  - status mapping helpers

- `src/hooks/useProcedureAnnotationPicking.ts`
  - Cesium double-click handler
  - entity picking
  - selected annotation update
  - highlight companion entity management

- `src/components/ProcedureAnnotationPopup.tsx`
  - floating detail inspector

- optional:
  - `src/utils/procedureEntityMetadata.ts`
  - if `useProcedureSegmentLayer.ts` becomes too large during implementation.

## 6. Annotation Content Mapping

### 6.1 Segment Centerline

Meaning:

> Nominal coded path for this procedure segment. It is the reference line used to place lateral protected areas.

Status:

- `SOURCE_BACKED` when segment centerline is built from positioned fixes/RF metadata.
- `ESTIMATED` when CA endpoint/centerline backfill was used.

Parameters:

- segment type
- nav spec
- XTT
- ATT
- start/end fix
- leg types

### 6.2 Primary / Secondary Envelope

Meaning:

> Primary and secondary protected lateral areas around the segment centerline. Primary is the central protected area; secondary is the outer buffer/taper area when available.

Status:

- `SOURCE_BACKED` for normal segment envelopes built from known fixes.
- `ESTIMATED` when tied to estimated CA centerline.

Parameters:

- primary/secondary
- XTT/ATT
- width mode
- segment type

### 6.3 LNAV OEA

Meaning:

> Lateral obstacle evaluation area for the final segment. This is a protected area reference used by vertical surface construction.

Status:

- current implementation should be described as source-backed lateral geometry when centerline exists, but not a certified final template.

### 6.4 LNAV/VNAV OCS

Meaning:

> Sloping obstacle clearance surface estimate for LNAV/VNAV-style vertical guidance, built from final centerline, GPA, and TCH when both are available.

Status:

- `ESTIMATED`, because current implementation is marked `GPA_TCH_SLOPE_ESTIMATE`.

Parameters:

- GPA
- TCH
- construction status

If TCH missing:

- no OCS geometry should be present.
- missing marker/status should explain `TCH required`.

### 6.5 W/X/Y Precision Surfaces

Meaning:

> Debug-estimate W/X/Y surfaces for LPV/GLS-style visualization. These help compare vertical guidance surfaces but are not certified surface construction.

Status:

- `DEBUG_ESTIMATE`.

### 6.6 CA Course Guide / Endpoint / Centerline

Meaning:

> Course-to-altitude missed-approach guidance. The guide shows published course direction; the endpoint/centerline may be estimated from climb model when no explicit terminating fix exists.

Status:

- guide: `SOURCE_BACKED` for course direction, but not a full termination geometry.
- endpoint/centerline: `ESTIMATED`.

Parameters:

- course
- target altitude
- climb gradient
- distance
- start fix

### 6.7 Missed Section Surface

Meaning:

> Missed-approach protected area for section 1 or straight section 2. It helps show how the protected area continues after MAPt.

Status:

- `SOURCE_BACKED` if built from source-backed centerline.
- `ESTIMATED` if built from CA estimated centerline.

### 6.8 Turning Missed Debug

Meaning:

> Debug-only visualization for turning missed approach concepts such as turn initiation boundary, early/late baseline, nominal turn path, or wind spiral placeholder.

Status:

- `DEBUG_ESTIMATE`.

### 6.9 Visual Turn Fill

Meaning:

> Visual fill patch between adjacent segment envelopes. It improves readability at turns but is not a compliant turn construction.

Status:

- `VISUAL_FILL_ONLY`.

### 6.10 Missing Final Surface Marker

Meaning:

> Marker showing that expected final vertical/protected surfaces were not constructed because required source data or implementation is missing.

Status:

- `MISSING_SOURCE`.

Parameters:

- missing surface types
- missing source fields if known

## 7. Label Placement Rules

### 7.1 Fix Labels

Use actual fix coordinates from `pkg.sharedFixes`.

Only label important fixes in Phase 1:

- roles include `IAF`, `IF`, `FAF`, `PFAF`, `MAP`, `MAHF`, `RWY`, `FROP`

If multiple procedures share same fix:

- allow duplicate labels in Phase 1 only if labels are visually acceptable.
- Phase 2 can de-duplicate by `(fixId, lon, lat)` and append role summary.

### 7.2 Segment Labels

Use representative centerline point:

- middle point of centerline for normal segments.
- first or endpoint for short/estimated CA segments if midpoint is too close to MAPt.

### 7.3 Surface Labels

Use representative polygon/ribbon point:

- midpoint along centerline when available.
- otherwise representative point from ribbon boundaries.

### 7.4 Label Styling

Suggested Cesium label style:

- small font: `12px sans-serif`
- fill: white/near-white
- outline: dark translucent
- vertical origin: bottom
- pixel offset: `(0, -10)`
- distance display condition:
  - visible roughly from near to medium camera distance
  - hide when zoomed too far out to reduce clutter

## 8. Picking And Popup Behavior

### 8.1 Single-Click Picking

Use Cesium `ScreenSpaceEventHandler`:

- event: `Cesium.ScreenSpaceEventType.LEFT_CLICK`
- call `viewer.scene.pick(event.position)`
- resolve picked object:
  - picked entity
  - label entity with `annotationId`
  - source geometry entity with `annotationId`
- look up annotation in ref map
- update `selectedProcedureAnnotation`

### 8.2 Popup Position

Phase 1:

- Render popup as React floating panel in overlay grid, fixed near the right-middle or bottom-right.
- Do not attempt exact screen coordinate anchoring yet.

Reason:

- Cesium screen-space anchoring can be added later.
- Fixed panel is more stable and easier to test.

Phase 2:

- Position popup near selected entity using `Cesium.SceneTransforms.worldToWindowCoordinates`.

### 8.3 Selection Cleanup

Clear selected annotation when:

- annotation mode is turned off.
- procedures layer is turned off.
- active airport changes.
- selected entity is removed.
- user clicks close.

## 9. Implementation Plan

### Phase A: State And Documentation-Safe Skeleton

Files:

- `src/context/AppContext.tsx`
- `src/data/procedureAnnotations.ts`
- `src/components/ProcedurePanel.tsx`

Work:

- Add annotation mode state.
- Add procedure display-level state.
- Add `Annotate` toggle to ProcedurePanel.
- Add `Display` selector to ProcedurePanel.
- Add unit test for toggle if practical.

Acceptance:

- Toggle appears.
- Display selector appears and defaults to `Protection`.
- State updates.
- No labels/picking yet.

### Phase B: Entity Annotation Metadata

Files:

- `src/data/procedureAnnotations.ts`
- `src/hooks/useProcedureSegmentLayer.ts`

Work:

- Build `ProcedureEntityAnnotation` for every procedure entity.
- Add pure display-level classification for every annotation kind/status pair.
- Store annotation metadata in a ref map.
- Attach `annotationId` property to entities.
- Keep geometry rendering unchanged.

Acceptance:

- Existing rendering unchanged.
- Tests cover annotation metadata builder functions, not Cesium internals.
- Tests cover display-level classification.

### Phase C: Labels

Files:

- `src/hooks/useProcedureSegmentLayer.ts`
- optional `src/hooks/useProcedureAnnotationLabels.ts`

Work:

- Add label entities when annotation mode is enabled.
- Label visibility follows:
  - procedure layer on/off
  - branch visibility
  - annotation mode on/off
  - procedure display-level filter
- Labels are cleaned up with procedure entities.

Acceptance:

- Annotate off: no labels.
- Annotate on: important fix/segment/protected geometry labels appear.
- Branch toggles hide corresponding labels.

### Phase D: Single-Click Selection And Popup

Files:

- `src/hooks/useProcedureAnnotationPicking.ts`
- `src/components/ProcedureAnnotationPopup.tsx`
- `src/App.tsx`

Work:

- Register single-click picking handler.
- Resolve picked procedure annotation.
- Show popup with metadata.
- Add close button.

Acceptance:

- Single-clicking a procedure geometry opens popup.
- Single-clicking non-procedure geometry does nothing.
- Popup clears correctly.

### Phase E: Highlight

Files:

- `src/hooks/useProcedureAnnotationPicking.ts`

Work:

- Add companion highlight entity for selected annotation.
- Cleanup old highlight before adding new one.

Acceptance:

- Selected geometry is visually distinguishable.
- No stale highlight remains after close/airport change/layer off.

## 10. Testing Plan

### Unit Tests

- `procedureAnnotations.test.ts`
  - maps geometry kinds to meaning/status.
  - formats parameters and diagnostics.
  - maps CA endpoint, W/X/Y, missing-final status correctly.

- `ProcedurePanel.test.tsx`
  - annotation toggle is rendered.
  - toggle updates state.

- `ProcedureAnnotationPopup.test.tsx`
  - renders title/context/status/diagnostics.
  - close button calls handler.

### Hook Tests

Mocking Cesium double-click picking is possible but brittle.

Recommended:

- Keep picking resolution logic as a pure function:

```ts
resolvePickedProcedureAnnotation(picked, annotationMap)
```

- Unit test the resolver with mocked picked entity objects.

### Manual Verification

Run:

- `npm test -- --run`
- `npm run build`

Then manually verify in browser:

- Annotate toggle appears.
- Labels appear/disappear.
- Branch visibility hides labels.
- Double-click opens popup.
- Popup content matches clicked geometry.
- Non-procedure entities do not open procedure popup.

## 11. Risks And Mitigations

### Label Clutter

Risk:

- RNAV procedures with transitions can produce too many labels.

Mitigation:

- Phase 1 labels only key fixes and segment/protected geometry categories.
- Add density control later.
- Use distance display conditions.

### Picking Ambiguity

Risk:

- Overlapping polygons/lines can cause Cesium to pick a less useful entity.

Mitigation:

- Prefer labels and points when clicked.
- Add pick priority in resolver:
  - label
  - point/status marker
  - centerline
  - surface polygon
  - boundary/fill

### Performance

Risk:

- Many label entities and per-frame hover logic can affect Cesium performance.

Mitigation:

- No hover in Phase 1.
- Labels only when annotation mode is on.
- Label set limited to key annotations.

### Misleading Compliance Interpretation

Risk:

- Users may interpret debug/estimated surfaces as certified surfaces.

Mitigation:

- Popup must always show construction status.
- Debug/estimate labels include `estimate` or `debug`.
- Footer wording: `Research visualization only; not for navigation.`

## 12. Review Questions

1. Should Phase 1 label only key fixes and geometry categories, or should every leg receive a label?
2. Should the popup be fixed-position first, or anchored near the clicked 3D object from the beginning?
3. Should annotation state be global in `AppContext`, or local to `ProcedurePanel` plus procedure hooks?
4. Should double-click work only when `Annotate` is on, or always work for procedure geometry?
5. Should label text be English-only to match current UI, or bilingual for procedure-specific technical terms?

## 13. Recommended Review Decision

Recommended Phase 1 scope:

- Add `Annotate` toggle.
- Show key labels only.
- Enable double-click detail popup only while annotate mode is on.
- Use fixed-position popup.
- Add companion highlight entity.
- Keep hover and label density controls for Phase 2.

This gives immediate explanatory value without adding too much interaction complexity to the Cesium scene.
