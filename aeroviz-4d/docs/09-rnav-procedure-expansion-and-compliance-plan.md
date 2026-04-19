# RNAV Procedure Expansion and Compliance Plan

## Goal

Expand the current single-procedure KRDU `R05LY` visualization into a runway-grouped procedure system where each runway's RNAV procedures can be turned on/off independently. Then improve the CIFP parsing and geometry so the displayed plan/profile/tunnel follows published CIFP procedure definitions as closely as practical for a thesis visualization tool.

The target remains research visualization, not certified navigation.

## Current State

- Current generated data contains one route only: `KRDU R05LY`, branch `R`, displayed as `RNAV(GPS) Y RW05L`.
- Current route points are `SCHOO -> WEPAS -> RW05L`.
- Current parser only renders `IF` and `TF` legs.
- Current missed approach legs are skipped with warnings: `CA`, `DF`, `HM`.
- Current tunnel is a fixed-width/fixed-height visualization volume, not an official RNAV containment or obstacle-protection surface.

## Implementation Status

- Implemented multi-procedure generation for `R05LY`, `R05RY`, `R23LY`, `R23RY`, and `R32`.
- Implemented `--include-all-rnav` and `--include-transitions`.
- Generated `procedures.geojson` now contains 17 drawable route branches across five runway groups.
- Implemented per-runway, per-procedure, and per-branch visibility controls in a dedicated `ProcedurePanel`.
- Final branches are visible by default; transition branches are generated but hidden until enabled.
- Unsupported or incomplete branches are preserved as warnings instead of silently disappearing.
- Still pending: `CF`, `DF`, `CA`, hold, `RF`, and `P` profile record support for closer official-procedure geometry.

## Priority 1: Generate Procedures For Other Runways

### Target Procedure Set

Generate these KRDU RNAV/GPS procedures first:

- `R05LY` for `RW05L`
- `R05RY` for `RW05R`
- `R23LY` for `RW23L`
- `R23RY` for `RW23R`
- `R32` for `RW32`

Then add the RNAV/RNP-style `H` procedures after confirming their CIFP semantics and chart naming:

- `H05LZ`
- `H05RZ`
- `H23LZ`
- `H23RZ`

Do not mix ILS/LOC procedures (`I05L`, `L05L`, etc.) into the first RNAV layer unless a separate procedure family filter is added.

### Parser Changes

- Change `preprocess_procedures.py` from one-procedure output to multi-procedure output.
- Add CLI options:
  - `--procedures R05LY,R05RY,R23LY,R23RY,R32`
  - `--include-all-rnav` to select the default RNAV set above.
  - `--include-transitions` to output initial transition branches, not just final branch `R`.
- For each procedure, output one route feature per branch:
  - final branch: `R`, `H`, or equivalent final segment branch
  - transition branches: examples already observed in CIFP include `ACHWDR`, `AOTTOS`, `ABUTTS`, `ADUWON`, `AESELL`, etc.
- Keep deterministic ordering:
  - runway order: `05L`, `05R`, `23L`, `23R`, `32`
  - procedure ident order within runway
  - branch order with final branch first, transitions after
  - leg sequence order inside each branch

### Output Schema Changes

Extend `procedures.geojson` so the frontend can group and toggle procedures:

- Add collection metadata:
  - `airport`
  - `sourceCycle`
  - `generatedAt`
  - `procedureFamilies`
- Add route properties:
  - `procedureFamily`: `RNAV_GPS`, `RNAV_RNP`, `ILS`, `LOC`, or `UNKNOWN`
  - `runwayIdent`: e.g. `RW05L`
  - `procedureVariant`: `Y`, `Z`, or `null`
  - `branchType`: `final`, `transition`, or `missed`
  - `branchIdent`
  - `legCoverage`: list of parsed, simplified, and skipped leg types
- Preserve existing fix, samples, tunnel, warnings, and `researchUseOnly` fields.

## Priority 2: Add A Dedicated Procedures Panel

Create a separate procedure control panel instead of placing all procedure controls in the generic layer list.

### UI Behavior

- Keep a top-level `RNAV Procedures` layer toggle in the existing `ControlPanel`.
- Add a new `ProcedurePanel` overlay with:
  - airport/source-cycle summary
  - family filters: `RNAV GPS`, `RNAV RNP`
  - runway groups: `RW05L`, `RW05R`, `RW23L`, `RW23R`, `RW32`
  - nested procedure toggles under each runway
  - nested branch toggles for final, transition, and missed approach
  - warning badge when a route has simplified/skipped legs
- Default visibility:
  - all runway groups collapsed except the current airport's selected/default runway
  - final branches visible
  - transitions hidden initially
  - missed approach hidden initially until leg support is improved

### State Changes

- Add procedure visibility state separate from generic layer booleans:
  - visible procedure IDs
  - visible branch IDs
  - active runway group
  - family filter state
- Keep `layers.procedures` as the master on/off switch.
- Do not reload `procedures.geojson` when toggling; update entity `show` flags only.

## Priority 3: Move Toward Official CIFP Geometry

The implementation should progressively replace approximations with CIFP-derived geometry.

### Leg-Type Support Order

Implement leg support in this order:

1. `IF`, `TF`: already implemented, keep and harden.
2. `CF`: course-to-fix, needed for many final approach segments.
3. `DF`: direct-to-fix, needed for missed approach.
4. `CA`, `VA`: course/heading to altitude, needed for climb-out.
5. `HM`, `HF`, `HA`: holds, needed for missed approach holding patterns.
6. `RF`: radius-to-fix arcs, especially important for RNAV/RNP procedures.

For any unsupported leg, preserve it in output as a `skipped` or `simplified` leg with a warning. Do not silently omit it.

### Vertical/Profile Definition

- Parse CIFP `P` profile records for each procedure where present.
- Use profile records to define:
  - threshold crossing height
  - glidepath angle
  - final approach vertical path
  - LPV/LNAV/VNAV/LNAV minima variants where encoded
- Stop using runway elevation alone as the final vertical definition when profile data provides a better path.
- Add a `profile` object to each route with distance-vs-altitude samples for future 2D profile view.

### Tunnel Definition

- Keep the current fixed tunnel as a fallback display mode.
- Add a derived mode based on procedure properties:
  - use RNP/required navigation performance where available
  - widen/narrow lateral volume by segment type when data supports it
  - split final, transition, and missed approach tunnels with different styling
- Label the tunnel mode in properties:
  - `visualApproximation`
  - `cifpDerived`
  - `unknown`

### Validation Against Official Sources

- Treat local CIFP cycle `2603` as the primary machine-readable source.
- Cross-check procedure names, runway mapping, and major fixes against official FAA chart products for the same cycle when available.
- For each procedure, record validation notes:
  - expected runway
  - expected IAF/IF/FAF/MAPt
  - expected missed approach hold fix
  - known simplified legs

## Priority 4: Testing And Acceptance Criteria

### Python Tests

- Verify multi-procedure generation produces at least the five KRDU RNAV/GPS procedures.
- Verify each generated runway group has at least one final branch.
- Verify every route has deterministic `routeId` and stable ordering.
- Verify unsupported legs are reported in `warnings` and `legCoverage`.
- Verify profile records are parsed for procedures with `P` records.
- Verify missing fix coordinates skip only the affected leg and do not abort the whole batch.

### Frontend Tests

- Verify `ProcedurePanel` groups routes by runway.
- Verify master `layers.procedures` hides all procedure entities.
- Verify runway-level toggle hides/shows all procedures under one runway.
- Verify branch-level toggle hides/shows only that branch.
- Verify warning badges appear for simplified/skipped routes.
- Verify cleanup removes all procedure entities and does not remove unrelated layers.

### Manual Acceptance

- Run:

```bash
python aeroviz-4d/python/preprocess_procedures.py \
  --cifp-root data/CIFP/CIFP_260319 \
  --airport KRDU \
  --include-all-rnav \
  --output aeroviz-4d/public/data/procedures.geojson
```

- Start AeroViz-4D and verify:
  - `RW05L`, `RW05R`, `RW23L`, `RW23R`, and `RW32` groups appear.
  - Each runway group can be opened/closed.
  - Each procedure can be toggled independently.
  - Final branches align visually with the correct runway end.
  - Simplified or skipped legs are visible in warnings.
  - Existing trajectories, runways, terrain, and obstacle layers still work.

## Implementation Order

1. Refactor Python output from single procedure to multi-procedure batch.
2. Generate KRDU `R05LY/R05RY/R23LY/R23RY/R32`.
3. Update frontend types and `useProcedureLayer` to handle multiple route IDs and group metadata.
4. Add `ProcedurePanel` with per-runway and per-procedure toggles.
5. Add branch toggles and warning badges.
6. Implement `CF` and `DF` geometry support.
7. Parse `P` profile records and update vertical path generation.
8. Implement missed approach `CA` and hold-related legs.
9. Implement `RF` arcs for RNAV/RNP procedures.
10. Add FAA chart cross-check notes to the generated metadata or docs.

## Risks And Guardrails

- CIFP fixed-width parsing must remain deterministic and tested; do not rely on loose string splitting for procedure records.
- Keep approximate geometry visibly labeled as approximate.
- Never silently drop a leg; every skipped/simplified leg must be represented in warnings.
- Do not claim the tunnel is official obstacle protection geometry unless it is backed by the correct procedure design criteria.
- Keep generated `procedures.geojson` reproducible so visual diffs and tests are meaningful.
