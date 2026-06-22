# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroViz-4D: Airport 4D trajectory and terrain digital-twin visualization system for thesis research. Combines a React/TypeScript/CesiumJS frontend with Python data pipeline tools to visualize aircraft trajectories (position + time) in 3D terminal airspace.

## Repository Layout

- **aeroviz-4d/** — Main visualization app (React + CesiumJS frontend, Python CZML generator)
- **trajectory_data_process/** — Trajectory acquisition, processing, and dataset helpers
- **bc_lidar_downloader/** — BC LiDAR terrain data downloader
- **run_asd-b_fetch_and_generate.py** — Orchestrator: fetch -> normalize -> generate CZML pipeline

## Build & Dev Commands

### Frontend (aeroviz-4d/)

```bash
cd aeroviz-4d
npm install
npm run dev                          # Vite dev server with HMR
npm run build                        # tsc + vite production build
npm test                             # Vitest (watch mode)
npx vitest run                       # Single run, no watch
npx vitest run src/utils/__tests__/ocsGeometry.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:dsm-tiles              # Build 3D Tiles from GeoTIFF
npm run build:dsm-heightmap-terrain  # Generate heightmap terrain tiles
```

### Python (aeroviz-4d/python/)

```bash
pip install -r aeroviz-4d/python/requirements.txt
python -m pytest aeroviz-4d/python/tests/test_generate_czml.py -v
python -m pytest aeroviz-4d/python/tests/ --cov=. --cov-report=html
```

### Data Pipeline (end-to-end)

```bash
# Full pipeline: fetch live data → normalize → generate CZML
python run_asd-b_fetch_and_generate.py --airport CYYC --mode live

# Reuse existing raw JSON (skip fetch, run normalization + generation)
python run_asd-b_fetch_and_generate.py --input-json trajectory_data_process/outputs/cyyc_raw_*.json

# Skip straight to CZML generation from already-normalized data
python run_asd-b_fetch_and_generate.py --input-json trajectory_data_process/outputs/cyyc_czml_input_*.json
```

## Architecture

### Frontend State & Component Structure

Global state lives in `AppContext` (context + useState, no Redux). Key state: `viewer` (CesiumJS Viewer instance), `airport` config, `selectedFlightId`, `layers` visibility toggles, `playbackSpeed`.

Components read context via `useApp()` hook. CesiumJS logic is encapsulated in custom hooks:
- `useCesiumViewer` — initializes Viewer, loads airport.json, sets camera
- `useCzmlLoader` — loads CZML data source, syncs Cesium clock
- `useRunwayLayer` / `useTerrainLayer` — data layer management
- `useDsmTerrainLayer` — loads preprocessed `.f32` heightmap tiles via `terrain/dsmHeightmapTerrain.ts`; returns `{ status, metadata, provider, error }` so consumers can display terrain info or manage toggling; controlled by `layers.dsmTerrain` toggle in the main app

UI components (ControlPanel, HUD, FlightTable) overlay on the Cesium canvas via CSS grid with `pointer-events: none`.

### Data Flow

```
OpenSky history DB → download_trajectories.py (Trajectory model, geometric altitude) → *_czml_input_*.json
    → generate_czml.py (bearing, velocity, orientation) → trajectories.czml
    → useCzmlLoader hook → CesiumJS rendering
```

Static data follows a similar pattern: OurAirports CSV → `preprocess_airports.py` → `runway.geojson`; ARINC 424 CIFP → `preprocess_waypoints.py` → `waypoints.geojson`.

### Key Data Formats

- **CZML**: JSON array where first element is a "document" packet (clock config), subsequent elements are entity packets with time-sampled positions via `cartographicDegrees: [secondsOffset, lon, lat, altMetres, ...]`
- **GeoJSON**: Used for static layers (runways, waypoints, OCS surfaces)
- Airport config: `public/data/airport.json` — `{code, lon, lat, height}`

### Utility Modules

- `ocsGeometry.ts` — Pure math for PANS-OPS obstacle clearance surface computation (no side effects)
- `czmlBuilder.ts` — Pure CZML packet construction helpers

## Environment

- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` which handles Cesium asset copying and `CESIUM_BASE_URL` setup
- TypeScript strict mode is enabled (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals enabled

## Domain Context

This is a thesis research project. Key aviation concepts in the code:
- **TMA** (Terminal Maneuvering Area) — controlled airspace around airports
- **OCS** (Obstacle Clearance Surface) — PANS-OPS geometry ensuring terrain clearance on approach
- **4D Trajectory** — aircraft position (lon, lat, alt) + time; the "4th dimension" is the scheduled arrival time
- **CTA** (Controlled Time of Arrival) — ATC-assigned time slot at a fix point

The project serves dual purposes: thesis visualization/validation, and reusable research component library.

## Coding Conventions

**Minimise defensive / patch-like code.** Prefer clear contracts over scattered guards.

- Don't sprinkle `if x is not None` / `try/except` / fallback branches to handle inputs that shouldn't occur. Instead: give the parameter a sensible **default**, or make it **required** — pick one. Validate once at the boundary if validation is truly needed; otherwise let it fail loudly (consistent with "fail loudly if data is missing").
- No band-aids that paper over a root cause. If something is wrong upstream, fix it upstream rather than adding a downstream workaround (e.g. fix the parser, not the consumer).
- Keep the happy path linear and readable; a single explicit assumption beats many repeated `None`/empty checks for the same condition.

## Changelog

### 2026-06-22 — Selectable collocation defect schemes (trapezoidal / Hermite-Simpson / RK4)

Made the direct-collocation defect "fitting equation" pluggable, so the optimiser, backend, and frontend can pick among three transcription schemes of increasing order. This grew out of the RW05L "terminal offset" investigation: the offset was **not** the sampling step nor a playback mismatch (the optimiser's own RHS and the playback integrator drift identically), but the **collocation accuracy** of an aggressive near-floor/near-stall solution at the default state density — i.e. a discretisation issue, of which the defect scheme is one lever (state density is the other).

- `4dTrajectory/optimization/casadi_direct_collocation_optimizer.py` — added `trapezoidal_defect_expr` (linear state, order 2) and `rk4_defect_expr` (RK4 integral defect, order 4, playback-consistent by construction; reuses `rk4_step_expr`), alongside the existing `hermite_simpson_defect_expr` (cubic state, order 4). Registry `_DEFECT_SCHEMES` + `_DEFAULT_SCHEME = "hermiteSimpson"`. `_build_collocation_decision` takes a `defect_fn`; both NLP builders + the optimiser class take `collocation_scheme` (validated once in `__init__`). Note: Hermite-Simpson is **4th-order** (Simpson quadrature), same order as RK4 — they differ in *construction* (implicit polynomial collocation vs explicit shooting), not order; trapezoidal is the only 2nd-order one.
- `aeroviz_backend/optimization_backend.py` — `DIRECT_COLLOCATION_SCHEMES` maps optimizer names (`casadiDirectCollocation` [=HS default], `…Trapezoidal`, `…HermiteSimpson`, `…Rk4`) to schemes; `SUPPORTED_OPTIMIZERS`/`CASADI_OPTIMIZERS` and the free-time dispatch + `make_optimizer` updated. **No optimizer code/handlers removed** (legacy ones still served). `trajectory_playback.simulation_mode_for_optimizer` now prefix-matches `casadiDirectCollocation*`.
- `aeroviz-4d/src/components/PilotPanel.tsx` + `pilot/trajectoryOptimizationClient.ts` — Optimizer dropdown now lists only the three direct-collocation scheme variants (Hermite-Simpson default / Trapezoidal / RK4); the legacy optimizers were dropped from the **list** but remain valid in the type/response-parser and on the backend. `trajectoryOptimizerSimulationMode` prefix-matches the variants. Removed the now-dead variable-time arrival-disable flag.
- `4dTrajectory/optimization/collocation_scheme_comparison.py` (new) — standalone apples-to-apples comparison (replay-vs-target accuracy ladder; trapezoidal ~5 m vs HS/RK4 sub-metre on a feasible A320 approach). Tests: `test_casadi_direct_collocation_optimizer.py` gains scheme-registry / defect-zero / unknown-scheme / accuracy-ladder cases; backend gains a name→scheme dispatch test. 256 frontend + 53 python pass; tsc clean.
- `4dTrajectory/docs/collocation_schemes.zh.html` (new) — beginner tutorial: optimization basics (state/control/dynamics/objective/constraints → transcription → NLP → **defect**), then the three schemes with intuition + formulas, with **live in-browser** demos (one-interval linear-vs-cubic fit; a log-log convergence plot showing empirical slopes ≈2 for trapezoidal and ≈4 for HS/RK4 on the toy ODE ẏ=−y²). Complements the technical reference `direct_collocation_hermite_simpson.zh.md`.

### 2026-06-22 — Optimized trajectory plays as backend CZML on the Cesium clock (was fixed-dt frontend replay)

"Trajectory Play" no longer re-simulates the optimized trajectory on the frontend with a fixed-dt `setInterval` loop (one `/simulation/step` per ~120 ms tick). Instead the backend rolls the optimizer's controls forward **once** and returns a CZML the frontend loads into a `CzmlDataSource` and plays on Cesium's own clock/timeline — exactly like a downloaded trajectory. The trail grows behind the aircraft (the aircraft sits at its leading edge) and is coloured **per control segment** by segment order (a smooth blue→red gradient — adjacent segments continuous yet distinct).

- `aeroviz_backend/trajectory_playback.py` (new) — `build_optimized_trajectory_playback()` rolls the N piecewise-constant controls forward through the **same** geodetic integrator the live simulator uses (`make_geodetic_simulator(aircraft, mode)`), so the replay matches the optimizer to sub-mm (CLAUDE.md 2026-06-21). Emits `{epochIso, multiplier, czml, samples}`: a document + aircraft + trail CZML, plus a dense sample series (state + control + aero) for the live readout. The trail is **one short polyline per sample interval**, each made available only once the aircraft has passed its far end (ms-precision `availability`), so it grows smoothly and trails the aircraft. Colour is by control-segment order via `_segment_color_rgba`. The aircraft packet has **no `orientation`** — the frontend sets it. Rollout truncates (not raises) if a pathological replay leaves the envelope, since it is a viz aid.
- `aeroviz_backend/optimization_backend.py` — `optimize()` attaches `playback` to its response (additive; existing keys unchanged, so the prior tests still pass).
- `aeroviz-4d/src/pilot/trajectoryOptimizationClient.ts` — added `TrajectorySample` / `TrajectoryPlayback` types and parsing of the new `playback` field.
- `aeroviz-4d/src/hooks/useOptimizedTrajectoryPlayback.ts` (new) — loads the CZML, drives `viewer.clock` (LOOP_STOP, multiplier from the doc), camera-follows the optimized aircraft, and on each clock tick samples the rollout (throttled ~12 Hz) to feed the live readout via `sampleTrajectoryAt()`. Sets the aircraft `orientation` from the sampled state (heading/pitch/**bank**) with the **same convention as the live Pilot aircraft** (`headingPitchRollQuaternion`, heading `-ψ`, pitch `γ+α`, roll `-μ`) — fixes the earlier CZML orientation that used ψ as a compass bearing and ignored bank.
- `aeroviz-4d/src/components/PilotPanel.tsx` — removed the fixed-dt trajectory replay effect + its refs; `Play/Pause/Reset` now drive the Cesium clock; the hand-built `usePilotAircraft` is gated to **Pilot** mode (Trajectory Play shows the CZML aircraft + coloured polylines instead). The live readout in Trajectory Play is sampled from the rollout, not stepped.
- Manual **Pilot** mode is unchanged (it is interactive and cannot be precomputed). Tests: 33 backend (6 new for the playback builder) + 255 frontend (PilotPanel + client tests updated for the new contract) pass; tsc + vite build clean.

### 2026-06-22 — Dense-state direct collocation (control N, state N·M); remove polish

Fixed the optimizer→playback target mismatch (km-scale on long, coarse-mesh approaches). Root cause: matching the *continuous* dynamics (geodetic ≈ re-anchored RK4) is necessary but not sufficient — the optimiser's *discrete* operator (Hermite-Simpson on a coarse mesh, h~30 s) differs from the playback's (fine RK4), so replaying the raw controls drifted. Fix: collocate the **state** on a finer grid while keeping **control** coarse.

- `4dTrajectory/optimization/casadi_direct_collocation_optimizer.py` — both NLP builders now take `sub_steps` (M): control is piecewise-constant over N segments, state collocated on N·M Hermite-Simpson sub-intervals (`_build_collocation_decision`). M auto-selected from the horizon (`select_state_substeps`, ~3 s state step, capped at 16). External contract unchanged: returns N controls + N segment-endpoint states. **The multiple-shooting polish (`_polish_with_multiple_shooting`, `_get_polisher`, `solution_to_initial_guess`, the `polish` arg, the lazy `casadi_optimizer` import) is removed** — the dense-state raw solution is playback-consistent.
- Verified: playback drift drops from km-scale (M=1, h~30 s) to <1 m; backend `optimize()` lands exactly on target; 124 tests pass; added a long-horizon coarse-control playback regression test.
- Why not just raise nSegments: that refines control too, causing the convergence/"wrinkle" pathology (old doc §5.4). Refining only the state mesh avoids it (control DOF stays 3N).
- Note: `4dTrajectory/docs/direct_collocation_hermite_simpson.zh.md` §5 and `geodetic_dynamics_transport.zh.html` still describe the old HS-planner + RK4-polish pipeline and are now historically inaccurate.

### 2026-06-22 — Fix CIFP transition-altitude misparse (bogus 18000 ft initial fixes)

RNAV initial fixes with no published crossing altitude (e.g. KRDU R32 CONCA/SINNO) were being placed at **18000 ft** because the CIFP parser read the ARINC 424 **Transition Altitude** field (a procedure-wide constant) as the leg's crossing altitude. Selecting such an IF as the optimizer start made the problem infeasible (~10° required descent) → `Maximum_Iterations`/`Infeasible`. Full root cause + evidence + regenerate command in `aeroviz-4d/docs/33-cifp-transition-altitude-misparse-postmortem.md`.

- `aeroviz-4d/python/cifp_parser.py` — production `parse_procedure_legs` (cifparse adapter) no longer falls back to `trans_alt`; sets the altitude qualifier from `alt_desc` (`+`→atOrAbove, etc.). Legacy `parse_leg_altitude_ft` scans only through Altitude 2 (drops the `line[94:99]` transition-altitude read); added `parse_leg_altitude_qualifier`.
- `aeroviz-4d/src/data/rnavInitialFixCandidates.ts` — `altitudeFtForInitialFix` returns a leg's own altitude only from a finite **positive** published value (never the transition altitude or `fix.elevationFt`). An IF with no own altitude is **not discarded**: `derivedInitialFixAltitudeFt` interpolates one from the nearest published fix(es) on the branch (so feeder IFs like CONCA derive ~3400 ft from the downstream NOSIC leg and stay selectable); dropped only if no neighbour has an altitude.
- Tests updated to assert the corrected behavior (two prior tests had locked in `== 18000`); added production-path + frontend regression tests. KRDU procedure data regenerated. Note: the ready-made `cifparse`/`arinc424` packages are only used in `validate_cifp_parser_packages.py` (cross-check), and their altitude extraction had the same `trans_alt` fallback — switching libraries would not have fixed this.

### 2026-06-21 — Geodetic continuous dynamics for direct collocation (replaces fixed-ENU + polish)

Replaced the fixed-ENU transcription in the direct-collocation optimiser with a single **continuous geodetic RHS**, so the optimiser and the playback simulator now share one continuous vector field. The coordinate transformation that the re-anchored ENU stepper did discretely (per-step frame swap) is folded into the continuous RHS as WGS84 curvature factors plus transport-rate terms — no tangent plane, no fixed-frame curvature error, and the multiple-shooting polish is no longer needed.

- `aerodynamic_model/casadi_simulator.py` — added `make_geodetic_dynamics_model(include_transport=True)` (continuous point-mass RHS in `(lat, lon, h, V, psi, gamma)`, lat/lon in radians; position kinematics via `R_M`/`R_N`; transport terms on `psi_dot`/`gamma_dot`) and `make_geodetic_step_integrator` (RK4 stepper, degrees externally). **Existing `make_dynamics_model` / `make_geo_step_from_enu_integrator` left untouched.**
- `4dTrajectory/optimization/casadi_direct_collocation_optimizer.py` — collocates directly on geodetic state (radians), boundary is now just deg↔rad; ENU anchor machinery removed. **Polish is disabled by default (`optimize_free_time(polish=False)`)** but the polisher code is kept temporarily for verification — *remove it once the geodetic path is fully validated*.
- `4dTrajectory/optimization/geodetic_vs_reanchored_error.py` (new) — 5 km comparison of the two discrete systems. Result: geodetic+transport tracks the re-anchored RK4 playback to ~0.3 mm; dropping transport drifts ~2.9 m / 1.3 m alt / 0.04° over 5 km (= the transport rate).
- `4dTrajectory/docs/geodetic_dynamics_transport.zh.html` (new) — interactive HTML (MathJax + Plotly 3D) explaining the geodetic RHS and the transport terms (meridian convergence + curvature pitch), with the 5 km validation.
- Tests updated; backend `optimize()` path verified end-to-end (terminal state matches target). Convergence robustness on aggressive cold-start targets is unchanged from the old optimiser (parity confirmed).

### 2026-04-20 — Finish OCS geometry and add final-approach OCS layer

Completed the PANS-OPS final-approach Obstacle Clearance Surface (OCS) pipeline: filled in the TODO in `src/utils/ocsGeometry.ts`, wrote full unit-test assertions, and added a new `useOcsLayer` hook that derives FAF→threshold pairs from `procedures.geojson` and renders three semi-transparent Cesium polygons per route (red primary + two orange 7:1-slope secondary panels, `perPositionHeight: true` for the slope to show).

- `src/utils/ocsGeometry.ts` — implemented `buildFinalApproachOCS` (bearing → perpendiculars → primary trapezoid → secondary outer edges with `faf.altM − secondaryWidthM/7` drop at FAF and `threshold.altM` at the runway end). 13/13 unit tests pass.
- `src/hooks/useOcsLayer.ts` (new) — dual-useEffect pattern matching `useObstacleLayer`; primary half-width pulled from the route's tunnel descriptor (`tunnel.lateralHalfWidthNm × 1852`), falls back to 150 m.
- `src/components/CesiumViewer.tsx` — activated `useOcsLayer()`.
- `src/components/ControlPanel.tsx` — added `ocsSurfaces` to `ACTIVE_LAYER_KEYS` so the toggle renders.
- `docs/03-ocs-geometry.zh.md` (new) — Chinese tutorial with the flat-earth math derivation, a worked KRDU R05LY example, the altitude-provenance section (geometry altitude vs MCA and how to switch), and a concepts clarifier for OCS vs OCH vs MCA.

Altitudes are currently read from the LineString z-values (i.e. CIFP `geometryAltitudeFt × 0.3048`). Switching to MCA (`altitudeFt`) is a one-function change documented in `docs/03-ocs-geometry.zh.md §5.6`.

### 2026-04-20 — Add FAA DOF obstacle visualization layer

Added end-to-end pipeline for rendering FAA Digital Obstacle File (DOF) obstacles as 3D cylinders in CesiumJS. Obstacles are color-coded by type (TOWER=red, BLDG=steelblue, WINDMILL=green, etc.) and positioned with `HeightReference.RELATIVE_TO_GROUND` so they sit on terrain.

- `python/preprocess_obstacles.py` — parses fixed-width DOF `.Dat` files, filters by haversine radius (default 20 km / ~10.8 NM to cover the approach corridor), outputs `obstacles.geojson`
- `useObstacleLayer` hook — loads GeoJSON, creates cylinder entities with AGL-height labels; follows `useWaypointLayer` dual-useEffect pattern
- Added `"obstacles"` to `LayerKey` with toggle in `ControlPanel`
- DOF data documentation at `data/DOF/README.md`

Usage: `python preprocess_obstacles.py --input <DOF .Dat> --airport`

### 2026-04-19 — Refactor DSM terrain into reusable hook

Rewrote `useDsmTerrainLayer` to use the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`) instead of decoding raw GeoTIFF in the browser. The hook now returns `{ status, metadata, provider, error }` and can be dropped into any page.

- `DsmTerrainDemoPage` delegates terrain loading to the hook (keeps its own overlay/camera logic)
- `CesiumViewer` wires the hook so DSM terrain is available in the main flight view
- Added `dsmTerrain` to `LayerKey` with a toggle in `ControlPanel`
