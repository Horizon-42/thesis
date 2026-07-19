# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroViz-4D: Airport 4D trajectory and terrain digital-twin visualization system for thesis research. Combines a React/TypeScript/CesiumJS frontend with Python data pipeline tools to visualize aircraft trajectories (position + time) in 3D terminal airspace.

## Repository Layout

- **aeroviz-4d/** — Main visualization app (React + CesiumJS frontend, Python CZML generator)
- **trajectory_data_process/** — Trajectory acquisition, processing, and dataset helpers
- **bc_lidar_downloader/** — BC LiDAR terrain data downloader
- **run_asd-b_fetch_and_generate.py** — Orchestrator: fetch -> normalize -> generate CZML pipeline
- **geokit/** — Shared geodesy/units package (src-layout, `pip install -e` into conda `aviation`)
- **flight_scenarios/** — Data→modeling seam (observed track → `FlightScenario`)
- **evaluation/** — File-based trajectory judging + batch metrics (geokit + stdlib only)
- **4dTrajectory/optimization/** — Optimizers, constraints, batch tooling
- **aeroviz_backend/** — Python HTTP backend (simulation / optimization / dynamics-comparison)
- **run_scenario_pipeline.py** — Batch runner: scenarios → optimize → CZML comparison + evaluation

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

# Scenario optimization batch (all 3 modes per airport when --target-type omitted)
python run_scenario_pipeline.py --jobs 6
```

## Architecture

### Frontend State & Component Structure

Global state lives in `AppContext` (context + useState, no Redux). Key state: `viewer` (CesiumJS Viewer instance), `airport` config, `selectedFlightId`, `layers` visibility toggles, `playbackSpeed`.

Components read context via `useApp()` hook. CesiumJS logic is encapsulated in custom hooks:
- `useCesiumViewer` — initializes Viewer, loads airport.json, sets camera
- `useCzmlLoader` — loads CZML data source, syncs Cesium clock
- `useRunwayLayer` / `useTerrainLayer` — data layer management
- `useDsmTerrainLayer` — loads preprocessed `.f32` heightmap tiles via `terrain/dsmHeightmapTerrain.ts`; returns `{ status, metadata, provider, error }`; controlled by `layers.dsmTerrain` toggle

UI components (ControlPanel, HUD, FlightTable) overlay on the Cesium canvas via CSS grid with `pointer-events: none`.

### Data Flow

```
OpenSky history DB → download_trajectories.py (Trajectory model, geometric altitude) → *_czml_input_*.json
    → generate_czml.py (bearing, velocity, orientation) → trajectories.czml
    → useCzmlLoader hook → CesiumJS rendering
```

Static data: OurAirports CSV → `preprocess_airports.py` → `runway.geojson`; ARINC 424 CIFP → `preprocess_waypoints.py` / `preprocess_procedures.py` → `waypoints.geojson` / procedure details.

Modeling pipeline: `*_czml_input_*.json` → `flight_scenarios` (`FlightScenario`: initial+target `GeodeticState`, `AircraftSpec`, `AeroParams`, source incl. `entry_time_utc`) → `4dTrajectory/optimization/scenario_optimization.py` (`*_states.json` + `*_eval.json`) → `build_scenario_comparison_czml.py` (3-colour comparison CZML) + `evaluation` (report JSON/HTML). `flight_scenarios` sits between the data plane and the modeling plane — depends downward on modeling primitives, imported upward by both consumers (no cycles); it's a top-level package deliberately (`4dTrajectory` isn't importable). `flight_scenarios.aircraft_for_code` OpenAP extension is WIP (left to the author).

### Key Data Formats

- **CZML**: JSON array where first element is a "document" packet (clock config), subsequent elements are entity packets with time-sampled positions via `cartographicDegrees: [secondsOffset, lon, lat, altMetres, ...]`
- **GeoJSON**: static layers (runways, waypoints, OCS surfaces)
- Airport config: `public/data/airport.json` — `{code, lon, lat, height}`
- Evaluation record (one JSON/trajectory): `{source, initial_state, target_state, final_time_s, states[], controls[]}` — controls 1:1 ZOH-aligned with states; unsolved = empty states+controls; reference records (observed track in same contract) have `controls == []`; solved records require `final_time_s == states[-1].t`

### Utility Modules

- `ocsGeometry.ts` — pure PANS-OPS obstacle clearance surface math
- `czmlBuilder.ts` — pure CZML packet construction helpers
- `utils/procedureGeoMath.ts` — the single TS geo/units module (constants imported from generated `geoConstants.json`)

## Environment

- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` (asset copying, `CESIUM_BASE_URL`)
- TypeScript strict mode (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals
- Python env: conda `aviation`; `geokit` is `pip install -e`'d
- This machine: 16 GB RAM, frequently swap-bound — memory pressure (Cesium + casadi + IDE + browser) causes UI lag independent of code changes

## Domain Context

Thesis research project. Key aviation concepts:
- **TMA** (Terminal Maneuvering Area) — controlled airspace around airports
- **OCS** (Obstacle Clearance Surface) — PANS-OPS geometry ensuring terrain clearance on approach
- **4D Trajectory** — position (lon, lat, alt) + time; the 4th dimension is scheduled arrival time
- **CTA** (Controlled Time of Arrival) — ATC-assigned time slot at a fix point

Dual purpose: thesis visualization/validation + reusable research component library.

## Coding Conventions

**Minimise defensive / patch-like code.** Prefer clear contracts over scattered guards.

- Don't sprinkle `if x is not None` / `try/except` / fallback branches for inputs that shouldn't occur. Give the parameter a sensible **default**, or make it **required** — pick one. Validate once at the boundary if truly needed; otherwise fail loudly.
- No band-aids over a root cause — fix upstream (the parser, not the consumer).
- Keep the happy path linear; one explicit assumption beats repeated `None`/empty checks.
- No silent approximations or caps: any approximation gets an explicit option + notice; any bounded coverage (top-N, sampling) is stated in output, never silent.
- Single source of truth: constants/conversions/course math defined once (geokit, `approach_constraints`, module constants) and imported everywhere; "MUST match" mirror comments only where an import is impossible (e.g. the import-light pipeline runner).

## Operational Gotchas (recurring, verified)

- **The backend does NOT hot-reload** — restart `./start_aeroviz_fullstack.sh` after backend changes. The launcher is a supervisor: restarts a dead child individually (crash-loop protection: >5 deaths within 8 s ⇒ give up); Ctrl-C/SIGTERM kills both subtrees.
- **casadi symbolic construction is NOT thread-safe** (global SXElem pools; concurrent NLP builds → heap corruption → C++ `abort()`/SIGABRT that Python can't catch). Casadi-heavy endpoints run in an isolated worker subprocess (`aeroviz_backend/isolated_backend.py`); all in-process casadi entry points serialize on `casadi_lock.CASADI_LOCK`. Toggle isolation off with `AEROVIZ_ISOLATE_SOLVER=0` (to get a native traceback).
- **IPOPT is sensitive to CasADi symbol-creation order** — `make_dynamics()` must run before NLP decision symbols are created.
- **Heading convention**: the dynamics model ψ is math-ENU (0 = East, CCW toward North; `V_east = V·cosγ·cosψ`). `approach_constraints.geometry.course_bearing` returns this convention (`atan2(Δn, Δe)`). Mixing in compass convention reads an aligned aircraft as a 90° intercept.
- **All aircraft CZML sets `forwardExtrapolationType:"HOLD"`** — `position.getValue` returns the frozen final position forever forward; any outward time-walk must stop when the position stops changing, not only on null.
- **Cesium `Clock.tick()` LOOP_STOP wrap preserves overshoot** (`currentTime = startTime + (currentTime − stopTime)`); use `clock.onStop` (fires at the stop time for both CLAMPED and LOOP_STOP) for exact end-of-playback emits — never elapsed-based heuristics.
- **CZML document clock intervals must use `iso_ms`** — second-precision `iso()` truncation made the clock stop up to 1 s before the last sample (~25–75 m phantom position error).
- **Probe bug**: `build_optimized_trajectory_playback` needs a REAL optimizer name — `simulation_mode_for_optimizer` on an unknown name selects the alpha-control mode and misreads casadi load-factor controls (fake 8–11 km "drift").
- **`geokit` is src-layout** (`geokit/src/geokit/`) because a top-level `geokit/` dir on sys.path (CWD under pytest) would shadow the installed package.
- **`aeroviz-4d/public/data` is git-ignored** (local artifacts; regenerate via preprocess scripts).
- **`.flight-ops-panel` has `backdrop-filter`** → it becomes the containing block for `position:fixed` descendants AND clips overflow; floating windows must render via React portal into `document.body`.
- **"Batch edition" seam class**: the batch callers in `scenario_optimization.py` duplicate wiring the backend HTTP path also has — several bugs (stale `_solve_iaf` unpack, trapezoidal left behind after the HS flip, missing `n_seg_per_phase`) came from updating one path and missing the other. When changing optimizer wiring, update BOTH and their seam tests.
- Pre-existing, unrelated: two `run_asd-b` orchestrator tests fail on this branch (missing `--include-transitions` / `_airport_output_dir`) — untouched.
- Stale docs (historically inaccurate, kept): `4dTrajectory/docs/direct_collocation_hermite_simpson.zh.md` §5 and `geodetic_dynamics_transport.zh.html` describe the old HS-planner + RK4-polish pipeline.

## Key Defaults & Constants (current)

- Mesh: `collocation/optimizer.py` `DEFAULT_N_SEGMENTS = 8`, `DEFAULT_N_SEG_PER_PHASE = 3` (single source; backend + batch import them; `run_scenario_pipeline.py` mirrors 8/3 with a "MUST match" comment). Frontend/backend unconstrained `n_segments` default = 10 (a different knob). Multiphase mesh = n_seg_per_phase × legs (`n_segments` doesn't apply).
- State substeps: auto per phase ≈ 3 s state step, cap 16 (`_TARGET_STATE_STEP_S`/`_MAX_STATE_SUBSTEPS`); explicit `--state-substeps`/frontend "State substeps" (0 = auto, clamp 0–64) overrides. Do NOT lower below auto (M=4 → 14.5 km rollout error); on unconstrained solves M=32 improves accuracy/optimum; constrained solves don't need big M (per-node inequality rows make big M explode solve time).
- Fitting: constrained + unconstrained default = Hermite-Simpson (`hermiteSimpsonNormalizedFullTransport`; frontend `DEFAULT_TRAJECTORY_OPTIMIZER = casadiMultiphaseNormalizedFullTransport`). `FITTING_SCHEMES`: `hs` / `trapezoidal` / `rk4` via `--fitting-type`. Trapezoidal is dynamically unfaithful on aggressive min-time floor-riding solves (5–15 km rollout drift vs HS metres); rk4 is basin-fragile there (needs M=64) — both kept for comparison studies only.
- IPOPT: `components.DEFAULT_MAX_ITERATIONS = 3000` (`ipopt.max_iter` set explicitly on all three solver constructions; request `maxIterations` reaches both backend branches). Linear solver: `AEROVIZ_IPOPT_LINSOL` (default `mumps`) + `AEROVIZ_IPOPT_HSLLIB` — HSL hook dormant (free MA27 measured 3–27× slower than MUMPS on these small NLPs; kept for a future MA57 attempt); batch speed lever is `--jobs`.
- Altitude floor: `altitude_floor_m(target) = target − 5 m` (a real operational floor min-time solves ride; was −300 m which they dove to). Transition-phase floor = min(start alt, first leg's published entry floor `_first_leg_entry_floor_m`) − margin. Rollout guard: `ROLLOUT_GUARD_MARGIN_M = 5.0`, `rollout_guard_altitude_m(target) = altitude_floor_m(target) − 5` (zero margin truncated faithful floor-riding replays on cm integration noise). `min_altitude_m` is REQUIRED on `rollout_controls`/`simulate_controls` (no silent sea-level default).
- ψ corridor: constrained solves bound the heading variable to the route's heading hull ± 90° (`_route_psi_profile`, `_PSI_CORRIDOR_SLACK_RAD`) — this killed the whole looping/crawling local-optimum family. Terminal ψ pinned on the route-unwrapped branch (`_route_unwrapped_target_psi`); per-phase heading guesses = own leg course.
- Join/passage constraints (`approach_constraints` + `collocation/optimizer.py`): the ONE forced fix passage is the pre-FAF fix, within its leg's k·RNP disc; FAC join = on-course (`fac_cross_track = 0`) with along-course distance in `[d_FAF + L_final/5, d_FAF + max_offset]` (max_offset auto = half the leg into the FAF; join guess = window middle; `max_join_offset_m=0` → the single point 1/5 before the FAF); branch-aware intercept box `|ψ_join − course_branch| ≤ 30°`; two-tier FAC alignment ±30° join→FAF, ±10° (`_FAC_ALIGN_TIGHT_DEG`) FAF→threshold; vertical glidepath window binds only `d ≤ d_faf_m`, upstream the published FAF minimum (`prefaf_floor_m`) applies. Tiny duration-split regularizer `1e-4·Σ(Tp/T_max)²` on free-time multi-phase solves. Constraint families are explicit functions dispatched from `_build`: `_terminal_pin_rows` / `_fac_join_rows` / `_prefaf_fix_rows` / `_leg_path_rows` / `_fac_alignment_rows`.
- Transition phase: prepended (unconstrained) when the start is farther than `_first_fix_join_tolerance_m` from the first fix (= first leg's k·RNP when it has one, 2 km fallback for LPV-first). Frame-anchor contract validated loudly (`segments[-1].end_ne` + `lpv.ltp_ne` at origin ±150 m). `DEFAULT_K_MARGIN` + `STANDARD_INTERCEPT_MAX_DEG` single-sourced in `approach_constraints`.
- Evaluation gates (`evaluation/thresholds.py`, regulation-derived from `docs/regulation/`, overridable): lateral ≤ **106.75 m** (FAA 8260.58D §3-1-5.c(3) Formula 3-1-1, 350 ft LPV semiwidth floor at threshold); vertical ∈ **[−3.05, +6.10] m** (§1-3-1.f(2)(b) TCH/WCH window). Gates judge the TRUE-dynamics rollout's final state. Batch metrics: solve/success rates, lateral mean/p95/max, vertical spreads, flight times; path-shape deviation vs reference = both paths resampled at 101 fractions of their own horizontal arc length.
- Playback drift guard: `playbackDriftM` on every optimize response; stderr WARNING above `PLAYBACK_DRIFT_WARN_M = 50`.
- Arrival truncation (`trajectory_data_process/arrival_segment.py`): `ENTRY_RADIUS_KM = 25` (builder `--entry-radius-km`, in (0, 30)), `ENTRY_HYSTERESIS_SAMPLES = 3`, `LOCAL_START_RADIUS_KM = 5`.
- Worker sessions: `AEROVIZ_WORKER_IDLE_TIMEOUT_S` (default 600) idle watchdog reclaims a stranded resident solver worker.
- Comparison CZML colour contract: group status lives on entity `properties.status` ∈ solved/offTarget/failed. Reference: white / dark-red (failed) / dark-amber `OFF_TARGET_REF_COLOR` (off-target); simulator/result path bakes bright yellow `OFF_TARGET_COLOR` (255,205,40) + "(off target)" name (frontend repaint skips `status=="offTarget"`); optimizer plan keeps legend orange/cyan.
- Categories manifest: `categories.json` entries carry an explicit `"constrained": bool` (frontend validator REQUIRES it; `_cons`-suffix detection deleted). Evaluation read side is manifest-ONLY: `load_records` reads a batch dir via `summary.json` roster (`results[].eval_file`); manifest-less dir / listed-missing file / empty roster raise (no glob fallback — globbing counted orphans).
- Stale-artifact hygiene (write side): `_clear_stale_records` deletes top-level `*_states.json`/`*_eval.json` at batch start; `write_reference_records` clears `references/*_reference_eval.json` first; CZML builder `clear_stale_outputs` deletes previous `comparison_*.czml` + a stale published `evaluation_report.json`. Record-filename suffixes single-sourced (`_STATES_SUFFIX`/`_EVAL_SUFFIX`/`_REFERENCE_EVAL_SUFFIX`).

## Changelog

The dated development log lives in **`docs/CHANGELOG.md`** — deliberately not loaded by default (it is long). Read it only when you need history: why a design is the way it is, when/why a default changed, what a past bug/postmortem looked like, or which outputs a change made stale.

Maintenance convention:
- **Append new dated entries to `docs/CHANGELOG.md`** (newest first, `### YYYY-MM-DD — title`), not here.
- When a change produces a durable fact (a gotcha, a default, a contract), also update "Operational Gotchas" / "Key Defaults & Constants" above — those, not the changelog, are what every session sees.
- Keep the Open Items list below current: add items as they arise, delete them when resolved.

## Open Items (current as of 2026-07-19)

- **All optimization batches / evaluation reports / comparison CZMLs are STALE** — built before arrival truncation, the altitude-floor/rollout-guard fixes, and the HS fitting flip. Re-run `python run_scenario_pipeline.py --jobs 6` before comparing anything; afterwards re-examine the `runway_cons` off-target populations (e.g. KRDU RW32's 77/200) for the wrongly-truncated family.
- Approach view: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view (Observe-with-comparison plots neither source); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case.
- Per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (~926 m at k=0.5) instead of ~278 m (RNP 0.3).
- CIFP leg speed restrictions not extracted (no speed-bearing data source in the dataset yet; the canonical `speedMaxKt` field is ready).
- HSL linear-solver hook dormant (free MA27 measured slower than MUMPS); revisit with an MA57 academic license.
- Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07 approach-view changes were verified via tests/tsc/build but not re-checked in-browser.
