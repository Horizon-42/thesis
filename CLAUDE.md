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

### 2026-06-23 — Dynamics Comparison: per-system aircraft models + fix stranded START preview

- **Colored aircraft model per trajectory** — each system's CZML entity now carries an aircraft `model` (`/models/aircraft.glb`) tinted with the system colour (`colorBlendMode: MIX` + matching silhouette), replacing the plain point marker; the reference B is slightly larger. The frontend hook sets each model's orientation with `VelocityOrientationProperty(entity.position)` so the nose points down its own path (the CZML carries position only). Paths/labels unchanged.
- **Fix: the START preview no longer stays stranded at the origin during a Compare run** — Compare never sets `snapshot`, so the `previewVisible` guard (`… && !snapshot`) never hid the static placement aircraft; added `&& !isComparisonPlaybackActive` so the START marker shows only while setting up and disappears once the comparison loads/plays.
- 48 backend + 273 frontend tests, tsc + vite build pass (the CZML entity test now asserts a colored model whose colour matches the path).

### 2026-06-23 — Dynamics Comparison: custom start state, run history, backend-averaged history

Three additions to the Compare mode (all averaging/persistence math on the backend).

- **Custom start state like trajectory mode** — Compare now sources RNAV initial fixes too: the rnav-candidate fetch effect runs in `comparison` mode (not just `trajectory`), a runway `<select>` in the Compare settings row picks the source runway, and the initial-state overlay shows the RNAV-fix dropdown in Compare. (The field editor + place-on-map already worked, being shared.) An empty RNAV list is only an error in trajectory mode (optional in Compare).
- **Run history (`aeroviz_backend/dynamics_comparison_history.py`, new)** — each `/dynamics-comparison/run` persists one JSON record (`meta` + `chart`) under `dynamics_comparison_history/` (git-ignored); `run()` returns `historyCount`. Unique filenames per run (threading server); a lock guards clear-all.
- **Backend-averaged history + button** — `POST /dynamics-comparison/history/average` resamples every stored run onto one common distance grid (0 .. the shortest run's range, linear `np.interp`) and averages per system/field (and each run's final value) — **all server-side**; returns a chart in the same shape the run uses. `GET /dynamics-comparison/history` (count) and `POST /dynamics-comparison/history/clear` round it out. Frontend: *Average history (N)* / *Clear history* buttons; the averaged chart reuses `DynamicsComparisonCharts` (new `subtitle` prop, `chartMode: "run" | "average"`). Client gains `fetchDynamicsComparisonHistoryCount` / `averageDynamicsComparisonHistory` / `clearDynamicsComparisonHistory` + `historyCount` on the run result.
- Tests: backend `TestDynamicsComparisonHistory` (count/average-spans-shortest/clear/no-history) + the existing run test now uses a temp history dir (no repo writes) + 3 new HTTP route tests; frontend 6 new client tests. **273 frontend + 48 backend + 67 optimizer python pass; tsc + vite build clean; no repo pollution.**

### 2026-06-23 — Hybrid local-ENU cold-start + geodetic free-time; whole-flow timing log

The direct-collocation free-time solve already cold-starts itself by first solving the **fixed-time** NLP (at `max_duration`) to get a dynamically-feasible seed, then shrinking `T` along the feasible manifold. Until now that cold-start solve used the **same** dynamics the caller asked for. This adds a **hybrid** where the cold start runs a cheaper/more-robust dynamics than the free-time refinement, plus timing of the whole optimisation pipeline (not just one solve).

- **`cold_start_scheme` on `CasadiDirectCollocationOptimizer`** (`4dTrajectory/optimization/casadi_direct_collocation_optimizer.py`) — when set, the **fixed-time solver** (the cold-start/warm-start seed in `_build_free_time_initial_guess` → `_solve_fixed_time_raw`, and `optimize_trajectory`) is built with `cold_start_scheme`, while the **free-time solver** keeps `collocation_scheme` (the "rhs dynamics"). Every scheme shares the **same decision-vector layout** (control 3N, state 6·N·M), so the cold-start raw solution drops straight in as the free-time seed (`x0 = w_fixed + [T]`); verified the two NLPs are 216 vs 217 vars (= +T) and align. `None` (default) → both solves share `collocation_scheme` (original behaviour). Validated in `__init__`.
- **New optimizer `casadiDirectCollocationLocalEnuColdStart`** (`aeroviz_backend/optimization_backend.py`) — free-time = geodetic `hermiteSimpson`, cold-start = fixed local-ENU (`localEnu`). A new `DIRECT_COLLOCATION_COLD_START_SCHEMES` maps the name → cold-start scheme; `make_optimizer` passes it through (`.get(name)` → `None` for every existing name, so they are unchanged). The local-ENU fixed-tangent dynamics is flat/cheap and converges robustly from the linear-interp guess, seeding the accurate geodetic free-time refinement.
- **Whole-flow timing → server log** — `optimize_free_time` records `last_solve_timings` (`coldStartS` / `freeTimeSolveS` / `solveTotalS`); the backend times the **entire** `optimize()` (NLP build + solve + playback rollout) and writes one breakdown line to **stderr** via `log_optimization_timing` (e.g. `[aeroviz-backend] optimization timing optimizer=… build=…s coldStart=…s freeTime=…s solve=…s playback=…s total=…s`). The previous log measured only a single lumped solve. **The response JSON is unchanged** (timing is log-only).
- **Frontend** — exposed as a fourth **Dynamics** option *“Geodetic RHS (local-ENU cold start)”* (`OptimizerDynamics = "geodeticColdStart"`). It composes a single optimizer (Hermite-Simpson free-time), so its **Fitting** axis is locked (`validFittingsForDynamics("geodeticColdStart") → ["hermiteSimpson"]`); `COMBO_TO_OPTIMIZER` / `OPTIMIZER_TO_COMBO` / `readOptimizer` updated (`trajectoryOptimizationClient.ts`, `PilotPanel.tsx`).
- Tests: optimizer (cold-start validation + a real local-ENU→geodetic solve that reaches the target with `last_solve_timings` populated); backend (name→cold-start mapping + timing-log emission); frontend (round-trip + locked-fitting hybrid). All python + frontend suites pass.

### 2026-06-23 — Dynamics Comparison mode (Pilot panel): play the 4-way dynamics study with deviation charts

Added a third Pilot-panel mode, **Dynamics Compare**, parallel to *Pilot* and *Trajectory Play*. It flies the start state forward under one **parameterised constant control** four ways — the same four systems as the 30 km study — replays them on Cesium's clock as four colored, hideable paths, and pops up a deviation chart (horizontal / altitude / heading / final-speed error vs the reference B).

- **Reusable core** — `4dTrajectory/optimization/dynamics_comparison.py` (new) extracts `compare_dynamics(start, control, aero_params, duration, dt, anchor_geo, …)` from the 30 km script: integrates A (fixed tangent ENU @ `anchor_geo`), B (per-step re-anchored = reference), C (geodetic RHS +transport), D (geodetic RHS no-transport) with one shared RK4 step, returning the full per-step geodetic paths + cumulative distance + time. Helpers `even_sample_indices` / `error_series` (A/C/D vs B: horiz, alt, head, speed). `dynamics_comparison_30km.py` now calls it (anchor = target, 30 km range stop) — output byte-identical, verified. **The Compare mode anchors A at the start** (common origin → diverge outward), the study keeps anchoring at the target.
- **Backend endpoint** — `aeroviz_backend/dynamics_comparison_backend.py` (new) `DynamicsComparisonBackend.run(payload)` reads `initialState` + constant `control` (thrustN/bankDeg/loadFactor) + `durationS`/`dtS`, calls `compare_dynamics`, and returns `{systems, playback:{czml,multiplier}, chart}`. The CZML has one entity per system (`dyncmp-{A,B,C,D}`: time-sampled position + colored `path` + point + label) so the frontend can hide any one; B is the reference (near-white), A warm, C/D cool. Doc multiplier targets ~40 s wall-time. Wired as `POST /dynamics-comparison/run` in `http_server.py` (+ `AeroVizBackendApp` third backend). All numbers coerced to plain floats (no numpy leaking into `json.dumps`).
- **Frontend** — `src/pilot/dynamicsComparisonClient.ts` (types + `runDynamicsComparison`), `src/hooks/useDynamicsComparisonPlayback.ts` (loads the multi-system CZML, drives `viewer.clock`, shows/hides entities from `hiddenKeys`, optional camera-follow of B), `src/components/DynamicsComparisonCharts.tsx` (2×2 SVG line charts over distance + final-value table incl. speed; interactive legend toggles). `PilotPanel.tsx` gains the 3-way mode switch, a constant-control + duration/dt input row, Compute/Play/Pause/Reset, an in-panel legend with per-trajectory hide checkboxes, and the charts window. CSS in `index.css`.
- **Charts are a centred, draggable floating window** (`.dyncmp-charts-overlay`) **rendered via a React portal into `document.body`** — NOT inside the panel. This is required: the `.flight-ops-panel` ancestor has `backdrop-filter`, which makes it the containing block for `position:fixed` descendants AND clips overflow, so a fixed window rendered inside it is pinned to (and clipped by) the panel rather than the viewport. The portal lets it centre on the page. It auto-opens dead-centre after Compute, has a fixed title-bar drag handle (pointer-capture drag, clamped to the viewport) over a scrollable body, and is closable (Close) + re-openable (the panel's *Show/Hide charts* toggle); reopening recentres.
- Tests: backend `test_dynamics_comparison_backend.py` (7) + http route test; frontend `dynamicsComparisonClient.test.ts` + `DynamicsComparisonCharts.test.tsx` (incl. centred-default + drag-to-position). Existing `PilotPanel.test.tsx` mocks the new hook/client. **267 frontend + 42 backend + 65 optimizer python tests, tsc + vite build all pass; 30 km study output unchanged.**
- **Review-fix pass (xhigh code-review):** rollout now computes-then-decides so it never records a sub-surface or non-finite sample (stops on any system below ground / NaN); `even_sample_indices` is endpoint-inclusive linspace (≤ MAX_SAMPLES, was off-by-one); integrator cache uses double-checked locking (ThreadingHTTPServer); backend returns `requestedDurationS` and the panel shows a truncation note when the flight hits ground early. Frontend: follow drops to a visible system when B is hidden; a new effect syncs `clockRange` to the auto-replay toggle after load; Reset is gated to active playback; charts memoize `visibleKeys`/lines and `MetricChart` is `React.memo` (no recompute while dragging); final-table uses magnitude formatting for horiz/heading (no misleading `+`/`-0.0`); dead `?? grey` colour fallback removed; mode switch uses `role="group"`+`aria-pressed`; the airport-reset effect reuses `clearComparisonPlayback()`; the charts test restores `window.innerWidth`. **Reuse:** shared `aeroviz_backend/czml_common.py` (epoch/iso/document-packet) and `aeroviz-4d/src/pilot/responseValidators.ts` now back both backends/clients. (Chart tick math left local — the runway panel's `niceTickStep` floors differently and unifying would risk it.)

### 2026-06-23 — Optimizer = dynamics × fitting; shared stall model; review fixes

Reworked the collocation scheme system so the **fitting (transcription) is orthogonal to the dynamics**, plus a consistency fix and review cleanups.

- **`localEnu` is now a CONTINUOUS dynamics, collocatable with any fitting.** A fixed ENU tangent frame has a continuous RHS (the flat point-mass dynamics), so — like the geodetic RHS — it takes trapezoidal / Hermite-Simpson / RK4, not just shooting. `_DEFECT_SCHEMES` entries are now `(make_dynamics, make_defect)`; the localEnu defect converts the geodetic nodes into the target-anchored ENU frame (`geodetic_state_to_enu_expr`, new in `casadi_simulator.py`) and applies the chosen fitting on the flat RHS in ENU coords. Only `reanchoredEnu` stays shooting-only (it re-anchors per step → discrete). New schemes `localEnuTrapezoidal` / `localEnuHermiteSimpson` / `localEnu` (= shooting); backend + frontend (validFittingsForDynamics: localEnu → all three). The frontend Dynamics × Fitting dropdowns now offer all three fittings for Local ENU. **Note:** `make_dynamics()` MUST run before the NLP decision symbols are created — IPOPT's solve is sensitive to CasADi symbol-creation order, so the builders resolve dynamics first, then bind the defect after the symbols.
- **Shared stall model `aero_params_for_aircraft(aircraft)`** (`casadi_simulator.py`) — mass-based `Cl_max` (A320≈2.7), used by BOTH the optimiser and the playback (`CasadiSimulator`) and the test/script fixtures. Previously the optimiser used `Cl_max=2.7` while the playback used the default `1.5`, so optimized trajectories replayed ~1.6 km off; now they share one stall model (this also raises the A320 stall Cl_max in casadi-mode Pilot sim from 1.5 to 2.7).
- Review fixes: `collocation_scheme_comparison.py` no longer KeyErrors on a new scheme (orders has all entries + `.get`); the dead `reanchored_enu_defect_expr` alias removed (the shared fn is `enu_step_defect_expr`); `make_local_enu_step_integrator` and `geodetic_state_to_enu_expr` share the forward conversion; the localEnu test now exercises the dynamics (conversion round-trip + solves with every fitting). 57 python + 258 frontend tests + tsc pass.

### 2026-06-22 — Fixed local-ENU dynamics: 30 km error study + `localEnu` optimizer scheme

Two related additions, both about the *fixed local-tangent ENU* frame (the transcription the geodetic RHS replaced):

- **Fixed-ref ENU stepper** — `aerodynamic_model/casadi_simulator.py` gains `make_local_enu_step_integrator()`: one RK4 step in a SINGLE ENU tangent frame anchored at a **given** `ref_geo` (not re-anchored at the current point). With `ref_geo` = current point it reduces exactly to `make_geo_step_from_enu_integrator` (verified to 1e-17).
- **30 km comparison** — `4dTrajectory/optimization/dynamics_comparison_30km.py` (new) flies one trajectory (documented start 30 km SW of target, constant control) four ways and measures error vs the per-step re-anchored reference (B): **(A) fixed local-ENU @ target ≈ 335 m horiz**, **(D) RHS no-transport ≈ 145 m / −69 m alt / 0.14°**, **(C) full RHS +transport ≈ 0.03 m** (validates the RHS = re-anchored). Writes `dynamics_comparison_30km_data.json`; `4dTrajectory/docs/dynamics_comparison_30km.zh.html` (new) documents the method (initial state, control, reference, error metrics) and plots the results.
- **`localEnu` optimizer scheme** — fixed-ref ENU stepper anchored at the **target** as a shooting defect, in the same dense-state transcription as the others. `_DEFECT_SCHEMES` gains `localEnu` → `(enu_step_defect_expr, "localEnuStep")`; the shared `enu_step_defect_expr` (renamed from `reanchored_enu_defect_expr`, alias kept) serves both ENU schemes; `_make_scheme_dynamics` builds the stepper and `_bake_local_enu_ref` binds the target as the fixed anchor in each builder. Backend name `casadiDirectCollocationLocalEnu`. It is a deliberately-degraded LOCAL approximation for comparison — its solution replayed shows the same fixed-tangent drift as system (A).
- **Frontend optimizer UI split into two dropdowns** — *Dynamics* (geodetic RHS / re-anchored ENU / local ENU @ target) × *Fitting* (Hermite-Simpson / trapezoidal / RK4-shooting), composing into the single optimizer string the backend already takes (no backend change). Polynomial fittings need a continuous RHS, so picking an ENU dynamics collapses the fitting to shooting-only. Helpers `optimizerToParts` / `partsToOptimizer` / `validFittingsForDynamics` in `trajectoryOptimizationClient.ts`. 56 python + 258 frontend tests + tsc pass.

### 2026-06-22 — Selectable NLP solver backend (ipopt / sqpmethod) + benchmark

Added a `solver_backend` switch to `CasadiDirectCollocationOptimizer` (and both NLP builders) so the interior-point default (`ipopt`) can be swapped for CasADi's SQP (`sqpmethod`, via `_make_nlp_solver`: qpoases QP + `convexify_strategy='regularize'`, fails gracefully). Validated once in `__init__`; default unchanged, so existing behaviour is identical. **Not exposed to the backend/frontend** — see why below.

- `4dTrajectory/optimization/solver_backend_benchmark.py` (new) — cold-solve wall-time + convergence + accuracy, ipopt vs sqpmethod, on feasible (fixed-T / free-T) and an aggressive near-stall straight-in. Finding on this problem: **ipopt is robust** (feasible fixed-T ~0.5 s, free-T ~10 s; aggressive correctly reported infeasible), **sqpmethod is not usable cold** — it bails in ~3 ms ("Search_Direction_Becomes_Too_Small") on every case (incl. feasible) from our linear-interp initial guess; over-tuning `min_step_size` even makes qpOASES `abort()` the process (SIGABRT). SQP's payoff needs warm-starting (not wired) + a feasible initial guess.
- Warm-start investigation (seed SQP with a converged IPOPT fixed-T solution): SQP warm-start needs the **duals** (`lam_g0`/`lam_x0`), not just `x0` — primal-only doesn't recognise the KKT point. Seeded with primal+duals it re-solves a *nearby* problem to the target, but only with the right config (exact Hessian, not `convexify_strategy='regularize'`, + a small `min_step_size`): ~4 iters / ~2.5 s. With `regularize` it takes 17–22 iters / **52–60 s** and mislabels the status. Even the best warm SQP (~2.5 s) is **slower than cold IPOPT (~0.2–0.7 s)** here, because CasADi `sqpmethod` uses the **dense** active-set QP (qpoases) — ~300× IPOPT's per-iteration cost on this 330-var problem; it never exploits the banded OCP structure. Conclusion: **IPOPT stays the default and the only frontend-exposed backend**; the real warm-start payoff would require a structure-exploiting SQP/RTI solver (e.g. acados/HPIPM), not CasADi's generic `sqpmethod`. The `solver_backend` switch + `solver_backend_benchmark.py` remain as documented infrastructure. Test: `test_solver_backend_validation_and_default`. 55 python optimizer/backend tests pass.

Made the direct-collocation defect "fitting equation" pluggable, so the optimiser, backend, and frontend can pick among four transcription schemes. Three collocate the continuous geodetic RHS (trapezoidal, Hermite-Simpson, RK4); a fourth (`reanchoredEnu`) uses the re-anchored ENU one-step integrator (`make_geo_step_from_enu_integrator`) — the exact model the playback runs — as a **shooting** defect. `_DEFECT_SCHEMES` is now `name → (defect_fn, dynamics_kind)` where `dynamics_kind ∈ {"continuous", "enuStep"}`; `_make_scheme_dynamics` builds the right callable per scheme (continuous RHS vs the ENU stepper), so `_build_collocation_decision` takes a generic `dynamics` arg. The ENU defect converts radians↔degrees (optimiser nodes are radians; the stepper is degrees). Note: a stepper can be a *shooting* defect but NOT a *polynomial* collocation defect (trapezoidal/HS need a continuous `f`) — the comment in `casadi_simulator.py` was corrected accordingly. Backend name `casadiDirectCollocationReanchoredEnu`; frontend dropdown lists all four. This grew out of the RW05L "terminal offset" investigation: the offset was **not** the sampling step nor a playback mismatch (the optimiser's own RHS and the playback integrator drift identically), but the **collocation accuracy** of an aggressive near-floor/near-stall solution at the default state density — i.e. a discretisation issue, of which the defect scheme is one lever (state density is the other).

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
