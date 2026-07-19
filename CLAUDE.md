# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroViz-4D: Airport 4D trajectory and terrain digital-twin visualization system for thesis research. Combines a React/TypeScript/CesiumJS frontend with Python data pipeline tools to visualize aircraft trajectories (position + time) in 3D terminal airspace.

## Repository Layout

- **aeroviz-4d/** — Main visualization app (React + CesiumJS frontend, Python CZML generator)
- **trajectory_data_process/** — Trajectory acquisition, processing, and dataset helpers
- **bc_lidar_downloader/** — BC LiDAR terrain data downloader
- **run_asd-b_fetch_and_generate.py** — Orchestrator: fetch -> normalize -> generate CZML pipeline
- **geokit/** — Shared geodesy/units package (src-layout, `pip install -e` into conda `aeroviz`)
- **flight_scenarios/** — Data→modeling seam (observed track → `FlightScenario`)
- **evaluation/** — File-based trajectory judging + batch metrics (geokit + stdlib only)
- **4dTrajectory/optimization/** — Optimizers, constraints, batch tooling
- **4dTrajectory/ts_transformer/** — Learned trajectory prediction (vendored iTransformer + PatchTST, torch)
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

### Learned prediction (4dTrajectory/ts_transformer/)

```bash
conda activate aeroviz                                     # the single thesis env (has torch)
TS=4dTrajectory/ts_transformer/__main__.py
python $TS train   --data <arrivals.json|dir> --airport KRDU --model itransformer \
                   --horizon-mode window --output-dir 4dTrajectory/outputs/KRDU/ts_itr
python $TS predict --checkpoint .../checkpoint.pt --data ... --output-dir .../ts_pred
python -m evaluation --input .../ts_pred                   # same gates as the optimizer
python -m pytest 4dTrajectory/ts_transformer/tests -q --import-mode=importlib
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

- **`aeroviz` (Python 3.12) is THE thesis env on this machine** — data acquisition (`traffic`,
  `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi` + IPOPT, `openap`, the
  conda-forge geospatial stack, editable `geokit`, and `torch`. One env runs everything;
  `run_all_tests.sh` picks it and its `4dTrajectory` entry covers the ts_transformer suite.
- **`aviation` on this machine is NOT the thesis env** — it belongs to
  `/home/supercomputing/studys/AivationTransformer` (a different project; pure-pip, py3.11).
  The name collides because on another machine the thesis env IS called `aviation`.
  **Do not install thesis packages into it and do not delete it.** `run_all_tests.sh` and
  `start_aeroviz_fullstack.sh` both resolve the env via `scripts/activate_aeroviz_env.sh`,
  which probes candidates with `import casadi` (so the wrong-project `aviation` here is
  skipped by content, not trusted by name), keeps a qualifying already-active env,
  ACTIVATES (never direct-execs `envs/<env>/bin/python` — activate.d hooks must run), and
  treats an explicit `AEROVIZ_CONDA_ENV` as the only candidate (a typo fails loudly).
- **Consolidating the thesis into a py3.11 env is BLOCKED**, tested: `cifparse` >= 2.0.4
  (aeroviz has 2.0.9) uses PEP 701 f-strings — nested same-type quotes — which is Python
  3.12+ syntax; every version from 2.0.4 up fails `compileall` on 3.11. Only 2.0.0 and
  earlier import there, i.e. a 9-patch regression in the ARINC 424 parser that feeds
  `approach_constraints`. Its PyPI metadata claims `>=3.10` and is simply wrong.
- Env spec backups (regenerate `aeroviz` if ever needed): `.env-backup/aeroviz-pip-freeze.txt`,
  `aeroviz-conda-explicit.txt`, `aeroviz-environment.yml`.
- GPU: RTX 4060, 8 GB (compute capability 8.9), cu128 wheels.
- Requires `VITE_CESIUM_ION_TOKEN` in `.env` (Cesium Ion access token)
- Vite config uses `vite-plugin-cesium` (asset copying, `CESIUM_BASE_URL`)
- TypeScript strict mode (strict null checks, noUnusedLocals, noUnusedParameters)
- Test environment: jsdom with vitest globals
- Python env: conda `aeroviz` (see the env bullets above — this line used to say `aviation`,
  which is a DIFFERENT project's env on this machine and caused a near-miss deletion)
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
- **`import torch` BEFORE `import traffic` used to break matplotlib** — pip's manylinux torch wheel resolves `libstdc++.so.6` from `/lib/x86_64-linux-gnu` (CXXABI ≤ 1.3.13); once that SONAME is loaded, conda-forge matplotlib's `_c_internal_utils.so` (needs CXXABI_1.3.15) fails. The reverse import order worked, and `run_all_tests.sh` runs both suites in ONE pytest process with `4dTrajectory` (torch) ahead of `trajectory_data_process` (traffic) — i.e. exactly the failing order. Fixed by `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh`, which prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (with a matching `deactivate.d`). **This only applies under `conda activate`** — invoking `envs/aeroviz/bin/python` directly bypasses it and the old failure returns.
- **A harvest directory holds five overlapping views of the same flights**: `*_arrivals.json` (truncated at the ring — the training input), `*_landings.json` (SAME flights untruncated), `*_combined_czml_input.json` (all runways merged), plus `*_heading_rejected.json` / `*_local_rejected.json` (tracks the harvester THREW OUT). A naive `glob("*.json")` loaded every flight three times over plus the known-bad ones — invisible in a loss curve. `dataset.select_flight_files` takes the first matching pattern only, never mixes, always excludes `*_rejected*`, and prints what it skipped.
- **Every harvested arrival has `"type": "UNK"`** (`czml_export` hardcodes it) — but that does NOT mean the batch is single-type. `_resolve_aircraft` tries declared type → **`icao24` via the OpenAP lookup** → `--aircraft-type` fallback, and the icao24 path resolves most flights to their REAL airframe: measured on 400 KRDU arrivals, **20 distinct types** (A320 224, B738 38, E75L 25, B737 25, CRJ9 23, … A333, GLF6, C550). Anything that assumes one airframe per batch is wrong — this is exactly how the flyability check first shipped, grading ~44% of flights against an A320's `Cl_max`/max thrust. It resolves per flight; only genuinely unresolvable ones hit the fallback. `ts_transformer` takes `--aircraft-type` (train default `A320`, printed on every run); the resolved value is a `TSConfig` field, so it is **recorded in the checkpoint and predict defaults to the train-time value** (an explicit differing `--aircraft-type` at predict prints a WARNING — it shifts the ENU frames and gate targets away from what the normalizer was fit under). Not cosmetic: it sets the target state's Vref and threshold-crossing height, which is what the evaluation gates measure the final state against.
- **ts_transformer is a purely kinematic BASELINE — no dynamics/aerodynamics is connected, by design.** Channels in, channels out; the only symbol it imports from `aerodynamic_model` is the `GeodeticState` dataclass (no equations), vs the optimizer's `CasadiSimulator` + `rollout_piecewise_constant`. Predictions therefore carry NO flyability guarantee (speeds/turn rates/thrust/`Cl_max` are unchecked) — the survey's "statistically plausible but unflyable" problem. Do NOT treat this as an unfinished TODO; it is what lets the learned component be measured on its own. The four routes if it is ever added (post-hoc flyability check → post-hoc casadi projection → soft physical loss → differentiable torch dynamics) are written up in the package README. Same for single-aircraft-only and deterministic-point-prediction: scope decisions, listed separately from real gaps in that README.
- **ts_transformer flyability (`flyability.py`): read the DELTA against the observed tracks, never the absolute rate.** The closed-form control inversion (no casadi, no solver) judges against ONE clean-configuration drag polar, and real approaches are flown dirty. Run on REAL flown tracks it first scored **0/149 fully flyable** — the check was wrong, not the flights: median required thrust on a real arrival is **0.43 kN** (idle), and negative required thrust just means drag augmentation (speedbrake/flaps/gear). `thrust_negative` is therefore in `SOFT_VIOLATIONS` (reported, not counted unflyable) and the observed baseline (≈56–58%) is the FLOOR, not 100%. `Cl_max` comes from `aero_params_for_aircraft` (A320: 2.7), NOT `LoadFactorSimulator`'s hardcoded 1.5 — an 80% disagreement. **Flyability alone is not a quality metric**: in the instance-norm ablation the WORSE predictor scores HIGHER on it in 3 of 4 cells (PatchTST window 89.3% vs 29.6%, while being 2.2× worse at the threshold) by predicting blander paths — a straight line is perfectly flyable and completely wrong. Always pair it with the error metrics. **Each flight is judged against its OWN airframe** (`report_for_records` takes one `Aircraft` per flight; the report carries `fleet` + `envelopes`) — a KRDU batch spans ~20 types, and the first version shared one envelope and mis-graded ~44% of it.
- **ts_transformer: instance normalisation is OFF and must stay off by default.** iTransformer's `use_norm` and PatchTST's `revin` are ON upstream; both strip a window's absolute level as "nuisance". In a threshold-anchored ENU frame absolute position IS the signal (it decides where the turn onto final is, when the descent starts, where the approach ends). Measured on synthetic KRDU, all four model×mode cells: off wins by 2.4–6.5× on ADE, and on *converges sooner* to the worse optimum. **Re-ablated on REAL KRDU data (all 8 cells, `4dTrajectory/outputs/KRDU/_ablation_norm/ablation_results.json`) — the default holds: off wins 19 of 20 accuracy comparisons with one tie** (every cell on val loss / FDE / ADE p95 / lateral p95; mean ADE 3 of 4, PatchTST window a 0.4% dead heat). Real-data gaps are smaller (1.2–2.7×), so judge on the sweep, not one metric — a single-metric partial pass had shown an apparent reversal in that same cell that did not survive consistent scoring. Signature of ON: lateral p95 pins at 14.28–14.50 km in ALL four cells (a model that cannot place the endpoint at all), vs 2.6–8.5 km for off.
- **ts_transformer: a flight's identity is `id_runway_icao24_landingTime` (`flight_scenarios.identity.flight_key`), never `id` alone.** `id` is the callsign and repeats daily and across runway files; keying the split on it leaks train/val/test and makes `predict --split test` return every namesake (48 flights for an 18-flight split). The SAME function produces the ts record stems AND the optimizer batch's record filenames (`_scenario_filename` wraps it), so split key and both writers' filenames cannot drift.
- **ts_transformer: prediction records are anchored at `t=0` = the anchor sample**, `initial_state` is the observed state THERE (not the track start), and the reference record covers the SAME span. `evaluation.reference.compare_to_reference` resamples both paths at 101 fractions of *their own* arc length, so a whole-track reference against an anchor→threshold prediction reports kilometres of pure span mismatch (measured: 4349 m → 833 m once span-matched).
- Pre-existing, unrelated: `4dTrajectory/optimization/collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` fails with `TypeError: only 0-dimensional arrays can be converted to Python scalars` (numpy scalar-conversion deprecation), independent of ts_transformer.

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
- **ts_transformer** (`config.py`, single source; all serialised into every checkpoint, incl. `aircraft_type`): `dt_s = 2.0`, `seq_len = 60` (120 s), `pred_len` = 30 (window, 60 s) / **300** (full, 600 s). Sized from the MEASURED duration distribution of the 3747 harvested arrivals (p50 328 s / p95 651 s / p99 920 s): full-mode L+H covers **97.8%** of flights (150 would cover 57.6%). The old "an arrival is ~3.5–5 min" straight-line estimate was WRONG (real arrivals are vectored) — do not resize from it. The ~2% of flights longer than the horizon are cut at H and flagged `horizonCapped`/`horizon_capped` in record + summary (predict prints a WARNING; their gate verdicts are cap artifacts). `load_checkpoint` refuses a checkpoint whose channel order differs from `channels.CHANNELS`, and loads with `weights_only=True`. Train/val/test split is per-flight sha256 of `(seed, flight_id)` — stable under harvest growth, fractions approximate. Channels = `(e, n, u, ve, vn, vu)` in a threshold-anchored ENU frame (`channels.CHANNELS`, order is load-bearing — it indexes tensors, normalizer stats and checkpoints). `psi`/`gamma` are never regressed directly (±π wrap); they fall out of the velocity components as `atan2(vn, ve)`, which IS the math-ENU convention. Records are reference-shaped (`controls == []`) and emitted via `optimization/evaluation_export.py` (casadi-free, so it imports into the torch env). `summary.json` carries an `accuracy` block (mean AND p95 — chained window-mode error compounds into the TAIL) plus per-row `ade_m`/`fde_m`/`overlap_steps`; `overlap` is a REQUIRED arg to `write_batch` (an optional metric is one that silently goes missing) and a length mismatch raises rather than zipping the tail away.
- Comparison CZML colour contract: group status lives on entity `properties.status` ∈ solved/offTarget/failed. Reference: white / dark-red (failed) / dark-amber `OFF_TARGET_REF_COLOR` (off-target); simulator/result path bakes bright yellow `OFF_TARGET_COLOR` (255,205,40) + "(off target)" name; optimizer plan keeps legend orange/cyan. **The frontend repaint skip is keyed on "a verdict colour was baked", NOT on `status` alone** — reference always, plus off-target optimizer/simulator paths. `build_scenario_comparison_czml.states_schema` dispatches on the record keys (`optimizer_states`/`simulator_states` → `opt-`+`sim-` entities; `predicted_states` → one `pred-` entity, purple `PREDICTION_COLOR`, frontend kind `predicted`). **Predictions never get the off-target bake** (`mark_off_target = off_target and schema == "optimizer"`): a forecast essentially always misses the 106.75 m gate, so marking it repainted 27/27 groups yellow and the kind colour was never visible. Their `status` stays accurate and they ARE repainted from the legend — so `PREDICTION_COLOR` and the TS legend entry are not required to agree.
- Categories manifest: `categories.json` entries carry an explicit `"constrained": bool` (frontend validator REQUIRES it; `_cons`-suffix detection deleted). Evaluation read side is manifest-ONLY: `load_records` reads a batch dir via `summary.json` roster (`results[].eval_file`); manifest-less dir / listed-missing file / empty roster raise (no glob fallback — globbing counted orphans).
- Stale-artifact hygiene (write side): `_clear_stale_records` deletes top-level `*_states.json`/`*_eval.json` at batch start; `write_reference_records` clears `references/*_reference_eval.json` first; CZML builder `clear_stale_outputs` deletes previous `comparison_*.czml` + a stale published `evaluation_report.json`. Record-filename suffixes + `REFERENCES_DIR` + the `summary.json` row shape (`summary_row`) single-sourced in `optimization/evaluation_export.py` (imported by both the batch and `ts_transformer/export.py`).

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
- **`ts_transformer` first real run is DONE (KRDU, 995 arrivals; see `docs/CHANGELOG.md` 2026-07-19)** — artifacts in `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` + `ts_pred_*`. **Retrained 2026-07-20 on the reproducible `flight_key` split (702/141/152)** — the earlier checkpoints' split reproduces for only 552/995 flights (clean but pre-fix); do not quote pre-retrain numbers. Headline: one-pass `full` beats chained `window` on whole-approach lateral error for BOTH models on BOTH splits (1.5–1.6× mean, 1.5–2.1× p95) — the robust result; iTransformer beats PatchTST at long lead (channel-independence can't represent the east/north coupling of a turn) and PatchTST wins at short lead where the aircraft is near-straight (full-mode pair: 184 vs 571 m at 10 s, reversing by 300 s). **Two earlier claims did NOT survive the split change**: "compounding cost lands in the TAIL not the mean" (mean and p95 ratios are now equal to within noise; the tail effect survives only in FDE) and "0 gate passes in all four runs" (iTransformer now 3/152 full, 1/152 chained). **Treat any single-split margin under ~1.5× as provisional** — that is the size of effect a split change moved. Remaining: only KRDU trained (4 other airports harvested, cross-airport generalisation untested and the per-threshold ENU frame makes pooling a real design question); `--instance-norm` still not re-ablated on real data.
- ts_transformer follow-ups: single-aircraft only (no traffic interaction / ATC intent) and deterministic (no multimodality) — both are the survey's named open problems. Flyability is MEASURED but not FIXED (nothing projects a prediction back inside the envelope — README routes 2–4), and its polar is clean-configuration only, which is why it is read as a delta.
- **The ts prediction categories in the frontend have NOT been verified in-browser** (Chrome extension not connected in the session that built them). Backed by tsc + 451 frontend tests + `npm run build` + a structural check of every contract point the frontend reads. Open http://localhost:5174 → Observe → Trajectories → "Optimizer comparison" → pick a `Predicted (…)` category and confirm purple prediction + white reference paths draw and the legend checkboxes toggle them.
- Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07 approach-view changes were verified via tests/tsc/build but not re-checked in-browser.
