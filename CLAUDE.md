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
- **final_approach/** — The single final-approach geometry (runway frame, segment fit, arg-min runway assignment). Pure `geokit` + stdlib, no I/O, no regulation constants; imported by BOTH `trajectory_data_process/harvest` and `evaluation/arrival.py`
- **trajectory_data_process/harvest/** — New download pipeline: fetch → reconstruct → assign (one runway per track) → `tracks/` + `approach/`. CLI: `python -m trajectory_data_process.harvest --airport KRDU`
- **flight_scenarios/** — Data→modeling seam (observed track → `FlightScenario`)
- **evaluation/** — File-based trajectory judging + batch metrics (geokit + stdlib only)
- **4dTrajectory/optimization/** — Optimizers, constraints, batch tooling
- **4dTrajectory/ts_transformer/** — Learned trajectory prediction (vendored iTransformer + PatchTST, torch)
- **aeroviz_backend/** — Python HTTP backend (simulation / optimization / dynamics-comparison)
- **run_scenario_pipeline.py** — Batch runner: arrivals (step 0, only when the combined czml-input is missing) → scenarios → optimize → CZML comparison + evaluation

## Build & Dev Commands

### Frontend (aeroviz-4d/)

```bash
cd aeroviz-4d
npm install
npm run dev                          # Vite dev server with HMR
npm run backend                      # the Python backend, from here
npm run build                        # tsc + vite production build
npm test                             # Vitest (watch mode)
npx vitest run                       # Single run, no watch
npx vitest run src/utils/__tests__/ocsGeometry.test.ts  # Single test file
npm run test:coverage                # Coverage report
npm run build:local-terrain          # Airport-local heightmap terrain tiles
npm run build:local-terrain:visual-assets
```

### Python (aeroviz-4d/python/)

```bash
pip install -r aeroviz-4d/python/requirements.txt
python -m pytest aeroviz-4d/python/tests/test_generate_czml.py -v
python -m pytest aeroviz-4d/python/tests/ --cov=. --cov-report=html
```

### Data Pipeline (end-to-end)

```bash
# Full pipeline: download from the OpenSky history DB → normalize → generate CZML
python run_asd-b_fetch_and_generate.py --airport CYYC

# Render an already-normalized *_czml_input_*.json directly (skips the download).
# --airport is REQUIRED in both forms.
python run_asd-b_fetch_and_generate.py --airport CYYC \
    --input-json trajectory_data_process/outputs/cyyc_czml_input_*.json

# Scenario optimization batch (all 3 modes per airport when --target-type omitted)
python run_scenario_pipeline.py --jobs 6

# ts_transformer full chain (2 models × 2 horizon modes: train → predict → eval → CZML;
# dataset build + split happen inside train, split persisted in the checkpoint)
python run_ts_pipeline.py --airport KRDU

# Wipe ALL generated pipeline data (scenarios, optimizer+ts outputs, frontend comparison
# + observed CZML). Raw downloads and _parked dirs kept unless --include-downloads /
# --include-parked; archive_pipeline_data.py is the reversible alternative.
python clean_pipeline_data.py --dry-run
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
- `useAirportLocalTerrainLayer` — loads preprocessed `.f32` heightmap tiles via `terrain/airportLocalTerrain.ts`; returns `{ status, metadata, provider, error, … }`; controlled by the `layers.airportLocalTerrain` toggle

UI components (ControlPanel, HUD, FlightTable) overlay on the Cesium canvas via CSS grid with `pointer-events: none`.

### Data Flow

```
OpenSky history DB → trajectory_data_process.harvest → tracks/manifest.json (all outcomes, HAE)
    ├→ observed evaluation + generate_czml.py → trajectories.czml → useCzmlLoader
    └→ arrivals/manifest.json (model-ready final arrivals)
         ├→ flight_scenarios → optimization/evaluation
         └→ ts_transformer training/prediction
```

Static data: OurAirports CSV → `preprocess_airports.py` → `runway.geojson`; ARINC 424 CIFP → `preprocess_waypoints.py` / `preprocess_procedures.py` → `waypoints.geojson` / procedure details.

Modeling pipeline: `arrivals/manifest.json` → `flight_scenarios` (`FlightScenario`: initial+target `GeodeticState`, `AircraftSpec`, `AeroParams`, source incl. `entry_time_utc`) → `4dTrajectory/optimization/scenario_optimization.py` (`*_states.json` + `*_eval.json`) → `build_scenario_comparison_czml.py` (3-colour comparison CZML) + `evaluation` (report JSON/HTML). TS training reads the same arrival manifest directly. `flight_scenarios` sits between the data plane and the modeling plane — depends downward on modeling primitives, imported upward by both consumers (no cycles); it's a top-level package deliberately (`4dTrajectory` isn't importable).

### Key Data Formats

- **CZML**: JSON array where first element is a "document" packet (clock config), subsequent elements are entity packets with time-sampled positions via `cartographicDegrees: [secondsOffset, lon, lat, altMetres, ...]`
- **GeoJSON**: static layers (runways, waypoints, OCS surfaces)
- Airport config: `public/data/airports/<CODE>/airport.json` (per airport, not a single file) — `{code, lon, lat, height}`
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
- **Machine-dependent (READ THIS FIRST on a new machine):** on THIS Mac there is no `aeroviz` env — `aviation` (py3.13, casadi 3.7.2) IS the thesis env, and `scripts/activate_aeroviz_env.sh` resolves to it correctly by probing for casadi. The warning below is written from the Linux compute box's perspective and misleads when read here; trust the resolver, which selects by CONTENT not name.
- **`aviation` on the LINUX box is NOT the thesis env** — it belongs to
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
- **Vite must never watch `public/data`** (`vite.config.ts` → `server.watch.ignored`). chokidar takes ONE inotify watch per file and that tree is ~40k files (39,307 local-terrain `.f32` heightmap tiles), so a single dev server ate 41,260 of the system's 65,536 `fs.inotify.max_user_watches`. Two at once — e.g. a forgotten `nohup npm run dev` from an earlier session still holding the port — blew the limit and vite died on boot with `ENOSPC: System limit for number of file watchers reached`, which the supervisor then restart-looped (frontend dying at ~2 s, backend healthy the whole time). With the ignore rule: **363 watches**, 113× less. Nothing under `data/` is a build input (git-ignored generated output, fetched over HTTP at runtime), so watching it only ever bought the crash. Symptom to recognise: `Port 5173 is in use, trying another one…` plus a restart streak means a stale dev server is alive — `ss -ltnp | grep 5173`, and count a pid's watches via `/proc/<pid>/fdinfo/<fd>`.
- **casadi symbolic construction is NOT thread-safe** (global SXElem pools; concurrent NLP builds → heap corruption → C++ `abort()`/SIGABRT that Python can't catch). Casadi-heavy endpoints run in an isolated worker subprocess (`aeroviz_backend/isolated_backend.py`); all in-process casadi entry points serialize on `casadi_lock.CASADI_LOCK`. Toggle isolation off with `AEROVIZ_ISOLATE_SOLVER=0` (to get a native traceback).
- **IPOPT is sensitive to CasADi symbol-creation order** — `make_dynamics()` must run before NLP decision symbols are created.
- **Heading convention**: the dynamics model ψ is math-ENU (0 = East, CCW toward North; `V_east = V·cosγ·cosψ`). `approach_constraints.geometry.course_bearing` returns this convention (`atan2(Δn, Δe)`). Mixing in compass convention reads an aligned aircraft as a 90° intercept.
- **`geokit.METRES_PER_DEG_LAT` is derived from `WGS84_A·π/180` (111319.4908…), not the old hand-rounded `111_320.0`** — the rounded value put a 4.6 ppm seam (~0.11 m at the 25 km ring) between every geokit-derived frame (ts channels, `start_state` velocity fits) and the optimizer's NE frame (`approach_constraints.frame` + the NLP's metric-position normalization, both `WGS84_A·DEG2RAD`). One definition, bit-identical everywhere; `metres_per_deg_lon = METRES_PER_DEG_LAT·cos(lat)` (pure cosine, no ellipsoidal correction). Frontend mirror regenerated (`src/generated/geoConstants.json`). The `hermiteSimpsonNormalizedFullTransport` scheme's decision state IS these threshold-anchored NE metres (`(lat−lat_t)·R`, `(lon−lon_t)·R·cos lat_t`) — an exact affine change of variables for Jacobian conditioning; the defect still evaluates the exact geodetic full-transport RHS on the reconstructed physical state (only the `localEnu` scheme family approximates dynamics in a flat frame).
- **Velocity is PHYSICAL at the `flight_scenarios` seam, and the ts channels are chart derivatives (both since 2026-07-20, B3.1).** `_velocity_lsq` projects through the true tangent scales (`R_M+h`, `(R_N+h)·cosφ`, via `geokit.wgs84_curvature_radii` — numeric single source; the casadi RHS and its mirror comment are the symbolic twin), so fitted `V/psi/gamma` mean what the dynamics model integrates; it used the flat chart constants before, overstating `V_north` by `a/R_M` (+0.33% at 36°). `channels.py` then maps physical → chart with the full-transport Jacobian, making `∫ edot dt` reproduce `e` exactly (unit-tested). Measured on 995 KRDU arrivals the residual integration drift is unbiased LSQ smoothing (~2.4–2.7 m/min median). The two seams MUST move together: fixing only the channels re-adds a +0.33% north systematic (measured 8.6 m/min).
- **Comparison-overlay entities are TIME-WINDOWED — an empty scene is not a broken overlay.** Each group only shows inside its own availability interval, so at a clock time outside it the map is legitimately blank. Pause inside a window before diagnosing (≈08:09 UTC on the KRDU data).
- **All aircraft CZML sets `forwardExtrapolationType:"HOLD"`** — `position.getValue` returns the frozen final position forever forward; any outward time-walk must stop when the position stops changing, not only on null.
- **Cesium `Clock.tick()` LOOP_STOP wrap preserves overshoot** (`currentTime = startTime + (currentTime − stopTime)`); use `clock.onStop` (fires at the stop time for both CLAMPED and LOOP_STOP) for exact end-of-playback emits — never elapsed-based heuristics.
- **CZML document clock intervals must use `iso_ms`** — second-precision `iso()` truncation made the clock stop up to 1 s before the last sample (~25–75 m phantom position error).
- **Probe bug**: `build_optimized_trajectory_playback` needs a REAL optimizer name — `simulation_mode_for_optimizer` on an unknown name selects the alpha-control mode and misreads casadi load-factor controls (fake 8–11 km "drift").
- **`geokit` is src-layout** (`geokit/src/geokit/`) because a top-level `geokit/` dir on sys.path (CWD under pytest) would shadow the installed package.
- **`aeroviz-4d/public/data` is git-ignored** (local artifacts; regenerate via preprocess scripts).
- **`.flight-ops-panel` has `backdrop-filter`** → it becomes the containing block for `position:fixed` descendants AND clips overflow; floating windows must render via React portal into `document.body`.
- **"Batch edition" seam class**: the batch callers in `scenario_optimization.py` duplicate wiring the backend HTTP path also has — several bugs (stale `_solve_iaf` unpack, trapezoidal left behind after the HS flip, missing `n_seg_per_phase`) came from updating one path and missing the other. When changing optimizer wiring, update BOTH and their seam tests.
- Stale docs (historically inaccurate, kept): `4dTrajectory/docs/direct_collocation_hermite_simpson.zh.md` §5 and `geodetic_dynamics_transport.zh.html` describe the old HS-planner + RK4-polish pipeline.
- **`import torch` BEFORE `import traffic` used to break matplotlib** — pip's manylinux torch wheel resolves `libstdc++.so.6` from `/lib/x86_64-linux-gnu` (CXXABI ≤ 1.3.13); once that SONAME is loaded, conda-forge matplotlib's `_c_internal_utils.so` (needs CXXABI_1.3.15) fails. The reverse import order worked, and `run_all_tests.sh` runs both suites in ONE pytest process with `4dTrajectory` (torch) ahead of `trajectory_data_process` (traffic) — i.e. exactly the failing order. Fixed by `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh`, which prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (with a matching `deactivate.d`). **This only applies under `conda activate`** — invoking `envs/aeroviz/bin/python` directly bypasses it and the old failure returns.
- **Observed ADS-B altitude is ELLIPSOIDAL (HAE); everything it is judged against is MSL.** OpenSky `geoaltitude` is height above the WGS84 ellipsoid; runway thresholds, CIFP altitudes and the 8260.58D gates are orthometric. The gap is the geoid undulation N ≈ −25 to −33 m over the US (KRDU −33.53). Uncorrected, real completed airline landings scored **1.8 % on the gates** (18/996 KRDU) and the vertical gate passed ~0 %. Converted once at the data→modeling seam by `flight_scenarios/datum.py` (EGM96 via pyproj). **Do NOT move this into the harvest**: CZML positions are consumed by Cesium as metres above the ellipsoid (`aeroviz-4d/src/types/czml.d.ts`) and are CORRECT as recorded — converting at the source fixes modeling and breaks the viewer by the same 33 m. The conversion is keyed on `altitude_source` (hence idempotent) and reaches THREE ingest paths — `load_model_arrivals`, `build_scenario`, and `ts_transformer/dataset.py` (which reads bare waypoints and so cannot self-protect); unknown/missing sources RAISE rather than defaulting, and `"synthetic"` is already-MSL. **The seam is symmetric on the way OUT**: modeling records (`*_states.json`, predictions) are MSL, and `build_scenario_comparison_czml._states_to_waypoints` — the single point every record-derived entity flows through — converts MSL→HAE via `aeroviz-4d/python/vertical_datum.py` (a deliberate MIRROR of `flight_scenarios/datum.py`, same KRDU N = −33.53 pin + ballpark probe; the modeling tree must not be imported there). The observed reference bypasses it (deep-copied from `trajectories.czml`, already HAE). Records are MSL by ASSUMPTION, not by tag — pre-datum-fix HAE-era artifacts are discarded wholesale (user decision); feeding one through the builder would double-shift it ~33.5 m low. **PROJ trap**: with the EGM96 grid missing and network off, pyproj silently returns a "ballpark" no-op vertical transform — a correction that looks applied and does nothing; `_geoid_transformer()` probes a known undulation and raises.
- **`runway_thresholds.json` holds LANDING thresholds (displaced), not pavement ends** (`runway-thresholds-v2`, `displaced_threshold_m` on every entry). `build_runway_config.py` used to ignore `*_displaced_threshold_ft`: KSJC 30L/30R are displaced **775 m**, which on a 3° glidepath is a **40.6 m** altitude error and moved the OPTIMIZER TARGET, not just the gates. Six thresholds are displaced (KSJC ×4, KSTL 12R 143 m, KMSY 29 93 m). Fix the GENERATOR, never the JSON. The landing-threshold computation is single-sourced in `acquisition/runways.py` (`landing_thresholds_from_row`), shared by the generator AND `resolve_runway_threshold` — the `download_trajectories.py --runway` path previously kept pavement ends, naming a point up to 775 m away from the config's and shifting `landing_time_utc` (hence `flight_key` identity) between the two harvest paths. Corollary worth remembering: KSJC looked *healthiest* before the fix (+9.7 m vs everyone else's −25 m) because its displaced-threshold and datum errors had opposite signs and nearly cancelled — the airport that looks best can be the one with two bugs.
- **A landing must be assigned to ONE runway; parallel runways used to capture it twice.** `classify_landing_flights` runs per threshold, `RUNWAY_THRESHOLD_RADIUS_M` is 1000 m, and parallels sit 250–400 m apart on an identical heading — so geometry AND heading accepted either. Measured: **169 of KSJC 30L's 200 flights were also in 30R's file** (12L∩12R 63, KSTL 30L∩30R 32); KRDU/KSMF/KMSY are unaffected (parallels exceed the radius). Downstream signature: an observed lateral error whose MEDIAN *is* the parallel separation. `sibling_thresholds` arbitrates, restricted to same-direction runways (the opposite end of the same runway must be excluded — a full rollout stops on top of it). The discriminator is **median lateral offset from the extended centreline, NOT distance to the threshold point**: threshold distance does not separate at all (kept 763 m vs dropped 791 m) because a displaced threshold sits 775 m past where ADS-B coverage ends.
- **An observed track's `states[-1]` is NOT its arrival at the target.** A solve terminates at its target by construction; a harvested arrival is a truncated recording of a flight that continues — 966/996 KRDU tracks end a median **325 m short** of the threshold, still airborne. Grading observed data on final-state deviation measures where ADS-B coverage stopped. The meaningful observed metric fits each flight's own established final-approach line and extrapolates to the threshold (which also validates itself: the fitted glidepath comes out 3.02–3.13° at all five airports).
- **Harvest is manifest-only.** `tracks/manifest.json` rosters all four measured outcomes; `arrivals/manifest.json` rosters only assigned, CIFP-targeted, final-entry-cropped model inputs and records every exclusion. Scenario, optimizer-reference, and TS loaders follow that roster and never glob, so an orphan/rejected/stale JSON cannot enter a model split.
- **`"type": "UNK"` on every harvested arrival does NOT mean the batch is single-type.** `_resolve_aircraft` tries declared type → **`icao24` via the OpenAP lookup** → `--aircraft-type` fallback, and the icao24 path recovers the REAL airframe for most flights: **20 distinct types** across 400 KRDU arrivals (A320 224, B738 38, E75L 25, CRJ9 23, … A333, GLF6, C550). Anything assuming one airframe per batch is wrong — that is exactly how the flyability check first shipped, grading ~44% of flights against an A320. The fallback (`--aircraft-type`, train default `A320`) is a `TSConfig` field, so it is recorded in the checkpoint and predict defaults to the train-time value; overriding it at predict shifts the ENU frames and the target Vref/threshold-crossing height the gates measure against, so it WARNS.
- **ts_transformer is a purely kinematic BASELINE — no dynamics/aerodynamics is connected, by design.** Channels in, channels out; the only symbol it imports from `aerodynamic_model` is the `GeodeticState` dataclass (no equations), vs the optimizer's `CasadiSimulator` + `rollout_piecewise_constant`. Predictions therefore carry NO flyability guarantee (speeds/turn rates/thrust/`Cl_max` are unchecked) — the survey's "statistically plausible but unflyable" problem. Do NOT treat this as an unfinished TODO; it is what lets the learned component be measured on its own. The four routes if it is ever added (post-hoc flyability check → post-hoc casadi projection → soft physical loss → differentiable torch dynamics) are written up in the package README. Same for single-aircraft-only and deterministic-point-prediction: scope decisions, listed separately from real gaps in that README.
- **ts_transformer flyability (`flyability.py`): read the DELTA against the observed tracks, never the absolute rate.** The closed-form control inversion (no casadi, no solver) judges against ONE clean-configuration drag polar, and real approaches are flown dirty — run on REAL flown tracks it first scored **0/149 fully flyable**, i.e. the check was wrong, not the flights. Median required thrust on a real arrival is **0.43 kN** (idle); negative required thrust just means drag augmentation (speedbrake/flaps/gear), so `thrust_negative` is a SOFT violation and the observed baseline (**63.2%** on KRDU) is the FLOOR, not 100%. `Cl_max` comes from `aero_params_for_aircraft` (A320: 2.7), NOT `LoadFactorSimulator`'s hardcoded 1.5 — an 80% disagreement. **Each flight is judged against its OWN airframe** (`report_for_records` takes one `Aircraft` per flight; the report carries `fleet` + `envelopes`) — the first version shared one envelope and mis-graded ~44% of a batch. **Flyability alone is not a quality metric**: in the ablation the WORSE predictor scores HIGHER on it in 3 of 4 cells (89.3% vs 29.6% while 2.2× worse at the threshold) by predicting blander paths — a straight line is perfectly flyable and completely wrong. Always pair it with the error metrics.
- **ts_transformer: instance normalisation is OFF and must stay off by default.** iTransformer's `use_norm` and PatchTST's `revin` are ON upstream; both strip a window's absolute level as "nuisance". In a threshold-anchored ENU frame absolute position IS the signal (it decides where the turn onto final is, when the descent starts, where the approach ends). Ablated on real KRDU data, all 8 cells (`outputs/KRDU/_ablation_norm/ablation_results.json`, tables in the package README): **off wins 19 of 20 accuracy comparisons, one tie.** Signature of ON: lateral p95 pins at 14.28–14.50 km in ALL four cells — a model that cannot place the endpoint at all — vs 2.6–8.5 km for off. Real-data gaps (1.2–2.7×) are much smaller than synthetic (2.4–6.5×), so judge on the sweep, not one metric: a single-metric partial pass showed an apparent reversal that did not survive consistent scoring.
- **A flight's identity is `flight_key` = `id_runway_icao24_landingTime` (`flight_scenarios.identity`), NEVER `id` alone — and this has bitten four separate layers.** The raw harvest carries **no unique flight id at all**: `id` is a copy of the callsign, and OpenSky stores state vectors by icao24 + time, so an "arrival" is a segment this project derives — identity is (which aircraft, when). Measured on the 996 KRDU arrivals: `id` → 552 distinct, `icao24` → 717, `id_runway` → 778, `id_runway_icao24` → 874, **`icao24`+landing time → 996**. The extra fields in `flight_key` are for filename readability, not uniqueness. Casualties: the ts train/val/test split (leaked; `predict --split test` returned every namesake, 48 flights for an 18-flight split), the comparison-CZML group key (`id_runway` silently dropped 22% of a full batch), the FlightTable optimizer join (callsign-keyed; namesakes swapped V/mass/verdicts), and the observed-layer CZML entity ids (bare callsigns; Cesium merges same-id packets — per-runway files had up to 128 duplicate ids, two flights garbled into one entity). The same function produces the ts record stems, the optimizer's record filenames (`_scenario_filename` wraps it), the CZML group key (via the record filename stem), **the observed-layer entity ids** (`generate_czml`, which RAISES on a duplicate identity), and the comparison reference lookup — so they cannot drift. `aeroviz-4d/python/flight_identity.py` is a deliberate MIRROR of the function (frontend tooling must not import the modeling tree); both copies are pinned to the vector `EJA969_05R_ad7f04_20260618T213736Z`, change them together. Corollaries: entity `name` (the callsign) is the ONLY display text (FlightTable/approach view render names, never ids); **positional `_N` id re-uniquing is deleted from the landing path** (`czml_export.classify_landing_flights`, `build_arrivals.merge_landing_flights`) because each harvest chunk restarted the numbering (merged files held duplicates anyway) and the combined-file renumbering gave the same flight DIFFERENT ids in different views — duplicate bare-callsign `id`s in landings/arrivals files are normal and correct; only the plain (non-landing) download path keeps `_unique_id`, since without `runway`/`landing_time_utc` the suffixed id is its only discriminator.
- **ts_transformer: prediction records are anchored at `t=0` = the anchor sample**, `initial_state` is the observed state THERE (not the track start), and the reference record covers the SAME span. `evaluation.reference.compare_to_reference` resamples both paths at 101 fractions of *their own* arc length, so a whole-track reference against an anchor→threshold prediction reports kilometres of pure span mismatch (measured: 4349 m → 833 m once span-matched). **That rebase does NOT survive into a shared clock — the CZML builder must add `source.anchorTimeS` back.** The reference copied out of `trajectories.czml` still starts at t=0 = the START of the track, so writing a prediction's own times through unshifted drew it a whole lookback early (KRDU 05L: the forecast's first sample, bit-identical to the reference's t=118 s sample, was plotted at t=0 — 12.0 km from where the reference then was). `observed_states` (whole track, negative t before the anchor) is REQUIRED in the prediction schema and is the only source for the `look-` lookback entity — without it the purple line begins in mid-air with nothing joining it to the start of the approach. Lookback = the `t ≤ 0` slice; the anchor sample belongs to both halves, so the join is exact, not approximate.
- **Pre-existing numpy failure in `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` is BACK (2026-07-21).** `float(np.array(grad(x0))[0])` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars` under numpy 2.x. It went green on 2026-07-20 and failed again on 07-21 with no optimizer change in between, so it tracks the numpy version, not the code. Verified unrelated to any working-tree change by re-running with the tree stashed. Modeling suite is otherwise 588 pass.

- **Observed ADS-B altitude is quantised to 25 ft = 7.62 m, which is 83 % of the whole 9.15 m vertical gate.** All 482 distinct altitudes in the KRDU set lie on that lattice; 54.8 % of consecutive samples report an IDENTICAL raw altitude (the aircraft crosses one step every ~2 samples at 3.81 m of descent per 1 Hz sample). Consequences: a single sample carries ±3.81 m of rounding and **cannot resolve the gate even in principle** — `states[-1]` is not a usable crossing measurement, the least-squares fit is what recovers sub-quantum precision (σ ≈ 1.7 m); and the quantisation staircase IS the residual autocorrelation (lag-1 ρ ≈ 0.43, n_eff ≈ 0.40 n), so the OLS variance must be autocorrelation-corrected. **Deflate BOTH variance terms** — `Sxx` sums over the same correlated samples as `1/n`; correcting only the first gives a 1.15× inflation where the honest figure is **1.58×**.
- **Per-runway TCH is published in the CIFP and is NOT 15 m.** ARINC 424 section P / subsection P "Path Point" records (`data/CIFP/<cycle>/FAACIFP18`) carry glidepath angle, course width and threshold crossing height per LPV approach; `harvest/cifp.py` decodes them. Every runway in the fleet publishes **15.27–18.11 m**, so the old flat 15 m assumption put a systematic 1.5–2.5 m bias into a 9.15 m window. Using the published value moved the measured KSMF crossing from +2.74 m to **+0.61 m** — real traffic crosses where the plate says, to within half a metre, which is the best end-to-end check that the datum, the fit and the TCH source all agree. The column decode is pinned by a coincidence-proof cross-check: **4795 of 4900 records decode a course width of exactly 106.75 m**, independently the LPV semiwidth in `evaluation/thresholds.py`. A runway with no Path Point record has **no LPV procedure** (KRDU 14/32) — its TCH is `None`, never defaulted, because it cannot be judged against LPV gates at all.
- **Archived history rows are in FEET; `fetch_history_dataframe` output is in METRES.** `_altitudes_to_metres` converts on the live path only, so rows read back from `outputs/history_rows/**.jsonl` are raw feet. Guessing wrong scales every altitude by 3.28 and does not crash — it turns a 3.06° approach into a 9.94° one (observed). `harvest.reconstruct_tracks` therefore takes a **required** `altitude_units` argument with no default and no sniffing.
- **Runway assignment must be an arg-min over ALL thresholds at once, not a per-threshold test plus a guard.** The predecessor classified per threshold and undid double-assignment with a pairwise parallel-runway guard; the guard's logic was right and the shipped artifacts *still* had **232 of KSJC's 319 unique landings (72.7 %) in two runways' files** (169 in 30L+30R, 63 in 12L+12R; 32 at KSTL). `final_approach.assign_runway` fits once per candidate and takes the arg-min, so two runways are unrepresentable rather than guarded against. What discriminates: **median absolute cross-track** separates parallels (13.9 m correct vs 230.5 m wrong at KSJC, whose separation is 228.4 m — the wrong file's median offset IS the separation); distance to threshold does NOT (763 vs 791 m) and is worst at displaced thresholds; **direction of travel** is the only thing separating the two ends of one runway (they share a centreline).
- **Fit only the FINAL INBOUND RUN, never an along-track range.** A real arrival occupies the same along-track band more than once — downwind leg, vectoring, a go-around, or a track exported against the wrong runway end so it holds the approach AND the landing roll. One shipped KSJC track ranged over −23.5 km to +18.7 km yet ended at +2.6 km; a range filter mixed downwind samples in and produced a **median cross-track of 8.7 km**, which then decided a runway assignment. `_final_inbound_run` walks backward from the window's inner edge and stops on a reversal; this also subsumes direction (an outbound track yields no fit at all).
- **Assignment must never filter on approach quality.** If the harvest rejected tracks on the criterion `evaluation` later reports, every survivor would pass by construction and the established rate would be 1.0 — manufactured by the selection, not measured. Hence `final_approach` exposes facts only (no `established` flag, no gate constant): assignment asks *which runway* (relative), `evaluation/arrival.py` asks *how good* (absolute). Same reason the harvest quota counts **assigned landings, not established ones**.
- **`unassignable` (the receiver lost it) and `not_established` (the approach was not stabilised) must stay distinct outcomes.** Conflating them charges a reception gap to the pilot. Both are counted and reported; neither is ever dropped (established rate is 21–54 % on real data).
- **The observed gate verdict is mostly undecidable, and the report must say so.** With σ ≈ 1.7 m the 95 % CI is 6.7 m wide against a 9.15 m window, and the fleet's median vertical deviation sits ~0.5 m from the upper bound — so **67 % of established KRDU flights have a CI straddling a gate boundary**. `evaluate_batch` reports a `marginal` count alongside pass/fail; a bare pass rate over 25 ft-quantised data claims more than it knows. Read the deviation distribution as primary, the pass rate as secondary.
- **The fit window is `[−5000, −300]` m and it is a real methodological choice, not a free parameter.** On a 3° path that spans ~900 ft down to ~107 ft above threshold — below the 1000 ft stabilisation gate, above flare (~50 ft). Measured on KRDU with published TCH: median crossing +5.43 m from a `[−8000,−300]` (≈ FAF) window vs **+3.66 m** at `[−5000,−300]` vs +4.04 m at `[−2000,−300]`, with σ climbing to 2.50 m at the short end. Starting at the FAF biases HIGH (aircraft still intercepting from above); shrinking below ~3 km leaves too short a baseline to pin the slope. Report the sensitivity table, never one number.
- **Track reconstruction: a landing ends at a sustained ON-GROUND run, and a spatial crop must keep only the final contiguous run.** An aircraft that lands keeps transmitting from the gate, so state vectors are continuous across a turnaround and the >900 s gap rule never fires; the radius filter then removed the middle and glued two passes together — KSTL AAL2717 carried a **6598 s** hole with two stray samples after it, which put the flight's `landing_time_utc` (part of its identity) on the WRONG pass. Also: `_complete()` must require **both** dep and arr airports, not either — `(None,'KRDU') → ('KATL','KRDU')` is one flight with the origin merely resolved, and the old rule cut the approach in half.

## Key Defaults & Constants (current)

- Mesh: `collocation/optimizer.py` `DEFAULT_N_SEGMENTS = 8`, `DEFAULT_N_SEG_PER_PHASE = 3` (single source; backend + batch import them; `run_scenario_pipeline.py` mirrors 8/3 with a "MUST match" comment). Frontend/backend unconstrained `n_segments` default = 10 (a different knob). Multiphase mesh = n_seg_per_phase × legs (`n_segments` doesn't apply).
- State substeps: auto per phase ≈ 3 s state step, cap 16 (`_TARGET_STATE_STEP_S`/`_MAX_STATE_SUBSTEPS`); explicit `--state-substeps`/frontend "State substeps" (0 = auto, clamp 0–64) overrides. Do NOT lower below auto (M=4 → 14.5 km rollout error); on unconstrained solves M=32 improves accuracy/optimum; constrained solves don't need big M (per-node inequality rows make big M explode solve time).
- Fitting: constrained + unconstrained default = Hermite-Simpson (`hermiteSimpsonNormalizedFullTransport`; frontend `DEFAULT_TRAJECTORY_OPTIMIZER = casadiMultiphaseNormalizedFullTransport`). `FITTING_SCHEMES`: `hs` / `trapezoidal` / `rk4` via `--fitting-type`. Trapezoidal is dynamically unfaithful on aggressive min-time floor-riding solves (5–15 km rollout drift vs HS metres); rk4 is basin-fragile there (needs M=64) — both kept for comparison studies only.
- IPOPT: `components.DEFAULT_MAX_ITERATIONS = 3000` (`ipopt.max_iter` set explicitly on BOTH IPOPT constructions — verbose and quiet; request `maxIterations` reaches both backend branches). The third construction in that function is the `sqpmethod` backend, which **hardcodes `max_iter: 100` and ignores `max_iterations` entirely**. Linear solver: `AEROVIZ_IPOPT_LINSOL` (default `mumps`) + `AEROVIZ_IPOPT_HSLLIB` — HSL hook dormant (free MA27 measured 3–27× slower than MUMPS on these small NLPs; kept for a future MA57 attempt); batch speed lever is `--jobs`.
- Altitude floor: `altitude_floor_m(target) = target − 5 m` (a real operational floor min-time solves ride; was −300 m which they dove to). Transition-phase floor = min(start alt, first leg's published entry floor `_first_leg_entry_floor_m`) − margin. Rollout guard: `ROLLOUT_GUARD_MARGIN_M = 5.0`, `rollout_guard_altitude_m(target) = altitude_floor_m(target) − 5` (zero margin truncated faithful floor-riding replays on cm integration noise). `min_altitude_m` is REQUIRED on `rollout_controls`/`simulate_controls` (no silent sea-level default).
- ψ corridor: constrained solves bound the heading variable to the route's heading hull ± 90° (`_route_psi_profile`, `_PSI_CORRIDOR_SLACK_RAD`) — this killed the whole looping/crawling local-optimum family. Terminal ψ pinned on the route-unwrapped branch (the first element of `_route_psi_profile`'s return); per-phase heading guesses = own leg course.
- Join/passage constraints (`approach_constraints` + `collocation/optimizer.py`): the ONE forced fix passage is the pre-FAF fix, within its leg's k·RNP disc; FAC join = on-course (`fac_cross_track = 0`) with along-course distance in `[d_FAF + L_final/5, d_FAF + max_offset]` (max_offset auto = half the leg into the FAF; join guess = window middle; `max_join_offset_m=0` → the single point 1/5 before the FAF); branch-aware intercept box `|ψ_join − course_branch| ≤ 30°`; two-tier FAC alignment ±30° join→FAF, ±10° (`_FAC_ALIGN_TIGHT_DEG`) FAF→threshold; vertical glidepath window binds only `d ≤ d_faf_m`, upstream the published FAF minimum (`prefaf_floor_m`) applies. Tiny duration-split regularizer `1e-4·Σ(Tp/T_max)²` on free-time multi-phase solves. Constraint families are explicit functions dispatched from `_build`: `_terminal_pin_rows` / `_fac_join_rows` / `_prefaf_fix_rows` / `_leg_path_rows` / `_fac_alignment_rows`.
- Transition phase: prepended (unconstrained) when the start is farther than `_first_fix_join_tolerance_m` from the first fix (= first leg's k·RNP when it has one, 2 km fallback for LPV-first). Frame-anchor contract validated loudly (`segments[-1].end_ne` + `lpv.ltp_ne` at origin ±150 m). `DEFAULT_K_MARGIN` + `STANDARD_INTERCEPT_MAX_DEG` single-sourced in `approach_constraints`.
- Evaluation gates (`evaluation/thresholds.py`, overridable; derived from FAA Order 8260.3F / 8260.58D, which are public and NOT vendored here — the section numbers are the citation): lateral ≤ **106.75 m** (FAA 8260.58D §3-1-5.c(3) Formula 3-1-1, 350 ft LPV semiwidth floor at threshold); vertical ∈ **[−3.05, +6.10] m** (§1-3-1.f(2)(b) TCH/WCH window). Gates judge the TRUE-dynamics rollout's final state. Batch metrics: solve/success rates, lateral mean/p95/max, vertical spreads, flight times; path-shape deviation vs reference = both paths resampled at 101 fractions of their own horizontal arc length.
- Playback drift guard: `playbackDriftM` on every optimize response; stderr WARNING above `PLAYBACK_DRIFT_WARN_M = 50`.
- Arrival truncation (`trajectory_data_process/arrival_segment.py`): `ENTRY_RADIUS_KM = 25` (builder `--entry-radius-km`, in (0, 30)), `ENTRY_HYSTERESIS_SAMPLES = 3`, `LOCAL_START_RADIUS_KM = 5`.
- Worker sessions: `AEROVIZ_WORKER_IDLE_TIMEOUT_S` (default 600) idle watchdog reclaims a stranded resident solver worker.
- **ts_transformer** (`config.py` is the single source; everything below is serialised into every checkpoint — mechanism and rationale in the package README): `dt_s = 2.0`, `seq_len = 60` (120 s), `pred_len` = 30 (window, 60 s) / **300** (full, 600 s). Channels = `(e, n, u, edot, ndot, udot)` threshold-anchored chart; **names AND order are load-bearing** (the tuple indexes tensors, normalizer stats and checkpoints — `load_checkpoint` refuses a mismatch, which is also what locks out pre-2026-07-20 `ve/vn/vu` checkpoints after the transport-consistency change). The velocity channels are the EXACT chart derivatives of the position channels (full-transport Jacobian, `geokit.wgs84_curvature_radii`), not raw physical ENU components — see the Operational Gotchas transport bullet. Traps: the horizon was sized from the MEASURED duration distribution (p50 328 s / p95 651 s), covering **97.8%** of flights — the old "an arrival is ~3.5–5 min" straight-line estimate was WRONG (real arrivals are vectored), do not resize from it; the ~2% over the horizon are cut at H and flagged `horizonCapped`, so their gate verdicts are cap artifacts, not model error. `summary.json` carries an `accuracy` block (mean AND p95) plus per-row `ade_m`/`fde_m`; `overlap` is a REQUIRED arg to `write_batch` — an optional metric is one that silently goes missing.
- Comparison CZML colour contract: group status lives on entity `properties.status` ∈ solved/offTarget/failed. Reference: white / dark-red (failed) / dark-amber `OFF_TARGET_REF_COLOR` (off-target); simulator/result path bakes bright yellow `OFF_TARGET_COLOR` (255,205,40) + "(off target)" name; optimizer plan keeps legend orange/cyan. **The frontend repaint skip is keyed on "a verdict colour was baked", NOT on `status` alone** — reference always, plus off-target optimizer/simulator paths. `build_scenario_comparison_czml.states_schema` dispatches on the record keys (`optimizer_states`/`simulator_states` → `opt-`+`sim-` entities; `predicted_states`+`observed_states` → `pred-` (purple `PREDICTION_COLOR`, kind `predicted`) **plus `look-`** (same RGB at alpha 85 `LOOKBACK_COLOR`, kind `lookback`) — see the anchor-shift gotcha). **Predictions never get the off-target bake** (`mark_off_target = off_target and schema == "optimizer"`): a forecast essentially always misses the 106.75 m gate, so marking it repainted 27/27 groups yellow and the kind colour was never visible. Their `status` stays accurate and they ARE repainted from the legend — so `PREDICTION_COLOR` and the TS legend entry are not required to agree.
- Categories manifest: `categories.json` entries carry an explicit `"constrained": bool` (frontend validator REQUIRES it; `_cons`-suffix detection deleted). Evaluation read side is manifest-ONLY: `load_records` reads a batch dir via `summary.json` roster (`results[].eval_file`); manifest-less dir / listed-missing file / empty roster raise (no glob fallback — globbing counted orphans).
- Stale-artifact hygiene (write side): `_clear_stale_records` deletes top-level `*_states.json`/`*_eval.json` at batch start; `write_reference_records` clears `references/*_reference_eval.json` first; CZML builder `clear_stale_outputs` deletes previous `comparison_*.czml` + a stale published `evaluation_report.json`. Record-filename suffixes + `REFERENCES_DIR` + the `summary.json` row shape (`summary_row`) single-sourced in `optimization/evaluation_export.py` (imported by both the batch and `ts_transformer/export.py`).

## Changelog

The dated development log lives in **`docs/CHANGELOG.md`** — deliberately not loaded by default (it is long). Read it only when you need history: why a design is the way it is, when/why a default changed, what a past bug/postmortem looked like, or which outputs a change made stale.

Maintenance convention:
- **Append new dated entries to `docs/CHANGELOG.md`** (newest first, `### YYYY-MM-DD — title`), not here.
- When a change produces a durable fact (a gotcha, a default, a contract), also update "Operational Gotchas" / "Key Defaults & Constants" above — those, not the changelog, are what every session sees.
- Keep the Open Items list below current: add items as they arise, delete them when resolved.

## Open Items (current as of 2026-07-20)
- **ALL observed-derived artifacts on disk are STALE as of the 2026-07-20 datum/threshold/assignment fixes** (see the CHANGELOG entry). Stale: `flight_scenarios/outputs`, every `4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}`, every `public/data/airports/*/comparison`, and the KRDU `ts_*` training data + checkpoints. Re-run order: scenarios → optimizer batch → CZML/report tails → ts retrain. **KSJC and KSTL additionally need a re-harvest** (bug ③ changed runway assignment; offline de-dup would cost KSJC 42 % of its flights, leaving 12L at 12 and 30R at 39). KRDU/KMSY/KSMF need no re-harvest. OpenSky history access verified working from this Mac on 2026-07-20 (credentials live in `~/Library/Application Support/pyopensky/settings.conf`, NOT `~/.config`).
- **The ts_transformer gate-pass conclusion needs re-deriving, not re-quoting.** The recorded "gate-pass counts 0–4 of 152" scored a ±3 m vertical window against data offset ~33 m by the datum bug. Accuracy metrics (ADE/FDE, deviation vs a reference in the same frame) should be nearly unchanged; the gate verdicts were not measuring what they claimed.
- **Local terrain and aircraft CZML disagree by ~33 m in the viewer.** `local-terrain` heightmaps come from USGS TNM DSM (NAVD88 ≈ MSL) and the metadata records `vertical: "Source GeoTIFF elevation values, used directly as metres"` — no datum handling — while Cesium expects ellipsoidal heights and the aircraft CZML correctly supplies them. Found while chasing bug ①; not investigated further.

- **KRDU RW32 is systematically hard, and it is NOT the old truncation artifact.** The full 2026-07-20 batch re-run (post-truncation/floor/HS/identity fixes, all 15 airport×category cells fresh) kept the concentration: KRDU runway_cons RW32 = 79 offTarget + 59 failed vs 60 clean solves (198 flights; every other runway ≤ 9 offTarget), and KRDU **asdb RW32 fails 197/198** (IPOPT infeasible). Runway/procedure-specific — check against the per-leg-RNP-not-extracted item (H05LZ is RNP-AR) before touching solver knobs. KSTL runway_cons has a milder cluster (12R 53/200, 30R 41/168, 30L 30/200, 24 21/80; the "all IAF(s) infeasible" rows repeatedly name `PAULY`).
- Approach view: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view (Observe-with-comparison plots neither source); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case.
- Per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (~926 m at k=0.5) instead of ~278 m (RNP 0.3).
- CIFP leg speed restrictions not extracted (no speed-bearing data source in the dataset yet; the canonical `speedMaxKt` field is ready).
- HSL linear-solver hook dormant (free MA27 measured slower than MUMPS); revisit with an MA57 academic license.
- **`ts_transformer` KRDU run is DONE (three generations; current = 2026-07-20 B3)** — artifacts in `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` + `ts_pred_*` (B3 transport-consistent channels + physical-velocity fit; the previous generation is parked in `outputs/KRDU/_pre_b3_transport/`, the first is not reproducible — quote ONLY current-artifact numbers), tables in the package README, history in `docs/CHANGELOG.md` (07-19, 07-20 ×2). Robust across all generations: one-pass `full` beats chained `window` on whole-approach lateral error for both models (1.5–1.7× mean); PatchTST leads at short lead while iTransformer leads at 600 s on the raw-tensor accounting (5438 vs 7384 m, n=893 — direction held in all three generations, margins 1.16–1.36×; channel-independence can't represent a turn's east/north coupling). NOT stable across retrains: gate-pass counts (0–4 of 152 — only "forecast ≠ certifiable approach" survives) and the tail-vs-mean story (architecture-dependent). Hence: **treat any margin under ~1.5× as provisional** — both a split change and a ≤0.3% data rescale moved effects of that size. The two lead-time accountings (record vs raw-tensor) are NOT comparable — the README states both with their n. Remaining: only KRDU trained (4 other airports harvested; the per-threshold ENU frame makes pooling a real design question, not a bigger `--data` glob).
- ts_transformer follow-ups: single-aircraft only (no traffic interaction / ATC intent) and deterministic (no multimodality) — both are the survey's named open problems. Flyability is MEASURED but not FIXED (nothing projects a prediction back inside the envelope — README routes 2–4), and its polar is clean-configuration only, which is why it is read as a delta.
- Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07 approach-view changes were verified via tests/tsc/build but not re-checked in-browser.
