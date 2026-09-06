# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroViz-4D: Airport 4D trajectory and terrain digital-twin visualization system for thesis research. Combines a React/TypeScript/CesiumJS frontend with Python data pipeline tools to visualize aircraft trajectories (position + time) in 3D terminal airspace.

Dual purpose: thesis visualization/validation + reusable research component library.

## Repository Layout

Subsystem notes live in per-directory `CLAUDE.md` files — they load automatically when you touch
files in that tree, so **read the one for the subsystem you are changing** (the ✎ marks below).

- **aeroviz-4d/** ✎ — Main visualization app (React + CesiumJS frontend, Python CZML generator)
- **aeroviz_backend/** ✎ — Python HTTP backend (simulation / optimization / dynamics-comparison)
- **trajectory_data_process/** ✎ — Trajectory acquisition, processing, and dataset helpers.
  `harvest/` is the download pipeline: fetch → reconstruct → assign (one runway per track) →
  `tracks/` + `approach/`. CLI: `python -m trajectory_data_process.harvest --airport KRDU`
- **final_approach/** ✎ — The single final-approach geometry (runway frame, segment fit, arg-min
  runway assignment). Pure `geokit` + stdlib, no I/O, no regulation constants; imported by BOTH
  `trajectory_data_process/harvest` and `evaluation/arrival.py`
- **flight_scenarios/** ✎ — Data→modeling seam (observed track → `FlightScenario`); owns the
  vertical datum, flight identity and velocity contracts
- **evaluation/** ✎ — File-based trajectory judging + batch metrics (geokit + stdlib only)
- **4dTrajectory/** ✎ — `optimization/` optimizers, constraints, batch tooling
- **4dTrajectory/ts_transformer/** ✎ — Learned trajectory prediction (vendored iTransformer +
  PatchTST, torch)
- **geokit/** ✎ — Shared geodesy/units package (src-layout, `pip install -e` into conda `aeroviz`)
- **bc_lidar_downloader/** — BC LiDAR terrain data downloader
- **prepare_scenario_inputs.py** — Rebuild arrivals/observed outputs from stored tracks and
  generate fitted-ADS-B + runway-target scenario JSON
- **run_scenario_optimization.py** — Consume prepared scenario JSON, optimize, evaluate, and
  publish comparison CZML

## Build & Dev Commands

```bash
# Full observed pipeline: download from OpenSky history → manifests → CZML/evaluation
python -m trajectory_data_process.harvest --airport CYYC

# Render an existing flight-array JSON directly (does not create a harvest).
python aeroviz-4d/python/generate_czml.py --airport CYYC \
    --input path/to/flight_array.json \
    --output aeroviz-4d/public/data/airports/CYYC/trajectories.czml

# Prepare inputs (default: 2000 arrivals per runway, evenly spaced over landing time;
# --max-per-runway 0 takes everything, and the choice is written to <scenarios>.selection.json)
python prepare_scenario_inputs.py --skip-observed
# Optimize all 3 modes per airport. --jobs defaults to cores-4; the run pre-checks free disk
# space and refuses to start if its estimate does not fit. --resume makes a crash cheap.
python run_scenario_optimization.py --resume --max-groups-per-czml 500
#   --max-iterations N   cap IPOPT per solve (biggest cost lever; default 3000)
#   --rollout-dt 1.0     halve the on-disk footprint (coarsens the evaluated states)
#   --continue-on-error  keep the sweep going past one failed airport/category

# ts_transformer full chain (2 models × 2 horizon modes: train → predict → eval → CZML;
# dataset build + split happen inside train, split persisted in the checkpoint)
python run_ts_pipeline.py --airport KRDU

# Preview the allow-listed regenerable outputs for one airport, then clean them.
# Training/experiment artifacts, final-test ledgers, downloaded tracks, unknown/manual
# outputs, mixed experiment comparison trees, static data, and archives are preserved.
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU --dry-run
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU

./run_all_tests.sh                   # both suites in one pytest process (env via the resolver)
./start_aeroviz_fullstack.sh         # supervisor: frontend + backend
cd aeroviz-4d && npm run dev         # frontend only — more commands in aeroviz-4d/CLAUDE.md
```

Per-subsystem commands (frontend/Vitest, ts_transformer train/predict/eval) live in that
subsystem's `CLAUDE.md`.

## Architecture

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

- **CZML**: JSON array where first element is a "document" packet (clock config), subsequent entity packets carry time-sampled positions via `cartographicDegrees: [secondsOffset, lon, lat, altMetres, ...]`
- **GeoJSON**: static layers (runways, waypoints, OCS surfaces)
- Airport config: `public/data/airports/<CODE>/airport.json` (per airport, not a single file) — `{code, lon, lat, height}`
- Evaluation record (one JSON/trajectory): `{source, initial_state, target_state, final_time_s, states[], controls[]}` — controls 1:1 ZOH-aligned with states; unsolved = empty states+controls; reference records (observed track in same contract) have `controls == []`; solved records require `final_time_s == states[-1].t`

## Environment

- **`aeroviz` (Python 3.12) is THE thesis env on this machine** — data acquisition (`traffic`,
  `pyopensky`), CIFP parsing (`cifparse`, `arinc424`), `casadi` + IPOPT, `openap`, the
  conda-forge geospatial stack, editable `geokit`, and `torch`. One env runs everything;
  `run_all_tests.sh` picks it and its `4dTrajectory` entry covers the ts_transformer suite.
- **Machine-dependent (READ THIS FIRST on a new machine):** on THIS Mac there is no `aeroviz` env
  — `aviation` (py3.13, casadi 3.7.2) IS the thesis env, and `scripts/activate_aeroviz_env.sh`
  resolves to it correctly by probing for casadi. The warning below is written from the Linux
  compute box's perspective and misleads when read here; trust the resolver, which selects by
  CONTENT not name.
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
- **`import torch` BEFORE `import traffic` used to break matplotlib** — pip's manylinux torch
  wheel resolves `libstdc++.so.6` from `/lib/x86_64-linux-gnu` (CXXABI ≤ 1.3.13); once that
  SONAME is loaded, conda-forge matplotlib's `_c_internal_utils.so` (needs CXXABI_1.3.15) fails.
  The reverse import order worked, and `run_all_tests.sh` runs both suites in ONE pytest process
  with `4dTrajectory` (torch) ahead of `trajectory_data_process` (traffic) — i.e. exactly the
  failing order. Fixed by `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh`, which prepends
  `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (with a matching `deactivate.d`). **This only applies
  under `conda activate`** — invoking `envs/aeroviz/bin/python` directly bypasses it and the old
  failure returns.
- Python env is conda **`aeroviz`**. This line used to say `aviation`, which is a DIFFERENT
  project's env on this machine, and that caused a near-miss deletion — hence the warning above.
- Env spec backups (regenerate `aeroviz` if ever needed): `.env-backup/aeroviz-pip-freeze.txt`,
  `aeroviz-conda-explicit.txt`, `aeroviz-environment.yml`.
- GPU: RTX 4060, 8 GB (compute capability 8.9), cu128 wheels.
- This machine: 16 GB RAM, frequently swap-bound — memory pressure (Cesium + casadi + IDE +
  browser) causes UI lag independent of code changes.
- Frontend build config (Cesium Ion token, vite-plugin-cesium, TS strict, jsdom):
  `aeroviz-4d/CLAUDE.md`.

## Domain Context

- **TMA** (Terminal Maneuvering Area) — controlled airspace around airports
- **OCS** (Obstacle Clearance Surface) — PANS-OPS geometry ensuring terrain clearance on approach
- **4D Trajectory** — position (lon, lat, alt) + time; the 4th dimension is scheduled arrival time
- **CTA** (Controlled Time of Arrival) — ATC-assigned time slot at a fix point

## Coding Conventions

**Minimise defensive / patch-like code.** Prefer clear contracts over scattered guards.

- Don't sprinkle `if x is not None` / `try/except` / fallback branches for inputs that shouldn't occur. Give the parameter a sensible **default**, or make it **required** — pick one. Validate once at the boundary if truly needed; otherwise fail loudly.
- No band-aids over a root cause — fix upstream (the parser, not the consumer).
- Keep the happy path linear; one explicit assumption beats repeated `None`/empty checks.
- No silent approximations or caps: any approximation gets an explicit option + notice; any bounded coverage (top-N, sampling) is stated in output, never silent.
- Single source of truth: constants/conversions/course math defined once (geokit, `approach_constraints`, module constants) and imported everywhere; "MUST match" mirror comments only where an import is impossible (e.g. the import-light pipeline runner).
- **A schema literal in a consumer is a mirror** — import it, or comment it as a mirror; never let
  a fixture restate it (a version pinned in a test is a version the test cannot check).
- **A bound that can never bind is worse than no bound** — if it cannot change an answer on the
  real fleet, delete it, or the reader will assume it did.
- **`get(key, DEFAULT)` returns `None` for a key present with a null value** — use
  `get(key) or DEFAULT` when a null must read as "unspecified"; and a check comparing two
  optional fields to each other passes when BOTH are missing.

## Cross-Cutting Invariants

Short index; the full text (with measurements) is in the named file, which loads when you work there.

- **Vertical datum**: observed ADS-B altitude is ELLIPSOIDAL (HAE), everything it is judged
  against is MSL (N ≈ −33 m over the US). Converted once **at the `flight_scenarios` seam**, and
  converted back MSL→HAE on the way out to CZML. Never at the harvest; never twice. Records are
  MSL by assumption, not by tag. → `flight_scenarios/CLAUDE.md`
- **Flight identity is `flight_key` = `id_runway_icao24_landingTime`, never `id` alone** — the
  raw harvest carries no unique flight id (`id` is the callsign). Four layers have already been
  bitten. → `flight_scenarios/CLAUDE.md`
- **Harvest is manifest-only**: `tracks/manifest.json` and `arrivals/manifest.json` are the
  rosters; scenario/optimizer/TS loaders follow them and never glob (globbing counts orphans).
  Evaluation's read side is likewise `summary.json`-rostered. →
  `trajectory_data_process/CLAUDE.md`, `evaluation/CLAUDE.md`
- **Derived repairs and slices are READ-TIME; `tracks/` is never edited.** Altitude-outlier repair
  and the arrival-window slice both happen on read, so no artifact needs rebuilding — and editing
  the store would break the per-record SHA-256, `--reclassify-existing` and
  `source_integrity.retained_rows`. → `trajectory_data_process/CLAUDE.md`
- **Assignment asks *which* runway (relative); evaluation asks *how good* (absolute).** The
  harvest must never filter on approach quality, or the established rate is manufactured rather
  than measured. → `final_approach/CLAUDE.md`
- **Observed tracks have TWO time windows** (first reception vs the 25 km arrival slice, median
  45 s apart). The comparison overlay must use the model one, or the group renders ~5 km early
  and it reads as model error. → `aeroviz_backend/CLAUDE.md`
- **One definition of every geodetic constant** (`geokit.METRES_PER_DEG_LAT`,
  `wgs84_curvature_radii`); the frontend `geoConstants.json` and the casadi RHS are generated
  mirrors. → `geokit/CLAUDE.md`
- **casadi symbolic construction is NOT thread-safe** — isolated worker subprocess +
  `CASADI_LOCK`. → `4dTrajectory/CLAUDE.md`
- **The backend does NOT hot-reload** — restart `./start_aeroviz_fullstack.sh` after backend
  changes. → `aeroviz_backend/CLAUDE.md`
- **`aeroviz-4d/public/data` is git-ignored** (local artifacts; regenerate via preprocess
  scripts), and **`geokit` is src-layout** so a top-level `geokit/` on sys.path can't shadow the
  installed package.

## Changelog

The dated development log lives in **`docs/CHANGELOG.md`** — deliberately not loaded by default
(it is long). Read it only when you need history: why a design is the way it is, when/why a
default changed, what a past bug/postmortem looked like, or which outputs a change made stale.

Maintenance convention:
- **Append new dated entries to `docs/CHANGELOG.md`** (newest first, `### YYYY-MM-DD — title`).
- When a change produces a durable fact (a gotcha, a default, a contract), also update the
  **subsystem's own `CLAUDE.md`** — or this file's Cross-Cutting Invariants if it spans trees.
  Those, not the changelog, are what a session actually sees.
- Status items go in **`docs/open-items.md`**; the root Open Items block keeps only hazards that
  must fire unprompted.
- Code-health findings noticed **outside** the change you are making go in
  **`docs/code-health-followups.md`** (deferred, one entry each, marked verified vs judgement) —
  not into the change, and not into Open Items unless they block something.

## Open Items

Full status — every campaign, measurement and blocked item — is **`docs/open-items.md`**.
Only the hazards that must fire unprompted are repeated here.

- **`--evaluate-only` DELETES the v5 arrivals roster.** All five
  `outputs/harvest/<ICAO>/arrivals/manifest.json` are `harvest-arrivals-v5-takeoff-excluded`
  (re-measured on disk 2026-09-06, **42,650** arrivals — KRDU 14,435 / KSJC 11,082 / KSTL 8,767 /
  KSMF 4,219 / KMSY 4,147) with their `lateral_pass_eligibility.json`.
  Do not re-run it without need.
- **All control-output ts checkpoints are stale as of 2026-08-18** — the control contract
  changed units (newtons → fraction of installed thrust) and `TSConfig` gained required fields,
  so `load_checkpoint` refuses them. `state` checkpoints are unaffected, but any trained before
  2026-08-24 predates the v5 cohort.
- **A per-airport ADE/FDE quoted without its route mix is not a comparison** (KSJC 483 → 1526 m
  reweighted, best of five to worst). Published tables predate the covariates and need
  re-deriving. → `4dTrajectory/ts_transformer/CLAUDE.md`
- **Quote only current-artifact numbers** — the KRDU ts run has three generations and the first
  is not reproducible; the gate-pass conclusion still needs re-deriving after the datum fix.
- **The optimizer batch has no SOLVES on disk, but the prepared inputs DO exist** —
  `flight_scenarios/outputs` is NOT empty (it holds all ten `*_scenarios.json` + their
  `.selection.json`, 92 MB, built 2026-08-23); what is missing is `4dTrajectory/outputs/<ICAO>`
  solve records, so `--skip-optimize` still has nothing to reuse. Whether the 08-23 scenarios can
  be reused as-is is UNVERIFIED — the runner validates prepared-input signatures, so let it
  decide rather than assuming. Scale, read off the selections on disk: **23,429 flights /
  70,287 solves** (`runway`; `fitted_adsb` selects the same flights and then drops the 20
  `UnusableFittedApproach` ones → 23,409 / 70,227), ~30 h at `--jobs 24`, 12.3 GiB
  (`--rollout-dt 1.0` → 8.1 GiB). Free space is the binding constraint and moves with the ts
  experiments, so check it right before launching; the runner refuses to start if its estimate
  does not fit.
