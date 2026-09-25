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
python run_ts.py pipeline --airport KRDU

# Preview the allow-listed regenerable outputs for one airport, then clean them.
# Training/experiment artifacts, final-test ledgers, downloaded tracks, unknown/manual
# outputs, mixed experiment comparison trees, static data, and archives are preserved.
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU --dry-run
conda run -n aeroviz python clean_pipeline_data.py --airport KRDU

./run_all_tests.sh                   # every Python suite, two pytest runs (env via the resolver); exit 0 expected
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

Full text, with the investigation behind each line: `docs/environment.md` (E1–E12).

- **`aeroviz` (Python 3.12) is THE thesis env on this Linux box** — `traffic`/`pyopensky`,
  `cifparse`/`arinc424`, `casadi` + IPOPT, `openap`, the geospatial stack, editable `geokit` and
  `torch`; `run_all_tests.sh` picks it and covers the ts_transformer suite (E1).
- **Resolve the env by CONTENT, never by name.** `scripts/activate_aeroviz_env.sh` (used by
  `run_all_tests.sh` and `start_aeroviz_fullstack.sh`) probes candidates with `import casadi`,
  keeps a qualifying active env, ACTIVATES, and treats an explicit `AEROVIZ_CONDA_ENV` as the only
  candidate. On the Mac the thesis env is `aviation` (py3.13) (E2); **on this Linux box `aviation`
  belongs to `/home/supercomputing/studys/AivationTransformer` — never install thesis packages into
  it, never delete it** (it once nearly was) (E3, E6).
- **Always `conda activate`; never run `envs/aeroviz/bin/python` directly** — the
  `activate.d/zz-libstdcxx.sh` hook is what stops `import torch` before `import traffic` (the order
  `run_all_tests.sh` uses) from breaking matplotlib's CXXABI (E5).
- A py3.11 consolidation is BLOCKED: `cifparse` ≥ 2.0.4 needs Python 3.12 syntax (E4).
- ts campaigns run from the main tree: `.git/info/exclude` must carry `.claude/worktrees/`, or a
  nested worktree dirties the tree and a formal run refuses to start (E7); a killed formal run
  leaves a `running` manifest — if the arm directory holds only `config.json` + the manifest, move
  it aside as `<arm>.aborted-<UTC>` and rerun the SAME campaign command (E8).
- Env spec backups in `.env-backup/` (E9); GPU RTX 4060 8 GB, cc 8.9, cu128 wheels (E10); 16 GB
  RAM, frequently swap-bound — UI lag is memory pressure, not a code change (E11); frontend build
  config → `aeroviz-4d/CLAUDE.md` (E12).

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
- **兼容 (compatibility) is a FORBIDDEN word** (user, 2026-09-19): no `.get(key, default)` fallbacks, no schema-version
  branches, no "an older artefact reads as …" — refuse by name at the boundary. Every compatibility decision needs
  the user's explicit permission, case by case. **A class / payload / schema an experiment reads or writes is
  settled BEFORE that experiment runs**; a campaign's gate compares only inputs its own queue produced under one
  code version. An artefact produced by unfinished code is superseded and deleted, never kept beside the real one.
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
- **Every published experiment states its INTENT** (the user's rule, 2026-09-12): each campaign's
  title + question and one line per run live in
  `4dTrajectory/ts_transformer/docs/experiments/intents.json`, and the publisher BLOCKS a run
  without an entry. Write the entries when the campaign is designed and commit them with its arm
  declaration BEFORE launch (editing the registry under a running campaign dirties the main tree);
  when reporting a publication, state each campaign's intent too. →
  `4dTrajectory/ts_transformer/CLAUDE.md` (L27)
- **A ts checkpoint's data identity is the eligible SET, never the eligibility roster's bytes** —
  the roster embeds upstream provenance that legitimately moves over an unchanged set (a
  byte-bound identity refused every checkpoint on 2026-09-07). →
  `4dTrajectory/ts_transformer/CLAUDE.md` (C26)
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
  not into the change, and not into Open Items unless they block something. Each entry has a row in
  the status table at the top of that file; fixing one flips its row (with the commit) and deletes the entry.
- **A subsystem `CLAUDE.md` is an INDEX, not a store** (2026-09-16: the tree's CLAUDE.md files
  had reached 221 KB, the ts one 118 KB). One line per contract/gotcha/default, ending in an ID;
  the full text — measurements, history, runner manuals — lives in that subsystem's reference doc
  (`ts_transformer/docs/reference/`, `4dTrajectory/docs/optimizer_reference.md`,
  `evaluation/docs/EVALUATION_REFERENCE.md`, `trajectory_data_process/docs/06-harvest-reference.md`,
  `aeroviz-4d/docs/35-viewer-reference.md`, `flight_scenarios/docs/population_reference.md`,
  `docs/environment.md`). A new fact gets a new ID there and ONE line in the index.

## Open Items

Full status — every campaign, measurement and blocked item — is **`docs/open-items.md`**.
Only the hazards that must fire unprompted are repeated here.

- **`--evaluate-only` / `--merge-source` DELETE `lateral_pass_eligibility.json` and re-roster
  `arrivals/`.** Since 2026-09-23 the live harvest is the v5 roster (5/1–7/22) plus the 8/22–9/22
  download, `harvest-arrivals-v7-measured-crossing-in-slice`: **72,574 arrivals, 72,247 eligible**
  (train 50,693 / val 10,635 / test 10,919, seed 1337); the v5 roster every older checkpoint trained
  on is FROZEN read-only as `outputs/harvest-v5-20260823` and replays resolve to it by digest
  (`trajectory_data_process/docs/11-2026-09-23-merge-new-data.zh.md`). A rebuild changes every ts
  dataset split; never under a running campaign. **To rebuild only the observed evaluation
  records/report (`approach/`) use `--observed-only`.**
- **All control-output ts checkpoints from before 2026-08-18 are stale** — the control contract
  changed units (newtons → fraction of installed thrust) and `TSConfig` gained required fields, so
  `load_checkpoint` refuses them; any ts checkpoint trained before 2026-08-24 predates the v5 cohort.
- **ts numbers: quote only current-artifact numbers, and never a per-airport ADE/FDE without its
  route mix** (KSJC 483 → 1526 m reweighted, best of five to worst); published tables predate the
  covariates, and the gate-pass conclusion still needs re-deriving after the datum fix. →
  `4dTrajectory/ts_transformer/CLAUDE.md` "How to read results"
- **`clean_pipeline_data.py` would DELETE the 70,267 records the next item says to regenerate
  FROM** (verified 2026-09-20; KRDU's dry run alone lists 2.2 GB of "optimizer + standalone
  predictions (allow-listed)", ~7.0 GB over five airports). Do not run it unattended; exclude
  those directories by hand. → `docs/open-items.md`
- **No optimizer pass rate on disk is quotable.** `4dTrajectory/outputs/<ICAO>/{runway,fitted_adsb,runway_cons}`
  hold 15 batches, **70,267 records**, whose v6 reports grade speed indeterminate on every row;
  every record carries `source.dynamics_typecode`, so regenerate the 15 reports
  (`run_all_evaluations.py`, ~70 GB of reads, between GPU campaigns) — no backfill, no re-solve (a
  re-solve is ~30 h at `--jobs 24`, 12.3 GiB; the runner refuses to start if it does not fit).
