# Coordinate-conversion consolidation — findings & proposal

**Status: IMPLEMENTED.** All five phases landed and verified. See **Outcome** below.

This documents every place the project re-implements coordinate / geodetic / unit
conversion math, the inconsistencies that duplication has introduced, and a concrete
plan to reduce it to **one canonical source per runtime domain**.

## Outcome (implemented)

The shared **`geokit`** package (src-layout, `pip install -e` into conda `aviation`) is now
the single source of truth for the Python side; the frontend mirrors it via a generated
`aeroviz-4d/src/generated/geoConstants.json`, consumed through `utils/procedureGeoMath.ts`.

- **Phase 0 — `geokit`**: `constants.py` (WGS84 ellipsoid, `SPHERE_RADIUS_M`=WGS84 a default +
  switchable `EARTH_RADIUS_MEAN_M`, `NM_M`/`FT_M`/`KT_MS`/`METRES_PER_DEG_LAT`/`DEG2RAD`) +
  `geodesy.py` (`haversine_m/km`, `equirectangular_distance_m`, `bearing_rad`,
  `flat_distance_m`, `metres_per_deg_lon`, `metres_per_degree_precise`, `bounds_from_radius_km`).
- **Phase 1 — data pipeline**: `trajectory_data_process/geo.py` is now a thin geokit re-export;
  `acquisition/*`, `generate_czml.py`, all `preprocess_*.py`, and `bc_lidar` source their
  haversine/bearing/bbox/ft·nm·deg constants from geokit. The 4 different "Earth radii"
  (`6371.0`/`6371.0088`/`6_371_000`/`6_371_008.8`) collapse to one (WGS84 a, spherical default).
- **Phase 2 — aero (constants-only)**: `casadi_coordinates_converter.py` imports WGS84 `a`/`e²`
  from geokit (re-exported, so `casadi_simulator`/`transport_term_comparison`/tests inherit it);
  the 3+ small-angle distance helpers (`dynamics_comparison`, `geodetic_vs_reanchored_error`,
  `transport_term_comparison`, `fixed_enu_frame_error`) delegate to geokit; the optimizer's
  normalization radius + duplicate `radians_expr`/`degrees_expr` are single-sourced; the
  `"must match"` coupling between the optimizer and `dynamics_comparison` is gone (both =
  `geokit.WGS84_A`). The **30 km study was regenerated** (JSON + HTML embedded data) for the
  ~0.1 % WGS84-a shift — A≈334.3 m, D≈144.9 m, C≈0.03 m (conclusions unchanged).
- **Phase 3 — frontend**: `geokit/scripts/export_constants_json.py` → `geoConstants.json`;
  `procedureGeoMath.ts` is the single TS geo module (imports the JSON). ~16 files migrated off
  local `toRadians/toDegrees`, `1852`, `0.3048`, `111320`, `6378137`. **The `6_371_008.8` vs
  `6_378_137` haversine split in `procedureGeometry.ts` is fixed** (now the shared WGS84 a).
- **Phase 4 — drift guard**: `geokit/tests/test_constants_json.py` fails if `geoConstants.json`
  drifts from `geokit.constants` (regenerate with the export script).

**Verification:** geokit 17 · trajectory_data_process 44 · aerodynamic_model 47 · optimizer 69 ·
backend 65 · aeroviz-4d/python 94 (+1 pre-existing unrelated `--include-transitions` orchestrator
failure) · frontend 292 + `tsc` + `vite build` — **no new failures**.

**Scoping note — kt→m/s deferred.** The `0.51444` knot→m/s factor (11 sites) was left as-is: it
is a *speed* unit conversion (not coordinate), already perfectly consistent everywhere, and
changing it to the exact `1852/3600` would alter thesis-optimizer physics by ~9e-6 across many
test fixtures — an unannounced change. `geokit.KT_MS` holds the exact value for new code.
Pedagogical helpers (`preprocess_airports.runway_bearing_rad`, the flat-Earth bearing) were kept
local but rebuilt on the shared constants.

---

## Decisions (final)

- **Scope:** consolidate **all three domains** (frontend, aero/optimization, data pipeline).
- **Domain C strategy:** **C2 — a shared `geokit` package** (single source for Python),
  not per-package mirror files.
- **Earth radius:** spherical helpers default to **WGS84 `a` = 6 378 137 m**, with the
  **mean radius 6 371 008.8 m kept as a switchable option** (flip one constant, or pass a
  per-call override). Ellipsoidal helpers always use the full WGS84 ellipsoid (`a` + `e²`).
- **`geokit` reach:** **constants-only into the aero domain.** `geokit` owns the numeric
  helpers + the single constants table; the CasADi converter
  (`casadi_coordinates_converter.py`) keeps its symbolic functions but **imports
  `geokit`'s constants** so values unify without touching the optimizer's tested math.
- **Packaging:** `geokit/` is a real package with `pyproject.toml`, installed
  **`pip install -e` into the conda `aviation` env**; consumers just `import geokit`.
- **TS sync:** a small script **generates `constants.json` from `geokit`**, which the TS
  geo module imports — single source, zero drift.
- **Aero study reproducibility:** **accept regeneration.** Everything uses WGS84 `a`
  uniformly; the 30 km dynamics-comparison study is regenerated and its expected numbers
  updated (~33 m / 30 km change) — an announced, deliberate change in its own commit.

---

## 1. TL;DR

- Coordinate math (haversine distance, bearing, ENU/ECEF, WGS84 radii, lat/lon→metres,
  deg↔rad, nm/ft conversions) is **re-derived in ~40 places** across the project.
- It cannot be literally *one* module, because the code spans **three runtimes that
  cannot share a source file** (browser TypeScript, CasADi/optimization Python,
  data‑pipeline Python). The honest target is **one canonical module per domain**,
  with **identical constant values** across all three.
- Duplication has already produced **real numeric inconsistencies** — e.g. the same
  "Earth radius" is written four different ways in the data pipeline (`6371.0`,
  `6371.0088`, `6_371_000`, `6_371_008.8`), and the frontend computes haversine with
  `6_378_137` in one file and `6_371_008.8` in another. These are latent bugs, not
  style nits.
- Two domains **already have a canonical module** to grow from
  (`aerodynamic_model/casadi_coordinates_converter.py` + `coordinates_convertor.py`;
  `aeroviz-4d/src/utils/procedureGeoMath.ts`). The work is mostly *routing the
  stragglers through them*, not building from scratch.

I'm asking you to approve **(a)** the per-domain-canonical approach, **(b)** the
data‑pipeline cross-package strategy (the one genuine architectural fork), and
**(c)** the scope (all three domains, or start with one).

---

## 2. Why it can't be a single module

The project is three independent runtimes. A `.ts` file can't be imported by Python,
and the two Python trees aren't one package:

| Domain | Where | Runtime | Can import the others? |
|---|---|---|---|
| **A. Frontend** | `aeroviz-4d/src/**` | browser TypeScript | no (different language) |
| **B. Optimization / aero** | `aerodynamic_model/**`, `4dTrajectory/**`, `aeroviz_backend/**` | Python + CasADi (symbolic) | no (different language from A) |
| **C. Data pipeline** | `trajectory_data_process/**`, `aeroviz-4d/python/**` | Python (numpy/stdlib) | partially — see §5 |

So "reuse one logic" realistically means: **one canonical module inside each domain**,
and a **single shared table of constant *values*** (§4) that all three mirror, so the
*numbers* never diverge even though the *code* can't be shared.

---

## 3. What was found (summary)

Full file:line inventory is in the Appendix. The headline duplication:

### Domain A — Frontend TypeScript

| Category | Copies | Notes |
|---|---|---|
| `toRadians` / `toDegrees` helpers | **6 files** | re-defined identically |
| `METERS_PER_NM` (1852) | **10+ sites** | exported once, re-declared 9×, hardcoded inline |
| `FEET_TO_METERS` (0.3048) | **7+ sites** | one file uses the reciprocal `3.280839895` |
| `METRES_PER_DEG_LAT` (111320) | **4 files** | all local |
| Haversine distance | 2 (+flat-Earth variants) | **inconsistent radius** (see §4) |
| Bearing / azimuth | 7 | spherical vs flat-Earth mix |
| Geo→ENU (`pointToEastNorth`) | 3 files | same formula re-typed |
| HPR→quaternion, ENU→ECEF | (Cesium native) | **good — leave alone** |

### Domain B — Optimization / aero Python

| Category | Copies | Notes |
|---|---|---|
| Geodetic↔ENU / ECEF | canonical exists, **+1 inline re-impl** | `geodetic_simulator.py` re-types e/n/u vectors instead of calling the converter |
| WGS84 radii `R_M`, `R_N` | **3** | `casadi_simulator.py`, `transport_term_comparison.py`, a test — *constants consistent* |
| Small-angle horizontal distance | **3** | `dynamics_comparison.py`, `geodetic_vs_reanchored_error.py`, `transport_term_comparison.py` — identical formula |
| `radians_expr` / `degrees_expr` | **2** | converter + optimizer re-declare |
| Normalization anchor radius | **2** | `optimizer.py` and `dynamics_comparison.py` with a literal `"Must match…"` comment — a coupling smell |
| Knots→m/s (`0.51444`) | **11+** | scattered across optimizer + tests |
| WGS84 `e²` derivation | **3** | converter, numeric converter, a docs script — re-derived from `f` each time |

### Domain C — Data-pipeline Python

| Category | Copies | Notes |
|---|---|---|
| Haversine distance | **5** | `geo.py`, `generate_czml.py`, `preprocess_waypoints/obstacles/procedures.py` — **4 different Earth radii** |
| Bearing | 2 | `generate_czml.py` (spherical), `preprocess_airports.py` (flat-Earth) |
| HPR→quaternion (`_mat3_*`, `_hpr_to_ecef_quaternion`) | 1 (only `generate_czml.py`) | hand-rolled; fine but isolated |
| metres-per-degree / bbox-from-radius | **6** | `geo.py`, `preprocess_airports.py`, `preprocess_usgs_tnm_terrain.py`, `bc_lidar_downloader.py` — **inconsistent deg→m** |
| `FT_TO_M` (0.3048) | **8 files** | each re-declares |

---

## 4. The inconsistencies worth fixing (latent bugs)

These are not cosmetic — different files silently compute different answers:

| Quantity | Values found | Where | Recommended canonical |
|---|---|---|---|
| **Earth radius (spherical/haversine)** | `6_378_137` (WGS84 *a*) vs `6_371_008.8` (mean) | frontend `procedureGeoMath.ts` vs `procedureGeometry.ts` | one value, project-wide |
| **Earth radius (data pipeline)** | `6371.0`, `6371.0088`, `6_371_000`, `6_371_008.8` | `geo.py`, `generate_czml.py`, `preprocess_*` | one value |
| **Mean Earth radius (aero)** | `6_371_000` | 3 distance helpers | align with the above |
| **metres per degree latitude** | `110.574` (km) vs `111.32` vs `111_320.0` vs a 4-term WGS84 polynomial | `geo.py` `bounds_from_radius_km` vs terrain/lidar vs `preprocess_airports.py` vs `preprocess_usgs_tnm_terrain.py` | one value (+ keep the precise polynomial only where precision matters, **named**) |
| **nm→m** | `1.852` (km) vs `1852.0` (m) | `geo.py` vs `preprocess_procedures.py` | `1852.0` m, one name |
| **WGS84 a / e²** | well-defined once, re-derived 3× | aero converter vs numeric converter vs docs script | import from the converter |

**Canonical values (decided):**
- **Spherical Earth radius** — default **WGS84 `a` = `6_378_137` m**, exposed as a single
  `SPHERE_RADIUS_M` constant the haversine/great-circle helpers default to. The **mean
  radius `6_371_008.8` m** (`EARTH_RADIUS_MEAN_M`) is kept alongside as the switchable
  option: flip `SPHERE_RADIUS_M` in one place to change everything, or pass
  `radius_m=EARTH_RADIUS_MEAN_M` to override a single call.
- **WGS84 `a = 6_378_137`, `e²` from `f = 1/298.257223563`** for *ellipsoidal* work
  (ECEF, R_M/R_N, the geodetic dynamics) — already correct in the aero converter, and
  **unaffected by the spherical switch**.
- **metres/deg lat = `111_320.0`** for cheap flat-Earth helpers; keep the precise
  polynomial (`preprocess_usgs_tnm_terrain.py`) as an explicitly-named
  `metres_per_degree_precise()` for terrain only.
- **`NM_M = 1852.0`, `FT_M = 0.3048`, `KT_MS = 0.514444`, `DEG2RAD/RAD2DEG`.**

The point: choosing the value is a one-time decision; consolidation makes it a
**single edit** forever after.

**Reproducibility note:** the aero dynamics-comparison study currently computes
horizontal error with the mean radius (`6_371_000`). Defaulting `SPHERE_RADIUS_M` to
WGS84 `a` shifts those numbers ~0.1% (~33 m / 30 km) and regenerates the study output.
Either pin those specific call sites to `EARTH_RADIUS_MEAN_M` (keeps the study
byte-identical) or accept the regeneration — see §7.

### Why the values diverged (root cause)

There's no shared home a constant could live in across the three runtimes (and the two
Python trees aren't one package), so each new file typed the number from memory rather
than importing it. The *specific* divergences are all **individually-valid** numbers,
because Earth is an ellipsoid (equatorial `a` 6 378 137 m vs polar `b` 6 356 752 m vs
mean R₁ 6 371 008.8 m), and each author reached for whichever "famous" radius they had
at hand, in whichever **base unit their module worked in** (km vs m → `6371.0` vs
`6_371_000`; `1.852` vs `1852.0`). Precision needs were judged locally (terrain wanted
the precise metres-per-degree polynomial; approach geometry didn't), and nothing ever
reconciled them. It went unnoticed because **every value is individually correct**, so
no test fails — there's no forcing function to converge.

### Where the spherical helpers are used (why the radius choice is low-stakes)

The haversine/spherical helpers feed three kinds of consumer; in almost all the radius
is an `argmin` (cancels), a fuzzy threshold (0.1 % irrelevant), or a difference/scale
factor (only needs to be *consistent*, not *accurate*):

1. **Proximity filtering & selection** (data pipeline) — keep-within-N-km, closest-point
   anchor, landing detection: `czml_export.py:58/157/173`, `preprocess_waypoints.py:171`,
   `preprocess_obstacles.py:338`. Radius effect: none (thresholds are fuzzy; argmin cancels).
2. **Procedure/profile geometry** (frontend + pipeline) — leg lengths, cumulative
   distance-from-start, fix-to-fix nm, horizontal run for pitch: `procedureGeoMath.ts:53/61`,
   `procedureSegmentGeometry.ts:178`, `procedureGeometry.ts:101`,
   `preprocess_procedures.py:141/1219`, `generate_czml.py:123`. Radius effect: ~0.1 %,
   below displayed/chart resolution.
3. **Deviation / error metrics** (aero + Pilot panel) — the dynamics-comparison study's
   horizontal error and the live target deviation: `dynamics_comparison.py:323/363`,
   `geodetic_vs_reanchored_error.py:190`, `transport_term_comparison.py:188+`,
   `PilotPanel.tsx:512`. Radius effect: a uniform scale on a small difference — only
   *consistency across systems* matters (already all mean radius).

So unifying the radius is about **consistency + stopping the drift recurring**, not
fixing a visible error. The one place absolute sizing genuinely matters is **bbox query
sizing** (`bounds_from_radius_km`, a metres-per-degree constant, not the radius) — too
small can drop edge rows.

---

## 5. Proposed solution — one canonical module per domain

### Domain A (frontend) — grow `procedureGeoMath.ts` into the geo module
It already exports `EARTH_RADIUS_M`, `METERS_PER_NM`, `FEET_TO_METERS`,
`haversineDistanceM`, `bearingRad`, `toCartesian`, `toRadians/toDegrees`. Plan:
- Add the missing shared pieces (`METRES_PER_DEG_LAT`, `pointToEastNorth`,
  `offsetGeoFromMetres`, ENU helpers).
- Replace the 6 local `toRadians`, the 10+ `1852`, the 7 `0.3048`, the 3
  `pointToEastNorth`, etc., with imports.
- Fix `procedureGeometry.ts` to use the shared radius (kills the `6_371_008.8` vs
  `6_378_137` split).
- Leave Cesium-native transforms as-is (they're the right tool).

### Domain B (optimization / aero) — already has the canonical converter
`casadi_coordinates_converter.py` (symbolic) + `coordinates_convertor.py` (numeric)
already hold WGS84 constants, ECEF, ENU, deg/rad. Plan:
- Add a tiny `aerodynamic_model/geo_constants.py` (or extend the converter) exporting
  `R_MEAN`, `KT_MS`, `NM_M`, `FT_M`, `M_PER_DEG_LAT`, re-exporting `WGS84_A/E2`.
- Route the 3 small-angle distance helpers, the duplicate `radians_expr`, the 11
  `0.51444`, and the "must match" normalization radius through it.
- Make `geodetic_simulator.py` call the converter's ENU rotation instead of inlining
  e/n/u vectors.
- WGS84 radii (`R_M`/`R_N`) → one helper in the converter, used by
  `casadi_simulator.py` and `transport_term_comparison.py`.

### Domain C (data pipeline) — **one decision needed** (see §7)
`trajectory_data_process/` is an importable package (`geo.py` is its natural home);
`aeroviz-4d/python/` is a *non-package* directory of standalone scripts that already
can't cleanly import the other tree (same hyphen / `sys.path` issue we hit with
`generate_czml.py`). Two ways to give this domain a single source of truth:

- **Option C1 — one module per package (recommended, low-risk).**
  `trajectory_data_process/geo.py` stays canonical for that package; add
  `aeroviz-4d/python/geo.py` as canonical for that tree. Both carry **identical
  values** from §4. Eliminates all *intra*-package duplication (5 haversines → 1 per
  tree) without any cross-tree `sys.path` coupling. Downside: two small mirror files
  (one per package), by necessity.
- **Option C2 — one shared installable `geokit` package both import.**
  Promote a real top-level Python package (e.g. `geokit/`) that *both*
  `trajectory_data_process` and `aeroviz-4d/python` (and optionally Domain B) depend
  on. True single source for all Python. Downside: larger change — needs packaging
  (`pyproject`/`__init__`), and `aeroviz-4d/python` would need to become importable
  (it currently isn't). This is the "do it properly" option.

---

## 6. What to leave alone (intentional / already good)

- **Cesium native transforms** (`Transforms.eastNorthUpToFixedFrame`,
  `headingPitchRollQuaternion`) — correct, full-ellipsoid, no reason to hand-roll.
- **Flat-Earth vs spherical vs ellipsoidal** is sometimes a *deliberate* accuracy/speed
  trade (e.g. <20 km approach geometry uses flat-Earth on purpose). Consolidation keeps
  both, but as **named functions** (`flatDistanceM` vs `haversineDistanceM`) so the
  choice is explicit, not accidental.
- The **geodetic dynamics RHS / transport terms** — one definition already in
  `casadi_simulator.py`; only the *constants* it uses get centralized.

---

## 7. Decisions — all resolved

See **Decisions (final)** at the top. Nothing outstanding.

---

## 8. Implementation plan

### 8.1 `geokit` package layout

```
geokit/
  pyproject.toml            # installable: pip install -e . into conda `aviation`
  geokit/
    __init__.py             # re-export the public surface
    constants.py            # the single source of truth (see below)
    geodesy.py              # numeric helpers: haversine_m, bearing_rad/deg,
                            #   geo<->ecef, geo<->enu, R_M/R_N, metres_per_degree,
                            #   metres_per_degree_precise (terrain), flat_distance_m
    units.py                # nm_to_m, ft_to_m, kt_to_ms, deg<->rad
  scripts/
    export_constants_json.py  # emits aeroviz-4d/src/generated/geoConstants.json
  tests/
    test_geodesy.py
```

`constants.py` (the canonical values):

```python
WGS84_A             = 6_378_137.0          # ellipsoid semi-major axis (equatorial)
WGS84_F             = 1.0 / 298.257223563
WGS84_E2            = WGS84_F * (2.0 - WGS84_F)
EARTH_RADIUS_MEAN_M = 6_371_008.8          # IUGG mean radius R₁ — switchable option
SPHERE_RADIUS_M     = WGS84_A              # ← default for haversine/great-circle; flip to switch
METRES_PER_DEG_LAT  = 111_320.0
NM_M = 1852.0;  FT_M = 0.3048;  KT_MS = 0.514444
```

### 8.2 Phasing (each phase ends green; one PR/commit per phase)

**Phase 0 — land `geokit` (additive, zero call-site changes).** Create the package,
constants, numeric helpers, tests; `pip install -e` into `aviation`. Nothing else
imports it yet → every existing suite still byte-identical.

**Phase 1 — Domain C (data pipeline) onto `geokit`.** Replace the 5 haversines, the 8
`0.3048`, the metres/deg + bbox helpers, the `generate_czml.py` bearing, etc., with
`geokit` imports. `trajectory_data_process/geo.py` becomes a thin re-export (keeps its
km-based public API where callers expect km). Run `pytest trajectory_data_process/tests`
+ the `generate_czml` golden + procedure goldens.

**Phase 2 — Domain B (aero) constants-only.** Point
`casadi_coordinates_converter.py`’s `WGS84_A/E2` and the 3 small-angle distance helpers,
the 11 `0.51444`, the "must-match" normalization radius, and the duplicate `radians_expr`
at `geokit`’s constants/units. **Symbolic functions stay put.** Then the
**study-regeneration commit** (separate): default spherical radius → WGS84 `a`, regenerate
the 30 km study + `dynamics_comparison_30km_data.json`, update expected numbers. Run the
aerodynamic_model + optimizer + backend suites.

**Phase 3 — `constants.json` codegen + Domain A (frontend).** Add
`scripts/export_constants_json.py` → `src/generated/geoConstants.json`; make
`procedureGeoMath.ts` the single TS geo module that imports it; replace the 6 `toRadians`,
10+ `1852`, 7 `0.3048`, 3 `pointToEastNorth`, and fix `procedureGeometry.ts`’s radius.
Run `vitest` + `tsc` + `vite build`.

**Phase 4 — guards.** A `pytest`/`vitest` test asserting `geoConstants.json` matches
`geokit.constants` (codegen can’t silently drift); a lint note discouraging re-declaring
these constants.

### 8.3 Risk

Low for the pure de-duplication phases (byte-identical, test-guarded). The single
**study-regeneration commit** in Phase 2 deliberately changes numbers ~0.1% and is
reviewed on its own. Rough effort: Phase 0 ≈ half a day, Phase 1 ≈ half a day, Phase 2 ≈
half a day (+ study regen), Phase 3 ≈ half a day, Phase 4 ≈ small.

---

## Appendix — full inventory

> Line numbers are from a survey sweep and will be reconfirmed during implementation.

### A. Frontend (`aeroviz-4d/src/`)
- Haversine: `utils/procedureGeoMath.ts:41` (`6_378_137`), `utils/procedureGeometry.ts:47` (`6_371_008.8`), flat-Earth variants `data/procedureRoutes.ts:90`, `data/rnavInitialFixCandidates.ts:240`.
- Bearing: `utils/procedureGeoMath.ts:87`, `utils/ocsGeometry.ts:84`, `utils/procedureGeometry.ts:59`, `utils/runwayProfileGeometry.ts:150`, `data/procedureConstraint.ts:89`, `data/runwayThresholdTargets.ts:120`, `data/rnavInitialFixCandidates.ts:269`.
- Geo→ENU: `utils/procedureDetailsGeometry.ts:69`, `utils/runwayProfileGeometry.ts:173`, `utils/procedureProtectionVolumeAssessment.ts:55`; ENU→Geo `hooks/usePilotTargetGate.ts:103`, `hooks/usePilotInitialPlacement.ts:445`.
- Geo→ECEF: `utils/procedureGeoMath.ts:29`; Cesium transforms `hooks/useRangeRingLayer.ts:40`, `hooks/usePilotAircraft.ts:68`, `hooks/usePilotInitialPlacement.ts:222`, `hooks/useOptimizedTrajectoryPlayback.ts:273`.
- `toRadians`/`toDegrees`: `procedureGeoMath.ts:17`, `procedureDetailsGeometry.ts:65`, `runwayProfileGeometry.ts:126`, `data/procedureConstraint.ts:78`, `data/runwayThresholdTargets.ts:141`, `data/rnavInitialFixCandidates.ts:289`.
- `1852`: `procedureGeoMath.ts:3` (canonical) + local in `procedureGeometry.ts:12`, `RunwayTrajectoryProfilePanel.tsx:28`, `ProcedureDetailsPage.tsx:74`, `usePilotInitialPlacement.ts:35`, `useOcsLayer.ts:47`, `pilot/trajectoryTargetConstraints.ts:2`, `data/procedureRoutes.ts:18`.
- `0.3048`: `procedureGeoMath.ts:2` (canonical) + `procedureGeometry.ts:13`, `data/procedureConstraint.ts:30`, `data/runwayThresholdTargets.ts:5`, `data/rnavInitialFixCandidates.ts:10`, `pilot/trajectoryTargetConstraints.ts:1`, reciprocal in `utils/procedureVerticalProfileOverlay.ts:4`.
- `111_320`: `ocsGeometry.ts:22`, `procedureGeometry.ts:11`, `usePilotTargetGate.ts:25`.

### B. Optimization / aero (`aerodynamic_model/`, `4dTrajectory/`, `aeroviz_backend/`)
- Geodetic↔ENU/ECEF (canonical): `casadi_coordinates_converter.py:17-89`, `coordinates_convertor.py:35-113`; high-level `casadi_simulator.py:143-173`; consumers `casadi_direct_collocation_optimizer.py:222-240`, `dynamics_comparison.py:267-272`; **inline re-impl** `geodetic_simulator.py:44-110`.
- WGS84 radii: `casadi_simulator.py:371-375`, `transport_term_comparison.py:94-99`, `tests/test_casadi_simulator.py:376-383`.
- Small-angle distance: `dynamics_comparison.py:82-87`, `geodetic_vs_reanchored_error.py:97-106`, `transport_term_comparison.py:82-87` (all `6_371_000`).
- `radians_expr`/`degrees_expr`: `casadi_coordinates_converter.py:9-14`, `casadi_direct_collocation_optimizer.py:95-100`.
- Normalization radius: `casadi_direct_collocation_optimizer.py:88`, `dynamics_comparison.py:62` (`"Must match…"`).
- Knots→m/s `0.51444`: `casadi_optimizer.py:139`, `casadi_direct_collocation_optimizer.py:960`, `geodetic_vs_reanchored_error.py:155`, + many tests.
- `FEET_TO_METERS`: `aeroviz_backend/procedure_constraint.py:23`.

### C. Data pipeline (`trajectory_data_process/`, `aeroviz-4d/python/`, `bc_lidar_downloader/`)
- Haversine: `trajectory_data_process/geo.py:12` (`6371.0`), `aeroviz-4d/python/generate_czml.py:75` (`6_371_000`), `preprocess_waypoints.py:106` (`6371.0088`), `preprocess_obstacles.py:74` (`6371.0`), `preprocess_procedures.py:96` (`6_371_008.8`).
- Bearing: `generate_czml.py:57`, `preprocess_airports.py:87`.
- HPR→quaternion: `generate_czml.py:132-222` (`_mat3_multiply`, `_mat3_to_quaternion`, `_hpr_to_ecef_quaternion`), velocity→heading/pitch `generate_czml.py:89-127`.
- metres/deg + bbox: `geo.py:26` (`110.574`/`111.320`), `preprocess_airports.py:70-125` (`111_320`), `preprocess_usgs_tnm_terrain.py:403-416` (polynomial) + `:123` (`111.32`), `bc_lidar_downloader/download_bc_lidar.py:348` (`111.32`).
- `0.3048`: `acquisition/airports.py:10`, `acquisition/runways.py:14`, `acquisition/opensky_history.py:43`, `preprocess_procedures.py:69`, `preprocess_obstacles.py:56`, `preprocess_airports.py:40`, `preprocess_usgs_tnm_terrain.py:61`.
- `nm`: `geo.py:9` (`1.852` km), `preprocess_procedures.py:70` (`1852.0` m).
