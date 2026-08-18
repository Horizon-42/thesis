# flight_scenarios — the data→modeling seam

Observed track → `FlightScenario` (initial + target `GeodeticState`, `AircraftSpec`,
`AeroParams`, source incl. `entry_time_utc`). Depends downward on modeling primitives,
imported upward by both consumers (optimizer and ts_transformer) — no cycles. It is a
**top-level package deliberately** (`4dTrajectory` isn't importable).

Everything below is a contract this seam owns; getting one wrong is silent, not loud.

## Vertical datum — HAE in, MSL out

- **Observed ADS-B altitude is ELLIPSOIDAL (HAE); everything it is judged against is MSL.**
  OpenSky `geoaltitude` is height above the WGS84 ellipsoid; runway thresholds, CIFP altitudes
  and the 8260.58D gates are orthometric. The gap is the geoid undulation N ≈ −25 to −33 m over
  the US (KRDU −33.53). Uncorrected, real completed airline landings scored **1.8 % on the
  gates** (18/996 KRDU) and the vertical gate passed ~0 %. Converted once at this seam by
  `flight_scenarios/datum.py` (EGM96 via pyproj).
- **Do NOT move this into the harvest**: CZML positions are consumed by Cesium as metres above
  the ellipsoid (`aeroviz-4d/src/types/czml.d.ts`) and are CORRECT as recorded — converting at
  the source fixes modeling and breaks the viewer by the same 33 m.
- The conversion is keyed on `altitude_source` (hence idempotent) and reaches THREE ingest paths
  — `load_model_arrivals`, `build_scenario`, and `ts_transformer/dataset.py` (which reads bare
  waypoints and so cannot self-protect); unknown/missing sources RAISE rather than defaulting,
  and `"synthetic"` is already-MSL.
- **The seam is symmetric on the way OUT**: modeling records (`*_states.json`, predictions) are
  MSL, and `build_scenario_comparison_czml._states_to_waypoints` — the single point every
  record-derived entity flows through — converts MSL→HAE via `aeroviz-4d/python/vertical_datum.py`
  (a deliberate MIRROR of `flight_scenarios/datum.py`, same KRDU N = −33.53 pin + ballpark probe;
  the modeling tree must not be imported there). The observed reference bypasses it (deep-copied
  from `trajectories.czml`, already HAE).
- Records are MSL by ASSUMPTION, not by tag — pre-datum-fix HAE-era artifacts are discarded
  wholesale (user decision); feeding one through the builder would double-shift it ~33.5 m low.
- **PROJ trap**: with the EGM96 grid missing and network off, pyproj silently returns a
  "ballpark" no-op vertical transform — a correction that looks applied and does nothing;
  `_geoid_transformer()` probes a known undulation and raises.

## Flight identity

- **A flight's identity is `flight_key` = `id_runway_icao24_landingTime`
  (`flight_scenarios.identity`), NEVER `id` alone — and this has bitten four separate layers.**
  The raw harvest carries **no unique flight id at all**: `id` is a copy of the callsign, and
  OpenSky stores state vectors by icao24 + time, so an "arrival" is a segment this project
  derives — identity is (which aircraft, when). Measured on the 996 KRDU arrivals: `id` → 552
  distinct, `icao24` → 717, `id_runway` → 778, `id_runway_icao24` → 874, **`icao24`+landing time
  → 996**. The extra fields in `flight_key` are for filename readability, not uniqueness.
- Casualties: the ts train/val/test split (leaked; `predict --split test` returned every
  namesake, 48 flights for an 18-flight split), the comparison-CZML group key (`id_runway`
  silently dropped 22% of a full batch), the FlightTable optimizer join (callsign-keyed;
  namesakes swapped V/mass/verdicts), and the observed-layer CZML entity ids (bare callsigns;
  Cesium merges same-id packets — per-runway files had up to 128 duplicate ids, two flights
  garbled into one entity).
- The same function produces the ts record stems, the optimizer's record filenames
  (`_scenario_filename` wraps it), the CZML group key (via the record filename stem), the
  observed-layer entity ids (`generate_czml`, which RAISES on a duplicate identity), and the
  comparison reference lookup — so they cannot drift. `aeroviz-4d/python/flight_identity.py` is
  a deliberate MIRROR (frontend tooling must not import the modeling tree); both copies are
  pinned to the vector `EJA969_05R_ad7f04_20260618T213736Z`, **change them together**.
- Corollaries: entity `name` (the callsign) is the ONLY display text (FlightTable/approach view
  render names, never ids); **positional `_N` id re-uniquing is deleted from the landing path**
  (`czml_export.classify_landing_flights`, `build_arrivals.merge_landing_flights`) because each
  harvest chunk restarted the numbering (merged files held duplicates anyway) and the
  combined-file renumbering gave the same flight DIFFERENT ids in different views — duplicate
  bare-callsign `id`s in landings/arrivals files are normal and correct; only the plain
  (non-landing) download path keeps `_unique_id`, since without `runway`/`landing_time_utc` the
  suffixed id is its only discriminator.

## Velocity / chart derivatives

- **Velocity is PHYSICAL at this seam, and the ts channels are chart derivatives (both since
  2026-07-20, B3.1).** `_velocity_lsq` projects through the true tangent scales (`R_M+h`,
  `(R_N+h)·cosφ`, via `geokit.wgs84_curvature_radii` — numeric single source; the casadi RHS and
  its mirror comment are the symbolic twin), so fitted `V/psi/gamma` mean what the dynamics model
  integrates; it used the flat chart constants before, overstating `V_north` by `a/R_M` (+0.33%
  at 36°). `channels.py` then maps physical → chart with the full-transport Jacobian, making
  `∫ edot dt` reproduce `e` exactly (unit-tested). Measured on 995 KRDU arrivals the residual
  integration drift is unbiased LSQ smoothing (~2.4–2.7 m/min median). **The two seams MUST move
  together**: fixing only the channels re-adds a +0.33% north systematic (measured 8.6 m/min).

## Runway target

- **THREE runway-threshold sources exist and two of them disagree by metres.**
  `flight_scenarios.runway_target.find_threshold` reads `runway_thresholds.json` (FAA NASR);
  `harvest.airports.Runway.lat/lon` prefers the CIFP Path Point LTP where an LPV procedure
  exists. Measured KRDU 05L: **6.69 m apart** (35.8745003/−78.802002 vs
  35.87444889/−78.80196361), and elevation 111.86 vs 111.80 m. The real pipeline is consistent
  (`arrivals._runway_target(runway)` copies the CIFP-resolved `Runway`, so scenario targets are
  bit-identical to the evaluation context), but `ts_transformer/synthetic.py` builds on the NASR
  point — which is why its test context pins the NASR coordinates explicitly.
  `evaluation.arrival._require_target_agrees_with_runway_data` now catches any such mix at 1 cm.

## Aircraft resolution

- **`"type": "UNK"` on every harvested arrival does NOT mean the batch is single-type.**
  `_resolve_aircraft` (`flight_scenarios/build.py`, mirrored in `ts_transformer/dataset.py`)
  tries declared type → **`icao24` via the OpenAP lookup** → `--aircraft-type` fallback, and the
  icao24 path recovers the REAL airframe for most flights: **20 distinct types** across 400 KRDU
  arrivals (A320 224, B738 38, E75L 25, CRJ9 23, … A333, GLF6, C550). Anything assuming one
  airframe per batch is wrong — that is exactly how the flyability check first shipped, grading
  ~44% of flights against an A320. The fallback (`--aircraft-type`, train default `A320`) is a
  `TSConfig` field, so it is recorded in the checkpoint and predict defaults to the train-time
  value; overriding it at predict shifts the ENU frames and the target Vref/threshold-crossing
  height the gates measure against, so it WARNS.
