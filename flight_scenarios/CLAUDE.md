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
  `flight_scenarios/datum.flight_to_msl`, which subtracts the flight's RUNWAY's CIFP offset
  `runway_target["hae_minus_msl_m"]` (HAE minus MSL elevation of the threshold: KRDU 05L −32.0 m) —
  one runway-local constant per flight, NOT EGM96 (−33.53 m there; 1.5 m apart). The measurements
  above were taken with EGM96. `datum.geoid_undulation_m` (EGM96 via pyproj) is a different tool:
  the Training view's MSL → HAE for executor-flown tracks (backend `autopilot_segment`,
  `executor_training_export`).
- **Do NOT move this into the harvest**: CZML positions are consumed by Cesium as metres above
  the ellipsoid (`aeroviz-4d/src/types/czml.d.ts`) and are CORRECT as recorded — converting at
  the source fixes modeling and breaks the viewer by the same 33 m.
- The conversion is keyed on `altitude_source` (hence idempotent) and reaches FOUR ingest paths
  — `load_model_arrivals`, `build_scenario`, `ts_transformer/data/dataset.py` (which reads bare
  waypoints and so cannot self-protect) and the harvest's observed records
  (`harvest/observed.observed_record`, since 2026-09-25); unknown/missing sources RAISE rather than
  defaulting, and `"synthetic"` is already-MSL. `HAE_ALTITUDE_SOURCE` is a MIRROR of
  `harvest.store.ALTITUDE_SOURCE` (the harvest imports `datum`, so importing back would cycle),
  pinned by `tests/test_datum.py`.
- **The seam is symmetric on the way OUT**: modeling records (`*_states.json`, predictions) are
  MSL, and `build_scenario_comparison_czml._states_to_waypoints` — the single point every
  record-derived entity flows through — adds the record's own `source.hae_minus_msl_m` back (the
  same runway offset, so the round trip is exact). The observed reference bypasses it (deep-copied
  from `trajectories.czml`, already HAE).
- Records are MSL by ASSUMPTION, not by tag — pre-datum-fix HAE-era artifacts are discarded
  wholesale (user decision); feeding one through the builder would double-shift it ~33.5 m low.
- **PROJ trap** (for `geoid_undulation_m`): with the EGM96 grid missing and network off, pyproj
  silently returns a "ballpark" no-op vertical transform — a correction that looks applied and does
  nothing; `_geoid_transformer()` probes a known undulation and raises.

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

## Population: who gets into a dataset

Full text: `docs/population_reference.md` (FS1–FS4, moved there verbatim 2026-09-16; FS5–FS6 added 2026-09-24;
FS7–FS9 2026-09-25).

- **`build_scenarios_from_arrivals` is strict; `dataset.build_scenario_dataset` is the batch
  layer**: a flight with no usable fitted final approach (~0.08 % of the roster) raises
  `UnusableFittedApproach` — its own type, so a broad `except ValueError` cannot swallow a real
  contract violation — and the batch layer drops and NAMES it (FS1).
- **`--max-per-runway` caps the population, and the cap is written down**: default 2000, evenly
  spaced over landing time, derived from the arrival roster alone, so both prepared datasets
  select the SAME flights (the `fitted_adsb` vs `runway` comparison stays paired); every decision
  lands in `<scenarios>.selection.json` — a bounded population not stated reads as a full one (FS2).
- **`source["dynamics_typecode"]` is what `evaluation`'s v9 threshold speed gate keys on** (the
  type the model FLEW); both record producers copy `scenario.source`, so this one write covers
  both; `source["landing_aero"]` is provenance only, no longer read (FS3).
- **`source["flight_key"]` is populated here** — only when the flight has an `id`, since the
  fallback would be a list index this function does not have (FS4).
- **A threshold target that cannot be built is refused**, never replaced by the track end
  (`build_scenario(target_from_threshold=True)`, 2026-09-25): on manifest input it cannot happen —
  the arrivals loader requires the TCH and glidepath, and no roster, live or frozen, has a flight
  without them (FS7).
- **The ts and optimizer populations are different flights and nothing joins them**: ts keeps what
  `UnusableFittedApproach` drops, applies its own aircraft filter and the lateral-pass roster; the
  optimizer applies the per-runway cap. Per-airport rates across the two are over different flights
  (FS8).

## Scene context

- **`scene_context` (no live consumer)**: a track that landed by t₀ is a landing, never a neighbour; a
  neighbour not closing on the threshold has no ETA; two samples at one time are refused; the
  membership constants are mirrors pinned to ts `final_approach_geometry`; `hour_utc` is UTC (FS9).

## Crossing span (observed records say where their crossing lives)

- **`crossing_span.crossing_span_from_event` is the producer half of the marker
  contract** (`final_approach.crossing` owns the schema + validator; see its
  `CLAUDE.md`). A direct event becomes a `measured_bracket` marker (no new rows);
  a censored event appends ONE inferred crossing row — event position, HAE→MSL via
  the caller's `hae_minus_msl_m` (converted exactly once, here), V = the event's
  crossing ground speed (fallback: last measured V, recorded as `v_source`),
  ψ/γ/m carried from the last established sample, t = trapezoidal
  `extrapolation / mean(V_last, V_cross)`. Consumed by
  `trajectory_data_process/harvest/observed.py` (the only span-producing record
  writer today — optimizer/ts references carry no event and stay markerless).

## Runway target

- **THREE runway-threshold sources exist and two of them disagree by metres.**
  `flight_scenarios.runway_target.find_threshold` reads `runway_thresholds.json` (FAA NASR);
  `harvest.airports.Runway.lat/lon` prefers the CIFP Path Point LTP where an LPV procedure
  exists. Measured KRDU 05L: **6.69 m apart** (35.8745003/−78.802002 vs
  35.87444889/−78.80196361), and elevation 111.86 vs 111.80 m. The real pipeline is consistent
  (`arrivals.runway_target(runway)` copies the CIFP-resolved `Runway`, so scenario targets are
  bit-identical to the evaluation context), but `ts_transformer/data/synthetic.py` builds on the NASR
  point — which is why its test context pins the NASR coordinates explicitly.
  `evaluation.arrival._require_target_agrees_with_runway_data` now catches any such mix at 1 cm.
- **The threshold target's speed is the airframe's PUBLISHED approach speed at the target mass**
  (since 2026-09-24): `approach.reference_speed_ms(m)` = the `aircraft/reference_speeds.json` row
  (FAA Aircraft Characteristics Database, at MALW) × sqrt(m / MALW) — the threshold speed gate's own
  law and upper reference, so the target sits inside the gate's window (on its lower edge for a
  single-valued type). Every 5.7–150 t type used to target 145 kt. A saved `runway_threshold` scenario
  whose V disagrees is refused at load; B3XM (no FAA row) has no dynamics; ts series never read the
  target speed (FS5).

## Aircraft resolution

- **Types without a native model are decided by `aircraft/performance_index.json`** (2026-09-24):
  preset → index row (own parameters / a substitute flown under ITS code / excluded) → OpenAP-direct;
  OpenAP synonyms are never flown under their own code; nothing flies as a stand-in (the A320 fallback is
  gone). The batch layer drops and names flights with no dynamics (`excluded_no_dynamics`, selection schema
  v2); a trajectory-only consumer asks `build_scenario(..., require_dynamics=False)` and gets a scenario
  WITHOUT dynamics (aircraft/aero `None`, mass and target V NaN), from which anything needing dynamics must
  ask `scenario.dynamics(purpose)` — it raises `NoAircraftDynamics` by name (FS6).
- **`"type": "UNK"` on every harvested arrival does NOT mean the batch is single-type.**
  `_resolve_aircraft` (`flight_scenarios/build.py`, mirrored in `ts_transformer/data/dataset.py`)
  tries declared type → **`icao24` via the identity resolver**, and the icao24 path recovers the
  REAL airframe for most flights: **20 distinct types** across 400 KRDU arrivals (A320 224, B738
  38, E75L 25, CRJ9 23, … A333, GLF6, C550). Anything assuming one airframe per batch is wrong —
  that is exactly how the flyability check first shipped, grading ~44% of flights against an
  A320. The `--aircraft-type` fallback that flew every unresolved type as an A320 was retired on
  2026-09-24 (ts: `aircraft_filter`, see FS6).
- **FAA registry → ICAO type: a documented model beats every heuristic** (2026-09-23).
  `aircraft/faa_icao_crosswalk.json` (TCDS / FSB report / JO 7360.1K + Doc 8643, one row per
  certificated model listing every registry spelling) overrides the name matcher and the OpenSky
  registration crosswalk, splits a model by serial where an FAA document prints the split, and an
  `unresolved` row or an uncovered serial leaves the airframe untyped — OpenSky may NOT fill it.
  Changing a row changes the openap-direct ts population; rebuild with
  `python -m aircraft.build_aircraft_identity_database --faa-zip … --icao-catalog aircraft/icao_doc8643.json`.
  Evidence and decisions: `docs/aircraft_identity/2026-09-23_faa_model_crosswalk_evidence.md`.
