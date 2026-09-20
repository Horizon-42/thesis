# trajectory_data_process — harvest contracts, static data, constants (reference)

Full text behind the index lines of `trajectory_data_process/CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' trajectory_data_process/docs/06-harvest-reference.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Contracts

### TD1 · the threshold event carries `crossing_ground_speed_m_s` (ground speed)

- **The threshold event carries `crossing_ground_speed_m_s` (additive within
  `runway-threshold-event-v1`, 2026-08-24), and it is GROUND speed.** Direct events
  interpolate the bracketing samples' `reported_ground_speed_m_s` at the position's
  own fraction; censored events OLS-extrapolate speed vs along-track over the SAME kept
  samples as the position fit (`final_approach.fit_line`, the shared estimator), with the
  fit's own sample/span standard — an unfittable speed omits the field and
  `diagnostics.ground_speed_fit.omitted_reason` says why. OPTIONAL on read: only a NEW
  harvest or `--reclassify-existing` populates stored events (`--evaluate-only`
  re-rosters them unchanged). Evaluation judges the observed baseline on it as a STATED
  ground-speed proxy (owner decision 2026-08-24 — see `evaluation/CLAUDE.md`); the
  wind caveat travels with the proxy criterion id, never silently.

### TD2 · observed evaluation records carry `crossing_span` and the resolved airframe

- **Observed evaluation records carry a `crossing_span` and their resolved airframe's
  stall facts (2026-08-24)** — `harvest/observed.py` marks the event's direct bracket or
  appends the one inferred crossing row (built by `flight_scenarios.crossing_span`), so
  evaluation grades the STATES through one shared interpolation instead of re-reading
  the event; and it resolves each flight's IDENTITY from its icao24
  (`flight_scenarios.resolve_airframe`, the scenarios' own resolver) to write
  `source.aircraft_type` — the ICAO type the baseline speed gate looks its PUBLISHED
  approach-speed window up by (v9) — **whenever the identity resolves, dynamics or
  not**: the states' mass is the type's OpenAP landing mass when OpenAP models the
  type, else `NOMINAL_MASS_KG` (`source.mass_source` = `openap_landing_mass` /
  `nominal` / `explicit`; the gate never reads an observed record's mass). Until
  2026-09-08 the type was written only when OpenAP had dynamics, which silently
  dropped 9,056 of the 10,541 untyped observed rows (BCS3 1,582, E55P 719, CRJ7 674,
  BCS1 482, P28A 409, … 167 types) — 21 % of the fleet graded speed-indeterminate for
  want of a mass nothing used. An unresolvable identity (1,485 rows: 1,164 FAA models
  with no unambiguous ICAO type, 283 unmatched, 38 OpenSky codes absent from the Doc
  8643 snapshot) gets no `aircraft_type`. `source.landing_aero` (v6–v8's stall inputs)
  is no longer written.

### TD3 · `--observed-only` rebuilds only `approach/`

- **`--observed-only` rebuilds ONLY `approach/`** (records, `summary.json`, the report and
  its publication) from stored tracks — `arrivals/` and `lateral_pass_eligibility.json`
  untouched, no CZML. It is the mode for an observed-record contract change (a new
  `source` field, a resolver fix); `--evaluate-only` also rebuilds `arrivals/` and
  deletes the lateral roster, `--reclassify-existing` re-derives assignment. The
  regenerated report's hash still moves the ts `data_provenance` (the roster records
  it; `docs/open-items.md`).
  Target kinematics and `final_time_s` stay anchored to the last MEASURED row. Records
  in `approach/records/` from before this date fail evaluation loudly ("no
  crossing_span") — rebuild with `--evaluate-only`.

**Correction (2026-09-16):** the last three lines of this item ("Target kinematics … rebuild with `--evaluate-only`") belong to TD2 — they were stranded when the `--observed-only` bullet was inserted above them — and the cure they name is out of date: a record without `crossing_span` is rebuilt with `--observed-only`, which regenerates `approach/` alone; `--evaluate-only` also rebuilds `arrivals/` and deletes `lateral_pass_eligibility.json` (TD17).

### TD4 · `unassignable` and `not_established` stay distinct

- **`unassignable` (the receiver lost it) and `not_established` (the approach was not
  stabilised) must stay distinct outcomes.** Conflating them charges a reception gap to the
  pilot. Both are counted and reported; neither is ever dropped (established rate is 21–54 % on
  real data).

### TD5 · archived history rows are in feet, live rows in metres

- **Archived history rows are in FEET; `fetch_history_dataframe` output is in METRES.**
  `_altitudes_to_metres` converts on the live path only, so rows read back from
  `outputs/history_rows/**.jsonl` are raw feet. Guessing wrong scales every altitude by 3.28 and
  does not crash — it turns a 3.06° approach into a 9.94° one (observed).
  `harvest.reconstruct_tracks` therefore takes a **required** `altitude_units` argument with no
  default and no sniffing.

### TD6 · an arrival segment must not begin on the ground (`takeoff_in_segment`)

- **An arrival segment must not BEGIN on the ground, and the local-circuit test cannot catch
  that on its own.** `LOCAL_START_RADIUS_KM` measures distance from the DESTINATION, so a
  takeoff from a NEIGHBOURING field inside the 25 km ring passed it and entered the dataset
  as a "coverage-limited arrival" starting on a runway a few km away. Measured: **75 flights,
  64 of them KSJC** (KRHV ×28 at 7 km, KPAO ×20 at 21 km, KNUQ ×16 at 11 km), KSMF 8, KMSY 3,
  KRDU and KSTL **zero** — KSJC is the only airport of the five ringed by satellite fields.
  Every one was long enough to reach the TS dataset. `arrival_segment` now returns a third
  kind, `"takeoff"`, rostered as `excluded.outcome = "takeoff_in_segment"`.
  **The test is ALTITUDE ONLY and must stay that way**: a jet at rotation reads 71–80 m/s on
  the runway, inside the approach-speed range, so adding a ground-speed condition would keep
  29 of the 75. It is applied to the segment the ring cut PRODUCED, not the raw track — a
  flight that departs a neighbour, leaves the ring and comes back is a genuine arrival whose
  takeoff was already cut away. Full analysis: `docs/2026-08-21_ksjc_route_mix_and_ade.md`.

### TD7 · a landing ends at a sustained on-ground run; keep the final contiguous run

- **Track reconstruction: a landing ends at a sustained ON-GROUND run, and a spatial crop must
  keep only the final contiguous run.** An aircraft that lands keeps transmitting from the gate,
  so state vectors are continuous across a turnaround and the >900 s gap rule never fires; the
  radius filter then removed the middle and glued two passes together — KSTL AAL2717 carried a
  **6598 s** hole with two stray samples after it, which put the flight's `landing_time_utc`
  (part of its identity) on the WRONG pass. Also: `_complete()` must require **both** dep and arr
  airports, not either — `(None,'KRDU') → ('KATL','KRDU')` is one flight with the origin merely
  resolved, and the old rule cut the approach in half.

## Runway thresholds & TCH (static data feeding the harvest)

### TD8 · `runway_thresholds.json` holds landing (displaced) thresholds

- **`runway_thresholds.json` holds LANDING thresholds (displaced), not pavement ends**
  (`runway-thresholds-v2`, `displaced_threshold_m` on every entry). `build_runway_config.py` used
  to ignore `*_displaced_threshold_ft`: KSJC 30L/30R are displaced **775 m**, which on a 3°
  glidepath is a **40.6 m** altitude error and moved the OPTIMIZER TARGET, not just the gates.
  Six thresholds are displaced (KSJC ×4, KSTL 12R 143 m, KMSY 29 93 m). **Fix the GENERATOR,
  never the JSON.** The landing-threshold computation is single-sourced in
  `acquisition/runways.py` (`landing_thresholds_from_row`), shared by the generator AND
  `resolve_runway_threshold` — the `download_trajectories.py --runway` path previously kept
  pavement ends, naming a point up to 775 m away from the config's and shifting
  `landing_time_utc` (hence `flight_key` identity) between the two harvest paths.
  Corollary worth remembering: KSJC looked *healthiest* before the fix (+9.7 m vs everyone
  else's −25 m) because its displaced-threshold and datum errors had opposite signs and nearly
  cancelled — the airport that looks best can be the one with two bugs.

### TD9 · per-runway TCH is published in the CIFP, not 15 m

- **Per-runway TCH is published in the CIFP and is NOT 15 m.** ARINC 424 section P / subsection P
  "Path Point" records (`data/CIFP/<cycle>/FAACIFP18`) carry glidepath angle, course width and
  threshold crossing height per LPV approach; `harvest/cifp.py` decodes them. Every runway in the
  fleet publishes **15.27–18.11 m**, so the old flat 15 m assumption put a systematic 1.5–2.5 m
  bias into a 9.15 m window. Using the published value moved the measured KSMF crossing from
  +2.74 m to **+0.61 m** — real traffic crosses where the plate says, to within half a metre,
  which is the best end-to-end check that the datum, the fit and the TCH source all agree.
  The column decode is pinned by a coincidence-proof cross-check: **4795 of 4900 records decode a
  course width of exactly 106.75 m**, independently the LPV semiwidth in `evaluation/thresholds.py`.
  A runway with no Path Point record has **no LPV procedure** (KRDU 14/32, KSMF 35R), but that
  is not the end of its vertical path: the RNAV (GPS) approach's **runway leg** (section P /
  subsection F, the leg whose fix is `RWxx`) codes the vertical angle and the altitude at which
  the path crosses the runway, and its approach-types continuation says whether LNAV/VNAV
  (Baro-VNAV) minima are published. `read_approach_verticals` decodes that and `load_airport`
  fills `Runway.threshold_crossing_height_m` / `published_glidepath_deg` from it when LNAV/VNAV
  is published (`tch_source == "faa_cifp_approach_leg"`, `baro_vnav_minima`): **KRDU 32 =
  3.50° / 470 ft − 425 ft = 45 ft (1,604 arrivals), KSMF 35R = 3.00° / 64 ft (259)**. The decode
  is pinned per airport against the Path Points (the LPV-bearing procedure's leg must equal
  LTP + TCH within 1 ft and match the glidepath — all 23 fleet LPV runways do, within 0.8 ft).
  Only a runway with **no RNAV procedure at all** keeps `None` (KRDU 14, 13 arrivals). The
  runway record (PG) publishes a TCH too, but it is the ILS/VGSI figure and differs from the RNAV
  one by ~4 ft at KSTL 06/24 — a per-procedure quantity, so the procedure's own leg is what is
  read. `tch_source` and `baro_vnav_minima` are NOT in `threshold_frame_snapshot`, so every
  stored threshold event on those runways stays valid. **Re-running the harvest after this
  change grows the arrivals rosters (KRDU +1,617, KSMF +259) and therefore every ts dataset
  split** — do it between campaigns and rebuild `lateral_pass_eligibility.json` afterwards.

**Note (2026-09-16):** "a 9.15 m window" is the vertical gate this text was written against; it has been ±22 m (`evaluation.thresholds.RNAV_TERMINAL_VERTICAL_BOUND_M`) since c9ca54b (2026-08-15). The TCH bias measurement itself stands.

## Altitude outlier repair

### TD10 · outliers are repaired in the view, read-time

- **ADS-B altitude outliers are repaired in the VIEW, never in `tracks/` — and the repair is
  read-time, so no artifact needs rebuilding to get it.** A few state vectors report an
  unreachable altitude (measured extremes: **20 147 m between neighbours at 724 m**, 35 189 m at
  556 m), which renders as a needle and drags any fit through it.
  `harvest/altitude_filter.py` replaces those altitudes where a stored track is read into a
  derived view — `store.read_track_view` (observed CZML, evaluation records) and
  `arrivals.write_arrival_records`/`load_arrival_flights` (training data), which hash the SOURCE
  bytes first and filter after. Editing the track files instead breaks three things at once: the
  arrival roster's per-record SHA-256, `--reclassify-existing`, and
  `source_integrity.retained_rows`.

### TD11 · the two-part criterion

- Criterion: deviation from the median of the ±2-sample window exceeding BOTH 100 m AND
  `25 m/s × min(adjacent gap)`. Both halves are load-bearing — 100 m alone repairs 10 real
  descents that stepped 107–160 m across 9–14 s reception gaps into lies, and a **chord/jump test
  attributes one bad sample to three** (the outlier's two neighbours fail too; measured 363
  runs-of-3 where the truth was 363 isolated samples).

### TD12 · measured incidence; only the altitude changes

- Measured incidence: **561 samples in 451 of 44 622 assigned tracks (0.0027 %)**, 421 inside a
  model arrival slice. Only `samples[i][3]` changes; dropping a row would silently renumber
  `landing_sample_index`, the arrival slice bounds, the event's `source_sample_range`, and
  `reported_ground_speeds_m_s`.

### TD13 · audit / republish; what is not covered

- Audit/republish with `python -m trajectory_data_process.altitude_outliers [--rerender-czml]`
  (reads `tracks/`, writes only `public/data`). **Not covered:** stored
  `observed_threshold_event`s were fitted from raw samples during assignment — 17 outliers land
  inside one, and only `--reclassify-existing` re-derives those.

## Constants

### TD14 · altitude outlier filter constants

- Altitude outlier filter (`harvest/altitude_filter.py`, single source; `AltitudePolicy`):
  `half_window = 2`, `min_deviation_m = 100.0`, `max_vertical_rate_m_s = 25.0`. Repair = linear
  interpolation in time between the nearest retained samples; an outlier at a track edge HOLDS
  the nearest retained altitude (labelled `held` vs `interpolated`). The 100 m floor is 13× the
  25 ft quantum, 3.3× the 100 ft quantum, and 2× the largest residual genuine flight produces
  (over 20.85 M samples: 20 847 051 below 25 m, 3 625 in [25, 50), 189 above 50 m). Counts are
  reported in `arrivals/manifest.json` → `altitude_filter`, `approach/summary.json` →
  `altitude_filter`, and `RenderedObserved.altitude_outliers`.

### TD15 · arrival truncation constants

- Arrival truncation (`trajectory_data_process/arrival_segment.py`): `ENTRY_RADIUS_KM = 25`
  (builder `--entry-radius-km`, in (0, 30)), `ENTRY_HYSTERESIS_SAMPLES = 3`,
  `LOCAL_START_RADIUS_KM = 5`, `GROUND_START_AGL_M = 100` (published per manifest as
  `ground_start_agl_m`). The 100 m sits in an EMPTY band: over 42 725 rostered arrivals the
  first-sample height above the landing runway is bimodal — 75 flights at or below 82.1 m, **zero
  between 100 and 150 m**, next at 175.3 m — so no flight in the fleet is near the boundary.
  `arrival_segment` takes `field_elevation_m` as a REQUIRED argument with no default: the
  waypoint rows are HAE and a silently MSL reference would shift the test by the geoid
  separation (~33 m) without failing. `harvest/arrivals.py` asserts the datum once, at the
  boundary.

### TD16 · arrival manifest schema

- Arrival manifest schema: **`harvest-arrivals-v5-takeoff-excluded`** (v4 excluded no
  takeoffs). Loaders compare exactly, so a v4 manifest on disk fails loudly; rebuild with
  `python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only`, which
  re-rosters from stored `tracks/` without downloading or reassigning anything.

**Correction (2026-09-16):** the WRITER is now `harvest-arrivals-v6-published-vertical-path` (`harvest/arrivals.py` `SCHEMA_VERSION`) and loaders accept v5 AND v6 (`READABLE_SCHEMA_VERSIONS`), so "loaders compare exactly" now means against that pair; a v4 manifest still fails loudly. All five manifests on disk were still v5 when read on 2026-09-16. A rebuild writes v6 and grows the rosters (TD9).

### TD17 · rebuilding `arrivals/` deletes `lateral_pass_eligibility.json`

- **Rebuilding `arrivals/` DELETES `lateral_pass_eligibility.json`, which nothing rebuilds
  for you.** `arrivals._clear()` unlinks every `*.json` under the directory before writing
  the new manifest, and the lateral-pass roster lives there but is owned by
  `ts_transformer/data/lateral_eligibility.py`. So any `--evaluate-only` or
  `--reclassify-existing` silently removes a file every TS train/predict requires, and the
  failure surfaces later as `FileNotFoundError` from whichever run touches it next — on
  2026-08-21 that was mid-campaign, after one arm had already trained. Rebuild it right
  after the harvest, from the regenerated approach report:
  `lateral_eligibility.ensure_lateral_pass_roster(<arrivals>/manifest.json)`.

### TD18 · `--reclassify-existing` is not the rebuild command

- **`--reclassify-existing` is NOT the rebuild command** — it re-derives runway assignment
  and is only for a changed runway-data cycle or assignment method. For a schema bump or an
  evaluation-policy change use `--evaluate-only`; reaching for the heavier one risks a
  different roster for no reason.

### TD19 · a harvest killed with SIGTERM can still finish its write

- **A harvest killed with SIGTERM can still finish its write.** On 2026-08-21 a
  `--reclassify-existing` run killed mid-flight completed several minutes later, cleared the
  directory again, and overwrote a manifest a training job was already reading — the arm had
  to be discarded. Confirm the process is actually gone (`kill -0`) before rebuilding
  anything downstream of it.

### TD20 · the published decision altitude comes from the approach plates, NOT the CIFP

- **The CIFP publishes no approach minima at all**, so a per-runway decision altitude cannot be
  read out of `FAACIFP18`. Checked three ways on the 2026-08-06 cycle: (1) the *CIFP Readme*'s
  own list of record types is airports/heliports, runways, VHF and NDB navaids, terminal navaids,
  localizer/glideslope, Path Points, MSA, en-route and terminal waypoints, SIDs, STARs,
  approaches, airways, Class B/C/D and special-use airspace, Grid MORA — there is no minima
  record; (2) the whole file's approach section (`P`/`F`) has exactly **one** kind of
  continuation record, application type `W` (Level of Service, 6,741 of them), and it carries
  service *letters* (whether LNAV / LP / LNAV-VNAV / LPV is authorised), never a value; (3) the
  Path Point record (TD9) carries geometry only — LTP/FTP, glidepath, course width, TCH and the
  two orthometric heights. The DA is a charting product, published on the plate.
- **The plates are already on disk**: `data/RNAV_CHARTS/<ICAO>/*.PDF`, the FAA d-TPP RNAV (GPS)
  and RNAV (RNP) approaches for all five airports. Only the **RNAV (GPS)** plates are read — the
  RNAV (RNP) Z plates publish RNP AR minima, a different service the fleet is not flying.
- **`extract_approach_minima.py` reads them once and writes the numbers into
  `config/runway_thresholds.json`** (`published_minima` on every threshold, schema
  `runway-thresholds-v3`). `load_airport` then reads config; nothing re-parses a PDF at harvest
  or training time. Re-run it only when the chart cycle changes.
- **What is stored is what the plate prints**: the decision altitude in feet MSL, the height
  above touchdown it is published against (`decision_height_above_touchdown_ft` — named in full
  because it is NOT the height above the threshold), and that runway's printed TDZE, plus the
  plate's file name, procedure title, amendment and 28-day validity band (`chart_effective`, so a
  verdict can name the plate it was graded against). No derived height is stored, because the
  height above the *landing threshold* depends on which threshold elevation the reader uses, and
  for an LPV runway `load_airport` replaces the configured elevation with the Path Point's LTP
  (TD9). `Runway` derives it at read time:
  `decision_height_above_threshold_m = DA(MSL) − elevation("msl")`, and RAISES on a runway with
  no vertically guided minima. Across the fleet the landing threshold is **0.0–23.9 ft below the
  touchdown zone** (largest KSTL 29), so reaching for the wrong one is a 7.3 m error.
- **Every parse is checked against the same plate**: `DA − height above touchdown` must equal the
  TDZE printed for THAT runway (a sidestep sheet prints two — KSJC 30L's prints `TDZE 30L 57` and
  `TDZE 30R 55` — and the labelled one is used; a sheet that labels neither must print exactly
  one, or the parse raises rather than choosing). All 25 published rows pass, including the one
  below sea level (KMSY 20, TDZE −1 ft). **This check cannot catch a wrong ROW**, because every
  row on a plate satisfies it (KRDU 05L: 598−214, 748−364 and 840−456 all give 384) — what
  catches a wrong row is that the parse refuses to fall through: a table that names LPV without
  an LPV row parsing RAISES instead of taking the LNAV/VNAV row 150 ft above it, and a plate
  claimed to have no vertical guidance must show an LNAV MDA row to prove it. Six sheets, one
  per shape, are pinned against figures read by eye.
- **Three statuses, no silence.** `lpv` where the plate publishes LPV (23 thresholds), `lnav_vnav`
  where it publishes Baro-VNAV minima but no LPV (**KRDU 32 = 820 ft / 391 ft HAT**, **KSMF 35R =
  311 ft / 287 ft**, the same two runways TD9 already singles out), and `none` where the airport
  publishes no instrument approach to that end at all (**KRDU 14**, which has no approach in the
  CIFP either). A runway whose plate publishes only an LNAV **MDA** is not vertically guided, so
  it gets no decision altitude — its missed approach point is a fix, not a height.
- **The fleet's published DAs, height above touchdown** (feet): KRDU 05L 214, 05R 200, 23L 200,
  23R 200, **32 391**; KSJC 12L 250, 12R 200, 30L 200, 30R 200; KSTL 06 359, 11 250, 12L 410,
  12R 200, 24 200, 29 363, 30L 200, 30R 200; KSMF 17L 200, 17R 200, 35L 200, **35R 287**;
  KMSY 02 396, 11 200, 20 250, 29 200. The spread is **200–410 ft above touchdown**.
- **Above the LANDING THRESHOLD, which is the frame a trajectory is judged in, the same set runs
  200.0–423.4 ft** (highest: KSTL 12L; lowest: KSJC 30L/30R at 199.9 ft). That whole range sits
  inside altitude word 0 of the instruction vocabulary, which spans the threshold ±500 ft — so no
  decision altitude in this fleet can be told from the altitude WORD, and a go-around's timing has
  to be judged on the real height. That is the measurement behind the two-tier plan's D75.
- **`build_runway_config.py` can no longer rebuild this file** and could not before this change
  either: the generator writes `name`/`length_ft`/`surface`/`thresholds` only, while the config on
  disk also carries `width_ft`, `runway_width_effective_date` and now `published_minima`. Running
  it would drop all three. Recorded in `docs/code-health-followups.md`; until it is fixed, edit
  the JSON through `extract_approach_minima.py`, never by regenerating.
