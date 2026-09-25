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

- **The record's HAE → MSL step is `flight_scenarios.datum.flight_to_msl` (2026-09-25)**, fed the
  runway as a `runway_target` (`arrivals.runway_target`, the arrivals' own); it used to subtract the
  same CIFP offset by hand. Same numbers (1,500 real records compared byte for byte), and a track not
  tagged HAE is now refused instead of converted.

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

**Correction (2026-09-23):** the non-LPV runways' threshold POSITION no longer comes from the
configured OurAirports end, and every runway's course no longer from the configured heading —
both come from the CIFP Runway records now (TD21). `threshold_frame_snapshot` changed (schema
`threshold-physical-frame-v2`), so every stored event is stale and is re-derived by the
reclassification a `--merge-source` runs.

### TD21 · every threshold and course comes from the CIFP (Runway records)

- **Threshold position.** An LPV runway's threshold is its Path Point LTP (unchanged); every
  other runway's is its **CIFP Runway record** (section P / subsection G) LTP —
  `cifp.read_runway_records`, ARINC 424-23 §4.1.10.1 (latitude cols 33–41, longitude 42–51,
  landing threshold elevation 67–71 in whole feet). §5.36/5.37 Note 5: "The Runway latitude and
  longitude shall represent the runway's Landing Threshold"; §5.57 repeats it. The configured
  OurAirports end put **KSMF 35R 39.4 m cross-track** off the published LTP; after v6 admitted
  35R (TD9) every one of its arrivals read as a ~41 m lateral miss (383/383 of the 2026-08-22..09-22
  download, 259/259 in the v5 root's observed report), so the lateral roster dropped the whole
  runway from ts and the optimizer targeted a point 40 m off. KRDU 32 was 0.4 m off, KRDU 14 3.9 m.
  `position_source == "faa_cifp_runway_record"`; the non-LPV MSL elevation is the record's
  landing threshold elevation (equal to the configured value on all three non-LPV runways),
  N still from the nearest Path Point. The record also publishes the LTP's ellipsoidal height
  (cols 61–66, 0.1 m), but N = HAE − MSL taken from it inherits the MSL elevation's whole-foot
  rounding (±0.15 m), while the nearest Path Point's N is exact at a point 1–2 km away where the
  geoid differs by centimetres — so N stays the Path Point's (KRDU 32: 0.14 m apart).
- **Course.** `Runway.course_deg` is the initial bearing from the runway end's Runway-record LTP
  to the opposite end's (`course_source == "faa_cifp_runway_records"`). Both are on the
  centreline, displaced or not, so this is the true centreline to ~0.01° over a 2–3 km runway.
  OurAirports publishes whole degrees: up to **0.45° off** this fleet (KMSY 11 105.55 not 106,
  KSMF 35R 0.75 not 1, KSTL 30R 302.32 not 302) — ~40 m of cross-track 5 km out, ~80 m at 10 km.
  The threshold crossing itself was never biased (the final-segment fit has a slope term); what
  moved is everything measured ALONG the final in the runway frame (path-shape deviation, the
  established cone, cross-track readouts, the headwind component). The configured heading stays
  as a cross-check: a CIFP centreline more than 1° from it raises (the two files would describe
  different runway ends).
- **The decode is pinned** like the other two CIFP readers: on every runway with both a Runway
  record and a Path Point, position within 1 m and elevation within 1 ft, on ≥ 75 % of them.
  File-wide over CIFP 2608's 4,670 LPV runway ends: position p50 0.12 m / p95 0.48 m, elevation
  p50 0.08 m / p95 0.19 m, 95.8 % meet both (the rest are mostly fictitious-threshold Path Points).
  A column misread by one digit misses by kilometres and fails the pin, not a parse.

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
- **`--rerender-czml` renders the default policy only (2026-09-25).** Every reader of a track view
  (`store.read_track_view`, the arrivals and the observed CZML) applies `DEFAULT_POLICY`, so a CZML
  rendered under an audit's trial policy (`--half-window`, `--min-deviation-m`,
  `--max-vertical-rate-m-s`) would publish repairs no model input saw. The CLI refuses the
  combination by name: audit with the trial policy, rerender without it.

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

**Re-measured on the v7 roster (2026-09-25): the band is no longer empty.** Over the 72,574
rostered arrivals and the 114 `takeoff_in_segment` exclusions of the live root (read-only, the
exclusions' first samples recomputed with `arrival_segment`): the exclusions reach **93.1 m**
(`ZEUS14_32_ae7488_20260521T191843Z`, 20 km out; then 91.0 m, `N5264V`, 19 km out) and the rostered
arrivals start from **114.9 m** (`N5306U_30R_a6b3aa_20260826T024539Z`, 9.7 km out; then 116.0 m,
`ZEUS11_32_…`, 20 km; 138.8 m, `ZEUS21_32_…`, 19 km); 3 rostered arrivals start in [100, 150) m, 9
in [150, 200) m. The flights on both sides of 100 m are helicopters (ZEUS\*, KRDU 32 — the new
download brought them) and light aircraft flying low 10–20 km out, not takeoffs: there the test
separates low cruising from high, and the exclusion reason's "begins on the ground at a field" is
wrong for them. The constant is unchanged (changing it moves the roster, hence every ts split); the
comment at `arrival_segment.GROUND_START_AGL_M` states both measurements.

### TD16 · arrival manifest schema

- Arrival manifest schema: **`harvest-arrivals-v5-takeoff-excluded`** (v4 excluded no
  takeoffs). Loaders compare exactly, so a v4 manifest on disk fails loudly; rebuild with
  `python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only`, which
  re-rosters from stored `tracks/` without downloading or reassigning anything.

**Correction (2026-09-16):** the WRITER is now `harvest-arrivals-v6-published-vertical-path` (`harvest/arrivals.py` `SCHEMA_VERSION`) and loaders accept v5 AND v6 (`READABLE_SCHEMA_VERSIONS`), so "loaders compare exactly" now means against that pair; a v4 manifest still fails loudly. All five manifests on disk were still v5 when read on 2026-09-16. A rebuild writes v6 and grows the rosters (TD9).

**Correction (2026-09-23):** the writer is **`harvest-arrivals-v7-measured-crossing-in-slice`**
(TD22) and `READABLE_SCHEMA_VERSIONS` is gone: the loader reads the CURRENT schema only, or a
manifest whose bytes a registered frozen generation recorded (TD23) — read as written, nothing
converted. The version string says which rule WROTE a roster; the gate exists so that new
training never picks up a roster written under an older rule, and a frozen roster is identified
by its content instead (the identity a checkpoint pins).

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

**Correction (2026-09-23):** the writer now builds the whole roster first and replaces
`arrivals/` only then (manifest written to `manifest.json.tmp` and renamed), so a rebuild that
raises part-way leaves the previous manifest AND its lateral roster intact. A successful rebuild
still deletes the lateral roster (it is bound to the manifest bytes it was joined against).
`observed.write_observed_records` likewise builds `records/` in a staging directory and swaps it
in with `summary.json` only on success.

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
  200.0–423.4 ft** (highest: KSTL 12L; lowest: KSJC 30L/30R at 199.9 ft).
- **`build_runway_config.py` can no longer rebuild this file** and could not before this change
  either: the generator writes `name`/`length_ft`/`surface`/`thresholds` only, while the config on
  disk also carries `width_ft`, `runway_width_effective_date` and now `published_minima`. Running
  it would drop all three. Recorded in `docs/code-health-followups.md`; until it is fixed, edit
  the JSON through `extract_approach_minima.py`, never by regenerating.

## Merging, slices and generations (2026-09-23)

### TD22 · the arrival slice contains its measured crossing (v7); `entry_time_utc` is exact

- `landing_sample_index` is whichever sample of the measured bracket is nearer the threshold
  (`classify._landing_sample_index`); it defines `landing_time_utc` and so the `flight_key`, and
  it does NOT move. Before v7 the arrival slice ENDED there, so about half the bracketed arrivals
  (KSMF new download: 2,396 left / 2,228 right; KSJC v5 sample 729 / 770) stopped one sample short
  of their own measured crossing, and `ts dataset._observed_threshold_crossing` (which needs both
  bracket samples) supervised those on a fitted tail instead — two supervision contracts for one
  kind of measurement, decided by which sample happened to be closer.
- v7 (`harvest-arrivals-v7-measured-crossing-in-slice`): when the event is a direct bracket
  (`method == direct_linear_bracket`), `last_sample_index` is the bracket's post-crossing sample
  (`arrivals._slice_end_index`); a landing sample outside its own bracket raises. Censored events
  keep the landing sample. For bracketed flights the slice now ends one sample (typically
  ~70 m) PAST the threshold — the ts loader cuts the grid before the crossing, the optimizer's
  observed reference carries the extra row.
- `entry_time_utc` is the first kept sample's own time — the track's `start_time_utc` plus that
  sample's offset, millisecond ISO (`…T10:03:07.123Z`). It was the whole-second landing time
  minus the segment duration: 0–2 s early, and it would have moved with the slice end.
  `arrival_segment.truncate_flights` therefore requires `start_time_utc` on every flight.

### TD23 · frozen harvest generations; a checkpoint finds its data by digest

- A ts checkpoint fingerprints the exact bytes it trained on (each airport's arrival-manifest
  SHA-256 and each source track's). Rebuilding or merging the live root moves those bytes, so a
  checkpoint can only replay against the GENERATION it trained on. Freezing keeps it: the root
  is MOVED aside (never edited), and `python -m trajectory_data_process.harvest.generations
  freeze <root> --reason …` writes `FROZEN.json` (schema `harvest-frozen-generation-v1`) with every
  airport's arrival- and tracks-manifest SHA-256. Registered in code:
  `generations.FROZEN_GENERATIONS` (a reviewed change, never a directory scan). A registered root
  that is absent is skipped (a clone may not hold it); one present without its marker raises.
- `arrivals.load_arrival_flights` reads a non-current schema only when its bytes are a frozen
  generation's (`generations.is_frozen_arrival_manifest`). ts replay runners find a checkpoint's
  manifests with `repo_layout.checkpoint_arrival_manifest(s)`: among the live root and the frozen
  roots, the file whose SHA-256 equals the recorded `arrival_manifest_sha256`; none raises by
  airport and digest. New training reads the live root (`repo_layout.HARVEST_ROOT`, now the one
  definition; four runners had their own copy). A frozen roster passed EXPLICITLY
  (`train --data <frozen manifest>`) is read too — deliberately: retraining an old recipe on
  its own generation is a legitimate, fully fingerprinted choice; no default path reaches it.
- **A frozen root is never written**: every harvest mode refuses an `--output` holding
  `FROZEN.json`, and the root is made read-only (`chmod -R a-w`) when frozen.
  `freeze` records only airport directories (`[A-Z]{4}`), so a killed reclassification's hidden
  `.KSJC-reclassify-*` leftover (170 MB in the v5 root) is carried along but not recorded.
- Hard links: a merge stages the destination's records as hard links and reclassification writes
  NEW files, so nothing is written through a link — a frozen root that shares inodes with the
  live one is not modified by a merge (tested: source bytes identical after a merge).

### TD24 · merging a new download (`--merge-source`)

- `python -m trajectory_data_process.harvest --airport <ICAO> --merge-source <root> --jobs N
  [--no-czml --no-publish]` validates every source roster and record, hard-links them into a
  staged tree, reclassifies EVERY track under current code and runway data, swaps `tracks/`, then
  DELETES `arrivals/` and `approach/` and rebuilds both in the same run (and the observed CZML /
  frontend publication unless told not to). Identical flight_keys or record paths across sources
  refuse the whole transaction before anything changes. `--jobs` reaches reclassification
  (it was single-process before 2026-09-23; output is identical at any value).
- Each source's exclusion audits (`store.integrity_audits`: a freshness rebuild's
  `source_integrity` — 2,458 excluded tracks over the five v5 roots — or an earlier merge's,
  flattened) are carried into its `provenance.merge.sources[i]` as `source_integrity_audits`
  with `sources_without_integrity_audit` (a direct download has none); the merged manifest has
  no single denominator and no top-level `source_integrity`, and `observed.source_event_availability`
  reads the flattened audits and reports the unaudited count. Before 2026-09-23 the merge
  dropped the audit and the availability denominator silently lost those candidates.
- **A plain download refuses a merged or rebuilt root** (`__main__._refuse_download_over_derived_tracks`,
  provenance key `merge` / `freshness_rebuild`): `store.write_tracks` clears `tracks/` in place,
  and such a root never reads as a completed download (no `radius_km`), so
  `python -m trajectory_data_process.harvest --airport KRDU` used to replace months of merged
  harvest with one 30-day window. Download into a new `--output` and merge it.
- `download_landings.py` imports the harvest CLI's defaults; its own copy had drifted to CIFP
  260319, and the 2026-08-22..09-22 download (`outputs/new_data_9_22`) was stamped with that cycle
  (the merge re-derives every event under 260806).

### TD25 · staging leftovers are listed by every run and removed only on request (2026-09-25)

- Every rewrite stages beside the real tree and swaps it in: the observed records
  (`approach/.records-staging-*`, the previous records moved aside as `approach/.records-previous-*`),
  a reclassification (`<root>/.<ICAO>-reclassify-*`) and a merge (`<root>/.<ICAO>-merge-*`). A SIGKILL
  mid-write leaves them (the v5 root held a 170 MB `.KSJC-reclassify-*` from 2026-08-24). Readers
  follow the rosters and never see them; they cost disk only.
- The prefixes live in `harvest/staging.py` (the writers take them from there). Nothing removes a
  leftover unasked — there is no harvest lock, so a sweep could delete another run's live staging:
  every harvest run ends by listing the airport's leftovers, and
  `python -m trajectory_data_process.harvest --airport <ICAO> --output <root> --remove-staging-leftovers`
  removes them and exits. Run it only when no other harvest writes that root.

### TD26 · the observed CZML's censored tail starts at the closest support sample (2026-09-25)

- A right-censored event's `extrapolation_distance_m` is measured from
  `diagnostics.closest_support_sample_index`, which can lie AFTER the fit's last sample
  (`source_sample_range[1]`). The drawn tail (`harvest/czml._extrapolated_waypoints`) used to start
  at the fit's last sample, so its crossing could come before the last supporting sample (1,871 of
  7,827 censored KRDU flights, 2026-09-23); on the live root the start moves in 12,958 of 25,064
  censored flights. It now starts at the support sample and is timed as the observed record's fitted
  crossing row (`flight_scenarios.crossing_span`): the extrapolation over the mean of the start's
  speed (finite difference to the previous sample) and the event's `crossing_ground_speed_m_s`, the
  start's speed again when the event fitted none. The silent 70 m/s fallback is gone: a start without
  a speed refuses the flight by name (none of the 25,064 does). Viewer only — nothing reads the tail's
  time.
