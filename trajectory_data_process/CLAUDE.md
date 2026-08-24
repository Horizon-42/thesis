# trajectory_data_process — acquisition, harvest, datasets

`harvest/` is the download pipeline: fetch → reconstruct → assign (one runway per track) →
`tracks/` + `approach/`. CLI: `python -m trajectory_data_process.harvest --airport KRDU`.
Geometry lives in `final_approach/` (see its `CLAUDE.md`); this tree is fetch, reconstruct,
store, roster.

## Contracts

- **Harvest is manifest-only.** `tracks/manifest.json` rosters all four measured outcomes;
  `arrivals/manifest.json` rosters only assigned, CIFP-targeted, final-entry-cropped model
  inputs and records every exclusion. Scenario, optimizer-reference, and TS loaders follow that
  roster and never glob, so an orphan/rejected/stale JSON cannot enter a model split.
- **The threshold event carries `crossing_ground_speed_m_s` (additive within
  `runway-threshold-event-v1`, 2026-08-24), and it is GROUND speed, audit-only.** Direct
  events interpolate the bracketing samples' `reported_ground_speed_m_s` at the position's
  own fraction; censored events OLS-extrapolate speed vs along-track over the SAME kept
  samples as the position fit (`final_approach.fit_line`, the shared estimator), with the
  fit's own sample/span standard — an unfittable speed omits the field and
  `diagnostics.ground_speed_fit.omitted_reason` says why. OPTIONAL on read: every event
  stored before this date lacks it, and only a NEW harvest or `--reclassify-existing`
  populates it (`--evaluate-only` re-rosters stored events unchanged). Wind is unmodelled,
  so it must never feed evaluation's stall-anchored airspeed gate
  (`evaluation/docs/THRESHOLD_SPEED_GATE.md`); observed subjects stay speed-ungraded.
- **Observed evaluation records carry a `crossing_span` (2026-08-24)** —
  `harvest/observed.py` marks the event's direct bracket or appends the one inferred
  crossing row (built by `flight_scenarios.crossing_span`), so evaluation grades the
  STATES through one shared interpolation instead of re-reading the event. Target
  kinematics and `final_time_s` stay anchored to the last MEASURED row. Records in
  `approach/records/` from before this date fail evaluation loudly ("no
  crossing_span") — rebuild with `--evaluate-only`.
- **`unassignable` (the receiver lost it) and `not_established` (the approach was not
  stabilised) must stay distinct outcomes.** Conflating them charges a reception gap to the
  pilot. Both are counted and reported; neither is ever dropped (established rate is 21–54 % on
  real data).
- **Archived history rows are in FEET; `fetch_history_dataframe` output is in METRES.**
  `_altitudes_to_metres` converts on the live path only, so rows read back from
  `outputs/history_rows/**.jsonl` are raw feet. Guessing wrong scales every altitude by 3.28 and
  does not crash — it turns a 3.06° approach into a 9.94° one (observed).
  `harvest.reconstruct_tracks` therefore takes a **required** `altitude_units` argument with no
  default and no sniffing.
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
- **Track reconstruction: a landing ends at a sustained ON-GROUND run, and a spatial crop must
  keep only the final contiguous run.** An aircraft that lands keeps transmitting from the gate,
  so state vectors are continuous across a turnaround and the >900 s gap rule never fires; the
  radius filter then removed the middle and glued two passes together — KSTL AAL2717 carried a
  **6598 s** hole with two stray samples after it, which put the flight's `landing_time_utc`
  (part of its identity) on the WRONG pass. Also: `_complete()` must require **both** dep and arr
  airports, not either — `(None,'KRDU') → ('KATL','KRDU')` is one flight with the origin merely
  resolved, and the old rule cut the approach in half.

## Runway thresholds & TCH (static data feeding the harvest)

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
- **Per-runway TCH is published in the CIFP and is NOT 15 m.** ARINC 424 section P / subsection P
  "Path Point" records (`data/CIFP/<cycle>/FAACIFP18`) carry glidepath angle, course width and
  threshold crossing height per LPV approach; `harvest/cifp.py` decodes them. Every runway in the
  fleet publishes **15.27–18.11 m**, so the old flat 15 m assumption put a systematic 1.5–2.5 m
  bias into a 9.15 m window. Using the published value moved the measured KSMF crossing from
  +2.74 m to **+0.61 m** — real traffic crosses where the plate says, to within half a metre,
  which is the best end-to-end check that the datum, the fit and the TCH source all agree.
  The column decode is pinned by a coincidence-proof cross-check: **4795 of 4900 records decode a
  course width of exactly 106.75 m**, independently the LPV semiwidth in `evaluation/thresholds.py`.
  A runway with no Path Point record has **no LPV procedure** (KRDU 14/32) — its TCH is `None`,
  never defaulted, because it cannot be judged against LPV gates at all.

## Altitude outlier repair

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
- Criterion: deviation from the median of the ±2-sample window exceeding BOTH 100 m AND
  `25 m/s × min(adjacent gap)`. Both halves are load-bearing — 100 m alone repairs 10 real
  descents that stepped 107–160 m across 9–14 s reception gaps into lies, and a **chord/jump test
  attributes one bad sample to three** (the outlier's two neighbours fail too; measured 363
  runs-of-3 where the truth was 363 isolated samples).
- Measured incidence: **561 samples in 451 of 44 622 assigned tracks (0.0027 %)**, 421 inside a
  model arrival slice. Only `samples[i][3]` changes; dropping a row would silently renumber
  `landing_sample_index`, the arrival slice bounds, the event's `source_sample_range`, and
  `reported_ground_speeds_m_s`.
- Audit/republish with `python -m trajectory_data_process.altitude_outliers [--rerender-czml]`
  (reads `tracks/`, writes only `public/data`). **Not covered:** stored
  `observed_threshold_event`s were fitted from raw samples during assignment — 17 outliers land
  inside one, and only `--reclassify-existing` re-derives those.

## Constants

- Altitude outlier filter (`harvest/altitude_filter.py`, single source; `AltitudePolicy`):
  `half_window = 2`, `min_deviation_m = 100.0`, `max_vertical_rate_m_s = 25.0`. Repair = linear
  interpolation in time between the nearest retained samples; an outlier at a track edge HOLDS
  the nearest retained altitude (labelled `held` vs `interpolated`). The 100 m floor is 13× the
  25 ft quantum, 3.3× the 100 ft quantum, and 2× the largest residual genuine flight produces
  (over 20.85 M samples: 20 847 051 below 25 m, 3 625 in [25, 50), 189 above 50 m). Counts are
  reported in `arrivals/manifest.json` → `altitude_filter`, `approach/summary.json` →
  `altitude_filter`, and `RenderedObserved.altitude_outliers`.
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
- Arrival manifest schema: **`harvest-arrivals-v5-takeoff-excluded`** (v4 excluded no
  takeoffs). Loaders compare exactly, so a v4 manifest on disk fails loudly; rebuild with
  `python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only`, which
  re-rosters from stored `tracks/` without downloading or reassigning anything.
- **Rebuilding `arrivals/` DELETES `lateral_pass_eligibility.json`, which nothing rebuilds
  for you.** `arrivals._clear()` unlinks every `*.json` under the directory before writing
  the new manifest, and the lateral-pass roster lives there but is owned by
  `ts_transformer/lateral_eligibility.py`. So any `--evaluate-only` or
  `--reclassify-existing` silently removes a file every TS train/predict requires, and the
  failure surfaces later as `FileNotFoundError` from whichever run touches it next — on
  2026-08-21 that was mid-campaign, after one arm had already trained. Rebuild it right
  after the harvest, from the regenerated approach report:
  `lateral_eligibility.ensure_lateral_pass_roster(<arrivals>/manifest.json)`.
- **`--reclassify-existing` is NOT the rebuild command** — it re-derives runway assignment
  and is only for a changed runway-data cycle or assignment method. For a schema bump or an
  evaluation-policy change use `--evaluate-only`; reaching for the heavier one risks a
  different roster for no reason.
- **A harvest killed with SIGTERM can still finish its write.** On 2026-08-21 a
  `--reclassify-existing` run killed mid-flight completed several minutes later, cleared the
  directory again, and overwrote a manifest a training job was already reading — the arm had
  to be discarded. Confirm the process is actually gone (`kill -0`) before rebuilding
  anything downstream of it.
