# AeroViz-4D Development Changelog

Dated log of significant changes, root causes, and decisions, referenced from `CLAUDE.md`. This file is deliberately NOT loaded into every session — read it when investigating history: why a design is the way it is, when/why a default changed, what a past bug or postmortem looked like, or which outputs a change made stale. Append new entries at the top (`### YYYY-MM-DD — title`); when a change produces a durable fact (gotcha, default, contract), also update the corresponding section in `CLAUDE.md`.

Entries verified via full test suites + tsc + vite build at the time; "verified in-browser" noted only where done. Merged same-day, same-topic entries.

### 2026-08-17 — Comparison references were the wrong window: full track vs model arrival slice

**Symptom.** In the comparison overlay the white observed reference did not start at the
same time or the same place as the `look-`/`pred-` group beside it. Measured on the KRDU
05L validation batch (471 groups): the group's first sample sat a median **5055 m** from
the reference (p75 5281 m, p95 47.1 km, max 54.9 km).

**Root cause — two time origins, only one of them reconciled.** Three timelines exist and
the publisher accounted for two:

1. a stored track's `samples[i][0]` is relative to first reception (`store.track_record`,
   absolute time in `start_time_utc`) — this is what `trajectories.czml` and the
   `/trajectories` backend served;
2. the model **arrival slice** is rebased at `harvest/arrivals.py` `load_arrival_flights`
   (`t0 = waypoints[0][0]`, `sample[0] - t0`), so every scenario, optimizer record and TS
   record has `t = 0` at the 25 km terminal-ring entry, and `t0` is discarded;
3. a prediction record rebases again to the anchor, recording the shift as
   `source.anchorTimeS`.

`build_scenario_comparison_czml` added ③ back (correctly — that was the 2026-07-20 anchor
fix) but nothing ever added ② back, so every group rendered `t0` early. Measured `t0` over
300 random KRDU arrivals: median **45.1 s**, p25 34.3, p75 55.6, p95 123.1, max 526.3 s.
The same seam applies to optimizer groups (`opt-`/`sim-` also start at ring entry); no
optimizer category happened to be published at the time, so it only showed on predictions.

Nothing downstream could detect it: both timelines start at `t = 0`, both name the right
flight, the schema is satisfied, and the drawn result reads as model error rather than a
publication bug. Proof that the two were the same measurement: fitting a per-flight pure
time shift dropped the lookback↔reference distance from a median 4900 m to **13.7 m** (2 s
resampling error).

**Fix — align the reference to the modeling window, at READ time.** The pre-entry segment
is not model input, not a supervision target and not evaluated; drawing it as the white
"truth" beside a forecast invites reading it as something the model failed to produce. So
the reference is now the arrival slice on the arrival origin, rather than the group being
pushed out onto the full track's origin.

- `aeroviz_backend/observed_trajectories.py` gained `window` ∈ `full` | `arrival`. `full`
  (default) is unchanged: the complete reconstructed track, rostered by
  `tracks/manifest.json`, for Observe/Baseline. `arrival` rosters from
  `arrivals/manifest.json` and builds flights through **`load_arrival_flights` itself** —
  the same loader the scenario/optimizer/training paths use — so there is no second
  implementation of the slice to drift from, and the source-hash check plus identity round
  trip come along for free.
- `tracks/` is untouched and no artifact is written: the slice is taken at read time, the
  same rule the altitude-outlier repair follows.
- Response schema bumped to `observed-trajectories-v2` with `trackWindow` echoed. The bump
  is load-bearing: a v1 backend ignores an unknown `window` argument and answers a
  comparison-reference request with full tracks, reproducing the bug silently. The
  frontend refuses anything but `arrival` for the comparison reference.
- `build_scenario_comparison_czml._require_reference_aligned` pins the embedded-reference
  path (`include_reference_entities=True`, currently unused in publishing) at 50 m — inside
  the gap between resampling noise (~14 m) and a wrong window (≥5 km).

**Verified end-to-end on real data**: over all 471 KRDU 05L groups the group-start-to-
reference distance is now **0.0 m** for every group (bit-identical samples), against a
median 5055 m before.

**No artifact is stale.** Published comparison CZMLs contain only `look-`/`pred-` (the
publisher passes `include_reference_entities=False`) and the reference is served live, so
restarting the backend is the whole deployment — no re-publish, no re-predict, no
re-optimize.

### 2026-08-17 — ADS-B altitude outliers filtered in the view, not in the tracks

**Symptom.** Observed trajectories rendered with needle-shaped vertical peaks: single
samples reporting an altitude nowhere near their neighbours. Measured extremes across the
five harvested airports are 20 147 m between neighbours at 724 m and 35 189 m at 556 m.

**What was rejected.** A stop-gap `fix_altitude_spikes.py` edited `tracks/*.json` in place.
That breaks three contracts at once — `arrivals/manifest.json` pins every source track by
SHA-256 (the loader refuses a changed file), `--reclassify-existing` re-derives assignment
from those exact samples, and `source_integrity.retained_rows` counts them. It also missed
the point: `tracks/` is the sensor reconstruction, and a repair is a property of the view.

**What shipped.** `trajectory_data_process/harvest/altitude_filter.py`, applied where a
stored track becomes a derived view and nowhere else:

- `store.read_track_view` → observed CZML (`czml.observed_czml_flights`, which also feeds
  the backend trajectory sampler) and evaluation records (`observed.write_observed_records`);
- `arrivals.write_arrival_records` / `load_arrival_flights` → all model input (ts training,
  `flight_scenarios`, `batch_benchmark`). Both hash the SOURCE bytes first and filter after,
  so the roster stays a statement about what the receiver recorded.

`store.iter_records` deliberately stays raw — reconstruction and reclassification must see
what was stored.

**Detection.** Deviation from the median of the ±2-sample window exceeding BOTH 100 m AND
`25 m/s × min(adjacent gap)`. Both halves earned their place on the data:

- a chord/jump test (the stop-gap's approach) attributes one bad sample to three, because
  the outlier's two neighbours have a chord running through it — 363 runs of exactly three
  where the truth was 363 isolated samples;
- the 100 m floor sits at 2× the largest residual genuine flight produces. Over 20 851 436
  assigned samples the residual is < 25 m for 20 847 051, 3 625 fall in [25, 50) (the 25 ft
  and 100 ft reporting lattices plus real motion), and only 189 exceed 50 m;
- the rate bound spares 10 real descents that stepped 107–160 m across 9–14 s reception
  gaps, which a bare deviation threshold "repairs" into a lie.

Incidence: **561 samples in 451 of 44 622 assigned tracks (0.0027 %)**; 421 sat inside a
model arrival slice. 479 isolated, longest run six.

**Repair replaces the altitude and never drops the sample.** `landing_sample_index`, the
arrival slice bounds, the threshold event's `source_sample_range` and the
`reported_ground_speeds_m_s` parallel array all index that array; deleting a row silently
renumbers every one of them. Replacement is a linear interpolation in time between the
nearest retained samples; at a track edge it holds the nearest retained altitude
(`held` vs `interpolated`, both labelled in the report).

**Stated, never silent.** `arrivals/manifest.json` and `approach/summary.json` each carry an
`altitude_filter` block (policy + repaired counts), arrival records carry a per-flight
`altitude_outliers`, and `RenderedObserved` carries the render's totals.

**Tooling.** `fix_altitude_spikes.py` deleted; `python -m trajectory_data_process.altitude_outliers`
replaces it as a read-only audit (`--report-json` gives the full per-track trail) plus
`--rerender-czml`, which republishes `public/data/<ICAO>/trajectories.czml` through the
pipeline's own renderer.

**Artifacts.** All five `trajectories.czml` republished. Batch comparison CZMLs resolve
their observed reference by entity id inside that canonical file, so they follow without a
rebuild. Training data needs no rebuild either — `load_arrival_flights` filters on the way
out — so the next dataset build is already clean; `--evaluate-only` is what refreshes the
roster counts and the evaluation records.

**Known gap.** Stored `observed_threshold_event`s were fitted from raw samples during
assignment, before this filter existed. 17 outliers (KRDU 15, KSTL 2) land inside an
event's source range; the audit lists them and `--reclassify-existing` is what re-derives
them.

### 2026-08-12 — Fail-closed pipeline cleaner

- `clean_pipeline_data.py` now requires an explicit airport scope and constructs its
  deletion plan from producer-owned artifact names instead of recursively treating
  `4dTrajectory/outputs` as disposable.
- Downloaded tracks, checkpoints/history, `test_release.json`, formal experiments,
  pooled roots, final-test and ambiguous predictions, parked/manual/unknown outputs,
  tracked/static data, archives, and mixed experiment comparison publications are
  protected.
- Only standalone predictions whose readable metadata explicitly says `split: "val"`
  are eligible. Comparison cleanup requires a readable registry that accounts for all
  content. The canonical observed filename is matched exactly, not by prefix.
- Destructive execution validates the complete plan and stages every selected file on
  the same filesystem; a staging failure rolls the move set back. Safety tests cover the
  allowlist, airport isolation, protected research state, mixed comparison output, exact
  CZML ownership, required scope, dry-run behavior, and rollback.

### 2026-07-23 — Fitted threshold kinematics; preparation/optimization runner split

- Fitted-ADS-B targets now derive `V/psi/gamma` from the same established final-approach
  fit as their threshold position. The along-track rate is fitted over that established
  segment only; the spatial tangent supplies heading and glide angle, so rollout
  or parked samples can no longer create a threshold target with `V=0`. The constant-rate
  helper is the single replacement seam for a future deceleration model.
- The former combined runner was deleted. `prepare_scenario_inputs.py` rebuilds
  arrivals/observed products and writes the two distinct scenario datasets;
  `run_scenario_optimization.py` consumes those datasets and owns optimization,
  evaluation, and comparison-CZML publication. Run preparation first, then optimization.
- `clean_pipeline_data.py` now removes preparation-derived `arrivals/` and `approach/`
  by default while preserving downloaded `tracks/`; `--include-downloads` only expands
  the deletion boundary to measured source tracks.
- Review hardening made datum provenance explicit on fitted results, rejects invalid
  `states_ref` ranges and duplicate source identities, and verifies reference identity
  plus SHA-256 before cache/batch reuse. `--skip-optimize` now validates the complete
  summary/eval/states/reference roster rather than treating `summary.json` as a marker.
- Comparison publication now writes immutable generation-suffixed CZML/report artifacts
  and atomically commits their index last; failures preserve the previous generation.
  The frontend follows the report named by that index, strictly rejects legacy observed
  and comparison manifests, never falls back to embedded references or fixed-name
  reports, and restores canonical entity styles on comparison exit.

### 2026-07-21 — final_approach + evaluation review fixes: observed-aware reporting surfaces, one deviation definition

Code review of `final_approach/` + `evaluation/` (no correctness bugs in the geometry or the
fit statistics; all findings were reporting/duplication/doc-drift). Fixes:

- **The human-facing surfaces caught up with the observed-subject work.** The measurement side
  (`arrival.py`, the report's `observed` block) was done, but `evaluation/visualize.py` and the
  `python -m evaluation` console summary still rendered pre-subject reports: a solve-rate card/
  line (1.0 by construction on observed data — the exact number the subject dispatch exists to
  stop reporting) and no established rate or marginal count anywhere. Both now suppress the
  solve rate for pure observed batches and report `established N/M` + marginal. The deviation
  charts also plotted `solvedRows`, which for observed batches includes not-established rows
  with NO deviation fields — `Math.max(undefined, 0.01)` → NaN → silent bar gaps; charts now
  draw `measuredRows` only. Verdict table gains established/marginal columns + ±95 % bounds on
  the deviations when observed rows exist; not-established rows are styled grey ("no arrival to
  judge"), not red ("judged and failed").
- **`EstablishedCriteria` became reachable**: `evaluate_batch(..., criteria=)` and CLI flags
  (`--fit-window-m/--max-cross-track-m/--glidepath-range-deg/--max-vertical-rms-m`), defined
  ONCE in the new `evaluation/cli.py` and shared by `__main__` and `visualize` so the JSON and
  HTML reports cannot be produced with silently different knobs (the gate flags were previously
  duplicated between the two entry points).
- **One final-state deviation definition.** `metrics.final_state_deviation` + its
  `FinalStateDeviation` dataclass duplicated `arrival._final_state` line for line (no external
  consumers, verified). The single definition now lives in `arrival.final_state_deviation`
  returning `ArrivalDeviation`; `FinalStateDeviation` deleted.
- **`_is_marginal` lateral fold fixed**: `abs(lateral − margin)` folded a signed CI containing
  the centreline past 0, misreading "CI straddles the gate" as a solid verdict whenever
  1.96σ > gate + offset. Lower bound is now `max(0, lateral − margin)`. Unreachable at real
  σ ≈ 1–2 m (needs σ > ~55 m) — fixed for correctness, with a regression test.
- **Dead code removed**: `frame.track_course_deg` / `heading_difference_deg` had no consumer
  anywhere (the promised harvest heading pre-filter never materialized — the arg-min design made
  it unnecessary). `fit_final_segment` now validates `min_samples >= 3` / `min_span_m > 0` at
  the boundary (previously a ZeroDivisionError deep in `_fit_line`). `visualize.RESAMPLE_N`
  deduped into `reference.N_RESAMPLE`.
- Docs: `evaluation` README/`__init__` (which still described the pre-subject package, exported
  nothing from `arrival`) updated; `Assignment.scores` docstring no longer claims rejections
  carry scores; stale CLAUDE.md open item ("observed evaluation designed but NOT built") removed.

Suites: final_approach + evaluation 88 pass, trajectory_data_process 93 pass.

Found while trying to answer "what do the 8260.58D gates score REAL ADS-B arrivals at?" —
the observed baseline the optimizer and the learned predictor are implicitly measured
against, which had never been computed. The naive answer was 1.8 % pass (18/996 KRDU), i.e.
completed, safe airline landings graded as failures. Three independent bugs, all upstream of
`evaluation/`, which is unchanged by this work.

**The measurement that separated them.** Fitting each flight's OWN established final-approach
line (position + cross-track vs along-track, extrapolated to the threshold) instead of reading
`states[-1]`: the fitted glidepath came out 3.02–3.13° at all five airports — textbook — while
the vertical intercept was 20–30 m low. A fleet flying a perfect glidepath to a uniform 25 m
error is not a fleet error; it is a reference error. Lateral was already 3–10 m median, so the
two axes were telling opposite stories and had to be chased separately.

**Bug ① — observed altitude is ellipsoidal, targets are MSL.** OpenSky `geoaltitude` is height
above the WGS84 ellipsoid (HAE); runway thresholds, CIFP altitudes and the gates are MSL. The
gap is the geoid undulation N ≈ −25 to −33 m over the US. Confirmed independently: KRDU's
lowest observed sample is 99.1 m against a predicted field elevation + N = 132.59 − 33.53 =
99.06 m (4 cm). Fixed in a new `flight_scenarios/datum.py` (EGM96 via pyproj), applied at the
data→modeling seam.

*Not* in the harvest, deliberately: the harvest feeds two consumers with opposite requirements
— CZML/Cesium positions are documented as metres above the WGS84 ellipsoid
(`aeroviz-4d/src/types/czml.d.ts`) and are correct as recorded. Converting at the source would
have fixed modeling and broken the viewer by the same ~33 m. The harvest stays a faithful
record of what the sensor said; the datum choice is made on the way in.

The conversion reached THREE separate ingest paths (`load_observed_flights`, `build_scenario`,
`ts_transformer/dataset.py` — the last reads bare waypoints, so it cannot self-protect). It is
keyed on `altitude_source` and therefore idempotent, and unknown/missing sources raise rather
than defaulting. Seam tests pin all three.

PROJ trap worth knowing: without the EGM96 grid and with network off, pyproj silently returns
a "ballpark" no-op vertical transform — a correction that looks applied and does nothing.
`_geoid_transformer()` probes a known undulation and raises instead.

**Bug ② — `runway_thresholds.json` stored pavement ends, not landing thresholds.**
`build_runway_config.py` read `le_latitude_deg`/`le_elevation_ft` and ignored
`le_displaced_threshold_ft` entirely. KSJC 30L/30R are displaced 775 m; on a 3° glidepath that
is a 40.6 m altitude error, and it moved the OPTIMIZER TARGET, not just the gates. Six
thresholds moved (KSJC ×4, KSTL 12R 143 m, KMSY 29 93 m); the other 20 are unchanged. Fixed in
the generator, not the JSON. Schema bumped to `runway-thresholds-v2`; thresholds now carry
`displaced_threshold_m`.

This is why KSJC looked HEALTHIEST before the fix (+9.7 m vs everyone else's −25 m): its two
bugs had opposite signs and nearly cancelled (+40.6 − 32.0 = +8.6 predicted). Chasing the
"anomaly" is what found bug ②.

**Bug ③ — parallel runways captured the same landing twice.** `classify_landing_flights` was
called once per threshold with no cross-threshold arbitration, and `RUNWAY_THRESHOLD_RADIUS_M`
is 1000 m while parallel runways sit 250–400 m apart on an identical heading — so both the
geometry and heading tests accepted either one. Measured: 169 of KSJC 30L's 200 flights were
also in 30R's file; KSJC 12L∩12R 63; KSTL 30L∩30R 32. KRDU/KSMF/KMSY are unaffected (their
parallels exceed the capture radius). It surfaced downstream as an observed lateral error
whose MEDIAN was the parallel separation (KSTL 30L 397 m, KSJC 30R 234 m).

Fixed with a `sibling_thresholds` arbitration restricted to same-direction runways — the
opposite end of the same runway must be excluded, since a full rollout stops on top of it.
The discriminator is the median lateral offset from the extended centreline, NOT distance to
the threshold point: a first attempt using threshold distance failed to separate at all (kept
763 m vs dropped 791 m) because a displaced threshold sits 775 m past where ADS-B coverage
ends, so every track is equidistant from it. On the centreline metric the split is clean
(kept 17.5 m vs dropped 232.8 m at KSJC; 35.9 vs 382.6 at KSTL).

**Effect, end to end** (established-approach threshold crossing, records regenerated through
the real code path, not a reimplementation):

| airport | vertical median before → after | vertical gate before → after |
|---|---|---|
| KRDU | −29.2 → **+4.3 m** | 1 % → **44 %** |
| KMSY | −19.6 → **+5.7 m** | 0 % → **50 %** |
| KSTL | −26.9 → **+4.9 m** | 0 % → **31 %** |
| KSJC | +9.9 → **+0.3 m** | 24 % → **51 %** |
| KSMF | −27.7 → **+2.7 m** | 0 % → **65 %** |

All five now sit at +0.3 to +5.7 m — a small POSITIVE bias, which is operationally right
(crossing at or slightly above TCH is correct; low is dangerous). Lateral was already correct
and is unchanged at 3–10 m median.

**What this makes stale.** Everything derived from observed tracks: `flight_scenarios/outputs`,
all `4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}`, all
`public/data/airports/*/comparison`, and the `ts_*` training data + checkpoints. Bug ③
additionally requires re-harvesting KSJC and KSTL (offline de-duplication is possible but
would cost KSJC 42 % of its flights, leaving 12L at 12 and 30R at 39).

Note for the ts_transformer re-run: the `u` channel shifts uniformly by +33.5 m at KRDU.
Accuracy metrics (ADE/FDE, deviation vs reference) are computed against a reference in the
same frame and should be nearly unchanged, but the GATE verdicts were biased — the recorded
"gate-pass counts 0–4 of 152" was a ±3 m window scored against data offset by 33 m, so that
conclusion needs re-deriving rather than quoting.

Tests: 692 pass (557 modeling+backend, 135 aeroviz-4d/python), both suites exit 0. The
"one known pre-existing failure" in `run_all_tests.sh`'s header
(`test_fixed_time_objective_weights_control_effort_at_one`, numpy scalar conversion) did NOT
reproduce — that note and the matching CLAUDE.md Open Item look stale.

**Post-review hardening (same day).** A recall-mode review of the three fixes surfaced and
closed: ① `resolve_runway_threshold` still returned pavement ends — the `--runway` download
path would have named a threshold up to 775 m from the config's and drifted
`landing_time_utc`/`flight_key` between harvest paths; landing-threshold interpolation is now
single-sourced in `acquisition/runways.py` (`landing_thresholds_from_row`), generator output
byte-identical, plus a loud `ValueError` when a displaced end has no usable length.
② `_wins_against_parallel_runways` crashed on a heading-less threshold
(`math.radians(None)`) and its `inf <= inf` tie silently re-admitted double-assignment when
no sample fell in the centreline window — now: no competitors → win (no offset computed,
also removing a dead full-track scan), unestablished centreline vs a competitor → lose from
both. Also: `_heading_diff` reused instead of an inlined twin; true `statistics.median`.
③ `datum.py`: the ballpark probe was NaN-transparent (`abs(nan−33.53) > 1.0` is False) —
inverted to not-within-tolerance; an operator's explicit `PROJ_NETWORK` is no longer
overridden; `waypoints_to_msl` transforms the altitudes directly (EGM96 N is
height-independent — verified: +1000 m in → exactly +1000 m out), removing the
negate-and-subtract dance. ④ `FlightScenario.source` now records `altitude_source`, so
saved scenario files carry datum provenance (pre-fix HAE-era files lack the key).
⑤ ts dataset conversion moved after the cheap skip checks; test fixtures now import
`METRES_PER_DEG_LAT`/`metres_per_deg_lon`/`FT_M` from geokit instead of retired literals.
⑥ The symmetric OUT seam, closed after review discussion: modeling records (now MSL)
were packed straight into Cesium ellipsoidal `cartographicDegrees`, so after the batch
re-run every opt-/sim-/pred- entity would have drawn ~33.5 m above the white HAE
reference. `build_scenario_comparison_czml._states_to_waypoints` (the single choke point
all record-derived entities share; the reference bypasses it) now converts MSL→HAE via a
new `aeroviz-4d/python/vertical_datum.py` — a mirror of `flight_scenarios/datum.py`
(same KRDU −33.53 pin, ballpark probe, PROJ_NETWORK respect) per the `flight_identity.py`
precedent. Records are assumed MSL rather than tagged: all pre-datum-fix artifacts are
discarded wholesale (user decision), never fed back in.

**New operational scripts (same day).** `run_ts_pipeline.py` — the ts_transformer sibling
of the scenario optimization runner: per airport runs the 2×2 grid (iTransformer/PatchTST ×
window/full) as train → predict(test split) → evaluation report/HTML → comparison-CZML
publish (categories `ts_{itr|ptst}_{mode}`, matching the published naming); dataset build
+ flight_key split happen inside train and travel in the checkpoint. `clean_pipeline_data.py`
— wipes every generated artifact of both chains (scenarios, 4dTrajectory outputs incl.
ts dirs, frontend comparison + observed-layer CZML) with plan-print + confirm/`--yes`;
raw OpenSky downloads and `_`-parked research dirs are kept unless `--include-downloads` /
`--include-parked`; static airport layers and `data/archive` are never touched.

### 2026-07-20 — prediction overlay: anchor-time alignment + the lookback window is drawn

Two defects in how a ts_transformer forecast reached the globe, both invisible in the record
files (which were correct) and both living in `build_scenario_comparison_czml.py`.

**The forecast was drawn a whole lookback early.** A prediction record rebases its own time so
`t = 0` is the ANCHOR — the last observed sample the model was shown, `seq_len - 1` samples
into the approach. The reference is copied out of the airport's `trajectories.czml` and still
starts at `t = 0` = the START of the track. The builder wrote the record's times straight
through as CZML offsets, so the two shared a clock they did not share a zero on. Measured on
KRDU 05L / `AAL542_…`: `pred[0]` is bit-identical to the reference's `t = 118 s` sample, and
was plotted at `t = 0` — **12.0 km** from where the reference was at that instant. Every
prediction-schema entity is now shifted by `source.anchorTimeS`. Optimizer records were never
affected: their `t = 0` already is the scenario start.

**The lookback was never rendered.** `export.py` has always written `observed_states` as the
WHOLE observed track (negative `t` before the anchor) explicitly so a viewer could show the
input the model was conditioned on — and no viewer ever read it. The purple line simply began
in mid-air at the anchor. It now emits a second entity per group, `look-{group}`: the `t ≤ 0`
slice, same hue as the forecast at alpha 85 (`LOOKBACK_COLOR`) vs 225, frontend kind
`lookback` with its own "Predictor input" legend row. The anchor sample belongs to both halves
— it is literally the same state object in the record — so the faded half meets the forecast
exactly, not merely closely (asserted, not eyeballed). `observed_states` moved into
`_PREDICTION_SCHEMA`: a record that cannot be drawn completely now fails loudly.

Supporting cleanups: the lookback retraces samples the reference already covers, so it renders
path-only (a model/point there would draw a second aircraft on top of the reference's for the
whole input window); the entity-id→kind prefix table is now one list shared by `kindOfEntityId`
and `isComparisonEntity`, which had drifted — the picker's list was missing `pred-`, so
prediction tracks silently could not be hovered for their callsign; per-kind alpha
(`COMPARISON_KIND_ALPHA`) drives the legend swatch too, since "Predicted" and "Predictor input"
share a colour and a solid swatch made the two rows identical.

Rebuilt all four KRDU ts categories (`ts_itr_full` / `ts_itr_window` / `ts_ptst_full` /
`ts_ptst_window`, 152 groups each). Verified structurally across all 608 lookback entities:
every one starts at the reference's own `t = 0`, carries exactly `seq_len = 60` samples, ends
where its forecast begins, and never outruns its reference. 138 CZML-package tests, 460
frontend tests, tsc clean, `npm run build` clean. NOT re-checked in-browser.

### 2026-07-20 — B3: transport-consistent velocity channels + physical-velocity fit; third ts training generation

The findings doc's B3 bundle (`docs/findings_and_open_items_2026-07-20.md`), executed. The
one deviation from B3's literal scope was forced by measurement — see the second bullet.

**B3.1 — the position↔velocity inconsistency (A7) closed, at BOTH seams.**

- New numeric single source `geokit.wgs84_curvature_radii(lat_deg) -> (R_M, R_N)` (exact
  WGS84 closed forms). `flyability._transport_rates` now imports it (was an inline copy);
  the casadi geodetic RHS keeps its symbolic twin with a MUST-match mirror comment (a CasADi
  expression cannot call a float function). Pinned in geokit's tests at the equator/pole/45°
  landmarks.
- `ts_transformer/channels.py`: velocity channels are now the exact **chart derivatives** of
  the position channels — the physical velocity mapped through the full-transport Jacobian
  (`ndot = V_north·a/(R_M+h)`, `edot = V_east·a·cos lat₀/((R_N+h)·cos lat)`, `udot`
  unchanged); `states_from_channels` inverts exactly. Renamed `ve/vn/vu → edot/ndot/udot`
  deliberately: the channel tuple is serialised into every checkpoint and `load_checkpoint`
  refuses a mismatch, so every pre-change checkpoint fails loudly instead of silently
  mis-scaling velocities. New tests pin the factor closed form and the integration identity
  (a sequence generated by the geodetic kinematics integrates its velocity channels back
  into its position channels).
- **The fix as literally scoped in B3.1 would have made the measured inconsistency WORSE** —
  found by measuring, not reasoning. On all 995 KRDU arrivals (median whole-track drift of
  ∫v dt against the position channels): original east 3.5 / north 2.7 m/min; channels fixed
  alone east 3.4 / **north 8.6** m/min. Cause: `flight_scenarios._velocity_lsq` fitted
  velocity through the flat chart scales (`a`, `a·cos lat`), so its `V_north` overstated the
  physical value by `a/R_M` (+0.33% at 36°) — the old *channel* code cancelled that bias by
  accident, and A7's "cos ratio + h/R + R_M/R_N" attribution was really describing the FIT.
  Per the fix-upstream convention, `_velocity_lsq` now projects through the true tangent
  scales at the window anchor (`(R_M+h)`, `(R_N+h)·cos lat`), making its output the physical
  ENU velocity every consumer already assumed (the geodetic RHS, flyability's inversion, the
  ts chart). Final measurement, both seams fixed: east **2.4** / north **2.7** / up 0.45
  m/min — unbiased LSQ smoothing, no systematic left.
- Blast radius of the fit change: every fitted `V/psi/gamma` moves ≤ 0.33% / ≤ 0.1°;
  positions are untouched, so all position-based metrics and gates are unaffected. The
  2026-07-20 optimizer batch artifacts predate it (same situation as the geokit constant
  alignment below: a re-run would move initial-state V by ~0.2 m/s, far below data noise).
  Fixture constants in `test_start_state.py` / `test_scenario_optimization.py` were rebuilt
  on the tangent scales.

**B3.2 — `dt = 1 s` considered and declined** (README "Sizing"): same coverage needs
L=120/H=600 (~2× training cost) for almost no information — the source reports at ≤ 1 Hz
with ragged gaps and the velocity channels come from a 15 s window fit. `--dt` stays a knob.

**B3.3 — the lead-time table is restored on the raw-tensor accounting** (A8). The README
now carries BOTH accountings with their n: record accounting (one forecast per flight,
threshold-truncated — n falls with lead; 600 s survives for ≤ 1 flight) and raw-tensor
accounting from `history.json` `metrics.test.by_horizon` (every test window, sees past
truncation — 600 s restored with n = 893). On it, **iTransformer leads PatchTST at 600 s in
all three training generations** (6135/7142, 5407/6962, 5438/7384 m) at margins of
1.16–1.36× — direction consistent, each margin under the provisional band.

**Retrain (third generation).** Same recipe (ep=120, patience=15, lr=5e-4, seed=1337) and
the same `flight_key` split (702/141/152); all four cells trained, predicted (test split),
evaluated, flyability-reported. Pre-B3 artifacts moved to
`4dTrajectory/outputs/KRDU/_pre_b3_transport/`. Headline (152 test flights): iTransformer
full lateral 868/2750 m mean/p95, chained 1302/6576; PatchTST full 2016/5598, chained
3433/8993 — **full beats chained on threshold lateral in every generation** (1.5–1.7× mean).
Gate passes are 0/152 in all four cells this generation (pre-B3: 4, 1, 0, 0) — across
generations the count ranges 0–2.6% and the README now states only the stable conclusion
(a forecast is not a certifiable approach; a borderline pass is jitter). Flyability floor
(observed tracks) stays 63.2%; iTransformer window is again the only cell above it (73.7%).
The instance-norm ablation (`_ablation_norm/`) was NOT re-run — measured pre-B3, margins
1.2–2.7× vs a ≤ 0.3% channel rescale, structural argument unchanged; dated note added.

Suites: run_all_tests.sh 520 + 135 passed (the one known pre-existing collocation failure),
geokit 29. The ts suite is 56 tests (2 new transport pins).

### 2026-07-20 — full batch re-run (15/15 fresh), geokit per-degree constant aligned to the optimizer

**Batch.** The former combined runner with `--jobs 6`, 5 airports × 3 categories, 10,449 solves,
≈4 h 39 m wall clock (one harness-side background-task reap mid-run; resumed detached with
`setsid nohup`, `overall_fail=0`). All artifacts now post-date every 2026-07 fix (arrival
truncation, altitude floor/rollout guard, HS flip, identity unification) — the standing
"all batches are STALE" open item is closed. ts predictions were regenerated the same day,
**test split only** (152 flights per the reproducible flight_key split), for all four
checkpoints.

Identity contract verified on the fresh artifacts, all green: record stems are full
flight_keys; summary rows all carry `landing_time_utc`; **reference hit rate 100% in all 19
comparison categories** (15 optimizer + KRDU's 4 ts_pred — the pre-refactor
duplicate-callsign dropout was 22%); zero duplicate entity ids in sampled CZMLs; every
airport's categories.json intact (KRDU keeps 7 categories).

Headline solve/gate rates (success among solved): runway_cons is the cleanest everywhere
(93–99%), asdb the hardest (76–89%). Finding worth keeping: **KRDU RW32 is systematically
hard and NOT the old truncation artifact** — runway_cons RW32 79 offTarget + 59 failed of
198 (all other runways ≤9), and asdb RW32 fails 197/198 (IPOPT infeasible). Likely
procedure-specific (RNP-AR H05LZ; per-leg RNP still not extracted). KSTL runway_cons has a
milder cluster (12R/30R/30L/24; repeated single-IAF `PAULY` infeasibility).

**geokit alignment.** `METRES_PER_DEG_LAT` was the hand-rounded `111_320.0`; the
optimizer's NE frame (`approach_constraints.frame` and the NLP's metric-position
normalization) derives `WGS84_A·DEG2RAD = 111319.4908…` — a 4.6 ppm seam (~0.11 m at the
25 km ring) between the two frame families. Now defined as `WGS84_A * (π/180)` in
`geokit.constants` — bit-identical to the optimizer's product (IEEE commutativity),
`metres_per_deg_lon` stays pure `·cos(lat)`. Frontend `geoConstants.json` regenerated.
Applied AFTER the batch finished so all 15 cells share one constant; ts checkpoints are
unaffected in practice (inputs move ≤0.11 m at the ring edge vs km-scale model error — no
retrain). Also corrected the `channels.py` projection docstring: the flat chart deviates
from a true tangent-plane ENU by up to ~40 m at the ring edge (`e·n·tanφ/R` cross term) and
`u = Δalt` ignores the ~49 m curvature drop by design — the old "well under a metre" claim
was true only of the quantities the metrics actually measure (same-chart comparisons;
~0.2% local scale distortion at the ring edge, → 0 at the threshold).

### 2026-07-20 — flight identity unified end-to-end: entity ids = flight_key, positional `_N` re-uniquing deleted

The last identity holdouts (CZML entity ids, the comparison reference lookup, the FlightTable
optimizer join) still ran on bare callsigns + positional `_2/_3` suffixes. Diagnosis on the real
KRDU harvest:

- **Per-runway landings/arrivals files held massive duplicate ids** (128 duplicates across five
  runways; `N993FG` ×10 on RW32). Root cause: `collect_landings` harvests in CHUNKS and
  `classify_landing_flights` restarted its `_unique_id` numbering per chunk, while the
  cross-chunk merge de-duplicated by `(icao24, landing_time_utc)` without re-uniquing ids. The
  per-runway CZMLs inherited them — Cesium merges same-id packets, so two namesake flights
  rendered as ONE garbled entity (both tracks' samples interleaved from t=0), and
  `flightSummaries`/React row keys collided.
- **Cross-view id aliasing**: `merge_landing_flights` re-uniqued ids positionally for the
  combined file, so the same string (`SWA1692_2`) named DIFFERENT physical flights in the
  per-runway vs combined views, and the same flight got different `flight_key` stems from the
  two record writers (optimizer eats combined, ts eats per-runway) — breaking identity.py's
  "same stem" promise and making the comparison builder's callsign reference-lookup resolve to
  whichever namesake came first (the "wrong white line" open item).
- **FlightTable's optimizer join was callsign-keyed** (`byFlightId` from `group.flightId`), so
  namesakes swapped each other's V/mass/failed/offTarget facts.

Fix — one identity everywhere, display strictly separated:

- `generate_czml.build_czml`: entity id = `flight_key(flight)`, `name` = callsign; RAISES on a
  duplicate identity (silent Cesium merge → loud input error). Both `trajectories.czml`
  producers go through it.
- `_unique_id` deleted from the landing path (`classify_landing_flights`,
  `merge_landing_flights`); kept ONLY in `trajectories_to_czml_input` (the plain download path
  has no runway/landing time — the suffixed id is its one discriminator).
- `aeroviz-4d/python/flight_identity.py`: deliberate MIRROR of
  `flight_scenarios.identity.flight_key` (frontend tooling can't import the modeling tree);
  both copies pinned to `EJA969_05R_ad7f04_20260618T213736Z` in their own suites.
- Comparison builder: reference lookup by `group` (= flight_key = the new entity id) in batch
  mode and by `flight_key(source)` in single mode; `scenario_initial_map` keyed by flight_key
  (was `(id, runway)` — namesakes shared one V/mass); `_group_key`'s file-less fallback
  reconstructs the identity from the row (rows now carry `landing_time_utc` via `summary_row`
  — it IS part of the identity and was the one missing field).
- Frontend: `useFlightOptimizerData` keys by `group.group` (renamed `byFlightKey`);
  FlightTable/approach view display the callsign (`name`) while ids stay the
  selection/join/cache identity; `ObservedFlightSummary` gained `callsign`.

Verified: 57 aeroviz-4d python tests, 59 trajectory_data_process, 84 optimization+ts, 453
vitest, tsc + vite build — all green. New pins: namesake entity ids distinct + duplicate
identity raises (generate_czml); each comparison group copies ITS OWN namesake's reference
track; FlightTable keeps namesake optimizer facts apart. Artifacts regenerated after this
change (arrivals/CZML rebuild + full batch re-run); ts checkpoints are UNAFFECTED (per-runway
arrivals `id` fields — and therefore split keys — are byte-identical; only the combined view
changed).

### 2026-07-19 — flyability check, instance-norm re-ablation on real data, predictions in the frontend

Three things asked for together: the post-hoc flyability check (route 1 of the README's four),
re-ablating `--instance-norm` on real data, and getting prediction results to render in the
frontend alongside the optimizer's.

**Post-hoc flyability (`4dTrajectory/ts_transformer/flyability.py`, 16 tests).** The
load-factor point-mass model inverts in closed form — `A = psi_dot V cos(gamma)/g = n sin(mu)`,
`B = gamma_dot V/g + cos(gamma) = n cos(mu)`, so `n = hypot(A,B)`, `mu = atan2(A,B)`; `n` fixes
`Cl`, `Cl` fixes drag, and `T = m(V_dot + g sin(gamma)) + D` closes it. One pass, no solver, no
casadi, so it lives in the torch env. Earth-frame transport terms are subtracted first.
`predict` writes `flyability_report.json` and prints a summary line.

- **The calibration matters more than the check.** First run against REAL flown tracks scored
  **0/149 fully flyable** — those are trajectories real aircraft flew, so the check was wrong.
  Cause: `thrust_negative`. Median required thrust on a real arrival is **0.43 kN** (idle), and
  a negative requirement means the aircraft needed more drag than a clean airframe has —
  speedbrake, flaps, gear. Every approach does this; one clean-configuration polar cannot
  represent it. Reclassified as SOFT (reported, not counted unflyable), and the report leads
  with the delta against the observed tracks measured by identical code, because both sides
  carry the same polar bias. **The observed baseline is the floor, not 100%.**
- `Cl_max` from `aero_params_for_aircraft` (2.7 for an A320), NOT `LoadFactorSimulator`'s
  hardcoded 1.5 — they disagree by 80% and `aero_params.py` is the documented source of truth.

**Instance-norm re-ablation on real KRDU data — the synthetic OFF default holds.** All 8 cells
(2 models × 2 horizon modes × on/off), same hyperparameters, each graded on its own
checkpoint's test split; artifacts in `4dTrajectory/outputs/KRDU/_ablation_norm/`, roll-up in
`ablation_results.json`.

- **OFF wins 19 of 20 accuracy comparisons, one tie**: every cell on val loss, FDE, ADE p95 and
  lateral p95; 3 of 4 on mean ADE with PatchTST window a dead heat (2580 vs 2571 m — the cell
  where OFF wins the other four metrics). A sweep that lopsided is not run-to-run variance,
  which was the open question: individual gaps here are much smaller than on synthetic data
  (1.2–2.7× vs 2.4–6.5×), and an earlier partial pass on ONE metric had shown an apparent
  reversal in that same PatchTST window cell. It did not survive consistent scoring.
- **The signature is lateral p95**: all four instance-norm-ON cells land at 14.28–14.50 km —
  near-constant across both architectures and both horizon modes. That is a model that cannot
  place the endpoint at all; strip the absolute level and the prediction ends at a distance set
  by the frame, not by the flight. OFF spans 2.6–8.5 km, i.e. it varies with the flight.
- **Flyability moves the OPPOSITE way** (ON better in 3 of 4 cells; PatchTST window 89.3% vs
  29.6%, i.e. the configuration 2.2× worse at the threshold looks three times more flyable).
  Instance norm is not flying better, it is predicting blander paths — easy to fly, far from
  the truth. A straight line is perfectly flyable and completely wrong. Recorded in the README
  because it is the standing argument for never reading flyability alone. (iTransformer full is
  the exception at 46.1% off vs 34.9% on.)

**Headline KRDU runs retrained, and two published conclusions did not survive it.** The 4
published checkpoints carried a split that `hash(flight_key)` reproduces for only 552/995
flights — they predate the per-flight-hash split fix, exactly the "retrain before comparing"
note in the entry below. The partition itself is clean (train/val/test verified disjoint), so
the published numbers were not wrong, just on a split no current run can reproduce and not
comparable with the ablation. Retrained on the current keying (702/141/152):

- **"The compounding cost lands in the tail, not the mean" — withdrawn.** It rested on lateral
  mean 1.5× vs p95 2.2×; on the new split the two ratios are equal to within noise
  (iTransformer 1.55× vs 1.61×). A tail effect survives in FDE (PatchTST p95 1.78× against a
  1.36× mean) but that is a different claim than the one written.
- **"Zero gate passes in all four runs" — withdrawn.** iTransformer now passes 3/152 (full)
  and 1/152 (chained). The substance holds (2% is not a usable predictor) but "zero, always"
  was a property of that split.
- **Survived:** one-pass full beats chained window on whole-approach lateral error for both
  models on both splits (1.5–1.6× mean, 1.5–2.1× p95), and the short-lead/long-lead crossover
  between PatchTST and iTransformer (now cleanest within the full-mode pair: 184 vs 571 m at
  10 s, reversing by 300 s).

Standing lesson recorded in the README and CLAUDE.md: **treat any single-split margin under
~1.5× as provisional** — that is the size of effect a split change moved here.

**Flyability shipped with a wrong assumption, caught by its own guard.** The check graded a
whole batch against ONE envelope, documented as safe because "every harvested arrival is type
UNK and resolves to the single `--aircraft-type` fallback". A boundary assertion added to state
that assumption fired on the first real batch: `_resolve_aircraft` falls through to an
**`icao24` → OpenAP lookup** that recovers the real airframe — 20 distinct types across 400 KRDU
arrivals, A320 only 224 of them. So ~44% of every batch was being judged by an A320's `Cl_max`
and max thrust, invisibly, because the report never named the airframe it used.
`report_for_records` now takes one `Aircraft` per flight, builds one envelope per distinct type,
and the roll-up carries `fleet` + `envelopes`. All flyability numbers moved: the observed-track
baseline went 58.4% → **63.2%**. The CLAUDE.md/README claims about UNK were corrected.

**`summary.json` now carries an `accuracy` block** (+ per-row `ade_m`/`fde_m`/`overlap_steps`).
A batch's error against the observed tracks is its headline result and it existed only as a
printed mean — comparing eight ablation cells meant scraping stdout. Mean AND p95, because
chained window-mode error compounds into the tail. `overlap` is a required argument to
`write_batch`, not optional: an optional metric is one that silently goes missing.

**Predictions render in the frontend.** `build_scenario_comparison_czml.py` detects the record
schema (`optimizer_states`/`simulator_states` vs `predicted_states`) and emits `pred-` entities
for a prediction batch; the entity-id prefix is already what the frontend keys `kind` off, so
`predicted` gets its own purple colour and legend checkbox. Two follow-on fixes:

- **Off-target recolouring restricted to the optimizer schema.** A forecast essentially always
  misses the 106.75 m gate, so it marked 27/27 groups off-target and repainted every prediction
  yellow — the kind colour was never once visible and a marker that fires on everything carries
  no information. `properties.status` stays accurate; deviation is reported by the evaluation
  report and comparison index.
- **The frontend repaint skip was keyed on the wrong thing.** It skipped legend repaint whenever
  `status === "offTarget"`, but what it exists to preserve is a baked VERDICT colour (the
  yellow). Predictions never get that bake yet are always `offTarget`, so they rendered from the
  CZML — matching their legend swatch only because `PREDICTION_COLOR` and the TS legend entry
  happen to hold the same RGB. Now keyed on whether a verdict colour was actually baked.

**Verified in-browser** (Linux Chrome, KRDU, all 4 categories): purple prediction + white
reference paths render together, the `Predicted` legend checkbox removes only the purple, and
the Optimization panel's metrics follow the selected category (752 m / 2.0% iTransformer full
vs 3184 m / 0.0% PatchTST window, matching the evaluation reports — and PatchTST window is
visibly the more scattered fan, an independent read of the same gap). Also backed by tsc clean,
451 frontend tests, `npm run build`, 54 ts_transformer + 25 CZML-builder tests, and a structural
check of the published artifacts against every contract point the frontend reads. Gotcha found
while verifying: comparison entities are time-windowed, so at a clock time outside a group's
availability the scene is legitimately empty — pause inside a window before concluding the
overlay is broken.

### 2026-07-19 — code review fixes: ts_transformer contracts, env resolution, identity single-sourcing

Applied the 15 findings of a full-diff review (ts_transformer + the uncommitted env-script changes):

- **train.py**: a non-finite train/val loss now RAISES ("training diverged…") instead of sailing through the best-val bookkeeping — previously a run that went NaN before any finite improvement wrote the last (NaN) weights as a "successful" checkpoint and emitted literal `NaN`/`Infinity` into history.json. `load_checkpoint` now uses `weights_only=True` (payload is tensors + primitives) and refuses a checkpoint whose serialized channel order differs from `channels.CHANNELS` (a same-length reorder loaded cleanly and silently mis-mapped every channel). Flights too short to yield one training window (window mode: `seq_len + pred_len`; build only requires `seq_len + 1`) are excluded — counted, not silent — before the split, so they no longer occupy split slots and produce the misleading "empty window set" abort.
- **Aircraft type is part of the checkpoint**: `TSConfig.aircraft_type` (default A320, moved to config.py) is serialised with everything else; predict defaults to the train-time value instead of its own CLI default, and warns when explicitly overridden — the type sets target Vref/TCH, i.e. the ENU frame and the gate target the normalizer stats were fit under.
- **predict --device**: the compute device is a runtime property, no longer read from the checkpoint (an explicit-`cuda` checkpoint was unusable on a CPU-only machine with no override).
- **Horizon cap is stated, never silent**: window-mode chaining still stops at 300 steps (600 s), but a forecast that ends short of the threshold is flagged `horizonCapped` (record source), `horizon_capped` (summary row), and predict prints a per-batch WARNING — previously those ~2% of flights were graded as huge gate failures indistinguishable from model error.
- **Split stability**: train/val/test assignment is now a per-flight sha256 of `(seed, flight_id)` instead of a positional permutation — adding/removing one flight to the harvest reshuffled the entire split (old test flights silently entered training on a retrain). Fractions are now approximate; empty train/val raises with the real cause (tiny datasets previously rounded val/test to zero and died later blaming window sizes). NOTE: this reshuffles the split once relative to the existing KRDU checkpoints — retrain before comparing new runs against them.
- **synthetic.py**: icao24 derived via `zlib.crc32(runway)` instead of the process-salted builtin `hash()` — identically-seeded synthetic data now has identical flight identities across interpreters (previously `predict --split test` on regenerated data intersected on zero flights).
- **Identity + filename single-sourcing**: `flight_key` moved to `flight_scenarios/identity.py`; `scenario_optimization._scenario_filename` and ts dataset/export both import it (the two copies had already drifted: `flight{i}` vs `scenario{i}` fallback — unified on `flight{i}`). Record-filename suffixes + `REFERENCES_DIR` hoisted into `optimization/evaluation_export.py`, imported by both writers; the "MUST match" mirror comments are gone.
- **dataset.py**: falling through to `*_landings.json` (a download-only dir with no `*_arrivals.json`) now prints an explicit UNTRUNCATED-tracks warning — a different task/duration distribution than the windows were sized for.
- **Env resolution deduplicated + made content-aware**: new `scripts/activate_aeroviz_env.sh`, sourced by `run_all_tests.sh` (warn-and-continue) and `start_aeroviz_fullstack.sh` (abort). Candidates are probed with `import casadi` instead of trusted by name (on this machine `aviation` exists but is another project's env — both scripts previously accepted it when active and fell back to it when `aeroviz` was missing or `AEROVIZ_CONDA_ENV` was mistyped; the launcher then crash-looped the backend on `import casadi`). An explicit `AEROVIZ_CONDA_ENV` is now the ONLY candidate. The launcher ACTIVATES the env rather than direct-exec'ing `envs/<env>/bin/python`, so activate.d hooks (the libstdc++ `LD_LIBRARY_PATH` fix) apply to the backend subtree; missing conda now reports "conda is not on PATH" instead of probing `/envs/...`.
- **Docs**: CLAUDE.md's Key Defaults ts bullet updated to the real 60/300 window sizes and the measured-distribution rationale (it still carried the superseded 30/150 + "~3.5–5 min arrival" story that config.py refutes); requirements.txt pin rationales corrected (the torch comment claimed `weights_only=True` while the code passed `False` — now the code matches the claim; the numpy `copy=None` rationale matched nothing).

Follow-up pass over the review's below-the-cap cleanup findings (same day):

- **forecast.py**: the HORIZON_FULL branch of `forecast_approach` was a copy of `recursive_forecast`'s single-pass body that also skipped its anchor validation — both modes now run through `recursive_forecast` (full mode's `pred_len` covers `max_steps`, so it is one pass; an explicit `anchor < seq_len - 1` now gets the clear ValueError instead of a cryptic vendor shape error). `_forward` forces `model.eval()` (dropout noise from a train-mode model would compound through the chain). `default_anchor` dropped its never-read `series` parameter.
- **models.py**: PatchTST now receives `act=config.activation` (the vendored Model takes activation as a bare kwarg, so `TSConfig.activation` previously applied to iTransformer only while the checkpoint recorded it for both).
- **metrics.py**: `displacement_metrics` folded into `trajectory_metrics`; `error_components` computed once per call (was twice, with per-step displacement recomputed twice more) via a `displacement_grid` key; `_positions` uses `channels.POSITION_IDX` instead of re-deriving it. `_spread` stays vectorised — delegating to the stdlib-only `evaluation/stats.signed_spread` would sort millions of boxed floats in `evaluate_split` — but a new seam test pins the two equal on the same input, replacing the mirror comment with a checked property.
- **export.py**: `PredictionRecord.source`/`final_time_s` became read-through properties (three stored copies of `final_time_s` per record could disagree after a future edit; `evaluation` rejects `final_time_s != states[-1].t`); the whole observed track is converted to states ONCE and the anchor sample + reference span are slices of it (the per-sample GeodeticState loop previously ran twice over the tail); summary rows go through the new `evaluation_export.summary_row` (shared with `scenario_optimization._summary_record`, which now wraps it); a redundant `min()` removed.
- **train.py**: `_predict_split` stores decoded windows as float32 (float64 doubled the live peak — hundreds of MB at full-mode scale on the 16 GB swap-bound machine — for precision metre-scale metrics cannot use); the duplicated val `evaluate_split` call in both ternary arms collapsed.
- **__main__.py predict**: the split filter now runs on the RAW flight dicts (keyed identically to `build_series` via `flight_key`) BEFORE the expensive series build — a default `--split test` predict previously built and discarded ~85% of the work. Verified live: "built 1/1 series" for a 14-flight file.
- **dataset.py**: the track-span check moved BEFORE `build_scenario` (too-short flights no longer pay aircraft resolution + per-sample least-squares fits before being skipped; the raw-waypoint span is the identical value); dead code removed (`Normalizer.decode_torch`, `FlightSeries.duration_s`/`.source`).
- **channels.py**: dead `VELOCITY_IDX` + `Frame.to_dict`/`from_dict` removed; new `Frame.latlon_from_en` single-sources the inverse projection, used by `states_from_channels` AND `synthetic.py`'s waypoint generator (which previously hand-rolled a fifth copy of the frame formula; it now builds a `channels.Frame` and dropped its direct geokit constant imports).
- **package.json**: `npm run backend` resolves its interpreter through `scripts/activate_aeroviz_env.sh` instead of bare `python` (it was the third, uncoordinated env-selection seam).
- Deliberately NOT applied: batching the predict loop (would complicate the single-flight inference seam for a non-bottleneck), early-stopping the window-mode chain at the threshold (changes the measured object — `truncate_at_threshold` takes a global argmin), the vendored PatchTST `pv()` NameError on non-default positional encodings (`layers.py` is documented "copied whole, unmodified"; fixing it breaks the byte-identical vendoring contract — landmine noted here instead), and a src-layout re-packaging of ts_transformer's flat modules (the sys.path-shadowing hazard is real but latent; disproportionate to fix now).

### 2026-07-19 — ts_transformer: first REAL-data run (KRDU, 995 arrivals) — both models × both horizon modes

Harvested ADS-B landed (3747 arrivals, 5 airports, 815 MB). Full matrix trained and graded on KRDU: 995 arrivals over 6 runways, split by flight 697/149/149, 96k training windows, 120-epoch cap / patience 15 / `lr=5e-4`, RTX 4060. All four prediction batches graded by `python -m evaluation`. Artifacts under `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` and `ts_pred_{model}_{mode}/`.

**Displacement error at matched lead times** (the only cross-mode-comparable axis — headline ADE/FDE average over different horizon-length distributions and must not be compared):

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | 259 | 390 | 687 | — | — | — |
| iTransformer | full | 587 | 611 | 840 | 1502 | 3227 | 6135 |
| PatchTST | window | 158 | 409 | 995 | — | — | — |
| PatchTST | full | 253 | 399 | 830 | 1914 | 3983 | 7142 |

**Whole-approach prediction graded at the threshold** (149 test flights, directly comparable):

| model | mode | lateral mean | lateral p95 | path deviation |
|---|---|---:|---:|---:|
| iTransformer | full | **1070 m** | **3136 m** | **1895 m** |
| iTransformer | window (chained ×10) | 1594 m | 6898 m | 1918 m |
| PatchTST | full | 1804 m | 4666 m | 2488 m |
| PatchTST | window (chained ×10) | 2815 m | 7354 m | 3152 m |

Findings: (1) **the cost of chaining lands in the tail, not the mean** — chained-window vs one-pass-full is 1.5× on lateral mean but 2.2× on p95, because once one chained pass goes wrong the next nine extrapolate from a wrong history; training directly for the long horizon beats chaining a short one. (2) **iTransformer beats PatchTST at long lead for a structural reason** — PatchTST is channel-independent (`TSTiEncoder`), so it cannot represent the east/north coupling of a turning aircraft, while iTransformer attends *across* variates; note the reversal at 10 s lead (PatchTST 158 vs 259 m), where the aircraft is nearly straight and independence costs nothing. (3) **0/149 gate passes in all four runs is the honest result** — 106.75 m is FAA containment for a planned/flown approach, not a forecast-accuracy target; the number measures the gap between a statistical prediction and a certifiable trajectory. (4) Real is much harder than synthetic (423 vs 286 m ADE): synthetic approaches are straight-in, real ones are vectored, and *when* the turn onto final happens is a controller's decision a single-aircraft model structurally cannot see.

**Three real bugs the first real-data run exposed**, none of which a loss curve would have shown:

- **Silent training-data contamination.** A harvest directory holds five overlapping views of the same flights — `*_arrivals.json` (truncated, the training input), `*_landings.json` (SAME flights untruncated), `*_combined_czml_input.json` (all runways merged), plus `*_heading_rejected.json` / `*_local_rejected.json` (tracks the harvester explicitly THREW OUT). `glob("*.json")` loaded every flight three times over plus the known-bad ones. `dataset.select_flight_files` now takes the first matching pattern only, never mixes, always excludes `*_rejected*`, and prints what it skipped.
- **Aircraft type is `"UNK"` for all 3747 flights** and `flight_scenarios._resolve_aircraft` raises rather than guessing — the batch died on flight #1. Added `--aircraft-type` (default `A320`, printed every run). Not cosmetic: it sets the target state's Vref and threshold-crossing height, i.e. what the gates measure the final state against.
- **Sizing was wrong twice** (see the environment entry below for the corrected defaults).

### 2026-07-19 — Environment consolidated into `aeroviz`; the `aviation` name collision documented

`aeroviz` (py3.12) is now the single thesis environment — `torch` installed alongside the existing acquisition (`traffic`, `pyopensky`), CIFP (`cifparse`, `arinc424`), `casadi`/IPOPT, `openap` and conda-forge geospatial stacks. `run_all_tests.sh` needs no change: its `4dTrajectory` entry now covers the ts_transformer suite too.

**The name collision.** CLAUDE.md said "Python env: conda `aviation`", carried over from another machine — `run_all_tests.sh`'s own comment records "Env names differ per machine (`aeroviz` here, `aviation` elsewhere)". On THIS machine `aviation` is the editable env of `/home/supercomputing/studys/AivationTransformer`, an unrelated project (pure-pip, py3.11, 254 packages, 8.6 GB). Acting on the stale line nearly deleted the real thesis env; the ts_transformer package was also initially targeted at `aviation` for the same reason. Both CLAUDE.md lines are corrected and the collision is now documented under Environment.

**Consolidating the other way is BLOCKED — measured, not assumed.** `cifparse` >= 2.0.4 (aeroviz has 2.0.9) uses PEP 701 f-strings (nested same-type quotes), which is Python 3.12+ syntax. Every version from 2.0.4 up fails `compileall` on py3.11; only 2.0.0 and earlier import there — a 9-patch regression in the ARINC 424 parser that feeds `approach_constraints`. Upstream's PyPI metadata claims `requires_python >=3.10` and is simply wrong. So a py3.11 env cannot host the thesis at current package versions. `casadi`/`openap`/`arinc424` install fine on 3.11; `cifparse` alone is the blocker.

**One real interaction between torch and the existing stack, found and fixed.** `import torch` then `import traffic` raised `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`; the reverse order worked. Cause: pip's manylinux torch wheel has no RPATH into the env, so it resolves `libstdc++.so.6` from the system (CXXABI ≤ 1.3.13), and once that SONAME is loaded conda-forge matplotlib's `_c_internal_utils.so` — which needs 1.3.15 — is answered by the already-loaded old one. `run_all_tests.sh` runs every suite in ONE pytest process with `4dTrajectory` (torch) listed before `trajectory_data_process` (traffic), i.e. precisely the failing order. Fixed with `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh` prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (plus a matching `deactivate.d`). Note it only takes effect under `conda activate` — calling `envs/aeroviz/bin/python` directly still reproduces the failure. Full suite after the fix: **495 passed, 1 failed** (the pre-existing `test_fixed_time_objective_weights_control_effort_at_one` numpy scalar-conversion `TypeError`); the count rose from 464 because the 32 ts_transformer tests now run in the same invocation.

Anything installed into `aviation` during the attempt (casadi, openap, arinc424, the broken cifparse, plus a tqdm/wcwidth bump and an editable geokit) was rolled back; that env is verified back at torch 2.9.1+cu128 / tqdm 4.67.1 / wcwidth 0.3.0 with no thesis packages. Env spec backups for `aeroviz` live in `.env-backup/` (pip freeze, conda explicit, environment.yml).

**ts_transformer defaults re-sized against the real data** (see the entry below for the original synthetic sizing). The 3747 harvested arrivals have durations p5 235 s / p50 328 s / p90 607 s / p99 920 s — much longer than the "25 km at 120 m/s ⇒ 3.5–5 min" straight-line estimate, because real arrivals are vectored (downwind legs, base turns, holds), so the flown path far exceeds the straight-line distance to the entry ring. `seq_len` 30 → **60** (120 s), `DEFAULT_PRED_LEN_FULL` 150 → **300** (600 s): full mode now covers the complete remaining approach for **97.8%** of flights, where 150 covered **57.6%**.

### 2026-07-19 — `4dTrajectory/ts_transformer`: iTransformer + PatchTST integrated as a learned-prediction sibling to the optimizer

New package answering *what trajectory WILL this aircraft fly* (learned, no dynamics model) alongside `optimization`'s *what SHOULD it fly*. Both emit the same evaluation records, so `python -m evaluation --input <dir>` grades either against the identical regulatory gates.

**Vendoring.** `vendor/itransformer/` (MIT, thuml @ `c2426e6`) and `vendor/patchtst/` (Apache-2.0, yuqinie98 @ `204c21e`) copied byte-identical with only import paths rewritten, each with its `LICENSE` + a `PROVENANCE.md` recording what was copied/dropped and why. Vendored rather than installed because neither upstream is a packaged library and both resolve internal imports through a top-level `layers/` — installed side by side they collide (both ship a different `layers/Embed.py` and `layers/SelfAttention_Family.py`). Dropped from iTransformer: the Reformer/Flowformer/Flashformer/Informer attention variants, and with them the `reformer_pytorch` + `einops` deps. One shared `TSConfig` drives both (upstream's own argparse-namespace contract); `models.py` adapts iTransformer's 4-arg call and PatchTST's 1-arg call to one `model(x)`.

**Env.** Installed editable `geokit` into conda `aviation` (torch 2.9.1+cu128); the package is casadi-free by design, so it lives in the torch env while the optimizer stays in `aeroviz`. `requirements.txt` written. Verified both models forward CPU+CUDA on the RTX 4060.

**Three findings that changed the design, each measured not assumed:**

1. **Sizing.** An arrival truncated at the 25 km ring is only **~3.5–5 min** (~110–150 samples at 2 s), not the 8–12 min first assumed. The initial defaults (L=60 @ 4 s = 4 min lookback) exceeded a whole approach and skipped **5 of every 6** flights as "shorter than one window". Now `dt=2 s`, `L=30`, `H=30` (window) / `150` (full) — all 120 synthetic flights build, ~65 anchors each.
2. **Instance normalisation must be OFF.** iTransformer's `use_norm` / PatchTST's `revin` are ON upstream and strip each window's absolute level as nuisance; in a threshold-anchored ENU frame absolute position *is* the signal. Off wins in **all four** model×mode cells by 2.4–6.5× on ADE (iTransformer window 771→286 m, full 1972→303 m; PatchTST window 910→672 m, full 2268→701 m) — and on converges *sooner* (49–56 vs 74–90 epochs), i.e. to a worse optimum, not undertrained. Defaults flipped; `--instance-norm` re-enables for re-ablation on real data.
3. **Two alignment bugs caught by running the real thing** (both would have produced plausible-looking wrong numbers): (a) `FlightSeries.flight_id` was the callsign, which repeats daily and across runway files — `predict --split test` returned 48 flights for an 18-flight split, and the train/val/test split leaked. Identity is now `dataset.flight_key` = `id_runway_icao24_landingTime`, the same function that produces the record filename. (b) The reference record covered the whole track while the prediction covered anchor→threshold; `evaluation.reference` resamples both at 101 fractions of *their own* arc length, so it reported 4349 m of "path deviation" that was pure span mismatch — **833 m** once span-matched. Records now anchor `t=0` at the anchor sample with `initial_state` the observed state there, and `states[0] == initial_state` as in an optimizer record.

**Contracts.** Channels `(e, n, u, ve, vn, vu)` in a threshold-anchored ENU frame; `psi`/`gamma` never regressed directly (±π wrap averages 179° and −179° to 0°, pointing the aircraft backwards exactly at the turn onto final) but derived as `atan2(vn, ve)`, which *is* the math-ENU convention, so the compass/ENU substitution has no place left to happen. `m` carried, never predicted (unobservable from ADS-B). Records are reference-shaped (`controls == []`) and built via `optimization/evaluation_export.py` — importable here because it is casadi-free — rather than hand-rolling a second copy of the record JSON. `final_time_s` always from `states[-1]["t"]`, never `pred_len × dt` (threshold truncation makes them differ; `evaluation.records` rejects >1e-6).

**Status.** 30 tests pass; verified end-to-end through the real CLI (train on GPU → predict → `python -m evaluation`). **Never trained on real data** — none existed in the tree (`trajectory_data_process/outputs/` absent, no `credentials.json`), so `synthetic.py` generates straight-in approaches for smoke-testing and fixtures. Every number above is synthetic and is plumbing evidence, not a result. Pre-existing unrelated failure noted: `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` (numpy scalar-conversion `TypeError`).

### 2026-07-07 — "Profile" → "Approach view" rename (systematic)

The 2D approach page/toggle is now **Approach view** throughout (UI text, identifiers, CSS, files; `git mv` preserved history): `useApproachView.ts`, `ApproachViewPanel.tsx`, `ApproachViewToggle.tsx`, `approachViewAnalysis.ts`, `approachViewSources.ts`; identifiers like `isApproachViewOpen`/`setApproachViewOpen`, `approachViewMode`, `ApproachViewTrack/Sample/Input`, `buildApproachViewTracks`, `planApproachViewSources`; CSS `.approach-view-*`; toggle labels "View"/"Hide view". Deliberately NOT renamed — the geometric "profile" (altitude cross-section) keeps the term: `runwayProfileGeometry`/`RunwayProfilePoint`, `procedureProfileProjection`, `procedureVerticalProfileOverlay`, `ProfilePlot`/`ProfileOverlay`, the "Vertical profile" side-view mode label. Not re-checked in-browser.

### 2026-07-07 — Approach view: whole-track plotting, review fixes, perf cache (three passes, same day)

- **Whole track**: `sampleEntityTrack` walks outward in both directions from the current time (`TRACK_SAMPLE_STEP_SECONDS = 5`, `MAX_TRACK_SAMPLES_PER_DIRECTION = 600` backstop) returning the whole time-ordered track; all points kept tagged by containment tier; an aircraft is plotted iff some sample reaches PRIMARY (`trackEngagesProcedure` — full approach for flights that fly the procedure, unrelated traffic excluded). `splitTrackByContainment` splits ordered samples into contiguous runs sharing boundary points (gapless lines); PRIMARY/SECONDARY drawn solid (primary brighter), OUTSIDE dashed `6 5` + dimmed.
- **Review fixes**: walks also stop on `samePosition` (the HOLD-tail gotcha — otherwise never terminates and pads 600 duplicate threshold points per entity per tick); landed/parked entities dropped via liveness check (sample one step back == current → past the real track end → null); not-yet-airborne (null current) dropped; plot domain grows to include plotted track samples (corner-cut stretches no longer clipped) and the selected-flight label gets the plot clipPath; `ApproachViewInput.current` tightened to non-null; engages-gate = `trail.some(PRIMARY)`.
- **Perf**: per-entity cache of the classified whole track (`trackCacheRef` keyed by flight id), rebuilt only when geometry deps change (loaded procedure/frame, active routes, source set) — never on clock ticks; each ~120 ms tick then only re-checks liveness (1–2 `getValue`) + classifies the current marker (O(aircraft × track length) → O(aircraft) per tick). `sampleEntityTrack` split into `currentIfLive` + `sampleWholeTrack`; analysis module exposes `classifyProfileSample` / `classifyTrackSamples` / `trackEngagesProcedure` / `sortProfileTracksBySelection` (`buildApproachViewTracks` kept as their batch composition for tests).
- DEFERRED: interior-gap `break` in the walk is latent (current CZMLs are single-interval); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case. Not re-checked in-browser.

### 2026-07-07 — Approach view mirrors the active tab

- New pure `src/data/approachViewSources.ts` `planApproachViewSources(...) → {observed, optimized}` — the single source for "which trajectory sources are the current tab's globe content": `observed` reuses `planObservedTracks(...).visible`; `optimized` = `mode === "optimize" && hasOptimizedSource`. Used by BOTH the hook (what it samples) and the panel's "CZML linked" badge (`profile.sourceLinked`). Observe plots observed; Optimize plots only the optimized playback; Fly/Compare plot neither.
- **Root-cause follow-up**: `planObservedTracks` no longer keeps the observed CZML loaded behind an open view outside Observe (`relevant = mode === "observe"`). `useCzmlLoader` drives the shared `viewer.clock` from the observed CZML's hours-long span, so loading it in Optimize hijacked the clock and made the optimized playback aircraft vanish; releasing it also saves a 100+ MB load per non-Observe tab. Runtime clock behavior not re-checked in-browser.
- KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view — Observe-with-comparison plots neither source; wiring it in is a follow-up.

### 2026-07-07 — Pipeline exposes the control mesh; mesh defaults single-sourced

- `run_scenario_optimization.py --n-segments` (unconstrained) / `--n-seg-per-phase` (constrained), either/or by mode, defaulting to CollocationOptimizer's own 8/3; validated ≥2/≥1 at both CLIs.
- The constrained batch path never passed `n_seg_per_phase` before (stuck at the default with no override) — now threaded main → `optimize_scenarios_constrained_iaf` → `_optimize_one_scenario_iaf` → both IAF selectors → `_solve_iaf` → `CollocationOptimizer`.
- Defaults single-sourced in `collocation/optimizer.py` (`DEFAULT_N_SEGMENTS`/`DEFAULT_N_SEG_PER_PHASE`); backend + batch import them; frontend "Control segs/leg" default aligned 2→3 (defaults had silently diverged: frontend 2, backend fallback 4, optimizer/batch 3). The frontend→backend HTTP path had always wired `nSegPerPhase` correctly — the gap was batch-only. Verified: nsp 2→4 changes the constrained plan 85→125 nodes.

### 2026-07-07 — Optimize tab: solve-time readout; HSL hook benchmarked, kept dormant

- `optimization_backend._optimize` puts `result["timings"] = {buildS, solveS, playbackS, totalS}` in the response (was log-only); frontend `OptimizationTimings` + `parseTimings` (absent → null) + a "Solve time" row. Plan readouts restyled to a single column of full-width rows (3-col grid truncated values).
- HSL: `collocation/components.py` reads `AEROVIZ_IPOPT_LINSOL`/`AEROVIZ_IPOPT_HSLLIB` (casadi's IPOPT has the `hsllib` loader — no rebuild; setup in `docs/hsl-linear-solver-setup.md`). Measured: free Coin-HSL Archive has only MA27, 3–27× slower than MUMPS here (small NLPs; OpenBLAS/OpenMP clash) → dormant, kept for a future MA57 academic license.

### 2026-07-06 — Review pass: toggle false-open, ownership seams, manifest-only eval reads, required guards

- **[BUG]** `runwayMatchesSelection(null, X)` is match-ALL, so with the top bar on "All runways" every toggle showed "Hide profile" and clicks closed the invisible page; `open` now also requires `selectedRunway !== null`.
- `RunwayProfileToggle` prop `borrowSelection` (PilotPanel, unconstrained only): opening saves the pre-open `{selectedRunway, isRunwayProfileOpen}` (only when opening actually changes the selection) and restores on close/unmount. Constrained Optimize doesn't borrow — the forced-display hook owns the runway there.
- `useForcedProcedureDisplay`: `SavedDisplay` now carries the profile-open state, restored together with the runway (ownsRunway only); `forceRunway: null` never touches either. `ready` flip only fires when a drive is pending; savedRef bakes `ownsRunway` (what makes a non-null→null `forceRunway` flip restore correctly).
- **Evaluation read side manifest-ONLY** (user decision, no glob fallback): `evaluation.records.load_records` reads a batch dir via its `summary.json` roster; manifest-less dir / listed-but-missing / empty roster raise; `--pattern` removed from both CLIs. (Globbing counted orphans — the KRDU 1023-vs-996 class.)
- **`min_altitude_m` REQUIRED** on `rollout_controls`/`simulate_controls` — the 0.0 default silently validated diverged replays km below elevated fields; target-less replays now pass an explicit 0.0.
- **Constrained-ness is an explicit manifest field**: `_upsert_category` stamps `"constrained": bool`; builder `--constrained`; frontend `ComparisonCategory` requires it, `_cons`-suffix detection deleted; all 5 airports' `categories.json` migrated in place.
- Cleanups: record-filename suffix constants; shared `_fake_optimizer` test factory.

### 2026-07-06 — Rollout guard margin (residual low-success root cause)

97% of KRDU/asdb gate-failures were rollouts truncated by the zero-margin ground guard at exactly the floor: min-time plans deliberately ride the floor, and cm-scale integration noise (measured: 3.9 cm dip for 1.5 s on a replay that lands 0.7 m out) tripped a guard meant for km-scale divergence. Fix: `ROLLOUT_GUARD_MARGIN_M = 5.0`, guard = floor − 5, both rollout call sites. (Diagnostic lesson: the eval record's `final_time_s` IS the truncated end — compare against the states-file's plan T to detect truncation.) Same day:

- `_clear_stale_records` deletes stale top-level records at batch start (27 orphan evals inflated a report).
- `--fitting-type {hs,trapezoidal,rk4}` end-to-end (`FITTING_SCHEMES`; default hs). rk4 verdict: fine on smooth constrained solves; basin-fragile on aggressive unconstrained min-time (auto M 9.1 km off; only M=64 recovers HS's optimum at ~4× cost) — HS stays default.
- `--state-substeps M` end-to-end through both solve paths + frontend "State substeps" input → backend `state_substeps` on both branches (cache keys include it). Measurements in Key Defaults.
- `ipopt.max_iter` wired ("backend never finishes" postmortem): constrained M=64 ≈ 640 subintervals × per-node inequality rows, no iter cap, solves serialized behind the worker lock. `DEFAULT_MAX_ITERATIONS = 3000`, `CollocationOptimizer(max_iterations=…)` on all three solver constructions; backend `maxIterations` reaches both branches (make_optimizer had accepted-and-ignored it).
- Stale-artifact audit closed the `write_reference_records` and CZML-builder (`clear_stale_outputs`) accumulation gaps; single-file overwrites audited clean.
- NOTE: `runway_cons` off-target populations likely contain the wrongly-truncated family — re-examine after re-run. All categories need re-running after preparation: `python run_scenario_optimization.py --jobs 6`.

### 2026-07-06 — Observe constrained auto-open + shared Profile toggle + forced-display hook

- Extracted `src/hooks/useForcedProcedureDisplay.ts` from PilotPanel. Contract `{active, forceRunway}`: non-null `forceRunway` → the hook OWNS `selectedRunway` (save-once/force/restore-on-inactive + dependency-free restore-on-unmount); `forceRunway: null` (Observe) → never reads/writes `selectedRunway`, drives only the panel + `procedures` layer.
- Observe trigger in ControlPanel: active = trajectories layer ON ∧ comparison overlay ON ∧ constrained category ∧ runway selected.
- Shared toggle component (one parameterised toggle driving the global open state; opening focuses `selectedRunway` on the governed runway) used in ControlPanel, PilotPanel's Target-State header, and ProcedurePanel.
- **Dock-handoff race**: two forcing docks switching share one React passive flush — the incoming hook read the outgoing dock's still-forced display and saved a polluted baseline. Fixed with a one-render `ready` gate (no-op on first commit; the sibling's restore is batched with the `ready` flip in React 18). Integration regression test with the real AppProvider. Verified in-browser (KRDU 05L).

### 2026-07-06 — Evaluation window symlog label pile-up

Under `symlog(v) = sign·log10(1+|v|)` every |v|<1 collapses to ~0, stacking the ±0.01/±0.1 decade labels; `EvaluationReportWindow.tsx` now skips sub-1 decades (keeps 0, ±1, ±10, …).

### 2026-07-06 — All-modes sweep + unconstrained batch's trapezoidal fitting (low-success root cause)

- Omitting `--target-type` runs all three modes (`asdb`/`runway`/`runway_cons`) per airport; `--with-constraint` without it is rejected.
- Unconstrained success 4–14% vs constrained 76–97%: the batch's `optimize_scenario` still used trapezoidal after the 07-05 HS flip (batch-edition seam class). Min-time solves ride the floor exactly where 2nd-order trapezoidal is dynamically unfaithful: plan on-target, rollout 5–15 km off (full-T, not truncated). A/B (DAL1407): trap 5950 m vs HS 3.4 m, and HS found a better optimum at ~3× solve time. Fixed to HS.

### 2026-07-06 — Arrival-segment truncation: 25 km entry ring; locals excluded

Landing tracks were validated by their END only, so depart-and-return flights started ON the field. New `trajectory_data_process/arrival_segment.py`: walk backward from touchdown; the arrival starts after the LAST run of ≥3 consecutive samples outside the 25 km ring (hysteresis: one jittery fix can't cut); plain arrivals also cropped to the ring so every arrival shares one entry boundary (the on-ring state distribution = interaction-study boundary condition). Never-outside tracks: `local` if start ≤5 km (takeoff→circuit, written to `<ICAO>_local_rejected.json`, never silently dropped), else coverage-limited arrival kept whole. `truncate_flights` rebases times to 0 and annotates `arrival_truncated`/`cut_samples`/`arrival_duration_s`/`entry_time_utc`. Integrated in `build_arrivals.py` (renamed from `landings_to_czml.py`): raw `*_landings.json` untouched; derived `*_arrivals.json` feed CZMLs + czml-input; airport centre from `config/runway_thresholds.json`. `FlightScenario.source` carries `entry_time_utc` (co-temporal placement key). All 5 airports regenerated: KMSY 400, KRDU 996, KSJC 319, KSMF 714, KSTL 1054 arrivals. All pre-existing batch outputs predate this (and the floor fixes) — stale.

### 2026-07-05 — Evaluation detail window: legible profiles + one colour language

`DeviationProfile` ranked-dot charts replace the illegible bar walls (lateral: log axis; vertical: signed symlog); each chart draws AND labels its own gate in-plot, dots coloured by that gate, legend carries the outside count. One colour language: red/green = per-flight gate verdict only (summary cards made neutral; scatter legend explains its colours). Values are the backend report's rows verbatim. Verified in-browser (KRDU runway_cons, 1001 trajectories).

### 2026-07-05 — Off-target marking moved onto the RESULT path

"Successful flights ending mid-air" = off-target flights with guard-truncated rollouts (correctly classified; they merely LOOKED successful because only the reference was marked). Now the simulator/result path bakes `OFF_TARGET_COLOR` yellow + "(off target)" name; reference drops to dark-amber `OFF_TARGET_REF_COLOR`; plan keeps legend orange. `useComparisonTrajectoryLayer` skips its repaint for `properties.status == "offTarget"`. NOTE: KRDU runway 32's 77/200 off-target (vs 6–7 on 23L/R) is a real quality signal — likely the replay-divergence family.

### 2026-07-05 — Below-ground trajectories: altitude floors + rollout ground guard

Three root causes for optimized trajectories below field elevation (per-leg step-down floors verified working — the dives lived where no floor existed):
1. The global altitude bound (`components.altitude_floor_m` = target − 300 m, documented as a never-binding box) BINDS — min-time solves dive to it. Fixed: margin 300 → 5 m, a real operational floor; `min(initial, target)` deliberately NOT used (a start below the floor is bad data and fails loudly).
2. The start→first-fix transition phase had no floor above the global one (FFT2071 dove to −173 m). Fixed: transition altitude bound = min(start alt, `_first_leg_entry_floor_m`) − margin (min() IS needed here — a start below the first fix's altitude is legitimate climb-to-join geometry).
3. The batch rollout had no ground envelope (`CasadiSimulator.step` has no checks): `_GroundCheckedSimulator` + `rollout_controls(min_altitude_m=…)`.
Also: evaluation gates judge the FINAL state only, so mid-flight dives never failed a gate — pre-fix success rates were inflated. All batch outputs stale.

### 2026-07-05 — In-app evaluation report window ("Details")

The comparison builder PUBLISHES the evaluation report verbatim (`publish_evaluation_report` → `comparison/<category>/evaluation_report.json`); the frontend fetches that copy and only formats/sorts/plots — no metric recomputed client-side (new metrics go in `evaluation/metrics.py`; contract documented in `src/data/evaluationReport.ts`). `EvaluationReportWindow.tsx`: draggable floating window (portal, same shell as Dynamics-Comparison) with summary cards, 8260.58D gates note (values from `report.thresholds`), aggregates table, three SVG charts, full verdict table. `OptimizationSummary` Details button; report fetched lazily, cached per (airport, category); missing report shows a helpful message. Verified in-browser (KSMF runway_cons).

### 2026-07-05 — Comparison CZML: off-target status + evaluation metrics into the frontend

Builder `--evaluation-report`: verdicts keyed by eval filename, joined to summary rows via `eval_file`; solved-but-failed-gates → status `offTarget` with `lateralErrM`/`verticalErrM` on the index record; `optimization_stats(summary, report)` builds the index `optimization` block (successful/successRate/avgStateErrorM/avgTimeS — nothing recomputed). No report → byte-compatible plain behavior. Runner: the (cheap) evaluation report now always runs before the tails; the CZML step passes it; reusing a pre-evaluation optimization skips evaluation with a loud note. Frontend: `ComparisonGroup.status = "solved"|"offTarget"|"failed"`; flight list flags off-target yellow (`.flight-table-offtarget`, #ffcd28); `OptimizationSummary` metrics were already wired.

### 2026-07-05 — Panel "final horiz err" ≠ playbackDrift, round 2 (verified in-browser)

Two stacked causes behind Δ25 m vs playbackDrift 0.6 m: (1) the LOOP_STOP wrap heuristic never fired (Cesium preserves overshoot on wrap) — replaced all elapsed-based heuristics with `clock.onStop`; `makeReadoutEmitter` returns `{tick, stop}` (stop = throttle-bypassed exact-`stopS` emit, deduped), both playback hooks subscribe to onTick + onStop; Reset/backward scrubs never raise onStop. (2) `czml_common.document_packet` truncated the clock interval end to whole seconds — now `iso_ms` (74.7 m/s × 0.338 s ≈ the phantom 25 m). Backend audited clean otherwise (`playbackDriftM` and the CZML doc end read the same terminal sample).

### 2026-07-05 — Constrained-IAF batch 0% solve rate (two stacked bugs)

1. `scenario_optimization._solve_iaf` still unpacked `segments, _spans = build_constraint_segments(...)` after the 07-03 change to a plain list return — every constrained batch since 07-03 was broken (backend HTTP path updated, batch caller missed). Seam regression test added.
2. The batch's threshold target (from `config/runway_thresholds.json`) sat up to 390 m from the procedure's CIFP threshold → frame-anchor guard fired. `_snap_target_to_procedure` snaps the solve target horizontally onto the procedure's last waypoint (keeps altitude/Vref/pavement heading/glidepath); `_iaf_result` writes the eval record against the SNAPPED target; reference records keep the scenario target. (Diagnostic note: the batch log truncates errors to exception TYPE names per IAF — reproduce one scenario for a real traceback.)

### 2026-07-05 — Evaluation review fixes (degenerate records + HTML escaping)

- One 1-sample "solved" record (rollout truncated at its first step) aborted the whole batch report. Fixed at both ends: producer `_require_usable_rollout` (<2 samples → recorded as a FAILED scenario); evaluation guards via shared `reference.horizontal_arc_length_m` — comparison skipped with a row note, visualize drops undrawable polylines.
- HTML: embedded JSON escapes `</` as `<\/`; `esc()` on every data-derived string reaching innerHTML/Plotly.
- Contract: `final_time_s == states[-1].t` required on solved records; `resample_by_arc_length` rejects n<2; chart labels use unique record file basenames.

### 2026-07-05 — Former combined scenario pipeline runner

The comparison + evaluation runners merged at the time. They duplicated the expensive steps and wrote divergent opt_dir contents (the comparison runner silently overwrote `reference_file` pointers away). Optimization always runs with `--reference-tracks`; tails remain selectable via `--outputs czml,eval` (default both); `--skip-optimize` reuses an existing summary.json. The combined preparation/optimization entry point described here was superseded by the 2026-07-23 split above.

### 2026-07-05 — `evaluation` package (regulation-derived gates)

New root package `evaluation/` — the file-based seam at the end of the pipeline (geokit + stdlib only, never imports the optimizer). Record contract + gates: see Key Defaults. Salient points:
- `evaluation_export.py` maps the true-dynamics rollout onto the contract — `rollout_piecewise_constant`'s samples already carry the active control per sample (nothing re-derived). Both batch modes roll out once and write `*_eval.json` next to every `*_states.json`, INCLUDING failed scenarios (empty record = how solve rate is computed from files alone); summary rows carry `eval_file`. Old `*_states.json` can't be converted offline (no controls) — re-run for eval records.
- Reference records: observed track in the same contract (`flight_scenarios.state_samples_from_track`, times rebased, same target; `controls == []`). `write_reference_records` looks flights up by full identity `(id, icao24, landing_time_utc)` (missing ⇒ raise) → `references/<identity>_reference_eval.json`; `reference_file` stamped on every eval record, failed included. `compare_to_reference` = flight-time delta + path-shape deviation (arc-length resampling — time-matching would conflate speed profiles with geometry).
- CLIs: `python -m evaluation --input <dir> --output report.json` (+ threshold flags); `python -m evaluation.visualize` renders one self-contained HTML (Plotly CDN + embedded DATA; English body, abbreviations expanded on use; 8260.58D citations in the gate-sources note; `--max-tracks` cap stated on the page). Verified in-browser.

### 2026-07-05 — ψ corridor kills the looping/crawling pathology; drift guard; HS default

- Postmortem (nsp=2 loops): the solve CONVERGED onto a local optimum with ±2π winding (feasible — the join ψ box pins one node; excursions cancel), and the node-pinned terminal masked a 4.5–4.9 km true-dynamics rollout drift (trapezoidal on a winding path is node-feasible but dynamically meaningless).
- **ψ corridor** (structural fix): constrained heading variable bounds tightened from ±3π to the route heading hull ± 90° — winding optima cease to exist. The whole 07-04 family of crawls/Max_Iterations/basin-twitchiness died: 2×2 (HEAVE/custom × trap/HS) 4/4 clean, nsp 2/3/4/6 all converge.
- **Drift guard**: `playback_terminal_drift_m` → `playbackDriftM` on every response + stderr WARNING > 50 m. Never touches the NLP.
- Fitting verdict (doglegged H05LZ): trapezoidal 226–296 m drift vs HS 0.6–0.9 m at every nsp; HS ≈ 2–2.5× solve time. **Constrained default flipped to HS** (`_DEFAULT_CONSTRAINED_SCHEME = hermiteSimpsonNormalizedFullTransport`); trapezoidal stays selectable.
- Readout artifact (wall-throttled ~12 Hz emitter missing the terminal sample by up to throttle×multiplier sim-seconds) fixed via `makeReadoutEmitter`, shared by the optimized-playback and Compare hooks — later superseded by the onStop rework above.

### 2026-07-04 — Join constraints: pre-FAF fix passage + flexible FAC intercept + alignment tiers

(Current semantics in Key Defaults.) Highlights and lessons:
- ⑪ Fix passage evolved to the PRE-FAF fix only (all-fixes and entry-fix variants superseded, per user); enforced as a smooth squared form rescaled by `1/(2·tol)` so violations read in metres (no |·| kink); tolerance = the leg's k·RNP halfwidth, procedure-sourced.
- ⑫ Flexible FAC intercept replaced the exact-FAF pin: linear cross-track equality + upstream-only window ≥ L_final/5 before the FAF; `fac_distance_to_ltp` is THE one distance measure (glidepath d reuses it). Vertical semantics stay published-geography-keyed (`LpvFinalSpec.d_faf_m`/`prefaf_floor_m`; both None → gate off, byte-identical).
- Two 2π-branch bugs found by evaluating `nlp.g` at failed iterates and ranking violated rows: the terminal ψ pinned the wrong branch on double-90° routes (fix: `_route_unwrapped_target_psi` walks chained leg courses), and the join intercept `cos(ψ−course) ≥ cos30°` was 2π-periodic (fix: branch-aware linear box). Plus the duration-split regularizer for the flat time-split direction.
- Constraint families refactored into explicit per-family row functions behind a dispatcher (uniform `list[(expr, lb, ub)]`); `components.unwrap_angle()` collapses four unwrap copies. Two-tier FAC heading alignment added (positions-only corridors let large heading errors fit between nodes).
- `log_optimizer_config` writes one stderr line before any solving (optimizer/scheme/fitting/dynamics/transport/constrained).
- Frontend follow-up: custom starts KEEP the RNAV IF selection (the selector names the procedure; the start is independent) so constrained solves can start off-fix; transition threshold unified with the passage tolerance (`_first_fix_join_tolerance_m`) — the old 1–2 km dead zone (no transition, no disc) removed. Verified end-to-end on H05LZ HEAVE with a custom start 9.2 km out.
- KNOWN DATA GAP: per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (926 m at k=0.5) instead of ~278 m (RNP 0.3).

### 2026-07-03 — CIFP thresholds everywhere; displaced-threshold root cause

- Root cause of up-to-970 m target gaps (KSJC 30R): runway.geojson `runway_surface` edges are PAVEMENT ends while CIFP's threshold is the DISPLACED landing threshold, plus a real `build_runway_ring` bug (declared length re-centred on the endpoint midpoint with asymmetric displaced offsets → rigid shift by `(he_disp − le_disp)/2`). OurAirports coordinates themselves verified accurate.
- Constrained target anchored on CIFP: `procedureThresholdAnchor(constraint, document)` — position = the constraint's last waypoint, altitude = CIFP threshold elevation, ψ = final course in the simulator convention; `PilotPanel.computeTrajectory` overrides the request target whenever a `procedureConstraint` is attached. runway.geojson still drives the unconstrained target and rendering.
- `procedure-details/index.json` hoists each runway's CIFP `threshold` `{lon, lat, elevationFt}` (null when uncoded); unconstrained targets prefer it too (`buildRunwayThresholdTargets(collection, index?)`; heading stays pavement-derived). CIFP is also more complete than OA (e.g. KSTL 30L's displacement missing from OA).
- `build_runway_ring` rewritten: `runway_surface` corners ON the OA endpoints; `landing_zone` ends moved inward by the displaced distance. All 5 airports regenerated; landing_zone matches CIFP ≤ 24 m except pure OA data gaps.

### 2026-07-03 — approach_constraints/collocation review fixes

- **One ψ convention**: `course_bearing` now returns the model convention (`atan2(Δn, Δe)`); the optimizer consumes it instead of re-deriving course math (guiding rule: constraint/course math has ONE source — `approach_constraints`).
- `ConstraintReport` unit-aware: metre and radian violations separated (`max_violation()` / `max_angular_violation()`, `is_feasible(tol_m, tol_rad)`) — 1 rad had counted as "1 m", a real false-feasible.
- Transition phase actually built: `_phase_plan` prepends the unconstrained start→first-fix phase (start > 2 km from the first leg's start fix); an approach whose first leg IS the final gets the FAF intercept on the transition phase. `build_constraint_segments` returns a plain list.
- Frame-anchor contract validated loudly; terminal-bank 1-node-phase latent bug fixed (`phase_starts`); per-phase auto state substeps (~3 s target; M-selection deliberately ignores `fixed_duration` so fixed- and free-time NLPs share one decision layout, the fixed solve seeding the free one); glidepath d measured on the same GARP→LTP axis as the lateral corridor; normalized position box 1e7→2e6 m; dead machinery removed (`partition_node_indices`, `LpvFinalSpec.da_hat_m`); README de-staled; mass-frozen approximation stated; HS docstring corrected (4th-order, O(h⁵) local).

### (undated, ~2026-06-28) — Backend SIGABRT root cause + solver isolation

The service "shut itself down": casadi's thread-unsafe symbolic construction under `ThreadingHTTPServer` (three concurrent NLP builds) corrupted the heap → SIGABRT; the old launcher then co-killed the frontend.
- `isolated_backend.py` `IsolatedRunner`: casadi-heavy endpoints (`/optimization/run`, `/dynamics-comparison/run`) run in a worker subprocess (`ProcessPoolExecutor(max_workers=1, mp_context="spawn")`); a native abort → `BrokenProcessPool` → clean `SolverCrashError` 500, next request spawns a fresh worker (self-heal). Decorator backends keep the same method interface (tests inject in-process fakes).
- Memory-aware worker lifecycle: `POST /optimization/session/{open,close}` (+ dynamics-comparison twins) ref-count ONE resident warm worker tied to the frontend tab (spawn costs ~1.2 s/call otherwise); no session → ephemeral worker per call; idle watchdog (`AEROVIZ_WORKER_IDLE_TIMEOUT_S`, 600 s) reclaims stranded workers. Frontend `workerSessionClient.ts` opens/closes per Pilot sub-mode + `navigator.sendBeacon` close on `pagehide`; all best-effort, the watchdog is the backstop.
- `casadi_lock.CASADI_LOCK` (RLock) serializes every in-process casadi entry point incl. `SimulationBackend.reset/step` (previously unlocked).
- `start_aeroviz_fullstack.sh` rewritten as the supervisor (see Gotchas). Spawn-picklable probes live in `isolation_probes.py`.

### 2026-06-28 — Scenario→optimization→CZML pipeline (scaffolds, since filled) + `flight_scenarios` seam

- `scenario_optimization.py` writes one `*_states.json` per scenario = `{source, final_time_s, optimizer_states[], simulator_states[]}` (states `{t,lat,lon,alt,V,psi,gamma,m}`); `optimizer_states` = NLP node states, `simulator_states` = controls rolled through `CasadiSimulator` (lives in the `aerodynamic_model` layer — never imports the backend above it). `build_scenario_comparison_czml.py` renders reference/optimizer/simulator as three coloured time-dynamic paths.
- `flight_scenarios` package: `scenario.py` (record + JSON round-trip + `aircraft_for_code`), `start_state.py` (track → initial state via two-sample finite difference), `build.py`, CLI `python -m flight_scenarios`. `final_state_from_track` populates `FlightScenario.target`.

### 2026-06-28 / 2026-06-27 — `geokit`: one geodesy/units source

- `geokit.constants`: WGS84 (`WGS84_A`/`_E2`/`_B`), `SPHERE_RADIUS_M` (default WGS84 a; switchable `EARTH_RADIUS_MEAN_M`), `NM_M`/`FT_M`/`KT_MS`/`METRES_PER_DEG_LAT`/`DEG2RAD`. `geokit.geodesy`: haversine, equirectangular, bearing, flat-distance, metres-per-degree, bounds. `geokit.units`: exact speed (`kt_to_ms` = 1852/3600, ft/min, km/h, mph) + length (`nm_to_m`, `ft_to_m`) conversions — replaced the truncated `0.51444` and 4 divergent Earth radii project-wide; `aircraft_sets.py` SI-mirror fields derive from it.
- Frontend: `geokit/scripts/export_constants_json.py` → `src/generated/geoConstants.json`, re-exported by `procedureGeoMath.ts` (~16 files migrated off local constants); drift-guard test fails if the JSON drifts. Fixed a real bug: two TS modules had used different haversine radii.
- Aero layer imports constants only (symbolic functions untouched). The 30 km study was regenerated for the ~0.1% WGS84-a shift (conclusions unchanged). Pedagogical flat-Earth `runway_bearing_rad` stays local by design.

### 2026-06-27 — Full (exact) geodetic transport as an explicit option

The geodetic RHS's ψ transport had silently dropped a cross term `V·sinγ·sinψ·cosψ·(1/(R_N+h) − 1/(R_M+h))` (~3–4 orders below the main meridian-convergence term; γ transport was already exact). Now `transport ∈ {"none","approx","full"}` on `make_geodetic_dynamics_model`/`make_geodetic_step_integrator`; `"approx"` = historical default (byte-identical). New `*FullTransport` (+ Normalized) schemes end-to-end (backend names `casadiDirectCollocationNormalizedFullTransport(+Trapezoidal/+Rk4)` etc.; frontend Dynamics options). Compare mode gains opt-in system F (full transport). `transport_term_comparison.py` + zh doc: divergence ~mm over 120 s, dt-independent (vector-field, not truncation). Decision: default stays approx; no silent approximations going forward.

### 2026-06-25 / 2026-06-24 — RNAV(GPS) tutorials + canonical ProcedureConstraint + CIFP block-altitude fix

- Tutorials: `aeroviz-4d/docs/34-how-to-read-rnav-gps-approach.{zh,en}.html` — self-contained, interactive (auto-wrapped glossary tooltips, SVG fix/segment hotspots, slide-in glossary panel), worked on KRDU RNAV (GPS) Y RWY 5L. Verified in-browser.
- **Canonical `ProcedureConstraint`** (front↔back): `src/data/procedureConstraint.ts` + Python mirror `aeroviz_backend/procedure_constraint.py` — one JSON shape (ordered waypoints with altitude windows + final course + glidepath + nominal speed); `buildProcedureConstraint(document, {branchId})`. CIFP→`AltitudeConstraint` conversion unified into one `altitudeConstraintFromCifp` (the two copies had diverged — one dropped block upper bounds).
- **CIFP block-altitude fix**: ARINC 424 "B" descriptor is a WINDOW (at-or-below Alt1, at-or-above Alt2); the parser had dropped Alt2. `ProcedureLeg.altitude_ft_2` added; KSTL regenerated. Chart-cross-referenced golden test (`test_krdu_r05ly_matches_published_rnav_gps_chart`) guards the parser against published-chart values. Documented gap: leg speed restrictions not extracted (cifparse exposes no speed field; 0 coded in the dataset) — `speedMaxKt` is ready when a source appears.

### 2026-06-24 / 2026-06-23 — Normalized geodetic scheme (conditioning fix)

- Root cause of Max_Iterations on H05LZ N=10 free-time solves: conditioning, not the seed — radian lat/lon (~1e-6 rad/s derivatives) next to metre/m-s states; the `1/(R_M+h)` factor makes position defect rows ~6–7 orders smaller than altitude rows.
- Fix: `*Normalized` schemes reparameterise the decision state to metres from the target anchor (`n=(lat−lat_t)·R`, `e=(lon−lon_t)·R·cos(lat_t)`) — an EXACT affine change of variables (unlike localEnu's flat-tangent approximation); same geodetic RHS inside the defect. Robust across N and arrival windows; identical trajectories on benign problems. The localEnu cold-start hybrid (an earlier workaround) was removed as superseded. Tutorial: `4dTrajectory/docs/geodetic_state_normalization.zh.md`. Compare mode gains system N (normalized) which overlays C — live proof the reparameterisation changes nothing.

### 2026-06-23 — Dynamics Compare mode (Pilot panel) + follow-ups

Third Pilot-panel mode: flies the start state under one constant control as the study's systems — A fixed-tangent ENU (anchored at the START in Compare mode; the 30 km study anchors at the target), B per-step re-anchored (reference), C geodetic RHS +transport, D no-transport, opt-in N (normalized) and F (full transport) — replayed as coloured, hideable CZML paths on Cesium's clock with deviation charts (horiz/alt/head/speed/fpa vs B) and a final-value table.
- Core extracted to `dynamics_comparison.py` `compare_dynamics(...)` (30 km study now calls it, output byte-identical). Endpoint `POST /dynamics-comparison/run` (`dynamics_comparison_backend.py`); shared `czml_common.py` (epoch/iso/document-packet) + `responseValidators.ts` back both backends/clients.
- Frontend: `dynamicsComparisonClient.ts`, `useDynamicsComparisonPlayback.ts` (loads CZML, drives the clock, hides systems, camera-follow), `DynamicsComparisonCharts.tsx` (draggable portal window — see the backdrop-filter gotcha). Per-system tinted aircraft models (`colorBlendMode: MIX`) oriented via `VelocityOrientationProperty`, wrapped in `makeStableVelocityOrientation` (a CallbackProperty returning the last valid orientation when HOLD extrapolation zeroes the velocity — otherwise the parked model snaps to a default attitude).
- Live State panel shows B's state (dense backend `samples` in the trajectory-play shape) + colored per-system delta chips interpolated from the chart (`interpolateComparisonDeltas`); `fpa` tracked end-to-end as an error metric. Trajectory Play reuses the same chip strip for a live Δ-vs-target readout (replaced the old Lat/Lon/Alt Error rows).
- Custom start state in Compare (RNAV fixes + runway select); run history persisted per run (`dynamics_comparison_history.py`, git-ignored dir) with backend-averaged history endpoints (common distance grid, shortest run's range) + frontend Average/Clear buttons.
- Review fixes: rollout never records sub-surface/non-finite samples (stops + truncation note via `requestedDurationS`); endpoint-inclusive `even_sample_indices`; double-checked-locked integrator cache; chart memoization; START preview hidden during comparison playback.

### 2026-06-23 — Optimizer = dynamics × fitting; shared stall model

- `_DEFECT_SCHEMES` entries are `(make_dynamics, make_defect)`: localEnu is a CONTINUOUS dynamics (fixed ENU tangent frame) collocatable with any fitting (defect converts geodetic nodes into the target-anchored ENU frame via `geodetic_state_to_enu_expr`); only `reanchoredEnu` stays shooting-only (per-step re-anchoring is discrete).
- Shared stall model `aero_params_for_aircraft(aircraft)` (mass-based Cl_max, A320 ≈ 2.7) used by optimizer AND playback — they had diverged (2.7 vs 1.5), replaying optimized trajectories ~1.6 km off.
- Cold-start hybrid (`cold_start_scheme`: fixed-time seed solved with a cheaper dynamics, free-time refines) was added with a whole-flow timing stderr line (`log_optimization_timing`: build/coldStart/freeTime/solve/playback/total) — the hybrid was later removed (superseded by normalization); the timing log remains.

### 2026-06-22 — Pluggable defect schemes; solver backend verdict; CZML playback; dense state

- **Defect schemes**: trapezoidal (order 2) / Hermite-Simpson (order 4, default) / RK4 (order 4, shooting) on the continuous geodetic RHS + `reanchoredEnu` (the playback integrator as a shooting defect). HS and RK4 are the same order — they differ in construction (implicit collocation vs explicit shooting). A stepper can be a shooting defect but NOT a polynomial collocation defect. `collocation_scheme_comparison.py` accuracy ladder: trapezoidal ~5 m vs HS/RK4 sub-metre. Frontend split the optimizer choice into Dynamics × Fitting dropdowns (`optimizerToParts`/`partsToOptimizer`/`validFittingsForDynamics`); legacy optimizer names remain valid on the backend. Tutorial `4dTrajectory/docs/collocation_schemes.zh.html` (interactive convergence demos).
- **`localEnu` scheme + 30 km study** (`dynamics_comparison_30km.py` + zh doc): fixed local-ENU @ target ≈ 335 m horiz error over 30 km, RHS-no-transport ≈ 145 m, full RHS +transport ≈ 0.03 m (validates RHS ≡ re-anchored). `make_local_enu_step_integrator(ref_geo)` reduces exactly to the re-anchored stepper when ref = current point.
- **Solver backend**: `solver_backend` switch (ipopt/sqpmethod) exists but sqpmethod is NOT usable — cold it bails instantly from linear-interp guesses; warm-started (needs duals, exact Hessian) it's still slower than cold IPOPT because CasADi's sqpmethod uses a dense active-set QP (~300× per-iteration cost, no banded OCP structure). IPOPT stays the only exposed backend; a real warm-start payoff would need acados/HPIPM. `solver_backend_benchmark.py` documents this.
- **Playback**: the optimized trajectory plays as backend-built CZML on Cesium's clock (`trajectory_playback.build_optimized_trajectory_playback` — rolls the N piecewise-constant controls once through the SAME geodetic integrator as the live sim, sub-mm match). Trail = one short polyline per sample interval with ms-precision availability (grows behind the aircraft), coloured by control segment (blue→red). The aircraft packet carries NO orientation — the frontend sets it from the sampled state with the live-Pilot convention (`headingPitchRollQuaternion`, heading −ψ, pitch γ+α, roll −μ). `useOptimizedTrajectoryPlayback.ts` drives the clock + throttled ~12 Hz readout sampling (`sampleTrajectoryAt`). Rollout truncates (not raises) on envelope exit — it's a viz aid. Manual Pilot mode unchanged (interactive).
- **Dense-state collocation**: control on N segments, state collocated on N·M sub-intervals (`sub_steps`; auto ~3 s, cap 16) — fixed the km-scale optimizer→playback mismatch (coarse discrete operator ≠ fine playback RK4). The multiple-shooting polish machinery was REMOVED (dense-state raw solutions are playback-consistent). Raising nSegments instead would refine control too → the "wrinkle" convergence pathology.
- **CIFP transition-altitude misparse**: initial fixes with no published crossing altitude were placed at the procedure-wide Transition Altitude (18000 ft) → infeasible starts. Parser no longer falls back to `trans_alt`; qualifier from `alt_desc`; an IF with no own altitude derives one by interpolating from the nearest published fix on the branch (`derivedInitialFixAltitudeFt`) rather than being dropped. Postmortem: `aeroviz-4d/docs/33-cifp-transition-altitude-misparse-postmortem.md`. (The ready-made cifparse/arinc424 packages had the same fallback bug.)

### 2026-06-21 — Geodetic continuous dynamics for direct collocation

Replaced the fixed-ENU transcription with one continuous geodetic RHS shared by optimizer and playback: `make_geodetic_dynamics_model` (point-mass RHS in `(lat, lon, h, V, psi, gamma)` radians; position kinematics via WGS84 `R_M`/`R_N`; transport terms on ψ̇/γ̇) + `make_geodetic_step_integrator` (RK4, degrees externally). Validation (`geodetic_vs_reanchored_error.py`): geodetic+transport tracks the re-anchored RK4 playback to ~0.3 mm over 5 km; without transport ~2.9 m drift. Interactive doc `geodetic_dynamics_transport.zh.html`.

### 2026-04-20 — OCS geometry + FAA DOF obstacle layer

- `ocsGeometry.ts` `buildFinalApproachOCS` implemented (primary trapezoid + two 7:1-slope secondary panels); `useOcsLayer` renders three semi-transparent polygons per route (FAF→threshold pairs from `procedures.geojson`; primary half-width from the route's tunnel descriptor, 150 m fallback); `ocsSurfaces` layer toggle. Altitudes read from LineString z-values (CIFP geometry alt); switching to MCA is a one-function change documented in `docs/03-ocs-geometry.zh.md` §5.6.
- `preprocess_obstacles.py` parses fixed-width DOF `.Dat` (haversine radius filter, default 20 km) → `obstacles.geojson`; `useObstacleLayer` renders type-coloured cylinders (`RELATIVE_TO_GROUND`) with AGL labels; `obstacles` layer toggle.

### 2026-04-19 — DSM terrain hook

`useDsmTerrainLayer` rewritten onto the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`); returns `{status, metadata, provider, error}`; wired in `CesiumViewer` + demo page; `dsmTerrain` layer toggle.
