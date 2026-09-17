# evaluation — gates, gotchas, observed-data grading, streaming, precision (reference)

Full text behind the index lines of `evaluation/CLAUDE.md`, moved here VERBATIM on 2026-09-16 so that
file stays a short index (it is injected into every session that touches the tree). Only the
`### <ID>` headings and the dated **Correction / Note** paragraphs are new; every heading has
exactly one index line in that CLAUDE.md ending in its ID (`grep -n '^### # ·' evaluation/docs/EVALUATION_REFERENCE.md`).

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.**

## Gates & batch metrics (`evaluation/thresholds.py`, `evaluation/speed_gate.py`)

### EV1 · lateral = half the published runway width

- **Lateral = half the published runway width, and that is the whole rule**
  (`LATERAL_CRITERION_ID = "runway_half_width_at_threshold"`, 15.24–22.86 m across this fleet).
  It is a LANDING-GEOMETRY claim — did the crossing lie over the pavement — NOT a
  navigation-containment one; `METHODOLOGY["terminal_lateral"]` states that inside every report.

### EV2 · vertical ∈ [−22, +22] m; LPV and LNAV/VNAV

- **Vertical ∈ [−22, +22] m** from the published LTP+TCH path (ICAO Doc 9613 Vol II Pt C Ch5
  §5.3.4.4.7, RNP APCH Baro-VNAV FAS), available for LPV always and for the LNAV/VNAV benchmark
  when the runway's own RNAV (GPS) approach publishes LNAV/VNAV minima AND a runway-leg path
  (`Runway.baro_vnav_minima`, `tch_source == "faa_cifp_approach_leg"` — KRDU 32 at 3.50°,
  KSMF 35R; `assessment_for_runway` derives `baro_vnav_approved` from that, there is no
  caller flag). A runway with no RNAV procedure at all (KRDU 14) keeps lateral and grades
  vertical indeterminate with `procedure_source == "no_published_vertical_guidance"`. Gates
  judge the TRUE-dynamics rollout's final state.

### EV3 · speed gate anchored on the published approach speed (v9)

- **Speed ∈ [V_ref,lo·√max(n,1), V_ref,hi + 20 kt] at the crossing, per record, anchored
  on the type's PUBLISHED approach speed** (v9, 2026-09-07; `evaluation/speed_gate.py`;
  design, sources and the measured results in `docs/THRESHOLD_SPEED_GATE.md`).
  `V_ref` is the FAA Aircraft Characteristics Database (October 2024) approach speed at
  MALW per ICAO type — lo/hi = the FSB's dual flap-configuration values where given —
  scaled by `√(m/MALW)`: computed records at their crossing mass (`mass_basis =
  crossing_mass`), observed records over the type's published mass range [minimum
  operating mass, MALW] (`type_mass_range`; an ADS-B track carries no mass). The table
  is `aircraft/reference_speeds.json` (`aircraft.reference_speeds`), every number with a
  source id; the documents are downloaded under `data/reference_speeds/` and indexed in
  `docs/reference_speeds/README.md` (URL, retrieval date, SHA-256, page). **Adding a
  type is a cited JSON row, never a code change.** The record's type comes from
  `source.dynamics_typecode` (computed: the type the model FLEW, which is the fallback
  type when the identity's type has no dynamics — the row's `reference_typecode` says
  which) or `source.aircraft_type` (observed: the resolved airframe). Only the lower
  edge moves with n (`controls[-1].load_factor`; MEASURED from ADS-B kinematics on
  observed baselines, `adsb_kinematics`; declared 1 g otherwise). +20 kt = FSF ALAR BN
  7.1. **Composed into the verdict for ALL subjects (owner decision 2026-08-24).**
  Computed subjects are judged on the crossing model airspeed (`crossing_speed_ms`);
  observed baselines on the fitted crossing GROUND speed **corrected by the field's
  METAR headwind** (`evaluation/wind.py`, `data/metar/<ICAO>/`, CLI `--metar-root`)
  into `crossing_airspeed_estimate_ms` (criterion `…_metar_airspeed_estimate`) — or,
  when no report within 30 min is usable, on the raw ground speed as a STATED PROXY
  (`…_ground_speed_proxy`; `wind.status` on the row says why, `wind_counts` on the
  batch). **The verdict is pass/fail against a determinate window** (owner decision
  2026-09-07: no probabilistic or three-valued verdict; v8's `speed_uncertainty_ms` /
  `speed_marginal` are gone); `speed_margin_ms` (signed distance to the nearest bound)
  stays as a plain number; `bounds.reference_sources` carries the table's source ids so
  each row traces to its documents. `indeterminate` only when nothing can be judged —
  no crossing, no type on the record, a type without a published entry, an observed
  row whose type publishes no minimum mass, no crossing speed — with the reason on the
  row (`speed_reason`, written even when the composite is a lateral fail) and counted
  in `speed_indeterminate_reasons` over the same rows as `speed_result_counts`. Never compare observed and computed speed rates as one
  quantity. `source.landing_aero` (the v6–v8 stall inputs) is no longer read or
  validated; the optimizer floor (1.10·Vs on the project stall model) sits BELOW the
  gate on purpose — floor-riding and observed-speed-target solves can legitimately fail
  speed; quote speed rates per `target_source`. `crossing_speed_ms` (airspeed) stays
  null on observed rows — the two speeds live in two fields and are never mixed; both
  are FLAT on the row (the frontend verdict table reads them there).

### EV4 · batch metrics

- Batch metrics: solve/success rates, lateral mean/p95/max, vertical spreads, flight times;
  path-shape deviation vs reference = both paths resampled at 101 fractions of their own
  horizontal arc length.

### EV5 · the read side is manifest-only

- Read side is **manifest-ONLY**: `load_records` reads a batch dir via `summary.json` roster
  (`results[].eval_file`); manifest-less dir / listed-missing file / empty roster raise (no glob
  fallback — globbing counted orphans). `categories.json` entries carry an explicit
  `"constrained": bool` (frontend validator REQUIRES it; `_cons`-suffix detection deleted).

## Gotchas (recurring, verified)

### EV6 · a bound that never binds is worse than no bound

- **A navigation-standard bound that never binds is worse than no bound at all.**
  `AssessmentContext.limits()` used to compute `min(guidance, runway_width/2)` where guidance was
  the LPV course width or the LNAV 0.15 NM allowance. Measured over all 26 thresholds at the five
  airports, the guidance term bound **0 times** — those widths are 2.3–18× the runway. Worse, the
  LPV branch divided 106.75 m by 2, and 106.75 m is ALREADY the semiwidth (centreline →
  full-scale deflection, FAA Formula 3-1-1), so every report published `guidance_lateral_m` wrong
  by 2×, unnoticed *because* it was inert. Now there is one bound (`ResolvedLimits.lateral_m` =
  runway half-width) and `lpv_course_width_m` is carried as procedure provenance only.
  **Corollary for any future bound: if it cannot change an answer on the real fleet, delete it or
  the reader will assume it did.**

### EV7 · the assessment context, not `target_state`, says where the runway is

- **A record's `target_state` is not authority on where the runway is — the assessment context
  is.** `_computed_arrival` used to build its `RunwayFrame` at the record's own `target_state`
  lat/lon while cross-checking only the ALTITUDE against the published LTP+TCH. A
  fitted-ADS-B-target record was therefore graded against its own flight's fitted crossing and
  scored ~0 lateral deviation by construction; a target displaced from the published threshold
  (the 775 m KSJC class of bug) produced clean numbers and no symptom. The frame origin is now
  `context.threshold_lat/lon` unconditionally, and both coordinates are cross-checked — but only
  when `source.target_source == "runway_threshold"`, since `fitted_adsb_crossing` and `track_end`
  (both live in `run_scenario_optimization.ALL_MODES`) aim elsewhere on purpose. A record that
  declares no `target_source` gets the STRICT reading, never a bypass.

### EV8 · the report schema version has four homes

- **The evaluation report's schema version has FOUR homes and a green test suite proves nothing
  about them.** Bumping `evaluation.metrics.REPORT_SCHEMA_VERSION` v4→v5 landed in the producer
  only; `4dTrajectory/ts_transformer/data/lateral_eligibility.py` and
  `aeroviz-4d/src/data/evaluationReport.ts` kept their own v4 literals, so the ts pipeline raised
  on every regenerated report and the frontend's `isEvaluationReport` rejected them — surfacing
  `"evaluation report is malformed"` (`EvaluationSummary.tsx`), which is a MISLEADING message:
  the report is well formed, it is the version that disagrees. Both suites stayed green because
  their fixtures also carried the v4 literal: **a version pinned in the test is a version the
  test cannot check.** Now: the ts seam **imports** `REPORT_SCHEMA_VERSION` (it is the one module
  allowed to know evaluation policy), the frontend exports `EVALUATION_REPORT_SCHEMA_VERSION` as
  a declared MUST-match mirror, and every fixture on both sides imports the constant instead of
  repeating the string. Same trap, same fix, for `thresholds.CONTEXT_SCHEMA_VERSION` (now
  `terminal-assessment-context-v2` after the payload gained `threshold_lat/lon`).
  **General rule: when a schema literal appears in a consumer, it is a mirror — either import it
  or comment it as a mirror, and never let a fixture restate it.**

## Judging observed (ADS-B) data

### EV9 · observed grading is the same state interpolation as computed grading

- **Observed grading is the SAME state interpolation as computed grading (2026-08-24).**
  The record's `source.crossing_span` says where its crossing lives
  (`final_approach.crossing`): a `measured_bracket` marker interpolates the marked
  state pair at the event's own fraction; a `fitted_tail` marker points at the one
  inferred crossing row the producer appended (MSL, timed by a trapezoidal
  ground-speed estimate — audit metadata, nothing grades on its clock).
  `_observed_arrival` no longer reads the event's geometry: the event stays for
  identity/staleness validation, the datum cross-check, the audit copy, and
  `crossing_ground_speed_ms`. **`final_time_s` stays pinned to the last MEASURED
  row** (span-aware contract in `records.py`), so flight-time and Δt-vs-observed
  statistics did not move; `TrajectoryRecord.measured_states` is what path-shape
  and span comparisons run over (an appended row is a modeling boundary, not flown
  trajectory). An estimated event on a record WITHOUT a span is a stale artifact
  and raises with its cure (`--evaluate-only`); a record with no event at all
  (optimizer/ts references) still reads `unavailable` → indeterminate. The two
  retained subject branches are POLICY, not plumbing: missing-crossing outcome
  (reception gap ≠ model shortfall) and the speed-gate scope (observed V is ground
  speed).

**Correction (2026-09-18):** that `ValueError` now names `--observed-only` (the rebuild that
leaves `arrivals/` and the lateral roster alone); it said `--evaluate-only` when this text was
written — follow-up #41, fixed.

### EV10 · an observed track's `states[-1]` is not its arrival

- **An observed track's `states[-1]` is NOT its arrival at the target.** A solve terminates at
  its target by construction; a harvested arrival is a truncated recording of a flight that
  continues — 966/996 KRDU tracks end a median **325 m short** of the threshold, still airborne.
  Grading observed data on final-state deviation measures where ADS-B coverage stopped. The
  meaningful observed metric fits each flight's own established final-approach line and
  extrapolates to the threshold (which also validates itself: the fitted glidepath comes out
  3.02–3.13° at all five airports).

### EV11 · ADS-B altitude is quantised to 25 ft

- **Observed ADS-B altitude is quantised to 25 ft = 7.62 m, which is 83 % of the whole 9.15 m
  vertical gate.** All 482 distinct altitudes in the KRDU set lie on that lattice; 54.8 % of
  consecutive samples report an IDENTICAL raw altitude (the aircraft crosses one step every ~2
  samples at 3.81 m of descent per 1 Hz sample). Consequences: a single sample carries ±3.81 m of
  rounding and **cannot resolve the gate even in principle** — `states[-1]` is not a usable
  crossing measurement, the least-squares fit is what recovers sub-quantum precision
  (σ ≈ 1.7 m); and the quantisation staircase IS the residual autocorrelation (lag-1 ρ ≈ 0.43,
  n_eff ≈ 0.40 n), so the OLS variance must be autocorrelation-corrected. **Deflate BOTH variance
  terms** — `Sxx` sums over the same correlated samples as `1/n`; correcting only the first gives
  a 1.15× inflation where the honest figure is **1.58×**.

**Status (2026-09-16):** the quantisation facts stand (`final_approach/fit.py` pins the lattice as `_ALTITUDE_QUANTUM_M` and uses it as its residual-scale floor), but no live code computes an autocorrelation-corrected crossing variance any more, and "83 % of the whole 9.15 m vertical gate" compares against the OLD gate — it has been ±22 m since c9ca54b (2026-08-15).

### EV12 · the observed gate verdict was mostly undecidable (SUPERSEDED)

- **The observed gate verdict is mostly undecidable, and the report must say so.** With σ ≈ 1.7 m
  the 95 % CI is 6.7 m wide against a 9.15 m window, and the fleet's median vertical deviation
  sits ~0.5 m from the upper bound — so **67 % of established KRDU flights have a CI straddling a
  gate boundary**. `evaluate_batch` reports a `marginal` count alongside pass/fail; a bare pass
  rate over 25 ft-quantised data claims more than it knows. **Read the deviation distribution as
  primary, the pass rate as secondary.**

**Superseded (checked 2026-09-16):** `evaluate_batch` no longer reports a `marginal` count — it was removed with the terminal-approach verdicts in fb7b173 (2026-08-12) — the vertical window is ±22 m since c9ca54b (2026-08-15), and v9 verdicts are pass/fail against a determinate window (EV3; `test_speed_gate.py` asserts `speed_marginal` is absent). Kept as the record of why observed verdicts were once read distribution-first.

### EV13 · the fit window is `[−5000, −300]` m

- **The fit window is `[−5000, −300]` m and it is a real methodological choice, not a free
  parameter.** On a 3° path that spans ~900 ft down to ~107 ft above threshold — below the
  1000 ft stabilisation gate, above flare (~50 ft). Measured on KRDU with published TCH: median
  crossing +5.43 m from a `[−8000,−300]` (≈ FAF) window vs **+3.66 m** at `[−5000,−300]` vs
  +4.04 m at `[−2000,−300]`, with σ climbing to 2.50 m at the short end. Starting at the FAF
  biases HIGH (aircraft still intercepting from above); shrinking below ~3 km leaves too short a
  baseline to pin the slope. **Report the sensitivity table, never one number.**

**Where it lives now (2026-09-16):** `final_approach.fit.DEFAULT_WINDOW_M = (-5000.0, -300.0)`, used by the harvest's segment fit (runway assignment and censored crossings; a direct bracket is interpolated instead); the rationale is the comment above that constant.

## The authoritative-threshold check is a SEAM, not a local rule

### EV14 · `_require_target_agrees_with_runway_data` binds every producer

- **`_require_target_agrees_with_runway_data` binds every producer that claims
  `target_source == "runway_threshold"` to `harvest.airports.Runway` within 1 cm.** Its
  position half landed 2026-08-17 and was validated against observed and prediction records
  only — there was no optimizer comparison tree on disk — so it silently broke the one
  producer that moved its target: the constrained-IAF optimizer snapped onto the procedure
  document's rendering of the same CIFP threshold, 0.05–0.22 m away, and **every**
  `runway_cons` record was rejected on the first row. Changing this tolerance, or any
  producer's target, needs both ends checked together;
  `tests/test_pipeline_integration.py::test_constrained_optimizer_target_is_graded_against_the_same_threshold`
  is that pin.

## Batches are STREAMED, not materialized

### EV15 · `load_records` / `iter_records` stream the batch

- **`load_records` resolves each `states_ref` into the full state list, and a batch is now
  tens of thousands of flights.** Measured 0.5 MB retained per record → ~7 GB on an uncapped
  KRDU batch. The roster names the airport (`summary_row` per row as `arr_airport`; the
  harvest's observed roster once, as `summary["airport"]`), so `cli.contexts_for_input`
  resolves the assessment contexts from `summary.json` alone and `iter_records` streams the
  records past `evaluate_batch` one at a time; everything `evaluate_batch` retains
  (`TrajectoryEvaluation`, rows, comparisons) is per-flight metadata, not trajectory arrays.
  A roster row that cannot be placed RAISES — the old fallback silently materialized the
  whole batch instead.

### EV16 · `evaluation.visualize` reads in two passes

- `evaluation.visualize` needs random access for its `--max-tracks` overlays, so it does the
  same in TWO passes: pass one streams the metrics and remembers only which FILES were
  drawable, pass two reloads the sampled few. Verdicts are identical either way (A/B pinned
  at 48/48, 44 pass) — if they ever differ, the streaming path is wrong, not the report.

### EV17 · `load_records` is kept for tests

- `load_records` is `list(iter_records())`, kept for tests; nothing in the pipeline holds a
  batch. `visualize.build_payload` is the streaming builder (there is no list variant).

## Records are written at a declared precision, and the boundary states are not

### EV18 · `STATE_DECIMALS` / `CONTROL_DECIMALS`

- **`evaluation_export.STATE_DECIMALS` / `CONTROL_DECIMALS` round the VALUES in the timed
  arrays.** JSON pays for every decimal digit: at full float repr one state row is 185 bytes
  (`"lat":35.766821578167715` — 17 significant digits for a quantity whose ADS-B source
  resolution is metres), and the arrays are ~98 % of a record. Rounding to 1.1 mm position /
  1 mm altitude / 0.1 mm/s speed takes records from 151 to 110 KB per flight and the
  comparison CZML from 45 to 32 KB; A/B on a KRDU batch: **0 verdict or event_status changes,
  largest deviation difference 0.63 mm**.

### EV19 · the three exemptions

- **Three exemptions, each for a reason, not an oversight:**
  - `initial_state` / `target_state` — `_require_target_agrees_with_runway_data` measures the
    target against the authoritative threshold at **1 cm**; that budget cannot absorb half a
    rounding step.
  - **`t`** — the one field with hard contracts (`final_time_s == states[-1]["t"]` to 1e-6,
    and strictly increasing offsets). `ts_transformer` exports normalized clocks whose
    relative offsets are tiny by construction and pins a 3.7e-13 s horizon surviving export.
    Quantizing it was worth 7 more percentage points of file size; an ordering invariant does
    not go on a budget.
  - `controls` use their own table (`CONTROL_DECIMALS`) — different units, different scale.

### EV20 · `final_time_s` is read back off the serialized array

- **`final_time_s` is read back OFF the serialized array, never recomputed.** With `t` exact
  the two agree anyway, but deriving it makes the invariant structural rather than a
  coincidence. Two writers got this wrong the moment precision entered the contract
  (`evaluation_record`, and `ts_transformer/inference/export.py`'s second unrounded copy of an array it
  already had) — both were caught by `record_from_dict` rejecting the batch, loudly.

### EV21 · the comparison-CZML builder mirrors the table

- `aeroviz-4d/python/build_scenario_comparison_czml.py` mirrors the table
  (`_DEG_DECIMALS`/`_ALT_DECIMALS`) because that package must not import the modeling tree;
  a test pins the two together, including that neither rounds time. **Change them together.**
