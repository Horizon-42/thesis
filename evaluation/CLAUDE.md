# evaluation — file-based trajectory judging + batch metrics

`geokit` + stdlib only. Judges records against published approach geometry; the harvest
(`final_approach`) decides *which* runway, this tree decides *how good*.
Report schema: `terminal-approach-evaluation-v5`.

## Gates & batch metrics (`evaluation/thresholds.py`)

- **Lateral = half the published runway width, and that is the whole rule**
  (`LATERAL_CRITERION_ID = "runway_half_width_at_threshold"`, 15.24–22.86 m across this fleet).
  It is a LANDING-GEOMETRY claim — did the crossing lie over the pavement — NOT a
  navigation-containment one; `METHODOLOGY["terminal_lateral"]` states that inside every report.
- **Vertical ∈ [−22, +22] m** from the published LTP+TCH path (ICAO Doc 9613 Vol II Pt C Ch5
  §5.3.4.4.7, RNP APCH Baro-VNAV FAS), available for LPV always and for LNAV/VNAV only with
  approved Baro-VNAV + a published reference. Gates judge the TRUE-dynamics rollout's final state.
- Batch metrics: solve/success rates, lateral mean/p95/max, vertical spreads, flight times;
  path-shape deviation vs reference = both paths resampled at 101 fractions of their own
  horizontal arc length.
- Read side is **manifest-ONLY**: `load_records` reads a batch dir via `summary.json` roster
  (`results[].eval_file`); manifest-less dir / listed-missing file / empty roster raise (no glob
  fallback — globbing counted orphans). `categories.json` entries carry an explicit
  `"constrained": bool` (frontend validator REQUIRES it; `_cons`-suffix detection deleted).

## Gotchas (recurring, verified)

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
- **The evaluation report's schema version has FOUR homes and a green test suite proves nothing
  about them.** Bumping `evaluation.metrics.REPORT_SCHEMA_VERSION` v4→v5 landed in the producer
  only; `4dTrajectory/ts_transformer/lateral_eligibility.py` and
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

- **An observed track's `states[-1]` is NOT its arrival at the target.** A solve terminates at
  its target by construction; a harvested arrival is a truncated recording of a flight that
  continues — 966/996 KRDU tracks end a median **325 m short** of the threshold, still airborne.
  Grading observed data on final-state deviation measures where ADS-B coverage stopped. The
  meaningful observed metric fits each flight's own established final-approach line and
  extrapolates to the threshold (which also validates itself: the fitted glidepath comes out
  3.02–3.13° at all five airports).
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
- **The observed gate verdict is mostly undecidable, and the report must say so.** With σ ≈ 1.7 m
  the 95 % CI is 6.7 m wide against a 9.15 m window, and the fleet's median vertical deviation
  sits ~0.5 m from the upper bound — so **67 % of established KRDU flights have a CI straddling a
  gate boundary**. `evaluate_batch` reports a `marginal` count alongside pass/fail; a bare pass
  rate over 25 ft-quantised data claims more than it knows. **Read the deviation distribution as
  primary, the pass rate as secondary.**
- **The fit window is `[−5000, −300]` m and it is a real methodological choice, not a free
  parameter.** On a 3° path that spans ~900 ft down to ~107 ft above threshold — below the
  1000 ft stabilisation gate, above flare (~50 ft). Measured on KRDU with published TCH: median
  crossing +5.43 m from a `[−8000,−300]` (≈ FAF) window vs **+3.66 m** at `[−5000,−300]` vs
  +4.04 m at `[−2000,−300]`, with σ climbing to 2.50 m at the short end. Starting at the FAF
  biases HIGH (aircraft still intercepting from above); shrinking below ~3 km leaves too short a
  baseline to pin the slope. **Report the sensitivity table, never one number.**
