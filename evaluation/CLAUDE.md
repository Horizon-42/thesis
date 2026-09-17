# evaluation — file-based trajectory judging + batch metrics

`geokit` + stdlib + the (stdlib-only) `aircraft` parameter package. Judges records against
published approach geometry; the harvest (`final_approach`) decides *which* runway, this
tree decides *how good*. Report schema: `terminal-approach-evaluation-v9`.

**This file is an index**: each line ends in the ID of its full text in
`docs/EVALUATION_REFERENCE.md` (moved there verbatim 2026-09-16, with dated corrections). A new
fact gets a new ID there and one line here.

## Gates & batch metrics (`evaluation/thresholds.py`, `evaluation/speed_gate.py`)

- **Lateral = half the published runway width, and that is the whole rule**
  (`LATERAL_CRITERION_ID = "runway_half_width_at_threshold"`, 15.24–22.86 m) — a
  landing-geometry claim, NOT navigation containment (EV1).
- **Vertical ∈ [−22, +22] m** from the published LTP+TCH path (ICAO Doc 9613 RNP APCH Baro-VNAV
  FAS): LPV always; LNAV/VNAV when the runway's own RNAV approach publishes those minima and a
  runway-leg path (`Runway.baro_vnav_minima`: KRDU 32, KSMF 35R — no caller flag); a runway with
  no RNAV procedure (KRDU 14) grades vertical indeterminate. Gates judge the TRUE-dynamics
  rollout's final state (EV2).
- **Speed ∈ [V_ref,lo·√max(n,1), V_ref,hi + 20 kt] at the crossing, anchored on the type's
  PUBLISHED approach speed** (v9; design and sources `docs/THRESHOLD_SPEED_GATE.md`). The table is
  `aircraft/reference_speeds.json` — **adding a type is a cited JSON row, never a code change**.
  Type from `source.dynamics_typecode` (computed) or `source.aircraft_type` (observed); composed
  into the verdict for ALL subjects; observed = fitted ground speed corrected by the METAR
  headwind, else a STATED ground-speed proxy; pass/fail against a determinate window;
  `indeterminate` only when nothing can be judged, reason on the row. **Never compare observed and
  computed speed rates as one quantity; quote speed rates per `target_source`** (EV3).
- Batch metrics: solve/success rates, lateral mean/p95/max, vertical spreads, flight times,
  path-shape deviation at 101 arc-length fractions (EV4).
- Read side is **manifest-ONLY** (`summary.json` roster; no glob fallback — globbing counted
  orphans); `categories.json` entries carry an explicit `"constrained": bool` (EV5).

## Gotchas (recurring, verified)

- **A navigation-standard bound that never binds is worse than no bound at all** — the old
  guidance term bound 0 times and published `guidance_lateral_m` 2× wrong, unnoticed because it
  was inert (EV6).
- **A record's `target_state` is not authority on where the runway is — the assessment context
  is**: the frame origin is `context.threshold_lat/lon`, both coordinates cross-checked when
  `target_source == "runway_threshold"`; a record with no `target_source` gets the STRICT
  reading (EV7).
- **The report schema version has FOUR homes** (producer; `ts_transformer/data/lateral_eligibility.py`
  imports it; `aeroviz-4d/src/data/evaluationReport.ts` mirrors it; fixtures import it). A frontend
  "evaluation report is malformed" usually means the VERSION disagrees. Same for
  `thresholds.CONTEXT_SCHEMA_VERSION` (EV8).

## Judging observed (ADS-B) data

- Observed grading is the SAME state interpolation as computed grading: `source.crossing_span`
  (`measured_bracket` / `fitted_tail`, `final_approach.crossing`) says where the crossing lives;
  `final_time_s` stays on the last MEASURED row; an estimated event on a record without a span is
  stale and raises (the harvest's `--observed-only` rebuilds it) (EV9).
- **An observed track's `states[-1]` is NOT its arrival** (966/996 KRDU tracks end a median 325 m
  short, airborne) — grading the final state measures where coverage stopped (EV10).
- ADS-B altitude is quantised to 25 ft = 7.62 m, so a single sample cannot resolve a crossing;
  the least-squares fit does (EV11). The fit window `[−5000, −300]` m
  (`final_approach.fit.DEFAULT_WINDOW_M`) is a methodological choice: starting at the FAF biases
  HIGH, below ~3 km the baseline is too short (EV13).
- **Superseded, kept as history**: the 9.15 m vertical gate, the autocorrelation-corrected
  crossing CI and the `marginal` count (removed in fb7b173, 2026-08-12; the gate is ±22 m since
  c9ca54b, 2026-08-15) (EV11, EV12).

## Seams, streaming, precision

- **`_require_target_agrees_with_runway_data` binds EVERY producer claiming
  `target_source == "runway_threshold"` to `harvest.airports.Runway` within 1 cm** — changing the
  tolerance or any producer's target needs both ends checked together
  (`tests/test_pipeline_integration.py::test_constrained_optimizer_target_is_graded_against_the_same_threshold`) (EV14).
- **Batches are STREAMED** (`iter_records`, contexts resolved from `summary.json` alone; ~0.5 MB
  per materialized record); a roster row that cannot be placed RAISES (EV15).
  `evaluation.visualize` reads in two passes and its verdicts must equal the streaming ones (EV16);
  `load_records` is kept for tests only (EV17).
- Records are written at a declared precision (`evaluation_export.STATE_DECIMALS` /
  `CONTROL_DECIMALS`; 0 verdict changes on A/B) (EV18) — EXCEPT `initial_state` / `target_state`
  (the 1 cm target check), `t` (hard contracts) and controls (their own table) (EV19);
  `final_time_s` is read back off the serialized array, never recomputed (EV20);
  `build_scenario_comparison_czml.py` mirrors the table — **change them together** (EV21).
