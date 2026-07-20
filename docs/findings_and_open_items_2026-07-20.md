# Findings & Open Items — 2026-07-20

Consolidated record of what this working session established (the flight-identity
unification refactor, the full optimization batch re-run, and the ts_transformer
test-split prediction regeneration). Overlaps with `CLAUDE.md`'s Open Items by design —
that list is the terse every-session version; this document carries the evidence and the
recommended actions. History lives in the two 2026-07-20 entries of `docs/CHANGELOG.md`.

---

## A. Established findings (each backed by measurement)

### A1. The flight-identity design was broken in three layers — all fixed
- **Per-runway CZML entity ids were massively duplicated** (pre-fix): the harvest
  downloads in chunks, `_unique_id` restarted its numbering per chunk, and the cross-chunk
  merge de-duplicated by `(icao24, landing_time)` without re-uniquing ids → 128 duplicate
  ids across KRDU's five runways (`N993FG` ×10 on RW32). Cesium merges same-id packets, so
  two namesake flights rendered as one garbled entity.
- **Cross-view id aliasing**: the combined file's positional `_N` renumbering made the
  same string (e.g. `SWA1692_2`) name *different physical flights* in the per-runway vs
  combined views; the optimizer (fed combined) and ts (fed per-runway) derived different
  `flight_key` stems for one flight, and the comparison CZML's white reference line could
  be the wrong aircraft.
- **Frontend callsign join collisions**: FlightTable looked optimizer facts up by
  callsign, so namesake flights swapped each other's V / mass / verdict.
- **Post-fix verification** (on the fully regenerated artifacts): reference hit rate
  **100% in all 19 comparison categories** (was a 22% duplicate-callsign dropout);
  per-runway and combined views agree on 996/996 flight ids; zero duplicate entity ids;
  zero summary rows missing `landing_time_utc`. The ts checkpoints' split keys are
  byte-identical — **no retrain needed**.

### A2. KRDU RW32 is systematically hard — and it is NOT the old truncation artifact
- After the full re-run (truncation / altitude-floor / HS / identity fixes all in):
  runway_cons RW32 = **79 offTarget + 59 failed of 198** (every other runway ≤ 9
  offTarget); asdb RW32 = **197/198 IPOPT infeasible**.
- Consistent across categories → runway/procedure-specific, not solver noise. Prime
  suspect: RW32's RNP-AR procedure (H05LZ) is constrained with the default RNP 1.0 disc
  (~926 m at k=0.5) because per-leg RNP is not extracted from CIFP; it should be ~278 m
  (RNP 0.3) — see item B1.

### A3. KSTL runway_cons has a milder failure cluster
- 12R 53/200, 30R 41/168, 30L 30/200, 24 21/80; the "all IAF(s) infeasible" rows
  repeatedly name the single IAF **`PAULY`**. Worth the same geometry audit as A2 (B2).

### A4. GPU nondeterminism is amplified by chained forecasting (measured)
- Re-running `predict` on identical checkpoints and data: one-pass (full) aggregate ADE
  reproduces to **< 0.1 m**; chained (window) cells drift **2–4%**; one borderline flight
  crossed the 106.75 m lateral gate (3 → 4 passes).
- Conclusion: **single-run numbers from chained mode cannot be read precisely**; this is a
  second, independent justification for the README's "treat margins under ~1.5× as
  provisional" rule. A jitter note is now in the README.

### A5. The flat chart's real error magnitude (a wrong doc claim corrected)
- `channels.py`'s docstring claimed "well under a metre over the 25 km ring" — **false as
  an absolute-coordinate statement**: equirectangular vs true tangent-plane ENU differs by
  **~40 m** at the ring edge on diagonal bearings (the `e·n·tanφ/R` cross term), and
  `u = Δalt` differs from tangent-plane z by the **~49 m** curvature drop (deliberate).
- What *is* true: none of it reaches the metrics — predictions, references, and the loss
  live in the SAME chart, so systematic distortion cancels; the residual is ~0.2% local
  scale distortion at the ring edge, → 0 at the threshold where the gates judge. The
  docstring now says exactly this.

### A6. The constant seam is closed: geokit and the optimizer's NE frame are now bit-identical
- Root cause: geokit's `METRES_PER_DEG_LAT = 111_320.0` was a hand-rounded convenience
  value; the optimizer derives `WGS84_A·π/180 = 111319.4908…` — a **4.6 ppm** seam
  (0.11 m at the 25 km ring). `metres_per_deg_lon` is pure `·cos(lat)` on both sides.
- Fixed: the geokit constant is now derived from `WGS84_A` (IEEE multiplication
  commutativity makes it bit-identical to `frame.py`'s product); the frontend
  `geoConstants.json` was regenerated; all suites green. Applied AFTER the batch finished,
  so all 15 cells share one constant.

### A7. ts channels mix chart-coordinate positions with physical velocity components
- Integrating the velocity channels does not exactly reproduce the position channels:
  mismatch O(0.2–0.5%) (the cos ratio, h/R, chart radius vs curvature radii R_M/R_N) —
  a few metres per minute of integration.
- Removable in closed form with transport factors (the Jacobian the optimizer's
  full-transport RHS already encodes) by redefining velocity channels as chart
  derivatives — but that changes channel semantics and therefore requires a retrain. See B3.

### A8. The two lead-time-error accountings are not directly comparable
- The table recomputed from exported records (predictions truncated at the threshold)
  vs training-time `error_by_horizon` on raw tensors (which sees past truncation) diverge
  at long leads: under the record accounting, 600 s has n=1 and at 300 s the two full-mode
  models are within 2% (3852 vs 3802 m); under the raw-tensor accounting iTransformer
  clearly led at 600 s (5407 vs 6962 m).
- The README table now states its accounting and per-cell n; "iTransformer leads at long
  lead" was downgraded to a caveated claim.

### A9. The correct reading of `hermiteSimpsonNormalizedFullTransport` (corrected once)
- The decision state **is** threshold-anchored metric NE (`(lat−lat_t)·R`,
  `(lon−lon_t)·R·cos lat_t`) — same form and anchor as the ts channels; but the dynamics
  are evaluated on the reconstructed physical geodetic state (exact full-transport RHS):
  a pure affine change of variables for Jacobian conditioning, zero modelling change.
  Only the `localEnu` scheme family approximates dynamics in a flat frame.

---

## B. Open items (by priority)

### Near-term, directly actionable
- **B1. Extract per-leg RNP from CIFP, then re-run KRDU RW32.** The leading hypothesis
  for A2: H05LZ (RNP-AR) currently gets the default RNP 1.0 disc (~926 m) instead of
  ~278 m (RNP 0.3). After extraction, re-run KRDU runway_cons/asdb alone and check whether
  the RW32 failure population dissolves. If 197/198 infeasible persists, audit RW32's
  approach geometry (course, intercept angles) next.
- **B2. Audit the KSTL `PAULY` IAF constraint geometry.** The repeated single-IAF
  infeasibility pattern is the same family as A2; plot that IAF→runway leg geometry and
  the ψ corridor first to rule out a coding/direction issue.
- **B3. Bundle these three into the next retrain** (none justifies a retrain alone):
  1. transport-consistent velocity channels (A7 — closed-form, removes the
     position↔velocity inconsistency);
  2. if finer time resolution is wanted: `dt=1 s` requires L=120 / H=600 to keep the same
     time coverage (~2× training cost, limited information gain — the source is ≤1 Hz and
     velocities come from a 15 s window fit);
  3. recompute the lead-time table with the raw-tensor accounting to restore the 600 s
     column (A8).
- **B4. Report the aircraft-type fallback hit rate in flyability output.** How often the
  `--aircraft-type` fallback is actually used is currently unreported per batch; one
  summary line prevents "UNK all graded as A320" from hiding again.

### Requires a design decision
- **B5. Multi-airport training / pooling.** 3747 arrivals harvested across 5 airports;
  only KRDU trained. The per-threshold ENU frame makes pooling a *frame design* question
  (not a bigger `--data` glob): candidates include per-threshold frames with shared
  normalization, or a canonical runway-heading-aligned rotation.
- **B6. Actually fixing prediction flyability (README routes 2–4).** Currently measured,
  not fixed; route 2 (post-hoc casadi projection) is the most conservative, route 4
  (differentiable dynamics in torch) the most invasive. Thesis-narrative-level decision,
  like B5.
- **B7. Feed the 3-colour comparison overlay to the Observe view.** The comparison
  datasource is not yet wired into the approach view, and `useCzmlLoader`'s clock write is
  still ungated for the Observe+comparison two-writer case.

### Long-term / known scope limits (not bugs)
- **B8.** ts is single-aircraft, no traffic interaction, no ATC intent, deterministic
  point prediction (no multimodality) — the survey's named open problems; these define the
  baseline's boundary.
- **B9.** CIFP leg speed restrictions not extracted (no speed-bearing data source yet;
  the canonical `speedMaxKt` field is ready).
- **B10.** HSL linear-solver hook dormant (free MA27 measured slower than MUMPS);
  revisit with an MA57 academic license.
- **B11.** The plain (non-landing) download path's identity is still `id(_N)_icao24`
  (no landing time exists there). That path is display-only and keeps `_unique_id` as its
  sole discriminator; if it ever feeds the modeling pipeline, add a timestamp field first.
- **B12.** Pre-existing, unrelated failure:
  `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one`
  (numpy scalar-conversion deprecation) — still to clean up.

---

*Verification baseline at time of writing: 191 (geokit + flight_scenarios + aeroviz-4d
python) + 84 (optimization + ts) + 453 (vitest) tests green; batch 15/15 with
`overall_fail=0`; ts predictions 4 × 152 test-split-only.*
