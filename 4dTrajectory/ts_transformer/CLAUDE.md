# ts_transformer — learned trajectory prediction

Vendored iTransformer + PatchTST (torch). Mechanism, rationale and result tables live in the
package README; history in `docs/CHANGELOG.md` (2026-07-19, 07-20 ×2).

## Commands

```bash
conda activate aeroviz                                     # the single thesis env (has torch)
TS=4dTrajectory/ts_transformer/__main__.py
python $TS train   --data <arrivals.json|dir> --airport KRDU --model itransformer \
                   --horizon-mode window --output-dir 4dTrajectory/outputs/KRDU/ts_itr
python $TS predict --checkpoint .../checkpoint.pt --data ... --output-dir .../ts_pred
python -m evaluation --input .../ts_pred                   # same gates as the optimizer
python -m pytest 4dTrajectory/ts_transformer/tests -q --import-mode=importlib

# whole chain, 2 models × 2 horizon modes (train → predict → eval → CZML; dataset build and
# split happen inside train, split persisted in the checkpoint)
python run_ts_pipeline.py --airport KRDU
```

## Config & constants (`config.py` is the single source)

Everything below is serialised into every checkpoint.

- `dt_s = 2.0`, `seq_len = 60` (120 s), `pred_len` = 30 (window, 60 s) / **300** (full, 600 s).
- Channels = `(e, n, u, edot, ndot, udot)` threshold-anchored chart; **names AND order are
  load-bearing** (the tuple indexes tensors, normalizer stats and checkpoints —
  `load_checkpoint` refuses a mismatch, which is also what locks out pre-2026-07-20 `ve/vn/vu`
  checkpoints after the transport-consistency change). The velocity channels are the EXACT chart
  derivatives of the position channels (full-transport Jacobian, `geokit.wgs84_curvature_radii`),
  not raw physical ENU components — see the velocity seam in `flight_scenarios/CLAUDE.md`.
- **Horizon trap**: the horizon was sized from the MEASURED duration distribution (p50 328 s /
  p95 651 s), covering **97.8 %** of flights — the old "an arrival is ~3.5–5 min" straight-line
  estimate was WRONG (real arrivals are vectored), do not resize from it. The ~2 % over the
  horizon are cut at H and flagged `horizonCapped`, so their gate verdicts are cap artifacts, not
  model error.
- `summary.json` carries an `accuracy` block (mean AND p95) plus per-row `ade_m`/`fde_m`;
  `overlap` is a REQUIRED arg to `write_batch` — an optional metric is one that silently goes
  missing.
- Aircraft-type resolution (`_resolve_aircraft`, `--aircraft-type`, why `"type": "UNK"` does not
  mean single-type): see `flight_scenarios/CLAUDE.md`.

## Gotchas (recurring, verified)

- **ts_transformer is a purely kinematic BASELINE — no dynamics/aerodynamics is connected, by
  design.** Channels in, channels out; the only symbol it imports from `aerodynamic_model` is the
  `GeodeticState` dataclass (no equations), vs the optimizer's `CasadiSimulator` +
  `rollout_piecewise_constant`. Predictions therefore carry NO flyability guarantee
  (speeds/turn rates/thrust/`Cl_max` are unchecked) — the survey's "statistically plausible but
  unflyable" problem. **Do NOT treat this as an unfinished TODO**; it is what lets the learned
  component be measured on its own. The four routes if it is ever added (post-hoc flyability
  check → post-hoc casadi projection → soft physical loss → differentiable torch dynamics) are
  written up in the package README. Same for single-aircraft-only and deterministic
  point-prediction: scope decisions, listed separately from real gaps in that README.
- **Flyability (`flyability.py`): read the DELTA against the observed tracks, never the absolute
  rate.** The closed-form control inversion (no casadi, no solver) judges against ONE
  clean-configuration drag polar, and real approaches are flown dirty — run on REAL flown tracks
  it first scored **0/149 fully flyable**, i.e. the check was wrong, not the flights. Median
  required thrust on a real arrival is **0.43 kN** (idle); negative required thrust just means
  drag augmentation (speedbrake/flaps/gear), so `thrust_negative` is a SOFT violation and the
  observed baseline (**63.2 %** on KRDU) is the FLOOR, not 100 %. `Cl_max` comes from
  `aero_params_for_aircraft` (A320: 2.7), NOT `LoadFactorSimulator`'s hardcoded 1.5 — an 80 %
  disagreement. **Each flight is judged against its OWN airframe** (`report_for_records` takes
  one `Aircraft` per flight; the report carries `fleet` + `envelopes`) — the first version shared
  one envelope and mis-graded ~44 % of a batch. **Flyability alone is not a quality metric**: in
  the ablation the WORSE predictor scores HIGHER on it in 3 of 4 cells (89.3 % vs 29.6 % while
  2.2× worse at the threshold) by predicting blander paths — a straight line is perfectly
  flyable and completely wrong. Always pair it with the error metrics.
- **Instance normalisation is OFF and must stay off by default.** iTransformer's `use_norm` and
  PatchTST's `revin` are ON upstream; both strip a window's absolute level as "nuisance". In a
  threshold-anchored ENU frame absolute position IS the signal (it decides where the turn onto
  final is, when the descent starts, where the approach ends). Ablated on real KRDU data, all 8
  cells (`outputs/KRDU/_ablation_norm/ablation_results.json`, tables in the package README):
  **off wins 19 of 20 accuracy comparisons, one tie.** Signature of ON: lateral p95 pins at
  14.28–14.50 km in ALL four cells — a model that cannot place the endpoint at all — vs
  2.6–8.5 km for off. Real-data gaps (1.2–2.7×) are much smaller than synthetic (2.4–6.5×), so
  judge on the sweep, not one metric: a single-metric partial pass showed an apparent reversal
  that did not survive consistent scoring.
- **Prediction records are anchored at `t=0` = the anchor sample**, `initial_state` is the
  observed state THERE (not the track start), and the reference record covers the SAME span.
  `evaluation.reference.compare_to_reference` resamples both paths at 101 fractions of *their
  own* arc length, so a whole-track reference against an anchor→threshold prediction reports
  kilometres of pure span mismatch (measured: 4349 m → 833 m once span-matched). **That rebase
  does NOT survive into a shared clock — the CZML builder must add `source.anchorTimeS` back.**
  The reference copied out of `trajectories.czml` still starts at t=0 = the START of the track,
  so writing a prediction's own times through unshifted drew it a whole lookback early (KRDU 05L:
  the forecast's first sample, bit-identical to the reference's t=118 s sample, was plotted at
  t=0 — 12.0 km from where the reference then was). `observed_states` (whole track, negative t
  before the anchor) is REQUIRED in the prediction schema and is the only source for the `look-`
  lookback entity — without it the purple line begins in mid-air with nothing joining it to the
  start of the approach. Lookback = the `t ≤ 0` slice; the anchor sample belongs to both halves,
  so the join is exact, not approximate.

## Open items

- **The KRDU run is DONE (three generations; current = 2026-07-20 B3)** — artifacts in
  `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` + `ts_pred_*` (B3 transport-consistent channels
  + physical-velocity fit; the previous generation is parked in `outputs/KRDU/_pre_b3_transport/`,
  the first is not reproducible — **quote ONLY current-artifact numbers**), tables in the package
  README.
  Robust across all generations: one-pass `full` beats chained `window` on whole-approach lateral
  error for both models (1.5–1.7× mean); PatchTST leads at short lead while iTransformer leads at
  600 s on the raw-tensor accounting (5438 vs 7384 m, n=893 — direction held in all three
  generations, margins 1.16–1.36×; channel-independence can't represent a turn's east/north
  coupling). NOT stable across retrains: gate-pass counts (0–4 of 152 — only "forecast ≠
  certifiable approach" survives) and the tail-vs-mean story (architecture-dependent).
  Hence: **treat any margin under ~1.5× as provisional** — both a split change and a ≤0.3 % data
  rescale moved effects of that size. The two lead-time accountings (record vs raw-tensor) are
  NOT comparable — the README states both with their n.
  Remaining: only KRDU trained (4 other airports harvested; the per-threshold ENU frame makes
  pooling a real design question, not a bigger `--data` glob).
- **The gate-pass conclusion needs re-deriving, not re-quoting.** The recorded "gate-pass counts
  0–4 of 152" scored a ±3 m vertical window against data offset ~33 m by the datum bug. Accuracy
  metrics (ADE/FDE, deviation vs a reference in the same frame) should be nearly unchanged; the
  gate verdicts were not measuring what they claimed.
- Follow-ups: single-aircraft only (no traffic interaction / ATC intent) and deterministic (no
  multimodality) — both are the survey's named open problems. Flyability is MEASURED but not
  FIXED (nothing projects a prediction back inside the envelope — README routes 2–4), and its
  polar is clean-configuration only, which is why it is read as a delta.
