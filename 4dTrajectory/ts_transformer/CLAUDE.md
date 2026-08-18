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
- **Controls are DIMENSIONLESS in this package** (`control_envelope.py`, the single source):
  `(thrust_fraction ∈ [-0.2, 1.0], bank_rad ∈ ±π/4, load_factor ∈ [0.2, 2.0])`, the same box on
  every airframe. Newtons appear in exactly two places — `physical_controls()` on the way into
  the dynamics, and `forecast.py` on the way out to the evaluation record, whose contract stays
  in newtons and is shared with the CasADi optimizer. **The thrust floor is negative on
  purpose**: an approach needs net-negative force (idle + speedbrake/flaps/gear drag this
  clean-configuration polar does not model), and with a 0 N floor **40 % of inverted teacher
  segments pinned at the bound** on the KSJC cohort — 0.33 % after. This is NOT the optimizer's
  envelope (`casadi_optimizer.make_control_bounds` keeps a non-negative floor and a 0.5 load
  floor): that box is a flyability claim, this one is a learned head's search space.
- **Two orthogonal dynamics axes.** `control_dynamics_model` ∈ `point-mass` |
  `first-order-lag` is the physics; `control_dynamics_backend` ∈ `reanchored-rk4` |
  `transport-chart-velocity` | `scaled-transport-chart-velocity` is the state representation the
  long rollout carries. The registry in `control_dynamics_backends.py` is keyed by the PAIR.

## Gotchas (recurring, verified)

- **The teacher inverse must be the inverse OF THE CONFIGURED FORWARD MODEL, and nothing else
  can catch it if it is not.** A schedule solved against the wrong equations is finite, bounded,
  the right shape, and its own optimizer reports a falling loss — it simply reproduces nothing.
  So `control_inverse_dynamics.py` registers each inverse under the SAME config key as its
  forward model, and `tests/test_control_inverse_dynamics.py` closes the loop numerically for
  every registered model (roll a known schedule → invert the dense result → require the schedule
  back: 5e-3 in thrust fraction and load factor, 0.5° in bank). A model added without an inverse
  fails at registry lookup. Two live bugs this replaced: `build_inverse_dynamics_target`
  hard-unpacked a 7-field batch, which only exists under `fixed-dt` — so the teacher builder was
  **unusable under every native-grid recipe including simple-v1**; and it rebuilt reference
  velocities with a hardcoded `smoothed-position-difference` regardless of
  `config.reference_velocity_source`, inverting a different velocity definition from the one the
  supervision uses. The inverse also adds back the `ω × v` transport term the chart RHS
  subtracts (worth only ~1.6e-4 g, but it is what makes it the exact inverse rather than a close
  one).
- **The lagged model is the point-mass model plus three actuators, not a second flight model.**
  `torch_lag_dynamics` *wraps* `transport_chart_rhs`, so the force equations, stall handling,
  transport term and chart projection are the same code. Verified both ways: at τ = 0.1 s the
  two trajectories end within 0.5 % of path length and the gap is first order in τ; at the 2 s
  default they differ by **~3 km over a 240 s rollout**, so it is a materially different model,
  not a smoothing pass. **τ shorter than the integrator step produces NaN, not a worse answer** —
  explicit RK4 on `y' = -y/τ` is unstable above `h/τ = 2.785` — so `TSConfig` refuses it at
  construction. Relevant when sweeping τ.
- **The anchor's control state is inverted from the observed lookback, never from the first
  command** (`dataset.anchor_controls`, 11 samples ≈ 20 s). Starting the actuators at the first
  command would place the aircraft in a bank it has not rolled into, and the lag would be paid
  twice. It reads only samples at or before the anchor, so it is as deployable as the history
  window itself.
- **A named recipe cannot be cross-validated as itself.** `simple-v1`/`simple-v1-lag` freeze
  `epochs`/`patience`, which a search deliberately shortens, so CV candidates carry
  `control_recipe_name='custom'` while the parent recipe stays in `base_config` inside the run
  contract. `simple-v1-lag` additionally leaves the three time constants open — τ_bank is the
  thing the sweep resolves — and `applicable_cv_parameters` drops that axis as inert under
  `point-mass` rather than multiplying the grid by 5 for identical folds.
- **`prediction_output` decides whether dynamics is connected at all, and the answer differs.**
  `state` is the purely kinematic BASELINE — channels in, channels out, the only symbol it takes
  from `aerodynamic_model` being the `GeodeticState` dataclass. Its predictions carry NO
  flyability guarantee (speeds/turn rates/thrust/`Cl_max` unchecked), which is the survey's
  "statistically plausible but unflyable" problem, and that is **deliberate, not an unfinished
  TODO**: it is what lets the learned component be measured on its own. `control` is the
  opposite — the model emits bounded controls and a differentiable RK4 rollout of the shared
  point-mass equations turns them into the trajectory, so every prediction is dynamically
  admissible by construction. The two are the experiment. Single-aircraft-only and deterministic
  point-prediction remain scope decisions for both; see the package README.
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
