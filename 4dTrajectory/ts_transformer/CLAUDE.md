# ts_transformer — learned trajectory prediction

Vendored iTransformer + PatchTST (torch). Predicts the remainder of an arrival from a
threshold-anchored lookback window.

**This file is the map: the contracts that break things if violated, and a one-line trigger for
every trap.** The evidence behind each line — measurements, campaign results, the causes already
ruled out — lives in `docs/ENGINEERING_NOTES.md`; status and next steps in `docs/OPEN_ITEMS.md`;
mechanism and result tables in the package `README.md`; history in the repo's `docs/CHANGELOG.md`
(2026-07-19, 07-20 ×2).
Read the notes before designing an experiment or touching the loss, rollout or output layer.

- **Seed noise on the control path is ~125 m of pooled ADE, not 30 m** (2026-09-08: `B1_point_matched` seeds 1337/2024 = 1248/1373 m around native32's 1322; duration MAE spread ~1 s). The 30 m line came from two-seed STATE arms. A single-seed control-arm ADE difference below ~125 m is not evidence; a gate on it needs a second seed (early stopping off, as A0.b's p180) or must be read against this line. Straight-in FDE and duration MAE spreads are smaller (~0.9 s MAE) but still single-seed unless replicated.

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

## The three prediction paths — this is the experiment

`prediction_output` decides whether dynamics is connected at all, and the answers are the
point of the package, not a migration in progress.

- **`state`** is the purely kinematic BASELINE: channels in, channels out, the only symbol it
  borrows from `aerodynamic_model` being the `GeodeticState` dataclass. Its predictions carry
  **no flyability guarantee** — speeds, turn rates, thrust and `Cl_max` are unchecked. That is
  the survey's "statistically plausible but unflyable" problem, and it is **deliberate**: it is
  what lets the learned component be measured on its own.
- **`control`** is the opposite: the model emits bounded controls and a differentiable RK4
  rollout of the shared point-mass equations turns them into the trajectory, so every prediction
  is dynamically admissible by construction.
- **`closure`** (scene design P1.c, 2026-09-05; **a COMPARISON ARM since 2026-09-07** — the
  latent-intent design demoted it, and its P1.d tracker was DELETED 2026-09-07 with its BLOCKER
  unfixed: a nearest-node search that jumps legs on 0.6 % of flights) regresses 14 DECISION numbers — the join
  distance, a via pose in runway axes, K=4 slowness knots (the duration is their integral),
  K=4 height knots — and `closure_output.reconstruct` draws the trajectory in closed form
  (`closure_geometry.via_dubins` + `closure_profile`; velocities = tangent × ground speed).
  Training is L1 regression on per-flight LABELS fitted from the truth
  (`docs/p1_closure_oracle.py labels` → `closure_labels_path`), carried as the batch context
  and never an input; `predict --closure-from-labels` draws every flight from its label (the
  family's ceiling, `source.closureFromLabels`). Locks: `enu` chart, `normalized` horizon,
  `checkpoint_selection_metric=fixed-anchor-objective` (the loop never draws the path), no
  random anchors; a labels file must be the airport's own and cover the cohort (a run refuses
  one that covers no flight, prints the covered share otherwise). The drawn path has no
  dynamics of its own (22 % fully flyable); the P1.d tracker that flew it with the point-mass
  rollout (`control/constraints/closure_tracking.py`, `predict --closure-track`) is RETIRED —
  code DELETED 2026-09-07, its numbers (+10.5 m of ADE, 92 % fully flyable) kept as history in
  `docs/2026-09-06_closure_p1d_tracking_results.zh.md`. Do not rebuild it.
- **The control path also carries two AXES (2026-09-07, `docs/2026-09-07_latent_intent_design.zh.md`)**:
  `latent_dim > 0` puts a latent intent z on the control output (`control/latent.py`:
  q(z | future) in training only, a K-component mixture prior from the context, z reaches
  the controls AND the duration; inference decodes the prior's top-1; `predict
  --latent-samples K / --latent-random K / --latent-shuffle / --z-from-posterior` write
  `modes/`, `random/`, `shuffled/` and the z-oracle; z never enters a record); and
  `cta_conditioning=given` makes the given arrival time BE the duration (`predict
  --cta-offset-s` is the scheduler's counterfactual). Both READ THE FUTURE in their oracle
  forms and the run name says so (`control+z8`, `z=posterior`, `cta=given`) — never a
  prediction result.
- **...and a third axis, the DURATION HEAD** (2026-09-07,
  `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §三):
  `duration_head=quantile` replaces the scalar `FinalTimeHead` with
  `QuantileFinalTimeHead`'s five `config.DURATION_QUANTILES`, monotone by cumulative
  softplus. Its MEDIAN is the duration the rollout flies (`final_time_s` unchanged), all
  five go to `source.durationQuantilesS`, and `run_ts_eta_calibration.py` turns them into a
  calibrated interval. `predict --cta-from-quantiles` then decodes each flight at its OWN
  quantile — the run name says `cta=self-q`, and that one IS a prediction result.
  **`duration_head=two-head` (B1.b, 2026-09-08) carries BOTH heads**: the point head drives
  the rollout duration exactly as `point` does and the quantile head emits only the
  published distribution. It exists because `B1_point_matched` showed the two gains come
  from different mechanisms and do not overlap — the PATH gain is the duration term's
  WEIGHT (point head at `final_time_loss_weight` 26: ADE 1248 vs native32's 1322) and the
  ARRIVAL-TIME gain is the quantile HEAD (MAE 23.9 vs 25.9 s pooled, 10.3 vs 13.6 s
  straight-in) — so under `quantile`, where the rollout flies q50, the head's path cost is
  forced onto the trajectory for nothing.

Single-aircraft-only and deterministic point-prediction are scope decisions for all three (README).

Two orthogonal dynamics axes underneath the control path: `control_dynamics_model` ∈
`point-mass` | `first-order-lag` is the physics; `control_dynamics_backend` ∈ `reanchored-rk4` |
`scaled-transport-chart-velocity` is the state representation the long rollout carries (the
unscaled `transport-chart-velocity` was RETIRED 2026-09-07 — a measured regression that the
nondimensional variant replaced on 2026-08-02; `run_naming` still abbreviates it so the 13
stored 2026-07/08 configs keep their `_tcv` names). **The registry in
`control/dynamics/backends.py` is keyed by the PAIR.**
The lagged model *wraps* `transport_chart_rhs` — same force equations, stall handling, transport
term and chart projection — so it is the point-mass model plus three actuators, not a second
flight model.

## Contracts (violating these breaks loading, or silently scores the wrong thing)

- **Channels = `(e, n, u, edot, ndot, udot)`, and names AND order are load-bearing.** The tuple
  indexes tensors, normalizer stats and checkpoints; `load_checkpoint` refuses a mismatch. The
  velocity channels are the EXACT chart derivatives of the position channels (full-transport
  Jacobian, `geokit.wgs84_curvature_radii`), not raw physical ENU components — see the velocity
  seam in `flight_scenarios/CLAUDE.md`.
- **The chart origin is NOT "the threshold"; `FlightSeries.target_chart` is.** `enu` /
  `runway-aligned` anchor at the assigned threshold (target_chart ≡ 0), `airport-enu` at the
  airport reference point. Every consumer that judges distance-to-go must measure from
  `target_chart`; **a new one that reads `hypot(e, n)` is silently wrong under the airport frame
  and only there.**
- **Controls are DIMENSIONLESS in this package** (`control/envelope.py` is the single source):
  `(thrust_fraction ∈ [-0.2, 1.0], bank_rad ∈ ±π/4, load_factor ∈ [0.2, 2.0])`, same box on every
  airframe. Newtons appear in exactly two places — `physical_controls()` into the dynamics, and
  `forecast.py` out to the evaluation record. **The thrust floor is negative on purpose** (an
  approach needs net-negative force this clean polar does not model). This is NOT the optimizer's
  envelope, which is a flyability claim; this one is a learned head's search space.
- **`config.py` is the single source** and everything in it is serialised into every checkpoint.
  `config.input_channels` is what the model sees, `config.channels` what it predicts.
- `dt_s = 2.0`, `seq_len = 60` (120 s), `pred_len` = 30 (window, 60 s) / **300** (full, 600 s).
  The horizon was sized from the MEASURED duration distribution (p50 328 s / p95 651 s), covering
  **97.8 %** of flights — the "an arrival is ~3.5–5 min" straight-line estimate was WRONG (real
  arrivals are vectored), **do not resize from it**. The ~2 % over the horizon are cut at H and
  flagged `horizonCapped`; their gate verdicts are cap artifacts, not model error.
- **A new loss term must be added to `objective.loss_component_names`**, not only to the
  objective's `extras` — otherwise `KeyError` on the first batch, *after* the slow dataset
  build.
- **A latent run's total KL says nothing about WHERE it is spent, and that is the whole
  failure mode.** Three 180-epoch arms were read as "the information is just small" while
  every posterior mean sat ON the prior mean (displacement 0.05–0.2 prior σ against a gate
  of one) and the budget bought only a narrower posterior. `history.json`'s `latent` block
  therefore carries `component_kl_mean_term_nats` / `component_kl_variance_term_nats` (the
  split; the variance term is the REMAINDER of `per_dimension_kl`, never a second closed
  form), `mean_displacement_sigma`, `component_kl_per_dim` and `active_units_0p05` — the
  FIXED 0.05-nat ruler, because `active_units` moves with the free-bits budget and made that
  gate unreadable across arms. **The three `component_*` numbers sum to
  `component_kl_nats_per_flight`, NOT to the charged `kl_nats_per_flight`** (free bits and
  the mixture estimator separate the two). Read them with `run_ts_latent_readout.py
  --history <run>/history.json` (which needs no `--arm`: this is the epoch-1 reading), or off
  a checkpoint with `run_ts_latent_probe.py`; the gate sentence is
  `control.latent.displacement_verdict` and its ruler `DEAD_MEAN_DISPLACEMENT_SIGMA`, so no
  surface restates either. A scalar auxiliary target concentrates the information in ONE
  dimension, so on such an arm the median displacement can miss what `component_kl_per_dim`
  shows — and that target is an INPUT of the posterior encoder, so a small `latent_aux`
  proves nothing on its own.
- **A supervision term on the turn rate reads the RHS, never a restated identity.** The
  shared point-mass RHS integrates `ψ̇ = g·n_realized·sin φ / (V·cos γ)` with the
  STALL-LIMITED load factor; the textbook `g·tan φ / V` equals it only on a coordinated
  level turn. `aerodynamic_model.torch_dynamics.heading_rate_rad_s` reads the ψ row out of
  `enu_rhs` itself for exactly this reason, and it is evaluated on
  `EndpointControlRollout.actual_controls` — the commands under the point-mass model, the
  ACTUATOR STATES under the lag. Pricing the command under the lag charges a turn the
  rollout never flew. Its target's uniform `(k+1)·T/N` endpoints coincide with the
  rollout's `cumsum(segment_durations)` ONLY through `true-time-position ⇒ uniform
  durations` (the config refuses anything else); `reference_control_supervision`'s midpoints
  ride on the same chain.
- **The anchor state is `batch_contract.anchor_state(x, C)`, never `x[:, -1]`** — a history
  is `[B, L, C + K]` with `K` input-only conditioning columns (`target_conditioning`,
  `intent_conditioning`), and the control loss refuses a `[B, C + K]` anchor on the first
  batch. Two call sites had the raw slice until 2026-09-05; any control run with conditioning
  died there.
- **`cta=self-q` and `cta=given` are two ARMS, and only the first is quotable.** `given`
  hands the decoder the truth duration; `self-q` (`predict --cta-from-quantiles`) hands it
  the model's OWN `q_τ` and reads no future. The B3 arm TRAINS as `given` — the truth CTA
  drives the rollout while the quantile head trains as a target beside it — so the run
  directory honestly wears `cta=given` and only the prediction directory says `self-q`.
  The fan is `quantiles/qNN/` (plus `a20lo` / `a20hi` when a conformal table exists), the
  top-1 records ARE the q50 decode, and every record carries `source.ctaFromQuantiles` /
  `ctaQuantile`. Refused with `--cta-offset-s` and with `--z-from-posterior`.
- **A quantile record says whether it is CALIBRATED, always.** `source.calibrated` is
  written on every quantile record — `false` is the claim that this checkpoint has no
  conformal table, not a missing key — and `durationIntervalS` / `durationIntervalStratum` /
  `durationIntervalCohort` (the table's split, airports and smoke flag) appear only with one.
  The table is bound to `checkpoint_sha256` AND to `DURATION_QUANTILES`: one left behind by
  other weights, or written under other levels, RAISES rather than widening this run's
  intervals by someone else's δ.
- **The calibrated coverage is MEASURED, never guaranteed, and the gate reads the DEPLOYED
  block.** Two separate traps. (i) The calibration split is also the split the checkpoint was
  SELECTED on, so the finite-sample split-conformal guarantee does not hold as constructed;
  every surface says "empirically measured cross-half coverage" and the guarantee-bearing
  read is pre-registered as one test-split measurement at `freeze-test`. (ii) A per-stratum δ
  is measured on its own stratum's members, but DEPLOYMENT falls through
  `INTERVAL_STRATUM_PRECEDENCE` — a refused stratum's flights take the pooled δ, and the
  pooled row says nothing about them (synthetic check: pooled row 0.794, the fallen-through
  group 0.167, deployed pooled 0.756). `calibration.calibrate` therefore publishes a
  `deployed` block that scores every held-out flight under the δ it would really get, and
  **design gate 3.4-2 reads that number**.
- **`intent_conditioning=truth-…` checkpoints read the FUTURE** (the truth join point, the
  lead's true landing time) — the Phase 0 upper-bound instrument of the scene design, never a
  result to quote as a predictor; the run name carries `intent=truth-…` so it cannot pass as
  one. Its lead and remaining-time channels are measured at the window's anchor, so it
  refuses random train anchors; `FlightSeries.lead_landing is None` means "roster never
  consulted" and raises, `LeadLanding(None)` means "no earlier landing" and reads as a
  clear runway. A `truth-join-duration` arm's `final_time_error_s` is an identity check
  (its input IS the duration target), never a duration result.
- **`overlap` is a REQUIRED arg to `write_batch`** — an optional metric is one that silently goes
  missing.
- **τ shorter than the integrator step produces NaN, not a worse answer** (explicit RK4 on
  `y' = -y/τ` is unstable above `h/τ = 2.785`); `TSConfig` refuses it at construction.
- **Instance normalisation is OFF and must stay off** (iTransformer `use_norm`, PatchTST
  `revin`). In a threshold-anchored frame absolute position IS the signal. Signature of ON:
  lateral p95 pins at 14.3–14.5 km in all cells — a model that cannot place the endpoint.
- **Prediction records are anchored at `t=0` = the anchor sample**; `initial_state` is the
  observed state THERE, and the reference record must cover the SAME span (a whole-track
  reference against an anchor→threshold prediction reports kilometres of pure span mismatch).
  **That rebase does not survive into a shared clock — the CZML builder must add
  `source.anchorTimeS` back.** `observed_states` is REQUIRED in the schema and is the only source
  for the `look-` lookback entity.
- **Prediction is the anchor's own past only**: the anchor's control state is inverted from the
  observed lookback (`dataset.anchor_controls`), never from the first command.
- **The teacher inverse must be the inverse OF THE CONFIGURED FORWARD MODEL.** A schedule solved
  against the wrong equations is finite, bounded, the right shape, and its own optimizer reports
  a falling loss — it simply reproduces nothing. `control/dynamics/inverse.py` registers each
  inverse under the SAME config key as its forward model; a model added without one fails at
  registry lookup. The transport term (ω×v) is UNCONDITIONAL and there is no correct "off".
- **A fitted teacher table belongs to ONE width, anchor and cohort** — `control_imitation_target=
  "fitted"` reads per-flight schedules that reproduce the truth only at the N, the anchor, the
  uniform partition and the total duration they were fitted under. Six things are checked, and
  the run is refused on any of them: the SCHEMA and the `uniform` duration mode at load, then the
  cohort's AIRPORTS (flight keys are unique within an airport only, so a foreign table would
  otherwise read as "covers 0 of N"), N, the anchors the dataset actually samples, coverage (with
  the count — there is no partial mode) and each flight's `truth_duration_s` to 1e-6 s.
- **The fitted teacher is a TRAINING-TIME INPUT, not something a dataset loads.** `train.fit_model`
  opens the file once and hands the table to the train and validation window sets; every replay
  path (`evaluate-fit`, `predict --z-from-posterior`, the approach-cohort comparison, any
  `forecast`) builds its own window set WITHOUT it and must keep working when the file is gone.
  Its digest rides in `checkpoint_metadata.json` / the payload's `fitted_teacher`, never in
  `data_provenance` — that object is compared for equality by `evaluate-fit` and `freeze-test`.

## Current defaults and their status

| axis | default | status |
|---|---|---|
| `coordinate_frame` | `enu` | keep — the airport frame makes the model average across parallel pairs |
| `state_position_reference` | `absolute` | `corridor-bounded` ADOPTED as candidate default (4 seeds, no regression); **`anchor-relative` is VETOED by its own pre-registered rule.** It follows the package's one mechanism for a value like this: it is in `STATE_POSITION_REFERENCES` (what a STORED config may say, so the 2026-09-03 `state_v2_20260903/A_anchor_relative` artifact still loads and names) and NOT in `STATE_POSITION_REFERENCES_AVAILABLE` (what a NEW run may select — the CLI's choices, and what `cli.common._refuse_unavailable_selection` checks so `--config-overrides` cannot get past it either). `control_command_hook="nominal-residual"` is the same pair |
| control recipe | `simple-v3` | = `simple-v2` + `control_imitation_loss_weight`; **its weight 64.0 does NOT transfer between airports — recalibrate per airport**. A named recipe is a published DETERMINISTIC arm: all seven `latent_*` fields are pinned at their defaults, so **a latent run is `custom`** (every latent arm file already says so; adopted 2026-09-07 after measuring that no stored artifact changes name, slug or loading) |
| `control_dynamics_model` | `point-mass` | `first-order-lag` buys smoothness + 3.4 % ADE; τ=2.0 s is defensible, not CV-selected |
| procedure penalty (state + control) | weights at 0 | NOT adopted — kept as an option. Its two hinge SCALES (100 m / 30 m) are `objective.PROCEDURE_{LATERAL,VERTICAL}_SCALE_M` module constants, not fields: units, never swept, retired 2026-09-07. The closure timing group's 60 s is `closure_output.CLOSURE_TIMING_SCALE_S` for the same reason. The four scales a named recipe PINS (`position_loss_scale_m`, `final_time_scale_s`, `control_velocity_loss_scale_mps`, `control_heading_rate_loss_scale_dps`) stay fields — a module constant there would silently redefine every published simple-v* comparison |
| command hook | off in training | **`predict --command-hook barrier --hook-saturation soft` is the ADOPTED use**; no arm trained THROUGH a hook beat its predict-time counterpart (six tried) |
| `--project-final` | off | deployment fallback; FAF-gated wrecks vectored flights |
| `target_conditioning` | off | `channels` helps only the duration head; PatchTST refuses it |
| `latent_dim` | 0 | the latent intent (L2); `latent_prior_components` / `latent_beta` / `latent_free_bits_nats` / `latent_posterior_init_std` and L2.f's two below mean nothing without it and are refused |
| `latent_beta_warmup_epochs` / `latent_aux_duration_weight` | 0 / 0.0 | L2.f's two levers on the POSTERIOR MEAN, which is where the information died (see the contract above). The first ramps β linearly from 0 over N epochs (`effective_latent_beta`, the one place the schedule is written; the epoch's objective is `replace(config, latent_beta=…)` and the validation pass is scored under the SAME one); the second adds a train-only `Linear(latent_dim→1)` on the POSTERIOR SAMPLE predicting `truth_duration_s / final_time_scale_s` (component `latent_aux`, registered iff weighted, never called in `decode`, refused under `cta_conditioning=given`). Named `beta-warmup=` / `aux-T=`. Arms: `docs/experiments/l2f_mean_information_arms.json` |
| `cta_conditioning` | `off` | `given` = the CTA is the duration (L3); a delivery-form demonstration, never a prediction result. **`self-q` is a PREDICT-TIME label, not a trainable value** (`CTA_CONDITIONINGS_AVAILABLE` keeps it out of a new run): `predict --cta-from-quantiles` stamps it on the config it writes beside the records, so the directory that read no future cannot be named like the one that did |
| `duration_head` | `point` | `quantile` = B1's five `DURATION_QUANTILES` of the SAME duration (`QuantileFinalTimeHead`, monotone by cumulative softplus; the median IS `final_time_s`). The loss swaps the squared final-time residual for the sum of five pinball losses **under the same component name**, so `loss_component_names` is unchanged. `two-head` = B1.b: BOTH heads, the POINT head driving the rollout (`final_time_head`, the same module and state-dict keys `point` trains) and `duration_quantile_head` emitting the interval. **The second head is built LAST** so a `two-head` arm and a `point` arm start parameter-for-parameter identical (measured: 34 shared keys, 0 differ) — `_initialize_duration_head` zeroes only the last layer, so building it earlier would shift the point head's hidden draw and unpair the arms gate 1 compares single-seed. The training dropout stream still cannot be matched — its pinball rides in a component of its OWN, `duration_quantile`, so `loss_component_names` gains one entry there and only there. Every value but `point` is refused off the control output and refused with `latent_dim > 0` — z reaches the duration by shifting the point head's ONE logit, and under a posterior sample the quantiles would be conditioned on the flight's own future. The named recipes pin it at `point`, so a quantile or two-head run is `custom` and wears `T=q5` / `T=2h` |
| `duration_quantile_loss_weight` | `1.0` | B1.b: what weighs the pinball sum. Under `quantile` it multiplies the `final_time` component the pinball replaced (the 1.0 that multiplied it before the field existed, so nothing moved); under `two-head` it weighs the separate `duration_quantile` component. **Refused non-default under `point`** — there is no such head — and symmetrically `final_time_loss_weight` is **refused non-default under `quantile`**, where the pinball replaced the point term outright. Named in a run only when it deviates (`pinball=3`) |
| `n_segments` (control) | 64 | **32 is free** (L1: 1322 vs 1333 m, bank skill 0.726 vs 0.728); the deployed head's 257 numbers become 96 |
| `control_state_loss_grid` | native | `fixed-dt` without the imitation term (it is not registered there) trips the straight-in veto (FDE 703 → 2863) and brings the bank wiggle back — the trajectory-error loss alone is not enough |
| `control_heading_rate_loss_weight` (+ `_scale_dps` 1.5) | 0 | L1.b arm ②: the teacherless way to name the bank — the rollout's own ψ̇ at the segment endpoints against the flown track's. `_scale_dps` is the UNIT the residual is read in (half a standard-rate turn), not the dose. Doses 1.0 / 8.0 bracket an unknown; **not measured yet** |
| `control_bank_tv_loss_weight` | 0 | L1.b arm ③: mean \|step\| between adjacent COMMANDED banks, in half-box units. **It prices REVERSALS, not slope** — `sign(x)` cancels on a monotone run's interior — and exact flatness is a STATIONARY POINT (value 0 and gradient 0), which is where the zeroed head init starts every run, so it never leaves a flat schedule on its own. "TV changed nothing" is that before it is a dose. **Not measured yet**. Both terms register under `true-time-position` ONLY (where `velocity`/`imitation` live) and `TSConfig` REFUSES a non-zero weight under any other objective rather than ignoring it |
| `checkpoint_selection_metric` | `fixed-anchor-common-grid-ade` | WHICH epoch's weights the checkpoint keeps — see the table below. Lower is better for all three; the value NAMES the run (`select=…`, a `META_FIELDS` entry) |
| `lr_plateau_metric` | `selection` | WHICH number `ReduceLROnPlateau` measures its plateau on — never which epoch is KEPT. `objective` steps it with the macro validation objective (the SAME `val_loss` the epoch record writes, not a third number), `selection` with the checkpoint-selection value. It matters exactly when the two PART, which a fixed-anchor arm never does: on A0-random's `_grid` arm the objective improved to epoch 60 (1.147 → 0.707) while the grid ADE stalled after epoch 8, so the scheduler halved the LR from epoch 20 and reached **9.4e-7 by 60** — the model stopped training at the epoch the READOUT stalled. `checkpoint_metadata.json`'s `lr_scheduler.metric` says which it stepped on; names the run `lr-metric=objective`. **`objective` is REFUSED while the objective itself moves under the model's feet**: a `latent_beta_warmup_epochs` ramp reweights it every epoch and a `procedure_loss_dual_step` reprices the same trajectory every epoch, and under either the plateau scheduler would cut the rate straight through a schedule that is still ramping. A0.b (i) |
| `random_train_anchor_sampling` | `uniform` | HOW a random train anchor is drawn — refused without `random_train_anchor`. It cannot change WHICH anchors are admissible (`eligible_random_train_anchors` + the future contract), so both policies train the identical cohort on the identical anchor population. `uniform` draws over the flight's admissible SAMPLES, i.e. uniformly in TIME; **pooled over flights that over-weights the near end relative to the stored population**, because every flight gets one draw whatever its length and a kilometre near the runway holds more samples than one at 25 km (whole KRDU val split, 1404 flights / 181,906 anchors / 200 epochs: draws **34.3 %** under 6 km against **25.1 %** of the population, **17.9 %** beyond 20 km against **32.9 %**). `remaining-path-uniform` places the draw uniformly across the flight's OWN admissible remaining-path span and takes the nearest anchor — equal weight per km — moving those to 31.0 % and 21.0 %. **A stratum draw was tried first and REJECTED**: it moved training toward the runway (≥ 20 km 16.9 → 4.1 %) because the grid cuts the near end into four 2-km strata and leaves one open stratum spanning 20–123 km. Every epoch records the drawn counts per stratum BESIDE the population they came from (`history.json`'s `train_anchor_sampling.remaining_path_strata{,_population}`), under both policies — neither number means anything alone. Names the run `anchors=remaining-path-uniform`; `sampling_version` distinguishes the two in `training_anchor_contract`. A0.b (ii) |
| `control_imitation_target` | `inverse-dynamics` | WHAT the imitation term imitates. The default's schedule, flown open-loop, lands **2.5–7.8 km** from the truth it was read off (L0), so "imitating it perfectly" is not "flying the truth". `fitted` reads the per-flight table `run_ts_control_basis_oracle.py --checkpoint` fits through the same rollout (88–433 m) and needs `control_fitted_teacher_path` (which names the run: `teacher=<dir>/<file>`); refused with `random_train_anchor`, with `control_imitation_loss_weight=0`, and off `control_state_objective=true-time-position` (the only objective the term is registered under). L5.a, **not yet measured** — the arms are `docs/experiments/l5_fitted_teacher_arms.json` |

**The three checkpoint-selection metrics** (`config.CHECKPOINT_SELECTION_METRICS`, dispatched
through `validation.VALIDATION_SELECTIONS`; the LR scheduler, early stopping and the kept
weights all read the one number):

| value | what it scores | use it when |
|---|---|---|
| `fixed-anchor-objective` | the validation objective itself | the loop never draws the path it would be judged on — **the closure output, which `TSConfig` requires it for**. **Refused for a latent run**: that objective decodes a posterior sample, so it reads the future and is stochastic |
| `fixed-anchor-common-grid-ade` | airport-macro common-grid ADE at the **L−1 anchor** | the default, and right for every fixed-anchor arm — L−1 is where they train and where they are judged |
| `anchor-grid-common-grid-ade` | the same ADE at **every anchor set the cohort supports**, equal weight per SET: L−1 plus whichever of `anchor_grid`'s 16 / 12 / 8 / 6 km CANDIDATE bins clears the coverage gate, each flight at its own closest admissible sample (full lookback, ≥ 60 s of truth after it) | **a random-anchor arm**. The fixed metric scores the ONE anchor such a model is least specialised for and froze `A0_random_hr8_tv1` at epoch 10 (L−1 ADE 2949 vs native32's 1322) while that checkpoint drew BETTER geometry than the fixed arm at every anchor (chamfer p50 −114…−524 m). Each set's ADE is the mean over the flights that HAVE that anchor — absent, never scored 0 — then the airport macro; the per-set values, counts and `dropped_bins` land in `history.json`'s `validation_anchor_grid` block every epoch, `fixed_anchor_common_grid_ade_m` among them so the L−1 veto stays readable. Costs ≈ 4–5× the selection stage (one extra deployable replay per surviving bin; the L−1 pass is reused), ≈ +12–15 % per epoch at KRDU scale. **Refused with `cta_conditioning=given` or `intent_conditioning=truth-…`** — the grid re-reads the oracle at every bin anchor, so the metric would be selecting on how fast it converges |

**`anchor_grid.py` is the one definition of the grid** — bins (`DEFAULT_ANCHOR_GRID_KM`
20/16/12/8/6/4/2 km, `VALIDATION_ANCHOR_GRID_KM` the four the metric may select on), the
60 s future floor, `PARTIAL_COVERAGE` = 0.5, the per-flight `bin_anchor` rule and the
fixed-at-L−1 `strata_fixed_at_l1` rule. `run_ts_anytime_curve.py` and the selection metric
import the SAME objects (`tests/test_anchor_grid.py` asserts identity, not equality): two
grids that merely agreed today would make "the curve improved" and "this epoch was selected
on the curve" claims about different anchors.

**The grid's VALUES live one level down, in the leaf `anchor_strata.py`**, and `anchor_grid`
re-exports them — because `anchor_grid` imports `dataset` while `dataset` needs the same
kilometres for the training-anchor draw, so a single home would mean a cycle or a second
copy. The leaf owns `DEFAULT_ANCHOR_GRID_KM`, the strata those values cut when read as EDGES
(`REMAINING_PATH_STRATA_EDGES_M`: seven edges, **eight** strata, the open ends `<2km` and
`>=20km`), `remaining_path_strata` (`np.digitize` on the same profile the bins are chosen
from) and the draw law `remaining_path_uniform_offset`. It must stay free of `dataset` and
torch — `tests/test_import_boundaries.py` pins that, and it is the whole reason `dataset`
can import it at module scope.

**A candidate bin is a candidate, not a guarantee.** Measured coverage of the KRDU
validation cohort (1404 flights, 60 s floor): **20 km 33.7 %** (excluded outright),
**16 km 37 %**, **12 km 78.7 %**, **8 / 6 km 99.7 %** — the median flight has **13.4 km**
left at L−1, and 4 / 2 km are partial-to-empty under the floor (≈ 53 s / 27 s of truth
left). So `build_anchor_grid_validation_plans` DROPS a bin under `PARTIAL_COVERAGE` for any
airport — on that cohort, 16 km — with a printed notice and a `dropped_bins` record, and
refuses the metric outright when fewer than two bins survive. An equal-weighted fifth of a
selection value must not come from a long-haul subcohort. **Which bins survive is a property
of the cohort, not of the model**, so every arm on the same split is selected on the same
sets and their values are comparable.

Command hooks are called once per control SEGMENT, at its start, with the rollout's own state,
returning the command flown — and **the record carries the schedule FLOWN, not the network's**.
Two rules that cost real trajectories when missed: a hook that changes the bank **must
re-coordinate the load factor**, and every rate gain must be `min(gain, 1/Δt)` because the
command is HELD.

## How to read results here (conventions that prevent wrong conclusions)

- **Bank skill is read against the random-flight floor and the same-runway twin ceiling that
  `docs/score_control_arms.py` prints per arm — never against 1.0**, which is unreachable.
- **Flyability: read the DELTA against observed tracks, never the absolute rate** — the polar is
  clean-configuration, real approaches are flown dirty, and on REAL tracks it first scored
  0/149. **Flyability alone is not a quality metric**: the WORSE predictor scores higher on it in
  3 of 4 cells by predicting blander paths. Always pair it with error metrics.
- **Read both metric families, never one.** Every stratum table prints the time-aligned
  ADE/FDE next to the time-free geometry from `geometric_metrics` (chamfer, discrete Fréchet,
  arc-aligned ADE, length ratio, duration error, along-path lag). Phase 0 (2026-09-05): the
  truth-duration arm cut vectored ADE 2356 → 2011 m with NO geometric gain — a timing-only
  improvement reads as a model improvement on ADE alone. Truth = observed rows CLOSED to the
  threshold at `true_final_time_s` (they stop a median 380 m / 6 s short at KRDU;
  `--geometry-truth observed` reproduces the Phase 0 diagnostics within 2 m). arc-ADE / lag
  are aggregated over the flights whose exported polyline is a route (heading reversals at
  ≤ 5 % of nodes) and print `n/a` below a 95 % share: the STATE output's node-scale
  saw-tooth reverses at ~50 % of its nodes and doubles its own arc length (control arms and
  the truth: 0), nothing smooths it, so on state campaigns the readable geometry is chamfer
  + Fréchet. The length ratio column is information, not a gate.
- **A per-airport ADE without its ROUTE MIX is not a comparison.** Inside a matched stratum every
  airport scores the same; the whole spread is the share of flights in that stratum. Reweighted
  to the pooled mix KSJC goes 483 → 1526 m, best of five to worst. The signature to recognise:
  ADE and cross-track improve while **FDE does not**. → `approach_difficulty.py`
- **Treat any margin under ~1.5× as provisional** — both a split change and a ≤0.3 % data rescale
  moved effects of that size. Seed floor on the frame axis: threshold arms move 5–22 m pooled
  ADE, airport arms up to 107 m.
- **At n=1404 the paired sign test returns p = 3e-16 for pure seed noise.** Read magnitudes,
  never p.
- Truth values (flown-track bank RMS, shared share) are **per-airport and must be measured**;
  they used to be hardcoded to KSJC's and every KRDU number was read against the wrong reference.
- **Quote ONLY current-artifact numbers.** The KRDU run has three generations; the first is not
  reproducible.

## Layout

**Five modules carry the training plane, and each answers one question** (T3, 2026-09-07 —
`train.py` was 3,027 lines and `__main__.main` 714):

| module | the question | lines |
|---|---|---|
| `objective.py` | what is a prediction scored against | 1,079 |
| `validation.py` | how is a fitted model replayed on a split, and which epoch is kept | 1,162 |
| `train.py` | the epoch, the cohort, the checkpoint | 1,248 |
| `dataset.py` | observed arrivals → model windows (`FixedAnchor` = one common anchor, `ExplicitAnchor` = one CALLER-SUPPLIED anchor per flight, `RandomAnchor` / `RemainingPathUniformAnchor` = one per flight per epoch, drawn uniformly over samples or over the flight's remaining-path span; `training_window_class` is the one place the two anchor axes pick a class) | 1,941 |
| `anchor_grid.py` | which remaining-path anchors a re-anchored reading is taken at (no torch) | 159 |
| `anchor_strata.py` | the remaining-path VALUES both sides of that edge read: the grid's km, the strata they cut, the train-anchor draw law (leaf — no `dataset`, no torch) | 105 |
| `data_provenance.py` | which arrival rosters produced this run (pure hashing, **no torch**) | 503 |
| `splits.py` | which split a flight belongs to | 197 |
| `cli/` | one module per subcommand (`common` 857, `predict` 484, `evaluate_fit` 119, `cross_validate` 78, `train` 49, `freeze` 42, `__init__` 15) | 1,644 |
| `__main__.py` | the bootstrap and the `COMMANDS` table it dispatches from | 131 |

Two edges that a change must not reverse: `evaluation_protocol` reaches
`data_provenance`, never `dataset` (a boundary test bans torch from that path), and
`batch_contract` sits BELOW `objective` — `closure_output` and `control/latent` return
`LossComponents`, so it cannot move up.

Control-specific code lives in **`control/`**, by role rather than behind a `control_`
prefix: `envelope`, `heads`, `conditioning`, `latent`, `basis_fit`,
`dynamics/{backends,rollout,inverse,hooks}`, `loss/{components,fixed_dt}`,
`training/diagnostics`, `constraints/{barrier_filter,gates}`.

**A dynamics backend is a ROW, not a class**: `control/dynamics/backends.py` maps the
`(control_dynamics_model, control_dynamics_backend)` PAIR to
`(endpoint_fn, dense_fn, post_fn, runs_hooks)`. `post_fn` turns whatever state the
integrator carries into the one public `(channels, geodetic)` pair; `runs_hooks` is why the
point-mass rows refuse a command hook. A new pair supplies three functions and a flag.

**`archive/` is not the package.** Completed campaigns are kept there (README each, naming
the result documents that cite them, the commit they were taken from, and anything vendored
in to keep them self-contained) and are OFF the import path:

- `archive/oracle_teacher_2026_08/` — the 2026-08 inverse-dynamics teacher: eight modules,
  five runners (four of them `run_ts_*`), two test files. Superseded by `simple-v3`'s
  in-training imitation term.
- `archive/nominal_law_hook_2026_09/` — the never-adopted nominal tracking law +
  bounded-residual command hook (`nominal_residual.py`, `guidance_laws.py`). The adopted hook
  is the predict-time barrier, which stays live.

`tests/test_architecture.py` asserts nothing live imports the archive — package modules,
`tests/` and the root `run_ts_*.py` runners alike — and that no `__init__.py` makes it
importable. A finished one-off driver belongs there, not beside the live runners.

**Runners for the anytime / calibrated-ETA line** (2026-09-07,
`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`):

- `run_ts_anytime_curve.py` — **A0**: replays `--checkpoint LABEL=PATH` (repeatable) from the
  `anchor_grid` REMAINING-PATH grid (which it imports, and which the `anchor-grid-common-grid-ade`
  selection metric selects four bins of) and reports, per bin per stratum, ADE (mean / p50 / p95) and
  FDE beside the time-free chamfer and Fréchet, |Δt| p50/p80 and the predicted duration p50.
  Five things it exists to keep right: the strata are computed once at **L−1 and fixed for
  every bin** (relabel per bin and a flight leaves the vectored stratum exactly when it rolls
  out on final — a survivor curve reads as an improving one); the bin coordinate is
  `approach_difficulty.remaining_path_profile_m`, which the covariate itself reads, so a
  consumer binning on distance-to-go never restates the arc length; **bins hold different
  flights**, so the monotonicity verdict is PAIRED over the flights present in both adjacent
  bins and prints that n (and reads the ADE **median**, the package's convention, not the
  mean); each checkpoint's arm is named from its OWN `random_train_anchor` (`A0-fixed` /
  `A0-random`) because one run may hold both and their difference IS the out-of-distribution
  cost; and every per-flight row stays in the artifact so another paired reading needs no
  re-run. Refused: `cta_conditioning=given` and `intent_conditioning=truth-…` (both read the
  future, the latter afresh at every anchor). `--command-hook` / `--hook-saturation` mean what
  they mean in `predict`. **`--min-future-s 60` empties the 2 km bin and most of the 4 km one**
  (≈27 s / 53 s of truth left at approach speed) — stated as `n=0 / partial`, but s_freeze can
  then only be read at ≥ 6 km, and the **~125 s duration-head floor** makes |Δt| p80 RISE
  toward the runway anyway (L1_native32 vectored: 64 s at 12 km, 141 s at 4 km).
  **`--write-records` makes a bin PUBLISHABLE**: the same forecasts the cells were scored on
  (ONE forward pass — the record summary's per-flight `ade_m` IS the curve's) also go through
  `export.write_batch` into `<out>/records/<label>/<bin>km/`, the shape `predict` writes, so
  `python -m evaluation` and the comparison-CZML publisher take it unchanged. Each
  `summary.json` gains an `anytime` block (schema `ts-anytime-records-v1`: campaign, arm, bin,
  the anchor rule, split/limit, `measured_flights`, `records`, coverage) — a bin is a
  re-anchored SUBSET of the split, and `publish_ts_experiment_trajectories.py` reads that block
  to give it its own category key (`…_a12km_val`), its own picker id (`<run>@12km`, because the
  picker dedupes by experiment id), the record campaign as its picker group, and a label that
  states the bin and BOTH denominators (`244 of 300 flights, --limit 300 of 1404 in the split`).
  Under `--write-records` the whole artifact is built inside the `.partial-*` staging directory
  and renamed on success, so a crash publishes no half-written record set — and the flag is
  STRICTER than the curve alone (the record contract refuses a non-finite metric that the curve
  would have printed as a NaN cell). **A bin of a POOLED checkpoint cannot be published per
  airport**: the runner replays the whole cohort its provenance names and offers no airport
  narrowing, so the directory holds every airport's flights; the publisher refuses it rather
  than filing all of them under each airport's category.
- `run_ts_eta_calibration.py` — **B2**: split-conformal (CQR) calibration of a
  quantile-bearing checkpoint's interval (`duration_head` ∈ `quantile`, `two-head`). Reads the DURATION HEAD ALONE
  (`forecast.duration_quantile_predictions` — one forward per flight, no rollout, no CTA),
  which is why it is seconds of CPU and why it may load a `cta=given` checkpoint
  (`load_arm(..., refuse_cta_given=False)`, the only instrument that may; the
  `intent_conditioning` refusal still applies — that oracle is IN the history). The VAL
  split is halved by the checkpoint's own `split_seed`: **half A fits the DEPLOYED δ and
  half B measures what it covered** — the mirror is a stability check, never averaged in.
  `--split test` / `train` are refused with the reason; a `--limit` SMOKE table is refused
  at the sidecar unless `--allow-smoke-table`. **The cut itself is probeable, READ-ONLY**:
  `--half-seed N` re-cuts the halves through the same `calibration_halves` and is refused
  without `--readout-only` (which writes the readout to `--out` and touches no sidecar), the
  table then carrying `half_seed` / `deployed_half_rule` and a `PROBE HALF RULE` banner that
  `write_conformal_table` refuses with no escape hatch — a deployed δ comes from the
  documented rule alone. **Read a single cut's deployed coverage with its CUT noise**: over
  five seeds on KRDU val the two coverages are strongly anti-correlated (a δ fitted on an
  easy half under-covers the other and over-covers when mirrored), so the deployed−mirror
  gap has sd ≈ 0.05 against a binomial SE of 0.015 and FLIPS SIGN; B1_quantile α=0.2 reads
  0.745 at the deployed seed 1337 and 0.791–0.818 at four others (mean 0.790), i.e. gate
  3.4-2's verdict on that arm is cut-dependent (2026-09-08, `docs/CHANGELOG.md`). Per stratum
  (`approach_difficulty.strata_masks`), and only the three
  `calibration.INTERVAL_STRATUM_PRECEDENCE` can deploy are fitted at all; below
  `calibration.MIN_CALIBRATION_FLIGHTS` = 30 in a half the stratum is REFUSED and a flight
  in it falls through to the pooled δ. The table lands in the checkpoint's
  `checkpoint_metadata.json` under `conformal` — a SIDECAR, never in `data_provenance`
  (which `evaluate-fit` / `freeze-test` compare for equality) — and carries its own cohort
  (`airports`, `limit`, `smoke_test`) plus the half rule.
- `run_ts_quantile_fan_readout.py` — **B3**: `--arm <pred_dir>` of a `--cta-from-quantiles`
  run; it reads the five `qNN` leaves only (the calibrated endpoints are `predict
  --interval-endpoints`, off by default and not part of any gate). Per stratum: the share of
  flights whose truth duration falls in `[q10, q90]` and in the calibrated interval, the
  median widths (the veto reads the vectored one against 120 s), and the truth path's
  chamfer to the NEAREST of the five decodes against its chamfer to q50 — gate 3.4-3, read
  on the in-fan subset with the whole cohort beside it. **That geometric column is a
  readout, not a coverage guarantee** (§六 6): five trajectories are not a distribution over
  trajectories. **Its `cal.hit` column is IN-SAMPLE on a val arm** — the flights it scores
  are the ones the δ was fitted on — and is marked `cal.hit*`; the gate's coverage is the
  calibration readout's DEPLOYED block.
- `run_ts_eta_error_readout.py` — **B0**: |`final_time_error_s`| p50/p80/p90 and the SIGNED
  p10/p50/p90 (same for `fde_m`) per stratum, straight out of existing `summary.json` files.
  A row is used only if it carries every metric AND every `STRATA_COVARIATES` field — a
  present-but-null `established_at_anchor` would otherwise read as False and change stratum.
  Measured 2026-09-07 on KRDU val: vectored |Δt| p80 **65.8–72.5 s** against straight-in
  **12.0–20.3 s**, so one pooled ETA interval cannot serve both strata.

**The latent line's own runner** (`docs/2026-09-07_latent_intent_design.zh.md` §六 L2.f):

- `run_ts_latent_probe.py` — **L2.f**: the training-side densities of one or more latent
  checkpoints on a split (`--checkpoint LABEL=PATH`, repeatable), through the SAME cohort
  rebuild as A0 (`load_arm` / `cohort_series`, so the roster rule has one owner): prior and
  posterior per-dimension spread, the posterior mean's displacement in prior sigmas, the
  per-dimension KL and its mean/variance split, the prior's total std against N(0, I).
  Refuses a non-latent checkpoint (no posterior) and, through the shared loader, a
  `cta=given` / `intent=truth-…` one (that loader takes an `instrument` name so each runner
  refuses in its own voice). `--limit N` is a PREFIX of the split, not a sample — the
  displacement median moved 25 % between 100 and 200 KRDU flights, so a limited table is a
  smoke test and the artifact says so. **The posterior reads the future: never a prediction
  result.**

**A replay runner fingerprints through `data_provenance.checkpoint_data_provenance(payload,
manifests)`, never `arrival_data_provenance(manifests)`** — the helper reads the pre-split
lateral-pass roster iff the checkpoint recorded one. Without the roster the v5 cohort
fingerprints as 14 435 KRDU arrivals against the checkpoint's 14 378 and
`require_matching_data_provenance` reports "the manifest changed" on a correct pair. The A0
runner hit it first; the L5.a fitter had the same plain call and died at startup on every v5
checkpoint until `e8df12f` (`docs/code-health-followups.md` §19).

**The ts identity of an eligibility roster is its eligible SET, never the roster file's
bytes** (`data_provenance`, schema `ts-arrival-data-v4-eligible-set`, 2026-09-08). The roster
embeds UPSTREAM provenance — `sources.evaluation_report_sha256` — so regenerating the observed
evaluation moves its bytes over an eligible set that did not change by one flight. That is not
hypothetical: on 2026-09-07 the five observed reports went v6 → v9 with byte-identical eligible
sets (KRDU 14 378 keys), and the byte-bound v3 identity then refused EVERY checkpoint trained
before it — `predict`, `evaluate-fit`, `run_ts_*` replay runners and `publish_ts_experiment_
trajectories.py` (which held a second copy of the comparison). So:
- the compared eligibility entry is `{schema_version, policy, eligible_set_sha256}`, hashed by
  `eligible_set_digest(keys)` — THE one definition, which `splits.data_selection_audit` also
  hashes its split rosters with, and which the publisher and `run_ts_pipeline` import rather
  than re-deriving. **The roster's `counts` are NOT in it**: three of the five are reject
  tallies read off the observed evaluation (`excluded_lateral_indeterminate`,
  `evaluation_only`), so a re-graded flight that never was eligible would move them and refuse
  the checkpoint — the same defect, one level down. The other two are already compared as
  `arrival_candidate_count` and `source_records`;
- the byte facts and the counts stay AUDITABLE and out of every comparison:
  `data_selection.pre_split_eligibility[*].roster_sources` (from `eligibility_sources`) keeps
  the roster path, its digest, its counts and the evaluation report it was joined against;
- a checkpoint carrying the retired `ts-arrival-data-v3-eligibility-bound` fingerprint stays
  usable EXACTLY: its eligible set is re-derivable from the checkpoint alone — **`source_records`
  IS the eligible set** (the roster's keys are validated to exist in the manifest, so the records
  that survive the filter are exactly the eligible ones) — so `require_matching_data_provenance`
  compares it to today's per airport, refuses by name with both set digests, and then reads the
  v3 entry as today's. It deliberately reads NOTHING else from the payload: an earlier version
  rebuilt the checkpoint's `TSConfig` to recompute `data_selection`'s split digests, which
  re-imposed the model-recipe contract on a reader that needs none of it and raised `TypeError`
  on three real pooled checkpoints. Never by a registry of old roster bytes;
- `checkpoint_metadata.json` / `history.json` name the map `eligible_sets`; artifacts written
  before 2026-09-08 name `eligibility_rosters` (roster bytes) and are read through the
  checkpoint payload instead — a CV `cv_results.json` of that generation can only be checked by
  bytes, so `run_ts_pipeline` re-runs CV when they moved (and prints which reason it was);
- **an IN-FLIGHT `cross_validation/cv_candidate_progress.json` from before 2026-09-08 is refused
  on resume**: the run contract it is bound to now names `eligible_sets` where it named the
  roster files, so `_load_candidate_progress` raises — naming the file and saying to delete it —
  rather than silently re-running finished candidates. Deleting that one file restarts the
  search from candidate 0; a finished `cv_results.json` is read, not refused.

**Measurement code is CODE.** Reusable logic goes in the package with tests
(`control/basis_fit.py`, `geometric_metrics.py`, `approach_difficulty.strata_masks`);
a runnable experiment goes in a top-level `run_ts_*.py` runner beside the others;
**`docs/` holds documents**. The `docs/*.py` scripts predate this rule and are a layout
defect, not a pattern to copy (`docs/code-health-followups.md`) — do not add to them, and
move what you touch. `tests/conftest.py` already puts the package on `sys.path`, so a new
test file needs no path preamble.

**Membership rule**: a module belongs in `control/` only if EVERY consumer of it is
control-specific. `prediction_outputs` (holds `StatePrediction`), `terminal_state_loss`,
`arc_length_geometry`, `fixed_dt_supervision` and `flyability` therefore stay at the top
level — `fixed_anchor_validation` and `dataset` share them with the state path, and filing
them under `control` would claim an ownership that does not exist.

**Direction**: the objective and the training loop import `control/`, never the reverse.
`control/` may import `dataset` (`Normalizer` and the window types are data-plane values it
genuinely consumes) but not `train`/`objective`/`validation`/`forecast`/`models`/`batching`.
**That `dataset` edge runs BOTH ways and only one direction is safe**: `dataset` imports
`control.{basis_fit, conditioning, dynamics.inverse, envelope}` to build a batch at all, so
those four must stay `dataset`-free or the cycle closes and fails at import in whichever
order a caller hits first. `tests/test_architecture.py` names the four and pins it.
`batch_contract.py` holds `unpack_batch`/`model_forward`/`anchor_state` and the
`LossComponents` contract so a loss module can read a batch and return an objective without
importing `objective`, which imports it. `tests/test_architecture.py` enforces all of it.

`control/__init__.py` re-exports nothing on purpose — flattening forty names into one
namespace would restore the undifferentiated listing the package exists to remove.

**Every CLI flag is named after the `TSConfig` field it sets** (`--dt-s`,
`--control-rollout-integrator-dt-s`, `--control-state-supervision-clock`, …; fifteen were
renamed on 2026-09-07). `cli/common.CLI_CONFIG_FIELDS` is therefore a list of field names,
asserted against `fields(TSConfig)` at import; `tests/test_architecture.py` checks both
remaining directions — each listed field has a flag spelled as itself, and every parser dest
is either a listed field or one of the frozen `NON_CONFIG_DESTS` (so DELETING a name from
the list fails instead of leaving its flag silently ignored). **Every parser passes
`allow_abbrev=False`**: without it argparse accepts any unambiguous prefix and four of the
renamed flags kept working under their old spellings. Exceptions, all recorded where they
occur — on train: `batch_size` (its flag also takes `"auto"`), `use_norm`/`revin` (one
`--instance-norm`), `control_recipe_name` (resolved against `--config-overrides` first); on
predict, whose config comes from the CHECKPOINT so its four flags are overrides rather than
settings (`cli/predict.PREDICT_CONFIG_FLAGS`, asserted the same way):
`--command-hook` / `--hook-saturation` keep their short names because `CLAUDE.md` names them
as the adopted delivery form and two arm files spell them in re-runnable `predict_args`.

**Run and category naming**: `run_naming.py` is the single source for one grammar —
`output · backbone · dynamics · loss · meta` — rendered from the run's serialized config by
every surface that names a trained run. A default change deliberately shifts old runs' names.
**On-disk run/category directories are historical record — never rename them.** Grammar,
fallbacks and the relabel tooling: `docs/ENGINEERING_NOTES.md`.

## Traps (one line each; evidence in `docs/ENGINEERING_NOTES.md`)

- **The objective must score VELOCITY, not just position** — scoring position at 64 endpoints
  alone let 71 % of predicted bank energy collapse into one profile shared by every flight.
- **Bank was never supervised, and unsupervised it lands BELOW a trivial baseline** — position is
  derivative order 0, velocity order 1, bank order 2, so no term ever named it. `simple-v3` fixes
  it (skill 0.124 → 0.735) at no accuracy cost.
- **Three causes of that bank wiggle were tested and are NOT it** — segment count, training
  budget, conditioning capacity. Do not re-litigate without new evidence.
- **The imitation dose curve is NOT a ramp**; below ~11.8× position it is a noisy plateau and
  past ~47× it overshoots into smoother-than-reality.
- **Loss weights are calibrated, not chosen** — raw velocity and position terms differ by 642× at
  the converged operating point. Over-constraining looks exactly like the blandness trap.
- **`DEFAULT_CV_PATIENCE = 6` is too small here** — both flight models pass through an early ADE
  transient, so patience 6 turns a τ ranking into a stopping artifact.
- **A named recipe cannot be cross-validated as itself** (frozen `epochs`/`patience`).
- **The duration head cannot predict below ~125 s** against a true range starting at 21 s —
  flights anchored close to the runway fly a full loop. Unfixed, present in every flight model.
- **The state model's KRDU endpoints sit ~250 m NW of every runway, and it is the model, not the
  data** — a world-fixed translation present from the FIRST predicted step. Read every arm-A
  per-runway cross-track number with it in mind.
- **`random_train_anchor=True` + the imitation term is a performance cliff** (the per-flight
  inverse would be recomputed per sample per epoch). A note, not a guard — no recipe uses it.
- **Diagnostic scripts that call `model(x)` directly cannot run a corridor-bounded checkpoint** —
  go through `batch_contract.model_forward` with the context row.
- **`StateOutputLayer.offset_mask` is a non-persistent buffer**; `load_checkpoint` drops the key
  if a checkpoint stored it. A buffer that IS learned scale stays persistent.

## Where to go next

| doing this | read first |
|---|---|
| designing an experiment / changing loss, rollout, output layer | `docs/ENGINEERING_NOTES.md` |
| picking up work, checking what a campaign settled | `docs/OPEN_ITEMS.md` |
| putting the procedure constraint into TRAINING as a hard constraint (either path), or the lazy-network / gate question | `docs/2026-09-08_hard_constraints_survey_and_integration_plan.md` (survey with formulas + H0–H6 plan; papers in repo `docs/literature/procedure_hard_constraints/`) |
| mechanism, architecture, result tables, deliberate scope | `README.md` |
| comparing airports or quoting an ADE | `approach_difficulty.py`, repo `docs/2026-08-21_ksjc_route_mix_and_ade.md` |
| anything about vertical datum, velocity seam, flight identity | `flight_scenarios/CLAUDE.md` |
