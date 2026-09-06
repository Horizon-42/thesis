# ts_transformer — learned trajectory prediction

Vendored iTransformer + PatchTST (torch). Predicts the remainder of an arrival from a
threshold-anchored lookback window.

**This file is the map: the contracts that break things if violated, and a one-line trigger for
every trap.** The evidence behind each line — measurements, campaign results, the causes already
ruled out — lives in `docs/ENGINEERING_NOTES.md`; status and next steps in `docs/OPEN_ITEMS.md`;
mechanism and result tables in the package `README.md`; history in the repo's `docs/CHANGELOG.md`
(2026-07-19, 07-20 ×2).
Read the notes before designing an experiment or touching the loss, rollout or output layer.

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
- **A new loss term must be added to `loss_component_names`**, not only to the objective's
  `extras` — otherwise `KeyError` on the first batch, *after* the slow dataset build.
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
| `state_position_reference` | `absolute` | `corridor-bounded` ADOPTED as candidate default (4 seeds, no regression); **`anchor-relative` is VETOED by its own pre-registered rule.** It follows the package's one mechanism for a value like this: it is in `STATE_POSITION_REFERENCES` (what a STORED config may say, so the 2026-09-03 `state_v2_20260903/A_anchor_relative` artifact still loads and names) and NOT in `STATE_POSITION_REFERENCES_AVAILABLE` (what a NEW run may select — the CLI's choices, and what `__main__._refuse_unavailable_selection` checks so `--config-overrides` cannot get past it either). `control_command_hook="nominal-residual"` is the same pair |
| control recipe | `simple-v3` | = `simple-v2` + `control_imitation_loss_weight`; **its weight 64.0 does NOT transfer between airports — recalibrate per airport** |
| `control_dynamics_model` | `point-mass` | `first-order-lag` buys smoothness + 3.4 % ADE; τ=2.0 s is defensible, not CV-selected |
| procedure penalty (state + control) | weights at 0 | NOT adopted — kept as an option |
| command hook | off in training | **`predict --command-hook barrier --hook-saturation soft` is the ADOPTED use**; no arm trained THROUGH a hook beat its predict-time counterpart (six tried) |
| `--project-final` | off | deployment fallback; FAF-gated wrecks vectored flights |
| `target_conditioning` | off | `channels` helps only the duration head; PatchTST refuses it |
| `latent_dim` | 0 | the latent intent (L2); `latent_prior_components` / `latent_beta` / `latent_free_bits_nats` mean nothing without it and are refused |
| `cta_conditioning` | `off` | `given` = the CTA is the duration (L3); a delivery-form demonstration, never a prediction result |
| `n_segments` (control) | 64 | **32 is free** (L1: 1322 vs 1333 m, bank skill 0.726 vs 0.728); the deployed head's 257 numbers become 96 |
| `control_state_loss_grid` | native | `fixed-dt` without the imitation term (it is not registered there) trips the straight-in veto (FDE 703 → 2863) and brings the bank wiggle back — the trajectory-error loss alone is not enough |
| `control_heading_rate_loss_weight` (+ `_scale_dps` 1.5) | 0 | L1.b arm ②: the teacherless way to name the bank — the rollout's own ψ̇ at the segment endpoints against the flown track's. `_scale_dps` is the UNIT the residual is read in (half a standard-rate turn), not the dose. Doses 1.0 / 8.0 bracket an unknown; **not measured yet** |
| `control_bank_tv_loss_weight` | 0 | L1.b arm ③: mean \|step\| between adjacent COMMANDED banks, in half-box units. **It prices REVERSALS, not slope** — `sign(x)` cancels on a monotone run's interior — and exact flatness is a STATIONARY POINT (value 0 and gradient 0), which is where the zeroed head init starts every run, so it never leaves a flat schedule on its own. "TV changed nothing" is that before it is a dose. **Not measured yet**. Both terms register under `true-time-position` ONLY (where `velocity`/`imitation` live) and `TSConfig` REFUSES a non-zero weight under any other objective rather than ignoring it |
| `control_imitation_target` | `inverse-dynamics` | WHAT the imitation term imitates. The default's schedule, flown open-loop, lands **2.5–7.8 km** from the truth it was read off (L0), so "imitating it perfectly" is not "flying the truth". `fitted` reads the per-flight table `run_ts_control_basis_oracle.py --checkpoint` fits through the same rollout (88–433 m) and needs `control_fitted_teacher_path` (which names the run: `teacher=<dir>/<file>`); refused with `random_train_anchor`, with `control_imitation_loss_weight=0`, and off `control_state_objective=true-time-position` (the only objective the term is registered under). L5.a, **not yet measured** — the arms are `docs/experiments/l5_fitted_teacher_arms.json` |

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

Control-specific code lives in **`control/`**, by role rather than behind a `control_`
prefix: `envelope`, `heads`, `conditioning`, `latent`, `basis_fit`,
`dynamics/{backends,rollout,inverse,hooks}`, `loss/{components,fixed_dt}`,
`training/diagnostics`, `constraints/{barrier_filter,gates}`.

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

- `run_ts_anytime_curve.py` — **A0-fixed**: replays `--checkpoint LABEL=PATH` (repeatable) from
  a grid of REMAINING-PATH anchors and reports ADE / FDE / |Δt| per bin per stratum. Three
  things it exists to keep right: the strata are computed once at **L−1 and fixed for every
  bin** (relabel per bin and a flight leaves the vectored stratum exactly when it rolls out on
  final — a survivor curve reads as an improving one); the bin coordinate is
  `approach_difficulty.remaining_path_profile_m`, which the covariate itself reads, so a
  consumer binning on distance-to-go never restates the arc length; and the reading carries the
  out-of-distribution cost of an L−1-trained checkpoint, so it is **never quoted without an
  A0-random arm**. `cta_conditioning=given` is refused (its duration IS the truth's).
  **`--min-future-s 60` empties the 2 km bin and most of the 4 km one** (≈27 s / 53 s of truth
  left at approach speed) — stated as `n=0 / partial`, but s_freeze can then only be read at
  ≥ 6 km.
- `run_ts_eta_error_readout.py` — **B0**: |`final_time_error_s`| p50/p80/p90 and the SIGNED
  p10/p50/p90 (same for `fde_m`) per stratum, straight out of existing `summary.json` files.
  Measured 2026-09-07 on KRDU val: vectored |Δt| p80 **65.8–72.5 s** against straight-in
  **12.0–20.3 s**, so one pooled ETA interval cannot serve both strata.

**A replay runner must fingerprint the data the way the checkpoint was trained** — with the
pre-split lateral-pass roster (`lateral_eligibility.default_lateral_pass_roster_path`) when the
checkpoint's provenance carries an `eligibility` block. Without it the v5 cohort fingerprints as
14 435 KRDU arrivals against the checkpoint's 14 378 and `require_matching_data_provenance`
reports "the manifest changed". `run_ts_control_basis_oracle.py --checkpoint` still has the
plain call and will hit this on every v5 checkpoint (`docs/code-health-followups.md`).

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

**Direction**: the training loop imports `control/`, never the reverse. `control/` may import
`dataset` (`Normalizer` and the window types are data-plane values it genuinely consumes) but
not `train`/`forecast`/`models`/`batching`. The oracle takes `train`'s objective dispatch as
an injected `loss_components` argument for exactly this reason. `batch_contract.py` holds
`unpack_batch`/`model_forward`/`anchor_state` and the `LossComponents` contract so a loss
module or the train-only oracle can read a batch and return an objective without importing
the loop it runs inside. `tests/test_architecture.py` enforces all of it.

`control/__init__.py` re-exports nothing on purpose — flattening forty names into one
namespace would restore the undifferentiated listing the package exists to remove.

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
| mechanism, architecture, result tables, deliberate scope | `README.md` |
| comparing airports or quoting an ADE | `approach_difficulty.py`, repo `docs/2026-08-21_ksjc_route_mix_and_ade.md` |
| anything about vertical datum, velocity seam, flight identity | `flight_scenarios/CLAUDE.md` |
