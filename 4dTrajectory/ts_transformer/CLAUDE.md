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

## The two prediction paths — this is the experiment

`prediction_output` decides whether dynamics is connected at all, and the two answers are the
point of the package, not a migration in progress.

- **`state`** is the purely kinematic BASELINE: channels in, channels out, the only symbol it
  borrows from `aerodynamic_model` being the `GeodeticState` dataclass. Its predictions carry
  **no flyability guarantee** — speeds, turn rates, thrust and `Cl_max` are unchecked. That is
  the survey's "statistically plausible but unflyable" problem, and it is **deliberate**: it is
  what lets the learned component be measured on its own.
- **`control`** is the opposite: the model emits bounded controls and a differentiable RK4
  rollout of the shared point-mass equations turns them into the trajectory, so every prediction
  is dynamically admissible by construction.

Single-aircraft-only and deterministic point-prediction are scope decisions for both (README).

Two orthogonal dynamics axes underneath the control path: `control_dynamics_model` ∈
`point-mass` | `first-order-lag` is the physics; `control_dynamics_backend` ∈ `reanchored-rk4` |
`transport-chart-velocity` | `scaled-transport-chart-velocity` is the state representation the
long rollout carries. **The registry in `control/dynamics/backends.py` is keyed by the PAIR.**
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

## Current defaults and their status

| axis | default | status |
|---|---|---|
| `coordinate_frame` | `enu` | keep — the airport frame makes the model average across parallel pairs |
| `state_position_reference` | `absolute` | `corridor-bounded` ADOPTED as candidate default (4 seeds, no regression); `anchor-relative` vetoed by its own pre-registered rule |
| control recipe | `simple-v3` | = `simple-v2` + `control_imitation_loss_weight`; **its weight 64.0 does NOT transfer between airports — recalibrate per airport** |
| `control_dynamics_model` | `point-mass` | `first-order-lag` buys smoothness + 3.4 % ADE; τ=2.0 s is defensible, not CV-selected |
| procedure penalty (state + control) | weights at 0 | NOT adopted — kept as an option |
| command hook | off in training | **`predict --command-hook barrier --hook-saturation soft` is the ADOPTED use**; no arm trained THROUGH a hook beat its predict-time counterpart (six tried) |
| `--project-final` | off | deployment fallback; FAF-gated wrecks vectored flights |
| `target_conditioning` | off | `channels` helps only the duration head; PatchTST refuses it |

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
prefix: `envelope`, `heads`, `duration`, `conditioning`, `dynamics/{backends,rollout,inverse}`,
`loss/{components,terminal_clock,fixed_dt,regularization}`, `training/{curriculum,diagnostics}`,
`constraints/{barrier_filter,nominal_residual,gates}`, `oracle/*` (which absorbed the old
`oracle_teacher/` package — two halves of one idea).

**Membership rule**: a module belongs in `control/` only if EVERY consumer of it is
control-specific. `prediction_outputs` (holds `StatePrediction`), `terminal_state_loss`,
`arc_length_geometry`, `fixed_dt_supervision` and `flyability` therefore stay at the top
level — `fixed_anchor_validation` and `dataset` share them with the state path, and filing
them under `control` would claim an ownership that does not exist.

**Direction**: the training loop imports `control/`, never the reverse. `control/` may import
`dataset` (`Normalizer` and the window types are data-plane values it genuinely consumes) but
not `train`/`forecast`/`models`/`batching`. The oracle takes `train`'s objective dispatch as
an injected `loss_components` argument for exactly this reason. `batch_contract.py` holds
`unpack_batch`/`model_forward` so the train-only oracle can read a batch without importing
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
