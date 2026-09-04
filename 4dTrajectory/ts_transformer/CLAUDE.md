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
- **The chart origin is NOT "the threshold"; `FlightSeries.target_chart` is.** Three frames:
  `enu` / `runway-aligned` anchor at the assigned threshold (target_chart ≡ 0) and
  `airport-enu` anchors at the airport reference point (`AirportENUFrame`, resolved from the
  same `runway_thresholds.json` entry the harvest reads). Every consumer that judges
  distance-to-go — the observed crossing plane, `forecast.truncate_at_threshold`, the
  fixed-anchor common-grid truncation that SELECTS checkpoints, `approach_difficulty`,
  `horizontal_distance_m` — measures from `target_chart`; a new one that reads `hypot(e, n)`
  is silently wrong under the airport frame and only there (2026-09-03).
- **`target_conditioning="channels"` appends five INPUT-ONLY channels** (`e/n/u_tgt`
  standardised with the position stats, `cos/sin ψ_rwy`) after normalization —
  `config.input_channels` is what the model sees, `config.channels` what it predicts,
  `enc_in` is the input width, and the checkpoint stores `input_channels` beside `channels`
  (`load_checkpoint` refuses a mismatch). iTransformer consumes them as covariate variate
  tokens (the vendored `x_mark_enc` path); PatchTST is refused (channel-independent). Every
  `x[:, -1]` anchor read goes through `batch_contract.anchor_state` so the conditioning
  columns never pose as state. Measured: the attention backbone ignores them for the
  trajectory; only the flattening duration head reads them (−10 % time MAE).
- **`state_position_reference`** (state output only): `absolute` (default, state-v1) or
  `anchor-relative` — `StateOutputLayer` adds the anchor's normalized position back onto the
  forecaster's position channels, so the network's zero output means "stay where the
  aircraft is". Measured 2026-09-03 (`docs/2026-09-03_state_v2_anchor_relative_results.md`):
  it removes the first-step translation (KRDU straight-in lateral miss +204 → +21 m, FDE
  643 → 492 m) but gives up the absolute output's implicit "end at the origin" prior —
  KRDU vectored FDE +350 m on both seeds, endpoint lateral p95 ×1.4 — so it is NOT the
  default. The two parametrizations are two priors; the next candidate must keep both.
- **`state_position_reference="corridor-bounded"` (state-v1 + the final-approach corridor by
  construction, measured 2026-09-04/05, ADOPTED as a candidate default —
  `docs/2026-09-05_final_constraint_results.zh.md`).** `StateOutputLayer` decodes with the
  normalizer statistics it carries as buffers, reads each row in runway axes
  (`final_approach_geometry`: `d` back from the threshold, `xt` right of the course), and on the
  rows the output itself places on the final — inside the full-scale LPV cone floored at 500 m
  AND the predicted PATH (position differences, never the unsupervised velocity channels)
  within 30° of the course — saturates `xt` inside `±k·hw(d)` and the height inside the
  `[−60, +120] m` glidepath window (`corridor_gate="on-final"`; `"faf"` binds by distance and is
  the ablation). Four seeds (KRDU/KSJC × 1337/2024): pooled FDE −51…−91 m, pooled ADE
  −8…−71 m, straight-in FDE −18…−96 m, corridor-violation rows on truth-established rows
  77→48 % / 34→21 %, no seed regresses; it does NOT touch the start of the path (first-step
  jump unchanged) and the KRDU endpoint now sits ON the corridor edge (+54 m) — the NW pull is
  capped, not removed. Needs a per-flight context row in the batch (`FINAL_APPROACH_KEYS`:
  runway course, tan GPA, FAF distance) — `dataset.batch()` puts it in the context slot for
  state recipes, `forecast` builds it per series, and the `enu` chart is REQUIRED
  (`TSConfig` refuses the others). Diagnostic scripts that call `model(x)` directly
  (`batch_benchmark`, `run_ts_overfit_diagnostic`, `run_ts_predictability_report`) cannot run a
  corridor-bounded checkpoint; go through `batch_contract.model_forward` with the context.
- **The procedure penalty on the CONTROL rollout (2026-09-05, `control_procedure_20260905`,
  simple-v3, one seed, KRDU + KSJC, openap-direct cohort — NOT the state campaign's flight
  set; `docs/2026-09-05_control_penalty_results.zh.md`) improves the endpoint and worsens the
  path middle.** λ = 1e-3 on the segment-endpoint rollout: pooled FDE −101 / −39 m, straight-in
  FDE −50 / −58 m, endpoint |xt| p95 2769→2203 m at KRDU (KSJC's tail does not improve),
  first-step offset ≈ 0 by construction — but vectored ADE +581 / +245 m (the rollout is
  sequential, so a penalty on the last rows reaches every earlier control and the cheapest fix
  is to turn earlier) and the corridor-violation rate barely moves (55→45 %, 27→25 %; ~60 % of
  the remaining rows are vectored flights kilometres away when the truth is on the final,
  the rest straight-ins). λ = 5e-3 collapses the bank schedule at KRDU (common-profile share
  3.8→13.7 %, bank skill 0.728→0.280). Relative to its own position term the hinge on rollout
  rows is 5.6× the state path's at epoch 1 and 2.4–2.9× at the end, so the state-path "parity"
  weight is not parity here. Not adopted; kept as an option. The control
  path's own mechanism is the rollout **command hook** below (design:
  `docs/2026-09-05_control_constraint_design.zh.md`).
- **Command hooks (`control_command_hook`, 2026-09-06): a constraint module called once per
  control SEGMENT, at its start, with the rollout's own state, returning the command flown.**
  Per segment, never per substep — the hand-written adjoints cover whole schedules, so the
  hooked engine integrates one segment at a time and the hook's state dependence is plain
  autograd (`aerodynamic_model.torch_piecewise_rollout.rollout_piecewise_constant_hooked_with_step`).
  Only the first-order-lag backends support it (`_refuse_hook` on the point-mass ones); the
  dense rollout settles the effective schedule first and re-integrates it; **the record
  carries the schedule FLOWN**, not the network's (`forecast` exports
  `rollout.controls`, `source.commandHook` = `hook/saturation`). Two modules in
  `control/constraints/`: `barrier` (corridor barriers → heading interval → bank interval,
  lateral only) and `nominal-residual` (L1 + glidepath law, tanh-bounded residual, both axes),
  sharing `gates.py` (on-final on the rollout's velocity). Gotchas: (1) the command is HELD
  for Δt, so every rate gain is `min(gain, 1/Δt)` — `RolloutStateView.duration_s` exists for
  this; (2) `hard` = hard saturation AND hard gate, training refuses it, so a trained arm is
  compared with the SOFT predict-only arm (`F_barrier_infer_soft`) for "training through the
  hook" and with the hard one for hardness; (3) the heading layer is a continuous barrier
  pair — a version that acted only outside the interval clamped a centred command to zero
  bank; (4) diagnostics are per-step shares in `EpochResult.command_hook` (`gated`,
  `clamped` / `*_residual_saturated`, `bank_change_rad`), the "lazy network" reading is
  clamped > 20 % with bank skill below baseline; (5) **a hook that changes the bank must
  re-coordinate the load factor** (`n' = n cos μ / cos μ'`) or it steals the vertical lift the
  network paired with its bank (v1 measured: −1° → −10° path angle, 93 → 200 m/s, 16 km past
  the threshold on one flight), and **a rate rule on a held command with a lagged actuator
  limit-cycles** — the barrier's heading layer therefore asks for a heading CHANGE over the
  hold, credits the bank already flown (`state.actuators[:, 1]`) and evaluates the corridor
  margins at the lag-lead position (v2, 2026-09-06; heading gain default 0.3 → 0.1);
  (6) **the nominal law needs a third law on thrust, and the network's intent is only in
  its own rollout** — a trim load factor reads as "hold the current path angle" at ANY
  path angle, so a hook that steers the glidepath cannot recover the schedule's intended
  speed from the segment's command (v1 measured: pulled up to the glidepath, 58 vs 88 m/s,
  584 m short). A hook declaring `needs_reference = True` gets `RolloutStateView.reference`,
  the unhooked schedule's state integrated alongside (an endpoint rollout per segment
  more: ~2× the hooked rollout's wall time and adjoint memory), and the nominal law holds
  thrust to that rollout's speed (`control_nominal_speed_gain`, 0.1/s, capped at `1/Δt`
  like every other rate gain — uncapped it bangs between the envelope corners at 8
  segments).
  Predict-time: `predict --command-hook barrier --hook-saturation soft` on any lag checkpoint
  — **this is the adopted use (2026-09-06, `docs/2026-09-06_control_hooks_results.zh.md`)**:
  the v2 soft barrier applied at prediction to the `simple-v3` baseline is a net gain at both
  airports (KRDU pooled FDE 1650 → 1449 m, ADE 1333 → 1278, FDE better on 84 % of flights and
  none worse by 1 km; KSJC 996 → 930 m, 90 % better; straight-in endpoint |xt| p95 1821/407 →
  70/46 m), soft beats hard everywhere (the hard gate's jump at the cone edge: 10 KRDU
  flights > 1 km worse), and **no arm trained THROUGH a hook beat its predict-time counterpart**
  (six: barrier v1/v2 + nominal v1/v2 at KRDU, barrier v2 + nominal v2 at KSJC; pooled ADE
  +2…+21 %, bank skill below the baseline, vectored path middle worse — the network leans on
  the hook; the pre-registered lazy veto fires at KRDU). The nominal-law hook is the vertical
  complement (KRDU endpoint height above the threshold −164 → −7 m median, vertical violation rows 46.6 → 29.1 %,
  small lateral gain) and exposes a baseline fact: the control baseline ends 157 / 162 m BELOW
  the glidepath (median, KRDU / KSJC). Next: a combined hook (barrier
  lateral + nominal vertical) at predict time. Campaigns `docs/experiments/control_hooks_arms.json`
  → `control_hooks_20260906` (v1, commit cd981f4; its trained `F_barrier_soft` checkpoint
  predates v2 and would run v2 at predict time — quote its `_pred_val` records, do not
  re-predict it) and `control_hooks_v2_arms.json` → `control_hooks_v2_20260906` (KRDU + KSJC);
  `readout.json/.txt` + `score.txt` live in each campaign dir.
- **The procedure PENALTY is not the way on the state path (same campaign).** `procedure_loss_*` (runway-scale
  hinge² on truth-gated rows, `ProcedureMultipliers` fixed or dual-ascended on the violation
  rate) is kept as an option with the weights at 0. Dual ascent toward `epsilon=0.05` diverged
  on all four runs (λ ×50 in 74 epochs, common-grid ADE 2420/2749 at KRDU, 2258/2195 at KSJC vs
  1383/1361/870/865): the tolerated rate must be REACHABLE (arm A sits at 0.77, the bounded
  output at 0.48) or the multiplier is a ramp, and the hinge is ~20× larger early in training
  than at the converged operating point the calibration used. At the calibrated parity weight
  (1e-3, no dual) it buys the same violation-rate drop as the bounded output but pays 42 m of
  pooled ADE for it (KSJC).
- **Inference-time projection (`predict --project-final GATE`) is row-by-row, the hard
  counterpart of the bounded layer's gate, and at KRDU it recovers most of arm B's FDE gain
  post hoc** (pooled FDE 1163→1090 vs B 1072, straight-in 643→545 vs 548) but not the
  violation rate (56 % vs 48 %) nor the endpoint |xt| p95 (480 vs 305 m), and less at KSJC
  (FDE 748 vs 725). A first version bound only the suffix from which every later row was on
  the final; it moved 1.35 % of the rows (one off-final row cancelled the tail) — a gate that
  is not the layer's gate is not a ceiling for it. FAF-gated it is the straight-in ceiling
  (KRDU FDE 643→455) at the cost of vectored flights (ADE 2599→4665 at KRDU, 2536→5281 at
  KSJC where 30L's FAF is 15 km out).
- **`StateOutputLayer.offset_mask` is a non-persistent buffer** (a pure function of the channel
  contract) and `load_checkpoint` drops that key if a checkpoint stored it — both the
  2026-09-03 generations (arm A without it, state-v2 with it) load. A new buffer that IS
  learned scale (`channel_mean/std`) stays persistent.
- **Horizon trap**: the horizon was sized from the MEASURED duration distribution (p50 328 s /
  p95 651 s), covering **97.8 %** of flights — the old "an arrival is ~3.5–5 min" straight-line
  estimate was WRONG (real arrivals are vectored), do not resize from it. The ~2 % over the
  horizon are cut at H and flagged `horizonCapped`, so their gate verdicts are cap artifacts, not
  model error.
- `summary.json` carries an `accuracy` block (mean AND p95) plus per-row `ade_m`/`fde_m`;
  `overlap` is a REQUIRED arg to `write_batch` — an optional metric is one that silently goes
  missing.
- **A per-airport ADE without its ROUTE MIX is not a comparison** (`approach_difficulty.py`).
  Every row carries `route_tortuosity`, `remaining_path_m`, `anchor_range_m`,
  `anchor_cross_track_m`, `established_at_anchor`, and `accuracy.difficulty` carries the
  batch mix + the thresholds the flag encodes. Measured 2026-08-20 on the pooled checkpoint:
  **inside a matched stratum every airport scores the same** (412–509 m median ADE on
  "straight, <13 km left"), and the whole spread between airports is the share of flights in
  that stratum — 78.6 % at KSJC against 41.8–61.0 % elsewhere. Reweighted to the pooled mix
  KSJC goes **483 → 1526 m, from best of five to worst**, and on its own vectored flights it
  IS the worst (3931 m vs 2000–2249 m at KSTL/KRDU). The signature to recognise: ADE and
  cross-track improve while **FDE does not** (KSJC 1000 m vs KRDU 1019 m) — a straight route
  makes only the lateral channel easy. Covariates come from the OBSERVED track the error is
  scored against, never the prediction, and are frame-independent (world EN, not chart axes).
  → `docs/2026-08-21_ksjc_route_mix_and_ade.md`
- Aircraft-type resolution (`_resolve_aircraft`, `--aircraft-type`, why `"type": "UNK"` does not
  mean single-type): see `flight_scenarios/CLAUDE.md`.
- **Controls are DIMENSIONLESS in this package** (`control/envelope.py`, the single source):
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
  long rollout carries. The registry in `control/dynamics/backends.py` is keyed by the PAIR.

## Run & category naming — `run_naming.py` is the single source

Every surface that names a trained run (frontend category labels + Experiments picker,
publication manifests, `INDEX.md`) renders ONE grammar derived from the run's serialized
config: `output · backbone · dynamics · loss · meta`. Loss = named recipe, else
**nearest recipe + edits** (`simple-v3+(imit=16)` — fewest loss-field edits, later recipe
wins ties), else `custom-<8-hex>`; meta = fields deviating from TODAY'S defaults (seed
first, capped at 6, `+N more`) plus caller extras (run id / campaign/arm / cohort) —
so a default change shifts old runs' names, deliberately. The frontend prefers the
stamped `experiment.label` and falls back to composing from metadata for pre-label
publishes. Relabeling published categories is metadata-only and never touches
records/checkpoints: `publish_ts_experiment_trajectories.py --refresh-labels-only`
(publisher-managed, from stored publication manifests) and
`docs/relabel_published_categories.py` (the 2026-08-24 one-off for legacy `ts_*` keys).
**On-disk run/category directories are historical record — never rename them**;
`run_slug()` is the grammar's filesystem form for FUTURE directories.
Worked vocabulary — every KSJC label expanded, slot by slot, with the campaigns'
meaning and codebase pointers: `docs/2026-08-24_ksjc_result_labels_explained.md`.

## Layout

Control-specific code lives in **`control/`**, by role rather than behind a `control_`
prefix: `envelope`, `heads`, `duration`, `conditioning`, `dynamics/{backends,rollout,inverse}`,
`loss/{components,terminal_clock,fixed_dt,regularization}`, `training/{curriculum,diagnostics}`,
`oracle/*` (which absorbed the old `oracle_teacher/` package — two halves of one idea).

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

## Gotchas (recurring, verified)

- **`simple-v3` is the control recipe to use.** It is `simple-v2` plus one field,
  `control_imitation_loss_weight = 64.0` (~47× the position term), which supervises the
  control schedule directly — see the bank entry below for why that was the missing piece and
  what it measures. `simple-v2` remains the correct reference point for anything that predates
  it, and everything in the next paragraph is still true of it.
- **The objective must score VELOCITY, not just position** (this is what `simple-v2` added). simple-v1's `true-time-position` objective scored position at 64 endpoints and
  nothing else, so a rollout was free to thread the right places on any route between them —
  and it did: **71 % of the predicted bank energy was one profile shared by every flight**
  (flown tracks: 3 %), that profile was not even the population mean (the flown tracks average
  ~0°, as left and right turns must cancel; the model's swung −5.9 to +10.8°), and per-flight
  bank skill was **−0.073**, i.e. nil. Straight-in references were flown with 3.65° RMS bank
  and 7 sign reversals where inverting the flown track asks for 0.55° and none. Adding
  `control_velocity_loss_weight` fixes it: 17.3 % shared, 0.79°, skill **+0.197**, and ADE
  better on **77.8 %** of flights (median −58.2 m, p=7.9e-31→4.7e-79). Smoothness and accuracy
  move TOGETHER, which is what separates this from the flyability blandness trap.
  **The weight is calibrated, not chosen**: the raw velocity and position terms differ by
  **642×** at the converged operating point, so 0.003 puts velocity at ~2× position. Past 8×
  the bank drops BELOW the flown tracks' own 0.55° while FDE climbs 818 → 1231 m — over-
  constraining looks exactly like the blandness trap, and only watching accuracy alongside
  shows it.
- **Three plausible causes of that wiggle were tested and are NOT it** — do not re-litigate
  them without new evidence. Segment count (halving N moves the shared share 12 pp and is not
  even monotone); training budget (the two ways of adding optimizer steps DISAGREE — `lr 1e-4`
  makes straight-reference bank *worse* than baseline, so more optimisation just fits the
  shared profile harder); conditioning capacity (depth does nothing; width was the best
  non-loss axis at 70.7→48.8 % but per-flight skill stayed exactly 0.000, and **with the
  velocity term present width adds nothing at all** while costing ADE on 71.8 % of flights).
  A fourth, `_initialize_control_head`'s zeroed weight, genuinely starves the backbone of
  gradient (0.000e+00, 20/20 tensors) but is also not the cause: seeding it raised the shared
  share to 81 % and regressed ADE on 84.8 % of flights. Full write-up with figures:
  `docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md`.
- **The duration head cannot predict below ~125 s** against a true range starting at 21 s, so
  a handful of flights whose anchor is already close to the runway (0.4 % on KSJC) fly a full
  loop — the rollout runs five minutes when the threshold is thirty seconds away. Separate
  from the bank wiggle, present in every flight model, unfixed.

- **The teacher inverse must be the inverse OF THE CONFIGURED FORWARD MODEL, and nothing else
  can catch it if it is not.** A schedule solved against the wrong equations is finite, bounded,
  the right shape, and its own optimizer reports a falling loss — it simply reproduces nothing.
  So `control/dynamics/inverse.py` registers each inverse under the SAME config key as its
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
- **The lagged model buys smoothness, and the honest reading is "artifact removed", not "more
  realistic".** Paired on the same 1083 KSJC validation flights (2026-08-19): jerk p95 **−28 %**
  (lower on 99.8 % of flights), turn rate −6 %, acceleration −5.5 %, and ADE **−3.4 %** (better
  on 67.4 %, p=7.9e-31) — smoothness and accuracy move TOGETHER, which is what separates this
  from the blandness trap flyability falls into. FDE and the arrival endpoint are ties (49 %): a
  lag smooths the turn geometry, not where the model thinks the runway is. But both models were
  already smoother than the flown tracks (jerk p95: observed 4.25, point-mass 0.30×, lag 0.21×),
  so the lag moves them FURTHER from observed statistics. Observed jerk is 2 s ADS-B positions
  differentiated three times — mostly quantisation noise, not a target. Gates are unchanged at
  **0 pass / 1083 fail** on both arms.
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
  `point-mass` rather than multiplying the grid by 5 for identical folds. **`DEFAULT_CV_PATIENCE
  = 6` is too small here**: both flight models pass through an early ADE transient (point-mass:
  1534 m at epoch 2 → 2132 m at epoch 6 → 1268 m at epoch 13, still falling) while every loss
  component falls monotonically. Patience 6 stops inside it and the τ ranking becomes a stopping
  artifact — measured, the same τ scored 1674.6 m on a caught fold and 1234.4 m on one that was
  not. The frozen recipes' patience=20 clears it; a sweep must raise `--cv-patience` to match.
  The τ sweep itself came out **unresolved**: best-to-worst 5.7 % against 11–23 % fold noise, so
  τ=2.0 s is a defensible default, not a CV-selected value.
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

- **Bank was never supervised, and unsupervised it lands BELOW a trivial baseline —
  `simple-v3` is the fix.** Position is
  derivative order 0 and `control_velocity_loss_weight` is order 1; bank lives at order 2, so no
  term in the loss ever named it. Measured consequence: the model's bank carries *less*
  information about the flown bank than a randomly chosen OTHER flight's does — per-flight skill
  +0.197 vs a random-flight floor of +0.312 on KSJC, +0.124 vs +0.170 on KRDU. The flown
  population's bank residual is dominated by one mode (55.2 % of the energy, same sign in 86 % of
  flights — the turn onto final); the model concentrates its bank into a single mode too, but a
  nearly orthogonal one (alignment 0.062). It learned that banking happens, never when.
  **Always read bank skill against the floor and the same-runway twin ceiling that
  `docs/score_control_arms.py` now prints per arm — never against 1.0**, which is unreachable
  because the entry state only partly determines the future. Doing so inverts the earlier
  loss-design reading: the velocity dose frozen into `simple-v2` is the only one below the floor.
  Measured on 1404 KRDU validation flights, `simple-v3` takes per-flight bank skill 0.124 →
  **0.735**, the flight-independent share 49.0 % → **3.3 %** (KRDU's own flown tracks: 1.8 %),
  straight-in bank RMS 3.92° → **0.36°** (0.41°), sign reversals 5 → **0**, and ADE on **57.0 %**
  of flights (median 656 → 501 m, p=1.9e-7) with FDE unchanged — unlike the velocity term,
  this structure costs no accuracy.
- **The inverse's transport term (ω×v) is UNCONDITIONAL, and that is a measured result.**
  It used to be gated on `frame_params is not None` while `_transport_rate` never read that
  array's values — a flag wearing a data parameter's clothes — so the scoring scripts
  silently dropped it while the training target kept it, and the measured "truth" bank was
  not the quantity being trained toward. The obvious repair, keying it on the backend, is
  **wrong**: `reanchored-rk4` writes no transport term in its RHS (it re-anchors into
  geodetic state each substep instead), so it reads as transport-free, but its rolled
  trajectories invert **~50× more accurately WITH the term** (load residual 9.7e-6 vs
  5.3e-4). The inverse works in a local ENU frame and a curved-earth trajectory carries ω×v
  there no matter how the forward integrator is written. There is no correct "off", so there
  is no longer a way to ask for one.
  **The round-trip test cannot check any of this** — the term moves recovered bank by
  ~0.007° against a 0.5° tolerance, and a mutation mislabelling a backend passes it
  unchanged. `test_the_transport_term_is_required_by_every_backend` zeroes the term and
  requires the fit to get 10× worse; it catches a dropped term and a flipped sign, but NOT
  a spherical-earth simplification (eccentricity is ~0.3 % of the radius, far below its
  margin). Note `dynamics_arrays`' own `frame_params` entry is unrelated and is a real
  rollout input.
- **`simple-v3` replicates across airports; its WEIGHT does not.** On KSJC the mechanism is
  if anything stronger (bank skill 0.197 → **0.678**, past that airport's 0.543 twin, better
  on 94.7 % of flights, p=1e-230) — but the same 64.0 overshoots: straight-in bank 0.18°
  against a flown 0.53°, and **FDE degrades on 68 % of flights** (p=9e-33) where KRDU paid
  nothing. The cause is the already-measured channel split — bank carries 18 % of the
  box-normalised term at KSJC against 41 % at KRDU — so **recalibrate the weight per airport
  off that split; do not inherit 64.0**.
- **The imitation dose curve is NOT a ramp, and the sign test is not an effect size.** Below
  ~11.8× position the ladder is a noisy plateau (the 1.47× arm came out worse than 0.74× on
  every metric), so sampling only that region concludes the term barely works — the geometric
  1/4/16/64 ladder is what made the effect visible. Past 47× the fit saturates and starts
  overshooting: at 188× straight-in bank is 0.24° and the shared share 3.0 %, both past the
  flown tracks' own 0.41° — smoother than reality rather than closer to it, where 47× sits
  12 % below. **The truth values are per-airport and must be measured, never assumed**:
  `score_control_arms.py` now prints them as a "flown tracks" row under each metric, because
  they used to be hardcoded to KSJC's 3.2 % / 0.55° and every KRDU number was read against
  the wrong reference.
  And at n=1404 the paired sign test returns **p = 3e-16 for pure seed noise**, so read
  magnitudes, never p: seed noise moves bank skill 0.019 and ADE 1.7 m, the dose effects 3-8×
  and 13-52× that.
- **`control_imitation_loss_weight` supervises the schedule directly**, against
  `control/dynamics/inverse.py` — the same registry the forward model dispatches on, so target and
  rollout can never be different equations. The target is built in
  `dataset.reference_control_supervision` on the training-only `_dynamics_arrays` path;
  **`dynamics_arrays()` itself must stay free of it** because forecast/predict call it and there
  is no future to invert there. Each channel is divided by half its box width
  (`control.envelope.CONTROL_HALF_WIDTH`), which on KRDU splits the term 57 / 41 / 2 % across
  thrust / bank / load — on KSJC it is 82 / 18 / 1, one reason KRDU is the better testbed.
  Calibration convention: at the converged KRDU baseline (`state` = 0.0417, unweighted term
  0.0308) **w = 1.36 is 1× the position term**.
- **A new loss term must be added to `loss_component_names`**, not only to the objective's
  `extras`. `fit_model` builds its accumulator from that list and then indexes it with whatever
  keys came back, so an undeclared term raises `KeyError` on the first batch — *after* the dataset
  build, which is the slow part. Covered by a test now.
- **`random_train_anchor=True` + the imitation term is a performance cliff.**
  `FixedAnchorTrajectoryWindows` caches `_dynamics_arrays` once, so the per-flight inverse is paid
  at construction; `RandomAnchorTrajectoryWindows` does not override it, so the inverse would be
  recomputed per sample per epoch. No current recipe uses random anchors, so this is a note, not a
  guard.

## Open items

- **Control command-hook campaigns DONE 2026-09-06 (`control_hooks_20260906` v1 at KRDU,
  `control_hooks_v2_20260906` at KRDU + KSJC; report
  `docs/2026-09-06_control_hooks_results.zh.md`).** Adopted: the v2 soft barrier as a
  predict-time safety layer; not adopted: any hook inside the training loop (six arms, none
  beat its predict-time counterpart), the hard gate. Open: the combined lateral-barrier +
  vertical-nominal hook at predict time; the baseline ending 157 / 162 m below the glidepath
  (data or model?); a "committed to the final" gate for the vectored flights the v1 / hard
  barrier hurt (gate opening at d < 8 km or ≥ 16 km; every bin is net positive under v2 soft); a second seed at KSJC (its −66 m FDE gain is the smaller
  effect); PatchTST and the other three airports.
- **Final-approach constraint campaign DONE 2026-09-04/05 (`final_constraint_20260904`, KRDU +
  KSJC, 3 predict-only + 5 trained arms per airport; report
  `docs/2026-09-05_final_constraint_results.zh.md`, readout `docs/compare_constraint_arms.py`).**
  Bounded output adopted as candidate default (see the config entry above); penalty vetoed;
  projection kept as deployment fallback. Not done: making `corridor-bounded` THE default
  (decide together with the state-v3 continuity term, which addresses the start of the path
  the corridor does not), PatchTST, control output.
- **Procedure constraints in the learned model (2026-09-04 design + measurement):**
  measured on every 3rd rostered arrival (`docs/measure_procedure_adherence.py`) that **0.0 %**
  of observed KRDU/KSJC flights pass an off-axis IAF of their runway's RNAV(GPS) procedure,
  that 85–97 % (KRDU) / 38–83 % (KSJC) are established in the k=0.5 LPV cone by the FAF,
  and that once established 87–99 % of samples sit inside the cone and the −60/+120 m
  glidepath window (the ±22 m gate is met over the whole final by only 14–69 %). So the
  only data-consistent procedure constraint is the final segment (corridor + glidepath,
  gated by each flight's own join distance, never `d_faf`); IAF legs / pre-FAF join
  window / fix discs are normative and must not enter a loss. Design + measurements:
  `docs/2026-09-04_procedure_constraints_design.zh.md`; the method survey (penalty,
  bounded reparametrization, projection layers, primal-dual, sampling, two-stage with the
  optimizer) with reading list and the P0–P3 order:
  `docs/2026-09-04_constraint_methods_survey.zh.md`.
- **Index of the 2026-09-03 frame / runway / state-output experiments (four docs, one
  narrative, Chinese): `docs/2026-09-03_runway_frame_experiments_index.zh.md`; the
  runway-assignment reading list is `docs/literature/runway_assignment/README.md`.**
- **Runway-hypothesis expansion DONE 2026-09-03 (`run_ts_runway_hypotheses.py`, no training):**
  one threshold-anchored forecast per candidate runway, scored in the true runway's chart.
  The assigned label reproduces the baseline bit-for-bit (the chain check). What the label is
  worth: at KRDU a causal "active configuration" rule (most-used runway among development
  landings in the 30 min before entry) recovers the DIRECTION (majority runways 80–83 %) but
  guesses the majority sibling for the minority runway (05R/23L 29–31 %), costing +19 %
  pooled FDE (+30 % straight-in), i.e. ~500–800 m on those flights; at KSJC the same rule is
  93.8 % right and costs nothing (30L/30R are 230 m apart). An oracle over the real sibling
  gains 79 m of median FDE at KRDU against 32 m for a mirror-image fake sibling at the same
  separation, so about half of a K=2 sibling oracle is picking the luckier forecast, not
  runway knowledge; at KSJC the fake sibling gains MORE than the real one. The forecast's own
  closest approach to its hypothesised runway is useless as a selector (37–45 %). Left/right
  between parallels is the genuine unresolved mode; direction is not. →
  `docs/2026-09-03_runway_hypothesis_expansion.md`
- **The state model's KRDU endpoints sit ~250 m NW of every runway, and it is the model,
  not the data** (`docs/2026-09-03_krdu_nw_endpoint_bias.md`): a world-fixed translation of
  the whole predicted path present from the FIRST predicted step (240–350 m off the
  aircraft's actual position, path then parallel to the truth within 1.3°), on straight-in
  flights (established +204 m lateral miss, vectored +24 m), reproduced by noise-free
  synthetic straight-in histories, both seeds. Sign = KRDU's population-mean lateral drift
  (63 % of anchors SE of the centreline; observed +60 s drift median 0, mean +192 m NW);
  KSJC's drift is SE-ward and shows no bias. The objective cannot see it: 300 m on a
  straight-in is ~9e-4 per point against a ~0.08 pooled loss dominated by vectored
  kilometres, and the state output has no continuity to the anchor and no cross-track
  term. Read every arm-A per-runway cross-track number with this translation in mind.
  The anchor-relative output (state-v2 candidate, same doc set) fixes the start of the
  path and the straight-in stratum but loses ~350 m of vectored FDE at KRDU on both
  seeds — vetoed by its own pre-registered rule; a continuity term on an absolute output
  (keeping the endpoint prior) is the open next candidate.
- **Airport-frame ablation DONE 2026-09-03 (14 runs, KRDU + KSJC, two seeds; keep `enu`).**
  Removing the threshold anchor makes the model average across each parallel pair (KRDU:
  endpoints nearer the sibling 1.5 % → 12–15 %, minority runway pulled ~600 m, its FDE
  +30–45 %); target coordinates as input channels change none of that; the vectored-stratum
  gain (H2) flipped sign on the second seed. Seed floor on this axis: the threshold arm moves
  5–22 m pooled ADE across seeds, the airport arms up to 107 m — read every margin against
  that. Runner `run_ts_frame_ablation.py` (state arms, val split, resumable, no CV, no CZML;
  `--experiment-id` runs refuse a dirty worktree at EVERY arm start), readout
  `docs/compare_frame_arms.py`, results
  `docs/2026-09-03_airport_frame_ablation_results.md`. Not done: PatchTST A/B, control
  output, a KSJC cohort with enough 30R/12L flights to test the parallel pair there.
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
