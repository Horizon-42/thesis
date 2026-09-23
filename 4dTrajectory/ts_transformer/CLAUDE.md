# ts_transformer — learned trajectory prediction

Vendored iTransformer + PatchTST (torch). Predicts the remainder of an arrival from a
threshold-anchored lookback window.

**This file is the INDEX: one line per contract, default, trap and module rule, each ending in
the ID of its full text** under `docs/reference/` — `P` prediction paths
(`prediction_paths.md`), `C` contracts (`contracts.md`), `D`/`S`/`G`/`H` defaults, selection
metrics, anchor grid, command hooks (`defaults.md`), `L` layout (`layout.md`), `R` runners
(`runners.md`), `T`/`W` traps (`traps.md`). Find one with `grep -n '^### C7 ·' docs/reference/*.md`.
That text was moved there verbatim on 2026-09-16 (this file had reached 118 KB),
and the 09-14…09-17 additions were placed there the same way on 09-18. The evidence
behind each line — measurements, campaign results, the causes already ruled out — lives in
`docs/ENGINEERING_NOTES.md`; status and next steps in `docs/OPEN_ITEMS.md`; mechanism and result
tables in the package `README.md`; history in the repo's `docs/CHANGELOG.md` (2026-07-19,
07-20 ×2; the move itself: 2026-09-16).
Read the notes before designing an experiment or touching the loss, rollout or output layer.

**Maintenance: a new fact goes into its `docs/reference/` file under a new ID, and this index
gains ONE line.** Measurements, campaign logs, module line counts and runner manuals never go here.

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
python run_ts.py pipeline --airport KRDU
```

## The prediction paths — this is the experiment

`prediction_output` decides whether dynamics is connected at all, and the answers are the point
of the package, not a migration in progress.

- **`state`** — the purely kinematic BASELINE; **no flyability guarantee, deliberately**, so the
  learned component is measured on its own (P1).
- **`control`** — the model emits bounded controls and a differentiable RK4 rollout of the shared
  point-mass equations flies them: dynamically admissible by construction (P2).
- **`closure`, `plan`, `segment-plan` — RETIRED 2026-09-18** (`config.PREDICTION_OUTPUTS_RETIRED`):
  code under `archive/{closure,plan_head,two_tier_v2}_2026_09/` (a README each), stored checkpoints
  refused at load, published categories kept (the frontend mirrors `PREDICTION_OUTPUTS_PUBLISHED`).
  Only the rule guidance stayed live, as `outputs/guidance/`. Their numbers:
  `docs/2026-09-09_plan_and_guidance_design.md` §12, `docs/2026-09-17_two_tier_plan_v2.zh.md` §10–§12 (P3, P4, P9).
- **`manoeuvre` — the SECOND LAYER's line; the second layer itself is not built.** The
  **intent-code** layer (a learned FSQ code per segment, the executor conditioned on it, a causal
  prior over codes) is **ARCHIVED 2026-09-20**: `archive/manoeuvre_codes_2026_09/` (README there;
  tokenizer, sequences, prior, readout, `plan_token.py`, gates T/X/P/E/S copied to
  `gates_manoeuvre.py`, the `manoeuvre_*` runners and `two_tier_b_queue`). Why: plan v3 §10's
  audit — both layers trained on truth and only ever evaluated closed-loop, and the truth codes
  were indexed by time, not by where the executor was. Numbers:
  `docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11 and the campaign trees
  `4dTrajectory/outputs/KRDU/experiments/{two_tier_v3_b_20260919,manoeuvre_tok_20260918}` +
  `outputs/codebooks/` — read them through §10 item 1 (open-loop-trained executors). `plan_conditioning`
  keeps ONLY `off`; `manoeuvre-code` is refused at load by name (`PLAN_CONDITIONINGS_RETIRED`).
  **Live**: the no-token closed loop — `lockstep.py` (protocol `none`, payload schema
  `ts-manoeuvre-lockstep-v4`), `gates.py` (grid + relative), `failure_modes.py`, runners
  `manoeuvre_lockstep`, `executor_*`, `two_tier_grid_queue` (R8) — i.e. two-tier v3 stage A
  (P10, D30 / D31 / C28 / R7 / R9 are the archived layer's records).
- **Control-path axes**: `latent_dim > 0` (latent intent z) and `cta_conditioning=given` (the
  given arrival time IS the duration) — their oracle forms READ THE FUTURE and the run name says so
  (`control+z8`, `z=posterior`, `cta=given`), never a prediction result (P5). The duration head:
  `quantile` (five `DURATION_QUANTILES`, the median flies; `predict --cta-from-quantiles` →
  `cta=self-q` IS a prediction result) and `two-head` (the point head drives the rollout, the
  quantile head only publishes the interval — the two gains come from different mechanisms) (P6).
- Single-aircraft-only and deterministic point prediction are scope decisions (P7). Dynamics axes
  under the control path: `control_dynamics_model` (`point-mass` | `first-order-lag`) ×
  `control_dynamics_backend` (`reanchored-rk4` | `scaled-transport-chart-velocity`; the unscaled
  one is RETIRED); `outputs/dynamics/backends.py` is keyed by the PAIR; the lag model wraps
  `transport_chart_rhs` — three actuators, not a second flight model (P8).

## Contracts (violating these breaks loading, or silently scores the wrong thing)

- Channels `(e, n, u, edot, ndot, udot)`: names AND order are load-bearing (`load_checkpoint`
  refuses a mismatch); velocities are exact chart derivatives, not raw ENU (C1).
- Distance-to-go is measured from `FlightSeries.target_chart`; a consumer reading `hypot(e, n)` is
  silently wrong under the airport frame only (C2).
- "It crossed the threshold" is `d ≤ 0` AND on the final
  (`final_approach_geometry.threshold_crossing_index`) — the plane alone fires ABEAM on a downwind;
  no crossing means say so, never cut somewhere (C3).
- **The control head has FOUR contracts and each is ONE row** (`control_thrust_parameterization` ∈
  `thrust-fraction` (default, every stored run) | `specific-force` | `speed-command` (failed, kept
  loadable) | `specific-force+path-angle`): the row is `ControlContract` in `outputs/envelope.py`
  (names, box, neutral, the lag law, teacher, saturation labels, `identity_suffix`) plus a
  `config.CONTROL_PARAMETERIZATION_SCOPES` row — **never branch on the value anywhere else**. The
  parameterisation is a REQUIRED argument of `dynamics_arrays` / `anchor_controls` /
  `actual_controls` / `commanded_controls`, and the condition feature set of `dynamics_arrays` /
  `condition_vector` (the sets share one width, so a forgotten one is a silent mislabel);
  `identity_suffix` MUST stay empty for
  thrust-fraction and specific-force or every stored checkpoint of those is refused; the same thrust
  RANGE is not the same dynamics (with drag cancelled the speed has no drag feedback). Across the
  2026-09-16 refactor every stored checkpoint replays bit-identically on CPU, ≤ 2e-13 relative on
  CUDA (C27).
- **ARCHIVED 2026-09-20** (`archive/manoeuvre_codes_2026_09/`): the codebook as a hashed artefact
  and the executor binding to its `codebook_sha256` (C28).
- Controls are DIMENSIONLESS (`outputs/envelope.py`); the thrust floor is negative on purpose; the
  head's n ≥ 0.2 and `flyability`'s n ≥ 0.5 differ on purpose and neither moves; a guidance layer
  commanding n reads `flyability`'s envelope (C4).
- `config.py` is the single source, serialised into every checkpoint; `TSConfig` is flat, owned by
  typed views — a new field goes on exactly ONE view, a foreign-owned field is refused off default (C5).
- `dt_s = 2.0`, `seq_len = 60`, `pred_len` 30 / **300** — sized from MEASURED durations (97.8 %
  covered); do not resize from the "3.5–5 min" estimate; over-horizon flights are `horizonCapped` (C6).
- A new loss term goes in the strategy's `loss_component_names`, or `KeyError` after the slow
  dataset build (C7).
- A latent run's total KL says nothing about WHERE it is spent: read the `component_kl_*` split,
  `mean_displacement_sigma`, `active_units_0p05` (`run_ts.py latent_readout` / `latent_probe`);
  the `component_*` numbers sum to `component_kl_nats_per_flight`, not the charged KL (C8).
- A turn-rate supervision term reads the RHS (`heading_rate_rad_s` on `actual_controls`), never
  `g·tan φ / V` (C9).
- The anchor state is `batch_contract.anchor_state(x, C)`, never `x[:, -1]` (C10).
- `cta=given` hands the decoder the truth duration; only `cta=self-q` is quotable (C11).
- Every quantile record carries `source.calibrated`; a conformal table is bound to
  `checkpoint_sha256` and `DURATION_QUANTILES` and RAISES otherwise (C12).
- Calibrated coverage is MEASURED, never guaranteed; design gate 3.4-2 reads the `deployed`
  block (a refused stratum falls through to the pooled δ) (C13).
- `intent_conditioning=truth-…` (FROZEN) reads the FUTURE — an upper-bound instrument, never a
  predictor (C14).
- `flight_metrics` is REQUIRED on `write_batch`; a non-finite ADE refuses the batch (C15).
- τ past RK4's stability limit gives NaN; `TSConfig` refuses exactly `h/τ > 2.785` (C16).
- The fixed anchor is `dataset.fixed_anchor_index(...)`, read off the window set — never restate
  `seq_len - 1` (C17).
- `probe_dynamics` carries every key a real batch carries (pinned in `test_supervision_terms.py`) (C18).
- Instance normalisation is OFF and stays off; signature of ON: lateral p95 pinned at 14.3–14.5 km (C19).
- Prediction records start at `t=0` = the anchor; the reference covers the SAME span; the CZML
  builder adds `source.anchorTimeS` back; `observed_states` is REQUIRED (C20).
- The anchor's control state is inverted from the observed lookback, never the first command (C21).
- The teacher inverse must invert the CONFIGURED forward model (same registry key); the transport
  term is unconditional (C22).
- A fitted teacher table belongs to one N, anchor and cohort — six checks, no partial mode (C23).
- A path's training-time input (fitted teacher, rolled windows) is opened once by `fit_model`;
  every replay path works without it; its provenance never enters `data_provenance` (C24).
- Replay runners fingerprint via `checkpoint_data_provenance(payload, manifests)`, never
  `arrival_data_provenance` (C25).
- **The ts data identity is the eligible SET** (`ts-arrival-data-v4-eligible-set`), never the
  roster's bytes or counts; v3 checkpoints still load; a pre-2026-09-08 in-flight
  `cv_candidate_progress.json` is refused on resume — delete that one file (C26).
- **A replay finds a checkpoint's manifests by the digest it recorded**
  (`repo_layout.checkpoint_arrival_manifests`): the live harvest or a FROZEN generation
  (`harvest-v5-20260823` holds every checkpoint of 2026-08-24..09-23); never by path (C29).
- **The instruction sentence artefact** (`instructions/`, 2026-09-23): one spec sha, written once and
  refused on a sha mismatch; the LABELLER's source hash is part of it (a spec measured by other code
  refuses to label); candidates = the manifest's CIFP runway geometry; one gate (`read.admit`) for
  labelling and measuring; a sentence's words align with the FIRST `len(words)` rows of its
  signals (it ends before the landing — since `instruction-v2` the harvest's condition, parallel runways
  from every runway end the harvest builds; the labeller and the display share one hold funnel) (C30).

## Current defaults and their status

| axis | default | status (full text: the ID) |
|---|---|---|
| `coordinate_frame` | `enu` | keep; `airport-enu` / `runway-aligned` FROZEN (D1) |
| `state_position_reference` | `absolute` | `corridor-bounded` adopted as candidate default; `anchor-relative` VETOED — stored configs load, new runs cannot select it (`*_AVAILABLE`) (D2) |
| control recipe | `simple-v3` | imitation weight 64.0 does NOT transfer between airports; a latent run is `custom` (D3) |
| `control_thrust_parameterization` | `thrust-fraction` | `specific-force` (the head predicts `n_x = (T−D)/W`) measured on N3: **NOT adopted** — straight-in FDE p50 +240 / +222 m, and the cause is the last 5 km's height/speed SPLIT (open-loop vertical load), not the missing drag feedback (D25) |
| `control_thrust_parameterization` = `speed-command` | — | N6: **ran 2026-09-15 and FAILED, unstable in training** — do not train it again as is; its premise was wrong too (D26) |
| `control_thrust_parameterization` = `specific-force+path-angle` | — | N7′: the third column is a path-angle target flown by a 3 s loop. **Passes every pre-registered gate on both seeds** — fully flyable 97.7 / 97.4 % against the twin's 0.4 / 0.2 %, straight-in FDE p50 543 / 559 vs 647 / 663, pooled ADE 1173 / 1182 vs 1325 / 1295. ONE regression: vectored FDE p50 +625 / +704 m, unexplained (not turn authority). **Adoption is the user's call** (D27) |
| `control_condition_features` | `raw` | `ratios` carries the SAME information at the SAME width (so a ratios arm starts from its raw twin's weights); 4 of the 8 raw channels are constant on this fleet. Built, not yet measured (D28) |
| `control_horizon_s` | `0` (whole approach) | two-tier L1: a FIXED rollout horizon Δ — ONE span definition (`dataset.target_horizon_s`), ONE floor rule (`dataset.effective_min_future_s`), and every duration-deciding axis (CTA, quantile head, `final_time_loss_weight`, latent, imitation teacher) refused. **An L1 arm's numbers are its readouts' [0, Δ], never the record summary's** whole-remainder ADE/FDE (D29) |
| `plan_conditioning` | `off` | **`off` is the only value**: `manoeuvre-code` ARCHIVED 2026-09-20 (D30 / D31 are its record) and `truth-next` / `waypoints` 2026-09-18 — all three refused at load by name, each pointing at its own archive (`PLAN_CONDITIONINGS_RETIRED`) |
| `control_dynamics_model` | `point-mass` | `first-order-lag`: smoothness + 3.4 % ADE, τ = 2.0 s not CV-selected; a pipeline lag cell carries `_lag` (D4) |
| procedure penalty | weights 0 | NOT adopted; hinge scales are module constants (D5) |
| command hook | off in training | **`predict --command-hook barrier --hook-saturation soft` is the ADOPTED use**; no arm trained through a hook beat it; `+` combinations are a lookup, their order the application order (D6) |
| `--truncate-at-threshold` | off | cuts at the first crossing ON THE FINAL; any flyability/geometry/CTA readout of a hooked, late-CTA arm needs it (D7) |
| `control_speed_floor_margin` | `1.10` | same coefficient as the optimizer floor and the anchor gate but a different speed (commanded n, local ISA) — never quote the three as equal (D8) |
| `--project-final` | off | deployment fallback, one gate `on-final`; the FAF gate is deleted (D9) |
| `target_conditioning` | off | `channels` helps only the duration head; PatchTST refuses it (D10) |
| `latent_dim` | 0 | the latent intent (L2); its other fields are refused without it (D11) |
| `latent_beta_warmup_epochs` / `latent_aux_duration_weight` | 0 / 0.0 | L2.f's two levers on the posterior mean (D12) |
| `cta_conditioning` | `off` | `given` is never a prediction result; `self-q` is a predict-time label only (D13) |
| `duration_head` | `point` | `quantile` / `two-head`; the second head is built LAST so the arms pair (D14) |
| `duration_quantile_loss_weight` | `1.0` | refused under `point`; `final_time_loss_weight` refused under `quantile` (D15) |
| `n_segments` (control) | 64 | 32 is free (D16) |
| `control_state_loss_grid` | native | `fixed-dt` FROZEN (D17) |
| `control_heading_rate_loss_weight` | 0 | L1.b arm ②; not measured when written (D18) |
| `control_bank_tv_loss_weight` | 0 | L1.b arm ③; prices reversals, not slope; not measured when written (D19) |
| `checkpoint_selection_metric` | `fixed-anchor-common-grid-ade` | which epoch is kept; names the run (D20, S1) |
| `lr_plateau_metric` | `selection` | which number the plateau scheduler watches; `objective` refused while the objective moves (D21) |
| `random_train_anchor_sampling` | `uniform` | `remaining-path-uniform` = equal weight per km; a stratum draw was REJECTED (D22) |
| `random_train_anchor_l1_share` | `0` | a MIXTURE reserving L−1; flights without an L−1 are counted, not refused (D23) |
| `control_imitation_target` | `inverse-dynamics` | `fitted` teacher table (L5.a); not measured when written (D24) |

- **Selection metrics**: `fixed-anchor-objective` (closure requires it; refused for latent runs),
  `fixed-anchor-common-grid-ade` (default, every fixed-anchor arm), `anchor-grid-common-grid-ade`
  (random-anchor arms; ≈ +12–15 % per epoch; refused with `cta=given` / `intent=truth-…`) (S1).
- **Anchor grid**: `data/anchor_grid.py` is the one definition, imported (not copied) by
  `anytime_curve` and the selection metric (G1); its values live in the torch-free leaf
  `data/anchor_strata.py` (G2); a bin under `PARTIAL_COVERAGE` is dropped per airport and which
  bins survive is a property of the cohort (G3).
- **Command hooks** run once per control SEGMENT and the record carries the schedule FLOWN; a bank
  change must re-coordinate the load factor; rate gains are `min(gain, 1/Δt)` (H1). A hook acts
  THROUGH the controls, never on the state — the speed floor raises thrust, UNGATED, reading the
  COMMANDED load factor (H2). The trombone is GEOMETRY: a dog-leg sized with
  `V_e = max(V_floor, V_h)`, which is what makes it terminate (H3); it opens only off-course and
  once per flight (H3.1); ~58 % of KRDU flights are already aligned — quote `tromboneDelayS`, do not
  loosen the rule (H3.2); the 15° turn cap comes from the speed floor (H3.3); `soft` softens bounds,
  not modes (H3.4); a falling `thrust_over_max` in its arms is geometry, not relief (H3.5); the 45°
  offset cap binds past ~41 % delay (H3.6); `barrier+trombone` is the stretch alone (H3.7);
  `trombone_surplus_reference` `beeline` (default, bit-identical to L3.e) vs `reference-rollout` —
  sizing against `L_ref` itself is the trap (H3.8). Hook diagnostics are per-FLIGHT and ABSENT
  without a hook, never zero (H4).

## How to read results here (conventions that prevent wrong conclusions)

- **Seed noise on the control path is ~125 m of pooled ADE, not 30 m** (2026-09-08: `B1_point_matched` seeds 1337/2024 = 1248/1373 m around native32's 1322; duration MAE spread ~1 s). The 30 m line came from two-seed STATE arms. A single-seed control-arm ADE difference below ~125 m is not evidence; a gate on it needs a second seed (early stopping off, as A0.b's p180) or must be read against this line. Straight-in FDE and duration MAE spreads are smaller (~0.9 s MAE) but still single-seed unless replicated. **The same config re-trained at the 2026-09-15 code as a pair (`sf_n4/N4_twin` / `_s2024`, 180 / 168 epochs) differs by only 30 m of pooled ADE and 16 m of straight-in FDE p50.** The stored pair's 125 m came with an early stop at 143 epochs, so the line may be inflated by convergence. Keep 125 m as the conservative line until more converged pairs are measured (specific-force design §7.3).
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
  ADE and cross-track improve while **FDE does not**. → `data/approach_difficulty.py`
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

- A regular package under `4dTrajectory/`; **every import is qualified**
  (`from ts_transformer.config import TSConfig`) and `4dTrajectory/` — NEVER `ts_transformer/`
  itself — goes on `sys.path`, or a SECOND copy of every module loads and identity checks fail
  silently (L1, L22).
- Grouped by plane: `data/`, `geometry/`, `backbone/`, `outputs/`, `training/`, `inference/`,
  `cli/`, `experiments/`, plus top-level `config`, `run_naming`, `io_utils`, `repo_layout`,
  `__main__`; a group's `__init__.py` re-exports nothing (L2).
- Runners are `experiments/<name>.py` behind `python run_ts.py <name>` (`--list`); nothing in the
  package imports a runner; `repo_layout.py` is the one definition of repository paths (L3).
- Training-plane module map, and two edges a change must not reverse: `evaluation_protocol` reaches
  `data_provenance`, never `dataset`; `batch_contract` sits BELOW `objective` (L4).
- One strategy per prediction path under `outputs/`; the spine calls
  `outputs.strategy(config).<method>` and never branches on `prediction_output`; the registry is
  lazy; a new path is one package and no spine edit (L5).
- `outputs/control/` is organised by role; what any rollout needs (`outputs/dynamics`,
  `outputs/constraints`, `outputs/envelope`, `outputs/conditioning`) is shared and imports no
  path (L6).
- `outputs/guidance/` (was `outputs/plan/guidance` + `skeleton`, lifted 2026-09-18): the rule
  guidance — `route` and its six measured route rules (L8), `controller.PlanGuidance`, hold ≤ ~3 s
  (L9), `timing`, `skeleton` (the CIFP reader); a BASELINE and diagnostic beside the learned
  second layer, never the model's fix (the user's rule). The
  plan head, its extractors and oracle runner are archived — L7, L10–L15 describe archived code.
- `constraints/saturation` is the ONE soft-saturation definition; `composite` refuses duplicate
  diagnostics; barrier and trombone gates are complementary by construction (L16).
  `RolloutStateView.reference` is the hook-free schedule, built only under `needs_reference` (L17).
- A dynamics backend is a ROW keyed by the model × backend pair (L18).
- `archive/` is off the import path (asserted by `tests/test_architecture.py`); finished one-off
  drivers belong there (L19).
- Measurement code is CODE — package + tests, or `experiments/`; `docs/` holds documents; do not
  add `docs/*.py` (L20).
- `tests/` is one file per topic; shared fixtures in `tests/support.py` (L21).
- A module belongs under `outputs/control/` only if EVERY consumer is control-specific (L23);
  import direction rules, all enforced by `tests/test_architecture.py` (L24). **Between paths**: the
  guidance layer never imports the control path (L28). **`manoeuvre/`**: **nothing under `outputs/`
  imports `manoeuvre`**; only the runners do (L29). **`instructions/`** (2026-09-23): the second
  layer's language — vocabulary, signals, envelopes, labeller, artefact; torch-free, below every
  model, consumed by the runners and the executor (L30). **`autopilot/`** (2026-09-24): the executor —
  flies the words through the control path's point-mass dynamics one 1 s cycle at a time (that backend
  runs no hooks), exact inverse, limits in order; imports no model, training or path package (L31).
- Every CLI flag is named after the `TSConfig` field it sets, parsers use `allow_abbrev=False`;
  the exceptions are listed (L25).
- `run_naming.py` is the single naming grammar and every field is named or excused;
  **on-disk run/category directories are historical record — never rename them** (L26).
  `run_parameter_rows` + `docs/experiments/intents.json` — **no intent entry ⇒ the publication
  is blocked** (L27).

**Runners** (`docs/reference/runners.md`): `anytime_curve` — A0; strata fixed at L−1 for every bin,
paired verdicts over flights present in both bins, `--write-records` makes a bin publishable, a
pooled checkpoint's bin cannot be published per airport (R1). `eta_calibration` — B2; the duration
head alone, half A fits the deployed δ and half B measures, a single cut's coverage carries cut
noise (sd ≈ 0.05, sign flips), the table is a sidecar never in `data_provenance` (R2).
`quantile_fan_readout` — B3; the geometric column is a readout, not a coverage guarantee;
`cal.hit*` is in-sample on val (R3). `eta_error_readout` — B0; |Δt| p80 is 65.8–72.5 s vectored against
12.0–20.3 s straight-in (KRDU val), so one pooled ETA interval cannot serve both (R4). `latent_probe` — L2.f;
`--limit N` is a prefix (a smoke test); the posterior reads the future (R5). `latent_fan_readout`
— 4(a); the RANDOM fan is the reading, not a footnote (R6). **`manoeuvre_codebook` / `manoeuvre_readout` /
`manoeuvre_prior` / `manoeuvre_prior_readout` / `manoeuvre_gates` / `manoeuvre_code_atlas`** —
ARCHIVED 2026-09-20 (`archive/manoeuvre_codes_2026_09/`; R7 is their record). **Two-tier v3
stage A** (2026-09-18): `plan_cohort --arms` writes one development cohort PER CELL from one load (an arm's
own `development_cohort` wins over the file's; `frame_ablation --only` trains a subset); `manoeuvre_lockstep`
flies the no-token closed loop (schema `ts-manoeuvre-lockstep-v4`) and `--anchor-remaining-km X` starts it at the remaining-path bin
(`lockstep.from_remaining_path` + `dataset.series_from_row`: the same flight first seen at the bin's row) and
`--first-prediction-row N` at one common row of every flight (reading (c), `lockstep.from_row`: the same segment
for every lookback — reading (a) from L−1 confounds lookback with starting point);
`executor_failure_modes` classifies the non-crossing flights (six modes, course frame; not a gate);
`executor_grid_gate` picks the cell (the seed line read off the grid's own seed pairs, p75; fully flyable
≥ 0.95; ties → shorter Δ then shorter L); `two_tier_grid_queue` trains and reads one cell at a time; `executor_relative_gate` judges a candidate reading against its
protocol-none baseline on the common flights (§3.3 rows A3 / B / B1, the seed line named, never typed in; it reads
THIS code's schema only, no compatibility) (R8). **Stage B's intent-code queue** (`two_tier_b_queue`, 2026-09-19):
ARCHIVED 2026-09-20 (R9 is its record); `manoeuvre_lockstep --cohort` stayed (B0's re-read on the B cohort = the
grid's L60_D60 cohort). **The instruction labeller** (2026-09-23): `instruction_signals` →
`instruction_spec` (measured on TRAIN only) → `instruction_labels` → `instruction_figures`, each into
one never-overwritten artefact directory (R10). `instruction_training_export` writes the frontend's Training sets
from that artefact — a seeded per-airport VAL sample, every flight re-read against its stored sentence, every envelope's
geometry from `instructions/display.py`, added to each airport's `training/index.json` without touching its other sets (R11).

## Traps (one line each; full text `docs/reference/traps.md`, evidence `docs/ENGINEERING_NOTES.md`)

- **A stored config lacks every field added after it trained — read the absence as `from_dict`
  does** (`config.absent_field_defaults`), never as `None` (T18).
- A specific-force rollout's speed drift is not a bug in the law (no drag feedback) — never give
  that contract a level-trim neutral (T19).
- **The third actuator is a load factor under every law EXCEPT `specific-force+path-angle`**, where
  it is a path-angle target in radians — ask the contract's law (`law.geodetic_load`) for the flown
  load, never read `actual_controls[..., 2]` blind (T20).
- A tracker scored against the rule guidance must be flown to the GUIDANCE's budget, or "established"
  is decided by a rounding error (T21).
- **A straight-in FDE is a TIMING number** (along-track 60–77 % of Σ FDE², vertical 1–2 %): read it
  with its along/cross/vertical split, and remember the vertical-load channel is open loop on every
  law (T22).
- A checkpoint trained before `c544db0` (2026-09-09) is NOT a same-code baseline for one trained
  after it — re-train the twin (T23).
- **Every ts number assumes the LANDED runway is known** (future information) — quote ADE/FDE as
  runway-given (T1).
- A candidate-symmetric runway head must not carry a per-runway constant; label-derived columns
  come from EARLIER training days, never out of fold (T2).
- Arrival separation is a DISTANCE rule compared on the APPROACH CLOCK —
  `inference/runway_schedule.py` is the one definition, every value cited (T3).
- A runway / configuration model cannot be judged on the per-flight split — split by OPERATING
  day (`data/runway_context.operational_day`, 09Z cut) (T4).
- The objective must score VELOCITY, not just position (T5).
- Bank unsupervised lands BELOW a trivial baseline; `simple-v3` fixes it (0.124 → 0.735) (T6).
- Three causes of the bank wiggle (segment count, budget, conditioning capacity) are NOT it — do
  not re-litigate without new evidence (T7).
- The imitation dose curve is NOT a ramp (plateau below ~11.8×, overshoot past ~47×) (T8).
- Loss weights are calibrated, not chosen; over-constraining looks exactly like blandness (T9).
- `DEFAULT_CV_PATIENCE = 6` is too small here (T10).
- A named recipe cannot be cross-validated as itself (T11).
- The duration head cannot predict below ~125 s — unfixed, in every flight model (T12).
- The state model's KRDU endpoints sit ~250 m NW of every runway — the model, not the data (T13).
- `random_train_anchor=True` + the imitation term is a performance cliff (T14).
- Diagnostic scripts calling `model(x)` cannot run a corridor-bounded checkpoint — use
  `batch_contract.model_forward` (T15).
- `StateOutputLayer.offset_mask` is a non-persistent buffer (T16).
- The root `.gitignore`'s `data` and `4dTrajectory/.gitignore`'s `outputs` match the package's
  `data/` and `outputs/` — a new group named like an artifact directory needs a negation; check
  `git status --ignored` (T17).

## Where to go next

| doing this | read first |
|---|---|
| designing an experiment / changing loss, rollout, output layer | `docs/ENGINEERING_NOTES.md` |
| picking up work, checking what a campaign settled | `docs/OPEN_ITEMS.md` |
| putting the procedure constraint into TRAINING as a hard constraint (either path), or the lazy-network / gate question | `docs/2026-09-08_hard_constraints_survey_and_integration_plan.md` (survey with formulas + H0–H6 plan; papers in repo `docs/literature/procedure_hard_constraints/`) |
| mechanism, architecture, result tables, deliberate scope | `README.md` |
| comparing airports or quoting an ADE | `data/approach_difficulty.py`, repo `docs/2026-08-21_ksjc_route_mix_and_ade.md` |
| predicting the landing runway (runway intent), multi-runway scheduling | `docs/2026-09-13_runway_intent_plan.zh.md` (status by stage R0–R4: W1). The separation rules themselves: `inference/runway_schedule.py` and repo `docs/literature/arrival_separation/` |
| building or reading the **second layer** (what the executor is told each segment, a causal prior over it, later a multi-aircraft graph with separation masks) | **`docs/2026-09-18_two_tier_plan_v3.zh.md`** — the OVERVIEW (intent, outline, the metric and readout protocol §3, the framework §4, the 2026-09-20 audit §10) and its stage A document **`…_v3_A.zh.md`** (the no-token executor's (L, Δ) grid, run). **The second layer now**: **`docs/2026-09-23_two_tier_framework.zh.md`** (layers, packages, artefact, stage gates), **`docs/2026-09-23_instruction_vocabulary_design.zh.md`** (the words, envelopes, labelling rules, values) and **`docs/2026-09-23_instruction_labels_readout.zh.md`** (the labeller's readout on the five-airport development set). The intent-CODE version (`2026-09-18_manoeuvre_token_plan.zh.md` + its readouts `…_results.zh.md`), the two-tier v2 plan (`2026-09-17_two_tier_plan_v2.zh.md`) and the 09-16 feasibility doc are SUPERSEDED: only their measurements are citable (v2 §10–§12; the code readouts §9–§11, read through v3 §10 item 1): W2 |
| the full text behind any line of this index | `docs/reference/*.md`, by ID |
| anything about vertical datum, velocity seam, flight identity | `flight_scenarios/CLAUDE.md` |
