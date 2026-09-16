# AeroViz-4D Development Changelog

Dated log of significant changes, root causes, and decisions, referenced from `CLAUDE.md`. This file is deliberately NOT loaded into every session — read it when investigating history: why a design is the way it is, when/why a default changed, what a past bug or postmortem looked like, or which outputs a change made stale. Append new entries at the top (`### YYYY-MM-DD — title`); when a change produces a durable fact (gotcha, default, contract), also update the corresponding section in `CLAUDE.md`.

Entries verified via full test suites + tsc + vite build at the time; "verified in-browser" noted only where done. Merged same-day, same-topic entries.

### 2026-09-16 — two-tier transformer feasibility report + `lead_time_error` readout

**Ask.** The user: is a two-tier transformer worth building — a short-horizon iTransformer predicting
control parameters (re-asked, "the short horizon should cut ADE a lot"), under a long-horizon layer that
tokenises time segments and predicts the flight plan, later a graph layer over several aircraft's predicted
segments; with a literature search, mathematical reasoning, and uses beyond trajectory prediction.

**Written** `4dTrajectory/ts_transformer/docs/2026-09-16_two_tier_transformer_feasibility.zh.md`:
- three evaluation protocols separated (A whole-approach from one anchor, B rolling re-anchored on real
  observations, C plan-given tracking oracle) — the premise holds under B for the EXISTING model
  (native32 ADE over its first 60 s: 250 m vectored / 115 m straight-in, whole approach 2870 / 445) and
  cannot hold under A for any model (`ADE_Δ / ADE_T = (Δ/T)^α`);
- the error decomposition `e ≤ e_track + e_plan + e_repr`: a stable short-horizon tracker bounds
  `e_track` independent of horizon (the chained model's `L^k` term becomes a constant), `e_plan` stays
  bounded below by the intent information (truth join + duration: 2011 m vectored), and the term a learned
  layer can move is `e_repr + e_track` — the rule-based guidance flies the truth's plan to 1847 m, the
  control basis reproduces it to 106 m p50;
- per-tier "iterated vs direct": tier 1 iterated in the (Markov, exact) dynamics, tier 2 direct and
  multi-modal in the (non-Markov) intent;
- the graph layer gated by an L4-type oracle (its measured signal is runway / configuration / order);
- plan T0–T4 with pre-registered gates; seven uses beyond trajectory prediction (CTA delivery, separation
  probability, instruction recognition / conformance, go-around detection, runway configuration, traffic
  generation, procedure-design feedback).

**Measured** (new runner `run_ts.py lead_time_error`, `experiments/lead_time_error.py` +
`tests/test_lead_time_error.py`): displacement against lead time per stratum off stored records, both
series on a 1 s grid, a flight absent (never 0, never held) at a lead its records do not reach; the growth
exponent as the median's log-log slope PAIRED over the flights present at both leads (an opus review
caught the unpaired form reading 1.34 on a fixture whose every flight grows linearly; also: leads off the
1 s grid refused instead of rounded, the `.txt` sidecar under the same immutability rule, the chart
from `geometric_metrics.chart_rows` instead of a restated one). Artifact
`4dTrajectory/outputs/KRDU/experiments/two_tier_feasibility_20260916/lead_time_error.{json,txt}` over
native32, L3_cta and the state arm A_threshold_enu: the control path's error starts from ~20 m at 10 s and
grows as h^1.55 (straight-in) / h^1.77 (vectored) to 60 s; the state path starts from 400–600 m (its known
first-step offset) and does not grow until 90 s; the given-true-duration arm matches native32 to 60 s and
only pulls the error back after 180 s.

**Literature** `docs/literature/hierarchical_prediction/` (README index + `download.sh`; PDFs untracked):
hierarchical / two-stage motion forecasting, multi-scale time-series transformers, the compounding-error
theory of iterated vs direct multi-step prediction, multi-aircraft graph and intent-input aviation papers.

Docs: `OPEN_ITEMS.md` entry, `CLAUDE.md` where-to-go-next row. No training run, nothing committed by the
session.

### 2026-09-16 — ts_transformer: the path-angle contract RAN and passes every pre-registered gate (N7′, both seeds)

Campaign `sf_n7` (KRDU, two seeds, same split and code as the δ twins), design §15.
- **The pre-registered break-even holds:** the head's per-flight γ* bias is 0.088 / 0.091° against the
  0.11° §13.4 fixed in advance.
- **The diagnosed defect is gone:** the stall-bound share of the last 5–1 km falls from 33 / 45 % (n_x) and
  64 / 68 % (δ) to **1 %**; the last-km height error from +39 / +54 m to +6 / +3 m; the end |vertical| from
  77 / 90 m to 27 / 28 m.
- **The veto that killed N3 becomes a win:** straight-in FDE p50 543 / 559 m against the δ twin's 647 / 663,
  better on BOTH components of the along/vertical split.
- **Pooled ADE 1173 / 1182 m against 1325 / 1295**, with 70 / 69 % of flights better paired.
- **Fully flyable 97.7 / 97.4 % against the twin's 0.4 / 0.2 %** (observed tracks: 98.4 %).
- Both N3 gates pass as well — the per-class n_x bias sits at the truth's own spread AND the heavy − 737
  speed gap narrows, which the specific force alone had widened.
- **One regression:** the vectored endpoint, FDE p50 +625 / +704 m. It decomposes as cross-track (vectored
  ADE is 242 / 147 m better, and the vertical component is better too), and the obvious explanation —
  coordination costing turn authority — is measured not to be the cause.
- Adoption is the user's call; the evidence supports the PAIR (specific force longitudinally, path angle
  vertically), not either half alone.

### 2026-09-16 — ts_transformer: the path-angle vertical contract, built (N7′ P1+P2, branch `sf-n7`)

A fourth `control_thrust_parameterization`, **`specific-force+path-angle`** (design §14): the control head's
THIRD column becomes a path-angle target γ*, and the lag RHS re-solves the load factor
`n = [cos γ + V(γ* − γ)/(g·τ_γ)]/cos φ` at every RK4 stage, clipped to the load box with the stall clamp
unchanged. Longitudinally it stays the specific force, so `Ė = V·n_x` and the vertical law can only move
energy between height and speed.
- **Why:** the load-factor column is an open-loop double integrator (design §7.5.3). Measured: 0.004 of load
  bias is +92 m of end height and −5.3 m/s; the same instantaneous error under a path-angle target is +17 m.
- **Contract constants** (box [−15°, +10°], neutral −2.9° — the teacher's own median, τ_γ = 3 s) are spelled
  into the target contract, so a checkpoint trained under other values is refused at load.
- **Refused:** the point-mass model, the `fitted` teacher, every command hook, and the heading-rate loss —
  that term reads the third actuator as a load factor, which under this contract is radians (the one real
  bug the P1 review found; it would have priced a sign-flipped, 20×-small turn rate).
- **Nothing stored moves:** defaults are bit-identical and the 247-history census is unchanged in name, slug
  and parameter rows. ts + `aerodynamic_model` 1350 passed; a 2-epoch smoke completed train → predict →
  evaluate on real KRDU data with 1404/1404 solved.
- **Campaign `sf_n7`** (two seeds) launched 2026-09-16 01:31 UTC; gates pre-registered in §14.8, the
  headline one being a per-flight γ* bias at or below 0.11°.

### 2026-09-16 — ts_transformer: the glidepath hook (N7) withdrawn as a model fix; a learned vertical target (N7′) proposed

- **Withdrawn (the user's objection):** the model's purpose is to LEARN to fly a procedure-conforming track.
  A hook that computes the final's vertical profile from the published glidepath answers that by
  construction, so any gain would be the rule's. It is not leakage, but it changes what is evaluated.
- **The collected literature agrees** (`docs/literature/procedure_hard_constraints/`):
  - none of the twelve aviation TP papers imposes the glidepath;
  - physics-as-generator work learns the intent and lets the physics integrate it;
  - residual policies and safety filters are claims about the composite system;
  - test-time-only filtering has a horizon-quadratic error (Geiger & Straehle 2022).
- **Kept only** as a labelled diagnostic or a procedure-only baseline component.
- **Proposed instead (not built, the user's call): N7′.** The head predicts a per-segment path-angle (or
  height) target, and a fixed tracking law with no procedure in it flies it. The profile stays the model's;
  the open-loop double integration diagnosed in design §7.5 goes.
- **Pre-measured the same day (design §13), read-only, 300 KRDU val flights:** each contract's teacher
  inverted from the truth and flown open loop through the package's own dynamics.
  - Straight-in replay ADE p50 **32 m** for the path-angle contract, against 63 m (n_x) and 145 m (δ),
    ending 5 m off in height against 27 / 42 m; vectored 1956 vs 6491 / 6680 m. τ_γ insensitive (2 vs 5 s).
  - The same instantaneous vertical error costs 5.3× less height under the path-angle contract; the
    load-factor row reproduces the N3 tail (+0.004 of load = +92 m, −5.3 m/s).
  - Break-even for a path-angle head is a per-flight bias of 0.11°, so the gain is not automatic; the
    path-angle target is however the more predictable one (lag-1 autocorrelation +0.61 against +0.26).

### 2026-09-15 — ts_transformer: the final-descent split diagnosed — the vertical channel is open loop; δ's straight-in FDE edge is a stall-bound dive

Read-only diagnosis on the four same-code runs (N3 and `N4_twin`, both seeds; specific-force design §7.5).
- **The truth never nears the model's stall boundary** (~0.6 Cl_max on the final descent). Rollout samples
  past it are 16–21 m/s slower than the truth.
- **Specific force:**
  - the error is a tail (32 / 45 % of straight-ins), with the n_x command at the teacher and the energy
    right;
  - the vertical-load command sits +0.004 above the truth. Integrated twice open loop, it predicts the
    height error at 5 km with Spearman +0.76 / +0.82;
  - under SF, `Ė = V·n_x`, so the height comes out of speed: the tail ends ~95 m high, ~8 m/s slow and
    1.0–1.1 km behind.
- **Thrust fraction:**
  - its load bias is 2–6 × larger, and 63 / 68 % of its flights reach the stall boundary;
  - the lift cap dives them. They end ~100 m low and on time, which is where δ's straight-in FDE edge
    comes from;
  - its other flights are worse than SF's, and it ends with the larger |vertical|.
- **A straight-in FDE is 60–77 % along-track and 1–2 % vertical** of Σ FDE², so the N3 veto rewards a
  compensating error.
- **Proposed, not built:** N7, an inference-time glidepath hook on the specific-force contract. There the
  vertical law is energy-neutral by construction, the third law the 2026-09-06 nominal-law hook lacked.

### 2026-09-15 — ts_transformer: N6 ran and FAILED (the speed loop is unstable in training); N3/N4 final

- **N3 final** (both seeds against same-code twins):
  - the specific force collapses the per-class n_x bias to the truth's own spread (0.0042 / 0.0044 vs
    0.0125 / 0.0118 g);
  - but the heavy − 737 speed gap grows, and straight-in FDE p50 is +240 / +222 m — the veto trips;
  - not adopted.
- **N4 final:** handing the δ head T_max/W and the stall speed moves its class bias only 0.0125 → 0.0119 /
  0.0118 → 0.0105 g. The class structure is the parameterisation's.
- **N6 ran and failed:**
  - the speed-command arm diverged from epoch 26 (pre-clip gradient norm 1e7–1e10) and early-stopped with its
    epoch-18 weights: pooled ADE 2335 vs 1325 m;
  - its records show zoom climbs, which the loop hides until the thrust clamp binds (design §12.9);
  - the second seed was stopped before training;
  - not adopted, and the next design goes back to the user.
- **The stored B1 twins drifted** because of `c544db0` (review A-3, no config field).
- **A converged same-code δ pair differs by only 30 m of pooled ADE.**
- **Correction, the same day (design §7.4): N3's FDE veto is NOT a missing restoring force**, the reading N6
  was built on.
  - Straight-in approaches fly below the model's minimum-drag speed (98 % of anchors; the last minute at
    1.2–1.4 V_s against V_md ≈ 1.95 V_s), where δ's drag feedback amplifies speed errors.
  - The laws match the truth equally to 5 km out.
  - On the last 5 km, n_x ends 36–54 m high and 4.4–6.4 m/s slow with its total energy right (+5 / +8 m
    equivalent). δ ends ~30 m low at the right speed.

### 2026-09-15 — ts_transformer: a third longitudinal contract, the speed command (N6, branch `sf-n6`)

**Why.** N3 (provisional, design §7.2) passed its first gate on both seeds: the per-class n_x bias collapsed to
the truth's own spread. It failed the second (the heavy − 737 speed gap grew) and tripped the straight-in FDE veto
(887 / 885 m p50; the last 5 km flown 4.4 m/s slow). The specific force cancels the drag and with it the speed's
only feedback, `∂V̇/∂V = 0`.

**What** (design §12):
- `control_thrust_parameterization=speed-command`: the head predicts Δv, a target airspeed relative to the
  anchor's. The lag RHS flies it through a first-order speed loop (τ_V = 8 s, a contract constant) and the
  specific-force law's own clamp: `V̇ = (V₀ + Δv − V)/τ_V`, the same on every airframe.
- New aerodynamic_model pieces: `SpeedCommandLaw`, `lag_rhs_speed_command`,
  `transport_chart_speed_command_thrust_n`, and one `speed_loop_specific_force` read by the RHS and the record.
- `actual_controls` returns the absolute target; `inverse.anchor_relative` makes it relative at the two callers
  that know the anchor.
- `Forecast.specific_force_commands` became `longitudinal_commands` + `longitudinal_parameterization`.
  Specific-force records keep their `specific_force` key.
- Refused off the lag model, with the fitted teacher, and with the speed floor.

**Nothing moved.** The thrust-fraction and specific-force laws are bit-identical to `e959bc0`: loss, gradients,
batch and records. Stored-run names and resume are identical. ts + aerodynamic_model: 1323 passed.

**Review (opus):** no blocker, and the fixes are in design §12.8:
- the hooked rollout's physics is now tested;
- the speed-command constants are spelled into the checkpoint's target contract;
- the neutral is now tested as flown.

The reviewer also measured each law's teacher flown open-loop on 300 KRDU val flights: speed-command ADE 375 m,
specific-force 2606 m, thrust-fraction 2700 m.

**The go/no-go:** the same-code twin `N4_twin` confirms N3's straight-in FDE veto for seed 1337 (887 vs 647 m p50,
design §7.3). N6 is queued after `sf_n4`.

### 2026-09-15 — ts_transformer: the control head's condition vector as ratios (N4, branch `sf-n4`)

**Why.** N3 (running) tests whether predicting the specific force removes the thrust-fraction head's class
structure. The alternative explanation is the conditioning. Under the raw vector the head sees the mass, the
installed thrust and the wing area separately, and the δ it needs is `(n_x + D/W)/(T_max/W)`, a RATIO of two
inputs. N4 is the control that separates the two explanations (design §11).

**What.**
- New control-output axis `control_condition_features ∈ {raw, ratios}`, default `raw`.
- `ratios` keeps the mass and the polar and replaces T_max and S by `T_max/(m g)` and the 1-g stall speed
  (`aircraft.aero_params.stall_speed_ms`). It is the same information in the same width, so a ratios head starts
  from its raw twin's weights.
- `condition_vector` and `dynamics_arrays` take the set as a REQUIRED keyword; `DYNAMICS_CONDITION_NAMES` and
  `CONDITION_CHANNELS` are replaced by `CONDITION_FEATURE_SETS`, `condition_names()` and `CONDITION_WIDTH`.
- The batch-size probe's condition row is written by `condition_vector`; under `raw` it is bit-identical to the
  literal it replaces.
- Every named recipe pins `raw`; a run is named `airframe=ratios`; `train --control-condition-features`.
- Arms `docs/experiments/sf_n4_arms.json` + intents key `sf_n4`: the two ratios arms, and first the two
  thrust-fraction twins re-trained at this code (below).

**Measured on the way.**
- On KRDU's 26 OpenAP-direct types, Cd0, k, stall_threshold and k_stall are the same for every type, so 4 of the
  8 condition channels carry nothing. The raw thrust channel spans 46× over the fleet; T_max/W spans 1.3×.
- **The stored twins do not reproduce** (review S1, design §11.6). `B1_point_matched` (trained at `c0f2b9e`),
  re-trained for 2 epochs with the same config, seed and split at this code, reads epoch-1 val **7.737 against
  8.381**. Two runs at this code agree bit for bit. The data, the initialisation (the duration head's
  first-update gradient to 12 digits) and torch/numpy are the same. So the default control training path
  changed in code between 2026-09-07 and `47b4b40`, and a stored-twin delta would carry that drift. `sf_n4`
  re-trains the twins. **Bisected to `c544db0`** (2026-09-09, review A-3). A flight whose observed track
  reaches the threshold (1.9 % of train) now carries terminal position emphasis. The fix was intended and
  documented, but it has no config field, and its effect on training was never measured: epoch-1 selection
  ADE −11 %. `val_loss` is not comparable across it. Reversing that one hunk restores the twin bit for bit.
- The M2 smoke note that quoted the twin's 6176 / 3535 m at epochs 1 / 2 is corrected in design §7.1: this
  code's thrust-fraction run reads 5482 / 3754.

**Nothing moved.** The default path is bit-identical to `47b4b40` on a lagged training step (loss, every
gradient, the batch, the records). Over the 245 stored runs, names, slugs, parameter rows and loading are
identical, and so is campaign resume over 105 stored arms. ts suite 1181 passed.

### 2026-09-14 — specific-force control parameterisation, M1 (branch `specific-force-control`)

**Ask.** The user: "现在的模型，是直接预测不同质量下的操作参数 … 是不是可以修改aerodynamic 或者增加一个normalize
处理？" Then: design it and implement it on a new branch, with a review at every key step; the user merges.

**Why.** The literature is `docs/literature/control_normalization/` (`683e604`). On KRDU `B1_point_matched`
(2 seeds), the head commands nearly the same δ = T/T_max on every aircraft class, while the truth needs
0.028–0.041. That leaves a class-dependent specific-force bias of 0.010–0.012 g against the truth's own
0.0045 g, and heavies fly 4.7–4.9 m/s fast relative to the 737 family.

**Change** (design `4dTrajectory/ts_transformer/docs/2026-09-14_specific_force_control_design.md`):
- **New axis** `control_thrust_parameterization ∈ {thrust-fraction (default), specific-force}`.
  - Under specific-force the head's first column is n_x = (T − D)/W.
  - The lag RHS re-solves the thrust at every RK4 stage as clamp(W·a_x + D, −0.2·T_max, T_max), so drag cancels
    and V̇ = g(a_x − sin γ) where the clamp does not bind (`aerodynamic_model/torch_lag_dynamics`:
    `SpecificForceLaw`, per-law step functions and compile caches).
  - First-order-lag only. Refused with the fitted teacher and, until M2, with the speed-floor hook.
  - Every named recipe pins thrust-fraction.
- **Per-contract box** (`outputs/envelope.ControlContract`): the specific-force box is [−0.20, 0.23] with half
  width 0.215 g (= 0.6 × the fleet-median T_max/W), and its neutral is −0.05.
  - The neutral is a descent speed hold, not level trim. With the drag cancelled the speed has no drag feedback,
    and a level-trim neutral flew untrained heads to 311–328 m/s against thrust-fraction's 209–213.
- **Plumbing.** `dynamics_arrays` / `actual_controls` / `commanded_controls` take the parameterisation as a
  REQUIRED argument. The record keeps newtons: under specific-force the segment-start thrust, plus
  `control_segments[*].specific_force` and `source.controlThrustParameterization`, both absent under the default.
- **Every default path is bit-identical to `775b59e`:**
  - lag, point-mass, CUDA-compiled, objective + backward, and record JSON;
  - the census of 245 stored `history.json`: 0 names, slugs or loadability changes.

**Campaign-resume behaviour change.**
- `experiments/frame_ablation.stale_arm_error` now reads a field a stored config lacks the way
  `TSConfig.from_dict` does: its default, except REQUIRED fields (`config.absent_field_defaults`).
- The new recipe pin had refused every recipe arm on disk: 4/4 in `b1_quantile_20260907`.
- The same rule makes **36 more stored arms resumable** that `775b59e` refused for other fields added after they
  trained: `duration_head`, `lr_plateau_metric`, `random_train_anchor_sampling`, the `latent_*` fields, the
  heading-rate and bank-TV weights, `control_imitation_target`. Their defaults are documented as the earlier
  behaviour.
- Re-running those campaigns now skips their trained arms and runs pending steps, where it used to stop.

**Also.**
- The predictability report labels its control columns by contract (`thrust_fraction`, unit 1). It used to say
  `thrust_N` / `N`, a mislabel of δ.
- The training diagnostics keep `thrust_N` for comparability.

**Review.** An opus subagent found 1 blocker (the resume refusal) and 3 should-fix issues. All are fixed and
re-verified (design §9).

**M2.**
- **The speed-floor hook works under specific-force.** The demand is drag- and mass-free, with the lag credit
  capped at the engine. It saturates at the ENGINE's `(T_max − D)/W`, not at the head's 0.23 g box, which sits
  below the engine on this fleet. Its soft form is inert by construction. The M1 refusal is lifted.
- **`train --control-thrust-parameterization`.**
- **Docs:** ts `CLAUDE.md` contract, defaults row and traps; `OPEN_ITEMS`; design §7.1.
- **N3 is prepared, not launched:** `docs/experiments/sf_n3_arms.json`, intents key `sf_n3`.
- **Smoke run** (KRDU, 2 epochs; train → predict → evaluate): the chain works end to end. The numbers are not
  results (design §7.1).

### 2026-09-14 — runway intent R3.3: R3's flown schedule as a held-out-days (`dayval`) publication; a test overwrote the live KRDU categories.json

**Ask.** The user: "先做1, 2, 最后3" — item 3, publish R3's trajectories (plan §18.3).

**Change** (`220138a`, then `a6d921a` after a second opus review):
- The split `run_naming.SPLIT_DAYVAL`: a day partition's validation days, flown by a checkpoint trained on
  that partition's training days.
- `runway_intent_r3 --write-records`. A record's prediction is its SCHEDULED landing time (as a CTA arm's
  is its CTA; `runwaySchedule.scheduledTimeS`, and the block's `timing` names which fields read it). The head's
  own ETA error and the flown time's ride in `source.runwaySchedule`. R3's flights
  go through the plan path's forecast, so records say `plan`.
- Publisher:
  - dayval only from a reused directory and only under Experiments;
  - every record's flight is checked against both halves of "held out": the locked outer-test hash and the
    checkpoint's own persisted train/val;
  - `--category-group` files a variant under the campaign that wrote the records.
- `category_display_label` raises on an unknown split.
- Frontend: switching keeps the split in view, and ranking reads the split in view only.
- Records: 5195 over five airports, identical to R3's formal flights (0 s / 0 m).
- Publication waits for the ff-merge: the worktree's `aeroviz-4d/public/data/airports` is a symlink into the
  main tree, whose frontend rejects an airport's whole list on an unknown split.

**Incident.**
- What happened: a new publisher test called `main()` with only `--output-root`, then wrote a fake
  `categories.json` next to `plan.comparison_dir`. Through the symlink, that overwrote the live KRDU
  `categories.json` (152 entries → 1). The file is git-ignored and had no backup.
- Rebuilt from on-disk sources:
  - publication manifests through the publisher's own refresh functions;
  - the relabeler for the legacy `ts_*` keys;
  - other airports' entries for the optimizer/observed rows;
  - each category's `comparison_index.json`.
- The same method reproduces the other four airports field for field, and the result passes the frontend's
  guards.
- Written back at 17:41 with the user's permission ("写回"); the main tree's `npm run check-publication --
  --airport KRDU` reads 152 categories, 0 errors, 0 warnings. The rebuilt file, the rebuild script and the
  damaged file are kept in `runway_intent_r3_20260914/incident_20260914_krdu_categories/`.
- The publisher tests now point `main()`'s default roots into tmp and refuse every JSON write inside the live
  trees (the publisher's writer and the tests' own), since `PublicationPlan`'s default roots are bound at import.

### 2026-09-14 — ts_transformer: runway intent R3.1 / R3.2 — scheduling under ETA uncertainty has no predictive value; the closure's lost time located, not fixed

**Ask.** The user: "先做1, 2, 最后3" (plan §17.8's three next steps, §18 of
`4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`).

**R3.1.** `inference/runway_schedule.sample_schedules` + `experiments/runway_intent_r31.py`: each day_a expert's
ETA error calibrated on its own validation split (training days), stratified by the head's own predicted remaining
time; 200 FCFS schedules on drawn free arrivals, each flight's prediction the median landing time, read against
the same error model without interaction. Reviewed (opus): draws taken with replacement put a flight alone 1-2.5 s
off its baseline and read as a 0.4 s interaction gain — the quantile grid removed it before the run. Result: no
pre-registered gate passes; the interaction moves the median |dt| by −0.4..+0.3 s over all hours and makes the
moved flights worse at 3-4 airports; the calibration's median error does not carry from training to validation
days. R4 stays off.

**R3.2.** `experiments/runway_intent_r32_diagnosis.py`: the v5.4 time closure reads a flight on an instruction leg
through the 8 km placeholder the leg's route carries past the fix, reads it minutes late (X −85 / −176 s at KSMF /
KSTL), flies it at maximum speed, and at the switch to the closing it is early by ~13 s with ~12 km to go and no
lever (+104 s jump). Three tail models (the head's path to go, the closing laid from the turn) removed the chase
but none improved delivery at both airports; stopped before tuning the closure on the evaluation flights — no
change adopted (v3 kept on `wip-r32-leg-timing`); two verified follow-ups in `docs/code-health-followups.md`
§31-32.

### 2026-09-14 — ts_transformer: runway intent R3 — the multi-runway arrival scheduler under the FAA minima, flown; separation rules sourced from JO 7110.65BB

**Ask.** The user: "do it; first write plan then implement" (plan §17 of
`4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`), then "按计划做" — and asked where the
plan's "3 NM ≈ 80 s" came from, wanting the most authoritative separation constraint from the documents, saved
and indexed for the multi-aircraft work.

**Sources.** `docs/literature/arrival_separation/` (new): FAA JO 7110.65BB Change 3 (in force 2026-07-09) quoted
with paragraph numbers — 3 NM terminal radar (5-5-4 a/b), 2.5 NM only by authorization (5-5-4 j), CWT wake
TBL 5-5-2 at the threshold (5-5-4 h; CWT merged into 7110.65BB by Change 2, JO 7110.126B cancelled), parallels
under 2,500 ft as one runway, dependent 1.0 / 1.5 / 2 NM diagonals (5-9-6), independent 3,600 ft with FMA + PRM
under 4,300 ft (5-9-7, 5-9-8), JO 7110.308E's CSPR list (STL yes, SJC no), intersecting runways (3-10-4); type ->
CWT from JO 7360.1K Appendix A (parsed CSV, 2,653 rows); ICAO / RECAT-EU beside them; what could not be
verified. `docs/literature/README.md` indexes every topic folder. The 80 s was an estimate (3 NM at ~70 m/s);
it is now 3 NM at each airport's measured approach speed (KRDU 69.8 m/s -> 79.6 s).

**Built.** `inference/runway_schedule.py` (the scheduler and the one definition of the rules; minima applied on
the APPROACH CLOCK — parallel thresholds are staggered, KRDU 23L/23R 0.67 NM, KSTL 11 vs 12L 1.99 NM),
`experiments/runway_intent_r3.py` (R2b's flights scheduled, flown with the time closure, checked) and its
readout. Reviewed (opus): threshold stagger ignored, an unmeasured gate read as passed, gate 2 blind to the
close pairs, the FCFS key read from runways never tried — fixed before the run. The first formal run compared
flown and unassigned forecasts by `fde_m`, the displacement at the TRUTH's landing time, which scores a late
assigned arrival as distance (31 -> 132 m at KSJC for a clock, not a path); re-run on the end point against the
true threshold (18-21 m flown vs 18-26 unassigned), the first kept as `superseded_8bbe25b/`.

**Result.** Gates: KSTL, KMSY pass all four; KRDU / KSJC / KSMF miss gate 3's busy clause by 0.3-1.2 s (a tie).
The schedule barely binds (2.4-7.3 % of flights delayed), keeps the runway intent, and does not make the moved
flights' times more accurate — ETA error is the size of the minimum. Flown, 71-88 % of undelayed flights land
within 10 s of their slot, delayed ones 37-81 %; the time is lost between asks. R4 is not triggered. Artifacts
`4dTrajectory/outputs/POOLED/experiments/runway_intent_r3_20260914/`.

### 2026-09-13 — ts_transformer: runway intent R2 — the runway head's pick flown end to end; the cost of not knowing the runway

**Ask.** The user: "go on R2". Plan §16 of `4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`
(four stages; gates written before each run).

**Built.** `experiments/runway_intent_r2.py` (the runway heads asked once at the plan expert's anchor
from the last track point at or before it; the expert flown under every candidate by R0b's mechanism;
`--roster expert-val | day-val`; `--write-records` writes the runway fan as prediction records with
`source.runwayIntent`), `runway_intent_r2b_cohort.py` (an expert's cohort restricted to day_a's
training days — the development-cohort mechanism takes it unchanged), `runway_intent_r2c.py` (the
belief re-asked every 30 s) and `runway_intent_r2c_e2e.py` (lock rules end to end, the expert
re-anchored at every ask), and their readouts. Reviewed (opus) at every step; the end-to-end lock
reading was corrected to cover the asks where all rules agree on a wrong runway.

**Result.** R2a (original experts, 1828 flights neither model saw): the head's pick costs +55 m of
paired mean FDE against the known runway, B1 +405 m — every gate passes. R2b (experts retrained on
day_a's training days, 5061 flights): four of six gates; KSJC and KMSY fail because the retrained
experts cannot fly the minority-direction runways (KSJC 12R, KMSY 02), where always-the-majority ends
nearer the truth — on the flights the expert flies when told the runway the choice costs +73 m
(B1 +466). R2c: the belief sharpens along the approach (99.2-100 % inside 3 km), flips mostly correct
it, never-lock is best end to end at four of five airports — adopted. R2d: the runway fan published as
picker variants of the per-airport experts. Artifacts `4dTrajectory/outputs/POOLED/experiments/runway_intent_r2{,b,c,d}_20260913/`.

### 2026-09-13 — ts_transformer: runway intent R1.1 / R1.1b — a candidate-symmetric runway head that survives an unseen configuration

**Ask.** The user: R1.1 before R2 — fix what the day split exposed (R1's head on KSJC's two
30L-closure days: 55-57 % against B1's 97-98 %). Plan §14 (design, gates written before the run)
and §15 (results) of `4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`.

**Built.** `runway_intent_r1.build_samples` / `RunwaySamples` (R1's sample construction as one
function; R1's outputs byte-identical at all five airports); `data/runway_features.candidate_rows`
(one row per candidate runway, every column in that runway's terms; the raw wind, report age, time of
day and one-hot identities left out) and `minutes_since_each`; `experiments/runway_intent_r11.py`
(`ListwiseBooster`: a gradient-boosted conditional logit — shared histogram Newton trees, a softmax
across a sample's candidates, R1's budget; sklearn's boosting cannot take the loss and a one-tree-per-
step sklearn version of the same Newton step took 23 min on KSMF against 58 s, same numbers within
0.9 points; R1's head retrained on the same samples, matched against R1's artifact bit for bit);
`experiments/runway_intent_r11_readout.py` (the §14 gates on every symmetric head).

**Result.** R1.1 as pre-registered failed gate (i): its rows kept two per-runway constants (the
runway's training share and the operator's share of its landings) that let the shared trees
re-identify the runway (a classifier recovers a row's runway from its other columns at 96.7 %,
review), and the closure days stayed at 53-59 %. R1.1b: `r11_noid` drops both (closure days fixed,
but KRDU / KSMF lose 5-9 side points — the airline is a real signal); `r11_lift` drops the prior and
moves the operator's share to its deviation from the base rate. The first R1.1b run missed gate (i)
by 0.1 point and review traced it to the out-of-fold airline share (training days in 5 hashed blocks
— a block's share is anti-correlated with its own labels, -0.84 to -1.00, and the verdict moved with
the hash); the share now counts only training days before the sample's day. Rerun: `r11_lift`
passes every gate — closure days 96.8 / 97.6 %, the side gain kept, G5 at all ten cells — and the
day-blocked vs per-flight gap fell to at most 0.5 points. Gate (i) is not blind (the variant was
chosen after reading those days); the sealed test days are. Next: R2. Artifacts
`4dTrajectory/outputs/POOLED/experiments/runway_intent_r11_20260913/`, `runway_intent_r11b_20260913/`.

### 2026-09-13 — ts_transformer: runway intent R1 — a learned runway head on a day-blocked split, and what the split changes

**Ask.** The user decided D1 = (a): a day-blocked split for the runway experiments, and to look at
its effect first. Plan §12 (design) / §13 (results) of
`4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`.

**Built.** `data/runway_features.py` (causal features — configuration, same-sector landing, wind,
own position and track in each candidate's runway axes, entry sector, airline, time of day — and
`ring_anchors`, the first crossing of 20 / 15 / 10 / 6 km rings around the airport reference);
`data/runway_context.operational_day` (the day cut at the overnight traffic minimum, UTC − 9 h) and
`RunwayContext.with_majority`; `experiments/runway_intent_r1.py` (one HistGradientBoostingClassifier
per airport; two day partitions and the per-flight control, read paired on the flights validation
under both; a no-wind / no-time-of-day ablation; grouped permutation importance; per-operating-day
blocks); `experiments/runway_intent_{r0,r1}_readout.py` (moved out of `docs/`, the package's layout
rule; the R0 readout reproduces its previous output byte for byte).

**Review (opus) before the numbers.** Three optimistic biases, fixed and re-run: the UTC-midnight day
cut split the evening peak between a validation and a training day (→ operating days); R0's
remaining-path anchors are measured to the TRUE threshold, so the anchor's location leaked the
landing runway's along-track distance (→ rings around the airport reference, runway-independent);
the static majority counted validation / test days' labels (→ each partition's own training days).
The pre-review run is kept in `superseded_db1e701/`, void. R0a was re-run on operating days (only
its day and flip counts moved).

**Result.** KRDU / KSMF: side +23–32 points over the best rule, minority runways 93–95 % vs 40–49 %,
every gate passes on both partitions. KSTL: a tie (+4.3 / −0.01; G1 fails mechanically on one
partition). The day split changes nothing on ordinary days (paired −0.2 to +1.1 points) and exposes
what the per-flight split hides: KSJC's two 30L-closure days read 55–57 % against B1's 97–98 %; the
per-flight model, trained on those days' other flights, reads 100 %. Measured cause: the hourly wind
and time-of-day columns fingerprint the usual configuration (without them 87–90 %). Next: R1.1, a
candidate-symmetric head (gates drafted, §13.7). Artifacts
`4dTrajectory/outputs/POOLED/experiments/runway_intent_r1_20260913/`.

### 2026-09-13 — ts_transformer: runway intent — plan, and R0 (what causal context says about the landing runway)

**Why.** Every ts path reads the LANDED runway (threshold anchor; the plan path's CIFP skeleton),
so every published number is runway-given; the user asked for the runway to be predicted — the
intent multi-aircraft interaction turns on. Plan: `4dTrajectory/ts_transformer/docs/2026-09-13_runway_intent_plan.zh.md`
(runway head × per-runway experts, multi-runway scheduling, R0–R4, decisions D1–D5).

**R0 (no training).** `data/runway_context.py` (causal rules B0–B4 over a pool that excludes every
outer-test-hash flight, label and track), `experiments/runway_intent_r0.py` (R0a),
`experiments/runway_hypotheses.py` v4 (R0b: the plan path, the rules as selectors, paired costs;
also fixes print_summary's pre-v3 strata keys, which raised before `hypotheses.json` was written).
On the plan heads' val splits at five airports: the landing direction is solved by co-temporal
landings (96–99 %); the parallel side is not (73–75 % KRDU, 61–65 % KSMF, ~90 % KSTL, minority
runways 33–49 %), and the rules are flat along the approach. Choosing by B1 / B3 costs the plan
heads +380–650 m of paired FDE mean at KRDU / KSTL / KSMF, +20–30 m at KSJC. R1's gates are
pre-registered in the plan (§11.4); R1 waits on D1 (a day-blocked split). Found on the way: the
plan heads fly rarely-landed runways badly even with the true runway (KMSY 02 FDE median 23 km).
Artifacts `4dTrajectory/outputs/POOLED/experiments/runway_intent_r0_20260913/`; the readout is
regenerated by `run_ts.py runway_intent_r0_readout` (moved there from `docs/` with R1). Code reviewed (opus): no sealing or causality
defect; the unpaired cost, a latent KeyError, flips counted across overnight gaps and a
one-sample look-ahead were fixed before the final readout.

### 2026-09-12 — Experiments picker: every run shows its intent and its parameters as named rows

**Ask.** The Trajectories → Experiments picker listed each run as one machine-grammar string
(`control · iTransformer · first-order-lag @… · simple-v2+(hr=8, bank-tv=1) · d-model=512, …, +9
more, <run>`) under a raw campaign id: no attribute names, no structure, no reason. The user asked
for named parameters and each experiment's intent, written into the data where it was missing, and
for a standing rule that every publication of experiment results states its intent.

**Data.** `run_naming.run_parameter_rows(config)` is the grammar in structured form (`Model` /
`Loss edits vs <base>` / Architecture / Training / Data & anchors / Supervision sources /
Conditioning / Control rollout; nothing folded or hashed; `SETTING_SECTIONS` import-guarded).
`loss_design_name` is now rendered from `loss_design_parts` — name/slug/loss name identical on all
513 stored configs (snapshot before/after). New tracked registry
`4dTrajectory/ts_transformer/docs/experiments/intents.json` (`ts-experiment-intents-v1`): 31
campaigns (title + question + design doc), 66 run intents, 36 variant intents, in Chinese, drafted
from the arm declarations' `_comment`s and the design docs (two record-campaign intents inferred
from their `anytime_curve.json`: `anytime_a0b_records_20260908`, and in part
`anytime_a2b_20260908`). The publisher stamps `experiment.{runName, variantLabel, parameters,
intent}`, blocks a publication whose group or run has no registry entry, and refreshes all-or-
nothing.

**Backfill (user-requested).** `--refresh-labels-only` over the three experiment publication
roots (KRDU 133, KSJC 15, POOLED 8 manifests): all **156** experiment categories on five airports
stamped; every other byte of the five `categories.json` unchanged (labels, keys, order — diffed
against backups). No CZML, records or directories touched. `npm run check-publication --server`:
all airports load.

**Frontend.** `ExperimentPicker` (trigger → portalled two-pane browser) + `ExperimentDetails`
(intent + named rows; compact card in the panel). Verified in-browser on KRDU (worktree dev server,
129 runs / 30 campaigns): browse, filter, preview, select → map loads the category; no console
errors.

### 2026-09-12 — aeroviz-4d: the `plan` output is a legal picker category (every airport's picker was empty), and the frontend mirror is now test-pinned

**Symptom.** After the plan-and-guidance heads were published (this evening: `experiment_step3g_fan4_head_*_val`, `experiment_step5b_<icao>_fan4_head*_val`), the comparison picker showed NOTHING on all five airports, and restarting the dev server did not help. Console: `[useComparisonCategories] Failed to load categories manifest: Error: comparison categories for KRDU is not a valid manifest`. Every file was served (200, `application/json`); the frontend rejected the manifest itself.

**Cause.** `aeroviz-4d/src/data/airportData.ts::EXPERIMENT_PREDICTION_OUTPUTS` is a hand-written mirror of `config.PREDICTION_OUTPUTS` — `["state", "control", "closure"]`, last extended on 2026-09-07 (`b28e9ac`) when `closure` had done exactly this. The publisher writes `experiment.predictionOutput` straight from the run's config, the plan path (`052cbaa`, 2026-09-11) writes `"plan"`, `isComparisonCategory` rejects it, and `isComparisonCategoriesManifest` is `.every(isComparisonCategory)` — one rejected category empties the whole airport's manifest. The publishing agents verified HTTP status and content type, not the frontend's validator, so the breakage was invisible to the publication step.

**Fix.** `"plan"` added to the mirror; `airportData.test.ts` iterates the exported array instead of restating it. **Both mirrors are pinned**: `4dTrajectory/ts_transformer/tests/test_frontend_mirrors.py` parses the TypeScript arrays and asserts `EXPERIMENT_PREDICTION_OUTPUTS == config.PREDICTION_OUTPUTS` and `EXPERIMENT_HORIZON_MODES == config.HORIZON_MODES` (the horizon modes were the same kind of inline literal in the validator, now an exported array with a guard), so the next value added to the package fails the ts suite until the frontend learns it. **And the check is a script** (the user's rule: not after every publication, but at a milestone or when results are explicitly asked to be published): `cd aeroviz-4d && npm run check-publication -- --airport KSMF [--server http://localhost:5173]` runs `categories.json` and every category's `comparison_index.json` through the frontend's OWN guards (`src/utils/checkPublication.ts`, unit-tested), names the category and the field the picker would reject, checks the referenced CZML/report files exist, and with `--server` fetches the same files from the running dev server to catch the SPA-fallback case (2026-09-07) as "restart the dev server". The vite public-file cache was NOT the cause tonight — every request was already answered. Verified: the frontend test files, `tsc --noEmit`, the pytest, the script on KSMF/KRDU against disk and the live server; in-browser after the merge (the picker lists and loads the plan categories).

### 2026-09-12 — ts_transformer: plan-and-guidance step 5b, the two-seed check — KSJC/KSTL within the line, KSMF/KMSY seed-fragile

Design §12.11's last block; measurement only (no code). The four per-airport K = 4 heads
re-trained at seed 2024 (`step5b_<ICAO>_fan4_head_s2024`), read at the 60 s anchor unassigned and
with the truth's time, paired flight by flight against the seed-1337 heads
(`step5b_pair_<ICAO>_s2024{,_time0}_vs_s1337_a60s`). Vectored ADE s1337 → s2024: KSJC 3145 → 2906
(paired +30 m at the median), KSTL 2838 → 2780 (+22), KSMF 2758 → 4043 (+358; the rolled cap
2.8 → 19 % of vectored flights), KMSY 3768 → 4044 (+215; established 67 → 56 %). Straight-in and
the time closure unchanged under both seeds (dt MAE 3.1–8.2 s with the time, fully flyable 100 %).
So the step-5b own-vs-pooled margins on KSMF (+79) and KMSY (+217) are inside those two cohorts'
seed line and those rows are undecided; KRDU, KSJC and KSTL stand. Delivery on KSMF/KMSY stays the
seed-1337 head as measured; a third seed or a per-airport fine-tune of the pooled head is listed.
Lesson recorded in the package `CLAUDE.md`: a single-seed per-airport claim on a ~2.4k-flight
cohort is not evidence. The five seed-1337 heads' validation predictions were published to the
frontend picker the same evening (`experiment_step3g_fan4_head_*_val`,
`experiment_step5b_<icao>_fan4_head_*_val`; the experiment index rebuilt, 76 → 97 entries).

### 2026-09-12 — ts_transformer: plan-and-guidance step 5b — per-airport K = 4 heads beat the pooled head on four of five airports

Design §12.11; measurement only (no code). KSJC, KSTL, KSMF and KMSY each got a K = 4 head on
their own cohort (`run_ts.py plan_cohort` per airport; the rolled windows drawn from the pooled
table `step5_pool_table`, which `require_cover` accepts for a subset cohort), read at the 60 s
anchor unassigned and with the truth's time, and paired against the pooled head on the same
flights (`plan_oracle_pair --common`). Vectored ADE, own vs pooled: KSJC 3145 vs 2782 (paired
−40 m for the pooled on 207 flights), KSTL 2838 vs 3274 (+209), KSMF 2758 vs 3245 (+79), KMSY
3768 vs 4159 (+217), KRDU 3087 vs 3678 (+227, §12.10); with the truth's time the own head is
lower on all five (+33 to +591). Straight-in and the time closure (dt MAE 2.8–8.2 s) are the
same under either head. Per-airport K = 4 heads are the delivery on every airport; the pooled
head is not. Artifacts `…/plan_guidance_20260910/step5b_*`. Next: §9 step 6.

### 2026-09-12 — ts_transformer: plan-and-guidance step 5 — the pooled five-airport training (v5.5): the closure transfers, the pooled head costs the home airport

`dev-plan-pool`; design v5.5 §9 step 5, §12.10. Tooling: `run_ts.py plan_cohort` writes the
development cohort a random-anchor plan run needs (the train CLI's flags and its exact
`usable_series → split_by_flight → filter_training_cohort → window set` sequence; every train
flight with no admissible anchor dropped; `<output-dir>/development_cohort.json` +
`data_selection.json` with the reasons; refuses a rolled-table path; checks the config before the
track load) — on KRDU it reproduces the hand-written `step3c_plan_head_full_cohort.json`
exactly; `plan_oracle --by-airport` (`summarize_by_airport`, `summary_by_airport`);
`plan_oracle_pair --common` (a pooled artifact against a single-airport one over the flights
both hold, the base inside the arm, schema v2 with the `cohort` counts).

Measured (one detached chain: the pooled cohort 21,911 / 4,496 flights over KRDU, KSJC, KSTL,
KSMF, KMSY; the K = 4 point head 41 min — one epoch would have done, the truth table never
reads the model; the truth table 292,057 samples / 26,407 flights / 175 MB in 33 min; the K = 4
head at share 0.75, best epoch 117). KRDU val, 60 s anchor, paired on the 1404 flights against
the KRDU-only head: vectored ADE 3087 → 3678 m (+227 m p50, lower on 37 %), established
94.2 → 83.9 %; with the truth's time 2655 → 3684 (+591) — the §9 gate (within the seed line)
fails: the airport-macro pooling takes the home airport's weight. The four other airports
(their first head): straight-in 810–1010 m unassigned, 270–570 m with the truth's time (dt MAE
3–5 s), fully flyable 99.6–100 %; vectored 2.7–4.2 km, established 55–85 % (KSMF 55, KMSY 64).
Decision: the pooled head is not KRDU's delivery; per-airport heads for the other four next
(5b). `docs/code-health-followups.md` §30: the rolled table is encoded once per window set
(six copies on a pooled run). Full suite 1045 passed. Artifacts `…/plan_guidance_20260910/step5_*`.

### 2026-09-12 — ts_transformer: plan-and-guidance step 4 — the assigned time and join (v5.4): the time delivered, the join not in this form

`dev-plan-cta`; design v5.4 §9 step 4, §12.9. The scheduler's assignment reaches the plan path:
`outputs/plan/strategy.Assignment(arrival_time_s — ABSOLUTE on the series clock — , d_join_m)`,
applied by `lockstep_model_policy(assignments=)` at every ask (`assigned_order`: the head's `T_s`
replaced by the assigned remaining time, its `d_join_m` by the assigned join; every other parameter
the head's; the head's own ETA kept as the prediction the rolled flight reports; the budget the
later of the head's time and the assigned one, so a flight the speed cannot bring forward lands
late with X < 0 instead of being cut short of the final). `fly_lockstep` closes the time on the
route in force at every ask (`outputs/plan/guidance/timing.py::close_time`): the route timed from
the aircraft's PROGRESS point (`route.route_time_s(start=)`); the speed lever — one factor on the
plan's held speed (`held_speed_mps`: the instruction's at its fix; on a closing `scaled()` adds a
new held speed reached at `DECEL_RATE_MPS2`), bisected between the floor (`stall_floor_mps`, the
stall margin × the 1 g stall speed at the flight's mass and altitude, or `V_final`) and
`SPEED_MAX_MPS`, the deceleration point moving out with the held speed (`decel_point_for`); the
path lever — `leg_route(stretch_m=)` (a closing through the plain builder's hold / dog-leg, an
instruction leg flying its heading after the fix longer), sized at the floor speed, bracketed over
`STRETCH_PROBES` lays (the builder lays in quanta), a late lay never taken, the plan as laid
re-anchored to the stretched path, the stretch in force kept across re-lays and dropped when the
assignment no longer needs it; X = assigned remaining − closed time (`planUnabsorbedFirstS` /
`…LastS` / `…StepP50S`, `planSpeedFactorFirst/Last`, `planStretchM`, `planStretchDrops`, every
step's record `closure`). `speed_floor.stall_speed_mps` is the one stall-speed expression (the
tensor floor and the scalar closure). `plan_oracle --assign-time {none,truth}
--assign-time-offset-s S --assign-join {none,truth}` (the truth's time and join as the assignment
— an oracle form, `assignment` in the artifact) and the X / speed-factor / stretch rows.

Measured (KRDU val 1404, the 3(g) seed-1337 K = 4 head, lockstep 30 s, 60 s anchor): the truth's
time assigned — dt MAE 35.1 → 10.2 s pooled (straight-in 20.9 → 3.2, |dt| p50 1.5 s), paired ADE
−447 m at the median (lower on 81 %; straight-in 749 → 281 m, vectored 3087 → 2655 m), chamfer
unchanged, fully flyable 100 % in every arm, established 97.3 → 95.0 %; L−1 the same (1802 →
1337). −60 s: the median flight of both strata arrives 61 s early (met), vectored established
72 %. +60 / +90 s: the vectored stratum absorbs at the floor (X p50 0, speed factor 0.77 / 0.73,
the flown arrival 6–22 s short of the assigned), the straight-in stratum cannot (X p50 45.6 /
75.6 s: near `V_final`, no path to stretch inside the RNP box) and reports it. The assigned join
as a `d_join` override: 3451 m with the time, 3876 alone, against 2655 / 3087 — an inconsistent
order under the head's own fix, not adopted. §7: the arrival-time and fully-flyable rows pass,
"assigned time" (1596) and "time + join" (1400) do not. Review (opus) findings fixed before the
measurement: the progress-point timing (from 0 the closure pushed every vectored flight to its
maximum speed and off its route), the quantised builder bracketed, the deceleration point, the
re-anchored plan, the stretch across re-lays, the head's ETA, one stall expression. Full suite
1045 passed. Artifacts `…/plan_guidance_20260910/step4_*`. Next: §9 step 5, the pooled
five-airport training on the K = 4 head.

### 2026-09-12 — ts_transformer: plan-and-guidance step 3(g) — the fan over the next fix (v5.3): the mixture objective adopted, the one-step fan not

`dev-plan-fan`; design v5.3 §9 step 3(g), §12.8. `plan_fan_components` = K (config field,
default 0 = the point head; `--plan-fan-components`; named `plan-v1(fan=K)`): for K ≥ 2 the
plan head's instruction group is a K-component diagonal-Gaussian mixture — K means decoded
as the point head decodes the group, K log σ as `FAN_LOG_SIGMA_MIN + softplus(raw)` (bounded
below, never a dead gradient), K logits — trained by its negative log-likelihood
(`outputs/plan/model.mixture_nll`) in place of the group's L1; the `kinematic` component
keeps its name and becomes an NLL (comparable within a fan run only). `PlanPrediction.values`
carries the top-weight component (`fan_values` / `fan_log_sigma` / `fan_logits` beside it), so
the order, the lockstep and the drawn replay are unchanged in form; `fan_rows` is every
component as a full target vector with its weight and physical σ. The point head (K = 0)
keeps its stored layout and start bit for bit (pinned in `tests/test_plan_fan.py`). A fan
member is the fan ONE STEP DEEP (`strategy.lockstep_model_policy(first_component=k,
first_order=)`: the first order from component k or transformed given the flight's state,
every later one the top-1's; refused with an order hold). New runner `run_ts.py
plan_fan_readout --checkpoint L=<ckpt> --anchor-s 60`: the single-step reading (fix errors,
2σ coverage, per-component usage) and the rolled one — the top-1, the K members and a
DISPLACED CONTROL fan (the top-1's first fix moved 5 km in runway axes at K bearings, the
schedule coordinate with it), with `quantile_fan_readout.geometry_cell` and the top-1 a
member of both sets; only flights whose first order flew a fix are fanned; the artifact is
written before the table (a 40-minute run died formatting a stratum with no fanned flight).

Measured (KRDU val 1404, lockstep 30 s, 60 s anchor, the §12.6 recipe at K = 4, two seeds,
paired within seed against the L1 point head of the same recipe and seed): the mixture's
TOP-1 — vectored ADE 3641 → 3087 m (seed 1337, −123 m paired p50, lower on 57 %) and
3439 → 2837 m (seed 2024, −377 m, 63 %), established 86.5 → 94.2 % and 89.5 → 95.2 %, FDE mean
2803 → 1459 and 1925 → 1356, straight-in unchanged (747 → 749, 736 → 743); the point head's own
two-seed vectored spread ~200 m. The 3(d) gate (2870 m) reached at one seed, missed at the
other. Both heads learn one sharp dominant component (weight 0.62–0.69, σ ≈ 2 × 0.6 km) and
three broad alternatives 6–10 km away; usage 29 / 16 / 3 / 52 % on the vectored stratum. As a
FAN the members are no better than the ring: vectored minADE_4 2454 against 2450, the nearest
member beats the top-1 on 58 % of fanned flights against the control's 79 %; 2σ coverage
98.7 % is the alternatives' width. Decision: the mixture objective is the plan head's recipe
(K = 4; the default stays 0), the one-step fan is not a deliverable. Review (opus) findings
fixed before the commit: the top-1 a member of both fan sets, the shared geometry cell, the
control in runway axes with the schedule coordinate moved, the σ parameterisation, the fan
member refused with an order hold, the K = 0 layout pinned. Full suite 1039 passed.

### 2026-09-12 — ts_transformer: plan-and-guidance step 3(f) — the order hold (v5.3), measured and not adopted; `plan_oracle_pair`

`dev-plan-hold`; design v5.3 §9 step 3(f), §12.7. §12.6's reading: the receding-horizon
head's orders still move from ask to ask and a fix moved over `RELAY_FIX_M` on ONE ask
re-lays the route. The lockstep gains an ORDER HOLD (`outputs/plan/forecast.held_order`;
`fly_lockstep(hold_asks=, hold_flips_only=)`): a materially different order
(`orders_differ`, the re-lay's own test factored out of `order_changed`) is adopted only
once given on `hold_asks` consecutive asks agreeing with each other, or with
`hold_flips_only` only a fix ↔ none flip; the lockstep's own rules — an executed
instruction, the leg cap, the final — are never held. Threaded through
`fly_lockstep_truth`, `rolled.record_lockstep`, `strategy.rolled_predictions_lockstep`,
`plan_oracle --hold-asks N [--hold-flips-only]` and `plan_rolled_windows` (the table header
and `provenance` carry the hold; a pre-v5.3 table reads as `PRE_HOLD_ASKS` = 1 and
`--extend` refuses a different hold). Every step's record says `held` and `flown_fix`;
every rolled flight reports `planHeldSteps` / `planOrderChanges` / `planHoldAsks` /
`planHoldFlipsOnly`. New runner `run_ts.py plan_oracle_pair --base L=<dir> --arm L=<dir>`:
two plan-oracle artifacts joined flight by flight (per stratum the means, the paired Δ p50
and the share the arm is lower on, the identity line).

Measured on §12.6's share-0.75 head, KRDU val 1404, lockstep 30 s, the 60 s anchor: hold 1
reproduces §12.6's artifact (78 of 1404 rows differ by ≤ 0.072 m of ADE, 51 of them at
step 0 — the head's float32 CPU forward is not bit-reproducible between runs). Hold 2 on
every material change: vectored ADE 3641 → 3816 m (paired +57, lower on 40 %), established
86.5 → 60.2 %, FDE mean 2803 → 5470 (L−1: 3719 → 3938, 84.2 → 55.4 %); flips only 3723 m,
79.7 %. The head's fix WALKS (consecutive asks 766 m apart at the median, 2.7 km at p75),
so two asks never agree within 1 km and the step-0 fix stays in force for the whole flight
(2038 of 3143 held steps a moved fix; the worst flight 9 of 10 steps held, ADE 4039 →
15036 m). Neither passes the 3(d) gate; the veto (established down) fires.
`ORDER_HOLD_ASKS` = 1 stays the default, the axis stays. Review (opus) findings fixed before
the commit: the extend check and the provenance name the hold; a held step that re-lays
(off the route) keeps the plan in force with the join in force; the held share is one
definition with no bound that cannot bind; `orders_differ` and the three-way jitter are
pinned in `tests/test_plan_rolling.py`. Artifacts:
`4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step3f_*`. Next: §9 step 3(g),
the fan over the next fix.

### 2026-09-11 — ts_transformer: plan-and-guidance step 3(e) — the head trained on rolled windows (v5.2)

`dev-plan-rolled`; design v5.2 §9 step 3(e), §12.6. §12.5's reading: from its second step
on the receding-horizon head is asked on windows of its OWN flown rows, and trained on
observed windows only it jittered from step to step and sent 29 % of vectored flights to
the time cap. The fix is on the training side. `outputs/plan/rolled.py`: `record_lockstep`
wraps any lockstep policy and records, at every step, each flight's input window
(`rolled_history`) with the target vector the truth defines AT THAT STATE — `targets_at`,
the one definition `targets_from_labels` now reads at an observed anchor too: the
instruction the truth's queue holds there (`TruthQueue`: executed once the aircraft is past
the fix on its heading, `forecast.turn_done_row`'s rule read at one state; a fix behind an
aircraft on the final is none ahead, `forecast.behind_on_final`, the rule `fly_lockstep`
flies), the truth's arrival time less the time flown, the schedule coordinate as the
lockstep tracks it, the flight's own plan. `RolledWindowTable` (one `.npz`, the header
inside; `require_cover`: the window contract and every flight, no partial mode; `extend`
for DAgger's aggregation) and `RolledDraw` (the per-flight per-epoch sha256 coin at
`plan_rolled_share`, its own salt). `run_ts.py plan_rolled_windows --policy truth|model`
writes the table over a checkpoint's split(s); `train --plan-rolled-windows-path
--plan-rolled-share` draws from it (`PlanContext.override` through
`TrajectoryWindows.batch(epoch_seed=)`, passed by the training iterator only; a
substituted sample carries zero truth-grid targets), reports the share realised
(`plan_rolled_training`) and the loss over the val split's rolled windows every epoch
(`plan_rolled_validation`, `OutputStrategy.validation_extras`) beside the observed
objective, which still selects the checkpoint.

- The spine's training-time input is no longer one path's: `OutputStrategy.training_teacher`
  / `fitted_teacher=` are `training_input` (an object with `provenance` and `metadata_key`;
  the control table keeps its on-disk key `fitted_teacher`, the plan table is
  `plan_rolled_windows`). `WindowContext.override` and `OutputStrategy.validation_extras`
  are the two new hooks; `EpochResult` gains `plan_rolled_training` / `plan_rolled_validation`.
- `PlanOutput.plan_rolled_windows_path` / `plan_rolled_share` (together or not at all; named
  `rolled=` / `rolled-share=`; old plan checkpoints load with the defaults);
  `PLAN_TARGET_CONTRACT` lives in `labels.py` (the table is stamped with it); the batch
  context gains `plan_rolled`; `lockstep_model_policy` is the head as a lockstep policy.
- KRDU val, the rolled prediction in lockstep from the 60 s anchor (§12.6): the truth
  policy's table over both splits (97,633 samples, 696 s) and the head at share 0.75 —
  vectored ADE **5391 → 3641 m** (the once-per-leg head 3781, the lockstep ceiling 2184),
  established 55 → 94 % pooled (vectored 49 → 87 %), capped 29 → 9 %; straight-in 747 m,
  chamfer 44. The share is the lever: 0.25 / 0.5 / 0.75 / 1.0 = 3833 / 3862 / 3641 / 3685 m,
  the rolled val loss 0.736 / 0.630 / 0.559 / 0.523 while the observed objective stays at
  0.64–0.65 (paired vs share 0.5: 0.75 −71 m, 1.0 −130 m); DAgger's first round (the round-0
  head's own states appended, 190k samples) +73 m paired and 255 of 601 vectored flights at
  the six-instruction cap. The 3(d) gate stays open (2870 m); what is left is the head's
  geometry (vectored chamfer ~1430 m against the ceiling's ~700).
- The review (opus) moved the label rule into ONE object, `labels.TruthExpert` — the truth's
  policy read at ANY pose: the queue's instruction in force (executed past the fix on its
  heading, or past it on the final; one on top of the aircraft skipped), and the schedule
  coordinate and arrival time as the truth's own at the nearest truth row of the leg in force
  plus the way there — exact at the anchor (so `targets_from_labels` reads the same object
  and the observed and rolled populations agree about every target), defined at a learner's
  state off the truth's path (the draft's "anchor's value less the path flown" ran out on a
  slow learner kilometres from the runway). The pose predicates and their constants
  (`fix_ahead`, `past_fix`, `turn_done`, `on_final_pose`; `LEG_MIN_S`, `TURN_DONE_RAD`,
  `ON_FINAL_XT_M`) live in `labels.py`, `forecast.py` reads them from there. Also from the
  review: the rolled val readout averages each component over its CARRIERS
  (`model.loss_group_carriers`, the groups in `PLAN_LOSS_GROUPS` — by batch size the
  `kinematic` number moved 70 % with the batch size); `extend` checks the whole window
  contract and the step and unions the airports; the table's windows are encoded once per
  window set; `--device` on the runner; `guidance_config` resets the plan run's own fields
  (a rolled-window table on the config was refused on the control config the guidance
  flies under — the first rolled readout died there).
- Tests: `tests/test_plan_rolled.py` — the labels at a flown state are the instruction the
  lockstep flies, the table round-trips and refuses another contract or cohort, the draw is
  deterministic at the share, a training batch carries the rolled window, one run records
  the share and the val readout. Acceptance: the full ts suite 1029 passed (7 m 33 s,
  foreground); stored-run census 114 configs / 72 load, 0 names moved.

### 2026-09-11 — ts_transformer: plan-and-guidance step 3(d) — receding-horizon, lockstep rolling (v5.1)

`dev-plan-lockstep`; design v5.1 §9 step 3(d), §12.5. The user's reading of §12.4: a
single-step head asked once per leg commits a 60 s window's guess at a turn 25–39 km away
for four minutes; ask it again every 30 s instead, and make the oracle faster. So the unit
of rolling is a time step (`LOCKSTEP_S` = 30 s): `fly_lockstep` steps a whole group
together (one guidance rollout of 10 holds per step, `fly_routes(n_segments=, progress=)`),
asks the policy every step (`truth_lockstep_policy` holds the truth's next instruction
until executed; `rolled_predictions_lockstep` runs one head forward per step over every
flight still flying), keeps the route to the instruction in force WHOLE and tracked from
the point reached (`PlanGuidance(progress=…)`), re-lays it only when the order changed
materially (`order_changed`) or the aircraft drifted (`off_route`), and cuts a step at the
execution of its instruction. `plan_oracle --rolling {leg,lockstep}` (lockstep the
default) and `wall_s` in the artifact; `PlanStrategy.forecast` rolls a batch in lockstep.

- KRDU val: the lockstep oracle 1847 m of vectored ADE at L−1 against the leg form's
  1492 (chamfer 700 against 324 m), 2417 against 2141 at the 60 s anchor —
  in 69 s and 79 s for the split against ~90 min each. **The full head (unchanged)
  re-asked every 30 s at 60 s is WORSE than asked once per leg**: vectored ADE 5391 m
  (4991 every 60 s) against 3781, established 55 % against 83 %, its
  lockstep ceiling 2184; straight-in unmoved (887 against 635, chamfer 43 m).
  The head is asked on its own flown windows, which it never trained on; its orders
  jitter and 29 % of vectored flights run to the time cap. The receding-horizon form is
  right (the oracle), the head is not yet a receding-horizon head: training on rolled
  windows is the next item; §12.4's leg-form numbers stay the best prediction.
- `plan_oracle --lockstep-s N` (the step-length axis); a flight ends at its threshold
  crossing; an order never extends the first budget; an aircraft on the final follows the
  final (a fix behind it is executed, one ahead ignored); the review's fixes (the phase
  route follows a re-lay, no floor past the budget, the instruction-leg cap, the cap on
  the state, `leg_route` owns the on-final height rule, one time-cap expression, the join
  distance in `order_changed`, no `leg` key). `leg_route(mid_flight=True)` gates the
  re-lay forms so the leg form and the drawn replay keep the fly-by through the fix.
- Five rules found on the 48-flight smoke (§12.5): the route in force (re-laid every step
  from mid-turn, the Dubins builder flipped its turn direction step after step); a
  converging closing starts its height from where the aircraft is; the height law is
  anchored at the height the route was laid from; the route is laid through a fix wherever
  it is (`ahead_of` turned a near fix into the closing); "executed" is on the outbound leg
  beyond the fix (`passed_fix`), not the along-track projection alone.
- Tests: the lockstep oracle reaches the threshold and a group flies as its members alone;
  the chain test rolls in lockstep. Acceptance: the full ts suite 1024 passed (7 m 23 s,
  foreground); stored-run census 221 configs / 114 load, 0 names moved (the two new
  configs are step 3's plan heads).

### 2026-09-11 — ts_transformer: plan-and-guidance step 3 — the next-instruction readout, the rolled oracle, the single-step plan head (`prediction_output=plan`)

`dev-plan-next`; design v5 §12.4. The route is predicted one instruction at a time (v5):
**(a)** `run_ts.py plan_next_readout` measures the lead a single step must predict (at the
60 s anchor a vectored flight's next fix is 25–39 km / 3–5 min ahead, the median baseline
5 km off; at random anchors 13 km / 112 s); **(b)** `run_ts.py plan_oracle --route next`
rolls the truth's own instructions — each leg to the FIRST pass of its fix (the nearest
route point lay on a later loop), the remaining path re-anchored to the instruction's (the
leg's inflated route asked the closing for 20 km and flew a loop), the closing's join never
behind the aircraft, an instruction a VECTOR (fix + heading on; fly-over semantics
overshot every corner) with its height (each leg descends to it by its fix), no extension
beyond a fix whose heading on points at the join (the onward point overshot it and the
polyline looped back: 30–40 km of closing path on the full run's worst flights), a leg
flown until the AIRCRAFT has executed its instruction (`turn_done_row`; cut on the
route's clock the next leg started 60° off its heading and the plan-route builder answered
with a loop), the closing onto the join a polyline with its corner BEFORE the join — and
on the 48-flight smoke within ~50 m of vectored ADE of the whole-path waypoints flight
(915 against 862; corridor after the join 11.5 % against 7.7 %);
**(c)** the fourth output path: `PREDICTION_PLAN`, `PlanOutput` (two loss weights, run word
`plan · … · guidance · plan-v1`), `outputs/plan/labels.py` (14 targets per anchor, the
validity mask, the no-fix flag; `PlanOrder` back from a prediction, clamps recorded),
`model.py` (the head on the shared backbone, L1 in each target's scale, the flag's BCE),
`strategy.py` (`PlanContext` reads the labels per drawn anchor at batch time, 0.4 ms each;
the DRAWN single-step replay for checkpoint selection; the ROLLED forecast through the
guidance, cut at the threshold; `planOrders` on the record), `plan_oracle --policy model`
and the readout's HEAD columns as the two §7 readings. The oracle's vertical verdict is
v2: the glidepath window inside the FAF, the coded floor before it, the truth's own rows
graded beside every flight (the truth failed the join-graded window on its own level
segment; it fails the pre-FAF floor on 29 % of smoke flights, so no share there is a gate).

- The first head (403 train flights, 30 epochs) sent every ESTABLISHED flight to a made-up
  fix — its "join next" flag was supervised off the final only (§6 as written) — 74 % of
  straight-in flights out of the corridor; the flag is now "no fix ahead" on every sample,
  `d_join` there the anchor's own remaining path, and a closing that starts at or inside
  the join flies the height from where the aircraft is: straight-in corridor 74 → 0 %,
  glidepath 82 → 0 %, legs 6.2 → 1.0 on the same 48. Vectored: ADE 7657 m against the
  paired ceiling's 2293 (a 403-flight head; the full-cohort head is the next number).
- The step-3 code review (2026-09-11) fixed: the drawn replay's height law is the
  controller's own (`guidance.controller.reference_height`, one definition — it ramped to
  the capture height by the fix and put the glidepath on the pre-final leg); the
  capture-height clamp is threaded into the rolled flight (its share read 0 by
  construction); a rolled flight's caps are reported (`horizon_capped`, `planCappedBy`);
  `join_valid` (a mask that could never bind) is gone; `rolled_history` refuses an anchor
  without a full lookback; the plan path is refused off the ENU chart; the target contract
  digests the target names; the strategy refuses a conformal table.
- `fly_rolling`'s join index at the fix's first pass overflowed the route on one flight of
  1404 (the pass at the route's last point) — clamped; the a60s rolled oracle re-run.
- One definition of the strata's short names (`approach_difficulty.STRATUM_SHORT`; three
  private mirrors gone); `Instruction` / `truth_instructions` live in `labels.py`.
- Tests: `test_plan_output.py` (the config contract, the labels → targets → order round
  trip, the loss on the labels and its masks, the context and the drawn replay, one whole
  chain train → checkpoint → rolled forecast → export → evaluate on synthetic arrivals);
  the rolling tests follow the `labels` seam. Full KRDU val: the rolled oracle sits
  333 m of vectored ADE above the whole-path waypoints oracle at L−1 (1492 against 1159)
  and 436 m at the 60 s anchor; the full head (6856 train flights, 67 epochs) rolled at
  the 60 s anchor: vectored ADE 3781 m against its paired ceiling's 1845 m, straight-in 635
  against 584 m, arrival-time MAE 34.6 s; its next fix 2.6 km off at 60 s / 1.6 km
  at random anchors on vectored flights against the median baseline's 6.5 / 10.7 km.
  Acceptance: the full ts suite 1023 passed (9 m 49 s); stored-run census 220 configs / 113
  load, 0 names moved (the one new config is this step's smoke head).

### 2026-09-11 — ts_transformer: plan-and-guidance step 2b — the fixed-K fly-by waypoints oracle, and the oracle at earlier anchors

`dev-plan-turns`; the design's §3b gains the representation, §12.3 the numbers. The three
route parameters say how long a vectored path is, not where it goes (§12.2: chamfer 636 m
from the truth's own three), so the pre-final path gains up to K = 4 fly-by FIXES — each
turn as the point where the two legs it joins intersect, the RNAV way of coding a route
and what a radar vector comes to (`outputs/plan/extractors.extract_waypoints`,
`PlanLabels.waypoints`; a reversal is two fixes; a turn's fix past the join on the
centreline is the turn onto the final and becomes the join). `build_route(waypoints=…)` lays
the polyline through them with the corners rounded at each fix's own speed — **every
fix carries the speed the truth had at its turn** (`waypoint_speeds`), and the schedule
runs through those speeds (`speed_schedule_mps(points=…)`); a vector is a heading and a
speed instruction (`KIND_WAYPOINTS`). Three representations were measured out on the way:
fixes without their speeds were rounded at `V_mid` and cut by the tracker (chamfer
296 m, 31 % of vectored flights out of the corridor after the join); (start, heading
change) drifted every later leg when laid at another radius than the truth's (the vectored
route 4.4 km longer than the plan); a turn's end point was ambiguous about the leg heading.
`run_ts.py plan_oracle` gains `--route {plan,waypoints}`, `--anchor-s N` (one row N seconds
after the slice starts, every flight) and `--anchor-km` (the anytime grid's remaining-path
bin — later than L−1 on a vectored track); `fly_plans` takes one anchor per flight.

- KRDU val, vectored stratum from the truth's own plan at L−1: chamfer 636 → 268 m,
  Fréchet 2166 → 1648 m, ADE 1591 → 1159 m, 2.69 fixes per plan;
  arrival-time MAE 27.3 → 16.1 s (the −21 s early arrival of §12.2 was the
  schedule holding `V_mid` over the vectors), corridor after the join 19.9 %
  (3.4 % under the plan route): the residual is bimodal — the flights the tracker can
  follow sit at ~85 m, the rest at ~900 m where a fly-by corner at the fix's speed is
  tighter than the aircraft can fly (short legs); the next step is the corner (a
  fly-over, or the truth's radius per fix), not more fixes. §8 risk 1 answered for the
  representation, not yet for the guidance.
- The anchors: L−1 censors 59.2 % of plans (joined inside the window), the 60 s anchor
  45.6 %; the 20 km remaining-path bin censors 42.5 % (it is later than L−1 on a
  vectored track). The earlier anchor for step 3 is a shorter window.
- Tests: the extractor on known turns and a reversal, the route's round trip through two
  fixes. Acceptance: the full ts suite 1015 passed (6 m 53 s); stored-run census 219 configs / 112 load, byte-identical.

### 2026-09-10 — ts_transformer: plan-and-guidance steps 1–2 — the procedure skeleton, the plan extractors, the guidance layer, the oracle ceiling on KRDU val (reviewed)

`dev-plan-guidance`, the design's §9 steps 1 and 2 (`4dTrajectory/ts_transformer/docs/2026-09-09_plan_and_guidance_design.md`;
the numbers in its §12). Reviewed (opus) after step 1 (nine findings) and after step 2 (eight
confirmed findings, all applied and re-measured — the six route rules below came out of them).

- **Step 1.** `flight_scenarios.procedure_final.procedure_skeleton` reads the whole coded
  RNAV(GPS) approach (the final to the MAPt with floors and ceilings, GPA, TCH, every
  transition to its merge fix; all 25 runways on this machine read; a transition off its
  merge fix is refused). `outputs/plan/skeleton.py` puts it in a flight's chart (runway axes
  about `target_chart`, the optimizer's 150 m threshold check — both mirrors pinned by
  `tests/test_plan_skeleton_mirrors.py` — the RNP-box distance to the published legs).
  `outputs/plan/extractors.py` reads the eight plan parameters off an observed track with
  their ranges (`PlanLabels`): a join BEFORE the anchor censors the capture height, the side
  and the pre-final path (59 % of KRDU val at L−1); `d_decel` is where the ground speed
  first drops under the target speed + 10 m/s; `V_mid` the mean speed over the path held
  before it. `run_ts.py plan_extractors` measures them per stratum with the median baseline
  the veto reads. KRDU val (1404): straight-in joins inside the window are published
  transitions (87.7 %), vectored joins radar vectors (2.2 %); vectored `L_pre` p50 35.6 km.
- **Step 2.** `outputs/plan/guidance/`: `route` lays a plan's route (the held heading to a
  turn point, the turns onto the join, the final leg) and reads the time the speed schedule
  needs for it; **six rules, each measured in**: the join's intercept is any angle within
  the 30° alignment limit, aligned where the length affords it (five discrete steps left 5
  of 48 flights 2–6 km short); every turn is sized at the speed the schedule has where it is
  flown (at the anchor's radius a downwind flight's base-turn lengths jump by a
  circumference exactly where the real path lies); two arcs over 300° together are a loop
  or a teardrop, not a route (a pose beside the centreline heading in takes the chord);
  inside the RNP box a join closer ahead than the converging leg is flown as the
  converge-then-final route (29 of 73 routed straight-in flights were routed through
  12–24 km of teardrop for 200–1600 m of plan; the tracker cut through them, the route's
  length and time were wrong); the hold and the dog-leg offset are searched coarse-then-fine (the length is
  piecewise smooth; the bisection landed 10–23 km OVER the plan on 3 of 48, +55…+200 s); a
  hold or dog-leg is taken only when it lays the length closer than the shortest path and
  never over it by more than 500 m — the gap is `Route.shortfall_m`, an extra is never
  flown. `controller.PlanGuidance` is the one command hook: an L1 tracker with the route's
  curvature fed forward (bank), the height profile to the capture height and the glidepath
  (load factor), the corridor barrier composed on the final BEFORE the thrust is priced
  (the floor and the drag read the load actually flown), the speed schedule — a
  ground-speed law converted to the dynamics' airspeed — through the speed floor's thrust
  inversion; thrust at idle counted beside thrust over the maximum; ~3 s holds, the hold
  flown on the record (`planHoldS`). `outputs/plan/forecast.py`'s `fly_plans` is the batch
  entry point (the strategy seam). `run_ts.py plan_oracle` flies every flight's own plan and
  grades it as a prediction and as a reference. Moved up as shared: `outputs/dynamics/context.py`,
  the dense query grid in `outputs/dynamics/rollout.py`, `per_flight_hook_diagnostics` into
  `outputs/dynamics/hooks.py`, the Dubins primitives in `geometry/dubins.py` (`dubins_csc`
  gains `end_radius`, `max_sweep_rad`), `forecast_geometry` into `experiments/support.py`.
  KRDU val (1404), the truth's own plan: straight-in ADE 204 m / chamfer 34 m /
  arrival-time MAE 4.8 s (native32: 109 m, 25.9 s pooled), vectored ADE 1591 m (2870),
  99.7 % fully flyable, 99.7 % established at the threshold, the corridor left after the
  join by 1.4 %, the glidepath window by 10.0 % (the height law's from-above capture —
  not yet by construction). The §7 veto does not fire on either stratum; the pre-review run
  (ADE 1113 / 210 / 2710 m, established 96 %, corridor 14 %) is kept as
  `step2_oracle.superseded-20260910T2330Z`. The step-3 decision is in the design's §12.2.
- Tests: `tests/test_plan_extractors.py`, `tests/test_plan_guidance.py` (the envelope, the
  route's alignment sweep, the dog-leg gate, the two-radius primitive, the chord, the bank
  sign), `tests/test_plan_skeleton_mirrors.py`, the seam's `test_procedure_final.py`.
  Acceptance: the full ts suite 1010 passed (7 m 03 s); stored-run census 219 configs / 112 load, byte-identical.

### 2026-09-10 — ts_transformer: what the rollout needs moves out of `outputs/control/` (plan-and-guidance, step 0)

The first commit of the plan-and-guidance path (`4dTrajectory/ts_transformer/docs/2026-09-09_plan_and_guidance_design.md`
§11), on `dev-plan-guidance`. The design's guidance layer consumes the point-mass rollout, the
command hooks, the dimensionless command box and the condition vector, so under the membership
rule (a module belongs under `outputs/control/` only if EVERY consumer is control-specific) they
are no path's: `outputs/control/dynamics/` → `outputs/dynamics/`, `outputs/control/constraints/`
→ `outputs/constraints/`, `outputs/control/conditioning.py` → `outputs/conditioning.py`,
`outputs/control/envelope.py` → `outputs/envelope.py` (the envelope goes with them because the
backends' newton conversion and the speed floor read it). Pure moves — every qualified import
rewritten, the live documents with them; the control strategy imports from the new place.
`tests/test_architecture.py` gains the rule that a shared part of `outputs/` imports no path.
Acceptance: full suite 994 passed; 219 stored `history.json` configs give byte-identical run names,
slugs and loadability.

### 2026-09-10 — ts_transformer: `outputs/` was git-ignored — fourteen §4.2 modules had never reached the branch

Found by the review of the §4.5–§4.7 commits on `dev-pkg-review`. `4dTrajectory/.gitignore`'s
bare `outputs` rule (meant for `4dTrajectory/outputs/`, the optimizer's artifacts) also
matched `ts_transformer/outputs/`, so every file the §4.2 strategies commit (`947c907`)
CREATED there — `outputs/__init__.py`, `base.py`, `duration_heads.py`, the three `strategy.py`,
the `state/` and `closure/` `forecast.py`/`loss.py`/`__init__.py`, `control/supervision.py`,
`control/forecast.py`, `control/loss/objective.py` — was silently left out of the commit; only
the files `git mv`ed in stayed tracked. The suite passed because the worktree had them. A
fresh clone of the branch could not import the package. Now re-included (`!ts_transformer/outputs/`,
the same fix the grouping commit made for the bare `data` rule) and committed: full suite 992 passed, 1 skipped; the modeling suites collect 1,370 tests with no error.

- `tests/test_architecture.py`: an absolute `from ts_transformer.training import train` now
  registers `training.train` (only `from ts_transformer.training.train import …` did), so the
  grouped layout's natural import form cannot slip a loop module under `outputs/`.
- `cli/benchmark_batch.py` loses its dead `__main__` guard (its `sys.path` bootstrap went with
  the §4.5 move; the door is `python -m ts_transformer benchmark-batch`).
- Root `pyproject.toml`: `norecursedirs` adds `archive`, so `./run_all_tests.sh` (which
  collects `4dTrajectory` whole) no longer aborts on the archived scene-encoder test's frozen
  imports — archived code stays off the import path, as `test_architecture.py` requires.
- After the fast-forward into `dev-leg-ctrl`: `flight_scenarios/tests/test_datum.py`'s seam test
  read `ts_transformer/dataset.py` by path and now reads `data/dataset.py`. `./run_all_tests.sh`
  in the main tree: modeling + backend 1,799 passed with the three failures the base commit
  already had (`test_optimizer`'s numpy 2.x scalar, the reference-record `arr_airport` check in
  `test_scenario_optimization`, the harvest checkpoint start in `test_download_landings` — all
  three reproduced at `e996d77`); aeroviz-4d/python 157 passed.

### 2026-09-10 — ts_transformer: the package grouped by plane — `data/`, `geometry/`, `backbone/`, `training/`, `inference/`

The last step of the package review (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.2's layout, resolution paragraph), on `dev-pkg-review`. Pure moves: every module keeps its
name and content, only its directory changed, and every qualified import across the repository
was rewritten (the package, its tests, the docs scripts, the runners, the publisher, `run_ts.py`).
Nothing on disk changed: full suite 992 passed, 1 skipped, and the 219 stored `history.json` configs give
byte-identical run names, slugs and loadability.

- `data/` (the data plane: `dataset`, `splits`, `data_provenance`, `channels`, the frames, the
  grids, the anchor axis, eligibility, the conditionings, `synthetic`, `approach_difficulty`,
  `fixed_dt_supervision`, `batch_contract`), `geometry/` (the corridor, arc length, the metrics,
  flyability), `backbone/` (`adapters`, formerly `models.py`, and `vendor/`), `training/` (the
  loop, the objective, the replay, the probe, cross-validation, the experiment index),
  `inference/` (`forecast`, `calibration`, `export`, the test-release protocol, the readout
  builders); `batch_benchmark.py` is `cli/benchmark_batch.py`. `outputs/`, `cli/`,
  `experiments/` and the top level (`config`, `run_naming`, `io_utils`, `repo_layout`) as before.
- A group's `__init__.py` re-exports nothing, so the torch-free boundaries
  (`tests/test_import_boundaries.py`) hold; `tests/test_architecture.py`'s layering rules are
  spelled in the new names.

### 2026-09-10 — ts_transformer: review §4.6 — `tests/support.py`, the 5,855-line test file split by topic, the suite green

Step 6b of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.6, resolution paragraph), on `dev-pkg-review`. Tests only: full suite 992 passed, 1 skipped — the same count as before the split, 993 collected.

- **`tests/support.py`** (T4-24): `fake_data_provenance` (nine byte-identical copies, now taking
  the airports the one general copy took), `dynamics_context` (three), `terminal_contexts` (two),
  imported as `ts_transformer.tests.support`. The per-file `_config` / `_series` recipes stay.
- **`test_ts_transformer.py` → 23 single-topic files** (T4-26), by its own section headers and
  the test order, every test verbatim and each file carrying only the helpers it references:
  evaluation protocol, channel contract, windows, state objective, data loading, anchor policies,
  validation replay, control heads and config, capacity report, control objective, arc geometry,
  common-grid selector, config contract, auto batch, pipeline recipes, experiment index,
  common-grid control, cross-validation, metrics spread, forecast paths, export seam, end to end,
  and the 2026-09-09 review's pins.

### 2026-09-10 — ts_transformer: review §4.5 — the runners are `ts_transformer/experiments/` behind `run_ts.py <name>`

Step 6a of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.5, resolution paragraph), on `dev-pkg-review`. Nothing on disk changed: full suite 992 passed, 1 skipped (the two `test_cta_from_quantiles` tests that loaded the fan readout by its old root path now import `experiments.quantile_fan_readout`), and the
219 stored `history.json` configs give byte-identical run names, slugs and loadability.

- **The 21 live `run_ts_*.py` runners are `experiments/<name>.py`** (`pipeline`, `cv`,
  `frame_ablation`, `anytime_curve`, `eta_calibration`, …; `plot_ts_results.py` is
  `experiments/plot_results.py`) behind ONE entry point: `python run_ts.py <name> [args]` at the
  repository root, or `python -m ts_transformer.experiments <name>` with `4dTrajectory/` on the
  path; `run_ts.py --list` prints the names with their first docstring line. The 22 hand-written
  `sys.path` bootstraps are gone. **`ts_transformer/repo_layout.py` is the one definition of the
  repository's paths** (`REPO_ROOT`, `HARVEST_ROOT`, `OPT_OUTPUTS_ROOT`, …, `discover_k_airports`)
  — `batch_benchmark` used to import the pipeline RUNNER for them, the one package-into-runner
  edge, now banned by `tests/test_architecture.py`. `experiments/support.py` re-exports them and
  holds `series_digest`; `parse_airports` (five byte-identical copies) is `cli.common`'s; the
  three private `file_sha256` and four `_write_json_atomic` copies read `io_utils`'s.
  `write_reports` (×3) stays per runner — three CSV schemas under one name.
- **Archived:** `archive/flight_model_paired_2026_09/` (the one-shot paired flight-model
  comparison, T4-27; its result is `CLAUDE.md`'s `control_dynamics_model` row).
- **Tests beside the runners:** the six `trajectory_data_process/tests/test_ts_*.py` moved into
  `ts_transformer/tests/` (their own path preambles dropped — `conftest.py` covers them); the
  three tests that loaded a runner by file path import it by name.
- **The twelve red `test_ts_pipeline.py` fixtures are fixed** (T4-23 — fixture rot, not bugs): the
  fixture harvest writes the lateral-pass roster beside the manifest and stamps the eligible-set
  digests the runner has required since 2026-09-08; the printed loss line is the state path's;
  the default grid is 45 candidates; the directory test uses a non-default selection metric;
  the label assertion no longer expects the default frame spelled. **The suite's exit code carries
  information again.**
- Live documents (`CLAUDE.md`, the package `CLAUDE.md` / `README.md` / `OPEN_ITEMS.md` /
  `ENGINEERING_NOTES.md`, `docs/code-health-followups.md`, the arm files' comments, the publisher's
  docstrings) name the new door; dated reports, the changelog and the archive keep the commands
  they were written under.
- **Not done:** `TrainingPlan` keeps its keyword constructor, and the pipeline's `_*_tag`
  directory grammar stays — recomputing it from `run_naming.run_slug` would move every pipeline
  output directory `--skip-train` / CV reuse reads (a decision for the pipeline's owner).

### 2026-09-10 — ts_transformer: review §4.4 — the training loop and the predict command extracted into named steps

Step 5 of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.4, resolution paragraph), on `dev-pkg-review`. A pure extraction, statement for statement,
so nothing on disk changed: full suite 972 passed (the 12 known `test_ts_pipeline.py` fixtures red before and after); the epoch record, the RNG order and the prediction
directories are the ones the monoliths produced.

- **`train.fit_model` (506 lines) → six functions and three records.** `prepare_session`
  builds a `TrainingSession` (the window sets, the validation plans, the model, the optimizer
  and scheduler, the procedure multipliers, the teacher — in the order the loop always built
  them); `describe_session` prints the header; `train_epoch` returns a `TrainEpoch`,
  `validate_epoch` a `ValidationEpoch` (the selection metric included), `procedure_update`
  the dual step's record, `describe_epoch` the epoch's lines; `fit_model` itself is the loop
  and the selection, 124 lines.
- **`cli/predict.run_cli` (467 lines) → `load_predict_checkpoint`, `load_predict_series`,
  `parse_predict_options`, `predict_sets`, `write_prediction_sets`, `report_predictions`**, and
  a 22-line `run_cli` that calls them in order. `parse_predict_options` holds every
  flag-combination rule (the 24 `parser.error` calls) and returns a frozen `PredictOptions`:
  the `ForecastOptions` every forecast is asked, the arms decoded beside the top-1 records
  (latent modes, N(0, I) controls, the quantile fan, the shuffled-latent diagnostic, the
  posterior and label oracles), and the cohort the CTA offset skipped. The decoders take the
  options, never `args`; a fan leaf's forecast options are `replace(options.forecast, …)`.
- **`SplitPredictionReplay.truth` / `.mask` removed**: write-only — a decode and a device→host
  copy of the batch targets per batch per airport per epoch that no metric read (the metrics
  compare against the validation plan's common-grid truth).

### 2026-09-10 — ts_transformer: review §4.2 — one strategy per prediction path under `outputs/`; the spine stops branching on `prediction_output`

Step 4 of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.2, resolution paragraph), on `dev-pkg-review`. Nothing on disk changed: full suite 972 passed (the 12 known `test_ts_pipeline.py` fixtures red before and after), and
the 219 stored `history.json` configs give byte-identical run names, slugs and loadability;
the prediction-record and `history.json` schemas are unchanged.

- **`outputs/`**: a lazy registry (`outputs.strategy(config)`), the `OutputStrategy` interface
  (`outputs/base.py`, one method per former branch site) and one package per path —
  `outputs/state/{strategy,model,loss,forecast}`, `outputs/closure/{strategy,model,geometry,
  profile,forecast}` (the former `closure_output` / `closure_geometry` / `closure_profile`),
  `outputs/control/{strategy,supervision,forecast,loss/objective,…}` (the former `control/`
  package, plus what `dataset`, `forecast` and `objective` carried for it). The point and
  quantile duration heads both paths build are `outputs/duration_heads.py`;
  `prediction_outputs.py` and `anchor_eligibility.py` are gone.
- **The spine calls the strategy** where it branched: `models.build_model`; `dataset`
  (`self.context = strategy.bind_windows(self)`, `batch()` asks it for the context row and the
  dense supervision, `__getitem__` deleted — `batch([i])` is the one door); `batching`'s probe;
  `objective`'s `target_contract` / `loss_component_names` / `prediction_loss_components`;
  `forecast.forecast_approaches` with ONE `ForecastOptions` value in place of seven keyword
  arguments; `validation`'s replay; `export`'s record fields; `train.fit_model`'s
  `check_trainable` / `training_teacher` / `epoch_config` / `training_diagnostics` /
  `epoch_record` / `checkpoint_metadata`. `dataset` imports nothing under `outputs/` any more
  (both import cycles are gone), and a fourth path is one package with no spine edit.
- **Layering, enforced** (`tests/test_architecture.py`): nothing under `outputs/` imports the
  loop / replay / export / CLI; only a path's strategy seam (`strategy`, `forecast`,
  `supervision`, `loss`) reaches `objective` / `forecast` / `models`; the registry and the base
  import no spine module at runtime; `dataset` reaches only the registry.
- Every import of a moved name was rewritten across the package, the tests, the docs scripts
  and the root runners; the now-shared helpers lost their underscore (`forecast.history_at_anchor`
  / `history_batch`, `outputs.state.forecast.forecast_state`, `outputs.control.forecast.forecast_control_batch`,
  `outputs.closure.forecast.forecast_closure_batch`).

### 2026-09-10 — ts_transformer: review §4.3 — `TSConfig` split into typed views; one ownership rule replaces twenty-three per-field checks

Step 3 of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§4.3, resolution paragraph), on `dev-pkg-review`. Nothing on disk changed: full suite 970
passed (the 12 known `test_ts_pipeline.py` fixtures red before and after), and the 219 stored
`history.json` configs give byte-identical run names, slugs and loadability.

- **The flat dataclass stays** (the vendored networks, `run_naming`, the CLI and every stored
  checkpoint read the flat dict). `__post_init__` now builds the views that OWN the fields —
  `CohortSpec`, `BackboneSpec`, `TrainingSpec`, and one `OutputSpec` per path (`StateOutput`,
  `ClosureOutput`, `ControlOutput` with `DurationSpec` / `DynamicsSpec` / `ControlObjective` /
  `HookSpec` / `LatentSpec`) — exposed as `config.cohort/.backbone/.training/.output`. Each
  view's `__post_init__` carries its own vocabulary, range and intra-view rules (the τ/RK4
  bound on `DynamicsSpec`, the four hook rules on `HookSpec`, the "mean nothing without a
  latent" rule on `LatentSpec`); the six rules that read two views sit on
  `TSConfig._validate_cross`. The five 775-line validators are gone.
- **One ownership rule** (`_validate_ownership`): a field owned by another output's view is
  refused off its default — "`control_command_hook='barrier'` belongs to the control output;
  prediction_output='state' never reads it and would still carry it into the checkpoint and the
  run name". It replaces the eleven "belongs to output X" and twelve "supported only by
  prediction_output='control'" checks, and closes the class C-1 / C-2 / C-4 came from by
  construction. `from_dict` normalises a stored config's foreign-output fields to their defaults
  (unread under that output by definition; measured on every stored config that loads: none
  moves). The partition — every field on exactly one view — is checked at import.
- **Not adopted:** sum types per variant (`Factorized(floor)`, one `DynamicsSpec` per backend).
  C-4's floor stays a field: every recipe pins `0.0` under `uniform`. Defaults unchanged (user
  decision). `run_ts_control_basis_oracle.py` no longer hands a control config
  `closure_labels_path=None` (it goes through `from_dict`, which normalises the seed
  checkpoint's foreign fields).
- Tests: eight assertions repointed from the per-field messages to the ownership message; the
  fixed-dt procedure-penalty test now builds a config whose only violation is the grid.

### 2026-09-10 — ts_transformer: review §5 — four axes frozen, the FAF gate and dead code deleted, the scene encoder archived

Step 2 of the package review's order (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`
§5, whose new resolution paragraph is the per-axis record), on `dev-pkg-review`. Nothing on
disk changed: full ts suite 970 passed, the 12 known trajectory_data_process/tests/test_ts_pipeline.py fixture failures red before and after (run_ts_cv.py --frame now offers COORDINATE_FRAMES_AVAILABLE), and the 219 stored `history.json` configs give
byte-identical run names, slugs and loadability before and after.

- **Frozen** (stored checkpoints load, predict and publish; a NEW run cannot select the value —
  `PREDICTION_OUTPUTS_AVAILABLE`, `INTENT_CONDITIONINGS_AVAILABLE`,
  `CONTROL_STATE_LOSS_GRIDS_AVAILABLE`, `COORDINATE_FRAMES_AVAILABLE`, all wired through
  `cli.common._NEW_RUN_VOCABULARIES`, so both CLI doors refuse them): the `closure` output,
  the `intent_conditioning=truth-*` oracles, `control_state_loss_grid=fixed-dt`, and the
  `airport-enu` / `runway-aligned` frames. `run_ts_pipeline.py` and `run_ts_history_ablation.py`
  offer the same vocabularies.
- **Not frozen, correcting the review's table:** `control_state_supervision_clock=observed` and
  `control_dynamics_backend=scaled-transport-chart-velocity` are pinned by every simple-v*
  recipe (the lag model needs the transport-chart backend); `corridor-bounded` is the state
  path's adopted candidate default.
- **Deleted:** the FAF-distance corridor gate end to end (`corridor_gate=faf` was never set in
  any stored run and FAF-gating wrecked vectored flights) — `final_approach_geometry.membership`
  has one gate and no `gate`/`d_faf` arguments, `FINAL_APPROACH_KEYS` lost `final_approach_fix_m`,
  `dataset` no longer reads the CIFP FAF, and the `corridor_gate` field is gone from `TSConfig`
  (a retired constant on load: the 67 stored configs carrying `on-final` drop it; a `faf` would
  be refused); `train_only_diagnostics.py` (no caller); the `_initialize_control_head(bank_rad=,
  feature_std=)` and `arm_steps(label, config)` parameters nobody passed.
- **Archived:** `archive/scene_encoder_2026_09/` — `scene/features.py`,
  `run_ts_scene_explainability.py` and its test (the L4 gate failed, the encoder was never
  built); `intent_explainability.py` stays live for `docs/phase0_intent_diagnostics.py`.
- **Kept, correcting §4.7:** `RolloutStateView.reference` has a live producer (the trombone's
  reference-rollout estimator, L3.f).

### 2026-09-09 — ts_transformer: the package review's remaining bugs fixed (A-1, A-2, B-1–B-4, C-1…C-20), and the package became a package

The 2026-09-09 review (`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`;
its new §7 is the finding-by-finding ledger) had 4 number-changing, 4 crash and ~20 contract
findings. A-3/A-4 were fixed at `c544db0`; this entry is the rest, on `dev-pkg-review`. Rule
applied throughout: a fix that could refuse a stored artifact was first counted against the 219
stored `history.json` configs (none refused), and no stored run's name or directory moved.
971 ts tests pass (`4dTrajectory/ts_transformer/tests` + the four `trajectory_data_process` ts test files); the 12 failures left are the pre-existing red fixtures of `trajectory_data_process/tests/test_ts_pipeline.py` (audit T4-23), verified to fail identically on the untouched tree.

**Package review §4.1 — `ts_transformer` is a package (same branch, second commit).** No
`__init__.py`, 63 files inserting `ts_transformer/` itself into `sys.path`, and module names
that were global (`config`, `dataset`, `train`, `models`, `metrics`, `validation`, `export`,
`calibration` — shadowed silently by any same-named module earlier on the path). Now
`ts_transformer/__init__.py` + `4dTrajectory/pyproject.toml` (editable install like `geokit`,
optional: the bootstraps insert `4dTrajectory/` instead), and every import in the package,
its tests, its `docs/*.py`, the 23 root runners, the publisher and the
`trajectory_data_process` ts tests is qualified (`from ts_transformer.config import …`).
Mechanical and behaviour-free: 973 ts tests pass (the same 12 pre-existing `test_ts_pipeline.py` fixtures fail before and after), and the 219 stored
`history.json` configs give byte-identical run names, slugs and loadability before and
after. Two new architecture tests refuse a flat import of a package module and a bootstrap
that puts `ts_transformer/` on the path — because with the directory on the path the flat
names would resolve to a SECOND copy of every module beside the qualified one. Checkpoints
are unaffected (`weights_only=True` payloads hold no module paths).

**Numbers that were wrong on a live path.**
- **A-1** `run_ts_pipeline.py`: `control_dynamics_model` was emitted to the training subprocess
  but missing from BOTH rebuilt override dicts, so a first-order-lag cell was rebuilt as
  point-mass for its label, its `--skip-train` reuse check (never reused, always retrained) and
  its CV reuse; and no directory/category tag read the model, so a lag cell overwrote its
  point-mass twin. One `TrainingPlan._plan_overrides` now feeds all three; a lag cell wears
  `_lag` (the point-mass tag is empty and no pipeline-shaped directory on disk held a lag run,
  so nothing is orphaned). The plan's never-emitted `control_bank_time_constant_s` is gone.
- **A-2** the fixed anchor: `fixed_anchor_common_truth`, the terminal-velocity weights, the
  arc-length diagnostics, `filter_training_cohort` and `fit_model`'s cohort check all restated
  `seq_len − 1` and ignored `minimum_anchor_index`, while the windows anchored at the floor.
  Under `run_ts_history_ablation.py` (whose point is identical anchors across candidate
  `seq_len`) the selection metric therefore scored predictions against a truth taken
  `(max L − L)·dt` earlier than their anchor. Now `dataset.fixed_anchor_index` is the one
  definition, `FixedAnchorTrajectoryWindows.anchor` / `.anchor_indices` expose it, every
  `fixed_anchor_*` function takes `anchor=` as a REQUIRED argument, and the validation plan's
  truth is built at the dataset's own anchors. **Every number that runner published before this
  is stale.** Code-health #26 (a pure rename to `default_anchor`) would have preserved the defect
  and is superseded.

**Crashes on reachable paths.** B-1 `probe_dynamics` now takes the config and carries the
imitation / heading-rate / CTA keys under the dataset's own conditions (a test pins its key set
equal to a real batch's — `--batch-size auto` died with a bare `KeyError` on every custom arm
that weighted either term). B-2 the heterogeneous probe uses `dataclasses.replace`, so a
quantile head's `duration_quantiles_s` survives. B-3 `write_batch` builds the accuracy block
BEFORE the first record file, so a non-finite ADE refuses the batch instead of leaving a record
directory with no `summary.json`. B-4 the heading-rate target refuses a one-sample remainder
(it was a NaN at weight one).

**Contract holes** (§7 of the review has the file for each): the velocity and imitation weights
are refused off `true-time-position` like their siblings (C-1); `off` is held to the barrier-gain
rule and a hard saturation under no hook is refused (C-2); `validation_common_grid_points` names
the run (`grid-points=`) and `run_naming.KNOWN_UNNAMED_FIELDS` guards the reverse direction at
import (C-3); a `corridor_gate` nothing reads is refused (C-4); predict's `--aircraft-type` is
written into the config beside the records and every predict directory states the same
`skipped` through one emitter (C-5, C-6); the test-release ledger's two checks that could never
fire are gone (C-7); a frame-ablation arm resumes only when `history.json` exists AND its
declared fields agree with the trained config — a checkpoint without a history is refused by
name, nothing deleted (C-8); the literal `invalid_flights: 0` and the segment mask that could
never be False are gone (C-10); `project_onto_final` is threshold-relative under `airport-enu`
(C-11); the actuator-τ rule is the RK4 bound it cites, `h/τ ≤ 2.785`, not `τ ≥ h` (C-13);
`control_state_supervision_clock` is a required serialized field (C-14); a calibration stratum
is deployable only if calibrated at EVERY α, and a present-null covariate is refused rather than
read as False (C-15); the clustering silhouette states its subsample (C-16); an all-zero
imitation mask is documented on both sides (C-17); `fixed_anchor_fraction` reads the floor
(C-18); predict refuses repeated `--data` with `--airport` like train (C-19); and the C-20
small items (present-null `input_channels`, the inert offsets line, the zero-ground-speed
`gamma` fallback, `strictly_increasing` on a decreasing clock, the CTA mode the dataset could
not fill, the `anchor_state` rename, `generated_at`, the `overlap` claim in `CLAUDE.md`).

**C-3 finished the same evening (decided): the three identity-bearing fields name the run**
(`lr-patience=`, `lr-factor=`, `anchor-min-future=`); the named recipes pin the scheduler pair,
so recipe runs are untouched. Measured over the 219 stored `history.json` configs before
landing it: 146 display names / slugs moved (75 gain a spelled token, 71 only their folded
`+N more` count and slug hash), 0 loadability changes, no directory or category key moves;
132 published frontend categories carry the old label until relabelled (109 publisher-managed
via `publish_ts_experiment_trajectories.py --refresh-labels-only`, 23 hand-published `ts_*`
via `docs/relabel_published_categories.py`; labels only). C-12 was decided too: the
grader's load-factor floor stays 0.5 and the head's box 0.2; the plan-and-guidance guidance
layer reads the grader's envelope.

**Left as user decisions, recorded in `4dTrajectory/ts_transformer/docs/OPEN_ITEMS.md`:**
refusing `control_duration_uniform_floor` under `uniform` (the
recipes pin 0.0 against a default of 0.8, 88 stored runs); binding the test-release ledger to the
checkpoint digest rather than its directory (needs a registry outside the run directory);
renaming `cv_results.json`'s `mean_/std_val_macro_loss` (a schema bump; two stored files);
and the review's §5 retirements and §4.2–4.5 restructuring.

### 2026-09-09 — ts_transformer: review A-3 and A-4 fixed — one terminal supervision contract; the barrier is confined to the hard gate under a trombone

Two of the four number-changing findings of the 2026-09-09 package review
(`4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md` §2 A),
on `dev-pg` (`c544db0`), each with the review's own measurement as a test; 909 ts tests pass.

**A-3 — `dataset._build_supervision`.** The observed threshold crossing returned early at a
flat `1/6` on all six channels, while a fitted crossing carried
`fitted_terminal_position_weight/3` on its position channels and nothing on velocity — two
contracts chosen by ADS-B coverage, 2.5× apart on the terminal position weight (1 of the first
60 KRDU arrivals). Now the contract is one: a measured row keeps `1/6` on all six channels, an
extrapolated row is position-only at the tail weight, and the TERMINAL row adds the terminal
emphasis on position whichever way it was obtained (so an observed crossing reads
`1/6 + 1/3` on position, `1/6` on velocity). Any state/control run trained from here on carries
it; stored checkpoints are unaffected (a data-side contract, not a config field). The
seed-pinned `test_the_loop_keeps_the_epoch_with_the_lowest_grid_mean` moved to data seed 4: under
the new weights seed 3's run improves on both metrics through epoch 10 and can no longer show
the disagreement it exists to show; seed 4 separates them (grid epoch 7, L−1 epoch 9) for every
torch seed 0–5 (measured, scratch script).

**A-4 — `control/constraints`.** `composite.py` claimed the barrier and the trombone have
complementary gates; under `hook_saturation=soft` that was false — the barrier's soft
alignment shoulder (non-zero for 30–40° of misalignment) is exactly the trombone's admission
band (> 30°), and there the barrier blended a correction the trombone then overwrote
(re-measured: 12 km back, 500 m right, 33° off course, 6 s hold — soft weight 0.148, barrier
bank +0.112 rad, the trombone's own answer moved 0.7° from reading the barrier's coordinated
load). `build_command_hook` now builds the barrier `confine_to_hard_gate=True` whenever a
trombone is a member: its blend is multiplied by the HARD on-final gate, so inside the gate it
is the standalone soft barrier to the bit, outside it is silent, and the trombone may engage
only where that gate has not opened — complementary by construction, the order cannot change
the answer, and under `hard` it is the identity. `barrier` alone and `barrier+speed-floor` are
untouched. The "25 deg turn cap" cited in `composite.py` / `config.py` was 15° in the code;
fixed. **The L3.e-r / L3.f-r reruns on disk were flown before this fix** (soft, the adopted
form); the magnitude on their readouts was measured by a pre-registered one-arm check the same
evening (`4dTrajectory/outputs/KRDU/experiments/l3f_a4_check_20260909`, the +60 s L3.f arm on the
fixed tree, paired): every aggregate within 1 % (fully flyable 20.01 % both, `tromboneDelayS`
p50 134.9 s both), the 842 flights established at the anchor byte-identical, 395 of 497 vectored
records changed, and L3.f-r's advantage over L3.e-r unchanged — the overlap was not its source.
Confining the barrier costs slightly (12 fewer flights on the final, ~1 % more stall/thrust
samples): the price of a correct composite, recorded, not reverted.

### 2026-09-09 — ts_transformer: L3.e-r / L3.f-r reruns read; the hook line is closed, the route decision moves to plan-and-guidance

Under the corrected cut (the entry below) both path-stretch campaigns were rerun
(`4dTrajectory/outputs/KRDU/experiments/l3e_path_stretch_20260909r`,
`l3f_path_stretch_ref_20260909r`; the flawed directories and, with the user's approval, the
unpublished latent sample subtrees are deleted — 39 GB freed). The corrected numbers are much
worse than the withdrawn ones, which is the point: the old cut had discarded the second half of
every vectored approach (+0 samples 389,909 → 780,672). Three-hook stack, fully flyable **46 %**
at offset 0 and **20 %** at +60 s (was 78 / 52 %); at +60 s **30 % of flights (≈ 80 % of vectored)
never become established on the final** and their rollouts run on for tens of kilometres;
`tromboneStretchM` p50 = 0 — the median flight gets no stretch. The reference-rollout surplus
(L3.f-r) is adopted as the trombone's sizing — 7–10 points more flights on the final, 9–13 %
fewer stall / thrust samples, vectored |xt| p95 −23–29 %, `tromboneDelayS` p50 386 → 136 s —
but both arms fail nearly every pre-registered gate (L3.e: 1 of 5; L3.f: 0 of 2). Reading: a
hook admitted only where the path is already 30° off course cannot choose a route for a flight
whose predicted path never turns onto the final; that is the route builder of
`2026-09-09_plan_and_guidance_design.md` (v3 today, with §11 placing it inside the package
review's `OutputStrategy` target). Final report §7.7 carries a correction notice and §7.7b the
reruns; the eight corrected arms are published to the KRDU picker (`stack-r*`, `ref-r*`).

### 2026-09-09 — ts_transformer: "it crossed the threshold" is the plane AND the final — the plane-only rule fired ABEAM, on the downwind

**The defect, measured on real records.** Two consumers ask a trajectory where it lands:
`forecast.cut_at_threshold_crossing` (`predict --truncate-at-threshold`, built 2026-09-08) and
the trombone's reference-rollout path length (`control/constraints/trombone.py`,
`trombone_surplus_reference=reference-rollout`, built 2026-09-09, the entry below). Both asked
the same wrong question: "the closest horizontal approach among the rows with along-course
distance `d ≤ 0`, within the first such run" — the threshold PLANE and nothing else. A vectored
flight's downwind runs parallel to the runway course and opposite it, several kilometres abeam,
and passes `d = 0` out there. On `l3e_path_stretch_20260908/L3e_stack_p60s_pred_val`, **96.5 %
of the vectored cuts lay more than 1 km from the threshold — median `|xt|` 8.7 km, a median
1.75 km above it** — while the straight-in cuts were clean (`|xt|` p95 **50 m**). So the L3.e /
L3.f vectored flyability and geometry were read on a window that ended on the downwind, and the
reference path length `L_ref` was measured to an abeam point: **~12 km on a 25 km approach**,
with `tromboneRefNoCrossing` at **31.6 %**.

**The fix: one rule, one function.** `final_approach_geometry.threshold_crossing_index(d, xt,
cos_align)` returns the first crossing row, the end of its run, and whether there is a crossing
at all. A crossing is `d ≤ 0` **AND** inside the `on-final` gate there — the membership cone
(`MEMBERSHIP_K`, floored at `MEMBERSHIP_FLOOR_M`) and `ALIGNMENT_MAX_DEG`, read from the gate
rather than restated beside it. The cone is what rejects the abeam pass; the alignment rejects
a plane crossed on a heading that is not the final's. The refinement inside that run stays the
caller's: closest horizontal approach for the record cut, `d = 0` interpolated inside the
crossing segment for the reference path. A trajectory that never satisfies it has NO crossing —
the existing "never reaches" semantics: the record is left WHOLE with `truncatedAtThreshold:
false`, the reference is cut at the end of its own schedule and flagged.

**What moves and what does not.** Straight-in records are bit-identical: there the first row
past the plane is inside the cone and aligned, so both rules answer the same row (checked
end-to-end on a straight-in synthetic rollout — cut row 303 either way, 287 m off the
centreline). Vectored records that never turn onto the final now read as what they are — no
crossing, no cut — instead of being cut 8.7 km abeam. `hook_trombone_ref_no_crossing` now means
"the reference never got ONTO the final", which is a stronger claim than "it never passed the
plane" and will read HIGHER, not lower, on arms whose rollouts reach the threshold across the
course: measured on the package's 48-segment fixture (a flight flown straight at the threshold
at 45° to the runway) it is 0 → 1, while the estimate itself does not move, because a straight
reference has no detour either way. Every path that does not use the crossing is untouched to
the bit: the L3.f equivalence harness, extended with the two `reference-rollout` replays the
crossing actually reaches (15 cases, both trombone stacks trained and replayed, soft and hard),
reports **24 of 2344 leaves moved — every one of them `tromboneRefPathM` /
`tromboneRefNoCrossing` / `tromboneDelayS` inside those two cases**, and nothing at all
elsewhere: no trajectory, control, record or state digest anywhere. Those synthetic references
overshoot the threshold wide of the final rather than flying a downwind, so what moves on them
is the FLAG (0 → 1) and 8–58 m of path (0.1–0.3 %, the cut moving to the end of the schedule);
the kilometres are on the vectored cohort the defect was measured on. **333 stored runs
recounted, 0 renamed.**

**Two consequences the pre-commit review found, both fixed here.** (1) The trombone's
no-crossing branch cut the reference one segment early: under the plane-only rule "no crossing"
implied no boundary was past the plane, so the in-segment fraction saturated to 1 on its own and
the cut landed on the last boundary — where the path length already counted to. The on-final rule
broke that (an overshoot IS past the plane and still never on the final), the fraction clamped to
0 instead, and `L_ref` and `S_ref` stopped ending at the same point: **340 m of invented detour
on a perfectly straight reference**, at every step, so it never burned down. The fraction is now
FORCED to 1 where there is no crossing, and a straight reference that overshoots wide of the cone
is a test case. (2) `truncatedAtThreshold` was shared with the fixed-time postprocessor's
closest-approach truncation, which has no plane in it at all — so a vectored state record could
end 8 km abeam and still claim to end at the threshold, and `predict`'s cut/whole count was wrong
for exactly that population. Where the crossing rule runs it now OWNS the flag: no crossing
clears it, and `horizonCapped` — which marks the records the fixed-time rule never reached —
still recovers whether that cut happened.

**One fixture changed with the rule, and the change is the point.** `_VECTORED_LEGS` in
`test_control_constraints` reached the threshold POINT at 45° to the course, which under the new
rule is not a landing; it gains a 5 km final (42.0 km of path against a 21.5 km beeline). The
same test file gains the downwind case the defect is about: a reference that starts 3 km past
the threshold and 8 km right of it, flies the reciprocal out to 12 km back, then a base and a
final — where the plane-only cut measured 8.5 km of path against the 30.6 km the network
intends, and reported **307 s** of surplus on a flight with none.

The `L_ref` phrasing in the entry below ("its first crossing of the threshold plane") is
superseded by this one.

### 2026-09-09 — ts_transformer: L3.f — the trombone's surplus is sized against the reference rollout's remaining path, not the beeline

**The defect, measured (L3.e).** The mechanism worked and the estimator did not. At the TRUE CTA
— where by construction there is nothing to absorb — `tromboneDelayS` came out at p50 **391 s**,
the endpoint `|xt|` p95 at **10 km**, and **46 %** of flights did not reach the threshold by
`T_cta`; meanwhile the delay was absorbed exactly 30 s per 30 s of offset and the envelope
violations fell 85 / 84 / 99 % (thrust / stall / load). The cause is the length the time is
compared against: `ΔL = V_e·T_r − D` with `D` the BEELINE, which is not what a vectored flight
intends to fly. Its downwind and base — 40 km of path under a 21 km beeline on the fixture —
read as time to burn, and the hook stretched a flight that was on schedule.

**The change (one axis, `trombone_surplus_reference ∈ {beeline, reference-rollout}`;
`predict --trombone-surplus`).** Under `reference-rollout` the length is the aircraft's own
beeline plus **the detour the network still intends on top of it**: `detour = L_ref − S_ref`,
where `L_ref` is the HOOK-FREE reference rollout's remaining horizontal path from this command
step to its first crossing of the threshold plane and `S_ref` is the straight line to that same
cut. `ΔL = V_e·T_r − D − detour`, positive part — exactly the beeline surplus less what the
model was already going to spend on vectoring — and the dog-leg realisation is untouched
(`cos θ = D/(D + ΔL)` about the same base). A reference that never crosses is cut at the end of
its own schedule and says so (`hook_trombone_ref_no_crossing`): that is a different claim.

**Why the detour and not `L_ref` itself** — the obvious reading, and it is wrong twice; both
were found by the pre-commit review, on the package's own fixtures. `L_ref` is indexed by the
SCHEDULE: it knows the step number and nothing about where the hooked aircraft has got to, so
the moment the excursion opens the estimate stops responding to it and the surplus never burns
down; past the reference's own crossing `L_ref` is zero and `ΔL` degenerates to the whole
remaining reach, its largest possible value. Measured on the 48-segment rollout fixture (60 s
late, a flight whose own plan IS the beeline, where the axis should be a no-op): **19 of 48
steps pinned at the 45° offset cap, ending 1.7 km wide** of the threshold, where `beeline`
rolls out with zero saturated steps and ends 159 m out. Second, a reference that decelerates
and stops SHORT of the threshold had its missing metres counted as surplus — on the on-time
fixture, **9.4 s of delay invented and 15 engaged steps** against `beeline`'s 0 and 0, i.e. the
hook stretching a flight that could not cover the path it already had, sign inverted. The
detour form has neither problem: `D` is the AIRCRAFT's own, and `detour` is non-negative and
non-increasing along the schedule by the triangle inequality (removing a leading chord takes at
least as much off the polyline as off the straight line). A rollout-level test over all 48
segments now pins both — it is the only place the estimate is read at `segment_index > 0`, and
it fails on the naive form at both offsets.

**The reference rollout is now a documented member of the composite.** `needs_reference` becomes
a per-INSTANCE flag on the trombone, so `barrier+speed-floor+trombone` gains the hook-free
rollout under this axis and nothing extra under the default. The engine's reference (the
mechanism the archived nominal law introduced — reused, not duplicated) changed from lock-step to
LOOK-AHEAD: `rollout_piecewise_constant_hooked_with_step` now integrates the unhooked schedule
once, before the first hooked segment, and hands the hook `RolloutStateView.reference` WHOLE
(`[B,N+1,7]`, one chart row per segment boundary, the anchor first). A remaining path is a future
quantity and no per-segment state can answer it. Cost: one extra integration of the same schedule
(~2× the SEGMENTED rollout; the dense re-integration at predict is untouched), paid once — the
trombone derives its `[B,N]` detour table at `segment_index == 0` and the per-step cost after
that is an index. The archived law's README now says the contract moved, rather than leaving
stale code that looks portable.

**Two approximations, stated rather than silent.** Both lengths are CHORD sums between segment
boundaries — the command is constant within a hold, so the segment is a circular arc and the
chord is short by `sinc(Δψ/2)`: 0.15 % at a 15° bank over the deployed ~5 s hold, 2.8 % at 45°
(and the two errors partly cancel in the difference). And the crossing is interpolated linearly
inside the one segment that contains it, because a segment is a kilometre of flying either way.

**The default is `beeline`, and it is bit-exact.** Measured with the L3.e equivalence harness
extended to both trombone stacks, trained and replayed, in both saturations (`off`, `barrier`,
`speed-floor`, `barrier+speed-floor`, `barrier+trombone`, `barrier+speed-floor+trombone`):
**1972 leaves compared, 0 differing** — trajectories, controls, records, diagnostics and run
names alike. That is also why the two reference-only counts (`tromboneRefPathM`,
`tromboneRefNoCrossing`) and the `tromboneSurplusReference` label are ABSENT rather than zero
under the default: a key added there would change the bytes of every L3.e record on disk. Run
names: **333 stored runs recounted, 0 renamed** (`run_naming._field_diffs` skips a META field a
stored config does not carry). The axis is refused away from its default under a hook that
contains no `trombone`, like the barrier's gains and the stall margin.

**New: `CommandHook.diagnostic_labels()`** — per-run STRINGS reported next to the counts and
never divided by `hook_steps`, merged by the composite under the same collision refusal. A
module with more than one way of computing the same quantity says which one it ran, so a record
carries the law that produced its numbers.

Tests: +9 (`tests/test_control_constraints.py`) — the vectored dog-leg fixture at offset 0 and
+60 s (beeline asks for 260 s + offset, the reference for the offset alone), the two-arm
rollout-level no-op test above, the crossing interpolation / never-crosses flag / zero-length
segment gradient, the three refusals, the composite's OR, the engine handing over the whole
schedule, and the record's keys under both settings. 899 pass, 1 skipped.
`docs/experiments/l3f_path_stretch_ref_arms.json` dry-runs 12/12; the arms are NOT run.
### 2026-09-09 — ts_transformer: package review — bugs and why it is still heavy after the audit

Docs only: `4dTrajectory/ts_transformer/docs/2026-09-09_package_review_bugs_and_architecture.md`.
Five parallel code reviews (data plane / training / control / inference+CLI / config), each
finding re-verified. Four number-changing bugs on live paths: `run_ts_pipeline` drops
`control_dynamics_model` from both rebuilt override dicts (lag runs labelled point-mass, never
reused, and colliding on `train_dir` with point-mass cells); the fixed-anchor common-grid truth
and cohort floor ignore `minimum_anchor_index` (the history ablation's selection metric is
measured from an anchor up to 120 s early); an observed threshold crossing gets 2.5× less
terminal-position weight than a fitted one; under soft saturation the barrier and the trombone
rewrite the same step in the 30–40° band (the composite's invariant is false). Four crash paths
(`--batch-size auto` on every custom control arm; `predict` leaving a record directory without
`summary.json`). ~20 contract holes (silently ignored loss weights, run-name collisions on
selection-changing fields, a test-release ledger bound to a directory). Diagnosis: one experiment
axis costs ten touch points; target architecture = a real package, one `OutputStrategy`
sub-package per prediction path, `TSConfig` split into owner sub-configs with sum types and a
flat-dict adapter, runners under the package. Census of 219 stored runs backs the freeze/delete
list. Nothing changed in code; decisions listed in the doc's §〇.

### 2026-09-08 — ts_transformer: L3.e — the trombone command hook (the delay gets a place to go), and `predict --truncate-at-threshold`

**The question.** L3.d's speed floor failed all three of its gates, and the reason is arithmetic,
not tuning: the duration head obeys the CTA exactly, so the rollout's TOTAL TIME is fixed; the
network's path is what it is; a fixed path flown in a fixed time has a fixed mean speed. A floor
that refuses the infeasibly slow commands therefore has nowhere to put the time it refuses to
waste — the demand came back as `thrust_over_max` (0 samples before the floor, ~10 % of samples
after) and stall samples rose 94–139 %. The geometry readout added the other half of the picture:
with the floor holding the speed up, the rollout reaches the threshold **early** and keeps flying
— endpoint |xt| p95 43–63 km, pooled ADE 840 → 3443 m at offset 0 — so every flyability, corridor
and CTA number taken over the whole record was scoring post-landing flying. On the approach proper
(truncated at the threshold) the same floor arms read fully-flyable **1.35 % → 48.9 %** at +60 s.
Two changes follow, and they are separate: one gives the delay somewhere to go, the other makes
the reports read the approach.

**The hook (`control/constraints/trombone.py`, predict-time; nothing retrained).** At each command
step it reads the schedule's own remaining time, the beeline distance `D` to the threshold in
runway axes, and the SAME `V_floor` the speed floor holds on that segment
(`speed_floor.floor_speed`, extracted so the two cannot disagree), and turns the delay the
remaining path cannot absorb into an extra path length by one identity:

    V_e = max(V_floor, V_h)
    Δt_unabsorbed = T_r − D / V_e            ΔL = V_e·T_r − D

`V_e·T_r` is how far the aircraft covers in the time it has; the surplus has to be spent on path,
and the only place a real arrival may buy path is BEFORE the final. It is spent as a dog-leg about
the beeline at

    cos θ = D / (V_e · T_r)

— the offset at which flying for the remaining time lands exactly on the threshold. **The speed is
the one being flown, floored at the speed floor, and that `max` is what makes the manoeuvre
terminate.** The floor is a LOWER bound — the module only ever raises thrust — so `V_h > V_floor`
is the ordinary case, and the first design, sized against `V_floor` alone, failed exactly there:
with the aircraft covering ground faster than the estimate assumed, ΔL stopped falling (measured at
commanded thrust 0.12: the surplus rose from 2981 m and plateaued), the half-way switch never fired
and the excursion pinned outbound until the threshold plane ended it, 2.9 km wide of the
centreline. With `V_e`, `dΔL/dt = −V_e + V_h·cos θ ≤ 0` for any speed the aircraft flies. That makes the whole law a function of the state: as the surplus is spent θ shrinks to
zero on its own, so the roll-out onto the beeline is guaranteed by the identity rather than by a
timer, and **a flight with nothing to absorb is bit-identical in both saturations**. The excursion
is held on ONE side (latched at engagement as the side the aircraft is already on, so it stays wide
of the extended centreline) until the surplus falls below half the value latched at engagement,
then mirrored — no accumulator, because ΔL burns down at exactly the rate the extra length is
bought. The bank that turns onto the offset is the barrier's own inversion, lag-compensated
(`τ_eff = τ_μ(1 − e^{−Δt/τ_μ})`, crediting the bank already rolled into), saturated soft or hard,
clamped to the envelope, with the load factor re-coordinated to keep `n cos μ`.

**Three design decisions the pre-registration left open, and what settled each.**

1. **The hand-over rule.** The hook may only OPEN an excursion where the predicted path is more
   than `ALIGNMENT_MAX_DEG` (30°) off the runway course. That is strictly inside "the on-final gate
   is closed" — the gate REQUIRES alignment — so it can never act inside the gate, and it also
   declines the one state that is outside the gate yet already lined up (wide of the membership
   cone but pointed down the final), which is exactly where a dog-leg is a turn the barrier undoes
   as soon as the cone is entered. Alignment gates the OPENING only: the outbound leg can swing the
   aircraft through the course, and a hook that fell silent there left it 40° off with no leg back
   (measured on the fixture: four engaged steps, then a parallel track 6.6 km wide of the
   threshold). Once the gate has opened for a flight the hook is disabled for the rest of that
   rollout, and it never acts at or past the threshold.
   **The price is measured and is not hidden: ~58 % of the fleet cannot be stretched at all.**
   KRDU, 2998 arrivals sampled, observed track at the L−1 anchor: 58.9 % are aligned within 30° of
   the course at the anchor, 58.4 % are aligned at EVERY row from it, 57.1 % are already inside the
   gate. Those are the straight-ins, and under the generous membership cone they are "on final"
   from tens of kilometres out. Their unabsorbed delay is published as
   `commandHookDiagnostics.tromboneDelayS` next to `tromboneEngagedSteps = 0` — the number to
   quote next to gate (5), rather than a reason to let the hook into the corridor.
2. **The turn cap is 15°, and the speed floor sets it.** The vocabulary's one guarantee is that a
   value's `+` order is its application order, so under `barrier+speed-floor+trombone` the floor
   prices the network's load factor and NOT the turn the trombone then adds. The turn costs the
   margin twice — the coordinated load factor raises the stall speed by √(1/cos μ), and the induced
   drag it adds (~tan²μ) is drag the floor's thrust was not sized for. Measured on the rollout
   fixture (48 segments, a base leg flown at the floor, 6 s holds, V_floor 61 m/s over a 55.5 m/s
   stall): stall slack −0.6 m/s at a 10° cap, −0.9 at 15°, −1.5 at 20°, −3.3 at 25°, and at 30° a
   first stall sample. The OFFSET cap (45°), not the bank cap, is what buys path — the arrival is
   indistinguishable down to 10° — so lowering the bank costs nothing and buys ~2× headroom.
   Reordering the composite would have let the floor price the turn exactly and is the better
   physics, but it would have made a value's spelling disagree with what it does, which is worse.
3. **`hook_saturation` softens the bound, not the mode.** The hand-over, the excursion's side and
   the half-way switch are hard predicates under both forms; `soft` selects the bank saturation,
   the gate blend and a ramp on the surplus that is exactly zero, with zero slope, at zero surplus
   — so inertness survives the soft form, which is the one the arms use.

**When it cannot absorb, it says so.** The 45° offset cap binds once the delay exceeds ~41 % of the
time the remaining path needs at the floor speed (`sec 45° = 1.41`), and
`hook_trombone_saturated_steps` counts those steps — the counterpart of the floor's "full thrust is
not enough" — and `hook_trombone_bank_capped_steps` does the same for the 15° turn cap, so the
argument that cap rests on is auditable on an arm. That is outside the arms' range (a 25 km run at ~70 m/s is 350 s, so the +90 s arm asks
for 26 % and lands near θ = 31°), but a short remaining path with a large delay is simply not
absorbable before the final, and the count is what says so rather than a silent shortfall.

**What the hook does NOT do, which matters for reading gate (1).** It is geometry. The rollout's
speed comes from the thrust commands, the schedule is open loop, and the floor's demand is a
function of speed and height — so within one rollout the trombone barely touches the thrust.
Measured on the fixture, the floor's bound and saturated step counts are IDENTICAL with and
without it (47/48 and 0/48 either way), and the synthetic fixtures do not reproduce L3.d's
fleet-level `thrust_over_max` at all (the floor's demand saturates once in 48 steps at worst).
What the stretch changes is WHERE the aircraft is when the schedule runs out, and — through the
truncation — which part of the trajectory a report scores. If `thrust_over_max` falls in the arms,
that is why; it is not the hook relieving the thrust demand.

**Vocabulary.** `barrier+trombone` and `barrier+speed-floor+trombone` are members of
`CONTROL_HOOK_MEMBERS`; `speed-floor+barrier`, `barrier+trombone+speed-floor`, `trombone+barrier`
and a solo `trombone` are not, and are refused with the vocabulary. There is deliberately no solo
value: the hook hands the command back at the final approach course and has nothing to hand it to
without the barrier in the stack. Barrier and trombone both write bank and load factor and still
compose, because their gates are COMPLEMENTARY — the barrier acts only inside the on-final gate,
the trombone only outside it, so no step is ever rewritten by both. `control_speed_floor_margin` is
now read by two modules (`CONTROL_SPEED_FLOOR_MARGIN_READERS`) and is accepted under either.

**`RolloutStateView` gains `remaining_s`** — how much of the schedule is left at a segment's start,
this hold included; the rollout knows every duration before the first segment is integrated, so it
is the reversed cumsum rather than something a hook accumulates and hopes it was called in order.
Under a CTA-conditioned decoder it IS `T_cta − t`.

**The truncation (`forecast.cut_at_threshold_crossing`, `predict --truncate-at-threshold`).** Cuts
every record at the closest horizontal approach to the target among the rows at or past the
threshold plane (`d ≤ 0`), scoped to the FIRST such run — "first" so a rollout that wanders off and
later passes near the threshold again is cut on its real arrival, "closest approach" so a laterally
displaced crossing is not cut a step early. A forecast whose rows never reach `d ≤ 0` never landed:
it is returned WHOLE and still says `truncatedAtThreshold: false`, because cutting it would invent
an arrival. `final_time_s` moves to the cut, which is the point — an early arrival stops being
invisible in the CTA readout and becomes the `final_time_error_s` it always was. The control
record's two clocks stay aligned (`export` refuses them otherwise): segments are cut to the one
holding the new end and its duration shortened to land exactly on it. Applies to every output kind
and to a run's latent/posterior/label diagnostic arms as well, so one output directory is not half
cut; refused together with `--no-truncate`. Off by default.

**Measured.** Equivalence vs `617539a`, 8 synthetic 2-epoch cases (no hook; trained through
barrier / speed-floor / barrier+speed-floor; and the barrier, floor and stack replayed soft AND
hard on a hook-free checkpoint): **1094 non-timing leaves, 0 differ** — every state-dict digest,
every loss and metric, every forecast and record digest, every hook diagnostic. Run names recounted
on disk over all 354 stored configs under `4dTrajectory/outputs/**/summary.json`: **0 renamed, 0
refusal messages changed**. On the rollout fixture the stack does what it is for: under
`barrier+speed-floor` the rollout reaches the threshold with a quarter of its schedule left and
flies on; adding the trombone moves the closest approach to the last segment (within 1.5 % of the
direct distance) and the flown path grows, against the same stack flown to the UNDELAYED arrival
time, by 3469 m where `V̄ × 60 s` is 3512 m — 1.2 %. Tests 887 passed, 1 skipped (was 869 + 1).
`l3e_path_stretch_arms.json` now carries `--truncate-at-threshold` on all four arms and dry-runs
clean (12/12 steps, KRDU val).

### 2026-09-08 — ts_transformer: L3.d — the speed floor is a command hook (`speed-floor`, `barrier+speed-floor`), and hook diagnostics become per-flight

**The question.** L3.c settled that the soft barrier hook holds the corridor under a late CTA
(+90 s: `xt@thr` p50 3288 → 5 m, |xt| p95 7694 → 2332 m) and is free at offset 0 (ADE 840 vs 842),
but that the trajectories are unflyable anyway: the fully-flyable share is 0 % from +60 s,
**99.9 % of the hard violations are the STALL term, and 77–94 % of the stall samples sit at
≥ 20 km remaining**. The model commands infeasibly low speeds on the OUTER segment and a late CTA
pushes more of the delay into exactly that segment. The barrier is lateral-only, so nothing in the
adopted delivery form touches it. User decision 3(a) of 2026-09-08: a speed floor inside the
rollout, acting THROUGH the controls, never by clipping V after the fact.

**The change (code on `dev-speed-floor`, predict-time; nothing retrained).**
`control/constraints/speed_floor.py` is the second constraint module. At each command step it
computes the stall-margin floor

    V_floor = control_speed_floor_margin × sqrt(2 n_commanded m g / (rho S Cl_max))

— `flyability`'s only hard term (`Cl_required > Cl_max`) read as a speed, at the flight's own
`(S, Cl_max)` row of the batch `aero_params` (the same numbers the RHS integrates and `flyability`
grades against) and at the ISA density of the height the command's effect is measured at, not at
sea level. It then inverts `V̇ = (T − D)/m − g sin γ` over the hold for the THRUST command that
leaves the speed on the floor when the hold ends, crediting the thrust already spooled for
`τ_eff = τ_T (1 − e^{−Δt/τ_T})` — the same lag compensation barrier v2 gives a bank already rolled
into, and the v2 lesson applied to a second channel. The command's thrust is raised to that demand
(soft: a scaled softplus; hard: a clamp) and clamped to the envelope. **Bank and load factor come
back bit-identical.**

Three design points, each of which could have been got wrong:
- **Ungated.** The barrier acts only where its on-final gate opens, because a corridor is a
  statement about the final approach. A stall is a statement about the airframe, and the
  measurement says it fires mostly where that gate is shut — a gate here would have left the whole
  problem untouched.
- **Through the controls.** Clipping V after the rollout was the three-line alternative and would
  have stopped the record being a trajectory of the dynamics — the same reason the corridor is a
  barrier on the bank rather than a projection of the path.
- **The COMMANDED load factor.** Under `barrier+speed-floor` the barrier runs first and
  re-coordinates the load, so the floor prices the manoeuvre that will actually be flown.

**Composition.** The rollout takes ONE hook, and that stays right — "the command flown" must be a
single answer — so composition lives in `control/constraints/composite.py`: each module in turn is
handed the same segment-start state and the previous module's command. The order is part of the
vocabulary value (`config.CONTROL_HOOK_MEMBERS`), not a free choice, and the `+` is a LOOKUP, never
a split: `barrier+speed-floor` is a member, `speed-floor+barrier` and `barrier+nominal-residual`
are not and are refused by the ordinary vocabulary check. The two modules write disjoint channels
(bank + load against thrust), which is what makes this combination well defined and every other one
absent. A composite refuses two modules that report a diagnostic under the same name.

**`control_speed_floor_margin` defaults to 1.10, whose COEFFICIENT is not a new number** — the
optimizer's NLP velocity floor (`optimization/scenario_optimization._STALL_MARGIN`) and this
package's control-anchor eligibility gate (`anchor_eligibility.CONTROL_ANCHOR_STALL_MARGIN`) are both
1.10. **The speed it multiplies is NOT the same one, and a results table must not quote the three as
equal**: those two take the 1-g stall speed at SEA-LEVEL density (the optimizer's is also capped at
V_ref), while the hook takes `V_stall(n_commanded)` at the LOCAL ISA density and applies no cap —
because it exists to defend `flyability`'s criterion, which is evaluated at each sample's own
altitude and inverted load factor. At 8000 ft ρ/ρ₀ = 0.79, so the hook's floor is ~12.5 % higher, an
effective margin near 1.24 (×√n in a turn). The direction is the safe one: the hook is strictly the
TIGHTEST of the three, so what it lets through the optimizer's floor would also have admitted. Not
one symbol on purpose: the other two are frozen policy constants (one is spelled into the stored
policy string `airborne-1.10-stall-margin-v1`) and this one is a per-run field a campaign may raise,
and aliasing them would let a run's knob rewrite a data policy's identity. It is refused away from
its default under a hook with no speed floor, and refused non-finite or below 1.0, and it is
deliberately absent from `REQUIRED_SERIALIZED_CONTROL_FIELDS`, like the barrier's gains — a config
without the hook could not read it, so the default IS what every stored artifact ran under.

**A second module made an old check positional.** Until now `barrier` was the only buildable hook, so
"a hook is on" and "the barrier is on" were the same condition and the barrier's gains always bound.
Under `--command-hook speed-floor` they no longer do, so `control_barrier_alpha` /
`control_barrier_heading_gain` are now refused away from their default on a hook with no barrier in
it — the same "a value that cannot change an answer" rule the margin gets — while the positivity
check stays wherever a barrier IS built. The archived `nominal-residual` builds nothing and is
outside both rules by name, so its six stored 2026-09-06 configs load exactly as they did.
Two import-time assertions pin the tables against each other: every selectable hook names its
modules (`CONTROL_HOOK_MEMBERS` vs `CONTROL_HOOKS_AVAILABLE`) and every module named has a class
(`_HOOKS`) — without them a new vocabulary value would pass construction and die inside the rollout.

**Hook diagnostics became per-FLIGHT.** `CommandHook` gains `per_flight_diagnostics()` returning
`[B]` rows; `diagnostics()` is those summed and still feeds the epoch record. A prediction record
now carries `source.commandHookDiagnostics` — the flight's own `steps` plus every other count as a
share of it — because `commandHook` alone cannot separate a flight the hook never touched from one
it rewrote at every step. The key is ABSENT without a hook, never zero.

**Also**: `soft_max`/`soft_min` moved out of `barrier_filter` into
`control/constraints/saturation.py` — one definition of what `hook_saturation=soft` MEANS, now that
two modules spell it; `RunwayAxesView` gains `vertical_speed` (the height rate the floor carries
over a hold), so the chart layout still has one reader.

**The soft form had to be made inert where it demands nothing** (found in review before any arm
ran). `soft_max(x, bound, s)` overshoots its bound by `s·ln 2`, so parking a non-binding demand at
`MIN_THRUST_FRACTION` added **0.0139 of installed thrust — 2.8 kN on a 66 t fixture, 0.042 m/s², about
12 m/s over a 300 s remainder** — to every command sitting at the envelope floor, and `[-0.2, -0.1]`
is the COMMON band on an approach (`control/envelope.py`: a real approach needs net-negative thrust),
not a corner. `soft` being the adopted delivery form, the L3.d arm would have measured that bias
rather than the floor, and the diagnostics would have hidden it (the step counted as "not bound"
while the thrust moved). The demand is now parked twenty softnesses below the box, where `softplus`
is linear and the soft and hard forms agree bit-for-bit; it is still a FLOOR, because an unclamped
`-1e6` cancels in `bound + s·softplus(...)` and reintroduces a larger error than it removes
(measured: 1.5e-8 in float32, 0 in float64, against 1.25e-2 unclamped). `hook_floor_bound_steps` now
reads the DEMAND against the command rather than a change threshold, so soft and hard count the same
steps, and `hook_floor_saturated_steps` no longer requires the floor to have changed anything — "full
thrust is not enough" is exactly the case where the network is already there.

**Equivalence, measured** (`scratchpad/l3d/l3d_equiv_harness.py`, eight 2-epoch synthetic cases on
`9b1f130` vs this branch; 2041 non-timing leaves compared): every state-dict digest, every forecast
digest — **including the barrier replayed soft and hard on a hook-free checkpoint** — every loss and
every metric is identical. **Two leaves differ, both 1 ULP**, and both are
`command_hook.bank_change_rad` in the epoch record of the arm TRAINED through the barrier: the
barrier's counters now accumulate per row and are summed at read time instead of being summed per
call, which reorders one float addition. Nothing reads that number but a readout. The only other
difference is the new config key itself.
**Run names recounted on disk**: 177 stored configs under `4dTrajectory/outputs` (from `history.json`
and `config.json`) rebuild and name identically on both trees — **0 renamed** — and the 149 that this
build refuses are refused with byte-identical messages on both, i.e. the change adds no refusal.

**Pre-registered**: `4dTrajectory/ts_transformer/docs/experiments/l3d_speed_floor_arms.json` —
predict-only on the `l3_cta_20260907/L3_cta` checkpoint at CTA offsets 0 / +30 / +60 / +90 s with
`--command-hook barrier+speed-floor --hook-saturation soft`, paired flight-by-flight with the L3.c
hook arms. Gates: (1) stall SAMPLES on the approach fall ≥ 90 % at every offset; (2) fully-flyable
share at +60 s ≥ 88.4 % (the observed tracks' 98.4 % floor minus 10 points) — conditional, because
if the floor makes the rollout arrive EARLY instead, the unabsorbed delay is the deliverable X and
is reported rather than gated; (3) endpoint |xt| p95 at +60 s ≤ 1.5 × the offset-0 arm, as L3.c.
Not run here.
### 2026-09-08 — ts_transformer: A2b — `random_train_anchor_l1_share`, a reserved share of the random-anchor draws for L−1

**The question.** A0.b settled the random-anchor line's first half and left the second open.
`A0b_lr_objective_path_uniform` ran all 180 epochs without early stopping, took the anchor-grid
selection metric to 710 (L−1 1782 / 12 km 561 / 8 km 283 / 6 km 212 m), and **reached s_freeze at
8 km** (vectored |Δt| p80 19.8 s) — gate 3 met for the first time in this project, because the
duration head stopped flooring at ~125 s. What it still fails is the pre-registered L−1 veto: on
1404 paired KRDU val flights its pooled ADE is **1782 m** against the fixed-anchor native32's
**1322**, far beyond the 125 m control-path seed line. The reading is mechanical — under a law
spread uniformly over the flight's own remaining-path span, L−1 is ONE POINT among many draws, so
the arm never specialises for the anchor the fixed arms train at exclusively. A2b asks whether
giving L−1 a fixed SHARE of the draws recovers it without losing the curve.

**The change (design §2.4d; branch `dev-a2b`).** `random_train_anchor_l1_share: float = 0.0`. Under
`random_train_anchor_sampling=remaining-path-uniform`, with that probability a flight's draw for the
epoch IS `default_anchor(config)` = `seq_len − 1` — exactly the anchor the fixed-anchor arms train
at — and otherwise the unchanged span draw.

**A mixture, not a reweighting.** The coin is a SECOND 8 bytes of the same per-flight per-epoch
sha256 the law already reads for its unit draw, and the digest's salt does not move with the share.
`sampling_version` stays the class's law AND the salt; a new `TrajectoryWindows.reported_sampling_version`
property is what the epoch record and the checkpoint's `training_anchor_contract` NAME, and the
path-uniform policy overrides it to `per-flight-hash-v4-remaining-path-uniform-l1-share` when the
share is non-zero. So the (1 − share) of draws the coin passes over are **literally the anchors the
share-0 arm drew**, and the A2b arms differ from `A0b_lr_objective_path_uniform` in the replaced
draws and in nothing else — the contrast the campaign is for, rather than a second silent reshuffle
of the other 70 %.

**Refused, never ignored.** A non-zero share under `uniform` sampling (that law draws over the
samples, where L−1 is already one of them, and mixing a reserved share in is a second unmeasured
axis); a non-zero share with `random_train_anchor=False` (the fixed policy already anchors every
flight at L−1); and a value outside [0, 1]. The named recipes deliberately do NOT pin the new field
— with `random_train_anchor` pinned False a non-zero share is already refused, and a bound that
cannot bind reads as though it had (the same reasoning as `duration_quantile_loss_weight`).

**The flights that have no L−1 to reserve: counted, not refused.** The reserved draw is the
flight's FIRST admissible anchor (offset 0), which is L−1 unless output eligibility removed it or a
`minimum_anchor_index` floor moved it — and under a floor it is exactly what
`FixedAnchorTrajectoryWindows` anchors at anyway. The first draft refused the eligibility case as a
loud precondition; **measuring it killed that**: on the whole KRDU arrivals roster (14,435 records,
9,720 in the `openap-direct` cohort under the 20 s future contract) **10 flights store no anchor 59
at all** — 0.10 %, nine of them runway 23R, first admissible anchor 90–163 — because
`airborne-1.10-stall-margin-v1` puts their observed state there outside the airborne model domain.
A refusal would therefore have aborted both A2b arms at dataset construction, and the only remedy
it named (`--eligibility-roster`) would have changed the split and destroyed the pairing the arms
exist for. Those flights are now reserved at the earliest anchor they DO have — the nearest thing
they possess to the start of the approach, and the only one the control rollout can integrate — and
the count is stated in output, never assumed: `l1_share_flights_without_l1` in every epoch record
and one line from `train()`. Consequence for readers: `fixed_anchor_fraction` bounds
`l1_share_drawn` from above only while that count is zero.

**Bookkeeping.** Under a non-zero share only, each epoch's `train_anchor_sampling` block gains
`l1_share`, `l1_share_drawn` and `l1_share_flights_without_l1` beside the strata histogram. `l1_share_drawn` is the COIN's realised
share; the long-standing `fixed_anchor_fraction` counts every drawn anchor that IS L−1 — the coin's
plus the span draws that landed at the far end anyway — so it bounds it from above rather than
restating it. Reading the coin needs the epoch's seed, so the `_sampling_extras` hook now takes it
beside the indices (a draw law can have a component its chosen anchors do not reveal).

**`default_anchor` moved to `config.py`.** `dataset` needs the same index and `forecast` imports
`dataset`, so the one-liner could not stay where it was; `forecast` re-exports it and every call
site and monkeypatch is unchanged. It also replaces the inline `seq_len - 1` that
`anchor_statistics` carried.

**Default bit-equivalence (measured against `9b1f130`).** Three 2-epoch synthetic trains — the
fixed-anchor control path and both random-anchor sampling laws — compared leaf by leaf:
**4312 shared non-wall-clock leaves (4069 numeric), 6 differing, and all 6 are
`validation_profile_by_airport.KRDU.wall_s`**. The per-epoch `sample_sha256` digests are identical
under both random laws, i.e. the draws themselves were not touched. The one added key is
`config.random_train_anchor_l1_share`. **Name recount**: 1128 stored configs under
`4dTrajectory/outputs` (348 summary.json, 184 each of history / checkpoint_metadata /
fit_evaluation, 117 experiment_manifest, 111 campaign config.json) named and slugged at both trees
— **0 renamed, 0 unloadable**; the A0.b arms' slugs still end `…_8-more_7cfd2b7f` /
`…_9-more_48f49b41`. Suite 835 → **851** green. (Reviewed by an opus subagent; its HIGH finding is
the L−1 precondition above, which the cohort measurement then settled, and three test-strength
findings are folded in — the mixture test now pins WHICH flights the coin fired on, and the range
refusal pins that it wins over the two cross-field ones.)

**Arms.** `4dTrajectory/ts_transformer/docs/experiments/a2b_l1_share_arms.json`: base = the stored
`A0b_lr_objective_path_uniform` config (verified field-by-field against
`outputs/KRDU/experiments/a0_random_20260907/A0b_lr_objective_path_uniform/config.json`), two arms
`A2b_l1_share_0p3` / `A2b_l1_share_0p5` at 180 epochs, `patience` 180,
`lr_plateau_metric=objective`. **Pre-registered gate, both must hold**: the L−1 pooled ADE not worse
than native32's 1322 m beyond the 125 m control-path seed line, AND the anytime curve's gate 3
(s_freeze) still reached at 8 km with |Δt| p80 < 30 s. **Veto**: the 12 km bin's vectored ADE p50
worsens by more than 100 m against `A0b_lr_objective_path_uniform`'s 561 m. Dry run clean; slugs
`…_10-more_fbde423f` / `…_10-more_e5b37c1c`. **Neither arm has been trained.**

### 2026-09-08 — ts_transformer: B1.b — the two-head duration (`duration_head=two-head`), the point head drives the rollout and the quantile head publishes the ETA

**The question.** B1 delivered two gains and `B1_point_matched` showed they come from different
mechanisms and do not overlap: the PATH gain is the duration term's WEIGHT (a point head at
`final_time_loss_weight` 26 reaches ADE 1248 / chamfer 177 against native32's 1322 / 224), and the
ARRIVAL-TIME gain is the quantile HEAD (pooled duration MAE 23.9 vs 25.9 s, straight-in 10.3 vs
13.6). Under `duration_head=quantile` the rollout's duration IS q50, so the quantile head's path
cost is forced onto the trajectory to buy the arrival time. Taking both needed a head that keeps
them apart.

**The change (design §三 3.1b, pre-registered 2026-09-08; code on `dev-b1b`).**
`duration_head` gains a third value, `two-head`. The POINT head stays `final_time_head` — the same
class, the same state-dict keys and the same rollout duration `point` has — and the quantile head is
a second module `duration_quantile_head` beside it, emitting nothing but the published distribution.
**The second head is built LAST**, after `final_time_head` and `control_head`, for the reason
`control/latent.py` builds `aux_duration` last: `_initialize_duration_head` zeroes only the last
layer, so the point head's HIDDEN layer is a live random draw, and an extra `nn.Linear` constructed
ahead of it shifts that draw. Built last, a `two-head` model and a `point` model at the same seed
agree on all 34 shared parameters (measured) — which is what gate 1 needs, being a single-seed ADE
comparison against `B1_point_matched` inside a 30 m band. The TRAINING dropout stream still cannot
be matched (the second head draws inside `duration()`): the two arms share an initialization, not a
trajectory.
`config.DURATION_HEADS_WITH_QUANTILES` / `DURATION_HEADS_WITH_POINT` are the two predicates every
consumer now asks (records, `forecast.duration_quantile_predictions`, `run_ts_eta_calibration.py`,
`predict --cta-from-quantiles`), so "does this checkpoint publish an interval" cannot drift from the
head table. `ControlFeatureModel.quantile_head()` is the one rule for WHICH module the quantiles
come from — a method, not an attribute, because binding one module under two names would publish
every one of its tensors twice in `state_dict`.

**Two terms, two weights.** `final_time` stays the point head's squared residual at
`final_time_loss_weight` under `point` and `two-head`; the pinball sum rides in a component of its
OWN, `duration_quantile`, at the new `duration_quantile_loss_weight` (default 1.0), and
`loss_component_names` gains that entry under `two-head` and nowhere else. Under `quantile` the
pinball keeps the name `final_time` (B1's contract — every stored history row keys on it) but is now
multiplied by the new field: both defaults are 1.0 and a non-default `final_time_loss_weight` is
refused there, so the number is the one it always was.

**Four refusals, each naming its reason.** `two-head` off the control output; `two-head` with
`latent_dim > 0` (the same reason as `quantile` — training decodes a posterior sample, so the five
would be quantiles of p(T | z ~ q(z | this flight's own future)), an interval conditioned on the
answer); a non-default `final_time_loss_weight` under `quantile` (no point term to weigh — every
stored `quantile` run carries the default, so nothing on disk is refused by it); a non-default
`duration_quantile_loss_weight` under `point` (no such head). The named recipes still pin the head
at `point` and deliberately do NOT pin the new weight: with the head pinned, a non-default value is
already refused, and a bound that cannot bind reads as though it had.

**The readout separates the two heads.** `run_ts_eta_error_readout.py`'s `final_time_error_s` is the
error of the duration the ROLLOUT flew — the POINT head under `two-head` — so an arm whose rows carry
`duration_quantiles_s` now also gets a `duration_q50_error_s` block, derived as `q50 − truth` from
the row's own published fields (no new stored number to disagree with them). **Gate 1 of §三 3.1b
reads its MAE**; every block gained `mae` beside the quantiles, and `RESULT_SCHEMA` is
`ts-eta-error-readout-b0-v3`. Under `quantile` the two blocks are the same measurement twice.
`control/training/diagnostics.py` counts the second head's gradients in the `final_time_head` group,
not the backbone's.

**Naming.** `T=2h` (a second `_VALUE_ABBREV` entry beside `T=q5`); the new weight enters
`CONTROL_LOSS_FIELDS` and shows only when it deviates (`pinball=3`). `B1b_two_head` dry-runs as
`control · iTransformer · first-order-lag @scaled-transport-chart-velocity · simple-v3+(final-time=26)
· T=2h, …` — the arm sets the pinball weight at its default, so it does not appear.

**Verified.** Bit-exact at every pre-existing value: 2-epoch synthetic trains for control / latent
k=1 / latent k=4 / state / closure / cta=given / **quantile** / **cta=given+quantile** against
`39e86ce`, comparing every history float, the state-dict digest and a forecast digest — **2021
non-wall-clock leaves, 0 differ**; the only added leaf is `/config/duration_quantile_loss_weight`.
Names recounted over every stored run config on disk: **175 configs, 0 renamed, 0 errors**. Suite
**830 passed, 1 skipped** (809 + 21 in `tests/test_two_head_duration.py`). The arm file
`docs/experiments/b1b_two_head_arms.json` needed no change — its field names were already these.

### 2026-09-08 — ts_transformer: hard procedure constraints in training — literature survey (83 papers) and the H0–H6 integration plan

**The question.** How to put the final-approach corridor + glidepath window into the TRAINING of
both output paths as a hard constraint, given that a hinge penalty was vetoed on both paths, the
tanh-bounded output was adopted on `state`, and the CBF barrier filter is a net gain at predict
time but made the `control` network lazy when trained through (six arms).

**What was done (docs only, no code, no training).** Three literature clusters collected and
annotated into `docs/literature/procedure_hard_constraints/` (`README.md` index, `notes_{A,B,C}.md`
with every core formula transcribed from the PDF, `download.sh`; 80 PDFs, three paywalled papers
quoted from abstracts and marked unverified). The survey + plan is
`4dTrajectory/ts_transformer/docs/2026-09-08_hard_constraints_survey_and_integration_plan.md`.

**What the literature settles.** (i) HardNet-Aff's parallel projection collapses to the runway-axis
clamp — the closed-form justification of `corridor-bounded`; its Prop. 7 (an in-corridor row is left
untouched, which the tanh does not do) is a cheap arm. (ii) The lazy network is a known, measured
effect of training through a filter with `CC` bookkeeping (OptLayer): Pizarro Bejarano 2025 removes
it with a correction penalty `α‖u_uncert − u_cert‖²` (uncertified return 11 → 211 as α 0.1 → 10),
Krasowski 2023's projection + adaption penalty gives the lowest intervention rate, Oh & Fisac 2026
prove no performance penalty for a least-restrictive filter used at train AND test (so an accuracy
loss reads as over-restriction or infeasible gate entry), and Geiger & Straehle 2022 prove test-time
-only filtering has a quadratic-in-horizon imitation error against linear for train-and-test — the
adopted predict-time-only form is the quadratic case. A saturating filter is many-to-one; the gauge
map (Tabas & Zhang) is a bijection onto the state-dependent safe set, which is the structural cure.
(iii) The C_dual divergence was correct optimisation of an unbounded dual (best response to a
violated constraint is `+∞`, Gallego-Posada 2022 / Chamon 2023): anneal the level, learn it (Hounie
2023 resilient constrained learning, whose `u*` is the measured reachable violation rate), or use
νPI / PID instead of plain ascent. (iv) The gate is a discrete mode to be PREDICTED (TNT / DenseTNT /
annealed WTA; the map-adaptive goal-based predictor supervises modes from exactly our cross-track
test). (v) Verified gap: none of twelve aviation TP papers enforces a corridor or glidepath window
or predicts the establishment point.

**Plan (§4, nothing built).** H0 gate readout (no training) → H1 `state`: learned monotone
commitment gate (hazard form, BCE on the truth gate, calibrated false-open threshold) × geometric
membership, clamp-vs-tanh and `xt(d)` side arms → H2 `control`: H2-0 predict-time `(α, k)`
permissiveness sweep + feasible-entry count; H2-a correction penalty at three doses with a
committed, feasible-entry gate; H2-b OptLayer-CPC form (raw rollout scored beside the filtered
one); H2-c bijective bank-interval map instead of saturation; every arm predicted with AND without
the hook → H3 vertical load-factor barrier + speed hold → H4 resilient / νPI Lagrangian only if
L1.c passes → H5 solver stage, H6 constrained generation. Runs in the `../thesis-hc` worktree
(`dev-hard-constraints`); the queued L1.c arm file is untouched.

### 2026-09-08 — B2 calibration: the half cut is probeable (`--half-seed` + `--readout-only`), and the deployed-vs-mirror coverage gap is mostly the cut

**The question.** On the KRDU B1/B3 quantile arms the DEPLOYED cross-half coverage (fit on half A,
score on half B) sat 5–10 points BELOW its own mirror (fit on B, score on A) — B1_quantile α=0.2
pooled 0.748 vs 0.839 at n≈702 per half, where binomial SE is ≈1.5 points. Either the one fixed
half rule (`sha256('conformal:{split_seed}:{flight_key}')`, `split_seed` = 1337) splits the cohort
systematically, or the gap is real. It could not be asked: there was no seed flag, and every runner
invocation rewrote the deployed `conformal` sidecar.

**The change.** `run_ts_eta_calibration.py` gains `--half-seed INT` (default: the checkpoint's
`split_seed`, i.e. today's cut, bit-identical numbers) and `--readout-only` (writes the readout to
`--out`, refuses to touch `checkpoint_metadata.json`). `--half-seed` is REFUSED without
`--readout-only`, before any checkpoint is read, and the refusal names both flags: a deployed δ
comes from the documented rule alone. The seed is threaded into the single
`calibration.calibration_halves` (no second hash); the table carries `half_seed` and
`deployed_half_rule`, a probe table's `half_rule` says `PROBE, NOT THE DEPLOYED RULE`, `render`
opens with a `PROBE HALF RULE` banner, and `calibration.write_conformal_table` refuses a probe
table with no escape hatch (unlike the smoke table's `--allow-smoke-table`). `CONFORMAL_SCHEMA`
stays `ts-conformal-cqr-v2` deliberately: no stored δ changes meaning and no reader of a stored
table reads the two new fields, so a bump would have refused every deployed sidecar over a
provenance field.

**The measurement** (KRDU val, 1404 flights, halves 702/702, CPU, ~11 s per run; five seeds
1337/2024/7/99/31337 × two arms, all `--readout-only`; both
`checkpoint_metadata.json` sha256 verified unchanged before and after). The gap **flips sign**:
B1_quantile α=0.2 pooled `coverage − stability` = −0.091 (seed 1337, the deployed cut), −0.040,
+0.006, −0.023, +0.036 — mean −0.023, and the deployed-block coverage moves 0.745 / 0.791 / 0.805 /
0.793 / 0.818 (mean **0.790**, nominal 0.80). At α=0.5 the deployed block reads 0.450 / 0.480 /
0.533 / 0.497 / 0.469 (mean **0.486**, nominal 0.50). B3_quantile_cta is the same picture at a
smaller amplitude: α=0.2 deployed 0.775 / 0.795 / 0.811 / 0.802 / 0.811 (mean **0.799**), α=0.5
0.477 / 0.480 / 0.517 / 0.484 / 0.486 (mean **0.489**).

**Reading.** The 5–10 point gap is an artefact of the ONE cut that was ever tried, not a cohort
split: averaged over five cuts both coverages bracket nominal (B1 α=0.2: 0.788 deployed / 0.810
mirror). The two are strongly ANTI-correlated by construction — a cut that hands half A the easier
flights fits a δ too small, which under-covers B and over-covers A when mirrored — so the GAP has a
much wider spread (sd ≈ 0.05 across seeds) than either coverage alone (binomial SE 0.015), and the
deployed cut sits ≈1.4 sd low. **Consequence for gate 3.4-2**: B1_quantile's α=0.2 verdict is
cut-dependent — 0.745 at the deployed seed is outside the [0.76, 0.84] band, the other four seeds
(0.791–0.818) are inside; B3 passes at all five. Quote the deployed number with its cut noise, or
read the gate on the five-cut mean. Neither this nor the mirror repairs the deeper issue already
recorded in `calibration.py`: the calibration split is also the selection split, and the
guarantee-bearing read stays the pre-registered single test-split measurement at `freeze-test`.
Probe artifacts (not committed):
`/tmp/.../scratchpad/b2a/{B1_quantile,B3_quantile_cta}_seed_{1337,2024,7,99,31337}/`.

### 2026-09-08 — ts_transformer straight-in residual decomposition: along-track deceleration-timing scatter, not lateral, not the total duration

`run_ts_straight_in_residual_readout.py` (new CPU runner) projects each flight's prediction error on
the observed heading / right normal / vertical over the common window and reads the speed schedule
by remaining distance. KRDU val straight-in (904): along-track RMS p50 333 m vs cross 112 m and
vertical 78 m (along = 89 % of horizontal error²), concentrated in the last 5 km (p10..p90
−819..+591 m); with the TRUE arrival time given (L3_cta) the bias goes (−191 → +8 m) but the RMS
does not (333 → 316 m). Speed residual per band ≤ 1.6 m/s, deceleration point median offset 0.0 km
with p10/p90 ±1.4 km, explaining 21–22 % of the last-band along variance (slope −225 m per km; the
tower wind explained 10 %). Reading: the residual is the deceleration schedule — a speed-instruction
intent — not dynamics. Results doc `4dTrajectory/ts_transformer/docs/2026-09-08_straight_in_residual_readout.zh.md`;
a deceleration-point oracle arm (L3.b) is drafted there, decision pending.
### 2026-09-08 — ts data identity is the eligible SET, not the eligibility roster's bytes

**The incident (2026-09-07, 08:15 and 08:55 local).** `run_all_evaluations.py --kind observed`
regenerated the five canonical observed reports (v6 → v9) and the five
`lateral_pass_eligibility.json` rosters were refreshed against them. The eligible sets came out
byte-for-byte identical (KRDU 14,378 keys, same order; v6 backup roster
`observed_v6_backup_2026-09-07/KRDU/…` sha `3e09d3e9…`, refreshed `0a90a923…`), but the roster
FILES moved, because a roster carries the upstream provenance it was joined against
(`sources.evaluation_report_sha256`). `data_provenance` compared exactly that: the v3 eligibility
entry held `roster_sha256` + `evaluation_report_sha256`. Consequence: `require_matching_data_
provenance` refused EVERY checkpoint trained before 08:55 — `predict`, `evaluate-fit`,
`run_ts_control_basis_oracle`, `run_ts_anytime_curve`, `run_ts_runway_hypotheses`,
`evaluation_protocol`, `approach_clustering` — and `publish_ts_experiment_trajectories.py`
blocked every publication through its OWN copy of the byte comparison. The arm trained at
08:55 (`l1b_full_20260907/L1b_hr16_tv1_full`, roster `e2b0fa52…`) would have refused at its own
predict step: that roster generation no longer exists on disk.

**The fix** (branch `dev-provenance`). The compared identity is the SET:
`ARRIVAL_DATA_PROVENANCE_SCHEMA` = `ts-arrival-data-v4-eligible-set`, whose eligibility entry is
`{schema_version, policy, eligible_set_sha256}` with
`eligible_set_sha256 = sha256("\n".join(sorted(keys)))` — one function,
`data_provenance.eligible_set_digest`, which `splits.data_selection_audit` now hashes its split
rosters with and which `run_ts_pipeline` and the publisher import instead of re-deriving. The
roster's `counts` left the compared entry too: three of the five are reject tallies read off the
observed evaluation (`excluded_lateral_indeterminate`, `evaluation_only`), so a re-graded flight
that never was eligible would move them and refuse the checkpoint — the same defect one level
down, reproduced on the real KRDU roster before it was removed. The byte facts and the counts
stay auditable and out of every comparison: `eligibility_sources(rosters)` records the roster
path, its digest, its counts and the evaluation report under
`data_selection.pre_split_eligibility[*].roster_sources`. `checkpoint_metadata.json` /
`history.json` name the map `eligible_sets`.

**Legacy checkpoints stay usable exactly, with no registry of old bytes and no flag.** A stored
`ts-arrival-data-v3-eligibility-bound` fingerprint carries the eligible set itself —
`source_records` IS that set, since the roster's keys are validated to exist in the manifest — so
it is compared to today's per airport, and a difference is refused by name with both set digests
instead of the generic "the manifest changed". The verification deliberately reads nothing else
from the payload: a first version recomputed the checkpoint's own
`data_selection.splits[*].eligible_identity_sha256` through `splits.flight_keys_by_split`, which
needed `TSConfig.from_dict(payload["config"])` and therefore re-imposed the model-recipe contract
on a reader that needs none of it — it raised `TypeError` (uncaught by the publisher and the
pipeline) on three real pooled checkpoints whose stored configs this build no longer accepts, and
it would have refused the 2026-07-29 `ts-data-selection-v1` audits, which name the same split
method but a different digest field. Verified by sweeping all 200 checkpoints under
`4dTrajectory/outputs` through `require_matching_data_provenance(payload,
checkpoint_data_provenance(payload, manifests))` at HEAD and after: 65 → 66 accepted, exactly one
newly accepted (`l1b_full_20260907/L1b_hr16_tv1_full`, the 08:55 arm), zero newly refused, zero
crashes; of the 134 still refused, 99 predate the multi-airport fingerprint entirely and 21 have
a genuinely changed eligible set (KSJC 18, KMSY 3) that now says so by name.

The publisher's byte comparison is deleted — it compares `eligible_set_digest` of the current
roster against the metadata's `eligible_sets`, and for older metadata loads the checkpoint
payload and goes through `require_matching_data_provenance(..., allow_subset=True)`.
`run_ts_pipeline`'s checkpoint reuse check does the same; its CV reuse check can only compare
bytes for pre-2026-09-08 `cv_results.json` (nothing in that artifact names the set), so a moved
roster re-runs CV there and only there. Four replay paths that fingerprinted WITHOUT the roster
(`run_ts_clock_attribution`, `run_ts_control_capacity_ceiling`, `run_ts_predictability_report`,
`approach_clustering.evaluation` — the `code-health-followups.md` §19 class, all four broken
against the v5 cohort) now go through `checkpoint_data_provenance`, and a fingerprint taken
without a roster the checkpoint recorded says exactly that instead of "the manifest changed".

**Two costs to know about.** The CV run contract renames its `eligibility_rosters` key, so an
IN-FLIGHT `cross_validation` progress file is refused on resume ("does not match the current CV
run contract") and its candidates must be re-run — finished `cv_results.json` files are read, not
refused. And a `checkpoint_metadata.json` written before this change carries no `eligible_sets`,
so the publisher and the pipeline pay one `torch.load` of the payload to answer the same question.

**Verification.** ts suite 693 passed / 1 skipped (679 before, +14 new in
`tests/test_eligible_set_identity.py`: byte-independence, one swapped key with the counts
unchanged, a re-graded reject accepted, a fingerprint taken without the roster, a v3 payload
accepted and refused, a v3 payload with no `data_selection` and an unreadable config, the predict
CLI on a synthetic v3 checkpoint, both publisher preflight paths, both pipeline reuse paths, and
the `pre_split_eligibility` byte facts). A 2-epoch synthetic train before
and after is bit-identical in model weights, forecast digest, metrics, splits, normalizer and
config — only the provenance/metadata blocks and wall-clock timings differ. Run-name recount over
169 stored configs: 0 renamed, 0 errors. On the real KRDU data (read-only):
`l1b_full_20260907/L1b_hr16_tv1_full` (the 08:55 arm) refused at HEAD and predicted all 1,404
val flights after; with a re-serialised roster copy (different `evaluation_report_sha256`,
reversed key order, different indentation, identical 14,378-key set) `L1b_hr8_tv1_full` and
`l1_lowdim_20260907/L1_native32` both refused at HEAD and both ran after, and the publisher's
`--dry-run` went from `blocked: eligibility roster SHA-256 mismatch` to `planned`.

### 2026-09-08 — ts_transformer wind readout: the tower headwind explains a tenth of the straight-in along-track residual

`run_ts_wind_residual_readout.py` (new CPU runner) joins each scored flight of a prediction
directory to the field's ASOS/METAR headwind at its landing time (`evaluation.wind`, the table
`dev-observed-load-factor-metar` brought in under `data/metar/`) and regresses the speed,
duration and endpoint along-track residuals on it. KRDU val straight-in (839 flights with a
report): speed-residual slope +0.62 ± 0.13 m/s per m/s headwind, duration slope −2.0 ± 0.5 s
per m/s, four wind bins monotone — the signal is physical — but R² = 0.10 (residual sd 4.8 → 4.5
m/s); the endpoint along-track error is unrelated (R² 0.01). Decision: no wind GPU arm; the
predicted rollout stays wind-free. Results doc
`4dTrajectory/ts_transformer/docs/2026-09-08_wind_residual_readout.zh.md`.

### 2026-09-08 — observed records name their type whenever the identity resolves; `--observed-only`

Of the 10,541 observed rows the v9 gate could not type (24.7 %), 9,056 resolve to an ICAO
type OpenAP does not model — the observed writer demanded OpenAP dynamics just to pick a
states mass the gate never reads, and dropped the type with it. `flight_scenarios.
resolve_airframe` now returns `(mass or None, typecode)` whenever the identity resolves;
`harvest/observed.py` writes `aircraft_type` for those rows with the nominal mass and says
so (`source.mass_source`). New harvest mode `--observed-only` rebuilds only `approach/`
(records, summary, report, publication; no arrivals rebuild, no roster deletion, no CZML).
The remaining 1,485 untyped rows are identity gaps (followups #24/#25). With the table
extended to 172 types the fleet's speed-indeterminate share fell from 24.7 % to 12.5 %
(96.0 % of 39,048 graded rows pass); what is left is unresolved identities (1,657) and bizjet
types with no published minimum mass (~3,900). The rebuild also brought KRDU 32 and KSMF 35R
into the observed batch — and exposed KSMF 35R's 39.4 m threshold error (its own entry). Tests: harvest writer (identity
without dynamics, mass_source), `resolve_airframe` unit test.
### 2026-09-07 — ts_transformer A0.b: the random-anchor arm was frozen by its own learning-rate schedule, and drew its anchors nearer the runway than its own anchor population

`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
§2.4c, branch `dev-a0b` (`12d35ce` the scheduler axis, `23cae12` the sampling axis,
`965077a` the two arms, `f8a1726` the docs, and one review-fix commit that REPLACED the
sampling law — see below). Two config axes, both defaulting to today's behaviour. **Neither
arm has been trained** — this is the code the §2.4c reading needs.

**What the second round measured.** `A0_random_hr8_tv1_p180` (early stopping off) still
peaked at epoch 10, and `A0_random_hr8_tv1_grid` (anchor-grid selection, the A1 lever)
peaked at epoch **8** — so neither pre-registered explanation survives: it was not the
stopping rule, and it was not a metric the grid could fix. The `_grid` arm's `history.json`
gave the mechanism instead. Its validation OBJECTIVE kept improving to epoch 60 (1.147 →
0.707, val state 0.264 → 0.102) while its selection metric stalled after epoch 8 (1100 →
1289 at 60). `ReduceLROnPlateau` was stepped with the **selection** value, so the learning
rate was halved from epoch 20 and reached **9.4e-7 by epoch 60** and 2.9e-8 by 100: from
about epoch 30 the model was not training, it was frozen at the epoch the *readout*
stalled. Per anchor set, 12 km went 722 → 1324 and only the 6 km set improved (291 → 218)
— the shape of a model that is mostly being shown the last few kilometres.

**`lr_plateau_metric ∈ {selection, objective}`** (default `selection`, pinned as a literal
in every named recipe). Under `objective` the scheduler steps on the macro validation
objective — the SAME `val_loss` the epoch record writes, chosen from the two numbers the
epoch already produced rather than computed again — and **checkpoint selection is
unchanged** either way. `checkpoint_metadata.json`'s `lr_scheduler.metric` says which it
stepped on; the run name carries `lr-metric=objective`.

**Two things the objective must not be watched through.** `objective` is REFUSED under
`latent_beta_warmup_epochs > 0` (the ramp reweights the objective every epoch) and under
`procedure_loss_dual_step > 0` (λ moves every epoch, so the same trajectory is priced
differently each time). In both the plateau would be the schedule's, not the model's, and
the scheduler would cut the rate straight through a ramp.

**`random_train_anchor_sampling ∈ {uniform, remaining-path-uniform}`** (default `uniform`,
refused without `random_train_anchor`). `uniform` draws one of a flight's admissible
SAMPLES, i.e. uniformly in TIME. **Per flight that is not a skew** — a flight's anchors
start at index 59 and the median flight has 84 of them, so a uniform draw centres at 100.8
by construction and lands at a measured median of 108; "median drawn anchor 107" was never
evidence of anything. **The skew is in the pooling**: every flight gets one draw whatever
its length, and a kilometre near the runway holds more samples than a kilometre at 25 km
because the aircraft is slower there, so the drawn distribution sits nearer the runway than
the anchor POPULATION it draws from. `remaining-path-uniform` places the draw uniformly
across the flight's OWN admissible remaining-path span and takes the nearest admissible
anchor — equal weight per kilometre — from the same per-flight per-epoch sha256, so
determinism is unchanged.

**Measured on the whole KRDU validation split** (1404 flights, 181,906 admissible anchors
under the 20 s contract, 200 epochs) — the share of the stored anchor POPULATION against
each law's share of the DRAWS:

| remaining path | population | `uniform` | `remaining-path-uniform` |
|---|---:|---:|---:|
| < 2 km | 3.8 % | 5.4 % | 4.4 % |
| 2–4 km | 10.7 % | 14.5 % | 13.4 % |
| 4–6 km | 10.6 % | 14.4 % | 13.2 % |
| 6–8 km | 10.3 % | 13.8 % | 13.1 % |
| 8–12 km | 17.7 % | 23.4 % | 24.5 % |
| 12–16 km | 8.5 % | 7.4 % | 7.3 % |
| 16–20 km | 5.5 % | 3.2 % | 3.2 % |
| **≥ 20 km** | **32.9 %** | **17.9 %** | **21.0 %** |
| mean / p50 / p90 | 17.2 / 11.2 / 40.8 km | 12.4 / 8.3 / 32.3 km | 13.4 / 8.9 / 35.3 km |

So the new law moves the draws TOWARD the population: ≥ 20 km 17.9 → 21.0 %, < 6 km 34.3 →
31.0 %, and the 8–16 km band carrying the 12 km selection anchor set 30.8 → 31.8 %.

**A stratum draw was written first and REJECTED by the review's own measurement** (700 KRDU
val flights): drawing an `anchor_strata` stratum uniformly and then a sample inside it moved
training the WRONG WAY — mean remaining path 12.2 → 8.0 km, p90 31.8 → 14.7 km, ≥ 20 km
16.9 → 4.1 %, and the 8–16 km band 31.6 → 27.4 %, i.e. the very set the arm exists to fix.
The cause is the grid's own shape: it cuts the near end into four 2-km strata while the far
end is ONE open stratum spanning 20–123 km that holds a third of the anchors, so equal
weight per stratum gives that third an eighth of the probability. **The strata survive only
as bookkeeping**: every epoch records the drawn counts per stratum BESIDE the population
they came from (`train_anchor_sampling.remaining_path_strata` and `_population`, under both
policies) — a drawn share alone cannot show over-weighting, which is exactly the mistake the
first draft made.

**The grid's values moved down a level.** `anchor_grid` imports `dataset` while `dataset`
needs the same kilometres to draw an anchor, so `DEFAULT_ANCHOR_GRID_KM`, the strata edges
and labels, `remaining_path_strata` and the draw law now live in the leaf `anchor_strata.py`
(no `dataset`, no torch) and `anchor_grid` re-exports the same objects — one import site for
every reading consumer, a module-scope import for the sampler, and no call-time import.
`tests/test_import_boundaries.py` pins the leaf.

**Admissibility is not part of the axis.** `eligible_random_train_anchors` and the 20 s
future contract still decide which anchors exist, so both policies store the identical
anchors for the identical cohort; only which one each epoch sees changes.

**Equivalence at the defaults, measured against `4943724`**: 2-epoch synthetic trains for
control / latent / state / closure are **byte-identical**, and a random-anchor `uniform`
arm differs only by the added `remaining_path_strata` keys — all 342 numeric leaves
unchanged, `sample_sha256` included. Recount over the **935** config-bearing artifacts
under `4dTrajectory/outputs` — every JSON with a top-level `config` object, plus every
campaign `config.json` override set rebuilt as a new run would render it (199
`history.json`, 255 `summary.json`, 193 `fit_evaluation.json`, 100
`experiment_manifest.json`, 95 campaign `config.json`, 93 others): **0** changed name, slug
or loading. Suite 680 → **719**.

**Arms** (`4dTrajectory/ts_transformer/docs/experiments/a0_random_arms.json`, both on the
`_grid` recipe — random anchors 20 s, hr=8 + bank TV=1, patience 180, anchor-grid
selection): `A0b_lr_objective` and `A0b_lr_objective_path_uniform`, so the pair separates
the two mechanisms instead of confounding them. The pre-registered reading: the L−1 and 12 km
anchor sets must stop degrading after epoch 10, the best epoch must be later than 60, and
the L−1 veto against native32's 1322 m is read as before — all three random-anchor arms so
far sit at 2949–2990 m, so this line's base is not established yet.
### 2026-09-07 — ts_transformer B1–B3: the arrival time becomes a calibrated interval, and a fan of flyable paths

`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
§三, branch `dev-b1` (`9f4149a` B1, `585b0e1` B2, `ebdb00c` B3, then the arms and the docs).
The B line's premise, measured by B0 the day before: the package's duration head is a POINT
estimate, there is no interval anywhere in it, and the |Δt| p80 a scheduler would have to
live with is 65.8–72.5 s on vectored flights against 12.0–20.3 s on straight-in ones. Code
only; **no arm has been trained**.

**B1 — the head.** `duration_head ∈ point | quantile`. `QuantileFinalTimeHead` emits the five
`config.DURATION_QUANTILES` (0.1/0.25/0.5/0.75/0.9) of the same quantity, **monotone by
construction** — cumulative softplus, so no input can produce a crossing pair and the loss
never has to price one. The median walks the existing `final_time_s` contract (it IS the
duration the rollout flies), so every downstream reader is untouched; the five go out as
`source.durationQuantilesS`. The loss REPLACES the `final_time` component under the same name
with the sum of five pinball losses on the same `(T − q)/final_time_scale_s` residual the
point head squares, so `loss_component_names` is unchanged. The median starts exactly where
the point head starts, so two arms that differ only in the head differ only in what they
learn.

Two things it refuses, and the second is the one the design left open. Off
`prediction_output=control` there is no such head. With `latent_dim > 0` it is refused
outright: z reaches the duration by SHIFTING the point head's single logit and a
cumulative-softplus head has five, and — the real reason — training decodes a POSTERIOR
sample, so the five would be quantiles of p(T | z ~ q(z | the flight's own future)), an
interval conditioned on the answer, which B2 would then calibrate as if it were
p(T | history).

**B2 — the calibration.** `calibration.py` (top level, torch-free, shared by `forecast`,
`predict` and both readouts) + `run_ts_eta_calibration.py --checkpoint PATH --out DIR`.
Conformalized quantile regression: score `max(q_lo − T, T − q_hi)`, δ_α its (1−α) quantile,
interval `[q_lo − δ, q_hi + δ]`, for α ∈ {0.2, 0.5} bound to the pairs (q10, q90) and
(q25, q75). The calibration set is the VALIDATION split and nothing else — the sealed outer
test would be spent on a number with no gate, and the training split's quantiles are fitted
to their own targets. It is halved by the checkpoint's own `split_seed`: **half A fits the
DEPLOYED δ and half B, which it never saw, measures what it covered**. The mirror (fit on B,
score on A) is printed beside it as a stability check and is never averaged in — a mean of
two δ is fitted on every flight it is then scored against, and nothing would measure it.

**The coverage is MEASURED, not guaranteed, and the language says so everywhere.** The
calibration split is also the split the checkpoint was SELECTED on: the LR schedule, the best
epoch and early stopping all step on a validation metric, and halving val does not repair
that — both halves fed the selection. So the artifact, the readout, the design doc and this
package's `CLAUDE.md` all say "empirically measured cross-half coverage", never "guarantee".
The coupling is worth naming as weak: selection reads a TRAJECTORY metric (common-grid ADE),
not the duration residual these δ are quantiles of. Pre-registered: the guarantee-bearing
number is a SINGLE held-out coverage read on the test split at the `freeze-test` ledger
stage, once every experiment decision is final.

**What flights actually GET is a different number from the per-stratum δ, and it is the one
the gate reads.** A per-stratum δ is measured on that stratum's own members, but deployment
takes the first stratum in the precedence a flight is in AND that has a δ — so a flight whose
stratum refused for thinness is handed the POOLED δ, about which the pooled row says nothing.
It need not be fine: the pooled δ is a mixture, and the fall-through group is by construction
the part of the cohort the mixture is least like. On a synthetic check (1 240 flights, 40 of
them vectored with a far wider truth spread) the pooled row reads 0.794 while the 18
fallen-through flights are covered **0.167**, and the deployed pooled number is 0.756.
`calibrate` therefore publishes a `deployed` block — every held-out flight scored under the δ
it would really get, pooled and broken out by `natural -> assigned` group — and design gate
3.4-2 reads it.

Three further decisions the design under-specified, written down where they live. δ uses the
finite-sample level `⌈(n+1)(1−α)⌉ / n`, not the plain (1−α) empirical quantile; when n is so
small that the level exceeds 1 the conformal answer is +∞ and that is raised, never clamped
to the largest score. The stratum rule is a PRECEDENCE (straight-in → vectored → pooled)
because of the FALL-THROUGH, not because of overlap — straight-in and vectored are disjoint
by construction, but they do not cover the cohort (a vectored flight already established is
in neither) and a refused stratum has no δ — with the record saying which it took
(`source.durationIntervalStratum`). And only the three strata the precedence can deploy are
fitted at all: a δ no record will ever read is a number in the artifact that nothing
measures.

The table is a SIDECAR in `checkpoint_metadata.json` under `conformal`, deliberately not in
`data_provenance` — `evaluate-fit` and `freeze-test` compare that object for equality, and a
calibration run would otherwise make every later replay report "the manifest changed". It is
bound to `checkpoint_sha256`; a table left behind by other weights RAISES. Without a table
`predict` writes `calibrated: false` and the raw quantiles, which is a claim, not a missing
key.

The runner reads the DURATION HEAD ALONE (`forecast.duration_quantile_predictions`: one
forward per flight, no rollout, no CTA). That is why it is seconds of CPU, and why it is the
one instrument allowed to load a `cta_conditioning=given` checkpoint
(`run_ts_anytime_curve.load_arm(..., refuse_cta_given=False)`) — the head reads the history
only, so its quantiles are the same function a prediction would use. The
`intent_conditioning` refusal still applies to it: that oracle is IN the history.

**B3 — the fan.** `predict --cta-from-quantiles`, for a checkpoint with BOTH
`cta_conditioning=given` and `duration_head=quantile`. In training the given (truth) CTA
drives the rollout exactly as L3 built it, and the quantile head trains as a target beside
it; at predict the rollout is driven by the model's OWN q_τ, one full prediction directory
per level under `quantiles/q10…q90/` (plus `a20lo` / `a20hi`, the calibrated α=0.2
endpoints, when a table exists), with the top-1 records BEING the q50 decode rather than a
sixth. Refused with `--cta-offset-s` (that shifts the truth) and with `--z-from-posterior`
(that reads the future).

**This is the first CTA arm that reads no future, and the naming had to be able to say so.**
`cta_conditioning` gained a third value, `self-q`, which no training run may select
(`CTA_CONDITIONINGS_AVAILABLE`, refused through `cli.common._refuse_unavailable_selection`,
the `--config-overrides` door included); `predict` stamps it on the config it writes beside
the records, so the run name, `summary.mode` and every record say `cta=self-q` while the
TRAINING directory still honestly wears `cta=given`. The readout is
`run_ts_quantile_fan_readout.py --arm <pred_dir>`: per stratum the share of flights whose
truth duration falls in [q10, q90] and in the calibrated interval, the median widths (the
design's veto reads the vectored one against 120 s), and gate 3.4-3 — the truth path's
chamfer to the NEAREST of the five decodes against its chamfer to q50, read on the in-fan
subset with the whole cohort beside it. That geometric column is a READOUT, not a coverage
guarantee: five trajectories are not a distribution over trajectories, and §六 6 forbids
writing it as one.

**Bit-exact at the defaults.** 2-epoch synthetic trains for control, latent k=1, latent k=4,
state, closure and cta=given against `c2fccda`, comparing every history float, a state-dict
digest and a forecast digest: **1519 non-wall-clock leaves, 0 differ**; the only new leaf is
`/config/duration_head`. Names recomputed for **168 stored configs on disk: 0 changed**.

**Opus review (2026-09-07), applied on the same branch.** It re-verified the equivalence, the
recount and the suite, and confirmed nine of the decisions. Four substantive changes came out
of it, all above: the `deployed` block and gate 3.4-2 pointing at it; the coverage language
downgraded from guarantee to measurement everywhere, with the test-split read pre-registered;
the deployed δ becoming half A's rather than the mean of both; and the table carrying its own
cohort so a `--limit` smoke table is refused at the sidecar unless `--allow-smoke-table` is
given. Also: the table's `quantiles` mirror is now CHECKED against `DURATION_QUANTILES`; the
fan readout marks its in-sample `cal.hit` column and looks intervals up by their own α; the
B1 arm discloses that the five pinball terms are ≈26× the point term at the same weight (from
B0's measured residual, mu 3.5 s / sigma 31.4 s) and pre-registers a conditional
`B1_point_matched` arm at `final_time_loss_weight: 26.0`; the calibrated interval ENDPOINTS
became `predict --interval-endpoints`, off by default, with the non-positive-CTA refusal now
tested; an inverted interval and an out-of-range conformal level raise instead of being
repaired; and `_NEW_RUN_VOCABULARIES` finally has a test that every entry on it bites.

Suite 680 → 729 (15 duration-head + 24 calibration + 9 fan tests, plus the `_NEW_RUN_VOCABULARIES` one).

**Arms**: `docs/experiments/b1_quantile_arms.json` — `B1_quantile` (point-vs-quantile
isolation on L1_native32's content) and `B3_quantile_cta` (+ `cta_conditioning=given`,
`predict_args: --cta-from-quantiles`), with the gates, the veto and the calibration COMMAND
between train and predict pre-registered in `_comment`. Nothing is trained yet.
### 2026-09-07 — 69 prediction categories published, and the identity that lets one checkpoint hold several

Branch `dev-publish2`. The A0 curve
(`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
§二) measured what a re-anchored forecast is worth and then threw the forecasts away — it kept
per-flight metric rows, no trajectories — so the anytime line existed only as a table.

**`run_ts_anytime_curve.py --write-records`** now also writes, per checkpoint and per bin, a full
prediction directory (`<out>/records/<label>/<bin>km/`) through `export.write_batch` — the same
shape `predict` writes, so `python -m evaluation` and `build_scenario_comparison_czml.py` take it
unchanged. The records come out of the SAME forward pass the cells were scored on (`measure_bin`
returns `(rows, records)`; a second replay would have made the published trajectory a different
measurement from the published number). `export.write_batch` gained `extra_summary`, refusing any
key that collides with the record contract's own, and the runner puts an `anytime` block there
(schema `ts-anytime-records-v1`) naming the campaign, arm, bin, anchor rule, split, `--limit`,
`measured_flights`, `records` and coverage. The artifact is now built INSIDE the `.partial-*`
staging directory and renamed on success, so a crash mid-run publishes no half-written record set.

**`publish_ts_experiment_trajectories.py`** reads that block off the reused prediction directory
and treats the bin as its own publication: category key `…_a12km_val`, its own raw-output
subdirectory (bins must not overwrite each other's evaluation report), `experiment.id`
`<run>@12km` (**the frontend picker dedupes by experiment id** — a shared id would have shown only
the first bin published), `experiment.group` = the record campaign so all bins sit under one
heading, and a label that states the bin AND both denominators (`@ 12km remaining (244 of 300
flights, --limit 300 of 1404 in the split)` — how much of the replayed cohort reached the bin,
and how much of the split was replayed): a subset's ADE must never read as the split's. The block
also rides in the publication manifest, so `--refresh-labels-only` recomputes the same name
without reopening the (read-only) record directory. Two refusals came out of review: a bin of a
POOLED checkpoint is one cohort over every airport it was trained on, and publishing it per
airport would file all of them under each — so a reused anytime directory whose rows are not all
this airport's is blocked; and repeating `--reuse-prediction-dir` for one checkpoint now errors
instead of silently keeping the last (publishing several bins is one invocation per bin).

**The CZML builder needed no change**, and there is now a test that says why: it already shifts
every prediction-schema entity by `source.anchorTimeS` (design §六 7), which is the general rule,
not the L−1 constant every earlier batch happened to share. `test_a_mid_approach_anchor_is_drawn_
where_the_flight_was_then` pins a 240 s anchor: the forecast is drawn from 240 s along the
observed flight's clock, its first sample IS the reference's sample at that offset, the lookback
fills [0, 240], and the arrival-window reference still spans the group (a full-track reference is
still refused).

**Published** (KRDU val, `--limit 300`, one replay process, 2 min 30 s wall on the RTX 4060):
three checkpoints — L1_native32, L2d_warm_beta0p01, A0_random_hr8_tv1_grid — × bins 12/8/6 km →
`4dTrajectory/outputs/KRDU/experiments/anytime_records_20260908/` (317 MB) and nine categories
under `aeroviz-4d/public/data/airports/KRDU/comparison/`. Coverage: 244/300 at 12 km, 300/300 at
8 and 6 km. Verified in-browser (KRDU → Experiments): the forecast starts mid-approach on the
observed track, not at the 25 km slice entry.

**Not fixed, written down instead** (`docs/code-health-followups.md` 21): the faded `look-`
entity is the observed track BEFORE the anchor, which is the model's input window only at L−1.
At a 12 km anchor it draws ~430 s of approach under a name ("Lookback" / legend "Predictor
input") that claims the model saw all of it. Nothing is mis-placed — only the segment's meaning
is overstated — and narrowing it needs a record-contract field (`source.lookbackSamples`) plus a
republish, so both ends now say what the line actually is.

**Hazard hit mid-session, since cleared**: KRDU's `arrivals/lateral_pass_eligibility.json` was
regenerated by another process at 08:15, after the publish. Its SHA-256 stopped matching what
these checkpoints recorded, so `preflight_error` blocked ANY (re)publication of them —
correctly, and including yesterday's L−1 categories. The cohort was never the thing that moved
(counts unchanged at 14,435/14,378 and every val flight of all four campaigns still eligible);
the roster's embedded `sources.evaluation_report_sha256` was, because the upstream v9 report had
been refreshed. Both airports' rosters were later restored to the v6-generation bytes the
checkpoints recorded, with the v9 refresh kept under `evaluation_reports/observed_v9_2026-09-07/`.
`--refresh-labels-only` was unaffected throughout (it reads stored manifests only).

### The second publish: 24 constraint-experiment categories, and one identity per publication

The earlier constraint / control-procedure / control-hook campaigns had prediction artifacts on
disk and no way into the picker. Twelve directories per airport were published for **KRDU and
KSJC** — `airport_frame_20260903/A_threshold_enu` (the state baseline, previously unpublished),
`final_constraint_20260904` {`B_corridor_bounded`, `C_procedure_fixed`, `C_procedure_dual`},
`control_procedure_20260905` {`A_control_v3`, `C_control_1e-3`, `C_control_5e-3`} and
`control_hooks_v2_20260906` {`F_barrier_soft`, `R_nominal_residual`}, plus the three predict-only
arms. `categories.json` went KRDU 32 → 44 and KSJC 31 → 43. Skipped deliberately: the `_s2024`
seed replicates, the superseded v1 `control_hooks_20260906`, and `F_barrier_infer` (the non-soft
v2 arm). The picker needs no per-airport work — it reads each airport's `categories.json` — and
the labels need no bespoke text: the campaign is the optgroup heading, the output path is the
run-name grammar's first token (`state` / `control`) and the arm is the run id.

**Three of those directories have no checkpoint of their own** — `A_project_on_final` and
`A_project_faf` reuse `airport_frame_20260903/A_threshold_enu`, `F_barrier_infer_soft` reuses
`control_procedure_20260905/A_control_v3`, each sitting beside that very checkpoint's own
baseline. Everything a category is named from comes from the checkpoint, so publishing them
would have silently replaced the baseline's CZML, evaluation report and picker entry. The
anytime bins had hit the same wall; `--category-variant SLUG[=LABEL]` (`ceab146`) generalises
their fix: `CategoryVariant` is the one thing that tells two publications of a checkpoint apart,
with exactly two sources — a directory that describes its own bin, or the flag — and **both at
once is refused**, because two answers to "which publication is this" is the ambiguity the
variant exists to remove. The other half of the rule is a `preflight_error` refusal: a directory
landing on a category another directory already holds is blocked, naming the flag, never
overwritten. The variant rides in the publication manifest so a label refresh needs no access to
the (read-only) prediction directory; manifests predating the field are read through their
anytime block. A refresh of all 38 publications reproduces every label byte-identically.

**KSJC's experiment index was five weeks stale** (2026-08-24) and listed none of the nine runs,
so the first KSJC pass failed with "checkpoint is not a completed indexed run" twelve times. All
nine carry `experiment_manifest.json` with `status: completed`, so `python -m experiment_index
--root .../KSJC/experiments` — the routine idempotent disk scan — fixed it. Worth knowing: the
publisher reads the index, not the disk, so a campaign that finished after the last rebuild is
invisible to it.

**Gotcha found while verifying** (now in `aeroviz-4d/CLAUDE.md`): a dev server that was already
running cannot serve a category directory created after it booted — `server.watch.ignored` on
`public/data` is exactly what would have refreshed vite's public-file list — and the SPA fallback
returns `index.html`, which the frontend reports as *"the airport data file is probably missing"*
for a file that is on disk. `categories.json` keeps working, so the picker lists a category it
cannot load. Restart the frontend after publishing new categories. Two further traps in that
restart, both hit today: killing the `npm run dev` wrapper leaves its **vite node child** holding
the port, and the supervisor's replacement then finds 5173 occupied and **silently falls back to
5175** — so the app looks restarted while the stale process still answers. Kill the node process.

### The rest of the day: the frame arms and the KRDU back catalogue

Two more passes finished the backlog. The remaining 2026-09-03 airport-frame ablation arms
(`B_airport_enu`, `C_airport_enu_target`, `A_threshold_enu_target`, seed 1337) went up for both
airports — six categories, each with its own checkpoint, no variant needed. Then thirty KRDU
categories in five groups: the six shifted-CTA counterfactuals (`l3_cta_counterfactual`, all
predict-only on the published `L3_cta` checkpoint, so all six are variants, and **each label
carries its scored count** because a shifted CTA below the 60 s floor drops flights — 1214 of
1404 at −90 s, 1392 at −60 s, 1400 at −30 s, 1402 at +30 s, all 1404 at +60/+90 s); the five
closure error-budget arms (four of them variants — the `C_pred` checkpoint now carries its
baseline plus three); the two teacher-free dense arms; ten remaining latent arms (three of them
z-oracle decodes of an already-published checkpoint); and seven screening/scene/anchor arms.
KRDU ended at **77** categories and KSJC at **46**, from 23 and 31 this morning. Nothing was
skipped beyond what was asked (the `_s2024` replicates, the superseded v1 hook campaign, and the
hard-barrier `F_barrier_infer` arms).

**Review of the variant flag** (`3ed0911`) found the guard that makes all of this safe was
itself unsafe. It compared the RENDERED `predictionDir` string — and every manifest on disk
stores an ABSOLUTE path, because a worktree resolves `4dTrajectory/outputs` into the main tree
and `relative_to(REPO_ROOT)` fails, while the same publication run from the main tree renders it
relative. A byte-identical republish therefore read as a collision, and the blocked branch then
wrote its document over the *completed* manifest, losing `accuracy`/`evaluation` and wedging the
category. The guard now resolves both sides (`samefile` when both exist), a blocked preflight
never overwrites a completed manifest, and new manifests render relative where they can — **the
manifests already written keep their absolute values and need no rewrite**. `--category-variant`
without `--reuse-prediction-dir` (a relabelled copy of the baseline) is refused at the boundary,
as is the both-identities conflict; only a completed manifest holds a category; the multi-airport
refusal now covers any reused directory; and `PUBLICATION_SCHEMA` is v2 so an old reader skips a
variant manifest instead of writing the baseline's id onto it.

### 2026-09-07 — ts_transformer A1: the selection metric was blind to what the random-anchor arm improved

`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
§2.4, branch `dev-a1` (`364adaf` the grid module, `4d563f5` the metric + tests, `04de172`
the arm, `d4ca2c3` the docs, and one review-fix commit carrying the coverage gate, the
oracle refusals and the dry-run fix). It is A1's PREREQUISITE, not the A1 row of that status
table: that row is the streaming evaluation protocol (`evaluation_protocol` anchor grid +
`compare_constraint_arms.py`), still not done, and this work is the new `A1.a` row beside it.

**The measurement that motivated it.** The A0-random arm (`docs/experiments/a0_random_arms.json`,
random train anchors + L1.b's teacherless supervision) early-stopped at epoch 30 with its
best at epoch 10 and failed the L−1 veto badly — ADE 2949 m against native32's 1322, FDE
p50 3784 vs 864. But the same checkpoint, replayed on the re-anchoring grid, draws **better
geometry than the fixed arm at every anchor** (chamfer p50 −114…−524 m) and better ADE at
≤ 8 km. The arm was not worse; it was **frozen early by a metric that cannot see what it
improves**. `checkpoint_selection_metric=fixed-anchor-common-grid-ade` scores the validation
set at the L−1 anchor ONLY — the one anchor a random-anchor model is least specialised for
— so once such a model starts spending capacity across the anchor range, the L−1 number
stalls and patience fires.

**One grid, two consumers.** The remaining-path grid moved out of `run_ts_anytime_curve.py`
into `4dTrajectory/ts_transformer/anchor_grid.py`: the bins
(`DEFAULT_ANCHOR_GRID_KM = 20, 16, 12, 8, 6, 4, 2` km), the 60 s future floor, the
per-flight `bin_anchor` rule (closest sample by remaining path, THEN admissibility) and the
fixed-at-L−1 `strata_fixed_at_l1` rule. The runner imports the same objects — a test asserts
identity, not equality — because two grids that merely agreed today would make "the curve
improved" and "this epoch was selected on the curve" statements about different anchors. The
runner's own 24 tests passed untouched.

**The metric.** `checkpoint_selection_metric=anchor-grid-common-grid-ade` averages the SAME
common-grid ADE over L−1 plus whichever of `VALIDATION_ANCHOR_GRID_KM = 16, 12, 8, 6` km the
cohort can cover, each flight at its own closest admissible sample. Equal weight **per anchor
set** — pooling all (flight, anchor) pairs would weight each bin by its coverage and fade the
far bins out exactly on the arms whose flights are vectored — and each set's ADE is the mean
over the flights that HAVE that anchor (per airport, then airport-macro, which is what the
existing metric already means, so the L−1 term IS its value). A flight absent from a bin is
absent from its mean, never scored 0. `history.json` gains a `validation_anchor_grid` block
every epoch (per set: ADE, flight count, per-airport split, plus `dropped_bins` and
`fixed_anchor_common_grid_ade_m` so the L−1 veto stays readable) and the epoch line prints
the values.

**A candidate bin is not a bin the metric will use.** Measured coverage of the KRDU
validation cohort (1404 flights, 60 s floor): 20 km **33.7 %**, 16 km **37 %**, 12 km
**78.7 %**, 8 and 6 km **99.7 %** — the median flight has only **13.4 km** left at L−1. So
20 km is not a candidate at all, 4 / 2 km are partial-to-empty under the floor (≈ 53 s /
27 s of truth left), and of the four candidates a bin under `anchor_grid.PARTIAL_COVERAGE`
(0.5, the SAME constant A0 marks a point `partial` with) is dropped at plan build with a
printed notice and a `dropped_bins` record — on that cohort, 16 km. Otherwise an
equal-weighted fifth of the selection value would come from the long-haul subcohort that
reached it. Fewer than two surviving bins refuses the metric outright: that is the L−1
metric with a companion, not a curve. **Which bins survive is a property of the cohort, not
of the model**, so every arm on the same split is selected on the same sets.

**Two oracle inputs are refused** under this metric (`TSConfig.__post_init__`):
`cta_conditioning=given` and `intent_conditioning != none`. Both read the future, and the
grid re-reads it AT EVERY BIN ANCHOR — the same reason `run_ts_anytime_curve.py` refuses
those checkpoints — so selecting an epoch on them would be selecting on how fast the oracle
converges. Both stay allowed under the L−1 metric, which reads the oracle once.

The ADE itself is never restated: `fixed_anchor_common_truth` became the L−1 case of
`common_truth_at_anchors`, each bin's cached plan carries its own truth, and
`dataset.ExplicitAnchorTrajectoryWindows` is the fixed-anchor policy with a caller-supplied
anchor per flight — same caching, same batch plan, same replay. The four extra sets are
built once per fit and replayed by `validation.replay_validation_plan` (one deployable
forward per batch, no objective); the L−1 pass is reused, not rebuilt.

**Cost**: the selection stage grows by about one deployable replay of the validation split
per surviving bin. Measured on a synthetic control run (40 flights, 7 val, 3 epochs, all four
bins alive): median `val_checkpoint_selection_s` **0.046 → 0.238 s** — 5.2× the selection
stage, 1.5× the whole validation stage (0.390 → 0.590 s). At KRDU scale, with three surviving
bins, the review's estimate is **≈ 4–5× the selection stage and +12–15 % per epoch**, i.e.
**+7–9 min on a 180-epoch arm**. That timer now also covers the selection call itself, which
it did not before.

**Nothing stored changes.** Every existing metric is bit-exact: 2-epoch synthetic trains for
control / latent / state / closure before and after, **232 history floats compared** (control
50, latent 78, state 54, closure 50), 0 changed — the only difference is the new
`validation_anchor_grid` key, empty on every other metric exactly as `latent` / `procedure`
/ `command_hook` already are. The naming/loading audit over every config-bearing artifact
under `4dTrajectory/outputs` — 816 stored configs (`summary.json` 239, `history.json` 196,
`fit_evaluation.json` 190, `experiment_manifest.json` 98 and eleven smaller kinds) recomputed
through `run_naming` and reloaded through `TSConfig.from_dict`, plus 93 campaign
`config.json` override sets rendered as a new run would be: **909 artifacts, 0 changed** in
name, slug or loading. (That tree is live — the GPU queue in the main worktree added 16
artifacts between the before and after scans — so the comparison is over the intersection
and the denominator is a snapshot, not a constant.)

**Naming.** `checkpoint_selection_metric` was already a `META_FIELDS` entry, so a
grid-selected run gets `select=anchor-grid-common-grid-ade` in its display name and always a
distinct slug; past six meta deviations it folds into `+N more`, where `run_slug` hashes it.
Moving it up the tuple so it is always spelled was measured and rejected: over every
config-bearing artifact under `4dTrajectory/outputs` it would rename **27 of 196
`history.json`** and **128 of 816 across all kinds** (all 38 `overfit_result.json`, all 15
`oracle_result.json`, 27 of 190 `fit_evaluation.json`, 6 of 239 `summary.json`, …). Those
totals move as the shared outputs tree grows — the review measured 27 of 165 and 64 of 667
on an earlier snapshot of the same tree — so the reproducible statement is the SCOPE (any
JSON under that root with a top-level `config` object, recomputed through `run_naming`), not
the denominator.

**The arm.** `A0_random_hr8_tv1_grid` = the p180 arm with that one field changed (verified by
diffing the generated configs). Not trained yet; the reading is pre-registered in its
`_comment`, including that the two earlier arms carry NO `validation_anchor_grid` block, so
comparing against them needs an A0 replay of their checkpoints, read mean-to-mean (A0's own
verdicts read the ADE median, which is not what this metric selects on).

**One runner archived, one dry run fixed.** `run_ts_frame_ablation.py` is now the only
arm-campaign driver; `run_ts_control_arms.py` moved to
`ts_transformer/archive/control_arms_runner_2026_08/` with a README, as audit T4-27
scheduled. Two behaviours found by using it made keeping it live a hazard: **it reads only
`base_recipe` and silently ignores an arm file's `base` block** (every declaration written
since 2026-09-03 has one, so it would have trained the bare recipe under the arm's name),
and **its `--dry-run` wrote `config.json` for every arm** — pointed at `4dTrajectory/outputs`,
a read-only symlink in a development worktree, a dry run left a campaign directory in the
shared tree. `run_ts_frame_ablation.arm_config` now takes `write=not --dry-run` and still
CONSTRUCTS every config (finding an unrunnable arm is most of what a dry run is for), and
`arm_steps` takes the resolved settings instead of reading the file back.

### 2026-09-07 — ts_transformer L2.f: the latent's information was in the wrong half of the KL

`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md` §六 L2.f, branch
`dev-l2f` (`be56088` diagnostics, `60dd620` β annealing, `6233967` the auxiliary duration
target, `63d1fdd` `run_ts_latent_probe.py`, `59640bd` the arm file, then one review-fix
commit). Nothing here changes an existing run: every code commit is bit-exact at the
defaults, proved by 2-epoch synthetic trains of four configs (plain control simple-v3/N=32,
latent control, state, closure) before and after each — 0 numbers changed, only the new keys
added (the review's own re-run moved exactly one value, `mean_displacement_sigma`, when its
median convention was fixed to `torch.quantile`; that key is new here, so nothing stored
changes). The audit scope for the naming/loading check is every config-bearing artifact
under `4dTrajectory/outputs`: **606 stored configs** (`history.json`, `fit_evaluation.json`, `summary.json`) recomputed through
`run_naming` and re-loaded through `TSConfig.from_dict`, plus **91 campaign `config.json`
override sets** reconstructed as a new run would — 697 artifacts, **0 changed** in name, slug
or loading verdict.

**Why.** L2.e' spent three 180-epoch arms on free bits as an information budget and read the
result as "the budget works, the information is just small". A direct probe of the three
checkpoints said otherwise: in every arm the posterior MEAN sat on the prior mean —
|μ_q − μ_p| under 0.2 prior σ per flight — and the whole budget went into narrowing the
posterior. z was a denoised constant. That explains the shuffled ΔADE of +47/+16 m, the
z-oracle's mere 227 m, the N(0, I) control beating the trained prior's best-of-6, and top-1
getting worse as the budget grew (1214 / 1260 / 1387 m). The mechanism: above the free-bits
floor the penalty's gradient on the KL's MEAN term is proportional to the displacement, so it
flattens each flight's posterior mean onto the prior within the first ten epochs — before the
decoder has learned to read z — and nothing afterwards brings the information back.

**Not one number a training run wrote could have shown this.** Total KL, active units and
shuffled ΔADE are all blind to WHERE the KL is spent. So the epoch record gained the split
(`component_kl_mean_term_nats` / `component_kl_variance_term_nats`, the variance term as the
REMAINDER of `per_dimension_kl` so the charged number stays bit-identical — and named
`component_*` because the three sum to the analytic `component_kl_nats_per_flight`, NOT to
the charged `kl_nats_per_flight`), `component_kl_per_dim`, `mean_displacement_sigma` (a
median is not summable: the flight-weighted mean of the per-batch medians, the same shape as
`active_units`, and taken with `torch.quantile(…, 0.5)` rather than `torch.median`, which
returns the lower of two middle values — an outside reader must land on the same number) and
`active_units_0p05` — the FIXED ruler, because `active_unit_threshold_nats` moves with the
free-bits budget and made that gate unreadable across arms. The gate sentence itself
(`displacement_verdict`) and its ruler (`DEAD_MEAN_DISPLACEMENT_SIGMA = 1.0`) live in
`control/latent.py`, so no surface restates either. `run_ts_latent_readout.py --history
<run>/history.json` prints the kept epoch's block and needs no `--arm` (this is the epoch-1
reading, before the arm has predicted anything); `run_ts_latent_probe.py --checkpoint
LABEL=PATH` measures the same quantities off a checkpoint (the scratch probe, now code beside
the other replay runners, sharing A0's cohort rebuild so the roster rule keeps one owner; its
prior total std comes from the MIXTURE's own moments, so a K>1 prior's range — which lives
between its components — is not read off one of them, and `--limit` is documented as a prefix
of the split, not a sample).

**Two levers, one arm each** (`docs/experiments/l2f_mean_information_arms.json`, base = L2.d's
warm posterior + free bits 0.05 + β 0.01): `latent_beta_warmup_epochs=40` ramps β linearly
from 0 so the decoder's z-weights grow before the mean term is charged (the L2.c "annealing
only postpones collapse" reading was about the VARIANCE term — recorded, not quietly
reversed); `latent_aux_duration_weight=1.0` adds a train-only `Linear(latent_dim→1)` on the
POSTERIOR SAMPLE predicting `truth_duration_s / final_time_scale_s`, a restoring force on the
mean that does not go through the decoder. The head is never called in `decode`, so z stays
latent and no record carries it; the combination with `cta_conditioning=given` is refused (the
CTA already hands the duration over).

**Three things worth keeping.** (1) The epoch's objective is now `replace(config,
latent_beta=effective)` and the VALIDATION pass is scored under the same one, so an epoch's
train and val `latent_kl` mean the same thing — the rule the procedure penalty's λ already
followed; `train_components.latent_kl` keeps its meaning (the term actually charged) and the
`latent` block is unscaled nats plus `beta_effective`. (2) Measured while building the aux
head (synthetic, three seeds, paired init): the target raises the KL's mean term in 3 of 3
seeds but the displacement MEDIAN in only 2 of 3, because a scalar read-out needs ONE latent
direction and a median over eight dimensions can miss it. The test asserts the monotone
quantity and the arm file says to read `component_kl_per_dim` beside gate (1)'s median rather
than being weakened to pass. **And the aux target is literally an INPUT of the posterior
encoder** (the true duration), so a small `latent_aux` proves only that one latent coordinate
can copy one input: the verdict needs the KL mean term to move AND a downstream gate to move.
(3) `simple-v*` now pins the WHOLE latent axis at its defaults, so a named recipe is
non-latent by definition and a latent run is `custom` — which every latent arm file already
said. Pinning only L2.f's two levers, as the first draft did, would have let a `simple-v3`
run choose its β but not its warm-up.

### 2026-09-07 — ts_transformer A0 / B0: the re-anchoring curve and the arrival-time error distribution

`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
§二 2.1–2.4 and §三 3.1, branch `dev-a0` (`3f7a849` A0, `fe84e76` B0, `5f408df` tests, then
the merge of `dev-l2` and this branch's review-fix commit). Both are MEASUREMENTS on existing
checkpoints and existing prediction directories — no training, no new model axis.

**Why.** Every evaluation in this package anchors at L−1 (`seq_len − 1`, 120 s after the 25 km
slice starts), which is the moment the ego history knows least about where the controller is
going to send the aircraft. The 962 m of intent in the C_pred → C_truth_intent budget is a
property of that INSTANT, not of the flight: it gets exposed as the aircraft flies. Nothing in
the package had ever measured that — each flight has exactly one anchor and one ADE. A0 asks
how the error falls as the remaining path shrinks; B0 asks how wide an arrival-time interval
would have to be, which is the number B1 (quantile head) and B2 (conformal) would be built
against.

**`run_ts_anytime_curve.py` (A0).** `--checkpoint LABEL=PATH`, repeatable; the cohort is each
checkpoint's own `val` split, rebuilt the way `predict` rebuilds it; bins are REMAINING PATH
(`--bins-km 20,16,12,8,6,4,2`), and a flight's anchor in a bin is the observed sample whose
remaining path is closest to the bin value, empty when it has no full lookback or under
`--min-future-s` of truth after it. Scored with the package's own `observed_series_metrics`.
Writes `anytime_curve.json` + `.txt` with the §2.4 readings. Six design points:

- **The strata are computed once at L−1 and fixed for every bin.** This is the whole
  difference between a curve and a survivor curve: relabel per bin and a flight leaves the
  vectored stratum exactly when it rolls out on the centreline, so the stratum improves
  because its hard members left it. Held down by a test with a two-flight cohort — one
  vectored at L−1 and established by 4 km, one straight-in — which must read vectored n=1 /
  established n=0 at the 4 km bin where per-bin labels would read 0 and 2.
- **The bin coordinate is the covariate's own arithmetic.** `approach_difficulty` gains
  `remaining_path_profile_m` (the per-sample form, one suffix sum) and
  `approach_difficulty()` now reads its own `remaining_path_m` out of it. The anchor grid and
  the NEAR / FAR strata therefore cannot become two definitions of one name.
- **Bins hold different flights, so the curve is read PAIRED.** A bin's population is the
  flights whose geometry put a sample there; an unpaired difference of two medians mixes "the
  model got better" with "this bin got easier flights". The monotonicity verdict compares
  adjacent bins over the flights present in BOTH and prints that n, and every per-flight row
  stays in the artifact so another paired reading needs no re-run. It reads the ADE
  **median** (the package's convention for a heavy-tailed error); mean and p95 are published
  beside it.
- **Both metric families, per cell.** Time-free chamfer and Fréchet
  (`geometric_metrics.path_metrics`, truth = the post-anchor supervision rows ADE is scored
  against) sit next to ADE / FDE / |Δt|. The closure arm is why: at 12 km it scores (SUPERSEDED: first replay before the closure fix `4ecfb69`; re-read 2026-09-08 from
  `anytime_a0_20260907/anytime_curve.json`, vectored: closure ADE mean 3649 / p50 621 m, FDE p50 402 m,
  |Δt| p50/p80 17/185 s vs native32 771 / 585 m, 1343 m, 45/85 s) — off its training
  anchor it is an out-of-distribution CONTROL, not a competitor, and reading one family alone
  says the opposite.
- **The arm is a per-checkpoint fact**, named from its own `random_train_anchor`
  (`A0-fixed` / `A0-random`): one run may hold both, their difference IS the
  out-of-distribution cost §六 3 asks for, and the out-of-distribution warning is printed only
  for the arms that need it.
- **`observed_series_metrics` was already anchor-aware** (`series.times[forecast.anchor]`,
  truth = supervision rows after it, difficulty at that anchor), so nothing at the export
  boundary had to change — asserted, not assumed: a forecast that copies the truth from a
  late anchor scores ADE, chamfer and Fréchet 0 there.

Refusals, all before the immutable output directory exists: `cta_conditioning=given` (its
duration IS the truth's) and `intent_conditioning=truth-…` (its oracle channels are re-read
at EVERY anchor, so the curve would measure how fast the oracle converges — three such
checkpoints exist under `scene_phase0_20260905`), the sealed `test` split, a malformed
`LABEL=PATH`, and a `--command-hook` without its saturation or on a non-control checkpoint.
The artifact is staged in a `.partial-*` directory and renamed, so a crash mid-measurement
never leaves a half-written curve under the name a reader will cite. `--limit N` is the smoke
test on a real checkpoint (the artifact says so, and the coverage denominators are the
flights actually built).

**`run_ts_eta_error_readout.py` (B0).** A pure readout of `summary.json`: per stratum the
|`final_time_error_s`| p50/p80/p90 and the SIGNED p10/p50/p90, and the same quantiles of
`fde_m`. A row is used only if it carries every metric AND every `STRATA_COVARIATES` field —
the tuple `strata_masks` reads, now exported, because a present-but-null
`established_at_anchor` would pass a numeric-only filter and then read as False, moving that
flight into the vectored stratum unnoticed. Measured on the three KRDU val arms (1404 flights
each, full coverage):

| arm | vectored \|Δt\| p80 | straight-in \|Δt\| p80 | signed p50 (all) |
|---|---:|---:|---:|
| native32 | 72.5 s | 20.3 s | +3.5 s |
| L2.d warm β=0.01 | 68.6 s | 18.9 s | +3.4 s |
| closure C_pred | 65.8 s | 12.0 s | −1.4 s |

Four things that decide B1/B2's shape: per-stratum calibration is necessary, not a refinement
(the vectored stratum is 3.6–5.5× the straight-in one, so a pooled interval is absurd for
straight-in flights); B is not dead on arrival but has only about one doubling of margin
against the §三 veto of a 120 s vectored interval; the error is near-symmetric but OFF-CENTRE
(+3.4/+3.5 s late for the control arms, −1.4 s early for closure), so a quantile head beats
"point ± δ"; and closure's FDE p50 of 11 m is a construction artifact (it draws the path onto
the threshold) sitting next to a 996 m ADE — the standing "read both metric families" rule,
with a fresh example.

**A replay runner must fingerprint the data the way the checkpoint was trained.** The v5
cohort is eligibility-bound: `arrival_data_provenance(manifests)` without
`eligibility_rosters` lists 14 435 KRDU arrival candidates where the checkpoint carries the
14 378 eligible ones, and `require_matching_data_provenance` then reports "the manifest
changed" on a checkpoint and a manifest that are both correct. Found here, fixed for the
package in `e8df12f` as `data_provenance.checkpoint_data_provenance(payload, manifests)` (the
roster is read iff the checkpoint recorded one), which this runner calls — its own copy was
deleted when `dev-l2` merged. It had been a live startup blocker for L5.a's unrun fit.

**Two things measured while verifying, which are not results**: all three §2.2 arms replay at
anchors 87–260 against L−1 = 59 (closure included — `closure_output.reconstruct` draws from
the ANCHOR STATE and its labels only ever enter training and `--closure-from-labels`); and
the whole A0-fixed run costs ≈76 min per control arm on CPU at 1404 flights × 7 bins, with
the closure arm roughly two orders of magnitude cheaper. Batching is by ANCHOR, and real
anchors are nearly all distinct, so the effective batch is ≈2.7 and `--batch-size` barely
matters.

**Two gaps this work found in the design's own plan**, both now written into it:

- `--min-future-s 60` empties the 2 km bin and most of the 4 km one (≈27 s and ≈53 s of truth
  left at approach speed). The readout states it (`n=0 / cov 0.00 / partial`) and gate 2 only
  asks about s > 4 km, but **s_freeze can then only be read at ≥ 6 km**. Reading it closer in
  needs a lower floor, which changes the bins' population and must be quoted with the result —
  and the **~125 s duration-head floor** makes |Δt| p80 RISE toward the runway anyway
  (L1_native32 vectored: 64 s at 12 km, 141 s at 4 km), so every cell now prints its predicted
  duration p50 and the freeze block names the floor.
- **The A0-random arm cannot be trained yet**: `random_train_anchor=True` raises
  `TypeError: RandomAnchorTrajectoryWindows.__init__() got an unexpected keyword argument
  'fitted_teacher'` before the first epoch (introduced by `e9e3639`, no current recipe uses
  random anchors, so the suite is green). Since §六 3 forbids publishing the fixed arm's
  curve alone, that one-line fix gates the whole A0 reading —
  `docs/code-health-followups.md` §20.

Tests: `4dTrajectory/ts_transformer/tests/test_anytime_curve.py` (24) and
`test_eta_error_readout.py` (10); package suite 616 → 626.

### 2026-09-07 — ts_transformer T3 review: the guards that did not grow with the module map

`e1fe10f` (code + tests) and this entry, branch `dev-t3`. The independent review passed on
behaviour — 15,241 in-process leaves and 316,993 CLI-path leaves identical, 0 import cycles,
every moved function body AST-identical, 0 stored names or load verdicts moved — and found
what a structural pass systematically misses: a guard that stayed where it was.

- **`dataset` was accidentally re-exporting `DYNAMICS_CONDITION_NAMES`.** The import had no
  use in `dataset`, but three tests read it THROUGH `dataset`, so the follow-up entry saying
  it was "safe to delete" would have cost seven failures. They import it from
  `control.conditioning` now.
- **Four renamed flags still worked.** argparse accepts any unambiguous PREFIX by default,
  so `--dt`, `--control-recipe`, `--closure-labels` and `--control-fitted-teacher` kept
  parsing after the T3-19 rename — a stale command line setting the field it looks like it
  sets while reading as up to date. `allow_abbrev=False` on every parser, and a test that
  runs all fifteen old spellings through `main()` and requires "unrecognized arguments".
- **`RETIRED_SERIALIZED_FIELDS` dropped non-default values silently**, and its header
  ("could not have changed the run") was false for the three fields T3-21 put in it — they
  are live loss denominators. Split into the unread kind (dropped by name) and
  `RETIRED_CONSTANT_FIELDS`, a `{field: constant}` map that drops a stored value only when
  it EQUALS the constant and otherwise refuses, naming the key and the value. Exposure is
  zero — 0 non-default across the artifacts on disk — so no stored config changes verdict.
  The three constants are single-sourced in `config.py` rather than mirrored.
- **The barrier gains could not reach the delivery form.** T3-21 put them on `train`, where
  the hook cannot be enabled without training through it, while the ADOPTED use is
  `predict --command-hook barrier`. Both are on `predict` now, folded into the existing
  `replace(...)` and refused without the hook, behind `cli.predict.PREDICT_CONFIG_FLAGS` —
  the same import-time assertion `CLI_CONFIG_FIELDS` carries. `--command-hook` /
  `--hook-saturation` keep their short names on purpose: two arm files spell them in
  still-re-runnable `predict_args`, and renaming rewrites a finished campaign's record.
- **One `latent_dim` refusal came back**, in `_latent_batch` — T3-22 removed four copies on
  the grounds that `__main__` checks, which is true of the CLI and not of the library's other
  callers. It is in `_latent_batch` rather than the fan-out because two of the four entries
  read the prior in between and would die on `AttributeError: prior_logits` first.
- **`CLI_CONFIG_FIELDS` was guarded in two directions of three**: deleting a name left its
  flag parsed and ignored. `NON_CONFIG_DESTS` freezes the infrastructure dests and the
  documented exceptions, and the test now asserts set equality.
- Also: `validation` added to the control-layering ban; the two-way `dataset` ↔ `control/`
  edge stated and pinned (the four modules `dataset` imports must stay `dataset`-free);
  `splits` guards its annotation-only `dataset` import and joins the torch ban;
  `validation._dataset_loss_components` moved into the test module that was its only caller;
  the import-boundary helper proves its poison is live before importing anything;
  `test_architecture` resolves relative imports; all four point-mass hook refusals covered.
- **`run_slug` now hashes the diffs `_MAX_LISTED_META` folds into `+N more`.** The barrier
  gains are LAST in `META_FIELDS`, so two runs differing only in them were the first pair to
  fold — and a slug names a FUTURE directory, so they would have shared one. 131 stored
  configs' slugs gain a 6-hex suffix; nothing on disk is renamed (existing directories are
  historical record), display names are untouched, and there are 0 actual collisions before
  or after, so the fix is prophylactic.

Numbers this entry and the one below it got wrong, re-counted: `__main__.main` was 679 lines
not 714; `__post_init__` had 97 `raise` statements not 123; `cli_values` was 58 pairs and
`CLI_CONFIG_FIELDS` 55 (57 after T3-21), not 60 and 54; `_refuse_hook` had 4 call sites not
six. The equivalence itemization is restated so it no longer reads as a partition of 9,197,
and two claims are softened: `evaluation_protocol` shed torch and openap but still loads
numpy and the harvest package, and the `cta_offset` guard is kept for its test as well as
for its non-CLI caller. Stale module references fixed in `config.py`, `forecast.py`,
`final_approach_geometry.py`, `control/loss/components.py`, `docs/ENGINEERING_NOTES.md` and
four `docs/*.md`; `README.md`'s Layout table gains `objective`, `validation`,
`data_provenance`, `splits` and `cli/`.

Suite 583 → **591 passed**.


### 2026-09-07 — ts_transformer package audit T3: the structural rearrangement (eight commits)

`4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md` §五, branch `dev-t3`,
base `4e00c59`. Nothing here changes a number: every item is a move, and the series is
closed by an equivalence proof rather than by an assertion. Suite 580 → 583 (three new
architecture / boundary tests); stored-config sweep at base and at HEAD over every
`history.json` / `summary.json` / `checkpoint_metadata.json` under
`4dTrajectory/outputs/*/experiments/**`: 306 shared configs, 233 load at both, **0 load
verdicts changed, 0 run names changed**.

**The training plane is five modules now, one question each** (`3d3dba0` T3-15,
`7a27791` T3-16, `ae9d83f` T3-18). `train.py` was 3,027 lines holding three of them.

| module | question | lines |
|---|---|---|
| `objective.py` | what a prediction is scored against | 1,079 (new) |
| `validation.py` | how a fitted model is replayed on a split, which epoch is kept | 869 (new) |
| `train.py` | the epoch, the cohort, the checkpoint | 1,194 |
| `data_provenance.py` | which arrival rosters produced this run | 258 (new) |
| `splits.py` | which split a flight belongs to | 192 (new) |
| `dataset.py` | observed arrivals → model windows | 1,742 |

Each carve was verified line-for-line against `4e00c59`: 924, 754, 221 and 147 non-blank
lines moved unchanged. Two consequences worth keeping:

- `batching._probe_training_step`'s three in-function imports are gone. The deferred
  `from train import prediction_loss` existed only to break a cycle (`train` imports
  `resolve_batch_size`); with the objective in its own module there is none, and `dataset` /
  `fixed_dt_supervision` had been deferred alongside it for no reason of their own.
- `evaluation_protocol` took `require_matching_data_provenance` from `dataset`, so reading
  the test-release ledger imported torch and openap to compare two dicts. It reaches
  `data_provenance` now, and
  `test_the_provenance_and_protocol_modules_do_not_reach_the_data_plane` imports it with
  `torch` banned. It is not import-free: `data_provenance` still pulls numpy and the harvest
  package through `trajectory_data_process.harvest.arrivals`. Torch and openap are what
  went.

**`TSConfig.__post_init__` is five named validators** (`24aee2d` T3-17): 544 lines and 97
raises, in an order nobody could hold, become `_validate_vocabulary` → `_validate_recipe` →
`_validate_ranges` → `_validate_output_contract` → `_validate_control_contract`, and the
order is the argument for it (a value must be in its vocabulary, then in its bounds, before
a cross-field rule can read it). The `n_segments` default sits BETWEEN the vocabulary pass
and the rest, because `DEFAULT_N_SEGMENTS_BY_MODEL[self.model]` would raise `KeyError`
instead of the vocabulary's `ValueError` if it ran first.

**The plan was wrong twice, and both are now measured facts in the code.**

1. `control_state_loss_grid` is NOT derivable from `control_state_objective` after T1-11.
   Census over the 306 stored configs: 159 `(control, true-time-position, native)`, 62 + 8
   `(state|closure, normalized-mse, native)`, **4 `(control, normalized-mse, fixed-dt)`** —
   the arms `CLAUDE.md`'s defaults table describes. `true-time-position` does pin the native
   grid; `normalized-mse` runs on both. Two values under one objective is not a function, so
   the six mutual raises stay, with the census in `_validate_control_contract`'s docstring.
2. Of the seven `*_scale` fields "never non-default on disk", only three may become module
   constants. `control_simple_v1_overrides` pins `position_loss_scale_m`,
   `final_time_scale_s`, `control_velocity_loss_scale_mps` and
   `control_heading_rate_loss_scale_dps` as LITERALS — that is what T0-7 did, so that a
   module default cannot silently redefine a published simple-v* comparison. Retiring them
   would reinstall exactly that hazard.

**Three units retired, two hook gains gained flags** (`4148a53` T3-21).
`objective.PROCEDURE_LATERAL_SCALE_M` (100 m), `objective.PROCEDURE_VERTICAL_SCALE_M` (30 m)
and `closure_output.CLOSURE_TIMING_SCALE_S` (60 s) leave `TSConfig` into
`RETIRED_SERIALIZED_FIELDS` (143 / 143 / 81 stored configs carry them, none at anything but
the default), out of both `run_naming` tuples. `--control-barrier-alpha` and
`--control-barrier-heading-gain` exist at last: `--config-overrides` JSON was the only way to
set either, which is how six 2026-09-06 configs carry the first campaign's heading gain 0.3
rather than a chosen value.

**A dynamics backend is a row, not a class** (`2fc5560` T3-20). Three near-identical classes
behind an ABC become `ControlDynamicsBackend(description, endpoint_fn, dense_fn, post_fn,
runs_hooks)`; `_TransportChartResults` was `post_fn` written twice, and `_refuse_hook`'s four
call sites become one `_admit` driven by `runs_hooks`. Bit-exactness was measured, not
asserted: all three registered pairs (and the lagged one again under a barrier hook), endpoint
and dense, 13 tensor digests plus the control gradient each — byte-identical.

**The four latent forecast entries share one fan-out** (`c48adf8` T3-22): `_latent_batch` +
`_latent_forecasts(latents, probabilities, …)`, where `probabilities is None` states once
what four call sites used to imply — latents that are not prior samples carry no mode index.
The four `latent_dim < 1` re-checks are gone (`__main__` refuses every latent flag against a
non-latent checkpoint, twice) — the T3 review put ONE back, in `_latent_batch`, for the
library's own callers. The `cta_offset` guard in `forecast_approaches` is NOT gone, against
the plan: `test_the_offset_is_refused_off_the_cta_path` pins it, and the entry has a caller
outside the CLI (`run_ts_runway_hypotheses.py`), so deleting it would have removed the only
check on that path and a green test with it.

**BREAKING (CLI): one module per subcommand, and fifteen flags renamed** (`dba2450` T3-19).
`__main__.main` 679 → `__main__.py` 131 lines (and `cli/` 1,542): a bootstrap plus a `COMMANDS` table of
`(help, add_cli_arguments, run_cli)`, the pattern `approach_clustering/cli.py` already used.
`cli/{train,cross_validate,evaluate_fit,freeze,predict}.py` + `cli/common.py`. Every
flag that sets a `TSConfig` field is now named after that field, so the 58-pair hand-written
`cli_values` mapping is a list of 55 field names with an import-time assertion (57 after
T3-21 adds the two barrier gains). Old spellings
are not accepted:

    --closure-labels                 -> --closure-labels-path
    --dt                             -> --dt-s
    --fitted-tail-weight             -> --fitted-tail-position-weight
    --fitted-terminal-weight         -> --fitted-terminal-position-weight
    --kinematic-consistency-weight   -> --kinematic-consistency-loss-weight
    --control-thrust-tau-s           -> --control-thrust-time-constant-s
    --control-bank-tau-s             -> --control-bank-time-constant-s
    --control-load-tau-s             -> --control-load-time-constant-s
    --control-state-clock            -> --control-state-supervision-clock
    --control-fitted-teacher         -> --control-fitted-teacher-path
    --control-heading-rate-weight    -> --control-heading-rate-loss-weight
    --control-heading-rate-scale-dps -> --control-heading-rate-loss-scale-dps
    --control-bank-tv-weight         -> --control-bank-tv-loss-weight
    --control-rollout-dt             -> --control-rollout-integrator-dt-s
    --control-recipe                 -> --control-recipe-name

Updated wherever spelled: `run_ts_flight_model_paired.py`, the two flags `run_ts_pipeline.py`
EMITS (its own same-named flags are a separate CLI surface and keep their names, with a
comment at the emission site), `tests/test_ts_transformer.py`, the package `README.md` and
eight `docs/*.md`. Older entries BELOW in this file (newest first) keep the spelling they
were written with.

**The equivalence proof that closes the series.** Six tiny CPU runs on synthetic KRDU
arrivals, fixed seeds, two epochs, trained once at `4e00c59` and once at `4148a53`: simple-v3
supervision content at N=32, the latent config (`latent_dim=8`, posterior init 0.1, β=0.01,
free bits 0.5), the L1.b terms armed (heading-rate 8, bank-TV 1), `state`, `closure`, and the
lagged model under `control_command_hook=barrier`. **9,197 leaf values compared, 8,951
identical.** The categories that carry the claim, each counted separately (they do not
partition the 9,197 — the rest are the strings and identities around them): 6,752
`history.json` floats, 6 state-dict SHA-256s, 6 target contracts, 6 checkpoint metadata
schemas, 750 forecast field digests (top-1 everywhere, plus 2 prior modes, the z-oracle and
the shuffle on the latent run), 6 run display names, 6 slugs, 72 normalizer statistics, 34
split rosters — 0 differences in any of them. The 246 that differ: 240 wall-clock timings,
and the 6 `.pt` file digests, which move because the serialized config lost the three fields
T3-21 retired (111 → 108 fields, every shared value equal).

### 2026-09-07 — evaluation v9: the speed gate anchors on the type's PUBLISHED approach speed (O4)

Branch `dev-observed-load-factor-metar`. Under the wind-corrected v8 gate 26 % of the
observed fleet still failed speed, by airframe family — the 737 bucket's stall-model
`1.23·Vs1g(MLW)` was 134 kt against a published 140–147 kt. All those flights landed, so the
anchor was the error (`evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` §9). Owner decision:
plan A (published V_ref) with plan B (data calibration) only as a fallback, determinate
bounds, no probabilistic or three-valued verdict, every source downloaded or cited in a
dedicated folder.

- **New `aircraft/reference_speeds.json` + `aircraft.reference_speeds`** (stdlib): per ICAO
  type the FAA Aircraft Characteristics Database (October 2024) approach speed at MALW
  (min/max over the FSB's flap configurations), the MALW, and the type's lowest published
  operating mass (manufacturer MFW/OEW/BOW, else OpenAP 2.4) — one source id per number.
  `docs/reference_speeds/README.md` indexes every document (URL, retrieval date, SHA-256,
  page, excerpt); the files live under `data/reference_speeds/` (git-ignored,
  `fetch_sources.sh` re-downloads). Sources: FAA ACD xlsx; Airbus A319/A320/A321 AC (3-5-0
  Final Approach Speed); Boeing 737NG/737 MAX/757/767/777/787 ACAP (§2.1 weights); Embraer
  175 APM; Bombardier CRJ900 APM; Eurocontrol Aircraft Performance Database (Vat).
- **`evaluation/speed_gate.py` rewritten**: `V_ref,x(m) = V_x·√(m/MALW)`; computed records
  `[V_ref,lo(m)·√max(n,1), V_ref,hi(m) + 20 kt]` at the crossing mass; observed records
  `[V_ref,lo(m_min)·√max(n,1), V_ref,hi(MALW) + 20 kt]` over the type's published mass range
  (an ADS-B track carries no mass). Criterion ids `published_vref_at_crossing_mass_and_n_…` /
  `published_vref_over_type_mass_range_and_n_…` (+ `_metar_airspeed_estimate` /
  `_ground_speed_proxy`). Bounds carry `reference_typecode`, `reference_sources` (the table's
  source ids, so each row traces to its documents), `mass_basis`, `vref_low_ms`,
  `vref_high_ms`; every row carries `speed_reason`. The record's type: `source.dynamics_typecode` (computed)
  or `source.aircraft_type` (observed); `source.landing_aero` is no longer read or validated,
  the harvest's observed writer no longer writes it, `backfill_landing_aero.py` deleted (the
  70,267 on-disk optimizer records carry `dynamics_typecode` and regrade by report
  regeneration).
- **Verdict is pass/fail against the window**; `speed_uncertainty_ms`, `speed_marginal`,
  `speed_uncertainty_unknown` and the ±5 kt declaration removed; `speed_margin_ms` kept;
  new batch `speed_indeterminate_reasons` (count per cause over the same rows as
  `speed_result_counts`: no crossing, no type, no published entry, no minimum mass, no
  crossing speed). METAR correction unchanged. `flight_scenarios.resolve_landing_aero` →
  `resolve_airframe` (mass + type; the stall block had no consumer left).
- Observed re-evaluation (`observed_v9_2026-09-07/`): **96.9 % of graded rows pass** (v8:
  73.8 %); 999 fails, 992 fast by a median 2.2 kt, 747 of them pushed over by the tower
  headwind, the type clusters (E75L, B737) on single-flap-value FAA rows — followups #21–#23;
  `BASELINE_SPEED_GATE_RESULTS.md` §10.
- Schema v8 → v9 in all four homes; ts seam readable set (v6…v9); frontend shows the
  published V_ref in the speed cell's tooltip and labels v6–v8 as stall-anchored.
- Docs: `THRESHOLD_SPEED_GATE.md` §1/§2/§3.1/§3.2/§4/§5/§6/§7/§8/§9/§10 rewritten;
  `evaluation/CLAUDE.md`, `README.md`, root open item, `open-items.md`, followups #16/#19
  resolved, `trajectory_data_process/CLAUDE.md`, `flight_scenarios/CLAUDE.md`.


### 2026-09-07 — ts_transformer L1.b: two supervision terms that replace the imitation teacher's job of naming the bank

`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md` §六 L1.b, branch
`dev-l1b`. The imitation teacher's one real job is to NAME THE BANK — position is derivative
order 0, velocity order 1, bank order 2, and nothing else in the objective reaches it
(unsupervised bank skill 0.124, below a trivial baseline; simple-v3's teacher takes it to
0.735). But its target is solved by inverse dynamics from the flown track and is NOT
consistent with the rollout: flown open-loop it drifts 2.5–7.8 km, so the decoder is pinned
to controls that do not reproduce the track and the position and imitation terms fight.
Both new terms price the same information THROUGH the rollout instead.

**`heading_rate` (arm ②, `control_heading_rate_loss_weight`, default 0).** Per flight, the
masked mean over the N segment endpoints of
`((ψ̇_pred − ψ̇_obs) / control_heading_rate_loss_scale_dps)²`, deg/s.

- *Target* — `dataset.reference_heading_rate_supervision`, training-only (it reads the
  future, so it sits beside `reference_control_supervision`, never in `dynamics_arrays`):
  the observed ψ from the velocity channels through `states_from_channels` (the modeling
  layer's math-ENU heading, so the sign convention is the rollout's own), `np.unwrap`-ed,
  then a CENTRAL DIFFERENCE over `HEADING_RATE_SMOOTHING_WINDOW_S = 10 s`, sampled at the
  endpoint times. ADS-B reports a quantised track angle; a single 2 s difference of it is
  noise, not a turn rate, so the window is a stated constant rather than a free parameter.
  **The constant is NOMINAL**: the half-width is `int(round(W / 2 / dt_s))` samples and the
  realized span `2 · half · dt_s`, so at the package default `dt_s = 2 s` it is
  half = round(2.5) = 2 and a span of **8 s, not 10** — quote the realized span. Rounding
  goes both ways (`dt_s = 3 s` → 12 s, `5.05 s` → 10.1 s; stated at the constant), and a
  `dt_s` too coarse for even one sample of half-width is REFUSED with its numbers rather
  than floored to one and silently widened. Weight zero past the last measured velocity,
  read at the ENDPOINTS rather than at the segment midpoints the imitation mask uses (a
  turn rate is a quantity at an instant, a piecewise-constant control one over a segment) —
  the same cut the velocity term makes, **not the same numbers**: this is a hard 0/1 step,
  the velocity term's an interpolated per-channel weight.
- *Index pairing* — the target's uniform `(k+1)·T/N` endpoints ARE the rollout's
  `cumsum(segment_durations)` only through one chain: `true-time-position` ⇒ `TSConfig`
  admits `control_duration_parameterization="uniform"` only ⇒ endpoints at `(k+1)·T/N`.
  Under a non-uniform partition row k of the two sides would be different physical instants.
  `reference_control_supervision`'s midpoints carry the identical unstated assumption; both
  docstrings now name the chain.
- *Model side* — `train.control_heading_rate_mse` calls
  `aerodynamic_model.torch_dynamics.heading_rate_rad_s`, which reads the ψ row out of
  `enu_rhs` ITSELF at the rollout's endpoint state. **The pre-registration wrote the
  relation as `g·tan φ / V`; the RHS actually integrates `g·n_realized·sin φ / (V·cos γ)`
  with the stall-limited load factor.** The two coincide only on a coordinated level turn
  (`n = cos γ / cos φ`), and calling the RHS is what makes "model-consistent by
  construction" true rather than approximately true. It is evaluated on the controls the
  aircraft had ACTUALLY reached: `EndpointControlRollout` gains `actual_controls` —
  `inputs.controls` under the point-mass backends, `lag_actuator_states` under the
  first-order lag. Pricing the command under the lag would charge a turn the rollout never
  flew. (The chart backends add the moving-frame `ω×v` term on top of the force part, which
  this expression omits: mean 3.5e-4 – 9.0e-4 deg/s, worst ~3e-3 at bank 0.3 rad / 60 m/s,
  i.e. ≤ 0.2 % of the 1.5 deg/s scale — stated in the docstring rather than silently
  dropped.)

**`bank_tv` (arm ③, `control_bank_tv_loss_weight`, default 0).** Per flight, the mean
absolute step between adjacent COMMANDED banks divided by `CONTROL_HALF_WIDTH[BANK_INDEX]`
(new constant, so nothing hard-codes a 1). The commanded schedule is the one the head owns —
penalising the lagged actual bank would charge the actuator for the command it was given.
**It prices REVERSALS, not slope, and it cannot start on its own** — both read off the
subgradient of `|x|` and both pinned by a test. On a monotone run the interior terms cancel
(`+1` from the left step, `−1` from the right), so a smooth roll-in costs exactly what a
single step of the same total size costs, while a schedule that turns around pays at every
turn. At EXACT flatness the value AND the gradient are zero — a stationary point, and the
one every run begins at, because `control.heads._initialize_control_head` zeroes the
projection so all N commands are identical. Only the other terms leave it. If arm ③ reads
"the TV term changed nothing", that is the live explanation to check before the dose.

Both register in `loss_component_names` under `control_state_objective=true-time-position`
ONLY, exactly where `velocity` and `imitation` do, and **`TSConfig` REFUSES a non-zero
weight under any other objective** rather than accepting a number that cannot change an
answer — the fixed-dt grid silently having no imitation term is the trap this avoids
repeating. All three fields are spelled as literals (0.0 / 1.5 / 0.0) in
`control_simple_v1_overrides()`, the ONE recipe literal dict every named recipe derives from,
so a default that later moves cannot redefine a published comparison. CLI
`--control-heading-rate-weight` / `--control-heading-rate-scale-dps` /
`--control-bank-tv-weight`; `run_naming` abbreviations `hr` / `hr-scale` / `bank-tv`.
**Run-name recount over the 306 stored configs on disk: 0 changed** (every one predates the
fields and reads their defaults).

Arms `docs/experiments/l1b_supervision_arms.json`, SCREENING tier, FOUR arms on
`base_recipe: simple-v3` with `control_recipe_name: custom`, `n_segments: 32`, imitation 0,
`epochs: 60`: `L1b_native32_e60` (the MATCHED CONTROL — the same base with the teacher back
at 64, i.e. L1_native32 at the screening budget), then `L1b_hr1` / `L1b_hr8` /
`L1b_hr8_tv1`. The control arm is not optional: the published native32 numbers (1322 m,
bank skill 0.726) come from a 180-epoch run that never early-stopped, i.e. a budget-limited
upper bound, so reading a 60-epoch screening number against them would confound supervision
with budget. Early stopping is LIVE at 60 epochs (patience 20), and the epoch an arm reaches
is part of its reading. The doses are a CALIBRATION, not a claim: the imitation dose curve
was a noisy plateau between ~11.8× and ~47× the position term, so the analogous
heading-rate dose is unknown and 1.0 / 8.0 bracket it; the TV dose is a first probe. The
bank-skill gate reads "≥ 0.70, i.e. within 0.03 of native32's 0.726" — note the same-runway
twin CEILING is 0.699, so the gate means "not below native32 beyond noise", never "reach the
ceiling". Naming: with the teacher at zero the nearest recipe by loss-field distance is
simple-v2, so the treatment arms resolve as `simple-v2+(hr=…)` — read that as "simple-v2 +
heading rate", not "v3 minus the teacher"; v2 IS the no-teacher loss design, and the control
arm wears the v3 name because it carries the teacher. Nothing has been trained; the campaign
is queued separately.

Suite 522 → 546 (`tests/test_supervision_terms.py`, 23, plus one in `test_run_naming.py`).
The sign convention is pinned by testing the TWO HALVES AGAINST EACH OTHER — fly a constant
positive bank, rebuild the observed target from the track that rollout flew, require the same
sign and magnitude — never against a hand-picked sign. Note what the constant-rate fixtures
canNOT catch, and say so in their docstrings: on a constant-rate turn every smoothing width
and every endpoint index gives the same answer, so those two are pinned by the jitter test
and the mask/anchor tests instead.

Reviewed 2026-09-07: behaviour clean (bit-exact at weight 0 over 22,428 leaves, sign
hand-checked on 185 real endpoints, lag indexing exact including the hooked path, 0 run names
moved); the findings above are the documentation and reading-rule fixes that followed, plus
the coarse-`dt_s` refusal and two new tests (the TV subgradient behaviour, the refusal).
### 2026-09-07 — L5.a: the FITTED imitation teacher — the fitter's checkpoint mode, `control_imitation_target`, the pre-registered arms

Latent-intent design §六 L5.a (`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md`),
branch `dev-l5`, commits `ca9b939`, `8b79623`, `7c097cc`. Code only — the fit itself is a
GPU job scheduled separately, so **no L5 number exists yet**. Suite 522 → 551.

**Why.** L1 settled that the imitation term cannot be dropped (the teacher-free dense arms
scored 2515/2603 m against native32's 1322 and brought the 2026-08 bank wiggle back), and
L0 measured that the teacher in service is bad in a quantified way: its target is the
inverse dynamics of the truth track, and flown open-loop that schedule lands **2.5–7.8 km**
from the truth it was read off (N=4 7850 m, N=8 6381, N=16 4095, N=32 2537). "Imitating the
teacher perfectly" is therefore not "flying the truth track" — the numeric form of the
control-training review's defect B. Fitted through the same differentiable rollout at the
same width, the schedule reproduces the truth to **88–433 m**.

**The fitter (`ca9b939`).** `run_ts_control_basis_oracle.py` gains a `--checkpoint` mode:
the cohort is the checkpoint's own `train,val` splits (`test` refused — a teacher fitted on
it trains on it), the width, anchor, dynamics and frame are inherited from its config, and
`--init network` (the default) seeds each flight from the checkpoint's own deterministic
forward through `batch_contract.model_forward` — a latent model decodes its prior's top-1
there, i.e. the schedule it would actually fly. `--init inverse-dynamics` keeps the width
study's seed. Pre-registered budget `--steps 400 --batch-size 1024` (L0's own convergence
evidence — the last 10 % of a 1200-step budget bought 0.4–1.2 % — puts 400 past the knee,
and the network seed starts far closer). The output is `basis_fit.json` under the schema
`ts-basis-fit-v2-teacher`, keyed by `flight_key`, carrying the N×3 dimensionless controls,
the total duration fitted over, fitADE / seedADE / best step / split, and the stamps a
consumer is checked by. The `--reference` width study is unchanged (defaults still
600 / 256); the two modes share only the batch mechanics.

**Two traps worth recording.** The table's stamped duration is compared against the
dataset's `truth_duration_s` to 1e-6 s, and the batch's own `final_time` is float32, whose
resolution at 300 s is ~3e-5 s — so the teacher path takes the float64 duration directly
and gives the fit the same value it stamps. And the dense supervision of a batch is padded
to its LONGEST flight: at KRDU the train split runs from a p50 of 183 s to 1454 s, so a
batch of 1024 drawn in split order integrates ~2.5x the flight-seconds it needs. The
cohort is therefore sorted by truth duration before batching, which also makes the table
independent of the order the splits happened to list (tested).

**The axis (`8b79623`).** `control_imitation_target ∈ {inverse-dynamics, fitted}` plus
`control_fitted_teacher_path`. The default and all four named recipes' literals stay
`inverse-dynamics` — every published simple-v* comparison was run against it. Under
`fitted` the dataset loads the table once at build time and substitutes
`reference_controls` with the row and `reference_control_weight` with **ones** (the
schedule was fitted over the whole supervised horizon, so there is no fitted tail to mask);
`train.control_imitation_mse` is untouched. The build REFUSES a table that is not this
cohort's — another schema, a free-duration partition, another airport, another width,
another anchor, a missing flight (with `covers a of b flights`), a duration off by more
than 1e-6 s — with no partial mode, which would train part of every batch on nothing while
the loss still reported an imitation number. The airport check comes first because a
foreign table fails every other check too and "covers 0 of 11082" does not name the cause.
Config refuses the incoherent combinations before any of that: fitted without a path, a
path without fitted, off the control path, with `random_train_anchor`, with
`control_imitation_loss_weight=0`, and off `control_state_objective=true-time-position` —
the only objective `loss_component_names` registers `imitation` under, which (because it
also requires the native grid and uniform durations) is what keeps a uniformly partitioned
table from claiming to supervise a learned partition. `imit-target` joins `run_naming.CONTROL_LOSS_FIELDS` and the table's path
joins `META_FIELDS` beside `closure_labels_path` (two generations of a table are two
different runs), so only the non-default values name a run: recount over every
config-bearing artifact under `4dTrajectory/outputs/*/experiments/**` — 728 files
(`history.json`, `summary.json`, `fit_evaluation.json`, `checkpoint_metadata.json`,
`config.json`, `best_config.json`) — **0 names changed**.

**The teacher is a training-time INPUT, and that is a contract, not a wording.**
`train.fit_model` opens the file once and hands the table to the two window sets it
supervises; the dataset never opens it. Every other consumer of `TrajectoryWindows`
replays a checkpoint — `evaluate-fit`, `predict --z-from-posterior`,
`approach_clustering` — and builds its window set without a teacher, so a fitted
checkpoint stays usable when the table is gone or covers a different cohort (both tested).
The table's path, sha256, width, anchor, airports and flight count go into
`checkpoint_metadata.json` and the checkpoint payload under `fitted_teacher`, deliberately
NOT inside `data_provenance`: that object is compared for EQUALITY by `evaluate-fit` and
`freeze-test` against a provenance rebuilt from the arrival manifests alone, so putting
the teacher in it would make every fitted checkpoint fail those two paths.

**The arms (`7c097cc`).** `docs/experiments/l5_fitted_teacher_arms.json`: L1_native32's
exact base plus the fitted teacher, `L5_fitted64` (the deployed dose) and `L5_fitted16` (a
better teacher may want less pull). Gates: per-stratum top-1 ADE/FDE p50 not worse than
native32, **bank skill ≥ 0.70** against the 0.17 random-flight floor and the 0.70
same-runway twin ceiling; veto on straight-in FDE p50 beyond seed noise. Pre-registered
risk: the fit minimises POSITION only, so the table may carry bank wiggle of its own — if
bank skill drops, the fix is at the FITTING end (a smoothness prior in the fit objective),
never at the training end, which would hide which of the two produced it.

### 2026-09-07 — Package audit T2: the 2026-08 oracle teacher and the nominal-law hook archived, the scene features' sequence half deleted

`4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md` §四 (`ed7fccc`,
`f701532`, `56643c4`). Three completed pieces of work leave the live package. Two are
ARCHIVED — kept in the repository under `4dTrajectory/ts_transformer/archive/`, off the
import path, each with a README naming the result documents that cite it and the commit it
was taken from; `tests/test_architecture.py` now asserts that nothing live imports `archive`
and that no `__init__.py` makes it importable. One is deleted outright.

**Suite 558 → 516.** Stored-config load sweep (all `history.json` / `summary.json` under
`4dTrajectory/outputs/*/experiments/**`), restricted to the 276 files present when T2 started:
**181 loaded at base, 181 after** — nothing stopped loading. Run-name recount over the same
276: **zero** names changed, raw or loaded. (The artifact tree grew by 15 files mid-audit —
the main tree's GPU campaign writes into the same `outputs/` — which is why the raw totals
move; the comparison is over the shared keys.)

**The 2026-08 oracle teacher (`ed7fccc`) → `archive/oracle_teacher_2026_08/`.** The eight
`control/oracle/*.py` modules, five runners (`run_ts_control_oracle.py`,
`run_ts_oracle_teacher_{audit,optimize}.py`, `run_ts_simple_teacher_paired_cv.py`, and
`plot_teacher_training.py` — the plan missed the last one; it plots the
`model_pretraining.history` block only the archived pretrainer writes), and the two test
files, renamed off pytest's `test_*.py` pattern so `run_all_tests.sh`'s whole-`4dTrajectory`
collection skips them. Five runners, FOUR of them `run_ts_*` — `plot_teacher_training.py`
never carried the prefix — so the glob T4-27's "20 → 14" target is measured on reads **20 →
16** at the repo root. It was superseded by `simple-v3`'s
in-training imitation term (`train.control_imitation_mse`); the archived
`imitation.control_imitation_loss` is a different formula and only one of the two stays alive.
`control/oracle/basis.py` → **`control/basis_fit.py`**: it is a fit, not a teacher, and the
only member with a live consumer (`run_ts_control_basis_oracle.py`, the width study's live
successor). Also deleted from the live package: `__main__ --control-teacher-schedules` with
its three companion flags and the whole `model_pretrainer` / `model_pretraining` path through
`train.py`; `dynamics/inverse.refine_piecewise_constant_schedule`; and
`physical_criteria.{physical_criteria_loss,smooth_maximum}`, orphaned by the archive.

**`transport-chart-velocity` retired.** The plan named only the `(first-order-lag, …)`
registry entry, but deleting `TransportChartVelocityBackend` necessarily takes the
`(point-mass, …)` one too, which would leave a vocabulary value with no backend at all — so
the VALUE left `CONTROL_DYNAMICS_BACKENDS` (the CLI choices derive from it) and
`run_ts_pipeline`'s filesystem-tag map. Evidence: 13 stored configs carry it, all from
2026-07-31 / 08-01 / 08-02, and **none of them loads** at base (the arc family and the
control-unit change had already refused them). `run_naming` keeps the `tcv` abbreviation on
purpose — those runs' on-disk directories are historical record — and `test_run_naming` pins
it. The lag model's remaining chart backend is the nondimensional one, which is what every
published control run uses.

**The nominal-law command hook (`f701532`) → `archive/nominal_law_hook_2026_09/`.**
`control/constraints/nominal_residual.py` + `control/guidance_laws.py`. The 2026-09-06 hooks
campaign ADOPTED the barrier as a predict-time safety layer and kept the nominal law "as an
option"; T1-9 deleted the P1.d closure tracker, its only other consumer. `barrier_filter.py`
and `gates.py` stay. **The `control_command_hook="nominal-residual"` VALUE is deliberately
NOT retired** — six stored 2026-09-06 configs carry it, all six load, and `load_checkpoint`
rebuilds a checkpoint through `TSConfig.from_dict`, so retiring it would make three
current-cohort checkpoints unloadable rather than merely unnameable (same call as T1-14's
`anchor-relative`). `CONTROL_HOOKS` is now "what a STORED config may say" and a new
`CONTROL_HOOKS_AVAILABLE` is "what a NEW run may select": `--command-hook` rejects the
archived value at the parser and `build_command_hook` refuses to construct it, pointing at
the archive and the results document. The six `control_nominal_*` gain fields ARE retired
(`RETIRED_SERIALIZED_FIELDS`) — measured first: not one stored config sets any of them away
from its default, so no run name moved. Both hook arm files
(`docs/experiments/control_hooks_{arms,v2_arms}.json`; the plan named only the first) lose
their `R_nominal_residual` arm with a `_comment` clause saying where the code and the numbers
went.

**The scene features' sequence half (`56643c4`) — deleted, not archived.**
`SceneArrays.neighbours` (`[N_MAX, L, 6]`), `neighbour_mask` and `_series_on_grid` were built
for the L4 explainability gate; the gate did not pass and the one consumer
(`run_ts_scene_explainability.py`) reads only the entity and scalar halves. `scene_arrays`
loses its `seq_len` / `dt_s` parameters and the "lookback longer than the scene window"
guard, which existed only to keep the resampling grid inside the window and can no longer
fire. `scene/__init__.py` and the module docstring say so, so adding a sequence back is a
deliberate re-derivation against a model that consumes one.

**Documents.** One-line banners only, no rewrites: `2026-08-02_oracle_teacher_experiment`,
`2026-08-16_control_simple_v1_development`, `2026-07-30_direct_control_oracle`,
`2026-08-24_ksjc_result_labels_explained`, `control_parameter_prediction` (the whole call
graph it draws is the archived chain), `2026-09-06_control_hooks_results`,
`2026-09-05_control_constraint_design`, plus path notes on the two 2026-09-07 docs that cite
`control/oracle/basis.py`. The live `CLAUDE.md`, `README.md`, `ENGINEERING_NOTES.md`,
`OPEN_ITEMS.md` and `barrier_filter.py` lines that named archived modules, flags, runners or
the retired backend are fixed; `CLAUDE.md`'s Layout section states the `archive/` convention,
and both `ENGINEERING_NOTES` and `OPEN_ITEMS` now say that reviving the combined
lateral-barrier + vertical-nominal hook is a deliberate un-archive, not an import.
### 2026-09-07 — observed speed gate: measured load factor + METAR headwind (report v8)

Branch `dev-observed-load-factor-metar`; plan `evaluation/docs/2026-09-07_observed_speed_gate_plan.zh.md`.
v7 anchored the gate's lower bound on the crossing load factor but declared 1 g for
observed records, so on the ADS-B baseline — the ground truth the method is judged
against — it changed nothing. Two changes make the baseline judged on what it flew:

- **The observed load factor is measured, not assumed.** `arrival._observed_load_factor`
  fits ψ and γ over the final 20 s of measured track before the crossing and inverts the
  point-mass rotational equations (`aircraft/kinematics.load_factor_from_rates`, the
  same inversion `ts_transformer/flyability.py` uses per sample); the window's facts are
  on the row. Measured over 42,732 observed rows: median n 1.002–1.005, p99 ≤ 1.024,
  max 1.122 (`THRESHOLD_SPEED_GATE.md` §3.5) — the 1-g assumption holds, and the 211
  verdicts that moved are flights within a knot of the floor.
- **The ground-speed proxy is corrected by the field's wind.** OpenSky carries no
  airspeed, so `trajectory_data_process/metar/fetch_iem_asos.py` fetches the airport's
  ASOS/METAR archive from IEM into `data/metar/<ICAO>/` (provenance beside each file) and
  `evaluation/wind.py` joins the report nearest each landing (≤ 30 min, direction not
  variable): airspeed estimate = ground speed + W·cos(direction − runway course), judged
  under `…_metar_airspeed_estimate` with a declared ±5 kt uncertainty; `speed_marginal`
  counts estimate-judged rows within it of a bound. No usable report ⇒ the proxy, said
  per row and counted per batch. The join is at read time: no harvest artifact changes.
  Measured effect and coverage: `BASELINE_SPEED_GATE_RESULTS.md` §9.
- Schema v7 → v8 in all four homes; the ts seam reads (v6, v7, v8); the frontend shows
  v6/v7 as prior. `--metar-root` on the evaluation CLIs, default `data/metar`.

### 2026-09-07 — evaluation review: load-factor speed gate (v7), LNAV/VNAV runways wired, one read path

Branch `dev-evaluation-review-fixes`; plan + status table
`evaluation/docs/2026-09-07_review_fix_plan.zh.md`. Four milestones, each opus-reviewed.

- **Speed gate anchors on the crossing load factor (report schema v6 → v7).** The gate
  used the 1-g stall speed regardless of what the aircraft was doing; the model's own
  dynamics fly `n` and its stall drag already uses `n·m·g`. Now `lower = 1.23·Vs1g·√max(n,1)`,
  `upper = 1.23·Vs1g + 20 kt` (an energy criterion, stays at 1 g); `n` =
  `controls[-1].load_factor` (the control active over the final rollout step) or a declared
  1 g on records without controls (`crossing_load_factor_source`). Measured on the optimizer
  batches: median n 1.01, tail to 1.37 (+17 % stall speed); scaling both edges would have
  flipped 4–227 verdicts per batch, all records piled against the upper edge, the lower-only
  rule 0–20 (`THRESHOLD_SPEED_GATE.md` §3.5). `aircraft.aero_params.stall_speed_ms` gained
  `load_factor=1.0`; the optimizer floor is unchanged. Four version homes moved together; the
  ts seam imports `READABLE_REPORT_SCHEMA_VERSIONS = (v6, v7)` since it reads only
  `lateral_result`, and the frontend shows v6 as "graded at 1 g" (`isPriorSpeedGateReport`).
- **Every v6 report contradicted itself**: `methodology.observed_crossing_ground_speed.use`
  still said the ground speed was "never composed into any verdict" beside the
  `terminal_speed` block that grades it (owner decision 2026-08-24); the HTML report note said
  "observed records are never speed-graded" and its V column showed "—" for observed rows;
  the frontend tooltips said "audit only, not graded". All say the same thing now, and a test
  pins the two methodology blocks to each other.
- **LNAV/VNAV runways are evaluated.** KRDU 32 (1,604 arrivals) and KSMF 35R (259) have no
  LPV Path Point but publish LNAV/VNAV minima: `harvest/cifp.read_approach_verticals` decodes
  the RNAV (GPS) procedure's runway leg (3.50°/470 ft − 425 ft = 45 ft; 3.00°/64 ft), pinned
  per airport against the Path Points (all 23 LPV runways within 0.8 ft). `Runway` gained
  `tch_source` / `baro_vnav_minima` (not in the frame fingerprint — all 26 frame fingerprints
  and the 23 LPV context fingerprints are pinned unchanged by
  `test_approach_verticals.py`); `assessment_for_runway` derives `baro_vnav_approved` from
  the runway, no caller flag. KRDU 14 (13 arrivals, no procedure) stays excluded. The
  previously inert `rnp_apch_lnav_vnav_baro` branch is now reachable. **Nothing on disk
  changed**: re-running the harvest grows the arrivals rosters (KRDU +1,617, KSMF +259) and
  every ts split — do it between campaigns.
- **One read path.** `roster_context_keys` no longer falls back to materializing the whole
  batch (~1 MB/flight on a 16 GB box) when a roster row lacks `arr_airport`; it raises.
  `visualize.build_payload` is the streaming builder (the list variant and
  `contexts_from_args` are gone); records validate `arr_airport`, `runway`,
  `hae_minus_msl_m`, `landing_aero` shape and a strictly increasing `t` at the boundary
  (followups #7–#10 closed); the reference guard walks `measured_states` like the comparison.
- **Found, not fixed here**: all 70,267 optimizer records on disk predate `landing_aero`
  and grade speed-indeterminate (three-gate pass = 0) — `4dTrajectory/optimization/
  backfill_landing_aero.py` adds the block through the producer's own chain without
  re-solving; run it and regenerate the reports when the GPU campaign is over. The root
  `CLAUDE.md` open item claiming "no SOLVES on disk" was stale and is corrected.
- Tests: fixtures import the constants they used to restate; negative pins on retired
  wording replaced by positive pins on the current policy; `validate_event` cases moved to
  `final_approach/tests/test_event_contract.py`; new coverage for the streaming payload,
  roster errors, composite precedence, interpolated-crossing mass, the empty batch. Docs:
  `BUG_FIX_GUIDE.md` and the v4 `FAILURE_REASON_ANALYSIS.html` archived under
  `evaluation/docs/archive/`; `THRESHOLD_SPEED_GATE.md` §2 working notes replaced by §3.5.

### 2026-09-07 — Package audit T1-10 / T1-11 / T1-13 / T1-14: the horizon curriculum, the arc-length-geometry objective family and the dual terminal clock deleted

`4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md` §三, rows 10, 11, 13
and 14 (`c5c3a32`, `09e4420`, `2692e61`). T1 is complete: the whole 2026-07/08 generation of
abolished control design is out of live code.

**T1-10 — the horizon curriculum (`c5c3a32`).** `control_horizon_curriculum_s` defaulted to
`()`, every named recipe pinned it to `()`, and no arm file ever set it. Its own validation
made it unreachable in practice besides: it required `control_state_objective=arc-length-geometry`,
the family retired the same day. Deleted: `control/training/curriculum.py`, the `training_stage`
parameter of eight loss functions and of `_evaluate_validation_airport`, the stage-view crop,
`EpochResult.training_stage`, and the whole staged branch of `fit_model` (scheduler reset,
`curriculum-prefix-objective` selection, stage prints). `close_duration_prefix` is KEPT and moved
to its only consumer, `control/loss/fixed_dt.py`; with the prefix mask gone it takes
`(durations, total_duration)` and its four unreachable opening guards go — the `allclose` on the
reconstructed total is the check that ever mattered. Two knock-ons: `include_deployable_replay`
could no longer be False (flag and both branches deleted), and `segment_valid` had no non-None
source, so the pass-through in `control/dynamics/{rollout,backends}.py` went too (the
`aerodynamic_model` kernels below keep the capability, with their own defaults and tests).
The oracle teacher's short-to-long refinement existed only to drive the crop, so
`teacher_optimization_stages` is gone and `optimize_teacher_controls` takes a flat `steps`;
`run_ts_oracle_teacher_optimize.py` and `run_ts_simple_teacher_paired_cv.py` swap
`--prefix-steps 30` / `--full-steps 150` for one `--steps 240`, the same budget.

**T1-11 + T1-13 — the arc-length-geometry objective family and the dual terminal clock
(`09e4420`).** No arm file has used `arc-length-geometry` since 2026-08-02; README /
OPEN_ITEMS / ENGINEERING_NOTES never mention it; its only setters were three 2026-08 teacher
runners whose checkpoints `load_checkpoint` already refuses (the 2026-08-18 control-unit
change). The dual terminal clock rode with it — its `state-supervision` strategy was literally
`return result`, and its two other values were reachable only under the arc objective. Deleted:
`components._arc_length_geometry_objective` with its four helper tables and the
`runway_heading_rad` argument, `control/loss/terminal_clock.py` (its deletion was already
staged when a concurrent docs commit swept it in, so it landed under `1c83d7c`, not
`09e4420`), the torch halves of `arc_length_geometry.py` and `terminal_state_loss.py`,
`train._arc_length_geometry_validation_selection`, and seventeen config fields — sixteen
`control_geometry_*` / `control_arc_*` / `control_terminal_*` plus the whole
`control_gradient_clip_policy` AXIS (a one-member vocabulary is the behaviour, not a choice;
`control_gradient_clip_norm` stays, and `clip_gradients_by_global_norm` now caps once and
returns the applied coefficient).

**What the arc deletion did NOT touch, deliberately.** The plan's row 11 asked for
`fixed_anchor_validation.py:110-315` AND for keeping the three `*_metrics_numpy` "the
validation uses every epoch"; those cannot both hold. `fixed_anchor_arc_length_geometry_metrics`
is a DIAGNOSTIC that `fixed_anchor_common_grid_metrics` computes for every control run
whatever its objective, and `train._common_grid_validation_details` plus
`build_multiflight_capacity_report.py` read ~30 of its `arc_length_*` keys today — so the
block stays, and its three shape parameters (position-end weight 4.0, terminal emphases
3.0 / 5.0) are frozen as module constants, which is what keeps those keys comparable across
the whole artifact history.

**A naming tie-break the deletion widened.** With the sixteen arc/terminal fields out of
`run_naming.CONTROL_LOSS_FIELDS`, a custom config can now tie a named recipe at ZERO
loss-field diffs (the v1 / v1-lag tie itself pre-existed: `control_dynamics_model` was never a
loss field; the deletion made it reach every point-mass run) — and the documented "a later
recipe wins ties" then named every point-mass `simple-v1` run `simple-v1-lag`, because those
two recipes are the same loss design and
differ only in the flight model. The nearest-recipe rank gained a second key: fewest edits
among the NON-loss fields the recipe also freezes.

**T1-14 — `state_position_reference="anchor-relative"` KEPT and marked vetoed (`2692e61`).**
It was vetoed by the 2026-09-03 state-v2 campaign's own pre-registered rule, but a
current-cohort artifact is stored under it (`state_v2_20260903/A_anchor_relative`), so
deleting the value would stop that run's config loading. The warning now lives at the
constant in `config.py` and in the package `CLAUDE.md` defaults table.

**Contract.** All nineteen removed fields go into `RETIRED_SERIALIZED_FIELDS` and out of
`REQUIRED_SERIALIZED_CONTROL_FIELDS`, so every stored config still loads through
`TSConfig.from_dict` — the lenient direction. 2026-08 teacher checkpoints stop loading; they
were already refused by the 2026-08-18 control-unit change.

**Stored-run-name recount** (`history.json` + `summary.json` under
`4dTrajectory/outputs/*/experiments/**`, read-only, `run_display_name` + `run_slug` before and
after each commit; the config count rises across the three commits — 261 → 263 → 276 — because
the L2 campaign kept writing in the main tree while this worktree was edited, and one
pre-existing `summary.json` carries no config dict throughout):
 - **T1-10: 20 names moved**, all the 2026-07-31/08-01/08-02 POOLED and KSJC control
   generation, which genuinely ran `60,120,240 s × 10 epochs`. Their `curriculum=60/120/240`
   meta item drops out, so a name ends one item earlier or shows one fewer `+N more`.
 - **T1-11 + T1-13: 24 names moved**, the same generation and one mechanism: without the
   arc/terminal loss fields those runs are within two edits of simple-v1, so
   `custom(obj=…, grid=…, clock=…, duration-grad=off)` (and one `custom-56aee7f8` content
   hash) becomes the spelled-out `simple-v1+(obj=…, grid=…)`, with `grad-clip=20` taking the
   meta slot the curriculum vacated. One further name was already wrong and is now right: a
   point-mass run that read `simple-v1-lag` now reads `simple-v1`.
 - **T1-14: 0 names moved** (comment-only).
Both families are the documented behaviour of the grammar ("names describe a config relative
to TODAY'S defaults"). No on-disk run directory is renamed, the cached `display_name` in
`outputs/*/INDEX.md` and `index.json` is a historical snapshot and was not touched, and no
moved name is quoted in any doc, arm file or source.

**Docs that quote the retired code got a one-line banner, never a rewrite**:
`2026-08-02_dual_clock_terminal_ablation.zh.md`, `2026-08-01_arc_length_geometry_loss_experiments.zh.md`,
`2026-07-31_deployable_control_training_optimizations.zh.md`, `2026-09-07_control_training_review.zh.md`
and `control_parameter_prediction.zh.md`.

Suite after each commit: 586 / 556 / 556 passed. `train --help`, `predict --help`,
`run_ts_pipeline.py --help` and the three teacher runners' `--help` all work.

### 2026-09-07 — Package audit T1-9 / T1-12: the closure tracker and the regularization axis deleted

`4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md` §三, rows 9 and 12
(`0702669`, `0898291`). Two generations of abolished design removed from live code.

**T1-9 — the closure tracker (`0702669`).** The P1.d delivery form — a drawn closure
reference flown by the point-mass rollout under a command hook — was abolished by the
latent-intent design (§四) with its review BLOCKER unfixed: the tracker's nearest-node
search snaps onto a self-approaching reference and jumps legs on 8 of 1404 via-Dubins
flights (endpoints 6–19 km off), which are four of the five largest "tracking gains", so
the honest tracking cost is **+10.5 m of ADE, not +9**. Deleted:
`control/constraints/closure_tracking.py`, `tests/test_closure_tracking.py`,
`docs/experiments/closure_p1d_arms.json`, `forecast.tracking_config` /
`track_closure_forecasts` / `Forecast.closure_tracked` and the `track`/`device` parameters
that fed them, `predict --closure-track`, and `source.closureTracked` in the record. The
closure OUTPUT is untouched and stays as a comparison arm (`closure_output.py`,
`prediction_output=closure`, `--closure-from-labels`, the labels/template chain). No
TSConfig field moved, so no serialization or run-name change. Suite 598 → 596 (the two
tracker tests).

**T1-12 — effort / smoothness regularization (`0898291`).** Every named recipe pinned both
weights to 0.0, no arm file had set them since 2026-08, and the term was computed on every
control batch and then multiplied by zero. Deleted `control/loss/regularization.py`, the
`control_effort_loss_weight` / `control_smoothness_loss_weight` fields with their raises,
recipe pins and `control_recipe()` entries, the whole block in
`train.control_prediction_loss_terms`, the two `ControlLossTerms` fields and their share of
`.total`, the two `CONTROL_LOSS_COMPONENT_NAMES` entries and extras, both CLI flags, and
the `run_ts_pipeline` / `run_ts_control_oracle` / `run_ts_control_capacity_ceiling`
plumbing. Both fields joined `RETIRED_SERIALIZED_FIELDS` so older checkpoints still load
through `from_dict`; neither was in `REQUIRED_SERIALIZED_CONTROL_FIELDS`.

**Run-name recount, 261 stored configs (`history.json` + `summary.json` under
`4dTrajectory/outputs/*/experiments/**`, read only): 23 names moved** — and unlike T0's
zero, these are the real thing, in two families. Ten are the 2026-07-29 / 08-01 POOLED
effort–smoothness SWEEPS (`stage_c_effort`, `stage_c_smoothness`,
`control_output/{effort,smoothness}_weight`), which genuinely set the weights:
`custom(effort=0.0001)` → `custom`. So these two fields retire on **weaker grounds than
`control_hook_gate` / `control_dense_state_loss_weight`** — the stored value could change
THOSE runs — and the `RETIRED_SERIALIZED_FIELDS` comment now says so instead of claiming a
blanket "could not change an answer". The other thirteen are 2026-08-01/02
arc-length-geometry runs carrying the old module defaults (1e-3 / 1e-2): losing two
residual diffs drops them under `_MAX_LISTED_DIFFS`, so the loss design UN-collapses from
the content hash `custom-1c5a2429` / `custom-3788b8a3` into the spelled-out
`custom(obj=arc-length-geometry, grid=fixed-dt, clock=observed, …)`. Both families are the
grammar's documented behaviour (names describe a config against TODAY'S defaults). No
on-disk run directory was renamed; the cached `display_name` in `outputs/*/INDEX.md` and
`index.json` is a historical snapshot and was left alone; neither moved name is quoted in
any doc, arm file or source.

Docs, one line each and no dated doc rewritten: the package `CLAUDE.md` closure paragraph
and Layout line, `docs/OPEN_ITEMS.md` (both tracker mentions),
`docs/2026-09-06_closure_p1d_tracking_results.zh.md` (its supersession banner now names
the deleted symbols and says its §一/§五 no longer execute) and
`docs/control_parameter_prediction.zh.md` (a second 2026-09-07 note in its existing
staleness banner).

### 2026-09-07 — Package audit T0: eight zero-risk cleanups, reviewed

`4dTrajectory/ts_transformer/docs/2026-09-07_package_audit_plan.zh.md` §二. One control
model class (`UniformDurationControlOutputModel` + `control/duration.py` folded into
`ControlOutputModel`; `UniformDurationControlHead` beside `ControlOutputHead`); two config
axes that could never bind retired (`control_hook_gate`, `control_dense_state_loss_weight`
— `RETIRED_SERIALIZED_FIELDS`, dropped by `from_dict` so stored configs keep loading);
one strata definition (`approach_difficulty.STRATUM_*` + `strata_masks`; the stratified
readout's own copy lacked `& ~established` on the vectored mask); six tests that could not
fail removed; the recipes spelled as literals; one `io_utils`; one `metrics.signed_spread`.
Run names recomputed over all 276 stored configs: 0 changed.

Review findings fixed in the same commit series: three `docs/` scripts rebuilt a STORED
config with `TSConfig(**summary["config"])`, bypassing `from_dict` — after the retirement
0 of the 111 on-disk summaries loaded there (`p1_closure_oracle.py`, which the closure arm
names as its label producer, included); `run_ts_runway_hypotheses.py` re-keyed its strata
without a schema bump (now `v3-stratum-labels`); the stratified test's fixture could not
exercise the mask fix (ten vectored rows are now established at the anchor and the
stratum count asserts it); the L2 arms' gate named the arm the comment reports at 2515 m
(now L1_native32, the base itself); a real-artifact checkpoint canary is back in skipif
form (`L1_native32/checkpoint.pt`, which stores both retired fields); the recipe-literal
guard is a `DEFAULT_*` regex over both recipe functions rather than an eight-name list.
`score_control_arms._tortuosity` and `approach_difficulty.route_tortuosity` are two
estimators (chord to the last observed row vs range to the threshold), not one statistic
at two cuts — the comment says so now; unifying them would move the published bank floor.
### 2026-09-06 — Disk reclaim (29.7 GB) and three stale roster facts corrected

The root filesystem hit 100 % (2.3 GB free). Reclaimed **29.7 GB** with no research
artifact touched: `~/.cache/opensky` (18 GB, 3133 raw parquet, untouched since
2026-08-14 — pyopensky's download cache, superseded by the materialised `tracks/`, and
its own config already says `purge = 90 days`), browser/vcpkg/bazelisk caches (2 GB),
`studys/**/__pycache__` (43 MB), and two superseded harvest generations:
`harvest-may-2026` (4.5 GB — already merged into the live harvest, `provenance.merge`
completed 2026-08-12) and `harvest-pre-source-timed-20260815` (5.0 GB — the pre-fix
backup of the current `harvest-tracks-v2-source-timing`). Checked first that the merge's
hard links no longer shared inodes, so both deletions were real space rather than
link-count decrements. `conda clean --all` reclaims nothing (already clean).
Still available and NOT taken: `journalctl --vacuum-size=200M` (3.9 GB, needs sudo);
`public/data/airports/trajectories.czml` (2.4 GB, one command to regenerate); the
21 GB of per-flight `*_states.json`/`*_eval.json` under the 116 `*_pred*` dirs — whose
`summary.json`/`evaluation_report.json` (1.2 GB) carry every published number, but which
a running experiment was reading at the time.

Verifying the survivors surfaced three documented facts that were wrong, all corrected:

- **The v5 arrivals roster is 42,650, not 42,725.** Corroborated independently by the five
  `arrivals/manifest.json` (`counts.included`, mtime 2026-08-24) and the ten
  `.selection.json` (`available`, 2026-08-23), which agree exactly: KRDU 14,435 /
  KSJC 11,082 / KSTL 8,767 / KSMF 4,219 / KMSY 4,147. 42,725 was the real count of the
  **2026-08-19** roster generation; the root `CLAUDE.md` claimed it was "verified on disk
  2026-09-03", i.e. the number was copied forward rather than re-measured.
- **`flight_scenarios/outputs` is NOT empty.** Both the root `CLAUDE.md` Open Items and
  `docs/open-items.md` said it was, and concluded "the run is from scratch". All ten
  `*_scenarios.json` + `.selection.json` are on disk (92 MB, built 2026-08-23). What is
  actually missing is the *solve* records under `4dTrajectory/outputs/<ICAO>`, so
  `--skip-optimize` still finds nothing — the conclusion held, the reason did not. Whether
  the 08-23 scenarios are reusable is left UNVERIFIED on purpose (the manifests were
  rewritten one day later, though their counts still match): the runner's prepared-input
  signature check decides.
- **The capped batch is 23,429 flights / 70,287 solves** (`runway`), not 23,453 / 70,359 —
  read off the selections rather than restated. `fitted_adsb` selects the same flights and
  then drops the 20 `UnusableFittedApproach` ones (23,409 / 70,227), which confirms the
  "both prepared datasets select the same flights" invariant rather than breaking it.

The `35 of 42,725 (0.08 %)` unfittable measurement in `flight_scenarios/CLAUDE.md` was
NOT rebased — changing a denominator under an unremeasured numerator would manufacture a
result — it is now attributed to its roster generation, with the current capped-selection
figure (20 dropped, same per-airport shape) beside it. The 2026-08-19 CHANGELOG entries
keep 42,725: they are dated history and correct as written.

Also found, not deleted: `outputs/harvest/.KSJC-reclassify-akrwpor_`, an orphaned
reclassify staging copy (170 MB, 8351 files, no `manifest.json` so nothing can read it,
every file separately inoded and duplicated in the live tree, timestamped 8 minutes
before the reclassify completed).

### 2026-09-06 — Snapshot before a redesign: the tracker review, the P2 data plane as WIP

The opus review of the closure tracker (`control/constraints/closure_tracking.py`)
found a BLOCKER — the nearest-node search over the whole reference lets a via-Dubins
path that passes within metres of itself snap the tracker to the wrong leg (8 of 1404
val flights, endpoints 6–19 km off; four of the five largest ADE "improvements" from
tracking in `2026-09-06_closure_p1d_tracking_results.zh.md` are those flights, the
honest pooled cost being +10.5 m) — and five SHOULD-FIXes (the vertical law's sign is
untested; the stall floor ignores the commanded load factor; ISA density uses chart
height; the height-profile anchor pinning of `4ecfb69` is outside the label contract;
the gain docstring). Nothing was changed: the user decided to redo the plan. The P2 data
plane built meanwhile is committed as WIP (`045c233`: `trajectory_data_process/scene_index.py`,
`flight_scenarios/scene_context.py` with a stricter leakage line — an airborne
neighbour's landing time AND eventual runway are future, kept in `future_label` only —
and `4dTrajectory/ts_transformer/scene/features.py`; 13 tests, no review, no
explainability measurement). The design doc's §〇.1 carries the full snapshot.

### 2026-09-06 — Closure decoder, P1.d: the drawn reference flown by the point-mass rollout — ≤ 100 m of ADE for 92 % flyability

Option (b) of the design's P1.d. `control/constraints/closure_tracking.py` (new): a
command hook that steers the rollout's own state toward the closure reference at every
segment start — L1 lateral guidance on the cross-track / heading error to the reference's
local course plus its curvature as a feed-forward bank, the glidepath law on the height
error with the reference's local slope (capped 0.3), and a PI speed hold toward the
reference speed (integrator from the anchor's implied thrust, proportional 0.3, the
reference acceleration ±0.5 m/s² as feed-forward) with the along-track error folded into
the target and a 1.15 × stall floor. `forecast.track_closure_forecasts` flies every
decision vector under it on the first-order-lag dynamics (`tracking_config` derives that
control config; the only backends that run hooks) for the reference's own duration in
64 held segments; `predict --closure-track`; records carry the controls flown and
`source.closureTracked`. Also fixed: the drawn height profile now starts at the anchor's
height (the first knot is a least-squares value, and the step it drew made the vertical
law ask for a −86° path angle). KRDU val, 1404 flights, pooled / vectored ADE:
C_pred 996 → 1005 / 2197 → 2222, C_truth_intent 595 → 611 / 1235 → 1244, C_oracle 183 →
256 / 458 → 554 m; fully flyable (clean polar) 22 → 92 / 90 / 88 % (observed 98 %, the
control baseline 0.1 %), per sample 99.9 %. The delivery form is closure + tracking
(`docs/2026-09-06_closure_p1d_tracking_results.zh.md`); the tracker's gains stay module
constants for now (a P1.d measurement setting, not yet a config field).

### 2026-09-06 — Closure decoder, P1.c-3: the campaign — both gates pass, the output side was the bottleneck

`4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905/` (arms
`docs/experiments/closure_p1c_arms.json` + `closure_p1c_oracle_arms.json`; report
`4dTrajectory/ts_transformer/docs/2026-09-06_closure_p1c_results.zh.md`). KRDU val, 1404
flights, closed truth, vectored stratum ADE / chamfer p50: simple-v3 2858 / 942; the Phase 0
truth-intent control arm 2011 / 847; **C_pred (closure, ego history only) 2197 / 722;
C_truth_intent (closure + truth (d_join, T) inputs) 1235 / 492; C_oracle (drawn from the
labels) 455 / 180**. Pooled ADE 1333 → 996 (C_pred) → 595 (C_truth_intent); straight-in
469 → 310. Gates (vectored ADE < 1.5 km and chamfer < 500 m with the truth intent; not
worse than the baseline without it) both pass. Training is 2–2.5 s per epoch (a pure
regression, no rollout). One decoder rule was added mid-campaign: a predicted via inside one
turn radius of the anchor is dropped for the plain CSC (`csc-via-at-anchor`) — the first
C_pred arm's straight-in stratum was 6260 m because a label via AT the anchor, predicted a
few tens of metres off, made the Dubins path a full circle (827 of 837 over-long paths);
the pre-rule readout is kept under `attempt1_readouts/`. The runner now substitutes
`{airport}` in a predict-only arm's `predict_args`. Flyability, read as a delta: closure
paths are 22 % fully flyable under the clean polar against the control baseline's 0.1 %
(59k stall samples) and the observed 98 %; per sample 99.8 % = the observed, the few
violations per flight being bank jumps at the CSC junctions and thrust jumps at the knots
(P1.d: a clothoid / bank-rate transition plus profile smoothing, or a post-hoc rollout). P2's acceptance becomes the 960 m of vectored
ADE between C_pred and C_truth_intent.

### 2026-09-05 — Closure decoder, P1.c-1/2: the `closure` prediction output and its labels

The third `prediction_output`. `closure_output.py` (new): `ClosurePrediction` (a 14-number
decision vector — join distance, via pose in runway axes as (d, xt, cos Δψ, sin Δψ), K=4
slowness knots whose integral is the duration, K=4 height knots with the threshold pinned
to 0), `ClosureOutputModel` (the backbone's tokens → MLP → the bounded vector: sigmoid /
tanh at the family's own reach, a softened unit vector for the heading), `fit_labels` (the
per-flight label from the truth: the canonical F3 geometry, both profile widths with
residuals, `valid` = canonical and within the 1 km cap, the difficulty covariates),
`closure_loss_components` (L1 per group over the valid flights; state = geometry,
final_time = slowness in seconds at a 60 s scale — measured at init the three groups then
sit within a factor of two — kinematic = height, terminal = 0), `reconstruct` /
`replay_batch` (the drawn trajectory: via-Dubins, else the plain CSC, else the straight
line, each recorded as `source.closureConstruction`). Wired through `config.py`
(`PREDICTION_CLOSURE`, the six `closure_*` fields, the locks: ENU chart, normalized horizon,
objective checkpoint selection, no random anchors, knot widths ∈ (4, 8)), `train.py`
(`closure-v1-slowness4-height4` target contract, loss registry, a replay branch that keeps
the reconstruction's exact velocities; `LossComponents` moved to `batch_contract` so the
loss side never imports the loop), `models.py`, `dataset.py` (the labels as the batch
context — a file covering no flight is refused, the covered share is printed),
`batching.py`, `forecast.py` (`closure_forecast`, `forecast_closure_from_labels` = the
oracle arm, `source.closureFromLabels`), `run_naming.py` (`closed-form`, `closure-v1`, the
labels file name in the run name), `anchor_eligibility.py`, `__main__.py`
(`--closure-labels`, `predict --closure-from-labels`), `export.py`; scipy joins
`requirements.txt`. `docs/p1_closure_oracle.py labels` writes a cohort's labels
(`closure-labels-v2`; KRDU: 9720 flights, 96.8 % valid). `tests/test_closure_output.py`
(6 tests) runs a whole train → checkpoint → forecast → export → evaluate chain. Review
(opus): the wiring script had applied every hunk 2–3× (the labels file was parsed twice
per dataset) — restored and re-applied once; the other findings (oracle provenance,
zero-coverage training to a perfect loss, unbounded head, 20× loss-scale gap, schema
bump, run-name identity) are the changes above. Arms for the campaign:
`docs/experiments/closure_p1c_arms.json` (C_pred, C_truth_intent, C_oracle).

### 2026-09-05 — Closure decoder, P1.a / P1.b: the path family and the profile parametrisation, measured before any network

Direction C of the scene design (fix the output side before the scene encoder) was
chosen. `4dTrajectory/ts_transformer/closure_geometry.py` (new) holds the closed-form
approach families the closure decoder will draw from — the Phase 0 rule template
(moved out of `docs/phase0_intent_diagnostics.py`, which now imports it), a
downwind-then-Dubins family and a via-pose Dubins family — with the vertical profile,
the truth / naive timings and per-flight fits; `closure_profile.py` (new) parametrises
speed as piecewise-linear SLOWNESS over progress (time is linear in the knots, the fit a
bounded linear least squares) and height as knots; `docs/p1_closure_oracle.py
geometry|speed` runs the two oracles on the reference arm's flights (post-anchor
supervision truth). Two opus review rounds shaped the fits: the objective is the
order-preserving arc-aligned horizontal error (chamfer let detours through), F3 is
multi-start and seeded from F1/F2 (the fitted residuals nest), and the labels are
canonicalised — the join at the localizer entry (a straight along the localizer made
d_join unidentifiable), the via as the earliest pose along the fitted path that still
reproduces it (identifiable: two best starts within 3 m median) — with `canonical:
False` kept on the 4 % of looping fits. Results at KRDU (497 vectored flights, truth
timing): F0 1688 m → F1 1301 → F2 794 → **F3 chamfer 180 / Fréchet 1179 / ADE 510 m,
gate passed**; F2 does not contain the trombone (its downwind runs along the anchor
heading). On the truth path: naive 1308 (Phase 0 reproduced) → naive × truth duration
622 → slowness knots K=2 166 / K=4 97 m; height knots K=4 30 m; combined K=4 110 m.
P1.c's decision vector is therefore ≈14 numbers (d_join, the via in runway axes with
its heading relative to the course, K=4 slowness with the duration as their integral,
K=4 height with the terminal knot pinned to the threshold — a third review round found
the unpinned fit landing up to 33 m below it on 28 % of flights, and the chart-frame via
splitting one decision into four per-runway label distributions). Design doc §〇 / §五
P1 updated with the plan, results and gates.

### 2026-09-05 — Geometric readout (P0 of the scene design): chamfer / Fréchet / arc-aligned ADE next to ADE/FDE

`4dTrajectory/ts_transformer/geometric_metrics.py` (new; `tests/test_geometric_metrics.py`,
23 tests) is the single source for the time-free path metrics, and both standard readouts
(`docs/compare_constraint_arms.py`, `docs/compare_frame_arms.py`) now print them in every
stratum table next to ADE/FDE: chamfer p50 (symmetric nearest point, horizontal, 100 m
resampling — moved out of `phase0_intent_diagnostics.py`, whose chamfer now comes from
here), discrete Fréchet p50 (order-preserving, anti-diagonal DP on the same resampling,
verified against the textbook recursion), arc-aligned ADE (3D at the same fraction of each
path's own horizontal arc length, 64 fractions excluding the anchor), the predicted/true
length ratio, |Δdur| p50 and the along-path lag p50 (predicted minus true time at the same
arc fraction; Δdur on the exported states' clock, so a horizon-capped flight reads the cap
here and the duration head in `final_time_error_s`). Truth (`--geometry-truth`): `closed`
(default) = the post-anchor observed rows closed to the threshold at `true_final_time_s`
— the exported states carry no fitted tail, and the observed track stops a median 379 m /
6 s short at KRDU (82 m / 2 s at KSJC), so both metric families now end at the same point
and time; `observed` = the rows as exported, which reproduces the Phase 0 diagnostics'
chamfer within 2 m (the resampler now includes the endpoint: 942 / 801 / 791 / 850 →
943 / 802 / 793 / 849). The readout's first line states the truth, the closure and every
parameter. **Review finding (opus, 2026-09-05): the STATE output's exported polyline is a
node-scale saw-tooth** (heading reversals > 90° at a median 50 % of its 2 s nodes, every
flight above 5 %; control rollouts and the observed truth at 0, truth max 0.008; raw length
ratio ≈ 2 against 1.01), which doubles anything parametrised by its own arc. Each flight
therefore carries `reversal_share` and enters the arc family only at ≤ 5 %; a block
aggregates arc-ADE / lag over those flights, writes `arc_family_share`, prints the share
beside the value when it is below 100 % and `n/a` below 95 % (null in the JSON, so a two-flight mean is never quotable) — no smoothing (a silent
approximation), and no length-ratio gate (the second review showed a block whose median
ratio sits in a band still averages in its out-of-band flights, +398 % on one state
stratum, and on control arms a ratio far from 1 is a real error, not an artifact). On
state campaigns the arc family is therefore mostly `n/a`; chamfer / Fréchet / Δdur carry
the geometry there. Also fixed while there: the constraint readout's `|xt| p95` header
broke its markdown table since the campaign it was written for.
Backfilled: `scene_phase0_20260905/readout_geometry.*` (four arms; results doc §三b) and
`control_hooks_v2_20260906/readout_geometry.*` at KRDU + KSJC (hook results doc, new
section). What the geometry says: the truth join point is worth 15–20 % of vectored geometry
(chamfer 942 → 795, Fréchet 3100 → 2468, arc-ADE 2175 → 1775), the truth-duration arm adds
none of it (its 2356 → 2011 ADE gain is timing alone — the case the revised success
criteria veto), Fréchet ≈ 3× chamfer (the paths are sequenced differently, not offset), and
the v2 soft barrier is the only hook that improves vectored geometry too (KRDU chamfer
942 → 886 on 81 % of flights; both trained-through arms lose 13–16 % on Fréchet / arc-ADE).
Design doc §〇 / §五 P0 updated; `CLAUDE.md` gained the "read both families" rule.

### 2026-09-05 — Phase 0 result (KRDU): the join point is worth −17 % vectored ADE, the gate was mis-sized

`4dTrajectory/ts_transformer/docs/2026-09-05_scene_phase0_results.zh.md`; campaign
`scene_phase0_20260905` (O_join_lead, O_join; simple-v3, paired with
`control_procedure_20260905/A_control_v3`). Vectored ADE 2858 → 2364 / 2356 m, duration
error 39 → 20 / 22 s, straight-in 469 → 442 / 458 m; the lead ETA adds nothing beyond the
join point. The pre-registered gate (< 1.5 km) was unreachable for a join-only oracle under
the time-aligned metric: on the same 497 flights the TRUTH path with a naive speed profile
scores 1.3 km and a trombone from the truth join point with the truth's timing 1.7 km
(`docs/phase0_intent_diagnostics.py`, five readings). The network reacts to the channel
(±5 km join → ∓43/+53 s), mostly in timing; its predicted join distance moves 6 → 11 km
toward the truth 14 km but the paths claim the final on fewer flights. Causal traffic counts
from the manifests explain little of the join (R² 0.34 → 0.38 on flights joining after the
anchor; the truth lead ETA 0.47); knowing the join halves the remaining-duration error
(35.8 → 22.8 s). Conclusion: for vectored arrivals the 4D error is dominated by along-path
timing and residual geometry, the join distance is one variable of ~three; the design's
decision variable should include time. New mode `truth-join-duration` (join + the true
remaining time, the duration head's own target) added to size the (where, when) ceiling
with this decoder — its duration error is an identity check. **Its arm: vectored ADE
2011 m (−30 %), pooled 1005 m, duration error 5 s, but the time-free path error does NOT
improve (chamfer 850 m vs 791 join-only, 942 baseline) and the claimed join distance falls
back to 7.4 km — with both truth decision variables this decoder still hedges the geometry,
while a crude trombone with the truth timing scores 1.7 km and the truth path with a naive
speed profile 1.3 km.** Recommendation in the results doc: fix the output side first
(geometric closure: predict (d_join, T) + a speed profile, construct the path), then the
scene encoder; the deterministic top-1 of the scene design is capped near 2 km by this
measurement. Reviewer caveat folded in: a
prediction's "establishes on the final" must be judged with the membership gate
(`hard_on_final` + `stays_mask`), not the k=0.5 truth gate, which the documented 250–350 m
endpoint translation saturates.

### 2026-09-05 — Phase 0 of the scene design: truth-intent conditioning (upper-bound instrument)

`4dTrajectory/ts_transformer/intent_conditioning.py` + `TSConfig.intent_conditioning` ∈
`none | truth-join | truth-join-lead`. The TRUTH join point (chart position of the first
observed row from which the track stays inside the k=0.5 cone, `truth_final_gate` on all
observed rows incl. the lookback) and, with `-lead`, the previous same-runway landing's
TRUE time relative to the window's anchor (`(t_lead − t_anchor)/600 s`, clipped ±1800 s;
lead from the tracks roster's `assigned` landings, model-ready or not) are appended after the
target conditioning as input-only constant channels through the covariate-token path. Reads
the future by design — a development measurement of what inferring the intent is worth,
never a deployable model; the run name carries `intent=truth-…`. Contracts: the row is built
at the window's actual anchor (`series_conditioning(..., anchor=)`), random train anchors are
refused with the lead channel, iTransformer only, `runway-aligned` refused (the gate reads
chart e/n against the world course). `FlightSeries.lead_landing: LeadLanding | None` — `None`
means the roster was never consulted (the channel raises), `LeadLanding(None)` means no
earlier landing (reads as the negative clip); `load_flight_dicts` attaches it for every
flight from the same manifest the flight came through. Data facts (KRDU/KSJC manifests):
`entry_time_utc` is within 1 s of the first slice sample; median lead ETA at the anchor
−27 s, 44 % of leads still airborne, 95 % within ±1800 s; a 150-flight KRDU sample shows
d_join p5/p50/p95 = 10.6/18.7/23.7 km with no never-established fallback.

Review found two pre-existing blockers on the way: the control training loop
(`train.py`) and the auto-batch probe (`batching.py`) still passed `x[:, -1]` — the whole
conditioned row — as the anchor state, so ANY control run with conditioning columns died on
the first batch (`anchor_state(x, C)` everywhere now, with a control-path training-step
test); and the named-recipe override check in `__main__.py` rejected fields outside
`PROCEDURE_LOSS_FIELDS | CONTROL_HOOK_FIELDS` (now `| INTENT_FIELDS`, plus
`--intent-conditioning`). Campaign declaration
`docs/experiments/scene_phase0_arms.json` (O_join_lead, O_join; paired with
`control_procedure_20260905/A_control_v3`; pre-registered gate: vectored ADE < 1.5 km from
2858 m continues the design, no movement stops it).

### 2026-09-07 — Design: scene encoder + join-anchor multimodal control prediction

`4dTrajectory/ts_transformer/docs/2026-09-07_scene_join_anchor_design.zh.discard.md`. Intent: give the
model the variables that decide the join (traffic context — the lead aircraft on the same
runway, queue, time since last landing — plus the final approach course, FAF/IF and STAR
legs as map tokens) and make the join distance an explicit K-way decision with a control
schedule per anchor (VectorNet/Wayformer-style entity encoder + scene attention, MTR/TNT-style
anchor queries, MultiPath winner-takes-all training; the differentiable rollout, envelope,
teacher and predict-time barrier all kept per mode). Deficiencies of the current
iTransformer-variate-token + flattened single head named with the measurements behind them.
Phases: 0 oracle upper bound (truth d_join + lead ETA as covariates), 1 data-plane scene index
/ context with leakage tests, 2 minimal-change context arm on the existing backbone, 3 the
new encoder/decoder with pre-registered vetoes (random-anchor control, straight-in top-1 must
not regress), 4 closure/wind/pooled airports. Reading list attached.

### 2026-09-07 — Control training design review: the below-glidepath endpoint is an objective-design fault

`4dTrajectory/ts_transformer/docs/2026-09-07_control_training_review.zh.md`. The control
baseline's endpoint 157 / 162 m below the glidepath (KRDU / KSJC medians) is the last
minute of the rollout: on the final (1–8 km) the prediction sits within ±13 m of the
glidepath, at the truth's landing time it is 540–680 m short and 140 m low with a path
angle of −4…−5° against the truth's −3°, and the rollout runs only 3–10 s past the truth
(at inference; in training simple-v3 rescales the durations to the truth's final time).
Ruled out: data and coordinates (truth endpoint +28 / +6 m), the fitted tail (median 6 / 2 s,
379 / 11 m before the threshold, no flight over 60 s). The model's last-minute controls
(load 1.014, thrust 0.069, speed 75 m/s) each sit near the inverted teacher's (0.999, ~0,
71 m/s) yet fly a different path — the open-loop imitation teacher (47× the position
term) says what the truth does at the truth's state, never what to do once drifted, and
the isotropic 10 km metre-scale position loss (state path too) prices a 150 m height
error at 2e-4 per endpoint averaged over 64 (terminal terms share the scale), with no
threshold-crossing event, no ground in the rollout, and no path gradient to the time grid. Proposals ranked: per-channel position scale / vertical-only procedure term /
threshold-plane crossing loss / ground hinge (P0), a closed-loop teacher built from the
guidance laws on the rollout's own state (P1, the nominal law moved from filter to
teacher), event-defined duration and closed-loop prediction (P2); first diagnostics named.

### 2026-09-06 — Command-hook campaigns (KRDU v1 + v2, KSJC v2): the barrier is a predict-time safety layer, training through a hook is not

Report `4dTrajectory/ts_transformer/docs/2026-09-06_control_hooks_results.zh.md`; campaigns
`control_hooks_20260906` (v1, KRDU) and `control_hooks_v2_20260906` (KRDU + KSJC), simple-v3,
one seed, paired against `control_procedure_20260905/A_control_v3`. The v2 soft barrier
applied at prediction to the untouched baseline: KRDU pooled FDE 1650 → 1449 m, ADE
1333 → 1278, FDE better on 84 % of flights, none worse by 1 km (v1: 54 such flights); KSJC
996 → 930 m, 90 % better; straight-in endpoint |xt| p95 1821 / 407 → 70 / 46 m; vertical
untouched. Hard saturation + hard gate is worse everywhere (the jump at the cone edge).
Training through a hook: six trained arms (barrier v1/v2 + nominal v1/v2 at KRDU, barrier v2 +
nominal v2 at KSJC), none beat its predict-time counterpart — pooled ADE +2…+21 %, bank skill
below the baseline, vectored path middle worse; the pre-registered lazy veto (clamped ≥ 20 % and skill below
baseline − seed noise) fires at KRDU. The nominal-law hook is the vertical complement (KRDU
endpoint height above the threshold −164 → −7 m median, vertical violation rows 46.6 → 29.1 %) and revealed that
the control baseline ends 157 / 162 m below the glidepath (median, KRDU / KSJC).
Adopted: `predict --command-hook barrier --hook-saturation soft`. Next: the combined hook.
Readouts (`readout.json/.txt`, `score.txt`) are in each campaign dir.

### 2026-09-06 — Nominal-law hook v2: thrust held to the unhooked rollout's speed (the third law)

The first campaign's `R_nominal_residual` arm (KRDU) lost the endpoint: pooled FDE 1650 →
2334 m, straight-in FDE median 703 → 1461 m, while its lateral was the most realistic of
any hook (bank skill 0.659, straight-reference bank RMS 0.54° against the flown 0.41°).
Endpoint analysis: the glidepath law pulled the aircraft up from below the glidepath
(endpoint height error −167 → −87 m) with the thrust passed through, the speed fell to
58 m/s (baseline 88) and the rollout ended 584 m short. A coordination that pays back
only each hold's path-angle change does not fix it: a trim load factor reads as "hold the
current path angle" at any path angle, so the network's intended path — and speed — is
unobservable from a segment's command. The engine
(`rollout_piecewise_constant_hooked_with_step(track_reference=True)`) now integrates the
schedule UNHOOKED alongside for hooks that declare `needs_reference`, the lag backend
passes it as `RolloutStateView.reference`, and the nominal law's thrust is
`T' = T + k·m·(V_reference − V)` (`guidance_laws.speed_hold_thrust`,
`control_nominal_speed_gain` = 0.1/s): the schedule's own deceleration is kept and only
the hook's energy cost is paid back. Tests: the speed-hold law, the thrust at and below
the reference speed, and the convergence rollout ending within 3 m/s of the unhooked
rollout's speed where the passthrough was more than 10 m/s off. R reruns in
`control_hooks_v2_20260906` (the first campaign's copies are removed).

### 2026-09-06 — Barrier filter v2: lag-aware, evaluated at the lead position, load-coordinated (after the first campaign's predict-only readout)

The first campaign's predict-only arms (hard / soft barrier on the `simple-v3` baseline, KRDU)
did on straight-ins what the penalty could not — lateral violation rows 35.5 → 1.6 %, endpoint
|xt| p95 1821 → 56 m, FDE median 703 → 492 m, ADE unchanged — but 40–49 vectored flights lost
more than 1 km of FDE (worst +14 km). Traced on the worst flight: the rate-only heading rule
(`ψ̇ = β·excess`) on 7 s holds with a 2 s bank actuator commanded +28° for two holds, then −29°
(a limit cycle), and because only the bank was changed the vertical lift component `n cos μ`
the network had paired with its load factor was lost — path angle −1° → −10°, 93 → 200 m/s,
16 km past the threshold. v2 (`control/constraints/barrier_filter.py`): the heading layer asks
for a heading CHANGE over the hold and credits the bank the actuator is already in
(`Δψ ≈ g/V_h·[tan μ_c (Δt − τ_eff) + tan μ_0 τ_eff]`, `τ_eff = τ(1 − e^{−Δt/τ})`, inverted for
`μ_c`); the corridor margins are evaluated where the aircraft will be when the command bites
(current velocity carried for `τ_eff`); the load factor is re-coordinated, `n' = n cos μ /
cos μ'`; the heading-gain default drops 0.3 → 0.1 (with the lag credit, `βΔt = 1` meant "close
the whole excess in one hold", 45° of bank for a 30° error). `RunwayAxesView` gains `hold_s`;
diagnostics gain `hook_load_change`. Tests: the lag credit and the lift invariant, and the
first campaign's worst entry replayed through the lagged rollout (capture in the first holds,
≤ 15° after, corridor bounce ≤ 120 m, path angle within [−4.5°, −1°], speed within 5 % of
the unfiltered rollout). The F arms rerun as `control_hooks_v2_20260906`
(`docs/experiments/control_hooks_v2_arms.json`); the nominal-law hook is unchanged.

### 2026-09-06 — Command hooks on the control rollout: barrier filter and nominal law + bounded residual (implemented; campaign `control_hooks_20260906`)

The control path's own constraint mechanism, per `docs/2026-09-05_control_constraint_design.zh.md`
P0 + P1. A **command hook** is called once per control segment, at the segment's start, with
the physical state the rollout carries there and the network's command, and returns the
command actually flown; the rollout integrates each segment on its own
(`aerodynamic_model.torch_piecewise_rollout.rollout_piecewise_constant_hooked_with_step`),
so the per-segment discrete adjoint stays exact and the hook's dependence on the state is
ordinary autograd across segments (finite-difference checked by the reviewer to ≤ 7e-7).
The hooked dense rollout settles the effective schedule first and then integrates it, so
dense and endpoint rollouts agree, and the **effective schedule is what a record carries**
(`EndpointControlRollout.controls`, `forecast` exports it in newtons, `source.commandHook`
names the hook). Only the first-order-lag backends support hooks (their state carries the
actuators; the point-mass backends refuse). Backends expose the state through
`control/dynamics/hooks.RolloutStateView` (transport chart + actuators + the segment's
hold `duration_s`).

Two hooks in `control/constraints/`, sharing the runway-axes adapter and the on-final gate
(`gates.py`; membership cone floored at 500 m + path alignment, read from the rollout's own
velocity): **`barrier`** — the corridor's two barriers `h = k·hw(d) ∓ xt` bound the sine
of the heading error, the heading interval is a second barrier pair
(`−β(ψ_err − lo) ≤ ψ̇ ≤ β(hi − ψ_err)`, one continuous rule — an earlier form that only
acted outside the interval clamped a centred aligned command to zero bank), the level-turn
relation makes that a bank interval, and the command's bank is saturated into it (scaled
softplus in training, hard clamp at inference), lateral only; **`nominal-residual`** — L1
lateral guidance + a glidepath flight-path-angle law (`control/guidance_laws.py`) give the
nominal bank and load factor, the command is a tanh-bounded residual on them (±5°, ±0.1).
Both are **discrete-time** rules: the command is held for Δt, so every rate gain is used as
`min(gain, 1/Δt)` (a faster rate crosses the edge inside the hold it protects). `hard`
selects the hard saturation AND the hard gate (a deployed filter has no partially-gated
rows); training refuses it. The closing term reads `corridor_halfwidth_slope`, the
derivative of the same half-width the corridor uses (zero past the threshold).

Config: `control_command_hook` (`off|barrier|nominal-residual`), `control_hook_gate`
(`on-final` only), `control_hook_saturation`, `control_barrier_alpha/heading_gain`,
`control_nominal_*`; needs control output, first-order-lag, native grid, `enu`. Predict:
`--command-hook`, `--hook-saturation` apply a hook to any lag checkpoint (the
`F_barrier_infer` arms). Epoch diagnostics `EpochResult.command_hook` (per-step gated /
clamped / saturated shares, mean bank change). Readout `compare_constraint_arms.py` gained
the outside-corridor recovery columns (first claimed on-final row outside → last inside).
Tests: `tests/test_command_hook.py` (identity hook bit-exact, state-reading hook
differentiable, point-mass refusal), `tests/test_control_constraints.py` (geometry adapter,
barrier bounds and continuity, adversarial rollout stays inside, nominal law converges
through the rollout, config guards, training refuses hard + logs the hook, export contract).
Arms: `docs/experiments/control_hooks_arms.json` (baseline + hard filter at prediction,
+ soft filter at prediction, trained through the soft filter, trained through the nominal
law; simple-v3, paired against `control_procedure_20260905/A_control_v3`).

### 2026-09-05 — Procedure penalty on the control rollout (wired, measured, not adopted); control constraint design

Code (9d9e66e): `procedure_loss` now charges the control path's native-grid rollout
endpoints against their aligned targets (`ControlStateLossResult.aligned_targets/weights`),
per-flight term + diagnostics through `ControlLossTerms`; TSConfig admits `procedure_loss_*`
on control (native grid only); control dynamics carry `glidepath_tan`; `PROCEDURE_LOSS_FIELDS`
is one source, open under named recipes (CLI + `run_naming` render `recipe+(edits)`);
`config.recipe_settings` is the single recipe-content helper for both arm runners;
`coerce_sequence_fields` makes the JSON round trip of tuple fields lossless for the dataclass
and the CLI's frozen-recipe check (subprocess-tested).

Campaign `control_procedure_20260905` (simple-v3, openap-direct cohort, KRDU + KSJC, one
seed, λ = 1e-3 / 5e-3): the penalty pulls the endpoint into the corridor (pooled FDE −101 /
−39 m, straight-in FDE −50 / −58 m, KRDU endpoint |xt| p95 2769→2203 m; KSJC's tail does not
improve) but pushes the vectored path middle away (vectored ADE +581 / +245 m), the violation
rate barely moves, and 5e-3 collapses the bank schedule at KRDU (bank skill 0.728→0.280). Report
`4dTrajectory/ts_transformer/docs/2026-09-05_control_penalty_results.zh.md`. Design for the
control path's own constraint mechanisms (rollout `command_hook`, barrier filter, nominal law
+ bounded residual; reference path deprioritised):
`docs/2026-09-05_control_constraint_design.zh.md`.

### 2026-09-05 — Final-approach corridor in the learned model: bounded output adopted, penalty vetoed, projection as fallback

Code (5b54fae): `flight_scenarios/fas_geometry.py` (FAS cone, one definition; the backend
bridge imports it), `flight_scenarios/procedure_final.py` (RNAV(GPS) document + FAF read;
`scenario_optimization` delegates), `ts_transformer/final_approach_geometry.py` (torch
corridor geometry tested against `approach_constraints`), `state_position_reference=
"corridor-bounded"` + `corridor_gate` (`StateOutputLayer` saturates cross-track/height on
the rows the output places on the final, direction from predicted positions),
`procedure_loss_*` with `ProcedureMultipliers` (fixed or dual on the violation rate,
"procedure" is the state objective's fifth component, λ logged per epoch and the selected
epoch's stored), `predict --project-final GATE`, per-flight final-approach context in the
batch context slot (`enu` chart required), predict-only arms in `run_ts_frame_ablation.py`,
readout `docs/compare_constraint_arms.py`. `StateOutputLayer.offset_mask` is no longer
persisted and `load_checkpoint` tolerates checkpoints that stored it (both 2026-09-03
generations load). Reviewed by a subagent before the commit.

Campaign `final_constraint_20260904` (KRDU + KSJC, val split, paired with the 2026-09-03 arm A,
two seeds): the bounded output improves pooled FDE −51…−91 m and pooled ADE −8…−71 m on all
four runs (two airports × two seeds) with corridor-violation rows 77→48 % (KRDU) / 34→21 % (KSJC) and the pre-registered
veto (a vectored regression on both seeds) not triggered at either airport (KRDU vectored ADE
−7 / +49 m, KSJC −28 / −204 m); the dual-ascent penalty diverged on all four runs (an
unreachable ε turns the multiplier into a ramp) and the fixed parity penalty pays 42 m of ADE
for the same violation drop; the row-by-row on-final projection (the layer's gate, hard) recovers most of B's KRDU
FDE gain post hoc but not its violation rate or endpoint tail (a first, tail-only version moved
1.35 % of rows and was replaced), the FAF-gated projection is the straight-in ceiling (KRDU FDE
643→455) at the price of vectored flights.
Report `4dTrajectory/ts_transformer/docs/2026-09-05_final_constraint_results.zh.md`.

### 2026-09-04 — Procedure constraints for the learned model: adherence measurement, design, method survey

No code path changed. `4dTrajectory/ts_transformer/docs/measure_procedure_adherence.py`
(read-only over the harvest + CIFP procedure documents, every 3rd rostered arrival) measures
how observed flights sit against the optimizer's constraint rows: 0.0 % pass an off-axis
IAF at either airport, 85–97 % (KRDU) are established in the k=0.5 LPV cone by the FAF, and
once established 87–99 % of samples satisfy the cone and the −60/+120 m glidepath window
(±22 m over the whole final: 14–69 %). `docs/2026-09-04_procedure_constraints_design.zh.md`
turns that into the design (final-segment corridor + glidepath only, gated by the flight's
own join distance, runway-scale hinge in training + projection / casadi tracking at
inference, shared geometry via a torch dispatch in `approach_constraints.mathx`);
`docs/2026-09-04_constraint_methods_survey.zh.md` surveys the alternatives (bounded output
reparametrization, differentiable projection layers, primal-dual training, sampling +
filtering, two-stage predictor + optimizer, data-side) with reading lists and the P0–P3
implementation order (shared geometry → projection arm → bounded reparametrization paired
with a primal-dual penalty arm → optimizer tracking objective).

### 2026-09-03 — state-v2 candidate: anchor-relative state positions (mechanism fixed, recipe vetoed)

`TSConfig.state_position_reference` (`absolute` default | `anchor-relative`):
`StateOutputLayer` can read the forecaster's position channels as displacements from the
anchor and add the anchor's normalized position back, so "start where the aircraft is"
is the zero output. `compare_frame_arms.py` now reports the first predicted step against
the anchor's kinematic extrapolation and takes `--only`; `run_ts_frame_ablation.py`
gains `--informal` (no experiment manifest / clean-tree guard, as
`run_ts_control_arms.py` runs). Four runs (KRDU + KSJC, two seeds) paired against arm A
(`docs/2026-09-03_state_v2_anchor_relative_results.md`): the first-step jump collapses
(KRDU established flights −348…+239 m → 0…7 m lateral), straight-in FDE 643 → 492 m and
lateral miss +204 → +21 m, KSJC pooled ADE −55…−86 m on both seeds — but KRDU vectored
FDE +350 m on both seeds (worse on 78–82 % of them), endpoint lateral p95 ×1.4, so the
pre-registered veto fires and `absolute` stays the default. Reading: absolute output
= "end at the origin" prior, anchor-relative = "start where you are"; each stratum
wants a different one. Next candidate: absolute output + first-step continuity term.

### 2026-09-03 — Runway-hypothesis expansion: what the runway label is worth, and what recovers it

`run_ts_runway_hypotheses.py` runs a trained threshold-anchored checkpoint once per
candidate runway for every validation flight (clone the flight dict with that runway's
CIFP target, same `build_series`/`forecast` chain, forecast mapped back to world
coordinates and scored in the true runway's chart) and evaluates selection rules over
the K hypotheses; the assigned label reproduces the baseline exactly. Includes a
mirror-image pseudo-sibling per flight (same separation and course as the real parallel
sibling, opposite side) as the noise control for any oracle. Result
(`4dTrajectory/ts_transformer/docs/2026-09-03_runway_hypothesis_expansion.md`, KRDU +
KSJC, two seeds each): a causal active-configuration rule from co-temporal development
landings recovers the runway DIRECTION (KSJC 93.8 % overall; KRDU majority runways
80–83 %) but not the parallel side (KRDU 05R/23L 29–31 %), which costs +19 % pooled FDE
at KRDU (+30 % on straight-in flights, ~500–800 m on the minority runways) and nothing at
KSJC (230 m separation). The real-sibling oracle gains 79 m median FDE at KRDU against
32 m for the fake sibling, so roughly half of a K=2 sibling oracle is luck; the
forecast's own closest approach is not a usable selector. The parallel side is the one
genuine unresolved mode. The 150–200 m NW endpoint bias of arm A at KRDU was then traced
(`docs/2026-09-03_krdu_nw_endpoint_bias.md`): a model-side, world-fixed translation of
the whole predicted path from the first step, on straight-in flights, with the sign of
KRDU's population-mean lateral drift and below the objective's resolution; observed
tracks and CIFP geometry are on the centreline to metres.

### 2026-09-03 — Airport-center frame ablation: the threshold anchor IS the target conditioning

Asked whether the ts_transformer's only runway knowledge — the chart being anchored at
the assigned threshold — is a prior worth keeping (plan
`4dTrajectory/ts_transformer/docs/2026-09-03_airport_frame_ablation_plan.md`). Three
commits of mechanism, one of tooling, then 14 runs. (1) `channels.target_chart_position` /
`FlightSeries.target_chart` names the target's chart position; the five consumers that had
silently equated it with the origin (observed crossing plane, inference truncation,
`horizontal_distance_m`, and two the plan missed — the fixed-anchor common-grid truncation
that selects checkpoints, and the difficulty covariates) now measure from it; verified
bit-identical on 300 real KRDU arrivals in both threshold frames. (2)
`coordinate_frame="airport-enu"`: `AirportENUFrame` anchored at the airport reference
point from the harvest's own `runway_thresholds.json` entry
(`flight_scenarios.runway_target.airport_reference_point`); `COORDINATE_FRAMES` moved to
`coordinate_frames.py`, config imports it. (3) `target_conditioning="channels"`: five
input-only channels through iTransformer's vendored covariate-token path;
`TSConfig.input_channels` / `enc_in` = input width; checkpoints serialise
`input_channels`; `batch_contract.anchor_state` replaces every `x[:, -1]`; PatchTST refused.
(4) `run_ts_frame_ablation.py` (state arms, paired split, resumable, no per-arm CV, no
CZML) + `docs/compare_frame_arms.py` (pre-registered stratified/paired readout with the
endpoint's cross-track against the assigned centreline).
Result (`docs/2026-09-03_airport_frame_ablation_results.md`): at KRDU the airport frame
makes the deterministic model average across each parallel pair — endpoints nearer the
sibling runway 1.5 % → 12–15 % on both seeds, the minority runway (05R, 23L) pulled
570–680 m toward its majority sibling, its median FDE +30–45 %, p95 lateral endpoint
error ~2× — and the target coordinates fed as data change none of it; the vectored-stratum
gain the plan hypothesised flipped sign on the second seed (KSJC −184 → +62 m); the only
consistent effect of the conditioning is −10 % final-time MAE (the flattening duration
head reads it, the attention backbone does not). Seed floor: threshold arm 5–22 m pooled
ADE, airport arms up to 107 m. Decision: keep `enu`. Also verified: the v5 re-roster
CLAUDE.md still called pending was already on disk for all five airports with rosters;
the 14 failures in `trajectory_data_process/tests/test_ts_pipeline.py` /
`test_download_landings.py` reproduce at the pre-change commit (recorded in
`docs/code-health-followups.md`). ts_transformer suite 419 pass.

### 2026-08-24 — Result pickers rank a split's results by mean/p95 ADE/FDE

The Prediction and Experiments pickers listed a split's results in name order only —
no way to see which model/arm actually scored best. Each category's aggregate accuracy
existed only inside its own `comparison_index.json` `prediction` block, too heavy to
fetch per category just to order a dropdown, so `build_scenario_comparison_czml.py`'s
`category_accuracy_summary` now stamps a compact `accuracy` block (`adeM`/`fdeM`, each
mean + p95 — a SUBSET of the index block, never recomputed) onto every prediction
category entry in `categories.json`, and
`ts_transformer/docs/backfill_category_accuracy.py` backfilled the 59 published
categories from their own indexes (idempotent, metadata-only). The frontend gains a
"Sort results" selector (Default / ADE mean / ADE p95 / FDE mean / FDE p95) shown for
the Prediction and Experiments sources: prediction categories are ranked best-first
WITHIN their split optgroup (cross-split error comparisons are meaningless, so the sort
never mixes groups), experiments are ranked within their campaign by the split
currently in view (val fallback), options show the metric value inline, and entries
without a value keep their order at the end. An earlier misread of this request
(per-trajectory ADE/FDE sorting inside one category, incl. a 140k-record group-level
backfill) was fully reverted the same day — group records carry no `adeM`/`fdeM`.
Verified: frontend 523 pass + tsc clean, aeroviz-4d python suite 155 pass.

### 2026-08-24 — One naming grammar for every ts_transformer run and published category

The frontend's learned-prediction categories carried three label dialects (verbose
pipeline concatenations, publisher `run_id (model, output, horizon)` strings, ad-hoc
hand labels like "budget: lr 1e-4"), and the Experiments picker showed raw run-directory
names — none of which said what was actually experimented. New
`ts_transformer/run_naming.py` is the single source of a canonical grammar derived
mechanically from a run's serialized config: `output · backbone · dynamics · loss ·
meta`. Dynamics distinguishes control-derivative handling (`point-mass` vs
`first-order-lag`, `@backend` when non-default); loss names a `custom` run against its
NEAREST recipe (fewest loss-field edits, e.g. `simple-v3+(imit=16)`; >4 edits → stable
`custom-<hash>` version); meta lists deviations from today's defaults (seed first,
capped, `+N more`) plus caller extras (run id, campaign/arm, cohort). Wired into
`run_ts_pipeline.py` (replacing ~60 lines of label concatenation and five dead label
helpers), `publish_ts_experiment_trajectories.py` (labels + a stamped
`experiment.label` the frontend now prefers; new `--refresh-labels-only` walks stored
publication manifests and re-derives labels without predicting anything), and
`experiment_index.py` (`display_name` + INDEX.md Name column). Applied on disk as
metadata-only edits — 45 legacy `ts_*` categories via the one-off
`docs/relabel_published_categories.py`, 14 publisher-managed categories via the refresh
mode, indexes rebuilt; no record, CZML, checkpoint, or directory was renamed or
deleted (category keys/dirs are historical record; `run_slug()` covers future ones).
Notable relabel finds: the "velocity 2x" wiggle arm is exactly `simple-v2`'s frozen
dose, and the imitation "~47x" arm is exactly `simple-v3`. Verified: ts_transformer
suite 407 pass (label-assertion tests updated to the new grammar; the terminal-clock
collision test now passes the arc-length selection metric its synthetic plan always
implied), frontend 520 pass + tsc clean.

### 2026-08-24 — Anchor calibration landed: manufacturer cluster gone; speed-gate toggle in the viewer

A320-family landing Cl_max 2.7 → 3.0 (calibrated from Airbus's VLS = 1.23·Vs1g +
published VLS figures; pin test holds the modeled floor at-or-just-below published
VLS) and the C56X airframe restored from OpenAP's C550 surrogate to certificated
facts (`6e31f2d`). Republished: A319/A320/A321 speed-pass 46.6/54.8/41.5 % →
**78.9/85.4/80.3 %**, C56X fast-fail 53.5 → 20.7 %, composite of-decided now nearly
uniform across airports (71.2–75.8 %) — confirming airport spread was anchor ×
fleet mix. Residuals documented, not patched: A21N/A20N (neo subfleet lands far
below MLW — needs an operational landing-mass model), B763, E75L. Results doc §8;
follow-up 14 resolved to those residuals. The same commit family adds the Details
window's **speed-gate toggle** (`af32454`): two-gate/three-gate verdicts re-derived
client-side from per-row component results via `composeVerdict` (a declared mirror
of `_composite`, truth-table pinned) — no schema change, published reports stay
three-gate.

### 2026-08-24 — First baseline speed-gate measurement: the anchor, not the weather, dominates

`evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` — full statistics of the first
three-gate fleet baseline (per airport, per runway, per airframe type, slow/fast
margins, wind-explainable bands). Headlines: fleet speed pass 22,473/32,191 graded
(69.8 %); fails split 66 % slow / 34 % fast with median margins 3–5 kt; ungraded is
almost entirely unresolvable airframes (10,306 — GA-heavy KRDU/KSJC hit hardest);
**the fail structure clusters by MANUFACTURER** (A21N 91.9 % slow vs B738 75.7 %
pass, same days, same airports), so the per-type stall-model anchor — not wind — is
the dominant effect, recorded as code-health follow-up 14 (review A320-family
Cl_max/landing mass; the same anchor is the optimizer's velocity floor). Airport
inversion (KMSY/KSTL fast-skewed vs KSJC 88 % slow) is fleet mix, the KSJC-ADE
lesson again: quote speed rates per type, never bare.

### 2026-08-24 — The baseline runs the speed gate: three gates for every subject

Owner decision, superseding v6's observed exclusion: the observed baseline is judged
on the SAME three gates as its modeled twins. The fitted crossing GROUND speed is
graded against the stall-anchored window as a STATED PROXY for airspeed — the wind
caveat (10 kt headwind = half the window) travels with a distinct criterion id
(`vref_1p23_vs1g_to_vref_plus_20kt_ground_speed_proxy` on every observed row's
bounds) and `methodology.terminal_speed.observed_proxy_caveat`, never silently.

- **The window anchors on the flight's own airframe**: `harvest/observed.py` resolves
  each record's icao24 through `flight_scenarios.resolve_landing_aero` (the scenarios'
  identity→OpenAP chain, no fallback type) and writes `source.landing_aero` + the
  type's landing mass — so baseline and modeled twins share one set of stall
  assumptions. Unresolvable airframe: NOMINAL_MASS_KG, no landing_aero, speed grades
  indeterminate loudly (reason named), composite indeterminate.
- **Composition**: speed ∧ lateral ∧ vertical for observed exactly as for computed —
  the headline baseline pass rates MOVE (previously two-gate). `crossing_speed_ms`
  (airspeed) stays null on observed rows; the proxy lives in
  `crossing_ground_speed_ms` — two quantities, two names, never mixed.
- Frontend: the observed Details speed-gate card returns (it is a real pass rate
  now); the gates note explains the proxy; per-row speed verdicts + windows render.
- `evaluation/docs/THRESHOLD_SPEED_GATE.md` §5 records the override and the caveats;
  the pipeline-integration test now pins the unresolved-airframe outcome.

### 2026-08-24 — Unified record design: observed records say where their crossing lives

Evaluation now grades every subject through ONE state interpolation. The design
("2b"): an observed record's `source.crossing_span` marks its crossing —
`measured_bracket` (the instrument-selected direct bracket: left_index + fraction,
reproduced from the event) or `fitted_tail` (exactly one inferred crossing row
appended after the measured states; MSL, V = event ground speed with a recorded
fallback, t = trapezoidal estimate nothing grades on). Computed records carry no
marker — the artifact under test cannot author the quantity it is graded on
(`record_from_dict` enforces observed-only).

- **One crossing definition fleet-wide**: `final_approach.crossing.bracket_fraction`
  + `interpolate_channels` replace three hand-rolled copies (harvest direct bracket,
  evaluator computed path, ts dataset supervision truncation). The marker schema +
  `validate_crossing_span` live beside the event contract — the same
  producer/evaluator seam pattern.
- **Producer half**: `flight_scenarios.crossing_span.crossing_span_from_event`;
  `harvest/observed.py` is the one span-producing writer (optimizer/ts references
  carry no event and stay markerless → still `unavailable`/indeterminate, unchanged).
- **`_observed_arrival` no longer reads event geometry** — the event remains for
  identity/staleness validation, the datum cross-check, audit copy, and the
  ground-speed audit stat. `final_time_s` stays pinned to the last MEASURED row
  (span-aware contract), so flight-time and Δt-vs-observed statistics do not move;
  `TrajectoryRecord.measured_states` keeps path-shape/span comparisons on flown
  trajectory only. Two subject branches remain BY POLICY: missing-crossing outcome
  (reception gap ≠ model shortfall) and speed-gate scope (ground speed never graded).
- Deliberate small semantic changes, all in descriptive columns: observed rows'
  `speed_ms`/`heading_rad` now measure the crossing state against the measured-end
  target (previously tautologically 0), and `along_track_m` is the projected ≈0
  instead of exactly 0. Verdict-bearing quantities (lateral, vertical, verdicts,
  flight times, ground speed) are preserved — pinned by the republish diff below.
- Fixtures made physically self-consistent: the test event's crossing position now
  ENCODES its cross-track offset instead of contradicting it (the old event-based
  reader never noticed; honest geometry does).
- Pre-span records in `approach/records/` fail loudly with the cure
  (`--evaluate-only`); all five airports republished after the change.

### 2026-08-24 — Observed crossing ground speed wired through the report to the Details window

The event's audit speed now actually reaches a reader — the missing consumer half of
today's harvest change (additive within report schema v6):

- `ArrivalDeviation.crossing_ground_speed_ms` is filled by `_observed_arrival` from the
  event; rows carry it flat + under `deviation`, batches carry a
  `crossing_ground_speed_ms` spread, and `METHODOLOGY["observed_crossing_ground_speed"]`
  declares it: ADS-B ground-referenced, audit-only, never composed, never fed to the
  stall-anchored gate (the graded-airspeed slots stay untouched — pinned by test).
- **Flat-row bug fixed on the way**: `crossing_speed_ms` lived only under
  `row["deviation"]` while the frontend verdict table reads it flat, so the "V crossing"
  column would have shown dashes on real v6 reports. Both crossing speeds are now flat.
- Details window: a "crossing ground speed (m/s, ADS-B)" aggregates row, and the
  per-row V-crossing cell falls back to the audit value with an explicit
  "audit only, not graded" tooltip. Observed reports still show no speed-gate card.
- Observed evaluation reports republished for all five airports (`--evaluate-only`)
  so the on-disk v6 artifacts carry the new fields; lateral-pass rosters rebuilt after
  (the `arrivals._clear` deletion footgun, again).

### 2026-08-24 — Fleet reclassified: every stored event now carries the crossing ground speed

All five airports ran `--reclassify-existing` (267,194 stored tracks), so the
`crossing_ground_speed_m_s` field from today's harvest change is now IN the data:

- **Rosters unchanged, as required**: tracks and arrivals counts are identical to the
  saved before-state at every airport (assignment is deterministic; only the event
  payload gained the field). Spot checks (400 assigned tracks/airport): **100 % of
  estimated events carry the speed**, direct and censored alike, medians 65.6–72.7 m/s
  (≈ 128–141 kt). KRDU is ~95 % censored events where KMSY/KSMF are ~99 % direct —
  the coverage-ends-short signature, visible per airport for the first time.
- KMSY/KSMF/KSTL/KRDU ran serial (~210 tracks/s, 91 s–7 min 21 s each); KSJC ran on
  the new `--jobs` path (24 workers) in **5 min 0 s** end-to-end for 106k tracks —
  the classify fan-out is no longer the bottleneck, the serial arrivals/CZML/report
  tail is.
- KSJC's first attempt died on `ENOSPC`: the stage-then-swap needs the airport's
  tracks footprint free (2.2 GiB) and the disk was at 100 %. Staging unwound cleanly
  (`tracks/` untouched — the design working as intended). Freed by deleting pure
  caches only (conda tarballs + unused packages, pip, vscode-cpptools; user-approved);
  `~/.cache/opensky` (18 G) deliberately untouched. Disk now ~14 G free — the pending
  optimizer batch (12.3 GiB) fits again.
- The five `lateral_pass_eligibility.json` rosters (deleted by the re-roster footgun,
  missing since 2026-08-21) were rebuilt via `ensure_lateral_pass_roster`. That fixed
  the 3 `test_ts_pipeline` failures that read the real harvest root; the other 11 are
  fixture rot at HEAD (hermetic fixtures predate the roster requirement) — recorded as
  code-health follow-up **13**, left to the active ts-pipeline workstream.

### 2026-08-24 — Threshold event carries the estimated crossing ground speed (audit-only)

The observed threshold event now serializes `crossing_ground_speed_m_s` — additive
within `runway-threshold-event-v1`, optional on read (every stored event predates it):

- **Direct events** interpolate the bracketing samples' `reported_ground_speed_m_s` at
  the position's own crossing fraction; those speeds were already required by the
  bracket's source-integrity checks, so the field exists for every direct event.
- **Censored events** OLS-extrapolate speed vs along-track to the plane over the SAME
  kept samples as the position fit, using the same estimator (`final_approach.fit_line`,
  the previously-private `_fit_line` made public — one OLS, not two) and the fit's own
  sample/span standard. When the speed-bearing subset cannot meet it, the field is
  omitted and `diagnostics.ground_speed_fit.omitted_reason` says why — distinguishing a
  post-change unfittable event from a pre-change one.
- The shared contract (`final_approach.event_contract.validate_event`) accepts
  absent/null as unspecified and rejects a present non-finite or non-positive value.
- **It is GROUND-referenced (ADS-B velocity, wind unmodelled) and audit-only**: nothing
  feeds it to evaluation's stall-anchored airspeed gate, observed subjects stay
  speed-ungraded, and no verdict anywhere changes.
- **Blast radius, verified before building**: ts_transformer training never reads the
  event (`dataset.py` computes its own crossing from waypoints) and optimizer targets
  come from `flight_scenarios/fitted_approach.py` / runway data — no retraining, no
  re-solving, no schema bump. Existing harvests gain the field only via
  `--reclassify-existing` (`--evaluate-only` re-rosters stored events unchanged).
- Suites: final_approach + harvest + evaluation + flight_scenarios + backend green. The
  14 `test_ts_pipeline.py` reuse-guard failures observed alongside are PRE-EXISTING
  disk state, not code: every airport's `arrivals/lateral_pass_eligibility.json` is
  missing (the documented re-roster deletion footgun) and those guards read the real
  harvest tree — rebuild via `lateral_eligibility.ensure_lateral_pass_roster`.

### 2026-08-23 — Observe: Optimization result source restored; legacy v5 reports displayable again

Two regressions the Observe panel had accumulated, both verified in-browser on KRDU:

- **The optimizer categories had no home in the result-source selector.** The 07-29
  experiment refactor partitioned categories into only `prediction | experiment`, so
  fitted_adsb / runway / runway_cons (whose manifest entries predate `resultSource` and
  never stamp it) fell through to the Prediction dropdown's "Other evaluation results",
  mixed in with the ts model outputs. `trajectoryResultSources.categoryResultSource` is
  now the single three-way classifier (`optimization | prediction | experiment`; absent
  `resultSource` + non-`ts_` key ⇒ optimization — the same legacy rule
  `EvaluationSummary.evaluationKind` already used, which now delegates to it), and the
  ControlPanel gained an "Optimization" source with its own category selector.
- **"Details" failed with "evaluation report is malformed" for every published category.**
  The 08-23 speed-gate commit bumped the report schema to v6 in all four homes, which made
  `isEvaluationReport`'s exact-version check reject every report on disk — all still v5,
  and the optimization ones cannot be regenerated (their record batches were cleaned up;
  the batch rerun is a standing open item). The reader now accepts the enumerated
  `LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS` (v5 only; the v6 speed fields are typed
  optional and nothing in the window renders them yet), and `EvaluationReportWindow`
  banners a legacy report as pre-speed-gate — its verdicts grade lateral+vertical only —
  instead of refusing to open. v4-and-earlier stay rejected (shape changes, not just
  grading).
- **Follow-up: the Details window now renders the v6 speed-gate statistics** — a
  pass/graded summary card (with the ungraded count), a crossing-speed aggregates row,
  the speed-gate sentence in the gates note, and per-row `speed_result` + `V crossing`
  columns whose tooltip carries the flight's own stall-anchored window (the bounds are
  mass-dependent, so no common band exists to chart). All of it keys on the schema
  version, so a legacy v5 report renders exactly as before, banner included. The card is
  additionally suppressed for OBSERVED-subject reports (the regenerated observed
  baseline is already v6): observed tracks are never speed-graded by policy, so the card
  could only ever read "— · N ungraded" and was mistaken for a data problem on sight.

### 2026-08-23 — Optimization-tree review: batch-driver merge, resume config guard, plan-timeline fix

Full review of `4dTrajectory/optimization/` + the pipeline runner; every confirmed finding
fixed in one pass (suites re-run green; the two pre-existing known failures — numpy 2.x
`test_fixed_time_objective_weights_control_effort_at_one`, `test_optimizer.py` — unchanged).

**Correctness fixes**

- **`--resume` now verifies the solver configuration, not just identity.** Every eval
  record (solved AND failed) is stamped with the batch's `optimization_config`;
  `_resumable_record` rejects a mismatch or a missing stamp, so a resume across a changed
  `--max-iterations`/`--fitting`/`--rollout-dt` re-solves instead of silently absorbing the
  old records and stamping the new config over the whole roster (which `--skip-optimize`
  would then have trusted). Records written before this date carry no stamp → a resumed
  batch over them re-solves everything, deliberately.
- **Summary rows quote the eval record's `final_time_s`** (the replay's last sample), not
  the planned NLP horizon — for guard-truncated replays (the `not_reached` family, 12.5 %
  of a measured KRDU runway batch) the two differ by seconds, and resumed rows (rebuilt
  from eval records) disagreed with their own fresh twins.
- **Constrained plan exports carried time-warped timestamps.** `_node_states_to_samples`
  spread the dense nodes evenly over `[0, T]`, but multiphase node spacing is
  `(T_p/n_seg)/m_sub_p` with per-phase auto `m_sub` — the orange "Optimizer plan" CZML
  track for every runway_cons flight animated wrong. `CollocationOptimizer._extract` now
  exports `last_dense_state_times_s` (pure helper `dense_node_times`), and both export
  paths use it. Positions were always right; existing states files have wrong
  `optimizer_states[].t` only.
- **Shortest-IAF ranking is now the 3D polyline length.** The old Lagrange-curve proxy
  inflated cornered routes (measured +38 % on a two-corner T-arrival, +7 % mild dogleg,
  exact on straight) and could pick a genuinely longer IAF; a fly-by path only cuts
  corners, so the polyline is the tighter monotone proxy (`_path_curve_length_m` →
  `_path_length_m`).
- **`_concat_to_runway` no longer treats two missing fixIds as a match** ("" == "" passed
  the fix_id half; mismatch now requires no PRESENT identifier to match) — the
  optional-fields-compared-to-each-other trap from the coding conventions.
- **Space pre-check refuses only runs that genuinely don't fit**: the estimate now drops
  the CZML family when 'czml' is not in `--outputs` (the refusal message already suggested
  that remedy without it working), and nets each artifact family against what its target
  directory already holds — a `--resume` restart or `--skip-optimize` rebuild no longer
  re-demands the full footprint (which forced `--skip-space-check` on a 98 %-full disk).
- `casadi_optimizer.py` (legacy multiple-shooting, live via the backend): terminal ψ pin
  is now `sin(Δψ/2) = 0` — the plain difference against ψ boxed to [−π, π] read a due-west
  target reached on the other branch as a 2π violation. Also removed the dead `dt` param
  of `segement_integrate_expr` and the broken dead `decision_vector_to_geo_state`.
- `variable_time_warm_start_transcription_optimizor`: the final-time guess is clamped into
  `build_final_time_bound()`'s box — any `arrival_time_s > 1000 s` used to make
  scipy.least_squares reject x0 outright (masked in production by the backend input clamp).
- `approach_constraints`: `ConstraintSet.evaluate` disambiguates colliding violation names
  by segment position (same-kind default-ident legs silently DROPPED the earlier leg's
  rows from the NumPy report; optimizer path unaffected); box-leg `lateral_left/right`
  labels un-inverted (box axis = flight direction, final axis opposes it — feasibility
  unaffected, only report naming); `ConstraintReport.summary(tol_m, tol_rad)` takes the
  caller's tolerances; `TargetFrame.to_ne` fails loudly across the antimeridian (both
  transforms share the non-wrapping assumption); `intercept_angle_deg` documented
  numeric-validation-only (its `fabs` kink sits at the aligned optimum).
- Experiment scripts: `transport_term_comparison` ψ-cross check is now relative
  (the 1e-18 absolute tolerance passed the committed run by 8.6e-19);
  `fixed_enu_frame_error --max-range-km` honoured by the grid radii; scheme-comparison
  orders derived from the fitting name (7-entry hand-list covered 7 of 16 schemes);
  30 km-study docstring named a nonexistent function for system A.

**Structure**

- `optimize_scenarios` / `optimize_scenarios_constrained_iaf` are thin fronts over ONE
  `_run_batch` driver — the "batch edition seam class" (three past bugs from updating one
  copy) is gone.
- `REFERENCE_CACHE_SCHEMA`, `file_sha256`, and the record→track path mapping
  (`observed_track_path`) live in `evaluation_export.py`; the batch AND the runner import
  them (the runner's restated mirror + pin test replaced by a shared-import seam test).
- Workers no longer ship the rollout states twice across the process boundary (the eval
  copy is emptied worker-side; the parent's `states_ref` points at the states file).
- The runner builds each `Plan` once (space check and run share the objects), and its
  reuse validator memoizes SHA-256 by (path, mtime, size) — runway/runway_cons validate
  the same shared reference set, which was ~1.5 GB re-hashed per category.

### 2026-08-23 — Threshold speed gate (report schema v6) + evaluation review

**A third component joins the terminal verdict: the crossing speed must lie in
[1.23·Vs1g, 1.23·Vs1g + 20 kt]**, with Vs1g the project's own 1-g stall model at the
record's crossing mass. 1.23 anchors on 14 CFR 25.125(b)(2)(i) (V_REF ≥ 1.23 V_SR0);
the +20 kt window on FSF ALAR Briefing Note 7.1's stabilized-approach speed element.
Design, worked numbers, rejected alternatives, and trackable sources:
`evaluation/docs/THRESHOLD_SPEED_GATE.md`. Mechanics:

- `aircraft.aero_params.stall_speed_ms` is the ONE stall-speed definition — the
  optimizer's velocity floor (`scenario_optimization._stall_speed_ms`, now a thin
  wrapper) and the new gate import it, so admitted solves and their judge share one
  stall model by construction.
- `flight_scenarios.build_scenario` writes `source.landing_aero =
  {wing_area_m2, cl_max_landing}`; both record producers copy `scenario.source`, so
  optimizer and ts records carry it going forward. A computed record WITHOUT the block
  grades speed-indeterminate loudly (absent = unspecified, incl. explicit null);
  a malformed block raises. Records written before this date lack it.
- **Observed subjects are reported `indeterminate` and never composed** — their V is
  ground speed (wind unmodelled) and ADS-B ends a median 325 m short of the threshold,
  so no crossing airspeed was ever measured; observed composites are bit-identical to
  v5. `ArrivalDeviation` gained `crossing_speed_ms`/`crossing_mass_kg` (None for
  observed).
- Report schema v5 → v6 in all four homes (producer, ts seam import, frontend mirror
  + fixtures via the constant). **Every on-disk v5 report is stale**; regenerate
  before the ts pipeline or frontend reads it. New surface: per-row `speed_result` +
  speed bounds in `bounds`, batch `speed_result_counts` + `crossing_speed_ms` spread,
  `methodology.terminal_speed`.
- Known interaction, deliberate: the optimizer floor (1.10·Vs, admits observed
  touchdown-speed targets) sits BELOW the gate's 1.23·Vs — floor-riding min-time
  solves and fitted-ADS-B/track-end-target solves can legitimately fail speed; and the
  category-default 145 kt target V_ref exceeds the window top for light narrow-bodies
  (E75L-class tops at ~137.5 kt), so `runway`-target solves for those types will fail
  speed until the per-type approach data is refined (follow-up recorded).

Also: an `evaluation` package review (this change's findings that were NOT fixed here
are in `docs/code-health-followups.md` — roster FileNotFoundError path, missing
monotonic-t validation, STATE_KEYS mirror in `arrival.py`, and more).

### 2026-08-21 — KSJC's ADE advantage is a route-mix artifact; 75 takeoffs were in the arrivals

Investigating why KSJC trajectories looked "weirdly short" and its ADE/FDE remarkably better
than the other four airports. Two separate findings; full write-up with every table in
`docs/2026-08-21_ksjc_route_mix_and_ade.md`.

**The ADE gap is composition, not skill, and it reverses under standardisation.** KSJC's data
is not truncated — it is *straight*. Its reception is the best of the five (99.0 % of tracks
start at the 30 km crop edge, max sample gap p50 2.1 s against KMSY's 6.6 s, 1 coverage gap in
400 tracks), and the arrival slice cuts the same ~5.1 km annulus everywhere. What differs is
the flying: whole-segment tortuosity p75 is **1.017** at KSJC against 1.96–2.38 elsewhere,
96.6 % are established on the centreline at 20 km to go (KSTL 38.7 %), and at 15 km out the
interquartile cross-track spread is **12 m**. It is visible before any slicing — median path
inside the raw 30 km crop is 29.1 km at KSJC against 34.6–44.8 km — and at the crop edge
44.8 % are already within 15° of the approach course (KSTL 3.2 %, median 126°, still outbound).
Cause is operational: 86 % of arrivals on 30L, Santa Clara valley plus SFO/OAK Class B funnel
traffic onto the centreline outside 30 km.

Stratifying evaluation flights by post-anchor tortuosity × remaining path, **ADE inside a
stratum is equal across airports** (412–509 m median on "straight, <13 km left"). KSJC simply
has 78.6 % of its flights there against 41.8–61.0 % elsewhere. Reweighted to the pooled mix,
KSJC's ADE median goes **483 → 1526 m — from best of five to worst**, and on its own vectored
flights it already was the worst (3931 m against 2000–2249 m at KSTL/KRDU). The recognisable
signature: ADE and cross-track improve while **FDE does not** (1000 m against KRDU's 1019 m),
because a straight route makes only the lateral channel easy.

Fixed by making the mix impossible to omit: new `ts_transformer/approach_difficulty.py` writes
`route_tortuosity`, `remaining_path_m`, `anchor_range_m`, `anchor_cross_track_m` and
`established_at_anchor` onto every prediction row, plus an `accuracy.difficulty` batch block
carrying the mix and the thresholds the flag encodes. Computed from the observed track the
error is scored against — never the prediction — and in world EN rather than chart axes, so
the numbers do not change meaning across a `coordinate_frame` ablation. Verified against the
independent measurement: KSJC 78.2 % established (78.3 % standalone), KSTL 53.8 % (54.0 %).

**Separately: 75 rostered "arrivals" were takeoffs.** `arrival_segment.py` classified a
never-left-the-ring track as a local circuit only if it started within
`LOCAL_START_RADIUS_KM = 5` of the DESTINATION, so a takeoff from a neighbouring field inside
the 25 km ring passed and was kept whole — first sample on a runway a few km away, on the
ground. **64 of the 75 are KSJC** (KRHV ×28 at 7 km, KPAO ×20 at 21 km, KNUQ ×16 at 11 km),
KSMF 8, KMSY 3, KRDU and KSTL zero; KSJC is the only one of the five ringed by satellite
fields. Every one was long enough to reach the TS dataset. Impact on the metrics is small
(0.24 % of KSJC evaluation flights anchor within 5 km of the threshold against 0.10–0.16 %
elsewhere) — a correctness fix, not the ADE explanation, though it overlaps the known
"duration head cannot predict below ~125 s" item.

`arrival_segment` gained a required `field_elevation_m` (no default — the rows are HAE and a
silently MSL reference would shift the test by the geoid separation without failing) and a
third outcome `"takeoff"`, rostered as `excluded.outcome = "takeoff_in_segment"` with
`ground_start_agl_m` published in the manifest. `truncate_flights` now returns
`(arrivals, locals, takeoffs)`.

Two decisions worth keeping. **The criterion is altitude alone**: over 42 725 arrivals the
first-sample height above the landing runway is bimodal with an empty band — 75 at or below 82.1 m,
**zero between 100 and 150 m**, next at 175.3 m — while ground speed does NOT separate the
populations, because a jet at rotation reads 71–80 m/s on the runway; a speed-and-altitude
rule would have kept 29 of the 75. **The test reads the segment the ring cut produced**, not
the raw track, so a flight that departs a neighbour, leaves the ring and comes back stays a
genuine arrival.

**Stale outputs.** The manifest schema is bumped to `harvest-arrivals-v5-takeoff-excluded`;
loaders compare exactly, so every on-disk v4 manifest now fails loudly. Rebuild with
`python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only` (re-rosters from
stored `tracks/`; no download, no reassignment). KSJC drops 11 146 → 11 082. Existing ts
checkpoints were trained on cohorts containing the removed flights, and published per-airport
ADE/FDE tables should be re-derived standardised, or at minimum quoted alongside
`established_at_anchor_fraction`.

### 2026-08-20 — `simple-v3`: the control schedule is now supervised directly

`simple-v2` scored position (derivative order 0) and velocity (order 1). Bank lives at
order 2, so nothing in the loss ever named it, and unsupervised it landed **below a
trivial baseline**: on KRDU the predicted bank carried less information about the flown
bank than a randomly chosen other flight's did (per-flight skill 0.124 against a
random-flight floor of 0.170; a same-runway twin reaches 0.679). The visible symptom was
the one reported from the viewer — curves and reversals on references that are dead straight.

New `control_imitation_loss_weight` scores the predicted schedule against the one
`control_inverse_dynamics` reads off the flown track, through the **same registry the
forward model dispatches on**, so target and rollout can never be different equations. The
target is built in `dataset.reference_control_supervision` on the training-only
`_dynamics_arrays` path — `dynamics_arrays()` stays free of it because forecast/predict call
it and there is no future to invert there. Default 0.0, so simple-v1/v2 stay bit-identical.

`simple-v3` = `simple-v2` + `control_imitation_loss_weight = 64.0` (~47x the position term).
On 1404 KRDU validation flights: bank skill 0.124 → **0.735**, flight-independent share of
the bank 49.0 % → **3.3 %** (KRDU's own flown tracks: 1.8 %), straight-in bank RMS 3.92 →
**0.36°** (0.41°), sign reversals 5 → **0**, ADE better on **57.0 %** of flights (656 → 501 m,
p=1.9e-7), FDE unchanged. First change in this investigation that buys control structure
without paying accuracy for it.

Dose chosen off an eight-point ladder (0 / 0.74 / 1.47 / 2.94 / 11.8 / 11.8-seed2024 / 47 /
188x). Two things worth keeping: below ~11.8x the ladder is a **noisy plateau**, not a ramp —
the 1.47x arm is worse than 0.74x on every metric, so sampling only that region concludes the
term barely works. And at 188x the fit **saturates and overshoots** — 0.24° straight-in bank
and a 3.0 % shared share are both past the flown tracks' own values, smoother than reality
rather than closer to it.

Methodology, recorded because it changed several readings: bank skill must be read against the
random-flight floor and same-runway twin that `docs/score_control_arms.py` now prints per arm,
never against 1.0 — doing so also inverted the earlier loss-design conclusion, since the
velocity dose frozen into simple-v2 is the only one below the floor. The twin is a yardstick,
not a bound (the 47x arm exceeds it). And at n=1404 the paired sign test returns p = 3e-16 for
**pure seed noise**, so its p values mean "reproducible direction", never "large effect".

Not done: KRDU only, `val` split only, and 47x has a single seed — the 11.8x seed pair shows
seed noise is 3-8x smaller than the dose effects, which is an inference, not a measurement.
Artifacts for all eight arms (checkpoints, 1404 predicted flights, evaluation reports,
comparison CZML) are under `4dTrajectory/outputs/KRDU/experiments/imitation_design/` and
published to the viewer. Full write-up:
`4dTrajectory/ts_transformer/docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md` §12.

### 2026-08-19 — the optimizer pipeline: two blockers, a bounded population, and eight cost levers

Audit of `run_scenario_optimization.py` and everything it shells out to, then the fixes.
Measured throughout on the real harvest (42,725 rostered arrivals across 5 airports) rather
than reasoned about; every number below was reproduced end-to-end.

**Blocker 1 — `runway_cons` evaluation crashed on the first record, on every runway.**
`_snap_target_to_procedure` moved the constrained solve's target onto the procedure
document's last waypoint. That waypoint and the arrival manifest's `runway_target` are two
renderings of the SAME CIFP threshold and round differently — measured 0.05–0.22 m over the
25 runways in service (KRDU 32 = 2.98 m, KSMF 35R = 39.45 m, neither in the arrival set).
On 2026-08-17 `evaluation.arrival._require_target_agrees_with_runway_data` gained its
POSITION half at a 1 cm tolerance, so **25 of 25 runways failed**. It went unnoticed because
that day's regeneration covered observed and prediction records only — the CHANGELOG entry
says so: "There is currently no optimizer comparison tree published at all". `run_for_airport`
uses `check=True` with no handler, so the first `runway_cons` cell would have aborted the
whole sweep about 2 h into KMSY.
*Fix:* the snap is gone. The scenario target is already the authoritative
`harvest.airports.Runway` point (that is why the unconstrained `runway` mode always passed),
so the constrained path now keeps it and VALIDATES the procedure against it at the
optimizer's own `_FRAME_ANCHOR_TOLERANCE_M` (150 m) — the same displaced-threshold
mis-anchor the snap existed to catch (KSJC 12L was 390 m off against the NASR config) still
fails, loudly. `evaluation/tests/test_pipeline_integration.py` now pins both ends together
across all of KRDU's runways; before, `_snap_target_to_procedure` was unit-tested only in
isolation and no test ran a constrained record through `python -m evaluation`.

**Blocker 2 — the fitted-ADS-B dataset could not be built for 4 of 5 airports.**
`build_scenario` raises when a flight has no usable `final_approach` fit, which aborts the
whole airport. Measured: **35 flights of 42,725 (0.08 %)** — KMSY 1, KRDU 1, KSJC 25,
KSMF 8, KSTL 0. So 0.08 % of the fleet blocked 82 % of the dataset.
*Fix:* `UnusableFittedApproach` (its own type, so nothing broader is swallowed) and a new
`flight_scenarios.dataset` batch layer that drops exactly those flights, names each one, and
writes the accounting to `<scenarios>.selection.json`.

**Blocker 3 — the run did not fit on the disk.** At the full population the artifacts came
to ~36 GB against 17.8 GiB free. `run_scenario_optimization.py` now estimates the footprint
from measured per-flight sizes and refuses to start rather than filling the filesystem
30 hours in (`--skip-space-check` overrides).

**Population.** `--max-per-runway` (default 2000 in `prepare_scenario_inputs.py`) keeps N
arrivals per runway, evenly spaced over landing time so a capped runway still spans the whole
harvest window. The rostered fleet is wildly unbalanced (KSJC 30L 9,603 vs 12L 14), so an
uncapped batch spends most of its compute re-measuring two runways. At 2000:
**23,453 flights / 70,359 solves** (KMSY 3534, KRDU 7491, KSJC 3543, KSMF 3737, KSTL 5148),
down from 42,725 / 128,175. The selection is derived from the ROSTER only — no source track
is opened for a discarded flight — so it is target-independent, both prepared datasets pick
the same flights, and KRDU's build dropped from 2.4 GB / 24 s to 0.9 GB / 7 s.

**Cost levers, measured.** 120 random KRDU arrivals, `--jobs 20`: runway 159 s wall / 18.1
CPU-s per flight, fitted_adsb 146 s / 17.6, runway_cons 175 s / 9.7.

1. **IPOPT iteration cap is the dominant lever and had no knob.** Serial A/B: the 8 flights
   that ended `Maximum_Iterations_Exceeded` cost 448 s (~56 s each, the full 3000-iteration
   budget); 8 that solved cost 45 s (~4.3 s each). **6.7 % of the flights, ~48 % of the CPU.**
   `--max-iterations` is now plumbed from both runners through both solve paths, and is
   recorded in `summary.json`'s `optimization_config` — a lower cap converts slow successes
   into failures, so it is a different experiment and `--skip-optimize` will not reuse across
   the change. The DEFAULT is unchanged at 3000: that is a research decision, not a
   performance one.
2. **`--jobs`.** The library auto is half the cores (right for a call that should leave the
   box usable); the pipeline driver, which owns the machine for the batch, now defaults to
   `cores - 4` (24 here) instead of 14.
3. Population cap, above.
4. **`--rollout-dt`** exposed on the runner (default unchanged, 0.5 s). The simulator array
   is ~75 % of every `*_states.json`; 1.0 s takes the estimated 2000/runway footprint from
   16.6 GiB to 10.4 GiB.
5. **The observed reference track was written twice.** The fitted-ADS-B and runway datasets
   reference the same flights and their reference records differed only in `target_state` —
   ~200 bytes of a ~67 KB record. The records now quote a shared sibling
   `shared_references/observed_tracks/` store through the contract's existing `states_ref`
   indirection, which `evaluation.records.load_record` already resolves. Measured
   134 KB/flight → 64 KB. Cache contract bumps to
   `optimization-references-v3-shared-tracks` and hashes the track alongside the record; the
   store is swept against the union of all sibling reference dirs, never one dataset's roster.
6. **`python -m evaluation` materialized every record.** `load_records` resolved each
   `states_ref` into the full state list and held the batch — measured 0.5 MB per record, so
   ~7 GB on uncapped KRDU. `summary_row` already carries `arr_airport`, so contexts resolve
   from the roster and the records stream (`iter_records`). `evaluation.visualize` does the
   same in two passes, remembering only which FILES were drawable and reloading the sampled
   30. Verdicts identical (48/48, 44 pass) on the A/B.
7. **`--max-groups-per-czml`.** One CZML per runway at 38–54 KB/flight makes a 2000-flight
   runway a single ~100 MB file. The frontend already loads CZMLs named by each index group's
   own `czml` field, so splitting is transparent to it.
8. **Resilience.** `--resume` reuses complete record pairs for scenarios in the current
   roster and solves only the rest (`_clear_stale_records` still sweeps orphans — resume
   narrows which files survive, it never turns the sweep off); summary counts now come from
   the roster rather than from what this process happened to write, and a roster that ends up
   incomplete raises. `--continue-on-error` keeps the sweep going past one failed cell and
   names the casualties at the end.

**Serialized precision — 30 % of the footprint was decimal digits nobody reads.** Asked why
the run needed 16.6 GiB, the answer turned out to be float text: a single state row is 185
bytes at full repr (`"lat":35.766821578167715`, 17 significant digits for a quantity whose
ADS-B source resolution is metres), and the timed arrays are ~98 % of a record. Records now
serialize at a declared precision (`evaluation_export.STATE_DECIMALS`: 1.1 mm position, 1 ms
time, 0.1 mm/s speed, 2e-9 rad angles) and the comparison CZML mirrors it. Measured: records
**31 %** smaller, reference + observed track **21 %**, comparison CZML **40 %**, and the
2000/runway estimate **16.6 GiB → 11.7 GiB**. A/B on a KRDU batch: **0 verdict or
event_status changes**, largest deviation difference 0.63 mm — four orders below the
metre-scale gates.
Two things this deliberately does NOT touch: `initial_state`/`target_state`, because
`_require_target_agrees_with_runway_data` measures them at **1 cm** and that budget cannot
absorb a rounding step; and `final_time_s`, which is now read back OFF the serialized array
rather than recomputed — the contract requires it to equal `states[-1]["t"]` to 1e-6, and
writing the two independently made `record_from_dict` reject the entire batch on the first
record (caught immediately, which is the contract doing its job).

**Also fixed:** every optimizer evaluation row reported `flight_key: null` while observed
rows carried it — `build_scenario` never copied it onto `scenario.source`. It does now
(only when the flight has an `id`, since `flight_key`'s fallback is a list index this
function does not have).

**Verified end-to-end** on 48 KRDU flights (cap 12/runway, binding on all four runways)
through the real runner: 3/3 cells complete, `runway_cons` 44 pass / 4 fail — the mode that
could not be evaluated at all before. Resume re-solved 6 deleted records in 4.15 s and
reproduced the same 47/48. Suites: 1091 passed, 13 failed — the same 13 pre-existing
failures as before the change (the documented numpy `test_optimizer` one, 11 `test_ts_pipeline`,
1 `test_download_landings`), plus 153 aeroviz-4d python.

### 2026-08-19 — Viewer: the predictor-input window takes its forecast's verdict colour

**Problem.** In Observe → prediction comparison, a group draws as two paths: the predictor
input window (`look-`) and the forecast it produced (`pred-`). The forecast was painted from
its terminal verdict — green pass / red fail / gray indeterminate — while the input window was
painted a fixed faded purple. Purple is a colour nothing else in the legend uses, so the input
half read as a THIRD kind of result rather than as the first half of the track it belongs to,
and a viewer could not tell at a glance which input fed which forecast in a crowded scene.

**Change (frontend rendering only).** `applyComparisonRenderModel` now resolves the outcome
colour for `look-` exactly as it does for `pred-`, at the kind's own alpha:
`predictionOutcomeColor(status, COMPARISON_KIND_ALPHA[kind])`. Both halves share the group key
(`groupOfEntityId` strips the prefix), so one comparison-index entry colours both. Hue is now
the group's verdict; **alpha alone** (85/255 vs 225/255) separates input from forecast. The
purple `COMPARISON_KIND_COLORS.lookback`/`.predicted` survives only as the no-verdict fallback,
which is why the two constants are deliberately equal.

The "Predictor input" checkbox swatch follows: `comparisonKindSwatch` now returns the
green/red/gray split gradient for `lookback` as well as `predicted`, and `ControlPanel` already
multiplies each swatch by `COMPARISON_KIND_ALPHA[kind]`, so the input row renders as the faded
version of the prediction row — the same relationship the two paths have in the scene.

**Not changed: the CZML.** `build_scenario_comparison_czml.py` still bakes `PREDICTION_COLOR`
and `LOOKBACK_COLOR` (both purple) into the packets; the viewer has always repainted prediction
paths from the legend, so the file colours are unobserved in the app but ARE what any external
CZML consumer sees. Logged as a future improvement in the root `README.md` ("Align the
comparison CZML's baked colours with the frontend contract") with two suggested fixes: generate
the builder's table from the frontend legend (the `geoConstants.json` pattern), or bake the
verdict the builder already knows and drop the repaint.

**Tests.** `comparisonRenderModel.test.ts`: the old "lookback keeps the legend purple on an
offTarget group" expectation was the assertion of the behaviour being replaced and is now
"repaints a failed lookback red"; added the no-verdict fallback case, an indexed-status-wins
case, and an explicit input-hue == forecast-hue equality (the previous version compared the
input against the purple constant, which passed trivially because both constants were purple).
81 files / 510 tests pass; `tsc --noEmit` clean.

### 2026-08-18 — ts_transformer: control-mode design pass, first-order control lag, teacher-inverse audit

Three connected pieces of work on `4dTrajectory/ts_transformer`, driven by the 2026-08-17
meeting note (`docs/MeetingNotes/note_8_17.md`).

**1. The control path carried every historical experiment axis; simple-v1 walks one of them.**
Removed, each with the recorded verdict that closed it: control-mixture (paused 2026-07-30),
direct durations (ablated negative 2026-07-30, ADE +21.0 % / FDE +36.6 %), trim-residual
controls (never published a result), the `physical-criteria` and `terminal-state` objectives and
their checkpoint-selection metrics, progressive-N teacher pretraining (rejected 2026-08-02), the
one-off rollout-finetuning gate, and `benchmark_validation_execution.py` (zero references, zero
tests). That is −2 of 5 tracking objectives, −2 of 5 selection metrics, −1 of 3 duration
parameterizations, the whole `control_value_parameterization` axis, and one of three prediction
outputs. `oracle_teacher.optimization` also owned a second stage type that only converted itself
into `ControlTrainingStage`, and the 60/120/240/full schedule was written out twice; both
runners now call one `teacher_optimization_stages()`.

Two live bugs surfaced only once the modes hiding them were gone: `run_ts_pipeline` had no
`objective_label` entry for `true-time-position`, so `PredictionPlan` raised `KeyError` on the
current frozen recipe; and the arc-length recipe builds a **365-byte** directory name against a
255-byte path-component cap (names over the cap now keep their head and end in a digest of the
whole name). `TSConfig.from_dict` restated the same missing-field check 25 times and is now one
table.

**2. A switchable flight model with first-order control lag.** The point-mass model applies a
piecewise-constant control instantly, so a learned schedule steps the bank angle N times across
an approach: curvature is discontinuous at every segment boundary and the implied roll rate is
unbounded. The meeting asked for continuity. The three controls become states chasing their
command — `d(mu)/dt = (mu_cmd - mu)/tau_mu`, likewise thrust and load factor — and
`aerodynamic_model/torch_lag_dynamics.py` **wraps** `transport_chart_rhs` rather than restating
it, so the force equations, stall handling, WGS84 transport term and chart projection are
literally the same code.

Measured both directions, because "it reduces to the old model" is a claim that has to be
checked: at `tau = 0.1 s` the two trajectories end **within 0.5 % of path length** and the gap is
first order in tau; at the 2 s default they differ by **~3 km over a 240 s rollout**. So it is a
materially different model, not a smoothing pass.

Switchable on one axis (`control_dynamics_model` ∈ `point-mass` | `first-order-lag`), orthogonal
to the state representation, with a `simple-v1-lag` recipe that is simple-v1 with that one field
changed so a paired comparison measures the flight model rather than a bundle. `tau_bank` joins
the CV grid (0.5/1/2/3/4 s) and is dropped as inert under `point-mass`.

**New numerical constraint, worth remembering:** explicit RK4 on `y' = -y/tau` is unstable above
`h/tau = 2.785`, so a swept time constant shorter than the integrator step produces **NaN, not a
degraded rollout**. `TSConfig` refuses it at construction rather than letting it be discovered as
a dead training run.

**3. Controls are dimensionless, and the thrust floor is negative.** `thrust_fraction = T/T_max`
puts the three controls on one magnitude and makes the same box mean the same thing on every
airframe — one sigmoid output used to mean 100 kN on a small jet and 400 kN on a heavy, so a
teacher schedule or a learned bias was not transferable across the fleet. Newtons now appear in
exactly two functions: into the dynamics, and out to the evaluation record (whose contract is
unchanged and shared with the CasADi optimizer).

The floor moved from 0 to **−0.2**, because a real approach needs net-negative force: idle thrust
plus the drag of speedbrake, flaps and gear, none of which the clean-configuration polar models.
This was measurable. On the KSJC outer-train cohort (24 flights × 64 segments), **39.7 % of
inverted teacher thrust segments pinned at the 0 N bound** — the teacher structurally could not
reproduce the deceleration the aircraft actually flew. After: **0.33 %**, with a median required
thrust of 3.6 % of installed (idle) and p5 at −12 % (drag augmentation). `flyability.py` already
treated negative required thrust as a SOFT violation for exactly this reason. The optimizer's own
`make_control_bounds` is deliberately NOT changed — that box is a flyability claim, this one is a
learned head's search space — so no optimizer artifact is restaled.

**4. The teacher inverse is now the inverse of the configured forward model, by construction.**
This was the specific thing asked for, and the audit found real problems.

A schedule solved against equations the training rollout does not integrate is finite, bounded,
the right shape, and its own optimizer reports a falling loss. It simply reproduces nothing, and
nothing downstream can tell. So `control_inverse_dynamics.py` registers each inverse under the
SAME config key as its forward model, and `tests/test_control_inverse_dynamics.py` closes the
loop numerically for every registered model: roll a known schedule, invert the dense result,
require the schedule back (recovered to 5e-3 in thrust fraction and load factor, 0.5° in bank).
A model added without an inverse fails at registry lookup.

Found and fixed in the old path:

- `build_inverse_dynamics_target` hard-unpacked a 7-field batch. `dataset.batch()` returns 7
  fields only under `control_state_loss_grid='fixed-dt'`, so under **every native-grid recipe,
  simple-v1 included, the teacher builder died on a bare tuple `ValueError`**. It only ever
  worked because `run_ts_oracle_teacher_optimize.py` builds its own `custom` fixed-dt config.
- It rebuilt reference velocities with a hardcoded `smoothed-position-difference` regardless of
  `config.reference_velocity_source`, i.e. it inverted a different velocity definition from the
  one the supervision targets are built from.
- The inversion was a hand-written numpy inverse of the flat-ENU RHS with no link to
  `config.control_dynamics_backend`. It agreed with the transport-chart backend by accident
  (measured 0.5 N / 1e-6 load factor), not by construction — and adding lag states would have
  broken that silently. It now adds back the `omega x v` term the chart RHS subtracts.
- The lagged inverse is the actual-control inverse plus `u_cmd = u + tau * du/dt`, sharing its
  first stage, so the two cannot drift apart.

The lagged model's actuator initial condition comes from the same inversion applied to the
observed lookback (`dataset.anchor_controls`, 11 samples ≈ 20 s), so it reads no future and the
"inverse must match the forward model" requirement is structural rather than teacher-only.

**Measured on real data (2026-08-19), paired.** Both frozen recipes trained 180 epochs on
KSJC with the same manifest, roster and `split_seed`, then predicted, evaluated and published;
both arms scored the SAME 1083 validation flights, so the per-flight sign tests below are
paired. Artifacts in `4dTrajectory/outputs/KSJC/experiments/flight_model_paired/`, published to
`aeroviz-4d/public/data/airports/KSJC/comparison/ts_ksjc_flight_model_{point_mass,first_order_lag}/`.

- **The lag does what it was added for.** jerk p95 **−28.0 %** (lower on 99.8 % of flights,
  p≈1e-320), turn rate p95 −6.0 % (86.4 %, p=5e-141), acceleration p95 −5.5 % (77.4 %, p=2e-76).
- **Without costing accuracy — it gains a little.** ADE **−3.4 %** (better on 67.4 % of flights,
  p=7.9e-31, bootstrap 95 % CI on the mean delta [−32.7, −17.0] m). This is what separates the
  result from the blandness trap the README documents for flyability, where a worse predictor
  scores better by drawing straighter lines: here smoothness and accuracy move together.
- **The endpoint does not move.** FDE and arrival-endpoint error are ties (49.1–49.2 % win rate,
  p=0.58–0.63). Coherent: a lag smooths the turn geometry along the way, not where the model
  believes the runway is.
- **Read the smoothness as a modelling artifact removed, NOT as "more realistic".** Both models
  were already smoother than the flown tracks (jerk p95: observed 4.253, point-mass 0.30×, lag
  0.21×), so the lag moves them further from observed statistics — it is closer to observed jerk
  on 0.2 % of flights. Observed jerk from 2 s ADS-B positions triple-differentiated is dominated
  by quantisation noise and is not a target to chase; what the lag removes is the curvature
  discontinuity a piecewise-constant control creates at every segment boundary.
- **Gates are unchanged: 0 pass / 1083 fail on both arms**, consistent with the standing
  "forecast ≠ certifiable approach" finding. Changing the flight model does not change that.

**τ_bank cross-validation: not resolved.** 3 folds × 36 epochs over {0.5, 1, 2, 3, 4} s put the
best (2.0 s, 1014.2 m) and worst (4.0 s, 1072.3 m) **5.7 %** apart against **11–23 %** fold noise;
τ ∈ {0.5, 1, 2} lie within 1.5 % and τ=1.0 beats τ=2.0 on 2 of 3 folds. By the recipe's own rule
(<2 % ⇒ no reliable difference) τ=2.0 is a defensible default, not a CV-selected value.

**A CV trap worth remembering: `DEFAULT_CV_PATIENCE = 6` is too small for this recipe.** BOTH
flight models pass through an early ADE transient while every loss component falls monotonically
— on the unchanged point-mass recipe: best 1534 m at epoch 2, bump to 2132 m at epoch 6, then
1268 m at epoch 13 and still falling. Patience 6 stops inside it, and the first τ sweep had to be
discarded: the same τ=0.5 scored 1674.6 m on a fold early stopping caught and 1234.4 m on one it
did not — a 26 % difference that was pure stopping artifact. The frozen recipes' own patience=20
clears the ~7-epoch bump; a CV sweep needs patience raised to match.

**Performance.** The lag step originally ran eager on CUDA while the point-mass backends cached a
`torch.compile`d one: ~95 s vs ~15 s per epoch. The blocker was passing `state_scale` through a
closure — `torch.compile` caches per code object, so a fresh closure per rollout rebuilt the
kernel every batch. Moving it into `step_context` took the lagged epoch to **8.2 s (11.6×)** and
the τ sweep from ~15 h to ~40 min.

**Stale artifacts.** Every existing control-output checkpoint. The control contract changed units
(newtons → fraction) and `TSConfig` gained required serialized fields, so `load_checkpoint`
refuses them rather than mis-scaling thrust by five orders of magnitude. `state`-output
checkpoints are unaffected. No optimizer, harvest, evaluation or comparison artifact changes.

### 2026-08-17 — Comparison references were the wrong window: full track vs model arrival slice

**Symptom.** In the comparison overlay the white observed reference did not start at the
same time or the same place as the `look-`/`pred-` group beside it. Measured on the KRDU
05L validation batch (471 groups): the group's first sample sat a median **5055 m** from
the reference (p75 5281 m, p95 47.1 km, max 54.9 km).

**Root cause — two time origins, only one of them reconciled.** Three timelines exist and
the publisher accounted for two:

1. a stored track's `samples[i][0]` is relative to first reception (`store.track_record`,
   absolute time in `start_time_utc`) — this is what `trajectories.czml` and the
   `/trajectories` backend served;
2. the model **arrival slice** is rebased at `harvest/arrivals.py` `load_arrival_flights`
   (`t0 = waypoints[0][0]`, `sample[0] - t0`), so every scenario, optimizer record and TS
   record has `t = 0` at the 25 km terminal-ring entry, and `t0` is discarded;
3. a prediction record rebases again to the anchor, recording the shift as
   `source.anchorTimeS`.

`build_scenario_comparison_czml` added ③ back (correctly — that was the 2026-07-20 anchor
fix) but nothing ever added ② back, so every group rendered `t0` early. Measured `t0` over
300 random KRDU arrivals: median **45.1 s**, p25 34.3, p75 55.6, p95 123.1, max 526.3 s.
The same seam applies to optimizer groups (`opt-`/`sim-` also start at ring entry); no
optimizer category happened to be published at the time, so it only showed on predictions.

Nothing downstream could detect it: both timelines start at `t = 0`, both name the right
flight, the schema is satisfied, and the drawn result reads as model error rather than a
publication bug. Proof that the two were the same measurement: fitting a per-flight pure
time shift dropped the lookback↔reference distance from a median 4900 m to **13.7 m** (2 s
resampling error).

**Fix — align the reference to the modeling window, at READ time.** The pre-entry segment
is not model input, not a supervision target and not evaluated; drawing it as the white
"truth" beside a forecast invites reading it as something the model failed to produce. So
the reference is now the arrival slice on the arrival origin, rather than the group being
pushed out onto the full track's origin.

- `aeroviz_backend/observed_trajectories.py` gained `window` ∈ `full` | `arrival`. `full`
  (default) is unchanged: the complete reconstructed track, rostered by
  `tracks/manifest.json`, for Observe/Baseline. `arrival` rosters from
  `arrivals/manifest.json` and builds flights through **`load_arrival_flights` itself** —
  the same loader the scenario/optimizer/training paths use — so there is no second
  implementation of the slice to drift from, and the source-hash check plus identity round
  trip come along for free.
- `tracks/` is untouched and no artifact is written: the slice is taken at read time, the
  same rule the altitude-outlier repair follows.
- Response schema bumped to `observed-trajectories-v2` with `trackWindow` echoed. The bump
  is load-bearing: a v1 backend ignores an unknown `window` argument and answers a
  comparison-reference request with full tracks, reproducing the bug silently. The
  frontend refuses anything but `arrival` for the comparison reference.
- `build_scenario_comparison_czml._require_reference_aligned` pins the embedded-reference
  path (`include_reference_entities=True`, currently unused in publishing) at 50 m — inside
  the gap between resampling noise (~14 m) and a wrong window (≥5 km).

**Verified end-to-end on real data**: over all 471 KRDU 05L groups the group-start-to-
reference distance is now **0.0 m** for every group (bit-identical samples), against a
median 5055 m before.

**No artifact is stale.** Published comparison CZMLs contain only `look-`/`pred-` (the
publisher passes `include_reference_entities=False`) and the reference is served live, so
restarting the backend is the whole deployment — no re-publish, no re-predict, no
re-optimize.

### 2026-08-17 — `final_approach` / `evaluation` design pass: one event validator, one lateral bound, authoritative threshold frame

Review of the two packages for design and defensive code. Report schema bumps to
`terminal-approach-evaluation-v5`.

**What "stale" means here, measured — no number changes.** Every shipped
`evaluation_report.json` must be regenerated, but only because its *shape and version*
changed; re-running produces the same verdicts and the same deviations:

- The effective lateral bound is unchanged. A shipped KRDU v4 row reads
  `{guidance_lateral_m: 53.375, runway_lateral_m: 22.86, effective_lateral_m: 22.86}` —
  the runway term won, as it does at all 26 thresholds — and v5 publishes `lateral_m:
  22.86`. Same number; what disappears is the `53.375`, which was the 2×-wrong half of an
  already-inert term.
- The authoritative-frame change (C) moves nothing on disk. Probed 500 observed + 500
  predicted KRDU records against the CIFP-resolved context: worst `target_state` offset
  from the authoritative threshold is **0.0000 m** (altitude likewise), so no frame origin
  moved and every deviation is bit-identical. Observed records carry no `target_source` at
  all and therefore take the STRICT branch — and pass it.
- So KRDU observed re-runs to the same `{pass: 14168, fail: 270, indeterminate: 1}`.

What actually changes: `schema_version`, the `bounds` / `resolved_limits` field names, the
new `methodology.terminal_lateral`, and `evaluation_context_fingerprint` (context v2 gained
`threshold_lat`/`threshold_lon`). **Comparison CZML files do NOT need rebuilding** — their
geometry comes from states and their `status`/colour from verdicts, and neither moved; only
`evaluation_report*.json` and the `evaluation.schemaVersion` string inside
`comparison_index.json` do.

Scope on disk: all 34 published comparison trees are PREDICTION trees (`ts_pooled_*`,
`prediction_ts_itr_*`, `experiment_*`) plus the per-airport `observed/` report. There is
currently **no optimizer comparison tree published at all**, so the optimizer batches were
not part of this regeneration.

**Regeneration completed 2026-08-17.** Everything below is v5 on disk:

- 5 observed batches via `--evaluate-only --no-czml` (harvest + published copy). Verdicts
  reproduce the historical table exactly — KRDU 14168/270/1, KMSY 3892/257/1, KSJC
  11144/7/6, KSMF 4221/8/2, KSTL 8485/280/4 — confirming the "no number changes" claim
  end-to-end. CZML deliberately not re-rendered (unchanged, and already post-altitude-filter).
- 34 prediction batches re-evaluated. 14 of them had their records archived into
  `prediction_records.tar.gz`, so `python -m evaluation` failed on them until the tarball was
  extracted, evaluated, and the extracted members removed again to restore the archived
  layout (the tarball does carry the `*_eval.json`, 2166 of them in the KSJC/val case).
- 34 published comparison trees refreshed report-only, following the publisher's own atomic
  pattern (new generation-named report → atomic index replace → prune) and using its own
  `publish_evaluation_report` / `evaluation_batch_stats` / `prune_unreferenced_outputs`, so
  the shapes cannot drift. The index carries no source path and the itr/ptst variants of a
  split share a flight set, so each tree was joined to its source by CONTENT — the exact
  per-row `(file, cross_track_m)` fingerprint — which matched all 34 uniquely, 0 ambiguous.
- 5 `lateral_pass_eligibility.json` rosters rebuilt (KRDU's was deleted by the arrivals
  rebuild, which clears its directory; the other four pinned the old manifest hash and v4).

Two things this surfaced, neither caused by the v5 work:

- Re-running the harvest **changes every `arrivals/manifest.json` hash**, because the
  altitude-outlier filter block (`altitude-outlier-filter-v1`) had never been written to
  them. The reports also shed five stale `deviation` keys (`glidepath_deg`, and the four
  `*_sigma_m`/`*_interval_m` fields) that the code had already stopped emitting. The on-disk
  artifacts were behind the code in more ways than the schema version.
- Prediction **checkpoint provenance was already broken before this session**: the KSJC
  checkpoint pins arrival-manifest `687f5c6c1d94bb54` while the tree held `362b0dd97ff823ef`
  even before the re-run. A full republish (`publish_ts_experiment_trajectories`) preflights
  that hash and would refuse — which is why the refresh above is report-only. Anything
  needing a real republish must re-run prediction against the current manifests first.
  `publication.json`'s `evaluation.schema_version` therefore still reads v4: it records the
  publish run that actually happened, and nothing validates it.

**Contract consolidation (behaviour-preserving).**
- The `observed_threshold_event` schema had TWO hand-rolled validators — `harvest
  .threshold_event.require_current_threshold_event` and the inline block in
  `evaluation.arrival._observed_arrival`, ~65 lines each over the same payload, free to
  drift. One `final_approach.event_contract.validate_event(event) -> status` now owns
  the schema; each side keeps only its own identity binding (producer: `Runway` frame
  fingerprint + snapshot; evaluator: `AssessmentContext` runway + fingerprint). Both run
  identity first, payload second, so a stale artifact reports as stale.
- `INBOUND_TOLERANCE_M = 100.0` existed twice with a "same as the final-approach fitter"
  comment; it is now imported from `final_approach`.
- Removed: `evaluation.metrics._validate_deviation` (re-checked values the same function
  produced three lines earlier); the 45-line self-consistency re-derivation in
  `_validated_observed_availability` (the counts are measured upstream from a roster
  evaluation never sees — now copied verbatim like `observed_threshold_event`, with only
  the denominator LABEL checked); `AssessmentContext.from_dict` (unused); the
  `source.get("arr_airport") or source.get("airport")` fallback duplicated in two modules
  (no producer has ever written `"airport"` — replaced by `TrajectoryRecord.airport`);
  `_line_inliers`' rejected-index half (both callers discarded and recomputed it); the
  provably unreachable span check after `_straight_final_suffix`; and the type-shape half
  of `AssessmentContext.__post_init__` (kept: benchmark whitelist, since `limits()`'s
  second branch is a fallthrough; positive runway width; NaN; LPV completeness).
  `METHODOLOGY` moved to module level. Net −115 lines with the new validator counted in.
- Single-statement functions that only added an indirection: `arrival.subject_of`
  (a `source["subject"]` lookup behind a cross-module import and a package export — now
  read directly at its two call sites), `stats.mean` (a reimplementation of
  `statistics.fmean`), and `thresholds._require_finite` (a module function taking the
  one object that called it — inlined into `__post_init__` over a `_FINITE_FIELDS`
  tuple). An AST sweep over both packages checked the rest: the ~13 remaining
  one-statement definitions are dataclass properties naming a domain quantity
  (`SegmentFit.cross_at_threshold_m`, `TrajectoryRecord.solved`), `to_dict`
  serializers, or public helpers with three or more callers — kept. `event_contract
  ._source_sample_range` returned a tuple no caller used; it is now
  `_require_source_sample_range() -> None`.

**Lateral criterion: one honest bound (methodology change).**
`limits()` computed `min(guidance, runway_width/2)` where guidance was the LPV course
width `/2` or LNAV 0.15 NM. Two findings: the LPV value 106.75 m is already a semiwidth,
so halving it published `guidance_lateral_m` wrong by 2×; and measured over all 26
thresholds at the five airports the guidance term bound **0 times** (2.3–18× wider than
the runway). `ResolvedLimits` now carries a single `lateral_m` = runway half-width,
tagged `LATERAL_CRITERION_ID = "runway_half_width_at_threshold"`, and the report's new
`METHODOLOGY["terminal_lateral"]` states the claim boundary: landing geometry, not
navigation containment. `lpv_lateral_fsd_m` → `lpv_course_width_m`, kept as procedure
provenance that bounds nothing. Lateral is now never indeterminate once a crossing was
measured, so the composite's only indeterminate route is a missing vertical reference.
No verdict changes (the min already selected the runway half-width everywhere).

**Deviations are measured in the authoritative runway frame.**
`_computed_arrival` built its `RunwayFrame` at the record's own `target_state` lat/lon
and cross-checked only the altitude against the published LTP+TCH. `AssessmentContext`
gains required `threshold_lat`/`threshold_lon`; the frame origin is now authoritative,
and `_require_target_agrees_with_runway_data` checks position AND altitude at 1 cm —
gated on `source.target_source == "runway_threshold"`, because `fitted_adsb_crossing`
and `track_end` (both in `run_scenario_optimization.ALL_MODES`) aim elsewhere by design.
An undeclared `target_source` gets the strict reading. Two consequences worth recording:
a fitted-ADS-B-target record used to be graded against its own flight's fitted crossing
(~0 lateral by construction), and the repo's two threshold sources — NASR
`runway_thresholds.json` vs CIFP Path Point LTP — sit **6.69 m apart at KRDU 05L**, which
nothing had ever noticed. The real pipeline is internally consistent; only
`ts_transformer/synthetic.py` builds on the NASR point, so its test context pins those
coordinates explicitly.

**Follow-up pass: the v5 bump reached the producer only, and two guards were over-deleted.**

*Alignment.* `REPORT_SCHEMA_VERSION` v4 → v5 landed in `evaluation/metrics.py` alone.
`4dTrajectory/ts_transformer/lateral_eligibility.py:18` and
`aeroviz-4d/src/data/evaluationReport.ts` kept their own v4 literals, so
`build_lateral_pass_roster` raised on every regenerated report and the frontend's
`isEvaluationReport` rejected them — the panel surfaces `"evaluation report is malformed"`,
which is a misleading message for what is purely a version disagreement. Both suites were green
because their fixtures also said v4: **a version pinned in the fixture is a version the
test cannot check**, and that is the whole reason this shipped. The ts seam now imports
`REPORT_SCHEMA_VERSION` (it is the one ts module allowed to know evaluation policy — model,
loss and loader code still must not); the frontend exports
`EVALUATION_REPORT_SCHEMA_VERSION` as a declared MUST-match mirror; all four fixtures
import the constant. `thresholds.CONTEXT_SCHEMA_VERSION` also became
`terminal-assessment-context-v2`, since its hashed payload gained `threshold_lat/lon` and
renamed `lpv_lateral_fsd_m`. Docs realigned: `FINAL_APPROACH_VERDICT_STANDARD.md` §3.3 now
documents the single runway-half-width rule and why the guidance term went (its §2/§5 v4
history is kept as history), plus `evaluation/README.md`, `BUG_FIX_GUIDE.md`, the zh
implementation doc and the ts evaluation doc. `EvaluationReportWindow.tsx` still advertised
"the tighter of the guidance bound and runway half-width" — the exact claim this pass
deleted — while rendering a report whose `METHODOLOGY` said the opposite; its test now
asserts the new claim and asserts the old wording is ABSENT.

*Over-deletion (both verified by running the code, then covered by tests).*
- `validate_event` compared `ESTIMATED_OBSERVABILITY_BY_METHOD.get(method) != observability`.
  With NEITHER field present that is `None != None` → False: an `estimated` event carrying
  no method and no observability validated clean, then fell through to the censored branch
  and was graded as a real crossing. Each single-field case already failed, which is why
  the gap survived. Now the lookup must resolve.
- `record.source.get("target_source", THRESHOLD_TARGET_SOURCE)` returns `None`, not the
  default, for a key present with a null value — so an explicit null took the "aims
  elsewhere on purpose" early return, bypassing the cross-check exactly where the docstring
  promised it could not be bypassed. Audited: all 144,764 shipped `*_eval.json` carry an
  explicit `"runway_threshold"`, so nothing on disk was mis-graded.
- Restored the positivity check on `threshold_crossing_height_m` (dropped with the type-shape
  half of `__post_init__`). It sets the vertical REFERENCE PLANE, published values run
  15.27–18.11 m, so a zero would move the plane by most of the ±22 m window with every
  verdict still looking clean — the same "parsed FAA data, not a Python literal" argument
  that kept the NaN check.

*Kept deleted, on review:* the `hae_minus_msl_m` presence check (the harvest writes it
unconditionally; `KeyError` is loud), the non-dict event check (a non-dict event is a corrupt
artifact), the empty-`frames` guard in `assign_runway` and the duplicate `max_tracks` guard
in `visualize` (both contracts documented, both still crash, and argparse rejects the latter
first). `METHODOLOGY` being shared by reference is real aliasing but nothing mutates it —
adding a deepcopy would re-add the defensive code this pass removed.

*Symmetry.* `evaluationReport.ts` also stopped re-deriving the observed-availability
arithmetic. `harvest/observed.py` computes `event_unavailable` and `event_estimated_rate`
FROM `denominator` and `estimated` in the same expression, so those are identities, not
invariants — and re-checking them cost a silent whole-report rejection. Both sides now check
only the denominator LABEL, which is the one field that says which population the rate
describes.

Out-of-scope findings from this pass are recorded in `docs/code-health-followups.md`
(duplicate `_iso` in harvest, `summary_row`'s explicit JSON nulls, the stale v2 fixture in
the comparison-CZML tests, and the 12 pre-existing `test_ts_pipeline` failures).

Deferred pending review: the dead `max(seed_scale, floor)` in `fit._straight_final_suffix`
(the "adaptive" residual limit is a constant 31.9 m for every track), and replacing the
O(n²) exact Theil–Sen seed in `_median_line`.

Verified: 908 passed across the Python suites (13 failures all pre-existing — 12 in
`test_ts_pipeline`/`test_download_landings`, plus the known numpy 2.x
`test_fixed_time_objective_weights_control_effort_at_one`); frontend `tsc --noEmit`
clean and 508 vitest tests passing.

### 2026-08-17 — ADS-B altitude outliers filtered in the view, not in the tracks

**Symptom.** Observed trajectories rendered with needle-shaped vertical peaks: single
samples reporting an altitude nowhere near their neighbours. Measured extremes across the
five harvested airports are 20 147 m between neighbours at 724 m and 35 189 m at 556 m.

**What was rejected.** A stop-gap `fix_altitude_spikes.py` edited `tracks/*.json` in place.
That breaks three contracts at once — `arrivals/manifest.json` pins every source track by
SHA-256 (the loader refuses a changed file), `--reclassify-existing` re-derives assignment
from those exact samples, and `source_integrity.retained_rows` counts them. It also missed
the point: `tracks/` is the sensor reconstruction, and a repair is a property of the view.

**What shipped.** `trajectory_data_process/harvest/altitude_filter.py`, applied where a
stored track becomes a derived view and nowhere else:

- `store.read_track_view` → observed CZML (`czml.observed_czml_flights`, which also feeds
  the backend trajectory sampler) and evaluation records (`observed.write_observed_records`);
- `arrivals.write_arrival_records` / `load_arrival_flights` → all model input (ts training,
  `flight_scenarios`, `batch_benchmark`). Both hash the SOURCE bytes first and filter after,
  so the roster stays a statement about what the receiver recorded.

`store.iter_records` deliberately stays raw — reconstruction and reclassification must see
what was stored.

**Detection.** Deviation from the median of the ±2-sample window exceeding BOTH 100 m AND
`25 m/s × min(adjacent gap)`. Both halves earned their place on the data:

- a chord/jump test (the stop-gap's approach) attributes one bad sample to three, because
  the outlier's two neighbours have a chord running through it — 363 runs of exactly three
  where the truth was 363 isolated samples;
- the 100 m floor sits at 2× the largest residual genuine flight produces. Over 20 851 436
  assigned samples the residual is < 25 m for 20 847 051, 3 625 fall in [25, 50) (the 25 ft
  and 100 ft reporting lattices plus real motion), and only 189 exceed 50 m;
- the rate bound spares 10 real descents that stepped 107–160 m across 9–14 s reception
  gaps, which a bare deviation threshold "repairs" into a lie.

Incidence: **561 samples in 451 of 44 622 assigned tracks (0.0027 %)**; 421 sat inside a
model arrival slice. 479 isolated, longest run six.

**Repair replaces the altitude and never drops the sample.** `landing_sample_index`, the
arrival slice bounds, the threshold event's `source_sample_range` and the
`reported_ground_speeds_m_s` parallel array all index that array; deleting a row silently
renumbers every one of them. Replacement is a linear interpolation in time between the
nearest retained samples; at a track edge it holds the nearest retained altitude
(`held` vs `interpolated`, both labelled in the report).

**Stated, never silent.** `arrivals/manifest.json` and `approach/summary.json` each carry an
`altitude_filter` block (policy + repaired counts), arrival records carry a per-flight
`altitude_outliers`, and `RenderedObserved` carries the render's totals.

**Tooling.** `fix_altitude_spikes.py` deleted; `python -m trajectory_data_process.altitude_outliers`
replaces it as a read-only audit (`--report-json` gives the full per-track trail) plus
`--rerender-czml`, which republishes `public/data/<ICAO>/trajectories.czml` through the
pipeline's own renderer.

**Artifacts.** All five `trajectories.czml` republished. Batch comparison CZMLs resolve
their observed reference by entity id inside that canonical file, so they follow without a
rebuild. Training data needs no rebuild either — `load_arrival_flights` filters on the way
out — so the next dataset build is already clean; `--evaluate-only` is what refreshes the
roster counts and the evaluation records.

**Known gap.** Stored `observed_threshold_event`s were fitted from raw samples during
assignment, before this filter existed. 17 outliers (KRDU 15, KSTL 2) land inside an
event's source range; the audit lists them and `--reclassify-existing` is what re-derives
them.

### 2026-08-12 — Fail-closed pipeline cleaner

- `clean_pipeline_data.py` now requires an explicit airport scope and constructs its
  deletion plan from producer-owned artifact names instead of recursively treating
  `4dTrajectory/outputs` as disposable.
- Downloaded tracks, checkpoints/history, `test_release.json`, formal experiments,
  pooled roots, final-test and ambiguous predictions, parked/manual/unknown outputs,
  tracked/static data, archives, and mixed experiment comparison publications are
  protected.
- Only standalone predictions whose readable metadata explicitly says `split: "val"`
  are eligible. Comparison cleanup requires a readable registry that accounts for all
  content. The canonical observed filename is matched exactly, not by prefix.
- Destructive execution validates the complete plan and stages every selected file on
  the same filesystem; a staging failure rolls the move set back. Safety tests cover the
  allowlist, airport isolation, protected research state, mixed comparison output, exact
  CZML ownership, required scope, dry-run behavior, and rollback.

### 2026-07-23 — Fitted threshold kinematics; preparation/optimization runner split

- Fitted-ADS-B targets now derive `V/psi/gamma` from the same established final-approach
  fit as their threshold position. The along-track rate is fitted over that established
  segment only; the spatial tangent supplies heading and glide angle, so rollout
  or parked samples can no longer create a threshold target with `V=0`. The constant-rate
  helper is the single replacement seam for a future deceleration model.
- The former combined runner was deleted. `prepare_scenario_inputs.py` rebuilds
  arrivals/observed products and writes the two distinct scenario datasets;
  `run_scenario_optimization.py` consumes those datasets and owns optimization,
  evaluation, and comparison-CZML publication. Run preparation first, then optimization.
- `clean_pipeline_data.py` now removes preparation-derived `arrivals/` and `approach/`
  by default while preserving downloaded `tracks/`; `--include-downloads` only expands
  the deletion boundary to measured source tracks.
- Review hardening made datum provenance explicit on fitted results, rejects invalid
  `states_ref` ranges and duplicate source identities, and verifies reference identity
  plus SHA-256 before cache/batch reuse. `--skip-optimize` now validates the complete
  summary/eval/states/reference roster rather than treating `summary.json` as a marker.
- Comparison publication now writes immutable generation-suffixed CZML/report artifacts
  and atomically commits their index last; failures preserve the previous generation.
  The frontend follows the report named by that index, strictly rejects legacy observed
  and comparison manifests, never falls back to embedded references or fixed-name
  reports, and restores canonical entity styles on comparison exit.

### 2026-07-21 — final_approach + evaluation review fixes: observed-aware reporting surfaces, one deviation definition

Code review of `final_approach/` + `evaluation/` (no correctness bugs in the geometry or the
fit statistics; all findings were reporting/duplication/doc-drift). Fixes:

- **The human-facing surfaces caught up with the observed-subject work.** The measurement side
  (`arrival.py`, the report's `observed` block) was done, but `evaluation/visualize.py` and the
  `python -m evaluation` console summary still rendered pre-subject reports: a solve-rate card/
  line (1.0 by construction on observed data — the exact number the subject dispatch exists to
  stop reporting) and no established rate or marginal count anywhere. Both now suppress the
  solve rate for pure observed batches and report `established N/M` + marginal. The deviation
  charts also plotted `solvedRows`, which for observed batches includes not-established rows
  with NO deviation fields — `Math.max(undefined, 0.01)` → NaN → silent bar gaps; charts now
  draw `measuredRows` only. Verdict table gains established/marginal columns + ±95 % bounds on
  the deviations when observed rows exist; not-established rows are styled grey ("no arrival to
  judge"), not red ("judged and failed").
- **`EstablishedCriteria` became reachable**: `evaluate_batch(..., criteria=)` and CLI flags
  (`--fit-window-m/--max-cross-track-m/--glidepath-range-deg/--max-vertical-rms-m`), defined
  ONCE in the new `evaluation/cli.py` and shared by `__main__` and `visualize` so the JSON and
  HTML reports cannot be produced with silently different knobs (the gate flags were previously
  duplicated between the two entry points).
- **One final-state deviation definition.** `metrics.final_state_deviation` + its
  `FinalStateDeviation` dataclass duplicated `arrival._final_state` line for line (no external
  consumers, verified). The single definition now lives in `arrival.final_state_deviation`
  returning `ArrivalDeviation`; `FinalStateDeviation` deleted.
- **`_is_marginal` lateral fold fixed**: `abs(lateral − margin)` folded a signed CI containing
  the centreline past 0, misreading "CI straddles the gate" as a solid verdict whenever
  1.96σ > gate + offset. Lower bound is now `max(0, lateral − margin)`. Unreachable at real
  σ ≈ 1–2 m (needs σ > ~55 m) — fixed for correctness, with a regression test.
- **Dead code removed**: `frame.track_course_deg` / `heading_difference_deg` had no consumer
  anywhere (the promised harvest heading pre-filter never materialized — the arg-min design made
  it unnecessary). `fit_final_segment` now validates `min_samples >= 3` / `min_span_m > 0` at
  the boundary (previously a ZeroDivisionError deep in `_fit_line`). `visualize.RESAMPLE_N`
  deduped into `reference.N_RESAMPLE`.
- Docs: `evaluation` README/`__init__` (which still described the pre-subject package, exported
  nothing from `arrival`) updated; `Assignment.scores` docstring no longer claims rejections
  carry scores; stale CLAUDE.md open item ("observed evaluation designed but NOT built") removed.

Suites: final_approach + evaluation 88 pass, trajectory_data_process 93 pass.

Found while trying to answer "what do the 8260.58D gates score REAL ADS-B arrivals at?" —
the observed baseline the optimizer and the learned predictor are implicitly measured
against, which had never been computed. The naive answer was 1.8 % pass (18/996 KRDU), i.e.
completed, safe airline landings graded as failures. Three independent bugs, all upstream of
`evaluation/`, which is unchanged by this work.

**The measurement that separated them.** Fitting each flight's OWN established final-approach
line (position + cross-track vs along-track, extrapolated to the threshold) instead of reading
`states[-1]`: the fitted glidepath came out 3.02–3.13° at all five airports — textbook — while
the vertical intercept was 20–30 m low. A fleet flying a perfect glidepath to a uniform 25 m
error is not a fleet error; it is a reference error. Lateral was already 3–10 m median, so the
two axes were telling opposite stories and had to be chased separately.

**Bug ① — observed altitude is ellipsoidal, targets are MSL.** OpenSky `geoaltitude` is height
above the WGS84 ellipsoid (HAE); runway thresholds, CIFP altitudes and the gates are MSL. The
gap is the geoid undulation N ≈ −25 to −33 m over the US. Confirmed independently: KRDU's
lowest observed sample is 99.1 m against a predicted field elevation + N = 132.59 − 33.53 =
99.06 m (4 cm). Fixed in a new `flight_scenarios/datum.py` (EGM96 via pyproj), applied at the
data→modeling seam.

*Not* in the harvest, deliberately: the harvest feeds two consumers with opposite requirements
— CZML/Cesium positions are documented as metres above the WGS84 ellipsoid
(`aeroviz-4d/src/types/czml.d.ts`) and are correct as recorded. Converting at the source would
have fixed modeling and broken the viewer by the same ~33 m. The harvest stays a faithful
record of what the sensor said; the datum choice is made on the way in.

The conversion reached THREE separate ingest paths (`load_observed_flights`, `build_scenario`,
`ts_transformer/dataset.py` — the last reads bare waypoints, so it cannot self-protect). It is
keyed on `altitude_source` and therefore idempotent, and unknown/missing sources raise rather
than defaulting. Seam tests pin all three.

PROJ trap worth knowing: without the EGM96 grid and with network off, pyproj silently returns
a "ballpark" no-op vertical transform — a correction that looks applied and does nothing.
`_geoid_transformer()` probes a known undulation and raises instead.

**Bug ② — `runway_thresholds.json` stored pavement ends, not landing thresholds.**
`build_runway_config.py` read `le_latitude_deg`/`le_elevation_ft` and ignored
`le_displaced_threshold_ft` entirely. KSJC 30L/30R are displaced 775 m; on a 3° glidepath that
is a 40.6 m altitude error, and it moved the OPTIMIZER TARGET, not just the gates. Six
thresholds moved (KSJC ×4, KSTL 12R 143 m, KMSY 29 93 m); the other 20 are unchanged. Fixed in
the generator, not the JSON. Schema bumped to `runway-thresholds-v2`; thresholds now carry
`displaced_threshold_m`.

This is why KSJC looked HEALTHIEST before the fix (+9.7 m vs everyone else's −25 m): its two
bugs had opposite signs and nearly cancelled (+40.6 − 32.0 = +8.6 predicted). Chasing the
"anomaly" is what found bug ②.

**Bug ③ — parallel runways captured the same landing twice.** `classify_landing_flights` was
called once per threshold with no cross-threshold arbitration, and `RUNWAY_THRESHOLD_RADIUS_M`
is 1000 m while parallel runways sit 250–400 m apart on an identical heading — so both the
geometry and heading tests accepted either one. Measured: 169 of KSJC 30L's 200 flights were
also in 30R's file; KSJC 12L∩12R 63; KSTL 30L∩30R 32. KRDU/KSMF/KMSY are unaffected (their
parallels exceed the capture radius). It surfaced downstream as an observed lateral error
whose MEDIAN was the parallel separation (KSTL 30L 397 m, KSJC 30R 234 m).

Fixed with a `sibling_thresholds` arbitration restricted to same-direction runways — the
opposite end of the same runway must be excluded, since a full rollout stops on top of it.
The discriminator is the median lateral offset from the extended centreline, NOT distance to
the threshold point: a first attempt using threshold distance failed to separate at all (kept
763 m vs dropped 791 m) because a displaced threshold sits 775 m past where ADS-B coverage
ends, so every track is equidistant from it. On the centreline metric the split is clean
(kept 17.5 m vs dropped 232.8 m at KSJC; 35.9 vs 382.6 at KSTL).

**Effect, end to end** (established-approach threshold crossing, records regenerated through
the real code path, not a reimplementation):

| airport | vertical median before → after | vertical gate before → after |
|---|---|---|
| KRDU | −29.2 → **+4.3 m** | 1 % → **44 %** |
| KMSY | −19.6 → **+5.7 m** | 0 % → **50 %** |
| KSTL | −26.9 → **+4.9 m** | 0 % → **31 %** |
| KSJC | +9.9 → **+0.3 m** | 24 % → **51 %** |
| KSMF | −27.7 → **+2.7 m** | 0 % → **65 %** |

All five now sit at +0.3 to +5.7 m — a small POSITIVE bias, which is operationally right
(crossing at or slightly above TCH is correct; low is dangerous). Lateral was already correct
and is unchanged at 3–10 m median.

**What this makes stale.** Everything derived from observed tracks: `flight_scenarios/outputs`,
all `4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}`, all
`public/data/airports/*/comparison`, and the `ts_*` training data + checkpoints. Bug ③
additionally requires re-harvesting KSJC and KSTL (offline de-duplication is possible but
would cost KSJC 42 % of its flights, leaving 12L at 12 and 30R at 39).

Note for the ts_transformer re-run: the `u` channel shifts uniformly by +33.5 m at KRDU.
Accuracy metrics (ADE/FDE, deviation vs reference) are computed against a reference in the
same frame and should be nearly unchanged, but the GATE verdicts were biased — the recorded
"gate-pass counts 0–4 of 152" was a ±3 m window scored against data offset by 33 m, so that
conclusion needs re-deriving rather than quoting.

Tests: 692 pass (557 modeling+backend, 135 aeroviz-4d/python), both suites exit 0. The
"one known pre-existing failure" in `run_all_tests.sh`'s header
(`test_fixed_time_objective_weights_control_effort_at_one`, numpy scalar conversion) did NOT
reproduce — that note and the matching CLAUDE.md Open Item look stale.

**Post-review hardening (same day).** A recall-mode review of the three fixes surfaced and
closed: ① `resolve_runway_threshold` still returned pavement ends — the `--runway` download
path would have named a threshold up to 775 m from the config's and drifted
`landing_time_utc`/`flight_key` between harvest paths; landing-threshold interpolation is now
single-sourced in `acquisition/runways.py` (`landing_thresholds_from_row`), generator output
byte-identical, plus a loud `ValueError` when a displaced end has no usable length.
② `_wins_against_parallel_runways` crashed on a heading-less threshold
(`math.radians(None)`) and its `inf <= inf` tie silently re-admitted double-assignment when
no sample fell in the centreline window — now: no competitors → win (no offset computed,
also removing a dead full-track scan), unestablished centreline vs a competitor → lose from
both. Also: `_heading_diff` reused instead of an inlined twin; true `statistics.median`.
③ `datum.py`: the ballpark probe was NaN-transparent (`abs(nan−33.53) > 1.0` is False) —
inverted to not-within-tolerance; an operator's explicit `PROJ_NETWORK` is no longer
overridden; `waypoints_to_msl` transforms the altitudes directly (EGM96 N is
height-independent — verified: +1000 m in → exactly +1000 m out), removing the
negate-and-subtract dance. ④ `FlightScenario.source` now records `altitude_source`, so
saved scenario files carry datum provenance (pre-fix HAE-era files lack the key).
⑤ ts dataset conversion moved after the cheap skip checks; test fixtures now import
`METRES_PER_DEG_LAT`/`metres_per_deg_lon`/`FT_M` from geokit instead of retired literals.
⑥ The symmetric OUT seam, closed after review discussion: modeling records (now MSL)
were packed straight into Cesium ellipsoidal `cartographicDegrees`, so after the batch
re-run every opt-/sim-/pred- entity would have drawn ~33.5 m above the white HAE
reference. `build_scenario_comparison_czml._states_to_waypoints` (the single choke point
all record-derived entities share; the reference bypasses it) now converts MSL→HAE via a
new `aeroviz-4d/python/vertical_datum.py` — a mirror of `flight_scenarios/datum.py`
(same KRDU −33.53 pin, ballpark probe, PROJ_NETWORK respect) per the `flight_identity.py`
precedent. Records are assumed MSL rather than tagged: all pre-datum-fix artifacts are
discarded wholesale (user decision), never fed back in.

**New operational scripts (same day).** `run_ts_pipeline.py` — the ts_transformer sibling
of the scenario optimization runner: per airport runs the 2×2 grid (iTransformer/PatchTST ×
window/full) as train → predict(test split) → evaluation report/HTML → comparison-CZML
publish (categories `ts_{itr|ptst}_{mode}`, matching the published naming); dataset build
+ flight_key split happen inside train and travel in the checkpoint. `clean_pipeline_data.py`
— wipes every generated artifact of both chains (scenarios, 4dTrajectory outputs incl.
ts dirs, frontend comparison + observed-layer CZML) with plan-print + confirm/`--yes`;
raw OpenSky downloads and `_`-parked research dirs are kept unless `--include-downloads` /
`--include-parked`; static airport layers and `data/archive` are never touched.

### 2026-07-20 — prediction overlay: anchor-time alignment + the lookback window is drawn

Two defects in how a ts_transformer forecast reached the globe, both invisible in the record
files (which were correct) and both living in `build_scenario_comparison_czml.py`.

**The forecast was drawn a whole lookback early.** A prediction record rebases its own time so
`t = 0` is the ANCHOR — the last observed sample the model was shown, `seq_len - 1` samples
into the approach. The reference is copied out of the airport's `trajectories.czml` and still
starts at `t = 0` = the START of the track. The builder wrote the record's times straight
through as CZML offsets, so the two shared a clock they did not share a zero on. Measured on
KRDU 05L / `AAL542_…`: `pred[0]` is bit-identical to the reference's `t = 118 s` sample, and
was plotted at `t = 0` — **12.0 km** from where the reference was at that instant. Every
prediction-schema entity is now shifted by `source.anchorTimeS`. Optimizer records were never
affected: their `t = 0` already is the scenario start.

**The lookback was never rendered.** `export.py` has always written `observed_states` as the
WHOLE observed track (negative `t` before the anchor) explicitly so a viewer could show the
input the model was conditioned on — and no viewer ever read it. The purple line simply began
in mid-air at the anchor. It now emits a second entity per group, `look-{group}`: the `t ≤ 0`
slice, same hue as the forecast at alpha 85 (`LOOKBACK_COLOR`) vs 225, frontend kind
`lookback` with its own "Predictor input" legend row. The anchor sample belongs to both halves
— it is literally the same state object in the record — so the faded half meets the forecast
exactly, not merely closely (asserted, not eyeballed). `observed_states` moved into
`_PREDICTION_SCHEMA`: a record that cannot be drawn completely now fails loudly.

Supporting cleanups: the lookback retraces samples the reference already covers, so it renders
path-only (a model/point there would draw a second aircraft on top of the reference's for the
whole input window); the entity-id→kind prefix table is now one list shared by `kindOfEntityId`
and `isComparisonEntity`, which had drifted — the picker's list was missing `pred-`, so
prediction tracks silently could not be hovered for their callsign; per-kind alpha
(`COMPARISON_KIND_ALPHA`) drives the legend swatch too, since "Predicted" and "Predictor input"
share a colour and a solid swatch made the two rows identical.

Rebuilt all four KRDU ts categories (`ts_itr_full` / `ts_itr_window` / `ts_ptst_full` /
`ts_ptst_window`, 152 groups each). Verified structurally across all 608 lookback entities:
every one starts at the reference's own `t = 0`, carries exactly `seq_len = 60` samples, ends
where its forecast begins, and never outruns its reference. 138 CZML-package tests, 460
frontend tests, tsc clean, `npm run build` clean. NOT re-checked in-browser.

### 2026-07-20 — B3: transport-consistent velocity channels + physical-velocity fit; third ts training generation

The findings doc's B3 bundle (`docs/findings_and_open_items_2026-07-20.md`), executed. The
one deviation from B3's literal scope was forced by measurement — see the second bullet.

**B3.1 — the position↔velocity inconsistency (A7) closed, at BOTH seams.**

- New numeric single source `geokit.wgs84_curvature_radii(lat_deg) -> (R_M, R_N)` (exact
  WGS84 closed forms). `flyability._transport_rates` now imports it (was an inline copy);
  the casadi geodetic RHS keeps its symbolic twin with a MUST-match mirror comment (a CasADi
  expression cannot call a float function). Pinned in geokit's tests at the equator/pole/45°
  landmarks.
- `ts_transformer/channels.py`: velocity channels are now the exact **chart derivatives** of
  the position channels — the physical velocity mapped through the full-transport Jacobian
  (`ndot = V_north·a/(R_M+h)`, `edot = V_east·a·cos lat₀/((R_N+h)·cos lat)`, `udot`
  unchanged); `states_from_channels` inverts exactly. Renamed `ve/vn/vu → edot/ndot/udot`
  deliberately: the channel tuple is serialised into every checkpoint and `load_checkpoint`
  refuses a mismatch, so every pre-change checkpoint fails loudly instead of silently
  mis-scaling velocities. New tests pin the factor closed form and the integration identity
  (a sequence generated by the geodetic kinematics integrates its velocity channels back
  into its position channels).
- **The fix as literally scoped in B3.1 would have made the measured inconsistency WORSE** —
  found by measuring, not reasoning. On all 995 KRDU arrivals (median whole-track drift of
  ∫v dt against the position channels): original east 3.5 / north 2.7 m/min; channels fixed
  alone east 3.4 / **north 8.6** m/min. Cause: `flight_scenarios._velocity_lsq` fitted
  velocity through the flat chart scales (`a`, `a·cos lat`), so its `V_north` overstated the
  physical value by `a/R_M` (+0.33% at 36°) — the old *channel* code cancelled that bias by
  accident, and A7's "cos ratio + h/R + R_M/R_N" attribution was really describing the FIT.
  Per the fix-upstream convention, `_velocity_lsq` now projects through the true tangent
  scales at the window anchor (`(R_M+h)`, `(R_N+h)·cos lat`), making its output the physical
  ENU velocity every consumer already assumed (the geodetic RHS, flyability's inversion, the
  ts chart). Final measurement, both seams fixed: east **2.4** / north **2.7** / up 0.45
  m/min — unbiased LSQ smoothing, no systematic left.
- Blast radius of the fit change: every fitted `V/psi/gamma` moves ≤ 0.33% / ≤ 0.1°;
  positions are untouched, so all position-based metrics and gates are unaffected. The
  2026-07-20 optimizer batch artifacts predate it (same situation as the geokit constant
  alignment below: a re-run would move initial-state V by ~0.2 m/s, far below data noise).
  Fixture constants in `test_start_state.py` / `test_scenario_optimization.py` were rebuilt
  on the tangent scales.

**B3.2 — `dt = 1 s` considered and declined** (README "Sizing"): same coverage needs
L=120/H=600 (~2× training cost) for almost no information — the source reports at ≤ 1 Hz
with ragged gaps and the velocity channels come from a 15 s window fit. `--dt` stays a knob.

**B3.3 — the lead-time table is restored on the raw-tensor accounting** (A8). The README
now carries BOTH accountings with their n: record accounting (one forecast per flight,
threshold-truncated — n falls with lead; 600 s survives for ≤ 1 flight) and raw-tensor
accounting from `history.json` `metrics.test.by_horizon` (every test window, sees past
truncation — 600 s restored with n = 893). On it, **iTransformer leads PatchTST at 600 s in
all three training generations** (6135/7142, 5407/6962, 5438/7384 m) at margins of
1.16–1.36× — direction consistent, each margin under the provisional band.

**Retrain (third generation).** Same recipe (ep=120, patience=15, lr=5e-4, seed=1337) and
the same `flight_key` split (702/141/152); all four cells trained, predicted (test split),
evaluated, flyability-reported. Pre-B3 artifacts moved to
`4dTrajectory/outputs/KRDU/_pre_b3_transport/`. Headline (152 test flights): iTransformer
full lateral 868/2750 m mean/p95, chained 1302/6576; PatchTST full 2016/5598, chained
3433/8993 — **full beats chained on threshold lateral in every generation** (1.5–1.7× mean).
Gate passes are 0/152 in all four cells this generation (pre-B3: 4, 1, 0, 0) — across
generations the count ranges 0–2.6% and the README now states only the stable conclusion
(a forecast is not a certifiable approach; a borderline pass is jitter). Flyability floor
(observed tracks) stays 63.2%; iTransformer window is again the only cell above it (73.7%).
The instance-norm ablation (`_ablation_norm/`) was NOT re-run — measured pre-B3, margins
1.2–2.7× vs a ≤ 0.3% channel rescale, structural argument unchanged; dated note added.

Suites: run_all_tests.sh 520 + 135 passed (the one known pre-existing collocation failure),
geokit 29. The ts suite is 56 tests (2 new transport pins).

### 2026-07-20 — full batch re-run (15/15 fresh), geokit per-degree constant aligned to the optimizer

**Batch.** The former combined runner with `--jobs 6`, 5 airports × 3 categories, 10,449 solves,
≈4 h 39 m wall clock (one harness-side background-task reap mid-run; resumed detached with
`setsid nohup`, `overall_fail=0`). All artifacts now post-date every 2026-07 fix (arrival
truncation, altitude floor/rollout guard, HS flip, identity unification) — the standing
"all batches are STALE" open item is closed. ts predictions were regenerated the same day,
**test split only** (152 flights per the reproducible flight_key split), for all four
checkpoints.

Identity contract verified on the fresh artifacts, all green: record stems are full
flight_keys; summary rows all carry `landing_time_utc`; **reference hit rate 100% in all 19
comparison categories** (15 optimizer + KRDU's 4 ts_pred — the pre-refactor
duplicate-callsign dropout was 22%); zero duplicate entity ids in sampled CZMLs; every
airport's categories.json intact (KRDU keeps 7 categories).

Headline solve/gate rates (success among solved): runway_cons is the cleanest everywhere
(93–99%), asdb the hardest (76–89%). Finding worth keeping: **KRDU RW32 is systematically
hard and NOT the old truncation artifact** — runway_cons RW32 79 offTarget + 59 failed of
198 (all other runways ≤9), and asdb RW32 fails 197/198 (IPOPT infeasible). Likely
procedure-specific (RNP-AR H05LZ; per-leg RNP still not extracted). KSTL runway_cons has a
milder cluster (12R/30R/30L/24; repeated single-IAF `PAULY` infeasibility).

**geokit alignment.** `METRES_PER_DEG_LAT` was the hand-rounded `111_320.0`; the
optimizer's NE frame (`approach_constraints.frame` and the NLP's metric-position
normalization) derives `WGS84_A·DEG2RAD = 111319.4908…` — a 4.6 ppm seam (~0.11 m at the
25 km ring) between the two frame families. Now defined as `WGS84_A * (π/180)` in
`geokit.constants` — bit-identical to the optimizer's product (IEEE commutativity),
`metres_per_deg_lon` stays pure `·cos(lat)`. Frontend `geoConstants.json` regenerated.
Applied AFTER the batch finished so all 15 cells share one constant; ts checkpoints are
unaffected in practice (inputs move ≤0.11 m at the ring edge vs km-scale model error — no
retrain). Also corrected the `channels.py` projection docstring: the flat chart deviates
from a true tangent-plane ENU by up to ~40 m at the ring edge (`e·n·tanφ/R` cross term) and
`u = Δalt` ignores the ~49 m curvature drop by design — the old "well under a metre" claim
was true only of the quantities the metrics actually measure (same-chart comparisons;
~0.2% local scale distortion at the ring edge, → 0 at the threshold).

### 2026-07-20 — flight identity unified end-to-end: entity ids = flight_key, positional `_N` re-uniquing deleted

The last identity holdouts (CZML entity ids, the comparison reference lookup, the FlightTable
optimizer join) still ran on bare callsigns + positional `_2/_3` suffixes. Diagnosis on the real
KRDU harvest:

- **Per-runway landings/arrivals files held massive duplicate ids** (128 duplicates across five
  runways; `N993FG` ×10 on RW32). Root cause: `collect_landings` harvests in CHUNKS and
  `classify_landing_flights` restarted its `_unique_id` numbering per chunk, while the
  cross-chunk merge de-duplicated by `(icao24, landing_time_utc)` without re-uniquing ids. The
  per-runway CZMLs inherited them — Cesium merges same-id packets, so two namesake flights
  rendered as ONE garbled entity (both tracks' samples interleaved from t=0), and
  `flightSummaries`/React row keys collided.
- **Cross-view id aliasing**: `merge_landing_flights` re-uniqued ids positionally for the
  combined file, so the same string (`SWA1692_2`) named DIFFERENT physical flights in the
  per-runway vs combined views, and the same flight got different `flight_key` stems from the
  two record writers (optimizer eats combined, ts eats per-runway) — breaking identity.py's
  "same stem" promise and making the comparison builder's callsign reference-lookup resolve to
  whichever namesake came first (the "wrong white line" open item).
- **FlightTable's optimizer join was callsign-keyed** (`byFlightId` from `group.flightId`), so
  namesakes swapped each other's V/mass/failed/offTarget facts.

Fix — one identity everywhere, display strictly separated:

- `generate_czml.build_czml`: entity id = `flight_key(flight)`, `name` = callsign; RAISES on a
  duplicate identity (silent Cesium merge → loud input error). Both `trajectories.czml`
  producers go through it.
- `_unique_id` deleted from the landing path (`classify_landing_flights`,
  `merge_landing_flights`); kept ONLY in `trajectories_to_czml_input` (the plain download path
  has no runway/landing time — the suffixed id is its one discriminator).
- `aeroviz-4d/python/flight_identity.py`: deliberate MIRROR of
  `flight_scenarios.identity.flight_key` (frontend tooling can't import the modeling tree);
  both copies pinned to `EJA969_05R_ad7f04_20260618T213736Z` in their own suites.
- Comparison builder: reference lookup by `group` (= flight_key = the new entity id) in batch
  mode and by `flight_key(source)` in single mode; `scenario_initial_map` keyed by flight_key
  (was `(id, runway)` — namesakes shared one V/mass); `_group_key`'s file-less fallback
  reconstructs the identity from the row (rows now carry `landing_time_utc` via `summary_row`
  — it IS part of the identity and was the one missing field).
- Frontend: `useFlightOptimizerData` keys by `group.group` (renamed `byFlightKey`);
  FlightTable/approach view display the callsign (`name`) while ids stay the
  selection/join/cache identity; `ObservedFlightSummary` gained `callsign`.

Verified: 57 aeroviz-4d python tests, 59 trajectory_data_process, 84 optimization+ts, 453
vitest, tsc + vite build — all green. New pins: namesake entity ids distinct + duplicate
identity raises (generate_czml); each comparison group copies ITS OWN namesake's reference
track; FlightTable keeps namesake optimizer facts apart. Artifacts regenerated after this
change (arrivals/CZML rebuild + full batch re-run); ts checkpoints are UNAFFECTED (per-runway
arrivals `id` fields — and therefore split keys — are byte-identical; only the combined view
changed).

### 2026-07-19 — flyability check, instance-norm re-ablation on real data, predictions in the frontend

Three things asked for together: the post-hoc flyability check (route 1 of the README's four),
re-ablating `--instance-norm` on real data, and getting prediction results to render in the
frontend alongside the optimizer's.

**Post-hoc flyability (`4dTrajectory/ts_transformer/flyability.py`, 16 tests).** The
load-factor point-mass model inverts in closed form — `A = psi_dot V cos(gamma)/g = n sin(mu)`,
`B = gamma_dot V/g + cos(gamma) = n cos(mu)`, so `n = hypot(A,B)`, `mu = atan2(A,B)`; `n` fixes
`Cl`, `Cl` fixes drag, and `T = m(V_dot + g sin(gamma)) + D` closes it. One pass, no solver, no
casadi, so it lives in the torch env. Earth-frame transport terms are subtracted first.
`predict` writes `flyability_report.json` and prints a summary line.

- **The calibration matters more than the check.** First run against REAL flown tracks scored
  **0/149 fully flyable** — those are trajectories real aircraft flew, so the check was wrong.
  Cause: `thrust_negative`. Median required thrust on a real arrival is **0.43 kN** (idle), and
  a negative requirement means the aircraft needed more drag than a clean airframe has —
  speedbrake, flaps, gear. Every approach does this; one clean-configuration polar cannot
  represent it. Reclassified as SOFT (reported, not counted unflyable), and the report leads
  with the delta against the observed tracks measured by identical code, because both sides
  carry the same polar bias. **The observed baseline is the floor, not 100%.**
- `Cl_max` from `aero_params_for_aircraft` (2.7 for an A320), NOT `LoadFactorSimulator`'s
  hardcoded 1.5 — they disagree by 80% and `aero_params.py` is the documented source of truth.

**Instance-norm re-ablation on real KRDU data — the synthetic OFF default holds.** All 8 cells
(2 models × 2 horizon modes × on/off), same hyperparameters, each graded on its own
checkpoint's test split; artifacts in `4dTrajectory/outputs/KRDU/_ablation_norm/`, roll-up in
`ablation_results.json`.

- **OFF wins 19 of 20 accuracy comparisons, one tie**: every cell on val loss, FDE, ADE p95 and
  lateral p95; 3 of 4 on mean ADE with PatchTST window a dead heat (2580 vs 2571 m — the cell
  where OFF wins the other four metrics). A sweep that lopsided is not run-to-run variance,
  which was the open question: individual gaps here are much smaller than on synthetic data
  (1.2–2.7× vs 2.4–6.5×), and an earlier partial pass on ONE metric had shown an apparent
  reversal in that same PatchTST window cell. It did not survive consistent scoring.
- **The signature is lateral p95**: all four instance-norm-ON cells land at 14.28–14.50 km —
  near-constant across both architectures and both horizon modes. That is a model that cannot
  place the endpoint at all; strip the absolute level and the prediction ends at a distance set
  by the frame, not by the flight. OFF spans 2.6–8.5 km, i.e. it varies with the flight.
- **Flyability moves the OPPOSITE way** (ON better in 3 of 4 cells; PatchTST window 89.3% vs
  29.6%, i.e. the configuration 2.2× worse at the threshold looks three times more flyable).
  Instance norm is not flying better, it is predicting blander paths — easy to fly, far from
  the truth. A straight line is perfectly flyable and completely wrong. Recorded in the README
  because it is the standing argument for never reading flyability alone. (iTransformer full is
  the exception at 46.1% off vs 34.9% on.)

**Headline KRDU runs retrained, and two published conclusions did not survive it.** The 4
published checkpoints carried a split that `hash(flight_key)` reproduces for only 552/995
flights — they predate the per-flight-hash split fix, exactly the "retrain before comparing"
note in the entry below. The partition itself is clean (train/val/test verified disjoint), so
the published numbers were not wrong, just on a split no current run can reproduce and not
comparable with the ablation. Retrained on the current keying (702/141/152):

- **"The compounding cost lands in the tail, not the mean" — withdrawn.** It rested on lateral
  mean 1.5× vs p95 2.2×; on the new split the two ratios are equal to within noise
  (iTransformer 1.55× vs 1.61×). A tail effect survives in FDE (PatchTST p95 1.78× against a
  1.36× mean) but that is a different claim than the one written.
- **"Zero gate passes in all four runs" — withdrawn.** iTransformer now passes 3/152 (full)
  and 1/152 (chained). The substance holds (2% is not a usable predictor) but "zero, always"
  was a property of that split.
- **Survived:** one-pass full beats chained window on whole-approach lateral error for both
  models on both splits (1.5–1.6× mean, 1.5–2.1× p95), and the short-lead/long-lead crossover
  between PatchTST and iTransformer (now cleanest within the full-mode pair: 184 vs 571 m at
  10 s, reversing by 300 s).

Standing lesson recorded in the README and CLAUDE.md: **treat any single-split margin under
~1.5× as provisional** — that is the size of effect a split change moved here.

**Flyability shipped with a wrong assumption, caught by its own guard.** The check graded a
whole batch against ONE envelope, documented as safe because "every harvested arrival is type
UNK and resolves to the single `--aircraft-type` fallback". A boundary assertion added to state
that assumption fired on the first real batch: `_resolve_aircraft` falls through to an
**`icao24` → OpenAP lookup** that recovers the real airframe — 20 distinct types across 400 KRDU
arrivals, A320 only 224 of them. So ~44% of every batch was being judged by an A320's `Cl_max`
and max thrust, invisibly, because the report never named the airframe it used.
`report_for_records` now takes one `Aircraft` per flight, builds one envelope per distinct type,
and the roll-up carries `fleet` + `envelopes`. All flyability numbers moved: the observed-track
baseline went 58.4% → **63.2%**. The CLAUDE.md/README claims about UNK were corrected.

**`summary.json` now carries an `accuracy` block** (+ per-row `ade_m`/`fde_m`/`overlap_steps`).
A batch's error against the observed tracks is its headline result and it existed only as a
printed mean — comparing eight ablation cells meant scraping stdout. Mean AND p95, because
chained window-mode error compounds into the tail. `overlap` is a required argument to
`write_batch`, not optional: an optional metric is one that silently goes missing.

**Predictions render in the frontend.** `build_scenario_comparison_czml.py` detects the record
schema (`optimizer_states`/`simulator_states` vs `predicted_states`) and emits `pred-` entities
for a prediction batch; the entity-id prefix is already what the frontend keys `kind` off, so
`predicted` gets its own purple colour and legend checkbox. Two follow-on fixes:

- **Off-target recolouring restricted to the optimizer schema.** A forecast essentially always
  misses the 106.75 m gate, so it marked 27/27 groups off-target and repainted every prediction
  yellow — the kind colour was never once visible and a marker that fires on everything carries
  no information. `properties.status` stays accurate; deviation is reported by the evaluation
  report and comparison index.
- **The frontend repaint skip was keyed on the wrong thing.** It skipped legend repaint whenever
  `status === "offTarget"`, but what it exists to preserve is a baked VERDICT colour (the
  yellow). Predictions never get that bake yet are always `offTarget`, so they rendered from the
  CZML — matching their legend swatch only because `PREDICTION_COLOR` and the TS legend entry
  happen to hold the same RGB. Now keyed on whether a verdict colour was actually baked.

**Verified in-browser** (Linux Chrome, KRDU, all 4 categories): purple prediction + white
reference paths render together, the `Predicted` legend checkbox removes only the purple, and
the Optimization panel's metrics follow the selected category (752 m / 2.0% iTransformer full
vs 3184 m / 0.0% PatchTST window, matching the evaluation reports — and PatchTST window is
visibly the more scattered fan, an independent read of the same gap). Also backed by tsc clean,
451 frontend tests, `npm run build`, 54 ts_transformer + 25 CZML-builder tests, and a structural
check of the published artifacts against every contract point the frontend reads. Gotcha found
while verifying: comparison entities are time-windowed, so at a clock time outside a group's
availability the scene is legitimately empty — pause inside a window before concluding the
overlay is broken.

### 2026-07-19 — code review fixes: ts_transformer contracts, env resolution, identity single-sourcing

Applied the 15 findings of a full-diff review (ts_transformer + the uncommitted env-script changes):

- **train.py**: a non-finite train/val loss now RAISES ("training diverged…") instead of sailing through the best-val bookkeeping — previously a run that went NaN before any finite improvement wrote the last (NaN) weights as a "successful" checkpoint and emitted literal `NaN`/`Infinity` into history.json. `load_checkpoint` now uses `weights_only=True` (payload is tensors + primitives) and refuses a checkpoint whose serialized channel order differs from `channels.CHANNELS` (a same-length reorder loaded cleanly and silently mis-mapped every channel). Flights too short to yield one training window (window mode: `seq_len + pred_len`; build only requires `seq_len + 1`) are excluded — counted, not silent — before the split, so they no longer occupy split slots and produce the misleading "empty window set" abort.
- **Aircraft type is part of the checkpoint**: `TSConfig.aircraft_type` (default A320, moved to config.py) is serialised with everything else; predict defaults to the train-time value instead of its own CLI default, and warns when explicitly overridden — the type sets target Vref/TCH, i.e. the ENU frame and the gate target the normalizer stats were fit under.
- **predict --device**: the compute device is a runtime property, no longer read from the checkpoint (an explicit-`cuda` checkpoint was unusable on a CPU-only machine with no override).
- **Horizon cap is stated, never silent**: window-mode chaining still stops at 300 steps (600 s), but a forecast that ends short of the threshold is flagged `horizonCapped` (record source), `horizon_capped` (summary row), and predict prints a per-batch WARNING — previously those ~2% of flights were graded as huge gate failures indistinguishable from model error.
- **Split stability**: train/val/test assignment is now a per-flight sha256 of `(seed, flight_id)` instead of a positional permutation — adding/removing one flight to the harvest reshuffled the entire split (old test flights silently entered training on a retrain). Fractions are now approximate; empty train/val raises with the real cause (tiny datasets previously rounded val/test to zero and died later blaming window sizes). NOTE: this reshuffles the split once relative to the existing KRDU checkpoints — retrain before comparing new runs against them.
- **synthetic.py**: icao24 derived via `zlib.crc32(runway)` instead of the process-salted builtin `hash()` — identically-seeded synthetic data now has identical flight identities across interpreters (previously `predict --split test` on regenerated data intersected on zero flights).
- **Identity + filename single-sourcing**: `flight_key` moved to `flight_scenarios/identity.py`; `scenario_optimization._scenario_filename` and ts dataset/export both import it (the two copies had already drifted: `flight{i}` vs `scenario{i}` fallback — unified on `flight{i}`). Record-filename suffixes + `REFERENCES_DIR` hoisted into `optimization/evaluation_export.py`, imported by both writers; the "MUST match" mirror comments are gone.
- **dataset.py**: falling through to `*_landings.json` (a download-only dir with no `*_arrivals.json`) now prints an explicit UNTRUNCATED-tracks warning — a different task/duration distribution than the windows were sized for.
- **Env resolution deduplicated + made content-aware**: new `scripts/activate_aeroviz_env.sh`, sourced by `run_all_tests.sh` (warn-and-continue) and `start_aeroviz_fullstack.sh` (abort). Candidates are probed with `import casadi` instead of trusted by name (on this machine `aviation` exists but is another project's env — both scripts previously accepted it when active and fell back to it when `aeroviz` was missing or `AEROVIZ_CONDA_ENV` was mistyped; the launcher then crash-looped the backend on `import casadi`). An explicit `AEROVIZ_CONDA_ENV` is now the ONLY candidate. The launcher ACTIVATES the env rather than direct-exec'ing `envs/<env>/bin/python`, so activate.d hooks (the libstdc++ `LD_LIBRARY_PATH` fix) apply to the backend subtree; missing conda now reports "conda is not on PATH" instead of probing `/envs/...`.
- **Docs**: CLAUDE.md's Key Defaults ts bullet updated to the real 60/300 window sizes and the measured-distribution rationale (it still carried the superseded 30/150 + "~3.5–5 min arrival" story that config.py refutes); requirements.txt pin rationales corrected (the torch comment claimed `weights_only=True` while the code passed `False` — now the code matches the claim; the numpy `copy=None` rationale matched nothing).

Follow-up pass over the review's below-the-cap cleanup findings (same day):

- **forecast.py**: the HORIZON_FULL branch of `forecast_approach` was a copy of `recursive_forecast`'s single-pass body that also skipped its anchor validation — both modes now run through `recursive_forecast` (full mode's `pred_len` covers `max_steps`, so it is one pass; an explicit `anchor < seq_len - 1` now gets the clear ValueError instead of a cryptic vendor shape error). `_forward` forces `model.eval()` (dropout noise from a train-mode model would compound through the chain). `default_anchor` dropped its never-read `series` parameter.
- **models.py**: PatchTST now receives `act=config.activation` (the vendored Model takes activation as a bare kwarg, so `TSConfig.activation` previously applied to iTransformer only while the checkpoint recorded it for both).
- **metrics.py**: `displacement_metrics` folded into `trajectory_metrics`; `error_components` computed once per call (was twice, with per-step displacement recomputed twice more) via a `displacement_grid` key; `_positions` uses `channels.POSITION_IDX` instead of re-deriving it. `_spread` stays vectorised — delegating to the stdlib-only `evaluation/stats.signed_spread` would sort millions of boxed floats in `evaluate_split` — but a new seam test pins the two equal on the same input, replacing the mirror comment with a checked property.
- **export.py**: `PredictionRecord.source`/`final_time_s` became read-through properties (three stored copies of `final_time_s` per record could disagree after a future edit; `evaluation` rejects `final_time_s != states[-1].t`); the whole observed track is converted to states ONCE and the anchor sample + reference span are slices of it (the per-sample GeodeticState loop previously ran twice over the tail); summary rows go through the new `evaluation_export.summary_row` (shared with `scenario_optimization._summary_record`, which now wraps it); a redundant `min()` removed.
- **train.py**: `_predict_split` stores decoded windows as float32 (float64 doubled the live peak — hundreds of MB at full-mode scale on the 16 GB swap-bound machine — for precision metre-scale metrics cannot use); the duplicated val `evaluate_split` call in both ternary arms collapsed.
- **__main__.py predict**: the split filter now runs on the RAW flight dicts (keyed identically to `build_series` via `flight_key`) BEFORE the expensive series build — a default `--split test` predict previously built and discarded ~85% of the work. Verified live: "built 1/1 series" for a 14-flight file.
- **dataset.py**: the track-span check moved BEFORE `build_scenario` (too-short flights no longer pay aircraft resolution + per-sample least-squares fits before being skipped; the raw-waypoint span is the identical value); dead code removed (`Normalizer.decode_torch`, `FlightSeries.duration_s`/`.source`).
- **channels.py**: dead `VELOCITY_IDX` + `Frame.to_dict`/`from_dict` removed; new `Frame.latlon_from_en` single-sources the inverse projection, used by `states_from_channels` AND `synthetic.py`'s waypoint generator (which previously hand-rolled a fifth copy of the frame formula; it now builds a `channels.Frame` and dropped its direct geokit constant imports).
- **package.json**: `npm run backend` resolves its interpreter through `scripts/activate_aeroviz_env.sh` instead of bare `python` (it was the third, uncoordinated env-selection seam).
- Deliberately NOT applied: batching the predict loop (would complicate the single-flight inference seam for a non-bottleneck), early-stopping the window-mode chain at the threshold (changes the measured object — `truncate_at_threshold` takes a global argmin), the vendored PatchTST `pv()` NameError on non-default positional encodings (`layers.py` is documented "copied whole, unmodified"; fixing it breaks the byte-identical vendoring contract — landmine noted here instead), and a src-layout re-packaging of ts_transformer's flat modules (the sys.path-shadowing hazard is real but latent; disproportionate to fix now).

### 2026-07-19 — ts_transformer: first REAL-data run (KRDU, 995 arrivals) — both models × both horizon modes

Harvested ADS-B landed (3747 arrivals, 5 airports, 815 MB). Full matrix trained and graded on KRDU: 995 arrivals over 6 runways, split by flight 697/149/149, 96k training windows, 120-epoch cap / patience 15 / `lr=5e-4`, RTX 4060. All four prediction batches graded by `python -m evaluation`. Artifacts under `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` and `ts_pred_{model}_{mode}/`.

**Displacement error at matched lead times** (the only cross-mode-comparable axis — headline ADE/FDE average over different horizon-length distributions and must not be compared):

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | 259 | 390 | 687 | — | — | — |
| iTransformer | full | 587 | 611 | 840 | 1502 | 3227 | 6135 |
| PatchTST | window | 158 | 409 | 995 | — | — | — |
| PatchTST | full | 253 | 399 | 830 | 1914 | 3983 | 7142 |

**Whole-approach prediction graded at the threshold** (149 test flights, directly comparable):

| model | mode | lateral mean | lateral p95 | path deviation |
|---|---|---:|---:|---:|
| iTransformer | full | **1070 m** | **3136 m** | **1895 m** |
| iTransformer | window (chained ×10) | 1594 m | 6898 m | 1918 m |
| PatchTST | full | 1804 m | 4666 m | 2488 m |
| PatchTST | window (chained ×10) | 2815 m | 7354 m | 3152 m |

Findings: (1) **the cost of chaining lands in the tail, not the mean** — chained-window vs one-pass-full is 1.5× on lateral mean but 2.2× on p95, because once one chained pass goes wrong the next nine extrapolate from a wrong history; training directly for the long horizon beats chaining a short one. (2) **iTransformer beats PatchTST at long lead for a structural reason** — PatchTST is channel-independent (`TSTiEncoder`), so it cannot represent the east/north coupling of a turning aircraft, while iTransformer attends *across* variates; note the reversal at 10 s lead (PatchTST 158 vs 259 m), where the aircraft is nearly straight and independence costs nothing. (3) **0/149 gate passes in all four runs is the honest result** — 106.75 m is FAA containment for a planned/flown approach, not a forecast-accuracy target; the number measures the gap between a statistical prediction and a certifiable trajectory. (4) Real is much harder than synthetic (423 vs 286 m ADE): synthetic approaches are straight-in, real ones are vectored, and *when* the turn onto final happens is a controller's decision a single-aircraft model structurally cannot see.

**Three real bugs the first real-data run exposed**, none of which a loss curve would have shown:

- **Silent training-data contamination.** A harvest directory holds five overlapping views of the same flights — `*_arrivals.json` (truncated, the training input), `*_landings.json` (SAME flights untruncated), `*_combined_czml_input.json` (all runways merged), plus `*_heading_rejected.json` / `*_local_rejected.json` (tracks the harvester explicitly THREW OUT). `glob("*.json")` loaded every flight three times over plus the known-bad ones. `dataset.select_flight_files` now takes the first matching pattern only, never mixes, always excludes `*_rejected*`, and prints what it skipped.
- **Aircraft type is `"UNK"` for all 3747 flights** and `flight_scenarios._resolve_aircraft` raises rather than guessing — the batch died on flight #1. Added `--aircraft-type` (default `A320`, printed every run). Not cosmetic: it sets the target state's Vref and threshold-crossing height, i.e. what the gates measure the final state against.
- **Sizing was wrong twice** (see the environment entry below for the corrected defaults).

### 2026-07-19 — Environment consolidated into `aeroviz`; the `aviation` name collision documented

`aeroviz` (py3.12) is now the single thesis environment — `torch` installed alongside the existing acquisition (`traffic`, `pyopensky`), CIFP (`cifparse`, `arinc424`), `casadi`/IPOPT, `openap` and conda-forge geospatial stacks. `run_all_tests.sh` needs no change: its `4dTrajectory` entry now covers the ts_transformer suite too.

**The name collision.** CLAUDE.md said "Python env: conda `aviation`", carried over from another machine — `run_all_tests.sh`'s own comment records "Env names differ per machine (`aeroviz` here, `aviation` elsewhere)". On THIS machine `aviation` is the editable env of `/home/supercomputing/studys/AivationTransformer`, an unrelated project (pure-pip, py3.11, 254 packages, 8.6 GB). Acting on the stale line nearly deleted the real thesis env; the ts_transformer package was also initially targeted at `aviation` for the same reason. Both CLAUDE.md lines are corrected and the collision is now documented under Environment.

**Consolidating the other way is BLOCKED — measured, not assumed.** `cifparse` >= 2.0.4 (aeroviz has 2.0.9) uses PEP 701 f-strings (nested same-type quotes), which is Python 3.12+ syntax. Every version from 2.0.4 up fails `compileall` on py3.11; only 2.0.0 and earlier import there — a 9-patch regression in the ARINC 424 parser that feeds `approach_constraints`. Upstream's PyPI metadata claims `requires_python >=3.10` and is simply wrong. So a py3.11 env cannot host the thesis at current package versions. `casadi`/`openap`/`arinc424` install fine on 3.11; `cifparse` alone is the blocker.

**One real interaction between torch and the existing stack, found and fixed.** `import torch` then `import traffic` raised `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`; the reverse order worked. Cause: pip's manylinux torch wheel has no RPATH into the env, so it resolves `libstdc++.so.6` from the system (CXXABI ≤ 1.3.13), and once that SONAME is loaded conda-forge matplotlib's `_c_internal_utils.so` — which needs 1.3.15 — is answered by the already-loaded old one. `run_all_tests.sh` runs every suite in ONE pytest process with `4dTrajectory` (torch) listed before `trajectory_data_process` (traffic), i.e. precisely the failing order. Fixed with `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh` prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (plus a matching `deactivate.d`). Note it only takes effect under `conda activate` — calling `envs/aeroviz/bin/python` directly still reproduces the failure. Full suite after the fix: **495 passed, 1 failed** (the pre-existing `test_fixed_time_objective_weights_control_effort_at_one` numpy scalar-conversion `TypeError`); the count rose from 464 because the 32 ts_transformer tests now run in the same invocation.

Anything installed into `aviation` during the attempt (casadi, openap, arinc424, the broken cifparse, plus a tqdm/wcwidth bump and an editable geokit) was rolled back; that env is verified back at torch 2.9.1+cu128 / tqdm 4.67.1 / wcwidth 0.3.0 with no thesis packages. Env spec backups for `aeroviz` live in `.env-backup/` (pip freeze, conda explicit, environment.yml).

**ts_transformer defaults re-sized against the real data** (see the entry below for the original synthetic sizing). The 3747 harvested arrivals have durations p5 235 s / p50 328 s / p90 607 s / p99 920 s — much longer than the "25 km at 120 m/s ⇒ 3.5–5 min" straight-line estimate, because real arrivals are vectored (downwind legs, base turns, holds), so the flown path far exceeds the straight-line distance to the entry ring. `seq_len` 30 → **60** (120 s), `DEFAULT_PRED_LEN_FULL` 150 → **300** (600 s): full mode now covers the complete remaining approach for **97.8%** of flights, where 150 covered **57.6%**.

### 2026-07-19 — `4dTrajectory/ts_transformer`: iTransformer + PatchTST integrated as a learned-prediction sibling to the optimizer

New package answering *what trajectory WILL this aircraft fly* (learned, no dynamics model) alongside `optimization`'s *what SHOULD it fly*. Both emit the same evaluation records, so `python -m evaluation --input <dir>` grades either against the identical regulatory gates.

**Vendoring.** `vendor/itransformer/` (MIT, thuml @ `c2426e6`) and `vendor/patchtst/` (Apache-2.0, yuqinie98 @ `204c21e`) copied byte-identical with only import paths rewritten, each with its `LICENSE` + a `PROVENANCE.md` recording what was copied/dropped and why. Vendored rather than installed because neither upstream is a packaged library and both resolve internal imports through a top-level `layers/` — installed side by side they collide (both ship a different `layers/Embed.py` and `layers/SelfAttention_Family.py`). Dropped from iTransformer: the Reformer/Flowformer/Flashformer/Informer attention variants, and with them the `reformer_pytorch` + `einops` deps. One shared `TSConfig` drives both (upstream's own argparse-namespace contract); `models.py` adapts iTransformer's 4-arg call and PatchTST's 1-arg call to one `model(x)`.

**Env.** Installed editable `geokit` into conda `aviation` (torch 2.9.1+cu128); the package is casadi-free by design, so it lives in the torch env while the optimizer stays in `aeroviz`. `requirements.txt` written. Verified both models forward CPU+CUDA on the RTX 4060.

**Three findings that changed the design, each measured not assumed:**

1. **Sizing.** An arrival truncated at the 25 km ring is only **~3.5–5 min** (~110–150 samples at 2 s), not the 8–12 min first assumed. The initial defaults (L=60 @ 4 s = 4 min lookback) exceeded a whole approach and skipped **5 of every 6** flights as "shorter than one window". Now `dt=2 s`, `L=30`, `H=30` (window) / `150` (full) — all 120 synthetic flights build, ~65 anchors each.
2. **Instance normalisation must be OFF.** iTransformer's `use_norm` / PatchTST's `revin` are ON upstream and strip each window's absolute level as nuisance; in a threshold-anchored ENU frame absolute position *is* the signal. Off wins in **all four** model×mode cells by 2.4–6.5× on ADE (iTransformer window 771→286 m, full 1972→303 m; PatchTST window 910→672 m, full 2268→701 m) — and on converges *sooner* (49–56 vs 74–90 epochs), i.e. to a worse optimum, not undertrained. Defaults flipped; `--instance-norm` re-enables for re-ablation on real data.
3. **Two alignment bugs caught by running the real thing** (both would have produced plausible-looking wrong numbers): (a) `FlightSeries.flight_id` was the callsign, which repeats daily and across runway files — `predict --split test` returned 48 flights for an 18-flight split, and the train/val/test split leaked. Identity is now `dataset.flight_key` = `id_runway_icao24_landingTime`, the same function that produces the record filename. (b) The reference record covered the whole track while the prediction covered anchor→threshold; `evaluation.reference` resamples both at 101 fractions of *their own* arc length, so it reported 4349 m of "path deviation" that was pure span mismatch — **833 m** once span-matched. Records now anchor `t=0` at the anchor sample with `initial_state` the observed state there, and `states[0] == initial_state` as in an optimizer record.

**Contracts.** Channels `(e, n, u, ve, vn, vu)` in a threshold-anchored ENU frame; `psi`/`gamma` never regressed directly (±π wrap averages 179° and −179° to 0°, pointing the aircraft backwards exactly at the turn onto final) but derived as `atan2(vn, ve)`, which *is* the math-ENU convention, so the compass/ENU substitution has no place left to happen. `m` carried, never predicted (unobservable from ADS-B). Records are reference-shaped (`controls == []`) and built via `optimization/evaluation_export.py` — importable here because it is casadi-free — rather than hand-rolling a second copy of the record JSON. `final_time_s` always from `states[-1]["t"]`, never `pred_len × dt` (threshold truncation makes them differ; `evaluation.records` rejects >1e-6).

**Status.** 30 tests pass; verified end-to-end through the real CLI (train on GPU → predict → `python -m evaluation`). **Never trained on real data** — none existed in the tree (`trajectory_data_process/outputs/` absent, no `credentials.json`), so `synthetic.py` generates straight-in approaches for smoke-testing and fixtures. Every number above is synthetic and is plumbing evidence, not a result. Pre-existing unrelated failure noted: `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` (numpy scalar-conversion `TypeError`).

### 2026-07-07 — "Profile" → "Approach view" rename (systematic)

The 2D approach page/toggle is now **Approach view** throughout (UI text, identifiers, CSS, files; `git mv` preserved history): `useApproachView.ts`, `ApproachViewPanel.tsx`, `ApproachViewToggle.tsx`, `approachViewAnalysis.ts`, `approachViewSources.ts`; identifiers like `isApproachViewOpen`/`setApproachViewOpen`, `approachViewMode`, `ApproachViewTrack/Sample/Input`, `buildApproachViewTracks`, `planApproachViewSources`; CSS `.approach-view-*`; toggle labels "View"/"Hide view". Deliberately NOT renamed — the geometric "profile" (altitude cross-section) keeps the term: `runwayProfileGeometry`/`RunwayProfilePoint`, `procedureProfileProjection`, `procedureVerticalProfileOverlay`, `ProfilePlot`/`ProfileOverlay`, the "Vertical profile" side-view mode label. Not re-checked in-browser.

### 2026-07-07 — Approach view: whole-track plotting, review fixes, perf cache (three passes, same day)

- **Whole track**: `sampleEntityTrack` walks outward in both directions from the current time (`TRACK_SAMPLE_STEP_SECONDS = 5`, `MAX_TRACK_SAMPLES_PER_DIRECTION = 600` backstop) returning the whole time-ordered track; all points kept tagged by containment tier; an aircraft is plotted iff some sample reaches PRIMARY (`trackEngagesProcedure` — full approach for flights that fly the procedure, unrelated traffic excluded). `splitTrackByContainment` splits ordered samples into contiguous runs sharing boundary points (gapless lines); PRIMARY/SECONDARY drawn solid (primary brighter), OUTSIDE dashed `6 5` + dimmed.
- **Review fixes**: walks also stop on `samePosition` (the HOLD-tail gotcha — otherwise never terminates and pads 600 duplicate threshold points per entity per tick); landed/parked entities dropped via liveness check (sample one step back == current → past the real track end → null); not-yet-airborne (null current) dropped; plot domain grows to include plotted track samples (corner-cut stretches no longer clipped) and the selected-flight label gets the plot clipPath; `ApproachViewInput.current` tightened to non-null; engages-gate = `trail.some(PRIMARY)`.
- **Perf**: per-entity cache of the classified whole track (`trackCacheRef` keyed by flight id), rebuilt only when geometry deps change (loaded procedure/frame, active routes, source set) — never on clock ticks; each ~120 ms tick then only re-checks liveness (1–2 `getValue`) + classifies the current marker (O(aircraft × track length) → O(aircraft) per tick). `sampleEntityTrack` split into `currentIfLive` + `sampleWholeTrack`; analysis module exposes `classifyProfileSample` / `classifyTrackSamples` / `trackEngagesProcedure` / `sortProfileTracksBySelection` (`buildApproachViewTracks` kept as their batch composition for tests).
- DEFERRED: interior-gap `break` in the walk is latent (current CZMLs are single-interval); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case. Not re-checked in-browser.

### 2026-07-07 — Approach view mirrors the active tab

- New pure `src/data/approachViewSources.ts` `planApproachViewSources(...) → {observed, optimized}` — the single source for "which trajectory sources are the current tab's globe content": `observed` reuses `planObservedTracks(...).visible`; `optimized` = `mode === "optimize" && hasOptimizedSource`. Used by BOTH the hook (what it samples) and the panel's "CZML linked" badge (`profile.sourceLinked`). Observe plots observed; Optimize plots only the optimized playback; Fly/Compare plot neither.
- **Root-cause follow-up**: `planObservedTracks` no longer keeps the observed CZML loaded behind an open view outside Observe (`relevant = mode === "observe"`). `useCzmlLoader` drives the shared `viewer.clock` from the observed CZML's hours-long span, so loading it in Optimize hijacked the clock and made the optimized playback aircraft vanish; releasing it also saves a 100+ MB load per non-Observe tab. Runtime clock behavior not re-checked in-browser.
- KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view — Observe-with-comparison plots neither source; wiring it in is a follow-up.

### 2026-07-07 — Pipeline exposes the control mesh; mesh defaults single-sourced

- `run_scenario_optimization.py --n-segments` (unconstrained) / `--n-seg-per-phase` (constrained), either/or by mode, defaulting to CollocationOptimizer's own 8/3; validated ≥2/≥1 at both CLIs.
- The constrained batch path never passed `n_seg_per_phase` before (stuck at the default with no override) — now threaded main → `optimize_scenarios_constrained_iaf` → `_optimize_one_scenario_iaf` → both IAF selectors → `_solve_iaf` → `CollocationOptimizer`.
- Defaults single-sourced in `collocation/optimizer.py` (`DEFAULT_N_SEGMENTS`/`DEFAULT_N_SEG_PER_PHASE`); backend + batch import them; frontend "Control segs/leg" default aligned 2→3 (defaults had silently diverged: frontend 2, backend fallback 4, optimizer/batch 3). The frontend→backend HTTP path had always wired `nSegPerPhase` correctly — the gap was batch-only. Verified: nsp 2→4 changes the constrained plan 85→125 nodes.

### 2026-07-07 — Optimize tab: solve-time readout; HSL hook benchmarked, kept dormant

- `optimization_backend._optimize` puts `result["timings"] = {buildS, solveS, playbackS, totalS}` in the response (was log-only); frontend `OptimizationTimings` + `parseTimings` (absent → null) + a "Solve time" row. Plan readouts restyled to a single column of full-width rows (3-col grid truncated values).
- HSL: `collocation/components.py` reads `AEROVIZ_IPOPT_LINSOL`/`AEROVIZ_IPOPT_HSLLIB` (casadi's IPOPT has the `hsllib` loader — no rebuild; setup in `docs/hsl-linear-solver-setup.md`). Measured: free Coin-HSL Archive has only MA27, 3–27× slower than MUMPS here (small NLPs; OpenBLAS/OpenMP clash) → dormant, kept for a future MA57 academic license.

### 2026-07-06 — Review pass: toggle false-open, ownership seams, manifest-only eval reads, required guards

- **[BUG]** `runwayMatchesSelection(null, X)` is match-ALL, so with the top bar on "All runways" every toggle showed "Hide profile" and clicks closed the invisible page; `open` now also requires `selectedRunway !== null`.
- `RunwayProfileToggle` prop `borrowSelection` (PilotPanel, unconstrained only): opening saves the pre-open `{selectedRunway, isRunwayProfileOpen}` (only when opening actually changes the selection) and restores on close/unmount. Constrained Optimize doesn't borrow — the forced-display hook owns the runway there.
- `useForcedProcedureDisplay`: `SavedDisplay` now carries the profile-open state, restored together with the runway (ownsRunway only); `forceRunway: null` never touches either. `ready` flip only fires when a drive is pending; savedRef bakes `ownsRunway` (what makes a non-null→null `forceRunway` flip restore correctly).
- **Evaluation read side manifest-ONLY** (user decision, no glob fallback): `evaluation.records.load_records` reads a batch dir via its `summary.json` roster; manifest-less dir / listed-but-missing / empty roster raise; `--pattern` removed from both CLIs. (Globbing counted orphans — the KRDU 1023-vs-996 class.)
- **`min_altitude_m` REQUIRED** on `rollout_controls`/`simulate_controls` — the 0.0 default silently validated diverged replays km below elevated fields; target-less replays now pass an explicit 0.0.
- **Constrained-ness is an explicit manifest field**: `_upsert_category` stamps `"constrained": bool`; builder `--constrained`; frontend `ComparisonCategory` requires it, `_cons`-suffix detection deleted; all 5 airports' `categories.json` migrated in place.
- Cleanups: record-filename suffix constants; shared `_fake_optimizer` test factory.

### 2026-07-06 — Rollout guard margin (residual low-success root cause)

97% of KRDU/asdb gate-failures were rollouts truncated by the zero-margin ground guard at exactly the floor: min-time plans deliberately ride the floor, and cm-scale integration noise (measured: 3.9 cm dip for 1.5 s on a replay that lands 0.7 m out) tripped a guard meant for km-scale divergence. Fix: `ROLLOUT_GUARD_MARGIN_M = 5.0`, guard = floor − 5, both rollout call sites. (Diagnostic lesson: the eval record's `final_time_s` IS the truncated end — compare against the states-file's plan T to detect truncation.) Same day:

- `_clear_stale_records` deletes stale top-level records at batch start (27 orphan evals inflated a report).
- `--fitting-type {hs,trapezoidal,rk4}` end-to-end (`FITTING_SCHEMES`; default hs). rk4 verdict: fine on smooth constrained solves; basin-fragile on aggressive unconstrained min-time (auto M 9.1 km off; only M=64 recovers HS's optimum at ~4× cost) — HS stays default.
- `--state-substeps M` end-to-end through both solve paths + frontend "State substeps" input → backend `state_substeps` on both branches (cache keys include it). Measurements in Key Defaults.
- `ipopt.max_iter` wired ("backend never finishes" postmortem): constrained M=64 ≈ 640 subintervals × per-node inequality rows, no iter cap, solves serialized behind the worker lock. `DEFAULT_MAX_ITERATIONS = 3000`, `CollocationOptimizer(max_iterations=…)` on all three solver constructions; backend `maxIterations` reaches both branches (make_optimizer had accepted-and-ignored it).
- Stale-artifact audit closed the `write_reference_records` and CZML-builder (`clear_stale_outputs`) accumulation gaps; single-file overwrites audited clean.
- NOTE: `runway_cons` off-target populations likely contain the wrongly-truncated family — re-examine after re-run. All categories need re-running after preparation: `python run_scenario_optimization.py --jobs 6`.

### 2026-07-06 — Observe constrained auto-open + shared Profile toggle + forced-display hook

- Extracted `src/hooks/useForcedProcedureDisplay.ts` from PilotPanel. Contract `{active, forceRunway}`: non-null `forceRunway` → the hook OWNS `selectedRunway` (save-once/force/restore-on-inactive + dependency-free restore-on-unmount); `forceRunway: null` (Observe) → never reads/writes `selectedRunway`, drives only the panel + `procedures` layer.
- Observe trigger in ControlPanel: active = trajectories layer ON ∧ comparison overlay ON ∧ constrained category ∧ runway selected.
- Shared toggle component (one parameterised toggle driving the global open state; opening focuses `selectedRunway` on the governed runway) used in ControlPanel, PilotPanel's Target-State header, and ProcedurePanel.
- **Dock-handoff race**: two forcing docks switching share one React passive flush — the incoming hook read the outgoing dock's still-forced display and saved a polluted baseline. Fixed with a one-render `ready` gate (no-op on first commit; the sibling's restore is batched with the `ready` flip in React 18). Integration regression test with the real AppProvider. Verified in-browser (KRDU 05L).

### 2026-07-06 — Evaluation window symlog label pile-up

Under `symlog(v) = sign·log10(1+|v|)` every |v|<1 collapses to ~0, stacking the ±0.01/±0.1 decade labels; `EvaluationReportWindow.tsx` now skips sub-1 decades (keeps 0, ±1, ±10, …).

### 2026-07-06 — All-modes sweep + unconstrained batch's trapezoidal fitting (low-success root cause)

- Omitting `--target-type` runs all three modes (`asdb`/`runway`/`runway_cons`) per airport; `--with-constraint` without it is rejected.
- Unconstrained success 4–14% vs constrained 76–97%: the batch's `optimize_scenario` still used trapezoidal after the 07-05 HS flip (batch-edition seam class). Min-time solves ride the floor exactly where 2nd-order trapezoidal is dynamically unfaithful: plan on-target, rollout 5–15 km off (full-T, not truncated). A/B (DAL1407): trap 5950 m vs HS 3.4 m, and HS found a better optimum at ~3× solve time. Fixed to HS.

### 2026-07-06 — Arrival-segment truncation: 25 km entry ring; locals excluded

Landing tracks were validated by their END only, so depart-and-return flights started ON the field. New `trajectory_data_process/arrival_segment.py`: walk backward from touchdown; the arrival starts after the LAST run of ≥3 consecutive samples outside the 25 km ring (hysteresis: one jittery fix can't cut); plain arrivals also cropped to the ring so every arrival shares one entry boundary (the on-ring state distribution = interaction-study boundary condition). Never-outside tracks: `local` if start ≤5 km (takeoff→circuit, written to `<ICAO>_local_rejected.json`, never silently dropped), else coverage-limited arrival kept whole. `truncate_flights` rebases times to 0 and annotates `arrival_truncated`/`cut_samples`/`arrival_duration_s`/`entry_time_utc`. Integrated in `build_arrivals.py` (renamed from `landings_to_czml.py`): raw `*_landings.json` untouched; derived `*_arrivals.json` feed CZMLs + czml-input; airport centre from `config/runway_thresholds.json`. `FlightScenario.source` carries `entry_time_utc` (co-temporal placement key). All 5 airports regenerated: KMSY 400, KRDU 996, KSJC 319, KSMF 714, KSTL 1054 arrivals. All pre-existing batch outputs predate this (and the floor fixes) — stale.

### 2026-07-05 — Evaluation detail window: legible profiles + one colour language

`DeviationProfile` ranked-dot charts replace the illegible bar walls (lateral: log axis; vertical: signed symlog); each chart draws AND labels its own gate in-plot, dots coloured by that gate, legend carries the outside count. One colour language: red/green = per-flight gate verdict only (summary cards made neutral; scatter legend explains its colours). Values are the backend report's rows verbatim. Verified in-browser (KRDU runway_cons, 1001 trajectories).

### 2026-07-05 — Off-target marking moved onto the RESULT path

"Successful flights ending mid-air" = off-target flights with guard-truncated rollouts (correctly classified; they merely LOOKED successful because only the reference was marked). Now the simulator/result path bakes `OFF_TARGET_COLOR` yellow + "(off target)" name; reference drops to dark-amber `OFF_TARGET_REF_COLOR`; plan keeps legend orange. `useComparisonTrajectoryLayer` skips its repaint for `properties.status == "offTarget"`. NOTE: KRDU runway 32's 77/200 off-target (vs 6–7 on 23L/R) is a real quality signal — likely the replay-divergence family.

### 2026-07-05 — Below-ground trajectories: altitude floors + rollout ground guard

Three root causes for optimized trajectories below field elevation (per-leg step-down floors verified working — the dives lived where no floor existed):
1. The global altitude bound (`components.altitude_floor_m` = target − 300 m, documented as a never-binding box) BINDS — min-time solves dive to it. Fixed: margin 300 → 5 m, a real operational floor; `min(initial, target)` deliberately NOT used (a start below the floor is bad data and fails loudly).
2. The start→first-fix transition phase had no floor above the global one (FFT2071 dove to −173 m). Fixed: transition altitude bound = min(start alt, `_first_leg_entry_floor_m`) − margin (min() IS needed here — a start below the first fix's altitude is legitimate climb-to-join geometry).
3. The batch rollout had no ground envelope (`CasadiSimulator.step` has no checks): `_GroundCheckedSimulator` + `rollout_controls(min_altitude_m=…)`.
Also: evaluation gates judge the FINAL state only, so mid-flight dives never failed a gate — pre-fix success rates were inflated. All batch outputs stale.

### 2026-07-05 — In-app evaluation report window ("Details")

The comparison builder PUBLISHES the evaluation report verbatim (`publish_evaluation_report` → `comparison/<category>/evaluation_report.json`); the frontend fetches that copy and only formats/sorts/plots — no metric recomputed client-side (new metrics go in `evaluation/metrics.py`; contract documented in `src/data/evaluationReport.ts`). `EvaluationReportWindow.tsx`: draggable floating window (portal, same shell as Dynamics-Comparison) with summary cards, 8260.58D gates note (values from `report.thresholds`), aggregates table, three SVG charts, full verdict table. `OptimizationSummary` Details button; report fetched lazily, cached per (airport, category); missing report shows a helpful message. Verified in-browser (KSMF runway_cons).

### 2026-07-05 — Comparison CZML: off-target status + evaluation metrics into the frontend

Builder `--evaluation-report`: verdicts keyed by eval filename, joined to summary rows via `eval_file`; solved-but-failed-gates → status `offTarget` with `lateralErrM`/`verticalErrM` on the index record; `optimization_stats(summary, report)` builds the index `optimization` block (successful/successRate/avgStateErrorM/avgTimeS — nothing recomputed). No report → byte-compatible plain behavior. Runner: the (cheap) evaluation report now always runs before the tails; the CZML step passes it; reusing a pre-evaluation optimization skips evaluation with a loud note. Frontend: `ComparisonGroup.status = "solved"|"offTarget"|"failed"`; flight list flags off-target yellow (`.flight-table-offtarget`, #ffcd28); `OptimizationSummary` metrics were already wired.

### 2026-07-05 — Panel "final horiz err" ≠ playbackDrift, round 2 (verified in-browser)

Two stacked causes behind Δ25 m vs playbackDrift 0.6 m: (1) the LOOP_STOP wrap heuristic never fired (Cesium preserves overshoot on wrap) — replaced all elapsed-based heuristics with `clock.onStop`; `makeReadoutEmitter` returns `{tick, stop}` (stop = throttle-bypassed exact-`stopS` emit, deduped), both playback hooks subscribe to onTick + onStop; Reset/backward scrubs never raise onStop. (2) `czml_common.document_packet` truncated the clock interval end to whole seconds — now `iso_ms` (74.7 m/s × 0.338 s ≈ the phantom 25 m). Backend audited clean otherwise (`playbackDriftM` and the CZML doc end read the same terminal sample).

### 2026-07-05 — Constrained-IAF batch 0% solve rate (two stacked bugs)

1. `scenario_optimization._solve_iaf` still unpacked `segments, _spans = build_constraint_segments(...)` after the 07-03 change to a plain list return — every constrained batch since 07-03 was broken (backend HTTP path updated, batch caller missed). Seam regression test added.
2. The batch's threshold target (from `config/runway_thresholds.json`) sat up to 390 m from the procedure's CIFP threshold → frame-anchor guard fired. `_snap_target_to_procedure` snaps the solve target horizontally onto the procedure's last waypoint (keeps altitude/Vref/pavement heading/glidepath); `_iaf_result` writes the eval record against the SNAPPED target; reference records keep the scenario target. (Diagnostic note: the batch log truncates errors to exception TYPE names per IAF — reproduce one scenario for a real traceback.)

### 2026-07-05 — Evaluation review fixes (degenerate records + HTML escaping)

- One 1-sample "solved" record (rollout truncated at its first step) aborted the whole batch report. Fixed at both ends: producer `_require_usable_rollout` (<2 samples → recorded as a FAILED scenario); evaluation guards via shared `reference.horizontal_arc_length_m` — comparison skipped with a row note, visualize drops undrawable polylines.
- HTML: embedded JSON escapes `</` as `<\/`; `esc()` on every data-derived string reaching innerHTML/Plotly.
- Contract: `final_time_s == states[-1].t` required on solved records; `resample_by_arc_length` rejects n<2; chart labels use unique record file basenames.

### 2026-07-05 — Former combined scenario pipeline runner

The comparison + evaluation runners merged at the time. They duplicated the expensive steps and wrote divergent opt_dir contents (the comparison runner silently overwrote `reference_file` pointers away). Optimization always runs with `--reference-tracks`; tails remain selectable via `--outputs czml,eval` (default both); `--skip-optimize` reuses an existing summary.json. The combined preparation/optimization entry point described here was superseded by the 2026-07-23 split above.

### 2026-07-05 — `evaluation` package (regulation-derived gates)

New root package `evaluation/` — the file-based seam at the end of the pipeline (geokit + stdlib only, never imports the optimizer). Record contract + gates: see Key Defaults. Salient points:
- `evaluation_export.py` maps the true-dynamics rollout onto the contract — `rollout_piecewise_constant`'s samples already carry the active control per sample (nothing re-derived). Both batch modes roll out once and write `*_eval.json` next to every `*_states.json`, INCLUDING failed scenarios (empty record = how solve rate is computed from files alone); summary rows carry `eval_file`. Old `*_states.json` can't be converted offline (no controls) — re-run for eval records.
- Reference records: observed track in the same contract (`flight_scenarios.state_samples_from_track`, times rebased, same target; `controls == []`). `write_reference_records` looks flights up by full identity `(id, icao24, landing_time_utc)` (missing ⇒ raise) → `references/<identity>_reference_eval.json`; `reference_file` stamped on every eval record, failed included. `compare_to_reference` = flight-time delta + path-shape deviation (arc-length resampling — time-matching would conflate speed profiles with geometry).
- CLIs: `python -m evaluation --input <dir> --output report.json` (+ threshold flags); `python -m evaluation.visualize` renders one self-contained HTML (Plotly CDN + embedded DATA; English body, abbreviations expanded on use; 8260.58D citations in the gate-sources note; `--max-tracks` cap stated on the page). Verified in-browser.

### 2026-07-05 — ψ corridor kills the looping/crawling pathology; drift guard; HS default

- Postmortem (nsp=2 loops): the solve CONVERGED onto a local optimum with ±2π winding (feasible — the join ψ box pins one node; excursions cancel), and the node-pinned terminal masked a 4.5–4.9 km true-dynamics rollout drift (trapezoidal on a winding path is node-feasible but dynamically meaningless).
- **ψ corridor** (structural fix): constrained heading variable bounds tightened from ±3π to the route heading hull ± 90° — winding optima cease to exist. The whole 07-04 family of crawls/Max_Iterations/basin-twitchiness died: 2×2 (HEAVE/custom × trap/HS) 4/4 clean, nsp 2/3/4/6 all converge.
- **Drift guard**: `playback_terminal_drift_m` → `playbackDriftM` on every response + stderr WARNING > 50 m. Never touches the NLP.
- Fitting verdict (doglegged H05LZ): trapezoidal 226–296 m drift vs HS 0.6–0.9 m at every nsp; HS ≈ 2–2.5× solve time. **Constrained default flipped to HS** (`_DEFAULT_CONSTRAINED_SCHEME = hermiteSimpsonNormalizedFullTransport`); trapezoidal stays selectable.
- Readout artifact (wall-throttled ~12 Hz emitter missing the terminal sample by up to throttle×multiplier sim-seconds) fixed via `makeReadoutEmitter`, shared by the optimized-playback and Compare hooks — later superseded by the onStop rework above.

### 2026-07-04 — Join constraints: pre-FAF fix passage + flexible FAC intercept + alignment tiers

(Current semantics in Key Defaults.) Highlights and lessons:
- ⑪ Fix passage evolved to the PRE-FAF fix only (all-fixes and entry-fix variants superseded, per user); enforced as a smooth squared form rescaled by `1/(2·tol)` so violations read in metres (no |·| kink); tolerance = the leg's k·RNP halfwidth, procedure-sourced.
- ⑫ Flexible FAC intercept replaced the exact-FAF pin: linear cross-track equality + upstream-only window ≥ L_final/5 before the FAF; `fac_distance_to_ltp` is THE one distance measure (glidepath d reuses it). Vertical semantics stay published-geography-keyed (`LpvFinalSpec.d_faf_m`/`prefaf_floor_m`; both None → gate off, byte-identical).
- Two 2π-branch bugs found by evaluating `nlp.g` at failed iterates and ranking violated rows: the terminal ψ pinned the wrong branch on double-90° routes (fix: `_route_unwrapped_target_psi` walks chained leg courses), and the join intercept `cos(ψ−course) ≥ cos30°` was 2π-periodic (fix: branch-aware linear box). Plus the duration-split regularizer for the flat time-split direction.
- Constraint families refactored into explicit per-family row functions behind a dispatcher (uniform `list[(expr, lb, ub)]`); `components.unwrap_angle()` collapses four unwrap copies. Two-tier FAC heading alignment added (positions-only corridors let large heading errors fit between nodes).
- `log_optimizer_config` writes one stderr line before any solving (optimizer/scheme/fitting/dynamics/transport/constrained).
- Frontend follow-up: custom starts KEEP the RNAV IF selection (the selector names the procedure; the start is independent) so constrained solves can start off-fix; transition threshold unified with the passage tolerance (`_first_fix_join_tolerance_m`) — the old 1–2 km dead zone (no transition, no disc) removed. Verified end-to-end on H05LZ HEAVE with a custom start 9.2 km out.
- KNOWN DATA GAP: per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (926 m at k=0.5) instead of ~278 m (RNP 0.3).

### 2026-07-03 — CIFP thresholds everywhere; displaced-threshold root cause

- Root cause of up-to-970 m target gaps (KSJC 30R): runway.geojson `runway_surface` edges are PAVEMENT ends while CIFP's threshold is the DISPLACED landing threshold, plus a real `build_runway_ring` bug (declared length re-centred on the endpoint midpoint with asymmetric displaced offsets → rigid shift by `(he_disp − le_disp)/2`). OurAirports coordinates themselves verified accurate.
- Constrained target anchored on CIFP: `procedureThresholdAnchor(constraint, document)` — position = the constraint's last waypoint, altitude = CIFP threshold elevation, ψ = final course in the simulator convention; `PilotPanel.computeTrajectory` overrides the request target whenever a `procedureConstraint` is attached. runway.geojson still drives the unconstrained target and rendering.
- `procedure-details/index.json` hoists each runway's CIFP `threshold` `{lon, lat, elevationFt}` (null when uncoded); unconstrained targets prefer it too (`buildRunwayThresholdTargets(collection, index?)`; heading stays pavement-derived). CIFP is also more complete than OA (e.g. KSTL 30L's displacement missing from OA).
- `build_runway_ring` rewritten: `runway_surface` corners ON the OA endpoints; `landing_zone` ends moved inward by the displaced distance. All 5 airports regenerated; landing_zone matches CIFP ≤ 24 m except pure OA data gaps.

### 2026-07-03 — approach_constraints/collocation review fixes

- **One ψ convention**: `course_bearing` now returns the model convention (`atan2(Δn, Δe)`); the optimizer consumes it instead of re-deriving course math (guiding rule: constraint/course math has ONE source — `approach_constraints`).
- `ConstraintReport` unit-aware: metre and radian violations separated (`max_violation()` / `max_angular_violation()`, `is_feasible(tol_m, tol_rad)`) — 1 rad had counted as "1 m", a real false-feasible.
- Transition phase actually built: `_phase_plan` prepends the unconstrained start→first-fix phase (start > 2 km from the first leg's start fix); an approach whose first leg IS the final gets the FAF intercept on the transition phase. `build_constraint_segments` returns a plain list.
- Frame-anchor contract validated loudly; terminal-bank 1-node-phase latent bug fixed (`phase_starts`); per-phase auto state substeps (~3 s target; M-selection deliberately ignores `fixed_duration` so fixed- and free-time NLPs share one decision layout, the fixed solve seeding the free one); glidepath d measured on the same GARP→LTP axis as the lateral corridor; normalized position box 1e7→2e6 m; dead machinery removed (`partition_node_indices`, `LpvFinalSpec.da_hat_m`); README de-staled; mass-frozen approximation stated; HS docstring corrected (4th-order, O(h⁵) local).

### (undated, ~2026-06-28) — Backend SIGABRT root cause + solver isolation

The service "shut itself down": casadi's thread-unsafe symbolic construction under `ThreadingHTTPServer` (three concurrent NLP builds) corrupted the heap → SIGABRT; the old launcher then co-killed the frontend.
- `isolated_backend.py` `IsolatedRunner`: casadi-heavy endpoints (`/optimization/run`, `/dynamics-comparison/run`) run in a worker subprocess (`ProcessPoolExecutor(max_workers=1, mp_context="spawn")`); a native abort → `BrokenProcessPool` → clean `SolverCrashError` 500, next request spawns a fresh worker (self-heal). Decorator backends keep the same method interface (tests inject in-process fakes).
- Memory-aware worker lifecycle: `POST /optimization/session/{open,close}` (+ dynamics-comparison twins) ref-count ONE resident warm worker tied to the frontend tab (spawn costs ~1.2 s/call otherwise); no session → ephemeral worker per call; idle watchdog (`AEROVIZ_WORKER_IDLE_TIMEOUT_S`, 600 s) reclaims stranded workers. Frontend `workerSessionClient.ts` opens/closes per Pilot sub-mode + `navigator.sendBeacon` close on `pagehide`; all best-effort, the watchdog is the backstop.
- `casadi_lock.CASADI_LOCK` (RLock) serializes every in-process casadi entry point incl. `SimulationBackend.reset/step` (previously unlocked).
- `start_aeroviz_fullstack.sh` rewritten as the supervisor (see Gotchas). Spawn-picklable probes live in `isolation_probes.py`.

### 2026-06-28 — Scenario→optimization→CZML pipeline (scaffolds, since filled) + `flight_scenarios` seam

- `scenario_optimization.py` writes one `*_states.json` per scenario = `{source, final_time_s, optimizer_states[], simulator_states[]}` (states `{t,lat,lon,alt,V,psi,gamma,m}`); `optimizer_states` = NLP node states, `simulator_states` = controls rolled through `CasadiSimulator` (lives in the `aerodynamic_model` layer — never imports the backend above it). `build_scenario_comparison_czml.py` renders reference/optimizer/simulator as three coloured time-dynamic paths.
- `flight_scenarios` package: `scenario.py` (record + JSON round-trip + `aircraft_for_code`), `start_state.py` (track → initial state via two-sample finite difference), `build.py`, CLI `python -m flight_scenarios`. `final_state_from_track` populates `FlightScenario.target`.

### 2026-06-28 / 2026-06-27 — `geokit`: one geodesy/units source

- `geokit.constants`: WGS84 (`WGS84_A`/`_E2`/`_B`), `SPHERE_RADIUS_M` (default WGS84 a; switchable `EARTH_RADIUS_MEAN_M`), `NM_M`/`FT_M`/`KT_MS`/`METRES_PER_DEG_LAT`/`DEG2RAD`. `geokit.geodesy`: haversine, equirectangular, bearing, flat-distance, metres-per-degree, bounds. `geokit.units`: exact speed (`kt_to_ms` = 1852/3600, ft/min, km/h, mph) + length (`nm_to_m`, `ft_to_m`) conversions — replaced the truncated `0.51444` and 4 divergent Earth radii project-wide; `aircraft_sets.py` SI-mirror fields derive from it.
- Frontend: `geokit/scripts/export_constants_json.py` → `src/generated/geoConstants.json`, re-exported by `procedureGeoMath.ts` (~16 files migrated off local constants); drift-guard test fails if the JSON drifts. Fixed a real bug: two TS modules had used different haversine radii.
- Aero layer imports constants only (symbolic functions untouched). The 30 km study was regenerated for the ~0.1% WGS84-a shift (conclusions unchanged). Pedagogical flat-Earth `runway_bearing_rad` stays local by design.

### 2026-06-27 — Full (exact) geodetic transport as an explicit option

The geodetic RHS's ψ transport had silently dropped a cross term `V·sinγ·sinψ·cosψ·(1/(R_N+h) − 1/(R_M+h))` (~3–4 orders below the main meridian-convergence term; γ transport was already exact). Now `transport ∈ {"none","approx","full"}` on `make_geodetic_dynamics_model`/`make_geodetic_step_integrator`; `"approx"` = historical default (byte-identical). New `*FullTransport` (+ Normalized) schemes end-to-end (backend names `casadiDirectCollocationNormalizedFullTransport(+Trapezoidal/+Rk4)` etc.; frontend Dynamics options). Compare mode gains opt-in system F (full transport). `transport_term_comparison.py` + zh doc: divergence ~mm over 120 s, dt-independent (vector-field, not truncation). Decision: default stays approx; no silent approximations going forward.

### 2026-06-25 / 2026-06-24 — RNAV(GPS) tutorials + canonical ProcedureConstraint + CIFP block-altitude fix

- Tutorials: `aeroviz-4d/docs/34-how-to-read-rnav-gps-approach.{zh,en}.html` — self-contained, interactive (auto-wrapped glossary tooltips, SVG fix/segment hotspots, slide-in glossary panel), worked on KRDU RNAV (GPS) Y RWY 5L. Verified in-browser.
- **Canonical `ProcedureConstraint`** (front↔back): `src/data/procedureConstraint.ts` + Python mirror `aeroviz_backend/procedure_constraint.py` — one JSON shape (ordered waypoints with altitude windows + final course + glidepath + nominal speed); `buildProcedureConstraint(document, {branchId})`. CIFP→`AltitudeConstraint` conversion unified into one `altitudeConstraintFromCifp` (the two copies had diverged — one dropped block upper bounds).
- **CIFP block-altitude fix**: ARINC 424 "B" descriptor is a WINDOW (at-or-below Alt1, at-or-above Alt2); the parser had dropped Alt2. `ProcedureLeg.altitude_ft_2` added; KSTL regenerated. Chart-cross-referenced golden test (`test_krdu_r05ly_matches_published_rnav_gps_chart`) guards the parser against published-chart values. Documented gap: leg speed restrictions not extracted (cifparse exposes no speed field; 0 coded in the dataset) — `speedMaxKt` is ready when a source appears.

### 2026-06-24 / 2026-06-23 — Normalized geodetic scheme (conditioning fix)

- Root cause of Max_Iterations on H05LZ N=10 free-time solves: conditioning, not the seed — radian lat/lon (~1e-6 rad/s derivatives) next to metre/m-s states; the `1/(R_M+h)` factor makes position defect rows ~6–7 orders smaller than altitude rows.
- Fix: `*Normalized` schemes reparameterise the decision state to metres from the target anchor (`n=(lat−lat_t)·R`, `e=(lon−lon_t)·R·cos(lat_t)`) — an EXACT affine change of variables (unlike localEnu's flat-tangent approximation); same geodetic RHS inside the defect. Robust across N and arrival windows; identical trajectories on benign problems. The localEnu cold-start hybrid (an earlier workaround) was removed as superseded. Tutorial: `4dTrajectory/docs/geodetic_state_normalization.zh.md`. Compare mode gains system N (normalized) which overlays C — live proof the reparameterisation changes nothing.

### 2026-06-23 — Dynamics Compare mode (Pilot panel) + follow-ups

Third Pilot-panel mode: flies the start state under one constant control as the study's systems — A fixed-tangent ENU (anchored at the START in Compare mode; the 30 km study anchors at the target), B per-step re-anchored (reference), C geodetic RHS +transport, D no-transport, opt-in N (normalized) and F (full transport) — replayed as coloured, hideable CZML paths on Cesium's clock with deviation charts (horiz/alt/head/speed/fpa vs B) and a final-value table.
- Core extracted to `dynamics_comparison.py` `compare_dynamics(...)` (30 km study now calls it, output byte-identical). Endpoint `POST /dynamics-comparison/run` (`dynamics_comparison_backend.py`); shared `czml_common.py` (epoch/iso/document-packet) + `responseValidators.ts` back both backends/clients.
- Frontend: `dynamicsComparisonClient.ts`, `useDynamicsComparisonPlayback.ts` (loads CZML, drives the clock, hides systems, camera-follow), `DynamicsComparisonCharts.tsx` (draggable portal window — see the backdrop-filter gotcha). Per-system tinted aircraft models (`colorBlendMode: MIX`) oriented via `VelocityOrientationProperty`, wrapped in `makeStableVelocityOrientation` (a CallbackProperty returning the last valid orientation when HOLD extrapolation zeroes the velocity — otherwise the parked model snaps to a default attitude).
- Live State panel shows B's state (dense backend `samples` in the trajectory-play shape) + colored per-system delta chips interpolated from the chart (`interpolateComparisonDeltas`); `fpa` tracked end-to-end as an error metric. Trajectory Play reuses the same chip strip for a live Δ-vs-target readout (replaced the old Lat/Lon/Alt Error rows).
- Custom start state in Compare (RNAV fixes + runway select); run history persisted per run (`dynamics_comparison_history.py`, git-ignored dir) with backend-averaged history endpoints (common distance grid, shortest run's range) + frontend Average/Clear buttons.
- Review fixes: rollout never records sub-surface/non-finite samples (stops + truncation note via `requestedDurationS`); endpoint-inclusive `even_sample_indices`; double-checked-locked integrator cache; chart memoization; START preview hidden during comparison playback.

### 2026-06-23 — Optimizer = dynamics × fitting; shared stall model

- `_DEFECT_SCHEMES` entries are `(make_dynamics, make_defect)`: localEnu is a CONTINUOUS dynamics (fixed ENU tangent frame) collocatable with any fitting (defect converts geodetic nodes into the target-anchored ENU frame via `geodetic_state_to_enu_expr`); only `reanchoredEnu` stays shooting-only (per-step re-anchoring is discrete).
- Shared stall model `aero_params_for_aircraft(aircraft)` (mass-based Cl_max, A320 ≈ 2.7) used by optimizer AND playback — they had diverged (2.7 vs 1.5), replaying optimized trajectories ~1.6 km off.
- Cold-start hybrid (`cold_start_scheme`: fixed-time seed solved with a cheaper dynamics, free-time refines) was added with a whole-flow timing stderr line (`log_optimization_timing`: build/coldStart/freeTime/solve/playback/total) — the hybrid was later removed (superseded by normalization); the timing log remains.

### 2026-06-22 — Pluggable defect schemes; solver backend verdict; CZML playback; dense state

- **Defect schemes**: trapezoidal (order 2) / Hermite-Simpson (order 4, default) / RK4 (order 4, shooting) on the continuous geodetic RHS + `reanchoredEnu` (the playback integrator as a shooting defect). HS and RK4 are the same order — they differ in construction (implicit collocation vs explicit shooting). A stepper can be a shooting defect but NOT a polynomial collocation defect. `collocation_scheme_comparison.py` accuracy ladder: trapezoidal ~5 m vs HS/RK4 sub-metre. Frontend split the optimizer choice into Dynamics × Fitting dropdowns (`optimizerToParts`/`partsToOptimizer`/`validFittingsForDynamics`); legacy optimizer names remain valid on the backend. Tutorial `4dTrajectory/docs/collocation_schemes.zh.html` (interactive convergence demos).
- **`localEnu` scheme + 30 km study** (`dynamics_comparison_30km.py` + zh doc): fixed local-ENU @ target ≈ 335 m horiz error over 30 km, RHS-no-transport ≈ 145 m, full RHS +transport ≈ 0.03 m (validates RHS ≡ re-anchored). `make_local_enu_step_integrator(ref_geo)` reduces exactly to the re-anchored stepper when ref = current point.
- **Solver backend**: `solver_backend` switch (ipopt/sqpmethod) exists but sqpmethod is NOT usable — cold it bails instantly from linear-interp guesses; warm-started (needs duals, exact Hessian) it's still slower than cold IPOPT because CasADi's sqpmethod uses a dense active-set QP (~300× per-iteration cost, no banded OCP structure). IPOPT stays the only exposed backend; a real warm-start payoff would need acados/HPIPM. `solver_backend_benchmark.py` documents this.
- **Playback**: the optimized trajectory plays as backend-built CZML on Cesium's clock (`trajectory_playback.build_optimized_trajectory_playback` — rolls the N piecewise-constant controls once through the SAME geodetic integrator as the live sim, sub-mm match). Trail = one short polyline per sample interval with ms-precision availability (grows behind the aircraft), coloured by control segment (blue→red). The aircraft packet carries NO orientation — the frontend sets it from the sampled state with the live-Pilot convention (`headingPitchRollQuaternion`, heading −ψ, pitch γ+α, roll −μ). `useOptimizedTrajectoryPlayback.ts` drives the clock + throttled ~12 Hz readout sampling (`sampleTrajectoryAt`). Rollout truncates (not raises) on envelope exit — it's a viz aid. Manual Pilot mode unchanged (interactive).
- **Dense-state collocation**: control on N segments, state collocated on N·M sub-intervals (`sub_steps`; auto ~3 s, cap 16) — fixed the km-scale optimizer→playback mismatch (coarse discrete operator ≠ fine playback RK4). The multiple-shooting polish machinery was REMOVED (dense-state raw solutions are playback-consistent). Raising nSegments instead would refine control too → the "wrinkle" convergence pathology.
- **CIFP transition-altitude misparse**: initial fixes with no published crossing altitude were placed at the procedure-wide Transition Altitude (18000 ft) → infeasible starts. Parser no longer falls back to `trans_alt`; qualifier from `alt_desc`; an IF with no own altitude derives one by interpolating from the nearest published fix on the branch (`derivedInitialFixAltitudeFt`) rather than being dropped. Postmortem: `aeroviz-4d/docs/33-cifp-transition-altitude-misparse-postmortem.md`. (The ready-made cifparse/arinc424 packages had the same fallback bug.)

### 2026-06-21 — Geodetic continuous dynamics for direct collocation

Replaced the fixed-ENU transcription with one continuous geodetic RHS shared by optimizer and playback: `make_geodetic_dynamics_model` (point-mass RHS in `(lat, lon, h, V, psi, gamma)` radians; position kinematics via WGS84 `R_M`/`R_N`; transport terms on ψ̇/γ̇) + `make_geodetic_step_integrator` (RK4, degrees externally). Validation (`geodetic_vs_reanchored_error.py`): geodetic+transport tracks the re-anchored RK4 playback to ~0.3 mm over 5 km; without transport ~2.9 m drift. Interactive doc `geodetic_dynamics_transport.zh.html`.

### 2026-04-20 — OCS geometry + FAA DOF obstacle layer

- `ocsGeometry.ts` `buildFinalApproachOCS` implemented (primary trapezoid + two 7:1-slope secondary panels); `useOcsLayer` renders three semi-transparent polygons per route (FAF→threshold pairs from `procedures.geojson`; primary half-width from the route's tunnel descriptor, 150 m fallback); `ocsSurfaces` layer toggle. Altitudes read from LineString z-values (CIFP geometry alt); switching to MCA is a one-function change documented in `docs/03-ocs-geometry.zh.md` §5.6.
- `preprocess_obstacles.py` parses fixed-width DOF `.Dat` (haversine radius filter, default 20 km) → `obstacles.geojson`; `useObstacleLayer` renders type-coloured cylinders (`RELATIVE_TO_GROUND`) with AGL labels; `obstacles` layer toggle.

### 2026-04-19 — DSM terrain hook

`useDsmTerrainLayer` rewritten onto the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`); returns `{status, metadata, provider, error}`; wired in `CesiumViewer` + demo page; `dsmTerrain` layer toggle.
