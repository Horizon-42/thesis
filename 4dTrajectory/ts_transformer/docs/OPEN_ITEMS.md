# ts_transformer — open items

Status log for the package: what each campaign settled, what it adopted, and what is left.
Moved out of `CLAUDE.md` so it is read when planning work, not injected into every session.
Newest campaigns first, long-standing scope limits last.

---

## Decisions left open by the 2026-09-09 package review (branch `dev-pkg-review`)

The review's bugs are fixed (its §7 is the ledger; changelog entry of the same date). These
are the items that change a stored artifact's name, reuse or number, or a protocol, and so
were NOT made without the owner:

- **C-3 — DECIDED and DONE 2026-09-09: the three identity-bearing fields name the run.**
  `lr_plateau_patience` / `lr_plateau_factor` (`lr-patience=` / `lr-factor=`) and
  `random_train_anchor_min_future_s` (`anchor-min-future=`) are in `META_FIELDS`; only `device`
  and the never-set backbone knobs stay in `KNOWN_UNNAMED_FIELDS`. Measured over the 219 stored
  `history.json` configs: 146 display names / slugs moved (75 gain a spelled token, 71 only
  their folded `+N more` count and slug hash — the named recipes pin the scheduler pair, so
  recipe runs are untouched), 0 loadability changes, no directory or category key moves. 132
  published frontend categories carry the old label until relabelled — 109 publisher-managed
  (`publish_ts_experiment_trajectories.py --refresh-labels-only`, once per publication root:
  `KRDU/`, `KSJC/`, `POOLED/experiment_predictions`, `POOLED/checkpoint_publications`) and 23
  hand-published `ts_*` keys (`docs/relabel_published_categories.py`); labels only, no CZML,
  records, keys or directories. Owner-run, one pass.
- **C-4, the duration floor under `uniform`.** `control_duration_uniform_floor` is read by the
  `factorized` head only, but its default is 0.8 and every recipe pins 0.0, so 88 stored
  `uniform` runs carry a non-default inert value and a refusal would stop them loading. It
  stays a field on `DurationSpec` after the §4.3 split (2026-09-10): a `Factorized(floor)`
  variant would refuse the `0.0` every recipe pins under `uniform`, i.e. every recipe. A custom
  `uniform` run can still wear `duration-floor=`; retiring the pin is a recipe-version change.
- **C-7, the test-release ledger is bound to the DIRECTORY.** `test_release.json` sits beside
  `checkpoint.pt`; a copy of the checkpoint elsewhere can be frozen and released again. The fix
  is a registry keyed by the checkpoint digest outside the run directory — where it lives and
  how tests isolate it is the decision. No ledger exists on disk today, so it costs nothing
  whenever it is made.
- **C-9, `cv_results.json` misnames the selection metric as a loss.** `mean_/std_val_macro_loss`
  at the candidate level hold the SELECTION value (ADE, metres); the fold rows' `best_val_macro_loss`
  is the loss. Renaming means a schema bump, after which the two stored files (2026-08-16
  `POOLED/ts_patchtst_normalized_time`, 2026-08-18 `KSJC/experiments/cv_tau_bank_20260818`)
  are no longer reusable by `--skip-cv`. The writer says so in a comment; nothing renamed.
- **C-12, the load-factor floor — DECIDED 2026-09-09: the grader stays at 0.5.** The learned
  head's box floor is 0.2 (`control/envelope.py`) and `flyability`'s hard floor is 0.5, so a
  control-path segment at n ∈ [0.2, 0.5) is unflyable by construction on the published metric.
  Neither number moves: the grader's floor is the flyability CLAIM every published number was
  read against, and the head's box is the old path's search space (moving it would change every
  control checkpoint's decoded controls). The plan-and-guidance design's guidance layer commands
  the load factor inside `flyability`'s envelope, so the new path cannot inherit the gap. Still
  unmeasured, and only of interest for reading the old arms: how often a trained head emits
  n < 0.5 (`control/training/diagnostics.py` saturation counts would say).
- **A-2's consequence.** Every number `run_ts.py history_ablation` published before 2026-09-09
  was scored against a truth taken `(max L − L)·dt` before its anchor; the runner is unchanged
  and correct now, its stored outputs are not.
- **§5 DONE 2026-09-10** (the review's §5 resolution paragraph has the per-axis outcome and
  the two corrections: the observed clock and scaled-tcv are recipe-pinned and stay
  selectable; corridor-bounded stays the candidate default). **Still to do, in the review's
  §6 order:** §4.3 config split DONE 2026-09-10 (typed views over the flat dataclass, the
  ownership rule; defaults unchanged; 0 names moved); §4.2 output strategies DONE 2026-09-10
  (`outputs/`, one strategy per path, the spine no longer branches; 0 names moved); §4.4 loop /
  predict extraction DONE 2026-09-10 (`prepare_session` / `train_epoch` / `validate_epoch`;
  `parse_predict_options` → `PredictOptions`); §4.5 runners DONE 2026-09-10
  (`experiments/<name>.py` behind `run_ts.py <name>`, `repo_layout.py`, the twelve red pipeline
  fixtures fixed — the suite is green); then §4.6 tests (support module, the split of
  `test_ts_transformer.py`), the folder grouping — each a commit with the full suite and the
  stored-run census as the acceptance test. The "name every field against
  the nearest recipe" grammar change is deferred to after the folder grouping (it moves
  stored names and needs its own relabel pass like C-3's).

## Current state (2026-09-07) — the latent-intent design supersedes everything below it

The control path was redesigned on 2026-09-07 (`docs/2026-09-07_latent_intent_design.zh.md`,
its §〇 status table is the live one). What that design settled and what is running:

| step | state | number |
|---|---|---|
| L0 width oracle | done | N\* = 32: 96 operating numbers replace 257 (vectored fit 203 m against 962 m of intent) |
| L1 low-dim head | done (`l1_lowdim_20260907`, `docs/2026-09-07_l1_lowdim_results.zh.md`) | native32 1322 m ≈ N=64 baseline 1333; dense / no-teacher 2515 — the trajectory-error loss alone is NOT enough, the imitation teacher stays |
| L2 latent intent | four campaigns run (L2.c β ladder, L2.d warm posterior, L2.e′ free bits, and the posterior probe that explains them); **L2.f is next** | L2.d's warm β=0.01 is the best point estimate so far (1214 m pooled against native32's 1322) and the fallback. The probe found the failure the totals hid: in all three L2.d/e′ arms the posterior MEAN sits on the prior mean (< 0.2 prior σ) and the KL is spent narrowing q, so z is a denoised constant — which is why shuffled ΔADE ≈ 0, the z-oracle bought 227 m and N(0, I) samples beat the trained prior. L2.f puts information back in the mean, one arm each: `latent_beta_warmup_epochs=40` and `latent_aux_duration_weight=1.0`, arms `docs/experiments/l2f_mean_information_arms.json`, code on `dev-l2f` with the epoch-1 diagnostics (`run_ts.py latent_readout --history`) and `run_ts.py latent_probe` |
| L3 CTA conditioning | code done + reviewed (`dev-l2`); the delivery layer for a LATE CTA is three predict-time hooks, L3.c/d run and L3.e built but **not run** | `cta_conditioning=given`, `predict --cta-offset-s`. L3.c: the barrier holds the corridor but the trajectories stall on the outer segment. L3.d: a speed floor alone has nowhere to put the delay (total time fixed by the CTA + path fixed ⇒ mean speed fixed) and trades stall for thrust-over-max, arriving early and flying on. L3.e (`dev-trombone`, 2026-09-08): `control/constraints/trombone.py` + `barrier+trombone` / `barrier+speed-floor+trombone`, and `predict --truncate-at-threshold` so the readouts cover the approach and not the flying after it. Arms `docs/experiments/l3e_path_stretch_arms.json` (KRDU val, four predict-only, dry-runs clean); **known cost: ~58 % of the fleet is already aligned with the course at the anchor and cannot be stretched at all** — its unabsorbed delay is reported as `tromboneDelayS` with an engaged share of zero. See §六 L3.c/d/e of `2026-09-07_latent_intent_design.zh.md` |
| L4 scene encoder | **NOT built — gate failed** | scene entity features add nothing (d_join R² 0.37 vs 0.38); the observable lead ETA correlates 0.11 with the lead's true landing |
| L5.a fitted teacher | code done + reviewed (`dev-l5`); its startup blocker fixed (`e8df12f`) | the fit itself is a GPU job and **has not run**: `docs/experiments/l5_fitted_teacher_arms.json` points at `4dTrajectory/outputs/KRDU/experiments/l5_fitted_teacher_20260907/basis_fit.json`, which **does not exist until** `run_ts.py control_basis_oracle --checkpoint .../l1_lowdim_20260907/L1_native32/checkpoint.pt --out <that directory> --splits train,val` has been run (8255 flights, ~2–3 h). Until then both arms die at the dataset build, by design — the config carries the path, the dataset opens it. It used to die EARLIER, at the provenance check, because it fingerprinted the manifests without the pre-split lateral-pass roster (14 435 candidates against the checkpoint's 14 378 eligible); `e8df12f` moved that rule into `data_provenance.checkpoint_data_provenance`, which every replaying runner now calls |
| A/B anytime prediction + calibrated ETA | **A0 / A1 / B0 measured; A0.b code done and reviewed (`dev-a0b`), its two arms NOT trained**; the rest design only | `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §〇 is the live status table (commands §〇.1, B0's numbers §〇.2). Built and measured: `run_ts.py anytime_curve` (A0-fixed over a remaining-path anchor grid, strata fixed at L−1) and `run_ts.py eta_error_readout` (B0, CPU readout) — KRDU val vectored \|Δt\| p80 **65.8–72.5 s** against straight-in **12.0–20.3 s**, so per-stratum ETA calibration is necessary and B's 120 s veto has about one doubling of margin. **Three A0-random arms have now run and all fail the L−1 veto** (2949–2990 m against native32's 1322); §2.4c's diagnosis is two mechanisms — the LR scheduler stepped on a stalled selection metric (9.4e-7 by epoch 60) and a time-uniform anchor draw that, pooled over flights, sits nearer the runway than its own anchor population (draws 34.3 % under 6 km against a population 25.1 %). **A0.b is the fix and is CODE ONLY**: `lr_plateau_metric` + `random_train_anchor_sampling=remaining-path-uniform`, arms `A0b_lr_objective` / `A0b_lr_objective_path_uniform` in `docs/experiments/a0_random_arms.json` — a GPU job that has not run, and until it does the random-anchor line has no established base. Still design: A2 / A3, B1–B3 (quantile duration head + split-conformal + `--cta-from-quantiles`, the first CTA arm that does not read the future), deliverable B4 = 80 % interval width vs remaining path and `s_freeze`; C = conditional diffusion over the 96 operating numbers trained on the L5.a fitted table, guided THROUGH the differentiable rollout (observed prefix / CTA / corridor), a REPLACEMENT for the CVAE if L2.e′ leaves z under ~1 nat, gated first by a retrieval ceiling on the same table (C0) |

**Abolished by that design** (entries below are history): the P1.d closure tracker (its BLOCKER
is not being fixed; the code was DELETED 2026-09-07 by audit T1-9), the K join-anchor decoder,
the `2026-09-07_control_training_review` P0/P1 objective fixes. The closure output stays as a comparison arm only.

---


- **Scene design Phase 0 DONE 2026-09-05 (`scene_phase0_20260905`, KRDU; results
  `docs/2026-09-05_scene_phase0_results.zh.md`, diagnostics
  `docs/phase0_intent_diagnostics.py`).** The TRUTH join point as input: vectored ADE
  2858 → 2356 m, duration error 39 → 22 s; + the truth lead ETA: no increment; + the truth
  remaining time: 2011 m (−30 %), pooled 1005 m, duration error 5 s — but the time-free path
  error never improves (chamfer 942 → 791 → 850 m). Pre-registered gate (< 1.5 km) not met
  and shown to be mis-sized (truth path + naive speed profile 1.3 km; trombone from the truth
  join + truth timing 1.7 km). Open — a decision for the user: (A) stop, (B) revise the design
  to (d_join, T) decisions, (C) fix the output side first (geometric closure); the results
  doc recommends C then B. **P0 geometric readout DONE 2026-09-05** (`geometric_metrics.py`
  in both readouts; `readout_geometry.*` backfilled in `scene_phase0_20260905` and
  `control_hooks_v2_20260906` KRDU + KSJC): the join arms gain 15–20 % on chamfer / Fréchet /
  arc-ADE, the duration arm gains none of it; the v2 soft barrier is the only hook that also
  improves vectored geometry (KRDU chamfer 942 → 886 on 81 % of flights). The state output's
  saw-tooth polyline (heading reversals at ~50 % of nodes, length ratio ≈ 2) takes it out
  of the arc family (chamfer / Fréchet still read) — a finding about the state export, not
  yet acted on. **Direction C chosen; P1.a / P1.b DONE 2026-09-05** (`closure_geometry.py`,
  `closure_profile.py`, `docs/p1_closure_oracle.py`; artifacts `closure_p1_20260905/`):
  the via-pose Dubins family (F3) fitted to the truth reaches vectored chamfer p50 180 m /
  Fréchet 1179 / truth-timed ADE 510 m (gate passed), with identifiable canonical labels
  on 96 % of fitted flights; on the truth path a K=4 slowness + K=4 height profile
  reaches 110 m. **P1.c DONE 2026-09-06 (`closure_p1c_20260905`, report
  `docs/2026-09-06_closure_p1c_results.zh.md`): both gates pass.** The `closure` output
  from the ego history alone beats simple-v3 on every stratum (pooled ADE 996 vs 1333,
  vectored 2197 vs 2858, straight-in 310 vs 469); with the truth (d_join, T) as inputs it
  reaches vectored ADE 1235 / chamfer 492 where the control head with the same inputs sat
  at 2011 / 847 — the output side WAS the bottleneck; the family's own ceiling is 455 m.
  Open: flyability, read as a delta — closure paths are 22 % fully flyable under the
  clean polar against the control baseline's 0.1 % (its violations are 59k stall
  samples) and the observed 98 %; per sample closure sits at 99.8 % = the observed, its
  few violations per flight being bank jumps at the CSC junctions and thrust jumps at
  the knots. **P1.d DONE 2026-09-06 (option b, report
  `docs/2026-09-06_closure_p1d_tracking_results.zh.md`)**: the drawn reference flown by
  the point-mass rollout under `control/constraints/closure_tracking.py` (RETIRED — code
  DELETED 2026-09-07) costs ≤ 100 m
  of ADE (C_pred +9 m pooled, +25 m vectored) and brings the fully-flyable rate from 22 %
  to 92 % (observed 98 %) — the delivery form is closure + tracking. **Review (opus, same
  day) found the tracker's nearest-node search unguarded: 8 of 1404 via-Dubins flights
  snap to the wrong leg (endpoints 6–19 km off) and four of the five largest "tracking
  gains" are those; the pooled cost is +10.5 m without them. Five SHOULD-FIXes
  (vertical-law sign untested, stall floor without the commanded load factor, ISA
  density on chart height, the height pinning outside the label contract, the gain
  docstring). Unfixed: the user decided to REDO the plan** — the design doc's §〇.1 is
  the snapshot to resume from. The P2 data plane (`trajectory_data_process/scene_index`,
  `flight_scenarios/scene_context`, `ts_transformer/scene/features`) is committed as WIP
  (`045c233`): 13 tests, not reviewed, no explainability measurement, no KRDU index
  built. Not started: any scene encoder.
- **Hard procedure constraints in TRAINING — survey + plan written 2026-09-08, nothing built
  (`docs/2026-09-08_hard_constraints_survey_and_integration_plan.md`; 83 papers annotated with
  their core formulas in repo `docs/literature/procedure_hard_constraints/`).** What the
  literature settles against our evidence: the tanh-bounded state output is HardNet-Aff's
  closed-form clamp (its Prop. 7 clamp-vs-tanh is a cheap arm); the lazy control network is the
  known result of training through a filter with `CC` bookkeeping and is removed by a swept
  correction penalty `α‖u_raw − u_filtered‖²` (Pizarro Bejarano 2025), the OptLayer-CPC form
  (raw rollout scored beside the filtered one), a feasible-entry committed gate, or a bijective
  gauge map onto the barrier's bank interval — and predict-time-only filtering carries a
  quadratic-in-horizon imitation error (Geiger & Straehle 2022), so it is not the end state; the
  C_dual divergence was an unreachable level (dual best response `+∞`), fixed by annealing /
  learning the level (Hounie 2023) with νPI. Plan H0–H6 in §4: H0 gate readout (no GPU) → H1
  learned monotone commitment gate on `state` → H2 control arms on the L1.c base (after L1.c,
  every arm predicted with AND without the hook) → H3 vertical barrier → H4 only if L1.c passes.
  Work lives in the `../thesis-hc` worktree (`dev-hard-constraints`), unmerged.
- **Control command-hook campaigns DONE 2026-09-06 (`control_hooks_20260906` v1 at KRDU,
  `control_hooks_v2_20260906` at KRDU + KSJC; report
  `docs/2026-09-06_control_hooks_results.zh.md`).** Adopted: the v2 soft barrier as a
  predict-time safety layer; not adopted: any hook inside the training loop (six arms, none
  beat its predict-time counterpart), the hard gate. The nominal law's code was archived
  2026-09-07 (`archive/nominal_law_hook_2026_09/`; it was the unadopted hook's only
  remaining consumer once T1-9 deleted the closure tracker). Open: the combined
  lateral-barrier + vertical-nominal hook at predict time — reviving the vertical half is a
  deliberate un-archive, not an import; the baseline ending 157 / 162 m below the glidepath
  — traced 2026-09-07 to the last minute of the rollout (on the final it sits within ±13 m
  of the glidepath; at the truth's landing time it is 540–680 m short and 140 m low, path
  angle −4…−5° vs −3°), NOT to data, coordinates or the fitted tail (2–6 s), and read as
  four objective-design faults — the isotropic 10 km metre-scale position loss (both
  paths) prices a 150 m height error at 2e-4 per endpoint (a mean over 64), the 47×
  open-loop imitation teacher never speaks to the rollout's own drift, the threshold
  anchors present (the 1.25-weight fitted terminal row, `state_endpoint_loss_weight`)
  share the 10 km scale and nothing stops at the ground, and the path loss carries no
  gradient to the time grid (training rescales durations to the truth; the overrun is
  inference-only):
  `docs/2026-09-07_control_training_review.zh.md` (P0: per-channel position scale, the
  vertical-only procedure term, a threshold-plane crossing loss; P1: a closed-loop
  DAgger-style teacher from the guidance laws). **The km-level error is elsewhere**: vectored
  flights carry 76 % / 60 % of pooled ADE and their error is the ATC join decision, which the
  ego-only input cannot see and a single-output head can only average — design for traffic
  context + join-anchor multimodal output (scene encoder, K join-distance anchors as decoder
  queries, WTA training, top-1 stays on the existing record contract, Phase 0 = oracle
  upper bound before any architecture work): `docs/2026-09-07_scene_join_anchor_design.zh.discard.md`; a "committed to the final" gate for the vectored flights the v1 / hard
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
- **Runway-hypothesis expansion DONE 2026-09-03 (`run_ts.py runway_hypotheses`, no training):**
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
  that. Runner `run_ts.py frame_ablation` (state arms, val split, resumable, no CV, no CZML;
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

- Open questions carried over from `docs/notes_7_20.md` (2026-07-20, deleted 2026-09-07):
  why 300 steps?
  figure to show fail of patchtst
  
  cross valition 
  accuracy ???
  hard constraints
  which baseline? with or without dynamic model
  multiple different 
  evaluation them with hard constraints and dynamic
  obliation study
  
  3 weeks for baselines 
  
  Diffusion Model??? guide diffusion; classifier free guidence; tricky to optimize.
  Temp output; video generation; low dimension;
  
  moudlize repo; interfeces
  
  improve;
