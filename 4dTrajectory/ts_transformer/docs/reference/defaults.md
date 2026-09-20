# ts_transformer reference — current defaults, selection metrics, anchor grid, command hooks

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new (the defaults table's rows became `D1`–`D24` sections: axis and default in the heading, the status cell verbatim below it). Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

## Current defaults and their status

### D1 · `coordinate_frame` · default `enu`

keep — the airport frame makes the model average across parallel pairs. `airport-enu` / `runway-aligned` are FROZEN (2026-09-09, `COORDINATE_FRAMES_AVAILABLE`): the 2026-09-03 arms load, no new run selects them

### D2 · `state_position_reference` · default `absolute`

`corridor-bounded` ADOPTED as candidate default (4 seeds, no regression); **`anchor-relative` is VETOED by its own pre-registered rule.** It follows the package's one mechanism for a value like this: it is in `STATE_POSITION_REFERENCES` (what a STORED config may say, so the 2026-09-03 `state_v2_20260903/A_anchor_relative` artifact still loads and names) and NOT in `STATE_POSITION_REFERENCES_AVAILABLE` (what a NEW run may select — the CLI's choices, and what `cli.common._refuse_unavailable_selection` checks so `--config-overrides` cannot get past it either). `control_command_hook="nominal-residual"` is the same pair

### D3 · control recipe · default `simple-v3`

= `simple-v2` + `control_imitation_loss_weight`; **its weight 64.0 does NOT transfer between airports — recalibrate per airport**. A named recipe is a published DETERMINISTIC arm: all seven `latent_*` fields are pinned at their defaults, so **a latent run is `custom`** (every latent arm file already says so; adopted 2026-09-07 after measuring that no stored artifact changes name, slug or loading)

### D4 · `control_dynamics_model` · default `point-mass`

`first-order-lag` buys smoothness + 3.4 % ADE; τ=2.0 s is defensible, not CV-selected. In `run_ts.py pipeline` the model is an axis of the cell: a lag cell's `train_dir` / `pred_dir` / category carry `_lag`, and the ONE override dict (`TrainingPlan._plan_overrides`) feeds the label, `--skip-train` and CV reuse — before 2026-09-09 two hand-written copies both lacked the field, so a lag cell was rebuilt as point-mass everywhere but the training command and shared its directory with the point-mass cell (review A-1)

### D5 · procedure penalty (state + control) · default weights at 0

NOT adopted — kept as an option. Its two hinge SCALES (100 m / 30 m) are `objective.PROCEDURE_{LATERAL,VERTICAL}_SCALE_M` module constants, not fields: units, never swept, retired 2026-09-07. The closure timing group's 60 s is `outputs.closure.model.CLOSURE_TIMING_SCALE_S` for the same reason. The four scales a named recipe PINS (`position_loss_scale_m`, `final_time_scale_s`, `control_velocity_loss_scale_mps`, `control_heading_rate_loss_scale_dps`) stay fields — a module constant there would silently redefine every published simple-v* comparison

### D6 · command hook · default off in training

**`predict --command-hook barrier --hook-saturation soft` is the ADOPTED use**; no arm trained THROUGH a hook beat its predict-time counterpart (six tried). THREE modules are live — `barrier` (lateral, gated ON the final), `speed-floor` (the stall margin on the thrust command, UNGATED — L3.d, 2026-09-08), `trombone` (the pre-final path stretch, gated OFF the final — L3.e, 2026-09-08) — and the vocabulary carries three combinations: `barrier+speed-floor`, `barrier+trombone`, `barrier+speed-floor+trombone`, each applied in the order it spells. The `+` is a LOOKUP in `config.CONTROL_HOOK_MEMBERS`, never a split: `speed-floor+barrier`, `barrier+trombone+speed-floor` and a SOLO `trombone` are not members and are refused with the vocabulary (the trombone hands the command back at the final approach course and has nothing to hand it to without the barrier)

### D7 · `--truncate-at-threshold` · default off

Predict-side, any output kind: cut every record where it FIRST crosses the threshold ON THE FINAL (`final_approach_geometry.threshold_crossing_index` — `d ≤ 0` and inside the on-final gate there; closest approach within that first run) and stamp `source.truncatedAtThreshold`. **Any flyability/geometry/CTA readout of a hooked, late-CTA arm needs it** — L3.d's floored rollouts arrive EARLY and fly on (endpoint \|xt\| p95 43–63 km, pooled ADE 840 → 3443 m at offset 0), and a report over the whole record is scoring that tail: on the approach proper the same arms read fully-flyable 1.35 % → 48.9 % at +60 s. `final_time_s` moves to the cut, which is the point — an early arrival stops being invisible and becomes the `final_time_error_s` it always was. A forecast that never crosses ON THE FINAL is left WHOLE and says `false`, and a vectored rollout that flies past abeam is exactly that case — until 2026-09-09 the rule read the PLANE alone and cut those on their downwind, 8.7 km out; one that crosses on its LAST row is whole with the flag TRUE (the flag means "ends at the threshold", and on a fixed-time STATE forecast the postprocessor's own closest-approach rule sets the same flag). Refused together with `--no-truncate`

### D8 · `control_speed_floor_margin` · default `1.10`

The speed floor's margin: `V_floor = margin × V_stall(n_commanded, mass, rho, Cl_max)`. The COEFFICIENT is the package's existing one — the optimizer's NLP velocity floor (`optimization/scenario_optimization._STALL_MARGIN`) and the control-anchor eligibility gate (`outputs.control.strategy.CONTROL_ANCHOR_STALL_MARGIN`) are both 1.10 — but **the SPEED it multiplies is not the same one, so do not quote the three as equal**: those two use the 1-g stall speed at SEA-LEVEL density (the optimizer's also capped at V_ref), the hook uses `V_stall(n_commanded)` at the LOCAL ISA density, uncapped, because it defends `flyability`'s criterion, which is evaluated at each sample's own altitude. At 8000 ft ρ/ρ₀ = 0.79, so the hook's floor is ~12.5 % higher — an effective margin near 1.24, ×√n in a turn — making the hook strictly the TIGHTEST of the three. Not one symbol on purpose: the other two are frozen policy constants (one is spelled into the stored `airborne-1.10-stall-margin-v1`) and this one is a per-run field. **Refused away from its default under a hook that contains neither `speed-floor` nor `trombone`** (`config.CONTROL_SPEED_FLOOR_MARGIN_READERS` — the trombone divides by the same `V_floor` to size its detour), and the barrier's two gains are refused the same way under a hook that has no barrier — before L3.d "a hook is on" and "the barrier is on" were the same condition. `predict --control-speed-floor-margin` overrides it; names a run `floor-margin=`

### D9 · `--project-final` · default off

deployment fallback: the post-hoc projection under the ONE gate, `on-final`. The FAF-distance gate and the `corridor_gate` field were DELETED 2026-09-09 (review §5): never set in any stored run (the field is a retired constant on load, 0 names moved), and FAF-gating wrecked vectored flights

### D10 · `target_conditioning` · default off

`channels` helps only the duration head; PatchTST refuses it

### D11 · `latent_dim` · default 0

the latent intent (L2); `latent_prior_components` / `latent_beta` / `latent_free_bits_nats` / `latent_posterior_init_std` and L2.f's two below mean nothing without it and are refused

### D12 · `latent_beta_warmup_epochs` / `latent_aux_duration_weight` · default 0 / 0.0

L2.f's two levers on the POSTERIOR MEAN, which is where the information died (see the contract above). The first ramps β linearly from 0 over N epochs (`effective_latent_beta`, the one place the schedule is written; the epoch's objective is `replace(config, latent_beta=…)` and the validation pass is scored under the SAME one); the second adds a train-only `Linear(latent_dim→1)` on the POSTERIOR SAMPLE predicting `truth_duration_s / final_time_scale_s` (component `latent_aux`, registered iff weighted, never called in `decode`, refused under `cta_conditioning=given`). Named `beta-warmup=` / `aux-T=`. Arms: `docs/experiments/l2f_mean_information_arms.json`

### D13 · `cta_conditioning` · default `off`

`given` = the CTA is the duration (L3); a delivery-form demonstration, never a prediction result. **`self-q` is a PREDICT-TIME label, not a trainable value** (`CTA_CONDITIONINGS_AVAILABLE` keeps it out of a new run): `predict --cta-from-quantiles` stamps it on the config it writes beside the records, so the directory that read no future cannot be named like the one that did

### D14 · `duration_head` · default `point`

`quantile` = B1's five `DURATION_QUANTILES` of the SAME duration (`QuantileFinalTimeHead`, monotone by cumulative softplus; the median IS `final_time_s`). The loss swaps the squared final-time residual for the sum of five pinball losses **under the same component name**, so `loss_component_names` is unchanged. `two-head` = B1.b: BOTH heads, the POINT head driving the rollout (`final_time_head`, the same module and state-dict keys `point` trains) and `duration_quantile_head` emitting the interval. **The second head is built LAST** so a `two-head` arm and a `point` arm start parameter-for-parameter identical (measured: 34 shared keys, 0 differ) — `_initialize_duration_head` zeroes only the last layer, so building it earlier would shift the point head's hidden draw and unpair the arms gate 1 compares single-seed. The training dropout stream still cannot be matched — its pinball rides in a component of its OWN, `duration_quantile`, so `loss_component_names` gains one entry there and only there. Every value but `point` is refused off the control output and refused with `latent_dim > 0` — z reaches the duration by shifting the point head's ONE logit, and under a posterior sample the quantiles would be conditioned on the flight's own future. The named recipes pin it at `point`, so a quantile or two-head run is `custom` and wears `T=q5` / `T=2h`

### D15 · `duration_quantile_loss_weight` · default `1.0`

B1.b: what weighs the pinball sum. Under `quantile` it multiplies the `final_time` component the pinball replaced (the 1.0 that multiplied it before the field existed, so nothing moved); under `two-head` it weighs the separate `duration_quantile` component. **Refused non-default under `point`** — there is no such head — and symmetrically `final_time_loss_weight` is **refused non-default under `quantile`**, where the pinball replaced the point term outright. Named in a run only when it deviates (`pinball=3`)

### D16 · `n_segments` (control) · default 64

**32 is free** (L1: 1322 vs 1333 m, bank skill 0.726 vs 0.728); the deployed head's 257 numbers become 96

### D17 · `control_state_loss_grid` · default native

`fixed-dt` without the imitation term (it is not registered there) trips the straight-in veto (FDE 703 → 2863) and brings the bank wiggle back — the trajectory-error loss alone is not enough. FROZEN 2026-09-09 (`CONTROL_STATE_LOSS_GRIDS_AVAILABLE`): its 26 stored runs load, no new run selects it

### D18 · `control_heading_rate_loss_weight` (+ `_scale_dps` 1.5) · default 0

L1.b arm ②: the teacherless way to name the bank — the rollout's own ψ̇ at the segment endpoints against the flown track's. `_scale_dps` is the UNIT the residual is read in (half a standard-rate turn), not the dose. Doses 1.0 / 8.0 bracket an unknown; **not measured yet**

### D19 · `control_bank_tv_loss_weight` · default 0

L1.b arm ③: mean \|step\| between adjacent COMMANDED banks, in half-box units. **It prices REVERSALS, not slope** — `sign(x)` cancels on a monotone run's interior — and exact flatness is a STATIONARY POINT (value 0 and gradient 0), which is where the zeroed head init starts every run, so it never leaves a flat schedule on its own. "TV changed nothing" is that before it is a dose. **Not measured yet**. Both terms register under `true-time-position` ONLY (where `velocity`/`imitation` live) and `TSConfig` REFUSES a non-zero weight under any other objective rather than ignoring it

### D20 · `checkpoint_selection_metric` · default `fixed-anchor-common-grid-ade`

WHICH epoch's weights the checkpoint keeps — see the table below. Lower is better for all three; the value NAMES the run (`select=…`, a `META_FIELDS` entry)

### D21 · `lr_plateau_metric` · default `selection`

WHICH number `ReduceLROnPlateau` measures its plateau on — never which epoch is KEPT. `objective` steps it with the macro validation objective (the SAME `val_loss` the epoch record writes, not a third number), `selection` with the checkpoint-selection value. It matters exactly when the two PART, which a fixed-anchor arm never does: on A0-random's `_grid` arm the objective improved to epoch 60 (1.147 → 0.707) while the grid ADE stalled after epoch 8, so the scheduler halved the LR from epoch 20 and reached **9.4e-7 by 60** — the model stopped training at the epoch the READOUT stalled. `checkpoint_metadata.json`'s `lr_scheduler.metric` says which it stepped on; names the run `lr-metric=objective`. **`objective` is REFUSED while the objective itself moves under the model's feet**: a `latent_beta_warmup_epochs` ramp reweights it every epoch and a `procedure_loss_dual_step` reprices the same trajectory every epoch, and under either the plateau scheduler would cut the rate straight through a schedule that is still ramping. A0.b (i)

### D22 · `random_train_anchor_sampling` · default `uniform`

HOW a random train anchor is drawn — refused without `random_train_anchor`. It cannot change WHICH anchors are admissible (`eligible_random_train_anchors` + the future contract), so both policies train the identical cohort on the identical anchor population. `uniform` draws over the flight's admissible SAMPLES, i.e. uniformly in TIME; **pooled over flights that over-weights the near end relative to the stored population**, because every flight gets one draw whatever its length and a kilometre near the runway holds more samples than one at 25 km (whole KRDU val split, 1404 flights / 181,906 anchors / 200 epochs: draws **34.3 %** under 6 km against **25.1 %** of the population, **17.9 %** beyond 20 km against **32.9 %**). `remaining-path-uniform` places the draw uniformly across the flight's OWN admissible remaining-path span and takes the nearest anchor — equal weight per km — moving those to 31.0 % and 21.0 %. **A stratum draw was tried first and REJECTED**: it moved training toward the runway (≥ 20 km 16.9 → 4.1 %) because the grid cuts the near end into four 2-km strata and leaves one open stratum spanning 20–123 km. Every epoch records the drawn counts per stratum BESIDE the population they came from (`history.json`'s `train_anchor_sampling.remaining_path_strata{,_population}`), under both policies — neither number means anything alone. Names the run `anchors=remaining-path-uniform`; `sampling_version` distinguishes the two in `training_anchor_contract`. A0.b (ii)

### D23 · `random_train_anchor_l1_share` · default `0`

How much of that draw is RESERVED for **L-1**, the anchor the fixed-anchor arms train at (A2b, 2026-09-08). With probability `l1_share` a flight's draw for the epoch IS `default_anchor(config)`; otherwise the unchanged remaining-path-uniform draw. **A mixture, not a reweighting**: the coin is a SECOND 8 bytes of the same per-flight per-epoch sha256, and the digest's salt (`sampling_version`) does NOT move with the share — only `reported_sampling_version` does — so the draws the coin passes over are literally the anchors the share-0 arm drew, and an arm differs from its base arm only in the replaced draws. WHY: A0.b's `A0b_lr_objective_path_uniform` wins at every re-anchored bin and still loses at L-1 (**1782 m** pooled ADE against fixed-anchor native32's **1322**), because under a law spread over the whole approach L-1 is one point among many. **Refused (never ignored)** under `uniform` sampling, with `random_train_anchor=False`, and outside [0, 1]. **The reserved draw is the flight's FIRST admissible anchor**, which is L-1 unless output eligibility removed it (`airborne-1.10-stall-margin-v1`: the observed state there is outside the airborne model domain, so there is no valid L-1 window to train) or a `minimum_anchor_index` floor moved it (where it is exactly what `FixedAnchorTrajectoryWindows` anchors at anyway). **That is NOT rare enough to assume away and NOT a refusal**: measured on the real KRDU roster, **10 of the 9,720 eligible arrivals** store no L-1 (0.10 %, nine of them runway 23R, first admissible anchor 90–163 against L-1 = 59) — a refusal would have aborted both A2b arms, and changing the roster to avoid it would have destroyed the pairing. It is COUNTED instead: `l1_share_flights_without_l1` in every epoch record, and one line from `train()`. Every epoch records `train_anchor_sampling.l1_share_drawn` beside the strata histogram — the coin's REALISED share, which `fixed_anchor_fraction` bounds from above (a span draw can land on L-1 too). Names the run `l1-share=0.3` when non-zero; a non-zero share reports `sampling_version=per-flight-hash-v4-remaining-path-uniform-l1-share`. Default 0 draws BYTE-IDENTICALLY to before the axis (digests pinned in `tests/test_random_anchor_sampling.py`). A2b

### D24 · `control_imitation_target` · default `inverse-dynamics`

WHAT the imitation term imitates. The default's schedule, flown open-loop, lands **2.5–7.8 km** from the truth it was read off (L0), so "imitating it perfectly" is not "flying the truth". `fitted` reads the per-flight table `run_ts.py control_basis_oracle --checkpoint` fits through the same rollout (88–433 m) and needs `control_fitted_teacher_path` (which names the run: `teacher=<dir>/<file>`); refused with `random_train_anchor`, with `control_imitation_loss_weight=0`, and off `control_state_objective=true-time-position` (the only objective the term is registered under). L5.a, **not yet measured** — the arms are `docs/experiments/l5_fitted_teacher_arms.json`

### D25 · `control_thrust_parameterization` · default `thrust-fraction`

`specific-force` = the head predicts `n_x = (T − D)/W` (the contract above). **Measured (N3, design §7.3, both seeds against same-code twins): NOT adopted.** It collapses the per-class n_x bias to the truth's own spread (0.0042 / 0.0044 vs 0.0125 / 0.0118 g), but the heavy − 737 speed gap grows and straight-in FDE p50 is +240 / +222 m. **That is NOT the missing drag feedback** (design §7.4): straight-in approaches fly below the minimum-drag speed, where δ's drag feedback amplifies errors. The error is the last 5 km's height/speed SPLIT: 36–54 m high and 4.4–6.4 m/s slow, with the total energy right. **Diagnosed (design §7.5):** the split is the vertical-load command integrated open loop, and δ's straight-in FDE edge is a stall-bound dive that ends ~100 m low and on time. An inference-time glidepath hook (N7) was WITHDRAWN as a fix: it computes the answer from the procedure. N7′ (a learned per-segment path-angle target flown by a fixed tracking law) is proposed, not built. First-order-lag only; refused with the fitted teacher. The speed floor inverts it drag-free and saturates at the engine's `(T_max − D)/W` — NOT at the head's 0.23 g box, which sits below the engine on this fleet (a box-capped floor had less authority than the thrust-fraction one). Names a run `first-order-lag+specific-force …` / slug `lag-sf-…`, only off the default. `control_recipe()` carries it only off the default, because `pipeline` compares that dict for checkpoint reuse

### D26 · `control_thrust_parameterization` = `speed-command` (no default of its own)

N6 (design §12, branch `sf-n6`): **ran 2026-09-15 and FAILED — unstable in training** (§12.9: the loop hides a pull-up's energy cost until the T_max clamp binds; zoom climbs, pre-clip gradients 1e7–1e10, pooled ADE 2335 vs 1325 m). Do not train it again as is. Its premise was also wrong (§7.4). `sf_n6_arms.json` is launched only if N3's straight-in FDE veto holds against the same-code twin (§12.6). Named `first-order-lag+speed-command` / slug `lag-sc-…`; the record carries `speed_command_delta` per segment

### D27 · `control_thrust_parameterization` = `specific-force+path-angle` (no default of its own)

N7′ (design §14, branch `sf-n7`): the one value that also moves the VERTICAL column. The head's third column is a path-angle target γ\* (box [−15°, +10°], neutral −2.9° = the teacher's own median, τ_γ = 3 s — all CONTRACT constants, spelled into the target contract by `ControlContract.identity_suffix`), and the lag RHS re-solves the load factor `n = [cos γ + V(γ\* − γ)/(g·τ_γ)]/cos φ` at every RK4 stage, clipped to the load box with the stall clamp unchanged; longitudinally it stays the specific force, so `Ė = V·n_x` and the vertical law is energy-neutral. **Why:** the load-factor column is an OPEN-LOOP double integrator — a bias δn grows a height error like `½·g·δn·t²`, measured at +92 m and −5.3 m/s at the end for δn = 0.004 against +17 m for the same instantaneous error here (design §7.5.3, §13.3). The target is ABSOLUTE, not anchor-relative: the anchor's own γ carries ADS-B noise the size of the whole error budget. **Refused with** the point-mass model and the `fitted` teacher. The command hooks are ADMITTED since 2026-09-16 (they read the load through the law, two-tier design §10.8), and so is the heading-rate loss: it reads the load the path loop resolved (`law.geodetic_load`), so γ\* reaches the turn rate through the lift. The record's `load_factor` is the load the loop resolves at each segment's START state and `path_angle_command` rides beside it. **MEASURED (N7′, design §15, both seeds against the same-code δ twins): it passes every pre-registered gate.** γ\* bias 0.088 / 0.091° against the pre-registered break-even 0.11°; the stall-bound share on the last 5–1 km falls from 33/45 % (n_x) and 64/68 % (δ) to **1 %**; the last-km height error from +39/+54 m to **+6/+3 m**; straight-in FDE p50 **543 / 559** against the twin's 647 / 663 (a win on BOTH components of the along/vertical split); pooled ADE **1173 / 1182** against 1325 / 1295, better on 70/69 % of flights paired; **fully flyable 97.7 / 97.4 % against the twin's 0.4 / 0.2 %** — a closed vertical loop keeps the rollout in the envelope. Both N3 gates pass as well, the heavy − 737 speed gap included (+3.98/+3.76 against the twin's +5.16/+4.89, where N3 alone widened it to +6.17/+6.31). ONE regression: the vectored endpoint, FDE p50 +625/+704 m, which decomposes as cross-track (vectored ADE is 242/147 m better) and is unexplained — turn authority is measured NOT to be the cause. Adoption is the user's call. Names a run `first-order-lag+specific-force+path-angle` / slug `lag-sfpa-…`

### D28 · `control_condition_features` · default `raw`

HOW the airframe is written into the control head's 8-channel condition vector (`outputs/conditioning.py`, design §11). `ratios` keeps the mass and the polar and replaces the installed thrust and the wing area by `T_max/(m g)` and the 1-g stall speed (`aircraft.aero_params.stall_speed_ms`) — the same information and the SAME WIDTH, so a ratios arm starts from its raw twin's weights. On KRDU's 26 types the raw thrust channel spans 46×, T_max/W 1.3×; Cd0, k and the stall parameters are constant on the whole fleet (4 of the 8 channels carry nothing). **Built, not yet measured**: N4 (`sf_n4_arms.json`) is the alternative-hypothesis control for N3, and it re-trains the δ twins because the stored `B1_point_matched` pair does not reproduce at the current code (design §11.6). Every named recipe pins `raw`; names a run `airframe=ratios`; `control_recipe()` carries it only off the default; `predict` takes the checkpoint's own value and has no override (a head trained on one set reads the other without a shape error)

### D29 · `control_horizon_s` · default `0` (the whole approach)

**Two-tier L1 (2026-09-17, `docs/2026-09-17_two_tier_plan_v2.zh.md` §3): a FIXED rollout horizon Δ.** The schedule is rolled over exactly Δ, the targets cover [0, Δ] at the N segment ends, and no duration head is built. **ONE definition of a sample's span, `dataset.target_horizon_s`** (the truth's remainder, or Δ) — `_sample_arrays`, the control supervision endpoints and the selection metric's `common_truth_at_anchors` all read it; the per-epoch report block reads the truth CUT at Δ (`series_within_horizon`), or a Δ-long prediction is scored against touchdown. **ONE floor rule, `dataset.effective_min_future_s`**: every window set, the anchor grid's bins and the anytime curve admit only anchors with ≥ Δ of truth after them (a flight with less is excluded and the notice names Δ; the epoch audit reports the raised floor). `ControlOutput` REFUSES every axis that would decide a duration under Δ — the CTA, a quantile head, a non-zero `final_time_loss_weight` (the residual is structurally zero; the component stays as a stated 0 like `kinematic`), the latent, the imitation teacher — and `intent=truth-join-duration`. Named `ctrl-horizon=`; the recipes pin 0, so a Δ run is `custom`. **The record's `ade_m`/`fde_m` are still export's whole-remainder accounting** (the prediction held at its last node to `true_final_time_s`); an L1 arm's numbers are its readouts' ([0, Δ]), never the summary's

### D30 · `plan_conditioning` = `manoeuvre-code` (no default of its own; the v2 tokens are archived)

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit, the 2026-09-20 rewrite of stage B to an instruction vocabulary). The text below describes the archived code and is kept as its record. The manoeuvre-token plan's executor conditioning (`docs/2026-09-18_manoeuvre_token_plan.zh.md` §2.6, P1.3, 2026-09-18). The token is the segment's code vector **z** from the tokenizer (`manoeuvre/tokenizer.py`), which the executor holds as a SUBMODULE (`heads.ControlFeatureModel.manoeuvre_tokenizer`): under `manoeuvre_codebook = ""` it trains JOINTLY — the control objective's gradient reaches the encoder through z (FSQ straight-through) and the tokenizer is exported afterwards (`run_ts.py manoeuvre_codebook --checkpoint … --out 4dTrajectory/outputs/codebooks/<name>`); under a codebook path it is the frozen codebook's (P3.3) and the checkpoint binds to `codebook_sha256` (C28). Fields: `manoeuvre_tokenizer` ∈ {`learned` (default), `command-vocabulary`}, `manoeuvre_fsq_levels` (one integer ≥ 3 per FSQ dimension, K = the product; empty for the command vocabulary), `manoeuvre_codebook`; all three refused off their defaults under any other plan value, and `manoeuvre-code` REQUIRES `control_horizon_s` > 0 — **the segment IS the fixed horizon**, there is no second `segment_s` field. The dynamics context carries the tokenizer's INPUTS, one source per batch: `manoeuvre_segment` [B, R, 6] (the TRUTH's segment from the anchor in its start frame, `manoeuvre/segments.py`) + `manoeuvre_state` [B, 6] (the anchor's chart row) in training, `predict` and protocol C (reads the future); or `manoeuvre_z` [B, Z] handed in (the prior's, protocol A). Both present or neither → refused (`plan_token.plan_z`). `plan_conditioning_dropout` is RETIRED at its constant 0 (`RETIRED_CONSTANT_FIELDS`): 0.5 taught the head to ignore the token (v2 §10.8). Names: `plan=manoeuvre-code, fsq=4/4, ctrl-horizon=60` (`tok=command-vocabulary`, `codebook=<parent/name>` off their defaults). Records carry `manoeuvreCode` + `manoeuvreCodeSource` (`truth` | `given`); the epoch record's `control_training_diagnostics` gains `manoeuvre_codes` (K, used, unused, max share, entropy, counts — gate T(iii)'s reading) and a `manoeuvre_tokenizer` gradient group (a stated 0 on every other run). The archived v2 tokens (`truth-next`, `waypoints`, K waypoints every 30 s relative to the current position; `archive/two_tier_v2_2026_09/plan_token_v2.py`) are refused at load by name; their readouts (`short_horizon_readout`, `tracker_lockstep`, `two_tier_gates`) are archived with them and the numbers live in v2 §10–§12

### D31 · `manoeuvre_token_s` / `manoeuvre_token_step_s` · defaults `0` / `0` (= the horizon)

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit, the 2026-09-20 rewrite of stage B to an instruction vocabulary). The text below describes the archived code and is kept as its record. Two-tier v3 stage B (`docs/2026-09-18_two_tier_plan_v3.zh.md` §5.2, plan decisions D31 / D38 / D37; 2026-09-19). `manoeuvre_token_s` is the token SPAN S — the seconds of the approach one manoeuvre code stands for: the tokenizer reads S seconds of truth from the span's start (`plan_token.training_plan_context`), the codebook records S as its `segment_s` (`manoeuvre_codebook`), the prior's sequences are S apart. `manoeuvre_token_step_s` is the STEP the closed loop flies between two predictions inside a span under a coded protocol (`lockstep.round_step_s`; the executor still forecasts its whole horizon), and the stride training draws the anchor's position inside the span on. Both 0 = the horizon, which every 2026-09-18 arm had (one token per forecast, `token_hold` 1 — those checkpoints load and fly bit-identically; an explicit value EQUAL to the horizon is refused so one behaviour has one identity). Rules (`TSConfig.__post_init__`, one place): S ≥ the horizon, the step ≤ the horizon, both whole multiples of `dt_s`, an explicit step a whole multiple of the integrator step (the leg is cut on the dense rollout grid), S a whole multiple of the step; `token_span_s` / `token_step_s` / `token_hold` resolve them. Names `tok-s=` / `tok-step=`; `control_recipe` carries `token_s` / `token_step_s` off default only. The three stage B configurations: S20 (nothing set), S60-held (`manoeuvre_token_s` 60 on a 20 s horizon: one token held over three 20 s rounds), S60-h60 (a 60 s horizon with `manoeuvre_token_step_s` 20: a 60 s forecast flown 20 s at a time, three rounds per token). **Training under a held token** (B-dev2): a training row draws φ ∈ {0, …, hold − 1} per flight, anchor and epoch (`plan_token.token_phase`, sha256 under `TOKEN_PHASE_SAMPLING_VERSION`; the epoch seed reaches `WindowContext.row(i, epoch_seed)` from `TrajectoryWindows.batch`; cached / validation rows are φ = 0), reads the span that started φ steps BEFORE the anchor and that span's start row, and — when the record cannot hold that span — the PREVIOUS span's (`token_span_start_s`), which is the closed loop's own rule past the truth's last full span (protocol C holds its last code). The random-anchor floor therefore stays the horizon's Δ, not S — a deliberate deviation from the plan's B-dev1 sentence, taken so the S60-held executor trains on the same anchor population as its baseline L60_D20 and IS trained in the last minute of the approach where "established" is decided; the user's decision of 2026-09-19 (plan D48). **The closed loop** (B-dev3): round k reads token k // hold at phase k % hold; protocol C's held code past the truth's last full span is counted in `held_predictions`; A / A-truth ask the prior at a refresh (phase 0) only, A-truth is exhausted at a refresh when the truth prefix runs out; a flown span is tokenised back once its last round is flown; every round's record carries `token_index` / `phase`, the row `token_refreshes`. **The prior's landing** (D37, B-dev5): `prior_landed_at_s` records when the prior first says the flight lands (the refresh's flown seconds + its fraction of the span) and `prior_landed_error_s` its error against the truth's arrival; the flight ends by it only under `manoeuvre_lockstep --prior-landing-ends-flight` (the 09-18 rule, which judged 732 / 1392 flights by the prior's clock).

### S1 · the three checkpoint-selection metrics

**The three checkpoint-selection metrics** (`config.CHECKPOINT_SELECTION_METRICS`, dispatched
through `validation.VALIDATION_SELECTIONS`; the LR scheduler, early stopping and the kept
weights all read the one number):

| value | what it scores | use it when |
|---|---|---|
| `fixed-anchor-objective` | the validation objective itself | the loop never draws the path it would be judged on — **the closure output, which `TSConfig` requires it for**. **Refused for a latent run**: that objective decodes a posterior sample, so it reads the future and is stochastic |
| `fixed-anchor-common-grid-ade` | airport-macro common-grid ADE at the **L−1 anchor** | the default, and right for every fixed-anchor arm — L−1 is where they train and where they are judged |
| `anchor-grid-common-grid-ade` | the same ADE at **every anchor set the cohort supports**, equal weight per SET: L−1 plus whichever of `anchor_grid`'s 16 / 12 / 8 / 6 km CANDIDATE bins clears the coverage gate, each flight at its own closest admissible sample (full lookback, ≥ 60 s of truth after it) | **a random-anchor arm**. The fixed metric scores the ONE anchor such a model is least specialised for and froze `A0_random_hr8_tv1` at epoch 10 (L−1 ADE 2949 vs native32's 1322) while that checkpoint drew BETTER geometry than the fixed arm at every anchor (chamfer p50 −114…−524 m). Each set's ADE is the mean over the flights that HAVE that anchor — absent, never scored 0 — then the airport macro; the per-set values, counts and `dropped_bins` land in `history.json`'s `validation_anchor_grid` block every epoch, `fixed_anchor_common_grid_ade_m` among them so the L−1 veto stays readable. Costs ≈ 4–5× the selection stage (one extra deployable replay per surviving bin; the L−1 pass is reused), ≈ +12–15 % per epoch at KRDU scale. **Refused with `cta_conditioning=given` or `intent_conditioning=truth-…`** — the grid re-reads the oracle at every bin anchor, so the metric would be selecting on how fast it converges |

### G1 · `data/anchor_grid.py` is the one definition of the grid

**`data/anchor_grid.py` is the one definition of the grid** — bins (`DEFAULT_ANCHOR_GRID_KM`
20/16/12/8/6/4/2 km, `VALIDATION_ANCHOR_GRID_KM` the four the metric may select on), the
60 s future floor, `PARTIAL_COVERAGE` = 0.5, the per-flight `bin_anchor` rule and the
fixed-at-the-evaluation-anchor `strata_fixed_at_anchor` rule. `run_ts.py anytime_curve` and the selection metric
import the SAME objects (`tests/test_anchor_grid.py` asserts identity, not equality): two
grids that merely agreed today would make "the curve improved" and "this epoch was selected
on the curve" claims about different anchors.

### G2 · the grid's values live in the leaf `data/anchor_strata.py`

**The grid's VALUES live one level down, in the leaf `data/anchor_strata.py`**, and `anchor_grid`
re-exports them — because `anchor_grid` imports `dataset` while `dataset` needs the same
kilometres for the training-anchor draw, so a single home would mean a cycle or a second
copy. The leaf owns `DEFAULT_ANCHOR_GRID_KM`, the strata those values cut when read as EDGES
(`REMAINING_PATH_STRATA_EDGES_M`: seven edges, **eight** strata, the open ends `<2km` and
`>=20km`), `remaining_path_strata` (`np.digitize` on the same profile the bins are chosen
from) and the draw law `remaining_path_uniform_offset`. It must stay free of `dataset` and
torch — `tests/test_import_boundaries.py` pins that, and it is the whole reason `dataset`
can import it at module scope.

### G3 · a candidate bin is a candidate, not a guarantee

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

### H1 · command hooks: once per segment, the schedule FLOWN is recorded

Command hooks are called once per control SEGMENT, at its start, with the rollout's own state,
returning the command flown — and **the record carries the schedule FLOWN, not the network's**.
Two rules that cost real trajectories when missed: a hook that changes the bank **must
re-coordinate the load factor**, and every rate gain must be `min(gain, 1/Δt)` because the
command is HELD.

### H2 · a hook acts through the controls — the speed floor (L3.d)

**A hook acts THROUGH the controls, never on the state.** The speed floor (L3.d) is the second
module and the rule is what makes it one: it raises the commanded THRUST so the speed is still
on the floor when the hold ends — `V̇ = (T − D)/m − g sin γ` inverted for the command, with the
thrust already spooled credited over `τ_eff = τ_T(1 − e^{−Δt/τ_T})` exactly as the barrier
credits a bank already rolled into. Clipping `V` after the rollout would have been three lines
and would have stopped the record being a trajectory of the dynamics. Three consequences worth
knowing before touching it: it is **UNGATED** (the barrier's on-final gate is a statement about
the final approach; a stall is a statement about the airframe, and 77–94 % of the measured stall
samples sit at ≥ 20 km remaining, where that gate is shut); it **owns one channel** and returns
bank and load factor bit-identical, which is what makes `barrier+speed-floor` well defined; and
the floor reads the **COMMANDED** load factor, so under the composite it prices the manoeuvre the
barrier just coordinated. The floor is imposed at the END of the hold and drag is frozen across
it, so the realised speed can dip slightly below the floor mid-hold (measured 0.2–0.7 m/s on the
synthetic fixtures, at 5–6 s holds) — that dip is against a 10 % margin, not against the stall
speed, so it does not reach `Cl_max`.

### H3 · the trombone is geometry, not thrust (L3.e)

**The trombone is the third module, and it is GEOMETRY, not thrust (L3.e).** With the CTA
fixing the total time and the path fixed, the mean speed is fixed — so a speed floor alone has
nowhere to put a late arrival, which is why L3.d traded stall for thrust-over-max and an early
threshold crossing. The trombone adds the missing degree of freedom. Per segment it reads the
schedule's own remaining time (`RolloutStateView.remaining_s`, new: the reversed cumsum of the
segment durations, which under a CTA-conditioned decoder IS `T_cta − t`), the beeline distance
`D` to the threshold, and the SAME `V_floor` the floor module holds
(`speed_floor.floor_speed`, one definition), and spends the surplus `ΔL = V_e·T_r − D` as a
dog-leg about the beeline at `cos θ = D/(V_e·T_r)` — the offset at which flying for the
remaining time lands exactly on the threshold. **`V_e = max(V_floor, V_h)`, the speed being
flown floored at the speed floor, and that `max` is what makes the manoeuvre terminate**: the
floor is a LOWER bound (the module only raises thrust), so `V_h > V_floor` is the ordinary
case, and sizing against `V_floor` alone leaves `ΔL` not falling — measured, the surplus rose
from 2981 m and plateaued at commanded thrust 0.12, the half-way switch never fired and the
excursion pinned outbound until the threshold plane ended it, 2.9 km wide. With `V_e`,
`dΔL/dt ≤ 0` for any speed flown. Five things to know before touching it:

### H3.1 · trombone — the hand-over rule

- **The hand-over rule.** It may only OPEN an excursion where the predicted path is more than
  `ALIGNMENT_MAX_DEG` off the runway course — strictly inside "the on-final gate is closed",
  so it can never act on the final and never fights the barrier over a state already lined up.
  Alignment gates the OPENING only: its own outbound leg can swing the aircraft through the
  course, and a hook that fell silent there would leave it 40° off with no leg back (measured:
  four engaged steps, then a parallel track 6.6 km wide of the threshold). Once the gate has
  opened for a flight it is disabled for the rest of the rollout, and it never acts past `d = 0`.

### H3.2 · trombone — the price of that rule (~58 % of the fleet)

- **The price of that rule is ~58 % of the fleet** (KRDU, 2998 arrivals sampled, observed track
  at the L−1 anchor: 58.9 % aligned within 30° at the anchor, 58.4 % aligned at EVERY row from
  it, 57.1 % already inside the gate). Those flights cannot be stretched at all; their
  unabsorbed delay is published as `commandHookDiagnostics.tromboneDelayS` next to
  `tromboneEngagedSteps = 0`, which is the number to quote, not a reason to loosen the rule.

### H3.3 · trombone — the 15° turn cap is set by the speed floor

- **The 15° turn cap is set by the speed floor, not by comfort.** The value's `+` order is its
  application order, so under `barrier+speed-floor+trombone` the floor prices the network's load
  factor and NOT the turn the trombone then adds; the turn costs the margin twice (stall speed
  ×√(1/cos μ), plus unpriced induced drag ~tan²μ). Measured on the rollout fixture: stall slack
  −0.6 m/s at 10°, −0.9 at 15, −1.5 at 20, −3.3 at 25, and a first stall sample at 30°. The
  offset cap (45°), not the bank cap, is what buys path, so lowering the bank costs nothing.

### H3.4 · trombone — `hook_saturation` softens the bound, not the mode

- **`hook_saturation` softens the bound, not the mode.** The hand-over, the side and the
  half-way switch are hard under both; `soft` softens the bank saturation, the gate blend, and
  a ramp on the surplus that is exactly zero (with zero slope) at zero surplus — so a flight
  with nothing to absorb comes back bit-identical in BOTH forms.

### H3.5 · trombone — geometry: within one rollout it barely touches the thrust

- **It is GEOMETRY: within one rollout it barely touches the thrust.** The speed comes from
  the thrust commands, the schedule is open loop, and the floor's demand is a function of speed
  and height — measured on the fixture the floor's bound/saturated step counts are IDENTICAL
  with and without it (47/48 and 0/48). What changes is WHERE the aircraft is when the schedule
  runs out, and (through `--truncate-at-threshold`) which part of the trajectory a report scores.
  **If `thrust_over_max` falls in the L3.e arms, that is why** — not the hook relieving thrust.

### H3.6 · trombone — the 45° offset cap binds past ~41 % delay

- **The 45° offset cap BINDS once the delay exceeds ~41 % of the time the remaining path
  needs** (`sec 45° = 1.41`), and `hook_trombone_saturated_steps` is how you see it — the
  counterpart of the floor's "full thrust is not enough". Outside the arms' range (a 25 km run
  at ~70 m/s is 350 s, so +90 s asks for 26 % and lands near 31°), but a short remaining path
  plus a large delay is simply not absorbable before the final, and the count is what says so
  rather than a silent shortfall. `hook_trombone_bank_capped_steps` does the same for the 15°
  turn cap, so the argument the cap rests on is auditable on an arm.

### H3.7 · trombone — under `barrier+trombone` the floor is an assumption

- **Under `barrier+trombone` the floor is an ASSUMPTION, not an enforced speed.** That stack is
  the ablation that isolates the stretch; nothing there holds the speed up, so read it as "the
  stretch alone", never as "the stretch under the floor it was sized against".

### H3.8 · trombone — the surplus reference is an axis (`beeline` default)

- **What the surplus is measured against is an AXIS, and the default is the one L3.e failed on
  (`trombone_surplus_reference`, L3.f, 2026-09-09).** `beeline` is the `D` above, and it reads a
  vectored flight's own downwind and base as surplus time: at the TRUE CTA, `tromboneDelayS`
  p50 391 s, endpoint `|xt|` p95 10 km, 46 % not reaching the threshold by `T_cta` — while the
  mechanism itself was clean (30 s absorbed per 30 s of offset, thrust/stall/load −85/−84/−99 %).
  `reference-rollout` (`predict --trombone-surplus reference-rollout`) adds to `D` the DETOUR
  the HOOK-FREE reference rollout still intends: `detour = L_ref − S_ref`, its remaining path
  to its first threshold crossing ON THE FINAL (the shared
  `final_approach_geometry.threshold_crossing_index`; the plane alone is crossed abeam, and
  reading it that way measured `L_ref` to a downwind — see the crossing contract above) less
  the straight line to that same cut. So
  `ΔL = V_e·T_r − D − detour`, positive part, then the same dog-leg `cos θ = D/(D + ΔL)`.
  **Sizing against `L_ref` itself — the tempting reading — breaks it**, and that is the trap
  to know: `L_ref` is indexed by the SCHEDULE and cannot see the excursion the hook is
  flying, so the surplus never burns down, and past the reference's own crossing it
  degenerates to the whole remaining reach. Measured on the 48-segment fixture: a clean 60 s
  absorption became 19 steps pinned at the 45° offset cap ending 1.7 km wide (beeline: 0
  saturated steps, 159 m), and a reference that decelerates and stops SHORT had its missing
  metres read as surplus (9.4 s of delay invented on a flight with nothing to absorb). The
  detour form has neither problem: `D` is the AIRCRAFT's own, and `detour` is non-negative
  and non-increasing by the triangle inequality, so `dΔL/dt ≤ 0` survives as the identity it
  is under `beeline`.
  **The default stays `beeline` so every L3.e artifact reproduces to the bit** — records
  included, which is why `tromboneRefPathM`, `tromboneRefNoCrossing` and the
  `tromboneSurplusReference` label are ABSENT rather than zero under it (measured 2026-09-09:
  1972 harness leaves over both stacks, trained and replayed, soft and hard — 0 differ; 333
  stored runs recounted, 0 renamed; re-measured the same day over 2344 leaves for the crossing
  rule — 24 differ, every one a reference-only diagnostic under `reference-rollout`). Under the
  axis the composite gains the hook-free rollout
  (`needs_reference` — one extra integration of the schedule, once, ~2× the SEGMENTED
  rollout; the per-step cost is an index into a table built at segment 0), and both lengths
  are CHORD sums between segment boundaries (short by `sinc(Δψ/2)`: 0.15 % at 15° over a 5 s
  hold, 2.8 % at 45°, and the two errors partly cancel in the difference). Refused away from
  its default under a hook with no `trombone` in it, like the barrier's gains and the stall
  margin.

### H4 · every hook reports per-flight counts

**Every hook reports per-FLIGHT counts, and that is what a record carries.**
`per_flight_diagnostics()` returns `[B]` rows; `diagnostics()` is those summed and is what an
epoch record reports. `source.commandHookDiagnostics` on a prediction record is the flight's own
`steps` plus every other count as a share of it — `commandHook` alone cannot separate a flight
the hook never touched from one it rewrote at every step. The key is ABSENT without a hook,
never zero.

### H5 · the altitude word's 1000 ft bin is indistinguishable from random rounding (2026-09-21)

Measured after the user challenged the bin ("不会太粗吗?"). Every number below is off the five
airports' full arrival manifests — **42,604 flights, 75,534 altitude instructions** (the labeller
refused 1 track, a KRDU record holding a stopped row; counted, not dropped). Wider than stage B's
8,257-flight cohort, which is stated because the cohort's own numbers differ.

- **D51's stated justification does not hold.** It reads "目标离档中心 p95 都在半档以内 (128 m)".
  `altitude_bin` rounds to nearest, so the distance from any value to its assigned centre is ≤ half
  a bin **by construction**. The check cannot fail; it measured nothing. (Repo rule: a bound that
  can never bind is worse than no bound.)
- **56 % of "altitude instructions" are not instructions.** The labeller reads the threshold
  crossing as a plateau and emits an altitude target for it: on the KRDU cohort, **8,212 of 8,257
  flights have exactly one** such event below 200 ft, 45 have none, and **none has two** — so it is
  structural, not an averaging artefact. Fleet-wide it is 42,621 events with p25/p50/p75 =
  **42 / 62 / 94 ft** above the threshold. That is the landing, which the terminal word (D72)
  already states.
- **On the real levels that remain (33,124 events), the current bin is random rounding**:
  mean 235.3 / p50 240.9 / p95 472.0 ft, against a uniform-random reference of 250 / 250 / 475.
  500 ft → 112.4, 250 ft → 62.4, **200 ft → 50.9 (51 words)**, 100 ft → 26.3 (101 words). The real
  levels sit at 1600–3700 ft above the threshold with only 24.7 % on a round 500 ft and 15.3 % on
  a round 1000 ft, because the bins are anchored at the threshold while the assignments are MSL.
- **Anchoring the bins on MSL is NOT the fix** (measured, since it was proposed): fleet-wide it is
  a wash (87.6 → 83.2 ft at a 500 ft bin). It rescues KRDU (205.6 → 63.2) and KSTL and damages
  KSJC (73.8 → 93.7) and KSMF. Per-airport the current bin runs 73.8 (KSJC) to 205.6 (KRDU), and
  the spread tracks threshold elevation.

### H6 · the vertical angle is a criterion, not a word — and the two angles are different (2026-09-21)

The user proposed a vertical-angle word mirroring the heading word ("就像 LPV 规定一样"). Measured
on 13,043 altitude instructions / 7,486 flights (first 1,500 per airport; coverage stated).
**Two distinct quantities, kept apart because an earlier readout of mine conflated them:**

- **Position angle** = `atan(height above threshold / along-track distance to it)` — "am I on the
  published path". On the final segment (course ±30°, cross-track < 1 NM, before the threshold):
  p5 2.29 / p25 2.96 / **p50 3.05** / p75 3.13 / p95 3.35°; minus that runway's published
  glidepath, p50 **+0.04°**, and **88.5 % within ±0.5°**.
- **Flight path angle** = the aircraft's own descent gradient. Same instants: p5 −0.00 / p25 1.75 /
  **p50 2.52** / p75 3.47 / p95 4.73°, with **9.4 % level** and 89.9 % descending. Real descents are
  built from level and steep segments alternating around the path (a sample: 4131 ft at 12.8 NM
  descending 5.53° to regain it; 3043 ft at 12.8 NM level at 0.00° waiting for it).

**Conclusions, both from the same numbers:**
1. The position angle is an excellent **criterion** for D61's vertical check: tight, per runway,
   and referenced to a published value (KRDU 32 is **3.50°**, so it must be read per runway and
   never hard-coded to 3°). The flight path angle is not — it is broad and noisy.
2. The position angle is a **bad word**: 88.5 % of its mass lands in three 0.1° bins, so a model
   learns to always emit bin 0 — the identical pathology to always emitting altitude word 0, in
   different units.
3. **One angular scale cannot serve the whole approach.** Off the final course (40.1 % of altitude
   instructions) the same 0.1° resolution needs **598 words** to cover p1–p99 (p95 is +14.69° above
   the published path), because on downwind and base the straight line to the threshold is not the
   path the aircraft will fly. So the "angular everywhere, far field reads as large angles" variant
   is refused by measurement, and with it the cost of a word whose unit changes by segment
   (`altitude_centre_m` stays one conversion; the gate, `plan_conditioning=instruction`, the
   frontend colouring and the conditioning scaling are all untouched).

### H7 · what the altitude word is FOR: an anchor against drift, not a carrier of precision (2026-09-21)

Three measurements, in the order the user pushed for them. Together they change what the bin
width is chosen to do, so they belong with H5 rather than inside it.

**(a) Non-uniform bins win, once the gradient points at the levels.** The user asked twice for
non-uniform bins; my first two attempts measured the wrong shapes (a hand ladder whose coarse
segment's centres drifted off the round numbers, then one that was uniform-250 with a coarse tail)
and I twice reported that non-uniform buys nothing. Fitting the centres to the data instead
(Lloyd-Max on the fleet's real levels, one bin reserved for the degenerate final-descent event):

| bins | uniform | fitted |
|---|---|---|
| 11 | 235.3 ft | **134.4** |
| 21 | 112.4 | **72.7** |
| 26 | — | **53.3** |
| 41 | 62.4 | **35.5** |

So **26 fitted bins beat 41 uniform ones** (53.3 vs 62.4 ft) with the median class holding 1,275
examples instead of 242. The fitted centres are dense over 1500–3200 ft and sparse elsewhere —
exactly the shape the user described, pointed at the band the levels actually occupy (1600–3700 ft),
not at the ground. Cost: the centres become part of the data, so they freeze with the sha and a
cohort change (a v6 rebuild) would want refitting; `altitude_centre_m` becomes a 26-entry lookup.

**(b) The descent angle DOES hold, and is bimodal.** I had dismissed it as noise; that was wrong.
Read with the labeller's own plateau rule (10 s smoothing, ≥ 20 s, ±0.5°) over ~2,000 flights, the
flight path angle holds **11,199 plateaus, median 32 s** — the same order as the height plateaus.
But 13.0 % of them are level, and the 9,723 genuinely descending ones have an interquartile of only
**2.70–3.22°** (p50 3.01). So the aircraft does two things — level, or descend at ~3° — across the
WHOLE approach, not just the final segment. As a word that is one bit, and the height word already
carries it (a lower target means descend, the same target means level). D77's conclusion stands;
its stated reason ("散且带噪") does not and was replaced.

**(c) Removing the altitude word entirely is refused by its tail, not its centre.** The user
proposed dropping it: with descent pinned at 3° the altitude follows from the duration word, and
the model learns it implicitly. Reconstructing every flight's vertical profile from
`level/descend + duration + TRUE ground speed`:

- per segment: median descent 473 ft, median error **−2.1 ft**, median |error| **43.5 ft** — the
  3° constant is genuinely good, better than expected;
- per approach, accumulated: median |error| **153.4 ft**, p75 299, p95 1,189.

**Correction (2026-09-21, same day): those three per-approach figures were mislabelled.** They are
the sum of the errors made INSIDE each plateau; the transitions between plateaus were not counted
at all, so they are not the end-to-end profile error they were called. Re-measured with the
instructions tiling the track — each one in force from its own event until the next, so the
transitions belong to the preceding instruction, which is what a closed loop actually does — the
end-to-end height error at the runway with no altitude word is **median 218.1 ft, p95 1,442.6,
p99 2,327.7**. The direction of the finding is unchanged and slightly strengthened; the three
numbers above are not. Also visible only end-to-end: the worst case does not depend on any of
these choices, because **a flight that never levels off never triggers an anchor at all**.

**Method note, because this was the third instance in one day.** Three readouts were wrong in the
same way — a local quantity used as if it were a global one: an angle computed from a future
TARGET height against the present distance; a flight path angle taken as `arctan2(-dh, -ds)`,
which wraps to ±180° on downwind where the along-track distance grows; and this one, segments that
do not tile what they claim to summarise. Each was caught by the numbers being implausible, not by
reading the code. Before quoting a per-flight or per-approach aggregate built from segments, check
that the segments cover the flight.

The centre passes and the tail fails, for a structural reason: an absolute target's error is
bounded by half a bin, a rate's is unbounded and compounds over segments. The median flatters
itself because the signed errors cancel (median +4.7 ft). And this is a LOWER bound — it used the
true ground speed, while in closed loop the speed is a predicted word too, so the two errors
multiply.

**What this changes:** the altitude word's job is to bound drift, not to carry precision, because
the segment shape is already accurate to 43 ft without it. A bin chosen as an anchor can be
coarser than one chosen as a measurement — which also relieves H5's class-count problem
(200 ft leaves a median of 9 examples per class on KRDU alone). Pending numbers to settle
together: bin width under the anchor reading, uniform vs fitted, and the cohort (KRDU alone vs
five airports, which moves the per-class counts by an order of magnitude).
