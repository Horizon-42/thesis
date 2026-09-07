# Stage report: the 2026-09-03 → 09-08 experiment programme

*What each line was for, what moved, and what did not. Every number cites its source.*

Scope: the learned approach-trajectory predictor in `4dTrajectory/ts_transformer`. All numbers are
the **KRDU validation split** unless stated otherwise. Branch context: `dev-multi` / `dev-l2` at
merge `fd71a01`. This report reads and cites; it runs nothing and changes nothing.

---

## Abstract

Twelve experiment lines ran in six days. The representation questions are settled: 32 control
segments do the work of 64 at no cost, and a trajectory-error loss alone is not enough. The two
lines that tried to *recover* the missing information failed cleanly — a latent intent variable
(seven arms) and scene conditioning (a pre-measurement gate) both returned nothing. The line that
tried to be *given* it succeeded: told the true arrival time, pooled ADE falls 1214 → 841 m and
vectored ADE 2643 → 1596 m, and a counterfactual scan shows the model obeys a shifted arrival time
exactly, with graded geometry. A separate line made the corridor a deployable inference-time safety
layer. The reading is not "the latent failed" but "the model must not guess the intent; the
scheduler must supply it as a time." The author's proposal — decision pending — is to make CTA the
spine.

---

## 1. What the programme is for

The product is not a forecast. It is a **reference trajectory for an arrival scheduling programme**.
A transformer encoder reads the aircraft's own history to an anchor and emits **32 control segments
× 3 bounded controls plus a duration**; a differentiable point-mass rollout (RK4, no learned
parameters) integrates those into a 4D trajectory. The output is therefore dynamically consistent by
construction and warm-startable by an optimizer (`2026-09-07_latent_intent_design.zh.md` §二).

One constraint governs everything: **intent stays latent, or comes from the scheduler. It is never
an output.** A model that emits "this aircraft will join final at 14 km" is emitting an ATC decision
it has no authority to make (§一.4, §三). IPOPT gives the *optimal* trajectory in 4.3–56 s; this
model gives the *realistic* one in milliseconds, with a distribution and counterfactuals (§2.2).

**Metrics, defined once.**

- **ADE** — mean horizontal distance between predicted and observed position on a common true-time
  grid, metres. *Time-aligned*: it charges wrong place and right place at the wrong time alike.
- **FDE** — distance at the last point, quoted as mean or p50 (always stated).
- **chamfer** — a *time-free* shape error: paths resampled at 100 m, each point's distance to the
  nearest point of the other, median. ADE moving while chamfer stands still means the clock
  improved, not the shape.
- **bank skill** — per-flight correlation of predicted and observed bank schedules, read against the
  two references printed with every arm: a random-other-flight **floor** (KRDU 0.170) and a
  same-runway-twin **ceiling** (0.699). Never against 1.0 (`ts_transformer/CLAUDE.md`).
- **corridor violation rate** — share of rows, counted only where the *observed* track was already
  established on final, on which the prediction lies outside the LPV lateral corridor. Observed is
  0 % laterally by definition, 6.1 % on the vertical window at KRDU.
- **flyability** — share of predictions inside the envelope at *every* sample (stall, thrust, load
  factor, bank), read as a delta against observed tracks scored the same way (98.4 % here). Never
  alone: a blander predictor scores higher.
- **strata** — **straight-in** = route tortuosity < 1.05; **vectored** = tortuosity ≥ 1.05 and not
  established at the anchor. KRDU val is 1404 flights: 904 and 497. Vectored is where the error is.

---

## 2. The error budget everything is read against

From `2026-09-07_latent_intent_design.zh.md` §〇, measured on `closure_p1c_20260905`, **vectored
stratum, 497 flights**, untracked, single seed:

```
C_pred 2197 m ──962──> C_truth_intent 1235 m ──777──> C_oracle 458 m ──> 0
                what the aircraft        still undecided given      family + label
                cannot see itself        the true (d_join, T)       + reconstruction floor
```

Of the 2197 m, **962 m disappears the moment the model is told the true join point and remaining
duration** — information the aircraft's own history does not contain. A further **777 m** survives
even with the true intent: regression error inside the decoder. The last **458 m** is the family's
representation floor. Baselines on the same cohort: `simple-v3` control 1333 pooled / 2858 vectored
/ 469 straight-in; closure `C_pred` 996 / 2197 / 310.

**A discrepancy worth recording.** The budget writes 458 m and 777 m. The artifact it is drawn from
(`closure_p1c_20260905/readout_geometry.txt`, vectored block) reads **455 m**, and
`2026-09-06_closure_p1c_results.zh.md` §四.3 quotes the residual as **780 m**. Three metres, no
consequence — but they are not the same number.

---

## 3. The lines, their intent, and their result

| Line | Intent | Status | Headline (KRDU val) | Verdict |
|---|---|---|---|---|
| **L0** control basis | How many numbers must the decoder emit? | Done | N\*=32; fitted vectored ADE 203 m; N=64 → 81 m | Gate **failed literally**; "otherwise" branch |
| **L1** low-dim head | Is 32 free? Is trajectory error enough? | Done | 1322 vs 1333 m pooled; teacher-free 2515 m | **Passed** (width) / **vetoed** (dense) |
| **L1.b** supervision vs teacher | Replace the teacher with rollout terms | 1 of 3 arms read | bank skill 0.713 vs gate 0.726; straight-in FDE p50 **632 vs 671** | **Split**; hr16 blocked |
| **L1.c** training-time corridor | Retest the vetoed penalty, teacher-free | Pre-registered, **not run** | — | Pending |
| **L2** latent intent z | Discover the intent as a latent | Closed but for one arm | best 1214 m; posterior-mean shift **0.05–0.18 σ**, 7 arms | **Negative — the key finding** |
| **L3** CTA conditioning | Let the scheduler supply the time | Done + counterfactual | 1214 → **841** pooled, 2643 → **1596** vectored, 87.3 % wins | **Positive** |
| **L4** scene conditioning | Take intent from nearby traffic | Pre-measurement gate | R² 0.37 vs 0.38; 34.7 vs 35.1 s | **Failed — encoder not built** |
| **L5.a** fitted teacher | Upper bound on teacher quality | Code done, fit **not run** | — | Deprioritised |
| **A** anytime prediction | Re-plan at any observation time | Measured; A0.b untrained | L−1 ADE **2949** vs 1322 (veto); chamfer better at every bin | Infrastructure; veto stands |
| **B** calibrated ETA | Deliver an arrival-time interval | B0 measured; B1–B3 **untrained** | vectored \|Δt\| p80 **65.8–72.5 s** vs 12.0–20.3 s | Pending |
| **Constraints** | Keep the reference in the corridor | Done | barrier hook: pooled FDE mean 1650 → **1449 m** | **Passed** as inference layer; penalties **vetoed** |
| **Wind** | Is the residual wind? | Measured | slope +0.62 m/s per m/s, **R² 0.10** | Hypothesis **excluded** |

### 3.1 L0 / L1 — the representation (foundational, not a gain)

**Intent.** Before asking what the model should know, ask how many numbers it must emit. The
deployed head had 257 (64 segments); nobody had measured whether that was needed.

L0 fitted control schedules directly to observed tracks with no network
(`2026-09-07_l0_control_basis_results.zh.md`, `l0_control_basis_20260907/`). The pre-registered gate
— "there exists N\* ≤ 16 with vectored ADE ≤ 200 m" — **failed as written** (N=16 gave 315–330 m);
the "otherwise" branch set N\* = 32 (uniform 203 m, free-duration 191 m; N=64 reaches 91 / 81 m).
L1 tested that in training (`2026-09-07_l1_lowdim_results.zh.md`, `l1_lowdim_20260907`, 1404
flights, seed 1337, single seed, paired):

| Arm | pooled ADE / FDE p50 | straight-in | vectored | bank skill |
|---|---|---|---|---|
| `A_control_v3` (N=64, teacher) | 1333 / 908 | 469 | 2858 | 0.728 |
| **`L1_native32`** (N=32, teacher) | **1322 / 864** | **445** | 2870 | 0.726 |
| `L1_dense32` (N=32, no teacher) | 2515 / 3333 | 1455 | 4393 | 0.360 |
| `L1_dense64` (N=64, no teacher) | 2603 / 3293 | 1364 | 4811 | 0.324 |

**32 segments are free** — 96 parameters replace 257 at a 52.6 % paired win rate. And **the
trajectory-error loss alone is not enough**: the teacher-free arms are 1.9× worse on ADE, 2.7× on
FDE p50, and the 2026-08 bank wiggle returns (bank skill 0.360 against a 0.170 floor; straight-in
reference bank RMS 0.94° against the observed 0.41°). The pre-registered veto fired on both dense
arms (straight-in FDE p50 703 → 2863). Foundation, not progress.

### 3.2 L1.b — supervision instead of the teacher (marginal, and blocked)

**Intent.** The imitation teacher is inverse dynamics of the truth, and L0 measured how bad it is:
flown open-loop it lands 2.5–7.8 km from the truth, while a fitted table of the same width lands
within 88–433 m. Can two terms passed *through the rollout itself* — a heading-rate loss and a bank
total-variation penalty — do the teacher's only real job, naming the bank?

The 60-epoch screen (`l1b_supervision_20260907`) picked `hr8 + TV=1`. The 180-epoch confirmation arm
(`l1b_full_20260907/L1b_hr8_tv1_full`; verified against `readout_arm1.txt` and
`readout_bank_arm1.txt`) is **a split, not a pass**: bank skill **0.713** against a gate of 0.726;
ADE slightly worse in every stratum (1332 / 461 / 2872 vs 1322 / 445 / 2870); **FDE p50 better in
every stratum** (848 / 632 / 1722 vs 864 / 671 / 1982); straight-in reference bank RMS **0.12°** —
straighter than the real aircraft — with vectored length ratio 0.88 and chamfer worse (1040 vs 901).
The source reads this as **corner-cutting**: a shorter, smoother path to the right endpoint.
Endpoint metrics win, path metrics lose, gate missed by 0.013. Single seed.

**The hr16 arms are blocked, and not for a scientific reason.** `L1b_hr16_tv1_full` has a trained
checkpoint but **no prediction directory** — its `predict` step aborted with `ValueError: prediction
data is not an exact airport subset of the checkpoint training data` (`l1b_full_20260907.log`, step
6/12). `L1b_hr16_full` has only a `config.json`; never trained. **No gate number exists for either.**
See §7.

### 3.3 L2 — latent intent (the negative result that matters)

**Intent.** If the 962 m is intent, and intent must stay latent, learn it as a latent variable: a
posterior `q(z | the flight's own future)` against a prior `p(z | context)`, z never an output.
**Seven arms across five campaigns, all single-seed:** `L2_gauss` (β=1), `L2b_beta0p1`, `L2_mix4`
(K=4); `L2.d` warm posterior at β=0.1 and β=0.01; `L2.e′` free-bits 0.5 and 1.0; `L2.f` β-annealing
and a z→T auxiliary head.

The point estimate is real: warm β=0.01 gives pooled 1214 / straight-in 402 / vectored 2643, paired
win 65.3 %, bank skill 0.729, chamfer p50 165 vs 224 — **z bought 108 m of pooled ADE**. The
pre-registered *distribution* gates all failed: shuffled-z ΔADE +19 m against a 200 m gate, and
minADE₆ of the trained prior (1004) **loses to an N(0, I) control** (947).

The diagnosis — post hoc; the probe was written *after* L2.e′, and the design document says so — is
the finding. In **every one of the seven arms the posterior mean sits on the prior mean**:
displacement 0.05–0.18 prior σ against a verdict threshold of 1.0 σ, with the learned prior's total
σ only 0.27–0.58 where N(0, I) is 1.0. The KL budget went into narrowing the posterior *variance*,
not moving its *mean*. **z is a denoised constant.** The z-oracle arm confirms the scale: decoding
with the flight's own posterior z buys vectored 2643 → 2416 (−227 m), a fifth of the gate, and it
improves the *path*, not the endpoint (FDE p50 827 → 836). The mechanism is units: the loss is in
(m / 10 km)², so position gains are worth ~0.01 loss units, while at β = 0.01 one nat also costs
0.01. Five levers failed to make `q(z | future)` depend on the future. **User decision, 2026-09-08:**
close the line with one units-test arm (`L2z_units`, `position_loss_scale_m` 1 km,
`l2_units_test_arms.json`) — no campaign directory for it exists yet — keeping warm β=0.01 as the
point-estimate fallback.

### 3.4 L4 — scene conditioning (negative, same message)

**Intent.** If the intent is not in the aircraft's own history, it may be in the traffic around it.

A **pre-measurement** gated it before any encoder was built (`run_ts_scene_explainability.py`,
`l4_scene_explainability_20260907/`, KRDU 14,418 flights, Phase 0's cohort and CV protocol). It
reproduced Phase 0 to the decimal, then measured the increment: runway-use scalars give R² 0.37 on
the join distance and 34.7 s median error on remaining duration; adding four neighbour aircraft as
static rows gives 0.36 / 36.1 s, against a Phase 0 baseline of **0.38 / 35.1 s**. The gate
(0.55 / 28 s) **failed with zero increment**. Why: the data plane finds a lead aircraft for 96 % of
vectored flights, but the lead's *estimated* ETA correlates with its own *actual* landing time at
only **0.11** (median 275 s late, p10/p90 −90 / +1067 s). The lead's remaining time is as
undetermined as the own-ship's — another output of the same sequencing decision, not an observable
cause. **The encoder was not built.** A data-plane review found no leak or frame error, so the
measurement stands.

### 3.5 L3 — CTA conditioning (the positive result)

**Intent.** Stop guessing. Feed the target arrival time as a conditioning scalar and let the network
decide only *how to be there then* — exactly the scheduler's interface.

**Given the true arrival time** (`l3_cta_20260907`, 180 epochs, no early stop, warm β=0.01 base,
paired against the same base without CTA):

| Stratum | no CTA (ADE / FDE p50 / chamfer) | given CTA | paired win (median Δ) |
|---|---|---|---|
| pooled (1404) | 1214 / 827 / 165 | **841 / 745 / 128** | 65.5 % (−57 m) |
| straight-in (904) | 402 / 594 / 84 | 392 / 586 / 66 | — |
| vectored (497) | 2643 / 1851 / 827 | **1596 / 1180 / 721** | **87.3 % (−896 m)** |

Knowing *when* is worth **373 m of pooled ADE**, essentially all of it vectored (−1047 m);
straight-in moves 10 m. Against the latent line's 108 m, that is the whole argument in two numbers.
Note the shape: chamfer barely moves (165 → 128) — much better at *when*, hardly better at *where*.
Flyability at offset 0 is 8.3 % (`l3_cta_20260907.log`) against 98.4 % for observed tracks. **This is
an oracle arm and can never be quoted as prediction accuracy** (§三.3 requires `cta=given` in the run
name for that reason); 841 m is the ceiling a perfect ETA would buy.

**The counterfactual scan** (`l3_cta_counterfactual_20260907`, predict-only from the same
checkpoint, seven offsets, paired on the 1214-flight intersection; verified against `readout.txt`
and the campaign log):

| offset | \|Δdur\| p50 | fully flyable | chamfer p50 | length ratio | xt@thr p50 | \|xt\| p95 | vert. viol. | ADE | FDE p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −90 s | 90.0 | 44.9 % | 1038 | 0.59 | −254 | 3615 | 41.3 % | 3039 | 5504 |
| −60 s | 60.0 | 44.2 % | 622 | 0.75 | −350 | 3063 | 41.9 % | 2072 | 3314 |
| −30 s | 30.0 | 30.0 % | 333 | 0.86 | −293 | 2607 | 44.0 % | 1344 | 1685 |
| **0** | 0.0 | 8.3 % | 232 | 0.98 | −4 | 2598 | 47.3 % | 929 | 786 |
| +30 s | 30.0 | 0.3 % | 364 | 1.12 | 558 | 3166 | 49.4 % | 1165 | 1332 |
| +60 s | 60.0 | 0.0 % | 661 | 1.28 | 1517 | 4903 | 54.4 % | 1672 | 2074 |
| +90 s | 90.0 | 0.0 % | 1160 | 1.51 | 3156 | 8177 | 60.1 % | 2235 | 3047 |

**(1) The CTA is a hard constraint** — duration error equals the offset exactly at all seven points.
**(2) The geometry responds gradedly and does not collapse** — chamfer and length ratio are monotone
on both sides: less time → a shorter, straighter path; more time → a longer one. **(3) The asymmetry
is the result.** Asked to arrive early the model becomes *more* flyable (8.3 → 44.9 %); asked to
arrive late it becomes unflyable (0 %), because its only way to absorb delay is to lengthen the path
laterally — cross-track at the threshold reaches a median of 3156 m and a p95 of 8177 m, and
vertical-window violations rise 47 → 60 %.

The pre-registered gate asked for **per-sample** flyability not below the L2 arm, plus linear CTA
following. Linearity **passed** exactly, but the recorded flyability is the *whole-trajectory* share;
**the per-sample rate is not recorded for these arms**. Two confounders are named and untested:
early arms have shorter rollouts (a smaller row denominator), and whether the barrier hook would
pull the late arms back into the corridor.

### 3.6 The constraint line (a deployable safety layer)

**Intent.** A reference handed to a controller must lie inside the final-approach corridor.

**(a) Training-time penalties are vetoed on both output paths.** On the state path the dual-ascent
arm diverged in 4 of 4 runs and the fixed-λ arm bought violation rate with accuracy
(`2026-09-05_final_constraint_results.zh.md` §5). On the control path λ = 1e-3 improved the endpoint
(pooled FDE 1650 → 1549) but wrecked the mid-path (vectored ADE 2858 → 3439, paired improvement
23 %, median +658 m); λ = 5e-3 collapsed the bank schedule to a shared shape (skill 0.728 → 0.280)
(`2026-09-05_control_penalty_results.zh.md`). **(b) Bounded output works on the state path** — the
programme's only **replicated** result: two airports × two seeds, FDE improved in all four
(−91 / −66 / −51 / −63 m), corridor violation KRDU 77 → 48 %, threshold |xt| p95 492 → 305 m.
**(c) The inference-time soft barrier hook is a net gain on the control path**, with no retraining:
pooled FDE mean 1650 → **1449 m**, pooled ADE 1333 → 1278, straight-in endpoint |xt| p95
**1821 → 70 m**, 84 % of flights better and none more than 1 km worse
(`2026-09-06_control_hooks_results.zh.md`, single seed, KRDU + KSJC). Training *through* a hook was
worse than inference-only in all six arms — the network learns to delegate.

**L1.c** re-opens (a): the 09-05 veto used a base carrying the teacher, whose imitation term was
87 % of the loss and fought the penalty through the dynamics. Four arms are pre-registered in
`l1c_procedure_arms.json` with gates (violation rate −5 points ∧ vectored ADE ≤ +30 m ∧ bank skill
≥ 0.70; veto at vectored ADE > +100 m). **Not run.**

### 3.7 The A line — anytime prediction (infrastructure, not accuracy)

**Intent.** A scheduler re-plans continuously. A model anchored only at L−1 is useless at 12 km.

A0-fixed replayed three L−1-trained checkpoints at seven remaining-path bins. Fixed-anchor models
degrade away from their anchor; `warm β=0.01` beats `native32` at every bin (−274 m at 20 km); the
closure arm is most accurate far out but **collapses between 8 and 6 km** (vectored ADE 613 → 3990).
Gate 3 (a freeze point) was **unreadable**: the duration head's ~125 s floor makes |Δt| p80 *rise*
toward the runway (`native32` vectored: 64 s at 12 km, 141 s at 4 km).

Random-anchor arms **fail the pre-registered L−1 veto by a wide margin** — 2949–2990 vs 1322, all
three — yet draw **better chamfer at every bin** (−114…−524 m). `history.json` gave both mechanisms:
the LR scheduler stepped on a *selection metric* blind to what the arm improved (lr 9.4e-7 by epoch
60; training effectively stopped around epoch 30), and time-uniform anchor sampling over-weighted
the near end (34.3 % of draws under 6 km against 25.1 % of the population). The **A0.b** fix
(scheduler on the objective; remaining-path-uniform sampling) is implemented, reviewed,
bit-equivalent at the defaults — **and not trained**.

### 3.8 The B line — calibrated ETA (built, not trained)

**Intent.** A scheduler cannot use a point ETA. It needs an interval with a known coverage.

**B0** measured the distribution (`b0_eta_error_20260907`, 1404 flights, coverage 1404/1404):
vectored |Δt| p80 is **65.8–72.5 s** against straight-in **12.0–20.3 s**, a factor of 3.6–5.5 — so
per-stratum calibration is necessary, not a refinement. The signed median is off-centre
(+3.4 / +3.5 s late for the control arms, −1.4 s early for closure), which argues for a quantile
head over "point ± δ". The veto is a 120 s vectored interval width, and a symmetric 80 % interval
around the current p80 is already about twice 65.8 s: **B has roughly one doubling of margin**.

**B1** (monotone five-quantile duration head), **B2** (split-conformal calibration: half A fits the
deployed δ, half B measures coverage) and **B3** (`--cta-from-quantiles`, a fan of paths and the
first CTA arm that reads no future) are **implemented and reviewed, with no arm trained**. Two review
outcomes govern the reading. The gate number is the **deployed** coverage block, not the per-stratum
rows — in a synthetic check the pooled row read 0.794 while the 18 fall-through flights were covered
0.167. And the coverage claim is downgraded everywhere to **"empirically measured cross-half
coverage, not a finite-sample guarantee"**, because the calibration split is also the selection
split; the guarantee-bearing number is pre-registered as one test-split read at `freeze-test`.

### 3.9 Wind (a hypothesis excluded)

**Intent.** The straight-in residual is along-track and about 3–5 m/s — the size of wind, and the
rollout has no wind.

Measured, not modelled (`2026-09-08_wind_residual_readout.zh.md`, `wind_readout.json`, 839
straight-in flights with a report). Tower-headwind slope **+0.62 ± 0.13 m/s per m/s** (final 120 s:
+0.78), duration slope −2.0 ± 0.5 s per m/s, four wind bins monotone — physical signal, right sign
and magnitude — but **R² = 0.10**: regressing it out moves the residual sd from 4.8 to 4.5 m/s. The
endpoint along-track error is unrelated (R² 0.01). **Decision: no wind GPU arm.** The R² is
explicitly a lower bound — a 10 m tower wind is not the wind at altitude.

---

## 4. What moved and what did not

**Moved.** The control head shrank from 257 numbers to 96 with no loss. The corridor became
deployable without retraining (pooled FDE 1650 → 1449 m). Bounded output on the state path is the
only **replicated** win. A latent variable bought 108 m of pooled ADE. And telling the model the
arrival time bought **373 m pooled / 1047 m vectored**, which the counterfactual showed to be a real
control response, not a fitted output.

**Did not move.** **Top-1 accuracy from anything the aircraft carries** — seven latent arms, no
posterior mean depended on the future; L4, zero increment from scene features. **Geometry under
CTA** — chamfer 165 → 128 while ADE fell 373 m: CTA solves the timing half, the lateral half is
untouched. **The teacher question** — L1.b is a split at one arm and both hr16 arms are blocked, so
the decision rule cannot fire. **Anytime prediction** — all three random-anchor arms fail the L−1
veto; the fix is untrained. **Flyability of a *late* reference** — zero at +60 s and +90 s.

**Results the brief's skeleton omits, found while checking.** (i) **2026-09-03, the state path**: the
airport-frame ablation (H1 supported — the threshold anchor *is* the model's runway knowledge;
removing it raises wrong-sibling endpoints 1.5 % → 12–15 %; H2 and H3 not supported), the state-v2
anchor-relative candidate (**mechanism fixed, recipe vetoed** — vectored FDE −350 m at KRDU on both
seeds), and the runway-hypothesis expansion — two-seed results, the most replicated work in the
window. (ii) **The closure line (P1.a–P1.d, 09-05/06)**, which produced the error budget itself:
`C_pred`, own-history only, beats `simple-v3` on every stratum (996 vs 1333 pooled, 2197 vs 2858
vectored, 310 vs 469 straight-in); its skeleton was discarded on 09-07 for positioning, not accuracy.
(iii) **Scene Phase 0 (09-05)**: the true join point cuts vectored ADE 2858 → 2356 (−17 %) and halves
the duration error, but the pre-registered gate was < 1.5 km and **was not met**. (iv) **The package
audit (T0–T3, 09-07)** — why several documents carry "archived" notes. None of this changes §5;
(i) and (ii) show that the window's most replicated work sits on an output path the programme has
since left.

---

## 5. The reading: CTA as the spine

The dominant residual is **along-track and timing**, in the vectored stratum. Two independent lines
say it is **not recoverable from what the aircraft can see**: seven latent arms could not make a
posterior depend on the future, and a scene pre-measurement found zero increment with the mechanism
named — the lead aircraft's own remaining time is as undetermined as the own-ship's, because both
are outputs of the same sequencing decision. One line says it is **largely resolved when the arrival
time is supplied**: −1047 m of vectored ADE, on 87.3 % of flights. So the conclusion is not "the
latent failed." It is:

> **The model should not guess the intent. The scheduler should supply it, as a time.**

That is not a retreat — it is the delivery form the design argued for from the start (§2.2:
distribution, counterfactual, warm-startable parameters), and it reframes L2's failure as
*consistent with the structure of the problem*: a negative result with a mechanism, not an absence.

Two cautions. The 841 m is an **oracle**; the honest version needs B, which is untrained, and B0 says
the margin against its own veto is about one doubling. And CTA fixes *when*, not *where*: chamfer
barely moved, and the counterfactual shows the model absorbs delay by leaving the corridor —
precisely what a scheduler will most often ask for.

---

## 6. Proposed next steps, and the gate each would carry

*The author's proposal. Decision pending; none of this is a recorded decision.*

1. **Train B1–B3 now.** Code, arms and commands exist. Gates as pre-registered: B1 — the median is
   not worse than the point head within seed noise; B2 — **deployed** 80 % coverage in [0.76, 0.84],
   read from the `DEPLOYED` block; B3 — each quantile path's flyability not below top-1, and the
   truth inside the fan at no less than nominal coverage. Veto: vectored interval width median
   > 120 s.
2. **New, to pre-register: delay absorption inside the corridor.** The counterfactual showed a late
   CTA is absorbed by lateral excursion (xt@thr p50 3156 m at +90 s), and a scheduler mostly asks for
   delay. So put the corridor and the CTA together on the `L3_cta` checkpoint: the inference-time
   soft barrier hook first (predict-only, hours), the L1.c training-time term second if the hook is
   not enough. Proposed gate: at +30 / +60 / +90 s the threshold |xt| p95 falls by at least half,
   vertical violation rows do not rise, and vectored ADE degrades by no more than 100 m against the
   un-hooked arm. Veto: flyability at offset 0 falls below the un-hooked 8.3 %.
3. **Anytime × CTA.** A0.b's arms are built and untrained. Run them under the existing L−1 veto (not
   worse than `native32` beyond seed noise) with the anchor-grid selection metric, then repeat the
   CTA arm at non-L−1 anchors.
4. **Deprioritise.** L5.a stays an upper bound on teacher quality (its fit has never run; the
   campaign directory is empty). L2 gets its one units-test arm and closes either way. **No further
   teacher arms after hr16** — and hr16 needs the provenance blocker cleared first.

---

## 7. Risks and open hazards

- **The roster/provenance incident of 2026-09-07 is live, and it has already cost an arm.** The five
  canonical observed reports were regenerated to v9 on 2026-09-07 08:14; the eligible flight sets are
  **identical** (KRDU 14,378 / KSJC 11,076 / KSTL 8,761 / KSMF 4,216 / KMSY 4,140), but the recorded
  report SHA-256 and roster digests moved. Because `data_provenance` is compared for equality,
  **every checkpoint trained before that moment refuses to replay against the current data**
  (`docs/open-items.md`); the concrete casualty is `L1b_hr16_tv1_full` (§3.2). A v6 backup exists at
  `outputs/evaluation_reports/observed_v6_backup_2026-09-07/`. A related bug — provenance rebuilt
  without the eligibility roster, reporting 14,435 candidates against a checkpoint's 14,378 — was
  fixed in `e8df12f`; it had been a live startup blocker for L5.a's unrun fit.
- **The KSMF / KRDU rebuild hazard.** KSMF 35R's configured threshold is **39.4 m** off (the
  OurAirports row gives both ends of 17L/35R the same longitude), failing all 259 of its observed
  crossings laterally. Fixing it changes the 35R physical frame, so KSMF needs
  `--reclassify-existing` plus an arrivals rebuild (+259 arrivals) — **which changes its ts split**.
  KRDU 32 is 3.0 m off and harmless, but a KRDU reclassify would add 1,604 RW32 arrivals. Separately,
  `--evaluate-only` **deletes** the v5 arrivals roster and since 2026-09-07 rewrites it as v6
  (+1,876 arrivals), changing every ts split. Do not run it under a campaign.
- **Almost everything here is single-seed** (1337): L1, L1.b, the seven L2 arms, L3, the control-path
  constraint work and the hooks; only the 09-03 state-path work and the bounded-output arms are
  two-seed. The repository's rule — margins under ~1.5× are provisional — applies to most of §3.
  Related: **a per-airport ADE without its route mix is not a comparison** (KSJC 483 → 1526 m
  reweighted), so every number here must stay labelled KRDU-only.
- **Oracle arms must never be quoted as accuracy** — L3's 841 m, the truth-intent arms and the
  z-oracle arm all read the future; hence the `cta=given` / `intent=truth-…` / `z=posterior` naming
  discipline.
- **Two small unresolved items**: the 458 / 455 m mismatch (§2), and the two documented gaps in the
  L3 counterfactual (per-sample flyability not recorded; the shorter-rollout confound untested).

---

## 8. Sources

**Design documents** (`4dTrajectory/ts_transformer/docs/`, status table in §〇 of each):
`2026-09-07_latent_intent_design.zh.md` (§〇 budget/status; §一–§三 positioning and contracts;
§六 L0–L5.a; §七 vetoes); `2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
(§〇 status and reference numbers; §〇.2 B0; §〇.3 commands; §〇.4 review fixes; §2.4b/2.4c A0; §三 B).

**Results documents** (same directory): `2026-09-07_l0_control_basis_results.zh.md`,
`2026-09-07_l1_lowdim_results.zh.md`, `2026-09-05_final_constraint_results.zh.md`,
`2026-09-05_control_penalty_results.zh.md`, `2026-09-06_control_hooks_results.zh.md`,
`2026-09-05_scene_phase0_results.zh.md`, `2026-09-06_closure_p1c_results.zh.md`,
`2026-09-06_closure_p1d_tracking_results.zh.md`, `2026-09-03_airport_frame_ablation_results.md`,
`2026-09-03_state_v2_anchor_relative_results.md`, `2026-09-03_runway_hypothesis_expansion.md`,
`2026-09-08_wind_residual_readout.zh.md`.

**Artifacts** (`4dTrajectory/outputs/KRDU/experiments/`):
`closure_p1c_20260905/readout_geometry.{txt,json}` (the budget's source rows);
`l1_lowdim_20260907/{readout.json,readout_bank.txt,wind_readout.json}`;
`l1b_full_20260907/{readout_arm1.txt,readout_bank_arm1.txt}` + `l1b_full_20260907.log` (the hr16
provenance failure, step 6/12); `l2f_mean_information_20260907/` incl. `probe/`;
`l2_warm_posterior_20260907/`; `l2e_free_bits_20260907/`; `l2d_zoracle_20260907/`;
`l3_cta_20260907/` + `.log`; `l3_cta_counterfactual_20260907/readout.{txt,json}` + `.log`;
`l4_scene_explainability_20260907/`; `l0_control_basis_20260907/`; `b0_eta_error_20260907/`;
`anytime_a0_20260907/`, `a0_random_20260907/`; `l5_fitted_teacher_20260907/` — **empty, the fit has
not run**.

**Pre-registered arm files** (`ts_transformer/docs/experiments/`): `l1_lowdim_arms.json`,
`l1b_full_arms.json`, `l1c_procedure_arms.json`, `l2f_mean_information_arms.json`,
`l2_units_test_arms.json`, `l3_cta_arms.json`, `l3_cta_counterfactual_arms.json`,
`a0_random_arms.json`, `b1_quantile_arms.json`, `l5_fitted_teacher_arms.json`.

**Standing rules and hazards.** `ts_transformer/CLAUDE.md` ("How to read results here" and the
config-axis table); `docs/CHANGELOG.md` entries 2026-09-03 → 09-08; `docs/open-items.md`; root
`CLAUDE.md` Open Items.
