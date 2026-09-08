# The 2026-09-03 → 09-08 experiment programme: results, conclusion and plan

*Final report. It supersedes `2026-09-08_programme_stage_report.md`, written on the morning of
09-08, which covers only the lines finished by then. This one carries the whole window, the overall
conclusion and the plan. It reads artifacts and documents; it runs nothing.*

Subject: the learned approach-trajectory predictor in `4dTrajectory/ts_transformer`. Unless stated
otherwise every number is the **KRDU validation split, 1404 flights, seed 1337, single seed, paired
flight by flight**. Source keys (`LID`, `AED`, `E/…`) are expanded in §7.

---

## 1. Purpose and deliverables

The product is a **reference-trajectory generator for an arrival scheduling programme** (an
AMAN-class tool), not a forecast for its own sake. The scheduler needs three things (`LID` §2.2):
**(a)** a calibrated arrival-time distribution when no time has been assigned; **(b)** a flyable
reference trajectory for an assigned arrival time (a CTA); **(c)** re-issued predictions as the
flight progresses.

One constraint governs everything: **the controller's intent is either latent inside the model or
supplied by the scheduler. It is never a model output** (`LID` §一.4, §三). A model that emits
"this aircraft will join final at 14 km" is emitting an air traffic control decision it has no
authority to make. IPOPT gives the *optimal* trajectory in 4.3–56 s; this model gives the
*realistic* one in milliseconds, with a distribution and counterfactuals.

**The error budget** (`E/closure_p1c_20260905`, vectored stratum, 497 flights, untracked, single
seed; `LID` §〇):

```
C_pred 2197 m ──962──> C_truth_intent 1235 m ──777──> C_oracle 458 m ──458──> 0
             what the aircraft's       still undecided given       family + label
             own history cannot see    the true (d_join, T)        + reconstruction floor
```

Of 2197 m of vectored ADE, **962 m disappears the moment the model is told the true join point and
remaining duration**; **777 m** survives even then (decoder regression error); the last **458 m**
is the closure family's representation floor. Baselines on the same cohort: `simple-v3` control
1333 pooled / 2858 vectored / 469 straight-in; closure `C_pred` 996 / 2197 / 310. **One recorded
inconsistency**: the budget writes 458 m and 777 m while
`E/closure_p1c_20260905/readout_geometry.txt` reads **455 m** and
`2026-09-06_closure_p1c_results.zh.md` §四.3 quotes **780 m**.

---

## 2. Data, model and metrics

**Data.** KRDU arrivals, v5 roster (`harvest-arrivals-v5-takeoff-excluded`). Validation holds
**1404 flights** on the control path (the `openap-direct` fleet subset the point-mass dynamics can
model) and **2104** on the state path (whole fleet), so the two paths are **not on the same
flights** and their numbers are not comparable (`CPR`). The test split is untouched.

**Model.** An iTransformer encoder reads the aircraft's own history to an anchor and emits
**N = 32 control segments × 3 bounded controls, plus a duration**. Controls are dimensionless:
thrust fraction ∈ [−0.2, 1.0], bank ∈ ±π/4, load factor ∈ [0.2, 2.0]. A differentiable RK4
point-mass rollout with **no learned parameters** integrates them into a 4D trajectory, so every
prediction is dynamically admissible by construction. The default anchor is L−1, the last sample of
the lookback window.

**Strata.** `approach_difficulty.strata_masks` cuts on route tortuosity at **1.05**:
**straight-in** = tortuosity < 1.05 (904 flights); **vectored** = tortuosity ≥ 1.05 and not
established at the anchor (497). Disjoint, and they do not cover the cohort. Almost all error is
vectored.

| Metric | Definition | How to read it |
|---|---|---|
| **ADE** | Average Displacement Error: mean horizontal distance between predicted and observed position on a common true-time grid, metres. Mean unless a percentile is named. | Time-aligned. Charges "wrong place" and "right place at the wrong time" alike. |
| **FDE** | Final Displacement Error: distance at the last point, mean or p50 (always stated). | Endpoint only. |
| **chamfer** | Time-free shape error: both paths resampled at 100 m, each point charged its distance to the nearest point of the other, median. | ADE falling while chamfer stands still means the clock improved, not the shape. |
| **Fréchet** | Discrete Fréchet distance between the two polylines, p50. | Second time-free measure; more sensitive to one bad excursion. |
| **bank skill** | Per-flight correlation of the predicted and observed bank schedules. | Read against the two references printed with every arm: random-other-flight **floor 0.170**, same-runway-twin **ceiling 0.699** (KRDU). Never against 1.0, which is unreachable. |
| **corridor violation rate** | Share of rows — counted only where the *observed* track was already established on final — on which the prediction lies outside the LPV lateral corridor. | Observed floor: 0 % lateral by definition, 6.1 % on the vertical window at KRDU (`FCR` §2). |
| **fully-flyable share** | Share of predictions inside the envelope at **every** sample of the whole trajectory (stall, thrust, load factor, bank). | Dominated by the segment furthest from the runway: on the L3.c arms **77–94 % of stall violations sit at remaining path ≥ 20 km** (`LID` §六 L3.c). Observed tracks score **98.4 %**, so 8–10 % is a delta of −88 to −90 points. A blander predictor scores higher, so it is never quoted alone. |
| **duration MAE** | Mean absolute error of the predicted total remaining duration, seconds. | `final_time_error_s` is the error of the duration the rollout actually flew. |
| **deployed coverage** | Share of held-out flights whose true duration falls in the conformal interval they would **really be given**, after the stratum precedence has fallen through. | The gate number, not the per-stratum rows: in a synthetic check the pooled row read 0.794 while the 18 fall-through flights were covered 0.167 (`AED` §3.2). |

Both metric families are read together, time-aligned and time-free (`LID` §八). Margins under about
1.5× are provisional.

**The seed-noise line, corrected on 2026-09-08 and applied throughout this report.** The
pre-registered 30 m line came from two-seed **state**-path arms, whose measured spread is 5–22 m of
pooled ADE. The only two-seed **control**-path pair measured in this window, `B1_point_matched` at
seeds 1337 and 2024, differs by **125 m of pooled ADE** (1248 against 1373) — four times that line.
Part of that is convergence rather than initialisation: one arm ran the full 180 epochs, the other
early-stopped at 143. **A single-seed control-arm pooled-ADE difference below about 125 m is
therefore not evidence**, and every such difference below is marked "not demonstrated" where it
appears. Duration-MAE seed spread is about 1 s pooled and 0.85 s straight-in; straight-in FDE
spread is smaller but also unreplicated. (`AED` §3.5; `ts_transformer/CLAUDE.md`, first bullet.)

---

## 3. Results

### 3.1 Does the model reproduce the approach path?

**Yes, geometrically. The residual is timing, not shape.**

**L0 — how many numbers must the decoder emit?** (pre-registered, no training; `LID` §六 L0;
`2026-09-07_l0_control_basis_results.zh.md`; `E/l0_control_basis_20260907`.) Control schedules
fitted to observed tracks through the same rollout, no network. The gate — "there exists N\* ≤ 16
with vectored ADE ≤ 200 m" — **failed as written** (N = 16 gave 315–330 m); the "otherwise" branch
set **N\* = 32** (uniform 203 m, free durations 191 m; N = 64 reaches 91 / 81 m).

**L1 — is 32 free, and is a trajectory-error loss enough?** (pre-registered; `L1R`;
`E/l1_lowdim_20260907`.)

| Arm | Width | Teacher | pooled ADE / FDE p50 / chamfer / Fréchet | straight-in | vectored | bank skill |
|---|---:|---|---|---|---|---:|
| `A_control_v3` | 64 | inverse-dynamics 64× | 1333 / 908 / 256 / 1426 | 469 / 703 / 134 | 2858 / 1971 / 942 | 0.728 |
| **`L1_native32`** | 32 | inverse-dynamics 64× | **1322 / 864 / 224 / 1357** | **445 / 671 / 109** | 2870 / 1982 / 901 | 0.726 |
| `L1_dense32` | 32 | none | 2515 / 3333 / 869 / 4264 | 1455 / 2863 / 371 | 4393 / 5138 / 1842 | 0.360 |
| `L1_dense64` | 64 | none | 2603 / 3293 / 731 / 4391 | 1364 / 2576 / 404 | 4811 / 6312 / 1950 | 0.324 |

**Gate passed on width**: 32 segments do the work of 64, so 96 free numbers replace 257 at a 52.6 %
paired win rate and equal bank skill. **Veto fired on both teacher-free arms**: straight-in FDE p50
703 → 2863 / 2576, and the bank-wiggle signature returned (skill 0.360 against a 0.170 floor;
straight-in reference bank RMS 0.94° against the observed 0.41°). `L1_native32` ran 180 epochs
without early stopping, so **1322 m is a budget-limited upper bound** (`L1R` §三.4).

**Straight-in residual decomposition** (measurement, no gate, post hoc; `SIR`;
`E/l1_lowdim_20260907/straight_in_residual.json`; 904 flights; errors projected onto the observed
heading, right normal and vertical).

| Component (p50 / mean / p90, m) | `L1_native32` | `L3_cta` (true duration given) |
|---|---|---|
| along-track RMS | **333 / 404 / 680** | **316 / 394 / 688** |
| cross-track RMS | 112 / 213 / 423 | 71 / 155 / — |
| vertical RMS | 78 / 86 / 121 | 65 / — / — |
| along-track share of horizontal error² (p50) | 0.89 | 0.95 |
| along-track mean (p50) | **−88** | **+8** |

**(1) The path is straight**: **89–95 % of the squared horizontal error is along-track**.
**(2) The along-track error concentrates in the last 5 km and is scatter, not bias**: beyond 15 km
the median is ±0 m, the 10–5 km band spreads p10..p90 −421..+315 m, the 5–0 km band −819..+591 m,
and `L1_native32` carries a **−191 m** median bias in the last band (a predicted duration about 2 s
long). **(3) The true total duration removes the bias but not the scatter**: last-band median
**−191 → +8 m**, along-track RMS only **333 → 316 m**. The source is the *shape* of the speed
profile, not the total time.

**What explains the scatter.** The deceleration point (remaining distance where predicted speed
first drops below V_target + 10 m/s) has a median offset of 0.0 km but p10/p90 **−1.4 / +1.5 km**,
about ±16 s at 85 m/s; correlated with the last-band along-track error at r = −0.46,
**R² = 0.21–0.22**, slope −225 m per km (`SIR` §二). The tower headwind explains **R² = 0.10** of
the straight-in speed residual: slope **+0.62 ± 0.13 m/s per m/s** (final 120 s +0.78), duration
slope **−2.0 ± 0.5 s per m/s**, four wind bins monotone, but regressing it out moves the residual
sd only 4.8 → 4.5 m/s, and the endpoint along-track error is unrelated (R² 0.01) (`WIN`;
`E/l1_lowdim_20260907/wind_readout.json`, 839 of 904 flights with a report). **Decision: no wind
GPU arm**; that R² is a lower bound, since a 10 m tower wind is not the wind at altitude. Where a
crew decelerates ("170 knots to 5 DME") is a speed instruction; like vectoring it is **not in the
aircraft's own history**.

### 3.2 Can the model infer the controller's intent from its own history?

**Only partly, and not the timing half. This is the programme's most-revised finding.**

**L2 — latent intent.** (pre-registered; `LID` §六 L2; five campaigns, **seven arms, all single
seed**: `L2_gauss` β=1, `L2b_beta0p1`, `L2_mix4` K=4, `L2.d` warm posterior at β=0.1 and β=0.01,
`L2.e′` free bits 0.5 and 1.0, `L2.f` β-annealing and a z→T auxiliary head.) A posterior
q(z | the flight's own future) trained against a prior p(z | context); z never leaves the model.
The point estimate was real — warm β=0.01 reaches pooled 1214 / straight-in 402 / vectored 2643,
chamfer 165 against native32's 224, so 108 m of pooled ADE — but the pre-registered **distribution
gates all failed**: shuffled-z ΔADE +19 m against a 200 m gate, and minADE₆ of the trained prior
(1004 m) **lost to an N(0, I) control** (947 m). The **post-hoc** diagnosis (the probe was written
after `L2.e′`) was that in all seven arms the posterior mean sat on the prior mean — displacement
**0.05–0.18 prior σ** against a 1.0 σ verdict, prior total σ only **0.27–0.58** where N(0, I) is
1.0 — and the proposed cause was units: the loss is scaled in (m / 10 km)², so a position gain is
worth about 0.01 loss units while at β = 0.01 one nat also costs 0.01.

**The units test (`L2z_units`) confirms that cause and reverses the reading.** One arm,
`position_loss_scale_m` 10 km → 1 km, everything else as the warm β=0.01 fallback (`LID` §六
"L2 量纲测试结果"; `E/l2_units_test_20260908/{readout.json, latent_readout_L2z_units.json,
probe/latent_probe.txt}`; 180/180 epochs, best epoch 180 — **budget-limited, so lower bounds**).

| Arm | pooled / straight-in / vectored ADE | FDE p50 pooled / s-i / vec | shuffled-z ΔADE pooled / vectored | minADE₆ prior vs N(0, I), pooled / vectored | prior total σ / active dims | posterior-mean displacement median (p90) |
|---|---|---|---:|---|---:|---:|
| `L1_native32` (no latent) | 1322 / 445 / 2870 | 864 / 671 / 1982 | — | — | — | — |
| `L2d_warm_beta0p01` | 1214 / 402 / 2643 | 827 / 594 / 1851 | ≤ 60 m | 1004 vs 947 (**loses**) | 0.583 / — | 0.079 (0.289) |
| **`L2z_units`** | 1249 / 447 / 2660 | **688 / 498 / 1519** | **+928 / +2444 m** | **1021 vs 1043 / 2084 vs 2163 (wins)** | **0.848 / 8** | 0.334 (**1.04**) |

Of the four pre-registered gates it **passes three and fails one**: shuffled-z ΔADE > 200 m
**passed by 4.6×**; minADE₆ better than the N(0, I) control **passed for the first time in five
campaigns**; top-1 not worse than native32 **passed**; posterior-mean displacement > 1 σ **failed**
at the median (0.334 σ, p90 1.04). Loss components at the best epoch — state 2.28, imitation 1.10,
terminal 0.40, velocity 0.05, `latent_kl` 0.04 — put the reconstruction term at the same order as
the nats, as the units argument predicted.

**(1) z is genuinely used for the first time**: shuffling it costs 928 m pooled and 2444 m vectored
where every earlier arm cost ≤ 60 m, and the learned prior is back to near unit width. That also
explains the result seen in five campaigns, that the trained prior scored worse than an N(0, I)
control: the prior had narrowed to a third of the correct width. **(2) The two verdicts split**:
the reading that z is a constant cannot stand against a 2444 m shuffle cost, and the 1 σ threshold
was calibrated on arms where both criteria agreed. **(3) It improves the
endpoint, not the path average**: FDE p50 is 170 m better than native32 pooled and 463 m better
vectored, while pooled ADE stands still (1249 against `L2d`'s 1214, still the best point estimate).
So the seven earlier arms failed **for a units reason**, not because a latent cannot carry the
intent — yet the corrected arm still does not recover the 962 m, its vectored top-1 being 2660 m
against a 1235 m target. The line is therefore **not closed but re-opened once**, as
**L2.g** (pre-registered `LID` §六 L2 末, `l2g_latent_distribution_arms.json`, campaign
`l2g_latent_distribution_20260908`, queued last): the same units recipe at **300 epochs with early
stopping off and two seeds**, read as a distribution. Its gates and veto are in §5 item 5. It is
**the only experiment still pending in this programme.**

**L4 — scene conditioning.** (pre-registered pre-measurement gate; `LID` §六 L4;
`E/l4_scene_explainability_20260907`, KRDU 14,418 flights, Phase 0 cohort and CV protocol.) It ran
**before** any encoder was built, reproduced Phase 0 exactly, then measured the increment:
runway-use scalars give R² 0.37 on the join distance and 34.7 s median error on remaining duration;
adding four neighbour aircraft as static rows gives 0.36 / 36.1 s against a Phase 0 baseline of
**0.38 / 35.1 s**. The gate (0.55 / 28 s) **failed with zero increment**. The mechanism was
measured too: the data plane finds a lead aircraft for 96 % of vectored flights, but the lead's
*estimated* ETA correlates with its own *actual* landing time at only **0.11** (median 275 s late,
p10/p90 −90 / +1067 s) — the lead's remaining time is as undetermined as the own-ship's, both being
outputs of the same sequencing decision. **The encoder was not built**; a data-plane review found
no leak or frame error, so the measurement stands.

Together: neighbouring traffic contributes nothing, and the aircraft's own history — once the
latent can use it — improves the endpoint but not the timing.

### 3.3 What happens when the scheduler supplies the arrival time?

**Most of the residual resolves, and the model obeys the assigned time exactly.**

**L3 — CTA conditioning.** (pre-registered; `LID` §六 L3; `E/l3_cta_20260907`, 180 epochs, no early
stop, warm β=0.01 base, paired against the same base with no CTA.) `cta_conditioning=given` makes
the target arrival time **be** the duration; the network decides only how to be there then.

| Stratum | no CTA: ADE / FDE p50 / chamfer | given true CTA | paired win (median Δ) |
|---|---|---|---|
| pooled (1404) | 1214 / 827 / 165 | **841 / 745 / 128** | 65.5 % (−57 m) |
| straight-in (904) | 402 / 594 / 84 | 392 / 586 / 66 | — |
| vectored (497) | 2643 / 1851 / 827 | **1596 / 1180 / 721** | **87.3 % (−896 m)** |

Knowing *when* is worth **373 m of pooled ADE**, essentially all vectored (**−1047 m**, on 87.3 %
of flights); straight-in moves 10 m, consistent with §3.1. Two limits: chamfer barely moves
(165 → 128), so CTA fixes *when*, not *where*; and this is an **oracle arm** whose run name carries
`cta=given`, so **841 m may never be quoted as prediction accuracy** (`LID` §三.3) — it is the
upper bound a perfect ETA would give.

**Counterfactual scan.** (pre-registered; `E/l3_cta_counterfactual_20260907`, predict-only from the
same checkpoint, seven offsets, paired on the 1214-flight intersection, where offset 0 reads ADE
929 / FDE p50 786 against the full cohort's 841 / 745.)

| offset | \|Δduration\| p50 | fully flyable | chamfer p50 | length ratio | xt@thr p50 | \|xt\| p95 | vert. viol. | ADE | FDE p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −90 s | 90.0 | 44.9 % | 1038 | 0.59 | −254 | 3615 | 41.3 % | 3039 | 5504 |
| −60 s | 60.0 | 44.2 % | 622 | 0.75 | −350 | 3063 | 41.9 % | 2072 | 3314 |
| −30 s | 30.0 | 30.0 % | 333 | 0.86 | −293 | 2607 | 44.0 % | 1344 | 1685 |
| **0** | 0.0 | 8.3 % | 232 | 0.98 | −4 | 2598 | 47.3 % | 929 | 786 |
| +30 s | 30.0 | 0.3 % | 364 | 1.12 | 558 | 3166 | 49.4 % | 1165 | 1332 |
| +60 s | 60.0 | 0.0 % | 661 | 1.28 | 1517 | 4903 | 54.4 % | 1672 | 2074 |
| +90 s | 90.0 | 0.0 % | 1160 | 1.51 | 3156 | 8177 | 60.1 % | 2235 | 3047 |

**(1) The CTA is a hard constraint**: duration error equals the offset exactly at all seven points.
**(2) The geometry responds gradedly**: chamfer and length ratio are monotone on both sides, with
and neither degrades at ±90 s. **(3) The asymmetry is the result.** Asked to arrive early the model becomes *more*
flyable (8.3 → 44.9 %); asked to arrive late it becomes unflyable (0 %), because its only way to
absorb delay is a lateral excursion — cross-track at the threshold reaches a median 3156 m and p95
8177 m at +90 s, with vertical violations rising 47 → 60 %. The gate asked for **per-sample**
flyability; the recorded number is the whole-trajectory share and **the per-sample rate is not
recorded for these arms**. Two confounders are named and untested: early arms have shorter
rollouts, hence a smaller row denominator; and whether the hook would contain the late arms, which
L3.c then measured.

**L3.c — delay absorption inside the corridor.** (pre-registered 2026-09-08; `LID` §六 L3.c;
`E/l3c_delay_corridor_20260908`, predict-only from the same `L3_cta` checkpoint, four offsets each
with the soft barrier hook, paired on 1402 flights.)

| Arm | ADE | FDE p50 | chamfer | length ratio | xt@thr p50 | \|xt\| p95 | lateral viol. | vert. viol. | fully flyable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offset 0, no hook (`L3_cta`) | 842 | 746 | 129 | 0.98 | −23 | 2402 | 84.4 % | 43.2 % | 8.3 % |
| offset 0 + hook | 840 | **582** | **70** | 0.98 | −1 | **1611** | **43.2 %** | 43.2 % | 8.3 % |
| +30 s + hook | 1045 | 891 | 176 | 1.13 | −5 | 2010 | 50.1 % | 45.5 % | 0.3 % |
| +60 s + hook | 1451 | 1101 | 426 | 1.30 | **16** | 2364 | 85.9 % | 50.5 % | 0.0 % |
| +90 s + hook | 1894 | 1328 | 822 | 1.53 | **5** | 2332 | 98.8 % | 56.2 % | 0.0 % |
| +30 s, no hook | 1051 | 1125 | 314 | 1.13 | 524 | 2975 | 96.1 % | 45.4 % | 0.3 % |
| +60 s, no hook | 1502 | 1601 | 624 | 1.30 | 1539 | 4737 | 99.4 % | 50.4 % | 0.0 % |
| +90 s, no hook | 2000 | 2431 | 1117 | 1.53 | 3288 | 7694 | 99.9 % | 56.2 % | 0.0 % |

**Gate 2 passed**: duration error equals the offset exactly for every arm and quantile — the delay
is absorbed in time. **Gate 1 split**: the endpoint clause passed (|xt| p95 2364 ≤ 1.5 × 1611), the
any-point lateral violation clause failed (85.9 % against 43.2 %). **Gate 3 failed**: the +60 s
arm's fully-flyable share is 0.0 %, flight for flight identical to the un-hooked arm. **The veto
did not fire** (offset 0 with hook 840 against 842 without).

**(1) The hook does what it was designed for, and on the CTA base it is free**: at +90 s
cross-track at the threshold goes **3288 → 5 m** (p95 7694 → 2332); at offset 0 lateral violations
84 → 43 %, chamfer 129 → 70, FDE p50 746 → 582, ADE unchanged. **(2) Absorption does not depend on
the hook**: both +60 s arms show the same per-band ground-speed difference (about 2 m/s in the
10–5 km band) and the same length ratio 1.30, and cross-track RMS in the 20–10 km segment is
**1–3 m** — a longer but smooth path, not weaving. **(3) The unflyability is the stall term**
(99.9 % of hard violations), and **77–94 % of stall samples are at remaining path ≥ 20 km**; the
rollout tail past the threshold accounts for only 5 % (22 % on the +60 s arm), and truncating there
moves the fully-flyable share 8.3 → 10.5 % (+60 s: 0 → 1.4 %). In the last 10 km the minimum ground
speed is **84.6 m/s**, above the 74.6 m/s target: the model does not slow down near the runway, it
slows too much in the outer segment (downwind and base for vectored flights) where required Cl
exceeds Cl_max. That is already true at offset 0, and the delay is placed in the same segment. The outcome
matches none of the three pre-registered branches — the delay was neither absorbed legitimately by
deceleration nor pushed upstream of the join.

### 3.4 Can the model give a calibrated arrival-time distribution?

**Yes for straight-in traffic. For vectored traffic the interval is honest but too wide, and the
pre-registered veto fires.**

**B0 — the distribution to be calibrated.** (measurement, no gate; `AED` §〇.2;
`E/b0_eta_error_20260907/eta_error.json`, 1404 flights, coverage 1404/1404, 2026-09-07.)
|Δt| = |`final_time_error_s`|.

| Arm | Stratum | n | \|Δt\| p50 | p80 | p90 | signed p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| native32 | pooled | 1404 | 14.9 | 39.9 | 63.8 | −42.3 | +3.5 | +38.3 |
| native32 | straight-in | 904 | 9.9 | 20.3 | 26.5 | −17.7 | +3.6 | +21.7 |
| native32 | vectored | 497 | 39.3 | **72.5** | 87.1 | −74.8 | +2.2 | +67.0 |
| L2.d β=0.01 | vectored | 497 | 39.2 | **68.6** | 82.8 | −70.7 | +3.8 | +68.3 |
| closure `C_pred` | vectored | 497 | 33.7 | **65.8** | 81.5 | −74.3 | −7.1 | +56.6 |

Vectored |Δt| p80 is **65.8–72.5 s** against straight-in **12.0–20.3 s**, a factor of 3.6–5.5, so
per-stratum calibration is necessary rather than a refinement. The signed median is off-centre
(+3.4/+3.5 s for the control arms, −1.4 s for closure), which argues for a quantile head rather
than "point estimate ± δ".

**B1/B2/B3 results.** (gates in `AED` §3.4; results §3.5; `E/b1_quantile_20260907`, **single
seed**, 2026-09-08. `B1_quantile` commit `73d829f`, 164/180 epochs, best 144; `B3_quantile_cta`
commit `77e6d3a`, 129/180, best 109.)

**Gate 1 — both clauses passed against the pre-registered line, but the ADE clause is now "not
demonstrated".** Against the pre-registered reference `native32`, `B1_quantile` is better in every
stratum: ADE 1282 / 420 / 2805 against 1322 / 445 / 2870; FDE p50 860 / 595 / 1935 against
864 / 671 / 1982; paired win 55.6–55.8 %. **The pooled ADE margin is 40 m, well inside the ~125 m
control-path seed line measured later the same day (§2), so it passed the line it was written
against and is not evidence of an improvement.** The MAE clause is unaffected. Duration MAE **23.9 vs 25.9 s** pooled
(straight-in 10.3 vs 13.6; vectored 45.4 vs 45.0), and native32's late bias (signed median +3.5 s)
disappears (+0.1 s). A first reading that used L2.d as the reference and concluded "worse in every
stratum by 68 m" is corrected in the source; L2.d is context only.

**Gate 2 — deployed coverage**, band [0.76, 0.84] at α = 0.2 and [0.45, 0.55] at α = 0.5:

| Arm | α | deployed (cut 1337) | five-cut mean | width p50, s | straight-in / vectored width, s | mirror half |
|---|---|---:|---:|---:|---:|---:|
| B1 | 0.2 | 0.745 | **0.790** | 33.5 | 28 / **142** | 0.839 |
| B1 | 0.5 | 0.450 | **0.486** | 16.5 | 14 / 74 | 0.557 |
| B3 | 0.2 | 0.775 | **0.799** | 38.2 | 32 / **146** | 0.823 |
| B3 | 0.5 | 0.477 | **0.489** | 18.4 | 15 / 76 | 0.548 |

The half cut itself moves the answer. A read-only five-seed probe (`212ec50`; seeds 1337, 2024, 7,
99, 31337) found deployed and mirror coverage strongly anti-correlated, with a cut-to-cut spread of
sd ≈ 0.05 against a binomial standard error of 0.015 and the gap changing sign across seeds; the
deployment cut (1337) is about 1.4 sd the least favourable of the five. **Post-hoc correction,
labelled as such in the source**: gate 2 is read on the five-cut mean with the cut noise beside it,
and on that reading **B1 passes both levels (0.790, 0.486) and B3 passes both (0.799, 0.489)**; on
the single deployment cut, B1 α = 0.2 (0.745) is outside the band. The deployed δ is unchanged.

**Gate 3 — passed.** Among flights whose true duration falls inside the fan (84 % of the cohort),
the truth path's chamfer to the **nearest of the five quantile decodes** beats its chamfer to top-1
on **70.9 %** of flights (107 vs 137 m; straight-in 53 vs 75; vectored 704 vs 949). This is a
readout, not a coverage guarantee: five trajectories are not a distribution over trajectories.

**The pre-registered veto fired.** The vectored 80 % interval median width is **142 / 146 s**
(149 s on the fan readout; five-cut mean 144 / 149 s) against a 120 s veto line; the straight-in
FDE clause is clean (595 vs 594). So the straight-in 80 % interval is about **28–32 s** wide with
compliant coverage and usable by a scheduler, while the vectored **±70 s** is real uncertainty —
the same 962 m of controller intent, in seconds — which calibration reports rather than narrows.
Whether to follow the veto literally and return to the A line is **pending with the user**. One
caution applies to every coverage number here: the calibration split is also the selection split,
so the finite-sample guarantee does not hold as constructed; every surface says "empirically
measured cross-half coverage", and the guarantee-bearing read is pre-registered as **one**
test-split measurement at `freeze-test`.

**`B1_point_matched`, both seeds — half the dissociation survives.** (pre-registered conditional
arm, triggered because B1's ADE change exceeded the then-assumed seed noise;
`E/b1_quantile_20260907/{B1_point_matched, B1_point_matched_s2024}`, readouts
`readout_point_matched.{txt,json}` and `readout_s2024.{txt,json}`; point head at
`final_time_loss_weight` 26.0. Seed 1337: 180/180 epochs, best 175. Seed 2024: 143/180,
early-stopped, best 123. `AED` §3.5.)

| | native32 | `B1_quantile` | point_matched **seed 1337** | point_matched **seed 2024** |
|---|---:|---:|---:|---:|
| ADE pooled / straight-in / vectored | 1322 / 445 / 2870 | 1282 / 420 / 2805 | 1248 / 413 / 2721 | **1373 / 451 / 3004** |
| FDE p50 pooled / straight-in | 864 / 671 | 860 / 595 | 877 / 582 | 937 / 659 |
| duration MAE pooled / straight-in | 25.9 / 13.6 s | **23.9 / 10.3 s** | 25.1 / 13.1 s | 26.2 / 13.9 s |
| paired ADE win vs native32 (median Δ) | — | 55.8 % | 61.4 % (−26 m) | 46.8 % (+9 m) |

**The 26× duration weight is not adopted, and the claim it supported is withdrawn.** The two seeds
land at 1248 and 1373, straddling native32's 1322, and their duration MAE straddles 25.9 in the
same way. There is no evidence that the heavier duration weight improves ADE, so the single-seed
reading "the path gain is the duration term's weight" **is withdrawn**.

**The surviving half is the quantile head's duration accuracy.** `B1_quantile` reaches 23.92
pooled / **10.31 s** straight-in, better than both point-matched seeds (25.08 / 26.19 pooled;
13.06 / 13.91 straight-in) and better than native32. The straight-in margin of about 3 s is well
outside the two-seed spread of 0.85 s; the pooled margin of 1.2–2.3 s sits at the edge of the
1.1 s spread and is therefore weaker.

**The methodological finding matters more than the arm.** The pre-registered 30 m seed-noise line
came from two-seed **state**-path arms (5–22 m). On the **control** path these two seeds differ by
**125 m of pooled ADE** — four times that line — partly through convergence, since one ran the full
180 epochs and the other early-stopped at 143. Every single-seed control-arm ADE difference below
about 125 m in this report is therefore **not demonstrated**, and each is marked where it appears
(§2, §3.4 gate 1, §3.5, §3.6, §3.7). Duration-MAE seed spread is about 1 s.

**`B1b_two_head` — the two heads do not compose, and the B line's deliverable is now fixed.**
(pre-registered `AED` §3.1b, implemented in `b859453` / `3833e8e`; result `AED` §3.5;
`E/b1b_two_head_20260908/{readout,eta_error}.{json,txt}` and `B1b_two_head_calibration/`;
180/180 epochs, best 163, single seed. Its gate 1 ADE clause was **reframed post hoc, stated as
such**, after the 1248 m it was written against did not replicate: not worse than the worse
point-matched seed (1373 m) and not worse than native32 beyond the ~125 m seed line.)

| | native32 | `B1_quantile` | `B1_point_matched` (1337 / 2024) | **`B1b_two_head`** |
|---|---:|---:|---:|---:|
| ADE pooled / straight-in / vectored | 1322 / 445 / 2870 | 1282 / 420 / 2805 | 1248 / 413 / 2721 · 1373 / 451 / 3004 | 1307 / **461** / 2799 |
| FDE p50 pooled / straight-in | 864 / 671 | 860 / 595 | 877 / 582 · 937 / 659 | 899 / **687** |
| chamfer pooled | 224 | 205 | 177 · 211 | **232** |
| duration MAE (the duration flown) pooled / straight-in | 25.9 / 13.6 s | 23.9 / 10.3 s | 25.1 / 13.1 · 26.2 / 13.9 s | 25.6 / 13.4 s |
| **q50 MAE** pooled / straight-in | — | **23.9 / 10.3 s** | — | **24.2 / 10.4 s** |
| deployed coverage α = 0.2 / 0.5 (single cut) | — | 0.745 / 0.450 | — | 0.748 / 0.452 |
| vectored 80 % width | — | 142 s | — | 143 s |

Loss components at the best epoch are `final_time` 0.164 and `duration_quantile` 0.075, so both
heads are carrying load. **(1) The MAE clause passes**: the two-head q50 reads 24.2 / 10.4 s
against `B1_quantile`'s 23.9 / 10.3 s, a difference of 0.3 / 0.1 s — the second head reproduces the
quantile head's duration accuracy in full. **(2) The reframed ADE clause fails**: pooled 1307 is
inside the seed line, but straight-in ADE 461 is the worst of the four arms, straight-in FDE p50
687 trips the standing veto (671), and chamfer 232 is the only value in the B family worse than
native32's 224. The arm inherited the quantile head's timing and lost the point head's path; the
two advantages did not add. **(3) Gate 2 and the width veto are unchanged**: α = 0.2 deployed
coverage 0.748 is the same single-cut near-miss as `B1_quantile`'s 0.745, which the five-cut probe
already showed to be cut noise, and the vectored width is 143 s against the 120 s veto.

**Decision: `B1b_two_head` is not adopted and gets no second seed. The B line's deliverable is
fixed as `B1_quantile`'s quantile head (the ETA distribution, calibrated by B2) plus B3's CTA
conditioned decoding, with the path model chosen separately** — the 26× duration weight that was
the other candidate is withdrawn (above).

### 3.5 Can the prediction be re-issued at any time?

**Yes at the far anchors, and the arrival time now freezes at 8 km. The veto at the training anchor
still fails.**

**A0-fixed.** (gates in `AED` §2.4; results §2.4b; `E/anytime_a0_20260907`, three L−1-trained
checkpoints replayed at seven remaining-path bins, `--min-future-s 60`.) Gate 1 (monotone vectored
ADE p50, read paired between adjacent bins) **passed** for native32 and warm β=0.01, which is
better at every bin (−274 m at 20 km). Gate 2 (farthest bin with vectored ADE p50 < 1.5 km) is
**16 km** for both (1382 / 1301 m). The closure arm is most accurate far out but **degrades sharply
between 8 and 6 km** (vectored ADE 613 → 3990 m), failing gate 1. **Gate 3, the freeze point, was
unreadable** for all three: the duration head's ~125 s floor makes |Δt| p80 *rise* toward the
runway (native32 vectored 64 s at 12 km, 141 s at 4 km).

**A0-random and its failure.** Three random-anchor arms **failed the pre-registered L−1 veto by a
wide margin** — pooled ADE 2949–2990 against 1322 — yet drew **better chamfer at every bin**
(−114 … −524 m) and better ADE inside 8 km. Two mechanisms were found in `history.json` and both
fixed. **(i)** `ReduceLROnPlateau` stepped on the *selection metric*, which scores only L−1, the
anchor a random-anchor model is least specialised for; it stalled after epoch 8–10, so the learning
rate was halved from epoch 20 and reached **9.4e-7 by epoch 60** while the validation objective
kept improving to epoch 60 (1.147 → 0.707). **(ii)** Anchor sampling was uniform in *time*, which
over-weights the near end when pooled over flights: over 1404 flights / 181,906 admissible anchors
/ 200 epochs, **34.3 % of draws under 6 km against 25.1 % of the population; 17.9 % beyond 20 km
against 32.9 %**. A stratum draw was tried first and **rejected by measurement** — it moved
training *toward* the runway (≥ 20 km 16.9 → 4.1 %) — and replaced by a remaining-path-uniform law
(31.0 % under 6 km, 21.0 % beyond 20 km).

**A0.b results.** (pre-registered fixes; `AED` §2.4d; `E/a0_random_20260907/A0b_*`, replay
`E/anytime_a0b_20260908`, single seed.) Both arms ran 180 epochs without early stopping, best
epochs **155 and 173** — both still improving at the cap.

| Arm | best epoch | anchor-grid selection metric | L−1 | 12 km | 8 km | 6 km |
|---|---:|---:|---:|---:|---:|---:|
| `A0_random_hr8_tv1_grid` (09-07) | 8 | 1100 | 2990 | 722 | 390 | 298 |
| `A0b_lr_objective` | 155 | 743 | 1871 | 581 | 303 | 217 |
| `A0b_lr_objective_path_uniform` | 173 | **710** | **1782** | **561** | **283** | **212** |

| Replay (vectored) | `A0b_lr_objective` | `A0b_…_path_uniform` | `native32` (fixed anchor) |
|---|---|---|---|
| Gate 1 (monotone) | passed | passed | passed |
| Gate 2 (farthest bin, ADE p50 < 1500 m) | 16 km (1321 m) | 16 km (1317 m) | 16 km (1382 m) |
| **Gate 3, s_freeze (\|Δt\| p80 < 30 s)** | **8 km (22.0 s)** | **8 km (19.8 s)** | not reached |
| predicted duration p50 (20/16/12/8/6 km) | 235/207/161/113/94 s | 239/207/162/114/91 s | 323/258/204/182/179 s |

The **ranking between the two A0.b arms** (path_uniform better by 20–90 m at each anchor set) is
inside the ~125 m control-path seed line (§2) and is **not demonstrated**. What is demonstrated is
the difference between either A0.b arm and the 09-07 grid arm — 1100–1200 m at L−1 and 140–160 m
at 12 km — and the change in best epoch from 8 to 155 / 173.

**The freeze point was reached for the first time in this project.** The duration head is no longer
on its ~125 s floor: native32 still reports 179 s at 6 km where A0.b reports 91–94 s, tracking the
true remaining time. That follows from the scheduler fix, not from a new head.

**The L−1 veto still fails.** Paired on 1404 flights: native32 1322 / 864 / 224 (ADE / FDE p50 /
chamfer) against `path_uniform` 1782 / 1477 / 229; straight-in 628 / 1315 / **84** (chamfer better
than native32's 109), vectored 3839 / 1981 / 1691. The gap is a third of the previous 2949 but
still far beyond seed noise. A random-anchor arm is **not a replacement at L−1** but is better
across the anchor range it trains on, so the two delivery forms should be kept separate. The A2
choice is **pending with the user**: (a) two models, or (b) mixed-anchor training with L−1 weighted.

### 3.6 How is the final-approach corridor enforced?

**As a bounded output on the state path and an inference-time hook on the control path.
Training-time penalties are vetoed on both, and now vetoed twice on the control path.**

**(a) Bounded output on the state path — adopted, and the programme's only replicated result.**
(pre-registered; `FCR`; `R-CON`; two airports × two seeds, KRDU val 2104 / KSJC 1666.)
`state_position_reference="corridor-bounded"` improved pooled FDE in all four runs
(−91 / −66 / −51 / −63 m) and pooled ADE in all four (−46 / −8 / −18 / −71 m). Corridor violation
rows KRDU 77 → 48 %, KSJC 34 → 21 %; glidepath-window rows 62 → 43 % and 38–46 → 21–23 %; threshold
|xt| p95 KRDU 492 → 305 m, KSJC 237 → 171 m. The veto (a vectored regression on both seeds) did not
fire. The checkpoint is selected on common-grid ADE, so part of the ADE gain is a selection effect;
FDE, violation rate and threshold |xt| p95 took no part in it.

**(b) Training-time penalties — vetoed twice.** On the state path the dual-ascent arm diverged in
4 of 4 runs (ε = 0.05 is unreachable — the baseline is at 0.77 — so the multiplier only rose, 50×
over 74 epochs), and the fixed-λ arm reduced the violation rate at the cost of accuracy (`FCR` §5). On the control
path with the teacher base, λ = 1e-3 improved the endpoint (pooled FDE 1650 → 1549) but damaged the
mid-path (vectored ADE 2858 → 3439, paired improvement 23 %, median +658 m), and λ = 5e-3 reduced
the bank schedule to one shape shared across flights (skill 0.728 → 0.280) (`CPR`). **L1.c re-opened that veto on a
teacher-free base**, because the 09-05 result could be explained by the imitation term, 87 % of the
loss, whose gradient opposed the penalty's through the dynamics. (pre-registered 2026-09-08; `LID` §六 L1.c;
`E/l1c_procedure_20260908`, base `L1b_hr8_tv1_full`, 180 epochs, paired on 1404 flights.)

| Arm | epochs | selection | ADE pooled / vectored | FDE p50 | chamfer | \|xt\| p95 pooled / straight-in | lateral viol. | vert. viol. | bank skill |
|---|---|---:|---|---:|---:|---|---:|---:|---:|
| base hr8+TV | 116, early stop | 1332 | 1332 / 2872 | 848 | 161 | 2601 / 1282 | 83.3 % | 55.9 % | 0.713 |
| λ = 2e-4 | 180, no stop | 1343 | 1344 / 2900 | 860 | 165 | 2206 / 1291 | 82.4 % | 51.2 % | 0.715 |
| λ = 1e-3 | 140, early stop | 1471 | 1471 / 3059 | 899 | 165 | 2111 / 1367 | 79.1 % | 52.9 % | 0.719 |
| **base + soft barrier hook** (predict-only) | — | — | **1296 / 2857** | **721** | **87** | 1977 / **55** | **43.8 %** | 55.9 % | 0.646 |
| λ = 2e-4 + hook (predict-only) | — | — | 1307 / 2884 | 750 | 85 | **1281** / 56 | 45.5 % | 51.2 % | 0.634 |

Gates (violation rate down ≥ 5 points **and** vectored ADE not worse by more than 30 m **and** bank
skill ≥ 0.70; veto at vectored ADE worse by more than 100 m). **λ = 2e-4 failed** — 0.9 points for
+28 m. Its ADE differences (+11 pooled, +28 vectored) are inside the ~125 m seed line and are
**not demonstrated**, so this arm fails on the violation-rate clause alone — a 4.1-point shortfall
against the 5 points the gate required, which is not a seed question. **λ = 1e-3 failed and
triggered the veto** — 4.2 points for +187 m, and +187 m is above the seed line. **The 09-05 veto
reproduces on a teacher-free base; the training-time LPV corridor term is settled as "not used".**
The mechanism is unchanged by removing the teacher: the penalty's gradient reaches the controls
through the dynamics and damages the mid-path, and a dose small enough not to damage it does not
reduce the violation rate.

**(c) The inference-time soft barrier hook — adopted.** On the same base with no retraining the
lateral violation rate falls **83.3 → 43.8 %** — 39.5 points, eight times what the gate asked of
the training term — while ADE, FDE and chamfer improve at once: 1332 → 1296, FDE p50 848 → 721,
chamfer 161 → 87, straight-in threshold |xt| p95 **1282 → 55 m**. The cost is bank skill
0.713 → 0.646, because the hook overwrites the network's commanded bank inside the gated segments.
An independent KRDU + KSJC campaign reaches the same verdict from a different base (`CHR`): pooled
FDE mean 1650 → 1449 m, pooled ADE 1333 → 1278, straight-in threshold |xt| p95 1821 → 70 m, 84 % of
flights better and **none worse by more than 1 km**. Training *through* a hook lost to
inference-only in all six arms tried.

**(d) The penalty's only stable gain is vertical.** Vertical violation rows fall 55.9 → 51.2 %
(vectored 82.9 → 66.6 %) at λ = 2e-4, and the hook does not touch the vertical axis by
construction. The two do not compose cleanly: stacked, they give the best pooled |xt| p95 (1281 m)
and the worst bank skill (0.634). A candidate follow-up — lateral by hook, vertical by a
training-time term only — is named but not queued.

### 3.7 Is the teacher necessary, and can it be replaced?

**Necessary — confirmed. Replaceable — no, and the fitted teacher does not clear its gate either.**

L1's teacher-free dense arms are 1.9× worse on ADE (2515 vs 1322) and 2.7× worse on FDE p50, and
the bank wiggle returns (§3.1). Position is derivative order 0, velocity order 1, bank order 2;
with no imitation term nothing in the loss names the bank.

**L1.b — two terms passed through the rollout itself** (a heading-rate loss against the smoothed
observed ψ̇, and a bank total-variation penalty). (pre-registered; `LID` §六 L1.b; screen
`E/l1b_supervision_20260907` at 60 epochs, confirmation `E/l1b_full_20260907` at 180, paired
against `L1_native32` at 180.)

| Arm | epochs | bank skill (gate 0.726) | straight-in reference bank RMS | ADE pooled / s-i / vectored | FDE p50 pooled / s-i / vectored | chamfer |
|---|---|---:|---:|---|---|---:|
| `native32` (teacher) | 180 | **0.726** | 0.34° | 1322 / 445 / 2870 | 864 / 671 / 1982 | 224 |
| hr8 + TV | 116, early stop | 0.713 | 0.12° | 1332 / 461 / 2872 | **848 / 632 / 1722** | 161 |
| hr16 + TV | 142, early stop | 0.677 | 0.12° | 1358 / 448 / 2968 | 921 / 658 / 1801 | 147 |
| hr16 | 180, no stop | 0.706 | 0.15° | 1344 / **442** / 2940 | 941 / 672 / 2329 | **146** |

**No arm passed.** The gate is bank skill, which is what decides this; the accompanying ADE
differences (hr8+TV +10 m pooled against native32) are inside the ~125 m seed line and are **not
demonstrated**. Dose is not a ramp: hr16 is worse than hr8, so the 1 → 8 trend (0.516 → 0.665 in
the screen) does not continue, and the total-variation term's sign flips with dose (+0.046 at hr8,
−0.029 at hr16). All three arms produce bank geometry straighter than the real aircraft (RMS
0.12–0.15° against the observed 0.41°) with vectored length ratio 0.87–0.88; the chamfer and
vectored-FDE gains and the bank-skill loss come from the same behaviour, a shorter and smoother
path to the correct endpoint. **Decision rule lands on "none passed": the teacher stays.**
`hr8 + TV` was nonetheless kept as the base for L1.c and A0.b, being the only teacher-free
supervision compatible with random anchors; this **deviates from L1.c's pre-registered wording**
("if none pass, the base returns to native32"), and the deviation is declared in the source.

**L5.a — the fitted teacher.** The deployed teacher is the inverse dynamics of the truth, and L0
measured how poor it is: flown open loop it lands **2.5–7.8 km** from the truth (N=4 7850, N=8
6381, N=16 4095, N=32 2537 m), while a fitted table of the same width lands within **88–433 m**.
The **fit is done** (`E/l5_fitted_teacher_20260907/basis_fit.json`, schema
`ts-basis-fit-v2-teacher`, N = 32, uniform durations, anchor index 59, network initialisation from
`L1_native32`, 400 steps × batch 1024, **wall time 2951 s = 49.2 min**, train 6851 + val 1404 =
8255 flights):

| Split | fitADE p50 | p90 | p99 | mean | seedADE p50 | best step p50 |
|---|---:|---:|---:|---:|---:|---:|
| val (1404) | **105.9 m** | 737.0 | 1320.4 | 313.5 | 542.4 | **400 (the cap)** |
| train (6851) | 107.6 m | 743.9 | 1602.1 | 322.6 | 564.4 | **400 (the cap)** |

The fit improves on its own seed by a factor of five (542 → 106 m), but the **median best step is
pinned at the 400-step cap on both splits and still falling**, so the bound is not tight and a weak
arm result must first be suspected of under-optimisation — which is why the L5.a arms were
re-prioritised to the end of the queue. Both have now trained, predicted and been read out
(`E/l5_fitted_arms_20260907/{readout.json, readout.txt, readout_bank.txt}`, 2026-09-08, paired on
1404 flights).

| Arm | dose (× position) | ADE pooled / s-i / vectored | FDE p50 pooled / s-i / vec | chamfer p50 | bank skill (gate 0.70) | common-profile share (observed 1.8 %) |
|---|---:|---|---|---:|---:|---:|
| `L1_native32` (reference) | 47.06 (inverse dynamics) | 1322 / 445 / 2870 | 864 / 671 / 1982 | 224 | 0.726 | 2.6 % |
| `L5_fitted64` | 47.06 (fitted) | **1280 / 438 / 2767** | 907 / 677 / **1964** | **200** | **0.707** | **2.3 %** |
| `L5_fitted16` | 11.76 (fitted) | 1471 / 508 / 3177 | 989 / 788 / 1879 | 274 | 0.554 | 19.5 % |
| `L2d_warm_beta0p01` (context) | 47.06 | 1214 / 402 / 2643 | 827 / 594 / 1851 | 165 | — | — |

Against the gate ("top-1 ADE **and** FDE p50 not worse than native32 in every stratum; bank skill
≥ 0.70; veto if straight-in FDE p50 degrades beyond seed noise"): **`L5_fitted64` fails on the FDE
clause.** ADE passes in all three strata (−42 / −7 / −103 m; paired wins 57.0 / 52.7 / 64.8 %,
vectored median Δ −86 m) and chamfer improves in all three (paired win 63.2 % pooled, 75.1 %
vectored); bank skill **passes** at 0.707 with the lowest common-profile share of any arm (2.3 %);
but FDE p50 is worse pooled (907 vs 864) and straight-in (677 vs 671). The veto does not fire (+6 m
is inside seed noise). **The pooled −42 m of ADE is itself inside the ~125 m seed line and is not
demonstrated**, so the arm neither passes its gate nor shows a measured accuracy gain. **`L5_fitted16` degrades sharply**: ADE 1471, bank skill 0.554, common-profile share
19.5 % against the observed 1.8 % — the bank-wiggle signature. **The pre-registered risk fired**:
the fit minimises position only, so the table carries wiggle, and the correction belongs at the
fitting end (a smoothness prior, more than 400 steps), not at the training end.
`L2d_warm_beta0p01` at 1214 m still beats both fitted arms.

With L1.b this closes the teacher line: **the teacher is necessary, is not replaceable by the two
rollout-side terms tried, and a 25× better teacher (106 m against 2537 m) gains about 40 m of pooled
ADE while costing about 40 m of endpoint error.**

---

## 4. Overall conclusion

This is the author's position, stated plainly.

**The model reproduces the approach path, and the residual is timing.** Thirty-two control segments
match the 64-segment baseline with 96 free numbers instead of 257; on straight-in flights
cross-track RMS is 112 m and vertical RMS 78 m, with 89–95 % of the squared horizontal error
along-track. The residual concentrates in the last 5 km of a straight-in approach and across the
whole of a vectored one, and it is scatter, not bias: the true total duration moves the last-band
bias from −191 m to +8 m and the RMS only 333 → 316 m. What varies is *when* the aircraft
decelerates and *when* it turns.

**The aircraft's own history does not contain enough of that timing.** Neighbouring traffic
contributes exactly nothing, because the lead aircraft's remaining time is as undetermined as the
own-ship's — both are outputs of the same sequencing decision. The latent line's seven failures,
however, were a **loss-scaling failure**: with the position term rescaled 100×, shuffling z costs
928 m pooled and 2444 m vectored, the trained prior beats an N(0, I) control for the first time,
and vectored FDE p50 improves 1982 → 1519 m. It still does not recover the 962 m — vectored top-1
is 2660 m against a 1235 m target. The honest statement is therefore narrower than "the intent is
unavailable": **what the aircraft carries improves the endpoint; it does not fix the timing.**

**Supplying the time resolves most of it.** Given the true arrival time, pooled ADE falls
1214 → 841 m and vectored 2643 → 1596 m on 87.3 % of flights, and the duration error equals a
requested offset exactly at seven points with monotone geometry on both sides — a real control
response, not a fitted output. **The model must not guess the intent; the scheduler must supply it,
as a time** — and the latent, now that it works, is the right instrument for the *distribution*
around that time rather than for the point estimate.

**The calibrated ETA is deliverable for straight-in traffic and honest for vectored traffic, and
its form is now settled.** The straight-in 80 % interval is about 28–32 s wide with coverage inside
the band and straight-in duration MAE falls 13.6 → 10.3 s; vectored reports **±70 s**, tripping the
120 s veto. That width is not a calibration defect but the 962 m of controller intent expressed in
seconds, and the correct response is for the scheduler to assign the time. **The delivered form is
`B1_quantile`'s quantile head, calibrated by B2, decoded per quantile by B3, with the path model
chosen separately.** Both attempts to fold the arrival-time gain into the path model failed: the
26× duration weight did not survive its second seed, and `B1b_two_head` reproduced the q50 accuracy
exactly (24.2 / 10.4 s against 23.9 / 10.3 s) while giving up the easy stratum (straight-in ADE
461, FDE p50 687, chamfer 232 — the only B-family chamfer worse than native32). Timing and path are
better delivered by two components than by one head.

**The model cannot absorb delay legitimately.** Asked to arrive 60 or 90 s later it produces a
reference with a 0 % fully-flyable share. The hook fixes the lateral symptom completely
(3288 → 5 m of cross-track at the threshold) and the delay is obeyed exactly in time, but the hard
violations are stall violations, 77–94 % of them at remaining path ≥ 20 km: the model is already
too slow in the outer segment at zero offset and the delay is placed in the same segment. Two things are
missing, neither a training trick — a **speed floor inside the rollout dynamics**
(V ≥ V_stall(n) × margin, in the equations rather than only in the metric) and an **upstream
path-stretch degree of freedom** so that delay can be absorbed by a modelled degree of freedom.

**The inference-time hook is the safety layer; the training-time penalty is settled.** The penalty
has been vetoed twice on the control path, with and without the teacher, so the explanation is the
mechanism and not the base. The hook cuts the lateral violation rate by 39.5 points at prediction
time with no retraining while improving ADE, FDE and chamfer at once, at a cost of bank skill
(0.713 → 0.646) that must be quoted with it.

**The heavier duration weight is not adopted, and that is the day's methodological lesson.** The
second seed overturned it: `B1_point_matched` reads 1248 m at seed 1337 and **1373 m at seed 2024**,
straddling native32's 1322 m, so the 74 m gain and the reading that "the path gain is the duration
term's weight" are **withdrawn**. What survives is the quantile head's duration accuracy:
`B1_quantile` at 23.92 pooled / **10.31 s** straight-in beats both point-matched seeds
(25.08 / 26.19 and 13.06 / 13.91) and native32, and the 3 s straight-in margin is well outside the
0.85 s two-seed spread. The lesson generalises beyond this arm: **the pre-registered 30 m seed line
came from state-path arms, while the control path's measured two-seed spread is 125 m of pooled
ADE**, so several single-seed differences in §3 are now labelled "not demonstrated", and every
future ADE gate on a control arm needs a second seed.

**Anytime prediction reaches its freeze point:** after the scheduler fix and the remaining-path-uniform anchor law the
arrival-time error first drops below 30 s at **8 km** (|Δt| p80 22.0 / 19.8 s) and the duration head
leaves its 125 s floor, though the random-anchor arm is still worse at the training anchor
(1782 vs 1322), so the two are different delivery forms rather than substitutes.

---

## 5. Next steps

Items 1–8 are proposals for decision, except item 4, which has been run and decided; item 9 is
closed. Each is described in one or two sentences, with its gate, veto and cost in the table that
follows. **Only one experiment in this list is still pending: L2.g, item 5.**

**1. Adopt two recipe changes; the third is now refused.** (a) `final_time_loss_weight` 26 into the
mainline recipe is **NOT adopted** — the second seed (1373 m against seed 1337's 1248 m, around
native32's 1322 m) did not reproduce the 74 m, so the change has no measured benefit. (b) `lr_plateau_metric=objective` plus
`random_train_anchor_sampling=remaining-path-uniform` as standard for every random-anchor arm.
(c) `predict --command-hook barrier --hook-saturation soft` as the default inference layer for
every delivered control reference, with its bank-skill cost published alongside.

**2. Delay absorption: a stall-margin speed floor in the rollout, plus an upstream path-stretch
degree of freedom.** The rollout would enforce V ≥ V_stall(n) × margin inside the dynamics rather
than reporting the violation afterwards, and the reference would gain a way to lengthen the path
before joining final. **This is the highest-value open item**: a scheduler requests delay far more
often than an early arrival.

**3. Anytime delivery: two models versus mixed-anchor training.** Option (a) delivers the
fixed-anchor arm at L−1 and the random-anchor arm later; option (b) trains one mixed-anchor model
(L−1 weighted, remaining-path-uniform elsewhere), the A2 arm. Both A0.b arms were still improving
at 180 epochs, so a longer budget is a confound to control.

**4. The B line's deliverable — done, and fixed.** `B1b_two_head` ran and was **declined**: the
MAE clause passed (q50 24.2 / 10.4 s against 23.9 / 10.3 s) but the post-hoc ADE clause failed
(straight-in ADE 461, FDE p50 687 against the standing 671 veto, chamfer 232), so the two heads do
not compose. No second seed. **The deliverable is fixed as `B1_quantile`'s quantile head + B2
calibration + B3 CTA decoding, with the path model chosen separately.** Nothing remains to run on
this step; what remains is the vectored width veto (§3.4), which is a scheduler-interface decision
rather than an experiment.

**5. L2.g — the latent line re-opened once, read as a distribution. This is the only experiment
still pending.** The units test passed three of four gates and was budget-limited at 180/180
epochs, so its numbers are lower bounds. L2.g (pre-registered `LID` §六 L2 末;
`l2g_latent_distribution_arms.json`; campaign `l2g_latent_distribution_20260908`; queued last)
keeps the units recipe — `position_loss_scale_m` 1 km, warm posterior, β 0.01, 8 dimensions — and
runs **300 epochs with early stopping off at two seeds** (1337 and 2024, same split seed), with the
units test's prediction settings (6 prior samples, 6 N(0, I) controls, shuffled z). The 26×
duration weight is **not** carried, having been withdrawn with item 1(a). Every number is reported
as a two-seed spread, and a difference inside that spread is not a finding. It overlaps the C line
— reading a sample fan as a trajectory distribution is the same objective — so the two should be
scoped together.

**6. Replicate the CTA and B results on KSJC before any claim generalises.** Everything in §3.3 and
§3.4 is KRDU and single-seed, and a per-airport ADE without its route mix is not a comparison:
reweighted to the pooled stratum mix KSJC moves 483 → 1526 m.

**7. The end-to-end scheduler demonstration — the thesis deliverable.** Assign CTAs to several
aircraft arriving at one runway, generate a reference for each from the same checkpoint, and check
that the set is separated. The model, the CTA conditioning, the hook and the evaluation contract
all exist; missing are the multi-aircraft driver and the separation metric. This turns the result
into the thesis claim, and it is gated on item 2: requesting delay is the scheduler's main use of
the reference.

| # | Gate | Veto | Cost |
|---|---|---|---|
| 1(a) | **failed**: `B1_point_matched_s2024` read 1373 m against seed 1337's 1248 m, straddling native32's 1322 m — not adopted | — | already spent (one training arm) |
| 1(b) | already met (A0.b, both arms, every anchor set better than the 09-07 arm) | — | none; a default change |
| 1(c) | already met on three independent bases (`simple-v3` in `CHR`, teacher-free `hr8+TV` in L1.c, the CTA base in L3.c) | — | none; a predict-time flag |
| 2 | the L3.c protocol re-run unchanged: duration error still equals the offset exactly (gate 2), threshold \|xt\| p95 within 1.5× the offset-0 hooked arm (gate 1), fully-flyable share at +60 s ≥ 8.3 % (gate 3, the clause that failed) | pooled ADE at offset 0 degrades by more than 100 m against 840 m | dynamics change (code + review) + four predict-only arms, ≈ 20 min GPU |
| 3 | pooled ADE at L−1 not worse than 1322 m beyond seed noise, anchor-grid selection metric, and no regression of the anytime gates (monotone; gate 2 at 16 km; s_freeze at 8 km) | — | one training arm ≈ 2–3 h GPU for (b); (a) trains nothing but doubles what must be shipped |
| 4 | **run and decided**: the MAE clause passed (q50 24.2 / 10.4 s within 1 s of 23.9 / 10.3 s); the ADE clause, reframed post hoc against the worse point-matched seed (1373 m) and the ~125 m seed line, **failed** on straight-in (ADE 461, FDE p50 687 against the standing 671 veto, chamfer 232) | — (declined before the veto was reached) | already spent (one training arm + calibration + prediction) |
| 5 (L2.g) | **all four must hold at BOTH seeds**: (1) shuffled-z ΔADE > 200 m; (2) minADE₆ below the N(0, I) control, pooled and vectored; (3) FDE p50 better than native32's 864 m by more than 100 m; (4) pooled top-1 ADE not worse than native32's 1322 m beyond the ~125 m seed line | either seed fails gate 1 — z is inert again — and the latent line closes permanently | 2 × ≈ 100 min GPU plus prediction |
| 6 | the sign and rough magnitude of both headline effects reproduce: a CTA gain concentrated in the vectored stratum, and a straight-in 80 % interval under ≈ 40 s with deployed coverage in the band | — | ≈ three training arms + calibration, ≈ 8–10 h GPU |
| 7 | for a realistic arrival stream the references satisfy the required in-trail separation at the threshold and along final, and every one passes the corridor and flyability checks the single-flight arms are judged by | — | integration only: a multi-aircraft driver and a separation metric |
| 8 | a control-arm ADE gate is decided by two seeds with early stopping off, or the difference is reported as "not demonstrated" when inside ~125 m | — | one extra training arm per gated comparison, ≈ 2–3 h GPU |

**8. Put every future control-arm ADE gate on two seeds.** The 30 m line this programme
pre-registered was imported from state-path arms and is four times too small for the control path.
From now on an ADE gate on a control arm is decided either by **two seeds with early stopping
turned off** — as A0.b's `p180` arm did, so that a convergence difference cannot be read as a seed
difference — or by reading the single-seed difference against the **~125 m** line and reporting it
as "not demonstrated" when it falls inside. The same applies to any published table: the
single-seed ADE differences already in §3 are labelled, not deleted. Cost: one extra training arm
per gated comparison (about 2–3 h GPU at KRDU scale), which is what the withdrawn 74 m result cost
to discover after the fact.

**9. Closed lines — do not reopen without new evidence.** **Training-time LPV corridor penalty**:
vetoed twice on the control path and once on the state path; the one live candidate is the
vertical-only variant, named but not queued. **Teacher replacement (L1.b) and teacher improvement
(L5.a)**: no L1.b arm passed, `L5_fitted64` fails its gate on the FDE clause and `L5_fitted16`
degrades sharply — no further teacher *training* arms, and if the fitted teacher is revisited the work is
at the **fitting** end (a smoothness prior, more than the 400-step cap), because the pre-registered
risk that a position-only fit produces bank wiggle is exactly what fired. **Wind**: measured and
excluded (R² 0.10 on the straight-in speed residual, 0.01 on the endpoint). **Deceleration-point
oracle (L3.b)**: drafted with a ceiling of straight-in ADE 400 → 330 m and a 30 m gate (`SIR` §四),
**declined** — it would measure the speed-instruction channel that §3.1 already attributes to
controller intent, and the CTA line answers the same question in the form the scheduler uses.
**Scene encoder (L4)**: not built; the pre-measurement gate failed with zero increment.

---

## 6. Incidents that changed the tools

Four operational failures changed instruments rather than results. Each cost measurements.

**(1) Roster provenance: a dataset's identity is its eligible set, not the roster file's bytes.**
On 2026-09-07 at 08:14 the five canonical observed evaluation reports were regenerated (v6 → v9)
and the five `lateral_pass_eligibility.json` rosters refreshed against them. The eligible sets came
out **byte-for-byte identical** (KRDU 14,378 keys) but the roster *files* moved, because a roster
embeds the upstream provenance it was joined against (`sources.evaluation_report_sha256`), and
`data_provenance` compared exactly those bytes. `require_matching_data_provenance` therefore
**refused every checkpoint trained before 08:55** — `predict`, `evaluate-fit`, every `run_ts_*`
replay runner, and the publisher through its own second copy of the comparison. The arm lost to it was
`L1b_hr16_tv1_full`, which has a trained checkpoint and no prediction directory. The
fix makes the compared identity the **set** (`ts-arrival-data-v4-eligible-set`,
`eligible_set_sha256` over the sorted keys, one function shared by the splits audit, the pipeline
and the publisher); the roster's byte facts and counts stay auditable and outside every comparison,
and a stored v3 fingerprint stays usable because it carries the eligible set itself. Verified by
sweeping all 200 on-disk checkpoints: 65 → 66 accepted, exactly one newly accepted (the 08:55 arm),
zero newly refused. A related bug — provenance rebuilt without the roster, reporting 14,435
candidates against a checkpoint's 14,378 — was fixed in `e8df12f`, having blocked the L5.a fit at
startup.

**(2) The dirty-tree refusal.** A formal campaign (`--campaign-id` / `--experiment-id`) refuses to
start on a dirty working tree, and the check fires only at the `train` step. Two campaigns were affected:
the L1 campaign had its tree dirtied mid-run by a parallel commit (`L1R` §五), and
`B3_quantile_cta` had to be restarted after a documentation commit in the main tree, which is why
`B1_quantile` and `B3_quantile_cta` carry different commits (`73d829f`, `77e6d3a`) despite
identical training paths (`AED` §3.5). The rule: commit before starting a campaign, stage explicit
paths, keep the tree clean while it runs, and list a nested git worktree in `.git/info/exclude`
before creating it beside a running campaign.

**(3) Runner dependency deferral.** `run_ts_frame_ablation.py` validated every arm's inputs at
*plan* time, so a predict-only arm reading a checkpoint an *earlier arm of the same campaign*
writes was rejected before the campaign started — exactly the pattern L1.c and L3.c need. Fixed in
`56a0d3b`: the check moved from plan time to the step and now refuses by naming the arm that was
supposed to produce the missing checkpoint. A smaller instance: the A0.b anytime replay died on the
legacy v3 fingerprint of the `_grid` checkpoint until `provenance_manifest_digests` learned to read
it (`b4d3d96`).

**(4) The cost of a 13-decode prediction.** A latent arm's prediction step writes thirteen full
prediction directories — six prior samples, six N(0, I) controls, one shuffled-z diagnostic —
beside the top-1 records. Measured on disk 2026-09-08,
`E/l2_units_test_20260908/L2z_units_pred_val` is **3.0 GB** against **222 MB** for the
single-decode `E/l1_lowdim_20260907/L1_native32_pred_val`, a factor of 13.5. Free disk space is
already the binding constraint on this machine: the root filesystem hit 100 % on 2026-09-06 and
29.7 GB had to be reclaimed with no research artifact touched, and the outstanding optimizer
re-solve is estimated at 12.3 GiB (8.1 GiB at `--rollout-dt 1.0`) with the runner refusing to start
if its estimate does not fit. Latent decode counts and multi-quantile fans are a storage decision
as much as a GPU one.

---

## 7. Sources

**Design documents** (`4dTrajectory/ts_transformer/docs/`; status table is §〇 of each).
**`LID`** = `2026-09-07_latent_intent_design.zh.md`: §〇 status and error budget; §一–§三
positioning and contracts; §六 L0 / L1 / L1.b / L1.c / L2 including "L2 量纲测试结果" / L3 / L3.c /
L4 / L5.a; §七 vetoes; §八 reading conventions. **`AED`** =
`2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`: §〇 status and reference numbers;
§〇.2 B0; §〇.3 commands; §〇.4 review corrections; §2.4 A0 gates, §2.4b A0, §2.4c the second round,
§2.4d A0.b; §三 3.1 B1, 3.1b B1.b, 3.2 B2, 3.3 B3, 3.4 gates, 3.5 results and `B1_point_matched`.

**Results documents** (same directory): **`L1R`** `2026-09-07_l1_lowdim_results.zh.md`; **`SIR`**
`2026-09-08_straight_in_residual_readout.zh.md`; **`WIN`** `2026-09-08_wind_residual_readout.zh.md`;
**`FCR`** `2026-09-05_final_constraint_results.zh.md`; **`CPR`**
`2026-09-05_control_penalty_results.zh.md`; **`CHR`** `2026-09-06_control_hooks_results.zh.md`.
Also cited: `2026-09-07_l0_control_basis_results.zh.md`, `2026-09-06_closure_p1c_results.zh.md`,
`2026-09-05_scene_phase0_results.zh.md`, `2026-09-03_airport_frame_ablation_results.md`.

**Sibling reports** (`4dTrajectory/ts_transformer/docs/reports/`), read for consistency: **`R-CON`**
`2026-09-08_final_approach_constraints_report.md` (§3.6 agrees with it and adds L1.c); **`R-FRAME`**
`2026-09-08_runway_frame_ablation_report.md` (the seed-noise floor in §2 comes from its "The seed
noise floor"); **`R-ANY`** `2026-09-08_anytime_prediction_report.md` (§3.5 agrees with it and adds
the A0.b replay verdicts); **`R-STAGE`** `2026-09-08_programme_stage_report.md` — **superseded by
this report**: written before L1.c, L3.c, the B1–B3 results, `B1_point_matched`, the A0.b replay,
the L2 units test and the L5.a arms existed. Its §3.4, §3.5, §3.9 and §7 remain accurate; its
§3.1b, §3.7 and §3.8 status lines ("untrained", "not run") are out of date, and its §3.3 conclusion
that the latent is inert is superseded by §3.2 here.

**Artifacts** (`E/` = `4dTrajectory/outputs/KRDU/experiments/`; KRDU val, seed 1337 unless stated):

| Path | Arms read | Split / n | Date |
|---|---|---|---|
| `E/closure_p1c_20260905/readout_geometry.{txt,json}` | `C_pred` | val 1404 (vectored 497) | 2026-09-05 |
| `E/l0_control_basis_20260907/` | N ∈ {4,8,16,32,64} × {uniform, free} | val 1404 | 2026-09-07 |
| `E/l1_lowdim_20260907/{readout.json,readout_bank.txt}` | `L1_native32`, `L1_dense32`, `L1_dense64` | val 1404 | 2026-09-07 |
| `E/l1_lowdim_20260907/wind_readout.json` | `L1_native32`, `L2d_warm_beta0p01` | val 1305 with a report (839 straight-in) | 2026-09-08 |
| `E/l1_lowdim_20260907/straight_in_residual.json` | `L1_native32`, `L3_cta` | val 904 straight-in | 2026-09-08 |
| `E/l1b_full_20260907/{readout.{json,txt},readout_bank.txt}` | hr8+TV, hr16+TV, hr16 | val 1404 | 2026-09-08 |
| `E/l1c_procedure_20260908/{readout.{json,txt},readout_bank.txt}` | base, λ 2e-4, λ 1e-3, two hook arms | val 1404 | 2026-09-08 |
| `E/l2_warm_posterior_20260907/L2d_warm_beta0p01_pred_val/summary.json` | `L2d_warm_beta0p01` + `modes/`, `random/`, `shuffled/` | val 1404 | 2026-09-07 |
| `E/l2f_mean_information_20260907/` incl. `probe/` | `L2f_anneal`, `L2f_aux_T` | val 1404 | 2026-09-08 |
| `E/l2_units_test_20260908/L2z_units{,_pred_val}` | `L2z_units` + 13 decodes | val 1404 | 2026-09-08 |
| `E/l3_cta_20260907/` + `.log` | `L3_cta` (`cta=given`, oracle) | val 1404 | 2026-09-07 |
| `E/l3_cta_counterfactual_20260907/readout.{txt,json}` | seven CTA offsets | val 1214 (seven-way intersection) | 2026-09-08 |
| `E/l3c_delay_corridor_20260908/{readout.{txt,json},straight_in_residual.txt}` | four hooked offsets | val 1402 paired | 2026-09-08 |
| `E/l4_scene_explainability_20260907/` | scene pre-measurement | Phase 0 cohort 14,418 | 2026-09-07 |
| `E/l5_fitted_teacher_20260907/basis_fit.json` | the fit (no training) | train 6851 + val 1404 | 2026-09-07 |
| `E/l5_fitted_arms_20260907/L5_fitted{64,16}{,_pred_val}` | `L5_fitted64`, `L5_fitted16` | val 1404 | 2026-09-08 |
| `E/anytime_a0_20260907/anytime_curve.{txt,json}` | native32, warm β=0.01, `C_pred` | val 1404 × 7 bins | 2026-09-07 |
| `E/a0_random_20260907/{readout.json,readout_rerun.json,readout_a0b.{txt,json}}` | four random-anchor arms | val 1404 | 2026-09-07/08 |
| `E/anytime_a0b_20260908/anytime_curve.{txt,json}` | `A0b_lr_objective`, `…_path_uniform`, native32 | val 1404 × 7 bins | 2026-09-08 |
| `E/b0_eta_error_20260907/eta_error.json` | native32, L2.d, `C_pred` | val 1404 (coverage 1404/1404) | 2026-09-07 |
| `E/b1_quantile_20260907/{readout_vs_native32,eta_error_vs_native32,quantile_fan,readout_point_matched,eta_error_point_matched}.{txt,json}` | `B1_quantile`, `B3_quantile_cta`, `B1_point_matched` (seed 1337) | val 1404 | 2026-09-08 |
| `E/b1_quantile_20260907/readout_s2024.{txt,json}` | `B1_point_matched_s2024` (seed 2024, 143/180, best 123) | val 1404 | 2026-09-08 |
| `E/b1_quantile_20260907/B*_calibration/eta_calibration.{txt,json}` | B1, B3 conformal tables | val halves, ≈700 each | 2026-09-08 |
| `E/b1b_two_head_20260908/{readout,eta_error}.{txt,json}`, `B1b_two_head_calibration/` | `B1b_two_head` (180/180, best 163, single seed) | val 1404 | 2026-09-08 |

**Two results landed during writing**, and their numbers come from the readouts, not from
derivation: the L2 units test (`LID` §六 "L2 量纲测试结果", commit `91c0f2f`; artifacts
`E/l2_units_test_20260908/{readout.json, latent_readout_L2z_units.json, probe/latent_probe.{txt,json}}`,
2026-09-08 12:23–12:25) and the two L5.a arms
(`E/l5_fitted_arms_20260907/{readout.json, readout.txt, readout_bank.txt}`, 2026-09-08 12:09–12:10).
Both changed a conclusion here: §3.2 now reads the latent failure as a units failure with z used
for the first time, and §3.7 records `L5_fitted64` as a gate failure on the FDE clause with bank
skill 0.707 recorded rather than missing.

**A third result landed after the first commit of this report** and changed §2, §3.4, §4 and §5:
`B1_point_matched_s2024` (`AED` §3.5, new block; `ts_transformer/CLAUDE.md` first bullet; commit
`e424ecd`; artifacts `E/b1_quantile_20260907/readout_s2024.{txt,json}`). It withdrew the 26×
duration-weight gain and replaced the 30 m seed-noise line with a measured ~125 m one on the
control path.

**Two more results landed after the seed-noise revision**: `B1b_two_head` (`AED` §3.5, "B1.b
结果"; commit `2d33f64`; artifacts `E/b1b_two_head_20260908/{readout,eta_error}.{json,txt}` and
`B1b_two_head_calibration/`), which fixed the B line's deliverable in §3.4, §4 and §5 item 4; and
the L2.g pre-registration (`LID` §六 L2 末; commit `bc3f2f7`;
`docs/experiments/l2g_latent_distribution_arms.json`).

**One experiment is still pending, and only one**: **L2.g** — two seeds of the units recipe at 300
epochs with early stopping off, campaign `l2g_latent_distribution_20260908`, queued last. Its gates
and veto are in §5 item 5. Nothing else in this report is waiting on a run.

**Pre-registered arm files** (`4dTrajectory/ts_transformer/docs/experiments/`):
`l1_lowdim_arms.json`, `l1b_full_arms.json`, `l1c_procedure_arms.json`,
`l2f_mean_information_arms.json`, `l2_units_test_arms.json`, `l3_cta_arms.json`,
`l3_cta_counterfactual_arms.json`, `l3c_delay_corridor_arms.json`, `a0_random_arms.json`,
`b1_quantile_arms.json`, `b1b_two_head_arms.json`, `l2g_latent_distribution_arms.json`,
`l5_fitted_teacher_arms.json`.

**Standing rules and hazards.** `4dTrajectory/ts_transformer/CLAUDE.md` — the config-axis table and
"How to read results here". `docs/CHANGELOG.md` entries 2026-09-03 → 2026-09-08.
`docs/open-items.md`. Root `CLAUDE.md` Open Items.

**Standing caveats for almost every number above.** Everything except the 09-03 state-path work and
the bounded-output arms is **single-seed (1337)**, and everything is **KRDU only**. Oracle arms —
`cta=given`, `intent=truth-…`, `z=posterior` — read the future and may never be quoted as
prediction accuracy; the naming discipline exists so they cannot be mistaken for one. The **test
split has not been touched**, and the one guarantee-bearing coverage measurement is reserved for
the `freeze-test` stage.
