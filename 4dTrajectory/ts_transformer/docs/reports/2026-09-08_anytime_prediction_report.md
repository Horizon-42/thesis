# Anytime prediction: how a forecast improves as the aircraft flies, and why the random-anchor arm was frozen by its own scheduler

*Experiments of 2026-09-07 and the early hours of 2026-09-08. Written 2026-09-08. All
numbers are from the KRDU validation split; the outer-test split has never been read.*

## Abstract

Every evaluation in this package makes exactly one forecast per flight, from one anchor:
L−1, the instant 120 seconds after the 25 km arrival slice begins. That is the moment the
aircraft's own history knows least about where the controller is going to send it. The A0
measurement replays the *same* trained checkpoint from a grid of later anchors and asks how
the error falls as the remaining path shrinks. Three fixed-anchor checkpoints produce a
monotone, well-behaved curve. A fourth family — models trained *at random anchors*, which
ought to be the right ones for this job — failed the pre-registered L−1 veto by a factor of
two, and yet drew **better time-free geometry than the fixed arms at every anchor on the
grid**. Two rounds of diagnosis found two independent mechanisms: the learning-rate
scheduler was stepping on a selection metric that had gone blind to what the arm improved,
so the model stopped training around epoch 30; and time-uniform anchor sampling
over-weighted the strata near the runway relative to the arm's own anchor population. Both
fixes (A0.b) and the anchor-grid selection metric (A1) are implemented, reviewed and merged.
**Neither A0.b arm has been trained.**

---

## Question

### What a bin is

A **bin** is a remaining-path anchor set. For a chosen distance — 12 km, say — each flight's
anchor in that bin is the observed sample whose **remaining path** (the horizontal arc length
the truth still flies to the threshold,
`approach_difficulty.remaining_path_profile_m`) is closest to the bin value. A flight has no
reading in a bin if that sample has no full 120 s lookback, or if less than `--min-future-s`
of truth remains after it. The measurement grid is
`DEFAULT_ANCHOR_GRID_KM = 20, 16, 12, 8, 6, 4, 2` km (source:
`4dTrajectory/ts_transformer/anchor_strata.py` docstring).

Two design choices are stated in the module docstrings rather than left implicit. **Remaining
path, not time**: it is the covariate the existing near/far strata are already cut on, and it
is comparable across flights of different speeds. And **strata are computed once at L−1 and
fixed for every bin**: relabel per bin and a flight leaves the vectored stratum exactly when
it rolls out on the centreline, so the stratum would improve because its hard members left it
— a survivor curve, not a learning curve. A two-flight test pins that down (source:
`anchor_grid.py` docstring; `docs/CHANGELOG.md`, 2026-09-07 "A0 / B0"). Because bins hold
different flights, the curve is read **paired**: adjacent bins are compared only over the
flights present in both, and the readout prints that count.

### Why a fixed-anchor model is out of distribution elsewhere

Every current checkpoint was trained at L−1 (`FixedAnchorTrajectoryWindows`). Replay it from
an anchor 8 km out and the input distribution has changed — closer, lower, slower, usually
already established. So A0 has to be **two arms**: *A0-fixed* (existing checkpoints, whose
curve carries the out-of-distribution cost as well as the information gain) and *A0-random*
(`random_train_anchor=True`, in distribution at every anchor). Their difference is that cost
(source: `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §2.2).

Why it is a deliverable rather than a curiosity: an arrival manager needs to know *how many
minutes before landing the arrival time can be trusted to ±30 s*, and that quantity did not
exist anywhere in this package — each flight had one anchor and one ADE (same document, §一).

## Setup

### Metrics, first use

- **ADE** — average displacement error: mean 3-D distance between the predicted and observed
  position on a common time grid. **FDE** — the same at the end of the forecast. The gates
  read the **median** ADE (`ade_p50_m`), this package's convention for a heavy-tailed error;
  mean and p95 are published beside it.
- **Chamfer** and **discrete Fréchet** — *time-free* distances between the two polylines,
  horizontal, resampled at 100 m. They say whether the shape is right where ADE says whether
  the aircraft is in the right place at the right time. The package rule is to read both
  families in every cell, and the closure arm below is exactly why.
- **|Δt|** — absolute error of the predicted flight duration, in seconds.
- **Coverage (`cov`)** — a bin's share of the stratum it is reported for. Below
  `anchor_grid.PARTIAL_COVERAGE = 0.5` the point is printed `partial` and no verdict reads
  it.
- **Strata** — `approach_difficulty.strata_masks`: **straight-in** (tortuosity < 1.05),
  **vectored** (tortuosity ≥ 1.05 and not established at the anchor).

### The A0-fixed run

Three checkpoints, KRDU validation split, **1,404 flights** (the `openap-direct` control
fleet), bins 20/16/12/8/6/4/2 km, `--min-future-s 60`, seed 1337 throughout (runner:
`run_ts_anytime_curve.py`; artifact:
`4dTrajectory/outputs/KRDU/experiments/anytime_a0_20260907/anytime_curve.{json,txt}`, written
2026-09-07 01:59):

- `L1_native32` — the current control base (L−1 pooled ADE 1322 m).
- `L2d_warm_beta0p01` — the warm-started latent arm, β = 0.01, the best point estimate at
  L−1 (1214 m).
- `C_pred` — the closure model, present as an **out-of-distribution control**, not a
  competitor (L−1 pooled ADE 996 m).

The runner refuses `cta_conditioning=given` and `intent_conditioning=truth-…` checkpoints:
both read the future, and the grid would re-read the oracle at every anchor, so the curve
would measure how fast the oracle converges. The sealed test split is refused too. Cost is
about 76 min per control arm on CPU: batching is per anchor, and real anchors are almost all
distinct, so the effective batch is ≈ 2.7.

### The pre-registered gates

1. Vectored ADE p50 falls monotonically as remaining path shrinks (a rise of up to 22 m, the
   seed-noise allowance, is tolerated), read paired.
2. There exists s\* > 4 km with vectored ADE(s\*) < 1.5 km.
3. **s_freeze** — the first bin whose vectored |Δt| p80 falls below 30 s. Reported, not
   gated.

**Veto:** an A0-random arm that is worse than `native32` at L−1 by more than seed noise means
random-anchor training damaged the base, and the whole A2/A3 line would be built on a worse
foundation — stop.

### The A0-random arms

`A0_random_hr8_tv1` = `native32` + L1.b's teacherless supervision (heading-rate 8 + bank
total-variation 1) + `random_train_anchor=True` with a 20 s training future contract. The
teacherless supervision is the only one compatible with random anchors: the imitation teacher
recomputes inverse dynamics per sample per epoch (a performance cliff), the fitted teacher
table is anchor-bound, and the closure labels refuse random anchors. Three arms were run
(artifact: `4dTrajectory/outputs/KRDU/experiments/a0_random_20260907/`, seed 1337):

| arm | selection metric | stopping | best epoch |
|---|---|---|---:|
| `A0_random_hr8_tv1` | L−1 common-grid ADE | patience 20, stopped at 30 | 10 |
| `A0_random_hr8_tv1_p180` | L−1 common-grid ADE | early stopping **off**, 180 epochs | 10 |
| `A0_random_hr8_tv1_grid` | **anchor-grid** common-grid ADE | 180 epochs | 8 |

---

## Results

### The A0 curve for fixed-anchor arms

Vectored stratum (n = 497 at L−1), median ADE in metres, from the artifact's per-cell tables:

| s (km) | `L1_native32` | `L2d_warm_beta0p01` | `C_pred` (closure) |
|---:|---:|---:|---:|
| 20 | 2708 | **2434** | 1568 |
| 16 | 1382 | **1301** | 693 |
| 12 | 585 | **528** | 621 |
| 8 | 385 | 390 | 614 |
| 6 | 338 | 353 | **3990** |

Verdicts as printed in `anytime_curve.txt`:

- **Gate 1 passes for both control arms.** `native32` steps −1329 (n = 489), −799 (492),
  −200 (497), −47 (497) m; the warm latent arm −1138, −774, −138, −37 m. `C_pred` **fails**:
  −876, −60, −8, then **+3376 m** from 8 km to 6 km.
- **Gate 2 lands at 16 km for both control arms** (ADE p50 1382 m for `native32`, 1301 m for
  the warm arm). The warm arm is better at every bin from 20 to 12 km (−274 m at 20 km) and
  the difference closes by 6 km.
- **Gate 3 is unreadable on all three arms.** The duration head has a known ~125 s floor, so
  |Δt| *rises* toward the runway however good the geometry gets: `native32`'s vectored |Δt|
  p80 goes 147 s at 20 km → 112 → 85 → 95 → 113 s at 6 km, and its predicted duration p50
  bottoms out at 179 s. The readout prints "not reached in the bins read", plus the predicted
  duration per bin so the floor is visible. **s_freeze cannot be defined from this run** — it
  is a problem of the time channel, i.e. the B line (quantile head plus conformal
  calibration), not of the curve.

The closure arm is why the package insists on both metric families. At 12 km its pooled FDE
p50 is **26 m**, the best endpoint number anywhere in the run, while its pooled ADE mean is
1858 m and its |Δt| p80 is 91 s; at 6 km it collapses (pooled ADE p50 3980 m, FDE p50 8797 m).
Read on ADE alone at 12 km it looks like a competitor, on FDE alone like an oracle. It is
neither: it measures what leaving the training anchor costs a closed-form closure model.

Bin populations, for honesty: 2 km is **empty** at the 60 s floor (~27 s of truth remains
there at 75 m/s), 4 km covers 0.30 and is marked partial, and 20 / 16 km cover 0.35 / 0.37 of
the whole cohort — though 0.98 / 0.99 of the *vectored* stratum, which is why the gates read
that stratum. The straight-in stratum is nearly empty far out (2 and 28 flights at 20 and
16 km).

### The random-anchor arms fail the veto and draw better geometry anyway

At L−1, paired over all 1,404 flights (artifact:
`4dTrajectory/outputs/KRDU/experiments/a0_random_20260907/readout.txt` and
`readout_rerun.txt`, produced by `docs/compare_constraint_arms.py`):

| arm | ADE | FDE mean | FDE p50 | chamfer p50 | ADE better than `native32` on |
|---|---:|---:|---:|---:|---:|
| `L1_native32` | **1322** | 1589 | **864** | **224** | — |
| `A0_random_hr8_tv1` / `_p180` | 2949 | 4551 | 3784 | 612 | 6.8 % |
| `A0_random_hr8_tv1_grid` | 2990 | 4107 | 2430 | 450 | 10.8 % |

The veto fires on all three arms, and not marginally: 2949–2990 m against 1322 m, where the
seed noise on this axis is 5–22 m of pooled ADE.

But replay the *same* first checkpoint on the grid (artifact:
`.../anytime_a0_random_20260907/anytime_curve.txt`, both arms in one run so the comparison is
bin by bin), vectored stratum:

| s (km) | `A0_random` ADE p50 | `native32` ADE p50 | `A0_random` chamfer p50 | `native32` chamfer p50 |
|---:|---:|---:|---:|---:|
| 20 | **2622** | 2708 | **2124** | 2584 |
| 16 | 1847 | **1382** | **1325** | 1439 |
| 12 | 756 | **585** | **426** | 652 |
| 8 | **377** | 385 | **326** | 850 |
| 6 | **298** | 338 | **606** | 1051 |

**Chamfer is better at every one of the five bins** (−114 to −524 m), ADE is better at 8 km and
closer, and gate 2 moves from 16 km to 12 km (ADE p50 756 m). A checkpoint that used 17 % of
its training budget and lost the L−1 comparison outright draws better geometry over the whole
approach. The 16 km degradation (+465 m ADE) is real and still unexplained.

### Round two: it was not the stopping rule, and not a metric the grid could fix

Both pre-registered explanations were tested and both failed (source: §2.4c; artifact:
`.../a0_random_20260907/`, arms `_p180` and `_grid`). With early stopping **off**, `_p180`
still peaked at epoch 10. With **anchor-grid selection**, `_grid` peaked at epoch **8** — grid
mean 1100 m, made of L−1 2990, 12 km 722, 8 km 390, 6 km 298. Per anchor set, L−1 and 12 km
*degrade* after epoch 8–10 (12 km 722 → 1324, nearly double), 8 km slips from 314 at epoch 3 to
406, and only the 6 km set keeps improving (291 → 218). There was no good late checkpoint to
select.

`history.json` of the `_grid` arm gave the mechanism:

| epoch | lr | train loss | validation objective | val state | selection metric (grid mean) |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.0e-5 | 0.884 | 1.147 | 0.264 | **1100** |
| 15 | 3.0e-5 | 0.466 | 0.836 | 0.150 | — |
| 20 | 1.5e-5 | 0.487 | 0.786 | 0.139 | 1384 |
| 30 | 7.5e-6 | 0.414 | 0.765 | 0.131 | — |
| 60 | **9.4e-7** | 0.424 | 0.707 | 0.102 | 1289 |
| 100 | 2.9e-8 | 0.487 | 0.705 | 0.101 | — |
| 180 | 1.5e-8 | 0.521 | 0.706 | 0.102 | 1285 |

Two independent diagnoses:

1. **The learning-rate scheduler was stepping on a stalled selection metric.** The validation
   *objective* kept improving to epoch 60 (1.147 → 0.707, val state 0.264 → 0.102) while the
   *selection metric* stalled after epoch 8. `ReduceLROnPlateau` read the selection value, so
   the rate was halved from epoch 20 and reached 9.4e-7 by epoch 60 and 2.9e-8 by 100. From
   about epoch 30 the model was not training; it was frozen at the epoch the *readout* stalled.
   Fixed-anchor arms never exposed this, because their selection metric improves for 100+
   epochs.
2. **Time-uniform anchor sampling over-weights the near strata.** Each flight draws one anchor
   per epoch, uniformly over its admissible *samples*, i.e. uniformly in time. Per flight that
   is not a skew — a flight's anchors start at index 59 and the median flight has 84 of them,
   so a uniform draw centres at 100.8 by construction and lands at a measured median of 108.
   The skew is in the **pooling**: every flight contributes one draw whatever its length, and a
   kilometre near the runway holds more samples than a kilometre at 25 km because the aircraft
   is slower there. Measured over the whole split, draws below 6 km are **34.3 %** against a
   population share of **25.1 %** (full table below). The 6 km set improving while L−1 and
   12 km degrade is exactly that shape.

### A0.b: the two fixes, and the review that reversed one of them

Two config axes, both defaulting to today's behaviour (source: §2.4c "Implementation";
`docs/CHANGELOG.md`, 2026-09-07 "A0.b"; branch `dev-a0b`, commits `12d35ce` the scheduler
axis, `23cae12` the sampling axis, `965077a` the two arms, `f8a1726` the docs, plus the
review-fix commit `45d015a`):

- **`lr_plateau_metric ∈ {selection, objective}`.** Under `objective` the scheduler steps on
  the macro validation objective the epoch record already writes — one of the two numbers the
  epoch already produced, not a third computed for the purpose — and **checkpoint selection is
  unchanged either way**. It is *refused* under `latent_beta_warmup_epochs > 0` (the β ramp
  reweights the objective every epoch) and `procedure_loss_dual_step > 0` (λ moves every
  epoch, so the same trajectory is priced differently each time): there a plateau would be the
  schedule's, not the model's, and the scheduler would cut the rate straight through a ramp.
- **`random_train_anchor_sampling ∈ {uniform, remaining-path-uniform}`.** The new law places
  each flight's draw uniformly across **its own** admissible remaining-path span and takes the
  nearest admissible anchor — equal weight per kilometre, not per sample — from the same
  per-flight, per-epoch sha256, so determinism is unchanged. The pre-registration had written
  a *stratum-first* law (draw a bin, then a sample in it); the review measured it and found it
  pushed training **toward** the runway, the opposite of the intent, because the grid cuts the
  near end into four 2 km strata while the far end is one open stratum spanning 20–123 km and
  holding 33 % of the anchors — equal weight per stratum gives that a mere eighth of the
  probability. On 700 KRDU validation flights the rejected law moved the mean draw 12.2 → 8.0
  km, p90 31.8 → 14.7 km and the ≥ 20 km share 16.9 % → 4.1 %. It was replaced; the strata
  survive only as bookkeeping.

The population-versus-draws table that settled it (1,404 flights, 181,906 admissible anchors,
200 epochs):

| remaining-path stratum | population | `uniform` draws | `remaining-path-uniform` draws |
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

The new law moves the draws *toward* the population without reaching it. Because only the
drawn share is observable in a training log, `history.json` now records both
`train_anchor_sampling.remaining_path_strata` (what was drawn) and `_population` (what it was
drawn from), under **both** strategies — a drawn share alone cannot show over-sampling.

Equivalence evidence, recorded because both defaults reproduce today's behaviour: the four
fixed-anchor training paths (control, latent, state, closure) produce **byte-identical**
two-epoch histories; the `uniform` random-anchor arm gains only the two bookkeeping keys, all
342 numeric leaves unchanged; 935 config-carrying artifacts under `4dTrajectory/outputs` were
re-derived with **zero** name, slug or loadability changes; the test suite went 680 → 719,
all green.

**The two arms `A0b_lr_objective` and `A0b_lr_objective_path_uniform` are declared in
`docs/experiments/a0_random_arms.json`, dry-run clean, and have NOT been trained.**
Pre-registered reading: the L−1 and 12 km anchor sets should stop degrading after epoch 10,
and the best epoch should land later than 60. The veto is unchanged, and all three existing
random-anchor arms are still 2949–2990 m at L−1, so this line's foundation does not yet
stand.

### A1: the selection metric the random-anchor arm needed

`checkpoint_selection_metric=anchor-grid-common-grid-ade` averages the *same* common-grid ADE
over L−1 plus whichever of `VALIDATION_ANCHOR_GRID_KM = 16, 12, 8, 6` km the cohort can cover
(source: §2.4; `docs/CHANGELOG.md`, 2026-09-07 "A1"; branch `dev-a1`). Three decisions carry
the weight:

- **Equal weight per anchor set**, not per (flight, anchor) pair. Pooling the pairs would weight
  each bin by its coverage and fade the far bins out exactly on the arms whose flights are
  vectored.
- **Each set's ADE is the mean over the flights that *have* that anchor.** A flight absent from
  a bin is absent from its mean, never scored zero.
- **A candidate bin is not a bin the metric will use.** Measured coverage of the KRDU validation
  cohort at the 60 s floor: 20 km 33.7 %, 16 km 37 %, 12 km 78.7 %, 8 and 6 km 99.7 % — the
  median flight has only **13.4 km** left at L−1. So 20 km is not a candidate, 4 and 2 km are
  partial-to-empty, and of the four candidates any bin below `anchor_grid.PARTIAL_COVERAGE`
  (0.5, the same constant A0 marks a point `partial` with) is dropped at plan build with a
  printed notice and a `dropped_bins` record — on this cohort, 16 km. Fewer than two surviving
  bins refuses the metric outright. **Which bins survive is a property of the cohort, not of
  the model**, so every arm on the same split is selected on the same sets.

L−1 is still recorded every epoch (`validation_anchor_grid.fixed_anchor_common_grid_ade_m`), so
the veto stays readable; it no longer decides which epoch is kept. The two oracle inputs
(`cta_conditioning=given`, `intent_conditioning != none`) are refused under this metric, for the
reason the A0 runner refuses those checkpoints. Cost: the selection pass is 4–5×, i.e. +12–15 %
per epoch at KRDU scale, 7–9 extra minutes over a 180-epoch arm.

One grid module serves both consumers, with a test asserting *identity* rather than equality;
its values live one level down in the leaf module `anchor_strata.py`, because training needs
them too and `dataset` cannot import `anchor_grid` (which imports `dataset`).

---

## Reading

1. **A model can be better everywhere the deliverable cares about and still lose the metric
   that selects it.** The random-anchor arm's failure was a selection failure compounded by a
   scheduler failure, not a training failure — the validation objective improved for another
   50 epochs after the selection metric stalled.
2. **Single-anchor evaluation was hiding a whole axis.** The curve turns one scalar per flight
   into a function of remaining path, and immediately shows gate 2 at 16 km for the fixed arms
   and 12 km for a random-anchor arm that "failed".
3. **Read both metric families, per cell.** The closure arm at 12 km — 26 m FDE p50, 1858 m
   pooled ADE mean, 91 s |Δt| p80 — is unreadable on either family alone.
4. **Near the runway the binding constraint is the time channel, not the geometry.** The
   duration head's ~125 s floor makes |Δt| rise as the aircraft approaches, so the freeze-point
   question the arrival manager actually asks cannot be answered from this curve. It belongs to
   the B line, whose code is built and whose arms are also untrained.
5. **A skew can be invisible per unit and obvious in aggregate.** The per-flight draw was
   provably unbiased; the pooled draw was 34.3 % against a population 25.1 % below 6 km.
   Logging the drawn share without its population would not have shown it — which is why both
   are now recorded.
6. **Pre-registration constrained the diagnosis, not only the verdict.** Both pre-registered
   explanations for the epoch-10 peak were tested and both failed before the training log was
   opened for a third; and the review reversed a pre-registered *implementation* after
   measuring it, which is recorded rather than quietly rewritten.

## Caveats

- **KRDU only, one seed (1337), one fleet** (`openap-direct`, 1,404 validation flights). No
  second airport and no seed replicate anywhere in this line.
- **The A0.b arms are not trained.** Nothing here says the fixes work; it says what they are,
  why, and what they are pre-registered to show.
- **The A0-random line's foundation does not stand.** All three arms are 2949–2990 m at L−1
  against `native32`'s 1322 m, so the curve comparisons are informative about mechanism, not
  about a deployable model. The 16 km degradation (+465 m ADE, +1530 m FDE) has no explanation.
- **Bins are not equally trustworthy.** 2 km is empty at the 60 s floor, 4 km covers 0.30, and
  the far bins cover about a third of the whole cohort (though nearly all of the vectored
  stratum). Gates skip the `partial` points and the readout says so.
- **s_freeze does not exist in this run.** Defining it needs a lower future floor (e.g.
  `--min-future-s 20`), which changes a bin's *population*, not only its range, and would have
  to be stated in the result.
- **The A0-fixed curve confounds two effects** — information gained by flying closer, and the
  out-of-distribution cost of an unseen anchor. Only the A0-random arm separates them, and it
  is the arm that failed the veto.
- **The per-bin records published to the frontend are a smoke test**, not a measurement.

## What happened next

**Merged, untrained.** A0.b (`lr_plateau_metric`, `random_train_anchor_sampling`) and A1.a
(`anchor_grid.py`, `anchor_strata.py`, `checkpoint_selection_metric=anchor-grid-common-grid-ade`)
are implemented, reviewed and on the branch. The A0.b arms sit in
`docs/experiments/a0_random_arms.json`, scheduled after the L1.b hr16 arms and the L1.c corridor
arms in the campaign queue.

**Still design-stage.** A1 proper (the streaming evaluation protocol: an anchor grid inside
`evaluation_protocol` and extra columns in `compare_constraint_arms.py`), A2 (candidate
re-weighting at prediction time — a particle filter over intent, gated on "no worse than the
stateless version at the same anchor"), A3 (a learned recursive prior), and B4, the frozen-point
curve, which is the actual deliverable and is A × B.

**Publication.** On 2026-09-07 the per-bin forecasts of three checkpoints at 12, 8 and 6 km were
published to the frontend's Experiments picker under the heading `anytime_records_20260908` —
nine categories at KRDU (`L1_native32`, `L2d_warm_beta0p01` and `A0_random_hr8_tv1_grid`, each at
three bins). **These records come from a `--limit 300` smoke run**, which the artifact labels "a
SMOKE TEST over the first flights of the split, not a measurement": 244 flights at 12 km and 300
at 8 and 6 km, out of the 1,404 in the split (artifacts:
`4dTrajectory/outputs/KRDU/experiments/anytime_records_20260908/anytime_curve.txt` and
`records/`; `aeroviz-4d/public/data/airports/KRDU/comparison/categories.json`, written
2026-09-07 08:04 and 09:51 UTC). They are there to be *looked at*; the numbers in this report
come from the full-cohort runs.

## Sources

**Design document** —
`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`:
§〇 status table and §〇.1 the A0 / B0 run commands with cost and refusal notes; §一 why this
line exists (the 962 m intent budget, the AMAN freeze point); §2.1 the definition of P(s);
§2.2 the out-of-distribution trap and the two-arm design; §2.3 the anchor grid; §2.4 the
pre-registered gates; **§2.4b** the A0 results; **§2.4c** the second random round, the
`history.json` mechanism table, the sampling diagnosis, the A0.b pre-registration and the
post-review implementation; §2.5 the A2 gates.

**Changelog** — `docs/CHANGELOG.md`, 2026-09-07 entries: "A0 / B0: the re-anchoring curve and
the arrival-time error distribution"; "A1: the selection metric was blind to what the
random-anchor arm improved"; "A0.b: the random-anchor arm was frozen by its own learning-rate
schedule, and drew its anchors nearer the runway than its own anchor population".

**Code** (docstrings quoted above) — `run_ts_anytime_curve.py` (grid, fixed-at-L−1 strata,
paired reading, both metric families, arm naming, refusals); `run_ts_eta_error_readout.py`
(the B0 companion readout); `4dTrajectory/ts_transformer/anchor_grid.py` (one grid, three
consumers); `.../anchor_strata.py` (the leaf module holding the values and the draw law).

**Artifacts** (KRDU validation split, seed 1337, outer test untouched; under
`4dTrajectory/outputs/KRDU/experiments/`)
- `anytime_a0_20260907/anytime_curve.{json,txt}` — A0-fixed: `L1_native32`,
  `L2d_warm_beta0p01`, `C_pred`; 1,404 of 1,404 flights; bins 20/16/12/8/6/4/2 km;
  2026-09-07 01:59.
- `anytime_a0_random_20260907/` — `A0_random_hr8_tv1` and `L1_native32` in one run, bin by
  bin; 1,404 flights; 02:28. `anytime_a0_random2_20260907/` — `A0_random_p180`,
  `A0_random_grid`, `L1_native32`; 06:54.
- `a0_random_20260907/` — the three trained arms with their `history.json`, plus
  `readout.{json,txt}` (first arm vs `native32`) and `readout_rerun.{json,txt}` (`_p180` and
  `_grid` vs `native32`); 1,404 paired flights; 02:24 and 06:47.
- `anytime_records_20260908/anytime_curve.txt` and `records/` — the published per-bin
  forecasts; **`--limit 300` smoke test**; 08:03.
- `b0_eta_error_20260907/eta_error.json` — the B0 arrival-time error readout; 00:15.
- `aeroviz-4d/public/data/airports/KRDU/comparison/categories.json` — the published
  Experiments picker categories.
