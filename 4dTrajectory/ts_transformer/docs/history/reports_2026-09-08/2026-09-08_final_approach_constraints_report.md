# Putting the final-approach procedure into a learned predictor: bounded outputs, penalties, projections and command hooks

*Experiments of 2026-09-04 → 2026-09-06. Written 2026-09-08. All numbers are from the
validation split; the outer-test split has never been read.*

## Abstract

An RNAV (GPS) approach ends in a well-defined piece of geometry: a lateral corridor that
narrows towards the runway, and a vertical window around the glidepath. The optimizer in
this repository already enforces both as inequality rows. The question here was whether a
learned trajectory predictor can be made to respect the same geometry, and by which
mechanism. Five mechanisms were tried across three campaigns. Two were adopted: a **bounded
output layer** on the state-output model, and a **soft barrier filter applied at inference
time** on the control-output model. Three were rejected: a training-time penalty (both with
a fixed weight and with a primal-dual multiplier), a post-hoc projection gated at the final
approach fix, and — the most surprising result — letting the network train *through* the
command hook it is later deployed with. Six training-through arms were run across two hooks
and two airports and not one of them beat its own inference-time control.

---

## Question

The optimizer's final leg (`4dTrajectory/optimization/approach_constraints`) constrains an
LPV **angular corridor**, |cross-track| ≤ k·hw(d) with hw(d) = cw·(d + d_GARP)/d_GARP and d
the along-course distance back from the threshold, and a **glidepath window**,
height ∈ [h_GP(d) − 60 m, h_GP(d) + 120 m]. Before any of it went into a loss function, the
design document measured which of the optimizer's constraint families the *observed* traffic
actually satisfies (source: `2026-09-04_procedure_constraints_design.zh.md` §1–§2; script
`docs/measure_procedure_adherence.py`, every third flight, KRDU n = 4,812, KSJC n = 3,694).

The answer decided the scope. **Zero per cent** of flights at either airport passed within
1 NM of any off-axis IAF of the published RNAV (GPS) procedure, and 15–62 % join the final
*inside* the final approach fix (FAF), which the optimizer forbids. Only the final segment
matches the data: once established, 87–99 % of samples lie inside the k = 0.5 corridor and
90–99 % inside the glidepath window. Everything upstream is prescriptive rather than
descriptive, and putting it into a loss would ask the model to fit paths that do not exist.

So the question became narrow: **on the final segment only, which mechanism makes a learned
predictor respect the corridor and the glidepath, and what does it cost?**

## Setup

### Data and splits

Two airports, per-airport training, validation split only, outer test sealed.

| Campaign | Output | Airport | Validation flights | Fleet |
|---|---|---|---:|---|
| `final_constraint_20260904` | state | KRDU / KSJC | 2,104 / 1,666 | all aircraft |
| `control_procedure_20260905` | control | KRDU / KSJC | 1,404 / 1,083 | `openap-direct` subset |
| `control_hooks_20260906` (v1) | control | KRDU | 1,404 | `openap-direct` |
| `control_hooks_v2_20260906` | control | KRDU / KSJC | 1,404 / 1,083 | `openap-direct` |

The two cohorts are **different flight sets**, so a state number and a control number may
be compared only for direction, never for magnitude (source:
`2026-09-05_control_penalty_results.zh.md`, preamble). The split figures are the same ones
the frame ablation reports (`2026-09-03_airport_frame_ablation_results.md`, "What was run").

### Metrics, first use

- **ADE** (average displacement error) — mean 3-D distance between predicted and observed
  position on a common time grid; **FDE** (final displacement error) — the same at the end
  of the forecast. Both time-aligned, metres.
- **Chamfer** and **discrete Fréchet** — *time-free* distances between the two polylines
  (horizontal, 100 m resampling): the right shape rather than the right place at the right
  time. Both families are always read together, because a timing-only change moves ADE and
  leaves chamfer where it was (source: `4dTrajectory/ts_transformer/CLAUDE.md`, "How to read
  results here").
- **Corridor violation rate** — share of *rows* where the prediction lies outside the k = 0.5
  corridor, counted only on rows where the observed track was already established on the
  final (the "truth gate"); **glidepath window violation rate** is the vertical analogue.
  Read against the observed baseline, not against zero.
- **Threshold cross-track p95** — 95th percentile of |lateral offset| of the predicted
  endpoint from the centreline.
- **Bank skill** — correlation between the predicted and flown bank schedule of the same
  flight after removing the airport's common bank profile. Read against two printed
  references, never against 1.0: a *random other real flight* is the floor (KRDU **0.170**),
  a *same-runway twin* the yardstick for how much future bank is knowable (KRDU **0.699**;
  KSJC 0.31 and 0.543). **Common-profile share** is how much of the bank is identical on
  every flight; a rising share signals a collapsed schedule (source:
  `docs/score_control_arms.py` docstring).
- **Flyability** — whether a record's true-dynamics rollout stays inside the aircraft
  envelope. Deliberately *not* a quality metric: a blander predictor scores higher on it, so
  it is read only as a delta against observed tracks.
- **Strata** (`approach_difficulty.strata_masks`) — **straight-in** (tortuosity < 1.05) and
  **vectored** (tortuosity ≥ 1.05 and not established at the anchor), where tortuosity is
  remaining flown path over straight-line range to the threshold.

### The shared geometry, and the two gates

One torch module, `4dTrajectory/ts_transformer/final_approach_geometry.py`, serves the
bounded output layer, the penalty and the projection, and shares `hw(d)` with the optimizer
through `flight_scenarios/fas_geometry.py`. Two gates decide which rows are "on the final":

- **`on-final`** — the row lies inside the full-scale cone (floored at 500 m near the
  threshold, because the corridor half-width there is only 107 m while the state model
  carries a 250–350 m bias) **and** the predicted *path* direction is within 30° of the
  course. Direction is read from the predicted positions, never from the velocity channels:
  the state objective supervises positions only, so the network could otherwise steer the
  velocities to switch the gate off. Deployable — it reads only the prediction.
- **`faf`** — every row with d ≤ d_FAF, the optimizer's convention; kept only as the ablation
  that shows why the self-gate is needed.

The **truth gate** used for scoring is the tail of rows from which the *observed* track stays
inside the k = 0.5 cone all the way to the threshold.

### Arms and what was pre-registered

State path (`final_constraint_20260904`; source: `2026-09-05_final_constraint_results.zh.md`
§1). B, C_dual and the on-final projection ran on **two seeds** (1337, 2024); C_fixed and
the FAF projection on one (1337). Training arms carried a formal experiment manifest
(commits `5b54fae`, `99ad2d3`).

| Arm | Mechanism |
|---|---|
| A `A_threshold_enu` | the 2026-09-03 baseline, reused, not retrained |
| A + proj(on-final) | per-row hard clamp of A's prediction, no training |
| A + proj(faf) | same, gated at the FAF distance — the post-hoc upper bound |
| B `B_corridor_bounded` | **bounded output**: the position output is tanh-saturated into ±k·hw(d) and [−60, +120] m in runway-axis coordinates, softly gated; the network trains through the map (`state_position_reference="corridor-bounded"`) |
| C_dual | runway-scale hinge² penalty, λ found by dual ascent towards a 5 % violation target |
| C_fixed | the same penalty at a fixed λ = 1e-3 |

Pre-registered veto, inherited from the earlier `state-v2` experiment: **degradation of the
vectored stratum on both seeds**.

Control path, penalty (`control_procedure_20260905`; source:
`2026-09-05_control_penalty_results.zh.md` §1): `A_control_v3` (`simple-v3` recipe, newly
trained because every control checkpoint predated the v5 roster), plus the same hinge
penalty at λ = 1e-3 and 5e-3, applied to the 64 rollout segment-endpoint states. One seed.

Control path, command hooks (`control_hooks_20260906` v1, `control_hooks_v2_20260906`;
source: `2026-09-06_control_hooks_results.zh.md`, "Arms"). Two hooks — a **barrier filter**
(changes the bank command only when the rollout would leave the corridor) and a **nominal
law + bounded residual** (a continuous tracking law with a ±5° bank / bounded load-factor
correction) — each in an inference-only form and a trained-through form. Pre-registered
"lazy network" veto: the trained arm is rejected if the share of *clipped steps* stays above
20 % **and** bank skill falls below the baseline minus seed noise.

---

## Results

### The compliance floor

On truth-gated rows the observed tracks violate the corridor 0 % of the time by definition,
and the glidepath window **6.1 %** at KRDU (2.6 % straight-in, 11.4 % vectored) and **1.7 %**
at KSJC (source: `2026-09-05_final_constraint_results.zh.md` §2). On the control cohort the
same observed figures are 5.7 % and 0.8 % — a different flight set, hence a different floor.
Any arm's violation rate is read against these, never against zero.

### State output: bounded output adopted, penalty rejected, projection is the ceiling

Bounded output B against A, paired flight-by-flight on the same seed (artifact:
`4dTrajectory/outputs/{KRDU,KSJC}/experiments/final_constraint_20260904/readout.json`;
source: `2026-09-05_final_constraint_results.zh.md` §4):

| | KRDU s1337 | KRDU s2024 | KSJC s1337 | KSJC s2024 |
|---|---|---|---|---|
| pooled ADE | 1383 → 1337 | 1361 → 1353 | 870 → 852 | 865 → 794 |
| pooled FDE mean | 1163 → 1072 | 1124 → 1058 | 776 → 725 | 755 → 692 |
| straight-in FDE | 643 → 548 | 622 → 526 | 449 → 411 | 430 → 412 |
| vectored ADE | 2599 → 2592 | 2574 → 2623 | 2536 → 2508 | 2521 → 2317 |
| corridor violation rows | 77.4 → 47.8 % | 74.7 → 48.1 % | 34.1 → 22.0 % | 33.6 → 20.2 % |
| glidepath violation rows | 62.1 → 42.3 % | 59.6 → 43.0 % | 38.5 → 22.6 % | 45.9 → 21.0 % |
| threshold \|xt\| p95 (m) | 492 → 305 | 477 → 298 | 237 → 171 | 195 → 140 |

All four runs improve FDE (−91 / −66 / −51 / −63 m) and all four improve ADE (−46 / −8 /
−18 / −71 m). The pre-registered veto did not fire: vectored ADE is one up and one down at
KRDU (−7 / +49 m, against a 25 m two-seed spread on A itself) and improves on both KSJC
seeds. Two honesty notes are recorded with the result. The checkpoint is *selected* on
common-grid ADE, so part of the ADE gain is a selection effect; the numbers that took no
part in selection are FDE, the violation rates and the threshold p95. And the 8 m ADE gain
on KRDU seed 2024 sits inside the seed noise band (A's own two seeds differ by 22 m pooled
ADE at KRDU, 5 m at KSJC).

The penalty is the clean contrast. **C_dual diverged on all four runs.** The tolerated
violation rate ε = 0.05 is unreachable (A itself sits at 0.77, B only reaches 0.48), so the
multiplier only ever climbed: λ grew 50-fold over 74 epochs and the penalty overtook the
position term by epoch 20 (0.191 versus 0.630). Best common-grid ADE landed at 2420 / 2749 m
(KRDU) and 2258 / 2195 m (KSJC) against A's 1383 / 1361 / 870 / 865. **C_fixed** at the
calibrated parity weight λ = 1e-3 did not diverge but bought nothing: KRDU 1383 → 1386 ADE,
KSJC 870 → 912. It lowered violation rates about as much as B did and paid accuracy for it.

Projection, applied post hoc to A with no training: the **on-final** gate recovers most of
B's endpoint gain at KRDU (pooled FDE 1163 → 1090 against B's 1072; straight-in FDE
643 → 545 against B's 548) but not the violation rate (56.3 % against 47.8 %) nor the
threshold p95 (480 m, unmoved, against 305 m). The **FAF** gate is the straight-in upper
bound (FDE 455 at KRDU, 399 at KSJC) and destroys the vectored stratum (ADE 2599 → 4665,
2536 → 5281), because every downwind and base row inside the FAF distance is crushed into
the corridor.

Bounded output did **not** fix the start of the path: the median first-step offset stays at
KRDU 390 → 377 and 368 → 389 m. The established-flight endpoint cross-track median moved
from +206 / +149 m to +54 / +54 m — exactly the corridor edge k·hw(0) = 53 m. The
north-west bias is still there; the wall is holding it.

### Control output: the penalty helps the endpoint and hurts the middle

Applying the same hinge to the 64 rollout segment-endpoint states (artifact:
`4dTrajectory/outputs/{KRDU,KSJC}/experiments/control_procedure_20260905/readout.json`,
seed 1337, KRDU n = 1,404, KSJC n = 1,083; source:
`2026-09-05_control_penalty_results.zh.md` §2):

| KRDU | A | λ=1e-3 | λ=5e-3 |
|---|---:|---:|---:|
| pooled ADE / FDE mean | 1333 / 1650 | 1537 / 1549 | 1917 / 1904 |
| straight-in ADE / FDE | 469 / 938 | 466 / 888 | 530 / 1001 |
| vectored ADE / FDE | 2858 / 2949 | 3439 / 2754 | 4396 / 3552 |
| corridor / glidepath violation | 54.8 / 46.6 % | 45.2 / 46.0 % | 46.6 / 53.1 % |
| threshold \|xt\| p95 (m) | 2769 | 2203 | 1999 |
| bank skill (floor 0.170, twin 0.699) | 0.728 | 0.674 | **0.280** |
| common-profile share (flown 1.8 %) | 3.8 % | 3.7 % | **13.7 %** |

The endpoint improves and the middle of the path gets worse: vectored ADE is +581 m at
λ = 1e-3, and at 5e-3 the bank schedule collapses to a shared shape (skill 0.280, near the
random-flight floor of 0.170) — the "blandness trap" signature. The mechanism is the
rollout: the penalty is charged only on rows where the truth is already established, but a
rollout integrates sequentially, so the cheapest way to bring the last rows into the
corridor is to turn early, and the downwind and base of a vectored flight get pushed off.
The state path has no such coupling, which is why the penalty there moved violation rates
and not endpoints.

A calibration lesson came out of it: the *same* λ is 4.4× the position term on the control
path in epoch 1 against 0.78× on the state path, and still 1.3–1.5× at the end of training
against 0.53–0.55× on state. λ = 1e-3 was never "parity" on the control path.

### Control output: the barrier filter at inference time

The v1 barrier filter had two design defects, both visible on its worst flight: a rate-based
heading rule that went bang-bang under "command held per segment plus first-order actuator
lag" (+28° then −29° of bank over four segments), and a bank-only correction that stole
vertical lift (flight path angle −1° → −10°, speed 93 → 200 m/s, overflying the threshold by
16 km). v2 added lag compensation, evaluated the corridor margin at the predicted position
when the command takes effect, coordinated the load factor as n′ = n·cos μ / cos μ′, and
recalibrated the heading gain 0.3 → 0.1. Applied at inference only to the same baseline
checkpoint (artifact:
`4dTrajectory/outputs/KRDU/experiments/control_hooks_v2_20260906/readout.json`, n = 1,404,
seed 1337; source: `2026-09-06_control_hooks_results.zh.md`):

| KRDU | A | v1 soft | **v2 soft** |
|---|---:|---:|---:|
| pooled ADE / FDE mean / FDE p50 | 1333 / 1650 / 908 | 1335 / 1634 / 756 | **1278 / 1449 / 700** |
| straight-in ADE / FDE mean | 469 / 938 | 455 / 842 | 398 / 715 |
| vectored ADE / FDE mean | 2858 / 2949 | 2889 / 3080 | 2831 / 2786 |
| straight-in threshold \|xt\| p95 | 1821 m | 56 m | 70 m |
| vertical hinge | 15.6 | 53.6 | 15.4 |
| paired FDE better / worse by > 1 km | — | 78.9 % / 54 flights | **84.3 % / 0 flights** |
| bank skill | 0.728 | 0.578 | 0.644 |

KSJC reproduces the direction at smaller magnitude (pooled FDE 996 → 930, better on 90.3 % of
flights, worst case +220 m; straight-in corridor violation 15.9 → 0.7 %, threshold |xt| p95
407 → 46 m). The time-free geometry agrees: KRDU pooled chamfer p50 256 → 107 m, better on
93 % of flights, and the soft v2 filter is the only hook that does not degrade the vectored
stratum geometrically (chamfer 942 → 886, better on 81 %).

**The soft form beat the hard form everywhere** — violation rate, endpoint tail and worst
case. Hard saturation with hard gating makes the command jump at the edge of the membership
cone; the soft gate's ~5 % ramp smooths it (KRDU: 10 flights worse by > 1 km under hard v2,
0 under soft).

### Training through the hook: six arms, no winners

| Arm | pooled ADE vs A | vs its own inference-time control |
|---|---|---|
| KRDU barrier v1, trained | 1539 (A 1333) | worse than 1335 |
| KRDU barrier v2, trained | 1404 | worse than 1278 |
| KRDU nominal law v1, trained | 1618 | — (v1 failed on energy) |
| KRDU nominal law v2, trained | 1421 | worse than 1299 |
| KSJC barrier v2, trained | 669 (A 654) | worse than 637 |
| KSJC nominal law v2, trained | 713 | — |

Every trained arm raised pooled ADE (+2 % to +21 %), every one lowered bank skill, and every
one degraded the middle of vectored paths. The pre-registered lazy-network veto fired
outright on the KRDU barrier v2 arm (clipped steps ≥ 20 % on 125 of 132 epochs, bank skill
0.586 against a 0.728 baseline). The KSJC arm's clipped share fell to 15 %, below the line,
but its bank skill dropped to 0.543 — exactly the same-runway twin level — and it was read
the same way.

The nominal law's own story is a vertical one. Its v1 failure was energy: changing the
flight path angle without adjusting thrust cost 30 m/s and left the aircraft 584 m short of
the threshold. v2 integrates a parallel hook-free reference rollout and holds thrust to the
reference speed; the endpoint height relative to the threshold moved from −164 m to −7 m and
the vertical violation rate from 46.6 % to 29.1 %, the best vertical channel of any arm. Its
lateral effect at inference is small (corridor violation 55 → 52 %).

---

## Reading

1. **A constraint on the feasible set beats the same constraint on the objective.** Bounded
   output and the fixed penalty enforce the *same* geometry with the *same* code. The
   penalty pays accuracy for the violation rate it buys; the bounded output improves both,
   and has no weight to calibrate.
2. **On the control path, where to act matters more than how hard.** The penalty acts on the
   objective and propagates back through the rollout to segments far upstream, so it fixes
   the endpoint by wrecking the downwind. The barrier filter acts on the command only when
   the gate is open, so the endpoint tail collapses and the middle is untouched.
3. **Safety layers belong at inference, not in the training loop.** Six arms, two hooks, two
   airports, no exceptions. The network learns that the hook will catch it.
4. **Softness is better engineering, not a compromise.** Under "command held per segment plus
   first-order actuator lag" a hard gate makes the command jump inside one segment. Even if
   the deployed form saturates hard, the *gate* should stay soft.
5. **Violation rates have a floor near 48 % (state) and 45 % (control) at KRDU.** About 60 %
   of the remaining violating rows are vectored flights whose truth has already rolled out
   while the prediction is still kilometres away. No corridor mechanism helps there — the
   prediction is not in the corridor to be constrained, and a barrier filter is inactive on
   rows already outside.
6. **The corridor governs the end of the path, not the start.** The anchor-continuity prior
   governs the beginning; the two are orthogonal, and the KRDU north-west translation
   survives the corridor unchanged.

## Caveats

- The control campaigns are **single-seed**. Two airports agreeing gives direction, not an
  effect-size interval. The only calibration available is the state path's two-seed spread
  (5–22 m pooled ADE); the barrier v2 effect at KRDU (−201 m FDE, −55 m ADE, zero flights
  worse by > 1 km) is far outside it, KSJC's −17 m ADE is not.
- The state cohort (2,104 / 1,666, all aircraft) and the control cohort (1,404 / 1,083,
  `openap-direct`) are different flight sets with different observed compliance floors;
  magnitudes are not comparable across the two campaigns.
- One recipe per path, iTransformer only. PatchTST was never run through the bounded map,
  and its channel-independent backbone may not be able to use it.
- KSJC's parallel pair is barely exercised (30L 1,444, 30R 133, 12R 86, 12L 3).
- **Flyability of the projected paths was not measured.** Per-row hard clamping makes corners
  at the edge of the membership cone; the cost is stated but unquantified.
- The nominal-law hook was **archived on 2026-09-07** (package audit T2, into
  `archive/nominal_law_hook_2026_09/`) and is no longer constructible; its numbers are kept
  as history, not as a reproducible arm.
- The projection's first implementation clamped only maximal on-final tails and moved 1.35 %
  of rows; the tables above are from the per-row re-run.

## What happened next

**Adopted.** `state_position_reference="corridor-bounded"` with `corridor_gate="on-final"`
became a candidate default for the state path (final commitment deferred until the
anchor-continuity term is decided alongside it, since one governs the endpoint and the other
the start). `control_command_hook="barrier"` with `control_hook_saturation="soft"` became
the **inference-time** default safety layer for the control path
(`predict --command-hook barrier --hook-saturation soft`), explicitly not part of the
training recipe. `procedure_loss_*` is kept as an option with weight 0 and dual step 0.
`predict --project-final` is kept: `on-final` as a deployment fallback, `faf` only as a
readout ceiling.

**Pre-registered, not run.** The 2026-09-05 penalty veto was measured on a base carrying an
imitation teacher that took 87 % of the loss; the teacher pins the controls to an
inverse-dynamics target the rollout cannot fly back to, which is consistent with the mid-path
cost and the bank collapse being a *fight* rather than a property of the penalty. **L1.c**
re-tests the training-time corridor penalty on a teacher-free base: two training arms
(λ = 2e-4, the corrected parity dose, and λ = 1e-3, the 09-05 dose) plus two predict-only
arms applying the soft barrier hook to the base and to the parity arm (source:
`docs/2026-09-07_latent_intent_design.zh.md` §六 L1.c; campaign `l1c_procedure_20260908`,
KRDU, 180 epochs, seed 1337, val n = 1,404). Gate: corridor violation down ≥ 5 points,
vectored ADE no worse than +30 m, bank skill ≥ 0.70; veto at vectored ADE worse by > 100 m.
**It has not been run.** Also listed as not done: PatchTST through the bounded map, a second
control-path seed, the other three airports, and the combined hook (lateral barrier plus
vertical nominal law) the two complementary results point at.

**Publication.** On 2026-09-07 the arms of all four campaigns were published to the
frontend's Experiments picker: fifteen categories per airport across the headings
`airport_frame_20260903` (6), `final_constraint_20260904` (3), `control_procedure_20260905`
(4) and `control_hooks_v2_20260906` (2), at both KRDU and KSJC (artifact:
`aeroviz-4d/public/data/airports/{KRDU,KSJC}/comparison/categories.json`, written
2026-09-07 09:51 UTC).

## Sources

All document paths are relative to `4dTrajectory/ts_transformer/docs/` unless stated.

**Design**
- `2026-09-04_procedure_constraints_design.zh.md` — §1 constraint families vs the data, §2
  per-runway adherence, §3 the three-layer implementation, §4 the pre-registered design.
- `2026-09-04_constraint_methods_survey.zh.md` — §1 the eight-method table, §3 the P0–P3
  order and why parameterisation was preferred to a penalty.
- `2026-09-05_control_constraint_design.zh.md` — the two command hooks (sixth revision).
- `4dTrajectory/ts_transformer/final_approach_geometry.py` — docstring: closed forms, the
  `on-final` / `faf` / truth gates, the mirrored optimizer constants.

**Results**
- `2026-09-05_final_constraint_results.zh.md` — state output: §1 arms and pre-registration,
  §2 observed floor, §3 projection, §4 bounded output, §5 penalties, §6 decisions.
- `2026-09-05_control_penalty_results.zh.md` — control penalty: §2 accuracy and violation,
  §2.2 control metrics, §2.3 training dynamics, §3 reading.
- `2026-09-06_control_hooks_results.zh.md` — hooks v1/v2, inference vs trained-through, the
  geometry backfill table, conclusions and next steps.
- `2026-09-07_latent_intent_design.zh.md` §六 L1.c — the pre-registered, unrun training-time
  corridor penalty on a teacher-free base.

**Artifacts** (validation split, outer test untouched; all under `4dTrajectory/outputs/`)
- `{KRDU,KSJC}/experiments/final_constraint_20260904/readout.{json,txt}` — arms
  `A_threshold_enu`, `A_project_on_final`, `A_project_faf`, `B_corridor_bounded`(`_s2024`),
  `C_procedure_dual`(`_s2024`), `C_procedure_fixed`; n = 2,104 / 1,666; seeds 1337 and 2024;
  2026-09-04/05.
- `{KRDU,KSJC}/experiments/control_procedure_20260905/readout.{json,txt}`, `score.txt` — arms
  `A_control_v3`, `C_control_1e-3`, `C_control_5e-3`; n = 1,404 / 1,083; seed 1337;
  2026-09-05.
- `KRDU/experiments/control_hooks_20260906/` (v1) and
  `{KRDU,KSJC}/experiments/control_hooks_v2_20260906/readout.{json,txt}`,
  `readout_geometry.{json,txt}`, `score.txt` (v2) — arms `F_barrier_infer`,
  `F_barrier_infer_soft`, `F_barrier_soft`, `R_nominal_residual`; seed 1337; 2026-09-06.
- `aeroviz-4d/public/data/airports/{KRDU,KSJC}/comparison/categories.json` — the published
  Experiments picker categories.
