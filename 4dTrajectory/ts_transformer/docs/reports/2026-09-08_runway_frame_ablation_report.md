# Which chart the model predicts in: the runway-frame ablation, the north-west endpoint bias, and what the runway label is worth

*Experiments of 2026-09-03. Written 2026-09-08. All numbers are from the validation split;
the outer-test split has never been read.*

## Abstract

Every forecast in this package is made in a chart whose origin is the assigned runway's
threshold. That is a coordinate convention only in appearance. Moving the origin to the
airport reference point shows what it really is: **the model's entire knowledge of which
runway it is flying to**. Without it, a deterministic point predictor averages between the
two members of each parallel pair, and the cost lands on the minority runway. Handing the
model the target's coordinates as input channels does not undo any of it — the trajectory
backbone ignores them; only the duration head reads them. A second experiment then asked
what the runway label is worth and whether the choice can be made outside the model: the
*direction* can be picked from co-temporal landings at 80–94 % accuracy for free, but the
left/right of a parallel pair cannot, and at KRDU that costs 500–800 m of final displacement
error on about a third of the flights. A third measurement traced a 150–200 m north-west
offset in every KRDU endpoint to the model, not the data, and a fourth attempt to fix it by
predicting anchor-relative displacements triggered its own pre-registered veto.

---

## Question

The chart used for training and inference is a local ENU (east-north-up) frame anchored at
the assigned runway threshold. The destination is therefore always the origin, and the
position channels *are* distance-to-go. Three hypotheses were pre-registered before any
run (source: `4dTrajectory/ts_transformer/docs/2026-09-03_airport_frame_ablation_plan.md`,
"Question"):

- **H1 — target conditioning.** Threshold anchoring is a strong prior; removing it should
  degrade final displacement error, worst at parallel-runway airports, because a
  deterministic model must average between the modes a parallel pair creates.
- **H2 — route stability.** Threshold anchoring *costs* accuracy on vectored flights,
  because the same physical airspace (downwind legs, STAR transitions) lands at different
  chart coordinates depending on which runway is anchoring. An airport-fixed chart should
  help exactly the stratum the model is worst on.
- **H3 — symbolic conditioning.** An airport-centred chart *plus* the target's chart
  position fed as explicit input should recover the threshold-anchored baseline, i.e. the
  model can consume coordinates as data rather than as geometry.

The plan wrote down both readings of H3 in advance: C ≈ A means symbolic coordinates work;
A ≻ C means geometric anchoring beats symbolic conditioning, and "either is a defensible
thesis finding".

Two follow-up questions came out of the result: **what is the runway label worth, and can it
be chosen outside the predictor?** And **why do KRDU endpoints sit 150–200 m north-west of
every runway?**

Background: an earlier pooled ablation had already chosen ENU over a runway-aligned
(rotated) chart on a five-airport cohort of 19,741 flights, by 5.01 % of the registered
normalised-MSE selection objective, winning all three paired folds and all four matched
hyperparameter trials (source: `docs/coordinate_frame_ablation_results.md`, "Executive
conclusion", "Cross-validation results"; recorded 2026-07-25, one seed, outer test sealed).
That settled the *rotation*; this experiment settles the *origin*.

## Setup

### Cohort and splits

Per-airport training. Validation split only; the outer test stays sealed (source:
`2026-09-03_airport_frame_ablation_results.md`, "What was run").

| Airport | train | val | test | validation route mix | validation runways |
|---|---:|---:|---:|---|---|
| KRDU | 10,105 | **2,104** | 2,160 | 55.5 % established at the anchor, median remaining path 14.1 km | 23R 950 · 05L 471 · 23L 470 · 05R 213 |
| KSJC | 7,801 | **1,666** | 1,602 | 78.3 % established, median remaining 12.3 km | 30L 1,444 · 30R 133 · 12R 86 · 12L 3 |

Strata come from `approach_difficulty.strata_masks`: **straight-in** is route tortuosity
< 1.05, **vectored** is tortuosity ≥ 1.05 and not established at the anchor. Tortuosity is
the flown remaining path over the straight-line range from the anchor to the threshold — 1.0
is a straight-in, 2.4 a full downwind and base.

### Metrics, first use

**ADE** (average displacement error) is the mean 3-D distance between prediction and
observation on a common time grid; **FDE** (final displacement error) is that distance at the
end of the forecast. **Endpoint cross-track** is the signed lateral offset of the predicted
endpoint from the assigned runway's centreline, positive to the right of the inbound course.
**Final-time MAE** is the mean absolute error of the predicted flight duration.
**`horizonCapped`** counts forecasts that never reach a closest approach to the target within
the 600 s horizon.

### Arms

One fixed recipe across every arm — `state-v1` defaults (iTransformer, d = 256, 3 layers,
8 heads, dropout 0.1, lr 5e-4 with plateau halving, batch 2048, ≤ 180 epochs, patience 20,
checkpoint selected on fixed-anchor common-grid ADE), `full` 600 s horizon, all aircraft.
**No per-arm hyperparameter search**, deliberately: a search picking different
hyperparameters per arm would have confounded the frame with the search. 1.77 M parameters
(1.84 M with the conditioning tokens), 106–162 epochs to early stop, 14 runs in total.

| Arm | Chart origin | Target given as | Config |
|---|---|---|---|
| A | assigned runway threshold | the origin (geometry) | `coordinate_frame="enu"` / `target_conditioning="none"` |
| B | airport reference point | nothing | `coordinate_frame="airport-enu"` |
| C | airport reference point | 5 input-only channels (e/n/u of the target, cos/sin of the runway course) | `airport-enu` + `target_conditioning="channels"` |
| A + cond (control) | assigned runway threshold | the origin **and** the 5 channels | `enu` + `channels` |

A, B and C were replicated with a second model seed (2024, same split seed 1337, same
cohort) to bound the noise floor.

### The audit that had to come first

The real risk was not the new frame class but every consumer that silently equated "chart
origin" with "runway threshold". The plan listed three; the audit found two more that would
have corrupted arm B without any error: `fixed_anchor_validation`'s resampling cut every
full-horizon prediction at its closest approach to the **origin** — which is the metric that
*selects the checkpoint*, so arm B's model selection itself would have been wrong — and
`approach_difficulty` measured anchor range and cross-track from the origin, so the strata of
this very readout would have shifted under arm B. All five moved onto a new
`FlightSeries.target_chart` value, and bit-identity of the refactor was verified on 300 real
KRDU arrivals in both threshold frames (6,000 arrays, 0 differ) *before* the new frame
existed.

---

## Results

### The seed noise floor — read every margin against this first

Seed 2024 minus seed 1337, same arm, same flights, metres:

| arm | KRDU ΔADE mean / med | KRDU ΔFDE mean / med | KSJC ΔADE mean / med | KSJC ΔFDE mean / med |
|---|---:|---:|---:|---:|
| A threshold | −22 / −19 | −40 / −32 | −5 / +5 | −21 / −2 |
| B airport | −1 / −10 | −19 / −17 | **+107** / +35 | +65 / +11 |
| C airport + target | −57 / −36 | −26 / −21 | +36 / +10 | +25 / −6 |

The threshold-anchored arm is the stable one (5–22 m of pooled ADE across seeds); the
airport-anchored arms swing up to 107 m at KSJC. Any pooled margin under about 60 m at KRDU
or 100 m at KSJC is noise on this axis, whatever a paired sign test says at n ≈ 2,000. The
plan had pre-registered this reading — "read magnitudes, not p-values" — because n ≈ 1,400
gives p = 3e-16 for pure seed noise elsewhere in this package.

### H1 — supported, where it can be tested

Pooled accuracy is flat: two-seed mean ADE is 1372 (A), 1395 (B), 1382 (C) at KRDU and 868 /
832 / 879 at KSJC, all inside the noise band. The endpoint is not. Share of predicted
endpoints that lie closer to the *sibling* runway's centreline (KRDU's 05L/05R and 23L/23R
pairs are about 1.1 km apart), and the p95 of |endpoint cross-track|:

| arm | seed | median xt | p95 \|xt\| | closer to sibling |
|---|---:|---:|---:|---:|
| A threshold | 1337 | +56 | 492 | **1.5 %** |
| A threshold | 2024 | +50 | 477 | **1.5 %** |
| A + cond | 1337 | +58 | 487 | 2.3 % |
| B airport | 1337 | +57 | 916 | **13.3 %** |
| B airport | 2024 | +38 | 855 | **11.6 %** |
| C airport + target | 1337 | +39 | 840 | **15.4 %** |
| C airport + target | 2024 | +34 | 852 | **14.3 %** |

Per runway, the pattern is exactly what a deterministic model that cannot tell the pair apart
must do — each pair's endpoints collapse toward a runway-blind average sitting nearer the
majority runway. On 05R (213 flights, whose sibling 05L has 471) the median endpoint
cross-track moves from −120…−150 m under arm A to **−573…−678 m** under arms B and C, i.e.
toward 05L; on 23L (470, sibling 23R has 950) it moves from +177…+198 m to **+350…+454 m**,
toward 23R. The majority runways gain a little (23R median FDE 594–620 → 502–598) while the
minority runways lose 30–45 % of their FDE (05R 966–983 → 1052–1159; 23L 789–801 →
1113–1170). The p95 lateral endpoint error nearly doubles.

At KSJC the fingerprint is at the noise level in every arm (sibling share 7.3–13.1 %, p95
|xt| 195–293 m): 87 % of the validation cohort lands on 30L, so the parallel pair is barely
exercised, and 30L/30R are only ~230 m apart. The parallel-runway hypothesis is effectively
tested at KRDU alone.

### H2 — not supported

Vectored stratum, median ADE / median FDE, and the paired share of flights on which the arm
beats arm A seed 1337 on ADE:

| arm | seed | KRDU (n = 827) | beats A | KSJC (n = 350) | beats A |
|---|---:|---:|---:|---:|---:|
| A threshold | 1337 | 2200 / 1356 | — | 1824 / 1109 | — |
| A threshold | 2024 | 2186 / 1327 | 53 % | 1826 / 1141 | 45 % |
| B airport | 1337 | 2221 / 1529 | 48 % | **1619 / 1103** | **76 %** |
| B airport | 2024 | 2211 / 1532 | 47 % | 1895 / 1225 | 39 % |
| C airport + target | 1337 | 2284 / 1566 | 43 % | 1792 / 1180 | 56 % |
| C airport + target | 2024 | 2205 / 1442 | 51 % | 1988 / 1260 | 37 % |

At KRDU the airport frame never wins on vectored flights and loses about 75 m of median FDE
on both seeds. KSJC's seed-1337 win — 184 m better on 76 % of flights — flipped to 62 m worse
on the replicate, which beats A on only 39 % of the same flights, while arm A's own two seeds
agree to within 2 m of median ADE. Where the airport frame *does* win consistently is the
**easy** stratum at KRDU (straight-in flights gain ~100 m of median FDE, better on 61–64 % on
both seeds) — which is the opposite of H2's mechanism, and is bought with the minority-runway
losses above.

### H3 — not supported for the trajectory, supported for the duration head

Arm C reproduces arm B's endpoint numbers to within seed noise on every measure. The one
thing the conditioning moves, on both airports and both seeds, is the flight duration:

| arm | KRDU final-time MAE (s) | KSJC |
|---|---:|---:|
| A / A seed 2024 | 33.5 / 33.0 | 24.5 / 25.1 |
| B / B seed 2024 | 32.6 / 32.4 | 24.9 / 24.4 |
| C / C seed 2024 | **29.9 / 29.5** | **22.0 / 22.3** |
| A + cond | **30.8** | **22.0** |

About −10 %. The flattening duration head reads the target position; the variate-token
attention backbone that places the trajectory does not.

The control arm (threshold frame **plus** the channels) is inert for the trajectory, as
predicted: KRDU 1401 / 1193 against A's 1361–1383 / 1124–1163, sibling share 2.3 % against
1.5 %. `horizonCapped` counts stayed low and comparable (KRDU: A 0–6, control 8, B 10–19,
C 12–13 of 2,104; KSJC ≤ 2 everywhere), confirming the label refactor did not change the
cohort.

### The KRDU north-west endpoint bias

Every arm-A per-runway cross-track median at KRDU is on the same world side: 05L −145,
05R −150, 23L +198, 23R +206 m. Follow-up measurement on the arm-A validation records of both
seeds, no retraining (source: `2026-09-03_krdu_nw_endpoint_bias.md`):

- **Not the data.** The last observed sample before the threshold crossing sits on the CIFP
  centreline (cross-track medians −2 / −2 / +1 / +1 m); the CIFP path point differs from the
  NASR threshold by ≤ 6.5 m on every KRDU runway; KSJC's observed and predicted endpoints are
  both within ±12 m of their centrelines.
- **Not vectored flights.** Median lateral miss of the predicted endpoint is +204 m for
  flights established at the anchor and +24 m for the rest — the bias lives in the *easy*
  stratum.
- **A translation, not a rotation.** The first predicted point is already 235–348 m off the
  aircraft's actual position and the offset stays roughly constant along a final parallel to
  the true one (heading error ≤ 1.3°). Left for 05, right for 23: the north-west side both
  times. Seed 2024 reproduces every number within 40 m.
- **World-fixed, not shrinkage.** The first-step jump projected on the world north-west unit
  vector is positive in all four anchor quadrants (NE +236, NW +332, SE +221, SW +135 m),
  while the projection toward the normalizer mean flips sign by quadrant. KSJC: +68 m. Even
  noise-free synthetic straight-in inputs through the same checkpoint jump +317 m (05L/05R)
  and +150 m (23L/23R) north-west.
- **Why north-west.** 63 % of KRDU anchors lie south-east of their runway's extended
  centreline (05L 263/471, 05R 121/213, 23L 368/470, 23R 593/950), so those flights must
  displace north-west to join the final. The observed displacement relative to the
  anchor-heading extrapolation has median +4 m but **mean +192 m** north-west at +60 s. An
  MSE-trained deterministic model learns conditional means, and its weakly determined
  residual lateral offset takes the sign of that skew.
- **Why the loss lets it stand.** A 300 m constant lateral offset on a straight-in flight
  costs (300 / 10,000)² ≈ 9e-4 per supervised point, against a pooled objective of ~0.08
  dominated by kilometre-scale errors on vectored flights; established flights are only about
  5 % of the position loss.

Every quoted per-runway cross-track median for arm A carries this ~250 m translation.

### What the runway label is worth

No training. For each validation flight and each of the four runways with a published CIFP
target: clone the flight with that runway's target, build the series in *that* threshold's
chart with the same `build_series` training uses, forecast with the same checkpoint, map back
to world coordinates, and score against the observation in the **true** runway's chart. Then
apply selection rules over the K hypotheses (runner: `run_ts_runway_hypotheses.py`; artifact:
`4dTrajectory/outputs/{KRDU,KSJC}/experiments/runway_hypotheses_20260903/A_seed{1337,2024}/hypotheses.json`).
Chain check: the `assigned` selector reproduces the checkpoints' own validation numbers to
the decimetre (KRDU 1383.4 / 1163.3 and 1361.0 / 1123.7; KSJC 869.6 / 775.8 and 864.6 / 755.1
m ADE / FDE).

KRDU, pooled mean ADE / mean FDE / median FDE and how often the pick equals the label:

| selector | seed 1337 | acc | seed 2024 | acc |
|---|---:|---:|---:|---:|
| assigned (the label) | 1383 / 1163 / 711 | 100 % | 1361 / 1124 / 690 | 100 % |
| oracle over the same-direction sibling | 1379 / 1053 / 632 | 75 % | 1358 / 1016 / 615 | 75 % |
| **mirror-image fake sibling (noise control)** | 1387 / 1077 / 679 | 84 % | 1363 / 1032 / 656 | 83 % |
| oracle over all four | 1343 / 773 / 578 | 59 % | 1326 / 737 / 561 | 58 % |
| self-consistency | 1772 / 1535 / 1299 | 37 % | 1713 / 1412 / 1144 | 45 % |
| **active configuration (co-temporal landings)** | 1495 / 1388 / 965 | **65 %** | 1464 / 1337 / 906 | **65 %** |

The `active_config` selector takes the most-used runway among *development-roster* landings
in the 30 minutes before the ego flight's terminal-ring entry — causal, reads no future, and
the harvest already holds the data. At KSJC it is right on 94 % of flights and costs nothing
(pooled 871 / 772 against the label's 870 / 776), because the pair is 230 m apart and the
model's own endpoint error is ~400 m. At KRDU it gets the *direction* right on 80–83 % of the
majority runways but guesses the majority sibling about 70 % of the time on the minority ones
— 05R accuracy 29 %, 23L 31 % — and that guess costs the whole separation: 05R median FDE
983 → 1456 m, 23L 789 → 1630 m, and +19 % pooled FDE (+30 % on straight-in flights).

The **mirror control** is the methodological point. A K = 2 sibling oracle at KRDU gains
79 / 75 m of median FDE; a *fake* sibling at the same separation and course but displaced to
the opposite side gains 32 / 34 m. So about half of any min-over-siblings score is selection
luck, not runway knowledge. At KSJC the fake sibling gains as much as or more than the real
one (28 vs 8 m) — all luck. Any minFDE@K on this data must be quoted against a mirror
control.

Two further readings: the predictor **cannot check its own runway** (how close a forecast gets
to the runway it was told about picks correctly 37–45 % at KRDU and 33 % at KSJC, with far
worse FDE than the label) — the flip side of the frame ablation's finding, since a
threshold-anchored forecast flies to whatever origin it is given. And 23L's mirror control
beats its real sibling (633 vs 773 m median FDE) precisely because the fake threshold,
displaced south-east, cancels the north-west bias.

---

## Reading

1. **The threshold anchor is not a convention, it is the model's runway identity.** Remove it
   and the deterministic point predictor averages between the modes a parallel pair creates,
   and the cost lands on the minority runway — exactly the flight the anchor was
   disambiguating.
2. **Geometric anchoring beats symbolic conditioning.** Of the plan's two pre-written
   readings of H3, this is the one the data support. Coordinates handed to the model as
   numbers are consumed only by the part of the network that flattens its input.
3. **The route-stability argument did not survive a second seed** on either airport. This is
   the clearest illustration in the campaign of why single-seed margins are not read here.
4. **Runway choice belongs outside the predictor.** The direction half is solved cheaply and
   causally by co-temporal landings; the parallel-side half is genuine multimodality that a
   deterministic single-output model cannot represent, and it is worth 500–800 m of FDE on
   about a third of KRDU's flights.
5. **The model has two weak ends and one prior each.** Absolute chart output carries the
   "endpoint is at the origin" prior and gets the start wrong (the north-west translation);
   anchor-relative output carries the "start where the aircraft is" prior and loses the
   endpoint. The next output parameterisation has to hold both.
6. **Multi-airport pooling is not addressed by any of this.** Each airport still gets its own
   chart; the airport-anchored frame only makes the *within*-airport chart runway-independent,
   which is a prerequisite, not the solution. The plan warned against overselling arm B as the
   pooling route, and the result gives no reason to.

## Caveats

- Two seeds per arm, one recipe, `state` output only, iTransformer only. PatchTST is
  structurally unable to use arm C (channel-independent backbone) and was not run for A or B;
  control-output arms were left as a follow-up.
- The parallel-runway hypothesis is tested at KRDU only; KSJC's validation cohort has 3
  flights on 12L and 133 on 30R.
- Everything was scored on validation; the outer test stays sealed, so no held-out
  generalisation claim is made. The earlier pooled ENU-vs-aligned ablation made none either —
  its report states the test set was untouched and supports a *selection decision* only.
- The 30-minute context window for the active-configuration cue was not tuned; what makes the
  majority guess wrong on minority-runway flights is KRDU's 05/23 split within a window, not
  the window length. Its context pool is the development roster (train + validation)
  restricted to landings before the ego flight's entry, so outer-test labels were never read;
  a production cue would use all traffic.
- The runway-hypothesis run produces **per-flight metric tables** (`hypotheses.json`: every
  flight × every candidate, plus every selector's pick), not trajectory records — there is
  nothing to publish to the viewer as trajectories, which is why it does not appear in the
  Experiments picker. Nothing from the frame ablation was published at the time either (disk
  was at 98 %); its arms were published later, see below.
- Fourteen pre-existing test failures in `trajectory_data_process/tests/` were reproduced at
  the pre-change commit and recorded separately; they are unrelated to these arms.

## What happened next

**Decision.** `coordinate_frame="enu"` (threshold-anchored) stays the default. The
airport-anchored chart is retained as a diagnostic of what the anchor buys, not as a pooling
route. The conditioning channels are worth keeping only for the duration head. Runway
assignment is treated as a scheduling-layer or controller input, not something the predictor
resolves (literature index: `docs/literature/runway_assignment/README.md`).

**The attempted fix, and its veto.** A `state-v2` candidate predicted positions as
displacements from the anchor (`state_position_reference="anchor-relative"`), two airports ×
two seeds, paired with arm A. It fixed the start — first-step lateral jump 250–350 → 0–7 m,
straight-in FDE 643/622 → 492/499 m at KRDU, pooled ADE 1383/1361 → 1322/1323 — and broke the
vectored stratum: FDE 1967/1900 → 2317/2278 m, worse on 78–82 % of flights, on **both** seeds.
That is the pre-registered veto condition, and it fired. The default stayed `absolute`
(source: `2026-09-03_runway_frame_experiments_index.zh.md`, experiment 4).

**The follow-up that ran instead.** The "runway-scale endpoint cross-track term" that the bias
diagnosis and the `state-v2` document both asked for became the final-approach corridor
constraint of 2026-09-04/05 — the bounded output layer, which cut KRDU's endpoint cross-track
p95 from 492 to 305 m and moved the established-flight endpoint median from +206 m to +54 m,
the corridor edge. It held the bias against a wall; it did not remove it.

**Open.** The parallel-side choice at KRDU; an output parameterisation that keeps the
start-of-path and end-of-path priors at once; multi-airport pooling.

**Publication.** On 2026-09-07 the frame-ablation arms were published to the frontend's
Experiments picker under the heading `airport_frame_20260903` — six categories per airport
(`A_threshold_enu`, `A_threshold_enu_target`, `B_airport_enu`, `C_airport_enu_target`, and the
two inference-projection variants of arm A), at both KRDU and KSJC, 2,104 and 1,666 groups
respectively (artifact:
`aeroviz-4d/public/data/airports/{KRDU,KSJC}/comparison/categories.json`, written
2026-09-07 09:51 UTC). The `runway_hypotheses_20260903` campaign is not among them, for the
reason given in the caveats.

## Sources

Document paths are relative to `4dTrajectory/ts_transformer/docs/`.

**Index — start here**
- `2026-09-03_runway_frame_experiments_index.zh.md` — the four experiments in causal order,
  the through-line, and the "settled vs open" table.

**Plan and results**
- `2026-09-03_airport_frame_ablation_plan.md` — "Question" (H1–H3), "Arms", "Phase 1 — the
  origin-⇔-threshold audit", "Phase 5 — readout (decide these BEFORE looking)",
  "Risks / traps".
- `2026-09-03_airport_frame_ablation_results.md` — "Headline", "What was run", "Phase 1
  audit", "Seed noise floor", "Pooled accuracy", "H1 …", "H2 …", "Control arm", "Other
  pre-registered checks", "Not done / caveats".
- `2026-09-03_krdu_nw_endpoint_bias.md` — "Verdict", "Evidence", "What follows".
- `2026-09-03_runway_hypothesis_expansion.md` — "Method", the KRDU and KSJC selector tables,
  "Reading", "What this means for runway choice", "Caveats".
- `coordinate_frame_ablation_results.md` — the earlier pooled ENU vs runway-aligned ablation:
  "Executive conclusion", "Experimental design and leakage controls", "Cross-validation
  results", "Limitations and threats to validity".
- `run_ts_runway_hypotheses.py` (repo root) — docstring: the four-step per-candidate
  procedure, the selector list, the development-scope statement.

**Artifacts** (validation split, outer test untouched)
- `4dTrajectory/outputs/{KRDU,KSJC}/experiments/airport_frame_20260903/` — arms
  `A_threshold_enu`, `A_threshold_enu_target`, `B_airport_enu`, `C_airport_enu_target`, each
  with a `_s2024` replicate for A/B/C, the matching `*_pred_val/` directories (0.98 GB KRDU,
  0.66 GB KSJC), and the campaign `readout.json` from `docs/compare_frame_arms.py`. Seeds
  1337 and 2024; n = 2,104 / 1,666; 2026-09-03.
- `.../experiments/runway_hypotheses_20260903/A_seed{1337,2024}/hypotheses.json` — every
  flight × every candidate runway plus every selector's pick; 6.8 MB each; 2026-09-03.
  Metric tables, not trajectory records.
- `aeroviz-4d/public/data/airports/{KRDU,KSJC}/comparison/categories.json` — the published
  Experiments picker categories.
