# Airport-center frame ablation — results (2026-09-03)

Plan: `2026-09-03_airport_frame_ablation_plan.md`. Readout script (the pre-registered
tables, regenerable from the artifacts): `docs/compare_frame_arms.py`, whose JSON output
sits beside each campaign as `readout.json`. Every number below is from the **validation
split** (outer-test untouched); arms are paired flight-by-flight on identical persisted
splits and data provenance (verified per checkpoint before any A-vs-B number was read).

## Headline

- **H1 (the threshold anchor is target conditioning) — supported, where it can be tested.**
  At KRDU, removing the anchor makes the deterministic model average across each parallel
  pair: predicted endpoints closer to the sibling runway rise from **1.5 % to 12–15 %** on
  both seeds, the minority runway of each pair is pulled **570–680 m** toward its majority
  sibling, and that runway's median FDE rises 30–45 %. Feeding the target's coordinates as
  input channels (arm C) does **not** undo any of it.
- **H2 (an airport-fixed frame helps vectored flights) — not supported.** On the vectored
  stratum the airport-anchored arms are no better than the baseline at KRDU on either seed
  (FDE worse by ~75 m median), and KSJC's one-seed win (184 m better on 76 % of flights)
  flipped to 62 m worse on the second seed. The threshold-anchored baseline is the
  seed-stable arm.
- **H3 (coordinates as data recover the geometric anchor) — not supported for the
  trajectory; supported only for the duration head.** Arm C is indistinguishable from arm B
  on every endpoint measure, but every conditioned arm (C and the A+conditioning control)
  lowers final-time MAE by ~10 % on both airports and both seeds — the flattening duration
  head reads the target position, the attention backbone does not.
- **Decision:** keep `coordinate_frame="enu"` (threshold-anchored) as the default. The
  airport-anchored chart is a diagnostic of what the anchor buys, not a pooling route; the
  conditioning channels are worth keeping only for the duration head.

## What was run

One fixed recipe across every arm — TSConfig `state-v1` defaults (iTransformer d=256,
3 layers, 8 heads, dropout 0.1, lr 5e-4 with plateau halving, batch 2048, ≤180 epochs,
patience 20, fixed-anchor common-grid ADE selection), `full` horizon (600 s), `all`
aircraft. **No per-arm cross-validation**: a search selecting different hyperparameters
per arm would have confounded the frame with the search. 1.77 M parameters (1.84 M with
the five conditioning tokens); 0.5–1.2 s per epoch on the RTX 4060, 106–162 epochs to
early stop. Campaign runner: `run_ts_frame_ablation.py`; arm declarations under
`docs/experiments/airport_frame_*.json`.

| Arm | Chart origin | Target given as | `coordinate_frame` / `target_conditioning` |
|---|---|---|---|
| A | assigned runway threshold | the origin (geometry) | `enu` / `none` |
| B | airport reference point | nothing | `airport-enu` / `none` |
| C | airport reference point | 5 input-only channels (`e/n/u_tgt`, `cos/sin ψ_rwy`) | `airport-enu` / `channels` |
| A+cond (control) | assigned runway threshold | the origin AND the 5 channels | `enu` / `channels` |

Seed replicates (`*_s2024`: model seed 2024, split seed 1337, same cohort) for A, B, C on
both airports bound the noise floor. 14 runs in all.

| Airport | train | val | test | val route mix | val runways |
|---|---:|---:|---:|---|---|
| KRDU | 10,105 | 2,104 | 2,160 | 55.5 % established at anchor, median remaining path 14.1 km | 23R 950 · 05L 471 · 23L 470 · 05R 213 |
| KSJC | 7,801 | 1,666 | 1,602 | 78.3 % established, median remaining 12.3 km | 30L 1,444 · 30R 133 · 12R 86 · 12L 3 |

Artifacts: `4dTrajectory/outputs/{KRDU,KSJC}/experiments/airport_frame_20260903/` —
`<arm>/` checkpoint + history + fit evaluation + experiment manifest, `<arm>_pred_val/`
records + `summary.json` + evaluation report (0.98 GB KRDU, 0.66 GB KSJC). Nothing was
published to the viewer (disk at 98 %).

## Phase 1 audit (before any run)

The plan's consumer list was a floor. Two more consumers equated the origin with the
threshold and would have corrupted arm B silently:

- `fixed_anchor_validation.resample_prediction_to_physical_time` cut every full-horizon
  prediction at its closest approach to the **origin** — this is the common-grid ADE that
  selects the checkpoint, so arm B's model selection itself would have been wrong.
- `approach_difficulty` measured `anchor_range_m` and the cross-track from the origin — the
  strata of this very readout would have shifted under arm B.

Both moved onto `FlightSeries.target_chart` with the three the plan named (crossing plane,
inference truncation, `horizontal_distance_m`). Bit-identity of the refactor was verified
on 300 real KRDU arrivals in both threshold frames (6,000 arrays, 0 differ) before the new
frame existed. The control path's terminal-loss axis table and the pipeline's frame tag
gained the new key. Normalizer sanity (plan 2.4): under `airport-enu` the KRDU position
spread grows 9,154 → 9,737 m (e) and 9,172 → 9,793 m (n); velocity statistics identical.
Under arm B the observed threshold crossing lands at the same time as under arm A to
5 µs (400 KSJC arrivals).

## Seed noise floor (read every margin against this first)

Seed 2024 minus seed 1337, same arm, same flights (m):

| arm | KRDU ΔADE mean / med | KRDU ΔFDE mean / med | KSJC ΔADE mean / med | KSJC ΔFDE mean / med |
|---|---:|---:|---:|---:|
| A threshold | −22 / −19 | −40 / −32 | −5 / +5 | −21 / −2 |
| B airport | −1 / −10 | −19 / −17 | **+107** / +35 | +65 / +11 |
| C airport + target | −57 / −36 | −26 / −21 | +36 / +10 | +25 / −6 |

The threshold-anchored arm is the stable one (5–22 m pooled ADE across seeds); the
airport-anchored arms swing up to 107 m at KSJC. Any pooled margin under ~60 m (KRDU) or
~100 m (KSJC) is noise on this axis, whatever a paired sign test says at n ≈ 2,000.

## Pooled accuracy (validation, both seeds)

Mean ADE / mean FDE in metres; two-seed means in brackets.

| arm | KRDU seed 1337 | KRDU seed 2024 | KRDU mean | KSJC seed 1337 | KSJC seed 2024 | KSJC mean |
|---|---:|---:|---:|---:|---:|---:|
| A threshold | 1383 / 1163 | 1361 / 1124 | [1372 / 1144] | 870 / 776 | 865 / 755 | [868 / 766] |
| B airport | 1395 / 1201 | 1395 / 1182 | [1395 / 1192] | 778 / 718 | 885 / 783 | [832 / 751] |
| C airport + target | 1410 / 1241 | 1353 / 1215 | [1382 / 1228] | 860 / 772 | 897 / 797 | [879 / 785] |
| A + cond (control) | 1401 / 1193 | — | | 916 / 767 | — | |

ADE is flat everywhere within the seed floor. FDE at KRDU degrades in the airport frame on
both seeds (B +48 m, C +84 m on the two-seed means, i.e. 4–7 %, against a 40 m seed swing);
at KSJC every FDE sits inside the airport frame's own swing. Final-time MAE (s), the one
metric the conditioning moves on both airports and both seeds:

| arm | KRDU | KSJC |
|---|---:|---:|
| A / A_s2024 | 33.5 / 33.0 | 24.5 / 25.1 |
| B / B_s2024 | 32.6 / 32.4 | 24.9 / 24.4 |
| C / C_s2024 | **29.9 / 29.5** | **22.0 / 22.3** |
| A + cond | **30.8** | **22.0** |

## H1 — the endpoint against the ASSIGNED runway centreline (KRDU)

Cross-track of the predicted endpoint from the assigned runway's centreline (+ = right of
the inbound course), and the share of endpoints that lie closer to the parallel sibling's
centreline (05L/05R and 23L/23R are ~1.1 km apart):

| arm | seed | median | p25 / p75 | p95 \|cross\| | closer to sibling |
|---|---:|---:|---:|---:|---:|
| A threshold | 1337 | +56 | −59 / +272 | 492 | 1.5 % |
| A threshold | 2024 | +50 | −59 / +234 | 477 | 1.5 % |
| A + cond | 1337 | +58 | −62 / +274 | 487 | 2.3 % |
| B airport | 1337 | +57 | −83 / +154 | **916** | **13.3 %** |
| B airport | 2024 | +38 | −73 / +118 | **855** | **11.6 %** |
| C airport + target | 1337 | +39 | −88 / +129 | **840** | **15.4 %** |
| C airport + target | 2024 | +34 | −76 / +104 | **852** | **14.3 %** |

Per runway (endpoint cross-track median / FDE median, m; n in the val split):

| runway | A 1337 | A 2024 | B 1337 | B 2024 | C 1337 | C 2024 |
|---|---:|---:|---:|---:|---:|---:|
| 05L (471) | −145 / 764 | −129 / 782 | −64 / 798 | −58 / 802 | −52 / 812 | −48 / 679 |
| 05R (213) | −150 / 983 | −120 / 966 | **−642 / 1144** | **−573 / 1159** | **−631 / 1069** | **−678 / 1052** |
| 23L (470) | +198 / 789 | +177 / 801 | **+353 / 1139** | **+350 / 1113** | **+454 / 1170** | **+354 / 1153** |
| 23R (950) | +206 / 620 | +148 / 594 | +67 / 520 | +57 / 502 | +58 / 592 | +45 / 598 |

The pattern is the one H1 predicted for a deterministic model that cannot tell the runways
apart: each pair's endpoints collapse toward a runway-blind average that sits nearer the
majority runway. 05R (213 flights, sibling 05L has 471) is pulled ~600 m left toward 05L;
23L (470, sibling 23R has 950) ~350–450 m right toward 23R; the majority runways gain a
little (23R FDE 594–620 → 502–598) while the minority ones lose 30–45 % of their FDE. The
p95 lateral endpoint error almost doubles. Arm C reproduces arm B's numbers to within seed
noise: the model does not use the target coordinates it is given to place the endpoint.

At KSJC the fingerprint is at the noise level (sibling share A 7.3–9.1 %, B 7.6–12.0 %,
C 9.1–13.1 %; p95 |cross| 195–293 m for every arm): the validation cohort is 87 % runway
30L, so the parallel pair is barely exercised, and 30L/30R are only ~230 m apart.

## H2 — the vectored stratum (tortuosity ≥ 1.05, not established at the anchor)

Median ADE / median FDE (m) and the paired share of flights on which the arm beats arm A
(seed 1337) on ADE:

| arm | seed | KRDU (n = 827) | beats A | KSJC (n = 350) | beats A |
|---|---:|---:|---:|---:|---:|
| A threshold | 1337 | 2200 / 1356 | — | 1824 / 1109 | — |
| A threshold | 2024 | 2186 / 1327 | 53 % | 1826 / 1141 | 45 % |
| B airport | 1337 | 2221 / 1529 | 48 % | **1619 / 1103** | **76 %** |
| B airport | 2024 | 2211 / 1532 | 47 % | 1895 / 1225 | 39 % |
| C airport + target | 1337 | 2284 / 1566 | 43 % | 1792 / 1180 | 56 % |
| C airport + target | 2024 | 2205 / 1442 | 51 % | 1988 / 1260 | 37 % |

At KRDU the airport frame never wins on vectored flights and loses ~75 m of median FDE on
both seeds. At KSJC the seed-1337 arm B win was seed luck: the replicate is worse than A
on 61 % of the same flights, while arm A's own two seeds agree to within 2 m of median ADE.
Where the airport frame does win consistently is the **easy** stratum at KRDU: straight-in /
established flights gain ~100 m of median FDE (better on 61–64 %, both seeds) and the
endpoint median drops 506 → 389–453 m — most of those flights land on the majority runway
of a pair, where the runway-blind average is close to right. That is the opposite of H2's
mechanism (a location-stable downwind), and it is bought with the minority-runway losses
above.

## Control arm (threshold frame + conditioning)

Under a threshold-anchored frame the position part of the conditioning is one constant per
run, but `cos/sin ψ_rwy` still varies by runway, so the control is "A plus the runway course
as data", not a pure no-op. It stays at arm A's level on everything but the duration head:
KRDU ADE/FDE 1401/1193 (A: 1361–1383 / 1124–1163), sibling share 2.3 % (A: 1.5 %); KSJC
916/767 (A: 865–870 / 755–776). The mechanism is harmless and, for the trajectory, inert.

## Other pre-registered checks

- `horizonCapped` (forecast never reaches a closest approach within 600 s): KRDU A 0–6,
  control 8, B 10–19, C 12–13 of 2,104; KSJC ≤ 2 in every arm. A handful more runway-blind
  forecasts never close on the target; strata are defined from observed covariates and are
  unaffected.
- Checkpoint selection under `airport-enu` used the target-relative truncation (Phase 1);
  with the pre-refactor code it would have cut predictions at the airport reference point.
- Flyability and gate verdicts were not read: the state output carries no flyability
  guarantee by design, and the gates are unchanged in kind from the baseline generation.

## Interpretation, written against the plan's pre-registered readings

The threshold anchor is not a coordinate convention; it is the model's only runway
identity. Take it away and the deterministic point predictor does what the plan's H1 said
it must — averages between the modes the parallel pair creates — and the cost lands on the
minority runway of each pair, which is exactly the flight the anchor was disambiguating.
The "route stability" argument for an airport-fixed chart (H2) did not survive a second
seed on either airport. And giving the model the target's coordinates as numbers (H3) is
consumed only by the part of the network that flattens its input (the duration head,
−10 % time MAE) and not by the variate-token attention that places the trajectory: the
plan's alternative reading — *geometric anchoring beats symbolic conditioning* — is the
one the data support. Multi-airport pooling is not addressed by any of this (each airport
still gets its own chart), as the plan warned.

## Not done / caveats

- Two seeds per arm, one recipe, `state` output only, iTransformer only (PatchTST cannot
  use arm C and was not run for A/B). Control-output arms remain a follow-up.
- KSJC's validation cohort has 3 runway-12L flights and 133 30R, so the parallel-runway
  hypothesis is effectively tested at KRDU only.
- Both airports' predictions were scored on the validation split; outer-test stays sealed.
- The 14 pre-existing failures in `trajectory_data_process/tests/test_ts_pipeline.py` and
  `test_download_landings.py` (label grammar of 2026-08-24, fixtures without rosters) were
  reproduced at the pre-change commit and are recorded in `docs/code-health-followups.md`.
