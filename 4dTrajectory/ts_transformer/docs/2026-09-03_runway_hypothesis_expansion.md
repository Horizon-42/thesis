# Runway-hypothesis expansion — what the runway label is worth (2026-09-03)

Follow-up to `2026-09-03_airport_frame_ablation_results.md`, which showed that the
threshold anchor is the model's only runway knowledge. This asks what that knowledge is
worth and whether the choice can be made outside the predictor. Runner:
`run_ts_runway_hypotheses.py`; artifacts under
`4dTrajectory/outputs/{KRDU,KSJC}/experiments/runway_hypotheses_20260903/A_seed{1337,2024}/`
(`hypotheses.json`: every flight × every candidate, all selector picks; 25 MB in total).

## Method

For every validation flight of the arm-A checkpoints (threshold-anchored, both seeds) and
every runway with a published CIFP target (four at each airport):

1. clone the flight dict with that runway's `runway_target` (the data source is untouched;
   `flight_to_msl` returns a new dict);
2. build the series in that threshold's chart with the same `build_series` training used;
3. forecast with the checkpoint through the same `forecast_approaches` `predict` uses;
4. map the forecast to world coordinates and score it against the observed track in the
   TRUE runway's chart with `observed_series_metrics` (same common true-time grid).

Then apply selection rules over the K hypotheses per flight. Two are bounds, the rest are
causal (nothing from the future):

| selector | picks |
|---|---|
| assigned | the harvest label — must reproduce the baseline exactly |
| oracle_fde / oracle_ade | min over all four (an upper bound that includes picking the luckiest of K) |
| oracle_same_direction | min FDE over the assigned runway and its parallel sibling(s) |
| **oracle_mirror_control** | min FDE over the assigned runway and a **fake** sibling: the assigned threshold displaced by the real sibling's offset in the opposite direction (same separation, same course). The noise control for the line above. |
| self_consistency | the hypothesis whose forecast gets closest to the runway it was told about |
| course_gate_then_self | same, among candidates whose inbound course is within 90° of the track at the anchor |
| active_config | the most-used runway among **development-roster** landings in the 30 min before the ego's terminal-ring entry (falls back to the gate when there is none: 52 KRDU / 46 KSJC flights) |
| active_config_then_gate | active_config if it passes the course gate, else the gate |

**Chain check:** the `assigned` row equals the checkpoints' own validation predictions to
the decimetre (KRDU 1383.4 / 1163.3 and 1361.0 / 1123.7; KSJC 869.6 / 775.8 and
864.6 / 755.1 m ADE / FDE).

## KRDU (2,104 validation flights; parallel pairs 05L/05R and 23L/23R, ~1.1 km apart)

Pooled, mean ADE / mean FDE / median FDE (m) and how often the pick equals the label:

| selector | seed 1337 | acc | seed 2024 | acc |
|---|---:|---:|---:|---:|
| assigned | 1383 / 1163 / 711 | 100 % | 1361 / 1124 / 690 | 100 % |
| oracle_same_direction | 1379 / 1053 / 632 | 75 % | 1358 / 1016 / 615 | 75 % |
| oracle_mirror_control | 1387 / 1077 / 679 | 84 % | 1363 / 1032 / 656 | 83 % |
| oracle_fde (all four) | 1343 / 773 / 578 | 59 % | 1326 / 737 / 561 | 58 % |
| self_consistency | 1772 / 1535 / 1299 | 37 % | 1713 / 1412 / 1144 | 45 % |
| course_gate_then_self | 1763 / 1417 / 1167 | 41 % | 1745 / 1366 / 1068 | 46 % |
| **active_config** | **1495 / 1388 / 965** | **65 %** | **1464 / 1337 / 906** | **65 %** |
| active_config_then_gate | 1746 / 1385 / 1085 | 43 % | 1743 / 1368 / 1031 | 44 % |

By stratum (FDE mean, seed 1337 / seed 2024):

| stratum | n | assigned | active_config | oracle_same_direction | oracle_mirror_control |
|---|---:|---:|---:|---:|---:|
| straight-in | 1,273 | 643 / 622 | 849 / 800 | 572 / 545 | 604 / 577 |
| vectored | 827 | 1967 / 1900 | 2224 / 2168 | 1796 / 1745 | 1806 / 1735 |

Per assigned runway (median FDE / pick accuracy, seed 1337; seed 2024 within 40 m):

| runway | n | assigned | active_config | oracle_same_direction | oracle_mirror_control |
|---|---:|---:|---:|---:|---:|
| 05L | 471 | 764 | 887 / 83 % | 761 / 91 % | 764 / 82 % |
| 05R | 213 | 983 | **1456 / 29 %** | 948 / 78 % | 948 / 88 % |
| 23L | 470 | 789 | **1630 / 31 %** | 773 / 96 % | 633 / 56 % |
| 23R | 950 | 620 | 681 / 80 % | 487 / 57 % | 614 / 98 % |

## KSJC (1,666 validation flights; 30L/30R and 12L/12R, ~230 m apart; 87 % land 30L)

| selector | seed 1337 | acc | seed 2024 | acc |
|---|---:|---:|---:|---:|
| assigned | 870 / 776 / 439 | 100 % | 865 / 755 / 402 | 100 % |
| oracle_same_direction | 869 / 764 / 431 | 77 % | 863 / 741 / 391 | 74 % |
| oracle_mirror_control | 869 / 757 / 411 | 63 % | 864 / 743 / 388 | 71 % |
| oracle_fde (all four) | 908 / 572 / 310 | 56 % | 904 / 558 / 305 | 58 % |
| self_consistency | 1040 / 948 / 692 | 34 % | 1104 / 999 / 744 | 33 % |
| **active_config** | **871 / 772 / 438** | **94 %** | **868 / 756 / 402** | **94 %** |

Per runway, `active_config` is right on 98 % of 30L and 93 % of 12R flights and on 49 % of
30R — and the 30R flights it sends to 30L lose nothing (median FDE 468 vs 473 m).

## Reading

1. **Direction is cheap; the parallel side is the real mode.** Co-temporal landings alone
   pick the runway direction on 93 % of KSJC flights and on 80–83 % of the KRDU majority
   runways. What they cannot do is tell 05R from 05L or 23L from 23R: for KRDU's minority
   runways the rule guesses the majority sibling 70 % of the time. That guess costs the
   whole separation — 05R median FDE 983 → 1456 m, 23L 789 → 1630 m — and +19 % pooled FDE
   (+30 % on straight-in flights, where the endpoint is most of the error). At KSJC the
   same mistake costs nothing because the pair is 230 m apart and the model's own endpoint
   error is ~400 m.
2. **A K=2 sibling oracle is half luck.** The real-sibling oracle gains 79 / 75 m of median
   FDE at KRDU; the mirror-image fake sibling, at the same separation, gains 32 / 34 m. So
   only about half of a min-over-siblings score reflects runway knowledge; at KSJC the fake
   sibling gains as much as or more than the real one (28 vs 8 m). Quote any minFDE@K on
   this data against a mirror control, never bare. The four-candidate oracle's −34 % FDE is
   selection noise plus opposite-direction hypotheses and moves ADE nowhere.
3. **The predictor cannot check its own runway.** How close a forecast gets to the runway
   it was told about is useless as a selector (37–45 % at KRDU, 33 % at KSJC, FDE far worse
   than the label): a threshold-anchored forecast flies to whatever origin it is given,
   which is the flip side of the frame ablation's finding.
4. **Per-runway asymmetry.** 23L's mirror control beats its real sibling (633 vs 773 m)
   because arm A's endpoints sit 150–200 m toward the NW side of every KRDU runway in both
   directions (frame-ablation readout: 05L −145, 05R −150, 23L +198, 23R +206 m cross-track
   medians, all the same world side). A threshold displaced to the SE cancels that bias.
   The bias itself is unexplained and worth a look (CIFP path point vs flown centreline,
   or a training-set geometry effect).

## What this means for runway choice

- Keep the threshold-anchored predictor and choose the frame outside it (as the frame
  ablation concluded). The direction half of the choice is solved by an active-configuration
  cue from co-temporal landings — data the harvest already holds (`landing_time_utc`,
  `entry_time_utc`, runway); no model change.
- The parallel-side half is the multimodality that remains. At KSJC it is not worth
  resolving; at KRDU it is worth ~500–800 m of FDE on a third of the flights. Candidates:
  (a) a side classifier with ATC/procedure context (STAR transition, downwind side, aircraft
  type/operator: 05R/23L may carry a different traffic mix); (b) emit both sibling
  hypotheses and evaluate as minFDE@2 **against the mirror control**; (c) treat it as
  irreducible and report per-runway error.

## Caveats

- Validation split only, two seeds, `state` output, iTransformer. The context pool is the
  development roster (train + validation) restricted to landings before the ego's entry, so
  outer-test labels were never read; a production cue would use all traffic.
- The 30 min context window was not tuned. KRDU's 05/23 split within a window is what makes
  the majority guess wrong on minority-runway flights, not the window length.
- KSJC's validation cohort has 3 flights on 12L; its 12L/12R row is anecdotal.
