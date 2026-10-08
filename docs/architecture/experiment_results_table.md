# Results of the three stages of the two-tier model

Source: the output directories on SP-AI (`4dTrajectory/outputs/POOLED/`), read 2026-10-08. LaTeX version:
`experiment_results_table.tex`.

## Table 1 — Stage A: the executor flies the closed-loop sentences of observed flights

Landing rate (the judge's outcome *landed*) at three row intervals Δ; the observed flights are the reference.

| Row interval Δ | Train sample: flights | Landed (%) | Select days: flights | Landed (%) |
|---|---|---|---|---|
| 2 s | 1,999 | 97.65 | 6,199 | 97.23 |
| **4 s (used)** | 1,999 | 97.45 | 6,199 | 97.08 |
| 8 s | 1,999 | 96.85 | 6,198 | 96.43 |

## Table 2 — Stage B: landing rate of the prior in free generation

200 aircraft × 2 samples per airport, single aircraft, no traffic. Base model: trained on all five airports, read once on the
validation days. Leave-one-airport-out: each airport is read with a model of the selected configuration that was trained on the
other four airports only, so the airport was never seen in training; it is read on the select days of that airport. The base
model's teacher-forced loss on the validation days is 1.084 per step.

| Airport | Base model (val days): sentences | Landed (%) | Leave-one-airport-out (select days): sentences | Landed (%) |
|---|---|---|---|---|
| KMSY | 370 | 93.0 | 360 | 91.7 |
| KRDU | 380 | 96.6 | 380 | 83.4 |
| KSJC | 382 | 95.8 | 388 | 90.7 |
| KSMF | 382 | 94.5 | 366 | 75.7 |
| KSTL | 380 | 93.2 | 374 | 71.4 |
| **All** | 1,894 | **94.6** | 1,868 | **82.6** |

## Table 3 — Stage C: post-training in windows of recorded traffic

Read on the same 1,000 select windows (200 per airport) with the same random numbers. *Landed*: the window ends in *landed*.
*Lost*: the commanded aircraft loses separation. *Reward*: mean reward (1 for a landing without a go-around on a runway of the
present landing direction, 0.9ⁿ after n go-arounds, 0 for every other outcome and every loss of separation). Each line gives its
best round (by landing rate). Differences below about 1.2 points are within the noise of 1,000 windows.

| Method | Start | Rounds | Landed (%) | Lost (%) | Reward |
|---|---|---|---|---|---|
| Base model, no post-training | — | — | 79.5 | 14.9 | 0.787 |
| Branch training (K = 8) | base | 14 | 85.7 | 11.7 | 0.849 |
| Branch training, K = 16 | round 8 | 4 | 85.8 | 11.5 | 0.849 |
| Branch training, 3× learning rate | round 8 | 6 | 85.8 | 11.3 | 0.847 |
| Branch training, K = 16, 3× learning rate | round 8 | 4 | 85.6 | 11.8 | 0.849 |
| Training on the landed sentences | round 8 | 4 | 85.1 | 11.3 | 0.843 |
| Value function (advantage per row) | round 8 | 6ᵃ | 84.3 | 12.5 | 0.834 |
| Branch training, credit per 60 s segment | round 8 | 8 | **87.7** | 9.9 | **0.869** |
| Branch training, credit per 32 s segment | segment-60, round 5 | 4 | **87.9** | **9.5** | **0.869** |

"Round 8" is round 8 of the first branch-training run. ᵃ The first two rounds only train the value function (the model does not
move). A model that says a landing sentence in at least one of 32 draws lands in 99.3 % of the windows (base and round 8:
99.3 / 99.4 %), so the gap to the one-draw rates above is probability, not capacity. Select days only; the validation days are
read once, for the round that is chosen.
