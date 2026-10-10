# Statistics of the dataset of the two-tier model

Source: `4dTrajectory/outputs/POOLED/instruction_language/v12_20261005` and `data/day_split_20260924.json` on SP-AI, read
2026-10-08. LaTeX version: `dataset_table.tex`.

## Table 1 — The split of the data

Flights are split by operating day, so that no day is in two splits. The test days are sealed: they are not labelled, counted or
read before the end. *Flown*: flights whose sentence the executor can fly (the others have no aircraft dynamics).

| Split | Operating days | Labelled flights | Refused | Flown (closed loop) |
|---|---|---|---|---|
| Train | 53 | 44,338 | 365 | 40,555 |
| Select | 9 | 6,837 | 76 | 6,201 |
| Val | 14 | 10,649 | 97 | 9,756 |
| Test | 14 | sealed | | |
| **Total** | 90 | 61,824 | 538 | 56,512 |

## Table 2 — The five airports

*E*: published field elevation. Candidates: runway ends with a published vertical path (no flight decides a candidate).
Instructions per flight: words said after the first row, mean over the train flights.

| Airport | E (m) | Candidates | Train flights | Select flights | Instructions per flight | Go-around flights (train / select) |
|---|---|---|---|---|---|---|
| KMSY | 1.2 | 4 | 4,862 | 670 | 29.3 | 12 / 0 |
| KRDU | 132.6 | 5 | 14,274 | 2,308 | 33.8 | 29 / 11 |
| KSJC | 18.9 | 4 | 10,027 | 1,683 | 14.9 | 1 / 2 |
| KSMF | 8.2 | 4 | 5,602 | 882 | 38.9 | 5 / 2 |
| KSTL | 188.4 | 8 | 9,573 | 1,294 | 37.9 | 33 / 3 |
| **All** | — | — | 44,338 | 6,837 | 30.6 | 80 / 18 |

## Table 3 — The two kinds of approach

A straight-in flight is on the final with few turns; a vectored flight is turned onto the final by the controller.

| Approach | Train flights (%) | Select flights (%) | Instructions per flight (train) |
|---|---|---|---|
| Straight-in | 26,721 (60.3) | 3,924 (57.4) | 10.8 |
| Vectored | 17,617 (39.7) | 2,913 (42.6) | 60.7 |
