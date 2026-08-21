# Why KSJC's ADE/FDE look better than everyone else's

**Date:** 2026-08-21 · **Data:** five-airport harvest (42 725 rostered arrivals),
pooled all-airport checkpoints in `4dTrajectory/outputs/<ICAO>/ts_pred_pooled_*`
· **Scripts:** analysis run ad hoc; the measurements below are reproduced by the
covariates now written into every prediction row (see "What changed").

## Summary

KSJC's arrival data is **not truncated and not short of samples — it is straight**. Its
apparent 1.7× accuracy advantage is a composition effect: 78 % of its evaluation flights
are already established straight-in at the prediction anchor, against 41–61 % elsewhere.
Standardise the route mix and KSJC becomes the **worst** of the five airports, not the best.

Separately and much smaller, 64 KSJC "arrivals" (0.57 %) were not arrivals at all: they are
takeoffs from neighbouring airfields inside the terminal-entry ring. That is a real defect
in `arrival_segment.py`, now fixed.

## The observation

Pooled all-airport checkpoint, iTransformer, normalized-time horizon, validation split:

| airport | n | ADE median | FDE median | cross-track p95 (median) | final-time MAE |
|---|---|---|---|---|---|
| KRDU | 2104 | 814 m | 1019 m | 408 m | 37.4 s |
| **KSJC** | 1675 | **483 m** | **1000 m** | **166 m** | **26.1 s** |
| KSMF | 581 | 1117 m | 1465 m | 1034 m | 45.6 s |
| KSTL | 1223 | 973 m | 1127 m | 783 m | 36.1 s |
| KMSY | 622 | 1153 m | 1224 m | 1314 m | 43.2 s |

The tell is already here: ADE and cross-track are 1.7–8× better at KSJC, but **FDE median
is not better at all** (1000 m against KRDU's 1019 m). The advantage lives entirely in the
lateral channel — which is exactly what a straight route makes trivial. PatchTST shows the
same pattern; train and validation splits agree to within a few percent, so it is not
overfitting either.

## 1. It is not a reception or slicing problem

KSJC has the **best** data of the five:

| airport | tracks starting at the 30 km crop edge | max sample gap p50 | coverage gaps / 400 tracks |
|---|---|---|---|
| KRDU | 99.5 % | 2.9 s | 0 |
| **KSJC** | **99.0 %** | **2.1 s** | **1** |
| KSMF | 99.2 % | 3.3 s | 0 |
| KSTL | 99.0 % | 4.9 s | 5 |
| KMSY | 98.8 % | 6.6 s | 4 |

The arrival slice cuts a median of ~5.1 km — the 25–30 km annulus — at every airport alike.
The re-entry case that could eat a downwind (a track leaving and re-entering the ring, so
the "final entry" cut lands late) fires on 4.4 % of KSJC tracks against 5.6 % at KRDU and
KSTL. Nothing is being removed at KSJC that is kept elsewhere.

## 2. KSJC arrivals are genuinely rail-straight

Tortuosity = flown path ÷ straight-line distance, over the whole ring-entry→threshold
segment:

| airport | p50 | p75 | straight (<1.05) | established on centreline at 20 km to go | IQR cross-track at 15 km |
|---|---|---|---|---|---|
| KRDU | 1.074 | 2.383 | 47.6 % | 62.7 % | 125 m wide |
| **KSJC** | **1.001** | **1.017** | **76.3 %** | **96.6 %** | **12 m wide** |
| KSMF | 1.201 | 2.094 | 22.8 % | 58.7 % | 3.3 km wide |
| KSTL | 1.101 | 2.355 | 35.1 % | 38.7 % | 100 m wide |
| KMSY | 1.293 | 1.960 | 40.8 % | 78.4 % | 32 m wide |

Three quarters of KSJC arrivals fly a dead-straight 24 km final, and at 15 km out the middle
half of them sit inside a **12-metre-wide** corridor.

This is present in the raw data before any slicing: median path length inside the raw 30 km
crop is 29.1 km at KSJC against 34.6 km (KRDU), 35.6 km (KSMF), 40.5 km (KMSY) and 44.8 km
(KSTL) — KSJC traffic flies almost a straight radial across the whole harvested circle. At
the 30 km crop edge, 44.8 % of KSJC arrivals are already within 15° of the final approach
course (median offset 22°), against 3.2 % at KSTL (median 126° — still outbound on a
downwind) and 12.2 % at KRDU.

The cause is operational, not code: 86 % of KSJC arrivals use 30L, and the Santa Clara valley
plus SFO/OAK Class B funnel traffic onto the extended centreline outside 30 km.

## 3. The ADE gap is a mix artifact, and it reverses under standardisation

Stratifying every evaluation flight by post-anchor tortuosity × remaining path length, ADE
**within a stratum** is roughly equal across airports:

| stratum | KRDU | KSJC | KSMF | KSTL | KMSY |
|---|---|---|---|---|---|
| straight, <13 km to fly | 461 m (n=892) | 412 m (n=1064) | 464 m | 509 m | 502 m |
| straight, 13–20 km | 682 m | 540 m | 772 m | 698 m | 809 m |
| tortuosity >1.5, 20–35 km | 3882 m | 2322 m | 2856 m | 3404 m | 2611 m |
| tortuosity >1.5, >35 km | 2249 m (n=694) | **3931 m** (n=75) | 2091 m | 2000 m (n=509) | 3386 m |

What differs is the **share** of flights in the easy stratum: 78.6 % at KSJC against
41.8–61.0 % elsewhere. Reweighting each airport to the pooled stratum mix:

| airport | raw ADE median | mix-standardised |
|---|---|---|
| KRDU | 815 m | 1244 m |
| **KSJC** | **483 m** | **1526 m ← worst of the five** |
| KSMF | 1117 m | 1085 m |
| KSTL | 973 m | 1093 m |
| KMSY | 1148 m | 1484 m |

So the headline number is not a skill result. On matched flights KSJC is ordinary; on its
rare vectored flights it is the worst in the fleet (3931 m against 2000–2249 m at KSTL and
KRDU), plausibly because the pooled model saw very few vectored KSJC examples.

A second-order consequence worth stating: KSJC contributes 7848 training flights of which
~78 % are the same rail. Its **effective** sample count is far below its flight count, and a
pooled model may be pulled toward a "go straight" prior by it.

## 4. The genuinely short flights — a separate, real bug

`arrival_segment.py` classified a never-left-the-ring track as a LOCAL circuit only if it
started within `LOCAL_START_RADIUS_KM = 5` of the **destination** airport. A takeoff from a
*neighbouring* field inside the 25 km ring passed that test and was kept whole as a
"coverage-limited arrival" — first sample on a runway a few kilometres away, on the ground.

| airport | ground-start "arrivals" | satellite fields | long enough to reach the TS dataset |
|---|---|---|---|
| **KSJC** | **64 (0.57 %)** | KRHV ×28 (7 km), KPAO ×20 (21 km), KNUQ ×16 (11 km) | all 64 |
| KSMF | 8 | KSAC, KMCC | 8 |
| KMSY | 3 | KNEW | 3 |
| KRDU, KSTL | 0 | none inside the ring | 0 |

KSJC takes essentially all of it because it is the only airport of the five ringed by
satellite fields inside 25 km. Example: `N6101G_30R_a7f140_20260720T231613Z` starts at 23 m
HAE, 0.3 km from Reid-Hillview, and was fed to the model as an arrival.

A further ~6 KSJC flights begin airborne at 5–7 km on final at 300–400 m — pure reception
limits, genuine arrivals. Those are shorter than 120 s and the `seq_len = 60` lookback
requirement already excluded them from training.

**Impact on the metrics is small** (0.24 % of KSJC evaluation flights anchor within 5 km of
the threshold, against 0.10–0.16 % elsewhere). This is a data-correctness fix, not the
explanation for the ADE gap. It does overlap the known "duration head cannot predict below
~125 s" item in `ts_transformer/CLAUDE.md`, which quoted 0.4 % on KSJC.

### Why the criterion is altitude alone

Over all 42 725 rostered arrivals, the first sample's height above the landing runway is
bimodal with an empty band on both sides of the chosen 100 m threshold:

| AGL band | count |
|---|---|
| < 50 m | 73 |
| 50–100 m | 2 |
| **100–150 m** | **0** |
| 150–200 m | 6 |
| 200–300 m | 6 |
| ≥ 300 m | 42 638 |

Every one of the 75 below 100 m is within 2 km of a satellite field; the next flight up is
at 175.3 m, airborne, 11 km from the nearest field.

**Ground speed does not separate the two populations** and must not be part of the test: a
jet at rotation still reads 71–80 m/s while on the runway, squarely inside the approach-speed
range. A speed-and-altitude rule would have kept 29 of the 75 takeoffs.

## What changed

Both fixes are on branch `feat/arrival-takeoff-filter`.

**1. `trajectory_data_process/arrival_segment.py` — takeoff exclusion.**
`arrival_segment()` gained a required `field_elevation_m` (no default: a wrong vertical
datum would shift the test by the geoid separation without failing) and a third outcome,
`"takeoff"`, when the segment the ring cut produced starts at or below
`GROUND_START_AGL_M = 100.0` above the landing runway. The test is applied to the segment
the cut produced, not the raw track, so a flight that departs a neighbouring field, leaves
the ring and comes back is still a genuine arrival. `truncate_flights()` now returns
`(arrivals, locals, takeoffs)`.

`harvest/arrivals.py` records them as `outcome: "takeoff_in_segment"` in the manifest's
`excluded` list — counted, never silently dropped — and publishes `ground_start_agl_m`
beside `entry_radius_km`. It also asserts the source `altitude_datum` is HAE once at the
boundary, because that is what makes comparing a sample altitude to `elevation_hae_m` valid.

The manifest schema is bumped to **`harvest-arrivals-v5-takeoff-excluded`**. The roster's
meaning changed, so an existing v4 manifest now fails loudly rather than quietly feeding 75
takeoffs into a model split.

**2. `4dTrajectory/ts_transformer/approach_difficulty.py` (new) — per-row difficulty
covariates.** Every prediction row in `summary.json` now carries `anchor_range_m`,
`remaining_path_m`, `route_tortuosity`, `anchor_cross_track_m` and `established_at_anchor`,
computed from the observed track the error is scored against (never from the prediction) and
independent of the `coordinate_frame` setting. The `accuracy` block gains a `difficulty`
sub-block with the batch's route mix and the thresholds `established_at_anchor` encodes, so
a per-airport ADE can no longer be read without the mix it was earned on.

Verified end to end: the new filter excludes exactly the 75 flights identified above and no
others; the new covariates reproduce the independently measured mix (KSJC 78.2 % established
against 78.3 % measured by standalone geometry, KSTL 53.8 % against 54.0 %).

## Required rebuild

The on-disk arrival manifests are v4 and every loader will now reject them. Rebuild from
stored tracks — no download, no re-assignment:

```bash
for ap in KRDU KSJC KSMF KSTL KMSY; do
  python -m trajectory_data_process.harvest --airport $ap --evaluate-only
done
```

Consequences to expect:

- 75 flights leave the rostered arrivals (KSJC 11 146 → 11 082).
- Existing ts checkpoints were trained on cohorts that included those flights; their split
  membership no longer matches the roster.
- Published per-airport ADE/FDE tables should be re-derived with the standardised figures,
  or at minimum quoted alongside `established_at_anchor_fraction`.

## What this does not fix

- **The route mix itself.** KSJC really is straight within 25 km; no filter changes that.
  Capturing whatever sequencing happens further out would need a larger crop and entry radius
  (`harvest/__main__.py` enforces `entry_radius_km < radius_km`), which changes the boundary
  condition for all five airports and invalidates every arrival manifest and checkpoint —
  a deliberate re-harvest decision, not a tweak.
- **KSJC's weight in the pooled cohort.** Reporting effective diversity, subsampling the
  straight mode, or weighting by stratum are all open options.
- **The duration head's ~125 s floor**, which the removed flights partly overlapped.
