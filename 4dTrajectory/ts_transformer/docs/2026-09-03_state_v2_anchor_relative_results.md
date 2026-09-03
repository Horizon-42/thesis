# state-v2 candidate: anchor-relative positions — results (2026-09-03)

Follow-up to `2026-09-03_krdu_nw_endpoint_bias.md`, which traced arm A's ~250 m NW endpoint
translation at KRDU to a first-step jump of the absolute position output. The candidate
fix: `state_position_reference="anchor-relative"` — the network's position channels are
read as displacements from the anchor and the anchor's normalized position is added back
inside `StateOutputLayer`, so the zero output means "the aircraft stays where it is".
Everything else is the state-v1 recipe (iTransformer, `enu`, `full` horizon, `all`
aircraft, two seeds). Arms: `docs/experiments/state_v2_anchor_relative_arms.json`;
artifacts `4dTrajectory/outputs/{KRDU,KSJC}/experiments/state_v2_20260903/` (informal
runs — no experiment manifest; code at commit `2f7f746`, runner `6b98ffc`); readout
`compare_frame_arms.py … --only A_threshold_enu,A_anchor_relative` (`readout.json` beside
each campaign). Paired flight-by-flight against the airport-frame campaign's arm A on the
same validation split.

## Verdict

**The mechanism is fixed; the recipe is not adopted.** The first-step jump and the
straight-in translation collapse on both airports and both seeds, and straight-in flights
gain ~140 m of FDE at KRDU — but vectored flights lose ~350 m of FDE at KRDU on both seeds
(worse on 78–82 % of them), which is the pre-registered veto. The absolute output carried
an implicit "the endpoint is the origin" prior that the residual output gives up; the
next candidate has to keep both (see below).

## Pre-registered reading (from the arm declaration, before any number)

- first-step jump and straight-in lateral miss should collapse toward zero — **yes**;
- straight-in FDE should improve by about the translation (~200 m) — **yes, ~140 m**;
- a vectored-stratum regression on both seeds would veto the recipe — **it did**.

## KRDU (2,104 validation flights)

Mean ADE / mean FDE / median FDE (m); paired share = flights on which the anchor-relative
arm beats arm A of the same seed.

| stratum | n | A seed 1337 | v2 seed 1337 | A seed 2024 | v2 seed 2024 | v2 better (ADE / FDE) |
|---|---:|---:|---:|---:|---:|---:|
| all | 2,104 | 1383 / 1163 / 711 | 1322 / 1209 / 762 | 1361 / 1124 / 690 | 1323 / 1198 / 762 | 64–67 % / 52–55 % |
| straight-in | 1,273 | 570 / 643 / 526 | **435 / 492 / 331** | 550 / 622 / 514 | **441 / 499 / 342** | 78–82 % / 74–77 % |
| vectored | 827 | 2599 / 1967 / 1356 | 2652 / **2317** / 1728 | 2574 / 1900 / 1327 | 2646 / **2278** / 1718 | 41–43 % / **18–22 %** |

The bias itself (established flights, prediction vs observation cross-track, m, + = right
of the inbound course):

| | first-step lateral jump | lateral miss at threshold | endpoint cross-track medians 05L / 05R / 23L / 23R |
|---|---:|---:|---|
| A (absolute) | −348 / −340 / +239 / +235 | +204 (seed 2024: +151) | −145 / −150 / +198 / +206 |
| v2 (anchor-relative), seed 1337 | −1 / 0 / 0 / +7 | +21 | −52 / −69 / +48 / +47 |
| v2, seed 2024 | −18 / −20 / +3 / +1 | | −77 / −112 / +61 / +59 |

The first step now starts at the aircraft (median |offset| 118–121 m, almost all of it
along-track, vs 368–390 m); a residual NW drift of 50–110 m remains and now BUILDS along
the final (v2 station profile: ±5 m at −10 km, ±30 m at −5 km, ±60–110 m at the threshold)
instead of being a translation — a milder version of the same population prior, expressed
as curvature rather than as an offset.

What the residual output costs: the endpoint lateral tail widens (p95 |cross-track|
640–668 m vs 477–492), endpoints closer to the parallel sibling 3.5–3.7 % vs 1.5 %,
vectored endpoint error 1630 vs 1250–1270 m, `horizonCapped` 8–11 vs 0–6, altitude p95
611–639 vs 466–469 m. The absolute output could "snap" a long forecast onto the origin;
the residual output has to integrate its way there.

Seed stability improves: seed-to-seed pooled ADE / FDE swing 1 / 11 m (A: 22 / 40 m).

## KSJC (1,666 validation flights)

| stratum | n | A seed 1337 | v2 seed 1337 | A seed 2024 | v2 seed 2024 | v2 better (ADE / FDE) |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,666 | 870 / 776 / 439 | **784 / 724 / 357** | 865 / 755 / 402 | **810 / 754 / 391** | 75–77 % / 64–66 % |
| straight-in | 1,316 | 426 / 449 / 320 | **365 / 386 / 244** | 424 / 430 / 286 | 377 / 419 / 289 | 76–79 % / 66–69 % |
| vectored | 350 | 2536 / 2006 / 1109 | 2357 / 1996 / 1251 | 2521 / 1977 / 1141 | 2440 / 2012 / 1206 | 70–71 % / 55–57 % |

At KSJC (few vectored flights, no lateral bias to begin with) the anchor-relative output
wins pooled ADE by 55–86 m on both seeds and even the vectored stratum's ADE; FDE is a
wash on vectored flights (paired medians go the other way from the distribution medians).
First-step |offset| 59–79 m vs 177–240; altitude p95 291–310 m vs 530; endpoint lateral
p95 383–428 m vs 195–237 (the same tail widening as KRDU).

## Reading

The two output parametrizations carry two different priors, and each stratum wants a
different one. Absolute positions are implicitly "end at the origin": good for long,
vectored forecasts whose endpoint the model cannot integrate its way to, bad at the start,
where the model must reconstruct the current position from a 120 s history and gets it
wrong by a population-dependent constant. Anchor-relative displacements are "start where
you are": exact at the start, good for straight-ins whose whole path is a short
extrapolation, bad at the far end of a 600 s vectored path where displacement errors
accumulate. The pooled numbers are the route mix deciding which prior wins, which is why
KSJC (79 % established) likes v2 and KRDU's vectored 40 % vetoes it.

## Next candidate (not run)

Keep both priors: an absolute output with a **continuity term** on the first steps
(penalise the offset of the first predicted rows from the anchor's kinematic
extrapolation, at the runway scale rather than the 10 km position scale), or a
progress-weighted blend of an anchor-relative and an absolute head (weight 0 at the
anchor, 1 at the endpoint). Either keeps the endpoint prior that vectored flights need
and removes the start-of-path translation that straight-ins pay for. The residual NW
curvature that survives in v2 says the population prior will still need the loss to see
lateral error at the runway, i.e. an endpoint cross-track term.

## Caveats

- Two seeds, one recipe, `state` output, iTransformer, validation split only. The runs
  carry no experiment manifest (an untracked file in the tree blocked the formal guard);
  the code commit is recorded above.
- `--only` reads the arm-A checkpoints from the airport-frame campaign; the splits are
  identical by construction (same split seed) and were verified per checkpoint there.
