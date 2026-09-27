# Why arm A's KRDU endpoints sit 150–200 m NW of every runway (2026-09-03)

Follow-up to `2026-09-03_airport_frame_ablation_results.md` (per-runway cross-track medians
05L −145, 05R −150, 23L +198, 23R +206 m — all the NW side in world coordinates) and
`2026-09-03_runway_hypothesis_expansion.md`. Measured on the validation records of the
`A_threshold_enu` checkpoints (seeds 1337 and 2024) with three throwaway scripts; nothing
was retrained.

## Verdict

The bias is in the **model's output**, not in the data or the readout. It is a
**world-fixed translation of the whole predicted path, present from the first predicted
step**, of the order of 250–350 m toward the NW, and it lands on straight-in flights. Its
sign is the sign of KRDU's population-mean lateral drift (63 % of arrivals join their final
from the SE side and therefore move NW relative to their heading), and the training
objective cannot see a constant lateral offset of that size on straight-in flights.

## Evidence

**Not the data.** The last observed sample before the threshold crossing sits on the CIFP
centreline: cross-track medians −2 / −2 / +1 / +1 m on 05L / 05R / 23L / 23R. The CIFP path
point differs from the NASR threshold by ≤ 6.5 m cross-track on every KRDU runway. KSJC's
observed and predicted endpoints are both within ±12 m of their centrelines.

**Not vectored flights.** Median lateral miss of the predicted endpoint (prediction minus
observation at the threshold): established-at-anchor flights +204 m (seed 2024: +151),
not-established +24 m (+9). The bias lives in the easy stratum.

**A translation from the first step, not a rotation.** For established flights, prediction
vs observation cross-track at fixed along-track stations (m, + = right of the inbound
course; seed 1337):

| runway | n | first step (Δcross / Δalong) | −10 km | −5 km | −2 km | threshold | final heading off course |
|---|---:|---:|---:|---:|---:|---:|---:|
| 05L | 196 | −348 / +361 | −308 / +3 | −333 / −1 | −390 / −1 | −438 / −2 | −1.2° |
| 05R | 78 | −340 / +184 | −318 / +2 | −332 / +1 | −400 / −2 | −458 / −2 | −1.3° |
| 23L | 245 | +239 / +364 | +229 / −9 | +271 / −3 | +284 / +1 | +302 / +1 | +0.3° |
| 23R | 649 | +235 / +184 | +213 / −10 | +262 / −3 | +271 / +1 | +283 / +1 | +0.2° |

The first predicted point is already 240–350 m off the aircraft's actual position (the
observed track's own first step matches the kinematic extrapolation to 1 m), and the
offset then stays roughly constant along a final that is parallel to the true one (heading
error ≤ 1.3°). Left for 05, right for 23: the NW side both times. Seed 2024 reproduces
every number within 40 m.

**World-fixed, not a pull toward the normalizer mean.** First-step jump (prediction minus
anchor + groundspeed × 2 s × heading) projected on the world NW unit vector, all 2,104 KRDU
validation flights: +236 m overall; by anchor quadrant NE +236, NW +332, SE +221, SW +135 —
positive everywhere. The projection toward the normalizer mean flips sign by quadrant
(NW −309, SE +302), so shrinkage toward the training mean is ruled out. KSJC: +68 m.

**The model does it to a clean history too.** Noise-free synthetic straight-in histories
(entry offset 0°, jitter 0) through the KRDU checkpoint: first-step NW jump +317 m on
05L/05R and +150 m on 23L/23R. The displacement does not need ADS-B noise or a side cue in
the history.

**Why NW: the population's lateral drift.** Observed displacement of the aircraft relative
to its anchor-heading extrapolation, projected on world NW, all KRDU validation flights:

| lead | median | mean |
|---|---:|---:|
| +60 s | +4 m | **+192 m** |
| +120 s | +3 m | **+314 m** |

The median is zero and the mean is NW: a skewed population, because 63 % of KRDU anchors
lie SE of their runway's extended centreline (05L 263 of 471, 05R 121 of 213, 23L 368 of
470, 23R 593 of 950) and those flights must displace NW to join the final. An MSE-trained
deterministic model learns conditional MEANS; its residual, weakly determined lateral
offset takes the sign of that skew. At KSJC the same statistic is SE-ward (−342 m median at
+60 s) and the KSJC model shows no NW bias.

**Why the loss lets it stand.** A 300 m constant lateral offset on a straight-in flight
costs `(300 / 10,000)² ≈ 9e-4` per supervised point in the position objective
(`position_loss_scale_m = 10 km`), against a pooled objective of ~0.08 dominated by
kilometre-scale errors on vectored flights; established flights as a whole contribute
roughly 5 % of the position loss. The state output predicts absolute chart positions with
no continuity to the anchor and no cross-track term, so nothing in the objective is
sharper than that.

## What follows

- Every quoted per-runway cross-track median for arm A carries this ~250 m NW translation.
  It is why 23L's mirror-image fake sibling (displaced SE) beat its real sibling in the
  hypothesis-expansion readout: the fake threshold cancels the bias.
- Candidate fixes, untested and outside the current experiments: predict displacements
  from the anchor (or add a first-step continuity term) so the path starts where the
  aircraft is; an endpoint cross-track term at the runway scale (the terminal-loss
  machinery exists on the control path); or per-flight loss scaling so straight-in flights
  are not drowned by the vectored tail. Any of these would be a recipe change (`state-v2`).
