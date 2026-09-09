# Plan-and-guidance: the next model (design v2, 2026-09-09)

Status: design only; nothing built. v1 (same day) split the plan into operating and route parameters
(decision (A)). v2 makes the published approach procedure the backbone of the whole design: the
learned plan lives inside the procedure, the guidance layer enforces the procedure by construction,
and the result is judged twice — as a prediction of what aircraft do, and as a reference that
conforms to the procedure. Source of the programme's numbers: `docs/reports/2026-09-08_programme_results_and_plan.md`.

## 1. Why change the architecture

Today's model is a transformer that outputs 32 control segments (thrust, bank, load factor) plus a
duration, and a point-mass rollout that flies them. The network flies the aircraft; inference-time
hooks correct it afterwards. Six days of experiments say this is the wrong division of labour:

- The network draws the path well (straight-in chamfer 65–109 m, cross-track ~100 m). What it
  cannot decide is *when*: when to turn base, when to slow down, how long to fly the downwind. That
  decision is the controller's. Every experiment that supplied it from outside moved the error by
  hundreds of metres (true arrival time: vectored ADE 2643 → 1596 m; true join point: 2858 → 2356 m);
  nothing learned from the aircraft's own history moved it.
- The network does not know the constraints. Training-time corridor penalties were rejected twice.
  Feasibility comes only from hooks added after the fact, and each fixed one thing and broke another
  (the speed floor removed stall and created thrust-over-max; the trombone removed both and
  over-stretched by 10 km because it measured the wrong distance). Stacked, the hooks take the
  flyable share from 8 % to 78 % — they already do the work, but as corrections of something that
  was never planned.
- The loss is a hand-balanced mixture of seven terms, and the balance is fragile: one unit change
  made the latent work and destroyed bank fidelity (skill 0.726 → 0.400); restoring the teacher's
  share repaired the bank shape and destroyed the latent.
- Seed noise on pooled ADE is ~125 m; half of the week's arms moved less than that.

The published approach procedure (the RNAV approach in the CIFP: legs, fixes, the final approach
segment with its course, glidepath, threshold-crossing height and LPV course width, the altitude
and speed limits at the fixes) already fixes most of what a trajectory must be. Today's model
rediscovers that from data and gets it partly wrong; the hooks put part of it back. The design
below stops learning what is published, and learns only what the procedure leaves free.

Relation to the current control-prediction model (decision (A)): the point-mass dynamics, bounded
controls, rollout, actuator model, CTA conditioning, random anchors, quantile head and calibration
are all kept. What changes is who chooses the controls: a guidance layer emits them from the plan,
so the controls become an inspectable intermediate rather than the learning target. The thesis rule
that intent is never an output is kept by construction (§3b).

## 2. The idea

Three layers:

1. **The procedure is the skeleton.** For the assigned runway, the CIFP gives the final approach
   segment (course, glidepath angle, threshold-crossing height, LPV course width, the fixes
   FAF/IF/IAF with their altitude and speed limits) and the published transitions onto it. This is
   data, read once, not learned. Everything the trajectory must satisfy is stated here as hard
   bounds.
2. **The plan is what the procedure leaves free.** Operating parameters — when the aircraft
   arrives, how fast it flies before the final, where it slows to its final speed, at what height
   it captures the glidepath — are predicted by the network, each inside the interval the
   procedure and the aircraft allow. Route parameters — where and from which side the flight joins
   the final, and how much path it flies before that — are the controller's intent: assigned by
   the scheduler, or kept as a distribution, never a committed point.
3. **The guidance layer flies the plan inside the skeleton.** A deterministic controller builds a
   route on the procedure's legs, tracks it with bounded bank inside the corridor, follows the
   glidepath from the capture height, holds the speed schedule between the stall margin and the
   thrust limit, and meets the assigned time exactly or reports the part it cannot absorb. Nothing
   it produces can violate the procedure or the envelope, because the constraints are in the
   construction, not in a loss.

The scheduler is inside the loop: it assigns the time and the join, and receives a
procedure-conforming reference plus the arrival-time distribution.

## 3. What the network predicts, and what it does not

Every parameter has a definition, a range set by the procedure and the aircraft, and an extractor
that reads it off an observed track (the supervision). All extractors exist in the package or in
this week's readouts. A predicted value outside its range is clamped before flying, and the clamp
is recorded in the record — the model is never allowed to plan an infeasible flight.

### 3a. Operating parameters — predicted (point + distribution)

| parameter | meaning | range (procedure / aircraft) | extractor |
|---|---|---|---|
| `T` | time from the anchor to the threshold | > the time the shortest legal route needs at the fastest legal speed | the track's duration (as today) |
| `V_mid` | speed held before deceleration | ≤ the speed limit at the next fix (procedure / ICAO category); ≥ stall margin | median ground speed over 20–10 km remaining |
| `d_decel` | remaining distance at which the speed first drops below `V_target + 10 m/s` | before the FAF if the procedure limits speed there; ≥ 0 | `run_ts_straight_in_residual_readout.decel_distance_km` |
| `V_final` | final approach speed | ≥ the type's published V_ref (`landing_aero`, the observed speed gate's window) | the threshold-crossing speed |
| `h_capture` | height at which the glidepath is captured | ≥ the platform / fix altitude at that distance; ≤ the glidepath | the track's altitude when the on-final gate opens |

Five numbers, all aircraft operating parameters. Each is predicted as a point and as a
distribution (the B-line quantile head, which is the one head that improved arrival-time
accuracy; or the L2.g latent fan). `T` keeps its calibrated interval (B2).

### 3b. Route parameters — assigned, or a distribution; never a point output

| parameter | meaning | range (procedure) | extractor (for the distribution head and the oracle test) |
|---|---|---|---|
| `d_join` | remaining path at which the flight becomes established on the final | on a published leg or transition of the assigned runway's approach (snapped to the nearest legal join; the raw value is kept as a diagnostic) | `final_approach_geometry.truth_final_gate` |
| `side` | which side the base leg comes from | {left, right}, restricted to the sides the procedure's transitions allow | sign of the cross-track offset at the gate opening |
| `L_pre` | path length flown before the join | ≥ the shortest legal route to `d_join`; extra length only as a lengthening of the pre-final legs | arc length of the track up to `d_join` |

In delivery these come from the scheduler, which decides the sequence and therefore the join. When
the scheduler has not decided, the model offers a distribution over them (the latent sampler, whose
nearest-of-6 sample beats the point prediction on 92 % of flights; or quantiles), and the guidance
layer flies each sample into a member of a fan. There is no committed point estimate of the route.

Together, 5 + 3 numbers replace the 32 × 3 control segments. The control-basis oracle of 09-07
showed a 32-segment basis reproduces the truth to ~100–200 m; §9 step 2 measures the ceiling of the
lower parametrisation before anything is trained.

## 4. The guidance layer

A fixed, deterministic controller that flies a plan on the procedure's skeleton. Its parts exist:

1. **Procedure reader** — from the CIFP for the assigned runway: the final approach segment
   (course, glidepath, TCH, LPV course width = the corridor `k·hw(d)` already in
   `fas_geometry`), the fixes with their altitude and speed limits, and the published transitions.
   `preprocess_procedures.py` and `approach_constraints` already parse this for the optimizer.
2. **Route builder** — from the anchor state, the plan, and the skeleton: fly the current heading
   to a turn point, a base leg from the allowed side, a join on a legal leg at `d_join`, then the
   final. The pre-final path length equals `L_pre`; extra time (an assigned arrival later than the
   plan's own `T`) becomes extra pre-final length, computed once at planning time against the
   route's true length (the trombone's mechanism, with the distance L3.e got wrong).
3. **Lateral tracking** — the nominal-law hook: line-of-sight to the route, bounded bank,
   coordinated load factor, actuator-lag compensation; inside the final the barrier keeps the
   aircraft inside the LPV corridor by construction.
4. **Vertical** — the nominal hook's glidepath law: level or descending to `h_capture` subject to
   the fix altitudes, then on the glidepath; TCH at the threshold.
5. **Speed schedule** — hold `V_mid` (capped by the fix speed limits), decelerate at `d_decel` to
   `V_final`, never below the stall margin (`V_floor`), never above the thrust limit; a plan that
   asks for more is clamped and the clamp recorded.
6. **Time closure** — route length and speed schedule together fix the arrival time. If the
   assigned time cannot be met inside the procedure and the envelope, the layer flies the closest
   feasible plan and reports the unabsorbable delay X per flight.

The point-mass rollout remains the simulator. Because every step is deterministic and bounded,
the trajectory conforms to the procedure and the envelope by construction: corridor, glidepath,
fix limits, stall, thrust, bank. These become properties to verify, not outcomes to hope for.

## 5. What the scheduler supplies and gets back

- Supplies: the assigned arrival time and the route (`d_join`, `side`, the pre-final length or a
  delay to absorb). If it assigns nothing, the model returns the fan over legal routes and the
  arrival-time distribution, never a single guess.
- Gets back: a procedure-conforming, flyable reference for the assigned time and route; the
  operating parameters behind it (when the aircraft slows, how fast, how high); the unabsorbable
  delay X if the assignment is infeasible; and, when nothing is assigned, the arrival-time interval
  (~30 s wide for straight-in traffic, ±70 s for vectored traffic until the join is decided) and the
  fan over routes.
- The multi-aircraft demonstration (assign times and joins to several arrivals, build references,
  check separation) follows directly, because every reference is a plan the scheduler can read.

## 6. Training

- **Supervision**: the five operating parameters extracted from each observed track, regressed
  directly in their own units (point and quantile heads); the three route parameters supervise
  only the distribution head. No inverse-dynamics teacher, no imitation weight, no position-unit
  balancing. Bank fidelity is not a training target; it is a property of the guidance layer.
- **Not through the guidance layer**, at first. This week's evidence (L1.c; the 09-06 training-
  through-hook arms) says training through a corrective layer teaches the network to lean on it.
  The plan is supervised directly; the guidance layer runs at inference. End-to-end fine-tuning
  through a differentiable guidance layer is a later, gated experiment.
- **Inputs**: the same history window and threshold-anchored ENU chart (the frame ablation showed
  the anchor is the model's runway knowledge), the procedure's skeleton as conditioning (the fix
  distances and limits, so the plan head knows its ranges), and the scheduler's inputs as
  conditioning (CTA conditioning exists; join conditioning is the same mechanism).
- **Anchors**: the remaining-path-uniform random-anchor sampler with the scheduler fix (A0.b), so
  the plan is re-issued at any point of the approach; the two-model rule stays available.
- **Data**: pooled over the five airports (42,650 arrivals) rather than KRDU alone; the
  procedure reader makes airports comparable, and more data is the only lever on the 125 m line.

## 7. Two evaluations, and the gates

The reference is judged twice, because two different things are being claimed:

**As a prediction** (against observed tracks, the metrics of this programme): ADE, FDE, chamfer,
duration MAE, calibrated coverage, per stratum. This is the honest reading of the model as a
forecaster. Note the trade stated in §8: where real arrivals are radar vectors rather than published
transitions, a procedure-conforming reference will predict the observed track *worse*; the readout
must show the vectored stratum separately, and the fan (not the point) is the fair comparison there.

**As a reference** (against the procedure and the envelope, the `evaluation` package's gates): the
corridor and glidepath verdicts, fix altitude and speed limits, stall / thrust / bank, and CTA
obedience with the unabsorbable delay X. This is the deliverable, and it should be near-perfect by
construction.

Gates for the first prototype, KRDU val 1404, paired, two seeds:

| question | today's best | gate |
|---|---|---|
| prediction, no assignment (vectored ADE / straight-in chamfer) | 2579–2870 m / 65–109 m | ≤ 2870 + 125 m / ≤ 110 m |
| prediction, assigned time (vectored ADE) | 1596 m | ≤ 1596 m |
| prediction, assigned time and join (vectored ADE) | 2356 m (join only, Phase 0) | ≤ 1400 m |
| arrival-time MAE, straight-in / pooled | 10.3 / 23.9 s | within 1 s |
| reference: fully flyable, true time (truncated at threshold) | 78 % (L3.e stack) | ≥ 95 % |
| reference: fully flyable, +60 s; X reported | 52 % | ≥ 90 % |
| reference: corridor / glidepath / fix-limit violations on the final | 16 % lateral (L3.e) | 0 % by construction |
| bank | skill 0.726 (native32) | not a target; bank RMS ≤ 0.5° on straight-in |

Vetoes: the plan head's parameters worse than trivial baselines (airport medians) — the learned
part is not learning; or the oracle ceiling (§9 step 2) worse than today's model — the
parametrisation is too coarse.

## 8. Reuse, new work, risks

- **Reused**: the point-mass rollout and actuator model; the nominal-law, barrier, speed-floor and
  trombone hooks as guidance parts; the corridor geometry and on-final gate; the CIFP procedure
  parsing from the optimizer; the CTA conditioning; the quantile head and split-conformal
  calibration; the random-anchor sampler and anytime replay; all readouts and the publisher; the
  `evaluation` package's procedure gates.
- **New**: the plan-parameter extractors as a dataset layer; the procedure reader as model
  conditioning and guidance skeleton; the plan head; the route builder with the time-closure step;
  the assembly of the hooks into one controller; the pooled-airport run.
- **Risks and their tests**: (1) the parametrisation may be too coarse for vectored flights with
  more than one turn — allow one or two optional extra waypoints, gated by a readout of how many
  truths need them; (2) procedure conformance versus what pilots actually fly — real KRDU vectored
  arrivals are radar vectors, so a conforming reference is a worse prediction of them; the two
  evaluations make this explicit rather than hiding it, and the fan carries the prediction claim
  there; (3) the join is as hard to predict as the intent — which is why it is assigned or a
  distribution, never a point; (4) the procedure data may be incomplete for some runways (KRDU 32
  has no published vertical path) — the skeleton then falls back to the runway geometry and says so.

## 9. Steps

1. Procedure reader + extractors: the skeleton for every runway in the fleet; the eight plan
   parameters for every KRDU arrival; their distributions; how well airport medians predict them;
   how many observed joins fall on a published transition versus a vector (CPU, one day).
2. Oracle ceiling: fly the *true* plans through the guidance layer on the skeleton and measure
   against the observed tracks and against the procedure. If the ceiling is worse than today's
   model as a prediction, revise the parametrisation before training anything.
3. Plan head on KRDU: point and quantile heads, one seed, both evaluations.
4. Assigned-time and assigned-join conditioning; the +60 s delay test with X reported.
5. Two seeds; pooled five-airport training; KSJC replication.
6. The multi-aircraft scheduler demonstration.

## 10. What stops

No more arms on the latent dose axis, teacher replacement, position units, hook doses or the
duration weight. L3.f runs because it is the route builder's specification test; nothing else from
the current line is queued.
