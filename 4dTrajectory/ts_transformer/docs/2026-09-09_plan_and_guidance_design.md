# Plan-and-guidance: the next model (design v5, 2026-09-11)

Status: **v5.2 BUILT and MEASURED (2026-09-11 night, `dev-plan-rolled`; §9 step 3(e), §12.6): the
head is TRAINED ON ROLLED WINDOWS** — the windows a lockstep flight produces, labelled with the
truth's policy read at each state (`labels.TruthExpert`; `run_ts.py plan_rolled_windows`), mixed
into every epoch's draw at `plan_rolled_share`, the val split's rolled windows scored every epoch
beside the observed objective. The receding-horizon head now works: re-asked every 30 s at the
60 s anchor, vectored ADE **5391 → 3641 m** (share 0.75; the once-per-leg head 3781, the lockstep
ceiling 2184), established 55 → 94 % pooled, capped 29 → 9 %; straight-in 747 m, chamfer 44.
The share is the lever (0.75 the candidate default, 1.0 its equal); DAgger's first round did not
help. Next: hold an order unless the change persists, then the fan over the next fix.
Previous status: **v5.1 BUILT and MEASURED (2026-09-11 night, `dev-plan-lockstep`; §12.5): the rolling is
RECEDING-HORIZON and LOCKSTEP — §9 step 3(d), decided by the user on reading §12.4.** The
lockstep oracle reads 1847 m of vectored ADE at L−1 against the leg form's 1492 at
78× the speed (69 s for the split). **Re-asking the head every step made its vectored
prediction worse** (5391 m every 30 s, 4991 every 60 s, 3781 once per leg; the lockstep
ceiling 2184): it is asked on its own flown windows, which it never trained on — the next
item is training-side (§12.5). The leg-form numbers of §12.4 remain the head's best prediction. Step 3 as built asked the head
once per instruction and flew its answer to the fix: at the 60 s anchor that is one guess for a
turn 25–39 km / 4 min away, committed for the whole leg, and the 2.6 km it misses by became the
3781 m of rolled ADE. A radar vector is issued when the aircraft is near its turn, and the head's
own error is 1.6 km at the near anchors against 2.6 km at the far one: so the head is asked again
every `LOCKSTEP_S` = 30 s, the aircraft flying its current heading meanwhile, and every flight's
next 30 s is rolled as ONE batch (the leg-at-a-time form rolled one flight at a time, ~4 s each,
90 min per full run; the user asked for the oracle to be faster). The step-3 artifacts and
numbers (§12.4) stay as they are; the lockstep numbers go to §12.5. Previous status: **v5 step 3 BUILT and run small-scale (2026-09-11 evening, `dev-plan-next`; §12.4)**: the
next-instruction readout (3a), the rolled oracle (3b) and the single-step plan head — the fourth
output path, `prediction_output=plan` — trained on a 403-flight development cohort and rolled
through the guidance (3c); the full-cohort head and the full rolled oracles are the numbers §12.4
waits on. Two things the small run found and fixed: the oracle graded the truth itself as a
glidepath violator (the window now binds inside the FAF, the coded floor before it, and the
truth's own rows are graded beside every flight), and a flight already on the final needs the
"no fix ahead" order too (the first head sent 23 of 48 established flights to a made-up fix).
**v5 (2026-09-11, drafted for the user's read): the route is predicted ONE INSTRUCTION AT
A TIME** — the next fly-by fix and the speed at it, from a 60 s window, at any anchor, rolled
by the guidance until the join (§3b, §4.2, §5, §6, §7, §9 step 3); the whole path's fixes stay
the offline label and the oracle's representation. Why: a radar vector is issued one at a time
and executed within seconds, so no aircraft ever holds the whole path the K-fix head would
predict; a single step is the quantity that exists, and the one the scheduler assigns. Nothing
below step 3 changes. Steps 0–2b BUILT, REVIEWED and MEASURED on `dev-plan-guidance` /
`dev-plan-turns`, 2026-09-10/11 — the shared
rollout parts moved up (`7cb58b4`), the procedure reader and the eight extractors (§12.1), the
guidance layer and the oracle ceiling (§12.2, after the step-2 review's fixes: the §7 veto does
not fire; straight-in ADE 204 m / chamfer 34 m / arrival-time MAE 4.8 s from the
truth's own plan, vectored ADE 1591 m against native32's 2870 m, 99.7 % flyable,
99.7 % established, corridor left after the join by 1.4 %). Step 3 (the plan head)
is unblocked; the decision it needs is in §12.2, read with §12.3 (step 2b, 2026-09-11: the
fixed-K fly-by waypoints oracle — vectored chamfer 636 → 268 m from the truth's own
K ≤ 4 fixes — and the oracle at earlier anchors). v1 (same day) split the plan into operating and route parameters
(decision (A)). v2 made the published approach procedure the backbone of the whole design: the
learned plan lives inside the procedure, the guidance layer enforces the procedure by construction,
and the result is judged twice — as a prediction of what aircraft do, and as a reference that
conforms to the procedure. v3 (evening) replaces the hook numbers with the corrected L3.e-r / L3.f-r
reruns (the earlier campaigns used a threshold-crossing rule that cut vectored flights abeam on the
downwind; their directories are deleted) and adds §11, which places this design inside the package
review's target architecture (`2026-09-09_package_review_bugs_and_architecture.md`) so it lands as
one output strategy rather than ten touch points. Source of the programme's numbers:
`docs/reports/2026-09-08_programme_results_and_plan.md`; the corrected hook numbers:
`2026-09-07_latent_intent_design.zh.md` §六, L3.e-r / L3.f-r. v4 (2026-09-10) rewrites §11
against the package as it now IS: every step of the review — the package (§4.1), the §5 freeze,
the config views (§4.3), the output strategies (§4.2), the loop/predict extraction (§4.4), the
runners (§4.5), the tests (§4.6) and the grouping by plane — landed on `dev-leg-ctrl` (`dfce744`),
so the plan strategy is described in the strategy interface's real member names and the grouped
paths, and this design's step 3 is unblocked.

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
  (the speed floor removed stall and created thrust-over-max; the trombone removed part of both, but
  the median flight receives no stretch at all). With the corrected threshold cut, the three-hook
  stack makes 46 % of flights fully flyable at the true arrival time and 20 % at +60 s; about 30 %
  of flights (about 80 % of vectored ones) never become established on the final and their rollouts
  run on for tens of kilometres. The hooks do part of the work, as corrections of something that was
  never planned; the stretch the delay needs is a route decision, which a hook cannot make.
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
| `T` | time from the anchor to the threshold | > the time the shortest legal route needs at the fastest legal speed (the beeline over the coded speed limit; no KRDU document codes one, so the floor is uncoded — `None`, never the flight's own maximum) | the track's duration (as today) |
| `V_mid` | speed held before deceleration | ≤ the speed limit at the next fix (procedure / ICAO category); ≥ stall margin | the mean ground speed over the path flown before `d_decel` (its length over its time); a flight already decelerating at the anchor holds its anchor speed (20 % of KRDU val) |
| `d_decel` | remaining path at which the ground speed first drops below `V_final + 10 m/s` | before the FAF if the procedure limits speed there; ≥ 0 | `experiments.straight_in_residual_readout.decel_distance_km` (moves into `outputs/plan/extractors.py`, §11) |
| `V_final` | final approach ground speed | the observed speed gate's WINDOW about the type's published V_ref (headwind-corrected IAS, `evaluation/docs/THRESHOLD_SPEED_GATE.md`) — not a floor on ground speed: 72 % of observed values sit below V_ref (§12.1) | the ground speed at the last observed row (~380 m short of the threshold) |
| `h_capture` | height (above the threshold aim point) at which the final is captured LATERALLY — the glidepath is captured later, from above, on most flights (§12.1) | ≥ the floor coded at the next fix ahead of the join; ≤ its ceiling where one is coded | the chart height at the row the truth gate (`truth_final_gate`) opens |

Five numbers, all aircraft operating parameters. Each is predicted as a point and as a
distribution (the B-line quantile head, which is the one head that improved arrival-time
accuracy; or the L2.g latent fan). `T` keeps its calibrated interval (B2).

### 3b. Route parameters — the NEXT instruction (v5); assigned, or a distribution; never a committed path

**v5 (2026-09-11).** The route the network predicts is the next radar instruction, not the
path: from the anchor, the next fly-by fix and the speed to be flying at it — a vector
"turn to heading X, reduce to Y" as the point it aims at — and whether there is one at all
(no next fix: the current leg runs onto the final and the join is next). The guidance flies
toward it and the network is asked again — v5.1: every 30 s, from wherever the aircraft is,
not once per leg (§9 step 3(d); the turn point it aims at is refined as it approaches, the
way a vector is issued near the turn) — until the join. This is
how the instruction reaches the cockpit (one at a time, executed within seconds, the next
unknown until it is given), what a scheduler assigns (the next vector, as ATC does), and the
part of the route a 60 s history can say anything about. The fix-and-speed pair is
`PlanLabels.waypoints[0]` / `waypoint_speeds[0]` of §3b's fixed-K representation below,
read at a random anchor; the whole path's fixes remain the offline label set and the
oracle's representation (§12.3).

| parameter (v5, predicted per anchor) | meaning | range | extractor |
|---|---|---|---|
| `next_fix` | the next fly-by fix, in runway axes about the anchor (`Δd`, `Δxt`); None when the current leg runs onto the final | inside the TMA the procedure's transitions cover; ahead of the anchor along the path | the first of `extract_waypoints` after the anchor (§3b below), or None |
| `V_next` | ground speed at that fix | stall margin … the coded limit / the type's approach maximum | the median ground speed over the turn's rows (`waypoint_speeds`) |
| `next_is_join` | no next fix: the leg runs onto the final | — | the join before any fix |

Each is predicted as a point and as a distribution; the distribution over `next_fix` is the
fan of §3b's last paragraph one step deep — a small, readable set of "the next vector is one of
these", which the scheduler can override with its own. The three whole-path parameters below
stay defined (the join is where the rolling ends; `side` and `L_pre` are what the rolled route
amounts to, reported per flight), and the scheduler may still assign them whole.

| parameter | meaning | range (procedure) | extractor (for the distribution head and the oracle test) |
|---|---|---|---|
| `d_join` | remaining path at which the flight becomes established on the final | on a published leg or transition of the assigned runway's approach (snapped to the nearest legal join; the raw value is kept as a diagnostic) | `geometry.final_approach_geometry.truth_final_gate` |
| `side` | which side the base leg comes from | {left, right}, restricted to the sides the procedure's transitions allow | sign of the cross-track offset at the gate opening |
| `L_pre` | path length flown before the join | ≥ the shortest legal route to `d_join`; extra length only as a lengthening of the pre-final legs | arc length of the track up to `d_join` |

**Fixed-K fly-by waypoints (step 2b, 2026-09-11; §12.3).** The three route parameters say
how long the pre-final path is, which side it comes from and where it joins — not where it
goes (§12.2: vectored chamfer 636 m from the truth's own three). The representation that
does, and stays fixed-size and continuous, is up to K fly-by FIXES: each of the path's
turns as the point where the two legs it joins intersect — the RNAV way of coding a route,
what a radar vector "fly heading X until Y" comes to, and directly the HDG/LNAV modes of an
autopilot (`PlanLabels.waypoints`, `extractors.extract_waypoints`, K = `MAX_WAYPOINTS` = 4;
a reversal is two fixes about its mid-tangent; a plan with fewer turns pads with none — a
fix on the line between its neighbours is no turn). **Each fix carries the speed the
truth had at its turn** (`PlanLabels.waypoint_speeds`: a vector is a heading AND a speed
instruction): the route rounds the corner at that speed's radius and the speed schedule
runs through the fix speeds (`speed_schedule_mps(points=…)`), then decelerates as before.
The route through them is the polyline with each corner rounded (`route.KIND_WAYPOINTS`);
a last fix on the centreline is the turn onto the final and becomes the join. Three
things were tried and measured out first: a turn as (start, heading change) drifts every
later leg when laid at another radius than the truth's (the vectored route came out
4.4 km longer than the plan); a turn's END as the point is ambiguous about the leg
heading it starts; fixes WITHOUT their speeds are rounded at `V_mid`, which the truth
had long left by its base turn — the tracker cut the corners (bank cap on 17 % of steps,
31 % of vectored flights out of the corridor after the join, chamfer 296 m). Whether the
head predicts the fixes (as a distribution, like the other route parameters) is the
step-3 decision; the fixes are the same kind of thing as the join — the controller's
intent.

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
   route's true length (the trombone's mechanism, with the distance L3.e got wrong). As built
   (`outputs/plan/guidance/route.py`, 2026-09-10): the join's intercept is any angle within the
   30° alignment limit (aligned where the length affords it), every turn is sized at the speed
   the schedule has where it is flown, a turn-straight-turn that loops is not a route (the
   chord onto the join is), and a length no hold or dog-leg lays is reported as the route's
   shortfall, never flown as an extra. **v5 — rolled**: with a `next_fix` the route is ONE leg
   — the turn onto the line to the fix at the anchor's speed, the corner at the fix rounded at
   `V_next` — followed by the closing onto the join from there (the aligned join, the final
   from `d_join`) as the fallback beyond the fix; the guidance flies to the fix, the state
   there is the next anchor, and the plan is asked again. Without a next fix the route is
   the closing itself (the 8-number route of step 2). The whole-path form
   (`build_route(waypoints=…)`, §12.3) stays for the oracle and for a scheduler that assigns
   the whole route.
3. **Lateral tracking** — the nominal-law hook (archived 2026-09-09 under
   `archive/nominal_law_hook_2026_09/`; the guidance layer takes it back as its own module, §11):
   line-of-sight to the route, bounded bank,
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

- Supplies: the assigned arrival time and the route — v5: the NEXT instruction (a fix and a
  speed, i.e. a vector), as ATC gives it, step by step; or the whole route (`d_join`, `side`,
  the pre-final length or a delay to absorb) at once. If it assigns nothing, the model returns
  the fan over the next instruction and the arrival-time distribution, never a single guess.
- Gets back: a procedure-conforming, flyable reference for the assigned time and route; the
  operating parameters behind it (when the aircraft slows, how fast, how high); the unabsorbable
  delay X if the assignment is infeasible; and, when nothing is assigned, the arrival-time interval
  (~30 s wide for straight-in traffic, ±70 s for vectored traffic until the join is decided) and the
  fan over routes.
- The multi-aircraft demonstration (assign times and joins to several arrivals, build references,
  check separation) follows directly, because every reference is a plan the scheduler can read.

## 6. Training

- **Supervision**: the five operating parameters extracted from each observed track, regressed
  directly in their own units (point and quantile heads); v5: the next instruction
  (`next_fix`, `V_next`, `next_is_join`) at the anchor, point + distribution, censored only
  where the flight is already established at the anchor (then there is no instruction to
  predict, and the sample supervises the operating parameters alone); the three whole-path
  route parameters are not a training target. No inverse-dynamics teacher, no imitation weight, no position-unit
  balancing. Bank fidelity is not a training target; it is a property of the guidance layer.
- **Not through the guidance layer**, at first. This week's evidence (L1.c; the 09-06 training-
  through-hook arms) says training through a corrective layer teaches the network to lean on it.
  The plan is supervised directly; the guidance layer runs at inference. End-to-end fine-tuning
  through a differentiable guidance layer is a later, gated experiment.
- **Inputs**: a 60 s history window (`seq_len` 30 at `dt_s` 2; v5 — the next instruction needs
  the state and its trend, and the 120 s window put the anchor 12 km into the slice, past the
  vectors for 59 % of flights, §12.3) in the threshold-anchored ENU chart (the frame ablation
  showed the anchor is the model's runway knowledge), the procedure's skeleton as conditioning (the fix
  distances and limits, so the plan head knows its ranges), and the scheduler's inputs as
  conditioning (CTA conditioning exists; join conditioning is the same mechanism).
- **Anchors**: the remaining-path-uniform random-anchor sampler with the scheduler fix (A0.b), so
  the plan is re-issued at any point of the approach — under v5 that is the training
  distribution itself: every sample is "here, now, what is the next vector"; the two-model
  rule stays available.
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
| reference: fully flyable, true time (truncated at threshold) | 46 % (L3.e-r / L3.f-r stack, corrected cut) | ≥ 95 % |
| reference: fully flyable, +60 s; X reported | 20 % | ≥ 90 % |
| reference: established on the final by the assigned time | 70 % of all, ~20 % of vectored (L3.f-r +60) | 100 % by construction, or X reported |
| reference: corridor / glidepath / fix-limit violations on the final | 12.7 % lateral, straight-in, true time (L3.e-r +0) | 0 % by construction |
| bank | skill 0.726 (native32) | not a target; bank RMS ≤ 0.5° on straight-in |

Provenance of the "today's best" column, re-read off the artifacts on 2026-09-10: the
straight-in chamfer "65–109 m" mixes `L1_native32` (109.0 m, the prediction result) with the
`cta=given` oracle arm `L3_cta` (65.9 m), and the hooked arms already read 42.3 m there; the
arrival-time MAE 10.3 / 23.9 s is `B1_quantile`'s; the "~70 % established" row is L3.e-r +60 s.
The gates stay as written; the first row's chamfer gate is read against 109 m.

The oracle ceiling (§12.2, the truth's own plans through the guidance) reads against this
table: vectored ADE 1591 m (row 1: inside the gate), straight-in chamfer 34 m (inside),
arrival-time MAE 4.8 s straight-in / 14.0 s pooled (inside), fully flyable 99.7 % (inside),
established 99.7 % (4 flights of 1404 arrive after the horizon), corridor violations after the
join 1.4 % (NOT yet 0 % — the tracker's entry, not the plan's; §12.2).

**v5 adds the single-step reading**, because a rolled prediction compounds: at every anchor of
the anchor grid (and at the 60 s anchor), per stratum, the next-fix error (its position in
runway axes, its speed, the time to it) against the truth's next fix, the share of
`next_is_join` right, and the fan's coverage of the truth's next fix — read against the
trivial baseline (the stratum's median next fix). The rolled reading is then the existing
one: from a fixed anchor, fly leg by leg to the threshold and score the whole trajectory as
above (the oracle of §12.3 is its ceiling, the plan-route oracle of §12.2 its floor).

Vetoes: the plan head's parameters worse than trivial baselines (the stratum medians — measured,
§12.1's last column; v5: the next fix no better than the stratum's median next fix) — the
learned part is not learning; or the oracle ceiling (§9 step 2) worse than today's model — the
parametrisation is too coarse.

## 8. Reuse, new work, risks

- **Reused**: the point-mass rollout and actuator model; the nominal-law (from the archive, §11),
  barrier, speed-floor and trombone hooks as guidance parts; the corridor geometry and on-final gate; the CIFP procedure
  parsing from the optimizer; the CTA conditioning; the quantile head and split-conformal
  calibration; the random-anchor sampler and anytime replay; all readouts and the publisher; the
  `evaluation` package's procedure gates.
- **New**: the plan-parameter extractors (`outputs/plan/extractors.py`, bound to a window set
  through the strategy's `bind_windows`, never a line in `data/dataset.py`); the procedure
  reader as model conditioning and guidance skeleton; the plan head; the route builder with the time-closure step;
  the assembly of the hooks into one controller; the pooled-airport run.
- **Risks and their tests**: (1) the parametrisation may be too coarse for vectored flights with
  more than one turn — allow one or two optional extra waypoints, gated by a readout of how many
  truths need them (MEASURED 2026-09-11, §12.3: it is — and K ≤ 4 fly-by fixes close it,
  vectored chamfer 636 → 268 m from the truth's own fixes); (2) procedure conformance versus what pilots actually fly — real KRDU vectored
  arrivals are radar vectors, so a conforming reference is a worse prediction of them; the two
  evaluations make this explicit rather than hiding it, and the fan carries the prediction claim
  there; (3) the join is as hard to predict as the intent — which is why it is assigned or a
  distribution, never a point; (4) the procedure data may be incomplete for some runways (KRDU 32
  has no published vertical path) — the skeleton then falls back to the runway geometry and says so.

## 9. Steps

1. Procedure reader + extractors: the skeleton for every runway in the fleet; the eight plan
   parameters for every KRDU arrival; their distributions; how well airport medians predict them;
   how many observed joins fall on a published transition versus a vector (CPU, one day).
   **DONE 2026-09-10** — `python run_ts.py plan_extractors`, artifact
   `4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step1_extractors/`; §12.1.
2. Oracle ceiling: fly the *true* plans through the guidance layer on the skeleton and measure
   against the observed tracks and against the procedure. If the ceiling is worse than today's
   model as a prediction, revise the parametrisation before training anything.
   **DONE 2026-09-10** — `python run_ts.py plan_oracle`, artifact
   `4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step2_oracle/`; §12.2.
   **2b (2026-09-11)**: the fixed-K fly-by waypoints oracle (`--route waypoints`) and the
   oracle at earlier anchors (`--anchor-s 60`, `--anchor-km 20`); artifacts
   `4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step2b_*/`; §12.3.
3. **v5 — the single-step plan head on KRDU** (one seed, both evaluations), in three parts:
   (a) the measurement first — from the step-2b labels, how far ahead the next fix lies
   (distance, time) per stratum and anchor, and how often the next thing is the join: the
   lead a single step must predict (CPU, an hour); (b) the single-step oracle — the truth's
   next instruction flown leg by leg from the 60 s anchor and from L−1, re-planned at each
   fix (the rolled form of §12.3's oracle; its numbers should match §12.3's waypoints columns,
   and where they do not the rolling itself is the difference); (c) training: the operating
   parameters as before, `next_fix` / `V_next` / `next_is_join` as point + distribution, 60 s
   window, remaining-path-uniform anchors; the single-step and the rolled readouts of §7.
   **(a) DONE, (b) DONE, (c) BUILT and run small-scale, 2026-09-11 (`dev-plan-next`; §12.4)** —
   `run_ts.py plan_next_readout` (3a; `step3a_next_readout/`), `run_ts.py plan_oracle --route
   next` (3b; `step3b_rolled_l1/`, `step3b_rolled_a60s/`), `prediction_output=plan` with
   `outputs/plan/{labels,model,strategy}.py`, `plan_oracle --policy model` (the rolled reading)
   and the readout's HEAD columns (the single-step reading); the point heads only — the
   distribution over the next fix (the fan) is the next item on (c).
   **(d) v5.1 — receding-horizon, lockstep rolling (2026-09-11 night; `dev-plan-lockstep`).**
   The unit of rolling is a TIME STEP, not an instruction: every `LOCKSTEP_S` = 30 s the
   policy is asked again from the aircraft's current pose and window — the oracle's policy
   holds the truth's next instruction until the aircraft has executed it (past its fix, on
   its heading), the head re-predicts it — the route to the current instruction is re-laid
   from where the aircraft is, and the next 30 s is flown. Every flight in a group is
   stepped together (`fly_lockstep`: one guidance rollout per step for the whole group,
   ~10 segments each, one head forward per step) — the batching the user asked for; the
   leg-at-a-time form stays as `--rolling leg` with its §12.4 numbers. What (d) must
   measure, in this order: (1) the lockstep ORACLE at L−1 and 60 s against §12.4's leg form
   (the same instructions, re-laid every 30 s: the rolling cost should fall, and the wall
   time by ~10×); (2) the full head re-asked every 30 s at 60 s, paired with its ceiling —
   the number §12.4's 3781 m is re-read against, with the head unchanged (no retraining);
   (3) the step length as an axis (15 / 30 / 60 s) only if (2) moves. Gate: the head's
   rolled vectored ADE against native32's 2870 m and the straight-in chamfer against 109 m,
   both at the 60 s anchor.
   **(e) v5.2 — the head trained on rolled windows (planned 2026-09-11 night; `dev-plan-rolled`).**
   §12.5's reading: from its second step on the head is asked on windows of its own flown
   rows, and it was trained on observed windows only — its orders jitter, and 29 % of vectored
   flights ran to the cap. The fix is on the training side, one instrument, two rounds:
   (1) the ROLLED-WINDOW TABLE — `run_ts.py plan_rolled_windows` flies every flight of a
   split in lockstep from L−1 and records, at EVERY step, the head's input window there
   (`rolled_history`: the observed track to the anchor continued by the flown rows) with the
   target vector the truth defines AT THAT STATE: the same vector `targets_from_labels` builds
   at an observed anchor, read about the aircraft's pose (the fix ahead / across it, the
   heading on; the truth's arrival time less the time flown; the schedule coordinate as the
   path to go through the fix in force — or to the join and down the final; the flight's own
   speeds, deceleration and join distances, capture height). Under `--policy truth` the flown
   states are the oracle's — closed-loop windows along the truth's route, the ceiling's own
   inputs; under `--policy model` a plan checkpoint's — the head's own states, each labelled
   by the truth's queue at that state (the instruction executed once the aircraft is past its
   fix on its heading; a fix behind an aircraft on the final is none ahead, `fly_lockstep`'s
   own rule): the expert at the learner's state, DAgger's aggregation, `--extend` carrying the
   previous table into the new one. One file per table (`.npz`, the provenance inside), bound
   to the window contract (L, dt, channels, the target contract) and to the split's flights.
   (2) TRAINING on it — `plan_rolled_windows_path` + `plan_rolled_share`: per flight per epoch
   the draw is, with probability `share`, one of the flight's rolled windows (uniform over its
   steps, from the same per-flight per-epoch digest discipline as the anchor draw, salted
   apart) and otherwise the observed anchor draw, unchanged; the batch carries `plan_rolled`,
   the epoch record the share realised. The loss is unchanged. The table must cover every
   train AND val flight (no partial mode: a head trained on rolled windows for some flights
   and not others is a mixture nobody asked for); the val split's rolled windows are scored
   every epoch as a READOUT beside the observed objective (`plan_rolled_validation`); the
   checkpoint stays selected on the observed L−1 objective until that readout says the two
   part. What to measure, in order: (1) round 0 — the table from the truth policy (train +
   val), the head at share 0.5 under step 3c's recipe, the rolled prediction in lockstep at
   the 60 s anchor and L−1 (§12.5's 5391 m / 4991 m and the leg form's 3781 are the numbers
   to beat, the lockstep ceiling 2184 the bound; established 55 % against 78 %); (2) round 1 —
   the table from round 0's head (`--policy model`) appended to round 0's, the head
   retrained; (3) the share as an axis (0.25 / 0.75) only if (1) moves. Gate as 3(d)'s: the
   rolled vectored ADE against native32's 2870 m and the straight-in chamfer against 109 m at
   the 60 s anchor. **BUILT and MEASURED 2026-09-11 night (§12.6)**: share 0.75 — vectored ADE
   3641 m (gate not passed against 2870), straight-in chamfer 44 (passed), established 94 %;
   the share is the lever, DAgger's first round is not.
4. Assigned-time and assigned-join conditioning; the +60 s delay test with X reported.
5. Two seeds; pooled five-airport training; KSJC replication.
6. The multi-aircraft scheduler demonstration.

## 10. What stops

No more arms on the latent dose axis, teacher replacement, position units, hook doses or the
duration weight. L3.e-r and L3.f-r have run (the route builder's specification test: the
reference-rollout surplus estimate is the better one — 7–10 points more flights on the final,
9–13 % fewer stall and thrust samples, `tromboneDelayS` 386 → 136 s — but both arms fail nearly
every gate, and the median flight gets no stretch). One pre-registered check remains on that line:
one L3.f-r arm re-flown under the A-4 fix (§11). Nothing else from the current line is queued.

## 11. Where this lands in the package (updated 2026-09-10 — the review is built)

The package review of 2026-09-09 measured why the package was heavy: every new prediction path
had cost edits in ten places (`TSConfig` + validators, the CLI, the run grammar, the dataset, the
objective, the forecast, the export, the training loop, a runner, a test file). Its target — one
`OutputStrategy` per path, one typed config view per output, one test file — is the package as it
stands on `dev-leg-ctrl` since 2026-09-10 (`dfce744`): the §2 fixes, the package (§4.1), the §5
freeze, the config views (§4.3), the strategies (§4.2), the loop/predict extraction (§4.4), the
runners (§4.5), the tests (§4.6) and the grouping by plane (`data/ geometry/ backbone/ outputs/
training/ inference/ cli/ experiments/`), each landed with the full suite and the stored-run
census as acceptance (992 tests; 219 stored configs, 112 load, 0 names moved). This design is the
fourth path, and what it adds is three things: one package under `outputs/`, one view in
`config.py`, one test file per topic.

**Plan-and-guidance is `outputs/plan/`.** `config.PREDICTION_OUTPUTS` gains `PREDICTION_PLAN =
"plan"` (and `PREDICTION_OUTPUTS_AVAILABLE`, since a new run may select it), and the lazy registry
`outputs._STRATEGY_MODULES` gains `"ts_transformer.outputs.plan.strategy"` — the registry refuses
to import if the two sets differ, and `tests/test_architecture.py` asserts every output has a
strategy. **Its config is a fourth VIEW, not a sum type**: the review's §4.3 was built as typed
views over the flat `TSConfig` (every stored checkpoint, `run_naming` and the CLI read the flat
dict), so `PlanOutput(OutputSpec)` declares the fields it OWNS, `_OUTPUT_VIEWS["plan"]` names it,
each field lives on `TSConfig` with its default, `_validate_ownership` refuses a plan field off
its default under another output (and a control field under `plan`), `from_dict` normalises them
on load, and `_check_view_partition` fails the import if a field is on no view. Every CLI flag is
named after its field (`cli.common.CLI_CONFIG_FIELDS`), and every field is in a `run_naming` list
or excused by name — the plan path names its runs through the shared grammar's `output` token
plus its own spelled fields (`plan · backbone · dynamics · loss · meta`); there is no
per-strategy grammar.

The strategy's members (`outputs/base.py`, the interface as built) map onto the sections above:

| `OutputStrategy` member | what it holds here | section |
|---|---|---|
| `name`, `view` | `"plan"`, `PlanOutput`: the five operating-parameter ranges, the route-parameter policy (assigned / distribution), the guidance gains, the two loss weights | §3, §4, §6 |
| `anchor_policy`, `eligible_anchors(series, anchors)` | which observed states the guidance layer can start from (the control path's airborne rule is the precedent); the remaining-path-uniform SAMPLER with the scheduler fix (A0.b) is a `TSConfig` field the loop reads, not the strategy's | §6 |
| `bind_windows(windows)` → `PlanContext(WindowContext)` | the per-flight labels from the extractors (the eight plan parameters, each with its range and its clamp) and the assigned runway's skeleton (fix distances, altitude and speed limits, LPV width); `row(i)` is the context slot `batch()` carries, `summary` the coverage line. The closure path's `ClosureContext` — labels from a file, refused at zero coverage, never a model input — is the template | §2 (1), §3 |
| `build_model(config, normalizer)` | the plan head over `backbone.adapters`'s forecaster: five point + quantile outputs (`outputs.duration_heads`'s monotone quantile head, reused per parameter), the route distribution head | §3a, §3b |
| `target_contract`, `loss_component_names`, `loss(...)` | direct regression in the parameters' own units; pinball for the quantiles; the route distribution's likelihood. Every term is in `loss_component_names` or the first batch raises | §6 |
| `probe_context`, `probe_dense_supervision`, `probe_prediction` | the `--batch-size auto` probe's batch, carrying every key `PlanContext.row` carries (review B-1's rule, pinned the way `tests/test_supervision_terms.py` pins the control path's) | — |
| `check_trainable`, `training_input`, `epoch_config`, `training_diagnostics`, `epoch_record`, `checkpoint_metadata` | refuse a cohort the skeleton does not cover; no teacher; no per-epoch schedule; the plan-parameter errors against the airport medians (the §7 veto) into `history.json`; the skeleton's source and cycle into `checkpoint_metadata.json` | §7 |
| `forecast(model, series, config, normalizer, anchor, device, options)` | the guidance layer (route builder → lateral / vertical / speed → time closure) driving the point-mass rollout. The scheduler's assignments arrive in `ForecastOptions` — new fields beside `cta_offset_s` / `cta_s` (the assigned time, `d_join`, `side`, the delay to absorb), filled from `PredictOptions` by `cli.predict.parse_predict_options`; a strategy refuses the options that do not apply to its path | §4, §5 |
| `replay(...)` → `Replay` | the validation replay on the plan's own clock (the control path's rollout-clock replay is the precedent) | §7 |
| `record_fields(forecast)` | the plan (with every clamp), the route actually flown, the unabsorbable delay X, the fan — the record's `source` block | §5, §7 |

**What it reuses (the move that reuse forced is DONE — the first commit of
`dev-plan-guidance`, 2026-09-10).** The point-mass rollout and actuator model
(`outputs/dynamics/{backends,rollout}`), the barrier / speed-floor / trombone modules with
their gates and saturation (`outputs/constraints/`), the dimensionless command box and its
newton conversion (`outputs/envelope`), CTA conditioning (`outputs/conditioning`), the
quantile head (`outputs/duration_heads`), split-conformal calibration
(`inference/calibration`), the corridor geometry, on-final gate and crossing rule
(`geometry/final_approach_geometry`), the anytime grid and replay (`data/anchor_grid`,
`experiments/anytime_curve`), the procedure documents through the seam's reader
(`flight_scenarios.procedure_final`, `flight_scenarios.fas_geometry` — the optimizer's
`approach_constraints` is not on the package's import path). The membership rule
(`CLAUDE.md`: a module belongs under `outputs/control/` only if EVERY consumer is
control-specific) is why the first four moved up from `outputs/control/` — pure moves, the
control strategy imports them from the new place, and `tests/test_architecture.py` refuses a
shared part that imports any path. The guidance layer imports those modules as parts of ONE
controller (`outputs/plan/guidance/`), never as post-hoc hooks. The nominal-law hook
(`guidance_laws.py`, `nominal_residual.py`) is ARCHIVED under `archive/nominal_law_hook_2026_09/`
and the archive is off the import path (`tests/test_architecture.py` refuses an import of it),
so the lateral and vertical laws come back as the guidance layer's own modules
(`outputs/plan/guidance/lateral.py`, `vertical.py`), the archive README naming their origin.

**Layering the plan package must keep** (`tests/test_architecture.py` enforces it): only its
strategy seam (`strategy`, `forecast`, `labels`, `loss`) reaches the spine — `training/objective`,
`inference/forecast`, `backbone/adapters`, `data/dataset`; its inner modules may import
`data/dataset` values and the shared `outputs/` parts but not the spine; nothing under it imports
`training/train`, `training/validation`, `training/batching`, `inference/export` or `cli`.
**Nothing in this design touches `data/dataset`, `training/objective`, `inference/forecast`,
`inference/export`, `training/train` or `training/validation`** — each calls
`outputs.strategy(config).<method>` where it used to branch on `prediction_output` (landed
2026-09-10, `947c907`), which is what the previous version of this section was waiting for.

**Runners, readouts, tests.** A runnable experiment is `experiments/<name>.py` behind
`python run_ts.py <name>` (a module with `main()`; `--list` finds it by its docstring's first
line), and measurement code is CODE in the package with tests. So: the extractors are
`outputs/plan/extractors.py` (`experiments.straight_in_residual_readout.decel_distance_km` moves
in with them), the step-1 readout (parameter distributions, airport-median baselines, how many
joins fall on a published transition) is `experiments/plan_extractors.py`, the step-2 oracle
ceiling is `experiments/plan_oracle.py`, the campaign driver `experiments/plan_campaign.py`.
Tests are one file per topic under `tests/`: `test_plan_output.py` for the strategy
(`test_closure_output.py` is the shape), `test_plan_guidance.py` for the controller's
by-construction properties (corridor, glidepath, fix limits, stall, thrust, bank, time closure
with X), `test_plan_extractors.py` for the extractors against synthetic tracks; the shared
fixtures come from `tests/support.py` (`fake_data_provenance`, `terminal_contexts`).

**Order, and what is already done (updated 2026-09-10, `dev-leg-ctrl` at `dfce744`).**
Everything the previous version of this section waited for has landed. The review's step 0 (the
A/B items and the C holes, `e2e1c84`; A-3 / A-4 at `c544db0`, the A-4 check at `a2aa8c4`),
step 1 (the package, `f08d196`), step 2 (§5 — FROZEN rather than deleted: closure, the airport /
runway frames, the fixed-dt grid and truth-intent stay loadable and no new run selects them,
`dddee9a`), step 3 (§4.3 views, defaults unchanged, `6600201`), step 4 (§4.2 strategies,
`947c907`), step 5 (§4.4, `d462c7a`), step 6 (§4.5 runners `ed708de`, §4.6 tests `1e637b9`)
and the grouping (`544c205`, its follow-up `b9c5fa1`). This design's step 3 (the plan head) is
therefore UNBLOCKED; steps 1–2 (the procedure reader, the extractors, the oracle ceiling) are
CPU-only readouts and never were blocked. What is left is the user's go on §9 step 1.

**Two review findings this design must not inherit.** C-12 (the head's load-factor floor 0.2
against the grader's 0.5) — DECIDED 2026-09-09: the grader stays at 0.5, and the guidance layer
commands the load factor inside `flyability`'s envelope (`geometry/flyability.py`, floor 0.5),
never inside the learned head's box in `outputs/envelope.py` (floor 0.2, a search-space
fact of the old path); the §7 "today's best" flyability baselines therefore stay as measured.
C-11 (the corridor geometry's origin) is fixed in `project_onto_final` (threshold-relative under
every frame), and remains the one geometry the procedure reader and the corridor gate must agree
on, so the reader is written against the threshold frame, never the airport reference point.
Naming: `random_train_anchor_min_future_s` (§6's sampler floor) names the run since 2026-09-09
(`anchor-min-future=`, review C-3 decided), so two plan arms differing only in it no longer share
a name; the plan's own fields follow the same import-time rule. Two review items are still open
and can touch a plan campaign: the "name every field against the nearest recipe" grammar change
is deferred (it moves stored names and needs its own relabel pass like C-3's), and the pipeline
runner's `TrainingPlan` keeps its keyword constructor — a plan cell added to `run_ts.py pipeline`
is one more keyword there, not a new copy of the override dict.

## 12. Measurements

### 12.1 Step 1 — the plan parameters on KRDU val (2026-09-10)

Cohort: `L1_native32`'s validation split (1404 flights: 904 straight-in, 497 vectored, 842
established at the anchor), anchor L−1, the observed track from the anchor on.
`python run_ts.py plan_extractors --checkpoint native32=<l1_lowdim_20260907/L1_native32>/checkpoint.pt
--out 4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step1_extractors`. The
skeletons are the five KRDU RNAV(GPS) documents (`R05LY`, `R05RY`, `R23LY`, `R23RY`, `R32`);
none codes a speed limit, and `R32` codes no TCH. Code: `flight_scenarios.procedure_final.
procedure_skeleton`, `outputs/plan/skeleton.py`, `outputs/plan/extractors.py`; reviewed
(opus) before the numbers below — the first readout had read 831 joins that happened BEFORE
the anchor as measured captures, and a deceleration reference that could never be undefined.

**831 of the 1404 flights (59 %) are already established at L−1** — the truth gate is open at
the anchor. Their `d_join` is the anchor's remaining path and their `h_capture`, `side` and
`L_pre` are CENSORED (the window never saw the join), so the three route parameters are
measured on the 573 flights whose join lies inside the window (76 straight-in, 497 vectored).

| parameter | all p50 [p10, p90] | straight-in | vectored | median-baseline MAE (straight-in / vectored) |
|---|---|---|---|---|
| `T` (s) | 186 [146, 524] | 166 [140, 200] | 486 [390, 566] | 19.4 / 57.8 |
| `V_mid` (m/s) | 91.2 [81.6, 116.9] | 88.2 [78.9, 93.1] | 114.0 [105.7, 123.0] | 4.4 / 6.3 |
| `d_decel` (m) | 10020 [7323, 14099] | 10209 [7409, 13549] | 9811 [7169, 16113] | 1992 / 2677 |
| `V_final` (m/s) | 71.4 [64.9, 78.5] | 71.1 [64.9, 78.3] | 71.9 [65.2, 78.9] | 4.2 / 4.2 |
| `h_capture` (m, in-window joins) | 597 [485, 820] | 615 [485, 788] | 591 [486, 820] | 107 / 114 |
| `d_join` (m) | 12627 [10670, 16253] | 12301 [10645, 14260] | 14133 [10868, 18130] | 1130 / 2346 |
| `side` (L / 0 / R, in-window joins) | 224 / 13 / 336 | 17 / 13 / 43 | 206 / 0 / 291 | — |
| `L_pre` (m, in-window joins) | 34858 [2770, 41105] | 1545 [420, 3609] | 35612 [27589, 41478] | 1135 / 5131 |

The last column is the §7 veto's reference: predicting the stratum's own median, per
parameter. A plan head must beat it on the parameters it claims to learn. `d_decel` is where
the ground speed first drops below the scenario TARGET speed + 10 m/s (the residual readout's
rule); one flight never does.

What the numbers change in the design:

- **Straight-in joins are published transitions; vectored joins are radar vectors.** Of the
  joins inside the window, 87.7 % of the straight-in ones (64/73) lie inside the RNP box of a
  coded pre-final leg over the 60 s before the join; 2.2 % of the vectored ones (11/497).
  §8 risk 2 is measured: for 35 % of the cohort a procedure-conforming reference is not what
  was flown, so the fan carries the prediction claim there and the point comparison is only
  fair on the straight-in stratum.
- **`V_final` has no floor at V_ref.** The published V_ref is an AIRSPEED window the
  evaluation reads with the METAR headwind (`evaluation/docs/THRESHOLD_SPEED_GATE.md`); the
  observed final ground speed carries the headwind and the last row is ~380 m short, so a
  ground-speed floor at the type's reference speed is violated by 72 % of the fleet. The
  plan's floor is the stall margin (14 flights under it); the gate grades the flown speed
  as it grades every record.
- **`h_capture` is the height at the LATERAL join, not the glidepath capture.** Against the
  glidepath at the join most flights are still above it (they capture it later), so the
  design's "≤ the glidepath" ceiling is dropped; the bound is the next fix's coded floor and
  ceiling. 207/573 (36 %) of the in-window joins sit below the FAF's 2200 ft floor at the join
  (p10 485 m above the aim point against the floor's 544 m): real traffic below the platform,
  which the guidance must be allowed to fly — the floor is a plan clamp the record reports,
  not a refusal.
- **The vectored pre-final path is long and regular.** `L_pre` p50 35.6 km [27.6, 41.5]
  against a `d_join` of 14.1 km: the route builder must lay 20–30 km of pre-final path, far
  past what the trombone's 45° offset could stretch, and the base leg comes from either side
  (206 left / 291 right) — the route parameters are a distribution, as §3b says.
- **Straight-in flights that do join inside the window join short and from a side**: 73
  flights, `L_pre` p50 1.5 km, 60 of them with a base leg wider than 500 m.
- **`d_decel` p50 10.0 km against the 05L FAF at 10.3 km**: the median flight starts its
  final deceleration at the FAF; "before the FAF if the procedure limits speed there"
  never binds on KRDU (no coded limits).
- **`V_mid` is the MEAN speed over the path held before the deceleration point** (its
  length over its time — the number the time closure needs; a 10–20 km band median, the
  first definition, missed the flown time by 61 s on the vectored stratum whose pre-final
  path is flown at 114 m/s). 288 flights (20 %, nearly all straight-in) are already
  decelerating at the anchor and hold their anchor speed (`v_mid_from_anchor`). `T`'s floor
  (the beeline at the type's approach maximum) is under-run by 160 flights (11 %), which
  flew faster than that maximum somewhere on the way.

### 12.2 Step 2 — the oracle ceiling on KRDU val (2026-09-10, after the step-2 review)

Every flight's OWN eight parameters (§12.1's extractor) laid as a route and flown by the
guidance layer on the shared point-mass rollout, from L−1, for the plan's `T` + 10 % (a late
arrival is measured as late); the record cut at the first threshold crossing on the final, as
every hooked-arm readout is. Same cohort as §12.1. `python run_ts.py plan_oracle --checkpoint
native32=… --out 4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/step2_oracle`
(the pre-review run is kept beside it as `step2_oracle.superseded-20260910T2330Z`).
Code: `outputs/plan/guidance/{route,controller}.py`, `outputs/plan/forecast.py`,
`experiments/plan_oracle.py`.

| metric | all (1404) | straight-in (904) | vectored (497) | today's best (§7, provenance-checked) |
|---|---|---|---|---|
| ADE mean (m) | 713 | 204 | 1591 | 1322 pooled, 2870 vectored (`L1_native32`) |
| ADE p50 / p95 (m) | 251 / 1939 | 158 / 481 | 1382 / 2559 | |
| FDE p50 (m) | 19 | 19 | 20 | |
| chamfer p50 (m) | 43 | 34 | 636 | 109 straight-in (`L1_native32`) |
| Fréchet p50 (m) | 145 | 134 | 2166 | |
| arrival-time MAE (s) | 14.0 | 4.8 | 27.3 | 25.9 pooled (`L1_native32`); 10.3 straight-in / 23.9 pooled (`B1_quantile`) |
| \|Δt\| p80 (s) | 25.0 | 7.1 | 36.5 | |
| signed Δt p10 / p50 / p90 (s) | -33.0 / -5.5 / 2.3 | -8.5 / -3.0 / 3.5 | -41.0 / -26.0 / -12.8 | |
| fully flyable | 99.7 % | 99.6 % | 100.0 % | 46 % (the three-hook stack, true time, cut) |
| established at the threshold | 99.7 % | 99.8 % | 99.6 % | ~70 % (L3.e-r +60 s) |
| corridor violation after the join | 1.4 % | 0.3 % | 3.4 % | 12.7 % lateral, straight-in (L3.e-r +0) |
| glidepath-window violation after the join | 10.0 % | 12.1 % | 6.0 % | |
| route laid | | final only 886, direct 16, held heading 2, dog-leg 0 | held heading 360, direct 130, dog-leg 7 | |
| route shortfall p50 (m); routed flights over 500 m | 0; 15 of 518 | 0; 3 of 18 | -2; 12 of 497 | |
| join intercept \|p50\| / \|p90\| (°, routed) | 1.4 / 21.8 | 1.5 / 30.0 | 0.0 / 18.0 | |
| route time − T, p50 (s) | -5.4 | -2.8 | -21.1 | |
| bank over the 25° cap / thrust at idle (share of steps) | 2.6 % / 9.6 % | 0.5 % / 12.1 % | 6.5 % / 5.0 % | |
| barrier gated / clamped (share of steps) | 80.7 % / 8.7 % | 99.3 % / 10.4 % | 46.8 % / 5.5 % | |
| capture height clamped into the window | 26.9 % | 3.3 % | 69.2 % | |

The pre-review run (the same cohort, kept as evidence) read ADE 1113 / 210 / 2710 m, ADE p95
3746 / 489 / 9543 m, chamfer 43 / 34 / 833 m, established 96.2 / 99.7 / 89.9 %, corridor left
after the join 14.2 / 0.9 / 38.6 %, glidepath 11.1 / 11.3 / 10.7 %. What moved it: the route
builder's join heading was bent twice (a previous choice's intercept re-added to the course),
its dog-leg bisection landed on the wrong branch of a non-monotone length (10–23 km MORE
path than the plan on 3 of 48 smoke flights, arriving 55–200 s late), five discrete
intercepts could not lay a downwind flight's length (2–6 km short), the base turn at the
anchor's speed jumped by a circumference where the real path lies, and a plan whose join
lay a few hundred metres ahead of a pose beside the centreline was routed through a
12–24 km teardrop the tracker then cut through (29 of 73 routed straight-in flights). All
six are the route's, not the controller's, and each is a rule in
`outputs/plan/guidance/route.py` now (`CLAUDE.md` names them).

What it says:

- **The veto does not fire, on both strata.** On the straight-in stratum the parametrisation
  reproduces the observed track far inside today's model (ADE 204 m, chamfer 34 m,
  arrival time 4.8 s MAE); on the vectored stratum the truth's own three route
  parameters now lay a path that beats today's model as a prediction (ADE 1591 against
  2870 m, Fréchet 2166 m) — but its chamfer (636 m) says the same thing the
  pre-review run said: with the plan handed the truth, `d_join`, `side` and `L_pre` do not say
  WHERE the 35 km of pre-final path go, only how long they are. §8 risk 1 stands measured:
  the vectored point claim needs the extra waypoints or stays with the fan, and step 3
  should be read on the straight-in stratum as the point claim and on the vectored one
  through the distribution.
- **The reference reads as designed on flyability and arrival**: 99.7 % fully flyable
  (4 straight-in and 0 vectored flights with 11 stall rows between them), 99.7 % established
  at the threshold (4 flights arrive after the horizon), against 46 % and ~70 % for
  the hook stack.
- **The corridor is nearly by construction; the glidepath is not yet.** After the join
  1.4 % of flights leave the design corridor (20 flights, excess p50 59 m against a
  half-width of ~260 m at 12 km — the barrier composed on the final clamps 8.7 % of steps) and
  10.0 % the glidepath window (140 flights; excess p50 68 m straight-in, 5 m
  vectored — the height law's gain on the from-above capture). The glidepath entry is
  controller tuning (the capture-descent angle, the height gain), not parametrisation — a
  step-3 sub-task before the reference claim is quoted; §7's "0 % by construction" is read
  against it.
- **Timing**: the speed schedule (`V_mid` held, a 0.5 m/s² deceleration to `V_final`)
  reproduces straight-in arrival times (median -3.0 s, p10/p90 -8 / 4 s); vectored
  flights arrive -26 s early at the median with the route laid to the plan's length
  (shortfall p50 -2 m; 12 of 497 routed vectored flights more than 500 m off,
  where no hold or dog-leg reaches the length and the gap is reported), so the schedule's
  speed over the vectors is higher than the truth's — the time closure's own reading
  (`route time − T`, -21 s) says the same, and the assigned-time stretch of step 4
  absorbs exactly this residual by construction.
- **The route's own diagnostics to watch in step 3**: the join intercept (0° p50 on the
  vectored stratum, 18° p90 — the aligned join where the length affords it), the
  15 routed flights of 518 whose length no route lays (3 straight-in,
  12 vectored — the gap reported, never flown as an extra), and the thrust at idle on 9.6 % of steps (a
  deceleration the airframe's drag cannot fly at the schedule's rate).
- **Two step-1 definitions were settled by the first run**: `V_mid` as the mean speed over
  the held path (a band median missed the vectored time by 61 s), and the deceleration as a
  0.5 m/s² law rather than a linear ramp (the ramp arrived 13 s early on straight-in).

Decision for step 3 (the user's): train the plan head as designed (five operating
parameters as point + quantile, the route as a distribution), reading the point claim on
the straight-in stratum and the vectored stratum through the fan — or first add the
optional extra waypoints of §8 risk 1 so the vectored point claim has something to learn.
Either way the guidance's glidepath entry above is on the critical path for the reference
claim, and costs no training.

### 12.3 Step 2b — the fixed-K waypoints oracle, and the oracle at earlier anchors (2026-09-11)

Same protocol as §12.2 (the truth's own plan flown, cut at the first threshold crossing on
the final, the strata fixed at L−1), on three anchors and two route representations.
`python run_ts.py plan_oracle … --route {plan,waypoints} [--anchor-s 60 | --anchor-km 20]`;
artifacts `step2b_waypoints_l1`, `step2b_plan_a60s`, `step2b_waypoints_a60s`,
`step2b_plan_r20km`, `step2b_waypoints_r20km` beside `step2_oracle`. The anchors: **L−1** is
the end of the 120 s window (median remaining path 13.4 km, 59.2 % of plans
censored — joined before the anchor); **60 s** is the one row 60 s after the slice starts,
every flight — the anchor a 60 s lookback would give (median remaining path
18.7 km, 45.6 % censored, 0 flights without one);
**20 km path** is the anytime grid's 20 km REMAINING-PATH bin, each flight's own nearest
sample (1399 flights have one; 5 do not) — note that remaining path is arc
length, so for a vectored flight this anchor is LATER than L−1, past most of its vectors,
and its 42.5 % censoring says so. The oracle needs no lookback (the control inversion's
{ANCHOR_CONTROL_SAMPLES} rows only); a model at these anchors would.

**Vectored stratum** (497 flights at L−1; "plan" = the three route parameters, "waypoints" = those plus the truth's own K ≤ 4 fixes):

| metric | L−1, plan | L−1, waypoints | 60 s, plan | 60 s, waypoints | 20 km path, plan | 20 km path, waypoints |
|---|---|---|---|---|---|---|
| flights in the stratum | 497 | 497 | 497 | 497 | 497 | 497 |
| ADE mean (m) | 1591 | 1159 | 2944 | 1705 | 746 | 989 |
| ADE p95 (m) | 2559 | 2595 | 6593 | 3173 | 910 | 1850 |
| chamfer p50 (m) | 636 | 268 | 1454 | 887 | 80 | 64 |
| Fréchet p50 (m) | 2166 | 1648 | 4392 | 3022 | 344 | 286 |
| arrival-time MAE (s) | 27.3 | 16.1 | 33.6 | 19.7 | 9.5 | 10.5 |
| route time − T, p50 (s) | -21.1 | -6.5 | -25.0 | -11.6 | -5.1 | -2.3 |
| fully flyable | 100.0 % | 99.6 % | 99.8 % | 99.6 % | 99.6 % | 99.6 % |
| established at the threshold | 99.6 % | 99.0 % | 98.8 % | 99.0 % | 98.4 % | 97.2 % |
| corridor violation after the join | 3.4 % | 19.9 % | 5.4 % | 19.7 % | 4.8 % | 14.5 % |
| glidepath-window violation after the join | 6.0 % | 8.2 % | 6.8 % | 9.1 % | 31.6 % | 30.2 % |
| waypoints per plan | — | 2.69 | 2.60 | 2.60 | 1.66 | 1.66 |
| route through waypoints | — | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 81.5 % |
| bank over the 25° cap (share of steps) | 6.5 % | 15.3 % | 6.5 % | 14.1 % | 8.6 % | 13.1 % |

**Straight-in stratum** (904 flights at L−1):

| metric | L−1, plan | L−1, waypoints | 60 s, plan | 60 s, waypoints | 20 km path, plan | 20 km path, waypoints |
|---|---|---|---|---|---|---|
| flights in the stratum | 904 | 904 | 904 | 904 | 899 | 899 |
| ADE mean (m) | 204 | 194 | 325 | 350 | 384 | 412 |
| ADE p95 (m) | 481 | 476 | 774 | 920 | 869 | 1043 |
| chamfer p50 (m) | 34 | 34 | 43 | 42 | 45 | 43 |
| Fréchet p50 (m) | 134 | 134 | 153 | 152 | 160 | 157 |
| arrival-time MAE (s) | 4.8 | 4.8 | 8.1 | 8.4 | 9.7 | 10.3 |
| route time − T, p50 (s) | -2.8 | -2.8 | -5.1 | -4.8 | -6.2 | -6.2 |
| fully flyable | 99.6 % | 99.6 % | 99.9 % | 99.9 % | 100.0 % | 100.0 % |
| established at the threshold | 99.8 % | 99.9 % | 99.2 % | 99.2 % | 98.6 % | 98.9 % |
| corridor violation after the join | 0.3 % | 0.6 % | 0.7 % | 2.9 % | 2.4 % | 4.4 % |
| glidepath-window violation after the join | 12.1 % | 11.6 % | 35.4 % | 36.8 % | 39.5 % | 40.6 % |
| waypoints per plan | — | 0.07 | 0.31 | 0.31 | 0.35 | 0.35 |
| route through waypoints | — | 1.8 % | 0.0 % | 17.4 % | 0.0 % | 21.4 % |
| bank over the 25° cap (share of steps) | 0.5 % | 0.5 % | 1.8 % | 2.2 % | 2.4 % | 2.7 % |

What it says:

- **The fixes close the shape gap.** From the truth's own fixes the vectored path is
  reproduced to a chamfer of 268 m (three route parameters: 636 m) and a Fréchet
  of 1648 m (2166 m), with 2.69 fixes per vectored plan (the arrival time is the
  next bullet). §8 risk 1 is answered for the representation: the parametrisation was
  too coarse for the vectored stratum, and K ≤ 4 fixes carry its shape wherever the
  guidance can follow their corners (the bullet after next).
- **The fix speeds close the timing too.** With each fix's speed on the schedule the
  vectored arrival-time MAE is 16.1 s (27.3 s under the plan's own law) and the
  route time − T median -6.5 s (-21.1 s): the −21 s of §12.2 was the schedule
  holding `V_mid` over the vectors where the truth had slowed. Without the speeds the fixes
  alone were rounded at `V_mid`'s radius, cut by the tracker (bank cap on 17 % of steps) and
  overshot at the centreline (31 % of vectored flights out of the corridor after the join,
  chamfer 296 m) — measured on the first full run of the day.
- **The residual is bimodal, and it is the corner, not the fix.** Under the waypoints route
  the vectored chamfer splits by the bank cap: the flights the tracker can follow (cap
  under 15 % of steps) sit at ~85 m, the rest at ~900 m; the corridor is left after the
  join by 19.9 % of vectored flights (3.4 % under the plan route), the cap
  binds on 15.3 % of steps (6.5 %), and 2595 m is the ADE p95. A fly-by
  corner at the fix's speed and a 20° bank needs legs twice its tangent long; where the
  truth's legs are shorter the corner is rounded tighter than the aircraft can fly and
  the tracker overshoots it. What the truth did there was not a fly-by of two straight
  legs — a continuous turn, a slower speed, or a wider bank — so the next step on this
  representation is the corner itself (a fly-over where the legs are short, or the
  truth's radius as a fourth number per fix), not more fixes: a 0.5°/s rule over ±3 rows
  that reads the wide turns of a fast vector was tried and measured worse on the smoke
  (chamfer 85 → 106 m, twice the fixes dropped).
- **The anchor decides what is predicted.** At L−1 59.2 % of plans are censored (the
  flight joined inside the window) and the route parameters exist for the rest only; at
  the 60 s anchor 45.6 % are — the vectors are ahead of the aircraft there, which is where a
  plan head has something to predict. The 20 km remaining-path bin is the wrong tool for
  that question (it is later than L−1 on a vectored track: 42.5 % censored); the earlier
  anchor for step 3 is a shorter window, not a nearer bin.

**Read for v5 (2026-09-11).** Three things §12.3 settles for the single-step design: the
fixed-K fixes with their speeds are the representation a rolled step is read from (its
ceiling, chamfer 268 m and arrival-time MAE 16 s on the vectored stratum from the truth's
own fixes; the plan-route form is the floor); the residual is in the corners, half the
vectored flights within ~85 m and half ~900 m where a fly-by at the fix's speed does not
fit the leg — the next item on the guidance, not on the head; and the anchor for step 3 is
the 60 s window's (46 % censored against 59 % at L−1), which is why v5 shortens the window.


### 12.4 Step 3 — the single-step head: the readout, the rolled oracle, the first head (2026-09-11)

`dev-plan-next`. Three instruments and one path, in §9's order. The grading changed first
(the truth graded itself a violator): **the glidepath window binds inside the FAF only; before
it, on the final, the coded floor at the next fix applies** (`experiments.plan_oracle.
corridor_verdicts`, the optimizer's `prefaf_floor_m` rule; oracle schema v2), and every table
grades the truth's OWN future rows under the same rule beside the flight's (`… the truth`
rows) — on the 48-flight smoke the truth fails the pre-FAF floor on 14/48 (vectored aircraft
are assigned altitudes below the coded IF floor; the 30 m tolerance of the observed
evaluation) and the glidepath window on 3/48, so neither share is a gate: a flown share
reads against the truth's.

**3a — the next-instruction readout** (`step3a_next_readout/`, KRDU val 1404, the fixed 60 s
anchor and 5 remaining-path-uniform random anchors per flight, seed 1337, 60 s of truth
after every anchor):

| | 60 s anchor, all | 60 s, straight-in | 60 s, vectored | random, all (7020) | random, vectored (2485) |
|---|---|---|---|---|---|
| censored (already on the final) | 45.6 % | 70.8 % | 0 % | 61.3 % | 17.0 % |
| the join is next (of open) | 3.4 % | 9.8 % | 0 % | 3.6 % | 0.9 % |
| fixes ahead (mean, open) | 2.07 | 1.07 | 2.60 | 2.12 | 2.46 |
| next fix: lead path p10 / p50 / p90 | 1.8 / 29.5 / 37.9 km | 1.0 / 3.2 / 6.3 km | 25.8 / 33.4 / 38.7 km | 1.2 / 7.7 / 32.4 km | 1.5 / 13.3 / 34.3 km |
| next fix: lead time p10 / p50 / p90 | 18 / 236 / 302 s | 10 / 32 / 63 s | 200 / 266 / 308 s | 12 / 70 / 258 s | 14 / 112 / 272 s |
| next fix: \|across\| p50 | 4.4 km | 2.2 km | 5.2 km | 1.5 km | 1.3 km |
| speed change to it p50 | −21.8 m/s | −6.7 | −28.5 | −10.0 | −11.3 |
| median-baseline fix error p50 / p90 | 11.0 / 31.0 km | 2.2 / 4.9 km | 5.1 / 15.9 km | 8.9 / 25.5 km | 12.5 / 21.1 km |
| next-is-join majority accuracy | 0.966 | 0.902 | 1.000 | 0.964 | 0.991 |

What the single step has to predict: at the 60 s anchor a vectored flight's next fix lies
25–39 km and 3.3–5.1 minutes ahead (the slice enters on a long downwind; the first turn is
the base turn), and the trivial baseline misses it by 5 km at the median; read at random
anchors over the approach the lead is 13 km / 112 s at the median and the baseline 12.5 km.
The join is almost never the next thing on a vectored track (0–1 %), so the "no fix
ahead" flag carries the ESTABLISHED state (below), not the join. Straight-in flights at
60 s are on the final on 71 % and their next fix (the turn onto the final) 3 km / 32 s away.

**3b — the rolled oracle** (`plan_oracle --route next`: the truth's own instructions flown ONE
AT A TIME — each leg to the first pass of its fix, the aircraft re-anchored there, the next
instruction laid from where it is, the closing onto the join and the final last; `outputs/
plan/forecast.fly_rolling`, `fly_legs`). What rolling had to get right, each measured on the
48-flight smoke before the full run: a leg is timed to the fix's FIRST pass, not the nearest
route point (the route runs on past the fix and can come back near it — the leg flew the
loop); the remaining path at the next anchor is the INSTRUCTION's, not the leg's inflated
route's (read off the route, the closing was asked for 20 km of pre-final path and flew a
loop); the closing's join is never behind the aircraft (the converge rule; a Dubins path
back to a pose behind it otherwise); an instruction is a VECTOR — the fix AND the heading
on after it (`Instruction`: fly-over semantics overshot every corner and the re-projection
skipped half the fixes) — and carries the height at the fix (each leg descends to it by its
fix: without it the leg flew to the plan's capture height over its inflated route and left
the window on 62 % of vectored flights); a last fix at the join heads down the course.
Three more came from the FULL L−1 run's worst flights (27 % of the vectored stratum
worse than the whole-path flight by over 1 km, four of them by 9–14 km with 30–40 km of
closing path — loops): a leg whose heading on points at the join gets NO extension beyond
its fix (the 8 km onward point overshot the join and the polyline looped back to it); a
leg is flown until the AIRCRAFT has executed the instruction — past the fix and on the
heading given within 5° (`turn_done_row`; the tracker lags the route by its 8 s
look-ahead, and cut on the route's clock the next leg started 60° off its heading 3 km
short of the join, which the plan-route builder answered with a 25 km loop); and the
closing from a pose heading at the join inside the 30° intercept is the polyline onto
the join with its corner the fly-by point BEFORE it (`d_join + tangent`, so the arc onto
the course ends at the join as the whole-path route's last corner does — at the join
itself the arc ended past it and the corridor grading flagged it). The rolled flight
reports where it was capped (`planCappedBy`: the 6-leg cap, the time cap) and the legs
whose turn the aircraft never completed (`planTurnsIncomplete`). The 48-flight smoke,
vectored stratum, against the whole-path waypoints flight on the same 48, at L−1:

| vectored, 48-flight smoke, L−1 | rolled (first form) | rolled (final) | whole path |
|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 1310 / 144 / 766 m | 915 / 119 / 650 m | 862 / 85 / 379 m |
| arrival-time MAE | 29.0 s | 16.7 s | 14.1 s |
| established / corridor after the join | 96 % / 3.8 % | 100 % / 11.5 % | 100 % / 7.7 % |
| glidepath (in FAF) flown / the truth | 11.5 % / 11.5 % | 7.7 % / 11.5 % | 3.8 % / 11.5 % |
| floor (pre-FAF) flown / the truth | 26.9 % / 50 % | 23.1 % / 50 % | 0 % / 50 % |
| legs / skipped / turn not completed | 3.69 / 0 % / — | 3.58 / 11.5 % / 7.7 % | 1 / — / — |

Full KRDU val (1404 flights at L−1, 1404 at the 60 s anchor; `step3b_rolled_l1/`,
`step3b_rolled_a60s/`; the whole-path columns are §12.3's artifacts, whose glidepath
share is the v1 join-graded one and whose floor share does not exist):

| vectored, KRDU val | rolled, L−1 | whole path, L−1 (§12.3, v1 grading) | rolled, 60 s | whole path, 60 s (§12.3, v1 grading) |
|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 1492 / 324 / 1800 m | 1159 / 268 / 1648 m | 2141 / 935 / 3121 m | 1705 / 887 / 3022 m |
| arrival-time MAE | 20.5 s | 16.1 s | 25.5 s | 19.7 s |
| established / corridor after the join | 94.2 % / 12.9 % | 99.0 % / 19.9 % | 92.4 % / 11.9 % | 99.0 % / 19.7 % |
| glidepath (in FAF) flown / the truth | 15.1 % / 1.4 % | 8.2 % / — | 13.5 % / 1.4 % | 9.1 % / — |
| floor (pre-FAF) flown / the truth | 19.7 % / 32.0 % | — | 19.7 % / 32.0 % | — |
| legs flown | 3.61 | 1 | 3.53 | 1 |
| bank capped | 19.1 % | 15.3 % | 18.2 % | 14.1 % |

| straight-in, KRDU val | rolled, L−1 | whole path, L−1 (§12.3, v1 grading) | rolled, 60 s | whole path, 60 s (§12.3, v1 grading) |
|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 288 / 35 / 134 m | 194 / 34 / 134 m | 567 / 42 / 152 m | 350 / 42 / 152 m |
| arrival-time MAE | 6.6 s | 4.8 s | 14.0 s | 8.4 s |
| established / corridor after the join | 99.8 % / 0.7 % | 99.9 % / 0.6 % | 99.3 % / 2.5 % | 99.2 % / 2.9 % |
| glidepath (in FAF) flown / the truth | 2.0 % / 0.3 % | 11.6 % / — | 3.2 % / 0.3 % | 36.8 % / — |
| floor (pre-FAF) flown / the truth | 4.1 % / 5.3 % | — | 6.6 % / 8.5 % | — |
| legs flown | 1.06 | 1 | 1.30 | 1 |
| bank capped | 0.7 % | 0.5 % | 3.0 % | 2.2 % |

**3c — the plan head** (`prediction_output=plan`; `outputs/plan/labels.py` the targets,
`model.py` the head and the loss, `strategy.py` the path; the point heads, no fan yet).
The head regresses 14 numbers per anchor in their own scales (L1 over the entries the
track defines): the operating group (`T`, `V_mid`, `d_decel`, `V_final`, `h_capture`,
`d_join`, and the REMAINING PATH — the schedule's coordinate, which the guidance needs and
a prediction must therefore carry) and the next instruction (the fix ahead / across the
anchor in runway axes, the heading on after it as a unit vector relative to the course,
the speed, the remaining path and the height at it), plus the logit of "no fix ahead". The
labels are read per drawn anchor at batch time (`extract_plan`, 0.4 ms each). The
deployable forecast is the ROLLED flight (`fly_rolling_orders`: the head's order at the
observed anchor, one leg on the guidance, the head asked again on the window of the
observed track continued by the flown rows, until the closing; cut at the threshold; at
most 6 instruction legs and 1.5× the first predicted arrival time); the validation replay
is the DRAWN single-step flight (`draw_order`: the route through the predicted fix at the
predicted schedule on the normalized grid, milliseconds per flight — the closure path's
precedent). `plan_oracle --policy model` rolls a plan checkpoint's orders through the
oracle's own instrument; `plan_next_readout` on a plan checkpoint adds the HEAD columns
(the single-step reading of §7).

The first run: 403 train / 117 val flights (every 17th / 12th of the locked split), 60 s
window, remaining-path-uniform anchors, d_model 128 × 2 layers, 30 epochs, ~1 min. The
validation objective fell 3.71 → 1.56 and was still falling (operating 0.29, T 0.51,
instruction 0.43, no-fix flag 0.32). Rolled through the guidance on the first 48 val
flights at the 60 s anchor, paired with the truth's own instructions rolled under the
same checkpoint (its strata: 27 straight-in / 21 vectored / 23 established at row 29):

| 48 flights, 60 s anchor | ceiling (truth's orders, first rolling) | the first head (first rolling) | ceiling (final rolling) | the head, corrected (final rolling) |
|---|---|---|---|---|
| vectored ADE / chamfer / Fréchet | 1913 / 820 / 2579 m | 6391 / 2613 / 10767 m | 2293 / 783 / 2567 m | 7657 / 2833 / 13015 m |
| vectored established / legs / skipped / turn not completed | 95 % / 3.2 / 0 % / — | 38 % / 5.2 / 67 % / — | 90 % / 3.19 / 4.8 % / 9.5 % | 38 % / 2.14 / 4.8 % / 24 % |
| straight-in ADE / chamfer | 550 / 43 m | 857 / 83 m | 554 / 39 m | 827 / 42 m |
| straight-in corridor / glidepath / established | 3.7 % / 0 % / 100 % | 74 % / 82 % / 93 % | 3.7 % / 0 % / 100 % | 0 % / 0 % / 93 % |
| arrival-time MAE flown / the head's T | 20.4 s / — | 53.6 / 52.5 s | 20.0 s / — | 60.6 / 57.0 s |

**What the first head got wrong is a design fact, not a training one**: its "the join is
next" flag was supervised only off the final (§6 as written: an established flight
supervises the operating parameters alone), so on every established flight (23 of 48 at
60 s) it predicted a fix from the prior and the guidance flew to it — 74 % of straight-in
flights out of the corridor, 6.2 legs each. An aircraft on the final has no next fix
either: the flag is now "no fix ahead" (the join next, OR already on the final),
supervised on every sample, with `d_join` there its own remaining path (the join is
here), and a closing that starts at or inside the join flies the height from where the
aircraft is. The corrected head: straight-in corridor 74 → 0 %, glidepath 82 → 0 %,
legs 6.2 → 1.0. The vectored stratum stays what a 403-flight head can do (the base turn
25–39 km ahead is predicted 2.8 km off at random anchors against the 8.5 km baseline;
11 km off at the 60 s anchor against 10.7 — the head has not learned the far lead; and
its no-fix flag, initialised at even odds, is below the majority):

| single-step readout, 48-flight smoke (the retrained head) | fixed 60 s | random anchors |
|---|---|---|
| HEAD next fix error p50 / p90 | 11.3 / 18.4 km | 2.8 / 9.2 km |
| median baseline p50 / p90 | 10.7 / 34.4 km | 8.5 / 26.6 km |
| HEAD next speed error p50 | 7.4 m/s | 7.7 m/s |
| HEAD T MAE | 57.0 s | 25.0 s |
| HEAD no-fix accuracy / majority | 0.72 / 0.96 | 0.72 / 0.98 |

Full KRDU train (6856 flights, d_model 256 × 3 layers, 67 epochs run, best epoch 52; `step3c_plan_head_full/`): the validation
objective's parts at the best epoch — operating 0.153, T 0.246, instruction 0.192, no-fix flag 0.067.
Rolled through the guidance on the whole val split at the 60 s anchor (`plan_oracle --policy model
--route next --anchor-s 60`, `step3c_rolled_model_a60s/`; 1404 flights), paired with the
truth's own instructions rolled under the same checkpoint (`step3c_rolled_truth_a60s/`; the strata
at this checkpoint's L−1, row 29: 799 straight-in / 601 vectored):

| KRDU val, 60 s anchor | ceiling (truth's orders) | the full head |
|---|---|---|
| vectored ADE / chamfer / Fréchet | 1845 / 797 / 2631 m | 3781 / 1524 / 4715 m |
| vectored established / corridor after the join | 93.7 % / 13.0 % | 83.4 % / 23.8 % |
| vectored legs / skipped / turn not completed | 3.31 / 6.5 % / 5.7 % | 3.60 / 14.3 % / 9.7 % |
| straight-in ADE / chamfer | 584 / 41 m | 635 / 42 m |
| straight-in corridor / glidepath / established | 0.4 % / 1.6 % / 99.2 % | 1.8 % / 2.6 % / 99.2 % |
| arrival-time MAE flown / the head's T (pooled) | 19.3 s / — | 34.6 / 24.4 s |
| glidepath (in FAF) / floor (pre-FAF), pooled, flown | 7.0 % / 11.3 % | 10.5 % / 3.6 % |
| … the truth | 0.9 % / 17.0 % | same |

The single-step reading (`plan_next_readout` on the head, `step3c_next_readout_head/`; the fixed
60 s anchor and 5 random anchors per flight):

| single-step readout, KRDU val | vectored, 60 s | vectored, random | straight-in, 60 s | straight-in, random |
|---|---|---|---|---|
| HEAD next fix error p50 / p90 | 2612 / 9062 m | 1625 / 4409 m | 903 / 2968 m | 827 / 2574 m |
| median baseline p50 / p90 | 6536 / 33725 m | 10686 / 22889 m | 1172 / 3719 m | 1488 / 4209 m |
| HEAD next speed error p50 | 6.2 m/s | 5.7 m/s | 5.3 m/s | 4.3 m/s |
| HEAD next remaining error p50 | 1923 m | 1652 m | 833 m | 758 m |
| HEAD T MAE | 39.5 s | 24.9 s | 11.0 s | 6.3 s |
| HEAD no-fix accuracy / majority | 0.998 / 0.998 | 0.987 / 0.983 | 0.855 / 0.843 | 0.846 / 0.842 |


**Read.** The line runs end to end — labels per drawn anchor, the head, the drawn replay
for checkpoint selection (the controller's own height law, `reference_height`), the
rolled flight through the guidance, the two readings against the same ceiling. The
rolled oracle (3b) sits 333 m of vectored ADE and 55 m of chamfer above the whole-path
waypoints oracle at L−1 on the full split (1492 against 1159; 2141 against 1705 at the 60 s
anchor), with 94.2 % of vectored flights established against 99.0 % — before the last three
rolling rules it sat 1078 m above, with 27 % of the vectored flights over 1 km worse. What
rolling still costs is the corner flown from a lagging pose (turn not completed on
5.8 % of vectored flights, a fix skipped on 8.0 %) — the same residual as §12.3's —
and the vertical: the rolled flight leaves the glidepath window inside the FAF on
15.1 % of vectored flights (the truth 1.4 %), each leg descending to its instruction's
height and the closing re-planning the capture from the last fix.

**The full head's reading** (1404 val flights at the 60 s anchor, paired with the
truth's orders under the same checkpoint): vectored ADE 3781 m against the ceiling's 1845 m, chamfer 1524 against 797 m,
83.4 % established against 93.7 %; straight-in ADE 635 against 584 m, corridor after the join 1.8 % against 0.4 %;
arrival-time MAE 34.6 s flown (24.4 s the head's own T) against the ceiling's 19.3 s and native32's 25.9 s. The single
step: the head's next fix is 2.6 km off at the 60 s anchor and 1.6 km at random anchors on vectored
flights, against the median baseline's 6.5 and 10.7 km (§7's veto does not fire); its arrival time 39.5 s / 24.9 s MAE there.
Next on (c): the fan over the next fix (§3b's distribution), the corner, and the head's
far lead (the base turn 25–39 km ahead at the 60 s anchor).


### 12.5 Step 3(d) — receding-horizon, lockstep rolling (v5.1, 2026-09-11 night)

`dev-plan-lockstep`; `plan_oracle --route next --rolling lockstep` (the default now; `--rolling
leg` is §12.4's form). The unit of rolling is a 30 s step: the policy is asked again every
step from the aircraft's pose and window, the route to the instruction in force is kept
WHOLE and tracked from the point reached on it (`FlightState.progress`, handed to the
guidance), re-laid only when the order changed materially (its fix over 1 km, its heading on
over 10°, the closing's join over 1 km, fix ↔ none) or the aircraft drifted over 1 km from
it; a step that executes its instruction (past the fix, on the heading given within 5°) is
cut there so the next instruction is laid from the turn's end; a flight ends at its threshold
crossing on the final; an aircraft on the final follows the final (a fix behind it counts as
executed, one ahead is not flown); the whole group is stepped together in one guidance
rollout of 10 holds. Eight rules had to be found on the 48-flight smoke and the full run's
worst flights before the oracle came within ~350 m of the leg form: re-laying the route every
step from a mid-turn pose made the turn-straight-turn builder flip its turn direction step
after step (the aircraft drifting 3 km sideways per step, two of three diagnosed flights
never executing their first instruction) — hence the route in force; a closing re-laid from
at or inside the join with the plan's capture height held an established flight level at
its anchor height — the height starts from where the aircraft is; a step's height law
anchored at the current height re-steepened the descent each step — anchored at the height
the route was laid from; an instruction within a kilometre turned into the closing for the
step (`ahead_of`) and was never executed — the route is laid through the fix wherever it is;
"executed" read off the along-track projection alone fired 20 km short of a base-turn fix
(arrivals 200–290 s early) and, once repaired, still fired mid-turn (38° off the heading given:
the next leg laid from mid-turn looped) — executed means the turn is DONE; a fly-by laid from
inside its turn or with a design radius the geometry cannot afford (6 km at 146 m/s, 3.6 km
short of the fix) looped — such a leg is "turn to heading X" with the onward point four radii
ahead (`leg_route(mid_flight=True)`); and a closing near the join but off its direction is the
intercept polyline at 30°, never the aligned-pose turn-straight-turn.

**The oracle** (the truth's own instructions, KRDU val 1404; wall time 69 s at L−1
and 79 s at 60 s against ~90 min for the leg form — 0.05 s per flight):

| vectored, KRDU val | lockstep, L−1 | leg form, L−1 (§12.4) | whole path, L−1 (§12.3) | lockstep, 60 s | leg form, 60 s (§12.4) | whole path, 60 s (§12.3) |
|---|---|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 1847 / 700 / 2466 m | 1492 / 324 / 1800 m | 1159 / 268 / 1648 m | 2417 / 1057 / 3402 m | 2141 / 935 / 3121 m | 1705 / 887 / 3022 m |
| arrival-time MAE | 28.7 s | 20.5 s | 16.1 s | 33.2 s | 25.5 s | 19.7 s |
| established / corridor after the join | 87.9 % / 12.5 % | 94.2 % / 12.9 % | 99.0 % / 19.9 % | 86.7 % / 13.3 % | 92.4 % / 11.9 % | 99.0 % / 19.7 % |
| glidepath (in FAF) / floor (pre-FAF) | 20.1 % / 14.3 % | 15.1 % / 19.7 % | 8.2 % / — | 19.1 % / 16.7 % | 13.5 % / 19.7 % | 9.1 % / — |
| legs / skipped / turn not completed | 3.80 / 6.0 % / 3.0 % | 3.61 / 8.0 % / 5.8 % | 0.00 / 0.0 % / 0.0 % | 3.69 / 6.2 % / 3.4 % | 3.53 / 7.0 % / 6.6 % | 0.00 / 0.0 % / 0.0 % |
| bank capped | 16.0 % | 19.1 % | 15.3 % | 15.5 % | 18.2 % | 14.1 % |

| straight-in, KRDU val | lockstep, L−1 | leg form, L−1 | whole path, L−1 | lockstep, 60 s | leg form, 60 s | whole path, 60 s |
|---|---|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 283 / 36 / 134 m | 288 / 35 / 134 m | 194 / 34 / 134 m | 653 / 43 / 152 m | 567 / 42 / 152 m | 350 / 42 / 152 m |
| arrival-time MAE | 6.6 s | 6.6 s | 4.8 s | 14.8 s | 14.0 s | 8.4 s |
| established / corridor after the join | 99.8 % / 0.3 % | 99.8 % / 0.7 % | 99.9 % / 0.6 % | 98.2 % / 2.9 % | 99.3 % / 2.5 % | 99.2 % / 2.9 % |
| glidepath (in FAF) / floor (pre-FAF) | 1.5 % / 4.1 % | 2.0 % / 4.1 % | 11.6 % / — | 4.1 % / 5.9 % | 3.2 % / 6.6 % | 36.8 % / — |
| legs / skipped / turn not completed | 1.08 / 0.8 % / 0.0 % | 1.06 / 0.8 % / 0.1 % | 0.00 / 0.0 % / 0.0 % | 1.38 / 1.3 % / 0.1 % | 1.30 / 1.5 % / 0.2 % | 0.00 / 0.0 % / 0.0 % |
| bank capped | 0.8 % | 0.7 % | 0.5 % | 2.8 % | 3.0 % | 2.2 % |

**The full head re-asked every 30 s** (the §12.4 checkpoint unchanged; the 60 s anchor,
paired with the truth's instructions rolled in lockstep under the same checkpoint,
799 straight-in / 601 vectored; wall time 92 s and 80 s):

| vectored, 60 s anchor | ceiling, lockstep 30 s | the head, re-asked every 30 s | the head, every 60 s | ceiling, leg form (§12.4) | the head, once per leg (§12.4) |
|---|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 2184 / 888 / 2816 m | 5391 / 2639 / 7750 m | 4991 / 2389 / 7028 m | 1845 / 797 / 2631 m | 3781 / 1524 / 4715 m |
| arrival-time MAE | 30.0 s | 66.4 s | 54.5 s | 22.8 s | 56.9 s |
| established / corridor after the join | 87.7 % / 14.8 % | 54.9 % / 4.7 % | 49.1 % / 29.0 % | 93.7 % / 13.0 % | 83.4 % / 23.8 % |
| glidepath (in FAF) / floor (pre-FAF) | 19.5 % / 14.5 % | 52.1 % / 0.3 % | 43.6 % / 2.3 % | 13.8 % / 17.8 % | 21.0 % / 0.5 % |
| legs / skipped / turn not completed | 3.48 / 5.5 % / 3.0 % | 5.18 / 0.0 % / 25.3 % | 5.61 / 0.0 % / 20.8 % | 3.31 / 6.5 % / 5.7 % | 3.60 / 14.3 % / 9.7 % |
| bank capped | 15.1 % | 29.7 % | 35.5 % | 17.6 % | 27.6 % |

| straight-in, 60 s anchor | ceiling, lockstep 30 s | the head, every 30 s | the head, every 60 s | ceiling, leg form | the head, once per leg |
|---|---|---|---|---|---|
| ADE mean / chamfer p50 / Fréchet p50 | 598 / 42 / 149 m | 887 / 43 / 150 m | 850 / 44 / 151 m | 584 / 41 / 148 m | 635 / 42 / 150 m |
| arrival-time MAE | 14.8 s | 28.3 s | 23.8 s | 14.6 s | 15.8 s |
| established / corridor after the join | 99.0 % / 0.4 % | 99.4 % / 1.3 % | 97.7 % / 4.6 % | 99.2 % / 0.4 % | 99.2 % / 1.8 % |
| glidepath (in FAF) / floor (pre-FAF) | 1.9 % / 6.0 % | 6.3 % / 5.9 % | 5.0 % / 6.1 % | 1.6 % / 6.4 % | 2.6 % / 6.0 % |
| legs / skipped / turn not completed | 1.23 / 1.3 % / 0.0 % | 1.24 / 0.0 % / 0.1 % | 1.23 / 0.0 % / 1.0 % | 1.17 / 1.3 % / 0.0 % | 1.16 / 0.3 % / 0.3 % |
| bank capped | 1.4 % | 3.2 % | 6.3 % | 1.3 % | 2.3 % |

Pooled arrival-time MAE: the head 45.8 s flown (24.4 s its own T at the first
step) against the lockstep ceiling's 22.5 s, the leg form's 34.6 s and native32's 25.9 s.

**Read.** Two things, one expected and one not. The lockstep ORACLE is 78× faster
(69 s for the split against ~90 min) and sits 355 m of vectored ADE above the leg
form at L−1 (1847 against 1492; chamfer 700 against 324 m; established
87.9 % against 94.2 %) — the median flight is the same (paired ΔADE p50 ≈ 50 m), a 13 % tail
is worse by over 1 km: the legs re-laid from where a 30 s step happened to end, not from
the turn's end, and the whole route tracked faithfully where the leg form's cut hid a
looping lay. **Re-asking the HEAD every step made the vectored prediction worse, not
better**: 5391 m every 30 s, 4991 m every 60 s, 3781 m asked once per leg (its
lockstep ceiling 2184); established 55 % / 49 % / 83 %; the straight-in
stratum is unmoved (887 against 635 m, chamfer 43 m). The cause is in the readout's own
numbers: the head was trained on OBSERVED windows and is asked, from the second step on,
on windows of its own flown rows; its orders jitter from step to step (a fix moved over
1 km re-lays the route, `RELAY_FIX_M`), and a spurious fix or a too-long arrival time on
any one of 10–20 asks derails a flight that a single early guess would have landed —
29 % of vectored flights ran to the time cap. The receding-horizon FORM is right (the
oracle proves the guidance can be re-issued every 30 s at a small cost); the HEAD is not
yet a receding-horizon head. Next, in this order: train the head on rolled windows (the
windows a lockstep flight produces — a closed-loop training set from the oracle's
flights, or noise on the observed windows), hold an order unless the head's change is
material for two consecutive asks, and only then the fan over the next fix. §12.4's
leg-form numbers remain the plan head's best prediction today.

### 12.6 Step 3(e) — the head trained on rolled windows (v5.2, 2026-09-11 night)

Planned before building (the §9 step 3(e) text is the plan). Every number is KRDU val (1404
flights), the rolled prediction in lockstep (`plan_oracle --route next --policy model --rolling
lockstep`), the 60 s anchor unless said; the head is step 3c's recipe (iTransformer d=256,
3 layers, 30-row window, remaining-path-uniform anchors, selected on the observed L−1
objective) with `plan_rolled_share` 0.5.

**The table.** The truth policy over the checkpoint's split from L−1 (`plan_rolled_windows
--policy truth --split train --split val`): 6856 + 1405 flights, **97,633 samples** (9 steps
per flight at the median, 35 at most), 53 % of them on the final and 57 % with no fix ahead
(the observed-anchor population has ~50 %); the oracle's own flown ADE along the way 612 m
straight-in / 2388 m vectored (train), 99 of 6856 flights to the time cap; 696 s of CPU for
both splits (57 MB). The head's own states labelled by the truth (`--policy model`, the
step-3c head, 48 val flights): 639 samples, its flown vectored ADE 4919 m, 8 of 48 capped —
round 1's input.

**The draft round (2026-09-11 night, before the review; artifacts `step3e_rolled_truth/`,
`step3e_rolled_head_r0/`, `step3e_lockstep_r0_a60s/`).** Labels with the schedule coordinate
TRACKED as the lockstep tracks it (the anchor's remaining path less the path flown, re-synced
at an executed fix) and `T` the truth's arrival time less the time flown — the review's
reading: right along the oracle's own path, wrong for a slow learner (the coordinate runs out
kilometres from the runway), and at step 0 not the observed label (the truth's curvature
before its first fix). The head trained 97 epochs, 5 s each; its rolled-window val loss
fell WITH the observed objective (0.695 against 0.644 at the kept epoch; no divergence, so
the observed selection stands). Rolled at 60 s:

| KRDU val, 60 s anchor, lockstep | all | straight-in | vectored |
|---|---|---|---|
| §12.5: the head re-asked every 30 s (observed windows only) — ADE mean | — | 887 | 5391 |
| … established | 0.55 | — | 0.49 |
| … rolled flight capped | 0.29 (vectored) | | |
| §12.4: the head asked once per leg — vectored ADE | | | 3781 |
| the lockstep ceiling (truth policy, 60 s) — vectored ADE | | | 2184 |
| **draft round 0 — ADE mean** | 2095 | 694 | 3930 |
| … ADE p50 / chamfer p50 | 1119 / 56 | 548 / 43 | 3087 / 1555 |
| … established | 0.915 | 0.994 | 0.809 |
| … rolled flight capped / turn not completed | 0.038 / 0.033 | 0.001 / 0.001 | 0.088 / 0.077 |
| … ETA MAE (the head's T) s / flown dt MAE s | 24.4 / 36.4 | 11.3 / 17.5 | 39.1 / 58.8 |

**Read.** Training on the oracle's own windows removes the derailing: capped 29 % → 3.8 %,
established 55 % → 91.5 % (vectored 49 % → 81 %), vectored ADE 5391 → 3930 m — now beside
the once-per-leg 3781 with none of its commitment, straight-in unchanged (694 against 635 at
its best, chamfer 43). What is left is the head's geometry itself (vectored chamfer 1555 m
against the ceiling's ~700): the fixes it names, not the derailing. The final rounds below
use the review's labels (`TruthExpert`: the truth's values at the nearest truth row of the
leg in force plus the way there — exact at the anchor, defined at any learner state).

**The rounds (2026-09-11 night, the review's labels; artifacts `step3e_r0_table/`,
`step3e_r0_head*/`, `step3e_r1_table/`, `step3e_r1_head/`, `step3e_*_lockstep_*/`).** Every
head trained 70–103 epochs (kept 55–88), 5 s per epoch; the rolled-window val loss is the
plan loss over the val split's 16,470 truth-policy windows (equal weight per sample; each
component over its carriers).

| KRDU val, 60 s anchor, lockstep | vectored ADE mean / p50 | vectored chamfer p50 | established vec. / all | capped vec. | straight-in ADE / chamfer | dt MAE all | rolled val loss (obs. objective) |
|---|---|---|---|---|---|---|---|
| §12.5: the head re-asked, observed windows only | 5391 / — | — | 0.49 / 0.55 | 0.29 | 887 / 43 | — | — |
| §12.4: asked once per leg | 3781 | — | 0.78 (vec.) | — | 635 | 34.6 | — |
| the lockstep ceiling (truth policy) | 2184 | — | — | — | — | 22.5 | — |
| round 0, share 0.25 | 3833 / 3263 | 1881 | 0.839 / 0.930 | 0.085 | 761 / 44 | 38.6 | 0.736 (0.649) |
| round 0, share 0.5 | 3862 / 3159 | 1467 | 0.779 / 0.901 | 0.116 | 756 / 44 | 38.3 | 0.630 (0.643) |
| **round 0, share 0.75** | **3641 / 2904** | 1434 | 0.865 / 0.940 | 0.090 | 747 / 44 | 37.1 | 0.559 (0.651) |
| round 0, share 1.0 | 3685 / 2862 | 1427 | 0.862 / 0.939 | 0.068 | 735 / 44 | 37.2 | 0.523 (0.646) |
| round 1 (DAgger: round 0's own states appended), share 0.5 | 3905 / 3376 | 1576 | 0.762 / 0.895 | 0.453 | 682 / 43 | 36.2 | 0.847 (0.654)¹ |

¹ over its own table's 32,057 val samples (the head's states are harder), not comparable
with the rows above.

Paired per flight over the 601 vectored flights, against §12.5's re-asked head: round 0 at
share 0.5 −1694 m of ADE at the median, better on 78 %; share 0.75 −1867 m, 83 %; share
1.0 −1962 m, 81 %. Against round 0 at share 0.5: share 0.75 −71 m (better on 55 %), share
1.0 −130 m (59 %), share 0.25 +241 m (38 %), round 1 +73 m (45 %). At L−1 the same
picture: share 0.5 3959 m / established 78.4 %, share 0.75 3719 m / 84.2 %.

**Read.** (1) Training on the rolled windows is what the receding-horizon head needed: the
derailing is gone (capped 29 % → 7–12 %; vectored established 49 % → 78–87 %, pooled 55 %
→ 90–94 %), and the vectored ADE 5391 → 3641 m, now BELOW the once-per-leg head's 3781 with
none of its four-minute commitment, straight-in at 735–761 m against its 635 (chamfer 44,
the ceiling's 43). (2) The share is the lever and more is better: the rolled-window val
loss falls monotonically with it (0.736 → 0.630 → 0.559 → 0.523) while the observed L−1
objective does not move (0.643–0.651) — the two populations are learnt side by side, and the
observed selection stays valid — and 0.75 and 1.0 are tied on the deployed reading
(−71 / −130 m paired, 55 / 59 %), both ahead of 0.5. **Share 0.75 is the candidate default**
(the observed anchors stay in the draw for the single-step readouts at any anchor); 1.0 is
its equal here. (3) DAgger's first round did NOT help: the head's own states appended to
the truth's (190k samples, 45 % of them with no fix ahead against the truth's 57 %) gave a
head that names a fix more often — 255 of 601 vectored flights reached the six-instruction
cap (`legs`) against 8 — and the same ADE (+73 m paired). The truth-policy windows are
already the right distribution for a head that then flies close to the truth; a second
round would want the cap in the labels or the head's spurious fixes suppressed first. (4)
What is left is the head's GEOMETRY: vectored chamfer ~1430 m against the ceiling's ~700,
ADE p50 2.9 km against 2.2 — the fixes it names, at the 60 s anchor, 25–39 km from the turn.
The 3(d) gate: vectored ADE 3641 m is still above native32's 2870 (not passed); straight-in
chamfer 44 m against 109 (passed); established 94 % against 70 %. Next: hold an order unless
the head's change persists two asks (the jitter is smaller now but `RELAY_FIX_M` re-lays
still fire), then the fan over the next fix (§9 step 3(c)'s distribution), then the seed.
