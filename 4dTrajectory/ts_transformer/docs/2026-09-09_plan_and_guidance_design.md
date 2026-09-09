# Plan-and-guidance: the next model (design, 2026-09-09)

Status: design only. Nothing here is built. Decision to write this document: the user, 2026-09-09,
after the review of the 09-03 → 09-09 programme (`docs/reports/2026-09-08_programme_results_and_plan.md`).

## 1. Why change the architecture

Today's model is a transformer that outputs 32 control segments (thrust, bank, load factor) and a
duration, and a point-mass rollout that flies those controls. The network flies the aircraft; a stack
of inference-time hooks (barrier, speed floor, trombone) corrects it afterwards. Six days of
experiments say this is the wrong division of labour:

- The network draws the path well. Straight-in chamfer is 65–109 m and the cross-track error is
  ~100 m. What it cannot do is decide *when*: when to turn base, when to slow down, how long to
  fly the downwind. That decision is the controller's, and every experiment that supplied it from
  outside moved the error by hundreds of metres (true arrival time: vectored ADE 2643 → 1596 m;
  true join point: 2858 → 2356 m). Nothing learned from the aircraft's own history moved it.
- The network does not know the constraints. Training-time corridor penalties were rejected twice
  (with and without the teacher). All feasibility comes from hooks added after the fact, and each
  hook fixed one thing and broke another (the speed floor removed stall and created thrust-over-max;
  the trombone removed both and over-stretched by 10 km because it measured the wrong distance).
  When the hooks are stacked, the flyable share goes from 8 % to 78 % — the hooks are already doing
  the work.
- The loss is a hand-balanced mixture of seven terms (position, velocity, imitation teacher,
  heading rate, bank smoothness, duration, and the unit scale of position). The balance is fragile:
  changing the position unit from 10 km to 1 km made the latent work and destroyed bank fidelity
  (skill 0.726 → 0.400); restoring the teacher's share repaired the bank shape and destroyed the
  latent. Every improvement in one term has cost another.
- Seed noise on pooled ADE is ~125 m. Half of the arms this week moved less than that.

The pattern is clear: we are tuning a model that has to rediscover feasibility from data, and
patching it where it fails. The alternative is to give the model only the part that needs learning.

Relation to the current control-prediction model (decision (A), 2026-09-09): the same point-mass
dynamics, bounded controls, rollout, actuator model, CTA conditioning, random anchors, quantile
head and calibration are all kept. What changes is who chooses the controls: the guidance layer
emits them from the plan, so the controls become an inspectable intermediate rather than the
learning target. The thesis rule that intent is never an output is kept by construction (§3b).

## 2. The idea in one paragraph

Split the problem in two. A learned **plan predictor** reads the aircraft's history (and the
scheduler's inputs) and outputs the **operating parameters** of the plan — when it arrives, how fast it flies, where and to
what speed it decelerates, how high it captures the glidepath — and a distribution over the
**route** (where it joins the final and how much path it flies before that), which is the
controller's intent and is either assigned by the scheduler or kept as a distribution, never a
point output. A deterministic **guidance layer** turns a plan into a trajectory that is feasible by
construction: it stays in the LPV corridor, respects the stall margin and the thrust limit, absorbs
extra time by lengthening the path before the final (never by slowing below the floor), and meets
the assigned arrival time exactly. The network never touches bank or thrust directly. The
scheduler is part of the loop: it assigns the arrival time (and, if it wants, the join point), and
gets back a flyable reference and a calibrated arrival-time distribution.

## 3. What the network predicts, and what it does not

A plan has two kinds of numbers, and the thesis rule "intent is never an output" decides where each
kind lives. **Operating parameters** describe how the aircraft is flown; the network predicts them.
**Route parameters** describe where the controller sends the aircraft; they are the controller's
intent, so the network never commits to them as a point output: they are either assigned by the
scheduler or represented only as a distribution (the latent fan or quantiles). The point reference
the scheduler receives is always "the reference for the route the scheduler decided".

Every parameter has a definition, a range, and an extractor that reads it off an observed track;
the extractor is the supervision. All extractors exist in the package or in this week's readouts.

### 3a. Operating parameters — predicted by the network (point + distribution)

| parameter | meaning | range / units | extractor (supervision) |
|---|---|---|---|
| `T` | time from the anchor to the threshold | seconds, > 0 | the track's duration (as today) |
| `V_mid` | the speed held on the segment before deceleration | m/s | median ground speed over 20–10 km remaining |
| `d_decel` | remaining distance where the speed first drops below `V_target + 10 m/s` | metres | `run_ts_straight_in_residual_readout.decel_distance_km` |
| `V_final` | the final approach speed | m/s | the threshold-crossing speed (the observed speed gate's quantity) |
| `h_capture` | the height at which the glidepath is captured | metres | the track's altitude when the on-final gate opens |

Five numbers, all aircraft operating parameters in the thesis sense: how fast, where to slow, how
high, how long. The network predicts each as a point and as a distribution (quantiles, the B-line
head; or the latent fan of L2.g). `T` keeps its calibrated interval (B2).

### 3b. Route parameters — assigned by the scheduler, or a distribution; never a point output

| parameter | meaning | range / units | extractor (for the distribution's supervision and for the oracle test) |
|---|---|---|---|
| `d_join` | remaining path at which the flight becomes established on the final | metres | `final_approach_geometry.truth_final_gate` (the on-final gate every readout uses) |
| `side` | which side the base leg comes from | {left, right} | the sign of the cross-track offset at the gate opening |
| `L_pre` | path length flown before the join | metres | arc length of the track up to `d_join` |

Three numbers. In delivery they come from the scheduler (it decides the sequence, so it decides
the join). When the scheduler has not decided, the model offers a **distribution** over them — the
latent sampler (whose nearest-of-6 sample beats the point prediction on 92 % of flights) or
quantiles — and the guidance layer flies each sample into a member of a fan. There is no committed
point estimate of the route: that is the rule, and it is also what the data support (the
programme found no way to predict the join from the aircraft's own history beyond a distribution).

Together, 5 + 3 numbers per flight replace the 32 × 3 control segments. The control-basis oracle
of 2026-09-07 showed that a 32-segment basis reproduces the truth to ~100–200 m, so a lower
parametrisation is not a loss as long as the guidance layer can fly it (§9 step 2 measures that
ceiling before anything is trained).

## 4. What the guidance layer does

A fixed, deterministic controller that flies a plan. It is assembled from parts that exist:

1. **Route builder** — from the anchor state and the plan: fly the current heading to a turn point,
   a base leg, and a final of length `d_join`, with the pre-final path length equal to `L_pre`;
   extra time (an assigned arrival later than the plan's own `T`) becomes extra pre-final path
   (this is the trombone, done at planning time with the right distance).
2. **Lateral tracking** — the nominal-law hook (line-of-sight to the route, bounded bank,
   coordinated load factor, actuator-lag compensation) keeps the aircraft on the route; inside the
   final the barrier keeps it inside the LPV corridor by construction.
3. **Speed schedule** — hold `V_mid`, decelerate at `d_decel` to `V_final`, never below the stall
   margin (the speed floor's `V_floor`), never above the thrust limit; if the plan's speeds are
   infeasible the layer clamps them and reports the clamp.
4. **Vertical** — the glidepath law of the nominal hook: capture the glidepath at `h_profile`, then
   follow it.
5. **Time closure** — the route length and the speed schedule together fix the arrival time; if the
   assigned time cannot be met within the envelope, the layer reports the unabsorbable part
   (the deliverable X of this week) instead of flying something infeasible.

The point-mass rollout stays as the simulator. The hooks stop being corrections and become the
controller. Because every step is deterministic and bounded, the trajectory is flyable by
construction: no stall, no thrust over max, no corridor excursion, bank within limits. Those become
things to verify, not things to hope for.

## 5. What the scheduler supplies and gets back

- Supplies: the assigned arrival time (CTA) and the route (`d_join`, `side`, and the pre-final length
  it wants, or a delay to absorb) — decisions the scheduler makes anyway. If it assigns nothing, the
  model returns the fan over routes and the arrival-time distribution, not a single guess. The counterfactual scan showed the
  model obeys an assigned time exactly and responds gradually to it; the plan-and-guidance layer
  keeps that property and adds feasibility.
- Gets back: a flyable reference trajectory for the assigned time and route, the operating
  parameters behind it (when it will slow down, how fast, how high), and — when nothing is assigned —
  the arrival-time distribution and the fan over routes (the B-line interval: ~30 s wide for straight-in traffic,
  ±70 s for vectored traffic until the join is decided).
- The multi-aircraft demonstration in the plan (assign times to several arrivals, build references,
  check separation) becomes straightforward, because every reference is a plan the scheduler can
  inspect and adjust.

## 6. Training

- **Supervision**: the operating parameters extracted from each observed track — direct regression
  and quantiles on five numbers in their own units; the route parameters supervise only the
  distribution head (latent or quantile), never a point head. No inverse-dynamics teacher, no imitation
  weight, no position-unit balancing. Bank fidelity is not a training target any more; it is a
  property of the guidance layer.
- **Through the guidance layer or not**: not, at first. This week's evidence (L1.c, the hook
  training arms of 09-06) says training through a corrective layer teaches the network to lean on
  it. The plan is supervised directly; the guidance layer is applied at inference. A later experiment
  can test end-to-end fine-tuning through a differentiable guidance layer, with a gate.
- **Inputs**: the same history window and the same chart (threshold-anchored ENU, which the frame
  ablation showed is the model's runway knowledge), plus the scheduler's inputs as conditioning
  (CTA conditioning exists; join-point conditioning is the same mechanism).
- **Anchors**: the remaining-path-uniform random-anchor sampler with the scheduler fix (A0.b), so
  the plan can be re-issued at any point of the approach; the two-model rule (fixed anchor at L−1)
  stays available.
- **Data**: pooled over the five airports (42,650 arrivals) rather than KRDU alone (9,720 in the
  fleet subset). The pipeline supports pooled training; the threshold-anchored chart makes airports
  comparable; more data is the only lever on the 125 m seed line.

## 7. Gates — measured against today's best results

The first prototype is judged on KRDU val (1404 flights), paired against the best numbers of the
current programme, and must replicate at two seeds:

| question | today's best | gate for the prototype |
|---|---|---|
| path, no assigned time (vectored ADE) | 2579–2870 m | not worse than 2870 + 125 m; straight-in chamfer ≤ 110 m |
| path, assigned time (vectored ADE with true CTA) | 1596 m | ≤ 1596 m |
| path, assigned time and join point (vectored ADE) | 2356 m (join only, Phase 0) | ≤ 1400 m — the first number that would show the plan works |
| arrival-time MAE, straight-in / pooled | 10.3 / 23.9 s | within 1 s |
| flyability, true time (fully flyable, truncated at threshold) | 78 % (L3.e stack) | ≥ 95 % |
| flyability, +60 s | 52 % | ≥ 90 %; unabsorbable delay X reported per flight |
| corridor violation on the final | 16 % (L3.e) | 0 % by construction |
| bank skill | 0.726 (native32) | not a target; report bank RMS ≤ 0.5° on straight-in |

Veto: if the plan predictor's own parameters are worse than trivial baselines (e.g. `d_join`
predicted worse than the airport median), the learned part is not learning and the design fails.

## 8. Reuse, new work, risks

- **Reused**: the point-mass rollout and its actuator model; the nominal-law, barrier, speed-floor
  and trombone hooks (as guidance parts); the on-final gate and corridor geometry; the CTA
  conditioning; the quantile head and split-conformal calibration; the random-anchor sampler and
  the anytime replay; all readouts and the publisher.
- **New**: the plan-parameter extractors as a dataset layer (most exist as readout functions), the
  plan head, the route builder, the assembly of the hooks into one controller with a time-closure
  step, and the pooled-airport training run.
- **Risks**: (1) the plan parametrisation may be too coarse for vectored flights with more than one
  turn — mitigation: allow one or two extra waypoints as optional parameters, gated by a readout of
  how many truths need them; (2) the guidance layer's route may be flyable but far from what
  pilots actually fly, so a "flyable" reference could still be a poor prediction — the chamfer and
  ADE gates catch this; (3) join-point prediction may be as hard as the intent itself — but it is
  also the thing the scheduler can assign, which is why the design keeps it as an input.

## 9. Steps

1. Extractors: compute the seven plan parameters for every KRDU arrival; report their
   distributions and how well trivial baselines predict them (one day, CPU).
2. Guidance layer without learning: fly the *true* plans through the guidance layer and measure
   against the observed tracks — the ceiling of the design (like the control-basis oracle). If the
   ceiling is worse than today's model, stop here.
3. Plan head on KRDU: point and quantile versions, single seed, all gates read.
4. Assigned-time and assigned-join conditioning; the +60 s delay test with X reported.
5. Two seeds; then pooled five-airport training; then KSJC replication.
6. The multi-aircraft scheduler demonstration.

## 10. What stops

No more arms on: the latent dose axis, teacher replacement, position units, hook doses, the
duration weight. L3.f (the trombone's surplus estimate) runs because it is the route builder's
specification test; nothing else from the current line is queued.
