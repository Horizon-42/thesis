# AeroViz-4D Development Changelog

Dated log of significant changes, root causes, and decisions, referenced from `CLAUDE.md`. This file is deliberately NOT loaded into every session — read it when investigating history: why a design is the way it is, when/why a default changed, what a past bug or postmortem looked like, or which outputs a change made stale. Append new entries at the top (`### YYYY-MM-DD — title`); when a change produces a durable fact (gotcha, default, contract), also update the corresponding section in `CLAUDE.md`.

Entries verified via full test suites + tsc + vite build at the time; "verified in-browser" noted only where done. Merged same-day, same-topic entries.

### 2026-09-06 — Snapshot before a redesign: the tracker review, the P2 data plane as WIP

The opus review of the closure tracker (`control/constraints/closure_tracking.py`)
found a BLOCKER — the nearest-node search over the whole reference lets a via-Dubins
path that passes within metres of itself snap the tracker to the wrong leg (8 of 1404
val flights, endpoints 6–19 km off; four of the five largest ADE "improvements" from
tracking in `2026-09-06_closure_p1d_tracking_results.zh.md` are those flights, the
honest pooled cost being +10.5 m) — and five SHOULD-FIXes (the vertical law's sign is
untested; the stall floor ignores the commanded load factor; ISA density uses chart
height; the height-profile anchor pinning of `4ecfb69` is outside the label contract;
the gain docstring). Nothing was changed: the user decided to redo the plan. The P2 data
plane built meanwhile is committed as WIP (`045c233`: `trajectory_data_process/scene_index.py`,
`flight_scenarios/scene_context.py` with a stricter leakage line — an airborne
neighbour's landing time AND eventual runway are future, kept in `future_label` only —
and `4dTrajectory/ts_transformer/scene/features.py`; 13 tests, no review, no
explainability measurement). The design doc's §〇.1 carries the full snapshot.

### 2026-09-06 — Closure decoder, P1.d: the drawn reference flown by the point-mass rollout — ≤ 100 m of ADE for 92 % flyability

Option (b) of the design's P1.d. `control/constraints/closure_tracking.py` (new): a
command hook that steers the rollout's own state toward the closure reference at every
segment start — L1 lateral guidance on the cross-track / heading error to the reference's
local course plus its curvature as a feed-forward bank, the glidepath law on the height
error with the reference's local slope (capped 0.3), and a PI speed hold toward the
reference speed (integrator from the anchor's implied thrust, proportional 0.3, the
reference acceleration ±0.5 m/s² as feed-forward) with the along-track error folded into
the target and a 1.15 × stall floor. `forecast.track_closure_forecasts` flies every
decision vector under it on the first-order-lag dynamics (`tracking_config` derives that
control config; the only backends that run hooks) for the reference's own duration in
64 held segments; `predict --closure-track`; records carry the controls flown and
`source.closureTracked`. Also fixed: the drawn height profile now starts at the anchor's
height (the first knot is a least-squares value, and the step it drew made the vertical
law ask for a −86° path angle). KRDU val, 1404 flights, pooled / vectored ADE:
C_pred 996 → 1005 / 2197 → 2222, C_truth_intent 595 → 611 / 1235 → 1244, C_oracle 183 →
256 / 458 → 554 m; fully flyable (clean polar) 22 → 92 / 90 / 88 % (observed 98 %, the
control baseline 0.1 %), per sample 99.9 %. The delivery form is closure + tracking
(`docs/2026-09-06_closure_p1d_tracking_results.zh.md`); the tracker's gains stay module
constants for now (a P1.d measurement setting, not yet a config field).

### 2026-09-06 — Closure decoder, P1.c-3: the campaign — both gates pass, the output side was the bottleneck

`4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905/` (arms
`docs/experiments/closure_p1c_arms.json` + `closure_p1c_oracle_arms.json`; report
`4dTrajectory/ts_transformer/docs/2026-09-06_closure_p1c_results.zh.md`). KRDU val, 1404
flights, closed truth, vectored stratum ADE / chamfer p50: simple-v3 2858 / 942; the Phase 0
truth-intent control arm 2011 / 847; **C_pred (closure, ego history only) 2197 / 722;
C_truth_intent (closure + truth (d_join, T) inputs) 1235 / 492; C_oracle (drawn from the
labels) 455 / 180**. Pooled ADE 1333 → 996 (C_pred) → 595 (C_truth_intent); straight-in
469 → 310. Gates (vectored ADE < 1.5 km and chamfer < 500 m with the truth intent; not
worse than the baseline without it) both pass. Training is 2–2.5 s per epoch (a pure
regression, no rollout). One decoder rule was added mid-campaign: a predicted via inside one
turn radius of the anchor is dropped for the plain CSC (`csc-via-at-anchor`) — the first
C_pred arm's straight-in stratum was 6260 m because a label via AT the anchor, predicted a
few tens of metres off, made the Dubins path a full circle (827 of 837 over-long paths);
the pre-rule readout is kept under `attempt1_readouts/`. The runner now substitutes
`{airport}` in a predict-only arm's `predict_args`. Flyability, read as a delta: closure
paths are 22 % fully flyable under the clean polar against the control baseline's 0.1 %
(59k stall samples) and the observed 98 %; per sample 99.8 % = the observed, the few
violations per flight being bank jumps at the CSC junctions and thrust jumps at the knots
(P1.d: a clothoid / bank-rate transition plus profile smoothing, or a post-hoc rollout). P2's acceptance becomes the 960 m of vectored
ADE between C_pred and C_truth_intent.

### 2026-09-05 — Closure decoder, P1.c-1/2: the `closure` prediction output and its labels

The third `prediction_output`. `closure_output.py` (new): `ClosurePrediction` (a 14-number
decision vector — join distance, via pose in runway axes as (d, xt, cos Δψ, sin Δψ), K=4
slowness knots whose integral is the duration, K=4 height knots with the threshold pinned
to 0), `ClosureOutputModel` (the backbone's tokens → MLP → the bounded vector: sigmoid /
tanh at the family's own reach, a softened unit vector for the heading), `fit_labels` (the
per-flight label from the truth: the canonical F3 geometry, both profile widths with
residuals, `valid` = canonical and within the 1 km cap, the difficulty covariates),
`closure_loss_components` (L1 per group over the valid flights; state = geometry,
final_time = slowness in seconds at a 60 s scale — measured at init the three groups then
sit within a factor of two — kinematic = height, terminal = 0), `reconstruct` /
`replay_batch` (the drawn trajectory: via-Dubins, else the plain CSC, else the straight
line, each recorded as `source.closureConstruction`). Wired through `config.py`
(`PREDICTION_CLOSURE`, the six `closure_*` fields, the locks: ENU chart, normalized horizon,
objective checkpoint selection, no random anchors, knot widths ∈ (4, 8)), `train.py`
(`closure-v1-slowness4-height4` target contract, loss registry, a replay branch that keeps
the reconstruction's exact velocities; `LossComponents` moved to `batch_contract` so the
loss side never imports the loop), `models.py`, `dataset.py` (the labels as the batch
context — a file covering no flight is refused, the covered share is printed),
`batching.py`, `forecast.py` (`closure_forecast`, `forecast_closure_from_labels` = the
oracle arm, `source.closureFromLabels`), `run_naming.py` (`closed-form`, `closure-v1`, the
labels file name in the run name), `anchor_eligibility.py`, `__main__.py`
(`--closure-labels`, `predict --closure-from-labels`), `export.py`; scipy joins
`requirements.txt`. `docs/p1_closure_oracle.py labels` writes a cohort's labels
(`closure-labels-v2`; KRDU: 9720 flights, 96.8 % valid). `tests/test_closure_output.py`
(6 tests) runs a whole train → checkpoint → forecast → export → evaluate chain. Review
(opus): the wiring script had applied every hunk 2–3× (the labels file was parsed twice
per dataset) — restored and re-applied once; the other findings (oracle provenance,
zero-coverage training to a perfect loss, unbounded head, 20× loss-scale gap, schema
bump, run-name identity) are the changes above. Arms for the campaign:
`docs/experiments/closure_p1c_arms.json` (C_pred, C_truth_intent, C_oracle).

### 2026-09-05 — Closure decoder, P1.a / P1.b: the path family and the profile parametrisation, measured before any network

Direction C of the scene design (fix the output side before the scene encoder) was
chosen. `4dTrajectory/ts_transformer/closure_geometry.py` (new) holds the closed-form
approach families the closure decoder will draw from — the Phase 0 rule template
(moved out of `docs/phase0_intent_diagnostics.py`, which now imports it), a
downwind-then-Dubins family and a via-pose Dubins family — with the vertical profile,
the truth / naive timings and per-flight fits; `closure_profile.py` (new) parametrises
speed as piecewise-linear SLOWNESS over progress (time is linear in the knots, the fit a
bounded linear least squares) and height as knots; `docs/p1_closure_oracle.py
geometry|speed` runs the two oracles on the reference arm's flights (post-anchor
supervision truth). Two opus review rounds shaped the fits: the objective is the
order-preserving arc-aligned horizontal error (chamfer let detours through), F3 is
multi-start and seeded from F1/F2 (the fitted residuals nest), and the labels are
canonicalised — the join at the localizer entry (a straight along the localizer made
d_join unidentifiable), the via as the earliest pose along the fitted path that still
reproduces it (identifiable: two best starts within 3 m median) — with `canonical:
False` kept on the 4 % of looping fits. Results at KRDU (497 vectored flights, truth
timing): F0 1688 m → F1 1301 → F2 794 → **F3 chamfer 180 / Fréchet 1179 / ADE 510 m,
gate passed**; F2 does not contain the trombone (its downwind runs along the anchor
heading). On the truth path: naive 1308 (Phase 0 reproduced) → naive × truth duration
622 → slowness knots K=2 166 / K=4 97 m; height knots K=4 30 m; combined K=4 110 m.
P1.c's decision vector is therefore ≈14 numbers (d_join, the via in runway axes with
its heading relative to the course, K=4 slowness with the duration as their integral,
K=4 height with the terminal knot pinned to the threshold — a third review round found
the unpinned fit landing up to 33 m below it on 28 % of flights, and the chart-frame via
splitting one decision into four per-runway label distributions). Design doc §〇 / §五
P1 updated with the plan, results and gates.

### 2026-09-05 — Geometric readout (P0 of the scene design): chamfer / Fréchet / arc-aligned ADE next to ADE/FDE

`4dTrajectory/ts_transformer/geometric_metrics.py` (new; `tests/test_geometric_metrics.py`,
23 tests) is the single source for the time-free path metrics, and both standard readouts
(`docs/compare_constraint_arms.py`, `docs/compare_frame_arms.py`) now print them in every
stratum table next to ADE/FDE: chamfer p50 (symmetric nearest point, horizontal, 100 m
resampling — moved out of `phase0_intent_diagnostics.py`, whose chamfer now comes from
here), discrete Fréchet p50 (order-preserving, anti-diagonal DP on the same resampling,
verified against the textbook recursion), arc-aligned ADE (3D at the same fraction of each
path's own horizontal arc length, 64 fractions excluding the anchor), the predicted/true
length ratio, |Δdur| p50 and the along-path lag p50 (predicted minus true time at the same
arc fraction; Δdur on the exported states' clock, so a horizon-capped flight reads the cap
here and the duration head in `final_time_error_s`). Truth (`--geometry-truth`): `closed`
(default) = the post-anchor observed rows closed to the threshold at `true_final_time_s`
— the exported states carry no fitted tail, and the observed track stops a median 379 m /
6 s short at KRDU (82 m / 2 s at KSJC), so both metric families now end at the same point
and time; `observed` = the rows as exported, which reproduces the Phase 0 diagnostics'
chamfer within 2 m (the resampler now includes the endpoint: 942 / 801 / 791 / 850 →
943 / 802 / 793 / 849). The readout's first line states the truth, the closure and every
parameter. **Review finding (opus, 2026-09-05): the STATE output's exported polyline is a
node-scale saw-tooth** (heading reversals > 90° at a median 50 % of its 2 s nodes, every
flight above 5 %; control rollouts and the observed truth at 0, truth max 0.008; raw length
ratio ≈ 2 against 1.01), which doubles anything parametrised by its own arc. Each flight
therefore carries `reversal_share` and enters the arc family only at ≤ 5 %; a block
aggregates arc-ADE / lag over those flights, writes `arc_family_share`, prints the share
beside the value when it is below 100 % and `n/a` below 95 % (null in the JSON, so a two-flight mean is never quotable) — no smoothing (a silent
approximation), and no length-ratio gate (the second review showed a block whose median
ratio sits in a band still averages in its out-of-band flights, +398 % on one state
stratum, and on control arms a ratio far from 1 is a real error, not an artifact). On
state campaigns the arc family is therefore mostly `n/a`; chamfer / Fréchet / Δdur carry
the geometry there. Also fixed while there: the constraint readout's `|xt| p95` header
broke its markdown table since the campaign it was written for.
Backfilled: `scene_phase0_20260905/readout_geometry.*` (four arms; results doc §三b) and
`control_hooks_v2_20260906/readout_geometry.*` at KRDU + KSJC (hook results doc, new
section). What the geometry says: the truth join point is worth 15–20 % of vectored geometry
(chamfer 942 → 795, Fréchet 3100 → 2468, arc-ADE 2175 → 1775), the truth-duration arm adds
none of it (its 2356 → 2011 ADE gain is timing alone — the case the revised success
criteria veto), Fréchet ≈ 3× chamfer (the paths are sequenced differently, not offset), and
the v2 soft barrier is the only hook that improves vectored geometry too (KRDU chamfer
942 → 886 on 81 % of flights; both trained-through arms lose 13–16 % on Fréchet / arc-ADE).
Design doc §〇 / §五 P0 updated; `CLAUDE.md` gained the "read both families" rule.

### 2026-09-05 — Phase 0 result (KRDU): the join point is worth −17 % vectored ADE, the gate was mis-sized

`4dTrajectory/ts_transformer/docs/2026-09-05_scene_phase0_results.zh.md`; campaign
`scene_phase0_20260905` (O_join_lead, O_join; simple-v3, paired with
`control_procedure_20260905/A_control_v3`). Vectored ADE 2858 → 2364 / 2356 m, duration
error 39 → 20 / 22 s, straight-in 469 → 442 / 458 m; the lead ETA adds nothing beyond the
join point. The pre-registered gate (< 1.5 km) was unreachable for a join-only oracle under
the time-aligned metric: on the same 497 flights the TRUTH path with a naive speed profile
scores 1.3 km and a trombone from the truth join point with the truth's timing 1.7 km
(`docs/phase0_intent_diagnostics.py`, five readings). The network reacts to the channel
(±5 km join → ∓43/+53 s), mostly in timing; its predicted join distance moves 6 → 11 km
toward the truth 14 km but the paths claim the final on fewer flights. Causal traffic counts
from the manifests explain little of the join (R² 0.34 → 0.38 on flights joining after the
anchor; the truth lead ETA 0.47); knowing the join halves the remaining-duration error
(35.8 → 22.8 s). Conclusion: for vectored arrivals the 4D error is dominated by along-path
timing and residual geometry, the join distance is one variable of ~three; the design's
decision variable should include time. New mode `truth-join-duration` (join + the true
remaining time, the duration head's own target) added to size the (where, when) ceiling
with this decoder — its duration error is an identity check. **Its arm: vectored ADE
2011 m (−30 %), pooled 1005 m, duration error 5 s, but the time-free path error does NOT
improve (chamfer 850 m vs 791 join-only, 942 baseline) and the claimed join distance falls
back to 7.4 km — with both truth decision variables this decoder still hedges the geometry,
while a crude trombone with the truth timing scores 1.7 km and the truth path with a naive
speed profile 1.3 km.** Recommendation in the results doc: fix the output side first
(geometric closure: predict (d_join, T) + a speed profile, construct the path), then the
scene encoder; the deterministic top-1 of the scene design is capped near 2 km by this
measurement. Reviewer caveat folded in: a
prediction's "establishes on the final" must be judged with the membership gate
(`hard_on_final` + `stays_mask`), not the k=0.5 truth gate, which the documented 250–350 m
endpoint translation saturates.

### 2026-09-05 — Phase 0 of the scene design: truth-intent conditioning (upper-bound instrument)

`4dTrajectory/ts_transformer/intent_conditioning.py` + `TSConfig.intent_conditioning` ∈
`none | truth-join | truth-join-lead`. The TRUTH join point (chart position of the first
observed row from which the track stays inside the k=0.5 cone, `truth_final_gate` on all
observed rows incl. the lookback) and, with `-lead`, the previous same-runway landing's
TRUE time relative to the window's anchor (`(t_lead − t_anchor)/600 s`, clipped ±1800 s;
lead from the tracks roster's `assigned` landings, model-ready or not) are appended after the
target conditioning as input-only constant channels through the covariate-token path. Reads
the future by design — a development measurement of what inferring the intent is worth,
never a deployable model; the run name carries `intent=truth-…`. Contracts: the row is built
at the window's actual anchor (`series_conditioning(..., anchor=)`), random train anchors are
refused with the lead channel, iTransformer only, `runway-aligned` refused (the gate reads
chart e/n against the world course). `FlightSeries.lead_landing: LeadLanding | None` — `None`
means the roster was never consulted (the channel raises), `LeadLanding(None)` means no
earlier landing (reads as the negative clip); `load_flight_dicts` attaches it for every
flight from the same manifest the flight came through. Data facts (KRDU/KSJC manifests):
`entry_time_utc` is within 1 s of the first slice sample; median lead ETA at the anchor
−27 s, 44 % of leads still airborne, 95 % within ±1800 s; a 150-flight KRDU sample shows
d_join p5/p50/p95 = 10.6/18.7/23.7 km with no never-established fallback.

Review found two pre-existing blockers on the way: the control training loop
(`train.py`) and the auto-batch probe (`batching.py`) still passed `x[:, -1]` — the whole
conditioned row — as the anchor state, so ANY control run with conditioning columns died on
the first batch (`anchor_state(x, C)` everywhere now, with a control-path training-step
test); and the named-recipe override check in `__main__.py` rejected fields outside
`PROCEDURE_LOSS_FIELDS | CONTROL_HOOK_FIELDS` (now `| INTENT_FIELDS`, plus
`--intent-conditioning`). Campaign declaration
`docs/experiments/scene_phase0_arms.json` (O_join_lead, O_join; paired with
`control_procedure_20260905/A_control_v3`; pre-registered gate: vectored ADE < 1.5 km from
2858 m continues the design, no movement stops it).

### 2026-09-07 — Design: scene encoder + join-anchor multimodal control prediction

`4dTrajectory/ts_transformer/docs/2026-09-07_scene_join_anchor_design.zh.md`. Intent: give the
model the variables that decide the join (traffic context — the lead aircraft on the same
runway, queue, time since last landing — plus the final approach course, FAF/IF and STAR
legs as map tokens) and make the join distance an explicit K-way decision with a control
schedule per anchor (VectorNet/Wayformer-style entity encoder + scene attention, MTR/TNT-style
anchor queries, MultiPath winner-takes-all training; the differentiable rollout, envelope,
teacher and predict-time barrier all kept per mode). Deficiencies of the current
iTransformer-variate-token + flattened single head named with the measurements behind them.
Phases: 0 oracle upper bound (truth d_join + lead ETA as covariates), 1 data-plane scene index
/ context with leakage tests, 2 minimal-change context arm on the existing backbone, 3 the
new encoder/decoder with pre-registered vetoes (random-anchor control, straight-in top-1 must
not regress), 4 closure/wind/pooled airports. Reading list attached.

### 2026-09-07 — Control training design review: the below-glidepath endpoint is an objective-design fault

`4dTrajectory/ts_transformer/docs/2026-09-07_control_training_review.zh.md`. The control
baseline's endpoint 157 / 162 m below the glidepath (KRDU / KSJC medians) is the last
minute of the rollout: on the final (1–8 km) the prediction sits within ±13 m of the
glidepath, at the truth's landing time it is 540–680 m short and 140 m low with a path
angle of −4…−5° against the truth's −3°, and the rollout runs only 3–10 s past the truth
(at inference; in training simple-v3 rescales the durations to the truth's final time).
Ruled out: data and coordinates (truth endpoint +28 / +6 m), the fitted tail (median 6 / 2 s,
379 / 11 m before the threshold, no flight over 60 s). The model's last-minute controls
(load 1.014, thrust 0.069, speed 75 m/s) each sit near the inverted teacher's (0.999, ~0,
71 m/s) yet fly a different path — the open-loop imitation teacher (47× the position
term) says what the truth does at the truth's state, never what to do once drifted, and
the isotropic 10 km metre-scale position loss (state path too) prices a 150 m height
error at 2e-4 per endpoint averaged over 64 (terminal terms share the scale), with no
threshold-crossing event, no ground in the rollout, and no path gradient to the time grid. Proposals ranked: per-channel position scale / vertical-only procedure term /
threshold-plane crossing loss / ground hinge (P0), a closed-loop teacher built from the
guidance laws on the rollout's own state (P1, the nominal law moved from filter to
teacher), event-defined duration and closed-loop prediction (P2); first diagnostics named.

### 2026-09-06 — Command-hook campaigns (KRDU v1 + v2, KSJC v2): the barrier is a predict-time safety layer, training through a hook is not

Report `4dTrajectory/ts_transformer/docs/2026-09-06_control_hooks_results.zh.md`; campaigns
`control_hooks_20260906` (v1, KRDU) and `control_hooks_v2_20260906` (KRDU + KSJC), simple-v3,
one seed, paired against `control_procedure_20260905/A_control_v3`. The v2 soft barrier
applied at prediction to the untouched baseline: KRDU pooled FDE 1650 → 1449 m, ADE
1333 → 1278, FDE better on 84 % of flights, none worse by 1 km (v1: 54 such flights); KSJC
996 → 930 m, 90 % better; straight-in endpoint |xt| p95 1821 / 407 → 70 / 46 m; vertical
untouched. Hard saturation + hard gate is worse everywhere (the jump at the cone edge).
Training through a hook: six trained arms (barrier v1/v2 + nominal v1/v2 at KRDU, barrier v2 +
nominal v2 at KSJC), none beat its predict-time counterpart — pooled ADE +2…+21 %, bank skill
below the baseline, vectored path middle worse; the pre-registered lazy veto (clamped ≥ 20 % and skill below
baseline − seed noise) fires at KRDU. The nominal-law hook is the vertical complement (KRDU
endpoint height above the threshold −164 → −7 m median, vertical violation rows 46.6 → 29.1 %) and revealed that
the control baseline ends 157 / 162 m below the glidepath (median, KRDU / KSJC).
Adopted: `predict --command-hook barrier --hook-saturation soft`. Next: the combined hook.
Readouts (`readout.json/.txt`, `score.txt`) are in each campaign dir.

### 2026-09-06 — Nominal-law hook v2: thrust held to the unhooked rollout's speed (the third law)

The first campaign's `R_nominal_residual` arm (KRDU) lost the endpoint: pooled FDE 1650 →
2334 m, straight-in FDE median 703 → 1461 m, while its lateral was the most realistic of
any hook (bank skill 0.659, straight-reference bank RMS 0.54° against the flown 0.41°).
Endpoint analysis: the glidepath law pulled the aircraft up from below the glidepath
(endpoint height error −167 → −87 m) with the thrust passed through, the speed fell to
58 m/s (baseline 88) and the rollout ended 584 m short. A coordination that pays back
only each hold's path-angle change does not fix it: a trim load factor reads as "hold the
current path angle" at any path angle, so the network's intended path — and speed — is
unobservable from a segment's command. The engine
(`rollout_piecewise_constant_hooked_with_step(track_reference=True)`) now integrates the
schedule UNHOOKED alongside for hooks that declare `needs_reference`, the lag backend
passes it as `RolloutStateView.reference`, and the nominal law's thrust is
`T' = T + k·m·(V_reference − V)` (`guidance_laws.speed_hold_thrust`,
`control_nominal_speed_gain` = 0.1/s): the schedule's own deceleration is kept and only
the hook's energy cost is paid back. Tests: the speed-hold law, the thrust at and below
the reference speed, and the convergence rollout ending within 3 m/s of the unhooked
rollout's speed where the passthrough was more than 10 m/s off. R reruns in
`control_hooks_v2_20260906` (the first campaign's copies are removed).

### 2026-09-06 — Barrier filter v2: lag-aware, evaluated at the lead position, load-coordinated (after the first campaign's predict-only readout)

The first campaign's predict-only arms (hard / soft barrier on the `simple-v3` baseline, KRDU)
did on straight-ins what the penalty could not — lateral violation rows 35.5 → 1.6 %, endpoint
|xt| p95 1821 → 56 m, FDE median 703 → 492 m, ADE unchanged — but 40–49 vectored flights lost
more than 1 km of FDE (worst +14 km). Traced on the worst flight: the rate-only heading rule
(`ψ̇ = β·excess`) on 7 s holds with a 2 s bank actuator commanded +28° for two holds, then −29°
(a limit cycle), and because only the bank was changed the vertical lift component `n cos μ`
the network had paired with its load factor was lost — path angle −1° → −10°, 93 → 200 m/s,
16 km past the threshold. v2 (`control/constraints/barrier_filter.py`): the heading layer asks
for a heading CHANGE over the hold and credits the bank the actuator is already in
(`Δψ ≈ g/V_h·[tan μ_c (Δt − τ_eff) + tan μ_0 τ_eff]`, `τ_eff = τ(1 − e^{−Δt/τ})`, inverted for
`μ_c`); the corridor margins are evaluated where the aircraft will be when the command bites
(current velocity carried for `τ_eff`); the load factor is re-coordinated, `n' = n cos μ /
cos μ'`; the heading-gain default drops 0.3 → 0.1 (with the lag credit, `βΔt = 1` meant "close
the whole excess in one hold", 45° of bank for a 30° error). `RunwayAxesView` gains `hold_s`;
diagnostics gain `hook_load_change`. Tests: the lag credit and the lift invariant, and the
first campaign's worst entry replayed through the lagged rollout (capture in the first holds,
≤ 15° after, corridor bounce ≤ 120 m, path angle within [−4.5°, −1°], speed within 5 % of
the unfiltered rollout). The F arms rerun as `control_hooks_v2_20260906`
(`docs/experiments/control_hooks_v2_arms.json`); the nominal-law hook is unchanged.

### 2026-09-06 — Command hooks on the control rollout: barrier filter and nominal law + bounded residual (implemented; campaign `control_hooks_20260906`)

The control path's own constraint mechanism, per `docs/2026-09-05_control_constraint_design.zh.md`
P0 + P1. A **command hook** is called once per control segment, at the segment's start, with
the physical state the rollout carries there and the network's command, and returns the
command actually flown; the rollout integrates each segment on its own
(`aerodynamic_model.torch_piecewise_rollout.rollout_piecewise_constant_hooked_with_step`),
so the per-segment discrete adjoint stays exact and the hook's dependence on the state is
ordinary autograd across segments (finite-difference checked by the reviewer to ≤ 7e-7).
The hooked dense rollout settles the effective schedule first and then integrates it, so
dense and endpoint rollouts agree, and the **effective schedule is what a record carries**
(`EndpointControlRollout.controls`, `forecast` exports it in newtons, `source.commandHook`
names the hook). Only the first-order-lag backends support hooks (their state carries the
actuators; the point-mass backends refuse). Backends expose the state through
`control/dynamics/hooks.RolloutStateView` (transport chart + actuators + the segment's
hold `duration_s`).

Two hooks in `control/constraints/`, sharing the runway-axes adapter and the on-final gate
(`gates.py`; membership cone floored at 500 m + path alignment, read from the rollout's own
velocity): **`barrier`** — the corridor's two barriers `h = k·hw(d) ∓ xt` bound the sine
of the heading error, the heading interval is a second barrier pair
(`−β(ψ_err − lo) ≤ ψ̇ ≤ β(hi − ψ_err)`, one continuous rule — an earlier form that only
acted outside the interval clamped a centred aligned command to zero bank), the level-turn
relation makes that a bank interval, and the command's bank is saturated into it (scaled
softplus in training, hard clamp at inference), lateral only; **`nominal-residual`** — L1
lateral guidance + a glidepath flight-path-angle law (`control/guidance_laws.py`) give the
nominal bank and load factor, the command is a tanh-bounded residual on them (±5°, ±0.1).
Both are **discrete-time** rules: the command is held for Δt, so every rate gain is used as
`min(gain, 1/Δt)` (a faster rate crosses the edge inside the hold it protects). `hard`
selects the hard saturation AND the hard gate (a deployed filter has no partially-gated
rows); training refuses it. The closing term reads `corridor_halfwidth_slope`, the
derivative of the same half-width the corridor uses (zero past the threshold).

Config: `control_command_hook` (`off|barrier|nominal-residual`), `control_hook_gate`
(`on-final` only), `control_hook_saturation`, `control_barrier_alpha/heading_gain`,
`control_nominal_*`; needs control output, first-order-lag, native grid, `enu`. Predict:
`--command-hook`, `--hook-saturation` apply a hook to any lag checkpoint (the
`F_barrier_infer` arms). Epoch diagnostics `EpochResult.command_hook` (per-step gated /
clamped / saturated shares, mean bank change). Readout `compare_constraint_arms.py` gained
the outside-corridor recovery columns (first claimed on-final row outside → last inside).
Tests: `tests/test_command_hook.py` (identity hook bit-exact, state-reading hook
differentiable, point-mass refusal), `tests/test_control_constraints.py` (geometry adapter,
barrier bounds and continuity, adversarial rollout stays inside, nominal law converges
through the rollout, config guards, training refuses hard + logs the hook, export contract).
Arms: `docs/experiments/control_hooks_arms.json` (baseline + hard filter at prediction,
+ soft filter at prediction, trained through the soft filter, trained through the nominal
law; simple-v3, paired against `control_procedure_20260905/A_control_v3`).

### 2026-09-05 — Procedure penalty on the control rollout (wired, measured, not adopted); control constraint design

Code (9d9e66e): `procedure_loss` now charges the control path's native-grid rollout
endpoints against their aligned targets (`ControlStateLossResult.aligned_targets/weights`),
per-flight term + diagnostics through `ControlLossTerms`; TSConfig admits `procedure_loss_*`
on control (native grid only); control dynamics carry `glidepath_tan`; `PROCEDURE_LOSS_FIELDS`
is one source, open under named recipes (CLI + `run_naming` render `recipe+(edits)`);
`config.recipe_settings` is the single recipe-content helper for both arm runners;
`coerce_sequence_fields` makes the JSON round trip of tuple fields lossless for the dataclass
and the CLI's frozen-recipe check (subprocess-tested).

Campaign `control_procedure_20260905` (simple-v3, openap-direct cohort, KRDU + KSJC, one
seed, λ = 1e-3 / 5e-3): the penalty pulls the endpoint into the corridor (pooled FDE −101 /
−39 m, straight-in FDE −50 / −58 m, KRDU endpoint |xt| p95 2769→2203 m; KSJC's tail does not
improve) but pushes the vectored path middle away (vectored ADE +581 / +245 m), the violation
rate barely moves, and 5e-3 collapses the bank schedule at KRDU (bank skill 0.728→0.280). Report
`4dTrajectory/ts_transformer/docs/2026-09-05_control_penalty_results.zh.md`. Design for the
control path's own constraint mechanisms (rollout `command_hook`, barrier filter, nominal law
+ bounded residual; reference path deprioritised):
`docs/2026-09-05_control_constraint_design.zh.md`.

### 2026-09-05 — Final-approach corridor in the learned model: bounded output adopted, penalty vetoed, projection as fallback

Code (5b54fae): `flight_scenarios/fas_geometry.py` (FAS cone, one definition; the backend
bridge imports it), `flight_scenarios/procedure_final.py` (RNAV(GPS) document + FAF read;
`scenario_optimization` delegates), `ts_transformer/final_approach_geometry.py` (torch
corridor geometry tested against `approach_constraints`), `state_position_reference=
"corridor-bounded"` + `corridor_gate` (`StateOutputLayer` saturates cross-track/height on
the rows the output places on the final, direction from predicted positions),
`procedure_loss_*` with `ProcedureMultipliers` (fixed or dual on the violation rate,
"procedure" is the state objective's fifth component, λ logged per epoch and the selected
epoch's stored), `predict --project-final GATE`, per-flight final-approach context in the
batch context slot (`enu` chart required), predict-only arms in `run_ts_frame_ablation.py`,
readout `docs/compare_constraint_arms.py`. `StateOutputLayer.offset_mask` is no longer
persisted and `load_checkpoint` tolerates checkpoints that stored it (both 2026-09-03
generations load). Reviewed by a subagent before the commit.

Campaign `final_constraint_20260904` (KRDU + KSJC, val split, paired with the 2026-09-03 arm A,
two seeds): the bounded output improves pooled FDE −51…−91 m and pooled ADE −8…−71 m on all
four runs (two airports × two seeds) with corridor-violation rows 77→48 % (KRDU) / 34→21 % (KSJC) and the pre-registered
veto (a vectored regression on both seeds) not triggered at either airport (KRDU vectored ADE
−7 / +49 m, KSJC −28 / −204 m); the dual-ascent penalty diverged on all four runs (an
unreachable ε turns the multiplier into a ramp) and the fixed parity penalty pays 42 m of ADE
for the same violation drop; the row-by-row on-final projection (the layer's gate, hard) recovers most of B's KRDU
FDE gain post hoc but not its violation rate or endpoint tail (a first, tail-only version moved
1.35 % of rows and was replaced), the FAF-gated projection is the straight-in ceiling (KRDU FDE
643→455) at the price of vectored flights.
Report `4dTrajectory/ts_transformer/docs/2026-09-05_final_constraint_results.zh.md`.

### 2026-09-04 — Procedure constraints for the learned model: adherence measurement, design, method survey

No code path changed. `4dTrajectory/ts_transformer/docs/measure_procedure_adherence.py`
(read-only over the harvest + CIFP procedure documents, every 3rd rostered arrival) measures
how observed flights sit against the optimizer's constraint rows: 0.0 % pass an off-axis
IAF at either airport, 85–97 % (KRDU) are established in the k=0.5 LPV cone by the FAF, and
once established 87–99 % of samples satisfy the cone and the −60/+120 m glidepath window
(±22 m over the whole final: 14–69 %). `docs/2026-09-04_procedure_constraints_design.zh.md`
turns that into the design (final-segment corridor + glidepath only, gated by the flight's
own join distance, runway-scale hinge in training + projection / casadi tracking at
inference, shared geometry via a torch dispatch in `approach_constraints.mathx`);
`docs/2026-09-04_constraint_methods_survey.zh.md` surveys the alternatives (bounded output
reparametrization, differentiable projection layers, primal-dual training, sampling +
filtering, two-stage predictor + optimizer, data-side) with reading lists and the P0–P3
implementation order (shared geometry → projection arm → bounded reparametrization paired
with a primal-dual penalty arm → optimizer tracking objective).

### 2026-09-03 — state-v2 candidate: anchor-relative state positions (mechanism fixed, recipe vetoed)

`TSConfig.state_position_reference` (`absolute` default | `anchor-relative`):
`StateOutputLayer` can read the forecaster's position channels as displacements from the
anchor and add the anchor's normalized position back, so "start where the aircraft is"
is the zero output. `compare_frame_arms.py` now reports the first predicted step against
the anchor's kinematic extrapolation and takes `--only`; `run_ts_frame_ablation.py`
gains `--informal` (no experiment manifest / clean-tree guard, as
`run_ts_control_arms.py` runs). Four runs (KRDU + KSJC, two seeds) paired against arm A
(`docs/2026-09-03_state_v2_anchor_relative_results.md`): the first-step jump collapses
(KRDU established flights −348…+239 m → 0…7 m lateral), straight-in FDE 643 → 492 m and
lateral miss +204 → +21 m, KSJC pooled ADE −55…−86 m on both seeds — but KRDU vectored
FDE +350 m on both seeds (worse on 78–82 % of them), endpoint lateral p95 ×1.4, so the
pre-registered veto fires and `absolute` stays the default. Reading: absolute output
= "end at the origin" prior, anchor-relative = "start where you are"; each stratum
wants a different one. Next candidate: absolute output + first-step continuity term.

### 2026-09-03 — Runway-hypothesis expansion: what the runway label is worth, and what recovers it

`run_ts_runway_hypotheses.py` runs a trained threshold-anchored checkpoint once per
candidate runway for every validation flight (clone the flight dict with that runway's
CIFP target, same `build_series`/`forecast` chain, forecast mapped back to world
coordinates and scored in the true runway's chart) and evaluates selection rules over
the K hypotheses; the assigned label reproduces the baseline exactly. Includes a
mirror-image pseudo-sibling per flight (same separation and course as the real parallel
sibling, opposite side) as the noise control for any oracle. Result
(`4dTrajectory/ts_transformer/docs/2026-09-03_runway_hypothesis_expansion.md`, KRDU +
KSJC, two seeds each): a causal active-configuration rule from co-temporal development
landings recovers the runway DIRECTION (KSJC 93.8 % overall; KRDU majority runways
80–83 %) but not the parallel side (KRDU 05R/23L 29–31 %), which costs +19 % pooled FDE
at KRDU (+30 % on straight-in flights, ~500–800 m on the minority runways) and nothing at
KSJC (230 m separation). The real-sibling oracle gains 79 m median FDE at KRDU against
32 m for the fake sibling, so roughly half of a K=2 sibling oracle is luck; the
forecast's own closest approach is not a usable selector. The parallel side is the one
genuine unresolved mode. The 150–200 m NW endpoint bias of arm A at KRDU was then traced
(`docs/2026-09-03_krdu_nw_endpoint_bias.md`): a model-side, world-fixed translation of
the whole predicted path from the first step, on straight-in flights, with the sign of
KRDU's population-mean lateral drift and below the objective's resolution; observed
tracks and CIFP geometry are on the centreline to metres.

### 2026-09-03 — Airport-center frame ablation: the threshold anchor IS the target conditioning

Asked whether the ts_transformer's only runway knowledge — the chart being anchored at
the assigned threshold — is a prior worth keeping (plan
`4dTrajectory/ts_transformer/docs/2026-09-03_airport_frame_ablation_plan.md`). Three
commits of mechanism, one of tooling, then 14 runs. (1) `channels.target_chart_position` /
`FlightSeries.target_chart` names the target's chart position; the five consumers that had
silently equated it with the origin (observed crossing plane, inference truncation,
`horizontal_distance_m`, and two the plan missed — the fixed-anchor common-grid truncation
that selects checkpoints, and the difficulty covariates) now measure from it; verified
bit-identical on 300 real KRDU arrivals in both threshold frames. (2)
`coordinate_frame="airport-enu"`: `AirportENUFrame` anchored at the airport reference
point from the harvest's own `runway_thresholds.json` entry
(`flight_scenarios.runway_target.airport_reference_point`); `COORDINATE_FRAMES` moved to
`coordinate_frames.py`, config imports it. (3) `target_conditioning="channels"`: five
input-only channels through iTransformer's vendored covariate-token path;
`TSConfig.input_channels` / `enc_in` = input width; checkpoints serialise
`input_channels`; `batch_contract.anchor_state` replaces every `x[:, -1]`; PatchTST refused.
(4) `run_ts_frame_ablation.py` (state arms, paired split, resumable, no per-arm CV, no
CZML) + `docs/compare_frame_arms.py` (pre-registered stratified/paired readout with the
endpoint's cross-track against the assigned centreline).
Result (`docs/2026-09-03_airport_frame_ablation_results.md`): at KRDU the airport frame
makes the deterministic model average across each parallel pair — endpoints nearer the
sibling runway 1.5 % → 12–15 % on both seeds, the minority runway (05R, 23L) pulled
570–680 m toward its majority sibling, its median FDE +30–45 %, p95 lateral endpoint
error ~2× — and the target coordinates fed as data change none of it; the vectored-stratum
gain the plan hypothesised flipped sign on the second seed (KSJC −184 → +62 m); the only
consistent effect of the conditioning is −10 % final-time MAE (the flattening duration
head reads it, the attention backbone does not). Seed floor: threshold arm 5–22 m pooled
ADE, airport arms up to 107 m. Decision: keep `enu`. Also verified: the v5 re-roster
CLAUDE.md still called pending was already on disk for all five airports with rosters;
the 14 failures in `trajectory_data_process/tests/test_ts_pipeline.py` /
`test_download_landings.py` reproduce at the pre-change commit (recorded in
`docs/code-health-followups.md`). ts_transformer suite 419 pass.

### 2026-08-24 — Result pickers rank a split's results by mean/p95 ADE/FDE

The Prediction and Experiments pickers listed a split's results in name order only —
no way to see which model/arm actually scored best. Each category's aggregate accuracy
existed only inside its own `comparison_index.json` `prediction` block, too heavy to
fetch per category just to order a dropdown, so `build_scenario_comparison_czml.py`'s
`category_accuracy_summary` now stamps a compact `accuracy` block (`adeM`/`fdeM`, each
mean + p95 — a SUBSET of the index block, never recomputed) onto every prediction
category entry in `categories.json`, and
`ts_transformer/docs/backfill_category_accuracy.py` backfilled the 59 published
categories from their own indexes (idempotent, metadata-only). The frontend gains a
"Sort results" selector (Default / ADE mean / ADE p95 / FDE mean / FDE p95) shown for
the Prediction and Experiments sources: prediction categories are ranked best-first
WITHIN their split optgroup (cross-split error comparisons are meaningless, so the sort
never mixes groups), experiments are ranked within their campaign by the split
currently in view (val fallback), options show the metric value inline, and entries
without a value keep their order at the end. An earlier misread of this request
(per-trajectory ADE/FDE sorting inside one category, incl. a 140k-record group-level
backfill) was fully reverted the same day — group records carry no `adeM`/`fdeM`.
Verified: frontend 523 pass + tsc clean, aeroviz-4d python suite 155 pass.

### 2026-08-24 — One naming grammar for every ts_transformer run and published category

The frontend's learned-prediction categories carried three label dialects (verbose
pipeline concatenations, publisher `run_id (model, output, horizon)` strings, ad-hoc
hand labels like "budget: lr 1e-4"), and the Experiments picker showed raw run-directory
names — none of which said what was actually experimented. New
`ts_transformer/run_naming.py` is the single source of a canonical grammar derived
mechanically from a run's serialized config: `output · backbone · dynamics · loss ·
meta`. Dynamics distinguishes control-derivative handling (`point-mass` vs
`first-order-lag`, `@backend` when non-default); loss names a `custom` run against its
NEAREST recipe (fewest loss-field edits, e.g. `simple-v3+(imit=16)`; >4 edits → stable
`custom-<hash>` version); meta lists deviations from today's defaults (seed first,
capped, `+N more`) plus caller extras (run id, campaign/arm, cohort). Wired into
`run_ts_pipeline.py` (replacing ~60 lines of label concatenation and five dead label
helpers), `publish_ts_experiment_trajectories.py` (labels + a stamped
`experiment.label` the frontend now prefers; new `--refresh-labels-only` walks stored
publication manifests and re-derives labels without predicting anything), and
`experiment_index.py` (`display_name` + INDEX.md Name column). Applied on disk as
metadata-only edits — 45 legacy `ts_*` categories via the one-off
`docs/relabel_published_categories.py`, 14 publisher-managed categories via the refresh
mode, indexes rebuilt; no record, CZML, checkpoint, or directory was renamed or
deleted (category keys/dirs are historical record; `run_slug()` covers future ones).
Notable relabel finds: the "velocity 2x" wiggle arm is exactly `simple-v2`'s frozen
dose, and the imitation "~47x" arm is exactly `simple-v3`. Verified: ts_transformer
suite 407 pass (label-assertion tests updated to the new grammar; the terminal-clock
collision test now passes the arc-length selection metric its synthetic plan always
implied), frontend 520 pass + tsc clean.

### 2026-08-24 — Anchor calibration landed: manufacturer cluster gone; speed-gate toggle in the viewer

A320-family landing Cl_max 2.7 → 3.0 (calibrated from Airbus's VLS = 1.23·Vs1g +
published VLS figures; pin test holds the modeled floor at-or-just-below published
VLS) and the C56X airframe restored from OpenAP's C550 surrogate to certificated
facts (`6e31f2d`). Republished: A319/A320/A321 speed-pass 46.6/54.8/41.5 % →
**78.9/85.4/80.3 %**, C56X fast-fail 53.5 → 20.7 %, composite of-decided now nearly
uniform across airports (71.2–75.8 %) — confirming airport spread was anchor ×
fleet mix. Residuals documented, not patched: A21N/A20N (neo subfleet lands far
below MLW — needs an operational landing-mass model), B763, E75L. Results doc §8;
follow-up 14 resolved to those residuals. The same commit family adds the Details
window's **speed-gate toggle** (`af32454`): two-gate/three-gate verdicts re-derived
client-side from per-row component results via `composeVerdict` (a declared mirror
of `_composite`, truth-table pinned) — no schema change, published reports stay
three-gate.

### 2026-08-24 — First baseline speed-gate measurement: the anchor, not the weather, dominates

`evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md` — full statistics of the first
three-gate fleet baseline (per airport, per runway, per airframe type, slow/fast
margins, wind-explainable bands). Headlines: fleet speed pass 22,473/32,191 graded
(69.8 %); fails split 66 % slow / 34 % fast with median margins 3–5 kt; ungraded is
almost entirely unresolvable airframes (10,306 — GA-heavy KRDU/KSJC hit hardest);
**the fail structure clusters by MANUFACTURER** (A21N 91.9 % slow vs B738 75.7 %
pass, same days, same airports), so the per-type stall-model anchor — not wind — is
the dominant effect, recorded as code-health follow-up 14 (review A320-family
Cl_max/landing mass; the same anchor is the optimizer's velocity floor). Airport
inversion (KMSY/KSTL fast-skewed vs KSJC 88 % slow) is fleet mix, the KSJC-ADE
lesson again: quote speed rates per type, never bare.

### 2026-08-24 — The baseline runs the speed gate: three gates for every subject

Owner decision, superseding v6's observed exclusion: the observed baseline is judged
on the SAME three gates as its modeled twins. The fitted crossing GROUND speed is
graded against the stall-anchored window as a STATED PROXY for airspeed — the wind
caveat (10 kt headwind = half the window) travels with a distinct criterion id
(`vref_1p23_vs1g_to_vref_plus_20kt_ground_speed_proxy` on every observed row's
bounds) and `methodology.terminal_speed.observed_proxy_caveat`, never silently.

- **The window anchors on the flight's own airframe**: `harvest/observed.py` resolves
  each record's icao24 through `flight_scenarios.resolve_landing_aero` (the scenarios'
  identity→OpenAP chain, no fallback type) and writes `source.landing_aero` + the
  type's landing mass — so baseline and modeled twins share one set of stall
  assumptions. Unresolvable airframe: NOMINAL_MASS_KG, no landing_aero, speed grades
  indeterminate loudly (reason named), composite indeterminate.
- **Composition**: speed ∧ lateral ∧ vertical for observed exactly as for computed —
  the headline baseline pass rates MOVE (previously two-gate). `crossing_speed_ms`
  (airspeed) stays null on observed rows; the proxy lives in
  `crossing_ground_speed_ms` — two quantities, two names, never mixed.
- Frontend: the observed Details speed-gate card returns (it is a real pass rate
  now); the gates note explains the proxy; per-row speed verdicts + windows render.
- `evaluation/docs/THRESHOLD_SPEED_GATE.md` §5 records the override and the caveats;
  the pipeline-integration test now pins the unresolved-airframe outcome.

### 2026-08-24 — Unified record design: observed records say where their crossing lives

Evaluation now grades every subject through ONE state interpolation. The design
("2b"): an observed record's `source.crossing_span` marks its crossing —
`measured_bracket` (the instrument-selected direct bracket: left_index + fraction,
reproduced from the event) or `fitted_tail` (exactly one inferred crossing row
appended after the measured states; MSL, V = event ground speed with a recorded
fallback, t = trapezoidal estimate nothing grades on). Computed records carry no
marker — the artifact under test cannot author the quantity it is graded on
(`record_from_dict` enforces observed-only).

- **One crossing definition fleet-wide**: `final_approach.crossing.bracket_fraction`
  + `interpolate_channels` replace three hand-rolled copies (harvest direct bracket,
  evaluator computed path, ts dataset supervision truncation). The marker schema +
  `validate_crossing_span` live beside the event contract — the same
  producer/evaluator seam pattern.
- **Producer half**: `flight_scenarios.crossing_span.crossing_span_from_event`;
  `harvest/observed.py` is the one span-producing writer (optimizer/ts references
  carry no event and stay markerless → still `unavailable`/indeterminate, unchanged).
- **`_observed_arrival` no longer reads event geometry** — the event remains for
  identity/staleness validation, the datum cross-check, audit copy, and the
  ground-speed audit stat. `final_time_s` stays pinned to the last MEASURED row
  (span-aware contract), so flight-time and Δt-vs-observed statistics do not move;
  `TrajectoryRecord.measured_states` keeps path-shape/span comparisons on flown
  trajectory only. Two subject branches remain BY POLICY: missing-crossing outcome
  (reception gap ≠ model shortfall) and speed-gate scope (ground speed never graded).
- Deliberate small semantic changes, all in descriptive columns: observed rows'
  `speed_ms`/`heading_rad` now measure the crossing state against the measured-end
  target (previously tautologically 0), and `along_track_m` is the projected ≈0
  instead of exactly 0. Verdict-bearing quantities (lateral, vertical, verdicts,
  flight times, ground speed) are preserved — pinned by the republish diff below.
- Fixtures made physically self-consistent: the test event's crossing position now
  ENCODES its cross-track offset instead of contradicting it (the old event-based
  reader never noticed; honest geometry does).
- Pre-span records in `approach/records/` fail loudly with the cure
  (`--evaluate-only`); all five airports republished after the change.

### 2026-08-24 — Observed crossing ground speed wired through the report to the Details window

The event's audit speed now actually reaches a reader — the missing consumer half of
today's harvest change (additive within report schema v6):

- `ArrivalDeviation.crossing_ground_speed_ms` is filled by `_observed_arrival` from the
  event; rows carry it flat + under `deviation`, batches carry a
  `crossing_ground_speed_ms` spread, and `METHODOLOGY["observed_crossing_ground_speed"]`
  declares it: ADS-B ground-referenced, audit-only, never composed, never fed to the
  stall-anchored gate (the graded-airspeed slots stay untouched — pinned by test).
- **Flat-row bug fixed on the way**: `crossing_speed_ms` lived only under
  `row["deviation"]` while the frontend verdict table reads it flat, so the "V crossing"
  column would have shown dashes on real v6 reports. Both crossing speeds are now flat.
- Details window: a "crossing ground speed (m/s, ADS-B)" aggregates row, and the
  per-row V-crossing cell falls back to the audit value with an explicit
  "audit only, not graded" tooltip. Observed reports still show no speed-gate card.
- Observed evaluation reports republished for all five airports (`--evaluate-only`)
  so the on-disk v6 artifacts carry the new fields; lateral-pass rosters rebuilt after
  (the `arrivals._clear` deletion footgun, again).

### 2026-08-24 — Fleet reclassified: every stored event now carries the crossing ground speed

All five airports ran `--reclassify-existing` (267,194 stored tracks), so the
`crossing_ground_speed_m_s` field from today's harvest change is now IN the data:

- **Rosters unchanged, as required**: tracks and arrivals counts are identical to the
  saved before-state at every airport (assignment is deterministic; only the event
  payload gained the field). Spot checks (400 assigned tracks/airport): **100 % of
  estimated events carry the speed**, direct and censored alike, medians 65.6–72.7 m/s
  (≈ 128–141 kt). KRDU is ~95 % censored events where KMSY/KSMF are ~99 % direct —
  the coverage-ends-short signature, visible per airport for the first time.
- KMSY/KSMF/KSTL/KRDU ran serial (~210 tracks/s, 91 s–7 min 21 s each); KSJC ran on
  the new `--jobs` path (24 workers) in **5 min 0 s** end-to-end for 106k tracks —
  the classify fan-out is no longer the bottleneck, the serial arrivals/CZML/report
  tail is.
- KSJC's first attempt died on `ENOSPC`: the stage-then-swap needs the airport's
  tracks footprint free (2.2 GiB) and the disk was at 100 %. Staging unwound cleanly
  (`tracks/` untouched — the design working as intended). Freed by deleting pure
  caches only (conda tarballs + unused packages, pip, vscode-cpptools; user-approved);
  `~/.cache/opensky` (18 G) deliberately untouched. Disk now ~14 G free — the pending
  optimizer batch (12.3 GiB) fits again.
- The five `lateral_pass_eligibility.json` rosters (deleted by the re-roster footgun,
  missing since 2026-08-21) were rebuilt via `ensure_lateral_pass_roster`. That fixed
  the 3 `test_ts_pipeline` failures that read the real harvest root; the other 11 are
  fixture rot at HEAD (hermetic fixtures predate the roster requirement) — recorded as
  code-health follow-up **13**, left to the active ts-pipeline workstream.

### 2026-08-24 — Threshold event carries the estimated crossing ground speed (audit-only)

The observed threshold event now serializes `crossing_ground_speed_m_s` — additive
within `runway-threshold-event-v1`, optional on read (every stored event predates it):

- **Direct events** interpolate the bracketing samples' `reported_ground_speed_m_s` at
  the position's own crossing fraction; those speeds were already required by the
  bracket's source-integrity checks, so the field exists for every direct event.
- **Censored events** OLS-extrapolate speed vs along-track to the plane over the SAME
  kept samples as the position fit, using the same estimator (`final_approach.fit_line`,
  the previously-private `_fit_line` made public — one OLS, not two) and the fit's own
  sample/span standard. When the speed-bearing subset cannot meet it, the field is
  omitted and `diagnostics.ground_speed_fit.omitted_reason` says why — distinguishing a
  post-change unfittable event from a pre-change one.
- The shared contract (`final_approach.event_contract.validate_event`) accepts
  absent/null as unspecified and rejects a present non-finite or non-positive value.
- **It is GROUND-referenced (ADS-B velocity, wind unmodelled) and audit-only**: nothing
  feeds it to evaluation's stall-anchored airspeed gate, observed subjects stay
  speed-ungraded, and no verdict anywhere changes.
- **Blast radius, verified before building**: ts_transformer training never reads the
  event (`dataset.py` computes its own crossing from waypoints) and optimizer targets
  come from `flight_scenarios/fitted_approach.py` / runway data — no retraining, no
  re-solving, no schema bump. Existing harvests gain the field only via
  `--reclassify-existing` (`--evaluate-only` re-rosters stored events unchanged).
- Suites: final_approach + harvest + evaluation + flight_scenarios + backend green. The
  14 `test_ts_pipeline.py` reuse-guard failures observed alongside are PRE-EXISTING
  disk state, not code: every airport's `arrivals/lateral_pass_eligibility.json` is
  missing (the documented re-roster deletion footgun) and those guards read the real
  harvest tree — rebuild via `lateral_eligibility.ensure_lateral_pass_roster`.

### 2026-08-23 — Observe: Optimization result source restored; legacy v5 reports displayable again

Two regressions the Observe panel had accumulated, both verified in-browser on KRDU:

- **The optimizer categories had no home in the result-source selector.** The 07-29
  experiment refactor partitioned categories into only `prediction | experiment`, so
  fitted_adsb / runway / runway_cons (whose manifest entries predate `resultSource` and
  never stamp it) fell through to the Prediction dropdown's "Other evaluation results",
  mixed in with the ts model outputs. `trajectoryResultSources.categoryResultSource` is
  now the single three-way classifier (`optimization | prediction | experiment`; absent
  `resultSource` + non-`ts_` key ⇒ optimization — the same legacy rule
  `EvaluationSummary.evaluationKind` already used, which now delegates to it), and the
  ControlPanel gained an "Optimization" source with its own category selector.
- **"Details" failed with "evaluation report is malformed" for every published category.**
  The 08-23 speed-gate commit bumped the report schema to v6 in all four homes, which made
  `isEvaluationReport`'s exact-version check reject every report on disk — all still v5,
  and the optimization ones cannot be regenerated (their record batches were cleaned up;
  the batch rerun is a standing open item). The reader now accepts the enumerated
  `LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS` (v5 only; the v6 speed fields are typed
  optional and nothing in the window renders them yet), and `EvaluationReportWindow`
  banners a legacy report as pre-speed-gate — its verdicts grade lateral+vertical only —
  instead of refusing to open. v4-and-earlier stay rejected (shape changes, not just
  grading).
- **Follow-up: the Details window now renders the v6 speed-gate statistics** — a
  pass/graded summary card (with the ungraded count), a crossing-speed aggregates row,
  the speed-gate sentence in the gates note, and per-row `speed_result` + `V crossing`
  columns whose tooltip carries the flight's own stall-anchored window (the bounds are
  mass-dependent, so no common band exists to chart). All of it keys on the schema
  version, so a legacy v5 report renders exactly as before, banner included. The card is
  additionally suppressed for OBSERVED-subject reports (the regenerated observed
  baseline is already v6): observed tracks are never speed-graded by policy, so the card
  could only ever read "— · N ungraded" and was mistaken for a data problem on sight.

### 2026-08-23 — Optimization-tree review: batch-driver merge, resume config guard, plan-timeline fix

Full review of `4dTrajectory/optimization/` + the pipeline runner; every confirmed finding
fixed in one pass (suites re-run green; the two pre-existing known failures — numpy 2.x
`test_fixed_time_objective_weights_control_effort_at_one`, `test_optimizer.py` — unchanged).

**Correctness fixes**

- **`--resume` now verifies the solver configuration, not just identity.** Every eval
  record (solved AND failed) is stamped with the batch's `optimization_config`;
  `_resumable_record` rejects a mismatch or a missing stamp, so a resume across a changed
  `--max-iterations`/`--fitting`/`--rollout-dt` re-solves instead of silently absorbing the
  old records and stamping the new config over the whole roster (which `--skip-optimize`
  would then have trusted). Records written before this date carry no stamp → a resumed
  batch over them re-solves everything, deliberately.
- **Summary rows quote the eval record's `final_time_s`** (the replay's last sample), not
  the planned NLP horizon — for guard-truncated replays (the `not_reached` family, 12.5 %
  of a measured KRDU runway batch) the two differ by seconds, and resumed rows (rebuilt
  from eval records) disagreed with their own fresh twins.
- **Constrained plan exports carried time-warped timestamps.** `_node_states_to_samples`
  spread the dense nodes evenly over `[0, T]`, but multiphase node spacing is
  `(T_p/n_seg)/m_sub_p` with per-phase auto `m_sub` — the orange "Optimizer plan" CZML
  track for every runway_cons flight animated wrong. `CollocationOptimizer._extract` now
  exports `last_dense_state_times_s` (pure helper `dense_node_times`), and both export
  paths use it. Positions were always right; existing states files have wrong
  `optimizer_states[].t` only.
- **Shortest-IAF ranking is now the 3D polyline length.** The old Lagrange-curve proxy
  inflated cornered routes (measured +38 % on a two-corner T-arrival, +7 % mild dogleg,
  exact on straight) and could pick a genuinely longer IAF; a fly-by path only cuts
  corners, so the polyline is the tighter monotone proxy (`_path_curve_length_m` →
  `_path_length_m`).
- **`_concat_to_runway` no longer treats two missing fixIds as a match** ("" == "" passed
  the fix_id half; mismatch now requires no PRESENT identifier to match) — the
  optional-fields-compared-to-each-other trap from the coding conventions.
- **Space pre-check refuses only runs that genuinely don't fit**: the estimate now drops
  the CZML family when 'czml' is not in `--outputs` (the refusal message already suggested
  that remedy without it working), and nets each artifact family against what its target
  directory already holds — a `--resume` restart or `--skip-optimize` rebuild no longer
  re-demands the full footprint (which forced `--skip-space-check` on a 98 %-full disk).
- `casadi_optimizer.py` (legacy multiple-shooting, live via the backend): terminal ψ pin
  is now `sin(Δψ/2) = 0` — the plain difference against ψ boxed to [−π, π] read a due-west
  target reached on the other branch as a 2π violation. Also removed the dead `dt` param
  of `segement_integrate_expr` and the broken dead `decision_vector_to_geo_state`.
- `variable_time_warm_start_transcription_optimizor`: the final-time guess is clamped into
  `build_final_time_bound()`'s box — any `arrival_time_s > 1000 s` used to make
  scipy.least_squares reject x0 outright (masked in production by the backend input clamp).
- `approach_constraints`: `ConstraintSet.evaluate` disambiguates colliding violation names
  by segment position (same-kind default-ident legs silently DROPPED the earlier leg's
  rows from the NumPy report; optimizer path unaffected); box-leg `lateral_left/right`
  labels un-inverted (box axis = flight direction, final axis opposes it — feasibility
  unaffected, only report naming); `ConstraintReport.summary(tol_m, tol_rad)` takes the
  caller's tolerances; `TargetFrame.to_ne` fails loudly across the antimeridian (both
  transforms share the non-wrapping assumption); `intercept_angle_deg` documented
  numeric-validation-only (its `fabs` kink sits at the aligned optimum).
- Experiment scripts: `transport_term_comparison` ψ-cross check is now relative
  (the 1e-18 absolute tolerance passed the committed run by 8.6e-19);
  `fixed_enu_frame_error --max-range-km` honoured by the grid radii; scheme-comparison
  orders derived from the fitting name (7-entry hand-list covered 7 of 16 schemes);
  30 km-study docstring named a nonexistent function for system A.

**Structure**

- `optimize_scenarios` / `optimize_scenarios_constrained_iaf` are thin fronts over ONE
  `_run_batch` driver — the "batch edition seam class" (three past bugs from updating one
  copy) is gone.
- `REFERENCE_CACHE_SCHEMA`, `file_sha256`, and the record→track path mapping
  (`observed_track_path`) live in `evaluation_export.py`; the batch AND the runner import
  them (the runner's restated mirror + pin test replaced by a shared-import seam test).
- Workers no longer ship the rollout states twice across the process boundary (the eval
  copy is emptied worker-side; the parent's `states_ref` points at the states file).
- The runner builds each `Plan` once (space check and run share the objects), and its
  reuse validator memoizes SHA-256 by (path, mtime, size) — runway/runway_cons validate
  the same shared reference set, which was ~1.5 GB re-hashed per category.

### 2026-08-23 — Threshold speed gate (report schema v6) + evaluation review

**A third component joins the terminal verdict: the crossing speed must lie in
[1.23·Vs1g, 1.23·Vs1g + 20 kt]**, with Vs1g the project's own 1-g stall model at the
record's crossing mass. 1.23 anchors on 14 CFR 25.125(b)(2)(i) (V_REF ≥ 1.23 V_SR0);
the +20 kt window on FSF ALAR Briefing Note 7.1's stabilized-approach speed element.
Design, worked numbers, rejected alternatives, and trackable sources:
`evaluation/docs/THRESHOLD_SPEED_GATE.md`. Mechanics:

- `aircraft.aero_params.stall_speed_ms` is the ONE stall-speed definition — the
  optimizer's velocity floor (`scenario_optimization._stall_speed_ms`, now a thin
  wrapper) and the new gate import it, so admitted solves and their judge share one
  stall model by construction.
- `flight_scenarios.build_scenario` writes `source.landing_aero =
  {wing_area_m2, cl_max_landing}`; both record producers copy `scenario.source`, so
  optimizer and ts records carry it going forward. A computed record WITHOUT the block
  grades speed-indeterminate loudly (absent = unspecified, incl. explicit null);
  a malformed block raises. Records written before this date lack it.
- **Observed subjects are reported `indeterminate` and never composed** — their V is
  ground speed (wind unmodelled) and ADS-B ends a median 325 m short of the threshold,
  so no crossing airspeed was ever measured; observed composites are bit-identical to
  v5. `ArrivalDeviation` gained `crossing_speed_ms`/`crossing_mass_kg` (None for
  observed).
- Report schema v5 → v6 in all four homes (producer, ts seam import, frontend mirror
  + fixtures via the constant). **Every on-disk v5 report is stale**; regenerate
  before the ts pipeline or frontend reads it. New surface: per-row `speed_result` +
  speed bounds in `bounds`, batch `speed_result_counts` + `crossing_speed_ms` spread,
  `methodology.terminal_speed`.
- Known interaction, deliberate: the optimizer floor (1.10·Vs, admits observed
  touchdown-speed targets) sits BELOW the gate's 1.23·Vs — floor-riding min-time
  solves and fitted-ADS-B/track-end-target solves can legitimately fail speed; and the
  category-default 145 kt target V_ref exceeds the window top for light narrow-bodies
  (E75L-class tops at ~137.5 kt), so `runway`-target solves for those types will fail
  speed until the per-type approach data is refined (follow-up recorded).

Also: an `evaluation` package review (this change's findings that were NOT fixed here
are in `docs/code-health-followups.md` — roster FileNotFoundError path, missing
monotonic-t validation, STATE_KEYS mirror in `arrival.py`, and more).

### 2026-08-21 — KSJC's ADE advantage is a route-mix artifact; 75 takeoffs were in the arrivals

Investigating why KSJC trajectories looked "weirdly short" and its ADE/FDE remarkably better
than the other four airports. Two separate findings; full write-up with every table in
`docs/2026-08-21_ksjc_route_mix_and_ade.md`.

**The ADE gap is composition, not skill, and it reverses under standardisation.** KSJC's data
is not truncated — it is *straight*. Its reception is the best of the five (99.0 % of tracks
start at the 30 km crop edge, max sample gap p50 2.1 s against KMSY's 6.6 s, 1 coverage gap in
400 tracks), and the arrival slice cuts the same ~5.1 km annulus everywhere. What differs is
the flying: whole-segment tortuosity p75 is **1.017** at KSJC against 1.96–2.38 elsewhere,
96.6 % are established on the centreline at 20 km to go (KSTL 38.7 %), and at 15 km out the
interquartile cross-track spread is **12 m**. It is visible before any slicing — median path
inside the raw 30 km crop is 29.1 km at KSJC against 34.6–44.8 km — and at the crop edge
44.8 % are already within 15° of the approach course (KSTL 3.2 %, median 126°, still outbound).
Cause is operational: 86 % of arrivals on 30L, Santa Clara valley plus SFO/OAK Class B funnel
traffic onto the centreline outside 30 km.

Stratifying evaluation flights by post-anchor tortuosity × remaining path, **ADE inside a
stratum is equal across airports** (412–509 m median on "straight, <13 km left"). KSJC simply
has 78.6 % of its flights there against 41.8–61.0 % elsewhere. Reweighted to the pooled mix,
KSJC's ADE median goes **483 → 1526 m — from best of five to worst**, and on its own vectored
flights it already was the worst (3931 m against 2000–2249 m at KSTL/KRDU). The recognisable
signature: ADE and cross-track improve while **FDE does not** (1000 m against KRDU's 1019 m),
because a straight route makes only the lateral channel easy.

Fixed by making the mix impossible to omit: new `ts_transformer/approach_difficulty.py` writes
`route_tortuosity`, `remaining_path_m`, `anchor_range_m`, `anchor_cross_track_m` and
`established_at_anchor` onto every prediction row, plus an `accuracy.difficulty` batch block
carrying the mix and the thresholds the flag encodes. Computed from the observed track the
error is scored against — never the prediction — and in world EN rather than chart axes, so
the numbers do not change meaning across a `coordinate_frame` ablation. Verified against the
independent measurement: KSJC 78.2 % established (78.3 % standalone), KSTL 53.8 % (54.0 %).

**Separately: 75 rostered "arrivals" were takeoffs.** `arrival_segment.py` classified a
never-left-the-ring track as a local circuit only if it started within
`LOCAL_START_RADIUS_KM = 5` of the DESTINATION, so a takeoff from a neighbouring field inside
the 25 km ring passed and was kept whole — first sample on a runway a few km away, on the
ground. **64 of the 75 are KSJC** (KRHV ×28 at 7 km, KPAO ×20 at 21 km, KNUQ ×16 at 11 km),
KSMF 8, KMSY 3, KRDU and KSTL zero; KSJC is the only one of the five ringed by satellite
fields. Every one was long enough to reach the TS dataset. Impact on the metrics is small
(0.24 % of KSJC evaluation flights anchor within 5 km of the threshold against 0.10–0.16 %
elsewhere) — a correctness fix, not the ADE explanation, though it overlaps the known
"duration head cannot predict below ~125 s" item.

`arrival_segment` gained a required `field_elevation_m` (no default — the rows are HAE and a
silently MSL reference would shift the test by the geoid separation without failing) and a
third outcome `"takeoff"`, rostered as `excluded.outcome = "takeoff_in_segment"` with
`ground_start_agl_m` published in the manifest. `truncate_flights` now returns
`(arrivals, locals, takeoffs)`.

Two decisions worth keeping. **The criterion is altitude alone**: over 42 725 arrivals the
first-sample height above the landing runway is bimodal with an empty band — 75 at or below 82.1 m,
**zero between 100 and 150 m**, next at 175.3 m — while ground speed does NOT separate the
populations, because a jet at rotation reads 71–80 m/s on the runway; a speed-and-altitude
rule would have kept 29 of the 75. **The test reads the segment the ring cut produced**, not
the raw track, so a flight that departs a neighbour, leaves the ring and comes back stays a
genuine arrival.

**Stale outputs.** The manifest schema is bumped to `harvest-arrivals-v5-takeoff-excluded`;
loaders compare exactly, so every on-disk v4 manifest now fails loudly. Rebuild with
`python -m trajectory_data_process.harvest --airport <ICAO> --evaluate-only` (re-rosters from
stored `tracks/`; no download, no reassignment). KSJC drops 11 146 → 11 082. Existing ts
checkpoints were trained on cohorts containing the removed flights, and published per-airport
ADE/FDE tables should be re-derived standardised, or at minimum quoted alongside
`established_at_anchor_fraction`.

### 2026-08-20 — `simple-v3`: the control schedule is now supervised directly

`simple-v2` scored position (derivative order 0) and velocity (order 1). Bank lives at
order 2, so nothing in the loss ever named it, and unsupervised it landed **below a
trivial baseline**: on KRDU the predicted bank carried less information about the flown
bank than a randomly chosen other flight's did (per-flight skill 0.124 against a
random-flight floor of 0.170; a same-runway twin reaches 0.679). The visible symptom was
the one reported from the viewer — curves and reversals on references that are dead straight.

New `control_imitation_loss_weight` scores the predicted schedule against the one
`control_inverse_dynamics` reads off the flown track, through the **same registry the
forward model dispatches on**, so target and rollout can never be different equations. The
target is built in `dataset.reference_control_supervision` on the training-only
`_dynamics_arrays` path — `dynamics_arrays()` stays free of it because forecast/predict call
it and there is no future to invert there. Default 0.0, so simple-v1/v2 stay bit-identical.

`simple-v3` = `simple-v2` + `control_imitation_loss_weight = 64.0` (~47x the position term).
On 1404 KRDU validation flights: bank skill 0.124 → **0.735**, flight-independent share of
the bank 49.0 % → **3.3 %** (KRDU's own flown tracks: 1.8 %), straight-in bank RMS 3.92 →
**0.36°** (0.41°), sign reversals 5 → **0**, ADE better on **57.0 %** of flights (656 → 501 m,
p=1.9e-7), FDE unchanged. First change in this investigation that buys control structure
without paying accuracy for it.

Dose chosen off an eight-point ladder (0 / 0.74 / 1.47 / 2.94 / 11.8 / 11.8-seed2024 / 47 /
188x). Two things worth keeping: below ~11.8x the ladder is a **noisy plateau**, not a ramp —
the 1.47x arm is worse than 0.74x on every metric, so sampling only that region concludes the
term barely works. And at 188x the fit **saturates and overshoots** — 0.24° straight-in bank
and a 3.0 % shared share are both past the flown tracks' own values, smoother than reality
rather than closer to it.

Methodology, recorded because it changed several readings: bank skill must be read against the
random-flight floor and same-runway twin that `docs/score_control_arms.py` now prints per arm,
never against 1.0 — doing so also inverted the earlier loss-design conclusion, since the
velocity dose frozen into simple-v2 is the only one below the floor. The twin is a yardstick,
not a bound (the 47x arm exceeds it). And at n=1404 the paired sign test returns p = 3e-16 for
**pure seed noise**, so its p values mean "reproducible direction", never "large effect".

Not done: KRDU only, `val` split only, and 47x has a single seed — the 11.8x seed pair shows
seed noise is 3-8x smaller than the dose effects, which is an inference, not a measurement.
Artifacts for all eight arms (checkpoints, 1404 predicted flights, evaluation reports,
comparison CZML) are under `4dTrajectory/outputs/KRDU/experiments/imitation_design/` and
published to the viewer. Full write-up:
`4dTrajectory/ts_transformer/docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md` §12.

### 2026-08-19 — the optimizer pipeline: two blockers, a bounded population, and eight cost levers

Audit of `run_scenario_optimization.py` and everything it shells out to, then the fixes.
Measured throughout on the real harvest (42,725 rostered arrivals across 5 airports) rather
than reasoned about; every number below was reproduced end-to-end.

**Blocker 1 — `runway_cons` evaluation crashed on the first record, on every runway.**
`_snap_target_to_procedure` moved the constrained solve's target onto the procedure
document's last waypoint. That waypoint and the arrival manifest's `runway_target` are two
renderings of the SAME CIFP threshold and round differently — measured 0.05–0.22 m over the
25 runways in service (KRDU 32 = 2.98 m, KSMF 35R = 39.45 m, neither in the arrival set).
On 2026-08-17 `evaluation.arrival._require_target_agrees_with_runway_data` gained its
POSITION half at a 1 cm tolerance, so **25 of 25 runways failed**. It went unnoticed because
that day's regeneration covered observed and prediction records only — the CHANGELOG entry
says so: "There is currently no optimizer comparison tree published at all". `run_for_airport`
uses `check=True` with no handler, so the first `runway_cons` cell would have aborted the
whole sweep about 2 h into KMSY.
*Fix:* the snap is gone. The scenario target is already the authoritative
`harvest.airports.Runway` point (that is why the unconstrained `runway` mode always passed),
so the constrained path now keeps it and VALIDATES the procedure against it at the
optimizer's own `_FRAME_ANCHOR_TOLERANCE_M` (150 m) — the same displaced-threshold
mis-anchor the snap existed to catch (KSJC 12L was 390 m off against the NASR config) still
fails, loudly. `evaluation/tests/test_pipeline_integration.py` now pins both ends together
across all of KRDU's runways; before, `_snap_target_to_procedure` was unit-tested only in
isolation and no test ran a constrained record through `python -m evaluation`.

**Blocker 2 — the fitted-ADS-B dataset could not be built for 4 of 5 airports.**
`build_scenario` raises when a flight has no usable `final_approach` fit, which aborts the
whole airport. Measured: **35 flights of 42,725 (0.08 %)** — KMSY 1, KRDU 1, KSJC 25,
KSMF 8, KSTL 0. So 0.08 % of the fleet blocked 82 % of the dataset.
*Fix:* `UnusableFittedApproach` (its own type, so nothing broader is swallowed) and a new
`flight_scenarios.dataset` batch layer that drops exactly those flights, names each one, and
writes the accounting to `<scenarios>.selection.json`.

**Blocker 3 — the run did not fit on the disk.** At the full population the artifacts came
to ~36 GB against 17.8 GiB free. `run_scenario_optimization.py` now estimates the footprint
from measured per-flight sizes and refuses to start rather than filling the filesystem
30 hours in (`--skip-space-check` overrides).

**Population.** `--max-per-runway` (default 2000 in `prepare_scenario_inputs.py`) keeps N
arrivals per runway, evenly spaced over landing time so a capped runway still spans the whole
harvest window. The rostered fleet is wildly unbalanced (KSJC 30L 9,603 vs 12L 14), so an
uncapped batch spends most of its compute re-measuring two runways. At 2000:
**23,453 flights / 70,359 solves** (KMSY 3534, KRDU 7491, KSJC 3543, KSMF 3737, KSTL 5148),
down from 42,725 / 128,175. The selection is derived from the ROSTER only — no source track
is opened for a discarded flight — so it is target-independent, both prepared datasets pick
the same flights, and KRDU's build dropped from 2.4 GB / 24 s to 0.9 GB / 7 s.

**Cost levers, measured.** 120 random KRDU arrivals, `--jobs 20`: runway 159 s wall / 18.1
CPU-s per flight, fitted_adsb 146 s / 17.6, runway_cons 175 s / 9.7.

1. **IPOPT iteration cap is the dominant lever and had no knob.** Serial A/B: the 8 flights
   that ended `Maximum_Iterations_Exceeded` cost 448 s (~56 s each, the full 3000-iteration
   budget); 8 that solved cost 45 s (~4.3 s each). **6.7 % of the flights, ~48 % of the CPU.**
   `--max-iterations` is now plumbed from both runners through both solve paths, and is
   recorded in `summary.json`'s `optimization_config` — a lower cap converts slow successes
   into failures, so it is a different experiment and `--skip-optimize` will not reuse across
   the change. The DEFAULT is unchanged at 3000: that is a research decision, not a
   performance one.
2. **`--jobs`.** The library auto is half the cores (right for a call that should leave the
   box usable); the pipeline driver, which owns the machine for the batch, now defaults to
   `cores - 4` (24 here) instead of 14.
3. Population cap, above.
4. **`--rollout-dt`** exposed on the runner (default unchanged, 0.5 s). The simulator array
   is ~75 % of every `*_states.json`; 1.0 s takes the estimated 2000/runway footprint from
   16.6 GiB to 10.4 GiB.
5. **The observed reference track was written twice.** The fitted-ADS-B and runway datasets
   reference the same flights and their reference records differed only in `target_state` —
   ~200 bytes of a ~67 KB record. The records now quote a shared sibling
   `shared_references/observed_tracks/` store through the contract's existing `states_ref`
   indirection, which `evaluation.records.load_record` already resolves. Measured
   134 KB/flight → 64 KB. Cache contract bumps to
   `optimization-references-v3-shared-tracks` and hashes the track alongside the record; the
   store is swept against the union of all sibling reference dirs, never one dataset's roster.
6. **`python -m evaluation` materialized every record.** `load_records` resolved each
   `states_ref` into the full state list and held the batch — measured 0.5 MB per record, so
   ~7 GB on uncapped KRDU. `summary_row` already carries `arr_airport`, so contexts resolve
   from the roster and the records stream (`iter_records`). `evaluation.visualize` does the
   same in two passes, remembering only which FILES were drawable and reloading the sampled
   30. Verdicts identical (48/48, 44 pass) on the A/B.
7. **`--max-groups-per-czml`.** One CZML per runway at 38–54 KB/flight makes a 2000-flight
   runway a single ~100 MB file. The frontend already loads CZMLs named by each index group's
   own `czml` field, so splitting is transparent to it.
8. **Resilience.** `--resume` reuses complete record pairs for scenarios in the current
   roster and solves only the rest (`_clear_stale_records` still sweeps orphans — resume
   narrows which files survive, it never turns the sweep off); summary counts now come from
   the roster rather than from what this process happened to write, and a roster that ends up
   incomplete raises. `--continue-on-error` keeps the sweep going past one failed cell and
   names the casualties at the end.

**Serialized precision — 30 % of the footprint was decimal digits nobody reads.** Asked why
the run needed 16.6 GiB, the answer turned out to be float text: a single state row is 185
bytes at full repr (`"lat":35.766821578167715`, 17 significant digits for a quantity whose
ADS-B source resolution is metres), and the timed arrays are ~98 % of a record. Records now
serialize at a declared precision (`evaluation_export.STATE_DECIMALS`: 1.1 mm position, 1 ms
time, 0.1 mm/s speed, 2e-9 rad angles) and the comparison CZML mirrors it. Measured: records
**31 %** smaller, reference + observed track **21 %**, comparison CZML **40 %**, and the
2000/runway estimate **16.6 GiB → 11.7 GiB**. A/B on a KRDU batch: **0 verdict or
event_status changes**, largest deviation difference 0.63 mm — four orders below the
metre-scale gates.
Two things this deliberately does NOT touch: `initial_state`/`target_state`, because
`_require_target_agrees_with_runway_data` measures them at **1 cm** and that budget cannot
absorb a rounding step; and `final_time_s`, which is now read back OFF the serialized array
rather than recomputed — the contract requires it to equal `states[-1]["t"]` to 1e-6, and
writing the two independently made `record_from_dict` reject the entire batch on the first
record (caught immediately, which is the contract doing its job).

**Also fixed:** every optimizer evaluation row reported `flight_key: null` while observed
rows carried it — `build_scenario` never copied it onto `scenario.source`. It does now
(only when the flight has an `id`, since `flight_key`'s fallback is a list index this
function does not have).

**Verified end-to-end** on 48 KRDU flights (cap 12/runway, binding on all four runways)
through the real runner: 3/3 cells complete, `runway_cons` 44 pass / 4 fail — the mode that
could not be evaluated at all before. Resume re-solved 6 deleted records in 4.15 s and
reproduced the same 47/48. Suites: 1091 passed, 13 failed — the same 13 pre-existing
failures as before the change (the documented numpy `test_optimizer` one, 11 `test_ts_pipeline`,
1 `test_download_landings`), plus 153 aeroviz-4d python.

### 2026-08-19 — Viewer: the predictor-input window takes its forecast's verdict colour

**Problem.** In Observe → prediction comparison, a group draws as two paths: the predictor
input window (`look-`) and the forecast it produced (`pred-`). The forecast was painted from
its terminal verdict — green pass / red fail / gray indeterminate — while the input window was
painted a fixed faded purple. Purple is a colour nothing else in the legend uses, so the input
half read as a THIRD kind of result rather than as the first half of the track it belongs to,
and a viewer could not tell at a glance which input fed which forecast in a crowded scene.

**Change (frontend rendering only).** `applyComparisonRenderModel` now resolves the outcome
colour for `look-` exactly as it does for `pred-`, at the kind's own alpha:
`predictionOutcomeColor(status, COMPARISON_KIND_ALPHA[kind])`. Both halves share the group key
(`groupOfEntityId` strips the prefix), so one comparison-index entry colours both. Hue is now
the group's verdict; **alpha alone** (85/255 vs 225/255) separates input from forecast. The
purple `COMPARISON_KIND_COLORS.lookback`/`.predicted` survives only as the no-verdict fallback,
which is why the two constants are deliberately equal.

The "Predictor input" checkbox swatch follows: `comparisonKindSwatch` now returns the
green/red/gray split gradient for `lookback` as well as `predicted`, and `ControlPanel` already
multiplies each swatch by `COMPARISON_KIND_ALPHA[kind]`, so the input row renders as the faded
version of the prediction row — the same relationship the two paths have in the scene.

**Not changed: the CZML.** `build_scenario_comparison_czml.py` still bakes `PREDICTION_COLOR`
and `LOOKBACK_COLOR` (both purple) into the packets; the viewer has always repainted prediction
paths from the legend, so the file colours are unobserved in the app but ARE what any external
CZML consumer sees. Logged as a future improvement in the root `README.md` ("Align the
comparison CZML's baked colours with the frontend contract") with two suggested fixes: generate
the builder's table from the frontend legend (the `geoConstants.json` pattern), or bake the
verdict the builder already knows and drop the repaint.

**Tests.** `comparisonRenderModel.test.ts`: the old "lookback keeps the legend purple on an
offTarget group" expectation was the assertion of the behaviour being replaced and is now
"repaints a failed lookback red"; added the no-verdict fallback case, an indexed-status-wins
case, and an explicit input-hue == forecast-hue equality (the previous version compared the
input against the purple constant, which passed trivially because both constants were purple).
81 files / 510 tests pass; `tsc --noEmit` clean.

### 2026-08-18 — ts_transformer: control-mode design pass, first-order control lag, teacher-inverse audit

Three connected pieces of work on `4dTrajectory/ts_transformer`, driven by the 2026-08-17
meeting note (`docs/MeetingNotes/note_8_17.md`).

**1. The control path carried every historical experiment axis; simple-v1 walks one of them.**
Removed, each with the recorded verdict that closed it: control-mixture (paused 2026-07-30),
direct durations (ablated negative 2026-07-30, ADE +21.0 % / FDE +36.6 %), trim-residual
controls (never published a result), the `physical-criteria` and `terminal-state` objectives and
their checkpoint-selection metrics, progressive-N teacher pretraining (rejected 2026-08-02), the
one-off rollout-finetuning gate, and `benchmark_validation_execution.py` (zero references, zero
tests). That is −2 of 5 tracking objectives, −2 of 5 selection metrics, −1 of 3 duration
parameterizations, the whole `control_value_parameterization` axis, and one of three prediction
outputs. `oracle_teacher.optimization` also owned a second stage type that only converted itself
into `ControlTrainingStage`, and the 60/120/240/full schedule was written out twice; both
runners now call one `teacher_optimization_stages()`.

Two live bugs surfaced only once the modes hiding them were gone: `run_ts_pipeline` had no
`objective_label` entry for `true-time-position`, so `PredictionPlan` raised `KeyError` on the
current frozen recipe; and the arc-length recipe builds a **365-byte** directory name against a
255-byte path-component cap (names over the cap now keep their head and end in a digest of the
whole name). `TSConfig.from_dict` restated the same missing-field check 25 times and is now one
table.

**2. A switchable flight model with first-order control lag.** The point-mass model applies a
piecewise-constant control instantly, so a learned schedule steps the bank angle N times across
an approach: curvature is discontinuous at every segment boundary and the implied roll rate is
unbounded. The meeting asked for continuity. The three controls become states chasing their
command — `d(mu)/dt = (mu_cmd - mu)/tau_mu`, likewise thrust and load factor — and
`aerodynamic_model/torch_lag_dynamics.py` **wraps** `transport_chart_rhs` rather than restating
it, so the force equations, stall handling, WGS84 transport term and chart projection are
literally the same code.

Measured both directions, because "it reduces to the old model" is a claim that has to be
checked: at `tau = 0.1 s` the two trajectories end **within 0.5 % of path length** and the gap is
first order in tau; at the 2 s default they differ by **~3 km over a 240 s rollout**. So it is a
materially different model, not a smoothing pass.

Switchable on one axis (`control_dynamics_model` ∈ `point-mass` | `first-order-lag`), orthogonal
to the state representation, with a `simple-v1-lag` recipe that is simple-v1 with that one field
changed so a paired comparison measures the flight model rather than a bundle. `tau_bank` joins
the CV grid (0.5/1/2/3/4 s) and is dropped as inert under `point-mass`.

**New numerical constraint, worth remembering:** explicit RK4 on `y' = -y/tau` is unstable above
`h/tau = 2.785`, so a swept time constant shorter than the integrator step produces **NaN, not a
degraded rollout**. `TSConfig` refuses it at construction rather than letting it be discovered as
a dead training run.

**3. Controls are dimensionless, and the thrust floor is negative.** `thrust_fraction = T/T_max`
puts the three controls on one magnitude and makes the same box mean the same thing on every
airframe — one sigmoid output used to mean 100 kN on a small jet and 400 kN on a heavy, so a
teacher schedule or a learned bias was not transferable across the fleet. Newtons now appear in
exactly two functions: into the dynamics, and out to the evaluation record (whose contract is
unchanged and shared with the CasADi optimizer).

The floor moved from 0 to **−0.2**, because a real approach needs net-negative force: idle thrust
plus the drag of speedbrake, flaps and gear, none of which the clean-configuration polar models.
This was measurable. On the KSJC outer-train cohort (24 flights × 64 segments), **39.7 % of
inverted teacher thrust segments pinned at the 0 N bound** — the teacher structurally could not
reproduce the deceleration the aircraft actually flew. After: **0.33 %**, with a median required
thrust of 3.6 % of installed (idle) and p5 at −12 % (drag augmentation). `flyability.py` already
treated negative required thrust as a SOFT violation for exactly this reason. The optimizer's own
`make_control_bounds` is deliberately NOT changed — that box is a flyability claim, this one is a
learned head's search space — so no optimizer artifact is restaled.

**4. The teacher inverse is now the inverse of the configured forward model, by construction.**
This was the specific thing asked for, and the audit found real problems.

A schedule solved against equations the training rollout does not integrate is finite, bounded,
the right shape, and its own optimizer reports a falling loss. It simply reproduces nothing, and
nothing downstream can tell. So `control_inverse_dynamics.py` registers each inverse under the
SAME config key as its forward model, and `tests/test_control_inverse_dynamics.py` closes the
loop numerically for every registered model: roll a known schedule, invert the dense result,
require the schedule back (recovered to 5e-3 in thrust fraction and load factor, 0.5° in bank).
A model added without an inverse fails at registry lookup.

Found and fixed in the old path:

- `build_inverse_dynamics_target` hard-unpacked a 7-field batch. `dataset.batch()` returns 7
  fields only under `control_state_loss_grid='fixed-dt'`, so under **every native-grid recipe,
  simple-v1 included, the teacher builder died on a bare tuple `ValueError`**. It only ever
  worked because `run_ts_oracle_teacher_optimize.py` builds its own `custom` fixed-dt config.
- It rebuilt reference velocities with a hardcoded `smoothed-position-difference` regardless of
  `config.reference_velocity_source`, i.e. it inverted a different velocity definition from the
  one the supervision targets are built from.
- The inversion was a hand-written numpy inverse of the flat-ENU RHS with no link to
  `config.control_dynamics_backend`. It agreed with the transport-chart backend by accident
  (measured 0.5 N / 1e-6 load factor), not by construction — and adding lag states would have
  broken that silently. It now adds back the `omega x v` term the chart RHS subtracts.
- The lagged inverse is the actual-control inverse plus `u_cmd = u + tau * du/dt`, sharing its
  first stage, so the two cannot drift apart.

The lagged model's actuator initial condition comes from the same inversion applied to the
observed lookback (`dataset.anchor_controls`, 11 samples ≈ 20 s), so it reads no future and the
"inverse must match the forward model" requirement is structural rather than teacher-only.

**Measured on real data (2026-08-19), paired.** Both frozen recipes trained 180 epochs on
KSJC with the same manifest, roster and `split_seed`, then predicted, evaluated and published;
both arms scored the SAME 1083 validation flights, so the per-flight sign tests below are
paired. Artifacts in `4dTrajectory/outputs/KSJC/experiments/flight_model_paired/`, published to
`aeroviz-4d/public/data/airports/KSJC/comparison/ts_ksjc_flight_model_{point_mass,first_order_lag}/`.

- **The lag does what it was added for.** jerk p95 **−28.0 %** (lower on 99.8 % of flights,
  p≈1e-320), turn rate p95 −6.0 % (86.4 %, p=5e-141), acceleration p95 −5.5 % (77.4 %, p=2e-76).
- **Without costing accuracy — it gains a little.** ADE **−3.4 %** (better on 67.4 % of flights,
  p=7.9e-31, bootstrap 95 % CI on the mean delta [−32.7, −17.0] m). This is what separates the
  result from the blandness trap the README documents for flyability, where a worse predictor
  scores better by drawing straighter lines: here smoothness and accuracy move together.
- **The endpoint does not move.** FDE and arrival-endpoint error are ties (49.1–49.2 % win rate,
  p=0.58–0.63). Coherent: a lag smooths the turn geometry along the way, not where the model
  believes the runway is.
- **Read the smoothness as a modelling artifact removed, NOT as "more realistic".** Both models
  were already smoother than the flown tracks (jerk p95: observed 4.253, point-mass 0.30×, lag
  0.21×), so the lag moves them further from observed statistics — it is closer to observed jerk
  on 0.2 % of flights. Observed jerk from 2 s ADS-B positions triple-differentiated is dominated
  by quantisation noise and is not a target to chase; what the lag removes is the curvature
  discontinuity a piecewise-constant control creates at every segment boundary.
- **Gates are unchanged: 0 pass / 1083 fail on both arms**, consistent with the standing
  "forecast ≠ certifiable approach" finding. Changing the flight model does not change that.

**τ_bank cross-validation: not resolved.** 3 folds × 36 epochs over {0.5, 1, 2, 3, 4} s put the
best (2.0 s, 1014.2 m) and worst (4.0 s, 1072.3 m) **5.7 %** apart against **11–23 %** fold noise;
τ ∈ {0.5, 1, 2} lie within 1.5 % and τ=1.0 beats τ=2.0 on 2 of 3 folds. By the recipe's own rule
(<2 % ⇒ no reliable difference) τ=2.0 is a defensible default, not a CV-selected value.

**A CV trap worth remembering: `DEFAULT_CV_PATIENCE = 6` is too small for this recipe.** BOTH
flight models pass through an early ADE transient while every loss component falls monotonically
— on the unchanged point-mass recipe: best 1534 m at epoch 2, bump to 2132 m at epoch 6, then
1268 m at epoch 13 and still falling. Patience 6 stops inside it, and the first τ sweep had to be
discarded: the same τ=0.5 scored 1674.6 m on a fold early stopping caught and 1234.4 m on one it
did not — a 26 % difference that was pure stopping artifact. The frozen recipes' own patience=20
clears the ~7-epoch bump; a CV sweep needs patience raised to match.

**Performance.** The lag step originally ran eager on CUDA while the point-mass backends cached a
`torch.compile`d one: ~95 s vs ~15 s per epoch. The blocker was passing `state_scale` through a
closure — `torch.compile` caches per code object, so a fresh closure per rollout rebuilt the
kernel every batch. Moving it into `step_context` took the lagged epoch to **8.2 s (11.6×)** and
the τ sweep from ~15 h to ~40 min.

**Stale artifacts.** Every existing control-output checkpoint. The control contract changed units
(newtons → fraction) and `TSConfig` gained required serialized fields, so `load_checkpoint`
refuses them rather than mis-scaling thrust by five orders of magnitude. `state`-output
checkpoints are unaffected. No optimizer, harvest, evaluation or comparison artifact changes.

### 2026-08-17 — Comparison references were the wrong window: full track vs model arrival slice

**Symptom.** In the comparison overlay the white observed reference did not start at the
same time or the same place as the `look-`/`pred-` group beside it. Measured on the KRDU
05L validation batch (471 groups): the group's first sample sat a median **5055 m** from
the reference (p75 5281 m, p95 47.1 km, max 54.9 km).

**Root cause — two time origins, only one of them reconciled.** Three timelines exist and
the publisher accounted for two:

1. a stored track's `samples[i][0]` is relative to first reception (`store.track_record`,
   absolute time in `start_time_utc`) — this is what `trajectories.czml` and the
   `/trajectories` backend served;
2. the model **arrival slice** is rebased at `harvest/arrivals.py` `load_arrival_flights`
   (`t0 = waypoints[0][0]`, `sample[0] - t0`), so every scenario, optimizer record and TS
   record has `t = 0` at the 25 km terminal-ring entry, and `t0` is discarded;
3. a prediction record rebases again to the anchor, recording the shift as
   `source.anchorTimeS`.

`build_scenario_comparison_czml` added ③ back (correctly — that was the 2026-07-20 anchor
fix) but nothing ever added ② back, so every group rendered `t0` early. Measured `t0` over
300 random KRDU arrivals: median **45.1 s**, p25 34.3, p75 55.6, p95 123.1, max 526.3 s.
The same seam applies to optimizer groups (`opt-`/`sim-` also start at ring entry); no
optimizer category happened to be published at the time, so it only showed on predictions.

Nothing downstream could detect it: both timelines start at `t = 0`, both name the right
flight, the schema is satisfied, and the drawn result reads as model error rather than a
publication bug. Proof that the two were the same measurement: fitting a per-flight pure
time shift dropped the lookback↔reference distance from a median 4900 m to **13.7 m** (2 s
resampling error).

**Fix — align the reference to the modeling window, at READ time.** The pre-entry segment
is not model input, not a supervision target and not evaluated; drawing it as the white
"truth" beside a forecast invites reading it as something the model failed to produce. So
the reference is now the arrival slice on the arrival origin, rather than the group being
pushed out onto the full track's origin.

- `aeroviz_backend/observed_trajectories.py` gained `window` ∈ `full` | `arrival`. `full`
  (default) is unchanged: the complete reconstructed track, rostered by
  `tracks/manifest.json`, for Observe/Baseline. `arrival` rosters from
  `arrivals/manifest.json` and builds flights through **`load_arrival_flights` itself** —
  the same loader the scenario/optimizer/training paths use — so there is no second
  implementation of the slice to drift from, and the source-hash check plus identity round
  trip come along for free.
- `tracks/` is untouched and no artifact is written: the slice is taken at read time, the
  same rule the altitude-outlier repair follows.
- Response schema bumped to `observed-trajectories-v2` with `trackWindow` echoed. The bump
  is load-bearing: a v1 backend ignores an unknown `window` argument and answers a
  comparison-reference request with full tracks, reproducing the bug silently. The
  frontend refuses anything but `arrival` for the comparison reference.
- `build_scenario_comparison_czml._require_reference_aligned` pins the embedded-reference
  path (`include_reference_entities=True`, currently unused in publishing) at 50 m — inside
  the gap between resampling noise (~14 m) and a wrong window (≥5 km).

**Verified end-to-end on real data**: over all 471 KRDU 05L groups the group-start-to-
reference distance is now **0.0 m** for every group (bit-identical samples), against a
median 5055 m before.

**No artifact is stale.** Published comparison CZMLs contain only `look-`/`pred-` (the
publisher passes `include_reference_entities=False`) and the reference is served live, so
restarting the backend is the whole deployment — no re-publish, no re-predict, no
re-optimize.

### 2026-08-17 — `final_approach` / `evaluation` design pass: one event validator, one lateral bound, authoritative threshold frame

Review of the two packages for design and defensive code. Report schema bumps to
`terminal-approach-evaluation-v5`.

**What "stale" means here, measured — no number changes.** Every shipped
`evaluation_report.json` must be regenerated, but only because its *shape and version*
changed; re-running produces the same verdicts and the same deviations:

- The effective lateral bound is unchanged. A shipped KRDU v4 row reads
  `{guidance_lateral_m: 53.375, runway_lateral_m: 22.86, effective_lateral_m: 22.86}` —
  the runway term won, as it does at all 26 thresholds — and v5 publishes `lateral_m:
  22.86`. Same number; what disappears is the `53.375`, which was the 2×-wrong half of an
  already-inert term.
- The authoritative-frame change (C) moves nothing on disk. Probed 500 observed + 500
  predicted KRDU records against the CIFP-resolved context: worst `target_state` offset
  from the authoritative threshold is **0.0000 m** (altitude likewise), so no frame origin
  moved and every deviation is bit-identical. Observed records carry no `target_source` at
  all and therefore take the STRICT branch — and pass it.
- So KRDU observed re-runs to the same `{pass: 14168, fail: 270, indeterminate: 1}`.

What actually changes: `schema_version`, the `bounds` / `resolved_limits` field names, the
new `methodology.terminal_lateral`, and `evaluation_context_fingerprint` (context v2 gained
`threshold_lat`/`threshold_lon`). **Comparison CZML files do NOT need rebuilding** — their
geometry comes from states and their `status`/colour from verdicts, and neither moved; only
`evaluation_report*.json` and the `evaluation.schemaVersion` string inside
`comparison_index.json` do.

Scope on disk: all 34 published comparison trees are PREDICTION trees (`ts_pooled_*`,
`prediction_ts_itr_*`, `experiment_*`) plus the per-airport `observed/` report. There is
currently **no optimizer comparison tree published at all**, so the optimizer batches were
not part of this regeneration.

**Regeneration completed 2026-08-17.** Everything below is v5 on disk:

- 5 observed batches via `--evaluate-only --no-czml` (harvest + published copy). Verdicts
  reproduce the historical table exactly — KRDU 14168/270/1, KMSY 3892/257/1, KSJC
  11144/7/6, KSMF 4221/8/2, KSTL 8485/280/4 — confirming the "no number changes" claim
  end-to-end. CZML deliberately not re-rendered (unchanged, and already post-altitude-filter).
- 34 prediction batches re-evaluated. 14 of them had their records archived into
  `prediction_records.tar.gz`, so `python -m evaluation` failed on them until the tarball was
  extracted, evaluated, and the extracted members removed again to restore the archived
  layout (the tarball does carry the `*_eval.json`, 2166 of them in the KSJC/val case).
- 34 published comparison trees refreshed report-only, following the publisher's own atomic
  pattern (new generation-named report → atomic index replace → prune) and using its own
  `publish_evaluation_report` / `evaluation_batch_stats` / `prune_unreferenced_outputs`, so
  the shapes cannot drift. The index carries no source path and the itr/ptst variants of a
  split share a flight set, so each tree was joined to its source by CONTENT — the exact
  per-row `(file, cross_track_m)` fingerprint — which matched all 34 uniquely, 0 ambiguous.
- 5 `lateral_pass_eligibility.json` rosters rebuilt (KRDU's was deleted by the arrivals
  rebuild, which clears its directory; the other four pinned the old manifest hash and v4).

Two things this surfaced, neither caused by the v5 work:

- Re-running the harvest **changes every `arrivals/manifest.json` hash**, because the
  altitude-outlier filter block (`altitude-outlier-filter-v1`) had never been written to
  them. The reports also shed five stale `deviation` keys (`glidepath_deg`, and the four
  `*_sigma_m`/`*_interval_m` fields) that the code had already stopped emitting. The on-disk
  artifacts were behind the code in more ways than the schema version.
- Prediction **checkpoint provenance was already broken before this session**: the KSJC
  checkpoint pins arrival-manifest `687f5c6c1d94bb54` while the tree held `362b0dd97ff823ef`
  even before the re-run. A full republish (`publish_ts_experiment_trajectories`) preflights
  that hash and would refuse — which is why the refresh above is report-only. Anything
  needing a real republish must re-run prediction against the current manifests first.
  `publication.json`'s `evaluation.schema_version` therefore still reads v4: it records the
  publish run that actually happened, and nothing validates it.

**Contract consolidation (behaviour-preserving).**
- The `observed_threshold_event` schema had TWO hand-rolled validators — `harvest
  .threshold_event.require_current_threshold_event` and the inline block in
  `evaluation.arrival._observed_arrival`, ~65 lines each over the same payload, free to
  drift. One `final_approach.event_contract.validate_event(event) -> status` now owns
  the schema; each side keeps only its own identity binding (producer: `Runway` frame
  fingerprint + snapshot; evaluator: `AssessmentContext` runway + fingerprint). Both run
  identity first, payload second, so a stale artifact reports as stale.
- `INBOUND_TOLERANCE_M = 100.0` existed twice with a "same as the final-approach fitter"
  comment; it is now imported from `final_approach`.
- Removed: `evaluation.metrics._validate_deviation` (re-checked values the same function
  produced three lines earlier); the 45-line self-consistency re-derivation in
  `_validated_observed_availability` (the counts are measured upstream from a roster
  evaluation never sees — now copied verbatim like `observed_threshold_event`, with only
  the denominator LABEL checked); `AssessmentContext.from_dict` (unused); the
  `source.get("arr_airport") or source.get("airport")` fallback duplicated in two modules
  (no producer has ever written `"airport"` — replaced by `TrajectoryRecord.airport`);
  `_line_inliers`' rejected-index half (both callers discarded and recomputed it); the
  provably unreachable span check after `_straight_final_suffix`; and the type-shape half
  of `AssessmentContext.__post_init__` (kept: benchmark whitelist, since `limits()`'s
  second branch is a fallthrough; positive runway width; NaN; LPV completeness).
  `METHODOLOGY` moved to module level. Net −115 lines with the new validator counted in.
- Single-statement functions that only added an indirection: `arrival.subject_of`
  (a `source["subject"]` lookup behind a cross-module import and a package export — now
  read directly at its two call sites), `stats.mean` (a reimplementation of
  `statistics.fmean`), and `thresholds._require_finite` (a module function taking the
  one object that called it — inlined into `__post_init__` over a `_FINITE_FIELDS`
  tuple). An AST sweep over both packages checked the rest: the ~13 remaining
  one-statement definitions are dataclass properties naming a domain quantity
  (`SegmentFit.cross_at_threshold_m`, `TrajectoryRecord.solved`), `to_dict`
  serializers, or public helpers with three or more callers — kept. `event_contract
  ._source_sample_range` returned a tuple no caller used; it is now
  `_require_source_sample_range() -> None`.

**Lateral criterion: one honest bound (methodology change).**
`limits()` computed `min(guidance, runway_width/2)` where guidance was the LPV course
width `/2` or LNAV 0.15 NM. Two findings: the LPV value 106.75 m is already a semiwidth,
so halving it published `guidance_lateral_m` wrong by 2×; and measured over all 26
thresholds at the five airports the guidance term bound **0 times** (2.3–18× wider than
the runway). `ResolvedLimits` now carries a single `lateral_m` = runway half-width,
tagged `LATERAL_CRITERION_ID = "runway_half_width_at_threshold"`, and the report's new
`METHODOLOGY["terminal_lateral"]` states the claim boundary: landing geometry, not
navigation containment. `lpv_lateral_fsd_m` → `lpv_course_width_m`, kept as procedure
provenance that bounds nothing. Lateral is now never indeterminate once a crossing was
measured, so the composite's only indeterminate route is a missing vertical reference.
No verdict changes (the min already selected the runway half-width everywhere).

**Deviations are measured in the authoritative runway frame.**
`_computed_arrival` built its `RunwayFrame` at the record's own `target_state` lat/lon
and cross-checked only the altitude against the published LTP+TCH. `AssessmentContext`
gains required `threshold_lat`/`threshold_lon`; the frame origin is now authoritative,
and `_require_target_agrees_with_runway_data` checks position AND altitude at 1 cm —
gated on `source.target_source == "runway_threshold"`, because `fitted_adsb_crossing`
and `track_end` (both in `run_scenario_optimization.ALL_MODES`) aim elsewhere by design.
An undeclared `target_source` gets the strict reading. Two consequences worth recording:
a fitted-ADS-B-target record used to be graded against its own flight's fitted crossing
(~0 lateral by construction), and the repo's two threshold sources — NASR
`runway_thresholds.json` vs CIFP Path Point LTP — sit **6.69 m apart at KRDU 05L**, which
nothing had ever noticed. The real pipeline is internally consistent; only
`ts_transformer/synthetic.py` builds on the NASR point, so its test context pins those
coordinates explicitly.

**Follow-up pass: the v5 bump reached the producer only, and two guards were over-deleted.**

*Alignment.* `REPORT_SCHEMA_VERSION` v4 → v5 landed in `evaluation/metrics.py` alone.
`4dTrajectory/ts_transformer/lateral_eligibility.py:18` and
`aeroviz-4d/src/data/evaluationReport.ts` kept their own v4 literals, so
`build_lateral_pass_roster` raised on every regenerated report and the frontend's
`isEvaluationReport` rejected them — the panel surfaces `"evaluation report is malformed"`,
which is a misleading message for what is purely a version disagreement. Both suites were green
because their fixtures also said v4: **a version pinned in the fixture is a version the
test cannot check**, and that is the whole reason this shipped. The ts seam now imports
`REPORT_SCHEMA_VERSION` (it is the one ts module allowed to know evaluation policy — model,
loss and loader code still must not); the frontend exports
`EVALUATION_REPORT_SCHEMA_VERSION` as a declared MUST-match mirror; all four fixtures
import the constant. `thresholds.CONTEXT_SCHEMA_VERSION` also became
`terminal-assessment-context-v2`, since its hashed payload gained `threshold_lat/lon` and
renamed `lpv_lateral_fsd_m`. Docs realigned: `FINAL_APPROACH_VERDICT_STANDARD.md` §3.3 now
documents the single runway-half-width rule and why the guidance term went (its §2/§5 v4
history is kept as history), plus `evaluation/README.md`, `BUG_FIX_GUIDE.md`, the zh
implementation doc and the ts evaluation doc. `EvaluationReportWindow.tsx` still advertised
"the tighter of the guidance bound and runway half-width" — the exact claim this pass
deleted — while rendering a report whose `METHODOLOGY` said the opposite; its test now
asserts the new claim and asserts the old wording is ABSENT.

*Over-deletion (both verified by running the code, then covered by tests).*
- `validate_event` compared `ESTIMATED_OBSERVABILITY_BY_METHOD.get(method) != observability`.
  With NEITHER field present that is `None != None` → False: an `estimated` event carrying
  no method and no observability validated clean, then fell through to the censored branch
  and was graded as a real crossing. Each single-field case already failed, which is why
  the gap survived. Now the lookup must resolve.
- `record.source.get("target_source", THRESHOLD_TARGET_SOURCE)` returns `None`, not the
  default, for a key present with a null value — so an explicit null took the "aims
  elsewhere on purpose" early return, bypassing the cross-check exactly where the docstring
  promised it could not be bypassed. Audited: all 144,764 shipped `*_eval.json` carry an
  explicit `"runway_threshold"`, so nothing on disk was mis-graded.
- Restored the positivity check on `threshold_crossing_height_m` (dropped with the type-shape
  half of `__post_init__`). It sets the vertical REFERENCE PLANE, published values run
  15.27–18.11 m, so a zero would move the plane by most of the ±22 m window with every
  verdict still looking clean — the same "parsed FAA data, not a Python literal" argument
  that kept the NaN check.

*Kept deleted, on review:* the `hae_minus_msl_m` presence check (the harvest writes it
unconditionally; `KeyError` is loud), the non-dict event check (a non-dict event is a corrupt
artifact), the empty-`frames` guard in `assign_runway` and the duplicate `max_tracks` guard
in `visualize` (both contracts documented, both still crash, and argparse rejects the latter
first). `METHODOLOGY` being shared by reference is real aliasing but nothing mutates it —
adding a deepcopy would re-add the defensive code this pass removed.

*Symmetry.* `evaluationReport.ts` also stopped re-deriving the observed-availability
arithmetic. `harvest/observed.py` computes `event_unavailable` and `event_estimated_rate`
FROM `denominator` and `estimated` in the same expression, so those are identities, not
invariants — and re-checking them cost a silent whole-report rejection. Both sides now check
only the denominator LABEL, which is the one field that says which population the rate
describes.

Out-of-scope findings from this pass are recorded in `docs/code-health-followups.md`
(duplicate `_iso` in harvest, `summary_row`'s explicit JSON nulls, the stale v2 fixture in
the comparison-CZML tests, and the 12 pre-existing `test_ts_pipeline` failures).

Deferred pending review: the dead `max(seed_scale, floor)` in `fit._straight_final_suffix`
(the "adaptive" residual limit is a constant 31.9 m for every track), and replacing the
O(n²) exact Theil–Sen seed in `_median_line`.

Verified: 908 passed across the Python suites (13 failures all pre-existing — 12 in
`test_ts_pipeline`/`test_download_landings`, plus the known numpy 2.x
`test_fixed_time_objective_weights_control_effort_at_one`); frontend `tsc --noEmit`
clean and 508 vitest tests passing.

### 2026-08-17 — ADS-B altitude outliers filtered in the view, not in the tracks

**Symptom.** Observed trajectories rendered with needle-shaped vertical peaks: single
samples reporting an altitude nowhere near their neighbours. Measured extremes across the
five harvested airports are 20 147 m between neighbours at 724 m and 35 189 m at 556 m.

**What was rejected.** A stop-gap `fix_altitude_spikes.py` edited `tracks/*.json` in place.
That breaks three contracts at once — `arrivals/manifest.json` pins every source track by
SHA-256 (the loader refuses a changed file), `--reclassify-existing` re-derives assignment
from those exact samples, and `source_integrity.retained_rows` counts them. It also missed
the point: `tracks/` is the sensor reconstruction, and a repair is a property of the view.

**What shipped.** `trajectory_data_process/harvest/altitude_filter.py`, applied where a
stored track becomes a derived view and nowhere else:

- `store.read_track_view` → observed CZML (`czml.observed_czml_flights`, which also feeds
  the backend trajectory sampler) and evaluation records (`observed.write_observed_records`);
- `arrivals.write_arrival_records` / `load_arrival_flights` → all model input (ts training,
  `flight_scenarios`, `batch_benchmark`). Both hash the SOURCE bytes first and filter after,
  so the roster stays a statement about what the receiver recorded.

`store.iter_records` deliberately stays raw — reconstruction and reclassification must see
what was stored.

**Detection.** Deviation from the median of the ±2-sample window exceeding BOTH 100 m AND
`25 m/s × min(adjacent gap)`. Both halves earned their place on the data:

- a chord/jump test (the stop-gap's approach) attributes one bad sample to three, because
  the outlier's two neighbours have a chord running through it — 363 runs of exactly three
  where the truth was 363 isolated samples;
- the 100 m floor sits at 2× the largest residual genuine flight produces. Over 20 851 436
  assigned samples the residual is < 25 m for 20 847 051, 3 625 fall in [25, 50) (the 25 ft
  and 100 ft reporting lattices plus real motion), and only 189 exceed 50 m;
- the rate bound spares 10 real descents that stepped 107–160 m across 9–14 s reception
  gaps, which a bare deviation threshold "repairs" into a lie.

Incidence: **561 samples in 451 of 44 622 assigned tracks (0.0027 %)**; 421 sat inside a
model arrival slice. 479 isolated, longest run six.

**Repair replaces the altitude and never drops the sample.** `landing_sample_index`, the
arrival slice bounds, the threshold event's `source_sample_range` and the
`reported_ground_speeds_m_s` parallel array all index that array; deleting a row silently
renumbers every one of them. Replacement is a linear interpolation in time between the
nearest retained samples; at a track edge it holds the nearest retained altitude
(`held` vs `interpolated`, both labelled in the report).

**Stated, never silent.** `arrivals/manifest.json` and `approach/summary.json` each carry an
`altitude_filter` block (policy + repaired counts), arrival records carry a per-flight
`altitude_outliers`, and `RenderedObserved` carries the render's totals.

**Tooling.** `fix_altitude_spikes.py` deleted; `python -m trajectory_data_process.altitude_outliers`
replaces it as a read-only audit (`--report-json` gives the full per-track trail) plus
`--rerender-czml`, which republishes `public/data/<ICAO>/trajectories.czml` through the
pipeline's own renderer.

**Artifacts.** All five `trajectories.czml` republished. Batch comparison CZMLs resolve
their observed reference by entity id inside that canonical file, so they follow without a
rebuild. Training data needs no rebuild either — `load_arrival_flights` filters on the way
out — so the next dataset build is already clean; `--evaluate-only` is what refreshes the
roster counts and the evaluation records.

**Known gap.** Stored `observed_threshold_event`s were fitted from raw samples during
assignment, before this filter existed. 17 outliers (KRDU 15, KSTL 2) land inside an
event's source range; the audit lists them and `--reclassify-existing` is what re-derives
them.

### 2026-08-12 — Fail-closed pipeline cleaner

- `clean_pipeline_data.py` now requires an explicit airport scope and constructs its
  deletion plan from producer-owned artifact names instead of recursively treating
  `4dTrajectory/outputs` as disposable.
- Downloaded tracks, checkpoints/history, `test_release.json`, formal experiments,
  pooled roots, final-test and ambiguous predictions, parked/manual/unknown outputs,
  tracked/static data, archives, and mixed experiment comparison publications are
  protected.
- Only standalone predictions whose readable metadata explicitly says `split: "val"`
  are eligible. Comparison cleanup requires a readable registry that accounts for all
  content. The canonical observed filename is matched exactly, not by prefix.
- Destructive execution validates the complete plan and stages every selected file on
  the same filesystem; a staging failure rolls the move set back. Safety tests cover the
  allowlist, airport isolation, protected research state, mixed comparison output, exact
  CZML ownership, required scope, dry-run behavior, and rollback.

### 2026-07-23 — Fitted threshold kinematics; preparation/optimization runner split

- Fitted-ADS-B targets now derive `V/psi/gamma` from the same established final-approach
  fit as their threshold position. The along-track rate is fitted over that established
  segment only; the spatial tangent supplies heading and glide angle, so rollout
  or parked samples can no longer create a threshold target with `V=0`. The constant-rate
  helper is the single replacement seam for a future deceleration model.
- The former combined runner was deleted. `prepare_scenario_inputs.py` rebuilds
  arrivals/observed products and writes the two distinct scenario datasets;
  `run_scenario_optimization.py` consumes those datasets and owns optimization,
  evaluation, and comparison-CZML publication. Run preparation first, then optimization.
- `clean_pipeline_data.py` now removes preparation-derived `arrivals/` and `approach/`
  by default while preserving downloaded `tracks/`; `--include-downloads` only expands
  the deletion boundary to measured source tracks.
- Review hardening made datum provenance explicit on fitted results, rejects invalid
  `states_ref` ranges and duplicate source identities, and verifies reference identity
  plus SHA-256 before cache/batch reuse. `--skip-optimize` now validates the complete
  summary/eval/states/reference roster rather than treating `summary.json` as a marker.
- Comparison publication now writes immutable generation-suffixed CZML/report artifacts
  and atomically commits their index last; failures preserve the previous generation.
  The frontend follows the report named by that index, strictly rejects legacy observed
  and comparison manifests, never falls back to embedded references or fixed-name
  reports, and restores canonical entity styles on comparison exit.

### 2026-07-21 — final_approach + evaluation review fixes: observed-aware reporting surfaces, one deviation definition

Code review of `final_approach/` + `evaluation/` (no correctness bugs in the geometry or the
fit statistics; all findings were reporting/duplication/doc-drift). Fixes:

- **The human-facing surfaces caught up with the observed-subject work.** The measurement side
  (`arrival.py`, the report's `observed` block) was done, but `evaluation/visualize.py` and the
  `python -m evaluation` console summary still rendered pre-subject reports: a solve-rate card/
  line (1.0 by construction on observed data — the exact number the subject dispatch exists to
  stop reporting) and no established rate or marginal count anywhere. Both now suppress the
  solve rate for pure observed batches and report `established N/M` + marginal. The deviation
  charts also plotted `solvedRows`, which for observed batches includes not-established rows
  with NO deviation fields — `Math.max(undefined, 0.01)` → NaN → silent bar gaps; charts now
  draw `measuredRows` only. Verdict table gains established/marginal columns + ±95 % bounds on
  the deviations when observed rows exist; not-established rows are styled grey ("no arrival to
  judge"), not red ("judged and failed").
- **`EstablishedCriteria` became reachable**: `evaluate_batch(..., criteria=)` and CLI flags
  (`--fit-window-m/--max-cross-track-m/--glidepath-range-deg/--max-vertical-rms-m`), defined
  ONCE in the new `evaluation/cli.py` and shared by `__main__` and `visualize` so the JSON and
  HTML reports cannot be produced with silently different knobs (the gate flags were previously
  duplicated between the two entry points).
- **One final-state deviation definition.** `metrics.final_state_deviation` + its
  `FinalStateDeviation` dataclass duplicated `arrival._final_state` line for line (no external
  consumers, verified). The single definition now lives in `arrival.final_state_deviation`
  returning `ArrivalDeviation`; `FinalStateDeviation` deleted.
- **`_is_marginal` lateral fold fixed**: `abs(lateral − margin)` folded a signed CI containing
  the centreline past 0, misreading "CI straddles the gate" as a solid verdict whenever
  1.96σ > gate + offset. Lower bound is now `max(0, lateral − margin)`. Unreachable at real
  σ ≈ 1–2 m (needs σ > ~55 m) — fixed for correctness, with a regression test.
- **Dead code removed**: `frame.track_course_deg` / `heading_difference_deg` had no consumer
  anywhere (the promised harvest heading pre-filter never materialized — the arg-min design made
  it unnecessary). `fit_final_segment` now validates `min_samples >= 3` / `min_span_m > 0` at
  the boundary (previously a ZeroDivisionError deep in `_fit_line`). `visualize.RESAMPLE_N`
  deduped into `reference.N_RESAMPLE`.
- Docs: `evaluation` README/`__init__` (which still described the pre-subject package, exported
  nothing from `arrival`) updated; `Assignment.scores` docstring no longer claims rejections
  carry scores; stale CLAUDE.md open item ("observed evaluation designed but NOT built") removed.

Suites: final_approach + evaluation 88 pass, trajectory_data_process 93 pass.

Found while trying to answer "what do the 8260.58D gates score REAL ADS-B arrivals at?" —
the observed baseline the optimizer and the learned predictor are implicitly measured
against, which had never been computed. The naive answer was 1.8 % pass (18/996 KRDU), i.e.
completed, safe airline landings graded as failures. Three independent bugs, all upstream of
`evaluation/`, which is unchanged by this work.

**The measurement that separated them.** Fitting each flight's OWN established final-approach
line (position + cross-track vs along-track, extrapolated to the threshold) instead of reading
`states[-1]`: the fitted glidepath came out 3.02–3.13° at all five airports — textbook — while
the vertical intercept was 20–30 m low. A fleet flying a perfect glidepath to a uniform 25 m
error is not a fleet error; it is a reference error. Lateral was already 3–10 m median, so the
two axes were telling opposite stories and had to be chased separately.

**Bug ① — observed altitude is ellipsoidal, targets are MSL.** OpenSky `geoaltitude` is height
above the WGS84 ellipsoid (HAE); runway thresholds, CIFP altitudes and the gates are MSL. The
gap is the geoid undulation N ≈ −25 to −33 m over the US. Confirmed independently: KRDU's
lowest observed sample is 99.1 m against a predicted field elevation + N = 132.59 − 33.53 =
99.06 m (4 cm). Fixed in a new `flight_scenarios/datum.py` (EGM96 via pyproj), applied at the
data→modeling seam.

*Not* in the harvest, deliberately: the harvest feeds two consumers with opposite requirements
— CZML/Cesium positions are documented as metres above the WGS84 ellipsoid
(`aeroviz-4d/src/types/czml.d.ts`) and are correct as recorded. Converting at the source would
have fixed modeling and broken the viewer by the same ~33 m. The harvest stays a faithful
record of what the sensor said; the datum choice is made on the way in.

The conversion reached THREE separate ingest paths (`load_observed_flights`, `build_scenario`,
`ts_transformer/dataset.py` — the last reads bare waypoints, so it cannot self-protect). It is
keyed on `altitude_source` and therefore idempotent, and unknown/missing sources raise rather
than defaulting. Seam tests pin all three.

PROJ trap worth knowing: without the EGM96 grid and with network off, pyproj silently returns
a "ballpark" no-op vertical transform — a correction that looks applied and does nothing.
`_geoid_transformer()` probes a known undulation and raises instead.

**Bug ② — `runway_thresholds.json` stored pavement ends, not landing thresholds.**
`build_runway_config.py` read `le_latitude_deg`/`le_elevation_ft` and ignored
`le_displaced_threshold_ft` entirely. KSJC 30L/30R are displaced 775 m; on a 3° glidepath that
is a 40.6 m altitude error, and it moved the OPTIMIZER TARGET, not just the gates. Six
thresholds moved (KSJC ×4, KSTL 12R 143 m, KMSY 29 93 m); the other 20 are unchanged. Fixed in
the generator, not the JSON. Schema bumped to `runway-thresholds-v2`; thresholds now carry
`displaced_threshold_m`.

This is why KSJC looked HEALTHIEST before the fix (+9.7 m vs everyone else's −25 m): its two
bugs had opposite signs and nearly cancelled (+40.6 − 32.0 = +8.6 predicted). Chasing the
"anomaly" is what found bug ②.

**Bug ③ — parallel runways captured the same landing twice.** `classify_landing_flights` was
called once per threshold with no cross-threshold arbitration, and `RUNWAY_THRESHOLD_RADIUS_M`
is 1000 m while parallel runways sit 250–400 m apart on an identical heading — so both the
geometry and heading tests accepted either one. Measured: 169 of KSJC 30L's 200 flights were
also in 30R's file; KSJC 12L∩12R 63; KSTL 30L∩30R 32. KRDU/KSMF/KMSY are unaffected (their
parallels exceed the capture radius). It surfaced downstream as an observed lateral error
whose MEDIAN was the parallel separation (KSTL 30L 397 m, KSJC 30R 234 m).

Fixed with a `sibling_thresholds` arbitration restricted to same-direction runways — the
opposite end of the same runway must be excluded, since a full rollout stops on top of it.
The discriminator is the median lateral offset from the extended centreline, NOT distance to
the threshold point: a first attempt using threshold distance failed to separate at all (kept
763 m vs dropped 791 m) because a displaced threshold sits 775 m past where ADS-B coverage
ends, so every track is equidistant from it. On the centreline metric the split is clean
(kept 17.5 m vs dropped 232.8 m at KSJC; 35.9 vs 382.6 at KSTL).

**Effect, end to end** (established-approach threshold crossing, records regenerated through
the real code path, not a reimplementation):

| airport | vertical median before → after | vertical gate before → after |
|---|---|---|
| KRDU | −29.2 → **+4.3 m** | 1 % → **44 %** |
| KMSY | −19.6 → **+5.7 m** | 0 % → **50 %** |
| KSTL | −26.9 → **+4.9 m** | 0 % → **31 %** |
| KSJC | +9.9 → **+0.3 m** | 24 % → **51 %** |
| KSMF | −27.7 → **+2.7 m** | 0 % → **65 %** |

All five now sit at +0.3 to +5.7 m — a small POSITIVE bias, which is operationally right
(crossing at or slightly above TCH is correct; low is dangerous). Lateral was already correct
and is unchanged at 3–10 m median.

**What this makes stale.** Everything derived from observed tracks: `flight_scenarios/outputs`,
all `4dTrajectory/outputs/<ICAO>/{asdb,runway,runway_cons}`, all
`public/data/airports/*/comparison`, and the `ts_*` training data + checkpoints. Bug ③
additionally requires re-harvesting KSJC and KSTL (offline de-duplication is possible but
would cost KSJC 42 % of its flights, leaving 12L at 12 and 30R at 39).

Note for the ts_transformer re-run: the `u` channel shifts uniformly by +33.5 m at KRDU.
Accuracy metrics (ADE/FDE, deviation vs reference) are computed against a reference in the
same frame and should be nearly unchanged, but the GATE verdicts were biased — the recorded
"gate-pass counts 0–4 of 152" was a ±3 m window scored against data offset by 33 m, so that
conclusion needs re-deriving rather than quoting.

Tests: 692 pass (557 modeling+backend, 135 aeroviz-4d/python), both suites exit 0. The
"one known pre-existing failure" in `run_all_tests.sh`'s header
(`test_fixed_time_objective_weights_control_effort_at_one`, numpy scalar conversion) did NOT
reproduce — that note and the matching CLAUDE.md Open Item look stale.

**Post-review hardening (same day).** A recall-mode review of the three fixes surfaced and
closed: ① `resolve_runway_threshold` still returned pavement ends — the `--runway` download
path would have named a threshold up to 775 m from the config's and drifted
`landing_time_utc`/`flight_key` between harvest paths; landing-threshold interpolation is now
single-sourced in `acquisition/runways.py` (`landing_thresholds_from_row`), generator output
byte-identical, plus a loud `ValueError` when a displaced end has no usable length.
② `_wins_against_parallel_runways` crashed on a heading-less threshold
(`math.radians(None)`) and its `inf <= inf` tie silently re-admitted double-assignment when
no sample fell in the centreline window — now: no competitors → win (no offset computed,
also removing a dead full-track scan), unestablished centreline vs a competitor → lose from
both. Also: `_heading_diff` reused instead of an inlined twin; true `statistics.median`.
③ `datum.py`: the ballpark probe was NaN-transparent (`abs(nan−33.53) > 1.0` is False) —
inverted to not-within-tolerance; an operator's explicit `PROJ_NETWORK` is no longer
overridden; `waypoints_to_msl` transforms the altitudes directly (EGM96 N is
height-independent — verified: +1000 m in → exactly +1000 m out), removing the
negate-and-subtract dance. ④ `FlightScenario.source` now records `altitude_source`, so
saved scenario files carry datum provenance (pre-fix HAE-era files lack the key).
⑤ ts dataset conversion moved after the cheap skip checks; test fixtures now import
`METRES_PER_DEG_LAT`/`metres_per_deg_lon`/`FT_M` from geokit instead of retired literals.
⑥ The symmetric OUT seam, closed after review discussion: modeling records (now MSL)
were packed straight into Cesium ellipsoidal `cartographicDegrees`, so after the batch
re-run every opt-/sim-/pred- entity would have drawn ~33.5 m above the white HAE
reference. `build_scenario_comparison_czml._states_to_waypoints` (the single choke point
all record-derived entities share; the reference bypasses it) now converts MSL→HAE via a
new `aeroviz-4d/python/vertical_datum.py` — a mirror of `flight_scenarios/datum.py`
(same KRDU −33.53 pin, ballpark probe, PROJ_NETWORK respect) per the `flight_identity.py`
precedent. Records are assumed MSL rather than tagged: all pre-datum-fix artifacts are
discarded wholesale (user decision), never fed back in.

**New operational scripts (same day).** `run_ts_pipeline.py` — the ts_transformer sibling
of the scenario optimization runner: per airport runs the 2×2 grid (iTransformer/PatchTST ×
window/full) as train → predict(test split) → evaluation report/HTML → comparison-CZML
publish (categories `ts_{itr|ptst}_{mode}`, matching the published naming); dataset build
+ flight_key split happen inside train and travel in the checkpoint. `clean_pipeline_data.py`
— wipes every generated artifact of both chains (scenarios, 4dTrajectory outputs incl.
ts dirs, frontend comparison + observed-layer CZML) with plan-print + confirm/`--yes`;
raw OpenSky downloads and `_`-parked research dirs are kept unless `--include-downloads` /
`--include-parked`; static airport layers and `data/archive` are never touched.

### 2026-07-20 — prediction overlay: anchor-time alignment + the lookback window is drawn

Two defects in how a ts_transformer forecast reached the globe, both invisible in the record
files (which were correct) and both living in `build_scenario_comparison_czml.py`.

**The forecast was drawn a whole lookback early.** A prediction record rebases its own time so
`t = 0` is the ANCHOR — the last observed sample the model was shown, `seq_len - 1` samples
into the approach. The reference is copied out of the airport's `trajectories.czml` and still
starts at `t = 0` = the START of the track. The builder wrote the record's times straight
through as CZML offsets, so the two shared a clock they did not share a zero on. Measured on
KRDU 05L / `AAL542_…`: `pred[0]` is bit-identical to the reference's `t = 118 s` sample, and
was plotted at `t = 0` — **12.0 km** from where the reference was at that instant. Every
prediction-schema entity is now shifted by `source.anchorTimeS`. Optimizer records were never
affected: their `t = 0` already is the scenario start.

**The lookback was never rendered.** `export.py` has always written `observed_states` as the
WHOLE observed track (negative `t` before the anchor) explicitly so a viewer could show the
input the model was conditioned on — and no viewer ever read it. The purple line simply began
in mid-air at the anchor. It now emits a second entity per group, `look-{group}`: the `t ≤ 0`
slice, same hue as the forecast at alpha 85 (`LOOKBACK_COLOR`) vs 225, frontend kind
`lookback` with its own "Predictor input" legend row. The anchor sample belongs to both halves
— it is literally the same state object in the record — so the faded half meets the forecast
exactly, not merely closely (asserted, not eyeballed). `observed_states` moved into
`_PREDICTION_SCHEMA`: a record that cannot be drawn completely now fails loudly.

Supporting cleanups: the lookback retraces samples the reference already covers, so it renders
path-only (a model/point there would draw a second aircraft on top of the reference's for the
whole input window); the entity-id→kind prefix table is now one list shared by `kindOfEntityId`
and `isComparisonEntity`, which had drifted — the picker's list was missing `pred-`, so
prediction tracks silently could not be hovered for their callsign; per-kind alpha
(`COMPARISON_KIND_ALPHA`) drives the legend swatch too, since "Predicted" and "Predictor input"
share a colour and a solid swatch made the two rows identical.

Rebuilt all four KRDU ts categories (`ts_itr_full` / `ts_itr_window` / `ts_ptst_full` /
`ts_ptst_window`, 152 groups each). Verified structurally across all 608 lookback entities:
every one starts at the reference's own `t = 0`, carries exactly `seq_len = 60` samples, ends
where its forecast begins, and never outruns its reference. 138 CZML-package tests, 460
frontend tests, tsc clean, `npm run build` clean. NOT re-checked in-browser.

### 2026-07-20 — B3: transport-consistent velocity channels + physical-velocity fit; third ts training generation

The findings doc's B3 bundle (`docs/findings_and_open_items_2026-07-20.md`), executed. The
one deviation from B3's literal scope was forced by measurement — see the second bullet.

**B3.1 — the position↔velocity inconsistency (A7) closed, at BOTH seams.**

- New numeric single source `geokit.wgs84_curvature_radii(lat_deg) -> (R_M, R_N)` (exact
  WGS84 closed forms). `flyability._transport_rates` now imports it (was an inline copy);
  the casadi geodetic RHS keeps its symbolic twin with a MUST-match mirror comment (a CasADi
  expression cannot call a float function). Pinned in geokit's tests at the equator/pole/45°
  landmarks.
- `ts_transformer/channels.py`: velocity channels are now the exact **chart derivatives** of
  the position channels — the physical velocity mapped through the full-transport Jacobian
  (`ndot = V_north·a/(R_M+h)`, `edot = V_east·a·cos lat₀/((R_N+h)·cos lat)`, `udot`
  unchanged); `states_from_channels` inverts exactly. Renamed `ve/vn/vu → edot/ndot/udot`
  deliberately: the channel tuple is serialised into every checkpoint and `load_checkpoint`
  refuses a mismatch, so every pre-change checkpoint fails loudly instead of silently
  mis-scaling velocities. New tests pin the factor closed form and the integration identity
  (a sequence generated by the geodetic kinematics integrates its velocity channels back
  into its position channels).
- **The fix as literally scoped in B3.1 would have made the measured inconsistency WORSE** —
  found by measuring, not reasoning. On all 995 KRDU arrivals (median whole-track drift of
  ∫v dt against the position channels): original east 3.5 / north 2.7 m/min; channels fixed
  alone east 3.4 / **north 8.6** m/min. Cause: `flight_scenarios._velocity_lsq` fitted
  velocity through the flat chart scales (`a`, `a·cos lat`), so its `V_north` overstated the
  physical value by `a/R_M` (+0.33% at 36°) — the old *channel* code cancelled that bias by
  accident, and A7's "cos ratio + h/R + R_M/R_N" attribution was really describing the FIT.
  Per the fix-upstream convention, `_velocity_lsq` now projects through the true tangent
  scales at the window anchor (`(R_M+h)`, `(R_N+h)·cos lat`), making its output the physical
  ENU velocity every consumer already assumed (the geodetic RHS, flyability's inversion, the
  ts chart). Final measurement, both seams fixed: east **2.4** / north **2.7** / up 0.45
  m/min — unbiased LSQ smoothing, no systematic left.
- Blast radius of the fit change: every fitted `V/psi/gamma` moves ≤ 0.33% / ≤ 0.1°;
  positions are untouched, so all position-based metrics and gates are unaffected. The
  2026-07-20 optimizer batch artifacts predate it (same situation as the geokit constant
  alignment below: a re-run would move initial-state V by ~0.2 m/s, far below data noise).
  Fixture constants in `test_start_state.py` / `test_scenario_optimization.py` were rebuilt
  on the tangent scales.

**B3.2 — `dt = 1 s` considered and declined** (README "Sizing"): same coverage needs
L=120/H=600 (~2× training cost) for almost no information — the source reports at ≤ 1 Hz
with ragged gaps and the velocity channels come from a 15 s window fit. `--dt` stays a knob.

**B3.3 — the lead-time table is restored on the raw-tensor accounting** (A8). The README
now carries BOTH accountings with their n: record accounting (one forecast per flight,
threshold-truncated — n falls with lead; 600 s survives for ≤ 1 flight) and raw-tensor
accounting from `history.json` `metrics.test.by_horizon` (every test window, sees past
truncation — 600 s restored with n = 893). On it, **iTransformer leads PatchTST at 600 s in
all three training generations** (6135/7142, 5407/6962, 5438/7384 m) at margins of
1.16–1.36× — direction consistent, each margin under the provisional band.

**Retrain (third generation).** Same recipe (ep=120, patience=15, lr=5e-4, seed=1337) and
the same `flight_key` split (702/141/152); all four cells trained, predicted (test split),
evaluated, flyability-reported. Pre-B3 artifacts moved to
`4dTrajectory/outputs/KRDU/_pre_b3_transport/`. Headline (152 test flights): iTransformer
full lateral 868/2750 m mean/p95, chained 1302/6576; PatchTST full 2016/5598, chained
3433/8993 — **full beats chained on threshold lateral in every generation** (1.5–1.7× mean).
Gate passes are 0/152 in all four cells this generation (pre-B3: 4, 1, 0, 0) — across
generations the count ranges 0–2.6% and the README now states only the stable conclusion
(a forecast is not a certifiable approach; a borderline pass is jitter). Flyability floor
(observed tracks) stays 63.2%; iTransformer window is again the only cell above it (73.7%).
The instance-norm ablation (`_ablation_norm/`) was NOT re-run — measured pre-B3, margins
1.2–2.7× vs a ≤ 0.3% channel rescale, structural argument unchanged; dated note added.

Suites: run_all_tests.sh 520 + 135 passed (the one known pre-existing collocation failure),
geokit 29. The ts suite is 56 tests (2 new transport pins).

### 2026-07-20 — full batch re-run (15/15 fresh), geokit per-degree constant aligned to the optimizer

**Batch.** The former combined runner with `--jobs 6`, 5 airports × 3 categories, 10,449 solves,
≈4 h 39 m wall clock (one harness-side background-task reap mid-run; resumed detached with
`setsid nohup`, `overall_fail=0`). All artifacts now post-date every 2026-07 fix (arrival
truncation, altitude floor/rollout guard, HS flip, identity unification) — the standing
"all batches are STALE" open item is closed. ts predictions were regenerated the same day,
**test split only** (152 flights per the reproducible flight_key split), for all four
checkpoints.

Identity contract verified on the fresh artifacts, all green: record stems are full
flight_keys; summary rows all carry `landing_time_utc`; **reference hit rate 100% in all 19
comparison categories** (15 optimizer + KRDU's 4 ts_pred — the pre-refactor
duplicate-callsign dropout was 22%); zero duplicate entity ids in sampled CZMLs; every
airport's categories.json intact (KRDU keeps 7 categories).

Headline solve/gate rates (success among solved): runway_cons is the cleanest everywhere
(93–99%), asdb the hardest (76–89%). Finding worth keeping: **KRDU RW32 is systematically
hard and NOT the old truncation artifact** — runway_cons RW32 79 offTarget + 59 failed of
198 (all other runways ≤9), and asdb RW32 fails 197/198 (IPOPT infeasible). Likely
procedure-specific (RNP-AR H05LZ; per-leg RNP still not extracted). KSTL runway_cons has a
milder cluster (12R/30R/30L/24; repeated single-IAF `PAULY` infeasibility).

**geokit alignment.** `METRES_PER_DEG_LAT` was the hand-rounded `111_320.0`; the
optimizer's NE frame (`approach_constraints.frame` and the NLP's metric-position
normalization) derives `WGS84_A·DEG2RAD = 111319.4908…` — a 4.6 ppm seam (~0.11 m at the
25 km ring) between the two frame families. Now defined as `WGS84_A * (π/180)` in
`geokit.constants` — bit-identical to the optimizer's product (IEEE commutativity),
`metres_per_deg_lon` stays pure `·cos(lat)`. Frontend `geoConstants.json` regenerated.
Applied AFTER the batch finished so all 15 cells share one constant; ts checkpoints are
unaffected in practice (inputs move ≤0.11 m at the ring edge vs km-scale model error — no
retrain). Also corrected the `channels.py` projection docstring: the flat chart deviates
from a true tangent-plane ENU by up to ~40 m at the ring edge (`e·n·tanφ/R` cross term) and
`u = Δalt` ignores the ~49 m curvature drop by design — the old "well under a metre" claim
was true only of the quantities the metrics actually measure (same-chart comparisons;
~0.2% local scale distortion at the ring edge, → 0 at the threshold).

### 2026-07-20 — flight identity unified end-to-end: entity ids = flight_key, positional `_N` re-uniquing deleted

The last identity holdouts (CZML entity ids, the comparison reference lookup, the FlightTable
optimizer join) still ran on bare callsigns + positional `_2/_3` suffixes. Diagnosis on the real
KRDU harvest:

- **Per-runway landings/arrivals files held massive duplicate ids** (128 duplicates across five
  runways; `N993FG` ×10 on RW32). Root cause: `collect_landings` harvests in CHUNKS and
  `classify_landing_flights` restarted its `_unique_id` numbering per chunk, while the
  cross-chunk merge de-duplicated by `(icao24, landing_time_utc)` without re-uniquing ids. The
  per-runway CZMLs inherited them — Cesium merges same-id packets, so two namesake flights
  rendered as ONE garbled entity (both tracks' samples interleaved from t=0), and
  `flightSummaries`/React row keys collided.
- **Cross-view id aliasing**: `merge_landing_flights` re-uniqued ids positionally for the
  combined file, so the same string (`SWA1692_2`) named DIFFERENT physical flights in the
  per-runway vs combined views, and the same flight got different `flight_key` stems from the
  two record writers (optimizer eats combined, ts eats per-runway) — breaking identity.py's
  "same stem" promise and making the comparison builder's callsign reference-lookup resolve to
  whichever namesake came first (the "wrong white line" open item).
- **FlightTable's optimizer join was callsign-keyed** (`byFlightId` from `group.flightId`), so
  namesakes swapped each other's V/mass/failed/offTarget facts.

Fix — one identity everywhere, display strictly separated:

- `generate_czml.build_czml`: entity id = `flight_key(flight)`, `name` = callsign; RAISES on a
  duplicate identity (silent Cesium merge → loud input error). Both `trajectories.czml`
  producers go through it.
- `_unique_id` deleted from the landing path (`classify_landing_flights`,
  `merge_landing_flights`); kept ONLY in `trajectories_to_czml_input` (the plain download path
  has no runway/landing time — the suffixed id is its one discriminator).
- `aeroviz-4d/python/flight_identity.py`: deliberate MIRROR of
  `flight_scenarios.identity.flight_key` (frontend tooling can't import the modeling tree);
  both copies pinned to `EJA969_05R_ad7f04_20260618T213736Z` in their own suites.
- Comparison builder: reference lookup by `group` (= flight_key = the new entity id) in batch
  mode and by `flight_key(source)` in single mode; `scenario_initial_map` keyed by flight_key
  (was `(id, runway)` — namesakes shared one V/mass); `_group_key`'s file-less fallback
  reconstructs the identity from the row (rows now carry `landing_time_utc` via `summary_row`
  — it IS part of the identity and was the one missing field).
- Frontend: `useFlightOptimizerData` keys by `group.group` (renamed `byFlightKey`);
  FlightTable/approach view display the callsign (`name`) while ids stay the
  selection/join/cache identity; `ObservedFlightSummary` gained `callsign`.

Verified: 57 aeroviz-4d python tests, 59 trajectory_data_process, 84 optimization+ts, 453
vitest, tsc + vite build — all green. New pins: namesake entity ids distinct + duplicate
identity raises (generate_czml); each comparison group copies ITS OWN namesake's reference
track; FlightTable keeps namesake optimizer facts apart. Artifacts regenerated after this
change (arrivals/CZML rebuild + full batch re-run); ts checkpoints are UNAFFECTED (per-runway
arrivals `id` fields — and therefore split keys — are byte-identical; only the combined view
changed).

### 2026-07-19 — flyability check, instance-norm re-ablation on real data, predictions in the frontend

Three things asked for together: the post-hoc flyability check (route 1 of the README's four),
re-ablating `--instance-norm` on real data, and getting prediction results to render in the
frontend alongside the optimizer's.

**Post-hoc flyability (`4dTrajectory/ts_transformer/flyability.py`, 16 tests).** The
load-factor point-mass model inverts in closed form — `A = psi_dot V cos(gamma)/g = n sin(mu)`,
`B = gamma_dot V/g + cos(gamma) = n cos(mu)`, so `n = hypot(A,B)`, `mu = atan2(A,B)`; `n` fixes
`Cl`, `Cl` fixes drag, and `T = m(V_dot + g sin(gamma)) + D` closes it. One pass, no solver, no
casadi, so it lives in the torch env. Earth-frame transport terms are subtracted first.
`predict` writes `flyability_report.json` and prints a summary line.

- **The calibration matters more than the check.** First run against REAL flown tracks scored
  **0/149 fully flyable** — those are trajectories real aircraft flew, so the check was wrong.
  Cause: `thrust_negative`. Median required thrust on a real arrival is **0.43 kN** (idle), and
  a negative requirement means the aircraft needed more drag than a clean airframe has —
  speedbrake, flaps, gear. Every approach does this; one clean-configuration polar cannot
  represent it. Reclassified as SOFT (reported, not counted unflyable), and the report leads
  with the delta against the observed tracks measured by identical code, because both sides
  carry the same polar bias. **The observed baseline is the floor, not 100%.**
- `Cl_max` from `aero_params_for_aircraft` (2.7 for an A320), NOT `LoadFactorSimulator`'s
  hardcoded 1.5 — they disagree by 80% and `aero_params.py` is the documented source of truth.

**Instance-norm re-ablation on real KRDU data — the synthetic OFF default holds.** All 8 cells
(2 models × 2 horizon modes × on/off), same hyperparameters, each graded on its own
checkpoint's test split; artifacts in `4dTrajectory/outputs/KRDU/_ablation_norm/`, roll-up in
`ablation_results.json`.

- **OFF wins 19 of 20 accuracy comparisons, one tie**: every cell on val loss, FDE, ADE p95 and
  lateral p95; 3 of 4 on mean ADE with PatchTST window a dead heat (2580 vs 2571 m — the cell
  where OFF wins the other four metrics). A sweep that lopsided is not run-to-run variance,
  which was the open question: individual gaps here are much smaller than on synthetic data
  (1.2–2.7× vs 2.4–6.5×), and an earlier partial pass on ONE metric had shown an apparent
  reversal in that same PatchTST window cell. It did not survive consistent scoring.
- **The signature is lateral p95**: all four instance-norm-ON cells land at 14.28–14.50 km —
  near-constant across both architectures and both horizon modes. That is a model that cannot
  place the endpoint at all; strip the absolute level and the prediction ends at a distance set
  by the frame, not by the flight. OFF spans 2.6–8.5 km, i.e. it varies with the flight.
- **Flyability moves the OPPOSITE way** (ON better in 3 of 4 cells; PatchTST window 89.3% vs
  29.6%, i.e. the configuration 2.2× worse at the threshold looks three times more flyable).
  Instance norm is not flying better, it is predicting blander paths — easy to fly, far from
  the truth. A straight line is perfectly flyable and completely wrong. Recorded in the README
  because it is the standing argument for never reading flyability alone. (iTransformer full is
  the exception at 46.1% off vs 34.9% on.)

**Headline KRDU runs retrained, and two published conclusions did not survive it.** The 4
published checkpoints carried a split that `hash(flight_key)` reproduces for only 552/995
flights — they predate the per-flight-hash split fix, exactly the "retrain before comparing"
note in the entry below. The partition itself is clean (train/val/test verified disjoint), so
the published numbers were not wrong, just on a split no current run can reproduce and not
comparable with the ablation. Retrained on the current keying (702/141/152):

- **"The compounding cost lands in the tail, not the mean" — withdrawn.** It rested on lateral
  mean 1.5× vs p95 2.2×; on the new split the two ratios are equal to within noise
  (iTransformer 1.55× vs 1.61×). A tail effect survives in FDE (PatchTST p95 1.78× against a
  1.36× mean) but that is a different claim than the one written.
- **"Zero gate passes in all four runs" — withdrawn.** iTransformer now passes 3/152 (full)
  and 1/152 (chained). The substance holds (2% is not a usable predictor) but "zero, always"
  was a property of that split.
- **Survived:** one-pass full beats chained window on whole-approach lateral error for both
  models on both splits (1.5–1.6× mean, 1.5–2.1× p95), and the short-lead/long-lead crossover
  between PatchTST and iTransformer (now cleanest within the full-mode pair: 184 vs 571 m at
  10 s, reversing by 300 s).

Standing lesson recorded in the README and CLAUDE.md: **treat any single-split margin under
~1.5× as provisional** — that is the size of effect a split change moved here.

**Flyability shipped with a wrong assumption, caught by its own guard.** The check graded a
whole batch against ONE envelope, documented as safe because "every harvested arrival is type
UNK and resolves to the single `--aircraft-type` fallback". A boundary assertion added to state
that assumption fired on the first real batch: `_resolve_aircraft` falls through to an
**`icao24` → OpenAP lookup** that recovers the real airframe — 20 distinct types across 400 KRDU
arrivals, A320 only 224 of them. So ~44% of every batch was being judged by an A320's `Cl_max`
and max thrust, invisibly, because the report never named the airframe it used.
`report_for_records` now takes one `Aircraft` per flight, builds one envelope per distinct type,
and the roll-up carries `fleet` + `envelopes`. All flyability numbers moved: the observed-track
baseline went 58.4% → **63.2%**. The CLAUDE.md/README claims about UNK were corrected.

**`summary.json` now carries an `accuracy` block** (+ per-row `ade_m`/`fde_m`/`overlap_steps`).
A batch's error against the observed tracks is its headline result and it existed only as a
printed mean — comparing eight ablation cells meant scraping stdout. Mean AND p95, because
chained window-mode error compounds into the tail. `overlap` is a required argument to
`write_batch`, not optional: an optional metric is one that silently goes missing.

**Predictions render in the frontend.** `build_scenario_comparison_czml.py` detects the record
schema (`optimizer_states`/`simulator_states` vs `predicted_states`) and emits `pred-` entities
for a prediction batch; the entity-id prefix is already what the frontend keys `kind` off, so
`predicted` gets its own purple colour and legend checkbox. Two follow-on fixes:

- **Off-target recolouring restricted to the optimizer schema.** A forecast essentially always
  misses the 106.75 m gate, so it marked 27/27 groups off-target and repainted every prediction
  yellow — the kind colour was never once visible and a marker that fires on everything carries
  no information. `properties.status` stays accurate; deviation is reported by the evaluation
  report and comparison index.
- **The frontend repaint skip was keyed on the wrong thing.** It skipped legend repaint whenever
  `status === "offTarget"`, but what it exists to preserve is a baked VERDICT colour (the
  yellow). Predictions never get that bake yet are always `offTarget`, so they rendered from the
  CZML — matching their legend swatch only because `PREDICTION_COLOR` and the TS legend entry
  happen to hold the same RGB. Now keyed on whether a verdict colour was actually baked.

**Verified in-browser** (Linux Chrome, KRDU, all 4 categories): purple prediction + white
reference paths render together, the `Predicted` legend checkbox removes only the purple, and
the Optimization panel's metrics follow the selected category (752 m / 2.0% iTransformer full
vs 3184 m / 0.0% PatchTST window, matching the evaluation reports — and PatchTST window is
visibly the more scattered fan, an independent read of the same gap). Also backed by tsc clean,
451 frontend tests, `npm run build`, 54 ts_transformer + 25 CZML-builder tests, and a structural
check of the published artifacts against every contract point the frontend reads. Gotcha found
while verifying: comparison entities are time-windowed, so at a clock time outside a group's
availability the scene is legitimately empty — pause inside a window before concluding the
overlay is broken.

### 2026-07-19 — code review fixes: ts_transformer contracts, env resolution, identity single-sourcing

Applied the 15 findings of a full-diff review (ts_transformer + the uncommitted env-script changes):

- **train.py**: a non-finite train/val loss now RAISES ("training diverged…") instead of sailing through the best-val bookkeeping — previously a run that went NaN before any finite improvement wrote the last (NaN) weights as a "successful" checkpoint and emitted literal `NaN`/`Infinity` into history.json. `load_checkpoint` now uses `weights_only=True` (payload is tensors + primitives) and refuses a checkpoint whose serialized channel order differs from `channels.CHANNELS` (a same-length reorder loaded cleanly and silently mis-mapped every channel). Flights too short to yield one training window (window mode: `seq_len + pred_len`; build only requires `seq_len + 1`) are excluded — counted, not silent — before the split, so they no longer occupy split slots and produce the misleading "empty window set" abort.
- **Aircraft type is part of the checkpoint**: `TSConfig.aircraft_type` (default A320, moved to config.py) is serialised with everything else; predict defaults to the train-time value instead of its own CLI default, and warns when explicitly overridden — the type sets target Vref/TCH, i.e. the ENU frame and the gate target the normalizer stats were fit under.
- **predict --device**: the compute device is a runtime property, no longer read from the checkpoint (an explicit-`cuda` checkpoint was unusable on a CPU-only machine with no override).
- **Horizon cap is stated, never silent**: window-mode chaining still stops at 300 steps (600 s), but a forecast that ends short of the threshold is flagged `horizonCapped` (record source), `horizon_capped` (summary row), and predict prints a per-batch WARNING — previously those ~2% of flights were graded as huge gate failures indistinguishable from model error.
- **Split stability**: train/val/test assignment is now a per-flight sha256 of `(seed, flight_id)` instead of a positional permutation — adding/removing one flight to the harvest reshuffled the entire split (old test flights silently entered training on a retrain). Fractions are now approximate; empty train/val raises with the real cause (tiny datasets previously rounded val/test to zero and died later blaming window sizes). NOTE: this reshuffles the split once relative to the existing KRDU checkpoints — retrain before comparing new runs against them.
- **synthetic.py**: icao24 derived via `zlib.crc32(runway)` instead of the process-salted builtin `hash()` — identically-seeded synthetic data now has identical flight identities across interpreters (previously `predict --split test` on regenerated data intersected on zero flights).
- **Identity + filename single-sourcing**: `flight_key` moved to `flight_scenarios/identity.py`; `scenario_optimization._scenario_filename` and ts dataset/export both import it (the two copies had already drifted: `flight{i}` vs `scenario{i}` fallback — unified on `flight{i}`). Record-filename suffixes + `REFERENCES_DIR` hoisted into `optimization/evaluation_export.py`, imported by both writers; the "MUST match" mirror comments are gone.
- **dataset.py**: falling through to `*_landings.json` (a download-only dir with no `*_arrivals.json`) now prints an explicit UNTRUNCATED-tracks warning — a different task/duration distribution than the windows were sized for.
- **Env resolution deduplicated + made content-aware**: new `scripts/activate_aeroviz_env.sh`, sourced by `run_all_tests.sh` (warn-and-continue) and `start_aeroviz_fullstack.sh` (abort). Candidates are probed with `import casadi` instead of trusted by name (on this machine `aviation` exists but is another project's env — both scripts previously accepted it when active and fell back to it when `aeroviz` was missing or `AEROVIZ_CONDA_ENV` was mistyped; the launcher then crash-looped the backend on `import casadi`). An explicit `AEROVIZ_CONDA_ENV` is now the ONLY candidate. The launcher ACTIVATES the env rather than direct-exec'ing `envs/<env>/bin/python`, so activate.d hooks (the libstdc++ `LD_LIBRARY_PATH` fix) apply to the backend subtree; missing conda now reports "conda is not on PATH" instead of probing `/envs/...`.
- **Docs**: CLAUDE.md's Key Defaults ts bullet updated to the real 60/300 window sizes and the measured-distribution rationale (it still carried the superseded 30/150 + "~3.5–5 min arrival" story that config.py refutes); requirements.txt pin rationales corrected (the torch comment claimed `weights_only=True` while the code passed `False` — now the code matches the claim; the numpy `copy=None` rationale matched nothing).

Follow-up pass over the review's below-the-cap cleanup findings (same day):

- **forecast.py**: the HORIZON_FULL branch of `forecast_approach` was a copy of `recursive_forecast`'s single-pass body that also skipped its anchor validation — both modes now run through `recursive_forecast` (full mode's `pred_len` covers `max_steps`, so it is one pass; an explicit `anchor < seq_len - 1` now gets the clear ValueError instead of a cryptic vendor shape error). `_forward` forces `model.eval()` (dropout noise from a train-mode model would compound through the chain). `default_anchor` dropped its never-read `series` parameter.
- **models.py**: PatchTST now receives `act=config.activation` (the vendored Model takes activation as a bare kwarg, so `TSConfig.activation` previously applied to iTransformer only while the checkpoint recorded it for both).
- **metrics.py**: `displacement_metrics` folded into `trajectory_metrics`; `error_components` computed once per call (was twice, with per-step displacement recomputed twice more) via a `displacement_grid` key; `_positions` uses `channels.POSITION_IDX` instead of re-deriving it. `_spread` stays vectorised — delegating to the stdlib-only `evaluation/stats.signed_spread` would sort millions of boxed floats in `evaluate_split` — but a new seam test pins the two equal on the same input, replacing the mirror comment with a checked property.
- **export.py**: `PredictionRecord.source`/`final_time_s` became read-through properties (three stored copies of `final_time_s` per record could disagree after a future edit; `evaluation` rejects `final_time_s != states[-1].t`); the whole observed track is converted to states ONCE and the anchor sample + reference span are slices of it (the per-sample GeodeticState loop previously ran twice over the tail); summary rows go through the new `evaluation_export.summary_row` (shared with `scenario_optimization._summary_record`, which now wraps it); a redundant `min()` removed.
- **train.py**: `_predict_split` stores decoded windows as float32 (float64 doubled the live peak — hundreds of MB at full-mode scale on the 16 GB swap-bound machine — for precision metre-scale metrics cannot use); the duplicated val `evaluate_split` call in both ternary arms collapsed.
- **__main__.py predict**: the split filter now runs on the RAW flight dicts (keyed identically to `build_series` via `flight_key`) BEFORE the expensive series build — a default `--split test` predict previously built and discarded ~85% of the work. Verified live: "built 1/1 series" for a 14-flight file.
- **dataset.py**: the track-span check moved BEFORE `build_scenario` (too-short flights no longer pay aircraft resolution + per-sample least-squares fits before being skipped; the raw-waypoint span is the identical value); dead code removed (`Normalizer.decode_torch`, `FlightSeries.duration_s`/`.source`).
- **channels.py**: dead `VELOCITY_IDX` + `Frame.to_dict`/`from_dict` removed; new `Frame.latlon_from_en` single-sources the inverse projection, used by `states_from_channels` AND `synthetic.py`'s waypoint generator (which previously hand-rolled a fifth copy of the frame formula; it now builds a `channels.Frame` and dropped its direct geokit constant imports).
- **package.json**: `npm run backend` resolves its interpreter through `scripts/activate_aeroviz_env.sh` instead of bare `python` (it was the third, uncoordinated env-selection seam).
- Deliberately NOT applied: batching the predict loop (would complicate the single-flight inference seam for a non-bottleneck), early-stopping the window-mode chain at the threshold (changes the measured object — `truncate_at_threshold` takes a global argmin), the vendored PatchTST `pv()` NameError on non-default positional encodings (`layers.py` is documented "copied whole, unmodified"; fixing it breaks the byte-identical vendoring contract — landmine noted here instead), and a src-layout re-packaging of ts_transformer's flat modules (the sys.path-shadowing hazard is real but latent; disproportionate to fix now).

### 2026-07-19 — ts_transformer: first REAL-data run (KRDU, 995 arrivals) — both models × both horizon modes

Harvested ADS-B landed (3747 arrivals, 5 airports, 815 MB). Full matrix trained and graded on KRDU: 995 arrivals over 6 runways, split by flight 697/149/149, 96k training windows, 120-epoch cap / patience 15 / `lr=5e-4`, RTX 4060. All four prediction batches graded by `python -m evaluation`. Artifacts under `4dTrajectory/outputs/KRDU/ts_{model}_{mode}/` and `ts_pred_{model}_{mode}/`.

**Displacement error at matched lead times** (the only cross-mode-comparable axis — headline ADE/FDE average over different horizon-length distributions and must not be compared):

| model | mode | 10 s | 30 s | 60 s | 120 s | 300 s | 600 s |
|---|---|---:|---:|---:|---:|---:|---:|
| iTransformer | window | 259 | 390 | 687 | — | — | — |
| iTransformer | full | 587 | 611 | 840 | 1502 | 3227 | 6135 |
| PatchTST | window | 158 | 409 | 995 | — | — | — |
| PatchTST | full | 253 | 399 | 830 | 1914 | 3983 | 7142 |

**Whole-approach prediction graded at the threshold** (149 test flights, directly comparable):

| model | mode | lateral mean | lateral p95 | path deviation |
|---|---|---:|---:|---:|
| iTransformer | full | **1070 m** | **3136 m** | **1895 m** |
| iTransformer | window (chained ×10) | 1594 m | 6898 m | 1918 m |
| PatchTST | full | 1804 m | 4666 m | 2488 m |
| PatchTST | window (chained ×10) | 2815 m | 7354 m | 3152 m |

Findings: (1) **the cost of chaining lands in the tail, not the mean** — chained-window vs one-pass-full is 1.5× on lateral mean but 2.2× on p95, because once one chained pass goes wrong the next nine extrapolate from a wrong history; training directly for the long horizon beats chaining a short one. (2) **iTransformer beats PatchTST at long lead for a structural reason** — PatchTST is channel-independent (`TSTiEncoder`), so it cannot represent the east/north coupling of a turning aircraft, while iTransformer attends *across* variates; note the reversal at 10 s lead (PatchTST 158 vs 259 m), where the aircraft is nearly straight and independence costs nothing. (3) **0/149 gate passes in all four runs is the honest result** — 106.75 m is FAA containment for a planned/flown approach, not a forecast-accuracy target; the number measures the gap between a statistical prediction and a certifiable trajectory. (4) Real is much harder than synthetic (423 vs 286 m ADE): synthetic approaches are straight-in, real ones are vectored, and *when* the turn onto final happens is a controller's decision a single-aircraft model structurally cannot see.

**Three real bugs the first real-data run exposed**, none of which a loss curve would have shown:

- **Silent training-data contamination.** A harvest directory holds five overlapping views of the same flights — `*_arrivals.json` (truncated, the training input), `*_landings.json` (SAME flights untruncated), `*_combined_czml_input.json` (all runways merged), plus `*_heading_rejected.json` / `*_local_rejected.json` (tracks the harvester explicitly THREW OUT). `glob("*.json")` loaded every flight three times over plus the known-bad ones. `dataset.select_flight_files` now takes the first matching pattern only, never mixes, always excludes `*_rejected*`, and prints what it skipped.
- **Aircraft type is `"UNK"` for all 3747 flights** and `flight_scenarios._resolve_aircraft` raises rather than guessing — the batch died on flight #1. Added `--aircraft-type` (default `A320`, printed every run). Not cosmetic: it sets the target state's Vref and threshold-crossing height, i.e. what the gates measure the final state against.
- **Sizing was wrong twice** (see the environment entry below for the corrected defaults).

### 2026-07-19 — Environment consolidated into `aeroviz`; the `aviation` name collision documented

`aeroviz` (py3.12) is now the single thesis environment — `torch` installed alongside the existing acquisition (`traffic`, `pyopensky`), CIFP (`cifparse`, `arinc424`), `casadi`/IPOPT, `openap` and conda-forge geospatial stacks. `run_all_tests.sh` needs no change: its `4dTrajectory` entry now covers the ts_transformer suite too.

**The name collision.** CLAUDE.md said "Python env: conda `aviation`", carried over from another machine — `run_all_tests.sh`'s own comment records "Env names differ per machine (`aeroviz` here, `aviation` elsewhere)". On THIS machine `aviation` is the editable env of `/home/supercomputing/studys/AivationTransformer`, an unrelated project (pure-pip, py3.11, 254 packages, 8.6 GB). Acting on the stale line nearly deleted the real thesis env; the ts_transformer package was also initially targeted at `aviation` for the same reason. Both CLAUDE.md lines are corrected and the collision is now documented under Environment.

**Consolidating the other way is BLOCKED — measured, not assumed.** `cifparse` >= 2.0.4 (aeroviz has 2.0.9) uses PEP 701 f-strings (nested same-type quotes), which is Python 3.12+ syntax. Every version from 2.0.4 up fails `compileall` on py3.11; only 2.0.0 and earlier import there — a 9-patch regression in the ARINC 424 parser that feeds `approach_constraints`. Upstream's PyPI metadata claims `requires_python >=3.10` and is simply wrong. So a py3.11 env cannot host the thesis at current package versions. `casadi`/`openap`/`arinc424` install fine on 3.11; `cifparse` alone is the blocker.

**One real interaction between torch and the existing stack, found and fixed.** `import torch` then `import traffic` raised `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found`; the reverse order worked. Cause: pip's manylinux torch wheel has no RPATH into the env, so it resolves `libstdc++.so.6` from the system (CXXABI ≤ 1.3.13), and once that SONAME is loaded conda-forge matplotlib's `_c_internal_utils.so` — which needs 1.3.15 — is answered by the already-loaded old one. `run_all_tests.sh` runs every suite in ONE pytest process with `4dTrajectory` (torch) listed before `trajectory_data_process` (traffic), i.e. precisely the failing order. Fixed with `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh` prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (plus a matching `deactivate.d`). Note it only takes effect under `conda activate` — calling `envs/aeroviz/bin/python` directly still reproduces the failure. Full suite after the fix: **495 passed, 1 failed** (the pre-existing `test_fixed_time_objective_weights_control_effort_at_one` numpy scalar-conversion `TypeError`); the count rose from 464 because the 32 ts_transformer tests now run in the same invocation.

Anything installed into `aviation` during the attempt (casadi, openap, arinc424, the broken cifparse, plus a tqdm/wcwidth bump and an editable geokit) was rolled back; that env is verified back at torch 2.9.1+cu128 / tqdm 4.67.1 / wcwidth 0.3.0 with no thesis packages. Env spec backups for `aeroviz` live in `.env-backup/` (pip freeze, conda explicit, environment.yml).

**ts_transformer defaults re-sized against the real data** (see the entry below for the original synthetic sizing). The 3747 harvested arrivals have durations p5 235 s / p50 328 s / p90 607 s / p99 920 s — much longer than the "25 km at 120 m/s ⇒ 3.5–5 min" straight-line estimate, because real arrivals are vectored (downwind legs, base turns, holds), so the flown path far exceeds the straight-line distance to the entry ring. `seq_len` 30 → **60** (120 s), `DEFAULT_PRED_LEN_FULL` 150 → **300** (600 s): full mode now covers the complete remaining approach for **97.8%** of flights, where 150 covered **57.6%**.

### 2026-07-19 — `4dTrajectory/ts_transformer`: iTransformer + PatchTST integrated as a learned-prediction sibling to the optimizer

New package answering *what trajectory WILL this aircraft fly* (learned, no dynamics model) alongside `optimization`'s *what SHOULD it fly*. Both emit the same evaluation records, so `python -m evaluation --input <dir>` grades either against the identical regulatory gates.

**Vendoring.** `vendor/itransformer/` (MIT, thuml @ `c2426e6`) and `vendor/patchtst/` (Apache-2.0, yuqinie98 @ `204c21e`) copied byte-identical with only import paths rewritten, each with its `LICENSE` + a `PROVENANCE.md` recording what was copied/dropped and why. Vendored rather than installed because neither upstream is a packaged library and both resolve internal imports through a top-level `layers/` — installed side by side they collide (both ship a different `layers/Embed.py` and `layers/SelfAttention_Family.py`). Dropped from iTransformer: the Reformer/Flowformer/Flashformer/Informer attention variants, and with them the `reformer_pytorch` + `einops` deps. One shared `TSConfig` drives both (upstream's own argparse-namespace contract); `models.py` adapts iTransformer's 4-arg call and PatchTST's 1-arg call to one `model(x)`.

**Env.** Installed editable `geokit` into conda `aviation` (torch 2.9.1+cu128); the package is casadi-free by design, so it lives in the torch env while the optimizer stays in `aeroviz`. `requirements.txt` written. Verified both models forward CPU+CUDA on the RTX 4060.

**Three findings that changed the design, each measured not assumed:**

1. **Sizing.** An arrival truncated at the 25 km ring is only **~3.5–5 min** (~110–150 samples at 2 s), not the 8–12 min first assumed. The initial defaults (L=60 @ 4 s = 4 min lookback) exceeded a whole approach and skipped **5 of every 6** flights as "shorter than one window". Now `dt=2 s`, `L=30`, `H=30` (window) / `150` (full) — all 120 synthetic flights build, ~65 anchors each.
2. **Instance normalisation must be OFF.** iTransformer's `use_norm` / PatchTST's `revin` are ON upstream and strip each window's absolute level as nuisance; in a threshold-anchored ENU frame absolute position *is* the signal. Off wins in **all four** model×mode cells by 2.4–6.5× on ADE (iTransformer window 771→286 m, full 1972→303 m; PatchTST window 910→672 m, full 2268→701 m) — and on converges *sooner* (49–56 vs 74–90 epochs), i.e. to a worse optimum, not undertrained. Defaults flipped; `--instance-norm` re-enables for re-ablation on real data.
3. **Two alignment bugs caught by running the real thing** (both would have produced plausible-looking wrong numbers): (a) `FlightSeries.flight_id` was the callsign, which repeats daily and across runway files — `predict --split test` returned 48 flights for an 18-flight split, and the train/val/test split leaked. Identity is now `dataset.flight_key` = `id_runway_icao24_landingTime`, the same function that produces the record filename. (b) The reference record covered the whole track while the prediction covered anchor→threshold; `evaluation.reference` resamples both at 101 fractions of *their own* arc length, so it reported 4349 m of "path deviation" that was pure span mismatch — **833 m** once span-matched. Records now anchor `t=0` at the anchor sample with `initial_state` the observed state there, and `states[0] == initial_state` as in an optimizer record.

**Contracts.** Channels `(e, n, u, ve, vn, vu)` in a threshold-anchored ENU frame; `psi`/`gamma` never regressed directly (±π wrap averages 179° and −179° to 0°, pointing the aircraft backwards exactly at the turn onto final) but derived as `atan2(vn, ve)`, which *is* the math-ENU convention, so the compass/ENU substitution has no place left to happen. `m` carried, never predicted (unobservable from ADS-B). Records are reference-shaped (`controls == []`) and built via `optimization/evaluation_export.py` — importable here because it is casadi-free — rather than hand-rolling a second copy of the record JSON. `final_time_s` always from `states[-1]["t"]`, never `pred_len × dt` (threshold truncation makes them differ; `evaluation.records` rejects >1e-6).

**Status.** 30 tests pass; verified end-to-end through the real CLI (train on GPU → predict → `python -m evaluation`). **Never trained on real data** — none existed in the tree (`trajectory_data_process/outputs/` absent, no `credentials.json`), so `synthetic.py` generates straight-in approaches for smoke-testing and fixtures. Every number above is synthetic and is plumbing evidence, not a result. Pre-existing unrelated failure noted: `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one` (numpy scalar-conversion `TypeError`).

### 2026-07-07 — "Profile" → "Approach view" rename (systematic)

The 2D approach page/toggle is now **Approach view** throughout (UI text, identifiers, CSS, files; `git mv` preserved history): `useApproachView.ts`, `ApproachViewPanel.tsx`, `ApproachViewToggle.tsx`, `approachViewAnalysis.ts`, `approachViewSources.ts`; identifiers like `isApproachViewOpen`/`setApproachViewOpen`, `approachViewMode`, `ApproachViewTrack/Sample/Input`, `buildApproachViewTracks`, `planApproachViewSources`; CSS `.approach-view-*`; toggle labels "View"/"Hide view". Deliberately NOT renamed — the geometric "profile" (altitude cross-section) keeps the term: `runwayProfileGeometry`/`RunwayProfilePoint`, `procedureProfileProjection`, `procedureVerticalProfileOverlay`, `ProfilePlot`/`ProfileOverlay`, the "Vertical profile" side-view mode label. Not re-checked in-browser.

### 2026-07-07 — Approach view: whole-track plotting, review fixes, perf cache (three passes, same day)

- **Whole track**: `sampleEntityTrack` walks outward in both directions from the current time (`TRACK_SAMPLE_STEP_SECONDS = 5`, `MAX_TRACK_SAMPLES_PER_DIRECTION = 600` backstop) returning the whole time-ordered track; all points kept tagged by containment tier; an aircraft is plotted iff some sample reaches PRIMARY (`trackEngagesProcedure` — full approach for flights that fly the procedure, unrelated traffic excluded). `splitTrackByContainment` splits ordered samples into contiguous runs sharing boundary points (gapless lines); PRIMARY/SECONDARY drawn solid (primary brighter), OUTSIDE dashed `6 5` + dimmed.
- **Review fixes**: walks also stop on `samePosition` (the HOLD-tail gotcha — otherwise never terminates and pads 600 duplicate threshold points per entity per tick); landed/parked entities dropped via liveness check (sample one step back == current → past the real track end → null); not-yet-airborne (null current) dropped; plot domain grows to include plotted track samples (corner-cut stretches no longer clipped) and the selected-flight label gets the plot clipPath; `ApproachViewInput.current` tightened to non-null; engages-gate = `trail.some(PRIMARY)`.
- **Perf**: per-entity cache of the classified whole track (`trackCacheRef` keyed by flight id), rebuilt only when geometry deps change (loaded procedure/frame, active routes, source set) — never on clock ticks; each ~120 ms tick then only re-checks liveness (1–2 `getValue`) + classifies the current marker (O(aircraft × track length) → O(aircraft) per tick). `sampleEntityTrack` split into `currentIfLive` + `sampleWholeTrack`; analysis module exposes `classifyProfileSample` / `classifyTrackSamples` / `trackEngagesProcedure` / `sortProfileTracksBySelection` (`buildApproachViewTracks` kept as their batch composition for tests).
- DEFERRED: interior-gap `break` in the walk is latent (current CZMLs are single-interval); the pre-existing `useCzmlLoader` clock write is still ungated for the Observe+comparison two-writer case. Not re-checked in-browser.

### 2026-07-07 — Approach view mirrors the active tab

- New pure `src/data/approachViewSources.ts` `planApproachViewSources(...) → {observed, optimized}` — the single source for "which trajectory sources are the current tab's globe content": `observed` reuses `planObservedTracks(...).visible`; `optimized` = `mode === "optimize" && hasOptimizedSource`. Used by BOTH the hook (what it samples) and the panel's "CZML linked" badge (`profile.sourceLinked`). Observe plots observed; Optimize plots only the optimized playback; Fly/Compare plot neither.
- **Root-cause follow-up**: `planObservedTracks` no longer keeps the observed CZML loaded behind an open view outside Observe (`relevant = mode === "observe"`). `useCzmlLoader` drives the shared `viewer.clock` from the observed CZML's hours-long span, so loading it in Optimize hijacked the clock and made the optimized playback aircraft vanish; releasing it also saves a 100+ MB load per non-Observe tab. Runtime clock behavior not re-checked in-browser.
- KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource not yet fed to the view — Observe-with-comparison plots neither source; wiring it in is a follow-up.

### 2026-07-07 — Pipeline exposes the control mesh; mesh defaults single-sourced

- `run_scenario_optimization.py --n-segments` (unconstrained) / `--n-seg-per-phase` (constrained), either/or by mode, defaulting to CollocationOptimizer's own 8/3; validated ≥2/≥1 at both CLIs.
- The constrained batch path never passed `n_seg_per_phase` before (stuck at the default with no override) — now threaded main → `optimize_scenarios_constrained_iaf` → `_optimize_one_scenario_iaf` → both IAF selectors → `_solve_iaf` → `CollocationOptimizer`.
- Defaults single-sourced in `collocation/optimizer.py` (`DEFAULT_N_SEGMENTS`/`DEFAULT_N_SEG_PER_PHASE`); backend + batch import them; frontend "Control segs/leg" default aligned 2→3 (defaults had silently diverged: frontend 2, backend fallback 4, optimizer/batch 3). The frontend→backend HTTP path had always wired `nSegPerPhase` correctly — the gap was batch-only. Verified: nsp 2→4 changes the constrained plan 85→125 nodes.

### 2026-07-07 — Optimize tab: solve-time readout; HSL hook benchmarked, kept dormant

- `optimization_backend._optimize` puts `result["timings"] = {buildS, solveS, playbackS, totalS}` in the response (was log-only); frontend `OptimizationTimings` + `parseTimings` (absent → null) + a "Solve time" row. Plan readouts restyled to a single column of full-width rows (3-col grid truncated values).
- HSL: `collocation/components.py` reads `AEROVIZ_IPOPT_LINSOL`/`AEROVIZ_IPOPT_HSLLIB` (casadi's IPOPT has the `hsllib` loader — no rebuild; setup in `docs/hsl-linear-solver-setup.md`). Measured: free Coin-HSL Archive has only MA27, 3–27× slower than MUMPS here (small NLPs; OpenBLAS/OpenMP clash) → dormant, kept for a future MA57 academic license.

### 2026-07-06 — Review pass: toggle false-open, ownership seams, manifest-only eval reads, required guards

- **[BUG]** `runwayMatchesSelection(null, X)` is match-ALL, so with the top bar on "All runways" every toggle showed "Hide profile" and clicks closed the invisible page; `open` now also requires `selectedRunway !== null`.
- `RunwayProfileToggle` prop `borrowSelection` (PilotPanel, unconstrained only): opening saves the pre-open `{selectedRunway, isRunwayProfileOpen}` (only when opening actually changes the selection) and restores on close/unmount. Constrained Optimize doesn't borrow — the forced-display hook owns the runway there.
- `useForcedProcedureDisplay`: `SavedDisplay` now carries the profile-open state, restored together with the runway (ownsRunway only); `forceRunway: null` never touches either. `ready` flip only fires when a drive is pending; savedRef bakes `ownsRunway` (what makes a non-null→null `forceRunway` flip restore correctly).
- **Evaluation read side manifest-ONLY** (user decision, no glob fallback): `evaluation.records.load_records` reads a batch dir via its `summary.json` roster; manifest-less dir / listed-but-missing / empty roster raise; `--pattern` removed from both CLIs. (Globbing counted orphans — the KRDU 1023-vs-996 class.)
- **`min_altitude_m` REQUIRED** on `rollout_controls`/`simulate_controls` — the 0.0 default silently validated diverged replays km below elevated fields; target-less replays now pass an explicit 0.0.
- **Constrained-ness is an explicit manifest field**: `_upsert_category` stamps `"constrained": bool`; builder `--constrained`; frontend `ComparisonCategory` requires it, `_cons`-suffix detection deleted; all 5 airports' `categories.json` migrated in place.
- Cleanups: record-filename suffix constants; shared `_fake_optimizer` test factory.

### 2026-07-06 — Rollout guard margin (residual low-success root cause)

97% of KRDU/asdb gate-failures were rollouts truncated by the zero-margin ground guard at exactly the floor: min-time plans deliberately ride the floor, and cm-scale integration noise (measured: 3.9 cm dip for 1.5 s on a replay that lands 0.7 m out) tripped a guard meant for km-scale divergence. Fix: `ROLLOUT_GUARD_MARGIN_M = 5.0`, guard = floor − 5, both rollout call sites. (Diagnostic lesson: the eval record's `final_time_s` IS the truncated end — compare against the states-file's plan T to detect truncation.) Same day:

- `_clear_stale_records` deletes stale top-level records at batch start (27 orphan evals inflated a report).
- `--fitting-type {hs,trapezoidal,rk4}` end-to-end (`FITTING_SCHEMES`; default hs). rk4 verdict: fine on smooth constrained solves; basin-fragile on aggressive unconstrained min-time (auto M 9.1 km off; only M=64 recovers HS's optimum at ~4× cost) — HS stays default.
- `--state-substeps M` end-to-end through both solve paths + frontend "State substeps" input → backend `state_substeps` on both branches (cache keys include it). Measurements in Key Defaults.
- `ipopt.max_iter` wired ("backend never finishes" postmortem): constrained M=64 ≈ 640 subintervals × per-node inequality rows, no iter cap, solves serialized behind the worker lock. `DEFAULT_MAX_ITERATIONS = 3000`, `CollocationOptimizer(max_iterations=…)` on all three solver constructions; backend `maxIterations` reaches both branches (make_optimizer had accepted-and-ignored it).
- Stale-artifact audit closed the `write_reference_records` and CZML-builder (`clear_stale_outputs`) accumulation gaps; single-file overwrites audited clean.
- NOTE: `runway_cons` off-target populations likely contain the wrongly-truncated family — re-examine after re-run. All categories need re-running after preparation: `python run_scenario_optimization.py --jobs 6`.

### 2026-07-06 — Observe constrained auto-open + shared Profile toggle + forced-display hook

- Extracted `src/hooks/useForcedProcedureDisplay.ts` from PilotPanel. Contract `{active, forceRunway}`: non-null `forceRunway` → the hook OWNS `selectedRunway` (save-once/force/restore-on-inactive + dependency-free restore-on-unmount); `forceRunway: null` (Observe) → never reads/writes `selectedRunway`, drives only the panel + `procedures` layer.
- Observe trigger in ControlPanel: active = trajectories layer ON ∧ comparison overlay ON ∧ constrained category ∧ runway selected.
- Shared toggle component (one parameterised toggle driving the global open state; opening focuses `selectedRunway` on the governed runway) used in ControlPanel, PilotPanel's Target-State header, and ProcedurePanel.
- **Dock-handoff race**: two forcing docks switching share one React passive flush — the incoming hook read the outgoing dock's still-forced display and saved a polluted baseline. Fixed with a one-render `ready` gate (no-op on first commit; the sibling's restore is batched with the `ready` flip in React 18). Integration regression test with the real AppProvider. Verified in-browser (KRDU 05L).

### 2026-07-06 — Evaluation window symlog label pile-up

Under `symlog(v) = sign·log10(1+|v|)` every |v|<1 collapses to ~0, stacking the ±0.01/±0.1 decade labels; `EvaluationReportWindow.tsx` now skips sub-1 decades (keeps 0, ±1, ±10, …).

### 2026-07-06 — All-modes sweep + unconstrained batch's trapezoidal fitting (low-success root cause)

- Omitting `--target-type` runs all three modes (`asdb`/`runway`/`runway_cons`) per airport; `--with-constraint` without it is rejected.
- Unconstrained success 4–14% vs constrained 76–97%: the batch's `optimize_scenario` still used trapezoidal after the 07-05 HS flip (batch-edition seam class). Min-time solves ride the floor exactly where 2nd-order trapezoidal is dynamically unfaithful: plan on-target, rollout 5–15 km off (full-T, not truncated). A/B (DAL1407): trap 5950 m vs HS 3.4 m, and HS found a better optimum at ~3× solve time. Fixed to HS.

### 2026-07-06 — Arrival-segment truncation: 25 km entry ring; locals excluded

Landing tracks were validated by their END only, so depart-and-return flights started ON the field. New `trajectory_data_process/arrival_segment.py`: walk backward from touchdown; the arrival starts after the LAST run of ≥3 consecutive samples outside the 25 km ring (hysteresis: one jittery fix can't cut); plain arrivals also cropped to the ring so every arrival shares one entry boundary (the on-ring state distribution = interaction-study boundary condition). Never-outside tracks: `local` if start ≤5 km (takeoff→circuit, written to `<ICAO>_local_rejected.json`, never silently dropped), else coverage-limited arrival kept whole. `truncate_flights` rebases times to 0 and annotates `arrival_truncated`/`cut_samples`/`arrival_duration_s`/`entry_time_utc`. Integrated in `build_arrivals.py` (renamed from `landings_to_czml.py`): raw `*_landings.json` untouched; derived `*_arrivals.json` feed CZMLs + czml-input; airport centre from `config/runway_thresholds.json`. `FlightScenario.source` carries `entry_time_utc` (co-temporal placement key). All 5 airports regenerated: KMSY 400, KRDU 996, KSJC 319, KSMF 714, KSTL 1054 arrivals. All pre-existing batch outputs predate this (and the floor fixes) — stale.

### 2026-07-05 — Evaluation detail window: legible profiles + one colour language

`DeviationProfile` ranked-dot charts replace the illegible bar walls (lateral: log axis; vertical: signed symlog); each chart draws AND labels its own gate in-plot, dots coloured by that gate, legend carries the outside count. One colour language: red/green = per-flight gate verdict only (summary cards made neutral; scatter legend explains its colours). Values are the backend report's rows verbatim. Verified in-browser (KRDU runway_cons, 1001 trajectories).

### 2026-07-05 — Off-target marking moved onto the RESULT path

"Successful flights ending mid-air" = off-target flights with guard-truncated rollouts (correctly classified; they merely LOOKED successful because only the reference was marked). Now the simulator/result path bakes `OFF_TARGET_COLOR` yellow + "(off target)" name; reference drops to dark-amber `OFF_TARGET_REF_COLOR`; plan keeps legend orange. `useComparisonTrajectoryLayer` skips its repaint for `properties.status == "offTarget"`. NOTE: KRDU runway 32's 77/200 off-target (vs 6–7 on 23L/R) is a real quality signal — likely the replay-divergence family.

### 2026-07-05 — Below-ground trajectories: altitude floors + rollout ground guard

Three root causes for optimized trajectories below field elevation (per-leg step-down floors verified working — the dives lived where no floor existed):
1. The global altitude bound (`components.altitude_floor_m` = target − 300 m, documented as a never-binding box) BINDS — min-time solves dive to it. Fixed: margin 300 → 5 m, a real operational floor; `min(initial, target)` deliberately NOT used (a start below the floor is bad data and fails loudly).
2. The start→first-fix transition phase had no floor above the global one (FFT2071 dove to −173 m). Fixed: transition altitude bound = min(start alt, `_first_leg_entry_floor_m`) − margin (min() IS needed here — a start below the first fix's altitude is legitimate climb-to-join geometry).
3. The batch rollout had no ground envelope (`CasadiSimulator.step` has no checks): `_GroundCheckedSimulator` + `rollout_controls(min_altitude_m=…)`.
Also: evaluation gates judge the FINAL state only, so mid-flight dives never failed a gate — pre-fix success rates were inflated. All batch outputs stale.

### 2026-07-05 — In-app evaluation report window ("Details")

The comparison builder PUBLISHES the evaluation report verbatim (`publish_evaluation_report` → `comparison/<category>/evaluation_report.json`); the frontend fetches that copy and only formats/sorts/plots — no metric recomputed client-side (new metrics go in `evaluation/metrics.py`; contract documented in `src/data/evaluationReport.ts`). `EvaluationReportWindow.tsx`: draggable floating window (portal, same shell as Dynamics-Comparison) with summary cards, 8260.58D gates note (values from `report.thresholds`), aggregates table, three SVG charts, full verdict table. `OptimizationSummary` Details button; report fetched lazily, cached per (airport, category); missing report shows a helpful message. Verified in-browser (KSMF runway_cons).

### 2026-07-05 — Comparison CZML: off-target status + evaluation metrics into the frontend

Builder `--evaluation-report`: verdicts keyed by eval filename, joined to summary rows via `eval_file`; solved-but-failed-gates → status `offTarget` with `lateralErrM`/`verticalErrM` on the index record; `optimization_stats(summary, report)` builds the index `optimization` block (successful/successRate/avgStateErrorM/avgTimeS — nothing recomputed). No report → byte-compatible plain behavior. Runner: the (cheap) evaluation report now always runs before the tails; the CZML step passes it; reusing a pre-evaluation optimization skips evaluation with a loud note. Frontend: `ComparisonGroup.status = "solved"|"offTarget"|"failed"`; flight list flags off-target yellow (`.flight-table-offtarget`, #ffcd28); `OptimizationSummary` metrics were already wired.

### 2026-07-05 — Panel "final horiz err" ≠ playbackDrift, round 2 (verified in-browser)

Two stacked causes behind Δ25 m vs playbackDrift 0.6 m: (1) the LOOP_STOP wrap heuristic never fired (Cesium preserves overshoot on wrap) — replaced all elapsed-based heuristics with `clock.onStop`; `makeReadoutEmitter` returns `{tick, stop}` (stop = throttle-bypassed exact-`stopS` emit, deduped), both playback hooks subscribe to onTick + onStop; Reset/backward scrubs never raise onStop. (2) `czml_common.document_packet` truncated the clock interval end to whole seconds — now `iso_ms` (74.7 m/s × 0.338 s ≈ the phantom 25 m). Backend audited clean otherwise (`playbackDriftM` and the CZML doc end read the same terminal sample).

### 2026-07-05 — Constrained-IAF batch 0% solve rate (two stacked bugs)

1. `scenario_optimization._solve_iaf` still unpacked `segments, _spans = build_constraint_segments(...)` after the 07-03 change to a plain list return — every constrained batch since 07-03 was broken (backend HTTP path updated, batch caller missed). Seam regression test added.
2. The batch's threshold target (from `config/runway_thresholds.json`) sat up to 390 m from the procedure's CIFP threshold → frame-anchor guard fired. `_snap_target_to_procedure` snaps the solve target horizontally onto the procedure's last waypoint (keeps altitude/Vref/pavement heading/glidepath); `_iaf_result` writes the eval record against the SNAPPED target; reference records keep the scenario target. (Diagnostic note: the batch log truncates errors to exception TYPE names per IAF — reproduce one scenario for a real traceback.)

### 2026-07-05 — Evaluation review fixes (degenerate records + HTML escaping)

- One 1-sample "solved" record (rollout truncated at its first step) aborted the whole batch report. Fixed at both ends: producer `_require_usable_rollout` (<2 samples → recorded as a FAILED scenario); evaluation guards via shared `reference.horizontal_arc_length_m` — comparison skipped with a row note, visualize drops undrawable polylines.
- HTML: embedded JSON escapes `</` as `<\/`; `esc()` on every data-derived string reaching innerHTML/Plotly.
- Contract: `final_time_s == states[-1].t` required on solved records; `resample_by_arc_length` rejects n<2; chart labels use unique record file basenames.

### 2026-07-05 — Former combined scenario pipeline runner

The comparison + evaluation runners merged at the time. They duplicated the expensive steps and wrote divergent opt_dir contents (the comparison runner silently overwrote `reference_file` pointers away). Optimization always runs with `--reference-tracks`; tails remain selectable via `--outputs czml,eval` (default both); `--skip-optimize` reuses an existing summary.json. The combined preparation/optimization entry point described here was superseded by the 2026-07-23 split above.

### 2026-07-05 — `evaluation` package (regulation-derived gates)

New root package `evaluation/` — the file-based seam at the end of the pipeline (geokit + stdlib only, never imports the optimizer). Record contract + gates: see Key Defaults. Salient points:
- `evaluation_export.py` maps the true-dynamics rollout onto the contract — `rollout_piecewise_constant`'s samples already carry the active control per sample (nothing re-derived). Both batch modes roll out once and write `*_eval.json` next to every `*_states.json`, INCLUDING failed scenarios (empty record = how solve rate is computed from files alone); summary rows carry `eval_file`. Old `*_states.json` can't be converted offline (no controls) — re-run for eval records.
- Reference records: observed track in the same contract (`flight_scenarios.state_samples_from_track`, times rebased, same target; `controls == []`). `write_reference_records` looks flights up by full identity `(id, icao24, landing_time_utc)` (missing ⇒ raise) → `references/<identity>_reference_eval.json`; `reference_file` stamped on every eval record, failed included. `compare_to_reference` = flight-time delta + path-shape deviation (arc-length resampling — time-matching would conflate speed profiles with geometry).
- CLIs: `python -m evaluation --input <dir> --output report.json` (+ threshold flags); `python -m evaluation.visualize` renders one self-contained HTML (Plotly CDN + embedded DATA; English body, abbreviations expanded on use; 8260.58D citations in the gate-sources note; `--max-tracks` cap stated on the page). Verified in-browser.

### 2026-07-05 — ψ corridor kills the looping/crawling pathology; drift guard; HS default

- Postmortem (nsp=2 loops): the solve CONVERGED onto a local optimum with ±2π winding (feasible — the join ψ box pins one node; excursions cancel), and the node-pinned terminal masked a 4.5–4.9 km true-dynamics rollout drift (trapezoidal on a winding path is node-feasible but dynamically meaningless).
- **ψ corridor** (structural fix): constrained heading variable bounds tightened from ±3π to the route heading hull ± 90° — winding optima cease to exist. The whole 07-04 family of crawls/Max_Iterations/basin-twitchiness died: 2×2 (HEAVE/custom × trap/HS) 4/4 clean, nsp 2/3/4/6 all converge.
- **Drift guard**: `playback_terminal_drift_m` → `playbackDriftM` on every response + stderr WARNING > 50 m. Never touches the NLP.
- Fitting verdict (doglegged H05LZ): trapezoidal 226–296 m drift vs HS 0.6–0.9 m at every nsp; HS ≈ 2–2.5× solve time. **Constrained default flipped to HS** (`_DEFAULT_CONSTRAINED_SCHEME = hermiteSimpsonNormalizedFullTransport`); trapezoidal stays selectable.
- Readout artifact (wall-throttled ~12 Hz emitter missing the terminal sample by up to throttle×multiplier sim-seconds) fixed via `makeReadoutEmitter`, shared by the optimized-playback and Compare hooks — later superseded by the onStop rework above.

### 2026-07-04 — Join constraints: pre-FAF fix passage + flexible FAC intercept + alignment tiers

(Current semantics in Key Defaults.) Highlights and lessons:
- ⑪ Fix passage evolved to the PRE-FAF fix only (all-fixes and entry-fix variants superseded, per user); enforced as a smooth squared form rescaled by `1/(2·tol)` so violations read in metres (no |·| kink); tolerance = the leg's k·RNP halfwidth, procedure-sourced.
- ⑫ Flexible FAC intercept replaced the exact-FAF pin: linear cross-track equality + upstream-only window ≥ L_final/5 before the FAF; `fac_distance_to_ltp` is THE one distance measure (glidepath d reuses it). Vertical semantics stay published-geography-keyed (`LpvFinalSpec.d_faf_m`/`prefaf_floor_m`; both None → gate off, byte-identical).
- Two 2π-branch bugs found by evaluating `nlp.g` at failed iterates and ranking violated rows: the terminal ψ pinned the wrong branch on double-90° routes (fix: `_route_unwrapped_target_psi` walks chained leg courses), and the join intercept `cos(ψ−course) ≥ cos30°` was 2π-periodic (fix: branch-aware linear box). Plus the duration-split regularizer for the flat time-split direction.
- Constraint families refactored into explicit per-family row functions behind a dispatcher (uniform `list[(expr, lb, ub)]`); `components.unwrap_angle()` collapses four unwrap copies. Two-tier FAC heading alignment added (positions-only corridors let large heading errors fit between nodes).
- `log_optimizer_config` writes one stderr line before any solving (optimizer/scheme/fitting/dynamics/transport/constrained).
- Frontend follow-up: custom starts KEEP the RNAV IF selection (the selector names the procedure; the start is independent) so constrained solves can start off-fix; transition threshold unified with the passage tolerance (`_first_fix_join_tolerance_m`) — the old 1–2 km dead zone (no transition, no disc) removed. Verified end-to-end on H05LZ HEAVE with a custom start 9.2 km out.
- KNOWN DATA GAP: per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc (926 m at k=0.5) instead of ~278 m (RNP 0.3).

### 2026-07-03 — CIFP thresholds everywhere; displaced-threshold root cause

- Root cause of up-to-970 m target gaps (KSJC 30R): runway.geojson `runway_surface` edges are PAVEMENT ends while CIFP's threshold is the DISPLACED landing threshold, plus a real `build_runway_ring` bug (declared length re-centred on the endpoint midpoint with asymmetric displaced offsets → rigid shift by `(he_disp − le_disp)/2`). OurAirports coordinates themselves verified accurate.
- Constrained target anchored on CIFP: `procedureThresholdAnchor(constraint, document)` — position = the constraint's last waypoint, altitude = CIFP threshold elevation, ψ = final course in the simulator convention; `PilotPanel.computeTrajectory` overrides the request target whenever a `procedureConstraint` is attached. runway.geojson still drives the unconstrained target and rendering.
- `procedure-details/index.json` hoists each runway's CIFP `threshold` `{lon, lat, elevationFt}` (null when uncoded); unconstrained targets prefer it too (`buildRunwayThresholdTargets(collection, index?)`; heading stays pavement-derived). CIFP is also more complete than OA (e.g. KSTL 30L's displacement missing from OA).
- `build_runway_ring` rewritten: `runway_surface` corners ON the OA endpoints; `landing_zone` ends moved inward by the displaced distance. All 5 airports regenerated; landing_zone matches CIFP ≤ 24 m except pure OA data gaps.

### 2026-07-03 — approach_constraints/collocation review fixes

- **One ψ convention**: `course_bearing` now returns the model convention (`atan2(Δn, Δe)`); the optimizer consumes it instead of re-deriving course math (guiding rule: constraint/course math has ONE source — `approach_constraints`).
- `ConstraintReport` unit-aware: metre and radian violations separated (`max_violation()` / `max_angular_violation()`, `is_feasible(tol_m, tol_rad)`) — 1 rad had counted as "1 m", a real false-feasible.
- Transition phase actually built: `_phase_plan` prepends the unconstrained start→first-fix phase (start > 2 km from the first leg's start fix); an approach whose first leg IS the final gets the FAF intercept on the transition phase. `build_constraint_segments` returns a plain list.
- Frame-anchor contract validated loudly; terminal-bank 1-node-phase latent bug fixed (`phase_starts`); per-phase auto state substeps (~3 s target; M-selection deliberately ignores `fixed_duration` so fixed- and free-time NLPs share one decision layout, the fixed solve seeding the free one); glidepath d measured on the same GARP→LTP axis as the lateral corridor; normalized position box 1e7→2e6 m; dead machinery removed (`partition_node_indices`, `LpvFinalSpec.da_hat_m`); README de-staled; mass-frozen approximation stated; HS docstring corrected (4th-order, O(h⁵) local).

### (undated, ~2026-06-28) — Backend SIGABRT root cause + solver isolation

The service "shut itself down": casadi's thread-unsafe symbolic construction under `ThreadingHTTPServer` (three concurrent NLP builds) corrupted the heap → SIGABRT; the old launcher then co-killed the frontend.
- `isolated_backend.py` `IsolatedRunner`: casadi-heavy endpoints (`/optimization/run`, `/dynamics-comparison/run`) run in a worker subprocess (`ProcessPoolExecutor(max_workers=1, mp_context="spawn")`); a native abort → `BrokenProcessPool` → clean `SolverCrashError` 500, next request spawns a fresh worker (self-heal). Decorator backends keep the same method interface (tests inject in-process fakes).
- Memory-aware worker lifecycle: `POST /optimization/session/{open,close}` (+ dynamics-comparison twins) ref-count ONE resident warm worker tied to the frontend tab (spawn costs ~1.2 s/call otherwise); no session → ephemeral worker per call; idle watchdog (`AEROVIZ_WORKER_IDLE_TIMEOUT_S`, 600 s) reclaims stranded workers. Frontend `workerSessionClient.ts` opens/closes per Pilot sub-mode + `navigator.sendBeacon` close on `pagehide`; all best-effort, the watchdog is the backstop.
- `casadi_lock.CASADI_LOCK` (RLock) serializes every in-process casadi entry point incl. `SimulationBackend.reset/step` (previously unlocked).
- `start_aeroviz_fullstack.sh` rewritten as the supervisor (see Gotchas). Spawn-picklable probes live in `isolation_probes.py`.

### 2026-06-28 — Scenario→optimization→CZML pipeline (scaffolds, since filled) + `flight_scenarios` seam

- `scenario_optimization.py` writes one `*_states.json` per scenario = `{source, final_time_s, optimizer_states[], simulator_states[]}` (states `{t,lat,lon,alt,V,psi,gamma,m}`); `optimizer_states` = NLP node states, `simulator_states` = controls rolled through `CasadiSimulator` (lives in the `aerodynamic_model` layer — never imports the backend above it). `build_scenario_comparison_czml.py` renders reference/optimizer/simulator as three coloured time-dynamic paths.
- `flight_scenarios` package: `scenario.py` (record + JSON round-trip + `aircraft_for_code`), `start_state.py` (track → initial state via two-sample finite difference), `build.py`, CLI `python -m flight_scenarios`. `final_state_from_track` populates `FlightScenario.target`.

### 2026-06-28 / 2026-06-27 — `geokit`: one geodesy/units source

- `geokit.constants`: WGS84 (`WGS84_A`/`_E2`/`_B`), `SPHERE_RADIUS_M` (default WGS84 a; switchable `EARTH_RADIUS_MEAN_M`), `NM_M`/`FT_M`/`KT_MS`/`METRES_PER_DEG_LAT`/`DEG2RAD`. `geokit.geodesy`: haversine, equirectangular, bearing, flat-distance, metres-per-degree, bounds. `geokit.units`: exact speed (`kt_to_ms` = 1852/3600, ft/min, km/h, mph) + length (`nm_to_m`, `ft_to_m`) conversions — replaced the truncated `0.51444` and 4 divergent Earth radii project-wide; `aircraft_sets.py` SI-mirror fields derive from it.
- Frontend: `geokit/scripts/export_constants_json.py` → `src/generated/geoConstants.json`, re-exported by `procedureGeoMath.ts` (~16 files migrated off local constants); drift-guard test fails if the JSON drifts. Fixed a real bug: two TS modules had used different haversine radii.
- Aero layer imports constants only (symbolic functions untouched). The 30 km study was regenerated for the ~0.1% WGS84-a shift (conclusions unchanged). Pedagogical flat-Earth `runway_bearing_rad` stays local by design.

### 2026-06-27 — Full (exact) geodetic transport as an explicit option

The geodetic RHS's ψ transport had silently dropped a cross term `V·sinγ·sinψ·cosψ·(1/(R_N+h) − 1/(R_M+h))` (~3–4 orders below the main meridian-convergence term; γ transport was already exact). Now `transport ∈ {"none","approx","full"}` on `make_geodetic_dynamics_model`/`make_geodetic_step_integrator`; `"approx"` = historical default (byte-identical). New `*FullTransport` (+ Normalized) schemes end-to-end (backend names `casadiDirectCollocationNormalizedFullTransport(+Trapezoidal/+Rk4)` etc.; frontend Dynamics options). Compare mode gains opt-in system F (full transport). `transport_term_comparison.py` + zh doc: divergence ~mm over 120 s, dt-independent (vector-field, not truncation). Decision: default stays approx; no silent approximations going forward.

### 2026-06-25 / 2026-06-24 — RNAV(GPS) tutorials + canonical ProcedureConstraint + CIFP block-altitude fix

- Tutorials: `aeroviz-4d/docs/34-how-to-read-rnav-gps-approach.{zh,en}.html` — self-contained, interactive (auto-wrapped glossary tooltips, SVG fix/segment hotspots, slide-in glossary panel), worked on KRDU RNAV (GPS) Y RWY 5L. Verified in-browser.
- **Canonical `ProcedureConstraint`** (front↔back): `src/data/procedureConstraint.ts` + Python mirror `aeroviz_backend/procedure_constraint.py` — one JSON shape (ordered waypoints with altitude windows + final course + glidepath + nominal speed); `buildProcedureConstraint(document, {branchId})`. CIFP→`AltitudeConstraint` conversion unified into one `altitudeConstraintFromCifp` (the two copies had diverged — one dropped block upper bounds).
- **CIFP block-altitude fix**: ARINC 424 "B" descriptor is a WINDOW (at-or-below Alt1, at-or-above Alt2); the parser had dropped Alt2. `ProcedureLeg.altitude_ft_2` added; KSTL regenerated. Chart-cross-referenced golden test (`test_krdu_r05ly_matches_published_rnav_gps_chart`) guards the parser against published-chart values. Documented gap: leg speed restrictions not extracted (cifparse exposes no speed field; 0 coded in the dataset) — `speedMaxKt` is ready when a source appears.

### 2026-06-24 / 2026-06-23 — Normalized geodetic scheme (conditioning fix)

- Root cause of Max_Iterations on H05LZ N=10 free-time solves: conditioning, not the seed — radian lat/lon (~1e-6 rad/s derivatives) next to metre/m-s states; the `1/(R_M+h)` factor makes position defect rows ~6–7 orders smaller than altitude rows.
- Fix: `*Normalized` schemes reparameterise the decision state to metres from the target anchor (`n=(lat−lat_t)·R`, `e=(lon−lon_t)·R·cos(lat_t)`) — an EXACT affine change of variables (unlike localEnu's flat-tangent approximation); same geodetic RHS inside the defect. Robust across N and arrival windows; identical trajectories on benign problems. The localEnu cold-start hybrid (an earlier workaround) was removed as superseded. Tutorial: `4dTrajectory/docs/geodetic_state_normalization.zh.md`. Compare mode gains system N (normalized) which overlays C — live proof the reparameterisation changes nothing.

### 2026-06-23 — Dynamics Compare mode (Pilot panel) + follow-ups

Third Pilot-panel mode: flies the start state under one constant control as the study's systems — A fixed-tangent ENU (anchored at the START in Compare mode; the 30 km study anchors at the target), B per-step re-anchored (reference), C geodetic RHS +transport, D no-transport, opt-in N (normalized) and F (full transport) — replayed as coloured, hideable CZML paths on Cesium's clock with deviation charts (horiz/alt/head/speed/fpa vs B) and a final-value table.
- Core extracted to `dynamics_comparison.py` `compare_dynamics(...)` (30 km study now calls it, output byte-identical). Endpoint `POST /dynamics-comparison/run` (`dynamics_comparison_backend.py`); shared `czml_common.py` (epoch/iso/document-packet) + `responseValidators.ts` back both backends/clients.
- Frontend: `dynamicsComparisonClient.ts`, `useDynamicsComparisonPlayback.ts` (loads CZML, drives the clock, hides systems, camera-follow), `DynamicsComparisonCharts.tsx` (draggable portal window — see the backdrop-filter gotcha). Per-system tinted aircraft models (`colorBlendMode: MIX`) oriented via `VelocityOrientationProperty`, wrapped in `makeStableVelocityOrientation` (a CallbackProperty returning the last valid orientation when HOLD extrapolation zeroes the velocity — otherwise the parked model snaps to a default attitude).
- Live State panel shows B's state (dense backend `samples` in the trajectory-play shape) + colored per-system delta chips interpolated from the chart (`interpolateComparisonDeltas`); `fpa` tracked end-to-end as an error metric. Trajectory Play reuses the same chip strip for a live Δ-vs-target readout (replaced the old Lat/Lon/Alt Error rows).
- Custom start state in Compare (RNAV fixes + runway select); run history persisted per run (`dynamics_comparison_history.py`, git-ignored dir) with backend-averaged history endpoints (common distance grid, shortest run's range) + frontend Average/Clear buttons.
- Review fixes: rollout never records sub-surface/non-finite samples (stops + truncation note via `requestedDurationS`); endpoint-inclusive `even_sample_indices`; double-checked-locked integrator cache; chart memoization; START preview hidden during comparison playback.

### 2026-06-23 — Optimizer = dynamics × fitting; shared stall model

- `_DEFECT_SCHEMES` entries are `(make_dynamics, make_defect)`: localEnu is a CONTINUOUS dynamics (fixed ENU tangent frame) collocatable with any fitting (defect converts geodetic nodes into the target-anchored ENU frame via `geodetic_state_to_enu_expr`); only `reanchoredEnu` stays shooting-only (per-step re-anchoring is discrete).
- Shared stall model `aero_params_for_aircraft(aircraft)` (mass-based Cl_max, A320 ≈ 2.7) used by optimizer AND playback — they had diverged (2.7 vs 1.5), replaying optimized trajectories ~1.6 km off.
- Cold-start hybrid (`cold_start_scheme`: fixed-time seed solved with a cheaper dynamics, free-time refines) was added with a whole-flow timing stderr line (`log_optimization_timing`: build/coldStart/freeTime/solve/playback/total) — the hybrid was later removed (superseded by normalization); the timing log remains.

### 2026-06-22 — Pluggable defect schemes; solver backend verdict; CZML playback; dense state

- **Defect schemes**: trapezoidal (order 2) / Hermite-Simpson (order 4, default) / RK4 (order 4, shooting) on the continuous geodetic RHS + `reanchoredEnu` (the playback integrator as a shooting defect). HS and RK4 are the same order — they differ in construction (implicit collocation vs explicit shooting). A stepper can be a shooting defect but NOT a polynomial collocation defect. `collocation_scheme_comparison.py` accuracy ladder: trapezoidal ~5 m vs HS/RK4 sub-metre. Frontend split the optimizer choice into Dynamics × Fitting dropdowns (`optimizerToParts`/`partsToOptimizer`/`validFittingsForDynamics`); legacy optimizer names remain valid on the backend. Tutorial `4dTrajectory/docs/collocation_schemes.zh.html` (interactive convergence demos).
- **`localEnu` scheme + 30 km study** (`dynamics_comparison_30km.py` + zh doc): fixed local-ENU @ target ≈ 335 m horiz error over 30 km, RHS-no-transport ≈ 145 m, full RHS +transport ≈ 0.03 m (validates RHS ≡ re-anchored). `make_local_enu_step_integrator(ref_geo)` reduces exactly to the re-anchored stepper when ref = current point.
- **Solver backend**: `solver_backend` switch (ipopt/sqpmethod) exists but sqpmethod is NOT usable — cold it bails instantly from linear-interp guesses; warm-started (needs duals, exact Hessian) it's still slower than cold IPOPT because CasADi's sqpmethod uses a dense active-set QP (~300× per-iteration cost, no banded OCP structure). IPOPT stays the only exposed backend; a real warm-start payoff would need acados/HPIPM. `solver_backend_benchmark.py` documents this.
- **Playback**: the optimized trajectory plays as backend-built CZML on Cesium's clock (`trajectory_playback.build_optimized_trajectory_playback` — rolls the N piecewise-constant controls once through the SAME geodetic integrator as the live sim, sub-mm match). Trail = one short polyline per sample interval with ms-precision availability (grows behind the aircraft), coloured by control segment (blue→red). The aircraft packet carries NO orientation — the frontend sets it from the sampled state with the live-Pilot convention (`headingPitchRollQuaternion`, heading −ψ, pitch γ+α, roll −μ). `useOptimizedTrajectoryPlayback.ts` drives the clock + throttled ~12 Hz readout sampling (`sampleTrajectoryAt`). Rollout truncates (not raises) on envelope exit — it's a viz aid. Manual Pilot mode unchanged (interactive).
- **Dense-state collocation**: control on N segments, state collocated on N·M sub-intervals (`sub_steps`; auto ~3 s, cap 16) — fixed the km-scale optimizer→playback mismatch (coarse discrete operator ≠ fine playback RK4). The multiple-shooting polish machinery was REMOVED (dense-state raw solutions are playback-consistent). Raising nSegments instead would refine control too → the "wrinkle" convergence pathology.
- **CIFP transition-altitude misparse**: initial fixes with no published crossing altitude were placed at the procedure-wide Transition Altitude (18000 ft) → infeasible starts. Parser no longer falls back to `trans_alt`; qualifier from `alt_desc`; an IF with no own altitude derives one by interpolating from the nearest published fix on the branch (`derivedInitialFixAltitudeFt`) rather than being dropped. Postmortem: `aeroviz-4d/docs/33-cifp-transition-altitude-misparse-postmortem.md`. (The ready-made cifparse/arinc424 packages had the same fallback bug.)

### 2026-06-21 — Geodetic continuous dynamics for direct collocation

Replaced the fixed-ENU transcription with one continuous geodetic RHS shared by optimizer and playback: `make_geodetic_dynamics_model` (point-mass RHS in `(lat, lon, h, V, psi, gamma)` radians; position kinematics via WGS84 `R_M`/`R_N`; transport terms on ψ̇/γ̇) + `make_geodetic_step_integrator` (RK4, degrees externally). Validation (`geodetic_vs_reanchored_error.py`): geodetic+transport tracks the re-anchored RK4 playback to ~0.3 mm over 5 km; without transport ~2.9 m drift. Interactive doc `geodetic_dynamics_transport.zh.html`.

### 2026-04-20 — OCS geometry + FAA DOF obstacle layer

- `ocsGeometry.ts` `buildFinalApproachOCS` implemented (primary trapezoid + two 7:1-slope secondary panels); `useOcsLayer` renders three semi-transparent polygons per route (FAF→threshold pairs from `procedures.geojson`; primary half-width from the route's tunnel descriptor, 150 m fallback); `ocsSurfaces` layer toggle. Altitudes read from LineString z-values (CIFP geometry alt); switching to MCA is a one-function change documented in `docs/03-ocs-geometry.zh.md` §5.6.
- `preprocess_obstacles.py` parses fixed-width DOF `.Dat` (haversine radius filter, default 20 km) → `obstacles.geojson`; `useObstacleLayer` renders type-coloured cylinders (`RELATIVE_TO_GROUND`) with AGL labels; `obstacles` layer toggle.

### 2026-04-19 — DSM terrain hook

`useDsmTerrainLayer` rewritten onto the preprocessed heightmap pipeline (`terrain/dsmHeightmapTerrain.ts`); returns `{status, metadata, provider, error}`; wired in `CesiumViewer` + demo page; `dsmTerrain` layer toggle.
