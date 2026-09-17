# ts_transformer reference — prediction paths

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new. Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

## The three prediction paths — this is the experiment

`prediction_output` decides whether dynamics is connected at all, and the answers are the
point of the package, not a migration in progress.

### P1 · `state` — the kinematic baseline

- **`state`** is the purely kinematic BASELINE: channels in, channels out, the only symbol it
  borrows from `aerodynamic_model` being the `GeodeticState` dataclass. Its predictions carry
  **no flyability guarantee** — speeds, turn rates, thrust and `Cl_max` are unchecked. That is
  the survey's "statistically plausible but unflyable" problem, and it is **deliberate**: it is
  what lets the learned component be measured on its own.

### P2 · `control` — bounded controls through the rollout

- **`control`** is the opposite: the model emits bounded controls and a differentiable RK4
  rollout of the shared point-mass equations turns them into the trajectory, so every prediction
  is dynamically admissible by construction.

### P3 · `closure` — FROZEN comparison arm, tracker deleted

- **`closure`** — **FROZEN 2026-09-09 (review §5)**: its two stored checkpoints load, predict
  and publish exactly as before; no NEW run may select it (`PREDICTION_OUTPUTS_AVAILABLE`,
  refused by `cli.common._refuse_unavailable_selection`). (Scene design P1.c, 2026-09-05; **a COMPARISON ARM since 2026-09-07** — the
  latent-intent design demoted it, and its P1.d tracker was DELETED 2026-09-07 with its BLOCKER
  unfixed: a nearest-node search that jumps legs on 0.6 % of flights) regresses 14 DECISION numbers — the join
  distance, a via pose in runway axes, K=4 slowness knots (the duration is their integral),
  K=4 height knots — and `outputs.closure.model.reconstruct` draws the trajectory in closed form
  (`outputs.closure.geometry.via_dubins` + `outputs.closure.profile`; velocities = tangent × ground speed).
  Training is L1 regression on per-flight LABELS fitted from the truth
  (`docs/p1_closure_oracle.py labels` → `closure_labels_path`), carried as the batch context
  and never an input; `predict --closure-from-labels` draws every flight from its label (the
  family's ceiling, `source.closureFromLabels`). Locks: `enu` chart, `normalized` horizon,
  `checkpoint_selection_metric=fixed-anchor-objective` (the loop never draws the path), no
  random anchors; a labels file must be the airport's own and cover the cohort (a run refuses
  one that covers no flight, prints the covered share otherwise). The drawn path has no
  dynamics of its own (22 % fully flyable); the P1.d tracker that flew it with the point-mass
  rollout (`outputs/control/constraints/closure_tracking.py`, `predict --closure-track`) is RETIRED —
  code DELETED 2026-09-07, its numbers (+10.5 m of ADE, 92 % fully flyable) kept as history in
  `docs/2026-09-06_closure_p1d_tracking_results.zh.md`. Do not rebuild it.

### P4.a · `plan` — plan-and-guidance, the lockstep, "no fix ahead"

- **`plan`** — the plan-and-guidance path (design v5, `docs/2026-09-09_plan_and_guidance_design.md`;
  step 3 built 2026-09-11): the network predicts the OPERATING PARAMETERS and the NEXT
  INSTRUCTION at any anchor (`outputs/plan/labels.TARGETS`: `T`, `V_mid`, `d_decel`,
  `V_final`, `h_capture`, `d_join`, the remaining path; the next fix ahead / across the
  anchor in runway axes, its heading on, speed, remaining path and height; the logit of
  "no fix ahead"), regressed L1 in each target's scale over the entries the track defines,
  and a deterministic guidance layer flies them, re-asking the head EVERY 30 s from the
  aircraft's pose and window (v5.1, `outputs/plan/forecast.fly_lockstep`: the route to the
  instruction in force is kept whole and tracked from the point reached, re-laid only when
  the order changed materially or the aircraft drifted; a step that executes its
  instruction — past the fix, on the heading given — is cut there; a whole group is stepped
  together, 0.05 s per flight against ~4 s one leg at a time). Two traps the lockstep
  found: re-laid every step from a mid-turn pose the turn-straight-turn builder flips its
  turn direction step after step; and "executed" must mean ON the outbound leg beyond the
  fix — the along-track test alone fires 20 km short of a base-turn fix. Labels are read per
  drawn anchor at batch time (`PlanContext`); the validation replay is the DRAWN single-step
  route (`draw_order`), the deployable forecast the rolled flight cut at the threshold.
  Locks: `normalized` horizon; no CTA yet. **An aircraft already on the final has no next
  fix either** — the flag is "no fix ahead", supervised on every sample; the first head,
  supervised off the final only, sent every established flight to a made-up fix (74 % of
  straight-in flights out of the corridor).

### P4.b · `plan` v5.2 — the head trained on rolled windows

**v5.2 (2026-09-11): the head is TRAINED ON
  ROLLED WINDOWS** (`outputs/plan/rolled.py`, design §9 step 3(e), §12.6): `run_ts.py
  plan_rolled_windows` flies every flight of a split in lockstep from L−1 and records, at
  EVERY step, the head's input window there (`rolled_history`) with the truth's target
  vector AT THAT STATE (`targets_at`, the one definition `targets_from_labels` reads at an
  observed anchor): the instruction the truth's queue holds (`TruthQueue` — executed once
  the aircraft is past the fix on its heading; a fix behind an aircraft on the final is
  none ahead, `behind_on_final`, the lockstep's own rule), the truth's arrival time less
  the time flown, the flight's own plan. `--policy truth` records the oracle's states,
  `--policy model` a head's own labelled by the truth (DAgger's round; `--extend` carries
  the previous table). `train --plan-rolled-windows-path T --plan-rolled-share p` replaces
  that share of each epoch's per-flight draws by one of the flight's rolled windows
  (`PlanContext.override`, through `batch(epoch_seed=)`; the coin has its OWN salt, so the
  draws it passes over are the observed law's); the table must cover every train AND
  val flight (no partial mode); the val split's rolled windows are scored every epoch
  (`plan_rolled_validation`, `describe_epoch`'s `rolled` line) BESIDE the observed
  objective, which still selects the checkpoint. Measured (§12.6, KRDU val, 60 s anchor):
  the re-asked head's vectored ADE 5391 → 3641 m at share 0.75 (established 55 → 94 %
  pooled; the once-per-leg head 3781, the ceiling 2184); the rolled val loss falls with the
  share while the observed objective stands still, 0.75 and 1.0 tied; DAgger's first round
  (the head's own states appended) did not help and taught it to name fixes past the cap.
  Two traps: the label's `remaining_m` at
  a flown state is the LOCKSTEP's coordinate (the anchor's remaining path less the path
  flown, re-synced at an executed fix), not the geometric path through the fix — at
  step 0 the two differ by the truth's own curvature before its first fix, so the rolled
  and observed populations would disagree about one target; and a substituted training
  sample carries ZERO truth-grid targets and weights, so a path whose loss reads `y` /
  `mask` cannot use `override`.

### P4.c · `plan` v5.3 — the order hold is an axis (default 1)

**v5.3 (2026-09-12): an ORDER HOLD in the lockstep is an
  AXIS, not the default** (`forecast.held_order`; `plan_oracle --hold-asks N
  [--hold-flips-only]`, `fly_lockstep(hold_asks=, hold_flips_only=)`; default 1 = adopt at
  once, v5.2's behaviour): measured at 2 on the share-0.75 head the head's fix WALKS between
  asks (766 m at the median, 2.7 km at p75), so two asks never agree within `RELAY_FIX_M`
  and the hold locked flights onto their step-0 fix — vectored established 86.5 → 60.2 %
  (flips only 79.7 %), design §12.7. Do not re-litigate without a head whose next fix is
  steady between asks. Every rolled flight and rolled table says which hold it flew under
  (`planHoldAsks` / `planHoldFlipsOnly`; the header's `hold_asks`, a pre-v5.3 table reading
  as `rolled.PRE_HOLD_ASKS` = 1).

### P4.d · `plan` — `run_ts.py plan_oracle_pair`, the one comparison instrument

**`run_ts.py plan_oracle_pair --base L=<dir> --arm
  L=<dir>` is how two plan-oracle artifacts are compared** — flight by flight, never two
  summary tables; its identity line is the refactor check, and a hold-1 re-roll differs
  from §12.6's artifact by ≤ 0.07 m of ADE because the head's float32 CPU forward is not
  bit-reproducible between runs (51 of 78 differing rows differ at step 0).

### P4.e · `plan` v5.3 — the K-component mixture head (the fan)

**v5.3 (2026-09-12,
  §9 step 3(g)): `plan_fan_components` = K ≥ 2 makes the instruction group a K-component
  diagonal-Gaussian MIXTURE** (`outputs/plan/model.py`: K means, K log σ as
  `FAN_LOG_SIGMA_MIN + softplus`, K logits; trained by `mixture_nll` in place of the group's
  L1 — **the `kinematic` component keeps its name and becomes a negative log-likelihood, often
  negative, comparable within a fan run only**). `values` carries the top-weight component so
  the rolled flight is unchanged in form; `fan_rows` is every component as a full target
  vector with its weight and σ. **A fan member is the fan ONE STEP DEEP**
  (`lockstep_model_policy(first_component=k)`: the first order from component k, every later
  one the top-1's — a component's index is not an identity across asks), refused with an
  order hold. The point head (K = 0) keeps its stored layout bit for bit (15 outputs; pinned
  in `tests/test_plan_fan.py`). `run_ts.py plan_fan_readout --checkpoint L=<ckpt> --anchor-s
  60` reads a fan head twice — single step (fix errors, 2σ coverage, per-component usage) and
  rolled (the top-1, the K members, a DISPLACED CONTROL fan of K: the top-1's first fix moved
  5 km in runway axes at K bearings, the schedule coordinate with it) — with the latent fan's
  `geometry_cell` and the top-1 a member of BOTH sets; only flights whose first order flew a
  fix are fanned. A fan's columns are read against the control's, never alone. **Measured
  (§12.8, two seeds): the mixture's TOP-1 beats the L1 point head by 550–600 m of vectored ADE
  at 60 s (3641 → 3087, 3439 → 2837; established +6 points; straight-in unchanged; the point
  head's own seed spread ~200 m) — the mixture objective is the plan head's recipe from here
  (`--plan-fan-components 4`; the config default stays 0 for the stored heads); the one-step
  fan's members are no better than a 5 km ring (minADE_4 2454 vs 2450, nearest-beats-top-1
  58 vs 79 %) and the 2σ coverage (98.7 %) is the alternatives' width, so the fan is NOT a
  deliverable in this form.**

### P4.f · `plan` v5.4 — the scheduler's assignment and the time closure

**v5.4 (2026-09-12, §9 step 4): the scheduler's ASSIGNMENT**
  (`strategy.Assignment(arrival_time_s, d_join_m)` — the arrival ABSOLUTE on the series clock;
  `lockstep_model_policy(assignments=)` / `rolled_predictions_lockstep(assignments=)` replace
  the head's `T_s` and `d_join_m` at EVERY ask, `assigned_order`; the leg order carries
  `assigned_arrival_s`). `fly_lockstep` then CLOSES THE TIME on the route in force at every ask
  (`guidance/timing.close_time`): the SPEED lever first — one factor on the plan's HELD speed
  (`held_speed_mps`: the instruction's at its fix; on a closing, where the schedule holds the
  anchor's own speed to the deceleration point, `scaled()` adds a new held speed reached at the
  deceleration rate — never `V_final`), bisected between the floor (`stall_floor_mps`, the stall
  margin × the 1 g stall speed at the flight's mass and altitude, or `V_final` if higher) and
  `SPEED_MAX_MPS`; then the PATH lever (`leg_route(stretch_m=)`: a closing goes through the
  plain builder's hold / dog-leg, an instruction leg flies its heading after the fix longer),
  sized at the FLOOR speed with a secant on the length ACTUALLY laid (`MIN_STRETCH_LAID_M`: the
  builder laid nothing more → the lever is exhausted); X = assigned remaining − closed time
  (`planUnabsorbedFirstS` / `planUnabsorbedLastS`, signed), `planSpeedFactorFirst/Last`,
  `planStretchM`, every step's record `closure`. **Three rules that cost real flights when
  missed**: the route time is integrated FROM THE PROGRESS POINT on a route in force
  (`route_time_s(start=)`; from 0 the time never fell as the aircraft flew and the closure
  pushed the speed to its maximum — every vectored flight derailed); the closure scales from
  the plan AS LAID (`FlightState.base_plan`), never the last closed plan (a factor on a factor
  compounds); and the assigned time is a TARGET, not the flight's budget — the budget is the
  later of the head's own time and the assigned one, so a flight the closure cannot bring
  forward lands late and reports dt with X < 0 instead of being cut short of the final
  (measured: vectored established 0.96 → 0.58 with the assigned time as the budget). The
  instrument is `plan_oracle --policy model --assign-time truth [--assign-time-offset-s S]
  [--assign-join truth]` — the truth's time and join as the assignment, an oracle form that
  READS THE FUTURE and says so (`assignment` in the artifact; the control line's `cta=given`
  convention); its summary gains the X distribution, the speed factor and the stretch share.

### P4.g · `plan` v5.5 — the pooled five-airport run and its tooling

**v5.5 (2026-09-12, §9 step 5): the POOLED five-airport run and its tooling.** `run_ts.py
  plan_cohort` writes the development cohort a random-anchor plan run needs (the train CLI
  refuses a run whose 20 s future contract covers fewer train flights than the locked split
  holds): the train CLI's own flags, the same `usable_series → split_by_flight →
  filter_training_cohort → window set` sequence, every train flight with no admissible anchor
  dropped, `<output-dir>/development_cohort.json` + `data_selection.json` (the REASON each
  excluded flight is out — a later `train` loads the cohort's keys only and rejects nothing);
  refuses a rolled-table path (a table follows a cohort) and checks the config before the
  track load. On KRDU alone it reproduces the hand-written `step3c_plan_head_full_cohort.json`
  exactly. **A rolled table needs a checkpoint to name its cohort, and the truth policy never
  reads the model: the bootstrap head of a pooled run needs ONE epoch, not 120** (the first
  pooled run spent 41 min of GPU on weights nothing read). `plan_oracle --by-airport` cuts the
  summary by the flight key's airport beside the strata (`summary_by_airport`); `plan_oracle_pair
  --common` pairs a pooled artifact against a single-airport one over the flights both hold
  (the base must be INSIDE the arm; the counts are in the header). The pooled table is encoded
  once per window set — six copies on five airports (`docs/code-health-followups.md` §30).

### P4.h · `plan` — a per-airport head's seed line scales with its cohort

**A per-airport plan head's SEED LINE scales with its cohort (2026-09-12, design §12.11's
  two-seed check)**: at seed 2024 the KSJC and KSTL heads move ≤ 30 m paired (means ≤ 240 m),
  the KSMF head +358 m paired (+1.3 km of vectored ADE mean, the rolled cap 2.8 → 19 %) and
  KMSY +215 — the two ~2.4k-flight cohorts. A single-seed own-vs-pooled margin there (+79 /
  +217) is NOT evidence; a per-airport claim on such a cohort needs both seeds
  (`step5b_<ICAO>_fan4_head_s2024` beside `step5b_<ICAO>_fan4_head`).

### P5 · control-path axes: latent intent and CTA conditioning

- **The control path also carries two AXES (2026-09-07, `docs/2026-09-07_latent_intent_design.zh.md`)**:
  `latent_dim > 0` puts a latent intent z on the control output (`outputs/control/latent.py`:
  q(z | future) in training only, a K-component mixture prior from the context, z reaches
  the controls AND the duration; inference decodes the prior's top-1; `predict
  --latent-samples K / --latent-random K / --latent-shuffle / --z-from-posterior` write
  `modes/`, `random/`, `shuffled/` and the z-oracle; z never enters a record); and
  `cta_conditioning=given` makes the given arrival time BE the duration (`predict
  --cta-offset-s` is the scheduler's counterfactual). Both READ THE FUTURE in their oracle
  forms and the run name says so (`control+z8`, `z=posterior`, `cta=given`) — never a
  prediction result.

### P6 · control-path axis: the duration head

- **...and a third axis, the DURATION HEAD** (2026-09-07,
  `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §三):
  `duration_head=quantile` replaces the scalar `FinalTimeHead` with
  `QuantileFinalTimeHead`'s five `config.DURATION_QUANTILES`, monotone by cumulative
  softplus. Its MEDIAN is the duration the rollout flies (`final_time_s` unchanged), all
  five go to `source.durationQuantilesS`, and `run_ts.py eta_calibration` turns them into a
  calibrated interval. `predict --cta-from-quantiles` then decodes each flight at its OWN
  quantile — the run name says `cta=self-q`, and that one IS a prediction result.
  **`duration_head=two-head` (B1.b, 2026-09-08) carries BOTH heads**: the point head drives
  the rollout duration exactly as `point` does and the quantile head emits only the
  published distribution. It exists because `B1_point_matched` showed the two gains come
  from different mechanisms and do not overlap — the PATH gain is the duration term's
  WEIGHT (point head at `final_time_loss_weight` 26: ADE 1248 vs native32's 1322) and the
  ARRIVAL-TIME gain is the quantile HEAD (MAE 23.9 vs 25.9 s pooled, 10.3 vs 13.6 s
  straight-in) — so under `quantile`, where the rollout flies q50, the head's path cost is
  forced onto the trajectory for nothing.

### P7 · scope decisions

Single-aircraft-only and deterministic point-prediction are scope decisions for all three (README).

### P8 · the two dynamics axes under the control path

Two orthogonal dynamics axes underneath the control path: `control_dynamics_model` ∈
`point-mass` | `first-order-lag` is the physics; `control_dynamics_backend` ∈ `reanchored-rk4` |
`scaled-transport-chart-velocity` is the state representation the long rollout carries (the
unscaled `transport-chart-velocity` was RETIRED 2026-09-07 — a measured regression that the
nondimensional variant replaced on 2026-08-02; `run_naming` still abbreviates it so the 13
stored 2026-07/08 configs keep their `_tcv` names). **The registry in
`outputs/dynamics/backends.py` is keyed by the PAIR.**
The lagged model *wraps* `transport_chart_rhs` — same force equations, stall handling, transport
term and chart projection — so it is the point-mass model plus three actuators, not a second
flight model.

### P9 · `prediction_output` = `segment-plan` — the two-tier L2 plan layer

(A row of the defaults table in the source; it is a prediction path, so it is indexed with the others. Text verbatim.)

**Two-tier L2 (2026-09-17, plan v2 §4): the plan layer as a fifth prediction path** (`outputs/segment_plan/`). The observed window is cut into K = (L − 1) / 15 coarse 30 s segments and every segment becomes one row of 27 features in RUNWAY AXES about the anchor (`features.SEGMENT_FEATURES`: start / end / direction / five path sub-samples / length, speed, heading change, plus the approach context measured from `target_chart`); `segment_plan_attention` picks the token axis — `channels` (one token per feature, its K-series through the vendored inverted embedding: the inversion the requirement names) or `segments` (one token per segment + a learned position) — and ONE learned query (`TokenPool`) reads the tokens out at d_model under both, so the arms differ in their tokens and nothing else (a flattened read-out gave the channels arm a 27·d_model head against K·d_model). The head decodes M = `segment_plan_segments` (10) segments in one shot: the position at each segment's end (along, across, up — `labels.runway_deltas`; the chart is `chart_deltas`), the logit of "arrived by its end" and the arrival fraction inside the arrival segment. **The arrival is the truth's end, `truth_duration_s`** — the one duration every path's head trains on (the supervision rows close at the threshold crossing, observed or fitted); a segment the truth never reaches has no position target (masked). Loss: masked L1 positions in `POSITION_SCALE` (`state`), BCE arrival bits (`terminal`), L1 arrival fraction (`final_time`), `kinematic` a stated 0. The encoder is the vendored iTransformer stack built in ONE place (`backbone.adapters.ITransformerEncoderStack`, the projector discarded), so `TSConfig` requires `model=itransformer`, the `enu` chart, `dt_s | 30`, `(seq_len − 1) % 15 == 0`, `checkpoint_selection_metric=fixed-anchor-objective` (the waypoints ARE the prediction) and refuses `use_norm` (the stack is read below the vendored instance norm — it would be recorded and never applied). Decode (`decode.decode_plan`): the arrival segment is the first whose probability clears `ARRIVAL_PROBABILITY` (0.5); a forecast carries the waypoints before it and then the THRESHOLD at the arrival time (`truncated_at_threshold`), or all M waypoints (`horizon_capped`) — every record says `segmentPlanArrivalSegment` / `segmentPlanArrivalProbability`. **The validation replay pads the plan to M rows with the threshold HELD at zero velocity, and says so** (`Replay.row_valid` → `raw_kinematic_metrics(row_valid=)`, the mask review C-10 removed while no live path padded); the common-grid report still holds the last waypoint flat past M × 30 s over the whole remaining approach and its cap rate cannot say so, so **the number to read is the epoch record's `segment_plan_validation` block** (`readout.py`: the ADE over [0, min(T_truth, M·30)], e(30k) per segment with n, the arrival confusion plan/truth within the span, the signed arrival-time error) — measured beside the objective, never selected on. Named `segment-plan · iTransformer · waypoints · segment-plan-v1 · … M=10 attend=segments`. The frontend mirror `EXPERIMENT_PREDICTION_OUTPUTS` carries the value. **Readout: `run_ts.py segment_plan_readout`** (every checkpoint from ONE common fixed anchor — the latest of their own — and the 12/8/6 km bins; the displacement at 60/120/180/300 s per stratum with the forecast's last row HELD past its end, only the truth's landing makes a lead absent; each checkpoint on its OWN split, paired over the flights both hold; a constant-velocity extrapolation built in) → **gate L2** in `two_tier_gates --segment-readout`: at 120 AND 180 s the vectored p50 at least `L2_SEED_LINE_M` (125 m) below EVERY whole-approach reference's over the same flights and the straight-in p50 not above it, both seeds. Arms: `docs/experiments/two_tier_l2_arms.json`. **E2E (S3): `tracker_lockstep --plan-head <segment-plan ckpt>` flies an L1 tracker under the L2 head's OWN waypoints** (`SegmentHeadWaypoints`, protocol A: `decode_series` on the rolled history at every ask, the K waypoints relative to the flown row, the head's arrival time as the first ask's horizon — the plan's span where it draws none; e_plan, the head's waypoints against `truth_waypoints` at the same flown time, recorded per ask beside the step error and never handed over); a head of the other token shape is refused; `--anchor-floor-index 60` reads every tracker at the L2 head's anchor (its 61-sample lookback does not fit before the L1 arms' 59; refused below a tracker's own anchor). Gate E2E = G3's criteria under the `segment-head` source. Tests: `tests/test_segment_plan.py`, `tests/test_segment_plan_readout.py`, the E2E block of `tests/test_tracker_lockstep.py`
