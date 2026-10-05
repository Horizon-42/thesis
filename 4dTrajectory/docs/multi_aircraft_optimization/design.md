# Multi-aircraft trajectory optimization: design

Style: ASD-STE100 (Simplified Technical English). Units: SI. Regulation values in feet or nautical
miles occur only in quoted text, with the SI value beside them.

## 0. Status

### 0.1 State

| Item | Value |
|---|---|
| Base commit | `dev-two-tier` `a804633e` |
| Code written | Branch `dev-optimizer-multi-aircraft`: T0 (`bbb18c52`, `5e3826f9`); T1–T6, M1 (§8, this commit). M2 (T8) not started |
| Companion document | `code_review.md` (findings F1 to F13). Step T0 of this design needs F4, F5, F6 (item 3) and F7 |
| Model of the scenario | The two-tier post-training "one aircraft commanded" (`ts_transformer/docs/two_tier/design/post_training.md` D29, D93) |
| Dependency on `ts_transformer` | Two pure modules, read-only, through one adapter (§6, §7). No other import. No shared output |
| Decision state | MD1, MD2, MD3 decided by the user (2026-10-05). MD4–MD6, MD9, MD10, MD12: built as recommended (Claude, within the rules). MD7, MD8: T5 measured (§8.1), the user decides. MD11: not built (open) |

### 0.2 Decisions

Each decision is open until the user decides. "Rec." is Claude's recommendation. A recommendation is not a
project rule. A decided row says "Decided" with the option, the person and the date.

| ID | Question | Options | Rec. |
|---|---|---|---|
| MD1 | Scope of the first stage | (a) M1 only: one optimized aircraft in recorded traffic. (b) M1 and M2 | **Decided: (b), M1 and M2** (user, 2026-10-05). M1 comes first in the plan, because M2 runs the M1 loop for each aircraft (§5.6) |
| MD2 | Which reading of the separation rules makes the constraint rows | (a) `VISUAL` makes the rows, `IFR` is reported beside it. (b) `IFR` makes the rows | **Decided: (a)** (user, 2026-10-05). It is the reading that the two-tier closed loop uses (`separation.py` docstring, user 2026-09-27, provisional). Then the two results compare |
| MD3 | How the optimizer gets the separation rules | (a) Read-only import of `ts_transformer.inference.separation` and `.runway_schedule` through one adapter. (b) Move the two modules to a neutral top-level package first | **Decided: (a)** (user, 2026-10-05). (b) stays a proposal (§7, R2): it needs a change in `ts_transformer` and the agreement of its stage owner |
| MD4 | Source of the recorded traffic | (a) The arrivals manifest (`arrivals/manifest.json`), every flight, not only the scenario sample. (b) The tracks roster with all outcomes | (a), built. Only (a) has the runway target that the datum conversion needs (`flight_scenarios/datum.py:63`) |
| MD5 | Scenario category for the first stage | (a) `runway_cons` (procedure-constrained). (b) `runway`. (c) both | (a), built. The "established" test reads the final approach geometry. The constrained solve already flies it |
| MD6 | Check step `h_c` of the judge on the replay | 1 s, 2 s, 4 s (the scene step Δ of the two-tier design, vocabulary D11) | 1 s, built, on UTC multiples (one grid for every aircraft). A 4 s result is not built yet |
| MD7 | Row window `W` (the time around a judged loss that gets rows) and the row margin `κ` | `W` in {30, 60, 120} s; `κ` in {0, 1, 2} % | T5 (§8.1): `W` = 15 s and 60 s gave the same outcomes on 50 KRDU windows. Built: `W` = 60 s, `κ` = 1 %, and a loss that comes back at the same instant asks for twice its margin. The user decides |
| MD8 | Largest number of loop iterations `K_max` | 3, 5, 8 | 5, built. T5: no window needed more than 2 re-solves (§8.1). The user decides |
| MD9 | Speed that changes a distance minimum into a time minimum (wake at the threshold, the M2 schedule) | (a) One approach speed per airport, as `runway_schedule.faa_separation(speed_mps=…)`. (b) The follower's published approach speed at its mass (K10 in `optimizer_reference.md`) | (b), built for M1: each window's minima are timed at the commanded aircraft's target speed. M2's schedule needs one speed per block (`faa_separation` takes one): open |
| MD10 | A window that starts in a loss (the record already has the loss at the first check step) | (a) No rows for that pair until its first step without loss; count the window. (b) Refuse the window | (a), built; the readout counts these windows |
| MD11 | Recorded aircraft with a faulty observed point (two-tier vocabulary D111) | (a) Count only: the summary reports the losses whose recorded aircraft has a jump at the loss step. (b) Mark with the D111 function | (a) is NOT built yet: no jump count. The D111 function reads the two-tier artefact (`instructions/faults.py:25`), not a track. (b) needs that function moved to a neutral module |
| MD12 | The runway in force of a recorded aircraft | (a) Its record's runway for the whole window. (b) As the two-tier design: only from its own first predicted step (vocabulary D23) | (a), built. The optimizer has no predicted steps. State the difference in every comparison |

## 1. Purpose and scope

### 1.1 Questions

1. **Q1 (M1).** The optimizer flies one arrival from its observed start to its runway. The other arrivals
   fly their records. Can the optimizer keep separation from them? What is the cost in flight time?
2. **Q2 (M2).** All arrivals of one time block get a landing slot and an optimized trajectory. How many
   losses of separation remain, and how much delay does the schedule add?
3. **Q3 (comparison).** For the same windows, how do the optimizer and the two-tier post-trained model
   compare (outcome, losses, flight time)?

### 1.2 Out of scope

- Go-around. The optimizer has no go-around. A recorded aircraft with a go-around flies its record,
  go-around included.
- Departures and overflights. They are not in the arrivals manifest (MD4).
- The choice of the runway. Each aircraft lands on the runway of its record.
- Wind. The dynamics have no wind.
- ATC speed instructions inside the FAF or 5 NM (9,260 m) (7110.65BB 5-7-1 b4). The optimizer does not
  model instructions. §10 item 4 lists this as an open question.

## 2. Terms

| Term | Meaning |
|---|---|
| Commanded aircraft | The aircraft that the optimizer flies. One per window in M1 |
| Recorded aircraft | An aircraft of the window that flies its observed record |
| Window | The time from the commanded aircraft's start to its threshold crossing (§3.2) |
| Baseline solve | The present single-aircraft solve of the commanded aircraft, without traffic |
| Replay | The optimizer's controls flown by the real simulator (`scenario_optimization.rollout_controls`) |
| Check step | One time of the judge's UTC grid, a multiple of `h_c` (MD6) |
| Judge | `ts_transformer.inference.separation.losses` applied to the aircraft at one check step |
| Loss | A pair of aircraft under its minimum at one check step (`separation.Loss`) |
| Responsible | The aircraft of a loss that the rules make responsible (`Loss.responsible`) |
| Established | On the final of its runway: inside the FAF and the LPV cone, track within 20° of the course (two-tier D92) |
| Approach clock | Position along the landing direction from the airport's common origin (`runway_schedule` module docstring) |
| Row | One constraint `lb ≤ expr ≤ ub` of the NLP (`collocation/optimizer.py:89-94`) |

## 3. Relation to the two-tier design

### 3.1 What this design takes

| Two-tier item | Here |
|---|---|
| D29: one aircraft commanded, the others fly their records | Mode M1 |
| D93: the window starts at the commanded aircraft's first state and ends when it is done or loses separation that it answers for | §3.2. The optimizer does not end a window at a loss. It removes the loss or reports it |
| D92: "established" | §4.3, same three conditions and the same 20° |
| The separation judge (`inference/separation.py`, readings `VISUAL` and `IFR`) | Same module, read-only (MD2, MD3) |
| D11, D25: scene step Δ = 4 s, UTC multiples | The check grid is UTC multiples of `h_c`. The 4 s result is reported (MD6) |
| Later, optional stage: every aircraft of a window commanded (`post_training.md` §2) | Mode M2 (§5.6) |

### 3.2 The window

- **Start.** The commanded scenario's initial state. Its UTC time is the scenario source field
  `entry_time_utc` (`flight_scenarios/build.py:154`). This is the time of waypoint offset 0 of the arrival
  slice (`trajectory_data_process/harvest/arrivals.py:142`, `:365-376`). A scenario without this field is
  refused.
- **End.** The commanded aircraft's threshold crossing: `t0 + Σ T_p` of the solve.
- **Other aircraft.** At each check step: every other flight of the airport in the arrivals manifest with a
  record at that time (MD4). A flight enters the scene at its first sample and leaves after its last
  sample (its threshold crossing). Between two samples, its position is the linear interpolation.

### 3.3 What is different, and why

| Item | Two-tier | Here | Why |
|---|---|---|---|
| How a loss counts | Reward 0 (D30). The policy learns to avoid it | A hard constraint row. A loss that remains makes the window `unresolved` | The optimizer has constraints, not rewards |
| Who flies the commanded aircraft | The prior speaks, the executor flies | The NLP and the real simulator (the replay) | — |
| The runway in force of a recorded aircraft | From its own first predicted step (D23) | The record's runway for the whole window (MD12) | The optimizer has no predicted steps |
| Time base | Rows on the 4 s grid | Free node times; the judge on a 1 s grid (MD6) | The NLP nodes are not on a grid |
| Population | The ts split (train, select, test days) | The optimizer's scenario population (a sample of at most 2,000 per runway, `prepare_scenario_inputs.py`) | `flight_scenarios/CLAUDE.md` "The ts and optimizer populations are different flights". For Q3 a window list joins them (§6.5) |

## 4. Data and geometry

### 4.1 Recorded traffic

1. Read the arrivals manifest of the live harvest root with `flight_scenarios.build.load_model_arrivals`
   (`flight_scenarios/build.py:328`). This converts the altitude from HAE to MSL with the runway's offset
   (`flight_scenarios/datum.py:79`). The commanded scenario came through the same function, so the two
   altitudes have the same datum.
2. Do not read `trajectory_data_process/outputs/harvest-heldout/` (KAUS, held-out test data).
3. Make an interval index in memory: per flight, `[entry_time_utc, entry_time_utc + last offset]`.
   Do not use `trajectory_data_process.scene_index`: on first use it writes `tracks/scene_index.json`
   into the live harvest (`scene_index.py:141`). This design writes nothing outside its own output
   directory.
4. The waypoint rows are `[offset_s, lon_deg, lat_deg, alt_m]` (`flight_scenarios/start_state.py:62`).
5. The aircraft type of a recorded flight comes from the same aircraft resolution that the scenarios use
   (`flight_scenarios/CLAUDE.md` "Aircraft resolution"). Its CWT category comes from
   `runway_schedule.wake_category` (`runway_schedule.py:260`). A flight with no type gets category `None`.
   The judge then applies the radar minimum only and marks the loss `wake_known = False`
   (`separation.py:122`). The summary counts these losses.

### 4.2 The frame

- The commanded NLP state is metric, anchored at its target (the LTP):
  `n = (lat − lat_t)·R`, `e = (lon − lon_t)·R·cos(lat_t)`, with `R = geokit.WGS84_A`
  (`collocation/schemes.py` `_normalization_cb`).
- Convert each recorded position into this frame with the same formula. Then a recorded position and a
  commanded node are in one frame, and a separation row is a function of the decision variables only.
- The judge gets the positions in the same frame (`Traffic.e_m`, `Traffic.n_m`, `separation.py:91`).

### 4.3 The judge's inputs per aircraft

`separation.Traffic` needs eight values per aircraft. This design computes them as follows.

| Field | Commanded aircraft | Recorded aircraft |
|---|---|---|
| `e_m`, `n_m`, `height_m` | Replay state at the check step | Interpolated record |
| `runway` | Its scenario runway | Its record's runway (MD12) |
| `along_m` | `Separation.along_nm[runway]·NM_M` less the distance to the threshold along the course | Same |
| `track_minus_course_deg` | Replay track less the runway course | Track from two consecutive samples, less the course |
| `right_of_course_m` | Signed distance right of the extended centreline | Same |
| `established` | D92: inside the FAF and the LPV cone of its runway, and track within 20° of the course | Same |
| `category` | CWT of its `dynamics_typecode` | §4.1 item 5 |

- The LPV cone and the FAF come from the procedure geometry: `approach_constraints.lateral`
  (`fac_distance_to_ltp`, `fac_cross_track`, `lpv_corridor_violation`, `lateral.py:142-193`) and the
  procedure bridge of `code_review.md` F4. The cone formula is defined once in
  `flight_scenarios/fas_geometry.py`.
- The course is a compass bearing in the manifest (`runway_schedule.parallel_relations` docstring) and
  math-ENU in the optimizer (`optimizer_reference.md` O3). Convert once, in the adapter (§6.2). A test
  checks one runway of each airport in both conventions.
- D92 is a two-tier decision that this design reads. If the two-tier code gets a public function for it,
  replace this computation with that function (§7, R3).
- The region of D92 is the LPV cone from the FAF to its apex at the GARP: it continues past the threshold.
  A replay ends some centimetres past the threshold, and the aircraft must stay established there. (A
  lower bound at the threshold made the commanded aircraft "not established" at its own landing and
  responsible for an arrival on a crossing runway: T5 found it.)
- A runway without an RNAV(GPS) procedure has no coded final, so nobody is established on it. A runway
  with a procedure whose FAF does not read is an error, not a missing final.
- The CWT category comes from the identity resolver's ICAO type. A type that the CWT tables do not list
  gets `None` (the judge then applies the radar minimum alone) and is named in the sidecar.

### 4.4 Frame error

The frame is a local equirectangular projection. Its east scale is exact only at the target latitude. The
relative error of an east distance is about `tan(lat)·Δlat`, with `Δlat` the latitude offset from the
target in radians. At 36° N and 40 km from the threshold, this is 0.46 %: 26 m on 3 NM (5,556 m). The
row margin `κ` (MD7) must cover it. The summary states the largest `tan(lat)·Δlat` of each batch.

## 5. Method

### 5.1 Overview

```
               commanded scenario + recorded traffic (§4)
                               │
   L1  baseline solve  (present optimizer, no traffic)
                               │
   L2  replay ─► judge on the check grid (VISUAL rows, IFR reported)
                               │
            no loss with the commanded aircraft responsible ──► outcome "separated"
                               │ else
   L3  add rows for each judged loss (§5.3) ─► re-solve, warm start ─► back to L2
                               │ after K_max iterations or a failed solve
                         outcome "unresolved" / "solve_failed"
```

M2 adds level L0 before L1: a landing schedule (§5.6).

### 5.2 L1 and L2

1. Solve the commanded scenario with the present optimizer and the present configuration
   (`optimize_scenario_*_iaf` for `runway_cons`). Use the record of the existing batch when its
   `optimization_config` and its target agree (the resume rule, O9). Do not overwrite that record.
2. Replay the controls (`rollout_controls`). The replay is the trajectory that `evaluation` grades. The judge
   reads the replay, not the plan.
3. At each check instant, build `Traffic` for the aircraft present and call `separation.losses`, and
   `wake_at_threshold` behind each aircraft that is over its threshold at that instant. The instants are the
   UTC multiples of `h_c`, the window's start and end, and every landing in the window. The judge sees the
   recorded aircraft within 30 km of the commanded one: every minimum is shorter than half of that (the
   largest, 8 NM = 14.8 km), and an aircraft between two others on one final is near both, so the cut
   changes no loss that involves the commanded aircraft.
4. Keep the losses with the commanded aircraft in `Loss.responsible` (two-tier D93: "a loss of separation
   that it answers for"). Count the other losses that touch the commanded aircraft. Do not make rows for
   them. Do not count losses between two recorded aircraft as the optimizer's losses. Report them as
   background.

### 5.3 Constraint rows

The judge decides the kind of each loss. The NLP gets a smooth row for that kind. The rows against one
recorded aircraft use one branch for the whole window. The branch is horizontal when any loss with that
aircraft is in trail. Else it is the larger-margin branch at the tightest instant. It changes only from
vertical to horizontal, when an in-trail loss appears later. (A branch chosen at each check instant gave
one node contradictory rows, above and below, or both branches: review finding, fixed.)

| Loss kind (`separation.py:82`) | Row on the commanded node `k` and the recorded aircraft `j` | Notes |
|---|---|---|
| `in_trail` | `(n_k − n_j(t_k))² + (e_k − e_j(t_k))² − (S·(1 + κ))² ≥ 0` | `S = Loss.required_m` |
| `diagonal`, `radar_or_vertical` (the rules allow horizontal OR vertical separation) | Branch frozen when the row is added. Horizontal branch: the row above. Vertical branch: `σ·(h_k − h_j(t_k)) − V·(1 + κ) ≥ 0`, with `V` = 304.8 m (1,000 ft, 7110.65BB 4-5-1 a) and `σ` the sign of `h_k − h_j` at the judged step | The branch is the one with the larger margin at the judged step: horizontal distance over `S`, or vertical distance over `V` (the `margin` of `separation.Judged`, `separation.py:162`) |
| `at_threshold` (TBL 5-5-2 wake, the leader over its threshold) | `t0 + Σ T_p − τ_j − g ≥ 0`, with `τ_j` the leader's threshold time and `g` the minimum time | Only when the commanded aircraft is the follower. `g` = distance over the speed of MD9, on the approach clock (`Separation.gap_s`, `runway_schedule.py:114`). The row is linear in the phase durations |

- **The disjunction is an approximation.** The rules allow either branch. A frozen branch forbids the
  other one for that row. The record of the window states the branch of each row.
- **Node times.** The time of node `k` in phase `P` is `t_k = t0 + Σ_{p<P} T_p + (i/N_P)·T_P`. It is affine in
  the phase durations. `code_review.md` F7 makes it available to the caller.
- **Recorded positions at a node time.** `n_j(t_k)`, `e_j(t_k)`, `h_j(t_k)` are `casadi.pw_lin` of the
  recorded track in the commanded frame, at the symbolic `t_k`, over a slice that reaches two row windows
  beyond the losses. T2: an inline casadi interpolant cannot take an `SX` argument (casadi 3.7.2:
  "eval_sx not defined"); `pw_lin` can, its values equal `np.interp`, and its derivatives agree with finite
  differences. About 10 `SX` nodes per knot.
- **An aircraft that is not in the air.** Before its first sample (its entry into the arrival slice) and after
  its last (its landing) the recorded aircraft holds its end position, and a presence weight `a(t)` ramps
  from 0 to 1 over 10 s and relaxes the row: `gap² + S'²·a ≥ S'²`, `σ·(h − h_j) + 2·V'·a ≥ V'`. A re-solve
  can then wait for a landing. (An extrapolated track kept a landed aircraft flying; a ramp that moved it
  away swept it across a final: both found and replaced.)
- **Which nodes get rows.** The nodes whose time in the last solve is inside `[τ − W, τ + W]` around a judged
  loss at time `τ` (MD7). One node gets one row per family and aircraft (the largest bound). Rows
  accumulate over the rounds of one window. A loss that comes back at the same instant asks for twice its
  margin in the next round: rows hold only at the nodes, and between nodes (or through replay drift) the
  record can still be some metres short.

### 5.4 L3: the loop

For one window:

1. `i = 0`. Do L1 and L2. If no loss has the commanded aircraft responsible, the outcome is `separated_at_baseline`.
2. Add the rows of §5.3 for each such loss.
3. Solve again. Use the last solution as the initial guess (`initial_guess`, `optimizer.py:320`).
   Keep the objective of the baseline (minimum time).
4. If the solve fails, the outcome is `solve_failed`. Keep the IPOPT status, and keep the last good solve
   as the window's record.
5. Replay and judge (L2). If no loss has the commanded aircraft responsible, the outcome is `separated`.
6. `i = i + 1`. If `i = K_max` (MD8), the outcome is `unresolved`. Else go to step 2.

A pair in loss at the first check step gets no rows until its first check step without loss (MD10). The
window then has the flag `starts_in_loss`.

Each window record has: the outcome, the rounds (the flight time, every loss with whether it counted, the
rows of the next solve, its solve time or its error), the branch per aircraft, the IFR losses at the
baseline and at the end, the recorded aircraft, MD10's start instants, the types without a category, the
runways without a coded final, and the largest frame error. A baseline that fails raises
`BaselineFailed`: the batch writes a failed record, and the readout counts it as `baseline_failed`.

### 5.5 Why a loop and not all rows at once

A row for every pair at every node makes the NLP large: about 20 recorded aircraft (an assumption, to
measure in step T4) times 100 to 300 nodes per window. Most of these rows can never bind. The judge also
needs discrete facts (the runway relation, "established", who is responsible) that a smooth NLP cannot
decide. The loop gives the discrete part to the judge and the continuous part to the NLP. This is the
method of lazy constraint generation.

### 5.6 M2: a block of arrivals

1. **Block.** All arrivals of one airport whose windows overlap one time interval (for example 1 h). All of
   them need a scenario: prepare the block with `--max-per-runway 0` for that interval only.
2. **ETA.** The baseline threshold time of each aircraft: `t0 + T_baseline`.
3. **L0 schedule.** `runway_schedule.schedule` (`runway_schedule.py:363`) with one `Arrival` per aircraft:
   its record's runway only, `logp = 0`, its category, its ETA; order `fcfs_by_eta`. The result is one
   slot (a controlled time of arrival, CTA) per aircraft. The speed of the time minima is MD9.
4. **Fixed-time solve.** Each aircraft flies to its CTA: `optimize_trajectory(duration = CTA − t0)`.
   A CTA that the procedure corridor cannot absorb fails the solve. Count these aircraft with their delay.
5. **Priority order.** Solve the aircraft in slot order. For aircraft `m`, the traffic is: the replays of
   the aircraft before it in the order, and the records of the flights outside the block. Run the loop of
   §5.4 with this traffic.
6. **Final check.** Judge all replays of the block together. Report every loss that remains.

### 5.7 M3: one joint NLP (not designed)

One NLP for all aircraft of a conflict cluster. It needs common node times for all aircraft, so the phase
durations become fixed or shared. Design it only if M2 leaves losses that a sequential order cannot
remove. §10 item 2 records the open questions.

## 6. Decoupling from `ts_transformer`

### 6.1 Rules

1. **New package.** `4dTrajectory/optimization/traffic/`. No module of `ts_transformer` imports it.
2. **One import point.** Only `traffic/rules.py` imports from `ts_transformer`, and only
   `ts_transformer.inference.separation` and `ts_transformer.inference.runway_schedule`. Both are pure:
   numpy, stdlib, `geokit`; no torch (`runway_schedule.py:15` docstring; `separation.py` imports).
3. **Architecture test.** A test in `optimization/tests/` reads the imports of every module in `traffic/`
   (with `ast`) and fails on any other `ts_transformer` import or on `torch`.
4. **Behaviour pins.** Tests with fixed inputs pin the adapter's results: a set of `Traffic` cases and their
   expected `Loss` lists, one case per loss kind and per reading. A change of the rules on the two-tier side
   then fails a test here. Do not pin a source hash (root `CLAUDE.md`: compare behaviour on fixed inputs,
   never an equal commit or source hash).
5. **No shared writes.** The package writes only into its own output directories (§9). It writes nothing
   into the harvest, the ts outputs or the existing optimizer category directories.
6. **No change to `ts_transformer`.** A change that this design needs there goes to the user as a request.
   The user forwards it to the stage owner.
7. **Import cost.** `runway_schedule` imports `ts_transformer.repo_layout` (`runway_schedule.py:39`), which
   imports `trajectory_data_process.harvest.generations`. This loads the `ts_transformer` package root but no
   torch. The architecture test also checks that `import traffic.rules` does not load `torch`
   (`sys.modules`).

### 6.2 The adapter `traffic/rules.py`

It holds every conversion between the two sides, and nothing else:

- builds `Separation` with `faa_separation(targets, speed_mps=…)` from the manifest runway targets
  (`runway_schedule.py:269`),
- builds `Traffic` from the values of §4.3, with the course conversion of §4.3,
- calls `losses` and `judged_pairs` for one reading,
- maps a missing type to category `None` (§4.1 item 5).

### 6.3 Dependency direction

```
traffic/ ──► collocation/, approach_constraints/, procedure bridge (F4)
         ──► flight_scenarios/ (scenarios, load_model_arrivals, identity, fas_geometry)
         ──► evaluation_export, batch driver (F6 item 3)
         ──► traffic/rules.py ──► ts_transformer.inference.{separation, runway_schedule}   (read-only)
```

### 6.4 Failure modes that the rules prevent

| Failure | Rule that stops it |
|---|---|
| A ts refactor breaks the optimizer silently | Behaviour pins (rule 4) fail in the optimizer suite |
| The optimizer imports torch through a ts module and slows the solver workers | Architecture test (rules 3, 7) |
| An optimizer run overwrites a ts artefact or the harvest | Rule 5 and own output directories (§9) |
| A ts campaign runs from the main tree while this code changes | This code is developed in its own worktree and branch (project memory "campaigns own the main tree") |

### 6.5 The seam for Q3

The only connection for the comparison is a data file: a list of windows by `flight_key` and start UTC
time. The two-tier side writes it only when the user orders it. The optimizer reads it as input. No code of
one side calls code of the other side for Q3.

## 7. Reuse

| ID | Code | Verdict | Reason |
|---|---|---|---|
| R1 | `ts_transformer/inference/separation.py` (judge) and `runway_schedule.py` (minima, CWT, schedule) | Reuse now, read-only, through §6.2 (MD3, decided) | Pure, tested (`tests/test_separation.py`, `tests/test_runway_schedule.py`), every value cited to 7110.65BB. A second copy would break the single-source rule |
| R2 | The same two modules moved to a neutral top-level package (for example `separation_rules/`) | A proposal only (MD3 decided (a)) | Removes the only `ts_transformer` import. Needs a change in `ts_transformer` and the stage owner's agreement |
| R3 | The two-tier "established" function (D92) | Later, when it is public | Today this design computes D92 from its own procedure geometry (§4.3). Two computations of one rule can drift. A behaviour test that compares them on fixed rows removes the risk |
| R4 | `ts_transformer/instructions/faults.py` (D111) | Not now (MD11) | It reads the two-tier artefact (`FlightSignals`), not a track. Its array part could move to a neutral module |
| R5 | `collocation.CollocationOptimizer`, `approach_constraints` | Reuse, with the F7 extension | The separation rows use the same row contract as the procedure rows |
| R6 | `rollout_controls`, `evaluation_export.evaluation_record` | Reuse unchanged | The commanded record keeps the evaluation contract, so `evaluation` grades it unchanged |
| R7 | `_run_batch` (pool, resume, sweep, summary) | Reuse after F6 item 3 | The traffic batch needs the same mechanics with a different worker and record writer |
| R8 | `trajectory_data_process/scene_index.py` | Do not reuse | It writes a cache into the live harvest on first use (§4.1 item 3) |
| R9 | `flight_scenarios/scene_context.py` | Do not reuse | It selects neighbours for an encoder (120 s, 40 km, at most 16). This design needs every aircraft present |

## 8. Plan

Each step has a gate. A step starts after the gate of the step before it passes. Code goes on its own
branch and worktree, with a review before each commit.

| Step | Work | Gate | Needs | State |
|---|---|---|---|---|
| T0 | Prerequisites: F7 (`extra_rows`, node times), F4 (procedure bridge out of the backend), F5 (one constrained solve function), F6 item 3 (batch driver module) | All optimizer and backend tests pass. A fixed set of scenarios (20 `runway`, 20 `runway_cons`) solves bit-identical before and after | User accepts F4 to F7 | Done: `bbb18c52`, `5e3826f9`; 40/40 solves bit-identical, 21 batch files byte-identical |
| T1 | Traffic assembly (§4.1, §4.2): interval index, frame conversion | Test: the commanded flight's own record, read as traffic, gives its scenario's initial position at `t0` within 1 mm in the frame and in MSL | T0 | Done: 200 KRDU scenarios, largest gap 0.000 mm horizontal and vertical |
| T2 | Spike: inline casadi interpolant in `SX` at a symbolic node time | Derivative equals a finite difference within 1e-6 relative on 10 random tracks. If not: user decision (§10 item 1) | T0 | Done: inline interpolant fails in `SX`; `pw_lin` passes (§5.3) |
| T3 | Adapter (§6.2), architecture test, behaviour pins, the inputs of §4.3 | Pins pass; the course check passes on one runway per airport | MD12 | Done: `traffic/rules.py`, `traffic/tests/test_rules.py` (pins, import boundary, no torch) |
| T4 | Baseline census: judge the baseline replays of a seeded sample of windows (no rows) | A readout to the user: windows with a loss that the commanded aircraft answers for, by reading, kind and airport; recorded aircraft per window; the frame error of §4.4 | T1, T3, MD5, MD6 | Done inside T5 (the baseline round of each window) |
| T5 | Rows and loop (§5.3, §5.4) on 50 windows with a loss | A readout: outcome counts, iterations, rows, flight time change, solve time per window, memory. The user then decides MD7 and MD8 | T2, T4 | Done (§8.1); the user decides MD7, MD8 |
| T6 | Traffic batch, records, resume (§9) | Full optimizer suite passes. Preflight on the real size: time and memory per window measured, disk estimate checked before the start | T5, MD7, MD8 | Done: `traffic_optimization.py`, `run_batch` sidecars; the preflight at the real size is open |
| T7 | Evaluation of the commanded records and the CZML comparison (a new category) | The frontend shows a published window; checked in the browser | T6 | Not started (frontend: a sonnet agent) |
| T8 | M2 (§5.6) | A readout like T5 for 5 blocks | T6, MD9 | Not started |
| T9 | M3 design (§5.7) | Only if T8 leaves losses that the order cannot remove | T8 | Not started |

### 8.1 T5 readout (KRDU, seed 11, 50 windows; the reviewed M1 code, scratch output)

- Cost: 50 windows on 8 workers in 1 min 31 s wall time; the largest resident memory is 4.2 GB (the parent, which reads the manifest once). The traffic of KRDU is 24,202 arrivals, read in 21–33 s.
- Outcomes: 43 `separated_at_baseline`; 3 `separated` after 1 or 2 re-solves (the landing 34 s to 43 s later); 4 `solve_failed` (IPOPT `Maximum_Iterations_Exceeded` on long "radar or vertical" losses: §10 item 1). No window needed more than 2 re-solves.
- Windows with a counted VISUAL loss: 7 at the baseline, 4 at the end (the 4 failures). Windows with an IFR loss that the commanded aircraft answers for: 17 at the baseline, 15 at the end (IFR is stricter and makes no rows, MD2). Windows with a VISUAL loss that it does not answer for (no rows, §5.2 item 4): 2 and 2.
- No window starts in a loss (MD10). No type without a CWT category. 92 loss instants between recorded aircraft near the commanded one at the baselines (background). The largest frame error is 0.30 %, inside κ = 1 %.
- `W` = 15 s and `W` = 60 s gave the same outcomes (before the margin doubling). Before the review fixes of §5.3 the same 50 windows gave 1 separated and 6 failures.
- The outputs are in a scratch directory, not in `4dTrajectory/outputs`: a measurement of the method, not a published result.

## 9. Outputs and records

- **Runner.** `4dTrajectory/optimization/traffic_optimization.py --airport <ICAO> --sample N --seed S
  --output-dir …` (a seeded sample of the airport's arrivals, stated in `summary.json`); the readout is
  `python -m traffic.readout <dir>` from `4dTrajectory/optimization`.
- **Directory.** `4dTrajectory/outputs/<ICAO>/traffic_m1_<category>/` (M1) and
  `4dTrajectory/outputs/<ICAO>/traffic_m2_<category>/` (M2). The existing directories
  `runway`, `fitted_adsb`, `runway_cons` stay unchanged.
- **Commanded record.** `*_states.json` and `*_eval.json` in the existing evaluation contract
  (`evaluation_export.py`), so `evaluation` grades them unchanged. The filename stem is `flight_key`.
- **Traffic sidecar.** One `*_traffic.json` per window: the outcome, the flags, the iterations, the rows,
  the losses per iteration and reading, the recorded aircraft (by `flight_key`), the flight times. Its schema
  name is `optimization-traffic-v1`. Settle every field before step T6 starts. A later change gets a new
  schema name; no reader accepts two versions (root `CLAUDE.md`, the compatibility rule).
- **Summary.** `summary.json` per directory: the configuration (with `h_c`, `W`, `κ`, `K_max`, the reading),
  the outcome counts, the counts of §5.2 item 4, §4.1 item 5, MD10 and MD11, and the largest frame error.

## 10. Open questions

1. **Re-solves that hit the IPOPT cap.** T5: 4 of 7 windows with a counted loss end `solve_failed` with
   `Maximum_Iterations_Exceeded`, all on long "radar or vertical" losses. Options: (a) soft rows (a slack
   per row and a penalty in the objective: needs an objective hook in the optimizer); (b) on a failure,
   try the other branch once; (c) accept and report. Not decided.
2. **M3.** Common node times for all aircraft; disjunction branches for pairs that change order; the size
   of a conflict cluster. Not designed.
3. **The objective in traffic.** M1 keeps minimum time. A delay then appears only where a row forces it.
   A controller would also give speed instructions earlier. The present objective does not model this.
4. **Speed instructions inside the FAF or 5 NM.** 7110.65BB 5-7-1 b4 forbids speed instructions there. The
   optimizer can still change its speed there. A row that holds the speed inside that region is possible.
   It is not in this design.
5. **Losses that the commanded aircraft does not answer for** (§5.2 item 4). The optimizer could also avoid
   them. This design does not, to match two-tier D93. The counts show if this choice matters.

## 11. Key code index

| What | Where |
|---|---|
| The optimizer, its build and the row contract | `4dTrajectory/optimization/collocation/optimizer.py:89-94`, `:320` (`optimize_free_time`), `:362` (`optimize_trajectory`), `:390` (`_build`), `:524-545` (row dispatcher), `:173` (`dense_node_times`) |
| The metric normalization (the frame) | `4dTrajectory/optimization/collocation/schemes.py` `_normalization_cb` |
| Final approach geometry | `4dTrajectory/optimization/approach_constraints/lateral.py:142` (`fac_cross_track`), `:150` (`fac_distance_to_ltp`), `:177` (`lpv_corridor_violation`) |
| Constrained solve, replay, batch | `4dTrajectory/optimization/scenario_optimization.py:359` (`_solve_iaf`), `scenario_replay.py:141` (`rollout_controls`), `scenario_batch.py:134` (`run_batch`) |
| Procedure bridge (moved in F4) | `4dTrajectory/optimization/procedure/segments.py:104` (`build_constraint_segments`), `procedure/constraint.py:170`, `procedure/iaf.py` |
| Scenario, identity, datum | `flight_scenarios/scenario.py:131` (`FlightScenario`), `flight_scenarios/identity.py:39` (`flight_key`), `flight_scenarios/build.py:328` (`load_model_arrivals`), `flight_scenarios/datum.py:79` (`flight_to_msl`) |
| Arrival slice time origin | `trajectory_data_process/harvest/arrivals.py:142`, `:365-376` |
| Separation judge | `4dTrajectory/ts_transformer/inference/separation.py:91` (`Traffic`), `:122` (`Loss`), `:162` (`Judged`), `:185` (`judged_pairs`), `:231` (`losses`), `:255` (`wake_at_threshold`) |
| Minima, CWT, schedule | `4dTrajectory/ts_transformer/inference/runway_schedule.py:63` (`Separation`), `:183-222` (constants), `:260` (`wake_category`), `:269` (`faa_separation`), `:313` (`earliest_time`), `:363` (`schedule`), `:386` (`violations`) |
| The traffic package | `4dTrajectory/optimization/traffic/`: `scene.py` (recorded traffic), `frame.py`, `runways.py` (D92 geometry), `rules.py` (the only `ts_transformer` import), `check.py` (the judge on a replay), `rows.py`, `loop.py` (M1), `readout.py`; the runner `traffic_optimization.py` |
| Two-tier decisions read here | `4dTrajectory/ts_transformer/docs/two_tier/design/post_training.md` D29, D30, D92, D93; `vocabulary.md` D11, D23, D25, D111 |

## 12. Regulation sources

All values come from FAA JO 7110.65BB Change 3 (2026-07-09), as encoded in `runway_schedule.py:159-222` and
quoted to the paragraph in `docs/literature/arrival_separation/README.md` §2 (§2.1 to §2.9). This design adds
no regulation value. The values that the rows use:

| Value | SI | Paragraph |
|---|---|---|
| Terminal radar minimum 3 NM | 5,556 m | 5-5-4 a, b |
| Vertical minimum 1,000 ft | 304.8 m | 4-5-1 a |
| Reduced radar minimum 2.5 NM (by authorization only; not the default) | 4,630 m | 5-5-4 j |
| Parallels under 2,500 ft count as one runway | 762 m | 5-5-4 h NOTE |
| Dependent parallels, diagonal 1.0 NM / 1.5 NM | 1,852 m / 2,778 m | 5-9-6 a2, a3 |
| Wake minima by CWT pair (directly behind / on approach) | `CWT_DIRECTLY_BEHIND_NM`, `CWT_ON_APPROACH_NM` | TBL 5-5-1, TBL 5-5-2 |
| Established: track within 20° of the course inside the FAF (two-tier D92) | — | 5-9-2 a, TBL 5-9-1 |
| No speed instructions inside the FAF or 5 NM | 9,260 m | 5-7-1 b4 |
