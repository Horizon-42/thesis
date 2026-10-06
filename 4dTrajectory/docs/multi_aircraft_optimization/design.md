# Multi-aircraft trajectory optimization: design

Style: ASD-STE100 (Simplified Technical English). Units: SI. Regulation values in feet or nautical
miles occur only in quoted text, with the SI value beside them.

## 0. Status

### 0.1 State

| Item | Value |
|---|---|
| Base commit | `dev-two-tier` `a804633e` |
| Code written | Merged into `dev-two-tier` (fast-forward to `b9a8afec`, 2026-10-06; the branch `dev-optimizer-multi-aircraft` deleted): T0 (`bbb18c52`, `5e3826f9`); T1–T6, M1 (`b5f4cd1a`); F9/F10/F13 (`18d73bd0`); T8, M2 (`958c4b71`); MD13 (`c7f2b99e`); T7 (`78ce2fcb`); T10 (`4e6c3ad0`, `87f5f2f2`); T11 (`7a6dca9b`) |
| Companion document | `code_review.md` (findings F1 to F13). Step T0 of this design needs F4, F5, F6 (item 3) and F7 |
| Model of the scenario | The two-tier post-training "one aircraft commanded" (`ts_transformer/docs/two_tier/design/post_training.md` D29, D93) |
| Dependency on `ts_transformer` | Two pure modules, read-only, through one adapter (§6, §7). No other import. No shared output |
| Decision state | MD1, MD2, MD3 decided by the user (2026-10-05). MD4–MD6, MD9, MD10, MD12: built as recommended (Claude, within the rules). MD13 decided by the user (2026-10-05). MD7, MD8: T5 measured (§8.1), the user decides. MD11: not built (open). T7: one M1 sample published (the user, 2026-10-05; §8.3) |

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
| MD13 | A re-solve that fails (T5: 4 of 50 windows, IPOPT `Maximum_Iterations_Exceeded`) | (a) Soft rows: a slack per row and a penalty in the objective (needs an objective hook in the optimizer). (b) Solve once more with the other branch. (c) Accept and report | **Decided: (b)** (user, 2026-10-05), built (§5.4 item 4) |
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
  model instructions. §11 item 3 lists this as an open question.

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
4. If the solve fails, solve once more with the other branch (MD13) for each recorded aircraft that has a
   position loss in this round, except when one of its losses is in trail (there only horizontal
   separation counts). A new vertical branch takes the side of this round's tightest loss with that
   aircraft. An aircraft whose rows already hold keeps its branch (Claude's reading of MD13). Keep the new
   branches after a good solve. If this solve also fails, the outcome is `solve_failed`. Keep both IPOPT
   messages, the rows and the time of each attempt, and keep the last good solve as the window's record.
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

`traffic/block.py`, runner `traffic_optimization.py m2`.

1. **Block.** The arrivals of one airport whose recorded landing is inside one time interval (for example
   1 h). Each is built as a scenario as in M1. An arrival without a dynamics model stays a record.
2. **ETA.** Each aircraft's free-time baseline without traffic (the shortest feasible IAF): `t0 + T`. An
   aircraft whose ETA solve fails stays a record.
3. **L0 schedule.** First come first served by ETA, each aircraft on its record's runway, each placed at the
   earliest time that every minimum allows (`runway_schedule.earliest_time`) against the slots placed before
   it and against the recorded landings of all arrivals outside the scheduled set (frozen slots: those
   aircraft fly their records). `runway_schedule.schedule` has no argument for frozen slots, so
   `block.place` repeats its loop for one runway per arrival. The result is one slot (a controlled time of
   arrival, CTA) per aircraft; the slot order is this placement order. The minima are timed at one speed for
   the block: the slowest target speed of its aircraft (`faa_separation` takes one speed). This is a
   choice for M2; MD9 for M2 stays open.
4. **Fixed-time solve.** Each aircraft flies to its CTA: a fixed-time solve (`solve_iaf(fixed_duration_s =
   CTA − t0)`) on the IAF of its ETA solve, started from that solution. A CTA later than `t0 + max_duration`
   is refused. A CTA that the procedure corridor cannot absorb fails the solve; the readout counts these
   failures with their delay. In fixed time the landing time is a constant, so a wake loss at the threshold
   gets no row: a window that is left with only such losses ends as `wake_at_fixed_time`.
5. **Priority order.** The aircraft fly in slot order. For aircraft `m`, the traffic is: the replays of the
   aircraft before it, and the records of every arrival outside the scheduled set. An aircraft whose slot
   solve fails flies its record for the aircraft after it (its slot was already used by the schedule; the
   readout reports it). An aircraft never sees the aircraft after it.
6. **Final check.** Each flown aircraft is judged against all the others (replays and records): the losses
   it answers for, the losses with it that it does not answer for, and the background near it.

### 5.7 M3: one joint NLP (not designed)

One NLP for all aircraft of a conflict cluster. It needs common node times for all aircraft, so the phase
durations become fixed or shared. Design it only if M2 leaves losses that a sequential order cannot
remove. §11 item 1 records the open questions.

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
| T2 | Spike: inline casadi interpolant in `SX` at a symbolic node time | Derivative equals a finite difference within 1e-6 relative on 10 random tracks. If not: user decision | T0 | Done: inline interpolant fails in `SX`; `pw_lin` passes (§5.3) |
| T3 | Adapter (§6.2), architecture test, behaviour pins, the inputs of §4.3 | Pins pass; the course check passes on one runway per airport | MD12 | Done: `traffic/rules.py`, `traffic/tests/test_rules.py` (pins, import boundary, no torch) |
| T4 | Baseline census: judge the baseline replays of a seeded sample of windows (no rows) | A readout to the user: windows with a loss that the commanded aircraft answers for, by reading, kind and airport; recorded aircraft per window; the frame error of §4.4 | T1, T3, MD5, MD6 | Done inside T5 (the baseline round of each window) |
| T5 | Rows and loop (§5.3, §5.4) on 50 windows with a loss | A readout: outcome counts, iterations, rows, flight time change, solve time per window, memory. The user then decides MD7 and MD8 | T2, T4 | Done (§8.1); the user decides MD7, MD8 |
| T6 | Traffic batch, records, resume (§9) | Full optimizer suite passes. Preflight on the real size: time and memory per window measured, disk estimate checked before the start | T5, MD7, MD8 | Done: `traffic_optimization.py`, `run_batch` sidecars; the preflight at the real size is open |
| T7 | Evaluation of the commanded records and the CZML comparison (a new category) | The frontend shows a published window; checked in the browser | T6 | Done (§8.3): one M1 sample, KRDU (user, 2026-10-05); frontend by a sonnet agent |
| T8 | M2 (§5.6) | A readout like T5 for 5 blocks | T6, MD9 | Done (§8.2) |
| T9 | M3 design (§5.7) | Only if T8 leaves losses that the order cannot remove | T8 | Not started |
| T10 | M2 in the viewer (the user, 2026-10-06): (a) one directory and one `summary.json` per M2 run; (b) the builder and the frontend show a run as one scene (§9); (c) one KRDU run published | Builder and frontend tests; the publication validator; checked in the browser | T7, T8 | Done (§8.4): (a) `4e6c3ad0`; (b) `87f5f2f2`; (c) published. The legend and M1's one window: in work with T11 |
| T11 | The Optimize task's multi-aircraft mode (§10; the user, 2026-10-06) | Backend and frontend tests; one M1 job and one 15-min M2 job on a test stack, checked in the browser | T10 | Built (§10; this commit); reviewed; real jobs on a test stack matched the batch; checked in the browser |
| T12 | The scenario list (§10.6; the user, 2026-10-06): the census runner, its catalog per airport, `GET /traffic/scenarios`, the panel's list instead of the date | Tests; the KRDU catalog; checked in the browser by someone who is NOT told a date or a flight | T11 | Design written (§10.6) |

### 8.1 T5 readout (KRDU, seed 11, 50 windows; the reviewed M1 code with the retry of MD13, scratch output)

- Cost: 50 windows on 3 workers at the lowest CPU priority (`nice -n 19`, beside a two-tier campaign) in 3 min 5 s wall time; the largest resident memory is 4.2 GB (the parent, which reads the manifest once). The traffic of KRDU is 24,202 arrivals, read in 21–33 s.
- Outcomes: 43 `separated_at_baseline`; 4 `separated` after 1 or 2 re-solves (the landing 34 s to 101 s later); 3 `solve_failed`. Four first re-solves failed (IPOPT `Maximum_Iterations_Exceeded` on long "radar or vertical" losses); each was retried with the other branch (MD13): one retry solved (horizontal to vertical; the landing 101 s later), three failed again (`Maximum_Iterations_Exceeded`). No window needed more than 2 re-solves.
- Windows with a counted VISUAL loss: 7 at the baseline, 3 at the end (the 3 failures). Windows with an IFR loss that the commanded aircraft answers for: 17 at the baseline, 14 at the end (IFR is stricter and makes no rows, MD2). Windows with a VISUAL loss that it does not answer for (no rows, §5.2 item 4): 2 and 2.
- No window starts in a loss (MD10). No type without a CWT category. 92 loss instants between recorded aircraft near the commanded one at the baselines (background). The largest frame error is 0.30 %, inside κ = 1 %.
- `W` = 15 s and `W` = 60 s gave the same outcomes (before the margin doubling and the retry).
- The outputs are in a scratch directory, not in `4dTrajectory/outputs`: a measurement of the method, not a published result.

### 8.2 T8 readout (KRDU, five 1-hour blocks from 2026-05-21 15:00 UTC; the reviewed M2 code with the retry of MD13, scratch output)

- Cost: 5 blocks on 3 workers at the lowest CPU priority (`nice -n 19`, beside a two-tier campaign) in 9 min 47 s wall time (one worker per block; inside a block the aircraft fly in slot order, so a block is serial); the largest resident memory is 4.2 GB (the parent).
- 77 aircraft (9 to 24 per block), all scheduled. Delay: median 0 s, largest 252 s; 17 aircraft delayed by more than 60 s.
- Outcomes: 58 `separated_at_baseline` (the CTA solve alone keeps separation); 6 `separated` after re-solves; 1 `unresolved`; 9 `solve_failed` (3 at no delay, 6 delayed; IPOPT's cap, as in M1); 3 `slot_failed` (the CTA solve on the ETA's IAF failed; no other IAF is tried). 10 re-solves failed and were retried with the other branch (MD13); 3 retries solved.
- After the block's final check, 16 flown aircraft have a VISUAL loss they answer for and 6 a VISUAL loss they do not answer for; IFR: 27 and 19.
- The outputs are in a scratch directory: a measurement of the method, not a published result.

### 8.3 T7 publication (KRDU, seed 11, 50 windows; code `78ce2fcb`)

- Output: `4dTrajectory/outputs/KRDU/traffic_m1_runway_cons/` (records, sidecars, `summary.json`,
  `evaluation_report.json` and `.html`, the run logs, `SHA256SUMS`; read-only). The same outcomes as §8.1.
- Category `traffic_m1_runway_cons` in `aeroviz-4d/public/data/airports/KRDU/comparison/` (4 CZML files, 50
  groups). `categories.json` got this one key and no other change (checked against a copy taken before the
  write). The evaluation: 46 pass, 4 fail.
- The viewer (AV46 in `aeroviz-4d/docs/35-viewer-reference.md`): each shown group with its recorded
  neighbours in pink, at their real time relative to the commanded aircraft, clipped to the groups' clock.
- Browser check (2026-10-06, the branch's frontend on a test port; Evaluate view, Result source
  Optimization): the category loads without errors; the pink neighbours move on their approaches with the
  commanded aircraft, some already in the air at the clock start, none held at a runway end; the Flights
  table shows each window's outcome; the Reference switch hides the neighbours too; another category
  shows none.
- A running dev server does not serve files that were published after it started (AV5): the frontend on
  5173 shows the category only after a restart.

### 8.4 T10 publication (KRDU, the five 1-hour blocks of §8.2; code `4e6c3ad0` run, `87f5f2f2` viewer)

- Output: `4dTrajectory/outputs/KRDU/traffic_m2_runway_cons/` (one directory, one `summary.json` of mode
  `traffic:m2`, the evaluation, the run logs, `SHA256SUMS`; read-only). The outcomes are those of §8.2, the
  same numbers. The evaluation: 61 pass, 16 fail.
- Category `traffic_m2_runway_cons`: one scene of 77 groups over 17,885 s (the scene starts at
  2026-05-21 14:55:39 UTC) and 48 background aircraft; `categories.json` got this key only.
- Browser check (2026-10-06, a clean checkout of `87f5f2f2` on a test port): the scene loads; the Flights table
  holds all 77 aircraft; one clock of about 5 h; each optimized path with its record beside it; landed aircraft
  leave the scene; tooltips with the outcome and the delay (33 aircraft delayed); the runway selector and the
  sample count do not reload it.
- The user's review (2026-10-06): no legend for the traffic colours, and M1 drew all its windows at once
  (§9 viewer contracts, fixed with T11).

## 9. Outputs and records

- **Runner.** `4dTrajectory/optimization/traffic_optimization.py m1 --airport <ICAO> --sample N --seed S
  --output-dir …` (a seeded sample of the airport's arrivals, stated in `summary.json`), and `… m2 --airport
  <ICAO> --block-start <UTC> --block-s 3600 --blocks N --output-dir …` (the records of every block in one
  directory; one `summary.json`, mode `traffic:m2`, with `results` for every block and `blocks`: each block's
  schedule, outcomes and final check). The `mode` values are `traffic.M1_MODE` and `traffic.M2_MODE`. The
  readout is `python -m traffic.readout <dir>` from `4dTrajectory/optimization` (it reads either mode).
- **M2 sidecar.** The M1 sidecar plus `slot` (ETA, CTA, delay) and `block_final` (the final check), schema
  `optimization-traffic-block-v2`.
- **Directory.** `4dTrajectory/outputs/<ICAO>/traffic_m1_<category>/` (M1) and
  `4dTrajectory/outputs/<ICAO>/traffic_m2_<category>/` (M2). The existing directories
  `runway`, `fitted_adsb`, `runway_cons` stay unchanged.
- **Commanded record.** `*_states.json` and `*_eval.json` in the existing evaluation contract
  (`evaluation_export.py`), so `evaluation` grades them unchanged. The filename stem is `flight_key`.
- **Traffic sidecar.** One `*_traffic.json` per window: the outcome, the flags, the iterations, the rows and solve time of each attempt (a retry of MD13 included),
  the losses per iteration and reading, the recorded aircraft (by `flight_key`), the flight times. Its schema
  name is `optimization-traffic-v2`. Settle every field before step T6 starts. A later change gets a new
  schema name; no reader accepts two versions (root `CLAUDE.md`, the compatibility rule).
- **Viewer, M1** (AV46 in `aeroviz-4d/docs/35-viewer-reference.md`): one window at a time — the selected
  flight's group (the first by default), with its controlled aircraft, its record and only its own neighbours.
  The other windows stay in the Flights table. All windows at once on overlapping clocks put one window's
  neighbours beside another window's controlled aircraft (the user found this, 2026-10-06).
- **Legend** (both traffic modes): "Controlled aircraft — optimized path", "Controlled aircraft — its record"
  (white), "Recorded traffic — not controlled" (pink; M2: outside the scheduled set).
- **Viewer, M2: one scene.** All aircraft of an M2 run fly on ONE clock: the scene starts at the earliest
  entry of its groups (`S`). Each group (solved or not) gets `scene: {startOffsetS, outcome, delayS}`
  (`startOffsetS` = its entry − `S`; its result paths are written on the scene clock, its reference is shifted
  by the same offset). The index gets `scene: {startUtc, background: {recorded, startOffsetsS}}`: the recorded
  aircraft that the flown aircraft saw and that are not groups (records outside the scheduled set), each
  once. The viewer shows every group of a scene (no sample), the background in the neighbours' pink under
  the same rules as M1 (model budget, the Reference switch, clipped to the clock). The roster for the entry
  times is `optimization_config.traffic.selection.manifest`, as in M1.
- **Summary.** `summary.json` per directory: the configuration (with `h_c`, `W`, `κ`, `K_max`, the reading),
  the outcome counts, the counts of §5.2 item 4, §4.1 item 5, MD10 and MD11, and the largest frame error.

## 10. Interactive mode: the Optimize task

The user, 2026-10-06: the Optimize task gets a multi-aircraft mode of two kinds, one aircraft controlled (M1)
and all aircraft controlled (M2). This section is the design (step T11). A sonnet agent writes the code (the
user's order). Claude reviews it.

### 10.1 What the user does

1. In the Optimize task, the user selects the mode: `Single aircraft` (the present mode, not changed),
   `Multi-aircraft: one controlled` (M1) or `Multi-aircraft: all controlled` (M2).
2. The panel shows the airport's scenario list (§10.6), not a date: for M1 the arrivals whose own record
   has a loss they answer for, for M2 the blocks of the most traffic and losses. Each row gives the time, the
   runway(s), the aircraft and the recorded losses. The list can be sorted and filtered by runway.
3. A click on a row selects the scenario (M1: that arrival; M2: that block, its length 15, 30 or 60 min).
4. The user starts the job. The panel shows the progress (aircraft done of the total, the aircraft last
   finished) and a Cancel button. Start is disabled while a job runs: the user cancels first.
5. When the job is done, the viewer shows the scene (§10.4). The panel shows one row per controlled aircraft
   (callsign, runway, outcome, and for M2 the delay) and one summary line (the outcome counts; for M2 also
   the delays and the losses left after the final check). A click on a row selects that flight.
6. A change of mode, another airport, or leaving the Optimize task cancels a running job and removes the
   scene. The single-aircraft panel stays mounted (hidden) in the multi-aircraft modes, so its state is kept.
7. A traffic scene shows a "Scene time (UTC)" readout: the display clock has a fixed epoch, the readout gives
   the real time.

### 10.2 Decisions

These are Claude's choices for this design. The user can change them.

| ID | Question | Choice | Reason |
|---|---|---|---|
| IM1 | Which aircraft can be controlled | Recorded arrivals only, built as the batch builds them: the record's start, the runway-threshold target, the constrained solve on the shortest IAF | The judge needs a record: a flight key, a runway, a type with a wake category. A hand-set aircraft has none of them |
| IM2 | The solver settings | The batch defaults (`LoopSettings`, `max_duration` 2000 s, rollout step 0.5 s, IPOPT cap 3000), shown read-only | The same code and settings as T5 and T8, so an interactive result compares with the batch |
| IM3 | Block length (M2) | 15, 30 or 60 min; 30 min by default | A block is serial. T8: a 1-hour block takes up to about 10 min |
| IM4 | Where a job runs | One process group per job, separate from the resident single-aircraft worker, at `nice 10`, with one solver thread (the batch's thread variables); one job per backend at a time, also across a backend restart (the job directory records its process group); the job root is per backend port | casadi is not thread-safe; a long job must not block the single-aircraft optimizer; the box also runs the two-tier experiments; one thread keeps a job comparable with the batch |
| IM5 | Which traffic a job reads | Only the roster rows whose arrival slice overlaps the job's time span plus `max_duration`, and only their tracks (`load_model_arrivals_subset`); plus one arrival per runway of the roster for the runway targets, so the job's runways are the batch's | The batch reads the whole roster (4.2 GB peak). The live backend must not |
| IM6 | Where the result goes | A job directory in the backend's cache, not `public/data`: the batch records and `summary.json` (mode `traffic:m1` or `traffic:m2`), the evaluation report, and the comparison builder's files (§9). No `categories.json`. The backend keeps the last 5 job directories | A job result is not a publication. The frontend reads the same files as a published category |
| IM7 | How the frontend shows the result | The comparison layer (AV46 for M1, AV47 for M2), fed from the job's files instead of `public/data` | One renderer for published and interactive results |

### 10.3 Backend

- `GET /traffic/arrivals?airport=<ICAO>&date=<YYYY-MM-DD>`: the roster rows that land on that UTC day (flight
  key, callsign, runway, type, entry and landing UTC). It reads only the roster.
- `POST /traffic/jobs` with `{mode: "m1", airport, flightKey}` or `{mode: "m2", airport, blockStartUtc,
  blockS}`: starts a job and returns `{jobId}`. 409 when a job runs. 400 for a bad request (an unknown flight
  key, a block length other than 900, 1800 or 3600 s).
- `GET /traffic/jobs/<jobId>`: `{state: running | done | failed | cancelled, progress: {done, total, current},
  error}`, and when the state is `done` also `summary` (the job's `traffic.readout`: the panel's summary line
  needs the losses left after the final check, which no served file holds). `current` is the aircraft last
  finished.
- `GET /traffic/jobs/<jobId>/files/<name>`: one file of the job directory that the comparison index lists (the
  index, the CZML files, the evaluation report). 404 for any other name.
- `POST /traffic/jobs/<jobId>/cancel`: stops the job process and its process group.
- The job process is `python 4dTrajectory/optimization/traffic_job.py --spec <file> --out <dir>`. It reads the
  near traffic (IM5), builds the scenarios as the batch does, flies M1 (`traffic.loop.fly_in_traffic`) or M2
  (`traffic.block.fly_block`), writes the records and `summary.json` with the batch writers, runs the
  evaluation and the comparison builder into `<dir>`, and writes `progress.json` after each aircraft
  (`fly_block` gets an optional progress callback). Its last write is `state.json`: `done`, or `failed` with
  the reason.
- Safety of the job's process group: the backend signals a group only when its leader is the job's own process
  (an unreaped child of this backend, or for a job of an earlier backend the same start time and boot id in
  `process.json`). A job whose process is gone without `state.json` is failed at once. Known limits (rare): a job
  of an earlier backend whose leader is gone but whose evaluation or builder step still runs is marked failed
  and that step is not stopped; a backend that dies between starting a job and writing `process.json` leaves a
  job the next backend cannot see. Linux only (`/proc`): elsewhere a job is refused by name.
- The backend does not hot-reload. Test this mode on a separate backend (the worktree, a free port) with the
  worktree's frontend, never on the live backend (8765).

### 10.4 Frontend

- A mode selector in the Optimize panel. The single-aircraft mode stays as it is.
- The comparison layer gets a source: a published category (as now) or a job (the base URL
  `/traffic/jobs/<jobId>/files/`). It runs in the Evaluate task for a category and in the Optimize task for a
  job.
- While a job runs, the panel reads `GET /traffic/jobs/<jobId>` every 2 s.
- The scene is the one of AV46 (M1: the controlled aircraft, its record, its neighbours) or AV47 (M2: one clock,
  every controlled aircraft, the background).

### 10.5 Tests and checks

- Backend: the job manager (one job at a time, cancel stops the process group, only listed files are served,
  the last 5 jobs kept) on a fake job process; the job runner (the near-traffic selection reads no other
  track; a fake solve end to end into a tmp directory; the progress file).
- Frontend: the mode selector, the job lifecycle (start, poll, cancel on a change of mode and on leaving the
  task), the comparison layer fed from a job.
- One M1 job and one 15-min M2 job on the test stack, checked in the browser.

### 10.6 Scenario list

The user, 2026-10-06: the scenario is chosen from a list computed in advance, never from a date (a date gave no
hint which days hold data and which flights meet traffic).

- **Census.** `4dTrajectory/optimization/traffic_scenarios.py --airport <ICAO>`, offline, once per airport and
  harvest. Each arrival that M1 can command (it has a dynamics model) is judged on its OWN RECORD as the flown
  track, in its recorded traffic, with the M1 judge (`traffic.check.check`, reading VISUAL, MD2). This is the
  same rule set as the M1 baseline, so a loss in the list means what a loss in a result means.
- **Check step.** The census check step is an option, stated in the catalog (the M1 loop uses 1 s, MD6).
- **M1 rows.** Every arrival with at least one loss it answers for: flight key, callsign, runway, type, its CWT
  category, entry and landing UTC, the loss instants it answers for, their kinds, the tightest one (distance /
  required), and the recorded aircraft in its window. Sorted by the loss instants. The catalog states how many
  arrivals were judged and how many have a loss; no row is dropped. The panel hides CWT category I (light
  aircraft) by default, with a checkbox: on KRDU, C172 practice traffic tops the ranking otherwise (the first
  browser check, 2026-10-06).
- **M2 rows.** For each block length (15, 30, 60 min, aligned to the length): every block in which at least
  one arrival lands, with its start, its arrivals, its runways, and the loss instants its arrivals answer
  for. Sorted by those losses, then by arrivals.
- **Where.** `4dTrajectory/outputs/<ICAO>/traffic_scenarios/catalog.json` (schema
  `traffic-scenario-catalog-v2`, with the configuration and the roster it judged), served by `GET
  /traffic/scenarios?airport=<ICAO>`; a missing catalog is a 404 naming the command that makes it. Not in
  `public/data`.
- **Interpretation.** A loss in the record is a loss between recorded aircraft as flown, judged by these rules.
  It marks dense traffic. It does not say that the optimized baseline will have a loss (T5: 7 of 50 sampled
  windows had one).
- **The panel.** Lists show 100 rows at a time with the count of the rest. The result shows, per controlled
  aircraft, what changed: the loss instants in its record, after the first solve and at the end, and its landing
  against its record (M2 also its delay; an aircraft that was not flown shows no delay). While a job runs the panel
  shows its phase and the elapsed time. While a scene is shown the clock widgets display its real UTC time.

## 11. Open questions

1. **M3.** Common node times for all aircraft; disjunction branches for pairs that change order; the size
   of a conflict cluster. Not designed.
2. **The objective in traffic.** M1 keeps minimum time. A delay then appears only where a row forces it.
   A controller would also give speed instructions earlier. The present objective does not model this.
3. **Speed instructions inside the FAF or 5 NM.** 7110.65BB 5-7-1 b4 forbids speed instructions there. The
   optimizer can still change its speed there. A row that holds the speed inside that region is possible.
   It is not in this design.
4. **Losses that the commanded aircraft does not answer for** (§5.2 item 4). The optimizer could also avoid
   them. This design does not, to match two-tier D93. The counts show if this choice matters.

## 12. Key code index

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

## 13. Regulation sources

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
