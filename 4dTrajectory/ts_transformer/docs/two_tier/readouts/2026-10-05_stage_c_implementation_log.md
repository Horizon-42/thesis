# Stage C: implementation log (post-training document §0.3, §8 C0–C12)

The implementer's log of stage C (outline §5 rule 10): the state, the commits, and each reading made where the design
says nothing. A reading is a proposal until the user decides. The design document (`design/post_training.md`) keeps
the rules and the specifications of the milestones not done; its §0.3 holds only a short status table.

Branch `dev-two-tier-v4-post`, worktree `.claude/worktrees/two-tier-v4-post`, made 2026-10-05 from
`dev-two-tier-v4-prior` `08b1c5e1` and merged with `dev-two-tier` `a472cb79` (`2a2d735f`). Its ignored data trees
(`data`, `trajectory_data_process/outputs`, `4dTrajectory/outputs`, `aeroviz-4d/public/data/airports`,
`aeroviz-4d/node_modules`) are absolute links to the live data (outline §5 rule 1).

---

## 1 State

| Part | State |
|---|---|
| C0: the package `post/`, its import rules | Done, `1b4d3cea` (reviewed) |
| C1: windows and scenes, the census runner `post_windows` | Done on synthetic artefacts, `1b4d3cea` (reviewed); smoke and census on A34's artefact (§3) |
| C2: the edge features and their conformance | Done on synthetic artefacts, `1b4d3cea` (reviewed); its reference on A34's train windows (§3) |
| C3: "established", the separation judge on v4, the speed-word mask, the regulation text | Code `1b4d3cea` (reviewed); the text `86fc86d6` (`docs/literature/arrival_separation/README.md` §8) |
| The user's choices from C1's census | Open: the count of each kind of window in a round (§2 item 4, D55); the shifts of A and D (§4 P7) |
| C4–C7 | Wait for B9 (prior D96) and the parts of A38 (vocabulary D97) on this branch |
| C8–C12 | Wait for B5's base, Claude's check of stage B, the moved start (C9) and the user's criteria (D7) |

## 2 C0–C3 (2026-10-05)

**Commits** (`dev-two-tier-v4-post`): `1b4d3cea` (code, reviewed by an independent reviewer; its two bugs — a NaN that
passed the edge check, the start-of-day side of the day cut not counted — and its contract points fixed before the
commit, the fixes reviewed again); `86fc86d6` (the regulation text). Tests: `test_post_scene.py`, `test_post_edges.py`,
`test_post_separation.py`, `test_architecture.py`: 57 passed. The full ts suite: §6.

**Code.**

| What | Where |
|---|---|
| Package `post/` (C0) | `post/__init__.py` |
| The motion of a scene: the 2 s displacement (a tested mirror of `prior.inputs.motion`) | `post/motion.py` `motion` |
| An airport's separation rules on the artefact's candidates; the approach clock | `post/runways.py` `airport_separation`, `approach_clock_m` |
| A recorded flight, the aircraft at a step, a scene, a window (real, A, D), the census (C1) | `post/scene.py` `Recorded`, `AircraftAt`, `Scene`, `MovedScene`, `Window`, `airport_scenes`, `real_windows`, `inserted_window`, `leader_moved_window`, `next_ahead`, `near_day_cut`, `census` |
| The tokens of the traffic attention (C2, D98) | `post/edges.py` `tokens`, `TOKEN_FEATURES`, `EDGES_SCHEMA` (`post-edges-v1`) |
| Their identity (§4 item 1) | `post/conformance.py` `reference_steps`, `write_edge_reference`, `require_conforming_edges` (`post-edges-reference-v1`) |
| "Established" (D92) | `post/established.py` `established` |
| The separation judge's inputs on v4; the event of a window | `post/traffic.py` `traffic`, `commanded_loss`, `joined` |
| The speed-word mask (§3) | `post/speed_mask.py` `speed_check`, `along_course_speeds`, `ramp_distance_m`, `ramp_time_s` |
| The census runner (C1) | `experiments/post_windows.py` (`post-windows-census-v1`) |
| Stage C's import rules (C0) | `tests/test_architecture.py` `POST_MAY_IMPORT`, `PRIOR_INTERFACE`, `test_the_post_training_reads_only_its_layers`, `test_stage_c_takes_from_the_prior_only_the_names_of_its_interface`, `test_only_the_runners_reach_the_post_training`, `test_the_stage_c_runners_take_from_autopilot_only_what_its_interface_lists` |

## 3 C1 and C2 on A34's artefact (read-only)

`instruction_language/v12_20261005`, Δ = 4 s, from `1b4d3cea` (the census code as committed), in the session's
scratchpad (outline §5 rules 7 and 12; nothing under `4dTrajectory/outputs/`). Only the signals, the candidates and the
index of the closed-loop sentences were read; the validation days were not read (D85).

- **Smoke** (`--sample 50`: 50 windows of each airport and split, seed 1337): 4.4 s, 0.7 GB.
- **Census of all train days and of select**: 7.7 s, 0.7 GB. The edge reference: 323 steps, 420 tokens, read again with
  the largest difference 0.0.

| Split | Airport | Windows | Others at the first predicted step: mean, p50, p90, max | Share with none | Share with a leader in the air | Windows near a day cut | A possible | D possible |
|---|---|---|---|---|---|---|---|---|
| train | KMSY | 4,808 | 0.75, 1, 2, 5 | 0.45 | 0.37 | 0 | 4,808 | 1,760 |
| train | KRDU | 11,864 | 1.71, 2, 3, 7 | 0.18 | 0.39 | 12 | 11,864 | 4,638 |
| train | KSJC | 9,101 | 0.95, 1, 2, 5 | 0.35 | 0.54 | 232 | 9,101 | 4,920 |
| train | KSMF | 5,377 | 1.06, 1, 2, 6 | 0.34 | 0.25 | 6 | 5,377 | 1,335 |
| train | KSTL | 9,380 | 1.31, 1, 3, 7 | 0.28 | 0.38 | 0 | 9,380 | 3,581 |
| train | all | 40,530 | | 0.294 | 0.401 | 250 | 40,530 | 16,234 |
| select | KMSY | 657 | 0.71, 1, 2, 4 | 0.48 | 0.34 | 0 | 657 | 226 |
| select | KRDU | 1,916 | 1.80, 2, 4, 7 | 0.15 | 0.41 | 0 | 1,916 | 788 |
| select | KSJC | 1,516 | 1.23, 1, 3, 7 | 0.31 | 0.51 | 74 | 1,516 | 772 |
| select | KSMF | 844 | 1.02, 1, 2, 5 | 0.36 | 0.23 | 2 | 844 | 190 |
| select | KSTL | 1,266 | 1.25, 1, 3, 6 | 0.30 | 0.36 | 0 | 1,266 | 462 |
| select | all | 6,199 | | 0.285 | 0.393 | 76 | 6,199 | 2,438 |

"A leader in the air": the aircraft next ahead on the approach clock at the first predicted step, on the commanded
flight's recorded runway or a runway separated as one, each on its record (P8): the windows that admit D. Evidence §6.1
says that about 45 % of the train flights have no leader at their first predicted step; here 60 % have none, by this
definition (the earlier readout's definition is not this one; not compared).

## 4 Proposals (readings where the design says nothing; each holds until the user decides)

- **P1. The first predicted step and R of a recorded aircraft.** Its first predicted step is its first row on the Δ
  grid + 16 s (`first_interval_row`, the sentence's own rule); its R is in force at the steps after it, not at it — as
  the commanded aircraft's runway word, said at its first predicted step, is in force from its next row. So D23's test
  ("up to the first predicted step") holds for both, inclusive.
- **P2. One motion for every aircraft.** The track of "established", the separation judge and the speed-word mask, and
  every motion of a token, is the 2 s displacement before the row (as the prior's motion inputs), for the commanded
  aircraft too, though its flown states have an exact track. An aircraft at its row 0 (no motion) is not established.
- **P3. A recorded aircraft's G is false.** A recorded flight has no words, so no go-around word ends or starts its G;
  "established" (D92) reads G false for it. A recorded go-around inside the final's region and lined up would read as
  established. Alternatives for the user: the labelled go-around rows of its open-loop sentence where it has one (a
  withheld field, not an input; the judge is not an input either).
- **P4. A recorded aircraft over its threshold** is so at its last step in the air (its last row is the last before the
  observed crossing, so the crossing is within one Δ after); the on-approach wake minimum behind it is judged then.
- **P5. The speed-word mask.** A recorded leader's speed target is its present speed along its course (it has no speed
  word). "Unchanged" and "unspecified" (the pilot's own speed, which the mask cannot predict) are never masked. The
  speeds are ground speeds along the course of R.
- **P6. The tokens.** The other aircraft's own motion is its ground speed, its vertical rate and its direction of motion
  less the commanded aircraft's (0 without a motion or a frame); `required` takes the aircraft ahead on the approach
  clock as the leader (a tie: the other aircraft); the scales: 3 NM (5,556 m, the radar minimum) for horizontal
  distances (asinh), 1,000 m, 100 m/s, 10 m/s, a closest-approach horizon of 120 s.
- **P7. Windows A and D** (§8 C1: the ranges are the implementer's proposals). A: a flight of the same airport and split
  whose recorded landing is more than 3,600 s from the commanded flight's, shifted so that it lands within ±180 s of the
  commanded flight's recorded landing (a whole number of Δ). D: the aircraft next ahead moved by a whole number of Δ in
  ±120 s, never 0. Both are drawn only from real windows.
- **P8. The census's definitions.** A leader in the air: the aircraft next ahead on the approach clock at the first
  predicted step, on the commanded flight's recorded runway or one separated as one (records' runways and positions). Near a day cut: the day beside the commanded flight's own day is not of its split, and either its recorded
  landing is less than the scene's longest record before the end of its day, or its row 0 is before the start of its
  day.
- **P9. Where the edge reference lives.** `post_windows` writes it beside its census (`conformance/edges.npz`), with its
  airports' geometry, so that the check reads fixed inputs; 10 windows of each airport (train, seed 1337), steps every
  60 s from the first predicted step while the record lasts; tolerance 1e-6 (float32 tokens). §4 item 1 says "written
  with the code": the user chooses whether the reference of the formal census is the one every later process reads
  (by its path), or a reference committed beside the code.
- **P10. Interface items read outside the listed names.** The census reads which flights have a closed-loop sentence by
  `instructions.artefact.closed_loop_indices` (the index only, no sentence; vocabulary §6 item 3 lists the reader of the
  sentences, not this function); the speed-word mask builds the caller's mask in `instructions.grammar.column_words`
  order, the order the speaker requires (prior §7 item 3; vocabulary §6 item 2 lists `apply`). Both are requests to
  stage A, through the user, to name them in vocabulary §6 — or another way the user chooses.

## 5 For the user (design gaps; Claude writes the decided text)

1. **D92 against the regulation text** (`docs/literature/arrival_separation/README.md` §8):
   - TBL 5-9-1 sets 20° (not 30°) for an interception less than 2 NM (3,704 m) from the approach gate; D92's region is
     inside the FAF, so inside the gate, where the table's row is 20°. In the text 30° is an intercept angle for an
     assigned heading, not a test for "established".
   - The text lets an aircraft be established on the final approach course outside the FAF (FIG 5-9-1 Example 4,
     8 NM out; 5-5-4 j "within 10 NM"); D92's region is narrower.
   - 5-7-1 b4: no speed adjustment "inside the final approach fix on final or a point 5 miles from the runway, whichever
     is closer to the runway"; D92 and §3 take 5 NM in every case.
   - Separation is the controller's (5-9-5 a); "the aircraft that answers" is the project's reading of 5-5-4 g/h,
     5-7-1 a3, 7-4-4 c.
2. **The speed-word mask can act only between 9,260 m and the FAF**, because it needs both aircraft established (inside
   the FAF, D92) and the commanded aircraft more than 5 NM from its threshold. The FAFs of A34's candidates (CIFP, m
   before the threshold): KMSY 11,286–11,387; KRDU 9,505–10,336; KSJC 9,926–15,068; KSMF 17L 10,603, 17R 10,022, 35L
   7,687, 35R 7,663; KSTL 8,362–10,963. So the band is 0.1–2 km at most runways (5.8 km at KSJC 30L), and the mask
   never acts at KSMF 35L/35R and KSTL 30L/30R (FAF inside 5 NM).
3. P3 (a recorded aircraft's G), P9 (the reference's place), P10 (the interface names) above.

## 6 The full ts suite

At `86fc86d6` (`1b4d3cea` + the regulation text), one test at a time, detached (outline §5 rule 4; stage A's A37
check and a stage B `prior_train` were running, rule 13): 1,688 passed, 1 skipped, 22 min 33 s.
