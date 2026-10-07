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
| C3: "established", the separation judge on v4, the speed-word mask, the regulation text | Code `1b4d3cea` (reviewed); the text `86fc86d6` (`docs/literature/arrival_separation/README.md` §8); the user's decisions of 2026-10-05 (§7) in `ca15a2e5` (reviewed) |
| The user's choices from C1's census | Decided: real : A : D = 1 : 1 : 1 in a round (§7). Open: the shifts of A and D (§4 P7) |
| C5, C7 | Done on synthetic artefacts and models, `3171da69` (reviewed), after B9 and `dev-two-tier` were merged (`52224585`, `68407e13`); §9 |
| C4 | Done on synthetic artefacts, `0168d457` (reviewed), after B10 was merged (`b6914acc`); §11 |
| C6 | Waits for A38 (vocabulary D97: the copy of a loop) on this branch |
| D105: the landings of a window's scene | Done, `00e81c88` (reviewed), after `dev-two-tier` `202b38d0` was merged (`65b4d782` merged as well); §10 |
| C8–C12 | Wait for B5's base, Claude's check of stage B, the moved start (C9) and the user's criteria (D7) |
| C14 (D157): C10 to 14 rounds | Code `21547ad4` (reviewed) on `dev-two-tier-v4-post`; rounds 10–13 wait for the user's merge and stage D's GPU and timing steps (§27) |

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
3. P9 (the reference's place), P10 (the interface names) above. (P3, items 1 and 2: decided, §7.)

## 6 The full ts suite

At `86fc86d6` (`1b4d3cea` + the regulation text), one test at a time, detached (outline §5 rule 4; stage A's A37
check and a stage B `prior_train` were running, rule 13): 1,688 passed, 1 skipped, 22 min 33 s.

After `ca15a2e5`: see §8.

## 7 Decided by the user (2026-10-05)

The user's answers to §5 and to the census. The design text of these decisions is the design author's to write (outline
§5 rule 10); until then this log records them, and the code follows them.

| Question | Decision | Code |
|---|---|---|
| The count of each kind of window in a round (§2 item 4, D55) | Real : A : D = 1 : 1 : 1 | C10 (the rounds) |
| The angle of "established" (D92 says 30°, the judge's lined-up angle) | **20°**: 7110.65BB 5-9-2 a, TBL 5-9-1, the row for an interception less than 2 NM from the approach gate, which holds inside the FAF. The judge's lined-up angle (30°) and the turned-in rule of the visual reading (7-4-4 c, 30°) do not change | `post/established.py` `ESTABLISHED_MAX_ANGLE_DEG`, `ca15a2e5` |
| The speed-word mask acts only between 5 NM and the FAF (§5 item 2) | Kept as designed; a stated limit: the readouts report how often it acts | — (C8 and the readouts report it) |
| A recorded aircraft's G (P3) | **From its labelled sentence**: G true at the rows after each labelled go-around row up to and including the row where the runway is said again (a word in force from the next row, as for the commanded aircraft); a flight without a sentence has G false. The labelled rows are a withheld field: only the separation judge and the masks read G. The commanded aircraft's G is its words', never its labelled one (a window refuses a commanded record with G). On A34's train days: 80 flights carry G, 3,656 rows | `post/scene.py` `go_around_rows`, `Recorded.go_around`, `AircraftAt.go_around`, `ca15a2e5` |

Why recorded aircraft had no G (for the record): D92 defines "established" through G and D31 makes it the same for
every aircraft; D93 and D98 make every other aircraft a recorded flight that no one speaks to. The labeller's reading of
a recorded flight (its sentence, for 44,338 of the 44,703 train flights) has the go-around rows; the user chose to read
G from them.

## 8 After the decisions

`ca15a2e5` on `dev-two-tier-v4-post`, reviewed by an independent reviewer (its findings fixed in the same commit: the
commanded record's labelled G stripped and refused, a row past the record refused, G checked as bool, the
signal-index mapping and the G of a shifted record tested). The four stage C test files: 59 passed. The full ts
suite at `ca15a2e5`, one test at a time (rule 13): 1,690 passed, 1 skipped, 18 min 11 s.

## 9 C5 and C7 (2026-10-05)

On the user's order: `dev-two-tier-v4-prior` (B9, `5313b6cd`) merged into this branch (`52224585`), then `dev-two-tier`
(`68407e13`); C5 and C7 (C4 and C6 wait: §1). The user's decisions on Claude's check of stage B, now post-training
D105–D107 and prior D105–D107 (B10): the window's landings follow its scene (C4), dropout off in the ratio and the KL and
on in the data term (C7, as built), the closed loop's step one module of stage B (C4 calls it).

**Commit** `3171da69` (code), reviewed by an independent reviewer (no blocking point; fixed before the commit: `one_pass`
refused before any update unless one data batch for each, the counted rows masked by `where` so that a value of a row
not counted cannot reach the loss, the copy at the start of the pass without gradients). Tests: the six stage C files
and `test_architecture.py`, 72 passed; the full ts suite on 8 workers: 1,712 passed, 1 skipped, 5 min 2 s (stage B's
smoke free generation and a test backend were running; no formal build).

| What | Where |
|---|---|
| The traffic attention (C5, D98) | `post/traffic_attention.py` `TrafficAttention`, `Traffic`, `traffic_of`, `TrafficConfig` (`post-traffic-attention-v1`), `add_traffic_attention`, `traffic_modules`, `parameter_groups` |
| The loss (C7) | `post/loss.py` `Samples`, `PassStart`, `surrogate`, `pull_to_base`, `data_term`, `update_loss`, `one_pass`, `stacked`; `CLIP`, `KL_WEIGHT`, `DATA_WEIGHT` |
| B9's names in the import rule | `tests/test_architecture.py` `PRIOR_INTERFACE`: `prior/loop.py` `LoopRows`, `speaker.draw`, `speaker.Permitted`, `train.masked_log_probability` (stage B's log gives them for prior §7's Code column; §7 still reads "new, B9": for Claude) |

**Proposals** (readings where the design says nothing):

- **P11. The ratio and the pull per word.** Each column of each row is one word with its own ratio, clipped on its own,
  and its own KL term, as the archived post-training (`archive/two_tier_v3_2026_10/prior/train.py` `flight_surrogate`,
  `flight_kl`), not one ratio for the row's five words.
- **P12. The ratio's denominator** is a frozen copy of the model at the start of the pass, scoring the same words under
  the same records through `masked_log_probability` (the speaker's own `drawn_probability` is not in prior §7). At the
  parameters that spoke it equals the speaker's probability within 1e-5 (a test).
- **P13. The weight of a sample.** Each sample's sum over its counted words is divided by its counted rows, and the
  samples are averaged. With C6, a continuation from a late branch point (few counted rows) then weighs as much as a
  whole first sentence. Alternatives: divide the whole batch by its counted rows (each row weighs the same).
- **P14. The pull reads the counted words only**, the same words as the surrogate; a first sentence that is in several
  branch groups (C6) has its words counted once in each group.
- **P15. No traffic is a sample without a scene.** `TrafficAttention(x, None)` gives zero: the data term's
  single-aircraft samples (D36) go through `batch_nll`, which gives the model no ``extra``.
- **P16. The shape of the module.** Each layer has its own token network (two linear layers, hidden 16 in the tests) and
  an attention of 4 heads at the prior's width; the formal shape is not chosen (C8). Each layer embeds every padded
  token, so the memory grows with B·R·N_max (N_max the most other aircraft of the batch): the reviewer's estimate at
  configuration A, B = 64, R = 300, N_max = 30, approximately 9 GB — more than the GPU. C8 measures it at the formal
  size; one token network shared by the layers would cut it (a design choice for the user).

**Requests to stage B** (through the user; in B10 as prior D105, D106): each aircraft's landings in `LoopRows`; the
closed loop's step as one module under `experiments/` with ``extra``, caller masks, an end by the caller and a copy.

## 10 The order of `notes/stage_c.md` (2026-10-05): steps 1–2

1. `dev-two-tier` merged into this branch (up to `65b4d782`, `202b38d0` included).
2. **D105** — `00e81c88`, reviewed by an independent reviewer (no bug; its two test gaps added). `post/landings.py`
   `window_landings(window, roster, days)`: a real window gives the roster's index itself; window A adds the inserted
   aircraft's landing at its source's roster landing plus the window's shift, under its own key (`…+inserted`), the
   source's landing kept; window D moves the leader's roster landing by its shift; a landing a shift puts on a sealed
   test day is left out and counted in `sealed`; the commanded aircraft's own landing stays in the index (the loop's row
   leaves it out by its key). `Landing` added to `PRIOR_INTERFACE` (prior §7 item 2). Tests: `test_post_landings.py`
   (6) and `test_architecture.py`, 39 passed; the full ts suite on 8 workers at `00e81c88`: 1,718 passed, 1 skipped,
   8 min 23 s (stage B's campaign smoke was running on the GPU; no formal build).

Steps 3–6 (the motion of prior §7 item 2, B10's names, C7's refusal of training mode, C4 through the shared step) wait
for B10 on `dev-two-tier-v4-prior` (at `5313b6cd` it is not there).

**Proposals.**

- **P17. A shifted landing on a day outside the day split** (no data that day) is counted: only a sealed test day is
  left out (D105). `roster_landings` refuses a landing on an unlisted day; a shift of at most ±180 s (A) or ±120 s (D)
  from a flight of a listed day can reach an unlisted day only at the edge of the data. B10's index decides it when it
  refuses test days itself.
- **P18. The landing that A inserts** is its source's roster landing shifted (on the source's runway), not the signals'
  landing time; the roster's time is the one a real window counts.

## 11 The order of `notes/stage_c.md`: steps 3–6, after B10 (2026-10-05)

On the user's word ("B10 合进来了"): `dev-two-tier-v4-prior` at `e4e7ba42` (B10) merged into this branch (`b6914acc`),
then `dev-two-tier` (`b1068f66`).

**Commits.**

- `3725565c` (steps 3–5, reviewed; the relative imports past the new rule fixed): `post/motion.py` deleted, `post/`
  reads `prior.inputs.motion` (prior §7 item 2); `post/landings.py` on B10's `LandingIndex` (its day split: a landing a
  shift puts on a test day left out and counted as sealed, one on a day outside the split refused by the index — P17
  settled by B10); `PRIOR_INTERFACE` with B10's names (`open_prior`, `OpenedPrior`, `motion`); stage C's runners take
  from `experiments/` only `post_*` modules and `prior_speaking_loop`'s `SpeakingLoop`, `Generated`, `flight_numbers`
  (prior §7 item 7), relative imports refused; C7's own check of the base dropped (B10's `masked_log_probability`
  refuses any module in training mode, D107) and D107 tested.
- `0168d457` (step 6, C4, reviewed): `experiments/post_window_loop.py` `WindowLoop`, `WindowResult`, `checked_edges`,
  `LOST_SEPARATION`; `post/reward.py` `reward`, `present_runways`, `LANDED` (a mirror of the judge's literal, pinned).

**C1 and C2 unchanged bit for bit** (step 3; A34's artefact v12, Δ 4 s, the scratchpad): the old and the new motion
equal on all 10,076,052 train and select rows; the full census's edge reference read again with the largest
difference 0.0; the census run again — its splits and every array of its reference equal.

**Tests.** The stage C files and `test_architecture.py`: 79 passed at `3725565c`; `test_post_window_loop.py` and
`test_architecture.py`: 40 passed at `0168d457`. The full ts suite on 8 workers at `0168d457`: 1,748 passed, 1
skipped, 4 min 51 s (no other job running).

**C4 as built.** Each window's commanded aircraft through B10's `SpeakingLoop` with the window's landings (D105). At
every row, observed and said, the tokens of the window's step as the traffic module's input, with the commanded
aircraft's runway and G in force before the row's words; with each row said, the speed-word mask from the state at the
start of the row (D110); after each row flown, the separation judge, a loss the commanded aircraft answers for ending
the window (`SpeakingLoop.end`, outcome `lost_separation`, reward 0). The judged scene of a row flown is kept for the
next row's mask and tokens (the reviewer's estimate before the cache: 3–4 ms a window-row in Python, approximately
40 min a pass of 2,048 windows × 300 rows without the executor). A window without other aircraft says and flies what free
generation does: words, states and the speaker's probabilities bit for bit, with any weights of the traffic module.

**Proposals.**

- **P19. The present landing direction with no landing in the 30 min** (§2 item 2, read literally): no runway is of it,
  so a landing earns 0. On A34's artefact (Δ 4 s, the real rosters): 1,119 of 40,530 train windows (2.8 %; select 3.0 %)
  have no landing in the 30 min before the first predicted step; in 1,379 (3.4 %; select 3.9 %) the recorded runway is
  not of the present direction. For the user (asked): every runway present when there is no landing; or as now.
- **P20. The judge runs after each row the executor flew**, never on the observed rows or on the state at the first
  predicted step. A window that opens inside a loss (the recorded traffic breaks the rules every hour at four of the
  five airports, readout 2026-09-27) is judged lost one row later, reward 0, whatever is said. For the user (asked).
- **P21.** A lost window's go-arounds are counted from its own words (the judge gives none).
- **P22.** The traffic module's input of a flight no longer flown is still computed (the speaker says a batch together)
  and never enters its sentence; its speed mask permits every word.

**Test gaps (the reviewer's)**, for the runner tests on real data (C8, C10): a batch of several windows (two airports,
one lost and one done, padding across windows) — A26's synthetic artefact has one flight; the speed mask acting in the
loop — a random prior does not fly a final.

## 12 The user's answers on P19 and P20 (2026-10-05)

- **P19 decided: with no landing in the 30 min before the first predicted step, every runway is of the present
  landing direction** (no direction to break). `a1fdd2a8` `post/reward.py` `present_runways` (reviewed; tested: no
  landing gives every runway, the commanded aircraft's own landing alone gives every runway, a runway 180° from a landing
  is not present). For the design author: §2 item 2's text.
- **P20: the user asked for a count first.** `a1fdd2a8` `post/traffic.py` `loss_at_first_step` and the census's
  `lost_at_first_step` (`experiments/post_windows.py`, finals from `airport_finals`, `--procedure-root`): the commanded
  aircraft on its record at its first predicted step, judged without a runway in force (the loop's state there) and with
  its recorded runway. On A34's artefact (v12, Δ 4 s, the scratchpad, 21 s): train 58 of 40,530 windows (0.14 %) without
  a runway (KRDU 37, KSTL 11, KSJC 7, KSMF 3, KMSY 0), 26 (0.06 %) with the recorded runway; select 12 and 6 of 6,199;
  every one `radar_or_vertical` (the general minimum, not in trail). For the user: keep such windows (the first row
  judged decides), or leave them out of the draw (stated in the readouts).

Tests: `test_post_scene.py`, `test_post_window_loop.py`, `test_architecture.py`, 53 passed; the full ts suite on 8
workers at `a1fdd2a8`: 1,749 passed, 1 skipped, 5 min 26 s.

## 13 P20 decided: a window that opens inside a loss is left out of the draw (2026-10-05)

The user, on §12's count: such windows are left out. `2bb77065` (reviewed; the A and D counts pinned in the runner
test): `post/traffic.py` `opens_inside_loss` — the commanded aircraft on its record at its first predicted step, with
no runway in force (the loop's state there, the larger count; a reading), loses separation that it answers for; real
and augmented windows alike, each judged on its own scene. The census counts the windows the draw keeps
(`experiments/post_windows.py` `draw_checks`: `real_kept`, `A_kept`, `D_kept`). On A34's artefact (v12, Δ 4 s, the
scratchpad):

| Split | Real kept | A kept | D kept |
|---|---|---|---|
| train | 40,472 of 40,530 (58 out) | 38,448 of 40,530 (2,082 out, 5.1 %) | 14,930 of 16,234 (1,304 out, 8.0 %) |
| select | 6,187 of 6,199 | 5,940 of 6,199 | 2,245 of 2,438 |

The augmented windows lose more: the shifts of D103 (A within ±180 s, D within ±120 s) often put the moved aircraft
inside 3 NM and 1,000 ft of the commanded aircraft at its first predicted step. For the user: whether D103's ranges stay.

Tests: `test_post_scene.py`, 13 passed; the full ts suite on 8 workers at `2bb77065`: 1,749 passed, 1 skipped, 7 min
17 s (another session's suite was running beside it; a first run was stopped from outside and run again).

## 14 C1's census of the recorded aircraft with a faulty observed track (vocabulary D111; 2026-10-05)

The order of `notes/stage_c.md` (the census of D111): stage A's line (`ed2530ae`, A40's `instructions/faults.py`)
reached this branch through stage B (`dev-two-tier-v4-prior` `c58d7772` merged, `fdfa81d3`; `dev-two-tier` with it).

**Commit** `35289115` (reviewed by an independent reviewer; fixed: the skipped steps widened by the offset of a pair
separated as one): `post/fault_census.py` `fault_census`, `window_faults`, `reads_fault`; `experiments/post_windows.py`
the census's block `faults` (`instructions.faults.faulty_flights`, vocabulary §6 item 3). Tests:
`test_post_fault_census.py`, `test_post_scene.py`, `test_architecture.py`, 53 passed; the full ts suite on 8 workers at
`35289115`: 1,782 passed, 1 skipped, 8 min 43 s.

**The census** on A34's artefact (v12, Δ 4 s), train and select, in the scratchpad, from the committed code on a clean
tree, 11 min 22 s; val not read, nothing under `outputs/`. Over each real window's steps from the commanded aircraft's
row 0 to the end of its record:

| Split | Airport | Windows | With a marked recorded aircraft in the air | Steps | Steps reading a faulty point | Tokens reading one | Losses on the records | Of them with a faulty point at or 2 Δ before the event |
|---|---|---|---|---|---|---|---|---|
| train | KMSY | 4,808 | 172 | 447,581 | 113 | 137 | 16 | 0 |
| train | KRDU | 11,864 | 413 | 1,174,512 | 291 | 327 | 345 | 0 |
| train | KSJC | 9,101 | 49 | 721,363 | 39 | 40 | 333 | 0 |
| train | KSMF | 5,377 | 112 | 547,412 | 86 | 90 | 321 | 1 |
| train | KSTL | 9,380 | 316 | 945,384 | 267 | 320 | 359 | 1 |
| train | all | 40,530 | 1,062 (2.6 %) | 3,836,252 | 796 (0.02 %) | 914 | 1,374 (3.4 % of windows) | 2 |
| select | KMSY | 657 | 25 | 61,408 | 11 | 11 | 2 | 0 |
| select | KRDU | 1,916 | 175 | 192,745 | 218 | 251 | 74 | 1 |
| select | KSJC | 1,516 | 111 | 130,476 | 38 | 43 | 66 | 0 |
| select | KSMF | 844 | 60 | 85,394 | 59 | 67 | 46 | 0 |
| select | KSTL | 1,266 | 60 | 127,866 | 81 | 99 | 46 | 0 |
| select | all | 6,199 | 431 (7.0 %) | 597,889 | 407 (0.07 %) | 471 | 234 (3.8 %) | 1 |

No criterion is applied (D7); the user decides whether such steps or windows need a rule.

**Proposals** (readings of the order and of C1's last item):

- **P23. Reading a faulty point**: a recorded aircraft reads one at a step when its row there is one of D111's fault
  rows or the row before it is (its 2 s motion reads the step that starts there). A fault's own row counts as the order
  says ("the row itself is a faulty point"), though for a jump or a held stretch its position and its motion are sound:
  the counts lean high by about one step for each fault (the reviewer).
- **P24. The steps counted**: each real window's steps from the commanded aircraft's row 0 to the end of its record
  (the census reads records; a loop flies longer or shorter).
- **P25. The losses on the records**: each window's first loss that the commanded aircraft answers for after its first
  predicted step (the event that would end the window, D93), the commanded aircraft on its record (its recorded runway
  after its first predicted step, its labelled G); "with a faulty point" when either aircraft of the pair reads one at
  the event's step or in the 2 Δ before it. A step with no other aircraft within 8 NM widened by 2,500 ft (the largest
  minimum, and the offset of a pair separated as one) is not judged.

## 15 The order of `notes/stage_c.md`: D114–D117, C6, C9 (2026-10-05)

`dev-two-tier` merged (`99d5af3d`, `092bb30d`). Each step reviewed by an independent reviewer, its findings fixed before
its commit.

| Step | Commit | What |
|---|---|---|
| C6 | `a81d0a5a` | Branch training (D37, D94): `post/branches.py` (the numbers, the branch points, `Group`, `samples`, `STATE_BOUND_M` a pinned mirror); `experiments/post_branches.py` `branch_round` (the two passes, K copies at each branch point flown as one batch, the second pass checked against the first, a differing window counted and without groups, a window halted after its last branch point); `WindowLoop.copy`, `end_step`, `samples`, `finish`, each row's tokens recorded. The review's bug fixed before the commit: the numbers keyed by the window's place in the round, not in its batch |
| D115 | `9f089780` | Every counted row weighs the same in the surrogate and the pull (the batch's sum over its counted rows) |
| D116 | `20adc232` | One token network shared by the layers (`TrafficTokens`), a step's tokens embedded once a forward pass by the first layer's module; `post-traffic-attention-v2`; `update_loss` clears the embedding after use |
| D114 | `66b8d4db` | `WindowResult.faulty_steps` and `loss_reads_fault` (C1's census definitions); `WindowLoop(faults=…)` |
| C9 | `2c60de9c` | Window B: `post/scene.py` `StartMove`, `moved_start_window`; the window loop speaks from the start's observed rows (`start_moved`); `moved_commanded` for D113 at the draw; the census's windows B |

Tests: each step's files (C6 6, D115 9, D116 16, D114 13, C9 70 with the files it touches); the full ts suite on 8
workers at `2c60de9c`: 1,795 passed, 1 skipped, 6 min 41 s (the GPU idle, no formal job running).

**C9's census** on A34's artefact (v12, Δ 4 s, the scratchpad, from `2c60de9c` on a clean tree, 8 min 26 s): windows B
drawn from a generator of their own (A's and D's draws, and their counts, are unchanged), kept unless they open inside
a loss on the moved record (D113):

| Split | Real kept | A kept | D kept | B kept |
|---|---|---|---|---|
| train | 40,472 of 40,530 | 38,448 of 40,530 | 14,930 of 16,234 | 40,423 of 40,530 (107 out, 0.26 %) |
| select | 6,187 of 6,199 | 5,940 of 6,199 | 2,245 of 2,438 | 6,184 of 6,199 |

The faulty-track counts of §14 are unchanged.

**Proposals.**

- **P26.** A window's random numbers are keyed by its place in the round (a round's windows are flown in batches).
- **P27.** The windows of one batch command different flights (a loop holds each flight once): a real window and its A,
  B or D go to different batches.
- **P28. Window B's ranges** (C9: the implementer's proposal, for the user): a turn about the airport reference within
  ±15°, a height within ±300 m, a speed scale within 1 ± 0.1, each uniform.
- **P29. D114's readings**: the steps are counted over all rows from row 0, each once; only the loss's other aircraft is
  checked (the commanded aircraft flies on the executor); an inserted aircraft keeps its source's fault rows, a D-moved
  leader's move with its record.
- **P30. C6's details**: a continuation starts at its branch row (that row's words are its first); the second pass is
  checked on the words before each window's last branch point and the positions and heights of its 2 s rows up to it;
  a differing window drops all its groups.
- **P31. For C10**: the groups of a round are large (the reviewer's estimate: several MB a group, tens of GB for a round
  held at once) — the loss is fed branch point by branch point; window B's D113 check at the draw uses `moved_commanded`;
  `start_loop` maps each window's move.

## 16 The code of C10 and C8 before B5's base (2026-10-05)

The user's order of 2026-10-05: write the code of C8, C10 and C11 now, tested on synthetic artefacts; the formal runs
wait for B5's base (and C10 for the user's criteria). `dev-two-tier` merged first.

| Step | Commit | What |
|---|---|---|
| C10, C8 | `69531aa3` | `experiments/post_train.py`: the rounds as one campaign (the draw by kind with D113, batches that command each flight once, the two passes with informative groups written per batch, one pass of the loss streamed over them, the selection readout on fixed select windows with fixed numbers, `round.json`, the checkpoint `ts-post-checkpoint-v1`, then the groups deleted); clean tree unless a smoke, an intent before a formal launch, the checks at the start; resumable, every random number keyed by the seed and the round. `experiments/post_profile.py`: one batch under cProfile by part, the round timed by part, the GPU's peak memory by part, the groups' bytes. `post/loss.py`: `one_pass` takes its pairs as a stream |

Reviewed (no bug; fixed before the commit: the groups deleted after the round's checkpoint, an empty data term refused
by name, the GPU peak reset for each part). A bug that the resume test found before the review: the traffic modules'
starting weights came from torch's unseeded generator, so a resumed campaign was not the campaign run through; the start
model and each pass's dropout are now seeded (the test: bit for bit on the CPU).

Tests: the C10 and C8 files with the loss, branch and architecture tests, 52 + 6 passed. The tests of a whole round fly
a short window (the flight inserted 8 s ahead of itself, lost at its first row); the real window's whole flight is
`test_post_branches`. The full suite waits: B5's campaign is running (a free generation) beside another session's tests.

**Proposals.**

- **P32. The windows of a round share their flights**: A, D and B are built from the same real windows that the real
  windows of the round are drawn from, so a round of 4N windows covers about N flights. D100 does not say whether each
  kind is a draw of its own. **The user, 2026-10-05: they share flights, as built.**
- **P33. A shortfall does not stop a round** (a kind with fewer windows than its count: the counts then differ from
  D100's equal counts); the record shows it.
- **P34. `rounds` is a setting of the campaign**: a campaign cannot be extended by more rounds after it ends.
- **P35. The groups at the formal size** (the reviewer's estimate: 5–9 MB a group, several GB a batch's file): read the
  profile's `groups_bytes` first; then, if needed, drop the uninformative groups inside `branch_round` and store each
  first sentence once.
- **P36. The data term reads every train sentence of the base's selection into memory** (as stage B's training): the
  profile measures it.

## 17 C11 parts 1–2; the order of `notes/stage_c.md` on stage B's round (2026-10-05)

| Step | Commit | What |
|---|---|---|
| C11 parts 1–2 | `1945e19c` | `experiments/post_training_export.py` (the windows of a campaign's rounds flown with fixed numbers: the commanded flight's head from stage A, the traffic on its records, window B's moved observed rows, each round's sentence with the window's end), `post/training_files.py` (`aeroviz-training-window-index-v1` / `-sample-v1`, `index_post_v1.json`), `aeroviz_backend/autopilot_segment/window.py` (`POST /autopilot/window-segment`: window B from its moved start, a window ended at a loss stopped there unjudged). Reviewed three times (window B's moved rows written into the set; the lost window's live segment cut at the loss) |
| Stage B's round, steps 1–3 | `6d5f2cc7`, merged into the branch | `dev-two-tier-v4-prior` (89845fa6 and later) and `dev-two-tier` merged; PRIOR_INTERFACE takes prior §7's new names (`readable_identity`, `validation_claim`, `holds_claim`, `require_selection_of`, `left_out`, `side`, `SIDES`, `REASONS`, `CELL`), each checked to exist (a new test, the reviewer's). No other change was needed: stage C takes the format names by import, calls no `kept`, reads its data term through `ArtefactSource` and takes a window ended at a loss from `said`/`states` (D118 item 6). Reviewed, no defect |

The user's decision of 2026-10-05: stage A's and stage B's Training export is listed in vocabulary §6 and prior §7, and
stage C imports exactly those names (`tests/test_architecture.py` TRAINING_EXPORT_NAMES): stage A's
`training_export` (`FORMATS`, `candidate_hae_minus_msl_m`, `candidates_block`, `events`, `split_flights`,
`vocabulary_block`), `training_flights` (`crossing_payload`, `last_state_cycle`), `training_attitude`
(`attitude_payload`, `executor_attitude`), and stage B's `prior_training_export` `procedure_block`. The rows of §6 and §7
are the designer's to write (requests file).

Tests: stage C's files, the architecture test and the backend's window and prior segments on the merged branch, 136
passed. The full suite waits for B5 (outline §5 rule 13). Step 6 waits for B12.

**Proposals.**

- **P37. The speaker's per-row records are not in a window set** (the probability of "go-around", the blocked words):
  stage B's loop gives them only for a flight the executor ended (`generated`). If the view needs them for windows ended
  at a loss, stage B's loop would give them for an ended flight too.
- **P38. A window set's `procedure` block is stage B's `procedure_block`**; the three copies of the set-file helpers
  (stages A, B, C) could become one module with the names as arguments (a follow-up, not built).
- **P39. The temperature (D121)**: stage C's loop draws at temperature 1, as stage B's runs; no other value is proposed.

## 18 Step 6 of `notes/stage_c.md`, after B12 (2026-10-06)

| Step | Commit | What |
|---|---|---|
| Merge | `814056cd` | `dev-two-tier` with B12 (`39d02ef8`) and A43; the conflict in `aeroviz_backend/autopilot_segment/backend.py` resolved to A43's warm-up with the window sets warmed after stage B's; `window.py`'s warm-up follows A43 (each set under its own lock, never the request lock) |
| Step 6 | `963ee427` | `WindowLoop` hands `start_moved`'s observed rows to `SpeakingLoop` as they are; `moved_sentences` deleted; the free-generation comparison starts through `start_moved(NO_MOVE)` |

Both reviewed by an independent reviewer, no defect. Tests: stage C's files, the architecture test and the backend's
window segment, 125 passed. The full suite waits for B5 (outline §5 rule 13).

Readings for step 6, as proposals:

- **The first-step mark**: stage C builds none. Every row is from `SpeakingLoop`'s `LoopRows` and the speaker's own state,
  copies included (branch training), so the speaker's new refusal holds without a change in stage C.
- **P40. A refused row** (`start.RowRefused`, the speaker's `accept`): B12 leaves the speaking loop as it was; the window
  loop records its tokens, speed-mask rows and faulty reads of a row before the step, so a caller that caught a refusal
  and went on would hold one row more. Today a refusal ends the run; records after the step if stage C ever retries one.
- **P41. The live check of a window's segment**: as stage B's (`prior.apart_from_exported`), the window's answer is
  compared with the export's track written to 0.1 m and not refused past a bound (A43's refusal is against unrounded
  stored states). A refusal at the rounding is a question for stages B and C together (stage B's requests).
- **The temperature (D121)**: 1, the speaking loop's default; no other value is proposed (P39).

## 19 The refused row (P40), C11 part 3 and the test stack (2026-10-06)

| Step | Commit | What |
|---|---|---|
| The order of `notes/stage_c.md` (P40) | `57578005` | A row the executor refuses (vocabulary D80) leaves the window's records as they were: its tokens and speed-mask count are kept only with the row. Reviewed, correct |
| C11 part 3 | `dfae102f` | The frontend's Training view of the windows (`aeroviz-4d/src/data/trainingWindowSample.ts`, `trainingWindowAutopilot.ts`, `hooks/useTrainingWindowLayer.ts`, `components/training/TrainingWindowSession.tsx`, "Stage C · windows" in the panel); `TrainingFlownEnd` with a loss of separation for the views (stage A's reader still takes only the judge's outcomes); the fixtures written by the export and the service. Reviewed twice (fixed: stage A's live hook also flew window flights; window B keeps the head's observed track, its moved start drawn on its own; a new set or airport never shows the old window) |

Tests: vitest 739 passed (103 files), tsc clean; stage C's Python files and the backend's window segment pass.

**A smoke on real data** (scratch only; nothing under `outputs/` or the published airports written, checked by the files'
times): the user's instruction was to copy the labelled data and read it only. A copy of A34's artefact and its
executor spec is refused by name — the executor's reference names its artefact's and its own path (the check works as
meant) — so they were read in place (every check only reads) and the copies deleted; the prior is a read-only copy of
B5's one finished fold (`prior_base_20261006/A_full_s1337/KMSY`, a fold model, not the base).

- `post_train --smoke`, one round, 5 train windows (real 2, A, D, B), K 2, on the CPU: 218 s, every check passed
  (labeller 336 flights, executor within its bounds, closed loop 258 flights at 0 m); the selection readout's reward
  1.0 at KMSY, KRDU, KSJC, KSMF and 0.0 at KSTL (one window each); no informative group at this size (no update).
- `post_training_export --smoke`: 10 windows at KRDU (3 flights; real, A, D, B), rounds start and 0, 0.7 MB, 220 s.
- A test stack (vite 5183, backend 8771, the worktree's airports link pointed at a scratch tree of links to the live
  files with the set's own `training/`): the backend warmed the set in 1.2 s; the browser check (a one-shot subagent)
  passed every step it could see — the set opens, the window block for each kind, the traffic tracks in their roles'
  colours, window B's moved start, a live segment flown (14 ms, 10 ms), no console error. The loss line was not seen:
  every window of this set landed.

**Readings and follow-ups, as proposals.**

- **P42.** The list names a window's recorded runway; the round's sentence may say another (seen at KRDU: recorded 23R,
  said 23L). The list could say "recorded".
- **P43.** Window A inserts a flight of another time: its shift is often days (−4,409,144 s ≈ 51 d in the smoke), shown
  in seconds; days or hours would read better.
- **P44.** The cursor starts at the commanded flight's row 0, before the window's row 0, so no other aircraft shows there.

## 20 D129: the window view's three changes (2026-10-06)

| Step | Commit | What |
|---|---|---|
| D129 | `18d570c4` | The window list names the recorded runway ("recorded 23R"); window A's shift in days or hours (`windowShiftText`: "−51.0 d"); the cursor at the window's row 0 once its window and round are on screen (`WindowCursorStart`, a leaf; never written on the flight before). Reviewed twice (fixed: the cursor written on the flight before for one render) |

Tests: vitest 741 passed (104 files), the cursor and panel tests after the last fix; tsc clean.

**The browser check** (a one-shot subagent, the test stack of §19: vite 5183, backend 8771): every step passed — the
list reads "DAL1427 recorded 23R …", window A's "Moved … by −51.0 d" (another: −118.0 d; window D's "−20 s"), the
other aircraft labelled at once after a window or a round is chosen (the bar at "t = 2 s · before the sentence"), the
four Draw switches stay, no console error. It saw once a jump back to the first window after A → round 0 → start that
two repeats did not show; the session file was changed (the cursor fix) while the check ran, and vite's reload of the
module mounts the session anew on its first window — the likely cause.

## 21 B13 followed (2026-10-06)

| Step | Commit | What |
|---|---|---|
| Merge | `138b0ca0`, `ed76d6bd` | B13 (`142272f0`) and `dev-two-tier` |
| B13 followed | `2933a66b` | PRIOR_INTERFACE: `holds_claim` (deleted by B13) replaced by `holds_written_claim`, with §7's `written_claim`, `spend_validation_claim`, `CLAIM_SPENT_BY` (each checked to exist). `prior.apart_from_exported` now refuses past the executor's bound or at another outcome or end cycle (D127), so the window export writes each round's track unrounded (`aeroviz-training-window-sample-v2`, the frontend mirror with it) and the window route passes the round's sentence; the traffic and window B's moved start stay rounded (drawn only). Reviewed, correct |

Tests: Python 144 passed (the backend's window and prior segments, the architecture test, every `test_post_*`), vitest
741 passed, tsc clean.

The test stack follows: the smoke set exported again (v2, 0.9 MB for 10 windows, from 0.7 MB rounded), the test backend
restarted on the new code; a live segment flown against it is the export's flight to 0.0 m over 126 rows (judged
landed, 26 ms).

**Reading, as a proposal.** P45: prior D127 is followed for windows (the window route uses stage B's check, which now
refuses past the bound); the unrounded track makes a set's file larger (here 1.3×) — a formal set's size is to be read
at its export.

## 22 The specifications of the milestones done (moved verbatim from the design, 2026-10-06)

Moved from `design/post_training.md` §8 when the design was cleared of the milestones done (Claude's check of stage C,
2026-10-06). The design keeps C8, C10 and C12; its §7 gives the code and the import rules.

**C0. Package and layout.**

- A new package `post/`: the scene and its steps, the edge features, "established", the separation masks, the traffic
  module, the reward, the branch groups and the loss. It reads `instructions/` and the names of prior §7; it does not
  import `autopilot/`. The window loop, which joins the speaker, the start of a closed loop and the scene, is a module
  shared by the runners of stage C (under `experiments/`, not a runner), as a runner joins a model to the executor; it
  runs the commanded aircraft through the prior's step of a speaker's closed loop (prior §7 item 7, D106).
  The separation judge stays in `inference/separation.py`.
- The architecture test (`tests/test_architecture.py`, read by the names imported, as prior D69's test): `post/` imports
  from `prior/` only the names of prior §7; the runners of stage C import from `autopilot/` only the names of
  vocabulary §6, from `prior/` only those of prior §7, and from the module of prior §7 item 7 only its names given
  there. After each merge of stage B, the test's list follows prior §7's "Code" column (B9's and B10's names).
- The runners: `post_windows` (the windows and their census, C1), `post_train` (the rounds, C10), `post_readout` (a
  window readout), `post_training_export` (C11). New code; the archived runners stay as they are.

**C1. Windows and scenes** (§3; D29, D93, C32).

- A window: one commanded flight with a closed-loop sentence at the chosen Δ (every such flight, inside the base's
  selection or not, D76), and its scene: the stored signals of every other flight of the airport and the split, on the
  scene's steps, from the commanded aircraft's row 0 to its end (D93).
- The augmented windows A (one recorded flight of the same airport and split, from another time, inserted with its
  record shifted in time) and D (the aircraft next ahead on the approach clock at the first predicted step, its record
  shifted in time). The shifts: D103. A window that opens inside a loss of separation is left out of the draw (D113).
- The census (`post_windows`, outline §5 rule 12): for each airport and split (train, select), the windows, the other
  aircraft at the first predicted step, the share with a leader in the air on the same runway or a runway that counts
  as one, and the windows near the cut between two operating days. Measured on all train days in a scratch directory
  after C1; the user chose the count of each kind of window in a round on it (D100).
- The recorded aircraft with a faulty observed track (vocabulary D111: the marks of `instructions/faults.py`), in the
  census, for each airport and split (train, select): the windows with a marked recorded aircraft in the air; the
  scene steps at which a recorded aircraft reads a faulty point (its row is one, or its 2 s motion reads one), and the
  tokens of the traffic attention at those steps; and, with every aircraft on its record, the losses of separation
  (the separation judge, reading VISUAL) of a pair in which one aircraft reads a faulty point at the event's step or in
  the 2 Δ before it, against all losses. A recorded aircraft's jump can make a loss that did not occur (reward 0, D30)
  and a token with a motion of hundreds of m/s. No criterion is applied: the user decides from the counts whether such
  steps or windows need a rule (Claude's check of stage B, 2026-10-05).
- Tests: a window never reads a test day (C32) or a flight of another split; its other aircraft at a step are exactly
  the flights in the air then; a recorded aircraft's motion comes from its 2 s displacement (a change of its stored
  track, ground speed or vertical rate changes nothing); its R is absent before its first predicted step; the steps are
  on UTC multiples of Δ (vocabulary §6 item 7).

**C2. Edge features** (§3, §4 item 1; D23, D31).

- The features of §3 for every other aircraft at each step, at fixed SI scales; the conformance reference (fixed
  scenes of the train days, seed 1337) written by the census beside its output (D104); the check run by every process
  that computes them.
- Tests: no feature uses a later row (a change of any row after the step changes nothing); D23 in a scene (a change of
  one aircraft's runway word changes no input and no edge feature up to its first predicted step, bit for bit); a
  permutation of the other aircraft permutes their features; an aircraft whose motion is unknown gives the flag and
  zeros.

**C3. Separation judge, "established" and the speed-word mask** (§3; D92).

- `inference/separation.py` on v4: its inputs from vocabulary §6 item 4 (the position relative to a candidate) and
  "established" of D92 (the region of prior §7 item 6, 20°; a recorded aircraft's G, D99); the reading VISUAL; the event of a
  window: the first loss of separation for which the commanded aircraft answers.
- The speed-word mask of §3, given to the speaker as a mask of a caller; its prediction to the threshold at the
  executor's rate (vocabulary §6 item 1).
- The regulation text that D92 reads ("established", the in-trail rule) cited to its paragraph, in
  `docs/literature/arrival_separation/`.
- Tests: "established" is a function of one row (a change of later rows changes nothing), the same for a commanded and
  a recorded aircraft, false while G is true; the speed-word mask masks nothing outside its conditions and nothing when
  every word falls short.

**C4. The window loop** (§2 items 1–3; D29–D31, D91, D93, D105; prior D96, D106; vocabulary D97).

- The commanded aircraft through the prior's step of a speaker's closed loop (prior §7 item 7, D106), which joins the
  start of a closed loop (vocabulary §6 item 5; the most go-arounds 2, D91), the prior's one function of a loop's row
  (prior §7 item 2) with the window's landings (D105) and the speaker with the random numbers of D94, the masks of a
  caller (D91, the speed-word mask) and the traffic module's input; the scene step by step; the separation judge at each step; the end of the window (D93); the reward of D30, with the
  present landing direction from the window's landings (D105). The motion of a recorded aircraft from prior §7 item 2
  (D106): the mirror `post/motion.py` is deleted.
- Tests: a window with no other aircraft says and flies what the prior's free generation says and flies for the same
  flight with the same random numbers and the same bound: the same words and states, bit for bit in the same batch (D29; §2 item 1; §6.4); the
  reward table of §2 item 2; the commanded aircraft's landing context never counts its own landing (D31) and counts an
  inserted or moved aircraft's landing at its time in the window (D105); the loop reads no time limit (vocabulary
  D90).

**C5. The traffic attention** (§3; D98).

- The module of §3 at each layer through prior §7 item 5, its input passed by the speaker (prior D96): the tokens of
  D98, embedded once by one token network shared by the layers (D116); its own learning rate.
- Tests: with the module added at its start, every output of the base is the same, bit for bit; with no other aircraft
  its output is zero at any weights; a permutation of the other aircraft changes nothing.

**C6. Branch training** (§2 item 9; D37, D94).

- The two passes of D94: the random numbers of each window and each continuation from the seed, the round, the window,
  the branch point and k; the second pass checked against the first; the copies of the loop, the speaker and the window
  state at each branch point (vocabulary D97, prior D96); the continuations of one branch point of all windows as one
  batch; the branch groups and their advantage after the branch point.
- Tests: the test of D94 (D37 item 4); the executor in inference mode flies the states that it flies with gradients
  (D37 item 6); a group whose rewards are all the same gives no sample; the advantage reaches only the words after its
  branch point; a window whose second pass differs from its first is counted and gives no sample.

**C7. The loss** (§2 item 5; D36, D76).

- The clipped-ratio surrogate (ε = 0.2) on the log-probabilities of the words said, under the speaker's records of the
  permitted words, with the traffic module's input (prior §7 item 4, D96); the pull to the base (0.04, the KL on the
  sampled words; the base's log-probabilities under the same records); the data term (1) on single-aircraft samples of
  the train days' closed-loop sentences in the selection `landed` (prior §7 item 4); the module's own learning rate;
  one pass over the samples of a round; every counted row weighs the same (D115). The surrogate and the KL with dropout off, the data term with the base's
  dropout (D107); the records of first sentences and continuations joined into one batch (prior §7 item 3, D106).
- Tests: at the parameters that spoke the words, the ratio is 1 within the float tolerance of §6.4; a word that a record
  blocks takes no probability; when the module gives zero, the data term equals the prior's teacher-forced loss on the
  same batch (the same dropout state); a surrogate or KL asked of a model in training mode is refused (D107).

**C9. Window B** (§2 item 4). After the moved start of vocabulary D97.

- Moved starts through the start of a closed loop; the ranges of the turn, the height change and the speed change:
  D123; windows B in the census of C1.
- Tests: the prior reads the moved observed rows that the start gives back; D23 holds; a move of zero gives the window
  without the move, bit for bit.

**C11. The Training view of stage C (outline §6).** The last milestone of the stage. The window export (the archived
`experiments/window_training_export.py`, R36, rewritten): windows of recorded traffic with the commanded aircraft
flown on the words of the post-trained model and the other aircraft on their records; the separation judge's events;
the outcome of each window; the rounds of the post-training side by side. The frontend's Training view shows the
traffic window with the five columns of the commanded aircraft, and a click on a word flies its segment live with the
executor of A23. New schema names; its own index beside the old one; the intent of each set; a test stack and the
browser check (outline §6).

## 23 After Claude's check of stage C: D130, the validation readout, the docstring (2026-10-06)

| Step | Commit | What |
|---|---|---|
| Merge | `3ccb82f8`, `4882c130` | `dev-two-tier` (the design cleared, `0f4f4fff`, `b29d2c6d`; D130; then D131 and prior §7 item 1 with the claim names) |
| D130, docstring | `02a85049` | `update_pairs` takes each groups file's branch groups in an order drawn by the pass's generator (the round's numbers: seed, round, 1), then `update_groups` at a time; the data term's sentences are drawn by the same generator after it. Test: the same numbers give the same updates, other numbers others, an update mixes groups spoken more than one update apart; the resume test stays bit for bit. `inference/separation.py`: "established" is the caller's, by D92 (comment only). Review: no S1/S2; two S3 (the test could not tell a shuffle inside an update from one of the file; a long line), both fixed |
| The validation readout | `0c95a821` | `experiments/post_validation.py` (§8 C10): refused unless the campaign has done every round (a smoke reads earlier) and holds the round; the checks first (closed loop, edge reference); then the read's lock, the round's checkpoint (`round_model`), the claim (in the campaign's directory, reader `post_validation`, options round and device), the val recount (`require_selection_of`), and only then val opened (`split_data`); the readout is `selection_readout` on val's windows, with each airport's coverage (real windows, left out inside a loss, read); `readout.json` last, then the claim spent. `post_train`: `open_context(splits=…)`, `readout_pool`, and `selection_windows` / `selection_readout` take a split (default select; the campaign, profile and export unchanged). PRIOR_INTERFACE: `claim_validation_read`, `lock_val_read`, `settle_written_claim` (user, 2026-10-06; prior §7 item 1). Tests on the synthetic campaign, its train split standing in for val: only val opened, after the claim, no data term; a second read refused (same output, another output); a read stopped before its readout runs again only with the same options; the lock refuses a second run; a kill after the readout is settled, then refused; another round, and a campaign with rounds to do, refused before the claim; a smoke reads select, claims nothing, and gives back round 0's selection readout exactly; the config shows no val counts. Review: no S1; one S2 (the coverage not recorded: val cannot be read again to learn it) and four S3 (the lock and the settle untested, the `data` flag untested, the formal read before the last round, no end-to-end equality), all fixed but the base's formality (P48); second round: no S1/S2, one S3 (the docstring claimed the weights), fixed |

Tests: every `test_post_*`, the architecture test and the backend's window segment: 135 passed.

**Readings, as proposals.**
- P46: the validation readout reads at most the campaign's `select_per_airport` real windows of each airport of the val
  days, drawn as the selection readout draws the select days' (the same seed stream, D113 applied), with the same random
  numbers, and states each airport's coverage. §8 C10 gives the readouts, not the count. Every val window is one option
  away if the user wants it.
- P47: the claim of the val read is held in the campaign's directory (the post-trained model's run, as prior D119 holds
  it in the prior's run); its options are the round and the device (as prior D128's validation readout records its
  split and device; this readout has one split). A second campaign has its own claim: D85's "once for each stage" is
  guarded per campaign, as the prior's per run. With the formal read refused before the last round, a campaign stopped
  on purpose before `rounds` has no formal read (its settings would be changed by a resume with fewer rounds).
- P48: the readout does not refuse a campaign whose base is a smoke prior or a fold (stage B's validation readout does,
  for its own read); `post_train` refuses neither today. The place for it is `post_train`'s formal start, which opens the
  base's run.

## 24 After B5: the formal census, C8 under way, the base on its device, D132 (2026-10-06)

| Step | Commit | What |
|---|---|---|
| Merges | `0f100083`, `812a47b1`, `fa141377` | `dev-two-tier` (B5's readout and intents; B6; D132). `812a47b1` fast-forwarded onto `dev-two-tier` (the user, 2026-10-06) |
| Full ts suite | — | 1,840 passed, 1 skipped, 8 min on 6 workers (stage B's export on the CPU beside it) |
| The formal census (D104, the user's order) | `44fb8af8` | `post_windows` writes it directly under `4dTrajectory/outputs/POOLED/post/<id>`, of every window of train and select, from a clean tree; anything else under the outputs refused (through the worktree's link or its target). Review: one S2 (the link untested), fixed with a test that stands the outputs in through a link; the outputs root imported from `repo_layout` |
| The base on its device | `6d6502fa` | C8's first profile on the GPU failed in the pass: the loss's pull read the base on the CPU (`post_profile` never moved it; `run_campaign` did). `open_context` now holds the base on the context's device in eval mode for every caller. Review: no findings |
| D132 | `483b81d7` | `open_context(formal=…)`: a formal `post_train` start and a formal `post_validation` read refuse a fold (`config.json` run `held_out`) or a smoke prior (checkpoint run `sample`) by name; smoke runs and the profile and the export do not check. On the real runs: B5's base passes, its fold `C_full_s1337/KMSY` is refused. Review: no S1/S2; two S3 (the message read `sample` as a count, the test did not pin "before the claim"), fixed |

**The formal census** (`outputs/POOLED/post/windows_20261006`, from `44fb8af8` on a clean tree, 7 min, read-only with
`SHA256SUMS` and its log): train KMSY 4,808, KRDU 11,864, KSJC 9,101, KSMF 5,377, KSTL 9,380 windows; select 657,
1,916, 1,516, 844, 1,266. Its edge reference: 323 steps, 420 tokens, read again with the largest difference 0.0.

**C8, the base's go-around on the final on the select days** (`outputs/POOLED/post/c8_20261006/free_generation_select`,
stage B's `prior_free_generation` on B5's base, the options of its val read: 200 flights × 2 samples an airport, seed
1337, chunk 400; from `44fb8af8`, read-only). Inside the selection (landed), pooled over the airports:

| Days | Stratum | Sentences | Go-arounds said | Final rows | Mean probability of "go-around" a final row |
|---|---|---|---|---|---|
| select | straight-in | 1,132 | 6 | 39,727 | 7.6e-05 |
| select | vectored | 746 | 5 | 24,027 | 1.2e-04 |
| val (B5's readout) | straight-in | 1,242 | 5 | 44,773 | 4.2e-05 |
| val (B5's readout) | vectored | 652 | 9 | 20,586 | 2.6e-04 |

By airport, the select days' go-arounds: KMSY 3, KRDU 2, KSJC 1, KSMF 4 (one at the bound of D91), KSTL 1. The base says
"go-around" by itself (§2 item 6); no criterion is applied (D7).

**The profile**: the first run (32 windows, batches of 16, K 8) failed in the pass (above) and was moved aside as
`profile_32.aborted-20261006T082024Z`; before it failed it spoke the round in about a minute and wrote 3.2 MB of groups
(2 of 4 files non-empty). Run again from `483b81d7`; its numbers follow.

**Stage C and D133.** `notes/stage_c.md` (on `docs-training-view`, `4fa9f2ee`) step 6: the Training view's files and
the backend's routes are not changed while the shared layout is built; none of the commits of this section touches
them.

## 25 C8: the profile at the formal size and the proposal of O13 (2026-10-06)

| Step | Commit | What |
|---|---|---|
| The profile's record and one update's memory | `c396f9be` | After the 256-window profile ran out of the GPU's memory in the pass (16 groups an update) and lost its speaking with its record: `profile.json` written after each part; `pass_memory` before the pass — one update's forward and backward (no step) on groups that bound any update of k ≥ 2 groups (the longest, the widest, then the next longest; for k = 1 the longest and the widest), an out-of-memory recorded as the measurement (`ts-post-profile-v2`). Review in three rounds: S2 (the first k groups written are the shortest), S2 (rows bound but not traffic), S3s; all fixed |
| Merge | `7288de0e` | Stage C onto `dev-two-tier` (the user, 2026-10-06), a merge in the main tree (the profile ran from the worktree); 140 stage C, architecture and window-segment tests passed on it |

**The profiles** (B5's base, A34's artefact v12 at Δ 4 s, executor v17, the formal census; read-only under
`outputs/POOLED/post/c8_20261006/`, groups deleted after reading). Round 0's draw, K 8, 64 data sentences an update,
placeholder rates (prior 1e-5, traffic 1e-4, weight decay 0.01, traffic 64 / 4):

| | `profile_32` (`483b81d7`) | `profile_256` (`c396f9be`) |
|---|---|---|
| Windows (real, A, D, B) | 8 each, batches 14 / 8 / 8 / 2 | 64 each, batches of 64 (8 left out inside a loss: 4 D, 3 A, 1 real) |
| One batch under cProfile | 14 windows, 55 s | 64 windows, 113 s (speaker masks 30 s, edge features 4.9 s, prior step 4.1 s, separation scene 3.5 s) |
| Speaking the round (D94's two passes) | 76 s (2.4 s a window) | 543 s (2.1 s a window) |
| First pass's ends | 29 landed, 3 lost separation | 189 landed, 53 lost separation, 5 timeout, 4 unstable at minimums, 4 ground contact, 1 crossed another runway |
| Informative groups, bytes | 5, 3.2 MB | 108 (42 % of the windows), 104 MB (0.96 MB a group) |
| GPU while speaking | 0.14 GB | 1.3 GB |
| The pass | 2 updates, 0.7 s, 2.8 GB (update groups 16, 5 held) | update groups 4: 29 updates, 10.5 s, 4.4 GB |
| Selection readout | 50 windows, 30 s | 50 windows, 14 s |
| Host | 5.1 GB peak resident | 5.3 GB peak resident |

One update's memory (`profile_256`, the bounding groups: 480 rows, traffic 4; the data term's 64 sentences, 175 rows):
1 group 2.2 GB, 2 groups 2.9 GB, 4 groups 4.3 GB, 8 groups out of memory (the RTX 4060's 7.6 GB, about 1.2 GB held by
other processes). The first 256-window run (update groups 16) ran out of memory in the pass. A run stopped when another
session's job took the host's memory (0 GB available, swap full) had written nothing.

**Reading, as a proposal: the settings of O13** (Claude's, from these numbers; the user chooses, D7 is not applied):

| Setting | Proposal | Basis |
|---|---|---|
| Update groups | 4 | 4.3 GB at the bound; 8 run out of memory |
| Data sentences an update | 64 | measured inside that bound |
| Batch windows | 64 | 2.1 s a window at 64 against 2.4 s (and 4 s under cProfile) at 14; 1.3 GB on the GPU while speaking; larger batches not measured |
| Continuations K | 8 | D94 |
| Windows of each kind in a round (real, A, D, B) | 1,000 each | D100 for real, A, D; B equal by the rule that paired quantities default to equal; the train days admit 40,472 real, 38,448 A, 14,930 D, 40,423 B windows (C9's census) |
| Rounds | 10 | at the sizes above a round is about 2.5 h (speaking 2.4 h, about 1,700 groups and 1.6 GB on disk, about 420 updates in 2.5 min, the selection readout below 5 min): about 25 h in all |
| Learning rate, prior | 1e-5 | a thirtieth of the base's 3e-4; at this rate one pass clipped 1.3 % of the words, its pull term 0.006 |
| Learning rate, traffic modules | 1e-4 | new modules starting at zero output (D116) |
| Weight decay | 0.01 | the base's |
| Select windows of each airport | 200 | 0.28 s a window: about 5 min a round; the validation readout reads the same count (D132) |
| Traffic attention: hidden width, heads | 64, 4 | as profiled (no measurement chose them) |

Not measured: the time with other jobs on the CPU (the 256 run met a load of about 2–3 of 28 cores); larger batches;
other traffic shapes. The first pass loses separation in 21 % of `profile_256`'s windows (the record does not split
the ends by kind).

## 26 C10: the formal post-training launched (2026-10-06)

The user's choice of O13 (2026-10-06): every setting as proposed in §25 (1,000 windows of each kind, 10 rounds, update
groups 4, 64 data sentences, batch windows 64, K 8, rates 1e-5 / 1e-4, weight decay 0.01, 200 select windows an airport,
traffic 64 / 4); the criterion that chooses the round (D7) later, before the validation readout.

| Step | Commit | What |
|---|---|---|
| The intent | `d5a2d322` | `post_train_20261006` in `docs/experiments/intents.json` (title, intent, design, one line per round), checked by the publisher's loader |
| The full ts suite | — | On the merged `dev-two-tier`: 1,845 passed, 1 skipped, 8 min 36 s on 6 workers |
| The launch | — | `outputs/POOLED/post/post_train_20261006`, from the main tree at `e5f54dd4` on a clean tree (2026-10-06 12:20 local), every path repository-relative. A first launch from the worktree with absolute paths stopped at the executor's check before writing anything (its reference names its inputs by repository-relative path) |

The worktree's `aeroviz-4d/public/data/airports` points at the scratch tree of the stage C test stack (log §19); the
formal census and the profiles read the CIFP procedures through it. Every live airport file is there as a link to
the live file (checked file by file), and the base's procedure-mask check (prior D106) passed at every open: their
results stand. The campaign reads the main tree's live airports.

**Stopped and relaunched with parallel speaking (2026-10-06, the user).** The first launch spoke on one CPU core (one
Python thread at 93 %, 1 of 28 cores; the GPU forward of the 3.8 M-parameter prior is a few % of a batch: one real
batch of 28 windows, 74 s under cProfile — the start's rebuild of the flights from disk, stage A's `start_moved`,
26 s for the two passes; the speaker's masks 14 s; compile set-up 10 s once; inputs, edges and scene 8 s; the prior's
forward 5 s). The user stopped it 10 min into round 0 (moved aside as `post_train_20261006.aborted-20261006T103231Z`)
and asked for the change:

| Step | Commit | What |
|---|---|---|
| Parallel speaking | `0268e0ad` | `--speak-workers N`: `Speakers`, a `ProcessPoolExecutor` forked before the process uses the GPU (sharing the context's memory), every worker started at once, one torch thread each (a fork after the parent's CPU threads ran hung in the reviewer's probe), one task a batch; a worker redraws the round's windows and refuses others; the round's model by a file; a dead worker fails the round (`BrokenProcessPool`). The worker count in `round.json` (information; `campaign.json` and its schema unchanged: the frontend fixture pins the name and step 6 keeps it). Review in two rounds: four S2 (the thread hang, a test that could not fail, a dead worker hanging the round, memory at N unchecked) and three S3; all fixed but the memory at N, watched in round 0 |
| The GPU check | — | 64 windows, one round, on the GPU, 1 against 3 workers (twice, on the pool and on the executor code): the windows, the speaking record, the groups' bytes and the selection readout identical; the pass and the weights equal to float rounding (largest weight difference 9e-7; the loss equal to its last printed digit): the parent's own GPU history changes its kernels, as on any resume |
| Merge | `e045d5c3` | `dev-two-tier` (docs) into the branch, `dev-two-tier` fast-forwarded to it |
| Relaunch | — | From the worktree at `e045d5c3` (the user: the branch, since `dev-two-tier` may change), repository-relative paths, `OMP_NUM_THREADS=1`, `--speak-workers 3`, 13:27 local. `campaign.json` records its inputs under the worktree's path: the worktree stays until the campaign's export and validation readout are done. At the start: three workers at 75–80 % CPU each, 14 GB of host memory available, 0.25 GB of GPU each |

**Round 0 and the workers (2026-10-06).** Round 0 ran 13:30–14:37 with three workers on `e045d5c3` (its
`round.json` records `98a89871`, the head at its end: the fix below was committed in the worktree meanwhile; it
releases cached memory only). The draw: 1,000 windows of each kind, 114 left out inside a loss (66 D, 45 A, 2 B,
1 real), no shortfall; 63 batches. The first pass: 2,907 landed, 866 lost separation, 95 timeout, 60 unstable at
minimums, 52 ground contact, 11 crossed too high, 5 crossed another runway, 4 crossed off the runway; 1,477 of 3,068
groups informative (1.4 GB, deleted after the round). The pass: 391 updates, clipped share 2.3 %, pull term 0.012,
data term 1.086. The selection readout's mean reward: KMSY 0.82, KRDU 0.75, KSJC 0.835, KSMF 0.785, KSTL 0.775. No
criterion is applied (D7).

| Step | Commit | What |
|---|---|---|
| The GPU cache | `ea2050ff` | With speakers on the GPU, the campaign's process releases the memory it cached in the last pass before each round's speaking (about 4.6 GB reserved after the pass; three workers speaking need about 3 GB: round 1 would have run out of memory). Review: no findings |
| A supervisor | — | A detached script beside the campaign (the user's rule: five workers when the load is below 10): at round 0's checkpoint it resumed the campaign on the fixed code with five workers (load 2.9, 13 GB of host memory, other processes 0.5 GB of the GPU) |
| Four workers | — | The GPU sampler showed round 0's pass at 7.0 of the GPU's 7.6 GB with three idle workers holding their contexts; five would very likely run round 1's pass out of memory. The five-worker process was stopped in its start checks and the campaign resumed at 14:41 with four (round 1 begun and moved aside as `round_1.aborted-…`); the supervisor reports only. The GPU, not the CPU, caps the workers at four on this card |

**Rounds 1–3, and round 4 out of the GPU's memory (2026-10-06).** Rounds 1–3 with four workers (14:41–16:44): the
selection readout's mean reward KMSY / KRDU / KSJC / KSMF / KSTL — round 1 0.8745 / 0.80 / 0.845 / 0.78 / 0.7945,
round 2 0.835 / 0.77 / 0.83 / 0.79 / 0.785, round 3 0.875 / 0.80 / 0.855 / 0.775 / 0.815 (no criterion, D7). Round 4's
pass ran out of the GPU's memory at 17:16 (four workers; the main process at 6.2 GB) and again on its resume with three
(the main process at 6.3 GB): round 4's groups reach 668 rows and traffic 7 (13 groups past 480 rows; the C8 profile
saw 480 and 4), and an update of four groups is padded to its longest and widest. Both half rounds moved aside
(`round_4.aborted-…`).

| Step | Commit | What |
|---|---|---|
| The pass in pieces | `832555a5` | The user's decision (2026-10-06; an exception to `notes/stage_c.md`'s "C runs C10 without changing code"): `post.loss.update_step` — each branch group a piece, its surrogate and pull divided by the whole update's counted rows, its backward at once; the data term whole (its dropout); `update_loss` kept as the reference and checked equal (parts to 1e-6, gradients to rounding, the clipped counts equal; a piece normalised by its own rows fails the test). `update_pairs` yields a piece a group; the profile measures pieces, with the group of the largest rows × traffic beside (a proxy, not a bound; `ts-post-profile-v3`). Review in two rounds: two S2 (a test of the branches not updated; the profile's bound wording), four S3; all fixed. On round 4's real groups with round 3's model: an update of the four largest (668 rows, traffic 7) peaks at 1.58 GB, the same for one, two or four groups |
| Merge | `af36ebe8` | `dev-two-tier` (docs only: D137–D139, the notes) into the branch; `dev-two-tier` fast-forwarded to it |
| Resume | — | From the worktree at `af36ebe8`, 18:36, five workers (load 1.3, 23 GB of host memory, the GPU 0.5 GB used; the speaking's peak about 1.24 GB a worker): round 4 from round 3's checkpoint. Rounds 0–3 ran the whole update; rounds 4–9 run it in pieces (the same loss, float rounding apart) |

**Round 4 and five workers (2026-10-06).** Round 4 ran 18:36–19:13 with five workers and the pass in pieces: the
selection readout's mean reward KMSY / KRDU / KSJC / KSMF / KSTL 0.865 / 0.805 / 0.865 / 0.785 / 0.81; landed 83.7 %,
lost separation 13.2 % of its 1,000 windows (round 0: 80.2 %, 14.9 %). The speaking with five workers peaked at 7.24 of
the GPU's 7.59 GB; round 5's speaking ran out of the GPU's memory at 19:47 (62 of about 63 batches spoken; the five
workers held 1.2–1.7 GB each at once). The campaign resumed at once with four workers (round 5 begun anew from round 4's
checkpoint; the half round moved aside): their speaking peaked at about 5.6 GB in rounds 1–3, and the pass in pieces
needs about 1.6 GB. Four is the most this GPU holds safely; the user's rule of five cannot hold on it.

**C10 done (2026-10-06, 22:46).** `post_train_20261006`: ten rounds, each closed by its checkpoint; the data read-only
(`SHA256SUMS`, 219 files; the run's logs in `logs/`). The aborted half rounds (`round_1/4/5.aborted-…`, 3.3 GB of
groups) are kept read-only with it. The selection readout (the same 1,000 select windows and numbers every round):

| Round | Workers | Landed | Lost separation | Mean reward | KMSY | KRDU | KSJC | KSMF | KSTL | Informative groups | Updates | Pull (KL) | Data term |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 80.2 % | 14.9 % | 0.793 | 0.820 | 0.750 | 0.835 | 0.785 | 0.775 | 1,477 | 391 | 0.0117 | 1.0860 |
| 1 | 4 | 82.6 % | 13.9 % | 0.819 | 0.875 | 0.800 | 0.845 | 0.780 | 0.794 | 1,253 | 331 | 0.0184 | 1.0807 |
| 2 | 4 | 81.0 % | 15.7 % | 0.802 | 0.835 | 0.770 | 0.830 | 0.790 | 0.785 | 1,237 | 332 | 0.0216 | 1.0773 |
| 3 | 4 | 83.2 % | 13.7 % | 0.824 | 0.875 | 0.800 | 0.855 | 0.775 | 0.815 | 1,144 | 307 | 0.0235 | 1.0728 |
| 4 | 5 | 83.7 % | 13.2 % | 0.826 | 0.865 | 0.805 | 0.865 | 0.785 | 0.810 | 1,205 | 324 | 0.0262 | 1.0713 |
| 5 | 4 | 83.8 % | 13.2 % | 0.830 | 0.894 | 0.800 | 0.850 | 0.785 | 0.820 | 1,231 | 331 | 0.0287 | 1.0783 |
| 6 | 4 | 85.1 % | 11.3 % | 0.842 | 0.870 | 0.830 | 0.850 | 0.795 | 0.865 | 1,106 | 303 | 0.0321 | 1.0763 |
| 7 | 4 | 84.8 % | 11.8 % | 0.840 | 0.895 | 0.830 | 0.840 | 0.780 | 0.855 | 1,201 | 326 | 0.0330 | 1.0744 |
| 8 | 4 | 85.7 % | 11.7 % | 0.849 | 0.890 | 0.805 | 0.855 | 0.815 | 0.880 | 870 | 244 | 0.0348 | 1.0687 |
| 9 | 4 | 85.0 % | 12.0 % | 0.840 | 0.885 | 0.810 | 0.865 | 0.820 | 0.820 | 1,161 | 315 | 0.0344 | 1.0759 |

Rounds 0–3 ran the whole update, 4–9 the update in pieces (the same loss to float rounding). No criterion is applied
(D7): the user's criterion chooses the round, then the validation readout reads val once (`post_validation`, D132).
A landing share over 1,000 windows carries about ±1.2 points of binomial noise. `campaign.json` records its inputs
under the worktree's path: the validation readout reads them, so the worktree is kept until it has run.

**The user's decisions after C10 (2026-10-06).** The criterion that chooses the round (D7): the earliest round within the
selection readout's noise (about one point) of the best — round 6 (landed 85.1 %, lost separation 11.3 %, mean reward
0.842; the best is round 8 at 0.849). The validation readout is NOT run yet: the post-training may be extended; the
worktree `two-tier-v4-post` is kept (its `campaign.json` paths) until the user decides.

## 27 C14: a resume may raise the rounds (2026-10-06)

The order is `notes/stage_c.md` of 2026-10-06 (late evening), and the design is post-training D157 and §8 C14. Before
the work, `dev-two-tier-v4-post` was fast-forwarded to `dev-two-tier` `63ca94e0`.

| Step | Commit | What |
|---|---|---|
| C14's code | `21547ad4` | See the list below |
| P47 | `75fe9ff2` | The user's decision (2026-10-07) on reading P47: `open_campaign` refuses a raise of the rounds once the campaign holds `post_validation`'s claim of its val read, spent or not. A resume of the same count still opens. `CLAIM_READER` moves into `post_train`, because `post_validation` imports it. Tests: an unspent claim (`test_post_train`) and a real formal read (`test_post_validation`); both fail without the refusal. Review: no findings. C10's directory holds no claim, so the `--rounds 14` resume passes |

**The code.**
- `experiments/post_train.py` (`open_campaign`): a resume is accepted when its inputs equal the record's with the
  rounds left out, and its rounds are at least the record's. More rounds raise the record's count.
- The recorded paths are compared through `inputs_here`: `training_export.this_checkout`, imported as `model_speed`
  imports it, over the keys of `INPUT_PATHS`. The record keeps its paths.
- Each resume's entry gives its time, its commit, its checks, the paths it read (`inputs`) and the rounds
  (`{"before", "after"}`).
- `experiments/post_validation.py` reads the campaign's paths through `inputs_here`.
- `docs/experiments/intents.json` gives rounds 10–13 their intent.

**The tests** (each fails when the mapping or the raise is taken out):
- 2 rounds raised to 3 runs round 2 only, with rounds 0 and 1 byte for byte unchanged. It gives the round 2 of a
  campaign of 3 rounds from its start: the checkpoint bit for bit (model, optimizer, identity), the `round.json` (its
  commit and time aside) and the readout.
- Fewer rounds are refused by name, and so are more rounds together with another seed, number of update groups,
  smoke flag or executor. The record is not touched.
- A campaign recorded under a deleted worktree's data trees resumes from the main checkout, and a path that maps to
  other data is refused.
- `post_validation` reads the mapped paths.
- Results: `test_post_train`, `test_post_validation` and `test_architecture` 62 passed. `test_post_training_export`,
  `test_model_speed`, `test_post_profile` and the publisher's tests also pass.

**Review** (opus, independent, `review_guide` §3 step 6): no S1 and no S2. Two S3 findings go to the requests note:
- `model_speed` and `post_training_export` keep their own five-key lists;
- the entry records the mapped path, not the one typed.

The reviewer also checked that the real campaign `post_train_20261006` will be accepted from the main checkout with
`--rounds 14`, using a read-only copy of its `campaign.json`. All five mapped paths exist. The record goes from 10 to 14
rounds and keeps the worktree paths.

**The readers of a campaign's `"inputs"`** in `experiments/`:
- `open_campaign` maps both sides; `settings_of` reads only the settings.
- `post_validation`, `post_training_export` and `model_speed` map through `this_checkout`. `model_speed.open_round` gets
  a record that is already mapped.
- `post_profile` and `prior_behaviour` write records of their own and do not read one.

**The run of rounds 10–13 (2026-10-07).**
- The user merged `dev-two-tier-v4-post` (`bd7929a5`), and stage D reported its GPU and timing steps done (`1c6146bb`).
- Before the launch, the host was checked: 17 GB of memory free, no experiment running. The GPU held 1.9 GB, all of
  it the desktop's (Xorg, Chrome, VS Code).
- Launch: from the main checkout, clean, at 00:09. The command was C10's last resume with `--rounds 14`, repository-
  relative paths, and 3 workers. Three rather than four, because the desktop's 1.9 GB left four workers at about
  7.0 of the GPU's 8.2 GB. The C10 directory was made writable for the run.
- The resume was accepted: `rounds {before: 10, after: 14}`, the main checkout's paths in its entry, the checks equal
  (0 m up to 2.2e-06 m). O15 measured one worker at 1.11 GB of GPU and 1.43 GB of host memory, and the pass at
  1.89 GB.
- The rounds ended 00:50, 01:28, 02:05 and 02:41 (about 37 min each, 3 workers); exit 0.
- **The main checkout moved under the run.** Other sessions fast-forwarded it eight times (00:12–02:21,
  `bd7929a5` → `cddf0c7f`), so the rounds' `git` records three heads. The only Python among the changes is
  `post_training_export.py` and `post/training_files.py`, which the campaign does not import, and all of its
  modules were imported at 00:09. The rounds are not affected.
- **Sealed:**
  - `"start": null` added to `campaign.json`'s settings (the user's choice of 2026-10-07, for the start of a campaign;
    nothing else changed, checked);
  - the run's log and launch script copied into `logs/` (`campaign_c14.log.copy`, `launch_c14.sh.copy`);
  - the 218 earlier files checked against `SHA256SUMS`, all matching (only `campaign.json` differed, from the raise);
  - `SHA256SUMS` rewritten with 229 files and checked; the directory read-only again.

**The selection readout of the 14 rounds** (the same 1,000 select windows and numbers every round; no criterion, D7):

| Round | Workers | Landed | Lost separation | Mean reward | KMSY | KRDU | KSJC | KSMF | KSTL | Informative groups | Updates | Pull (KL) | Data term |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 80.2 % | 14.9 % | 0.793 | 0.820 | 0.750 | 0.835 | 0.785 | 0.775 | 1,477 | 391 | 0.0117 | 1.0860 |
| 1 | 4 | 82.6 % | 13.9 % | 0.819 | 0.875 | 0.800 | 0.845 | 0.780 | 0.794 | 1,253 | 331 | 0.0184 | 1.0807 |
| 2 | 4 | 81.0 % | 15.7 % | 0.802 | 0.835 | 0.770 | 0.830 | 0.790 | 0.785 | 1,237 | 332 | 0.0216 | 1.0773 |
| 3 | 4 | 83.2 % | 13.7 % | 0.824 | 0.875 | 0.800 | 0.855 | 0.775 | 0.815 | 1,144 | 307 | 0.0235 | 1.0728 |
| 4 | 5 | 83.7 % | 13.2 % | 0.826 | 0.865 | 0.805 | 0.865 | 0.785 | 0.810 | 1,205 | 324 | 0.0262 | 1.0713 |
| 5 | 4 | 83.8 % | 13.2 % | 0.830 | 0.894 | 0.800 | 0.850 | 0.785 | 0.820 | 1,231 | 331 | 0.0287 | 1.0783 |
| 6 | 4 | 85.1 % | 11.3 % | 0.842 | 0.870 | 0.830 | 0.850 | 0.795 | 0.865 | 1,106 | 303 | 0.0321 | 1.0763 |
| 7 | 4 | 84.8 % | 11.8 % | 0.840 | 0.895 | 0.830 | 0.840 | 0.780 | 0.855 | 1,201 | 326 | 0.0330 | 1.0744 |
| 8 | 4 | 85.7 % | 11.7 % | 0.849 | 0.890 | 0.805 | 0.855 | 0.815 | 0.880 | 870 | 244 | 0.0348 | 1.0687 |
| 9 | 4 | 85.0 % | 12.0 % | 0.840 | 0.885 | 0.810 | 0.865 | 0.820 | 0.820 | 1,161 | 315 | 0.0344 | 1.0759 |
| 10 | 3 | 85.0 % | 11.7 % | 0.842 | 0.880 | 0.820 | 0.850 | 0.785 | 0.875 | 1,058 | 290 | 0.0371 | 1.0720 |
| 11 | 3 | 83.7 % | 13.4 % | 0.831 | 0.870 | 0.810 | 0.845 | 0.775 | 0.855 | 1,036 | 284 | 0.0369 | 1.0644 |
| 12 | 3 | 84.1 % | 12.2 % | 0.833 | 0.880 | 0.815 | 0.850 | 0.794 | 0.825 | 986 | 270 | 0.0395 | 1.0679 |
| 13 | 3 | 84.8 % | 12.0 % | 0.839 | 0.900 | 0.830 | 0.835 | 0.800 | 0.830 | 1,051 | 287 | 0.0387 | 1.0715 |

**What the table shows:**
- Rounds 10–13 did not go past rounds 6–9. A landing share over 1,000 windows carries about ±1.2 points of noise, and
  every round from 6 on lies within it of the best (round 8, 85.7 %, 0.849).
- The training windows show the same picture. Lost separation stays at 17–19 %, about half of the branch groups have
  K equal rewards, and the clipped share is 0.4 % from round 1 on. The pull to the base keeps rising, 0.034 → 0.039.
- The user's criterion (D7) chooses the round. The ceiling readout (requests P48) comes first, on the user's word of
  2026-10-07.

## 28 C15, C16 and the intent at launch (2026-10-07)

The order is `notes/stage_c.md` of 2026-10-07 (`8012daff`), and the design is post-training D161, D162 and D163, §8 C15
and C16.

| Step | Commit | What |
|---|---|---|
| C15, C16, the intent | `3c10e08f` | See the list below |
| Merge | `67f5d25c` | `dev-two-tier` (docs only) into the branch; no Python changed against `3c10e08f`. `dev-two-tier` can fast-forward to the branch |

**C15 (D161), the ceiling readout.**
- `experiments/post_ceiling.py`: each model reads the select windows N times. Draw 0 is checked against the round's
  `round.json` before the other draws.
- `post_train`: `readout_numbers(seed, place, draw=0)`, with draw 0 unchanged bit for bit; `read_batch`,
  `Speakers.read` and `_read` take the draw; `counted_ends` is split out of `selection_readout`.
- `runners.md` R64, and one index line in the package's `CLAUDE.md`.

**C16 (D162), a campaign's start.**
- `Settings.start` and `campaign_start`. The start's bytes, identity and seed are checked (D162: refused when its seed
  is the source campaign's).
- The start is checked before the workers fork and before `campaign.json` is written.
- `--start-campaign` / `--start-round`; the path is recorded repo-relative through `this_checkout`.
- `round_model(None)` gives the campaign's start; the optimizer starts afresh; the pull term still pulls toward the
  base.
- The checkpoint identity is unchanged.
- The stage C frontend fixtures were written again by the export: their `settings` gain `"start":null`, which the
  frontend never reads.

**The intent at launch.** `post_train` no longer checks the intent when a campaign starts (the user; D163). The
publisher still checks it (L27).

**Tests.**
- The affected files: 83 passed. `test_post_ceiling`: 5 passed. The 11 vitest files that read the stage C fixtures: 75
  passed.
- New tests:
  - a start from round 0 speaks its round 0 with the start's weights, and its optimizer counts only its own updates;
  - the seed rule, the bytes and the identity refused by name;
  - the CLI's refusals, each checked by its message;
  - draw d reaches every window's numbers in one process and through the workers.
- Each new test fails under the mutation it guards against.

**Review** (opus, independent, two rounds):
- Four S2 findings, all fixed:
  - the D162 seed rule, decided after the review started;
  - the start test could not see the start's weights;
  - no test that the draw reaches the numbers;
  - the memory of 4 reading workers (the run uses 3).
- The S3 findings are fixed: the order of the models (a round first, so that draw 0 is checked first); a refused start
  wrote `campaign.json`; the start's path given from a worktree; a bad model name; the CLI messages; the start checked
  after the workers forked.
- The reviewer checked, on the CPU, that draw 0 of C10's rounds 6 and 8 gives their `round.json` readout exactly.

**The run.**
- First launch: 03:20 from the development worktree, on the user's word "先跑上限检测", before their merge. Draw 0 of
  round 6 equalled its `round.json`.
- Stopped after 4 draws (10 min), when the newest note asked for a detached run worktree (D163). The partial output
  (`config.json` only) was moved aside as `ceiling_20261007.aborted-20261007T013056Z`.
- Relaunch: from `.claude/worktrees/run-post-ceiling`, detached at `67f5d25c`, its data trees linked to the live ones.
  It waited for another session's tests to end (rule 13). Output: `outputs/POOLED/post/ceiling_20261007`.

**The run, continued (2026-10-07).**
- On the GPU with 3 workers a draw of 1,000 windows took about 2 min (96 draws: about 3.2 h). The user asked for the
  CPU with 8 workers. The GPU run was stopped after 15 draws, and its partial output (`config.json`) moved aside as
  `.aborted-20261007T020311Z`.
- CPU, 8 workers (04:03): about 47 s a draw. Each CPU worker holds about 0.55 GB of its own memory, against 1.5 GB for
  a GPU worker.
- The user asked for more workers. Restarted at 04:14 with 16 (a draw is 16 batches of 64 windows). The 8-worker
  partial output was moved aside too.
- With 16 workers, a draw still took about 50 s. A draw waits for its slowest batch before the next draw starts, and
  the host's load was about 12. A runner that sends the next draws' batches before the current draw ends would use the
  workers fully; it is not built.
- Draw 0 equalled its round's `round.json` readout on the GPU and on the CPU, for round 6 and for round 8.
- The run ended 05:27, exit 0. Sealed: `logs/` (the run's log and launch script), `SHA256SUMS` (7 files, checked), the
  directory read-only. The run worktree was removed, its data links unlinked first.

**The ceiling readout** (`outputs/POOLED/post/ceiling_20261007`; the 1,000 select windows of the selection readout, 32
draws each; the share of windows landed in at least one of the first n draws; no conclusion is drawn here, D7):

| Model | n = 1 | 2 | 4 | 8 | 16 | 32 | Never landed | Lost to separation in every draw | Mean best reward, n = 1 / 32 |
|---|---|---|---|---|---|---|---|---|---|
| Start (the base with zero-output traffic modules) | 79.5 % | 88.1 % | 94.3 % | 97.5 % | 98.7 % | 99.3 % | 7 | 5 | 0.787 / 0.987 |
| Round 6 | 85.1 % | 92.3 % | 95.7 % | 97.9 % | 98.6 % | 99.3 % | 7 | 5 | 0.842 / 0.987 |
| Round 8 | 85.7 % | 92.4 % | 96.5 % | 98.2 % | 99.0 % | 99.4 % | 6 | 5 | 0.849 / 0.988 |

By airport (200 windows each; n = 1 / 4 / 32, and the windows never landed):

| Airport | Start | Round 6 | Round 8 |
|---|---|---|---|
| KMSY | 83.5 / 98.0 / 100 %, 0 | 88.0 / 98.5 / 100 %, 0 | 90.0 / 99.5 / 100 %, 0 |
| KRDU | 72.0 / 93.5 / 99.0 %, 2 | 84.0 / 93.5 / 99.0 %, 2 | 81.0 / 97.0 / 99.0 %, 2 |
| KSJC | 80.0 / 92.0 / 99.0 %, 2 | 86.0 / 96.5 / 99.0 %, 2 | 86.5 / 96.5 / 99.5 %, 1 |
| KSMF | 82.0 / 96.0 / 99.5 %, 1 | 81.0 / 95.0 / 99.0 %, 2 | 83.0 / 94.0 / 99.5 %, 1 |
| KSTL | 80.0 / 92.0 / 99.0 %, 2 | 86.5 / 95.0 / 99.5 %, 1 | 88.0 / 95.5 / 99.0 %, 2 |

**The windows never landed** (10 windows across the three models, all real windows). Four were never landed by any of
the three models:
- 263 (KRDU, EDV5114): `dynamics_failure` in all 96 draws;
- 350 (KRDU, N592DR), 549 (KSJC, SWA3106) and 700 (KSMF, SWA2234): lost separation in all 96 draws.

Pairs: rounds 6 and 8 share 5, round 6 and the start 5, round 8 and the start 4. The other six windows lost
separation in most draws and landed in a few:
- 486 (KSJC) and 685 (KSMF) were never landed by the start or round 6 respectively;
- 871 and 876 (KSTL) were never landed by the start, but landed 3–4 and 10–11 times in 32 by rounds 6 and 8;
- 900 and 906 (KSTL) were never landed by round 8, or by rounds 6 and 8.

## 29 C17: training on the landed sentences (2026-10-07)

The order is `notes/stage_c.md` of 2026-10-07 (`77517176`), and the design is post-training D165 and §8 C17.

**The user's decisions after the note:**
- the form: a required `Settings.method` in the same runner and campaign schema, rather than D165's own runner and
  schema;
- a required `Settings.select_seed`;
- C10's record gains both fields in the same step as the merge.

| Step | Commit | What |
|---|---|---|
| Merge | `2816f2a6` and before | `dev-two-tier` merged in, including stage D's code (the `Stage` skeleton of §9 items 11 and 12). The work was carried onto it by a three-way patch, its conflicts resolved toward stage D's structure |
| C17 | `6afa3c0b` | See the list below |

**C17, the code.**
- `Settings.method`: `branch` or `landed`; a resume compares it.
- `STAGE_C_LANDED`:
  - `speak_landed_batch`: N draws a window; draw 0 is the window's first sentence (`landed_numbers`).
  - `best_landed`: the landed sentence of the highest reward above 0, a tie to the lowest draw, kept in
    `kept_<k>.pt`.
  - `landed_train_pass` → `post.loss.landed_step`: the kept words' negative log-likelihood, the pull and the data term,
    in pieces, normalised per counted row (D115).
  - `landed_pass_memory`: one update measured for O15.
- `Stage` gains two parts, `train` and `pass_memory`, with branch training's defaults, so stage D's code is unchanged.
  `speak_round` sums whatever its batches report, so the branch record is unchanged.
- `Settings.select_seed`: the selection readout's windows and numbers, read by `selection_windows`, `read_batch` and
  `model_speed`.
- The options `--method` and `--select-seed` are required.
- `intents.json`: `post_landed_20261007`; `runners.md` R65.
- The stage C frontend fixtures were written again by the export: their settings gain `method` and `select_seed`.

**Two readings of mine, told to the user:**
- 16 kept sentences an update, about 250 updates a round, against C10's 244–391;
- a landing that the reward scores 0 (D105) is not kept.

**Tests.**
- The affected files, stage D's included: 143 passed; after the last fixes, 69. `test_post_landed`: 10. The vitest files
  on the fixtures: 75.
- The new tests:
  - the keep rule;
  - the draws' numbers;
  - the loss against the whole expression (value and gradient);
  - a real kept sentence spoken here and through two workers (byte-equal kept files), then passed;
  - a landed campaign resumed, equal to the campaign run through, started from a round of another campaign;
  - the runner handing the method's stage to the workers and the campaign;
  - the readout's seed: seed 2024 reads C10's windows and numbers.
- The guarded tests fail under their mutations.

**Review** (opus, independent, two rounds):
- S1, fixed: the readout took its seed from the campaign's seed, so seed 2024 would have read about 195 of C10's 1,000
  windows.
- S2, fixed:
  - the update size;
  - reward-0 landings;
  - two missing tests;
  - a select-seed test that could not fail.
- S3: fixed or noted.

**The run waits.** Stage D's `multi_profile` (MC5 at the formal size) runs from `dev-multi-control` with 4 workers, and
reads C10 as its start.
- It must not be slowed (rule 13).
- C10's record edit would make its older code fail by name when it reads C10's settings.
- When it ends: C10's record edit, `dev-two-tier` fast-forwarded, the run worktree, the launch.

**C17's run (2026-10-07).**
- Launched 08:18 from `run-post-landed` (`61e718ba`), CPU, 16 workers; O15 measured one worker at 0.70 GB.
- Round 0 ended 09:07: speaking 22 min, the pass on one CPU thread about 23 min. Landed 84.2 %, lost separation 12.9 %,
  mean reward 0.835. 3,646 windows kept; 78.4 % of the draws landed.
- Round 1 ended 09:48: 84.5 %, 12.2 %, 0.838. 3,701 windows kept; 80.8 % of the draws landed.
- Round 2's speaking ended 10:07. The process was then killed in round 2's pass, with no error and no out-of-memory
  kill. The shells ran inside the GNOME Terminal tab's systemd scope (`vte-spawn-….scope`), and when the previous
  Claude Code session's terminal closed, systemd stopped the scope and every process in it; `nohup setsid` changes the
  session, not the cgroup.
- Resumed at 11:52 with the same command, now in a systemd user unit of its own (`systemd-run --user`, unit
  `post-landed-115211`, its cgroup checked). The half round is moved aside by `open_campaign`. The first run's log is
  kept as `campaign_1.log`.
- Round 2 rerun ended 12:39 (84.8 %); round 3 ended 13:21 (85.1 %). The user's rule, 2026-10-07: if this round is
  still flat, stop. My reading of "flat": below the start's 85.7 % plus the readout's noise, 86.9 %. A watcher in its
  own systemd unit read round 3's readout after its checkpoint and stopped the campaign at 13:21:47. Round 4, just
  begun, was moved aside as `round_4.aborted-…`.
- Sealed: `logs/` (both runs' logs, the launch and the watcher), `SHA256SUMS` 78 files checked, read-only. The run
  worktree was removed, its data links unlinked first.
- Rounds 0–3 against the start (C10 round 8: 85.7 %, 11.7 %, 0.849): 84.2 / 84.5 / 84.8 / 85.1 % landed, 12.9 / 12.2 /
  12.0 / 11.3 % lost separation. The table, the intent and my reading of why it did not help are in the experiment log,
  `readouts/2026-10-07_stage_c_experiments.zh.md`, which also holds the next experiment: branch training with K = 16
  (the user's priority, 2026-10-07; an experiment within the design, not a design change).

## 30 C19 and C18: gradient clipping, the measure reused (2026-10-07)

Built on `dev-two-tier-v4-post` after fast-forwarding it to `dev-two-tier` (`b849340f`), while the campaign with K = 16
runs from its own run worktree (only the changed modules' tests, 2–3 processes, rule 13).

**C19 (D168), `f3ce543f`.**
- `Settings.clip_norm`, default None: not clipped, the code's behaviour before it. A record without the field reads as
  None (the user's standing permission); no record is edited. `SETTINGS_ADDED` lists the settings with a default, and a
  test pins it to the dataclass.
- `post.loss.one_pass(..., clip_norm=)`: after an update's backward, the gradient's norm over the optimizer's parameters
  (`torch.nn.utils.get_total_norm`, read-only); with a clip, `clip_grads_with_norm_` before the step (what
  `clip_grad_norm_` does). Each update keeps its norm before the clip and whether it was clipped.
- `round.json`'s `pass` gains `grad_norm_mean`, `grad_norm_max` and `grad_clipped_share`. They are recorded with no
  clip too (share 0), so the norms of a campaign at the old setting show where a clip would act.
- Stage C's pass reads the setting through its stage (`STAGE_C`, `STAGE_C_LANDED`). The skeleton's default pass, which
  stage D's campaign uses, clips nothing: its `MultiSettings` has no such field (D168: stage D takes the defaults).
- `open_campaign` compares the recorded settings as the stage's settings class reads them (`settings_type`). Without
  this, an old record (no `clip_norm`) would differ from a resume's settings (`clip_norm: None`). Stage D's runner and
  its tests pass `MultiSettings` (a one-word change in `multi_train.py` and its tests).
- Tests: with None the pass is the loop before D168 bit for bit; below every norm each stepped gradient has the clip's
  norm and every update is counted clipped; above every norm the pass is the unclipped one bit for bit; an old record
  opens with the default and resumes with it, another value refused by name, and the reverse; stage C hands its setting
  to `one_pass`, stage D's pass hands None.
- Review (opus, independent): no S1. S2: `test_multi_validation` opened a stage D campaign without `settings_type`
  (safe only because it never resumed) — fixed. S3: the stage D test now calls `stage_d()`. S3 noted: a record that
  lacks an older required field now stops with a TypeError rather than "other settings" (no live record does: the four
  records under `POOLED/post` were checked by the reviewer; only `post_train_20261006.aborted-…` lacks fields).

**C18 (D167), `364c2f4c`.**
- The first launch with speaking workers measures (O15, `require_workers_fit`) and adds the measure to
  `campaign.json`'s `measures`: the devices (the pass's and the workers'), the round, the time, the workers it admitted,
  the worker's and the pass's measures, the memory free then.
- A later launch takes the newest measure on its devices. When that measure admitted at least as many workers, it is
  read before this process uses the GPU, and only the memory free now is checked (`workers_fit(held_now=False)`, the
  form of stage D's branch: the workers' and the pass's whole peaks, the workers on the GPU beside what this process
  holds after a pass). Workers that do not fit are refused by name, naming the measure. Otherwise it measures again.
- My reading of D167's "more workers than the measure allows": more than the measure was checked for and admitted. A
  record without `measures` (every campaign so far, the one with K = 16 included) measures at its next launch.
- Each launch's entry (the record itself for the first launch, `resumed[-1]` for a resume) names the measure it read:
  `fit` = its place in `measures`, whether it measured, the workers, the memory read; None without workers or with no
  round left.
- `main` now moves the context to the GPU after the recorded fit is read (inside `try`, so the workers are closed on a
  refusal).
- APPROXIMATION, stated in `workers_fit`: with nothing held yet, the host's free memory is read before this process
  starts CUDA, so its own host memory for CUDA (a few hundred MB, not measured) is not taken out.
- Tests: the arithmetic with nothing held (the speaking line on the GPU deciding too); the newest measure on the
  devices, more workers, other devices, a record from before D167, a refusal naming the measure; through the runner:
  measured once, read on a resume, measured again for more workers, none without workers, a record without measures
  measures, the free memory read before the move.
- Review (opus, independent): S2, fixed: on a resume the workers on the GPU were checked without what this process
  holds after a pass. S3: the host approximation now stated; the order in `main` pinned by a test.
- Merge note for stage D: `dev-multi-control` changes `workers_fit` the same way (`held_now`), but its pass line has no
  guard for workers on the CPU (`--speak-device`, `ac455d2e`). The merge keeps the guard; its `profiled_fit` already
  takes `passed["now"]` out of the GPU share.

`dev-two-tier-v4-post` (`364c2f4c`) fast-forwards `dev-two-tier`; the user merges. C20 (D169) follows the campaign
with K = 16.

**The runs after C18 (2026-10-07).**
- The campaign with K = 16 (`post_branch16_20261007`) stopped after round 3 on the user's word: 84.4 / 84.9 / 85.8 /
  85.1 % landed, no large change against the start's 85.7 %. Round 4, just begun, moved aside (`round_4.aborted-…`);
  sealed (`logs/`, `SHA256SUMS` 12 files, read-only); its run worktree removed.
- The user then chose the next experiment (candidate 3a: the prior's learning rate 3e-5, the traffic modules' 3e-4,
  `--clip-norm 1.0`, K = 8, from C10's round 8, seed 2026) and allowed the fast-forward of `dev-two-tier` to
  `dev-two-tier-v4-post` (`704879d7`). Launched 16:00 local from `run-post-lr3` in the systemd unit `post-lr3-160054`.
- Intents and results: `readouts/2026-10-07_stage_c_experiments.zh.md` (experiments 2 and 3); `intents.json`.

**C20 (D169), `2e2e860f`, merged into `dev-two-tier` on the user's word ("审完测完直接合并").**
- `Settings.epochs`, default 1 (a record without it reads as 1; none edited); `SETTINGS_ADDED` = clip_norm, epochs.
- `post.loss.passes`: E passes, one `PassStart` (the model at the round's start) for all, each pass's updates drawn
  lazily from its own iterable (a pass's groups are loaded only when it runs); `one_pass` is a pass of it.
- `pass_orders`: the first pass reads the round's numbers (`[seed, round, 1]`, so E = 1 is the pass before, bit for
  bit); pass e ≥ 1 `default_rng([seed, round, 1, 1 << 31, e])`. The reviewer found that `Generator.spawn` children
  would have equalled a continuation's key (`[seed, round, 1, 0, k]`, branch 0, never drawn); the explicit key avoids it.
- `round.json`'s pass: the means over every pass and `passes` (each pass's); the landed method renames each alike.
  Nothing reads `round.json["pass"]` (the reviewer's search: backend, exports, readouts).
- Stage D keeps one pass (the skeleton's default `train_pass`, epochs 1).
- `--epochs` (default from `Settings`).
- Tests: two passes share one frozen `PassStart` and draw in order; the pass orders (the first the round's, the later
  ones their own, no continuation's key); stage C hands epochs and clip, stage D one pass, no clip; a record without
  `epochs` opens and resumes as 1, 2 refused; the pass record over two passes, branch and landed.
- Review (opus, independent): no S1, no S2; four S3 fixed (above, and a test that compared `one_pass` with itself).
- Found while testing: `test_post_generalised`'s campaign digest had failed since `ac455d2e` (`--speak-device`), whose
  test run left that file out: `round.json` gained `speak_device`. The digest now leaves out the information keys added
  since (`speak_device`, the gradient norms, `passes`), and with them out it is the digest recorded before the
  skeleton: models and optimizers unchanged bit for bit. A lesson for me: the generalisation test runs with every
  change to `round.json`.
- The pass's rng `[seed, round, 1]` is the same stream as `first_numbers(seed, round, window=1)` (the reviewer's note,
  older than C20). The two read different things (a shuffle and a data draw, a sentence's words), so no result is
  affected; a key apart would change every campaign's numbers, so I leave it as it is and note it here.
