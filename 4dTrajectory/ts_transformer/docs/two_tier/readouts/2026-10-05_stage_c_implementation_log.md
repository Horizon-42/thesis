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
