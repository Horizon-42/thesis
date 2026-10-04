# Two-tier model: design of the vocabulary, the labeller and the executor

**Summary.** The two-tier model has two parts. The upper part is the prior: a language model of the controller. Every
2 s it says one row of words for each aircraft. The lower part is the executor: an autopilot that flies only these
words with point-mass dynamics. This document gives the next design of four parts: the words (the vocabulary), the
program that reads words from observed tracks (the labeller), the executor and the prior. It gives only an outline for
the post-training and the multi-aircraft work.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that §2 defines. The words were not checked one by one against
the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete: it gives every rule of the design, and it needs no other design
document. The code is on the branch `dev-two-tier-v4` (§0.3, §14). The evidence (§11) cites the readouts and the data it
comes from. Paths are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or the
repository root.

---

## 0 Status

### 0.1 Decisions

| # | Item | State | Source |
|---|---|---|---|
| D1 | The runway column says the expected runway and "go-around". There is no approach column and no word "cleared". The model says the runway at the first predicted step | Decided | User, 2026-10-03 |
| D2 | The model flies the aircraft to the runway with heading words. No executor law turns the aircraft onto the final | Decided | User, 2026-10-03 |
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The LPV minimums are a judgement only. The model must learn to land or to go around | Decided | User, 2026-10-03 |
| D4 | The speed value "unspecified" starts at the capture row of the labeller | Decided | User, 2026-10-03 |
| D5 | Prior inputs are in the frame of the runway in force. The prior has no airport embedding, no absolute position and no absolute direction | Decided | User, 2026-10-03 |
| D6 | The grids of the words: heading 5°, four descent classes. A measurement or a rough labelling decides if they are sufficient (D35) | Decided | User, 2026-10-03 |
| D7 | This document sets no test criteria and no measurement criteria. The user sets them after the design is settled | Decided | User, 2026-10-03 |
| D8 | The heading grid is aligned to the course of the runway in force (§3.3) | Decided | User, 2026-10-03 |
| D9 | The executor has none of the laws of §5.7, the glidepath floor included | Decided | User, 2026-10-03 |
| D10 | Go-around is an event: "abandon this approach". It does not change the runway in force; a runway word ends it. Its effects on the executor, the judge and the masks: §3.2 | Decided | User, 2026-10-03 |
| D11 | The labeller stage includes an ablation of the row interval Δ (§4.8) | Decided. Values: D25. Readings: D34 | User, 2026-10-03 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§6.1) | Decided | User, 2026-10-03 |
| D14 | While G is true: "no level-off" is not permitted (rule 5), and the procedure mask "no climb below the entry height" does not apply (§3.7) | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§6.1) | Decided | User, 2026-10-03 |
| D17 | Every column has the input "time since this column said its word in force", in seconds; the runway column too (§6.1) | Decided | User, 2026-10-03 |
| D18 | The labeller reads the real go-arounds inside a sentence and says "go-around" at the go-around point (§4.6) | Decided | User, 2026-10-03 |
| D19 | After a labelled go-around, the runway word that ends G is at the first level-off after the go-around climb, and not later than the row of the next "no level-off" (§4.6) | Decided | User, 2026-10-03 |
| D20 | This design is a new version, on its own branch and worktree. It reads no artefact, executor spec or prior that an earlier version made (`v1`–`v6`). The running experiments keep their own checkouts | Decided | User, 2026-10-03 |
| D21 | Identities bind format and data rules only. A code identity is a behaviour check on fixed inputs, never a hash of source bytes; data are identified by their flights, never by the bytes of a manifest (§9.2) | Decided | User, 2026-10-03 |
| D22 | Altitude words use the grid "optimal 40 levels" of the altitude-grid proposal: 60 m steps from 0 to 1,260 m, 120 m steps to 2,700 m, 450 m steps to 5,400 m (§3.4) | Decided | User, 2026-10-03 |
| D23 | How the prior gets the frame of R (D5) and the glidepath height (D13). The own state of the aircraft has no frame. Every position and direction is in the candidate vectors, each in the frame of its own candidate. R is an input only as its candidate vector. No input of a row up to the first predicted step is computed from R. Each candidate vector has the height above its own glidepath; the value of R is the input of D13 (§6.1) | Decided | User, 2026-10-03 |
| D24 | A candidate vector has no constant of its runway: no length, no elevation, no layout relative to R. Such a value comes back only as a variant that D39 selects (§6.1) | Decided | User, 2026-10-03 |
| D25 | The ablation reads Δ = 2, 4, 8 s: each divides the 16 s observation of the prior. At each Δ, the motion inputs of a row come from the 2 s before the row (§4.8, §6.1) | Decided | User, 2026-10-03 |
| D26 | The labeller reads a flight with a go-around approach by approach. An approach ends at the landing or at a go-around row. The last descent that reaches a go-around row says "no level-off". Each approach has its own capture row and its own "unspecified". The first row says the runway of the first approach. The go-around row is the row of the climb word (§4.2, §4.4–§4.6) | Decided | User, 2026-10-03 |
| D27 | "Go-around" changes no target of another column. A row that says "go-around" while "no level-off" is in force also says a level above the aircraft (rule 6). The executor keeps the heading word in force; it does not fly the course of R. While G is true, "unspecified" holds the airspeed (§3.2, §3.7, §5.4–§5.6) | Decided | User, 2026-10-03 |
| D28 | The vocabulary has one climb word. Its angle is a value of the executor. While G is true, the executor climbs at the steady climb angle that the thrust limit permits, not more than 3° and not less than 1.885° (200 ft per NM). While G is false, it climbs at the nominal angle of the climb class; that value comes from the spec measurement with D15 (§3.5, §5.5) | Decided (the range 1.885°–3° and the nominal from D15: the user; the thrust rule inside the range: Claude's proposal) | User, 2026-10-03 |
| D29 | Post-training starts from the base model in the multi-aircraft setting "one aircraft commanded". There is no single-aircraft post-training stage (§7) | Decided | User, 2026-10-03 |
| D30 | The post-training reward comes only from the outcome: 1 for `landed` without a go-around; 0.9ⁿ for `landed` after n go-arounds; 0 for every other outcome and for every loss of separation. No payment for a go-around without a landing. No mask on where "go-around" can be said (D10) (§7) | Decided | User, 2026-10-03 |
| D31 | Multi-aircraft inputs and judgements (§8): the landing context of every aircraft of a scene counts the landings of the closed loop; D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: O6). The prior's training stops on the select days (§6.3) | Decided | User, 2026-10-03 |
| D32 | Closed-loop reading. The labeller flies its sentence with the executor. When the flown path leaves the observed path by more than a tolerance, it says a correction word, and the observed word again when the flown path is back. The prior trains on the flown states of these sentences (§4.9, §6.1). Tolerances: lateral 30 m, vertical 15 m | Decided | User, 2026-10-03 |
| D33 | While G is true, no crossing of a threshold is an event: of R or of another candidate (§3.2, §5.8) | Decided | User, 2026-10-03 |
| D34 | The ablation of the row interval reads, at each Δ: the correction words of the closed-loop reading for each flight and each column; the errors left where §4.9 makes no correction; the replay outcomes. The user compares the Δ values on these readings (§4.8) | Decided | User, 2026-10-03 |
| D35 | No finer grids. A finer grid does not remove the drift of open-loop words (§11.9); the closed-loop reading does (D32). The grids are those of D6 | Decided | User, 2026-10-03 |
| D36 | The teacher-forced data term of the post-training uses single-aircraft samples of the closed-loop sentences, not scene samples. The flown states keep the observed path, not the observed time (§4.9), so two aircraft of one scene do not keep their observed spacing (§7, §8) | Decided | User, 2026-10-04 |
| D37 | Branch training. Each training aircraft is spoken one time. When its reward is less than 1, it is spoken again from saved states at its first predicted step and every 120 s after it, before the event that ended it. Each branch group compares only the words after its branch point (§7) | Decided | User, 2026-10-04 |
| D38 | The DA check uses one definition with the evaluation module. Vertical: within ±22 m of the published glidepath of R, the bound of `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`, ICAO Doc 9613). Lateral: inside the FAS cone at the distance of the DA point. Neither is a parameter (§5.8) | Decided | User, 2026-10-04 |
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). Two variants: `full` (§6.1) and `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation). `constants` is chosen only if it is better by more than twice the seed scale (§6.3) | Decided | User, 2026-10-04 |
| D40 | Hyperparameters: configuration A (§6.3) is the start. The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§6.3) | Decided | User, 2026-10-04 |
| D41 | The design runs at an airport that is not in the training data: the number of candidates is not fixed; every input has a fixed physical scale, never a statistic of the training data; each fold of D39 flies the full closed loop at its held-out airport. The altitude grid limits this to airports whose approach levels lie in its 60 m segment (§6.4) | Decided | User, 2026-10-04 |
| D42 | The closed-loop reading says each observed word at the place where the observed aircraft heard it, not at the time: at the Δ row nearest to the place where the matched point of the flown aircraft reaches it (D45). The correction words do not change. The flight ends when the executor is done or at the time limit of the replay (§4.9) | Decided | User, 2026-10-04 |
| D43 | Speed words in steps. In a change of speed, the labeller says each grid value on the way, where the observed speed comes nearer to it than to the value before. The executor changes the speed of a speed word at the largest acceleration of the speed envelope (a_max, a value of the spec). The rate of a change thus comes from the words, as the turn rate comes from the heading words. "Unspecified" has its own rate (§3.6, §4.5, §5.6) | Decided | User, 2026-10-04 |
| D44 | No correction past the end of the observed path (the observed slice ends before the threshold; the flown aircraft flies on to it). There the closed-loop reading measures the lateral error against the line of the last segment, for the readouts only, and no vertical error (there is no observed height); it says no lateral or vertical correction: a correction in force ends, and the observed word is said again; the aircraft flies the last observed words to the threshold. The rows count as rows without correction (D34). The DA point always lies before the end of the observed path (§4.9) | Decided | User, 2026-10-04 |
| D45 | A word goes to the nearest row, not to the next row. The Δ grid puts each word of the 2 s reading on the nearest Δ row (§4.8). The closed-loop reading takes the observed words with their 2 s times and says a word at the first Δ row whose matched observed time is less than Δ/2 before the time of the word (§4.9). A word exactly between two rows goes to the later row in both. The mean lateness of a word is then zero in the closed loop, and on the Δ grid zero at 2 s and 1 s (half a 2 s row, from the ties) at 4 and 8 s | Decided | User, 2026-10-04 |
| D46 | A heading word is said in the frame where the executor hears it. Where the Δ grid or the closed-loop reading moves a heading word across a runway word, the word is the class nearest to its absolute track minus the course of the R in force when it is heard. A heading word is said when its absolute track differs from the track in force, also when its class is the class in force said under another course. A change of runway alone says no heading word. No sentence is refused for a heading word across a change of runway (§3.3, §4.8, §4.9) | Decided | User, 2026-10-04 |
| D47 | The turn law of the executor stays (§5.4). The measurement of A13: the words alone (the 5° grid and the lead) end a turn approximately 35 m inside the observed turn; the executor gives back approximately 18 m of it; the stopping-rate limit changes approximately 3 m (§11.14). The readout of a turn is its own part (the change of the displacement from the observed aircraft of the same time); the change of e_y is read beside it, because it holds the offsets that earlier turns left | Decided | User, 2026-10-04 |
| D48 | The 2 s reading follows D46 too. After a change of R, a heading word whose class is the class in force is said, because its absolute track differs from the track in force. No flight is refused for it (§4.3) | Decided | User, 2026-10-04 (Claude's check of stage A, §3 item 2) |
| D49 | The labeller conformance also covers the Δ grid: its reference sample stores the sentences at Δ = 4 and 8 s, and the check compares them again with the code on disk (§9.2) | Decided | User, 2026-10-04 (check §3 item 3) |
| D50 | The closed-loop reading does not keep every row inside the tolerances of D32. A correction is one class and needs time to bring the flown path back (§11.11: a 100 m offset takes approximately 20 s at one 5° class); in the stage A smoke 10–25 % of the correctable rows are more than 30 m off laterally. The check of §14.6 is a rule on the stored data: on every row where §4.9 permits a correction, \|e_y\| > Y is followed by a heading correction in force after that row, and \|e_h\| > H by an angle correction. The p90 of \|e_y\| at the last row is information. The share of the correctable rows outside the tolerances is a reading of the ablation (D34), not a check (§4.8, §14.6). Since D53 the reading keeps it on every row | Decided | User, 2026-10-04 (check §3 item 4) |
| D51 | The closed-loop artefact stores the flown states on the data's 2 s rows at every Δ, with the Δ rows marked. Observed and flown rows are then on one grid, and stage B builds the motion input of D25 with one piece of code for both. The artefact stores raw states, not a derived input, so a change of the motion window needs no rebuild. The replay check of a closed-loop sentence compares on the 2 s rows. At Δ = 2 s nothing changes (§4.8, §4.9) | Decided | User, 2026-10-04 (check §3 item 1) |
| D52 | ε of a level = half the larger gap to its neighbouring levels + 10 m: 40 m for 0–1,200 m, 70 m for 1,260–2,580 m, 235 m for 2,700–5,400 m. A level's band then covers the largest rounding error of its word; the top level of a segment collects the heights up to half the next segment's step above it. One band for both sides (a one-sided band would differ at two levels only). "No level-off" has ε = 40 m: it has no level, so no rounding to cover; its tube's width comes mainly from the edges of the angle class (§3.4, §10) | Decided | User, 2026-10-04 (check §3 item 5) |
| D53 | An overshoot. When a correction is in force and the error changes its sign while it is more than the tolerance (the flown aircraft crossed the observed path in one row), the labeller says the opposite correction in the same row: one class toward the path from the observed word. When the error changes its sign and is within the tolerance, it says the observed word again. Laterally and vertically alike. The rule of D50 then holds on every row (§4.9, §14.6). Why: a correction ends at a sign change so that it does not push the aircraft further on the other side; waiting one row to correct that side adds a lag of one row Δ that comes from the reading, not from Δ, and the ablation compares Δ (A15 smoke: every break of D50 at Δ = 4 and 8 s was such a row) | Decided | User, 2026-10-04 |
| D54 | The descent classes are a k-means on tan(angle) of the descent pieces, with each piece weighted by the square of its length. A piece flown at the nominal angle of its class ends length · \|tan a − tan c\| from its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest; D15 gives the same error for each candidate. The climb nominal is the length-weighted median of the climb pieces (§3.5) | Decided | User, 2026-10-04 |
| D55 | A value that a runner fits from data and the user chooses (D15) is measured on all train days, in a scratch directory, directly after the milestone that writes the runner; the user chooses before a later milestone reads the value. A smoke build uses the chosen spec, and its flights are a random sample for each airport and split (seed 1337), not the first flights of the sorted flight keys (a key starts with the callsign, so the first flights are mostly one airline). Why: the smoke of stage A fitted its own spec on approximately 400 train flights, 373 of them one airline, with one climb piece; every smoke reading of A9–A16 used it (§14.1 rules 7 and 12) | Decided | User, 2026-10-04 |
| D56 | The values of D15, chosen: the candidate rounded to 0.25° of the spec measurement on all train days (stage A code at `42f3ff62`, 44,703 train flights, 227,559 descent pieces, 1,339 climb pieces). Descent nominals 1.5 / 2.5 / 3.0 / 4.5°, edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; climb nominal (G false) 1.5°. End-of-piece height error p50 / p90: 15.0 / 51.0 m (fitted 1.51 / 2.45 / 3.09 / 4.45°: 14.7 / 47.8 m). Descent 3 is the published 3.0° glidepath of 24 of the 25 candidate runways (§3.5, §10) | Decided | User, 2026-10-04 |

### 0.2 Open items, in the order of discussion

| # | Item | Proposal | §  |
|---|---|---|---|
| O6 | A mask for the spacing on the final (this design has no clearance word to hold back); the rule "established on the final" of the separation judge and the masks (D31) | Discuss with §8 | 8 |
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |
| O9 | The word clock of an open-loop replay. The executor spec chooses `time`, `distance` or `track` (`autopilot/sentence.py` `CLOCKS`); the formal executor spec `v12_20261004` uses `track`: a word is said at the row whose observed point is nearest the flown aircraft (forward, at most 60 s ahead, at most one row a cycle). It serves only the open-loop replays and the reference flights of the executor conformance. The closed-loop reading says its words at its own matched point (D42), and a closed-loop sentence is said on its own rows, so no training sentence and no reading of D34 uses it | At the next change of the executor spec: `time` only (an open-loop sentence said on its own rows), the other two clocks removed. Until then the value is part of the executor spec | 5.8, 9.2 |

### 0.3 Implementation

| Part | State |
|---|---|
| Stage A: vocabulary, labeller, identities, executor, judge, replay, closed-loop reading (§14.2) | Built on `dev-two-tier-v4`, each milestone reviewed: A0 `e57c62d8` + `d2917c06`; A1–A3 `63639a54` + review fixes `feaf9558`, `d1e277f7`; A4–A6 `330ffbaf` + review fixes `c7603a4c`; A7 `14eb7946`; A8 `05f00be7` + review fixes in `c41ce7af`; A9 `c41ce7af` + review fixes `2e182e8d`, `983f2847`; D38 and D34 `6d1e8c4a` + review fixes `fdc3b832`; A10 `4ac34679` + review fixes `d1258c00`; A11 `efcbb5dc` + review fixes `0c0778c8`; A12 `b8077c34` + review fixes `b5b76c5a`; A13 `47865c8e` + review fixes `249804bd`, `b8994f58`; A14 `66daefa2` + review fix `ab295b18`. Smoke build of A12 and A13 (80 flights an airport and split, from the open-loop reading again, in the ignored `smoke_v4/data/a12/` of the worktree): the labeller's, the executor's and the closed loop's conformance checks pass; the closed loop read at Δ = 2, 4 and 8 s; the replays ran at Δ = 2 and 4 s open loop and 2, 4, 8 s closed loop; the turn readout of A13 ran on train and select; the full ts suite passes (1,523). Reports: `readouts/2026-10-04_stage_a_a8_a9_report.zh.md`, `readouts/2026-10-04_stage_a_a10_a11_report.zh.md`, `readouts/2026-10-04_stage_a_a12_a13_report.zh.md`. The closed loop of A14 built again in `smoke_v4/data/a14/`; the full ts suite at `ab295b18` passes (1,524). Claude's check of stage A (§14.6) at `ab295b18`: `readouts/2026-10-04_stage_a_check.zh.md` — items 2, 4, 5 pass; items 1 and 3 hold with points for the user (the motion input 2 s before a row is not stored at Δ = 4 and 8 s). The user's decisions on the check: D48–D52 and A15. A15 `4a2f4fc5` (three review rounds); smoke `smoke_v4/data/a15/`: the three conformance checks pass, the closed-loop sentences and the replays equal A14's on every Δ row, the replays pass on every 2 s row; the rule of D50 held at Δ = 2 s and broke at 4 and 8 s only on overshoot rows; the full ts suite passes (1,535). Report: `readouts/2026-10-04_stage_a_a15_report.zh.md`. A16 (D53, the overshoot corrected in its row) `3e70b4f3`, reviewed; smoke `smoke_v4/data/a16/`: the conformance checks pass, the rule of D50 holds on every row at Δ = 2, 4, 8 s, Δ = 2 s unchanged; the full ts suite passes (1,535). Report: `readouts/2026-10-04_stage_a_a16_report.zh.md` |
| Stage B: prior (§14.3) | Not started; outline |
| Stage C: post-training and multi-aircraft (§14.4) | Not started; outline |
| Stage D: frontend and backend (§14.5) | Not started; outline |

### 0.4 Plan

1. Stage A (§14.2), on the branch `dev-two-tier-v4`, by another agent: A0–A17 are done; A18 builds the formal artefact
   with the spec of D56 and makes the readings of D34 at Δ = 2, 4, 8 s.
2. Claude checks A15–A18 and the formal artefact (§14.6).
3. The user compares the readings of D34 and chooses Δ (D7, D11). The replay of the val days waits for the user.
4. Stages B, C, D (§14.3–§14.5): the prior from the start, the post-training and the multi-aircraft work, the frontend.
5. The user merges the branch.

---

## 1 Scope and principles

1. **The words are the only interface.** The prior gives no control values. The executor reads the words, the dynamics
   of the aircraft and the runway geometry in the vocabulary (§5.2). The executor reads no procedure data.
2. **The model decides the path.** No executor law turns onto the final, follows the centreline or follows the
   glidepath. The model must say the words that do these things (D2, D3). This is the user's rule of 2026-09-16: a
   track that the procedure computes is not a skill of the model.
3. **Procedures are masks and judgements only.** The procedure data (glidepath, decision altitude, final approach
   segment) blocks words during decoding (masks) and judges the flown track (judge, evaluation). It does not make a
   track.
4. **Airport facts are inputs, not words and not identities.** A word has the same meaning at all airports. Facts
   of the airport (runway geometry, landings before now) go into the inputs as measured values. No input is a code
   for the airport.
5. **One frame for the words of direction.** Heading words are relative to the course of the runway in force (D8).
   The same heading word means "downwind" or "final" at all airports.
6. **Each instruction has a target and an envelope.** The labeller, the judge and the display use the same envelope
   code (`instructions/envelope.py`). A containment rate is always given with the width of its envelope.
7. **Inputs give only what a controller knows before the step.** Labelled words are targets of the prior, never
   inputs, except the words said before the step.
8. **No compatibility.** A changed word, spec or payload gets a new name. The code refuses an artefact of another name.

---

## 2 Terms

| Term | Meaning |
|---|---|
| Row, step | One line of a sentence. A row is Δ seconds, on the UTC multiples of Δ: 2 s, or 4 or 8 s in the ablation of §4.8 (D25) |
| Sentence | The rows of one flight, from its first row to the row before the landing |
| Column | One kind of word in a row. A row has five columns (§3.1) |
| Word | The value of one column in one row. The value "unchanged" means that the column says nothing new |
| Word in force | The last word that a column said |
| Runway in force (R) | The candidate runway that the runway column said last |
| Go-around state (G) | True after the word "go-around", false after a runway word (§3.2) |
| Candidate runway | A landing threshold of the airport in the arrival manifest (`runway_targets`), with FAA CIFP geometry |
| Course | The true direction of a candidate runway, in compass degrees |
| Relative heading | The target track minus the course of R, in (−180°, +180°]. Positive is clockwise |
| Final | The extended centreline of R, before the threshold, in the direction of the course |
| Capture corridor | On the final: lateral offset ≤ 20 m + d·tan 0.45°, track within 2° of the course (d: distance to the threshold). Spec values (§10) |
| Approach | The rows of a sentence from its first row, or from the runway word that ends G, to the landing or to the next go-around row (D26) |
| Go-around row | The row where the runway column says "go-around". In a labelled sentence: the first row of the climb after the low pass (§4.6) |
| Capture row | The labeller's first row from which the observed track stays in the capture corridor to the end of its approach (D26). Each approach has at most one. Only the labeller uses it (D4). It uses later rows, so it is never an input |
| Go-around angle | The angle of a climb word while G is true: a value of the executor, 1.885°–3° (D28, §5.5) |
| Open-loop reading | The labeller's reading of the words from the observed track alone (§4.1–§4.8). Its words are the observed words |
| Closed-loop reading | The labeller's second pass: it flies the sentence with the executor and adds correction words (D32, §4.9) |
| Flown states | The states of the executor in the closed-loop reading, on the rows of the sentence. The prior trains on them (§6.1) |
| Correction word | A word that the closed-loop reading adds to bring the flown path back to the observed path (§4.9) |
| Matched point | The point of the observed path nearest to the flown position, searched forward from the matched point of the row before (§4.9). The closed-loop reading compares the flown state with the observed path there, and says the observed words there (D42) |
| First predicted step | The row 16 s after row 0 (`N_LOOK` = 8 rows at 2 s). The prior observes the rows before it and says nothing there |
| Executor cycle | 1 s. The executor hears the words at the start of each row |
| Labeller | The program that reads a sentence from an observed track (`instructions/labeller/`) |
| Judge | The part of the executor package that classifies what the executor flew (`autopilot/judge.py`) |
| Decision altitude (DA) | The published DA of the LPV minimums of R, as a height above the threshold (`Runway.decision_height_above_threshold_m`) |
| DA point | The first flown row where the aircraft, on the final of R and with G false, descends through the DA |
| Final approach segment (FAS) cone | The LPV lateral containment: half-width 106.7 m (350 ft) at the threshold, larger with distance (FAA Order 8260.58D Formula 3-1-1, `flight_scenarios/fas_geometry.py`) |

---

## 3 Vocabulary

### 3.1 Columns and order

| # | Column | Values (with "unchanged") | Classes |
|---|---|---|---|
| 0 | Runway | a candidate runway (pointer), "go-around" | candidates of the airport + 2 |
| 1 | Heading | relative heading, 5° grid (72 values) | 73 |
| 2 | Altitude | target level, geometric MSL, 40 levels on three segments (60 / 120 / 450 m steps), 0–5,400 m; "no level-off" | 42 |
| 3 | Angle | level; descent 1–4; climb | 7 |
| 4 | Speed | target ground speed, 5 m/s grid, 20–250 m/s (47 values); "unspecified" | 49 |

In one row the prior says the columns in the order of the table. A later column sees the values that the earlier columns said in the same row.

### 3.2 Runway column

**Meaning of a runway word.** "Expect runway k." The executor uses R for two things only: the conversion of the
heading words (§5.4) and the distance for the deceleration to the approach speed (§5.6). The judge uses R for the landing (§5.8). The prior gets R as the candidate vector of R (§6.1, D23). The multi-aircraft
loop uses R for the relations between aircraft (§8).

**Regulation.** Approach control gives the expected approach and runway at the first contact (FAA JO 7110.65BB
4-7-10 a: "Approach clearance or type approach to be expected ... Runway if different from that to which the
instrument approach is made"). Thus the runway is known before the approach clearance. The approach clearance is an
authorization ("CLEARED (Type of) APPROACH − ATC authorization for an aircraft to execute a specific instrument approach
procedure", Pilot/Controller Glossary). After it, the aircraft flies the procedure (AIM 5-5-4 a.3). The labeller cannot
see when a controller gave it: the data has no radio. Thus this design has no clearance word. The model says the turns
with heading words (D2).

**States.** The column keeps two states: R (the runway in force) and G (the go-around state).

| Word said | R after it | G after it | Permitted when |
|---|---|---|---|
| candidate k at the first predicted step | k | false | always; the first predicted step must say a candidate |
| candidate k later, G false | k | false | k ≠ R (the word in force is not said again) |
| candidate k, G true | k | false | any candidate, R included |
| go-around | R (no change) | true | G false, after the first predicted step |

**No runway lock (D12).** While G is false the model can change the runway at any time, also on the final (a change to
a parallel runway is a real manoeuvre). The judge reads R at the crossing.

**Go-around (D10): "abandon this approach".** A change of runway does not stop a descent and does not tell the judge
anything. A go-around does. It does not change R: R stays the frame of the heading words while the model vectors the
aircraft back. Its effects:

| Who | Effect of "go-around" |
|---|---|
| Executor (§5.4–§5.6, D27, D28) | "Go-around" changes no target. While G is true, a climb word climbs at the go-around angle (1.885°–3°), not at the nominal angle of the climb class, and "unspecified" holds the airspeed. The climb comes from the level that the go-around row says (rule 6). The heading word in force stays |
| Judge (§5.8, D33) | While G is true, no crossing of a threshold is an event, of R or of another candidate (not a landing and not a failure): the flight continues. A go-around before the DA point answers a failed DA check. The time limit of the flight grows by 900 s (the user, 2026-10-02) |
| Masks (§3.7, D14, D27) | Rule 5: no "no level-off" while G is true. Rule 6: a go-around row with "no level-off" in force also says a level above the aircraft. The procedure mask "no climb below the entry height" does not apply while G is true |
| Prior (§6.1) | G is an input: the aircraft is in a missed approach |
| Multi-aircraft (§8) | The decision is explicit: it can be counted and rewarded, and the separation judge can treat the aircraft as no longer on the approach (open with O6) |

A runway word ends G. The model can say R again or another candidate ("expect runway k" again). Thus one column says
the sequence approach → go-around → approach.

**Why "go-around" changes no target (D27).** The model flies the final with its own heading and altitude words (D2,
D3). An executor target for a go-around would replace a word of the model that is in force: a climb against "no
level-off", the course of R against the heading word. Then a target would come from the executor and not from a word
(§5.1). Thus a go-around changes only the angle of a climb word and the meaning of "unspecified" (§5.5, §5.6). The model
says the climb and the heading.

### 3.3 Heading column

**Meaning.** "Fly track θ and keep it", with θ = course(R) + 5k°, k the class of the word (D8). The executor turns in
the shorter direction for the first heading word (Pilot/Controller Glossary, FLY HEADING). Class 0 is the course of R.
Class 36 is the opposite direction (downwind). Classes 18 and 54 are the two base legs.

**Why a relative grid (D8).** Two reasons, both measured on the v6 train split (§11):

1. **The final must be on the grid.** The capture corridor permits 2° from the course. The four courses of KSTL (62.7°,
   122.3°, 242.7°, 302.3°) are 2.26–2.32° from the nearest absolute 5° value. With absolute words, no heading word holds
   a KSTL final. With D2 and D3 the model must hold the final with heading words.
2. **The same manoeuvre gets the same word at all airports.** The heading in force at the end of the labelled turn onto
   the final is very different between airports in absolute classes. It is much more similar in relative classes
   (§11.2). This helps the prior to use the data of one airport at another airport.

**Conversion and change of runway.** The executor converts a heading word to an absolute track when it hears the word.
It keeps that absolute target until the next heading word. A change of R does not turn the aircraft. The prior gets the
heading in force as the sine and cosine of its angle relative to the course of the current R (§6.1), not as a class.

**The frame where a word is heard (D46).** The labeller says each heading word relative to the course of the R in force
when the executor hears it. Where the Δ grid (§4.8) or the closed-loop reading (§4.9) moves a heading word across a
runway word, the word is the class nearest to its absolute track minus the course of the new R. For parallel runways
(courses 0.002–0.01° apart) the class does not change; for other runways the absolute track stays within half a step,
and the closed-loop reading corrects the rest. A class gives a track only with its course: a heading word is said when
its absolute track differs from the track in force, also when its class is the class in force said under another course
(the executor takes a heading word as new because it is said, not because its class is new). A change of R alone says no
heading word.

**Reading.** At each row, the labeller takes the smoothed track L = 4 s later (the lead; near the end, the track of the
last row). The word is the grid value, relative to the course of R at that row, nearest to that track. A new word is
said where that track leaves the target in force by more than half a step (2.5°). Thus a continuous turn is a series of
words, one for each 5°, and the time between the words gives the turn rate. The words continue from row 0 to the last
row of the sentence: they describe the turns onto the final and the final itself (§4.3).

**Envelope.** A word said at row r: from row r + L to the next heading word + L, the track is within ±4.5° of θ
(`envelope.heading_words_inside`). The rows of a word continue to the next heading word or to the end of the
sentence.

**Grid (D6, D35).** The grid is 5°. A word of class 0 gives no lateral correction: the aircraft keeps its lateral
offset until the model says a word of ±5°. The corrections come from the closed-loop reading (D32, §4.9), not from a
finer grid (§11.9).

### 3.4 Altitude column

**Meaning.** "Descend (or climb) to level T and keep it" (7110.65BB 4-5-7, DESCEND AND MAINTAIN). T is geometric MSL.
The direction comes from T and the present height. The angle column says how steep.

**Grid (D22).** 40 levels on three segments, each segment uniform:

| Segment | Levels | Step | Largest rounding error |
|---|---|---|---|
| 1 | 0, 60, …, 1,260 m | 60 m | 30 m |
| 2 | 1,380, …, 2,700 m | 120 m | 60 m |
| 3 | 3,150, …, 5,400 m | 450 m | 225 m |

The steps and the break points are constants of the vocabulary and go into the spec. A word is the nearest level; its
meaning is an absolute height, the same at every airport and from every position. The grid is an exact
dynamic-programming fit of at most three uniform segments that makes the squared rounding error of the observed
level-offs smallest for 40 levels (§11.7).

**Why a coarser grid higher up.** (1) 99.7 % of the level words are under 3,000 m (§11.7). (2) A controller assigns a
pressure altitude; the geometric height of one assigned level moves from day to day by about height × ΔT / 273 (ΔT: the
day's temperature deviation; the model does not see it). At KRDU this is about 15 m at 600–900 m and about 90 m at
2,000–3,000 m (§11.7). A fine step high up only divides this spread.

**What a coarse segment costs.** In segment 3 two assigned levels 1,000 ft (304.8 m) apart can round to one word, and
a small change of level inside one step has no word. Such words are rare (above 2,700 m: approximately 0.3 % of the
words, mostly the entry level at row 0). The 1,000 ft between opposite base legs (7110.65BB 5-9-1 b) is at low
altitude, where the step is 60 m. At an airport above approximately 600 m elevation the final approach uses the 120 m
segment (§6.4).

**"No level-off".** "Descend at the angle in force. Do not level off." The executor flies the nominal angle of the
class in force and has no other law for it (§5.5, §5.7). The judge stops the flight at the threshold or at the ground.

**Why MSL and not height above the threshold.** Controllers assign MSL levels. Relative to the threshold elevation, the
level distributions of the five airports become less similar, not more (§11.2). KSJC has 28 % of its level words on
levels that the other four airports almost never use. This is airspace, not frame.

**Envelope.** The tube from the row of the word: max(T, h0 − s·tan γ_hi) − ε ≤ h ≤ max(T, h0 − s·tan γ_lo) + ε while
it descends, T ± ε after it arrives (s is the horizontal distance flown from the row of the word). ε depends on the
level (D52): **ε of a level = half the larger gap to its neighbouring levels + 10 m** (the fit residual). That gives
40 m for 0–1,200 m, 70 m for 1,260–2,580 m and 235 m for 2,700–5,400 m. The band covers the largest rounding error of
the word: the top level of a segment collects the heights up to half the next segment's step above it (60 m above
1,260 m, 225 m above 2,700 m). For "no level-off" there is no lower bound T, and **ε = 40 m**: the word has no level, so
no rounding to cover, and the tube's width comes mainly from the edges of the angle class. A new angle word starts a new tube. A containment rate is given
with these widths (principle 6). The level detection of the labeller does not use ε (§4.4).

### 3.5 Angle column

| Class | Nominal | Range |
|---|---|---|
| Level | 0° | used only to hold a level that the aircraft reached |
| Descent 1 | 1.5° | −0.5° to 2.0° |
| Descent 2 | 2.5° | 2.0° to 2.75° |
| Descent 3 | 3.0° | 2.75° to 3.75° |
| Descent 4 | 4.5° | 3.75° to 10° |
| Climb | No angle in the word. The executor's angle (D28): the nominal of D15 while G is false (1.5°, D56); 1.885°–3° while G is true | 0.5° to 15° climb (the labeller's range of a climb piece) |

The four descent classes are a k-means on tan(angle) of the descent pieces of the train days, with each piece weighted
by the square of its length (D54): a piece flown at the nominal angle of its class ends length · |tan a − tan c| from
its observed end, so the k-means makes the sum of the squared end-of-piece height errors smallest. The values of the
table are the user's choice (D56): the fit on all train days rounded to 0.25° (fitted 1.51 / 2.45 / 3.09 / 4.45°, edges
1.98 / 2.77 / 3.77°, climb 1.47°). The old labeller's pieces (artefact `v6_20261002`) gave 0.92 / 2.13 / 3.06 / 4.41°
under the same rule: the same flights and a median piece angle of 2.98° against 2.97°, but its shallowest class (12 % of
the weight at 0.92°) is not in the new pieces.
With D3 the model holds the glidepath with these classes; the closed-loop reading gives the corrections (D32, §4.9).

**Rounder values (D15, D56).** The spec measurement writes each value that it fits from data (the descent nominals and
edges, the climb nominal) with candidates rounded to 0.5°, 0.25° and 0.1°, and for each candidate the end-of-piece
height error that it leaves on the descent and on the climb pieces (`instructions/measure.py` `rounding_candidates`).
The spec takes the row that the user chose (`instruction_spec --candidate`; D56: 0.25°). Descent 3 is then the 3.0°
glidepath that 24 of the 25 candidate runways publish (§11.4). The values that the measurement takes as percentiles
are rounded by their own rules (turn rates to 0.1°/s, the bank limit to 1°, the corridor to 5 m, 0.05° and 1°, the
acceleration to 0.1 m/s²).

**Climb (D28).** The vocabulary has one climb word: "climb". The word has no angle. The angle is a value of the
executor (§5.5), from the state G:

- **G true (a missed approach).** The executor climbs at the go-around angle: the steady climb angle that the thrust
  limit permits at the present airspeed, not more than 3° and not less than 1.885°. 1.885° is the minimum gradient of a
  missed approach, 200 ft per NM (AIM 5-4-21 b). 3° is the upper limit. The real go-around climbs are steeper (§11.6),
  so the 3° limit usually applies.
- **G false.** The executor climbs at the nominal angle of the climb class, 1.5° (D56): the length-weighted median of
  the climb pieces of the train days (1.47°), rounded to 0.25°. The train days have 1,339 climb pieces against 227,559
  descent pieces (0.6 %).

The labeller does not read a go-around angle: a climb piece is a climb when its angle is 0.5° to 15°, in and out of G.

### 3.6 Speed column

**Meaning.** "Hold ground speed V." Controllers assign indicated airspeed; the data has no airspeed and no wind, so the
words use ground speed. "Unspecified" means that the pilot flies the approach speed of the aircraft type. An approach
clearance cancels the assigned speeds (7110.65BB 5-7-1 d). In this design the labeller starts "unspecified" at the
capture row (D4, §4.5).

**Steps (D43).** In a change of speed, the labeller says each grid value on the way (§4.5). Each speed word is then a
step of 5 m/s, and the time between the words gives the rate of the change, as the time between the heading words
gives the rate of a turn. The executor makes each step at a_max (§5.6).

**Envelope.** A transition is monotone (a step back ≤ 5 m/s) with acceleration ≤ a_max (1.4 m/s², spec).
After it, the speed stays within V ± 5 m/s. "Unspecified" checks only the range.

**Wind (O8, open).** At constant airspeed, the ground speed changes when the aircraft turns in wind. Such a change can
become a speed word that no controller said. A check against the surface winds of the METAR data is open.

### 3.7 Grammar rules

The labeller checks these rules. The prior applies them as masks when it decodes (`instructions/grammar.py`). The
labeller conformance (§9.2) covers them.

1. At the first predicted step, each column says a value. The runway column says a candidate.
2. The runway column follows the table of §3.2.
3. When a new level T needs a direction that the angle in force does not give (level, or the opposite direction), the
   same row says an angle of the correct direction.
4. "No level-off" needs a descent class in force or in the same row.
5. "No level-off" is not permitted while G is true (D14). The model first says a runway word, which ends G; in the same
   row (the runway column comes first) or later it can say "no level-off".
6. A row that says "go-around" while "no level-off" is in force also says a level T above the present height of the
   aircraft (D27). Rule 3 then makes the same row say "climb". With rules 5 and 6, "no level-off" is never in force
   while G is true, so every climb of a go-around has a target from an altitude word.

**Procedure masks.** The prior also decodes under three masks from the procedure of R (principle 3). The first two
block the altitude and angle words that would take the aircraft below a lower limit; the third blocks the climb:

- inside the FAF and the LPV cone: the glidepath lower edge, the published glidepath − 60 m (nowhere else: the RNAV
  floors outside the FAF disagree with 10–14 % of the recorded tracks);
- before the join (the first row inside that region): the published DA;
- once the aircraft is below the entry height before the join: no climb.

**Masks while G is true (D14).** Only one mask stops a go-around: "no climb below the entry height". It does not apply
while G is true, and its stretch starts again after the go-around. The other two masks are lower limits; a go-around
climbs above them, so they apply. Rules 1–4 apply while G is true as at every row.

### 3.8 Example

A KRDU arrival that lands on 23R (course 225.0°). At row 0 it is on the left downwind of 23R: 3.0 km before the
threshold, 5.0 km left of the final, track 045°, level at 1,200 m MSL, 115 m/s. Only rows with a word are shown; an empty
cell is "unchanged". Relative headings are given in degrees.

| t (s) | Runway | Heading | Altitude | Angle | Speed | What happens |
|---|---|---|---|---|---|---|
| 0 | 23R | +180 | 1,200 m | level | 115 m/s | Expected runway 23R. Downwind, level |
| 40 | | | | | 105 m/s | Speed reduction |
| 106–134 | | +175 … +90 (one word each row) | | | | Left turn onto the base leg, 4 s before the track |
| 120 | | | 900 m | descent 2 | | Descend to 900 m |
| 156–200 | | +85 … 0 (one word each row) | | | | Left turn onto the final, with heading words (D2) |
| 204 | | | | | unspecified | Capture row; the pilot's own speed (D4) |
| 210 | | | no level-off | descent 3 | | Final descent |

At approximately t = 390 s the aircraft crosses the threshold of 23R. The judge classifies the crossing and the DA
check (§5.8). The table is the labelled sentence. The prior observes rows 0–7 and starts to speak at row 8
(t = 16 s); there it says the five words in force at that row.

---

## 4 Labeller

The labeller reads one sentence from one observed flight (`instructions/labeller/read.read_flight`), in two passes. The
open-loop reading (§4.1–§4.8) reads the observed words from the observed track. The closed-loop reading (§4.9) flies
them with the executor and adds the correction words; its sentences are the training sentences.

### 4.1 Signals, cut and gate

- **Signals** come from the ts data plane (`build_series`): the altitude repaired at read time, MSL, 2 s rows on even
  UTC seconds, the airport frame. The test days are sealed: the labeller never opens them (contract C32).
- **Smoothing:** centred moving means of the track (6 s), the altitude (10 s) and the ground speed (10 s). The flown
  distance is the integral of the smoothed ground speed.
- **Cut** (`read.admit`). A landing passage is a row where the flight crosses the threshold plane of its landed runway
  inside the landing screen of the harvest: lateral ≤ 1,000 m and ≤ half the spacing to a parallel runway, height
  ≤ 100 m, both interpolated between the two rows (`read.landing_passages`). The sentence ends before the last landing
  passage after which the flight does not come back before the threshold.
- **Gate.** The labeller refuses a flight, with its reason, when: its rows are not 2 s apart; its landed runway is not
  a candidate; fewer than two rows remain before the threshold; a raw ground speed is outside 15–350 m/s; a smoothed
  ground speed is outside the speed words (20–250 m/s ± 5 m/s); a landing passage that it comes back from is not inside
  the low pass of a go-around (§4.6); a go-around point is on the runway (§4.6). The readings of §4.2–§4.7 add their own
  refusals.

### 4.2 Runway word

The first row says the runway of the first approach (D26). For a flight without a go-around, this is the runway on
which the flight landed (the harvest assignment). For a flight with a go-around, it is the runway of the first low pass
(§4.6). The data shows a change of runway only at a go-around, so a labelled sentence changes R only with the runway
word that ends G (D19). A candidate without a published threshold crossing height (TCH), a
glidepath angle or an LPV DA is refused before labelling (the judge needs all three, §5.8).

### 4.3 Heading words

- **Frame:** the course of R at each row (D8): the runway of the first approach, then, from the runway word that ends
  a go-around, the runway of the next approach (§4.2, §4.6).
- **Reading:** §3.3, at every row from row 0 to the last row of the sentence. A word in force keeps its track when R
  changes; only a new word reads the new course. After a change of R, a new word with the class in force is said,
  because its absolute track differs from the track in force (D48, as D46 on the Δ grid and in the closed loop).
- **Capture row** (D4, D26): the first row of an approach from which the track stays in the capture corridor of its
  runway, before the threshold, to the end of the approach. The landing approach must end in the corridor; otherwise
  the flight is refused. An approach that ends at a go-around row outside the corridor has no capture row.
- **Strata of the readouts.** A flight is vectored when, before the capture row of its landing approach, the net turns
  of its runs of rows that turn one way faster than 0.2°/s add up to 90° or more. Otherwise it is straight-in.

### 4.4 Altitude and angle words

**Pieces.** The smoothed altitude is fitted against the flown distance by straight pieces. From the first row of a
piece, the piece grows as long as the least-squares line through its rows keeps every residual within 10 m (the fit
residual). A piece has at least two rows.

1. **Level.** A piece is level when it lasts at least 20 s and every row is within 25 m of the piece's own median. This
   test is physical and does not use the grid: a test against the 60–450 m steps would miss a level between two grid
   levels (a level at 630 m is 30 m from both neighbours) or take a slow descent for a level. The word of a level is the
   grid level nearest to its median. Consecutive level pieces with the same word are one level.
2. **Move.** Every other piece is a move, with its path angle (descending positive). Consecutive moves in one direction
   are a run.

**Words.**

1. A level at row 0 says its level and the angle "level".
2. A run says its altitude word at its first row: the word of the next level; the grid level nearest to the height
   where the direction changes without a level; or "no level-off" when the run is the last descent of its approach (it
   reaches the threshold, or the go-around row; D26, §4.6). A run whose word is the word in force (a move between two
   heights that round to one level) says nothing: the word cannot be said (§3.4).
3. Each move says its angle class where the class differs from the class in force: the descent class whose range holds
   the angle, or "climb" for a climb of 0.5° to 15° (§3.5).
4. A step is two levels with different words and no move between them. The altitude word is said at the last row of
   the first level. Its angle is read from there to the first row within 10 m of the height of the second level.
5. The flight is refused when it climbs at the end of the sentence, when a level is outside the grid, when an angle is
   outside the classes, or when a level word is on the wrong side of the height where its run starts (by more than its
   ε).

The tube of §3.4 checks each word, with the ε of its level (D52).

### 4.5 Speed words

**Pieces.** The smoothed ground speed is fitted against time by straight pieces, as the altitude (§4.4), with every
residual within 1.5 m/s. A piece is a hold when it lasts at least 20 s, its slope is at most 0.1 m/s², and every row is
within ±5 m/s of the grid value nearest to its median. A hold that follows a hold and stays inside the band of its value
is the same hold. Every other piece is a transition, accelerating or decelerating. Consecutive transitions in one
direction are a run. The target of a run is the value of the next hold; or the grid value nearest to the speed where the
direction changes without a hold; or, when the run continues into "unspecified" or to the end of the approach, the grid
value nearest to the speed at its last row before that.

**Words.**

1. Row 0 says the value of the hold that starts there, or else the grid value nearest to the smoothed speed there.
2. A hold that follows another hold says its value at its first row.
3. **Steps (D43).** In a run, the labeller says each grid value from the word in force to the target, in order. It says
   a grid value at the first row at which the smoothed speed is nearer to that value than to the value before it, and
   the target at the latest at the last row of the run. Thus, in a change of speed, the word in force is approximately
   the grid value nearest to the observed speed. Inside a run the labeller says no value in the other direction, so the
   noise of the speed gives no words.
4. A target outside the grid (20–250 m/s) refuses the flight.

**"Unspecified" (D4, D26).** Each approach has its own reading. "Unspecified" starts at the capture row of the
approach, with two exceptions:

1. A hold of at least 30 s that ends at or after the capture row, and ends 9,260 m (5 NM, 7110.65BB 5-7-1 b.4) or
   more before the threshold, keeps the speed word in force. "Unspecified" starts at the end of the last such hold.
2. A change of speed under way at that row, which started less than 20 s before it, is already the pilot's speed.
   "Unspecified" starts where it started.

From that row to the end of the approach the column says no other word. An approach that ends at a go-around row
outside the capture corridor has no capture row: its speed words continue to its end, without "unspecified". The
reading of the next approach starts at the go-around row.

**Why steps (D43):**

- A speed word gives a target, not a rate. The rate of a real change of speed is a choice of the pilot or the
  controller. On the smoke artefact it is 0.24 m/s² in level flight and 0.19 m/s² in a descent (median), and it changes
  by more than three times in each (§11.12). It does not follow from the aircraft.
- The dynamics cannot give it. Their drag polar is clean: no flaps, no landing gear, no speedbrakes. The thrust floor of
  −20 % of the installed thrust is a stand-in for these devices (`outputs/envelope.py`). Thus "the deceleration that
  the aircraft can do" is not a physical value in this model.
- One target word with a fixed rate of the executor moves the flown aircraft ahead of the observed aircraft along the
  path, or behind it, by more than 1 km in 31 % of the flights (before "unspecified"). A faster rate makes this worse.
  With the steps and the executor at a_max, no flight is more than 1 km ahead or behind (§11.12, a one-dimensional
  estimate).
- The prior says the steps. Thus the prior, not the executor, sets the rate of a change of speed. A controller who
  keeps the spacing between two aircraft does this with speed.

### 4.6 Go-around words (D18, D19)

**The data.** The training days hold 109 go-arounds (R40 v2, `go_around_census`; §11.6). In 73 the go-around point
lies inside a labelled sentence of artefact `v5`. In 11 more it lies inside the arrival slice of a flight that crosses
a threshold plane in its low pass; the gate of §4.1 reads these as go-arounds. In 23 it lies before the sentence: the
aircraft left the 25 km slice, and the slice starts at its return.

**The reading.**

1. **The go-around** (`instructions/labeller/go_around.py`).
   - A low pass is a run of rows (gaps of at most 3 rows) on the final of one candidate: at most 500 m from its
     centreline, from 10 km before its threshold to 3 km past it, at most 600 m above it, with at least 1 km of
     progress along the landing direction between its first and last rows (it flies the final; it does not cross it).
     Two passes on two candidates with common rows (close parallels) are one pass, on the candidate with the smaller
     median offset. The lowest row of the pass is the go-around point.
   - A low pass is a go-around when, before it (since the previous pass), the aircraft held a level for at least 20 s at
     least 150 m above the go-around point, and after it (before the next pass or the end) held a level for at least
     20 s at least 150 m above the point. The last pass of a sentence (the landing approach) is never a go-around.
   - A go-around point past the threshold and at most 15 m above it is on the runway: a touch-and-go, or a landing
     balked in the flare. The vocabulary cannot say that landing, so the flight is refused.
2. The go-around row is the first row of the climb after the low pass: the row where the altitude reading starts the
   climb run, with its level word and its climb word (D26). The runway column says "go-around" in that row; G becomes
   true. A go-around without a climb word is refused, with its reason.
3. The climb is read as altitude and angle words, as every climb. Its level word and its climb word are in the
   go-around row (rule 6). While G is true the executor flies the climb at the go-around angle (D28, §5.5).
4. The runway word that ends G (D19): at the first level-off after the go-around climb, and not later than the row of
   the next "no level-off" (rule 5). It says the runway of the next approach: the runway on which the flight landed,
   after its last go-around.
5. A landing passage that the flight comes back from must lie inside the low pass of a go-around; otherwise the
   flight is refused (§4.1).
6. Each approach is read separately (D26): the last descent that reaches the go-around row says "no level-off" (§4.4);
   the capture row and "unspecified" belong to the approach (§4.5); the first row says the runway of the first low pass
   (§4.2).

**Why each approach is read separately (D26).** A word must not tell the prior what only later rows show (the rule of
D23). Two readings use the end of an approach: "no level-off" is the last descent that reaches the end, and the capture
row needs the track in the corridor up to the end. If the end were the landing for every approach, the words before a
go-around would be a low level and a speed value, never "no level-off" or "unspecified" (§11.6). These words then tell
the prior that this approach will not land, a fact that only the rows after the go-around show. The prior would also
learn "go-around" only after a low level and a speed value. In closed loop the model flies its final with "no
level-off" and "unspecified" in force, and there the prior would give "go-around" a probability near zero, where the DA
check needs it (D3). With each approach read separately, "go-around" occurs with "no level-off" and "unspecified" in
force, as on a real final.

**Not read.** The 23 go-arounds before their sentence need a longer arrival slice, and 9 sentences of `v5`
end at a low go-around that the harvest took for the landing (it takes the best-aligned crossing under 100 m, not the
last; R40 v2 §5). Both are changes of the harvest or of the data plane, which other lines (evaluation, the optimizer,
the one-tier models) share.

### 4.7 Assembly and artefact

- All words go on the row grid (Δ; §4.8). A word equal to the word in force is not a word. The first row says all five columns.
  Two different words in one column in one row: refused.
- The reading is named `instruction-v4`. The spec records the grids, the classes, the tolerances and the reading name.
  The labeller conformance (§9.2) checks the reading code, the grammar included.
- The open-loop sentences are the input of the closed-loop reading (§4.9). The training sentences of the prior are the
  closed-loop sentences with their flown states.
- The code refuses an artefact of another reading name, and every prior trained on one (principle 8).

### 4.8 Row interval: an ablation (D11)

**What the row interval Δ sets.** The data plane gives a row every 2 s. Δ sets the grid of the words, how
often the prior speaks and the executor hears, the length of a sentence (median 154 rows at 2 s) and the step of a
multi-aircraft scene. Each column changes its word at only 1–2 % of the 2 s rows (artefact `v6_20261002`), so a larger
Δ can be sufficient. The ablation measures this at the labeller stage, before the prior trains.

**How the ablation changes only Δ.** The labeller reads each flight at the 2 s rows of the data. Then it puts
the words on a Δ grid:

1. The Δ rows are the rows on UTC multiples of Δ, so the aircraft of a scene are on one grid.
2. Each word of the 2 s reading goes to the nearest Δ row; a word exactly between two Δ rows goes to the later one
   (D45). A Δ row says, for each column, the last word that goes to it, if that word differs from the word in force. A
   heading word that moves across a runway word is said in the frame where it is heard (D46, §3.3).
3. The first Δ row says all five columns. The grammar rules (§3.7) are checked again on the Δ grid. The rounding keeps
   the order of the words, and the words of one 2 s row stay in one Δ row (a level word and its angle word, a
   go-around row).

**Why the nearest row (D45).** A word put on the next Δ row is late by (Δ − 2)/2 on average, and a word said when the
matched point has passed its place (§4.9) is late by Δ/2 more: approximately 1 s at Δ = 2 s, 3 s at 4 s and 7 s at 8 s.
A late heading word makes a turn end late: 3 s late at 70–100 m/s leaves approximately 200–300 m of lateral offset after
a turn of 90°. The ablation would then read the rounding, not the row interval (§11.13). With the nearest row, the mean
lateness is zero at Δ = 2 s and 1 s at 4 and 8 s (a word exactly between two rows goes to the later one). The spread of
±Δ/2 stays; that spread is the cost of a coarser Δ.

Thus one reading gives every Δ. Δ must be a multiple of 2 s, and it must divide the 16 s observation of the prior
(D25): the first predicted step is a Δ row.

**What a larger Δ changes (to read in the ablation; no criteria here, D7):**

| Part | Change |
|---|---|
| Heading words | In a 3°/s turn the track moves 12° in 4 s, so one word jumps two or three 5° classes. The lead L = 4 s is one row at Δ = 4 s and less than one row above it |
| Executor | The cycle stays 1 s. It hears the words at each Δ row. The heading law (arrive L after the hearing, the stopping rate) flies larger steps |
| Speed words | A change of speed faster than 5 m/s in Δ puts two steps (D43) into one row; the row says the last one, and the executor flies it at a_max |
| Final approach | With D2 and D3 the model corrects the final only every Δ; an error grows for a longer time before the next word |
| Prior | Fewer steps for each flight and more changes for each step. The observation stays 16 s (`N_LOOK` = 16 s / Δ rows: 8, 4, 2). The motion inputs do not change with Δ: they come from the 2 s before the row (D25, §6.1). A loss for each step cannot be compared between two Δ; a loss for each flight or each second can |
| Multi-aircraft | The scene step is Δ |

**Values (D25).** Δ = 2, 4, 8 s. A Δ of 6 s is not used: 16 s is not a whole number of 6 s rows.

**Readings (D34).** The closed-loop reading (§4.9) makes each replay follow the observed path, so the landed share no
longer shows how well a Δ carries a flight. At each Δ the ablation reads instead:

1. the correction words of the closed-loop reading, for each flight and each column;
2. the errors left where §4.9 makes no correction (a level hold, a climb, no steeper or shallower class): the largest
   |e_y| and |e_h| for each flight;
3. the share of the rows where §4.9 permits a correction and the flown path is outside the tolerances (|e_y| > Y,
   |e_h| > H; D50);
4. the replay outcomes (§5.8).

The user compares the Δ values on these readings; this document sets no threshold (D7).

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ; the closed-loop artefact stores the flown states on
the 2 s rows (D51). At Δ = 2 s it is the value of now.

### 4.9 Closed-loop reading (D32)

**What it does.** The open-loop reading (§4.1–§4.8) gives the observed words. Then the labeller flies the sentence with
the executor of the artefact's executor spec and compares the flown path with the observed path at each row. When the
flown path leaves the observed path by more than a tolerance, the labeller adds a correction word. The result is the
training sentence: the observed words, the correction words and the flown states.

**Why.** A heading word gives a direction, and an angle word gives a path angle. Neither gives a position. The real
flights stayed on their paths because of the aircraft's own lateral and vertical guidance (LOC, LPV) and the
controller's radar. Their corrections are smaller than half a grid step, so the open-loop reading shows none: in the
smoke artefact of stage A, the last heading word is class 0 in every sentence. Flown without these corrections, the
errors of the directions add up over 10–25 km. With an exact model of the words, 47 % of the vectored flights end more
than 106.7 m from the centreline, and the final descents end between 45 m low and 53 m high (p10, p90; §11.9). A finer
grid does not remove this: the real corrections are smaller than any grid step. The closed-loop reading puts the
missing corrections into words. Each correction is a reaction to an offset of the flown states, and the prior trains
on the flown states (§6.1). Thus the prior learns "an offset gives a correction word" from inputs that the closed loop
also gives.

**The flight.**

1. The executor starts at the first predicted step, from the observed state of that row, as in free generation. The
   rows before it stay observed.
2. The aircraft flies with its own dynamics or with a stand-in's, by the rule of the replay (`autopilot/replay.py`
   `group_of`). A flight that the replay does not fly (no identified type, no aircraft dynamics, or no published approach
   speed) gives no training sentence. The artefact counts these flights by reason.
3. At each row, the labeller finds the matched point, then decides the words of the row: the observed words that the
   matched point has reached ("When the observed words are said", below) and the correction words. The executor hears
   them together.
4. The closed-loop reading runs at each row interval Δ of the ablation. It takes the observed words with their 2 s
   times (the open-loop reading at 2 s, not the Δ grid of §4.8). The words, the corrections and the flown states are on
   the Δ rows and belong to that Δ.
5. The observed path includes a go-around and the next approach (D26). The comparison continues while G is true.
6. The flight ends when the executor is done (a crossing, the ground, the dynamics) or at the time limit of the replay
   (§5.8). The closed-loop sentence has one row for each Δ row of the flight: its length is the flown time, not the
   observed time.

**The comparison.** The matched point is the point of the observed path nearest to the flown position. The search
starts at the matched point of the row before and goes forward, so that a path that crosses itself does not jump. At
the matched point:

- the lateral error e_y is the distance of the flown position from the observed path, perpendicular to the observed
  track there (positive to the right);
- the vertical error e_h is the flown height minus the observed height.

The reference is the matched point, not the observed point at the same time. Thus a difference along the path (a
difference of time) is not corrected, and the speed words stay the observed ones.

**Past the end of the observed path (D44).** The observed slice ends before the threshold, and the flown aircraft flies
on to it: approximately 150 m (median), at most approximately 750 m (§11.13). There the path continues along the line of
its last segment, and the lateral error is measured against that line for the readouts; no vertical error is measured
there. The labeller says no correction there, lateral or vertical: a correction in force ends at the end of the path, and
the observed word of its column is said again (in the readouts this word counts as a correction word; the prior and the
executor read every word alike). The aircraft flies the last observed words to the threshold. These rows count as rows
without correction (D34). Why: there is no
observation there; the slope of one 2 s segment is not a measurement (the ADS-B altitude has steps of 7.6 m, so one
segment can be approximately ±3° wrong, ±20 m over 400 m); and the DA point, 0.9–2.1 km before the threshold, always
lies before the end of the observed path, so the DA check does not change.

**When the observed words are said (D42, D45, D46).** The observed time of the matched point is the time at which the
observed aircraft was at that point. At each Δ row, the labeller says every observed word whose 2 s time is less
than Δ/2 after the observed time of the matched point, and that it did not say before (D45; a word exactly Δ/2 after
waits for the next row, as on the Δ grid). Thus a word is said at the Δ row nearest to the place where the observed
aircraft heard it, and its mean lateness is zero. When one Δ row says
more than one observed word of a column, it says the last one. A heading word is said in the frame where the executor
hears it (D46, §3.3). When the flown aircraft is behind the observed aircraft, the observed words wait. The first
predicted step says every column (rule 1): the observed words in force before Δ/2 after its observed time.

Why:

- A word gives a target, not a time. The speed steps (D43) give the rate of a change of speed, but the steps, the
  dynamics and the 5 m/s grid still leave a difference along the path: the flown aircraft moves ahead of the observed
  aircraft, or falls behind it (closed loop, select, Δ = 2 s, vectored: p50 221 m, p90 348 m; §11.13). The comparison
  does not correct this difference.
- A turn word must come where the observed aircraft turned. A word said at the observed time comes where the flown
  aircraft is at that time: an aircraft d ahead along the path before a turn of 90° is d to the side after the turn.
  One correction class closes 6–9 m/s, too slowly for such an offset (§11.11).
- The controller gives a turn at a place, not at a time. The prior learns from the flown states when to say a word, so
  the words must agree with the flown states.
- A go-around word said at the observed time can come after the flown aircraft has crossed the threshold. Said at the
  place, it comes where the observed aircraft went around.

Thus the flown time to the threshold is not the observed time. The flown states keep the observed path, not the
observed time (D36).

**Lateral correction.**

1. When |e_y| > Y (the lateral tolerance), and the row says no new observed heading word, the labeller says the
   heading class one step (5°) from the observed word in force, toward the observed path.
2. When |e_y| < Y / 2, or e_y changes its sign, the labeller says the observed word in force again — except an
   overshoot (D53): when e_y changes its sign and |e_y| > Y, it says the opposite correction in the same row (item 1
   toward the other side).
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.

**Vertical correction.**

1. Only while a descent class is in force (toward a level or with "no level-off"). During a level hold the level word
   is the target; its rounding to the grid (§3.4) is not corrected. A climb has one class, so a climb gets no
   correction.
2. When e_h > H (too high), the labeller says the next steeper descent class. When e_h < −H (too low), it says the next
   shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then there is
   no correction.
3. When |e_h| < H / 2, or e_h changes its sign, the labeller says the observed angle class in force again — except an
   overshoot (D53): when e_h changes its sign and |e_h| > H, it says the opposite correction in the same row (item 2,
   where that class exists). A new observed altitude word or angle word ends a correction.

**The words.** A correction word is an ordinary word of its column (§3.3, §3.5). The grammar (§3.7) checks it. The
envelopes of the closed-loop sentence are checked on the flown states. The envelopes of the observed words on the
observed track stay as the readout of the open-loop reading. The capture row, the runway words and the go-around rows
come from the open-loop reading: the decisions come from the observed track; the closed-loop reading only adds
corrections. The executor reads only words (D2, D3).

**Tolerances (D32).** Y = 30 m and H = 15 m, constants of the labeller in the spec. With Y = 30 m the flown path ends
within approximately 30 m of the observed path, which ends 2 m from the centreline (median), far inside the runway
limit of 106.7 m. H = 15 m is larger than the fit residual of the altitude pieces (10 m, §4.4), so that the noise of
the fit starts no correction.

**Where the code goes.** The labeller package (`instructions/`) does not import the executor; only the runners and
`autopilot/` read `instructions/` (`tests/test_architecture.py`). The closed-loop reading is a module in `autopilot/`.
It reads the open-loop artefact and the executor spec.

**Artefact.** For each split and each Δ: the closed-loop sentences (each correction word marked as a correction); the
flown states on the data's 2 s rows, with the Δ rows marked (position in the airport frame, MSL height, track, ground
speed, vertical rate; D51); the count of
correction words for each column; the flights without a training sentence, by reason.

---

## 5 Executor

### 5.1 Principle

The executor flies the words with the point-mass dynamics of the ts control path. It has no state that the words do not
give. It has no law that makes a track from the runway or from a procedure (D2, D3). The user's rule of 2026-09-24: the
executor uses only the vocabulary; a parameter mined from data shows that it does not generalize.

### 5.2 What the executor reads

| Input | Use |
|---|---|
| The vocabulary spec | Grids, nominal angles, tolerances, lead, turn-rate limit, bank limit, speed change rate |
| The words of each row | The targets |
| The course of each candidate runway (`candidates.json`) | Conversion of heading words (§5.4) |
| The distance to the threshold of R | The deceleration to the approach speed (§5.6) |
| The aircraft: aerodynamic data, installed thrust, published maximum landing mass, published approach speed | Dynamics and "unspecified" speed |

The executor laws do not read the TCH, the glidepath angle or the DA. Only the judge reads them (§5.8).

### 5.3 One cycle

The laws of §5.4–§5.6 change the words in force into three wanted rates: the track rate χ̇*, the path-angle rate γ̇*
and the airspeed rate V̇*. The inner loop solves the point-mass equations of the ts control path
(`aerodynamic_model.torch_dynamics`) for the three controls that give these rates:

```
A = −χ̇*·V·cos γ / g          (= n·sin φ; compass and mathematical turns have opposite signs)
B =  γ̇*·V / g + cos γ        (= n·cos φ)
φ = atan2(A, B),   n = B / cos φ
T = m·(V̇* + g·sin γ) + D(V, n, h)
```

D is the drag of the dynamics' own polar at the commanded load factor and the ISA density at the height. The limits act
in this order, each on the value that it bounds:

1. Bank: not more than the bank limit (32°, spec), and changed at not more than the roll rate p = 5°/s. Then
   n = B / cos φ keeps γ̇*: a limited bank costs turn rate, not the path.
2. Load factor: in [0.5, 2.0].
3. Stall floor: the speed law asks for no airspeed below 1.10·V_stall(n) at that load factor (§5.6).
4. Thrust: in [−0.2, 1.0]·T_max. The drag polar is clean: it has no flaps, no landing gear and no speedbrakes. The
   negative part of the range stands for these devices. A limited thrust costs airspeed rate.

One cycle is 1 s: the controls are held (zero-order hold), and RK4 integrates the point-mass equations. Layer 1 of the
judge records each limit that bound (§5.8).

### 5.4 Lateral law

- **Heading words.** When the executor hears a heading word, it sets the absolute target θ = course(R) + 5k° and keeps
  it until the next heading word. The word says the track L = 4 s later, so the executor turns to arrive at θ at that
  time. The track rate is the smallest of three rates (`lateral.word_rate`): the error e divided by the time left until
  L runs out (not less than two cycles; after that, the executor holds θ); the stopping rate √(2·g·p·|e| / V), the
  fastest rate whose bank the executor can still take out before e is gone (it cannot know which word ends a turn); and
  the turn-rate limit 4.7°/s. The first word turns in the shorter direction. A later word measures e from the target
  before it, so a series of words never reverses a turn while the aircraft lags behind it.
- **Go-around (D27).** "Go-around" does not change the heading target. The executor keeps flying the heading word in
  force. The model turns the aircraft with heading words, as at every other row.
- Nothing else. There is no capture state, no turn onto the final, no centreline tracking (§5.7).

### 5.5 Vertical law

The inner path-angle loop: γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max), τ_γ = 2 s. γ̇_max is twice the smallest rate that keeps
an entry into the steepest descent class inside its tube: 2 × V·γ_lo² / (2ε), with γ_lo the lower edge of the steepest
class and ε the narrowest level band of the grid. The modes choose γ_ref (climbing positive):

| Words in force | Mode | γ_ref |
|---|---|---|
| Level T + level | Hold | −(h − T) / (V·τ_h), τ_h = 4·τ_γ = 8 s; within the nominal angle of the steepest descent class and the climb angle (γ_GA while G is true) |
| Level T + descent class | Descend to T | −nominal angle of the class; level-off starts at V·γ²/(2·γ̇_max) above T |
| Level T + climb, G false | Climb to T | +nominal angle of the climb class (D15, D28); level-off as above |
| Level T + climb, G true | Go-around climb to T | +γ_GA, the go-around angle (below); level-off as above |
| "No level-off" + descent class | Descend | −nominal angle of the class; no level-off |

A target that the executor has captured (its level-off started) stays captured until a new altitude word or angle word,
so the mode does not change back and forth at the level.

**The go-around angle (D28).** γ_GA = min(3°, max(1.885°, γ_T)). γ_T is the steady climb angle at the thrust limit and
the present airspeed: sin γ_T = (T_max − D) / (m·g), with the drag D of the present state at load factor 1 (the
point-mass model of §5.3). The executor computes γ_T at each cycle from the aircraft data that it already reads (§5.2).
1.885° is the minimum gradient of a missed approach, 200 ft per NM (AIM 5-4-21 b); 3° is the upper limit. When γ_T is
less than 1.885°, the reference is 1.885° and the inner loop gives what the thrust permits;
layer 1 of the judge records the limit that bound (§5.8). No word of the vocabulary says an angle of a climb: the
vocabulary has one climb word (§3.5).

**No climb without a target.** Rules 5 and 6 (§3.7) make sure that "no level-off" is never in force while G is true.
Thus the executor has no mode "climb with no target", and "go-around" alone starts no climb.

### 5.6 Speed law (D43)

V_ref = V_g / cos γ (ground speed to airspeed without wind), not below 1.10·V_stall(n). For a speed word, the speed
changes at a_max, the largest acceleration of the speed envelope (§3.6, a value of the spec), and goes exponentially
into the last 5 m/s. A speed word is a step of 5 m/s (D43), so the executor makes each step in a few seconds; the rate
of a longer change comes from the words. Where the thrust limits cannot give a_max, the thrust limit binds (§5.3), and
the replay counts it. "Unspecified" is the published approach speed of the type (an indicated airspeed at the maximum
landing mass), scaled by √(m / maximum landing mass) to the mass of the aircraft and flown as the true airspeed at the
present density. The deceleration to it is the larger of a_U = 0.25 m/s² (a speed step of 5 m/s divided by the minimum hold of
20 s) and the rate that reaches it at the threshold of R, not more than a_max. While G is true,
"unspecified" means the pilot's own speed in a missed approach: the executor holds the airspeed that the aircraft has
at the go-around row (D27; Claude's reading, not checked in the regulation text). A speed word of the model ends it at
any row.

### 5.7 What the executor does not do (D9)

The executor has no law of its own for the approach. It has:

- no runway lock (D12);
- no capture state, no turn onto the final, no heading bend and no own intercept (D2);
- no centreline tracking (D3);
- no special law for "no level-off": no aim at the threshold crossing point and no tube limit (D3);
- no glidepath floor and no level flight below the glidepath (D3, D9).

The words do these things. Only the judge and the masks read the procedure (principles 2 and 3).

### 5.8 Judge

**Layer 1, each cycle.** The wanted rates, the rates given, and the limit that bound.

**Layer 2, each word.** The labeller's checks on the flown track, from the row where the executor heard the
word: heading words to the end of the flight (§3.3); altitude and angle words in their tubes; speed words in their
spans.

**Layer 3, each flight.** A crossing of R is an approach crossing when G is false and the aircraft crosses the
threshold plane of R lined up: track within 30° of the course, lateral offset inside the landing screen (≤ 1,000 m and
≤ half the spacing to a parallel). The runway limit is the FAS half-width at the threshold (106.7 m), and not more than
half the spacing to a parallel. The judge gives the first outcome that occurs. At one row it uses the order of the table.

| Outcome | Condition |
|---|---|
| `dynamics_failure` | A state that is not finite, no airspeed, or the stall cut-off of the dynamics |
| `ground_contact` | Below the threshold elevation of R while before the threshold |
| `crossed_too_high` | An approach crossing higher than 100 m above the threshold |
| `crossed_off_runway` | An approach crossing at ≤ 100 m, outside the runway limit |
| `unstable_at_minimums` | An approach crossing at ≤ 100 m, inside the runway limit, after a failed DA check (or with no DA point: the aircraft crossed above the DA) |
| `landed` | An approach crossing at ≤ 100 m, inside the runway limit, after a DA check that passed |
| `crossed_other_runway` | G false: the threshold plane of another candidate crossed lined up, inside that runway's own limit, at any height |
| `timeout` | None of these within the time limit (the remaining observed time × 1.5, plus 900 s for each go-around) |

A crossing that is not lined up (for example, abeam the threshold on a downwind) is not an event. While G is true, no
crossing is an event, of R or of another candidate (D33): the flight continues.

**The DA check (D3, D38).** At the DA point the judge checks that the aircraft is stable:

- **Vertical:** the height within ±22 m of the published glidepath of R (straight-line reference, §11.4). The value is
  the vertical bound of the evaluation module (`evaluation/thresholds.py` `RNAV_TERMINAL_VERTICAL_BOUND_M`; ICAO Doc
  9613, RNP APCH Baro-VNAV final approach segment). The judge imports it; it is not a parameter.
- **Lateral:** the lateral offset inside the FAS cone at the distance of the DA point (FAA Order 8260.58D Formula 3-1-1,
  `flight_scenarios/fas_geometry.py`), the same geometry as the runway limit of the crossing. The lateral bound of the
  evaluation module (half the runway width, 15.24–22.86 m) is a landing-geometry criterion at the threshold (evaluation
  EV1). At the DA point, 0.9–2.1 km before the threshold, it would be narrower than the lateral tolerance of the
  closed-loop reading (30 m, D32): a sentence that follows a real flight could then fail the check.

The regulation basis: the missed approach starts at the DA (AIM 5-4-21 b: "Obstacle
protection for missed approach is predicated on the missed approach being initiated at the decision altitude"). If
the model says "go-around" before the DA point, the flight continues (§3.2).

**Quality** belongs to the evaluation module (lateral, vertical and speed gates). The judge does not repeat it.

---

## 6 Prior

### 6.1 Inputs

- **Source of the states (D32).** The rows before the first predicted step are observed (ADS-B), as in closed loop.
  From the first predicted step on, every input that comes from a state (the own state, the candidate vectors, the
  motion) comes from the flown states of the closed-loop reading (§4.9), not from the observed track. The words in
  force are those of the closed-loop sentence, corrections included. In closed loop the states come from the executor.
- **Own state (D5, D23).** The own state of the aircraft has no frame: its MSL height (the level words are MSL), its
  ground speed and its vertical rate.
- **Frame (D5, D23).** Every position and every direction is in the candidate vectors. Each candidate vector gives the
  aircraft in the frame of that candidate: the distance to its threshold along its course, the offset right of its
  final, the height above its threshold, the motion direction minus its course (sine, cosine)
  (`instructions.airport.relative_to_runway`). "The frame of R" is the candidate vector of R, which the prior gets as
  the runway in force. There is no second set of values relative to R.
- **No input is computed from R before R is said (D23).** The first predicted step says R. The inputs of the rows before
  it and of the first predicted step itself use no value computed from R. (The heads of a row see the runway that the
  same row said, §3.1; that is an output, not an input of the row.) The reason: the artefact writes the runway word at
  row 0 (§4.7), and its value is the runway on which the flight landed. An input of these rows computed from it gives
  the answer of the first predicted step, and in closed loop the same input does not exist. A test: a change of the
  runway word of a sentence leaves the inputs of rows 0 to `N_LOOK` the same, bit for bit.
- **No airport identity (D5, D24).** The prior has no airport embedding, no absolute position (E, N), no absolute
  motion direction, and no table of candidate constants (the absolute threshold position, the course, the elevation,
  the length).
- **Candidates (D24).** A candidate vector has only values that change with the aircraft: the values of the frame above,
  the glidepath height below, and the landings on the candidate in the 30 min before the step. It has no
  constant of its runway. Such a constant identifies the runway:
  - In the candidate table of `v6_20261002`, the pair (length, elevation) is different for 24 of the 25 candidates. The
    length alone tells the left runway of KRDU from the right one (05L / 23R 3,048 m, 05R / 23L 2,286 m).
  - The spacing of parallel runways is different at each airport: KRDU 1,068 m, KSJC 213 m, KSMF 1,827 m, KSTL 397 /
    855 / 1,251 m. A layout relative to R is a code of the airport when R is known.
  - The runway-intent study found this channel (R1.1, gradient-boosted trees, split by day; [plan](../history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md)
    §15): with a constant of each runway (`prior_share`), the trees recognised "30L" and were right on only 53 / 59 % of
    the flights on the two KSJC days with 30L closed. Without it (R1.1b) they were right on 96.8 / 97.6 %. For the prior,
    R46 shows that it uses an airport identity when it gets one (§11.1).

  The constants are also not necessary. The length is a cause only together with the aircraft type, and the prior has
  no aircraft type. The elevation is the MSL height minus the height above the threshold. Which of two parallel runways
  is the left one shows at each step in the offsets right of their finals (the value of the left runway is always
  larger). The relative positions of all candidates together still show the layout of the airport. That is real
  geometry, and the prior has it; only the held-out airports of D39 can measure how much the prior uses it. A constant comes back
  only as the variant `constants` of D39.
- **Words in force:** the runway in force as its candidate vector ("none yet" up to the first predicted step), the
  go-around state G, the heading in force as the sine and cosine of its angle relative to the course of R, the other
  columns as embeddings, and the time since each column said its word (D17).
- **Glidepath (D13, D23).** Each candidate vector has the height above the published glidepath of that candidate at
  the aircraft's distance before its threshold, with the straight-line reference of §11.4 (TCH + d·tan(angle) +
  d²/(2R_e), R_e the earth's radius of curvature along the course). The value in the vector of R is the input of D13.
  Thus the value exists at every row, also before R is said. Every candidate has a TCH and a glidepath angle (§4.2
  refuses a candidate without them). It is procedure geometry as an input; the model still decides the profile
  (principle 2). The value is only meaningful near the final; the model also has the offset from the final to weigh it.
- **Motion (D25).** The ground speed, the vertical rate and the motion direction (in each candidate vector, minus its
  course) come from the displacement in the 2 s before the row, at every row interval Δ (§4.8). Only past positions
  give them; never the fitted track, ground speed or vertical rate of the signals, which use 7.5 s of the future.
- **Time (D16).** No row position embedding and no input "time from row 0": both measure the time since the aircraft entered the 25 km slice, a cut of the data, and a long sentence (a
  go-around adds up to 900 s) reaches rows that training seldom saw. The causal time attention gives the order. RoPE in
  the time attention gives how far back each earlier row is: the rotation of a row's query and key uses its time in
  seconds, so the attention reads only time differences. Seconds, not rows, so that every row interval of D11 reads the
  same time. The time of a row is in seconds from the aircraft's own row 0. Only differences count, so the zero changes
  no result; but a UTC time (approximately 1.76e9 s) in float32 has a step of 128 s and loses the rows. RoPE works with
  the row-by-row cache of the speaker (`Prior.extend`): a key is rotated once, when it is written. The RoPE base is set
  at implementation.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force"
  (`since`): log(1 + t / 2 s) / 5, with t in seconds, counted from the first predicted step at the earliest. It is in
  seconds for every column, so that every row interval of D11 reads the same time. The runway column's value is, in the labelled data, the time since the first predicted step, because a
  labelled sentence says its runway only there, and again only at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or given again
  (a change of runway, or the runway word that ends a go-around), so it measures a real fact.

### 6.2 Outputs

Five heads in the order of §3.1. The runway head scores each candidate (a pointer) plus "unchanged" and
"go-around". The number of candidates is the airport's own: no slot count is fixed by the model (D41). The first
predicted step masks "unchanged" in each column and "go-around" in the runway column.

### 6.3 Training

Teacher forcing on the closed-loop sentences of the artefact, split by operating day (`data/day_split_20260924.json`; test days sealed). KAUS is
a held-out test airport only: it is read one time, at the end of the whole chain.

**Stop on the select days (D31).** The training keeps the epoch with the smallest per-step loss on the select days. The
training does not read the validation days: they are read one time for each stage, as in the post-training (§7).

**Hyperparameters (D40).** Configuration A, the start: d_model 192, 4 layers, 6 attention heads, feed-forward 768,
dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ with 500 warm-up steps, gradient clip 1.0, 16,384 aircraft-steps
in a batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is set at
implementation; a test makes sure that a shift of all times changes nothing.

**Cross-validation and the choice of the design (D39, D40).** The folds are the five airports: a fold trains on the
train days of four airports, stops on their select days and reads the select days of the fifth (the held-out airport).
The goal is a new airport, so the folds are airports, not days. All runs use the chosen Δ (§4.8).

| Step | Runs | Training runs |
|---|---|---|
| 1 | Variant `full`, four configurations, each on the 5 folds. A: the start. B: d_model 128, feed-forward 512. C: d_model 256, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. B, C and D have the layers and the learning rate of A | 20 |
| 2 | The seed scale: configuration A with a second seed, on the 5 folds | 5 |
| 3 | Variant `constants` with the configuration chosen in step 1, on the 5 folds | 5 |
| 4 | The base: the chosen configuration and variant on all five airports, stopped on their select days; the validation days read one time | 1 |

The rules, fixed before the runs:

- **The score** of a run is the mean, over the 5 folds, of the per-step loss on the held-out airport's select days.
- **The seed scale** is the absolute difference between the scores of configuration A at the two seeds.
- **Configuration:** among the configurations whose score is within twice the seed scale of the best, the one with the
  fewest parameters.
- **Variant:** `constants` only if its score is lower than that of `full` (same configuration) by more than twice the
  seed scale; else `full`.
- **Readouts of each fold, not used for the choice:** the runway word of the first predicted step at the held-out
  airport (top-1); free generation at the held-out airport (200 flights × 2 sentences): its outcomes (§5.8) and the words
  for each column.

The cost, Claude's estimate before the smoke of B3: approximately 50 min for one run on four fifths of the data, so
approximately 25 h of GPU for steps 1–3, and 1–2 h for the free generation of the folds.

### 6.4 Running at a new airport (D41)

The prior has no identity of an airport (D5, D23, D24). A new airport needs only data that the five airports also
come from:

| What the airport needs | Source |
|---|---|
| Candidate runways and their geometry: threshold, course, TCH, glidepath angle, LPV DA | FAA CIFP and the runway data (§4.2 refuses a candidate without them) |
| The procedure data for the procedure masks: the FAF, the LPV cone | FAA CIFP procedure details |
| The landings in the 30 min before the step | The airport's tracks (offline) or a live surveillance feed |
| The dynamics of each aircraft type | Independent of the airport |

The design makes sure of three more points:

1. **No fixed count of candidates.** The runway head points at the airport's candidates, whatever their number; no
   slot count comes from the training airports.
2. **Fixed physical scales.** Every input is scaled by a constant in SI units. No input uses a mean, a spread or another
   statistic of the training data or of an airport: a new airport has none.
3. **The check is a flight.** Each fold of D39 runs the full closed loop at its held-out airport (the prior speaks,
   the executor flies, the judge decides), as well as the teacher-forced loss. The final test on KAUS is the same.

**Known limit: the altitude grid.** The altitude words are MSL, and the 60 m steps reach to 1,260 m (§3.4). At an
airport whose elevation is above approximately 600 m, the levels of the final approach fall in the 120 m segment,
coarser than at the five training airports. The five airports and KAUS lie below 200 m, so this design does not meet
the limit.

---

## 7 Post-training (outline)

1. **One stage, from the base model, with one aircraft commanded (D29).** The closed loop is a window of 20 minutes of
   recorded traffic with one commanded aircraft. The model speaks for that aircraft; the other aircraft fly their
   records. The start model is the base model with a traffic attention whose
   output is zero (§8). There is no single-aircraft stage. The reasons:
   - The single-aircraft closed loop is a special case of this loop. With no other aircraft in its window, the
     commanded aircraft says and flies what single-aircraft free generation does, with the same loss and gradients.
     Approximately 45 % of the train flights have no leader in the air at their first predicted step (§11.8).
   - A single-aircraft reward does not see when an aircraft lands, so it does not limit the spread of the landing times.
     In traffic, that spread puts the aircraft into the sequence of the other aircraft (§11.8).
   - Without a capture law (D2, D3), the base model must learn more to land. It learns it one time, in the loop where
     it is used.
2. **Reward (D30).**

   | Outcome of the sentence (§5.8) | Reward |
   |---|---|
   | `landed`, no go-around, on a runway of the airport's present landing direction | 1 |
   | `landed` after n go-arounds, on such a runway | 0.9ⁿ |
   | Every other outcome; every loss of separation (the separation judge, §8) | 0 |

   "The airport's present landing direction": a runway within 90° of a runway with a landing in the 30 min before the
   first predicted step. The reward gives "a reward for a go-around that the DA check needs" without a term of its own.
   An unstable approach that continues ends as `unstable_at_minimums`: 0. A go-around before the DA point continues the
   flight with 900 s more time (§3.2); a landing then gives 0.9. Thus the model gains from a go-around only when the
   go-around changes a probable failure into a probable landing: with the same probability p of a landing before and
   after the go-around, 0.9·p < p. A go-around that prevents a loss of separation keeps the chance of 0.9, so the same
   reward also gives the go-around for traffic. The power n stops a chain of go-arounds that only adds time.

   **No payment for a go-around without a landing.** Such a payment f makes a go-around better than the continued
   approach when p < f / (f + 0.1), with the same p before and after the go-around (Claude's arithmetic). For example,
   f = 0.28 gives p < 0.74. "Go-around" is permitted at any row after the first predicted step (D10), and p is low on
   many flights when the training starts. The payment then teaches a go-around where the model is not sure, not where
   the approach is unstable. In this stage the other aircraft fly their records and do not react: a go-around cannot
   help them land. A stage with every aircraft of a window commanded (item 8) decides for itself if it pays for that
   help.
3. **Masks.** The vocabulary rules (§3.7); the procedure masks (glidepath lower edge inside the FAF, DA before the
   join, no climb back; D14 while G is true); the separation masks (§8, O6). A prior records the masks that it was
   trained under (`procedure_masks.json`, contract C35) and speaks under them.
4. **Windows.** Real windows and augmented windows of the train days: B (a moved start: a turn about the airport, a
   height change and a speed change), A (one inserted aircraft that flies its record), D (the aircraft ahead moved). The
   counts are set at stage C.
5. **Loss**: the clipped-ratio surrogate (ε = 0.2) with the advantage inside each branch group (item 9); the pull to
   the base model (0.04, the KL on the sampled words, masked distribution); the teacher-forced data term (1) on
   single-aircraft samples of the closed-loop sentences of the train days (D36); the traffic attention has its own
   learning rate. One pass over the sentences of a round.
6. **Go-around sampling first.** Before the training, measure the probability that the base model gives "go-around" on
   the final (with D26 the data has go-arounds with "no level-off" and "unspecified" in force). If the model says
   "go-around" by itself, the training uses no probes. If it does not, probes (a cross-entropy on a forced "go-around" at chosen steps)
   are discussed with the evidence of §11.6: a cross-entropy on a forced word teaches the word, not when to say it.
7. **Selection.** On the select days; the validation days are read one time for each stage. The airport
   generalization of the design is chosen in stage B (D39) and tested at the end on KAUS.
8. **Later, optional.** A stage with every aircraft of a window commanded starts from
   the model of item 7.
9. **Branch training (D37).** The samples of a round:
   1. Each training window (one commanded aircraft) is spoken one time: the first sentence.
   2. If the reward of the first sentence is 1, the window gives no sample: a group whose rewards are all the same
      gives no gradient.
   3. If the reward is less than 1, the event time t_E is the step where the first sentence ended: the first step of
      the loss of separation that ended it, the step of its judged outcome, or the time limit. The branch points are
      the first predicted step and every 120 s after it, before t_E.
   4. At each branch point, the window is spoken again K = 8 times from the state saved there. The saved state holds
      the executor state, the speaker's cache, the judge state and the window loop; the other aircraft fly their
      records, so their states come from the time. Each continuation has its own random numbers, from the seed, the
      round, the window, the branch point and k. A continuation with the random numbers of the first sentence repeats
      the first sentence, bit for bit (a test).
   5. A branch group is the first sentence and the K continuations of one branch point; they differ only after the
      branch point. The advantage is the reward minus the mean of the group. It applies only to the words after the
      branch point; a group whose rewards are all the same gives no sample.
   6. The executor flies without gradients while it speaks (inference mode); the states are the same as with gradients
      (a test). The training scores the prior only.

   **Why.**
   - Cost: with K full sentences for each window, the speaking took approximately 55 % of a round, and only 37–47 % of
     the sentences carried a gradient (§11.10). Branch training spends the extra sentences only where the first
     sentence failed, and a continuation flies only the part after its branch point.
   - Credit: a reward for a whole sentence gives every word the same advantage. In a branch group, only the words
     after the branch point differ, so the advantage goes to them (§11.8, §11.10).
   - Branch points: the readout of multi-aircraft step 7.7 found that a new sentence from the start rescues the most
     events, then from 120 s before the event; 60 s and less rescue few (§11.10). Its reading, decided before the
     run, puts the branch points at the start and at fixed times, not some tens of seconds before the event.
   - A window whose first sentence lands gives no sample in this round. Its chance to fail comes again in a later
     round, as the windows are drawn again.

---

## 8 Multi-aircraft (outline)

- **Time.** A scene puts its aircraft on one UTC grid: a scene step is one instant for every aircraft. That is the layout of the data, not a model input, and D16 does not change it. RoPE runs along each
  aircraft's own rows; the attention among the aircraft and the traffic attention read one scene step, which is one
  instant, and use no position. Edge features are relative (the approach clock is a distance; the closest-approach time
  is a time difference). The landing context counts the landings in the 30 min before the step's UTC time.
- R exists at every step for every aircraft from its first predicted step on. Before it, no input reads the aircraft's
  R (D23), the edge features included. Thus the relations of the edge features (the approach clock, the same runway,
  the parallel runways) and the separation judge have a runway at every step.
- **Separation judge** (`inference/separation.py`, reading VISUAL): the radar minima of 7110.65BB with the rules for
  visual approaches (7-4-4 c), never visual separation. Parallel runways at least 2,500 ft (762 m) apart are free once
  both aircraft are turned in (within 30° of the course, each on its own side of the midline); closer pairs count as one
  runway; established crossing finals are not judged.
- **Spacing on the final.** This design has no word "cleared", so no mask can hold a clearance back. A mask that keeps
  an aircraft from the final too close behind another is open (O6).
- **Speed-word mask.** It applies only when the aircraft and the aircraft next ahead on the approach clock (the same
  runway, or a pair that counts as one runway) are both established on their finals (O6), and while the aircraft is
  more than 9,260 m (5 NM) from its threshold (7110.65BB 5-7-1 b.4). Both aircraft are predicted to the time when the
  aircraft ahead crosses its threshold, each along its course toward its speed target at the rate of the executor
  (§5.6). A speed word whose predicted gap is then less than the required distance is masked. When every word falls
  short, nothing is masked.
- The traffic attention (initial output zero) on the base model is the start of the post-training (D29, §7).
- **Landing context (D31).** In the closed loop, the landing context of every aircraft of the scene, the replayed
  aircraft included, counts the landings that occur in the loop: the recorded landings of the replayed aircraft and of
  the aircraft before the window, and the landing of a commanded aircraft when it lands in the loop. The recorded
  landing of a commanded aircraft is never in it: that landing is the future of the commanded aircraft. Through the
  traffic attention, a commanded aircraft that flies later than its record would read a landing on its own landed
  runway.
- **States in a scene (D32, D36).** A scene occurs only in the closed loop. The commanded aircraft flies with the
  executor; the other aircraft fly their records. There is no teacher-forced scene sample: the flown states keep the
  observed path but not the observed time, so the spacing between two aircraft would not be the observed one.
- **D23 in a scene (D31).** At the rows up to the first predicted step of an aircraft, no input of any aircraft and no
  edge feature uses a value computed from that aircraft's R. A test: a change of one aircraft's runway word leaves all
  inputs and edge features of the scene at those rows the same, bit for bit.
- **Established on the final (D31, O6).** The separation judge (the in-trail rule: which aircraft is
  responsible) and the separation masks must know if an aircraft is established on the final of its R. This comes from
  a function of one row: the state of the aircraft at that row and its R. It is the same for every aircraft
  (commanded, labelled, replayed). It does not come from the capture row, which uses later rows (§2), and not from an
  executor state, because the executor has none (§5.8). It is not an executor law: only the separation judge and the
  masks read it. Its rule is open with O6.

---

## 9 Gates and identities

### 9.1 Gates

Each stage has a gate: the labeller (completeness, envelope containment with envelope width), the replay (the executor
flies the closed-loop sentences; §4.8, reading 4), the prior (teacher-forced likelihood against baselines, free
generation), the post-training, the multi-aircraft stages. The user sets the criteria of each gate after the design is
settled (D7). This document sets none.

### 9.2 Identities (D21)

**Rule** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the payloads mean)
and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a conformance check), not
by the bytes of its source. Data are identified by their flights, not by the bytes of a file they came from.

| # | What | Its identity | Stage |
|---|---|---|---|
| 1 | The vocabulary (grids, classes, tolerances, reading name) | The spec sha. It is the format | A |
| 2 | The labeller, open-loop and closed-loop reading | The labeller conformance: the artefact holds a reference sample (`conformance/`; train, seed 1337, 50 labelled and 10 refused flights for each airport, with their signals); `instruction_conformance` labels it again with the code on disk and requires the same words and the same refusals, on the 2 s rows and on the Δ grid at Δ = 4 and 8 s (D49). The closed-loop reading (D32) has its own reference sample (train, seed 1337, 10 flights for each airport, at each Δ of the artefact): with the artefact's executor spec, the same correction words and the same flown states on the 2 s rows (D51), within the tolerance of the executor conformance. Each check writes a passed record named by a digest of the code by its logic; a runner that labels, reads the closed loop or replays sentences requires the passed record of the code on disk. The artefact records which code wrote it, as information | A |
| 3 | The executor | The sha of the spec's parameters and the conformance of its reference tracks (C33): the spec's labelled train flights, flown again in every way the executor flies (single-aircraft batch, multi-aircraft batch, single flight) within 1e-6 m. The open-loop sentences of these flights are said on the spec's word clock (O9) | A |
| 4 | The flights of an artefact | The stored signals. A consumer that rebuilds a flight from the harvest compares it row by row with them (`autopilot/flights.py` `require_same_flight`). No byte hash of an arrival manifest | A |
| 5 | The artefact of a prior | The spec sha, the day split, the candidate table and the sha256 of the sentence files | B |
| 6 | The edge features of a scene | A conformance check: fixed reference scenes give the same edge features | C |
| 7 | The procedure masks of a prior | The set name, the checkpoint sha and the digests of the procedure data (C35). The procedure data are the format of the masks | B |
| 8 | A window readout | The code version and the conformance R44 | C |
| 9 | What the frontend reads | The reading name | D |
| 10 | The day split and the sealed test days (C32) | The day split file. It is a data rule | A |

---

## 10 Values

| Item | Value | Source |
|---|---|---|
| Row | Δ = 2 s, on even UTC seconds; the ablation also reads 4 and 8 s | D11, D25 |
| Observation of the prior | 16 s (8, 4, 2 rows at Δ = 2, 4, 8 s) | `N_LOOK`; D25 |
| Motion inputs of the prior | Displacement in the 2 s before the row, at every Δ | D25 |
| Executor cycle | 1 s | Fixed choice |
| Heading grid | 5°, relative to the course of R | Spec (grid); D8 (frame) |
| Heading lead L | 4 s | Measured (vocabulary design §10.1) |
| Heading tolerance | 4.5° | Half grid 2.5° + 2° |
| Turn-rate limit | 4.7°/s | Spec, measured (p99.9) |
| Bank limit | 32° | Spec, measured (p99.9) |
| Roll rate p | 5°/s | FAA Order 8260.3G Appendix E §4 ¶6.a ("roll-in rates of up to five degrees per second") |
| Level grid | 60 m to 1,260 m, 120 m to 2,700 m, 450 m to 5,400 m MSL; 40 levels | D22 (fit of the altitude-grid proposal) |
| Level envelope ε | half the larger gap to the neighbouring levels + 10 m: 40 m (0–1,200 m), 70 m (1,260–2,580 m), 235 m (2,700–5,400 m); "no level-off" 40 m | D22, D52 |
| Level detection | ≥ 20 s, rows within 25 m of the piece's median | Labeller constant (§4.4) |
| Descent classes | edges −0.5 / 2.0 / 2.75 / 3.75 / 10°; nominal 1.5 / 2.5 / 3.0 / 4.5° | Spec; k-means on the train days, weight length² (D54), rounded to 0.25° (D56) |
| Climb word | One class; a climb piece is 0.5–15° (labeller); executor angle while G is false: the nominal of D15, 1.5° | Spec; D15, D28, D56 |
| Go-around angle (executor, while G is true) | The steady climb angle at the thrust limit, within 1.885°–3° | AIM 5-4-21 b (minimum, 200 ft per NM); D28 (maximum) |
| Reward of a landing after n go-arounds | 0.9ⁿ | D30 |
| Speed grid, range, tolerance | 5 m/s, 20–250 m/s, ±5 m/s | Spec |
| Speed change rate (executor), speed word | a_max (the transition acceleration limit below) | D43 |
| Speed change rate (executor), "unspecified" | a_U = 0.25 m/s², or the rate that reaches the approach speed at the threshold when larger; not more than a_max | 5 m/s ÷ 20 s (user, 2026-09-24) |
| Transition acceleration limit | 1.4 m/s² | Spec, measured (p99.9) |
| Capture corridor (labeller only) | 20 m + d·tan 0.45°, course ±2° | Spec, measured (p99) |
| Landing screen | lateral ≤ 1,000 m and ≤ half the parallel spacing; height ≤ 100 m | `final_approach.assign.LandingScreen` |
| Landed lateral limit | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| DA check | vertical ±22 m of the published glidepath; lateral inside the FAS cone at the DA distance | D38 (`evaluation/thresholds.py`; FAA Order 8260.58D Formula 3-1-1) |
| Time limit | remaining observed time × 1.5, plus 900 s for each go-around | Fixed choice; the 900 s: the user, 2026-10-02 |
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; a correction ends below half the tolerance or at a change of sign; a change of sign beyond the tolerance says the opposite correction in the same row | D32, D53 |
| Closed-loop reading: correction | heading: one class (5°) toward the observed path; angle: the next descent class | D32 |
| Closed-loop reading: observed words | at the first Δ row whose matched observed time is less than Δ/2 before the word's 2 s time (the Δ row nearest to its place; a tie: the later row); a heading word in the frame where it is heard | D42, D45, D46 |
| Closed-loop reading past the end of the observed path | no correction (one in force ends); lateral error for the readouts, no vertical error | D44 |
| Turn rate of a heading word | the smallest of: the error over the time left until L (not less than two cycles), the stopping rate √(2·g·p·\|e\| / V), the turn-rate limit | §5.4; D47 |
| Δ grid of the open-loop reading | each 2 s word on the nearest Δ row (a tie: the later row) | D45 |
| Word clock of an open-loop replay | `track` (executor spec `v12_20261004`) | O9 |

---

## 11 Evidence

All counts in §11.2–§11.4 come from two runners on the train split of artefact `v6_20261002` (44,375 sentences),
2026-10-03: R48 `instruction_word_frames` (§11.2, §11.3; output
`4dTrajectory/outputs/POOLED/analyses/word_frames_20261003/word_frames.json`) and R49 `instruction_final_approach`
(§11.4; output `4dTrajectory/outputs/POOLED/analyses/final_approach_20261003/final_approach.json`). Each output gives
the same numbers for each airport too. `docs/reference/runners.md` R48, R49 give the definitions.

The smoke builds of §11.9 and §11.11–§11.14 took the first 80 sorted flight keys of each airport and split (a key starts
with the callsign: approximately 400 train flights, 373 of them one airline), and each fitted its spec on its own train
flights (the A9 smoke: descent nominals 1.51 / 2.36 / 3.08 / 4.36°, the climb 4.24° from one climb piece). Their numbers
show how the mechanisms act; they are not numbers of the formal artefact (D55).

### 11.1 The prior uses its airport embedding (R46)

[Readout](readouts/2026-10-03_airport_embedding.zh.md). On the validation days, a replacement of the airport embedding
with the mean of the four other embeddings increases the teacher-forced loss by 0.024 (base) and 0.028 (augmented) for
each step. It decreases the landed share in free generation by 5.8 and 8.0 points. Approximately 60 % of the increase is
in the heading column. The five embeddings are almost orthogonal: an identity table, not a property of the airports.
The candidate table (absolute threshold positions and courses) is a second identity: R46 §6.

### 11.2 Frames of the heading and level words

Mean pairwise Jensen–Shannon divergence between the word distributions of the five airports (bits; 0 is the same
distribution, 1 is no class in common):

| Words | Absolute | Relative to R |
|---|---|---|
| All heading words | 0.108 | 0.052 |
| Heading in force at the clearance row of `instruction-v3` (the intercept heading) | 0.619 | 0.195 |
| Level words (relative = height above the threshold) | 0.426 | 0.484 |

Share of each airport's MSL level words on levels that the other four airports use for less than 0.2 % of theirs: KSJC
27.8 %, KRDU 6.8 %, KSTL 4.8 %, KSMF 3.7 %, KMSY 0.4 %.

### 11.3 Courses of KSTL against the 5° grid

KSTL courses 62.7°, 122.3°, 242.7°, 302.3° (all eight candidates): the nearest absolute grid values are 2.26–2.32° away.
The capture corridor permits 2°. The courses of the other four airports are 0.01–1.07° from the grid (KSJC the
largest).

### 11.4 Clearance, final descent and glidepath in the labelled data

- 47.1 % of all sentence rows are before the clearance row of `instruction-v3`. 25.3 % of the flights are cleared at
  row 0.
- "Descend to land" comes before the clearance row in 49.9 % of the flights (median 96 s before, p90 332 s). It comes
  before the capture row in 59.0 %. Its distance before the threshold along the course is −0.9 / 15.7 / 24.0 km
  (p10 / p50 / p90; a negative distance is past the threshold plane, on a downwind).
- After the capture row and more than 300 m before the threshold, the observed height minus the published glidepath of
  the landed runway, against two references (R49). "Flat" is TCH + d·tan(angle), the formula that the executor and the
  procedure masks use; it leaves out the earth's curvature. "Straight line" is the published path as a straight line in
  space; over the curved earth it is d²/(2R) higher (approximately 15 m at 14 km, 31 m at 20 km).

  | Measure | Flat | Straight line |
  |---|---|---|
  | Rows: p10 / p50 / p90 | −9 / +6 / +31 m | −20 / +2 / +21 m |
  | Rows within ±30 m / ±60 m | 83.6 % / 92.5 % | 86.2 % / 92.8 % |
  | Flights with ≥ 90 % of their rows within ±60 m | 78.0 % | 78.9 % |
  | At the capture row (median 13.9 km before the threshold): more than 60 m above / below | 11.2 % / 21.0 % | 5.7 % / 24.9 % |

  The flat formula of the executor is a few metres to 31 m low far from the threshold. This matters only for a design
  that uses it as a reference.
- Published glidepaths: 3.0° at all candidates, 3.5° at KRDU 32. TCH 13.7–19.5 m.

### 11.5 Free generation and the heading column

- Base model, validation days: per-step loss 0.2648, of which heading 0.1214 (46 %) (R46, own embedding).
- Base model free generation (validation, 400 × 4 for each airport): landed 88.7 %, timeout 6.5 %, crossed too high 3.5 %
  (R46). The timeouts are mostly a flight that gets heading words and no clearance (stage notes §3.2).
- Labelled heading words for each flight: straight-in 3.0, vectored 33.7 (vocabulary design §10.1).

### 11.6 Go-around

- The 9.4 readouts: the start model gives the go-around word approximately 1e−11 at a probe. With a cross-entropy
  weight of 10–100 on the probes, the model says go-around, but 88 of its 108 own go-arounds end worse than round 0
  ([readout](readouts/2026-10-03_9_4_weights.zh.md)).
- R40 v2: 105 real go-arounds on the training days; from the go-around to the next turn onto the final, median 421 s,
  p95 598 s ([readout](readouts/2026-10-01_go_arounds.zh.md)).
- Words in force at the 70 real go-arounds inside the train sentences of `v5`, where each sentence is one
  approach to the landing (R40 v2 joined to the `v5` sentences; a one-off script, 2026-10-03): "descend to land" 0 of
  70; a level 70 of 70 (median 450 m MSL; the go-around heights above the threshold p10 / p50 / p90 90 / 236 / 481 m);
  "unspecified" 0 of 70; the capture row after the go-around row 70 of 70; a landing on another runway after the
  go-around 9 of 70 (D26).
- The row of the climb word in `v5`, relative to the lowest point of the go-around (69 of the 70 have a climb word):
  1–3 rows before 7; the same row 6; 1–4 rows after 45; 5–23 rows after 11. D26 puts "go-around" in the row of the
  climb word.
- The climb after a real go-around: median gradient 9.3 % (5.3°); 96 % are at least 200 ft per NM (R40 v2, readout §7).

---

### 11.7 The altitude grid

From the proposal [altitude_word_grid.zh.md](altitude_word_grid.zh.md) (`v6_20261002`; fit on 11,937 train flights,
read on 3,964 validation flights; continuous level-off heights). Rounding error of the level words other than row 0,
p50 / p95 / largest: the chosen grid (40 levels) 17 / 30 / 58 m; the same shape with steps doubling and break points at
1,200 and 2,400 m (44 levels) 17 / 38 / 79 m; a uniform grid of 48 levels over the range 35 / 55 / 57 m. The
`instruction-v3` grid (30 m, 182 classes) has at most 15 m. These numbers come from a one-off script of that proposal.
The spec measurement gives them from a runner (`instruction_spec`, `level_rounding`): on all train days with the labeller
of stage A (code at `42f3ff62`, 25,654 level words), 16 / 34 / 223 m; the largest is in the 450 m segment.

- In `v6_20261002`, 99.7 % of the level words are under 3,000 m; its 30 m grid used 116 of its 182 classes.
- The geometric height of one assigned pressure level moves from day to day by about height × ΔT / 273 (ΔT: the day's
  temperature deviation). At KRDU this is about 15 m at 600–900 m and about 90 m at 2,000–3,000 m (vocabulary design
  §2.4).

### 11.8 The post-training in traffic

From the multi-aircraft readouts of `instruction-v3` (`readouts/2026-09-28_m3_free_generation.zh.md`,
`readouts/2026-09-30_m4_passes.zh.md`; multi-aircraft design §6.6 step 9.4):

- The start model of the multi-aircraft work (augmented: landing and augmented starts, single-aircraft rewards) with
  one aircraft commanded, on the select days: it landed between 111 s earlier and 121 s later than its record (p10,
  p90); the labelled words, −36 s and +21 s. A loss of separation with the recorded traffic ended 11.5 % of its
  sentences; 12.3 % when it did not see the traffic. Most losses were on vectored approaches (21.6 %; straight-in 4.0 %).
- The selected round (5 of 8) of the multi-aircraft training, validation days: 1.75 points fewer losses of separation
  than the start model (11.1 % → 9.3 %); the labelled words 4.4 %, the records 2.9 %.
- No readout has the base model in this loop. The numbers show that single-aircraft rewards do not limit the spread of
  the landing times; they do not show that those rewards make it.
- In the window loop with one commanded aircraft and no other aircraft, the words, the flown states, the loss and the
  gradients are the same as in single-aircraft free generation and training (tests of step 9.4). On the train days,
  approximately 45 % of the flights have no leader in the air at their first predicted step.

### 11.9 Open-loop words on the smoke artefact

From the smoke build of stage A (`dev-two-tier-v4`, 80 flights for each airport and split; 1,189 sentences; the replay
at Δ = 2 s, 150 train flights on their own dynamics). One-off scripts, 2026-10-03; information, not a criterion (D7).

- Replay outcomes: landed 49, `crossed_off_runway` 36, `ground_contact` 23, `unstable_at_minimums` 16,
  `crossed_too_high` 15, `timeout` 6, `crossed_other_runway` 5. Lateral offset at the crossing, median / p90:
  straight-in 54 / 155 m, vectored 143 / 265 m.
- The last heading word is class 0 in 1,189 of 1,189 sentences. Where it takes effect (median 22.7 km before the
  threshold straight-in, 11.2 km vectored), the observed offset is 28 m and 30 m (median); the observed track after it
  stays within 2.1° of the course (median of the largest deviation, vectored); the observed offset at the last row is
  2 m (median).
- Where that word takes effect, the flown offset of the vectored flights is 157 m (median; p90 378 m) against 30 m
  observed; at the end, 155 m. The error is made before the final.
- An exact model of the heading words (each word's track exactly, reached 4 s after the word, at the observed ground
  speed, from the observed position at row 0), offset at the last row:

  | Heading grid | Straight-in: median / p90 | Vectored: median / p90 | Vectored: more than 106.7 m |
  |---|---|---|---|
  | Observed track | 2 / 5 m | 2 / 4 m | 0 % |
  | 5° | 34 / 82 m | 99 / 382 m | 47 % |
  | 2.5° | 31 / 63 m | 74 / 196 m | 31 % |
  | 1° | 40 / 72 m | 70 / 118 m | 15 % |

- The final descent ("no level-off" in force, median 20 km) flown at the nominal angles of its labelled classes, from
  the observed height where it starts: the height at the last row minus the observed height, p10 / p50 / p90:
  −45 / −4 / +53 m; 22 % more than 30 m high, 19 % more than 30 m low.

### 11.10 Cost and credit of the post-training

- Multi-aircraft step 9.4, one aircraft commanded, 700 windows × K = 8 = 5,600 sentences each round, four speaking
  processes (`outputs/POOLED/prior/step9_4_traffic_20261003.run/run.log`):

  | Part of a round | Round 1 | Round 2 |
  |---|---|---|
  | Speaking the 5,600 sentences | 2,050 s | 2,157 s |
  | Training (3 passes) | 891 s | 1,051 s |
  | Selection readout | 694 s | 748 s |
  | Sentences that carried a gradient | 2,063 (37 %) | 2,609 (47 %) |

- Multi-aircraft step 7.7 (`readouts/2026-10-02_window_rewind.zh.md` §2.1; real windows of the select days, the
  responsible aircraft spoken again 8 times): events with at least one rescue in 8, and the mean rescue of one
  sentence: from the start 69.6 % and 32.7 %; 120 s before the event 41.7 % and 17.4 %; 60 s 25.4 % and 9.1 %;
  30 s 13.0 % and 4.9 %; 10 s 3.1 % and 1.1 %. Each branch flew the whole window again from its start: 13.5 h for the
  readout.
- Three passes on the same sentences against one (multi-aircraft M4, second run against the first): the KL to the base
  model grew 1.4 times faster (0.0137 against 0.0098 for each round), and the rewards of rounds 1–7 did not differ
  from the first run's, sentence by sentence (`readouts/2026-09-30_m4_passes.zh.md`).

### 11.11 Closed-loop sentences with the observed words at the observed time

From the smoke build of A9 (`dev-two-tier-v4` at `751d0ec8`, `smoke_v4/data/`): closed-loop sentences at Δ = 2 s, each
observed word said at its observed time, replayed; the select split, the 395 flights on their own dynamics.
One-off scripts, 2026-10-04; information, not a criterion (D7).

- Landed: straight-in 222 of 229 (96.9 %), vectored 122 of 166 (73.5 %). The vectored failures: `timeout` 17,
  `crossed_off_runway` 13, `unstable_at_minimums` 6, `crossed_too_high` 5, `crossed_other_runway` 3.
- 41 of the 44 vectored failures leave the observed path laterally by more than 300 m (largest offset: median 1.55 km).
  At the row before, the flown aircraft is ahead of or behind the observed aircraft of the same time, along the
  observed track, by 1.3 km (median; p90 3.6 km; more than 500 m in 88 %).
- 76 of the 122 landed vectored flights also leave the path by more than 300 m (along the track before it: median
  0.6 km); they come back before the threshold. Heading correction words: 12.4 for each sentence (train, Δ = 2 s).
- The straight-in flights are also ahead or behind (median of the largest difference 0.7 km), but they do not turn:
  13 of the 222 landed straight-in flights leave the path by more than 300 m.
- KMSY `AAL2354_11_a9942e_20260826T153443Z` hears 70 m/s at 62 s. The observed aircraft slows at 0.36 m/s², the
  executor at 0.25 m/s². At the start of a turn of approximately 100° the flown aircraft is 1.8 km ahead; after the
  turn it is 2.5 km to the side (`timeout`).
- KSTL `AAL1010_30R_aae316_20260722T024358Z` hears 105 m/s at the first predicted step. The observed aircraft takes
  approximately 290 s from 146 m/s to 105 m/s, the executor approximately 160 s. The flown aircraft is 4 km behind; after
  the turns it is 11 km to the side (`timeout`).
- One correction class (5°) closes 6–9 m/s at a ground speed of 70–100 m/s: 2 km in 4–5 minutes. A new observed heading
  word ends a correction (§4.9).
- The 8 vertical failures of the DA check (select, 22–54 m above the glidepath at the DA point): in 3, the flown path is
  within 3 m of the observed path and the observed path is 19–23 m above the glidepath; in 3, the flown path is 9–16 m
  above the observed path and the observed path is 12–15 m above the glidepath; in 1, the flown path is 44 m above the
  observed path; in 1, the flown path is 850 m to the side of the observed path.

### 11.12 Speed words and the rate of a change of speed

From the smoke build of A9 (`dev-two-tier-v4` at `751d0ec8`, `smoke_v4/data/`), train and select. One-off scripts,
2026-10-04; information, not a criterion (D7).

- The observed rate of a deceleration (20 s windows that slow down; ground speed, 10 s mean), p25 / p50 / p75: level
  flight (|vertical rate| < 0.5 m/s, 844 windows) 0.12 / 0.24 / 0.40 m/s²; descent (vertical rate below −3 m/s, 3,540
  windows) 0.11 / 0.19 / 0.31 m/s².
- A one-dimensional estimate: 634 flights with at least 30 rows of 2 s before "unspecified"; from the first predicted
  step, the speed goes toward the word in force at a constant rate, the words at the observed time, no dynamics. The
  "steps" are the grid value nearest to the observed speed at each row. The largest difference along the path from the
  observed aircraft, for each flight:

  | Words | Rate | p50 | p90 | More than 1 km |
  |---|---|---|---|---|
  | One word for each run (its target) | 0.25 m/s² | 499 m | 2,235 m | 31 % |
  | One word for each run (its target) | 0.5 m/s² | 537 m | 2,359 m | 32 % |
  | One word for each run (its target) | 1.5 m/s² | 1,276 m | 4,527 m | 57 % |
  | Steps | 0.25 m/s² | 743 m | 2,251 m | 39 % |
  | Steps | 0.5 m/s² | 255 m | 563 m | 1 % |
  | Steps | 1.0 m/s² | 148 m | 297 m | 0 % |
  | Steps | 1.5 m/s² | 120 m | 243 m | 0 % |

- Speed words before "unspecified", for each flight (mean): 2.2 for one word for each run, 11.1 for the steps.
- With a thrust floor of 0 N, 40 % of the inverted teacher segments of the KSJC outer-train cohort needed less thrust
  than 0 N: the real aircraft slowed down more than the clean polar permits (`outputs/envelope.py`).

### 11.13 Closed-loop sentences with the words at the place and the speed words in steps

From the smoke build of A10 and A11 (`dev-two-tier-v4` at `0c0778c8`, `smoke_v4/data/a11/`, the spec of the A9 smoke;
words said at the first Δ row after their place, speed words in steps). Readouts of the stage A report
(`readouts/2026-10-04_stage_a_a10_a11_report.zh.md`) and one-off scripts, 2026-10-04; information, not a criterion (D7).

- Landed in the replay of the closed-loop sentences, select, Δ = 2 s: straight-in 225 of 231, vectored 157 of 166
  (94.6 %). Vectored flights that leave the observed path by more than 300 m: 4 (§11.11: 117).
- The 14 failures of select, Δ = 2 s (9 vectored, 5 straight-in), are DA checks that fail vertically: 13 between 22 m
  and 43 m above the glidepath, 1 23 m below it; 9 at KMSY.
- Heading correction words for each sentence, train, Δ = 2 s: straight-in 4.9, vectored 25.7.
- The largest difference along the path from the observed aircraft of the same time, before "unspecified", closed
  loop, select, Δ = 2 s, vectored: p50 221 m, p90 348 m; more than 1 km 1 %.
- Δ = 4 s, train: 35 of 40 vectored flights leave the observed path by more than 300 m (Δ = 2 s: 2 of 40); 39 of 40
  land.
- The flight past the end of the observed path, train and select: p50 149 m, p90 393 m, at most 754 m. Train, Δ = 2 s:
  82 of the 2,621 angle corrections and 6 heading corrections were said there.
- A synthetic flight at the observed speed, flown open loop: 75–180 m of lateral offset after each turn (a test of A11
  holds 93 m after the second turn).
- The courses of parallel runways differ by 0.002–0.01°.

### 11.14 The nearest row and the turn of the executor

From the smoke build of A12 and A13 (`dev-two-tier-v4` at `fd235e19` and `b8994f58`, `smoke_v4/data/a12/`, the spec of
the A9 smoke; the closed loop with a tie at the earlier row, before A14). Readouts of the stage A report
(`readouts/2026-10-04_stage_a_a12_a13_report.zh.md`); information, not a criterion (D7).

- Closed-loop sentences replayed, train, vectored: more than 300 m from the observed path at Δ = 4 s 0 of 40 (§11.13: 35
  of 40), at Δ = 8 s 2 of 40; all 40 land at each Δ. Select, Δ = 2 s, vectored: 159 of 166 land.
- The lateness of the observed heading words said in closed loop (the matched observed time of the row that says a word
  minus the word's 2 s time), train, mean: +0.03 s at Δ = 2 s, −0.3 s at 4 s, −1.6 s at 8 s; inside ±Δ/2. In a turn
  several words fall in one row and only the last, the earliest in time, is said: thus the words said are early on
  average at Δ = 8 s.
- The open-loop sentences at Δ = 4 s do not change: on a 4 s grid, the nearest row with a tie at the later row is the
  next row.
- The turn of the executor (A13, `experiments/executor_turns.py`), select, 837 turns measured in every way, the turn's
  own part toward the outside, mean: the exact words −36 m, the executor −18 m, the executor without the stopping-rate
  limit −21 m, the observed track integrated (the control) −2 m. Without the 292 turns whose next turn starts within
  30 s of the end: −34, −16, −18, +1 m. Larger at higher speed: at 100 m/s or more the words −39 m, the executor −18 m.
  The change of e_y across a turn: p50 58 m, p90 390 m (the executor); it holds the offsets of earlier turns.

## 12 Regulation sources

Checked on 2026-10-03 in the local PDF files (`docs/literature/runway_assignment/official/`, repository root; not in git):

| Paragraph | Text used |
|---|---|
| Pilot/Controller Glossary, CLEARED (Type of) APPROACH | "ATC authorization for an aircraft to execute a specific instrument approach procedure to an airport" |
| AIM 5-5-4 a.3 | When radar vectored: "Maintains the last assigned altitude until established on a segment of a published route or IAP, at which time published altitudes apply" |
| AIM 5-5-4 b.1, b.2 | The controller "Issues an approach clearance based on known traffic"; "only after the aircraft is established ... or assigns an appropriate altitude ... until so established" |
| AIM 5-4-21 b | Missed approach at the DA; climb gradient of at least 200 feet per nautical mile (1.885°) |
| 7110.65BB 4-7-10 a | Approach information at the first contact: the approach to expect and the runway |
| 7110.65BB 5-9-1 a, c | Intercept the final at least 2 miles (3,704 m) outside the approach gate; for a precision approach "at an altitude not above the glideslope/glidepath" |
| 7110.65BB 5-9-2, TBL 5-9-1 | Intercept angle: 30°, or 20° less than 2 miles (3,704 m) from the approach gate |
| 7110.65BB 5-9-4 | Arrival instructions before the approach gate; example 2: "Turn right heading three four zero. Maintain two thousand until established on the localizer. Cleared I-L-S runway three six approach." |
| 7110.65BB 5-7-1 b.4, d | No speed assignment inside 5 NM (9,260 m) or the FAF; an approach clearance cancels assigned speeds (vocabulary design §9) |
| FAA Order 8260.58D Formula 3-1-1 | FAS course half-width ≥ 350 ft (106.7 m) at the threshold (cited in `flight_scenarios/fas_geometry.py`; not checked again on 2026-10-03) |

---

## 13 Key code index

The current code that this design changes. Line numbers are at `dev-two-tier` `26e96c5e`.

| What | Where |
|---|---|
| Columns; the approach classes | `instructions/words.py:16`, `:20`–`:23` |
| Absolute heading classes | `instructions/words.py:56` `heading_index`, `:59` `heading_deg` |
| "Descend to land" value | `instructions/words.py:37` |
| Approach transitions (decode rule) | `instructions/grammar.py:30` `APPROACH_NEXT`, `:53` `approach_words_allowed` |
| Row rules (altitude and angle) | `instructions/grammar.py:36` `step_allowed` |
| Capture row | `instructions/labeller/lateral.py:48` `capture_row` |
| Heading words cut at the clearance | `instructions/labeller/lateral.py:85` `per_step_words`, `:130` `read_lateral` |
| Clearance at the turn onset | `instructions/labeller/lateral.py:116` `capture_turn_onset` |
| "Unspecified" anchored to the clearance | `instructions/labeller/speed.py:83` `read_speed` (`join_row`) |
| Heading envelope | `instructions/envelope.py:19` `heading_word_rows`, `:31` `heading_words_inside` |
| Convergence and corridor | `instructions/envelope.py:78` `heading_converges`, `:96` `corridor` |
| Tube checks | `instructions/labeller/vertical.py:169` `tube_bounds`, `:200` `tube_checks` |
| Landing screen | `instructions/labeller/read.py:61` `landing_passages` |
| Heading law | `autopilot/lateral.py:99` `word_rate` |
| Capture start, wait, bend, lock | `autopilot/lateral.py:262` `Lateral.rate` (`:284` capture start, `:293` wait) |
| Vertical modes; glidepath floor; go-around climb | `autopilot/vertical.py:140` `Vertical.rate`, `:74` `GLIDEPATH_BELOW_M`, `:81` `GO_AROUND_CLIMB_GRADIENT` |
| Runway lock | `autopilot/executor.py:149` `runway_locked` |
| TCH and glidepath read by the executor | `autopilot/runway_data.py:26` `VerticalPath`, `:51` `published_vertical_paths` |
| Outcomes | `autopilot/judge.py:82` `OUTCOMES`, `:126` `runway_lateral_limit_m`, `:144` `_outcome` |
| FAS cone | `flight_scenarios/fas_geometry.py:46` `fas_course_geometry` (repository root) |
| DA above the threshold | `trajectory_data_process/harvest/airports.py:169` (repository root) |
| Prior: airport embedding, position embedding, candidate sum | `prior/model.py:312`, `:313`, `:382`, `:385` |
| Prior: step, relative and candidate features | `prior/data.py:63`, `:64`, `:70` |
| Clearance mask | `inference/separation_masks.py:131` `clearance_check` |
| Edge features | `inference/scene_edges.py:51` `EDGE_FEATURES` |

---

## 14 Implementation plan

This section tells the implementing agent what to build. The sections above give the design; this one gives the work.
Read the design sections that a milestone names before you start it.

### 14.1 Branch and rules

1. The branch is `dev-two-tier-v4`, in the worktree `.claude/worktrees/two-tier-v4`, made from `dev-two-tier`. Before a
   milestone that a new design commit holds, merge `dev-two-tier` into it. The ignored data trees of the worktree
   (`data`, `trajectory_data_process/outputs`, `4dTrajectory/outputs`, `aeroviz-4d/public/data/airports`) are absolute
   links to LIVE data.
2. Each milestone: read the code that it changes; write the code and its tests; run the milestone's test files; get a
   code review from a separate reviewer (code only, never documents); correct; commit with explicit paths. Never use
   `git add -A`. Before each commit, read `git diff --cached --stat`.
3. A test never writes under a live root. A test that calls a runner's `main()` gives every write root a tmp path
   (`tests/support.py` `labelled_instruction_artefact` is an example).
4. Run single test files in the foreground. Run the full ts suite (approximately 55 min, detached: `nohup setsid`, the
   script writes its own PID file) at the end of each milestone that changes code, before its report, and before a
   formal build.
5. No compatibility (principle 8). Every changed format gets a new name (the reading name, each schema name; the
   names are constants of the code). The new code refuses an old artefact by its name. No `.get(key, default)`
   fallbacks, no branches on a schema version.
6. SI units only. A value from a regulation has its paragraph in a comment and is defined one time.
7. A formal build or a formal readout only where the user orders it (A18). A smoke build (a stated limit for each
   airport) goes to a tmp or scratch directory. It uses the spec that the user chose (D15), and its flights are a random
   sample for each airport and split, seed 1337 (D55). The replay of the val days and every criterion wait for the user
   (D7). Never write into an existing directory under `4dTrajectory/outputs/`.
8. Do not touch the checkouts of running experiments (`.claude/worktrees/step9-run` and others) or the main checkout.
9. A defect that you find outside the milestone goes to `docs/code-health-followups.md` (an entry and a table row), not
   into the change.
10. At each milestone, update §0.3 (state, commit). At the end of stage A, update §13 (key code index) to the new code.
11. The branch is not merged before stage D: the backend's live executor (`aeroviz_backend/autopilot_segment/`) and the
    frontend's Training view read the old format until then. The user merges.
12. A value that a runner fits from data and the user chooses (D15) is measured on all train days in a scratch
    directory, directly after the milestone that writes the runner. This measurement is not a formal build: it writes
    nothing under `4dTrajectory/outputs/`. Report the fitted values and their candidates to the user, and wait for the
    choice before a later milestone reads the value (D55).

### 14.2 Stage A: vocabulary, labeller, identities, executor, judge, replay

A0–A17 are done (§0.3 gives the commits; the reports are in `readouts/`). The rules are in the design sections above;
the table gives what each milestone built. The module and the test that carry each decision of A0–A14 are in the table
of Claude's check (`readouts/2026-10-04_stage_a_check.zh.md` §2), those of A15–A17 in their reports.

| Milestone | What it built | Decisions |
|---|---|---|
| A0 | The archive of the modules of the old vocabulary that stage A does not rewrite (`archive/two_tier_v3_2026_10/`, with a `README.md`: what moved, why, which stage brings each part back; the backend tests that use the two-tier code fail until stage D) | D20 |
| A1 | The vocabulary: five columns, the runway/G table, heading classes relative to the course of R, the altitude grid, grammar rules 1–5 as one function for the labeller and the speaker | D1, D8, D10, D12, D14, D22 |
| A2 | The labeller: heading words from row 0 to the end, the go-around reading, the level test without the grid, "unspecified" from the capture row, the Δ grid; candidates without TCH, glidepath angle or LPV DA refused | D4, D11, D18, D19, D25 |
| A3 | The artefact and the identities: the labeller conformance (§9.2 #2), no byte checks of the arrival manifests, the sentences schema, the spec measurement with its rounding candidates and the level rounding errors | D15, D21 |
| A4 | The executor without the laws of §5.7; heading words converted with the course of R when heard; the go-around; the words heard at each Δ row; the same in `autopilot/single.py` | D2, D3, D9, D10, D27 |
| A5 | The judge: the outcomes of §5.8 in their order, no event while G is true, the time limit + 900 s for each go-around, the DA check | D3, D33 |
| A6 | The executor spec (`ts-executor-spec-v7`) and its conformance; `executor_replay --row-interval-s` | D21 |
| A7 | The close of the first part: the full suite, the report `readouts/2026-10-03_stage_a_report.zh.md` | — |
| A8 | Each approach read separately; rule 6; "go-around" changes no target; the go-around angle | D26–D28 |
| A9 | The closed-loop reading (`autopilot/closed_loop.py`, runner `instruction_closed_loop`) and its conformance; the replay of closed-loop sentences; the DA check with the bounds of D38; the readings of D34 | D32–D34, D38 |
| A10 | The observed words said at the place; the flight to the end of the executor or the time limit | D42 |
| A11 | Speed words in steps; the executor at a_max | D43 |
| A12 | Each word on the nearest Δ row; a heading word in the frame where it is heard; no correction past the end of the observed path | D44–D46 |
| A13 | The turn measurement (`experiments/executor_turns.py`, R51); the user kept the turn law | D47 |
| A14 | A tie at the later row in the closed loop too | D45 |
| A15 | A heading word of the class in force after a change of R; the flown states on the 2 s rows; the Δ grid in the labeller conformance; the architecture test that the executor's laws do not import the runway data | D48, D49, D51 |
| A16 | An overshoot corrected in its row | D53 |
| A17 | The smoke sample random for each airport and split | D55 |

**A18. The chosen spec and the formal artefact (D56).** After A17.

- `experiments/instruction_spec.py`: `--candidate NAME` (required when the runner measures): the spec takes the descent
  nominals and edges and the climb nominal of that row of `rounding_candidates` (`fitted`, `0.5`, `0.25`, `0.1`);
  `measurements.json` records the chosen row beside the fitted values. Tests: the spec holds the chosen row; a name not
  in the rows is refused.
- The formal artefact (the user, 2026-10-04: "完成正式数据的读数"), from a clean checkout, into new directories:
  `4dTrajectory/outputs/POOLED/instruction_language/v7_20261004/` (signals of every development flight, the spec with
  `--candidate 0.25`, labels, the labeller conformance, the closed-loop reading at Δ = 2, 4, 8 s on every split) and
  `4dTrajectory/outputs/POOLED/executor/v12_20261004/` (the executor spec, word clock `track` as in the smokes, its
  conformance).
- The readings of D34 at Δ = 2, 4, 8 s: the closed-loop summary of every split (items 1–3) and the closed-loop replays
  (item 4) of train (400 flights an airport, seed 1337) and select (every labelled flight). The val replay waits for the
  user (as for every executor before). The data are made read-only with a `SHA256SUMS` beside them. The report gives
  the readings side by side; it sets no criterion (D7).

### 14.3 Stage B: prior

**Start.** After Claude's check of stage A (§14.6). The code is written and tested on a sample of the formal artefact
`v7_20261004` at Δ = 2 s, read-only. The smoke artefacts of stage A are not used: their spec is not the spec of D56
(D55). The formal runs of B5 need the chosen Δ (§0.4 item 3). The rules of §14.1 apply.

**Written from §6, not patched from the archive.** The archived `prior/` (`archive/two_tier_v3_2026_10/prior/`) stays
unchanged. A part of it comes back only where its logic fits §6, rewritten into the new modules: the row-by-row encoding
with a cache (`Prior.extend`), the candidate tokens with one shared network and the pointer head, the ordered heads,
the training loop (AdamW, warm-up, clipping, early stopping), the index of the airport's landings, the rules of the
procedure masks. Nothing else comes back: not the six-column layout, the airport and row-position embeddings, the
candidate constant table, the fixed count of candidate slots, the variants and selection rule of `instruction-v3`,
the aircraft attention of a one-aircraft scene, the checkpoint formats `v3`–`v5`. No compatibility (principle 8).

**B0. Package and layout.**

- A new package `prior/`. It reads `instructions/` and the artefact files; it does not import `autopilot/`; only the
  runners use it (two-tier framework §2). These rules go into `tests/test_architecture.py` (the prior's rules cut into
  `tests/test_mirrors_cut_from_live_tests.py` come back, rewritten).
- The runners of B: `prior_train` (one run: all airports, or one fold with a held-out airport), `prior_select` (reads
  the campaign of B5 and writes the choice), `prior_free_generation` (the prior speaks, the executor flies). New code;
  the archived runners of the same names stay as they are.

**B1. Data** (§6.1; D13, D17, D23–D25, D32, D41).

- The rows: before the first predicted step, the observed states; from it on, the flown states of the closed-loop
  sentences, on the rows of the chosen Δ.
- The own state: MSL height, ground speed, vertical rate; the motion from the displacement in the 2 s before the row.
- One vector for each candidate: the distance before its threshold along its course, the offset right of its final,
  the height above its threshold, the motion direction minus its course (sine, cosine), the height above its glidepath
  (straight-line reference), the landings on it in the 30 min before the step. No constant of the runway, except in the
  variant `constants` of D39 (length, threshold elevation).
- The words in force: the runway as the candidate vector of R ("none yet" up to the first predicted step); G; the
  heading as the sine and cosine of its angle relative to the course of R; the other columns as embeddings; the time
  since each column said its word, in seconds.
- The targets: the five columns of the closed-loop sentence.
- Every scale is a constant in SI units (D41). A flight without a training sentence (§4.9) is not read.
- Tests: a change of the runway word leaves the inputs of the rows up to the first predicted step the same, bit for bit
  (D23); the glidepath height against a hand computation; the motion from the 2 s displacement at Δ = 2, 4, 8 s; the
  flown states from the first predicted step on; a permutation of the candidates permutes their vectors and nothing
  else; an airport with more candidates than any training airport is read.

**B2. Model** (§6.1, §6.2; D16, D41).

- The time attention with RoPE on the seconds from the aircraft's row 0; no row-position and no airport embedding.
- The candidate tokens through one shared network; five heads in the order of §3.1; the runway head points at the
  candidates, plus "unchanged" and "go-around"; the first predicted step masks "unchanged" and "go-around".
- Each layer has a place where stage C adds the traffic attention with a zero output (§8).
- The checkpoint: a new format name; its identity as §9.2 #5 and #7.
- Tests: the row-by-row encoding gives what the encoding of the whole sentence gives; a shift of all times changes no
  output; a permutation of the candidates permutes the runway scores; any number of candidates.

**B3. Training** (§6.3; D31, D40).

- Teacher forcing; the airports of a run (all, or a fold without its held-out airport); the stop on the select days;
  the validation days not read.
- Before a formal run: the check at the formal size of the host memory and the GPU memory of the largest batch.
- A smoke run on a sample of the formal artefact; it gives the time of one run for the cost of B5.
- Tests: the stop reads only the select days; a fold never reads its held-out airport in training.

**B4. Speaking and free generation** (§3.7, §5.8; D14, D22, D33, D38).

- The masks: grammar rules 1–6 and the runway/G table; the procedure masks (the glidepath lower edge inside the FAF,
  the DA before the join, no climb back; lifted while G is true as D14 says), the tolerance of a word half the step of
  its segment (D22).
- The closed loop: the prior speaks, the executor flies, the judge decides (D33, D38); 900 s more time at each
  go-around; the observed rows before the first predicted step, the executor's states after it.
- The readout: the outcomes for each airport and each kind of approach; the words for each column against the
  labelled ones; the go-arounds said; the probability of "go-around" on the final (§7 item 6).
- Tests: one flight spoken and flown to its outcome; each mask; G; the time limit; the same seed gives the same
  sentence.

**B5. Cross-validation and the base** (D39, D40, D41).

- The 31 training runs of §6.3, as one campaign from one commit on a clean checkout, one at a time on the GPU.
- For each fold: the held-out loss, the first-step runway at the held-out airport, the free generation at the held-out
  airport (200 flights × 2).
- `prior_select` applies the rules of §6.3 and writes the choice. Then the base on all five airports, and its one
  validation readout: the teacher-forced loss, the free generation, the share of the labelled words that the masks
  block, the probability of "go-around" on the final.
- No criterion is applied: the user reads the results (D7).

**B6. Close of stage B.** The full ts suite passes (run detached). §0.3, §13 and `docs/reference/runners.md` are
updated. Report to the user: the commits, the readings of each fold and of the base, the choice and its rule, and what
stage C needs.

### 14.4 Stage C: post-training and multi-aircraft (outline)

One post-training stage from the base model with one aircraft commanded (D29, §7): the window loop of the multi-aircraft
design with `--commanded one`, the traffic attention, the reward of D30, the masks, real and augmented windows. The
training is branch training (D37, §7 item 9): the saved state of a window at a branch point (executor, speaker cache,
judge, loop) and its restart, the branch groups and their advantage after the branch point, one pass, the executor in
inference mode while it speaks. The data term uses single-aircraft samples of the closed-loop sentences (D36). Before
it: a profile of one speaking batch (the prior, the executor, the masks, the edge features), and the go-around
probability of the base model on the final (§7 item 6). The multi-aircraft parts of §8 (D31): the
landing context from the loop for every aircraft (the archived `experiments/traffic_window.py` gives a replayed aircraft
its recorded context), the D23 test over a scene, the rule "established on the final" and
the replacement of the clearance mask (O6). The edge-feature conformance (§9.2 #6). The multi-aircraft step-8.9 reward is
not built (D30). A later stage with every aircraft of a window commanded is optional (§7 item 8).

### 14.5 Stage D: frontend and backend (outline)

The Training exports and the backend's live executor (`aeroviz_backend/autopilot_segment/`) on the new format; the
frontend reads the reading name, not the spec sha (§9.2 #9). After stage D the user merges the branch.

### 14.6 What Claude checks before stage B

The check of A0–A14 is done (`readouts/2026-10-04_stage_a_check.zh.md`, at `ab295b18`; its points became D48–D52).
Before stage B, Claude checks A15–A18 and the formal artefact:

1. D48, D49, D51, D53, D55 and D56 against the code: the module and the test that carry each (a table in the report).
2. The targeted tests of A15–A18 pass on the commit of the formal build (run again); the full suite of that commit
   passed (§14.1 rule 4; its log).
3. The formal artefact: the spec holds the values of D56 and its measurement records the chosen row; the labeller, the
   executor and the closed-loop conformance records pass for the code of the build; at each Δ, the closed-loop sentences
   of the replays give their flown states again on the 2 s rows, and the rule of D50 holds on every stored row; the
   readings of D34 (items 1–4) exist for every Δ, each split that A18 reads and each airport.
4. No write under a live root except the new directories of A18; no existing directory under `4dTrajectory/outputs/`
   changed; the `SHA256SUMS` beside the formal data match.
5. Nothing in the archive was edited after the move.
