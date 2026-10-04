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

**Relation to the current documents.** This design is NOT implemented. The current code implements reading
`instruction-v3` (spec `145d6911e75b`) and executor spec `v11_20260927`. The Chinese documents describe that code:
[vocabulary](instruction_vocabulary_design.zh.md), [executor](executor_design.zh.md), [prior](prior_design.zh.md),
[post-training](post_training_design.zh.md), [multi-aircraft](multi_aircraft_design.zh.md). When the code of this design
is merged, this document replaces the vocabulary and executor documents. Paths are relative to
`4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or the repository root.

---

## 0 Status

### 0.1 Decisions

| # | Item | State | Source |
|---|---|---|---|
| D1 | Merge the approach column into the runway column. The runway word is the expected runway. The model says it at the first predicted step. "Cleared" is not a word. "Go-around" is a value of the runway column | Decided | User, 2026-10-03 |
| D2 | The model flies the aircraft to the runway with heading words. No executor law turns the aircraft onto the final | Decided | User, 2026-10-03 |
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The LPV minimums are a judgement only. The model must learn to land or to go around | Decided | User, 2026-10-03 |
| D4 | The speed value "unspecified" starts at the capture row of the labeller, not at a clearance | Decided | User, 2026-10-03 |
| D5 | Prior inputs are in the frame of the runway in force. The prior has no airport embedding, no absolute position and no absolute direction | Decided | User, 2026-10-03 |
| D6 | The grids of the words (heading 5°, four descent classes) stay as they are now. A measurement or a rough labelling decides if they are sufficient | Decided | User, 2026-10-03 |
| D7 | This document sets no test criteria and no measurement criteria. The user sets them after the design is settled | Decided | User, 2026-10-03 |
| D8 | The heading grid is aligned to the course of the runway in force (§3.3) | Decided | User, 2026-10-03 |
| D9 | The executor laws in §5.7 are removed, the glidepath floor included | Decided | User, 2026-10-03 |
| D10 | Go-around is an event: "abandon this approach". It does not change the runway in force; a runway word ends it. Its effects on the executor, the judge and the masks: §3.2 | Decided | User, 2026-10-03 |
| D11 | The labeller stage includes an ablation of the row interval: 2 s now, larger intervals possible (§4.8) | Decided. Values: D25. Readings: D34 | User, 2026-10-03 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§6.1) | Decided | User, 2026-10-03 |
| D14 | While G is true: "no level-off" is not permitted (rule 5), and the procedure mask "no climb below the entry height" does not apply (§3.7) | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§6.1) | Decided | User, 2026-10-03 |
| D17 | Every column keeps the input "time since this column said its word in force", in seconds; the runway column's too (§6.1) | Decided | User, 2026-10-03 |
| D18 | The labeller reads the real go-arounds inside a sentence and says "go-around" at the go-around point (§4.6) | Decided | User, 2026-10-03 |
| D19 | After a labelled go-around, the runway word that ends G is at the first level-off after the go-around climb, and not later than the row of the next "no level-off" (§4.6) | Decided | User, 2026-10-03 |
| D20 | This design is a new version: it is developed on a new branch in a new worktree. Artefacts `v1`–`v6`, the executor specs and every prior are superseded and are not kept readable. The running experiments keep their own checkouts | Decided | User, 2026-10-03 |
| D21 | Identities bind format and data rules only. A code identity is a behaviour check on fixed inputs, never a hash of source bytes; data are identified by their flights, never by the bytes of a manifest (§9.2) | Decided | User, 2026-10-03 |
| D22 | Altitude words use the grid "optimal 40 levels" of the altitude-grid proposal: 60 m steps from 0 to 1,260 m, 120 m steps to 2,700 m, 450 m steps to 5,400 m (§3.4) | Decided | User, 2026-10-03 |
| D23 | How the prior gets the frame of R (D5) and the glidepath height (D13). The own state of the aircraft has no frame. Every position and direction is in the candidate vectors, each in the frame of its own candidate. R is an input only as its candidate vector. No input of a row up to the first predicted step is computed from R. Each candidate vector has the height above its own glidepath; the value of R is the input of D13 (§6.1) | Decided | User, 2026-10-03 |
| D24 | A candidate vector has no constant of its runway: no length, no elevation, no layout relative to R. Such a value comes back only as a variant that D39 selects (§6.1) | Decided | User, 2026-10-03 |
| D25 | The ablation reads Δ = 2, 4, 8 s: each divides the 16 s observation of the prior. At each Δ, the motion inputs of a row come from the 2 s before the row (§4.8, §6.1) | Decided | User, 2026-10-03 |
| D26 | The labeller reads a flight with a go-around approach by approach. An approach ends at the landing or at a go-around row. The last descent that reaches a go-around row says "no level-off". Each approach has its own capture row and its own "unspecified". The first row says the runway of the first approach. The go-around row is the row of the climb word (§4.2, §4.4–§4.6) | Decided | User, 2026-10-03 |
| D27 | "Go-around" changes no target of another column. A row that says "go-around" while "no level-off" is in force also says a level above the aircraft (rule 6). The executor keeps the heading word in force; it does not fly the course of R. While G is true, "unspecified" holds the airspeed (§3.2, §3.7, §5.4–§5.6) | Decided | User, 2026-10-03 |
| D28 | The vocabulary has one climb word. Its angle is a value of the executor. While G is true, the executor climbs at the steady climb angle that the thrust limit permits, not more than 3° and not less than 1.885° (200 ft per NM). While G is false, it climbs at the nominal angle of the climb class; that value comes from the spec measurement with D15 (§3.5, §5.5) | Decided (the range 1.885°–3° and the nominal from D15: the user; the thrust rule inside the range: Claude's proposal) | User, 2026-10-03 |
| D29 | Post-training starts from the base model in the multi-aircraft setting "one aircraft commanded". There is no single-aircraft post-training stage (§7) | Decided | User, 2026-10-03 |
| D30 | The post-training reward comes only from the outcome: 1 for `landed` without a go-around; 0.9ⁿ for `landed` after n go-arounds; 0 for every other outcome and for every loss of separation. No payment for a go-around without a landing. No mask on where "go-around" can be said (D10). The step-8.9 reward of the multi-aircraft design is not used (§7) | Decided | User, 2026-10-03 |
| D31 | Multi-aircraft inputs and judgements (§8): the landing context of every aircraft of a scene counts the landings of the closed loop; D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: O6). The prior's training stops on the select days (§6.3) | Decided | User, 2026-10-03 |
| D32 | Closed-loop reading. The labeller flies its sentence with the executor. When the flown path leaves the observed path by more than a tolerance, it says a correction word, and the observed word again when the flown path is back. The prior trains on the flown states of these sentences (§4.9, §6.1). Tolerances: lateral 30 m, vertical 15 m | Decided | User, 2026-10-03 |
| D33 | While G is true, no crossing of a threshold is an event: of R or of another candidate (§3.2, §5.8) | Decided | User, 2026-10-03 |
| D34 | The ablation of the row interval reads, at each Δ: the correction words of the closed-loop reading for each flight and each column; the errors left where §4.9 makes no correction; the replay outcomes. The user compares the Δ values on these readings (§4.8) | Decided | User, 2026-10-03 |
| D35 | No finer grids. A finer grid does not remove the drift of open-loop words (§11.9); the closed-loop reading does (D32). The heading grid stays 5°, the descent classes stay (D6) | Decided | User, 2026-10-03 |
| D36 | The teacher-forced data term of the post-training uses single-aircraft samples of the closed-loop sentences, not scene samples. The flown states keep the observed path, not the observed time (§4.9), so two aircraft of one scene do not keep their observed spacing (§7, §8) | Decided | User, 2026-10-04 |
| D37 | Branch training. Each training aircraft is spoken one time. When its reward is less than 1, it is spoken again from saved states at its first predicted step and every 120 s after it, before the event that ended it. Each branch group compares only the words after its branch point (§7) | Decided | User, 2026-10-04 |
| D38 | The DA check uses one definition with the evaluation module. Vertical: within ±22 m of the published glidepath of R, the bound of `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`, ICAO Doc 9613). Lateral: inside the FAS cone at the distance of the DA point. Neither is a parameter (§5.8) | Decided | User, 2026-10-04 |
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). Two variants: `full` (§6.1) and `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation). `constants` is chosen only if it is better by more than twice the seed scale (§6.3) | Decided | User, 2026-10-04 |
| D40 | Hyperparameters: the values of `instruction-v3` are the start (configuration A). The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§6.3) | Decided | User, 2026-10-04 |
| D41 | The design runs at an airport that is not in the training data: the number of candidates is not fixed; every input has a fixed physical scale, never a statistic of the training data; each fold of D39 flies the full closed loop at its held-out airport. The altitude grid limits this to airports whose approach levels lie in its 60 m segment (§6.4) | Decided | User, 2026-10-04 |
| D42 | The closed-loop reading says each observed word at the place where the observed aircraft heard it, not at the time: at the first Δ row at which the matched point of the flown aircraft has reached that place. The correction words do not change. The flight ends when the executor is done or at the time limit of the replay (§4.9) | Decided | User, 2026-10-04 |

### 0.2 Open items, in the order of discussion

| # | Item | Proposal | §  |
|---|---|---|---|
| O6 | Replacement for the clearance mask of the multi-aircraft loop; the rule "established on the final" of the separation judge and masks (D31) | Discuss with §8 | 8 |
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |

### 0.3 Implementation

| Part | State |
|---|---|
| Stage A: vocabulary, labeller, identities, executor, judge, replay (§14.2) | Not started. Another agent implements and reviews; Claude checks the result (§14.6) |
| Stage B: prior (§14.3) | Not started; outline |
| Stage C: post-training and multi-aircraft (§14.4) | Not started; outline |
| Stage D: frontend and backend (§14.5) | Not started; outline |

### 0.4 Plan

1. Stage A (§14.2): write the vocabulary, the labeller, the identities, the executor, the judge and the replay on the
   branch `dev-two-tier-v4`, milestones A0–A10, each with tests and a code review. Another agent does this.
2. Claude checks the result of stage A against this document (§14.6).
3. The user chooses the fitted values of D15. Then the formal artefact is built, and the
   readings of D34 are made at each row interval of the ablation (D11, D25: 2, 4, 8 s). The user compares them (D7).
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
8. **No compatibility.** A changed word, spec or payload gets a new name. The new code refuses the old artefacts.

---

## 2 Terms

| Term | Meaning |
|---|---|
| Row, step | One line of a sentence. A row is Δ = 2 s now, on even UTC seconds (artefact `v6_20261002`); the ablation of §4.8 also reads Δ = 4 and 8 s (D25) |
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
| Executor cycle | 1 s. The executor hears the words at the start of each 2 s row |
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

The approach column of reading `instruction-v3` does not exist (D1). In one row the prior says the columns in the order
of the table. A later column sees the values that the earlier columns said in the same row.

### 3.2 Runway column

**Meaning of a runway word.** "Expect runway k." The executor uses R for two things only: the conversion of the
heading words (§5.4) and the distance for the deceleration to the approach speed (§5.6). The judge uses R for the landing (§5.8). The prior gets R as the candidate vector of R (§6.1, D23). The multi-aircraft
loop uses R for the relations between aircraft (§8).

**Regulation.** Approach control gives the expected approach and runway at the first contact (FAA JO 7110.65BB
4-7-10 a: "Approach clearance or type approach to be expected ... Runway if different from that to which the
instrument approach is made"). Thus the runway is known before the approach clearance. The approach clearance is an
authorization ("CLEARED (Type of) APPROACH − ATC authorization for an aircraft to execute a specific instrument approach
procedure", Pilot/Controller Glossary). After it, the aircraft flies the procedure (AIM 5-5-4 a.3). The labeller cannot
see when a controller gave it: the data has no radio. In `instruction-v3` the labeller computed the clearance from the
start of the turn onto the final. This design has no clearance word. The model says the turns with heading words (D2).

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
| Judge (§5.8, D33) | While G is true, no crossing of a threshold is an event, of R or of another candidate (not a landing and not a failure): the flight continues. A go-around before the DA point answers a failed DA check. The time limit of the flight grows by 900 s (multi-aircraft design §6.6 step 8, the user 2026-10-02) |
| Masks (§3.7, D14, D27) | Rule 5: no "no level-off" while G is true. Rule 6: a go-around row with "no level-off" in force also says a level above the aircraft. The procedure mask "no climb below the entry height" does not apply while G is true |
| Prior (§6.1) | G is an input: the aircraft is in a missed approach |
| Multi-aircraft (§8) | The decision is explicit: it can be counted and rewarded, and the separation judge can treat the aircraft as no longer on the approach (open with O6) |

A runway word ends G. The model can say R again or another candidate ("expect runway k" again). This keeps the step-8
sequence of the multi-aircraft design (approach → go-around → approach) in one column.

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

**Reading (unchanged from `instruction-v3`, except the frame and the end).** Each row gets the grid value nearest to the
smoothed track 4 s later (the lead L). A new word occurs when the value changes. A continuous turn is a series of words
2 s apart. Turn rate shows in the time between the words. The heading words now continue to the end of the sentence.
They describe the turn onto the final and the final itself (§4.3).

**Envelope.** A word said at row r: from row r + L to the next heading word + L, the track is within ±4.5° of θ
(`envelope.heading_words_inside`). The rows of a word continue to the end of the sentence. There is no cut at a
clearance or a capture.

**Grid (D6, D35).** The grid stays 5°. A word of class 0 gives no lateral correction: the aircraft keeps its lateral
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
meaning stays an absolute height, the same at every airport and from every position. The grid comes from the proposal
[altitude_word_grid.zh.md](altitude_word_grid.zh.md): an exact dynamic-programming fit of at most three uniform
segments that makes the squared rounding error of the observed level-offs smallest for 40 levels (§11.7).

**Why a coarser grid higher up.** (1) The `instruction-v3` grid used 116 of its 182 classes; 99.7 % of the level words
are under 3,000 m. (2) A controller assigns a pressure altitude; the geometric height of one assigned level moves from
day to day by about height × ΔT / 273 (ΔT: the day's temperature deviation; the model does not see it). At KRDU this is
about 15 m at 600–900 m and about 90 m at 2,000–3,000 m ([vocabulary design](instruction_vocabulary_design.zh.md)
§2.4). A fine step high up only divides this spread.

**What a coarse segment costs.** In segment 3 two assigned levels 1,000 ft (304.8 m) apart can round to one word, and
a small change of level inside one step has no word. Such words are rare (above 2,700 m: approximately 0.3 % of the
words, mostly the entry level at row 0). The 1,000 ft between opposite base legs (7110.65BB 5-9-1 b) is at low
altitude, where the step is 60 m. At an airport above approximately 600 m elevation the final approach uses the 120 m
segment (§6.4).

**"No level-off".** "Descend at the angle in force. Do not level off." The judge stops the flight at the threshold or
at the ground. This value replaces "descend to land" of `instruction-v3`. The name changes because the meaning changes:
the executor has no special law for it (§5.5, §5.7).

**Why MSL and not height above the threshold.** Controllers assign MSL levels. Relative to the threshold elevation, the
level distributions of the five airports become less similar, not more (§11.2). KSJC has 28 % of its level words on
levels that the other four airports almost never use. This is airspace, not frame.

**Envelope.** The tube from the row of the word: max(T, h0 − s·tan γ_hi) − ε ≤ h ≤ max(T, h0 − s·tan γ_lo) + ε while
it descends, T ± ε after it arrives (s is the horizontal distance flown from the row of the word). ε depends on the
segment of T: half its step plus the fit residual 10 m, that is 40 / 70 / 235 m (in `instruction-v3`, 25 m for every
level). For "no level-off" there is no lower bound T. A new angle word starts a new tube. A containment rate is given
with these widths (principle 6). The level detection of the labeller does not use ε (§4.4).

### 3.5 Angle column

| Class | Nominal | Range |
|---|---|---|
| Level | 0° | used only to hold a level that the aircraft reached |
| Descent 1 | 0.92° | −0.5° to 1.52° |
| Descent 2 | 2.13° | 1.52° to 2.59° |
| Descent 3 | 3.06° | 2.59° to 3.74° |
| Descent 4 | 4.41° | 3.74° to 10° |
| Climb | No angle in the word. The executor's angle (D28): the nominal of D15 while G is false (fitted: 1.32°); 1.885°–3° while G is true | 0.5° to 15° climb (the labeller's range of a climb piece) |

The four descent classes come from a length-weighted k-means on the train days (spec `145d6911e75b`). The grid stays
(D6, D35). With D3 the model holds the glidepath with these classes; the closed-loop reading gives the corrections
(D32, §4.9).

**Rounder values (D15).** The spec measurement gives each value that it fits from data (the descent centres and edges,
the climb centre) with rounder candidates (for example to 0.5°, 0.25°, 0.1°) and, for each candidate, the fit that it
leaves: the end-of-piece height error that the measurement already reports (`instructions/measure.py`
`fit_descent_classes`), and the same for the climb pieces. The user chooses (criteria: D7). Two facts for that choice:
24 of the 25 candidate runways publish a 3.0° glidepath (§11.4), and with D3 the model holds the final with an angle
word, so a centre of 3.06° flies 10 m from a 3.0° glidepath in 10 km; the k-means with three classes put a centre at
3.006°. The values that the measurement takes as percentiles are already rounded by their rules (turn rates to
0.1°/s, the bank limit to 1°, the corridor to 5 m, 0.05° and 1°, the acceleration to 0.1 m/s²).

**Climb (D28).** The vocabulary has one climb word: "climb". The word has no angle. The angle is a value of the
executor (§5.5), from the state G:

- **G true (a missed approach).** The executor climbs at the go-around angle: the steady climb angle that the thrust
  limit permits at the present airspeed, not more than 3° and not less than 1.885°. 1.885° is the minimum gradient of a
  missed approach, 200 ft per NM (AIM 5-4-21 b). 3° is the upper limit. The real go-around climbs are steeper (§11.6),
  so the 3° limit usually applies.
- **G false.** The executor climbs at the nominal angle of the climb class. The spec measurement fits it from the climb
  pieces of the train days, with rounder candidates (D15); the user chooses. The train days have 1,498 climb pieces
  among 233,649 vertical pieces (0.6 %); their length-weighted median is 1.32°.

The labeller does not read a go-around angle: a climb piece is a climb when its angle is 0.5° to 15°, in and out of G.

### 3.6 Speed column

**Meaning.** "Hold ground speed V." Controllers assign indicated airspeed; the data has no airspeed and no wind, so the
words use ground speed. "Unspecified" means that the pilot flies the approach speed of the aircraft type. An approach
clearance cancels the assigned speeds (7110.65BB 5-7-1 d). In this design the labeller starts "unspecified" at the
capture row (D4, §4.5).

**Envelope (unchanged).** A transition is monotone (a step back ≤ 5 m/s) with acceleration ≤ 1.4 m/s². After it, the
speed stays within V ± 5 m/s. "Unspecified" checks only the range.

**Wind (O8, open).** At constant airspeed, the ground speed changes when the aircraft turns in wind. Such a change can
become a speed word that no controller said. A check against the surface winds of the METAR data is open.

### 3.7 Grammar rules

The labeller checks these rules. The prior applies them as masks when it decodes (`instructions/grammar.py`). The
labeller identity hash must include `grammar.py` (code-health follow-up, row "`instructions/grammar.py` is outside the
labeller sha").

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

**Masks while G is true (D14).** Only one mask stops a go-around: the procedure rule "no climb after the aircraft is
below the entry height before the join". It does not apply while G is true; the current code already lifts it while a
go-around is in force (`prior/procedure.py:126` `climb_barred`), and its stretch starts again after the go-around. The
other procedure masks are lower limits (the glidepath lower edge inside the FAF, the DA before the join). A go-around
climbs above them, so they stay. The vocabulary rules 1–4 stay.

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

The labeller reads one sentence from one observed flight (`instructions/labeller/read.read_flight`). This section gives
only what differs from `instruction-v3` and what stays the same. The detailed rules of the readings that stay are in the
[vocabulary design](instruction_vocabulary_design.zh.md) §3.

### 4.1 Signals, cut and gate (unchanged)

- Signals come from the ts data plane (`build_series`): read-time repair, MSL, 2 s rows on even UTC seconds, the airport
  frame. Test days are sealed: the labeller never opens them (contract C32).
- The sentence ends before the first crossing of a threshold plane that meets the landing screen of the harvest
  (lateral ≤ 1,000 m and ≤ half the spacing to a parallel runway; height ≤ 100 m; `read.landing_passages`).
- The labeller refuses a flight that crosses and then comes back before the threshold, a flight with a bad raw ground
  speed, and the other refusals of `instruction-v3`.
- Smoothing: track 6 s, altitude 10 s, ground speed 10 s, centred.

### 4.2 Runway word

The first row says the runway of the first approach (D26). For a flight without a go-around, this is the runway on
which the flight landed (the harvest assignment). For a flight with a go-around, it is the runway of the first low pass
(§4.6). The data shows a change of runway only at a go-around, so a labelled sentence changes R only with the runway
word that ends G (D19). A candidate without a published threshold crossing height (TCH), a
glidepath angle or an LPV DA is refused before labelling (the judge needs all three, §5.8).

### 4.3 Heading words

- Frame: the course of the landed runway (D8).
- Reading: the per-row reading of §3.3, from row 0 to the last row of the sentence. In `instruction-v3` the words
  stopped at the clearance row. Now they also cover the turn onto the final and the final itself.
- Removed: the clearance placement (`read_lateral`: the turn onset and the convergence test), and the turn check of the
  capture turn (`turn_check` as an envelope of a clearance).
- Kept for other uses: the capture row (`capture_row`), for D4 and for the readout groups (straight-in, vectored).

### 4.4 Altitude and angle words

Piecewise-linear fit of the smoothed altitude against distance (residual ≤ 10 m), as in `instruction-v3`. One change
follows from D22. In `instruction-v3` a piece is level when its rows lie within 25 m of the nearest 30 m level
(`instructions/labeller/vertical.py:60`); with 60–450 m steps that test would miss real level-offs (a level at 630 m is
30 m from both neighbours) or swallow descents. So the two questions separate:

1. **Is the piece level?** At least 20 s, and every row within 25 m of the piece's own median. This is a physical test
   with a constant of the labeller; it does not use the grid.
2. **Which word?** The nearest level of the grid to the median.

Two level pieces in a row with the same word merge. Moving pieces give the level words and the angle words. A move
between two levels that round to the same word gives no level word (it cannot be said; §3.4). The last descent of
each approach gives "no level-off" when it reaches the end of the approach: the threshold, or the go-around row (D26,
§4.6). The tube of §3.4 checks each word, with the ε of its segment.

### 4.5 Speed words

The reading of `instruction-v3` stays, with two changes. (1) "Unspecified" starts at the capture row, not at the
clearance row (D4). (2) Each approach has its own reading (D26): speed words before its capture row, "unspecified" from
its capture row to the end of the approach. An approach that ends at a go-around row outside the capture corridor has
no capture row; its speed word in force stays. After the go-around row, the reading of the next approach starts. The
two exceptions stay, with the new anchor:

1. A hold of at least 30 s that ends at or after the capture row, and ends 9,260 m (5 NM, 7110.65BB 5-7-1 b.4) or
   more before the threshold, keeps the speed word in force. "Unspecified" starts at the end of the last such hold.
2. A deceleration that started less than 20 s before that row is already the pilot's speed. "Unspecified" starts where
   it started.

### 4.6 Go-around words (D18, D19)

**The data.** R40 v2 (`go_around_census`, readout `readouts/2026-10-01_go_arounds.zh.md` §3) found 109 go-arounds on the
training days. In 73 of them the go-around point is inside a labelled sentence of `v5`: `instruction-v3` read the first
approach and the climb as heading and altitude words, and put the clearance at the second capture. In 11 more the
point is inside the arrival slice, but the labeller refused the flight. In 23 the go-around point is before the sentence
(the aircraft left 25 km; the slice starts at its return). Without a go-around word the prior gives the word a
probability of approximately 1e−11 (the 9.4 readouts).

**The reading.**

1. The labeller finds a go-around with the rule of R40: a low pass on the final of a runway end, down from a level held
   for 20 s at least 150 m higher, then up to a level held for 20 s at least 150 m higher. The rule moves from the runner
   into `instructions/labeller/` (the labeller cannot import a runner), with its tests.
2. The go-around row is the first row of the climb after the low pass: the row where the altitude reading starts the
   climb run, with its level word and its climb word (D26). The runway column says "go-around" in that row; G becomes
   true. A go-around without a climb word is refused, with its reason.
3. The climb is read as altitude and angle words, as every climb. Its level word and its climb word are in the
   go-around row (rule 6). While G is true the executor flies the climb at the go-around angle (D28, §5.5).
4. The runway word that ends G (D19): at the first level-off after the go-around climb, and not later than the row of
   the next "no level-off" (rule 5). It says the runway of the next approach: the runway on which the flight landed,
   after its last go-around.
5. A flight that the labeller refused only because it crossed a threshold and came back is read again with this rule.
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

**Not in this version.** The 23 go-arounds before their sentence need a longer arrival slice, and 9 sentences of `v5`
end at a low go-around that the harvest took for the landing (it takes the best-aligned crossing under 100 m, not the
last; R40 v2 §5). Both are changes of the harvest or of the data plane, which other lines (evaluation, the optimizer,
the one-tier models) share.

### 4.7 Assembly and artefact

- All words go on the row grid (2 s now; §4.8). A word equal to the word in force is not a word. The first row says all five columns.
  Two different words in one column in one row: refused.
- The artefact has a new reading name and a new spec format (principle 8). The spec records the grids, the classes, the
  tolerances and the reading. The labeller identity hash covers the reading code and `grammar.py`.
- The open-loop sentences are the input of the closed-loop reading (§4.9). The training sentences of the prior are the
  closed-loop sentences with their flown states.
- The new code refuses artefacts `v1`–`v6` and every prior trained on them.

### 4.8 Row interval: an ablation (D11)

**What the row interval Δ sets.** Δ is 2 s now: the resample step of the data plane. Δ sets the grid of the words, how
often the prior speaks and the executor hears, the length of a sentence (median 154 rows at 2 s) and the step of a
multi-aircraft scene. Each column changes its word at only 1–2 % of the 2 s rows ([prior design](prior_design.zh.md)
§1), so a larger Δ can be sufficient. The user asked to measure this at the labeller stage, before the prior trains.

**How the ablation changes only Δ.** The labeller reads each flight at the 2 s rows of the data, as now. Then it puts
the words on a Δ grid:

1. The Δ rows are the rows on UTC multiples of Δ (the rule of artefact `v6_20261002` for Δ = 2 s), so the aircraft of a
   scene stay on one grid.
2. At each Δ row, each column says the word in force at that 2 s row if it differs from the word in force at the
   previous Δ row. Inside one interval, only the last word of a column stays.
3. The first Δ row says all five columns. The grammar rules (§3.7) are checked again on the Δ grid.

Thus one reading gives every Δ. Δ must be a multiple of 2 s, and it must divide the 16 s observation of the prior
(D25): the first predicted step is a Δ row.

**What a larger Δ changes (to read in the ablation; no criteria here, D7):**

| Part | Change |
|---|---|
| Heading words | In a 3°/s turn the track moves 12° in 4 s, so one word jumps two or three 5° classes. The lead L = 4 s is one row at Δ = 4 s and less than one row above it |
| Executor | The cycle stays 1 s. It hears the words at each Δ row. The heading law (arrive L after the hearing, the stopping rate) flies larger steps |
| Final approach | With D2 and D3 the model corrects the final only every Δ; an error grows for a longer time before the next word |
| Prior | Fewer steps for each flight and more changes for each step. The observation stays 16 s (`N_LOOK` = 16 s / Δ rows: 8, 4, 2). The motion inputs do not change with Δ: they come from the 2 s before the row (D25, §6.1). A loss for each step cannot be compared between two Δ; a loss for each flight or each second can |
| Multi-aircraft | The scene step is Δ |

**Values (D25).** Δ = 2, 4, 8 s. A Δ of 6 s is not used: 16 s is not a whole number of 6 s rows.

**Readings (D34).** The closed-loop reading (§4.9) makes each replay follow the observed path, so the landed share no
longer shows how well a Δ carries a flight. At each Δ the ablation reads instead:

1. the correction words of the closed-loop reading, for each flight and each column;
2. the errors left where §4.9 makes no correction (a level hold, a climb, no steeper or shallower class): the largest
   |e_y| and |e_h| for each flight;
3. the replay outcomes (§5.8).

The user compares the Δ values on these readings; this document sets no threshold (D7).

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ. At Δ = 2 s it is the value of now.

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
4. The closed-loop reading runs on the sentence of each row interval (§4.8). The corrections are on the Δ rows; the
   corrections and the flown states belong to that Δ.
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

**When the observed words are said (D42).** The observed time of the matched point is the time at which the observed
aircraft was at that point. At each Δ row, the labeller says the observed words of each Δ row of the open-loop sentence
whose time is not later than the observed time of the matched point, and that it did not say before. When the matched
point passes more than one of these rows in one Δ row, the labeller says the last observed word of each column. When
the flown aircraft is behind the observed aircraft, the observed words wait. The first predicted step says every
column (rule 1): the observed words in force there.

Why:

- A word gives a target, not a rate. A speed word does not tell how fast to change the speed, and the executor changes
  the speed at its own fixed rate (§5.6). Thus the flown aircraft moves ahead of the observed aircraft along the path,
  or falls behind it. The comparison does not correct this difference.
- A turn word must come where the observed aircraft turned. A word said at the observed time comes where the flown
  aircraft is at that time: 2 km ahead along the path before a turn of 90° gives a lateral offset of 2 km after the
  turn. One correction class removes such an offset too slowly (§11.11).
- The controller gives a turn at a place, not at a time. The prior learns from the flown states when to say a word, so
  the words must agree with the flown states.
- A go-around word said at the observed time can come after the flown aircraft has crossed the threshold. Said at the
  place, it comes where the observed aircraft went around.

Thus the flown time to the threshold is not the observed time. The flown states keep the observed path, not the
observed time (D36).

**Lateral correction.**

1. When |e_y| > Y (the lateral tolerance), and the row says no new observed heading word, the labeller says the
   heading class one step (5°) from the observed word in force, toward the observed path.
2. When |e_y| < Y / 2, or e_y changes its sign, the labeller says the observed word in force again.
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.

**Vertical correction.**

1. Only while a descent class is in force (toward a level or with "no level-off"). During a level hold the level word
   is the target; its rounding to the grid (§3.4) is not corrected. A climb has one class, so a climb gets no
   correction.
2. When e_h > H (too high), the labeller says the next steeper descent class. When e_h < −H (too low), it says the next
   shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then there is
   no correction.
3. When |e_h| < H / 2, or e_h changes its sign, the labeller says the observed angle class in force again. A new
   observed altitude word or angle word ends a correction.

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
flown states on the rows (position in the airport frame, MSL height, track, ground speed, vertical rate); the count of
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

### 5.3 One cycle (unchanged)

The outer loop changes the words in force to three wanted rates: track rate, path-angle rate, airspeed rate. The inner
loop inverts the point-mass equations for bank, load factor and thrust, in this order: bank limit 32° and roll rate
p = 5°/s, load factor in [0.5, 2.0], stall floor 1.10·V_stall(n), thrust in [−0.2, 1.0]·T_max. One cycle is 1 s,
integrated with RK4 under zero-order hold. Details: [executor design](executor_design.zh.md) §2, §3, §7.

### 5.4 Lateral law

- **Heading words** (unchanged law, new conversion). When the executor hears a heading word, it sets the absolute target
  θ = course(R) + 5k°. It turns so that the track arrives at θ L = 4 s after the hearing time. The rate is the smallest
  of |e| / (time to go), the stopping rate √(2·g·p·|e| / V) and the turn-rate limit 4.7°/s (`lateral.word_rate`). The
  first word turns in the shorter direction. A later word continues from the target before it (a series of words never
  reverses a turn).
- **Go-around (D27).** "Go-around" does not change the heading target. The executor keeps flying the heading word in
  force. The model turns the aircraft with heading words, as at every other row.
- Nothing else. There is no capture state, no turn onto the final, no centreline tracking (§5.7).

### 5.5 Vertical law

The inner path-angle loop stays: γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max), τ_γ = 2 s (executor design §5.1). The modes:

| Words in force | Mode | γ_ref |
|---|---|---|
| Level T + level | Hold | Altitude hold law, τ_h = 8 s (executor design §5.5) |
| Level T + descent class | Descend to T | −nominal angle of the class; level-off starts at V·γ²/(2·γ̇_max) above T |
| Level T + climb, G false | Climb to T | +nominal angle of the climb class (D15, D28); level-off as above |
| Level T + climb, G true | Go-around climb to T | +γ_GA, the go-around angle (below); level-off as above |
| "No level-off" + descent class | Descend | −nominal angle of the class; no level-off |

**The go-around angle (D28).** γ_GA = min(3°, max(1.885°, γ_T)). γ_T is the steady climb angle at the thrust limit and
the present airspeed: sin γ_T = (T_max − D) / (m·g), with the drag D of the present state at load factor 1 (the
point-mass model of §5.3). The executor computes γ_T at each cycle from the aircraft data that it already reads (§5.2).
1.885° is the minimum gradient of a missed approach, 200 ft per NM (AIM 5-4-21 b); 3° is the upper limit. When γ_T is
less than 1.885°, the reference is 1.885° and the inner loop gives what the thrust permits;
layer 1 of the judge records the limit that bound (§5.8). No word of the vocabulary says an angle of a climb: the
vocabulary has one climb word (§3.5).

**No climb without a target.** Rules 5 and 6 (§3.7) make sure that "no level-off" is never in force while G is true.
Thus the executor has no mode "climb with no target", and "go-around" alone starts no climb.

### 5.6 Speed law (unchanged)

V_ref = V_g / cos γ (ground speed to airspeed without wind), not below 1.10·V_stall(n). The speed changes at
a = 0.25 m/s² (a speed step of 5 m/s divided by the minimum hold of 20 s) and goes exponentially into the last 5 m/s.
"Unspecified" is the published approach speed of the type at the published maximum landing mass. The deceleration to it
is the larger of a and the rate that reaches it at the threshold of R, not more than 1.4 m/s². While G is true,
"unspecified" means the pilot's own speed in a missed approach: the executor holds the airspeed that the aircraft has
at the go-around row (D27; Claude's reading, not checked in the regulation text). This replaces no word of the model:
a speed word of the model replaces it at any row.

### 5.7 Removed laws (D9)

| Law in `v11` | Where | Why it goes |
|---|---|---|
| Runway lock after a clearance or a capture | `executor.runway_locked`, `lateral.Lateral.rate` | No clearance word; D12 |
| Capture state and the capture turn (an arc tangent to the final, planned at 1.53°/s) | `lateral.py:284`, executor design §4.4 | D2 |
| Heading bend of ±4.5° and the own intercept at 30° after a clearance | `lateral.py:293`, executor design §4.3 | D2 |
| Centreline tracking after capture | executor design §4.5 | D3 |
| "Descend to land": aim at the TCH crossing point, the tube limit, the exit from the tube, not below the crossing height | `vertical.py:140`, executor design §5.3.1–§5.3.3 | D3 |
| Glidepath floor (published glidepath − 60 m) and level flight below the glidepath after capture | `vertical.py:74`, executor design §5.3.4–§5.3.5 | D3, D9 |

### 5.8 Judge

**Layer 1, each cycle (unchanged).** The wanted rates, the rates given, and the limit that bound.

**Layer 2, each word (changed).** The labeller's checks on the flown track, from the row where the executor heard the
word: heading words to the end of the flight (§3.3); altitude and angle words in their tubes; speed words in their
spans. The checks of the clearance and of the corridor go.

**Layer 3, each flight (changed).** A crossing of R is an approach crossing when G is false and the aircraft crosses the
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
crossing is an event, of R or of another candidate (D33): the flight continues. `crossed_without_capture` goes, because the executor has no capture
state.

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

**Quality** stays with the evaluation module (lateral, vertical and speed gates). The judge does not repeat it.

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
  final, the height above its threshold, the motion direction minus its course (sine, cosine), as now
  (`instructions.airport.relative_to_runway`, `prior/data.py:198`). "The frame of R" is the candidate vector of R, which
  the prior gets as the runway in force (`prior/model.py:388` `_runway`, as now). There is no second set of values
  relative to R.
- **No input is computed from R before R is said (D23).** The first predicted step says R. The inputs of the rows before
  it and of the first predicted step itself use no value computed from R. (The heads of a row see the runway that the
  same row said, §3.1; that is an output, not an input of the row.) The reason: the artefact writes the runway word at
  row 0 (§4.7), and its value is the runway on which the flight landed. An input of these rows computed from it gives
  the answer of the first predicted step, and in closed loop the same input does not exist. A test: a change of the
  runway word of a sentence leaves the inputs of rows 0 to `N_LOOK` the same, bit for bit.
- **Removed (D5, D24):** the airport embedding (`prior/model.py:312`), the absolute position E, N and the absolute
  motion direction (`prior/data.py:63`), and the whole table of candidate constants (`prior/data.py:70`): the absolute
  threshold position, the course, the elevation and the length.
- **Candidates (D24).** A candidate vector has only values that change with the aircraft: the values of the frame above,
  the glidepath height below, and the landings on the candidate in the 30 min before the step (as now). It has no
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
  geometry and stays; only the held-out airports of D39 can measure how much the prior uses it. A constant comes back
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
  give them; never the fitted track, ground speed or vertical rate of the signals, which use 7.5 s of the future (as
  now, `prior/data.py:163` `_motion`).
- **Time (D16).** No row position embedding (`prior/model.py:313`) and no input "time from row 0" (`prior/data.py:63`
  `time`): both measure the time since the aircraft entered the 25 km slice, a cut of the data, and a long sentence (a
  go-around adds up to 900 s) reaches rows that training seldom saw. The causal time attention gives the order. RoPE in
  the time attention gives how far back each earlier row is: the rotation of a row's query and key uses its time in
  seconds, so the attention reads only time differences. Seconds, not rows, so that every row interval of D11 reads the
  same time. The time of a row is in seconds from the aircraft's own row 0. Only differences count, so the zero changes
  no result; but a UTC time (approximately 1.76e9 s) in float32 has a step of 128 s and loses the rows. RoPE works with
  the row-by-row cache of the speaker (`Prior.extend`): a key is rotated once, when it is written. The RoPE base is set
  at implementation.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force"
  (`since`, now `log1p(rows) / 5`, counted from the first predicted step at the earliest). It is in seconds, for every
  column, so that every row interval of D11 reads the same time. The runway column's value is, in the labelled data, the time since the first predicted step, because a
  labelled sentence says its runway only there, and again only at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or given again
  (a change of runway, or the runway word that ends a go-around), so it measures a real fact.

### 6.2 Outputs

Five heads in the order of §3.1. The runway head scores each candidate (a pointer) plus "unchanged" and
"go-around". The number of candidates is the airport's own: no slot count is fixed by the model (D41). The first
predicted step masks "unchanged" in each column and "go-around" in the runway column.

### 6.3 Training

Teacher forcing on the new artefact, split by operating day (`data/day_split_20260924.json`; test days sealed). KAUS is
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
| 1 | Variant `full`, four configurations, each on the 5 folds. A: the start. B: d_model 128, feed-forward 512. C: d_model 256, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. The layers and the learning rate stay | 20 |
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

1. **One stage, from the base model, with one aircraft commanded (D29).** The closed loop is the multi-aircraft window
   loop with one commanded aircraft (R37 `--commanded one`, multi-aircraft design §6.6 step 9.4). The model speaks for
   one aircraft; the other aircraft fly their records. The start model is the base model with a traffic attention whose
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
   | Every other outcome; every loss of separation (multi-aircraft design §3.4) | 0 |

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
   join, no climb back; D14 while G is true); the separation masks (§8, O6). The masks stay with the model
   (`procedure_masks.json`, contract C35).
4. **Windows.** Real windows and augmented windows of the train days: B (a moved start: a turn about the airport, a
   height change and a speed change, post-training design §4), A (one inserted aircraft that flies its record), D (the
   leader moved) (multi-aircraft design §5, step 9.4). The counts are set at stage C.
5. **Loss**: the clipped-ratio surrogate (ε = 0.2) with the advantage inside each branch group (item 9); the pull to
   the base model (0.04, the KL on the sampled words, masked distribution); the teacher-forced data term (1) on
   single-aircraft samples of the closed-loop sentences of the train days (D36); the traffic attention has its own
   learning rate. One pass over the sentences of a round.
6. **Go-around sampling first.** Before the training, measure the probability that the base model gives "go-around" on
   the final (with D26 the data has go-arounds with "no level-off" and "unspecified" in force). If the model says
   "go-around" by itself, the training uses no probes. If it does not, the probes of the multi-aircraft design (8.10)
   are discussed with the evidence of §11.6: a cross-entropy on a forced word teaches the word, not when to say it.
7. **Selection.** On the select days; the validation days are read one time for each stage. The airport
   generalization of the design is chosen in stage B (D39) and tested at the end on KAUS.
8. **Later, optional.** A stage with every aircraft of a window commanded (multi-aircraft design 7.6, 9.5) starts from
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

- **Time.** A scene puts its aircraft on one UTC grid: a scene step is one instant for every aircraft (multi-aircraft
  design §2.1). That is the layout of the data, not a model input, and D16 does not change it. RoPE runs along each
  aircraft's own rows; the attention among the aircraft and the traffic attention read one scene step, which is one
  instant, and use no position. Edge features are relative (the approach clock is a distance; the closest-approach time
  is a time difference). The landing context counts the landings in the 30 min before the step's UTC time, as now.
- R exists at every step for every aircraft from its first predicted step on. Before it, no input reads the aircraft's
  R (D23), the edge features included. Thus the relations of the edge features (`inference/scene_edges.py`: the
  approach clock, the same runway, the parallel runways) and the separation judge (`inference/separation.py`) have a
  runway at every step.
- The clearance mask (`inference/separation_masks.py:131`) blocks "cleared" while a cleared aircraft ahead is too close.
  This design has no "cleared" word. A replacement is open (O6).
- The speed-word mask stays.
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
- **Established on the final (D31, O6).** The separation judge (multi-aircraft design §3.2: the in-trail rule, who is
  responsible) and the separation masks must know if an aircraft is established on the final of its R. This comes from
  a function of one row: the state of the aircraft at that row and its R. It is the same for every aircraft
  (commanded, labelled, replayed). It does not come from the capture row, which uses later rows (§2), and not from an
  executor state, because the executor has none (§5.8). It is not an executor law: only the separation judge and the
  masks read it. Its rule is open with O6.

---

## 9 Gates and identities

### 9.1 Gates

Each stage keeps a gate: the labeller (completeness, envelope containment with envelope width), the replay (the
executor flies the labelled sentences), the prior (teacher-forced likelihood against baselines, free generation), the
post-training, the multi-aircraft stages. The user sets the criteria of each gate after the design is settled (D7).
This document sets none.

### 9.2 Identities (D21)

**Rule** (the user, 2026-10-02 and 2026-10-03). An identity binds the format (what the words and the payloads mean)
and the data rules (the sealed test days). Code is identified by what it does on fixed inputs (a conformance check), not
by the bytes of its source. Data are identified by their flights, not by the bytes of a file they came from.

| # | Identity in `instruction-v3` | Decision | Where (stage) |
|---|---|---|---|
| 1 | Vocabulary spec sha (grids, classes, tolerances, reading name) | Keep. It is the format | A |
| 2 | Labeller source hash over bytes (`instructions/artefact.py:132`, checked by `require_current_labeller` and `load_sentences`) | Remove. Replace it by the labeller conformance (§14.2 A3): a fixed reference sample labelled again by today's code gives the same words. The conformance covers the closed-loop reading too (D32): with the artefact's executor spec, the same correction words and the same flown states, within the tolerance of the executor conformance. The artefact records which code wrote it as information | A |
| 3 | Executor spec: the parameters' sha and the conformance of the reference tracks (C33) | Keep. It is the model for #2 and #6 | A |
| 4 | Each airport's arrival manifest bytes (`arrival_manifest_sha256s`) | Remove as a check. The artefact keeps the signals; a consumer that rebuilds a flight from the harvest compares it row by row with the stored signals (`autopilot/flights.py` `require_same_flight`, kept) | A |
| 5 | Prior → artefact: spec sha, labeller sha, day split, candidate table (C34) | Keep the spec sha, the day split and the candidate table. Replace the labeller sha by the artefact's identity: the sha256 of its sentence files | B |
| 6 | Edge source hash (`EDGE_SOURCES`) | Remove. Replace it by a conformance check: the edge features of fixed reference scenes are the same | C |
| 7 | Procedure masks: the set name, the checkpoint sha, the digests of the procedure data (C35) | Keep. The procedure data are the format of the masks | B |
| 8 | Window readouts: the code version and the conformance R44 | Keep | C |
| 9 | The frontend pins the vocabulary spec sha; the pin of the single-flight executor | Replace by the reading name | D |
| 10 | The day split file and the sealed test days (C32) | Keep. It is a data rule | A |

---

## 10 Values

| Item | Value | Source |
|---|---|---|
| Row | 2 s, on even UTC seconds; the ablation also reads 4 and 8 s | Artefact `v6_20261002`; D11, D25 |
| Observation of the prior | 16 s (8, 4, 2 rows at Δ = 2, 4, 8 s) | Prior design §10 (`N_LOOK`); D25 |
| Motion inputs of the prior | Displacement in the 2 s before the row, at every Δ | D25 |
| Executor cycle | 1 s | Fixed choice |
| Heading grid | 5°, relative to the course of R | Spec (grid); D8 (frame) |
| Heading lead L | 4 s | Measured (vocabulary design §10.1) |
| Heading tolerance | 4.5° | Half grid 2.5° + 2° |
| Turn-rate limit | 4.7°/s | Spec, measured (p99.9) |
| Bank limit | 32° | Spec, measured (p99.9) |
| Roll rate p | 5°/s | Source cited in executor design §9 |
| Level grid | 60 m to 1,260 m, 120 m to 2,700 m, 450 m to 5,400 m MSL; 40 levels | D22 (fit of the altitude-grid proposal) |
| Level envelope ε | half the segment's step + 10 m: 40 / 70 / 235 m | D22 |
| Level detection | ≥ 20 s, rows within 25 m of the piece's median | Labeller constant (§4.4) |
| Descent classes | edges −0.5 / 1.52 / 2.59 / 3.74 / 10°; nominal 0.92 / 2.13 / 3.06 / 4.41° | Spec, k-means on train days |
| Climb word | One class; a climb piece is 0.5–15° (labeller); executor angle while G is false: the nominal of D15 (fitted 1.32°) | Spec; D15, D28 |
| Go-around angle (executor, while G is true) | The steady climb angle at the thrust limit, within 1.885°–3° | AIM 5-4-21 b (minimum, 200 ft per NM); D28 (maximum) |
| Reward of a landing after n go-arounds | 0.9ⁿ | D30 |
| Speed grid, range, tolerance | 5 m/s, 20–250 m/s, ±5 m/s | Spec |
| Speed change rate (executor) | 0.25 m/s² | 5 m/s ÷ 20 s (user, 2026-09-24) |
| Transition acceleration limit | 1.4 m/s² | Spec, measured (p99.9) |
| Capture corridor (labeller only) | 20 m + d·tan 0.45°, course ±2° | Spec, measured (p99) |
| Landing screen | lateral ≤ 1,000 m and ≤ half the parallel spacing; height ≤ 100 m | `final_approach.assign.LandingScreen` |
| Landed lateral limit | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| DA check | vertical ±22 m of the published glidepath; lateral inside the FAS cone at the DA distance | D38 (`evaluation/thresholds.py`; FAA Order 8260.58D Formula 3-1-1) |
| Time limit | remaining observed time × 1.5 | Fixed choice |
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; a correction ends below half the tolerance or at a change of sign | D32 |
| Closed-loop reading: correction | heading: one class (5°) toward the observed path; angle: the next descent class | D32 |
| Closed-loop reading: observed words | at the first Δ row at which the matched point reaches the place where the observed aircraft heard the word | D42 |

---

## 11 Evidence

All counts in §11.2–§11.4 come from two runners on the train split of artefact `v6_20261002` (44,375 sentences),
2026-10-03: R48 `instruction_word_frames` (§11.2, §11.3; output
`4dTrajectory/outputs/POOLED/analyses/word_frames_20261003/word_frames.json`) and R49 `instruction_final_approach`
(§11.4; output `4dTrajectory/outputs/POOLED/analyses/final_approach_20261003/final_approach.json`). Each output gives
the same numbers for each airport too. `docs/reference/runners.md` R48, R49 give the definitions.

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
`instruction-v3` grid (30 m, 182 classes) has at most 15 m. These numbers come from a one-off script of that proposal;
the spec measurement of stage A gives them again from a runner (§14.2 A3). The proposal did not fly the grid: the replay
of stage A does.

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

1. Make the branch from `dev-two-tier` at the commit that holds this section:
   `git worktree add -b dev-two-tier-v4 .claude/worktrees/two-tier-v4 dev-two-tier`. In the worktree, link the ignored
   data trees with absolute links: `data`, `trajectory_data_process/outputs`, `4dTrajectory/outputs`,
   `aeroviz-4d/public/data/airports`. These links point at LIVE data.
2. Each milestone: read the code that it changes; write the code and its tests; run the milestone's test files; get a
   code review from a separate reviewer (code only, never documents); correct; commit with explicit paths. Never use
   `git add -A`. Before each commit, read `git diff --cached --stat`.
3. A test never writes under a live root. A test that calls a runner's `main()` gives every write root a tmp path
   (`tests/support.py` `labelled_instruction_artefact` is an example).
4. Run single test files in the foreground. Run the full ts suite (approximately 55 min) only at the end of the stage,
   detached (`nohup setsid`, the script writes its own PID file).
5. No compatibility (principle 8). Every changed format gets a new name: reading `instruction-v4`, spec schema
   `ts-instruction-spec-v5`, sentences schema `ts-instruction-sentences-v3`, executor spec schema `ts-executor-spec-v7`.
   The new code refuses an old artefact by its name. No `.get(key, default)` fallbacks, no branches on a schema version.
6. SI units only. A value from a regulation has its paragraph in a comment and is defined one time.
7. No formal data build and no formal readout. A smoke build (a stated limit for each airport) goes to a tmp or
   scratch directory. The formal artefact, the replay gate and every criterion wait for the user (D7). Never write into
   an existing directory under `4dTrajectory/outputs/`.
8. Do not touch the checkouts of running experiments (`.claude/worktrees/step9-run` and others) or the main checkout.
9. A defect that you find outside the milestone goes to `docs/code-health-followups.md` (an entry and a table row), not
   into the change.
10. At each milestone, update §0.3 (state, commit). At the end of stage A, update §13 (key code index) to the new code.
11. The branch is not merged before stage D: the backend's live executor (`aeroviz_backend/autopilot_segment/`) and the
    frontend's Training view read the old format until then. The user merges.

### 14.2 Stage A: vocabulary, labeller, identities, executor, judge, replay

**A0. Archive.** Move to `archive/two_tier_v3_2026_10/`, unmodified and with their tests, the modules that belong to
the old vocabulary and that stage A does not rewrite. Write a `README.md` there: what moved, why (D20), the last commit
before the move, which stage brings each part back. The list:

- `prior/` (the whole package; stage B brings it back, rewritten);
- every `experiments/prior_*.py` and `experiments/traffic_*.py`, `experiments/window_training_export.py`,
  `inference/separation_masks.py`, `inference/scene_edges.py` (stages B and C);
- the Training exports: `experiments/instruction_training_export.py`, `experiments/executor_training_export.py`,
  `experiments/training_attitude.py`, `instructions/training_files.py`, `instructions/display.py` (stage D);
- the readouts of old artefacts: `experiments/go_around_census.py` (R40; A2 moves its detection rule into the
  labeller), `experiments/instruction_word_frames.py` (R48), `experiments/instruction_final_approach.py` (R49),
  `experiments/heading_lead_ablation.py` (R23).

Remove their names from `experiments/__main__.py` `NOT_RUNNERS` where they occur. If a later milestone breaks another
module that stage A does not rewrite, archive it in that milestone and add it to the README. `tests/test_architecture.py`
must pass. Note in the README that the backend tests under `aeroviz_backend/tests/` that use the two-tier code
(`test_single_executor.py`, `test_autopilot_segment.py`, `test_http_server.py` where it calls that route) fail until
stage D.

**A1. Vocabulary** (§3; `instructions/spec.py`, `words.py`, `grammar.py`).

- Columns: `("runway", "heading", "altitude", "angle", "speed")`. The approach column and its constants go.
- Runway column: the candidate pointer plus the value "go-around". The states R and G and the table of §3.2.
- Heading column: class k is the relative heading 5k° from the course of the runway in force (D8). `Words` gets the
  conversions both ways with the course as an argument; there is no absolute heading class any more.
- Altitude column: the grid of D22 (§3.4): 40 levels on three uniform segments, with the steps and break points as
  spec constants; `Words` maps a height to the nearest level and a level to its height, and gives the ε of a level's
  segment. The value "descend to land" becomes "no level-off", the index after the levels.
- Angle and speed columns: unchanged.
- Grammar rules 1–6 of §3.7 and the runway/G table of §3.2, as one function that the labeller checks and the speaker
  masks with. There is no runway lock (D12).
- Spec: reading `instruction-v4`, schema `ts-instruction-spec-v5`; the spec sha over the format fields, as now.
- Tests: the conversion relative ↔ absolute at several courses (KSTL 122.3°, wrap at 0°/360°); every row of the
  runway/G table, permitted and refused; rules 5 and 6; the first step.

**A2. Labeller** (§4; `instructions/labeller/`).

- Heading words (§4.3): the per-row reading in the frame of the landed runway, from row 0 to the last row. No clearance
  placement, no capture-turn check. `capture_row` stays (for "unspecified" and the readout groups).
- Go-around (§4.6, D18, D19): move the detection rule of `experiments/go_around_census.py` (the low pass, the levels
  held 20 s, 150 m below and above) into a new module `instructions/labeller/go_around.py` with its tests (move the
  relevant tests from `tests/test_go_around_census.py`). Say "go-around" at the first row of the climb (the row of the
  climb word, D26); read the climb as altitude and angle words; say the runway of the next approach at the first
  level-off after the climb, not later than the next "no level-off". Read each approach separately (D26, §4.6 item 6):
  "no level-off" for the last descent that reaches the go-around row, a capture row and "unspecified" for each approach,
  the runway of the first low pass at the first row. A flight that `instruction-v3` refused only because it crossed and came back is read with this
  rule. The landing cut is the last qualifying crossing, not one that a go-around follows.
- Altitude and angle (§4.4): the level test no longer uses the grid (rows within 25 m of the piece's own median, at
  least 20 s); the word is the nearest level; level pieces with the same word merge; a move between two levels with the
  same word gives no level word; the tube uses the ε of the word's segment; the name "no level-off".
- Speed (§4.5): "unspecified" starts at the capture row, with the two exceptions re-anchored (D4).
- Assembly (§4.7): five columns; the grammar of A1 checked on the 2 s rows.
- Row interval (§4.8, D11): a pure function that puts a 2 s sentence on a grid of Δ (a multiple of 2 s, rows on UTC
  multiples of Δ), keeping the last word of a column inside an interval, the first Δ row saying all five columns, and
  the grammar checked again. It refuses a Δ that does not divide 16 s (D25). Its tests: Δ = 2 gives the sentence back;
  Δ = 4 and 8 on hand-built sentences; Δ = 6 is refused.
- Candidates: refuse a candidate without TCH, glidepath angle or LPV DA before labelling (§4.2).
- Tests: a level at the middle of two 60 m levels is found as level; a slow descent inside one 450 m step is not a
  level; synthetic flights (`tests/support.py` `fly_legs`, `instruction_flight`) for a downwind–base–final with
  heading words to the end; a go-around and a second approach (the go-around row, the runway word's row, rule 5);
  "unspecified" at the capture row with each exception.

**A3. Artefact and identities** (§9.2, D21; `instructions/artefact.py`, `instructions/measure.py`, runners).

- Remove `labeller_source_sha256`, `require_current_labeller` and the labeller check in `load_sentences`.
- Labeller conformance: when `instruction_labels` writes the artefact, it also writes `conformance/`: a fixed sample of
  250 labelled train flights (seed 1337, every airport) with their signals and their words. A new runner
  `instruction_conformance` labels them again with today's code and requires the same words and the same refusals,
  word for word; it writes `conformance/passed-<code>.json`, where `<code>` is a digest of the labeller's code by its
  logic (as `autopilot/spec.py` does for the executor). Any runner that labels, or reads sentences for a replay,
  requires a passed record for today's code.
- Arrival manifests: remove the byte checks of the manifest (`arrival_manifest_sha256s` stays as recorded information
  only). `autopilot/flights.py` `require_same_flight` stays.
- Sentences schema `ts-instruction-sentences-v3`: five columns; per sentence the runway, the capture row, the go-around
  rows; no `join_row`.
- The spec measurement (D15): for the descent centres and edges and the climb centre, write the fitted value and the
  candidates rounded to 0.5°, 0.25° and 0.1°, each with the end-of-piece height error that it leaves (the existing
  `fit_descent_classes` error, and the same for the climb pieces). Also write the distribution of the climb angles
  (D15, D28). The spec keeps the fitted values until the user chooses.
- The spec measurement also writes the rounding error of the level words (row 0 apart, p50 / p95 / largest) under the
  grid of D22 and under a uniform 30 m grid, on the train flights, so that §11.7 comes from a runner.
- Runners kept and changed: `instruction_signals`, `instruction_spec`, `instruction_labels`, `instruction_figures`.
- Tests: the conformance passes on a tmp artefact and fails when a labeller rule changes; an old artefact is refused by
  its schema name.

**A4. Executor** (§5; `autopilot/`).

- Remove the laws of §5.7: the runway lock (`Executor.runway_locked`), the capture state and turn, the heading bend and
  own intercept, the centreline tracking, every special law of "descend to land", the glidepath floor and the level
  flight below the glidepath. The executor laws no longer read the TCH or the glidepath (`runway_data` stays for the
  judge).
- Heading: convert each heading word with the course of R at the hearing time; keep the absolute target until the
  next heading word (§3.3, §5.4).
- Go-around (§5.4–§5.6, D27, D28): G from the runway column; while G is true a climb word climbs at the go-around
  angle (the thrust-limited steady climb angle within 1.885°–3°); the heading word in force stays; "unspecified"
  holds the airspeed; a runway word ends G. No mode climbs without an altitude word.
- Vertical modes of §5.5; speed law unchanged (§5.6).
- Row interval: the executor hears the words at the start of each row of Δ (the word clocks of `autopilot/sentence.py`
  take Δ, not the spec's 2 s).
- Port every change to the single-flight executor `autopilot/single.py` (the conformance flies it too).
- Tests: the heading conversion and a runway change that does not turn the aircraft; a go-around (the go-around angle,
  the heading word in force kept, end by a runway word); "no level-off" flies the class angle without a level-off;
  nothing captures or locks.

**A5. Judge** (§5.8; `autopilot/judge.py`).

- The outcomes of the §5.8 table, in the order of the table: `dynamics_failure`, `ground_contact`, `crossed_too_high`,
  `crossed_off_runway`, `unstable_at_minimums`, `landed`, `crossed_other_runway`, `timeout`. The four crossing outcomes
  of R need an approach crossing (G false, lined up within 30°, inside the landing screen). `crossed_without_capture`
  goes.
- A crossing while G is true is not an event. The time limit grows by 900 s at each go-around
  (`Executor.extend_time_limit`).
- The DA check: at the DA point, the lateral offset against the FAS cone at that distance
  (`flight_scenarios/fas_geometry.py`) and the height against the published glidepath, straight-line reference
  (§11.4). The bounds of D38: ±22 m imported from `evaluation/thresholds.py`, and the FAS cone itself. They are not
  parameters.
- Layer 2: the heading words to the end of the flight; the clearance and corridor checks go.
- Tests: each outcome on a hand-built flown track; the order of two events at one row; a go-around low pass that is not
  an event; the DA check pass and fail.

**A6. Executor spec, conformance and replay.**

- `executor_spec`: schema `ts-executor-spec-v7`; the parameters and the conformance reference
  (250 flights) as now; `executor_conformance` as now (§9.2 #3).
- `executor_replay`: a new option `--row-interval-s` (default 2; a multiple of 2): the sentences go through the A2
  projection before they are flown. The summary gives each outcome, the words inside their envelopes (with the
  envelope widths), the go-around sentences apart, for each airport and pooled. It requires the labeller and the
  executor conformance records.
- Smoke only (14.1 rule 7): build a smoke artefact (for example 30 flights for each airport) in a scratch directory,
  pass both conformance checks, and run the replay at Δ = 2 and 4 to its end. No criterion is read.

**A7. Close of stage A.** The full ts suite passes (run detached). §0.3 and §13 are updated. Report to the user: the
commits, the archive list, the smoke results (as information, not as a verdict), and what stage B needs.

**A8. Go-around reading and the go-around in the executor (D26–D28).** First merge the `dev-two-tier` commit that
holds this milestone into `dev-two-tier-v4`.

- Labeller (§4.2, §4.4–§4.6): read each approach separately. `vertical.py`: the last descent that reaches a go-around
  row says "no level-off". `lateral.py` `capture_row`: one per approach, to the end of the approach; an approach that
  ends at a go-around row outside the corridor has no capture row and is not refused for it. `speed.py` `read_speed`:
  one reading per approach. `read.py`: the first row says the runway of the first low pass; the go-around row is the
  first row of the climb; a go-around without a climb word is refused, with its reason.
- Grammar (§3.7): rule 6, in the one function that the labeller checks and the speaker masks with.
- Executor (`autopilot/`, and `autopilot/single.py`): remove "fly the course of R after a go-around" and the climb with
  "no level-off" in force; the go-around angle of §5.5 (γ_T from the dynamics at each cycle, limited to 1.885°–3°);
  the nominal climb angle while G is false. "Unspecified" holds the airspeed while G is true, as now.
- Names: keep the names of §14.1 rule 5. No formal artefact exists yet. Delete the smoke artefact of A6 (it was made
  by superseded code) and build it again.
- Tests: a synthetic flight with a go-around and a second approach: "no level-off" and "unspecified" in force at the
  go-around row; "go-around", the level word and "climb" in one row; the first row's runway is the low-pass runway when
  the flight lands on another runway; rule 6 refuses a go-around row without a level while "no level-off" is in force.
  Executor: the heading word in force stays after "go-around"; the go-around angle is 3° with ample thrust, γ_T between
  the limits, and a 1.885° reference when the thrust is short; "go-around" alone starts no climb.
- Smoke: the replay at Δ = 2 and 4 to its end; each labelled go-around flies as a go-around (G, the climb from its
  level word, a new runway word, a landing after it). A readout of the smoke artefact gives, for each go-around row,
  the altitude and speed words in force (the counts of §11.6 for the new labeller), as information.
- Then the full ts suite again (as A7), and the report.

**A9. Closed-loop reading (D32, D33).** After A8.

- Module in `autopilot/` (§4.9, "Where the code goes"): for one flight, it flies the sentence of a row interval with
  the executor from the first predicted step and returns the closed-loop sentence (correction words marked) and the
  flown states on the rows. The comparison, the lateral and the vertical corrections exactly as in §4.9.
- Spec: the two tolerances Y = 30 m and H = 15 m are labeller constants of the instruction spec (D32).
- Runner: after `executor_spec`, a runner writes the closed-loop sentences and the flown states for each split and each
  Δ of the ablation into the artefact, with the counts of correction words and the flights without a training sentence
  by reason (§4.9, "Artefact"). Sentences schema: a new name (principle 8).
- Labeller conformance (§9.2 #2): it covers the closed-loop reading, with the artefact's executor spec.
- Judge (D33): while G is true, no crossing is an event, of R or of another candidate.
- DA check (D38): the vertical bound imported from `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`), the
  lateral bound the FAS cone at the DA distance; the two required DA parameters of the executor spec go (A5 built them).
  Tests: a flight 21 m above the glidepath at the DA point passes, 23 m fails; inside and outside the cone.
- `executor_replay` flies the closed-loop sentences. It reports, besides the outcomes: for each flight, the largest
  |e_y| and |e_h| against the observed path, and the correction words for each column. Replaying a closed-loop sentence
  gives its flown states again (a test).
- Tests: a synthetic flight whose open-loop words leave a lateral offset gets heading corrections of one class toward
  the path and ends within the tolerance; the correction ends below Y / 2 and at a change of sign; a new observed word
  ends a correction; a final descent that is too high gets the next steeper class, and the observed class again; no
  vertical correction during a level hold or a climb; a path that crosses itself keeps the forward matched point; a
  flight without dynamics gives no training sentence and is counted; the rows before the first predicted step stay
  observed; a crossing of another runway while G is true is not an event.
- Smoke: the closed-loop reading at Δ = 2 and 4 on the smoke artefact; the replay of its sentences; as information,
  the outcomes, the largest errors against the observed path and the correction words for each flight.
- Then the full ts suite again, and the report.

**A10. Observed words at the place (D42).** After A9.

- `autopilot/closed_loop.py`: at each Δ row, the observed words of each open-loop Δ row whose observed time is not
  later than the observed time of the matched point, and that were not said before; when one Δ row passes more than one
  of these rows, the last word of each column (§4.9, "When the observed words are said"). One matched point for each
  row serves the comparison and the words (`ObservedPath.match`, the forward search). The corrections do not change: a
  row that says a new observed word of a column ends a correction of that column, as before.
- The flight ends when the executor is done or at the time limit of the replay (§5.8: the remaining observed time ×
  1.5, plus 900 s for each go-around), not at the last row of the open-loop sentence. The closed-loop sentence has the
  flown rows.
- The replay of a closed-loop sentence does not change: the sentence is said on its own rows and gives its flown states
  again (the test of A9).
- The closed-loop conformance (§9.2 #2) gets a new reference from the new code. The closed-loop sentences of the A9
  smoke build come from superseded code: delete them and write them again.
- Tests: a flown aircraft behind the observed one hears a turn word where the observed aircraft heard it, not at the
  observed time, and ends within Y of the path; a flown aircraft ahead passes two observed heading words in one Δ row
  and hears the last one; the observed words wait while the flown aircraft is behind; the first predicted step says
  every column; a flown aircraft ahead of the observed one hears a go-around word before the threshold where the
  observed aircraft went around before it (synthetic flights); a flight longer than the observed time is not cut at the
  end of the open-loop sentence.
- Smoke: the closed-loop reading at Δ = 2 and 4 and the replay of its sentences (train 30 flights for each airport,
  select every flight). As information: the readouts of A9 and, for each stratum, the number of flights that leave the
  observed path by more than 300 m and the correction words for each sentence (compare §11.11).
- Then the full ts suite again, and the report.

### 14.3 Stage B: prior

**Start.** After Claude's check of stage A (§14.6). The code is written and tested on the smoke artefact at Δ = 2 s.
The formal runs of B5 need the formal artefact (§0.4 item 3) and the chosen Δ. The rules of §14.1 apply.

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
- A smoke run on the smoke artefact; it gives the time of one run for the cost of B5.
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

### 14.6 What Claude checks at the end of stage A

1. Each decision that stage A carries (D1–D22, the Δ values of D25, D26–D28, D32, D33, D38 and D42) against the code: the module and the
   test that carry it (a table in the report).
2. The targeted tests and the full suite pass on the branch head (run again, not read from the report).
3. The smoke build: both conformance checks pass; the replay at Δ = 2 and 4 runs to its end; a labelled go-around
   flies as a go-around (G, the climb from its level word, a new runway word, a landing after it); at each go-around
   row, "no level-off" and "unspecified" are in force where the approach reached them (D26); the closed-loop sentences
   at Δ = 2 and 4 replay to their flown states, and their flown paths stay within the tolerances of the observed
   paths except where §4.9 permits no correction (D32); each observed word of a closed-loop sentence comes at the first Δ
   row at which the matched point reaches it (D42), and a labelled go-around is in its closed-loop sentence and flies as
   a go-around.
4. No write under a live root, and no existing directory under `4dTrajectory/outputs/` changed.
5. Nothing in the archive was edited after the move.
