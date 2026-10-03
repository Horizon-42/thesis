# Two-tier model: design of the vocabulary, the labeller and the executor

**Summary.** The two-tier model has two parts. The upper part is the prior: a language model of the controller. Every
2 s it says one row of words for each aircraft. The lower part is the executor: an autopilot that flies only these
words with point-mass dynamics. This document gives the next design of three parts: the words (the vocabulary), the
program that reads words from observed tracks (the labeller), and the executor. It gives only an outline for the
prior, the post-training and the multi-aircraft work.

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
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The LPV minimums are a judgement only. The model must learn to land or to go around | Decided. It replaces "the executor tracks the centreline after capture" (decided earlier the same day) | User, 2026-10-03 |
| D4 | The speed value "unspecified" starts at the capture row of the labeller, not at a clearance | Decided | User, 2026-10-03 |
| D5 | Prior inputs are in the frame of the runway in force. The prior has no airport embedding, no absolute position and no absolute direction | Decided | User, 2026-10-03 |
| D6 | The grids of the words (heading 5°, four descent classes) stay as they are now. A measurement or a rough labelling decides if they are sufficient | Decided | User, 2026-10-03 |
| D7 | This document sets no test criteria and no measurement criteria. The user sets them after the design is settled | Decided | User, 2026-10-03 |
| D8 | The heading grid is aligned to the course of the runway in force (§3.3) | Decided | User, 2026-10-03 |
| D9 | The executor laws in §5.7 are removed, the glidepath floor included | Decided | User, 2026-10-03 |
| D10 | Go-around is an event: "abandon this approach". It does not change the runway in force; a runway word ends it. Its effects on the executor, the judge and the masks: §3.2 | Decided | User, 2026-10-03 |
| D11 | The labeller stage includes an ablation of the row interval: 2 s now, larger intervals possible (§4.8) | Decided. Values: D25. Criteria: O10 | User, 2026-10-03 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided (the user considered a lock that a go-around lifts, and chose no lock) | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§6.1) | Decided (was O3) | User, 2026-10-03 |
| D14 | While G is true: "no level-off" is not permitted (rule 5), and the procedure mask "no climb below the entry height" does not apply (§3.7) | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§6.1) | Decided (was the first half of O4) | User, 2026-10-03 |
| D17 | Every column keeps the input "time since this column said its word in force", in seconds; the runway column's too (§6.1) | Decided (was O4) | User, 2026-10-03 |
| D18 | The labeller reads the real go-arounds inside a sentence and says "go-around" at the go-around point (§4.6) | Decided (was O5) | User, 2026-10-03 |
| D19 | After a labelled go-around, the runway word that ends G is at the first level-off after the go-around climb, and not later than the row of the next "no level-off" (§4.6) | Decided | User, 2026-10-03 |
| D20 | This design is a new version: it is developed on a new branch in a new worktree. Artefacts `v1`–`v6`, the executor specs and every prior are superseded and are not kept readable. The running experiments keep their own checkouts | Decided | User, 2026-10-03 |
| D21 | Identities bind format and data rules only. A code identity is a behaviour check on fixed inputs, never a hash of source bytes; data are identified by their flights, never by the bytes of a manifest (§9.2) | Decided (was O13) | User, 2026-10-03 |
| D22 | Altitude words use the grid "optimal 40 levels" of the altitude-grid proposal: 60 m steps from 0 to 1,260 m, 120 m steps to 2,700 m, 450 m steps to 5,400 m (§3.4) | Decided | User, 2026-10-03 |
| D23 | How the prior gets the frame of R (D5) and the glidepath height (D13). The own state of the aircraft has no frame. Every position and direction is in the candidate vectors, each in the frame of its own candidate. R is an input only as its candidate vector. No input of a row up to the first predicted step is computed from R. Each candidate vector has the height above its own glidepath; the value of R is the input of D13 (§6.1) | Decided | User, 2026-10-03 |
| D24 | A candidate vector has no constant of its runway: no length, no elevation, no layout relative to R. Such a value comes back only as a variant that O7 selects (§6.1) | Decided | User, 2026-10-03 |
| D25 | The ablation reads Δ = 2, 4, 8 s: each divides the 16 s observation of the prior. At each Δ, the motion inputs of a row come from the 2 s before the row (§4.8, §6.1) | Decided (the values of O10) | User, 2026-10-03 |

### 0.2 Open items, in the order of discussion

| # | Item | Proposal | §  |
|---|---|---|---|
| O2 | Tolerances of the stability check at the decision altitude (DA) | No values in this document (D7) | 5.8 |
| O6 | Replacement for the clearance mask of the multi-aircraft loop | Discuss with §8 | 8 |
| O7 | Selection of designs by leave-one-airport-out (train on four airports, read the fifth) | Discuss with §7 | 7 |
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |
| O9 | Finer grids near the runway course and near the glidepath angle | Decided by D6: measure first | 3.3, 3.5 |
| O10 | How to compare the row intervals of the ablation (the values are D25: 2, 4, 8 s) | Criteria by the user after the design is settled (D7) | 4.8 |
| O12 | Climb classes: keep one, add more, or set the climb to the missed-approach gradient | Proposal: keep one climb class; its value from the measurement with D15; a second class only if the climb angles have two clear groups | 3.5 |

### 0.3 Implementation

| Part | State |
|---|---|
| Stage A: vocabulary, labeller, identities, executor, judge, replay (§14.2) | Built on `dev-two-tier-v4`, each milestone reviewed: A0 `e57c62d8` + `d2917c06`; A1–A3 `63639a54` + review fixes `feaf9558`, `d1e277f7`; A4–A6 `330ffbaf` + review fixes `c7603a4c`; A7: this section, §13, and the full ts suite (1,469 passed at `c7603a4c`). Smoke build (80 flights an airport and split, in the ignored `smoke_v4/data/` of the worktree): both conformance checks pass; the replay ran to its end at Δ = 2 and 4 s. Claude's readings that the user should confirm: listed in the stage A report (level band at a segment's top, the grammar's held height, "no level-off"'s band, LNAV/VNAV DAs, the DA lateral tolerance as a share of the cone). Claude checks the result (§14.6) |
| Stage B: prior (§14.3) | Not started; outline |
| Stage C: post-training and multi-aircraft (§14.4) | Not started; outline |
| Stage D: frontend and backend (§14.5) | Not started; outline |

### 0.4 Plan

1. Stage A (§14.2): write the vocabulary, the labeller, the identities, the executor, the judge and the replay on the
   branch `dev-two-tier-v4`, milestone by milestone, each with tests and a code review. Another agent does this.
2. Claude checks the result of stage A against this document (§14.6).
3. The user sets the criteria (D7, O10) and the values of O2 and O12. Then the formal artefact is built, and the
   replay gate is read at each row interval of the ablation (D11, D25: 2, 4, 8 s). The same readings answer D6 / O9 (are the grids sufficient).
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
| Capture row | The labeller's first row from which the observed track stays in the capture corridor to the threshold. Only the labeller uses it (D4). It uses later rows, so it is never an input |
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

**Meaning of a runway word.** "Expect runway k." The executor uses R for three things only: the conversion of the
heading words (§5.4), the course of a go-around (§5.4), and the distance for the deceleration to the approach speed
(§5.6). The judge uses R for the landing (§5.8). The prior gets R as the candidate vector of R (§6.1, D23). The multi-aircraft
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
| Executor (§5.4, §5.5) | Climb at the missed-approach gradient, 200 ft per NM (1.885°, AIM 5-4-21 b). No other word gives it: the climb class is 1.32°. The climb replaces "no level-off" in force. Fly the course of R until a heading word. Hold the airspeed |
| Judge (§5.8) | While G is true, a crossing of the threshold is not an event (not a landing and not a failure): the flight continues. A go-around before the DA point answers a failed DA check. The time limit of the flight grows by 900 s (multi-aircraft design §6.6 step 8, the user 2026-10-02) |
| Masks (§3.7, D14) | Rule 5: no "no level-off" while G is true. The procedure mask "no climb below the entry height" does not apply while G is true |
| Prior (§6.1) | G is an input: the aircraft is in a missed approach |
| Multi-aircraft (§8) | The decision is explicit: it can be counted and rewarded, and the separation judge can treat the aircraft as no longer on the approach (open with O6) |

A runway word ends G. The model can say R again or another candidate ("expect runway k" again). This keeps the step-8
sequence of the multi-aircraft design (approach → go-around → approach) in one column.

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

**Grid (D6, O9).** The grid stays 5°. A measurement or a rough labelling decides if 5° holds the final. A word of class 0
gives no lateral correction. The aircraft keeps its lateral offset until the model says a word of ±5°.

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
altitude, where the step is 60 m.

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
| Climb | 1.32° | 0.5° to 15° climb |

The four descent classes come from a length-weighted k-means on the train days (spec `145d6911e75b`). The grid stays
(D6). With D3 the model holds the glidepath with these classes. A measurement or a rough labelling decides if they are
sufficient (O9). An alternative for discussion: a fixed angle grid, not a k-means result.

**Rounder values (D15).** The spec measurement gives each value that it fits from data (the descent centres and edges,
the climb centre) with rounder candidates (for example to 0.5°, 0.25°, 0.1°) and, for each candidate, the fit that it
leaves: the end-of-piece height error that the measurement already reports (`instructions/measure.py`
`fit_descent_classes`), and the same for the climb pieces. The user chooses (criteria: D7). Two facts for that choice:
24 of the 25 candidate runways publish a 3.0° glidepath (§11.4), and with D3 the model holds the final with an angle
word, so a centre of 3.06° flies 10 m from a 3.0° glidepath in 10 km; the k-means with three classes put a centre at
3.006°. The values that the measurement takes as percentiles are already rounded by their rules (turn rates to
0.1°/s, the bank limit to 1°, the corridor to 5 m, 0.05° and 1°, the acceleration to 0.1 m/s²).

**Climb (O12).** The train days have 1,498 climb pieces among 233,649 vertical pieces (0.6 %); their length-weighted
median is 1.32°. The missed-approach climb is now the go-around's (D10, 1.885°). Proposal: one climb class; its value
from the measurement with D15; a second class only if the climb angles show two clear groups. Setting the climb class to
1.885° is also possible (one climb gradient in the vocabulary, from a regulation, not from data), but 1.885° is a
minimum for a missed approach, and the climbs in the data are shallower.

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

The first row says the runway on which the flight landed (the harvest assignment). The data cannot show a change of
runway, so a labelled sentence never changes R. A candidate without a published threshold crossing height (TCH), a
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
between two levels that round to the same word gives no level word (it cannot be said; §3.4). The last descent to the
threshold gives "no level-off". The tube of §3.4 checks each word, with the ε of its segment.

### 4.5 Speed words

The reading of `instruction-v3` stays, with one change: "unspecified" starts at the capture row, not at the clearance
row (D4). The two exceptions stay, with the new anchor:

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
2. The go-around row is the row of the lowest point. The runway column says "go-around" there; G becomes true.
3. The climb after it is read as altitude and angle words, as every climb. While G is true the executor flies it at
   the missed-approach gradient (§5.5); the observed climbs are steeper (median 9.3 %), inside the climb class's range.
4. The runway word that ends G (D19): at the first level-off after the go-around climb, and not later than the row of
   the next "no level-off" (rule 5). It says the runway on which the flight landed.
5. A flight that the labeller refused only because it crossed a threshold and came back is read again with this rule.

**Not in this version.** The 23 go-arounds before their sentence need a longer arrival slice, and 9 sentences of `v5`
end at a low go-around that the harvest took for the landing (it takes the best-aligned crossing under 100 m, not the
last; R40 v2 §5). Both are changes of the harvest or of the data plane, which other lines (evaluation, the optimizer,
the one-tier models) share.

### 4.7 Assembly and artefact

- All words go on the row grid (2 s now; §4.8). A word equal to the word in force is not a word. The first row says all five columns.
  Two different words in one column in one row: refused.
- The artefact has a new reading name and a new spec format (principle 8). The spec records the grids, the classes, the
  tolerances and the reading. The labeller identity hash covers the reading code and `grammar.py`.
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

**Values (D25).** Δ = 2, 4, 8 s. A Δ of 6 s is not used: 16 s is not a whole number of 6 s rows. The user sets the
criteria after the design is settled (O10, D7).

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ. At Δ = 2 s it is the value of now.

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
- **Go-around.** After the word "go-around", the executor flies the course of R until the model says a heading word at
  or after the go-around row. Then it flies that word (shorter direction from the present track).
- Nothing else. There is no capture state, no turn onto the final, no centreline tracking (§5.7).

### 5.5 Vertical law

The inner path-angle loop stays: γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max), τ_γ = 2 s (executor design §5.1). The modes:

| Words in force | Mode | γ_ref |
|---|---|---|
| Level T + level | Hold | Altitude hold law, τ_h = 8 s (executor design §5.5) |
| Level T + descent class | Descend to T | −nominal angle of the class; level-off starts at V·γ²/(2·γ̇_max) above T |
| Level T + climb | Climb to T | +1.32°; level-off as above |
| "No level-off" + descent class | Descend | −nominal angle of the class; no level-off |
| G true, with "no level-off" or a higher T in force | Go-around climb | +1.885° (200 ft per NM, AIM 5-4-21 b): with "no level-off", until a new altitude word; with a higher T, to T, then hold |

### 5.6 Speed law (unchanged)

V_ref = V_g / cos γ (ground speed to airspeed without wind), not below 1.10·V_stall(n). The speed changes at
a = 0.25 m/s² (a speed step of 5 m/s divided by the minimum hold of 20 s) and goes exponentially into the last 5 m/s.
"Unspecified" is the published approach speed of the type at the published maximum landing mass. The deceleration to it
is the larger of a and the rate that reaches it at the threshold of R, not more than 1.4 m/s². During a go-around the
executor holds the airspeed.

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
| `crossed_other_runway` | The threshold plane of another candidate crossed lined up, inside that runway's own limit, at any height |
| `timeout` | None of these within the time limit (the remaining observed time × 1.5, plus 900 s for each go-around) |

A crossing that is not lined up (for example, abeam the threshold on a downwind) is not an event. A crossing of R while G
is true is not an event: the flight continues. `crossed_without_capture` goes, because the executor has no capture
state.

**The DA check (D3, O2).** At the DA point the judge checks that the aircraft is stable: lateral offset inside the FAS
cone at that distance, height inside a tolerance of the published glidepath of R (TCH + d·tan(glidepath angle)). The
tolerances are open (D7). The regulation basis: the missed approach starts at the DA (AIM 5-4-21 b: "Obstacle
protection for missed approach is predicated on the missed approach being initiated at the decision altitude"). If
the model says "go-around" before the DA point, the flight continues (§3.2).

**Quality** stays with the evaluation module (lateral, vertical and speed gates). The judge does not repeat it.

---

## 6 Prior (outline)

### 6.1 Inputs

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
  geometry and stays; only O7 can measure how much the prior uses it. A constant comes back only as a variant that O7
  selects.
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
  labelled sentence says its runway only there. In closed loop it is the time since the runway was given or given again
  (a change of runway, or the runway word that ends a go-around), so it measures a real fact.

### 6.2 Outputs

Five heads in the order of §3.1. The runway head scores each candidate (pointer, as now) plus "unchanged" and
"go-around". The first predicted step masks "unchanged" in each column and "go-around" in the runway column.

### 6.3 Training

Teacher forcing on the new artefact, split by operating day (`data/day_split_20260924.json`; test days sealed). KAUS is
a held-out test airport only. The selection method between designs is open (O7).

---

## 7 Post-training (outline)

1. Landing reward in the single-aircraft closed loop (the clipped-ratio surrogate, the pull to the base model, the
   teacher-forced data term), as `prior_landing_reward` now. The reward reads the outcomes of §5.8.
2. Procedure masks (glidepath lower edge inside the FAF, DA before the join, no climb back) and augmented starts, as
   stage 2 now. The masks stay with the model (`procedure_masks.json`, contract C35).
3. Go-around: a reward for a go-around that the DA check needs (§5.8). The step-8 machinery (probes, a group advantage)
   moves to the runway column.
4. Selection: on the select days; the validation days are read one time for each stage. O7 asks for leave-one-airport-out
   as the selection method for the airport generalization.

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
  runway at every step. In `instruction-v3` this was also true; this design keeps it.
- The clearance mask (`inference/separation_masks.py:131`) blocks "cleared" while a cleared aircraft ahead is too close.
  This design has no "cleared" word. A replacement is open (O6).
- The speed-word mask stays.
- The traffic attention (initial output zero) on the single-aircraft prior stays as the start of the multi-aircraft
  post-training.

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
| 2 | Labeller source hash over bytes (`instructions/artefact.py:132`, checked by `require_current_labeller` and `load_sentences`) | Remove. Replace it by the labeller conformance (§14.2 A3): a fixed reference sample labelled again by today's code gives the same words. The artefact records which code wrote it as information | A |
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
| Climb class | 0.5–15°, nominal 1.32° | Spec |
| Go-around climb | 1.885° (200 ft per NM) | AIM 5-4-21 b |
| Speed grid, range, tolerance | 5 m/s, 20–250 m/s, ±5 m/s | Spec |
| Speed change rate (executor) | 0.25 m/s² | 5 m/s ÷ 20 s (user, 2026-09-24) |
| Transition acceleration limit | 1.4 m/s² | Spec, measured (p99.9) |
| Capture corridor (labeller only) | 20 m + d·tan 0.45°, course ±2° | Spec, measured (p99) |
| Landing screen | lateral ≤ 1,000 m and ≤ half the parallel spacing; height ≤ 100 m | `final_approach.assign.LandingScreen` |
| Landed lateral limit | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| DA check tolerances | open | O2, D7 |
| Time limit | remaining observed time × 1.5 | Fixed choice |

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

---

### 11.7 The altitude grid

From the proposal [altitude_word_grid.zh.md](altitude_word_grid.zh.md) (`v6_20261002`; fit on 11,937 train flights,
read on 3,964 validation flights; continuous level-off heights). Rounding error of the level words other than row 0,
p50 / p95 / largest: the chosen grid (40 levels) 17 / 30 / 58 m; the same shape with steps doubling and break points at
1,200 and 2,400 m (44 levels) 17 / 38 / 79 m; a uniform grid of 48 levels over the range 35 / 55 / 57 m. The
`instruction-v3` grid (30 m, 182 classes) has at most 15 m. These numbers come from a one-off script of that proposal;
the spec measurement of stage A gives them again from a runner (§14.2 A3). The proposal did not fly the grid: the replay
of stage A does.

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

The code of this design (stage A). Line numbers are at the end of stage A on `dev-two-tier-v4`; paths are relative to
`4dTrajectory/ts_transformer/`.

| What | Where |
|---|---|
| Columns; "go-around" in the runway column | `instructions/words.py:20` `COLUMNS`, `:24` `RUNWAY_GO_AROUND` |
| Heading classes relative to the course of R | `instructions/words.py:80` `heading_class`, `:84` `heading_track_deg` |
| The 40 levels, their bands | `instructions/words.py:47` `altitude_tolerances`, `:89` `altitude_index` |
| Grammar: rules 1–5 and the runway/G table, one function | `instructions/grammar.py:54` `apply`, `:117` `runway_words_allowed` |
| Capture row; heading words to the end, relative frame | `instructions/labeller/lateral.py:38` `capture_row`, `:57` `per_step_words`, `:90` `read_lateral` |
| Level test by the piece's own median; held height; tubes | `instructions/labeller/vertical.py:57` `vertical_pieces`, `:78` `held_altitude`, `:187` `tube_bounds`, `:218` `tube_checks` |
| "Unspecified" from the capture row | `instructions/labeller/speed.py:86` `read_speed` |
| Go-around rule (R40's, on the rows) | `instructions/labeller/go_around.py:61` `low_passes`, `:108` `go_arounds` |
| Landing cut, go-arounds in the gate, runway words (D19) | `instructions/labeller/read.py:106` `admit`, `:147` `flight_go_arounds`, `:166` `runway_words`, `:192` `read_flight` |
| Row interval (D11, D25) | `instructions/labeller/interval.py:45` `first_interval_row`, `:69` `on_interval` |
| Labeller conformance (§9.2 #2) | `instructions/conformance.py:59` `labeller_code_sha256`, `:148` `check`, `:202` `require_conforming_labeller` |
| Heading envelope | `instructions/envelope.py:31` `heading_words_inside` |
| Straight-line glidepath (D13, the DA check) | `instructions/airport.py:183` `glidepath_height_m` |
| Heading law; conversion with the course of R; go-around course | `autopilot/lateral.py:54` `word_rate`, `:138` `Lateral.word_error`, `:172` `Lateral.rate` |
| Vertical modes; go-around climb | `autopilot/vertical.py:43` `GO_AROUND_CLIMB_GRADIENT`, `:89` `Vertical.rate` |
| The cycle; go-around time; approach crossing ends the flight | `autopilot/executor.py:60` `GO_AROUND_EXTRA_S`, `:160` `Executor.cycle` |
| R and G per row | `autopilot/sentence.py:89` `_filled` |
| Outcomes, their order; the DA check | `autopilot/judge.py:74` `OUTCOMES`, `:77` `EVENT_ORDER`, `:135` `decision_check`, `:158` `_outcome`, `:319` `judge` |
| TCH, glidepath, DA read by the judge | `autopilot/runway_data.py:25` `VerticalPath` |
| A sentence on Δ; the readout | `autopilot/replay.py:77` `sentence_on_interval`, `:367` `summary`; `experiments/executor_replay.py:99` `readout_table` |
| Clearance mask (stage C) | archived: `archive/two_tier_v3_2026_10/inference/separation_masks.py` |
| Prior: airport embedding, inputs (stage B) | archived: `archive/two_tier_v3_2026_10/prior/model.py`, `prior/data.py` |
| FAS cone; DA above the threshold | `flight_scenarios/fas_geometry.py:46` `fas_course_geometry`; `trajectory_data_process/harvest/airports.py:169` (repository root) |

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
- Grammar rules 1–5 of §3.7 and the runway/G table of §3.2, as one function that the labeller checks and the speaker
  masks with. There is no runway lock (D12).
- Spec: reading `instruction-v4`, schema `ts-instruction-spec-v5`; the spec sha over the format fields, as now.
- Tests: the conversion relative ↔ absolute at several courses (KSTL 122.3°, wrap at 0°/360°); every row of the
  runway/G table, permitted and refused; rule 5; the first step.

**A2. Labeller** (§4; `instructions/labeller/`).

- Heading words (§4.3): the per-row reading in the frame of the landed runway, from row 0 to the last row. No clearance
  placement, no capture-turn check. `capture_row` stays (for "unspecified" and the readout groups).
- Go-around (§4.6, D18, D19): move the detection rule of `experiments/go_around_census.py` (the low pass, the levels
  held 20 s, 150 m below and above) into a new module `instructions/labeller/go_around.py` with its tests (move the
  relevant tests from `tests/test_go_around_census.py`). Say "go-around" at the row of the lowest point; read the climb
  as altitude and angle words; say the landed runway again at the first level-off after the climb, not later than the
  next "no level-off". A flight that `instruction-v3` refused only because it crossed and came back is read with this
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
  (O12). The spec keeps the fitted values until the user chooses.
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
- Go-around (§5.4, §5.5): G from the runway column; the climb at 1.885° (`GO_AROUND_CLIMB_GRADIENT`) for "no
  level-off" or a higher level; the course of R until a heading word; hold the airspeed; a runway word ends G.
- Vertical modes of §5.5; speed law unchanged (§5.6).
- Row interval: the executor hears the words at the start of each row of Δ (the word clocks of `autopilot/sentence.py`
  take Δ, not the spec's 2 s).
- Port every change to the single-flight executor `autopilot/single.py` (the conformance flies it too).
- Tests: the heading conversion and a runway change that does not turn the aircraft; a go-around (climb gradient,
  course, end by a runway word); "no level-off" flies the class angle without a level-off; nothing captures or locks.

**A5. Judge** (§5.8; `autopilot/judge.py`).

- The outcomes of the §5.8 table, in the order of the table: `dynamics_failure`, `ground_contact`, `crossed_too_high`,
  `crossed_off_runway`, `unstable_at_minimums`, `landed`, `crossed_other_runway`, `timeout`. The four crossing outcomes
  of R need an approach crossing (G false, lined up within 30°, inside the landing screen). `crossed_without_capture`
  goes.
- A crossing while G is true is not an event. The time limit grows by 900 s at each go-around
  (`Executor.extend_time_limit`).
- The DA check: at the DA point, the lateral offset against the FAS cone at that distance
  (`flight_scenarios/fas_geometry.py`) and the height against the published glidepath, straight-line reference
  (§11.4), within two tolerances. The tolerances are REQUIRED parameters of the executor spec, given on the command line
  of `executor_spec`, with no default (O2: the user gives the values).
- Layer 2: the heading words to the end of the flight; the clearance and corridor checks go.
- Tests: each outcome on a hand-built flown track; the order of two events at one row; a go-around low pass that is not
  an event; the DA check pass and fail.

**A6. Executor spec, conformance and replay.**

- `executor_spec`: schema `ts-executor-spec-v7`; the parameters (with the DA tolerances) and the conformance reference
  (250 flights) as now; `executor_conformance` as now (§9.2 #3).
- `executor_replay`: a new option `--row-interval-s` (default 2; a multiple of 2): the sentences go through the A2
  projection before they are flown. The summary gives each outcome, the words inside their envelopes (with the
  envelope widths), the go-around sentences apart, for each airport and pooled. It requires the labeller and the
  executor conformance records.
- Smoke only (14.1 rule 7): build a smoke artefact (for example 30 flights for each airport) in a scratch directory,
  pass both conformance checks, and run the replay at Δ = 2 and 4 to its end. No criterion is read.

**A7. Close of stage A.** The full ts suite passes (run detached). §0.3 and §13 are updated. Report to the user: the
commits, the archive list, the smoke results (as information, not as a verdict), and what stage B needs.

### 14.3 Stage B: prior (outline)

Copy `prior/` out of the archive and change the copy to §6. The archived copy stays unchanged: it is the record of
`instruction-v3`. The changes: an own state without a frame, every position and direction in the candidate vectors,
and no input computed from R up to the first predicted step, with its test (D5, D23); no candidate constant (D24); no
airport embedding; the glidepath height in each candidate vector (D13, D23); the motion from the 2 s before each row
(D25); RoPE with seconds from the aircraft's row 0 and no row embedding (D16); the time since each word in seconds
(D17); the go-around state as an input; five heads with the runway head scoring the candidates, "unchanged" and
"go-around". The observation is 16 s at every Δ (8, 4, 2 rows). The checkpoint identity of §9.2 #5. The procedure masks' word tolerance becomes half
the step of the level's segment (D22; `prior/procedure.py` `word_tolerance_m`). The selection method is open (O7). Free
generation with the masks of §3.7 and the procedure masks (D14). Milestones, tests and reviews as in stage A.

### 14.4 Stage C: post-training and multi-aircraft (outline)

The landing reward and stage 2 (§7) on the new outcomes, the go-around reward (multi-aircraft design §6.6 step 8), the
traffic attention and the window loop (§8), the replacement of the clearance mask (O6), the edge-feature conformance
(§9.2 #6).

### 14.5 Stage D: frontend and backend (outline)

The Training exports and the backend's live executor (`aeroviz_backend/autopilot_segment/`) on the new format; the
frontend reads the reading name, not the spec sha (§9.2 #9). After stage D the user merges the branch.

### 14.6 What Claude checks at the end of stage A

1. Each decision that stage A carries (D1–D22, and the Δ values of D25) against the code: the module and the test that
   carry it (a table in the report).
2. The targeted tests and the full suite pass on the branch head (run again, not read from the report).
3. The smoke build: both conformance checks pass; the replay at Δ = 2 and 4 runs to its end; a labelled go-around
   flies as a go-around (G, climb, a new runway word, a landing after it).
4. No write under a live root, and no existing directory under `4dTrajectory/outputs/` changed.
5. Nothing in the archive was edited after the move.
