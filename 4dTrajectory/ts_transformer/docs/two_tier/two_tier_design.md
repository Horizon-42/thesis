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
| D11 | The labeller stage includes an ablation of the row interval: 2 s now, larger intervals possible (§4.8) | Decided. Values and criteria: O10 | User, 2026-10-03 |
| D12 | No runway lock. While G is false the model can change the runway at any time; the judge reads R at the crossing (§3.2) | Decided (the user considered a lock that a go-around lifts, and chose no lock) | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§6.1) | Decided (was O3) | User, 2026-10-03 |
| D14 | While G is true: "no level-off" is not permitted (rule 5), and the procedure mask "no climb below the entry height" does not apply (§3.7) | Decided | User, 2026-10-03 |
| D15 | The spec measurement gives each value that it fits from data with rounder candidates and the fit that each leaves; the user chooses (§3.5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§6.1) | Decided (was the first half of O4) | User, 2026-10-03 |
| D17 | Every column keeps the input "time since this column said its word in force", in seconds; the runway column's too (§6.1) | Decided (was O4) | User, 2026-10-03 |

### 0.2 Open items, in the order of discussion

| # | Item | Proposal | §  |
|---|---|---|---|
| O2 | Tolerances of the stability check at the decision altitude (DA) | No values in this document (D7) | 5.8 |
| O5 | Label the real go-arounds (R40 found 105 on the training days) | Discuss after O2 and O4 | 4.6 |
| O6 | Replacement for the clearance mask of the multi-aircraft loop | Discuss with §8 | 8 |
| O7 | Selection of designs by leave-one-airport-out (train on four airports, read the fifth) | Discuss with §7 | 7 |
| O8 | Speed words in ground speed: wind can make speed words at turns | A check on data, later | 3.6 |
| O9 | Finer grids near the runway course and near the glidepath angle | Decided by D6: measure first | 3.3, 3.5 |
| O10 | The row intervals of the ablation and how to compare them | Proposal: 2, 4, 6, 8 s. Criteria by the user after the design is settled (D7) | 4.8 |
| O12 | Climb classes: keep one, add more, or set the climb to the missed-approach gradient | Proposal: keep one climb class; its value from the measurement with D15; a second class only if the climb angles have two clear groups | 3.5 |

### 0.3 Implementation

| Part | State |
|---|---|
| Vocabulary spec, words, grammar (`instructions/`) | Not started |
| Labeller (`instructions/labeller/`) | Not started |
| Executor and judge (`autopilot/`) | Not started |
| Prior, post-training, multi-aircraft | Outline only (§6–§8) |

### 0.4 Plan

1. Settle the open items (§0.2) with the user. Update this document after each decision.
2. Do the measurement or rough labelling of D6 (the user sets the criteria, D7).
3. Write the vocabulary, the labeller and the executor on a branch in a worktree. Each step gets a code review.
4. Fly the labelled sentences with the executor (the replay gate, §9), at each row interval of the ablation (D11, §4.8).
   The user sets the criteria and chooses the interval.
5. Train the prior from the start, then the post-training, then the multi-aircraft stages (§6–§8).

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
| Row, step | One line of a sentence. A row is Δ = 2 s now, on even UTC seconds (artefact `v6_20261002`); the ablation of §4.8 reads larger Δ |
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
| 2 | Altitude | target level, geometric MSL, 30 m grid, 0–5,400 m (181 values); "no level-off" | 183 |
| 3 | Angle | level; descent 1–4; climb | 7 |
| 4 | Speed | target ground speed, 5 m/s grid, 20–250 m/s (47 values); "unspecified" | 49 |

The approach column of reading `instruction-v3` does not exist (D1). In one row the prior says the columns in the order
of the table. A later column sees the values that the earlier columns said in the same row.

### 3.2 Runway column

**Meaning of a runway word.** "Expect runway k." The executor uses R for three things only: the conversion of the
heading words (§5.4), the course of a go-around (§5.4), and the distance for the deceleration to the approach speed
(§5.6). The judge uses R for the landing (§5.8). The prior uses R as the frame of its inputs (§6.1). The multi-aircraft
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

**Meaning.** "Descend (or climb) to level T and keep it" (7110.65BB 4-5-7, DESCEND AND MAINTAIN). T is geometric MSL, on
a 30 m grid, 0–5,400 m. The direction comes from T and the present height. The angle column says how steep.

**"No level-off".** "Descend at the angle in force. Do not level off." The judge stops the flight at the threshold or
at the ground. This value replaces "descend to land" of `instruction-v3`. The name changes because the meaning changes:
the executor has no special law for it (§5.5, §5.7).

**Why MSL and not height above the threshold.** Controllers assign MSL levels. Relative to the threshold elevation, the
level distributions of the five airports become less similar, not more (§11.2). KSJC has 28 % of its level words on
levels that the other four airports almost never use. This is airspace, not frame.

**Envelope (unchanged).** The tube from the row of the word: max(T, h0 − s·tan γ_hi) − ε ≤ h ≤ max(T, h0 − s·tan γ_lo) + ε
while it descends, T ± ε after it arrives (ε = 25 m; s is the horizontal distance flown from the row of the word). For
"no level-off" there is no lower bound T. A new angle word starts a new tube.

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

### 4.4 Altitude and angle words (unchanged except the name)

Piecewise-linear fit of the smoothed altitude against distance (residual ≤ 10 m). Level segments ≥ 20 s inside a
30 m level ± 25 m. Moving segments give the level words and the angle words. The last descent to the threshold gives
"no level-off" (the old "descend to land"). The tube of §3.4 checks each word.

### 4.5 Speed words

The reading of `instruction-v3` stays, with one change: "unspecified" starts at the capture row, not at the clearance
row (D4). The two exceptions stay, with the new anchor:

1. A hold of at least 30 s that ends at or after the capture row, and ends 9,260 m (5 NM, 7110.65BB 5-7-1 b.4) or
   more before the threshold, keeps the speed word in force. "Unspecified" starts at the end of the last such hold.
2. A deceleration that started less than 20 s before that row is already the pilot's speed. "Unspecified" starts where
   it started.

### 4.6 Go-around words

The labelled data has no go-around: the arrival slice keeps only the last approach, and the labeller refuses a flight
that crosses and comes back. Thus the prior gives the word a probability near zero (approximately 1e−11 in the 9.4
readouts). O5 asks if the labeller must read the real go-arounds (R40 v2 found 105 on the training days). That needs a
slice that includes the second approach (median 421 s from the go-around to the next turn onto the final).

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

Thus one reading gives every Δ. Δ must be a multiple of 2 s.

**What a larger Δ changes (to read in the ablation; no criteria here, D7):**

| Part | Change |
|---|---|
| Heading words | In a 3°/s turn the track moves 12° in 4 s, so one word jumps two or three 5° classes. The lead L = 4 s is one row at Δ = 4 s and less than one row above it |
| Executor | The cycle stays 1 s. It hears the words at each Δ row. The heading law (arrive L after the hearing, the stopping rate) flies larger steps |
| Final approach | With D2 and D3 the model corrects the final only every Δ; an error grows for a longer time before the next word |
| Prior | Fewer steps for each flight and more changes for each step. The observation stays 16 s (`N_LOOK` = 16 s / Δ rows). A loss for each step cannot be compared between two Δ; a loss for each flight or each second can |
| Multi-aircraft | The scene step is Δ |

**Values (O10).** Proposal: Δ = 2, 4, 6, 8 s. The user sets the values and the criteria after the design is settled.

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

- **Frame (D5).** All positions and directions are relative to R: distance to the threshold along the course, offset
  right of the final, height above the threshold, motion direction minus the course (sine, cosine). The MSL height of
  the aircraft stays, because the level words are MSL.
- **Removed (D5):** the airport embedding (`prior/model.py:312`), the absolute position E, N and the absolute motion
  direction (`prior/data.py:63`), the absolute threshold position and course of each candidate (`prior/data.py:70`).
- **Candidates:** for each candidate, its position relative to the aircraft (as now) and relative to R (the layout of
  parallel runways), its length and its elevation. The landings on it in the 30 min before the step (as now).
- **Words in force:** the runway in force as its candidate vector, the go-around state G, the heading in force as the
  sine and cosine of its angle relative to the course of R, the other columns as embeddings, and the steps since each
  column said its word.
- **Glidepath (D13):** the height above the published glidepath of R at the aircraft's distance before the threshold,
  with the straight-line reference of §11.4 (TCH + d·tan(angle) + d²/(2R_e), R_e the earth's radius of curvature along
  the course). It is procedure geometry as an input; the model still decides the profile (principle 2). The value is
  only meaningful near the final; the model also has the offset from the final to weigh it.
- **Time (D16).** No row position embedding (`prior/model.py:313`) and no input "time from row 0" (`prior/data.py:63`
  `time`): both measure the time since the aircraft entered the 25 km slice, a cut of the data, and a long sentence (a
  go-around adds up to 900 s) reaches rows that training seldom saw. The causal time attention gives the order. RoPE in
  the time attention gives how far back each earlier row is: the rotation of a row's query and key uses its time in
  seconds, so the attention reads only time differences. Seconds, not rows, so that every row interval of D11 reads the
  same time. RoPE works with the row-by-row cache of the speaker (`Prior.extend`): a key is rotated once, when it is
  written. The RoPE base is set at implementation.
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
- R exists at every step for every aircraft. Thus the relations of the edge features (`inference/scene_edges.py`: the
  approach clock, the same runway, the parallel runways) and the separation judge (`inference/separation.py`) have a
  runway at every step. In `instruction-v3` this was also true; this design keeps it.
- The clearance mask (`inference/separation_masks.py:131`) blocks "cleared" while a cleared aircraft ahead is too close.
  This design has no "cleared" word. A replacement is open (O6).
- The speed-word mask stays.
- The traffic attention (initial output zero) on the single-aircraft prior stays as the start of the multi-aircraft
  post-training.

---

## 9 Gates

Each stage keeps a gate: the labeller (completeness, envelope containment with envelope width), the replay (the
executor flies the labelled sentences), the prior (teacher-forced likelihood against baselines, free generation), the
post-training, the multi-aircraft stages. The user sets the criteria of each gate after the design is settled (D7).
This document sets none.

---

## 10 Values

| Item | Value | Source |
|---|---|---|
| Row | 2 s, on even UTC seconds; larger intervals in the ablation (D11) | Artefact `v6_20261002` |
| Executor cycle | 1 s | Fixed choice |
| Heading grid | 5°, relative to the course of R | Spec (grid); D8 (frame) |
| Heading lead L | 4 s | Measured (vocabulary design §10.1) |
| Heading tolerance | 4.5° | Half grid 2.5° + 2° |
| Turn-rate limit | 4.7°/s | Spec, measured (p99.9) |
| Bank limit | 32° | Spec, measured (p99.9) |
| Roll rate p | 5°/s | Source cited in executor design §9 |
| Level grid, range, tolerance | 30 m, 0–5,400 m MSL, ±25 m | Spec |
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
