# Two-tier model: the prior

**Summary.** The prior is the upper part of the two-tier model (outline §1): a language model of the controller. At
each row it says one row of words for each aircraft, and the executor flies them. This document gives its inputs, its
outputs, its decoding, its training and how it runs at a new airport. It is stage B of the plan. It reads the
vocabulary only through the vocabulary's public interface (vocabulary §6) and its decisions; the post-training reads
this document only through its public interface (§7) and its decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete for its part. The principles and the shared rules are in the
outline (`outline.md`). The code of stage B is not written yet (§0.3); the code that it replaces is archived (§11).
Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the
repository root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`,
`two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D5 | Prior inputs are in the frame of the runway in force. The prior has no airport embedding, no absolute position and no absolute direction | Decided | User, 2026-10-03 |
| D13 | The prior gets the height above the published glidepath of R at each step (§2) | Decided | User, 2026-10-03 |
| D14 | While G is true, the procedure mask "no climb below the entry height" does not apply (§4). (The same decision has a part in the vocabulary: rule 5) | Decided | User, 2026-10-03 |
| D16 | The prior has no row position embedding and no input "time from row 0". The time attention uses RoPE with the row's time in seconds (§2) | Decided | User, 2026-10-03 |
| D17 | Every column has the input "time since this column said its word in force", in seconds; the runway column too (§2) | Decided | User, 2026-10-03 |
| D23 | How the prior gets the frame of R (D5) and the glidepath height (D13). The own state of the aircraft has no frame. Every position and direction is in the candidate vectors, each in the frame of its own candidate. R is an input only as its candidate vector. No input of a row up to the first predicted step is computed from R. Each candidate vector has the height above its own glidepath; the value of R is the input of D13 (§2) | Decided | User, 2026-10-03 |
| D24 | A candidate vector has no constant of its runway: no length, no elevation, no layout relative to R. Such a value comes back only as a variant that D39 selects (§2) | Decided | User, 2026-10-03 |
| D25 | At each Δ of the ablation (2, 4, 8 s), the motion inputs of a row come from the 2 s before the row (§2). (The Δ values: D25 in the vocabulary) | Decided | User, 2026-10-03 |
| D31 | The prior's training stops on the select days (§5). (The multi-aircraft part: D31 in the post-training) | Decided | User, 2026-10-03 |
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). Two variants: `full` (§2) and `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation). `constants` is chosen only if it is better by more than twice the seed scale (§5) | Decided | User, 2026-10-04 |
| D40 | Hyperparameters: configuration A (§5) is the start. The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§5). Every configuration has attention heads 32 wide: A 6 heads, B (d_model 128) 4, C (d_model 256) 8 — 128 and 256 are not six heads of one width | Decided | User, 2026-10-04; the head width: user, 2026-10-04, on Claude's proposal |
| D41 | The design runs at an airport that is not in the training data: the number of candidates is not fixed; every input has a fixed physical scale, never a statistic of the training data; each fold of D39 flies the full closed loop at its held-out airport. The altitude words are above the airport elevation (D58), so the levels of an approach lie in the 60 m segment at any airport (§6) | Decided | User, 2026-10-04 |
| D58 | The prior's own height is the height above the airport elevation E, as the altitude words are (vocabulary, D58). The prior gets neither E nor the MSL height (D24), so it cannot know where the round MSL levels that controllers assign lie (§2). (The words: D58 in the vocabulary) | Decided | User, 2026-10-04 |
| D60 | Row 0 of an aircraft has no motion inputs: no state 2 s before it is stored. There, the ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and the own input `no_motion` is 1 (0 at every other row). The same at every Δ, in training, in closed loop and in a loop of several aircraft. Not the fitted values of the data, which use 7.5 s after the row (§2) | Decided | User, 2026-10-04, on Claude's proposal |
| D63 | The identity of a prior's data (§8, item 1) also holds the landings that the candidate vectors count: for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would change the inputs and leave the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |

### 0.2 Open items

None.

### 0.3 Implementation

The implementer's log (outline §5 rule 10). Branch `dev-two-tier-v4-prior`, worktree `.claude/worktrees/two-tier-v4-prior`.
A proposal is a reading where the design says nothing; it holds only until the user decides.

| Part | State |
|---|---|
| B0: the package `prior/`; its import rules in `tests/test_architecture.py` (`prior/` reads only `instructions/`, the day split and the plain utilities; only the runners import it) | Done, `f3070978` |
| B2: the model (`prior/model.py`), a sentence's rows and their batch (`prior/batch.py`), the checkpoint `ts-prior-checkpoint-v6` (`prior/checkpoint.py`); `no_motion` (D60) | Done on synthetic sentences, `f3070978`, `de7d4994`. Full ts suite at `f3070978`: 1,559 passed |
| B3: the training loop (`prior/train.py`), the data of a run or a fold (`prior/runs.py`) | The loop done on synthetic sentences, `f3070978`. The runner `prior_train`, the smoke run, its time and the memory check at the formal size wait for A21 |
| B1: the inputs of a row (`prior/inputs.py`: `state_inputs`, `Heard`; a sentence and a loop use both), the landings (`prior/landings.py`), the artefact as sentences and the identity of the data (`prior/source.py`) | Done on synthetic artefacts, `278b626b`; the landings digest as D63, `ff514325` |
| B4: the speaker (`prior/speaker.py`), the procedure masks (`prior/procedure.py`, set `procedure-masks-v4`) | Done on synthetic inputs, `278b626b`; the finals read on KRDU's CIFP. Free generation (the speaker with the executor, the judge, the time limit) waits for A21 |
| The full ts suite at `278b626b` / `ff514325` | Not run yet: stage A's A21 build runs (outline §5 rule 13) |
| B5, B6 | Wait for Claude's check of stage A and the user's choice of Δ |

Proposals (where the design says nothing):

1. The candidate tokens reach a row through one attention over them, whose weights sum to one: not a sum, which grows
   with the number of candidates (D41).
2. The RoPE base is 10,000 (a head of 32: periods from 6.3 s to approximately 35,000 s).
3. The runway head's classes: "unchanged", "go-around", then the candidates.
4. The fixed scales (D41): distances along and across a candidate asinh(d / 1 km); heights 1 km; the height above the
   glidepath 100 m; ground speed 100 m/s; vertical rate 10 m/s; landings in 30 min 10; length 1 km.
5. The variant `constants` gives the threshold elevation as MSL.
6. The runway column's `since` starts again at a candidate word, not at "go-around".
7. The word rules of the procedure masks: a level only where it is not lower than the edge (inside the region) or the
   DA (before the join) by more than its ε; "no level-off" not inside the region where the aircraft is more than its ε
   below the edge; where the climb is barred no level higher than the aircraft's height plus its ε and no climb class;
   "unchanged" in the altitude column only where the word in force passes the rules that apply.
8. "Its stretch starts again after the go-around" (D14): a go-around clears where the aircraft joined and dipped; the
   next approach is read as new.
9. The entry height is passed when the aircraft is below it, with no band.

### 0.4 Plan

1. Stage B is developed in parallel with the end of stage A (the user, 2026-10-04), on its own branch (outline §5
   rule 1). B0–B6 (§12): the package, the data, the model, the training, the speaking and free generation, the
   cross-validation and the base model. A milestone starts when the parts of stage A that it reads are on
   `dev-two-tier-v4` (outline §4):

   | When | What |
   |---|---|
   | Now | B0. B2 and the training loop of B3, tested on synthetic inputs. The words of each column and their number come from the vocabulary spec (vocabulary §6, item 1), never from constants of the prior |
   | After A19, A20 and A22 of stage A (the altitude words above E, the new format names, the executor that flies T + E; the vertical path of each candidate in `candidates.json`, the reader of a closed-loop file, D61; the grammar's column mask, D62) | B1, tested on synthetic artefacts (`tests/support.py`). The speaker and the masks of B4 |
   | After A21 of stage A (the formal artefact) | B1–B4 on a sample of the formal artefact at Δ = 2 s: the smoke run of B3, its time and the memory check at the formal size; free generation with the executor of the formal artefact |
   | After Claude's check of stage A and the user's choice of Δ (outline §4) | B5, B6 |

2. Then the post-training (`post_training.md`).

---

## 1 Scope

- **This document owns** the package `prior/` (the inputs, the model, the training, the speaker and its masks) and its
  runners.
- **It reads** the outline (the principles, the shared decisions D7, D20, D21, D55, the rules of the implementation)
  and the vocabulary's public interface (vocabulary §6): the vocabulary spec and the grammar (items 1, 2), the sentence
  artefact (item 3), the candidates and their geometry (item 4) and the row grid (item 7). The package `prior/` does
  not import the executor or the judge (items 5, 6): a runner joins them (free generation, B4).
- **It gives** the public interface of §7, and nothing else, to the post-training.

---

## 2 Inputs

- **Source of the states (D32).** The rows before the first predicted step are observed (ADS-B), as in closed loop. From
  the first predicted step on, every input that comes from a state (the own state, the candidate vectors, the motion)
  comes from the flown states of the closed-loop sentences (vocabulary §6, item 3), not from the observed track. The
  words in force are those of the closed-loop sentence, corrections included. In closed loop the states come from the
  executor.
- **Own state (D5, D23, D58, D60).** The own state of the aircraft has no frame: its height above the airport elevation
  (the level words are above it), its ground speed, its vertical rate and `no_motion` (1 only at row 0, below). No MSL
  height: with it, the airport elevation (a constant of the airport) would be an input (D24).
- **Frame (D5, D23).** Every position and every direction is in the candidate vectors. Each candidate vector gives the
  aircraft in the frame of that candidate: the distance to its threshold along its course, the offset right of its
  final, the height above its threshold, the motion direction minus its course (sine, cosine)
  (`instructions.airport.relative_to_runway`). "The frame of R" is the candidate vector of R, which the prior gets as
  the runway in force. There is no second set of values relative to R.
- **No input is computed from R before R is said (D23).** The first predicted step says R. The inputs of the rows before
  it and of the first predicted step itself use no value computed from R. (The heads of a row see the runway that the
  same row said, in the order of the columns; that is an output, not an input of the row.) The reason: the artefact
  writes the runway word at row 0 (vocabulary §6, item 3), and its value is the runway on which the flight landed. An
  input of these rows computed from it gives the answer of the first predicted step, and in closed loop the same input
  does not exist. A test: a change of the runway word of a sentence leaves the inputs of rows 0 to `N_LOOK` the same,
  bit for bit.
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
  - The runway-intent study found this channel (R1.1, gradient-boosted trees, split by day;
    [plan](../../history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md) §15): with a constant of each runway
    (`prior_share`), the trees recognised "30L" and were right on only 53 / 59 % of the flights on the two KSJC days
    with 30L closed. Without it (R1.1b) they were right on 96.8 / 97.6 %. For the prior, R46 shows that it uses an
    airport identity when it gets one (§10.1).

  The constants are also not necessary. The length is a cause only together with the aircraft type, and the prior has no
  aircraft type. The elevation is not needed: the words and the own height are above the airport, and each candidate
  gives the height above its own threshold. (The own height minus a candidate's height above its threshold is that
  threshold's height relative to E: 0–28 m at the five airports, a small constant of the runway that two physical inputs
  give together.) Which of two parallel runways is the left one shows at each step in the offsets right of their finals
  (the value of the left runway is always larger). The relative positions of all candidates together still show the
  layout of the airport. That is real geometry, and the prior has it; only the held-out airports of D39 can measure how
  much the prior uses it. A constant comes back only as the variant `constants` of D39.
- **Landings (D63).** Offline, the landings come from the airport's tracks roster: every flight that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). A flight
  never counts its own landing: a closed-loop sentence can fly past the time at which the observed aircraft landed, and
  the runway of that landing is the answer of the runway word. In a loop, the caller gives the landings that the loop
  knows. The roster is outside the artefact, so the identity of a prior's data holds a digest of the landings (§8,
  item 1).
- **Words in force:** the runway in force as its candidate vector ("none yet" up to the first predicted step), the
  go-around state G, the heading in force as the sine and cosine of its angle relative to the course of R, the other
  columns as embeddings, and the time since each column said its word (D17).
- **Glidepath (D13, D23).** Each candidate vector has the height above the published glidepath of that candidate at the
  aircraft's distance before its threshold, with the straight-line reference of the vocabulary (vocabulary §6, item 4:
  TCH + d·tan(angle) + d²/(2R_e), R_e the earth's radius of curvature along the course). The value in the vector of R is
  the input of D13. Thus the value exists at every row, also before R is said. Every candidate has a TCH and a glidepath
  angle (the vocabulary refuses a candidate without them). It is procedure geometry as an input; the model still decides
  the profile (principle 2). The value is only meaningful near the final; the model also has the offset from the final
  to weigh it.
- **Motion (D25).** The ground speed, the vertical rate and the motion direction (in each candidate vector, minus its
  course) come from the displacement in the 2 s before the row, at every row interval Δ. Only past positions
  give them; never the fitted track, ground speed or vertical rate of the signals: a least-squares fit over 15 s
  centred on the row, which uses 7.5 s of the future.
- **Row 0 (D60).** Row 0 of an aircraft has no state 2 s before it: the observed data start at the entry of the 25 km
  slice, and the sentence artefact stores the states from row 0 of each sentence (vocabulary §6, item 3). At row 0 the
  ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and
  `no_motion` is 1; at every other row `no_motion` is 0. The rule is the same at every Δ, also when the data have a 2 s
  row before row 0, so that only the interval changes with Δ (D25). It is the same in training, in closed loop and in a
  loop of several aircraft (§7, item 2): the first row of an aircraft never has a state before it. The zeros alone
  already mark the row (no aircraft in flight has a ground speed of 0, and (0, 0) is not a direction); `no_motion` says
  it directly, so that the network does not have to learn it from an extreme value. Two other values were not chosen:
  - The fitted values of the data at row 0. They use 7.5 s after the row. For the aircraft's own first predicted step,
    16 s after row 0, this is no leak; but in a loop of several aircraft another aircraft reads the row at its own time,
    and then reads 7.5 s of the future (principle 7). Also, it would be a 15 s fit at one row and a 2 s displacement at
    every other row.
  - The displacement from row 0 to the next row: in a loop of several aircraft, that is 2 s of the future.
- **Time (D16).** No row position embedding and no input "time from row 0": both measure the time since the aircraft
  entered the 25 km slice, a cut of the data, and a long sentence (a go-around adds up to 900 s) reaches rows that
  training seldom saw. The causal time attention gives the order. RoPE in the time attention gives how far back each
  earlier row is: the rotation of a row's query and key uses its time in seconds, so the attention reads only time
  differences. Seconds, not rows, so that every row interval of D11 reads the same time. The time of a row is in seconds
  from the aircraft's own row 0. Only differences count, so the zero changes no result; but a UTC time (approximately
  1.76e9 s) in float32 has a step of 128 s and loses the rows. RoPE works with the row-by-row cache of the speaker
  (`Prior.extend`): a key is rotated once, when it is written. The RoPE base is set at implementation.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force" (`since`):
  log(1 + t / 2 s) / 5, with t in seconds, counted from the first predicted step at the earliest. It is in seconds for
  every column, so that every row interval of D11 reads the same time. The runway column's value is, in the labelled
  data, the time since the first predicted step, because a labelled sentence says its runway only there, and again only
  at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or given
  again (a change of runway, or the runway word that ends a go-around), so it measures a real fact.

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation of Δ would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ, at every row after row 0 (D60); the closed-loop
artefact stores the flown states on the 2 s rows (D51). At Δ = 2 s it is the value of now.

## 3 Outputs

Five heads in the order of the columns (vocabulary §6, item 1). The runway head scores each candidate (a pointer) plus
"unchanged" and "go-around". The number of candidates is the airport's own: no slot count is fixed by the model (D41).

## 4 Decoding

The speaker says a row column by column, in the order of the columns. A later column sees the words of the earlier
columns of the same row. Three kinds of masks block words:

**The grammar.** Rules 1–6 and the runway/G table, the vocabulary's one function (vocabulary §6, item 2). The first
predicted step masks "unchanged" in each column and "go-around" in the runway column. The speaker asks the grammar
column by column, after the earlier columns of the row (D62). It gives the grammar, for each later column, the words
that its other masks (below) permit there, so that a row never reaches a column with no permitted word.

**Procedure masks.** The prior decodes under three masks from the procedure of R (principle 3). The first two
block the altitude and angle words that would take the aircraft below a lower limit; the third blocks the climb:

- inside the FAF and the LPV cone: the glidepath lower edge, the published glidepath − 60 m (nowhere else: the RNAV
  floors outside the FAF disagree with 10–14 % of the recorded tracks);
- before the join (the first row inside that region): the published DA;
- once the aircraft is below the entry height before the join: no climb.

**Masks while G is true (D14).** Only one mask stops a go-around: "no climb below the entry height". It does not apply
while G is true, and its stretch starts again after the go-around. The other two masks are lower limits; a go-around
climbs above them, so they apply. The grammar applies while G is true as at every row.

A level word is checked against these limits with the band ε of its level (vocabulary §6, item 1). A prior records the
set of procedure masks that it was trained under (`procedure_masks.json`, contract C35) and speaks under it.

**Masks of a caller.** The loop that runs the speaker can give a set of forbidden words for each column (§7, item 3).
The speaker applies them as it applies the others. The prior does not know what they mean.

## 5 Training

Teacher forcing on the closed-loop sentences of the artefact, split by operating day (`data/day_split_20260924.json`;
test days sealed). KAUS is a held-out test airport only: it is read one time, at the end of the whole chain.

**Stop on the select days (D31).** The training keeps the epoch with the smallest per-step loss on the select days. The
training does not read the validation days: they are read one time for each stage.

**Hyperparameters (D40).** Configuration A, the start: d_model 192, 4 layers, 6 attention heads, feed-forward 768,
dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ with 500 warm-up steps, gradient clip 1.0, 16,384 aircraft-steps
in a batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is set at
implementation; a test makes sure that a shift of all times changes nothing.

**Cross-validation and the choice of the design (D39, D40).** The folds are the five airports: a fold trains on the
train days of four airports, stops on their select days and reads the select days of the fifth (the held-out airport).
The goal is a new airport, so the folds are airports, not days. All runs use the chosen Δ (D11).

| Step | Runs | Training runs |
|---|---|---|
| 1 | Variant `full`, four configurations, each on the 5 folds. A: the start. B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. B, C and D have the layers and the learning rate of A; every head is 32 wide | 20 |
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
  airport (top-1); free generation at the held-out airport (200 flights × 2 sentences): its outcomes (vocabulary §6,
  item 6) and the words for each column.

The cost, Claude's estimate before the smoke of B3: approximately 50 min for one run on four fifths of the data, so
approximately 25 h of GPU for steps 1–3, and 1–2 h for the free generation of the folds.

## 6 Running at a new airport

The prior has no identity of an airport (D5, D23, D24). A new airport needs only data that the five airports also
come from:

| What the airport needs | Source |
|---|---|
| Candidate runways and their geometry: threshold, course, TCH, glidepath angle, LPV DA; the airport elevation | FAA CIFP and the runway data (the vocabulary refuses a candidate without them) |
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

**Heights and speeds at a high airport.** The altitude words are above the airport elevation (D58), so a level word
means the same height above the airport, and the final approach lies in the 60 m segment, at every airport. The speed
words are ground speeds: at a high airport one indicated airspeed is a larger ground speed (at E = 1,650 m approximately
8 %, about 6 m/s at an approach speed of 70 m/s: one step). The speed words of a high airport thus lie about one step
above those of the training airports for the same phase. The five airports and KAUS lie below 200 m.

---

## 7 Public interface

The post-training and its code use this document only through these items and the decisions (the D numbers).

| # | Item | What it gives |
|---|---|---|
| 1 | The checkpoint | A trained prior in a format with its own name; its identity (§8) |
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said, the candidates and the landings before the step to the inputs of a row (§2). Training, free generation and a loop of several aircraft use the same function |
| 3 | The speaker | Says one row from the inputs of the row, column by column, with a cache from row to row; under the masks of §4, the masks of a caller included; with a random generator that the caller gives |
| 4 | The teacher-forced loss | The loss of each step and each column for a batch of sentences |
| 5 | A place in each layer | A module added at each layer whose output starts at zero leaves every output of the prior unchanged until it learns |

---

## 8 Gates and identities

The gate of this document is the prior: the teacher-forced likelihood against baselines and free generation. The user
sets its criteria (D7). The identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table, the sha256 of the sentence files, and the landings that the candidate vectors count (D63): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number left out on the sealed test days. A run that reads the landings again computes the digest again and refuses a difference |
| 2 | The procedure masks of a prior | The set name, the checkpoint sha and the digests of the procedure data (C35). The procedure data are the format of the masks |

---

## 9 Values

| Item | Value | Source |
|---|---|---|
| Motion inputs | Displacement in the 2 s before the row, at every Δ; at row 0 of an aircraft 0, with `no_motion` = 1 | D25, D60 |
| Rows before the first predicted step | 16 s: 8, 4, 2 rows at Δ = 2, 4, 8 s | Vocabulary §6, item 7 |
| Time since a word | log(1 + t / 2 s) / 5, t in seconds, from the first predicted step at the earliest | D17 |
| Landings of a candidate | In the 30 min before the step | §2 |
| Configuration A | d_model 192, 4 layers, 6 heads, feed-forward 768, dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴, 500 warm-up steps, clip 1.0, 16,384 aircraft-steps a batch, at most 30 epochs, stop after 3 | D40 |
| Configurations B, C, D | B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: A with dropout 0.2, weight decay 0.05. A head is 32 wide in each | D40 |
| Procedure masks | Glidepath lower edge: published glidepath − 60 m inside the FAF and the LPV cone; the DA before the join; no climb below the entry height before the join | §4 |

---

## 10 Evidence

### 10.1 The prior uses its airport embedding (R46)

[Readout](../readouts/2026-10-03_airport_embedding.zh.md). On the validation days, a replacement of the airport
embedding with the mean of the four other embeddings increases the teacher-forced loss by 0.024 (base) and 0.028
(augmented) for each step. It decreases the landed share in free generation by 5.8 and 8.0 points. Approximately 60 % of
the increase is in the heading column. The five embeddings are almost orthogonal: an identity table, not a property of
the airports. The candidate table (absolute threshold positions and courses) is a second identity: R46 §6.

### 10.2 Free generation and the heading column of the earlier prior

- Base model, validation days: per-step loss 0.2648, of which heading 0.1214 (46 %) (R46, own embedding).
- Base model free generation (validation, 400 × 4 for each airport): landed 88.7 %, timeout 6.5 %, crossed too high 3.5
  % (R46). The timeouts are mostly a flight that gets heading words and no clearance (`two_tier_stage_notes.zh.md`
  §3.2).
- Labelled heading words for each flight: straight-in 3.0, vectored 33.7 (`instruction_vocabulary_design.zh.md` §10.1).

---

## 11 Key code index

The code that this design replaces, archived by stage A unchanged.

| What | Where |
|---|---|
| Prior: airport embedding, position embedding, candidate sum (line numbers at the move) | `archive/two_tier_v3_2026_10/prior/model.py:312`, `:313`, `:382`, `:385` |
| Prior: step, relative and candidate features (line numbers at the move) | `archive/two_tier_v3_2026_10/prior/data.py:63`, `:64`, `:70` |

---

## 12 Implementation plan: stage B

**Start.** In parallel with the end of stage A, on the branch `dev-two-tier-v4-prior` (outline §5 rule 1); §0.4 gives
when each milestone starts. Until the formal artefact of A21 exists, the tests use synthetic inputs and artefacts. Then
the code is tested on a sample of the formal artefact at Δ = 2 s, read-only. The smoke artefacts of stage A are not
used: their spec is not the spec of D56 and D58 (D55). The formal runs of B5 need Claude's check of stage A and the
chosen Δ (D11). Stage B changes no code of `instructions/` or `autopilot/`; it reads them only through the
vocabulary's public interface (vocabulary §6). The rules of outline §5 apply.

**Written from this document, not patched from the archive.** The archived `prior/`
(`archive/two_tier_v3_2026_10/prior/`) stays unchanged. A part of it comes back only where its logic fits this document,
rewritten into the new modules: the row-by-row encoding with a cache (`Prior.extend`), the candidate tokens with one
shared network and the pointer head, the ordered heads, the training loop (AdamW, warm-up, clipping, early stopping),
the index of the airport's landings, the rules of the procedure masks. Nothing else comes back: not the six-column
layout, the airport and row-position embeddings, the candidate constant table, the fixed count of candidate slots, the
variants and selection rule of `instruction-v3`, the aircraft attention of a one-aircraft scene, the checkpoint formats
`v3`–`v5`. No compatibility (principle 8).

**B0. Package and layout.**

- A new package `prior/`. It reads `instructions/` and the artefact files; it does not import `autopilot/`; only the
  runners join it to the executor (§1). These rules go into `tests/test_architecture.py` (the prior's rules cut into
  `tests/test_mirrors_cut_from_live_tests.py` come back, rewritten).
- The runners of B: `prior_train` (one run: all airports, or one fold with a held-out airport), `prior_select` (reads
  the campaign of B5 and writes the choice), `prior_free_generation` (the prior speaks, the executor flies). New code;
  the archived runners of the same names stay as they are.

**B1. Data** (§2, §7 item 2; D13, D17, D23–D25, D32, D41, D60, D63).

- The rows: before the first predicted step, the observed states; from it on, the flown states of the closed-loop
  sentences, on the rows of the chosen Δ.
- The own state: height above the airport elevation (D58), ground speed, vertical rate, `no_motion`; the motion from the
  displacement in the 2 s before the row; at row 0 the motion inputs 0 and `no_motion` 1 (D60).
- One vector for each candidate: the distance before its threshold along its course, the offset right of its final,
  the height above its threshold, the motion direction minus its course (sine, cosine), the height above its glidepath
  (straight-line reference), the landings on it in the 30 min before the step. No constant of the runway, except in the
  variant `constants` of D39 (length, threshold elevation).
- The words in force: the runway as the candidate vector of R ("none yet" up to the first predicted step); G; the
  heading as the sine and cosine of its angle relative to the course of R; the other columns as embeddings; the time
  since each column said its word, in seconds.
- The targets: the five columns of the closed-loop sentence.
- Every scale is a constant in SI units (D41). A flight without a training sentence (vocabulary §6, item 3) is not read.
- The landings from each airport's tracks roster, less the sealed test days, never a flight's own (§2, D63); their
  digest in the identity of the data (§8, item 1).
- Tests: a change of the runway word leaves the inputs of the rows up to the first predicted step the same, bit for bit
  (D23); the glidepath height against a hand computation; the motion from the 2 s displacement at Δ = 2, 4, 8 s; at row
  0 the motion inputs 0 and `no_motion` 1 at Δ = 2, 4, 8 s, also when the data have a 2 s row before row 0, and
  `no_motion` 0 at every other row (D60); a change of the stored track, ground speed and vertical rate of the states
  changes no input (only positions and heights give the motion); the flown states from the first predicted step on; a
  permutation of the candidates permutes their vectors and nothing else; an airport with more candidates than any
  training airport is read; a change of a landing's time or runway in the roster changes the digest of the landings, and
  a change of another field of the roster does not; a run refuses landings whose digest differs from the identity (D63).

**B2. Model** (§2, §3; D16, D41).

- The time attention with RoPE on the seconds from the aircraft's row 0; no row-position and no airport embedding.
- The candidate tokens through one shared network; five heads in the order of the columns; the runway head points at the
  candidates, plus "unchanged" and "go-around"; the first predicted step masks "unchanged" and "go-around".
- Each layer has a place where a module with a zero output can be added (§7, item 5).
- The checkpoint: a new format name; its identity as §8.
- Tests: the row-by-row encoding gives what the encoding of the whole sentence gives; a shift of all times changes no
  output; a permutation of the candidates permutes the runway scores; any number of candidates.

**B3. Training** (§5; D31, D40).

- Teacher forcing; the airports of a run (all, or a fold without its held-out airport); the stop on the select days;
  the validation days not read.
- Before a formal run: the check at the formal size of the host memory and the GPU memory of the largest batch.
- A smoke run on a sample of the formal artefact; it gives the time of one run for the cost of B5.
- Tests: the stop reads only the select days; a fold never reads its held-out airport in training.

**B4. Speaking and free generation** (§4, §7 item 3; vocabulary §6 items 2, 5, 6; D14, D33, D38, D52, D62).

- The masks of §4: the grammar; the procedure masks (the glidepath lower edge inside the FAF, the DA before the join,
  no climb back; lifted while G is true as D14 says), a level checked with the band ε of its level (D52); the masks of
  a caller.
- The closed loop: the prior speaks, the executor flies, the judge decides (D33, D38); 900 s more time at each
  go-around; the observed rows before the first predicted step, the executor's states after it.
- The readout: the outcomes for each airport and each kind of approach; the words for each column against the
  labelled ones; the go-arounds said; the probability of "go-around" on the final.
- Tests: one flight spoken and flown to its outcome; each mask; G; the time limit; the same seed gives the same
  sentence; no row reaches a column with no permitted word (D62).

**B5. Cross-validation and the base** (D39, D40, D41).

- The 31 training runs of §5, as one campaign from one commit on a clean checkout, one at a time on the GPU.
- For each fold: the held-out loss, the first-step runway at the held-out airport, the free generation at the held-out
  airport (200 flights × 2).
- `prior_select` applies the rules of §5 and writes the choice. Then the base on all five airports, and its one
  validation readout: the teacher-forced loss, the free generation, the share of the labelled words that the masks
  block, the probability of "go-around" on the final.
- No criterion is applied: the user reads the results (D7).

**B6. Close of stage B.** The full ts suite passes (run detached). §0.3 (the log) and `docs/reference/runners.md` are
updated; the report gives the new code index for §11 (outline §5 rule 10). `dev-two-tier-v4` merges `dev-two-tier-v4-prior` (outline §5 rule 1). Report to the user: the commits, the
readings of each fold and of the base, the choice and its rule, and what stage C needs.
