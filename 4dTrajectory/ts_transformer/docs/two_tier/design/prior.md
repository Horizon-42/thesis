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
outline (`outline.md`). The code of stage B is on the branch `dev-two-tier-v4-prior` (§0.3); the code that it
replaces is archived (§11).
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
| D64 | The word rules of the procedure masks (§4). They block a word when it is said and never block "unchanged". Inside the region (the FAF and the LPV cone of R) a level T only where T ≥ the glidepath lower edge − ε(T), and "no level-off" not where the aircraft is more than ε("no level-off") below the edge. Wherever the aircraft is not inside the region (before the join, or after it when the aircraft has left the region), a level T only where T ≥ DA − ε(T). Before the join, once the aircraft has been below the entry height by more than the band ε of the level nearest the entry height, no level above the aircraft's height + ε and no climb class. A go-around clears the join and the passage below the entry height; while G is true neither is kept | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D65 | The inputs and the model where §2 left them open: the fixed scales of §9 (the height above the glidepath as asinh(h / 100 m)); the candidate tokens reach a row through one attention over them whose weights sum to one; the RoPE base is 10,000; the variant `constants` gives the threshold elevation MSL; the runway column's `since` starts again at a candidate word, not at "go-around" (§2) | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D68 | Free generation lets a flight say at most 2 go-arounds. After its second, the runner forbids "go-around" in the runway column (a mask of a caller, §4), and it gives 2 as the most go-arounds to the start of the closed loop (vocabulary §6, item 5; D67). The readout counts the flights that reached the bound. Why: the executor lays out the time that go-arounds add (900 s each) when it starts, and the number that a model says is not known before; no closed-loop sentence of `v9_20261004` at Δ = 2 s has more than one go-around (train 68 of 40,534 sentences, select 12 of 6,199, val 20 of 9,750); 2 lets a model say one more than the data, and no flight goes around without end (B4) | Decided | User, 2026-10-04, on the report of the stage B agent |
| D72 | The readout's probability of "go-around" on the final (B4, B5) is read on the rows on the final (inside the region of the runway in force, D64) where the masks permit "go-around": not while G is true, not after the bound of D68. The probability is that of the distribution the speaker draws the runway word from, the masks applied. Why: at a row where the word is masked, its probability is 0 because of the mask, not the model, and would pull the mean down | Decided | User, 2026-10-04, on the proposal of the stage B agent |
| D75 | The base learns only from closed-loop sentences whose own words land: a selection of the artefact's sentences by a named rule (`all`, `landed`), applied when they are read; the artefact is not changed. `landed` keeps a sentence whose stored outcome (vocabulary §6, item 3; D74) is a landing. It applies to train and select (the stop and the choices of D39 and D40 read the same kind of sentence that the training reads) and to the teacher-forced loss of the validation readout; free generation starts from every flight, and the readout gives the flights outside the selection apart. A run records the rule and, for each split, airport, stratum and outcome, the sentences kept and left out (§8, item 1). Why: about 2.2–2.6 % of the sentences do not land when their own words are flown (A25, Δ = 2 s: train 1,955 of 1,999 replayed, select 6,036 of 6,199; KMSY 94–95 %), mostly high at the DA; their last words teach an end that does not land, and the vocabulary and the executor, not the words, are often the cause, so imitation cannot correct them. They come back in the post-training as starts (post-training D76) (milestone B8) | Decided | User, 2026-10-04 |
| D96 | The public interface gives what the post-training needs (§7), and stage B makes the changes (the user, 2026-10-05): (1) the speaker passes the caller's input of the added modules (§7 item 5) to the model at every row that it encodes or says; (2) the caller gives the random numbers: one uniform number for each aircraft, row and column, and the speaker draws, from the masked distribution, the word whose cumulative probability (in the class order of the heads) first passes the number; an aircraft's words thus depend only on its own numbers and inputs (the probabilities up to their last bits: post-training §6.4); free generation's files get a new format name; (3) for each row said, the speaker records the words that all masks permitted in each column, as an object that the caller keeps and gives back but does not read; the teacher-forced part gives the log-probability of given words under such records, with the input of the added modules, with gradients; at the parameters that spoke, it equals the probability that the speaker drew from, within the float tolerance (a test); (4) the inputs of a row in a loop are one function of `prior/` (until now `row` inside `experiments/prior_free_generation.py` `speak_and_fly`): from each aircraft's states on the 2 s rows (`STATE_COLUMNS`), its landings and the speaker's words in force, the inputs of the next Δ row and the position that the masks read; free generation calls it; (5) a copy of chosen aircraft of a speaker, repeats permitted: the cache, the state of the procedure masks, the words heard, the go-arounds said and the records; a copy continued with the same inputs and numbers says what the original says, bit for bit in a batch of the same layout (a test); (6) §7 gives the code of each item, the region of a candidate's final (D64) and the reading of a split's sentences under a selection rule (D75) as batches included; the post-training's architecture test holds its code to these names (post-training C0). Why: the post-training's branch groups need each aircraft's own random numbers and a copy of the state at a branch point (post-training D94); its loss needs the masked distribution (post-training §2 item 5); its traffic attention needs its input while the prior speaks; and it needs one function for a loop's row and one region of the final (post-training D92). Without these, it would copy code of `prior/`: a second definition (milestone B9, before B5's formal campaign) | Decided | User, 2026-10-05, on Claude's report of stage C's readiness |
| D105 | The landings that a loop's row counts (§2, §7 item 2) are given for each aircraft, not one index for each airport: the function of a loop's row takes each aircraft's landings. A caller builds them from the roster's landings (`Landing`, `LandingIndex`, in §7 item 2); the index refuses a landing on a sealed test day by itself (C32), not only the reader of the roster. Free generation gives each flight its airport's roster landings, as now. (The window's landings: D105 in the post-training) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D106 | Stage B gives what the post-training's window loop needs, so that it copies no code of `prior/` or of free generation (extends D96; milestone B10, before B5's formal campaign): (1) the step of a speaker's closed loop as one module under `experiments/`, shared by free generation and the post-training's window loop: the observed rows up to the first predicted step, then at each Δ row the inputs of the row (§7 item 2), the speaker with the caller's random numbers, masks of a caller and input of the added modules (given by the caller for each row), the executor's step through the start of a closed loop (vocabulary §6 item 5), the flights that are done halted; the caller can end a flight (a loss of separation, post-training D93); its state can be copied for chosen flights (post-training D94; the loop's own copy is vocabulary D97); free generation calls it, and its words, states and readout stay the same, bit for bit (a test); (2) one function that opens a prior run: the checkpoint, the identity of its data and the check of its procedure masks (§8 items 1, 2); every runner of the prior uses it; (3) the rows of a sentence that a loop said, with its words as targets, as one batch for the log-probability of §7 item 4; (4) a join of the records of several speakers into one batch, padded; (5) the motion of rows (the 2 s displacement, D25, D60), with the velocities east and north, in §7 item 2; the post-training deletes its mirror of it; (6) the state of a speaker that a loop reads, named in §7 item 3, read only: the words in force, the go-arounds said, the number of candidates. Why: Claude's check of stage B (`readouts/2026-10-05_stage_b_check.zh.md`) found the step of the closed loop only inside the runner (`speak_and_fly`: no input of the added modules, no other masks of a caller, no end by the caller), so the window loop would copy it, and post-training C4's test (a window without other aircraft equals free generation, bit for bit) would rest on a copy — the observed rows encoded in one call instead of one at a time already move the probabilities by 1.2e-7; and stage C needs names that §7 did not give (it mirrored the motion already) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D107 | The speaker and the log-probability of given words under records (§7 items 3, 4) refuse a model of which any module is in training mode: the speaker speaks with dropout off, and the log-probability under its records is that of the policy that spoke only with dropout off. The teacher-forced loss of item 4 (`batch_nll`) does not refuse. (Which term of the post-training runs in which mode: D107 in the post-training) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D108 | B5's campaign runs on a clean checkout. Before each step it runs a behaviour check of the prior's code on fixed inputs: on the CPU with one thread, two steps of training on a fixed synthetic batch and some rows said by the speaker with fixed numbers; the losses, the words and the probabilities are compared, bit for bit, with those that the campaign recorded at its start, and a difference stops the campaign by name. The commit of each step is recorded as information and never compared. Why: the campaign compared the commit before each step, a guard by an equal commit, against the user's rules (results of different code are comparable after a behaviour check on fixed inputs, never by an equal commit, 2026-10-02; no code fingerprint as the guard of a run, 2026-10-04): a commit that changes only documents stopped the campaign (Claude's check of stage B) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D111 | The selection `landed` (D75) also leaves out a flight that stage A marks as having a faulty observed track (vocabulary D111, §6 item 3), whatever its outcome; `all` keeps every sentence. The run's record gives, for each split, airport and stratum, the sentences left out for a faulty track apart from those left out by their outcome (§8 item 1). Free generation and the post-training's starts are not changed (the user: the post-training may start from flights that do not land). Why: the closed-loop sentences of some flights with ADS-B faults land, and their observed rows — which the prior reads — hold the fault | Decided | User, 2026-10-05 |

### 0.2 Open items

None.

### 0.3 Implementation

The commits, dates, branches, test counts and measurements of every milestone are in
`readouts/2026-10-05_stage_b_implementation_log.md` §1 (the design document holds no log). Branch
`dev-two-tier-v4-prior`, worktree `.claude/worktrees/two-tier-v4-prior`.

| Part | State |
|---|---|
| B0–B4, B8 (§12) | Done on synthetic artefacts. B3's smoke, free generation and the formal-size check are run again on A34's artefact (§0.4) |
| B3's smoke and the formal-size check | Done at Δ = 2 s on `v9_20261004`, at Δ = 4 s on `v11_20261004` / `v16_20261004`; on A34's artefact the memory check (A, C) and the timing (A 26 s, C 36 s an epoch) done (the log §1) |
| After A32 (D77–D87) | The prior reads a closed-loop sentence's rows alone and the withheld fields apart (D82); its export calls `split_flights` |
| B5's tools | Built on synthetic artefacts: the first-step runway at the held-out airport, the base's validation readout, the campaign runner |
| B6 | Started 2026-10-05 (the log §1) |
| B9 (D96) | Done, reviewed twice: `1204fd8f`, `6a68ac67`, `0b050797`, `bb86a39f`, `5313b6cd`; full suite passed; the names for §7's "Code" column in the log §1 |
| D90 / A37 | `Loop.executor` stays public (D90 narrowed): the gap is closed; stage B follows the changed A37 after its merge (the log §1) |
| Claude's check of stage B | Done at `5313b6cd` (`readouts/2026-10-05_stage_b_check.zh.md`): no leak into the inputs; the corrections are B10 (D105–D108) |
| B10 | Done (`e4e7ba42`, reviewed twice; full suite passed); D109 waits for A39, the loop's copy for A38; the names for §7 in the log §1 |
| B11 (D111) | Planned (§12): after vocabulary A40, before B5's formal campaign |
| B5, B7 | Wait for B10, B11, A34's artefact and Claude's check of stage A (§0.4) |

Proposals (where the design says nothing): none open.

### 0.4 Plan

1. Stage B is developed in parallel with the end of stage A (the user, 2026-10-04), on its own branch (outline §5
   rule 1). B0–B7 and B9 (§12): the package, the data, the model, the training, the speaking and free generation, the
   cross-validation and the base model, the Training view of stage B, the close, the interface for the post-training.
   A milestone starts when the parts of stage A that it reads are on `dev-two-tier-v4` (outline §4). The milestones
   that have started are in the log (§2 there); these are the ones that wait:

   | When | What |
   |---|---|
   | After A34 of stage A (the formal artefact with D77–D84) | The smoke of B3, free generation and the check at the formal size at Δ = 4 s again, on A34's artefact (the candidates of an airport can change, D78); the formal runs read A34's artefact |
   | Now, before B5's formal campaign (the user, 2026-10-05) | B9: the interface for the post-training (D96), on synthetic artefacts. Its change of free generation's draws comes before any formal free generation: a payload that an experiment writes is settled before the experiment runs |
   | Now, before B5's formal campaign (the user, 2026-10-05) | B10: the corrections of Claude's check of stage B (D105–D108), on synthetic artefacts; then B3's smoke and free generation on A34's artefact again. Its changes of free generation (the masks of D64 at the row that ends G, the region of D72, the shared step) come before any formal free generation |
   | Now: stage A's line is merged (`dev-two-tier-v4` at `ed2530ae`, the user, 2026-10-05: A37–A40) | Stage B merges it and follows stage A's changes (`requests_from_a_to_designer.md` §2): A37 — the reason a flight ended from the judge's outcome (`judge.TIMEOUT`, vocabulary D90), the export's runway offset from `training_export.candidate_hae_minus_msl_m`; A38 — the copy of the closed loop's step (D106 (1)) takes the loop's own copy (`Loop.copy`), and a halted flight hears no words; A39 — B10's item of outline D109 with the splits given to the backend's service and to the frontend's reader. Then B11 (D111) |
   | After B10 and B11 (Claude's check of stage A, A32–A40, is done: `readouts/2026-10-05_stage_a_check_a32_a40.zh.md`; Δ = 4 s is chosen, vocabulary D11; A34's artefact exists); A32 holds vocabulary D86 (the export gives val flights for the base's validation readout) | B5; B6's publication of the folds and the base (the base's val set after vocabulary A39, outline D109); B7 |

2. The post-training (`post_training.md`) is developed in parallel with the end of stage B, on its own branch, made
   from this one (outline §4, §5 rule 1; D95). It reads only §7; it merges this branch when B9 is committed.

---

## 1 Scope

- **This document owns** the package `prior/` (the inputs, the model, the training, the speaker and its masks), its
  runners, and the Training view of its results (B6, outline §6).
- **It reads** the outline (the principles, the shared decisions D7, D20, D21, D55, the rules of the implementation)
  and the vocabulary's public interface (vocabulary §6): the vocabulary spec and the grammar (items 1, 2), the sentence
  artefact (item 3), the candidates and their geometry (item 4) and the row grid (item 7). The package `prior/` does
  not import the executor or the judge (items 5, 6): a runner joins them through the start of a closed loop (item 5,
  D67; free generation, B4).
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
  much the prior uses it. A constant comes back only as the variant `constants` of D39: the length and the threshold
  elevation MSL (D65). That variant thus also gives the airport elevation E, which `full` does not (D58).
- **Landings (D63).** Offline, the landings come from the airport's tracks roster: every flight that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). A flight
  never counts its own landing: a closed-loop sentence can fly past the time at which the observed aircraft landed, and
  the runway of that landing is the answer of the runway word. In a loop, the caller gives the landings that the loop
  knows. The roster is outside the artefact, so the identity of a prior's data holds a digest of the landings (§8,
  item 1).
- **Candidate tokens (D41, D65).** Each candidate vector goes through one shared network. The candidate tokens reach a
  row through one attention over them whose weights sum to one: a sum would grow with the number of candidates.
- **Scales (D41, D65).** Every input has a fixed scale in SI units (§9): the distances along and across a candidate
  asinh(d / 1 km), linear within a few hundred metres of a final; the heights (above E, above a threshold) h / 1 km; the
  height above a glidepath asinh(h / 100 m), linear near the glidepath, where the value has a meaning, and compressed
  far from it (an aircraft high above an opposite runway's extended glidepath is thousands of metres above it); the
  ground speed v / 100 m/s; the vertical rate v / 10 m/s; the landings in 30 min n / 10; in the variant `constants` the
  length l / 1 km and the threshold elevation MSL h / 1 km.
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
  (`Prior.extend`): a key is rotated once, when it is written. The RoPE base is 10,000 (D65): with heads 32 wide, the
  periods go from 6.3 s to approximately 35,000 s.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force" (`since`):
  log(1 + t / 2 s) / 5, with t in seconds, counted from the first predicted step at the earliest. It is in seconds for
  every column, so that every row interval of D11 reads the same time. The runway column's value is, in the labelled
  data, the time since the first predicted step, because a labelled sentence says its runway only there, and again only
  at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or given
  again (a change of runway, or the runway word that ends a go-around), so it measures a real fact. A candidate word
  starts it again; "go-around" does not, because it keeps R (D10) and G is an input of its own (D65).

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

**Procedure masks (D64).** The prior decodes under three masks from the procedure of R (principle 3). The first two
are lower limits on the altitude words; the third blocks the climb. The region is inside the FAF and the LPV cone of R;
the join is the first row inside it. ε is the band of a level (vocabulary §6, item 1; D52):

- **The glidepath lower edge**, inside the region: the published glidepath − 60 m (nowhere else: the RNAV floors outside
  the FAF disagree with 10–14 % of the recorded tracks). A level T only where T ≥ edge − ε(T). "No level-off" not where
  the aircraft is already more than ε("no level-off") below the edge. The edge is taken at the aircraft's position; it
  only falls toward the threshold, so a level permitted there stays permitted inbound.
- **The DA**, wherever the aircraft is not inside the region: before the join, and after it when the aircraft has left
  the region (out of the LPV cone before the threshold). A level T only where T ≥ DA − ε(T). After the join, outside the
  region, no other limit holds, so the DA holds there too.
- **No climb**, before the join, once the aircraft has been below the entry height (the glidepath at the FAF) by more
  than the band ε of the level nearest the entry height: no level T above the aircraft's height + ε(T), and no climb
  class. The band: the levels are on a grid and the executor holds a level exactly, so an aircraft at the level nearest
  the FAF altitude can be up to half a step below the entry height without a descent below it.

**A mask blocks a word when it is said, never "unchanged".** A word in force was permitted when it was said. A mask
that made the speaker change it, for example a level forced above an aircraft that sank under the edge with
"no level-off" in force, would fly the aircraft along a line that the procedure computes: the glidepath floor that the
executor does not have (D9). Whether to correct or to go around is the model's decision, and the DA check judges it
(vocabulary §6, item 6).

**Masks while G is true (D14).** Only one mask stops a go-around: "no climb below the entry height". It does not apply
while G is true. A go-around clears the join and the passage below the entry height, and while G is true neither is
kept: the next approach is read as a new one (D26). The other two masks are lower limits; a go-around climbs above
them, so they apply. The grammar applies while G is true as at every row.

A prior records the set of procedure masks that it was trained under (`procedure_masks.json`, contract C35) and speaks
under it.

**Masks of a caller.** The loop that runs the speaker can give a set of forbidden words for each column (§7, item 3).
The speaker applies them as it applies the others. The prior does not know what they mean. Free generation gives one:
"go-around" after a flight's second go-around (D68).

**Drawing (D96).** The caller gives one uniform random number for each aircraft, row and column. The speaker draws,
from the distribution with all masks applied, the word whose cumulative probability, in the class order of the heads,
first passes the number. An aircraft's words thus depend only on its own numbers and its own inputs, not on the other
aircraft of the batch. For each row, the speaker records the words that all masks permitted in each column (§7, items 3
and 4).

## 5 Training

Teacher forcing on the closed-loop sentences of the artefact, split by operating day (`data/day_split_20260924.json`;
test days sealed). KAUS is a held-out test airport only: it is read one time, at the end of the whole chain.

**Stop on the select days (D31).** The training keeps the epoch with the smallest per-step loss on the select days. The
training does not read the validation days: they are read one time for each stage.

**Hyperparameters (D40).** Configuration A, the start: d_model 192, 4 layers, 6 attention heads, feed-forward 768,
dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ with 500 warm-up steps, gradient clip 1.0, 16,384 aircraft-steps
in a batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is 10,000 (D65); a
test makes sure that a shift of all times changes nothing.

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

The cost, Claude's estimate from the smoke of B3 (`v9_20261004`, Δ = 2 s, configuration A): approximately 45 s an
epoch on a fold and 55 s on all five airports, so at most approximately 25 min for one run of 30 epochs, and
approximately 13 h of GPU for the 31 runs if each run costs as much as A (B is smaller, C is larger). The runs use
Δ = 4 s, where a sentence has half the rows. Before B5, the memory check and the time are measured again at Δ = 4 s
with configurations A and C (the log §1); that measurement replaces this estimate. The free generation of the folds is not
measured on real data: Claude's estimate before the smoke, 1–2 h.

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

The post-training and its code use this document only through these items and the decisions (the D numbers). Its
code imports from `prior/` only the names in the column "Code" (an architecture test, post-training C0); everything
else in `prior/` can change without a change of the post-training. Item 7 is a module under `experiments/`, which the
post-training's runners and its window loop import by the names given. "New, B10" marks a part that B10 adds (D105–D107);
its names are given to Claude in B10's report, and Claude writes them here.

| # | Item | What it gives | Code |
|---|---|---|---|
| 1 | The checkpoint | A trained prior in a format with its own name; its identity (§8); the set of procedure masks that it speaks under (§8 item 2); one function that opens a prior run: the checkpoint, the identity of its data and the check of its procedure masks (D106) | `prior/checkpoint.py` `load_checkpoint`, `CHECKPOINT_SCHEMA`; `prior/model.py` `Prior`; `prior/procedure.py` `PROCEDURE_MASKS`, `procedure_digests`; the function that opens a prior run (new, B10) |
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said, the candidates and the landings before the step to the inputs of a row (§2). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states, its own landings (D105) and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings: from the tracks roster, less the sealed test days (the index refuses them, D105), counted before a time without a given flight's own landing (D63). The motion of rows: the 2 s displacement, with the velocities east and north (D106) | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, the motion of rows (new, B10); `prior/loop.py` `LoopRows` (`select`; each aircraft's landings: new, B10); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings` |
| 3 | The speaker | Says one row from the inputs of the row, column by column, with a cache from row to row; under the masks of §4, the masks of a caller included; draws each word with the caller's random numbers (§4, D96); passes the caller's input of the added modules to the model; records the permitted words of each row and column; gives a copy of chosen aircraft (D96); refuses a model with any module in training mode (D107). A loop reads, and does not change, the words in force, the go-arounds said and the number of candidates (D106) | `prior/speaker.py` `Speaker` (`observe`, `speak`, `permitted`, `copy`; `heard`, `in_force`, `go_arounds`, `n_candidates`: read only), `Position`, `Permitted` (`select`; the join of several records: new, B10), `draw`, `go_around_bound`, `MOST_GO_AROUNDS` |
| 4 | The teacher-forced loss | The loss of each step and each column for a batch of sentences; the sentences of a split under a selection rule (D75) as batches; the log-probability of given words under the speaker's records of the permitted words, with the input of the added modules, with gradients (D96), refused for a model with any module in training mode (D107); the rows of a sentence that a loop said, with its words as targets, as one batch (D106) | `prior/train.py` `batch_nll`, `masked_log_probability`; `prior/source.py` `ArtefactSource`; `prior/selection.py` `require_rule`, `kept`; `prior/batch.py` `collate`, `RowTensors`; the rows of a sentence a loop said (new, B10) |
| 5 | A place in each layer | A module added at each layer whose output starts at zero leaves every output of the prior unchanged until it learns. Its input is what the caller gives the model and the speaker (item 3) | `prior/model.py` `Prior.add_at_each_layer` |
| 6 | The region of a final | For each candidate, whether a position is inside the FAF and the LPV cone: the region of the procedure masks (D64) and of the rows "on the final" (D72) | `prior/procedure.py` `airport_finals`, `Final.inside` |
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows, then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it | A module under `experiments/` (new, B10); free generation's numbers `flight_numbers` (moved into it, B10) |

---

## 8 Gates and identities

The gate of this document is the prior: the teacher-forced likelihood against baselines and free generation. The user
sets its criteria (D7). The identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table, the sha256 of the sentence files, the selection of its sentences (D75: the rule, and the sentences kept and left out for each split, airport, stratum and outcome), and the landings that the candidate vectors count (D63): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number left out on the sealed test days. A run that reads the landings again computes the digest again and refuses a difference |
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
| Procedure masks | Glidepath lower edge: published glidepath − 60 m inside the FAF and the LPV cone; the DA wherever the aircraft is not inside them; no climb before the join, once more than ε of the level nearest the entry height below the entry height; never "unchanged" blocked | §4, D64 |
| Input scales | Distances asinh(d / 1 km); heights / 1 km; height above a glidepath asinh(h / 100 m); ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65 |
| RoPE base | 10,000 (heads 32 wide: periods 6.3 s to approximately 35,000 s) | D65 |
| Go-arounds of a flight in free generation | At most 2; after the second, "go-around" is masked | D68 |
| Sentences of the base, the folds and the configurations | `landed`: only sentences whose own words land; train, select and the validation's teacher-forced loss | D75 |

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
when each milestone starts. Until the formal artefact (A21, then A25) exists, the tests use synthetic inputs and
artefacts. Then the code is tested on a sample of the formal artefact at Δ = 2 s, read-only. The smoke artefacts of
stage A are not used: their spec is not the spec of D56 and D58 (D55). The formal runs of B5 need Claude's check of
stage A and the chosen Δ (D11). Stage B changes no code of `instructions/` or `autopilot/`; it reads them only through
the vocabulary's public interface (vocabulary §6). The rules of outline §5 apply.

**Written from this document, not patched from the archive.** The archived `prior/`
(`archive/two_tier_v3_2026_10/prior/`) stays unchanged. A part of it comes back only where its logic fits this document,
rewritten into the new modules: the row-by-row encoding with a cache (`Prior.extend`), the candidate tokens with one
shared network and the pointer head, the ordered heads, the training loop (AdamW, warm-up, clipping, early stopping),
the index of the airport's landings, the rules of the procedure masks. Nothing else comes back: not the six-column
layout, the airport and row-position embeddings, the candidate constant table, the fixed count of candidate slots, the
variants and selection rule of `instruction-v3`, the aircraft attention of a one-aircraft scene, the checkpoint formats
`v3`–`v5`. No compatibility (principle 8).

**B0–B4 and B8 are done**; their specifications (what each builds, its tests) are in
`readouts/2026-10-05_stage_b_implementation_log.md` §3. The rules they build are in §2–§9.

| Milestone | What it built | Decisions |
|---|---|---|
| B0 | The package `prior/` and its import rules in `tests/test_architecture.py`; the runners `prior_train`, `prior_select`, `prior_free_generation` | — |
| B1 | The data: the inputs of a row, the landings, the artefact as sentences and the identity of the data | D13, D17, D23–D25, D32, D41, D60, D63 |
| B2 | The model and its checkpoint | D16, D41, D65 |
| B3 | The training loop and the runner `prior_train` | D31, D40 |
| B4 | The speaker, the procedure masks, free generation and its readout | D14, D33, D38, D52, D62, D64, D67–D72 |
| B8 | The selection of the base's sentences | D74, D75 |

**B5. Cross-validation and the base** (D39, D40, D41).

- The 31 training runs of §5, as one campaign on a clean checkout, one at a time on the GPU; before each step the
  behaviour check of D108, the commit of each step recorded as information.
- Every run reads the sentences of the selection `landed` (D75). For each fold: the held-out loss, the first-step
  runway at the held-out airport, the free generation at the held-out airport (200 flights × 2, from every flight;
  the flights outside the selection given apart).
- `prior_select` applies the rules of §5 and writes the choice. Then the base on all five airports, and its one
  validation readout: the teacher-forced loss, the free generation, the share of the labelled words that the masks
  block, the probability of "go-around" on the final (D72).
- No criterion is applied: the user reads the results (D7).

**B11. The selection leaves out flights with a faulty observed track (D111).** After vocabulary A40 is on the stage A
line that stage B merges; before B5's formal campaign. `prior/selection.py`: `landed` also leaves out a flight that
stage A's `instructions/faults.py` marks; the selection record counts them apart (by split, airport and stratum). The
change of the record is a change of the identity of a run's data (the checkpoint's schema; one new name with B10's
changes, as both come before B5). The mark is read like the stored outcome (D75): it keeps or leaves a sentence and
never reaches an input. Of the val days, the marks are counted in the identity and never printed (D85, as B10). Free
generation's readout gives the flights left out for a faulty track apart from those left out by their outcome (a
flight with a faulty start fails whatever is said: vocabulary D111). The report gives the selection's names for §7
item 4 if they change. Tests: a marked flight that landed is left out under `landed` and kept under `all`; the record
counts it apart; a change of a flight's mark changes no input of its rows; a marked flight's free generation is
unchanged; the readout counts the two reasons apart.

**B10. The corrections of Claude's check of stage B** (`readouts/2026-10-05_stage_b_check.zh.md`; D64, D72, D85,
D105–D108). Now; before B5's formal campaign, because it changes free generation and the campaign runner. On synthetic
artefacts; then B3's smoke and free generation on A34's artefact again. The report gives the names of the new parts
for the column "Code" of §7.

- D85: no printed text, log or summary of the prior shows a reading of the val days before the base's one validation
  readout: `prior_train` prints the selection of train and select only (the identity keeps every split, D75). A
  validation readout of the base is exported only when it is the one that the claim of the val read names (its output
  and the base's run).
- D105: the function of a loop's row takes each aircraft's landings; `Landing` and the construction of a
  `LandingIndex` from given landings are public; the index refuses a landing on a sealed test day.
- D106: (1) the step of a speaker's closed loop as one module under `experiments/`, with `flight_numbers` moved into it;
  free generation calls it; (2) the function that opens a prior run, used by every runner of the prior; (3) the rows of
  a sentence a loop said as one batch with its targets; (4) the join of records; (5) the motion of rows in §7; (6) the
  speaker's state named in §7.
- D107: the speaker and the log-probability under records refuse a model of which any module is in training mode (not
  only the top module).
- D108: the campaign's behaviour check before each step in place of the comparison of the commit.
- The speaker changes nothing before a row is said (as vocabulary D80 for the start): the state of the procedure
  masks, the cache and the records change only after every column of the row has a word; `observe` is refused after
  the first `speak`; both refuse a row not later than the last row encoded, observed or said.
- D64: at the row whose runway word ends G, the procedure masks keep the row's state (G is false after that word):
  the join and the passage below the entry height start again at that row, not one row later.
- D72: "on the final" is the region of the runway in force before the row, the runway under which the speaker drew
  the word.
- Free generation's memory at the formal size (outline §5 rule 13): a chunk in which one sentence reaches the time of
  its go-arounds (the cache of every aircraft of the chunk grows with it), measured on the GPU before B5.
- Tests: a change of the observed samples after the first predicted step and of the time limit changes no input of a
  loop's row (vocabulary D90); D23's test also changes the flight's record (its runway and landing time) and its own
  landing in the index; in free generation, a change of every field that it must not read (the stored flown states,
  the words, the correction marks, the withheld fields of vocabulary D82, the landing time, the own landing) changes
  no word, state or readout; the loop's rows equal the sentence's rows at Δ = 8 s too; a smoke of the validation
  readout reads no outcome of a val sentence (a test that fails on such a read); the export of a val readout (refused
  unless claimed); the frontend's fixtures of stage B written by the export, the index by its own writer; a refused row
  leaves the speaker as it was; a module in training mode inside an eval model is refused; free generation through the
  shared step gives the same words, states and readout, bit for bit, as before the move.
- Outline D109, after vocabulary A39 is on this branch: the prior's sets give the Training view's reader and live
  segment their splits; a set exported from the base's claimed validation readout gives val too, and only it. Tests: a
  val set of a claimed readout opens and flies live; any other set with a val flight is refused.

**B6. The Training view of stage B (outline §6).** After A23 is on this branch; the publication of the folds and the
base after B5. The user sees what the prior says and how the executor flies it.

- **Export** (the archived prior Training exports, rewritten). For each flight of the sample: the observed rows before
  the first predicted step; the sentences that the prior says in free generation and their flown states (several
  sentences of one flight side by side); the closed-loop sentence of the same flight; the outcome and the DA check of
  each; at each row, the words that the procedure masks blocked; the region, the glidepath lower edge, the DA and the
  entry height of R. Sets: each fold of B5 at its held-out airport (the flights of its free generation), and the base
  model (its one validation readout). Before B5, a smoke set from the smoke model of B3 checks the view. New schema
  names; its own index beside the old one (outline §6 item 3).
- **Frontend.** The Training view of A23 with the prior's sentences: the five columns, a choice of sentence, the
  blocked words at a row, the procedure's limits drawn, the outcome. A click on a word flies its segment live with the
  executor of A23.
- **Publication and view.** The intent of each set in `docs/experiments/intents.json`; a test stack from the worktree
  (outline §6 items 4, 5).
- **Tests.** The export (a sample written and read again); the frontend's readers on fixtures that the export writes;
  a live segment equals the export's flown states; the browser check (outline §6 item 6).

**B7. Close of stage B.** The full ts suite passes (run detached). §0.3 (the log) and `docs/reference/runners.md` are
updated; the report gives the new code index for §11 (outline §5 rule 10). `dev-two-tier-v4` merges
`dev-two-tier-v4-prior` (outline §5 rule 1). Report to the user: the commits, the readings of each fold and of the base,
the choice and its rule, and what stage C needs.
