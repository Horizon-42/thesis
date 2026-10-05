# Two-tier model: the prior

**Summary.** The prior is the upper part of the two-tier model (outline §1): a language model of the controller. At
each Δ row it says one row of words for each aircraft, and the executor flies them. This document gives its inputs,
its outputs, its decoding, its training and how it runs at a new airport. It is stage B of the plan. It reads the
vocabulary only through the vocabulary's public interface (vocabulary §6) and its decisions; the post-training reads
this document only through its public interface (§7) and its decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete for its part. The principles and the shared rules are in the
outline (`outline.md`). The code is on the branch `dev-two-tier-v4-prior` (§0.3). Paths in backticks are relative to
`4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the repository root; `readouts/` is
`docs/two_tier/readouts/`.

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
| D40 | Hyperparameters: configuration A (§5) is the start. The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§5); a tie of parameters among them goes to the lower score (A and D have one shape). Every configuration has attention heads 32 wide: A 6 heads, B (d_model 128) 4, C (d_model 256) 8 | Decided | User, 2026-10-04; the head width: user, 2026-10-04, on Claude's proposal; the tie: user, 2026-10-05 |
| D41 | The design runs at an airport that is not in the training data: the number of candidates is not fixed; every input has a fixed physical scale, never a statistic of the training data; each fold of D39 flies the full closed loop at its held-out airport. The altitude words are above the airport elevation (D58), so the levels of an approach lie in the 60 m segment at any airport (§6) | Decided | User, 2026-10-04 |
| D58 | The prior's own height is the height above the airport elevation E, as the altitude words are (vocabulary, D58). The prior gets neither E nor the MSL height (D24), so it cannot know where the round MSL levels that controllers assign lie (§2). (The words: D58 in the vocabulary) | Decided | User, 2026-10-04 |
| D60 | Row 0 of an aircraft has no motion inputs: no state 2 s before it is stored. There, the ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and the own input `no_motion` is 1 (0 at every other row). The same at every Δ, in training, in closed loop and in a loop of several aircraft. Never a stored or fitted velocity, which reads rows after the row (§2) | Decided | User, 2026-10-04, on Claude's proposal |
| D63 | The identity of a prior's data (§8, item 1) also holds the landings that the candidate vectors count: for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would change the inputs and leave the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |
| D64 | The word rules of the procedure masks (§4). They block a word when it is said and never block "unchanged". Inside the region (the FAF and the LPV cone of R) a level T only where T ≥ the glidepath lower edge − ε(T), and "no level-off" not where the aircraft is more than ε("no level-off") below the edge. Wherever the aircraft is not inside the region (before the join, or after it when the aircraft has left the region), a level T only where T ≥ DA − ε(T). Before the join, once the aircraft has been below the entry height by more than the band ε of the level nearest the entry height, no level T above the aircraft's height + ε(T) and no climb class. A go-around clears the join and the passage below the entry height; while G is true neither is kept; at the row whose runway word ends G, they start again with that row's own state (a change of runway outside a go-around: D122) | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D65 | The fixed values of the inputs and the model: the scales of §9 (the height above the glidepath as asinh(h / 100 m)); the candidate tokens reach a row through one attention over them whose weights sum to one; the RoPE base is 10,000; the variant `constants` gives the threshold elevation MSL; the runway column's `since` starts again at a candidate word, not at "go-around" (§2) | Decided | User, 2026-10-04, on the proposals of the stage B agent and Claude's review |
| D68 | Free generation lets a flight say at most 2 go-arounds. The start of the closed loop gets 2 as the most go-arounds (vocabulary §6, item 5; D67), and after a flight's second, the step of a speaker's closed loop (§7 item 7) forbids "go-around" in the runway column for every caller, from the loop's most go-arounds (a mask joined with the caller's runway mask, §4). The readout counts the flights that reached the bound. Why: the executor lays out the time that go-arounds add (900 s each) when it starts, and the number that a model says is not known before; no closed-loop sentence of `v9_20261004` at Δ = 2 s has more than one go-around (train 68 of 40,534 sentences, select 12 of 6,199); 2 lets a model say one more than the data, and no flight goes around without end | Decided | User, 2026-10-04, on the report of the stage B agent |
| D72 | The readout's probability of "go-around" on the final (B4, B5) is read on the rows on the final (inside the region of the runway in force before the row — the runway under which the speaker drew the row's runway word; D64) where the masks permit "go-around": not while G is true, not after the bound of D68. The probability is that of the distribution the speaker draws the runway word from, the masks applied. Why: at a row where the word is masked, its probability is 0 because of the mask, not the model, and would pull the mean down | Decided | User, 2026-10-04, on the proposal of the stage B agent |
| D75 | The base learns only from closed-loop sentences whose own words land: a selection of the artefact's sentences by a named rule (`all`, `landed`), applied when they are read; the artefact is not changed. `landed` keeps a sentence whose stored outcome (vocabulary §6, item 3; D74) is a landing. It applies to train and select (the stop and the choices of D39 and D40 read the same kind of sentence that the training reads) and to the teacher-forced loss of the validation readout; free generation starts from every flight, and the readout gives the flights outside the selection apart. A run records the rule and, for each split, airport, stratum and outcome, the sentences kept and left out (§8, item 1). Why: about 2.2–2.6 % of the sentences do not land when their own words are flown (A25, Δ = 2 s: train 1,955 of 1,999 replayed, select 6,036 of 6,199; KMSY 94–95 %), mostly high at the DA; their last words teach an end that does not land, and the vocabulary and the executor, not the words, are often the cause, so imitation cannot correct them. They come back in the post-training as starts (post-training D76) (milestone B8) | Decided | User, 2026-10-04 |
| D96 | The public interface gives what the post-training needs (§7), made by stage B: (1) the speaker passes the caller's input of the added modules (§7 item 5) to the model at every row that it encodes or says; (2) the caller gives the random numbers: one uniform number for each aircraft, row and column, and the speaker draws, from the masked distribution, the word whose cumulative probability (in the class order of the heads) first passes the number; an aircraft's words thus depend only on its own numbers and inputs (the probabilities up to their last bits: post-training §6.4); (3) for each row said, the speaker records the words that all masks permitted in each column, as an object that the caller keeps and gives back but does not read; the teacher-forced part gives the log-probability of given words under such records, with the input of the added modules, with gradients; at the parameters that spoke, it equals the probability that the speaker drew from, within the float tolerance (a test); (4) the inputs of a row in a loop are one function of `prior/`: from each aircraft's states on the 2 s rows (`STATE_COLUMNS`), its landings and the speaker's words in force, the inputs of the next Δ row and the position that the masks read; free generation calls it; (5) a copy of chosen aircraft of a speaker, repeats permitted: the cache, the state of the procedure masks, the words heard, the go-arounds said and the records; a copy continued with the same inputs and numbers says what the original says, bit for bit in a batch of the same layout (a test); (6) §7 gives the code of each item, the region of a candidate's final (D64) and the reading of a split's sentences under a selection rule (D75) as batches included; the post-training's architecture test holds its code to these names (post-training C0). Why: the post-training's branch groups need each aircraft's own random numbers and a copy of the state at a branch point (post-training D94); its loss needs the masked distribution (post-training §2 item 5); its traffic attention needs its input while the prior speaks; and it needs one function for a loop's row and one region of the final (post-training D92). Without these, it would copy code of `prior/`: a second definition (milestone B9) | Decided | User, 2026-10-05, on Claude's report of stage C's readiness |
| D105 | The landings that a loop's row counts (§2, §7 item 2) are given for each aircraft, not one index for each airport: the function of a loop's row takes each aircraft's landings. A caller builds them from the roster's landings (`Landing`, `LandingIndex`, in §7 item 2); the index refuses a landing on a sealed test day by itself (C32), not only the reader of the roster. Free generation gives each flight its airport's roster landings. (The window's landings: D105 in the post-training) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D106 | Stage B gives what the post-training's window loop needs, so that it copies no code of `prior/` or of free generation (extends D96; milestone B10): (1) the step of a speaker's closed loop as one module under `experiments/`, shared by free generation and the post-training's window loop: the observed rows up to the first predicted step, then at each Δ row the inputs of the row (§7 item 2), the speaker with the caller's random numbers, masks of a caller and input of the added modules (given by the caller for each row), the executor's step through the start of a closed loop (vocabulary §6 item 5), the flights that are done halted; the caller can end a flight (a loss of separation, post-training D93); its state can be copied for chosen flights (post-training D94; the loop's own copy is vocabulary D97); free generation calls it; (2) one function that opens a prior run: the checkpoint, the identity of its data and the check of its procedure masks (§8 items 1, 2); every runner of the prior uses it; (3) the rows of a sentence that a loop said, with its words as targets, as one batch for the log-probability of §7 item 4; (4) a join of the records of several speakers into one batch, padded; (5) the motion of rows (the 2 s displacement, D25, D60), with the velocities east and north, in §7 item 2; (6) the state of a speaker that a loop reads, named in §7 item 3, read only: the words in force, the go-arounds said, the number of candidates. Why: with one step for both loops, post-training C4's test (a window without other aircraft equals free generation, bit for bit) runs one code, not a copy — the observed rows encoded in one call instead of one at a time already move the probabilities by 1.2e-7 (Claude's check of stage B, `readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D107 | The speaker and the log-probability of given words under records (§7 items 3, 4) refuse a model of which any module is in training mode: the speaker speaks with dropout off, and the log-probability under its records is that of the policy that spoke only with dropout off. The teacher-forced loss of item 4 (`batch_nll`) does not refuse. (Which term of the post-training runs in which mode: D107 in the post-training) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D108 | B5's campaign runs on a clean checkout. Before each step it runs a behaviour check of the prior's code on fixed inputs: on the CPU with one thread, two steps of training on a fixed synthetic batch and some rows said by the speaker with fixed numbers; the losses, the words and the probabilities are compared, bit for bit, with those that the campaign recorded at its start, and a difference stops the campaign by name. The commit of each step is recorded as information and never compared. Why: results of different code are comparable once the code is shown to behave the same on fixed inputs, never by an equal commit (the user, 2026-10-02); no code fingerprint guards a run (the user, 2026-10-04) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D111 | The selection `landed` (D75) also leaves out a flight that stage A marks as having a faulty observed track (vocabulary D111, §6 item 3), whatever its outcome; `all` keeps every sentence. The run's record gives, for each split, airport and stratum, the sentences left out for a faulty track apart from those left out by their outcome (§8 item 1). Free generation and the post-training's starts are not changed (the user: the post-training may start from flights that do not land). Why: the closed-loop sentences of some flights with ADS-B faults land, and their observed rows — which the prior reads — hold the fault | Decided | User, 2026-10-05 |
| D118 | Stage B's readings of B9–B11, accepted by the user: (1) a faulty flight that also did not land is left out for its fault (the mark is asked first); (2) the identity of a prior's data also holds the sha256 of each split's stored signals (§8 item 1): the faulty-track marks are read from the signals, so a reader before the claim of the val read (`open_prior`) checks the val marks by the files' sha256 without reading them (D85) — the training run itself reads val's outcomes and marks to count its identity, held and never shown (D120); it binds data, not code — it changes only when the artefact is built again, when the sentence files' sha256 changes too, so no change of code refuses a prior unless it changes which flights the selection keeps; (3) the claim of the val read names its readout relative to the repository; the base's one validation readout is two readers, each with its own claim (`prior_validation`: the teacher-forced loss and the masks on the labelled words; `prior_free_generation`: the free generation, whose set the Training view opens, D109); (4) the claimed validation set flies on a second service of stage A's live segment that permits val alone (outline D109), so stage A's code is not changed; (5) after the claim, the val readers count val's selection again and refuse a difference before they read (with (2), only a change of stage A's fault rule that moves no train or select mark can cause it; nothing is written, so the claim may be run again, D119); (6) a flight that the caller ends (post-training D93) gets the caller's outcome: the closed loop's step gives its words and states and refuses to give it an outcome of the judge; (7) D108's behaviour check takes the artefact's vocabulary spec and its first airport's finals as its fixed inputs, and covers two steps of training, the speaker, the input functions, the variant `constants`, the first-step runway and the closed loop's step on a straight-flying stand-in; the campaign also compares its own settings (the seeds, the selection, the configurations, free generation, the temperature, D68's bound); (8) a flight's random numbers in free generation come from numpy `default_rng([seed, sample, index])`; the log-probability under records is 0 at a row not asked, and an asked row reads the record said at its time, its own-state inputs checked against the record's. Free generation and the validation readout write the identity without the val counts (D85) | Decided | User, 2026-10-05 |
| D119 | The one val read of a reader (D85, D118 item 3) is spent when the reader has written its readout. A reader claims the val read before it reads, and the claim names its output; a claim whose output holds no written readout (a run that stopped before it wrote: a crash, a kill) may be run again to the same output, and only to it. A runner checks its own options before it claims. A copy of a prior's run directory could read val again: a deliberate bypass, which the claim does not guard (it guards against accidents). B5's code spends the claim when it is written: if a val readout of B5 stops before it writes, Claude reports it and the user decides on the claim file (milestone B12 changes the code). Why: a claim spent at once refused the base's val readouts for ever after a crash, and the campaign's resume could not complete its last steps (Claude's second check of stage B, `readouts/2026-10-05_stage_b_check_2.zh.md` §2 item 1) | Decided | User, 2026-10-05, on Claude's second check of stage B |
| D120 | The identity of a prior run's data holds the counts of the selection of every split (§8 item 1), val included: the training run counts them. No text shows the val counts before the base's one validation readout (D85): the readouts write `checkpoint.readable_identity` (train and select only), and a report or summary reads no `identity.selection.counts.val` of a run's `config.json` or checkpoint. Why: they are counts of stage A's closed-loop sentences on val (outcomes, faulty-track marks); no input, stop or choice reads them; a change of the format after B5 would refuse B5's checkpoints (Claude's second check, §2 item 2) | Decided | User, 2026-10-05, on Claude's second check of stage B |
| D121 | The speaker draws from the masked distribution at a temperature: the logits divided by it before the softmax (§4). Free generation, the readouts and the campaign of stage B use 1, the model's own distribution. The records of the permitted words hold the temperature, the log-probability under records uses it, and one batch of records has one temperature (§7 items 3, 4). A caller may set another (the post-training) | Decided | User, 2026-10-05, on Claude's second check of stage B |
| D122 | The procedure masks keep, for each candidate, whether the aircraft has joined (been inside its region) and whether it has been below its entry height before the join, from its row 0 or from its last go-around (D64), whatever runway was in force at those rows: a change of runway outside a go-around starts neither again, and the new runway has the rows flown under the one before. Why: on the select days, none of the 5,861 sentences of `landed` at Δ = 4 s changes its runway outside a go-around, and the two readings differ on 0 rows (Claude's second check, §3 item 5): it bears on free generation | Decided | User, 2026-10-05, on Claude's second check of stage B |
| D126 | §7 item 8 lists the function of stage B's Training export that the post-training's window export imports (`procedure_block`); the post-training imports exactly this name. Names only: no code changes. (The part of the vocabulary: D126 there) | Decided | User, 2026-10-05, on stage C's request |

### 0.2 Open items

None.

### 0.3 Implementation

The commits, test counts and measurements of every milestone are in `readouts/2026-10-05_stage_b_implementation_log.md`
§1; the specifications of the milestones that are done are in its §3.

| Part | State |
|---|---|
| B0–B4, B8–B11 | Done, each reviewed. The code is on `dev-two-tier-v4-prior` (worktree `.claude/worktrees/two-tier-v4-prior`) and in `dev-two-tier` at `44f6000f`. B9: `1204fd8f`–`5313b6cd`; B10: `e4e7ba42`; B11 and outline D109: `df68c927`; stage A's line A37–A42 followed: `c072e355`, `89845fa6`, `5818fbbf` |
| B6 | The export, the frontend and the live segment are done and checked on smoke sets (`b079ebef`, `ce601bd7`, `aac93945`, `201b4afd`, `a280d984`; the claimed val set `df68c927`). The publication of B5's sets waits for B5 |
| B5 | Running since 2026-10-05 20:10 UTC: `4dTrajectory/outputs/POOLED/prior/prior_base_20261005`, on the artefact `v12_20261005` and the executor `v17_20261005` at Δ = 4 s, from `dev-two-tier` at `deaf9690` (`44f6000f` and its intent) |
| B12 | Ordered (`notes/stage_b.md`); on `dev-two-tier-v4-prior`, merged into `dev-two-tier` after B5 |
| B7 | After B5, B12 and B6's publication |
| Claude's checks | At `5313b6cd` (`readouts/2026-10-05_stage_b_check.zh.md`; corrections B10, D105–D108) and at `44f6000f` (`readouts/2026-10-05_stage_b_check_2.zh.md`; corrections B12, D119–D122). Neither found a leak into an input, a mask or a choice |

Proposals (where the design says nothing): none open.

### 0.4 Plan

1. **B5** (running, §12). A report of it reads no val count of a run's identity (D120). If a val readout stops after its
   claim, the user decides on the claim file (D119).
2. **B12** (now, §12), on stage B's branch. No code goes into `dev-two-tier` while B5 runs: B5's guard before each step
   does not see every change of the code (B12 item 2), and B12's behaviour check gives another answer, which would stop
   B5.
3. **B6's publication**, after B5: the set of each fold at its held-out airport and the base's claimed val set (outline
   D109).
4. **B7**, after B5, B12 and B6's publication.
5. The post-training (`post_training.md`) is built in parallel on its own branch (outline §4, D95). It reads only §7;
   its formal runs wait for B5's base.

---

## 1 Scope

- **This document owns** the package `prior/` (the inputs, the model, the training, the speaker and its masks), its
  runners (`experiments/prior_*.py`), and the Training view of its results (B6, outline §6).
- **It reads** the outline (the principles, the shared decisions D7, D20, D21, D55, D85, the rules of the
  implementation) and the vocabulary's public interface (vocabulary §6): the vocabulary spec and the grammar (items 1,
  2), the sentence artefact (item 3), the candidates and their geometry (item 4) and the row grid (item 7). The package
  `prior/` does not import the executor or the judge (items 5, 6): a runner joins them through the start of a closed
  loop (item 5, D67), in the step of a speaker's closed loop (§7 item 7).
- **It gives** the public interface of §7, and nothing else, to the post-training.

---

## 2 Inputs

The inputs of a row are what a controller knows before the row's words (outline principle 7). The words of a row are
its targets; the inputs of the row are computed before them.

- **Source of the states (D32).** The rows before the first predicted step are observed (ADS-B), as in closed loop.
  From the first predicted step on, every input that comes from a state (the own state, the candidate vectors, the
  motion) comes from the flown states of the closed-loop sentences (vocabulary §6, item 3), not from the observed track.
  The motion of the first predicted step itself is its state minus the observed row 2 s before it: D60 leaves no other
  value, and the position of the first predicted step is that of the observed row (the start rule, vocabulary D77). The
  words in force are those of the closed-loop sentence, corrections included. In closed loop the states come from the
  executor.
- **Own state (D5, D23, D58, D60).** The own state of the aircraft has no frame: its height above the airport elevation
  E (the level words are above it), its ground speed, its vertical rate and `no_motion` (1 only at row 0, below). No MSL
  height: with it, the airport elevation (a constant of the airport) would be an input (D24).
- **Frame (D5, D23).** Every position and every direction is in the candidate vectors. Each candidate vector gives the
  aircraft in the frame of that candidate: the distance before its threshold along its course, the offset right of its
  final, the height above its threshold, the motion direction minus its course (sine, cosine)
  (`instructions.airport.relative_to_runway`). "The frame of R" is the candidate vector of R, which the prior gets as
  the runway in force. There is no second set of values relative to R.
- **No input is computed from R before R is said (D23).** The first predicted step says R. The inputs of the rows before
  it and of the first predicted step itself use no value computed from R. (The heads of a row see the runway that the
  same row said, in the order of the columns; that is an output, not an input of the row.) The reason: the artefact
  writes the runway word at the first predicted step (vocabulary §6, item 3), and its value is the runway on which the
  flight landed. An input of these rows computed from it gives the answer of the first predicted step, and in closed
  loop the same input does not exist. A test: a change of the runway word of a sentence, of its flight's record (its
  runway and landing time) and of its own landing in the index leaves the inputs of rows 0 to the first predicted step
  the same, bit for bit.
- **No airport identity (D5, D24).** The prior has no airport embedding, no absolute position (E, N), no absolute
  motion direction, and no table of candidate constants (the absolute threshold position, the course, the elevation,
  the length).
- **Candidates (D24).** A candidate vector has only values that change with the aircraft: the values of the frame above,
  the glidepath height below, and the landings on the candidate in the 30 min before the step. It has no constant of
  its runway. Such a constant identifies the runway:
  - In the candidate table of the formal artefact (`v12_20261005`: 25 candidates at five airports), the pair (length,
    threshold elevation) differs for 24 of the 25 candidates (KSJC 12L and 12R share it). The length alone tells the
    left runway of KRDU from the right one (05L / 23R 3,048 m, 05R / 23L 2,286 m).
  - The spacing of parallel runways is different at each airport: KRDU 1,068 m, KSJC 213 m, KSMF 1,827 m, KSTL 397 /
    855 / 1,251 m. A layout relative to R is a code of the airport when R is known.
  - The runway-intent study found this channel (R1.1, gradient-boosted trees, split by day;
    [plan](../../history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md) §15): with a constant of each runway
    (`prior_share`), the trees recognised "30L" and were right on only 53 / 59 % of the flights on the two KSJC days
    with 30L closed. Without it (R1.1b) they were right on 96.8 / 97.6 %. A prior uses an airport identity when it gets
    one (§10).

  The constants are also not necessary. The length is a cause only together with the aircraft type, and the prior has no
  aircraft type. The elevation is not needed: the words and the own height are above the airport, and each candidate
  gives the height above its own threshold. (The own height minus a candidate's height above its threshold is that
  threshold's height relative to E: from 0 to 28 m below E at the five airports, a small constant of the runway that two
  physical inputs give together.) Which of two parallel runways is the left one shows at each step in the offsets right
  of their finals (the value of the left runway is always larger). The relative positions of all candidates together
  still show the layout of the airport. That is real geometry, and the prior has it; only the held-out airports of D39
  can measure how much the prior uses it. A constant comes back only as the variant `constants` of D39: the length and
  the threshold elevation MSL (D65). That variant thus also gives the airport elevation E, which `full` does not (D58).
- **Landings (D63, D105).** The number of landings on each candidate in the 30 min before the step: in [t − 30 min, t),
  t the row's UTC time. Offline, the landings come from the airport's tracks roster: every record that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). The index of
  landings refuses a landing on a test day or on a day outside the day split by itself. A flight never counts its own
  landing: a closed-loop sentence can fly past the time at which the observed aircraft landed, and the runway of that
  landing is the answer of the runway word; the own landing must be in the index. In a loop, the caller gives each
  aircraft the landings that the loop knows (D105). The roster is outside the artefact, so the identity of a prior's
  data holds a digest of the landings (§8, item 1).
- **Candidate tokens (D41, D65).** Each candidate vector goes through one shared network. The candidate tokens reach a
  row through one attention over them whose weights sum to one: a sum would grow with the number of candidates.
- **Scales (D41, D65).** Every input has a fixed scale in SI units (§9): the distances along and across a candidate
  asinh(d / 1 km), linear within a few hundred metres of a final; the heights (above E, above a threshold) h / 1 km; the
  height above a glidepath asinh(h / 100 m), linear near the glidepath, where the value has a meaning, and compressed
  far from it (an aircraft high above an opposite runway's extended glidepath is thousands of metres above it); the
  ground speed v / 100 m/s; the vertical rate v / 10 m/s; the landings in 30 min n / 10; in the variant `constants` the
  length l / 1 km and the threshold elevation MSL h / 1 km.
- **Words in force** before the row: the runway in force as its candidate's token ("none yet" up to and with the first
  predicted step), the go-around state G, the heading in force as the sine and cosine of its angle relative to the
  course of R, the altitude, angle and speed words as embeddings (each "none yet" up to and with the first predicted
  step), and the time since each column said its word (D17).
- **Glidepath (D13, D23).** Each candidate vector has the height above the published glidepath of that candidate at the
  aircraft's distance before its threshold, with the straight-line reference of the vocabulary (vocabulary §6, item 4:
  TCH + d·tan(angle) + d²/(2R_e), R_e the earth's radius of curvature along the course). The value in the vector of R is
  the input of D13. Thus the value exists at every row, also before R is said. Every candidate has a TCH and a glidepath
  angle (the vocabulary refuses a candidate without them). It is procedure geometry as an input; the model still decides
  the profile (principle 2). The value is only meaningful near the final; the model also has the offset from the final
  to weigh it.
- **Motion (D25).** The ground speed, the vertical rate and the motion direction (in each candidate vector, minus its
  course) come from the displacement in the 2 s before the row, at every row interval Δ. Only past positions give them;
  never the stored track, ground speed or vertical rate of the states: on the observed rows these hold the velocity of
  the start rule (vocabulary D77), which at row 0 reads the row after it, and the signals hold a least-squares fit over
  15 s centred on the row, which uses 7.5 s of the future.
- **Row 0 (D60).** Row 0 of an aircraft has no state 2 s before it: the observed data start at the entry of the 25 km
  slice, and the sentence artefact stores the states from row 0 of each sentence (vocabulary §6, item 3). At row 0 the
  ground speed, the vertical rate and the motion direction in each candidate vector (sine, cosine) are 0, and
  `no_motion` is 1; at every other row `no_motion` is 0. The rule is the same at every Δ, also when the data have a 2 s
  row before row 0, so that only the interval changes with Δ (D25). It is the same in training, in closed loop and in a
  loop of several aircraft (§7, item 2): the first row of an aircraft never has a state before it. The zeros alone
  already mark the row (no aircraft in flight has a ground speed of 0, and (0, 0) is not a direction); `no_motion` says
  it directly, so that the network does not have to learn it from an extreme value. Two other values are not used:
  - A fitted or stored velocity at row 0. It reads rows after the row. For the aircraft's own first predicted step, 16 s
    after row 0, this is no leak; but in a loop of several aircraft another aircraft reads the row at its own time, and
    then reads the future (principle 7). Also, it would be another kind of value at one row than at every other row.
  - The displacement from row 0 to the next row: in a loop of several aircraft, that is 2 s of the future.
- **Time (D16).** No row position embedding and no input "time from row 0": both measure the time since the aircraft
  entered the 25 km slice, a cut of the data, and a long sentence (a go-around adds up to 900 s) reaches rows that
  training seldom saw. The causal time attention gives the order. RoPE in the time attention gives how far back each
  earlier row is: the rotation of a row's query and key uses its time in seconds, so the attention reads only time
  differences. Seconds, not rows, so that every row interval of D11 reads the same time. The time of a row is in seconds
  from the aircraft's own Δ row 0. Only differences count, so the zero changes no result; but a UTC time (approximately
  1.76e9 s) in float32 has a step of 128 s and loses the rows. RoPE works with the row-by-row cache of the speaker
  (`Prior.extend`): a key is rotated once, by its own time, when it is written. The RoPE base is 10,000 (D65): with heads
  32 wide, the periods go from 6.3 s to approximately 35,000 s.
- **Time since each word (D17).** Each column has the input "time since this column said its word in force" (`since`):
  log(1 + t / 2 s) / 5, with t in seconds; 0 up to and with the first predicted step, which says every column. It is in seconds
  for every column, so that every row interval of D11 reads the same time. In the labelled data the runway column's
  value is the time since the first predicted step, because a labelled sentence says its runway only there, and again
  only at the runway word that ends a go-around (D19, D26). In closed loop it is the time since the runway was given or
  given again (a change of runway, or the runway word that ends a go-around), so it measures a real fact. A candidate
  word starts it again; "go-around" does not, because it keeps R (D10) and G is an input of its own (D65).

**Why the motion inputs do not change with Δ (D25).** If the motion of a row were the displacement since the row before,
it would be the mean over Δ. In a 3°/s turn, the direction of that mean is approximately 12° behind the track at
Δ = 8 s (3° at 2 s). The ablation of Δ would then compare a coarser input as well as a longer interval. With the
displacement in the 2 s before the row, only the interval changes. The data has a row every 2 s, and in closed loop the
executor has a state every 1 s, so the value exists at every Δ, at every row after row 0 (D60); the closed-loop
artefact stores the flown states on the 2 s rows (D51). At Δ = 2 s it is the value of now.

## 3 Outputs

Five heads in the order of the columns (vocabulary §6, item 1). Each head reads the row's hidden state and the words of
the earlier columns of the same row: the true words in teacher forcing, the words said when the prior speaks. The runway
head scores "unchanged", "go-around" and each candidate; a candidate's score is the row's query against the candidate's
token (a pointer), so the number of candidates is the airport's own and no slot count is fixed by the model (D41). Each
other head scores "unchanged" and each word of its column in the vocabulary spec.

## 4 Decoding

The speaker says a row column by column, in the order of the columns. A later column sees the words of the earlier
columns of the same row. A word is said only where every mask permits it:

**The model's own.** At the first predicted step, "unchanged" in every column and "go-around" in the runway column; a
padded candidate everywhere.

**The grammar.** Rules 1–6 and the runway/G table, the vocabulary's one function (vocabulary §6, item 2). The speaker
asks the grammar column by column, after the earlier columns of the row (D62). It gives the grammar, for each later
column, the words that its other masks permit there; in the runway column it asks once for each runway word, with the
later columns' masks under that word. A row thus never reaches a column with no permitted word.

**Procedure masks (D64).** The prior decodes under three masks from the procedure of R (principle 3), on the altitude
and angle words. The first two are lower limits on the altitude words; the third blocks the climb. The region of a
candidate is between its threshold and its FAF along its course, inside its LPV cone (the FAF and the cone from the FAA
CIFP procedure, §6); the join is the first row inside the region. Heights are above E. ε is the band of a level
(vocabulary §6, item 1; D52):

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

The masks of a row read the runway and G in force after the row's runway word, and the aircraft's position at the row.

**A mask blocks a word when it is said, never "unchanged".** A word in force was permitted when it was said. A mask
that made the speaker change it, for example a level forced above an aircraft that sank under the edge with
"no level-off" in force, would fly the aircraft along a line that the procedure computes: the glidepath floor that the
executor does not have (D9). Whether to correct or to go around is the model's decision, and the DA check judges it
(vocabulary §6, item 6).

**Masks while G is true (D14).** Only one mask stops a go-around: "no climb below the entry height". It does not apply
while G is true. A go-around clears the join and the passage below the entry height, and while G is true neither is
kept: the next approach is read as a new one (D26). At the row whose runway word ends G, they start again with that
row's own state. The other two masks are lower limits; a go-around climbs above them, so they apply. The grammar
applies while G is true as at every row.

**A change of runway outside a go-around (D122).** The masks keep, for each candidate, the join and the passage below
its entry height from row 0 or from the last go-around, whatever runway was in force at those rows. A change of runway
starts neither again: the new runway has the rows flown under the one before.

A prior speaks under the set of procedure masks that it was trained under, on the procedure data whose digests it
records (§8 item 2).

**Masks of a caller.** The loop that runs the speaker can give, for each column, the words it permits (§7, item 3).
The speaker applies them as it applies the others. The prior does not know what they mean. The step of a speaker's
closed loop (§7 item 7) gives one for every caller, joined with the caller's own: "go-around" after a flight's second
go-around (D68).

**Drawing (D96, D121).** The caller gives one uniform random number in [0, 1) for each aircraft, row and column. The
speaker draws, from the distribution with all masks applied at a temperature (the logits divided by it; 1 in every run
of stage B), the word whose cumulative probability, in the class order of the heads, first passes the number. An
aircraft's words thus depend only on its own numbers and its own inputs, not on the other aircraft of the batch (the
probabilities up to their last bits: 2.2e-10 measured, Claude's second check). For each row, the speaker records the
words that all masks permitted in each column (§7, items 3 and 4).

## 5 Training

**The data.** Teacher forcing on the closed-loop sentences of the selection `landed` (D75, D111), split by operating
day (the artefact's day split, `data/day_split_20260924.json`; test days sealed). A run reads the train and select days
of its airports only; the validation days are read one time, by the base's validation readout. KAUS is a held-out test
airport only: it is read one time, at the end of the whole chain.

**The loss.** At every asked row (a row of the sentence at or after the first predicted step), the cross entropy of each
of the five columns, each head reading the true words of the earlier columns of the row. The per-step loss of a set of
sentences is the sum over its asked rows and the five columns, divided by the number of asked rows. A training batch's
loss is its own per-step loss.

**The loop (D31, D40).** AdamW; the learning rate rises linearly over the warm-up steps and then stays constant; the
gradient norm is clipped. A batch holds sentences of similar length, at most the given number of padded rows (the
observed rows before the first predicted step and the padding included; a longer sentence is a batch alone); new
batches and a new order every epoch. After each epoch, the per-step loss on the select sentences, with dropout off. The
run keeps the weights of the epoch with the smallest one and stops after 3 epochs without a smaller one, or at 30
epochs. The seed sets the first weights, the batches and the dropout.

**Hyperparameters (D40).** Configuration A, the start: d_model 192, 4 layers, 6 attention heads, feed-forward 768,
dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ with 500 warm-up steps, gradient clip 1.0, 16,384 padded rows in a
batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is 10,000 (D65); a test
makes sure that a shift of all times changes nothing.

**Cross-validation and the choice of the design (D39, D40).** The folds are the five airports: a fold trains on the
train days of four airports, stops on their select days and reads the select days of the fifth (the held-out airport)
after its training. The goal is a new airport, so the folds are airports, not days. All runs use the chosen Δ = 4 s
(vocabulary D11).

| Step | Runs | Training runs |
|---|---|---|
| 1 | Variant `full`, four configurations, each on the 5 folds, seed 1337. A: the start. B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. B, C and D have the layers and the learning rate of A; every head is 32 wide | 20 |
| 2 | The seed scale: configuration A with the seed 2024, on the 5 folds | 5 |
| 3 | Variant `constants` with the configuration chosen after step 2, on the 5 folds, seed 1337 | 5 |
| 4 | The base: the chosen configuration and variant on all five airports, seed 1337, stopped on their select days; then its one validation readout | 1 |

The rules, fixed before the runs:

- **The score** of a run is the mean, over the 5 folds, of the per-step loss on the held-out airport's select days.
- **The seed scale** is the absolute difference between the scores of configuration A at the two seeds.
- **Configuration:** among the configurations whose score is within twice the seed scale of the best (the bound
  included), the one with the fewest parameters; a tie of parameters goes to the lower score (A and D have one shape)
  (D40).
- **Variant:** `constants` only if its score is lower than that of `full` (same configuration) by more than twice the
  seed scale; else `full`.
- **Readouts of each fold, not used for the choice:** the runway word of the first predicted step at the held-out
  airport (top-1: the model's most probable runway class there, teacher-forced up to that row); free generation at the
  held-out airport on its select days (200 flights × 2 sentences, drawn from every flight with the seed 1337): its
  outcomes (vocabulary §6, item 6), the words for each column, the go-arounds and the probability of "go-around" on the
  final (D72), the flights outside the selection apart (D75, D111).
- **The base's one validation readout (D85):** on the validation days of every airport, the per-step loss under
  `landed` and the share of the labelled words that the procedure masks block (`prior_validation`); free generation
  (200 flights × 2 sentences an airport, as a fold's; `prior_free_generation`). Each of the two claims its read (D118
  item 3, D119).

**The cost, measured.** At Δ = 4 s on `v12_20261005` (the log §1): configuration A 26 s, C 36 s an epoch on all five
airports; the largest batches reserve 1.73 GB (A) and 2.26 GB (C) of the GPU; a chunk of free generation (400 flights,
one of them to its two go-arounds, the cache 1,056 rows) 4.16 GB (A) and 5.55 GB (C). In B5, configuration A's first
folds took 11–14 min a training run (with its process's checks) and 3–6 min its free generation.

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
post-training's runners and its window loop import by the names given. "B12" marks a change that milestone B12 makes;
its report gives the names that change, and Claude writes them here.

| # | Item | What it gives | Code |
|---|---|---|---|
| 1 | The checkpoint | A trained prior in a format with its own name; its identity (§8); the set of procedure masks that it speaks under (§8 item 2); one function that opens a prior run: the checkpoint, the identity of its data and the check of its procedure masks (D106) | `prior/checkpoint.py` `load_checkpoint`, `CHECKPOINT_SCHEMA`; `prior/model.py` `Prior`; `prior/procedure.py` `PROCEDURE_MASKS`, `procedure_digests`; `prior/checkpoint.py` `open_prior`, `OpenedPrior` (the function that opens a prior run); the identity as a readout shows it and the one val read (D85, D118, D119): `prior/checkpoint.py` `readable_identity`, `validation_claim`, `holds_claim`; `prior/source.py` `require_selection_of` |
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said, the candidates and the landings before the step to the inputs of a row (§2). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states, its own landings (D105) and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings: from the tracks roster, less the sealed test days (the index refuses them, D105), counted before a time without a given flight's own landing (D63). The motion of rows: the 2 s displacement, with the velocities east and north (D106) | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, `motion` (the motion of rows); `prior/loop.py` `LoopRows` (`select`; one `LandingIndex` for each aircraft; one first predicted step for the batch, B12); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings` |
| 3 | The speaker | Says one row from the inputs of the row, column by column, with a cache from row to row; under the masks of §4, the masks of a caller included; draws each word with the caller's random numbers at a temperature (§4, D96, D121); refuses a row whose mark of the first predicted step is not the aircraft's own (no word in force), before it changes anything (B12); passes the caller's input of the added modules to the model; records the permitted words of each row and column; gives a copy of chosen aircraft (D96); refuses a model with any module in training mode (D107). A loop reads, and does not change, the words in force, the go-arounds said and the number of candidates (D106) | `prior/speaker.py` `Speaker` (`observe`, `speak`, `permitted`, `copy`; `heard`, `in_force`, `go_arounds`, `n_candidates`: read only), `Position`, `Permitted` (`select`, `join`: the join of several records), `draw`, `go_around_bound`, `MOST_GO_AROUNDS` |
| 4 | The teacher-forced loss | The loss of each step and each column for a batch of sentences (§5); the sentences of a split under a selection rule (D75) as batches; the log-probability of given words under the speaker's records of the permitted words, with the input of the added modules, with gradients (D96), refused for a model with any module in training mode (D107); the rows of a sentence that a loop said, with its words as targets, as one batch (D106) | `prior/train.py` `batch_nll`, `masked_log_probability`; `prior/source.py` `ArtefactSource`; `prior/selection.py` `require_rule`, `kept(rule, outcome, faulty)`, `left_out`, `side`, `SIDES`, `REASONS`, `CELL` (a record's cells `kept`, `left_out_fault`, `left_out_outcome`; D111); `prior/batch.py` `collate`, `RowTensors`; the rows of a sentence a loop said: `SpeakingLoop.sentences` (item 7) |
| 5 | A place in each layer | A module added at each layer whose output starts at zero leaves every output of the prior unchanged until it learns. Its input is what the caller gives the model and the speaker (item 3) | `prior/model.py` `Prior.add_at_each_layer` |
| 6 | The region of a final | For each candidate, whether a position is inside the FAF and the LPV cone: the region of the procedure masks (D64) and of the rows "on the final" (D72) | `prior/procedure.py` `airport_finals`, `Final.inside` |
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows that the start of the closed loop gives back (vocabulary §6 item 5: `start_moved`'s, a moved start's moved rows; never the stored sentence's, B12), then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it | `experiments/prior_speaking_loop.py` `SpeakingLoop` (the start's observed rows as an argument, B12; `copy(flights)` on the loop's own `Loop.copy`; `said(b)`, `states(b)`; `generated(flights)`, which refuses a flight the caller ended, D118), `Generated`; free generation's numbers `flight_numbers` |
| 8 | The Training export's procedure block | The block of a set that gives each candidate's region, glidepath lower edge, DA and entry height, as stage B's sets give it (D126) | `experiments/prior_training_export.py` `procedure_block` |

---

## 8 Gates and identities

The gate of this document is the prior: the teacher-forced likelihood against baselines and free generation. The user
sets its criteria (D7). The identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table (`candidates.json`), the row interval Δ, the sha256 of each split's closed-loop sentence file at Δ and of its stored signals (D118: they bind the faulty-track marks, D111), the selection of its sentences (D75: the rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason, D111; val's counts held and never shown, D120), and the landings that the candidate vectors count (D63): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number left out on the sealed test days. A checkpoint opens only for an equal identity, compared whole; a reader before the val read compares val's counts by the files' sha256, and the val readers count them again after their claim (D118). A run that reads the landings again computes the digest again and refuses a difference |
| 2 | The procedure masks of a prior | The set name (`procedure-masks-v5`), the checkpoint's sha256 and the sha256 of each candidate's CIFP procedure document (`procedure_masks.json` beside the checkpoint). The procedure data are the format of the masks |

---

## 9 Values

| Item | Value | Source |
|---|---|---|
| Motion inputs | Displacement in the 2 s before the row, at every Δ; at row 0 of an aircraft 0, with `no_motion` = 1 | D25, D60 |
| Rows before the first predicted step | 16 s: 8, 4, 2 rows at Δ = 2, 4, 8 s | Vocabulary §6, item 7 |
| Time since a word | log(1 + t / 2 s) / 5, t in seconds; 0 up to and with the first predicted step | D17 |
| Landings of a candidate | In [t − 30 min, t), the flight's own landing not counted | §2, D63 |
| Input scales | Distances asinh(d / 1 km); heights / 1 km; height above a glidepath asinh(h / 100 m); ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65 |
| RoPE base | 10,000 (heads 32 wide: periods 6.3 s to approximately 35,000 s) | D65 |
| Configuration A | d_model 192, 4 layers, 6 heads, feed-forward 768, dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ (linear warm-up over 500 steps, then constant), clip 1.0, 16,384 padded rows a batch, at most 30 epochs, stop after 3 | D40 |
| Configurations B, C, D | B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: A with dropout 0.2, weight decay 0.05. A head is 32 wide in each | D40 |
| Seeds | 1337 for every run; 2024 for step 2 (the seed scale) | §5 |
| Procedure masks | `procedure-masks-v5`. Glidepath lower edge: published glidepath − 60 m inside the FAF and the LPV cone; the DA wherever the aircraft is not inside them; no climb before the join, once more than ε of the level nearest the entry height below the entry height; never "unchanged" blocked | §4, D64 |
| Go-arounds of a flight in free generation | At most 2; after the second, "go-around" is masked | D68 |
| Temperature of the speaker | 1 in every run of stage B | D121 |
| Free generation of a fold and of the base | 200 flights × 2 sentences an airport, seed 1337; a flight's numbers from `default_rng([seed, sample, index])` | §5, D118 |
| Sentences of the base, the folds and the configurations | `landed`: only sentences whose own words land and whose flight has no faulty observed track; train, select and the validation's teacher-forced loss | D75, D111 |

---

## 10 Evidence

**The earlier prior uses its airport embedding (R46).** [Readout](../readouts/2026-10-03_airport_embedding.zh.md). The
prior of instruction-v3 (archived: `archive/two_tier_v3_2026_10/`) had an airport embedding. On the validation days, a
replacement of the embedding with the mean of the four other embeddings increased the teacher-forced loss by 0.024
(base) and 0.028 (augmented) for each step and decreased the landed share in free generation by 5.8 and 8.0 points;
approximately 60 % of the increase was in the heading column. The five embeddings were almost orthogonal: an identity
table, not a property of the airports. Its candidate table (absolute threshold positions and courses) was a second
identity (R46 §6). What transfers to this prior: a network that gets an airport identity uses it (D5, D24); the
measured sizes belong to that prior and are not a baseline of this one.

---

## 11 Key code index

At `44f6000f` (the code of B5). Paths relative to `4dTrajectory/ts_transformer/`.

| What | Where |
|---|---|
| The features of a row: own, candidate, the variant `constants`, the words in force | `prior/batch.py:42`, `:47`, `:51`, `:54` |
| A sentence's rows and targets (checked on construction); the heads' classes; a batch | `prior/batch.py:68` `SentenceRows`, `:150` `target_classes`, `:183` `collate` |
| The motion of rows (D25, D60) | `prior/inputs.py:59` `motion` |
| The inputs from the states (D13, D23, D24, D58, the scales of D41/D65) | `prior/inputs.py:85` `state_inputs` |
| The words in force and `since` (D17), through the grammar | `prior/inputs.py:120` `Heard` |
| A closed-loop sentence's rows: the inputs before the row's words | `prior/inputs.py:164` `sentence_rows` |
| The flight's own key in the roster | `prior/inputs.py:203` `own_flight_key` |
| The landings: the index, its digest, the count in [t − 30 min, t) without the own landing, the roster | `prior/landings.py:38` `LandingIndex`, `:64` `digest`, `:71` `counts_before`, `:90` `roster_landings` |
| A loop's row (D96 item 4, D105) | `prior/loop.py:31` `LoopRows` |
| The model: RoPE, a layer's causal time attention with its cache, the candidate pool, the first layer's input, a column's logits | `prior/model.py:96` `rope_angles`, `:155` `Layer.extend`, `:184` `CandidatePool`, `:198` `Prior`, `:246` `Prior._inputs`, `:266` `Prior.column_logits` |
| A place in each layer (§7 item 5); the refusal of training mode (D107) | `prior/model.py:307` `Prior.add_at_each_layer`, `:315` `require_eval` |
| The loss of each row and column; the per-step loss; the training loop and its stop (D31) | `prior/train.py:72` `step_nll`, `:92` `evaluate`, `:172` `train` |
| Configuration A's training values (D40) | `prior/train.py:34` `TrainConfig` |
| The log-probability under records (D96 item 3); the first-step runway | `prior/train.py:106` `masked_log_probability`, `:152` `first_step_runway` |
| The speaker: the bound of D68, the draw, the records, observe, speak, copy, the masks of a row | `prior/speaker.py:72` `go_around_bound`, `:82` `draw`, `:95` `Permitted`, `:152` `Speaker`, `:208` `observe`, `:224` `speak`, `:298` `copy`, `:347` `_allowed` |
| A final (region, edge, DA, entry height); the finals from the CIFP; the procedure masks (D64, D14, D122) | `prior/procedure.py:54` `Final`, `:84` `Final.inside`, `:95` `airport_finals`, `:119` `ProcedureMasks`, `:178` `permitted` |
| The selection (D75, D111) and its record | `prior/selection.py:51` `left_out`, `:80` `selection_record` |
| The identity of the data (§8 item 1); the val recount after the claim | `prior/source.py:45` `artefact_identity`, `:68` `require_selection_of` |
| The artefact as sentences of a selection | `prior/source.py:87` `ArtefactSource` |
| A run's data and a fold's held-out sentences (D39) | `prior/runs.py:51` `RunData`, `:78` `held_out_sentences` |
| The checkpoint; the claim of the val read; the identity a readout shows; opening a prior run | `prior/checkpoint.py:65` `load_checkpoint`, `:98` `claim_validation_read`, `:112` `readable_identity`, `:130` `open_prior` |
| The step of a speaker's closed loop (§7 item 7) and a flight's numbers | `experiments/prior_speaking_loop.py:76` `SpeakingLoop`, `:140` `SpeakingLoop.step`, `:46` `flight_numbers` |
| The runners: training, free generation and its readout, the validation readout, the choice, the campaign, the behaviour check, the export | `experiments/prior_train.py:69`, `experiments/prior_free_generation.py:162`, `:120` `readout`, `experiments/prior_validation.py:94` `masks_readout`, `experiments/prior_select.py:113` `choose_configuration`, `:124` `choose_variant`, `experiments/prior_campaign.py:119` `plan`, `:242` `run_campaign`, `experiments/prior_behaviour.py:171` `behaviour`, `experiments/prior_training_export.py:169` `fly_again` |
| The backend's live segment of a prior sentence | `aeroviz_backend/autopilot_segment/prior.py:82` `PriorSegments` |

---

## 12 Implementation plan: stage B

**Rules.** On the branch `dev-two-tier-v4-prior` (outline §5 rule 1). Stage B changes no code of `instructions/` or
`autopilot/` and no runner of stage A; it reads them only through the vocabulary's public interface (vocabulary §6).
Tests use synthetic artefacts; a check on real data reads the formal artefact on the train and select days only (D85).
The formal runs read `v12_20261005` / `v17_20261005` at Δ = 4 s. The rules of outline §5 apply.

| Milestone | What it built | Decisions | State |
|---|---|---|---|
| B0 | The package `prior/` and its import rules in `tests/test_architecture.py`; the runners `prior_train`, `prior_select`, `prior_free_generation` | — | Done |
| B1 | The data: the inputs of a row, the landings, the artefact as sentences and the identity of the data | D13, D17, D23–D25, D32, D41, D60, D63 | Done |
| B2 | The model and its checkpoint | D16, D41, D65 | Done |
| B3 | The training loop and the runner `prior_train` | D31, D40 | Done |
| B4 | The speaker, the procedure masks, free generation and its readout | D14, D33, D38, D52, D62, D64, D67–D72 | Done |
| B8 | The selection of the base's sentences | D74, D75 | Done |
| B9 | The interface for the post-training | D96 | Done |
| B10 | The corrections of Claude's check of stage B; the base's val set (outline D109) | D64, D72, D85, D105–D108 | Done |
| B11 | The selection leaves out flights with a faulty observed track | D111, D118 | Done |
| B5 | Cross-validation and the base | D39–D41, D75, D108 | Running |
| B12 | The corrections of Claude's second check of stage B | D108, D119–D122 | Ordered |
| B6 | The Training view of stage B | Outline §6, D109 | The view done; the publication after B5 |
| B7 | The close of stage B | — | After B5, B12, B6 |

The specifications of the milestones that are done are in `readouts/2026-10-05_stage_b_implementation_log.md` §3.

**B5. Cross-validation and the base** (D39, D40, D41, D75, D108).

- The 31 training runs of §5 as one campaign (`prior_campaign`) on a clean checkout, one at a time on the GPU: each
  fold's training and then its free generation at its held-out airport; after step 2 the choice of the configuration,
  after step 3 the choice of the variant (`prior_select`, the rules of §5, written as a choice); then the base and its
  one validation readout (`prior_validation`, then free generation on the val days). 65 steps in all.
- Before each step the behaviour check of D108; the commit of each step recorded as information. A step whose output
  lacks its last file (a crash, a kill) is moved aside and run again; a run is never repeated otherwise.
- Every run reads the sentences of the selection `landed` (D75, D111). For each fold: the held-out loss, the first-step
  runway at the held-out airport, the free generation at the held-out airport (§5).
- No criterion is applied: the user reads the results (D7).

**B12. The corrections of Claude's second check of stage B** (`readouts/2026-10-05_stage_b_check_2.zh.md`; D108,
D119–D122). On stage B's branch, on synthetic artefacts; merged into `dev-two-tier` only after B5 has ended: its
behaviour check gives another answer, so a campaign started before it stops by name under it (§0.4). The report gives
the changed names for §7 items 2, 3 and 7.

- D119: a claim of the val read whose output holds no written readout may be run again to the same output (both
  readers, `prior_validation` and `prior_free_generation`); free generation checks its options (`--airports`) before
  it claims. The export and the backend take a val readout only when it is written and claimed.
- D108's guard (D118 item 7): the campaign's settings (the seeds, the selection, the configurations, free generation,
  the temperature, D68's bound) come from the code on the disk — the process of the behaviour check gives them in its
  answer —, not from the campaign's own process. The behaviour check also covers `inputs.sentence_rows` on a fixed
  synthetic sentence (with a flight key of the real format), the selection (`kept`, `left_out`) over every rule ×
  outcome × mark, and `prior_select`'s rules on a fixed table of scores (a tie of parameters, a score at exactly twice
  the seed scale, a seed scale of 0). The campaign's format gets a new name.
- The speaker (§7 item 3): a row whose mark of the first predicted step differs, for an aircraft, from "no word in
  force" is refused; `observe` refuses positions that are not one for each row and each aircraft; both before any
  change.
- The step of a speaker's closed loop (§7 item 7): `SpeakingLoop` takes the observed rows before the first predicted
  step as the start gives them back (`start_moved`'s; `NO_MOVE` is `start` bit for bit) and refuses rows of another
  shape; of a sentence it reads only its first row and its first predicted step. Free generation gives it the rows of
  `start_moved` with `NO_MOVE`: its words, states and readout stay the same, bit for bit (a test). A row that the
  executor refuses (vocabulary D80) leaves the speaker, the loop's records and its row as they were.
- `LoopRows` takes one first predicted step for the batch.
- `prior_select` refuses a score that is not finite, by name; it checks every training value of a fold against its
  arm (the learning rate, the weight decay, the patience, the epochs, the warm-up, the clip, the padded rows of a batch,
  the RoPE base); the campaign's selection rule has one definition.
- D121, D122: no change of the code; a test holds D122 (a change of runway outside a go-around keeps the join and the
  passage of the candidate).
- `prior/inputs.py`'s docstring and `tests/test_prior_inputs.py` name what the motion must not read as §2 does.
- Tests: the D23 test and free generation's test of the fields it must not read, with a flight key of the real format
  (`<airport>:<id>_<runway>_<icao24>_<landing time>`), its runway and time changed with the own landing; free
  generation's test gives the changed sentences to the start too; `prior_train` prints train and select only (its
  output tested); the val readers refuse every read of val before the claim (the sentences, the stored outcomes, the
  faulty-track marks); a fold through the runner: a change of the held-out airport's data leaves the checkpoint the
  same; the choice at its edges; the campaign's settings from a second process; the speaker's two refusals; a moved
  start's rows reach the inputs of the first predicted step.

**B6. The Training view of stage B (outline §6).** The user sees what the prior says and how the executor flies it.

- **Export** (`prior_training_export`). For each flight of the sample: the observed rows before the first predicted
  step; the sentences that the prior says in free generation and their flown states (several sentences of one flight
  side by side); the closed-loop sentence of the same flight; the outcome and the DA check of each; at each row, the
  words that the procedure masks blocked; the region, the glidepath lower edge, the DA and the entry height of R. Every
  prior sentence is flown again and must equal its readout's states within the executor's bound. Sets: each fold of B5
  at its held-out airport (the flights of its free generation), and the base model (its one validation readout, only
  when claimed, outline D109). Its own schema names and its own index beside stage A's (outline §6 item 3).
- **Frontend.** The Training view of stage A with the prior's sentences: the five columns, a choice of sentence, the
  blocked words at a row, the procedure's limits drawn, the outcome. A click on a word flies its segment live with the
  executor of stage A; the base's val set opens and flies live, and no other set with a val flight opens (outline
  D109).
- **Publication and view** (after B5). The intent of each set in `docs/experiments/intents.json`; a test stack from the
  worktree; the browser check (outline §6 items 4–6).
- **Tests.** The export (a sample written and read again); the frontend's readers on fixtures that the export writes;
  a live segment equals the export's flown states.

**B7. Close of stage B.** The full ts suite passes (run detached). The implementation log and
`docs/reference/runners.md` are updated (the runners `prior_train`, `prior_select`, `prior_free_generation`,
`prior_validation`, `prior_campaign`, `prior_behaviour`, `prior_training_export`); the report gives the code index for
§11 (outline §5 rule 10). `dev-two-tier-v4` merges `dev-two-tier-v4-prior` (outline §5 rule 1). Report to the user: the
commits, the readings of each fold and of the base, the choice and its rule, and what stage C needs.
