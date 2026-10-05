# Two-tier model: the post-training and the multi-aircraft work

**Summary.** The post-training trains the prior further in a closed loop: the prior speaks, the executor flies, the
judge decides, and the reward comes from the outcome. It runs in windows of recorded traffic, so this document also
gives the multi-aircraft work: scenes, the separation judge and the separation masks. It is stage C of the plan. It
reads the vocabulary and the prior only through their public interfaces (vocabulary §6, prior §7) and their decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is complete for its part, except O12 (what the traffic attention reads of
another aircraft, §3). The principles and the shared rules are in the outline (`outline.md`). The code of stage C is on
the branch `dev-two-tier-v4-post` (§0.3; outline §5 rule 1); the code that it replaces is archived (§7). Paths in
backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the repository
root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`,
`two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D29 | Post-training starts from the base model in the multi-aircraft setting "one aircraft commanded". There is no single-aircraft post-training stage (§2) | Decided | User, 2026-10-03 |
| D30 | The post-training reward comes only from the outcome: 1 for `landed` without a go-around; 0.9ⁿ for `landed` after n go-arounds; 0 for every other outcome and for every loss of separation. No payment for a go-around without a landing. No mask on where "go-around" can be said (D10) (§2) | Decided | User, 2026-10-03 |
| D31 | Multi-aircraft inputs and judgements (§3): the landing context of every aircraft of a scene counts the landings of the closed loop; D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: D92). (The part of the prior: D31 there) | Decided | User, 2026-10-03 |
| D36 | The teacher-forced data term of the post-training uses single-aircraft samples of the closed-loop sentences, not scene samples. The flown states keep the observed path, not the observed time (vocabulary, D42), so two aircraft of one scene do not keep their observed spacing (§2, §3) | Decided | User, 2026-10-04 |
| D37 | Branch training. Each training aircraft is spoken one time. When its reward is less than 1, it is spoken again from saved states at its first predicted step and every 120 s after it, before the event that ended it. Each branch group compares only the words after its branch point (§2). (How the state at a branch point is made, and the test of item 4: D94) | Decided | User, 2026-10-04 |
| D76 | The flights outside the base's selection (prior D75) come back in the post-training only as starts of the closed loops, rewarded by their outcome (D30). The teacher-forced data term (D36) uses the same selection `landed`, so a sentence whose own words do not land is never imitated | Decided | User, 2026-10-04 |
| D91 | The most go-arounds of a flight in the loops of the post-training is 2, as in the prior's free generation (prior D68): the start of a closed loop gets 2, and after a flight's second go-around the loop forbids "go-around" (a mask of a caller, prior §7 item 3). Why: a window without other aircraft must say and fly what free generation says and flies (§2 item 1, a test of C4), and that needs the same bound; no closed-loop sentence of the artefact has more than one go-around (prior D68); D30 already makes a chain of go-arounds cost reward (O9 until then) | Decided | User, 2026-10-05 |
| D92 | "Established on the final" (D31) is a function of one row: G is false, the aircraft is inside the region of the final of R (inside the FAF and the LPV cone: the region of the prior's procedure masks, prior D64, prior §7 item 6), and its track is within 30° of the course of R (the judge's "lined up", 7110.65BB 5-9-2, TBL 5-9-1). After "go-around" the aircraft is not established until a runway word ends G. This stage has no mask for the spacing on the final: the reward teaches it (D30: 0 for a loss of separation), and the speed-word mask stays (§3). The rule is Claude's reading; C3 cites the regulation text that it reads ("established", the in-trail rule) to its paragraph (O6 until then) | Decided | User, 2026-10-05, on Claude's proposal |
| D93 | A window has no length of its own. It starts at the commanded aircraft's row 0 and ends where the commanded aircraft is done (the judge's outcome, its time limit included: vocabulary §6 items 5 and 6) or has a loss of separation that it answers for. At each step, its other aircraft are every other flight of the airport and the split in the air at that time, replayed along its record (§3). The 20 minutes of §2 item 1 are a typical length, not a bound. Stage C reads no time limit (vocabulary D90). Why: with two go-arounds (D91), a flight can fly for more than 20 minutes after its first predicted step; a cut at 20 minutes would leave it without traffic or end it early | Decided | User, 2026-10-05 |
| D94 | How branch training (D37) gets the state at a branch point, and its test. A round has two passes. First, every window is spoken one time, each with its own random numbers (from the seed, the round and the window). Then the windows whose reward is less than 1 are spoken again with the random numbers of their first sentence, and at each branch point the state of the window is copied K = 8 times: the loop of the executor (vocabulary §6 item 5, D97), the speaker (prior §7 item 3, D96) and the window's own state (its step, its separation judge, its landings). Each copy continues with its own random numbers (from the seed, the round, the window, the branch point and k). The second pass is checked against the first: the same words and the same flown states up to the last branch point; a window that differs is counted, reported and gives no sample in the round. The test of D37 item 4: a continuation with the random numbers of the first sentence says the same words and flies the same states as the first sentence, bit for bit; the probabilities can differ in their last bits when the batch differs (§6.4). Why: the copies spend the extra speaking only after the branch points (§2 item 9), and only the windows that need a branch are spoken again, so no state of a window that lands is saved | Decided | User, 2026-10-05, on Claude's report of stage C's readiness |

### 0.2 Open items

| # | Item | Proposal | §  |
|---|---|---|---|
| O12 | What the traffic attention reads of another aircraft at a step. (a) A token from its recorded state only: its edge features to the commanded aircraft and its own motion. The prior runs only on the commanded aircraft; the commanded aircraft's landing context is the prior's count of the landings before the step without its own landing (prior §7 item 2), and D31 holds with no more code. (b) The prior's hidden state of that aircraft, from the prior run over its recorded rows and words: every recorded aircraft then needs words said (a flight without a sentence has none) and a landing context from the loop (D31), and its hidden states change each round with the prior's weights | (a). The user decides (asked 2026-10-05) | 3 |

### 0.3 Implementation

The implementer's log (outline §5 rule 10). Branch `dev-two-tier-v4-post`, worktree `.claude/worktrees/two-tier-v4-post`.
A proposal is a reading where the design says nothing; it holds only until the user decides.

| Part | State |
|---|---|
| Stage C (§8) | Not started. The interfaces that it needs were requested on 2026-10-05: prior D96 (B9), vocabulary D97 |

### 0.4 Plan

1. Stage C is developed in parallel with the end of stage B (outline §4, D95), on its own branch (outline §5 rule 1), by
   its own implementer. C0–C12 (§8). A milestone starts when the parts that it reads are on its branch:

   | When | What |
   |---|---|
   | Now | C0. C1 and C2 on synthetic artefacts (`tests/support.py`). C3 |
   | After A34's artefact (vocabulary §0.4) | C1 and C2 also on a sample of A34's artefact, read-only; C1's census on all train days in a scratch directory (outline §5 rule 12, D55); the user chooses the counts of each kind of window |
   | After B9 of stage B (prior D96) is on `dev-two-tier-v4-prior` and merged into this branch; after the parts of vocabulary D97 that a milestone reads (the stored signals, the copy of a loop, the test of a flight in a batch) reach this branch through stage B | C4, C5, C6, C7 on synthetic artefacts and the smoke model of B3 |
   | After B5's base and Claude's check of stage B | C8. Then C10, after the user sets its criteria (D7) |
   | After the moved start of vocabulary D97 | C9 (window B) |
   | After C10 | C11, C12 |

2. The post-training of §2 (C10) starts from the base of stage B (D29) and runs only after C8's report.

---

## 1 Scope

- **This document owns** the window loop, the reward, the branch training, the traffic attention, the edge features,
  the separation judge and the separation masks, and their runners.
- **It reads** the outline; the vocabulary's public interface (vocabulary §6): the grammar, the sentence artefact and the
  stored signals of every flight of a split (item 3), the candidates, the executor and the start of a closed loop with a
  copy of a loop's flights and a moved start (item 5), the judge and the row grid; and the prior's public interface
  (prior §7): the checkpoint, the inputs of a row (one function, also in a loop) and the landings, the speaker (with the
  input of the added modules, the caller's random numbers, the records of the permitted words and a copy of chosen
  aircraft), the teacher-forced loss with the sentences under a selection and the log-probability of given words under
  a record, the place for an added module, and the region of a final. The parts of these interfaces that stage C
  needed were requested by D96 (prior) and D97 (vocabulary).
- **It gives** the post-trained checkpoints and the window readouts, and their Training view (§8, outline §6).

---

## 2 Post-training

1. **One stage, from the base model, with one aircraft commanded (D29).** The closed loop is a window of recorded
   traffic with one commanded aircraft, typically 20 minutes long (D93: it lasts as long as the commanded aircraft
   flies). The model speaks for that aircraft; the other aircraft fly their records. The start model is the base model
   with a traffic attention whose output is zero (§3; prior §7, item 5). There is no single-aircraft stage. The reasons:
   - The single-aircraft closed loop is a special case of this loop. With no other aircraft in its window, the
     commanded aircraft says and flies what single-aircraft free generation does, with the same loss and gradients.
     Approximately 45 % of the train flights have no leader in the air at their first predicted step (§6.1).
   - A single-aircraft reward does not see when an aircraft lands, so it does not limit the spread of the landing times.
     In traffic, that spread puts the aircraft into the sequence of the other aircraft (§6.1).
   - Without a capture law (D2, D3), the base model must learn more to land. It learns it one time, in the loop where
     it is used.
2. **Reward (D30).**

   | Outcome of the sentence (vocabulary §6, item 6) | Reward |
   |---|---|
   | `landed`, no go-around, on a runway of the airport's present landing direction | 1 |
   | `landed` after n go-arounds, on such a runway | 0.9ⁿ |
   | Every other outcome; every loss of separation (the separation judge, §3) | 0 |

   "The airport's present landing direction": a runway within 90° of a runway with a landing in the 30 min before the
   first predicted step. The reward gives "a reward for a go-around that the DA check needs" without a term of its own.
   An unstable approach that continues ends as `unstable_at_minimums`: 0. A go-around before the DA point continues the
   flight with 900 s more time (vocabulary §6, item 6); a landing then gives 0.9. Thus the model gains from a go-around
   only when the go-around changes a probable failure into a probable landing: with the same probability p of a landing
   before and after the go-around, 0.9·p < p. A go-around that prevents a loss of separation keeps the chance of 0.9, so
   the same reward also gives the go-around for traffic. The power n stops a chain of go-arounds that only adds time;
   after the second, "go-around" is masked (D91).

   **No payment for a go-around without a landing.** Such a payment f makes a go-around better than the continued
   approach when p < f / (f + 0.1), with the same p before and after the go-around (Claude's arithmetic). For example,
   f = 0.28 gives p < 0.74. "Go-around" is permitted at any row after the first predicted step (D10), and p is low on
   many flights when the training starts. The payment then teaches a go-around where the model is not sure, not where
   the approach is unstable. In this stage the other aircraft fly their records and do not react: a go-around cannot
   help them land. A stage with every aircraft of a window commanded (item 8) decides for itself if it pays for that
   help.
3. **Masks.** The masks of the prior: the grammar and the procedure masks, under the set that the prior was trained
   under (prior §7, item 3). The masks of a caller (prior §7, item 3): "go-around" after a flight's second go-around
   (D91), and the speed-word mask of §3.
4. **Windows.** Real windows and augmented windows of the train days: B (a moved start: a turn about the airport, a
   height change and a speed change; the start of vocabulary §6 item 5, D97), A (one inserted aircraft that flies its
   record), D (the aircraft ahead moved). The user chooses the count of each kind in a round from C1's census (D55).
5. **Loss**: the clipped-ratio surrogate (ε = 0.2) with the advantage inside each branch group (item 9); the pull to
   the base model (0.04, the KL on the sampled words, masked distribution); the teacher-forced data term (1) on
   single-aircraft samples of the closed-loop sentences of the train days in the base's selection `landed` (D36,
   D76); the flights outside it are starts like the others (D76); the traffic attention has its own
   learning rate. One pass over the sentences of a round. The masked distribution is the one that the speaker drew
   from: the training reads it from the speaker's records of the permitted words (prior §7, items 3 and 4).
6. **Go-around sampling first.** Before the training, measure the probability that the base model gives "go-around" on
   the final, on the select days, with the prior's free generation and its readout (prior D72) (with D26 the data has
   go-arounds with "no level-off" and "unspecified" in force). If the model says "go-around" by itself, the training
   uses no probes. If it does not, probes (a cross-entropy on a forced "go-around" at chosen steps) are discussed with
   the evidence of §6.3: a cross-entropy on a forced word teaches the word, not when to say it.
7. **Selection.** On the select days; the validation days are read one time for each stage. The airport
   generalization of the design is chosen in stage B (D39) and tested at the end on KAUS.
8. **Later, optional.** A stage with every aircraft of a window commanded starts from
   the model of item 7.
9. **Branch training (D37, D94).** The samples of a round:
   1. Each training window (one commanded aircraft) is spoken one time: the first sentence, with the window's own
      random numbers (D94).
   2. If the reward of the first sentence is 1, the window gives no sample: a group whose rewards are all the same
      gives no gradient.
   3. If the reward is less than 1, the event time t_E is the step where the first sentence ended: the first step of
      the loss of separation that ended it, the step of its judged outcome, or the time limit. The branch points are
      the first predicted step and every 120 s after it, before t_E.
   4. A second pass speaks these windows again with the random numbers of their first sentences. At each branch point,
      the state of the window is copied K = 8 times: the executor's loop, the speaker (its cache and its masks' state),
      the separation judge and the window's own state; the other aircraft fly their records, so their states come from
      the time. Each continuation has its own random numbers, from the seed, the round, the window, the branch point
      and k. A continuation with the random numbers of the first sentence says the same words and flies the same
      states, bit for bit (a test; D94). The branch points are at the same times after every window's first predicted
      step, so the continuations of one branch point of all windows are at the same row and speak as one batch.
   5. A branch group is the first sentence and the K continuations of one branch point; they differ only after the
      branch point. The advantage is the reward minus the mean of the group. It applies only to the words after the
      branch point; a group whose rewards are all the same gives no sample.
   6. The executor flies without gradients while it speaks (inference mode); the states are the same as with gradients
      (a test). The training scores the prior only.

   **Why.**
   - Cost: with K full sentences for each window, the speaking took approximately 55 % of a round, and only 37–47 % of
     the sentences carried a gradient (§6.2). Branch training spends the extra sentences only where the first
     sentence failed, and a continuation flies only the part after its branch point. The second pass speaks only the
     windows that failed, one time each (D94).
   - Credit: a reward for a whole sentence gives every word the same advantage. In a branch group, only the words
     after the branch point differ, so the advantage goes to them (§6.1, §6.2).
   - Branch points: the readout of multi-aircraft step 7.7 found that a new sentence from the start rescues the most
     events, then from 120 s before the event; 60 s and less rescue few (§6.2). Its reading, decided before the
     run, puts the branch points at the start and at fixed times, not some tens of seconds before the event.
   - A window whose first sentence lands gives no sample in this round. Its chance to fail comes again in a later
     round, as the windows are drawn again.

---

## 3 Multi-aircraft

- **Time.** A scene puts its aircraft on one UTC grid: a scene step is one instant for every aircraft. That is the
  layout of the data, not a model input, and D16 (prior) does not change it. The prior's time attention runs along each
  aircraft's own rows; the traffic attention reads one scene step, which is one instant, and uses no position. Edge
  features are relative (the approach clock is a distance; the closest-approach time is a time difference). The landing
  context counts the landings in the 30 min before the step's UTC time.
- R exists at every step for every aircraft from its first predicted step on. Before it, no input reads the aircraft's
  R (D23), the edge features included. Thus the relations of the edge features (the approach clock, the same runway,
  the parallel runways) and the separation judge have a runway at every step.
- **Recorded traffic (D93).** The other aircraft of a window come from the stored signals of the split (vocabulary §6
  item 3, D97): every flight of the airport and the split, with a sentence or without one, in the air at the step, from
  its entry into the arrival slice to its landing. A stated limit of the scene: an aircraft outside the slice (before
  its entry, a departure, an overflight) is not in the data, and a flight of a day of another split is never read (the
  test days are sealed, C32), so near the cut between two operating days a scene can lack aircraft. The motion of a
  recorded aircraft is its displacement in the 2 s before the row, as the prior's motion inputs (prior D25, D60) —
  never the signals' track, ground speed or vertical rate, which are fits that use later rows. Its R is the runway of
  its record, from its own first predicted step on (D23), and not before.
- **Separation judge** (`inference/separation.py`, reading VISUAL): the radar minima of 7110.65BB with the rules for
  visual approaches (7-4-4 c), never visual separation. Parallel runways at least 2,500 ft (762 m) apart are free once
  both aircraft are turned in (within 30° of the course, each on its own side of the midline); closer pairs count as one
  runway; established crossing finals are not judged. Its inputs on v4: each aircraft's position relative to its R
  (vocabulary §6 item 4) and "established" (D92). An event of a window is the first loss of separation for which the
  commanded aircraft answers.
- **Established on the final (D31, D92).** The separation judge (the in-trail rule: which aircraft is responsible) and
  the separation masks must know if an aircraft is established on the final of its R. This is a function of one row:
  the state of the aircraft at that row and its R. It is the same for every aircraft (commanded or recorded). It does
  not come from the capture row, which uses later rows (vocabulary §6, item 3), and not from an executor state, because
  the executor has none (vocabulary §6, item 5). It is not an executor law: only the separation judge and the masks read
  it. The rule (D92): G false; inside the region of the final of R (the FAF and the LPV cone; prior §7 item 6: the
  same region that the procedure masks and the readout of the go-around probability read, prior D64, D72); the track
  within 30° of the course of R.
- **Spacing on the final (D92).** This design has no word "cleared", so no mask can hold a clearance back. This stage
  has no mask that keeps an aircraft from joining the final too close behind another: the reward (0 for a loss of
  separation) teaches the spacing. A mask is designed only if the readouts show that the losses at the join dominate.
- **Speed-word mask.** It applies only when the aircraft and the aircraft next ahead on the approach clock (the same
  runway, or a pair that counts as one runway) are both established on their finals (D92), and while the aircraft is
  more than 9,260 m (5 NM) from its threshold (7110.65BB 5-7-1 b.4). Both aircraft are predicted to the time when the
  aircraft ahead crosses its threshold, each along its course toward its speed target at the rate of the executor
  (vocabulary §6, item 1). A speed word whose predicted gap is then less than the required distance is masked. When
  every word falls short, nothing is masked.
- **Traffic attention.** At each layer of the prior, a module (prior §7, item 5) reads the other aircraft of the step:
  an attention from the commanded aircraft's row to one token for each other aircraft. Its output starts at zero, and
  it gives zero when the step has no other aircraft, so that a window without traffic is free generation (§2 item 1).
  The speaker passes it the tokens of each row (prior §7, item 3). It has its own learning rate (§2 item 5). What a
  token holds: O12. The start of the post-training is the base with this module (D29, §2).
- **Edge features.** For each other aircraft at each step, what it is to the commanded aircraft: the archived set (§7),
  rewritten: its place in the commanded aircraft's frame (ahead, to the left) and its height difference; its position
  on the approach clock less the commanded aircraft's, and whether the two can be compared; the relation of the two
  runways in force (same, single, dependent, independent, unrelated); the rate at which the horizontal distance closes;
  the closest point of approach on the present motions (its time, horizontal distance and height difference); the
  distance that the rules require between the two; a flag for an unknown motion. Every scale is a constant in SI units
  (as prior D41). No feature uses a value of a row after the step; the motion is the 2 s displacement; the runway of an
  aircraft is used only from its own first predicted step on (D23). Their identity: §4 item 1.
- **Landing context (D31).** In the closed loop, the landing context of every aircraft of the scene, the recorded
  aircraft included, counts the landings that occur in the loop: the recorded landings of the recorded aircraft and of
  the aircraft before the window, and the landing of a commanded aircraft when it lands in the loop. The recorded
  landing of a commanded aircraft is never in it: that landing is the future of the commanded aircraft. Through the
  traffic attention, a commanded aircraft that flies later than its record would read a landing on its own landed
  runway. With O12 (a), only the commanded aircraft has inputs, and its landing context is the prior's count of the
  landings before the step without its own landing (prior §7, item 2): the recorded aircraft land at their recorded
  times, and the commanded aircraft's landing ends its window.
- **States in a scene (D32, D36).** A scene occurs only in the closed loop. The commanded aircraft flies with the
  executor; the other aircraft fly their records. There is no teacher-forced scene sample: the flown states keep the
  observed path but not the observed time, so the spacing between two aircraft would not be the observed one.
- **D23 in a scene (D31).** At the rows up to the first predicted step of an aircraft, no input of any aircraft and no
  edge feature uses a value computed from that aircraft's R. A test: a change of one aircraft's runway word leaves all
  inputs and edge features of the scene at those rows the same, bit for bit.
- **Go-around in traffic (D92).** The word "go-around" is an explicit decision: it can be counted and rewarded. After
  it, the aircraft is not established until a runway word ends G; the separation judge treats it as any aircraft that
  is not established.

---

## 4 Gates and identities

The gates of this document are the post-training and the multi-aircraft stages. The user sets their criteria (D7). The
identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The edge features of a scene | A conformance check: fixed reference scenes (train, seed 1337) give the same edge features. The references are written with the code (C2); every process that computes edge features runs the check first, every time (as vocabulary D73) |
| 2 | A window readout | The checks of the code that it uses, run at its start: the edge features' conformance, and the executor's and the closed loop's through the start of a closed loop (vocabulary D71, D73). The commit is recorded as information. No digest of code (D21) |
| 3 | A post-trained checkpoint | Its own format name; the identity of the base checkpoint (prior §8 item 1); the set of procedure masks that it speaks under (prior §8 item 2); the shape of the traffic module; the seed, and the windows of each round by their flights and their start times |

---

## 5 Values

| Item | Value | Source |
|---|---|---|
| Reward of a landing after n go-arounds | 0.9ⁿ | D30 |
| Most go-arounds of a flight in a loop | 2; after the second, "go-around" is masked | D91 |
| Window | One commanded aircraft, from its row 0 to its end; its other aircraft every flight of the airport and the split in the air at each step; typically 20 minutes | D29, D93 |
| Branch points | The first predicted step and every 120 s after it, before the event; K = 8 continuations, copied from a second pass with the first sentence's random numbers | D37, D94 |
| Established on the final | G false, inside the FAF and the LPV cone of R, track within 30° of the course of R | D92 |
| Loss | Clipped ratio ε = 0.2; pull to the base model 0.04; teacher-forced data term 1 | §2 item 5 |
| Present landing direction | A runway within 90° of a runway with a landing in the 30 min before the first predicted step | §2 item 2 |

---

## 6 Evidence

### 6.1 The post-training in traffic

From the multi-aircraft readouts of `instruction-v3` (`readouts/2026-09-28_m3_free_generation.zh.md`,
`readouts/2026-09-30_m4_passes.zh.md`; `multi_aircraft_design.zh.md` §6.6 step 9.4):

- The start model of the multi-aircraft work (augmented: landing and augmented starts, single-aircraft rewards) with one
  aircraft commanded, on the select days: it landed between 111 s earlier and 121 s later than its record (p10, p90);
  the labelled words, −36 s and +21 s. A loss of separation with the recorded traffic ended 11.5 % of its sentences;
  12.3 % when it did not see the traffic. Most losses were on vectored approaches (21.6 %; straight-in 4.0 %).
- The selected round (5 of 8) of the multi-aircraft training, validation days: 1.75 points fewer losses of separation
  than the start model (11.1 % → 9.3 %); the labelled words 4.4 %, the records 2.9 %.
- No readout has the base model in this loop. The numbers show that single-aircraft rewards do not limit the spread of
  the landing times; they do not show that those rewards make it.
- In the window loop with one commanded aircraft and no other aircraft, the words, the flown states, the loss and the
  gradients are the same as in single-aircraft free generation and training (tests of step 9.4). On the train days,
  approximately 45 % of the flights have no leader in the air at their first predicted step.

### 6.2 Cost and credit of the post-training

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

### 6.3 Go-around probes

- The 9.4 readouts: the start model gives the go-around word approximately 1e−11 at a probe. With a cross-entropy
  weight of 10–100 on the probes, the model says go-around, but 88 of its 108 own go-arounds end worse than round 0
  ([readout](../readouts/2026-10-03_9_4_weights.zh.md)).

### 6.4 One aircraft in batches of different size

- The prior (Claude's check, 2026-10-05, on `dev-two-tier-v4-prior` at `649a3d74`; a script in the session's
  scratchpad, not kept): a prior of the shape of configuration A (d_model 192, 4 layers, 6 heads, random weights),
  synthetic sentences of 60 rows (`tests/support.py` `prior_sentence`), encoded row by row on the CPU with one thread.
  The hidden state of one aircraft encoded alone and in batches of 2, 8 and 64 differs by at most 9.5e-7 to 1.2e-6:
  not the same bit for bit. Thus a speaker's probabilities depend in their last bits on the other aircraft of its
  batch. A word drawn with the same random number differs only when the number falls that close to a boundary between
  two words; the executor then flies the same states, because it reads only the words (D94).
- The executor: the closed-loop reading of the A16 smoke (2026-10-04, 395 train flights, Δ = 2 s) in chunks of 256
  flights with four threads and in chunks of 2,048 with one thread gave the same words, corrections, states and errors,
  bit for bit. Vocabulary D97 makes this a tested property of item 5.

---

## 7 Key code index

The code that this design replaces, archived by stage A unchanged, and the live code that it owns.

| What | Where |
|---|---|
| Separation judge (live; stage C ports its inputs to v4, C3) | `inference/separation.py`; the rules `inference/runway_schedule.py` |
| Clearance mask (line number at the move) | `archive/two_tier_v3_2026_10/inference/separation_masks.py:131` `clearance_check` |
| Edge features (line number at the move) | `archive/two_tier_v3_2026_10/inference/scene_edges.py:51` `EDGE_FEATURES` |
| Window loop, windows, the landings of a window (instruction-v3) | `archive/two_tier_v3_2026_10/experiments/traffic_window.py` `WindowLoop`, `draw_windows`, `window_landings` |
| One commanded aircraft's scene, its masks and edges (instruction-v3) | `archive/two_tier_v3_2026_10/experiments/traffic_speaking.py` |

---

## 8 Implementation plan: stage C

**Start.** In parallel with the end of stage B, on the branch `dev-two-tier-v4-post` (outline §5 rule 1, D95); §0.4
gives when each milestone starts. Until the interfaces of prior D96 and vocabulary D97 are on this branch, the code uses
synthetic artefacts and models (`tests/support.py`), and no milestone that reads those interfaces starts. The formal
runs need B5's base, Claude's check of stage B and the user's criteria (D7). Stage C changes no code of
`instructions/`, `autopilot/` or `prior/`; it reads them only through vocabulary §6 and prior §7. A defect in them goes
to their stage, through the user; a missing part of an interface is requested from its stage, as D96 and D97 were. The
rules of outline §5 apply.

**Written from this document, not patched from the archive.** The archived multi-aircraft code
(`archive/two_tier_v3_2026_10/`, §7) stays unchanged. A part of it comes back only where its logic fits this document,
rewritten into the new modules: the drawing of windows, the edge features, the step of a scene, the separation masks'
prediction to the threshold. Nothing else comes back: not the clearance mask, the scene prior of several commanded
aircraft, the rewards of steps 8.9 and 9.4, the formats of `instruction-v3`. No compatibility (principle 8).

**C0. Package and layout.**

- A new package `post/`: the scene and its steps, the edge features, "established", the separation masks, the traffic
  module, the reward, the branch groups and the loss. It reads `instructions/` and the names of prior §7; it does not
  import `autopilot/`. The window loop, which joins the speaker, the start of a closed loop and the scene, is a module
  shared by the runners of stage C (under `experiments/`, not a runner), as a runner joins a model to the executor.
  The separation judge stays in `inference/separation.py`.
- The architecture test (`tests/test_architecture.py`, read by the names imported, as prior D69's test): `post/` imports
  from `prior/` only the names of prior §7; the runners of stage C import from `autopilot/` only the names of
  vocabulary §6 and from `prior/` only those of prior §7.
- The runners: `post_windows` (the windows and their census, C1), `post_train` (the rounds, C10), `post_readout` (a
  window readout), `post_training_export` (C11). New code; the archived runners stay as they are.

**C1. Windows and scenes** (§3; D29, D93, C32).

- A window: one commanded flight with a closed-loop sentence at the chosen Δ (every such flight, inside the base's
  selection or not, D76), and its scene: the stored signals of every other flight of the airport and the split, on the
  scene's steps, from the commanded aircraft's row 0 to its end (D93).
- The augmented windows A (one recorded flight of the same airport and split, from another time, inserted with its
  record shifted in time) and D (the aircraft next ahead on the approach clock at the first predicted step, its record
  shifted in time). The ranges of the shifts are the implementer's proposals for the user.
- The census (`post_windows`, outline §5 rule 12): for each airport and split (train, select), the windows, the other
  aircraft at the first predicted step, the share with a leader in the air on the same runway or a runway that counts
  as one, and the windows near the cut between two operating days. Measured on all train days in a scratch directory
  after C1; the user chooses the count of each kind of window in a round.
- Tests: a window never reads a test day (C32) or a flight of another split; its other aircraft at a step are exactly
  the flights in the air then; a recorded aircraft's motion comes from its 2 s displacement (a change of its stored
  track, ground speed or vertical rate changes nothing); its R is absent before its first predicted step; the steps are
  on UTC multiples of Δ (vocabulary §6 item 7).

**C2. Edge features** (§3, §4 item 1; D23, D31).

- The features of §3 for every other aircraft at each step, at fixed SI scales; the conformance references (fixed
  scenes of the train days, seed 1337) written with the code; the check run by every process that computes them.
- Tests: no feature uses a later row (a change of any row after the step changes nothing); D23 in a scene (a change of
  one aircraft's runway word changes no input and no edge feature up to its first predicted step, bit for bit); a
  permutation of the other aircraft permutes their features; an aircraft whose motion is unknown gives the flag and
  zeros.

**C3. Separation judge, "established" and the speed-word mask** (§3; D92).

- `inference/separation.py` on v4: its inputs from vocabulary §6 item 4 (the position relative to a candidate) and
  "established" of D92 (the region of prior §7 item 6, the 30° of the judge); the reading VISUAL; the event of a
  window: the first loss of separation for which the commanded aircraft answers.
- The speed-word mask of §3, given to the speaker as a mask of a caller; its prediction to the threshold at the
  executor's rate (vocabulary §6 item 1).
- The regulation text that D92 reads ("established", the in-trail rule) cited to its paragraph, in
  `docs/literature/arrival_separation/`.
- Tests: "established" is a function of one row (a change of later rows changes nothing), the same for a commanded and
  a recorded aircraft, false while G is true; the speed-word mask masks nothing outside its conditions and nothing when
  every word falls short.

**C4. The window loop** (§2 items 1–3; D29–D31, D91, D93; prior D96; vocabulary D97).

- The commanded aircraft through the start of a closed loop (vocabulary §6 item 5; the most go-arounds 2, D91); the
  inputs of its rows through the prior's one function of a loop's row (prior §7 item 2); the speaker with the random
  numbers of D94, the masks of a caller (D91, the speed-word mask) and the traffic module's input; the scene step by
  step; the separation judge at each step; the end of the window (D93); the reward of D30, with the present landing
  direction from the landings of prior §7 item 2.
- Tests: a window with no other aircraft says and flies what the prior's free generation says and flies for the same
  flight with the same random numbers and the same bound: the same words and states, bit for bit (D29; §2 item 1); the
  reward table of §2 item 2; the commanded aircraft's landing context never counts its own landing (D31); the loop
  reads no time limit (vocabulary D90).

**C5. The traffic attention** (§3; O12 decides what a token holds).

- The module of §3 at each layer through prior §7 item 5, its input passed by the speaker (prior D96); its own learning
  rate.
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
  one pass over the samples of a round.
- Tests: at the parameters that spoke the words, the ratio is 1 within the float tolerance of §6.4; a word that a record
  blocks takes no probability; when the module gives zero, the data term equals the prior's teacher-forced loss on the
  same batch.

**C8. Profile and the go-around probability** (§2 item 6). After B5's base.

- One speaking batch at the formal size: the time of the prior, the executor, the masks and the edge features; the
  memory of the host and of the GPU at the formal size (the user's rule of 2026-09-30: a smoke at a small size proves
  nothing about the formal size; outline §5 rule 13); the time of a round, the second pass of D94 included.
- The base's probability of "go-around" on the final, on the select days, through the prior's free generation and its
  readout (prior D72). No probes unless the user decides them (§2 item 6).
- The report to the user; no criterion is applied (D7).

**C9. Window B** (§2 item 4). After the moved start of vocabulary D97.

- Moved starts through the start of a closed loop; the ranges of the turn, the height change and the speed change are
  the implementer's proposals for the user; windows B in the census of C1.
- Tests: the prior reads the moved observed rows that the start gives back; D23 holds; a move of zero gives the window
  without the move, bit for bit.

**C10. The post-training** (§2). After C8 and the user's criteria (D7).

- The rounds as one campaign from one commit on a clean checkout: in each round the windows drawn (C1's counts), the
  two passes of D94, the training of C7, the selection readout on the select days; resumable. The validation days are
  read one time, for the chosen round. The intents in `docs/experiments/intents.json` before the launch.

**C11. The Training view of stage C (outline §6).** The last milestone of the stage. The window export (the archived
`experiments/window_training_export.py`, R36, rewritten): windows of recorded traffic with the commanded aircraft
flown on the words of the post-trained model and the other aircraft on their records; the separation judge's events;
the outcome of each window; the rounds of the post-training side by side. The frontend's Training view shows the
traffic window with the five columns of the commanded aircraft, and a click on a word flies its segment live with the
executor of A23. New schema names; its own index beside the old one; the intent of each set; a test stack and the
browser check (outline §6).

**C12. Close of stage C.** The full ts suite passes. §0.3 (the log) and `docs/reference/runners.md` are updated; the
report gives the new code index for §7 (outline §5 rule 10). Report to the user: the commits, the rounds and their
readings. The user merges (outline §5 rule 11). A later stage with every aircraft of a window commanded is optional
(§2 item 8).
