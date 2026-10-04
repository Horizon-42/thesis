# Two-tier model: the post-training and the multi-aircraft work

**Summary.** The post-training trains the prior further in a closed loop: the prior speaks, the executor flies, the
judge decides, and the reward comes from the outcome. It runs in windows of recorded traffic, so this document also
gives the multi-aircraft work: scenes, the separation judge and the separation masks. It is stage C of the plan, in
outline. It reads the vocabulary and the prior only through their public interfaces (vocabulary §6, prior §7) and their
decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document is an outline: stage C sets the counts and the rules that are open here
(§0.2). The principles and the shared rules are in the outline (`outline.md`). Paths in backticks are relative to
`4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the repository root; `readouts/` is
`docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`, `two_tier_framework.zh.md`,
`two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3).

| # | Item | State | Source |
|---|---|---|---|
| D29 | Post-training starts from the base model in the multi-aircraft setting "one aircraft commanded". There is no single-aircraft post-training stage (§2) | Decided | User, 2026-10-03 |
| D30 | The post-training reward comes only from the outcome: 1 for `landed` without a go-around; 0.9ⁿ for `landed` after n go-arounds; 0 for every other outcome and for every loss of separation. No payment for a go-around without a landing. No mask on where "go-around" can be said (D10) (§2) | Decided | User, 2026-10-03 |
| D31 | Multi-aircraft inputs and judgements (§3): the landing context of every aircraft of a scene counts the landings of the closed loop; D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: O6). (The part of the prior: D31 there) | Decided | User, 2026-10-03 |
| D36 | The teacher-forced data term of the post-training uses single-aircraft samples of the closed-loop sentences, not scene samples. The flown states keep the observed path, not the observed time (vocabulary, D42), so two aircraft of one scene do not keep their observed spacing (§2, §3) | Decided | User, 2026-10-04 |
| D37 | Branch training. Each training aircraft is spoken one time. When its reward is less than 1, it is spoken again from saved states at its first predicted step and every 120 s after it, before the event that ended it. Each branch group compares only the words after its branch point (§2) | Decided | User, 2026-10-04 |
| D76 | The flights outside the base's selection (prior D75) come back in the post-training only as starts of the closed loops, rewarded by their outcome (D30). The teacher-forced data term (D36) uses the same selection `landed`, so a sentence whose own words do not land is never imitated | Decided | User, 2026-10-04 |

### 0.2 Open items

| # | Item | Proposal | §  |
|---|---|---|---|
| O6 | A mask for the spacing on the final (this design has no clearance word to hold back); the rule "established on the final" of the separation judge and the masks (D31) | Discuss with §3 | 3 |
| O9 | The most go-arounds of a flight in the loops of the post-training: the start of a closed loop takes this number and lays out the time that go-arounds add (vocabulary §6, item 5; D67). The prior's free generation uses 2 (prior D68); D30 already makes a chain of go-arounds cost reward | Decide with §2 | 2 |

### 0.3 Implementation

| Part | State |
|---|---|
| Stage C: post-training and multi-aircraft (§8) | Not started; outline |

### 0.4 Plan

1. Stage C starts from the base model of stage B (outline §4).
2. The parts of §8, then the post-training stage of §2, then the Training view of stage C (§8).

---

## 1 Scope

- **This document owns** the window loop, the reward, the branch training, the traffic attention, the edge features,
  the separation judge and the separation masks, and their runners.
- **It reads** the outline; the vocabulary's public interface (vocabulary §6): the grammar, the sentence artefact, the
  candidates, the executor, the judge and the row grid; and the prior's public interface (prior §7): the checkpoint,
  the inputs of a row, the speaker, the teacher-forced loss and the place for an added module.
- **It gives** the post-trained checkpoints and the window readouts, and their Training view (§8, outline §6).

---

## 2 Post-training

1. **One stage, from the base model, with one aircraft commanded (D29).** The closed loop is a window of 20 minutes of
   recorded traffic with one commanded aircraft. The model speaks for that aircraft; the other aircraft fly their
   records. The start model is the base model with a traffic attention whose
   output is zero (§3; prior §7, item 5). There is no single-aircraft stage. The reasons:
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
   the same reward also gives the go-around for traffic. The power n stops a chain of go-arounds that only adds time.

   **No payment for a go-around without a landing.** Such a payment f makes a go-around better than the continued
   approach when p < f / (f + 0.1), with the same p before and after the go-around (Claude's arithmetic). For example,
   f = 0.28 gives p < 0.74. "Go-around" is permitted at any row after the first predicted step (D10), and p is low on
   many flights when the training starts. The payment then teaches a go-around where the model is not sure, not where
   the approach is unstable. In this stage the other aircraft fly their records and do not react: a go-around cannot
   help them land. A stage with every aircraft of a window commanded (item 8) decides for itself if it pays for that
   help.
3. **Masks.** The masks of the prior: the grammar and the procedure masks, under the set that the prior was trained
   under (prior §7, item 3); and the separation masks of §3 (O6), given to the speaker as masks of a caller.
4. **Windows.** Real windows and augmented windows of the train days: B (a moved start: a turn about the airport, a
   height change and a speed change), A (one inserted aircraft that flies its record), D (the aircraft ahead moved). The
   counts are set at stage C.
5. **Loss**: the clipped-ratio surrogate (ε = 0.2) with the advantage inside each branch group (item 9); the pull to
   the base model (0.04, the KL on the sampled words, masked distribution); the teacher-forced data term (1) on
   single-aircraft samples of the closed-loop sentences of the train days in the base's selection `landed` (D36,
   D76); the flights outside it are starts like the others (D76); the traffic attention has its own
   learning rate. One pass over the sentences of a round.
6. **Go-around sampling first.** Before the training, measure the probability that the base model gives "go-around" on
   the final (with D26 the data has go-arounds with "no level-off" and "unspecified" in force). If the model says
   "go-around" by itself, the training uses no probes. If it does not, probes (a cross-entropy on a forced "go-around"
   at chosen steps) are discussed with the evidence of §6.3: a cross-entropy on a forced word teaches the word, not when
   to say it.
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
     the sentences carried a gradient (§6.2). Branch training spends the extra sentences only where the first
     sentence failed, and a continuation flies only the part after its branch point.
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
  runway, or a pair that counts as one runway) are both established on their finals (O6), and while the aircraft is more
  than 9,260 m (5 NM) from its threshold (7110.65BB 5-7-1 b.4). Both aircraft are predicted to the time when the
  aircraft ahead crosses its threshold, each along its course toward its speed target at the rate of the executor
  (vocabulary §6, item 5). A speed word whose predicted gap is then less than the required distance is masked. When
  every word falls short, nothing is masked.
- The traffic attention (initial output zero; prior §7, item 5) on the base model is the start of the post-training
  (D29, §2).
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
- **Established on the final (D31, O6).** The separation judge (the in-trail rule: which aircraft is responsible) and
  the separation masks must know if an aircraft is established on the final of its R. This comes from a function of one
  row: the state of the aircraft at that row and its R. It is the same for every aircraft (commanded, labelled,
  replayed). It does not come from the capture row, which uses later rows (vocabulary §6, item 3), and not from an
  executor state, because the executor has none (vocabulary §6, item 5). It is not an executor law: only the separation
  judge and the masks read it. Its rule is open with O6.
- **Go-around in traffic.** The word "go-around" is an explicit decision: it can be counted and rewarded, and the
  separation judge can treat the aircraft as no longer on the approach (open with O6).

---

## 4 Gates and identities

The gates of this document are the post-training and the multi-aircraft stages. The user sets their criteria (D7). The
identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The edge features of a scene | A conformance check: fixed reference scenes give the same edge features |
| 2 | A window readout | The code version and the conformance R44 |

---

## 5 Values

| Item | Value | Source |
|---|---|---|
| Reward of a landing after n go-arounds | 0.9ⁿ | D30 |
| Window | 20 minutes of recorded traffic, one commanded aircraft | D29 |
| Branch points | The first predicted step and every 120 s after it, before the event; K = 8 continuations | D37 |
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

---

## 7 Key code index

The code that this design replaces, archived by stage A unchanged.

| What | Where |
|---|---|
| Clearance mask (line number at the move) | `archive/two_tier_v3_2026_10/inference/separation_masks.py:131` `clearance_check` |
| Edge features (line number at the move) | `archive/two_tier_v3_2026_10/inference/scene_edges.py:51` `EDGE_FEATURES` |

---

## 8 Implementation plan: stage C (outline)

One post-training stage from the base model with one aircraft commanded (D29, §2): the window loop of the multi-aircraft
design with `--commanded one`, the traffic attention, the reward of D30, the masks, real and augmented windows. The
training is branch training (D37, §2 item 9): the saved state of a window at a branch point (executor, speaker cache,
judge, loop) and its restart, the branch groups and their advantage after the branch point, one pass, the executor in
inference mode while it speaks. The data term uses single-aircraft samples of the closed-loop sentences (D36). Before
it: a profile of one speaking batch (the prior, the executor, the masks, the edge features), and the go-around
probability of the base model on the final (§2 item 6). The multi-aircraft parts of §3 (D31): the
landing context from the loop for every aircraft (the archived `experiments/traffic_window.py` gives a replayed aircraft
its recorded context), the D23 test over a scene, the rule "established on the final" and
the replacement of the clearance mask (O6). The edge-feature conformance (§4). The multi-aircraft step-8.9 reward is
not built (D30). A later stage with every aircraft of a window commanded is optional (§2 item 8).

**The Training view of stage C (outline §6).** The last milestone of the stage. The window export (the archived
`experiments/window_training_export.py`, R36, rewritten): windows of recorded traffic with the commanded aircraft
flown on the words of the post-trained model and the other aircraft on their records; the separation judge's events;
the outcome of each window; the rounds of the post-training side by side. The frontend's Training view shows the
traffic window with the five columns of the commanded aircraft, and a click on a word flies its segment live with the
executor of A23. New schema names; its own index beside the old one; the intent of each set; a test stack and the
browser check (outline §6).
