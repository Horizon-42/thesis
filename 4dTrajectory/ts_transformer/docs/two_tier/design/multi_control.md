# Two-tier model: the multi-aircraft control

**Summary.** The multi-aircraft control trains the prior further in windows of recorded traffic in which every arrival
of a time span is commanded by the model: the model speaks for several aircraft at one time, each aircraft flies its
own words, and the aircraft must keep their separation from each other and from the recorded traffic. It is stage D of
the plan, the optional stage of post-training §2 item 8. It starts from the round that stage C chose. Its first
subject is the reward: what one window's sentences earn, and how the gain or the loss of a window goes to the words
of the aircraft that caused it. Its code is the post-training's code, generalised where one commanded aircraft becomes
several; it adds only what is its own.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document gives the design of stage D: its decisions, its rules, its values, the
evidence for them, its code and its milestones. The principles and the shared rules are in the outline (`outline.md`).
Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name the
repository root; `readouts/` is `docs/two_tier/readouts/`. The design of instruction-v3's multi-aircraft work, which
this stage reads as evidence only (§9), is `archive/two_tier_v3_2026_10/docs/multi_aircraft_design.zh.md` ("the
archived design").

**State of this document.** Written 2026-10-06 on the user's request, before any of it is built. Every decision in
§0.1 is a proposal of Claude until the user decides it (state "Proposed"); the user decided D141, D142 and D150
on 2026-10-06. Nothing of stage D starts before stage C's
campaign (C10) ends and its round is chosen.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3). A decision gives the rule and, after "Why", its reason;
the evidence is in §9.

| # | Item | State | Source |
|---|---|---|---|
| D139 | Stage D commands every arrival of a window's time span (§3). It starts from the round of stage C that the user chooses (post-training §2 item 7) and pulls to the base, as every stage of the post-training does. The model keeps stage C's shape: the prior runs on each commanded aircraft, and the traffic attention reads every other aircraft as a token from its state only (post-training D98) — a recorded aircraft's state from its record, a commanded one's from the states that the loop flew. No attention between the commanded aircraft's hidden states, no new head. Why: stage C's module already reads the other aircraft of a step; a commanded aircraft is, to another one, an aircraft with a state, as a recorded one is. The archived design encoded every aircraft of a scene together and needed a tensor of steps × aircraft² × width in each layer (5.3 GB a layer for its largest window, §9.1) | Proposed | Claude, 2026-10-06 |
| D140 | **The reward of an aircraft** r_i is post-training D30, unchanged, for each commanded aircraft: 1 for `landed` with no go-around, 0.9ⁿ after n go-arounds, on a runway of the airport's present landing direction (D147); 0 for every other outcome and when the aircraft answers for a loss of separation (the judge's responsibility, §3). Why: D30's reasons hold for each aircraft (§2.1) | Proposed | Claude, 2026-10-06 |
| D141 | **The reward of a window** W is the sum of r_i over the window's commanded aircraft. A recorded aircraft adds nothing. The sum is not divided by the number of aircraft. Why: one controller speaks for every aircraft, and its task is that all of them land safely; a landing is worth the same in a window of one aircraft and in a window of six (§2.2) | Decided | User, 2026-10-06, on Claude's proposal |
| D142 | **Credit by one varied aircraft.** A branch group (post-training D37, D94) varies one commanded aircraft v: from the branch point on, the K continuations draw new random numbers for v only; every other commanded aircraft draws the numbers of the first sentence (each keeps its own stream). The advantage of each sentence of the group is its W less the mean of the group's W, and it applies only to v's rows after the branch point. The other aircraft's rows of the group carry no advantage. Why: inside a group, only v's draws differ, so a change of W is caused by v's words — directly or through the other aircraft's answers to v. A window's reward can then be the team's sum without the noise of the other aircraft's own draws (§2.3) | Decided | User, 2026-10-06, on Claude's proposal |
| D143 | **Which aircraft are varied, and where.** A window whose W is less than its count of commanded aircraft is spoken again (post-training D94's second pass). The varied aircraft are each commanded aircraft with r_i < 1 and each commanded aircraft of a loss of separation that one of them answers for. The branch points of a varied aircraft v are v's own first predicted step and the points of the window's grid after it (the anchor's first predicted step and every 120 s after it), before v's event: the step of v's own end, or of the loss it is in, whichever is earlier. K = 8 (D94). Why the window's grid: the continuations of one point of all windows of a batch are at one row of the loop and speak as one batch, as in stage C | Proposed | Claude, 2026-10-06 |
| D144 | **An aircraft that answers for a loss of separation becomes silent.** Its reward is 0 from that step. It is not removed: it flies on its words in force until the executor is done with it, the other aircraft must keep their separation from it, and it never answers for a loss again. It is flown through the loop with a caller's mask that permits only "unchanged" in every column (prior §7 item 3), so its rows are said with probability 1 and carry no gradient. **A window ends when none of its commanded aircraft is spoken** (each is done, or silent). Why: an aircraft that disappears gives the others free space, so a loss would pay the others; a window of one commanded aircraft then ends at its loss, as post-training D93 says (§2.4) | Proposed | Claude, 2026-10-06; the aircraft that stays in the scene: the user's rule of 2026-09-30 for the archived design (§9.1), proposed again |
| D145 | **A loss with a recorded aircraft.** A commanded aircraft answers for a loss with a recorded aircraft only where the rules make it responsible, as in stage C. A loss that only the recorded aircraft answers for is counted in the readouts and earns nothing and costs nothing. Why: a recorded aircraft does not react; its record also loses separation by itself (2.76 % of the flights of the archived census, §9.1). The alternative is O17 | Proposed | Claude, 2026-10-06 |
| D146 | **The window.** A window is an anchor flight and every flight of the same airport and split with a closed-loop sentence at the chosen Δ whose row 0 is in [the anchor's row 0, the anchor's row 0 + L): these are its commanded aircraft. Every other flight of the airport and the split in the air at a step is recorded and flies its record (post-training D93). L is a setting that the user chooses after the census (MC1, O16). With L = 0 the window is stage C's real window. Kinds: real windows and compressed windows (each commanded aircraft other than the anchor moved toward the anchor by a whole number of Δ: its offset from the anchor times c, c drawn uniformly in [c_min, 1]); their counts are settings (O16). A window is left out of the draw when a commanded aircraft opens inside a loss of separation on the records at its first predicted step (post-training D113, for each commanded aircraft) | Proposed | Claude, 2026-10-06 |
| D147 | **Landings in the loop.** A commanded aircraft's recorded landing is never counted by any aircraft of its window: it is the future of that aircraft. Its landing in the loop is counted from the time of its crossing by every other aircraft of the window, in their inputs (prior §7 item 2) and in the present landing direction of the reward (post-training §2 item 2). The landings of the recorded aircraft follow post-training D105. Why: the inputs give only what a controller knows before the step (principle 7); with one commanded aircraft this is post-training D31 and D105 | Proposed | Claude, 2026-10-06 |
| D148 | **One row at a time for all aircraft.** At a Δ row, every commanded aircraft is spoken at once, from the state at the start of the row; no aircraft reads the words that another one says in the same row. The masks of a caller are computed once for the row from that state (post-training D110). The speed-word mask takes a commanded leader as it takes a recorded one: its speed target is its present speed along its course (post-training D117 P5). Why: the archived design ordered the aircraft of a step along the approach clock so that a later aircraft's mask read the earlier one's new words (§9.1); that needed an order, several passes of the speaker in each step and a rule for ties, for a mask that acted on 0.18 % of the labelled speed words | Proposed | Claude, 2026-10-06 |
| D149 | **Stage D's code depends on stage C's code, through a public interface.** The post-training document gets a public interface (a new §9, items in §6.1 here); stage D imports from `post/` and from stage C's runners only its names. Where one commanded aircraft becomes several (the window, the window loop, the loss of separation, the two passes of a round, the campaign's steps), stage C's code is generalised, not copied. A generalisation keeps stage C's behaviour: with one commanded aircraft in each window, the same words, states, rewards and branch groups, bit for bit, on fixed inputs (the CPU, one thread), checked by a test before stage D uses it (as outline D138: the old form is the reference). Stage C's formal results stay valid | Proposed | Claude, 2026-10-06 |
| D150 | **The aircraft of one closed loop join it at their own Δ rows, in one batch** (§6.3). A loop has a row clock (its ticks, one Δ each); each aircraft has a join tick j: before it the aircraft is absent (its rows not present, not spoken, not flown, its executor waiting), from it the aircraft's own row 0, then its observed rows to its own first predicted step, then its said rows. Built in stages A's and B's code (the start and `Loop`; `LoopRows`, the speaker, `SpeakingLoop`), each written first in its stage's public interface (vocabulary §6 item 5, prior §7 items 2, 3, 7). The prior's network does not change: its time attention already reads only present rows (`Past.present`) and time differences (RoPE on each aircraft's own seconds). With every join tick 0 the loop is today's loop, bit for bit (a test); free generation and stage C call it so. Also: a landing added to chosen aircraft's landings while the loop runs (D147). Why: the commanded aircraft of a window enter at different times; one loop for each entry row would speak a row in up to L / Δ small batches (150 at L = 10 min), each with its fixed cost in Python, and the speaking is bound by the CPU (outline D138) | Decided | User, 2026-10-06 (one batch with each aircraft's own first row, Claude's option 1); the design of §6.3: Claude |
| D151 | **What the reward does not have.** No term for the time an aircraft takes (the judge's time limit bounds it; vocabulary §6 item 6); no shaping by the margin of separation; no payment for a go-around of its own; no probes (forced words). Why: each of them was built or measured in the archived design or in stage C and gave the model a target that is not the outcome (§2.5, §9) | Proposed | Claude, 2026-10-06 |

### 0.2 Open items

| # | Item | Proposal | § |
|---|---|---|---|
| O16 | The span L of a window, the kinds of window and their counts, c_min | Measured by the census (MC1) on the train days for L of 0, 5, 10 and 20 min; the user chooses before MC2's smoke (outline D55) | 3.1 |
| O17 | A loss that a recorded aircraft answers for, with a commanded aircraft as its other aircraft (a commanded aircraft that slows in front of a recorded follower) | D145: readout only. Alternative: 0 for the commanded aircraft of such a loss when the two records kept their separation at that step. Decided with the counts of MC1 and of round 0 | 2.4 |
| O18 | The time that the aircraft take | D151: readout only (the delay of each landing against its record, the spacing at the threshold). A term only if the readouts show that the model spreads the aircraft out; the user decides with the numbers | 2.5 |

### 0.3 Implementation

Not started. Development waits for the user's decisions of §0.1; the formal runs wait for C10's chosen round.

| Milestone | State |
|---|---|
| MC0–MC8 | Not started |

### 0.4 Plan

1. The user decides §0.1 and O16–O18 (by `AskUserQuestion`); Claude writes the decided text and the interfaces of
   §6 into the vocabulary, prior and post-training documents.
2. MC0 (the interfaces) and MC1 (the census) while C10 runs, on `dev-two-tier-v4` (outline §5 rule 1), with no job
   that slows C10 (outline §5 rule 13). The user chooses O16.
3. MC2–MC5. The formal campaign MC6 after C10's chosen round and the user's settings; the Training view MC7; MC8.

The milestones are named MC (multi-aircraft control), not D: D numbers are decisions.

---

## 1 Scope

- **This document owns** the windows with several commanded aircraft, the window's reward and its credit, the varied
  aircraft and their branch points, the landings of a loop, their runners (`experiments/multi_*.py`), the package
  `multi/`, and the Training view of stage D.
- **It reads** the outline; the vocabulary's public interface (vocabulary §6: the grammar, the sentence artefact and
  the stored signals, the candidates, the executor and the start of a closed loop, the judge, the row grid); the
  prior's public interface (prior §7: the checkpoint, the inputs of a row, the speaker and its masks of a caller, the
  teacher-forced loss and the log-probability under records, the step of a speaker's closed loop); and the
  post-training's public interface (§6.1, D149).
- **It gives** the checkpoints of the multi-aircraft control, its window readouts and its Training view.

---

## 2 Reward

The reward is the first subject of this stage. One window holds several commanded aircraft, and one controller (the
same prior) speaks for all of them. Three questions follow: what one aircraft earns (§2.1), what a window earns (§2.2),
and to which words a window's gain or loss goes (§2.3). §2.4 gives the losses of separation and the aircraft that
cause them; §2.5 what the reward leaves out; §2.6 the incentives that result, case by case.

### 2.1 One aircraft: D30, unchanged (D140)

| Outcome of the aircraft | r_i |
|---|---|
| `landed`, no go-around, on a runway of the present landing direction | 1 |
| `landed` after n go-arounds, on such a runway | 0.9ⁿ |
| every other outcome of the judge; a loss of separation that the aircraft answers for | 0 |

The present landing direction (post-training §2 item 2, D112) is read for each aircraft at its own first predicted
step, from the landings of its window (D147). The bound of two go-arounds (post-training D91) holds for each aircraft.

D30's reasons hold for each aircraft: a go-around gains only when it changes a probable failure into a probable
landing (0.9·p < p); a chain of go-arounds costs; an unstable approach that continues ends as `unstable_at_minimums`.
Nothing of D30 is changed, so that a window of one commanded aircraft earns what stage C's window earns.

### 2.2 One window: the sum (D141)

W = Σ r_i over the commanded aircraft of the window.

- **Why a sum.** The model is one controller. Its task in a window is that every aircraft lands safely; an aircraft
  that lands because another one made room for it is the controller's success, not only that aircraft's. With r_i
  alone, an aircraft gains nothing when it helps another one: the leader that keeps its speed up for its follower, the
  aircraft that extends its downwind so that another one can join first, the go-around that saves the aircraft behind
  it.
- **Why not a mean.** A landing is worth the same in a window of one aircraft and in a window of six. A mean would
  make each aircraft of a busy window worth less, and the busy windows are where the multi-aircraft control is learnt.
  The advantage (§2.3) is a difference inside a group of one window, so W's range (0 to the count of aircraft) does
  not bias one window against another; the scale of a group's advantage is that of the outcomes that changed in it,
  mostly one or two aircraft.
- **The recorded aircraft add nothing.** They fly their records whatever is said; a term for them would pay or charge
  the model for the record.
- **With one commanded aircraft**, W = r of that aircraft: stage C's reward.

### 2.3 Credit: one varied aircraft (D142, D143)

**The problem.** A sum over several aircraft whose words are all drawn at random mixes the effects of every aircraft's
draws. The archived design gave each aircraft its own reward and compared each aircraft's K whole-window sentences,
in which every other aircraft was also spoken again with new draws (§9.1). Its rewards of the aircraft of one window
were nearly not correlated (0.05), and its trained model (`window r5`) had no gain that held on the validation days.
A team's sum compared over such samples would be noisier still: a loss of one aircraft in one sample moves every
aircraft's advantage.

**The rule.** A branch group varies one aircraft. Stage C's branch training (post-training D37, D94) already copies a
window's state at a branch point and speaks K continuations; here, in a continuation of the varied aircraft v:

1. v draws its words from new random numbers (`continuation_numbers` of the seed, the round, the window, v, the
   branch point and k);
2. every other commanded aircraft draws from the numbers of its first sentence, continued from where its stream was
   at the branch point (its own stream copied with the window's state);
3. every aircraft is spoken by the model as in the first sentence: an aircraft that is not varied answers v's new words
   through the traffic attention and its masks, with its own old numbers;
4. the group's sentences are the first sentence and the K continuations; the advantage of each is its W less the
   mean of the group's W; it applies to v's rows after the branch point, and to no other aircraft's rows.

**Why this gives v's credit (Claude's reasoning).** The draws of the aircraft are independent. Inside a group, the
numbers of every aircraft but v are fixed, so a difference of W between two sentences of the group is caused by v's
draws: by v's own outcome, and by the other aircraft's outcomes as they answer v. The expectation of the gradient over
the other aircraft's numbers is the gradient of the expected W, so fixing them in a group removes their noise and
adds no bias. An aircraft's changed words in a continuation are its answers to v; they get their own credit when that
aircraft is varied. The idea is close to the counterfactual baselines of multi-agent learning (for example, COMA,
Foerster et al., 2018) and to the advantage from several continuations of one state (VinePPO, Kazemnejad et al.,
2024); the texts were not read for this design.

A residual noise: an aircraft that is not varied draws with the same number from a distribution that changed a
little (its inputs changed), so a draw near a boundary between two words can change (as post-training §6.4). That is
part of its answer to v, not an independent draw.

**Which aircraft and where (D143).** As in stage C, the extra speaking is spent only where the first sentence failed:

- a window whose W equals its count of commanded aircraft (every one landed with no go-around) gives no sample;
- else the varied aircraft are each aircraft with r_i < 1 and each commanded aircraft of a loss that one of them
  answers for (the other aircraft of that loss, even with r = 1: it is often the aircraft that could have made room);
- the branch points of v are v's first predicted step and the points of the window's grid after it (the anchor's
  first predicted step and every 120 s after it), before v's event: the step of v's own end or of the loss that v is
  in, whichever is earlier (post-training D37's points, on the window's clock);
- a group whose W is the same in all its sentences gives no sample.

Most branch points are on the window's grid, so the continuations of one point of all windows of a batch are at one
row of the loop and speak as one batch, as in stage C.

**With one commanded aircraft** the varied aircraft is that aircraft, its branch points are stage C's, and the groups
are stage C's (D149's test).

### 2.4 Losses of separation and their aircraft (D144, D145)

The separation judge is stage C's (`inference/separation.py`, reading `VISUAL`, post-training §3): after each row
flown, it judges every pair of the scene. For each loss, the rules name the responsible aircraft: the aircraft behind
in trail; the joining aircraft where one is established; both where neither is.

| Pair | What it gives |
|---|---|
| Two commanded aircraft | Each responsible one that is still spoken answers: r = 0, and it becomes silent (D144). The other one flies on |
| A commanded and a recorded aircraft, the commanded one responsible | As above |
| A commanded and a recorded aircraft, only the recorded one responsible | Counted in the readouts; nothing for W (D145, O17) |
| A silent aircraft and a spoken one | The spoken one answers when it is responsible; the silent one never answers again |
| Two recorded aircraft | Not judged (stage C) |

**Silent, not removed (D144).** An aircraft that answers for a loss keeps flying on its words in force, through the
loop, with a caller's mask that permits only "unchanged" in every column. The mask makes its rows certain (probability
1), so they add nothing to the surrogate or the pull; the executor flies it to its end (a landing, a crossing, its time
limit). The other aircraft must keep their separation from it. Removing it would give the others the space that the
loss took, so a loss would pay the window.

**The end of a window.** A window ends when none of its commanded aircraft is spoken. A silent aircraft is then
halted: no reward can change after that row. With one commanded aircraft, the window ends at its loss, as post-training
D93 says.

### 2.5 What the reward leaves out (D151)

- **No term for the time.** A late landing earns as much as an early one. The judge's time limit (the remaining
  observed time × 1.5, and 900 s for each go-around) bounds a delay, and the model cannot read it (vocabulary D90), so
  a long delay risks the whole reward. The readouts give each landing's delay against its record and the spacing at
  the threshold beside the records (§5.4). If they show the model spreading the aircraft out, the user decides a term
  with the numbers (O18). A delay term needs a scale (how many seconds a landing is worth), which is not in the
  regulation and would be a value fitted to the data.
- **No shaping by the margin of separation.** The judge gives each pair's margin (`separation.Judged.margin`), and the
  archived design built a reward from it after a go-around (§9.1). A margin term pays for distance that no rule asks
  for, and it is a second target beside the outcome. The margin stays a readout.
- **No payment for a go-around.** As D30. With the sum (D141), a go-around that saves another aircraft is paid by that
  aircraft's landing, so no payment of its own is needed (§2.6).
- **No probes.** Post-training §2 item 6 and §6.3: a cross-entropy on a forced "go-around" taught the word and not
  when to say it.

### 2.6 The incentives, case by case (Claude's arithmetic)

| Case | Change of W in a group (v's continuation against the first sentence) | Learnt |
|---|---|---|
| v keeps its speed up, its follower lands instead of losing separation | +1 | The help |
| v (the follower) goes around before the loss and lands later | +0.9 | D30's go-around, unchanged |
| v (the leader) goes around so that its follower lands; v lands later | +1 − 0.1 = +0.9 | A go-around for another aircraft, paid by that aircraft's landing |
| v goes around for another aircraft and does not land (time limit) | +1 − 1 = 0 | Nothing: no sacrifice of one aircraft for one other |
| v gives up its landing so that two others land | +2 − 1 = +1 | The sacrifice of one for two (a controller's choice) |
| v delays its landing by 120 s and nothing else changes | 0 | Nothing: no cost of the time (§2.5) |
| v cuts in front of a recorded follower that then loses separation (only the follower responsible) | 0 | Nothing (D145; O17) |

---

## 3 The window and the loop

### 3.1 The window (D146)

- **The commanded aircraft**: the anchor and every flight of the same airport and split with a closed-loop sentence at
  the chosen Δ (inside the base's selection or not, post-training D76) whose row 0 is in [the anchor's row 0, + L).
  Each has its own row 0, first predicted step, start and time limit (the start of a closed loop, vocabulary §6 item
  5).
- **The recorded aircraft**: every other flight of the airport and the split in the air at the step, from its entry
  into the arrival slice to its landing (post-training D93), with its R and G as in stage C (post-training D99, D117).
  This includes the aircraft already in the air at the anchor's row 0 and every aircraft that enters after the span.
- **The draw of a round** (post-training D124, generalised): the train's flights in one permutation (the seed and the
  round); each flight in turn is an anchor; a window is kept when it does not share a commanded flight with a window
  already in the round's batch (a loop holds each flight once). Windows of different batches may share flights.
- **Compressed windows**: each commanded aircraft other than the anchor is moved earlier, its offset from the anchor's
  row 0 multiplied by c (c uniform in [c_min, 1], the moved offset rounded to a whole Δ); the recorded aircraft do not
  move. A moved aircraft's record moves with it: its observed rows, its row 0, its first predicted step; its states
  and words do not change (the executor and the inputs read no absolute time, except the landings it counts, which are
  read at its new time).
- **Left out** (post-training D113, for each commanded aircraft): a window in which a commanded aircraft, on its
  record at its first predicted step with no runway in force, loses separation that it answers for against the
  records of every other aircraft of the window. A stated limit: in the loop the earlier commanded aircraft fly the
  model's states, so an aircraft can still open inside a loss that they made; that loss is then charged by its rule,
  and the varied earlier aircraft get the credit (§2.3).
- **With L = 0**: the window is stage C's real window (one commanded aircraft, every other one recorded).

### 3.2 One step of the loop

At each Δ row of the window's grid (one UTC instant for every aircraft):

1. The aircraft whose row 0 is this row join the loop (D150). Before its first predicted step an aircraft's rows are
   its observed rows, encoded as in stage C.
2. For each commanded aircraft still flown: its tokens (post-training §3, `post.edges.tokens`) — the recorded aircraft
   from their records, every other commanded aircraft from the states that the loop flew (its position at the row and
   the 2 s row before it), with its R and G from its words in force (silent aircraft included); the speed-word mask
   (D148).
3. Every spoken aircraft says one row, all at once (D148); a silent one says "unchanged" (D144); the executor flies
   every one of them for Δ.
4. The separation judge on the scene at the next row (§2.4); a responsible spoken aircraft becomes silent.
5. A commanded aircraft that crossed its threshold adds its landing to the other aircraft's landings from its crossing
   time (D147).
6. The window ends when no commanded aircraft is spoken (D144).

**R and D23.** An aircraft's R is used, by any input or token, only from the row after its first predicted step
(post-training §3); the test of D31 (a change of one aircraft's runway word leaves every input and token of the scene
at the rows up to its first predicted step the same, bit for bit) runs with several commanded aircraft.

**The faulty points** of the recorded aircraft are counted as in stage C (post-training D114). A commanded aircraft
flies on the executor and has none.

### 3.3 Speaking for several aircraft

- **One batch.** The commanded aircraft of all windows of a batch are one batch of the speaker (prior §7 item 3),
  each joining at its own tick (D150, §6.3). The prior's forward pass, the masks over the batch (prior B14) and the
  executor's steps are those of stage C; an aircraft that has not joined is not spoken and not flown.
- **Random numbers.** Each commanded aircraft has its own stream, from the seed, the round, the window's place in the
  round and the aircraft's place in the window; a copy of a window copies every stream (D142).
- **The two passes** of a round are post-training D94's: the first pass speaks every window once; the second speaks
  the windows with W below their count again with the first pass's numbers, and at each branch point of each varied
  aircraft copies the window's state K times. The check of D94 holds for every aircraft: the second pass says the
  first pass's words and flies its states within `STATE_BOUND_M` up to the last branch point; a window that differs is
  counted, reported and gives no group.

---

## 4 Loss and masks

**Masks.** The prior's grammar and procedure masks under the set that the prior was trained under; the masks of a
caller: "go-around" after an aircraft's second go-around (post-training D91), the speed-word mask (post-training §3,
D148), and the silent aircraft's mask (D144).

**Loss.** Post-training §2 item 5, unchanged (`post.loss`): the clipped-ratio surrogate (ε = 0.2), word by word, on
the counted rows of each sample; the pull to the base (weight 0.04); the teacher-forced data term (weight 1) on
single-aircraft samples of the train days in the base's selection `landed` (post-training D36, D76). A sample is one
varied aircraft's sentence of one group: its rows, the speaker's records of them and its tokens, as a sample of stage
C (`post.branches.Sentence`). Every counted row weighs the same (post-training D115); the groups of a file are taken
in a shuffled order (D130); the surrogate and the pull in eval mode, the data term with the base's dropout (D107); the
ratio's denominator is the model at the start of the pass; the traffic attention at its own learning rate.

---

## 5 Selection, validation and readouts

1. **The start (round 0)** is stage C's chosen round, read in stage D's windows.
2. **The selection readout** of each round: a fixed set of windows of the select days (drawn once with the seed,
   D146's rule), spoken once with the same numbers every round. Its size is set so that the paired standard error of
   W per aircraft between two rounds can resolve the change that the user wants to see: MC5 proposes it from round 0's
   spread. Why: the archived campaign's selection set could not resolve a change of 0.012, and its chosen round's
   gain (+0.025) came mostly from taking the best of 8 rounds (§9.1).
3. **The validation readout**: the chosen round, read one time on the val days (outline D85), with a claim before the
   read (as post-training D132).
4. **The readouts** of each window set (no criterion: D7): by airport and kind,
   - W per commanded aircraft; each aircraft's outcome; landed; go-arounds; silent aircraft;
   - the losses of separation by pair (commanded–commanded, commanded–recorded with the commanded one responsible,
     recorded responsible: D145), by kind of the judge;
   - the delay of each landing against its record, the spacing at the threshold of successive landings (p1, p5, p50)
     and the landing order against the recorded order, beside the records of the same windows (O18);
   - the share of informative groups, the windows whose second pass differed, the rows where the speed-word mask
     acted, the faulty points (post-training D114);
   - a baseline, once (MC1): the commanded aircraft flown on their own closed-loop sentences of the artefact in the
     same windows (the executor's own losses of separation; the archived design measured 1.17 points above the
     records, §9.1).

The user sets the criterion that chooses the round (D7) before the validation readout.

---

## 6 What stage D needs of the other stages

### 6.1 The post-training's public interface (D149)

The post-training document gets a public interface (a new §9). Stage D imports only its names. The table gives what
stage D needs; the names are the implementer's, written into post-training §9 when the code is built.

| # | Item | Today | The generalisation |
|---|---|---|---|
| 1 | The scene and a window | `post/scene.py` `Window` (one `commanded`), `Recorded`, `AircraftAt`, `Scene` | A window holds its commanded aircraft (one or more, each with its own shift); stage C's windows hold one |
| 2 | The loss of separation an aircraft answers for | `post/traffic.py` `commanded_loss` (aircraft 0), `joined` (one commanded) | The loss that a given aircraft of the scene answers for; `commanded_loss` is that of aircraft 0 |
| 3 | The tokens and their conformance | `post/edges.py` `tokens`, `experiments/post_window_loop.py` `checked_edges` | Unchanged: a commanded other aircraft is an `AircraftAt` with its flown states |
| 4 | The reward of an aircraft | `post/reward.py` `reward`, `present_runways` | Unchanged; the landings are D147's |
| 5 | The window loop | `experiments/post_window_loop.py` `WindowLoop` (a window is one row of the batch) | A window is one or more rows of the batch; silent aircraft (D144); the end of a window when none is spoken; each aircraft's result; the copy of every aircraft's state |
| 6 | Branch training | `post/branches.py` `first_numbers`, `continuation_numbers`, `branch_points`, `Group`, `samples`; `experiments/post_branches.py` `branch_round` | A group names its varied aircraft; the numbers of an aircraft in a window; `branch_round` takes the varied aircraft and their points as a rule that the caller gives (stage C's: the one aircraft, its reward below 1) and the window's reward (stage C's: r) |
| 7 | The loss | `post/loss.py` `Samples`, `update_loss`, `one_pass`; `post/traffic_attention.py` `parameter_groups` | Unchanged |
| 8 | The campaign's steps | `experiments/post_train.py` `batches`, `Speakers`, `train_pass`, the resume, the checks at the start | The round's skeleton given the draw, the windows' loop and the readout of a stage; stage C's campaign is that skeleton with its own |
| 9 | The checkpoint | `ts-post-checkpoint-v1`, its identity (post-training §4 item 3) | A reader that opens a chosen round (the start of stage D) |

Each generalisation runs D149's test before stage D uses it: stage C's windows give the same words, states, rewards and
groups, bit for bit, on fixed inputs (the CPU, one thread).

### 6.2 The vocabulary's and the prior's public interfaces (D150)

Stage D does not change the code of `instructions/`, `autopilot/` or `prior/` for itself (as outline D95): each change
is written in that stage's public interface first and built by the one implementer (outline §5 rule 1), with that
stage's checks.

| # | Interface | The change | Its test |
|---|---|---|---|
| 1 | Vocabulary §6 item 5, the start of a closed loop and `Loop` | Each flight's join tick (§6.3 item 1) | Every join tick 0: today's loop, bit for bit; a flight that joins at tick j flies, with the same words, the states that it flies alone, within `STATE_BOUND_M` |
| 2 | Prior §7 items 2, 3 and 7: `LoopRows`, the speaker, `SpeakingLoop` | Each aircraft's join tick; one row of the batch holds absent, observed and said aircraft (§6.3 items 2–4) | Every join tick 0: today's words, records and states, bit for bit; an aircraft that joins at tick j says, with the same numbers, what it says alone, to the tolerance of post-training §6.4 |
| 3 | Prior §7 item 2, the landings of a loop | A landing added to chosen aircraft's landings while the loop runs; copied with the loop (§6.3 item 5) | No landing added: today's inputs, bit for bit; a landing added at a time changes only the counts of the rows after it |
| 4 | Prior §7 item 3, the masks of a caller | A caller's mask in every column (D144's silent aircraft): to check that the speaker takes one in each column and that the grammar permits "unchanged" in every column after the first predicted step | A silent aircraft flies its words in force; its rows' log-probability under records is 0 |

### 6.3 The design of D150: aircraft that join a loop

One loop holds the commanded aircraft of every window of a batch. The loop counts **ticks**, one a Δ row, on the
window's UTC grid (tick 0 at the earliest row 0 of the window; the windows of a batch are aligned by their own tick
0, as stage C's are by their row 0). Each aircraft has a **join tick** j: its own row 0. At tick t an aircraft is

- **absent** for t < j: nothing of it is encoded, said or flown;
- **observed** for j ≤ t < j + s (s the Δ rows to the first predicted step, 4 at Δ = 4 s): its own row t − j is its
  observed row, as the start gives it back;
- **said** for t ≥ j + s, while it is flown: the speaker says its own row t − j and the executor flies it;
- **done** after the executor is done with it or the caller ends it, as today.

The parts, each in its own stage's code, each with the readable form kept as today's call with every join tick 0:

1. **The start and `Loop`** (stage A). `Start.moved` takes each flight's join tick; `Loop` gives the executor the start
   cycle of each flight (join tick + s) × the cycles of a Δ row: the executor's multi-aircraft batch, which already
   holds a flight before its own cycle 0 and counts its time and its time limit from it (`Executor(start_cycle=…)`,
   checked by the executor conformance in that way of flying). A flight that has not started is not heard: the grammar
   and the bound of go-arounds read only started flights, as they read only flights not done or halted today; the
   sentence time of each flight is its own. `Loop.copy` keeps the join ticks.
2. **`LoopRows`** (stage B). Each aircraft's join tick in place of the one first predicted step of the batch: the
   inputs of aircraft b at tick t are those of its own row t − j_b, at its own UTC time (its entry time and its first
   row, as today); an absent aircraft's row is not present.
3. **The speaker** (stage B). One row of the batch may hold absent, observed and said aircraft. A row is given with
   each aircraft's role; the network runs once on the whole row; an absent aircraft's row is written into the cache as
   not present (`Past.present` false, so no later row reads it: the attention already masks rows not present and always
   lets a row read itself, so no row has every key masked); an observed one's as `observe` writes it; a said one's words
   are drawn as `speak` draws them, with its numbers and masks, and only its records are kept. The mark of the first
   predicted step is each aircraft's own (its first said row). `observe` and `speak` stay, as the forms of a row in
   which every aircraft has the same role.
4. **`SpeakingLoop`** (prior §7 item 7). `step` advances one tick for every aircraft (absent, observed or said by its
   own clock); `observe` stays as today's phase for a loop whose join ticks are all 0. Each aircraft's records, words,
   states and sentence (`sentences`, `said`, `states`, `generated`) are of its own rows from its row 0, as today, so
   stage C's samples and the loss read them unchanged.
5. **The landings of a loop** (stage B, `LoopRows` and `SpeakingLoop`). A landing (its flight, runway and time) added
   to chosen aircraft's landings: each aircraft's `LandingIndex` replaced by one that holds it (the index keeps its
   checks: time order, the sealed days, one landing a flight). The counts of a row read the landings before its time,
   so a landing added at its crossing time changes only later rows. Copied with the loop.

**What it costs.** An absent aircraft's row is computed by the network and thrown away (it is in the batch): the
waste is the share of absent rows in a batch, at most L over the window's length. The masks, the inputs and the
executor skip absent aircraft. The model's checkpoint, the base and C10's rounds are unchanged; free generation and
stage C call the loop with every join tick 0.

**Tests** (before stage D uses it): every join tick 0 gives today's words, records, states and outcomes, bit for bit
(the CPU, one thread), for free generation and for stage C's window loop; an aircraft that joins at tick j, beside
others that joined earlier, says with the same numbers the words it says alone and flies its states within
`STATE_BOUND_M` (its probabilities to the tolerance of post-training §6.4); an absent aircraft is never read by
another aircraft's row, never heard by the executor and never in a record; a copy of a loop with join ticks continued
with the same numbers says what the original says.

---

## 7 Gates and identities

The gate of this document is the multi-aircraft control. The user sets its criteria (D7). The identities follow D21
(outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The edge features | Stage C's conformance check (post-training §4 item 1), unchanged, run in every process that computes tokens |
| 2 | A window readout | The checks of the code it uses, at its start: the edge features', the executor's and the closed loop's (vocabulary D71, D73), and D149's check of the generalised loop on fixed windows. The commit as information. No digest of code |
| 3 | A checkpoint of stage D | Its format name (`ts-multi-checkpoint-v1`); the identity of its start (the chosen round's checkpoint identity, post-training §4 item 3, which holds the base's); the set of procedure masks; the shape of the traffic attention; the seed; L, the kinds and c_min; the windows of each round by their anchors, their commanded flights and their shifts |
| 4 | A campaign | `ts-multi-train-v1`; its settings in the campaign's record; a resume requires them the same |

---

## 8 Values

| Item | Value | Source |
|---|---|---|
| Reward of an aircraft | D30: 1; 0.9ⁿ after n go-arounds; 0 otherwise and for a loss it answers for | D140 |
| Reward of a window | The sum over its commanded aircraft | D141 |
| Advantage | The window's reward less the mean of its group; on the varied aircraft's rows after the branch point | D142 |
| Varied aircraft | Each aircraft with r < 1 and the commanded aircraft of the losses they answer for | D143 |
| Branch points | The varied aircraft's first predicted step and the points of the window's grid after it (the anchor's first predicted step and every 120 s after it), before its event; K = 8 | D143, post-training D37, D94 |
| Window | An anchor and every flight with a sentence whose row 0 is within L after it; L, the kinds, c_min: O16 | D146 |
| Loss | Post-training §2 item 5: ε = 0.2; pull to the base 0.04; data term 1 | §4 |
| Most go-arounds of an aircraft | 2 | Post-training D91 |
| Speed-word mask | Post-training's; a commanded leader's target is its present speed along its course | D148 |

---

## 9 Evidence

### 9.1 The archived multi-aircraft work (instruction-v3; read as evidence only)

From the archived design (`multi_aircraft_design.zh.md` §2.2, §6.2, §6.6 steps 7 and 8) and its readouts. Each
number was measured on instruction-v3's vocabulary, executor and models, not on this design's; it says what happened
there, not what will happen here.

- **Windows with several commanded aircraft** (step 7, `readouts/2026-09-30_m3_window.zh.md`): windows of 20 min,
  one every 10 min; on the select days, the losses of separation were 13.4 % with the model, 4.9 % with the labelled
  words flown and 3.8 % with the records. The correlation of the rewards of the aircraft of one window: 0.05.
- **Training in those windows** (step 7.6, `readouts/2026-10-01_m4_window.zh.md`): each aircraft's own reward (0 or
  1), the advantage over the K whole-window sentences of the same aircraft, 8 rounds. The chosen round (`window r5`)
  had no gain on the real windows of the validation days (reward −0.0050 ± 0.0032); the gain on the selection set
  (+0.025) came mostly from taking the best of 8 rounds, and the set could not resolve a change of 0.012. The user
  then kept the one-commanded model (2026-10-01).
- **Joint encoding** (step 7.6 item 5): every aircraft of a scene encoded together, with a tensor of steps ×
  aircraft² × 192 in each layer; 5.3 GB a layer for the largest augmented window (23 aircraft), so the scoring was cut
  into blocks of steps and recomputed in the backward pass.
- **The aircraft that lost separation** stayed in the scene, flying its words in force, with reward 0: the user's rule
  of 2026-09-30 (archived design §9 item 29), "not removed: removing it gives the others space, a hidden reward for the
  failure".
- **The order inside a step** (archived design §2.6): the masks were computed along the approach clock, front to back,
  so that a later aircraft's speed mask read the earlier one's new words. The two separation masks blocked 0.18 % of
  the labelled speed words and 0.71 % of the clearance words (the clearance word does not exist in this vocabulary).
- **The executor and the records** (archived M0 step 5): with the labelled words flown in traffic, 3.93 % of the
  flights lost separation; with the same flights on their records, 2.76 %: the executor added 1.17 points.
- **A composite reward after a go-around** (archived step 8.9): 0.48·L + 0.14·S + 0.14·H + 0.14·Q, with times taken
  from the data's p50 and p95 (421 s, 598 s) and a margin term. Replaced by D30's outcome only (post-training §2
  item 2).
- **Probes** (archived step 8.10): a forced "go-around" with a cross-entropy; 88 of 108 of the model's own go-arounds
  ended worse than round 0 (post-training §6.3).
- **Rescue by speaking again** (step 7.7, `readouts/2026-10-02_window_rewind.zh.md`, one aircraft spoken again 8
  times): an event rescued at least once in 8 from the start in 69.6 %, from 120 s before it in 41.7 %, from 60 s in
  25.4 %. Post-training D37's branch points come from it.

### 9.2 Stage C

- One commanded aircraft per window: on A34's train days, 40.1 % of the windows have a leader in the air at their
  first predicted step, 29.4 % no other aircraft at all (post-training §6.1). The windows of stage D add the aircraft
  that enter after the anchor; their count for each L is MC1's census.
- The cost of a round of C10's settings: about 60 min with three speaking workers, the speaking bound by the CPU
  (outline D138). Stage D speaks more aircraft for each window and varies more aircraft for each failed window; MC5
  measures it at the formal size.

### 9.3 What this design takes from the archived work and what it leaves

| Taken | Left |
|---|---|
| The failed aircraft stays in the scene (D144) | The joint encoding of every aircraft (D139) |
| A commanded aircraft's recorded landing is not counted; its loop landing is (D147) | The order of the aircraft inside a step (D148) |
| Each aircraft's reward by its outcome and its responsibility (D140) | The advantage over K whole-window sentences of every aircraft (D142) |
| Compressed windows as a kind (D146) | The composite reward, the margin term and the probes (D151) |
| One executor batch for aircraft that start at different times (D150) | Fixed overlapping windows of 20 min with the aircraft in the air at the start recorded: here every flight can be an anchor, and L = 0 is stage C (D146) |
| A selection set sized to resolve the change sought (§5) | The source-hash identity of the edge features (stage C replaced it by a conformance check) |

---

## 10 Code

**Packages and import rules** (`tests/test_architecture.py`):

- `multi/` — what is stage D's own: the window draw (anchors, spans, compressed windows, the census), the window's
  reward (`reward.window_reward`, the sum of `post.reward.reward`), the varied aircraft and their branch points
  (`credit.varied`, `credit.branch_points`). It imports `instructions/`, the names of the post-training's public
  interface (§6.1) and the names of prior §7; it does not import `autopilot/`. `post/` never imports `multi/`.
- The runners of stage D (`experiments/multi_*.py`): `multi_windows` (the census, MC1), `multi_train` (the campaign,
  through the post-training's campaign steps, item 8 of §6.1), `multi_validation`, `multi_profile`,
  `multi_training_export`. They import from `autopilot/` only the names of vocabulary §6, from `prior/` only those of
  prior §7, from `post/` and stage C's runners only those of §6.1, and `multi/`.
- The window loop stays one class (`experiments/post_window_loop.py`), generalised (§6.1 item 5): no second loop.

**Formats** (new names, principle 8): `multi-windows-census-v1`, `ts-multi-train-v1`, `ts-multi-checkpoint-v1`, the
Training sets' index and sample formats (named in MC7).

**Size, estimated.** `multi/` about 300 lines; the generalisations of stage C about 300 lines changed; the interfaces
of §6.2 about 200 lines in `autopilot/start.py`, `prior/loop.py`, `prior/landings.py` and
`experiments/prior_speaking_loop.py`; the runners about 600 lines; tests about 1,000 lines.

---

## 11 Milestones

The rules of outline §5 apply: one implementer on `dev-two-tier-v4` (rule 1), a review before each commit, light
tests while C10 runs (only the changed modules' tests; the full suite at the big commit points).

**MC0. The interfaces** (§6). After the user's decisions and Claude's text in the vocabulary, prior and post-training
documents.

- §6.2 items 1–4 in stages A's and B's code, each with its test; the checks of vocabulary D73 on the formal artefact
  (outline §5 rule 2) after the change of `autopilot/`.
- §6.1 in stage C's code, with D149's test on fixed windows (C10's census, 10 windows of each kind and airport, seed
  1337) before and after each generalisation.
- C10 keeps its own worktree; nothing of MC0 touches it.

**MC1. The census of multi-aircraft windows.** On the train and select days of A34's artefact, in a scratch directory
(outline D55), for L of 0, 5, 10 and 20 min: the commanded aircraft of a window (p50, p90, largest), the recorded
aircraft of a step, the windows left out (D146) and why, the losses of separation on the records between two
commanded aircraft and between a commanded and a recorded one (by who answers: O17), and the baseline of §5 item 4
(the commanded aircraft flown on their closed-loop sentences in the windows). For compressed windows at c_min of 0.6
and 0.8, the same counts. Report to the user, who chooses O16.

**MC2. The window loop with several commanded aircraft** (§3). Tests: L = 0 equals stage C (D149); every input and
token of the scene at an aircraft's rows up to its first predicted step is unchanged by a change of its runway word
(D23, D31); an aircraft that answers for a loss is silent and keeps flying; a window ends when none is spoken; a
landing in the loop changes the other aircraft's counts after it only (D147); a copy of a window continued with the
same numbers says and flies what the original does (D94).

**MC3. The window's reward and the credit** (§2). Tests: W is the sum; the varied aircraft and the branch points of
D143; in a continuation only the varied aircraft's numbers change; the advantage applies to the varied aircraft's
rows after the branch point only; with one commanded aircraft the groups are stage C's.

**MC4. The runners**: the census (MC1's code as a runner), the campaign, the selection readout, the validation readout
(its claim, as post-training D132). Smoke: two rounds, a few windows of each airport, with the user's O16 (outline
§5 rule 7); `--resume` continues.

**MC5. The profile at the formal size** (as post-training C8): the time of a batch and of a round with the two passes,
the memory of the host and of the GPU for N speaking workers (post-training O15's check), the bytes of the groups; the
spread of round 0's W per aircraft on the select windows, and from it the size of the selection set (§5 item 2).
Claude proposes the campaign's settings; the user decides them and the criterion that chooses the round (D7).

**MC6. The formal campaign** from the main checkout after the user's merge (outline §5 rule 1), its intent in
`docs/experiments/intents.json` before the launch; the validation readout of the chosen round.

**MC7. The Training view of stage D** (outline §6): the window view of stage C with several commanded aircraft — the
list of windows, the tabs Labelled · Start · r1 …, each commanded aircraft with its own sentence, the silent aircraft
marked from its loss on, the losses by pair, the envelopes of each flown sentence (outline D135); the results page's
sections Rounds, Speed, The checks, The window (outline D134). A word of a commanded aircraft flies its segment live
in the window.

**MC8. Close of stage D.** The full ts suite; `docs/reference/runners.md`; the code index for §10; the report to the
user. The user merges (outline §5 rule 11).
