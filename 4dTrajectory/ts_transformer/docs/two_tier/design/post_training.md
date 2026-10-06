# Two-tier model: the post-training and the multi-aircraft work

**Summary.** The post-training trains the prior further in a closed loop: the prior speaks, the executor flies, the
judge decides, and the reward comes from the outcome. It runs in windows of recorded traffic, so this document also
gives the multi-aircraft work: scenes, the separation judge and the separation masks. It is stage C of the plan. It
reads the vocabulary and the prior only through their public interfaces (vocabulary §6, prior §7) and their decisions.

**Language.** This document uses the writing rules of ASD-STE100 (Simplified Technical English): short sentences, active
voice, one meaning for each word, and technical names that the terms define. The words were not checked one by one
against the STE dictionary. Units are SI only (m, m/s, deg, s). Values in feet, knots or nautical miles occur only in
quotations of regulations, with the SI value given one time.

**Scope of this document.** This document gives the design of stage C: its decisions, its rules, its values, the
evidence for them, its code and the milestones not done. The principles and the shared rules are in the outline
(`outline.md`). The state of the work, the commits and the implementer's readings are in the implementation log
(`readouts/2026-10-05_stage_c_implementation_log.md`); the specifications of the milestones done are there too (log
§22). Paths in backticks are relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or name
the repository root; `readouts/` is `docs/two_tier/readouts/`. The documents of the earlier design (`*_design.zh.md`,
`two_tier_framework.zh.md`, `two_tier_stage_notes.zh.md`) are in `archive/two_tier_v3_2026_10/docs/`.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3). A decision gives the rule and, after "Why", its reason;
the measurements behind it are in §6 or in the readout named.

| # | Item | State | Source |
|---|---|---|---|
| D29 | Post-training starts from the base model in the multi-aircraft setting "one aircraft commanded". There is no single-aircraft post-training stage (§2) | Decided | User, 2026-10-03 |
| D30 | The post-training reward comes only from the outcome: 1 for `landed` without a go-around; 0.9ⁿ for `landed` after n go-arounds; 0 for every other outcome and for every loss of separation. No payment for a go-around without a landing. No mask on where "go-around" can be said (D10) (§2) | Decided | User, 2026-10-03 |
| D31 | Multi-aircraft inputs and judgements (§3): the landing context of an aircraft of a scene counts the landings of the closed loop (with one commanded aircraft, only that aircraft has inputs: D98, D105); D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: D92). (The part of the prior: D31 there) | Decided | User, 2026-10-03 |
| D36 | The teacher-forced data term of the post-training uses single-aircraft samples of the closed-loop sentences, not scene samples. The flown states keep the observed path, not the observed time (vocabulary D42), so two aircraft of one scene do not keep their observed spacing (§2, §3) | Decided | User, 2026-10-04 |
| D37 | Branch training. Each training aircraft is spoken one time. When its reward is less than 1, it is spoken again from its state at its first predicted step and every 120 s after it, before the event that ended it. Each branch group compares only the words after its branch point (§2 item 9). How the state at a branch point is made, and the test of item 4: D94 | Decided | User, 2026-10-04 |
| D76 | The flights outside the base's selection (prior D75) come back in the post-training only as starts of the closed loops, rewarded by their outcome (D30). The teacher-forced data term (D36) uses the same selection `landed`, so a sentence whose own words do not land is never imitated | Decided | User, 2026-10-04 |
| D91 | The most go-arounds of a flight in a loop of the post-training is 2, as in the prior's free generation (prior D68): the start of a closed loop gets 2, and after a flight's second go-around the loop forbids "go-around" (a mask of a caller, prior §7 item 3). Why: a window without other aircraft must say and fly what free generation says and flies (§2 item 1, a test of C4), and that needs the same bound; no closed-loop sentence of the artefact has more than one go-around (prior D68); D30 already makes a chain of go-arounds cost reward | Decided | User, 2026-10-05 (O9) |
| D92 | "Established on the final" (D31) is a function of one row: G is false; the aircraft is inside the region of the final of R (inside the FAF and the LPV cone: the region of the prior's procedure masks and of its readout of the go-around probability, prior D64, D72, §7 item 6); and its track is within 20° of the course of R (7110.65BB 5-9-2 a, TBL 5-9-1: the row for an interception less than 2 NM from the approach gate, which holds inside the FAF). After "go-around", the aircraft is not established until a runway word ends G. The judge's "lined up" (vocabulary §5.8) and the turned-in rule of the visual reading (7-4-4 c) keep 30°. This stage has no mask for the spacing on the final: the reward teaches it (D30: 0 for a loss of separation); the speed-word mask stays (§3). The regulation text also lets an aircraft be established on the final approach course outside the FAF (FIG 5-9-1 Example 4; 5-5-4 j, "within 10 NM"); the region stays narrower, so that the masks, the judge and the readout read one region. The rule is Claude's reading of the text, which `docs/literature/arrival_separation/README.md` §8 quotes to its paragraph | Decided | User, 2026-10-05 (O6), on Claude's proposal; 20° and the region: user, 2026-10-05, on the regulation text |
| D93 | A window has no length of its own. It starts at the commanded aircraft's row 0 and ends where the commanded aircraft is done (the judge's outcome, its time limit included: vocabulary §6 items 5 and 6) or has a loss of separation that it answers for (§3). At each step, its other aircraft are every other flight of the airport and the split in the air at that time, replayed along its record. Stage C reads no time limit (vocabulary D90). Why: with two go-arounds (D91), a flight can fly for more than 20 minutes after its first predicted step; a cut at a fixed length would leave it without traffic or end it early | Decided | User, 2026-10-05 |
| D94 | How branch training (D37) gets the state at a branch point, and its test. A round has two passes. First, every window is spoken one time with its own random numbers (from the seed, the round and the window's place in the round). Then the windows whose reward is less than 1 are spoken again with the random numbers of their first sentence; at each branch point, the state of the window is copied K = 8 times: the loop of the executor (vocabulary §6 item 5, D97), the speaker (prior §7 item 3, D96) and the window's own state (its step, its separation judge, its landings). Each copy continues with its own random numbers (from the seed, the round, the window's place, the branch point and k), from its branch row: that row's words are its first. The second pass is checked against the first on the words before the window's last branch point and on the positions and heights of its 2 s rows up to that point (within `STATE_BOUND_M`, vocabulary D97 (3)); a window that differs is counted, reported and gives none of its groups. The test of D37 item 4: a continuation with the random numbers of the first sentence says the same words as the first sentence and flies its states within `STATE_BOUND_M`. Why: the copies spend the extra speaking only after the branch points (§2 item 9), and only the windows that need a branch are spoken again, so no state of a window that lands is saved. In another batch, the probabilities can differ in their last bits, so a word drawn very near a boundary can differ, and the executor's states differ by a rounding (§6.4) | Decided | User, 2026-10-05, on Claude's report of stage C's readiness; the second pass's check and the branch row: the implementer's readings P26, P30, accepted (D125) |
| D98 | The traffic attention reads each other aircraft as a token from its recorded state only: its edge features to the commanded aircraft (§3) and its own motion. The prior runs only on the commanded aircraft. The commanded aircraft's landing context is the prior's count of the landings before the step without its own landing (prior §7 item 2, with the window's landings of D105); with one commanded aircraft, that is the rule of D31. Why: the recorded states are data, computed one time for a window; the prior's hidden state of a recorded aircraft would change each round with the prior's weights, and it would need words said to that aircraft (a flight without a sentence has none) and a landing context from the loop | Decided | User, 2026-10-05 (O12), on Claude's proposal |
| D99 | A recorded aircraft's G comes from its labelled sentence: G is true at the rows after each labelled go-around row, up to and including the row where the runway is said again (a word is in force from the next row, as for the commanded aircraft); a flight without a sentence has G false. Only the separation judge and the masks read this G ("established", D92); no input reads it. The labelled go-around rows are a withheld field (vocabulary §6 item 3). The commanded aircraft's G is that of its own words, never its labelled one: a window refuses a commanded record that would carry a labelled G. Why: a recorded flight has no words said in the loop, and with G false a recorded go-around inside the final's region and lined up would read as established (on A34's train days, 80 flights carry G, 3,656 rows) | Decided | User, 2026-10-05, on the implementer's reading P3 |
| D100 | The windows of a round are real, A and D in equal counts (1 : 1 : 1). The count of windows B is a setting of the campaign (O13). Why: C1's census admits A for every real window and D for 40 % of them (§6.1) | Decided | User, 2026-10-05, on C1's census |
| D101 | The speed-word mask acts only between 9,260 m (5 NM) from the threshold and the FAF, because it needs both aircraft established (inside the FAF, D92). That band is 0.1–2 km at most runways of A34's candidates (5.8 km at KSJC 30L), and the mask never acts at KSMF 35L/35R and KSTL 30L/30R (their FAF is inside 5 NM). A stated limit: the readouts report how often the mask acts | Decided | User, 2026-10-05, on the implementer's report |
| D103 | The augmented windows A and D (§2 item 4) are built from real windows. A: one flight of the same airport and split whose recorded landing is more than 3,600 s from the commanded flight's recorded landing, drawn uniformly among such flights, inserted with its record shifted by a whole number of Δ, drawn uniformly, so that it lands within ±180 s of the commanded flight's recorded landing; its own landing is its roster landing shifted by the same amount (D105). D: the aircraft next ahead on the approach clock at the first predicted step, its record shifted by a whole number of Δ drawn uniformly in [−120 s, +120 s], never 0. The shifts often put the moved aircraft inside 3 NM and 1,000 ft at the first predicted step; such windows are left out (D113) | Decided | User, 2026-10-05, on the implementer's proposals P7, P18; the ranges kept on D113's count |
| D104 | The reference of the edge features' conformance (§4 item 1) is written by the formal census (`post_windows`) beside its output, `conformance/edges.npz`, with the geometry of its airports, so that the check reads fixed inputs: 10 windows of each airport (train, seed 1337), steps every 60 s from the first predicted step while the record lasts, tolerance 1e-6 (float32 tokens). Every later process that computes edge features reads it by its path and runs the check first, as the references of the labeller and the executor are written beside the artefact that they were written from (vocabulary §7.2). The formal census is a formal build: it waits for the user's order (outline §5 rule 7) | Decided | User, 2026-10-05, on the implementer's proposal P9 |
| D105 | The landings that the commanded aircraft's inputs count follow the window's scene (§3): the tracks roster's landings with the window's changes — an inserted aircraft's landing (A) added at its shifted time, a moved aircraft's landing (D) moved by its shift — less the commanded aircraft's own landing (D31) and never a landing on a sealed test day (C32). A real window counts the roster's landings, so a window without other aircraft reads what free generation reads (§2 item 1). The present landing direction of the reward (§2 item 2) reads the same landings. The window loop gives them to the prior's function of a loop's row for each aircraft (prior D105). Why: with one index for each airport, the commanded aircraft read a value of the record that the augmentation changed — one landing on one candidate, up to 120 s in D and to the end of the window in A (`readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
| D107 | The clipped-ratio surrogate and the pull to the base (the KL) are computed with dropout off (eval mode), the base's side always; the teacher-forced data term with the dropout of the base's training (its configuration's: 0.1 in A, B and C, 0.2 in D; prior D40). The speaker and the log-probability under records refuse a model with any module in training mode (prior D107), so each step of the training computes the two kinds of terms in their own mode (the data term is a batch of its own, D36). Why: the speaker speaks with dropout off, so the ratio and the KL score the policy that spoke only with dropout off; with dropout on, at the parameters that spoke, the ratio was off by 3 % on average and by up to 0.18–0.26 for one word, and 22.5 % of the rows fell outside the clipping band (configuration A, random weights; `readouts/2026-10-05_stage_b_check.zh.md`). The data term is the base's own loss, which reached its select optimum with that dropout. Not measured on a post-training run | Decided | User, 2026-10-05, on Claude's recommendation |
| D110 | A mask of a caller (the speed-word mask, §3) is computed once for a row, from the state at the start of the row; it does not change with the row's own runway word. A row that says "go-around" thus keeps the speed mask that "established" (D92) gave before the word, though the aircraft is not established after it. Why: a caller's masks are fixed for the row (prior §4); the effect is small: the mask never blocks "unchanged" or "unspecified", and it acts only on rows where the aircraft and the one ahead are both established | Decided | User, 2026-10-05, on Claude's check of stage B |
| D112 | With no landing in the 30 min before the first predicted step, every runway is of the airport's present landing direction (§2 item 2): there is no direction to break. Why: read literally, no runway would be of it and every landing would earn 0; 1,119 of 40,530 train windows (2.8 %; select 3.0 %) of A34's artefact have no landing in those 30 min | Decided | User, 2026-10-05, on the implementer's proposal P19 |
| D113 | A window that opens inside a loss of separation is left out of the draw: the commanded aircraft on its record at its first predicted step, with no runway in force (the loop's state there), loses separation that it answers for. The rule holds for every kind of window, each judged on its own scene (window B on its moved record); the readouts state the windows left out. Why: the judge runs only after a row that the executor flew, so such a window would be judged lost one row later, reward 0, whatever is said. Counts on A34's artefact: §6.1 | Decided | User, 2026-10-05, on the implementer's count (P20) |
| D114 | A recorded aircraft whose observed track has faults (vocabulary D111) gets no rule in the scene: no window is left out and no step is skipped. Every readout reports the steps at which a recorded aircraft reads a faulty point (its row or the row before it is a fault row: its 2 s motion reads it) and the losses of separation in which the other aircraft reads one at the event's step or in the 2 Δ before it. The steps are counted from the commanded aircraft's row 0. Why: on A34's train days, 1,062 of 40,530 windows (2.6 %) have a marked recorded aircraft in the air, 796 of 3.8 million steps read a faulty point, and 2 of 1,374 losses on the records have one at or 2 Δ before the event | Decided | User, 2026-10-05, on C1's census of D111; the counting: readings P23, P29 |
| D115 | Every counted row weighs the same in the surrogate and in the pull to the base: the sum over a batch's counted words is divided by the batch's counted rows, not each sample by its own. Why: divided sample by sample, a continuation from a late branch point (few counted rows) weighs as much as a whole first sentence, so each of its words weighs more | Decided | User, 2026-10-05, on the implementer's proposal P13 |
| D116 | The traffic attention has one token network shared by the layers: the tokens of a step are embedded once, and each layer keeps its own attention over them. Why: with a token network in each layer, the memory grows with the layers × B·R·N_max (N_max the most other aircraft of the batch): at configuration A, B = 64, R = 300, N_max = 30, approximately 9 GB, more than the GPU. C8 measures the memory at the formal size | Decided | User, 2026-10-05, on the implementer's proposal P16 |
| D117 | The implementer's readings of C1–C7, accepted as built (each is a rule of this design): (P1) a recorded aircraft's first predicted step is its first row on the Δ grid + 16 s, and its R is in force at the steps after it; (P2) the motion of every aircraft, the commanded one too, is the 2 s displacement before the row, and an aircraft at its row 0 has no motion and is not established; (P4) a recorded aircraft is over its threshold at its last step in the air; (P5) a recorded leader's speed target is its present speed along its course, and the speed-word mask never blocks "unchanged" or "unspecified"; (P6) a token's own motion is the other aircraft's ground speed, vertical rate and direction of motion less the commanded aircraft's (0 without a motion or a frame), the required distance takes the aircraft ahead on the approach clock as the leader (a tie: the other aircraft), and the scales are 5,556 m (3 NM, asinh) for horizontal distances, 1,000 m for heights, 100 m/s for speeds, 10 m/s for vertical rates and 120 s for the closest-approach time; (P8) the census's leader in the air is the aircraft next ahead on the approach clock at the first predicted step, on the commanded flight's recorded runway or one separated as one, and a window is near a day cut when the day beside its flight's own day is not of its split and either its recorded landing is less than the scene's longest record before the end of its day or its row 0 is before the start of its day; (P11) the ratio and the pull are word by word, each word clipped on its own; (P12) the ratio's denominator is a frozen copy of the model at the start of the pass, scored through `masked_log_probability`; (P14) the pull reads the counted words only, and a first sentence in several branch groups counts once in each; (P15) no traffic gives the module no input and a zero output; (P21) a lost window's go-arounds are counted from its own words; (P22) a done flight's tokens are computed and never used, and its speed mask permits every word; (P24, P25) the census counts each real window's steps from the commanded aircraft's row 0 to the end of its record, and its losses on the records are each window's first loss that the commanded aircraft answers for after its first predicted step, and a step with no other aircraft within 8 NM plus 2,500 ft (the largest minimum, widened by the offset of a pair separated as one) is not judged | Decided | User, 2026-10-05, on Claude's review of the readings |
| D123 | Window B's move (§2 item 4): a turn about the airport reference within ±15°, a change of height within ±300 m and a scale of the speed within 1 ± 0.1, each drawn uniformly. Counts on A34's artefact: §6.1 | Decided | User, 2026-10-05, on the implementer's proposal P28 |
| D124 | The windows of a round share their flights: the round takes the train's real windows in one permutation (the seed and the round); the first ones make its real windows, and each kind A, D and B is built from the real windows in the same order until its count is reached (D admits only about 40 % of them, so it reads further down the permutation). A batch commands each flight once (a loop holds each flight once), so a real window and its A, B or D go to different batches. D113 is checked at the draw. The loss reads a round's branch groups branch point by branch point, never the whole round at once (a group is several MB; a round tens of GB) | Decided | User, 2026-10-05 (P32); the rest on the implementer's readings P27, P31 |
| D125 | The implementer's readings of C6–C11, accepted as built: (P33) a kind with fewer windows than its count does not stop the round, and the round's record shows the shortfall; (P34) the number of rounds is a setting of the campaign, which cannot be extended after it ends; (P35, P36) C8 measures the bytes of the branch groups and the memory of the data term (every train sentence of the base's selection in memory) before the formal run; (P37) a window set carries no per-row speaker records (the probability of "go-around", the blocked words), and the window view does not show them; (P38) the window set's `procedure` block is stage B's (prior §7 item 8); (P39) the loop draws at temperature 1 (prior D121) | Decided | User, 2026-10-05, on Claude's review of the readings |
| D129 | The window view (C11): the list names a window's runway as its recorded runway, since the round's sentence may say another; the shift of window A is shown in days or hours; the cursor starts at the window's row 0, so that the other aircraft show from the start | Decided | User, 2026-10-06, on the implementer's proposals P42–P44 |
| D130 | The updates of a round's pass take the branch groups of each groups file in an order shuffled by the round's random numbers (the seed and the round), then `update_groups` at a time; so an update mixes branch points and windows, and a resumed campaign is the same. Why: in the order spoken, an update's groups came mostly from one branch time of one batch | Decided | User, 2026-10-06, on Claude's check of stage C |
| D132 | The validation readout of a chosen round (C10, `post_validation`): it reads, for each airport, at most the campaign's select windows of each airport (`select_per_airport`) among the val days' real windows that do not open inside a loss, drawn as the selection readout draws the select days', with the same random numbers, and states each airport's coverage (real windows, left out, read). The one read of the val days (outline D85) is held for each campaign: its claim is in the campaign's directory. A formal campaign (`post_train`) and its validation readout refuse a base that is not stage B's formal base (a smoke prior or a fold): the formal start checks the base's run | Decided | User, 2026-10-06, on the implementer's readings P46, P47 and Claude's check (P48) |
| D137 | The settings of the formal campaign (C10; was O13), chosen by the user as proposed from C8's profile: 1,000 windows of each kind (real, A, D, B) a round; 10 rounds; 4 branch groups an update; 64 sentences of the data term an update; batches of 64 windows; K = 8; learning rates 1e-5 (the prior) and 1e-4 (the traffic modules); weight decay 0.01; 200 select windows an airport for the selection readout; the traffic attention 64 wide with 4 heads. The criterion that chooses the round (D7) is the user's, before the validation readout. Why: C8 measured the time and the memory at the formal size (log §25) | Decided | User, 2026-10-06 |

### 0.2 Open items

| # | Item | Proposal | § |
|---|---|---|---|
| O15 | The memory of N speaking workers is not checked before a campaign starts (outline §5 rule 13; D131: an S2 not corrected before C10). C10's round 0 is watched: a sampler records the GPU's peak; the main process releases its cache before each round's speaking (`ea2050ff`) | C13 checks it before the next campaign: one worker's host and GPU memory measured at the start, N workers refused by name where they do not fit | 8 |

### 0.3 Implementation

The status (outline §5 rule 10). Branch `dev-two-tier-v4-post`, worktree `.claude/worktrees/two-tier-v4-post`; the
commits, the tests and the readings are in the implementation log.

| Milestone | State |
|---|---|
| C0–C7, C9, C11 | Built and reviewed on synthetic artefacts; C1, C2 and C9 also on A34's artefact (log §2–§21) |
| C8 | Done on B5's base (log §24, §25): the formal census `outputs/POOLED/post/windows_20261006` (D104, `44fb8af8`), the base's go-around on the select days, the profiles at 32 and 256 windows (`483b81d7`, `c396f9be`); Claude's proposal of O13 in log §25, for the user |
| C10 | Running: `outputs/POOLED/post/post_train_20261006`, relaunched 2026-10-06 13:27 from the worktree at `e045d5c3` with the user's settings of O13 and three speaking workers (`0268e0ad`; log §25, §26). The criterion that chooses the round (D7) is the user's, before the validation readout |
| C13 (D138) | Built and reviewed by stage B's implementer: `65a21c57` on `dev-two-tier-v4`, O15's measure corrected (D139) `8cac5191`; its check on the GPU (one round, one process against N workers) after C10 ends. C10 gets none of it. Not merged: the user merges |
| The validation readout of the chosen round | Built and reviewed on synthetic artefacts (`post_validation`, `0c95a821`, log §23; D132 `483b81d7`) |
| C12 | After C10 |

### 0.4 Plan

1. C10 runs (D137; from `.claude/worktrees/two-tier-v4-post`, untouched until it ends); the user's criterion for the
   round (D7); the validation readout of the chosen round; then its speed (outline §6.2 item 10, `model_speed`), with no
   other job on the host or the GPU.
2. C13 (outline D138), by stage B's implementer after vocabulary A44, before the next campaign; C10 gets none of it.
3. C12.
4. The window view (C11) and the window export change with the one layout of the three stages' Training views (outline
   §6.2, D133–D135: the results page, the rounds' envelopes, one block of a flown sentence), built by stage B's
   implementer.
5. The generalisations of §9 marked "to be built" (multi-aircraft control D149, its MC0), each checked against stage
   C's code before the change, bit for bit; nothing of them goes into C10's worktree.

Every milestone from now on is built by stage B's implementer on `dev-two-tier-v4`, in the worktree
`.claude/worktrees/two-tier-v4` (outline §5 rule 1). `dev-two-tier-v4-post` is merged into `dev-two-tier`; it and its
worktree are deleted after C10 ends.

---

## 1 Scope

- **This document owns** the window loop, the reward, the branch training, the traffic attention, the edge features,
  the separation judge and the separation masks, their runners, and the Training view of stage C.
- **It reads** the outline; the vocabulary's public interface (vocabulary §6): the grammar, the sentence artefact and the
  stored signals of every flight of a split (item 3), the candidates, the executor and the start of a closed loop with a
  copy of a loop's flights and a moved start (item 5), the judge, the row grid and the Training export (item 8); and the
  prior's public interface (prior §7): the checkpoint, the inputs of a row (one function, also in a loop) and the
  landings, the speaker (with the input of the added modules, the caller's random numbers, the records of the permitted
  words and a copy of chosen aircraft), the teacher-forced loss with the sentences under a selection and the
  log-probability of given words under a record, the place for an added module, the region of a final, the step of a
  speaker's closed loop and the Training export's procedure block (items 1–8).
- **It gives** the post-trained checkpoints and the window readouts, and their Training view (outline §6); to the
  multi-aircraft control, its public interface (§9).

---

## 2 Post-training

1. **One stage, from the base model, with one aircraft commanded (D29).** The closed loop is a window of recorded
   traffic with one commanded aircraft; it lasts as long as the commanded aircraft flies (D93). The model speaks for
   that aircraft; the other aircraft fly their records. The start model is the base model with a traffic attention whose
   output is zero (§3; prior §7 item 5). There is no single-aircraft stage. The reasons:
   - The single-aircraft closed loop is a special case of this loop. With no other aircraft in its window, the
     commanded aircraft says and flies what single-aircraft free generation does, with the same loss and gradients.
     On the train days of A34's artefact, 60 % of the windows have no leader in the air at their first predicted step,
     and 29 % have no other aircraft at all (§6.1).
   - A single-aircraft reward does not see when an aircraft lands, so it does not limit the spread of the landing times.
     In traffic, that spread puts the aircraft into the sequence of the other aircraft (§6.1).
   - Without a capture law (D2, D3), the base model must learn more to land. It learns it one time, in the loop where
     it is used.
2. **Reward (D30).**

   | Outcome of the sentence (vocabulary §6 item 6) | Reward |
   |---|---|
   | `landed`, no go-around, on a runway of the airport's present landing direction | 1 |
   | `landed` after n go-arounds, on such a runway | 0.9ⁿ |
   | Every other outcome; every loss of separation (the separation judge, §3) | 0 |

   "The airport's present landing direction": a runway within 90° of a runway with a landing in the 30 min before the
   first predicted step, among the window's landings (D105); with no landing in those 30 min, every runway (D112).

   The reward gives a go-around that the DA check needs without a term of its own. An unstable approach that continues
   ends as `unstable_at_minimums`: 0. A go-around before the DA point continues the flight with 900 s more time
   (vocabulary §6 item 6); a landing then gives 0.9. Thus the model gains from a go-around only when the go-around
   changes a probable failure into a probable landing: with the same probability p of a landing before and after the
   go-around, 0.9·p < p. A go-around that prevents a loss of separation keeps the chance of 0.9, so the same reward also
   gives the go-around for traffic. The power n stops a chain of go-arounds that only adds time; after the second,
   "go-around" is masked (D91).

   **No payment for a go-around without a landing.** Such a payment f makes a go-around better than the continued
   approach when p < f / (f + 0.1), with the same p before and after the go-around (Claude's arithmetic). For example,
   f = 0.28 gives p < 0.74. "Go-around" is permitted at any row after the first predicted step (D10), and p is low on
   many flights when the training starts. The payment then teaches a go-around where the model is not sure, not where
   the approach is unstable. In this stage the other aircraft fly their records and do not react: a go-around cannot
   help them land. A stage with every aircraft of a window commanded (item 8) decides for itself if it pays for that
   help.
3. **Masks.** The masks of the prior: the grammar and the procedure masks, under the set that the prior was trained
   under (prior §7 item 3). The masks of a caller (prior §7 item 3): "go-around" after a flight's second go-around
   (D91), and the speed-word mask of §3.
4. **Windows.** Windows of the train days, of four kinds: real windows; A (one inserted aircraft that flies its record,
   D103); D (the aircraft ahead moved, D103); B (the commanded aircraft's start moved by a turn about the airport, a
   height change and a speed change, D123, through the moved start of vocabulary §6 item 5). Real, A and D in equal
   counts; B's count is a setting (D100, O13). The kinds share the round's flights (D124). A window that opens inside a
   loss of separation is left out of the draw (D113).
5. **Loss.** For each update (D124), three terms:
   - the clipped-ratio surrogate (ε = 0.2), word by word, with the advantage inside each branch group (item 9);
   - the pull to the base model (weight 0.04): the KL on the sampled words, under the same masked distribution;
   - the teacher-forced data term (weight 1) on single-aircraft samples of the closed-loop sentences of the train days
     in the base's selection `landed` (D36, D76); the flights outside it are starts like the others (D76).

   The masked distribution is the one that the speaker drew from: the training reads it from the speaker's records of
   the permitted words (prior §7 items 3 and 4). Every counted row weighs the same (D115); the groups of a file are taken in a shuffled order (D130). The surrogate and the pull
   with dropout off, the data term with the base's dropout (D107). The ratio's denominator is the model at the start of
   the pass (D117). The traffic attention has its own learning rate (O13). One pass over the samples of a round.

   **An update in pieces** (the user, 2026-10-06, after C10's round 4 ran out of the GPU's memory twice): the
   surrogate and the pull of an update are computed piece by piece, one branch group a piece, each summed over its
   counted words and divided by the counted rows of the whole update (D115), its backward pass taken before the next
   piece; then the data term in one piece, with its dropout; one optimizer step. The loss and its gradient are the
   whole update's to float rounding; the memory is one piece's (round 4's largest update: 6.3–6.5 GB whole, 1.58 GB in
   pieces). The whole update stays as the reference (outline D138). C10's rounds 0–3 ran the whole update.
6. **Go-around sampling first.** Before the training, measure the probability that the base model gives "go-around" on
   the final, on the select days, with the prior's free generation and its readout (prior D72) (with D26, the data has
   go-arounds with "no level-off" and "unspecified" in force). If the model says "go-around" by itself, the training
   uses no probes. If it does not, probes (a cross-entropy on a forced "go-around" at chosen steps) are discussed with
   the evidence of §6.3: a cross-entropy on a forced word teaches the word, not when to say it.
7. **Selection.** Each round's selection readout reads a fixed set of real windows of the select days, the same windows
   and random numbers every round. The user sets the criterion that chooses the round (D7). The validation days are read
   one time, by the validation readout of the chosen round (outline D85). The airport generalization of the design is
   chosen in stage B (D39) and tested at the end on KAUS.
8. **Later, optional.** A stage with every aircraft of a window commanded starts from the model of item 7: stage D,
   its own document `multi_control.md`.
9. **Branch training (D37, D94).** The samples of a round:
   1. Each training window (one commanded aircraft) is spoken one time: the first sentence, with the window's own
      random numbers.
   2. If the reward of the first sentence is 1, the window gives no sample: a group whose rewards are all the same
      gives no gradient.
   3. If the reward is less than 1, the event time t_E is the step where the first sentence ended: the step of the loss
      of separation that ended it, the step of its judged outcome, or the time limit. The branch points are the first
      predicted step and every 120 s after it, before t_E.
   4. A second pass speaks these windows again with the random numbers of their first sentences, and copies the state
      of each window K = 8 times at each branch point (D94). The other aircraft fly their records, so their states come
      from the time. The branch points are at the same times after every window's first predicted step, so the
      continuations of one branch point of all windows are at the same row and speak as one batch.
   5. A branch group is the first sentence and the K continuations of one branch point; they differ only after the
      branch point. The advantage is the reward minus the mean of the group. It applies only to the words after the
      branch point; a group whose rewards are all the same gives no sample.
   6. The executor flies without gradients while it speaks (inference mode); the states are the same as with gradients
      (a test). The training scores the prior only.

   **Why.**
   - Cost: with K full sentences for each window, the speaking took approximately 55 % of a round, and only 37–47 % of
     the sentences carried a gradient (§6.2). Branch training spends the extra sentences only where the first sentence
     failed, and a continuation flies only the part after its branch point.
   - Credit: a reward for a whole sentence gives every word the same advantage. In a branch group, only the words after
     the branch point differ, so the advantage goes to them (§6.2).
   - Branch points: the readout of multi-aircraft step 7.7 found that a new sentence from the start rescues the most
     events, then from 120 s before the event; 60 s and less rescue few (§6.2). Its reading, decided before the run,
     puts the branch points at the start and at fixed times, not some tens of seconds before the event.
   - A window whose first sentence lands gives no sample in this round. Its chance to fail comes again in a later
     round, as the windows are drawn again.

---

## 3 Multi-aircraft

- **Time.** A scene puts its aircraft on one UTC grid: a scene step is one instant for every aircraft. That is the
  layout of the data, not a model input, and D16 (prior) does not change it. The prior's time attention runs along the
  commanded aircraft's own rows; the traffic attention reads one scene step, which is one instant, and takes the other
  aircraft as a set, with no order and no position code. Edge features are relative (the approach clock is a distance;
  the closest-approach time is a time difference).
- **R and D23.** R exists at every step for every aircraft from the step after its first predicted step (D117). Before
  it, no input reads the aircraft's R (D23), the edge features included. At the rows up to the first predicted step of
  an aircraft, no input and no edge feature uses a value computed from that aircraft's R. A test: a change of one
  aircraft's runway word leaves all inputs and edge features of the scene at those rows the same, bit for bit (D31).
- **Recorded traffic (D93).** The other aircraft of a window come from the stored signals of the split (vocabulary §6
  item 3): every flight of the airport and the split, with a sentence or without one, in the air at the step, from its
  entry into the arrival slice to its landing. A stated limit of the scene: an aircraft outside the slice (before its
  entry, a departure, an overflight) is not in the data, and a flight of a day of another split is never read (the test
  days are sealed, C32), so near the cut between two operating days a scene can lack aircraft. The motion of a recorded
  aircraft is its displacement in the 2 s before the row, as the prior's motion inputs (prior D25, D60) — never the
  signals' track, ground speed or vertical rate, which are fits that use later rows. Its R is the runway of its record.
  Its G comes from its labelled sentence (D99).
- **Separation judge** (`inference/separation.py`, reading VISUAL; contract C36): the radar minima of 7110.65BB with
  the rules for visual approaches (7-4-4 c), never visual separation. Parallel runways at least 2,500 ft (762 m) apart
  are free once both aircraft are turned in (within 30° of the course, each on its own side of the midline); closer
  pairs count as one runway; established crossing finals are not judged. Its inputs: each aircraft's position relative
  to its R (vocabulary §6 item 4) and "established" (D92). It runs after each row that the executor flew. The event of a
  window is the first loss of separation for which the commanded aircraft answers.
- **Established on the final (D31, D92).** The separation judge (the in-trail rule: which aircraft is responsible) and
  the speed-word mask read whether an aircraft is established on the final of its R. It is a function of one row — the
  state of the aircraft at that row, its R and its G — the same for every aircraft (commanded or recorded). It does not
  come from the capture row, which uses later rows (vocabulary §6 item 3), and not from an executor state, because the
  executor has none (vocabulary §6 item 5). It is not an executor law and not an input. The rule: D92.
- **Spacing on the final (D92).** This design has no word "cleared", so no mask can hold a clearance back. This stage
  has no mask that keeps an aircraft from joining the final too close behind another: the reward (0 for a loss of
  separation) teaches the spacing. A mask is designed only if the readouts show that the losses at the join dominate.
- **Speed-word mask.** A mask of a caller on the commanded aircraft's speed column. It applies only when the aircraft
  and the aircraft next ahead on the approach clock (the same runway, or a pair that counts as one runway) are both
  established on their finals (D92), and while the aircraft is outside the limit of 7110.65BB 5-7-1 b4: "the final
  approach fix on final or a point 5 miles from the runway, whichever is closer to the runway" (5 NM = 9,260 m). With
  both aircraft established, so inside the FAF, that is the same as more than 9,260 m from the threshold (D101). Both
  aircraft are predicted to the time when the aircraft ahead crosses its threshold, each along its course toward its
  speed target at the rate of the executor (vocabulary §6 item 1); a recorded leader's target is its present speed
  along its course (D117). A speed word whose predicted gap is then less than the required distance is masked. When
  every word falls short, nothing is masked. "Unchanged" and "unspecified" are never masked. The mask is computed once
  for a row, from the state at the start of the row (D110).
- **Traffic attention (D98, D116).** At each layer of the prior, a module (prior §7 item 5) reads the other aircraft of
  the step: an attention from the commanded aircraft's row to one token for each other aircraft. A token is the other
  aircraft's edge features to the commanded aircraft and its own motion (D117), from its recorded state only. One token
  network, shared by the layers, embeds the tokens of a step once; each layer has its own attention over them. The
  module's output starts at zero, and it is zero when the step has no other aircraft, so that a window without traffic
  is free generation (§2 item 1). The speaker passes it the tokens of each row (prior §7 item 3). The start of the
  post-training is the base with this module (D29).
- **Edge features.** For each other aircraft at each step, what it is to the commanded aircraft: its place in the
  commanded aircraft's frame (ahead, to the left) and its height difference; its position on the approach clock less
  the commanded aircraft's, and whether the two can be compared; the relation of the two runways in force (same,
  single, dependent, independent, unrelated); the rate at which the horizontal distance closes; the closest point of
  approach on the present motions (its time, horizontal distance and height difference); the distance that the rules
  require between the two; a flag for an unknown motion. The scales are constants in SI units (D117). No feature uses a
  value of a row after the step. Their identity: §4 item 1.
- **Landing context (D31, D98, D105).** Only the commanded aircraft has inputs. Its landing context counts the landings
  of the window's scene before the step: the recorded landings of the recorded aircraft and of the aircraft before the
  window, with the changes of an augmented window (D105), never its own recorded landing (that landing is the future of
  the commanded aircraft) and never a landing on a sealed test day. The commanded aircraft's own landing in the loop
  ends its window, so it never counts it either.
- **States in a scene (D32, D36).** A scene occurs only in the closed loop. The commanded aircraft flies with the
  executor; the other aircraft fly their records. There is no teacher-forced scene sample: the flown states keep the
  observed path but not the observed time, so the spacing between two aircraft would not be the observed one.
- **Go-around in traffic (D92).** The word "go-around" is an explicit decision: it can be counted and rewarded. After
  it, the aircraft is not established until a runway word ends G; the separation judge treats it as any aircraft that is
  not established.

---

## 4 Gates and identities

The gates of this document are the post-training and the multi-aircraft stages. The user sets their criteria (D7). The
identities follow D21 (outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The edge features of a scene | A conformance check: fixed reference scenes give the same edge features (`post-edges-v1`; the reference `post-edges-reference-v1`, written by the formal census beside its output, D104). Every process that computes edge features runs the check first, every time (as vocabulary D73) |
| 2 | A window readout | The checks of the code that it uses, run at its start: the edge features' conformance, and the executor's and the closed loop's through the start of a closed loop (vocabulary D71, D73). The commit is recorded as information. No digest of code (D21) |
| 3 | A post-trained checkpoint | Its format name (`ts-post-checkpoint-v1`); the identity of the base checkpoint (prior §8 item 1); the set of procedure masks that it speaks under (prior §8 item 2); the shape of the traffic attention (`post-traffic-attention-v2`); the seed, and the windows of each round by their flights and their start times. The other settings (O13) are in the campaign's record; a resume requires them the same |

---

## 5 Values

| Item | Value | Source |
|---|---|---|
| Reward of a landing after n go-arounds | 0.9ⁿ | D30 |
| Most go-arounds of a flight in a loop | 2; after the second, "go-around" is masked | D91 |
| Window | One commanded aircraft, from its row 0 to its end; its other aircraft every flight of the airport and the split in the air at each step | D29, D93 |
| Windows of a round | Real : A : D = 1 : 1 : 1; B a setting | D100, O13 |
| Window A | A flight landing more than 3,600 s from the commanded flight's landing, shifted to land within ±180 s of it (whole Δ, uniform) | D103 |
| Window D | The aircraft next ahead moved by a whole Δ in [−120 s, +120 s], never 0 (uniform) | D103 |
| Window B | A turn within ±15°, a height change within ±300 m, a speed scale within 1 ± 0.1 (uniform) | D123 |
| Branch points | The first predicted step and every 120 s after it, before the event; K = 8 continuations, copied in a second pass with the first sentence's random numbers | D37, D94 |
| Established on the final | G false (a recorded aircraft's from its labelled sentence), inside the FAF and the LPV cone of R, track within 20° of the course of R | D92, D99 |
| A token of the traffic attention | The other aircraft's edge features to the commanded aircraft and its own motion, from its recorded state | D98, D117 |
| Token scales | 5,556 m (asinh) for horizontal distances; 1,000 m; 100 m/s; 10 m/s; 120 s | D117 |
| Loss | Clipped ratio ε = 0.2, word by word; pull to the base model 0.04; teacher-forced data term 1; every counted row weighs the same | §2 item 5, D115, D117 |
| Present landing direction | A runway within 90° of a runway with a landing in the 30 min before the first predicted step; every runway with no such landing | §2 item 2, D112 |
| Speed-word mask | Between 9,260 m from the threshold and the FAF, both aircraft established | §3, D101 |
| Temperature of the speaker | 1 | D125 |

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
  approximately 45 % of the flights had no leader in the air at their first predicted step (instruction-v3's
  definition, not the census's below; the two are not compared).

From C1's census on A34's artefact (`instruction_language/v12_20261005`, Δ = 4 s, train and select, in a scratch
directory; log §3, §13, §15):

- 40,530 train windows and 6,199 select windows. With a leader in the air (D117, P8): train 40.1 % (59.9 % none), by
  airport 25 % (KSMF) to 54 % (KSJC); select 39.3 %. With no other aircraft at all: train 29.4 %, select 28.5 %.
- Left out by D113: train 58 of 40,530 real windows, 2,082 of 40,530 A (5.1 %), 1,304 of 16,234 D (8.0 %), 107 of
  40,530 B (0.26 %); select 12 of 6,199 real, 259 A, 193 of 2,438 D, 15 B.

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

- The prior (Claude's check, 2026-10-05, on `dev-two-tier-v4-prior` at `649a3d74`): a prior of the shape of
  configuration A (d_model 192, 4 layers, 6 heads, random weights), synthetic sentences of 60 rows, encoded row by row
  on the CPU with one thread. The hidden state of one aircraft encoded alone and in batches of 2, 8 and 64 differs by at
  most 9.5e-7 to 1.2e-6: not the same bit for bit. Thus a speaker's probabilities depend in their last bits on the
  other aircraft of its batch. A word drawn with the same random number differs only when the number falls that close
  to a boundary between two words.
- The executor (stage A's A38, 2026-10-05, A34's artefact, train, Δ = 4 s, on the CPU): 200 flights flown alone and in
  chunks of 2,048 gave the same words, corrections and outcomes, and states that differ by at most 7.9e-10 m (65 of
  them; the others by at most 1.1e-12 m); A30's train read in parts differed from train read whole on 6–7 flights of
  about 40,500 by at most 9.1e-13 m. Vocabulary D97 (3) bounds this: the words, the end and the outcome are the same,
  the states within `STATE_BOUND_M` (1e-6 m). A rounding can in principle move a discrete step, so a word or an outcome
  can differ in a rare flight; a window that differs is counted (D94).

---

## 7 Code

**Packages and import rules** (`tests/test_architecture.py`, read by the names imported):

- `post/` — the scene and its steps, the edge features, "established", the separation judge's inputs and the event, the
  speed-word mask, the reward, the traffic attention, the branch groups and the loss, the window sets' files. It reads
  `instructions/` and, from `prior/`, only the names of prior §7's "Code" column; it does not import `autopilot/`.
- The window loop (`experiments/post_window_loop.py`) joins the speaker, the start of a closed loop and the scene, and
  runs the commanded aircraft through the prior's step of a speaker's closed loop (prior §7 item 7). It is shared by
  the runners of stage C, not a runner.
- The runners of stage C import from `autopilot/` only the names of vocabulary §6, from `prior/` only those of prior §7,
  from `experiments/` only the module of prior §7 item 7 and the Training export's names (vocabulary §6 item 8, prior §7
  item 8; D126), and `post_*` modules. After each merge of stage B, the test's list follows prior §7's "Code" column.
- The separation judge is `inference/separation.py`, with the rules of `inference/runway_schedule.py`.

**Formats.** `post-edges-v1`, `post-edges-reference-v1`, `post-traffic-attention-v2`, `ts-post-train-v1` (a
campaign), `ts-post-checkpoint-v1`, `post-windows-census-v1`, `aeroviz-training-window-index-v1` (the index
`index_post_v1.json`), `aeroviz-training-window-sample-v2`.

**Key code index.**

| What | Where |
|---|---|
| A recorded flight, its G, the aircraft at a step, a scene, the windows (real, A, D, B), the census | `post/scene.py` `Recorded`, `go_around_rows`, `AircraftAt`, `Scene`, `MovedScene`, `Window`, `StartMove`, `airport_scenes`, `real_windows`, `inserted_window`, `leader_moved_window`, `moved_start_window`, `next_ahead`, `near_day_cut`, `census` |
| The landings of a window (D105) | `post/landings.py` `window_landings` |
| An airport's separation rules; the approach clock | `post/runways.py` `airport_separation`, `approach_clock_m` |
| The tokens (edge features and own motion) and their conformance | `post/edges.py` `tokens`; `post/conformance.py` `write_edge_reference`, `require_conforming_edges` |
| "Established" (D92) | `post/established.py` `established` |
| The separation judge's inputs; the event; D113 | `post/traffic.py` `traffic`, `commanded_loss`, `loss_at_first_step`, `opens_inside_loss` |
| The speed-word mask | `post/speed_mask.py` `speed_check`, `along_course_speeds` |
| The reward | `post/reward.py` `reward`, `present_runways` |
| The faulty points (D114) | `post/fault_census.py` `reads_fault`, `window_faults`, `fault_census` |
| The traffic attention | `post/traffic_attention.py` `TrafficConfig`, `TrafficTokens`, `TrafficAttention`, `add_traffic_attention`, `parameter_groups` |
| Branch training's numbers, branch points and groups | `post/branches.py` `first_numbers`, `continuation_numbers`, `branch_points`, `Group`, `samples` |
| The loss | `post/loss.py` `surrogate`, `pull_to_base`, `data_term`, `update_loss`, `one_pass` |
| The window sets' files | `post/training_files.py` |
| The window loop | `experiments/post_window_loop.py` `WindowLoop`, `WindowResult`, `checked_edges`, `moved_commanded` |
| The two passes of a round | `experiments/post_branches.py` `branch_round` |
| The census (runner) | `experiments/post_windows.py` |
| The campaign (runner, C10) | `experiments/post_train.py` `draw_round`, `batches`, `speak_round`, `train_pass`, `selection_readout`, `run_campaign` |
| The profile (runner, C8) | `experiments/post_profile.py` |
| The Training export (runner, C11) | `experiments/post_training_export.py` |
| The live segment of a window | `aeroviz_backend/autopilot_segment/window.py` (`POST /autopilot/window-segment`) |
| The window view | `aeroviz-4d/src/data/trainingWindowSample.ts`, `trainingWindowAutopilot.ts`, `hooks/useTrainingWindowLayer.ts`, `components/training/TrainingWindowSession.tsx` |
| Replaced, archived by stage A: the clearance mask, the edge features, the window loop of instruction-v3 | `archive/two_tier_v3_2026_10/inference/separation_masks.py`, `inference/scene_edges.py`, `experiments/traffic_window.py`, `experiments/traffic_speaking.py` |

---

## 8 Milestones not done

C0–C7, C9 and C11 are done; their specifications are in the implementation log (§22). The rules of outline §5 apply:
from 2026-10-06 stage B's implementer builds every milestone of every stage (outline §5 rule 1); stage C's implementer
runs C10 to its end.

**C8. Profile and the go-around probability** (§2 item 6). After B5's base and Claude's check of stage B.

- One speaking batch at the formal size (`post_profile`): the time of the prior, the executor, the masks and the edge
  features; the memory of the host and of the GPU at the formal size (the user's rule of 2026-09-30: a smoke at a small
  size proves nothing about the formal size; outline §5 rule 13); the time of a round, the second pass of D94
  included; the bytes of the branch groups and the memory of the data term (D125).
- The base's probability of "go-around" on the final, on the select days, through the prior's free generation and its
  readout (prior D72). No probes unless the user decides them (§2 item 6).
- The report to the user, with Claude's proposal of the settings of O13. No criterion is applied (D7).

**C10. The post-training** (§2). After C8, O13 and the user's criteria (D7).

- The rounds as one campaign (`post_train`) from one commit on a clean checkout: in each round the windows drawn (D100,
  D113, D124), the two passes of D94, one pass of the loss (§2 item 5), the selection readout on the select days (§2
  item 7); resumable; the intent in `docs/experiments/intents.json` before the launch.
- The validation readout of the chosen round (`post_validation`, D132): the round's model on real windows of the validation days,
  read one time (outline D85; prior D119, D128: a claim before the read), with the readouts of the selection readout
  (the rewards, the outcomes, the losses of separation, the rows the speed-word mask acted, D114's counts). Its checks
  as §4 item 2. Tests: it refuses a second read; it reads no train or select window.
- A formal campaign and its validation readout refuse a base that is not stage B's formal base (D132).

**C13. The speed of a round** (outline D138; built and reviewed, `65a21c57`; the memory rule's correction and its GPU
run after C10 open), by stage B's implementer (outline §5 rule 1: one implementer for all
stages) on `dev-two-tier-v4`, after vocabulary A44, before the next campaign; C10 gets none of it.

- The selection readout's batches are spoken by the speaking workers (`Speakers`), each window with its own seeded
  numbers as now (`readout_numbers`); the main process sums the counts. Speaking in one process (no workers) stays a
  mode, the reference.
- The windows are started through vocabulary A44's `Start`, one for each split, opened in the campaign's process before
  the workers fork, so that they share its memory.
- O15: before a campaign starts, one worker's host and GPU memory is measured, and N workers are refused by name where
  they do not fit: N times one worker's peak, the main process's growth in one update (on the measured batch's groups)
  and the other workers' held memory must fit the host and the GPU (outline D139 (9)). A worker's peak leaves out the
  parent's pages that it shares (`SwapPss`, or its reading at its start taken off) and counts the reader model and the
  series it keeps over a round (Claude's review, findings 9 and 10).
- Tests: the selection readout through two workers equals the one-process readout on a synthetic campaign; a round
  started through `Start` equals one started through `start_moved` (the words, the rewards, the states).
- Size: about 80 lines of `post_train.py`, 30 of the window loop's start, 80 of tests. After it, `post_profile` at C10's
  settings measures a batch and a round again.

**C12. Close of stage C.** The full ts suite passes. The implementation log and `docs/reference/runners.md` are
updated; the report gives the code index for §7 (outline §5 rule 10). Report to the user: the commits, the rounds and
their readings. The user merges (outline §5 rule 11). A later stage with every aircraft of a window commanded is
optional (§2 item 8).

---

## 9 Public interface

What the multi-aircraft control (stage D, `multi_control.md`) reads of the post-training, and nothing else: its code
imports from `post/` and from stage C's runners only the names of the "Code" column (multi-aircraft control D149).
Where one commanded aircraft becomes several, stage C's code is generalised, not copied: each generalisation is
written here before it is built ("to be built"), keeps stage C's behaviour (with one commanded aircraft in each
window: the same words, states, rewards and branch groups, bit for bit, on fixed inputs, against the outputs of the
code before the change) and changes no format of this document, so C10's checkpoints and readouts stay valid. The
implementer's report gives the names that change, and Claude writes them here.

| # | Item | What it gives | Code |
|---|---|---|---|
| 1 | The scene and a window | The recorded flights of a split, an aircraft at a step, a scene; a window and its kinds (real, A, D, B), the census. To be built (multi-aircraft control D149, MC0): a window holds one or more commanded aircraft, each with its own shift (a whole number of Δ); stage C's windows hold one | `post/scene.py` `Recorded`, `AircraftAt`, `Scene`, `MovedScene`, `Window`, `StartMove`, `real_windows`, `inserted_window`, `leader_moved_window`, `moved_start_window`, `census` |
| 2 | The landings of a window | The roster's landings with the window's changes, less the commanded aircraft's own (D105). To be built (multi-aircraft control D149, MC0): less every commanded aircraft's own | `post/landings.py` `window_landings` |
| 3 | The separation judge's inputs and the losses | The aircraft of a step for the judge, "established" (D92), the separation rules of an airport, the approach clock; the loss that the commanded aircraft answers for; D113. To be built (multi-aircraft control D149, MC0): every loss of a step with its two aircraft and the ones the rules make responsible, so that a caller applies its own rule (stage D's D145); `commanded_loss` stays the loss of aircraft 0 | `post/traffic.py` `traffic`, `commanded_loss`, `loss_at_first_step`, `opens_inside_loss`, `joined`; `post/established.py` `established`; `post/runways.py` `airport_separation`, `approach_clock_m` |
| 4 | The tokens and their conformance | The edge features and the motion of each other aircraft (`post-edges-v1`) and their check (§4 item 1, D104) | `post/edges.py` `tokens`, `TOKEN_FEATURES`; `post/conformance.py` `require_conforming_edges`; `experiments/post_window_loop.py` `checked_edges` |
| 5 | The speed-word mask | A caller's mask on the speed column (§3, D110) | `post/speed_mask.py` `speed_check`, `along_course_speeds` |
| 6 | The reward | D30's reward of an aircraft; the present landing direction (D112) | `post/reward.py` `reward`, `present_runways` |
| 7 | The traffic attention | The module at each layer, its input, its learning rate (`post-traffic-attention-v2`). To be built (multi-aircraft control D149, MC0): a token part that a caller adds (its own token inputs beside the edge features, through its own projection that starts at zero, under its own format name); without it, today's module and tokens, bit for bit | `post/traffic_attention.py` `TrafficConfig`, `Traffic`, `traffic_of`, `TrafficTokens`, `TrafficAttention`, `add_traffic_attention`, `parameter_groups` |
| 8 | The window loop | The commanded aircraft of a batch of windows through the speaker's closed loop (prior §7 item 7), the scene at each step, the judge, the masks of a caller, the copy of a window at a branch point. To be built (multi-aircraft control D149, MC0): a window is one or more rows of the batch, each joining at its own tick (prior §7 item 7); an aircraft made silent (a caller's mask that permits only "unchanged" in every column); a window ends when every commanded aircraft is done or silent; each aircraft's result; a landing in the loop added to the other aircraft's landings; the copy copies every aircraft of the window | `experiments/post_window_loop.py` `WindowLoop`, `WindowResult`, `moved_commanded` |
| 9 | Branch training | The random numbers of a round, the branch points, the groups and their samples (D37, D94). To be built (multi-aircraft control D149, MC0): the numbers of each aircraft, the varied aircraft and their branch points, and a window's reward are rules that the caller gives (stage C's: D94's streams, its one aircraft with a reward below 1, its reward); a group names its varied aircraft | `post/branches.py` `first_numbers`, `continuation_numbers`, `branch_points`, `Group`, `samples`, `BRANCH_EVERY_S`, `CONTINUATIONS`; `experiments/post_branches.py` `branch_round` |
| 10 | The loss | The surrogate, the pull to the base and the data term of an update (§2 item 5, D107, D115) | `post/loss.py` `Samples`, `surrogate`, `pull_to_base`, `data_term`, `update_step` (an update in pieces, §2 item 5), `update_loss` (the whole update, the reference), `one_pass` |
| 11 | The campaign's steps | The draw of a round, the batches, the speaking workers, the pass, the selection readout, the checks at the start, the resume. To be built (multi-aircraft control D149, MC0): the round's skeleton given a stage's draw, window loop and readouts; stage C's campaign is that skeleton with its own | `experiments/post_train.py` `draw_round`, `batches`, `Speakers`, `speak_round`, `train_pass`, `selection_readout`, `open_context`, `run_campaign` |
| 12 | The checkpoint | A post-trained round (`ts-post-checkpoint-v1`) and its identity (§4 item 3). To be built (multi-aircraft control D149, MC0): one function that opens a chosen round as the start of a later stage, with its identity | `experiments/post_train.py` `POST_CHECKPOINT_SCHEMA` |
| 13 | The window sets | The files of a window set and its Training export (for stage D's Training view) | `post/training_files.py`; `experiments/post_training_export.py` |
