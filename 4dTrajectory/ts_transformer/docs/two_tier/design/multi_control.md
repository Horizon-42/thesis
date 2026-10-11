# Two-tier model: the multi-aircraft control

**Summary.** The multi-aircraft control trains the prior further in windows of recorded traffic in which every arrival
of a time span is commanded by the model: the model speaks for several aircraft at one time, each aircraft flies its
own words, and the aircraft must keep their separation from each other and from the recorded traffic. It is stage D of
the plan (outline §4 item 7), the optional later stage of the post-training. It starts from the round that stage C chose. Its first
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

**State of this document.** The decisions are in §0.1 and the open items in §0.2; stage D's implementer keeps the
state of the work in §0.3 and its log.

---

## 0 Status

### 0.1 Decisions

The decision numbers are shared by all documents (outline §3). A decision gives the rule and, after "Why", its reason;
the evidence is in §9.

| # | Item | State | Source |
|---|---|---|---|
| D140 | **The reward of an aircraft** r_i is post-training D30, unchanged, for each commanded aircraft: 1 for `landed` with no go-around, 0.9ⁿ after n go-arounds, on a runway of the airport's present landing direction (D147); 0 for every other outcome and when the aircraft answers for a loss of separation (the judge's responsibility, §2.4). Why: D30's reasons hold for each aircraft (§2.1) | Decided | User, 2026-10-06, on Claude's proposal |
| D141 | **The reward of a window** W is the sum of r_i over the window's commanded aircraft. A recorded aircraft adds nothing. The sum is not divided by the number of aircraft. Why: one controller speaks for every aircraft, and its task is that all of them land safely; a landing is worth the same in a window of one aircraft and in a window of six (§2.2) | Decided | User, 2026-10-06, on Claude's proposal |
| D142 | **Credit by one varied aircraft.** A branch group (post-training D37, D94) varies one commanded aircraft v: from the branch point on, the K continuations draw new random numbers for v only; every other commanded aircraft draws the numbers of the first sentence (each keeps its own stream). The advantage of each sentence of the group is its W less the mean of the group's W, and it applies only to v's rows after the branch point, up to v's event (its end, or the row from which it is silent, D144). The other aircraft's rows of the group carry no advantage. An aircraft's random numbers are the caller's: stage D gives each commanded aircraft a stream of its own (§3.3); stage C keeps D94's (D149). Why: inside a group, only v's draws differ, so a change of W is caused by v's words — directly or through the other aircraft's answers to v. A window's reward can then be the team's sum without the noise of the other aircraft's own draws (§2.3). Kept (the user, 2026-10-09) against a cheaper form in which the other aircraft replay their first sentence's words and states in a continuation (only v spoken and flown): their answers to v stay part of v's credit | Decided | User, 2026-10-06, on Claude's proposal; kept 2026-10-09 |
| D143 | **Which aircraft are varied, and where.** A window whose W is less than its count of commanded aircraft is spoken again (post-training D94's second pass). The varied aircraft are each commanded aircraft with r_i < 1 and each commanded aircraft of a loss of separation that one of them answers for. The branch points of a varied aircraft v are v's own first predicted step and the points of the window's grid after it (the anchor's first predicted step and every 120 s after it), before v's event: the step of v's own end, or of the loss it is in, whichever is earlier. K = 8 (D94). Why the window's grid: the continuations of one point of all windows of a batch are at one row of the loop and speak as one batch, as in stage C | Decided | User, 2026-10-06, on Claude's proposal |
| D144 | **An aircraft that answers for a loss of separation becomes silent.** Its reward is 0 from that step. It is not removed: it flies on its words in force until the executor is done with it, the other aircraft must keep their separation from it, and it never answers for a loss again. It is flown through the loop with a caller's mask that permits only "unchanged" in every column (prior §7 item 3), so its rows are said with probability 1 and carry no gradient. **A window ends when every commanded aircraft is done or silent**: an aircraft that has not joined yet, or is in its observed rows, keeps the window going. A silent aircraft's rows are not counted rows of any sample (post-training D115). Why: an aircraft that disappears gives the others free space, so a loss would pay the others; a window of one commanded aircraft then ends at its loss, as post-training D93 says (§2.4) | Decided | User, 2026-10-06, on Claude's proposal; the aircraft that stays in the scene: the user's rule of 2026-09-30 for the archived design (§9.1), proposed again |
| D145 | **A loss with a recorded aircraft.** A commanded aircraft answers for a loss with a recorded aircraft where the rules make it responsible, as in stage C; and also where the rules make only the recorded aircraft responsible but the two records (moved, in a compressed window) kept their separation at that step: the loss is then made by the model's change, and the commanded aircraft is the only one that can be moved. A loss that only the recorded aircraft answers for and that the records also have at that step is counted in the readouts and earns nothing and costs nothing. Why: the model is the controller of its commanded aircraft (the user: a loss that the model makes and does not answer for "does not fit simultaneous command"); a recorded aircraft does not react; and the records lose separation by themselves (2.76 % of the flights of the archived census, §9.1), which the model must not pay for | Decided | User, 2026-10-06 (O17's alternative; O17 closed) |
| D146 | **The window.** A window is an anchor flight and every flight of the same airport and split with a closed-loop sentence at the chosen Δ whose row 0 is in [the anchor's row 0, the anchor's row 0 + L): these are its commanded aircraft. Every other flight of the airport and the split in the air at a step is recorded and flies its record (post-training D93). L is a setting of the training data only: the model reads no L, and a window ends when every commanded aircraft is done or silent, so any span can be flown. A campaign mixes the spans L = 5, 10 and 20 min (the user, 2026-10-07, after MC1's census, O16): each kind's 1,000 windows of a round in equal parts of the spans, the span of each window drawn with the round's numbers, a shortfall of a span recorded; the windows of one batch have one L, so that a batch's rows and ticks are alike and its speaking stays one call of the network a tick, and the batches of each L are spoken by the speaking workers as now (nothing serial beyond today's). With L = 0 the window is stage C's real window (not one of stage D's spans). Kinds: real windows and compressed windows (each commanded aircraft other than the anchor moved toward the anchor by a whole number of Δ: its offset from the anchor times c, c drawn uniformly in [c_min, 1], c_min = 0.8). The windows of a round share their flights as in post-training D124: a batch commands each flight once, so windows that share a commanded flight go to different batches. A window is left out of the draw when a commanded aircraft opens inside a loss of separation on the records at its first predicted step (post-training D113, for each commanded aircraft) | Decided | User, 2026-10-06, on Claude's proposal; the spans, the kinds, their counts and c_min: user, 2026-10-07 (O16, after MC1), the rest of the mix on Claude's readings |
| D147 | **The landings that a window counts.** Two quantities count landings: the input of the landings on each candidate in the 30 min before a row (prior §7 item 2) and the present landing direction of the reward (post-training D112). In a window: (1) a commanded aircraft's landing in the data is not counted (it is that aircraft's future); (2) a commanded aircraft whose outcome in the loop is `landed` is counted for every other aircraft of its window, on its landed runway, from its crossing time; (3) a commanded aircraft with any other outcome is not counted; (4) a recorded aircraft is counted at its landing time in the data, as in stage C (post-training D105). Why: the inputs give only what a controller knows before the step (principle 7); with one commanded aircraft this is post-training D31 and D105 | Decided | User, 2026-10-06, on Claude's proposal |
| D148 | **One row at a time for all aircraft.** At a Δ row, every commanded aircraft is spoken at once, from the state at the start of the row; no aircraft reads the words that another one says in the same row. The masks of a caller are computed once for the row from that state (post-training D110). The speed-word mask takes a commanded leader as it takes a recorded one: its speed target is its present speed along its course (post-training D117 P5). Why: the archived design ordered the aircraft of a step along the approach clock so that a later aircraft's mask read the earlier one's new words (§9.1); that needed an order, several passes of the speaker in each step and a rule for ties, for a mask that acted on 0.18 % of the labelled speed words | Decided | User, 2026-10-06, on Claude's proposal |
| D149 | **Stage D's code depends on stage C's code, through a public interface.** The post-training document gets a public interface (its §9; §6.1 here); stage D imports from `post/` and from stage C's runners only its names. Where one commanded aircraft becomes several (the window, the window loop, the loss of separation, the two passes of a round, the campaign's steps), stage C's code is generalised, not copied. A generalisation keeps stage C's behaviour: with one commanded aircraft in each window, the same words, states, rewards and branch groups, bit for bit, on fixed inputs (the CPU, one thread). The reference is stage C's code before the change: its outputs on fixed windows are kept and compared with the generalised code's, by a test on synthetic windows and by a check on real windows, before stage D uses it. Stage C's formal results stay valid | Decided | User, 2026-10-06, on Claude's proposal |
| D150 | **The aircraft of one closed loop join it at their own Δ rows, in one batch** (§6.3). A loop has a row clock (its ticks, one Δ each); each aircraft has a join tick j: before it the aircraft is absent (its rows not present, not spoken, not flown, its executor waiting), from it the aircraft's own row 0, then its observed rows to its own first predicted step, then its said rows. Built in stages A's and B's code (the start and `Loop`; `LoopRows`, the speaker, `SpeakingLoop`), each written first in its stage's public interface (vocabulary §6 item 5, prior §7 items 2, 3, 7). The prior's network does not change: its time attention already reads only present rows (`Past.present`) and time differences (RoPE on each aircraft's own seconds). With every join tick 0 the loop is today's loop, bit for bit (a test); free generation and stage C call it so. Also: a landing added to chosen aircraft's landings while the loop runs (D147). Why: the commanded aircraft of a window enter at different times; one loop for each entry row would speak a row in up to L / Δ small batches (150 at L = 10 min), each with its fixed cost in Python, and the speaking is bound by the CPU (outline D138) | Decided | User, 2026-10-06 (one batch with each aircraft's own first row, Claude's option 1); the design of §6.3: Claude |
| D151 | **What the reward does not have.** No term for the time an aircraft takes (the judge's time limit bounds it; vocabulary §6 item 6); no shaping by the margin of separation; no payment for a go-around of its own; no probes (forced words). Why: each of them was built or measured in the archived design or in stage C and gave the model a target that is not the outcome (§2.5, §9) | Decided | User, 2026-10-06, on Claude's proposal |
| D152 | **The model, version 1: the states and the words in force of the other aircraft.** Stage D commands every arrival of a window's time span (§3). It starts from the round of stage C that the user chooses (post-training §9 item 12) and pulls to the base, as every stage of the post-training does. The prior runs on each commanded aircraft; the traffic attention reads every other aircraft of the step as a token (post-training D98): a recorded aircraft's from its record, a commanded one's from its states in the loop. The token of another commanded aircraft also carries what the model told it: a flag that it is commanded, a flag that it is silent (D144), and its words in force in the columns altitude, angle, speed and heading (none before its first said row). A recorded aircraft has the flags 0 and no words: its labelled words use later rows and are never an input (post-training D99). These features enter through a token part of stage D's own, beside stage C's token network, whose projection starts at zero, so the start says what stage C's chosen round says; stage C's token features and module stay as they are (post-training §9 item 7). Why: one model speaks for every aircraft at once and, for each one, knows where every other aircraft is and what it told each of them: a controller's knowledge. A word that one aircraft is told is read by the others one row (Δ) later (D148) | Decided | User, 2026-10-06 (the states and the words in force, Claude's option; the other aircraft's intent next, D153) |
| D153 | **The model, version 2: the other aircraft's intent.** After version 1's campaign, a module of stage D at each layer lets a commanded aircraft's row read the hidden states, at that layer, of the other commanded aircraft of its window at the same tick (the aircraft that have joined; an absent one never). Only the same tick: a hidden state at tick t comes from the inputs at the start of row t and holds no word of that row, so D148 holds. Its output starts at zero; version 2 starts from version 1's chosen round and pulls to the base. A sample of its loss is a whole window: every commanded aircraft's rows on the window's ticks, encoded together; only the varied aircraft's rows after the branch point, up to its event, are counted (D142). What it changes in the earlier stages' code, and why no trained artefact becomes invalid: §6.4. Why: version 1 knows what each aircraft was told, not what the model is about to tell it; the user wants that too, trained on top of version 1 | Decided | User, 2026-10-06 ("first train version 1, then add version 2 to it") |
| D164 | **Stage D's start follows stage C's rule of a start from a round** (post-training D162). Stage D's campaign starts from the round of stage C that the user chooses through the one function of post-training §9 item 12: its settings record the start as `{campaign, round, checkpoint_sha256}`; the start is refused by name unless the bytes are the recorded ones, the identity is of this base, today's masks, stage C's traffic shape and that round, a formal campaign starts from a formal one, and stage D's seed is not the source campaign's (stage D's draw takes the train's flights in a permutation of the seed and the round, D124 generalised, so the same seed would give its round r the flights of stage C's round r first); the optimizer starts afresh; the pull term pulls toward the base (D152). Version 2's start from version 1's chosen round follows the same rule. Why: one way to open a round as a start, with the same checks, for every stage | Decided | User, 2026-10-07 |
| D166 | **Stage D's implementer's readings of MC0–MC4, accepted** (`requests_from_d_to_designer.md`, 2026-10-07; the numbers are that note's). (6) A loop landing on a sealed test day is left out and counted with the sealed landings, as stage C's window landings. (7) A commanded aircraft that ends `landed` is judged once, at the row after it, as over its threshold at its crossing state, so the wake minimum behind a commanded leader is judged. (19) A formal start of stage D refuses a smoke campaign and a campaign with rounds still to run (as `post_validation`). (21) The token part's ten features (`multi/tokens.py`, `multi-commanded-tokens-v1`): the flags commanded, silent and words in force; the heading word as the prior reads it (sine and cosine of its track less the course in force); the level over 1,000 m with a flag for no level-off; the angle word's nominal angle over 3°; the speed target over 100 m/s with a flag for unspecified; a recorded aircraft's part 0; the scales the edge features' (a mirror, pinned by a test). (22) With one commanded aircraft, stage D's rules give stage C's round only where no `records_kept` loss occurs (D145); D149 holds for stage C's own code. (25) The silent flag carries at most one bit of the records (a `records_kept` loss) once an aircraft: a stated limit of D144 × D145 × D152. (29) `Stage.start` and `Stage.start_model`; `Stage.read_batch` takes the draw (post-training D161). (30) The context records whether the campaign is formal (`Context.formal`); `round_start` refuses a smoke source under it, and `start_of` checks the source before anything is opened. (31) A landed leader past its plane is not established; at its threshold row only the wake minimum is judged (its state up to 2 s behind, up to about 140 m, 2.5 % of the least minimum): a stated limit. (32) A compressed window whose shifts all round to 0 is counted as real; a window without a later aircraft has no compressed form. (33) The select set: real windows of each span (§5 item 2). (34) The readout's numbers `[seed, 2^30, place, member]`, a later draw appended; the readouts of O18 read each aircraft's crossing (`WindowResult.crossing`). (35) A landed commanded aircraft counts as recorded in the readout's pairs; the census's baseline never judges one (a stated limit, below 0.1 % of steps). The names they give are in vocabulary §6 item 5, prior §7 items 2, 3, 7 and post-training §9 | Decided | User, 2026-10-07 ((6), (7), (21), (25) one by one; (19), (22), (29)–(35) together) |
| D172 | **Stage D's speaking workers are sized from its profile** (MC5; stage D's items 46–48). A campaign of stage D sizes its speaking workers from its profile (`multi_train.profiled_fit`, `--profile`): the profile's measure of one worker and of the pass, and the memory free now; refused by name for a profile of other inputs or of other settings that set the memory (`PROFILED_SETTINGS`: the counts, the spans, c_min, the batch rows, K, the update groups, the data sentences); each worker's GPU capped at its share. Nothing is measured before a run. One batch size a span (`MultiSettings.batch_rows`, a number of rows for each span): a batch's GPU memory grows with its ticks as well as its rows. The profile speaks each span's batch once, in the worker whose memory it measures (in its own process only without workers). Stage C's campaign keeps its own measure (post-training O15, D167). Why: a measure at the formal size at every launch took many minutes; the profile is measured once and read in seconds | Decided | User, 2026-10-07 |
| D179 | **MC5 measures the memory only** (stage D's items 49 and 50; the user, 2026-10-08: a whole round spoken before the campaign "这个行为很愚蠢"). The profile (`multi_profile`) measures, at the formal size, one worker on each span's first batch and on its batch of the most rows, and the pass; `profile.json` records the peaks of the host and of the GPU and each measured batch's time; it speaks no round, reads no readout and measures no spread. The fit (`profiled_fit`, D172) multiplies each measured peak by 1.3 before it counts the workers, and the record states the factor (a batch that was not measured needed about 30 % more than its span's measured peak: three workers judged to fit ran out of the GPU's memory); since D191's peaks last, the factor sizes the host's budget and the pass's only, and a GPU worker counts at what it holds between batches (no measure of a few batches bounds a round's batches; one past its share is spoken again later); stage C's measure (post-training D167) is unchanged. The size of the select set (`select_per_airport`, §5 item 2) is a setting fixed before the campaign, not measured from a round's spread. A round's times and the readout's noise come from the campaign's own round 0 (`round.json`). A later profile (MC9) follows the same rule: measure only what must be known before the first round | Decided | User, 2026-10-08 (the measure only; the margin 1.3, on Claude's proposal); the GPU workers counted between batches: the user, 2026-10-10 (D191) |
| D180 | **Stage D's speaking workers on a device of their own** (stage D's items 52 and 53; the user, 2026-10-08). `multi_train --speak-device` and `multi_profile --speak-device`, as stage C's campaign (post-training §9 item 11): by default the campaign's `--device` (the behaviour before it); refused with one worker, and `cuda` beside a campaign on the CPU; workers on the CPU beside the pass on the GPU. The profile measures on the workers' device and records it; `profiled_fit` refuses a measure of another device kind, and counts a worker on the CPU against the host's memory and none of the GPU. Before MC6, a measure on each device (minutes) chooses the workers by the two speeds and the batch rows of each span (D172, a setting). Why: speaking is a loop of small forward passes and the executor's rows; which device speaks a stage's batches faster, and how many workers each holds, is measured, not assumed | Decided | User, 2026-10-08 |
| D181 | **The speed of a round: engineering that keeps the rules** (MC11; stage D's items 54–62, the user's request of 2026-10-09; the evaluation left to Claude). Nothing of D141–D146 changes. (54) The memory measure takes each span's first batch and the one batch of the most rows over all spans (`post_train.measured_batches`): accepted as built. (55) `select_per_airport` leaves `PROFILED_SETTINGS`: the measure does not read the select set (D179). (56) A speed runner, `experiments/multi_speed.py`, speaks one batch of a campaign's draw with timers at its parts — the first pass, the second pass, the continuations, the speaker's steps and the executor's (between two synchronisations, as `model_speed`) — and the GPU's peak reset at each part; a profiler only on the CPU, only where the timers leave a part unexplained (on the GPU it nearly doubles a batch and counts a kernel's time where the host waits); it runs with no other job on the host or the GPU; `multi_profile` stays the memory measure (D179). (58) A copy's samples are built for the rows read only (the varied aircraft's, `WindowLoop.samples(split, rows)`); (59) a row's scene is computed once a tick; (60) the campaign's process freezes its objects (`gc.freeze()`) before the speaking workers fork. Items 58–60 change no number: before use they are checked bit for bit on fixed inputs (the CPU, one thread) against the code before them, on stage C's windows and stage D's (D149); a campaign may then resume on them at a round's boundary (post-training D157; root `CLAUDE.md`: a behaviour check on fixed inputs). (57) Only after item 56 shows that the continuations set a worker's peak: a campaign setting, `continuation_rows`, at most that many rows of copies flown at a time, each piece whole windows (default: all at once, the behaviour before it); a piece's batch shape can change a word drawn near a boundary (post-training §6.4), so its check on fixed inputs requires the same words and outcomes but for such draws, counted, and the states within `STATE_BOUND_M`, as D94's second pass. Items 61 and 62: O20. Why: a round of stage D speaks for hours with the workers that the GPU holds (stage D's log §15), and a window's cost grows about with the square of its aircraft, since every continuation flies the whole window (D141); the gains must come from each continuation's work and from the workers' memory. Stage D's readings of MC11, accepted as built: (63) the records and sentences of chosen aircraft (`Speaker.permitted(indices)`, `SpeakingLoop.sentences(split, flights)`), padded to the most rows said among them, so that a sample's shape does not depend on the rows asked; (64) the objects are frozen only while the workers live (unfrozen at `Speakers.close`); (65) `multi_speed`'s parts are entered by the window loop's methods, the speaker's time net of the executor's steps it calls | Decided | User, 2026-10-09 (the request; the evaluation and the readings: Claude) |
| D182 | **Branch points found backward from each event** (MC12; the user's proposal of 2026-10-09, stage D's item 66; the form on the user's choices of 2026-10-09). A campaign setting, `MultiSettings.branching`: `grid` (D143's points; the default, the behaviour before it) or `backward`. With `backward`, the varied aircraft are D143's and each varied aircraft v is searched on its own (D142: one varied aircraft a group): (1) v's event is D143's (the step of v's own end or of the loss that v is in, the earlier), and for a landing after a go-around v's first go-around; (2) candidate points go back from the event in steps of 32 s on the window's grid of Δ rows, the latest first, at most 8 points and never before v's first predicted step; (3) at a point, K = 8 continuations (D142's streams; the numbers `[seed, round, place, member, tick, k]`, D166 (26)) are flown to 60 s after the event; a continuation **avoids** the event when, to that time, v answers for no loss and the event does not happen: for a loss, the pair of the loss keeps its separation; for a go-around, v says none; for another end, v's same outcome does not happen (a continuation that trades the event for a loss that v answers for does not avoid it); (4) the search stops at the first point (going back) where at least one continuation avoids the event — the decision point; its K continuations are then flown to the window's end and, with the first sentence, make a group whose advantage is D142's (W of D141 less the group's mean); (5) one point 16 s after the decision point is tried the same way and gives a group when one of its continuations avoids the event; (6) the search goes on back, within the same bound, to the first point at which a continuation avoids the event with no go-around said by v from the point to the event + 60 s (point B): its K continuations flown to the window's end make a group too; when one of the decision point's avoiding continuations has no go-around, B is the decision point and gives one group; (7) any other point tried gives no group (its continuations, flown only to the event + 60 s, give no W); an event that no point within the bound avoids gives no group. The loop's state at each candidate point is copied on the second pass (D94) and held in the host's memory until its point is tried; the device holds the current points of a batch's searches, stepped together (D188). Every group records the depth of its point (the seconds from it to the event). Stage D's readings of MC12, accepted as built (its item 77): (a) for a landing after a go-around the event is its first go-around when that is earlier than D143's event (a loss before the go-around stays the event); (c) the first point is 32 s before the event, never the event's own tick; (d) the 16 s point's state is the decision point's held copy flown 16 s on every aircraft's first-sentence numbers; (e) one point of one window at a time, replaced by D188 (1) after MC12's measure of time; (f) a window whose second pass differed (D94) is not searched; (g) a backward batch's record adds each group's span and depth (`group_depths_s`), the searches and those that found a decision point, and the continuations flown to the event + 60 s only and to the end, a grid batch's record unchanged; (h) the readout's go-arounds per aircraft are its `go_arounds` / `aircraft` of each span; (i) `branching` is a profiled setting (a profile without it reads as the grid); (j) an interval Δ at which 60 s is no whole number of ticks is refused at the first search. Built as `experiments/multi_train.py` `MultiSettings.branching` (`--branching`), `multi/backward.py` (`GRID`, `BACKWARD`, `event`, `points`, `avoids`, `search`), `experiments/post_branches.py` `backward_round`; the loop's state copied to the host through the copy chain's `device` (§6). **A stated limit:** the credit goes to the latest words that change the event (the decision point) and to the latest that change it without a go-around (B, where D30 pays 1 against a go-around's 0.9), so earlier planning gets credit only within 256 s of an event; the readouts give each round's go-arounds per aircraft and the depth of the groups' points, by span. Not taken: an extra point at v's first predicted step (it would fly the whole window again, as D143 does; the user, 2026-10-09); a reward of the separation's margin (D151); a continuation ended at a repeated loss in a group that is kept (W needs every aircraft's end, D141). Why: with D143's fixed points a group whose K continuations end alike gives no sample but is flown to the window's end (O21); searching back from the event puts the groups where the event can still change, flies the continuations of the points not chosen only to the event + 60 s, and keeps D141's W and D142's streams | Decided | User, 2026-10-09 (the backward search, coarse then fine, the stop at the first avoiding point; the rewards of D141; every failure searched; 8 points and 60 s; no early point; point B on Claude's proposal; (3) no loss answered for, for every kind of event, and the readings of stage D's item 77, on Claude's review of `de0711aa`) |
| D183 | **The speaker's cache holds the rows written and one step** (MC11; stage D's items 67 and 69, written at the user's word, 2026-10-09; the evaluation: Claude). Three places where the speaker's cache (`prior/model.py` `Past`, `prior/speaker.py` `Speaker.copy`) holds room that no row reads: (1) a copy indexes the cache by a tensor, which already makes the copy's own storage, and clones it again — the clone goes; (2) a copy takes the whole room of the loop it is copied from — it keeps the rows written and one step of room, in one storage of that size made once, into which the rows chosen are indexed (no storage of the rows in between, which a copy then grown would make); (3) a line that outgrows its room doubles it — it grows by one step of 64 rows (256 s at Δ = 4 s); (4) (stage D's item 69) a branch point's copies and their records stay bound in `branch_round` (`experiments/post_branches.py`) after its groups are taken, so the next point's copies are built while the old ones live — they are released once the point's groups are taken (the groups hold only arrays on the host), a change of lifetime only, checked by a test that the old copies are collected, with no check bit for bit. The attention reads only the rows written, so no number changes: items (1)–(3) checked bit for bit on fixed inputs (the CPU, one thread) as D181's items 58–60, with the prior's behaviour check (prior D108); on the GPU a cache's room changes its strides and can choose another kernel, a rounding of the class of post-training §6.4. Stage B's code, its interface (prior §7 items 1 and 3) unchanged, built by stage D's implementer (outline §5 rule 1); a campaign may resume on it at a round's boundary. Item 56's measure before and after gives the gain (the peak reserved, which a cache grown in many small steps can fragment). Why: the copies' caches are most of a worker's GPU peak (Claude's estimate, item 56 to confirm), and with D182 a continuation of a point not chosen flies only to the event + 60 s, so a copy that holds the room to the window's end holds mostly nothing; this removes that waste without changing the attention, before the shared prefix of O20 (62) is weighed | Decided | User, 2026-10-09 (the request; the evaluation: Claude; (2)'s one storage: the user, 2026-10-09, on Claude's review of `317979f9`) |
| D188 | **A backward batch's searches step together; the speaking workers' allocator expands its segments** (MC12, MC11; stage D's items 78 and 79, from the backward measure of stage D's log §18; the evaluation: Claude, 2026-10-09). (1) In a backward batch (D182) the searches of all its windows and varied aircraft go step by step together: each search's current point is copied into its K continuations, and the continuations of every search are flown in one loop, each to its own judged tick (the event + 60 s); a point that a search chooses is flown on to its window's end in the same loop, beside the others then running. Each continuation keeps its own numbers (D166 (26)); the search's rules are D182's. One point of one window at a time (D182's reading (e)) is deleted (outline D187), and the device holds the current points of the batch's searches. The batch of the model and of the executor changes, so the check before use is D181 (57)'s, not bit for bit: on fixed inputs (the CPU, one thread) the searches' decision points, 16 s points, points B, groups and depths are those of the code before, but for draws near a boundary, counted, and the states are within `STATE_BOUND_M`; grid is not touched and stays bit for bit. The new form is written to be read (outline D187): how a step's batch is made, when a continuation stops, how a chosen point goes on. (2) The speaking workers set the CUDA allocator's `expandable_segments` before their first CUDA call; the pass's process is unchanged. A setting of the process: no number changes, checked on one GPU batch spoken with and without it (the same record and group files). Then a new memory measure (D179) at the settings of stage D's log §18 sets the workers. Why: in a backward batch the search took 64–79 % of the time and a search step cost about 21 ms above its rows, so the number of steps, not their rows, set the time; on the 1200 s batch the reserved peak was 2.2 times the allocated one after D183's steps of 64 rows, so the allocator's fragments, not the cache, set a worker's GPU share (3 workers fit, 4 did not; log §18) | Decided | User, 2026-10-09 (the evaluation left to Claude; (1)'s check by D181 (57)'s bound: Claude's reading) |
| D189 | **A step's own work cut where it is computed again for nothing** (MC11; the user, 2026-10-09: no measure first). Each a fast form that replaces the code before it after one check on fixed inputs before its commit, bit for bit, and written to be read (outline D187): (0) a commanded aircraft's landings (post-training §3, D105) hold only what its rows can count — the landings from 30 min before the window's first row to the window's end, and its own — so that a landing added in the loop (D186 (1)) copies tens of landings, not the airport's roster; every count stays the same, since a row counts only the 30 min before it; (1) each step, each window's per-aircraft values are computed once for all its aircraft, commanded and recorded — the motion, the scene's fields (the approach clock, the track less the course, the distance right of the centreline, established) and the speeds along the course — and each commanded aircraft's own view (itself first, then the recorded aircraft, then the other commanded ones, the order of today) is that computation reordered, not computed again; the judged set after a landing computes the landed aircraft only; the traffic tokens read the motions and the approach clock from it, and a pair's runway relation and required distance from a table by runway pair and wake category pair; the tokens' other features stay one call for each commanded aircraft, over all its others, so that every token keeps its order and its bits; (2) the speaker brings back from the device, for each column, only the word drawn, and the row's records (the probability forbidden, the words permitted, the probability drawn) once a row; the masks of the later columns other than the grammar's (the procedure's and the caller's), fixed once the runway word is drawn, are computed once a row and read again for the records; (3) then the tokens of all pairs of a window in one computation: a product that changes its kernel can change a last bit, and a difference found on fixed inputs goes to the user before use, as D186 (3). Why: each commanded aircraft held its airport's whole roster of landings, copied whole for each landing added in the loop; in stage D every aircraft of a window is commanded, and the scene, the speeds along the course and the tokens' motions and clocks were computed again for each commanded aircraft (n + 1 scenes a window a step for n commanded aircraft), though each value depends on its aircraft only; the speaker brought back about four results a column, twenty a row | Decided | User, 2026-10-09 (cut at once, no measure; the form: Claude) |
| D190 | **The joint loop's memory: a rebuild's room, and the K copies of a point hold their front once** (MC11; stage D's item 80, the memory by part of stage D's log §19; the evaluation: Claude, 2026-10-09; (2) and its bound the user's choice). After D188 and D189 a batch took about half its time, but a worker's GPU peak rose from 1.6 to 2.21 GiB and 2 workers fit where 3 did, so a round's speaking took about as long as before: the joint loop's cache is most of a worker's memory, and of it the rows written but not present (a line aligned to the joint loop's clock, an aircraft not yet joined) 37–60 %, the history 26–42 %, a step's room 14–26 %. (1) After a rebuild of the joint loop its room is the steps to its next rebuild, not D183's 64 rows (a cache that outlives it still grows by D183's step): no number changes, checked bit for bit (outline D187). (2) The K copies of one point, the same bit for bit up to where they part (their history and the padding before it), hold that front once, and each continuation's attention reads the shared front and its own rows after the point (O20 (62), taken). A change of stage B's cache and attention (prior §7 items 1 and 3), built by stage D's implementer (outline D184), the old form (each copy holding the shared rows) deleted and the new one written to be read (D187). How a line reads its front is a campaign setting, both ways kept at the user's word (2026-10-09): `front_attention` `copied` — the front copied back to each line for one layer's attention at a time, the same tensors, bit for bit the behaviour before it (the default; a record without it reads so) — or `merged` — one softmax over the front's and the line's own scores, no copy, its last bits moved (stage C's `Settings` and stage D's `MultiSettings`, profiled; `--front-attention`). `merged`'s check before use is D181 (57)'s bound (the user's choice): on fixed inputs (the CPU, one thread) the same words and outcomes but for draws near a boundary, counted, and the states within `STATE_BOUND_M`, on stage D's windows and on stage C's (its K copies of a point share their front too: for this change D149's bit for bit gives way to the same bound); a form that stays bit for bit is better and is stated. Then a new memory measure (D179) at the settings of stage D's log §18 sets the workers (expected 3–4). Not taken: a campaign setting that bounds the joint loop's rows (D181 (57)'s `continuation_rows`): it gives back the steps D188 saved. Stage D's readings of D190, accepted as built: (85) the room after a rebuild is the steps to the first part's end by the parts' schedule, D183's step where every part is a point chosen, and a point chosen that goes on alone is rebuilt as it is; (88) in `merged`'s check, records that differ only in rows past a sentence's end (an aircraft said on after it is done) lie outside the bound, since no sentence keeps those words. Stage D's readings of D189, accepted as built (items 81–84): (81) a commanded aircraft's landings run from 30 min and one Δ before its window's row 0 to the latest a row of its loop is said (`WindowLoop` reach), and its own; (82) no assertion of one airport in the loop, since `post.scene.Window` refuses another airport and it could never bind; (83) the later columns' procedure and caller masks of an aircraft not said are computed under the runway as said (every class is permitted to it, so the records are the same), and D62's check is read at the row's end, so a row refused before any change can raise a later column's message; (84) `search_steps` and `search_rows` count the joint loop's steps and rows, not comparable with the steps of stage D's log §18 (the rows are the same work). Why: the cache held one front K times and rows that no line reads; one front frees 61–79 % of the cache (log §19) | Decided | User, 2026-10-09 ((2) and its bound: the user's choice of item 80 (c); (1), the readings and the evaluation: Claude) |
| D191 | **Speaking workers on both devices, each batch's device by a fixed rule** (O20 (61), taken; the user, 2026-10-10, on Claude's reading of the speed after D190). A campaign's speaking workers may be on the GPU and on the CPU at once (stage C's and stage D's runners: the GPU workers as today and a number of CPU workers besides). Each batch of a round is spoken on the device that a fixed rule gives it by its place (`cpu_batches` of every `cpu_period`) — a campaign setting, recorded, whose default puts every batch on the campaign's speak device (the behaviour before it; a record without it reads so), which a resume may change as it may change the rounds (the user, 2026-10-10: "下一轮 用 cpu gpu 混合的方式说"; the record holds the rule before and after) — so that a round is the same whatever the workers' timing, and each device's workers take that device's batches in order. The selection readout is read by the speak device's workers alone, so every round's readout is read on one device whatever rule spoke its round. The rule's share is set from a batch's measured time on each device, so that both devices finish a round together (Claude's reading: of every period of batches by index, a fixed number on the CPU). D185's resume checks each done batch's device kind against the rule's for that batch, in place of one kind for the rest of the round; a round left half done under another rule is refused by name (`--restart-round`). The fit (D179, O15) counts each budget apart: the GPU's for the GPU workers and the pass, the host's for every worker's host memory (a CPU worker's whole peak, a GPU worker's host part) and the campaign's process. A CPU worker speaks on one thread. **The peaks last** (the user, 2026-10-10: the batches that need the most memory go last, and more workers): a GPU batch that runs out of its worker's share (CUDA's out of memory under its cap) does not fail the round; once the round's other batches are spoken it is spoken again at fewer at once — half the GPU workers, a quarter, …, one —, each level's budget the GPU less the campaign process's hold and what each idle worker holds, shared by the level's batches; past the last level the round fails. The worker frees the batch's memory before it takes another. A batch's numbers come from its place, so a batch spoken later or by another worker of its device gives the same result. The fit (D179) counts a GPU worker at what it holds between batches and refuses only where the last level cannot hold a batch at the profile's GPU peak × D179's margin; the margin stays for the host and the pass, which have no second try. Stage C's campaign (no cap) is unchanged: an out of memory raises. A batch spoken on the CPU and on the GPU can differ in its last bits (post-training §6.4): the rule makes a round repeatable, and nothing is compared across the two devices. First `multi_profile --speak-device cpu` at the settings of stage D's log §18 measures a CPU worker (its host peak and its batches' time); the mixed workers are built if the host holds at least two CPU workers beside three GPU workers (Claude's reading), else the measure is reported. Stage D's readings, accepted as built (items 91, 94): `round.json` holds `cpu_workers` (information, as `speak_device`); a stage C measure (`campaign.json` `measures`) holds `cpu_workers` and `cpu_measured`, and a launch with CPU workers measures again where the measure holds none (as D167's record without a measure); `profiled_fit`'s line names `cpu_profile`, `cpu_workers` and the levels (`deferred`). Why: after D186–D190 a batch on the CPU took about 1.4–1.5 times its time on the GPU (stage D's checks, five processes at once, `merged`; at D180, three to nine times), the GPU holds three workers, and the host has 28 cores and about 26 GiB free: two or three CPU workers beside three GPU workers would speak a round about 1.4–1.7 times as fast (Claude's estimate) | Decided | User, 2026-10-10 (the evaluation and the form: Claude; the peaks last and a resume's change of the rule: the user, 2026-10-10) |
| D192 | **The second group at the point most continuations avoid** (stage D's plan of 2026-10-10, R1; stage D's item 95). A new value of `MultiSettings.branching`: `majority`. It is `backward` (D182) with point B replaced: after the decision point and its 16 s point (D182 (4), (5), unchanged), the search goes on back, within D182's bound (8 points of 32 s, never before v's first predicted step), to the first point at which at least 5 of its K = 8 continuations avoid the event (D182 (3)'s avoiding, unchanged) — the majority point; its K continuations flown to the window's end make a group with the first sentence. When the decision point has 5 avoiding continuations it is the majority point and gives one group; when no point within the bound has 5, the search gives the decision point's groups only (the bound is not widened); any other point tried gives no group (D182 (7)). No condition on go-arounds: a continuation that avoids the event by a go-around counts, and D30 pays its 0.9. `grid` and `backward` are unchanged, bit for bit (the campaigns run with `backward` stay readable and resumable); `majority` is profiled and compared by a resume as `branching` is. A `majority` batch's record adds each group's kind (decision, 16 s, majority) beside its depth, and the searches that found a majority point; the readouts give the majority points' depths by span. Stage D's readings, accepted as built (item 103): the majority is `continuations // 2 + 1`, more than half; the kinds (`group_kinds`) and the majority points found (`majority_depths_s`, each its span and depth, the decision point counted when it is the majority point) are in a `majority` batch's record only, `backward`'s record keys unchanged so that its campaigns stay resumable; a campaign with `majority` has its own profile (D182 (i)). Checks: the search's points on a synthetic window whose event 1, 5 and 8 continuations avoid at known points (a majority point behind the decision point, the decision point as the majority point, none within the bound). Built at the user's word (2026-10-10 22:10, "现在实现"), 172bc07d. Why: at the decision point often one continuation in 8 avoids, by a late word that fits that one geometry, and one round's read of the gradients (stage D's plan §1, R70) found the prior's reward gradient no more alike between updates than noise; at a point most continuations avoid, the usual words mostly work, and the group compares earlier, milder changes (a speed, a spacing) that more conflicts share (Claude's reading) | Decided; built | User, 2026-10-10 evening ("直接把 B 换成多数能避开的点，至少 5 条；但是现在先不实现"); the details (the bound not widened, no go-around condition, the 16 s point kept, the name): Claude |
| D193 | **Speed after MC6's rounds: the executor on the CPU, the rows no longer flown** (stage D's plan of 2026-10-10, S1, S2, S4, S6; stage D's items 96–98; the evaluation: Claude, 2026-10-10). (1) S1, built: the process option `--executor-device {speak,cpu}` (`multi_train`, `multi_speed`) flies a GPU worker's executor on the CPU while its speaker stays on the GPU (`Context.executor_device`; `SpeakingLoop` and `WindowLoop` take `executor_device` and keep it through D182's host copies and D188's joins); default `speak`, the behaviour before it, bit for bit; each batch's record holds `executor_device` (information, as `speak_device`, D180). A device, as the speak device, not a campaign setting. Its last bits move, so before a campaign uses it D181 (57)'s bound is checked on fixed inputs on stage D's real windows (MC6's batches 0, 23 and 86 of round 0) and one batch is timed with each executor device — checked (stage D's log §22: the same groups, words and ends, the states well within `STATE_BOUND_M`). (2) S2, built: the rows of a window that are not flown take their tokens in one call per window (`post/edges.each_tokens`, `WindowLoop._alone`), bit for bit, checked once (outline D187). (3) S6: a row no longer flown (its aircraft landed, or ended at a loss, while others of its window still fly) leaves the joint loop (D188) at the loop's next rebuild (D190 (1)): it is no longer said, flown or given tokens, since nothing reads its later words or states. The batch's rows change, so its check before use is D188 (1)'s (D181 (57)'s bound on fixed inputs, the CPU, one thread; bit for bit where it holds, stated); it replaces the form before it and is written to be read (D187). The first pass's window loop is unchanged. Stage D's readings of (3), accepted as built (items 99, 100, 104): a flight that has left the loop keeps a frozen record (`SpeakingLoop.left`), from which its states, sentence and outcome are read; `SpeakingLoop.permitted(flights)` gives one record a flight, its sentence's rows, not padded to the others' (a padded record would depend on which flights left; the loss reads a record's row by its time, so it is the same, bit for bit; `Speaker.permitted` keeps D181 (63)'s padding); stage C's pinned digests of D149, which hash these records, were written again once the code before with the record cut alone gave the same digests; the speaker keeps a row of records for every row from the first in which some aircraft is said, one with none said included (a joint loop rebuilt with only an observed flight left had no row to record). (4) S4: a CPU batch past its worker's host share is spoken again after the others at fewer CPU workers at once, as D191's peaks last on the GPU (on the CPU still: a batch keeps its rule's device): each CPU worker has a host cap at which an allocation fails in the worker (an error it catches), before the host swaps; the fit counts a CPU worker at what it holds between batches and refuses where the last level cannot hold the profile's host peak × D179's margin. Stage D's readings of (4), accepted as built (item 101): the worker reads its own memory every `HOST_CAP_SAMPLE_S` = 1 s (a read costs tens of ms) and raises the error in its main thread by a signal, so a batch can pass its cap for about a second; a capped worker gives the heap's free pages back after each batch; the GPU's levels are spoken before the CPU's. (3) and (4) built and merged. Not now: stage D's item 86 (the fronts of a cache in two blocks of own rows: the GPU's compute, not its memory, sets a round's time; it returns when memory does). Why: on a 1,200 s batch the GPU's executor took a third of the batch and the CPU's much less, a third of a backward search's rows were rows no longer flown, and a CPU batch past its share made four CPU workers swap (stage D's plan of 2026-10-10 §1, §3) | Decided | User, 2026-10-10 (S1 and S2 ordered); the evaluation of S4, S6 and item 86: Claude |
| D194 | **The arms of 2026-10-10 night: the data term's weight; speed that changes no group** (stage D's plan of 2026-10-10, R2; stage D's items 105–107; the evaluation of (2)–(4): Claude, 2026-10-11). (1) The data term's weight is a campaign setting of stage D, `MultiSettings.data_weight` (`--data-weight`; `post.loss.update_step(…, data_weight=)` through `train_pass`; R70 reads at the campaign's weight): default `post.loss.DATA_WEIGHT` = 1.0, the behaviour before it (a record without it reads so); not profiled (the memory does not change); a resume with another weight is refused; stage C's campaign unchanged. The data term is a mean over its sentences' words, so `data_sentences` sets its noise and its cost, not its weight. Two arms of two rounds each from MC6's start: A `majority` at 1.0, B `majority` at 0.25. (2) A round's batches not done are spoken in the order of their span, longest first, those of one span by index (stage C's windows, span 0: by index, as before); a batch's device is still its rule's (D191), deferred batches still go last, and the record and the pass read the batches in index order, so nothing learned changes. (3) A point ends its continuations, while it is tried, once it can no longer be chosen: a point tried for the majority point (D192) once fewer continuations are left than the majority (5 of 8: once 4 have failed); by the same rule a point tried for the decision point, or as the 16 s point, once all K have failed. A continuation fails only where nothing later can undo it, read at each step (`_failed`): v answered for a loss; for a loss event, the event's partner answered for one with v; for an end, v ended with the event's outcome; a go-around said is read at the judged tick only (so such a point stops later, never wrongly). A point chosen is flown to the window's end and never ended early. The search's answers (its points, groups, kinds, depths and counts) are those without the stop; only its steps, rows and time fall. Before use it is checked on the synthetic window and on stage D's real windows (MC6's round 0 batches 0, 23, 86, `majority`, the CPU, one thread, with and without the stop): the same groups, kinds and depths, by D181 (57)'s bound where the batch's rows change a last bit. (4) The number of CPU workers is the launch's, set by the fit (D191): a batch's device is its rule's by index, so the number of workers changes no group. Arms compared with each other keep one rule (`--cpu-batches`, `--cpu-period`), so that they differ in their setting alone; a new campaign sets its rule from the speeds measured after S1 and S6 (D193). Why: (1) one round's read of the gradients found the prior's data term about half its update and pointing one way, its reward term no more alike than noise (stage D's plan §1, R70), and fewer sentences would only make the term noisier; (2) sent by index, a round's 1,200 s batches came last and its last minutes ran on a few workers; (3) a point's continuations after it can no longer be chosen give nothing | Decided | User, 2026-10-10 23:20 ((1): the weight 0.25, arms A and B), 2026-10-11 (speed-ups that change no group); (2)–(4), and (3) for the decision and 16 s points: Claude |
| D186 | **The window loop's own work, cut** (MC11; stage D's item 70, the profile of the continuations; the evaluation: Claude, 2026-10-09). The continuations spend more time in the loop's own Python than in the model and the executor together (stage D's log). Three changes, each in its stage's code by stage D's implementer (outline D184), each before use checked on fixed inputs: (1) a landing added to a loop (`SpeakingLoop.add_landing`, `LandingIndex.with_landing`, stage B) checks and inserts the new landing only instead of building and checking the airport's whole roster again — the same index, so the same numbers, bit for bit; (2) the grammar's masks of a row computed over the batch (`instructions/grammar.column_mask`, stage A), which replaces the readable mask after its check (outline D187; the labeller reads `apply`) — boolean logic, so equal bit for bit; it rests on a property of rules 1–6 that vocabulary §6 item 2 states (a heading or a speed word enters the rules only through rule 1), which a change of the rules must keep or change the mask with; (3) the inputs of a row computed over the batch (`prior/inputs.state_inputs` and `instructions/airport.relative_to_runway`, stages B and A), which replaces the computation one aircraft at a time after its check (outline D187), equal bit for bit as D138 asks; if a vectorised function differs in its last bits (a library's own sine or cosine), the differences are reported and the user decides whether a bound (words and outcomes the same but for draws near a boundary, counted; D181 (57)) may stand in for equality. Each with its stage's checks (the prior's behaviour check, prior D108; the labeller's conformance where the grammar is read; D149's check on stage C's and stage D's windows), once before its commit (outline D187). Item 56's runner measures each before and after. Not now: the traffic module's tokens, the model's step and the executor | Decided | User, 2026-10-09 (the request; the evaluation: Claude) |

### 0.2 Open items

| # | Item | Proposal | § |
|---|---|---|---|
| O18 | The time that the aircraft take | D151: readout only (the delay of each landing against its record, the spacing at the threshold). A term only if the readouts show that the model spreads the aircraft out; the user decides with the numbers | 2.5 |
| O19 | **Stage C's later credit methods in stage D** (post-training D170, D171). Stage C's value network cannot serve stage D as it is: it values one aircraft's r, not the window's W (D141); it reads the future of the recorded aircraft, and a commanded aircraft's future depends on the words, so no record holds it; and it reads one commanded aircraft, not a window of several. What can carry over: the code of the advantages and of V's file (post-training §9 item 10), V's weights as the start of the part of a stage D value that reads one aircraft and the recorded traffic, and D170's settings. A start from a value round reads the checkpoint as any other (D164). A value for stage D would save most of its speaking (each varied aircraft's continuations fly the whole window again), but it must keep the credit between the aircraft of one tick, which a value of W alone does not give (§2.3) | MC6 keeps D142, D143 and D170's defaults (the user, 2026-10-07). After stage C's campaign with a value function is read, and only if it gains over the branch training, Claude writes stage D's value (its inputs and its credit between the aircraft) for the user. With `segment_only`, v's segment would end at v's next point on the window's grid. Not taken up (the user, 2026-10-10: stage C's value function was tried and fell) | 2.3 |
| O20 | **Two more ways to a faster round** (stage D's items 61 and 62). (61) Workers on two devices at once (taken by D191; the GPU's memory and the host's are separate budgets: 2 GPU and 2–3 CPU workers together): a batch's device by a fixed rule (its place in the round) keeps a round repeatable with a poorer balance; a device taken as workers free up makes a round depend on timing, and a CPU and a GPU batch can differ by a rounding. (62) The K copies of a branch point share the speaker's cache up to the point: taken by D190 (2) | Closed: (62) taken by D190 (2), (61) by D191 | 11 |
| O21 | **Where a round's cost goes** (Claude's reading of stage D's measures, 2026-10-09). With D146's equal parts of the spans, the 1,200 s windows take most of a round's speaking: a window's cost grows about with the square of its commanded aircraft, since each continuation flies the whole window (D141, D142). A group whose W is the same in all its sentences costs K continuations and gives no sample, and D143 also varies partners whose r is 1 | After MC6's rounds, read its readouts by span (§5 item 4): the gains and the share of informative groups (and, where cheap to count, the groups of a failed aircraft against a partner's). If the long windows or the partners' groups give less than their cost, the user may give the spans counts in proportion to their cost or narrow D143's varied aircraft. No change before. D182's backward points address the groups that give no sample, for a campaign that uses them. MC6's rounds give the cost by span (stage D's plan of 2026-10-10, S3, a proposal): the user decides, after D192's campaign | 3.1 |

### 0.3 Implementation

Stage D's implementer keeps this table and its log, `readouts/2026-10-06_stage_d_implementation_log.md` (outline §5
rule 10); its readings and requests go to `requests_from_d_to_designer.md`. State, 2026-10-11 (09:30): everything up
to D194 merged (226ed303; D194 (3) for every point tried, d3dc3b26); arms A and B done (A 82.2 / 83.3 %, B 82.8 /
82.3 % landed); the formal campaign `multi_train_majority_formal_20261011` (arm A's settings, 8 rounds) runs (log §22).

| Milestone | State |
|---|---|
| MC0 | Committed 9c98a8d6, d7bf55f5, 3a9b7153, 80fe774c, 648b6591, 5e456406, a1ea664f; D73's checks on the formal artefact and D149's real-window check (200 of 200 windows bit for bit) passed (log §2–§4, §9) |
| MC1 | Committed e6f0e564; the census read on a stated sample, 500 anchors an airport and split (log §5, §10) |
| MC2, MC3 | Committed fd720c75 (log §7) |
| MC4 | Committed 054727db, ac919f9c, 2b370d37, fccbab8d, 6cc49ad1; the smoke and its resume passed (log §7, §8, §11) |
| MC5 | The memory measure (D179, D180): committed 48c21e50; measured once for MC6's settings, the GPU's profile `outputs/POOLED/multi/profile_mc6_gpu32_20261009` (log §14, §15) |
| Frontend D177 (12), D178 (6) | `multi.census.judged_steps` 897e433f, `loop_positions` a332466a (log §13, §14) |
| MC6 | First launch stopped by the user inside round 0 (2026-10-09 14:31, 134 of 224 batches), `outputs/POOLED/multi/multi_train_20261009` kept as it was (log §15, §16). Second launch (`multi_train_20261010`, 2026-10-10 03:20, e2b69736 then c7c35b67; backward, merged): rounds 0–4 landed 82.1–82.9 % against the start's 81.5 %, stopped by the user in round 5 (92 of 217 batches), not resumed; its arms lr × 3 (`multi_train_lr3_20261010`, 80.4 % after 2 rounds) and traffic only (`multi_train_traffic_only_20261010`, 82.4 / 82.5 %) (log §22) |
| MC11 (D181) | Merged: 55 (92341758), 58–60 (37526695), 56 `multi_speed` (a8df27f7); item 56 measured on the GPU (log §16) |
| MC11 (D183, D186) | Merged (88b5d671): 69 (0fa9df2e), D183 (317979f9; its copy allocated once, order B, 64e62b4e), D186 (1) (ab6b2e8e), (2) (3) (ad950c8a; the fast forms replace the old code, order A, ae4249fd); the behaviour check `window_behaviour` (cfe5da0f); every change the same on stage C's and D's fixed windows and in prior_behaviour; the CPU batch 86: 189 → 130 s (log §17) |
| Post-training C27 (D185) | Merged (88b5d671): 33e24bbe; the resume through the speaking workers (order C, 646ffb3a) (log §17) |
| MC12 (D182) | Merged (88b5d671): de0711aa; the grid the same on stage C's and D's fixed windows; the GPU measures of grid against backward wait for the user's word (log §17) |
| MC12 (D188) | Merged (88b5d671): (1) the searches stepped together, efef5c4d (against the code before on MC6's batches 0, 23, 86: every group and continuation bit for bit on the CPU); (2) the speaking workers' allocator, c024a333 (the group files the same with and without it) (log §19) |
| MC11 (D189) | Merged (88b5d671): (0)–(2) ceccd417, (3) 9c2161fa; window_behaviour on stage C's and D's windows and prior_behaviour the same bit for bit; the profile at §18's settings, `outputs/POOLED/multi/profile_d189_20261009`: batches 0 / 23 / 86 in 55 / 65 / 61 s (§18: 112 / 117 / 81 s), a worker's GPU peak 2.21 GiB: 2 workers fit (log §19) |
| MC11 (D190) | Merged (88b5d671): (1) the joint loop's room to its next rebuild, ada6a79d (bit for bit: window_behaviour on stage C's and D's windows, prior_behaviour, MC6's backward batches); (2) the copies of a point hold their front once, 7893d787 (the stage C frontend fixtures 82dd7695), `front_attention` `copied` (the default, bit for bit) or `merged` (D181 (57)'s bound: the words of 1 of 1,600 continuations differ); the profile at §18's settings, `outputs/POOLED/multi/profile_d190_{copied,merged}_20261010`: a worker's GPU peak 1.53 / 1.33 GiB, 3 workers fit (log §20) |
| MC11 (D191) | Merged (88b5d671): 5b4cd739 (workers on the GPU and the CPU, each batch's device by a rule; the default bit for bit), e2b69736 (a GPU batch past its worker's share spoken again after the others at fewer at once, the user's word); a CPU worker's profile `outputs/POOLED/multi/profile_d191_cpu_20261010`; the full suite 2,113 passed (log §21) |
| MC11 (R69, R70) | Merged (88b5d671): `multi_compare` 35a6e072, `pass_gradients` a23a49d7, a2f51add; read on MC6's round 5 (log §22) |
| MC11 (D193 (1) (2)) | Merged (88b5d671): S1 2961b99a, S2 88a3990e (bit for bit); S1's check on the GPU on MC6's batches 0 / 23 / 86: the same groups, words and ends, states within 2.2e-8 m; batch 86 on the GPU 53.9 → 48.8 s with the executor on the CPU (log §22) |
| MC11 (D193 (3)) | Merged (19d90d45): 9d6886ec, stage C's digests repinned 6c5269af; window_behaviour on stage C's and D's windows and prior_behaviour the same bit for bit; batch 86 on the CPU 68.1 → 51.0 s, on the GPU 62.1 → 53.9 s; each flight's own permitted record (requests item 99) (log §22) |
| MC11 (D193 (4)) | Merged (19d90d45): 22fd1031; a CPU batch past its host share spoken again at the CPU's levels (log §22) |
| MC12 (D192) | Merged (19d90d45): 172bc07d, `--branching majority`; backward the same behaviour on MC6's windows; profiled at MC6's settings, `outputs/POOLED/multi/profile_d192_{gpu,cpu}_20261010` (3 GPU, 4 CPU workers fit); no campaign (log §22, requests item 103) |
| MC11 (D194 (2), (3)) | Merged (ca53c517, 226ed303): ce886c4a, 9bdbe6c5, d3dc3b26 (every point tried) (the longest batches first; a majority point out of reach ends its continuations); window_behaviour on arm A's campaign against 14e53df6: every digest the same (log §22) |
| MC12 (D194 (1)) | Merged (e77bc220): f81b5a13, `--data-weight` (default `DATA_WEIGHT`) (log §22) |
| Arms A, B (D192, D194 (1)) | A, `outputs/POOLED/multi/multi_train_majority_20261010`: done 2026-10-11 02:57, rounds 0 / 1 landed 82.2 / 83.3 %, lost separation 15.5 / 14.7 % (MC6's rounds 0–4: 82.1–82.9 %, 14.6–15.9 %); read-only with its checksums. B (data weight 0.25), `…/multi_train_majority_data025_20261011`: done 05:10, 82.8 / 82.3 %, 15.4 / 15.3 %; read-only (readout `readouts/2026-10-11_stage_d_arms_ab.zh.md`, log §22) |
| Formal campaign (D192, arm A's settings) | `…/multi_train_majority_formal_20261011`, 8 rounds, launched 2026-10-11 09:25 at 226ed303 (D194 (3) for every point tried, d3dc3b26), 3 GPU + 4 CPU workers, 8 of every 15 batches on the CPU (log §22) |
| MC8–MC10 | Not started |

### 0.4 Plan

1. MC0–MC5 are done: the interfaces, the census, the loop of several commanded aircraft, the credit, the runners and
   the memory measure; their specifications and checks are in stage D's log.
2. MC6, version 1's formal campaign from stage C's chosen round (D164) in a run worktree (outline D163): launched
   again on 2026-10-10 on MC11's and MC12's code (D181–D191, `backward`), stopped in round 5 (stage D's handover of
   2026-10-10). D193 and MC13 (D192, the majority point) are built and merged. Two arms from MC6's start test D192 and the data
   term's weight (D194 (1)); D194 (2), (3) speed them without changing a group. The other proposals of stage D's plan
   of 2026-10-10 are experiments, chosen by the user.
3. MC7, the Training view of stage D, by fronter. O18 from MC6's readouts; O21 for the user after D192's campaign; O19
   not taken up (the user, 2026-10-10).
4. Version 2 (D153): MC8 (its code), MC9 (its memory measure, its campaign from version 1's chosen round, its
   validation readout); MC10 closes stage D.

The milestones are named MC (multi-aircraft control), not D: D numbers are decisions.

---

## 1 Scope

- **This document owns** the windows with several commanded aircraft, the window's reward and its credit, the varied
  aircraft and their branch points, the landings of a loop, their runners (`experiments/multi_*.py`), the package
  `multi/`, and the Training view of stage D.
- **It reads** the outline; the vocabulary's public interface (vocabulary §6: the grammar, the sentence artefact and
  the stored signals, the candidates, the executor and the start of a closed loop, the judge, the row grid, the
  Training export); the prior's public interface (prior §7: the checkpoint, the inputs of a row, the speaker and its
  masks of a caller, the teacher-forced loss and the log-probability under records, the step of a speaker's closed
  loop, the Training export's procedure block); and the
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

The present landing direction (post-training D112, §9 item 6) is read for each aircraft at its own first predicted
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
   mean of the group's W; it applies to v's rows after the branch point up to v's event (its end, or the row from
   which it is silent, D144), and to no other aircraft's rows.

**Why this gives v's credit (Claude's reasoning).** The draws of the aircraft are independent. Inside a group, the
numbers of every aircraft but v are fixed, so a difference of W between two sentences of the group is caused by v's
draws: by v's own outcome, and by the other aircraft's outcomes as they answer v. The expectation of the gradient over
the other aircraft's numbers is the gradient of the expected W through v's words, so fixing them in a group removes
their noise from that part of the gradient. The part through the other aircraft's words is not in v's group: an
aircraft's changed words in a continuation are its answers to v, and they get their own credit when that aircraft is
varied. The group's mean includes the sentence itself, as in stage C. The idea is close to the counterfactual baselines of multi-agent learning (for example, COMA,
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

**The points found backward (D182, a setting).** With `branching = backward`, the points of a varied aircraft are not
fixed in advance: from its event, back in steps of 32 s (at most 8), the first point at which one of K continuations
avoids the event (to 60 s after it, v answering for no loss) is its decision point, one point 16 s after it is tried
too, and the search goes on back to the first point at which one avoids it with no go-around (B); only those points give groups, flown
to the window's end, while the points tried before them are flown only to 60 s after the event. With
`branching = majority` (D192) the search goes on back instead to the first point at which at least 5 of the K
continuations avoid the event (the majority point), in place of B. The points of
different windows are then at different ticks, so a batch's continuations no longer share one row of the loop.

### 2.4 Losses of separation and their aircraft (D144, D145)

The separation judge is stage C's (`inference/separation.py`, reading `VISUAL`, post-training §9 item 3): after each row
flown, it judges every pair of the scene. For each loss, the rules name the responsible aircraft: the aircraft behind
in trail; the joining aircraft where one is established; both where neither is.

| Pair | What it gives |
|---|---|
| Two commanded aircraft | Each responsible one that is still spoken answers: r = 0, and it becomes silent (D144). The other one flies on |
| A commanded and a recorded aircraft, the commanded one responsible | As above |
| A commanded and a recorded aircraft, only the recorded one responsible, the two records kept their separation at that step | The commanded one answers, as above (D145): the model made the loss |
| A commanded and a recorded aircraft, only the recorded one responsible, the records also lose separation at that step | Counted in the readouts; nothing for W (D145) |
| A silent aircraft and a spoken one | The spoken one answers when it is responsible; the silent one never answers again |
| A commanded aircraft in its observed rows (from its row 0 to its first predicted step) and another aircraft | The observed one flies its record and answers for nothing, as a recorded aircraft (D145); the other one answers when it is responsible. From its first said row on, the observed one is judged as every commanded aircraft |
| Two recorded aircraft | Not judged (stage C) |

**Silent, not removed (D144).** An aircraft that answers for a loss keeps flying on its words in force, through the
loop, with a caller's mask that permits only "unchanged" in every column. The mask makes its rows certain (probability
1), so they add nothing to the surrogate or the pull, and they are not counted rows (post-training D115); the
executor flies it to its end (a landing, a crossing, its time
limit). The other aircraft must keep their separation from it. Removing it would give the others the space that the
loss took, so a loss would pay the window.

**The end of a window.** A window ends when every commanded aircraft is done or silent. An aircraft that has not
joined yet, or is in its observed rows, keeps the window going. A silent aircraft is then halted: no reward can change
after that row. With one commanded aircraft, the window ends at its loss, as post-training
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
| v cuts in front of a recorded follower that then loses separation (only the follower responsible; the records kept their separation) | −1 | Not to cut in front of the recorded traffic (D145) |

---

## 3 The window and the loop

### 3.1 The window (D146)

- **The commanded aircraft**: the anchor and every flight of the same airport and split with a closed-loop sentence at
  the chosen Δ (inside the base's selection or not, post-training D76) whose row 0 is in [the anchor's row 0, + L),
  L the window's span (5, 10 or 20 min, D146).
  Each has its own row 0, first predicted step, start and time limit (the start of a closed loop, vocabulary §6 item
  5).
- **The recorded aircraft**: every other flight of the airport and the split in the air at the step, from its entry
  into the arrival slice to its landing (post-training D93), with its R and G as in stage C (post-training D99, D117).
  This includes the aircraft already in the air at the anchor's row 0 and every aircraft that enters after the span.
- **The draw of a round** (post-training D124, generalised): the train's flights in one permutation (the seed and the
  round); each flight in turn is an anchor with its span drawn by the round's numbers, until each kind's 1,000 windows
  are reached in equal parts of the spans (D146; a shortfall of a span recorded). The windows of a round share their
  flights; a batch holds windows of one L and commands each flight once (a loop holds each flight once), so windows
  that share a commanded flight go to different batches. D146's rule of the windows left out is checked at the draw.
  A compressed window whose shifts all round to 0 flies the real window and is counted as real in the readouts (the
  draw counts it as drawn compressed); a window without a later aircraft has no compressed form (D166).
- **Compressed windows**: each commanded aircraft other than the anchor is moved earlier, its offset from the anchor's
  row 0 multiplied by c (c uniform in [c_min, 1], the moved offset rounded to a whole Δ); the recorded aircraft do not
  move. A moved aircraft's record moves with it: its observed rows, its row 0, its first predicted step; its states
  and words do not change (the executor and the inputs read no absolute time, except the landings it counts, which are
  read at its new time: its moved entry time in `LoopRows`, §6.3 item 2).
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
2. For each commanded aircraft still flown: its tokens (post-training §9 item 4, `post.edges.tokens`) — the recorded aircraft
   from their records, every other commanded aircraft that has joined from its states in the loop (its observed rows
   before its first predicted step, then the states that the loop flew: its position at the row and the 2 s row
   before it), with its R and G from its words in force (silent aircraft included; before its first predicted step no
   R, D23), and the token part of D152: the flags commanded and silent and its words in force; the speed-word mask
   (D148).
3. Every spoken aircraft says one row, all at once (D148); a silent one says "unchanged" (D144); the executor flies
   every one of them for Δ.
4. The separation judge on the scene at the next row (§2.4); a responsible spoken aircraft becomes silent; an
   aircraft in its observed rows answers for nothing.
5. A commanded aircraft whose judged outcome is `landed` adds its landing on its landed runway to the other
   aircraft's landings from its crossing time (D147).
6. The window ends when every commanded aircraft is done or silent (D144).

**R and D23.** An aircraft's R is used, by any input or token, only from the row after its first predicted step
(post-training D31, D117); the test of D31 (a change of one aircraft's runway word leaves every input and token of the scene
at the rows up to its first predicted step the same, bit for bit) runs with several commanded aircraft.

**The faulty points** of the recorded aircraft are counted as in stage C (post-training D114). A commanded aircraft
flies on the executor and has none.

### 3.3 Speaking for several aircraft

- **One batch.** The commanded aircraft of all windows of a batch are one batch of the speaker (prior §7 item 3),
  each joining at its own tick (D150, §6.3). The prior's forward pass, the masks over the batch (prior B14) and the
  executor's steps are those of stage C; an aircraft that has not joined is not spoken and not flown.
- **Random numbers.** The numbers of each aircraft are the caller's (post-training §9 item 9). Stage D gives each commanded
  aircraft its own stream, from the seed, the round, the window's place in the round and the aircraft's place in the
  window, and a continuation's from these, the branch point and k; a copy of a window copies every stream (D142).
  Stage C keeps D94's streams (the seed, the round and the window; with the branch point and k), so that its windows
  say what they say today (D149).
- **The two passes** of a round are post-training D94's: the first pass speaks every window once; the second speaks
  the windows with W below their count again with the first pass's numbers, and at each branch point of each varied
  aircraft copies the window's state K times. The check of D94 holds for every aircraft: the second pass says the
  first pass's words and flies its states within `STATE_BOUND_M` up to the last branch point; a window that differs is
  counted, reported and gives no group.

---

## 4 Loss and masks

**Masks.** The prior's grammar and procedure masks under the set that the prior was trained under; the masks of a
caller: "go-around" after an aircraft's second go-around (post-training D91), the speed-word mask (post-training §9 item 5,
D148), and the silent aircraft's mask (D144).

**Loss.** Post-training §9 item 10, unchanged (`post.loss`): the clipped-ratio surrogate (ε = 0.2), word by word, on
the counted rows of each sample; the pull to the base (weight 0.04); the teacher-forced data term (weight 1) on
single-aircraft samples of the train days in the base's selection `landed` (post-training D36, D76). A sample is one
varied aircraft's sentence of one group: its rows, the speaker's records of them and its tokens, as a sample of stage
C (`post.branches.Sentence`). Every counted row weighs the same (post-training D115); the groups of a file are taken
in a shuffled order (D130); the surrogate and the pull in eval mode, the data term with the base's dropout (D107); the
ratio's denominator is the model at the start of the pass; the traffic attention at its own learning rate.

---

## 5 Selection, validation and readouts

1. **The start (round 0)** is stage C's chosen round, read in stage D's windows.
2. **The selection readout** of each round: a fixed set of real windows of the select days of each span (anchors in an
   order drawn once with the seed, each kept unless D146 leaves it out, at most `select_per_airport` an airport and
   span), spoken once with the same numbers every round (each aircraft's `[seed, 2^30, place, member]`, D166). Its size (`select_per_airport`) is a setting fixed before
   the campaign (D179), large enough that the paired standard error of W per aircraft between two rounds can resolve
   the change that the user wants to see. Why: the archived campaign's selection set could not resolve a change of 0.012, and its chosen round's
   gain (+0.025) came mostly from taking the best of 8 rounds (§9.1).
3. **The validation readout**: the chosen round, read one time on the val days (outline D85), with a claim before the
   read (as post-training D132).
4. **The readouts** of each window set (no criterion: D7): by airport, kind and span (the validation readout too),
   - W per commanded aircraft; each aircraft's outcome; landed; go-arounds; silent aircraft;
   - the losses of separation by pair (commanded–commanded; commanded–recorded, the commanded one answering by the
     rules or because the records kept their separation; recorded only, with the records also losing it: D145), by
     kind of the judge;
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

Stage D imports from `post/` and from stage C's runners only the names of post-training §9. Where one commanded
aircraft becomes several, stage C's code was generalised there, not copied (MC0): the scene and a window (§9 item 1),
the landings (item 2), the losses that an aircraft answers for (item 3), the traffic attention's token part (item 7,
D152), the window loop (item 8), the branch training's rules (item 9), the campaign's skeleton (item 11) and the start
from a round (item 12); the tokens (item 4) and the loss (item 10) are unchanged. Each generalisation keeps stage C's
behaviour: with one commanded aircraft, the same words, states, rewards and groups, bit for bit, on fixed inputs (the
CPU, one thread), against stage C's code before the change, on synthetic and on real windows; stage C's formats keep
their names. A later change of the shared code runs the same check (D181's items 58–60).

### 6.2 The vocabulary's and the prior's public interfaces (D150)

A change that stage D needs of `instructions/`, `autopilot/` or `prior/` is written in that stage's public interface
and built by stage D's implementer with that stage's checks; a small defect of their code is fixed directly (outline
D184), its names written into the interface after.

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
   row, as today; in a compressed window its moved entry time); an absent aircraft's row is not present.
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


### 6.4 Version 2: the other aircraft's intent (D153)

Version 1 (D152) gives each commanded aircraft the states of the other aircraft and the words that the model told each
of them. Version 2 adds what the model is about to tell them: a commanded aircraft's row reads the hidden states of
the other commanded aircraft of its window at the same tick. It is built and trained after version 1's campaign (MC8,
MC9), on version 1's chosen round.

**The module** (stage D's, `multi/`). At each layer, in the place of an added module (prior §7 item 5), beside stage
C's traffic attention: for the row of commanded aircraft i at tick t, an attention over the layer's input rows of the
other commanded aircraft of i's window at tick t that have joined (observed, said or silent; an absent aircraft is
never a key). Its output layer starts at zero, so version 2's start says what version 1's chosen round says, bit for
bit. A hidden state at tick t is computed from the inputs at the start of row t; it holds no word of row t, so no
aircraft reads a word said in its own row (D148). The recorded aircraft stay tokens (D152). The caller's input of the
added modules tells the module which rows of the batch are one window and which have joined.

**Where the rows meet.** In the loop, the speaker already runs one tick of every aircraft of a batch in one call of
the network (D150): the module reads the other rows of that call. In the loss, the hidden states of the other aircraft
at every tick are needed, so a sample is the whole window: the rows of every commanded aircraft of the window, each
sentence placed from its own join tick on the window's ticks (the ticks before it not present), encoded together;
only the varied aircraft's rows after the branch point, up to its event, are counted (D142). The pull to the base and
the data term are as in version 1 (the data term's single-aircraft samples have no other commanded aircraft: the
module adds zero).

**What it changes in the code of the earlier stages** (each written in that stage's public interface before it is
built, MC8):

| Stage | Code | The change | With the change not used |
|---|---|---|---|
| B | Prior §7 items 4 and 7: `SpeakingLoop.sentences`, `prior/batch.py` `collate`, `masked_log_probability` | The rows of several sentences that a loop said, placed on the loop's ticks (each from its join tick; the rows before it not present), as one batch | Every join tick 0, or one sentence: today's batch and log-probabilities, bit for bit (a test) |
| B | Prior §7 item 5, `Layer.added` | None: a caller's module already gets the whole batch's rows and the caller's input | — |
| C | Post-training §9 items 9 and 10 (`post/branches.py` `samples`, `post/loss.py` `Samples`) | A sample of several sentences (a window) with its counted rows | One sentence: stage C's samples and loss, bit for bit (D149's test) |
| D | `multi/` | The module, its format name, the window samples | — |

**Why no trained artefact becomes invalid.** A trained artefact stays valid while the code that reads it behaves as it
did and its format keeps its name. Every change above is added beside today's behaviour, which stays the default and
is checked bit for bit; no format of an earlier stage changes: the base's checkpoint (`ts-prior-checkpoint-v8`), stage
C's checkpoint, token features and module (`ts-post-checkpoint-v1`, `post-edges-v1`, `post-traffic-attention-v2`),
and version 1's checkpoint and token part keep their names and their shapes. So the base, every round of C10 and every
checkpoint and readout of version 1 stay valid; version 2's checkpoint and campaign are new names
(`ts-multi-checkpoint-v2`, `ts-multi-train-v2`). Forbidden, because it would invalidate them: a change of stage C's
token features or of the shape of an existing module (C10's checkpoints would no longer load), and a change of the
default behaviour of the prior's or stage C's code. Version 1's branch groups are not read by version 2: each
campaign speaks its own rounds.

**What it costs.** The bytes of the branch groups and the work of the loss grow with the commanded aircraft of a
window (MC1 gives their p50 and p90): a sample holds every one of them. An update runs in pieces of one branch group
each (post-training §9 item 10, `update_step`), so the GPU's memory is that of one window's group. The module's attention scores are ticks ×
aircraft² for each head and layer (about 100 for each tick of a window of ten aircraft); there are no vectors for each
pair of aircraft, which made the archived design's 5.3 GB a layer (§9.1). MC9 measures the memory before the campaign (as MC5, D179);
a round's time comes from its round 0.

---

## 7 Gates and identities

The gate of this document is the multi-aircraft control. The user sets its criteria (D7). The identities follow D21
(outline §3):

| # | What | Its identity |
|---|---|---|
| 1 | The edge features | Stage C's conformance check (post-training §9 item 4), unchanged, run in every process that computes tokens |
| 2 | A window readout | The checks of the code it uses, at its start: the edge features', the executor's and the closed loop's (vocabulary D71, D73), and D149's check of the generalised loop on fixed windows. The commit as information. No digest of code |
| 3 | A checkpoint of stage D | Its format name (version 1: `ts-multi-checkpoint-v1`; version 2: `ts-multi-checkpoint-v2`); the identity of its start (version 1: the chosen round of stage C, whose identity holds the base's; version 2: version 1's chosen round); the set of procedure masks; the shape of the traffic attention and of stage D's token part (D152) and, in version 2, of its module (D153), each with its format name; the seed; the spans and their counts, the kinds and c_min; the windows of each round by their anchors, their commanded flights and their shifts |
| 4 | A campaign | `ts-multi-train-v1` (version 1), `ts-multi-train-v2` (version 2); its settings in the campaign's record; a resume requires them the same |

---

## 8 Values

| Item | Value | Source |
|---|---|---|
| Reward of an aircraft | D30: 1; 0.9ⁿ after n go-arounds; 0 otherwise and for a loss it answers for | D140 |
| Reward of a window | The sum over its commanded aircraft | D141 |
| Advantage | The window's reward less the mean of its group; on the varied aircraft's rows after the branch point | D142 |
| Varied aircraft | Each aircraft with r < 1 and the commanded aircraft of the losses they answer for | D143 |
| Branch points | The varied aircraft's first predicted step and the points of the window's grid after it (the anchor's first predicted step and every 120 s after it, post-training D170's default), before its event; K = 8 | D143, post-training D37, D94 |
| Branch points found backward (a setting) | From each varied aircraft's event back in 32 s steps, at most 8 points; avoided = through the event + 60 s, v answering for no loss; the decision point, one 16 s after it and the latest point avoiding with no go-around (B) give groups; with `majority`, in place of B the first point at which at least 5 of 8 avoid | D182, D192 |
| Window | An anchor and every flight with a sentence whose row 0 is within L after it; L = 5, 10, 20 min mixed, one L a batch; real and compressed, c_min = 0.8; 1,000 windows of each kind a round, in equal parts of the spans | D146 |
| Loss | Post-training §9 item 10: ε = 0.2; pull to the base 0.04; data term 1 | §4 |
| Most go-arounds of an aircraft | 2 | Post-training D91 |
| Speed-word mask | Post-training's; a commanded leader's target is its present speed along its course | D148 |
| Token of another commanded aircraft | Stage C's edge features and motion; the flags commanded and silent; its words in force (altitude, angle, speed, heading) | D152 |
| A loss that only a recorded aircraft answers for | The commanded aircraft answers when the records kept their separation at that step | D145 |
| Version 2's module | An attention at each layer over the other commanded aircraft of the window at the same tick; starts at zero | D153 |

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
- The cost of a round of C10's settings: round 0 took 67 min with three speaking workers (stage C's log §26), the
  speaking bound by the CPU (outline D138). A window of stage D costs more than its count of aircraft: each
  continuation flies the whole window (D141, D142), so its speaking grows about with the square of its commanded
  aircraft (stage D's log §15; O21).

### 9.3 What this design takes from the archived work and what it leaves

| Taken | Left |
|---|---|
| The failed aircraft stays in the scene (D144) | The joint encoding of every aircraft at every step, with a vector of width 192 for each pair (version 2 reads only the other commanded aircraft's hidden states at the same tick, D153) |
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
  (`credit.varied`, `credit.branch_points`), the token part of D152 (`tokens`), version 2's module and its window
  samples (D153). It imports `instructions/`, the names of `post/` in the post-training's
  public interface (§6.1) and the names of `prior/` in prior §7; it imports neither `autopilot/` nor any runner
  (`experiments/`). `post/` never imports `multi/`.
- The runners of stage D (`experiments/multi_*.py`): `multi_windows` (the census, MC1), `multi_train` (the campaign,
  through the post-training's campaign steps, post-training §9 item 11), `multi_validation`, `multi_profile` (the memory measure, D179), `multi_speed` (D181, to be built),
  `multi_training_export` (written by fronter, frontend F4). They import from `autopilot/` only the names of vocabulary §6, from `prior/` only those of
  prior §7, from `post/` and stage C's runners only those of post-training §9, and `multi/`.
- The window loop stays one class (`experiments/post_window_loop.py`), generalised (post-training §9 item 8): no second loop.

**Formats** (new names, principle 8): `multi-windows-census-v1`, the token part `multi-commanded-tokens-v1` (D152),
`ts-multi-train-v1`, `ts-multi-checkpoint-v1` (version 1), version 2's module `multi-joint-attention-v1`,
`ts-multi-train-v2`, `ts-multi-checkpoint-v2` (D153), the Training sets' index and sample formats (frontend §7).

**Size of version 2, estimated** (D153): the module and the window samples about 250 lines in `multi/`, the
sentences on the loop's ticks about 80 lines in `prior/` and `experiments/prior_speaking_loop.py`, stage C's samples
of several sentences about 70 lines; tests about 400 lines.

---

## 11 Milestones

The rules of outline §5 apply: stage D's implementer on `dev-multi-control` (rule 1), a review before each commit (root
`CLAUDE.md`, "Code review"), light tests while a campaign runs (only the changed modules' tests; the full suite at
the big commit points).

MC0–MC5 are done (§0.3); their specifications and checks are in stage D's log.

**MC6. The formal campaign** from a run worktree at the commit that the user merged (outline D163), its start by D164,
its intent in `docs/experiments/intents.json` before its publication; the validation readout of the chosen round.

**MC11. The speed of a round** (D181, D183, D186; outline D187), beside MC6 (code only while MC6 holds the GPU). First items 55, 58, 59 and 60
(done), then D183's cache, each
with their check bit for bit on fixed inputs of stages C and D (a round's groups, its samples and its readout against
the code before them), and item 56's runner tested on a smoke batch. Then, on the user's word (after MC6, or at a
round's boundary of MC6, which may resume on items 58–60), item 56's measure of one batch, before and after D183; item 57 only if it shows
the continuations set the peak, with its check. After their checks D186's fast forms and B14's replace the code before them, and the masks' modes are deleted (outline D187). The post-training profile's `executor_step` part that reads 0.0 s
(`post_profile.PARTS`) goes to `docs/code-health-followups.md`. Then D193: (3) the rows no longer flown leave the joint loop, with its check; (1)'s check on real windows and its time once the GPU is free; (4) the CPU batches' host share and their levels.

**MC12. Branch points found backward** (D182), after MC11's items 58–60 are merged. `MultiSettings.branching`
(default `grid`); the search in `multi/` on the shared window loop (`WindowLoop.copy`, `finish`, `samples`; post-training
§9 items 8 and 9, stage C's behaviour unchanged, D149); the second pass's copies at the candidate points held in the
host's memory; each group's point depth in `round.json`; the readouts' go-arounds per aircraft and the points' depths
by span. How the continuations of points at different ticks are batched is stage D's reading, built and stated in its
report. Tests: with `grid` a round is the code's before, bit for bit; the search's points, the stop, the
16 s point, point B (and B at the decision point: one group) and the bound on a synthetic window whose event is avoided first at a known point; a point not chosen flies
to the event + 60 s only; a decision group's W is D141's; a resume compares the setting. Then `multi_speed` on one
batch of each setting (the groups that give samples, the continuations flown to the end, the time) before any
campaign uses it, the campaign the user's choice.

**MC13. The majority point** (D192; done, 172bc07d), on the user's word: `branching = majority`, each group's kind in the batch's record, the searches that found a majority point, D192's checks; `grid` and `backward` bit for bit; a resume compares the setting.

**MC7. The Training view of stage D** (frontend §5, D156): stage C's window view with several commanded aircraft,
built by fronter (frontend F2–F4), not by stage D's implementer. Stage D gives it what the export reads: the windows
of a campaign with each commanded aircraft's sentence of each round, its reward, its silence and the losses by pair,
in the files of MC4 and MC6, named in this document (§7, §10). A change of those files is written here first. A set
of listed windows (post-training D176) takes stage D's selection windows by span and its readout's numbers (D166
(34)). The losses that no aircraft answers for come to the readout and to the export from one generator and one
helper of `multi/` (frontend D177 (12), D178 (6)).

**MC8. Version 2's code** (§6.4, D153), after MC6. First Claude writes its interface changes into prior §7 items 4
and 7 and post-training §9 items 9 and 10. Then: the sentences of a loop on its ticks (stage B), stage C's samples of
several sentences (stage C), the module and the window samples (`multi/`). Tests: every join tick 0 or one sentence
gives today's batches, log-probabilities, samples and loss, bit for bit; the module at zero gives version 1's words,
bit for bit; a window sample's log-probability of each aircraft's rows equals the speaker's records of them (to the
tolerance of post-training §6.4); an absent aircraft is never a key; no row reads a word of its own tick.

**MC9. Version 2's memory measure and campaign**: the memory measure (as MC5, D179), the user's settings, the formal
campaign from version 1's chosen round (D164; a run worktree, outline D163), the validation readout of its chosen round.

**MC10. Close of stage D.** The full ts suite; `docs/reference/runners.md`; the code index for §10; the report to the
user. The user merges (outline §5 rule 11).
