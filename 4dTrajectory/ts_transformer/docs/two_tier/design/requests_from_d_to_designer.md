# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62; D181–D186: 63–70; the interfaces and D182:
71–77; D188: 78–79; D190: 80–84; D190, D191: 85, 87, 88; D191–D193 and the log: 86, 90–98), item 23 (it asked for 6 and
7), item 20 (built: item 40) and the names of MC0–MC4 (written into the interfaces).

State: 2026-10-11 (01:15): arm A (`multi_train_majority_20261010`, D192, 2 rounds) runs; it stopped once in round 0
on a defect of S6 with join ticks, fixed (14e53df6, merged) and resumed with its done batches kept (item 104). D194
(2), (3) built on `dev-speed` (ce886c4a, 9bdbe6c5; reviewed, the second round clean), not merged; they go into arm B.
89 is a measure kept from D190; 99–103 are of the night before; 104–107 are new.

## MC0 · stage A (vocabulary §6 item 5, D150)

1. **The executor's clock.** Multi-aircraft control §6.3 item 1 gives a flight's start cycle as "(join tick + s) × the
   cycles of a Δ row". Read: the loop's executor starts at tick s (the first predicted step of a flight of join tick 0,
   `Loop.steps` 0), so a flight's start cycle in the executor is j × the cycles of a Δ row; its first predicted step is
   the loop's tick j + s, as the design says. With every join tick 0 the executor's batch is the single-aircraft batch
   (no start cycle), so the loop is today's, bit for bit (`autopilot/start.py` module docstring, `Loop.started`).

2. **"Not heard" includes the column values.** A done or halted flight's words are not read by the grammar or the bound
   of go-arounds, but their values are still checked against their columns (D80). A flight that has not started is
   read not at all: neither the grammar, nor the bound, nor the values (its words are the caller's filler).

## MC0 · stage B (prior §7 items 2, 3, 7, D150)

3. **Which phase a loop with join ticks has.** `SpeakingLoop.observe` stays the phase of a loop whose join ticks are all
   0 (as the design says); a loop with any join tick above 0 has no `observe` phase: `step` runs from tick 0, each
   flight absent, observed or said by its own clock. A tick before tick s holds no said flight and does not step the
   executor. The speaking loop reads the join ticks from the start's `Loop` (one definition), not from its caller.

4. **The speaker's records of a row of several roles.** The records of a row (`forbidden`, the probability and the
   permission of "go-around", `procedure_blocked`, `drawn_probability`) are kept for every row in which some aircraft
   is said, `[B, …]` each, with `said_rows` (which aircraft were said in it); `Speaker.permitted` gives each aircraft
   its own said rows, first (padded as `Permitted.join` pads). As today, a done or ended flight that is still in the
   batch is said on (its later rows are recorded, not its sentence's). A flight not said in a row draws a word that no
   one keeps from every class of the model (no mask), so a flight that has not joined can never make a row refuse.

5. **An absent flight's inputs.** An aircraft that has not joined gets a row that is not present, computed from finite
   states (its observed row 0), so that its keys and values in the cache are finite (a masked key with a value that
   is not a number would still give a NaN through the attention's product).

## MC0 · stage C (post-training §9 items 1, 2, 3, 7, 8)

8. **A commanded aircraft that has ended leaves its window's scene.** Built so: the judged set and the other aircraft
   of a row hold only commanded aircraft that have joined and are still flown (a silent one included); one the executor
   has ended (landed, crossed, timed out) is in no later scene. (Linked with item 7.)

9. **A silent aircraft's landing adds nothing** (D147 items 2–3). Built so: a commanded aircraft that is silent (it
   answered for a loss, D144) and then ends `landed` adds no landing to its window's other aircraft; its reward is 0.
   Proposal: confirm (its outcome in the loop is its loss).

10. **The order of a row's other aircraft.** Built so: the recorded aircraft (the scene's order), then the window's
    other commanded aircraft (their order in the window); with one commanded aircraft, stage C's order.

11. **An aircraft in its observed rows is judged as a recorded one** (D145's table): it answers for nothing, the said
    aircraft answers when the rules make it responsible. In the synthetic test, a copy joining on the anchor's path
    makes the anchor silent at the copy's row 0 (neither established: both responsible).

## MC0 · stage C, items 9, 11, 12

12. **The rules of branch training are the caller's** (post-training §9 item 9): `experiments.post_branches.Rules`
    (each aircraft's first and continuation numbers, the window's reward, spoken again, the varied aircraft with their
    branch points as ticks of the window's grid); `stage_c_rules` are stage C's. A group's branch point is in the
    varied aircraft's own rows (`Group.branch`), its sentences count up to that aircraft's event (`Sentence.until`).
    Stage D's own rules (D142's streams, D143's varied aircraft) are MC3's, in `multi/credit`.

13. **The campaign's skeleton** (post-training §9 item 11): `post_train.Stage` and `STAGE_C`; the speaking workers
    (`Speakers`) speak the stage's batches; stage D's own speaking (its rules, its loop options) is its stage's
    `speak_batch` (MC4). The start of a later stage is the one start function of §9 item 12 (D164).

## MC1 · the census

14. **The kind of a window of stage D is stage D's own** (`multi.windows.Drawn`: real or compressed, with its c); the
    window it holds is a real window of stage C's code (`post.scene.REAL`) with its commanded aircraft joined, so stage
    C's kinds (real, A, D, B) stay as they are.

15. **`multi/` imports the separation judge's types and rules** (`inference.separation`, `inference.runway_schedule`)
    as `post/` does; §10 names `instructions/`, `post/` and `prior/` only. Proposal: add them to §10.

16. **The baseline's split of D145** (review S1, built): a loss of the baseline that only the recorded aircraft answers
    for is judged again on the records at that step: `records_kept` (the loop charges the commanded aircraft) when the
    records did not lose that pair there, else `recorded_only`. A commanded aircraft is never over its threshold, on the
    records as in the loop (with item 7).

17. **An aircraft in its observed rows counts as commanded in the census** (review S3): the loop charges it nothing
    (D145), so the census's commanded pairs include losses in the first 16 s of a later aircraft that the loop would
    not charge. Proposal: leave as is (after the left-out rule, only losses ending within those 16 s remain), or count
    them as recorded.

## MC2, MC3 · the window loop and the credit

24. **"The loss that v is in"** (D143, v's event): read as the earliest loss of the first sentence in which v is either
    aircraft — its own, or one that another commanded aircraft answered for against v. A varied aircraft whose event is
    at or before its first predicted step has no branch point (no group). An aircraft that answers two losses at one
    step records one other aircraft (the loop's first), so the second partner is not varied (the review, S3: rare).

26. **Stage D's numbers** (§3.3): `default_rng([seed, round, place, member])` for an aircraft's first sentence and
    `default_rng([seed, round, place, member, tick, k])` for a continuation of the varied aircraft `member` at tick
    `tick` (`multi.credit`).

## MC4 · the round's stage

27. **The campaign's round is the stage's throughout** (post-training §9 item 11): `Stage` also gives its batch speaker,
    its batch reader and its token part's width (which the pass's samples read); `Stage.selection` takes the split;
    the speaking workers run the stage's parts. Stage C's campaign is the campaign as before (D149's campaign digests
    unchanged). Still stage C's only: the memory measure of the pass (`pass_memory_of`, `post_profile`); stage D's comes
    with MC5.

28. **The losses by pair in stage D's readouts** (§5 item 4; reading for MC4): the loop records only the losses that
    an aircraft answers for. A readout's counts by pair (including `recorded_only`) are judged again on the flown states
    of the window's commanded aircraft with the census's machinery (`multi.census.window_losses`, positions from the
    loop's states and words), the same rule as the census's baseline. Stage C's loop is not changed for it.

## MC4 · the windows of several spans (D146, 6cc49ad1)

37. **Equal parts of the spans** (D146): each kind's count of a round divided by the spans, the remainder one each to
    the first spans in the order of the setting (`multi_train.span_counts`: 1,000 over 5, 10 and 20 min is 334, 333,
    333).
38. **The span of a window** (D146: "drawn with the round's numbers"): for each anchor in the round's permutation and
    each kind still short, drawn uniformly with the round's numbers among the spans of which that kind is still short,
    so each span's count is reached while anchors remain; the real and the compressed window of one anchor draw their
    spans each (they may differ). What the draw counts (drawn, left out, without a later aircraft, the shortfall) is
    recorded by kind and span.
39. **A window holds its span** (post-training §9 item 1): `post.scene.Window.span_s`, 0 for stage C's windows (D146:
    L = 0 is stage C's window), set by `multi.windows.Anchors.window_of`; a batch and a window's record read it
    (`multi_train.window_record` gains `span_s`).
40. **A batch is sized by its rows** (item 20, built; MC5's "a batch sized by its rows"): `post_train.batches(windows,
    size)` holds at most `size` commanded aircraft of windows of one span, each flight once; stage C's windows have one
    row and span 0, so its batches are as before (D149). A window of more rows than `size` is a batch alone. The
    stage gives its batches (`Stage.batches`: stage C's of `batch_windows`; stage D's setting `batch_rows`). The
    memory measure before a campaign (`_measure`, O15) speaks the first batch of each span and the batch of the most
    rows (`post_train.measured_batches`; stage C's: its first batch, as before), their largest peaks taken.
41. **The select set by span** (§5 item 2, D166 item 33): each airport's anchors in one order drawn once with the seed,
    used for every span, so the same anchors stand in each span where none is left out (the spans read on the same
    flights). The readouts (and the validation readout) by airport, span and kind, each with its `all`; an airport's
    `reward_mean` (the campaign's log) is its `all` of both; the validation readout's coverage gives the windows read
    of each span.
42. **D166 item 19 in stage D only**: `multi_train.require_finished`, in `main` before anything is opened and in stage
    D's start under a formal context; `post_train.source_campaign` (stage C's start from a round, C16) is unchanged.
43. **The identity** (§7 row 3): `spans_s`, `per_kind` and `per_span` (each kind's count of each span) in place of
    one L, and each round's windows with their spans; the campaign's settings hold `spans_s` and `batch_rows`.

## MC5 · the profile (bd6dd1be)

44. **The worker's memory measure speaks with the round's model** (post-training O15; the review of MC5): before,
    `Speakers.measure` spoke with the stage's start model (the base with zero-output traffic modules), so a campaign
    started from a round of another (C16, stage D's D164) measured other words than its own. Now the round's model goes
    to the worker (`Speakers.measure(round, directory, model)`, `post_train.campaign_model`); a new stage C campaign
    without a start measures as before.
45. **The size of the select set** (§5 item 2) is read from round 0's spread on two draws of the same windows with
    independent numbers: an upper bound of the noise of comparing two rounds, which share their numbers. Over all spans
    the unit is an anchor (its windows of the spans are nested and share their aircraft); within a span, windows of
    other anchors may still share flights in a dense select set (stated).

## D190, D191 (stage D's log §20, §21; 2026-10-10)

89. **The two ways' time** (measured, the profile at §18's settings on 7893d787). Copied: batches 0 / 23 / 86 in 56.8 /
    68.0 / 63.0 s; merged: 55.5 / 63.5 / 59.9 s (D189: 55.3 / 64.9 / 61.0). The campaign of item 90 reads `merged`.

## D193 on `dev-multi-s6` (stage D's log §22; 2026-10-10 night)

The code says the details; read the two commits.

99. **Each flight's own permitted record** (reading; changes D181 (63) at the loop's level). With S6 a flight that has
    left the loop has no rows in the speaker, so a record padded to the most rows among the flights asked would depend
    on which flights left. `SpeakingLoop.permitted(flights)` now gives a list, one record a flight: its sentence's rows
    (`len` of its said words) and its airport's classes (`column_words`). The loss finds a record's row by its time and
    reads a padded class as forbidden, so the loss is the same: checked on MC6's round 5 groups, 144 sentences, 143 cut
    shorter, the log-probabilities equal bit for bit. `Speaker.permitted(indices)` (prior §7 item 3) is unchanged and
    keeps D181 (63)'s padding. Stage C's pinned digests of D149 hash these records: two of them ("round real", the
    batch scenario) changed, and d5cc5d6a with the record cut alone gives the same two, so they were written again
    (6c5269af). Please write this into D193 (3) or D181 (63).
100. **New and removed public names** (for prior §7 item 7, post-training §9 and the key-code index). `SpeakingLoop`:
    `held` (the flights still in the engines), `left` (each left flight's frozen `Left` record), `inside` (the held
    flights' places), `outcome(flight)`, `landings(flight)`, `in_force(flight)`, `permitted(flights)` (item 99); a left
    flight's states, sentences and outcome come from its record. `WindowLoop`: `landings` removed (read
    `speaking.landings(flight)`), `results(rows=None)` per row. A loop with no flight held has no speaker or executor.
    The line numbers of `SpeakingLoop` and `SpeakingLoop.step` in prior §12's index moved.
101. **S4's readings** (D193 (4), 22fd1031). (a) The cap reads the worker's own memory (`_own_memory`, as O15's
    sampler) every `HOST_CAP_SAMPLE_S` = 1 s, a constant: one `smaps_rollup` read took 12–53 ms in a worker of 2.4 GB at
    a load of 35, so O15's 0.2 s would cost 6–25 % of a core; a batch can grow for 1 s past its cap, and the stop waits
    for the torch operation in progress. (b) The error is raised in the worker's main thread by a signal (SIGUSR1, the
    worker's handler), so it comes where the batch is and the worker catches it as CUDA's out of memory. (c) A capped
    CPU worker gives the heap's free pages back after each batch (glibc `malloc_trim`): without it the next batch
    starts past its cap (tested). (d) The host's part for the CPU workers is the free host less the GPU workers' (at
    their peak with the margin, and their series); a CPU worker holds between batches its profile's `now`, the reader's
    model and the round's series. (e) The GPU's levels are spoken before the CPU's, one device after the other.
102. **For the user: the measures of tonight** (stage D's log §22; no criterion). Batch 86 of MC6's round 0 on the
    GPU: 62.1 s before S6, 53.9 s with S6, 48.8 s with S6 and the executor on the CPU (S1); on the CPU 68.1 → 51.0 s.
    The rule of a mixed launch (`--cpu-batches`, `--cpu-period`) was set by the speeds before S1 and S6 and is to be set
    again at the next launch. The traffic-only arm lands 82.4 / 82.5 %, inside MC6's rounds; R5 finds no sign that
    the first rows after a point decide.
103. **D192 as built** (172bc07d; please check against the row). (a) The count is `continuations // 2 + 1`, more than
    half: 5 of D192's 8. (b) The kinds go into the record under `majority` only (`group_kinds`: decision, fine, majority),
    not into backward's: the row says "a backward batch's record adds each group's kind", but a new key in backward's
    record would change the records of MC6's campaign, which is to stay resumable; backward's record keys are pinned by
    a test. (c) "The searches that found a majority point" are the length of `majority_depths_s` (each its span and
    depth), the decision point counted when it is the majority point. (d) `branching` is a profiled setting, so a
    campaign with `majority` needs its own profile.

## A fix and the speed-ups of 2026-10-11 (D194 proposed)

104. **A speaker keeps a row of records for each tick after its first said row** (fix, 14e53df6, merged; prior §7
    item 3, D193 (3)). Arm A stopped in round 0: a joint loop of the backward search was rebuilt with the said flights
    of a window left (S6), its other flight still in its observed rows; at the next tick no aircraft was said,
    `Speaker.say` kept no row of records, and the next `SpeakingLoop.together` refused the loop (its rows said ≠ its
    ticks). Now the records (`forbidden` … `drawn_probability`, `said_rows`) have a row for every row `say` says from the
    first in which an aircraft is said, a row with no aircraft said included; the rows before it stay unrecorded, as
    before. A flight's own records (`permitted`, the sentences) read its own said rows only: unchanged. The check on
    fixed inputs (R68 on arm A's campaign, the code before and after) gave every digest equal, so A's 174 done
    batches were kept (D185). Please write this into prior §7 item 3's records ("every row in which some aircraft is
    said" → "every row from the first in which some aircraft is said").
105. **D194 (2): a round's batches longest span first** (proposal; `dev-speed` ce886c4a, the user's choice of
    2026-10-11: speed-ups that change no group). `speak_round` speaks the batches not done in the order of their span,
    longest first, those of one span by index (stage C's windows have span 0: by index, as before). Why: the batches
    were sent by index, which is by span ascending, so the 1200 s batches came last and the round's last ~10 min ran on
    a few workers (MC6 round 4: 7 batches in its last 10 min, 33–47 in each 10 min before). The record still sums the
    batches in batch order, a batch's device is still its rule's (by index, D191), and the pass reads the groups by
    batch index, so nothing learned changes. The batches deferred to a level (D191) are still spoken last, by index.
106. **D194 (3): a majority point out of reach ends its continuations early** (proposal; ce886c4a, 9bdbe6c5). In a
    search for the majority point (D192), a point tried behind the decision point leaves the joint loop as soon as
    fewer of its continuations are left than the majority (`multi.backward.out_of_reach`: K − failed < majority; 5 of
    8: once 4 have failed). A failure is counted only where it stays to the judged tick (`_failed`): v answered for a
    loss; for a loss event, the event's partner answered for one with v; for an end, v ended with the event's
    outcome. A go-around said is read at the judged tick only (it would need v's sentence read at every step), so a
    go-around event stops later, never wrongly. The search's answers (the points chosen, the groups, their kinds and
    depths, the counts of points and continuations) are those of the search without the stop; only the steps, the
    rows and the seconds of the search fall. The decision point and the fine point are never stopped early. Tested
    (`test_multi_backward.py`: the rule; `_failed` against `_trials` on one state for each kind of event; on the
    synthetic window the same answers with and without the stop). `continuations_to_event` keeps counting each
    continuation of a point tried, also one that stopped before the event + 60 s.
107. **D194 (4): arm B with 8 CPU workers** (a launch setting; no code). The profiles of D192 (`profile_d192_gpu`,
    `profile_d192_cpu`) fit 3 GPU + 8 CPU workers; A runs with 6. A batch's device is its rule's (`--cpu-batches 7
    --cpu-period 10`, by index), so the number of workers changes no group. Expected: about 10 % faster speaking (a
    CPU worker slowed from 0.61 to 0.53 batches a minute from 4 to 6 workers; 8 of the 20 cores are performance cores).
