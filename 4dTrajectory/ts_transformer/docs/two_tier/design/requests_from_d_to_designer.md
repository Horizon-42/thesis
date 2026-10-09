# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62; D181–D186: 63–70; the interfaces and D182:
71–77; D188: 78–79), item 23 (it asked for 6 and 7), item 20 (built: item 40) and the names of MC0–MC4 (written into
the interfaces).

State: 2026-10-09 (night): items 78–79 decided (D188) and deleted; D188 and D189 built on `dev-multi-control`
(stage D's log §19). Items 80–84 come from them (80 measured: the workers the joint loop leaves room for; 81–84
readings); the others are MC0–MC5's readings still open.

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

## D188, D189 (stage D's log §19; 2026-10-09, night)

80. **The joint loop's memory sets the workers** (D188 (1), measured by the profile at §18's settings,
    `outputs/POOLED/multi/profile_d189_20261009`, and by part, stage D's log §19). A batch is faster (batch 0 111.8 →
    55.3 s, 23 116.8 → 64.9 s, 86 80.8 → 61.0 s), but a worker's GPU peak rose from 1.6 to 2.21 GiB (batch 23: 2.07
    GiB reserved, 1.77 allocated): with D179's margin 1.3, 2 workers fit where 3 did (3 need 8.7 GiB against 6.8). At
    a round's batches of MC6 (23 / 63 / 138 by span) the speaking is about 13,800 worker-seconds against §18's 21,100:
    on 2 workers about 1.9 h, the time 3 workers took before.

    By part (one process on the GPU, at each batch's step of the most allocated; GiB): the joint loop's cache is most
    of it — batch 0: 0.40 of 0.63 allocated, 23: 1.10 of 1.44, 86: 0.96 of 1.24; the executor and words 0.02 / 0.12 /
    0.10; a rebuild's peak 0.79 / 1.77 / 1.53. Of the cache, the rows of history 37 / 26 / 42 %, the rows written but
    not present 37 / 60 / 44 % (a line is aligned to the joint loop's clock: a point from an early tick is padded to
    the latest part's length, and an aircraft not yet joined writes rows not present) and the room of a step (D183's
    64 rows, after each rebuild) 26 / 14 / 14 %. The K copies of one point are the same, bit for bit, up to where they
    part — their history and the padding before it: one copy of that front instead of K would free 0.24 / 0.76 / 0.75
    (61–79 % of the cache), of which history alone 0.12 / 0.19 / 0.36.

    Proposals, by what each frees (the user chooses; none changes a number but (c)'s kernel on the GPU):
    (a) the room after a rebuild of the joint loop: the steps to its next rebuild, not 64 rows (rebuilds come every
        few steps): about 0.10–0.16 GiB, the room no row reads (the cache grows by D183's step if it outlives it);
    (b) a campaign setting that bounds the rows the joint loop holds, a point waiting for room (D181 (57)'s
        `continuation_rows`, never built; default "all", today's behaviour): the memory in proportion, at the price of
        steps;
    (c) the shared prefix (O20 (62); the condition it was deferred on — that the cache sets the peak — now holds): the
        K copies of a point read one front, each its own rows after the point; frees most of the cache (above); a change
        of the speaker's cache and its attention (stage B), two pieces of attention merged — on the GPU a rounding,
        checked by D181 (57)'s bound. Estimate with (a) and (c): a worker's peak about 1.2–1.5 GiB, 3–4 workers.
    Today, without them, the next campaign runs 2 workers.
81. **D189 (0)'s "window's end"** (reading). A window has no end of its own: its aircraft fly until the executor ends
    them, and a batch's windows are said together (a window's halted rows are said while another's fly). Read: a
    commanded aircraft's landings are those from 30 min and one Δ before its window's row 0 to the latest a row of its
    loop is said (`WindowLoop` `reach`: each aircraft's first predicted step and the executor's time limit with every
    go-around's, from its window's row 0, the latest of the loop, one Δ more; a row's time is its flight's, the
    window's to rounding), and its own (`LandingIndex.counted`, `post.landings.commanded_landings`). Every count is the
    same (window_behaviour on stage C's and D's windows, a test that every count read lies in the cut).
82. **D189 (1)'s assertion of one airport** (reading). Not added as code: `post.scene.Window` refuses a commanded
    aircraft of another airport, and each row's geometry and finals are its window's by construction
    (`WindowLoop.__init__`), so an assertion in the loop could never bind (root `CLAUDE.md`); `_of_row`'s docstring
    names the rule it rests on.
83. **D189 (2)'s masks of an aircraft not said** (reading). The later columns' procedure and caller masks are computed
    once a row under the runway as said (an aircraft not said keeps its own), where the old code read the raw word
    drawn for an aircraft not said; every class is permitted to such an aircraft, so the records are the same, and
    the row's `blocked` record reads the same masks as before. The check that an aircraft has a permitted word (D62) is
    read at the row's end, before the executor flies it: a row that fails it is refused before any change, but a
    later column's refusal can be the message raised. The row's records come back in four transfers at its end.
84. **The search's counts in a batch's record** (reading). `search_steps` and `search_rows` count the joint loop's steps
    and their rows (D188 (1)): they are not comparable with §18's (one point of one window a loop); `search_rows` is
    the same work (on MC6's batches 0, 23, 86 equal to §18's).
