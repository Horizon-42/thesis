# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62; D181–D186: 63–70), item 23 (it asked for 6 and 7), item 20 (built: item 40) and the names of MC0–MC4
(written into the interfaces).

State: 2026-10-09 (evening): items 63–70 decided (D181–D186) and deleted. On `dev-multi-control`: item 69, D183, D186,
D185 (C27), MC12 (D182) and the designer's orders A–D (D187) built; the readings of that work are items 71–77 (stage D's
log §17).

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

## MC11, MC12 · the speed of a round and the points found backward (D182, D183, D185, D186, D187; 2026-10-09)

71. **The behaviour check is a runner** (`experiments/window_behaviour.py`, R68; D181's one-off script made formal, as
    a readout used more than once is a runner). From a stage C campaign (its branch method; landed and value refused
    by name) or a stage D one, on the CPU with one thread: the first batch of each span of a round's draw spoken by the
    stage's own batch speaker and a few readout windows of each span, digested; `--against` another record names what
    differs (a backward round's depths and counts too). It is also D185's check before a resume on changed code.
    Proposal: name it in post-training §9 item 11 beside D185.
72. **The cache's room** (D183 (2), (3), order B): a copy's cache is allocated once with its rows written and 64 rows
    (`prior.model.CACHE_STEP_ROWS`, `Past.taken`: the rows selected into its front by `torch.index_select(out=)`); a
    line that outgrows its room grows by as many 64-row steps as its rows need at once. On another device (D182's host
    copies) the selection is made where the rows are and then moved: an intermediate there that cannot be avoided.
    Proposal: write the constant and `Past.taken` into prior §7 item 1.
73. **An index of landings in time, then flight order** (D186 (1)): `LandingIndex` refuses landings out of (time,
    flight key) order (every builder already sorted so; the insert of D186 (1) equals the old sort only then). A stage
    B contract, refused by name. Proposal: write it into prior §7 item 2.
74. **The names after order A** (D187 rewritten): one form of each — `grammar.column_mask` is the batched mask (the
    readable one deleted, its test's reference `apply` itself); `ProcedureMasks(finals, words)`, `Speaker(…)`,
    `SpeakingLoop(…)` and `LoopRows(…)` take no mode; `PER_AIRCRAFT`, `BATCH`, `MASK_MODES`, `require_same_masks`
    and `check_rows` are gone; `procedure._FinalArrays` (was `_Gathered`). Vectorised numpy gave the same bits as single
    rows on this host (sin, cos, arctan2, hypot, arcsinh, log1p over 200,000 values) and the rows' inputs of many
    aircraft equal each one's alone (the test), so no bound was needed. A test now pins the masks' arrays to `Final`'s
    definition (`Final.inside`, `Final.edge_m`, `entry_low_m`), which the per-aircraft mode used to. Proposal: the names
    into vocabulary §6 item 2 and prior §7 items 2, 7 (the review found prior §7 still listing ``masks``,
    `PER_AIRCRAFT`, `BATCH`, `MASK_MODES` and the per-process check).
75. **A batch's record** (D185): `record_<k>.json` holds the batch's own record, its windows' records, the launch (time,
    commit, whether the tree was dirty: information) and the kind of device; files are written as `<stem>.tmp` and
    renamed (`groups_<k>.tmp`: torch names a file's inner folder after its stem, so the bytes are those of a file
    written in place). `--restart-round` is recorded in each resume's entry of `campaign.json`. New names:
    `post_train.save_batch_file`, `batch_record_path`, `spoken_batch`, `done_batches`, `launch_of`;
    `Stage.speak(…, launch=)`, `Speakers.speak(…, batches, launch)`, `open_campaign(…, restart_round=False)`.
    Proposal: write them into post-training §9 item 11.
76. **The copy chain takes a device** (D182's copies on the host): `WindowLoop.copy`, `SpeakingLoop.copy`,
    `Speaker.copy`, `LoopRows.select`, `autopilot.start.Loop.copy`, `Executor.take` and `Spoken.take` take an optional
    `device` (None: the copied one's, as before). Proposal: name it in vocabulary §6 item 5, prior §7 items 1, 2, 7 and
    post-training §9 item 8.
77. **Readings of D182 built in MC12** (`multi/backward.py`, `experiments/post_branches.backward_round`):
    (a) the event of a landing after a go-around is its first go-around when that is earlier than D143's event (a loss
    before the go-around stays the event); (b) for an event that is no loss and no go-around, a continuation in which v
    answers for a loss instead ends with another outcome and so "avoids" the event — the letter of D182; the user may
    want such a continuation not to count; (c) the first point is 32 s before the event (the event's own tick is never
    tried); (d) the fine point's state is the decision point's held copy flown 16 s on every aircraft's first-sentence
    numbers; (e) the batching: one point of one window at a time, its K continuations one batch (the device holds one
    point's continuations); (f) a window whose second pass differed (D94) is not searched; (g) the record of a
    backward batch adds each group's span and depth (`group_depths_s`), the searches and those that found a decision
    point, and the continuations flown to the event + 60 s only and to the end; a grid batch's record is unchanged;
    (h) the readout's go-arounds per aircraft are its `go_arounds` / `aircraft` of each span (no new key, so a grid
    round's `round.json` stays the one before); (i) `branching` is a profiled setting (a profile of the grid does not
    size a backward campaign; a profile without it reads as the grid); (j) at Δ = 8 s, 60 s is no whole number of
    ticks: refused, but only at the first search, not when the campaign opens. Proposal: confirm each, or decide (b).
