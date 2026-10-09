# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62), item 23 (it asked for 6 and 7), item 20 (built: item 40) and the names of MC0–MC4
(written into the interfaces).

State: 2026-10-09: items 54–62 decided (D181, O20) and deleted. MC11's code (D181 items 55, 58–60, 56) is built on
`dev-multi-control`; items 63–65 are readings of it for the designer. Item 66 is the user's proposal of 2026-10-09 (a
new rule of branch training) item 67 the speaker's cache allocated as needed and item 68 a round resumed by its batches; written here at
the user's word. MC6 was stopped by the user inside its round 0 (2026-10-09 14:31, 134 of 224 batches; "太慢而且不一定有用").

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

## MC11 · the speed of a round (D181)

63. **Stage B's records of chosen aircraft.** Item 58 needs the speaker's records and the loop's sentences of some rows
    only: `Speaker.permitted(indices)` (`prior/speaker.py`) and `SpeakingLoop.sentences(split, flights)`
    (`experiments/prior_speaking_loop.py`) take an optional list (repeats permitted, in its order); without it, every
    aircraft, as before. The records of chosen aircraft are those of every aircraft selected
    (`permitted().select(indices)`), padded to the rows of the aircraft said most among all of them, so a sample's shape
    does not depend on which rows were asked (the test). Proposal: name the argument in prior §7 item 3 (the records)
    and post-training §9 (`WindowLoop.samples(split, rows)`).
64. **The objects frozen only while the workers live.** Item 60: `Speakers` collects and freezes (`gc.freeze`) before
    the fork and unfreezes when it is closed (`Speakers.close`), so a process that opens workers twice (a test, the
    profile beside a campaign) does not keep its earlier objects out of the collector for good. The campaign's process
    holds its context the whole campaign anyway. Proposal: confirm.
65. **What `multi_speed` counts where** (item 56). A part is entered by the window loop's method that opens it (`run`:
    the first pass; `observe`, `step` outside it: the second pass; `copy`, `finish` of the copies: the continuations;
    `samples`: both passes' samples); inside a part the speaker's time (`Speaker.say`, `observe`) is taken net of the
    executor's steps that it calls (`accept`), and the window loop's own time is the rest. The batch's starts of its
    loops and its groups written are `other`. The first part holds the device's warm-up (a few seconds on the GPU, as a
    worker's first batch); the timers' synchronisations slow the GPU a little, as in `model_speed` (D136). Proposal:
    keep, stated in the report.

## A new rule of branch training: the decision point found backward (the user's proposal, 2026-10-09)

66. **Branch backward from each loss, coarse then fine, and stop at the decision point.** The user's proposal, the
    readings the user chose marked. It changes D143's branch points, not D141's W, D142's streams or the windows (D146).
    - *Today* (D143, `multi/credit.points`): a window below its count is spoken again; each varied aircraft is branched
      at its first predicted step and every `BRANCH_EVERY_S` (120 s) of the window's grid before its event, K = 8
      continuations at each point, each flown to the window's end. Points far before the loss give groups whose K
      continuations end alike (all land or all lose): no information (O21).
    - *The rule.* A window whose first pass lands every aircraft gives no sample (as today). Otherwise, for each loss
      of separation of the first pass, the earliest first, and for each varied aircraft of that loss (D143's: the one
      that answers and the commanded aircraft it lost separation with): branch points from the loss's tick backward
      in steps of 32 s (on the window's grid of Δ rows), tried one at a time, the latest first. At each point, K
      continuations as today (the prefix copied: `WindowLoop.copy`; the varied aircraft on its continuation numbers, the
      others on their streams, D142). **The search stops at the first point (going back) where at least one
      continuation avoids that loss** (the user's reading (a), 2026-10-09): the decision point. Then **one 16 s point**
      between it and the next later point tried (coarse then fine, the user 2026-10-09): the decision point to 16 s at
      the cost of one more point. A loss that no point avoids stops at a bound (the varied aircraft's first predicted
      step, or at most N points: the user's to set). Each failure is searched from the first pass's record; the order
      (the earliest first) only orders the work.
    - *Open for the designer (proposals):* "avoids that loss" = the same pair keeps its separation up to the original
      loss's tick, and the varied aircraft is in no loss in that span. Which points give groups: those tried
      (the ones before the decision point, where all K lose, are uninformative and may be dropped from the pass).
    - *Rewards* (the user asked whether they can be improved; proposals, the user decides): (1) a group's continuations
      compared with each other (the group's mean as the baseline), the question being "does this word at this point avoid
      that loss"; (2) a continuation that loses the same separation again is ended there, its outcome decided — a saving
      of time, the reward's meaning unchanged; one that avoids it is flown to the end (a later go-around or loss still
      counts); (3) optional: a continuous term of the separation's margin (the closest distance over the minimum), so a
      group whose K all lose is still graded — an auxiliary term only (risk: flying at the minimum), its weight the
      user's; (4) not proposed: a local reward alone (the loss avoided, the landing ignored): a go-around or a delay
      would "avoid" a loss.
    - *Cost.* The search is serial within a window (a point's K results decide the next), parallel over a batch's
      windows (every window's current point at once). When the decision point is near the loss (G1: most losses 10–20 km
      out, decided a minute or two before), far fewer continuations than today's every-120 s points flown to the end;
      when no point avoids it, more points than today, so the bound matters.
    - *Memory.* Going backward needs the loop's state at each candidate point, saved on a forward second pass. At any
      point the K continuations' peak is today's (it does not depend on the step). The saved states: as many as the
      span over the step, each a copy of the speaker's cache to its tick — kept in the host's memory and moved to the
      device when their point is tried, so the GPU holds one; coarse then fine keeps their number at the 32 s one. A
      tree of shared prefixes (O20's item 62) would make them nearly free, but it is not proposed before item 56's
      measure shows that the cache sets the peak.
    - *Storage.* A group's first sentence and its continuations share their rows up to the point; stored as a tree (a
      node holds the rows after its parent), the groups' files shrink; the loss still reads whole sentences.

## The speaker's cache allocated as needed (the user's word, 2026-10-09)

67. **The cache's room: what is written, a step more, and no second copy.** Claude's estimate (to be confirmed by item
    56's measure, D181): the K continuations' speaker caches are most of a stage D worker's GPU peak. A row of an
    aircraft costs 8 KiB (the base: 4 layers, d_model 256, float32; a key and a value each). The cache is
    `[B, heads, room, head]` a layer (`prior/model.py` `Past`), its room one for the whole batch: at first the latest
    join tick + the observed steps + `FIRST_ROWS` (128, `experiments/prior_speaking_loop.py`), doubled when a row
    outgrows it (`Past.grown`, `Layer.extend`: `max(end, 2 × room)`). A 1200 s window: about 430 rows at first, about
    860 once doubled, so 3.4–6.9 MB an aircraft. At a branch tick of a 1200 s batch (24 aircraft), e.g. 2 windows × 3
    varied aircraft × K 8 × 6 aircraft a window = 288 rows: about 1.0–2.0 GB, against a measured peak of about 2.5 GiB
    (the CUDA context included). Three places where room is held that no row uses:
    - **The copy clones twice.** `Speaker.copy` (`prior/speaker.py`) takes `p.keys[index].clone()`: indexing by a
      tensor already makes a new tensor, so the clone is a second whole copy, alive at once with the first — at a
      branch tick the copies' cache momentarily twice over. Proposal: drop the `.clone()` (the indexed tensor is the
      copy's own storage; `present` likewise).
    - **The copy takes the whole room.** A copy holds the room of the loop it is copied from, rows not yet written
      included. Proposal: a copy keeps the rows written (`rows`) and a step of room after them.
    - **Doubling.** A cache that outgrows its room becomes twice as large; at the moment of growth the old and the new
      are both alive (`torch.cat`: about three times the old). Proposal: grow by a fixed step (e.g. 64 rows, about
      256 s at Δ = 4 s), so the room is at most a step past the rows; more growths (each copies the cache once), each
      smaller.

    **No number changes, on the CPU and the GPU:** the attention reads only the rows written (`Layer.extend`:
    `past.keys[:, :, :end]`, the mask over `:end`); the room never enters a computation. Before use, the check of
    D181 (58–60) on fixed inputs (stage C's and stage D's windows; groups, samples, readout bit for bit), and item 56's
    measure before and after for the gain. Gain (estimate): the continuations' peak lower by about a third to a half,
    so possibly a third GPU worker. The code is stage B's speaker and model (prior §7), shared by stages C and D; the
    designer names the owner. It needs no tree of shared prefixes (O20's item 62) and does not exclude one: the tree
    would also share the copies' prefix, this only stops holding room that no row uses.

## A round resumed by its batches (the user's word, 2026-10-09)

68. **Resume inside a round, from its batches spoken.** Today (post-training D157, `post_train.open_campaign`): a round
    without its checkpoint is moved aside whole (`round_<r>.aborted-<UTC>`) and spoken again from its first batch.
    MC6's round 0 lost 113 batches (about 2.5 h) and then a few more to two bugs on 2026-10-09. What a batch leaves on
    disk is its groups (`groups_<k>.pt`, written by the worker); its record (counts of groups and informative groups,
    the windows spoken again and those that differed, the reward's sum, the faulty steps, the outcomes) stays in the
    campaign's memory until the round's end (`speak_round`), so a stopped round cannot be put together from its files.
    Proposal:
    - a batch's record written beside its groups (`record_<k>.json`), and both written to a temporary name and renamed
      (a stop never leaves half a file);
    - on a resume, a round without its checkpoint is kept: its batches with both files are read back, the others spoken;
      the round's record sums them in batch order as today. Refused by name if the round's draw differs (its windows'
      records, as the workers already check, `post_train._speak`), or the settings;
    - why it holds: a batch reads only its own windows' random numbers (`Speakers`: the same groups and records whatever
      process speaks it), and the round's draw comes from the seed. Batches spoken by other code may be joined once
      the codes are shown to behave the same on fixed inputs (root `CLAUDE.md`, as D181 items 58–60), else refused;
    - tests: a round stopped after some batches and resumed gives the groups and the record of a round spoken whole,
      bit for bit; a half-written file is never read; another draw is refused.
    It protects against loss (a stop costs the batches in flight, not the round); it does not make a round shorter
    (items 66 and 67 do). It changes D157 (stage C's and stage D's campaigns alike); the designer and the user decide.
