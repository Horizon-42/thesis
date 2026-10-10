# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62; D181–D186: 63–70; the interfaces and D182:
71–77; D188: 78–79; D190: 80–84; D190, D191: 85, 87, 88), item 23 (it asked for 6 and 7), item 20 (built: item 40) and
the names of MC0–MC4 (written into the interfaces).

State: 2026-10-10 (evening, after the handover): the user decided R1 of the plan (item 95) and ordered the speed-ups S1
and S2, built on `dev-multi-speed` (items 96, 97); the traffic-only arm (the plan's R3) runs. 86 and 90 wait for the
user; 89, 91–94 come from D191; 95–98 are new.

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

86. **The front's length: one for a cache** (proposal; measured, stage D's log §20). One length S for every front of a
    cache; two blocks of own rows would hold the joint loop's cache at its largest in 44,158 rows instead of 58,480 on
    batch 23 (20,930 / 23,616 instead of 24,626 / 34,032 on batches 0 / 86). The user chooses; not built.
89. **The two ways' time** (measured, the profile at §18's settings on 7893d787). Copied: batches 0 / 23 / 86 in 56.8 /
    68.0 / 63.0 s; merged: 55.5 / 63.5 / 59.9 s (D189: 55.3 / 64.9 / 61.0). The campaign of item 90 reads `merged`.
90. **The next campaign: launched** (the user, 2026-10-10 at night: once the tests pass, run it; the GPU alone this
    time; decide and report). MC6's start (stage C's `post_seg60_20261007` round 5, seed 2035) and settings, branch
    points found backward, `front_attention` `merged`, 8 rounds, 5 GPU speaking workers sized by item 91,
    `outputs/POOLED/multi/multi_train_20261010` from a run worktree at the commit of item 91 (stage D's log §21). Its
    first round's time is the measure of its speed (no measure before it, the user's word).
91. **The peaks last** (built at the user's word of 2026-10-10: "the peaks last, or more workers"). A GPU batch past
    its worker's share (CUDA's out of memory under its cap) no longer fails the round: it is spoken again after the
    others at fewer at once — half the workers, a quarter, … one —, each level's budget the GPU less the campaign
    process's hold and what each idle worker holds, shared by its batches at once; past the last level the round
    fails. The worker frees the batch's memory before it is free for another (the error raised again without the
    batch's frames). So the GPU's fit counts a worker at what it holds between batches (D179's margin stays for the
    host and the pass, which have no second try), and refuses only workers whose last level cannot hold a batch at the
    profile's peak × the margin. Reading, for the designer to write into D179/D191: the margin of D179 no longer sizes
    GPU workers; 5 workers on the GPU (the host's 3.24 GiB each with the margin of 1.3 sets the number).
92. **The profile does not bound a round's batches** (measured). The same 56 batches of MC6's round 0 on 3 GPU workers
    stopped: one batch reached 2.05 GiB allocated in a rebuild of the joint loop, past its worker's share of 2.21 GiB,
    where the profile's three batches peaked at 1.33 GiB with the CUDA context (allocated 0.94). A batch's memory is set
    by the searches flown together, not by its rows: no measure of a few batches bounds it, the levels of item 91 do
    not need one.
93. **Mixed workers, measured once** (D191 step 4, 5b4cd739). Of MC6's round 0, every 4th batch (56), 3 GPU and 3 CPU
    workers, 6 of every 13 on the CPU: the GPU's 30 batches in 929 s, the CPU's 26 in 995 s — a batch and worker 93 and
    115 s against 60 and 69 s alone (six processes at once slow each); about 66 min a round. Not used by item 90 (the
    user: the GPU alone this time); the rule (`--cpu-batches`, `--cpu-period`) is a setting for a later campaign.
94. **What the records hold more** (reading). `round.json` holds `cpu_workers` (information, as `speak_device`); a
    stage C measure (`campaign.json` `measures`) holds `cpu_workers` and `cpu_measured`, and a measure taken before
    D191 holds no CPU worker's: a launch with CPU workers measures again (as D167's record without a measure);
    `profiled_fit`'s line names `cpu_profile`, `cpu_workers` and the levels (`deferred`).

## The plan of 2026-10-10 evening and the speed-ups S1, S2 (`dev-multi-speed`)

Please review the plan and write what it decides into `multi_control.md`. The code says the details; read the commits.

95. **The plan**: `readouts/2026-10-10_stage_d_next_plan.zh.md` (uncommitted until this note's commit). Its status table
    lists each item and who decided it. The user's decision: R1, point B replaced by the point that at least 5 of 8
    continuations avoid (not built; for D192). The user's other words: R6 (value function) not done; R3 (traffic-only
    arm) launched. The rest are proposals.
96. **S1, built** (2961b99a): the process option `--executor-device cpu` (`multi_train`, `multi_speed`) flies the
    executor on the CPU while the speaker stays on the GPU; default `speak`, bit for bit the code before. Not a campaign
    setting. Not yet done: the D181 (57) bound on real windows and the GPU timing (the GPU runs the arm).
97. **S2, built** (88a3990e): the rows not flown read their tokens in one call per window; bit for bit
    (`window_behaviour` against 2961b99a on stage D's and stage C's windows, `prior_behaviour`). Batch 86 on the CPU:
    the window loop 20.1 → 17.6 s, the batch 84.3 → 81.5 s. One part was tried and dropped: the speaker's runway mask
    in one call for all runway words is slower above about 64 rows (the plan, S2).
98. **S6, a proposal** (changes D188's joint loop): rows no longer flown stay in the joint loop and are still said and
    flown, about 46 of 135 rows a step on batch 86; nothing reads their words or states. Dropping them at a rebuild
    would save about a third of the speaker's, the executor's and the tokens' work. For the designer and the user.
