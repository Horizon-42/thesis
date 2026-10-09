# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48), item 23 (it asked for 6 and 7), item 20 (built: item 40) and the names of MC0–MC4
(written into the interfaces).

State: 2026-10-09: items 49–53 decided (D179, D180) and deleted; `loop_positions` (a332466a) and D179 / D180
(48c21e50) merged. Items 54–55 are readings of that code for the designer. Items 56–62 (the user's request, 2026-10-09):
the engineering ways to a faster round, none of them a change of D141–D146's rules.

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

## MC5 · the memory measure (D179, D180; 48c21e50)

54. **"Each span's first and largest batch."** D179 says one worker on each span's first batch and on its batch of the
    most rows. Built with stage C's `post_train.measured_batches` (unchanged): the first batch of each span and the one
    batch of the most rows over all spans. In stage D a span's batches are filled to its `batch_rows` in order, so a
    span's first batch is its batch of the most rows but for a shorter one where a window's rows do not divide the
    setting (the profile of 2026-10-08: 64 / 64, 48 / 48, 24 / 24); the reading measures 3 or 4 batches. Proposal:
    keep it (the measure stays minutes); a batch of the most rows of each span would be at most one more a span.
55. **`select_per_airport` is still among `PROFILED_SETTINGS`.** The measure no longer reads the select set (D179), so a
    campaign with another select size than its profile's is refused and must be measured again, for nothing the
    measure reads (the reviewer's judgement, not a decided rule). Proposal: drop it from `PROFILED_SETTINGS` (one
    name, a small change), so that the select size is set freely before the campaign (D179).

## MC6 · the speed of a round (engineering only; the user's request, 2026-10-09)

**What was measured.** MC6's round 0, the attempt of 08:41–11:08 (the times of its `groups_<k>.pt`; 224 batches in
`profile_mc6_gpu32_20261009`'s draw, 2 GPU workers): the 23 batches of 300 s took 35 min, the 63 of 600 s 72 min, the
138 of 1200 s about 1.75 min each (about 4 h) — about 5–6 h of speaking a round (log §15: about 5 h). In worker-seconds
per commanded aircraft: 2.9 / 4.3 / 8.9 s at 2.1 / 3.0 / 4.9 aircraft a window (300 / 600 / 1200 s), against stage C's
2.1 s a window of one aircraft (C10, one worker; §25 of stage C's log): a window's cost grows about with the square of
its aircraft, since every continuation flies the whole window to its end (`experiments/post_branches.py:168`,
`WindowLoop.copy`, D141's W). Stage C's P55 spoke with 9 CPU workers (about 25 min a round); stage D fits 2 GPU workers
(a worker's peak 2.4–2.6 GiB whatever its rows, log §15) or 3 CPU workers (4.3 GiB host each). The rules' own levers
(K, the varied aircraft, the branch points, the share of 1200 s windows; a 60 s interval would about double the
speaking) are the designer's and not proposed here.

56. **A speed profile of stage D, first** (it measures the items below). One 1200 s batch of MC6's round-0 draw under
    cProfile: the first pass, the second pass and the continuations apart; the executor timed apart between two
    synchronisations, as `model_speed` times it (D136); the GPU's memory by part (the speaker's cache of the copies, the
    traffic attention, the executor's CUDA graphs). A runner of its own beside `multi_profile`, which stays a memory
    measure (D179). Also found: `post_profile.PARTS["executor_step"]` reads 0.0 s in both C8 profiles
    (`outputs/POOLED/post/c8_20261006/profile_{32,256}`) although the executor flies every cycle — the part's match
    misses it; the named parts sum to 43 of 113 s. Proposal: the runner, and the part's match fixed.
57. **The continuations in pieces of bounded rows.** `branch_round` flies every copy of a branch tick in one
    `copies.finish` (`post_branches.py:168–176`): K × the due varied aircraft × each window's aircraft, e.g. 4 windows ×
    3 varied × 8 × 6 aircraft ≈ 576 rows from a batch of 24 — Claude's reading of why a worker's peak is set by the
    longest windows, not by `batch_rows`. Proposal: a setting, at most P rows a piece (default: all at once, the old
    behaviour), each piece a `WindowLoop.copy` of whole windows. A row reads only its own window (Claude's reading of
    the traffic attention's input), so the pieces say what the whole says: on the CPU to the bit expected, on the GPU up
    to the kernels' choice by batch shape — a behaviour check on fixed inputs before use. Gain: a bounded peak, so more
    GPU workers or rows (item 56 measures it).
58. **The samples of the varied aircraft only.** `copies.samples(split)` (`post_branches.py:177`) builds the sentence of
    every row of every copy, and `Speaker.permitted()` stacks every row's masks over every row said; only
    `made[members[v]]` is read, one row of a copy's n. Proposal: `WindowLoop.samples(split, rows)` for the rows asked.
    The groups bit for bit; n − 1 of n of that time and host memory saved.
59. **A row's scene once a tick.** In a window of several commanded aircraft, `WindowLoop._of_row` (`scene_aircraft` and
    `separation_traffic`, `post_window_loop.py:277`) runs twice for each said row in a tick: in `_speed_masks` (:350)
    and in `_traffic` (:314). Proposal: kept for the tick. The same numbers.
60. **A worker's host memory.** A CPU worker peaks at 4.3 GiB (`profile_mc6_20261009`), a GPU worker's host at about
    1.8 GiB; the context is shared by the fork, but Python's reference counts and its collector write into shared
    pages, which are then copied. Proposal: `gc.freeze()` in the campaign's process before `Speakers` forks. The same
    results; it pays only once a new memory measure (D179) reads the lower peak.
61. **Workers on two devices (the user's decision).** D180 gives every worker one device. The GPU's memory and the
    host's are separate budgets: 2 GPU workers and 2–3 CPU workers fit together; a CPU worker speaks a 1200 s aircraft
    in about 10.9 s (521 s / 48 rows, `profile_mc6_20261009`) against about 8.75 s on the GPU (round 0), so the
    speaking would be about 1.8–2× faster. A batch spoken on the CPU and on the GPU may differ by float rounding (a word
    near a draw's boundary). Either the device is the batch's by a fixed rule (its place in the round; two pools, a
    poorer balance), so a round repeats, or the device a batch spoke on is recorded and the difference accepted.
62. **The copies' shared prefix (larger work, only if item 56 shows the cache is the peak).** The K copies of one branch
    point hold the same speaker cache up to the point, each a clone (`Speaker.copy`, `prior/speaker.py:390`). A prefix
    shared and a tail of each copy's own would cut the copies' cache by about (K − 1)/K; it changes the attention's
    arithmetic (two blocks), so it would be a fast mode beside the readable one, checked equal (D138).

Considered and not proposed:
- The executor's CUDA graphs (`aerodynamic_model/torch_scaled_transport_chart_dynamics.py:151`, `reduce-overhead`,
  `dynamic=True`; MC6's log warns of "9 distinct sizes"): a graph per batch size, but the step's tensors are `[B, 7]` —
  tens to a few hundred MB at most — and a replay saves microseconds of an executor row of 12–16 ms (`model_speed`,
  D136: batch 1 to 400, CPU and GPU alike). Only if item 56 finds its pool large: off in the speaking workers only
  (`post_train._initialise_speaker`), never in `aerodynamic_model`'s default (training uses it), then
  `executor_conformance`. Compiling it out altogether would slow the executor (about a hundred small kernels a step).
- The longest batches first: a round's tail is one batch (2–4 min of about 5 h).
- Every branch point's copies flown as one batch (they share the tick): more copies alive at once raise the peak that
  limits the workers now.
- The second pass from snapshots of the first: the second pass is about one first pass of the windows spoken again,
  small beside the continuations, and the snapshots cost memory.

Order proposed: 56 (the measure), 58 and 59 (the same numbers, small), 57 (with its check), 60, 61 for the user, 62 if
56 points to it. None touches the running MC6 (run worktree at `e4bbafac`); a resume on changed code needs a behaviour
check on fixed inputs first (root `CLAUDE.md`).
