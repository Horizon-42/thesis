# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`. The numbers are kept from the earlier
notes; the items the user decided are deleted (D166: 6, 7, 19, 21, 22, 25, 29–35; O16: 36, written into D146; the
census's sample: 18; D172: 46–48; D179, D180: 49–53; D181, O20: 54–62; D181–D186: 63–70; the interfaces and D182:
71–77; D188: 78–79; D190: 80–84; D190, D191: 85, 87, 88; D191–D193 and the log: 86, 90–98; D193, D194 and the log: 89, 99–107), item 23 (it asked for 6 and
7), item 20 (built: item 40) and the names of MC0–MC4 (written into the interfaces).

State: 2026-10-12 (05:30): the formal campaign `multi_train_majority_formal_20261011` resumed at 01:50 (round 5's 216
done batches kept, batch 186 spoken on the GPU, `--gpu-squares 200`) and ended its 8 rounds at 03:44 (round 7: select
landed 83.7 %, 604 losses of separation, reward 3727.2, the best of both arms); round 7 published to the Training view
(two sets); the user extended the campaign in place to 12 rounds (rounds 8–11 run, item 119). The control arm
`multi_train_minority_side_20261011` ended its 8 rounds (rounds 5–7 at 3 × the learning rates: no gain, item 116).
Built and merged since item 114: ae3839bd (item 115), 16bacde0 (item 116), cf7e21f0 (item 117), 70d87089 (item 118).
108–120 are open (stage D's log §22 is not yet written for 115–120).

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

## D194 and arms A, B (stage D's log §22; 2026-10-11)

108. **The CPU's levels and the CPU profile of D192** (measured in A's round 1; for D191, D193 (4) and O15). With 6 CPU
    workers a worker's host share was 2.43 GiB; about 40 CPU batches, nearly all of span 1200 s, went past it and were
    spoken again after the first pass, most one at a time (levels 3 at 2.78 GiB, 1 at 4.19 GiB): 20 of them took
    02:06–02:33, while the first pass took 37 min for 183 batches. With 4 workers (a share of 3.64 GiB; the user's
    word, A resumed with its batches kept) the other 20 took 8 min. The CPU profile (batches 0 / 23 / 86) holds below
    the host peak of the majority's batches of 1200 s, so the fit's levels assume a tail that does not happen. Proposal,
    for the user: (a) a CPU profile of D192 on the batches of 1200 s with the most aircraft, or (b) a batch spoken again
    goes to the level whose budget holds the peak it reached, not to each level in turn. B runs with 4 CPU workers;
    its round 0 measures the first pass and the tail with 4.
109. **D194 (3) on the GPU: one search of 224 batches differs** (measured; please write the bound into D194 (3)). Arm B's
    round 0 speaks from A's start with A's settings (the data weight acts in the pass only), so its groups are A round
    0's where the code behaves the same. Of 224 batch records, 4 are the same and 219 differ only in the search's seconds, steps or rows
    (D194 (3)); batch 147, on the GPU, differs in one search: A (c60f59c6) chose the decision point at 128 s with its
    majority point at 224 s, B (e77bc220) found no avoiding continuation at 128 s and chose 160 s, its fine point at
    144 s and the majority point at 224 s (12 → 13 groups). The search's own stop acts only behind its decision point,
    so this is the other searches' stop: the joint loop's rows fall earlier, and on the GPU a row's rounding depends
    on the batch's shape. Every CPU batch (158) is the same; window_behaviour (CPU) found every digest equal. Reading:
    D194 (3) keeps the answers on the CPU bit for bit, on the GPU within the GPU's rounding (as D193 S1: states within
    2.2e-8 m), here 1 search in 224 batches.
110. **For the user: B's round 0 with 4 CPU workers** (D194 (4)'s measure). Spoken 03:00–03:56 (A round 0: 69 min
    with its stop). The GPU's 66 batches were done by 03:27; the CPU's 158 batches took to 03:56, the last 1200 s batch
    alone at the end; no tail of the levels as in A's round 1. The CPU is now the slower device: 7 of every 10 batches
    go to it (`--cpu-batches 7 --cpu-period 10`, set before S1 and S6, item 102). A rule with fewer CPU batches would
    end the round earlier; arms compared keep one rule (D194), so it is for a next campaign.
111. **D194 (3) for every point tried, as built** (notes/stage_d.md order 1; d3dc3b26, merged 226ed303). `_Part.need`:
    1 for a point tried before the decision point, the fine point and point B, the majority behind the decision point
    in a search for the majority point; `out_of_reach(failed, continuations, need)`; the decision point's copy is never
    stopped. Order 2: window_behaviour on arm A's campaign (MC6's round 0 draw, batches 0 / 23 / 86, majority, the CPU,
    one process) against 14e53df6 (no D194 (3)): "the same behaviour". Tests: a synthetic search whose points before
    the decision point and fine point all fail stops them at their first step and finds the same; the batch test of
    D188 allows a point judged early where every continuation failed. Review: no S1/S2; S3: point B's own early stop
    has no test of its own (the same expression); on the GPU the answer is the same within the GPU's rounding only
    (item 109), now for more points; `_failed` now runs for every point tried at every step (its seconds against the
    rows saved are to be read in the formal campaign's records).

112. **The group weight: a decision of the user to write as D195** (the user, 2026-10-11; built afb9aea2, merged into
    dev-two-tier; the code and the intents already name it D195, the next free number). The user's words: weigh each
    branch group so that a lucky group (one continuation better than the first sentence of 8) still counts but less,
    and train every kind of group together. As built (`MultiSettings.group_weight`, `--group-weight`): `equal` (the
    default, the behaviour before it; a record without it reads so) or `minority-side`: each group's advantages times
    its minority side over the round's mean minority side. The minority side (`post.branches.Group.minority`) is the
    smaller count of the group's sentences above and below its mean reward, a sentence within `SIDE_TOLERANCE` (1e-9)
    of the mean on neither side; 1–4 with 8 continuations. The mean comes from each batch record's `group_minorities`
    (written under `minority-side` only), the pass record holds it as `minority_mean`, and pass_gradients (R70) reads
    with the campaign's weight; not profiled (no memory change); stage C unchanged. Why the minority side and not the
    count of better continuations (the user chose it after both were shown): a group whose first sentence is its one
    good sentence has the same one-against-eight form, and a mean difference is only as sure as its smaller side. On
    round 2's groups the summed |advantage| share of one-against-eight groups falls from 19 to 7 %, the majority
    groups' rises from 23 to 28 %. The user's plan: the control arm `multi_train_minority_side_20261011` (the formal
    campaign's settings with `minority-side`, 3 rounds, compared with the formal campaign's rounds 0–2) if the formal
    campaign's next two rounds stay flat. Please write D195 into multi_control.md and its arm into the plan.

## The formal campaign's stop in round 5 (stage D's log §22; 2026-10-11)

113. **A resume keeps each done batch whatever device spoke it: a decision of the user that reverses part of D185 and
    D191** (the user, 2026-10-11: "works数目改一下，不应该成为从中途resume 某一round的障碍"; built 67e45506). D185 / D191
    as built refused a done batch spoken on another kind of device than the launch's rule gives it, so a resume with
    another number of workers or another rule had to speak the round again. Now `done_batches(directory, windows, places,
    stage)` keeps each done batch; each batch record still holds its device; the launch's rule and workers place only the
    batches not done; a batch of another draw is still refused. A round so resumed has batches spoken on both kinds of
    device in other places than one rule gives (D194 (3): the answers are the same on the CPU bit for bit, on the GPU
    within its rounding). Please write it into D185 / D191.
114. **A CPU batch past the CPU's last level is spoken once more on the speak device: a decision of the user for D191 /
    D193 (4)** (the user, 2026-10-11: "CPU 超了自动换到 GPU 再说一次"; built 9f50be58). After the GPU's own levels and
    the CPU's levels, a CPU batch still past its share goes once to the speak device's pool at that device's last level
    (`Speakers._results`); only a batch past that too fails the round; no GPU level or a CPU speak device: no second try.
    The batch's record holds the GPU. Whether a batch goes there depends on its sampled host peak, so its device can
    differ between two runs of the same round. Also built the same afternoon (small change): every worker, GPU workers
    too, calls `malloc_trim(0)` after a task (1efca2f8). Batch 186 (measured alone on the CPU): a peak
    of 12.0 GB, 9.5 GB for the batch; its 14 searches stepped 433,231 rows, the most of all 1,309 batches (p99 155,400),
    from one KSTL window of 13 commanded aircraft (a correction: the census's "12 joined" are the 12 commanded aircraft
    besides the anchor, not 12 more aircraft). A batch's memory follows its search rows, which the batching (by
    commanded aircraft) does not see. The proposal written here (count a window's aircraft when batches are made or
    placed) was built as item 115. Please write 113's fallback into D191 / D193 (4).

## The formal campaign's end, its publication and its extension (2026-10-11 evening to 2026-10-12)

115. **Batches bounded by their commanded aircraft squared, and heavy batches on the speak device: a decision of the
    user to write into D191 / D193 (4) and the settings of §7** (the user, 2026-10-11, after batch 186; built ae3839bd).
    A window's searches fly each of its aircraft for each continuation, so the measure taken is a window's commanded
    aircraft squared (over rounds 0–5 the sum of squares follows a batch's search rows at r = 0.47, the sum of rows at
    0.03). Two settings, both off by default (the behaviour before them; a record without them reads so):
    `--batch-squares` (one bound a span; `post_train.batches` also keeps a batch's sum of squares at most the bound, a
    window above it a batch alone; a campaign setting, in `PROFILED_SETTINGS`) and `--gpu-squares` (a batch whose sum
    reaches it is spoken on the speak device, else `batch_device`'s rule; a resume may change it, the resume record's
    "rule" holds it). Measured after (each batch alone on the CPU, its added memory and search rows): squares 45 →
    0.37 GB / 1,906 rows; 100 → 0.32 GB / 0; 128 → 2.31 GB / 90,439; 161 → 0.85 GB / 16,106; 194 → 2.88 GB / 88,613;
    210 → 2.25 GB / 100,929; 600 s span, 179 → 2.07 GB / 91,017; batch 186, 218 → 9.5 GB / 433,231. So the memory
    follows the search rows (about 23–25 KB a row), the squares only roughly; batch 186 on the GPU (its round-4 model,
    round 5's draw, executor on the CPU as launched): allocated peak 3.36 GB, reserved 6.95 GB (the GPU's last level
    was 6.81 GB), host peak 9.25 GB, 397 s. The formal campaign resumed with `--gpu-squares 200` only
    (`--batch-squares` off: it would change the formal campaign's batching). Open, for the user: a bound on a window's
    search rows is not known before it is spoken; the squares are the measure available.

116. **A resume may change the two learning rates: a decision of the user for the resume rules (D185) and D195's control
    arm** (the user, 2026-10-11; built 16bacde0, small change). `prior_lr` and `traffic_lr` join `RESUME_MAY_CHANGE`;
    each resume entry holds the rates before and after and the round they act from; the optimizer state is loaded and
    each group's rate set from the settings (Adam's moments kept). Used once: the control arm's rounds 5–7 at 3 × (3e-5
    / 3e-4). Result (select readout, landed / losses of separation / reward): round 4 83.1 % / 644 / 3702.5; round 5
    83.1 % / 631 / 3707.8; 6 82.5 % / 644 / 3672.8; 7 83.0 % / 635 / 3701.7; KL 0.061 → 0.074 → 0.092 → 0.100, clipped
    share 0.7 → 2.7 %, other failures 116 → 129. The larger rate did not lift the plateau. Reading (pass_gradients,
    R70): the prior's reward gradient between updates has a cosine near 0 while the data term's is 0.95, and the credit
    is spread (median 33 rows a group); a larger step moves further along noise. Please write the rule into D185 and the
    arm's result into the plan.

117. **A stage D round read again window by window, and stage D's window lists: readings for §5 and the Training sets**
    (built af98c354, d0b9c202, 5139728a, merged cf7e21f0; runners.md R71). `experiments/multi_window_read.py` (schema
    `ts-multi-window-read-v1`, `config.json` + `per_window.jsonl`, sealed) speaks a done round's model again on its
    select windows with the round's own numbers and writes a line a window (`multi_train.window_line`; each commanded
    aircraft's outcome, reward and losses); it stops unless the summed lines are the round's `selection_readout`
    (`selection_readout(..., lines=)`; without `lines` the readout is unchanged); it refuses a round not done, a smoke
    campaign and a dirty tree (unless `--smoke`). `window_list --multi DIR` lists a stage D window when ANY of its
    commanded aircraft has an outcome in `--outcomes`; it refuses a read of another stage. The list feeds
    `multi_training_export --windows`, so a listed set's round column is the sentence the readout judged. Readings to
    confirm: (a) "any commanded aircraft" as the rule of a stage D list; (b) the list's identity of a window is written
    twice (the export's lambda and `post.window_lists.identity_of`, `docs/code-health-followups.md`). Round 7's read:
    1,500 windows; not landed (every outcome but `landed`): 453 windows (KMSY 55, KRDU 118, KSJC 95, KSMF 95, KSTL 90).

118. **The Training view counts a stage D aircraft as the readouts do; a closed loop of one word is read: for frontend
    §5.1 / §5.6 and the stage A reader** (built 70d87089, review clean after one S2). (a) An aircraft that answers for a
    loss is `lost_separation` in the loop (`post_window_loop.results`), a silent one flown on to a landing too; the
    export writes the silent aircraft's sentence with the end it was flown on to. The statistics already counted the
    loop's way; the window list (tally and tooltip), "this round" and the rounds summary counted the export's end, so
    a window with a silent aircraft read as all landed (2 of KMSY's 55 listed windows). Now one rule,
    `countedOutcome(window, place, aircraft)` (`trainingWindowSample.ts`); the aircraft line still shows the export's
    end with "silent from row N". Please write the rule into frontend §5.1. (b) A flight whose executor failed in its
    first word has a closed loop of one word, `flownFromRow + 1` state rows (the start state only); the reader asked
    `flownFromRow + 2` and refused three listed sets (KRDU, KSJC, KSTL). The reader now asks only the rows' equation.

119. **The formal campaign's end, the choice of the round to publish, and its extension: decisions of the user for the
    plan** (the user, 2026-10-12). (a) Order: finish the formal campaign first, then choose the best round, then
    export (the control arm's round-4 publication was stopped before it wrote anything). (b) The round chosen is the
    one with the highest select reward over both arms' 16 rounds: the formal campaign's round 7. Formal rounds 0–7,
    landed / losses / reward: 81.9 / 720; 81.6 / 730; 82.5 / 690; 83.1 / 648 / 3703.7; 82.8 / 663 / 3690.9; 83.3 / 625 /
    3711.8; 83.5 / 625 / 3718.8; 83.7 / 604 / 3727.2 (start 81.5 %). Round 7 by span: 300 s 83.4 % / 118; 600 s 83.9 % /
    189; 1200 s 83.7 % / 297. KL 0.044 → 0.070. The other failures rose 95 → 129 (unstable at minimums 24 → 36, timeout
    30 → 42): the gain is all in the losses of separation. (c) Published from round 7 (rounds start and 7, select
    split): `majority_formal_r7_drawn_20261012` (10 windows an airport, 5.0–6.6 MB) and
    `majority_formal_r7_not_landed_20261012` (item 117's list, 29–85 MB an airport). (d) Extended in place to 12 rounds
    (rounds 8–11; a new campaign cannot start from a stage D round, and its draws `[seed, round]` would repeat rounds
    0–3), every setting kept, at 70d87089 after a behaviour check. Stop rules (Claude's proposal, accepted by the user):
    two rounds in a row not above round 7's reward 3727.2, KL above 0.10 (the control arm's level at its plateau), or
    other failures above 150. Please write (b)'s rule and (d) into the plan.

120. **A run worktree needs `aeroviz-4d/public/data/airports` linked** (a gap of the run worktree's setup, found
    2026-10-12): `multi_window_read` and the export read the procedure details from there; the run worktree's first
    read stopped on a missing `KMSY/procedure-details/index.json`. The publication script now links it. Please add it
    to the run worktree's setup (D163) beside the other data links.
