# Stage D's requests to the designer

Stage D's implementer writes this file (outline §5 rule 10, multi-aircraft control §0.3); it is rewritten in full each
time. Each item is a reading the implementer made where the design says nothing, or a gap; a reading is a proposal
until the user decides. Paths are relative to `4dTrajectory/ts_transformer/`.

State: 2026-10-07: MC0 (stages A, B, C) and MC1 built on `dev-multi-control`, not yet run on real data.

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
6. **A landing added while a loop runs, on a sealed test day.** §6.3 item 5: "the index keeps its checks". Built so: a
   loop landing whose time falls on a sealed test day (a crossing just after a day cut) is refused by the index
   (`SealedDay`), which would stop a campaign. Stage C's own window landings (`post.landings.window_landings`) instead
   leave a shifted landing on a test day out and count it with the sealed landings. Proposal: the same for a landing
   of the loop — left out and counted as sealed — decided by the user before MC2 (the window loop of several commanded
   aircraft) uses it.

## MC0 · stage C (post-training §9 items 1, 2, 3, 7, 8)

7. **The wake minimum at the threshold behind a commanded leader is never judged** (review of item 8, S2, not
   measured). A recorded aircraft is "over its threshold" at its last row in the air (`AircraftAt.last_step`); a
   commanded aircraft has `last_step` false and leaves the scene in the row after the executor ends it, so
   `wake_at_threshold` never runs with a commanded leader. In a window of several commanded aircraft in trail to one
   runway, a follower inside the on-approach wake minimum when the leader crosses, but outside the pair minima at every
   Δ step, is not judged lost. Stage C never reaches it (a recorded follower answers for nothing). A reading is needed
   before MC2: for example, at the row after a commanded aircraft ends `landed`, judge it once as over its threshold,
   at its crossing state.
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

## Names that changed in the public interfaces (for the designer to write in, outline §5 rule 10)

- Vocabulary §6 item 5: `Loop(…, join_ticks=…)`, `Loop.join_ticks`, `Loop.started()`, `Start.moved(…, join_ticks=…)`.
- Prior §7 item 2: `LoopRows(…, join_ticks=…)`, `LoopRows.join_ticks`, `LoopRows.add_landing`,
  `LandingIndex.with_landing`, `batch.row_tensors(…, present=…)`. Item 3: `Speaker.say`, `speaker.ABSENT`, `OBSERVED`,
  `SAID`, `Speaker.said_rows`; `procedure.ProcedureMasks.keep`. Item 7: `SpeakingLoop.join_ticks`, `roles`, `own_row`,
  `joined`, `said_now`, `add_landing`.
- Post-training §9 item 1: `scene.Joined`, `Window.joined`, `commanded_all`, `signal_indices`, `join_steps`. Item 2:
  `landings.commanded_landings`. Item 3: `traffic.scene_aircraft`, `step_losses`, `answered_loss`. Item 7:
  `traffic_attention.Traffic.part`, `Traffic.select`, `traffic_of(…, part_width=…)`, `TokenPart`, `add_token_part`.
  Item 8: `post_window_loop.WindowLoop(…, answering=…, token_part=…, part_width=…)`, `responsible`, `Answering`,
  `TokenPartOf`, `WindowLoop.members`, `window_of`, `member_of`, `records`, `silent`, `landings` (now the speaking
  loop's, with the loop's own landings).

## MC0 · stage C, items 9, 11, 12

12. **The rules of branch training are the caller's** (post-training §9 item 9): `experiments.post_branches.Rules`
    (each aircraft's first and continuation numbers, the window's reward, spoken again, the varied aircraft with their
    branch points as ticks of the window's grid); `stage_c_rules` are stage C's. A group's branch point is in the
    varied aircraft's own rows (`Group.branch`), its sentences count up to that aircraft's event (`Sentence.until`).
    Stage D's own rules (D142's streams, D143's varied aircraft) are MC3's, in `multi/credit`.
13. **The campaign's skeleton** (post-training §9 item 11): `post_train.Stage` and `STAGE_C`; the speaking workers
    (`Speakers`) speak stage C's batches; stage D's own speaking (its rules, its loop options) is given as its stage's
    `speak` at MC4. **The start of a later stage** (item 12): `post_train.open_round(campaign, round, context)`.

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
18. **The cost of the census at the formal size** (the reviewer's estimate, not measured on real data): 0.2–0.9 ms a
    judged step, about 5,200 steps an anchor (4 spans, 3 kinds, records and baseline), so about 16–45 h in one process
    for 45,000 train anchors, the select days besides. Proposal: after C10, measure seconds per anchor on a sample
    (`--sample`, D55), then choose: the full census with one worker per airport, or a stated sample of anchors per
    airport (never silent).
19. **A formal start of a later stage and a campaign not finished** (review of item 12, proposal for the user):
    `open_round(…, formal=True)` refuses a smoke campaign (as `post_validation` does). `post_validation` also refuses a
    formal read while the campaign has rounds still to run (D7, D85: the round is chosen on every round's selection
    readout). Proposal: the same for a formal start of stage D (MC6 starts from the chosen round, after C10 ends) — the
    user decides.
20. **A batch holds windows, not aircraft** (review S3, for MC5): `batches` counts windows; with several commanded
    aircraft a batch's rows grow with L, and the memory measure of C13 (`_measure`) times one batch. MC5's preflight sizes
    a batch by its rows.
