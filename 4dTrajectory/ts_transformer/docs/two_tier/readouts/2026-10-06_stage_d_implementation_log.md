# Stage D: implementation log

Stage D's implementer keeps this log (outline §5 rule 10, multi-aircraft control §0.3). It holds the state, the commits,
the checks and each reading made where the design says nothing; a reading is a proposal until the user decides
(`docs/two_tier/design/requests_from_d_to_designer.md`). Paths are relative to `4dTrajectory/ts_transformer/` unless
they start with `4dTrajectory/` or `.claude/`.

## 0 State

| Part | Branch | State |
|---|---|---|
| Branches (notes/stage_d.md "分支和工作树" 1–3) | dev-two-tier-v4, dev-multi-control | Done 2026-10-06 |
| MC0 · stage A (vocabulary §6 item 5) | dev-multi-control | Built, reviewed, committed (9c98a8d6) |
| MC0 · stage B (prior §7 items 2, 3, 7) | dev-multi-control | Built, reviewed, committed (d7bf55f5) |
| MC0 · the silent aircraft (multi-aircraft control §6.2 item 4) | dev-multi-control | Checked by a test (in d7bf55f5): it holds |
| MC0 · stage C (post-training §9 items 1–3, 7–9, 11, 12) | dev-multi-control | Built, reviewed, committed (3a9b7153, 80fe774c, 648b6591, 5e456406, a1ea664f) |
| MC1 · `multi/`, `experiments/multi_windows.py` | dev-multi-control | Built, reviewed, committed (e6f0e564); not run on real data |
| The full ts suite on dev-multi-control (a1ea664f) | dev-multi-control | 1,918 passed, 1 skipped (8 workers, 10.6 min, 2026-10-06 after C10) |
| Stage B's steps after C10 (notes/stage_d.md 一 1–5) | dev-two-tier-v4 | Done 2026-10-07 (§6; stage B's log §6); dev-two-tier fast-forwarded to v4 (1e829e95) |
| dev-two-tier into dev-multi-control (notes 二 1) | dev-multi-control | Merged 2026-10-07 (9b90d5d7), no conflict; the changed modules' tests 72 passed |
| MC2, MC3 · stage D's rules (`multi/separation`, `tokens`, `credit`) | dev-multi-control | Built, reviewed, committed (fd720c75); synthetic tests only (C14 runs) |
| MC4 · the round's stage (first part) | dev-multi-control | Built, in review |
| Cleanup (notes 三) | — | Done 2026-10-07 (§7): seven worktrees removed, their data links unlinked first; ten branches deleted with `-d` |
| MC0's checks and MC1's census on real data | dev-multi-control | After C14's rounds (outline §4 item 7) |

## 1 The branches (2026-10-06)

1. The user let the implementer fast-forward `dev-two-tier-v4` to `dev-multi-control-design` (2f834c89; asked through
   the question tool, the order's step 1 said the user would).
2. `dev-two-tier` (stage C's 832555a5, the update in pieces) merged into `dev-two-tier-v4`: c7a0b6b0. git merged it
   with no conflict; the review (opus reviewer) found one semantic conflict, S2: with the update in pieces, O15's
   memory rule (`post_train.pass_memory_of`) asked the profile for the update's groups only, so the widest and the
   densest group alone were never measured. Corrected in the merge: it asks for one group too
   (`sorted({1, update_groups})`) and takes the largest peak, with a test. The 13 `test_post_*` files passed (one
   thread, one at a time); `test_post_train.py` again after the correction (18 passed). S3, one line: the refusal "fewer
   groups held" now mostly blocks a first batch that wrote fewer groups than an update takes (memory grows little with
   their count).
3. `dev-multi-control` from `dev-two-tier-v4` in `.claude/worktrees/multi-control`, its data trees absolute links to
   the live ones; fast-forwarded to c7a0b6b0.

## 2 MC0 · stage A: join ticks in the start (9c98a8d6)

`autopilot/start.py`: `Loop(join_ticks=…)`, `Loop.started`, `Loop.copy` keeps them, `Start.moved(join_ticks=…)`. The
executor's step 0 is tick s; flight b starts at the loop's step j_b (the executor's start cycle j_b × the cycles of a Δ
row, `Spoken`'s start step j_b). A flight not started is not heard. Readings: requests items 1, 2.

Tests (`tests/test_start.py`): every join tick 0 equals no join ticks; a flight that joins at its tick flies what it
flies alone (states within `STATE_BOUND_M`, done, halted, time limit, go-arounds, record and outcome exact; words it may
not say before its tick are not heard); a copy keeps the join ticks; refusals. `test_start.py` 24 passed, with
`test_autopilot.py` 64, `test_closed_loop.py` 60, `test_architecture.py` 35.

Review (opus reviewer): no S1, no S2 in stage A. S3, one line: the zero-tick test of `Loop` compares the new code with
itself (the reviewer measured the old and the new loop's record equal, digest 2d939d76…).

Vocabulary D73's checks (`closed_loop_start_check` first, then the executor conformance) on the formal artefact: after C10
ends (notes/stage_d.md "三" 1).

## 3 MC0 · stage B: aircraft of several roles in one speaking loop (d7bf55f5)

`prior/loop.py` (`LoopRows` join ticks, `add_landing`), `prior/landings.py` (`LandingIndex.with_landing`),
`prior/batch.py` (`row_tensors(present=…)`), `prior/procedure.py` (`ProcedureMasks.keep`), `prior/speaker.py`
(`Speaker.say`, `ABSENT`/`OBSERVED`/`SAID`, `said_rows`; `speak` is `say` with every aircraft said),
`experiments/prior_speaking_loop.py` (`SpeakingLoop`: join ticks from the loop, `step` by each flight's own clock,
`roles`, `own_row`, `add_landing`). Readings: requests items 3–6.

Tests (`tests/test_prior_join_ticks.py`, 8 passed): every join tick 0 and no join ticks give the digest written by
c7a0b6b0's code (`BEFORE_JOIN_TICKS`: words, states, sentence rows, drawn probabilities, records); a flight that joins
says and flies what it does alone; absent flights; copies; landings added; refusals; B12 with join ticks; the silent
aircraft. The changed modules' files: `test_prior_free_generation.py` 23, `test_prior_speaker.py` 36,
`test_prior_inputs.py` 35, `test_prior_model.py` 13, `test_prior_train.py` 17, `test_prior_campaign.py` 13,
`test_prior_training_export.py` 12, `test_prior_validation.py` 14, `test_model_speed.py` 9, `test_post_window_loop.py`
11, `test_post_train.py` 18, `test_post_training_export.py` 5, `test_post_loss.py` 10, `test_post_landings.py` 6,
`test_post_traffic_attention.py` 7 — all passed.

Review (opus reviewer, who ran the scenario on c7a0b6b0's code from `git archive` and got the pinned digest): two S2,
corrected and checked again — `prior_behaviour`'s stand-in loop had no join ticks (the D108 behaviour check failed; its
answer unchanged after the correction, c13881a5…), and a row refused at a flight's first predicted step left its start
state in the flight's record (the start state is now kept only once the row is, with a test); a refusal that could not
fire, deleted. S3, one line each: the digest leaves out some records (the reviewer measured them equal, 5b55b096…); the
speaker keeps, in its raw lists, the rows of aircraft not said (only `said_rows` tells them; `permitted` and the loop
read them through it); `model_speed`'s `TimedLoop.started` hides `Loop.started` and resets its timer at each
`rows()` (only with join ticks, which `model_speed` does not use).

**§6.2 item 4, the silent aircraft.** Checked: the speaker takes a caller's mask in every column, and after the first
predicted step the grammar permits "unchanged" in every column (rule 3 acts only when an altitude or angle word is
said; the procedure masks always permit "unchanged", D64). Under a mask of "unchanged" only, the flight says
"unchanged" with probability 1 and its rows' log-probability under the records is 0 (the test).

## 4 MC0 · stage C: post-training §9 items 1–3, 7–9, 11, 12

Each generalisation keeps stage C's behaviour bit for bit (D149): `tests/test_post_generalised.py` holds digests written
by c7a0b6b0's code (stage C before any change; the CPU, one thread) — the traffic module's state (names, shapes, first
values: C10's checkpoints load), the window loop of a real window, of window A and of window B with their samples, a
stage C batch whose first window ends while the other flies on, two branch rounds and a group's samples, and two
campaigns of two rounds (stage C's own draw; a pass that updates) — and passes after every commit.

| Commit | Items | Review (opus reviewer, D131's standard) |
|---|---|---|
| 3a9b7153 | the reference | — |
| 80fe774c | 1 (`Joined`, `Window.joined`), 2 (`commanded_landings`), 3 (`step_losses`, `answered_loss`, `scene_aircraft`), 7 (a caller's token part: `TokenPart`, `add_token_part`, `Traffic.part`) | S2 fixed: the module's state pinned against the old code |
| 648b6591 | 8: the window loop of several commanded aircraft (join steps, each row's others, a caller's rule of who answers, silent aircraft, a landed aircraft's landing to its window's others, the window's end) | S2 fixed: the silent mask on rows still flown only (a stage C row ended with its window is said on as before); tests of a compressed window and of token values. To the designer: the wake minimum behind a commanded leader (requests item 7) |
| 5e456406 | 9: branch training (`Rules`, `stage_c_rules`; a continuation copies the window, the varied aircraft draws new numbers, the others go on from their streams; `Sentence.until`, `Group.varied`) | S1 fixed: an aircraft ended before the window's last branch point made every such window differ; S2 fixed: a crash at a point that is an aircraft's first predicted step, a test that could not catch a wrong stream |
| a1ea664f | 11 (`Stage`, `STAGE_C`: the round's skeleton), 12 (`open_round`, a formal start refusing a smoke campaign) | S2 fixed: a reference against the old campaign, the smoke refusal, the window's record as a part of the stage |

The review's depth: after the user's word of 2026-10-06 ("注意review的深度"), every review order names review_guide §3
step 6 and §6 and the stage's checklist (§5), and only S1 and cheap S2 are acted on.

## 5 MC1 · the census (e6f0e564)

`multi/windows.py` (the windows of D146: `Anchors`, `window_of`, `compressed`, `Drawn`, `left_out`), `multi/census.py`
(the losses of a window's steps by pair — `commanded_commanded`, `commanded_answers`, `records_kept`, `recorded_only`
— on the records and on the closed-loop states, the baseline of §5 item 4), `experiments/multi_windows.py` (train and
select, each span and kind, records and baseline, a scratch directory, each split written as it is done); the import
rules of §10 in `tests/test_architecture.py`. Review: S1 fixed (the baseline's split of D145), S2 fixed (the baseline
judged to its own end; one rule of a commanded leader's threshold; tests that can fail). The reviewer's estimate of the
formal size: about 16–45 h in one process (requests item 18). Not run on real data.

## 6 Stage B's steps after C10 (notes/stage_d.md 一; outline §4 item 6)

Done on `dev-two-tier-v4` in `.claude/worktrees/two-tier-v4` with no other job on the host or the GPU; the details in
stage B's log, §6.

1. **B14's real-data check** (2026-10-06): B5's fold `C_full_s1337/KRDU` spoken again on the select days with the batch
   masks, its readout's options (200 flights, 2 samples, seed 1337, chunk 400, CUDA), into a scratch directory: all 400
   sentences word for word its readout's, and every other field of `sentences.npz` (states, probabilities, procedure
   blocks) equal.
2. **C13's check on the GPU** (2026-10-06; reading: one smoke round at C10's settings with 32 windows of each kind and 20
   select windows an airport, one process against two speaking workers): the draw, the windows, the speaking record,
   the groups' bytes, the selection readout, the pass's numbers and the identity identical; the weights within 2.0e-7
   (float rounding of the pass after workers, as `post_train`'s docstring says).
3. **The base's speed** (2026-10-06): `4dTrajectory/outputs/POOLED/speed/prior_base_20261006/speed.json` (`model_speed`,
   CPU one thread and CUDA, batch 1 and 400).
4. **The re-export** (2026-10-07): `sha256sum -c` of `prior_base_20261006` before and after: both OK. A's
   `closed_loop_v12_20261005` and B's six sets of `prior_sets_20261006` exported in the new formats with the same ids,
   B's naming the speed readout. The old sets moved aside first (the user's choice of 2026-10-07: the same id is the
   same directory, which an export never overwrites): `<set-id>.v10-old` (A) and `<set-id>.v3-old` (B), the old
   indexes left as they were.

**Items 2–4 are done (2026-10-07): C14 may start.**

## 7 The multi-aircraft control after stage B's wrap-up (notes/stage_d.md 二, 三; 2026-10-07)

While C14's rounds run (from 00:10, 2026-10-07): code and synthetic tests only, on the CPU with at most 4 processes
(rule 13).

| Commit | What | Review |
|---|---|---|
| 9b90d5d7 | dev-two-tier (1e829e95: C14's code, D157, P47) merged into dev-multi-control; no conflict (C14's change of `open_campaign` and the stage of MC0 touch other functions) | The changed modules' tests: 72 passed |
| fd720c75 | MC2, MC3: `multi/separation.py` (the census's judge on the records, shared; `Answering`, D145), `multi/tokens.py` (`TokenPart`, `multi-commanded-tokens-v1`, D152), `multi/credit.py` (W, spoken again, the varied aircraft and their branch points, stage D's numbers, `rules()`); `Rules.varied` also reads the first pass's ends | No S1. S2 fixed: a test that the loop asks the rule at the judged step. S2 to the user: requests item 25 (the silent flag reads the records under D145). The touched modules' tests: 81 passed |

**The cleanup of notes 三** (git only): for each worktree, its data links listed (`find -maxdepth 4 -type l`) and
unlinked, then `git worktree remove` (none refused: no worktree had changes), then `git branch -d` (none refused: each
in dev-two-tier). Removed: merge-a43, training-attitude, stage2-restart, a25-build, stage2-real400, traffic-scenarios;
branches deleted: dev-two-tier-merge-a43, dev-training-attitude, dev-stage2-restart, docs-optimizer-multi-aircraft,
dev-frontend-design, dev-multi-control-design, dev-traffic-scenarios. The links' targets (the live data) checked
afterwards: all there.
