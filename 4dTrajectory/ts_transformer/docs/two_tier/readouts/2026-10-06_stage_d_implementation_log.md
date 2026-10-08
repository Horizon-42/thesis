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
| MC4 · the round's stage (first part) | dev-multi-control | Built, reviewed (three rounds), committed (054727db) |
| Items 6 and 7 (the user's decisions of 2026-10-07) | dev-multi-control | Built, reviewed, committed (ac919f9c) |
| dev-two-tier (C15, C16) into dev-multi-control (notes 二 1) | dev-multi-control | Merged (fc962015), one conflict resolved; reviewed with the next row |
| One start function (notes 二 2, §9 item 12, D164) and MC4 · stage D's runner `multi_train` | dev-multi-control | Built, reviewed (three rounds), committed (2b370d37); touched tests 162 passed |
| Cleanup (notes 三) | — | Done 2026-10-07 (§7): seven worktrees removed, their data links unlinked first; ten branches deleted with `-d` |
| MC4 · the time readouts (O18) and `multi_validation` | dev-multi-control | Built, reviewed (three rounds), committed (fccbab8d) |
| MC0's D73 checks on the formal artefact (notes 三) | dev-multi-control | Passed 2026-10-07 (§9) |
| D149's check on real windows (notes 三) | dev-multi-control | Passed 2026-10-07: 200 of 200 windows bit for bit (§9) |
| MC1's census on real data (notes 三) | dev-multi-control | A sample of 20 anchors measured; the full census stopped at the user's word ("sample, … finish it as soon as possible"); a stated sample of 500 anchors an airport and split running in 6 processes (§9) |
| The full ts suite on dev-multi-control (fccbab8d) | dev-multi-control | 1,963 passed, 1 skipped (8 workers, 13.2 min, 2026-10-07) |
| dev-multi-control into dev-two-tier | dev-two-tier | Merged at the user's word ("合进去", 2026-10-07): b967d2b1, no conflict, no code differs from the tested branch |

### 0.1 Handover: where the next implementer of stage D starts (2026-10-07)

Stage D's first implementer stopped here at the user's word; the next one takes the work from this section. Paths are
relative to `4dTrajectory/ts_transformer/` unless they start with `4dTrajectory/` or `/`.

**Read first, in this order**: `docs/two_tier/notes/stage_d.md` (the current orders, 2026-10-07: part 一 is done,
二 and 三 remain); `docs/two_tier/design/multi_control.md` (§0 status, D140–D172, O18–O19, §3, §5, §7, §11 MC5–MC6);
`docs/two_tier/review_guide.md` §3 (the procedure and the severities S1–S3, outline D131); this log §11–§12 (what was
built and measured last); `docs/two_tier/design/requests_from_d_to_designer.md` (the readings still open).

**State of the code** (everything merged; nothing uncommitted):
- `dev-two-tier` = `dev-multi-control` = **246eca69** (fast-forwarded at the user's word). The worktree
  `/home/supercomputing/studys/thesis/.claude/worktrees/multi-control` (branch `dev-multi-control`, data trees linked to
  the live ones) is clean and kept for stage D; merge dev-two-tier into it before new work.
- Built and reviewed: MC0–MC4 (version 1: windows of several commanded aircraft, the loop, the credit, the runners
  `multi_train`, `multi_validation`), the spans of D146 (5 / 10 / 20 min mixed, one span a batch, select set and
  readouts by span, 6cc49ad1), D166 item 19 (a formal start refuses a source with rounds to run), MC5's runner
  `experiments/multi_profile.py` (bd6dd1be), D172 (fffe900d: the workers sized from the profile with `--profile`, each
  worker's GPU capped at its share; `MultiSettings.batch_rows` one number a span; the profile speaks each span's batch
  once), the merge of stage C's C17–C20 (f2bb4a5d: C18's `workers_fit`, C19's `settings_type`, stage D on
  `train_pass`'s defaults — no clipping, one pass).
- Tests: the changed files' tests and `test_architecture` pass (last: 114 + 2, beside stage C's campaign, low
  priority). **The full ts suite has not been run since fccbab8d** (1,963 passed then): run it once the host is free,
  before MC6 (`-n 8 --dist worksteal`, `OMP_NUM_THREADS=1`, about 3 min).

**Work left, in order**:
1. **MC5, the profile at the formal size** (orders 二; multi_control §11 MC5). Only when the host and the GPU are free:
   check `ps aux | grep run_ts.py` and `nvidia-smi` first (stage C's campaigns run from their own worktrees; on
   2026-10-07 evening `post_train --method …` was running). Run from the multi-control worktree at 246eca69 (or later
   merged code) **in a systemd unit** (`systemd-run --user --unit=mc5-profile …`; nohup/setsid died with a session
   once), output to a NEW directory under `4dTrajectory/outputs/POOLED/multi/` (the user, 2026-10-07: MC6 reads it
   with `--profile`; `multi_profile` refuses an existing directory; the old `/tmp/claude-1000/mc5/profile` holds the
   old integer `batch_rows` and `profiled_fit` refuses it):

       python run_ts.py multi_profile --prior 4dTrajectory/outputs/POOLED/prior/prior_base_20261006/base/run \
         --instructions 4dTrajectory/outputs/POOLED/instruction_language/v12_20261005 \
         --executor 4dTrajectory/outputs/POOLED/executor/v17_20261005 \
         --windows 4dTrajectory/outputs/POOLED/post/windows_20261006 \
         --start-campaign 4dTrajectory/outputs/POOLED/post/post_train_20261006 --start-round 6 \
         --out 4dTrajectory/outputs/POOLED/multi/profile_<YYYYMMDD> --rounds 1 --windows-real 1000 --windows-compressed 1000 --spans-s 300 600 1200 \
         --c-min 0.8 --batch-rows 64 48 24 --continuations 8 --seed 2027 --prior-lr 1e-5 --traffic-lr 1e-4 \
         --weight-decay 0.01 --update-groups 4 --data-sentences 64 --select-per-airport 100 \
         --device cuda --speak-workers 3

   (`OMP_NUM_THREADS=1`; start round 6 and seed 2027 as the smoke: the profile's start, not the formal one; 3 workers is
   Claude's estimate from §12 — each span's batch about 2 GiB at these rows; the profile records what 1…N need and
   stops by name where N does not fit, its record written up to there: then run again with fewer). It writes
   `profile.json` after each part: the draw and batches by span; one worker's measure on each span's first batch and the
   largest (`workers.measured.batches`: each batch's time and GPU peak), the pass's memory, what fits for each N; the
   round's speaking time with the workers; the selection readout read twice (draws 0, 1) and its time; the pass's time;
   `spread` (W per aircraft by span and over all, its standard error, the paired one of the two draws, `units_for` the
   select windows for a standard error of 0.01 / 0.02, in all and an airport). Expect several hours (one round at the
   formal size: about 12.6 h of speaking in one process at 64 rows, §12, divided by the workers). Arm a Monitor on its
   end and errors; the user must not learn hours later that it ended.
   After it ends: the profile's directory read-only, with a `SHA256SUMS` (MC6 reads it; a campaign refuses a profile
   of other inputs or settings, `profiled_fit`).
2. **Record it and propose MC6's settings to the user** (as a question, AskUserQuestion): the profile's numbers in a
   new section of this log; then the proposals — `--speak-workers` (the largest N with an empty `workers.short[N]`),
   `--batch-rows` (keep 64 / 48 / 24 unless one span's GPU peak in `workers.measured.batches` is far from the others),
   `--select-per-airport` (from `spread[<span>].units_for["0.01" or "0.02"].paired_se.an_airport`, the user picks the
   resolution), `--rounds` (from the round's time), and the start (`--start-campaign`, `--start-round`: a formal,
   finished campaign of stage C, D164 and D166 item 19; the user chooses which campaign and round; the seed must differ
   from that campaign's seed, 1337 for `post_train_20261006`). The user also sets the criterion that chooses the round
   (D7). MC6 keeps D142 / D143 (O19: no value function in stage D for now).
3. **MC6, the formal campaign** (orders 三; outline D163), after the user's settings and merge: a detached run worktree
   at the merged commit, data trees as absolute links to the live data, a clean tree, `multi_train` without `--smoke`,
   with `--profile <the profile>` and `--speak-workers N`, in a systemd unit, output under
   `4dTrajectory/outputs/POOLED/multi/<campaign id>`. Write its intent into `docs/experiments/intents.json` before any
   publication. After it ends: data read-only and `SHA256SUMS`, each round's selection readout into this log, the run
   worktree unlinked and removed; report to the user, who chooses the round (D7); then `multi_validation` (the val days,
   read once, with its claim) only at the user's word.
4. **Later** (not now): MC7 is fronter's (the Training view of stage D, frontend F3–F4); MC8–MC10 (version 2, D153)
   after MC6.

**Open for the user** (in the requests note): readings 1–5, 8–17, 24, 26–28, 37–45 (none blocks MC5 or MC6); whether
stage C's campaigns should read a profile too (D172 covers stage D only; stage C measures once and records it, D167).

**Rules that bind this work** (the user's, 2026-10-05…07):
- Each step: the changed files' tests and an independent review **started together** → fixes → an explicit-path commit
  (never `git add -A`; `git diff --cached --stat` first) → one line in this log. The reviewer is never the author
  (`opus-code-reviewer`).
- A review is scoped to the changed lines (a merge: its conflict resolutions); S1 fixed, S2 fixed when cheap (else an
  O item), **S3 one line and no fix**; a second round only on what was fixed. A large change goes to 2–3 reviewers by
  area.
- While another stage's campaign runs: no GPU, no full suite, only the changed files' tests with at most 2 processes at
  `nice -n 19`; the host's RAM is shared (31 GB; a stage C campaign held 20–24 GB).
- Measure once and reuse (the profile); never add a per-run cost (a check, a measure) without stating it to the user
  first.
- No `.get(key, default)` fallbacks, no compatibility code (one standing exception: a new campaign setting may default
  to the old code's behaviour); tests never write under live roots.
- The design text is not the implementer's: only this log, multi_control §0.3 and other documents' §0.3 status lines,
  and the requests note (rewritten whole, the decided items deleted). Docs commit straight to `dev-two-tier`.

**Code to know**: `multi/{windows,census,separation,tokens,credit,timing}.py`; `experiments/multi_train.py`
(`MultiSettings`, `span_counts`, `span_batches`, `draw_windows`, `speak_batch`, `selection_windows`, `read_batch`,
`selection_readout`, `profiled_fit`, `require_finished`, `round_model`, `stage_d()`, `add_arguments`, `settings_from`);
`experiments/multi_profile.py` (`profile`, `spread`, `ratio_se`, `windows_for`); `experiments/multi_validation.py`;
`experiments/post_train.py` (`Stage` with `batches`, `train`, `pass_memory`; `batches`, `measured_batches`,
`Speakers(…, gpu_budget=)`, `_measure`, `workers_fit(held_now=)`, `campaign_model`, `round_start`, `start_of`,
`source_campaign`); tests `tests/test_multi_{train,profile,validation,control,windows}.py`.

**Scratch** (`/tmp/claude-1000/`, not durable): `mc4/smoke` (the smoke campaign, 3 rounds), `mc5/profile` (the profile
at 64 rows, §12; old format), `d73/`, `d149/`, `mc1/s500/` (earlier checks). No test stack of stage D is running.

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

## 8 After the user's decisions and the merge of C15 and C16 (2026-10-07)

| Commit | What | Review |
|---|---|---|
| 054727db | MC4, first part: the campaign's round is the stage's throughout — `Stage.speak_batch`, `read_batch`, `part_width`; the workers, the measure and the one-process mode run the stage's parts; `run_campaign` hands the stage to its speaking and readout and refuses workers of another stage | Three rounds: S2 fixed (one process and the workers on the same stage's batches; the refusal; tests that can fail), then clean |
| ac919f9c | Items 6 and 7 (the user): a loop landing on a sealed test day left out and counted (`LandingIndex.with_landing`, its other checks first); a commanded leader judged once more over its threshold at the row after it lands (`WindowLoop._judged`) | No S1; S2 fixed (tests of the judged set's order with a recorded aircraft, of what the loop gives the rule of who answers, of `_landed`'s rows) |
| fc962015 | dev-two-tier (C15, C16; 67cd9bc8) merged into dev-multi-control: the conflict in `post_train.py` resolved with both kept — `Stage.start` (a new campaign's start: stage C's `campaign_start`) beside `Stage.start_model` (the shape a resume and a worker load into), `Stage.read_batch` with C15's draw, `counted_ends` kept; the stubs of C15's and C16's tests take the stage | Stage C's tests with C15's and C16's: 37 passed; the review with the next step's |

| 2b370d37 | Notes 二 2–3 and MC4: `round_start` (one start function: bytes, the source's rules via `source_campaign`, identity, formal from formal via `Context.formal`, the seed), `campaign_start` and stage D's start call it, `start_of`, `open_round` removed; `experiments/multi_train.py` (stage D's campaign); `WindowResult.crossing`; the readout's landed aircraft judged as the loop judges it, as a recorded one in the pairs (`classify(…, over)`) | The merge, this step and the runner in one review: no S1; S2 fixed in three rounds (the readout's landed leader where and as the loop judges it, faulty steps once a window, the measure's series, tests of the readout with several aircraft and of main's wiring), then clean. Tests: stage C's (C15, C16 and D149's digests) and stage D's, 162 passed |

Before (in review, now committed above): the one start function (`round_start`, `source_campaign`, `start_of`; `campaign_start` and stage D's start
call it; `open_round` removed; `Context.formal`) and stage D's runner `experiments/multi_train.py` (requests items
29–34).

## 9 The parts on real data (notes/stage_d.md 三; 2026-10-07, after C15's ceiling readout)

The host and the GPU free (rule 13: 19 GB of RAM free, the GPU idle, only idle backends); every job on the CPU with one
thread, its outputs in a scratch directory (`/tmp/claude-1000/`), the code of `dev-multi-control` (fccbab8d).

1. **MC0's D73 checks** on the formal artefact (`v12_20261005`, executor `v17_20261005`), the code with the start's join
   ticks (vocabulary §6 item 5): `executor_conformance` — conforming in every way of flying (batch, staggered, single,
   moved; 285 flights, 0 beyond the bounds, the largest difference 2.2e-06 m); `closed_loop_start_check` on train and
   select at Δ = 2, 4 and 8 s, 50 flights an airport, seed 1337 — 250 flights each, 0 failed, the largest position
   difference 0 m: PASSED both.
2. **D149's check on real windows** (§6.1): 10 windows of each of stage C's kinds and each airport of the train days of
   the formal census (`windows_20261006`; stage C's own draw, seed 1337), each batch's branch round (K = 8) with C10's
   round-0 weights, on the CPU with one thread, by the code before the generalisation (c7a0b6b0, in a detached
   worktree, removed after) and by today's: each window's first-pass ends and branch groups digested by field
   (`test_post_generalised._Digest`'s fields, stage D's added ones left out). **200 of 200 windows identical**, the
   same windows drawn, no window whose second pass differed. Time: 248 s for the old code, 255 s for the new (+3 %).
   (A one-off script beside the two checkouts, not a runner: it runs on a commit that has no runner of it.)
3. **MC1's census, a sample first** (requests item 18): `multi_windows --sample 20` (20 anchors of each airport and
   split, seed 1337, every span and kind, the records and the baseline): 167 s with the closed loop's checks. The full
   census is 40,530 train and 6,199 select anchors, so about 6–7 h in one process (about 0.5 s an anchor); run in full
   (not a sample), detached, started 2026-10-07. The sample's commanded aircraft of a window (train, real): L = 0:
   1; 5 min: p50 2, p90 3; 10 min: p50 2, p90 4, largest 8; 20 min: p50 4, p90 7, largest 11. On the records
   `records_kept` is 0 (as it must be: on the records every loss is the records'). The full census's report goes to the
   user for O16.

## 10 MC1's census (a stated sample, 2026-10-07)

The user's choice ("sample, and you may use more cpu, just finish it as soon as possible"): 500 anchors of each airport
and split (2,500 of 40,530 train anchors, 2,500 of 6,199 select anchors), seed 1337, every span and kind, in 6 processes
(the longest 537 s), on dev-multi-control's code (fccbab8d); the records in `/tmp/claude-1000/mc1/s500/`. A compressed
window needs a later aircraft (an anchor without one has no compressed form), so its windows are fewer. The pairs: cc
commanded–commanded, ca a commanded aircraft responsible against a recorded one, rk only the recorded one responsible
and the records kept their separation (D145: the loop charges the commanded one), ro the records lost it too. On the
records rk is 0 by construction; the baseline is the commanded aircraft flown on their stored closed-loop sentences (the
labelled words: not a model's result, §5 item 4).

| Split | L | Kind | Windows | Left out (anchor, later) | Commanded p50 / p90 / largest | Recorded at a first step p50 / p90 | Records: cc / ca / rk / ro (% of windows) | Baseline: cc / ca / rk / ro |
|---|---|---|---|---|---|---|---|---|
| train | 0min | real | 2500 | 2 (anchor 2, later 0) | 1.0 / 1.0 / 1.0 | 1.0 / 3.0 | 0.0 / 3.2 / 0.0 / 2.2 | 0.0 / 4.0 / 0.9 / 1.5 |
| train | 5min | compressed_0.6 | 1413 | 31 (anchor 0, later 31) | 2.0 / 3.0 / 5.0 | 1.0 / 2.0 | 16.5 / 11.9 / 0.0 / 3.0 | 13.7 / 13.5 / 1.7 / 2.7 |
| train | 5min | compressed_0.8 | 1413 | 6 (anchor 0, later 6) | 2.0 / 3.0 / 5.0 | 1.0 / 2.0 | 7.9 / 9.7 / 0.0 / 2.8 | 8.0 / 11.0 / 1.4 / 2.1 |
| train | 5min | real | 2500 | 5 (anchor 2, later 3) | 2.0 / 3.0 / 5.0 | 1.0 / 2.0 | 2.3 / 4.6 / 0.0 / 2.5 | 2.6 / 5.3 / 1.4 / 2.0 |
| train | 10min | compressed_0.6 | 1974 | 88 (anchor 1, later 87) | 3.0 / 5.0 / 9.0 | 1.0 / 2.0 | 23.0 / 12.1 / 0.0 / 5.3 | 21.8 / 13.1 / 2.0 / 5.1 |
| train | 10min | compressed_0.8 | 1974 | 32 (anchor 1, later 31) | 3.0 / 5.0 / 9.0 | 0.0 / 2.0 | 12.5 / 9.4 / 0.0 / 3.1 | 12.4 / 10.5 / 1.9 / 2.9 |
| train | 10min | real | 2500 | 8 (anchor 2, later 6) | 2.0 / 5.0 / 9.0 | 0.0 / 2.0 | 4.7 / 4.5 / 0.0 / 3.2 | 5.5 / 5.6 / 1.7 / 2.6 |
| train | 20min | compressed_0.6 | 2313 | 140 (anchor 2, later 138) | 4.0 / 7.0 / 14.0 | 0.0 / 2.0 | 30.0 / 12.5 / 0.0 / 6.8 | 28.2 / 13.3 / 3.2 / 6.6 |
| train | 20min | compressed_0.8 | 2313 | 82 (anchor 2, later 80) | 4.0 / 7.0 / 14.0 | 0.0 / 2.0 | 17.9 / 11.5 / 0.0 / 5.0 | 17.7 / 12.4 / 2.6 / 4.9 |
| train | 20min | real | 2500 | 15 (anchor 2, later 13) | 4.0 / 7.0 / 14.0 | 0.0 / 2.0 | 7.0 / 5.7 / 0.0 / 4.2 | 8.3 / 6.4 / 2.6 / 3.8 |
| select | 0min | real | 2500 | 2 (anchor 2, later 0) | 1.0 / 1.0 / 1.0 | 1.0 / 3.0 | 0.0 / 3.6 / 0.0 / 2.0 | 0.0 / 4.7 / 1.4 / 1.4 |
| select | 5min | compressed_0.6 | 1430 | 26 (anchor 2, later 24) | 2.0 / 3.0 / 5.0 | 1.0 / 3.0 | 15.5 / 10.2 / 0.0 / 2.1 | 15.8 / 11.8 / 0.9 / 1.9 |
| select | 5min | compressed_0.8 | 1430 | 7 (anchor 2, later 5) | 2.0 / 3.0 / 5.0 | 1.0 / 3.0 | 8.3 / 8.0 / 0.0 / 2.2 | 9.4 / 9.1 / 0.8 / 1.8 |
| select | 5min | real | 2500 | 4 (anchor 2, later 2) | 2.0 / 3.0 / 5.0 | 1.0 / 2.0 | 2.5 / 4.1 / 0.0 / 2.4 | 3.5 / 4.8 / 1.3 / 1.7 |
| select | 10min | compressed_0.6 | 1992 | 72 (anchor 2, later 70) | 3.0 / 5.0 / 8.0 | 1.0 / 2.0 | 22.5 / 11.7 / 0.0 / 4.2 | 22.2 / 12.8 / 1.6 / 3.9 |
| select | 10min | compressed_0.8 | 1992 | 29 (anchor 2, later 27) | 3.0 / 5.0 / 8.0 | 0.0 / 2.0 | 12.5 / 8.5 / 0.0 / 2.6 | 13.7 / 9.3 / 1.2 / 2.3 |
| select | 10min | real | 2500 | 6 (anchor 2, later 4) | 3.0 / 5.0 / 8.0 | 0.0 / 2.0 | 4.5 / 4.4 / 0.0 / 2.9 | 6.1 / 4.9 / 1.8 / 2.3 |
| select | 20min | compressed_0.6 | 2330 | 152 (anchor 2, later 150) | 4.0 / 8.0 / 13.0 | 0.0 / 2.0 | 30.2 / 12.0 / 0.0 / 5.7 | 30.1 / 12.9 / 2.3 / 5.2 |
| select | 20min | compressed_0.8 | 2330 | 63 (anchor 2, later 61) | 4.0 / 8.0 / 13.0 | 0.0 / 2.0 | 18.2 / 10.5 / 0.0 / 5.6 | 20.2 / 11.4 / 2.7 / 4.9 |
| select | 20min | real | 2500 | 11 (anchor 2, later 9) | 4.0 / 8.0 / 13.0 | 0.0 / 2.0 | 7.4 / 5.7 / 0.0 / 4.6 | 9.8 / 6.2 / 2.7 / 3.9 |

## 11 MC4 · the windows of several spans (notes/stage_d.md 一; 2026-10-07)

- dev-two-tier (8eeaf4bf) merged into `dev-multi-control` (a fast-forward). The windows of several spans (D146 after
  O16) and D166 item 19 built, reviewed (opus, two rounds: S2 the memory measure of every span, fixed by
  `post_train.measured_batches`; S2 the readout of two spans untested, tested; clean), committed **6cc49ad1**: each
  kind's count in equal parts of the spans (1,000 → 334 / 333 / 333), each window's span drawn with the round's numbers,
  the counts by kind and span; a batch of one span up to `batch_rows` commanded aircraft (`post_train.batches`, stage
  C's batches unchanged, checked by the reviewer on 3,000 random inputs); the select set, the readouts and the
  validation readout by span; the identity with `spans_s`, `per_kind`, `per_span`; `multi_train.require_finished`.
  Tests: the post_* and multi_* files and `test_architecture` (D149's digests and MC2's check with stage C's numbers
  among them) passed. Readings 37–43 in the requests note (rewritten, the decided items deleted).
- MC4's smoke (notes/stage_d.md 二 1; scratch `/tmp/claude-1000/mc4/smoke`, from 6cc49ad1 with the token part's fix of
  bd6dd1be in the tree for the resume): `multi_train --smoke`, start round 6 of `post_train_20261006`, seed 2027, 6
  windows of each kind over the spans 300, 600 and 1200 s, `batch_rows` 64, 2 select windows an airport and span; two
  rounds, exit 0 (round 0 about 5 min: 6 batches of one span each, 46 commanded aircraft, 31 landed, 14 lost separation,
  1 timeout); the draw 2 of each kind and span, no shortfall, the windows without a later aircraft counted by span (2,
  1, 1). The resume (`--rounds 3`, 2 speaking workers): the O15 measure on the first batch of each span (9, 13 and 18
  rows; one worker's host peak 1.5 GB, GPU 2.0 GB; the pass 2.1 GB), round 2 spoken by the workers, exit 0; round 2's
  identity holds three rounds, the spans and their counts, stage C's round as its start. The smoke's log warned of
  `log1p` (invalid value) on every call: the token part read `Heard.inputs` at 0, so the unused times since went
  negative (no value read changed); fixed in bd6dd1be.
- MC5's runner `experiments/multi_profile.py` built, reviewed (opus, two rounds: S2 the workers not refused where they
  do not fit, S2 the worker's measure spoken with the stage's start model and not the round's — the campaign's O15 check
  too, fixed by `Speakers.measure(…, model)` and `post_train.campaign_model`, S2 the spread over all by window where an
  anchor's windows of the spans are nested — now an anchor the unit; clean), committed **bd6dd1be**. MC5 at the formal
  size started 2026-10-07 (scratch `/tmp/claude-1000/mc5/profile`): 1,000 windows of each kind, spans 300, 600, 1200 s,
  `batch_rows` 64, 4 speaking workers, 100 select windows an airport and span read twice, start round 6, seed 2027.

## 12 MC5 · the profile at the formal size (2026-10-07, bd6dd1be; scratch `/tmp/claude-1000/mc5/profile/profile.json`)

Settings: 1,000 windows of each kind, spans 300, 600, 1200 s, c_min 0.8, `batch_rows` 64, K = 8, update groups 4, data
sentences 64, 100 select windows an airport and span; start round 6 of `post_train_20261006`, seed 2027; 4 speaking
workers asked. Started 05:39 UTC, stopped by name at part 2 at 06:19 UTC (exit 1): 4 workers do not fit on the GPU.

- **The draw** (round 0): each kind 334 / 333 / 333 over the spans, no shortfall; left out (D146) real 0 / 2 / 3,
  compressed 3 / 6 / 10; without a later aircraft (no compressed form) 245 / 83 / 24.
- **The batches** (64 rows, one span each): 23 / 31 / 50 batches of 668 / 666 / 666 windows; almost every batch full (no
  window above 64 rows).
- **One batch of each span** in the campaign's process (GPU, two passes):

  | Span | Windows | Time | Groups (informative) | Groups' bytes | GPU peak (allocator) | Host peak RSS |
  |---|---|---|---|---|---|---|
  | 300 s | 34 | 191 s | 29 | 28 MiB | 1.37 GiB | 4.69 GiB |
  | 600 s | 24 | 284 s | 30 | 34 MiB | 1.89 GiB | 4.86 GiB |
  | 1200 s | 15 | 641 s | 37 | 51 MiB | 3.57 GiB | 5.02 GiB |

  The round's speaking in one process, from these: 23 × 191 + 31 × 284 + 50 × 641 s ≈ 12.6 h (an estimate: one batch a
  span); the 20 min windows are 71 % of it.
- **One speaking worker** (`Speakers.measure` on the first batch of each span, with round 0's model): its peak on the
  GPU 3.71 GiB (CUDA context and allocator; 0.51 GiB after), on the host 1.69 GiB; besides in a round 0.02 GiB (the
  readout's model) and 0.13 GiB of series (4,167 flights). The pass's update (4 groups, 96 groups written): its peak
  2.18 GiB. Free: host 15.8 GiB, GPU 5.96 GiB.
- **What fits**: 1 worker; 2, 3, 4 workers need 7.5, 11.2, 14.9 GiB of the GPU against 6.5 GiB (the host would hold
  about 8). The GPU binds, through the 20 min batch: its peak grows with the batch's ticks, not only its rows.
- Not measured (the profile stopped before part 3): the round's speaking by the workers, the selection readout and its
  spread, the pass's time.
- The user (2026-10-07): a campaign's workers are sized from the profile, never measured before each run (the measure
  of every span was about 20 min a launch); built on `dev-multi-control` (`multi_train.profiled_fit`, `--profile`, each
  worker's GPU capped at its share), in review.
- The user's decisions after the profile (2026-10-07): (a) the workers are sized from the profile, never measured
  before each run (the user: "每次跑之前花20分钟测内存" — the measure of every span, added by the review of 6cc49ad1, cost
  about 20 min a launch and a resume); (b) one batch size a span, 64 / 48 / 24 rows for 300 / 600 / 1200 s; (c) the
  profile speaks each span's batch once. Built, reviewed (opus, two rounds: S2 each measured batch's own GPU peak, S2
  four test gaps; clean), committed **fffe900d**: `multi_train.profiled_fit`, `--profile`, each worker's GPU capped at
  its share (`Speakers(gpu_budget=)`), `MultiSettings.batch_rows` a list, `span_batches`; stage C's own measure
  unchanged. Tests run beside C17 in one process at the lowest priority.
- dev-two-tier merged into `dev-multi-control` (**ea9d5217**; C17's `Settings.method`, `select_seed`, `STAGE_C_LANDED`;
  the conflicts in `post_train` resolved by keeping both: `Stage.batches` beside `Stage.train` and `pass_memory`;
  `test_post_train` and `test_multi_train` 53 passed). The profile is run again with 64 / 48 / 24 after C17 ends.
- notes/stage_d.md 一 (2026-10-07, after D172 and O19): dev-two-tier (C17–C20) merged into `dev-multi-control`,
  **f2bb4a5d**: `workers_fit` is C18's (the check of workers on the CPU kept; with nothing held, `passed["now"]`
  taken out once, so `profiled_fit`'s own share check, the same condition, dropped; its GPU budget kept); `STAGE_C`
  with both `batches` and C's `train` (`clip_norm`, `epochs`); stage C's `main` with C18's recorded measure, the
  measure spoken with the round's model; `settings_type=MultiSettings` kept; stage D runs `train_pass`'s defaults (no
  clipping, one pass). Tests of the changed files and `test_architecture` (2 low-priority processes beside stage C's
  campaign): 114 passed, 1 failed (stage D's runner test replaced `MultiSettings`, now records the spans through
  `settings_from`; passed). Reviewed in parallel (opus: no S1; S2 `test_post_landed.py` to be staged — done; two S3).

## 13 The census's step loop as one generator; MC5's profile launched (notes/stage_d.md 一, 二; 2026-10-08)

**MC5's profile, launched 2026-10-08 21:17 local** (orders 二): the host free (no campaign, no GPU job; only fronter's
test services), from a detached run worktree `.claude/worktrees/run-mc5-profile` at `ad564a87` (dev-two-tier, holding
fffe900d; data trees linked), systemd unit `mc5-profile-211748`, the command of §0.1 item 1 with
`--out 4dTrajectory/outputs/POOLED/multi/profile_20261008` (start `post_train_20261006` round 6, seed 2027, batch rows
64 / 48 / 24, 3 workers on the GPU); launch script and log in the session's scratch (`tmp/mc5/`). The checks passed
(the labeller's 336 reference flights, the executor's, the closed loop's 258). Running; MC6's proposals follow its end.

**The generator (orders 一; frontend D177 (12)):** `dev-two-tier` fast-forwarded into `dev-multi-control` first
(`ad564a87`, no conflict). In `multi/census.py`, `judged_steps(window, positions, separation, finals, step_s)` yields a
`JudgedStep(step, aircraft, commanded, losses)` for each Δ step from the anchor's first predicted step to the end of
the positions at which a commanded aircraft is in the air; its `losses` are `StepLoss(pair, keys, loss)`, each loss
of the judge that holds a commanded aircraft, its pair as `classify` reads it (a landed commanded aircraft over its
threshold counted as recorded; `recorded_only` before the records are read). `window_losses` (the readout's census,
through `multi_train.window_losses_of`) counts its steps and splits off `records_kept` from it as before. Fronter's
`census_losses` becomes `[(s.step, x.pair, list(x.keys), x.loss) for s in judged_steps(...) for x in s.losses]` (the
reviewer checked the same list, order and objects); `multi_training_export.py` unchanged (fronter's, F5).
Tests (`tests/test_multi_windows.py`): the census counts bit for bit as the old loop, kept in the test as the reference
(`_counts_before`), on twelve fixed inputs reaching every pair and every branch of the loop (a landed commanded aircraft
over its threshold, losses holding none, two recorded-only losses in a step, a recorded-only loss on the records); the
generator's steps on a flight alone and with a follower 8 s behind. `test_multi_windows`, `test_multi_train`,
`test_multi_training_export` (the MIRROR test) and `test_architecture`: 78 passed (two processes, beside the profile).
Reviewed in two rounds: the first found one S2 (the fixed inputs reached none of the moved branches), fixed with the
reviewer's three inputs; the second none. S3 left (one line each): the last checks compare with `window_count`, which
runs on the generator; `JudgedStep.aircraft` is read by no consumer and a frozen dataclass of numpy arrays cannot be
compared; the export will still repeat the positions rule (a helper in `multi_train` would remove it, F5); `judged_step`
and `judged_steps` in one module read alike; a case picked by its place from the end of `_cases`; the two-recorded-only
count also met on the records. Commit `897e433f`.

