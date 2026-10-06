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
| MC0 · stage C (post-training §9) | dev-multi-control | The reference written (3a9b7153); the generalisations next |
| MC1 · `multi/`, `experiments/multi_windows.py` | dev-multi-control | Not started |
| Stage B's steps after C10 (notes/stage_d.md "二") | dev-two-tier-v4 | Wait for C10's end |
| MC0's checks and MC1's census on real data ("三") | dev-multi-control | Wait for C10's end and "二" |

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

## 4 MC0 · stage C (in progress)

The reference of D149 (3a9b7153, `tests/test_post_generalised.py`): stage C's window loop on a real window, window A
and window B with their samples, two branch rounds and the samples of a group, as digests written by c7a0b6b0's code
(the CPU, one thread). It passes on MC0 A and B: stages A's and B's changes leave stage C's outputs as they were.
