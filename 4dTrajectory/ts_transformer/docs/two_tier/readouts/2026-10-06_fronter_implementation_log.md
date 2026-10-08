# Fronter's implementation log

The log of the Training view's implementer (frontend D154; outline §5 rule 10). Branch `dev-frontend`, worktree
`.claude/worktrees/frontend`, made from `dev-two-tier` (`c633acc0`), with `dev-two-tier` merged in before each milestone.
Readings where the design says nothing are proposals in `docs/two_tier/design/requests_from_fronter_to_designer.md`.
Every commit was reviewed by an opus reviewer that did not write it (review guide §3 step 6, outline D131).

| Date | Milestone | What was built | Commits | Tests | Review |
|---|---|---|---|---|---|
| 2026-10-07 | F1 the cursor slider (D155) | `components/training/CursorSlider.tsx` and `data/trainingSlider.ts`: one slider under the readouts in every stage (drag, click, ← → one row, Shift ten, Home/End; `role="slider"`). Marks: the first predicted step in every stage; the corrections (A); the go-around line on a sample's tab (B, replacing its clickable strip, whose "At the cursor" line stays); the loss of separation (C). `WindowCursor` keeps a window's instant over a change of round (D129's start at the window's row 0 kept) | `e6f5c5fd` | Vitest 1102, tsc clean | One S2: Labelled ↔ the round it is read on did not clamp. Fixed and re-verified |
| 2026-10-07 | F2 windows of several commanded aircraft (D156) | Sample v4 / index v3 (`index_post_v3.json`, kind `training-windows`): a window's commanded aircraft are a list (stage C's export writes one), each with its join offset, move, first step and rounds (reward, silent row); the window's losses carry both aircraft, who answers and whether it costs W. `POST /autopilot/window-segment` names the aircraft (answer v2). The reader keeps two clocks (`onAircraftClock` / `onWindowClock`). The session, scene layer, bar chips and live request read the list. Stage C's fixtures were written again by the export and the backend | `05963721`, `1452cccb` | Vitest, tsc; pytest `test_post_training_export`, `test_window_segment`, `test_training_results` 21 passed; `test_training_export`'s check of the flown blocks reads them under the aircraft, same digests | One S2: no test told the clocks apart (the fixture's clock is 0). Fixed with `onWindowClock`, a test at clockS = 300 and a `firstStepS` check in the reader. Re-verified |
| 2026-10-07 | F0 the user's three corrections (D159) | The details page has two sections, the set and the experiment and the models' statistics (`data/trainingStatistics.ts`, `components/training/StatisticsSection.tsx`). The other section parts and their tests are removed. Every ground line is dashed, ≤ 3 px, at 0.6 opacity (`groundLine`). One colour map `TRAINING_SENTENCE_COLOR` is used by the 3D flown track, trace, end and aircraft, the read-back charts, both legends and the tabs' swatches | `01a8ab58` | Vitest 127 files / 1110, tsc clean | Two S2s: B's readout counted another population than the set's; the tests could not tell the readings apart. Both fixed and re-verified |
| 2026-10-07 | F1's three small points (stage D's browser check) | Stage A's intent line names `index_v5.json`; the read-back title names the sentence ("sample 0 · Δ 4 s"); `.training-readout-line` has its layout (B's "At the cursor" no longer overlaps) | `8c4d6af4` | the changed tests | No S1/S2 |
| 2026-10-07 | D160 (11), (12): the user's two changes to the accepted readings | The statistics' go-around column is "go-arounds / sentence" (a number per sentence, its count beneath); `trainingOutcomeColour` gives a landed outcome the pass green `#4ade80`, so teal means only the closed loop (tab dots, the bar's chip and end mark, the read-back's outcome line, other sentences and rounds) | `da1c6bba` | Training Vitest 87 files / 782, tsc clean | No S1/S2; the reviewer's colour distance between the green and the post-trained yellow-green is in the requests note |
| 2026-10-08 | The start's name after D162 (frontend §4.3; orders of 2026-10-08, item 一) | The window reader reads `model.settings.start` (required, null = the base, else `{campaign, round, checkpoint_sha256}`); `startName` names the start in the tabs, statistics rows, lines and 3D titles; a start from a round is post-trained (yellow-green, requests item 2). Fixture `start_from_round.json` written by `post_train.start_of`; the stage C fixtures rewritten by their writers (C21/C22's settings had moved them). Export docstring | `f10e9ee5` | Vitest 50 files / 404, tsc clean; pytest `test_post_training_export`, `test_training_export` 24, `test_window_segment`, `test_training_results` 16 | Round 1: one S2 (the from-a-round shape tested on a hand-written literal), fixed with the writer's fixture; round 2: no S1/S2 |
| 2026-10-08 | The start's formal readout (requests item 3, the user's word 2026-10-08) | The results route answers `rounds.start` (`aeroviz_backend/training_results.py`): null from the base; from another campaign's round, that round's `selection_readout` when the two campaigns' `select_seed` and `select_per_airport` are equal (a mirror of `post_train.close_value_round`'s condition), else `why`. The statistics' start row reads it, with a note. Also `7c917285`: stage C's fixtures written again after C23 (two settings keys) | `a9d296d5`, `7c917285` | Vitest 59 files / 566, tsc clean; pytest `test_training_results` 9, `test_window_segment`, `test_post_training_export`, `test_training_export` | No S1/S2; S3s applied (mirror label, unused fields dropped, the source round pinned). Every real campaign record has the keys read (only `post_train_20261006.aborted-…` lacks `select_seed`; it is no source) |
| 2026-10-08 | Stage C's first formal set (orders of 2026-10-08, item 二; the user's choice P55) | Speed readout `4dTrajectory/outputs/POOLED/speed/post_seg60_r5_20261008` (model_speed on `post_seg60_20261007` round 5; about 75 min: CPU / GPU, batch 1 / 400); intent line `windows_seg60_r5_20261008` (`d2491d66`); export `post_training_export --rounds start 5 --split select --kinds real A D B --per-airport 10` from a run worktree at `1de73bb0`: KMSY 32, KRDU 32, KSJC 34, KSMF 32, KSTL 33 windows (163); the main backend and vite restarted under the user's supervisor (children only); run worktree removed, its links unlinked first | — | — | Browser check (one-shot sonnet, 5173/8765, KRDU): A–F passed — the set loads, tabs `Start (post_train_20261006 r8)` and `r5`, a loss's red tick and 3D line, a live word (12.0 s flown, 49 ms), the statistics' start row read from C10 r8 with its note; G (console) not read |
| 2026-10-08 | F3 stage D's parts of the window view (D156, §5.1–§5.6, §6.1) | Stage switch D (`index_multi_v1.json`, `post/training_files.MULTI_FILES`); aircraft strip, `[` `]`, scene pick; the cursor on the window's clock; D's readouts, list item, silence band; 3D: other commanded tracks and points (from the row they join), silence dashed slate, every loss by pair (dashed unanswered); bar chips (k of n, silent from row, W) and notes; `stage=D` of the results route and D's statistics; stage C's export writes windows of several commanded aircraft (fixture `stage_c_two`, a synthetic window of two); the window segment names the stage (answer v3). D160 (7) checked against MC0: a silent aircraft answers for nothing (`post_window_loop` skips silent rows), the reader's rule holds | `a7ab9bb6` | Vitest 129 files / 1130, tsc clean; pytest export, training export, window multi, architecture, window segment, results | Round 1: one S1 (a commanded aircraft before its first predicted step missing from 3D), three S2 (loss counts' units, the bar's loss, a negative cursor), fixed; round 2: no S1/S2; S3s applied or noted. No browser check (§8: stage D's is F4's, on a real set) |
| 2026-10-08 | F4 stage D's export and a smoke set (§8 F4) | `experiments/multi_training_export.py` (stage D's sets in `index_multi_v1.json`: readout windows drawn per airport, the start and rounds, stage D's loop options, batches and numbers; the loop's answered losses plus each unanswered run once at its first step, by a declared mirror of the readout's census pinned against `window_losses_of`; no speed readout); `post_training_export` refactored (`Flying`, `window_set`, `StageExport`, `export_sets`; stage C byte-identical); a silent aircraft the executor finished written to the judge's end (found on real data: an IndexError); the statistics count an answering aircraft as lost separation; the stage D fixture from stage D's export. Smoke: a new one-round stage D campaign (`/tmp/claude-1000/f4/smoke`; MC4's smoke predates D172 and is not read), set `multi_smoke_v1` (KRDU, 3 windows) in the scratch tree | `7b3e7ae5` | Vitest 129 files / 1130, tsc clean; pytest multi export, export, architecture, results, window segment | Three rounds: round 1 two S2, round 2 four S2, all fixed; round 3 clean. Browser check on the test stack (5186/8796, stage D): A–H passed — the strip, `[` `]`, the readouts, the tabs and chips, the silence band, 3D points and tracks, a live word (48 s flown, 23 ms), the details page with its smoke banner and notes |
| 2026-10-08 | F5 sets of listed windows (D176) with D177's changes | The shared export's `--windows <list.json>` (refused with `--airports`, `--per-airport`, `--seed`, `--kinds`; each window at its place among the selection windows by its identity, flown with its readout's numbers: stage C `select_seed` + place, stage D the campaign's seed + place + member); sample v5 / index v4 (`index_post_v4.json`, `index_multi_v2.json`), cohort `drawn` or `listed`, v4 refused by name; the details page's line. D177 (1): the landed green is **#16a34a** (green-600; OKLab ΔE 23.5 from the yellow-green, 12.2 from the teal, 47.5 from the bar's surface). D177 (12, 15): stage D's export reads `multi.census.judged_steps` and `WindowLoop.fault_readings`; its private read and census mirror gone | `b8646c12` | Vitest 120 Training files / 1051, tsc clean; pytest export, multi export, architecture, window segment | Two rounds: three S2 (the listed test could not fail, `--airports` dropped silently, the positions unpinned), then one S2 (the landed-runway clause unpinned), all fixed. The browser check of the green and the listed sets comes with the exports after the merge |
| 2026-10-08 | The speed readout unbound; `model_speed` samples; F5's exports (the user's decisions of 2026-10-08) | No Training set names a speed readout any more (stages B, C, D: no `--speed`, no `source.speed`; the results route and its TS reader lose `speed`; `model_speed.speed_source` gone; format names kept, `training_files` says why). `model_speed` stage C speaks a seeded draw of the selection windows (`--per-airport` 20; `setting_draws`: a batch draws at least its batch). Exports from the run worktree at the merge (CPU): `windows_seg60_r5_20261008` in v5 (163 windows: KMSY 32, KRDU 32, KSJC 34, KSMF 32, KSTL 33; the v4 set and the v3 indexes deleted), `loss_windows_seg60_20261008` (`post_seg60_20261007` start + 5) and `loss_windows_c10_20261008` (`post_train_20261006` start + 8), each the list's 193 windows (KMSY 22, KRDU 52, KSJC 40, KSMF 38, KSTL 41); main services restarted | `2972cf9f`, merge `698e94b1` | pytest model_speed, the three exports, architecture, training results, window/prior segment; Vitest 130 files / 1133, tsc clean | One round: two S2 (a dead allow-list entry; the format names' rule unstated), fixed; S3 listed, no action (speed.json's own format moved under v1 — its one full readout `post_seg60_r5_20261008` is now unread; a batch that does not divide by the airports draws a few more). Browser: the P55 sets A–E pass (landed green #16a34a beside the yellow-green #a3e635); `loss_windows_c10_20261008` opens, its list line and intent show, tabs Start (base) / r8 (KRDU 15.4 % → 36.5 % landed), loss ticks and red lines, no speed section |

**The browser check** (2026-10-07, a one-shot sonnet agent; no screenshot entered the main context) ran on a test stack from
the worktree: vite 5186, backend 8796, and a scratch airports tree `/tmp/claude-1000/fronter-stack/airports`. The tree links
to the live data (A `closed_loop_v12_20261005`, B `prior_fold_C_KRDU_20261006` / `prior_base_val_20261006`) and holds a v4
smoke window set, `windows_smoke_v4` (KRDU, 6 windows, rounds start and 0, exported with `--smoke` from the smoke campaign
of C11 into the scratch tree only). 13 of 15 items passed:
- the slider: its keys, its marks and a window's instant;
- the tabs' swatches;
- the two-section details page, with A's formal readout 2257/2316 landed at Δ 2 s and B's samples 337/400;
- the dashed ground lines;
- the kinds' colours in 3D and in the read-back charts;
- the read-back title;
- the "At the cursor" layout;
- a live word in each stage.

Two items were not seen in this set: a loss tick (no window of the smoke set lost separation; covered by the unit tests),
and a formal readout in C (the smoke campaign lies outside `4dTrajectory/outputs`, and the page names it).

**Merged** into `dev-two-tier` by the user's word, 2026-10-07: a fast-forward to `c4def61b`. C14 was running from the
main checkout; it imports neither changed Python module. The main backend was not restarted: it still answers the
window segment v1, which only stage C uses, and no stage C set is published.

**The user's decision on requests item 10** (2026-10-07): stage B's formal readout counts every side of the prior's
selection, as built.

Waiting: F3 after stage D's MC0 is on `dev-two-tier`; F4 after its MC4.

## Handover (2026-10-07)

The first fronter session ends here; the user hands the Training view to another agent. This section is that agent's
starting point. Its orders are in `docs/two_tier/notes/fronter.md`; its design is `docs/two_tier/design/frontend.md`.

**State.** F0, F1, F2 and D160's two changes are built, reviewed and merged into `dev-two-tier` (the last merge is a
fast-forward to `43d2e992`). The commits and their reviews are in the table above. Nothing of fronter's is
uncommitted. `requests_from_fronter_to_designer.md` holds one open item (the landed green beside the post-trained
yellow-green, ΔE 9.5).

**Where.**
- Branch `dev-frontend`, worktree `.claude/worktrees/frontend`, level with `dev-two-tier`. Keep it for F3 and F4.
- Its ignored data trees (`data`, `trajectory_data_process/outputs`, `4dTrajectory/outputs`,
  `aeroviz-4d/node_modules`, `aeroviz-4d/public/data/airports`) are absolute links to the live data. Write nothing there.

**The next milestones.**
1. **F3, stage D's parts of the window view** (frontend §5.1–§5.6, §8 F3). Start once stage D's MC0 is on
   `dev-two-tier`. Merge `dev-two-tier` into `dev-frontend` first. Things to carry in:
   - Check again, against MC0's code, whether a silent aircraft can answer for a loss (D160 (7)). The window reader
     today refuses an aircraft that answers two losses in a round, and one that is silent while answering none
     (`data/trainingWindowSample.ts` `parseWindow`).
   - The cursor across a change of the selected aircraft (§6.1). `WindowCursor` (`components/training/
     TrainingWindowSession.tsx`) keeps an instant per window on ONE aircraft's flight clock. With several aircraft it
     must convert through the window's clock: `atS − old.clockS + new.clockS` (`onAircraftClock` / `onWindowClock`).
   - The fixtures come from stage C's export on a synthetic window of two commanded aircraft (§8 F3), through post-training
     §9 items 1, 3 and 8. Stage C's export today writes one aircraft (`window_payload` in
     `experiments/post_training_export.py`); the v4 format already holds a list, `c`, `joinS`, `shiftS`,
     `silentFromRow`, and losses by pair with `answering` and `costsW`.
   - Stage D's index is `index_multi_v1.json`, in the same format as stage C's (`post/training_files.py`: the set kind
     `training-windows` is shared; stage D gets its own `TrainingFiles` with that index file).
   - The window segment's request already names `aircraft` (`aeroviz_backend/autopilot_segment/window.py`, answer v2).
2. **F4, stage D's export and sets** (§8 F4), after stage D's MC4.

**How things are done here.**
- Fixtures are written by the writers, never by hand. Stage C's are rewritten with `AEROVIZ_WRITE_FIXTURES=1` on
  `tests/test_post_training_export.py -k frontend_fixtures`, then on `aeroviz_backend/tests/test_window_segment.py -k
  frontend_fixtures`. Stage A's and B's have the same kind of test.
- `tests/test_training_export.py::test_the_flown_block_of_every_stage_is_its_earlier_one_with_the_envelopes` reads
  stage C's fixture; it reads the sentences under `commanded[*].rounds`.
- While a campaign runs, use light tests: the changed Vitest files (`npx vitest run --maxWorkers 3 …`) and the changed
  pytest files (`-n 3`, `OMP_NUM_THREADS=1`, `nice -n 19`). Each commit is reviewed by an opus reviewer that did not
  write it, and the browser check goes to a one-shot sonnet agent.
- Since 2026-10-07, ts campaigns run from a detached run worktree, so merges into `dev-two-tier` go on during a run
  (root CLAUDE.md, outline D163).

**The test stack** (still running, from the worktree; it reads a scratch tree):
- vite 5186 (`http://192.168.1.103:5186`), backend 8796.
- Scripts and PID files are in `/tmp/claude-1000/fronter-stack/` (`vite.sh`, `backend.sh`, `vite.config.mts`). The
  scratch airports tree is `/tmp/claude-1000/fronter-stack/airports`: links to the live A and B sets, plus a v4 smoke
  set `windows_smoke_v4` at KRDU.
- That set was exported with `python run_ts.py post_training_export --campaign <C11's smoke campaign> --rounds start 0
  --split select --per-airport 2 --kinds real A B --airports KRDU --set-id windows_smoke_v4 --root
  /tmp/claude-1000/fronter-stack/airports --speed <the smoke speed readout of stage C> --smoke`. The two paths are in
  `windows_smoke_v4/sample.json`'s `source`.
- Stop it with `kill $(lsof -t -iTCP:5186 -sTCP:LISTEN) $(lsof -t -iTCP:8796 -sTCP:LISTEN)`. `/tmp` may be gone after a
  reboot; then build the tree again in the same way.

**The main checkout's backend** (8765) was not restarted after the merges. It still answers the window segment v1. Only
stage C uses it, and no stage C set is published. Restart it (`./start_aeroviz_fullstack.sh`) when a stage C or D set
is published.

**Left as S3 by the reviews** (listed once, no action taken):
- the slider: a cursor past the last row (a chart hover, the axis end) makes → and End step back to the last row;
  jsdom-only guards in `CursorSlider.tsx` (`box.width > 0`, `setPointerCapture?.`); the first-step tick is teal for
  every kind;
- the reader's refusal "not in the order they join" has no test (a one-aircraft fixture cannot make it);
- `data/trainingSetResults.ts` still parses the results route's sections the page no longer shows, so a malformed
  unread section would refuse the whole answer;
- `TRAINING_WINDOW_ROLE_COLOR` is exported but used only in its own module;
- the 3D layers' kind colour and the tabs' swatch have no unit test (the browser check saw them);
- `frontend.md`'s key code index still names `ResultSections.tsx`, which was removed (the designer's text);
- `aeroviz-4d/CLAUDE.md`'s lines on the details page (AV33) and the colours describe the view before F0. Updating them
  is for whoever keeps that file.
