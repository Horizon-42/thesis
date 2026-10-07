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
