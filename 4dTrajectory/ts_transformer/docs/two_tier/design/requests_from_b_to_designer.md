# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1, §4).

State of 2026-10-06. Stage B is closed (B7). The Training view's one layout (outline §6.2, D133) is built on
`dev-training-layout` (`38297528`, `e0aceffd`, `aa221e86`; reviewed, the full suite passed, checked in the browser; not
merged). The user found its details page wrong (§3 item 1): it must be redesigned before the branch is merged.

## 1 Decided by the user, 2026-10-06

1. **The details page shows the experiment's statistics**, as the old page did (`TrainingResults.tsx` before `0c8e4c13`:
   the executor's replay gate, the prior's val readout per column against the baselines, each model's landed beside its
   formal val free generation). What D133's table puts there now is not that: the first section is the intent plus a
   list of paths, seeds and hashes; "What this view shows" is usage notes; B's "Sentences" counts only the set's 20
   sentences on screen; B's "Row inspector" repeats the sentence bar. **Whose error:** the sections in outline §6.2 item 3
   were Claude's design on stage B's request (D133: "the parts: Claude"), which read the old page as a place for long
   content instead of a page of results; the user decided only the layout, the intent's source and the order.
2. **The statistics come from the artefacts the set already names**, read by the backend through a read-only route
   (the user's permission for D133 covers the backend's routes); no export, set or index format changes, nothing is
   exported again.
3. **Removed from the details page:** the Row inspector (the left panel's one-line "At the cursor" and its probability
   strip stay); "What this view shows" (the bar's ⓘ notes explain the tabs and chips); the provenance facts (paths,
   seeds, hashes) cut to one line, the path of the readout the set came from.

## 2 For stage A

None.

## 3 For the design text

1. **Outline §6.2 item 3, rewritten: the details page is the experiment's results.** Claude's proposal (for the
   designer to settle with the user; the artefacts read on 2026-10-06):

   | Stage | Sections after "The set and the experiment" | Read from (the set's own `source` / `model`) |
   |---|---|---|
   | A | **Labelling** (labelled and refused per split, the refusal reasons) · **Closed loop** (the executor's replay at each Δ for the set's split: landed, the outcomes) · **Flown flights** (the set's own flights at each Δ) · Vocabulary (kept) | `instruction_language/<run>/readout.json`; `executor/<run>/replay-closed-<split>-<Δ>s/replay.json` |
   | B | **Free generation** (the formal readout: outcomes by airport and kind of route, inside / outside_fault / outside_outcome, words per sentence against the labelled; beside it the set's own sentences counted) · **Validation** (teacher-forced loss per step and per column; the masks on the labelled words) · **Training** (loss per epoch, train and select; the best epoch) · for a fold set **The choice** (each arm's score and folds, the seed scale, the chosen configuration and variant) | `source.readout/readout.json`; `<prior>/../validation/readout.json`; `model.prior/history.json`; `prior_base_<date>/choice_{configuration,variant}.json` |
   | C | **Rounds** (the formal campaign's rounds: the reward, landed, the loss of separation, the go-arounds) · **The checks** (the labeller's and the executor's conformance at the campaign's start) · The window (kept: the window's kind, moves, other aircraft) | `model.campaign/campaign.json`; `round_<n>/round.json` |

   "The set and the experiment" keeps the campaign's title, intent, design and the set's own line, and one line of
   provenance (the path of the readout the set came from). "At the cursor" stays a line in B's left panel, not a link
   (it follows the cursor; it has no section).
2. **The route** (Claude's proposal): `GET /training/results?airport=<ICAO>&set=<id>` — the backend finds the set in the
   airport's Training index and reads only the files its `source`/`model` name, each under `4dTrajectory/outputs/`; a
   set whose artefacts lie elsewhere (C's smoke set under a scratch directory) or are missing is answered by name, and
   the page says so; it reads at each request and writes nothing.
3. **Readings where §6.2 says nothing, as built** (proposals): a tab chosen stays chosen on the next item of the same
   stage when it has that tab, else the stage's first choice (B the first sample, C the last round, A the set's first
   Δ); the intent's answers 404 (none), 409 (several), 400 (no run asked), a found intent kept for the page's life;
   with no session on screen the details page opens with its first section saying why; `ExperimentIntentBlock`
   extracted unchanged from `ExperimentDetails.tsx` (outside item 7's list of files).
4. **The short tab labels** (the user, 2026-10-06): B's samples by their number (0, 1 …), C's rounds r1, r2 … (Start
   (base) unchanged); §6.2 item 2's table says "Sample 0" and "Round 1".

## 4 The plan

- The designer rewrites outline §6.2 item 3 (with a new D number) and an order to stage B; stage B builds it on
  `dev-training-layout`, then review, tests, the browser check; the user merges.
