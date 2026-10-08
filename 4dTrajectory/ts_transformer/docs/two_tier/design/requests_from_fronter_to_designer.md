# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. Items 1–15 were decided (frontend D177) and are removed. F5, D177's changes, the speed unbinding
and the `model_speed` sample are merged into `dev-two-tier` (`b8646c12`, `2972cf9f`); the items below are readings
(built so; the user can change any).

## F5 (2026-10-08)

1. **`--windows` takes the list file.** The shared export reads `--windows <…>/list.json` (the file `window_list`
   writes), and a set records the file's path and sha256 — no mirror of the runner's file name.
2. **A listed set's airports are the list's.** `--airports` is refused with `--windows`, as `--per-airport`, `--seed`
   and `--kinds` are; an airport without a listed window gets no set.
3. **The draw's defaults moved from the arguments to the export.** `--per-airport`, `--seed` and `--kinds` have no
   argument default (so that giving one with `--windows` can be refused); a drawn set without them uses 10, 1337 and
   `real` (`DRAWN_PER_AIRPORT`, `DRAWN_SEED`), as before.
4. **A listed cohort's fields.** `form` `listed`, `split`, `list` (repository-relative), `sha256`, `chose` (the list's
   sentence), `listCount` (the list's windows), `windows` (this airport's), `selectSeed`; a drawn cohort is v4's fields
   with `form` `drawn`. The index entry's title says "listed windows". The details page's first section gives one line:
   drawn (count, windows an airport, seed, kinds, the draw's sentence) or listed (count of the list's, the list, its
   sentence, the select seed).
5. **The landed green is green-600, `#16a34a`.** OKLab ΔE 23.5 from the post-trained yellow-green, 12.2 from the closed
   loop's teal, 47.5 from the bar's surface; it is also the DA check's "passed" green (one constant,
   `TRAINING_DECISION_PASS_COLOR`).
6. **Two small copies wait for one helper each.** The positions stage D's export hands `multi.census.judged_steps`
   are built as `multi_train.window_losses_of` builds them (a MIRROR, pinned by a test): proposed to stage D, one
   helper both call. A window's identity and the selection's fields as a list names them are spelled in each stage's
   `StageExport` beside `post.window_lists.IDENTITY_FIELDS` / `SELECTION_FIELDS` (pinned by a test): proposed to stage
   C, `identity_of(stage, window)` in `post/window_lists.py`.

## The speed readout (2026-10-08, the user's decisions)

7. **The speed readout is unbound from the Training sets (the user: "解绑，B 和 C 都改").** No Training set names a
   speed readout any more: the exports of stages B, C and D lost `--speed` and `source.speed`, the results route
   (`aeroviz_backend/training_results.py`) lost its `speed` sections, the TS results reader its `TrainingSpeed`
   (D159 had already taken the section off the details page, so nothing showed it). `model_speed` stays a runner of its
   own that writes `speed.json`; a page that shows speed reads that file. To the designer: D136 and frontend §3 item 10
   say a set names its readout — to rewrite. **Reading (fronter): the format names stay** (sample v5 / index_post_v4,
   index_prior_v3): no reader read `source.speed`, so the published sets of stages B and C keep a field nobody reads
   and new sets lack it; renaming would mean re-exporting every published B set for a field no one reads.
8. **`model_speed` times a sample (the user: "sample 一部分就可以").** Stage C now speaks a seeded draw of the
   selection windows, `--per-airport` an airport (default 20, as stage B's), each at its place with its readout's
   numbers; a batched setting draws at least its batch (an equal share an airport), so batch 400 holds 400 windows
   where each airport holds its share; `speed.json` records the draw. `model_speed` is frontend §3 item 10 (D136),
   fronter's runner — item 7 of the last version said stage B's, which was wrong.
