# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. Items 1–15 were decided (frontend D177) and are removed. F5 and D177's changes are built on
`dev-frontend`; the items below are F5's readings (built so; the user can change any).

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
