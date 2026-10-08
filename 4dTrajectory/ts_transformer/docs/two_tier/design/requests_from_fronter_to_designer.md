# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. The start's name and readout (§4.3) are built and merged; stage C's first formal set is published
(P55 round 5); F3 is merged; F4's export and smoke set are built on `dev-frontend` (`7b3e7ae5`). Items 3–9 are F3's
readings, 10–15 F4's. Next: stage D's formal sets after MC6, their intent first.

## After D160

1. **The landed green beside the post-trained yellow-green.** With D160 (12) a landed outcome is the pass green
   `#4ade80`, and a post-trained round is yellow-green `#a3e635` (§3 item 11). Their OKLab ΔE on the bar's surface is
   9.5, below the 11.9 floor that `utils/trainingWordColors.ts` cites for the palette; teal against yellow-green was 22.2
   (the reviewer's computation). They are drawn together in two places:
   - in a stage C window, a landed "other round" (thin, half-transparent) beside the round on screen;
   - on a round's tab, the yellow-green kind swatch (square) beside the green outcome dot (round).

   Nothing is changed; the designer may judge it in the browser.

## After §4.3's start (2026-10-08)

2. **A start from another campaign's round is a post-trained sentence (built so).** §3 item 11 gives the base magenta
   and a post-trained round yellow-green; §4.3 says nothing of the colour of a D162 start. The model at such a start is
   a post-trained one (e.g. C10's round 8), so its tab swatch, 3D track, read-back line and statistics row are
   yellow-green unless the campaign starts from the base. Stage D's start (a stage C round, D164) is yellow-green too;
   stage D's rounds are stage D's colour (`multi`).

## F3 (2026-10-08, built so; the user can change any)

3. **The window segment names the stage.** §5.5 names the aircraft only. A stage D set is listed in another index
   (`index_multi_v1.json`), and set ids are not unique across the two, so the request names `stage` (C or D) and the
   answer is v3 (`aeroviz-autopilot-window-segment-v3`); a request without it is refused by name.
4. **Stage D's Training files live beside stage C's.** `post/training_files.py` holds `MULTI_FILES` (stage D's index
   file, the one format) rather than a module of stage D's own (the old docstring said "its own module"; F4's export
   reads it from there).
5. **Loss counts and their units.** "This round" (§5.1) counts the round's losses (each pair at one step once):
   "N losses of separation between commanded aircraft, M with recorded". The details page (§5.6, "counted once a pair")
   counts the windows with a loss of each kind, once a kind a window — the unit of the readout's `loss_windows` — in a
   note under the table for each round, beside the readout's four pairs (`multi.separation.PAIRS`). The set's two kinds
   split by whether both aircraft are commanded; the readout counts a commanded aircraft that has landed as recorded, so
   the two can differ on such a loss.
6. **Stage D's start has no formal readout.** The start of a stage D campaign is a stage C round (D164), read on stage
   C's windows: the results route answers why, and the start row shows "—" with that note.
7. **A commanded aircraft in 3D from the row it joins.** Before its first predicted step it is drawn on its observed
   rows (window B's: its moved start, which ends one 2 s row before that step); after it, on its flown track. A loss is
   drawn between any two aircraft at its time, so a loss at an aircraft's join (it answers for nothing then, D145) is
   drawn too. Stage C now also draws a loss that falls in its commanded aircraft's observed rows (the old view skipped
   it).
8. **The bar's loss chip names the loss the aircraft answers for.** A stage D aircraft may be in a loss it does not
   answer; the chip then says "no loss it answers for", and the slider still marks every loss it is in.
9. **Stage C's export flies a window of several commanded aircraft with stage C's loop and rule** (for §8 F3's
   fixture): each aircraft's numbers are the anchor's stage C readout numbers and, for a later member, a mirror of stage
   D's (`multi_train.readout_numbers`). Stage D's export (F4) uses stage D's loop options and numbers of its own.

## F4 (2026-10-08, built so; the user can change any)

10. **Stage D's sets name no speed readout.** `model_speed` times stages B and C (D136); a stage D set's `source.speed`
    is null and the details page says so. A speed readout of stage D would need `model_speed` to speak stage D's loop.
11. **A loss written once.** The loop records an answered loss at the step it answers it; the export adds every other
    loss holding a commanded aircraft (D145: answered by none, dashed), one entry per run of a pair's loss over
    successive steps, at its first step; a run holding an answered step is that answer's and is not written again.
12. **The census is mirrored.** The unanswered losses come from a copy of the readout census's step loop
    (`multi_training_export.census_losses`, declared a MIRROR and pinned against `multi_train.window_losses_of`).
    Proposed to stage D: one generator in `multi/census.py` that both use (fronter does not change stage D's code).
13. **A silent aircraft that flew on is written to its own end.** When the executor finishes an aircraft after it
    went silent (D144), its block has the judge's outcome, end and crossing (e.g. landed); its reward stays 0 and its
    loss stays answered. The details page's set column counts it as lost separation, as the readout does; the strip,
    "This round" and the list count its written outcome (it did land) — the two counts can differ by such aircraft.
14. **Stage D's sets: readout windows only.** A set draws from the campaign's readout windows (real, every span, D166
    (33)), so a window is never compressed (`c` null); each aircraft flies with `multi_train.readout_numbers` of the
    set's seed and its place in the set (not the readout's own places), as stage C's sets do.
15. **The private record read.** The export reads `WindowLoop._reading` (the steps at which a recorded aircraft read a
    faulty point) for the unanswered losses' `readsFault` (D114). Proposed to stage C: a public reader of it.
