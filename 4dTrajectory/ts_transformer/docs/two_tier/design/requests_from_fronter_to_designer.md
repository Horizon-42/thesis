# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides. Paths are relative
to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-07. F0, F1 and F2 are built, reviewed and browser-checked on `dev-frontend` (log
`readouts/2026-10-06_fronter_implementation_log.md`). F3 waits for stage D's MC0 on `dev-two-tier`, F4 for its MC4.

## F1 · the cursor slider (§6.1, D155)

1. **The slider's rows.** §6.1 says "from the sentence's row 0 to its last row". Read: the track is the sentence bar's
   axis, from 0 to where the sentence or its judged track ends (`readingAxisEndS`, one definition for the bar and the
   slider). The slider stops at the observed track's 2 s rows before the sentence opens (a closed-loop sentence opens
   at the first predicted step; a window's row 0 is before it, D129), then at the sentence's own rows: Δ apart on a
   closed-loop sentence, 2 s apart on a labelled one. ← and → move one of these stops. `aria-valuenow` is the stop's
   place in that list; `aria-valuetext` is "t = 128 s, row 32 of 210", with both numbers counted from 0.
2. **Stage A's labelled tab has no first-step mark.** In stage A the first predicted step differs by Δ (29 of the 40
   flights of `closed_loop_v12_20261005` at KRDU). So a labelled sentence, read against no Δ, gets no tick. On a Δ tab
   the tick is that Δ's first predicted step. In B and C the labelled tab keeps the tick of the item's own first
   predicted step (B: the prior's Δ; C: the window's).
3. **Labelled and the round it is read on.** In a stage C window, the Labelled tab reads the head of the last round's
   derived flight. Moving between it and that round therefore keeps the flight, and so the cursor. The cursor is then
   clamped to the new sentence's axis, as for any change of round (§6.1 "In a window").

## F2 · windows of several commanded aircraft (§5.7, D156)

4. **"A new request name on both sides."** Read: the request gets a required field `aircraft` (the commanded aircraft's
   dataset id), and the answer's schema becomes `aeroviz-autopilot-window-segment-v2`, which echoes it. The route stays
   `POST /autopilot/window-segment`. An old page that sends no `aircraft` is refused by name (400). An old backend's
   v1 answer is refused by name by the new page. An aircraft that the window does not command is refused by name (404).
5. **The window sample v4's fields.** A window: `kind`, `c` (null for a window that is not compressed: every stage C
   window), `row0S`, `commanded` (a list in the order the aircraft join), `moved`, `traffic`, `rounds`. A commanded
   aircraft: `datasetId`, `joinS` (from the window's row 0; the first is 0), `shiftS` (its move in a compressed window,
   else null), `firstStepS`, `startMove`, `movedStart`, and `rounds`. Each of its rounds is the flown-sentence block
   plus `reward`, `silentFromRow` (null in stage C) and `speedMaskRows`. A window round: `losses` (each with `step`,
   `timeS`, `aircraft` [two keys], `answering`, `kind`, `relation`, the distances, `wakeKnown`, `readsFault`, `costsW`)
   and `faultySteps`. Stage C writes one aircraft; its loss has its commanded aircraft answering and costs W.
6. **The index's set kind.** "One index format for both; the stage is the file's". Read: one set kind for stages C and
   D, `training-windows` (it was `post-training-windows`). Index schema `aeroviz-training-window-index-v3`; stage C's
   file is `index_post_v3.json`.
7. **The checks of a loss and the end of the aircraft that answers it.** In a round, an aircraft whose flight ended
   `lost_separation` answers for one of its losses. An aircraft that answers ended `lost_separation` or is silent from
   a row on (D144). The reader refuses anything else by name. The live segment stops at the loss when the aircraft's own
   outcome is `lost_separation`.
8. **Two clocks in the reader.** The window's times (traffic, losses) are seconds from its row 0. Each aircraft has its
   own flight clock, and `clockS` is its flight time at the window's row 0. Every consumer converts a window time to the
   aircraft on screen by adding its `clockS`. For stage D, the cursor's change of the selected aircraft (§6.1) can use
   the same conversion.

## F0 · the user's three corrections (D159)

9. **The rows of the models' statistics.** Stage A has one row for each Δ of the set: every Δ tab is a closed-loop
   sentence, and the design's "at the set's Δ" has no single Δ in stage A. Stage B has the closed loop at the prior's Δ
   and "the prior's samples" (magenta, the base's colour; a fold's prior is not the base, but it is the same model kind).
   Stage C has the start ("start (base)", magenta) and each round (yellow-green). The labelled sentence is not flown, so
   it has no row.
10. **The formal readout of each row.**
    - A: the executor's replays of the set's flights' splits at the row's Δ, at the set's airport, summed over the
      dynamics groups (own and stand-in).
    - B: the free generation of `source.readout` at the set's airport, summed over every side of the prior's selection
      (inside, outside for a faulty track, outside for the outcome) and every stratum. The set's flights are drawn from all
      the sides (`prior_training_export.chosen_flights`), so the two columns count the same population. B5's report
      (`readouts/2026-10-06_b5_campaign.zh.md` §3.2) quotes the "inside" side as its headline. On the folds' select days
      the two differ by about 1 point of landed: KSTL 71.4 % inside vs 70.5 % over every side, KMSY 91.7 vs 90.5, KRDU
      83.4 vs 84.3. If the page should show the headline instead, it is a one-line change.
    - B's closed loop row and C's start row: "—", because the route answers no count of them.
    - C's rounds: the round's selection readout at the set's airport.
11. **The columns.**
    - "go-arounds" is the go-arounds said, summed, over the sentences: the readouts count them so (`go_arounds` is a sum).
      With at most 2 go-arounds a flight, the share can exceed 100 %; on real data it is a few in a thousand.
    - "timed out" is the judge's outcome `timeout`.
    - Stage A has no go-around count on either side ("—").
    - Why a readout section is missing (its file elsewhere or missing) is written above the table.
12. **The colours.**
    - The start of a campaign is the base: magenta.
    - A tab carries its kind's square swatch beside the outcome's round dot.
    - The aircraft model at the cursor on a flown sentence is tinted in its kind's colour.
    - Not covered by §3 item 11, so left as they were: the outcome's colour (`trainingOutcomeColour`: landed teal, other
      red) in the tab dots and in the "other sentences" and "other rounds" overlays, and the slider's first-step tick (teal).
      A landed base sentence therefore has a magenta swatch beside a teal dot.
