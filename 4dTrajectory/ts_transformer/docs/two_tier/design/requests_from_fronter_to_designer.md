# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. The start's name after D162 (frontend §4.3) is built on `dev-frontend`; items 2 and 3 are its
readings. Next: stage C's first formal set (P55 round 5), then F3.

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
   a post-trained one (e.g. C10's round 8), so `roundKind("start", start)` is `postTrained` unless the campaign starts
   from the base: its tab swatch, 3D track, read-back line and statistics row are yellow-green, like the rounds after
   it. The tab's name tells the start from the rounds (`Start (post_train_20261006 r8)`).

3. **The start's row of the statistics has no readout (proposal, not built).** For a campaign from the base the start
   has no selection readout, so its row shows "—". A D162 start has one: the source campaign's round's own
   `round.json` `selection_readout` (on the same select windows when the readout seed and windows match, as for every
   campaign of 2026-10-07). The results route (`aeroviz_backend/training_results.py` `rounds`) could read it from
   `campaign.json` `inputs.settings.start` and answer it as the start's row, so the page shows e.g. 85.7 % beside
   87.7 %. Small (the route and `trainingStatistics.ts`); needs the user's word, since the source's readout may have
   been read on other windows (then: not shown, and why).
