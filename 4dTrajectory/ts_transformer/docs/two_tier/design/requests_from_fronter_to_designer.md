# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. The start's name after D162 (frontend §4.3) is built and merged (`f10e9ee5`); item 2 is its reading.
The start's formal readout (the old item 3) was decided by the user and is built (`a9d296d5`). Next: stage C's first
formal set (P55 round 5), then F3.

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
