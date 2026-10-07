# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-07. The twelve readings of F0–F2 were accepted by the user (frontend D160); (11) and (12) are built on
`dev-frontend` (`da1c6bba`). One new item below. F3 waits for stage D's MC0 on `dev-two-tier`, F4 for its MC4.

## After D160

1. **The landed green beside the post-trained yellow-green.** With D160 (12) a landed outcome is the pass green
   `#4ade80`, and a post-trained round is yellow-green `#a3e635` (§3 item 11). Their OKLab ΔE on the bar's surface is
   9.5, below the 11.9 floor that `utils/trainingWordColors.ts` cites for the palette; teal against yellow-green was 22.2
   (the reviewer's computation). They are drawn together in two places:
   - in a stage C window, a landed "other round" (thin, half-transparent) beside the round on screen;
   - on a round's tab, the yellow-green kind swatch (square) beside the green outcome dot (round).

   Nothing is changed; the designer may judge it in the browser.
