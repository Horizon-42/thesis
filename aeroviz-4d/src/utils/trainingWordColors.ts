/**
 * trainingWordColors.ts
 * ---------------------
 * The Training views' one palette. The sentence bar, the read-back charts and the 3D scene must
 * agree about what "the heading colour" or "a funnel" looks like — two copies of a hex string is
 * how they stop agreeing.
 *
 * Only colours that CARRY MEANING live here; the panel's chrome is styled by class in
 * `index.css`. These are chosen per column or per envelope at render time.
 */

import type { TrainingColumn } from "../data/trainingSample";

/** One colour per column, in `TRAINING_COLUMNS` order.
 *
 *  Speed is a purple (2026-09-24): its amber was the selection's yellow to the eye (OKLab ΔE 3.4).
 *  Checked with the dataviz palette validator on the bar's surface (#0f131e): ΔE 39 from the
 *  selection, 23 from the verdict red, 18 from the raw grey, ≥ 10 from every other colour here under
 *  simulated colour blindness — a teal read 4.6 from the approach pink and 5.3 from the raw grey for
 *  a deuteranope. Altitude and angle are near each other (ΔE 4.3); their rows are labelled. */
export const TRAINING_COLUMN_COLOR: Record<TrainingColumn, string> = {
  runway: "#86efac",
  approach: "#f472b6",
  heading: "#7dd3fc",
  altitude: "#a5b4fc",
  angle: "#c4b5fd",
  speed: "#be76ff",
};

/** The SELECTED word (one column's, at the cursor), in ONE colour across every view: its band's
 *  stroke, its envelope's edge and the rows it is in force — the eye should be able to follow it
 *  from the bar to the charts to the scene. An envelope keeps its own hue as its fill. */
export const TRAINING_WORD_COLOR = "#facc15";

/** The smoothed signal the labeller read — the bright line every envelope is judged against. */
export const TRAINING_TRACE_COLOR = "#e2e8f0";
/** The raw rows behind it: what the aircraft did, before the reading's smoothing. */
export const TRAINING_RAW_COLOR = "#94a3b8";

/** A heading word's BAND: its target ± the heading tolerance over the rows it is judged on — on the heading chart, and
 *  those rows of the ground trace in 3D (instruction-v3). The blue of the heading family, deeper than the column's. */
export const TRAINING_HEADING_BAND_COLOR = "#38bdf8";
/** The CAPTURE TURN: its rows, from the clearance onto the course. */
export const TRAINING_CAPTURE_TURN_COLOR = "#fb923c";
/** The CAPTURE CORRIDOR from the capture to the threshold (and the course band after it). */
export const TRAINING_CORRIDOR_COLOR = "#4ade80";
/** An altitude word's TUBE — the altitude column's own colour. */
export const TRAINING_TUBE_COLOR = TRAINING_COLUMN_COLOR.altitude;
/** A speed word's transition and band — the speed column's own colour. */
export const TRAINING_SPEED_COLOR = TRAINING_COLUMN_COLOR.speed;

/** How opaque each envelope is in the 3D scene at rest, and a fill when it is the selected word's — one table, so the
 *  legend's swatches are the scene's. The heading bands and the capture turn are lines on the ground. */
export const TRAINING_ENVELOPE_ALPHA = {
  headingBand: 0.85, captureTurn: 0.85, corridor: 0.3, tube: 0.28, selected: 0.45,
} as const;

/** Every candidate runway (the pointer's choices) — and the one pointed at. */
export const TRAINING_CANDIDATE_COLOR = "#94a3b8";
export const TRAINING_DESIGNATED_COLOR = TRAINING_COLUMN_COLOR.runway;

/** The signal where it sits OUTSIDE the envelope of the word in force — the one thing on these
 *  views that is a disagreement rather than a drawing. */
export const TRAINING_OUTSIDE_COLOR = "#f87171";

/** THE EXECUTOR's flown track and its verdicts "inside" (the replay of the truth sentence, `trainingOverlays.ts`).
 *  Teal (2026-09-24): OKLab ΔE on the palette above is ≥ 11.9 from every colour here (nearest: the raw grey 11.9,
 *  the funnel blue 12.3, the corridor green 13.8) and 53 from the bar's surface (#0f131e); every candidate hue
 *  between them sat under 10 from one of the columns. */
export const TRAINING_EXECUTOR_COLOR = "#14b8a6";

/** THE EXECUTOR FLOWN LIVE: the selected word's segment, flown by the backend when it is selected
 *  (`trainingAutopilot.ts`) — the replay's teal would read as the precomputed replay. Royal blue (2026-09-25): the dataviz
 *  validator on the bar's surface (#0f131e) puts it ≥ 21.8 OKLab ΔE from every colour above under normal vision (nearest:
 *  the speed purple) and ≥ 9.3 under simulated colour blindness, contrast ≥ 3:1; the violet #7c3aed passed too (17.3 /
 *  12.5) but sits in the speed column's hue. */
export const TRAINING_AUTOPILOT_COLOR = "#2563eb";
