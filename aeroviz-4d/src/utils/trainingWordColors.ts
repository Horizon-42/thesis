/**
 * trainingWordColors.ts
 * ---------------------
 * The Training views' one palette. The sentence bar, the read-back charts and the 3D scene must
 * agree about what "the heading colour" or "a heading band" looks like — two copies of a hex string is
 * how they stop agreeing.
 *
 * Only colours that CARRY MEANING live here; the panel's chrome is styled by class in
 * `index.css`. These are chosen per column or per envelope at render time.
 */

import type { TrainingColumn, TrainingFlownEnd } from "../data/trainingSample";

/** The sentence bar's surface — every colour here is validated against it; `index.css` draws `.training-sentence-bar` in it
 *  at 0.94 opacity (MIRROR: CSS cannot import it). The bar shades with it. */
export const TRAINING_SURFACE_COLOR = "#0f131e";

/** One colour per column, in `TRAINING_COLUMNS` order.
 *
 *  Speed is a purple (2026-09-24): its amber was the selection's yellow to the eye (OKLab ΔE 3.4).
 *  Checked with the dataviz palette validator on the bar's surface (#0f131e): ΔE 39 from the
 *  selection, 23 from the verdict red, 18 from the raw grey, ≥ 10 from every other colour here under
 *  simulated colour blindness. Altitude and angle are near each other (ΔE 4.3); their rows are labelled. */
export const TRAINING_COLUMN_COLOR: Record<TrainingColumn, string> = {
  runway: "#86efac",
  heading: "#7dd3fc",
  altitude: "#a5b4fc",
  angle: "#c4b5fd",
  speed: "#be76ff",
};

/** The SELECTED word (one column's, at the cursor), in ONE colour across every view: its band's
 *  stroke, its envelope's edge and the rows it is in force — the eye should be able to follow it
 *  from the bar to the charts to the scene. An envelope keeps its own hue as its fill. */
export const TRAINING_WORD_COLOR = "#facc15";

/** A CORRECTION WORD — one the closed-loop reading added to the labelled sentence. Orange: its band is also dashed and
 *  carries a mark (colour alone is not a signal), so it reads apart from the column hues and the selection's yellow. */
export const TRAINING_CORRECTION_COLOR = "#fb923c";

/** The observed track — the bright line every envelope of the labelled sentence is judged against. */
export const TRAINING_TRACE_COLOR = "#e2e8f0";
/** The observed track beside the flown one: dimmer, the reference the flown path is compared with. */
export const TRAINING_RAW_COLOR = "#94a3b8";

/** A heading word's BAND: its target ± the heading tolerance over the rows it is judged on — on the heading chart, and
 *  those rows of the ground trace in 3D. The blue of the heading family, deeper than the column's. */
export const TRAINING_HEADING_BAND_COLOR = "#38bdf8";
/** An altitude word's TUBE — the altitude column's own colour. */
export const TRAINING_TUBE_COLOR = TRAINING_COLUMN_COLOR.altitude;
/** A speed word's band — the speed column's own colour. */
export const TRAINING_SPEED_COLOR = TRAINING_COLUMN_COLOR.speed;

/** How opaque each envelope is in the 3D scene at rest, and a fill when it is the selected word's — one table, so the
 *  legend's swatches are the scene's. The heading bands are lines on the ground, at every ground line's opacity
 *  (`GROUND_LINE_ALPHA`, `scene/trainingEntities.ts`). */
export const TRAINING_ENVELOPE_ALPHA = {
  tube: 0.28, selected: 0.45,
} as const;

/** Every candidate runway (the runway word's choices) — and the one the flight lands on. */
export const TRAINING_CANDIDATE_COLOR = "#94a3b8";
export const TRAINING_DESIGNATED_COLOR = TRAINING_COLUMN_COLOR.runway;

/** The signal where it sits OUTSIDE the envelope of the word in force — the one thing on these
 *  views that is a disagreement rather than a drawing. */
export const TRAINING_OUTSIDE_COLOR = "#f87171";

/** THE FLOWN PATH: the closed-loop sentence flown by the executor (the replay's states) beside the observed track.
 *  Teal (2026-09-24): OKLab ΔE on the palette above is ≥ 11.9 from every colour here and 53 from the bar's surface. */
export const TRAINING_EXECUTOR_COLOR = "#14b8a6";

/** EACH KIND OF SENTENCE HAS ITS COLOUR (frontend §3 item 11, D159; the instruction-v3 view's validated palette, its
 *  §4.8, checked on the bar's surface #0f131e): the observed track near-white and the labelled sentence flown by the
 *  executor (the closed loop) teal, as before; the base's samples magenta; a post-trained round yellow-green; stage D's
 *  model raspberry. Every round of one model has the model's colour (the tab names the round). The colour draws the flown
 *  track in 3D and its ground trace, the read-back charts' flown line, the legend's swatch and a swatch on the bar's
 *  tab; a model's name and numbers are in the body text's colour beside the swatch (raspberry's contrast is 3.3:1:
 *  enough for lines and swatches, not for text). THE ONE MAP: no colour of a sentence kind is written elsewhere. */
export type TrainingSentenceKind = "observed" | "closedLoop" | "base" | "postTrained" | "multi";
export const TRAINING_SENTENCE_COLOR: Record<TrainingSentenceKind, string> = {
  observed: TRAINING_TRACE_COLOR,
  closedLoop: TRAINING_EXECUTOR_COLOR,
  base: "#d946ef",
  postTrained: "#a3e635",
  multi: "#b82e7a",
};

/** THE EXECUTOR FLOWN LIVE: the clicked word's segment, flown by the backend when it is clicked
 *  (`trainingAutopilot.ts`) — the flown path's teal would read as the exported one. Royal blue (2026-09-25): the dataviz
 *  validator on the bar's surface (#0f131e) puts it ≥ 21.8 OKLab ΔE from every colour above under normal vision and
 *  ≥ 9.3 under simulated colour blindness, contrast ≥ 3:1. */
export const TRAINING_AUTOPILOT_COLOR = "#2563eb";
/** A FAILURE: where a flight failed, drawn as the answer to the question it was flown for — louder than the per-row
 *  `TRAINING_OUTSIDE_COLOR`: the flight list's outcome tag, the DA check that failed, the live segment flown on to an
 *  outcome other than a landing. ΔE 41.6 from the autopilot blue (29.0 under CVD); contrast 5.0:1. */
export const TRAINING_FAILURE_COLOR = "#ff2d2d";

/** The DA point — the decision-altitude check of the crossing: green when it passed, the failure red when it did not. */
export const TRAINING_DECISION_PASS_COLOR = "#4ade80";
export const TRAINING_DECISION_FAIL_COLOR = TRAINING_FAILURE_COLOR;

/** How a flown flight ended, in one colour where it is said (the flight list's tag, the bar's chip): the replay's teal
 *  when it landed, the failure red for any other outcome. */
export function trainingOutcomeColour(outcome: TrainingFlownEnd): string {
  return outcome === "landed" ? TRAINING_EXECUTOR_COLOR : TRAINING_FAILURE_COLOR;
}
