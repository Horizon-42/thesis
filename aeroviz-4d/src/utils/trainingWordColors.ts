/**
 * trainingWordColors.ts
 * ---------------------
 * The Training views' one palette. The sentence bar draws six rows and the
 * read-back window three charts, and they must agree about what "the altitude
 * colour" is — two copies of `#a5b4fc` is how they stop agreeing.
 *
 * Only the colours that CARRY MEANING live here. The panel's chrome is styled by
 * class in `index.css`; these are inline because they are chosen per word kind at
 * render time, which a stylesheet cannot do.
 */

import type { TrainingKind } from "../data/trainingSample";

/** One colour per word kind, in `TRAINING_KINDS` order. */
export const TRAINING_KIND_COLOR: Record<TrainingKind, string> = {
  heading: "#7dd3fc",
  altitude: "#a5b4fc",
  speed: "#fbbf24",
  runway: "#86efac",
  duration: "#94a3b8",
  terminal: "#f472b6",
};

/**
 * The word in force, drawn in ONE colour across every chart: it is one sentence,
 * and the eye should be able to follow it down the page.
 *
 * YELLOW, not the orange the hand-check pages use. Those pages have no flown
 * track on them; this window draws one, the design fixes it as orange (§5.5,
 * "真实白、规则橙、模型紫"), and two orange lines on one chart are one line as far
 * as a reader is concerned.
 */
export const TRAINING_WORD_COLOR = "#facc15";

/** The measured signal, against which the words are read. */
export const TRAINING_TRACE_COLOR = "#cbd5e1";

/** The sentence flown by rule. Orange, and never the same style as a model's
 *  output: it is a baseline and a diagnostic (design V9 / §5.5), and drawing it
 *  like a prediction is how it would come to be read as one. */
export const TRAINING_FLOWN_COLOR = "#fb923c";
