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
 */
export const TRAINING_WORD_COLOR = "#facc15";

/** The measured signal, against which the words are read. */
export const TRAINING_TRACE_COLOR = "#cbd5e1";

/** The ENVELOPE the sentence makes — the chain of boxes, and the wedge wall the
 *  3D scene draws. Orange, the colour the design fixes for "what the words say"
 *  as opposed to what the aircraft did (§5.5, 真实白、规则橙、模型紫). Under this
 *  rule the words say a region rather than a line, so the colour moved from a
 *  track onto the region — there is no rule-flown track any more (the replay gate
 *  is not built). */
export const TRAINING_FLOWN_COLOR = "#fb923c";

/**
 * THE BOX ITSELF — the interval a word names, drawn as a filled band with its two
 * edges. Under `box-v2-wedge` the box IS the word, so this is not a decoration
 * around a line: it is the thing the chart is about, and the measured signal is
 * read against it.
 *
 * Cyan rather than the retired corridor's orange, because there is no flown line
 * for it to be the slack of any more.
 *
 * A box is drawn only for the three kinds that ARE boxes. The runway word names
 * the frame, the duration word the hold and the terminal word the ending; a band
 * on any of them would be an invented number.
 */
export const TRAINING_BAND_FILL = "rgba(56, 189, 248, 0.14)";
export const TRAINING_BAND_EDGE = "rgba(56, 189, 248, 0.55)";

/**
 * WHAT THE MODEL SAID. Purple, because the design fixed the three lines as white
 * (the aircraft), orange (the words flown by rule) and purple (the model) — and
 * because it must never be drawn in the rule-follower's colour: one is a
 * baseline, the other is the thing being judged.
 */
export const TRAINING_MODEL_COLOR = "#c084fc";

/** The observed signal where it sits OUTSIDE the band of the word in force —
 *  the one thing on these charts that is a disagreement rather than a drawing. */
export const TRAINING_OUTSIDE_COLOR = "#f87171";
