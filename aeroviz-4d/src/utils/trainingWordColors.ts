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

/** The word in force, drawn in ONE colour across every chart: it is one sentence,
 *  and the eye should be able to follow it down the page. */
export const TRAINING_WORD_COLOR = "#fb923c";

/** The measured signal, against which the words are read. */
export const TRAINING_TRACE_COLOR = "#cbd5e1";
