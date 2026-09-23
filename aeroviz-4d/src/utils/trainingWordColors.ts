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

/** One colour per column, in `TRAINING_COLUMNS` order. */
export const TRAINING_COLUMN_COLOR: Record<TrainingColumn, string> = {
  runway: "#86efac",
  approach: "#f472b6",
  heading: "#7dd3fc",
  altitude: "#a5b4fc",
  angle: "#c4b5fd",
  speed: "#fbbf24",
};

/** The word in force at the cursor, in ONE colour across every view: it is one sentence, and
 *  the eye should be able to follow it from the bar to the charts to the scene. */
export const TRAINING_WORD_COLOR = "#facc15";

/** The smoothed signal the labeller read — the bright line every envelope is judged against. */
export const TRAINING_TRACE_COLOR = "#e2e8f0";
/** The raw rows behind it: what the aircraft did, before the reading's smoothing. */
export const TRAINING_RAW_COLOR = "#94a3b8";

/** A heading word's TURN REGION: the arcs of every allowed bank from its issue point. */
export const TRAINING_TURN_COLOR = "#fb923c";
/** A heading word's HOLD FUNNEL: the nominal line along θ and its widening half width. */
export const TRAINING_FUNNEL_COLOR = "#38bdf8";
/** The CAPTURE CORRIDOR from the capture to the threshold (and the course band after it). */
export const TRAINING_CORRIDOR_COLOR = "#4ade80";
/** An altitude word's TUBE — the altitude column's own colour. */
export const TRAINING_TUBE_COLOR = TRAINING_COLUMN_COLOR.altitude;
/** A speed word's transition and band — the speed column's own colour. */
export const TRAINING_SPEED_COLOR = TRAINING_COLUMN_COLOR.speed;

/** Every candidate runway (the pointer's choices) — and the one pointed at. */
export const TRAINING_CANDIDATE_COLOR = "#94a3b8";
export const TRAINING_DESIGNATED_COLOR = TRAINING_COLUMN_COLOR.runway;

/** The signal where it sits OUTSIDE the envelope of the word in force — the one thing on these
 *  views that is a disagreement rather than a drawing. */
export const TRAINING_OUTSIDE_COLOR = "#f87171";
