/**
 * observedVerdictColors.ts
 * ------------------------
 * How an observed ADS-B track is coloured by its terminal-event verdict.
 *
 * THREE STATES, NOT TWO — and the third is the common one.
 *
 * A pass/fail pair would be a lie when a required operational bound is unavailable or
 * a fitted observed threshold event overlaps a bound once its uncertainty is included.
 * LPV has a resolved vertical bound, but an observed fit can still be indeterminate when
 * its confidence interval overlaps that bound; painting that state red would report
 * uncertainty as a flying error.
 *
 * An unavailable observed threshold estimate is folded into the same neutral colour.
 * It has no defensible crossing to judge and is not itself a flying failure.
 *
 *   green   passed both gates, and the interval clears both boundaries
 *   red     missed a gate, and the interval clears the boundary it missed
 *   grey    indeterminate — a bound/event is unavailable or an interval overlaps it
 *
 * The backend owns the three-way verdict (`evaluation/metrics.py`); this module only
 * maps it to colours, so the UI cannot invent a stronger conclusion.
 */

import type { ObservedVerdict } from "../data/observedTracks";

/** Path colours. Deliberately desaturated for grey so it recedes behind the verdicts. */
export const OBSERVED_VERDICT_COLORS: Record<ObservedVerdict, string> = {
  pass: "rgb(60, 200, 90)",
  fail: "rgb(230, 70, 70)",
  undecided: "rgb(150, 150, 155)",
};

/**
 * Fitted threshold extensions remain visibly distinct from measured paths by retaining
 * 35% of their generated ice-blue hue and mixing in 65% of the terminal verdict hue.
 * These are applied only at render time to the entity's independent `polylineVolume`;
 * generated CZML and its canonical ice-blue material remain unchanged.
 */
export const OBSERVED_FITTED_TAIL_COLORS: Record<ObservedVerdict, string> = {
  pass: "rgba(76, 202, 148, 0.58)",
  fail: "rgba(186, 117, 135, 0.58)",
  undecided: "rgba(134, 169, 190, 0.58)",
};

/** Higher-contrast versions of the same mix for the fitted volume's outline. */
export const OBSERVED_FITTED_TAIL_OUTLINE_COLORS: Record<ObservedVerdict, string> = {
  pass: "rgba(100, 211, 148, 0.9)",
  fail: "rgba(211, 126, 135, 0.9)",
  undecided: "rgba(159, 178, 190, 0.9)",
};

/** Short human-readable label, used by the legend and the tooltip. */
export const OBSERVED_VERDICT_LABELS: Record<ObservedVerdict, string> = {
  pass: "Terminal verdict: pass",
  fail: "Terminal verdict: fail",
  undecided: "Terminal verdict: indeterminate",
};

/**
 * One-line explanation per state. The legend shows these verbatim: a reader who does
 * not know the measurement limits would otherwise read grey as "no data".
 */
export const OBSERVED_VERDICT_HINTS: Record<ObservedVerdict, string> = {
  pass: "Both required components pass their runway/procedure-specific bounds.",
  fail: "At least one required component fails its bound.",
  undecided: "A required event, bound, or uncertainty interval does not support pass/fail.",
};
