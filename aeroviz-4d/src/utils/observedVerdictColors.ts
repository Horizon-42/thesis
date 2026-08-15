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

import type { EvaluationRow } from "../data/evaluationReport";

export type ObservedVerdict = "pass" | "fail" | "undecided";
export type ObservedVerdictFilter = "all" | ObservedVerdict;

/** Path colours. Deliberately desaturated for grey so it recedes behind the verdicts. */
export const OBSERVED_VERDICT_COLORS: Record<ObservedVerdict, string> = {
  pass: "rgb(60, 200, 90)",
  fail: "rgb(230, 70, 70)",
  undecided: "rgb(150, 150, 155)",
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

/**
 * The verdict for one report row.
 *
 * `undecided` is the UI spelling of the report's `indeterminate` verdict.
 */
export function verdictOfRow(row: EvaluationRow): ObservedVerdict {
  return row.verdict === "indeterminate" ? "undecided" : row.verdict;
}

/**
 * Index a report's rows by `flight_key` for joining against rendered entity ids.
 *
 * Keyed on `flight_key`, never on `id`: `id` is the callsign, and a callsign is not
 * unique (552 distinct callsigns across 996 KRDU arrivals). Joining on it swaps
 * verdicts between namesakes — a mistake this project has already made in four
 * separate layers. Rows without a `flight_key` are skipped rather than guessed at.
 */
export function verdictsByFlightKey(rows: EvaluationRow[]): Map<string, ObservedVerdict> {
  const out = new Map<string, ObservedVerdict>();
  for (const row of rows) {
    if (row.flight_key) out.set(row.flight_key, verdictOfRow(row));
  }
  return out;
}

/** Verdict counts, for the legend's "n of m" readout. */
export function countVerdicts(
  verdicts: Iterable<ObservedVerdict>,
): Record<ObservedVerdict, number> {
  const counts: Record<ObservedVerdict, number> = { pass: 0, fail: 0, undecided: 0 };
  for (const v of verdicts) counts[v] += 1;
  return counts;
}

/**
 * Whether one already-sampled observed trajectory passes the display-only verdict filter.
 * Sampling deliberately happens before this predicate is applied; `all` therefore means
 * all trajectories in the current sample, never all trajectories in the loaded harvest.
 */
export function matchesObservedVerdictFilter(
  verdict: ObservedVerdict,
  filter: ObservedVerdictFilter,
): boolean {
  return filter === "all" || verdict === filter;
}
