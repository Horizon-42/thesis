/**
 * observedVerdictColors.ts
 * ------------------------
 * How an observed ADS-B track is coloured by its FAA 8260.58D gate verdict.
 *
 * THREE STATES, NOT TWO — and the third is the common one.
 *
 * A pass/fail pair would be a lie about this data. Observed altitudes arrive
 * quantised to 25 ft (7.62 m) and the vertical gate window is only 9.15 m wide, so
 * after the least-squares fit the 95 % confidence interval on a threshold crossing is
 * ~6.7 m — three quarters of the window. Measured on real KRDU arrivals, **71 % of
 * established flights have an interval that straddles a gate boundary**: the data
 * cannot decide those verdicts. Painting them red would report a measurement limit as
 * a flying error.
 *
 * A fourth situation is folded into the same neutral colour: a track that never flew a
 * fittable, stabilised final approach (`established === false`) has no crossing to
 * judge at all. It is not a failure either.
 *
 *   green   passed both gates, and the interval clears both boundaries
 *   red     missed a gate, and the interval clears the boundary it missed
 *   grey    undecidable — interval straddles a boundary, or never established
 *
 * The backend decides `marginal` and `established` (`evaluation/metrics.py`); this
 * module only maps them to colours, so the UI can never invent a verdict the
 * evaluation did not reach.
 */

import type { EvaluationRow } from "../data/evaluationReport";

export type ObservedVerdict = "pass" | "fail" | "undecided";

/** Path colours. Deliberately desaturated for grey so it recedes behind the verdicts. */
export const OBSERVED_VERDICT_COLORS: Record<ObservedVerdict, string> = {
  pass: "rgb(60, 200, 90)",
  fail: "rgb(230, 70, 70)",
  undecided: "rgb(150, 150, 155)",
};

/** Short human-readable label, used by the legend and the tooltip. */
export const OBSERVED_VERDICT_LABELS: Record<ObservedVerdict, string> = {
  pass: "Within both gates",
  fail: "Outside a gate",
  undecided: "Undecidable",
};

/**
 * One-line explanation per state. The legend shows these verbatim: a reader who does
 * not know the measurement limits would otherwise read grey as "no data".
 */
export const OBSERVED_VERDICT_HINTS: Record<ObservedVerdict, string> = {
  pass: "Lateral ≤ 106.75 m and vertical inside −3.05…+6.10 m at the threshold.",
  fail: "Missed a gate by more than the measurement uncertainty.",
  undecided:
    "Either the 95 % interval straddles a gate (25 ft altitude quantisation vs a 9.15 m window), " +
    "or the flight never flew a stabilised final approach to fit.",
};

/**
 * The verdict for one report row.
 *
 * `undecided` wins over the pass/fail bit whenever the evaluation flagged the row as
 * marginal or not established — the point of the third state is that it OVERRIDES a
 * verdict the data cannot support.
 */
export function verdictOfRow(row: EvaluationRow): ObservedVerdict {
  if (row.established === false || row.marginal) return "undecided";
  return row.success ? "pass" : "fail";
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
