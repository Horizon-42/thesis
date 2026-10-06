/**
 * trafficOutcome.ts
 * -----------------
 * The plain names of a traffic run's outcomes — the ONE mapping behind the Flights table tooltip and the
 * multi-aircraft result table.
 *
 * An outcome is the loop's (`TRAFFIC_OUTCOMES`, a MIRROR of the constants in
 * `4dTrajectory/optimization/traffic/loop.py`, pinned by `aeroviz-4d/python/tests/test_traffic_mirrors.py`) or, for an
 * aircraft that was not optimized, the reason of its failed record (`BaselineFailed: …`, `block failed: …`). An
 * outcome this file does not know is shown as it is: a name made up for it would be a guess.
 */

/** The outcomes of an aircraft that was optimized, in the order `loop.py` declares them. */
export const TRAFFIC_OUTCOMES = [
  "separated_at_baseline",
  "separated",
  "unresolved",
  "solve_failed",
  "wake_at_fixed_time",
] as const;

export type TrafficOutcome = typeof TRAFFIC_OUTCOMES[number];

const LOSS_LEFT = "loss left — the best of its solves is shown (fewest losses)";

const OUTCOME_NAMES: Record<TrafficOutcome, string> = {
  separated_at_baseline: "separated at the first solve (no re-solve)",
  separated: "separated after re-solve",
  // MD14: the loop keeps the solve with the fewest counted loss instants, so in both the record is the best of its solves
  unresolved: LOSS_LEFT,
  solve_failed: LOSS_LEFT,
  wake_at_fixed_time: "wake loss left (landing time fixed)",
};

/** Reasons of aircraft that were not optimized, matched by prefix, the most specific first. */
const REASON_NAMES: ReadonlyArray<readonly [string, string]> = [
  ["BaselineFailed: slot solve", "not optimized: its slot could not be flown"],
  ["BaselineFailed: ETA solve", "not optimized: no ETA"],
  ["block failed:", "not optimized: block failed"],
  ["BaselineFailed", "not optimized"],
];

function isTrafficOutcome(raw: string): raw is TrafficOutcome {
  return (TRAFFIC_OUTCOMES as readonly string[]).includes(raw);
}

/** The plain name of an outcome; an unknown one comes back as it is. */
export function trafficOutcomeName(raw: string): string {
  if (isTrafficOutcome(raw)) return OUTCOME_NAMES[raw];
  return REASON_NAMES.find(([prefix]) => raw.startsWith(prefix))?.[1] ?? raw;
}

/** The plain name, then the raw string (title text of a row); an outcome shown raw is not repeated. */
export function trafficOutcomeTitle(raw: string): string {
  const name = trafficOutcomeName(raw);
  return name === raw ? raw : `${name} (${raw})`;
}
