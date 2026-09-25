/**
 * trainingText.ts
 * ---------------
 * The Training views' words for what the files hold — one spelling of each, shared by the panel, the sentence bar, the
 * read-back and prior windows, the live executor's card and the 3D labels, so the replay and the live executor never
 * word one judge's outcome two ways.
 */

import type { TrainingCrossing, TrainingExecutorCheck, TrainingExecutorOutcome } from "./trainingOverlays";
import type { TrainingAutopilotStatus } from "./trainingAutopilot";
import type { TrainingColumn } from "./trainingSample";

/** The six columns as the views name them. */
export const TRAINING_COLUMN_LABEL: Record<TrainingColumn, string> = {
  runway: "Runway", approach: "Approach", heading: "Heading", altitude: "Altitude", angle: "Angle", speed: "Speed",
};

/** How the executor's flight ended (the judge's outcomes), in words. */
export const TRAINING_OUTCOME_TEXT: Record<TrainingExecutorOutcome, string> = {
  landed: "landed",
  crossed_without_capture: "crossed the threshold without capturing the final",
  crossed_off_runway: "crossed the threshold off the runway",
  ground_contact: "reached the threshold's elevation before the threshold",
  timeout: "did not get there within its time limit",
  dynamics_failure: "left the dynamics (a non-finite state, no airspeed or a stall)",
};

/** The same, as a short tag: the flight list, the 3D label. */
export const TRAINING_OUTCOME_TAG: Record<TrainingExecutorOutcome, string> = {
  landed: "landed",
  crossed_without_capture: "no capture",
  crossed_off_runway: "off the runway",
  ground_contact: "ground contact",
  timeout: "timed out",
  dynamics_failure: "dynamics failure",
};

/** The live executor's verdict on its word, as it is read first. */
export const TRAINING_VERDICT_TEXT: Record<TrainingAutopilotStatus, string> = {
  inside: "✓ inside its envelope",
  outside: "✗ outside its envelope",
  "not judged": "not judged",
  "no check": "no envelope of its own",
};

export function checkMark(ok: boolean): string {
  return ok ? "✓" : "✗";
}

/** A judge's check: its name, its mark, and over rows how many were inside. */
export function checkText(check: TrainingExecutorCheck): string {
  return `${checkMark(check.ok)} ${check.name}${check.rows === null ? "" : ` (${check.inside}/${check.rows} rows)`}`;
}

/** Where the threshold was passed: "1.5 m right of the centreline, 20.8 m above the threshold". */
export function crossingText(crossing: TrainingCrossing): string {
  return `${Math.abs(crossing.crossM).toFixed(1)} m ${crossing.crossM >= 0 ? "right" : "left"} of the centreline, ` +
    `${crossing.heightM.toFixed(1)} m above the threshold`;
}

/** A sha as the views show it: its first 12 characters. */
export function shortSha(sha: string): string {
  return sha.slice(0, 12);
}

/** An elapsed time as the times read: milliseconds under a second, then seconds (two decimals under ten, one above) —
 *  the unit chosen after rounding, so 0.9996 s reads "1.00 s", never "1000 ms". */
export function formatElapsed(seconds: number): string {
  const ms = Math.round(seconds * 1000);
  if (ms < 1000) return `${ms} ms`;
  return Math.round(seconds * 100) < 1000 ? `${seconds.toFixed(2)} s` : `${seconds.toFixed(1)} s`;
}
