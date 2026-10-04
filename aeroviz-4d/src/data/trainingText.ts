/**
 * trainingText.ts
 * ---------------
 * The Training views' words for what the files hold — one spelling of each, shared by the panel, the sentence bar (and
 * its live-executor line), the read-back window and the 3D labels, so the export and the live executor never word one
 * judge's outcome two ways.
 */

import type { TrainingAutopilotSegment } from "./trainingAutopilot";
import {
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotView,
} from "./trainingAutopilot";
import type { TrainingColumn, TrainingCrossing, TrainingDecision, TrainingOutcome, TrainingReplay } from "./trainingSample";

/** The five columns as the views name them. */
export const TRAINING_COLUMN_LABEL: Record<TrainingColumn, string> = {
  runway: "Runway", heading: "Heading", altitude: "Altitude", angle: "Angle", speed: "Speed",
};

/** What each column's words say, for the legend and the notes. */
export const TRAINING_COLUMN_MEANING: Record<TrainingColumn, string> = {
  runway: "the runway the flight is cleared to land on, or go-around",
  heading: "the track to fly, relative to the course of the runway in force (+ to the right)",
  altitude: "a level above the airport elevation E to hold, or \"no level-off\"",
  angle: "the descent angle to fly (level, a descent class, or a climb)",
  speed: "the ground speed to fly, or left to the pilot",
};

/** How the executor's flight ended (the judge's outcomes, `autopilot/judge.py`), in words. */
export const TRAINING_OUTCOME_TEXT: Record<TrainingOutcome, string> = {
  landed: "landed: crossed the threshold lined up, inside the runway limit, after a decision-altitude check that passed",
  unstable_at_minimums: "crossed the threshold on the runway, but unstable at minimums: the decision-altitude check failed or had no point",
  crossed_too_high: "crossed the threshold on the runway's centreline, too high to land",
  crossed_off_runway: "crossed the threshold wide of the runway",
  crossed_other_runway: "crossed another runway's threshold, lined up to land on it",
  ground_contact: "reached the threshold's elevation before the threshold",
  timeout: "did not get there within its time limit",
  dynamics_failure: "left the dynamics (a non-finite state, no airspeed or a stall)",
};

/** The same, as a short tag: the flight list, the bar's chip. */
export const TRAINING_OUTCOME_TAG: Record<TrainingOutcome, string> = {
  landed: "landed",
  unstable_at_minimums: "unstable at minimums",
  crossed_too_high: "too high",
  crossed_off_runway: "off the runway",
  crossed_other_runway: "other runway",
  ground_contact: "ground contact",
  timeout: "timed out",
  dynamics_failure: "dynamics failure",
};

export function checkMark(ok: boolean): string {
  return ok ? "✓" : "✗";
}

/** Where the threshold was passed: "1.5 m right of the centreline, 20.8 m above the threshold". */
export function crossingText(crossing: TrainingCrossing): string {
  return `${Math.abs(crossing.crossM).toFixed(1)} m ${crossing.crossM >= 0 ? "right" : "left"} of the centreline, ` +
    `${crossing.heightM.toFixed(1)} m above the threshold`;
}

/** The decision-altitude check in one line: its verdict, and each of the two values it is made of with its mark. */
export function decisionText(decision: TrainingDecision): string {
  return `DA check ${decision.passed ? "passed" : "failed"} · ` +
    `${Math.abs(decision.rightM).toFixed(1)} m ${decision.rightM >= 0 ? "right" : "left"} of the centreline ` +
    `(cone half width ${decision.coneHalfWidthM.toFixed(1)} m) ${checkMark(decision.lateralOk)} · ` +
    `${Math.abs(decision.aboveGlidepathM).toFixed(1)} m ${decision.aboveGlidepathM >= 0 ? "above" : "below"} the glidepath ` +
    `${checkMark(decision.verticalOk)}`;
}

/** A flown flight's end as one line: its outcome, the crossing and the DA check. */
export function replayText(replay: TrainingReplay): string {
  const crossing = replay.crossing;
  return [
    TRAINING_OUTCOME_TEXT[replay.outcome],
    crossing === null ? "no threshold crossing" : crossingText(crossing),
    crossing === null ? null : crossing.decision === null ? "no DA check (none was made)" : decisionText(crossing.decision),
  ].filter((part) => part !== null).join(" · ");
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

/** How the live segment ended, in words. */
export function segmentEndText(segment: TrainingAutopilotSegment): string {
  const { end } = segment.segment;
  if (end === TRAINING_AUTOPILOT_SEGMENT_END) {
    return `reached the point where the next ${segment.segment.column} word is said`;
  }
  const crossing = segment.crossing;
  return `${TRAINING_OUTCOME_TEXT[end]}${crossing === null ? "" : ` — ${crossingText(crossing)}`}`;
}

/** The live answer, one short phrase for the bar. */
export function autopilotStatusText(view: Extract<TrainingAutopilotView, { status: "ready" }>): string {
  const { segment } = view;
  const { end } = segment.segment;
  return end === TRAINING_AUTOPILOT_SEGMENT_END ? "flown to the next word" : TRAINING_OUTCOME_TAG[end];
}
