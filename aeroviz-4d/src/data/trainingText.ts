/**
 * trainingText.ts
 * ---------------
 * The Training views' words for what the files hold — one spelling of each, shared by the panel, the sentence bar (and
 * its live-executor line), the read-back and prior windows and the 3D labels, so the replay and the live executor never
 * word one judge's outcome two ways.
 */

import {
  countedTwice,
  replayVerdict,
  trainingModelLabel,
  trainingRunName,
  type TrainingCrossing,
  type TrainingExecutorFlown,
  type TrainingExecutorCheck,
  type TrainingGenerationModel,
  type TrainingSentenceOutcome,
} from "./trainingOverlays";
import type { TrainingAutopilotStatus } from "./trainingAutopilot";
import {
  trainingWordLabel, TRAINING_COLUMNS, type TrainingCandidate, type TrainingColumn, type TrainingVocabulary,
} from "./trainingSample";

/** The six columns as the views name them — the angle column as "Descent": its classes are descent angles (and level,
 *  and the climb). */
export const TRAINING_COLUMN_LABEL: Record<TrainingColumn, string> = {
  runway: "Runway", approach: "Approach", heading: "Heading", altitude: "Altitude", angle: "Descent", speed: "Speed",
};

/** How the executor's flight ended (the judge's outcomes, a model's sentence stopped below the glidepath, and in a window
 *  one the judge ended for a loss of separation), in words. */
export const TRAINING_OUTCOME_TEXT: Record<TrainingSentenceOutcome, string> = {
  landed: "landed",
  crossed_too_high: "crossed the threshold on the runway's centreline, too high to land",
  crossed_off_runway: "crossed the threshold wide of the runway",
  crossed_other_runway: "crossed another runway's threshold, lined up to land on it",
  crossed_without_capture: "crossed the threshold without capturing the final",
  ground_contact: "reached the threshold's elevation before the threshold",
  timeout: "did not get there within its time limit",
  dynamics_failure: "left the dynamics (a non-finite state, no airspeed or a stall)",
  below_glidepath: "sank below the glidepath's lower edge (the procedure's altitudes stop the sentence there)",
  lost_separation: "lost separation from another aircraft it answers for (the window's judge ends it there; it flies on, silent)",
};

/** The same, as a short tag: the flight list, the 3D label. */
export const TRAINING_OUTCOME_TAG: Record<TrainingSentenceOutcome, string> = {
  landed: "landed",
  crossed_too_high: "too high",
  crossed_off_runway: "off the runway",
  crossed_other_runway: "other runway",
  crossed_without_capture: "no capture",
  ground_contact: "ground contact",
  timeout: "timed out",
  dynamics_failure: "dynamics failure",
  below_glidepath: "below glidepath",
  lost_separation: "lost separation",
};

/** What went wrong in the executor's replay of a flight (`replayVerdict`), in the fewest words — "2 words out", "timed
 *  out · 1 word out", "track refused" — or null when it landed clean: the sentence bar's header says it only then (the
 *  user, 2026-09-28). The words out are the gate's count, as the flight list's "43/45" says them. */
export function replayIssueText(flight: TrainingExecutorFlown): string | null {
  const { kind, wordsOut, refused } = replayVerdict(flight);
  if (kind === "clean") return null;
  return [
    ...(flight.outcome === "landed" ? [] : [TRAINING_OUTCOME_TAG[flight.outcome]]),
    ...(refused ? ["track refused"] : []),
    ...(wordsOut === 0 ? [] : [`${wordsOut} word${wordsOut === 1 ? "" : "s"} out`]),
  ].join(" · ");
}

/** The words the replay flew outside their envelopes, named: "heading 095° at step 164" — the word the gate counts twice
 *  (`countedTwice`) says so when both its checks failed, so the names add up to the words out. */
export function replayOutsideWords(
  flight: TrainingExecutorFlown, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[],
): string[] {
  return flight.words.filter((word) => word.status === "outside").map((word) => {
    const column = TRAINING_COLUMNS[word.column];
    const twice = countedTwice(word) && !word.checks[0].ok;
    return `${column} ${trainingWordLabel(vocabulary, candidates, column, word.value)} at step ${word.row}` +
      (twice ? " (counted twice: its band and the intercept)" : "");
  });
}

/** The live executor's verdict on its word, as it is read first. */
export const TRAINING_VERDICT_TEXT: Record<TrainingAutopilotStatus, string> = {
  inside: "in envelope",
  outside: "out of envelope",
  "not judged": "not judged",
  "no check": "no envelope",
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

/** How a model was trained: on data alone, or post-trained from which model by which method. */
export function trainingModelOrigin(model: TrainingGenerationModel): string {
  const { fineTuning } = model;
  return fineTuning === null ? "trained on data alone"
    : `post-trained from ${trainingModelLabel({ name: fineTuning.fromName, round: fineTuning.fromRound })} by ${fineTuning.schema}`;
}

/** A model named in full: its name and round, the run its rounds come from and the model it started from — a tab's,
 *  a round's, a table row's tooltip. */
export function trainingModelText(model: TrainingGenerationModel): string {
  return `${trainingModelLabel(model)} (${trainingRunName(model.run)}, ${trainingModelOrigin(model)})`;
}
