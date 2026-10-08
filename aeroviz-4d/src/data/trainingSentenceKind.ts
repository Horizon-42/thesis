/**
 * trainingSentenceKind.ts
 * -----------------------
 * Which kind of sentence a flown flight on screen is (frontend §3 item 11, D159), from what the flight was derived from:
 * a sample of stage B's prior is the base's; a round of a window is the base's at the start of a campaign that starts
 * from the base and a post-trained model's otherwise (after a round, or at a start from another campaign's round, D162); anything else is the labelled sentence flown by the executor (the closed loop). Its colour
 * is `TRAINING_SENTENCE_COLOR`'s, the one map.
 */

import { TRAINING_SENTENCE_COLOR, type TrainingSentenceKind } from "../utils/trainingWordColors";
import { trainingPriorOriginOf } from "./trainingPriorSample";
import type { TrainingFlight } from "./trainingSample";
import {
  roundLabel, TRAINING_WINDOW_START, trainingWindowOriginOf, type TrainingWindowRound, type TrainingWindowStage,
  type TrainingWindowStart,
} from "./trainingWindowSample";

/** The kind of the sentences a round of a campaign of ``stage`` says: the start is the base's when the campaign starts
 *  from the base, a post-trained model's when it starts from another campaign's round (D162; stage D's from stage C's,
 *  D164); a round after it is stage C's post-trained model's, or stage D's model's (§3 item 11). */
export function roundKind(round: TrainingWindowRound, start: TrainingWindowStart | null, stage: TrainingWindowStage): TrainingSentenceKind {
  if (round === TRAINING_WINDOW_START) return start === null ? "base" : "postTrained";
  return stage === "D" ? "multi" : "postTrained";
}

/** What each kind is, in a few words (the legends' names). */
export const TRAINING_SENTENCE_KIND_TEXT: Record<TrainingSentenceKind, string> = {
  observed: "the observed track",
  closedLoop: "the closed loop (the labelled sentence flown by the executor)",
  base: "the base's sentence, flown by the executor",
  postTrained: "a post-trained round's sentence, flown by the executor",
  multi: "stage D's model's sentence, flown by the executor",
};

/** The name of the sentence on screen, as the read-back window's title says it: the labelled sentence, the closed loop at
 *  its Δ, a sample of the prior, a round of a campaign. */
export function sentenceName(flight: TrainingFlight, intervalS: number | null): string {
  if (intervalS === null) return "labelled sentence";
  const prior = trainingPriorOriginOf(flight);
  if (prior?.sentence) return `sample ${prior.sentence.sample} · Δ ${intervalS} s`;
  const window = trainingWindowOriginOf(flight);
  if (window !== undefined) return `${roundLabel(window.round, window.sample.model.start)} · Δ ${intervalS} s`;
  return `closed loop · Δ ${intervalS} s`;
}

/** The kind of the flown sentence of ``flight`` (its closed-loop reading at the Δ on screen). */
export function flownSentenceKind(flight: TrainingFlight): TrainingSentenceKind {
  if (trainingPriorOriginOf(flight)?.sentence) return "base";
  const window = trainingWindowOriginOf(flight);
  if (window !== undefined) return roundKind(window.round, window.sample.model.start, window.sample.stage);
  return "closedLoop";
}

/** The colour of the flown sentence of ``flight``. */
export function flownSentenceColour(flight: TrainingFlight): string {
  return TRAINING_SENTENCE_COLOR[flownSentenceKind(flight)];
}
