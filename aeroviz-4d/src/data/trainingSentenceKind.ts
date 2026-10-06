/**
 * trainingSentenceKind.ts
 * -----------------------
 * Which kind of sentence a flown flight on screen is (frontend §3 item 11, D159), from what the flight was derived from:
 * a sample of stage B's prior is the base's; a round of a window is the base's at the start of the campaign and a
 * post-trained model's after; anything else is the labelled sentence flown by the executor (the closed loop). Its colour
 * is `TRAINING_SENTENCE_COLOR`'s, the one map.
 */

import { TRAINING_SENTENCE_COLOR, type TrainingSentenceKind } from "../utils/trainingWordColors";
import { trainingPriorOriginOf } from "./trainingPriorSample";
import type { TrainingFlight } from "./trainingSample";
import { TRAINING_WINDOW_START, trainingWindowOriginOf, type TrainingWindowRound } from "./trainingWindowSample";

/** The kind of the sentences a round of a campaign says: the start is the base's. */
export function roundKind(round: TrainingWindowRound): TrainingSentenceKind {
  return round === TRAINING_WINDOW_START ? "base" : "postTrained";
}

/** What each kind is, in a few words (the legends' names). */
export const TRAINING_SENTENCE_KIND_TEXT: Record<TrainingSentenceKind, string> = {
  observed: "the observed track",
  closedLoop: "the closed loop (the labelled sentence flown by the executor)",
  base: "the base's sentence, flown by the executor",
  postTrained: "a post-trained round's sentence, flown by the executor",
  multi: "stage D's model's sentence, flown by the executor",
};

/** The kind of the flown sentence of ``flight`` (its closed-loop reading at the Δ on screen). */
export function flownSentenceKind(flight: TrainingFlight): TrainingSentenceKind {
  if (trainingPriorOriginOf(flight)?.sentence) return "base";
  const window = trainingWindowOriginOf(flight);
  if (window !== undefined) return roundKind(window.round);
  return "closedLoop";
}

/** The colour of the flown sentence of ``flight``. */
export function flownSentenceColour(flight: TrainingFlight): string {
  return TRAINING_SENTENCE_COLOR[flownSentenceKind(flight)];
}
