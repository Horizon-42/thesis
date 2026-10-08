/**
 * The stage-D fixtures the Python code WRITES (stage D's export, `experiments/multi_training_export.py`, on a synthetic
 * window of two commanded aircraft — `4dTrajectory/ts_transformer/tests/test_multi_training_export.py` writes
 * `fixtures/stage_d/`; never edited by hand): the anchor and a copy of it joining 32 s behind it on the same path; the
 * anchor answers its loss with the copy, is silent from there and flies on (the loss between them goes on while the copy
 * is observed: one loss, not written again); the copy answers its own.
 */
import indexFile from "./fixtures/stage_d/index_multi_v1.json";
import sampleFile from "./fixtures/stage_d/fixture-windows-two/sample.json";
import {
  parseTrainingWindowSample,
  trainingWindowFlightView,
  trainingWindowSelectionOf,
  type TrainingWindowRound,
  type TrainingWindowSample,
} from "../trainingWindowSample";
import type { TrainingSelection } from "../trainingSample";

export const stageDIndex = (): Record<string, any> => structuredClone(indexFile) as Record<string, any>;
export const stageDSampleFile = (): Record<string, any> => structuredClone(sampleFile) as Record<string, any>;

export const TWO_SET_ID = "fixture-windows-two";

export function stageDSample(): TrainingWindowSample {
  const parsed = parseTrainingWindowSample(stageDSampleFile(), "D");
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

/** The sentence bar's selection of window ``place``'s commanded aircraft ``member`` with its ``round`` sentence on screen. */
export function stageDSelection(sample: TrainingWindowSample, place: number, member: number,
  round: TrainingWindowRound = "start"): TrainingSelection {
  const window = sample.windows[place];
  return trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, window, window.commanded[member], round));
}
