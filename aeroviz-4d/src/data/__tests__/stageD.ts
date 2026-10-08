/**
 * The stage-D fixtures the Python code WRITES (frontend §8 F3: stage C's export on a synthetic window of two commanded
 * aircraft — `4dTrajectory/ts_transformer/tests/test_post_training_export.py` writes `fixtures/stage_c_two/`; never edited
 * by hand), read as stage D's set (its index file is stage D's, the format one for both stages): the anchor and a copy of
 * it joining 32 s behind it on the same path; the anchor answers its loss with the copy, is silent from there and flies on.
 */
import indexFile from "./fixtures/stage_c_two/index_post_v3.json";
import sampleFile from "./fixtures/stage_c_two/fixture-windows-two/sample.json";
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
