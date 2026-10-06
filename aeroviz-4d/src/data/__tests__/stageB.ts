/**
 * The stage-B fixtures the Python code WRITES (`4dTrajectory/ts_transformer/tests/test_prior_training_export.py` writes
 * `fixtures/stage_b/index_prior_v3.json`, `fixture_set/sample.json` (train) and `fixture_val/sample.json` (the claimed validation set); `aeroviz_backend/tests/test_prior_segment.py` writes
 * `autopilot_prior_segment.json`; they are never edited by hand): the airport's index of prior sets, a set's sample and two
 * live answers of the backend on a sentence the prior said. A test that needs a broken file changes a clone of one of them.
 */
import indexFile from "./fixtures/stage_b/index_prior_v3.json";
import sampleFile from "./fixtures/stage_b/fixture_set/sample.json";
import valSampleFile from "./fixtures/stage_b/fixture_val/sample.json";
import answersFile from "./fixtures/stage_b/autopilot_prior_segment.json";
import { parseTrainingPriorSample, trainingPriorFlightView, trainingPriorSelectionOf, type TrainingPriorSample, type TrainingPriorWhich } from "../trainingPriorSample";
import type { TrainingAutopilotRequest } from "../trainingAutopilot";
import type { TrainingSelection } from "../trainingSample";

export const stageBIndex = (): Record<string, any> => structuredClone(indexFile) as Record<string, any>;
export const stageBSampleFile = (): Record<string, any> => structuredClone(sampleFile) as Record<string, any>;
export const stageBValSampleFile = (): Record<string, any> => structuredClone(valSampleFile) as Record<string, any>;
export const stageBAnswers = (): Array<Record<string, any>> => structuredClone(answersFile) as Array<Record<string, any>>;

export const PRIOR_SET_ID = "fixture_set";
export const PRIOR_VAL_SET_ID = "fixture_val";

export function stageBSample(): TrainingPriorSample {
  const parsed = parseTrainingPriorSample(stageBSampleFile());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

/** The sentence bar's selection of the set's first flight with ``which`` of its sentences on screen. */
export function stageBSelection(sample: TrainingPriorSample = stageBSample(), which: TrainingPriorWhich = 0): TrainingSelection {
  return trainingPriorSelectionOf(sample, trainingPriorFlightView(sample, sample.flights[0], which));
}

/** The request the first live answer of the fixture answers, as the view makes it (the derived flight's key). */
export function priorRequestOf(answer: Record<string, any>, selection: TrainingSelection): TrainingAutopilotRequest {
  return {
    airport: answer.airport, setId: answer.setId, flightKey: selection.flight.flightKey, rowIntervalS: answer.rowIntervalS,
    column: "heading", row: answer.segment.row,
  };
}
