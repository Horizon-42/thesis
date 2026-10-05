/**
 * The stage-A fixtures the Python code WRITES (`4dTrajectory/ts_transformer/tests/test_training_export.py` writes
 * `fixtures/stage_a/`; they are never edited by hand): the airport's index, a set's sample and two live answers of the
 * backend. A test that needs a broken file changes a clone of one of them — the file it breaks is the real one.
 */
import indexFile from "./fixtures/stage_a/index_v4.json";
import sampleFile from "./fixtures/stage_a/fixture_set/sample.json";
import answersFile from "./fixtures/stage_a/autopilot_segment.json";
import { parseTrainingSample, TRAINING_SPLITS, trainingSelectionOf, type TrainingSample, type TrainingSelection } from "../trainingSample";
import type { TrainingAutopilotRequest } from "../trainingAutopilot";

export const stageAIndex = (): Record<string, any> => structuredClone(indexFile) as Record<string, any>;
export const stageASampleFile = (): Record<string, any> => structuredClone(sampleFile) as Record<string, any>;
export const stageAAnswers = (): Array<Record<string, any>> => structuredClone(answersFile) as Array<Record<string, any>>;

export const AIRPORT = "KXXX";
export const SET_ID = "fixture_set";
export const FLIGHT_KEY = "KXXX:test_fixture";

export function stageASample(): TrainingSample {
  const parsed = parseTrainingSample(stageASampleFile(), TRAINING_SPLITS);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

export function stageASelection(): TrainingSelection {
  const sample = stageASample();
  return trainingSelectionOf(sample, sample.flights[0]);
}

/** The request the first live answer of the fixture answers: a heading word of the closed-loop sentence at Δ = 2 s. */
export function requestOf(answer: Record<string, any>): TrainingAutopilotRequest {
  return {
    airport: answer.airport, setId: answer.setId, flightKey: answer.flightKey, rowIntervalS: answer.rowIntervalS,
    column: "heading", row: answer.segment.row,
  };
}
