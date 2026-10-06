/**
 * The stage-C fixtures the Python code WRITES (`4dTrajectory/ts_transformer/tests/test_post_training_export.py` writes
 * `fixtures/stage_c/index_post_v2.json` and `fixture-windows/sample.json`; `aeroviz_backend/tests/test_window_segment.py`
 * writes `autopilot_window_segment.json`; they are never edited by hand): the airport's index of window sets, a set's
 * sample of three windows of one synthetic flight (real; A, the flight itself inserted 8 s ahead, lost at its first row
 * flown; B, its start moved), each flown by the start model, and two live answers of the backend.
 */
import indexFile from "./fixtures/stage_c/index_post_v2.json";
import sampleFile from "./fixtures/stage_c/fixture-windows/sample.json";
import answersFile from "./fixtures/stage_c/autopilot_window_segment.json";
import {
  parseTrainingWindowSample,
  trainingWindowFlightView,
  trainingWindowSelectionOf,
  type TrainingWindowRound,
  type TrainingWindowSample,
} from "../trainingWindowSample";
import type { TrainingAutopilotRequest } from "../trainingAutopilot";
import type { TrainingSelection } from "../trainingSample";

export const stageCIndex = (): Record<string, any> => structuredClone(indexFile) as Record<string, any>;
export const stageCSampleFile = (): Record<string, any> => structuredClone(sampleFile) as Record<string, any>;
export const stageCAnswers = (): Array<Record<string, any>> => structuredClone(answersFile) as Array<Record<string, any>>;

export const WINDOW_SET_ID = "fixture-windows";

export function stageCSample(): TrainingWindowSample {
  const parsed = parseTrainingWindowSample(stageCSampleFile());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

/** The sentence bar's selection of window ``place`` with ``round``'s sentence on screen. */
export function stageCSelection(sample: TrainingWindowSample, place: number, round: TrainingWindowRound = "start"): TrainingSelection {
  return trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, sample.windows[place], round));
}

/** The request a live answer of the fixture answers, as the view makes it (the derived flight's key). */
export function windowRequestOf(answer: Record<string, any>, selection: TrainingSelection): TrainingAutopilotRequest {
  return {
    airport: answer.airport, setId: answer.setId, flightKey: selection.flight.flightKey, rowIntervalS: answer.rowIntervalS,
    column: ["runway", "heading", "altitude", "angle", "speed"][answer.segment.column] as TrainingAutopilotRequest["column"],
    row: answer.segment.row,
  };
}
