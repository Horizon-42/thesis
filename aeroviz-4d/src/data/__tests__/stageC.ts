/**
 * The stage-C fixtures the Python code WRITES (`4dTrajectory/ts_transformer/tests/test_post_training_export.py` writes
 * `fixtures/stage_c/index_post_v4.json` and `fixture-windows/sample.json`; `aeroviz_backend/tests/test_window_segment.py`
 * writes `autopilot_window_segment.json`; the same Python test file writes `start_from_round.json`, `post_train.start_of`'s
 * setting of a start from another campaign's round, D162; they are never edited by hand): the airport's index of window sets, a set's
 * sample of three windows of one synthetic flight (real; A, the flight itself inserted 8 s ahead, lost at its first row
 * flown; B, its start moved), each flown by the start model, and two live answers of the backend.
 */
import indexFile from "./fixtures/stage_c/index_post_v4.json";
import sampleFile from "./fixtures/stage_c/fixture-windows/sample.json";
import answersFile from "./fixtures/stage_c/autopilot_window_segment.json";
import startFile from "./fixtures/stage_c/start_from_round.json";
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

/** The sample file with its campaign starting from another campaign's round (the setting `post_train.start_of` writes). */
export function stageCSampleFileFromRound(): Record<string, any> {
  const raw = stageCSampleFile();
  raw.model.settings.start = structuredClone(startFile.start);
  return raw;
}

/** `stageCSample` with its campaign starting from another campaign's round (`stageCSampleFileFromRound`). */
export function stageCSampleFromRound(): TrainingWindowSample {
  const parsed = parseTrainingWindowSample(stageCSampleFileFromRound(), "C");
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

export function stageCSample(): TrainingWindowSample {
  const parsed = parseTrainingWindowSample(stageCSampleFile(), "C");
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

/** The sentence bar's selection of window ``place`` with its (one) commanded aircraft's ``round`` sentence on screen. */
export function stageCSelection(sample: TrainingWindowSample, place: number, round: TrainingWindowRound = "start"): TrainingSelection {
  const window = sample.windows[place];
  return trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, window, window.commanded[0], round));
}

/** The request a live answer of the fixture answers, as the view makes it (the derived flight's key). */
export function windowRequestOf(answer: Record<string, any>, selection: TrainingSelection): TrainingAutopilotRequest {
  return {
    airport: answer.airport, setId: answer.setId, flightKey: selection.flight.flightKey, rowIntervalS: answer.rowIntervalS,
    column: ["runway", "heading", "altitude", "angle", "speed"][answer.segment.column] as TrainingAutopilotRequest["column"],
    row: answer.segment.row,
  };
}
