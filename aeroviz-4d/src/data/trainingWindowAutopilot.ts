/**
 * trainingWindowAutopilot.ts
 * --------------------------
 * THE EXECUTOR, LIVE, on a window of a Training set: the word the user CLICKED in the sentence on screen — one commanded
 * aircraft's, one round's — flown by the backend now (`POST /autopilot/window-segment`,
 * `aeroviz_backend/autopilot_segment/window.py`), from the aircraft's first predicted step — window B's from its moved
 * start — with the round's own words; the other aircraft fly their states of the round and do not react (frontend §5.5);
 * an aircraft whose flight ended at a loss of separation is flown to its loss at the latest. The answer is stage A's
 * segment answer plus the window, the aircraft and the round it flew; it is read by stage A's reader
 * (`trainingAutopilot.ts`) after the checks here.
 *
 * The flight on screen is a DERIVED flight (`trainingWindowFlightView`): its key names its window, aircraft and round, so
 * the pick, the cursor and the live answer belong to them. The request carries the window's place in the set, the
 * aircraft's dataset id and the round.
 */

import {
  parseTrainingAutopilot,
  TRAINING_AUTOPILOT_CLIENT_ID,
  type TrainingAutopilotRequest,
  type TrainingAutopilotSegment,
  TRAINING_AUTOPILOT_SCHEMA,
} from "./trainingAutopilot";
import { attempt, Reader, type Parsed } from "./trainingReader";
import { trainingWindowOriginOf } from "./trainingWindowSample";
import type { TrainingSelection } from "./trainingSample";

/** MIRROR of `aeroviz_backend/autopilot_segment/window.py` `SCHEMA`: the backend's answer; anything else is refused by
 *  name. A name changes with the payload's shape, on both sides, in one change (v2: the request names the aircraft). */
export const TRAINING_WINDOW_AUTOPILOT_SCHEMA = "aeroviz-autopilot-window-segment-v2";
export const TRAINING_WINDOW_AUTOPILOT_PATH = "/autopilot/window-segment";
/** This page's window requests, as the backend knows them: a client id of their own, apart from stage A's and B's. */
export const TRAINING_WINDOW_CLIENT_ID = `${TRAINING_AUTOPILOT_CLIENT_ID}-window`;
let requestSeq = 0;

/** The body of the request for the flown flight's pick: the set's window, the aircraft and the round, not the derived
 *  flight. */
export function trainingWindowRequestBody(request: TrainingAutopilotRequest, selection: TrainingSelection): Record<string, unknown> {
  const origin = trainingWindowOriginOf(selection.flight);
  if (origin === undefined) throw new Error(`${selection.flight.flightKey} is not a window of a window set`);
  return { airport: request.airport, setId: request.setId, window: origin.window.index, aircraft: origin.aircraft.datasetId,
    round: origin.round, column: request.column, row: request.row };
}

/** The backend's answer against the request and the window on screen: refused by name unless it is the schema, the
 *  window, the aircraft and the round asked for, then read as stage A's answer. */
export function parseTrainingWindowAutopilot(
  raw: unknown, request: TrainingAutopilotRequest, selection: TrainingSelection,
): Parsed<TrainingAutopilotSegment> {
  const checked = attempt(() => {
    const answer = Reader.of(raw, "autopilot answer");
    answer.oneOf("schema", [TRAINING_WINDOW_AUTOPILOT_SCHEMA]);
    const origin = trainingWindowOriginOf(selection.flight);
    if (origin === undefined) return answer.fail(`${selection.flight.flightKey} is not a window of a window set`);
    if (answer.raw("window") !== origin.window.index) answer.fail(`window is ${JSON.stringify(answer.raw("window"))}, but ${origin.window.index} was asked for`);
    if (answer.raw("round") !== origin.round) answer.fail(`round is ${JSON.stringify(answer.raw("round"))}, but ${JSON.stringify(origin.round)} was asked for`);
    if (answer.string("aircraft") !== origin.aircraft.datasetId) answer.fail(`aircraft is ${answer.raw("aircraft")}, but ${origin.aircraft.datasetId} was asked for`);
    if (answer.string("datasetId") !== origin.aircraft.datasetId) answer.fail(`datasetId is ${answer.raw("datasetId")}, but the aircraft's is ${origin.aircraft.datasetId}`);
    // stage A's reader reads the segment under its own name and the derived flight's key (what the views hold)
    return { ...(raw as Record<string, unknown>), schema: TRAINING_AUTOPILOT_SCHEMA, flightKey: request.flightKey };
  });
  return checked.ok ? parseTrainingAutopilot(checked.value, request, selection) : checked;
}

/** Ask the backend to fly the word's segment of the window on screen. Resolves to its answer's JSON (parse it with
 *  `parseTrainingWindowAutopilot`); rejects with the backend's own refusal, or with the reason nothing answered. */
export async function requestTrainingWindowAutopilot(
  backendUrl: string, body: Record<string, unknown>, signal?: AbortSignal,
): Promise<unknown> {
  const url = `${backendUrl.replace(/\/+$/, "")}${TRAINING_WINDOW_AUTOPILOT_PATH}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, clientId: TRAINING_WINDOW_CLIENT_ID, seq: (requestSeq += 1) }), signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(`the backend at ${backendUrl} did not answer (is it running? ./start_aeroviz_fullstack.sh): ${
      error instanceof Error ? error.message : String(error)}`);
  }
  const text = await response.text();
  let answer: unknown;
  try {
    answer = JSON.parse(text);
  } catch {
    throw new Error(`the backend at ${backendUrl} answered HTTP ${response.status} with something that is not JSON: ${text.slice(0, 120)}`);
  }
  if (!response.ok) {
    const refusal = typeof answer === "object" && answer !== null && typeof (answer as { error?: unknown }).error === "string"
      ? (answer as { error: string }).error : `HTTP ${response.status}`;
    throw new Error(`the backend refused (${response.status}): ${refusal}`);
  }
  return answer;
}
