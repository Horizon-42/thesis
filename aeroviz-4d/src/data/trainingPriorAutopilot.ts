/**
 * trainingPriorAutopilot.ts
 * -------------------------
 * THE EXECUTOR, LIVE, on a sentence of a Training set of stage B: the word the user CLICKED in the sentence on screen —
 * one the prior said, or the flight's closed-loop sentence at the prior's Δ — flown by the backend now
 * (`POST /autopilot/prior-segment`, `aeroviz_backend/autopilot_segment/prior.py`), from the sentence's first predicted step
 * with the sentence's own words. The answer is stage A's segment answer plus the sentence it flew; it is read by stage A's
 * reader (`trainingAutopilot.ts`: the segment is the word's, its cycles are on the flight clock) after the checks here.
 *
 * The flight on screen is a DERIVED flight (`trainingPriorFlightView`): its key names its sentence, so the pick, the cursor
 * and the live answer belong to the sentence. The request carries the set's own flight key and the sentence.
 */

import {
  parseTrainingAutopilot,
  TRAINING_AUTOPILOT_CLIENT_ID,
  type TrainingAutopilotRequest,
  type TrainingAutopilotSegment,
  TRAINING_AUTOPILOT_SCHEMA,
} from "./trainingAutopilot";
import { attempt, Reader, type Parsed } from "./trainingReader";
import { trainingPriorOriginOf, type TrainingPriorWhich } from "./trainingPriorSample";
import type { TrainingSelection } from "./trainingSample";

/** MIRROR of `aeroviz_backend/autopilot_segment/prior.py` `SCHEMA`: the backend's answer; anything else is refused by name.
 *  A name changes with the payload's shape, on both sides, in one change. */
export const TRAINING_PRIOR_AUTOPILOT_SCHEMA = "aeroviz-autopilot-prior-segment-v1";
/** MIRROR of `prior.py` `CLOSED_LOOP`: the request's sentence that names the flight's closed-loop sentence. */
export const TRAINING_PRIOR_CLOSED_LOOP = "closedLoop";
export const TRAINING_PRIOR_AUTOPILOT_PATH = "/autopilot/prior-segment";
/** This page's prior requests, as the backend knows them: a client id of their own (a request of the same page supersedes an
 *  older one still waiting or flying there), numbered in the order the page makes them, apart from stage A's. */
export const TRAINING_PRIOR_CLIENT_ID = `${TRAINING_AUTOPILOT_CLIENT_ID}-prior`;
let requestSeq = 0;

/** The body of the request for the flown flight's pick: the set's flight and the sentence, not the derived flight. */
export function trainingPriorRequestBody(request: TrainingAutopilotRequest, selection: TrainingSelection): Record<string, unknown> {
  const origin = trainingPriorOriginOf(selection.flight);
  if (origin === undefined) throw new Error(`${selection.flight.flightKey} is not a sentence of a prior set`);
  return { airport: request.airport, setId: request.setId, flightKey: origin.flightKey, sentence: origin.which, column: request.column, row: request.row };
}

/** The backend's answer against the request and the sentence on screen: refused by name unless it is the schema, the flight
 *  and the sentence asked for, then read as stage A's answer. */
export function parseTrainingPriorAutopilot(
  raw: unknown, request: TrainingAutopilotRequest, selection: TrainingSelection,
): Parsed<TrainingAutopilotSegment> {
  const checked = attempt(() => {
    const answer = Reader.of(raw, "autopilot answer");
    answer.oneOf("schema", [TRAINING_PRIOR_AUTOPILOT_SCHEMA]);
    const origin = trainingPriorOriginOf(selection.flight);
    if (origin === undefined) return answer.fail(`${selection.flight.flightKey} is not a sentence of a prior set`);
    if (answer.string("flightKey") !== origin.flightKey) answer.fail(`flightKey is ${answer.raw("flightKey")}, but ${origin.flightKey} was asked for`);
    const which = answer.raw("sentence") as TrainingPriorWhich;
    if (which !== origin.which) answer.fail(`sentence is ${JSON.stringify(which)}, but ${JSON.stringify(origin.which)} was asked for`);
    // stage A's reader reads the segment under its own name and the derived flight's key (what the views hold)
    return { ...(raw as Record<string, unknown>), schema: TRAINING_AUTOPILOT_SCHEMA, flightKey: request.flightKey };
  });
  return checked.ok ? parseTrainingAutopilot(checked.value, request, selection) : checked;
}

/** Ask the backend to fly the word's segment of the sentence on screen. Resolves to its answer's JSON (parse it with
 *  `parseTrainingPriorAutopilot`); rejects with the backend's own refusal, or with the reason nothing answered. */
export async function requestTrainingPriorAutopilot(
  backendUrl: string, body: Record<string, unknown>, signal?: AbortSignal,
): Promise<unknown> {
  const url = `${backendUrl.replace(/\/+$/, "")}${TRAINING_PRIOR_AUTOPILOT_PATH}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, clientId: TRAINING_PRIOR_CLIENT_ID, seq: (requestSeq += 1) }), signal,
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
