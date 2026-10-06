/**
 * trainingSetIntent.ts
 * --------------------
 * A Training set's intent (outline §6.2 item 4): the backend's `GET /experiments/intent?run=<set id>`, which reads the
 * registry `docs/experiments/intents.json` (the one source) at each request and answers the one campaign that lists the
 * set among its runs — or names the set and the campaigns found when none or several do. Shown in the experiments
 * picker's one form (`ExperimentIntent`): the campaign's title and intent, its design, and the set's own line.
 */

import { useEffect, useState } from "react";
import type { ExperimentIntent } from "./airportData";
import type { Parsed } from "./trainingReader";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";

/** MIRROR of the backend's route (`aeroviz_backend/http_server.py`, `experiment_intent`). */
export const TRAINING_SET_INTENT_PATH = "/experiments/intent";

/** A set's intent: the picker's form, and the campaign's id. */
export interface TrainingSetIntent extends ExperimentIntent {
  campaign: string;
}

/** The backend's answer as the view reads it: the intent, or the problem by name (the set still opens). */
export function parseTrainingSetIntent(ok: boolean, raw: unknown, setId: string): Parsed<TrainingSetIntent> {
  const answer = (raw ?? {}) as Record<string, unknown>;
  if (!ok || answer.ok !== true) {
    const error = typeof answer.error === "string" ? answer.error : "the backend gave no intent";
    return { ok: false, problem: `No intent for ${setId}: ${error}` };
  }
  const fields = ["campaign", "title", "intent", "design", "line"] as const;
  const missing = fields.filter((field) => typeof answer[field] !== "string");
  if (missing.length > 0) return { ok: false, problem: `The intent of ${setId} is not readable: ${missing.join(", ")}` };
  return {
    ok: true,
    value: {
      campaign: answer.campaign as string, groupTitle: answer.title as string, group: answer.intent as string,
      run: answer.line as string, design: answer.design as string,
    },
  };
}

export async function fetchTrainingSetIntent(setId: string, backendUrl: string = AEROVIZ_BACKEND_URL
): Promise<Parsed<TrainingSetIntent>> {
  const url = `${backendUrl.replace(/\/+$/, "")}${TRAINING_SET_INTENT_PATH}?run=${encodeURIComponent(setId)}`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    return { ok: false, problem: `No intent for ${setId}: the backend at ${backendUrl} did not answer (${
      error instanceof Error ? error.message : String(error)})` };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(await response.text());
  } catch {
    return { ok: false, problem: `No intent for ${setId}: the backend answered HTTP ${response.status} with something that is not JSON` };
  }
  return parseTrainingSetIntent(response.ok, raw, setId);
}

export type TrainingSetIntentState = { status: "loading" } | { status: "ready"; intent: TrainingSetIntent }
  | { status: "absent"; problem: string };

/** The answers asked in this page, by set id: the panel and the sentence bar ask once between them. An answer that names a
 *  problem is not kept — the next view of the set asks again (an entry committed since shows). */
const asked = new Map<string, Promise<Parsed<TrainingSetIntent>>>();

function intentOf(setId: string): Promise<Parsed<TrainingSetIntent>> {
  const known = asked.get(setId);
  if (known !== undefined) return known;
  const answer = fetchTrainingSetIntent(setId).then((parsed) => {
    if (!parsed.ok) asked.delete(setId);
    return parsed;
  });
  asked.set(setId, answer);
  return answer;
}

/** The intent of ``setId`` (null: none asked). */
export function useTrainingSetIntent(setId: string | null): TrainingSetIntentState {
  const [state, setState] = useState<{ setId: string | null; value: TrainingSetIntentState }>(
    { setId: null, value: { status: "loading" } });
  useEffect(() => {
    if (setId === null) return;
    let live = true;
    setState({ setId, value: { status: "loading" } });
    intentOf(setId).then((parsed) => {
      if (live) setState({ setId, value: parsed.ok ? { status: "ready", intent: parsed.value } : { status: "absent", problem: parsed.problem } });
    });
    return () => {
      live = false;
    };
  }, [setId]);
  return state.setId === setId ? state.value : { status: "loading" };
}
