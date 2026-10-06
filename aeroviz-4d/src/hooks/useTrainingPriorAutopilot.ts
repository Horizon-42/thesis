/**
 * useTrainingPriorAutopilot.ts
 * ----------------------------
 * The live executor on a sentence of a prior set (`data/trainingPriorAutopilot.ts`): the PICKED word's segment of the sentence
 * on screen is flown by the backend the moment it is picked — as stage A's `useTrainingAutopilot` does for a stage-A flight,
 * which skips a derived flight (its `trainingSelection` is one of a prior set) — and the answer is published as
 * `trainingAutopilot`. Run while the prior's session is open; every pick is flown again, nothing is cached.
 */

import { useEffect, useMemo, useRef } from "react";
import { useApp } from "../context/AppContext";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import { trainingAutopilotRequest, type TrainingAutopilotRequest } from "../data/trainingAutopilot";
import {
  parseTrainingPriorAutopilot,
  requestTrainingPriorAutopilot,
  trainingPriorRequestBody,
} from "../data/trainingPriorAutopilot";
import { trainingPriorOriginOf } from "../data/trainingPriorSample";

export default function useTrainingPriorAutopilot(backendUrl: string = AEROVIZ_BACKEND_URL): void {
  const { trainingSelection, trainingPick, setTrainingAutopilot } = useApp();
  const selection = useRef(trainingSelection);
  selection.current = trainingSelection;

  const key = useMemo(() => {
    if (trainingSelection === null || trainingPick === null || trainingPriorOriginOf(trainingSelection.flight) === undefined) return null;
    return JSON.stringify({ request: trainingAutopilotRequest(trainingSelection, trainingPick), attempt: trainingPick.attempt });
  }, [trainingSelection, trainingPick]);

  useEffect(() => {
    if (key === null) return;
    const { request } = JSON.parse(key) as { request: TrainingAutopilotRequest };
    const controller = new AbortController();
    setTrainingAutopilot({ status: "flying", request });
    requestTrainingPriorAutopilot(backendUrl, trainingPriorRequestBody(request, selection.current!), controller.signal)
      .then((raw) => {
        if (controller.signal.aborted) return;
        const parsed = parseTrainingPriorAutopilot(raw, request, selection.current!);
        setTrainingAutopilot(parsed.ok
          ? { status: "ready", request, segment: parsed.value, playedAt: Date.now() }
          : { status: "failed", request, problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setTrainingAutopilot({ status: "failed", request, problem: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, [key, backendUrl, setTrainingAutopilot]);

  // a pick is reset with the sentence on screen; its answer goes with it
  useEffect(() => {
    if (key === null) setTrainingAutopilot(null);
  }, [key, setTrainingAutopilot]);
  useEffect(() => () => setTrainingAutopilot(null), [setTrainingAutopilot]);
}
