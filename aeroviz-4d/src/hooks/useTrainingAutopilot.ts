/**
 * useTrainingAutopilot.ts
 * -----------------------
 * The Training panel's live executor (`data/trainingAutopilot.ts`): the PICKED word's segment of the flight on screen —
 * `trainingPick`, set by the sentence bar's "Fly this segment" button or by a band clicked while the panel's switch is on,
 * and reset with the flight (`AppContext`) — is flown by the backend the moment it is picked, and the answer is published
 * as `trainingAutopilot` for the sentence bar, the read-back window and 3D.
 *
 * EVERY PICK IS FLOWN AGAIN: nothing is cached here and no overlay is read — the point is to see what the executor does
 * now. The cursor asks nothing (the charts move it on hover); a new pick, or a new attempt at the same one ("Fly again",
 * `nextPick`), does. A request still in flight when the pick moves is aborted, and an answer to anything but the current
 * pick is never published.
 */

import { useEffect, useMemo, useRef } from "react";
import { useApp } from "../context/AppContext";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import {
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  trainingAutopilotRequest,
  type TrainingAutopilotRequest,
} from "../data/trainingAutopilot";

export default function useTrainingAutopilot(backendUrl: string = AEROVIZ_BACKEND_URL): void {
  const { trainingSelection, trainingPick, setTrainingAutopilot } = useApp();
  // The answer is read against the flight on screen; the pick is reset with it, so the latest is the one asked about.
  const selection = useRef(trainingSelection);
  selection.current = trainingSelection;

  // The segment picked and the attempt at it, as a key.
  const key = useMemo(() => (trainingSelection === null || trainingPick === null ? null
    : JSON.stringify({ request: trainingAutopilotRequest(trainingSelection, trainingPick), attempt: trainingPick.attempt })),
  [trainingSelection, trainingPick]);

  useEffect(() => {
    if (key === null) {
      setTrainingAutopilot(null);
      return;
    }
    const { request } = JSON.parse(key) as { request: TrainingAutopilotRequest };
    const controller = new AbortController();
    const sent = performance.now();
    setTrainingAutopilot({ status: "flying", request });
    requestTrainingAutopilot(backendUrl, request, controller.signal)
      .then((raw) => {
        if (controller.signal.aborted || selection.current === null) return;
        const parsed = parseTrainingAutopilot(raw, request, selection.current);
        setTrainingAutopilot(parsed.ok
          ? { status: "ready", request, segment: parsed.value, playedAt: Date.now(), roundTripS: (performance.now() - sent) / 1000 }
          : { status: "failed", request, problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setTrainingAutopilot({ status: "failed", request, problem: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, [key, backendUrl, setTrainingAutopilot]);

  useEffect(() => () => setTrainingAutopilot(null), [setTrainingAutopilot]);
}
