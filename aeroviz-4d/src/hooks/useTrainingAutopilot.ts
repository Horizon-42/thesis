/**
 * useTrainingAutopilot.ts
 * -----------------------
 * The Training panel's live executor (`data/trainingAutopilot.ts`): while `on`, the PICKED word's segment of the
 * selected flight — `trainingPick`, set by a click on a band of the sentence bar — is flown by the backend the moment it
 * is picked, and the answer is published as `trainingAutopilot` for the sentence bar, the read-back window and 3D.
 *
 * EVERY PICK IS FLOWN AGAIN: nothing is cached here and no overlay is read — the point is to see what the executor does
 * now. The cursor asks nothing (the charts move it on hover, and the lines drawn are there to be hovered); picking another
 * word, or the same one again after clearing it, does. `flyAgain` asks for the current segment once more. A request
 * still in flight when the pick moves is aborted, and an answer to anything but the current pick is never published.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import {
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  type TrainingAutopilotRequest,
} from "../data/trainingAutopilot";
import type { TrainingSample } from "../data/trainingSample";

export default function useTrainingAutopilot(
  on: boolean, airport: string | null, sample: TrainingSample | null, backendUrl: string = AEROVIZ_BACKEND_URL,
) {
  const { trainingSelection, trainingPick, setTrainingAutopilot } = useApp();
  const [attempt, setAttempt] = useState<number>(0);

  // The segment picked, as a key: of the flight on screen, in the set on screen.
  const key = useMemo(() => {
    if (!on || !airport || !sample || sample.airport !== airport || !trainingSelection || trainingPick === null) return null;
    const { flight } = trainingSelection;
    if (trainingPick.flightKey !== flight.flightKey || !sample.flights.some((item) => item.flightKey === flight.flightKey)) {
      return null;
    }
    const request: TrainingAutopilotRequest = {
      airport, setId: sample.setId, flightKey: flight.flightKey, column: trainingPick.column, row: trainingPick.row,
    };
    return JSON.stringify(request);
  }, [on, airport, sample, trainingSelection, trainingPick]);

  useEffect(() => {
    if (key === null || sample === null) {
      setTrainingAutopilot(null);
      return;
    }
    const request = JSON.parse(key) as TrainingAutopilotRequest;
    const controller = new AbortController();
    setTrainingAutopilot({ status: "flying", request });
    requestTrainingAutopilot(backendUrl, request, controller.signal)
      .then((raw) => {
        if (controller.signal.aborted) return;
        const parsed = parseTrainingAutopilot(raw, request, sample);
        setTrainingAutopilot(parsed.ok
          ? { status: "ready", request, segment: parsed.value, playedAt: Date.now() }
          : { status: "failed", request, problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setTrainingAutopilot({ status: "failed", request, problem: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, [key, sample, attempt, backendUrl, setTrainingAutopilot]);

  useEffect(() => () => setTrainingAutopilot(null), [setTrainingAutopilot]);

  const flyAgain = useCallback(() => setAttempt((count) => count + 1), []);
  return { selected: key !== null, flyAgain };
}
