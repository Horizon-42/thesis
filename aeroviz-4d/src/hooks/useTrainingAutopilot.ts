/**
 * useTrainingAutopilot.ts
 * -----------------------
 * The Training panel's live executor (`data/trainingAutopilot.ts`): the PICKED word's segment of the selected flight —
 * `trainingPick`, set by the sentence bar's "Fly this segment" button or by a band clicked while the panel's switch is on
 * — is flown by the backend the moment it is picked, and the answer is published as `trainingAutopilot` for the sentence
 * bar, the read-back window and 3D.
 *
 * EVERY PICK IS FLOWN AGAIN: nothing is cached here and no overlay is read — the point is to see what the executor does
 * now. The cursor asks nothing (the charts move it on hover, and the lines drawn are there to be hovered); a new pick, or a
 * new attempt at the same one ("Fly again", `nextPick`), does. A request still in flight when the pick moves is aborted,
 * and an answer to anything but the current pick is never published.
 */

import { useEffect, useMemo } from "react";
import { useApp } from "../context/AppContext";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import {
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  type TrainingAutopilotRequest,
} from "../data/trainingAutopilot";
import type { TrainingSample } from "../data/trainingSample";

export default function useTrainingAutopilot(
  airport: string | null, sample: TrainingSample | null, backendUrl: string = AEROVIZ_BACKEND_URL,
): void {
  const { trainingSelection, trainingPick, setTrainingAutopilot } = useApp();

  // The segment picked and the attempt at it, as a key: of the flight on screen, in the set on screen.
  const key = useMemo(() => {
    if (!airport || !sample || sample.airport !== airport || !trainingSelection || trainingPick === null) return null;
    const { flight } = trainingSelection;
    if (trainingPick.flightKey !== flight.flightKey || !sample.flights.some((item) => item.flightKey === flight.flightKey)) {
      return null;
    }
    const request: TrainingAutopilotRequest = {
      airport, setId: sample.setId, flightKey: flight.flightKey, column: trainingPick.column, row: trainingPick.row,
    };
    return JSON.stringify({ request, attempt: trainingPick.attempt });
  }, [airport, sample, trainingSelection, trainingPick]);

  useEffect(() => {
    if (key === null || sample === null) {
      setTrainingAutopilot(null);
      return;
    }
    const { request } = JSON.parse(key) as { request: TrainingAutopilotRequest };
    const controller = new AbortController();
    const sent = performance.now();
    setTrainingAutopilot({ status: "flying", request });
    requestTrainingAutopilot(backendUrl, request, controller.signal)
      .then((raw) => {
        if (controller.signal.aborted) return;
        const parsed = parseTrainingAutopilot(raw, request, sample);
        setTrainingAutopilot(parsed.ok
          ? { status: "ready", request, segment: parsed.value, playedAt: Date.now(), roundTripS: (performance.now() - sent) / 1000 }
          : { status: "failed", request, problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setTrainingAutopilot({ status: "failed", request, problem: error instanceof Error ? error.message : String(error) });
      });
    return () => controller.abort();
  }, [key, sample, backendUrl, setTrainingAutopilot]);

  useEffect(() => () => setTrainingAutopilot(null), [setTrainingAutopilot]);
}
