/**
 * useTrainingAutopilot.ts
 * -----------------------
 * The Training panel's live executor (`data/trainingAutopilot.ts`): the PICKED word's segment of the flight on screen —
 * `trainingPick`, set by the sentence bar's Fly button or by a band clicked, and reset with the flight (`AppContext`) —
 * is flown by the backend the moment it is picked, and the answer is published as `trainingAutopilot` for the sentence
 * bar, the read-back window and 3D.
 *
 * EVERY PICK IS FLOWN AGAIN: nothing is cached here — the point is to see what the executor does now. A MODEL's word
 * (`TrainingPick.source`) is asked with the model's words as the view read them (its sample's sentence): the backend flies
 * them, it reads no overlay. The cursor asks nothing (the charts move it on hover); a new pick, or a new attempt at the
 * same one ("Fly again", `nextPick`), does. A request still in flight when the pick moves is aborted, and an answer to anything but the current
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
import { generationOnScreen } from "../data/trainingOverlays";

export default function useTrainingAutopilot(backendUrl: string = AEROVIZ_BACKEND_URL): void {
  const { trainingSelection, trainingPick, setTrainingAutopilot, trainingGenerations } = useApp();
  // The answer is read against the flight on screen; the pick is reset with it, so the latest is the one asked about.
  const selection = useRef(trainingSelection);
  selection.current = trainingSelection;

  // The segment picked and the attempt at it, as a key — a model's word with the sentence it is a word of (none, when
  // the model's overlay is no longer published for the flight: nothing is asked).
  const key = useMemo(() => {
    // a window's aircraft is not flown live: the backend opens read-back sets only (`TrainingSelection.liveExecutor`)
    if (trainingSelection === null || trainingPick === null || !trainingSelection.liveExecutor) return null;
    const read = trainingPick.source === null ? null
      : generationOnScreen(trainingGenerations, trainingPick.source, trainingSelection);
    const sentence = read?.sentence ?? null;
    if (trainingPick.source !== null && sentence?.sample !== trainingPick.source.sample) return null;
    // a sample from an augmented start is flown from that start (`start`: its overlay's flight's; null from its own)
    const model = read === null || sentence === null ? null
      : { sentence, procedureMasks: read.view.overlay.generation.procedureMasks,
        augmentation: read.view.flight.start?.augmentation ?? null };
    return JSON.stringify({ request: trainingAutopilotRequest(trainingSelection, trainingPick, model), attempt: trainingPick.attempt });
  }, [trainingSelection, trainingPick, trainingGenerations]);

  useEffect(() => {
    if (key === null) {
      setTrainingAutopilot(null);
      return;
    }
    const { request } = JSON.parse(key) as { request: TrainingAutopilotRequest };
    const controller = new AbortController();
    setTrainingAutopilot({ status: "flying", request });
    requestTrainingAutopilot(backendUrl, request, controller.signal)
      .then((raw) => {
        if (controller.signal.aborted) return;
        // not aborted: the pick, and so the flight on screen, is the one asked about
        const parsed = parseTrainingAutopilot(raw, request, selection.current!);
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

  useEffect(() => () => setTrainingAutopilot(null), [setTrainingAutopilot]);
}
