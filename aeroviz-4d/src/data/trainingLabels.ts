/**
 * trainingLabels.ts
 * -----------------
 * The two label switches of every Training stage's Draw box (the user, 2026-10-12): the AIRCRAFT LABELS (what is written
 * beside an aircraft: the attitude under the aircraft at the cursor, the key of another aircraft, the callsign of another
 * commanded aircraft, the live executor's clock) and the FAILURE LABELS (what is written at an outcome: a loss of
 * separation, where a flown path ended, the DA check). A switch shows or hides the text only; the points and the aircraft
 * stay. A store of its own, read with `useTrainingLabels` by the scene's leaves and the panel.
 */

import { useSyncExternalStore } from "react";

export interface TrainingLabels {
  aircraft: boolean;
  failures: boolean;
}

let labels: TrainingLabels = { aircraft: true, failures: true };
const listeners = new Set<() => void>();

export function setTrainingLabels(name: keyof TrainingLabels, shown: boolean): void {
  labels = { ...labels, [name]: shown };
  listeners.forEach((listener) => listener());
}

export function useTrainingLabels(): TrainingLabels {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => labels,
  );
}
