/**
 * trainingPriorLayers.ts
 * ----------------------
 * What the prior's sets draw in 3D beyond stage A's layers, and the row the panel's inspector shows: the switches of the
 * procedure's limits and of the flight's other sentences, a store of their own (stage A's `trainingLayers` is a closed set of
 * switches and is not changed for stage B). Read with `useTrainingPriorLayers`; the scene's leaf and the panel share it.
 */

import { useSyncExternalStore } from "react";

export interface TrainingPriorLayers {
  /** The procedure's limits: the region outline, the glidepath lower edge, the DA point and the entry point at the FAF. */
  procedure: boolean;
  /** The flight's other sentences (the prior's and the closed-loop one), thin and faded beside the one on screen. */
  otherSentences: boolean;
}

let layers: TrainingPriorLayers = { procedure: true, otherSentences: true };
const listeners = new Set<() => void>();

export function setTrainingPriorLayer(name: keyof TrainingPriorLayers, shown: boolean): void {
  layers = { ...layers, [name]: shown };
  listeners.forEach((listener) => listener());
}

export function useTrainingPriorLayers(): TrainingPriorLayers {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => layers,
  );
}
