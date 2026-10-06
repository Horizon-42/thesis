/**
 * trainingWindowLayers.ts
 * -----------------------
 * What the window sets of stage C draw in 3D beyond stage A's layers: the switches of the other aircraft (their tracks and
 * where each is at the cursor), the loss of separation and the window's other rounds — a store of their own (stage A's and
 * stage B's switches are not changed for stage C). Read with `useTrainingWindowLayers`; the scene's leaf and the panel share
 * it.
 */

import { useSyncExternalStore } from "react";

export interface TrainingWindowLayers {
  /** The other aircraft at the cursor's time, each where its record has it. */
  traffic: boolean;
  /** Each other aircraft's track over the window, thin. */
  trafficTracks: boolean;
  /** The loss of separation that ended the round on screen: the two aircraft at that time, joined. */
  loss: boolean;
  /** The flown paths of the window's other rounds, thin and faded beside the one on screen. */
  otherRounds: boolean;
}

let layers: TrainingWindowLayers = { traffic: true, trafficTracks: true, loss: true, otherRounds: true };
const listeners = new Set<() => void>();

export function setTrainingWindowLayer(name: keyof TrainingWindowLayers, shown: boolean): void {
  layers = { ...layers, [name]: shown };
  listeners.forEach((listener) => listener());
}

export function useTrainingWindowLayers(): TrainingWindowLayers {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => layers,
  );
}
