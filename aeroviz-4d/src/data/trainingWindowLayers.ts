/**
 * trainingWindowLayers.ts
 * -----------------------
 * What the window sets of stages C and D draw in 3D beyond stage A's layers: the switches of the other aircraft (their
 * tracks and where each is at the cursor), the loss of separation, the window's other rounds and (D) the tracks of its
 * other commanded aircraft — a store of their own (stage A's and stage B's switches are not changed for stage C). Read with
 * `useTrainingWindowLayers`; the scene's leaf and the panel share it. Beside it, the commanded aircraft last clicked in the
 * scene (`pickTrainingWindowAircraft`), which the session selects (frontend §5.2).
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
  /** Stage D: the flown tracks of the window's other commanded aircraft in the round (never their points). */
  otherCommanded: boolean;
}

let layers: TrainingWindowLayers = { traffic: true, trafficTracks: true, loss: true, otherRounds: true, otherCommanded: true };
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

/** A commanded aircraft clicked in the scene: its dataset id, and a count so that the same aircraft clicked again is a new
 *  pick. */
export interface TrainingWindowPick {
  datasetId: string;
  count: number;
}

let pick: TrainingWindowPick | null = null;
const pickListeners = new Set<() => void>();

export function pickTrainingWindowAircraft(datasetId: string): void {
  pick = { datasetId, count: (pick?.count ?? 0) + 1 };
  pickListeners.forEach((listener) => listener());
}

export function useTrainingWindowPick(): TrainingWindowPick | null {
  return useSyncExternalStore(
    (listener) => {
      pickListeners.add(listener);
      return () => pickListeners.delete(listener);
    },
    () => pick,
  );
}
