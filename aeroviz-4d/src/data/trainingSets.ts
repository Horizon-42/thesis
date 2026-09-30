/**
 * trainingSets.ts
 * ---------------
 * A Training set as it is OPENED, by its kind — one door for the panel and `check-publication`: a read-back set's sample
 * (`trainingSample.ts`: val flights one at a time) or a window set (`trainingTraffic.ts`: multi-aircraft windows) — and
 * which overlays are drawn over which kind, each through its own reader against the set it names.
 */

import type { Parsed } from "./trainingReader";
import {
  parseTrainingSample,
  TRAINING_READBACK_SET_KIND,
  TRAINING_TRAFFIC_SET_KIND,
  trainingFilePath,
  trainingSetRefusal,
  type TrainingReadableSetKind,
  type TrainingSample,
  type TrainingSetEntry,
  type TrainingSetHead,
} from "./trainingSample";
import { parseTrainingTrafficSet, parseTrainingWindowGenerationOverlay, type TrainingTrafficSet } from "./trainingTraffic";
import {
  parseTrainingExecutorOverlay,
  parseTrainingGenerationOverlay,
  parseTrainingPriorOverlay,
  type TrainingOverlayEntry,
  type TrainingOverlayKind,
} from "./trainingOverlays";
import { fetchJson } from "../utils/fetchJson";

export type TrainingOpenSet =
  | { kind: typeof TRAINING_READBACK_SET_KIND; sample: TrainingSample }
  | { kind: typeof TRAINING_TRAFFIC_SET_KIND; traffic: TrainingTrafficSet };

/** A listed set as this reader takes it, from the manifest alone: the kind it opens it as, or why it refuses it
 *  (`trainingSample.trainingSetRefusal`: another vocabulary or spec, a kind it does not open). */
export function openable(entry: TrainingSetEntry): { kind: TrainingReadableSetKind; refusal: null } | { kind: null; refusal: string } {
  const refusal = trainingSetRefusal(entry);
  return refusal === null ? { kind: entry.kind as TrainingReadableSetKind, refusal } : { kind: null, refusal };
}

/** An opened set's head: what its flights and its overlays' binding are read from. */
export function openSetHead(open: TrainingOpenSet): TrainingSetHead {
  return open.kind === TRAINING_READBACK_SET_KIND ? open.sample : open.traffic;
}

/** A set's file through its kind's reader. */
export function parseTrainingSet(kind: TrainingReadableSetKind, raw: unknown): Parsed<TrainingOpenSet> {
  if (kind === TRAINING_READBACK_SET_KIND) {
    const parsed = parseTrainingSample(raw);
    return parsed.ok ? { ok: true, value: { kind, sample: parsed.value } } : parsed;
  }
  const parsed = parseTrainingTrafficSet(raw);
  return parsed.ok ? { ok: true, value: { kind, traffic: parsed.value } } : parsed;
}

export async function fetchTrainingSet(airportCode: string, kind: TrainingReadableSetKind, file: string): Promise<Parsed<TrainingOpenSet>> {
  return parseTrainingSet(kind, await fetchJson<unknown>(trainingFilePath(airportCode, file)));
}

/** The readers of the overlays drawn over a read-back set's flights, by kind. */
const READBACK_OVERLAY_READERS = {
  "executor-replay": parseTrainingExecutorOverlay,
  "prior-prediction": parseTrainingPriorOverlay,
  "prior-generation": parseTrainingGenerationOverlay,
  "prior-generation-augmented": parseTrainingGenerationOverlay,
} satisfies Partial<Record<TrainingOverlayKind,
  (raw: unknown, entry: TrainingOverlayEntry, sample: TrainingSample) => Parsed<{ flights: unknown[] }>>>;

/** The overlay kinds drawn over each kind of set: another model's output on a read-back set's flights, or a model's
 *  sentences in a window set's windows. Every kind is drawn over exactly one (tested). */
export const TRAINING_OVERLAYS_OVER: Record<TrainingReadableSetKind, readonly TrainingOverlayKind[]> = {
  [TRAINING_READBACK_SET_KIND]: Object.keys(READBACK_OVERLAY_READERS) as TrainingOverlayKind[],
  [TRAINING_TRAFFIC_SET_KIND]: ["window-generation"],
};

/** An overlay's file through its kind's reader, against the set it is drawn over — refused when that set is of another
 *  kind: the flights it holds (a window overlay: the set's commanded flights, its windows read one by one). */
export function parseOverlayOver(entry: TrainingOverlayEntry, raw: unknown, open: TrainingOpenSet): Parsed<{ flights: number }> {
  if (!TRAINING_OVERLAYS_OVER[open.kind].includes(entry.kind)) {
    return { ok: false, problem: `a ${entry.kind} overlay is not drawn over a ${open.kind} set` };
  }
  if (open.kind === TRAINING_TRAFFIC_SET_KIND) {
    const parsed = parseTrainingWindowGenerationOverlay(raw, entry, open.traffic);
    return parsed.ok ? { ok: true, value: { flights: open.traffic.flights.length } } : parsed;
  }
  const parsed = READBACK_OVERLAY_READERS[entry.kind as keyof typeof READBACK_OVERLAY_READERS](raw, entry, open.sample);
  return parsed.ok ? { ok: true, value: { flights: parsed.value.flights.length } } : parsed;
}
