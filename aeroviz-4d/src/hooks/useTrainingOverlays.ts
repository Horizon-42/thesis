/**
 * useTrainingOverlays.ts
 * ----------------------
 * The Training panel's second fetch: `training/overlays.json`, and the overlays drawn over the set it has open —
 * the executor's replay and the prior's predictions (`data/trainingOverlays.ts`). Each kind has its own switch (on at
 * first), its overlay is downloaded the first time it is switched on for a set, and while it is on the panel's
 * selected flight is published (`trainingExecutor` / `trainingPrior`) for the sentence bar, the windows and 3D.
 *
 * The states all name what failed, as the panel's own do: no manifest is "none published" (the command that writes
 * one is shown), a manifest that is not one or an overlay that does not bind to the set says why, and only that
 * overlay is dropped.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingExecutorOverlay,
  fetchTrainingOverlays,
  fetchTrainingPriorOverlay,
  trainingOverlaysOf,
  type TrainingExecutorOverlay,
  type TrainingOverlayEntry,
  type TrainingOverlayKind,
  type TrainingOverlays,
  type TrainingPriorOverlay,
} from "../data/trainingOverlays";
import type { Parsed, TrainingSample } from "../data/trainingSample";

export type OverlaysManifestState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; overlays: TrainingOverlays };

export type OverlayLoad<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; overlay: T };

export interface OverlayKindState<T> {
  kind: TrainingOverlayKind;
  /** The overlays of this kind drawn over the open set, the latest listed last. */
  entries: TrainingOverlayEntry[];
  /** The one shown: the latest unless another is chosen. */
  entry: TrainingOverlayEntry | null;
  choose: (id: string) => void;
  shown: boolean;
  setShown: (shown: boolean) => void;
  load: OverlayLoad<T>;
}

type Fetcher<T> = (airport: string, entry: TrainingOverlayEntry, sample: TrainingSample) => Promise<Parsed<T>>;

function useOverlayKind<T extends { flights: Array<{ flightKey: string }> }>(
  kind: TrainingOverlayKind, manifest: OverlaysManifestState, airport: string | null, sample: TrainingSample | null,
  fetcher: Fetcher<T>,
): OverlayKindState<T> {
  // Only the open airport's own manifest and sample: for one render after a switch both are still the last
  // airport's, and an entry of theirs must not be fetched under the new airport's path.
  const entries = useMemo(
    () => (manifest.status === "ready" && sample && manifest.overlays.airport === airport && sample.airport === airport
      ? trainingOverlaysOf(manifest.overlays, sample.setId, kind) : []),
    [manifest, sample, kind, airport],
  );
  const [chosen, setChosen] = useState<string | null>(null);
  const entry = entries.find((item) => item.id === chosen) ?? entries[entries.length - 1] ?? null;
  const [shown, setShown] = useState<boolean>(true);
  // One download per overlay and sample: switching off and on again reuses it.
  const [loaded, setLoaded] = useState<{ key: string; load: OverlayLoad<T> } | null>(null);
  const key = entry && sample && airport ? `${airport}/${entry.id}@${sample.writtenUtc}` : null;
  const requested = useRef<string | null>(null);

  useEffect(() => {
    if (!shown || key === null || !entry || !sample || !airport || requested.current === key) return;
    requested.current = key;
    setLoaded({ key, load: { status: "loading" } });
    fetcher(airport, entry, sample)
      .then((parsed) => {
        setLoaded((current) => (current?.key !== key ? current : {
          key, load: parsed.ok ? { status: "ready", overlay: parsed.value } : { status: "invalid", problem: parsed.problem },
        }));
      })
      .catch((error: unknown) => {
        setLoaded((current) => (current?.key !== key ? current : {
          key, load: { status: "invalid", problem: error instanceof Error ? error.message : String(error) },
        }));
      });
  }, [shown, key, entry, sample, airport, fetcher]);

  const load: OverlayLoad<T> = loaded !== null && loaded.key === key ? loaded.load : { status: "idle" };
  return { kind, entries, entry, choose: setChosen, shown, setShown, load };
}

export default function useTrainingOverlays(airport: string | null, sample: TrainingSample | null, flightKey: string | null) {
  const { setTrainingExecutor, setTrainingPrior } = useApp();
  const [manifest, setManifest] = useState<OverlaysManifestState>({ status: "loading" });

  useEffect(() => {
    if (!airport) return;
    let live = true;
    setManifest({ status: "loading" });
    fetchTrainingOverlays(airport)
      .then((parsed) => {
        if (!live) return;
        setManifest(parsed.ok ? { status: "ready", overlays: parsed.value } : { status: "invalid", problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (!live) return;
        // None published is an answer, not an error: most sets have no overlay.
        if (isMissingJsonAsset(error)) setManifest({ status: "absent" });
        else setManifest({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [airport]);

  const executor = useOverlayKind<TrainingExecutorOverlay>("executor-replay", manifest, airport, sample, fetchTrainingExecutorOverlay);
  const prior = useOverlayKind<TrainingPriorOverlay>("prior-prediction", manifest, airport, sample, fetchTrainingPriorOverlay);

  const executorOverlay = executor.shown && executor.load.status === "ready" ? executor.load.overlay : null;
  const priorOverlay = prior.shown && prior.load.status === "ready" ? prior.load.overlay : null;
  useEffect(() => {
    const flight = executorOverlay?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingExecutor(executorOverlay && flight ? { overlay: executorOverlay, flight } : null);
  }, [executorOverlay, flightKey, setTrainingExecutor]);
  useEffect(() => {
    const flight = priorOverlay?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingPrior(priorOverlay && flight ? { overlay: priorOverlay, flight } : null);
  }, [priorOverlay, flightKey, setTrainingPrior]);
  useEffect(() => () => {
    setTrainingExecutor(null);
    setTrainingPrior(null);
  }, [setTrainingExecutor, setTrainingPrior]);

  return { manifest, executor, prior };
}
