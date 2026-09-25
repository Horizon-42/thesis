/**
 * useTrainingOverlays.ts
 * ----------------------
 * The Training panel's second fetch: `training/overlays.json`, and the overlays drawn over the set it has open —
 * the executor's replay and the prior's predictions (`data/trainingOverlays.ts`). Each kind has its own switch (on at
 * first), its overlay is downloaded the first time it is switched on for a set, and while it is on the panel's
 * selected flight is published (`trainingExecutor` / `trainingPrior`) for the sentence bar, the windows and 3D.
 *
 * THE PRIOR'S OWN SENTENCES (`prior-generation`) have no switch: every one published for the set is downloaded — each
 * model is a tab of the sentence bar, which is what chooses the sentence read (`trainingSource`) — and the selected
 * flight's view of each loaded one is published together (`trainingGenerations`).
 *
 * The states all name what failed, as the panel's own do: no manifest is "none published" (the command that writes
 * one is shown), a manifest that is not one or an overlay that does not bind to the set says why, and only that
 * overlay is dropped.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingExecutorOverlay,
  fetchTrainingGenerationOverlay,
  fetchTrainingOverlays,
  fetchTrainingPriorOverlay,
  trainingOverlaysOf,
  type TrainingExecutorOverlay,
  type TrainingGenerationOverlay,
  type TrainingGenerationView,
  type TrainingOverlayEntry,
  type TrainingOverlayKind,
  type TrainingOverlays,
  type TrainingPriorOverlay,
} from "../data/trainingOverlays";
import type { Parsed } from "../data/trainingReader";
import type { TrainingSample } from "../data/trainingSample";

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
  // the overlay chosen among several belongs to the set it was chosen over: another set or airport starts at the latest
  const scope = sample && airport ? `${airport}/${sample.setId}` : null;
  const [chosen, setChosen] = useState<{ scope: string | null; id: string } | null>(null);
  const choose = useCallback((id: string) => setChosen({ scope, id }), [scope]);
  const chosenId = chosen !== null && chosen.scope === scope ? chosen.id : null;
  const entry = entries.find((item) => item.id === chosenId) ?? entries[entries.length - 1] ?? null;
  const [shown, setShown] = useState<boolean>(true);
  // One download per overlay and sample: switching off and on again reuses it — unless it failed, which switching off and
  // on again tries anew (a file rewritten since is read). Only the latest download's answer is kept: one started before it,
  // for this overlay or another, answers into nothing, even when it is the same file asked for again.
  const [loaded, setLoaded] = useState<{ key: string; load: OverlayLoad<T> } | null>(null);
  const key = entry && sample && airport ? `${airport}/${entry.id}@${sample.writtenUtc}` : null;
  const requested = useRef<{ key: string } | null>(null);

  useEffect(() => {
    if (!shown || key === null || !entry || !sample || !airport || requested.current?.key === key) return;
    const request = { key };
    requested.current = request;
    setLoaded({ key, load: { status: "loading" } });
    const settle = (load: OverlayLoad<T>) => {
      if (requested.current !== request) return;
      if (load.status !== "ready") requested.current = null;
      setLoaded({ key, load });
    };
    fetcher(airport, entry, sample)
      .then((parsed) => settle(parsed.ok ? { status: "ready", overlay: parsed.value } : { status: "invalid", problem: parsed.problem }))
      .catch((error: unknown) => settle({ status: "invalid", problem: error instanceof Error ? error.message : String(error) }));
  }, [shown, key, entry, sample, airport, fetcher]);

  const load: OverlayLoad<T> = loaded !== null && loaded.key === key ? loaded.load : { status: "idle" };
  return { kind, entries, entry, choose, shown, setShown, load };
}

/** One generation overlay published for the open set, its download, and — when that failed — a way to ask again. */
export interface GenerationItem {
  entry: TrainingOverlayEntry;
  load: OverlayLoad<TrainingGenerationOverlay>;
  retry: () => void;
}

/** Every generation overlay drawn over the open set, each downloaded once per set and sample; a failed one says why and is
 *  asked again only by its `retry`. Only the open set's are kept: another set's are dropped (they are large). */
function useGenerationOverlays(manifest: OverlaysManifestState, airport: string | null, sample: TrainingSample | null): GenerationItem[] {
  const entries = useMemo(
    () => (manifest.status === "ready" && sample && manifest.overlays.airport === airport && sample.airport === airport
      ? trainingOverlaysOf(manifest.overlays, sample.setId, "prior-generation") : []),
    [manifest, sample, airport],
  );
  const [loads, setLoads] = useState<Record<string, OverlayLoad<TrainingGenerationOverlay>>>({});
  const requested = useRef<Set<string>>(new Set());
  const [attempt, setAttempt] = useState<number>(0);
  const keyOf = useCallback((entry: TrainingOverlayEntry) => `${airport}/${entry.id}@${sample?.writtenUtc}`, [airport, sample]);

  useEffect(() => {
    if (!sample || !airport) return;
    // only the open set's downloads are kept, and only theirs are asked for: another set's answer lands in nothing
    const current = new Set(entries.map(keyOf));
    requested.current = new Set([...requested.current].filter((key) => current.has(key)));
    setLoads((held) => {
      const kept = Object.fromEntries(Object.entries(held).filter(([key]) => current.has(key)));
      return Object.keys(kept).length === Object.keys(held).length ? held : kept;
    });
    const settle = (key: string, load: OverlayLoad<TrainingGenerationOverlay>) =>
      setLoads((held) => (requested.current.has(key) ? { ...held, [key]: load } : held));
    for (const entry of entries) {
      const key = keyOf(entry);
      if (requested.current.has(key)) continue;
      requested.current.add(key);
      setLoads((current) => ({ ...current, [key]: { status: "loading" } }));
      fetchTrainingGenerationOverlay(airport, entry, sample)
        .then((parsed) => settle(key, parsed.ok ? { status: "ready", overlay: parsed.value } : { status: "invalid", problem: parsed.problem }))
        .catch((error: unknown) => settle(key, { status: "invalid", problem: error instanceof Error ? error.message : String(error) }));
    }
  }, [entries, sample, airport, keyOf, attempt]);

  // not asked yet: the effect above asks for it after this render
  return useMemo(() => entries.map((entry) => ({
    entry,
    load: loads[keyOf(entry)] ?? { status: "idle" },
    retry: () => {
      requested.current.delete(keyOf(entry));
      setAttempt((count) => count + 1);
    },
  })), [entries, loads, keyOf]);
}

export default function useTrainingOverlays(airport: string | null, sample: TrainingSample | null, flightKey: string | null) {
  const { setTrainingExecutor, setTrainingPrior, setTrainingGenerations } = useApp();
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
  const generations = useGenerationOverlays(manifest, airport, sample);

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
  const generationViews = useMemo(() => generations.flatMap(({ load }): TrainingGenerationView[] => {
    if (load.status !== "ready") return [];
    const flight = load.overlay.flights.find((item) => item.flightKey === flightKey);
    return flight ? [{ overlay: load.overlay, flight }] : [];
  }), [generations, flightKey]);
  useEffect(() => {
    setTrainingGenerations(generationViews);
  }, [generationViews, setTrainingGenerations]);
  useEffect(() => () => {
    setTrainingExecutor(null);
    setTrainingPrior(null);
    setTrainingGenerations([]);
  }, [setTrainingExecutor, setTrainingPrior, setTrainingGenerations]);

  return { manifest, executor, prior, generations };
}
