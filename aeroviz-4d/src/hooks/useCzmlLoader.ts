/**
 * useCzmlLoader.ts
 * ----------------
 * Custom hook: load a CZML file and synchronise the Cesium Clock so
 * the timeline bar spans the exact time window of the trajectory data.
 *
 * What this hook does:
 *   1. Loads the CZML file into a CzmlDataSource.
 *   2. Reads the clock interval embedded in the CZML "document" packet.
 *   3. Writes those times into viewer.clock so the timeline bar shows the
 *      correct start/end, and animation begins from the start.
 *   4. Optionally tracks (camera follows) the first aircraft entity.
 *   5. Returns loading state so the UI can show a spinner or error message.
 *
 * Key Cesium concepts:
 *   • CzmlDataSource  — loads and drives animated entities from CZML.
 *   • viewer.clock    — the master simulation clock (a Cesium.Clock instance).
 *   • JulianDate      — Cesium's internal time representation (Julian Day Number).
 *                       Always use Cesium.JulianDate methods; never raw Date math.
 *   • viewer.timeline — the UI bar at the bottom; call `.zoomTo()` to set range.
 *
 * 📖 Tutorial: see docs/04-czml-loader.md
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { sampleSubset } from "../utils/sampleTrajectories";
import { planTrajectoryModels, TRAJECTORY_PATH_WIDTH } from "../utils/trajectoryRenderModel";
import { addDataSourceHidden } from "../utils/cesiumDataSource";

const LAYER_NAME = "czml-trajectories";

// ── Return type ───────────────────────────────────────────────────────────────
export interface CzmlLoaderState {
  isLoaded: boolean;
  /** IDs of all aircraft entities found in the CZML (excludes "document") */
  flightIds: string[];
  /** Non-fatal data issue that should be shown to the user. */
  warning: string | null;
  error: string | null;
}

/**
 * Load a CZML trajectory file and drive the Cesium clock from it.
 *
 * Loading is keyed on `czmlUrl` ONLY — the source is loaded once per file and released when the
 * url changes (airport/runway) or empties. `visible` toggles the layer's visibility WITHOUT
 * reloading, so callers can hide the (large) observed layer while the comparison view is on and
 * bring it back instantly, instead of re-parsing 100 MB on every toggle.
 *
 * @param czmlUrl - path to the CZML file, e.g. "/data/airports/KRDU/trajectories.czml" ("" = none)
 * @param visible - whether the layer should be shown (still gated by the Trajectories toggle)
 */
export function useCzmlLoader(czmlUrl: string, visible: boolean = true): CzmlLoaderState {
  const {
    viewer,
    layers,
    autoReplay,
    trajectorySampleCount,
    selectedFlightId,
    setSelectedFlightId,
    setTrajectoryDataSource,
  } = useApp();
  // Hold a direct reference — the CZML document packet can overwrite the
  // datasource name, making getByName() unreliable for visibility sync.
  const dsRef = useRef<Cesium.CzmlDataSource | null>(null);
  // The shown sample + the (random) subset that carries an aircraft model. Held in
  // refs so re-selecting a flight re-applies the render model WITHOUT reshuffling the
  // subset (the random pick only recomputes when the data / sample count changes).
  const shownIdsRef = useRef<Set<string>>(new Set());
  const modelIdsRef = useRef<Set<string>>(new Set());
  const [state, setState] = useState<CzmlLoaderState>({
    isLoaded: false,
    flightIds: [],
    warning: null,
    error: null,
  });

  useEffect(() => {
    if (!viewer || !czmlUrl) {
      setTrajectoryDataSource(null);
      setState({ isLoaded: false, flightIds: [], warning: null, error: null });
      return;
    }

    // We need to hold a reference to the DataSource so we can clean it up.
    let dataSource: Cesium.CzmlDataSource | undefined;
    let cancelled = false;

    setState({ isLoaded: false, flightIds: [], warning: null, error: null });
    setTrajectoryDataSource(null);

    // ── Step 1: Fetch once, then hand the parsed packet array to Cesium.
    // This preserves the missing-asset guard without downloading the CZML twice.
    const ds = new Cesium.CzmlDataSource(LAYER_NAME);
    fetchJson<unknown>(czmlUrl)
      .then((czml) => ds.load(czml))
      .then((loadedDs) => {
        if (cancelled) return;

        // ── Inside .then(ds => { ... }): ─────────────────────────────────────────

        const ids = loadedDs.entities.values
          .filter((e) => e.id !== "document")
          .map((e) => e.id);

        if (ids.length === 0) {
          const warning =
            `No trajectory entities were found in ${czmlUrl}. ` +
            "The globe will stay open, but playback is disabled until CZML data is generated.";

          console.warn(`[useCzmlLoader] ${warning}`);
          viewer.trackedEntity = undefined;
          setSelectedFlightId(null);
          setTrajectoryDataSource(null);
          setState({ isLoaded: true, flightIds: [], warning, error: null });
          return;
        }

        dataSource = loadedDs;
        dsRef.current = loadedDs;
        // Add hidden: the render-model pass below samples + sets each entity's `show`, and the
        // visibility-sync effect then reveals the source. Adding it visible here would flash
        // EVERY flight for a frame before the sample is applied (see addDataSourceHidden).
        addDataSourceHidden(viewer, loadedDs);
        setTrajectoryDataSource(loadedDs);

        let warning: string | null = null;
        if (loadedDs.clock) {
          const startTime = loadedDs.clock.startTime.clone();
          const stopTime = loadedDs.clock.stopTime.clone();

          if (Cesium.JulianDate.lessThan(startTime, stopTime)) {
            viewer.clock.startTime = startTime;
            viewer.clock.stopTime = stopTime;
            viewer.clock.currentTime = startTime.clone();
            viewer.clock.clockRange = autoReplay
              ? Cesium.ClockRange.LOOP_STOP
              : Cesium.ClockRange.CLAMPED;
            viewer.clock.multiplier = 60;
            viewer.clock.shouldAnimate = true;
            viewer.timeline?.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
          } else {
            warning =
              `The CZML clock interval in ${czmlUrl} has no duration. ` +
              "Trajectory entities were loaded, but playback timing was not changed.";
            console.warn(`[useCzmlLoader] ${warning}`);
          }
        }

        // Keep camera fixed at the airport by default; tracking starts only
        // when the user clicks a flight row in FlightTable.
        viewer.trackedEntity = undefined;
        setSelectedFlightId(null);

        setState({ isLoaded: true, flightIds: ids, warning, error: null });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (isMissingJsonAsset(err)) {
          const warning =
            `${czmlUrl} was not found. ` +
            "The globe will stay open, but playback is disabled until CZML data is generated.";
          console.warn(`[useCzmlLoader] ${warning}`);
          viewer.trackedEntity = undefined;
          setSelectedFlightId(null);
          setTrajectoryDataSource(null);
          setState({ isLoaded: true, flightIds: [], warning, error: null });
          return;
        }

        const message = err instanceof Error ? err.message : String(err);
        setTrajectoryDataSource(null);
        setState({ isLoaded: false, flightIds: [], warning: null, error: message });
      });

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      cancelled = true;
      dsRef.current = null;
      setTrajectoryDataSource(null);
      if (dataSource && isCesiumViewerUsable(viewer)) {
        viewer.dataSources.remove(dataSource, true);
        viewer.trackedEntity = undefined;
      }
    };
  }, [viewer, czmlUrl, setSelectedFlightId, setTrajectoryDataSource]);

  // ── Sync visibility (toggles the loaded layer's show WITHOUT reloading) ──────
  useEffect(() => {
    if (dsRef.current) dsRef.current.show = visible && layers.trajectories;
  }, [visible, layers.trajectories, state.isLoaded]);

  // ── Recompute the shown sample + the model subset (random) ───────────────────
  // `trajectorySampleCount` flights are shown (0 = all); of those, a capped subset is
  // chosen to carry an aircraft model. Only recomputes when the data / count changes,
  // so selecting a flight (below) never reshuffles which flights are planes.
  useEffect(() => {
    const ds = dsRef.current;
    if (!ds || !state.isLoaded) return;
    const ids = ds.entities.values
      .filter((entity) => entity.id !== "document")
      .map((entity) => entity.id);
    const allShown = !(trajectorySampleCount > 0) || trajectorySampleCount >= ids.length;
    shownIdsRef.current = allShown ? new Set(ids) : new Set(sampleSubset(ids, trajectorySampleCount));
    modelIdsRef.current = planTrajectoryModels([...shownIdsRef.current], null).modelIds;
  }, [trajectorySampleCount, state.isLoaded, state.flightIds]);

  // ── Apply the render model to entities ───────────────────────────────────────
  // Every shown flight draws as a uniform-width path; only the subset (plus the
  // selected flight) carries an aircraft model; glTF animation is off; the label
  // shows for the selected flight alone (1000 labels is the dominant Cesium cost).
  // Re-runs on selection change so the tracked aircraft always shows its model+label.
  useEffect(() => {
    const ds = dsRef.current;
    if (!ds || !state.isLoaded) return;
    for (const entity of ds.entities.values) {
      if (entity.id === "document") continue;
      const shown = shownIdsRef.current.has(entity.id);
      entity.show = shown;
      if (entity.path) entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
      if (entity.model) {
        const hasModel = shown && (modelIdsRef.current.has(entity.id) || entity.id === selectedFlightId);
        entity.model.show = new Cesium.ConstantProperty(hasModel);
        entity.model.runAnimations = new Cesium.ConstantProperty(false);
      }
      if (entity.label) {
        entity.label.show = new Cesium.ConstantProperty(shown && entity.id === selectedFlightId);
      }
    }
  }, [trajectorySampleCount, state.isLoaded, state.flightIds, selectedFlightId]);

  return state;
}
