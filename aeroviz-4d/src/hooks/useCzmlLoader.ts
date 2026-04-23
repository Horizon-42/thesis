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
 * @param czmlUrl - path to the CZML file, e.g. "/data/airports/KRDU/trajectories.czml"
 */
export function useCzmlLoader(czmlUrl: string): CzmlLoaderState {
  const { viewer, layers, setSelectedFlightId } = useApp();
  // Hold a direct reference — the CZML document packet can overwrite the
  // datasource name, making getByName() unreliable for visibility sync.
  const dsRef = useRef<Cesium.CzmlDataSource | null>(null);
  const [state, setState] = useState<CzmlLoaderState>({
    isLoaded: false,
    flightIds: [],
    warning: null,
    error: null,
  });

  useEffect(() => {
    if (!viewer || !czmlUrl) {
      setState({ isLoaded: false, flightIds: [], warning: null, error: null });
      return;
    }

    // We need to hold a reference to the DataSource so we can clean it up.
    let dataSource: Cesium.CzmlDataSource | undefined;
    let cancelled = false;

    setState({ isLoaded: false, flightIds: [], warning: null, error: null });

    // ── Step 1: Preflight the CZML URL so missing files don't parse index.html.
    const ds = new Cesium.CzmlDataSource(LAYER_NAME);
    fetchJson<unknown>(czmlUrl)
      .then(() => ds.load(czmlUrl))
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
          setState({ isLoaded: true, flightIds: [], warning, error: null });
          return;
        }

        dataSource = loadedDs;
        dsRef.current = loadedDs;
        viewer.dataSources.add(loadedDs);
        loadedDs.show = layers.trajectories;

        let warning: string | null = null;
        if (loadedDs.clock) {
          const startTime = loadedDs.clock.startTime.clone();
          const stopTime = loadedDs.clock.stopTime.clone();

          if (Cesium.JulianDate.lessThan(startTime, stopTime)) {
            viewer.clock.startTime = startTime;
            viewer.clock.stopTime = stopTime;
            viewer.clock.currentTime = startTime.clone();
            viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
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
          setState({ isLoaded: true, flightIds: [], warning, error: null });
          return;
        }

        const message = err instanceof Error ? err.message : String(err);
        setState({ isLoaded: false, flightIds: [], warning: null, error: message });
      });

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      cancelled = true;
      dsRef.current = null;
      if (dataSource) {
        viewer.dataSources.remove(dataSource, true);
        viewer.trackedEntity = undefined;
      }
    };
  }, [viewer, czmlUrl, setSelectedFlightId]);

  // ── Sync visibility ───────────────────────────────────────────────────────
  useEffect(() => {
    if (dsRef.current) dsRef.current.show = layers.trajectories;
  }, [layers.trajectories, state.isLoaded]);

  return state;
}
