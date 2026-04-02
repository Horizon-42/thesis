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

import { useEffect, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

// ── Return type ───────────────────────────────────────────────────────────────
export interface CzmlLoaderState {
  isLoaded: boolean;
  /** IDs of all aircraft entities found in the CZML (excludes "document") */
  flightIds: string[];
  error: string | null;
}

/**
 * Load a CZML trajectory file and drive the Cesium clock from it.
 *
 * @param czmlUrl - path to the CZML file, e.g. "/data/trajectories.czml"
 */
export function useCzmlLoader(czmlUrl: string): CzmlLoaderState {
  const { viewer, layers, setSelectedFlightId } = useApp();
  const [state, setState] = useState<CzmlLoaderState>({
    isLoaded: false,
    flightIds: [],
    error: null,
  });

  useEffect(() => {
    if (!viewer) return;

    // We need to hold a reference to the DataSource so we can clean it up.
    let dataSource: Cesium.CzmlDataSource | undefined;

    // ── Step 1: Load the CZML file ────────────────────────────────────────────
    // TODO ① — Call `Cesium.CzmlDataSource.load(czmlUrl)`.
    //   This is a STATIC method that returns a Promise<CzmlDataSource>.
    //   Chain `.then(ds => { ... })` for the happy path and
    //   `.catch(err => { ... })` for error handling.
    //
    // In the .catch handler, call:
    //   setState({ isLoaded: false, flightIds: [], error: err.message });
    //
    // Reference: docs/04-czml-loader.md § "Loading CZML"

    // ── Inside .then(ds => { ... }): ─────────────────────────────────────────

    // TODO ② — Store the DataSource and add it to the viewer:
    //   dataSource = ds;
    //   viewer.dataSources.add(ds);

    // TODO ③ — Synchronise the viewer clock with the CZML time window.
    //
    //   The CZML "document" packet embeds a clock interval.  After loading,
    //   `ds.clock` exposes startTime and stopTime as Cesium.JulianDate objects.
    //   You must copy these into `viewer.clock` so the timeline bar is correct.
    //
    //   Required assignments:
    //     viewer.clock.startTime   = ds.clock.startTime.clone();
    //     viewer.clock.stopTime    = ds.clock.stopTime.clone();
    //     viewer.clock.currentTime = ds.clock.startTime.clone();
    //     viewer.clock.clockRange  = Cesium.ClockRange.LOOP_STOP;
    //     viewer.clock.multiplier  = 60;   // 60× speed (1 real sec = 1 sim min)
    //     viewer.clock.shouldAnimate = true;
    //
    //   Then tell the timeline UI to zoom to match:
    //     viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
    //
    // Hint: `ds.clock` is a DataSourceClock, NOT a Cesium.Clock — they have
    // the same shape but are different types.  Always `.clone()` JulianDate
    // values before assigning them to avoid aliasing bugs.

    // TODO ④ — Collect entity IDs (skip "document"):
    //   const ids = ds.entities.values
    //     .filter(e => e.id !== "document")
    //     .map(e => e.id);

    // TODO ⑤ — Track the first aircraft with the camera (optional but cool):
    //   if (ids.length > 0) {
    //     viewer.trackedEntity = ds.entities.getById(ids[0]) ?? undefined;
    //     setSelectedFlightId(ids[0]);
    //   }

    // TODO ⑥ — Update state:
    //   setState({ isLoaded: true, flightIds: ids, error: null });

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      if (dataSource) {
        viewer.dataSources.remove(dataSource, true);
        viewer.trackedEntity = undefined;
      }
    };
  }, [viewer, czmlUrl, setSelectedFlightId]);

  // ── Sync visibility ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!viewer) return;
    const ds = viewer.dataSources.getByName("czml-trajectories")[0];
    if (ds) ds.show = layers.trajectories;
  }, [viewer, layers.trajectories]);

  return state;
}
