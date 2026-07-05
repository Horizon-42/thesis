/**
 * useDynamicsComparisonPlayback.ts
 * --------------------------------
 * Plays the backend-generated dynamics-comparison CZML on Cesium's own clock.
 * Unlike the single-aircraft optimized playback, this CZML holds one entity per
 * dynamics system (A/B/C/D, plus N = the geodetic RHS in the optimizer's
 * normalized coords, which overlays C), each a colored growing path. The hook:
 *   1. loads the CZML into a CzmlDataSource and drives viewer.clock from its
 *      document clock (start/stop/multiplier), and
 *   2. shows/hides each system's entity from `hiddenKeys`, so the user can hide
 *      any trajectory.
 *
 * Colors, paths and labels all live in the CZML; the hook only manages its
 * lifecycle, the clock, and per-system visibility.
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import { makeReadoutEmitter, sampleTrajectoryAt } from "./useOptimizedTrajectoryPlayback";
import type { TrajectorySample } from "../pilot/trajectoryOptimizationClient";
import {
  interpolateComparisonDeltas,
  type DynamicsComparisonChart,
  type DynamicsComparisonDeltas,
} from "../pilot/dynamicsComparisonClient";

const DATASOURCE_NAME = "dynamics-comparison";
const ENTITY_PREFIX = "dyncmp-";

/** The system key (e.g. "B") for a dyncmp entity, or null for any other entity. */
function systemKeyOf(entity: Cesium.Entity): string | null {
  return entity.id.startsWith(ENTITY_PREFIX) ? entity.id.slice(ENTITY_PREFIX.length) : null;
}

export type DynamicsComparisonPlaybackStatus = "idle" | "loading" | "loaded" | "error";

interface UseDynamicsComparisonPlaybackParams {
  /** When false, any loaded playback is removed. */
  enabled: boolean;
  /** Parsed CZML packet array from the comparison response, or null. */
  czml: unknown[] | null;
  /** System keys (e.g. "A") whose entity should be hidden. */
  hiddenKeys: readonly string[];
  /** Camera-follow the reference (B) trajectory. */
  follow: boolean;
  /** Dense reference-B samples used to drive the Live-State readout. */
  samples: TrajectorySample[];
  /** Receives the reference-B state at the current clock time (null when idle). */
  onSample: (sample: TrajectorySample | null) => void;
  /** Error series (A/C/D vs B) used to overlay live deviations on the readout. */
  chart: DynamicsComparisonChart | null;
  /** Receives the A/C/D deviations at the current clock time (null when idle). */
  onDeltas: (deltas: DynamicsComparisonDeltas | null) => void;
}

interface UseDynamicsComparisonPlaybackResult {
  status: DynamicsComparisonPlaybackStatus;
  error: string | null;
}

export function useDynamicsComparisonPlayback({
  enabled,
  czml,
  hiddenKeys,
  follow,
  samples,
  onSample,
  chart,
  onDeltas,
}: UseDynamicsComparisonPlaybackParams): UseDynamicsComparisonPlaybackResult {
  const { viewer, setPlaybackSpeed, autoReplay } = useApp();
  const dsRef = useRef<Cesium.CzmlDataSource | null>(null);
  const startTimeRef = useRef<Cesium.JulianDate | null>(null);
  const [status, setStatus] = useState<DynamicsComparisonPlaybackStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  // Keep the latest samples / callback reachable from the clock tick without
  // re-subscribing the listener every render.
  const samplesRef = useRef(samples);
  samplesRef.current = samples;
  const onSampleRef = useRef(onSample);
  onSampleRef.current = onSample;
  const chartRef = useRef(chart);
  chartRef.current = chart;
  const onDeltasRef = useRef(onDeltas);
  onDeltasRef.current = onDeltas;
  // Read the live follow state from inside the async load without re-running
  // effect 1 when it flips.
  const followRef = useRef(follow);
  followRef.current = follow;

  // Join so the visibility effect re-runs only when the hidden set changes.
  const hiddenKey = [...hiddenKeys].sort().join(",");

  // ── Effect 1: load / unload the CZML and sync the clock ──────────────────────
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || !enabled || !czml) {
      setStatus("idle");
      setError(null);
      return;
    }

    let cancelled = false;
    let dataSource: Cesium.CzmlDataSource | undefined;
    setStatus("loading");
    setError(null);

    const ds = new Cesium.CzmlDataSource(DATASOURCE_NAME);
    ds.load(czml as Cesium.CzmlDataSource.LoadOptions | object)
      .then((loadedDs) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        dataSource = loadedDs;
        dsRef.current = loadedDs;
        viewer.dataSources.add(loadedDs);

        // Orient each system's aircraft model along its own path (the CZML only
        // carries position; velocity orientation points the nose down-track).
        // Wrap it so the final attitude holds the last in-flight orientation: at
        // the end the position extrapolation is HOLD, so the velocity → 0 and a
        // plain VelocityOrientationProperty returns undefined and snaps the model
        // to a default heading — not the final state's heading/pitch.
        loadedDs.entities.values.forEach((entity) => {
          if (systemKeyOf(entity) !== null && entity.position) {
            entity.orientation = makeStableVelocityOrientation(entity.position);
          }
        });

        if (loadedDs.clock) {
          const start = loadedDs.clock.startTime.clone();
          const stop = loadedDs.clock.stopTime.clone();
          if (Cesium.JulianDate.lessThan(start, stop)) {
            const multiplier = loadedDs.clock.multiplier || 1;
            startTimeRef.current = start.clone();
            viewer.clock.startTime = start;
            viewer.clock.stopTime = stop;
            viewer.clock.currentTime = start.clone();
            viewer.clock.clockRange = autoReplay
              ? Cesium.ClockRange.LOOP_STOP
              : Cesium.ClockRange.CLAMPED;
            viewer.clock.multiplier = multiplier;
            // Load PAUSED — the CZML is loaded whenever you're in Compare with a
            // result (so the shared clock transport is bound to it), but it animates
            // only when the user presses Play (panel button or bottom bar/native dial).
            viewer.clock.shouldAnimate = false;
            viewer.timeline?.zoomTo(start, stop);
            setPlaybackSpeed(multiplier);
          }
        }

        // Frame the whole comparison in view on load (reference B spans the full
        // start→target extent that A/C/D share). Skip when following: effect 4
        // takes over the camera in that case.
        if (!followRef.current) {
          frameTrajectoryCamera(viewer, samplesRef.current);
        }

        setStatus("loaded");
      })
      .catch((loadError: unknown) => {
        if (cancelled) return;
        setStatus("error");
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      });

    return () => {
      cancelled = true;
      dsRef.current = null;
      startTimeRef.current = null;
      onSampleRef.current(null);
      onDeltasRef.current(null);
      if (dataSource && isCesiumViewerUsable(viewer)) {
        if (viewer.trackedEntity && dataSource.entities.contains(viewer.trackedEntity)) {
          viewer.trackedEntity = undefined;
        }
        viewer.dataSources.remove(dataSource, true);
      }
      setStatus("idle");
    };
  }, [viewer, enabled, czml]);

  // ── Effect 2: keep clockRange in sync with the auto-replay toggle ─────────────
  // (Effect 1 only runs on load, so a later autoReplay flip must be applied here.)
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || status !== "loaded") return;
    viewer.clock.clockRange = autoReplay
      ? Cesium.ClockRange.LOOP_STOP
      : Cesium.ClockRange.CLAMPED;
  }, [viewer, status, autoReplay]);

  // ── Effect 3: per-system visibility ──────────────────────────────────────────
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || status !== "loaded") return;
    const ds = dsRef.current;
    if (!ds) return;
    const hidden = new Set(hiddenKey ? hiddenKey.split(",") : []);
    ds.entities.values.forEach((entity) => {
      const key = systemKeyOf(entity);
      if (key !== null) entity.show = !hidden.has(key);
    });
  }, [viewer, status, hiddenKey]);

  // ── Effect 4: camera follow ──────────────────────────────────────────────────
  // Follow the reference B; if B is hidden, fall back to the first visible system
  // so the camera never locks onto a hidden (invisible) entity.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || status !== "loaded") return;
    const ds = dsRef.current;
    if (!ds) return;
    if (!follow) {
      viewer.trackedEntity = undefined;
      return;
    }
    const hidden = new Set(hiddenKey ? hiddenKey.split(",") : []);
    const visible = ds.entities.values.filter((entity) => {
      const key = systemKeyOf(entity);
      return key !== null && !hidden.has(key);
    });
    const referenceFirst =
      visible.find((entity) => systemKeyOf(entity) === "B") ?? visible[0];
    viewer.trackedEntity = referenceFirst ?? undefined;
  }, [viewer, status, follow, hiddenKey]);

  // ── Effect 5: sample B + the A/C/D deviations on each clock tick ──────────────
  // The readout shows B's state (from the dense samples) and overlays each other
  // system's live error (interpolated from the chart's error series at the same
  // clock time), so both come from one throttled tick.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || status !== "loaded") return;

    const emitter = makeReadoutEmitter(viewer, startTimeRef, (tSimS) => {
      onSampleRef.current(sampleTrajectoryAt(samplesRef.current, tSimS));
      const chartNow = chartRef.current;
      onDeltasRef.current(chartNow ? interpolateComparisonDeltas(chartNow, tSimS) : null);
    });

    emitter.tick();
    const removeTick = viewer.clock.onTick.addEventListener(emitter.tick);
    const removeStop = viewer.clock.onStop.addEventListener(emitter.stop);
    return () => {
      removeTick();
      removeStop();
    };
  }, [viewer, status]);

  return { status, error };
}
