/**
 * useOptimizedTrajectoryPlayback.ts
 * ---------------------------------
 * Plays a backend-generated optimized-trajectory CZML on Cesium's own clock —
 * exactly like a downloaded trajectory — instead of re-simulating with a fixed
 * dt on the frontend.
 *
 * Responsibilities:
 *   1. Load the optimized CZML (a parsed packet array, not a URL) into a
 *      CzmlDataSource and add it to the viewer.
 *   2. Drive viewer.clock from the CZML document clock (start/stop/multiplier),
 *      so the existing timeline + ControlPanel transport control playback.
 *   3. Optionally track (camera-follow) the optimized aircraft entity.
 *   4. On every clock tick, sample the dense rollout at the current clock time
 *      and report it (throttled) so the live numeric readout stays in sync
 *      without the frontend stepping the dynamics.
 *
 * The colored per-segment polylines and the aircraft model live entirely in the
 * CZML; this hook only manages its lifecycle, the clock, and readout sampling.
 */

import { useEffect, useRef, useState, type MutableRefObject } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import type { TrajectorySample } from "../pilot/trajectoryOptimizationClient";

const OPTIMIZED_DATASOURCE_NAME = "optimized-trajectory";
const OPTIMIZED_AIRCRAFT_ID = "optimized-trajectory-aircraft";
/** Min wall-clock gap between readout updates, to cap re-renders at ~12 Hz. */
const READOUT_THROTTLE_MS = 80;

export type OptimizedTrajectoryPlaybackStatus = "idle" | "loading" | "loaded" | "error";

interface UseOptimizedTrajectoryPlaybackParams {
  /** When false, any loaded playback is removed and the readout is cleared. */
  enabled: boolean;
  /** Parsed CZML packet array from the optimizer response, or null. */
  czml: unknown[] | null;
  /** Dense rollout samples used to drive the live readout. */
  samples: TrajectorySample[];
  /** Camera-follow the optimized aircraft. */
  follow: boolean;
  /** Receives the sampled state at the current clock time (null when idle). */
  onSample: (sample: TrajectorySample | null) => void;
}

interface UseOptimizedTrajectoryPlaybackResult {
  status: OptimizedTrajectoryPlaybackStatus;
  error: string | null;
}

export function useOptimizedTrajectoryPlayback({
  enabled,
  czml,
  samples,
  follow,
  onSample,
}: UseOptimizedTrajectoryPlaybackParams): UseOptimizedTrajectoryPlaybackResult {
  const { viewer, setPlaybackSpeed, playbackSpeed, autoReplay, setOptimizedTrajectoryDataSource } =
    useApp();
  const dsRef = useRef<Cesium.CzmlDataSource | null>(null);
  const aircraftRef = useRef<Cesium.Entity | null>(null);
  const startTimeRef = useRef<Cesium.JulianDate | null>(null);
  // Remember the playback rate the user last used for Trajectory Play, so a new
  // run keeps it instead of snapping back to the CZML doc multiplier (1×). null
  // until the first run, where we fall back to the doc multiplier.
  const lastSpeedRef = useRef<number | null>(null);
  const [status, setStatus] = useState<OptimizedTrajectoryPlaybackStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  // Keep the latest samples / callback reachable from the clock tick without
  // re-subscribing the listener every render.
  const samplesRef = useRef(samples);
  samplesRef.current = samples;
  const onSampleRef = useRef(onSample);
  onSampleRef.current = onSample;
  // Read the live follow state from inside the async load without re-running
  // effect 1 when it flips.
  const followRef = useRef(follow);
  followRef.current = follow;

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

    const ds = new Cesium.CzmlDataSource(OPTIMIZED_DATASOURCE_NAME);
    ds.load(czml as Cesium.CzmlDataSource.LoadOptions | object)
      .then((loadedDs) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        dataSource = loadedDs;
        dsRef.current = loadedDs;
        aircraftRef.current = loadedDs.entities.getById(OPTIMIZED_AIRCRAFT_ID) ?? null;
        viewer.dataSources.add(loadedDs);
        // Expose it so the runway profile can plot the optimized track.
        setOptimizedTrajectoryDataSource(loadedDs);

        if (loadedDs.clock) {
          const start = loadedDs.clock.startTime.clone();
          const stop = loadedDs.clock.stopTime.clone();
          if (Cesium.JulianDate.lessThan(start, stop)) {
            // Keep the user's last Trajectory-Play rate across runs; fall back to
            // the CZML doc multiplier (1×, not the default 60×) on the first run.
            const multiplier = lastSpeedRef.current ?? (loadedDs.clock.multiplier || 1);
            lastSpeedRef.current = multiplier;
            startTimeRef.current = start.clone();
            viewer.clock.startTime = start;
            viewer.clock.stopTime = stop;
            viewer.clock.currentTime = start.clone();
            viewer.clock.clockRange = autoReplay
              ? Cesium.ClockRange.LOOP_STOP
              : Cesium.ClockRange.CLAMPED;
            viewer.clock.multiplier = multiplier;
            viewer.clock.shouldAnimate = true;
            viewer.timeline?.zoomTo(start, stop);
            // Keep the ControlPanel's displayed speed in step with the clock.
            setPlaybackSpeed(multiplier);
          }
        }

        attachOrientation(aircraftRef.current, startTimeRef, samplesRef);

        // Frame the whole trajectory in view on load. Skip when following: the
        // camera-follow effect below takes over the camera in that case.
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
      aircraftRef.current = null;
      startTimeRef.current = null;
      onSampleRef.current(null);
      setOptimizedTrajectoryDataSource(null);
      if (dataSource && isCesiumViewerUsable(viewer)) {
        if (viewer.trackedEntity === dataSource.entities.getById(OPTIMIZED_AIRCRAFT_ID)) {
          viewer.trackedEntity = undefined;
        }
        viewer.dataSources.remove(dataSource, true);
      }
      setStatus("idle");
    };
  }, [viewer, enabled, czml]);

  // ── Effect 1b: remember the user's live rate changes during playback ─────────
  // While a Trajectory Play is loaded, mirror any speed the user picks (via the
  // ControlPanel) into lastSpeedRef, so the NEXT run resumes at that rate. Gated
  // on "loaded" so speeds set in other modes (e.g. a downloaded trajectory at
  // 60×) don't leak into the Trajectory-Play default.
  useEffect(() => {
    if (status === "loaded") lastSpeedRef.current = playbackSpeed;
  }, [playbackSpeed, status]);

  // ── Effect 2: camera follow ──────────────────────────────────────────────────
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    if (status !== "loaded") return;
    viewer.trackedEntity = follow ? aircraftRef.current ?? undefined : undefined;
  }, [viewer, follow, status]);

  // ── Effect 3: sample the rollout on each clock tick for the live readout ─────
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || status !== "loaded") return;

    let lastEmitMs = 0;
    const emit = () => {
      const start = startTimeRef.current;
      if (!start) return;
      const now = performance.now();
      if (now - lastEmitMs < READOUT_THROTTLE_MS) return;
      lastEmitMs = now;
      const elapsedS = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start);
      onSampleRef.current(sampleTrajectoryAt(samplesRef.current, elapsedS));
    };

    emit();
    const remove = viewer.clock.onTick.addEventListener(emit);
    return () => {
      remove();
    };
  }, [viewer, status]);

  return { status, error };
}

/**
 * Linearly interpolate the trajectory at time `t` (seconds from start).
 * Continuous state (position, speed, flight path, aero) is interpolated;
 * piecewise-constant fields (segment index, controls) are taken from the
 * sample at or before `t`.
 */
export function sampleTrajectoryAt(
  samples: TrajectorySample[],
  t: number,
): TrajectorySample | null {
  if (samples.length === 0) return null;
  if (t <= samples[0].t) return samples[0];
  const last = samples[samples.length - 1];
  if (t >= last.t) return last;

  // Binary search for the interval [lo, lo+1] containing t.
  let low = 0;
  let high = samples.length - 1;
  while (high - low > 1) {
    const mid = (low + high) >> 1;
    if (samples[mid].t <= t) low = mid;
    else high = mid;
  }

  const a = samples[low];
  const b = samples[high];
  const span = b.t - a.t;
  const f = span > 1e-9 ? (t - a.t) / span : 0;
  const lerp = (x: number, y: number) => x + (y - x) * f;

  return {
    t,
    lon: lerp(a.lon, b.lon),
    lat: lerp(a.lat, b.lat),
    altM: lerp(a.altM, b.altM),
    speedMps: lerp(a.speedMps, b.speedMps),
    headingDeg: lerpAngleDeg(a.headingDeg, b.headingDeg, f),
    flightPathDeg: lerp(a.flightPathDeg, b.flightPathDeg),
    bankDeg: a.bankDeg,
    thrustN: a.thrustN,
    segmentIndex: a.segmentIndex,
    liftCoefficient: lerp(a.liftCoefficient, b.liftCoefficient),
    dragCoefficient: lerp(a.dragCoefficient, b.dragCoefficient),
    actualLoadFactor: lerp(a.actualLoadFactor, b.actualLoadFactor),
    ...(a.loadFactor === undefined ? {} : { loadFactor: a.loadFactor }),
    ...(a.attackDeg === undefined ? {} : { attackDeg: a.attackDeg }),
  };
}

/** Shortest-arc interpolation between two headings in degrees. */
function lerpAngleDeg(a: number, b: number, f: number): number {
  const delta = ((b - a + 540) % 360) - 180;
  return (((a + delta * f) % 360) + 360) % 360;
}

/**
 * Drive the optimized aircraft's orientation from the sampled state, matching
 * the live Pilot aircraft exactly (heading/pitch/bank, with the same axis
 * conventions). Replaces any baked CZML orientation so the model banks and
 * points along the trajectory instead of using the math-frame heading.
 */
function attachOrientation(
  aircraft: Cesium.Entity | null,
  startTimeRef: MutableRefObject<Cesium.JulianDate | null>,
  samplesRef: MutableRefObject<TrajectorySample[]>,
): void {
  if (!aircraft) return;
  const positionProperty = aircraft.position;
  if (!positionProperty) return;

  aircraft.orientation = new Cesium.CallbackProperty((time) => {
    const start = startTimeRef.current;
    if (!start || !time) return undefined;
    const position = positionProperty.getValue(time);
    if (!position) return undefined;
    const elapsedS = Cesium.JulianDate.secondsDifference(time, start);
    const sample = sampleTrajectoryAt(samplesRef.current, elapsedS);
    if (!sample) return undefined;
    const hpr = new Cesium.HeadingPitchRoll(
      Cesium.Math.toRadians(-sample.headingDeg),
      Cesium.Math.toRadians(sample.flightPathDeg + (sample.attackDeg ?? 0)),
      Cesium.Math.toRadians(-sample.bankDeg),
    );
    return Cesium.Transforms.headingPitchRollQuaternion(position, hpr);
  }, false);
}
