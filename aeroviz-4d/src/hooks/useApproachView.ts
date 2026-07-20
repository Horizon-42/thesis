import { useEffect, useMemo, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import { buildProcedureProfileProjection } from "../data/procedureProfileProjection";
import { loadProcedureRenderBundleData } from "../data/procedureRenderBundle";
import {
  activeHorizontalPlateRoutes,
  classifyApproachViewSample,
  classifyTrackSamples,
  colorForFlightId,
  sortApproachViewTracksBySelection,
  trackEngagesProcedure,
  type ApproachViewSample,
  type ApproachViewTrack,
  type SampledRunwayPoint,
} from "../data/approachViewAnalysis";
import { planApproachViewSources } from "../data/approachViewSources";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { normalizeRunwayIdent } from "../utils/runwayIdent";
import {
  buildRunwayReferenceMarksFromPlateRoutes,
  buildRunwayFrame,
  projectPositionToRunwayFrame,
  type HorizontalPlateRoute,
  type RunwayReferenceMark,
  type RunwayFeatureCollection,
  type RunwayFrame,
} from "../utils/runwayProfileGeometry";

export type {
  ApproachViewSample,
  ApproachViewTrack,
} from "../data/approachViewAnalysis";

const TICK_THROTTLE_MS = 120;
// The profile plots each aircraft's WHOLE track across the procedure, not a short trailing
// window: we sample outward from the current time in both directions until the entity has no
// position (before its first / after its last sample). The step sets the plotted resolution;
// the per-direction cap is only a runaway backstop — break-on-null stops at the real ends.
const TRACK_SAMPLE_STEP_SECONDS = 5;
const MAX_TRACK_SAMPLES_PER_DIRECTION = 600;

export interface ApproachViewState {
  isLoading: boolean;
  error: string | null;
  currentTimeIso: string | null;
  runwayFrame: RunwayFrame | null;
  plateRoutes: HorizontalPlateRoute[];
  referenceMarks: RunwayReferenceMark[];
  procedureNames: string[];
  sourceCycle: string | null;
  aircraftTracks: ApproachViewTrack[];
  /** Whether a trajectory source belonging to the CURRENT tab is loaded (the
   *  profile's "CZML linked" indicator). Mirrors what the approach view actually plots. */
  sourceLinked: boolean;
}

interface LoadedApproachViewData {
  runwayFrame: RunwayFrame;
  plateRoutes: HorizontalPlateRoute[];
  sourceCycle: string | null;
}

function formatJulianTime(time: Cesium.JulianDate): string {
  return Cesium.JulianDate.toDate(time).toISOString();
}

function sampleRunwayPoint(
  entity: Cesium.Entity,
  time: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
  timeIso: string,
): SampledRunwayPoint | null {
  if (!entity.position) return null;

  const cartesian = entity.position.getValue(time, new Cesium.Cartesian3());
  if (!cartesian) return null;

  const cartographic = Cesium.Cartographic.fromCartesian(
    cartesian,
    Cesium.Ellipsoid.WGS84,
    new Cesium.Cartographic(),
  );
  if (!cartographic) return null;

  const geoPosition = {
    lonDeg: Cesium.Math.toDegrees(cartographic.longitude),
    latDeg: Cesium.Math.toDegrees(cartographic.latitude),
    altM: cartographic.height,
  };
  return {
    ...projectPositionToRunwayFrame(
      runwayFrame,
      geoPosition.lonDeg,
      geoPosition.latDeg,
      geoPosition.altM,
    ),
    geoPosition,
    timeIso,
  };
}

// Two sampled points are the "same position" (within interpolation noise). Used to detect
// the HOLD tail: every aircraft CZML sets forwardExtrapolationType "HOLD" (so the parked
// model stays on the globe), so getValue returns the FROZEN final position for all time past
// the last real sample — a naive walk-until-undefined never terminates there.
function samePosition(a: SampledRunwayPoint, b: SampledRunwayPoint): boolean {
  return (
    Math.abs(a.geoPosition.lonDeg - b.geoPosition.lonDeg) < 1e-9 &&
    Math.abs(a.geoPosition.latDeg - b.geoPosition.latDeg) < 1e-9 &&
    Math.abs(a.geoPosition.altM - b.geoPosition.altM) < 1e-3
  );
}

/**
 * The current-time sample IF the aircraft is LIVE (actually moving) there, else `null`.
 *
 * "Live" matters because of the HOLD extrapolation (see `samePosition`): a landed flight
 * reads as frozen on the threshold forever. An entity whose sample one step back equals
 * `current` is on that held tail (landed/parked, not live); a `null` current means it is not
 * airborne yet (before its first sample). This is the ONLY per-tick work for most entities —
 * one or two cheap `getValue`s — so non-flying traffic is rejected before the expensive walk.
 */
function currentIfLive(
  entity: Cesium.Entity,
  currentTime: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
): SampledRunwayPoint | null {
  const current = sampleRunwayPoint(entity, currentTime, runwayFrame, formatJulianTime(currentTime));
  if (!current) return null;
  const priorTime = Cesium.JulianDate.addSeconds(
    currentTime,
    -TRACK_SAMPLE_STEP_SECONDS,
    new Cesium.JulianDate(),
  );
  const prior = sampleRunwayPoint(entity, priorTime, runwayFrame, formatJulianTime(priorTime));
  if (prior && samePosition(prior, current)) return null; // past its last sample — landed/parked
  return current;
}

/**
 * Sample an entity's WHOLE track (the real approach, both directions from an anchor time),
 * stopping each walk at the track end: `null` (NONE extrapolation before the first sample) or
 * an unchanged position (the HOLD-extrapolated tail past the last real sample). The result is
 * INDEPENDENT of where playback sits — walking out from any in-track anchor yields the same
 * path — so the hook samples it ONCE per entity and caches it, rather than every clock tick.
 */
function sampleWholeTrack(
  entity: Cesium.Entity,
  anchorTime: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
): SampledRunwayPoint[] {
  const anchor = sampleRunwayPoint(entity, anchorTime, runwayFrame, formatJulianTime(anchorTime));
  if (!anchor) return [];

  const walk = (direction: 1 | -1): SampledRunwayPoint[] => {
    const points: SampledRunwayPoint[] = [];
    let previous = anchor;
    for (let step = 1; step <= MAX_TRACK_SAMPLES_PER_DIRECTION; step += 1) {
      const time = Cesium.JulianDate.addSeconds(
        anchorTime,
        direction * step * TRACK_SAMPLE_STEP_SECONDS,
        new Cesium.JulianDate(),
      );
      const point = sampleRunwayPoint(entity, time, runwayFrame, formatJulianTime(time));
      if (!point || samePosition(point, previous)) break;
      points.push(point);
      previous = point;
    }
    return points;
  };

  return [...walk(-1).reverse(), anchor, ...walk(1)];
}

/** A live aircraft's current sample + its whole track, or null if not live. The tested
 *  composition of {@link currentIfLive} + {@link sampleWholeTrack}; the hook uses the two
 *  parts separately so the whole track can be cached across ticks. */
export function sampleEntityTrack(
  entity: Cesium.Entity,
  currentTime: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
): { current: SampledRunwayPoint; trail: SampledRunwayPoint[] } | null {
  const current = currentIfLive(entity, currentTime, runwayFrame);
  if (!current) return null;
  return { current, trail: sampleWholeTrack(entity, currentTime, runwayFrame) };
}

export function useApproachView(): ApproachViewState {
  const {
    viewer,
    mode,
    activeAirportCode,
    procedureVisibility,
    selectedRunway,
    isApproachViewOpen,
    trajectoryComparison,
    trajectoryDataSource,
    optimizedTrajectoryDataSource,
    selectedFlightId,
  } = useApp();
  // Which trajectory sources are the CURRENT tab's globe content — the approach view plots
  // only those, so it mirrors the active task instead of also drawing the observed
  // tracks that stay loaded (but hidden) behind an approach view opened in Optimize/Fly/Compare.
  const sourceSelection = useMemo(
    () =>
      planApproachViewSources({
        mode,
        activeAirportCode,
        selectedRunway,
        trajectoryComparison,
        hasOptimizedSource: optimizedTrajectoryDataSource !== null,
      }),
    [
      mode,
      activeAirportCode,
      selectedRunway,
      trajectoryComparison,
      optimizedTrajectoryDataSource,
    ],
  );
  const sourceLinked =
    (sourceSelection.observed && trajectoryDataSource !== null) ||
    (sourceSelection.optimized && optimizedTrajectoryDataSource !== null);
  // The approach view's runway is the global Landing-Runway selection, in the procedure
  // (RW-prefixed) spelling. `null` when "All runways" is selected — no single profile.
  const approachRunwayIdent = selectedRunway ? normalizeRunwayIdent(selectedRunway) : null;
  const [currentTime, setCurrentTime] = useState<Cesium.JulianDate | null>(null);
  const [loadedData, setLoadedData] = useState<LoadedApproachViewData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Per-entity cache of the classified whole track (keyed by flight id); rebuilt only when the
  // geometry it depends on changes, never per clock tick. See the aircraftTracks memo.
  const trackCacheRef = useRef<{ deps: unknown[]; tracks: Map<string, ApproachViewSample[]> }>({
    deps: [],
    tracks: new Map(),
  });

  useEffect(() => {
    if (!viewer || !isApproachViewOpen) {
      setCurrentTime(null);
      return;
    }

    setCurrentTime(viewer.clock.currentTime.clone());
    let lastUpdateMs = 0;
    const removeListener = viewer.clock.onTick.addEventListener((clock) => {
      const nowMs = typeof performance !== "undefined" ? performance.now() : Date.now();
      if (nowMs - lastUpdateMs < TICK_THROTTLE_MS) return;
      lastUpdateMs = nowMs;
      setCurrentTime(clock.currentTime.clone());
    });

    return () => {
      removeListener();
    };
  }, [viewer, isApproachViewOpen]);

  useEffect(() => {
    if (!activeAirportCode || !approachRunwayIdent || !isApproachViewOpen) {
      setLoadedData(null);
      setIsLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    Promise.all([
      fetchJson<RunwayFeatureCollection>(airportDataUrl(activeAirportCode, "runway.geojson")),
      loadProcedureRenderBundleData(activeAirportCode),
    ])
      .then(([runwayCollection, procedureRenderData]) => {
        if (cancelled) return;

        const runwayFrame = buildRunwayFrame(runwayCollection, approachRunwayIdent);
        const procedureProfileProjection = buildProcedureProfileProjection(
          procedureRenderData,
          runwayFrame,
          approachRunwayIdent,
        );
        setLoadedData({
          runwayFrame,
          plateRoutes: procedureProfileProjection.plateRoutes,
          sourceCycle: procedureProfileProjection.sourceCycle,
        });
        setIsLoading(false);
      })
      .catch((loadError) => {
        if (cancelled) return;
        const message = isMissingJsonAsset(loadError)
          ? `Missing runway or procedure-details data for ${activeAirportCode}`
          : loadError instanceof Error
            ? loadError.message
            : String(loadError);
        setLoadedData(null);
        setIsLoading(false);
        setError(message);
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, isApproachViewOpen, approachRunwayIdent]);

  const activePlateRoutes = useMemo(
    () => activeHorizontalPlateRoutes(loadedData?.plateRoutes ?? [], procedureVisibility),
    [loadedData, procedureVisibility],
  );

  const aircraftTracks = useMemo<ApproachViewTrack[]>(() => {
    // Sample only the sources that are the CURRENT tab's globe content (sourceSelection):
    // the observed ADS-B tracks in Observe, the optimized playback in Optimize. This is
    // what keeps the approach view in step with the active task — the observed tracks stay
    // loaded behind an approach view opened in another tab, but are NOT plotted there.
    const sources = [
      sourceSelection.observed ? trajectoryDataSource : null,
      sourceSelection.optimized ? optimizedTrajectoryDataSource : null,
    ].filter((dataSource): dataSource is Cesium.CzmlDataSource => dataSource !== null);
    if (sources.length === 0 || !currentTime || !loadedData || activePlateRoutes.length === 0) {
      return [];
    }
    const { runwayFrame } = loadedData;

    // Drop the per-entity whole-track cache when the geometry it depends on changes (the loaded
    // procedure/frame, the active routes, or the source set) — but NOT when currentTime moves.
    // This is the performance fix: each entity's track is sampled + classified ONCE (both are
    // expensive: getValue + projection + a protection-surface/segment classification per point),
    // and every clock tick then only re-checks liveness and the current marker.
    const cacheDeps = [loadedData, activePlateRoutes, trajectoryDataSource, optimizedTrajectoryDataSource];
    if (cacheDeps.some((dep, index) => dep !== trackCacheRef.current.deps[index])) {
      trackCacheRef.current = { deps: cacheDeps, tracks: new Map() };
    }
    const cache = trackCacheRef.current.tracks;

    // Only entities with a time-dynamic position are aircraft (the optimized CZML also
    // carries trail polylines, which have no `position`).
    const trajectoryEntities = sources.flatMap((dataSource) =>
      dataSource.entities.values.filter((entity) => entity.id !== "document" && entity.position),
    );

    const tracks: ApproachViewTrack[] = [];
    for (const entity of trajectoryEntities) {
      const currentPoint = currentIfLive(entity, currentTime, runwayFrame);
      if (!currentPoint) continue; // not airborne / parked — the only work for most entities

      let trail = cache.get(entity.id);
      if (!trail) {
        trail = classifyTrackSamples(
          sampleWholeTrack(entity, currentTime, runwayFrame),
          activePlateRoutes,
          runwayFrame,
        );
        cache.set(entity.id, trail);
      }
      const current = classifyApproachViewSample(currentPoint, activePlateRoutes, runwayFrame);
      if (!current) continue;
      // Plot only aircraft that fly this procedure (trail or the current point reaches PRIMARY).
      if (!trackEngagesProcedure(trail) && current.segmentAssessment.containment !== "PRIMARY") {
        continue;
      }
      tracks.push({
        flightId: entity.id,
        name: entity.name || undefined,
        color: colorForFlightId(entity.id),
        current,
        trail,
        isSelected: entity.id === selectedFlightId,
      });
    }
    return sortApproachViewTracksBySelection(tracks);
  }, [
    activePlateRoutes,
    currentTime,
    loadedData,
    selectedFlightId,
    sourceSelection,
    trajectoryDataSource,
    optimizedTrajectoryDataSource,
  ]);

  const activeReferenceMarks = useMemo(
    () =>
      loadedData && approachRunwayIdent
        ? buildRunwayReferenceMarksFromPlateRoutes(activePlateRoutes, approachRunwayIdent)
        : [],
    [activePlateRoutes, loadedData, approachRunwayIdent],
  );
  const activeProcedureNames = useMemo(
    () => [...new Set(activePlateRoutes.map((route) => route.procedureName))],
    [activePlateRoutes],
  );

  return {
    isLoading,
    error,
    currentTimeIso: currentTime ? formatJulianTime(currentTime) : null,
    runwayFrame: loadedData?.runwayFrame ?? null,
    plateRoutes: activePlateRoutes,
    referenceMarks: activeReferenceMarks,
    procedureNames: activeProcedureNames,
    sourceCycle: loadedData?.sourceCycle ?? null,
    aircraftTracks,
    sourceLinked,
  };
}
