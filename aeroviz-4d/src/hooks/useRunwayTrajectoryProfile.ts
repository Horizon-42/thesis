import { useEffect, useMemo, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import { buildProcedureProfileProjection } from "../data/procedureProfileProjection";
import { loadProcedureRenderBundleData } from "../data/procedureRenderBundle";
import {
  activeHorizontalPlateRoutes,
  buildProfileAircraftTracks,
  type ProfileAircraftInput,
  type ProfileAircraftTrack,
  type SampledRunwayPoint,
} from "../data/runwayTrajectoryProfileAnalysis";
import { planProfileTrajectorySources } from "../data/profileTrajectorySources";
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
  ProfileAircraftSample,
  ProfileAircraftTrack,
} from "../data/runwayTrajectoryProfileAnalysis";

const TICK_THROTTLE_MS = 120;
// The profile plots each aircraft's WHOLE track across the procedure, not a short trailing
// window: we sample outward from the current time in both directions until the entity has no
// position (before its first / after its last sample). The step sets the plotted resolution;
// the per-direction cap is only a runaway backstop — break-on-null stops at the real ends.
const TRACK_SAMPLE_STEP_SECONDS = 5;
const MAX_TRACK_SAMPLES_PER_DIRECTION = 600;

export interface RunwayTrajectoryProfileState {
  isLoading: boolean;
  error: string | null;
  currentTimeIso: string | null;
  runwayFrame: RunwayFrame | null;
  plateRoutes: HorizontalPlateRoute[];
  referenceMarks: RunwayReferenceMark[];
  procedureNames: string[];
  sourceCycle: string | null;
  aircraftTracks: ProfileAircraftTrack[];
  /** Whether a trajectory source belonging to the CURRENT tab is loaded (the
   *  profile's "CZML linked" indicator). Mirrors what the profile actually plots. */
  sourceLinked: boolean;
}

interface LoadedProfileData {
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
 * Sample a LIVE aircraft's whole track around the current time, or `null` if it is not live.
 *
 * "Live" = the aircraft is actually MOVING at currentTime. This matters because of the HOLD
 * extrapolation (see `samePosition`): a landed flight reads as frozen on the threshold
 * forever, and a walk-until-undefined would run to the cap over the held tail. So we:
 *   • drop the entity when its sample one step back equals `current` — that is the held tail
 *     (landed/parked, not live; a `null` prior instead means it just appeared = live), and
 *   • stop each walk when the position stops changing (the real track end for a HOLD entity)
 *     as well as on `null` (the start, where backward extrapolation is NONE).
 * The result spans exactly the real approach — the whole flown + remaining path — both
 * directions from currentTime, and never samples the held tail.
 */
export function sampleEntityTrack(
  entity: Cesium.Entity,
  currentTime: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
): { current: SampledRunwayPoint; trail: SampledRunwayPoint[] } | null {
  const current = sampleRunwayPoint(entity, currentTime, runwayFrame, formatJulianTime(currentTime));
  if (!current) return null; // not airborne at currentTime (before its first sample)

  const priorTime = Cesium.JulianDate.addSeconds(
    currentTime,
    -TRACK_SAMPLE_STEP_SECONDS,
    new Cesium.JulianDate(),
  );
  const prior = sampleRunwayPoint(entity, priorTime, runwayFrame, formatJulianTime(priorTime));
  if (prior && samePosition(prior, current)) return null; // past its last sample — landed/parked

  const walk = (direction: 1 | -1): SampledRunwayPoint[] => {
    const points: SampledRunwayPoint[] = [];
    let previous = current;
    for (let step = 1; step <= MAX_TRACK_SAMPLES_PER_DIRECTION; step += 1) {
      const time = Cesium.JulianDate.addSeconds(
        currentTime,
        direction * step * TRACK_SAMPLE_STEP_SECONDS,
        new Cesium.JulianDate(),
      );
      const point = sampleRunwayPoint(entity, time, runwayFrame, formatJulianTime(time));
      // null = the track end (NONE extrapolation before the first sample); an unchanged
      // position = the HOLD-extrapolated tail past the last real sample. Either ends the walk.
      if (!point || samePosition(point, previous)) break;
      points.push(point);
      previous = point;
    }
    return points;
  };

  const backward = walk(-1).reverse(); // oldest → toward current
  const forward = walk(1); // after current → newest
  return { current, trail: [...backward, current, ...forward] };
}

export function useRunwayTrajectoryProfile(): RunwayTrajectoryProfileState {
  const {
    viewer,
    mode,
    activeAirportCode,
    procedureVisibility,
    selectedRunway,
    isRunwayProfileOpen,
    trajectoryComparison,
    trajectoryDataSource,
    optimizedTrajectoryDataSource,
    selectedFlightId,
  } = useApp();
  // Which trajectory sources are the CURRENT tab's globe content — the profile plots
  // only those, so it mirrors the active task instead of also drawing the observed
  // tracks that stay loaded (but hidden) behind a profile opened in Optimize/Fly/Compare.
  const sourceSelection = useMemo(
    () =>
      planProfileTrajectorySources({
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
  // The profile's runway is the global Landing-Runway selection, in the procedure
  // (RW-prefixed) spelling. `null` when "All runways" is selected — no single profile.
  const profileRunwayIdent = selectedRunway ? normalizeRunwayIdent(selectedRunway) : null;
  const [currentTime, setCurrentTime] = useState<Cesium.JulianDate | null>(null);
  const [loadedData, setLoadedData] = useState<LoadedProfileData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!viewer || !isRunwayProfileOpen) {
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
  }, [viewer, isRunwayProfileOpen]);

  useEffect(() => {
    if (!activeAirportCode || !profileRunwayIdent || !isRunwayProfileOpen) {
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

        const runwayFrame = buildRunwayFrame(runwayCollection, profileRunwayIdent);
        const procedureProfileProjection = buildProcedureProfileProjection(
          procedureRenderData,
          runwayFrame,
          profileRunwayIdent,
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
  }, [activeAirportCode, isRunwayProfileOpen, profileRunwayIdent]);

  const activePlateRoutes = useMemo(
    () => activeHorizontalPlateRoutes(loadedData?.plateRoutes ?? [], procedureVisibility),
    [loadedData, procedureVisibility],
  );

  const aircraftTracks = useMemo<ProfileAircraftTrack[]>(() => {
    // Sample only the sources that are the CURRENT tab's globe content (sourceSelection):
    // the observed ADS-B tracks in Observe, the optimized playback in Optimize. This is
    // what keeps the profile in step with the active task — the observed tracks stay
    // loaded behind a profile opened in another tab, but are NOT plotted there.
    const sources = [
      sourceSelection.observed ? trajectoryDataSource : null,
      sourceSelection.optimized ? optimizedTrajectoryDataSource : null,
    ].filter((dataSource): dataSource is Cesium.CzmlDataSource => dataSource !== null);
    if (sources.length === 0 || !currentTime || !loadedData || activePlateRoutes.length === 0) {
      return [];
    }

    // Only entities with a time-dynamic position are aircraft (the optimized CZML also
    // carries trail polylines, which have no `position`).
    const trajectoryEntities = sources.flatMap((dataSource) =>
      dataSource.entities.values.filter((entity) => entity.id !== "document" && entity.position),
    );
    const aircraft: ProfileAircraftInput[] = trajectoryEntities
      .map((entity) => {
        const sampled = sampleEntityTrack(entity, currentTime, loadedData.runwayFrame);
        return sampled ? { flightId: entity.id, ...sampled } : null;
      })
      .filter((input): input is ProfileAircraftInput => input !== null);

    return buildProfileAircraftTracks({
      aircraft,
      activePlateRoutes,
      runwayFrame: loadedData.runwayFrame,
      selectedFlightId,
    });
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
      loadedData && profileRunwayIdent
        ? buildRunwayReferenceMarksFromPlateRoutes(activePlateRoutes, profileRunwayIdent)
        : [],
    [activePlateRoutes, loadedData, profileRunwayIdent],
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
