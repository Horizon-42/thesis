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
const TRAIL_LOOKBACK_SECONDS = 150;
const TRAIL_SAMPLE_STEP_SECONDS = 5;

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

export function useRunwayTrajectoryProfile(): RunwayTrajectoryProfileState {
  const {
    viewer,
    activeAirportCode,
    procedureVisibility,
    selectedRunway,
    isRunwayProfileOpen,
    trajectoryDataSource,
    optimizedTrajectoryDataSource,
    selectedFlightId,
  } = useApp();
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
    // Sample every loaded trajectory source — the observed ADS-B tracks AND the
    // optimized playback — so the profile shows whichever passes through the
    // procedure, in any mode (observed in Observe/Procedures, optimized in Optimize).
    const sources = [trajectoryDataSource, optimizedTrajectoryDataSource].filter(
      (dataSource): dataSource is Cesium.CzmlDataSource => dataSource !== null,
    );
    if (sources.length === 0 || !currentTime || !loadedData || activePlateRoutes.length === 0) {
      return [];
    }

    const currentTimeIso = formatJulianTime(currentTime);
    // Only entities with a time-dynamic position are aircraft (the optimized CZML also
    // carries trail polylines, which have no `position`).
    const trajectoryEntities = sources.flatMap((dataSource) =>
      dataSource.entities.values.filter((entity) => entity.id !== "document" && entity.position),
    );
    const aircraft: ProfileAircraftInput[] = trajectoryEntities
      .map((entity): ProfileAircraftInput => {
        const current = sampleRunwayPoint(
          entity,
          currentTime,
          loadedData.runwayFrame,
          currentTimeIso,
        );
        const trail: SampledRunwayPoint[] = [];
        for (
          let offsetSeconds = TRAIL_LOOKBACK_SECONDS;
          offsetSeconds >= TRAIL_SAMPLE_STEP_SECONDS;
          offsetSeconds -= TRAIL_SAMPLE_STEP_SECONDS
        ) {
          const sampleTime = Cesium.JulianDate.addSeconds(
            currentTime,
            -offsetSeconds,
            new Cesium.JulianDate(),
          );
          const samplePoint = sampleRunwayPoint(
            entity,
            sampleTime,
            loadedData.runwayFrame,
            formatJulianTime(sampleTime),
          );
          if (!samplePoint) continue;
          trail.push(samplePoint);
        }

        return {
          flightId: entity.id,
          current,
          trail,
        };
      });

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
  };
}
