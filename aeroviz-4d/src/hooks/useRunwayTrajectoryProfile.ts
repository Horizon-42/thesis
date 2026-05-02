import { useEffect, useMemo, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import { loadProcedureRenderBundleData } from "../data/procedureRenderBundle";
import { buildProcedureRoutes } from "../data/procedureRoutes";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import {
  attachRenderBundleAssessmentSegments,
  buildHorizontalPlateRoutes,
  buildRunwayReferenceMarksFromPlateRoutes,
  buildRunwayFrame,
  projectPositionToRunwayFrame,
  type HorizontalPlateRoute,
  type RunwayReferenceMark,
  type RunwayFeatureCollection,
  type RunwayFrame,
  type RunwayProfilePoint,
} from "../utils/runwayProfileGeometry";
import {
  classifyPointAgainstHorizontalPlateRoutes,
  type HorizontalPlateSegmentAssessment,
} from "../utils/procedureSegmentAssessment";

const TICK_THROTTLE_MS = 120;
const TRAIL_LOOKBACK_SECONDS = 150;
const TRAIL_SAMPLE_STEP_SECONDS = 5;

export interface ProfileAircraftSample extends RunwayProfilePoint {
  timeIso: string;
  segmentAssessment: HorizontalPlateSegmentAssessment;
}

export interface ProfileAircraftTrack {
  flightId: string;
  color: string;
  current: ProfileAircraftSample;
  trail: ProfileAircraftSample[];
  isSelected: boolean;
}

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

function colorForFlightId(flightId: string): string {
  let hash = 0;
  for (let index = 0; index < flightId.length; index += 1) {
    hash = (hash * 31 + flightId.charCodeAt(index)) >>> 0;
  }
  return `hsl(${hash % 360} 72% 58%)`;
}

function sampleRunwayPoint(
  entity: Cesium.Entity,
  time: Cesium.JulianDate,
  runwayFrame: RunwayFrame,
): RunwayProfilePoint | null {
  if (!entity.position) return null;

  const cartesian = entity.position.getValue(time, new Cesium.Cartesian3());
  if (!cartesian) return null;

  const cartographic = Cesium.Cartographic.fromCartesian(
    cartesian,
    Cesium.Ellipsoid.WGS84,
    new Cesium.Cartographic(),
  );
  if (!cartographic) return null;

  return projectPositionToRunwayFrame(
    runwayFrame,
    Cesium.Math.toDegrees(cartographic.longitude),
    Cesium.Math.toDegrees(cartographic.latitude),
    cartographic.height,
  );
}

function routeIsActive(
  route: HorizontalPlateRoute,
  procedureVisibility: Record<string, boolean>,
): boolean {
  return procedureVisibility[route.branchId] ?? route.defaultVisible;
}

export function useRunwayTrajectoryProfile(): RunwayTrajectoryProfileState {
  const {
    viewer,
    activeAirportCode,
    procedureVisibility,
    selectedProfileRunwayIdent,
    isRunwayProfileOpen,
    trajectoryDataSource,
    selectedFlightId,
  } = useApp();
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
    if (!activeAirportCode || !selectedProfileRunwayIdent || !isRunwayProfileOpen) {
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

        const runwayFrame = buildRunwayFrame(runwayCollection, selectedProfileRunwayIdent);
        const procedureRoutes = buildProcedureRoutes(procedureRenderData.documents);
        const plateRoutes = attachRenderBundleAssessmentSegments(
          buildHorizontalPlateRoutes(
            procedureRoutes,
            runwayFrame,
            selectedProfileRunwayIdent,
          ),
          procedureRenderData.renderBundles,
          runwayFrame,
          selectedProfileRunwayIdent,
        );
        setLoadedData({
          runwayFrame,
          plateRoutes,
          sourceCycle: procedureRenderData.index.sourceCycle ?? null,
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
  }, [activeAirportCode, isRunwayProfileOpen, selectedProfileRunwayIdent]);

  const activePlateRoutes = useMemo(
    () => loadedData?.plateRoutes.filter((route) => routeIsActive(route, procedureVisibility)) ?? [],
    [loadedData, procedureVisibility],
  );

  const aircraftTracks = useMemo<ProfileAircraftTrack[]>(() => {
    if (!trajectoryDataSource || !currentTime || !loadedData || activePlateRoutes.length === 0) {
      return [];
    }

    const currentTimeIso = formatJulianTime(currentTime);

    return trajectoryDataSource.entities.values
      .filter((entity) => entity.id !== "document")
      .map((entity) => {
        const currentPoint = sampleRunwayPoint(entity, currentTime, loadedData.runwayFrame);
        if (!currentPoint) return null;
        const currentAssessment = classifyPointAgainstHorizontalPlateRoutes(
          currentPoint,
          activePlateRoutes,
        );
        if (!currentAssessment || currentAssessment.containment !== "PRIMARY") {
          return null;
        }

        const trail: ProfileAircraftSample[] = [];
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
          const samplePoint = sampleRunwayPoint(entity, sampleTime, loadedData.runwayFrame);
          if (!samplePoint) continue;
          const sampleAssessment = classifyPointAgainstHorizontalPlateRoutes(
            samplePoint,
            activePlateRoutes,
          );
          if (!sampleAssessment || sampleAssessment.containment !== "PRIMARY") continue;
          trail.push({
            ...samplePoint,
            timeIso: formatJulianTime(sampleTime),
            segmentAssessment: sampleAssessment,
          });
        }

        trail.push({
          ...currentPoint,
          timeIso: currentTimeIso,
          segmentAssessment: currentAssessment,
        });

        return {
          flightId: entity.id,
          color: colorForFlightId(entity.id),
          current: {
            ...currentPoint,
            timeIso: currentTimeIso,
            segmentAssessment: currentAssessment,
          },
          trail,
          isSelected: entity.id === selectedFlightId,
        };
      })
      .filter((track): track is ProfileAircraftTrack => track !== null)
      .sort((left, right) => {
        if (left.isSelected === right.isSelected) {
          return left.flightId.localeCompare(right.flightId);
        }
        return left.isSelected ? -1 : 1;
      });
  }, [activePlateRoutes, currentTime, loadedData, selectedFlightId, trajectoryDataSource]);

  const activeReferenceMarks = useMemo(
    () =>
      loadedData && selectedProfileRunwayIdent
        ? buildRunwayReferenceMarksFromPlateRoutes(activePlateRoutes, selectedProfileRunwayIdent)
        : [],
    [activePlateRoutes, loadedData, selectedProfileRunwayIdent],
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
