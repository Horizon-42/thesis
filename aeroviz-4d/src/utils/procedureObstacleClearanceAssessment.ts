import type {
  ProcedureProtectionSurface,
} from "../data/procedureProtectionSurfaces";
import type { ObstacleProperties } from "../types/geojson-aviation";
import { FEET_TO_METERS } from "./procedureGeoMath";
import {
  assessPointAgainstProtectionSurface,
  type ProtectionVolumeAssessment,
  type ProtectionVolumeContainment,
} from "./procedureProtectionVolumeAssessment";

export type ObstacleClearanceStatus =
  | "OCS_CLEAR"
  | "OCS_PENETRATION"
  | "LATERAL_ONLY"
  | "ALTITUDE_PROFILE_ONLY"
  | "OUTSIDE";

export type ObstacleClearanceRuleStatus =
  | "PRIMARY_OCS"
  | "SECONDARY_RAW_OCS_NO_REDUCED_ROC"
  | "LATERAL_OEA_ONLY"
  | "ALTITUDE_PROFILE_NOT_OCS"
  | "OUTSIDE_LATERAL_AREA";

export interface ProcedureObstaclePoint {
  obstacleId: string;
  lonDeg: number;
  latDeg: number;
  topAltitudeFtMsl: number;
  obstacleType?: string;
  source?: string;
}

export interface ObstaclePointFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number] | [number, number, number];
  };
  properties: ObstacleProperties;
}

export interface ObstacleClearanceAssessment {
  obstacleId: string;
  obstacleType?: string;
  surfaceId: string;
  segmentId: string;
  surfaceKind: ProcedureProtectionSurface["kind"];
  surfaceStatus: ProcedureProtectionSurface["status"];
  containment: ProtectionVolumeContainment;
  stationNm: number;
  lateralDistanceNm: number;
  primaryHalfWidthNm: number;
  secondaryOuterHalfWidthNm: number | null;
  obstacleTopFtMsl: number;
  surfaceAltitudeFtMsl: number | null;
  clearanceFt: number | null;
  status: ObstacleClearanceStatus;
  ruleStatus: ObstacleClearanceRuleStatus;
  notes: string[];
  surfaceAssessment: ProtectionVolumeAssessment;
}

export interface ObstacleClearanceAssessmentOptions {
  includeDebug?: boolean;
  includeOutside?: boolean;
}

export function obstaclePointFromFeature(
  feature: ObstaclePointFeature,
  fallbackIndex = 0,
): ProcedureObstaclePoint | null {
  if (feature.geometry.type !== "Point") return null;
  const [lonDeg, latDeg] = feature.geometry.coordinates;
  const topAltitudeFtMsl = feature.properties.amsl_ft;
  if (
    !Number.isFinite(lonDeg) ||
    !Number.isFinite(latDeg) ||
    !Number.isFinite(topAltitudeFtMsl)
  ) {
    return null;
  }

  return {
    obstacleId: feature.properties.oas_number || `obstacle:${fallbackIndex}`,
    lonDeg,
    latDeg,
    topAltitudeFtMsl,
    obstacleType: feature.properties.obstacle_type,
    source: feature.properties.source,
  };
}

function ruleStatusForAssessment(
  assessment: ProtectionVolumeAssessment,
): ObstacleClearanceRuleStatus {
  if (assessment.containment === "OUTSIDE") return "OUTSIDE_LATERAL_AREA";
  if (assessment.verticalKind === "NONE") return "LATERAL_OEA_ONLY";
  if (assessment.verticalKind !== "OCS") return "ALTITUDE_PROFILE_NOT_OCS";
  return assessment.containment === "SECONDARY"
    ? "SECONDARY_RAW_OCS_NO_REDUCED_ROC"
    : "PRIMARY_OCS";
}

function clearanceStatusForAssessment(
  assessment: ProtectionVolumeAssessment,
  clearanceFt: number | null,
): ObstacleClearanceStatus {
  if (assessment.containment === "OUTSIDE") return "OUTSIDE";
  if (assessment.verticalKind === "NONE") return "LATERAL_ONLY";
  if (assessment.verticalKind !== "OCS") return "ALTITUDE_PROFILE_ONLY";
  return (clearanceFt ?? 0) < 0 ? "OCS_PENETRATION" : "OCS_CLEAR";
}

function notesForAssessment(
  assessment: ProtectionVolumeAssessment,
): string[] {
  if (assessment.containment === "OUTSIDE") {
    return ["Obstacle is outside the surface lateral primary/secondary area."];
  }
  if (assessment.verticalKind === "NONE") {
    return ["Surface is lateral OEA only; no vertical obstacle clearance is computed."];
  }
  if (assessment.verticalKind !== "OCS") {
    return ["Surface vertical data is an altitude/profile aid, not an OCS clearance surface."];
  }
  if (assessment.containment === "SECONDARY") {
    return [
      "Obstacle is in secondary area; clearance is a raw OCS comparison because reduced secondary ROC rules are not modeled yet.",
    ];
  }
  return ["Obstacle is assessed against the primary OCS altitude at the projected station."];
}

export function assessObstacleAgainstProtectionSurface(
  obstacle: ProcedureObstaclePoint,
  surface: ProcedureProtectionSurface,
  options: ObstacleClearanceAssessmentOptions = {},
): ObstacleClearanceAssessment | null {
  if (!options.includeDebug && surface.status === "DEBUG_ESTIMATE") return null;
  const surfaceAssessment = assessPointAgainstProtectionSurface(
    {
      lonDeg: obstacle.lonDeg,
      latDeg: obstacle.latDeg,
      altM: obstacle.topAltitudeFtMsl * FEET_TO_METERS,
    },
    surface,
  );
  if (!surfaceAssessment) return null;
  if (!options.includeOutside && surfaceAssessment.containment === "OUTSIDE") return null;

  const clearanceFt =
    surfaceAssessment.verticalKind === "OCS" && surfaceAssessment.verticalDeltaFt !== null
      ? -surfaceAssessment.verticalDeltaFt
      : null;
  const surfaceAltitudeFtMsl =
    surfaceAssessment.verticalKind === "OCS"
      ? obstacle.topAltitudeFtMsl + (clearanceFt ?? 0)
      : null;

  return {
    obstacleId: obstacle.obstacleId,
    obstacleType: obstacle.obstacleType,
    surfaceId: surface.surfaceId,
    segmentId: surface.segmentId,
    surfaceKind: surface.kind,
    surfaceStatus: surface.status,
    containment: surfaceAssessment.containment,
    stationNm: surfaceAssessment.stationNm,
    lateralDistanceNm: surfaceAssessment.lateralDistanceNm,
    primaryHalfWidthNm: surfaceAssessment.primaryHalfWidthNm,
    secondaryOuterHalfWidthNm: surfaceAssessment.secondaryOuterHalfWidthNm,
    obstacleTopFtMsl: obstacle.topAltitudeFtMsl,
    surfaceAltitudeFtMsl,
    clearanceFt,
    status: clearanceStatusForAssessment(surfaceAssessment, clearanceFt),
    ruleStatus: ruleStatusForAssessment(surfaceAssessment),
    notes: notesForAssessment(surfaceAssessment),
    surfaceAssessment,
  };
}

function assessmentSortKey(assessment: ObstacleClearanceAssessment): number {
  if (assessment.status === "OCS_PENETRATION") return -1_000_000 + (assessment.clearanceFt ?? 0);
  if (assessment.clearanceFt !== null) return assessment.clearanceFt;
  if (assessment.containment === "PRIMARY") return 100_000 + assessment.lateralDistanceNm;
  if (assessment.containment === "SECONDARY") return 200_000 + assessment.lateralDistanceNm;
  return 300_000 + assessment.lateralDistanceNm;
}

export function assessObstaclesAgainstProtectionSurfaces(
  obstacles: ProcedureObstaclePoint[],
  surfaces: ProcedureProtectionSurface[],
  options: ObstacleClearanceAssessmentOptions = {},
): ObstacleClearanceAssessment[] {
  return obstacles
    .flatMap((obstacle) =>
      surfaces
        .map((surface) => assessObstacleAgainstProtectionSurface(obstacle, surface, options))
        .filter((assessment): assessment is ObstacleClearanceAssessment => assessment !== null),
    )
    .sort((left, right) => assessmentSortKey(left) - assessmentSortKey(right));
}
