import type {
  ProcedureProtectionSurface,
  ProtectionSurfaceStatus,
} from "../data/procedureProtectionSurfaces";
import {
  FEET_TO_METERS,
  METERS_PER_NM,
  distanceNm,
  interpolateGreatCircle,
  type GeoPoint,
} from "./procedureGeoMath";

export type ProtectionVolumeContainment = "PRIMARY" | "SECONDARY" | "OUTSIDE";
export type ProtectionVolumeVerticalRelation =
  | "NO_VERTICAL_LIMIT"
  | "ABOVE_SURFACE"
  | "BELOW_SURFACE"
  | "ON_SURFACE";

export interface ProtectionVolumeAssessment {
  surfaceId: string;
  segmentId: string;
  surfaceKind: ProcedureProtectionSurface["kind"];
  surfaceStatus: ProtectionSurfaceStatus;
  verticalKind: ProcedureProtectionSurface["vertical"]["kind"];
  verticalOrigin: ProcedureProtectionSurface["vertical"]["origin"];
  stationNm: number;
  segmentIndex: number;
  lateralOffsetNm: number;
  lateralDistanceNm: number;
  primaryHalfWidthNm: number;
  secondaryOuterHalfWidthNm: number | null;
  containment: ProtectionVolumeContainment;
  verticalDeltaFt: number | null;
  verticalRelation: ProtectionVolumeVerticalRelation;
  closestPoint: GeoPoint;
}

interface CandidateAssessment extends ProtectionVolumeAssessment {
  distanceToCenterlineNm: number;
}

interface LocalPoint {
  eastM: number;
  northM: number;
}

interface WidthAtStation {
  primaryHalfWidthNm: number;
  secondaryOuterHalfWidthNm: number | null;
}

const VERTICAL_ON_SURFACE_TOLERANCE_FT = 1;

function pointToLocal(point: GeoPoint, origin: GeoPoint): LocalPoint {
  const meanLatRad = (((point.latDeg + origin.latDeg) / 2) * Math.PI) / 180;
  return {
    eastM:
      ((point.lonDeg - origin.lonDeg) * Math.PI * 6_378_137 * Math.cos(meanLatRad)) / 180,
    northM: ((point.latDeg - origin.latDeg) * Math.PI * 6_378_137) / 180,
  };
}

function cumulativeStationsNm(points: GeoPoint[]): number[] {
  let stationNm = 0;
  return points.map((point, index) => {
    if (index > 0) {
      stationNm += distanceNm(points[index - 1], point);
    }
    return stationNm;
  });
}

function interpolateValue(
  samples: Array<{ stationNm: number; value: number }>,
  stationNm: number,
): number | null {
  if (samples.length === 0) return null;
  const sorted = samples.slice().sort((left, right) => left.stationNm - right.stationNm);
  if (stationNm <= sorted[0].stationNm) return sorted[0].value;
  const last = sorted[sorted.length - 1];
  if (stationNm >= last.stationNm) return last.value;

  for (let index = 0; index < sorted.length - 1; index += 1) {
    const start = sorted[index];
    const end = sorted[index + 1];
    if (stationNm < start.stationNm || stationNm > end.stationNm) continue;
    const span = end.stationNm - start.stationNm;
    if (span <= 0) return start.value;
    const ratio = (stationNm - start.stationNm) / span;
    return start.value + ratio * (end.value - start.value);
  }

  return null;
}

function widthAtStation(
  surface: ProcedureProtectionSurface,
  stationNm: number,
): WidthAtStation | null {
  const widthSamples = surface.lateral.widthSamples;
  const primaryHalfWidthNm = interpolateValue(
    widthSamples.map((sample) => ({
      stationNm: sample.stationNm,
      value: sample.primaryHalfWidthNm,
    })),
    stationNm,
  );
  if (primaryHalfWidthNm === null) return null;

  const secondaryOuterHalfWidthNm = interpolateValue(
    widthSamples
      .filter((sample) => typeof sample.secondaryOuterHalfWidthNm === "number")
      .map((sample) => ({
        stationNm: sample.stationNm,
        value: sample.secondaryOuterHalfWidthNm as number,
      })),
    stationNm,
  );

  return {
    primaryHalfWidthNm,
    secondaryOuterHalfWidthNm,
  };
}

function verticalReferenceFt(
  surface: ProcedureProtectionSurface,
  centerlineStart: GeoPoint,
  centerlineEnd: GeoPoint,
  ratio: number,
  stationNm: number,
): number | null {
  if (surface.vertical.kind === "NONE") return null;
  const verticalSample = interpolateValue(
    surface.vertical.samples.map((sample) => ({
      stationNm: sample.stationNm,
      value: sample.altitudeFtMsl,
    })),
    stationNm,
  );
  if (verticalSample !== null) return verticalSample;

  const startFt = centerlineStart.altM / FEET_TO_METERS;
  const endFt = centerlineEnd.altM / FEET_TO_METERS;
  return startFt + ratio * (endFt - startFt);
}

function verticalRelation(verticalDeltaFt: number | null): ProtectionVolumeVerticalRelation {
  if (verticalDeltaFt === null) return "NO_VERTICAL_LIMIT";
  if (Math.abs(verticalDeltaFt) <= VERTICAL_ON_SURFACE_TOLERANCE_FT) return "ON_SURFACE";
  return verticalDeltaFt > 0 ? "ABOVE_SURFACE" : "BELOW_SURFACE";
}

function containmentFromWidths(
  lateralDistanceNm: number,
  width: WidthAtStation,
): ProtectionVolumeContainment {
  if (lateralDistanceNm <= width.primaryHalfWidthNm) return "PRIMARY";
  if (
    width.secondaryOuterHalfWidthNm !== null &&
    lateralDistanceNm <= width.secondaryOuterHalfWidthNm
  ) {
    return "SECONDARY";
  }
  return "OUTSIDE";
}

function containmentRank(containment: ProtectionVolumeContainment): number {
  if (containment === "PRIMARY") return 0;
  if (containment === "SECONDARY") return 1;
  return 2;
}

function surfaceVerticalRank(surface: ProcedureProtectionSurface): number {
  if (surface.vertical.kind === "OCS") return 0;
  if (surface.vertical.kind === "ALTITUDE_PROFILE") return 1;
  return 2;
}

function assessPointOnSegment(
  point: GeoPoint,
  surface: ProcedureProtectionSurface,
  stations: number[],
  segmentIndex: number,
): CandidateAssessment | null {
  const start = surface.centerline.geoPositions[segmentIndex];
  const end = surface.centerline.geoPositions[segmentIndex + 1];
  if (!start || !end) return null;

  const endLocal = pointToLocal(end, start);
  const pointLocal = pointToLocal(point, start);
  const segmentLengthM = Math.hypot(endLocal.eastM, endLocal.northM);
  if (segmentLengthM <= 0) return null;

  const ratio = Math.max(
    0,
    Math.min(
      1,
      (pointLocal.eastM * endLocal.eastM + pointLocal.northM * endLocal.northM) /
        (segmentLengthM * segmentLengthM),
    ),
  );
  const closestHorizontalPoint = interpolateGreatCircle(start, end, ratio);
  const closestPoint = {
    ...closestHorizontalPoint,
    altM: start.altM + ratio * (end.altM - start.altM),
  };
  const closestLocal = {
    eastM: ratio * endLocal.eastM,
    northM: ratio * endLocal.northM,
  };
  const crossTrackM =
    (pointLocal.eastM * endLocal.northM - pointLocal.northM * endLocal.eastM) /
    segmentLengthM;
  const distanceToCenterlineNm =
    Math.hypot(
      pointLocal.eastM - closestLocal.eastM,
      pointLocal.northM - closestLocal.northM,
    ) / METERS_PER_NM;
  const stationNm =
    stations[segmentIndex] +
    ratio * Math.max(0, (stations[segmentIndex + 1] ?? stations[segmentIndex]) - stations[segmentIndex]);
  const width = widthAtStation(surface, stationNm);
  if (!width) return null;

  const verticalFt = verticalReferenceFt(surface, start, end, ratio, stationNm);
  const verticalDeltaFt =
    verticalFt === null ? null : point.altM / FEET_TO_METERS - verticalFt;
  const lateralOffsetNm = crossTrackM / METERS_PER_NM;
  const lateralDistanceNm = distanceToCenterlineNm;

  return {
    surfaceId: surface.surfaceId,
    segmentId: surface.segmentId,
    surfaceKind: surface.kind,
    surfaceStatus: surface.status,
    verticalKind: surface.vertical.kind,
    verticalOrigin: surface.vertical.origin,
    stationNm,
    segmentIndex,
    lateralOffsetNm,
    lateralDistanceNm,
    primaryHalfWidthNm: width.primaryHalfWidthNm,
    secondaryOuterHalfWidthNm: width.secondaryOuterHalfWidthNm,
    containment: containmentFromWidths(lateralDistanceNm, width),
    verticalDeltaFt,
    verticalRelation: verticalRelation(verticalDeltaFt),
    closestPoint: {
      ...closestPoint,
      altM: verticalFt === null ? closestPoint.altM : verticalFt * FEET_TO_METERS,
    },
    distanceToCenterlineNm,
  };
}

export function assessPointAgainstProtectionSurface(
  point: GeoPoint,
  surface: ProcedureProtectionSurface,
): ProtectionVolumeAssessment | null {
  const positions = surface.centerline.geoPositions;
  if (positions.length < 2) return null;
  const stations = cumulativeStationsNm(positions);
  let best: CandidateAssessment | null = null;

  for (let index = 0; index < positions.length - 1; index += 1) {
    const candidate = assessPointOnSegment(point, surface, stations, index);
    if (!candidate) continue;
    if (!best || candidate.distanceToCenterlineNm < best.distanceToCenterlineNm) {
      best = candidate;
    }
  }

  if (!best) return null;
  const { distanceToCenterlineNm: _distanceToCenterlineNm, ...assessment } = best;
  return assessment;
}

export function classifyPointAgainstProtectionSurfaces(
  point: GeoPoint,
  surfaces: ProcedureProtectionSurface[],
  options: { includeDebug?: boolean } = {},
): ProtectionVolumeAssessment | null {
  const assessments = surfaces
    .filter((surface) => options.includeDebug || surface.status !== "DEBUG_ESTIMATE")
    .map((surface) => ({
      surface,
      assessment: assessPointAgainstProtectionSurface(point, surface),
    }))
    .filter(
      (entry): entry is { surface: ProcedureProtectionSurface; assessment: ProtectionVolumeAssessment } =>
        entry.assessment !== null,
    );

  return assessments.sort((left, right) => {
    const containmentDelta =
      containmentRank(left.assessment.containment) - containmentRank(right.assessment.containment);
    if (containmentDelta !== 0) return containmentDelta;
    const verticalDelta = surfaceVerticalRank(left.surface) - surfaceVerticalRank(right.surface);
    if (verticalDelta !== 0) return verticalDelta;
    return left.assessment.lateralDistanceNm - right.assessment.lateralDistanceNm;
  })[0]?.assessment ?? null;
}
