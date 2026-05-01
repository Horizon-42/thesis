import type { PolylineGeometry3D } from "./procedureSegmentGeometry";
import {
  METERS_PER_NM,
  bearingRad,
  distanceNm,
  interpolateGreatCircle,
  type GeoPoint,
} from "./procedureGeoMath";
import {
  buildVariableWidthRibbon,
  type VariableWidthRibbonGeometry,
} from "./procedureSurfaceGeometry";

export interface TurnJunctionPatch {
  geometryId: string;
  envelopeType: "PRIMARY" | "SECONDARY";
  ribbon: VariableWidthRibbonGeometry;
  halfWidthNm: number;
}

export interface TurnJunctionGeometry {
  geometryId: string;
  segmentId: string;
  turnPointIndex: number;
  stationNm: number;
  turnAngleDeg: number;
  turnDirection: "LEFT" | "RIGHT";
  constructionStatus: "VISUAL_FILL_ONLY";
  primaryPatch: TurnJunctionPatch;
  secondaryPatch?: TurnJunctionPatch;
}

export interface TurnJunctionBuildOptions {
  insetNm?: number;
  minTurnAngleDeg?: number;
}

const DEFAULT_TURN_JUNCTION_OPTIONS: Required<TurnJunctionBuildOptions> = {
  insetNm: 0.5,
  minTurnAngleDeg: 3,
};

function toDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

function signedAngleDifferenceDeg(fromBearingRad: number, toBearingRad: number): number {
  const fromDeg = toDegrees(fromBearingRad);
  const toDeg = toDegrees(toBearingRad);
  let delta = ((toDeg - fromDeg + 540) % 360) - 180;
  if (delta === -180) delta = 180;
  return delta;
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

function pointBeforeTurn(previous: GeoPoint, turnPoint: GeoPoint, insetNm: number): GeoPoint {
  const legLengthNm = distanceNm(previous, turnPoint);
  if (legLengthNm <= 1e-6) return turnPoint;
  const ratio = Math.max(0, (legLengthNm - insetNm) / legLengthNm);
  return interpolateGreatCircle(previous, turnPoint, ratio);
}

function pointAfterTurn(turnPoint: GeoPoint, next: GeoPoint, insetNm: number): GeoPoint {
  const legLengthNm = distanceNm(turnPoint, next);
  if (legLengthNm <= 1e-6) return turnPoint;
  const ratio = Math.min(1, insetNm / legLengthNm);
  return interpolateGreatCircle(turnPoint, next, ratio);
}

function buildPatch(
  geometryId: string,
  envelopeType: "PRIMARY" | "SECONDARY",
  centerline: PolylineGeometry3D,
  stationsNm: number[],
  halfWidthNm: number,
): TurnJunctionPatch {
  return {
    geometryId,
    envelopeType,
    ribbon: buildVariableWidthRibbon(
      geometryId,
      centerline,
      stationsNm,
      () => halfWidthNm,
    ),
    halfWidthNm,
  };
}

export function buildTfTurnJunctions(
  segmentId: string,
  centerline: PolylineGeometry3D,
  primaryHalfWidthNm: number,
  secondaryOuterHalfWidthNm: number | null,
  options: TurnJunctionBuildOptions = {},
): TurnJunctionGeometry[] {
  if (centerline.geoPositions.length < 3) return [];

  const opts = { ...DEFAULT_TURN_JUNCTION_OPTIONS, ...options };
  const stations = cumulativeStationsNm(centerline.geoPositions);
  const turnJunctions: TurnJunctionGeometry[] = [];

  for (let index = 1; index < centerline.geoPositions.length - 1; index += 1) {
    const previous = centerline.geoPositions[index - 1];
    const turnPoint = centerline.geoPositions[index];
    const next = centerline.geoPositions[index + 1];
    const inboundBearing = bearingRad(previous, turnPoint);
    const outboundBearing = bearingRad(turnPoint, next);
    const signedTurnAngleDeg = signedAngleDifferenceDeg(inboundBearing, outboundBearing);
    const turnAngleDeg = Math.abs(signedTurnAngleDeg);

    if (turnAngleDeg < opts.minTurnAngleDeg) continue;

    const inboundLengthNm = distanceNm(previous, turnPoint);
    const outboundLengthNm = distanceNm(turnPoint, next);
    const insetNm = Math.min(opts.insetNm, inboundLengthNm / 2, outboundLengthNm / 2);
    if (insetNm <= 1 / METERS_PER_NM) continue;

    const patchGeoPositions = [
      pointBeforeTurn(previous, turnPoint, insetNm),
      turnPoint,
      pointAfterTurn(turnPoint, next, insetNm),
    ];
    const patchStationsNm = [stations[index] - insetNm, stations[index], stations[index] + insetNm];
    const patchCenterline: PolylineGeometry3D = {
      geoPositions: patchGeoPositions,
      worldPositions: [],
      geodesicLengthNm: insetNm * 2,
      isArc: false,
    };
    const geometryId = `${segmentId}:turn:${index}`;

    turnJunctions.push({
      geometryId,
      segmentId,
      turnPointIndex: index,
      stationNm: stations[index],
      turnAngleDeg,
      turnDirection: signedTurnAngleDeg > 0 ? "RIGHT" : "LEFT",
      constructionStatus: "VISUAL_FILL_ONLY",
      primaryPatch: buildPatch(
        `${geometryId}:primary`,
        "PRIMARY",
        patchCenterline,
        patchStationsNm,
        primaryHalfWidthNm,
      ),
      secondaryPatch:
        secondaryOuterHalfWidthNm === null
          ? undefined
          : buildPatch(
              `${geometryId}:secondary`,
              "SECONDARY",
              patchCenterline,
              patchStationsNm,
              secondaryOuterHalfWidthNm,
            ),
    });
  }

  return turnJunctions;
}
