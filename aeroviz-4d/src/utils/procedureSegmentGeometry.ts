import type {
  BuildDiagnostic,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
  SourceRef,
} from "../data/procedurePackage";
import {
  FEET_TO_METERS,
  METERS_PER_NM,
  distanceNm,
  haversineDistanceM,
  interpolateGreatCircle,
  localBearing,
  offsetPoint,
  toCartesian,
  type CartesianPoint,
  type GeoPoint,
} from "./procedureGeoMath";
import {
  buildTfTurnJunctions,
  type TurnJunctionGeometry,
} from "./procedureTurnGeometry";
import { buildRfLeg, buildRfParallelEnvelope } from "./procedureRfGeometry";

export type { CartesianPoint, GeoPoint } from "./procedureGeoMath";

export interface PolylineGeometry3D {
  worldPositions: CartesianPoint[];
  geoPositions: GeoPoint[];
  geodesicLengthNm: number;
  isArc: boolean;
  arcCenter?: GeoPoint;
  arcRadiusNm?: number;
  arcStartAngleRad?: number;
  arcSweepRad?: number;
  turnDirection?: "LEFT" | "RIGHT";
}

export interface StationAxisSample {
  stationNm: number;
  position: CartesianPoint;
  geoPosition: GeoPoint;
}

export interface StationAxis {
  samples: StationAxisSample[];
  totalLengthNm: number;
}

export interface WidthSample {
  stationNm: number;
  halfWidthNm: number;
}

export interface LateralEnvelopeGeometry {
  geometryId: string;
  envelopeType: "PRIMARY" | "SECONDARY";
  leftBoundary: CartesianPoint[];
  rightBoundary: CartesianPoint[];
  leftGeoBoundary: GeoPoint[];
  rightGeoBoundary: GeoPoint[];
  halfWidthNmSamples: WidthSample[];
}

export interface SegmentGeometryBundle {
  segmentId: string;
  centerline: PolylineGeometry3D;
  stationAxis: StationAxis;
  primaryEnvelope?: LateralEnvelopeGeometry;
  secondaryEnvelope?: LateralEnvelopeGeometry;
  turnJunctions: TurnJunctionGeometry[];
  diagnostics: BuildDiagnostic[];
}

export interface GeometryBuildContext {
  samplingStepNm: number;
  enableDebugPrimitives: boolean;
}

export const DEFAULT_GEOMETRY_BUILD_CONTEXT: GeometryBuildContext = {
  samplingStepNm: 0.25,
  enableDebugPrimitives: false,
};

function sourceRefsFor(leg: ProcedurePackageLeg | undefined): SourceRef[] {
  return leg?.sourceRefs ?? [];
}

function diagnostic(
  code: BuildDiagnostic["code"],
  message: string,
  severity: BuildDiagnostic["severity"],
  segmentId?: string,
  legId?: string,
  sourceRefs: SourceRef[] = [],
): BuildDiagnostic {
  return { code, message, severity, segmentId, legId, sourceRefs };
}

function pointFromFix(fix: ProcedurePackageFix): GeoPoint | null {
  if (fix.latDeg === null || fix.lonDeg === null) return null;
  return {
    lonDeg: fix.lonDeg,
    latDeg: fix.latDeg,
    altM: (fix.altFtMsl ?? 0) * FEET_TO_METERS,
  };
}

export function buildTfLeg(
  leg: ProcedurePackageLeg,
  fixes: Map<string, ProcedurePackageFix>,
  ctx: GeometryBuildContext = DEFAULT_GEOMETRY_BUILD_CONTEXT,
): { geometry: PolylineGeometry3D | null; diagnostics: BuildDiagnostic[] } {
  const diagnostics: BuildDiagnostic[] = [];

  if (leg.legType !== "TF") {
    diagnostics.push(
      diagnostic(
        "UNSUPPORTED_LEG_TYPE",
        `${leg.legId}: buildTfLeg only accepts TF legs; received ${leg.legType}.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  const startFix = leg.startFixId ? fixes.get(leg.startFixId) : undefined;
  const endFix = leg.endFixId ? fixes.get(leg.endFixId) : undefined;
  const start = startFix ? pointFromFix(startFix) : null;
  const end = endFix ? pointFromFix(endFix) : null;

  if (!start || !end) {
    diagnostics.push(
      diagnostic(
        "SOURCE_INCOMPLETE",
        `${leg.legId}: TF geometry requires positioned start and end fixes.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  const lengthNm = distanceNm(start, end);
  const sampleCount = Math.max(1, Math.ceil(lengthNm / Math.max(ctx.samplingStepNm, 0.01)));
  const geoPositions = Array.from({ length: sampleCount + 1 }, (_, index) =>
    interpolateGreatCircle(start, end, index / sampleCount),
  );

  return {
    geometry: {
      worldPositions: geoPositions.map(toCartesian),
      geoPositions,
      geodesicLengthNm: lengthNm,
      isArc: false,
    },
    diagnostics,
  };
}

export function computeStationAxis(centerline: PolylineGeometry3D): StationAxis {
  let totalM = 0;
  const samples = centerline.geoPositions.map((geoPosition, index) => {
    if (index > 0) {
      totalM += haversineDistanceM(centerline.geoPositions[index - 1], geoPosition);
    }
    return {
      stationNm: totalM / METERS_PER_NM,
      position: centerline.worldPositions[index],
      geoPosition,
    };
  });

  return {
    samples,
    totalLengthNm: totalM / METERS_PER_NM,
  };
}

export function buildStraightEnvelope(
  geometryId: string,
  envelopeType: "PRIMARY" | "SECONDARY",
  centerline: PolylineGeometry3D,
  halfWidthNm: number,
): LateralEnvelopeGeometry {
  const halfWidthM = halfWidthNm * METERS_PER_NM;
  const stationAxis = computeStationAxis(centerline);
  const leftGeoBoundary = centerline.geoPositions.map((point, index) =>
    offsetPoint(point, localBearing(centerline.geoPositions, index) - Math.PI / 2, halfWidthM),
  );
  const rightGeoBoundary = centerline.geoPositions.map((point, index) =>
    offsetPoint(point, localBearing(centerline.geoPositions, index) + Math.PI / 2, halfWidthM),
  );

  return {
    geometryId,
    envelopeType,
    leftBoundary: leftGeoBoundary.map(toCartesian),
    rightBoundary: rightGeoBoundary.map(toCartesian),
    leftGeoBoundary,
    rightGeoBoundary,
    halfWidthNmSamples: stationAxis.samples.map((sample) => ({
      stationNm: sample.stationNm,
      halfWidthNm,
    })),
  };
}

function buildSegmentEnvelope(
  geometryId: string,
  envelopeType: "PRIMARY" | "SECONDARY",
  centerline: PolylineGeometry3D,
  halfWidthNm: number,
): LateralEnvelopeGeometry {
  return (
    buildRfParallelEnvelope(geometryId, envelopeType, centerline, halfWidthNm) ??
    buildStraightEnvelope(geometryId, envelopeType, centerline, halfWidthNm)
  );
}

export function buildSegmentGeometryBundle(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  fixes: Map<string, ProcedurePackageFix>,
  ctx: GeometryBuildContext = DEFAULT_GEOMETRY_BUILD_CONTEXT,
): SegmentGeometryBundle {
  const diagnostics: BuildDiagnostic[] = [];
  const segmentLegs = legs.filter((leg) => leg.segmentId === segment.segmentId);
  const constructibleLegs = segmentLegs.filter(
    (leg) => leg.legType === "TF" || leg.legType === "RF",
  );

  if (constructibleLegs.length === 0) {
    diagnostics.push(
      diagnostic(
        "UNSUPPORTED_LEG_TYPE",
        `${segment.segmentId}: segment geometry kernel requires at least one TF or RF leg.`,
        "WARN",
        segment.segmentId,
      ),
    );
  }

  const legGeometries = constructibleLegs
    .map((leg) => {
      const result =
        leg.legType === "RF"
          ? buildRfLeg(leg, fixes, ctx)
          : buildTfLeg(leg, fixes, ctx);
      diagnostics.push(...result.diagnostics);
      return result.geometry;
    })
    .filter((geometry): geometry is PolylineGeometry3D => geometry !== null);

  const geoPositions = legGeometries.flatMap((geometry, geometryIndex) =>
    geometry.geoPositions.slice(geometryIndex === 0 ? 0 : 1),
  );
  const worldPositions = geoPositions.map(toCartesian);
  const centerline: PolylineGeometry3D = {
    geoPositions,
    worldPositions,
    geodesicLengthNm: Math.max(0, legGeometries.reduce((sum, geometry) => sum + geometry.geodesicLengthNm, 0)),
    isArc: legGeometries.some((geometry) => geometry.isArc),
    ...(legGeometries.length === 1 && legGeometries[0].isArc
      ? {
          arcCenter: legGeometries[0].arcCenter,
          arcRadiusNm: legGeometries[0].arcRadiusNm,
          arcStartAngleRad: legGeometries[0].arcStartAngleRad,
          arcSweepRad: legGeometries[0].arcSweepRad,
          turnDirection: legGeometries[0].turnDirection,
        }
      : {}),
  };
  const stationAxis = computeStationAxis(centerline);
  const turnJunctions =
    geoPositions.length >= 3 && !centerline.isArc
      ? buildTfTurnJunctions(
          segment.segmentId,
          centerline,
          segment.xttNm * 2,
          segment.secondaryEnabled ? segment.xttNm * 3 : null,
        )
      : [];

  if (segment.segmentType.startsWith("FINAL") && turnJunctions.length > 0) {
    diagnostics.push(
      diagnostic(
        "FINAL_HAS_TURN",
        `${segment.segmentId}: TF turn junctions were detected inside a final segment; visual fill patches are not a compliant final turn construction.`,
        "WARN",
        segment.segmentId,
        undefined,
        segment.sourceRefs,
      ),
    );
  }

  return {
    segmentId: segment.segmentId,
    centerline,
    stationAxis,
    primaryEnvelope: geoPositions.length >= 2
      ? buildSegmentEnvelope(`${segment.segmentId}:primary`, "PRIMARY", centerline, segment.xttNm * 2)
      : undefined,
    secondaryEnvelope: geoPositions.length >= 2 && segment.secondaryEnabled
      ? buildSegmentEnvelope(
          `${segment.segmentId}:secondary`,
          "SECONDARY",
          centerline,
          segment.xttNm * 3,
        )
      : undefined,
    turnJunctions,
    diagnostics,
  };
}
