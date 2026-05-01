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

export type { CartesianPoint, GeoPoint } from "./procedureGeoMath";

export interface PolylineGeometry3D {
  worldPositions: CartesianPoint[];
  geoPositions: GeoPoint[];
  geodesicLengthNm: number;
  isArc: boolean;
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

export function buildSegmentGeometryBundle(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  fixes: Map<string, ProcedurePackageFix>,
  ctx: GeometryBuildContext = DEFAULT_GEOMETRY_BUILD_CONTEXT,
): SegmentGeometryBundle {
  const diagnostics: BuildDiagnostic[] = [];
  const tfLegs = legs.filter((leg) => leg.segmentId === segment.segmentId && leg.legType === "TF");

  if (tfLegs.length === 0) {
    diagnostics.push(
      diagnostic(
        "UNSUPPORTED_LEG_TYPE",
        `${segment.segmentId}: initial segment geometry kernel only builds TF centerlines.`,
        "WARN",
        segment.segmentId,
      ),
    );
  }

  const legGeometries = tfLegs
    .map((leg) => {
      const result = buildTfLeg(leg, fixes, ctx);
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
    isArc: false,
  };
  const stationAxis = computeStationAxis(centerline);

  return {
    segmentId: segment.segmentId,
    centerline,
    stationAxis,
    primaryEnvelope: geoPositions.length >= 2
      ? buildStraightEnvelope(`${segment.segmentId}:primary`, "PRIMARY", centerline, segment.xttNm * 2)
      : undefined,
    secondaryEnvelope: geoPositions.length >= 2 && segment.secondaryEnabled
      ? buildStraightEnvelope(
          `${segment.segmentId}:secondary`,
          "SECONDARY",
          centerline,
          segment.xttNm * 3,
        )
      : undefined,
    diagnostics,
  };
}
