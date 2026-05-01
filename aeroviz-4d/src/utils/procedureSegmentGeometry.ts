import type {
  BuildDiagnostic,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
  SourceRef,
} from "../data/procedurePackage";

const EARTH_RADIUS_M = 6_378_137;
const FEET_TO_METERS = 0.3048;
const METERS_PER_NM = 1852;

export interface CartesianPoint {
  x: number;
  y: number;
  z: number;
}

export interface GeoPoint {
  lonDeg: number;
  latDeg: number;
  altM: number;
}

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

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function toDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

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

function toCartesian(point: GeoPoint): CartesianPoint {
  const lon = toRadians(point.lonDeg);
  const lat = toRadians(point.latDeg);
  const radius = EARTH_RADIUS_M + point.altM;
  const cosLat = Math.cos(lat);
  return {
    x: radius * cosLat * Math.cos(lon),
    y: radius * cosLat * Math.sin(lon),
    z: radius * Math.sin(lat),
  };
}

function haversineDistanceM(start: GeoPoint, end: GeoPoint): number {
  const lat1 = toRadians(start.latDeg);
  const lat2 = toRadians(end.latDeg);
  const dLat = toRadians(end.latDeg - start.latDeg);
  const dLon = toRadians(end.lonDeg - start.lonDeg);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function interpolateGreatCircle(start: GeoPoint, end: GeoPoint, ratio: number): GeoPoint {
  const lat1 = toRadians(start.latDeg);
  const lon1 = toRadians(start.lonDeg);
  const lat2 = toRadians(end.latDeg);
  const lon2 = toRadians(end.lonDeg);
  const centralAngle = haversineDistanceM(start, end) / EARTH_RADIUS_M;

  if (centralAngle < 1e-9) {
    return {
      lonDeg: start.lonDeg + (end.lonDeg - start.lonDeg) * ratio,
      latDeg: start.latDeg + (end.latDeg - start.latDeg) * ratio,
      altM: start.altM + (end.altM - start.altM) * ratio,
    };
  }

  const sinCentral = Math.sin(centralAngle);
  const a = Math.sin((1 - ratio) * centralAngle) / sinCentral;
  const b = Math.sin(ratio * centralAngle) / sinCentral;
  const x = a * Math.cos(lat1) * Math.cos(lon1) + b * Math.cos(lat2) * Math.cos(lon2);
  const y = a * Math.cos(lat1) * Math.sin(lon1) + b * Math.cos(lat2) * Math.sin(lon2);
  const z = a * Math.sin(lat1) + b * Math.sin(lat2);
  const lat = Math.atan2(z, Math.hypot(x, y));
  const lon = Math.atan2(y, x);

  return {
    lonDeg: toDegrees(lon),
    latDeg: toDegrees(lat),
    altM: start.altM + (end.altM - start.altM) * ratio,
  };
}

function bearingRad(start: GeoPoint, end: GeoPoint): number {
  const lat1 = toRadians(start.latDeg);
  const lat2 = toRadians(end.latDeg);
  const dLon = toRadians(end.lonDeg - start.lonDeg);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return Math.atan2(y, x);
}

function offsetPoint(point: GeoPoint, bearing: number, distanceM: number): GeoPoint {
  const angularDistance = distanceM / EARTH_RADIUS_M;
  const lat1 = toRadians(point.latDeg);
  const lon1 = toRadians(point.lonDeg);
  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angularDistance) +
      Math.cos(lat1) * Math.sin(angularDistance) * Math.cos(bearing),
  );
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angularDistance) * Math.cos(lat1),
      Math.cos(angularDistance) - Math.sin(lat1) * Math.sin(lat2),
    );
  return {
    lonDeg: toDegrees(lon2),
    latDeg: toDegrees(lat2),
    altM: point.altM,
  };
}

function localBearing(points: GeoPoint[], index: number): number {
  if (points.length < 2) return 0;
  if (index === 0) return bearingRad(points[0], points[1]);
  if (index === points.length - 1) {
    return bearingRad(points[index - 1], points[index]);
  }
  return bearingRad(points[index - 1], points[index + 1]);
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

  const lengthNm = haversineDistanceM(start, end) / METERS_PER_NM;
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
