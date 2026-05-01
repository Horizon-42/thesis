import type { BuildDiagnostic, ProcedureSegment, SourceRef } from "../data/procedurePackage";
import type { PolylineGeometry3D } from "./procedureSegmentGeometry";
import {
  METERS_PER_NM,
  bearingRad,
  clamp,
  distanceNm,
  interpolateGreatCircle,
  localBearing,
  offsetPoint,
  toCartesian,
  type CartesianPoint,
  type GeoPoint,
} from "./procedureGeoMath";

export interface WidthStationSample {
  stationNm: number;
  halfWidthNm: number;
}

export interface VariableWidthRibbonGeometry {
  geometryId: string;
  leftBoundary: CartesianPoint[];
  rightBoundary: CartesianPoint[];
  leftGeoBoundary: GeoPoint[];
  rightGeoBoundary: GeoPoint[];
  halfWidthNmSamples: WidthStationSample[];
}

export interface LnavFinalOeaGeometry {
  geometryId: string;
  segmentId: string;
  surfaceType: "LNAV_FINAL_OEA";
  centerline: PolylineGeometry3D;
  primary: VariableWidthRibbonGeometry;
  secondaryOuter: VariableWidthRibbonGeometry;
  taper: {
    startStationNm: number;
    endStationNm: number;
    initialPrimaryHalfWidthNm: number;
    stablePrimaryHalfWidthNm: number;
    secondaryWidthNm: number;
  };
}

export interface LnavFinalOeaOptions {
  startBeforePfafNm?: number;
  endAfterThresholdNm?: number;
  stableAfterPfafNm?: number;
  initialPrimaryHalfWidthNm?: number;
  stablePrimaryHalfWidthNm?: number;
  secondaryWidthNm?: number;
  samplingStepNm?: number;
}

const DEFAULT_LNAV_FINAL_OEA_OPTIONS: Required<LnavFinalOeaOptions> = {
  startBeforePfafNm: 0.3,
  endAfterThresholdNm: 0.3,
  stableAfterPfafNm: 1,
  initialPrimaryHalfWidthNm: 0.3,
  stablePrimaryHalfWidthNm: 0.6,
  secondaryWidthNm: 0.3,
  samplingStepNm: 0.25,
};

function diagnostic(
  code: BuildDiagnostic["code"],
  message: string,
  severity: BuildDiagnostic["severity"],
  segmentId?: string,
  sourceRefs: SourceRef[] = [],
): BuildDiagnostic {
  return { code, message, severity, segmentId, sourceRefs };
}

export function sampleStationValues(startStationNm: number, endStationNm: number, stepNm: number): number[] {
  const safeStepNm = Math.max(stepNm, 0.01);
  const count = Math.max(1, Math.ceil((endStationNm - startStationNm) / safeStepNm));
  return Array.from({ length: count + 1 }, (_, index) => {
    if (index === count) return endStationNm;
    return startStationNm + index * safeStepNm;
  });
}

function pointAtStation(centerline: GeoPoint[], stationNm: number): GeoPoint {
  if (centerline.length === 1) return centerline[0];

  if (stationNm <= 0) {
    const bearing = bearingRad(centerline[1], centerline[0]);
    return offsetPoint(centerline[0], bearing, Math.abs(stationNm) * METERS_PER_NM);
  }

  let travelledNm = 0;
  for (let index = 0; index < centerline.length - 1; index++) {
    const start = centerline[index];
    const end = centerline[index + 1];
    const legLengthNm = distanceNm(start, end);
    if (stationNm <= travelledNm + legLengthNm || index === centerline.length - 2) {
      const ratio = legLengthNm <= 1e-9 ? 0 : clamp((stationNm - travelledNm) / legLengthNm, 0, 1);
      return interpolateGreatCircle(start, end, ratio);
    }
    travelledNm += legLengthNm;
  }

  return centerline[centerline.length - 1];
}

function pointAfterEnd(centerline: GeoPoint[], thresholdStationNm: number, stationNm: number): GeoPoint {
  if (centerline.length < 2 || stationNm <= thresholdStationNm) {
    return pointAtStation(centerline, stationNm);
  }

  const last = centerline[centerline.length - 1];
  const prev = centerline[centerline.length - 2];
  return offsetPoint(last, bearingRad(prev, last), (stationNm - thresholdStationNm) * METERS_PER_NM);
}

export function buildSampledCenterline(
  source: PolylineGeometry3D,
  startStationNm: number,
  endStationNm: number,
  samplingStepNm: number,
): PolylineGeometry3D {
  const stations = sampleStationValues(startStationNm, endStationNm, samplingStepNm);
  const thresholdStationNm = source.geodesicLengthNm;
  const geoPositions = stations.map((stationNm) =>
    pointAfterEnd(source.geoPositions, thresholdStationNm, stationNm),
  );
  return {
    geoPositions,
    worldPositions: geoPositions.map(toCartesian),
    geodesicLengthNm: endStationNm - startStationNm,
    isArc: source.isArc,
  };
}

function primaryHalfWidthAtStation(
  stationNm: number,
  opts: Required<LnavFinalOeaOptions>,
): number {
  const taperStart = -opts.startBeforePfafNm;
  const taperEnd = opts.stableAfterPfafNm;
  if (stationNm >= taperEnd) return opts.stablePrimaryHalfWidthNm;
  const ratio = clamp((stationNm - taperStart) / (taperEnd - taperStart), 0, 1);
  return (
    opts.initialPrimaryHalfWidthNm +
    (opts.stablePrimaryHalfWidthNm - opts.initialPrimaryHalfWidthNm) * ratio
  );
}

export function buildVariableWidthRibbon(
  geometryId: string,
  centerline: PolylineGeometry3D,
  stations: number[],
  halfWidthAtStation: (stationNm: number) => number,
): VariableWidthRibbonGeometry {
  const leftGeoBoundary = centerline.geoPositions.map((point, index) =>
    offsetPoint(
      point,
      localBearing(centerline.geoPositions, index) - Math.PI / 2,
      halfWidthAtStation(stations[index]) * METERS_PER_NM,
    ),
  );
  const rightGeoBoundary = centerline.geoPositions.map((point, index) =>
    offsetPoint(
      point,
      localBearing(centerline.geoPositions, index) + Math.PI / 2,
      halfWidthAtStation(stations[index]) * METERS_PER_NM,
    ),
  );

  return {
    geometryId,
    leftBoundary: leftGeoBoundary.map(toCartesian),
    rightBoundary: rightGeoBoundary.map(toCartesian),
    leftGeoBoundary,
    rightGeoBoundary,
    halfWidthNmSamples: stations.map((stationNm) => ({
      stationNm,
      halfWidthNm: halfWidthAtStation(stationNm),
    })),
  };
}

export function buildLnavFinalOea(
  segment: ProcedureSegment,
  centerline: PolylineGeometry3D,
  options: LnavFinalOeaOptions = {},
): { geometry: LnavFinalOeaGeometry | null; diagnostics: BuildDiagnostic[] } {
  const opts = { ...DEFAULT_LNAV_FINAL_OEA_OPTIONS, ...options };
  const diagnostics: BuildDiagnostic[] = [];

  if (centerline.geoPositions.length < 2 || centerline.geodesicLengthNm <= 0) {
    diagnostics.push(
      diagnostic(
        "SOURCE_INCOMPLETE",
        `${segment.segmentId}: LNAV final OEA requires a positioned final centerline.`,
        "ERROR",
        segment.segmentId,
        segment.sourceRefs,
      ),
    );
    return { geometry: null, diagnostics };
  }

  if (segment.segmentType !== "FINAL_LNAV" && segment.segmentType !== "FINAL_LNAV_VNAV") {
    diagnostics.push(
      diagnostic(
        "MODE_COLLAPSED_TO_LNAV",
        `${segment.segmentId}: using LNAV OEA dimensions for ${segment.segmentType}.`,
        "INFO",
        segment.segmentId,
        segment.sourceRefs,
      ),
    );
  }

  const startStationNm = -opts.startBeforePfafNm;
  const endStationNm = centerline.geodesicLengthNm + opts.endAfterThresholdNm;
  const stations = sampleStationValues(startStationNm, endStationNm, opts.samplingStepNm);
  const oeaCenterline = buildSampledCenterline(
    centerline,
    startStationNm,
    endStationNm,
    opts.samplingStepNm,
  );
  const primary = buildVariableWidthRibbon(
    `${segment.segmentId}:lnav-oea-primary`,
    oeaCenterline,
    stations,
    (stationNm) => primaryHalfWidthAtStation(stationNm, opts),
  );
  const secondaryOuter = buildVariableWidthRibbon(
    `${segment.segmentId}:lnav-oea-secondary-outer`,
    oeaCenterline,
    stations,
    (stationNm) => primaryHalfWidthAtStation(stationNm, opts) + opts.secondaryWidthNm,
  );

  return {
    geometry: {
      geometryId: `${segment.segmentId}:lnav-oea`,
      segmentId: segment.segmentId,
      surfaceType: "LNAV_FINAL_OEA",
      centerline: oeaCenterline,
      primary,
      secondaryOuter,
      taper: {
        startStationNm,
        endStationNm: opts.stableAfterPfafNm,
        initialPrimaryHalfWidthNm: opts.initialPrimaryHalfWidthNm,
        stablePrimaryHalfWidthNm: opts.stablePrimaryHalfWidthNm,
        secondaryWidthNm: opts.secondaryWidthNm,
      },
    },
    diagnostics,
  };
}
