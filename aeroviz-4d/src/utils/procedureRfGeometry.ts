import type {
  BuildDiagnostic,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  SourceRef,
} from "../data/procedurePackage";
import type { PolylineGeometry3D } from "./procedureSegmentGeometry";
import {
  FEET_TO_METERS,
  METERS_PER_NM,
  bearingRad,
  distanceNm,
  offsetPoint,
  toCartesian,
  type GeoPoint,
} from "./procedureGeoMath";

const RADIUS_TOLERANCE_NM = 0.05;

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

function normalizeAngle(value: number): number {
  return ((value % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
}

function localCircleAngle(center: GeoPoint, point: GeoPoint): number {
  const distanceM = distanceNm(center, point) * METERS_PER_NM;
  const bearing = bearingRad(center, point);
  const eastM = Math.sin(bearing) * distanceM;
  const northM = Math.cos(bearing) * distanceM;
  return normalizeAngle(Math.atan2(northM, eastM));
}

function circularSweepRad(
  startAngle: number,
  endAngle: number,
  turnDirection: "LEFT" | "RIGHT",
): number {
  if (turnDirection === "LEFT") {
    return normalizeAngle(endAngle - startAngle);
  }
  return -normalizeAngle(startAngle - endAngle);
}

function pointOnCircle(
  center: GeoPoint,
  angle: number,
  radiusNm: number,
  altitudeM: number,
): GeoPoint {
  const eastM = Math.cos(angle) * radiusNm * METERS_PER_NM;
  const northM = Math.sin(angle) * radiusNm * METERS_PER_NM;
  const bearing = Math.atan2(eastM, northM);
  return offsetPoint(
    { ...center, altM: altitudeM },
    bearing,
    Math.hypot(eastM, northM),
  );
}

export function buildRfLeg(
  leg: ProcedurePackageLeg,
  fixes: Map<string, ProcedurePackageFix>,
  ctx: { samplingStepNm: number },
): { geometry: PolylineGeometry3D | null; diagnostics: BuildDiagnostic[] } {
  const diagnostics: BuildDiagnostic[] = [];

  if (leg.legType !== "RF") {
    diagnostics.push(
      diagnostic(
        "UNSUPPORTED_LEG_TYPE",
        `${leg.legId}: buildRfLeg only accepts RF legs; received ${leg.legType}.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  if (
    leg.arcRadiusNm === undefined ||
    leg.centerLatDeg === undefined ||
    leg.centerLonDeg === undefined
  ) {
    diagnostics.push(
      diagnostic(
        "RF_RADIUS_MISSING",
        `${leg.legId}: RF geometry requires arcRadiusNm, centerLatDeg, and centerLonDeg.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  const arcRadiusNm = leg.arcRadiusNm;
  const centerLatDeg = leg.centerLatDeg;
  const centerLonDeg = leg.centerLonDeg;

  if (!leg.turnDirection) {
    diagnostics.push(
      diagnostic(
        "SOURCE_INCOMPLETE",
        `${leg.legId}: RF geometry requires an explicit turnDirection.`,
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
        `${leg.legId}: RF geometry requires positioned start and end fixes.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  const center: GeoPoint = {
    lonDeg: centerLonDeg,
    latDeg: centerLatDeg,
    altM: start.altM,
  };
  const startRadiusErrorNm = Math.abs(distanceNm(center, start) - arcRadiusNm);
  const endRadiusErrorNm = Math.abs(distanceNm(center, end) - arcRadiusNm);

  if (startRadiusErrorNm > RADIUS_TOLERANCE_NM || endRadiusErrorNm > RADIUS_TOLERANCE_NM) {
    diagnostics.push(
      diagnostic(
        "SOURCE_INCOMPLETE",
        `${leg.legId}: RF start/end fixes are inconsistent with the supplied center and radius.`,
        "ERROR",
        leg.segmentId,
        leg.legId,
        sourceRefsFor(leg),
      ),
    );
    return { geometry: null, diagnostics };
  }

  const startAngle = localCircleAngle(center, start);
  const endAngle = localCircleAngle(center, end);
  const sweepRad = circularSweepRad(startAngle, endAngle, leg.turnDirection);
  const arcLengthNm = Math.abs(sweepRad) * arcRadiusNm;
  const sampleCount = Math.max(1, Math.ceil(arcLengthNm / Math.max(ctx.samplingStepNm, 0.01)));
  const geoPositions = Array.from({ length: sampleCount + 1 }, (_, index) => {
    if (index === 0) return start;
    if (index === sampleCount) return end;
    const ratio = index / sampleCount;
    return pointOnCircle(
      center,
      startAngle + sweepRad * ratio,
      arcRadiusNm,
      start.altM + (end.altM - start.altM) * ratio,
    );
  });

  return {
    geometry: {
      worldPositions: geoPositions.map(toCartesian),
      geoPositions,
      geodesicLengthNm: arcLengthNm,
      isArc: true,
    },
    diagnostics,
  };
}
