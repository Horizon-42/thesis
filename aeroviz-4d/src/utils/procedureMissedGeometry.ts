import type {
  BuildDiagnostic,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../data/procedurePackage";
import {
  FEET_TO_METERS,
  METERS_PER_NM,
  offsetPoint,
  toCartesian,
  toRadians,
  type CartesianPoint,
  type GeoPoint,
} from "./procedureGeoMath";
import type {
  LateralEnvelopeGeometry,
  SegmentGeometryBundle,
} from "./procedureSegmentGeometry";

const DEFAULT_CA_COURSE_GUIDE_LENGTH_NM = 3;
const DEFAULT_CA_CLIMB_GRADIENT_FT_PER_NM = 200;

export type MissedSectionSurfaceType =
  | "MISSED_SECTION1_ENVELOPE"
  | "MISSED_SECTION2_STRAIGHT_ENVELOPE";

export interface MissedSectionSurfaceGeometry {
  segmentId: string;
  surfaceType: MissedSectionSurfaceType;
  primary: LateralEnvelopeGeometry;
  secondaryOuter: LateralEnvelopeGeometry | null;
}

export interface MissedCourseGuideGeometry {
  segmentId: string;
  legId: string;
  legType: "CA";
  startFixId: string;
  courseDeg: number;
  guideLengthNm: number;
  requiredAltitudeFtMsl: number | null;
  constructionStatus: "COURSE_DIRECTION_ONLY";
  geoPositions: [GeoPoint, GeoPoint];
  worldPositions: [CartesianPoint, CartesianPoint];
}

export type MissedCaEndpointStatus =
  | "ESTIMATED_ENDPOINT"
  | "SOURCE_EXACT"
  | "INSUFFICIENT_CLIMB_MODEL";

export interface MissedCaEndpointGeometry {
  segmentId: string;
  legId: string;
  startFixId: string;
  courseDeg: number;
  startAltitudeFtMsl: number;
  targetAltitudeFtMsl: number;
  climbGradientFtPerNm: number;
  distanceNm: number;
  constructionStatus: MissedCaEndpointStatus;
  geoPositions: [GeoPoint, GeoPoint];
  worldPositions: [CartesianPoint, CartesianPoint];
  notes: string[];
}

export interface MissedCaEndpointOptions {
  climbGradientFtPerNm?: number;
  useDefaultClimbGradient?: boolean;
}

export interface MissedTurnDebugPointGeometry {
  segmentId: string;
  debugType: "TURNING_MISSED_ANCHOR";
  anchorFixId: string;
  triggerLegTypes: string[];
  constructionStatus: "DEBUG_MARKER_ONLY";
  geoPosition: GeoPoint;
  worldPosition: CartesianPoint;
}

function diagnostic(
  segment: ProcedureSegment,
  message: string,
): BuildDiagnostic {
  return {
    severity: "WARN",
    segmentId: segment.segmentId,
    code: "SOURCE_INCOMPLETE",
    message,
    sourceRefs: segment.sourceRefs,
  };
}

function legDiagnostic(
  segment: ProcedureSegment,
  leg: ProcedurePackageLeg,
  message: string,
  code: BuildDiagnostic["code"] = "SOURCE_INCOMPLETE",
): BuildDiagnostic {
  return {
    severity: "WARN",
    segmentId: segment.segmentId,
    legId: leg.legId,
    code,
    message,
    sourceRefs: leg.sourceRefs,
  };
}

function pointFromFix(fix: ProcedurePackageFix): GeoPoint | null {
  if (fix.latDeg === null || fix.lonDeg === null) return null;
  return {
    lonDeg: fix.lonDeg,
    latDeg: fix.latDeg,
    altM: (fix.altFtMsl ?? 0) * FEET_TO_METERS,
  };
}

function altitudeTargetFt(leg: ProcedurePackageLeg): number | null {
  const altitude = leg.requiredAltitude;
  if (!altitude) return null;
  if (typeof altitude.minFtMsl === "number" && Number.isFinite(altitude.minFtMsl)) {
    return altitude.minFtMsl;
  }
  if (typeof altitude.maxFtMsl === "number" && Number.isFinite(altitude.maxFtMsl)) {
    return altitude.maxFtMsl;
  }
  return null;
}

export function buildMissedSectionSurface(
  segment: ProcedureSegment,
  segmentGeometry: SegmentGeometryBundle,
): { geometry: MissedSectionSurfaceGeometry | null; diagnostics: BuildDiagnostic[] } {
  if (segment.segmentType !== "MISSED_S1" && segment.segmentType !== "MISSED_S2") {
    return { geometry: null, diagnostics: [] };
  }

  if (!segmentGeometry.primaryEnvelope) {
    return {
      geometry: null,
      diagnostics: [
        diagnostic(
          segment,
          `${segment.segmentId}: missed section surface requires a primary envelope.`,
        ),
      ],
    };
  }

  if (segment.segmentType === "MISSED_S2" && segmentGeometry.centerline.isArc) {
    return {
      geometry: null,
      diagnostics: [
        diagnostic(
          segment,
          `${segment.segmentId}: turning missed section 2 surface is not implemented yet.`,
        ),
      ],
    };
  }

  return {
    geometry: {
      segmentId: segment.segmentId,
      surfaceType:
        segment.segmentType === "MISSED_S1"
          ? "MISSED_SECTION1_ENVELOPE"
          : "MISSED_SECTION2_STRAIGHT_ENVELOPE",
      primary: segmentGeometry.primaryEnvelope,
      secondaryOuter: segmentGeometry.secondaryEnvelope ?? null,
    },
    diagnostics: [],
  };
}

export function buildMissedCourseGuides(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  fixes: Map<string, ProcedurePackageFix>,
  guideLengthNm = DEFAULT_CA_COURSE_GUIDE_LENGTH_NM,
): { geometries: MissedCourseGuideGeometry[]; diagnostics: BuildDiagnostic[] } {
  if (segment.segmentType !== "MISSED_S1" && segment.segmentType !== "MISSED_S2") {
    return { geometries: [], diagnostics: [] };
  }

  const diagnostics: BuildDiagnostic[] = [];
  const geometries: MissedCourseGuideGeometry[] = [];
  const segmentLegs = legs.filter((leg) => leg.segmentId === segment.segmentId);

  segmentLegs.forEach((leg) => {
    if (leg.legType !== "CA") return;

    const courseDeg = leg.outboundCourseDeg;
    if (typeof courseDeg !== "number" || !Number.isFinite(courseDeg)) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA course guide requires outbound course metadata.`,
        ),
      );
      return;
    }

    const startFixId = leg.startFixId ?? leg.endFixId;
    const startFix = startFixId ? fixes.get(startFixId) : undefined;
    const start = startFix ? pointFromFix(startFix) : null;
    if (!startFixId || !start) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA course guide requires a positioned start fix.`,
        ),
      );
      return;
    }

    const target = offsetPoint(
      start,
      toRadians(courseDeg),
      guideLengthNm * METERS_PER_NM,
    );
    const geoPositions: [GeoPoint, GeoPoint] = [start, target];

    geometries.push({
      segmentId: segment.segmentId,
      legId: leg.legId,
      legType: "CA",
      startFixId,
      courseDeg,
      guideLengthNm,
      requiredAltitudeFtMsl: altitudeTargetFt(leg),
      constructionStatus: "COURSE_DIRECTION_ONLY",
      geoPositions,
      worldPositions: [toCartesian(geoPositions[0]), toCartesian(geoPositions[1])],
    });
  });

  return { geometries, diagnostics };
}

function climbGradientFor(options: MissedCaEndpointOptions): number | null {
  if (
    typeof options.climbGradientFtPerNm === "number" &&
    Number.isFinite(options.climbGradientFtPerNm) &&
    options.climbGradientFtPerNm > 0
  ) {
    return options.climbGradientFtPerNm;
  }
  if (options.useDefaultClimbGradient === false) return null;
  return DEFAULT_CA_CLIMB_GRADIENT_FT_PER_NM;
}

export function buildMissedCaEndpoints(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  fixes: Map<string, ProcedurePackageFix>,
  options: MissedCaEndpointOptions = {},
): { geometries: MissedCaEndpointGeometry[]; diagnostics: BuildDiagnostic[] } {
  if (segment.segmentType !== "MISSED_S1" && segment.segmentType !== "MISSED_S2") {
    return { geometries: [], diagnostics: [] };
  }

  const diagnostics: BuildDiagnostic[] = [];
  const geometries: MissedCaEndpointGeometry[] = [];
  const segmentLegs = legs.filter((leg) => leg.segmentId === segment.segmentId);

  segmentLegs.forEach((leg) => {
    if (leg.legType !== "CA") return;

    const courseDeg = leg.outboundCourseDeg;
    if (typeof courseDeg !== "number" || !Number.isFinite(courseDeg)) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires outbound course metadata.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }

    const targetAltitudeFtMsl = altitudeTargetFt(leg);
    if (targetAltitudeFtMsl === null) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires a target altitude constraint.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }

    const startFixId = leg.startFixId ?? leg.endFixId;
    const startFix = startFixId ? fixes.get(startFixId) : undefined;
    const start = startFix ? pointFromFix(startFix) : null;
    if (!startFixId || !start) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires a positioned start fix.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }
    if (typeof startFix?.altFtMsl !== "number" || !Number.isFinite(startFix.altFtMsl)) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires start-fix elevation metadata.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }

    const climbGradientFtPerNm = climbGradientFor(options);
    if (climbGradientFtPerNm === null) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires an explicit climb gradient when default climb model is disabled.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }

    const altitudeDeltaFt = targetAltitudeFtMsl - startFix.altFtMsl;
    if (altitudeDeltaFt <= 0) {
      diagnostics.push(
        legDiagnostic(
          segment,
          leg,
          `${leg.legId}: CA endpoint requires target altitude above start-fix elevation.`,
          "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        ),
      );
      return;
    }

    const distanceNm = altitudeDeltaFt / climbGradientFtPerNm;
    const endpoint = {
      ...offsetPoint(start, toRadians(courseDeg), distanceNm * METERS_PER_NM),
      altM: targetAltitudeFtMsl * FEET_TO_METERS,
    };
    const geoPositions: [GeoPoint, GeoPoint] = [start, endpoint];
    const notes = options.climbGradientFtPerNm
      ? ["Endpoint estimated from explicit climb gradient; not certified source geometry."]
      : [`Endpoint estimated from default ${DEFAULT_CA_CLIMB_GRADIENT_FT_PER_NM} ft/NM climb model; not certified source geometry.`];

    geometries.push({
      segmentId: segment.segmentId,
      legId: leg.legId,
      startFixId,
      courseDeg,
      startAltitudeFtMsl: startFix.altFtMsl,
      targetAltitudeFtMsl,
      climbGradientFtPerNm,
      distanceNm,
      constructionStatus: "ESTIMATED_ENDPOINT",
      geoPositions,
      worldPositions: [toCartesian(geoPositions[0]), toCartesian(geoPositions[1])],
      notes,
    });
  });

  return { geometries, diagnostics };
}

export function buildMissedTurnDebugPoint(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  fixes: Map<string, ProcedurePackageFix>,
): { geometry: MissedTurnDebugPointGeometry | null; diagnostics: BuildDiagnostic[] } {
  if (segment.segmentType !== "MISSED_S2" || !segment.constructionFlags.isTurningMissedApproach) {
    return { geometry: null, diagnostics: [] };
  }

  const segmentLegs = legs.filter((leg) => leg.segmentId === segment.segmentId);
  const triggerLegTypes = [
    ...new Set(
      segmentLegs
        .map((leg) => leg.legType)
        .filter((legType) => legType === "HM" || legType === "HA" || legType === "HF" || legType === "RF"),
    ),
  ];
  const anchorFixId =
    segment.startFixId ??
    segmentLegs.find((leg) => leg.startFixId)?.startFixId ??
    segmentLegs.find((leg) => leg.endFixId)?.endFixId ??
    null;
  const anchorFix = anchorFixId ? fixes.get(anchorFixId) : undefined;
  const anchor = anchorFix ? pointFromFix(anchorFix) : null;

  if (!anchorFixId || !anchor) {
    return {
      geometry: null,
      diagnostics: [
        diagnostic(
          segment,
          `${segment.segmentId}: turning missed debug anchor requires a positioned section 2 start fix.`,
        ),
      ],
    };
  }

  return {
    geometry: {
      segmentId: segment.segmentId,
      debugType: "TURNING_MISSED_ANCHOR",
      anchorFixId,
      triggerLegTypes,
      constructionStatus: "DEBUG_MARKER_ONLY",
      geoPosition: anchor,
      worldPosition: toCartesian(anchor),
    },
    diagnostics: [],
  };
}
