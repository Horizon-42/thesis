import type {
  BuildDiagnostic,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../data/procedurePackage";
import {
  FEET_TO_METERS,
  METERS_PER_NM,
  distanceNm,
  interpolateGreatCircle,
  offsetPoint,
  toCartesian,
  toRadians,
  type CartesianPoint,
  type GeoPoint,
} from "./procedureGeoMath";
import type {
  LateralEnvelopeGeometry,
  PolylineGeometry3D,
  SegmentGeometryBundle,
} from "./procedureSegmentGeometry";
import {
  buildStraightEnvelope,
  computeStationAxis,
} from "./procedureSegmentGeometry";

const DEFAULT_CA_COURSE_GUIDE_LENGTH_NM = 3;
const DEFAULT_CA_CLIMB_GRADIENT_FT_PER_NM = 200;

export type MissedSectionSurfaceType =
  | "MISSED_SECTION1_ENVELOPE"
  | "MISSED_SECTION2_STRAIGHT_ENVELOPE";

export interface MissedSectionSurfaceGeometry {
  segmentId: string;
  surfaceType: MissedSectionSurfaceType;
  constructionStatus: "SOURCE_BACKED" | "ESTIMATED_CA";
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

export interface MissedCaCenterlineGeometry extends PolylineGeometry3D {
  segmentId: string;
  legId: string;
  sourceEndpointStatus: MissedCaEndpointStatus;
  constructionStatus: "ESTIMATED_CENTERLINE";
  notes: string[];
}

export interface MissedCaCenterlineOptions {
  samplingStepNm?: number;
}

export interface MissedCaSegmentGeometryResult {
  geometry: SegmentGeometryBundle;
  backfilled: boolean;
  diagnostics: BuildDiagnostic[];
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
      constructionStatus: segmentGeometry.diagnostics.some(
        (diagnostic) => diagnostic.code === "ESTIMATED_CA_GEOMETRY",
      )
        ? "ESTIMATED_CA"
        : "SOURCE_BACKED",
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

export function buildMissedCaCenterlines(
  endpoints: MissedCaEndpointGeometry[],
  options: MissedCaCenterlineOptions = {},
): MissedCaCenterlineGeometry[] {
  const samplingStepNm = Math.max(options.samplingStepNm ?? 0.25, 0.01);

  return endpoints.map((endpoint) => {
    const [start, end] = endpoint.geoPositions;
    const lengthNm = distanceNm(start, end);
    const sampleCount = Math.max(1, Math.ceil(lengthNm / samplingStepNm));
    const geoPositions = Array.from({ length: sampleCount + 1 }, (_, index) =>
      interpolateGreatCircle(start, end, index / sampleCount),
    );
    geoPositions[geoPositions.length - 1] = end;

    return {
      segmentId: endpoint.segmentId,
      legId: endpoint.legId,
      sourceEndpointStatus: endpoint.constructionStatus,
      constructionStatus: "ESTIMATED_CENTERLINE",
      notes: [
        ...endpoint.notes,
        "Centerline is sampled from estimated CA endpoint geometry; not certified source geometry.",
      ],
      geoPositions,
      worldPositions: geoPositions.map(toCartesian),
      geodesicLengthNm: lengthNm,
      isArc: false,
    };
  });
}

function isMissedSegment(segment: ProcedureSegment): boolean {
  return segment.segmentType === "MISSED_S1" || segment.segmentType === "MISSED_S2";
}

function filterBackfilledCaDiagnostics(
  baseGeometry: SegmentGeometryBundle,
  caLegIds: Set<string>,
): BuildDiagnostic[] {
  return baseGeometry.diagnostics.filter((diagnostic) => {
    if (diagnostic.code !== "UNSUPPORTED_LEG_TYPE") return true;
    if (diagnostic.legId && caLegIds.has(diagnostic.legId)) return false;
    if (!diagnostic.legId && diagnostic.segmentId === baseGeometry.segmentId) return false;
    return true;
  });
}

export function buildMissedCaSegmentGeometry(
  segment: ProcedureSegment,
  legs: ProcedurePackageLeg[],
  baseGeometry: SegmentGeometryBundle,
  centerlines: MissedCaCenterlineGeometry[],
): MissedCaSegmentGeometryResult {
  if (!isMissedSegment(segment) || centerlines.length === 0) {
    return { geometry: baseGeometry, backfilled: false, diagnostics: [] };
  }

  const segmentLegs = legs.filter((leg) => leg.segmentId === segment.segmentId);
  const geometryLegs = segmentLegs.filter((leg) => leg.legType !== "IF");
  const caLegs = geometryLegs.filter((leg) => leg.legType === "CA");
  if (geometryLegs.length === 0 || caLegs.length !== geometryLegs.length) {
    return { geometry: baseGeometry, backfilled: false, diagnostics: [] };
  }

  const centerlineByLegId = new Map(centerlines.map((centerline) => [centerline.legId, centerline]));
  const orderedCenterlines = caLegs
    .map((leg) => centerlineByLegId.get(leg.legId))
    .filter((centerline): centerline is MissedCaCenterlineGeometry => centerline !== undefined);

  if (orderedCenterlines.length !== caLegs.length) {
    return { geometry: baseGeometry, backfilled: false, diagnostics: [] };
  }

  const geoPositions = orderedCenterlines.flatMap((centerline, index) =>
    centerline.geoPositions.slice(index === 0 ? 0 : 1),
  );
  if (geoPositions.length < 2) {
    return { geometry: baseGeometry, backfilled: false, diagnostics: [] };
  }

  const centerline: PolylineGeometry3D = {
    geoPositions,
    worldPositions: geoPositions.map(toCartesian),
    geodesicLengthNm: orderedCenterlines.reduce(
      (sum, centerlineGeometry) => sum + centerlineGeometry.geodesicLengthNm,
      0,
    ),
    isArc: false,
  };
  const caLegIds = new Set(caLegs.map((leg) => leg.legId));
  const estimatedDiagnostic: BuildDiagnostic = {
    severity: "WARN",
    segmentId: segment.segmentId,
    legId: caLegs.length === 1 ? caLegs[0].legId : undefined,
    code: "ESTIMATED_CA_GEOMETRY",
    message:
      `${segment.segmentId}: CA missed segment geometry was estimated from course, altitude, and climb model; ` +
      "it is debug geometry and not certified source protection.",
    sourceRefs: caLegs.flatMap((leg) => leg.sourceRefs),
  };
  const diagnostics = [
    ...filterBackfilledCaDiagnostics(baseGeometry, caLegIds),
    estimatedDiagnostic,
  ];

  return {
    geometry: {
      segmentId: segment.segmentId,
      centerline,
      stationAxis: computeStationAxis(centerline),
      primaryEnvelope: buildStraightEnvelope(
        `${segment.segmentId}:ca-estimated-primary`,
        "PRIMARY",
        centerline,
        segment.xttNm * 2,
      ),
      secondaryEnvelope: segment.secondaryEnabled
        ? buildStraightEnvelope(
            `${segment.segmentId}:ca-estimated-secondary`,
            "SECONDARY",
            centerline,
            segment.xttNm * 3,
          )
        : undefined,
      turnJunctions: [],
      diagnostics,
    },
    backfilled: true,
    diagnostics: [estimatedDiagnostic],
  };
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
