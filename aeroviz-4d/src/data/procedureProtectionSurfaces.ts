import type {
  BuildDiagnostic,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "./procedurePackage";
import type {
  LnavFinalOeaGeometry,
  LnavVnavOcsGeometry,
  PrecisionFinalSurfaceGeometry,
} from "../utils/procedureSurfaceGeometry";
import type {
  MissedConnectorSurfaceGeometry,
  MissedSectionSurfaceGeometry,
  MissedTurnDebugPrimitiveGeometry,
} from "../utils/procedureMissedGeometry";
import {
  computeStationAxis,
  type LateralEnvelopeGeometry,
  type PolylineGeometry3D,
  type SegmentGeometryBundle,
} from "../utils/procedureSegmentGeometry";
import {
  FEET_TO_METERS,
  distanceNm,
  toCartesian,
  type GeoPoint,
} from "../utils/procedureGeoMath";
import {
  buildVariableWidthRibbon,
  type VariableWidthRibbonGeometry,
} from "../utils/procedureSurfaceGeometry";

export type ProtectionSurfaceKind =
  | "FINAL_LNAV_OEA"
  | "FINAL_LNAV_VNAV_OCS"
  | "FINAL_PRECISION_DEBUG"
  | "MISSED_SECTION_1"
  | "MISSED_SECTION_2_STRAIGHT"
  | "MISSED_CONNECTOR"
  | "TURNING_MISSED_DEBUG";

export type ProtectionSurfaceStatus =
  | "SOURCE_BACKED"
  | "TERPS_ESTIMATE"
  | "DISPLAY_ESTIMATE"
  | "DEBUG_ESTIMATE"
  | "MISSING_SOURCE";

export type ProtectionSurfaceVerticalKind =
  | "NONE"
  | "ALTITUDE_PROFILE"
  | "OCS";

export type ProtectionSurfaceVerticalOrigin =
  | "SOURCE"
  | "GPA_TCH"
  | "MISSED_CLIMB"
  | "CENTERLINE_ALTITUDE_ONLY"
  | "ESTIMATED_CLIMB";

export type ProtectionSurfaceRibbon =
  | VariableWidthRibbonGeometry
  | LateralEnvelopeGeometry;

export interface ProtectionSurfaceWidthSample {
  stationNm: number;
  primaryHalfWidthNm: number;
  secondaryOuterHalfWidthNm?: number;
}

export interface ProtectionSurfaceVerticalSample {
  stationNm: number;
  altitudeFtMsl: number;
}

export interface ProcedureProtectionSurface {
  surfaceId: string;
  segmentId: string;
  sourceLegIds: string[];
  kind: ProtectionSurfaceKind;
  status: ProtectionSurfaceStatus;
  centerline: PolylineGeometry3D;
  lateral: {
    primary: ProtectionSurfaceRibbon;
    secondaryOuter: ProtectionSurfaceRibbon | null;
    widthSamples: ProtectionSurfaceWidthSample[];
    rule: string;
    notes: string[];
  };
  vertical: {
    kind: ProtectionSurfaceVerticalKind;
    origin: ProtectionSurfaceVerticalOrigin;
    samples: ProtectionSurfaceVerticalSample[];
    slopeFtPerNm?: number;
    notes: string[];
  };
  diagnostics: BuildDiagnostic[];
}

export interface ProtectionSurfaceSegmentInput {
  segment: ProcedureSegment;
  legs: ProcedurePackageLeg[];
  segmentGeometry: SegmentGeometryBundle;
  finalOea: LnavFinalOeaGeometry | null;
  lnavVnavOcs: LnavVnavOcsGeometry | null;
  precisionFinalSurfaces: PrecisionFinalSurfaceGeometry[];
  missedSectionSurface: MissedSectionSurfaceGeometry | null;
  missedTurnDebugPrimitives: MissedTurnDebugPrimitiveGeometry[];
  diagnostics: BuildDiagnostic[];
}

export interface ProtectionSurfaceBranchInput {
  segmentBundles: ProtectionSurfaceSegmentInput[];
  missedConnectorSurfaces: MissedConnectorSurfaceGeometry[];
}

const TURNING_MISSED_DEBUG_RIBBON_HALF_WIDTH_NM = 0.03;

function nearestSecondaryHalfWidth(
  samples: ProtectionSurfaceRibbon["halfWidthNmSamples"],
  stationNm: number,
): number | undefined {
  if (samples.length === 0) return undefined;
  return samples.reduce((nearest, candidate) =>
    Math.abs(candidate.stationNm - stationNm) < Math.abs(nearest.stationNm - stationNm)
      ? candidate
      : nearest,
  ).halfWidthNm;
}

function widthSamplesFromRibbons(
  primary: ProtectionSurfaceRibbon,
  secondaryOuter: ProtectionSurfaceRibbon | null,
): ProtectionSurfaceWidthSample[] {
  return primary.halfWidthNmSamples.map((sample) => {
    const secondaryOuterHalfWidthNm = secondaryOuter
      ? nearestSecondaryHalfWidth(secondaryOuter.halfWidthNmSamples, sample.stationNm)
      : undefined;
    return {
      stationNm: sample.stationNm,
      primaryHalfWidthNm: sample.halfWidthNm,
      ...(secondaryOuterHalfWidthNm === undefined ? {} : { secondaryOuterHalfWidthNm }),
    };
  });
}

function sourceLegIds(legs: ProcedurePackageLeg[]): string[] {
  return legs.map((leg) => leg.legId);
}

function verticalSamplesFromCenterline(
  centerline: PolylineGeometry3D,
): ProtectionSurfaceVerticalSample[] {
  const stationAxis = computeStationAxis(centerline);
  return stationAxis.samples.map((sample) => ({
    stationNm: sample.stationNm,
    altitudeFtMsl: sample.geoPosition.altM / FEET_TO_METERS,
  }));
}

function polylineFromGeoPositions(geoPositions: GeoPoint[]): PolylineGeometry3D {
  let geodesicLengthNm = 0;
  geoPositions.forEach((point, index) => {
    if (index > 0) {
      geodesicLengthNm += distanceNm(geoPositions[index - 1], point);
    }
  });
  return {
    geoPositions,
    worldPositions: geoPositions.map(toCartesian),
    geodesicLengthNm,
    isArc: false,
  };
}

function closedDebugBoundaryRibbon(
  geometryId: string,
  boundary: GeoPoint[],
): VariableWidthRibbonGeometry | null {
  if (boundary.length < 4) return null;
  const uniqueBoundary =
    distanceNm(boundary[0], boundary[boundary.length - 1]) < 1e-5
      ? boundary.slice(0, -1)
      : boundary;
  if (uniqueBoundary.length < 4) return null;

  const splitIndex = Math.floor(uniqueBoundary.length / 2);
  const leftGeoBoundary = uniqueBoundary.slice(0, splitIndex + 1);
  const rightGeoBoundary = uniqueBoundary.slice(splitIndex + 1).reverse();

  return {
    geometryId,
    leftBoundary: leftGeoBoundary.map(toCartesian),
    rightBoundary: rightGeoBoundary.map(toCartesian),
    leftGeoBoundary,
    rightGeoBoundary,
    halfWidthNmSamples: [],
  };
}

function debugRibbonForTurnPrimitive(
  primitive: MissedTurnDebugPrimitiveGeometry,
): VariableWidthRibbonGeometry | null {
  if (
    primitive.debugType === "TIA_BOUNDARY" ||
    primitive.debugType === "WIND_SPIRAL"
  ) {
    return closedDebugBoundaryRibbon(`${primitive.primitiveId}:debug-area`, primitive.geoPositions);
  }

  if (primitive.geoPositions.length < 2) return null;
  const centerline = polylineFromGeoPositions(primitive.geoPositions);
  const stationAxis = computeStationAxis(centerline);
  return buildVariableWidthRibbon(
    `${primitive.primitiveId}:debug-ribbon`,
    centerline,
    stationAxis.samples.map((sample) => sample.stationNm),
    () => TURNING_MISSED_DEBUG_RIBBON_HALF_WIDTH_NM,
  );
}

function buildFinalOeaProtectionSurface(
  segmentBundle: ProtectionSurfaceSegmentInput,
  finalOea: LnavFinalOeaGeometry,
): ProcedureProtectionSurface {
  return {
    surfaceId: finalOea.geometryId,
    segmentId: finalOea.segmentId,
    sourceLegIds: sourceLegIds(segmentBundle.legs),
    kind: "FINAL_LNAV_OEA",
    status: "SOURCE_BACKED",
    centerline: finalOea.centerline,
    lateral: {
      primary: finalOea.primary,
      secondaryOuter: finalOea.secondaryOuter,
      widthSamples: widthSamplesFromRibbons(finalOea.primary, finalOea.secondaryOuter),
      rule: "LNAV final OEA taper from the coded final segment centerline.",
      notes: [
        `Primary half-width tapers from ${finalOea.taper.initialPrimaryHalfWidthNm} NM to ${finalOea.taper.stablePrimaryHalfWidthNm} NM.`,
        `Secondary outer boundary adds ${finalOea.taper.secondaryWidthNm} NM outside primary.`,
      ],
    },
    vertical: {
      kind: "NONE",
      origin: "SOURCE",
      samples: [],
      notes: ["LNAV final OEA is represented here as a lateral footprint, not a sloping OCS."],
    },
    diagnostics: segmentBundle.diagnostics,
  };
}

function buildLnavVnavProtectionSurface(
  segmentBundle: ProtectionSurfaceSegmentInput,
  ocs: LnavVnavOcsGeometry,
): ProcedureProtectionSurface {
  return {
    surfaceId: ocs.geometryId,
    segmentId: ocs.segmentId,
    sourceLegIds: sourceLegIds(segmentBundle.legs),
    kind: "FINAL_LNAV_VNAV_OCS",
    status: "TERPS_ESTIMATE",
    centerline: ocs.centerline,
    lateral: {
      primary: ocs.primary,
      secondaryOuter: ocs.secondaryOuter,
      widthSamples: widthSamplesFromRibbons(ocs.primary, ocs.secondaryOuter),
      rule: "LNAV/VNAV OCS lateral limits sampled from the LNAV final OEA width model.",
      notes: ocs.notes,
    },
    vertical: {
      kind: "OCS",
      origin: "GPA_TCH",
      samples: ocs.verticalProfile.samples,
      notes: [
        `GPA ${ocs.verticalProfile.gpaDeg} deg with TCH ${ocs.verticalProfile.tchFt} ft.`,
        "Simplified LNAV/VNAV OCS adapter; VEB-specific certified construction remains future work.",
      ],
    },
    diagnostics: segmentBundle.diagnostics,
  };
}

function buildPrecisionProtectionSurface(
  segmentBundle: ProtectionSurfaceSegmentInput,
  surface: PrecisionFinalSurfaceGeometry,
): ProcedureProtectionSurface {
  return {
    surfaceId: surface.geometryId,
    segmentId: surface.segmentId,
    sourceLegIds: sourceLegIds(segmentBundle.legs),
    kind: "FINAL_PRECISION_DEBUG",
    status: "DEBUG_ESTIMATE",
    centerline: surface.centerline,
    lateral: {
      primary: surface.ribbon,
      secondaryOuter: null,
      widthSamples: widthSamplesFromRibbons(surface.ribbon, null),
      rule: `${surface.surfaceType} debug width uses scaled LNAV outer lateral samples.`,
      notes: surface.notes,
    },
    vertical: {
      kind: "OCS",
      origin: "GPA_TCH",
      samples: verticalSamplesFromCenterline(surface.centerline),
      notes: [
        `GPA ${surface.verticalProfile.gpaDeg} deg with TCH ${surface.verticalProfile.tchFt} ft.`,
        "Debug estimate only; certified W/X/Y construction remains future work.",
      ],
    },
    diagnostics: segmentBundle.diagnostics,
  };
}

function missedSectionKind(surface: MissedSectionSurfaceGeometry): ProtectionSurfaceKind {
  return surface.sectionKind === "SECTION_1"
    ? "MISSED_SECTION_1"
    : "MISSED_SECTION_2_STRAIGHT";
}

function missedSectionStatus(surface: MissedSectionSurfaceGeometry): ProtectionSurfaceStatus {
  return surface.constructionStatus === "SOURCE_BACKED"
    ? "SOURCE_BACKED"
    : "DISPLAY_ESTIMATE";
}

function buildMissedSectionProtectionSurface(
  segmentBundle: ProtectionSurfaceSegmentInput,
  surface: MissedSectionSurfaceGeometry,
): ProcedureProtectionSurface {
  const slopeFtPerNm = surface.verticalProfile.climbGradientFtPerNm ?? undefined;
  return {
    surfaceId: `${surface.segmentId}:${surface.surfaceType.toLowerCase()}`,
    segmentId: surface.segmentId,
    sourceLegIds: sourceLegIds(segmentBundle.legs),
    kind: missedSectionKind(surface),
    status: missedSectionStatus(surface),
    centerline: segmentBundle.segmentGeometry.centerline,
    lateral: {
      primary: surface.primary,
      secondaryOuter: surface.secondaryOuter,
      widthSamples: widthSamplesFromRibbons(surface.primary, surface.secondaryOuter),
      rule: `${surface.lateralWidthRule.ruleId}: ${surface.lateralWidthRule.transitionStatus}`,
      notes: [
        ...surface.lateralWidthRule.notes,
        surface.constructionStatus === "ESTIMATED_CA"
          ? "CA subsection centerline was estimated from course, altitude target, and climb model."
          : "Centerline was built from source-coded missed approach geometry.",
      ],
    },
    vertical: {
      kind: "OCS",
      origin: surface.verticalProfile.constructionStatus === "SOURCE_CLIMB_GRADIENT"
        ? "MISSED_CLIMB"
        : "CENTERLINE_ALTITUDE_ONLY",
      samples: surface.verticalProfile.samples,
      ...(slopeFtPerNm === undefined ? {} : { slopeFtPerNm }),
      notes: [
        surface.verticalProfile.constructionStatus === "SOURCE_CLIMB_GRADIENT"
          ? "Vertical samples use the missed approach climb gradient."
          : "No explicit climb gradient was available; samples follow centerline altitude only.",
      ],
    },
    diagnostics: segmentBundle.diagnostics,
  };
}

function buildConnectorProtectionSurface(
  surface: MissedConnectorSurfaceGeometry,
): ProcedureProtectionSurface {
  return {
    surfaceId: surface.surfaceId,
    segmentId: surface.sourceSegmentId,
    sourceLegIds: [surface.sourceLegId],
    kind: "MISSED_CONNECTOR",
    status: "DISPLAY_ESTIMATE",
    centerline: surface.centerline,
    lateral: {
      primary: surface.primary,
      secondaryOuter: surface.secondaryOuter,
      widthSamples: widthSamplesFromRibbons(surface.primary, surface.secondaryOuter),
      rule:
        `${surface.lateralWidthRule.ruleId}: CA endpoint to MAHF/HOLD constant terminal missed width ` +
        surface.lateralWidthRule.transitionStatus,
      notes: [
        `Target fix ${surface.targetFixIdent} (${surface.targetFixRole}).`,
        ...surface.notes,
      ],
    },
    vertical: {
      kind: "ALTITUDE_PROFILE",
      origin: "ESTIMATED_CLIMB",
      samples: surface.verticalProfile.samples,
      notes: [
        surface.verticalProfile.constructionStatus,
        "Connector vertical samples are continuity estimates, not certified OCS construction.",
      ],
    },
    diagnostics: [],
  };
}

function buildTurningMissedDebugProtectionSurface(
  segmentBundle: ProtectionSurfaceSegmentInput,
  primitive: MissedTurnDebugPrimitiveGeometry,
): ProcedureProtectionSurface | null {
  const primary = debugRibbonForTurnPrimitive(primitive);
  if (!primary) return null;
  const centerline = polylineFromGeoPositions(primitive.geoPositions);
  return {
    surfaceId: primitive.primitiveId,
    segmentId: primitive.segmentId,
    sourceLegIds: [primitive.legId],
    kind: "TURNING_MISSED_DEBUG",
    status: "DEBUG_ESTIMATE",
    centerline,
    lateral: {
      primary,
      secondaryOuter: null,
      widthSamples: widthSamplesFromRibbons(primary, null),
      rule: `${primitive.debugType}: debug-only turning missed placeholder; not certified TIA, wind spiral, or protected-area construction.`,
      notes: [
        ...primitive.notes,
        `Turn trigger ${primitive.turnTrigger}; turn case ${primitive.turnCase}.`,
        "Closed TIA/wind placeholders are rendered as debug areas; baseline and nominal-turn placeholders are rendered as narrow debug ribbons.",
      ],
    },
    vertical: {
      kind: "ALTITUDE_PROFILE",
      origin: "CENTERLINE_ALTITUDE_ONLY",
      samples: verticalSamplesFromCenterline(centerline),
      notes: [
        "Altitude follows the debug primitive source points only; no TERPS vertical construction is implied.",
      ],
    },
    diagnostics: segmentBundle.diagnostics,
  };
}

export function buildSegmentProtectionSurfaces(
  segmentBundle: ProtectionSurfaceSegmentInput,
): ProcedureProtectionSurface[] {
  const surfaces: ProcedureProtectionSurface[] = [];
  if (segmentBundle.finalOea) {
    surfaces.push(buildFinalOeaProtectionSurface(segmentBundle, segmentBundle.finalOea));
  }
  if (segmentBundle.lnavVnavOcs) {
    surfaces.push(buildLnavVnavProtectionSurface(segmentBundle, segmentBundle.lnavVnavOcs));
  }
  segmentBundle.precisionFinalSurfaces.forEach((surface) => {
    surfaces.push(buildPrecisionProtectionSurface(segmentBundle, surface));
  });
  if (segmentBundle.missedSectionSurface) {
    surfaces.push(
      buildMissedSectionProtectionSurface(segmentBundle, segmentBundle.missedSectionSurface),
    );
  }
  segmentBundle.missedTurnDebugPrimitives.forEach((primitive) => {
    const debugSurface = buildTurningMissedDebugProtectionSurface(segmentBundle, primitive);
    if (debugSurface) surfaces.push(debugSurface);
  });
  return surfaces;
}

export function buildBranchProtectionSurfaces(
  input: ProtectionSurfaceBranchInput,
): ProcedureProtectionSurface[] {
  return [
    ...input.segmentBundles.flatMap(buildSegmentProtectionSurfaces),
    ...input.missedConnectorSurfaces.map(buildConnectorProtectionSurface),
  ];
}
