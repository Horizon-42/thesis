import { altitudeConstraintReferenceFt } from "./altitudeConstraints";
import type {
  BranchGeometryBundle,
  ProcedureSegmentRenderBundle,
} from "./procedureRenderBundle";
import type { ProcedurePackage } from "./procedurePackage";
import type { ProcedureProtectionSurface } from "./procedureProtectionSurfaces";
import { distanceNm, type GeoPoint } from "../utils/procedureGeoMath";
import {
  buildVariableWidthRibbon,
  type VariableWidthRibbonGeometry,
} from "../utils/procedureSurfaceGeometry";

const FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM = 0.15;
const FINAL_VERTICAL_REFERENCE_PROTECTION_WIDTH_RATIO = 1;

export interface SegmentVerticalProfilePoint extends GeoPoint {
  fixIdent: string;
  altitudeFtMsl: number;
  halfWidthNm: number;
  segmentId?: string;
}

export interface SegmentVerticalProfileSection {
  sectionId: string;
  points: SegmentVerticalProfilePoint[];
  segmentIds: string[];
  segmentTypes: string[];
}

function isFinalSegment(segmentType: string): boolean {
  return segmentType.startsWith("FINAL");
}

function nearestHalfWidthNm(
  samples: Array<{ stationNm: number; halfWidthNm: number }>,
  stationNm: number,
): number {
  const nearest = samples.reduce<{ stationNm: number; halfWidthNm: number } | null>(
    (best, sample) => {
      if (!best) return sample;
      return Math.abs(sample.stationNm - stationNm) < Math.abs(best.stationNm - stationNm)
        ? sample
        : best;
    },
    null,
  );
  return nearest?.halfWidthNm ?? FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM * 2;
}

function protectionSurfaceForSegment(
  surfaces: ProcedureProtectionSurface[],
  segmentId: string,
  kind: ProcedureProtectionSurface["kind"],
): ProcedureProtectionSurface | null {
  return surfaces.find((surface) => surface.segmentId === segmentId && surface.kind === kind) ?? null;
}

function primaryProtectionWidthSamples(
  segmentBundle: ProcedureSegmentRenderBundle,
  protectionSurfaces: ProcedureProtectionSurface[],
): Array<{ stationNm: number; halfWidthNm: number }> {
  const finalOeaSurface = protectionSurfaceForSegment(
    protectionSurfaces,
    segmentBundle.segment.segmentId,
    "FINAL_LNAV_OEA",
  );
  return (
    finalOeaSurface?.lateral.primary.halfWidthNmSamples ??
    segmentBundle.segmentGeometry.primaryEnvelope?.halfWidthNmSamples ??
    []
  );
}

/**
 * Builds the final-approach vertical reference as domain geometry.
 *
 * LNAV/VNAV uses the source-backed OCS centerline when available. Otherwise we
 * reconstruct the GPA/TCH plane from the segment station axis so every view
 * reads the same vertical reference instead of recomputing its own estimate.
 */
export function finalVerticalReferencePoints(
  segmentBundle: ProcedureSegmentRenderBundle,
  protectionSurfaces: ProcedureProtectionSurface[] = [],
): GeoPoint[] {
  if (!isFinalSegment(segmentBundle.segment.segmentType)) return [];
  const gpaDeg = segmentBundle.segment.verticalRule?.gpaDeg;
  if (typeof gpaDeg !== "number" || !Number.isFinite(gpaDeg) || gpaDeg <= 0) return [];

  const lnavVnavSurface = protectionSurfaceForSegment(
    protectionSurfaces,
    segmentBundle.segment.segmentId,
    "FINAL_LNAV_VNAV_OCS",
  );
  if (lnavVnavSurface?.centerline.geoPositions.length) {
    return lnavVnavSurface.centerline.geoPositions;
  }

  const centerline = segmentBundle.segmentGeometry.centerline;
  if (centerline.geoPositions.length < 2 || centerline.geodesicLengthNm <= 0) return [];

  const samples = segmentBundle.segmentGeometry.stationAxis.samples.length >= 2
    ? segmentBundle.segmentGeometry.stationAxis.samples.map((sample) => ({
        stationNm: sample.stationNm,
        geoPosition: sample.geoPosition,
      }))
    : centerline.geoPositions.map((geoPosition, index) => ({
        stationNm: centerline.geodesicLengthNm * (
          index / Math.max(centerline.geoPositions.length - 1, 1)
        ),
        geoPosition,
      }));
  if (samples.length < 2) return [];

  const thresholdSample = samples[samples.length - 1];
  const thresholdElevationFtMsl = thresholdSample.geoPosition.altM / 0.3048;
  const thresholdReferenceAltitudeFtMsl =
    thresholdElevationFtMsl + (segmentBundle.segment.verticalRule?.tchFt ?? 0);
  const gpaRad = (gpaDeg * Math.PI) / 180;
  const totalStationNm = thresholdSample.stationNm;

  return samples.map((sample) => {
    const distanceBeforeThresholdNm = Math.max(0, totalStationNm - sample.stationNm);
    const altitudeFtMsl =
      thresholdReferenceAltitudeFtMsl +
      (Math.tan(gpaRad) * distanceBeforeThresholdNm * 1852) / 0.3048;
    return {
      lonDeg: sample.geoPosition.lonDeg,
      latDeg: sample.geoPosition.latDeg,
      altM: altitudeFtMsl * 0.3048,
    };
  });
}

export function buildFinalVerticalReferenceRibbon(
  segmentBundle: ProcedureSegmentRenderBundle,
  points: GeoPoint[],
  protectionSurfaces: ProcedureProtectionSurface[] = [],
): VariableWidthRibbonGeometry | null {
  if (points.length < 2) return null;
  const stations: number[] = [];
  let cumulativeNm = 0;
  points.forEach((point, index) => {
    if (index > 0) {
      cumulativeNm += distanceNm(points[index - 1], point);
    }
    stations.push(cumulativeNm);
  });
  const protectionWidthSamples = primaryProtectionWidthSamples(segmentBundle, protectionSurfaces);
  const nearestProtectionHalfWidthNm = (stationNm: number) =>
    nearestHalfWidthNm(protectionWidthSamples, stationNm);

  return buildVariableWidthRibbon(
    `${segmentBundle.segment.segmentId}:final-vertical-reference-band`,
    {
      geoPositions: points,
      worldPositions: [],
      geodesicLengthNm: cumulativeNm,
      isArc: false,
    },
    stations,
    (stationNm) =>
      Math.max(
        FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM,
        nearestProtectionHalfWidthNm(stationNm) * FINAL_VERTICAL_REFERENCE_PROTECTION_WIDTH_RATIO,
      ),
  );
}

function segmentProtectionHalfWidthNm(
  segmentBundle: ProcedureSegmentRenderBundle,
  pointIndex: number,
  pointCount: number,
  protectionSurfaces: ProcedureProtectionSurface[],
): number {
  const protectionWidthSamples = primaryProtectionWidthSamples(segmentBundle, protectionSurfaces);
  const totalStationNm =
    protectionWidthSamples[protectionWidthSamples.length - 1]?.stationNm ??
    segmentBundle.segmentGeometry.centerline.geodesicLengthNm;
  const stationNm =
    pointCount <= 1 ? totalStationNm : totalStationNm * (pointIndex / Math.max(1, pointCount - 1));
  return nearestHalfWidthNm(protectionWidthSamples, stationNm);
}

function segmentVerticalProfilePointsForSegment(
  segmentBundle: ProcedureSegmentRenderBundle,
  pkg: ProcedurePackage | null,
  protectionSurfaces: ProcedureProtectionSurface[],
): SegmentVerticalProfilePoint[] {
  if (!pkg) return [];
  const fixById = new Map(pkg.sharedFixes.map((fix) => [fix.fixId, fix]));
  const missedCaEndpointByLegId = new Map(
    segmentBundle.missedCaEndpoints.map((endpoint) => [endpoint.legId, endpoint]),
  );
  const points: SegmentVerticalProfilePoint[] = [];
  segmentBundle.legs.forEach((leg) => {
    if (
      (segmentBundle.segment.segmentType === "MISSED_S1" ||
        segmentBundle.segment.segmentType === "MISSED_S2") &&
      leg.legType === "CA"
    ) {
      const endpoint = missedCaEndpointByLegId.get(leg.legId);
      if (!endpoint) return;
      const point = endpoint.geoPositions[1];
      points.push({
        lonDeg: point.lonDeg,
        latDeg: point.latDeg,
        altM: point.altM,
        fixIdent: "CA endpoint",
        altitudeFtMsl: endpoint.targetAltitudeFtMsl,
        halfWidthNm: 0,
        segmentId: segmentBundle.segment.segmentId,
      });
      return;
    }

    const fix = leg.endFixId ? fixById.get(leg.endFixId) : undefined;
    if (!fix || fix.lonDeg === null || fix.latDeg === null) return;
    const altitudeFtMsl = altitudeConstraintReferenceFt(leg.requiredAltitude) ?? fix.altFtMsl;
    if (altitudeFtMsl === null || !Number.isFinite(altitudeFtMsl)) return;
    points.push({
      lonDeg: fix.lonDeg,
      latDeg: fix.latDeg,
      altM: altitudeFtMsl * 0.3048,
      fixIdent: fix.ident,
      altitudeFtMsl,
      halfWidthNm: 0,
      segmentId: segmentBundle.segment.segmentId,
    });
  });

  return points
    .map((point, index) => ({
      ...point,
      halfWidthNm: segmentProtectionHalfWidthNm(
        segmentBundle,
        index,
        points.length,
        protectionSurfaces,
      ),
    }))
    .filter(
      (point, index) =>
        index === 0 ||
        point.fixIdent !== points[index - 1].fixIdent ||
        distanceNm(point, points[index - 1]) > 1e-5,
    );
}

function isBranchVerticalProfileSegment(segmentType: string): boolean {
  return !segmentType.startsWith("MISSED") && segmentType !== "HOLDING";
}

function isSameVerticalProfilePoint(
  point: SegmentVerticalProfilePoint,
  previous: SegmentVerticalProfilePoint,
): boolean {
  return (
    point.fixIdent === previous.fixIdent &&
    distanceNm(point, previous) <= 1e-5 &&
    Math.abs(point.altitudeFtMsl - previous.altitudeFtMsl) <= 1e-3
  );
}

function appendVerticalProfilePoint(
  points: SegmentVerticalProfilePoint[],
  point: SegmentVerticalProfilePoint,
): void {
  const previous = points[points.length - 1];
  if (previous && isSameVerticalProfilePoint(point, previous)) return;
  points.push(point);
}

export function branchVerticalProfileSections(
  branchBundle: BranchGeometryBundle,
  pkg: ProcedurePackage | null,
): SegmentVerticalProfileSection[] {
  const sections: SegmentVerticalProfileSection[] = [];
  let currentSection: SegmentVerticalProfileSection | null = null;

  branchBundle.segmentBundles.forEach((segmentBundle) => {
    if (!isBranchVerticalProfileSegment(segmentBundle.segment.segmentType)) {
      currentSection = null;
      return;
    }

    const points = segmentVerticalProfilePointsForSegment(
      segmentBundle,
      pkg,
      branchBundle.protectionSurfaces,
    );
    if (points.length === 0) return;

    if (!currentSection) {
      currentSection = {
        sectionId: `section-${sections.length + 1}`,
        points: [],
        segmentIds: [],
        segmentTypes: [],
      };
      sections.push(currentSection);
    }

    const section = currentSection;
    if (!section.segmentIds.includes(segmentBundle.segment.segmentId)) {
      section.segmentIds.push(segmentBundle.segment.segmentId);
    }
    if (!section.segmentTypes.includes(segmentBundle.segment.segmentType)) {
      section.segmentTypes.push(segmentBundle.segment.segmentType);
    }
    points.forEach((point) => appendVerticalProfilePoint(section.points, point));
  });

  return sections.filter((section) => section.points.length >= 2);
}

export function buildVerticalProfileRibbon(
  geometryId: string,
  points: SegmentVerticalProfilePoint[],
): VariableWidthRibbonGeometry | null {
  if (points.length < 2) return null;

  const stations: number[] = [];
  let cumulativeNm = 0;
  points.forEach((point, index) => {
    if (index > 0) {
      cumulativeNm += distanceNm(points[index - 1], point);
    }
    stations.push(cumulativeNm);
  });

  const halfWidthSamples = stations.map((stationNm, index) => ({
    stationNm,
    halfWidthNm: points[index].halfWidthNm,
  }));

  return buildVariableWidthRibbon(
    geometryId,
    {
      geoPositions: points,
      worldPositions: [],
      geodesicLengthNm: cumulativeNm,
      isArc: false,
    },
    stations,
    (stationNm) => nearestHalfWidthNm(halfWidthSamples, stationNm),
  );
}
