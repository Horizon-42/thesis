import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  loadProcedureRenderBundleData,
  type BranchGeometryBundle,
  type ProcedureRenderBundle,
  type ProcedureSegmentRenderBundle,
} from "../data/procedureRenderBundle";
import {
  altitudeConstraintLabel,
  altitudeConstraintReferenceFt,
  altitudeConstraintText,
  type DisplayAltitudeConstraint,
} from "../data/altitudeConstraints";
import type {
  ProcedurePackage,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
  SourceRef,
} from "../data/procedurePackage";
import {
  attachProcedureAnnotation,
  getProcedureAnnotation,
  isProcedureAnnotationVisibleAtDisplayLevel,
  procedureAnnotationMeaning,
  type ProcedureDisplayLevel,
  type ProcedureAnnotationKind,
  type ProcedureAnnotationStatus,
  type ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";
import type { ProcedureProtectionSurface } from "../data/procedureProtectionSurfaces";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { distanceNm, interpolateGreatCircle, type GeoPoint } from "../utils/procedureGeoMath";
import {
  buildVariableWidthRibbon,
  type VariableWidthRibbonGeometry,
} from "../utils/procedureSurfaceGeometry";

const PROCEDURE_SEGMENT_ENTITY_PREFIX = "procedure-segment-";
const CENTERLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.95);
const PRIMARY_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.18);
const SECONDARY_COLOR = Cesium.Color.YELLOW.withAlpha(0.1);
const LNAV_VNAV_OCS_PRIMARY_COLOR = Cesium.Color.MAGENTA.withAlpha(0.34);
const LNAV_VNAV_OCS_SECONDARY_COLOR = Cesium.Color.MAGENTA.withAlpha(0.18);
const LNAV_VNAV_OCS_EDGE_COLOR = Cesium.Color.MAGENTA.withAlpha(0.92);
const LNAV_VNAV_OCS_SECONDARY_EDGE_COLOR = Cesium.Color.MAGENTA.withAlpha(0.62);
const LNAV_VNAV_OCS_RIB_COLOR = Cesium.Color.MAGENTA.withAlpha(0.72);
const PRECISION_FINAL_SURFACE_COLOR = Cesium.Color.MAGENTA.withAlpha(0.18);
const FINAL_VERTICAL_REFERENCE_COLOR = Cesium.Color.CYAN.withAlpha(0.88);
const FINAL_VERTICAL_REFERENCE_BAND_COLOR = Cesium.Color.CYAN.withAlpha(0.16);
const SEGMENT_VERTICAL_PROFILE_AID_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.13);
const TURN_FILL_COLOR = Cesium.Color.ORANGE.withAlpha(0.22);
const CONNECTOR_COLOR = Cesium.Color.ORANGE.withAlpha(0.06);
const CONNECTOR_LINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.92);
const FINAL_OEA_TAPER_MARKER_COLOR = Cesium.Color.YELLOW.withAlpha(0.96);
const FINAL_OEA_PFAF_MARKER_COLOR = Cesium.Color.LIME.withAlpha(0.98);
const FINAL_OEA_PRIMARY_WIDTH_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.92);
const FINAL_OEA_SECONDARY_WIDTH_COLOR = Cesium.Color.YELLOW.withAlpha(0.88);
const MISSED_SURFACE_COLOR = Cesium.Color.YELLOW.withAlpha(0.24);
const MISSED_CA_ESTIMATED_SURFACE_COLOR = Cesium.Color.ORANGE.withAlpha(0.26);
const MISSED_CONNECTOR_SURFACE_COLOR = Cesium.Color.ORANGE.withAlpha(0.18);
const CA_COURSE_GUIDE_COLOR = Cesium.Color.ORANGE.withAlpha(0.98);
const CA_CENTERLINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.86);
const CA_MAHF_CONNECTOR_COLOR = Cesium.Color.ORANGE.withAlpha(0.62);
const TURNING_MISSED_DEBUG_COLOR = Cesium.Color.YELLOW.withAlpha(0.98);
const TURNING_MISSED_DEBUG_SURFACE_COLOR = Cesium.Color.YELLOW.withAlpha(0.14);
const TURNING_MISSED_PRIMITIVE_COLOR = Cesium.Color.ORANGE.withAlpha(0.78);
const FINAL_SURFACE_STATUS_COLOR = Cesium.Color.ORANGE.withAlpha(0.9);
const CA_ENDPOINT_COLOR = Cesium.Color.ORANGE.withAlpha(0.98);
const OUTLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.28);
const ENVELOPE_HEIGHT_OFFSET_M = 8;
const OEA_HEIGHT_OFFSET_M = 18;
const FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M = OEA_HEIGHT_OFFSET_M + 12;
const FINAL_OEA_WIDTH_RIB_HEIGHT_OFFSET_M = FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M + 3;
const LNAV_VNAV_OCS_HEIGHT_OFFSET_M = 28;
const LNAV_VNAV_OCS_EDGE_HEIGHT_OFFSET_M = LNAV_VNAV_OCS_HEIGHT_OFFSET_M + 2;
const LNAV_VNAV_OCS_RIB_HEIGHT_OFFSET_M = LNAV_VNAV_OCS_HEIGHT_OFFSET_M + 3;
const PRECISION_FINAL_SURFACE_HEIGHT_OFFSET_M = 34;
const FINAL_VERTICAL_REFERENCE_HEIGHT_OFFSET_M = 40;
const FINAL_VERTICAL_REFERENCE_BAND_HEIGHT_OFFSET_M = 38;
const SEGMENT_VERTICAL_PROFILE_HEIGHT_OFFSET_M = 44;
const FINAL_ALTITUDE_CONSTRAINT_HEIGHT_OFFSET_M = 46;
const ALTITUDE_CONSTRAINT_LINK_HEIGHT_OFFSET_M = 12;
const FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM = 0.15;
const FINAL_VERTICAL_REFERENCE_PROTECTION_WIDTH_RATIO = 1;
const CONNECTOR_HEIGHT_OFFSET_M = 45;
const ALIGNED_CONNECTOR_FILL_HEIGHT_OFFSET_M = ENVELOPE_HEIGHT_OFFSET_M + 4;
const ALIGNED_CONNECTOR_LINE_HEIGHT_OFFSET_M = CONNECTOR_HEIGHT_OFFSET_M;
const MISSED_SURFACE_HEIGHT_OFFSET_M = 58;
const CA_COURSE_GUIDE_HEIGHT_OFFSET_M = 82;
const CA_CENTERLINE_HEIGHT_OFFSET_M = 88;
const CA_MAHF_CONNECTOR_HEIGHT_OFFSET_M = 90;
const TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M = 96;
const FINAL_SURFACE_STATUS_HEIGHT_OFFSET_M = 110;
const CA_ENDPOINT_HEIGHT_OFFSET_M = 92;
const PROCEDURE_ANNOTATION_LABEL_PREFIX = "procedure-annotation-label-";
const PROCEDURE_MEASUREMENT_LABEL_PREFIX = "procedure-measurement-label-";
const FINAL_OEA_STATION_MARKERS = [
  {
    key: "pfaf-minus-03",
    label: "PFAF -0.3",
    stationNm: -0.3,
    description: "LNAV final OEA start and taper start.",
    pixelSize: 9,
    color: FINAL_OEA_TAPER_MARKER_COLOR,
  },
  {
    key: "pfaf",
    label: "PFAF",
    stationNm: 0,
    description: "PFAF station on the final approach course.",
    pixelSize: 11,
    color: FINAL_OEA_PFAF_MARKER_COLOR,
  },
  {
    key: "pfaf-plus-10",
    label: "PFAF +1.0",
    stationNm: 1,
    description: "LNAV final OEA taper end; fixed-width final OEA begins here.",
    pixelSize: 9,
    color: FINAL_OEA_TAPER_MARKER_COLOR,
  },
] as const;

function elevatedPoint(point: GeoPoint, altitudeOffsetM: number): GeoPoint {
  return { ...point, altM: point.altM + altitudeOffsetM };
}

function geoToCartesian(point: GeoPoint, altitudeOffsetM = 0): Cesium.Cartesian3 {
  return Cesium.Cartesian3.fromDegrees(
    point.lonDeg,
    point.latDeg,
    point.altM + altitudeOffsetM,
  );
}

function addPolyline(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  points: GeoPoint[],
  visible: boolean,
  width: number,
  material: Cesium.Color,
  annotation?: ProcedureEntityAnnotation,
): void {
  if (points.length < 2) return;
  const entity = viewer.entities.add({
    id,
    name,
    show: visible,
    polyline: {
      positions: Cesium.Cartesian3.fromDegreesArrayHeights(
        points.flatMap((point) => [point.lonDeg, point.latDeg, point.altM]),
      ),
      width,
      material,
    },
  });
  if (annotation) attachProcedureAnnotation(entity, annotation);
}

function addPoint(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  point: GeoPoint,
  visible: boolean,
  pixelSize: number,
  color: Cesium.Color,
  altitudeOffsetM = 0,
  annotation?: ProcedureEntityAnnotation,
): void {
  const entity = viewer.entities.add({
    id,
    name,
    show: visible,
    position: geoToCartesian(point, altitudeOffsetM),
    point: {
      pixelSize,
      color,
      outlineColor: OUTLINE_COLOR,
      outlineWidth: 2,
    },
  });
  if (annotation) attachProcedureAnnotation(entity, annotation);
}

function representativePoint(points: GeoPoint[]): GeoPoint | null {
  if (points.length === 0) return null;
  return points[Math.floor((points.length - 1) / 2)];
}

function pointAtStationFromSamples(
  centerline: GeoPoint[],
  samples: Array<{ stationNm: number }>,
  stationNm: number,
): GeoPoint | null {
  const count = Math.min(centerline.length, samples.length);
  if (count === 0) return null;
  if (count === 1 || stationNm <= samples[0].stationNm) return centerline[0];

  for (let index = 0; index < count - 1; index += 1) {
    const startStationNm = samples[index].stationNm;
    const endStationNm = samples[index + 1].stationNm;
    if (stationNm <= endStationNm || index === count - 2) {
      const spanNm = endStationNm - startStationNm;
      const ratio = spanNm <= 1e-9
        ? 0
        : Math.min(1, Math.max(0, (stationNm - startStationNm) / spanNm));
      return interpolateGreatCircle(centerline[index], centerline[index + 1], ratio);
    }
  }

  return centerline[count - 1];
}

function isAnnotationLabelId(entityId: string): boolean {
  return entityId.startsWith(PROCEDURE_ANNOTATION_LABEL_PREFIX);
}

function isMeasurementEntityId(entityId: string): boolean {
  return (
    entityId.startsWith(PROCEDURE_MEASUREMENT_LABEL_PREFIX) ||
    entityId.includes("-oea-station-") ||
    entityId.includes("-envelope-width-")
  );
}

function procedureEntityShow(
  baseVisible: boolean,
  annotation: ProcedureEntityAnnotation | null,
  displayLevel: ProcedureDisplayLevel,
  isLabel = false,
  annotationVisible = false,
): boolean {
  return (
    baseVisible &&
    isProcedureAnnotationVisibleAtDisplayLevel(annotation, displayLevel) &&
    (!isLabel || annotationVisible)
  );
}

function sourceRefsFromSegment(segment: ProcedureSegment): string[] {
  return (segment.sourceRefs ?? []).map(formatSourceRef);
}

function sourceRefsFromLegs(legs: ProcedurePackageLeg[]): string[] {
  return [...new Set(legs.flatMap((leg) => (leg.sourceRefs ?? []).map(formatSourceRef)))];
}

function formatSourceRef(ref: SourceRef): string {
  return ref.rawRef ?? [ref.docId, ref.chapter, ref.section, ref.paragraph, ref.figure, ref.formula]
    .filter(Boolean)
    .join(" ");
}

function param(label: string, value: unknown): { label: string; value: string } | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return { label, value: Number.isInteger(value) ? String(value) : value.toFixed(2) };
  return { label, value: String(value) };
}

function isFinalSegment(segment: ProcedureSegment): boolean {
  return segment.segmentType.startsWith("FINAL");
}

function altitudeConstraintColor(
  constraint: DisplayAltitudeConstraint | null | undefined,
  alpha: number,
): Cesium.Color {
  if (constraint?.kind === "AT_OR_ABOVE") return Cesium.Color.CYAN.withAlpha(alpha);
  if (constraint?.kind === "AT_OR_BELOW") return Cesium.Color.ORANGE.withAlpha(alpha);
  if (constraint?.kind === "WINDOW") return Cesium.Color.MAGENTA.withAlpha(alpha);
  if (constraint?.kind === "UNKNOWN") return Cesium.Color.WHITE.withAlpha(alpha);
  return Cesium.Color.LIME.withAlpha(alpha);
}

function fixGeoPointAtAltitude(fix: ProcedurePackageFix, altitudeFtMsl: number): GeoPoint | null {
  if (
    typeof fix.lonDeg !== "number" ||
    typeof fix.latDeg !== "number" ||
    !Number.isFinite(fix.lonDeg) ||
    !Number.isFinite(fix.latDeg)
  ) {
    return null;
  }
  return {
    lonDeg: fix.lonDeg,
    latDeg: fix.latDeg,
    altM: altitudeFtMsl * 0.3048,
  };
}

function finalVerticalReferencePoints(segmentBundle: ProcedureSegmentRenderBundle): GeoPoint[] {
  if (!isFinalSegment(segmentBundle.segment)) return [];
  const gpaDeg = segmentBundle.segment.verticalRule?.gpaDeg;
  if (typeof gpaDeg !== "number" || !Number.isFinite(gpaDeg) || gpaDeg <= 0) return [];

  if (segmentBundle.lnavVnavOcs?.centerline.geoPositions.length) {
    return segmentBundle.lnavVnavOcs.centerline.geoPositions;
  }

  const centerline = segmentBundle.segmentGeometry.centerline;
  if (centerline.geoPositions.length < 2 || centerline.geodesicLengthNm <= 0) return [];

  const samples = segmentBundle.segmentGeometry.stationAxis.samples.length >= 2
    ? segmentBundle.segmentGeometry.stationAxis.samples.map((sample) => ({
        stationNm: sample.stationNm,
        geoPosition: sample.geoPosition,
      }))
    : centerline.geoPositions.map((geoPosition, index) => ({
        stationNm: centerline.geodesicLengthNm * (index / Math.max(centerline.geoPositions.length - 1, 1)),
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

function buildFinalVerticalReferenceRibbon(
  segmentBundle: ProcedureSegmentRenderBundle,
  points: GeoPoint[],
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
  const protectionWidthSamples =
    segmentBundle.finalOea?.primary.halfWidthNmSamples ??
    segmentBundle.segmentGeometry.primaryEnvelope?.halfWidthNmSamples ??
    [];
  const nearestProtectionHalfWidthNm = (stationNm: number) => {
    const nearest = protectionWidthSamples.reduce<
      { stationNm: number; halfWidthNm: number } | null
    >((best, sample) => {
      if (!best) return sample;
      return Math.abs(sample.stationNm - stationNm) < Math.abs(best.stationNm - stationNm)
        ? sample
        : best;
    }, null);
    return nearest?.halfWidthNm ?? FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM * 2;
  };

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

interface SegmentVerticalProfilePoint extends GeoPoint {
  fixIdent: string;
  altitudeFtMsl: number;
  halfWidthNm: number;
  segmentId?: string;
}

interface SegmentVerticalProfileSection {
  sectionId: string;
  points: SegmentVerticalProfilePoint[];
  segmentIds: string[];
  segmentTypes: string[];
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

function segmentProtectionHalfWidthNm(
  segmentBundle: ProcedureSegmentRenderBundle,
  pointIndex: number,
  pointCount: number,
): number {
  const protectionWidthSamples =
    segmentBundle.finalOea?.primary.halfWidthNmSamples ??
    segmentBundle.segmentGeometry.primaryEnvelope?.halfWidthNmSamples ??
    [];
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
      halfWidthNm: segmentProtectionHalfWidthNm(segmentBundle, index, points.length),
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

function branchVerticalProfileSections(
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

    const points = segmentVerticalProfilePointsForSegment(segmentBundle, pkg);
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

function buildVerticalProfileRibbon(
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

function compactSegmentType(segmentType: string | undefined): string {
  if (segmentType === "FINAL_RNAV_GPS") return "FINAL RNAV(GPS)";
  return (segmentType ?? "UNKNOWN").replace(/_/g, " ");
}

function entityIdSuffix(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function annotationBase(args: {
  entityId: string;
  label: string;
  title: string;
  kind: ProcedureAnnotationKind;
  status: ProcedureAnnotationStatus;
  bundle: ProcedureRenderBundle;
  branchBundle: BranchGeometryBundle;
  segment?: ProcedureSegment;
  legs?: ProcedurePackageLeg[];
  leg?: ProcedurePackageLeg;
  parameters?: Array<{ label: string; value: string } | null>;
  diagnostics?: string[];
  sourceRefs?: string[];
}): ProcedureEntityAnnotation {
  return {
    entityId: args.entityId,
    label: args.label,
    title: args.title,
    kind: args.kind,
    status: args.status,
    airportId: args.bundle.airportId,
    runwayId: args.branchBundle.runwayId,
    procedureUid: args.bundle.packageId,
    procedureId: args.bundle.procedureId,
    procedureName: args.bundle.procedureName,
    branchId: args.branchBundle.branchId,
    branchName: args.branchBundle.branchName,
    branchRole: args.branchBundle.branchRole,
    segmentId: args.segment?.segmentId,
    segmentType: args.segment?.segmentType,
    legId: args.leg?.legId,
    legType: args.leg?.legType,
    meaning: procedureAnnotationMeaning(args.kind, args.status),
    parameters: (args.parameters ?? []).filter(
      (item): item is { label: string; value: string } => item !== null,
    ),
    diagnostics: args.diagnostics ?? [],
    sourceRefs:
      args.sourceRefs ??
      (args.leg
        ? (args.leg.sourceRefs ?? []).map(formatSourceRef)
        : args.segment
          ? sourceRefsFromSegment(args.segment)
          : []),
  };
}

function segmentParams(segment: ProcedureSegment, legs: ProcedurePackageLeg[]): Array<{ label: string; value: string } | null> {
  return [
    param("Segment", compactSegmentType(segment.segmentType)),
    param("Nav spec", segment.navSpec),
    param("XTT", `${segment.xttNm} NM`),
    param("ATT", `${segment.attNm} NM`),
    param("Width", segment.widthChangeMode),
    param("Legs", legs.map((leg) => leg.legType).join(", ")),
  ];
}

function protectionSurfaceIdForMissedSection(segmentId: string, surfaceType: string): string {
  return `${segmentId}:${surfaceType.toLowerCase()}`;
}

function findProtectionSurface(
  branchBundle: BranchGeometryBundle,
  surfaceId: string | null | undefined,
): ProcedureProtectionSurface | null {
  if (!surfaceId) return null;
  return (branchBundle.protectionSurfaces ?? []).find((surface) => surface.surfaceId === surfaceId) ?? null;
}

function formatEnumLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function formatNmRange(values: number[]): string | null {
  if (values.length === 0) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 1e-6) return `${min.toFixed(2)} NM`;
  return `${min.toFixed(2)}-${max.toFixed(2)} NM`;
}

function protectionSurfaceParams(
  surface: ProcedureProtectionSurface | null,
): Array<{ label: string; value: string } | null> {
  if (!surface) return [];
  return [
    param("Surface kind", formatEnumLabel(surface.kind)),
    param("Surface status", formatEnumLabel(surface.status)),
    param("Lateral rule", surface.lateral.rule),
    param(
      "Primary half-width",
      formatNmRange(surface.lateral.widthSamples.map((sample) => sample.primaryHalfWidthNm)),
    ),
    param(
      "Secondary outer half-width",
      formatNmRange(
        surface.lateral.widthSamples
          .map((sample) => sample.secondaryOuterHalfWidthNm)
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value)),
      ),
    ),
    param("Vertical kind", formatEnumLabel(surface.vertical.kind)),
    param("Vertical origin", formatEnumLabel(surface.vertical.origin)),
    param("Climb gradient", surface.vertical.slopeFtPerNm === undefined
      ? null
      : `${surface.vertical.slopeFtPerNm} ft/NM`),
  ];
}

function protectionSurfaceAnnotationStatus(
  surface: ProcedureProtectionSurface,
): ProcedureAnnotationStatus {
  if (surface.status === "SOURCE_BACKED") return "SOURCE_BACKED";
  if (surface.status === "DEBUG_ESTIMATE") return "DEBUG_ESTIMATE";
  if (surface.status === "MISSING_SOURCE") return "MISSING_SOURCE";
  return "ESTIMATED";
}

function segmentBundleForSurface(
  branchBundle: BranchGeometryBundle,
  surface: ProcedureProtectionSurface,
): ProcedureSegmentRenderBundle | null {
  return branchBundle.segmentBundles.find(
    (segmentBundle) => segmentBundle.segment.segmentId === surface.segmentId,
  ) ?? null;
}

function segmentBaseEntityId(
  bundle: ProcedureRenderBundle,
  segmentId: string,
): string {
  return `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${segmentId}`;
}

function diagnosticsForProtectionSurface(
  surface: ProcedureProtectionSurface,
  segmentBundle: ProcedureSegmentRenderBundle | null,
): string[] {
  const surfaceDiagnostics = (surface.diagnostics ?? []).map((diagnostic) => diagnostic.message);
  if (surfaceDiagnostics.length > 0) return surfaceDiagnostics;
  return segmentBundle?.diagnostics.map((diagnostic) => diagnostic.message) ?? [];
}

function surfaceSegmentParams(
  segmentBundle: ProcedureSegmentRenderBundle | null,
): Array<{ label: string; value: string } | null> {
  return segmentBundle
    ? segmentParams(segmentBundle.segment, segmentBundle.legs)
    : [];
}

function precisionSurfaceLabel(surface: ProcedureProtectionSurface): string {
  const suffix = surface.surfaceId.slice(surface.segmentId.length + 1);
  return suffix.replace(/-/g, " ").toUpperCase();
}

function interpolatedHalfWidthNm(
  samples: Array<{ stationNm: number; halfWidthNm: number }>,
  stationNm: number,
): number | null {
  if (samples.length === 0) return null;
  if (samples.length === 1 || stationNm <= samples[0].stationNm) return samples[0].halfWidthNm;

  for (let index = 0; index < samples.length - 1; index += 1) {
    const start = samples[index];
    const end = samples[index + 1];
    if (stationNm <= end.stationNm || index === samples.length - 2) {
      const spanNm = end.stationNm - start.stationNm;
      const ratio = spanNm <= 1e-9
        ? 0
        : Math.min(1, Math.max(0, (stationNm - start.stationNm) / spanNm));
      return start.halfWidthNm + (end.halfWidthNm - start.halfWidthNm) * ratio;
    }
  }

  return samples[samples.length - 1].halfWidthNm;
}

function interpolatedSurfaceWidthSample(
  samples: ProcedureProtectionSurface["lateral"]["widthSamples"],
  stationNm: number,
): ProcedureProtectionSurface["lateral"]["widthSamples"][number] | null {
  if (samples.length === 0) return null;
  if (samples.length === 1 || stationNm <= samples[0].stationNm) return samples[0];

  for (let index = 0; index < samples.length - 1; index += 1) {
    const start = samples[index];
    const end = samples[index + 1];
    if (stationNm <= end.stationNm || index === samples.length - 2) {
      const spanNm = end.stationNm - start.stationNm;
      const ratio = spanNm <= 1e-9
        ? 0
        : Math.min(1, Math.max(0, (stationNm - start.stationNm) / spanNm));
      return {
        stationNm,
        primaryHalfWidthNm:
          start.primaryHalfWidthNm +
          (end.primaryHalfWidthNm - start.primaryHalfWidthNm) * ratio,
        secondaryOuterHalfWidthNm:
          start.secondaryOuterHalfWidthNm === undefined ||
          end.secondaryOuterHalfWidthNm === undefined
            ? undefined
            : start.secondaryOuterHalfWidthNm +
              (end.secondaryOuterHalfWidthNm - start.secondaryOuterHalfWidthNm) * ratio,
      };
    }
  }

  return samples[samples.length - 1];
}

function finalOeaWidthSamplesFromRibbons(
  primary: VariableWidthRibbonGeometry,
  secondaryOuter: VariableWidthRibbonGeometry | null,
): ProcedureProtectionSurface["lateral"]["widthSamples"] {
  return primary.halfWidthNmSamples.map((sample) => ({
    stationNm: sample.stationNm,
    primaryHalfWidthNm: sample.halfWidthNm,
    secondaryOuterHalfWidthNm: secondaryOuter
      ? (interpolatedHalfWidthNm(secondaryOuter.halfWidthNmSamples, sample.stationNm) ?? undefined)
      : undefined,
  }));
}

function formatWidthNm(halfWidthNm: number): string {
  const rounded = Math.round((halfWidthNm + Number.EPSILON) * 100) / 100;
  const text = rounded.toFixed(2).replace(/\.?0+$/, "");
  return `${text} NM`;
}

function surfaceWidthLabel(
  sample: ProcedureProtectionSurface["lateral"]["widthSamples"][number] | null,
): string | null {
  if (!sample) return null;
  const primary = `P half ${formatWidthNm(sample.primaryHalfWidthNm)}`;
  const secondary = sample.secondaryOuterHalfWidthNm === undefined
    ? null
    : `S outer ${formatWidthNm(sample.secondaryOuterHalfWidthNm)}`;
  return secondary ? `${primary}\n${secondary}` : primary;
}

function addSurfaceWidthRib(args: {
  viewer: Cesium.Viewer;
  id: string;
  name: string;
  ribbon: ProcedureProtectionSurface["lateral"]["primary"] | null;
  stationNm: number;
  visible: boolean;
  material: Cesium.Color;
  annotation?: ProcedureEntityAnnotation;
}): string[] {
  if (!args.ribbon) return [];
  const left = pointAtStationFromSamples(
    args.ribbon.leftGeoBoundary,
    args.ribbon.halfWidthNmSamples,
    args.stationNm,
  );
  const right = pointAtStationFromSamples(
    args.ribbon.rightGeoBoundary,
    args.ribbon.halfWidthNmSamples,
    args.stationNm,
  );
  if (!left || !right) return [];

  addPolyline(
    args.viewer,
    args.id,
    args.name,
    [
      elevatedPoint(left, FINAL_OEA_WIDTH_RIB_HEIGHT_OFFSET_M),
      elevatedPoint(right, FINAL_OEA_WIDTH_RIB_HEIGHT_OFFSET_M),
    ],
    args.visible,
    3,
    args.material,
    args.annotation,
  );
  return [args.id];
}

function addFinalOeaStationMarkers(args: {
  viewer: Cesium.Viewer;
  bundle: ProcedureRenderBundle;
  branchBundle: BranchGeometryBundle;
  segmentBundle: ProcedureSegmentRenderBundle | null;
  baseId: string;
  segmentName: string;
  centerline: GeoPoint[];
  primary: ProcedureProtectionSurface["lateral"]["primary"];
  secondaryOuter: ProcedureProtectionSurface["lateral"]["secondaryOuter"];
  widthSamples: ProcedureProtectionSurface["lateral"]["widthSamples"];
  visible: boolean;
  annotationVisible: boolean;
  displayLevel: ProcedureDisplayLevel;
  diagnostics: string[];
}): string[] {
  const ids: string[] = [];
  if (args.centerline.length === 0 || args.widthSamples.length === 0) return ids;

  FINAL_OEA_STATION_MARKERS.forEach((marker) => {
    const point = pointAtStationFromSamples(args.centerline, args.widthSamples, marker.stationNm);
    if (!point) return;

    const widthSample = interpolatedSurfaceWidthSample(args.widthSamples, marker.stationNm);
    const widthLabel = surfaceWidthLabel(widthSample);
    const markerId = `${args.baseId}-oea-station-${marker.key}`;
    const annotation = annotationBase({
      entityId: markerId,
      label: widthLabel ? `${marker.label}\n${widthLabel}` : marker.label,
      title: `${args.segmentName} ${marker.label} LNAV OEA station`,
      kind: "FINAL_OEA",
      status: "SOURCE_BACKED",
      bundle: args.bundle,
      branchBundle: args.branchBundle,
      segment: args.segmentBundle?.segment,
      legs: args.segmentBundle?.legs,
      parameters: [
        ...surfaceSegmentParams(args.segmentBundle),
        param(
          "Station",
          `${marker.stationNm >= 0 ? "+" : ""}${marker.stationNm.toFixed(1)} NM from PFAF`,
        ),
        param("Marker", marker.description),
        param("Primary half-width", widthSample ? formatWidthNm(widthSample.primaryHalfWidthNm) : null),
        param(
          "Secondary outer half-width",
          widthSample?.secondaryOuterHalfWidthNm === undefined
            ? null
            : formatWidthNm(widthSample.secondaryOuterHalfWidthNm),
        ),
        param("Taper rule", "FAA 8260.58D formula 3-2-1; fixed width after PFAF +1.0 NM"),
      ],
      diagnostics: args.diagnostics,
    });
    addPoint(
      args.viewer,
      markerId,
      `${args.segmentName} ${marker.label} LNAV OEA station`,
      point,
      procedureEntityShow(args.visible, annotation, args.displayLevel),
      marker.pixelSize,
      marker.color,
      FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M,
      annotation,
    );
    ids.push(markerId);

    const labelId = addMeasurementLabel(
      args.viewer,
      `${PROCEDURE_MEASUREMENT_LABEL_PREFIX}${markerId}`,
      `${args.segmentName} ${marker.label} LNAV OEA width label`,
      elevatedPoint(point, FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M),
      widthLabel ? `${marker.label}\n${widthLabel}` : marker.label,
      procedureEntityShow(args.visible, annotation, args.displayLevel),
      annotation,
    );
    if (labelId) ids.push(labelId);

    ids.push(
      ...addSurfaceWidthRib({
        viewer: args.viewer,
        id: `${markerId}-primary-width`,
        name: `${args.segmentName} ${marker.label} LNAV OEA primary width`,
        ribbon: args.primary,
        stationNm: marker.stationNm,
        visible: procedureEntityShow(args.visible, annotation, args.displayLevel),
        material: FINAL_OEA_PRIMARY_WIDTH_COLOR,
        annotation,
      }),
    );
    if (args.secondaryOuter) {
      ids.push(
        ...addSurfaceWidthRib({
          viewer: args.viewer,
          id: `${markerId}-secondary-outer-width`,
          name: `${args.segmentName} ${marker.label} LNAV OEA secondary outer width`,
          ribbon: args.secondaryOuter,
          stationNm: marker.stationNm,
          visible: procedureEntityShow(args.visible, annotation, args.displayLevel),
          material: FINAL_OEA_SECONDARY_WIDTH_COLOR,
          annotation,
        }),
      );
    }
  });

  return ids;
}

function widthMarkerStations(
  samples: Array<{ stationNm: number }>,
): Array<{ key: string; label: string; stationNm: number }> {
  if (samples.length === 0) return [];
  const startStationNm = samples[0].stationNm;
  const endStationNm = samples[samples.length - 1].stationNm;
  const candidates = [
    { key: "start", label: "Start", stationNm: startStationNm },
    { key: "mid", label: "Mid", stationNm: (startStationNm + endStationNm) / 2 },
    { key: "end", label: "End", stationNm: endStationNm },
  ];
  return candidates.filter(
    (candidate, index) =>
      candidates.findIndex(
        (other) => Math.abs(other.stationNm - candidate.stationNm) < 1e-6,
      ) === index,
  );
}

function addSegmentEnvelopeWidthMarkers(args: {
  viewer: Cesium.Viewer;
  bundle: ProcedureRenderBundle;
  branchBundle: BranchGeometryBundle;
  segmentBundle: ProcedureSegmentRenderBundle;
  baseId: string;
  segmentName: string;
  primary: ProcedureProtectionSurface["lateral"]["primary"] | null | undefined;
  secondaryOuter: ProcedureProtectionSurface["lateral"]["secondaryOuter"];
  visible: boolean;
  displayLevel: ProcedureDisplayLevel;
  diagnostics: string[];
}): string[] {
  const ids: string[] = [];
  const primary = args.primary;
  const secondaryOuter = args.secondaryOuter ?? null;
  if (!primary || primary.halfWidthNmSamples.length === 0) return ids;

  const widthSamples = finalOeaWidthSamplesFromRibbons(
    primary,
    secondaryOuter,
  );
  const centerline = args.segmentBundle.segmentGeometry.centerline.geoPositions;
  widthMarkerStations(primary.halfWidthNmSamples).forEach((marker) => {
    const point = pointAtStationFromSamples(
      centerline,
      primary.halfWidthNmSamples,
      marker.stationNm,
    );
    if (!point) return;

    const widthSample = interpolatedSurfaceWidthSample(widthSamples, marker.stationNm);
    const widthLabel = surfaceWidthLabel(widthSample);
    const markerId = `${args.baseId}-envelope-width-${marker.key}`;
    const annotation = annotationBase({
      entityId: markerId,
      label: widthLabel ? `${marker.label}\n${widthLabel}` : marker.label,
      title: `${args.segmentName} ${marker.label.toLowerCase()} envelope width`,
      kind: "SEGMENT_ENVELOPE_PRIMARY",
      status: args.segmentBundle.segmentGeometry.diagnostics.some(
        (diagnostic) => diagnostic.code === "ESTIMATED_CA_GEOMETRY",
      )
        ? "ESTIMATED"
        : "SOURCE_BACKED",
      bundle: args.bundle,
      branchBundle: args.branchBundle,
      segment: args.segmentBundle.segment,
      legs: args.segmentBundle.legs,
      parameters: [
        ...segmentParams(args.segmentBundle.segment, args.segmentBundle.legs),
        param("Station", `${marker.stationNm.toFixed(2)} NM from segment start`),
        param("Primary half-width", widthSample ? formatWidthNm(widthSample.primaryHalfWidthNm) : null),
        param(
          "Secondary outer half-width",
          widthSample?.secondaryOuterHalfWidthNm === undefined
            ? null
            : formatWidthNm(widthSample.secondaryOuterHalfWidthNm),
        ),
      ],
      diagnostics: args.diagnostics,
      sourceRefs: sourceRefsFromSegment(args.segmentBundle.segment),
    });
    const visibleAtLevel = procedureEntityShow(args.visible, annotation, args.displayLevel);
    addPoint(
      args.viewer,
      markerId,
      `${args.segmentName} ${marker.label.toLowerCase()} envelope width`,
      point,
      visibleAtLevel,
      7,
      FINAL_OEA_PRIMARY_WIDTH_COLOR,
      FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M,
      annotation,
    );
    ids.push(markerId);

    const labelId = addMeasurementLabel(
      args.viewer,
      `${PROCEDURE_MEASUREMENT_LABEL_PREFIX}${markerId}`,
      `${args.segmentName} ${marker.label.toLowerCase()} envelope width label`,
      elevatedPoint(point, FINAL_OEA_STATION_MARKER_HEIGHT_OFFSET_M),
      widthLabel ? `${marker.label}\n${widthLabel}` : marker.label,
      visibleAtLevel,
      annotation,
    );
    if (labelId) ids.push(labelId);

    ids.push(
      ...addSurfaceWidthRib({
        viewer: args.viewer,
        id: `${markerId}-primary-width`,
        name: `${args.segmentName} ${marker.label.toLowerCase()} primary envelope width`,
        ribbon: primary,
        stationNm: marker.stationNm,
        visible: visibleAtLevel,
        material: FINAL_OEA_PRIMARY_WIDTH_COLOR,
        annotation,
      }),
    );
    if (secondaryOuter) {
      ids.push(
        ...addSurfaceWidthRib({
          viewer: args.viewer,
          id: `${markerId}-secondary-outer-width`,
          name: `${args.segmentName} ${marker.label.toLowerCase()} secondary envelope width`,
          ribbon: secondaryOuter,
          stationNm: marker.stationNm,
          visible: visibleAtLevel,
          material: FINAL_OEA_SECONDARY_WIDTH_COLOR,
          annotation,
        }),
      );
    }
  });

  return ids;
}

function addAnnotationLabel(
  viewer: Cesium.Viewer,
  annotation: ProcedureEntityAnnotation,
  anchor: GeoPoint | null,
  visible: boolean,
): string | null {
  if (!anchor) return null;
  const id = `${PROCEDURE_ANNOTATION_LABEL_PREFIX}${annotation.entityId}`;
  const entity = viewer.entities.add({
    id,
    name: `${annotation.title} label`,
    show: visible,
    position: geoToCartesian(anchor, 18),
    label: {
      text: annotation.label,
      font: "12px sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 3,
      showBackground: true,
      backgroundColor: Cesium.Color.BLACK.withAlpha(0.55),
      scale: 0.9,
    },
  });
  attachProcedureAnnotation(entity, annotation);
  return id;
}

function addMeasurementLabel(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  anchor: GeoPoint | null,
  text: string,
  visible: boolean,
  annotation?: ProcedureEntityAnnotation,
): string | null {
  if (!anchor || !text) return null;
  const entity = viewer.entities.add({
    id,
    name,
    show: visible,
    position: geoToCartesian(anchor, 18),
    label: {
      text,
      font: "12px sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 3,
      showBackground: true,
      backgroundColor: Cesium.Color.BLACK.withAlpha(0.66),
      scale: 0.95,
    },
  });
  if (annotation) attachProcedureAnnotation(entity, annotation);
  return id;
}

function importantFixRole(fix: ProcedurePackageFix): string | null {
  const importantRoles = ["IAF", "IF", "PFAF", "FAF", "MAP", "MAHF", "RWY", "FROP"];
  return fix.role.find((role) => importantRoles.includes(role)) ?? null;
}

function addFixLabelEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  pkg: ProcedurePackage,
  visible: boolean,
  annotationVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  pkg.sharedFixes.forEach((fix) => {
    const role = importantFixRole(fix);
    if (!role || fix.latDeg === null || fix.lonDeg === null) return;
    const entityId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-fix-${fix.fixId}`;
    const annotation = annotationBase({
      entityId,
      label: `${role} ${fix.ident}`,
      title: `${fix.ident} procedure fix`,
      kind: "FIX",
      status: "SOURCE_BACKED",
      bundle,
      branchBundle,
      parameters: [
        param("Fix", fix.ident),
        param("Role", fix.role.join(", ")),
        param("Elevation", fix.altFtMsl === null ? null : `${fix.altFtMsl} ft`),
      ],
      sourceRefs: fix.sourceRefs.map(formatSourceRef),
    });
    const labelId = addAnnotationLabel(
      viewer,
      annotation,
      { lonDeg: fix.lonDeg, latDeg: fix.latDeg, altM: (fix.altFtMsl ?? 0) * 0.3048 },
      procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
    );
    if (labelId) ids.push(labelId);
  });
  return ids;
}

function addRibbonPolygon(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  ribbon: VariableWidthRibbonGeometry | undefined,
  visible: boolean,
  material: Cesium.Color,
  altitudeOffsetM = 0,
  annotation?: ProcedureEntityAnnotation,
): void {
  if (!ribbon || ribbon.leftGeoBoundary.length < 2 || ribbon.rightGeoBoundary.length < 2) return;
  const polygonPoints = [...ribbon.leftGeoBoundary, ...ribbon.rightGeoBoundary.slice().reverse()];
  const entity = viewer.entities.add({
    id,
    name,
    show: visible,
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        polygonPoints.map((point) => geoToCartesian(point, altitudeOffsetM)),
      ),
      material,
      perPositionHeight: true,
      outline: true,
      outlineColor: OUTLINE_COLOR,
    },
  });
  if (annotation) attachProcedureAnnotation(entity, annotation);
}

function addRibbonBoundaryPolylines(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  ribbon: VariableWidthRibbonGeometry | undefined,
  visible: boolean,
  material: Cesium.Color,
  altitudeOffsetM: number,
  annotation?: ProcedureEntityAnnotation,
): string[] {
  if (!ribbon || ribbon.leftGeoBoundary.length < 2 || ribbon.rightGeoBoundary.length < 2) return [];
  const leftId = `${id}-left`;
  const rightId = `${id}-right`;
  addPolyline(
    viewer,
    leftId,
    `${name} left boundary`,
    ribbon.leftGeoBoundary.map((point) => elevatedPoint(point, altitudeOffsetM)),
    visible,
    3,
    material,
    annotation,
  );
  addPolyline(
    viewer,
    rightId,
    `${name} right boundary`,
    ribbon.rightGeoBoundary.map((point) => elevatedPoint(point, altitudeOffsetM)),
    visible,
    3,
    material,
    annotation,
  );
  return [leftId, rightId];
}

function addRibbonCrossRibs(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  ribbon: VariableWidthRibbonGeometry | undefined,
  visible: boolean,
  material: Cesium.Color,
  altitudeOffsetM: number,
  annotation?: ProcedureEntityAnnotation,
): string[] {
  if (!ribbon || ribbon.leftGeoBoundary.length < 2 || ribbon.rightGeoBoundary.length < 2) return [];

  const ribIds: string[] = [];
  const lastIndex = Math.min(ribbon.leftGeoBoundary.length, ribbon.rightGeoBoundary.length) - 1;
  const indexes = Array.from(
    new Set([
      0,
      Math.round(lastIndex * 0.25),
      Math.round(lastIndex * 0.5),
      Math.round(lastIndex * 0.75),
      lastIndex,
    ]),
  ).filter((index) => index >= 0 && index <= lastIndex);

  indexes.forEach((index) => {
    const ribId = `${id}-rib-${index}`;
    addPolyline(
      viewer,
      ribId,
      `${name} station rib ${index}`,
      [
        elevatedPoint(ribbon.leftGeoBoundary[index], altitudeOffsetM),
        elevatedPoint(ribbon.rightGeoBoundary[index], altitudeOffsetM),
      ],
      visible,
      2,
      material,
      annotation,
    );
    ribIds.push(ribId);
  });

  return ribIds;
}

function addSegmentEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  segmentBundle: ProcedureSegmentRenderBundle,
  pkg: ProcedurePackage | null,
  visible: boolean,
  annotationVisible: boolean,
  widthMeasurementVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${segmentBundle.segment.segmentId}`;
  const segmentName = `${bundle.procedureName} ${segmentBundle.segment.segmentType}`;
  const segmentDiagnostics = segmentBundle.diagnostics.map((diagnostic) => diagnostic.message);
  const centerlineEstimated = segmentBundle.segmentGeometry.diagnostics.some(
    (diagnostic) => diagnostic.code === "ESTIMATED_CA_GEOMETRY",
  );
  const hasUnifiedSurfaces = (branchBundle.protectionSurfaces ?? []).length > 0;
  const hasUnifiedTurningMissedDebugSurfaces = (branchBundle.protectionSurfaces ?? []).some(
    (surface) =>
      surface.segmentId === segmentBundle.segment.segmentId &&
      surface.kind === "TURNING_MISSED_DEBUG",
  );

  const centerlineId = `${baseId}-centerline`;
  const centerlineAnnotation = annotationBase({
    entityId: centerlineId,
    label: compactSegmentType(segmentBundle.segment.segmentType),
    title: `${segmentName} centerline`,
    kind: "SEGMENT_CENTERLINE",
    status: centerlineEstimated ? "ESTIMATED" : "SOURCE_BACKED",
    bundle,
    branchBundle,
    segment: segmentBundle.segment,
    legs: segmentBundle.legs,
    parameters: segmentParams(segmentBundle.segment, segmentBundle.legs),
    diagnostics: segmentDiagnostics,
    sourceRefs: sourceRefsFromLegs(segmentBundle.legs),
  });
  addPolyline(
    viewer,
    centerlineId,
    `${segmentName} centerline`,
    segmentBundle.segmentGeometry.centerline.geoPositions,
    procedureEntityShow(visible, centerlineAnnotation, displayLevel),
    4,
    CENTERLINE_COLOR,
    centerlineAnnotation,
  );
  ids.push(centerlineId);
  const centerlineLabelId = addAnnotationLabel(
    viewer,
    centerlineAnnotation,
    representativePoint(segmentBundle.segmentGeometry.centerline.geoPositions),
    procedureEntityShow(visible, centerlineAnnotation, displayLevel, true, annotationVisible),
  );
  if (centerlineLabelId) ids.push(centerlineLabelId);

  if (isFinalSegment(segmentBundle.segment)) {
    const verticalReferencePoints = finalVerticalReferencePoints(segmentBundle);
    if (verticalReferencePoints.length >= 2) {
      const verticalReferenceBandId = `${baseId}-final-vertical-reference-band`;
      const verticalReferenceId = `${baseId}-final-vertical-reference`;
      const estimatedFromThreshold =
        segmentBundle.lnavVnavOcs === null &&
        typeof segmentBundle.segment.verticalRule?.tchFt !== "number";
      const verticalReferenceAnnotation = annotationBase({
        entityId: verticalReferenceId,
        label: `GPA ${segmentBundle.segment.verticalRule?.gpaDeg?.toFixed(1) ?? "?"} deg`,
        title: `${segmentName} final vertical reference`,
        kind: "FINAL_VERTICAL_REFERENCE",
        status: "ESTIMATED",
        bundle,
        branchBundle,
        segment: segmentBundle.segment,
        legs: segmentBundle.legs,
        parameters: [
          ...segmentParams(segmentBundle.segment, segmentBundle.legs),
          param("GPA", `${segmentBundle.segment.verticalRule?.gpaDeg} deg`),
          param("TCH", segmentBundle.segment.verticalRule?.tchFt === undefined
            ? "Not available"
            : `${segmentBundle.segment.verticalRule.tchFt} ft`),
          param("Basis", estimatedFromThreshold ? "Threshold/MAPt elevation estimate" : "GPA/TCH surface centerline"),
        ],
        diagnostics: estimatedFromThreshold
          ? [
              ...segmentDiagnostics,
              "TCH is not available; final vertical reference is anchored at runway/MAPt elevation.",
            ]
          : segmentDiagnostics,
        sourceRefs: sourceRefsFromSegment(segmentBundle.segment),
      });
      const referenceRibbon = buildFinalVerticalReferenceRibbon(
        segmentBundle,
        verticalReferencePoints,
      );
      addRibbonPolygon(
        viewer,
        verticalReferenceBandId,
        `${segmentName} final vertical reference band`,
        referenceRibbon ?? undefined,
        procedureEntityShow(visible, verticalReferenceAnnotation, displayLevel),
        FINAL_VERTICAL_REFERENCE_BAND_COLOR,
        FINAL_VERTICAL_REFERENCE_BAND_HEIGHT_OFFSET_M,
        verticalReferenceAnnotation,
      );
      ids.push(verticalReferenceBandId);
      addPolyline(
        viewer,
        verticalReferenceId,
        `${segmentName} final vertical reference`,
        verticalReferencePoints.map((point) => elevatedPoint(point, FINAL_VERTICAL_REFERENCE_HEIGHT_OFFSET_M)),
        procedureEntityShow(visible, verticalReferenceAnnotation, displayLevel),
        5,
        FINAL_VERTICAL_REFERENCE_COLOR,
        verticalReferenceAnnotation,
      );
      ids.push(verticalReferenceId);
      const verticalReferenceLabelId = addAnnotationLabel(
        viewer,
        verticalReferenceAnnotation,
        representativePoint(verticalReferencePoints.map((point) => elevatedPoint(point, FINAL_VERTICAL_REFERENCE_HEIGHT_OFFSET_M))),
        procedureEntityShow(visible, verticalReferenceAnnotation, displayLevel, true, annotationVisible),
      );
      if (verticalReferenceLabelId) ids.push(verticalReferenceLabelId);
    }

  }

  if (pkg) {
    const fixById = new Map(pkg.sharedFixes.map((fix) => [fix.fixId, fix]));
    segmentBundle.legs.forEach((leg) => {
      const constraint = leg.requiredAltitude;
      const altitudeFtMsl = altitudeConstraintReferenceFt(constraint);
      const endFix = leg.endFixId ? fixById.get(leg.endFixId) : undefined;
      if (altitudeFtMsl === null || !endFix) return;
      const point = fixGeoPointAtAltitude(endFix, altitudeFtMsl);
      if (!point) return;

      const constraintId = `${baseId}-altitude-${leg.legId}`;
      const constraintLabel = altitudeConstraintLabel(endFix.ident, constraint);
      const constraintAnnotation = annotationBase({
        entityId: constraintId,
        label: constraintLabel,
        title: `${segmentName} altitude constraint`,
        kind: "ALTITUDE_CONSTRAINT",
        status: "SOURCE_BACKED",
        bundle,
        branchBundle,
        segment: segmentBundle.segment,
        legs: segmentBundle.legs,
        leg,
        parameters: [
          param("Fix", endFix.ident),
          param("Constraint type", constraint?.kind),
          param("Constraint", altitudeConstraintText(constraint)),
          param("Min altitude", constraint?.minFtMsl === undefined ? null : `${constraint.minFtMsl} ft MSL`),
          param("Max altitude", constraint?.maxFtMsl === undefined ? null : `${constraint.maxFtMsl} ft MSL`),
          param("Source text", constraint?.sourceText),
          param("Leg", leg.legType),
        ],
        diagnostics: segmentDiagnostics,
        sourceRefs: (leg.sourceRefs ?? []).map(formatSourceRef),
      });
      addPoint(
        viewer,
        constraintId,
        `${segmentName} altitude ${endFix.ident} ${Math.round(altitudeFtMsl)} ft`,
        point,
        procedureEntityShow(visible, constraintAnnotation, displayLevel),
        9,
        altitudeConstraintColor(constraint, 0.98),
        FINAL_ALTITUDE_CONSTRAINT_HEIGHT_OFFSET_M,
        constraintAnnotation,
      );
      ids.push(constraintId);
      const linkId = `${constraintId}-link`;
      addPolyline(
        viewer,
        linkId,
        `${segmentName} altitude constraint link ${endFix.ident}`,
        [
          elevatedPoint(point, ALTITUDE_CONSTRAINT_LINK_HEIGHT_OFFSET_M),
          elevatedPoint(point, FINAL_ALTITUDE_CONSTRAINT_HEIGHT_OFFSET_M),
        ],
        procedureEntityShow(visible, constraintAnnotation, displayLevel),
        2,
        altitudeConstraintColor(constraint, 0.48),
        constraintAnnotation,
      );
      ids.push(linkId);
      const constraintLabelId = addAnnotationLabel(
        viewer,
        constraintAnnotation,
        elevatedPoint(point, FINAL_ALTITUDE_CONSTRAINT_HEIGHT_OFFSET_M),
        procedureEntityShow(visible, constraintAnnotation, displayLevel, true, annotationVisible),
      );
      if (constraintLabelId) ids.push(constraintLabelId);
    });
  }

  const primaryId = `${baseId}-envelope-primary`;
  const primaryAnnotation = annotationBase({
    entityId: primaryId,
    label: "Primary",
    title: `${segmentName} primary envelope`,
    kind: "SEGMENT_ENVELOPE_PRIMARY",
    status: centerlineEstimated ? "ESTIMATED" : "SOURCE_BACKED",
    bundle,
    branchBundle,
    segment: segmentBundle.segment,
    legs: segmentBundle.legs,
    parameters: [
      ...segmentParams(segmentBundle.segment, segmentBundle.legs),
      param("Surface meaning", "Lateral protection footprint"),
      param("Vertical meaning", "Not OCS; not a vertical clearance surface"),
    ],
    diagnostics: segmentDiagnostics,
    sourceRefs: sourceRefsFromSegment(segmentBundle.segment),
  });
  addRibbonPolygon(
    viewer,
    primaryId,
    `${segmentName} primary envelope`,
    segmentBundle.segmentGeometry.primaryEnvelope,
    procedureEntityShow(visible, primaryAnnotation, displayLevel),
    PRIMARY_COLOR,
    ENVELOPE_HEIGHT_OFFSET_M,
    primaryAnnotation,
  );
  ids.push(primaryId);

  const secondaryId = `${baseId}-envelope-secondary`;
  const secondaryAnnotation = annotationBase({
    entityId: secondaryId,
    label: "Secondary",
    title: `${segmentName} secondary envelope`,
    kind: "SEGMENT_ENVELOPE_SECONDARY",
    status: centerlineEstimated ? "ESTIMATED" : "SOURCE_BACKED",
    bundle,
    branchBundle,
    segment: segmentBundle.segment,
    legs: segmentBundle.legs,
    parameters: [
      ...segmentParams(segmentBundle.segment, segmentBundle.legs),
      param("Surface meaning", "Secondary lateral protection footprint"),
      param("Vertical meaning", "Not OCS; not a vertical clearance surface"),
    ],
    diagnostics: segmentDiagnostics,
    sourceRefs: sourceRefsFromSegment(segmentBundle.segment),
  });
  addRibbonPolygon(
    viewer,
    secondaryId,
    `${segmentName} secondary envelope`,
    segmentBundle.segmentGeometry.secondaryEnvelope,
    procedureEntityShow(visible, secondaryAnnotation, displayLevel),
    SECONDARY_COLOR,
    ENVELOPE_HEIGHT_OFFSET_M,
    secondaryAnnotation,
  );
  ids.push(secondaryId);

  if (!segmentBundle.finalOea) {
    ids.push(
      ...addSegmentEnvelopeWidthMarkers({
        viewer,
        bundle,
        branchBundle,
        segmentBundle,
        baseId,
        segmentName,
        primary: segmentBundle.segmentGeometry.primaryEnvelope,
        secondaryOuter: segmentBundle.segmentGeometry.secondaryEnvelope ?? null,
        visible: visible && widthMeasurementVisible,
        displayLevel,
        diagnostics: segmentDiagnostics,
      }),
    );
  }

  segmentBundle.segmentGeometry.turnJunctions.forEach((junction) => {
    const turnPrimaryId = `${baseId}-turn-${junction.turnPointIndex}-primary`;
    const turnPrimaryAnnotation = annotationBase({
      entityId: turnPrimaryId,
      label: "Turn fill",
      title: `${segmentName} visual turn fill primary`,
      kind: "TURN_FILL",
      status: "VISUAL_FILL_ONLY",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: [
        param("Turn", `${Math.round(junction.turnAngleDeg)} deg ${junction.turnDirection}`),
        param("Station", `${junction.stationNm.toFixed(2)} NM`),
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      turnPrimaryId,
      `${segmentName} visual turn fill primary`,
      junction.primaryPatch.ribbon,
      procedureEntityShow(visible, turnPrimaryAnnotation, displayLevel),
      TURN_FILL_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
      turnPrimaryAnnotation,
    );
    ids.push(turnPrimaryId);

    if (junction.secondaryPatch) {
      const turnSecondaryId = `${baseId}-turn-${junction.turnPointIndex}-secondary`;
      const turnSecondaryAnnotation = {
        ...turnPrimaryAnnotation,
        entityId: turnSecondaryId,
        title: `${segmentName} visual turn fill secondary`,
      };
      addRibbonPolygon(
        viewer,
        turnSecondaryId,
        `${segmentName} visual turn fill secondary`,
        junction.secondaryPatch.ribbon,
        procedureEntityShow(visible, turnSecondaryAnnotation, displayLevel),
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
        turnSecondaryAnnotation,
      );
      ids.push(turnSecondaryId);
    }
  });

  if (!hasUnifiedSurfaces && segmentBundle.finalOea) {
    const protectionSurface = findProtectionSurface(branchBundle, segmentBundle.finalOea.geometryId);
    const oeaPrimaryId = `${baseId}-oea-primary`;
    const oeaAnnotation = annotationBase({
      entityId: oeaPrimaryId,
      label: "LNAV OEA",
      title: `${segmentName} LNAV OEA primary`,
      kind: "FINAL_OEA",
      status: "SOURCE_BACKED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: [
        ...segmentParams(segmentBundle.segment, segmentBundle.legs),
        ...protectionSurfaceParams(protectionSurface),
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      oeaPrimaryId,
      `${segmentName} LNAV OEA primary`,
      segmentBundle.finalOea.primary,
      procedureEntityShow(visible, oeaAnnotation, displayLevel),
      PRIMARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
      oeaAnnotation,
    );
    ids.push(oeaPrimaryId);
    const oeaLabelId = addAnnotationLabel(
      viewer,
      oeaAnnotation,
      representativePoint(segmentBundle.finalOea.primary.leftGeoBoundary),
      procedureEntityShow(visible, oeaAnnotation, displayLevel, true, annotationVisible),
    );
    if (oeaLabelId) ids.push(oeaLabelId);

    const oeaSecondaryId = `${baseId}-oea-secondary`;
    const oeaSecondaryAnnotation = {
      ...oeaAnnotation,
      entityId: oeaSecondaryId,
      title: `${segmentName} LNAV OEA secondary`,
    };
    addRibbonPolygon(
      viewer,
      oeaSecondaryId,
      `${segmentName} LNAV OEA secondary`,
      segmentBundle.finalOea.secondaryOuter,
      procedureEntityShow(visible, oeaSecondaryAnnotation, displayLevel),
      SECONDARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
      oeaSecondaryAnnotation,
    );
    ids.push(oeaSecondaryId);
    ids.push(
      ...addFinalOeaStationMarkers({
        viewer,
        bundle,
        branchBundle,
        segmentBundle,
        baseId,
        segmentName,
        centerline: segmentBundle.finalOea.centerline.geoPositions,
        primary: segmentBundle.finalOea.primary,
        secondaryOuter: segmentBundle.finalOea.secondaryOuter,
        widthSamples: finalOeaWidthSamplesFromRibbons(
          segmentBundle.finalOea.primary,
          segmentBundle.finalOea.secondaryOuter,
        ),
        visible: visible && widthMeasurementVisible,
        annotationVisible,
        displayLevel,
        diagnostics: segmentDiagnostics,
      }),
    );
  }

  if (!hasUnifiedSurfaces && segmentBundle.lnavVnavOcs) {
    const protectionSurface = findProtectionSurface(branchBundle, segmentBundle.lnavVnavOcs.geometryId);
    const ocsPrimaryId = `${baseId}-lnav-vnav-ocs-primary`;
    const ocsAnnotation = annotationBase({
      entityId: ocsPrimaryId,
      label: "LNAV/VNAV OCS",
      title: `${segmentName} LNAV/VNAV OCS primary`,
      kind: "LNAV_VNAV_OCS",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: [
        ...segmentParams(segmentBundle.segment, segmentBundle.legs),
        param("GPA", `${segmentBundle.lnavVnavOcs.verticalProfile.gpaDeg} deg`),
        param("TCH", `${segmentBundle.lnavVnavOcs.verticalProfile.tchFt} ft`),
        param("Status", segmentBundle.lnavVnavOcs.constructionStatus),
        ...protectionSurfaceParams(protectionSurface),
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      ocsPrimaryId,
      `${segmentName} LNAV/VNAV OCS primary`,
      segmentBundle.lnavVnavOcs.primary,
        procedureEntityShow(visible, ocsAnnotation, displayLevel),
        LNAV_VNAV_OCS_PRIMARY_COLOR,
        LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
        ocsAnnotation,
    );
    ids.push(ocsPrimaryId);
    ids.push(
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-lnav-vnav-ocs-primary-boundary`,
        `${segmentName} LNAV/VNAV OCS primary`,
        segmentBundle.lnavVnavOcs.primary,
        procedureEntityShow(visible, ocsAnnotation, displayLevel),
        LNAV_VNAV_OCS_EDGE_COLOR,
        LNAV_VNAV_OCS_EDGE_HEIGHT_OFFSET_M,
        ocsAnnotation,
      ),
      ...addRibbonCrossRibs(
        viewer,
        `${baseId}-lnav-vnav-ocs-primary`,
        `${segmentName} LNAV/VNAV OCS primary`,
        segmentBundle.lnavVnavOcs.primary,
        procedureEntityShow(visible, ocsAnnotation, displayLevel),
        LNAV_VNAV_OCS_RIB_COLOR,
        LNAV_VNAV_OCS_RIB_HEIGHT_OFFSET_M,
        ocsAnnotation,
      ),
    );
    const ocsLabelId = addAnnotationLabel(
      viewer,
      ocsAnnotation,
      representativePoint(segmentBundle.lnavVnavOcs.centerline.geoPositions),
      procedureEntityShow(visible, ocsAnnotation, displayLevel, true, annotationVisible),
    );
    if (ocsLabelId) ids.push(ocsLabelId);

    const ocsSecondaryId = `${baseId}-lnav-vnav-ocs-secondary`;
    const ocsSecondaryAnnotation = {
      ...ocsAnnotation,
      entityId: ocsSecondaryId,
      title: `${segmentName} LNAV/VNAV OCS secondary`,
    };
    addRibbonPolygon(
      viewer,
      ocsSecondaryId,
      `${segmentName} LNAV/VNAV OCS secondary`,
      segmentBundle.lnavVnavOcs.secondaryOuter,
      procedureEntityShow(visible, ocsSecondaryAnnotation, displayLevel),
      LNAV_VNAV_OCS_SECONDARY_COLOR,
      LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
      ocsSecondaryAnnotation,
    );
    ids.push(ocsSecondaryId);
    ids.push(
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-lnav-vnav-ocs-secondary-boundary`,
        `${segmentName} LNAV/VNAV OCS secondary`,
        segmentBundle.lnavVnavOcs.secondaryOuter,
        procedureEntityShow(visible, ocsSecondaryAnnotation, displayLevel),
        LNAV_VNAV_OCS_SECONDARY_EDGE_COLOR,
        LNAV_VNAV_OCS_EDGE_HEIGHT_OFFSET_M,
        ocsSecondaryAnnotation,
      ),
    );
  }

  if (!hasUnifiedSurfaces) {
    segmentBundle.precisionFinalSurfaces.forEach((surface) => {
      const protectionSurface = findProtectionSurface(branchBundle, surface.geometryId);
      const surfaceId = `${baseId}-precision-${surface.surfaceType.toLowerCase().replace(/_/g, "-")}`;
      const surfaceAnnotation = annotationBase({
        entityId: surfaceId,
        label: `${surface.surfaceType.replace(/_/g, " ")} estimate`,
        title: `${segmentName} ${surface.surfaceType} debug estimate`,
        kind: "PRECISION_SURFACE",
        status: "DEBUG_ESTIMATE",
        bundle,
        branchBundle,
        segment: segmentBundle.segment,
        legs: segmentBundle.legs,
        parameters: [
          param("Surface", surface.surfaceType),
          param("GPA", `${surface.verticalProfile.gpaDeg} deg`),
          param("TCH", `${surface.verticalProfile.tchFt} ft`),
          param("Status", surface.constructionStatus),
          ...protectionSurfaceParams(protectionSurface),
        ],
        diagnostics: [...segmentDiagnostics, ...surface.notes],
      });
      addRibbonPolygon(
        viewer,
        surfaceId,
        `${segmentName} ${surface.surfaceType} debug estimate`,
        surface.ribbon,
        procedureEntityShow(visible, surfaceAnnotation, displayLevel),
        PRECISION_FINAL_SURFACE_COLOR,
        PRECISION_FINAL_SURFACE_HEIGHT_OFFSET_M,
        surfaceAnnotation,
      );
      ids.push(surfaceId);
      const surfaceLabelId = addAnnotationLabel(
        viewer,
        surfaceAnnotation,
        representativePoint(surface.centerline.geoPositions),
        procedureEntityShow(visible, surfaceAnnotation, displayLevel, true, annotationVisible),
      );
      if (surfaceLabelId) ids.push(surfaceLabelId);
    });
  }

  if (
    segmentBundle.finalSurfaceStatus &&
    segmentBundle.finalSurfaceStatus.missingSurfaceTypes.length > 0
  ) {
    const statusPoint = representativePoint(segmentBundle.segmentGeometry.centerline.geoPositions);
    if (statusPoint) {
      const statusId = `${baseId}-final-surface-status`;
      const statusAnnotation = annotationBase({
        entityId: statusId,
        label: `Missing: ${segmentBundle.finalSurfaceStatus.missingSurfaceTypes.join(", ")}`,
        title: `${segmentName} missing final surfaces`,
        kind: "MISSING_FINAL_SURFACE",
        status: "MISSING_SOURCE",
        bundle,
        branchBundle,
        segment: segmentBundle.segment,
        legs: segmentBundle.legs,
        parameters: [
          param("Requested", segmentBundle.finalSurfaceStatus.requestedModes.join(", ")),
          param("Constructed", segmentBundle.finalSurfaceStatus.constructedSurfaceTypes.join(", ")),
          param("Missing", segmentBundle.finalSurfaceStatus.missingSurfaceTypes.join(", ")),
        ],
        diagnostics: [...segmentDiagnostics, ...segmentBundle.finalSurfaceStatus.notes],
      });
      addPoint(
        viewer,
        statusId,
        `${segmentName} missing final surfaces: ${segmentBundle.finalSurfaceStatus.missingSurfaceTypes.join(", ")}`,
        statusPoint,
        procedureEntityShow(visible, statusAnnotation, displayLevel),
        11,
        FINAL_SURFACE_STATUS_COLOR,
        FINAL_SURFACE_STATUS_HEIGHT_OFFSET_M,
        statusAnnotation,
      );
      ids.push(statusId);
      const statusLabelId = addAnnotationLabel(
        viewer,
        statusAnnotation,
        elevatedPoint(statusPoint, FINAL_SURFACE_STATUS_HEIGHT_OFFSET_M),
        procedureEntityShow(visible, statusAnnotation, displayLevel, true, annotationVisible),
      );
      if (statusLabelId) ids.push(statusLabelId);
    }
  }

  if (segmentBundle.alignedConnector) {
    const connectorPrimaryId = `${baseId}-connector-primary`;
    const connectorAnnotation = annotationBase({
      entityId: connectorPrimaryId,
      label: "Connector",
      title: `${segmentName} aligned connector primary`,
      kind: "ALIGNED_CONNECTOR",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: segmentParams(segmentBundle.segment, segmentBundle.legs),
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      connectorPrimaryId,
      `${segmentName} aligned connector primary`,
      segmentBundle.alignedConnector.primary,
      procedureEntityShow(visible, connectorAnnotation, displayLevel),
      CONNECTOR_COLOR,
      ALIGNED_CONNECTOR_FILL_HEIGHT_OFFSET_M,
      connectorAnnotation,
    );
    ids.push(connectorPrimaryId);
    const connectorLabelId = addAnnotationLabel(
      viewer,
      connectorAnnotation,
      representativePoint(segmentBundle.alignedConnector.primary.leftGeoBoundary),
      procedureEntityShow(visible, connectorAnnotation, displayLevel, true, annotationVisible),
    );
    if (connectorLabelId) ids.push(connectorLabelId);

    const connectorSecondaryId = `${baseId}-connector-secondary`;
    const connectorSecondaryAnnotation = {
      ...connectorAnnotation,
      entityId: connectorSecondaryId,
      title: `${segmentName} aligned connector secondary`,
    };
    addRibbonPolygon(
      viewer,
      connectorSecondaryId,
      `${segmentName} aligned connector secondary`,
      segmentBundle.alignedConnector.secondaryOuter,
      procedureEntityShow(visible, connectorSecondaryAnnotation, displayLevel),
      CONNECTOR_COLOR,
      ALIGNED_CONNECTOR_FILL_HEIGHT_OFFSET_M,
      connectorSecondaryAnnotation,
    );
    ids.push(connectorSecondaryId);

    ids.push(
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-connector-primary-boundary`,
        `${segmentName} aligned connector primary`,
        segmentBundle.alignedConnector.primary,
        procedureEntityShow(visible, connectorAnnotation, displayLevel),
        CONNECTOR_LINE_COLOR,
        ALIGNED_CONNECTOR_LINE_HEIGHT_OFFSET_M,
        connectorAnnotation,
      ),
      ...addRibbonCrossRibs(
        viewer,
        `${baseId}-connector-primary`,
        `${segmentName} aligned connector primary`,
        segmentBundle.alignedConnector.primary,
        procedureEntityShow(visible, connectorAnnotation, displayLevel),
        CONNECTOR_LINE_COLOR,
        ALIGNED_CONNECTOR_LINE_HEIGHT_OFFSET_M,
        connectorAnnotation,
      ),
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-connector-secondary-boundary`,
        `${segmentName} aligned connector secondary`,
        segmentBundle.alignedConnector.secondaryOuter,
        procedureEntityShow(visible, connectorSecondaryAnnotation, displayLevel),
        CONNECTOR_LINE_COLOR,
        ALIGNED_CONNECTOR_LINE_HEIGHT_OFFSET_M,
        connectorSecondaryAnnotation,
      ),
    );
  }

  if (!hasUnifiedSurfaces && segmentBundle.missedSectionSurface) {
    const protectionSurface = findProtectionSurface(
      branchBundle,
      protectionSurfaceIdForMissedSection(
        segmentBundle.missedSectionSurface.segmentId,
        segmentBundle.missedSectionSurface.surfaceType,
      ),
    );
    const isEstimatedCaSurface =
      segmentBundle.missedSectionSurface.constructionStatus === "ESTIMATED_CA";
    const missedSurfaceColor = isEstimatedCaSurface
      ? MISSED_CA_ESTIMATED_SURFACE_COLOR
      : MISSED_SURFACE_COLOR;
    const missedSurfaceName = isEstimatedCaSurface
      ? `${segmentName} CA estimated missed section`
      : `${segmentName} missed section`;
    const missedPrimaryId = `${baseId}-missed-surface-primary`;
    const missedAnnotation = annotationBase({
      entityId: missedPrimaryId,
      label: isEstimatedCaSurface ? "Missed CA estimate" : "Missed surface",
      title: `${missedSurfaceName} primary`,
      kind: "MISSED_SURFACE",
      status: isEstimatedCaSurface ? "ESTIMATED" : "SOURCE_BACKED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: [
        param("Surface", segmentBundle.missedSectionSurface.surfaceType),
        param("Status", segmentBundle.missedSectionSurface.constructionStatus),
        ...protectionSurfaceParams(protectionSurface),
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      missedPrimaryId,
      `${missedSurfaceName} primary`,
      segmentBundle.missedSectionSurface.primary,
      procedureEntityShow(visible, missedAnnotation, displayLevel),
      missedSurfaceColor,
      MISSED_SURFACE_HEIGHT_OFFSET_M,
      missedAnnotation,
    );
    ids.push(missedPrimaryId);
    const missedLabelId = addAnnotationLabel(
      viewer,
      missedAnnotation,
      representativePoint(segmentBundle.missedSectionSurface.primary.leftGeoBoundary),
      procedureEntityShow(visible, missedAnnotation, displayLevel, true, annotationVisible),
    );
    if (missedLabelId) ids.push(missedLabelId);

    const missedSecondaryId = `${baseId}-missed-surface-secondary`;
    const missedSecondaryAnnotation = {
      ...missedAnnotation,
      entityId: missedSecondaryId,
      title: `${missedSurfaceName} secondary`,
    };
    addRibbonPolygon(
      viewer,
      missedSecondaryId,
      `${missedSurfaceName} secondary`,
      segmentBundle.missedSectionSurface.secondaryOuter ?? undefined,
      procedureEntityShow(visible, missedSecondaryAnnotation, displayLevel),
      missedSurfaceColor,
      MISSED_SURFACE_HEIGHT_OFFSET_M,
      missedSecondaryAnnotation,
    );
    ids.push(missedSecondaryId);
  }

  segmentBundle.missedCourseGuides.forEach((guide) => {
    const guideId = `${baseId}-ca-course-guide-${guide.legId}`;
    const leg = segmentBundle.legs.find((candidate) => candidate.legId === guide.legId);
    const guideAnnotation = annotationBase({
      entityId: guideId,
      label: `CA guide ${Math.round(guide.courseDeg)} deg`,
      title: `${segmentName} CA course guide`,
      kind: "CA_COURSE_GUIDE",
      status: "SOURCE_BACKED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      leg,
      parameters: [
        param("Course", `${guide.courseDeg} deg`),
        param("Guide length", `${guide.guideLengthNm} NM`),
        param("Required altitude", guide.requiredAltitudeFtMsl === null ? null : `${guide.requiredAltitudeFtMsl} ft`),
        param("Start fix", guide.startFixId),
      ],
      diagnostics: segmentDiagnostics,
    });
    addPolyline(
      viewer,
      guideId,
      `${segmentName} CA course guide ${Math.round(guide.courseDeg)} deg`,
      guide.geoPositions.map((point) => elevatedPoint(point, CA_COURSE_GUIDE_HEIGHT_OFFSET_M)),
      procedureEntityShow(visible, guideAnnotation, displayLevel),
      5,
      CA_COURSE_GUIDE_COLOR,
      guideAnnotation,
    );
    ids.push(guideId);
    const guideLabelId = addAnnotationLabel(
      viewer,
      guideAnnotation,
      representativePoint(guide.geoPositions.map((point) => elevatedPoint(point, CA_COURSE_GUIDE_HEIGHT_OFFSET_M))),
      procedureEntityShow(visible, guideAnnotation, displayLevel, true, annotationVisible),
    );
    if (guideLabelId) ids.push(guideLabelId);
  });

  segmentBundle.missedCaCenterlines.forEach((centerline) => {
    const centerlineId = `${baseId}-ca-centerline-${centerline.legId}`;
    const leg = segmentBundle.legs.find((candidate) => candidate.legId === centerline.legId);
    const caCenterlineAnnotation = annotationBase({
      entityId: centerlineId,
      label: "CA centerline estimate",
      title: `${segmentName} CA estimated centerline`,
      kind: "CA_CENTERLINE",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      leg,
      parameters: [
        param("Source endpoint", centerline.sourceEndpointStatus),
        param("Length", `${centerline.geodesicLengthNm.toFixed(2)} NM`),
      ],
      diagnostics: [...segmentDiagnostics, ...centerline.notes],
    });
    addPolyline(
      viewer,
      centerlineId,
      `${segmentName} CA estimated centerline`,
      centerline.geoPositions.map((point) => elevatedPoint(point, CA_CENTERLINE_HEIGHT_OFFSET_M)),
      procedureEntityShow(visible, caCenterlineAnnotation, displayLevel),
      4,
      CA_CENTERLINE_COLOR,
      caCenterlineAnnotation,
    );
    ids.push(centerlineId);
    const centerlineLabelId = addAnnotationLabel(
      viewer,
      caCenterlineAnnotation,
      representativePoint(centerline.geoPositions.map((point) => elevatedPoint(point, CA_CENTERLINE_HEIGHT_OFFSET_M))),
      procedureEntityShow(visible, caCenterlineAnnotation, displayLevel, true, annotationVisible),
    );
    if (centerlineLabelId) ids.push(centerlineLabelId);
  });

  segmentBundle.missedCaEndpoints.forEach((endpoint) => {
    const endpointId = `${baseId}-ca-endpoint-${endpoint.legId}`;
    const leg = segmentBundle.legs.find((candidate) => candidate.legId === endpoint.legId);
    const endpointAnnotation = annotationBase({
      entityId: endpointId,
      label: `CA ${Math.round(endpoint.targetAltitudeFtMsl)} ft`,
      title: `${segmentName} CA estimated endpoint`,
      kind: "CA_ENDPOINT",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      leg,
      parameters: [
        param("Course", `${endpoint.courseDeg} deg`),
        param("Target altitude", `${endpoint.targetAltitudeFtMsl} ft`),
        param("Start altitude", `${endpoint.startAltitudeFtMsl} ft`),
        param("Climb gradient", `${endpoint.climbGradientFtPerNm} ft/NM`),
        param("Distance", `${endpoint.distanceNm.toFixed(2)} NM`),
      ],
      diagnostics: [...segmentDiagnostics, ...endpoint.notes],
    });
    addPoint(
      viewer,
      endpointId,
      `${segmentName} CA estimated endpoint ${Math.round(endpoint.targetAltitudeFtMsl)} ft`,
      endpoint.geoPositions[1],
      procedureEntityShow(visible, endpointAnnotation, displayLevel),
      10,
      CA_ENDPOINT_COLOR,
      CA_ENDPOINT_HEIGHT_OFFSET_M,
      endpointAnnotation,
    );
    ids.push(endpointId);
    const endpointLabelId = addAnnotationLabel(
      viewer,
      endpointAnnotation,
      elevatedPoint(endpoint.geoPositions[1], CA_ENDPOINT_HEIGHT_OFFSET_M),
      procedureEntityShow(visible, endpointAnnotation, displayLevel, true, annotationVisible),
    );
    if (endpointLabelId) ids.push(endpointLabelId);
  });

  if (segmentBundle.missedTurnDebugPoint) {
    const turnDebugId = `${baseId}-turning-missed-anchor`;
    const debugAnnotation = annotationBase({
      entityId: turnDebugId,
      label: "Turning missed debug",
      title: `${segmentName} turning missed debug anchor`,
      kind: "TURNING_MISSED_DEBUG",
      status: "DEBUG_ESTIMATE",
      bundle,
      branchBundle,
      segment: segmentBundle.segment,
      legs: segmentBundle.legs,
      parameters: [
        param("Anchor", segmentBundle.missedTurnDebugPoint.anchorFixId),
        param("Triggers", segmentBundle.missedTurnDebugPoint.triggerLegTypes.join(", ")),
      ],
      diagnostics: segmentDiagnostics,
    });
    addPoint(
      viewer,
      turnDebugId,
      `${segmentName} turning missed debug anchor`,
      segmentBundle.missedTurnDebugPoint.geoPosition,
      procedureEntityShow(visible, debugAnnotation, displayLevel),
      12,
      TURNING_MISSED_DEBUG_COLOR,
      TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M,
      debugAnnotation,
    );
    ids.push(turnDebugId);
    const debugLabelId = addAnnotationLabel(
      viewer,
      debugAnnotation,
      elevatedPoint(segmentBundle.missedTurnDebugPoint.geoPosition, TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M),
      procedureEntityShow(visible, debugAnnotation, displayLevel, true, annotationVisible),
    );
    if (debugLabelId) ids.push(debugLabelId);
  }

  if (!hasUnifiedTurningMissedDebugSurfaces) {
    segmentBundle.missedTurnDebugPrimitives.forEach((primitive) => {
      const primitiveId = `${baseId}-turning-missed-${primitive.debugType.toLowerCase().replace(/_/g, "-")}`;
      const leg = segmentBundle.legs.find((candidate) => candidate.legId === primitive.legId);
      const primitiveAnnotation = annotationBase({
        entityId: primitiveId,
        label: primitive.debugType.replace(/_/g, " "),
        title: `${segmentName} turning missed ${primitive.debugType.toLowerCase().replace(/_/g, " ")}`,
        kind: "TURNING_MISSED_DEBUG",
        status: "DEBUG_ESTIMATE",
        bundle,
        branchBundle,
        segment: segmentBundle.segment,
        legs: segmentBundle.legs,
        leg,
        parameters: [
          param("Debug type", primitive.debugType),
          param("Turn trigger", primitive.turnTrigger),
          param("Turn case", primitive.turnCase),
          param("Course", `${primitive.courseDeg} deg`),
          param("Turn direction", primitive.turnDirection),
        ],
        diagnostics: [...segmentDiagnostics, ...primitive.notes],
      });
      addPolyline(
        viewer,
        primitiveId,
        `${segmentName} turning missed ${primitive.debugType.toLowerCase().replace(/_/g, " ")}`,
        primitive.geoPositions.map((point) => elevatedPoint(point, TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M)),
        procedureEntityShow(visible, primitiveAnnotation, displayLevel),
        primitive.debugType === "NOMINAL_TURN_PATH" ? 4 : 3,
        TURNING_MISSED_PRIMITIVE_COLOR,
        primitiveAnnotation,
      );
      ids.push(primitiveId);
      const primitiveLabelId = addAnnotationLabel(
        viewer,
        primitiveAnnotation,
        representativePoint(primitive.geoPositions.map((point) => elevatedPoint(point, TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M))),
        procedureEntityShow(visible, primitiveAnnotation, displayLevel, true, annotationVisible),
      );
      if (primitiveLabelId) ids.push(primitiveLabelId);
    });
  }

  return ids;
}

function addBranchTurnJunctionEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
  annotationVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];

  branchBundle.turnJunctions.forEach((junction) => {
    const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${junction.geometryId}`;
    const junctionName = `${bundle.procedureName} visual inter-segment turn fill`;

    const primaryId = `${baseId}-primary`;
    const primaryAnnotation = annotationBase({
      entityId: primaryId,
      label: "Inter-segment turn fill",
      title: `${junctionName} primary`,
      kind: "TURN_FILL",
      status: "VISUAL_FILL_ONLY",
      bundle,
      branchBundle,
      parameters: [
        param("From", junction.fromSegmentId),
        param("To", junction.toSegmentId),
        param("Turn", `${Math.round(junction.turnAngleDeg)} deg ${junction.turnDirection}`),
      ],
    });
    addRibbonPolygon(
      viewer,
      primaryId,
      `${junctionName} primary`,
      junction.primaryPatch.ribbon,
      procedureEntityShow(visible, primaryAnnotation, displayLevel),
      TURN_FILL_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
      primaryAnnotation,
    );
    ids.push(primaryId);
    const labelId = addAnnotationLabel(
      viewer,
      primaryAnnotation,
      representativePoint(junction.primaryPatch.ribbon.leftGeoBoundary),
      procedureEntityShow(visible, primaryAnnotation, displayLevel, true, annotationVisible),
    );
    if (labelId) ids.push(labelId);

    if (junction.secondaryPatch) {
      const secondaryId = `${baseId}-secondary`;
      const secondaryAnnotation = {
        ...primaryAnnotation,
        entityId: secondaryId,
        title: `${junctionName} secondary`,
      };
      addRibbonPolygon(
        viewer,
        secondaryId,
        `${junctionName} secondary`,
        junction.secondaryPatch.ribbon,
        procedureEntityShow(visible, secondaryAnnotation, displayLevel),
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
        secondaryAnnotation,
      );
      ids.push(secondaryId);
    }
  });

  return ids;
}

function addBranchProtectionSurfaceEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
  annotationVisible: boolean,
  widthMeasurementVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  const protectionSurfaces = branchBundle.protectionSurfaces ?? [];
  const missedConnectorSurfaces = protectionSurfaces.filter(
    (surface) => surface.kind === "MISSED_CONNECTOR",
  );

  protectionSurfaces.forEach((surface) => {
    const segmentBundle = segmentBundleForSurface(branchBundle, surface);
    const baseId = segmentBaseEntityId(bundle, surface.segmentId);
    const segmentName = `${bundle.procedureName} ${segmentBundle?.segment.segmentType ?? surface.kind}`;
    const diagnostics = diagnosticsForProtectionSurface(surface, segmentBundle);
    const status = protectionSurfaceAnnotationStatus(surface);
    const commonParameters = [
      ...surfaceSegmentParams(segmentBundle),
      ...protectionSurfaceParams(surface),
    ];

    if (surface.kind === "FINAL_LNAV_OEA") {
      const primaryId = `${baseId}-oea-primary`;
      const annotation = annotationBase({
        entityId: primaryId,
        label: "LNAV OEA",
        title: `${segmentName} LNAV OEA primary`,
        kind: "FINAL_OEA",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: commonParameters,
        diagnostics,
      });
      addRibbonPolygon(
        viewer,
        primaryId,
        `${segmentName} LNAV OEA primary`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        PRIMARY_COLOR,
        OEA_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(primaryId);
      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.lateral.primary.leftGeoBoundary),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);

      if (surface.lateral.secondaryOuter) {
        const secondaryId = `${baseId}-oea-secondary`;
        const secondaryAnnotation = {
          ...annotation,
          entityId: secondaryId,
          title: `${segmentName} LNAV OEA secondary`,
        };
        addRibbonPolygon(
          viewer,
          secondaryId,
          `${segmentName} LNAV OEA secondary`,
          surface.lateral.secondaryOuter,
          procedureEntityShow(visible, secondaryAnnotation, displayLevel),
          SECONDARY_COLOR,
          OEA_HEIGHT_OFFSET_M,
          secondaryAnnotation,
        );
        ids.push(secondaryId);
      }
      ids.push(
        ...addFinalOeaStationMarkers({
          viewer,
          bundle,
          branchBundle,
          segmentBundle,
          baseId,
          segmentName,
          centerline: surface.centerline.geoPositions,
          primary: surface.lateral.primary,
          secondaryOuter: surface.lateral.secondaryOuter,
          widthSamples: surface.lateral.widthSamples,
          visible: visible && widthMeasurementVisible,
          annotationVisible,
          displayLevel,
          diagnostics,
        }),
      );
      return;
    }

    if (surface.kind === "FINAL_LNAV_VNAV_OCS") {
      const primaryId = `${baseId}-lnav-vnav-ocs-primary`;
      const annotation = annotationBase({
        entityId: primaryId,
        label: "LNAV/VNAV OCS",
        title: `${segmentName} LNAV/VNAV OCS primary`,
        kind: "LNAV_VNAV_OCS",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: commonParameters,
        diagnostics,
      });
      addRibbonPolygon(
        viewer,
        primaryId,
        `${segmentName} LNAV/VNAV OCS primary`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        LNAV_VNAV_OCS_PRIMARY_COLOR,
        LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(primaryId);
      ids.push(
        ...addRibbonBoundaryPolylines(
          viewer,
          `${baseId}-lnav-vnav-ocs-primary-boundary`,
          `${segmentName} LNAV/VNAV OCS primary`,
          surface.lateral.primary,
          procedureEntityShow(visible, annotation, displayLevel),
          LNAV_VNAV_OCS_EDGE_COLOR,
          LNAV_VNAV_OCS_EDGE_HEIGHT_OFFSET_M,
          annotation,
        ),
        ...addRibbonCrossRibs(
          viewer,
          `${baseId}-lnav-vnav-ocs-primary`,
          `${segmentName} LNAV/VNAV OCS primary`,
          surface.lateral.primary,
          procedureEntityShow(visible, annotation, displayLevel),
          LNAV_VNAV_OCS_RIB_COLOR,
          LNAV_VNAV_OCS_RIB_HEIGHT_OFFSET_M,
          annotation,
        ),
      );
      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.centerline.geoPositions),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);

      if (surface.lateral.secondaryOuter) {
        const secondaryId = `${baseId}-lnav-vnav-ocs-secondary`;
        const secondaryAnnotation = {
          ...annotation,
          entityId: secondaryId,
          title: `${segmentName} LNAV/VNAV OCS secondary`,
        };
        addRibbonPolygon(
          viewer,
          secondaryId,
          `${segmentName} LNAV/VNAV OCS secondary`,
          surface.lateral.secondaryOuter,
          procedureEntityShow(visible, secondaryAnnotation, displayLevel),
          LNAV_VNAV_OCS_SECONDARY_COLOR,
          LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
          secondaryAnnotation,
        );
        ids.push(secondaryId);
        ids.push(
          ...addRibbonBoundaryPolylines(
            viewer,
            `${baseId}-lnav-vnav-ocs-secondary-boundary`,
            `${segmentName} LNAV/VNAV OCS secondary`,
            surface.lateral.secondaryOuter,
            procedureEntityShow(visible, secondaryAnnotation, displayLevel),
            LNAV_VNAV_OCS_SECONDARY_EDGE_COLOR,
            LNAV_VNAV_OCS_EDGE_HEIGHT_OFFSET_M,
            secondaryAnnotation,
          ),
        );
      }
      return;
    }

    if (surface.kind === "FINAL_PRECISION_DEBUG") {
      const label = precisionSurfaceLabel(surface);
      const surfaceId = `${baseId}-precision-${surface.surfaceId.slice(surface.segmentId.length + 1)}`;
      const annotation = annotationBase({
        entityId: surfaceId,
        label: `${label} estimate`,
        title: `${segmentName} ${label} debug estimate`,
        kind: "PRECISION_SURFACE",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: commonParameters,
        diagnostics,
      });
      addRibbonPolygon(
        viewer,
        surfaceId,
        `${segmentName} ${label} debug estimate`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        PRECISION_FINAL_SURFACE_COLOR,
        PRECISION_FINAL_SURFACE_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(surfaceId);
      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.centerline.geoPositions),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);
      return;
    }

    if (surface.kind === "TURNING_MISSED_DEBUG") {
      const suffix = entityIdSuffix(
        surface.surfaceId.slice(surface.segmentId.length + 1) || surface.surfaceId,
      );
      const surfaceEntityId = `${baseId}-turning-missed-debug-${suffix}`;
      const annotation = annotationBase({
        entityId: surfaceEntityId,
        label: "Turning missed debug",
        title: `${segmentName} turning missed debug surface`,
        kind: "TURNING_MISSED_DEBUG",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: [
          param("Geometry meaning", "Debug placeholder only; not certified TIA, wind spiral, or TERPS protection"),
          ...commonParameters,
        ],
        diagnostics,
      });
      addRibbonPolygon(
        viewer,
        surfaceEntityId,
        `${segmentName} turning missed debug surface`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        TURNING_MISSED_DEBUG_SURFACE_COLOR,
        TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(surfaceEntityId);
      ids.push(
        ...addRibbonBoundaryPolylines(
          viewer,
          `${surfaceEntityId}-boundary`,
          `${segmentName} turning missed debug surface`,
          surface.lateral.primary,
          procedureEntityShow(visible, annotation, displayLevel),
          TURNING_MISSED_PRIMITIVE_COLOR,
          TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M + 2,
          annotation,
        ),
      );
      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.centerline.geoPositions),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);
      return;
    }

    if (surface.kind === "MISSED_SECTION_1" || surface.kind === "MISSED_SECTION_2_STRAIGHT") {
      const estimated = surface.status !== "SOURCE_BACKED";
      const color = estimated ? MISSED_CA_ESTIMATED_SURFACE_COLOR : MISSED_SURFACE_COLOR;
      const name = estimated
        ? `${segmentName} CA estimated missed section`
        : `${segmentName} missed section`;
      const primaryId = `${baseId}-missed-surface-primary`;
      const annotation = annotationBase({
        entityId: primaryId,
        label: estimated ? "Missed CA estimate" : "Missed surface",
        title: `${name} primary`,
        kind: "MISSED_SURFACE",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: commonParameters,
        diagnostics,
      });
      addRibbonPolygon(
        viewer,
        primaryId,
        `${name} primary`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        color,
        MISSED_SURFACE_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(primaryId);
      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.lateral.primary.leftGeoBoundary),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);

      if (surface.lateral.secondaryOuter) {
        const secondaryId = `${baseId}-missed-surface-secondary`;
        const secondaryAnnotation = {
          ...annotation,
          entityId: secondaryId,
          title: `${name} secondary`,
        };
        addRibbonPolygon(
          viewer,
          secondaryId,
          `${name} secondary`,
          surface.lateral.secondaryOuter,
          procedureEntityShow(visible, secondaryAnnotation, displayLevel),
          color,
          MISSED_SURFACE_HEIGHT_OFFSET_M,
          secondaryAnnotation,
        );
        ids.push(secondaryId);
      }
      return;
    }

    if (surface.kind === "MISSED_CONNECTOR") {
      const connectorIndex = missedConnectorSurfaces.findIndex(
        (candidate) => candidate.surfaceId === surface.surfaceId,
      );
      const index = connectorIndex < 0 ? 0 : connectorIndex;
      const baseConnectorId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${branchBundle.branchId}-missed-connector-surface-${index}`;
      const connector = branchBundle.missedConnectorSurfaces.find(
        (candidate) => candidate.surfaceId === surface.surfaceId,
      );
      const primaryId = `${baseConnectorId}-primary`;
      const annotation = annotationBase({
        entityId: primaryId,
        label: connector ? `Connector to ${connector.targetFixIdent}` : "Missed connector",
        title: `${bundle.procedureName} estimated missed connector surface`,
        kind: "MISSED_CONNECTOR_SURFACE",
        status,
        bundle,
        branchBundle,
        segment: segmentBundle?.segment,
        legs: segmentBundle?.legs,
        parameters: [
          param("Source CA leg", connector?.sourceLegId ?? surface.sourceLegIds.join(", ")),
          param("Endpoint status", connector?.sourceEndpointStatus),
          param("Target fix", connector ? `${connector.targetFixIdent} ${connector.targetFixRole}` : null),
          param("Distance", `${surface.centerline.geodesicLengthNm.toFixed(2)} NM`),
          param("Geometry meaning", "Estimated connector surface; not certified TERPS construction"),
          ...commonParameters,
        ],
        diagnostics: connector?.notes ?? diagnostics,
      });
      addRibbonPolygon(
        viewer,
        primaryId,
        `${bundle.procedureName} estimated missed connector surface primary`,
        surface.lateral.primary,
        procedureEntityShow(visible, annotation, displayLevel),
        MISSED_CONNECTOR_SURFACE_COLOR,
        MISSED_SURFACE_HEIGHT_OFFSET_M,
        annotation,
      );
      ids.push(primaryId);

      const labelId = addAnnotationLabel(
        viewer,
        annotation,
        representativePoint(surface.lateral.primary.leftGeoBoundary),
        procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
      );
      if (labelId) ids.push(labelId);

      if (surface.lateral.secondaryOuter) {
        const secondaryId = `${baseConnectorId}-secondary`;
        const secondaryAnnotation = {
          ...annotation,
          entityId: secondaryId,
          title: `${bundle.procedureName} estimated missed connector surface secondary`,
        };
        addRibbonPolygon(
          viewer,
          secondaryId,
          `${bundle.procedureName} estimated missed connector surface secondary`,
          surface.lateral.secondaryOuter,
          procedureEntityShow(visible, secondaryAnnotation, displayLevel),
          MISSED_CONNECTOR_SURFACE_COLOR,
          MISSED_SURFACE_HEIGHT_OFFSET_M,
          secondaryAnnotation,
        );
        ids.push(secondaryId);
      }
    }
  });

  return ids;
}

function addBranchVerticalProfileEntity(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  pkg: ProcedurePackage | null,
  visible: boolean,
  annotationVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const verticalProfileSections = branchVerticalProfileSections(branchBundle, pkg);
  if (verticalProfileSections.length === 0) return [];

  const ids: string[] = [];
  const baseEntityId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${branchBundle.branchId}-vertical-profile`;
  const hasMultipleSections = verticalProfileSections.length > 1;

  verticalProfileSections.forEach((section, index) => {
    const verticalProfileRibbon = buildVerticalProfileRibbon(
      `${branchBundle.branchId}:vertical-profile:${section.sectionId}`,
      section.points,
    );
    if (!verticalProfileRibbon) return;

    const entityId = hasMultipleSections ? `${baseEntityId}-${index + 1}` : baseEntityId;
    const sectionSegmentIdSet = new Set(section.segmentIds);
    const sectionSegmentBundles = branchBundle.segmentBundles.filter((segmentBundle) =>
      sectionSegmentIdSet.has(segmentBundle.segment.segmentId),
    );
    const sectionLegs = sectionSegmentBundles.flatMap((segmentBundle) => segmentBundle.legs);
    const diagnostics = sectionSegmentBundles.flatMap((segmentBundle) =>
      segmentBundle.diagnostics.map((diagnostic) => diagnostic.message),
    );
    const profileTitle = `${bundle.procedureName} ${branchBundle.branchName} fix vertical profile aid`;
    const annotation = annotationBase({
      entityId,
      label: "Profile aid",
      title: hasMultipleSections ? `${profileTitle} ${index + 1}` : profileTitle,
      kind: "SEGMENT_VERTICAL_PROFILE",
      status: "PROFILE_AID",
      bundle,
      branchBundle,
      legs: sectionLegs,
      parameters: [
        param("Fixes", section.points.map((point) => point.fixIdent).join(" -> ")),
        param("Segments", section.segmentTypes.map(compactSegmentType).join(" -> ")),
        param("Aid type", "Fix altitude profile"),
        param("Width basis", "Primary footprint width for readability"),
        param("Protection meaning", "Display aid only; not OEA, OCS, TERPS, or a protected surface"),
      ],
      diagnostics,
      sourceRefs: sourceRefsFromLegs(sectionLegs),
    });

    addRibbonPolygon(
      viewer,
      entityId,
      annotation.title,
      verticalProfileRibbon,
      procedureEntityShow(visible, annotation, displayLevel),
      SEGMENT_VERTICAL_PROFILE_AID_COLOR,
      SEGMENT_VERTICAL_PROFILE_HEIGHT_OFFSET_M,
      annotation,
    );
    ids.push(entityId);

    const labelId = addAnnotationLabel(
      viewer,
      annotation,
      representativePoint(section.points),
      procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
    );
    if (labelId) ids.push(labelId);
  });

  return ids;
}

function addBranchMissedCaMahfConnectorEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
  annotationVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  (branchBundle.missedCaMahfConnectors ?? []).forEach((connector, index) => {
    const connectorId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${branchBundle.branchId}-ca-mahf-connector-${index}`;
    const annotation = annotationBase({
      entityId: connectorId,
      label: `CA to ${connector.targetFixIdent}`,
      title: `${bundle.procedureName} estimated CA to MAHF connector`,
      kind: "CA_MAHF_CONNECTOR",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      parameters: [
        param("Source CA leg", connector.sourceLegId),
        param("Endpoint status", connector.sourceEndpointStatus),
        param("Target fix", `${connector.targetFixIdent} ${connector.targetFixRole}`),
        param("Distance", `${connector.geodesicLengthNm.toFixed(2)} NM`),
        param("Geometry meaning", "Estimated continuity connector; not source-coded leg geometry"),
      ],
      diagnostics: connector.notes,
    });
    addPolyline(
      viewer,
      connectorId,
      `${bundle.procedureName} estimated CA to ${connector.targetFixIdent} connector`,
      connector.geoPositions.map((point) => elevatedPoint(point, CA_MAHF_CONNECTOR_HEIGHT_OFFSET_M)),
      procedureEntityShow(visible, annotation, displayLevel),
      4,
      CA_MAHF_CONNECTOR_COLOR,
      annotation,
    );
    ids.push(connectorId);
    const labelId = addAnnotationLabel(
      viewer,
      annotation,
      representativePoint(connector.geoPositions.map((point) => elevatedPoint(point, CA_MAHF_CONNECTOR_HEIGHT_OFFSET_M))),
      procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
    );
    if (labelId) ids.push(labelId);
  });
  return ids;
}

function addBranchMissedConnectorSurfaceEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
  annotationVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  (branchBundle.missedConnectorSurfaces ?? []).forEach((surface, index) => {
    const protectionSurface = findProtectionSurface(branchBundle, surface.surfaceId);
    const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${branchBundle.branchId}-missed-connector-surface-${index}`;
    const primaryId = `${baseId}-primary`;
    const annotation = annotationBase({
      entityId: primaryId,
      label: `Connector to ${surface.targetFixIdent}`,
      title: `${bundle.procedureName} estimated missed connector surface`,
      kind: "MISSED_CONNECTOR_SURFACE",
      status: "ESTIMATED",
      bundle,
      branchBundle,
      parameters: [
        param("Source CA leg", surface.sourceLegId),
        param("Endpoint status", surface.sourceEndpointStatus),
        param("Target fix", `${surface.targetFixIdent} ${surface.targetFixRole}`),
        param("Distance", `${surface.centerline.geodesicLengthNm.toFixed(2)} NM`),
        param("Surface status", surface.constructionStatus),
        param("Vertical", surface.verticalProfile.constructionStatus),
        param("Geometry meaning", "Estimated connector surface; not certified TERPS construction"),
        ...protectionSurfaceParams(protectionSurface),
      ],
      diagnostics: surface.notes,
    });
    addRibbonPolygon(
      viewer,
      primaryId,
      `${bundle.procedureName} estimated missed connector surface primary`,
      surface.primary,
      procedureEntityShow(visible, annotation, displayLevel),
      MISSED_CONNECTOR_SURFACE_COLOR,
      MISSED_SURFACE_HEIGHT_OFFSET_M,
      annotation,
    );
    ids.push(primaryId);

    const labelId = addAnnotationLabel(
      viewer,
      annotation,
      representativePoint(surface.primary.leftGeoBoundary),
      procedureEntityShow(visible, annotation, displayLevel, true, annotationVisible),
    );
    if (labelId) ids.push(labelId);

    if (surface.secondaryOuter) {
      const secondaryId = `${baseId}-secondary`;
      const secondaryAnnotation = {
        ...annotation,
        entityId: secondaryId,
        title: `${bundle.procedureName} estimated missed connector surface secondary`,
      };
      addRibbonPolygon(
        viewer,
        secondaryId,
        `${bundle.procedureName} estimated missed connector surface secondary`,
        surface.secondaryOuter,
        procedureEntityShow(visible, secondaryAnnotation, displayLevel),
        MISSED_CONNECTOR_SURFACE_COLOR,
        MISSED_SURFACE_HEIGHT_OFFSET_M,
        secondaryAnnotation,
      );
      ids.push(secondaryId);
    }
  });
  return ids;
}

function packageBranchDefaultVisible(
  pkg: ProcedurePackage | null,
  branchId: string,
): boolean {
  return pkg?.branches.find((branch) => branch.branchId === branchId)?.legacy.defaultVisible ?? true;
}

function packageBranchVisible(
  pkg: ProcedurePackage | null,
  branchId: string,
  procedureVisibility: Record<string, boolean>,
): boolean {
  return procedureVisibility[branchId] ?? packageBranchDefaultVisible(pkg, branchId);
}

function addBranchEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  branchIndex: number,
  pkg: ProcedurePackage | null,
  visible: boolean,
  annotationVisible: boolean,
  widthMeasurementVisible: boolean,
  displayLevel: ProcedureDisplayLevel,
): string[] {
  const ids: string[] = [];
  if (pkg && branchIndex === 0) {
    ids.push(
      ...addFixLabelEntities(
        viewer,
        bundle,
        branchBundle,
        pkg,
        visible,
        annotationVisible,
        displayLevel,
      ),
    );
  }
  branchBundle.segmentBundles.forEach((segmentBundle) => {
    ids.push(
      ...addSegmentEntities(
        viewer,
        bundle,
        branchBundle,
        segmentBundle,
        pkg,
        visible,
        annotationVisible,
        widthMeasurementVisible,
        displayLevel,
      ),
    );
  });
  ids.push(
    ...addBranchVerticalProfileEntity(
      viewer,
      bundle,
      branchBundle,
      pkg,
      visible,
      annotationVisible,
      displayLevel,
    ),
    ...addBranchTurnJunctionEntities(
      viewer,
      bundle,
      branchBundle,
      visible,
      annotationVisible,
      displayLevel,
    ),
    ...addBranchProtectionSurfaceEntities(
      viewer,
      bundle,
      branchBundle,
      visible,
      annotationVisible,
      widthMeasurementVisible,
      displayLevel,
    ),
    ...((branchBundle.protectionSurfaces ?? []).length === 0
      ? addBranchMissedConnectorSurfaceEntities(
          viewer,
          bundle,
          branchBundle,
          visible,
          annotationVisible,
          displayLevel,
        )
      : []),
    ...addBranchMissedCaMahfConnectorEntities(
      viewer,
      bundle,
      branchBundle,
      visible,
      annotationVisible,
      displayLevel,
    ),
  );
  return ids;
}

export function useProcedureSegmentLayer({ enabled = true }: { enabled?: boolean } = {}): void {
  const {
    viewer,
    layers,
    procedureVisibility,
    activeAirportCode,
    procedureAnnotationEnabled,
    procedureWidthMeasurementEnabled,
    procedureDisplayLevel,
  } = useApp();
  const visibleRef = useRef(layers.procedures);
  const annotationVisibleRef = useRef(procedureAnnotationEnabled);
  const widthMeasurementVisibleRef = useRef(procedureWidthMeasurementEnabled);
  const displayLevelRef = useRef(procedureDisplayLevel);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const branchEntityIdsRef = useRef<Record<string, string[]>>({});
  const allEntityIdsRef = useRef<string[]>([]);
  const renderDataRef = useRef<{
    renderBundles: ProcedureRenderBundle[];
    packageById: Map<string, ProcedurePackage>;
  } | null>(null);

  const addBranchEntityIds = (branchId: string, entityIds: string[]) => {
    allEntityIdsRef.current.push(...entityIds);
    const existing = branchEntityIdsRef.current[branchId] ?? [];
    branchEntityIdsRef.current[branchId] = [...existing, ...entityIds];
  };

  useEffect(() => {
    visibleRef.current = layers.procedures;
    annotationVisibleRef.current = procedureAnnotationEnabled;
    widthMeasurementVisibleRef.current = procedureWidthMeasurementEnabled;
    displayLevelRef.current = procedureDisplayLevel;
    procedureVisibilityRef.current = procedureVisibility;

    if (!enabled || !isCesiumViewerUsable(viewer)) return;
    Object.entries(branchEntityIdsRef.current).forEach(([branchId, entityIds]) => {
      const branchVisible = procedureVisibility[branchId] ?? true;
      entityIds.forEach((entityId) => {
        const entity = viewer.entities.getById(entityId);
        if (entity) {
          const annotation = getProcedureAnnotation(entity);
          const baseVisible = layers.procedures && branchVisible;
          if (isMeasurementEntityId(entityId)) {
            entity.show =
              procedureWidthMeasurementEnabled &&
              procedureEntityShow(baseVisible, annotation, procedureDisplayLevel);
            return;
          }
          entity.show = procedureEntityShow(
            baseVisible,
            annotation,
            procedureDisplayLevel,
            isAnnotationLabelId(entityId),
            procedureAnnotationEnabled,
          );
        }
      });
    });

    if (layers.procedures && renderDataRef.current) {
      renderDataRef.current.renderBundles.forEach((bundle) => {
        const pkg = renderDataRef.current?.packageById.get(bundle.packageId) ?? null;
        bundle.branchBundles.forEach((branchBundle, branchIndex) => {
          const branchVisible = packageBranchVisible(pkg, branchBundle.branchId, procedureVisibility);
          if (!branchVisible || branchEntityIdsRef.current[branchBundle.branchId]?.length) return;
          addBranchEntityIds(
            branchBundle.branchId,
            addBranchEntities(
              viewer,
              bundle,
              branchBundle,
              branchIndex,
              pkg,
              true,
              procedureAnnotationEnabled,
              procedureWidthMeasurementEnabled,
              procedureDisplayLevel,
            ),
          );
        });
      });
    }
  }, [
    enabled,
    viewer,
    layers.procedures,
    procedureVisibility,
    procedureAnnotationEnabled,
    procedureWidthMeasurementEnabled,
    procedureDisplayLevel,
  ]);

  useEffect(() => {
    if (!enabled) return;
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    branchEntityIdsRef.current = {};
    allEntityIdsRef.current = [];
    renderDataRef.current = null;

    loadProcedureRenderBundleData(activeAirportCode)
      .then(({ renderBundles, packages }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        const packageById = new Map(packages.map((pkg) => [pkg.packageId, pkg]));
        renderDataRef.current = { renderBundles, packageById };
        renderBundles.forEach((bundle) => {
          const pkg = packageById.get(bundle.packageId);
          bundle.branchBundles.forEach((branchBundle, branchIndex) => {
            const branchVisible = packageBranchVisible(
              pkg ?? null,
              branchBundle.branchId,
              procedureVisibilityRef.current,
            );
            const visible = visibleRef.current && branchVisible;
            if (!visible) return;
            addBranchEntityIds(
              branchBundle.branchId,
              addBranchEntities(
                viewer,
                bundle,
                branchBundle,
                branchIndex,
                pkg ?? null,
                visible,
                annotationVisibleRef.current,
                widthMeasurementVisibleRef.current,
                displayLevelRef.current,
              ),
            );
          });
        });
      })
      .catch((error) => {
        if (isMissingJsonAsset(error)) {
          console.warn(
            `[useProcedureSegmentLayer] procedure-details data for ${activeAirportCode} not found. ` +
              "Run: python aeroviz-4d/python/preprocess_procedures.py --airport <ICAO>",
          );
        } else {
          console.error("[useProcedureSegmentLayer]", error);
        }
      });

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        allEntityIdsRef.current.forEach((id) => viewer.entities.removeById(id));
      }
      branchEntityIdsRef.current = {};
      allEntityIdsRef.current = [];
      renderDataRef.current = null;
    };
  }, [enabled, viewer, activeAirportCode]);
}
