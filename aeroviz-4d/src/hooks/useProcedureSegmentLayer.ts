import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  loadProcedureRenderBundleData,
  type BranchGeometryBundle,
  type ProcedureRenderBundle,
  type ProcedureSegmentRenderBundle,
} from "../data/procedureRenderBundle";
import type {
  ProcedurePackage,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../data/procedurePackage";
import {
  attachProcedureAnnotation,
  procedureAnnotationMeaning,
  type ProcedureAnnotationKind,
  type ProcedureAnnotationStatus,
  type ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import type { GeoPoint } from "../utils/procedureGeoMath";
import type { VariableWidthRibbonGeometry } from "../utils/procedureSurfaceGeometry";

const PROCEDURE_SEGMENT_ENTITY_PREFIX = "procedure-segment-";
const CENTERLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.95);
const PRIMARY_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.18);
const SECONDARY_COLOR = Cesium.Color.YELLOW.withAlpha(0.1);
const LNAV_VNAV_OCS_COLOR = Cesium.Color.LIME.withAlpha(0.2);
const PRECISION_FINAL_SURFACE_COLOR = Cesium.Color.MAGENTA.withAlpha(0.18);
const TURN_FILL_COLOR = Cesium.Color.ORANGE.withAlpha(0.22);
const CONNECTOR_COLOR = Cesium.Color.ORANGE.withAlpha(0.32);
const CONNECTOR_LINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.92);
const MISSED_SURFACE_COLOR = Cesium.Color.YELLOW.withAlpha(0.24);
const MISSED_CA_ESTIMATED_SURFACE_COLOR = Cesium.Color.ORANGE.withAlpha(0.26);
const CA_COURSE_GUIDE_COLOR = Cesium.Color.ORANGE.withAlpha(0.98);
const CA_CENTERLINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.86);
const TURNING_MISSED_DEBUG_COLOR = Cesium.Color.YELLOW.withAlpha(0.98);
const TURNING_MISSED_PRIMITIVE_COLOR = Cesium.Color.ORANGE.withAlpha(0.78);
const FINAL_SURFACE_STATUS_COLOR = Cesium.Color.ORANGE.withAlpha(0.9);
const CA_ENDPOINT_COLOR = Cesium.Color.ORANGE.withAlpha(0.98);
const OUTLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.28);
const ENVELOPE_HEIGHT_OFFSET_M = 8;
const OEA_HEIGHT_OFFSET_M = 18;
const LNAV_VNAV_OCS_HEIGHT_OFFSET_M = 28;
const PRECISION_FINAL_SURFACE_HEIGHT_OFFSET_M = 34;
const CONNECTOR_HEIGHT_OFFSET_M = 45;
const MISSED_SURFACE_HEIGHT_OFFSET_M = 58;
const CA_COURSE_GUIDE_HEIGHT_OFFSET_M = 82;
const CA_CENTERLINE_HEIGHT_OFFSET_M = 88;
const TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M = 96;
const FINAL_SURFACE_STATUS_HEIGHT_OFFSET_M = 110;
const CA_ENDPOINT_HEIGHT_OFFSET_M = 92;
const PROCEDURE_ANNOTATION_LABEL_PREFIX = "procedure-annotation-label-";

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

function isAnnotationLabelId(entityId: string): boolean {
  return entityId.startsWith(PROCEDURE_ANNOTATION_LABEL_PREFIX);
}

function sourceRefsFromSegment(segment: ProcedureSegment): string[] {
  return (segment.sourceRefs ?? []).map((ref) => ref.sourceId);
}

function sourceRefsFromLegs(legs: ProcedurePackageLeg[]): string[] {
  return [...new Set(legs.flatMap((leg) => (leg.sourceRefs ?? []).map((ref) => ref.sourceId)))];
}

function param(label: string, value: unknown): { label: string; value: string } | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return { label, value: Number.isInteger(value) ? String(value) : value.toFixed(2) };
  return { label, value: String(value) };
}

function compactSegmentType(segmentType: string | undefined): string {
  return (segmentType ?? "UNKNOWN").replace(/_/g, " ");
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
        ? (args.leg.sourceRefs ?? []).map((ref) => ref.sourceId)
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
      sourceRefs: fix.sourceRefs.map((ref) => ref.sourceId),
    });
    const labelId = addAnnotationLabel(
      viewer,
      annotation,
      { lonDeg: fix.lonDeg, latDeg: fix.latDeg, altM: (fix.altFtMsl ?? 0) * 0.3048 },
      visible,
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
  );
  addPolyline(
    viewer,
    rightId,
    `${name} right boundary`,
    ribbon.rightGeoBoundary.map((point) => elevatedPoint(point, altitudeOffsetM)),
    visible,
    3,
    material,
  );
  return [leftId, rightId];
}

function addSegmentEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  segmentBundle: ProcedureSegmentRenderBundle,
  visible: boolean,
  annotationVisible: boolean,
): string[] {
  const ids: string[] = [];
  const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${segmentBundle.segment.segmentId}`;
  const segmentName = `${bundle.procedureName} ${segmentBundle.segment.segmentType}`;
  const segmentDiagnostics = segmentBundle.diagnostics.map((diagnostic) => diagnostic.message);
  const centerlineEstimated = segmentBundle.segmentGeometry.diagnostics.some(
    (diagnostic) => diagnostic.code === "ESTIMATED_CA_GEOMETRY",
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
    visible,
    4,
    CENTERLINE_COLOR,
    centerlineAnnotation,
  );
  ids.push(centerlineId);
  const centerlineLabelId = addAnnotationLabel(
    viewer,
    centerlineAnnotation,
    representativePoint(segmentBundle.segmentGeometry.centerline.geoPositions),
    visible && annotationVisible,
  );
  if (centerlineLabelId) ids.push(centerlineLabelId);

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
    parameters: segmentParams(segmentBundle.segment, segmentBundle.legs),
    diagnostics: segmentDiagnostics,
    sourceRefs: sourceRefsFromSegment(segmentBundle.segment),
  });
  addRibbonPolygon(
    viewer,
    primaryId,
    `${segmentName} primary envelope`,
    segmentBundle.segmentGeometry.primaryEnvelope,
    visible,
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
    parameters: segmentParams(segmentBundle.segment, segmentBundle.legs),
    diagnostics: segmentDiagnostics,
    sourceRefs: sourceRefsFromSegment(segmentBundle.segment),
  });
  addRibbonPolygon(
    viewer,
    secondaryId,
    `${segmentName} secondary envelope`,
    segmentBundle.segmentGeometry.secondaryEnvelope,
    visible,
    SECONDARY_COLOR,
    ENVELOPE_HEIGHT_OFFSET_M,
    secondaryAnnotation,
  );
  ids.push(secondaryId);

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
      visible,
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
        visible,
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
        turnSecondaryAnnotation,
      );
      ids.push(turnSecondaryId);
    }
  });

  if (segmentBundle.finalOea) {
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
      parameters: segmentParams(segmentBundle.segment, segmentBundle.legs),
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      oeaPrimaryId,
      `${segmentName} LNAV OEA primary`,
      segmentBundle.finalOea.primary,
      visible,
      PRIMARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
      oeaAnnotation,
    );
    ids.push(oeaPrimaryId);
    const oeaLabelId = addAnnotationLabel(
      viewer,
      oeaAnnotation,
      representativePoint(segmentBundle.finalOea.primary.leftGeoBoundary),
      visible && annotationVisible,
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
      visible,
      SECONDARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
      oeaSecondaryAnnotation,
    );
    ids.push(oeaSecondaryId);
  }

  if (segmentBundle.lnavVnavOcs) {
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
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      ocsPrimaryId,
      `${segmentName} LNAV/VNAV OCS primary`,
      segmentBundle.lnavVnavOcs.primary,
      visible,
      LNAV_VNAV_OCS_COLOR,
      LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
      ocsAnnotation,
    );
    ids.push(ocsPrimaryId);
    const ocsLabelId = addAnnotationLabel(
      viewer,
      ocsAnnotation,
      representativePoint(segmentBundle.lnavVnavOcs.centerline.geoPositions),
      visible && annotationVisible,
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
      visible,
      LNAV_VNAV_OCS_COLOR,
      LNAV_VNAV_OCS_HEIGHT_OFFSET_M,
      ocsSecondaryAnnotation,
    );
    ids.push(ocsSecondaryId);
  }

  segmentBundle.precisionFinalSurfaces.forEach((surface) => {
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
      ],
      diagnostics: [...segmentDiagnostics, ...surface.notes],
    });
    addRibbonPolygon(
      viewer,
      surfaceId,
      `${segmentName} ${surface.surfaceType} debug estimate`,
      surface.ribbon,
      visible,
      PRECISION_FINAL_SURFACE_COLOR,
      PRECISION_FINAL_SURFACE_HEIGHT_OFFSET_M,
      surfaceAnnotation,
    );
    ids.push(surfaceId);
    const surfaceLabelId = addAnnotationLabel(
      viewer,
      surfaceAnnotation,
      representativePoint(surface.centerline.geoPositions),
      visible && annotationVisible,
    );
    if (surfaceLabelId) ids.push(surfaceLabelId);
  });

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
        visible,
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
        visible && annotationVisible,
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
      visible,
      CONNECTOR_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
      connectorAnnotation,
    );
    ids.push(connectorPrimaryId);
    const connectorLabelId = addAnnotationLabel(
      viewer,
      connectorAnnotation,
      representativePoint(segmentBundle.alignedConnector.primary.leftGeoBoundary),
      visible && annotationVisible,
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
      visible,
      CONNECTOR_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
      connectorSecondaryAnnotation,
    );
    ids.push(connectorSecondaryId);

    ids.push(
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-connector-primary-boundary`,
        `${segmentName} aligned connector primary`,
        segmentBundle.alignedConnector.primary,
        visible,
        CONNECTOR_LINE_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
      ),
      ...addRibbonBoundaryPolylines(
        viewer,
        `${baseId}-connector-secondary-boundary`,
        `${segmentName} aligned connector secondary`,
        segmentBundle.alignedConnector.secondaryOuter,
        visible,
        CONNECTOR_LINE_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
      ),
    );
  }

  if (segmentBundle.missedSectionSurface) {
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
      ],
      diagnostics: segmentDiagnostics,
    });
    addRibbonPolygon(
      viewer,
      missedPrimaryId,
      `${missedSurfaceName} primary`,
      segmentBundle.missedSectionSurface.primary,
      visible,
      missedSurfaceColor,
      MISSED_SURFACE_HEIGHT_OFFSET_M,
      missedAnnotation,
    );
    ids.push(missedPrimaryId);
    const missedLabelId = addAnnotationLabel(
      viewer,
      missedAnnotation,
      representativePoint(segmentBundle.missedSectionSurface.primary.leftGeoBoundary),
      visible && annotationVisible,
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
      visible,
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
      visible,
      5,
      CA_COURSE_GUIDE_COLOR,
      guideAnnotation,
    );
    ids.push(guideId);
    const guideLabelId = addAnnotationLabel(
      viewer,
      guideAnnotation,
      representativePoint(guide.geoPositions.map((point) => elevatedPoint(point, CA_COURSE_GUIDE_HEIGHT_OFFSET_M))),
      visible && annotationVisible,
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
      visible,
      4,
      CA_CENTERLINE_COLOR,
      caCenterlineAnnotation,
    );
    ids.push(centerlineId);
    const centerlineLabelId = addAnnotationLabel(
      viewer,
      caCenterlineAnnotation,
      representativePoint(centerline.geoPositions.map((point) => elevatedPoint(point, CA_CENTERLINE_HEIGHT_OFFSET_M))),
      visible && annotationVisible,
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
      visible,
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
      visible && annotationVisible,
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
      visible,
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
      visible && annotationVisible,
    );
    if (debugLabelId) ids.push(debugLabelId);
  }

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
      visible,
      primitive.debugType === "NOMINAL_TURN_PATH" ? 4 : 3,
      TURNING_MISSED_PRIMITIVE_COLOR,
      primitiveAnnotation,
    );
    ids.push(primitiveId);
    const primitiveLabelId = addAnnotationLabel(
      viewer,
      primitiveAnnotation,
      representativePoint(primitive.geoPositions.map((point) => elevatedPoint(point, TURNING_MISSED_DEBUG_HEIGHT_OFFSET_M))),
      visible && annotationVisible,
    );
    if (primitiveLabelId) ids.push(primitiveLabelId);
  });

  return ids;
}

function addBranchTurnJunctionEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
  annotationVisible: boolean,
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
      visible,
      TURN_FILL_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
      primaryAnnotation,
    );
    ids.push(primaryId);
    const labelId = addAnnotationLabel(
      viewer,
      primaryAnnotation,
      representativePoint(junction.primaryPatch.ribbon.leftGeoBoundary),
      visible && annotationVisible,
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
        visible,
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
        secondaryAnnotation,
      );
      ids.push(secondaryId);
    }
  });

  return ids;
}

export function useProcedureSegmentLayer({ enabled = true }: { enabled?: boolean } = {}): void {
  const { viewer, layers, procedureVisibility, activeAirportCode, procedureAnnotationEnabled } = useApp();
  const visibleRef = useRef(layers.procedures);
  const annotationVisibleRef = useRef(procedureAnnotationEnabled);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const branchEntityIdsRef = useRef<Record<string, string[]>>({});

  useEffect(() => {
    visibleRef.current = layers.procedures;
    annotationVisibleRef.current = procedureAnnotationEnabled;
    procedureVisibilityRef.current = procedureVisibility;

    if (!enabled || !isCesiumViewerUsable(viewer)) return;
    Object.entries(branchEntityIdsRef.current).forEach(([branchId, entityIds]) => {
      const branchVisible = procedureVisibility[branchId] ?? true;
      entityIds.forEach((entityId) => {
        const entity = viewer.entities.getById(entityId);
        if (entity) {
          entity.show =
            layers.procedures &&
            branchVisible &&
            (!isAnnotationLabelId(entityId) || procedureAnnotationEnabled);
        }
      });
    });
  }, [enabled, viewer, layers.procedures, procedureVisibility, procedureAnnotationEnabled]);

  useEffect(() => {
    if (!enabled) return;
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    const addedIds: string[] = [];
    branchEntityIdsRef.current = {};

    const addBranchEntityIds = (branchId: string, entityIds: string[]) => {
      addedIds.push(...entityIds);
      const existing = branchEntityIdsRef.current[branchId] ?? [];
      branchEntityIdsRef.current[branchId] = [...existing, ...entityIds];
    };

    loadProcedureRenderBundleData(activeAirportCode)
      .then(({ renderBundles, packages }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        const packageById = new Map(packages.map((pkg) => [pkg.packageId, pkg]));
        renderBundles.forEach((bundle) => {
          const pkg = packageById.get(bundle.packageId);
          bundle.branchBundles.forEach((branchBundle, branchIndex) => {
            const visible = visibleRef.current && (procedureVisibilityRef.current[branchBundle.branchId] ?? true);
            if (pkg && branchIndex === 0) {
              addBranchEntityIds(
                branchBundle.branchId,
                addFixLabelEntities(
                  viewer,
                  bundle,
                  branchBundle,
                  pkg,
                  visible && annotationVisibleRef.current,
                ),
              );
            }
            branchBundle.segmentBundles.forEach((segmentBundle) => {
              addBranchEntityIds(
                branchBundle.branchId,
                addSegmentEntities(
                  viewer,
                  bundle,
                  branchBundle,
                  segmentBundle,
                  visible,
                  annotationVisibleRef.current,
                ),
              );
            });
            addBranchEntityIds(
              branchBundle.branchId,
              addBranchTurnJunctionEntities(
                viewer,
                bundle,
                branchBundle,
                visible,
                annotationVisibleRef.current,
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
        addedIds.forEach((id) => viewer.entities.removeById(id));
      }
      branchEntityIdsRef.current = {};
    };
  }, [enabled, viewer, activeAirportCode]);
}
