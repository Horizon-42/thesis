import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  loadProcedureRenderBundleData,
  type BranchGeometryBundle,
  type ProcedureRenderBundle,
  type ProcedureSegmentRenderBundle,
} from "../data/procedureRenderBundle";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import type { GeoPoint } from "../utils/procedureGeoMath";
import type { VariableWidthRibbonGeometry } from "../utils/procedureSurfaceGeometry";

const PROCEDURE_SEGMENT_ENTITY_PREFIX = "procedure-segment-";
const CENTERLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.95);
const PRIMARY_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.18);
const SECONDARY_COLOR = Cesium.Color.YELLOW.withAlpha(0.1);
const TURN_FILL_COLOR = Cesium.Color.ORANGE.withAlpha(0.22);
const CONNECTOR_COLOR = Cesium.Color.ORANGE.withAlpha(0.32);
const CONNECTOR_LINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.92);
const OUTLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.28);
const ENVELOPE_HEIGHT_OFFSET_M = 8;
const OEA_HEIGHT_OFFSET_M = 18;
const CONNECTOR_HEIGHT_OFFSET_M = 45;

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
): void {
  if (points.length < 2) return;
  viewer.entities.add({
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
}

function addRibbonPolygon(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  ribbon: VariableWidthRibbonGeometry | undefined,
  visible: boolean,
  material: Cesium.Color,
  altitudeOffsetM = 0,
): void {
  if (!ribbon || ribbon.leftGeoBoundary.length < 2 || ribbon.rightGeoBoundary.length < 2) return;
  const polygonPoints = [...ribbon.leftGeoBoundary, ...ribbon.rightGeoBoundary.slice().reverse()];
  viewer.entities.add({
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
  segmentBundle: ProcedureSegmentRenderBundle,
  visible: boolean,
): string[] {
  const ids: string[] = [];
  const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${segmentBundle.segment.segmentId}`;
  const segmentName = `${bundle.procedureName} ${segmentBundle.segment.segmentType}`;

  const centerlineId = `${baseId}-centerline`;
  addPolyline(
    viewer,
    centerlineId,
    `${segmentName} centerline`,
    segmentBundle.segmentGeometry.centerline.geoPositions,
    visible,
    4,
    CENTERLINE_COLOR,
  );
  ids.push(centerlineId);

  const primaryId = `${baseId}-envelope-primary`;
  addRibbonPolygon(
    viewer,
    primaryId,
    `${segmentName} primary envelope`,
    segmentBundle.segmentGeometry.primaryEnvelope,
    visible,
    PRIMARY_COLOR,
    ENVELOPE_HEIGHT_OFFSET_M,
  );
  ids.push(primaryId);

  const secondaryId = `${baseId}-envelope-secondary`;
  addRibbonPolygon(
    viewer,
    secondaryId,
    `${segmentName} secondary envelope`,
    segmentBundle.segmentGeometry.secondaryEnvelope,
    visible,
    SECONDARY_COLOR,
    ENVELOPE_HEIGHT_OFFSET_M,
  );
  ids.push(secondaryId);

  segmentBundle.segmentGeometry.turnJunctions.forEach((junction) => {
    const turnPrimaryId = `${baseId}-turn-${junction.turnPointIndex}-primary`;
    addRibbonPolygon(
      viewer,
      turnPrimaryId,
      `${segmentName} visual turn fill primary`,
      junction.primaryPatch.ribbon,
      visible,
      TURN_FILL_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
    );
    ids.push(turnPrimaryId);

    if (junction.secondaryPatch) {
      const turnSecondaryId = `${baseId}-turn-${junction.turnPointIndex}-secondary`;
      addRibbonPolygon(
        viewer,
        turnSecondaryId,
        `${segmentName} visual turn fill secondary`,
        junction.secondaryPatch.ribbon,
        visible,
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
      );
      ids.push(turnSecondaryId);
    }
  });

  if (segmentBundle.finalOea) {
    const oeaPrimaryId = `${baseId}-oea-primary`;
    addRibbonPolygon(
      viewer,
      oeaPrimaryId,
      `${segmentName} LNAV OEA primary`,
      segmentBundle.finalOea.primary,
      visible,
      PRIMARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
    );
    ids.push(oeaPrimaryId);

    const oeaSecondaryId = `${baseId}-oea-secondary`;
    addRibbonPolygon(
      viewer,
      oeaSecondaryId,
      `${segmentName} LNAV OEA secondary`,
      segmentBundle.finalOea.secondaryOuter,
      visible,
      SECONDARY_COLOR,
      OEA_HEIGHT_OFFSET_M,
    );
    ids.push(oeaSecondaryId);
  }

  if (segmentBundle.alignedConnector) {
    const connectorPrimaryId = `${baseId}-connector-primary`;
    addRibbonPolygon(
      viewer,
      connectorPrimaryId,
      `${segmentName} aligned connector primary`,
      segmentBundle.alignedConnector.primary,
      visible,
      CONNECTOR_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
    );
    ids.push(connectorPrimaryId);

    const connectorSecondaryId = `${baseId}-connector-secondary`;
    addRibbonPolygon(
      viewer,
      connectorSecondaryId,
      `${segmentName} aligned connector secondary`,
      segmentBundle.alignedConnector.secondaryOuter,
      visible,
      CONNECTOR_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
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

  return ids;
}

function addBranchTurnJunctionEntities(
  viewer: Cesium.Viewer,
  bundle: ProcedureRenderBundle,
  branchBundle: BranchGeometryBundle,
  visible: boolean,
): string[] {
  const ids: string[] = [];

  branchBundle.turnJunctions.forEach((junction) => {
    const baseId = `${PROCEDURE_SEGMENT_ENTITY_PREFIX}${bundle.packageId}-${junction.geometryId}`;
    const junctionName = `${bundle.procedureName} visual inter-segment turn fill`;

    const primaryId = `${baseId}-primary`;
    addRibbonPolygon(
      viewer,
      primaryId,
      `${junctionName} primary`,
      junction.primaryPatch.ribbon,
      visible,
      TURN_FILL_COLOR,
      CONNECTOR_HEIGHT_OFFSET_M,
    );
    ids.push(primaryId);

    if (junction.secondaryPatch) {
      const secondaryId = `${baseId}-secondary`;
      addRibbonPolygon(
        viewer,
        secondaryId,
        `${junctionName} secondary`,
        junction.secondaryPatch.ribbon,
        visible,
        TURN_FILL_COLOR,
        CONNECTOR_HEIGHT_OFFSET_M,
      );
      ids.push(secondaryId);
    }
  });

  return ids;
}

export function useProcedureSegmentLayer({ enabled = true }: { enabled?: boolean } = {}): void {
  const { viewer, layers, procedureVisibility, activeAirportCode } = useApp();
  const visibleRef = useRef(layers.procedures);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const branchEntityIdsRef = useRef<Record<string, string[]>>({});

  useEffect(() => {
    visibleRef.current = layers.procedures;
    procedureVisibilityRef.current = procedureVisibility;

    if (!enabled || !isCesiumViewerUsable(viewer)) return;
    Object.entries(branchEntityIdsRef.current).forEach(([branchId, entityIds]) => {
      const branchVisible = procedureVisibility[branchId] ?? true;
      entityIds.forEach((entityId) => {
        const entity = viewer.entities.getById(entityId);
        if (entity) entity.show = layers.procedures && branchVisible;
      });
    });
  }, [enabled, viewer, layers.procedures, procedureVisibility]);

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
      .then(({ renderBundles }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        renderBundles.forEach((bundle) => {
          bundle.branchBundles.forEach((branchBundle) => {
            const visible = visibleRef.current && (procedureVisibilityRef.current[branchBundle.branchId] ?? true);
            branchBundle.segmentBundles.forEach((segmentBundle) => {
              addBranchEntityIds(
                branchBundle.branchId,
                addSegmentEntities(viewer, bundle, segmentBundle, visible),
              );
            });
            addBranchEntityIds(
              branchBundle.branchId,
              addBranchTurnJunctionEntities(viewer, bundle, branchBundle, visible),
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
