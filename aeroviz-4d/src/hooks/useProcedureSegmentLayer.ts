import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  loadProcedureRenderBundleData,
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
const CONNECTOR_COLOR = Cesium.Color.ORANGE.withAlpha(0.14);
const OUTLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.28);

function geoToCartesian(point: GeoPoint): Cesium.Cartesian3 {
  return Cesium.Cartesian3.fromDegrees(point.lonDeg, point.latDeg, point.altM);
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
): void {
  if (!ribbon || ribbon.leftGeoBoundary.length < 2 || ribbon.rightGeoBoundary.length < 2) return;
  const polygonPoints = [...ribbon.leftGeoBoundary, ...ribbon.rightGeoBoundary.slice().reverse()];
  viewer.entities.add({
    id,
    name,
    show: visible,
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(polygonPoints.map(geoToCartesian)),
      material,
      perPositionHeight: true,
      outline: true,
      outlineColor: OUTLINE_COLOR,
    },
  });
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
  );
  ids.push(secondaryId);

  if (segmentBundle.finalOea) {
    const oeaPrimaryId = `${baseId}-oea-primary`;
    addRibbonPolygon(
      viewer,
      oeaPrimaryId,
      `${segmentName} LNAV OEA primary`,
      segmentBundle.finalOea.primary,
      visible,
      PRIMARY_COLOR,
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
    );
    ids.push(connectorSecondaryId);
  }

  return ids;
}

export function useProcedureSegmentLayer(): void {
  const { viewer, layers, procedureVisibility, activeAirportCode } = useApp();
  const visibleRef = useRef(layers.procedures);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const branchEntityIdsRef = useRef<Record<string, string[]>>({});

  useEffect(() => {
    visibleRef.current = layers.procedures;
    procedureVisibilityRef.current = procedureVisibility;

    if (!isCesiumViewerUsable(viewer)) return;
    Object.entries(branchEntityIdsRef.current).forEach(([branchId, entityIds]) => {
      const branchVisible = procedureVisibility[branchId] ?? true;
      entityIds.forEach((entityId) => {
        const entity = viewer.entities.getById(entityId);
        if (entity) entity.show = layers.procedures && branchVisible;
      });
    });
  }, [viewer, layers.procedures, procedureVisibility]);

  useEffect(() => {
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
  }, [viewer, activeAirportCode]);
}
