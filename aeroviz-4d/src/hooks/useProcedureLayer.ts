/**
 * useProcedureLayer.ts
 * --------------------
 * Loads public/data/airports/<ICAO>/procedure-details/*.json and renders procedure routes,
 * fixes, and an approximate 3D RNAV tunnel in Cesium.
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { loadProcedureRouteData, type ProcedureRouteViewModel } from "../data/procedureRoutes";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  DEFAULT_NOMINAL_SPEED_KT,
  DEFAULT_TUNNEL_HALF_HEIGHT_M,
  DEFAULT_TUNNEL_HALF_WIDTH_M,
  DEFAULT_TUNNEL_SAMPLE_SPACING_M,
  buildTunnelSections,
  type ProcedurePoint3D,
  type TunnelSection,
} from "../utils/procedureGeometry";

const PROCEDURE_ENTITY_PREFIX = "procedure-";
const ROUTE_COLOR = Cesium.Color.CYAN.withAlpha(0.95);
const FIX_COLOR = Cesium.Color.YELLOW.withAlpha(0.95);
const TUNNEL_COLOR = Cesium.Color.DEEPSKYBLUE.withAlpha(0.16);
const TUNNEL_OUTLINE_COLOR = Cesium.Color.CYAN.withAlpha(0.25);

function toCartesian(point: ProcedurePoint3D): Cesium.Cartesian3 {
  return Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.altM);
}

function toProcedurePoint(point: ProcedureRouteViewModel["points"][number]): ProcedurePoint3D {
  return {
    lon: point.lon,
    lat: point.lat,
    altM: point.altM,
  };
}

function addTunnelQuad(
  viewer: Cesium.Viewer,
  id: string,
  points: ProcedurePoint3D[],
  visible: boolean,
): void {
  viewer.entities.add({
    id,
    name: id,
    show: visible,
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(points.map(toCartesian)),
      material: TUNNEL_COLOR,
      perPositionHeight: true,
      outline: true,
      outlineColor: TUNNEL_OUTLINE_COLOR,
    },
  });
}

function addTunnelSegment(
  viewer: Cesium.Viewer,
  baseId: string,
  previous: TunnelSection,
  next: TunnelSection,
  visible: boolean,
): string[] {
  const quads: Array<{ suffix: string; points: ProcedurePoint3D[] }> = [
    {
      suffix: "left",
      points: [previous.leftBottom, next.leftBottom, next.leftTop, previous.leftTop],
    },
    {
      suffix: "right",
      points: [previous.rightBottom, previous.rightTop, next.rightTop, next.rightBottom],
    },
    {
      suffix: "top",
      points: [previous.leftTop, next.leftTop, next.rightTop, previous.rightTop],
    },
    {
      suffix: "bottom",
      points: [previous.leftBottom, previous.rightBottom, next.rightBottom, next.leftBottom],
    },
  ];

  return quads.map((quad) => {
    const id = `${baseId}-${quad.suffix}`;
    addTunnelQuad(viewer, id, quad.points, visible);
    return id;
  });
}

export function useProcedureLayer(): void {
  const { viewer, layers, procedureVisibility, activeAirportCode } = useApp();
  const visibleRef = useRef(layers.procedures);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const routeEntityIdsRef = useRef<Record<string, string[]>>({});
  const routeDefaultsRef = useRef<Record<string, boolean>>({});

  useEffect(() => {
    visibleRef.current = layers.procedures;
    procedureVisibilityRef.current = procedureVisibility;

    if (!isCesiumViewerUsable(viewer)) return;
    Object.entries(routeEntityIdsRef.current).forEach(([routeId, entityIds]) => {
      const routeVisible =
        procedureVisibility[routeId] ?? routeDefaultsRef.current[routeId] ?? true;
      entityIds.forEach((entityId) => {
        const entity = viewer.entities.getById(entityId);
        if (entity) entity.show = layers.procedures && routeVisible;
      });
    });
  }, [viewer, layers.procedures, procedureVisibility]);

  useEffect(() => {
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    const addedIds: string[] = [];
    routeEntityIdsRef.current = {};
    routeDefaultsRef.current = {};

    const addRouteEntityId = (routeId: string, entityId: string) => {
      addedIds.push(entityId);
      const existing = routeEntityIdsRef.current[routeId] ?? [];
      routeEntityIdsRef.current[routeId] = [...existing, entityId];
    };

    const isRouteVisible = (routeId: string) => {
      const routeVisible =
        procedureVisibilityRef.current[routeId] ?? routeDefaultsRef.current[routeId] ?? true;
      return visibleRef.current && routeVisible;
    };

    loadProcedureRouteData(activeAirportCode)
      .then(({ routes }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        routes.forEach((route, routeIndex) => {
          const routeId = route.routeId ?? `route-${routeIndex}`;
          const baseId = `${PROCEDURE_ENTITY_PREFIX}${routeId}`;
          if (route.points.length < 2) {
            console.warn(`[useProcedureLayer] Skipping ${routeId}: fewer than two points`);
            return;
          }

          routeDefaultsRef.current[routeId] = route.defaultVisible;
          const visible = isRouteVisible(routeId);
          const lineId = `${baseId}-line`;
          viewer.entities.add({
            id: lineId,
            name: route.procedureName,
            show: visible,
            polyline: {
              positions: Cesium.Cartesian3.fromDegreesArrayHeights(
                route.points.flatMap((point) => [point.lon, point.lat, point.altM]),
              ),
              width: 5,
              material: ROUTE_COLOR,
            },
          });
          addRouteEntityId(routeId, lineId);

          const tunnel = route.tunnel;
          const routePoints = route.points.map(toProcedurePoint);
          const tunnelSections = buildTunnelSections(routePoints, {
            halfWidthM: tunnel?.lateralHalfWidthNm
              ? tunnel.lateralHalfWidthNm * 1852
              : DEFAULT_TUNNEL_HALF_WIDTH_M,
            halfHeightM: tunnel?.verticalHalfHeightFt
              ? tunnel.verticalHalfHeightFt * 0.3048
              : DEFAULT_TUNNEL_HALF_HEIGHT_M,
            sampleSpacingM: tunnel?.sampleSpacingM ?? DEFAULT_TUNNEL_SAMPLE_SPACING_M,
            nominalSpeedKt: route.nominalSpeedKt ?? DEFAULT_NOMINAL_SPEED_KT,
          });

          for (let index = 0; index < tunnelSections.length - 1; index++) {
            const segmentIds = addTunnelSegment(
              viewer,
              `${baseId}-tunnel-${index}`,
              tunnelSections[index],
              tunnelSections[index + 1],
              visible,
            );
              segmentIds.forEach((entityId) => addRouteEntityId(routeId, entityId));
          }

          route.points.forEach((point, index) => {
            const id = `${PROCEDURE_ENTITY_PREFIX}${route.routeId}-fix-${point.sequence}-${index}`;
            viewer.entities.add({
              id,
              name: point.fixIdent,
              show: isRouteVisible(route.routeId),
              position: Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.altM),
              point: {
                pixelSize: point.role === "MAPt" ? 13 : 11,
                color: point.role === "FAF" ? Cesium.Color.ORANGE.withAlpha(0.95) : FIX_COLOR,
                outlineColor: Cesium.Color.BLACK,
                outlineWidth: 2,
              },
              label: {
                text: `${point.fixIdent}\n${point.role}`,
                font: "13px monospace",
                fillColor: Cesium.Color.WHITE,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                outlineWidth: 2,
                verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                pixelOffset: new Cesium.Cartesian2(0, -18),
                distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 80000),
              },
            });
            addRouteEntityId(route.routeId, id);
          });
        });
      })
      .catch((error) => {
        if (isMissingJsonAsset(error)) {
          console.warn(
            `[useProcedureLayer] procedure-details data for ${activeAirportCode} not found. ` +
              "Run: python aeroviz-4d/python/preprocess_procedures.py --airport <ICAO>",
          );
        } else {
          console.error("[useProcedureLayer]", error);
        }
      });

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        addedIds.forEach((id) => viewer.entities.removeById(id));
      }
      routeEntityIdsRef.current = {};
      routeDefaultsRef.current = {};
    };
  }, [viewer, activeAirportCode]);
}
