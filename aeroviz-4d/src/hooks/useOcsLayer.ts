/**
 * useOcsLayer.ts
 * --------------
 * Legacy debug layer for simple FAF-to-threshold obstacle clearance surfaces.
 *
 * Data source:
 *   public/data/airports/<ICAO>/procedure-details/*.json
 *
 * This predates the v3 procedure segment layer. Keep it opt-in so the default
 * scene does not mix these simplified red/orange surfaces with annotated v3
 * procedure OEA/OCS geometry.
 *
 * For every final approach route we:
 *   1. Locate the FAF point and the runway/MAPt point in the canonical route model
 *   2. Read their (lon, lat, altM) from procedure-details-derived points
 *   3. Call `buildFinalApproachOCS()` with primary + secondary widths derived
 *      from the route's tunnel descriptor (fallback to PANS-OPS defaults)
 *   4. Draw three semi-transparent polygons (primary + left/right secondary)
 *
 * Follows the same dual-useEffect pattern as useObstacleLayer / useWaypointLayer.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { loadProcedureRouteData, type ProcedureRouteViewModel } from "../data/procedureRoutes";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  buildFinalApproachOCS,
  type GeoPoint3D,
  type Polygon3D,
} from "../utils/ocsGeometry";

// ── Constants ───────────────────────────────────────────────────────────────

const OCS_ENTITY_PREFIX = "ocs-";

/** Fallback primary half-width if the procedure has no tunnel descriptor.
 *  PANS-OPS Vol II Part I — typical ILS Cat I final-approach primary half-width
 *  is ~150 m at the FAF; for RNAV it scales with RNP (0.3 NM ≈ 556 m). */
const DEFAULT_PRIMARY_HALF_WIDTH_M = 150;

/** Fallback secondary width. Typical PANS-OPS value: equal to primary half-width. */
const DEFAULT_SECONDARY_WIDTH_M = 150;

const NAUTICAL_MILE_M = 1852;

// ── Styling ────────────────────────────────────────────────────────────────

const PRIMARY_FILL_COLOR = Cesium.Color.RED.withAlpha(0.28);
const PRIMARY_OUTLINE_COLOR = Cesium.Color.RED.withAlpha(0.9);
const SECONDARY_FILL_COLOR = Cesium.Color.ORANGE.withAlpha(0.22);
const SECONDARY_OUTLINE_COLOR = Cesium.Color.ORANGE.withAlpha(0.85);

// ── Helpers ────────────────────────────────────────────────────────────────

function routePointToGeoPoint(point: ProcedureRouteViewModel["points"][number]): GeoPoint3D {
  return { lon: point.lon, lat: point.lat, altM: point.altM };
}

/**
 * Find the (FAF, runway-threshold) pair in a route's samples array.
 *
 * The runway end is conventionally tagged with role "MAPt" (Missed Approach
 * Point) for non-precision approaches — it coincides with the runway threshold.
 * If no MAPt is present we fall back to the final sample.
 */
function findFafAndThreshold(
  route: ProcedureRouteViewModel,
): { faf: GeoPoint3D; threshold: GeoPoint3D } | null {
  const samples = route.points;
  if (samples.length < 2) return null;

  const fafIdx = samples.findIndex((s) => s.role === "FAF");
  if (fafIdx < 0) return null;

  let thrIdx = samples.findIndex((s) => s.role === "MAPt");
  if (thrIdx < 0) thrIdx = samples.length - 1;
  if (thrIdx <= fafIdx) return null;

  return {
    faf: routePointToGeoPoint(samples[fafIdx]),
    threshold: routePointToGeoPoint(samples[thrIdx]),
  };
}

function polygonHierarchy(polygon: Polygon3D): Cesium.PolygonHierarchy {
  return new Cesium.PolygonHierarchy(
    polygon.map((p) => Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.altM)),
  );
}

function addOcsPolygon(
  viewer: Cesium.Viewer,
  id: string,
  name: string,
  polygon: Polygon3D,
  fill: Cesium.Color,
  outline: Cesium.Color,
  visible: boolean,
): void {
  viewer.entities.add({
    id,
    name,
    show: visible,
    polygon: {
      hierarchy: polygonHierarchy(polygon),
      material: fill,
      perPositionHeight: true,
      outline: true,
      outlineColor: outline,
    },
  });
}

// ── Hook ────────────────────────────────────────────────────────────────────

export function useOcsLayer({ enabled = false }: { enabled?: boolean } = {}): void {
  const { viewer, layers, activeAirportCode } = useApp();

  // Effect 1 — load canonical procedure details, build OCS entities.
  useEffect(() => {
    if (!enabled || !viewer || !activeAirportCode) return;

    let cancelled = false;
    const addedIds: string[] = [];

    loadProcedureRouteData(activeAirportCode)
      .then(({ routes }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        routes
          .filter((route) => route.branchType.toLowerCase() === "final")
          .forEach((route, routeIndex) => {
            const routeId = route.routeId ?? `route-${routeIndex}`;
            const pair = findFafAndThreshold(route);
            if (!pair) {
              console.warn(
                `[useOcsLayer] Skipping ${routeId}: could not resolve FAF→threshold pair`,
              );
              return;
            }

            // Derive widths from the published tunnel descriptor when available.
            // `lateralHalfWidthNm` is the navigation tunnel half-width; we use it
            // as the PRIMARY half-width (full containment) and extend by the same
            // amount to form the SECONDARY (7:1) fringe.
            const tunnelHalfWidthM = route.tunnel?.lateralHalfWidthNm
              ? route.tunnel.lateralHalfWidthNm * NAUTICAL_MILE_M
              : DEFAULT_PRIMARY_HALF_WIDTH_M;

            const geom = buildFinalApproachOCS({
              faf: pair.faf,
              threshold: pair.threshold,
              primaryHalfWidthM: tunnelHalfWidthM,
              secondaryWidthM: tunnelHalfWidthM > 0
                ? tunnelHalfWidthM
                : DEFAULT_SECONDARY_WIDTH_M,
            });

            const visible = enabled;
            const baseId = `${OCS_ENTITY_PREFIX}${routeId}`;

            const primaryId = `${baseId}-primary`;
            addOcsPolygon(
              viewer,
              primaryId,
              `${route.procedureName} OCS primary`,
              geom.primaryPolygon,
              PRIMARY_FILL_COLOR,
              PRIMARY_OUTLINE_COLOR,
              visible,
            );
            addedIds.push(primaryId);

            const leftId = `${baseId}-secondary-left`;
            addOcsPolygon(
              viewer,
              leftId,
              `${route.procedureName} OCS secondary (L)`,
              geom.secondaryLeft,
              SECONDARY_FILL_COLOR,
              SECONDARY_OUTLINE_COLOR,
              visible,
            );
            addedIds.push(leftId);

            const rightId = `${baseId}-secondary-right`;
            addOcsPolygon(
              viewer,
              rightId,
              `${route.procedureName} OCS secondary (R)`,
              geom.secondaryRight,
              SECONDARY_FILL_COLOR,
              SECONDARY_OUTLINE_COLOR,
              visible,
            );
            addedIds.push(rightId);
          });
      })
      .catch((err) => {
        if (isMissingJsonAsset(err)) {
          console.warn(
            `[useOcsLayer] procedure-details data for ${activeAirportCode} not found. ` +
              "Run: python aeroviz-4d/python/preprocess_procedures.py --airport <ICAO>",
          );
        } else {
          console.error("[useOcsLayer]", err);
        }
      });

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        addedIds.forEach((id) => viewer.entities.removeById(id));
      }
    };
  }, [viewer, activeAirportCode, enabled]);

  // Effect 2 — sync visibility when the layer toggle changes.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    viewer.entities.values.forEach((entity) => {
      if (String(entity.id).startsWith(OCS_ENTITY_PREFIX)) {
        entity.show = enabled && layers.ocsSurfaces;
      }
    });
  }, [viewer, enabled, layers.ocsSurfaces]);
}
