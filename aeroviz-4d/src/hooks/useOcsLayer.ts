/**
 * useOcsLayer.ts
 * --------------
 * Renders PANS-OPS Final Approach Obstacle Clearance Surfaces (OCS)
 * derived from procedure FAF → threshold pairs.
 *
 * Data source:
 *   public/data/procedures.geojson   (produced by preprocess_procedures.py)
 *
 * For every `procedure-route` feature we:
 *   1. Locate the FAF sample and the runway/MAPt sample in `properties.samples`
 *   2. Read their (lon, lat, altM) from the matching LineString vertices
 *   3. Call `buildFinalApproachOCS()` with primary + secondary widths derived
 *      from the route's tunnel descriptor (fallback to PANS-OPS defaults)
 *   4. Draw three semi-transparent polygons (primary + left/right secondary)
 *
 * Follows the same dual-useEffect pattern as useObstacleLayer / useWaypointLayer.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  buildFinalApproachOCS,
  type GeoPoint3D,
  type Polygon3D,
} from "../utils/ocsGeometry";
import type {
  ProcedureFeature,
  ProcedureFeatureCollection,
  ProcedureRouteProperties,
} from "../types/geojson-aviation";

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

function isRouteFeature(
  feature: ProcedureFeature,
): feature is ProcedureFeature & {
  geometry: { type: "LineString"; coordinates: Array<[number, number, number]> };
  properties: ProcedureRouteProperties;
} {
  return (
    feature.geometry.type === "LineString" &&
    feature.properties.featureType === "procedure-route"
  );
}

function coordToPoint(coord: [number, number, number]): GeoPoint3D {
  return { lon: coord[0], lat: coord[1], altM: coord[2] ?? 0 };
}

/**
 * Find the (FAF, runway-threshold) pair in a route's samples array.
 *
 * The runway end is conventionally tagged with role "MAPt" (Missed Approach
 * Point) for non-precision approaches — it coincides with the runway threshold.
 * If no MAPt is present we fall back to the final sample.
 */
function findFafAndThreshold(
  props: ProcedureRouteProperties,
  coords: Array<[number, number, number]>,
): { faf: GeoPoint3D; threshold: GeoPoint3D } | null {
  const samples = props.samples;
  if (!samples || samples.length < 2) return null;
  if (samples.length !== coords.length) return null;

  const fafIdx = samples.findIndex((s) => s.role === "FAF");
  if (fafIdx < 0) return null;

  let thrIdx = samples.findIndex((s) => s.role === "MAPt");
  if (thrIdx < 0) thrIdx = samples.length - 1;
  if (thrIdx <= fafIdx) return null;

  return {
    faf: coordToPoint(coords[fafIdx]),
    threshold: coordToPoint(coords[thrIdx]),
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

export function useOcsLayer(): void {
  const { viewer, layers } = useApp();

  // Effect 1 — load procedures.geojson, build OCS entities.
  useEffect(() => {
    if (!viewer) return;

    let cancelled = false;
    const addedIds: string[] = [];

    fetch("/data/procedures.geojson")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading procedures.geojson`);
        return r.json() as Promise<ProcedureFeatureCollection>;
      })
      .then((geojson) => {
        if (cancelled) return;

        geojson.features.filter(isRouteFeature).forEach((feature, routeIndex) => {
          const props = feature.properties;
          const routeId = props.routeId ?? `route-${routeIndex}`;
          const pair = findFafAndThreshold(props, feature.geometry.coordinates);
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
          const tunnelHalfWidthM = props.tunnel?.lateralHalfWidthNm
            ? props.tunnel.lateralHalfWidthNm * NAUTICAL_MILE_M
            : DEFAULT_PRIMARY_HALF_WIDTH_M;

          const geom = buildFinalApproachOCS({
            faf: pair.faf,
            threshold: pair.threshold,
            primaryHalfWidthM: tunnelHalfWidthM,
            secondaryWidthM: tunnelHalfWidthM > 0
              ? tunnelHalfWidthM
              : DEFAULT_SECONDARY_WIDTH_M,
          });

          const visible = layers.ocsSurfaces;
          const baseId = `${OCS_ENTITY_PREFIX}${routeId}`;

          const primaryId = `${baseId}-primary`;
          addOcsPolygon(
            viewer,
            primaryId,
            `${props.procedureName} OCS primary`,
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
            `${props.procedureName} OCS secondary (L)`,
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
            `${props.procedureName} OCS secondary (R)`,
            geom.secondaryRight,
            SECONDARY_FILL_COLOR,
            SECONDARY_OUTLINE_COLOR,
            visible,
          );
          addedIds.push(rightId);
        });
      })
      .catch((err) => {
        if (err instanceof Error && err.message.includes("404")) {
          console.warn(
            "[useOcsLayer] procedures.geojson not found. " +
              "Run: python aeroviz-4d/python/preprocess_procedures.py",
          );
        } else {
          console.error("[useOcsLayer]", err);
        }
      });

    return () => {
      cancelled = true;
      addedIds.forEach((id) => viewer.entities.removeById(id));
    };
    // layers.ocsSurfaces is intentionally NOT in the dep list — toggling
    // visibility is handled by Effect 2 without rebuilding geometry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewer]);

  // Effect 2 — sync visibility when the layer toggle changes.
  useEffect(() => {
    if (!viewer) return;
    viewer.entities.values.forEach((entity) => {
      if (String(entity.id).startsWith(OCS_ENTITY_PREFIX)) {
        entity.show = layers.ocsSurfaces;
      }
    });
  }, [viewer, layers.ocsSurfaces]);
}
