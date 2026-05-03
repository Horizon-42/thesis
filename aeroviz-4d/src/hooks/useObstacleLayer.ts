/**
 * useObstacleLayer.ts
 * -------------------
 * Custom hook: render FAA DOF obstacles (towers, windmills, buildings, etc.)
 * as 3D cylinder markers, with optional text labels in the Cesium scene.
 *
 * Data source: public/data/airports/<ICAO>/obstacles.geojson, produced by
 *   python preprocess_obstacles.py --input <DOF .Dat file> --airport
 *
 * Follows the same dual-useEffect pattern as useWaypointLayer:
 *   Effect 1 — load GeoJSON, create entities, track IDs for cleanup
 *   Effect 2 — sync visibility with layers.obstacles / layers.obstacleLabels
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import type { ObstacleProperties } from "../types/geojson-aviation";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

// ── Colour by obstacle type ─────────────────────────────────────────────────

const OBSTACLE_COLORS: Record<string, Cesium.Color> = {
  TOWER:          Cesium.Color.RED.withAlpha(0.85),
  "T-L TWR":      Cesium.Color.RED.withAlpha(0.85),
  "CTRL TWR":     Cesium.Color.DARKRED.withAlpha(0.85),
  BLDG:           Cesium.Color.STEELBLUE.withAlpha(0.8),
  "BLDG-TWR":     Cesium.Color.STEELBLUE.withAlpha(0.8),
  STACK:          Cesium.Color.ORANGE.withAlpha(0.85),
  "COOL TWR":     Cesium.Color.ORANGE.withAlpha(0.85),
  TANK:           Cesium.Color.DARKGOLDENROD.withAlpha(0.8),
  SILO:           Cesium.Color.DARKGOLDENROD.withAlpha(0.8),
  WINDMILL:       Cesium.Color.LIMEGREEN.withAlpha(0.85),
  ANTENNA:        Cesium.Color.MAGENTA.withAlpha(0.8),
  CATENARY:       Cesium.Color.YELLOW.withAlpha(0.8),
  POLE:           Cesium.Color.SALMON.withAlpha(0.7),
  "UTILITY POLE": Cesium.Color.SALMON.withAlpha(0.7),
  CRANE:          Cesium.Color.DARKCYAN.withAlpha(0.8),
  SIGN:           Cesium.Color.SANDYBROWN.withAlpha(0.7),
  BRIDGE:         Cesium.Color.SLATEGRAY.withAlpha(0.8),
};
const DEFAULT_OBSTACLE_COLOR = Cesium.Color.WHITE.withAlpha(0.7);
const OBSTACLE_MARKER_PREFIX = "obstacle-marker-";
const OBSTACLE_LABEL_PREFIX = "obstacle-label-";

// ── GeoJSON types ───────────────────────────────────────────────────────────

interface ObstacleFeature {
  geometry: {
    type: string;
    coordinates: [number, number, number?];
  };
  properties: ObstacleProperties;
}

interface ObstacleFeatureCollection {
  features: ObstacleFeature[];
}

// ── Constants ───────────────────────────────────────────────────────────────

const CYLINDER_RADIUS = 20; // metres — fixed for all obstacles
const MIN_CYLINDER_LENGTH = 10; // metres — minimum visible height
const LABEL_VERTICAL_GAP_M = 8;
const LABEL_MAX_DISTANCE_M = 15000;

function formatObstacleLabel(props: ObstacleProperties): string {
  const type = props.obstacle_type.trim() || "OBSTACLE";
  return `${type} · ${Math.round(props.agl_ft)} ft AGL`;
}

// ── Hook ────────────────────────────────────────────────────────────────────

export function useObstacleLayer(): void {
  const { viewer, layers, activeAirportCode } = useApp();
  const layersRef = useRef(layers);

  useEffect(() => {
    layersRef.current = layers;
  }, [layers]);

  // Effect 1: Load GeoJSON, create entities
  useEffect(() => {
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    const addedIds: string[] = [];
    const obstacleUrl = airportDataUrl(activeAirportCode, "obstacles.geojson");

    fetchJson<ObstacleFeatureCollection>(obstacleUrl)
      .then((geojson) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        geojson.features.forEach((feature, index) => {
          if (feature.geometry.type !== "Point") return;

          const [lon, lat] = feature.geometry.coordinates;
          const props = feature.properties;
          const aglM = props.agl_m ?? 0;
          const cylinderLength = Math.max(aglM, MIN_CYLINDER_LENGTH);

          // Position the cylinder centre at half the AGL height RELATIVE TO
          // GROUND.  This makes the bottom of the cylinder touch the terrain
          // surface regardless of any MSL ↔ ellipsoid mismatch.
          const centerAboveGroundM = cylinderLength / 2;

          const color =
            OBSTACLE_COLORS[props.obstacle_type] ?? DEFAULT_OBSTACLE_COLOR;
          const entityKey = `${props.oas_number}-${index}`;
          const markerId = `${OBSTACLE_MARKER_PREFIX}${entityKey}`;
          const labelId = `${OBSTACLE_LABEL_PREFIX}${entityKey}`;
          const labelAboveGroundM = cylinderLength + LABEL_VERTICAL_GAP_M;
          const currentLayers = layersRef.current;

          viewer.entities.add({
            id: markerId,
            name: `${props.obstacle_type} (${props.oas_number})`,
            show: currentLayers.obstacles,
            position: Cesium.Cartesian3.fromDegrees(
              lon,
              lat,
              centerAboveGroundM,
            ),
            cylinder: {
              length: cylinderLength,
              topRadius: CYLINDER_RADIUS,
              bottomRadius: CYLINDER_RADIUS,
              material: color,
              heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
            },
          });
          addedIds.push(markerId);

          viewer.entities.add({
            id: labelId,
            name: `${props.obstacle_type} label (${props.oas_number})`,
            show: currentLayers.obstacles && currentLayers.obstacleLabels,
            position: Cesium.Cartesian3.fromDegrees(
              lon,
              lat,
              labelAboveGroundM,
            ),
            label: {
              text: formatObstacleLabel(props),
              font: "11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineColor: Cesium.Color.BLACK.withAlpha(0.9),
              outlineWidth: 3,
              showBackground: true,
              backgroundColor: Cesium.Color.BLACK.withAlpha(0.48),
              backgroundPadding: new Cesium.Cartesian2(5, 3),
              heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
              horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -4),
              scaleByDistance: new Cesium.NearFarScalar(500, 1.0, LABEL_MAX_DISTANCE_M, 0.62),
              translucencyByDistance: new Cesium.NearFarScalar(
                LABEL_MAX_DISTANCE_M * 0.65,
                1.0,
                LABEL_MAX_DISTANCE_M,
                0.0,
              ),
              distanceDisplayCondition: new Cesium.DistanceDisplayCondition(
                0,
                LABEL_MAX_DISTANCE_M,
              ),
              disableDepthTestDistance: 4000,
            },
          });
          addedIds.push(labelId);
        });
      })
      .catch((err) => {
        // Graceful degradation: if obstacles.geojson doesn't exist yet,
        // just log a warning — the user hasn't run preprocess_obstacles.py.
        if (isMissingJsonAsset(err)) {
          console.warn(
            `[useObstacleLayer] ${obstacleUrl} not found. ` +
              "Run: python preprocess_obstacles.py --input <DOF .Dat> --airport-code <ICAO>",
          );
        } else {
          console.error("[useObstacleLayer]", err);
        }
      });

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        addedIds.forEach((id) => viewer.entities.removeById(id));
      }
    };
  }, [viewer, activeAirportCode]);

  // Effect 2: Sync visibility with layer toggles
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    viewer.entities.values.forEach((entity) => {
      const id = String(entity.id);
      if (id.startsWith(OBSTACLE_MARKER_PREFIX)) {
        entity.show = layers.obstacles;
      } else if (id.startsWith(OBSTACLE_LABEL_PREFIX)) {
        entity.show = layers.obstacles && layers.obstacleLabels;
      }
    });
  }, [viewer, layers.obstacles, layers.obstacleLabels]);
}
