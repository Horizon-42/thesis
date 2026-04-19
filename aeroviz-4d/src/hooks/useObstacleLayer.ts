/**
 * useObstacleLayer.ts
 * -------------------
 * Custom hook: render FAA DOF obstacles (towers, windmills, buildings, etc.)
 * as 3D cylinder markers with text labels in the Cesium scene.
 *
 * Data source: public/data/obstacles.geojson, produced by
 *   python preprocess_obstacles.py --input <DOF .Dat file> --airport
 *
 * Follows the same dual-useEffect pattern as useWaypointLayer:
 *   Effect 1 — load GeoJSON, create entities, track IDs for cleanup
 *   Effect 2 — sync visibility with layers.obstacles toggle
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import type { ObstacleProperties } from "../types/geojson-aviation";

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

// ── Hook ────────────────────────────────────────────────────────────────────

export function useObstacleLayer(): void {
  const { viewer, layers } = useApp();

  // Effect 1: Load GeoJSON, create entities
  useEffect(() => {
    if (!viewer) return;

    let cancelled = false;
    const addedIds: string[] = [];

    fetch("/data/obstacles.geojson")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading obstacles.geojson`);
        return r.json() as Promise<ObstacleFeatureCollection>;
      })
      .then((geojson) => {
        if (cancelled) return;

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
          const id = `obstacle-${props.oas_number}-${index}`;

          viewer.entities.add({
            id,
            name: `${props.obstacle_type} (${props.oas_number})`,
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
            label: {
              text: `${props.obstacle_type}\n${props.agl_ft} ft AGL`,
              font: "12px monospace",
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -20),
              distanceDisplayCondition: new Cesium.DistanceDisplayCondition(
                0,
                50000,
              ),
            },
          });
          addedIds.push(id);
        });
      })
      .catch((err) => {
        // Graceful degradation: if obstacles.geojson doesn't exist yet,
        // just log a warning — the user hasn't run preprocess_obstacles.py.
        if (err instanceof Error && err.message.includes("404")) {
          console.warn(
            "[useObstacleLayer] obstacles.geojson not found. " +
              "Run: python preprocess_obstacles.py --input <DOF .Dat> --airport",
          );
        } else {
          console.error("[useObstacleLayer]", err);
        }
      });

    return () => {
      cancelled = true;
      addedIds.forEach((id) => viewer.entities.removeById(id));
    };
  }, [viewer]);

  // Effect 2: Sync visibility with layer toggle
  useEffect(() => {
    if (!viewer) return;
    viewer.entities.values.forEach((entity) => {
      const id = String(entity.id);
      if (id.startsWith("obstacle-")) {
        entity.show = layers.obstacles;
      }
    });
  }, [viewer, layers.obstacles]);
}
