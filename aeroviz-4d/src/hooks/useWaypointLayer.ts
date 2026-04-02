/**
 * useWaypointLayer.ts
 * -------------------
 * Custom hook: render approach waypoints (IAF, IF, FAF, MAPt) as 3D cylinder
 * markers with text labels in the Cesium scene.
 *
 * Why cylinders instead of billboard icons?
 *   Cylinders are visible at a wide range of altitudes (5 km to 50 km).
 *   Billboard icons disappear when too close or too far.
 *   For a research demo, 3D geometry is more visually striking.
 *
 * 📖 Tutorial: see docs/02-runway-layer.md § "Waypoints"
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import type { WaypointProperties } from "../types/geojson-aviation";

/** Cylinder colour per waypoint type */
const WAYPOINT_COLORS: Record<string, Cesium.Color> = {
  IAF: Cesium.Color.CYAN.withAlpha(0.8),
  IF: Cesium.Color.LIGHTBLUE.withAlpha(0.8),
  FAF: Cesium.Color.YELLOW.withAlpha(0.9),
  MAPt: Cesium.Color.ORANGE.withAlpha(0.9),
};
const DEFAULT_COLOR = Cesium.Color.WHITE.withAlpha(0.7);

/** Container name used to look up and remove the layer */
const LAYER_NAME = "waypoints";

export function useWaypointLayer(): void {
  const { viewer, layers } = useApp();

  useEffect(() => {
    if (!viewer) return;

    // We collect the entity IDs we add so we can clean up precisely.
    const addedIds: string[] = [];

    fetch("/data/waypoints.geojson")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading waypoints.geojson`);
        return r.json() as Promise<GeoJSON.FeatureCollection>;
      })
      .then((geojson) => {
        geojson.features.forEach((feature) => {
          // GeoJSON Point geometry: [longitude, latitude, altitude_metres]
          const [lon, lat, altM] = feature.geometry.type === "Point"
            ? (feature.geometry.coordinates as [number, number, number])
            : [0, 0, 0];

          const props = feature.properties as WaypointProperties;

          // TODO ① — Add an entity with:
          //   id:       `waypoint-${props.name}`
          //   name:     props.name
          //   position: Cesium.Cartesian3.fromDegrees(lon, lat, altM)
          //   cylinder: { length: 400, topRadius: 200, bottomRadius: 200,
          //               material: WAYPOINT_COLORS[props.type] ?? DEFAULT_COLOR }
          //   label:    { text: props.name, font: "14px monospace",
          //               fillColor: Cesium.Color.WHITE,
          //               style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          //               outlineWidth: 2,
          //               verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          //               pixelOffset: new Cesium.Cartesian2(0, -30) }
          //
          // After calling viewer.entities.add(...), push the id to addedIds[].
          //
          // Hint: Cesium.Cartesian3.fromDegrees expects (longitude, latitude, height).
          // Altitude from GeoJSON is in metres — pass it directly.
        });
      })
      .catch(console.error);

    return () => {
      // Remove only the entities this hook added — don't wipe the whole scene.
      addedIds.forEach((id) => viewer.entities.removeById(id));
    };
  }, [viewer]);

  // Sync visibility
  useEffect(() => {
    if (!viewer) return;
    // TODO ② — Loop over viewer.entities.values and for each entity whose
    // id starts with "waypoint-", set entity.show = layers.waypoints.
    //
    // Hint: id.startsWith("waypoint-")
  }, [viewer, layers.waypoints]);
}
