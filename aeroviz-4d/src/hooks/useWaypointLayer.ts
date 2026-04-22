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
import { airportDataUrl } from "../data/airportData";
import type { WaypointProperties } from "../types/geojson-aviation";

/** Cylinder colour per waypoint type */
const WAYPOINT_COLORS: Record<string, Cesium.Color> = {
  IAF: Cesium.Color.CYAN.withAlpha(0.8),
  IF: Cesium.Color.LIGHTBLUE.withAlpha(0.8),
  FAF: Cesium.Color.YELLOW.withAlpha(0.9),
  MAPt: Cesium.Color.ORANGE.withAlpha(0.9),
};
const DEFAULT_COLOR = Cesium.Color.WHITE.withAlpha(0.7);

interface WaypointFeature {
  geometry: {
    type: string;
    coordinates: [number, number, number?];
  };
  properties: WaypointProperties;
}

interface WaypointFeatureCollection {
  features: WaypointFeature[];
}

export function useWaypointLayer(): void {
  const { viewer, layers, activeAirportCode } = useApp();

  useEffect(() => {
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    const waypointUrl = airportDataUrl(activeAirportCode, "waypoints.geojson");

    // We collect the entity IDs we add so we can clean up precisely.
    const addedIds: string[] = [];

    fetch(waypointUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading waypoints.geojson`);
        return r.json() as Promise<WaypointFeatureCollection>;
      })
      .then((geojson) => {
        if (cancelled) return;
        geojson.features.forEach((feature, index) => {
          // GeoJSON Point geometry: [longitude, latitude, altitude_metres]
          if (feature.geometry.type !== "Point") {
            return;
          }

          const [lon, lat, altMaybe] = feature.geometry.coordinates;
          const altM = altMaybe ?? 0;

          const props = feature.properties;

          const id = `waypoint-${props.name}-${index}`;
          viewer.entities.add({
            id,
            name: props.name,
            position: Cesium.Cartesian3.fromDegrees(lon, lat, altM),
            cylinder: {
              length: 400,
              topRadius: 200,
              bottomRadius: 200,
              material: WAYPOINT_COLORS[props.type] ?? DEFAULT_COLOR,
            },
            label: {
              text: props.name,
              font: "14px monospace",
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -30),
            },
          });
          addedIds.push(id);
        });
      })
      .catch(console.error);

    return () => {
      cancelled = true;
      // Remove only the entities this hook added — don't wipe the whole scene.
      addedIds.forEach((id) => viewer.entities.removeById(id));
    };
  }, [viewer, activeAirportCode]);

  // Sync visibility
  useEffect(() => {
    if (!viewer) return;
    viewer.entities.values.forEach((entity) => {
      const id = String(entity.id);
      if (id.startsWith("waypoint-")) {
        entity.show = layers.waypoints;
      }
    });
  }, [viewer, layers.waypoints]);
}
