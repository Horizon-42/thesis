/**
 * useRunwayLayer.ts
 * -----------------
 * Custom hook: load runway polygons from runway.geojson and render them
 * clamped to the terrain surface in the Cesium Viewer.
 *
 * Key concepts:
 *   • GeoJsonDataSource  — Cesium's built-in parser for GeoJSON files.
 *   • clampToGround      — ensures polygons stick to the terrain surface
 *                          even as the terrain elevation changes.
 *   • ClassificationType — controls whether a clamped shape occludes terrain,
 *                          3D tiles, or both.
 *
 * 📖 Tutorial: see docs/02-runway-layer.md
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

const RUNWAY_SURFACE_FILL = new Cesium.Color(0.15, 0.15, 0.15, 0.85);
const RUNWAY_SURFACE_STROKE = Cesium.Color.YELLOW;
const LANDING_ZONE_FILL = new Cesium.Color(0.65, 0.9, 0.65, 0.35);
const LANDING_ZONE_STROKE = new Cesium.Color(0.3, 0.7, 0.3, 0.9);

export function useRunwayLayer(): void {
  const { viewer, layers } = useApp();

  useEffect(() => {
    // Don't do anything until the Viewer is ready.
    if (!viewer) return;

    // ── Step 1: Create a named DataSource container ───────────────────────────
    // Naming it "runways" lets us find and remove it precisely later.
    const dataSource = new Cesium.GeoJsonDataSource("runways");

    // ── Step 2: Load the GeoJSON file ─────────────────────────────────────────
    // ① — Call `dataSource.load(url, options)` with:
    //   url:     "/data/runway.geojson"   (served from public/data/)
    //   options:
    //     • clampToGround: true
    //     • fill:          a dark-grey Color with ~0.9 alpha
    //                      (hint: new Cesium.Color(0.15, 0.15, 0.15, 0.9))
    //     • stroke:        yellow, full alpha
    //     • strokeWidth:   2
    //
    // Then chain `.then(ds => { ... })` to do steps 3-4.
    //
    // Reference: docs/02-runway-layer.md § "GeoJsonDataSource options"
    dataSource.load("/data/runway.geojson", {
      clampToGround: true,
      fill: RUNWAY_SURFACE_FILL,
      stroke: RUNWAY_SURFACE_STROKE,
      strokeWidth: 2
    }).then(ds => {
      // ── Step 3 (inside .then): Add the data source to the viewer ─────────────
      // ② — Call `viewer.dataSources.add(ds)`.
      // This callback runs after the GeoJSON is loaded and parsed into entities.
      // The `ds` argument is the same DataSource we created above, but now it
      // contains entities representing the runway polygons.
      viewer.dataSources.add(ds);

      // ── Step 4 (inside .then): Set ClassificationType on each entity ──────────
      // ③ — Loop over `ds.entities.values`.  For each entity, if
      //   `entity.polygon` exists, set:
      //     entity.polygon.classificationType =
      //       new Cesium.ConstantProperty(Cesium.ClassificationType.TERRAIN);
      //
      // Why: ClassificationType.TERRAIN means the polygon only drapes on terrain
      // and does NOT occlude 3D Tileset buildings or aircraft models.
      //
      // Hint: ConstantProperty wraps a static value for Cesium's property system.
      ds.entities.values.forEach(entity => {
        if (entity.polygon) {
          const zoneType = entity.properties?.zone_type?.getValue(Cesium.JulianDate.now());
          const isLandingZone = zoneType === "landing_zone";

          entity.polygon.material = new Cesium.ColorMaterialProperty(
            isLandingZone ? LANDING_ZONE_FILL : RUNWAY_SURFACE_FILL
          );
          entity.polygon.outline = new Cesium.ConstantProperty(true);
          entity.polygon.outlineColor = new Cesium.ConstantProperty(
            isLandingZone ? LANDING_ZONE_STROKE : RUNWAY_SURFACE_STROKE
          );
          // Draw landing zone above the runway surface where they overlap.
          entity.polygon.zIndex = new Cesium.ConstantProperty(isLandingZone ? 2 : 1);
          entity.polygon.classificationType = new Cesium.ConstantProperty(Cesium.ClassificationType.TERRAIN);
        }
      });

    }).catch(error => {
      console.error("[RunwayLayer] Failed to load runway.geojson:", error);
    });



    // ── Step 5: Respect the layer visibility flag ─────────────────────────────
    // ④ — After adding the DataSource, set its `show` property:
    dataSource.show = layers.runways;
    //
    // This way, toggling layers.runways in ControlPanel will hide/show runways.

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      // Remove the DataSource (and all its entities) when the hook unmounts
      // or when `viewer` changes.
      viewer.dataSources.remove(dataSource, true);
    };
  }, [viewer]); // Re-run if the Viewer instance changes

  // ── Separate effect: sync visibility when the layer flag changes ───────────
  // We split this into its own effect so toggling a layer does NOT reload the
  // GeoJSON from the network — it just flips the DataSource's show flag.
  useEffect(() => {
    if (!viewer) return;
    const ds = viewer.dataSources.getByName("runways")[0];
    if (ds) {
      // ⑤ — Set `ds.show = layers.runways;`
      ds.show = layers.runways;
    }
  }, [viewer, layers.runways]);
}
