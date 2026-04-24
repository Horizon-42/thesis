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

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

const RUNWAY_SURFACE_FILL = new Cesium.Color(0.15, 0.15, 0.15, 0.85);
const RUNWAY_SURFACE_STROKE = Cesium.Color.YELLOW;
const LANDING_ZONE_FILL = new Cesium.Color(0.65, 0.9, 0.65, 0.35);
const LANDING_ZONE_STROKE = new Cesium.Color(0.3, 0.7, 0.3, 0.9);

export function useRunwayLayer(): void {
  const { viewer, layers, activeAirportCode } = useApp();
  // Hold a direct reference — GeoJsonDataSource.load() overwrites the name
  // from the URL, so getByName() is unreliable for visibility sync.
  const dsRef = useRef<Cesium.GeoJsonDataSource | null>(null);

  useEffect(() => {
    // Don't do anything until the Viewer is ready.
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    let added = false;
    const runwayUrl = airportDataUrl(activeAirportCode, "runway.geojson");

    // ── Step 1: Create a named DataSource container ───────────────────────────
    // Naming it "runways" lets us find and remove it precisely later.
    const dataSource = new Cesium.GeoJsonDataSource("runways");

    // ── Step 2: Load the GeoJSON file ─────────────────────────────────────────
    // ① — Call `dataSource.load(url, options)` with:
    //   url:     "/data/airports/<ICAO>/runway.geojson"   (served from public/data/)
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
    fetchJson<unknown>(runwayUrl)
      .then(() => dataSource.load(runwayUrl, {
      clampToGround: true,
      fill: RUNWAY_SURFACE_FILL,
      stroke: RUNWAY_SURFACE_STROKE,
      strokeWidth: 2
    }))
    .then(ds => {
      if (cancelled) return;
      // ── Step 3 (inside .then): Add the data source to the viewer ─────────────
      // ② — Call `viewer.dataSources.add(ds)`.
      // This callback runs after the GeoJSON is loaded and parsed into entities.
      // The `ds` argument is the same DataSource we created above, but now it
      // contains entities representing the runway polygons.
      viewer.dataSources.add(ds);
      added = true;
      dsRef.current = ds;
      ds.show = layers.runways;

      // ── Step 4 (inside .then): Set ClassificationType on each entity ──────────
      // ③ — Loop over `ds.entities.values`.  For each entity, if
      //   `entity.polygon` exists, set:
      //     entity.polygon.classificationType =
      //       new Cesium.ConstantProperty(Cesium.ClassificationType.TERRAIN);
      //
      // Why: ClassificationType.TERRAIN means the polygon only drapes on terrain
      // and does NOT occlude aircraft models or other scene primitives.
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
      if (isMissingJsonAsset(error)) {
        console.warn(`[RunwayLayer] ${runwayUrl} not found.`);
      } else {
        console.error("[RunwayLayer] Failed to load runway.geojson:", error);
      }
    });



    // ── Step 5: Respect the layer visibility flag ─────────────────────────────
    // ④ — After adding the DataSource, set its `show` property:
    dataSource.show = layers.runways;
    //
    // This way, toggling layers.runways in ControlPanel will hide/show runways.

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      cancelled = true;
      dsRef.current = null;
      if (added && isCesiumViewerUsable(viewer)) {
        viewer.dataSources.remove(dataSource, true);
      }
    };
  }, [viewer, activeAirportCode]); // Re-run if the Viewer instance changes

  // ── Separate effect: sync visibility when the layer flag changes ───────────
  useEffect(() => {
    if (dsRef.current) dsRef.current.show = layers.runways;
  }, [layers.runways]);
}
