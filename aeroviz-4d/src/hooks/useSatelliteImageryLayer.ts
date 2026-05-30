import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  applyNoImageryTerrainStyle,
  captureGlobeRenderState,
  createNoImageryTerrainMaterial,
  noImageryTerrainHeightRange,
  restoreGlobeRenderState,
  type GlobeRenderState,
} from "../terrain/terrainRuntime";

export {
  NO_IMAGERY_TERRAIN_MATERIAL_SOURCE,
  noImageryTerrainHeightRange,
} from "../terrain/terrainRuntime";

/**
 * Toggle the Cesium base imagery layer.  The Viewer creates this layer during
 * initialization, so this hook syncs visibility and provides a gray shaded
 * terrain style for the no-imagery state.
 *
 * Rendering policy lives in terrainRuntime; this hook only binds that policy to
 * the AppContext layer toggle and guarantees the captured globe state is
 * restored when imagery comes back.
 */
export function useSatelliteImageryLayer(): void {
  const { viewer, layers, airportLocalTerrain } = useApp();
  const previousGlobeStateRef = useRef<GlobeRenderState | null>(null);
  const noImageryMaterialRef = useRef<Cesium.Material | null>(null);

  useEffect(() => {
    if (!viewer || viewer.isDestroyed()) return;

    if (viewer.imageryLayers.length === 0) return;

    const baseLayer = viewer.imageryLayers.get(0);
    if (!baseLayer) return;

    baseLayer.show = layers.satelliteImagery;
    if (layers.satelliteImagery) {
      if (previousGlobeStateRef.current) {
        restoreGlobeRenderState(viewer, previousGlobeStateRef.current);
        previousGlobeStateRef.current = null;
        if (noImageryMaterialRef.current && !noImageryMaterialRef.current.isDestroyed()) {
          noImageryMaterialRef.current.destroy();
        }
        noImageryMaterialRef.current = null;
      } else {
        viewer.scene.requestRender();
      }
      return;
    }

    if (!previousGlobeStateRef.current) {
      previousGlobeStateRef.current = captureGlobeRenderState(viewer);
    }
    if (!noImageryMaterialRef.current || noImageryMaterialRef.current.isDestroyed()) {
      noImageryMaterialRef.current = createNoImageryTerrainMaterial();
    }
    applyNoImageryTerrainStyle(
      viewer,
      noImageryMaterialRef.current,
      noImageryTerrainHeightRange(airportLocalTerrain),
    );
  }, [
    viewer,
    layers.satelliteImagery,
    airportLocalTerrain.minimumHeightM,
    airportLocalTerrain.maximumHeightM,
  ]);

  useEffect(() => {
    return () => {
      if (!viewer || viewer.isDestroyed() || !previousGlobeStateRef.current) return;

      restoreGlobeRenderState(viewer, previousGlobeStateRef.current);
      previousGlobeStateRef.current = null;
      if (noImageryMaterialRef.current && !noImageryMaterialRef.current.isDestroyed()) {
        noImageryMaterialRef.current.destroy();
      }
      noImageryMaterialRef.current = null;
    };
  }, [viewer]);
}
