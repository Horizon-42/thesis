import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

/**
 * Toggle world terrain on/off.
 *
 * ON  → CesiumTerrainProvider from Ion asset 1 (Cesium World Terrain)
 * OFF → EllipsoidTerrainProvider (flat, no elevation — imagery stays visible)
 *
 * On first mount we skip: useCesiumViewer already set world terrain via the
 * Viewer constructor.  Re-applying would cause a redundant network fetch.
 */
export function useTerrainLayer(): void {
  const { viewer, layers } = useApp();
  const initializedRef = useRef(false);

  useEffect(() => {
    if (!viewer) return;

    // Skip the very first run — terrain is already correct from the constructor.
    if (!initializedRef.current) {
      initializedRef.current = true;
      if (layers.terrain) return;
    }

    // Cancelled by cleanup if the user toggles again before the async load finishes.
    let cancelled = false;

    if (layers.terrain) {
      // Re-create world terrain from Cesium Ion (asset 1).
      // This avoids holding stale provider references that conflict
      // with Cesium's internal async Terrain management.
      Cesium.CesiumTerrainProvider.fromIonAssetId(1, {
        requestVertexNormals: true,
        requestWaterMask: true,
      }).then((provider) => {
        if (!cancelled && !viewer.isDestroyed()) {
          viewer.scene.terrainProvider = provider;
        }
      });
    } else {
      viewer.scene.terrainProvider = new Cesium.EllipsoidTerrainProvider();
    }

    return () => { cancelled = true; };
  }, [viewer, layers.terrain]);
}
