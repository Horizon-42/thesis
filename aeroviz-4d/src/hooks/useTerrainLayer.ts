import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  applyWorldTerrainStreamingSettings,
  captureTerrainStreamingSettings,
  restoreTerrainStreamingSettings,
  type GlobeTerrainStreamingSettings,
} from "../terrain/terrainRuntime";

/**
 * Toggle world terrain on/off.
 *
 * ON  → CesiumTerrainProvider from Ion asset 1 (Cesium World Terrain)
 * OFF → EllipsoidTerrainProvider (flat, no elevation — imagery stays visible)
 *
 * The terrain runtime module owns the Cesium globe streaming policy. This hook
 * is only the React lifecycle adapter that chooses world versus ellipsoid when
 * airport local terrain is not already controlling the provider seam.
 *
 * The Viewer now starts on ellipsoid terrain.  Loading World Terrain is
 * deliberately opt-in because it creates network and GPU work before the user
 * asks for elevation context.
 */
export function useTerrainLayer(): void {
  const { viewer, layers, airportLocalTerrain } = useApp();
  const worldTerrainProviderRef = useRef<Cesium.TerrainProvider | null>(null);
  const previousStreamingSettingsRef = useRef<GlobeTerrainStreamingSettings | null>(null);
  const airportLocalTerrainOwnsProvider =
    layers.airportLocalTerrain && airportLocalTerrain?.status === "active";

  useEffect(() => {
    if (!viewer) return;
    if (airportLocalTerrainOwnsProvider) return;

    // Cancelled by cleanup if the user toggles again before the async load finishes.
    let cancelled = false;

    if (layers.terrain) {
      if (!previousStreamingSettingsRef.current) {
        previousStreamingSettingsRef.current = captureTerrainStreamingSettings(
          viewer.scene.globe,
        );
      }

      if (worldTerrainProviderRef.current) {
        viewer.scene.terrainProvider = worldTerrainProviderRef.current;
        applyWorldTerrainStreamingSettings(viewer);
        return;
      }

      // Re-create world terrain from Cesium Ion (asset 1).
      // This avoids holding stale provider references that conflict
      // with Cesium's internal async Terrain management.
      Cesium.CesiumTerrainProvider.fromIonAssetId(1, {
        requestVertexNormals: true,
        requestWaterMask: true,
      }).then((provider) => {
        if (viewer.isDestroyed()) return;
        worldTerrainProviderRef.current = provider;
        if (!cancelled) {
          viewer.scene.terrainProvider = provider;
          applyWorldTerrainStreamingSettings(viewer);
        }
      });
    } else {
      viewer.scene.terrainProvider = new Cesium.EllipsoidTerrainProvider();
      if (previousStreamingSettingsRef.current) {
        restoreTerrainStreamingSettings(viewer, previousStreamingSettingsRef.current);
        previousStreamingSettingsRef.current = null;
      }
    }

    return () => { cancelled = true; };
  }, [viewer, layers.terrain, airportLocalTerrainOwnsProvider]);
}
