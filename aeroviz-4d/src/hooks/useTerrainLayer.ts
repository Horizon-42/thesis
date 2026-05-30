import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

const WORLD_TERRAIN_MAXIMUM_SCREEN_SPACE_ERROR = 1;
const WORLD_TERRAIN_MIN_TILE_CACHE_SIZE = 512;
const WORLD_TERRAIN_LOADING_DESCENDANT_LIMIT = 20;

interface GlobeTerrainStreamingSettings {
  maximumScreenSpaceError: number;
  tileCacheSize: number;
  loadingDescendantLimit: number;
  preloadAncestors: boolean;
  preloadSiblings: boolean;
}

function captureTerrainStreamingSettings(
  globe: Cesium.Globe,
): GlobeTerrainStreamingSettings {
  return {
    maximumScreenSpaceError: globe.maximumScreenSpaceError,
    tileCacheSize: globe.tileCacheSize,
    loadingDescendantLimit: globe.loadingDescendantLimit,
    preloadAncestors: globe.preloadAncestors,
    preloadSiblings: globe.preloadSiblings,
  };
}

function applyWorldTerrainStreamingSettings(viewer: Cesium.Viewer): void {
  const { globe } = viewer.scene;

  globe.maximumScreenSpaceError = WORLD_TERRAIN_MAXIMUM_SCREEN_SPACE_ERROR;
  globe.tileCacheSize = Math.max(globe.tileCacheSize, WORLD_TERRAIN_MIN_TILE_CACHE_SIZE);
  globe.loadingDescendantLimit = WORLD_TERRAIN_LOADING_DESCENDANT_LIMIT;
  globe.preloadAncestors = true;
  globe.preloadSiblings = true;
  viewer.scene.requestRender();
}

function restoreTerrainStreamingSettings(
  viewer: Cesium.Viewer,
  settings: GlobeTerrainStreamingSettings,
): void {
  const { globe } = viewer.scene;

  globe.maximumScreenSpaceError = settings.maximumScreenSpaceError;
  globe.tileCacheSize = settings.tileCacheSize;
  globe.loadingDescendantLimit = settings.loadingDescendantLimit;
  globe.preloadAncestors = settings.preloadAncestors;
  globe.preloadSiblings = settings.preloadSiblings;
  viewer.scene.requestRender();
}

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
  const { viewer, layers, airportLocalTerrain } = useApp();
  const initializedRef = useRef(false);
  const worldTerrainProviderRef = useRef<Cesium.TerrainProvider | null>(null);
  const previousStreamingSettingsRef = useRef<GlobeTerrainStreamingSettings | null>(null);
  const airportLocalTerrainOwnsProvider =
    layers.dsmTerrain && airportLocalTerrain?.status === "active";

  useEffect(() => {
    if (!viewer) return;
    if (airportLocalTerrainOwnsProvider) return;

    // Skip the very first run — terrain is already correct from the constructor.
    if (!initializedRef.current) {
      initializedRef.current = true;
      if (layers.terrain) {
        worldTerrainProviderRef.current = viewer.scene.terrainProvider;
        previousStreamingSettingsRef.current = captureTerrainStreamingSettings(
          viewer.scene.globe,
        );
        applyWorldTerrainStreamingSettings(viewer);
        return;
      }
    }

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
