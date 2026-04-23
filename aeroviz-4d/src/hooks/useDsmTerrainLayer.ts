import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { dsmHeightmapTerrainMetadataUrl } from "../terrain/dsmHeightmapTerrain";
import {
  loadDsmHeightmapTerrain,
  type DsmHeightmapTerrainMetadata,
} from "../terrain/dsmHeightmapTerrain";
import { isMissingJsonAsset } from "../utils/fetchJson";

export type DsmTerrainStatus = "idle" | "loading" | "active" | "error";

const DSM_MAXIMUM_SCREEN_SPACE_ERROR = 0.5;
const DSM_MIN_TILE_CACHE_SIZE = 256;

export interface DsmTerrainState {
  status: DsmTerrainStatus;
  metadata: DsmHeightmapTerrainMetadata | null;
  /** The loaded terrain provider, or null if not yet loaded / disabled. */
  provider: Cesium.CustomHeightmapTerrainProvider | null;
  error: string | null;
}

export interface UseDsmTerrainLayerOptions {
  enabled?: boolean;
  metadataUrl?: string;
  maximumScreenSpaceError?: number;
}

/**
 * Load preprocessed DSM heightmap terrain into the Cesium Viewer.
 *
 * Uses pre-built `.f32` height tiles produced by `npm run build:dsm-heightmap-terrain`
 * and served from `public/data/airports/<ICAO>/dsm/heightmap-terrain/`. The browser fetches only the tiles it needs
 * instead of decoding a full GeoTIFF.
 *
 * Returns metadata and loading status so callers can display terrain info if desired.
 * On cleanup, restores the previous terrain provider.
 */
export function useDsmTerrainLayer(
  options: UseDsmTerrainLayerOptions = {},
): DsmTerrainState {
  const { viewer, activeAirportCode } = useApp();
  const enabled = options.enabled ?? true;
  const maximumScreenSpaceError =
    options.maximumScreenSpaceError ?? DSM_MAXIMUM_SCREEN_SPACE_ERROR;
  const metadataUrl = options.metadataUrl ?? (
    activeAirportCode ? dsmHeightmapTerrainMetadataUrl(activeAirportCode) : null
  );

  const [state, setState] = useState<DsmTerrainState>({
    status: "idle",
    metadata: null,
    provider: null,
    error: null,
  });

  const providerRef = useRef<Cesium.CustomHeightmapTerrainProvider | null>(null);
  const previousProviderRef = useRef<Cesium.TerrainProvider | null>(null);
  const previousMaximumScreenSpaceErrorRef = useRef<number | null>(null);
  const previousTileCacheSizeRef = useRef<number | null>(null);
  const previousPreloadSiblingsRef = useRef<boolean | null>(null);
  const previousPreloadAncestorsRef = useRef<boolean | null>(null);

  // ── Load terrain provider ───────────────────────────────────────────────
  useEffect(() => {
    if (!viewer || !enabled || !metadataUrl) {
      setState({ status: "idle", metadata: null, provider: null, error: null });
      return;
    }

    let cancelled = false;

    previousProviderRef.current = viewer.scene.terrainProvider;
    previousMaximumScreenSpaceErrorRef.current =
      viewer.scene.globe.maximumScreenSpaceError;
    previousTileCacheSizeRef.current = viewer.scene.globe.tileCacheSize;
    previousPreloadSiblingsRef.current = viewer.scene.globe.preloadSiblings;
    previousPreloadAncestorsRef.current = viewer.scene.globe.preloadAncestors;
    setState({ status: "loading", metadata: null, provider: null, error: null });

    loadDsmHeightmapTerrain(metadataUrl)
      .then(({ metadata, provider }) => {
        if (cancelled || viewer.isDestroyed()) return;

        providerRef.current = provider;
        viewer.scene.terrainProvider = provider;
        viewer.scene.globe.maximumScreenSpaceError = maximumScreenSpaceError;
        viewer.scene.globe.tileCacheSize = Math.max(
          viewer.scene.globe.tileCacheSize,
          metadata.tileCount + 32,
          DSM_MIN_TILE_CACHE_SIZE,
        );
        viewer.scene.globe.preloadSiblings = true;
        viewer.scene.globe.preloadAncestors = true;
        viewer.scene.globe.depthTestAgainstTerrain = true;
        viewer.scene.requestRender();

        setState({ status: "active", metadata, provider, error: null });
      })
      .catch((error) => {
        if (cancelled) return;
        if (isMissingJsonAsset(error)) {
          console.warn(`[useDsmTerrainLayer] ${metadataUrl} not found.`);
          setState({ status: "idle", metadata: null, provider: null, error: null });
          return;
        }

        const message =
          error instanceof Error ? error.message : String(error);
        console.error("[useDsmTerrainLayer] Failed to load DSM terrain:", error);
        setState({ status: "error", metadata: null, provider: null, error: message });
      });

    return () => {
      cancelled = true;
      if (!viewer.isDestroyed() && providerRef.current) {
        if (viewer.scene.terrainProvider === providerRef.current) {
          viewer.scene.terrainProvider =
            previousProviderRef.current ?? new Cesium.EllipsoidTerrainProvider();
        }
        viewer.scene.globe.maximumScreenSpaceError =
          previousMaximumScreenSpaceErrorRef.current ?? 2;
        viewer.scene.globe.tileCacheSize =
          previousTileCacheSizeRef.current ?? 100;
        viewer.scene.globe.preloadSiblings =
          previousPreloadSiblingsRef.current ?? false;
        viewer.scene.globe.preloadAncestors =
          previousPreloadAncestorsRef.current ?? false;
      }
      providerRef.current = null;
      previousProviderRef.current = null;
      previousMaximumScreenSpaceErrorRef.current = null;
      previousTileCacheSizeRef.current = null;
      previousPreloadSiblingsRef.current = null;
      previousPreloadAncestorsRef.current = null;
    };
  }, [viewer, enabled, metadataUrl, maximumScreenSpaceError]);

  return state;
}
