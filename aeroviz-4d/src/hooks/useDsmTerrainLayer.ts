import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { dsmHeightmapTerrainMetadataUrl } from "../terrain/dsmHeightmapTerrain";
import {
  dsmTerrainTileRefsNearCoordinate,
  loadDsmHeightmapTerrain,
  type DsmHeightmapTerrain,
  type DsmHeightmapTerrainMetadata,
} from "../terrain/dsmHeightmapTerrain";
import { isMissingJsonAsset } from "../utils/fetchJson";

export type DsmTerrainStatus = "idle" | "loading" | "preloading" | "active" | "error";

const DSM_MAXIMUM_SCREEN_SPACE_ERROR = 0.5;
const DSM_MIN_TILE_CACHE_SIZE = 256;
const DSM_FOCUS_PRELOAD_CONCURRENCY = 24;
const DSM_BACKGROUND_PRELOAD_CONCURRENCY = 8;
const DSM_FOCUS_MAX_LEVEL_TILE_RADIUS = 4;

export interface DsmTerrainState {
  status: DsmTerrainStatus;
  metadata: DsmHeightmapTerrainMetadata | null;
  /** The loaded terrain provider, or null if not yet loaded / disabled. */
  provider: Cesium.CustomHeightmapTerrainProvider | null;
  loadedTiles: number;
  totalTiles: number;
  error: string | null;
}

export interface UseDsmTerrainLayerOptions {
  enabled?: boolean;
  metadataUrl?: string;
  maximumScreenSpaceError?: number;
}

function terrainHeightRange(metadata: DsmHeightmapTerrainMetadata): {
  minimumHeightM: number;
  maximumHeightM: number;
} {
  return {
    minimumHeightM: metadata.stats.min,
    maximumHeightM: metadata.stats.max,
  };
}

function terrainFocusCoordinate(metadata: DsmHeightmapTerrainMetadata): {
  lon: number;
  lat: number;
} {
  return {
    lon: (metadata.bounds.west + metadata.bounds.east) / 2,
    lat: (metadata.bounds.south + metadata.bounds.north) / 2,
  };
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
  const { viewer, activeAirportCode, setAirportLocalTerrain } = useApp();
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
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  });

  const providerRef = useRef<Cesium.CustomHeightmapTerrainProvider | null>(null);
  const terrainCacheRef = useRef<Map<string, Promise<DsmHeightmapTerrain>>>(new Map());
  const previousProviderRef = useRef<Cesium.TerrainProvider | null>(null);
  const previousMaximumScreenSpaceErrorRef = useRef<number | null>(null);
  const previousTileCacheSizeRef = useRef<number | null>(null);
  const previousPreloadSiblingsRef = useRef<boolean | null>(null);
  const previousPreloadAncestorsRef = useRef<boolean | null>(null);
  const previousLoadingDescendantLimitRef = useRef<number | null>(null);

  // ── Load terrain provider ───────────────────────────────────────────────
  useEffect(() => {
    if (!viewer || !enabled || !metadataUrl) {
      setState({
        status: "idle",
        metadata: null,
        provider: null,
        loadedTiles: 0,
        totalTiles: 0,
        error: null,
      });
      setAirportLocalTerrain({
        status: enabled ? "missing" : "disabled",
        airportCode: activeAirportCode || null,
        sourceLabel: null,
        minimumHeightM: null,
        maximumHeightM: null,
        loadedTiles: 0,
        totalTiles: 0,
        error: null,
      });
      return;
    }

    let cancelled = false;

    previousProviderRef.current = viewer.scene.terrainProvider;
    previousMaximumScreenSpaceErrorRef.current =
      viewer.scene.globe.maximumScreenSpaceError;
    previousTileCacheSizeRef.current = viewer.scene.globe.tileCacheSize;
    previousPreloadSiblingsRef.current = viewer.scene.globe.preloadSiblings;
    previousPreloadAncestorsRef.current = viewer.scene.globe.preloadAncestors;
    previousLoadingDescendantLimitRef.current =
      viewer.scene.globe.loadingDescendantLimit;
    setState({
      status: "loading",
      metadata: null,
      provider: null,
      loadedTiles: 0,
      totalTiles: 0,
      error: null,
    });
    setAirportLocalTerrain({
      status: "loading",
      airportCode: activeAirportCode,
      sourceLabel: "Airport local DSM heightmap",
      minimumHeightM: null,
      maximumHeightM: null,
      loadedTiles: 0,
      totalTiles: 0,
      error: null,
    });

    let terrainPromise = terrainCacheRef.current.get(metadataUrl);
    if (!terrainPromise) {
      terrainPromise = loadDsmHeightmapTerrain(metadataUrl);
      terrainCacheRef.current.set(metadataUrl, terrainPromise);
    }

    terrainPromise
      .then(async (terrain) => {
        if (cancelled || viewer.isDestroyed()) return;
        const { metadata, provider } = terrain;
        const heightRange = terrainHeightRange(metadata);
        const focus = terrainFocusCoordinate(metadata);
        const focusedTiles = dsmTerrainTileRefsNearCoordinate(
          metadata,
          focus.lon,
          focus.lat,
          {
            maxLevelRadius: DSM_FOCUS_MAX_LEVEL_TILE_RADIUS,
            ancestorRadius: 0,
          },
        );

        setState({
          status: "preloading",
          metadata,
          provider: null,
          loadedTiles: 0,
          totalTiles: focusedTiles.length,
          error: null,
        });
        setAirportLocalTerrain({
          status: "preloading",
          airportCode: activeAirportCode,
          sourceLabel: "Airport local DSM heightmap",
          ...heightRange,
          loadedTiles: 0,
          totalTiles: focusedTiles.length,
          error: null,
        });

        await terrain.preloadTiles({
          tiles: focusedTiles,
          concurrency: DSM_FOCUS_PRELOAD_CONCURRENCY,
          onProgress: ({ loadedTiles, totalTiles }) => {
            if (cancelled) return;
            setState({
              status: "preloading",
              metadata,
              provider: null,
              loadedTiles,
              totalTiles,
              error: null,
            });
            setAirportLocalTerrain({
              status: "preloading",
              airportCode: activeAirportCode,
              sourceLabel: "Airport local DSM heightmap",
              ...heightRange,
              loadedTiles,
              totalTiles,
              error: null,
            });
          },
        });

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
        viewer.scene.globe.loadingDescendantLimit = 1000;
        viewer.scene.globe.depthTestAgainstTerrain = true;
        viewer.scene.requestRender();

        setState({
          status: "active",
          metadata,
          provider,
          loadedTiles: focusedTiles.length,
          totalTiles: metadata.tileCount,
          error: null,
        });
        setAirportLocalTerrain({
          status: "active",
          airportCode: activeAirportCode,
          sourceLabel: "Airport local DSM heightmap",
          ...heightRange,
          loadedTiles: focusedTiles.length,
          totalTiles: metadata.tileCount,
          error: null,
        });

        void terrain.preloadTiles({
          concurrency: DSM_BACKGROUND_PRELOAD_CONCURRENCY,
          onProgress: ({ loadedTiles, totalTiles }) => {
            if (cancelled) return;
            setState((current) => {
              if (current.status !== "active") return current;
              return {
                ...current,
                loadedTiles,
                totalTiles,
              };
            });
            setAirportLocalTerrain({
              status: "active",
              airportCode: activeAirportCode,
              sourceLabel: "Airport local DSM heightmap",
              ...heightRange,
              loadedTiles,
              totalTiles,
              error: null,
            });
          },
        }).catch((error) => {
          if (cancelled) return;
          console.error("[useDsmTerrainLayer] Failed to warm DSM terrain cache:", error);
        });
      })
      .catch((error) => {
        if (cancelled) return;
        if (isMissingJsonAsset(error)) {
          terrainCacheRef.current.delete(metadataUrl);
          setState({
            status: "idle",
            metadata: null,
            provider: null,
            loadedTiles: 0,
            totalTiles: 0,
            error: null,
          });
          setAirportLocalTerrain({
            status: "missing",
            airportCode: activeAirportCode,
            sourceLabel: null,
            minimumHeightM: null,
            maximumHeightM: null,
            loadedTiles: 0,
            totalTiles: 0,
            error: null,
          });
          return;
        }

        const message =
          error instanceof Error ? error.message : String(error);
        console.error("[useDsmTerrainLayer] Failed to load DSM terrain:", error);
        terrainCacheRef.current.delete(metadataUrl);
        setState({
          status: "error",
          metadata: null,
          provider: null,
          loadedTiles: 0,
          totalTiles: 0,
          error: message,
        });
        setAirportLocalTerrain({
          status: "error",
          airportCode: activeAirportCode,
          sourceLabel: "Airport local DSM heightmap",
          minimumHeightM: null,
          maximumHeightM: null,
          loadedTiles: 0,
          totalTiles: 0,
          error: message,
        });
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
        viewer.scene.globe.loadingDescendantLimit =
          previousLoadingDescendantLimitRef.current ?? 10;
      }
      providerRef.current = null;
      previousProviderRef.current = null;
      previousMaximumScreenSpaceErrorRef.current = null;
      previousTileCacheSizeRef.current = null;
      previousPreloadSiblingsRef.current = null;
      previousPreloadAncestorsRef.current = null;
      previousLoadingDescendantLimitRef.current = null;
    };
  }, [
    viewer,
    enabled,
    metadataUrl,
    maximumScreenSpaceError,
    activeAirportCode,
    setAirportLocalTerrain,
  ]);

  return state;
}
