import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportLocalTerrainMetadataUrl } from "../terrain/airportLocalTerrain";
import {
  loadAirportLocalTerrain,
  type AirportLocalTerrain,
  type AirportLocalTerrainMetadata,
} from "../terrain/airportLocalTerrain";
import {
  applyLocalTerrainStreamingSettings,
  airportLocalTerrainProgressState,
  buildLocalTerrainActivationPlan,
  captureTerrainProviderRestorePoint,
  disabledAirportLocalTerrainState,
  installTerrainProvider,
  LOCAL_TERRAIN_SETTINGS,
  missingAirportLocalTerrainState,
  restoreTerrainProvider,
  type TerrainProviderRestorePoint,
} from "../terrain/terrainRuntime";
import { isMissingJsonAsset } from "../utils/fetchJson";

export type AirportLocalTerrainLayerStatus = "idle" | "loading" | "preloading" | "active" | "error";

export interface AirportLocalTerrainLayerState {
  status: AirportLocalTerrainLayerStatus;
  metadata: AirportLocalTerrainMetadata | null;
  /** The loaded terrain provider, or null if not yet loaded / disabled. */
  provider: Cesium.CustomHeightmapTerrainProvider | null;
  loadedTiles: number;
  totalTiles: number;
  error: string | null;
}

export interface UseAirportLocalTerrainLayerOptions {
  enabled?: boolean;
  /**
   * Warm every generated tile after the provider becomes active.
   * Disabled by default because large airports contain thousands of 65 KB
   * height tiles; Cesium can stream non-focused tiles on demand instead.
   */
  backgroundPreload?: boolean;
  metadataUrl?: string;
  maximumScreenSpaceError?: number;
}

/**
 * Load preprocessed airport-local heightmap terrain into the Cesium Viewer.
 *
 * Uses pre-built `.f32` height tiles produced by `npm run build:local-terrain`
 * and served from `public/data/airports/<ICAO>/local-terrain/heightmap/`. The browser fetches only the tiles it needs
 * instead of decoding a full GeoTIFF.
 *
 * TerrainRuntime owns the activation policy: focused warm before provider
 * switch, background warm after switch, and Cesium globe streaming settings.
 *
 * Returns metadata and loading status so callers can display terrain info if desired.
 * On cleanup, restores the previous terrain provider.
 */
export function useAirportLocalTerrainLayer(
  options: UseAirportLocalTerrainLayerOptions = {},
): AirportLocalTerrainLayerState {
  const { viewer, activeAirportCode, setAirportLocalTerrain } = useApp();
  const enabled = options.enabled ?? true;
  const backgroundPreload = options.backgroundPreload ?? false;
  const maximumScreenSpaceError =
    options.maximumScreenSpaceError ?? LOCAL_TERRAIN_SETTINGS.maximumScreenSpaceError;
  const metadataUrl = options.metadataUrl ?? (
    activeAirportCode ? airportLocalTerrainMetadataUrl(activeAirportCode) : null
  );

  const [state, setState] = useState<AirportLocalTerrainLayerState>({
    status: "idle",
    metadata: null,
    provider: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  });

  const providerRef = useRef<Cesium.CustomHeightmapTerrainProvider | null>(null);
  const terrainCacheRef = useRef<Map<string, Promise<AirportLocalTerrain>>>(new Map());
  const restorePointRef = useRef<TerrainProviderRestorePoint | null>(null);

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
      setAirportLocalTerrain(
        enabled
          ? missingAirportLocalTerrainState(activeAirportCode || null)
          : disabledAirportLocalTerrainState(activeAirportCode || null),
      );
      return;
    }

    let cancelled = false;
    const preloadAbortController = new AbortController();

    restorePointRef.current = captureTerrainProviderRestorePoint(viewer);
    setState({
      status: "loading",
      metadata: null,
      provider: null,
      loadedTiles: 0,
      totalTiles: 0,
      error: null,
    });
    setAirportLocalTerrain(airportLocalTerrainProgressState({
      status: "loading",
      airportCode: activeAirportCode,
    }));

    let terrainPromise = terrainCacheRef.current.get(metadataUrl);
    if (!terrainPromise) {
      terrainPromise = loadAirportLocalTerrain(metadataUrl);
      terrainCacheRef.current.set(metadataUrl, terrainPromise);
    }

    terrainPromise
      .then(async (terrain) => {
        if (cancelled || viewer.isDestroyed()) return;
        const { metadata, provider } = terrain;
        const activationPlan = buildLocalTerrainActivationPlan(metadata);

        setState({
          status: "preloading",
          metadata,
          provider: null,
          loadedTiles: 0,
          totalTiles: activationPlan.focusedTiles.length,
          error: null,
        });
        setAirportLocalTerrain(airportLocalTerrainProgressState({
          status: "preloading",
          airportCode: activeAirportCode,
          heightRange: activationPlan.heightRange,
          metadata,
          loadedTiles: 0,
          totalTiles: activationPlan.focusedTiles.length,
        }));

        await terrain.preloadTiles({
          tiles: activationPlan.focusedTiles,
          concurrency: activationPlan.focusedPreloadConcurrency,
          signal: preloadAbortController.signal,
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
            setAirportLocalTerrain(airportLocalTerrainProgressState({
              status: "preloading",
              airportCode: activeAirportCode,
              heightRange: activationPlan.heightRange,
              metadata,
              loadedTiles,
              totalTiles,
            }));
          },
        });

        if (cancelled || viewer.isDestroyed()) return;

        providerRef.current = provider;
        installTerrainProvider(viewer, provider);
        applyLocalTerrainStreamingSettings(viewer, metadata, maximumScreenSpaceError);
        const activeLoadedTiles = activationPlan.focusedTiles.length;
        const activeTotalTiles = backgroundPreload
          ? activationPlan.activeTotalTiles
          : activeLoadedTiles;

        setState({
          status: "active",
          metadata,
          provider,
          loadedTiles: activeLoadedTiles,
          totalTiles: activeTotalTiles,
          error: null,
        });
        setAirportLocalTerrain(airportLocalTerrainProgressState({
          status: "active",
          airportCode: activeAirportCode,
          heightRange: activationPlan.heightRange,
          metadata,
          loadedTiles: activeLoadedTiles,
          totalTiles: activeTotalTiles,
        }));

        if (backgroundPreload) {
          void terrain.preloadTiles({
            concurrency: activationPlan.backgroundPreloadConcurrency,
            signal: preloadAbortController.signal,
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
              setAirportLocalTerrain(airportLocalTerrainProgressState({
                status: "active",
                airportCode: activeAirportCode,
                heightRange: activationPlan.heightRange,
                metadata,
                loadedTiles,
                totalTiles,
              }));
            },
          }).catch((error) => {
            if (cancelled) return;
            console.error("[useAirportLocalTerrainLayer] Failed to warm local terrain cache:", error);
          });
        }
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
          setAirportLocalTerrain(missingAirportLocalTerrainState(activeAirportCode));
          return;
        }

        const message =
          error instanceof Error ? error.message : String(error);
        console.error("[useAirportLocalTerrainLayer] Failed to load local terrain:", error);
        terrainCacheRef.current.delete(metadataUrl);
        setState({
          status: "error",
          metadata: null,
          provider: null,
          loadedTiles: 0,
          totalTiles: 0,
          error: message,
        });
        setAirportLocalTerrain(airportLocalTerrainProgressState({
          status: "error",
          airportCode: activeAirportCode,
          error: message,
        }));
      });

    return () => {
      cancelled = true;
      preloadAbortController.abort();
      if (!viewer.isDestroyed() && providerRef.current) {
        if (viewer.scene.terrainProvider === providerRef.current) {
          restoreTerrainProvider(viewer, restorePointRef.current);
        }
      }
      providerRef.current = null;
      restorePointRef.current = null;
    };
  }, [
    viewer,
    enabled,
    backgroundPreload,
    metadataUrl,
    maximumScreenSpaceError,
    activeAirportCode,
    setAirportLocalTerrain,
  ]);

  return state;
}
