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
  captureTerrainStreamingSettings,
  disabledAirportLocalTerrainState,
  LOCAL_TERRAIN_SETTINGS,
  missingAirportLocalTerrainState,
  restoreTerrainStreamingSettings,
  type GlobeTerrainStreamingSettings,
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
  metadataUrl?: string;
  maximumScreenSpaceError?: number;
}

/**
 * Load preprocessed airport-local heightmap terrain into the Cesium Viewer.
 *
 * Uses pre-built `.f32` height tiles produced by `npm run build:local-terrain`
 * and served from `public/data/airports/<ICAO>/dsm/heightmap-terrain/`. The browser fetches only the tiles it needs
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
  const previousProviderRef = useRef<Cesium.TerrainProvider | null>(null);
  const previousStreamingSettingsRef = useRef<GlobeTerrainStreamingSettings | null>(null);

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

    previousProviderRef.current = viewer.scene.terrainProvider;
    previousStreamingSettingsRef.current = captureTerrainStreamingSettings(viewer.scene.globe);
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
          loadedTiles: 0,
          totalTiles: activationPlan.focusedTiles.length,
        }));

        await terrain.preloadTiles({
          tiles: activationPlan.focusedTiles,
          concurrency: activationPlan.focusedPreloadConcurrency,
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
              loadedTiles,
              totalTiles,
            }));
          },
        });

        if (cancelled || viewer.isDestroyed()) return;

        providerRef.current = provider;
        viewer.scene.terrainProvider = provider;
        applyLocalTerrainStreamingSettings(viewer, metadata, maximumScreenSpaceError);

        setState({
          status: "active",
          metadata,
          provider,
          loadedTiles: activationPlan.focusedTiles.length,
          totalTiles: activationPlan.activeTotalTiles,
          error: null,
        });
        setAirportLocalTerrain(airportLocalTerrainProgressState({
          status: "active",
          airportCode: activeAirportCode,
          heightRange: activationPlan.heightRange,
          loadedTiles: activationPlan.focusedTiles.length,
          totalTiles: activationPlan.activeTotalTiles,
        }));

        void terrain.preloadTiles({
          concurrency: activationPlan.backgroundPreloadConcurrency,
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
              loadedTiles,
              totalTiles,
            }));
          },
        }).catch((error) => {
          if (cancelled) return;
          console.error("[useAirportLocalTerrainLayer] Failed to warm local terrain cache:", error);
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
      if (!viewer.isDestroyed() && providerRef.current) {
        if (viewer.scene.terrainProvider === providerRef.current) {
          viewer.scene.terrainProvider =
            previousProviderRef.current ?? new Cesium.EllipsoidTerrainProvider();
        }
        if (previousStreamingSettingsRef.current) {
          restoreTerrainStreamingSettings(viewer, previousStreamingSettingsRef.current);
        }
      }
      providerRef.current = null;
      previousProviderRef.current = null;
      previousStreamingSettingsRef.current = null;
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
