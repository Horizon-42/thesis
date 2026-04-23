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

  // ── Load terrain provider ───────────────────────────────────────────────
  useEffect(() => {
    if (!viewer || !enabled || !metadataUrl) {
      setState({ status: "idle", metadata: null, provider: null, error: null });
      return;
    }

    let cancelled = false;

    previousProviderRef.current = viewer.scene.terrainProvider;
    setState({ status: "loading", metadata: null, provider: null, error: null });

    loadDsmHeightmapTerrain(metadataUrl)
      .then(({ metadata, provider }) => {
        if (cancelled || viewer.isDestroyed()) return;

        providerRef.current = provider;
        viewer.scene.terrainProvider = provider;
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
      if (
        !viewer.isDestroyed() &&
        providerRef.current &&
        viewer.scene.terrainProvider === providerRef.current
      ) {
        viewer.scene.terrainProvider =
          previousProviderRef.current ?? new Cesium.EllipsoidTerrainProvider();
      }
      providerRef.current = null;
      previousProviderRef.current = null;
    };
  }, [viewer, enabled, metadataUrl]);

  return state;
}
