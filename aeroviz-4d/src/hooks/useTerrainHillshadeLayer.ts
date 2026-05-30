import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  airportLocalTerrainMetadataUrl,
  type AirportLocalTerrainMetadata,
} from "../terrain/airportLocalTerrain";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

const DEFAULT_HILLSHADE_ALPHA = 0.34;

function boundedAlpha(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return DEFAULT_HILLSHADE_ALPHA;
  return Math.max(0, Math.min(1, value));
}

function removeLayer(viewer: Cesium.Viewer, layer: Cesium.ImageryLayer | null): void {
  if (!layer || viewer.isDestroyed()) return;
  viewer.imageryLayers.remove(layer, true);
  viewer.scene.requestRender();
}

/**
 * Adds the generated airport-local hillshade PNG as a bounded imagery overlay.
 *
 * The layer is intentionally imagery-only: terrain heights and vertical
 * exaggeration remain controlled by the existing terrain provider and HUD.
 */
export function useTerrainHillshadeLayer(): void {
  const { viewer, layers, activeAirportCode } = useApp();
  const layerRef = useRef<Cesium.ImageryLayer | null>(null);

  useEffect(() => {
    if (!viewer || viewer.isDestroyed()) return;

    let cancelled = false;
    const clearCurrentLayer = () => {
      removeLayer(viewer, layerRef.current);
      layerRef.current = null;
    };

    clearCurrentLayer();

    if (!layers.terrainHillshade || !activeAirportCode) {
      return () => {
        cancelled = true;
      };
    }

    const metadataUrl = airportLocalTerrainMetadataUrl(activeAirportCode);
    void fetchJson<AirportLocalTerrainMetadata>(metadataUrl)
      .then((metadata) => {
        if (cancelled || viewer.isDestroyed()) return;
        if (!metadata.hillshade?.url) return;

        const layer = viewer.imageryLayers.addImageryProvider(
          new Cesium.SingleTileImageryProvider({
            url: metadata.hillshade.url,
            tileWidth: metadata.hillshade.width,
            tileHeight: metadata.hillshade.height,
            rectangle: Cesium.Rectangle.fromDegrees(
              metadata.bounds.west,
              metadata.bounds.south,
              metadata.bounds.east,
              metadata.bounds.north,
            ),
            credit: "Airport terrain hillshade",
          }),
        );
        layer.alpha = boundedAlpha(metadata.hillshade.alpha);
        layerRef.current = layer;
        viewer.scene.requestRender();
      })
      .catch((error) => {
        if (cancelled || isMissingJsonAsset(error)) return;
        console.error("[useTerrainHillshadeLayer] Failed to load hillshade metadata:", error);
      });

    return () => {
      cancelled = true;
      clearCurrentLayer();
    };
  }, [viewer, activeAirportCode, layers.terrainHillshade]);
}
