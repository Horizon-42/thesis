import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  airportLocalTerrainMetadataUrl,
  type AirportLocalTerrainMetadata,
} from "../terrain/airportLocalTerrain";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

const DEFAULT_HEIGHT_TINT_ALPHA = 0.38;
const DEFAULT_HEIGHT_TINT_BRIGHTNESS = 1.05;
const DEFAULT_HEIGHT_TINT_CONTRAST = 1.12;
const DEFAULT_HEIGHT_TINT_SATURATION = 0.9;

function boundedAlpha(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return DEFAULT_HEIGHT_TINT_ALPHA;
  return Math.max(0, Math.min(1, value));
}

function removeLayer(viewer: Cesium.Viewer, layer: Cesium.ImageryLayer | null): void {
  if (!layer || viewer.isDestroyed()) return;
  viewer.imageryLayers.remove(layer, true);
  viewer.scene.requestRender();
}

/**
 * Drapes the generated local-terrain height-tint PNG over the active airport.
 *
 * This is intentionally imagery-only: it helps flat-looking tiles expose subtle
 * local height differences without changing DEM/DSM geometry or exaggeration.
 */
export function useTerrainHeightTintLayer(): void {
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

    if (!layers.terrainHeightTint || !activeAirportCode) {
      return () => {
        cancelled = true;
      };
    }

    const metadataUrl = airportLocalTerrainMetadataUrl(activeAirportCode);
    void fetchJson<AirportLocalTerrainMetadata>(metadataUrl)
      .then((metadata) => {
        if (cancelled || viewer.isDestroyed()) return;
        if (!metadata.overlay?.url) return;

        const layer = viewer.imageryLayers.addImageryProvider(
          new Cesium.SingleTileImageryProvider({
            url: metadata.overlay.url,
            tileWidth: metadata.overlay.width,
            tileHeight: metadata.overlay.height,
            rectangle: Cesium.Rectangle.fromDegrees(
              metadata.bounds.west,
              metadata.bounds.south,
              metadata.bounds.east,
              metadata.bounds.north,
            ),
            credit: "Airport terrain height tint",
          }),
        );
        layer.alpha = boundedAlpha(metadata.overlay.alpha);
        layer.brightness = DEFAULT_HEIGHT_TINT_BRIGHTNESS;
        layer.contrast = DEFAULT_HEIGHT_TINT_CONTRAST;
        layer.saturation = DEFAULT_HEIGHT_TINT_SATURATION;
        layerRef.current = layer;
        viewer.scene.requestRender();
      })
      .catch((error) => {
        if (cancelled || isMissingJsonAsset(error)) return;
        console.error("[useTerrainHeightTintLayer] Failed to load height tint metadata:", error);
      });

    return () => {
      cancelled = true;
      clearCurrentLayer();
    };
  }, [viewer, activeAirportCode, layers.terrainHeightTint]);
}
