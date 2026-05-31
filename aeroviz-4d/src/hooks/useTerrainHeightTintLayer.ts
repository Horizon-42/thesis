import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import {
  airportLocalTerrainImageryPlan,
  airportLocalTerrainMetadataUrl,
  type AirportLocalTerrainMetadata,
} from "../terrain/airportLocalTerrain";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

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
        const plan = airportLocalTerrainImageryPlan(metadata, "heightTint");
        if (!plan) return;

        const layer = viewer.imageryLayers.addImageryProvider(
          new Cesium.SingleTileImageryProvider({
            url: plan.url,
            tileWidth: plan.tileWidth,
            tileHeight: plan.tileHeight,
            rectangle: plan.rectangle,
            credit: plan.credit,
          }),
        );
        layer.alpha = plan.alpha;
        layer.brightness = plan.brightness ?? layer.brightness;
        layer.contrast = plan.contrast ?? layer.contrast;
        layer.saturation = plan.saturation ?? layer.saturation;
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
