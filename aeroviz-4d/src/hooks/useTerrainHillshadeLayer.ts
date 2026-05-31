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
        const plan = airportLocalTerrainImageryPlan(metadata, "hillshade");
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
        // Hillshade is a perceptual relief cue: tone controls make low-relief
        // airport terrain read as 3D without changing the terrain heights.
        layer.brightness = plan.brightness ?? layer.brightness;
        layer.contrast = plan.contrast ?? layer.contrast;
        layer.gamma = plan.gamma ?? layer.gamma;
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
