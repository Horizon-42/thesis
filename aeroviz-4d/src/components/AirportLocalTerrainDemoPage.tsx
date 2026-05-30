import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import AirportLocalTerrainAlert from "./AirportLocalTerrainAlert";
import HUD from "./HUD";
import { useAirportLocalTerrainLayer } from "../hooks/useAirportLocalTerrainLayer";
import type { AirportLocalTerrainMetadata } from "../terrain/airportLocalTerrain";

const TERRAIN_VERTICAL_EXAGGERATION = 25;
const SATELLITE_IMAGERY_URL =
  "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

const DEFAULT_LAYER_STATE = {
  terrain: true,
  satellite: true,
  originalTifHeatmap: true,
  terrainTint: false,
};

interface LocalTerrainDemoState {
  status: string;
  rasterSize: string;
  sourceTiles: string;
  tileSize: string;
  levels: string;
  tiles: string;
  heights: string;
  heightMin: string;
  heightMax: string;
  center: string;
  source: string;
}

type LayerToggleKey = keyof typeof DEFAULT_LAYER_STATE;
type LayerToggleState = typeof DEFAULT_LAYER_STATE;

function centerOfBounds(metadata: AirportLocalTerrainMetadata): { lon: number; lat: number } {
  return {
    lon: (metadata.bounds.west + metadata.bounds.east) / 2,
    lat: (metadata.bounds.south + metadata.bounds.north) / 2,
  };
}

function rectangleFromBounds(metadata: AirportLocalTerrainMetadata): Cesium.Rectangle {
  return Cesium.Rectangle.fromDegrees(
    metadata.bounds.west,
    metadata.bounds.south,
    metadata.bounds.east,
    metadata.bounds.north
  );
}

function describeMetadata(metadata: AirportLocalTerrainMetadata): LocalTerrainDemoState {
  const center = centerOfBounds(metadata);
  const heightMin = `${metadata.stats.min.toFixed(2)} m`;
  const heightMax = `${metadata.stats.max.toFixed(2)} m`;

  return {
    status: "Airport local heightmap terrain active",
    rasterSize: `${metadata.raster.width} x ${metadata.raster.height}`,
    sourceTiles: metadata.raster.sourceTileCount?.toLocaleString() ?? "1",
    tileSize: `${metadata.tileWidth} x ${metadata.tileHeight}`,
    levels: `${metadata.minLevel}-${metadata.maxLevel}`,
    tiles: metadata.tileCount.toLocaleString(),
    heights: `${heightMin}-${heightMax}`,
    heightMin,
    heightMax,
    center: `${center.lat.toFixed(6)}, ${center.lon.toFixed(6)}`,
    source: "CustomHeightmapTerrainProvider",
  };
}

function addSatelliteImagery(viewer: Cesium.Viewer): Cesium.ImageryLayer {
  const provider = new Cesium.UrlTemplateImageryProvider({
    url: SATELLITE_IMAGERY_URL,
    credit: "Esri World Imagery",
    maximumLevel: 19,
  });
  const layer = viewer.imageryLayers.addImageryProvider(provider, 0);
  layer.brightness = 1.03;
  layer.contrast = 1.08;
  layer.saturation = 0.95;
  return layer;
}

async function addSingleTileLayer(
  viewer: Cesium.Viewer,
  url: string,
  rectangle: Cesium.Rectangle,
  style: Partial<Pick<Cesium.ImageryLayer, "alpha" | "brightness" | "contrast" | "saturation">>
): Promise<Cesium.ImageryLayer> {
  const provider = await Cesium.SingleTileImageryProvider.fromUrl(url, {
    rectangle,
  });
  const layer = viewer.imageryLayers.addImageryProvider(provider);
  if (style.alpha !== undefined) layer.alpha = style.alpha;
  if (style.brightness !== undefined) layer.brightness = style.brightness;
  if (style.contrast !== undefined) layer.contrast = style.contrast;
  if (style.saturation !== undefined) layer.saturation = style.saturation;
  return layer;
}

async function addLocalTerrainHeightTint(
  viewer: Cesium.Viewer,
  metadata: AirportLocalTerrainMetadata
): Promise<Cesium.ImageryLayer | undefined> {
  if (!metadata.overlay?.url) return undefined;

  return addSingleTileLayer(viewer, metadata.overlay.url, rectangleFromBounds(metadata), {
    alpha: 0.44,
    brightness: 1.08,
    contrast: 1.12,
  });
}

async function addOriginalTifHeatmap(
  viewer: Cesium.Viewer,
  metadata: AirportLocalTerrainMetadata
): Promise<Cesium.ImageryLayer | undefined> {
  if (!metadata.originalTifHeatmap?.url) return undefined;

  const bounds = metadata.originalTifHeatmap.bounds ?? metadata.bounds;
  return addSingleTileLayer(
    viewer,
    metadata.originalTifHeatmap.url,
    Cesium.Rectangle.fromDegrees(bounds.west, bounds.south, bounds.east, bounds.north),
    {
      alpha: 0.62,
      brightness: 1.04,
      contrast: 1.08,
    },
  );
}

export default function AirportLocalTerrainDemoPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const flatTerrainProviderRef = useRef<Cesium.EllipsoidTerrainProvider | null>(null);
  const satelliteLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const terrainTintLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const originalTifHeatmapLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const layerStateRef = useRef<LayerToggleState>(DEFAULT_LAYER_STATE);
  const { setAirport, setViewer, activeAirportCode } = useApp();

  const terrain = useAirportLocalTerrainLayer();

  const [displayState, setDisplayState] = useState<LocalTerrainDemoState>({
    status: "Loading preprocessed airport local terrain",
    rasterSize: "",
    sourceTiles: "",
    tileSize: "",
    levels: "",
    tiles: "",
    heights: "",
    heightMin: "",
    heightMax: "",
    center: "",
    source: "Heightmap terrain",
  });
  const [layers, setLayers] = useState<LayerToggleState>(DEFAULT_LAYER_STATE);

  function setLayerEnabled(layer: LayerToggleKey, enabled: boolean) {
    setLayers((current) => {
      const next = { ...current, [layer]: enabled };
      layerStateRef.current = next;
      return next;
    });
  }

  // ── Sync layer visibility toggles ─────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    if (satelliteLayerRef.current) satelliteLayerRef.current.show = layers.satellite;
    if (terrainTintLayerRef.current) terrainTintLayerRef.current.show = layers.terrainTint;
    if (originalTifHeatmapLayerRef.current) {
      originalTifHeatmapLayerRef.current.show = layers.originalTifHeatmap;
    }

    const terrainProvider = layers.terrain
      ? terrain.provider
      : flatTerrainProviderRef.current;
    if (terrainProvider) {
      viewer.scene.terrainProvider = terrainProvider;
      viewer.scene.globe.depthTestAgainstTerrain = layers.terrain;
    }

    viewer.scene.requestRender();
  }, [layers, terrain.provider]);

  // ── Create Viewer ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;

    const viewer = new Cesium.Viewer(containerRef.current, {
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      infoBox: false,
      selectionIndicator: false,
      fullscreenButton: false,
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    });

    viewerRef.current = viewer;
    setViewer(viewer);
    flatTerrainProviderRef.current = new Cesium.EllipsoidTerrainProvider();
    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#08110f");
    viewer.scene.highDynamicRange = false;
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#1f4f46");
    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;
    viewer.scene.verticalExaggeration = TERRAIN_VERTICAL_EXAGGERATION;
    viewer.scene.verticalExaggerationRelativeHeight = 0;

    const satelliteLayer = addSatelliteImagery(viewer);
    satelliteLayer.show = layerStateRef.current.satellite;
    satelliteLayerRef.current = satelliteLayer;

    return () => {
      satelliteLayerRef.current = null;
      terrainTintLayerRef.current = null;
      originalTifHeatmapLayerRef.current = null;
      flatTerrainProviderRef.current = null;
      viewerRef.current = null;
      viewer.destroy();
    };
  }, [setViewer]);

  // ── Set up overlays and camera once terrain metadata is available ─────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed() || !terrain.metadata) return;

    let cancelled = false;
    const metadata = terrain.metadata;

    (async () => {
      const originalTifHeatmapLayer = await addOriginalTifHeatmap(viewer, metadata);
      if (cancelled || viewer.isDestroyed()) return;
      if (originalTifHeatmapLayer) {
        originalTifHeatmapLayer.show = layerStateRef.current.originalTifHeatmap;
        originalTifHeatmapLayerRef.current = originalTifHeatmapLayer;
      }

      const terrainTintLayer = await addLocalTerrainHeightTint(viewer, metadata);
      if (cancelled || viewer.isDestroyed()) return;
      if (terrainTintLayer) {
        terrainTintLayer.show = layerStateRef.current.terrainTint;
        terrainTintLayerRef.current = terrainTintLayer;
      }

      const center = centerOfBounds(metadata);
      setAirport({
        code: activeAirportCode || "LOCAL",
        lon: center.lon,
        lat: center.lat,
        height: 4300,
      });
      const focus = Cesium.Cartesian3.fromDegrees(
        center.lon,
        center.lat,
        Math.max(120, metadata.stats.max * TERRAIN_VERTICAL_EXAGGERATION)
      );
      viewer.camera.lookAt(
        focus,
        new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(-38),
          Cesium.Math.toRadians(-62),
          4300
        )
      );
      viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
      viewer.scene.requestRender();
      setDisplayState(describeMetadata(metadata));
    })();

    return () => {
      cancelled = true;
    };
  }, [terrain.metadata, setAirport, activeAirportCode]);

  // ── Update display when terrain status changes ────────────────────────────
  useEffect(() => {
    if (terrain.status === "error") {
      setDisplayState((current) => ({
        ...current,
        status: "Airport local terrain failed to load",
        source: "Failed",
      }));
    }
  }, [terrain.status]);

  return (
    <main className="local-terrain-page">
      <div ref={containerRef} className="local-terrain-viewer" />
      <div className="cesium-overlay-container local-terrain-hud-layer">
        <AirportLocalTerrainAlert />
        <HUD />
      </div>
      <section className="local-terrain-panel">
        <nav className="local-terrain-nav">
          <a href="/" className="local-terrain-link">Flight view</a>
        </nav>
        <h1>{activeAirportCode || "Airport"} Local Terrain</h1>
        <p>{displayState.status}</p>
        <dl>
          <div>
            <dt>Raster</dt>
            <dd>{displayState.rasterSize || "Pending"}</dd>
          </div>
          <div>
            <dt>Source TIFFs</dt>
            <dd>{displayState.sourceTiles || "Pending"}</dd>
          </div>
          <div>
            <dt>Tile grid</dt>
            <dd>{displayState.tileSize || "Pending"}</dd>
          </div>
          <div>
            <dt>Levels</dt>
            <dd>{displayState.levels || "Pending"}</dd>
          </div>
          <div>
            <dt>Tiles</dt>
            <dd>{displayState.tiles || "Pending"}</dd>
          </div>
          <div>
            <dt>Heights</dt>
            <dd>{displayState.heights || "Pending"}</dd>
          </div>
          <div>
            <dt>Center</dt>
            <dd>{displayState.center || "Pending"}</dd>
          </div>
          <div>
            <dt>Provider</dt>
            <dd>{displayState.source}</dd>
          </div>
        </dl>
        <div className="local-terrain-layer-toggles" aria-label="Local terrain layer toggles">
          <h2>Layers</h2>
          <label className="local-terrain-layer-toggle">
            <input
              type="checkbox"
              checked={layers.terrain}
              onChange={(event) => setLayerEnabled("terrain", event.currentTarget.checked)}
            />
            <span>Local terrain surface</span>
          </label>
          <label className="local-terrain-layer-toggle">
            <input
              type="checkbox"
              checked={layers.originalTifHeatmap}
              onChange={(event) =>
                setLayerEnabled("originalTifHeatmap", event.currentTarget.checked)
              }
            />
            <span>Original TIFF heatmap</span>
          </label>
          <label className="local-terrain-layer-toggle">
            <input
              type="checkbox"
              checked={layers.terrainTint}
              onChange={(event) => setLayerEnabled("terrainTint", event.currentTarget.checked)}
            />
            <span>Terrain-sampled heat tint</span>
          </label>
          <label className="local-terrain-layer-toggle">
            <input
              type="checkbox"
              checked={layers.satellite}
              onChange={(event) => setLayerEnabled("satellite", event.currentTarget.checked)}
            />
            <span>Satellite image</span>
          </label>
        </div>
        <p className="local-terrain-note">
          Turn off the terrain surface to inspect the original TIFF heatmap on a flat globe. View
          exaggeration is {TERRAIN_VERTICAL_EXAGGERATION}x; tile heights remain real local-terrain metres.
        </p>
      </section>
      <div className="local-terrain-legend">
        <span>{displayState.heightMin || "Low"}</span>
        <div />
        <span>{displayState.heightMax || "High"}</span>
      </div>
    </main>
  );
}
