import { useEffect } from "react";
import { fromArrayBuffer } from "geotiff";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

export const DEFAULT_DSM_TIFF_URL =
  "/data/DSM/CYVR/bc_092g015_3_3_3_xli1m_utm10_20240217_20250425.tif";

const DEFAULT_TILE_SIZE = 65;
const DEFAULT_UTM_ZONE = 10;
const UTM_K0 = 0.9996;
const WGS84_A = 6378137.0;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2);

type RasterData = Float32Array | Float64Array | Int16Array | Uint16Array | Int32Array | Uint32Array | number[];

export interface DsmRaster {
  data: RasterData;
  width: number;
  height: number;
  originX: number;
  originY: number;
  resolutionX: number;
  resolutionY: number;
  noData: number | null;
  rectangle: Cesium.Rectangle;
  utmZone: number;
}

export interface DsmStats {
  min: number;
  max: number;
  mean: number;
}

interface DsmTerrainSurface {
  provider: Cesium.CustomHeightmapTerrainProvider;
  rectangle: Cesium.Rectangle;
  stats: DsmStats;
}

interface UseDsmTerrainLayerOptions {
  enabled?: boolean;
  url?: string;
  tileSize?: number;
  utmZone?: number;
}

function centralMeridianRad(zone: number): number {
  return Cesium.Math.toRadians((zone - 1) * 6 - 180 + 3);
}

function utmToLonLat(easting: number, northing: number, zone: number): [number, number] {
  const x = easting - 500000.0;
  const m = northing / UTM_K0;
  const mu =
    m /
    (WGS84_A *
      (1 -
        WGS84_E2 / 4 -
        (3 * WGS84_E2 ** 2) / 64 -
        (5 * WGS84_E2 ** 3) / 256));

  const e1 = (1 - Math.sqrt(1 - WGS84_E2)) / (1 + Math.sqrt(1 - WGS84_E2));
  const j1 = (3 * e1) / 2 - (27 * e1 ** 3) / 32;
  const j2 = (21 * e1 ** 2) / 16 - (55 * e1 ** 4) / 32;
  const j3 = (151 * e1 ** 3) / 96;
  const j4 = (1097 * e1 ** 4) / 512;
  const fp =
    mu +
    j1 * Math.sin(2 * mu) +
    j2 * Math.sin(4 * mu) +
    j3 * Math.sin(6 * mu) +
    j4 * Math.sin(8 * mu);

  const sinFp = Math.sin(fp);
  const cosFp = Math.cos(fp);
  const tanFp = Math.tan(fp);
  const c1 = WGS84_EP2 * cosFp ** 2;
  const t1 = tanFp ** 2;
  const n1 = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinFp ** 2);
  const r1 = (WGS84_A * (1 - WGS84_E2)) / (1 - WGS84_E2 * sinFp ** 2) ** 1.5;
  const d = x / (n1 * UTM_K0);

  const lat =
    fp -
    ((n1 * tanFp) / r1) *
      (d ** 2 / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * WGS84_EP2) * d ** 4) / 24 +
        ((61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * WGS84_EP2 - 3 * c1 ** 2) *
          d ** 6) /
          720);

  const lon =
    centralMeridianRad(zone) +
    (d -
      ((1 + 2 * t1 + c1) * d ** 3) / 6 +
      ((5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * WGS84_EP2 + 24 * t1 ** 2) *
        d ** 5) /
        120) /
      cosFp;

  return [Cesium.Math.toDegrees(lon), Cesium.Math.toDegrees(lat)];
}

function lonLatToUtm(lonDeg: number, latDeg: number, zone: number): [number, number] {
  const lon = Cesium.Math.toRadians(lonDeg);
  const lat = Cesium.Math.toRadians(latDeg);
  const lon0 = centralMeridianRad(zone);

  const sinLat = Math.sin(lat);
  const cosLat = Math.cos(lat);
  const tanLat = Math.tan(lat);
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat ** 2);
  const t = tanLat ** 2;
  const c = WGS84_EP2 * cosLat ** 2;
  const a = cosLat * (lon - lon0);
  const m =
    WGS84_A *
    ((1 - WGS84_E2 / 4 - (3 * WGS84_E2 ** 2) / 64 - (5 * WGS84_E2 ** 3) / 256) * lat -
      ((3 * WGS84_E2) / 8 + (3 * WGS84_E2 ** 2) / 32 + (45 * WGS84_E2 ** 3) / 1024) *
        Math.sin(2 * lat) +
      ((15 * WGS84_E2 ** 2) / 256 + (45 * WGS84_E2 ** 3) / 1024) * Math.sin(4 * lat) -
      ((35 * WGS84_E2 ** 3) / 3072) * Math.sin(6 * lat));

  const easting =
    500000 +
    UTM_K0 *
      n *
      (a +
        ((1 - t + c) * a ** 3) / 6 +
        ((5 - 18 * t + t ** 2 + 72 * c - 58 * WGS84_EP2) * a ** 5) / 120);
  const northing =
    UTM_K0 *
    (m +
      n *
        tanLat *
        (a ** 2 / 2 +
          ((5 - t + 9 * c + 4 * c ** 2) * a ** 4) / 24 +
          ((61 - 58 * t + t ** 2 + 600 * c - 330 * WGS84_EP2) * a ** 6) / 720));

  return [easting, northing];
}

function dsmRectangle(bbox: number[], zone: number): Cesium.Rectangle {
  const [west, south] = utmToLonLat(bbox[0], bbox[1], zone);
  const [east, north] = utmToLonLat(bbox[2], bbox[3], zone);
  return Cesium.Rectangle.fromDegrees(west, south, east, north);
}

export async function loadDsmRaster(
  url: string,
  utmZone: number = DEFAULT_UTM_ZONE,
  signal?: AbortSignal
): Promise<DsmRaster> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }

  // Fetch once instead of using geotiff.fromUrl(). This DSM is stored with one
  // row per strip, so range-based decoding can trigger thousands of requests.
  const tiff = await fromArrayBuffer(await response.arrayBuffer(), signal);
  const image = await tiff.getImage();
  const origin = image.getOrigin();
  const resolution = image.getResolution();
  const raster = (await image.readRasters({
    samples: [0],
    interleave: true,
  })) as unknown as RasterData;

  const noData = image.getGDALNoData();

  return {
    data: raster,
    width: image.getWidth(),
    height: image.getHeight(),
    originX: origin[0],
    originY: origin[1],
    resolutionX: resolution[0],
    resolutionY: resolution[1],
    noData,
    rectangle: dsmRectangle(image.getBoundingBox(), utmZone),
    utmZone,
  };
}

export function calculateDsmStats(dsm: DsmRaster): DsmStats {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;

  for (let i = 0; i < dsm.data.length; i += 1) {
    const value = Number(dsm.data[i]);
    if (!Number.isFinite(value) || value === dsm.noData) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    count += 1;
  }

  return {
    min: count > 0 ? min : 0,
    max: count > 0 ? max : 0,
    mean: count > 0 ? sum / count : 0,
  };
}

function sampleDsm(dsm: DsmRaster, lonRad: number, latRad: number): number {
  const lon = Cesium.Math.toDegrees(lonRad);
  const lat = Cesium.Math.toDegrees(latRad);
  const [easting, northing] = lonLatToUtm(lon, lat, dsm.utmZone);
  const x = (easting - dsm.originX) / dsm.resolutionX;
  const y = (northing - dsm.originY) / dsm.resolutionY;

  if (x < 0 || y < 0 || x > dsm.width - 1 || y > dsm.height - 1) {
    return 0;
  }

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, dsm.width - 1);
  const y1 = Math.min(y0 + 1, dsm.height - 1);
  const tx = x - x0;
  const ty = y - y0;

  const valueAt = (col: number, row: number): number => {
    const value = Number(dsm.data[row * dsm.width + col]);
    if (!Number.isFinite(value) || value === dsm.noData) return 0;
    return value;
  };

  const top = valueAt(x0, y0) * (1 - tx) + valueAt(x1, y0) * tx;
  const bottom = valueAt(x0, y1) * (1 - tx) + valueAt(x1, y1) * tx;
  return top * (1 - ty) + bottom * ty;
}

export function createDsmTerrainSurface(
  dsm: DsmRaster,
  tileSize: number = DEFAULT_TILE_SIZE
): DsmTerrainSurface {
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const provider = new Cesium.CustomHeightmapTerrainProvider({
    width: tileSize,
    height: tileSize,
    tilingScheme,
    credit: "BC DSM GeoTIFF",
    callback: (x, y, level) => {
      const tileRectangle = tilingScheme.tileXYToRectangle(x, y, level);
      if (!Cesium.Rectangle.simpleIntersection(tileRectangle, dsm.rectangle)) {
        return undefined;
      }

      const heights = new Float32Array(tileSize * tileSize);
      for (let row = 0; row < tileSize; row += 1) {
        const v = tileSize === 1 ? 0 : row / (tileSize - 1);
        const lat = Cesium.Math.lerp(tileRectangle.north, tileRectangle.south, v);

        for (let col = 0; col < tileSize; col += 1) {
          const u = tileSize === 1 ? 0 : col / (tileSize - 1);
          const lon = Cesium.Math.lerp(tileRectangle.west, tileRectangle.east, u);
          heights[row * tileSize + col] = sampleDsm(dsm, lon, lat);
        }
      }

      return heights;
    },
  });

  return {
    provider,
    rectangle: dsm.rectangle,
    stats: calculateDsmStats(dsm),
  };
}

function rampColor(t: number): [number, number, number] {
  const clamped = Cesium.Math.clamp(t, 0, 1);
  if (clamped < 0.33) {
    const u = clamped / 0.33;
    return [
      Math.round(Cesium.Math.lerp(30, 34, u)),
      Math.round(Cesium.Math.lerp(86, 173, u)),
      Math.round(Cesium.Math.lerp(168, 126, u)),
    ];
  }
  if (clamped < 0.66) {
    const u = (clamped - 0.33) / 0.33;
    return [
      Math.round(Cesium.Math.lerp(34, 245, u)),
      Math.round(Cesium.Math.lerp(173, 196, u)),
      Math.round(Cesium.Math.lerp(126, 71, u)),
    ];
  }

  const u = (clamped - 0.66) / 0.34;
  return [
    Math.round(Cesium.Math.lerp(245, 198, u)),
    Math.round(Cesium.Math.lerp(196, 55, u)),
    Math.round(Cesium.Math.lerp(71, 52, u)),
  ];
}

export function createDsmColorRampDataUrl(
  dsm: DsmRaster,
  maxDimension: number = 1024
): { url: string; stats: DsmStats; width: number; height: number } {
  const stats = calculateDsmStats(dsm);
  const scale = Math.min(1, maxDimension / Math.max(dsm.width, dsm.height));
  const width = Math.max(1, Math.round(dsm.width * scale));
  const height = Math.max(1, Math.round(dsm.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Unable to create DSM color ramp canvas");
  }

  const imageData = ctx.createImageData(width, height);
  const range = stats.max - stats.min || 1;

  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(dsm.height - 1, Math.floor((y / height) * dsm.height));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(dsm.width - 1, Math.floor((x / width) * dsm.width));
      const value = Number(dsm.data[sourceY * dsm.width + sourceX]);
      const offset = (y * width + x) * 4;

      if (!Number.isFinite(value) || value === dsm.noData) {
        imageData.data[offset + 3] = 0;
        continue;
      }

      const [r, g, b] = rampColor((value - stats.min) / range);
      imageData.data[offset] = r;
      imageData.data[offset + 1] = g;
      imageData.data[offset + 2] = b;
      imageData.data[offset + 3] = 245;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return {
    url: canvas.toDataURL("image/png"),
    stats,
    width,
    height,
  };
}

/**
 * Load a local GeoTIFF DSM through Cesium.CustomHeightmapTerrainProvider.
 *
 * CesiumJS cannot use a GeoTIFF URL directly as terrain. This hook decodes the
 * TIFF in the browser and samples it into Cesium heightmap tiles. It replaces
 * the current terrain provider while mounted, so enable it only for DSM tests.
 */
export function useDsmTerrainLayer(options: UseDsmTerrainLayerOptions = {}): void {
  const { viewer } = useApp();
  const enabled = options.enabled ?? false;
  const url = options.url ?? DEFAULT_DSM_TIFF_URL;
  const tileSize = options.tileSize ?? DEFAULT_TILE_SIZE;
  const utmZone = options.utmZone ?? DEFAULT_UTM_ZONE;

  useEffect(() => {
    if (!viewer || !enabled) return;

    let cancelled = false;
    const abortController = new AbortController();
    const previousTerrainProvider = viewer.scene.terrainProvider;
    const dsmPromise = loadDsmRaster(url, utmZone, abortController.signal);
    let provider: Cesium.CustomHeightmapTerrainProvider | null = null;

    dsmPromise
      .then((dsm) => {
        provider = createDsmTerrainSurface(dsm, tileSize).provider;

        if (!cancelled && !viewer.isDestroyed()) {
          viewer.scene.terrainProvider = provider;
        }
      })
      .catch((error) => {
        console.error("[DsmTerrainLayer] Failed to load DSM GeoTIFF:", error);
      });

    return () => {
      cancelled = true;
      abortController.abort();
      if (!viewer.isDestroyed() && provider && viewer.scene.terrainProvider === provider) {
        viewer.scene.terrainProvider = previousTerrainProvider;
      }
    };
  }, [enabled, tileSize, url, utmZone, viewer]);
}
