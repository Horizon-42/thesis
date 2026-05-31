import * as Cesium from "cesium";
import { airportLocalTerrainUrl } from "../data/airportData";
import { fetchJson } from "../utils/fetchJson";

export function airportLocalTerrainMetadataUrl(airportCode: string): string {
  return airportLocalTerrainUrl(airportCode, "metadata.json");
}

export interface AirportLocalTerrainBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface AirportLocalTerrainCorner {
  lon: number;
  lat: number;
}

export interface AirportLocalTerrainLevelRange {
  level: number;
  xRange: [number, number];
  yRange: [number, number];
  tileCount: number;
}

export type AirportLocalTerrainSourceKind = "dem" | "dsm" | "unknown";

export interface AirportLocalTerrainPrecisionMetadata {
  /** Ground sample distance in metres. Smaller values are higher precision. */
  horizontalResolutionM: number;
  verticalAccuracyM?: number | null;
  source: string;
  notes?: string[];
}

export interface AirportLocalTerrainSourceMetadata {
  kind: AirportLocalTerrainSourceKind;
  label: string;
  sourceDir?: string;
}

export interface AirportLocalTerrainMetadata {
  format: "float32-little-endian-heightmap";
  tileWidth: number;
  tileHeight: number;
  tilingScheme: "geographic";
  tilesBaseUrl: string;
  source?: AirportLocalTerrainSourceMetadata;
  precision: AirportLocalTerrainPrecisionMetadata;
  overlay?: {
    url: string;
    width: number;
    height: number;
    note?: string;
  };
  originalTifHeatmap?: {
    url: string;
    width: number;
    height: number;
    bounds?: AirportLocalTerrainBounds;
    note?: string;
  };
  hillshade?: {
    url: string;
    width: number;
    height: number;
    alpha?: number;
    note?: string;
  };
  minLevel: number;
  maxLevel: number;
  tileCount: number;
  fallbackHeightM: number;
  raster: {
    width: number;
    height: number;
    noData: number | null;
    sourceTileCount?: number;
    validSampleCount?: number;
  };
  bounds: AirportLocalTerrainBounds;
  corners: {
    northWest: AirportLocalTerrainCorner;
    northEast: AirportLocalTerrainCorner;
    southEast: AirportLocalTerrainCorner;
    southWest: AirportLocalTerrainCorner;
  };
  levels: AirportLocalTerrainLevelRange[];
  stats: {
    min: number;
    max: number;
    mean: number;
  };
}

export class AirportLocalTerrainMetadataError extends Error {
  readonly metadataUrl: string;
  readonly code: "missing-precision-metadata" | "invalid-metadata";

  constructor(
    message: string,
    options: {
      metadataUrl: string;
      code: AirportLocalTerrainMetadataError["code"];
    },
  ) {
    super(message);
    this.name = "AirportLocalTerrainMetadataError";
    this.metadataUrl = options.metadataUrl;
    this.code = options.code;
  }
}

export interface AirportLocalTerrain {
  metadata: AirportLocalTerrainMetadata;
  provider: Cesium.CustomHeightmapTerrainProvider;
  rectangle: Cesium.Rectangle;
  preloadTiles: (options?: PreloadAirportLocalTerrainTilesOptions) => Promise<void>;
}

export interface AirportLocalTerrainTileRef {
  level: number;
  x: number;
  y: number;
}

export interface PreloadAirportLocalTerrainTilesOptions {
  concurrency?: number;
  tiles?: AirportLocalTerrainTileRef[];
  signal?: AbortSignal;
  onProgress?: (progress: AirportLocalTerrainPreloadProgress) => void;
}

export interface AirportLocalTerrainPreloadProgress {
  loadedTiles: number;
  totalTiles: number;
}

const HEIGHT_EPSILON_M = 0.001;
const FALLBACK_FILL_LEVEL_OFFSET = 0;
const DEFAULT_MAX_CACHED_HEIGHT_TILES = 512;
const HOST_USES_LITTLE_ENDIAN_FLOAT32 =
  new Uint8Array(new Float32Array([1]).buffer)[0] === 0;

function tileKey(level: number, x: number, y: number): string {
  return `${level}/${x}/${y}`;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function preloadAbortError(): Error {
  const error = new Error("Airport local terrain preload was aborted");
  error.name = "AbortError";
  return error;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw preloadAbortError();
}

function assertValidAirportLocalTerrainMetadata(
  value: AirportLocalTerrainMetadata,
  metadataUrl: string,
): void {
  if (!value.precision || !isFiniteNumber(value.precision.horizontalResolutionM)) {
    throw new AirportLocalTerrainMetadataError(
      [
        `Local terrain package ${metadataUrl} is missing precision.horizontalResolutionM.`,
        "Regenerate the airport local terrain package so source priority can be based on precision.",
      ].join(" "),
      { metadataUrl, code: "missing-precision-metadata" },
    );
  }

  if (value.precision.horizontalResolutionM <= 0) {
    throw new AirportLocalTerrainMetadataError(
      `Local terrain package ${metadataUrl} has invalid precision.horizontalResolutionM.`,
      { metadataUrl, code: "invalid-metadata" },
    );
  }
}

function enumerateAvailableTiles(
  metadata: AirportLocalTerrainMetadata,
): AirportLocalTerrainTileRef[] {
  return metadata.levels.flatMap((range) => {
    const tiles: AirportLocalTerrainTileRef[] = [];
    for (let x = range.xRange[0]; x <= range.xRange[1]; x += 1) {
      for (let y = range.yRange[0]; y <= range.yRange[1]; y += 1) {
        tiles.push({ level: range.level, x, y });
      }
    }
    return tiles;
  });
}

function isTileAvailable(
  metadata: AirportLocalTerrainMetadata,
  level: number,
  x: number,
  y: number
): boolean {
  const range = metadata.levels.find((item) => item.level === level);
  if (!range) return false;

  return x >= range.xRange[0] && x <= range.xRange[1] && y >= range.yRange[0] && y <= range.yRange[1];
}

function uniqueAvailableTiles(
  metadata: AirportLocalTerrainMetadata,
  tiles: AirportLocalTerrainTileRef[],
): AirportLocalTerrainTileRef[] {
  const seen = new Set<string>();
  const uniqueTiles: AirportLocalTerrainTileRef[] = [];

  for (const tile of tiles) {
    if (!isTileAvailable(metadata, tile.level, tile.x, tile.y)) continue;
    const key = tileKey(tile.level, tile.x, tile.y);
    if (seen.has(key)) continue;
    seen.add(key);
    uniqueTiles.push(tile);
  }

  return uniqueTiles;
}

export function airportLocalTerrainTileRefsNearCoordinate(
  metadata: AirportLocalTerrainMetadata,
  lon: number,
  lat: number,
  options: {
    maxLevelRadius?: number;
    ancestorRadius?: number;
  } = {},
): AirportLocalTerrainTileRef[] {
  const maxLevelRadius = Math.max(0, Math.floor(options.maxLevelRadius ?? 4));
  const ancestorRadius = Math.max(0, Math.floor(options.ancestorRadius ?? 0));
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const position = Cesium.Cartographic.fromDegrees(lon, lat);
  const tiles: AirportLocalTerrainTileRef[] = [];

  for (const range of metadata.levels) {
    const centerTile = tilingScheme.positionToTileXY(position, range.level);
    if (!centerTile) continue;

    const radius = range.level === metadata.maxLevel ? maxLevelRadius : ancestorRadius;
    for (let x = centerTile.x - radius; x <= centerTile.x + radius; x += 1) {
      for (let y = centerTile.y - radius; y <= centerTile.y + radius; y += 1) {
        tiles.push({ level: range.level, x, y });
      }
    }
  }

  return uniqueAvailableTiles(metadata, tiles);
}

function parseFloat32LittleEndian(buffer: ArrayBuffer): Float32Array {
  // Generated tiles are little-endian Float32. Every supported browser/Node
  // target is little-endian today, so avoid a per-sample DataView loop on the
  // hot terrain tile path and keep the portable path for unusual runtimes.
  if (HOST_USES_LITTLE_ENDIAN_FLOAT32) return new Float32Array(buffer);

  const view = new DataView(buffer);
  const values = new Float32Array(buffer.byteLength / Float32Array.BYTES_PER_ELEMENT);

  for (let i = 0; i < values.length; i += 1) {
    values[i] = view.getFloat32(i * Float32Array.BYTES_PER_ELEMENT, true);
  }

  return values;
}

export function airportLocalTerrainDisplayFallbackHeight(
  metadata: AirportLocalTerrainMetadata,
): number {
  const fallbackHeight = metadata.fallbackHeightM;
  const { min, max } = metadata.stats;
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) {
    return Number.isFinite(fallbackHeight) ? fallbackHeight : 0;
  }

  if (!Number.isFinite(fallbackHeight) || fallbackHeight < min || fallbackHeight > max) {
    return min;
  }

  return fallbackHeight;
}

export function shouldFillAirportLocalTerrainFallbackHeights(
  metadata: AirportLocalTerrainMetadata,
  level: number,
): boolean {
  return level >= metadata.maxLevel - FALLBACK_FILL_LEVEL_OFFSET;
}

function isFallbackHeightSample(
  metadata: AirportLocalTerrainMetadata,
  value: number,
): boolean {
  if (!Number.isFinite(value)) return true;

  const { min, max } = metadata.stats;
  if (
    Number.isFinite(min) &&
    Number.isFinite(max) &&
    min <= max &&
    (value < min - HEIGHT_EPSILON_M || value > max + HEIGHT_EPSILON_M)
  ) {
    return true;
  }

  const fallbackHeight = metadata.fallbackHeightM;
  return (
    Number.isFinite(fallbackHeight) &&
    Math.abs(value - fallbackHeight) <= HEIGHT_EPSILON_M &&
    Number.isFinite(min) &&
    Number.isFinite(max) &&
    (fallbackHeight < min || fallbackHeight > max)
  );
}

export function fillAirportLocalTerrainFallbackHeights(
  metadata: AirportLocalTerrainMetadata,
  heights: Float32Array,
): Float32Array {
  const expectedLength = metadata.tileWidth * metadata.tileHeight;
  if (heights.length !== expectedLength) return heights;

  const width = metadata.tileWidth;
  const height = metadata.tileHeight;
  const valid = new Uint8Array(expectedLength);
  const fallbackHeight = airportLocalTerrainDisplayFallbackHeight(metadata);
  let validCount = 0;

  for (let i = 0; i < heights.length; i += 1) {
    if (isFallbackHeightSample(metadata, heights[i])) {
      heights[i] = fallbackHeight;
    } else {
      valid[i] = 1;
      validCount += 1;
    }
  }

  if (validCount === 0) {
    heights.fill(fallbackHeight);
    return heights;
  }
  if (validCount === heights.length) return heights;

  for (let row = 0; row < height; row += 1) {
    let lastValidHeight: number | null = null;
    const rowOffset = row * width;
    for (let col = 0; col < width; col += 1) {
      const index = rowOffset + col;
      if (valid[index]) {
        lastValidHeight = heights[index];
      } else if (lastValidHeight !== null) {
        heights[index] = lastValidHeight;
        valid[index] = 1;
      }
    }

    lastValidHeight = null;
    for (let col = width - 1; col >= 0; col -= 1) {
      const index = rowOffset + col;
      if (valid[index]) {
        lastValidHeight = heights[index];
      } else if (lastValidHeight !== null) {
        heights[index] = lastValidHeight;
        valid[index] = 1;
      }
    }
  }

  for (let col = 0; col < width; col += 1) {
    let lastValidHeight: number | null = null;
    for (let row = 0; row < height; row += 1) {
      const index = row * width + col;
      if (valid[index]) {
        lastValidHeight = heights[index];
      } else if (lastValidHeight !== null) {
        heights[index] = lastValidHeight;
        valid[index] = 1;
      }
    }

    lastValidHeight = null;
    for (let row = height - 1; row >= 0; row -= 1) {
      const index = row * width + col;
      if (valid[index]) {
        lastValidHeight = heights[index];
      } else if (lastValidHeight !== null) {
        heights[index] = lastValidHeight;
        valid[index] = 1;
      }
    }
  }

  return heights;
}

function createFlatHeightTile(metadata: AirportLocalTerrainMetadata): Float32Array {
  const values = new Float32Array(metadata.tileWidth * metadata.tileHeight);
  values.fill(metadata.fallbackHeightM);
  return values;
}

async function fetchHeightTile(
  metadata: AirportLocalTerrainMetadata,
  level: number,
  x: number,
  y: number,
  signal?: AbortSignal,
): Promise<Float32Array> {
  throwIfAborted(signal);
  const response = await fetch(`${metadata.tilesBaseUrl}/${level}/${x}/${y}.f32`, {
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch local terrain height tile ${tileKey(level, x, y)}: ${response.status}`);
  }

  const heights = parseFloat32LittleEndian(await response.arrayBuffer());
  const expectedLength = metadata.tileWidth * metadata.tileHeight;
  if (heights.length !== expectedLength) {
    throw new Error(
      `Local terrain height tile ${tileKey(level, x, y)} has ${heights.length} samples; expected ${expectedLength}`
    );
  }

  return shouldFillAirportLocalTerrainFallbackHeights(metadata, level)
    ? fillAirportLocalTerrainFallbackHeights(metadata, heights)
    : heights;
}

export async function loadAirportLocalTerrain(
  metadataUrl: string
): Promise<AirportLocalTerrain> {
  const metadata = await fetchJson<AirportLocalTerrainMetadata>(metadataUrl);
  assertValidAirportLocalTerrainMetadata(metadata, metadataUrl);
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const tileCache = new Map<string, Promise<Float32Array>>();
  const maxCachedHeightTiles = Math.max(
    1,
    Math.min(metadata.tileCount, DEFAULT_MAX_CACHED_HEIGHT_TILES),
  );
  const flatTile = createFlatHeightTile(metadata);

  const touchCachedTile = (key: string, tilePromise: Promise<Float32Array>): void => {
    tileCache.delete(key);
    tileCache.set(key, tilePromise);
  };

  const evictStaleHeightTiles = (): void => {
    while (tileCache.size > maxCachedHeightTiles) {
      const oldestKey = tileCache.keys().next().value as string | undefined;
      if (!oldestKey) return;
      tileCache.delete(oldestKey);
    }
  };

  const getCachedHeightTile = (
    level: number,
    x: number,
    y: number,
    signal?: AbortSignal,
  ): Promise<Float32Array> => {
    throwIfAborted(signal);
    const key = tileKey(level, x, y);
    let tilePromise = tileCache.get(key);
    if (tilePromise) {
      touchCachedTile(key, tilePromise);
      return tilePromise;
    }

    const newTilePromise = fetchHeightTile(metadata, level, x, y, signal)
      .then((heights) => {
        if (tileCache.get(key) === newTilePromise) {
          touchCachedTile(key, newTilePromise);
          evictStaleHeightTiles();
        }
        return heights;
      })
      .catch((error) => {
        tileCache.delete(key);
        throw error;
      });
    tileCache.set(key, newTilePromise);
    return newTilePromise;
  };

  const provider = new Cesium.CustomHeightmapTerrainProvider({
    width: metadata.tileWidth,
    height: metadata.tileHeight,
    tilingScheme,
    credit: "Airport heightmap terrain",
    callback: (x, y, level) => {
      if (level > metadata.maxLevel) return undefined;
      if (!isTileAvailable(metadata, level, x, y)) {
        // Ancestor tiles report all children as available because
        // CustomHeightmapTerrainProvider does not expose childTileMask control.
        // Returning a flat tile for non-local siblings avoids holes around the
        // patch while keeping network fetches limited to real local terrain tiles.
        return flatTile.slice();
      }

      // Cesium calls this frequently while refining terrain. Cache promises, not
      // only resolved arrays, so repeated requests share the same in-flight fetch.
      return getCachedHeightTile(level, x, y);
    },
  });

  const preloadTileRefs = enumerateAvailableTiles(metadata);
  const preloadTiles = (
    options: PreloadAirportLocalTerrainTilesOptions = {},
  ): Promise<void> => {
    const concurrency = Math.max(1, Math.floor(options.concurrency ?? 16));
    const signal = options.signal;
    const tilesToLoad = options.tiles
      ? uniqueAvailableTiles(metadata, options.tiles)
      : preloadTileRefs;
    const totalTiles = tilesToLoad.length;

    let loadedTiles = 0;
    let nextTileIndex = 0;
    throwIfAborted(signal);
    options.onProgress?.({ loadedTiles, totalTiles });

    const preloadWorker = async () => {
      while (nextTileIndex < totalTiles) {
        throwIfAborted(signal);
        const tile = tilesToLoad[nextTileIndex];
        nextTileIndex += 1;
        await getCachedHeightTile(tile.level, tile.x, tile.y, signal);
        throwIfAborted(signal);
        loadedTiles += 1;
        options.onProgress?.({ loadedTiles, totalTiles });
      }
    };

    const workerCount = Math.min(concurrency, totalTiles);
    return Promise.all(
      Array.from({ length: workerCount }, () => preloadWorker()),
    ).then(() => undefined);
  };

  return {
    metadata,
    provider,
    preloadTiles,
    rectangle: Cesium.Rectangle.fromDegrees(
      metadata.bounds.west,
      metadata.bounds.south,
      metadata.bounds.east,
      metadata.bounds.north
    ),
  };
}

export function airportLocalTerrainFootprintDegrees(metadata: AirportLocalTerrainMetadata): number[] {
  return [
    metadata.corners.northWest,
    metadata.corners.northEast,
    metadata.corners.southEast,
    metadata.corners.southWest,
    metadata.corners.northWest,
  ].flatMap(({ lon, lat }) => [lon, lat]);
}
