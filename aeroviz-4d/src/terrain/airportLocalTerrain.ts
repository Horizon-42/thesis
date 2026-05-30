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
  onProgress?: (progress: AirportLocalTerrainPreloadProgress) => void;
}

export interface AirportLocalTerrainPreloadProgress {
  loadedTiles: number;
  totalTiles: number;
}

function tileKey(level: number, x: number, y: number): string {
  return `${level}/${x}/${y}`;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
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
  const view = new DataView(buffer);
  const values = new Float32Array(buffer.byteLength / Float32Array.BYTES_PER_ELEMENT);

  for (let i = 0; i < values.length; i += 1) {
    values[i] = view.getFloat32(i * Float32Array.BYTES_PER_ELEMENT, true);
  }

  return values;
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
  y: number
): Promise<Float32Array> {
  const response = await fetch(`${metadata.tilesBaseUrl}/${level}/${x}/${y}.f32`);
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

  return heights;
}

export async function loadAirportLocalTerrain(
  metadataUrl: string
): Promise<AirportLocalTerrain> {
  const metadata = await fetchJson<AirportLocalTerrainMetadata>(metadataUrl);
  assertValidAirportLocalTerrainMetadata(metadata, metadataUrl);
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const tileCache = new Map<string, Promise<Float32Array>>();
  const flatTile = createFlatHeightTile(metadata);

  const getCachedHeightTile = (level: number, x: number, y: number): Promise<Float32Array> => {
    const key = tileKey(level, x, y);
    let tilePromise = tileCache.get(key);
    if (!tilePromise) {
      tilePromise = fetchHeightTile(metadata, level, x, y).catch((error) => {
        tileCache.delete(key);
        throw error;
      });
      tileCache.set(key, tilePromise);
    }
    return tilePromise;
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
    const tilesToLoad = options.tiles
      ? uniqueAvailableTiles(metadata, options.tiles)
      : preloadTileRefs;
    const totalTiles = tilesToLoad.length;

    let loadedTiles = 0;
    let nextTileIndex = 0;
    options.onProgress?.({ loadedTiles, totalTiles });

    const preloadWorker = async () => {
      while (nextTileIndex < totalTiles) {
        const tile = tilesToLoad[nextTileIndex];
        nextTileIndex += 1;
        await getCachedHeightTile(tile.level, tile.x, tile.y);
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
