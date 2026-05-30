import * as Cesium from "cesium";
import { airportDsmHeightmapTerrainUrl } from "../data/airportData";
import { fetchJson } from "../utils/fetchJson";

export function dsmHeightmapTerrainMetadataUrl(airportCode: string): string {
  return airportDsmHeightmapTerrainUrl(airportCode, "metadata.json");
}

export interface DsmTerrainBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface DsmTerrainCorner {
  lon: number;
  lat: number;
}

export interface DsmTerrainLevelRange {
  level: number;
  xRange: [number, number];
  yRange: [number, number];
  tileCount: number;
}

export interface DsmHeightmapTerrainMetadata {
  format: "float32-little-endian-heightmap";
  tileWidth: number;
  tileHeight: number;
  tilingScheme: "geographic";
  tilesBaseUrl: string;
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
  bounds: DsmTerrainBounds;
  corners: {
    northWest: DsmTerrainCorner;
    northEast: DsmTerrainCorner;
    southEast: DsmTerrainCorner;
    southWest: DsmTerrainCorner;
  };
  levels: DsmTerrainLevelRange[];
  stats: {
    min: number;
    max: number;
    mean: number;
  };
}

export interface DsmHeightmapTerrain {
  metadata: DsmHeightmapTerrainMetadata;
  provider: Cesium.CustomHeightmapTerrainProvider;
  rectangle: Cesium.Rectangle;
  preloadTiles: (options?: PreloadDsmHeightmapTerrainTilesOptions) => Promise<void>;
}

export interface DsmTerrainTileRef {
  level: number;
  x: number;
  y: number;
}

export interface PreloadDsmHeightmapTerrainTilesOptions {
  concurrency?: number;
  tiles?: DsmTerrainTileRef[];
  onProgress?: (progress: DsmHeightmapTerrainPreloadProgress) => void;
}

export interface DsmHeightmapTerrainPreloadProgress {
  loadedTiles: number;
  totalTiles: number;
}

function tileKey(level: number, x: number, y: number): string {
  return `${level}/${x}/${y}`;
}

function enumerateAvailableTiles(metadata: DsmHeightmapTerrainMetadata): DsmTerrainTileRef[] {
  return metadata.levels.flatMap((range) => {
    const tiles: DsmTerrainTileRef[] = [];
    for (let x = range.xRange[0]; x <= range.xRange[1]; x += 1) {
      for (let y = range.yRange[0]; y <= range.yRange[1]; y += 1) {
        tiles.push({ level: range.level, x, y });
      }
    }
    return tiles;
  });
}

function isTileAvailable(
  metadata: DsmHeightmapTerrainMetadata,
  level: number,
  x: number,
  y: number
): boolean {
  const range = metadata.levels.find((item) => item.level === level);
  if (!range) return false;

  return x >= range.xRange[0] && x <= range.xRange[1] && y >= range.yRange[0] && y <= range.yRange[1];
}

function uniqueAvailableTiles(
  metadata: DsmHeightmapTerrainMetadata,
  tiles: DsmTerrainTileRef[],
): DsmTerrainTileRef[] {
  const seen = new Set<string>();
  const uniqueTiles: DsmTerrainTileRef[] = [];

  for (const tile of tiles) {
    if (!isTileAvailable(metadata, tile.level, tile.x, tile.y)) continue;
    const key = tileKey(tile.level, tile.x, tile.y);
    if (seen.has(key)) continue;
    seen.add(key);
    uniqueTiles.push(tile);
  }

  return uniqueTiles;
}

export function dsmTerrainTileRefsNearCoordinate(
  metadata: DsmHeightmapTerrainMetadata,
  lon: number,
  lat: number,
  options: {
    maxLevelRadius?: number;
    ancestorRadius?: number;
  } = {},
): DsmTerrainTileRef[] {
  const maxLevelRadius = Math.max(0, Math.floor(options.maxLevelRadius ?? 4));
  const ancestorRadius = Math.max(0, Math.floor(options.ancestorRadius ?? 0));
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const position = Cesium.Cartographic.fromDegrees(lon, lat);
  const tiles: DsmTerrainTileRef[] = [];

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

function createFlatHeightTile(metadata: DsmHeightmapTerrainMetadata): Float32Array {
  const values = new Float32Array(metadata.tileWidth * metadata.tileHeight);
  values.fill(metadata.fallbackHeightM);
  return values;
}

async function fetchHeightTile(
  metadata: DsmHeightmapTerrainMetadata,
  level: number,
  x: number,
  y: number
): Promise<Float32Array> {
  const response = await fetch(`${metadata.tilesBaseUrl}/${level}/${x}/${y}.f32`);
  if (!response.ok) {
    throw new Error(`Failed to fetch DSM height tile ${tileKey(level, x, y)}: ${response.status}`);
  }

  const heights = parseFloat32LittleEndian(await response.arrayBuffer());
  const expectedLength = metadata.tileWidth * metadata.tileHeight;
  if (heights.length !== expectedLength) {
    throw new Error(
      `DSM height tile ${tileKey(level, x, y)} has ${heights.length} samples; expected ${expectedLength}`
    );
  }

  return heights;
}

export async function loadDsmHeightmapTerrain(
  metadataUrl: string
): Promise<DsmHeightmapTerrain> {
  const metadata = await fetchJson<DsmHeightmapTerrainMetadata>(metadataUrl);
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
        // Returning a flat tile for non-DSM siblings avoids holes around the
        // patch while keeping network fetches limited to real DSM tiles.
        return flatTile.slice();
      }

      // Cesium calls this frequently while refining terrain. Cache promises, not
      // only resolved arrays, so repeated requests share the same in-flight fetch.
      return getCachedHeightTile(level, x, y);
    },
  });

  const preloadTileRefs = enumerateAvailableTiles(metadata);
  const preloadTiles = (
    options: PreloadDsmHeightmapTerrainTilesOptions = {},
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

export function dsmTerrainFootprintDegrees(metadata: DsmHeightmapTerrainMetadata): number[] {
  return [
    metadata.corners.northWest,
    metadata.corners.northEast,
    metadata.corners.southEast,
    metadata.corners.southWest,
    metadata.corners.northWest,
  ].flatMap(({ lon, lat }) => [lon, lat]);
}
