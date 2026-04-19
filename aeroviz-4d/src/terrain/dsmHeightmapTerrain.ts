import * as Cesium from "cesium";

export const DSM_HEIGHTMAP_TERRAIN_METADATA_URL =
  "/data/DSM/CYVR/heightmap-terrain/metadata.json";

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
}

function tileKey(level: number, x: number, y: number): string {
  return `${level}/${x}/${y}`;
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
  metadataUrl: string = DSM_HEIGHTMAP_TERRAIN_METADATA_URL
): Promise<DsmHeightmapTerrain> {
  const response = await fetch(metadataUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch DSM terrain metadata: ${response.status}`);
  }

  const metadata = (await response.json()) as DsmHeightmapTerrainMetadata;
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const tileCache = new Map<string, Promise<Float32Array>>();
  const flatTile = createFlatHeightTile(metadata);

  const provider = new Cesium.CustomHeightmapTerrainProvider({
    width: metadata.tileWidth,
    height: metadata.tileHeight,
    tilingScheme,
    credit: "BC DSM heightmap terrain",
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
      const key = tileKey(level, x, y);
      let tilePromise = tileCache.get(key);
      if (!tilePromise) {
        tilePromise = fetchHeightTile(metadata, level, x, y);
        tileCache.set(key, tilePromise);
      }

      return tilePromise;
    },
  });

  return {
    metadata,
    provider,
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
