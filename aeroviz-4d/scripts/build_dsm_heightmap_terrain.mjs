import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";
import { fromFile } from "geotiff";
import * as Cesium from "cesium";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const AIRPORT_CODE = (cliOption("--airport") ?? "CYVR").toUpperCase();
const rawOpenTopographyDemInputDir = path.resolve(repoRoot, `../data/opentopography/${AIRPORT_CODE}/dem`);
const rawLidarInputDir = path.resolve(repoRoot, `../data/bc_lidar/${AIRPORT_CODE}/dsm`);
const rawUsgsLidarInputDir = path.resolve(repoRoot, `../data/usgs_lidar/${AIRPORT_CODE}/dsm`);
const defaultInputDir = path.resolve(repoRoot, `public/data/airports/${AIRPORT_CODE}/dsm/source`);
const fallbackInputDir = path.resolve(repoRoot, `public/data/DSM/${AIRPORT_CODE}`);
const legacySingleInput = path.resolve(
  repoRoot,
  `../data/DSM/${AIRPORT_CODE}/bc_092g015_3_3_3_xli1m_utm10_20240217_20250425.tif`
);
const outputDir = path.resolve(
  repoRoot,
  `public/data/airports/${AIRPORT_CODE}/dsm/heightmap-terrain`
);
const tilesDir = path.join(outputDir, "tiles");

let UTM_ZONE = Number(cliOption("--utm-zone") ?? Number.NaN);
const TILE_SIZE = 129;
const MIN_LEVEL = 0;
const MAX_LEVEL = 16;
const FALLBACK_HEIGHT_M = 0;
const OVERLAY_MAX_WIDTH = 1024;
const SOURCE_TILE_INDEX_CELL_SIZE_M = 512;
const SOURCE_TILE_INDEX_CELL_SIZE_DEGREES = 0.01;

const UTM_K0 = 0.9996;
const WGS84_A = 6378137.0;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2);

const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n += 1) {
  let c = n;
  for (let k = 0; k < 8; k += 1) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  crcTable[n] = c >>> 0;
}

function cliOption(name) {
  const index = process.argv.indexOf(name);
  if (index !== -1) return process.argv[index + 1];

  const prefix = `${name}=`;
  const value = process.argv.find((arg) => arg.startsWith(prefix));
  return value ? value.slice(prefix.length) : undefined;
}

function numericGeoKey(geoKeys, key) {
  const value = Number(geoKeys?.[key]);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function projectedEpsgFromGeoKeys(geoKeys) {
  return (
    numericGeoKey(geoKeys, "ProjectedCSTypeGeoKey") ??
    numericGeoKey(geoKeys, "ProjectedCRSGeoKey")
  );
}

function geographicEpsgFromGeoKeys(geoKeys) {
  return numericGeoKey(geoKeys, "GeographicTypeGeoKey");
}

function resolveSourceCrs(geoKeys, filePath) {
  const modelType = numericGeoKey(geoKeys, "GTModelTypeGeoKey");
  const projectedEpsg = projectedEpsgFromGeoKeys(geoKeys);
  const geographicEpsg = geographicEpsgFromGeoKeys(geoKeys);

  if (projectedEpsg || modelType === 1) {
    const utmZone = resolveUtmZone(geoKeys, filePath);
    UTM_ZONE = utmZone;
    return {
      kind: "utm",
      units: "metres",
      epsg: projectedEpsg,
      utmZone,
      horizontal: projectedEpsg
        ? `EPSG:${projectedEpsg} / UTM zone ${utmZone} projected metres`
        : `UTM zone ${utmZone} projected metres`,
    };
  }

  if (geographicEpsg || modelType === 2) {
    return {
      kind: "geographic",
      units: "degrees",
      epsg: geographicEpsg,
      utmZone: null,
      horizontal: geographicEpsg
        ? `EPSG:${geographicEpsg} geographic degrees`
        : "geographic degrees",
    };
  }

  const zoneFromPath = inferUtmZoneFromPath(filePath);
  if (zoneFromPath || (Number.isFinite(UTM_ZONE) && UTM_ZONE >= 1 && UTM_ZONE <= 60)) {
    const utmZone = zoneFromPath ?? UTM_ZONE;
    UTM_ZONE = utmZone;
    return {
      kind: "utm",
      units: "metres",
      epsg: null,
      utmZone,
      horizontal: `UTM zone ${utmZone} projected metres`,
    };
  }

  throw new Error(`Could not infer GeoTIFF CRS from ${filePath}`);
}

function inferUtmZoneFromGeoKeys(geoKeys) {
  const projectedCode =
    Number(geoKeys?.ProjectedCSTypeGeoKey) ||
    Number(geoKeys?.ProjectedCRSGeoKey) ||
    Number.NaN;
  if (!Number.isFinite(projectedCode)) return null;

  for (const base of [32600, 32700, 26900]) {
    const zone = projectedCode - base;
    if (zone >= 1 && zone <= 60) return zone;
  }

  return null;
}

function inferUtmZoneFromPath(filePath) {
  const match = String(filePath).match(/[_-]utm(\d{1,2})(?:[_./-]|$)/i);
  if (!match) return null;

  const zone = Number(match[1]);
  if (zone >= 1 && zone <= 60) return zone;
  return null;
}

function resolveUtmZone(geoKeys, filePath) {
  if (Number.isFinite(UTM_ZONE) && UTM_ZONE >= 1 && UTM_ZONE <= 60) {
    return UTM_ZONE;
  }

  const inferredZone = inferUtmZoneFromGeoKeys(geoKeys);
  if (inferredZone) return inferredZone;

  const zoneFromPath = inferUtmZoneFromPath(filePath);
  if (zoneFromPath) return zoneFromPath;

  throw new Error(
    `Could not infer a UTM zone from ${filePath}. Pass --utm-zone <zone> explicitly.`
  );
}

function resolveInputDir() {
  const requestedInputDir = cliOption("--input-dir") ?? cliOption("--input");
  if (requestedInputDir) {
    return path.resolve(process.cwd(), requestedInputDir);
  }

  if (directoryHasGeoTiffs(rawOpenTopographyDemInputDir)) return rawOpenTopographyDemInputDir;
  if (directoryHasGeoTiffs(rawLidarInputDir)) return rawLidarInputDir;
  if (directoryHasGeoTiffs(rawUsgsLidarInputDir)) return rawUsgsLidarInputDir;
  if (directoryHasGeoTiffs(defaultInputDir)) return defaultInputDir;
  if (directoryHasGeoTiffs(fallbackInputDir)) return fallbackInputDir;
  return path.dirname(legacySingleInput);
}

function directoryHasGeoTiffs(inputDir) {
  if (!existsSync(inputDir)) return false;

  try {
    return readdirSync(inputDir, { withFileTypes: true }).some(
      (entry) => entry.isFile() && /\.tiff?$/i.test(entry.name)
    );
  } catch {
    return false;
  }
}

async function listGeoTiffPaths(inputDir) {
  const entries = await readdir(inputDir, { withFileTypes: true });
  const tiffPaths = entries
    .filter((entry) => entry.isFile() && /\.tiff?$/i.test(entry.name))
    .map((entry) => path.join(inputDir, entry.name))
    .sort((left, right) => left.localeCompare(right));

  if (tiffPaths.length > 0) return tiffPaths;
  if (existsSync(legacySingleInput)) return [legacySingleInput];

  throw new Error(`No GeoTIFF files found in ${inputDir}`);
}

function centralMeridianRad(zone) {
  return Cesium.Math.toRadians((zone - 1) * 6 - 180 + 3);
}

function utmToLonLat(easting, northing, zone) {
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

function lonLatToUtm(lonDeg, latDeg, zone) {
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

function isValidRasterValue(value, noData) {
  return Number.isFinite(value) && (noData === null || value !== noData);
}

function rasterStats(data, noData) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;

  for (let i = 0; i < data.length; i += 1) {
    const value = Number(data[i]);
    if (!isValidRasterValue(value, noData)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    count += 1;
  }

  return {
    min: count > 0 ? min : FALLBACK_HEIGHT_M,
    max: count > 0 ? max : FALLBACK_HEIGHT_M,
    mean: count > 0 ? sum / count : FALLBACK_HEIGHT_M,
    sum,
    count,
  };
}

function aggregateRasterStats(statsByTile) {
  const validStats = statsByTile.filter((stats) => stats.count > 0);
  const count = validStats.reduce((total, stats) => total + stats.count, 0);
  const sum = validStats.reduce((total, stats) => total + stats.sum, 0);

  return {
    min: validStats.length > 0 ? Math.min(...validStats.map((stats) => stats.min)) : FALLBACK_HEIGHT_M,
    max: validStats.length > 0 ? Math.max(...validStats.map((stats) => stats.max)) : FALLBACK_HEIGHT_M,
    mean: count > 0 ? sum / count : FALLBACK_HEIGHT_M,
    sum,
    count,
  };
}

function combineSourceBounds(tiles) {
  return [
    Math.min(...tiles.map((tile) => tile.sourceBounds[0])),
    Math.min(...tiles.map((tile) => tile.sourceBounds[1])),
    Math.max(...tiles.map((tile) => tile.sourceBounds[2])),
    Math.max(...tiles.map((tile) => tile.sourceBounds[3])),
  ];
}

function sourceBoundsToObject(sourceCrs, [west, south, east, north]) {
  return { crs: sourceCrs.horizontal, west, south, east, north };
}

function mosaicRasterDescriptor(sourceBounds, resolution) {
  const resolutionX = Math.abs(resolution[0] || 1);
  const resolutionY = Math.abs(resolution[1] || resolutionX);

  return {
    width: Math.max(1, Math.round((sourceBounds[2] - sourceBounds[0]) / resolutionX)),
    height: Math.max(1, Math.round((sourceBounds[3] - sourceBounds[1]) / resolutionY)),
    resolutionX,
    resolutionY,
  };
}

async function readDsmSourceTile(filePath, index) {
  const tiff = await fromFile(filePath);
  const image = await tiff.getImage();
  const noData = image.getGDALNoData();
  const raster = await image.readRasters({ samples: [0], interleave: true });

  return {
    id: index,
    fileName: path.basename(filePath),
    path: filePath,
    width: image.getWidth(),
    height: image.getHeight(),
    origin: image.getOrigin(),
    resolution: image.getResolution(),
    sourceBounds: image.getBoundingBox(),
    geoKeys: image.getGeoKeys ? image.getGeoKeys() : {},
    noData,
    raster,
    stats: rasterStats(raster, noData),
  };
}

function clampIndex(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function createSourceTileIndex(tiles, sourceBounds, sourceCrs) {
  const [west, south, east, north] = sourceBounds;
  const cellSize =
    sourceCrs.kind === "geographic"
      ? SOURCE_TILE_INDEX_CELL_SIZE_DEGREES
      : SOURCE_TILE_INDEX_CELL_SIZE_M;
  const width = Math.max(1, Math.ceil((east - west) / cellSize));
  const height = Math.max(1, Math.ceil((north - south) / cellSize));
  const cells = new Map();

  const addTileToCell = (x, y, tile) => {
    const key = `${x},${y}`;
    const cellTiles = cells.get(key);
    if (cellTiles) {
      cellTiles.push(tile);
    } else {
      cells.set(key, [tile]);
    }
  };

  for (const tile of tiles) {
    const [tileWest, tileSouth, tileEast, tileNorth] = tile.sourceBounds;
    const minX = clampIndex(Math.floor((tileWest - west) / cellSize), 0, width - 1);
    const maxX = clampIndex(Math.floor((tileEast - west) / cellSize), 0, width - 1);
    const minY = clampIndex(Math.floor((tileSouth - south) / cellSize), 0, height - 1);
    const maxY = clampIndex(Math.floor((tileNorth - south) / cellSize), 0, height - 1);

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        addTileToCell(x, y, tile);
      }
    }
  }

  return { sourceBounds, cellSize, width, height, cells };
}

function createDsmDataset(tiles, inputDir, sourceCrs) {
  if (tiles.length === 0) {
    throw new Error("Cannot create height dataset without source tiles");
  }

  const sourceBounds = combineSourceBounds(tiles);
  const resolution = tiles[0].resolution;
  const raster = mosaicRasterDescriptor(sourceBounds, resolution);
  const stats = aggregateRasterStats(tiles.map((tile) => tile.stats));

  return {
    inputDir,
    tiles,
    sourceCrs,
    sourceBounds,
    resolution,
    raster,
    stats,
    tileIndex: createSourceTileIndex(tiles, sourceBounds, sourceCrs),
  };
}

function crc32(buffer) {
  let c = 0xffffffff;
  for (let i = 0; i < buffer.length; i += 1) {
    c = crcTable[(c ^ buffer[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBuffer = Buffer.from(type);
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  typeBuffer.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 8 + data.length);
  return chunk;
}

function encodePngRgba(width, height, rgba) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * (width * 4 + 1);
    raw[rowStart] = 0;
    rgba.copy(raw, rowStart + 1, y * width * 4, (y + 1) * width * 4);
  }

  return Buffer.concat([
    signature,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rampColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const stops = [
    [0, [25, 91, 161]],
    [0.35, [25, 151, 130]],
    [0.7, [238, 205, 82]],
    [1, [205, 70, 62]],
  ];
  let left = stops[0];
  let right = stops[stops.length - 1];
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      left = stops[i - 1];
      right = stops[i];
      break;
    }
  }

  const u = (t - left[0]) / (right[0] - left[0] || 1);
  return [
    Math.round(lerp(left[1][0], right[1][0], u)),
    Math.round(lerp(left[1][1], right[1][1], u)),
    Math.round(lerp(left[1][2], right[1][2], u)),
  ];
}

function createOverlayPng(dataset, stats, bounds) {
  const overlayWidth = Math.min(OVERLAY_MAX_WIDTH, dataset.raster.width);
  const overlayHeight = Math.max(1, Math.round((dataset.raster.height / dataset.raster.width) * overlayWidth));
  const rgba = Buffer.alloc(overlayWidth * overlayHeight * 4);

  for (let y = 0; y < overlayHeight; y += 1) {
    const v = overlayHeight === 1 ? 0 : y / (overlayHeight - 1);
    const lat = lerp(bounds.north, bounds.south, v);
    for (let x = 0; x < overlayWidth; x += 1) {
      const u = overlayWidth === 1 ? 0 : x / (overlayWidth - 1);
      const lon = lerp(bounds.west, bounds.east, u);
      const value = sampleDatasetAtLonLatOrNull(dataset, lon, lat);
      const offset = (y * overlayWidth + x) * 4;

      if (value === null) {
        rgba[offset + 3] = 0;
        continue;
      }

      const [r, g, b] = rampColor(value, stats.min, stats.max);
      rgba[offset] = r;
      rgba[offset + 1] = g;
      rgba[offset + 2] = b;
      rgba[offset + 3] = 210;
    }
  }

  return {
    width: overlayWidth,
    height: overlayHeight,
    png: encodePngRgba(overlayWidth, overlayHeight, rgba),
  };
}

function createOriginalTifHeatmapPng(dataset, stats) {
  const heatmapWidth = Math.min(OVERLAY_MAX_WIDTH, dataset.raster.width);
  const heatmapHeight = Math.max(1, Math.round((dataset.raster.height / dataset.raster.width) * heatmapWidth));
  const rgba = Buffer.alloc(heatmapWidth * heatmapHeight * 4);
  const [west, south, east, north] = dataset.sourceBounds;

  for (let y = 0; y < heatmapHeight; y += 1) {
    const v = heatmapHeight === 1 ? 0 : y / (heatmapHeight - 1);
    const sourceY = lerp(north, south, v);

    for (let x = 0; x < heatmapWidth; x += 1) {
      const u = heatmapWidth === 1 ? 0 : x / (heatmapWidth - 1);
      const sourceX = lerp(west, east, u);
      const value = sampleDatasetAtSourceOrNull(dataset, sourceX, sourceY);
      const offset = (y * heatmapWidth + x) * 4;

      if (value === null) {
        rgba[offset + 3] = 0;
        continue;
      }

      const [r, g, b] = rampColor(value, stats.min, stats.max);
      rgba[offset] = r;
      rgba[offset + 1] = g;
      rgba[offset + 2] = b;
      rgba[offset + 3] = 220;
    }
  }

  return {
    width: heatmapWidth,
    height: heatmapHeight,
    png: encodePngRgba(heatmapWidth, heatmapHeight, rgba),
  };
}

function lonLatBoundsFromSourceBounds(sourceCrs, [west, south, east, north]) {
  const corners =
    sourceCrs.kind === "geographic"
      ? [
          { name: "northWest", lonLat: [west, north] },
          { name: "northEast", lonLat: [east, north] },
          { name: "southEast", lonLat: [east, south] },
          { name: "southWest", lonLat: [west, south] },
        ]
      : [
          { name: "northWest", lonLat: utmToLonLat(west, north, sourceCrs.utmZone) },
          { name: "northEast", lonLat: utmToLonLat(east, north, sourceCrs.utmZone) },
          { name: "southEast", lonLat: utmToLonLat(east, south, sourceCrs.utmZone) },
          { name: "southWest", lonLat: utmToLonLat(west, south, sourceCrs.utmZone) },
        ];

  return {
    corners,
    bounds: {
      west: Math.min(...corners.map(({ lonLat: [lon] }) => lon)),
      south: Math.min(...corners.map(({ lonLat: [, lat] }) => lat)),
      east: Math.max(...corners.map(({ lonLat: [lon] }) => lon)),
      north: Math.max(...corners.map(({ lonLat: [, lat] }) => lat)),
    },
  };
}

function tileRangeForBounds(tilingScheme, bounds, level) {
  const cartographics = [
    Cesium.Cartographic.fromDegrees(bounds.west, bounds.north),
    Cesium.Cartographic.fromDegrees(bounds.east, bounds.north),
    Cesium.Cartographic.fromDegrees(bounds.east, bounds.south),
    Cesium.Cartographic.fromDegrees(bounds.west, bounds.south),
  ];
  const tileCoordinates = cartographics
    .map((position) => tilingScheme.positionToTileXY(position, level))
    .filter(Boolean);

  return {
    level,
    minX: Math.min(...tileCoordinates.map(({ x }) => x)),
    maxX: Math.max(...tileCoordinates.map(({ x }) => x)),
    minY: Math.min(...tileCoordinates.map(({ y }) => y)),
    maxY: Math.max(...tileCoordinates.map(({ y }) => y)),
  };
}

function sampleRasterAtSourceOrNull(dsm, sourceX, sourceY) {
  // image.getOrigin() describes the outer raster origin. Subtracting 0.5 is the
  // inverse of the pixel-center mapping used when writing coordinates out.
  const x = (sourceX - dsm.origin[0]) / dsm.resolution[0] - 0.5;
  const y = (sourceY - dsm.origin[1]) / dsm.resolution[1] - 0.5;

  if (x < 0 || y < 0 || x > dsm.width - 1 || y > dsm.height - 1) {
    return null;
  }

  const x0 = Math.max(0, Math.min(dsm.width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(dsm.height - 1, Math.floor(y)));
  const x1 = Math.min(x0 + 1, dsm.width - 1);
  const y1 = Math.min(y0 + 1, dsm.height - 1);
  const tx = x - x0;
  const ty = y - y0;

  const valueAt = (col, row) => {
    const value = Number(dsm.raster[row * dsm.width + col]);
    return isValidRasterValue(value, dsm.noData) ? value : null;
  };

  const topLeft = valueAt(x0, y0);
  const topRight = valueAt(x1, y0);
  const bottomLeft = valueAt(x0, y1);
  const bottomRight = valueAt(x1, y1);

  const weightedSamples = [
    [topLeft, (1 - tx) * (1 - ty)],
    [topRight, tx * (1 - ty)],
    [bottomLeft, (1 - tx) * ty],
    [bottomRight, tx * ty],
  ].filter(([value]) => value !== null);

  if (weightedSamples.length === 0) {
    return null;
  }

  const weightSum = weightedSamples.reduce((sum, [, weight]) => sum + weight, 0);
  return weightedSamples.reduce((sum, [value, weight]) => sum + value * weight, 0) / weightSum;
}

function sourceTilesForCoordinate(dataset, sourceX, sourceY) {
  const [west, south, east, north] = dataset.sourceBounds;
  if (sourceX < west || sourceX > east || sourceY < south || sourceY > north) {
    return [];
  }

  const { cellSize, width, height, cells } = dataset.tileIndex;
  const x = clampIndex(Math.floor((sourceX - west) / cellSize), 0, width - 1);
  const y = clampIndex(Math.floor((sourceY - south) / cellSize), 0, height - 1);

  return cells.get(`${x},${y}`) ?? [];
}

function sampleDatasetAtSourceOrNull(dataset, sourceX, sourceY) {
  // Indexing candidates in source CRS space keeps every Cesium terrain sample
  // from scanning every source GeoTIFF.
  const candidates = sourceTilesForCoordinate(dataset, sourceX, sourceY);
  if (candidates.length === 0) return null;

  let sum = 0;
  let count = 0;
  for (const tile of candidates) {
    const value = sampleRasterAtSourceOrNull(tile, sourceX, sourceY);
    if (value === null) continue;
    sum += value;
    count += 1;
  }

  // Source tiles can overlap by a few metres. Averaging valid samples gives a
  // deterministic seam rule without guessing that one survey date is preferred.
  return count > 0 ? sum / count : null;
}

function sourceCoordinateFromLonLat(dataset, lonDeg, latDeg) {
  if (dataset.sourceCrs.kind === "geographic") {
    return [lonDeg, latDeg];
  }

  return lonLatToUtm(lonDeg, latDeg, dataset.sourceCrs.utmZone);
}

function sampleDatasetAtLonLatOrNull(dataset, lonDeg, latDeg) {
  const [sourceX, sourceY] = sourceCoordinateFromLonLat(dataset, lonDeg, latDeg);
  return sampleDatasetAtSourceOrNull(dataset, sourceX, sourceY);
}

function sampleDatasetAtLonLat(dataset, lonDeg, latDeg) {
  return sampleDatasetAtLonLatOrNull(dataset, lonDeg, latDeg) ?? FALLBACK_HEIGHT_M;
}

function buildHeightTile(dataset, tilingScheme, x, y, level) {
  const rectangle = tilingScheme.tileXYToRectangle(x, y, level);
  const heights = new Float32Array(TILE_SIZE * TILE_SIZE);

  for (let row = 0; row < TILE_SIZE; row += 1) {
    const v = TILE_SIZE === 1 ? 0 : row / (TILE_SIZE - 1);
    const latRad = Cesium.Math.lerp(rectangle.north, rectangle.south, v);
    const latDeg = Cesium.Math.toDegrees(latRad);

    for (let col = 0; col < TILE_SIZE; col += 1) {
      const u = TILE_SIZE === 1 ? 0 : col / (TILE_SIZE - 1);
      const lonRad = Cesium.Math.lerp(rectangle.west, rectangle.east, u);
      const lonDeg = Cesium.Math.toDegrees(lonRad);
      heights[row * TILE_SIZE + col] = sampleDatasetAtLonLat(dataset, lonDeg, latDeg);
    }
  }

  return heights;
}

function float32ArrayToLittleEndianBuffer(heights) {
  const buffer = Buffer.alloc(heights.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  for (let i = 0; i < heights.length; i += 1) {
    view.setFloat32(i * Float32Array.BYTES_PER_ELEMENT, heights[i], true);
  }
  return buffer;
}

async function writeHeightTile(dataset, tilingScheme, x, y, level) {
  const heights = buildHeightTile(dataset, tilingScheme, x, y, level);
  const tileDir = path.join(tilesDir, String(level), String(x));
  await mkdir(tileDir, { recursive: true });
  await writeFile(path.join(tileDir, `${y}.f32`), float32ArrayToLittleEndianBuffer(heights));
}

async function main() {
  const inputDir = resolveInputDir();
  const inputPaths = await listGeoTiffPaths(inputDir);
  const sourceTiles = [];

  console.log(`Reading ${inputPaths.length} height GeoTIFF source tile(s) from ${path.relative(repoRoot, inputDir)}`);
  for (let index = 0; index < inputPaths.length; index += 1) {
    sourceTiles.push(await readDsmSourceTile(inputPaths[index], index));
    if ((index + 1) % 10 === 0 || index + 1 === inputPaths.length) {
      console.log(`  loaded ${index + 1}/${inputPaths.length}`);
    }
  }

  const sourceCrs = resolveSourceCrs(sourceTiles[0]?.geoKeys, sourceTiles[0]?.path ?? inputDir);
  const dataset = createDsmDataset(sourceTiles, inputDir, sourceCrs);
  const { bounds, corners } = lonLatBoundsFromSourceBounds(dataset.sourceCrs, dataset.sourceBounds);
  const stats = dataset.stats;
  const overlay = createOverlayPng(dataset, stats, bounds);
  const originalTifHeatmap = createOriginalTifHeatmapPng(dataset, stats);
  const tilingScheme = new Cesium.GeographicTilingScheme();
  const levels = [];
  let tileCount = 0;

  await mkdir(outputDir, { recursive: true });
  await rm(tilesDir, { recursive: true, force: true });
  await writeFile(path.join(outputDir, "dsm_height_overlay.png"), overlay.png);
  await writeFile(path.join(outputDir, "dsm_original_tif_heatmap.png"), originalTifHeatmap.png);
  for (let level = MIN_LEVEL; level <= MAX_LEVEL; level += 1) {
    const range = tileRangeForBounds(tilingScheme, bounds, level);
    let levelTileCount = 0;

    for (let x = range.minX; x <= range.maxX; x += 1) {
      for (let y = range.minY; y <= range.maxY; y += 1) {
        await writeHeightTile(dataset, tilingScheme, x, y, level);
        levelTileCount += 1;
      }
    }

    tileCount += levelTileCount;
    levels.push({
      level,
      xRange: [range.minX, range.maxX],
      yRange: [range.minY, range.maxY],
      tileCount: levelTileCount,
    });
  }

  await writeFile(
    path.join(outputDir, "metadata.json"),
    JSON.stringify(
      {
        input: path.relative(repoRoot, inputDir),
        inputs: sourceTiles.map((tile) => ({
          path: path.relative(repoRoot, tile.path),
          width: tile.width,
          height: tile.height,
          noData: tile.noData,
          resolution: {
            x: tile.resolution[0],
            y: tile.resolution[1],
          },
          sourceBounds: sourceBoundsToObject(dataset.sourceCrs, tile.sourceBounds),
          ...(dataset.sourceCrs.kind === "utm"
            ? { projectedBounds: sourceBoundsToObject(dataset.sourceCrs, tile.sourceBounds) }
            : {}),
          stats: {
            min: tile.stats.min,
            max: tile.stats.max,
            mean: tile.stats.mean,
            validSampleCount: tile.stats.count,
          },
        })),
        format: "float32-little-endian-heightmap",
        tileWidth: TILE_SIZE,
        tileHeight: TILE_SIZE,
        tilingScheme: "geographic",
        tilesBaseUrl: `/data/airports/${AIRPORT_CODE}/dsm/heightmap-terrain/tiles`,
        overlay: {
          url: `/data/airports/${AIRPORT_CODE}/dsm/heightmap-terrain/dsm_height_overlay.png`,
          width: overlay.width,
          height: overlay.height,
          note: "Height tint for visual inspection; the terrain provider still supplies the actual heights.",
        },
        originalTifHeatmap: {
          url: `/data/airports/${AIRPORT_CODE}/dsm/heightmap-terrain/dsm_original_tif_heatmap.png`,
          width: originalTifHeatmap.width,
          height: originalTifHeatmap.height,
          note:
            "Raw GeoTIFF pixel heatmap, draped over the source lon/lat bounds for source-data inspection.",
        },
        minLevel: MIN_LEVEL,
        maxLevel: MAX_LEVEL,
        tileCount,
        fallbackHeightM: FALLBACK_HEIGHT_M,
        raster: {
          width: dataset.raster.width,
          height: dataset.raster.height,
          sourceTileCount: sourceTiles.length,
          validSampleCount: stats.count,
          noData: sourceTiles[0].noData,
        },
        sourceCrs: {
          kind: dataset.sourceCrs.kind,
          horizontal: dataset.sourceCrs.horizontal,
          units: dataset.sourceCrs.units,
          epsg: dataset.sourceCrs.epsg,
          vertical: "Source GeoTIFF elevation values, used directly as metres",
        },
        sourceBounds: sourceBoundsToObject(dataset.sourceCrs, dataset.sourceBounds),
        ...(dataset.sourceCrs.kind === "utm"
          ? { projectedBounds: sourceBoundsToObject(dataset.sourceCrs, dataset.sourceBounds) }
          : {}),
        bounds,
        corners: Object.fromEntries(
          corners.map(({ name, lonLat: [lon, lat] }) => [name, { lon, lat }])
        ),
        levels,
        stats,
      },
      null,
      2
    )
  );

  console.log(`Wrote ${tileCount} heightmap tiles to ${path.relative(repoRoot, outputDir)}`);
  console.log(
    `Mosaic ${dataset.raster.width} x ${dataset.raster.height} source ${dataset.sourceCrs.units} from ${sourceTiles.length} GeoTIFFs`
  );
  console.log(`Levels ${MIN_LEVEL}-${MAX_LEVEL}, ${TILE_SIZE} x ${TILE_SIZE} samples per tile`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
