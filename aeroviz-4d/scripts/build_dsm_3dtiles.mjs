import { mkdir, readdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";
import { fromFile } from "geotiff";
import Martini from "@mapbox/martini";
import { Accessor, Document, NodeIO, Primitive } from "@gltf-transform/core";
import * as Cesium from "cesium";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const AIRPORT_CODE = (cliOption("--airport") ?? "CYVR").toUpperCase();
const defaultInputDir = path.resolve(repoRoot, `public/data/airports/${AIRPORT_CODE}/dsm/source`);
const fallbackInput = path.resolve(
  repoRoot,
  `public/data/DSM/${AIRPORT_CODE}/bc_092g015_3_3_3_xli1m_utm10_20240217_20250425.tif`
);
const outputDir = path.resolve(repoRoot, `public/data/airports/${AIRPORT_CODE}/dsm/3dtiles`);

let UTM_ZONE = Number(cliOption("--utm-zone") ?? Number.NaN);
const GRID_SIZE = 513;
const OVERLAY_MAX_WIDTH = 1024;
const MAX_ERROR_M = 0.08;
const VERTICAL_EXAGGERATION = 80;
const EXACT_OVERLAY_CLEARANCE_M = 25;

const UTM_K0 = 0.9996;
const WGS84_A = 6378137.0;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2);

function cliOption(name) {
  const index = process.argv.indexOf(name);
  if (index !== -1) return process.argv[index + 1];

  const prefix = `${name}=`;
  const value = process.argv.find((arg) => arg.startsWith(prefix));
  return value ? value.slice(prefix.length) : undefined;
}

async function resolveInputPath() {
  const requestedInput = cliOption("--input");
  if (requestedInput) {
    return path.resolve(process.cwd(), requestedInput);
  }

  const requestedInputDir = cliOption("--input-dir");
  const inputDir = requestedInputDir
    ? path.resolve(process.cwd(), requestedInputDir)
    : defaultInputDir;
  if (existsSync(inputDir)) {
    const entries = await readdir(inputDir, { withFileTypes: true });
    const tiffPaths = entries
      .filter((entry) => entry.isFile() && /\.tiff?$/i.test(entry.name))
      .map((entry) => path.join(inputDir, entry.name))
      .sort((left, right) => left.localeCompare(right));
    if (tiffPaths.length > 0) return tiffPaths[0];
  }

  return fallbackInput;
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

function padBuffer(buffer, boundary, padByte) {
  const remainder = buffer.length % boundary;
  if (remainder === 0) return buffer;
  return Buffer.concat([buffer, Buffer.alloc(boundary - remainder, padByte)]);
}

const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n += 1) {
  let c = n;
  for (let k = 0; k < 8; k += 1) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  crcTable[n] = c >>> 0;
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
    [0.0, [21, 74, 141]],
    [0.35, [21, 160, 130]],
    [0.68, [238, 208, 75]],
    [1.0, [198, 62, 58]],
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

function createOverlayPng(raster, width, height, noData, stats) {
  const overlayWidth = Math.min(OVERLAY_MAX_WIDTH, width);
  const overlayHeight = Math.max(1, Math.round((height / width) * overlayWidth));
  const rgba = Buffer.alloc(overlayWidth * overlayHeight * 4);

  for (let y = 0; y < overlayHeight; y += 1) {
    const sourceY = (y / Math.max(overlayHeight - 1, 1)) * (height - 1);
    for (let x = 0; x < overlayWidth; x += 1) {
      const sourceX = (x / Math.max(overlayWidth - 1, 1)) * (width - 1);
      const nearestX = Math.max(0, Math.min(width - 1, Math.round(sourceX)));
      const nearestY = Math.max(0, Math.min(height - 1, Math.round(sourceY)));
      const nearest = Number(raster[nearestY * width + nearestX]);
      const offset = (y * overlayWidth + x) * 4;
      if (!Number.isFinite(nearest) || nearest === noData) {
        rgba[offset + 3] = 0;
        continue;
      }

      const value = sampleRaster(raster, width, height, noData, stats.min, sourceX, sourceY);
      const [r, g, b] = rampColor(value, stats.min, stats.max);
      rgba[offset] = r;
      rgba[offset + 1] = g;
      rgba[offset + 2] = b;
      rgba[offset + 3] = 190;
    }
  }

  return {
    width: overlayWidth,
    height: overlayHeight,
    png: encodePngRgba(overlayWidth, overlayHeight, rgba),
  };
}

function createB3dm(glb) {
  const featureJson = padBuffer(Buffer.from(JSON.stringify({ BATCH_LENGTH: 0 })), 8, 0x20);
  const glbPadded = padBuffer(Buffer.from(glb), 8, 0x00);
  const byteLength = 28 + featureJson.length + glbPadded.length;
  const header = Buffer.alloc(28);
  header.write("b3dm", 0);
  header.writeUInt32LE(1, 4);
  header.writeUInt32LE(byteLength, 8);
  header.writeUInt32LE(featureJson.length, 12);
  header.writeUInt32LE(0, 16);
  header.writeUInt32LE(0, 20);
  header.writeUInt32LE(0, 24);
  return Buffer.concat([header, featureJson, glbPadded]);
}

function projectedRasterPoint(origin, resolution, sourceX, sourceY) {
  return [
    origin[0] + (sourceX + 0.5) * resolution[0],
    origin[1] + (sourceY + 0.5) * resolution[1],
  ];
}

function localEnuFromProjected(easting, northing, height, inverseCenterTransform) {
  const [lon, lat] = utmToLonLat(easting, northing, UTM_ZONE);
  const world = Cesium.Cartesian3.fromDegrees(lon, lat, height);
  return Cesium.Matrix4.multiplyByPoint(
    inverseCenterTransform,
    world,
    new Cesium.Cartesian3()
  );
}

function positionBounds(positions) {
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[i + axis];
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }
  return { min, max };
}

async function createExactOverlayGlb(overlayPng, cornersLocal) {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positions = new Float32Array([
    cornersLocal.southWest.x,
    cornersLocal.southWest.y,
    cornersLocal.southWest.z,
    cornersLocal.southEast.x,
    cornersLocal.southEast.y,
    cornersLocal.southEast.z,
    cornersLocal.northEast.x,
    cornersLocal.northEast.y,
    cornersLocal.northEast.z,
    cornersLocal.northWest.x,
    cornersLocal.northWest.y,
    cornersLocal.northWest.z,
  ]);
  const texcoords = new Float32Array([
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
  ]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const positionAccessor = doc
    .createAccessor("POSITION", buffer)
    .setArray(positions)
    .setType(Accessor.Type.VEC3);
  const texcoordAccessor = doc
    .createAccessor("TEXCOORD_0", buffer)
    .setArray(texcoords)
    .setType(Accessor.Type.VEC2);
  const indexAccessor = doc
    .createAccessor("indices", buffer)
    .setArray(indices)
    .setType(Accessor.Type.SCALAR);
  const texture = doc
    .createTexture("DSM exact overlay texture")
    .setImage(overlayPng)
    .setMimeType("image/png");
  const material = doc
    .createMaterial("DSM exact overlay")
    .setBaseColorTexture(texture)
    .setBaseColorFactor([1, 1, 1, 0.72])
    .setAlphaMode("BLEND")
    .setMetallicFactor(0)
    .setRoughnessFactor(1)
    .setDoubleSided(true);
  const primitive = doc
    .createPrimitive()
    .setAttribute("POSITION", positionAccessor)
    .setAttribute("TEXCOORD_0", texcoordAccessor)
    .setIndices(indexAccessor)
    .setMode(Primitive.Mode.TRIANGLES)
    .setMaterial(material);
  const mesh = doc.createMesh(`${AIRPORT_CODE} DSM exact overlay`).addPrimitive(primitive);
  const scene = doc.createScene("DSM Exact Overlay Scene");
  scene.addChild(doc.createNode(`${AIRPORT_CODE} DSM exact overlay`).setMesh(mesh));
  doc.getRoot().setDefaultScene(scene);
  return new NodeIO().writeBinary(doc);
}

function rasterStats(data, noData) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 1) {
    const value = Number(data[i]);
    if (!Number.isFinite(value) || value === noData) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    count += 1;
  }
  return { min, max, mean: sum / count };
}

function sampleRaster(data, width, height, noData, fallback, x, y) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const tx = x - x0;
  const ty = y - y0;
  const valueAt = (col, row) => {
    const value = Number(data[row * width + col]);
    return Number.isFinite(value) && value !== noData ? value : fallback;
  };
  const top = valueAt(x0, y0) * (1 - tx) + valueAt(x1, y0) * tx;
  const bottom = valueAt(x0, y1) * (1 - tx) + valueAt(x1, y1) * tx;
  return top * (1 - ty) + bottom * ty;
}

function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  const fixedIndices = new Uint32Array(indices.length);
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  const c = [0, 0, 0];
  const ab = [0, 0, 0];
  const ac = [0, 0, 0];

  const read = (index, target) => {
    const offset = index * 3;
    target[0] = positions[offset];
    target[1] = positions[offset + 1];
    target[2] = positions[offset + 2];
  };

  for (let i = 0; i < indices.length; i += 3) {
    let ia = indices[i];
    let ib = indices[i + 1];
    let ic = indices[i + 2];
    read(ia, a);
    read(ib, b);
    read(ic, c);
    ab[0] = b[0] - a[0];
    ab[1] = b[1] - a[1];
    ab[2] = b[2] - a[2];
    ac[0] = c[0] - a[0];
    ac[1] = c[1] - a[1];
    ac[2] = c[2] - a[2];
    let nx = ab[1] * ac[2] - ab[2] * ac[1];
    let ny = ab[2] * ac[0] - ab[0] * ac[2];
    let nz = ab[0] * ac[1] - ab[1] * ac[0];
    if (nz < 0) {
      [ib, ic] = [ic, ib];
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    fixedIndices[i] = ia;
    fixedIndices[i + 1] = ib;
    fixedIndices[i + 2] = ic;
    for (const vertex of [ia, ib, ic]) {
      const offset = vertex * 3;
      normals[offset] += nx;
      normals[offset + 1] += ny;
      normals[offset + 2] += nz;
    }
  }

  for (let i = 0; i < normals.length; i += 3) {
    const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= length;
    normals[i + 1] /= length;
    normals[i + 2] /= length;
  }

  return { normals, indices: fixedIndices };
}

async function main() {
  const inputPath = await resolveInputPath();
  const tiff = await fromFile(inputPath);
  const image = await tiff.getImage();
  UTM_ZONE = resolveUtmZone(image.getGeoKeys ? image.getGeoKeys() : {}, inputPath);
  const width = image.getWidth();
  const height = image.getHeight();
  const origin = image.getOrigin();
  const resolution = image.getResolution();
  const [westEasting, southNorthing, eastEasting, northNorthing] = image.getBoundingBox();
  const noData = image.getGDALNoData();
  const raster = await image.readRasters({ samples: [0], interleave: true });
  const stats = rasterStats(raster, noData);

  const terrain = new Float32Array(GRID_SIZE * GRID_SIZE);
  for (let y = 0; y < GRID_SIZE; y += 1) {
    const sourceY = (y / (GRID_SIZE - 1)) * (height - 1);
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const sourceX = (x / (GRID_SIZE - 1)) * (width - 1);
      terrain[y * GRID_SIZE + x] = sampleRaster(
        raster,
        width,
        height,
        noData,
        stats.min,
        sourceX,
        sourceY
      );
    }
  }

  const martini = new Martini(GRID_SIZE);
  const mesh = martini.createTile(terrain).getMesh(MAX_ERROR_M);
  const positions = new Float32Array((mesh.vertices.length / 2) * 3);
  const centerEasting = (westEasting + eastEasting) / 2;
  const centerNorthing = (northNorthing + southNorthing) / 2;
  const [centerLon, centerLat] = utmToLonLat(centerEasting, centerNorthing, UTM_ZONE);
  const centerTransform = Cesium.Transforms.eastNorthUpToFixedFrame(
    Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0)
  );
  const inverseCenterTransform = Cesium.Matrix4.inverseTransformation(
    centerTransform,
    new Cesium.Matrix4()
  );
  const corners = [
    { name: "northWest", lonLat: utmToLonLat(westEasting, northNorthing, UTM_ZONE) },
    { name: "northEast", lonLat: utmToLonLat(eastEasting, northNorthing, UTM_ZONE) },
    { name: "southEast", lonLat: utmToLonLat(eastEasting, southNorthing, UTM_ZONE) },
    { name: "southWest", lonLat: utmToLonLat(westEasting, southNorthing, UTM_ZONE) },
  ];
  const bounds = {
    west: Math.min(...corners.map(({ lonLat: [lon] }) => lon)),
    south: Math.min(...corners.map(({ lonLat: [, lat] }) => lat)),
    east: Math.max(...corners.map(({ lonLat: [lon] }) => lon)),
    north: Math.max(...corners.map(({ lonLat: [, lat] }) => lat)),
  };

  for (let i = 0; i < mesh.vertices.length; i += 2) {
    const gridX = mesh.vertices[i];
    const gridY = mesh.vertices[i + 1];
    const sourceX = (gridX / (GRID_SIZE - 1)) * (width - 1);
    const sourceY = (gridY / (GRID_SIZE - 1)) * (height - 1);
    const [easting, northing] = projectedRasterPoint(origin, resolution, sourceX, sourceY);
    const heightM = sampleRaster(raster, width, height, noData, stats.min, sourceX, sourceY);
    const vertexIndex = i / 2;
    const local = localEnuFromProjected(
      easting,
      northing,
      (heightM - stats.min) * VERTICAL_EXAGGERATION,
      inverseCenterTransform
    );
    positions[vertexIndex * 3] = local.x;
    positions[vertexIndex * 3 + 1] = local.y;
    positions[vertexIndex * 3 + 2] = local.z;
  }

  const { normals, indices } = computeNormals(positions, mesh.triangles);
  const gltfIndices = Uint16Array.from(indices);
  const doc = new Document();
  const buffer = doc.createBuffer();
  const positionAccessor = doc
    .createAccessor("POSITION", buffer)
    .setArray(positions)
    .setType(Accessor.Type.VEC3);
  const normalAccessor = doc
    .createAccessor("NORMAL", buffer)
    .setArray(normals)
    .setType(Accessor.Type.VEC3);
  const indexAccessor = doc
    .createAccessor("indices", buffer)
    .setArray(gltfIndices)
    .setType(Accessor.Type.SCALAR);
  const material = doc
    .createMaterial("DSM surface")
    .setBaseColorFactor([0.48, 0.82, 0.5, 1])
    .setEmissiveFactor([0.18, 0.28, 0.12])
    .setMetallicFactor(0)
    .setRoughnessFactor(0.9)
    .setDoubleSided(true);
  const primitive = doc
    .createPrimitive()
    .setAttribute("POSITION", positionAccessor)
    .setAttribute("NORMAL", normalAccessor)
    .setIndices(indexAccessor)
    .setMode(Primitive.Mode.TRIANGLES)
    .setMaterial(material);
  const meshNode = doc.createMesh(`${AIRPORT_CODE} DSM`).addPrimitive(primitive);
  const scene = doc.createScene("DSM Scene");
  scene.addChild(doc.createNode(`${AIRPORT_CODE} DSM`).setMesh(meshNode));
  doc.getRoot().setDefaultScene(scene);

  const glb = await new NodeIO().writeBinary(doc);
  const b3dm = createB3dm(glb);
  const overlay = createOverlayPng(raster, width, height, noData, stats);
  const transform = Cesium.Matrix4.toArray(centerTransform);
  const localBounds = positionBounds(positions);
  const modelHeight = (stats.max - stats.min) * VERTICAL_EXAGGERATION;
  const exactOverlayHeight = modelHeight + EXACT_OVERLAY_CLEARANCE_M;
  const exactOverlayCorners = {
    northWest: localEnuFromProjected(
      westEasting,
      northNorthing,
      exactOverlayHeight,
      inverseCenterTransform
    ),
    northEast: localEnuFromProjected(
      eastEasting,
      northNorthing,
      exactOverlayHeight,
      inverseCenterTransform
    ),
    southEast: localEnuFromProjected(
      eastEasting,
      southNorthing,
      exactOverlayHeight,
      inverseCenterTransform
    ),
    southWest: localEnuFromProjected(
      westEasting,
      southNorthing,
      exactOverlayHeight,
      inverseCenterTransform
    ),
  };
  const exactOverlayGlb = await createExactOverlayGlb(overlay.png, exactOverlayCorners);
  const boxCenter = localBounds.min.map((value, index) => (value + localBounds.max[index]) / 2);
  const halfAxes = localBounds.min.map((value, index) =>
    Math.max((localBounds.max[index] - value) / 2, 1)
  );
  const tileset = {
    asset: {
      version: "1.0",
      gltfUpAxis: "Z",
      generator: "scripts/build_dsm_3dtiles.mjs",
    },
    geometricError: 0,
    root: {
      boundingVolume: {
        box: [
          boxCenter[0],
          boxCenter[1],
          boxCenter[2],
          halfAxes[0],
          0,
          0,
          0,
          halfAxes[1],
          0,
          0,
          0,
          halfAxes[2],
        ],
      },
      transform,
      geometricError: 0,
      refine: "ADD",
      content: {
        uri: "dsm.b3dm",
      },
    },
  };

  await mkdir(outputDir, { recursive: true });
  await writeFile(path.join(outputDir, "dsm_overlay.png"), overlay.png);
  await writeFile(path.join(outputDir, "dsm_overlay_exact.glb"), Buffer.from(exactOverlayGlb));
  await writeFile(path.join(outputDir, "dsm.glb"), Buffer.from(glb));
  await writeFile(path.join(outputDir, "dsm.b3dm"), b3dm);
  await writeFile(path.join(outputDir, "tileset.json"), JSON.stringify(tileset, null, 2));
  await writeFile(
    path.join(outputDir, "metadata.json"),
    JSON.stringify(
      {
        input: path.relative(repoRoot, inputPath),
        raster: { width, height },
        overlay: {
          url: `/data/airports/${AIRPORT_CODE}/dsm/3dtiles/dsm_overlay.png`,
          width: overlay.width,
          height: overlay.height,
        },
        exactOverlay: {
          url: `/data/airports/${AIRPORT_CODE}/dsm/3dtiles/dsm_overlay_exact.glb`,
          heightAboveBaseM: exactOverlayHeight,
          note: "Textured overlay plane generated from the same projected-corner transform as the DSM mesh.",
        },
        center: { lon: centerLon, lat: centerLat },
        corners: Object.fromEntries(
          corners.map(({ name, lonLat: [lon, lat] }) => [name, { lon, lat }])
        ),
        bounds,
        projectedBounds: {
          crs: `UTM zone ${UTM_ZONE} projected metres`,
          west: westEasting,
          south: southNorthing,
          east: eastEasting,
          north: northNorthing,
        },
        localBounds,
        gridSize: GRID_SIZE,
        maxErrorM: MAX_ERROR_M,
        vertices: mesh.vertices.length / 2,
        triangles: mesh.triangles.length / 3,
        stats,
        modelAxes: {
          x: "local ENU east, computed by projected metres -> lon/lat -> ECEF -> ENU",
          y: "local ENU north, computed by projected metres -> lon/lat -> ECEF -> ENU",
          z: "local ENU up, vertically exaggerated DSM height above min",
          cesiumUpAxis: "Z",
          cesiumForwardAxis: "X",
        },
        verticalExaggeration: VERTICAL_EXAGGERATION,
      },
      null,
      2
    )
  );
  console.log(`Wrote ${path.relative(repoRoot, outputDir)}/tileset.json`);
  console.log(`${mesh.vertices.length / 2} vertices, ${mesh.triangles.length / 3} triangles`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
