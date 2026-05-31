import { describe, expect, it, vi } from "vitest";

const { fetchJson } = vi.hoisted(() => ({
  fetchJson: vi.fn(),
}));

vi.mock("../../utils/fetchJson", () => ({
  fetchJson,
}));

vi.mock("cesium", () => ({
  GeographicTilingScheme: class GeographicTilingScheme {},
  Rectangle: {
    fromDegrees: vi.fn((west, south, east, north) => ({ west, south, east, north })),
  },
}));

import {
  AirportLocalTerrainMetadataError,
  airportLocalTerrainImageryPlan,
  airportLocalTerrainDisplayFallbackHeight,
  fillAirportLocalTerrainFallbackHeights,
  shouldFillAirportLocalTerrainFallbackHeights,
  type AirportLocalTerrainMetadata,
  loadAirportLocalTerrain,
} from "../airportLocalTerrain";

describe("loadAirportLocalTerrain", () => {
  it("rejects old terrain metadata that lacks precision metadata", async () => {
    fetchJson.mockResolvedValueOnce({
      format: "float32-little-endian-heightmap",
      tileWidth: 129,
      tileHeight: 129,
      tilingScheme: "geographic",
      tilesBaseUrl: "/tiles",
      minLevel: 0,
      maxLevel: 16,
      tileCount: 1,
      fallbackHeightM: 0,
      raster: { width: 1, height: 1, noData: null },
      bounds: { west: 0, south: 0, east: 1, north: 1 },
      corners: {
        northWest: { lon: 0, lat: 1 },
        northEast: { lon: 1, lat: 1 },
        southEast: { lon: 1, lat: 0 },
        southWest: { lon: 0, lat: 0 },
      },
      levels: [],
      stats: { min: 0, max: 1, mean: 0.5 },
    });

    await expect(loadAirportLocalTerrain("/metadata.json")).rejects.toMatchObject({
      name: "AirportLocalTerrainMetadataError",
      code: "missing-precision-metadata",
    } satisfies Partial<AirportLocalTerrainMetadataError>);
  });
});

function metadata(
  overrides: Partial<AirportLocalTerrainMetadata> = {},
): AirportLocalTerrainMetadata {
  return {
    format: "float32-little-endian-heightmap",
    tileWidth: 3,
    tileHeight: 3,
    tilingScheme: "geographic",
    tilesBaseUrl: "/tiles",
    precision: {
      horizontalResolutionM: 2,
      source: "test",
    },
    minLevel: 0,
    maxLevel: 16,
    tileCount: 1,
    fallbackHeightM: 0,
    raster: { width: 3, height: 3, noData: null },
    bounds: { west: 0, south: 0, east: 1, north: 1 },
    corners: {
      northWest: { lon: 0, lat: 1 },
      northEast: { lon: 1, lat: 1 },
      southEast: { lon: 1, lat: 0 },
      southWest: { lon: 0, lat: 0 },
    },
    levels: [],
    stats: { min: 10, max: 20, mean: 15 },
    ...overrides,
  };
}

describe("fillAirportLocalTerrainFallbackHeights", () => {
  it("uses the source minimum for fallback when zero would create an edge cliff", () => {
    expect(airportLocalTerrainDisplayFallbackHeight(metadata())).toBe(10);
  });

  it("fills no-data edge samples from nearby valid heights", () => {
    const heights = new Float32Array([
      0, 0, 0,
      0, 12, 13,
      0, 14, 0,
    ]);

    fillAirportLocalTerrainFallbackHeights(metadata(), heights);

    expect([...heights]).toEqual([
      12, 12, 13,
      12, 12, 13,
      14, 14, 14,
    ]);
  });

  it("preserves valid zero-height samples when zero is within the source range", () => {
    const heights = new Float32Array([
      0, 1, 2,
      0, 3, 4,
      0, 5, 6,
    ]);

    fillAirportLocalTerrainFallbackHeights(
      metadata({
        fallbackHeightM: 0,
        stats: { min: -5, max: 10, mean: 2 },
      }),
      heights,
    );

    expect([...heights]).toEqual([
      0, 1, 2,
      0, 3, 4,
      0, 5, 6,
    ]);
  });

  it("only fills fallback samples near the highest-detail local terrain levels", () => {
    expect(shouldFillAirportLocalTerrainFallbackHeights(metadata({ maxLevel: 16 }), 15)).toBe(false);
    expect(shouldFillAirportLocalTerrainFallbackHeights(metadata({ maxLevel: 16 }), 16)).toBe(true);
  });
});

describe("airportLocalTerrainImageryPlan", () => {
  it("builds a bounded hillshade plan from terrain metadata", () => {
    const plan = airportLocalTerrainImageryPlan(
      metadata({
        hillshade: {
          url: "/hillshade.png",
          width: 1024,
          height: 768,
          alpha: 1.4,
          brightness: 2,
          contrast: 0.1,
          gamma: 0.2,
        },
      }),
      "hillshade",
    );

    expect(plan).toEqual({
      url: "/hillshade.png",
      tileWidth: 1024,
      tileHeight: 768,
      rectangle: { west: 0, south: 0, east: 1, north: 1 },
      credit: "Airport terrain hillshade",
      alpha: 1,
      brightness: 1.5,
      contrast: 0.5,
      gamma: 0.5,
    });
  });

  it("builds a height-tint plan with terrain package defaults", () => {
    const plan = airportLocalTerrainImageryPlan(
      metadata({
        overlay: {
          url: "/height-tint.png",
          width: 900,
          height: 700,
          alpha: -0.2,
        },
      }),
      "heightTint",
    );

    expect(plan).toEqual({
      url: "/height-tint.png",
      tileWidth: 900,
      tileHeight: 700,
      rectangle: { west: 0, south: 0, east: 1, north: 1 },
      credit: "Airport terrain height tint",
      alpha: 0,
      brightness: 1.05,
      contrast: 1.12,
      saturation: 0.9,
    });
  });
});
