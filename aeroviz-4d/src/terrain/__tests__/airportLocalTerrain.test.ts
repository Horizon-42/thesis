import { describe, expect, it, vi } from "vitest";

const { fetchJson } = vi.hoisted(() => ({
  fetchJson: vi.fn(),
}));

vi.mock("../../utils/fetchJson", () => ({
  fetchJson,
}));

vi.mock("cesium", () => ({
  GeographicTilingScheme: class GeographicTilingScheme {},
}));

import {
  AirportLocalTerrainMetadataError,
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
