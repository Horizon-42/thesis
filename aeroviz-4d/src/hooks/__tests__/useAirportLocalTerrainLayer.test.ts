import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JsonFetchError } from "../../utils/fetchJson";

const {
  loadAirportLocalTerrain,
  setAirportLocalTerrain,
  mockViewer,
  getActiveAirportCode,
  setActiveAirportCode,
  providerByAirport,
  preloadTilesByAirport,
  focusTilesByAirport,
} = vi.hoisted(() => {
  const providerByAirport = {
    CYVR: { _tag: "local-terrain-CYVR" },
    KSJC: { _tag: "local-terrain-KSJC" },
  } as Record<string, any>;
  const tileCountByAirport = {
    CYVR: 12,
    KSJC: 10,
  } as Record<string, number>;
  const focusTilesByAirport = {
    CYVR: [{ level: 16, x: 20686, y: 14857 }],
    KSJC: [{ level: 16, x: 21142, y: 19164 }],
  } as Record<string, any[]>;
  const preloadTilesByAirport = {
    CYVR: vi.fn(({ tiles, onProgress } = {}) => {
      const totalTiles = tiles?.length ?? tileCountByAirport.CYVR;
      onProgress?.({ loadedTiles: 0, totalTiles });
      onProgress?.({ loadedTiles: totalTiles, totalTiles });
      return Promise.resolve();
    }),
    KSJC: vi.fn(({ tiles, onProgress } = {}) => {
      const totalTiles = tiles?.length ?? tileCountByAirport.KSJC;
      onProgress?.({ loadedTiles: totalTiles, totalTiles });
      return Promise.resolve();
    }),
  } as Record<string, any>;
  const loadAirportLocalTerrain = vi.fn((metadataUrl: string) => {
    const airportCode = metadataUrl.includes("KSJC") ? "KSJC" : "CYVR";
    return Promise.resolve({
      metadata: {
        tileCount: tileCountByAirport[airportCode],
        bounds: {
          west: airportCode === "KSJC" ? -121.99 : -123.28,
          south: airportCode === "KSJC" ? 37.31 : 49.13,
          east: airportCode === "KSJC" ? -121.87 : -123.09,
          north: airportCode === "KSJC" ? 37.41 : 49.25,
        },
        stats: {
          min: airportCode === "KSJC" ? 0.7 : -9,
          max: airportCode === "KSJC" ? 50.7 : 243.8,
        },
        precision: {
          horizontalResolutionM: airportCode === "KSJC" ? 10 : 1,
          source: "test",
        },
      },
      provider: providerByAirport[airportCode],
      preloadTiles: preloadTilesByAirport[airportCode],
    });
  });
  const setAirportLocalTerrain = vi.fn();
  const mockViewer = {
    scene: {
      terrainProvider: { _tag: "world-terrain" } as any,
      globe: {
        maximumScreenSpaceError: 2,
        tileCacheSize: 100,
        loadingDescendantLimit: 10,
        preloadSiblings: false,
        preloadAncestors: false,
        depthTestAgainstTerrain: false,
      },
      requestRender: vi.fn(),
    },
    isDestroyed: () => false,
  };
  let activeAirportCode = "CYVR";

  return {
    loadAirportLocalTerrain,
    setAirportLocalTerrain,
    mockViewer,
    getActiveAirportCode: () => activeAirportCode,
    setActiveAirportCode: (airportCode: string) => {
      activeAirportCode = airportCode;
    },
    providerByAirport,
    preloadTilesByAirport,
    focusTilesByAirport,
  };
});

vi.mock("cesium", () => ({
  EllipsoidTerrainProvider: class EllipsoidTerrainProvider {
    _tag = "ellipsoid";
  },
}));

vi.mock("../../terrain/airportLocalTerrain", () => ({
  airportLocalTerrainMetadataUrl: (airportCode: string) => (
    `/data/airports/${airportCode}/dsm/heightmap-terrain/metadata.json`
  ),
  airportLocalTerrainTileRefsNearCoordinate: (_metadata: any, lon: number) => (
    lon < -122 ? focusTilesByAirport.CYVR : focusTilesByAirport.KSJC
  ),
  loadAirportLocalTerrain,
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    activeAirportCode: getActiveAirportCode(),
    setAirportLocalTerrain,
  }),
}));

import { useAirportLocalTerrainLayer } from "../useAirportLocalTerrainLayer";

describe("useAirportLocalTerrainLayer", () => {
  beforeEach(() => {
    setActiveAirportCode("CYVR");
    mockViewer.scene.terrainProvider = { _tag: "world-terrain" } as any;
    mockViewer.scene.globe.maximumScreenSpaceError = 2;
    mockViewer.scene.globe.tileCacheSize = 100;
    mockViewer.scene.globe.loadingDescendantLimit = 10;
    mockViewer.scene.globe.preloadSiblings = false;
    mockViewer.scene.globe.preloadAncestors = false;
    mockViewer.scene.globe.depthTestAgainstTerrain = false;
    mockViewer.scene.requestRender.mockClear();
    setAirportLocalTerrain.mockClear();
    loadAirportLocalTerrain.mockClear();
    preloadTilesByAirport.CYVR.mockClear();
    preloadTilesByAirport.KSJC.mockClear();
    loadAirportLocalTerrain.mockImplementation((metadataUrl: string) => {
      const airportCode = metadataUrl.includes("KSJC") ? "KSJC" : "CYVR";
      return Promise.resolve({
        metadata: {
          tileCount: airportCode === "KSJC" ? 10 : 12,
          bounds: {
            west: airportCode === "KSJC" ? -121.99 : -123.28,
            south: airportCode === "KSJC" ? 37.31 : 49.13,
            east: airportCode === "KSJC" ? -121.87 : -123.09,
            north: airportCode === "KSJC" ? 37.41 : 49.25,
          },
          stats: {
            min: airportCode === "KSJC" ? 0.7 : -9,
            max: airportCode === "KSJC" ? 50.7 : 243.8,
          },
          precision: {
            horizontalResolutionM: airportCode === "KSJC" ? 10 : 1,
            source: "test",
          },
        },
        provider: providerByAirport[airportCode],
        preloadTiles: preloadTilesByAirport[airportCode],
      });
    });
  });

  it("preloads focused local terrain tiles before activating the airport provider", async () => {
    const { result } = renderHook(() => useAirportLocalTerrainLayer());

    await waitFor(() => expect(result.current.status).toBe("active"));

    expect(loadAirportLocalTerrain).toHaveBeenCalledWith(
      "/data/airports/CYVR/dsm/heightmap-terrain/metadata.json",
    );
    expect(preloadTilesByAirport.CYVR).toHaveBeenCalledWith(
      expect.objectContaining({
        concurrency: 24,
        tiles: focusTilesByAirport.CYVR,
      }),
    );
    expect(preloadTilesByAirport.CYVR).toHaveBeenCalledWith(
      expect.objectContaining({ concurrency: 8 }),
    );
    expect(mockViewer.scene.terrainProvider).toBe(providerByAirport.CYVR);
    expect(mockViewer.scene.globe.maximumScreenSpaceError).toBe(0.5);
    expect(mockViewer.scene.globe.tileCacheSize).toBe(256);
    expect(mockViewer.scene.globe.loadingDescendantLimit).toBe(1000);
    expect(mockViewer.scene.globe.preloadSiblings).toBe(true);
    expect(mockViewer.scene.globe.preloadAncestors).toBe(true);
    expect(mockViewer.scene.globe.depthTestAgainstTerrain).toBe(true);
    await waitFor(() => expect(result.current.loadedTiles).toBe(12));
    expect(result.current.totalTiles).toBe(12);
    expect(setAirportLocalTerrain).toHaveBeenLastCalledWith({
      status: "active",
      airportCode: "CYVR",
      sourceLabel: "Airport local heightmap terrain",
      minimumHeightM: -9,
      maximumHeightM: 243.8,
      loadedTiles: 12,
      totalTiles: 12,
      error: null,
    });
  });

  it("loads a separate cached provider per active airport", async () => {
    const { rerender, result } = renderHook(() => useAirportLocalTerrainLayer());

    await waitFor(() => expect(result.current.status).toBe("active"));
    expect(mockViewer.scene.terrainProvider).toBe(providerByAirport.CYVR);

    setActiveAirportCode("KSJC");
    rerender();
    await waitFor(() => expect(mockViewer.scene.terrainProvider).toBe(providerByAirport.KSJC));

    expect(loadAirportLocalTerrain).toHaveBeenCalledTimes(2);
    expect(loadAirportLocalTerrain).toHaveBeenLastCalledWith(
      "/data/airports/KSJC/dsm/heightmap-terrain/metadata.json",
    );
    expect(preloadTilesByAirport.KSJC).toHaveBeenCalled();
    expect(setAirportLocalTerrain).toHaveBeenLastCalledWith({
      status: "active",
      airportCode: "KSJC",
      sourceLabel: "Airport local heightmap terrain",
      minimumHeightM: 0.7,
      maximumHeightM: 50.7,
      loadedTiles: 10,
      totalTiles: 10,
      error: null,
    });
  });

  it("reports missing terrain metadata without replacing the existing provider", async () => {
    loadAirportLocalTerrain.mockRejectedValueOnce(
      new JsonFetchError("missing metadata", {
        url: "/data/airports/CYVR/dsm/heightmap-terrain/metadata.json",
        status: 404,
      }),
    );

    const { result } = renderHook(() => useAirportLocalTerrainLayer());

    await waitFor(() => expect(result.current.status).toBe("idle"));

    expect(mockViewer.scene.terrainProvider._tag).toBe("world-terrain");
    expect(preloadTilesByAirport.CYVR).not.toHaveBeenCalled();
    expect(setAirportLocalTerrain).toHaveBeenLastCalledWith({
      status: "missing",
      airportCode: "CYVR",
      sourceLabel: null,
      minimumHeightM: null,
      maximumHeightM: null,
      loadedTiles: 0,
      totalTiles: 0,
      error: null,
    });
  });
});
