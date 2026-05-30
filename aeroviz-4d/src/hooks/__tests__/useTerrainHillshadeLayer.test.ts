import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JsonFetchError } from "../../utils/fetchJson";

const {
  fetchJson,
  mockViewer,
  getActiveAirportCode,
  setActiveAirportCode,
  getTerrainHillshadeEnabled,
  setTerrainHillshadeEnabled,
  addImageryProvider,
  removeImageryLayer,
  requestRender,
} = vi.hoisted(() => {
  const addImageryProvider = vi.fn((provider: any) => ({
    _tag: "hillshade-layer",
    alpha: 1,
    provider,
  }));
  const removeImageryLayer = vi.fn();
  const requestRender = vi.fn();
  const mockViewer = {
    imageryLayers: {
      addImageryProvider,
      remove: removeImageryLayer,
    },
    scene: {
      requestRender,
    },
    isDestroyed: () => false,
  };
  const fetchJson = vi.fn();
  let activeAirportCode = "KRDU";
  let terrainHillshadeEnabled = true;

  return {
    fetchJson,
    mockViewer,
    getActiveAirportCode: () => activeAirportCode,
    setActiveAirportCode: (airportCode: string) => {
      activeAirportCode = airportCode;
    },
    getTerrainHillshadeEnabled: () => terrainHillshadeEnabled,
    setTerrainHillshadeEnabled: (enabled: boolean) => {
      terrainHillshadeEnabled = enabled;
    },
    addImageryProvider,
    removeImageryLayer,
    requestRender,
  };
});

vi.mock("cesium", () => ({
  Rectangle: {
    fromDegrees: vi.fn((west, south, east, north) => ({ west, south, east, north })),
  },
  SingleTileImageryProvider: class SingleTileImageryProvider {
    options: any;

    constructor(options: any) {
      this.options = options;
    }
  },
}));

vi.mock("../../utils/fetchJson", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../utils/fetchJson")>();
  return {
    ...actual,
    fetchJson,
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    activeAirportCode: getActiveAirportCode(),
    layers: {
      terrainHillshade: getTerrainHillshadeEnabled(),
    },
  }),
}));

import { useTerrainHillshadeLayer } from "../useTerrainHillshadeLayer";

function metadata(alpha = 0.42) {
  return {
    bounds: {
      west: -79.02,
      south: 35.69,
      east: -78.55,
      north: 36.07,
    },
    hillshade: {
      url: "/data/airports/KRDU/local-terrain/heightmap/local_terrain_hillshade.png",
      width: 1024,
      height: 830,
      alpha,
    },
  };
}

describe("useTerrainHillshadeLayer", () => {
  beforeEach(() => {
    setActiveAirportCode("KRDU");
    setTerrainHillshadeEnabled(true);
    fetchJson.mockReset();
    fetchJson.mockResolvedValue(metadata());
    addImageryProvider.mockClear();
    removeImageryLayer.mockClear();
    requestRender.mockClear();
  });

  it("loads the active airport hillshade as a bounded single-tile imagery layer", async () => {
    renderHook(() => useTerrainHillshadeLayer());

    await waitFor(() => expect(addImageryProvider).toHaveBeenCalledTimes(1));

    expect(fetchJson).toHaveBeenCalledWith(
      "/data/airports/KRDU/local-terrain/heightmap/metadata.json",
    );
    const provider = addImageryProvider.mock.calls[0][0];
    expect(provider.options).toEqual({
      url: "/data/airports/KRDU/local-terrain/heightmap/local_terrain_hillshade.png",
      tileWidth: 1024,
      tileHeight: 830,
      rectangle: {
        west: -79.02,
        south: 35.69,
        east: -78.55,
        north: 36.07,
      },
      credit: "Airport terrain hillshade",
    });
    expect(addImageryProvider.mock.results[0].value.alpha).toBe(0.42);
    expect(requestRender).toHaveBeenCalled();
  });

  it("removes the hillshade layer when the toggle is disabled", async () => {
    const { rerender } = renderHook(() => useTerrainHillshadeLayer());

    await waitFor(() => expect(addImageryProvider).toHaveBeenCalledTimes(1));

    setTerrainHillshadeEnabled(false);
    rerender();

    expect(removeImageryLayer).toHaveBeenCalledWith(
      addImageryProvider.mock.results[0].value,
      true,
    );
  });

  it("ignores airports without local terrain metadata", async () => {
    fetchJson.mockRejectedValueOnce(
      new JsonFetchError("missing metadata", {
        url: "/data/airports/KRDU/local-terrain/heightmap/metadata.json",
        status: 404,
      }),
    );

    renderHook(() => useTerrainHillshadeLayer());

    await waitFor(() => expect(fetchJson).toHaveBeenCalled());

    expect(addImageryProvider).not.toHaveBeenCalled();
  });
});
