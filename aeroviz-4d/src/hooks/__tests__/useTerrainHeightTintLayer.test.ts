import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JsonFetchError } from "../../utils/fetchJson";

const {
  fetchJson,
  mockViewer,
  getActiveAirportCode,
  setActiveAirportCode,
  getTerrainHeightTintEnabled,
  setTerrainHeightTintEnabled,
  addImageryProvider,
  removeImageryLayer,
  requestRender,
} = vi.hoisted(() => {
  const addImageryProvider = vi.fn((provider: any) => ({
    _tag: "height-tint-layer",
    alpha: 1,
    brightness: 1,
    contrast: 1,
    saturation: 1,
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
  let terrainHeightTintEnabled = true;

  return {
    fetchJson,
    mockViewer,
    getActiveAirportCode: () => activeAirportCode,
    setActiveAirportCode: (airportCode: string) => {
      activeAirportCode = airportCode;
    },
    getTerrainHeightTintEnabled: () => terrainHeightTintEnabled,
    setTerrainHeightTintEnabled: (enabled: boolean) => {
      terrainHeightTintEnabled = enabled;
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
      terrainHeightTint: getTerrainHeightTintEnabled(),
    },
  }),
}));

import { useTerrainHeightTintLayer } from "../useTerrainHeightTintLayer";

function metadata(alpha = 0.31) {
  return {
    bounds: {
      west: -79.02,
      south: 35.69,
      east: -78.55,
      north: 36.07,
    },
    overlay: {
      url: "/data/airports/KRDU/local-terrain/heightmap/local_terrain_height_overlay.png",
      width: 1024,
      height: 830,
      alpha,
    },
  };
}

describe("useTerrainHeightTintLayer", () => {
  beforeEach(() => {
    setActiveAirportCode("KRDU");
    setTerrainHeightTintEnabled(true);
    fetchJson.mockReset();
    fetchJson.mockResolvedValue(metadata());
    addImageryProvider.mockClear();
    removeImageryLayer.mockClear();
    requestRender.mockClear();
  });

  it("loads the active airport height tint as a bounded single-tile imagery layer", async () => {
    renderHook(() => useTerrainHeightTintLayer());

    await waitFor(() => expect(addImageryProvider).toHaveBeenCalledTimes(1));

    expect(fetchJson).toHaveBeenCalledWith(
      "/data/airports/KRDU/local-terrain/heightmap/metadata.json",
    );
    const provider = addImageryProvider.mock.calls[0][0];
    expect(provider.options).toEqual({
      url: "/data/airports/KRDU/local-terrain/heightmap/local_terrain_height_overlay.png",
      tileWidth: 1024,
      tileHeight: 830,
      rectangle: {
        west: -79.02,
        south: 35.69,
        east: -78.55,
        north: 36.07,
      },
      credit: "Airport terrain height tint",
    });
    expect(addImageryProvider.mock.results[0].value.alpha).toBe(0.31);
    expect(addImageryProvider.mock.results[0].value.brightness).toBe(1.05);
    expect(addImageryProvider.mock.results[0].value.contrast).toBe(1.12);
    expect(addImageryProvider.mock.results[0].value.saturation).toBe(0.9);
    expect(requestRender).toHaveBeenCalled();
  });

  it("removes the height tint layer when the toggle is disabled", async () => {
    const { rerender } = renderHook(() => useTerrainHeightTintLayer());

    await waitFor(() => expect(addImageryProvider).toHaveBeenCalledTimes(1));

    setTerrainHeightTintEnabled(false);
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

    renderHook(() => useTerrainHeightTintLayer());

    await waitFor(() => expect(fetchJson).toHaveBeenCalled());

    expect(addImageryProvider).not.toHaveBeenCalled();
  });
});
