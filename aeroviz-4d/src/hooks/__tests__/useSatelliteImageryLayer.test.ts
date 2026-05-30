import { describe, expect, it } from "vitest";
import type { AirportLocalTerrainState } from "../../context/AppContext";
import {
  NO_IMAGERY_TERRAIN_MATERIAL_SOURCE,
  noImageryTerrainHeightRange,
} from "../useSatelliteImageryLayer";

function localTerrainState(
  overrides: Partial<AirportLocalTerrainState> = {},
): AirportLocalTerrainState {
  return {
    status: "active",
    airportCode: "KSJC",
    sourceLabel: "Airport local heightmap terrain",
    minimumHeightM: 0.7,
    maximumHeightM: 50.7,
    loadedTiles: 1428,
    totalTiles: 1428,
    error: null,
    ...overrides,
  };
}

describe("useSatelliteImageryLayer", () => {
  it("keeps the no-imagery material compatible with local heightmap terrain", () => {
    expect(NO_IMAGERY_TERRAIN_MATERIAL_SOURCE).not.toMatch(/\bslope\b/);
    expect(NO_IMAGERY_TERRAIN_MATERIAL_SOURCE).not.toContain("normalEC");
    expect(NO_IMAGERY_TERRAIN_MATERIAL_SOURCE).not.toMatch(/contour/i);
  });

  it("uses active local terrain height stats for grayscale contrast", () => {
    expect(noImageryTerrainHeightRange(localTerrainState())).toEqual({
      minimumHeight: 0.7,
      maximumHeight: 50.7,
    });
  });

  it("pads tiny local height spans so flat airports still get visible contrast", () => {
    expect(
      noImageryTerrainHeightRange(
        localTerrainState({
          minimumHeightM: 10,
          maximumHeightM: 12,
        }),
      ),
    ).toEqual({
      minimumHeight: 6,
      maximumHeight: 16,
    });
  });
});
