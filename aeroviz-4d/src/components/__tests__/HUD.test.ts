/**
 * The HUD's "Local" line: the local terrain's phase from its state, the tile counts from their own context — while the
 * focused tiles warm ("Preload n/m"), and while the rest do after activation ("Active n/m").
 */
import { describe, expect, it } from "vitest";

import { localTerrainLabel } from "../HUD";

const tiles = (loadedTiles: number, totalTiles: number) => ({ loadedTiles, totalTiles });

describe("localTerrainLabel", () => {
  it("counts the tiles while they warm, and only then", () => {
    expect(localTerrainLabel("preloading", tiles(6, 41))).toBe("Preload 6/41");
    expect(localTerrainLabel("preloading", tiles(0, 0))).toBe("Preloading");
    expect(localTerrainLabel("active", tiles(12, 1428))).toBe("Active 12/1428");
    expect(localTerrainLabel("active", tiles(41, 41))).toBe("Active");
    expect(localTerrainLabel("active", tiles(0, 0))).toBe("Active");
  });

  it("names the other phases whatever the counts say", () => {
    expect(localTerrainLabel("loading", tiles(3, 9))).toBe("Loading");
    expect(localTerrainLabel("missing", tiles(0, 0))).toBe("Missing");
    expect(localTerrainLabel("error", tiles(3, 9))).toBe("Error");
    expect(localTerrainLabel("disabled", tiles(0, 0))).toBe("Off");
  });
});
