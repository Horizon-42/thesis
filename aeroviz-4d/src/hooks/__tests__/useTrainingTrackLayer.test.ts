/**
 * The 3D layer's one piece of arithmetic (T6): both tracks flattened into Cesium's
 * [lon, lat, height, …] with the altitude it actually wants.
 */
import { describe, expect, it } from "vitest";

import { trainingBandWall, trainingTrackPositions } from "../useTrainingTrackLayer";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";

function flight() {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value.flights[0];
}

describe("trainingTrackPositions", () => {
  it("flattens each track to lon, lat, height per row", () => {
    const item = flight();
    const { observed, flown } = trainingTrackPositions(item);
    expect(observed).toHaveLength(item.observed.tS.length * 3);
    expect(flown).toHaveLength(item.geometric.tS.length * 3);
    expect(observed.slice(0, 3)).toEqual([
      item.observed.lon[0],
      item.observed.lat[0],
      item.observed.altHaeM[0],
    ]);
  });

  // The altitude handed to Cesium is HAE, because that is what
  // `cartographicDegrees` means. Passing the MSL a record carries would float both
  // lines ~33.5 m above where they belong — together, so the mistake would be
  // invisible in the comparison and visible only against the terrain.
  it("takes the HAE column, not a height above the threshold", () => {
    const item = flight();
    const { observed } = trainingTrackPositions(item);
    expect(observed[2]).toBe(item.observed.altHaeM[0]);
    expect(observed[2]).not.toBe(item.observed.heightM[0]);
    // and the two tracks share a first point, as the words' rules require
    const { flown } = trainingTrackPositions(item);
    expect(flown[2]).toBeCloseTo(observed[2], 6);
  });

  // The wall is the vertical tolerance made visible in space. Its two heights
  // are HAE like everything else here, and `lo` — the SHALLOWER descent — is the
  // MAXIMUM, because losing less height means staying higher. Reading the names
  // as heights would turn the wall inside out and leave it exactly as thick.
  it("walls the corridor between the two heights the word allows", () => {
    const item = flight();
    const wall = trainingBandWall(item);
    expect(wall.positions).toHaveLength(item.geometric.tS.length * 2);
    expect(wall.maximumHeights).toEqual(item.geometric.verticalBand.altHaeLoM);
    expect(wall.minimumHeights).toEqual(item.geometric.verticalBand.altHaeHiM);
    wall.maximumHeights.forEach((top, row) => {
      expect(top).toBeGreaterThanOrEqual(wall.minimumHeights[row]);
    });
    // and it opens with distance: the far end is wider than the near one
    const last = wall.maximumHeights.length - 1;
    expect(wall.maximumHeights[last] - wall.minimumHeights[last]).toBeGreaterThan(
      wall.maximumHeights[1] - wall.minimumHeights[1],
    );
  });

  // The wall rides the FLOWN track's own ground positions — the commanded angle
  // never moves the horizontal step, which is what makes two height columns a
  // legal stand-in for two tracks.
  it("puts the wall on the flown track's own ground positions", () => {
    const item = flight();
    const wall = trainingBandWall(item);
    expect(wall.positions.slice(0, 2)).toEqual([item.geometric.lon[0], item.geometric.lat[0]]);
    expect(wall.positions).not.toContain(item.observed.lon[5]);
  });

  it("draws both tracks whole — the flown one is not cut to the observation", () => {
    const item = flight();
    const { observed, flown } = trainingTrackPositions(item);
    expect(flown.length).not.toBe(observed.length);
    expect(item.geometric.tS[item.geometric.tS.length - 1]).toBeGreaterThan(
      item.observed.tS[item.observed.tS.length - 1],
    );
  });
});
