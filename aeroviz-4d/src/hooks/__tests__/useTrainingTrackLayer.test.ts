/**
 * The 3D layer's one piece of arithmetic (T6): both tracks flattened into Cesium's
 * [lon, lat, height, …] with the altitude it actually wants.
 */
import { describe, expect, it } from "vitest";

import { trainingTrackPositions } from "../useTrainingTrackLayer";
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

  it("draws both tracks whole — the flown one is not cut to the observation", () => {
    const item = flight();
    const { observed, flown } = trainingTrackPositions(item);
    expect(flown.length).not.toBe(observed.length);
    expect(item.geometric.tS[item.geometric.tS.length - 1]).toBeGreaterThan(
      item.observed.tS[item.observed.tS.length - 1],
    );
  });
});
