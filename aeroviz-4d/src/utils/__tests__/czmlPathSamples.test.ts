import { describe, expect, it } from "vitest";
import { czmlPathSamples } from "../czmlPathSamples";

describe("czmlPathSamples", () => {
  const packet = (id: string, samples: number[] | undefined) => ({
    id, ...(samples ? { position: { epoch: "2026-04-01T08:00:00Z", cartographicDegrees: samples } } : {}) });

  it("reads lon, lat and altitude of every sample of the packets whose id has a wanted prefix", () => {
    const czml = [
      { id: "document" },
      packet("sim-A", [0, -78, 35, 1000, 10, -77, 35.5, 900]),
      packet("ref-A", [0, -70, 30, 100]),
      packet("sim-B", [0, -76, 36, 800]),
    ];
    expect(czmlPathSamples(czml, ["sim-A", "sim-B"])).toEqual([
      { lon: -78, lat: 35, altM: 1000 }, { lon: -77, lat: 35.5, altM: 900 }, { lon: -76, lat: 36, altM: 800 }]);
    expect(czmlPathSamples(czml, ["ref-"])).toEqual([{ lon: -70, lat: 30, altM: 100 }]);
  });

  it("gives nothing for a packet without samples, a prefix nobody has, or something that is not a CZML array", () => {
    expect(czmlPathSamples([packet("sim-A", undefined)], ["sim-"])).toEqual([]);
    expect(czmlPathSamples([packet("sim-A", [0, 1, 2, 3])], ["pred-"])).toEqual([]);
    expect(czmlPathSamples({}, ["sim-"])).toEqual([]);
  });
});
