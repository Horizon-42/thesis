import { describe, expect, it } from "vitest";
import {
  summarizeCartographicDegrees,
  summarizeObservedCzml,
} from "../observedFlightSummary";

describe("summarizeCartographicDegrees", () => {
  it("takes the duration from first→last sample time", () => {
    const samples = [0, -78.861065, 36.14, 2667, 1, -78.86, 36.14, 2650, 562, -78.80, 35.87, 114];
    expect(summarizeCartographicDegrees(samples)!.durationS).toBe(562);
  });

  it("returns null for empty samples and 0 duration for a single sample", () => {
    expect(summarizeCartographicDegrees([])).toBeNull();
    expect(summarizeCartographicDegrees([5, 0, 0, 0])!.durationS).toBe(0);
  });
});

describe("summarizeObservedCzml", () => {
  it("maps each non-document packet by id, skipping document + position-less packets", () => {
    const czml = [
      { id: "document", clock: {} },
      { id: "UPS1276", position: { cartographicDegrees: [0, 0, 0, 0, 10, 0, 0.001, 0] } },
      { id: "NOPOS", name: "no position" },
    ];
    const out = summarizeObservedCzml(czml);
    expect(Object.keys(out)).toEqual(["UPS1276"]);
    expect(out.UPS1276.durationS).toBe(10);
  });

  it("returns an empty map for non-array input", () => {
    expect(summarizeObservedCzml(null)).toEqual({});
    expect(summarizeObservedCzml({})).toEqual({});
  });
});
