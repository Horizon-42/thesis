import { describe, expect, it } from "vitest";
import {
  summarizeCartographicDegrees,
  summarizeObservedCzml,
} from "../observedFlightSummary";

describe("summarizeCartographicDegrees", () => {
  it("takes the duration from first→last sample time", () => {
    const samples = [0, -78.861065, 36.14, 2667, 1, -78.86, 36.14, 2650, 562, -78.80, 35.87, 114];
    expect(summarizeCartographicDegrees(samples)).toBe(562);
  });

  it("returns null for empty samples and 0 duration for a single sample", () => {
    expect(summarizeCartographicDegrees([])).toBeNull();
    expect(summarizeCartographicDegrees([5, 0, 0, 0])).toBe(0);
  });
});

describe("summarizeObservedCzml", () => {
  it("maps each non-document packet by entity id, with the callsign from `name`", () => {
    // Entity ids are flight_keys; `name` carries the display callsign.
    const czml = [
      { id: "document", clock: {} },
      { id: "UPS1276_05L_a1b2c3_20260618T213736Z", name: "UPS1276",
        position: { cartographicDegrees: [0, 0, 0, 0, 10, 0, 0.001, 0] } },
      { id: "NOPOS", name: "no position" },
    ];
    const out = summarizeObservedCzml(czml);
    expect(Object.keys(out)).toEqual(["UPS1276_05L_a1b2c3_20260618T213736Z"]);
    expect(out["UPS1276_05L_a1b2c3_20260618T213736Z"].durationS).toBe(10);
    expect(out["UPS1276_05L_a1b2c3_20260618T213736Z"].callsign).toBe("UPS1276");
  });

  it("falls back to the entity id when a packet has no name", () => {
    const czml = [{ id: "NONAME", position: { cartographicDegrees: [0, 0, 0, 0] } }];
    expect(summarizeObservedCzml(czml).NONAME.callsign).toBe("NONAME");
  });

  it("returns an empty map for non-array input", () => {
    expect(summarizeObservedCzml(null)).toEqual({});
    expect(summarizeObservedCzml({})).toEqual({});
  });
});
