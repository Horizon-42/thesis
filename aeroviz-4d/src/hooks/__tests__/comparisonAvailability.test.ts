import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import { availabilityByEntityId } from "../useComparisonTrajectoryLayer";

const EPOCH = "2026-04-01T08:00:00Z";

/** Seconds from EPOCH to a JulianDate. */
function offset(jd: Cesium.JulianDate): number {
  return Cesium.JulianDate.secondsDifference(jd, Cesium.JulianDate.fromIso8601(EPOCH));
}

describe("availabilityByEntityId", () => {
  it("derives each entity's [firstSample, lastSample] interval from position epoch + offsets", () => {
    // cartographicDegrees = [t, lon, lat, alt, ...]; first t = 0, last t = 576.
    const czml = [
      { id: "document", clock: {} },
      {
        id: "sim-UPS1276_05L",
        position: { epoch: EPOCH, cartographicDegrees: [0, -78.86, 36.14, 2667, 300, -78.85, 36.02, 900, 576, -78.85, 36.02, 114] },
      },
    ];
    const map = availabilityByEntityId(czml);
    expect([...map.keys()]).toEqual(["sim-UPS1276_05L"]);
    const interval = map.get("sim-UPS1276_05L")!.get(0)!;
    expect(offset(interval.start)).toBe(0);
    expect(offset(interval.stop)).toBe(576); // the LAST sample time, not an earlier one
  });

  it("skips the document packet and packets without a cartographicDegrees position", () => {
    const czml = [
      { id: "document" },
      { id: "no-position", name: "x" },
      { id: "no-epoch", position: { cartographicDegrees: [0, 0, 0, 0, 10, 0, 0, 0] } },
      { id: "ok", position: { epoch: EPOCH, cartographicDegrees: [0, 0, 0, 0, 10, 0, 0, 0] } },
    ];
    expect([...availabilityByEntityId(czml).keys()]).toEqual(["ok"]);
  });

  it("returns an empty map for non-array input", () => {
    expect(availabilityByEntityId(null).size).toBe(0);
    expect(availabilityByEntityId({}).size).toBe(0);
  });
});
