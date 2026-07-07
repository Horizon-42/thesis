import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import { sampleEntityTrack } from "../useApproachView";
import type { RunwayFrame } from "../../utils/runwayProfileGeometry";

// A runway frame at KRDU-ish coordinates; the projection only needs a valid frame — the test
// asserts on sample COUNTS and liveness, not exact runway-frame values.
const FRAME: RunwayFrame = {
  runwayIdent: "RW05L",
  thresholdLon: -78.8,
  thresholdLat: 35.9,
  thresholdAltM: 100,
  approachUnitEast: 1,
  approachUnitNorth: 0,
  leftUnitEast: 0,
  leftUnitNorth: 1,
};

const EPOCH = Cesium.JulianDate.fromIso8601("2026-05-01T00:00:00Z");
const at = (seconds: number) => Cesium.JulianDate.addSeconds(EPOCH, seconds, new Cesium.JulianDate());

/**
 * A fake entity whose position moves in longitude for t in [0, lastT], then HOLDs the final
 * position forever after (forwardExtrapolationType "HOLD"), and is undefined before t=0
 * (backwardExtrapolationType NONE) — exactly the CZML contract the real tracks use.
 */
function heldTrackEntity(lastT: number): Cesium.Entity {
  const position = {
    getValue(time: Cesium.JulianDate, result?: Cesium.Cartesian3): Cesium.Cartesian3 | undefined {
      const t = Cesium.JulianDate.secondsDifference(time, EPOCH);
      if (t < 0) return undefined; // before the first sample (NONE extrapolation)
      const held = Math.min(t, lastT); // HOLD the final position past lastT
      // Move ~1e-3 deg/s in longitude so 5s steps are clearly distinct positions.
      const lon = -78.85 + held * 1e-3;
      return Cesium.Cartesian3.fromDegrees(lon, 35.88, 1500 - held * 2, Cesium.Ellipsoid.WGS84, result);
    },
  };
  return { id: "AAL1", position } as unknown as Cesium.Entity;
}

describe("sampleEntityTrack", () => {
  it("spans the whole real track and STOPS at the HOLD tail (no 600-sample overrun)", () => {
    // Real samples span t=0..200 (step 5s -> 41 points); getValue holds the final position
    // for every t>200. Anchored mid-track, the walk must cover the real track and stop at the
    // held tail, NOT run to MAX_TRACK_SAMPLES_PER_DIRECTION.
    const result = sampleEntityTrack(heldTrackEntity(200), at(100), FRAME);
    expect(result).not.toBeNull();
    // 0..200 at 5s = 41 distinct points; a naive walk-until-undefined would return 600+.
    expect(result!.trail.length).toBe(41);
    expect(result!.trail.length).toBeLessThan(100);
  });

  it("shows the whole track when paused at the START (all-forward)", () => {
    const result = sampleEntityTrack(heldTrackEntity(200), at(0), FRAME);
    expect(result).not.toBeNull();
    expect(result!.trail.length).toBe(41); // current (t=0) + forward to t=200
    expect(result!.trail[0].timeIso).toBe(result!.current.timeIso); // current is the earliest point
  });

  it("shows the whole track when parked at the END (all-backward, no held-tail padding)", () => {
    const result = sampleEntityTrack(heldTrackEntity(200), at(200), FRAME);
    expect(result).not.toBeNull();
    expect(result!.trail.length).toBe(41); // backward to t=0 + current (t=200)
    expect(result!.trail[result!.trail.length - 1].timeIso).toBe(result!.current.timeIso); // current is last
  });

  it("drops a LANDED aircraft parked on the held tail (no ghost frozen at the threshold)", () => {
    // currentTime is well past the last real sample, so the position is the frozen final one
    // and the step before it is identical — not live.
    expect(sampleEntityTrack(heldTrackEntity(200), at(500), FRAME)).toBeNull();
  });

  it("drops an aircraft not yet airborne (no position at currentTime)", () => {
    expect(sampleEntityTrack(heldTrackEntity(200), at(-50), FRAME)).toBeNull();
  });

  it("spans the same track regardless of playback position (so the hook can cache it once)", () => {
    // The whole-track geometry does not depend on where playback sits — the hook relies on
    // this to sample+classify each track ONCE and reuse it across ticks (the perf fix).
    const early = sampleEntityTrack(heldTrackEntity(200), at(40), FRAME)!;
    const late = sampleEntityTrack(heldTrackEntity(200), at(160), FRAME)!;
    expect(early.trail[0].timeIso).toBe(late.trail[0].timeIso); // same start
    expect(early.trail[early.trail.length - 1].timeIso).toBe(late.trail[late.trail.length - 1].timeIso);
    expect(early.trail.length).toBe(late.trail.length);
  });
});
