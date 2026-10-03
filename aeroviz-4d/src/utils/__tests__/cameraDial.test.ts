import { describe, expect, it } from "vitest";

import {
  ORBIT_PITCH_MAX_DEG, ORBIT_PITCH_MIN_DEG, ZOOM_MIN_RANGE_M,
  dialFlatten, dialPoint, orbitDrag, quantiseHeading, quantiseReadout, sameReadout, wrapDegrees360, zoomRange,
} from "../cameraDial";

describe("dialFlatten", () => {
  it("is a full disc from above and flattens towards the horizon, never past its floor", () => {
    expect(dialFlatten(-90)).toBeCloseTo(1, 12);
    expect(dialFlatten(-30)).toBeCloseTo(0.5, 12);
    expect(dialFlatten(-1)).toBe(0.28);
    expect(dialFlatten(0)).toBe(0.28);
  });
});

describe("dialPoint", () => {
  it("puts the bearing the camera faces at the far side, north to its right when facing west", () => {
    const ahead = dialPoint(90, 90, 30, 1);
    expect(ahead.x).toBeCloseTo(0, 12);
    expect(ahead.y).toBeCloseTo(-30, 12);
    expect(ahead.depth).toBeCloseTo(1, 12);
    const north = dialPoint(0, 270, 30, 1);
    expect(north.x).toBeCloseTo(30, 12);
    expect(north.y).toBeCloseTo(0, 12);
  });

  it("squashes only the vertical", () => {
    const behind = dialPoint(180, 0, 30, 0.5);
    expect(behind.x).toBeCloseTo(0, 12);
    expect(behind.y).toBeCloseTo(15, 12);
    expect(behind.depth).toBeCloseTo(-1, 12);
  });
});

describe("orbitDrag", () => {
  it("turns the disc with the finger: right lowers the heading, down tips towards a top view", () => {
    const moved = orbitDrag(100, -40, 50, 25);
    expect(moved.headingDeg).toBeCloseTo(80, 12);
    expect(moved.pitchDeg).toBeCloseTo(-50, 12);
  });

  it("wraps the heading and clamps the pitch", () => {
    expect(orbitDrag(5, -40, 50, 0).headingDeg).toBeCloseTo(345, 12);
    expect(orbitDrag(0, -40, 0, 1000).pitchDeg).toBe(ORBIT_PITCH_MIN_DEG);
    expect(orbitDrag(0, -40, 0, -1000).pitchDeg).toBe(ORBIT_PITCH_MAX_DEG);
  });
});

describe("zoomRange", () => {
  it("closes in on an upward drag, backs off on a downward one, proportionally", () => {
    expect(zoomRange(1000, -50)).toBeLessThan(1000);
    expect(zoomRange(1000, 50)).toBeGreaterThan(1000);
    expect(zoomRange(2000, -50) / 2000).toBeCloseTo(zoomRange(1000, -50) / 1000, 12);
  });

  it("stops at the minimum range", () => {
    expect(zoomRange(12, -1000)).toBe(ZOOM_MIN_RANGE_M);
  });
});

describe("wrapDegrees360", () => {
  it("maps onto [0, 360)", () => {
    expect(wrapDegrees360(-10)).toBe(350);
    expect(wrapDegrees360(370)).toBe(10);
    expect(wrapDegrees360(0)).toBe(0);
  });
});

describe("readout quantising", () => {
  const read = { heading: 359.94, pitch: -42.16, altitude: 5348.6, lat: 35.877641, lon: -78.787495 };

  it("keeps the heading on [0, 360)", () => {
    expect(quantiseHeading(359.94)).toBe(359.9);
    expect(quantiseHeading(359.95)).toBe(0);
    expect(quantiseHeading(0.04)).toBe(0);
  });

  it("rounds each field to what the panel prints", () => {
    expect(quantiseReadout(read)).toEqual({ heading: 359.9, pitch: -42.2, altitude: 5349, lat: 35.8776, lon: -78.7875 });
  });

  it("reads a pitch that rounds to −0 as 0, so equal states compare equal", () => {
    expect(Object.is(quantiseReadout({ ...read, pitch: -0.04 }).pitch, 0)).toBe(true);
  });

  it("treats two reads inside one printed step as the same state, and a visible change as different", () => {
    const a = quantiseReadout(read);
    expect(sameReadout(a, quantiseReadout({ ...read, heading: 359.9401, altitude: 5348.9 }))).toBe(true);
    expect(sameReadout(a, quantiseReadout({ ...read, altitude: 5350 }))).toBe(false);
  });
});
