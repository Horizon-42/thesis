/**
 * ocsGeometry.test.ts
 * -------------------
 * Unit tests for the OCS geometry utility functions.
 *
 * These tests verify the math BEFORE the output is plugged into CesiumJS.
 * Catching a bug here (with clear numeric expected values) is much easier
 * than debugging a wrong polygon in a 3D scene.
 */

import { describe, it, expect } from "vitest";
import {
  bearingRad,
  offsetPoint,
  buildFinalApproachOCS,
  type OCSParams,
} from "../ocsGeometry";

const PRECISION = 4; // 4 decimal places ≈ ±0.00005°, good enough for geodesy

// Metres per degree constants mirrored from the implementation so we can
// derive expected values analytically without importing private helpers.
const METRES_PER_DEG_LAT = 111_320;
const metresPerDegLon = (latDeg: number) =>
  METRES_PER_DEG_LAT * Math.cos((latDeg * Math.PI) / 180);

// ─────────────────────────────────────────────────────────────────────────────
describe("bearingRad", () => {
  it("returns 0 (north) when B is directly north of A", () => {
    const result = bearingRad(-119.38, 49.90, -119.38, 49.95);
    expect(result).toBeCloseTo(0, PRECISION);
  });

  it("returns π/2 (east) when B is directly east of A", () => {
    const result = bearingRad(-119.38, 49.95, -119.28, 49.95);
    expect(result).toBeCloseTo(Math.PI / 2, PRECISION);
  });

  it("returns -π/2 (west) when B is directly west of A", () => {
    const result = bearingRad(-119.28, 49.95, -119.38, 49.95);
    expect(result).toBeCloseTo(-Math.PI / 2, PRECISION);
  });

  it("returns approximately ±π (south) when B is directly south of A", () => {
    // atan2 returns (-π, π]; due-south maps to -π (or +π).
    const result = bearingRad(-119.38, 49.95, -119.38, 49.90);
    expect(Math.abs(result)).toBeCloseTo(Math.PI, PRECISION);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("offsetPoint", () => {
  it("moving due east increases longitude and keeps latitude constant", () => {
    const result = offsetPoint(-119.38, 49.95, 3000, Math.PI / 2, 1000);
    const expectedDeltaLon = 1000 / metresPerDegLon(49.95);

    expect(result.lon).toBeCloseTo(-119.38 + expectedDeltaLon, PRECISION);
    expect(result.lat).toBeCloseTo(49.95, PRECISION);
    expect(result.altM).toBe(3000);
  });

  it("moving due north increases latitude and keeps longitude constant", () => {
    const result = offsetPoint(-119.38, 49.95, 3000, 0, 1000);
    const expectedDeltaLat = 1000 / METRES_PER_DEG_LAT;

    expect(result.lon).toBeCloseTo(-119.38, PRECISION);
    expect(result.lat).toBeCloseTo(49.95 + expectedDeltaLat, PRECISION);
    expect(result.altM).toBe(3000);
  });

  it("distance of zero returns the original point", () => {
    const result = offsetPoint(-119.38, 49.95, 4500, Math.PI / 4, 0);
    expect(result.lon).toBeCloseTo(-119.38, PRECISION);
    expect(result.lat).toBeCloseTo(49.95, PRECISION);
    expect(result.altM).toBe(4500);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildFinalApproachOCS", () => {
  // A simple north-to-south approach. FAF is north (higher),
  // threshold is south (lower). Centreline bearing ≈ -π (due south).
  const params: OCSParams = {
    faf:       { lon: -119.38, lat: 49.95, altM: 4500 },
    threshold: { lon: -119.38, lat: 49.90, altM: 430  },
    primaryHalfWidthM: 75,
    secondaryWidthM:   75,
  };

  it("primary polygon has exactly 4 corners", () => {
    const { primaryPolygon } = buildFinalApproachOCS(params);
    expect(primaryPolygon).toHaveLength(4);
  });

  it("secondary polygons each have exactly 4 corners", () => {
    const { secondaryLeft, secondaryRight } = buildFinalApproachOCS(params);
    expect(secondaryLeft).toHaveLength(4);
    expect(secondaryRight).toHaveLength(4);
  });

  it("primary polygon corners straddle the centreline symmetrically", () => {
    // For a due-south approach, centreline is constant longitude.
    // Each FAF corner should be ±(75 m / metresPerDegLon(faf.lat)) from faf.lon.
    const { primaryPolygon } = buildFinalApproachOCS(params);
    const dLonAtFaf = 75 / metresPerDegLon(params.faf.lat);
    const dLonAtThr = 75 / metresPerDegLon(params.threshold.lat);

    // Polygon order: [fafLeft, fafRight, thrRight, thrLeft]
    const [fafLeft, fafRight, thrRight, thrLeft] = primaryPolygon;

    // Latitudes unchanged from FAF / threshold respectively.
    expect(fafLeft.lat).toBeCloseTo(params.faf.lat, PRECISION);
    expect(fafRight.lat).toBeCloseTo(params.faf.lat, PRECISION);
    expect(thrLeft.lat).toBeCloseTo(params.threshold.lat, PRECISION);
    expect(thrRight.lat).toBeCloseTo(params.threshold.lat, PRECISION);

    // Bearing is ≈ -π (south). perpLeft = -π - π/2 = -3π/2 → east (+lon).
    // perpRight = -π + π/2 = -π/2 → west (-lon).
    expect(fafLeft.lon).toBeCloseTo(params.faf.lon + dLonAtFaf, PRECISION);
    expect(fafRight.lon).toBeCloseTo(params.faf.lon - dLonAtFaf, PRECISION);
    expect(thrLeft.lon).toBeCloseTo(params.threshold.lon + dLonAtThr, PRECISION);
    expect(thrRight.lon).toBeCloseTo(params.threshold.lon - dLonAtThr, PRECISION);

    // Altitudes: FAF corners at FAF altitude, threshold corners at threshold.
    expect(fafLeft.altM).toBe(params.faf.altM);
    expect(fafRight.altM).toBe(params.faf.altM);
    expect(thrLeft.altM).toBe(params.threshold.altM);
    expect(thrRight.altM).toBe(params.threshold.altM);
  });

  it("secondary polygons are outboard of the primary edge", () => {
    const { primaryPolygon, secondaryLeft, secondaryRight } =
      buildFinalApproachOCS(params);

    // Primary left corners are at ±75 m; secondary left outer corners at ±150 m.
    // For this south-bound approach: "left" corner has larger lon (east),
    // "right" corner has smaller lon (west).
    const primFafLeftLon = primaryPolygon[0].lon;
    const primFafRightLon = primaryPolygon[1].lon;

    // secondaryLeft = [fafLeft, secFafLeft, secThrLeft, thrLeft]
    const secFafLeftLon = secondaryLeft[1].lon;
    const secFafRightLon = secondaryRight[1].lon;

    // Outer secondary corner is further east than the primary left edge.
    expect(secFafLeftLon).toBeGreaterThan(primFafLeftLon);
    // Outer secondary corner is further west than the primary right edge.
    expect(secFafRightLon).toBeLessThan(primFafRightLon);
  });

  it("secondary outer altitude at FAF is reduced by 1/7 of secondaryWidthM", () => {
    // PANS-OPS 7:1 slope: outer_alt = faf.altM - secondaryWidthM / 7
    const expectedOuterAlt = params.faf.altM - params.secondaryWidthM / 7;
    const { secondaryLeft, secondaryRight } = buildFinalApproachOCS(params);

    // secondaryLeft corners: [fafLeft (inner), secFafLeft (outer),
    //                         secThrLeft (outer), thrLeft (inner)]
    expect(secondaryLeft[1].altM).toBeCloseTo(expectedOuterAlt, 1);
    expect(secondaryRight[1].altM).toBeCloseTo(expectedOuterAlt, 1);

    // At the threshold end, the outer altitude equals the threshold elevation.
    expect(secondaryLeft[2].altM).toBe(params.threshold.altM);
    expect(secondaryRight[2].altM).toBe(params.threshold.altM);
  });

  it("inner edges of the secondary polygons match the primary edges", () => {
    const { primaryPolygon, secondaryLeft, secondaryRight } =
      buildFinalApproachOCS(params);

    // primaryPolygon = [fafLeft, fafRight, thrRight, thrLeft]
    // secondaryLeft  = [fafLeft, secFafLeft, secThrLeft, thrLeft]
    expect(secondaryLeft[0]).toEqual(primaryPolygon[0]);
    expect(secondaryLeft[3]).toEqual(primaryPolygon[3]);

    // secondaryRight = [fafRight, secFafRight, secThrRight, thrRight]
    expect(secondaryRight[0]).toEqual(primaryPolygon[1]);
    expect(secondaryRight[3]).toEqual(primaryPolygon[2]);
  });
});
