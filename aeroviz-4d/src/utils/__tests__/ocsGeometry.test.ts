/**
 * ocsGeometry.test.ts
 * -------------------
 * Unit tests for the OCS geometry utility functions.
 *
 * These tests verify your math BEFORE you plug the output into CesiumJS.
 * Catching a bug here (with clear numeric expected values) is much easier
 * than debugging a wrong polygon in a 3D scene.
 *
 * How to run:
 *   npm test                         (watch mode)
 *   npm run test -- ocsGeometry      (run only this file)
 *   npm run test:coverage            (coverage report)
 *
 * Testing strategy:
 *   Each test uses a controlled scenario where the expected answer can be
 *   computed by hand.  Use `toBeCloseTo(expected, decimalPlaces)` for
 *   floating-point comparisons — never `toBe` for floats.
 */

import { describe, it, expect } from "vitest";
import {
  bearingRad,
  offsetPoint,
  buildFinalApproachOCS,
  type OCSParams,
} from "../ocsGeometry";

// ── Numeric tolerance (decimal places for toBeCloseTo) ────────────────────────
const PRECISION = 4; // 4 decimal places ≈ ±0.00005°, good enough for geodesy

// ─────────────────────────────────────────────────────────────────────────────
describe("bearingRad", () => {
  it("returns 0 (north) when B is directly north of A", () => {
    // Same longitude, latitude increases → bearing should be 0 (due north)
    const result = bearingRad(-119.38, 49.90, -119.38, 49.95);
    // TODO — replace the assertion below with a real check:
    //   expect(result).toBeCloseTo(0, PRECISION);
    expect(result).toBeDefined(); // ← placeholder, replace this line
  });

  it("returns π/2 (east) when B is directly east of A", () => {
    // Same latitude, longitude increases → bearing should be π/2 (due east)
    const result = bearingRad(-119.38, 49.95, -119.28, 49.95);
    // TODO — expect(result).toBeCloseTo(Math.PI / 2, PRECISION);
    expect(result).toBeDefined();
  });

  it("returns -π/2 (west) when B is directly west of A", () => {
    // TODO — Write the full assertion.
    //   lon decreases → bearing is -π/2
    const result = bearingRad(-119.28, 49.95, -119.38, 49.95);
    expect(result).toBeDefined();
  });

  it("returns approximately -π (south) when B is directly south of A", () => {
    // Same longitude, latitude decreases → bearing ≈ -π or π (wraps at π)
    // atan2 returns values in (-π, π], so due south = -π
    const result = bearingRad(-119.38, 49.95, -119.38, 49.90);
    // Hint: Math.PI ≈ 3.14159; expect the absolute value to be close to Math.PI
    // TODO — Write the assertion.
    expect(result).toBeDefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("offsetPoint", () => {
  it("moving due east increases longitude and keeps latitude constant", () => {
    // 1000 m due east from (-119.38, 49.95, 3000)
    // Expected Δlon ≈ 1000 / metresPerDegLon(49.95)
    //              ≈ 1000 / (111320 × cos(49.95° × π/180))
    //              ≈ 1000 / 71680 ≈ 0.01395°
    const result = offsetPoint(-119.38, 49.95, 3000, Math.PI / 2, 1000);

    // TODO ① — Assert result.lon is approximately -119.38 + 0.01395
    expect(result.lon).toBeDefined();

    // TODO ② — Assert result.lat is approximately 49.95 (unchanged)
    expect(result.lat).toBeDefined();

    // TODO ③ — Assert result.altM === 3000 (altitude unchanged)
    expect(result.altM).toBeDefined();
  });

  it("moving due north increases latitude and keeps longitude constant", () => {
    // 1000 m due north from (-119.38, 49.95, 3000)
    // Expected Δlat ≈ 1000 / 111320 ≈ 0.008983°
    const result = offsetPoint(-119.38, 49.95, 3000, 0, 1000);

    // TODO — Assert result.lat ≈ 49.95 + 0.008983, result.lon ≈ -119.38
    expect(result).toBeDefined();
  });

  it("distance of zero returns the original point", () => {
    const result = offsetPoint(-119.38, 49.95, 4500, Math.PI / 4, 0);
    // TODO — Assert all three coordinates equal the input
    expect(result.lon).toBeCloseTo(-119.38, PRECISION);
    // Fill in lat and altM assertions:
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildFinalApproachOCS", () => {
  // Test parameters: a simple north-to-south approach
  // FAF is north (higher), threshold is south (lower)
  const params: OCSParams = {
    faf:       { lon: -119.38, lat: 49.95, altM: 4500 },
    threshold: { lon: -119.38, lat: 49.90, altM: 430  },
    primaryHalfWidthM: 75,
    secondaryWidthM:   75,
  };

  it("primary polygon has exactly 4 corners", () => {
    const { primaryPolygon } = buildFinalApproachOCS(params);
    // TODO — expect(primaryPolygon).toHaveLength(4);
    expect(primaryPolygon).toBeDefined();
  });

  it("primary polygon corners straddle the centreline symmetrically", () => {
    // The two FAF corners should be equidistant (75 m) left and right
    // of the FAF lon, and the two threshold corners equidistant of the threshold lon.
    const { primaryPolygon } = buildFinalApproachOCS(params);

    // TODO ① — primaryPolygon[0] (fafLeft) and primaryPolygon[1] (fafRight)
    //   should have the same lat as faf.lat and lons that are roughly ±0.00105°
    //   from -119.38 (75 m ÷ metresPerDegLon(49.95) ≈ 0.00105°).
    //   Write two toBeCloseTo assertions.
    expect(primaryPolygon[0]).toBeDefined();
    expect(primaryPolygon[1]).toBeDefined();

    // TODO ② — primaryPolygon[0].altM should equal faf.altM (4500)
    //   primaryPolygon[2].altM should equal threshold.altM (430)
  });

  it("secondary polygons are outboard of the primary edge", () => {
    const { secondaryLeft, secondaryRight } = buildFinalApproachOCS(params);
    // The outer edge of the secondary area should be 150 m from the centreline
    // (primaryHalfWidthM + secondaryWidthM = 75 + 75).
    // That's approximately 0.00209° at lat 49.95.
    //
    // TODO — Assert that secondaryLeft has outer corners further from centre
    //   than primaryPolygon's left corners.
    expect(secondaryLeft).toBeDefined();
    expect(secondaryRight).toBeDefined();
  });

  it("secondary polygon outer altitude at FAF is reduced by 1/7 of secondaryWidthM", () => {
    // PANS-OPS 7:1 slope: outer_alt = faf.altM - secondaryWidthM / 7
    //                               = 4500 - 75/7 ≈ 4489.3 m
    const { secondaryLeft } = buildFinalApproachOCS(params);
    // TODO — Find the corner with the outer-FAF altitude and assert:
    //   expect(outerFafCorner.altM).toBeCloseTo(4500 - 75/7, 1);
    expect(secondaryLeft).toBeDefined();
  });
});
