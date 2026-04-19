import { describe, expect, it } from "vitest";
import {
  buildTunnelSections,
  distanceMeters,
  type ProcedurePoint3D,
} from "../procedureGeometry";

describe("procedureGeometry", () => {
  const route: ProcedurePoint3D[] = [
    { lon: -78.9251472, lat: 35.7734139, altM: 914.4 },
    { lon: -78.8829556, lat: 35.8087667, altM: 670.56 },
    { lon: -78.8019631, lat: 35.87445, altM: 243.23 },
  ];

  it("computes non-zero route distances", () => {
    expect(distanceMeters(route[0], route[1])).toBeGreaterThan(5000);
  });

  it("generates connected tunnel cross-sections from route points", () => {
    const sections = buildTunnelSections(route, {
      halfWidthM: 556,
      halfHeightM: 91,
      sampleSpacingM: 250,
      nominalSpeedKt: 140,
    });

    expect(sections.length).toBeGreaterThan(route.length);
    expect(sections[0].distanceFromStartM).toBe(0);
    expect(sections[0].timeSeconds).toBe(0);
    expect(sections[0].leftTop.altM - sections[0].leftBottom.altM).toBeCloseTo(182);
    const finalSection = sections[sections.length - 1];
    expect(finalSection.distanceFromStartM).toBeGreaterThan(10000);
    expect(finalSection.timeSeconds).toBeGreaterThan(100);
  });

  it("returns no sections for malformed empty or single-point routes", () => {
    expect(buildTunnelSections([])).toEqual([]);
    expect(buildTunnelSections([route[0]])).toEqual([]);
  });
});
