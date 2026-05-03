import { describe, expect, it } from "vitest";
import type { ProcedureProtectionSurface } from "../../data/procedureProtectionSurfaces";
import { FEET_TO_METERS, METERS_PER_NM } from "../procedureGeoMath";
import {
  assessPointAgainstProtectionSurface,
  classifyPointAgainstProtectionSurfaces,
} from "../procedureProtectionVolumeAssessment";

function latOffsetDeg(nm: number): number {
  return ((nm * METERS_PER_NM) / 6_378_137) * (180 / Math.PI);
}

function testSurface(
  overrides: Partial<ProcedureProtectionSurface> = {},
): ProcedureProtectionSurface {
  const centerline = {
    geoPositions: [
      { lonDeg: 0, latDeg: 0, altM: 1000 * FEET_TO_METERS },
      { lonDeg: 0.1, latDeg: 0, altM: 500 * FEET_TO_METERS },
    ],
    worldPositions: [],
    geodesicLengthNm: 6,
    isArc: false,
  };
  const ribbon = {
    geometryId: "surface:primary",
    leftBoundary: [],
    rightBoundary: [],
    leftGeoBoundary: [],
    rightGeoBoundary: [],
    halfWidthNmSamples: [
      { stationNm: 0, halfWidthNm: 0.5 },
      { stationNm: 6, halfWidthNm: 0.5 },
    ],
  };

  return {
    surfaceId: "surface:lnav-vnav-ocs",
    segmentId: "segment:final",
    sourceLegIds: ["leg:final"],
    kind: "FINAL_LNAV_VNAV_OCS",
    status: "TERPS_ESTIMATE",
    centerline,
    lateral: {
      primary: ribbon,
      secondaryOuter: null,
      widthSamples: [
        { stationNm: 0, primaryHalfWidthNm: 0.5, secondaryOuterHalfWidthNm: 1 },
        { stationNm: 6, primaryHalfWidthNm: 0.5, secondaryOuterHalfWidthNm: 1 },
      ],
      rule: "test rule",
      notes: [],
    },
    vertical: {
      kind: "OCS",
      origin: "GPA_TCH",
      samples: [
        { stationNm: 0, altitudeFtMsl: 1000 },
        { stationNm: 6, altitudeFtMsl: 500 },
      ],
      notes: [],
    },
    diagnostics: [],
    ...overrides,
  };
}

describe("procedure protection volume assessment", () => {
  it("checks a 3D point against lateral width and vertical samples from one protection surface", () => {
    const assessment = assessPointAgainstProtectionSurface(
      {
        lonDeg: 0.05,
        latDeg: latOffsetDeg(0.4),
        altM: 800 * FEET_TO_METERS,
      },
      testSurface(),
    );

    expect(assessment).toMatchObject({
      surfaceId: "surface:lnav-vnav-ocs",
      segmentId: "segment:final",
      surfaceKind: "FINAL_LNAV_VNAV_OCS",
      containment: "PRIMARY",
      verticalRelation: "ABOVE_SURFACE",
    });
    expect(assessment?.stationNm).toBeCloseTo(3, 0);
    expect(assessment?.lateralDistanceNm).toBeCloseTo(0.4, 1);
    expect(assessment?.verticalDeltaFt).toBeCloseTo(50, 0);
  });

  it("prefers OCS surfaces over lateral-only OEA surfaces when containment is equivalent", () => {
    const oea = testSurface({
      surfaceId: "surface:oea",
      kind: "FINAL_LNAV_OEA",
      status: "SOURCE_BACKED",
      vertical: {
        kind: "NONE",
        origin: "SOURCE",
        samples: [],
        notes: [],
      },
    });
    const ocs = testSurface();

    const assessment = classifyPointAgainstProtectionSurfaces(
      {
        lonDeg: 0.05,
        latDeg: latOffsetDeg(0.2),
        altM: 700 * FEET_TO_METERS,
      },
      [oea, ocs],
    );

    expect(assessment?.surfaceId).toBe("surface:lnav-vnav-ocs");
    expect(assessment?.verticalKind).toBe("OCS");
  });

  it("does not use debug-estimate surfaces for operational assessment by default", () => {
    const debugSurface = testSurface({
      surfaceId: "surface:turn-debug",
      kind: "TURNING_MISSED_DEBUG",
      status: "DEBUG_ESTIMATE",
    });

    expect(
      classifyPointAgainstProtectionSurfaces(
        { lonDeg: 0.05, latDeg: 0, altM: 700 * FEET_TO_METERS },
        [debugSurface],
      ),
    ).toBeNull();
    expect(
      classifyPointAgainstProtectionSurfaces(
        { lonDeg: 0.05, latDeg: 0, altM: 700 * FEET_TO_METERS },
        [debugSurface],
        { includeDebug: true },
      )?.surfaceId,
    ).toBe("surface:turn-debug");
  });
});
