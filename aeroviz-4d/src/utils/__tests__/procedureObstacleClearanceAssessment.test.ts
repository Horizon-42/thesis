import { describe, expect, it } from "vitest";
import type { ProcedureProtectionSurface } from "../../data/procedureProtectionSurfaces";
import { FEET_TO_METERS, METERS_PER_NM } from "../procedureGeoMath";
import {
  assessObstacleAgainstProtectionSurface,
  assessObstaclesAgainstProtectionSurfaces,
  obstaclePointFromFeature,
  type ObstaclePointFeature,
  type ProcedureObstaclePoint,
} from "../procedureObstacleClearanceAssessment";

function latOffsetDeg(nm: number): number {
  return ((nm * METERS_PER_NM) / 6_378_137) * (180 / Math.PI);
}

function surface(
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
    halfWidthNmSamples: [],
  };

  return {
    surfaceId: "surface:ocs",
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

const primaryObstacle: ProcedureObstaclePoint = {
  obstacleId: "obs-primary",
  obstacleType: "BLDG",
  lonDeg: 0.05,
  latDeg: latOffsetDeg(0.2),
  topAltitudeFtMsl: 700,
};

describe("procedure obstacle clearance assessment", () => {
  it("computes OCS clearance for obstacles inside the primary area", () => {
    const assessment = assessObstacleAgainstProtectionSurface(primaryObstacle, surface());

    expect(assessment).toMatchObject({
      obstacleId: "obs-primary",
      obstacleType: "BLDG",
      surfaceId: "surface:ocs",
      containment: "PRIMARY",
      status: "OCS_CLEAR",
      ruleStatus: "PRIMARY_OCS",
    });
    expect(assessment?.surfaceAltitudeFtMsl).toBeCloseTo(750, 0);
    expect(assessment?.clearanceFt).toBeCloseTo(50, 0);
  });

  it("flags OCS penetration when obstacle top is above the surface", () => {
    const assessment = assessObstacleAgainstProtectionSurface(
      { ...primaryObstacle, obstacleId: "obs-penetration", topAltitudeFtMsl: 820 },
      surface(),
    );

    expect(assessment).toMatchObject({
      obstacleId: "obs-penetration",
      status: "OCS_PENETRATION",
      ruleStatus: "PRIMARY_OCS",
    });
    expect(assessment?.clearanceFt).toBeLessThan(0);
  });

  it("keeps lateral-only OEA results separate from vertical clearance", () => {
    const assessment = assessObstacleAgainstProtectionSurface(
      primaryObstacle,
      surface({
        surfaceId: "surface:oea",
        kind: "FINAL_LNAV_OEA",
        status: "SOURCE_BACKED",
        vertical: {
          kind: "NONE",
          origin: "SOURCE",
          samples: [],
          notes: [],
        },
      }),
    );

    expect(assessment).toMatchObject({
      surfaceId: "surface:oea",
      containment: "PRIMARY",
      status: "LATERAL_ONLY",
      ruleStatus: "LATERAL_OEA_ONLY",
      clearanceFt: null,
      surfaceAltitudeFtMsl: null,
    });
  });

  it("reports secondary OCS comparisons as raw because reduced ROC is not modeled", () => {
    const assessment = assessObstacleAgainstProtectionSurface(
      {
        ...primaryObstacle,
        obstacleId: "obs-secondary",
        latDeg: latOffsetDeg(0.8),
      },
      surface(),
    );

    expect(assessment).toMatchObject({
      obstacleId: "obs-secondary",
      containment: "SECONDARY",
      ruleStatus: "SECONDARY_RAW_OCS_NO_REDUCED_ROC",
    });
    expect(assessment?.notes[0]).toContain("reduced secondary ROC");
  });

  it("excludes outside and debug surfaces by default", () => {
    const outside = assessObstacleAgainstProtectionSurface(
      {
        ...primaryObstacle,
        obstacleId: "obs-outside",
        latDeg: latOffsetDeg(1.4),
      },
      surface(),
    );
    const debug = assessObstacleAgainstProtectionSurface(
      primaryObstacle,
      surface({
        surfaceId: "surface:debug",
        kind: "TURNING_MISSED_DEBUG",
        status: "DEBUG_ESTIMATE",
      }),
    );

    expect(outside).toBeNull();
    expect(debug).toBeNull();
  });

  it("sorts penetrations before clearances and lateral-only inclusions", () => {
    const assessments = assessObstaclesAgainstProtectionSurfaces(
      [
        primaryObstacle,
        { ...primaryObstacle, obstacleId: "obs-penetration", topAltitudeFtMsl: 850 },
      ],
      [
        surface(),
        surface({
          surfaceId: "surface:oea",
          kind: "FINAL_LNAV_OEA",
          status: "SOURCE_BACKED",
          vertical: {
            kind: "NONE",
            origin: "SOURCE",
            samples: [],
            notes: [],
          },
        }),
      ],
    );

    expect(assessments[0]).toMatchObject({
      obstacleId: "obs-penetration",
      surfaceId: "surface:ocs",
      status: "OCS_PENETRATION",
    });
  });

  it("converts airport obstacle GeoJSON features into assessment points", () => {
    const feature: ObstaclePointFeature = {
      type: "Feature",
      geometry: { type: "Point", coordinates: [0.05, 0] },
      properties: {
        oas_number: "37-000001",
        verified: true,
        country: "US",
        state: "NC",
        city: "TEST",
        obstacle_type: "TOWER",
        quantity: 1,
        agl_ft: 100,
        amsl_ft: 700,
        agl_m: 30.48,
        amsl_m: 213.36,
        lighting: "N",
        horizontal_accuracy: "1",
        vertical_accuracy: "A",
        marking: "N",
        source: "DOF",
      },
    };

    expect(obstaclePointFromFeature(feature)).toMatchObject({
      obstacleId: "37-000001",
      obstacleType: "TOWER",
      lonDeg: 0.05,
      topAltitudeFtMsl: 700,
    });
  });
});
