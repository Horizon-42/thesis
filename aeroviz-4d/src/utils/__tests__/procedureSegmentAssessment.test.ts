import { describe, expect, it } from "vitest";
import type { HorizontalPlateRoute } from "../runwayProfileGeometry";
import {
  classifyGeoPointAgainstHorizontalPlateRoutes,
  classifyPointAgainstHorizontalPlateRoutes,
  projectPointToHorizontalPlateRoute,
} from "../procedureSegmentAssessment";
import { FEET_TO_METERS, METERS_PER_NM } from "../procedureGeoMath";

function latOffsetDeg(nm: number): number {
  return ((nm * METERS_PER_NM) / 6_378_137) * (180 / Math.PI);
}

const route: HorizontalPlateRoute = {
  routeId: "KRDU-R23RY-R",
  branchId: "branch:R",
  procedureName: "RNAV(GPS) Y RWY 23R",
  procedureFamily: "RNAV_GPS",
  procedureIdent: "R23RY",
  branchIdent: "R",
  transitionIdent: null,
  branchType: "final",
  defaultVisible: true,
  halfWidthM: 500,
  points: [
    {
      xM: 10_000,
      yM: 0,
      zM: 1_000,
      fixIdent: "IF",
      role: "IF",
      altitudeConstraint: null,
    },
    {
      xM: 5_000,
      yM: 0,
      zM: 500,
      fixIdent: "FAF",
      role: "FAF",
      altitudeConstraint: null,
    },
    {
      xM: 0,
      yM: 0,
      zM: 0,
      fixIdent: "RW23R",
      role: "MAPt",
      altitudeConstraint: null,
    },
  ],
};

describe("procedure segment assessment", () => {
  it("projects a profile point onto the nearest horizontal plate segment", () => {
    const assessment = projectPointToHorizontalPlateRoute({ xM: 4_000, yM: 120, zM: 460 }, route);

    expect(assessment).toMatchObject({
      routeId: "KRDU-R23RY-R",
      branchId: "branch:R",
      activeSegmentId: "branch:R:profile-segment:2",
      segmentIndex: 1,
      containment: "PRIMARY",
    });
    expect(assessment?.stationM).toBeCloseTo(6_000, 6);
    expect(Math.abs(assessment?.crossTrackErrorM ?? 0)).toBeCloseTo(120, 6);
    expect(assessment?.closestPoint).toMatchObject({
      xM: 4_000,
      yM: 0,
      zM: 400,
    });
    expect(assessment?.verticalErrorM).toBeCloseTo(60, 6);
    expect(assessment?.events).toContainEqual({
      kind: "VERTICAL_DEVIATION",
      label: "ABOVE_PROFILE",
      valueM: 60,
    });
  });

  it("classifies outside points while still returning nearest segment context", () => {
    const assessment = classifyPointAgainstHorizontalPlateRoutes(
      { xM: 4_000, yM: 700 },
      [route],
    );

    expect(assessment?.containment).toBe("OUTSIDE");
    expect(assessment?.activeSegmentId).toBe("branch:R:profile-segment:2");
    expect(Math.abs(assessment?.crossTrackErrorM ?? 0)).toBeCloseTo(700, 6);
    expect(assessment?.events).toContainEqual({
      kind: "LATERAL_CONTAINMENT",
      label: "OUTSIDE",
    });
  });

  it("uses render-bundle assessment segment widths when they are available", () => {
    const assessment = classifyPointAgainstHorizontalPlateRoutes(
      { xM: 7_500, yM: 700 },
      [
        {
          ...route,
          assessmentSegments: [
            {
              segmentId: "TEST-R23RY:branch:R:segment:final:1",
              primaryHalfWidthM: 500,
              secondaryHalfWidthM: 900,
              points: [
                { xM: 10_000, yM: 0, zM: 1_000 },
                { xM: 0, yM: 0, zM: 0 },
              ],
            },
          ],
        },
      ],
    );

    expect(assessment?.activeSegmentId).toBe("TEST-R23RY:branch:R:segment:final:1");
    expect(assessment?.containment).toBe("SECONDARY");
    expect(Math.abs(assessment?.crossTrackErrorM ?? 0)).toBeCloseTo(700, 6);
  });

  it("labels vertical deviations against an LNAV/VNAV OCS reference", () => {
    const assessment = classifyPointAgainstHorizontalPlateRoutes(
      { xM: 4_000, yM: 120, zM: 250 },
      [
        {
          ...route,
          assessmentSegments: [
            {
              segmentId: "TEST-R23RY:branch:R:segment:final:1",
              primaryHalfWidthM: 500,
              secondaryHalfWidthM: 900,
              points: [
                { xM: 5_000, yM: 0, zM: 500 },
                { xM: 0, yM: 0, zM: 0 },
              ],
              lnavVnavOcs: {
                kind: "LNAV_VNAV_OCS",
                label: "LNAV/VNAV OCS",
                gpaDeg: 3,
                tchFt: 50,
                primaryHalfWidthM: 500,
                secondaryHalfWidthM: 900,
                points: [
                  { xM: 5_000, yM: 0, zM: 500 },
                  { xM: 0, yM: 0, zM: 0 },
                ],
              },
            },
          ],
        },
      ],
    );

    expect(assessment?.activeSegmentId).toBe("TEST-R23RY:branch:R:segment:final:1");
    expect(assessment?.verticalErrorM).toBeCloseTo(-150, 6);
    expect(assessment?.events).toContainEqual({
      kind: "VERTICAL_OCS",
      label: "BELOW_OCS",
      valueM: -150,
    });
  });

  it("can classify aircraft points directly against route protection surfaces", () => {
    const assessment = classifyGeoPointAgainstHorizontalPlateRoutes(
      {
        lonDeg: 0.05,
        latDeg: latOffsetDeg(0.2),
        altM: 600 * FEET_TO_METERS,
      },
      [
        {
          ...route,
          protectionSurfaces: [
            {
              surfaceId: "surface:lnav-vnav-ocs",
              segmentId: "segment:final",
              sourceLegIds: ["leg:final"],
              kind: "FINAL_LNAV_VNAV_OCS",
              status: "TERPS_ESTIMATE",
              centerline: {
                geoPositions: [
                  { lonDeg: 0, latDeg: 0, altM: 1000 * FEET_TO_METERS },
                  { lonDeg: 0.1, latDeg: 0, altM: 500 * FEET_TO_METERS },
                ],
                worldPositions: [],
                geodesicLengthNm: 6,
                isArc: false,
              },
              lateral: {
                primary: {
                  geometryId: "surface:primary",
                  leftBoundary: [],
                  rightBoundary: [],
                  leftGeoBoundary: [],
                  rightGeoBoundary: [],
                  halfWidthNmSamples: [],
                },
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
            },
          ],
        },
      ],
      {
        runwayIdent: "RW",
        thresholdLon: 0,
        thresholdLat: 0,
        thresholdAltM: 0,
        approachUnitEast: 1,
        approachUnitNorth: 0,
        leftUnitEast: 0,
        leftUnitNorth: 1,
      },
    );

    expect(assessment).toMatchObject({
      activeSegmentId: "surface:lnav-vnav-ocs",
      containment: "PRIMARY",
      surfaceAssessment: expect.objectContaining({
        surfaceId: "surface:lnav-vnav-ocs",
        verticalKind: "OCS",
      }),
    });
    expect(assessment?.events).toContainEqual({
      kind: "VERTICAL_OCS",
      label: "BELOW_OCS",
      valueM: expect.any(Number),
    });
  });
});
