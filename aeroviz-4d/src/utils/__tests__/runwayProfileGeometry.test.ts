import { describe, expect, it } from "vitest";
import {
  attachRenderBundleAssessmentSegments,
  buildHorizontalPlateRoutes,
  buildRunwayReferenceMarks,
  buildRunwayFrame,
  pointIsInsideHorizontalPlate,
  projectPositionToRunwayFrame,
  type RunwayFeatureCollection,
} from "../runwayProfileGeometry";
import type { ProcedureRouteViewModel } from "../../data/procedureRoutes";

const runwayCollection: RunwayFeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [0.0, -0.00005],
          [0.0, 0.00005],
          [0.001, 0.00005],
          [0.001, -0.00005],
          [0.0, -0.00005],
        ]],
      },
      properties: {
        airport_ident: "TEST",
        runway_ident: "09/27",
        zone_type: "runway_surface",
        le_ident: "09",
        he_ident: "27",
        length_ft: 10000,
        width_ft: 150,
        surface: "ASP",
        lighted: 1,
        le_elevation_ft: 100,
        he_elevation_ft: 120,
      },
    },
  ],
};

const baseRoute = {
  airport: "TEST",
  procedureUid: "TEST-R09-RW09",
  procedureType: "R",
  procedureIdent: "R09",
  procedureName: "RNAV(GPS) RW09",
  procedureFamily: "RNAV_GPS",
  procedureVariant: null,
  runwayIdent: "RW09",
  branchProcedureType: "R",
  nominalSpeedKt: 140,
  tunnel: {
    lateralHalfWidthNm: 0.3,
    verticalHalfHeightFt: 300,
    sampleSpacingM: 250,
  },
};

const procedureRoutes: ProcedureRouteViewModel[] = [
  {
    ...baseRoute,
    routeId: "TEST-R09-R",
    branchId: "branch:R",
    branchKey: "R",
    branchIdent: "R",
    transitionIdent: null,
    branchType: "final",
    defaultVisible: true,
    warnings: [],
    points: [
      {
        fixId: "fix:SCHOO",
        fixIdent: "SCHOO",
        sequence: 10,
        legType: "IF",
        role: "IF",
        lon: -0.010,
        lat: 0.0,
        altitudeFt: 1600,
        altitudeConstraint: { kind: "AT" as const, minFtMsl: 1600, maxFtMsl: 1600, sourceText: "1600 ft" },
        geometryAltitudeFt: 1600,
        altM: 500,
        distanceFromStartM: 0,
        timeSeconds: 0,
        sourceLine: 1,
      },
      {
        fixId: "fix:WEPAS",
        fixIdent: "WEPAS",
        sequence: 20,
        legType: "TF",
        role: "FAF",
        lon: -0.005,
        lat: 0.0,
        altitudeFt: 900,
        altitudeConstraint: { kind: "AT" as const, minFtMsl: 900, maxFtMsl: 900, sourceText: "900 ft" },
        geometryAltitudeFt: 900,
        altM: 300,
        distanceFromStartM: 500,
        timeSeconds: 50,
        sourceLine: 2,
      },
      {
        fixId: "fix:RW09",
        fixIdent: "RW09",
        sequence: 30,
        legType: "TF",
        role: "MAPt",
        lon: 0.0,
        lat: 0.0,
        altitudeFt: 110,
        altitudeConstraint: { kind: "AT" as const, minFtMsl: 110, maxFtMsl: 110, sourceText: "110 ft" },
        geometryAltitudeFt: 110,
        altM: 35,
        distanceFromStartM: 1000,
        timeSeconds: 100,
        sourceLine: 3,
      },
    ],
  },
  {
    ...baseRoute,
    routeId: "TEST-R09-TRANS",
    branchId: "branch:TRANS",
    branchKey: "TRANS",
    branchIdent: "TRANS",
    transitionIdent: "TRANS",
    branchType: "transition",
    defaultVisible: false,
    warnings: [],
    points: [
      {
        fixId: "fix:CONCA",
        fixIdent: "CONCA",
        sequence: 5,
        legType: "IF",
        role: "IF",
        lon: -0.015,
        lat: 0.0002,
        altitudeFt: null,
        altitudeConstraint: null,
        geometryAltitudeFt: 1600,
        altM: 500,
        distanceFromStartM: 0,
        timeSeconds: 0,
        sourceLine: 4,
      },
      {
        fixId: "fix:SCHOO",
        fixIdent: "SCHOO",
        sequence: 10,
        legType: "TF",
        role: "IF",
        lon: -0.010,
        lat: 0.0,
        altitudeFt: 1600,
        altitudeConstraint: { kind: "AT_OR_ABOVE" as const, minFtMsl: 1600, sourceText: "1600 ft" },
        geometryAltitudeFt: 1600,
        altM: 500,
        distanceFromStartM: 500,
        timeSeconds: 40,
        sourceLine: 5,
      },
      {
        fixId: "fix:WEPAS",
        fixIdent: "WEPAS",
        sequence: 20,
        legType: "TF",
        role: "FAF",
        lon: -0.005,
        lat: 0.0,
        altitudeFt: 900,
        altitudeConstraint: { kind: "AT" as const, minFtMsl: 900, maxFtMsl: 900, sourceText: "900 ft" },
        geometryAltitudeFt: 900,
        altM: 300,
        distanceFromStartM: 1000,
        timeSeconds: 90,
        sourceLine: 2,
      },
      {
        fixId: "fix:RW09",
        fixIdent: "RW09",
        sequence: 30,
        legType: "TF",
        role: "MAPt",
        lon: 0.0,
        lat: 0.0,
        altitudeFt: 110,
        altitudeConstraint: { kind: "AT" as const, minFtMsl: 110, maxFtMsl: 110, sourceText: "110 ft" },
        geometryAltitudeFt: 110,
        altM: 35,
        distanceFromStartM: 1500,
        timeSeconds: 140,
        sourceLine: 3,
      },
    ],
  },
];

describe("runwayProfileGeometry", () => {
  it("builds a runway frame with threshold-centered runway coordinates", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");

    expect(frame.runwayIdent).toBe("RW09");
    expect(frame.thresholdLon).toBeCloseTo(0, 6);
    expect(frame.thresholdLat).toBeCloseTo(0, 6);
    expect(frame.approachUnitEast).toBeLessThan(-0.99);
    expect(Math.abs(frame.approachUnitNorth)).toBeLessThan(0.02);
  });

  it("filters points by the RNAV horizontal plate around the selected runway", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const routes = buildHorizontalPlateRoutes(procedureRoutes, frame, "RW09");

    const inside = projectPositionToRunwayFrame(frame, -0.0045, 0.00003, 220);
    const outside = projectPositionToRunwayFrame(frame, -0.0045, 0.008, 220);

    expect(routes).toHaveLength(2);
    expect(routes.map((route) => route.branchId)).toContain("TEST-R09-RW09:branch:R");
    expect(pointIsInsideHorizontalPlate(inside, routes)).toBe(true);
    expect(pointIsInsideHorizontalPlate(outside, routes)).toBe(false);
  });

  it("builds x-axis reference marks from important runway procedure fixes", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const marks = buildRunwayReferenceMarks(procedureRoutes, frame, "RW09");

    expect(marks.some((mark) => mark.label === "RW09" && mark.detail === "Threshold")).toBe(true);
    expect(marks.some((mark) => mark.label === "RW09" && mark.detail === "MAPt")).toBe(true);
    expect(marks.some((mark) => mark.label === "WEPAS" && mark.detail === "FAF")).toBe(true);
    expect(
      marks.some((mark) => mark.label === "CONCA" && mark.detail === "IF" && mark.zM > 400),
    ).toBe(true);
  });

  it("fills unknown zero-altitude transition endpoints from nearby valid route heights", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const routes = buildHorizontalPlateRoutes(procedureRoutes, frame, "RW09");
    const transitionRoute = routes.find((route) => route.routeId === "TEST-R09-TRANS");

    expect(transitionRoute).toBeTruthy();
    expect(transitionRoute?.points[0].zM).toBeCloseTo(transitionRoute?.points[1].zM ?? 0, 6);
    expect(transitionRoute?.points[0].zM ?? 0).toBeGreaterThan(400);
  });

  it("omits degenerate assessment segments that would render as standalone protection caps", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const routes = buildHorizontalPlateRoutes(procedureRoutes, frame, "RW09");
    const enriched = attachRenderBundleAssessmentSegments(
      routes,
      [
        {
          packageId: "TEST-R09-RW09",
          procedureId: "R09",
          procedureName: "RNAV(GPS) RW09",
          airportId: "TEST",
          diagnostics: [],
          branchBundles: [
            {
              branchId: "TEST-R09-RW09:branch:R",
              branchName: "R",
              branchRole: "STRAIGHT_IN",
              runwayId: "RW09",
              turnJunctions: [],
              missedCaMahfConnectors: [],
              segmentBundles: [
                {
                  segment: {
                    segmentId: "TEST-R09-RW09:branch:R:segment:if:1",
                    segmentType: "INITIAL",
                    xttNm: 0.3,
                    secondaryEnabled: true,
                  },
                  legs: [],
                  diagnostics: [],
                  finalOea: null,
                  lnavVnavOcs: null,
                  precisionFinalSurfaces: [],
                  alignedConnector: null,
                  segmentGeometry: {
                    segmentId: "TEST-R09-RW09:branch:R:segment:if:1",
                    centerline: {
                      geoPositions: [
                        { lonDeg: -0.01, latDeg: 0, altM: 500 },
                        { lonDeg: -0.01, latDeg: 0, altM: 500 },
                      ],
                      worldPositions: [],
                      geodesicLengthNm: 0,
                      isArc: false,
                    },
                    stationAxis: { samples: [], totalLengthNm: 0 },
                    primaryEnvelope: {
                      geometryId: "primary",
                      envelopeType: "PRIMARY",
                      leftBoundary: [],
                      rightBoundary: [],
                      leftGeoBoundary: [],
                      rightGeoBoundary: [],
                      halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 0.6 }],
                    },
                    secondaryEnvelope: null,
                    turnJunctions: [],
                    diagnostics: [],
                  },
                },
              ],
            },
          ],
        },
      ] as any,
      frame,
      "RW09",
    );
    const finalRoute = enriched.find((route) => route.branchId === "TEST-R09-RW09:branch:R");

    expect(finalRoute?.assessmentSegments).toBeUndefined();
  });

  it("attaches render-bundle segment geometry for profile assessment without replacing fix display points", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const routes = buildHorizontalPlateRoutes(procedureRoutes, frame, "RW09");
    const enriched = attachRenderBundleAssessmentSegments(
      routes,
      [
        {
          packageId: "TEST-R09-RW09",
          procedureId: "R09",
          procedureName: "RNAV(GPS) RW09",
          airportId: "TEST",
          diagnostics: [],
          branchBundles: [
            {
              branchId: "TEST-R09-RW09:branch:R",
              branchName: "R",
              branchRole: "STRAIGHT_IN",
              runwayId: "RW09",
              turnJunctions: [],
              missedCaMahfConnectors: [],
              missedConnectorSurfaces: [],
              protectionSurfaces: [
                {
                  surfaceId: "TEST-R09-RW09:branch:R:segment:final:1:lnav-oea",
                  segmentId: "TEST-R09-RW09:branch:R:segment:final:1",
                  kind: "FINAL_LNAV_OEA",
                  status: "SOURCE_BACKED",
                  centerline: {
                    geoPositions: [
                      { lonDeg: -0.005, latDeg: 0, altM: 300 },
                      { lonDeg: 0, latDeg: 0, altM: 35 },
                    ],
                  },
                  lateral: {
                    widthSamples: [
                      { stationNm: 0, primaryHalfWidthNm: 0.65, secondaryOuterHalfWidthNm: 0.95 },
                    ],
                  },
                  vertical: { kind: "NONE", origin: "SOURCE", samples: [] },
                  diagnostics: [],
                },
                {
                  surfaceId: "TEST-R09-RW09:branch:R:segment:final:1:lnav-vnav-ocs",
                  segmentId: "TEST-R09-RW09:branch:R:segment:final:1",
                  kind: "FINAL_LNAV_VNAV_OCS",
                  status: "TERPS_ESTIMATE",
                  centerline: {
                    geoPositions: [
                      { lonDeg: -0.005, latDeg: 0, altM: 250 },
                      { lonDeg: 0, latDeg: 0, altM: 60 },
                    ],
                  },
                  lateral: {
                    widthSamples: [
                      { stationNm: 0, primaryHalfWidthNm: 0.5, secondaryOuterHalfWidthNm: 0.75 },
                    ],
                  },
                  vertical: { kind: "OCS", origin: "GPA_TCH", samples: [] },
                  diagnostics: [],
                },
              ],
              segmentBundles: [
                {
                  segment: {
                    segmentId: "TEST-R09-RW09:branch:R:segment:final:1",
                    segmentType: "FINAL_LNAV_VNAV",
                    xttNm: 0.3,
                    secondaryEnabled: true,
                    verticalRule: { kind: "BARO_GLIDEPATH", gpaDeg: 3, tchFt: 50 },
                  },
                  legs: [],
                  diagnostics: [],
                  finalOea: null,
                  lnavVnavOcs: {
                    verticalProfile: { gpaDeg: 3, tchFt: 50 },
                    centerline: {
                      geoPositions: [
                        { lonDeg: -0.005, latDeg: 0, altM: 250 },
                        { lonDeg: 0, latDeg: 0, altM: 60 },
                      ],
                    },
                    primary: {
                      halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 0.4 }],
                    },
                    secondaryOuter: {
                      halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 0.7 }],
                    },
                  },
                  alignedConnector: null,
                  segmentGeometry: {
                    segmentId: "TEST-R09-RW09:branch:R:segment:final:1",
                    centerline: {
                      geoPositions: [
                        { lonDeg: -0.005, latDeg: 0, altM: 300 },
                        { lonDeg: 0, latDeg: 0, altM: 35 },
                      ],
                      worldPositions: [],
                      geodesicLengthNm: 0.3,
                      isArc: false,
                    },
                    stationAxis: { samples: [], totalLengthNm: 0.3 },
                    primaryEnvelope: {
                      geometryId: "primary",
                      envelopeType: "PRIMARY",
                      leftBoundary: [],
                      rightBoundary: [],
                      leftGeoBoundary: [],
                      rightGeoBoundary: [],
                      halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 0.6 }],
                    },
                    secondaryEnvelope: {
                      geometryId: "secondary",
                      envelopeType: "SECONDARY",
                      leftBoundary: [],
                      rightBoundary: [],
                      leftGeoBoundary: [],
                      rightGeoBoundary: [],
                      halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 0.9 }],
                    },
                    turnJunctions: [],
                    diagnostics: [],
                  },
                },
              ],
            },
          ],
        },
      ] as any,
      frame,
      "RW09",
    );
    const finalRoute = enriched.find((route) => route.branchId === "TEST-R09-RW09:branch:R");

    expect(finalRoute?.points).toHaveLength(3);
    expect(finalRoute?.assessmentSegments).toHaveLength(1);
    expect(finalRoute?.assessmentSegments?.[0]).toMatchObject({
      segmentId: "TEST-R09-RW09:branch:R:segment:final:1",
      finalVerticalReference: {
        kind: "FINAL_VERTICAL_REFERENCE",
        label: "GPA 3.0 deg",
      },
      lnavVnavOcs: {
        kind: "LNAV_VNAV_OCS",
        label: "LNAV/VNAV OCS",
      },
    });
    expect(finalRoute?.assessmentSegments?.[0].primaryHalfWidthM).toBeCloseTo(1203.8, 6);
    expect(finalRoute?.assessmentSegments?.[0].secondaryHalfWidthM).toBeCloseTo(1759.4, 6);
    expect(finalRoute?.assessmentSegments?.[0].finalVerticalReference?.halfWidthM).toBeCloseTo(
      555.6,
      6,
    );
    expect(finalRoute?.assessmentSegments?.[0].lnavVnavOcs?.primaryHalfWidthM).toBeCloseTo(
      926,
      6,
    );
    expect(finalRoute?.protectionSurfaces?.map((surface) => surface.surfaceId)).toEqual([
      "TEST-R09-RW09:branch:R:segment:final:1:lnav-oea",
      "TEST-R09-RW09:branch:R:segment:final:1:lnav-vnav-ocs",
    ]);
  });
});
