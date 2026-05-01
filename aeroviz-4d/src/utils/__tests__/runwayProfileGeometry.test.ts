import { describe, expect, it } from "vitest";
import {
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
});
