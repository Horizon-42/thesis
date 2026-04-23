import { describe, expect, it } from "vitest";
import {
  buildHorizontalPlateRoutes,
  buildRunwayReferenceMarks,
  buildRunwayFrame,
  pointIsInsideHorizontalPlate,
  projectPositionToRunwayFrame,
  type RunwayFeatureCollection,
} from "../runwayProfileGeometry";
import type { ProcedureFeatureCollection } from "../../types/geojson-aviation";

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

const procedureCollection: ProcedureFeatureCollection = {
  type: "FeatureCollection",
  metadata: {
    airport: "TEST",
    sourceCycle: "2603",
  },
  features: [
    {
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: [
          [-0.010, 0.0, 500],
          [-0.005, 0.0, 300],
          [0.0, 0.0, 35],
        ],
      },
      properties: {
        featureType: "procedure-route",
        airport: "TEST",
        procedureType: "SIAP",
        procedureIdent: "R09",
        procedureName: "RNAV(GPS) RW09",
        branch: "R",
        branchIdent: "R",
        branchType: "final",
        procedureFamily: "RNAV_GPS",
        runway: "RW09",
        runwayIdent: "RW09",
        routeId: "TEST-R09-R",
        source: "FAA-CIFP",
        sourceCycle: "2603",
        researchUseOnly: true,
        nominalSpeedKt: 140,
        tunnel: {
          lateralHalfWidthNm: 0.3,
          verticalHalfHeightFt: 300,
          sampleSpacingM: 250,
        },
        samples: [
          {
            sequence: 10,
            fixIdent: "SCHOO",
            legType: "IF",
            role: "IF",
            altitudeFt: 1600,
            geometryAltitudeFt: 1600,
            distanceFromStartM: 0,
            timeSeconds: 0,
            sourceLine: 1,
          },
          {
            sequence: 20,
            fixIdent: "WEPAS",
            legType: "TF",
            role: "FAF",
            altitudeFt: 900,
            geometryAltitudeFt: 900,
            distanceFromStartM: 500,
            timeSeconds: 50,
            sourceLine: 2,
          },
          {
            sequence: 30,
            fixIdent: "RW09",
            legType: "TF",
            role: "MAPt",
            altitudeFt: 110,
            geometryAltitudeFt: 110,
            distanceFromStartM: 1000,
            timeSeconds: 100,
            sourceLine: 3,
          },
        ],
        warnings: [],
      },
    },
  ],
};

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
    const routes = buildHorizontalPlateRoutes(procedureCollection, frame, "RW09");

    const inside = projectPositionToRunwayFrame(frame, -0.0045, 0.00003, 220);
    const outside = projectPositionToRunwayFrame(frame, -0.0045, 0.008, 220);

    expect(routes).toHaveLength(1);
    expect(pointIsInsideHorizontalPlate(inside, routes)).toBe(true);
    expect(pointIsInsideHorizontalPlate(outside, routes)).toBe(false);
  });

  it("builds x-axis reference marks from important runway procedure fixes", () => {
    const frame = buildRunwayFrame(runwayCollection, "RW09");
    const marks = buildRunwayReferenceMarks(procedureCollection, frame, "RW09");

    expect(marks.some((mark) => mark.label === "RW09" && mark.detail === "Threshold")).toBe(true);
    expect(marks.some((mark) => mark.label === "RW09" && mark.detail === "MAPt")).toBe(true);
    expect(marks.some((mark) => mark.label === "WEPAS" && mark.detail === "FAF")).toBe(true);
  });
});
