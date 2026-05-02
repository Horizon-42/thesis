import { describe, expect, it } from "vitest";
import type { ProcedureSegment } from "../../data/procedurePackage";
import type { PolylineGeometry3D } from "../procedureSegmentGeometry";
import { toCartesian, type GeoPoint } from "../procedureGeoMath";
import {
  buildFinalApproachSurfaceStatus,
  buildLnavFinalOea,
  buildLnavVnavOcs,
} from "../procedureSurfaceGeometry";

const finalGeoPositions: GeoPoint[] = [
  { lonDeg: -78.84, latDeg: 35.84, altM: 670.56 },
  { lonDeg: -78.8, latDeg: 35.87, altM: 243.84 },
];

const finalCenterline: PolylineGeometry3D = {
  geoPositions: finalGeoPositions,
  worldPositions: finalGeoPositions.map(toCartesian),
  geodesicLengthNm: 2.65,
  isArc: false,
};

const finalSegment: ProcedureSegment = {
  segmentId: "segment:final",
  branchId: "branch:R",
  segmentType: "FINAL_LNAV",
  navSpec: "RNP_APCH",
  startFixId: "fix:FAF",
  endFixId: "fix:RW",
  legIds: ["leg:R:030"],
  xttNm: 0.3,
  attNm: 0.3,
  secondaryEnabled: true,
  widthChangeMode: "LINEAR_TAPER",
  transitionRule: null,
  verticalRule: { kind: "LEVEL_ROC" },
  constructionFlags: {},
  sourceRefs: [],
  legacy: {
    rawSegmentType: "final",
    sequenceRange: [30, 30],
  },
};

function sampleAtStation(
  samples: Array<{ stationNm: number; halfWidthNm: number }>,
  stationNm: number,
): number {
  const sample = samples.reduce((nearest, candidate) =>
    Math.abs(candidate.stationNm - stationNm) < Math.abs(nearest.stationNm - stationNm)
      ? candidate
      : nearest,
  );
  return sample.halfWidthNm;
}

describe("procedure surface geometry", () => {
  it("builds the minimal LNAV final OEA anchors and taper for G-11", () => {
    const result = buildLnavFinalOea(finalSegment, finalCenterline, {
      samplingStepNm: 0.1,
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).not.toBeNull();
    if (!result.geometry) return;

    expect(result.geometry.surfaceType).toBe("LNAV_FINAL_OEA");
    expect(result.geometry.taper.startStationNm).toBeCloseTo(-0.3, 8);
    expect(result.geometry.taper.endStationNm).toBeCloseTo(1, 8);
    expect(result.geometry.centerline.geodesicLengthNm).toBeCloseTo(
      finalCenterline.geodesicLengthNm + 0.6,
      8,
    );

    const primarySamples = result.geometry.primary.halfWidthNmSamples;
    const secondarySamples = result.geometry.secondaryOuter.halfWidthNmSamples;
    expect(primarySamples[0].stationNm).toBeCloseTo(-0.3, 8);
    expect(primarySamples[primarySamples.length - 1].stationNm).toBeCloseTo(
      finalCenterline.geodesicLengthNm + 0.3,
      8,
    );
    expect(primarySamples[0].halfWidthNm).toBeCloseTo(0.3, 8);
    expect(sampleAtStation(primarySamples, 1)).toBeCloseTo(0.6, 8);
    expect(primarySamples[primarySamples.length - 1].halfWidthNm).toBeCloseTo(0.6, 8);
    expect(sampleAtStation(secondarySamples, 1)).toBeCloseTo(0.9, 8);
  });

  it("returns diagnostics instead of fake OEA geometry for incomplete centerlines", () => {
    const result = buildLnavFinalOea(finalSegment, {
      geoPositions: [],
      worldPositions: [],
      geodesicLengthNm: 0,
      isArc: false,
    });

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toHaveLength(1);
    expect(result.diagnostics[0].code).toBe("SOURCE_INCOMPLETE");
  });

  it("reports missing mode-specific final surfaces when modes are collapsed to LNAV", () => {
    const oea = buildLnavFinalOea(finalSegment, finalCenterline).geometry;
    const result = buildFinalApproachSurfaceStatus(
      {
        ...finalSegment,
        constructionFlags: { collapsedApproachModes: ["LPV", "LNAV/VNAV", "LNAV"] },
        verticalRule: { kind: "LPV_GLS_SURFACES" },
      },
      oea,
    );

    expect(result.status).toMatchObject({
      requestedModes: ["LPV", "LNAV/VNAV", "LNAV"],
      constructedSurfaceTypes: ["LNAV_FINAL_OEA"],
      missingSurfaceTypes: ["LPV_W", "LPV_X", "LPV_Y", "LNAV_VNAV_OCS"],
      constructionStatus: "COLLAPSED_TO_LNAV_BASELINE",
    });
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED",
        severity: "WARN",
      }),
    ]);
  });

  it("builds LNAV/VNAV OCS from explicit GPA and TCH source data", () => {
    const lnavVnavSegment: ProcedureSegment = {
      ...finalSegment,
      segmentType: "FINAL_LNAV_VNAV",
      verticalRule: { kind: "BARO_GLIDEPATH", gpaDeg: 3, tchFt: 50 },
    };
    const oea = buildLnavFinalOea(lnavVnavSegment, finalCenterline, {
      samplingStepNm: 0.25,
    }).geometry;
    const ocsResult = buildLnavVnavOcs(lnavVnavSegment, finalCenterline, oea, {
      samplingStepNm: 0.25,
    });

    expect(ocsResult.diagnostics).toEqual([]);
    expect(ocsResult.geometry).toMatchObject({
      surfaceType: "LNAV_VNAV_OCS",
      constructionStatus: "GPA_TCH_SLOPE_ESTIMATE",
      verticalProfile: {
        gpaDeg: 3,
        tchFt: 50,
      },
    });
    expect(ocsResult.geometry?.primary.leftGeoBoundary.length).toBeGreaterThan(2);
    const samples = ocsResult.geometry?.verticalProfile.samples ?? [];
    expect(samples[0].altitudeFtMsl).toBeGreaterThan(
      samples[samples.length - 1].altitudeFtMsl,
    );

    const statusResult = buildFinalApproachSurfaceStatus(
      lnavVnavSegment,
      oea,
      ocsResult.geometry,
    );
    expect(statusResult.status).toMatchObject({
      constructedSurfaceTypes: ["LNAV_FINAL_OEA", "LNAV_VNAV_OCS"],
      missingSurfaceTypes: [],
      constructionStatus: "MODE_SPECIFIC_SURFACES_CONSTRUCTED",
    });
  });

  it("diagnoses LNAV/VNAV OCS when GPA or TCH is missing", () => {
    const lnavVnavSegment: ProcedureSegment = {
      ...finalSegment,
      segmentType: "FINAL_LNAV_VNAV",
      verticalRule: { kind: "BARO_GLIDEPATH" },
    };
    const oea = buildLnavFinalOea(lnavVnavSegment, finalCenterline).geometry;
    const result = buildLnavVnavOcs(lnavVnavSegment, finalCenterline, oea);

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
        severity: "WARN",
      }),
    ]);
  });
});
