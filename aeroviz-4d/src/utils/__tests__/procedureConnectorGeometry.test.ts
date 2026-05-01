import { describe, expect, it } from "vitest";
import type { ProcedureSegment } from "../../data/procedurePackage";
import type { PolylineGeometry3D } from "../procedureSegmentGeometry";
import { toCartesian, type GeoPoint } from "../procedureGeoMath";
import { buildAlignedLnavConnector } from "../procedureConnectorGeometry";

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
  transitionRule: {
    kind: "INTERMEDIATE_TO_FINAL_LNAV",
    anchorFixId: "fix:FAF",
    beforeNm: 2,
    afterNm: 1,
    notes: [],
  },
  verticalRule: { kind: "LEVEL_ROC" },
  constructionFlags: {},
  sourceRefs: [],
  legacy: {
    rawSegmentType: "final",
    sequenceRange: [30, 30],
  },
};

describe("procedure connector geometry", () => {
  it("builds an aligned LNAV connector with PFAF -2 NM and +1 NM anchors for G-09", () => {
    const result = buildAlignedLnavConnector(finalSegment, finalCenterline, {
      samplingStepNm: 0.25,
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).not.toBeNull();
    if (!result.geometry) return;

    expect(result.geometry.connectorType).toBe("ALIGNED_LNAV_INTERMEDIATE_TO_FINAL");
    expect(result.geometry.anchors.startStationNm).toBe(-2);
    expect(result.geometry.anchors.endStationNm).toBe(1);
    expect(result.geometry.centerline.geodesicLengthNm).toBe(3);
    expect(result.geometry.primary.leftBoundary.length).toBe(
      result.geometry.primary.rightBoundary.length,
    );
    expect(result.geometry.primary.leftBoundary.length).toBeGreaterThan(2);

    const primarySamples = result.geometry.primary.halfWidthNmSamples;
    const secondarySamples = result.geometry.secondaryOuter.halfWidthNmSamples;
    expect(primarySamples[0].stationNm).toBe(-2);
    expect(primarySamples[0].halfWidthNm).toBe(2);
    expect(primarySamples[primarySamples.length - 1].stationNm).toBe(1);
    expect(primarySamples[primarySamples.length - 1].halfWidthNm).toBeCloseTo(0.6, 8);
    expect(secondarySamples[0].halfWidthNm).toBe(3);
    expect(secondarySamples[secondarySamples.length - 1].halfWidthNm).toBeCloseTo(0.9, 8);
  });

  it("returns diagnostics for connector inputs without a usable final centerline", () => {
    const result = buildAlignedLnavConnector(finalSegment, {
      geoPositions: [],
      worldPositions: [],
      geodesicLengthNm: 0,
      isArc: false,
    });

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toHaveLength(1);
    expect(result.diagnostics[0].code).toBe("CONNECTOR_NOT_CONSTRUCTIBLE");
  });
});
