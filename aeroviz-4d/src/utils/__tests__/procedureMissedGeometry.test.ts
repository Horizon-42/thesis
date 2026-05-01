import { describe, expect, it } from "vitest";
import type { ProcedureSegment } from "../../data/procedurePackage";
import type { SegmentGeometryBundle } from "../procedureSegmentGeometry";
import { buildMissedSectionSurface } from "../procedureMissedGeometry";

const missedSegment: ProcedureSegment = {
  segmentId: "segment:missed-s1",
  branchId: "branch:R",
  segmentType: "MISSED_S1",
  navSpec: "RNP_APCH",
  startFixId: "fix:RW",
  endFixId: "fix:MIS1",
  legIds: ["leg:R:040"],
  xttNm: 1,
  attNm: 1,
  secondaryEnabled: true,
  widthChangeMode: "NONE",
  transitionRule: null,
  verticalRule: { kind: "MISSED_CLIMB_SURFACE" },
  constructionFlags: {},
  sourceRefs: [],
  legacy: {
    rawSegmentType: "missed_s1",
    sequenceRange: [40, 40],
  },
};

const envelope = {
  geometryId: "segment:missed-s1:primary",
  envelopeType: "PRIMARY" as const,
  leftBoundary: [],
  rightBoundary: [],
  leftGeoBoundary: [
    { lonDeg: -78.8, latDeg: 35.87, altM: 250 },
    { lonDeg: -78.74, latDeg: 35.91, altM: 550 },
  ],
  rightGeoBoundary: [
    { lonDeg: -78.79, latDeg: 35.87, altM: 250 },
    { lonDeg: -78.73, latDeg: 35.91, altM: 550 },
  ],
  halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 2 }],
};

const geometryBundle: SegmentGeometryBundle = {
  segmentId: "segment:missed-s1",
  centerline: {
    worldPositions: [],
    geoPositions: [
      { lonDeg: -78.8, latDeg: 35.87, altM: 250 },
      { lonDeg: -78.74, latDeg: 35.91, altM: 550 },
    ],
    geodesicLengthNm: 4,
    isArc: false,
  },
  stationAxis: { samples: [], totalLengthNm: 4 },
  primaryEnvelope: envelope,
  secondaryEnvelope: { ...envelope, geometryId: "segment:missed-s1:secondary", envelopeType: "SECONDARY" },
  turnJunctions: [],
  diagnostics: [],
};

describe("procedure missed geometry", () => {
  it("wraps missed section one envelopes as an independent surface object", () => {
    const result = buildMissedSectionSurface(missedSegment, geometryBundle);

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).toMatchObject({
      segmentId: "segment:missed-s1",
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      primary: expect.objectContaining({ geometryId: "segment:missed-s1:primary" }),
      secondaryOuter: expect.objectContaining({ geometryId: "segment:missed-s1:secondary" }),
    });
  });

  it("diagnoses missed sections that have no primary envelope", () => {
    const result = buildMissedSectionSurface(missedSegment, {
      ...geometryBundle,
      primaryEnvelope: undefined,
    });

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
        severity: "WARN",
      }),
    ]);
  });
});
