import { describe, expect, it } from "vitest";
import type {
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../../data/procedurePackage";
import {
  buildSegmentGeometryBundle,
  buildStraightEnvelope,
  buildTfLeg,
  computeStationAxis,
} from "../procedureSegmentGeometry";

const fixes = new Map<string, ProcedurePackageFix>([
  [
    "fix:A",
    {
      fixId: "fix:A",
      ident: "A",
      role: ["IF"],
      lonDeg: -78.9,
      latDeg: 35.8,
      altFtMsl: 3000,
      annotations: [],
      sourceRefs: [],
    },
  ],
  [
    "fix:B",
    {
      fixId: "fix:B",
      ident: "B",
      role: ["FAF"],
      lonDeg: -78.84,
      latDeg: 35.84,
      altFtMsl: 2200,
      annotations: [],
      sourceRefs: [],
    },
  ],
  [
    "fix:RW",
    {
      fixId: "fix:RW",
      ident: "RW05L",
      role: ["MAP", "RWY"],
      lonDeg: -78.8,
      latDeg: 35.87,
      altFtMsl: 800,
      annotations: [],
      sourceRefs: [],
    },
  ],
]);

const tfLeg: ProcedurePackageLeg = {
  legId: "leg:R:020",
  segmentId: "segment:final",
  legType: "TF",
  rawPathTerminator: "TF",
  startFixId: "fix:A",
  endFixId: "fix:B",
  requiredAltitude: null,
  requiredSpeed: null,
  navSpecAtLeg: "RNP_APCH",
  xttNm: 0.3,
  attNm: 0.3,
  secondaryEnabled: true,
  notes: [],
  sourceRefs: [],
  legacy: {
    sequence: 20,
    constructionMethod: "track_to_fix",
    roleAtEnd: "FAF",
    qualityStatus: "exact",
    renderedInPlanView: true,
  },
};

const finalSegment: ProcedureSegment = {
  segmentId: "segment:final",
  branchId: "branch:R",
  segmentType: "FINAL_LNAV",
  navSpec: "RNP_APCH",
  startFixId: "fix:A",
  endFixId: "fix:RW",
  legIds: ["leg:R:020", "leg:R:030"],
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
    sequenceRange: [20, 30],
  },
};

const secondTfLeg: ProcedurePackageLeg = {
  ...tfLeg,
  legId: "leg:R:030",
  startFixId: "fix:B",
  endFixId: "fix:RW",
  legacy: {
    ...tfLeg.legacy,
    sequence: 30,
    roleAtEnd: "MAPt",
  },
};

describe("procedure segment geometry kernel", () => {
  it("builds a sampled geodesic TF centerline without Cesium", () => {
    const result = buildTfLeg(tfLeg, fixes, {
      samplingStepNm: 1,
      enableDebugPrimitives: false,
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).not.toBeNull();
    expect(result.geometry?.isArc).toBe(false);
    expect(result.geometry?.geodesicLengthNm).toBeGreaterThan(3.5);
    expect(result.geometry?.geodesicLengthNm).toBeLessThan(4.5);
    const firstPoint = result.geometry?.geoPositions[0];
    const lastPoint = result.geometry?.geoPositions[result.geometry.geoPositions.length - 1];
    expect(firstPoint?.lonDeg).toBeCloseTo(-78.9, 8);
    expect(firstPoint?.latDeg).toBeCloseTo(35.8, 8);
    expect(lastPoint?.lonDeg).toBeCloseTo(-78.84, 8);
    expect(lastPoint?.latDeg).toBeCloseTo(35.84, 8);
  });

  it("computes station axis and straight primary envelope samples", () => {
    const result = buildTfLeg(tfLeg, fixes, {
      samplingStepNm: 1,
      enableDebugPrimitives: false,
    });
    const geometry = result.geometry;
    expect(geometry).not.toBeNull();
    if (!geometry) return;

    const stationAxis = computeStationAxis(geometry);
    const envelope = buildStraightEnvelope("segment:final:primary", "PRIMARY", geometry, 0.6);

    expect(stationAxis.samples[0].stationNm).toBe(0);
    expect(stationAxis.totalLengthNm).toBeCloseTo(geometry.geodesicLengthNm, 3);
    expect(envelope.envelopeType).toBe("PRIMARY");
    expect(envelope.leftBoundary).toHaveLength(geometry.worldPositions.length);
    expect(envelope.rightBoundary).toHaveLength(geometry.worldPositions.length);
    expect(envelope.halfWidthNmSamples.every((sample) => sample.halfWidthNm === 0.6)).toBe(true);
  });

  it("builds a first-pass segment bundle for TF-only final segments", () => {
    const bundle = buildSegmentGeometryBundle(
      finalSegment,
      [tfLeg, secondTfLeg],
      fixes,
      {
        samplingStepNm: 1,
        enableDebugPrimitives: false,
      },
    );

    expect(bundle.segmentId).toBe("segment:final");
    expect(bundle.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["FINAL_HAS_TURN"]);
    expect(bundle.centerline.geodesicLengthNm).toBeGreaterThan(6);
    expect(bundle.stationAxis.totalLengthNm).toBeCloseTo(bundle.centerline.geodesicLengthNm, 3);
    expect(bundle.primaryEnvelope?.halfWidthNmSamples[0].halfWidthNm).toBe(0.6);
    expect(bundle.secondaryEnvelope?.halfWidthNmSamples[0].halfWidthNm).toBeCloseTo(0.9, 8);
    expect(bundle.turnJunctions).toHaveLength(1);
  });

  it("marks visual turn-fill patches inside final segments as diagnostics", () => {
    const sharpTurnFixes = new Map(fixes);
    sharpTurnFixes.set("fix:C", {
      fixId: "fix:C",
      ident: "C",
      role: ["MAP"],
      lonDeg: -78.84,
      latDeg: 35.9,
      altFtMsl: 1000,
      annotations: [],
      sourceRefs: [],
    });
    const sharpSecondLeg: ProcedurePackageLeg = {
      ...secondTfLeg,
      endFixId: "fix:C",
    };

    const bundle = buildSegmentGeometryBundle(
      finalSegment,
      [tfLeg, sharpSecondLeg],
      sharpTurnFixes,
      {
        samplingStepNm: 10,
        enableDebugPrimitives: false,
      },
    );

    expect(bundle.turnJunctions).toHaveLength(1);
    expect(bundle.turnJunctions[0].constructionStatus).toBe("VISUAL_FILL_ONLY");
    expect(bundle.diagnostics.map((diagnostic) => diagnostic.code)).toContain("FINAL_HAS_TURN");
  });
});
