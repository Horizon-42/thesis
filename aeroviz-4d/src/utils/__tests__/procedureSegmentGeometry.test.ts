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
import { buildRfLeg, buildRfParallelEnvelope } from "../procedureRfGeometry";
import { METERS_PER_NM, distanceNm, offsetPoint } from "../procedureGeoMath";

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

const rfCenter = { lonDeg: -78.86, latDeg: 35.84, altM: 0 };
const rfStart = offsetPoint(rfCenter, Math.PI / 2, 2 * METERS_PER_NM);
const rfEnd = offsetPoint(rfCenter, 0, 2 * METERS_PER_NM);
const rfFixes = new Map<string, ProcedurePackageFix>([
  [
    "fix:RF_START",
    {
      fixId: "fix:RF_START",
      ident: "RFSTART",
      role: ["IF"],
      lonDeg: rfStart.lonDeg,
      latDeg: rfStart.latDeg,
      altFtMsl: 3000,
      annotations: [],
      sourceRefs: [],
    },
  ],
  [
    "fix:RF_END",
    {
      fixId: "fix:RF_END",
      ident: "RFEND",
      role: ["FAF"],
      lonDeg: rfEnd.lonDeg,
      latDeg: rfEnd.latDeg,
      altFtMsl: 2200,
      annotations: [],
      sourceRefs: [],
    },
  ],
]);

const rfLeg: ProcedurePackageLeg = {
  legId: "leg:R:RF",
  segmentId: "segment:rf",
  legType: "RF",
  rawPathTerminator: "RF",
  startFixId: "fix:RF_START",
  endFixId: "fix:RF_END",
  turnDirection: "LEFT",
  arcRadiusNm: 2,
  centerLatDeg: rfCenter.latDeg,
  centerLonDeg: rfCenter.lonDeg,
  requiredAltitude: null,
  requiredSpeed: null,
  navSpecAtLeg: "RNP_APCH",
  xttNm: 0.3,
  attNm: 0.3,
  secondaryEnabled: true,
  notes: [],
  sourceRefs: [],
  legacy: {
    sequence: 10,
    constructionMethod: "radius_to_fix",
    roleAtEnd: "IF",
    qualityStatus: "exact",
    renderedInPlanView: true,
  },
};

const rfSegment: ProcedureSegment = {
  segmentId: "segment:rf",
  branchId: "branch:R",
  segmentType: "INTERMEDIATE",
  navSpec: "RNP_APCH",
  startFixId: "fix:RF_START",
  endFixId: "fix:RF_END",
  legIds: ["leg:R:RF"],
  xttNm: 0.3,
  attNm: 0.3,
  secondaryEnabled: true,
  widthChangeMode: "NONE",
  transitionRule: null,
  verticalRule: { kind: "LEVEL_ROC" },
  constructionFlags: {},
  sourceRefs: [],
  legacy: {
    rawSegmentType: "intermediate",
    sequenceRange: [10, 10],
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

  it("builds a sampled RF arc when radius, center, and direction metadata are present", () => {
    const result = buildRfLeg(rfLeg, rfFixes, {
      samplingStepNm: 0.5,
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).not.toBeNull();
    if (!result.geometry) return;

    expect(result.geometry.isArc).toBe(true);
    expect(result.geometry.geodesicLengthNm).toBeCloseTo(Math.PI, 2);
    expect(result.geometry.geoPositions.length).toBeGreaterThan(4);
    expect(result.geometry.geoPositions[0].lonDeg).toBeCloseTo(rfStart.lonDeg, 8);
    expect(result.geometry.geoPositions[0].latDeg).toBeCloseTo(rfStart.latDeg, 8);
    const lastPoint = result.geometry.geoPositions[result.geometry.geoPositions.length - 1];
    expect(lastPoint.lonDeg).toBeCloseTo(rfEnd.lonDeg, 8);
    expect(lastPoint.latDeg).toBeCloseTo(rfEnd.latDeg, 8);

    const maxRadiusErrorNm = Math.max(
      ...result.geometry.geoPositions.map((point) =>
        Math.abs(distanceNm({ ...rfCenter, altM: point.altM }, point) - 2),
      ),
    );
    expect(maxRadiusErrorNm).toBeLessThan(0.005);
  });

  it("returns RF diagnostics instead of constructing arcs without required metadata", () => {
    const result = buildRfLeg(
      {
        ...rfLeg,
        arcRadiusNm: undefined,
      },
      rfFixes,
      {
        samplingStepNm: 0.5,
      },
    );

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "RF_RADIUS_MISSING",
        severity: "ERROR",
      }),
    ]);
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

  it("integrates RF legs into segment bundles without adding TF visual turn fills", () => {
    const bundle = buildSegmentGeometryBundle(
      rfSegment,
      [rfLeg],
      rfFixes,
      {
        samplingStepNm: 0.5,
        enableDebugPrimitives: false,
      },
    );

    expect(bundle.diagnostics).toEqual([]);
    expect(bundle.centerline.isArc).toBe(true);
    expect(bundle.centerline.geodesicLengthNm).toBeCloseTo(Math.PI, 2);
    expect(bundle.primaryEnvelope).toBeDefined();
    expect(bundle.secondaryEnvelope).toBeDefined();
    expect(bundle.primaryEnvelope?.constructionKind).toBe("RF_PARALLEL_ARC");
    expect(bundle.primaryEnvelope?.rfEnvelopeCase).toBe("RF_CASE_1");
    expect(bundle.primaryEnvelope?.radialBoundsNm).toEqual({
      innerRadiusNm: 1.4,
      outerRadiusNm: 2.6,
      nominalRadiusNm: 2,
    });
    const leftRadiusErrors = bundle.primaryEnvelope?.leftGeoBoundary.map((point) =>
      Math.abs(distanceNm({ ...rfCenter, altM: point.altM }, point) - 1.4),
    ) ?? [];
    const rightRadiusErrors = bundle.primaryEnvelope?.rightGeoBoundary.map((point) =>
      Math.abs(distanceNm({ ...rfCenter, altM: point.altM }, point) - 2.6),
    ) ?? [];
    expect(Math.max(...leftRadiusErrors)).toBeLessThan(0.005);
    expect(Math.max(...rightRadiusErrors)).toBeLessThan(0.005);
    expect(bundle.turnJunctions).toEqual([]);
  });

  it("keeps RF Case 2 inner envelope geometry finite when the inside radius collapses", () => {
    const case2Center = { lonDeg: -78.86, latDeg: 35.84, altM: 0 };
    const case2Start = offsetPoint(case2Center, Math.PI / 2, 0.5 * METERS_PER_NM);
    const case2End = offsetPoint(case2Center, 0, 0.5 * METERS_PER_NM);
    const case2Fixes = new Map<string, ProcedurePackageFix>([
      [
        "fix:CASE2_START",
        {
          fixId: "fix:CASE2_START",
          ident: "CASE2START",
          role: ["IF"],
          lonDeg: case2Start.lonDeg,
          latDeg: case2Start.latDeg,
          altFtMsl: 3000,
          annotations: [],
          sourceRefs: [],
        },
      ],
      [
        "fix:CASE2_END",
        {
          fixId: "fix:CASE2_END",
          ident: "CASE2END",
          role: ["FAF"],
          lonDeg: case2End.lonDeg,
          latDeg: case2End.latDeg,
          altFtMsl: 2200,
          annotations: [],
          sourceRefs: [],
        },
      ],
    ]);
    const case2Leg: ProcedurePackageLeg = {
      ...rfLeg,
      startFixId: "fix:CASE2_START",
      endFixId: "fix:CASE2_END",
      arcRadiusNm: 0.5,
      centerLatDeg: case2Center.latDeg,
      centerLonDeg: case2Center.lonDeg,
    };
    const centerline = buildRfLeg(case2Leg, case2Fixes, {
      samplingStepNm: 0.25,
    }).geometry;
    expect(centerline).not.toBeNull();
    if (!centerline) return;

    const envelope = buildRfParallelEnvelope(
      "segment:rf-case2:primary",
      "PRIMARY",
      centerline,
      0.6,
    );

    expect(envelope).not.toBeNull();
    if (!envelope) return;
    expect(envelope.rfEnvelopeCase).toBe("RF_CASE_2_INNER_COLLAPSED");
    expect(envelope.radialBoundsNm).toEqual({
      innerRadiusNm: 0,
      outerRadiusNm: 1.1,
      nominalRadiusNm: 0.5,
    });
    expect(
      envelope.leftGeoBoundary.every((point) =>
        Number.isFinite(point.lonDeg) && Number.isFinite(point.latDeg),
      ),
    ).toBe(true);
    expect(
      Math.max(
        ...envelope.leftGeoBoundary.map((point) =>
          distanceNm({ ...case2Center, altM: point.altM }, point),
        ),
      ),
    ).toBeLessThan(0.005);
    expect(
      Math.max(
        ...envelope.rightGeoBoundary.map((point) =>
          Math.abs(distanceNm({ ...case2Center, altM: point.altM }, point) - 1.1),
        ),
      ),
    ).toBeLessThan(0.005);
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

  it("builds straight DF missed approach geometry when start and end fixes are positioned", () => {
    const missedSegment: ProcedureSegment = {
      ...rfSegment,
      segmentId: "segment:missed-s1",
      segmentType: "MISSED_S1",
      legIds: ["leg:R:040"],
      startFixId: "fix:RW",
      endFixId: "fix:B",
      xttNm: 1,
      attNm: 1,
      legacy: {
        rawSegmentType: "missed_s1",
        sequenceRange: [40, 50],
      },
    };
    const dfLeg: ProcedurePackageLeg = {
      ...tfLeg,
      legId: "leg:R:040",
      segmentId: "segment:missed-s1",
      legType: "DF",
      rawPathTerminator: "DF",
      startFixId: "fix:RW",
      endFixId: "fix:B",
      legacy: {
        ...tfLeg.legacy,
        sequence: 40,
        constructionMethod: "direct_to_fix",
      },
    };

    const bundle = buildSegmentGeometryBundle(missedSegment, [dfLeg], fixes, {
      samplingStepNm: 1,
      enableDebugPrimitives: false,
    });

    expect(bundle.diagnostics).toEqual([]);
    expect(bundle.centerline.geoPositions.length).toBeGreaterThan(2);
    expect(bundle.centerline.geoPositions[0].lonDeg).toBeCloseTo(fixes.get("fix:RW")?.lonDeg ?? 0, 8);
    expect(bundle.centerline.geoPositions[0].latDeg).toBeCloseTo(fixes.get("fix:RW")?.latDeg ?? 0, 8);
    expect(bundle.primaryEnvelope).toBeDefined();
    expect(bundle.secondaryEnvelope).toBeDefined();
  });

  it("diagnoses preserved but unsupported missed approach leg geometry", () => {
    const missedSegment: ProcedureSegment = {
      ...rfSegment,
      segmentId: "segment:missed-s1",
      segmentType: "MISSED_S1",
      legIds: ["leg:R:040", "leg:R:050"],
      startFixId: "fix:RW",
      endFixId: "fix:B",
      xttNm: 1,
      attNm: 1,
      legacy: {
        rawSegmentType: "missed_s1",
        sequenceRange: [40, 50],
      },
    };
    const caLeg: ProcedurePackageLeg = {
      ...tfLeg,
      legId: "leg:R:040",
      segmentId: "segment:missed-s1",
      legType: "CA",
      rawPathTerminator: "CA",
      startFixId: "fix:RW",
      endFixId: "fix:B",
      legacy: {
        ...tfLeg.legacy,
        sequence: 40,
        constructionMethod: "course_to_altitude",
      },
    };
    const hmLeg: ProcedurePackageLeg = {
      ...caLeg,
      legId: "leg:R:050",
      legType: "HM",
      rawPathTerminator: "HM",
      legacy: {
        ...caLeg.legacy,
        sequence: 50,
        constructionMethod: "hold_to_manual",
      },
    };

    const bundle = buildSegmentGeometryBundle(missedSegment, [caLeg, hmLeg], fixes, {
      samplingStepNm: 1,
      enableDebugPrimitives: false,
    });

    expect(bundle.centerline.geoPositions).toEqual([]);
    expect(bundle.primaryEnvelope).toBeUndefined();
    expect(bundle.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "UNSUPPORTED_LEG_TYPE",
      "UNSUPPORTED_LEG_TYPE",
      "UNSUPPORTED_LEG_TYPE",
    ]);
    expect(bundle.diagnostics[0].message).toContain("CA is preserved");
    expect(bundle.diagnostics[1].message).toContain("HM is preserved");
  });
});
