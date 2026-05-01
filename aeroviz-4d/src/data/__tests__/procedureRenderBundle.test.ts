import { describe, expect, it } from "vitest";
import type { ProcedurePackage } from "../procedurePackage";
import { buildProcedureRenderBundle } from "../procedureRenderBundle";

const samplePackage: ProcedurePackage = {
  packageId: "KRDU-R05LY-RW05L",
  airportId: "KRDU",
  runwayId: "RW05L",
  procedureId: "R05LY",
  procedureName: "RNAV(GPS) Y RW05L",
  procedureFamily: "RNAV_GPS",
  sourceMeta: {
    cifpCycle: "2501",
    sourceFiles: [],
    chartLinks: [],
    notes: [],
    authority: "AEROVIZ_SOURCE",
  },
  branches: [
    {
      branchId: "branch:R",
      runwayId: "RW05L",
      branchName: "RW05L",
      branchRole: "STRAIGHT_IN",
      segmentIds: ["segment:final"],
      legacy: {
        branchIdent: "R",
        branchKey: "R",
        defaultVisible: true,
        mergeFixRef: null,
        continuesWithBranchId: null,
      },
    },
  ],
  segments: [
    {
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
    },
  ],
  legs: [
    {
      legId: "leg:R:030",
      segmentId: "segment:final",
      legType: "TF",
      rawPathTerminator: "TF",
      startFixId: "fix:FAF",
      endFixId: "fix:RW",
      requiredAltitude: null,
      requiredSpeed: null,
      navSpecAtLeg: "RNP_APCH",
      xttNm: 0.3,
      attNm: 0.3,
      secondaryEnabled: true,
      notes: [],
      sourceRefs: [],
      legacy: {
        sequence: 30,
        constructionMethod: "track_to_fix",
        roleAtEnd: "MAPt",
        qualityStatus: "exact",
        renderedInPlanView: true,
      },
    },
  ],
  sharedFixes: [
    {
      fixId: "fix:FAF",
      ident: "FAF",
      role: ["FAF", "PFAF"],
      lonDeg: -78.84,
      latDeg: 35.84,
      altFtMsl: 2200,
      annotations: [],
      sourceRefs: [],
    },
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
  validationConfig: {
    expectedRunwayIdent: "RW05L",
    expectedIF: null,
    expectedFAF: "FAF",
    expectedMAP: "RW05L",
    expectedMissedHoldFix: null,
    knownSimplifications: [],
  },
  diagnostics: [],
  legacyDocument: {
    schemaVersion: "1.0",
    modelType: "rnav-procedure-runway",
    procedureUid: "KRDU-R05LY-RW05L",
  },
};

describe("procedure render bundle", () => {
  it("aggregates segment, OEA, and aligned connector geometry from a ProcedurePackage", () => {
    const bundle = buildProcedureRenderBundle(samplePackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });

    expect(bundle.packageId).toBe("KRDU-R05LY-RW05L");
    expect(bundle.branchBundles).toHaveLength(1);
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];
    expect(segmentBundle.segment.segmentId).toBe("segment:final");
    expect(segmentBundle.segmentGeometry.centerline.geoPositions.length).toBeGreaterThan(2);
    expect(segmentBundle.finalOea?.surfaceType).toBe("LNAV_FINAL_OEA");
    expect(segmentBundle.alignedConnector?.connectorType).toBe(
      "ALIGNED_LNAV_INTERMEDIATE_TO_FINAL",
    );
    expect(bundle.diagnostics).toEqual([]);
  });
});
