/**
 * Shared test fixture: a compact KRDU RNAV(GPS) Y RWY 05L detail document
 * (SCHOO IF -> WEPAS FAF -> RW05L threshold). Used by the procedureConstraint
 * unit tests and the PilotPanel constrained-optimize integration test.
 * Not a test file itself (vitest only collects *.test.* / *.spec.*).
 */
import type {
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailLeg,
} from "../procedureDetails";

export function leg(
  partial: Pick<ProcedureDetailLeg, "legId" | "sequence" | "path" | "roleAtEnd"> &
    Partial<ProcedureDetailLeg>,
): ProcedureDetailLeg {
  return {
    segmentType: "final",
    termination: { kind: "fix", fixRef: partial.path.endFixRef },
    constraints: { altitude: null, speedKt: null, geometryAltitudeFt: null },
    sourceRefs: ["src:cifp-detail"],
    quality: { status: "exact", sourceLine: partial.sequence, renderedInPlanView: true },
    ...partial,
  };
}

export const krduR05lyFinalBranch: ProcedureDetailBranch = {
  branchId: "branch:R",
  branchKey: "R",
  branchIdent: "R",
  branchRole: "final",
  sequenceOrder: 1,
  mergeFixRef: null,
  continuesWithBranchId: null,
  defaultVisible: true,
  warnings: [],
  legs: [
    leg({
      legId: "leg:R:010",
      sequence: 10,
      path: { pathTerminator: "IF", constructionMethod: "if_to_fix", startFixRef: null, endFixRef: "fix:SCHOO" },
      roleAtEnd: "IF",
      constraints: {
        altitude: { qualifier: "atOrAbove", valueFt: 3000, rawText: "3000 ft" },
        speedKt: null,
        geometryAltitudeFt: 3000,
      },
    }),
    leg({
      legId: "leg:R:020",
      sequence: 20,
      path: { pathTerminator: "TF", constructionMethod: "track_to_fix", startFixRef: "fix:SCHOO", endFixRef: "fix:WEPAS" },
      roleAtEnd: "FAF",
      constraints: {
        altitude: { qualifier: "atOrAbove", valueFt: 2200, rawText: "2200 ft" },
        speedKt: 170,
        geometryAltitudeFt: 2200,
      },
    }),
    leg({
      legId: "leg:R:030",
      sequence: 30,
      path: { pathTerminator: "TF", constructionMethod: "track_to_fix", startFixRef: "fix:WEPAS", endFixRef: "fix:RW05L" },
      roleAtEnd: "MAPt",
      constraints: {
        altitude: { qualifier: "at", valueFt: 424, rawText: "424 ft" },
        speedKt: null,
        geometryAltitudeFt: 367,
      },
    }),
  ],
};

export const krduR05lyDocument: ProcedureDetailDocument = {
  schemaVersion: "1.0.0",
  modelType: "rnav-procedure-runway",
  procedureUid: "KRDU-R05LY-RW05L",
  provenance: { assemblyMode: "cifp_primary_export", researchUseOnly: true, sources: [], warnings: [] },
  airport: { icao: "KRDU", faa: "RDU", name: "Raleigh-Durham International Airport" },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: "fix:RW05L",
    threshold: { lon: -78.80196389, lat: 35.87445, elevationFt: 367 },
  },
  procedure: {
    procedureType: "SIAP",
    procedureFamily: "RNAV_GPS",
    procedureIdent: "R05LY",
    chartName: "RNAV(GPS) Y RWY 05L",
    variant: "Y",
    runwayIdent: "RW05L",
    baseBranchIdent: "R",
    approachModes: ["LPV", "LNAV/VNAV", "LNAV"],
  },
  fixes: [
    { fixId: "fix:SCHOO", ident: "SCHOO", kind: "named_fix", position: { lon: -78.92647222, lat: 35.77341389 }, elevationFt: null, roleHints: ["IF"], sourceRefs: [] },
    { fixId: "fix:WEPAS", ident: "WEPAS", kind: "final_approach_fix", position: { lon: -78.88295556, lat: 35.80876667 }, elevationFt: null, roleHints: ["FAF"], sourceRefs: [] },
    { fixId: "fix:RW05L", ident: "RW05L", kind: "runway_threshold", position: { lon: -78.80196389, lat: 35.87445 }, elevationFt: 367, roleHints: ["MAPt"], sourceRefs: [] },
  ],
  branches: [krduR05lyFinalBranch],
  verticalProfiles: [
    {
      profileId: "vp:final",
      appliesToModes: ["LPV", "LNAV/VNAV", "LNAV"],
      branchId: "branch:R",
      fromFixRef: "fix:SCHOO",
      toFixRef: "fix:RW05L",
      basis: "cifp_leg_constraints",
      glidepathAngleDeg: 3.0,
      thresholdCrossingHeightFt: 57,
      constraintSamples: [],
      warnings: [],
    },
  ],
  validation: {
    expectedRunwayIdent: "RW05L",
    expectedIF: "fix:SCHOO",
    expectedFAF: "fix:WEPAS",
    expectedMAPt: "fix:RW05L",
    expectedMissedHoldFix: null,
    knownSimplifications: [],
  },
  displayHints: {
    nominalSpeedKt: 140,
    defaultVisibleBranchIds: ["branch:R"],
    tunnelDefaults: { lateralHalfWidthNm: 0.3, verticalHalfHeightFt: 300, sampleSpacingM: 250, mode: "visualApproximation" },
  },
};
