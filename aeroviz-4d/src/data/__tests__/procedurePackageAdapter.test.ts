import { describe, expect, it } from "vitest";
import type { ProcedureDetailDocument } from "../procedureDetails";
import { normalizeProcedurePackage } from "../procedurePackageAdapter";

const sampleDocument: ProcedureDetailDocument = {
  schemaVersion: "1.0.0",
  modelType: "rnav-procedure-runway",
  procedureUid: "KRDU-R05LY-RW05L",
  provenance: {
    assemblyMode: "cifp_primary_export",
    researchUseOnly: true,
    sources: [
      {
        sourceId: "src:cifp",
        kind: "cifp",
        cycle: "2603",
        path: "data/CIFP/CIFP_260319/FAACIFP18",
      },
    ],
    warnings: ["CA missed approach leg is parsed but not geometry-supported yet."],
  },
  airport: {
    icao: "KRDU",
    faa: "RDU",
    name: "Raleigh-Durham International Airport",
  },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: "fix:RW05L",
    threshold: {
      lon: -78.80196389,
      lat: 35.87445,
      elevationFt: 798,
    },
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
    {
      fixId: "fix:SCHOO",
      ident: "SCHOO",
      kind: "named_fix",
      position: { lon: -78.92647222, lat: 35.77341389 },
      elevationFt: null,
      roleHints: ["IF"],
      sourceRefs: ["src:cifp-detail:1"],
    },
    {
      fixId: "fix:WEPAS",
      ident: "WEPAS",
      kind: "final_approach_fix",
      position: { lon: -78.88295556, lat: 35.80876667 },
      elevationFt: null,
      roleHints: ["FAF"],
      sourceRefs: ["src:cifp-detail:2"],
    },
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      kind: "runway_threshold",
      position: { lon: -78.80196389, lat: 35.87445 },
      elevationFt: 798,
      roleHints: ["MAPt"],
      sourceRefs: ["src:cifp-detail:3"],
    },
  ],
  branches: [
    {
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
        {
          legId: "leg:R:010",
          sequence: 10,
          segmentType: "intermediate",
          path: {
            pathTerminator: "IF",
            constructionMethod: "if_to_fix",
            startFixRef: null,
            endFixRef: "fix:SCHOO",
          },
          termination: { kind: "fix", fixRef: "fix:SCHOO" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 3000, rawText: "3000 ft" },
            speedKt: null,
            geometryAltitudeFt: 3000,
          },
          roleAtEnd: "IF",
          sourceRefs: ["src:cifp-detail:1"],
          quality: { status: "exact", sourceLine: 1, renderedInPlanView: true },
        },
        {
          legId: "leg:R:020",
          sequence: 20,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track_to_fix",
            startFixRef: "fix:SCHOO",
            endFixRef: "fix:WEPAS",
          },
          termination: { kind: "fix", fixRef: "fix:WEPAS" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 2200, rawText: "2200 ft" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "FAF",
          sourceRefs: ["src:cifp-detail:2"],
          quality: { status: "exact", sourceLine: 2, renderedInPlanView: true },
        },
        {
          legId: "leg:R:030",
          sequence: 30,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track_to_fix",
            startFixRef: "fix:WEPAS",
            endFixRef: "fix:RW05L",
          },
          termination: { kind: "fix", fixRef: "fix:RW05L" },
          constraints: {
            altitude: null,
            speedKt: null,
            geometryAltitudeFt: 798,
          },
          roleAtEnd: "MAPt",
          sourceRefs: ["src:cifp-detail:3"],
          quality: { status: "exact", sourceLine: 3, renderedInPlanView: true },
        },
      ],
    },
  ],
  verticalProfiles: [],
  validation: {
    expectedRunwayIdent: "RW05L",
    expectedIF: "fix:SCHOO",
    expectedFAF: "fix:WEPAS",
    expectedMAPt: "fix:RW05L",
    expectedMissedHoldFix: null,
    knownSimplifications: ["RF legs are not exported in this sample."],
  },
  displayHints: {
    nominalSpeedKt: 140,
    defaultVisibleBranchIds: ["branch:R"],
    tunnelDefaults: {
      lateralHalfWidthNm: 0.3,
      verticalHalfHeightFt: 300,
      sampleSpacingM: 250,
      mode: "visualApproximation",
    },
  },
};

describe("normalizeProcedurePackage", () => {
  it("maps the current procedure-detail document into the v3 package hierarchy", () => {
    const pkg = normalizeProcedurePackage(sampleDocument);

    expect(pkg.packageId).toBe("KRDU-R05LY-RW05L");
    expect(pkg.airportId).toBe("KRDU");
    expect(pkg.runwayId).toBe("RW05L");
    expect(pkg.procedureFamily).toBe("RNAV_GPS");
    expect(pkg.sourceMeta).toMatchObject({
      cifpCycle: "2603",
      authority: "FAA_8260_58D",
      sourceFiles: ["data/CIFP/CIFP_260319/FAACIFP18"],
    });
    expect(pkg.sharedFixes.map((fix) => [fix.fixId, fix.role])).toEqual([
      ["fix:SCHOO", ["IF"]],
      ["fix:WEPAS", ["FAF"]],
      ["fix:RW05L", ["MAP", "RWY"]],
    ]);
    expect(pkg.branches).toEqual([
      expect.objectContaining({
        branchId: "branch:R",
        branchRole: "STRAIGHT_IN",
        segmentIds: [
          "branch:R:segment:intermediate:1",
          "branch:R:segment:final_lnav:2",
        ],
      }),
    ]);
    expect(pkg.segments.map((segment) => ({
      segmentId: segment.segmentId,
      segmentType: segment.segmentType,
      legIds: segment.legIds,
      xttNm: segment.xttNm,
      attNm: segment.attNm,
      verticalRule: segment.verticalRule?.kind,
    }))).toEqual([
      {
        segmentId: "branch:R:segment:intermediate:1",
        segmentType: "INTERMEDIATE",
        legIds: ["leg:R:010"],
        xttNm: 1,
        attNm: 1,
        verticalRule: "NONE",
      },
      {
        segmentId: "branch:R:segment:final_lnav:2",
        segmentType: "FINAL_LNAV",
        legIds: ["leg:R:020", "leg:R:030"],
        xttNm: 0.3,
        attNm: 0.3,
        verticalRule: "LPV_GLS_SURFACES",
      },
    ]);
    expect(pkg.legs.map((leg) => ({
      legId: leg.legId,
      segmentId: leg.segmentId,
      legType: leg.legType,
      altitude: leg.requiredAltitude,
    }))).toEqual([
      {
        legId: "leg:R:010",
        segmentId: "branch:R:segment:intermediate:1",
        legType: "IF",
        altitude: {
          kind: "AT",
          minFtMsl: 3000,
          maxFtMsl: 3000,
          sourceText: "3000 ft",
        },
      },
      {
        legId: "leg:R:020",
        segmentId: "branch:R:segment:final_lnav:2",
        legType: "TF",
        altitude: {
          kind: "AT",
          minFtMsl: 2200,
          maxFtMsl: 2200,
          sourceText: "2200 ft",
        },
      },
      {
        legId: "leg:R:030",
        segmentId: "branch:R:segment:final_lnav:2",
        legType: "TF",
        altitude: null,
      },
    ]);
  });

  it("surfaces migration diagnostics instead of silently inventing missing v3 data", () => {
    const pkg = normalizeProcedurePackage(sampleDocument);

    expect(pkg.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "DEFAULT_NAV_SPEC",
      "DEFAULT_TOLERANCE",
      "DEFAULT_NAV_SPEC",
      "DEFAULT_TOLERANCE",
      "MODE_COLLAPSED_TO_LNAV",
      "SOURCE_INCOMPLETE",
      "SOURCE_INCOMPLETE",
    ]);
    expect(
      pkg.diagnostics.find((diagnostic) => diagnostic.code === "MODE_COLLAPSED_TO_LNAV")
        ?.message,
    ).toContain("LPV / LNAV/VNAV / LNAV");
  });
});
