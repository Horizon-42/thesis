import { describe, expect, it } from "vitest";
import type { ProcedureDetailDocument } from "../procedureDetails";
import { buildProcedureRoutes } from "../procedureRoutes";

const sampleDocument: ProcedureDetailDocument = {
  schemaVersion: "1.0.0",
  modelType: "rnav-procedure-runway",
  procedureUid: "KRDU-R05LY-RW05L",
  provenance: {
    assemblyMode: "cifp_primary_export",
    researchUseOnly: true,
    sources: [],
    warnings: [],
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
      fixId: "fix:CHWDR",
      ident: "CHWDR",
      kind: "named_fix",
      position: { lon: -78.959, lat: 35.747 },
      elevationFt: null,
      roleHints: ["IAF"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:SCHOO",
      ident: "SCHOO",
      kind: "named_fix",
      position: { lon: -78.92647222, lat: 35.77341389 },
      elevationFt: null,
      roleHints: ["IF"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:WEPAS",
      ident: "WEPAS",
      kind: "final_approach_fix",
      position: { lon: -78.88295556, lat: 35.80876667 },
      elevationFt: null,
      roleHints: ["FAF"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      kind: "runway_threshold",
      position: { lon: -78.80196389, lat: 35.87445 },
      elevationFt: 798,
      roleHints: ["MAPt"],
      sourceRefs: ["src:cifp-detail"],
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
          sourceRefs: ["src:cifp-detail"],
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
            altitude: null,
            speedKt: null,
            geometryAltitudeFt: null,
          },
          roleAtEnd: "FAF",
          sourceRefs: ["src:cifp-detail"],
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
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 3, renderedInPlanView: true },
        },
      ],
    },
    {
      branchId: "branch:CHWDR",
      branchKey: "CHWDR",
      branchIdent: "CHWDR",
      transitionIdent: "CHWDR",
      branchRole: "transition",
      sequenceOrder: 2,
      mergeFixRef: "fix:WEPAS",
      continuesWithBranchId: "branch:R",
      defaultVisible: false,
      warnings: ["Transition branch is generated from CIFP transition records."],
      legs: [
        {
          legId: "leg:CHWDR:010",
          sequence: 10,
          segmentType: "initial",
          path: {
            pathTerminator: "IF",
            constructionMethod: "if_to_fix",
            startFixRef: null,
            endFixRef: "fix:CHWDR",
          },
          termination: { kind: "fix", fixRef: "fix:CHWDR" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 4000, rawText: "4000 ft" },
            speedKt: null,
            geometryAltitudeFt: 4000,
          },
          roleAtEnd: "IAF",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 10, renderedInPlanView: true },
        },
        {
          legId: "leg:CHWDR:020",
          sequence: 20,
          segmentType: "intermediate",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track_to_fix",
            startFixRef: "fix:CHWDR",
            endFixRef: "fix:WEPAS",
          },
          termination: { kind: "fix", fixRef: "fix:WEPAS" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 2200, rawText: "2200 ft" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "FAF",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 11, renderedInPlanView: true },
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
    knownSimplifications: [],
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

describe("buildProcedureRoutes", () => {
  it("preserves the current legacy route view model contract", () => {
    const routes = buildProcedureRoutes([sampleDocument]);

    expect(
      routes.map((route) => ({
        routeId: route.routeId,
        branchType: route.branchType,
        branchIdent: route.branchIdent,
        transitionIdent: route.transitionIdent,
        defaultVisible: route.defaultVisible,
        pointIdents: route.points.map((point) => point.fixIdent),
        pointRoles: route.points.map((point) => point.role),
        tunnel: route.tunnel,
      })),
    ).toMatchInlineSnapshot(`
      [
        {
          "branchIdent": "R",
          "branchType": "final",
          "defaultVisible": true,
          "pointIdents": [
            "SCHOO",
            "WEPAS",
            "RW05L",
          ],
          "pointRoles": [
            "IF",
            "FAF",
            "MAPt",
          ],
          "routeId": "KRDU-R05LY-R",
          "transitionIdent": null,
          "tunnel": {
            "lateralHalfWidthNm": 0.3,
            "mode": "visualApproximation",
            "sampleSpacingM": 250,
            "verticalHalfHeightFt": 300,
          },
        },
        {
          "branchIdent": "CHWDR",
          "branchType": "transition",
          "defaultVisible": false,
          "pointIdents": [
            "CHWDR",
            "WEPAS",
            "RW05L",
          ],
          "pointRoles": [
            "IAF",
            "FAF",
            "MAPt",
          ],
          "routeId": "KRDU-R05LY-CHWDR",
          "transitionIdent": "CHWDR",
          "tunnel": {
            "lateralHalfWidthNm": 0.3,
            "mode": "visualApproximation",
            "sampleSpacingM": 250,
            "verticalHalfHeightFt": 300,
          },
        },
      ]
    `);
  });

  it("keeps repaired altitude and timing behavior stable", () => {
    const [finalRoute] = buildProcedureRoutes([sampleDocument]);

    expect(finalRoute.points.map((point) => point.geometryAltitudeFt)).toEqual([
      3000,
      1899,
      798,
    ]);
    expect(finalRoute.points.map((point) => point.altM)).toEqual([914.4, 578.82, 243.23]);
    expect(finalRoute.points[0].distanceFromStartM).toBe(0);
    expect(finalRoute.points[1].distanceFromStartM).toBeGreaterThan(0);
    expect(finalRoute.points[2].distanceFromStartM).toBeGreaterThan(
      finalRoute.points[1].distanceFromStartM,
    );
    expect(finalRoute.points[2].timeSeconds).toBeGreaterThan(finalRoute.points[1].timeSeconds);
    expect(finalRoute.warnings).toContain(
      "WEPAS: altitude missing in procedure constraints; geometry altitude interpolated from neighboring constraints.",
    );
  });
});
