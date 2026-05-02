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
        branchId: "KRDU-R05LY-RW05L:branch:R",
        branchRole: "STRAIGHT_IN",
        segmentIds: [
          "KRDU-R05LY-RW05L:branch:R:segment:intermediate:1",
          "KRDU-R05LY-RW05L:branch:R:segment:final_rnav_gps:2",
        ],
        legacy: expect.objectContaining({
          sourceBranchId: "branch:R",
        }),
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
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:intermediate:1",
        segmentType: "INTERMEDIATE",
        legIds: ["leg:R:010"],
        xttNm: 1,
        attNm: 1,
        verticalRule: "NONE",
      },
      {
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:final_rnav_gps:2",
        segmentType: "FINAL_RNAV_GPS",
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
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:intermediate:1",
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
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:final_rnav_gps:2",
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
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:final_rnav_gps:2",
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

  it("passes vertical profile GPA and TCH into final vertical rules", () => {
    const pkg = normalizeProcedurePackage({
      ...sampleDocument,
      verticalProfiles: [
        {
          profileId: "profile:R:lnav-vnav",
          appliesToModes: ["LNAV/VNAV"],
          branchId: "branch:R",
          fromFixRef: "fix:WEPAS",
          toFixRef: "fix:RW05L",
          basis: "chart_profile",
          glidepathAngleDeg: 3,
          thresholdCrossingHeightFt: 50,
          constraintSamples: [],
          warnings: [],
        },
      ],
    });
    const finalSegment = pkg.segments.find((segment) => segment.segmentType.startsWith("FINAL"));

    expect(finalSegment?.verticalRule).toMatchObject({
      kind: "LPV_GLS_SURFACES",
      gpaDeg: 3,
      tchFt: 50,
    });
  });

  it("classifies route detail segments by branch role instead of collapsing them to unknown", () => {
    const transitionDocument: ProcedureDetailDocument = {
      ...sampleDocument,
      fixes: [
        ...sampleDocument.fixes,
        {
          fixId: "fix:CHWDR",
          ident: "CHWDR",
          kind: "named_fix",
          position: { lon: -79.1, lat: 35.7 },
          elevationFt: null,
          roleHints: ["IAF"],
          sourceRefs: ["src:cifp-detail:transition"],
        },
      ],
      branches: [
        {
          branchId: "branch:CHWDR",
          branchKey: "CHWDR",
          branchIdent: "CHWDR",
          transitionIdent: "CHWDR",
          branchRole: "transition",
          sequenceOrder: 0,
          mergeFixRef: "fix:SCHOO",
          continuesWithBranchId: "branch:R",
          defaultVisible: false,
          warnings: [],
          legs: [
            {
              ...sampleDocument.branches[0].legs[0],
              legId: "leg:CHWDR:010",
              sequence: 10,
              segmentType: "route",
              path: {
                pathTerminator: "TF",
                constructionMethod: "track_to_fix",
                startFixRef: "fix:CHWDR",
                endFixRef: "fix:SCHOO",
              },
              termination: { kind: "fix", fixRef: "fix:SCHOO" },
              sourceRefs: ["src:cifp-detail:transition"],
            },
          ],
        },
        sampleDocument.branches[0],
      ],
    };
    const transitionPackage = normalizeProcedurePackage(transitionDocument);
    const transitionSegment = transitionPackage.segments.find((segment) =>
      segment.segmentId.includes("branch:CHWDR"),
    );

    expect(transitionSegment).toMatchObject({
      segmentType: "TRANSITION_ROUTE",
      navSpec: "RNAV_1",
      xttNm: 1,
      attNm: 1,
      verticalRule: { kind: "NONE" },
    });
    expect(transitionSegment?.segmentId).toBe(
      "KRDU-R05LY-RW05L:branch:CHWDR:segment:transition_route:1",
    );

    const procedureRoutePackage = normalizeProcedurePackage({
      ...sampleDocument,
      branches: [
        {
          ...sampleDocument.branches[0],
          legs: [
            {
              ...sampleDocument.branches[0].legs[0],
              segmentType: "route",
            },
            ...sampleDocument.branches[0].legs.slice(1),
          ],
        },
      ],
    });
    const procedureRouteSegment = procedureRoutePackage.segments.find(
      (segment) => segment.segmentType === "PROCEDURE_ROUTE",
    );

    expect(procedureRouteSegment).toMatchObject({
      segmentId: "KRDU-R05LY-RW05L:branch:R:segment:procedure_route:1",
      segmentType: "PROCEDURE_ROUTE",
      navSpec: "RNP_APCH",
      xttNm: 1,
      attNm: 1,
      verticalRule: { kind: "NONE" },
    });
  });

  it("passes exported RF metadata through to the canonical package leg", () => {
    const rfDocument: ProcedureDetailDocument = {
      ...sampleDocument,
      branches: [
        {
          ...sampleDocument.branches[0],
          legs: [
            sampleDocument.branches[0].legs[0],
            {
              ...sampleDocument.branches[0].legs[1],
              legId: "leg:R:020",
              path: {
                ...sampleDocument.branches[0].legs[1].path,
                pathTerminator: "RF",
                constructionMethod: "radius_to_fix",
                turnDirection: "LEFT",
                arcRadiusNm: 2,
                centerLatDeg: 35.82,
                centerLonDeg: -78.86,
              },
            },
          ],
        },
      ],
    };

    const pkg = normalizeProcedurePackage(rfDocument);
    const rfLeg = pkg.legs.find((leg) => leg.legType === "RF");

    expect(rfLeg).toMatchObject({
      legType: "RF",
      turnDirection: "LEFT",
      arcRadiusNm: 2,
      centerLatDeg: 35.82,
      centerLonDeg: -78.86,
    });
    expect(pkg.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain(
      "RF_RADIUS_MISSING",
    );
  });

  it("passes exported course metadata through to CA package legs", () => {
    const caDocument: ProcedureDetailDocument = {
      ...sampleDocument,
      branches: [
        {
          ...sampleDocument.branches[0],
          legs: [
            {
              ...sampleDocument.branches[0].legs[2],
              legId: "leg:R:040",
              sequence: 40,
              segmentType: "missed",
              path: {
                ...sampleDocument.branches[0].legs[2].path,
                pathTerminator: "CA",
                constructionMethod: "course_to_altitude",
                startFixRef: "fix:RW05L",
                endFixRef: "fix:RW05L",
                courseDeg: 305,
              },
              roleAtEnd: "Route",
            },
          ],
        },
      ],
    };

    const pkg = normalizeProcedurePackage(caDocument);
    const caLeg = pkg.legs.find((leg) => leg.legType === "CA");

    expect(caLeg).toMatchObject({
      legType: "CA",
      outboundCourseDeg: 305,
    });
  });

  it("splits missed approach legs into section one and section two at the hold boundary", () => {
    const missedDocument: ProcedureDetailDocument = {
      ...sampleDocument,
      fixes: [
        ...sampleDocument.fixes,
        {
          fixId: "fix:MIS1",
          ident: "MIS1",
          kind: "named_fix",
          position: { lon: -78.76, lat: 35.9 },
          elevationFt: null,
          roleHints: ["UNKNOWN"],
          sourceRefs: ["src:cifp-detail:4"],
        },
        {
          fixId: "fix:HOLD",
          ident: "HOLD",
          kind: "missed_hold_fix",
          position: { lon: -78.7, lat: 35.95 },
          elevationFt: null,
          roleHints: ["MAHF"],
          sourceRefs: ["src:cifp-detail:5"],
        },
      ],
      branches: [
        {
          ...sampleDocument.branches[0],
          legs: [
            ...sampleDocument.branches[0].legs,
            {
              legId: "leg:R:040",
              sequence: 40,
              segmentType: "missed",
              path: {
                pathTerminator: "DF",
                constructionMethod: "direct_to_fix",
                startFixRef: "fix:RW05L",
                endFixRef: "fix:MIS1",
              },
              termination: { kind: "fix", fixRef: "fix:MIS1" },
              constraints: {
                altitude: { qualifier: "at", valueFt: 1500, rawText: "1500 ft" },
                speedKt: null,
                geometryAltitudeFt: 1500,
              },
              roleAtEnd: "UNKNOWN",
              sourceRefs: ["src:cifp-detail:4"],
              quality: { status: "exact", sourceLine: 4, renderedInPlanView: true },
            },
            {
              legId: "leg:R:050",
              sequence: 50,
              segmentType: "missed",
              path: {
                pathTerminator: "HM",
                constructionMethod: "hold_to_manual",
                startFixRef: "fix:MIS1",
                endFixRef: "fix:HOLD",
              },
              termination: { kind: "fix", fixRef: "fix:HOLD" },
              constraints: {
                altitude: { qualifier: "at", valueFt: 3000, rawText: "3000 ft" },
                speedKt: null,
                geometryAltitudeFt: 3000,
              },
              roleAtEnd: "MAHF",
              sourceRefs: ["src:cifp-detail:5"],
              quality: { status: "exact", sourceLine: 5, renderedInPlanView: false },
            },
          ],
        },
      ],
    };

    const pkg = normalizeProcedurePackage(missedDocument);
    const missedSegments = pkg.segments.filter((segment) =>
      segment.segmentType.startsWith("MISSED"),
    );

    expect(missedSegments.map((segment) => ({
      segmentId: segment.segmentId,
      segmentType: segment.segmentType,
      legIds: segment.legIds,
      isTurningMissedApproach: segment.constructionFlags.isTurningMissedApproach,
    }))).toEqual([
      {
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:missed_s1:3",
        segmentType: "MISSED_S1",
        legIds: ["leg:R:040"],
        isTurningMissedApproach: undefined,
      },
      {
        segmentId: "KRDU-R05LY-RW05L:branch:R:segment:missed_s2:4",
        segmentType: "MISSED_S2",
        legIds: ["leg:R:050"],
        isTurningMissedApproach: true,
      },
    ]);
    expect(pkg.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "TURNING_MISSED_UNIMPLEMENTED",
          segmentId: "KRDU-R05LY-RW05L:branch:R:segment:missed_s2:4",
          severity: "WARN",
        }),
      ]),
    );
  });

  it("scopes straight-in branch ids by procedure and runway", () => {
    const rw05Package = normalizeProcedurePackage(sampleDocument);
    const rw32Package = normalizeProcedurePackage({
      ...sampleDocument,
      procedureUid: "KRDU-R32-RW32",
      runway: {
        ...sampleDocument.runway,
        ident: "RW32",
        landingThresholdFixRef: "fix:RW32",
      },
      procedure: {
        ...sampleDocument.procedure,
        procedureIdent: "R32",
        chartName: "RNAV(GPS) RWY 32",
        runwayIdent: "RW32",
      },
    });

    expect(rw05Package.branches[0].legacy.sourceBranchId).toBe("branch:R");
    expect(rw32Package.branches[0].legacy.sourceBranchId).toBe("branch:R");
    expect(rw05Package.branches[0].branchId).toBe("KRDU-R05LY-RW05L:branch:R");
    expect(rw32Package.branches[0].branchId).toBe("KRDU-R32-RW32:branch:R");
    expect(rw05Package.branches[0].branchId).not.toBe(rw32Package.branches[0].branchId);
  });
});
