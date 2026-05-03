import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchJson } from "../../utils/fetchJson";
import { METERS_PER_NM, offsetPoint } from "../../utils/procedureGeoMath";
import type { ProcedureDetailDocument, ProcedureDetailsIndexManifest } from "../procedureDetails";
import type { ProcedurePackage } from "../procedurePackage";
import { normalizeProcedurePackage } from "../procedurePackageAdapter";
import {
  buildProcedureRenderBundle,
  clearProcedureRenderBundleDataCache,
  loadProcedureRenderBundleData,
} from "../procedureRenderBundle";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn(),
}));

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
        sourceBranchId: "branch:R",
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

const sampleIndex: ProcedureDetailsIndexManifest = {
  airport: "KRDU",
  airportName: "Raleigh-Durham International Airport",
  sourceCycle: "2501",
  researchUseOnly: true,
  runways: [
    {
      runwayIdent: "RW05L",
      chartName: "RW05L",
      procedureUids: ["KRDU-R05LY-RW05L"],
      procedures: [
        {
          procedureUid: "KRDU-R05LY-RW05L",
          procedureIdent: "R05LY",
          chartName: "RNAV(GPS) Y RW05L",
          procedureFamily: "RNAV_GPS",
          variant: "Y",
          approachModes: ["LNAV"],
          runwayIdent: "RW05L",
          defaultBranchId: "branch:R",
        },
      ],
    },
  ],
};

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
  airport: { icao: "KRDU", faa: "RDU", name: "Raleigh-Durham International Airport" },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: null,
    threshold: null,
  },
  procedure: {
    procedureType: "SIAP",
    procedureFamily: "RNAV_GPS",
    procedureIdent: "R05LY",
    chartName: "RNAV(GPS) Y RW05L",
    variant: "Y",
    runwayIdent: "RW05L",
    baseBranchIdent: "R",
    approachModes: ["LNAV"],
  },
  fixes: [],
  branches: [],
  verticalProfiles: [],
  validation: {
    expectedRunwayIdent: "RW05L",
    expectedIF: null,
    expectedFAF: null,
    expectedMAPt: null,
    expectedMissedHoldFix: null,
    knownSimplifications: [],
  },
  displayHints: {
    nominalSpeedKt: 140,
    defaultVisibleBranchIds: [],
    tunnelDefaults: {
      lateralHalfWidthNm: 0.3,
      verticalHalfHeightFt: 300,
      sampleSpacingM: 250,
      mode: "visualApproximation",
    },
  },
};

const twoSegmentTurnPackage: ProcedurePackage = {
  ...samplePackage,
  branches: [
    {
      ...samplePackage.branches[0],
      segmentIds: ["segment:intermediate", "segment:final"],
    },
  ],
  segments: [
    {
      segmentId: "segment:intermediate",
      branchId: "branch:R",
      segmentType: "INTERMEDIATE",
      navSpec: "RNP_APCH",
      startFixId: "fix:IF",
      endFixId: "fix:FAF",
      legIds: ["leg:R:020"],
      xttNm: 0.3,
      attNm: 0.3,
      secondaryEnabled: true,
      widthChangeMode: "LINEAR_TAPER",
      transitionRule: null,
      verticalRule: { kind: "LEVEL_ROC" },
      constructionFlags: {},
      sourceRefs: [],
      legacy: {
        rawSegmentType: "intermediate",
        sequenceRange: [20, 20],
      },
    },
    samplePackage.segments[0],
  ],
  legs: [
    {
      legId: "leg:R:020",
      segmentId: "segment:intermediate",
      legType: "TF",
      rawPathTerminator: "TF",
      startFixId: "fix:IF",
      endFixId: "fix:FAF",
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
        roleAtEnd: "IF",
        qualityStatus: "exact",
        renderedInPlanView: true,
      },
    },
    samplePackage.legs[0],
  ],
  sharedFixes: [
    {
      fixId: "fix:IF",
      ident: "IF",
      role: ["IF"],
      lonDeg: -78.88,
      latDeg: 35.84,
      altFtMsl: 2400,
      annotations: [],
      sourceRefs: [],
    },
    ...samplePackage.sharedFixes,
  ],
};

const rfCenter = { lonDeg: -78.86, latDeg: 35.84, altM: 0 };
const rfStart = offsetPoint(rfCenter, Math.PI / 2, 2 * METERS_PER_NM);
const rfEnd = offsetPoint(rfCenter, 0, 2 * METERS_PER_NM);

const rfDocument: ProcedureDetailDocument = {
  ...sampleDocument,
  procedureUid: "KRDU-RFTEST-RW05L",
  procedure: {
    ...sampleDocument.procedure,
    procedureIdent: "RFTEST",
    chartName: "RNAV RF TEST",
  },
  fixes: [
    {
      fixId: "fix:RFSTART",
      ident: "RFSTART",
      kind: "named_fix",
      position: { lon: rfStart.lonDeg, lat: rfStart.latDeg },
      elevationFt: 3000,
      roleHints: ["IF"],
      sourceRefs: [],
    },
    {
      fixId: "fix:RFEND",
      ident: "RFEND",
      kind: "named_fix",
      position: { lon: rfEnd.lonDeg, lat: rfEnd.latDeg },
      elevationFt: 2200,
      roleHints: ["FAF"],
      sourceRefs: [],
    },
    {
      fixId: "fix:CENTER",
      ident: "CENTER",
      kind: "rf_center",
      position: { lon: rfCenter.lonDeg, lat: rfCenter.latDeg },
      elevationFt: null,
      roleHints: [],
      sourceRefs: [],
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
            endFixRef: "fix:RFSTART",
          },
          termination: { kind: "fix", fixRef: "fix:RFSTART" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 3000, rawText: "3000 ft" },
            speedKt: null,
            geometryAltitudeFt: 3000,
          },
          roleAtEnd: "IF",
          sourceRefs: [],
          quality: { status: "exact", sourceLine: 1, renderedInPlanView: true },
        },
        {
          legId: "leg:R:020",
          sequence: 20,
          segmentType: "intermediate",
          path: {
            pathTerminator: "RF",
            constructionMethod: "radius_to_fix",
            startFixRef: "fix:RFSTART",
            endFixRef: "fix:RFEND",
            turnDirection: "LEFT",
            arcRadiusNm: 2,
            centerFixRef: "fix:CENTER",
            centerLatDeg: rfCenter.latDeg,
            centerLonDeg: rfCenter.lonDeg,
          },
          termination: { kind: "fix", fixRef: "fix:RFEND" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 2200, rawText: "2200 ft" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "FAF",
          sourceRefs: [],
          quality: { status: "exact", sourceLine: 2, renderedInPlanView: true },
        },
      ],
    },
  ],
};

const missedDfDocument: ProcedureDetailDocument = {
  ...sampleDocument,
  procedureUid: "KRDU-MISSEDDF-RW05L",
  procedure: {
    ...sampleDocument.procedure,
    procedureIdent: "MISSEDDF",
    chartName: "RNAV MISSED DF TEST",
  },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: "fix:RW05L",
    threshold: { lon: -78.8, lat: 35.87, elevationFt: 800 },
  },
  fixes: [
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      kind: "runway_threshold",
      position: { lon: -78.8, lat: 35.87 },
      elevationFt: 800,
      roleHints: ["MAPt"],
      sourceRefs: [],
    },
    {
      fixId: "fix:MIS1",
      ident: "MIS1",
      kind: "named_fix",
      position: { lon: -78.74, lat: 35.91 },
      elevationFt: 1800,
      roleHints: ["UNKNOWN"],
      sourceRefs: [],
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
            altitude: { qualifier: "at", valueFt: 1800, rawText: "1800 ft" },
            speedKt: null,
            geometryAltitudeFt: 1800,
          },
          roleAtEnd: "UNKNOWN",
          sourceRefs: [],
          quality: { status: "exact", sourceLine: 40, renderedInPlanView: true },
        },
      ],
    },
  ],
};

const missedCaDocument: ProcedureDetailDocument = {
  ...missedDfDocument,
  procedureUid: "KRDU-MISSEDCA-RW05L",
  procedure: {
    ...missedDfDocument.procedure,
    procedureIdent: "MISSEDCA",
    chartName: "RNAV MISSED CA TEST",
  },
  branches: [
    {
      ...missedDfDocument.branches[0],
      legs: [
        {
          ...missedDfDocument.branches[0].legs[0],
          legId: "leg:R:035",
          sequence: 35,
          path: {
            pathTerminator: "CA",
            constructionMethod: "course_to_altitude",
            startFixRef: "fix:RW05L",
            endFixRef: "fix:RW05L",
            courseDeg: 305,
          },
          termination: { kind: "fix", fixRef: "fix:RW05L" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 1000, rawText: "1000 ft" },
            speedKt: null,
            geometryAltitudeFt: 1000,
          },
        },
      ],
    },
  ],
};

const missedCaDfDocument: ProcedureDetailDocument = {
  ...missedDfDocument,
  procedureUid: "KRDU-MISSEDCADF-RW05L",
  procedure: {
    ...missedDfDocument.procedure,
    procedureIdent: "MISSEDCADF",
    chartName: "RNAV MISSED CA DF TEST",
  },
  fixes: [
    ...missedDfDocument.fixes,
    {
      fixId: "fix:HOLD",
      ident: "HOLD",
      kind: "missed_hold_fix",
      position: { lon: -78.7, lat: 35.95 },
      elevationFt: null,
      roleHints: ["MAHF"],
      sourceRefs: [],
    },
  ],
  branches: [
    {
      ...missedDfDocument.branches[0],
      legs: [
        {
          ...missedDfDocument.branches[0].legs[0],
          legId: "leg:R:035",
          sequence: 35,
          path: {
            pathTerminator: "CA",
            constructionMethod: "course_to_altitude",
            startFixRef: "fix:RW05L",
            endFixRef: "fix:",
            courseDeg: 305,
          },
          termination: { kind: "fix", fixRef: "fix:" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 1000, rawText: "1000 ft" },
            speedKt: null,
            geometryAltitudeFt: 1000,
          },
        },
        {
          ...missedDfDocument.branches[0].legs[0],
          legId: "leg:R:040",
          sequence: 40,
          path: {
            pathTerminator: "DF",
            constructionMethod: "direct_to_fix",
            startFixRef: "fix:",
            endFixRef: "fix:MIS1",
          },
          termination: { kind: "fix", fixRef: "fix:MIS1" },
        },
        {
          ...missedDfDocument.branches[0].legs[0],
          legId: "leg:R:050",
          sequence: 50,
          path: {
            pathTerminator: "HM",
            constructionMethod: "hold_to_manual",
            startFixRef: "fix:HOLD",
            endFixRef: "fix:HOLD",
            courseDeg: 305,
            turnDirection: "RIGHT",
          },
          termination: { kind: "fix", fixRef: "fix:HOLD" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 2200, rawText: "2200 ft" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "MAHF",
        },
      ],
    },
  ],
};

const missedTurningDocument: ProcedureDetailDocument = {
  ...missedDfDocument,
  procedureUid: "KRDU-MISSEDTURN-RW05L",
  procedure: {
    ...missedDfDocument.procedure,
    procedureIdent: "MISSEDTURN",
    chartName: "RNAV MISSED TURN TEST",
  },
  fixes: [
    ...missedDfDocument.fixes,
    {
      fixId: "fix:HOLD",
      ident: "HOLD",
      kind: "missed_hold_fix",
      position: { lon: -78.7, lat: 35.95 },
      elevationFt: 3000,
      roleHints: ["MAHF"],
      sourceRefs: [],
    },
  ],
  branches: [
    {
      ...missedDfDocument.branches[0],
      legs: [
        missedDfDocument.branches[0].legs[0],
        {
          ...missedDfDocument.branches[0].legs[0],
          legId: "leg:R:050",
          sequence: 50,
          path: {
            pathTerminator: "HM",
            constructionMethod: "hold_to_manual",
            startFixRef: "fix:MIS1",
            endFixRef: "fix:HOLD",
            courseDeg: 305,
            turnDirection: "RIGHT",
          },
          termination: { kind: "fix", fixRef: "fix:HOLD" },
          roleAtEnd: "MAHF",
        },
      ],
    },
  ],
};

describe("procedure render bundle", () => {
  beforeEach(() => {
    vi.mocked(fetchJson).mockReset();
    clearProcedureRenderBundleDataCache();
  });

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
    expect(segmentBundle.finalSurfaceStatus).toMatchObject({
      constructedSurfaceTypes: ["LNAV_FINAL_OEA"],
      missingSurfaceTypes: [],
      constructionStatus: "LNAV_BASELINE_CONSTRUCTED",
    });
    expect(segmentBundle.alignedConnector?.connectorType).toBe(
      "ALIGNED_LNAV_INTERMEDIATE_TO_FINAL",
    );
    expect(bundle.branchBundles[0].protectionSurfaces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          surfaceId: "segment:final:lnav-oea",
          kind: "FINAL_LNAV_OEA",
          status: "SOURCE_BACKED",
          vertical: expect.objectContaining({ kind: "NONE" }),
          lateral: expect.objectContaining({
            widthSamples: expect.arrayContaining([
              expect.objectContaining({
                primaryHalfWidthNm: expect.any(Number),
                secondaryOuterHalfWidthNm: expect.any(Number),
              }),
            ]),
          }),
        }),
      ]),
    );
    expect(bundle.branchBundles[0].turnJunctions).toEqual([]);
    expect(bundle.diagnostics).toEqual([]);
  });

  it("exposes missing final vertical surfaces when approach modes collapse to LNAV", () => {
    const collapsedPackage: ProcedurePackage = {
      ...samplePackage,
      segments: [
        {
          ...samplePackage.segments[0],
          verticalRule: { kind: "LPV_GLS_SURFACES" },
          constructionFlags: {
            collapsedApproachModes: ["LPV", "LNAV/VNAV", "LNAV"],
          },
        },
      ],
    };
    const bundle = buildProcedureRenderBundle(collapsedPackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.finalSurfaceStatus).toMatchObject({
      requestedModes: ["LPV", "LNAV/VNAV", "LNAV"],
      constructedSurfaceTypes: ["LNAV_FINAL_OEA"],
      missingSurfaceTypes: ["LPV_W", "LPV_X", "LPV_Y", "LNAV_VNAV_OCS"],
      constructionStatus: "COLLAPSED_TO_LNAV_BASELINE",
    });
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED",
          segmentId: "segment:final",
          severity: "WARN",
        }),
      ]),
    );
  });

  it("exports LNAV/VNAV OCS geometry when GPA and TCH are available", () => {
    const lnavVnavPackage: ProcedurePackage = {
      ...samplePackage,
      segments: [
        {
          ...samplePackage.segments[0],
          segmentType: "FINAL_LNAV_VNAV",
          verticalRule: { kind: "BARO_GLIDEPATH", gpaDeg: 3, tchFt: 50 },
          constructionFlags: {},
        },
      ],
    };
    const bundle = buildProcedureRenderBundle(lnavVnavPackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.lnavVnavOcs).toMatchObject({
      surfaceType: "LNAV_VNAV_OCS",
      constructionStatus: "GPA_TCH_SLOPE_ESTIMATE",
      verticalProfile: {
        gpaDeg: 3,
        tchFt: 50,
      },
    });
    expect(segmentBundle.finalSurfaceStatus).toMatchObject({
      constructedSurfaceTypes: ["LNAV_FINAL_OEA", "LNAV_VNAV_OCS"],
      missingSurfaceTypes: [],
      constructionStatus: "MODE_SPECIFIC_SURFACES_CONSTRUCTED",
    });
    expect(bundle.branchBundles[0].protectionSurfaces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          surfaceId: "segment:final:lnav-vnav-ocs",
          kind: "FINAL_LNAV_VNAV_OCS",
          status: "TERPS_ESTIMATE",
          vertical: expect.objectContaining({
            kind: "OCS",
            origin: "GPA_TCH",
            samples: expect.arrayContaining([
              expect.objectContaining({ altitudeFtMsl: expect.any(Number) }),
            ]),
          }),
        }),
      ]),
    );
    expect(bundle.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain(
      "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED",
    );
  });

  it("exports LPV W/X/Y debug-estimate surfaces with independent ids", () => {
    const lpvPackage: ProcedurePackage = {
      ...samplePackage,
      segments: [
        {
          ...samplePackage.segments[0],
          segmentType: "FINAL_LNAV",
          verticalRule: { kind: "LPV_GLS_SURFACES", gpaDeg: 3, tchFt: 50 },
          constructionFlags: { collapsedApproachModes: ["LPV", "LNAV"] },
        },
      ],
    };
    const bundle = buildProcedureRenderBundle(lpvPackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.precisionFinalSurfaces.map((surface) => surface.surfaceType)).toEqual([
      "LPV_W",
      "LPV_X",
      "LPV_Y",
    ]);
    expect(segmentBundle.precisionFinalSurfaces.map((surface) => surface.geometryId)).toEqual([
      "segment:final:lpv-w",
      "segment:final:lpv-x",
      "segment:final:lpv-y",
    ]);
    expect(bundle.branchBundles[0].protectionSurfaces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          surfaceId: "segment:final:lpv-w",
          kind: "FINAL_PRECISION_DEBUG",
          status: "DEBUG_ESTIMATE",
        }),
      ]),
    );
    expect(segmentBundle.finalSurfaceStatus).toMatchObject({
      constructedSurfaceTypes: ["LNAV_FINAL_OEA", "LPV_W", "LPV_X", "LPV_Y"],
      missingSurfaceTypes: [],
      constructionStatus: "MODE_SPECIFIC_SURFACES_CONSTRUCTED",
    });
  });

  it("reports RNP AR final templates as distinct missing geometry", () => {
    const rnpArPackage: ProcedurePackage = {
      ...samplePackage,
      procedureFamily: "RNP_AR_APCH",
      segments: [
        {
          ...samplePackage.segments[0],
          segmentType: "FINAL_RNP_AR",
          navSpec: "RNP_AR_0_3",
          verticalRule: { kind: "RNP_AR_VERTICAL" },
          secondaryEnabled: false,
          constructionFlags: {},
        },
      ],
    };
    const bundle = buildProcedureRenderBundle(rnpArPackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.finalSurfaceStatus).toMatchObject({
      requestedModes: ["RNP/AR"],
      constructedSurfaceTypes: [],
      missingSurfaceTypes: ["RNP_AR_FINAL_TEMPLATE"],
      constructionStatus: "UNSUPPORTED_FINAL_VERTICAL_SURFACES",
    });
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED",
          segmentId: "segment:final",
        }),
      ]),
    );
  });

  it("reports generated RNAV(RNP) RNP AR documents as missing RNP AR final templates", () => {
    const pkg = normalizeProcedurePackage({
      ...rfDocument,
      procedureUid: "KRDU-H05LZ-RW05L",
      procedure: {
        ...rfDocument.procedure,
        procedureFamily: "RNAV_RNP",
        procedureIdent: "H05LZ",
        chartName: "RNAV(RNP) Z RWY 05L",
        variant: "Z",
        approachModes: ["RNP AR"],
      },
      branches: [
        {
          ...rfDocument.branches[0],
          legs: [
            rfDocument.branches[0].legs[0],
            {
              ...rfDocument.branches[0].legs[1],
              segmentType: "final",
            },
          ],
        },
      ],
    });
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const finalSegment = bundle.branchBundles[0].segmentBundles.find((segmentBundle) =>
      segmentBundle.segment.segmentType.startsWith("FINAL"),
    );

    expect(finalSegment?.segment.segmentType).toBe("FINAL_RNP_AR");
    expect(finalSegment?.finalSurfaceStatus).toMatchObject({
      requestedModes: ["RNP/AR"],
      constructedSurfaceTypes: [],
      missingSurfaceTypes: ["RNP_AR_FINAL_TEMPLATE"],
      constructionStatus: "UNSUPPORTED_FINAL_VERTICAL_SURFACES",
    });
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED",
          segmentId: "KRDU-H05LZ-RW05L:branch:R:segment:final_rnp_ar:2",
        }),
      ]),
    );
  });

  it("adds visual inter-segment turn junctions at adjacent segment joins", () => {
    const bundle = buildProcedureRenderBundle(twoSegmentTurnPackage, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });

    expect(bundle.branchBundles[0].turnJunctions).toHaveLength(1);
    expect(bundle.branchBundles[0].turnJunctions[0]).toMatchObject({
      branchId: "branch:R",
      fromSegmentId: "segment:intermediate",
      toSegmentId: "segment:final",
      constructionStatus: "VISUAL_FILL_ONLY",
    });
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "TURN_VISUAL_FILL_ONLY",
          severity: "WARN",
          segmentId: "segment:intermediate",
        }),
      ]),
    );
  });

  it("builds RF arc geometry from procedure-detail RF path metadata", () => {
    const pkg = normalizeProcedurePackage(rfDocument);
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(pkg.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain(
      "RF_RADIUS_MISSING",
    );
    expect(segmentBundle.segmentGeometry.centerline.isArc).toBe(true);
    expect(segmentBundle.segmentGeometry.centerline.geodesicLengthNm).toBeCloseTo(Math.PI, 2);
    expect(segmentBundle.segmentGeometry.turnJunctions).toEqual([]);
  });

  it("builds DF missed approach geometry from procedure-detail path metadata", () => {
    const pkg = normalizeProcedurePackage(missedDfDocument);
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.segment.segmentType).toBe("MISSED_S1");
    expect(segmentBundle.segmentGeometry.centerline.geoPositions.length).toBeGreaterThan(2);
    expect(segmentBundle.segmentGeometry.primaryEnvelope).toBeDefined();
    expect(segmentBundle.segmentGeometry.secondaryEnvelope).toBeDefined();
    expect(segmentBundle.missedSectionSurface).toMatchObject({
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      constructionStatus: "SOURCE_BACKED",
      primary: expect.any(Object),
      secondaryOuter: expect.any(Object),
    });
    expect(bundle.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain(
      "UNSUPPORTED_LEG_TYPE",
    );
  });

  it("exports CA missed course guides through render bundles", () => {
    const pkg = normalizeProcedurePackage(missedCaDocument);
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.segment.segmentType).toBe("MISSED_S1");
    expect(segmentBundle.segmentGeometry.centerline.geoPositions.length).toBeGreaterThan(2);
    expect(segmentBundle.segmentGeometry.primaryEnvelope).toBeDefined();
    expect(segmentBundle.segmentGeometry.secondaryEnvelope).toBeDefined();
    expect(segmentBundle.missedSectionSurface).toMatchObject({
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      constructionStatus: "ESTIMATED_CA",
      primary: expect.any(Object),
      secondaryOuter: expect.any(Object),
    });
    expect(segmentBundle.missedCourseGuides).toHaveLength(1);
    expect(segmentBundle.missedCourseGuides[0]).toMatchObject({
      legId: "leg:R:035",
      courseDeg: 305,
      requiredAltitudeFtMsl: 1000,
      constructionStatus: "COURSE_DIRECTION_ONLY",
    });
    expect(segmentBundle.missedCaEndpoints).toHaveLength(1);
    expect(segmentBundle.missedCaEndpoints[0]).toMatchObject({
      legId: "leg:R:035",
      courseDeg: 305,
      startAltitudeFtMsl: 800,
      targetAltitudeFtMsl: 1000,
      climbGradientFtPerNm: 200,
      distanceNm: 1,
      constructionStatus: "ESTIMATED_ENDPOINT",
    });
    expect(segmentBundle.missedCaCenterlines).toHaveLength(1);
    expect(segmentBundle.missedCaCenterlines[0]).toMatchObject({
      legId: "leg:R:035",
      sourceEndpointStatus: "ESTIMATED_ENDPOINT",
      constructionStatus: "ESTIMATED_CENTERLINE",
      isArc: false,
    });
    expect(segmentBundle.missedCaCenterlines[0].geoPositions.length).toBeGreaterThan(2);
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "ESTIMATED_CA_GEOMETRY",
          legId: "leg:R:035",
        }),
      ]),
    );
    expect(bundle.diagnostics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "UNSUPPORTED_LEG_TYPE",
          legId: "leg:R:035",
        }),
      ]),
    );
  });

  it("exports a CA-only estimated missed surface when a following DF leg is not constructible", () => {
    const pkg = normalizeProcedurePackage(missedCaDfDocument);
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const segmentBundle = bundle.branchBundles[0].segmentBundles[0];

    expect(segmentBundle.segment.segmentType).toBe("MISSED_S1");
    expect(segmentBundle.legs.map((leg) => leg.legType)).toEqual(["CA", "DF"]);
    expect(segmentBundle.segmentGeometry.centerline.geoPositions.length).toBeGreaterThan(2);
    expect(segmentBundle.missedSectionSurface).toMatchObject({
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      constructionStatus: "ESTIMATED_CA",
      primary: expect.objectContaining({
        geometryId: expect.stringContaining(":ca-estimated-primary"),
      }),
      secondaryOuter: expect.objectContaining({
        geometryId: expect.stringContaining(":ca-estimated-secondary"),
      }),
    });
    expect(segmentBundle.missedCaEndpoints).toHaveLength(1);
    expect(bundle.branchBundles[0].missedCaMahfConnectors).toHaveLength(1);
    expect(bundle.branchBundles[0].missedConnectorSurfaces).toHaveLength(1);
    expect(bundle.branchBundles[0].missedConnectorSurfaces[0]).toMatchObject({
      sourceLegId: "leg:R:035",
      targetFixIdent: "HOLD",
      constructionStatus: "ESTIMATED_CONNECTOR_SURFACE",
      lateralWidthRule: {
        ruleId: "MISSED_CONNECTOR_TERMINAL_WIDTH",
        ruleStatus: "SOURCE_SURFACE_TERMINAL_WIDTH",
        transitionStatus: "TERMINAL_WIDTH_HELD_TO_MAHF",
      },
      primary: expect.any(Object),
      secondaryOuter: expect.any(Object),
    });
    expect(bundle.branchBundles[0].protectionSurfaces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "MISSED_SECTION_1",
          status: "DISPLAY_ESTIMATE",
          sourceLegIds: ["leg:R:035", "leg:R:040"],
          vertical: expect.objectContaining({
            kind: "OCS",
            origin: "CENTERLINE_ALTITUDE_ONLY",
          }),
        }),
        expect.objectContaining({
          kind: "MISSED_CONNECTOR",
          status: "DISPLAY_ESTIMATE",
          sourceLegIds: ["leg:R:035"],
          lateral: expect.objectContaining({
            rule: expect.stringContaining("CA endpoint to MAHF"),
            notes: expect.arrayContaining([expect.stringContaining("Target fix HOLD")]),
          }),
        }),
      ]),
    );
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "ESTIMATED_CA_GEOMETRY",
          legId: "leg:R:035",
        }),
        expect.objectContaining({
          code: "SOURCE_INCOMPLETE",
          legId: "leg:R:040",
        }),
      ]),
    );
    expect(bundle.diagnostics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "UNSUPPORTED_LEG_TYPE",
          legId: "leg:R:035",
        }),
      ]),
    );
  });

  it("exports turning missed debug anchors through render bundles", () => {
    const pkg = normalizeProcedurePackage(missedTurningDocument);
    const bundle = buildProcedureRenderBundle(pkg, {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });
    const missedS2Bundle = bundle.branchBundles[0].segmentBundles.find(
      (segmentBundle) => segmentBundle.segment.segmentType === "MISSED_S2",
    );

    expect(missedS2Bundle?.segment.constructionFlags.isTurningMissedApproach).toBe(true);
    expect(missedS2Bundle?.missedTurnDebugPoint).toMatchObject({
      debugType: "TURNING_MISSED_ANCHOR",
      anchorFixId: "fix:MIS1",
      triggerLegTypes: ["HM"],
      constructionStatus: "DEBUG_MARKER_ONLY",
    });
    expect(missedS2Bundle?.missedTurnDebugPrimitives.map((primitive) => primitive.debugType)).toEqual([
      "TIA_BOUNDARY",
      "EARLY_TURN_BASELINE",
      "LATE_TURN_BASELINE",
      "NOMINAL_TURN_PATH",
      "WIND_SPIRAL",
    ]);
    expect(bundle.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "SOURCE_INCOMPLETE",
          legId: "leg:R:050",
        }),
      ]),
    );
  });

  it("loads procedure details through the package normalizer and render bundle builder", async () => {
    vi.mocked(fetchJson)
      .mockResolvedValueOnce(sampleIndex)
      .mockResolvedValueOnce(sampleDocument);

    const data = await loadProcedureRenderBundleData("krdu", {
      samplingStepNm: 0.5,
      enableDebugPrimitives: false,
    });

    expect(fetchJson).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");
    expect(fetchJson).toHaveBeenCalledWith(
      "/data/airports/KRDU/procedure-details/KRDU-R05LY-RW05L.json",
    );
    expect(data.documents).toEqual([sampleDocument]);
    expect(data.packages[0].packageId).toBe("KRDU-R05LY-RW05L");
    expect(data.renderBundles[0].procedureId).toBe("R05LY");
  });

  it("reuses cached render bundle loads for the same airport and geometry context", async () => {
    vi.mocked(fetchJson)
      .mockResolvedValueOnce(sampleIndex)
      .mockResolvedValueOnce(sampleDocument);

    const [first, second] = await Promise.all([
      loadProcedureRenderBundleData("krdu", {
        samplingStepNm: 0.5,
        enableDebugPrimitives: false,
      }),
      loadProcedureRenderBundleData("KRDU", {
        samplingStepNm: 0.5,
        enableDebugPrimitives: false,
      }),
    ]);

    expect(first).toBe(second);
    expect(fetchJson).toHaveBeenCalledTimes(2);
    expect(fetchJson).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");
    expect(fetchJson).toHaveBeenCalledWith(
      "/data/airports/KRDU/procedure-details/KRDU-R05LY-RW05L.json",
    );
  });
});
