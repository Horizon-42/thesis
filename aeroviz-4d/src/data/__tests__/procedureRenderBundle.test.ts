import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchJson } from "../../utils/fetchJson";
import type { ProcedureDetailDocument, ProcedureDetailsIndexManifest } from "../procedureDetails";
import type { ProcedurePackage } from "../procedurePackage";
import { buildProcedureRenderBundle, loadProcedureRenderBundleData } from "../procedureRenderBundle";

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

describe("procedure render bundle", () => {
  beforeEach(() => {
    vi.mocked(fetchJson).mockReset();
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
    expect(segmentBundle.alignedConnector?.connectorType).toBe(
      "ALIGNED_LNAV_INTERMEDIATE_TO_FINAL",
    );
    expect(bundle.diagnostics).toEqual([]);
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
});
