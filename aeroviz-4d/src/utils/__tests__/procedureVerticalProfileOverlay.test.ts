import { describe, expect, it } from "vitest";
import type { ProcedureDetailDocument } from "../../data/procedureDetails";
import type { ProcedureBranchPolyline } from "../procedureDetailsGeometry";
import { buildFinalVerticalProfileOverlays } from "../procedureVerticalProfileOverlay";

const document: ProcedureDetailDocument = {
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
      elevationFt: 367,
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
  fixes: [],
  branches: [],
  verticalProfiles: [
    {
      profileId: "profile:R05LY:RW05L",
      appliesToModes: ["LPV", "LNAV/VNAV", "LNAV"],
      branchId: "branch:R",
      fromFixRef: "fix:SCHOO",
      toFixRef: "fix:RW05L",
      basis: "cifp_leg_constraints",
      glidepathAngleDeg: 3,
      thresholdCrossingHeightFt: null,
      constraintSamples: [
        {
          fixRef: "fix:SCHOO",
          ident: "SCHOO",
          role: "IF",
          distanceFromStartM: 0,
          altitudeFt: 3000,
          geometryAltitudeFt: 3000,
          sourceLine: 10,
        },
        {
          fixRef: "fix:WEPAS",
          ident: "WEPAS",
          role: "FAF",
          distanceFromStartM: 5555.1,
          altitudeFt: 2200,
          geometryAltitudeFt: 2200,
          sourceLine: 20,
        },
        {
          fixRef: "fix:RW05L",
          ident: "RW05L",
          role: "MAPt",
          distanceFromStartM: 15881.8,
          altitudeFt: 424,
          geometryAltitudeFt: 367,
          sourceLine: 30,
        },
      ],
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
    tunnelDefaults: {
      lateralHalfWidthNm: 0.3,
      verticalHalfHeightFt: 300,
      sampleSpacingM: 250,
      mode: "legacy",
    },
  },
};

const finalPolyline: ProcedureBranchPolyline = {
  branchId: "branch:R",
  branchIdent: "R",
  branchRole: "final",
  routeId: "KRDU-R05LY-R",
  branchKey: "R",
  transitionIdent: null,
  procedureIdent: "R05LY",
  procedureName: "RNAV(GPS) Y RWY 05L",
  procedureFamily: "RNAV_GPS",
  defaultVisible: true,
  warnings: [],
  points: [
    {
      fixId: "fix:SCHOO",
      ident: "SCHOO",
      role: "IF",
      branchId: "branch:R",
      branchIdent: "R",
      branchRole: "final",
      routeId: "KRDU-R05LY-R",
      branchKey: "R",
      transitionIdent: null,
      procedureIdent: "R05LY",
      procedureName: "RNAV(GPS) Y RWY 05L",
      procedureFamily: "RNAV_GPS",
      lon: -78.92,
      lat: 35.77,
      altitudeFt: 3000,
      geometryAltitudeFt: 3000,
      altM: 914.4,
      sequence: 10,
      legType: "IF",
      sourceLine: 10,
      timeSeconds: 0,
      xM: -10000,
      yM: -10000,
      distanceM: 0,
    },
    {
      fixId: "fix:WEPAS",
      ident: "WEPAS",
      role: "FAF",
      branchId: "branch:R",
      branchIdent: "R",
      branchRole: "final",
      routeId: "KRDU-R05LY-R",
      branchKey: "R",
      transitionIdent: null,
      procedureIdent: "R05LY",
      procedureName: "RNAV(GPS) Y RWY 05L",
      procedureFamily: "RNAV_GPS",
      lon: -78.88,
      lat: 35.8,
      altitudeFt: 2200,
      geometryAltitudeFt: 2200,
      altM: 670.56,
      sequence: 20,
      legType: "TF",
      sourceLine: 20,
      timeSeconds: 120,
      xM: -5000,
      yM: -5000,
      distanceM: 5555.1,
    },
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      role: "MAPt",
      branchId: "branch:R",
      branchIdent: "R",
      branchRole: "final",
      routeId: "KRDU-R05LY-R",
      branchKey: "R",
      transitionIdent: null,
      procedureIdent: "R05LY",
      procedureName: "RNAV(GPS) Y RWY 05L",
      procedureFamily: "RNAV_GPS",
      lon: -78.8,
      lat: 35.87,
      altitudeFt: 367,
      geometryAltitudeFt: 367,
      altM: 111.86,
      sequence: 30,
      legType: "TF",
      sourceLine: 30,
      timeSeconds: 320,
      xM: 0,
      yM: 0,
      distanceM: 15881.8,
    },
  ],
};

describe("buildFinalVerticalProfileOverlays", () => {
  it("anchors final constraints at the MAPT/runway station", () => {
    const [overlay] = buildFinalVerticalProfileOverlays(document, [finalPolyline]);

    expect(overlay.profileId).toBe("profile:R05LY:RW05L");
    expect(overlay.constraintPoints.map((point) => [point.ident, Math.round(point.stationM)])).toEqual([
      ["SCHOO", -15882],
      ["WEPAS", -10327],
      ["RW05L", 0],
    ]);
    expect(overlay.constraintPoints.at(-1)).toMatchObject({
      fixRef: "fix:RW05L",
      altitudeFt: 367,
    });
  });

  it("builds an estimated GPA reference line when CIFP has glidepath angle but no TCH", () => {
    const [overlay] = buildFinalVerticalProfileOverlays(document, [finalPolyline]);

    expect(overlay.glidepathReference).toMatchObject({
      kind: "gpa_from_threshold",
      gpaDeg: 3,
      thresholdFixRef: "fix:RW05L",
      thresholdAltitudeFt: 367,
      estimated: true,
    });
    expect(overlay.glidepathReference?.points).toHaveLength(3);
    expect(overlay.glidepathReference?.points[0].altitudeFt).toBeGreaterThan(
      overlay.glidepathReference?.points[1].altitudeFt ?? 0,
    );
    expect(overlay.glidepathReference?.points.at(-1)).toMatchObject({
      stationM: 0,
      altitudeFt: 367,
    });
  });

  it("skips profiles whose branch is not currently charted", () => {
    expect(buildFinalVerticalProfileOverlays(document, [])).toEqual([]);
  });
});
