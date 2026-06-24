import { describe, expect, it } from "vitest";
import type {
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailLeg,
} from "../procedureDetails";
import {
  buildProcedureConstraint,
  procedureConstraintAltitudesM,
} from "../procedureConstraint";

function leg(
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

const finalBranch: ProcedureDetailBranch = {
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

const document: ProcedureDetailDocument = {
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
  branches: [finalBranch],
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

describe("buildProcedureConstraint", () => {
  it("derives the canonical waypoint sequence from the detail document", () => {
    const constraint = buildProcedureConstraint(document);
    expect(constraint).not.toBeNull();
    expect(constraint!.procedureUid).toBe("KRDU-R05LY-RW05L");
    expect(constraint!.runwayIdent).toBe("RW05L");
    expect(constraint!.branchId).toBe("branch:R");
    expect(constraint!.nominalSpeedKt).toBe(140);
    expect(constraint!.waypoints.map((wp) => wp.ident)).toEqual(["SCHOO", "WEPAS", "RW05L"]);
    expect(constraint!.waypoints.map((wp) => wp.role)).toEqual(["IF", "FAF", "MAPt"]);
  });

  it("carries the canonical altitude window, reference altitude and speed", () => {
    const constraint = buildProcedureConstraint(document)!;
    const [schoo, wepas, threshold] = constraint.waypoints;

    expect(schoo.altitude).toEqual({ kind: "AT_OR_ABOVE", minFtMsl: 3000, sourceText: "3000 ft" });
    expect(schoo.altitudeRefFt).toBe(3000);
    expect(wepas.altitude).toEqual({ kind: "AT_OR_ABOVE", minFtMsl: 2200, sourceText: "2200 ft" });
    expect(wepas.speedMaxKt).toBe(170);
    expect(threshold.altitude).toEqual({ kind: "AT", minFtMsl: 424, maxFtMsl: 424, sourceText: "424 ft" });
    expect(threshold.geometryAltFt).toBe(367);
  });

  it("derives the final approach course and coded glidepath from the source", () => {
    const constraint = buildProcedureConstraint(document)!;
    // WEPAS -> RW05L true bearing (~054 mag at RDU, ~9 deg W variation -> ~045 true).
    expect(constraint.approachCourseDeg).toBeGreaterThan(40);
    expect(constraint.approachCourseDeg).toBeLessThan(50);
    expect(constraint.glidepath).toEqual({ angleDeg: 3.0, tchFt: 57 });
  });

  it("exposes reference altitudes in metres for the optimizer", () => {
    const constraint = buildProcedureConstraint(document)!;
    const altitudesM = procedureConstraintAltitudesM(constraint);
    expect(altitudesM[0]).toBeCloseTo(3000 * 0.3048, 6);
    expect(altitudesM[2]).toBeCloseTo(424 * 0.3048, 6);
  });

  it("returns null when no renderable branch yields two waypoints", () => {
    const empty: ProcedureDetailDocument = { ...document, branches: [] };
    expect(buildProcedureConstraint(empty)).toBeNull();
  });
});
