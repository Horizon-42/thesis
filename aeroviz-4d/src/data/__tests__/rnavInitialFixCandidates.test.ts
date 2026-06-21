import { describe, expect, it } from "vitest";
import {
  buildRnavInitialFixCandidates,
} from "../rnavInitialFixCandidates";
import type {
  ProcedureDetailDocument,
  ProcedureDetailFix,
  ProcedureDetailLeg,
} from "../procedureDetails";

describe("rnavInitialFixCandidates", () => {
  it("builds initial aircraft candidates from RNAV IF legs", () => {
    const candidates = buildRnavInitialFixCandidates([
      makeDocument({
        fixes: [
          makeFix("fix:IFIX", "IFIX", 0, 0, ["IF"]),
          makeFix("fix:NEXT", "NEXT", 0.01, 0, ["FAF"]),
          makeFix("fix:RW09", "RW09", 0.02, 0, ["MAPt"]),
        ],
        legs: [
          makeLeg("leg:R:010", 10, null, "fix:IFIX", "IF", "IF", 3000),
          makeLeg("leg:R:020", 20, "fix:IFIX", "fix:NEXT", "TF", "FAF", 2200),
          makeLeg("leg:R:030", 30, "fix:NEXT", "fix:RW09", "TF", "MAPt", 400),
        ],
      }),
    ], "RW09");

    expect(candidates).toHaveLength(1);
    const candidate = candidates[0];
    expect(candidate).toMatchObject({
      key: "TEST-R09-RW09|branch:R|fix:IFIX|fix:NEXT|914.4000000000001",
      runwayIdent: "RW09",
      procedureUid: "TEST-R09-RW09",
      procedureIdent: "R09",
      chartName: "RNAV(GPS) RWY 09",
      branchId: "branch:R",
      branchIdent: "R",
      fixId: "fix:IFIX",
      fixIdent: "IFIX",
      nextFixId: "fix:NEXT",
      nextFixIdent: "NEXT",
      lon: 0,
      lat: 0,
      headingDeg: 0,
    });
    expect(candidate?.altM).toBeCloseTo(914.4);
  });

  it("skips IF fixes that cannot produce a full initial state", () => {
    const candidates = buildRnavInitialFixCandidates([
      makeDocument({
        fixes: [
          makeFix("fix:NOALT", "NOALT", 0, 0, ["IF"]),
          makeFix("fix:NEXT", "NEXT", 0, 0.01, ["FAF"]),
        ],
        legs: [
          makeLeg("leg:R:010", 10, null, "fix:NOALT", "IF", "IF", null),
          makeLeg("leg:R:020", 20, "fix:NOALT", "fix:NEXT", "TF", "FAF", 2200),
        ],
      }),
    ], "RW09");

    expect(candidates).toEqual([]);
  });
});

function makeDocument({
  fixes,
  legs,
}: {
  fixes: ProcedureDetailFix[];
  legs: ProcedureDetailLeg[];
}): ProcedureDetailDocument {
  return {
    schemaVersion: "1.0.0",
    modelType: "rnav-procedure-runway",
    procedureUid: "TEST-R09-RW09",
    provenance: {
      assemblyMode: "test",
      researchUseOnly: true,
      sources: [],
      warnings: [],
    },
    airport: {
      icao: "TEST",
      faa: "TST",
      name: "Test Airport",
    },
    runway: {
      ident: "RW09",
      landingThresholdFixRef: "fix:RW09",
      threshold: {
        lon: 0.02,
        lat: 0,
        elevationFt: 400,
      },
    },
    procedure: {
      procedureType: "SIAP",
      procedureFamily: "RNAV_GPS",
      procedureIdent: "R09",
      chartName: "RNAV(GPS) RWY 09",
      variant: null,
      runwayIdent: "RW09",
      baseBranchIdent: "R",
      approachModes: ["LNAV"],
    },
    fixes,
    branches: [{
      branchId: "branch:R",
      branchKey: "R",
      branchIdent: "R",
      procedureType: "R",
      transitionIdent: null,
      branchRole: "final",
      sequenceOrder: 1,
      mergeFixRef: null,
      continuesWithBranchId: null,
      defaultVisible: true,
      warnings: [],
      legs,
    }],
    verticalProfiles: [],
    validation: {
      expectedRunwayIdent: "RW09",
      expectedIF: "fix:IFIX",
      expectedFAF: "fix:NEXT",
      expectedMAPt: "fix:RW09",
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
}

function makeFix(
  fixId: string,
  ident: string,
  lon: number,
  lat: number,
  roleHints: string[],
): ProcedureDetailFix {
  return {
    fixId,
    ident,
    kind: "named_fix",
    position: { lon, lat },
    elevationFt: null,
    roleHints,
    sourceRefs: [],
  };
}

function makeLeg(
  legId: string,
  sequence: number,
  startFixRef: string | null,
  endFixRef: string,
  pathTerminator: string,
  roleAtEnd: string,
  geometryAltitudeFt: number | null,
): ProcedureDetailLeg {
  return {
    legId,
    sequence,
    segmentType: "intermediate",
    path: {
      pathTerminator,
      constructionMethod: "track_to_fix",
      startFixRef,
      endFixRef,
    },
    termination: {
      kind: "fix",
      fixRef: endFixRef,
    },
    constraints: {
      altitude: geometryAltitudeFt === null
        ? null
        : {
            qualifier: "at",
            valueFt: geometryAltitudeFt,
            rawText: `${geometryAltitudeFt} ft`,
          },
      speedKt: null,
      geometryAltitudeFt,
    },
    roleAtEnd,
    sourceRefs: [],
    quality: {
      status: "exact",
      sourceLine: sequence,
      renderedInPlanView: true,
    },
  };
}
