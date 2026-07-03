import { describe, expect, it } from "vitest";
import type { ProcedureDetailDocument } from "../procedureDetails";
import {
  buildProcedureConstraint,
  procedureConstraintAltitudesM,
  procedureThresholdAnchor,
} from "../procedureConstraint";
import {
  krduR05lyDocument,
  krduR05lyFinalBranch as finalBranch,
  leg,
} from "./krduR05lyDetailDocument.fixture";

const document = krduR05lyDocument;


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

  it("ends at the runway threshold and drops a post-MAPt missed-approach leg", () => {
    // A final branch that continues PAST the MAPt — the published missed approach (a climb back
    // up, like KRDU R32's RW32 -> DUHAM). The constraint must end at the threshold; the missed
    // approach must NOT constrain a landing optimization. (Render/profile views keep the full one.)
    const withMissed: ProcedureDetailDocument = {
      ...document,
      fixes: [
        ...document.fixes,
        { fixId: "fix:MISSD", ident: "MISSD", kind: "named_fix", position: { lon: -78.70, lat: 35.95 }, elevationFt: null, roleHints: [], sourceRefs: [] },
      ],
      branches: [
        {
          ...finalBranch,
          legs: [
            ...finalBranch.legs,
            leg({
              legId: "leg:R:040",
              sequence: 40,
              path: { pathTerminator: "TF", constructionMethod: "track_to_fix", startFixRef: "fix:RW05L", endFixRef: "fix:MISSD" },
              roleAtEnd: "Route",
              constraints: { altitude: { qualifier: "atOrAbove", valueFt: 2200, rawText: "2200 ft" }, speedKt: null, geometryAltitudeFt: 2200 },
            }),
          ],
        },
      ],
    };
    const constraint = buildProcedureConstraint(withMissed)!;
    expect(constraint.waypoints.map((wp) => wp.ident)).toEqual(["SCHOO", "WEPAS", "RW05L"]);
    expect(constraint.waypoints[constraint.waypoints.length - 1].role).toBe("MAPt");
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

  it("anchors the optimizer target on the CIFP threshold, course in simulator convention", () => {
    // The optimizer target must be the procedure's OWN threshold (the frame anchor the backend
    // validates), not the runway.geojson pavement midpoint (can be hundreds of metres off).
    const constraint = buildProcedureConstraint(document)!;
    const anchor = procedureThresholdAnchor(constraint, document);
    expect(anchor.lon).toBeCloseTo(-78.80196389, 8);
    expect(anchor.lat).toBeCloseTo(35.87445, 8);
    expect(anchor.elevationM).toBeCloseTo(367 * 0.3048, 3);
    // WEPAS -> RW05L runs ~045 deg true (compass). The SIMULATOR convention is 0 = East, CCW
    // toward North, so the same course reads as 90 - 45 = ~45 deg — and critically it must NOT
    // equal the compass approachCourseDeg except by the 45-deg coincidence of this runway; assert
    // the exact convention via the complement identity.
    expect(anchor.psiDeg).toBeCloseTo(90 - constraint.approachCourseDeg!, 6);
  });

  it("threshold anchor reports a null elevation when the document has none", () => {
    const constraint = buildProcedureConstraint(document)!;
    const withoutElevation: ProcedureDetailDocument = {
      ...document,
      runway: { ...document.runway, threshold: { lon: -78.80196389, lat: 35.87445, elevationFt: null } },
    };
    expect(procedureThresholdAnchor(constraint, withoutElevation).elevationM).toBeNull();
  });

  it("returns null when no renderable branch yields two waypoints", () => {
    const empty: ProcedureDetailDocument = { ...document, branches: [] };
    expect(buildProcedureConstraint(empty)).toBeNull();
  });
});
