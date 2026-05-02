import { describe, expect, it } from "vitest";
import {
  attachProcedureAnnotation,
  annotationStatusLabel,
  procedureAnnotationMeaning,
  resolvePickedProcedureAnnotation,
  type ProcedureEntityAnnotation,
} from "../procedureAnnotations";

const annotation: ProcedureEntityAnnotation = {
  entityId: "entity-1",
  label: "LNAV/VNAV OCS",
  title: "Test OCS",
  kind: "LNAV_VNAV_OCS",
  status: "ESTIMATED",
  airportId: "KRDU",
  runwayId: "RW32",
  procedureUid: "KRDU-R32-RW32",
  procedureId: "R32",
  procedureName: "RNAV(GPS) RWY 32",
  branchId: "branch:R",
  branchName: "RW32",
  branchRole: "STRAIGHT_IN",
  segmentId: "segment:final",
  segmentType: "FINAL_LNAV_VNAV",
  meaning: "test",
  parameters: [],
  diagnostics: [],
  sourceRefs: [],
};

describe("procedureAnnotations", () => {
  it("formats construction status labels", () => {
    expect(annotationStatusLabel("DEBUG_ESTIMATE")).toBe("Debug Estimate");
  });

  it("explains estimated LNAV/VNAV OCS geometry", () => {
    expect(procedureAnnotationMeaning("LNAV_VNAV_OCS", "ESTIMATED")).toContain(
      "Sloping LNAV/VNAV",
    );
  });

  it("resolves annotations from picked Cesium entities", () => {
    const entity = attachProcedureAnnotation({ id: "entity-1" }, annotation);

    expect(resolvePickedProcedureAnnotation({ id: entity })).toEqual(annotation);
  });
});
