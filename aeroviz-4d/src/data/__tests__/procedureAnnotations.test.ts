import { describe, expect, it } from "vitest";
import {
  attachProcedureAnnotation,
  annotationStatusLabel,
  isProcedureAnnotationVisibleAtDisplayLevel,
  procedureAnnotationDisplayLevel,
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
    expect(procedureAnnotationMeaning("SEGMENT_VERTICAL_PROFILE", "ESTIMATED")).toContain(
      "adjacent procedure fixes",
    );
  });

  it("resolves annotations from picked Cesium entities", () => {
    const entity = attachProcedureAnnotation({ id: "entity-1" }, annotation);

    expect(resolvePickedProcedureAnnotation({ id: entity })).toEqual(annotation);
  });

  it("classifies annotation display levels by kind and status", () => {
    expect(procedureAnnotationDisplayLevel({ ...annotation, kind: "FIX", status: "SOURCE_BACKED" })).toBe("CORE");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "SEGMENT_ENVELOPE_PRIMARY",
        status: "SOURCE_BACKED",
      }),
    ).toBe("PROTECTION");
    expect(procedureAnnotationDisplayLevel(annotation)).toBe("ESTIMATED");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "SEGMENT_ENVELOPE_PRIMARY",
        status: "ESTIMATED",
      }),
    ).toBe("ESTIMATED");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "SEGMENT_VERTICAL_PROFILE",
        status: "ESTIMATED",
      }),
    ).toBe("ESTIMATED");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "ALTITUDE_CONSTRAINT",
        status: "SOURCE_BACKED",
      }),
    ).toBe("PROTECTION");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "TURN_FILL",
        status: "VISUAL_FILL_ONLY",
      }),
    ).toBe("VISUAL_AID");
    expect(
      procedureAnnotationDisplayLevel({
        ...annotation,
        kind: "PRECISION_SURFACE",
        status: "DEBUG_ESTIMATE",
      }),
    ).toBe("DEBUG");
  });

  it("applies cumulative procedure display-level visibility", () => {
    expect(isProcedureAnnotationVisibleAtDisplayLevel(annotation, "PROTECTION")).toBe(false);
    expect(isProcedureAnnotationVisibleAtDisplayLevel(annotation, "ESTIMATED")).toBe(true);
    expect(isProcedureAnnotationVisibleAtDisplayLevel(null, "PROTECTION")).toBe(false);
    expect(isProcedureAnnotationVisibleAtDisplayLevel(null, "DEBUG")).toBe(true);
  });
});
