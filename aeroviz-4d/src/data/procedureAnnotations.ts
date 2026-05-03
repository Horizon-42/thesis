export type ProcedureAnnotationKind =
  | "FIX"
  | "SEGMENT_CENTERLINE"
  | "SEGMENT_ENVELOPE_PRIMARY"
  | "SEGMENT_ENVELOPE_SECONDARY"
  | "SEGMENT_VERTICAL_PROFILE"
  | "FINAL_OEA"
  | "FINAL_VERTICAL_REFERENCE"
  | "ALTITUDE_CONSTRAINT"
  | "FINAL_ALTITUDE_CONSTRAINT"
  | "LNAV_VNAV_OCS"
  | "PRECISION_SURFACE"
  | "ALIGNED_CONNECTOR"
  | "MISSED_SURFACE"
  | "CA_COURSE_GUIDE"
  | "CA_CENTERLINE"
  | "CA_ENDPOINT"
  | "CA_MAHF_CONNECTOR"
  | "TURNING_MISSED_DEBUG"
  | "TURN_FILL"
  | "MISSING_FINAL_SURFACE";

export type ProcedureAnnotationStatus =
  | "SOURCE_BACKED"
  | "ESTIMATED"
  | "DEBUG_ESTIMATE"
  | "VISUAL_FILL_ONLY"
  | "MISSING_SOURCE";

export type ProcedureDisplayLevel =
  | "CORE"
  | "PROTECTION"
  | "ESTIMATED"
  | "VISUAL_AID"
  | "DEBUG";

export const PROCEDURE_DISPLAY_LEVEL_OPTIONS: Array<{
  value: ProcedureDisplayLevel;
  label: string;
  description: string;
}> = [
  { value: "CORE", label: "Core", description: "Source-backed fixes and coded paths" },
  { value: "PROTECTION", label: "Protection", description: "Core plus source-backed protection" },
  { value: "ESTIMATED", label: "Estimated", description: "Adds inferred operational geometry" },
  { value: "VISUAL_AID", label: "Visual Aid", description: "Adds readability fill geometry" },
  { value: "DEBUG", label: "Debug", description: "Adds debug and missing-source markers" },
];

export interface ProcedureAnnotationParameter {
  label: string;
  value: string;
}

export interface ProcedureEntityAnnotation {
  entityId: string;
  label: string;
  title: string;
  kind: ProcedureAnnotationKind;
  status: ProcedureAnnotationStatus;
  airportId: string;
  runwayId: string | null;
  procedureUid: string;
  procedureId: string;
  procedureName: string;
  branchId: string;
  branchName: string;
  branchRole: string;
  segmentId?: string;
  segmentType?: string;
  legId?: string;
  legType?: string;
  meaning: string;
  parameters: ProcedureAnnotationParameter[];
  diagnostics: string[];
  sourceRefs: string[];
}

export const PROCEDURE_ANNOTATION_ENTITY_FIELD = "__aeroVizProcedureAnnotation";

export function annotationStatusLabel(status: ProcedureAnnotationStatus): string {
  return status
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function displayLevelRank(level: ProcedureDisplayLevel): number {
  if (level === "CORE") return 1;
  if (level === "PROTECTION") return 2;
  if (level === "ESTIMATED") return 3;
  if (level === "VISUAL_AID") return 4;
  return 5;
}

export function procedureAnnotationDisplayLevel(
  annotation: ProcedureEntityAnnotation,
): ProcedureDisplayLevel {
  if (
    annotation.kind === "PRECISION_SURFACE" ||
    annotation.kind === "TURNING_MISSED_DEBUG" ||
    annotation.kind === "MISSING_FINAL_SURFACE"
  ) {
    return "DEBUG";
  }
  if (annotation.kind === "TURN_FILL") return "VISUAL_AID";
  if (
    annotation.kind === "LNAV_VNAV_OCS" ||
    annotation.kind === "ALIGNED_CONNECTOR" ||
    annotation.kind === "FINAL_VERTICAL_REFERENCE" ||
    annotation.kind === "SEGMENT_VERTICAL_PROFILE" ||
    annotation.kind === "CA_CENTERLINE" ||
    annotation.kind === "CA_ENDPOINT" ||
    annotation.kind === "CA_MAHF_CONNECTOR" ||
    (annotation.kind === "SEGMENT_CENTERLINE" && annotation.status === "ESTIMATED") ||
    (annotation.kind === "SEGMENT_ENVELOPE_PRIMARY" && annotation.status === "ESTIMATED") ||
    (annotation.kind === "SEGMENT_ENVELOPE_SECONDARY" && annotation.status === "ESTIMATED") ||
    (annotation.kind === "MISSED_SURFACE" && annotation.status === "ESTIMATED")
  ) {
    return "ESTIMATED";
  }
  if (
    annotation.kind === "SEGMENT_ENVELOPE_PRIMARY" ||
    annotation.kind === "SEGMENT_ENVELOPE_SECONDARY" ||
    annotation.kind === "FINAL_OEA" ||
    annotation.kind === "ALTITUDE_CONSTRAINT" ||
    annotation.kind === "FINAL_ALTITUDE_CONSTRAINT" ||
    annotation.kind === "MISSED_SURFACE"
  ) {
    return "PROTECTION";
  }
  return "CORE";
}

export function isProcedureAnnotationVisibleAtDisplayLevel(
  annotation: ProcedureEntityAnnotation | null,
  selectedLevel: ProcedureDisplayLevel,
): boolean {
  if (!annotation) return selectedLevel === "DEBUG";
  return displayLevelRank(procedureAnnotationDisplayLevel(annotation)) <= displayLevelRank(selectedLevel);
}

export function procedureAnnotationMeaning(
  kind: ProcedureAnnotationKind,
  status: ProcedureAnnotationStatus,
): string {
  if (kind === "FIX") {
    return "Procedure fix used by one or more coded legs. Role labels identify chart functions such as IF, FAF, MAP, runway, or missed hold.";
  }
  if (kind === "SEGMENT_CENTERLINE") {
    return status === "ESTIMATED"
      ? "Estimated nominal path for this segment. It is used to place display geometry when source legs do not provide a normal terminating fix."
      : "Nominal coded path for this procedure segment. It is the reference line used to place lateral protected areas.";
  }
  if (kind === "SEGMENT_ENVELOPE_PRIMARY") {
    return "Primary lateral protected-area footprint around the segment centerline. It shows horizontal containment only and is not an OCS or vertical clearance surface.";
  }
  if (kind === "SEGMENT_ENVELOPE_SECONDARY") {
    return "Secondary lateral protected-area footprint outside the primary envelope. It shows the outer horizontal buffer or taper and is not an OCS or vertical clearance surface.";
  }
  if (kind === "SEGMENT_VERTICAL_PROFILE") {
    return "Estimated vertical profile surface connecting adjacent procedure fixes on non-missed segments. Its lateral width follows the primary protected area; missed CA/DF/HM continuity is shown by missed-approach layers instead.";
  }
  if (kind === "FINAL_OEA") {
    return "Final-segment obstacle evaluation area used as the lateral reference for final protected-area visualization.";
  }
  if (kind === "FINAL_VERTICAL_REFERENCE") {
    return "Estimated final-approach vertical reference built from coded GPA. When source TCH is available, the reference is GPA/TCH-aligned; otherwise it is anchored at runway/MAPt elevation for display only.";
  }
  if (kind === "ALTITUDE_CONSTRAINT") {
    return "Published altitude constraint at the leg endpoint. It anchors the segment vertical profile when source altitude data is available.";
  }
  if (kind === "FINAL_ALTITUDE_CONSTRAINT") {
    return "Published final-approach altitude constraint shown at its terminal fix for vertical cross-checking.";
  }
  if (kind === "LNAV_VNAV_OCS") {
    return "Sloping LNAV/VNAV obstacle clearance surface estimate built from final centerline plus source-backed GPA and TCH path point data.";
  }
  if (kind === "PRECISION_SURFACE") {
    return "Debug-estimate LPV/GLS precision surface. It is useful for visual comparison but is not certified W/X/Y construction.";
  }
  if (kind === "ALIGNED_CONNECTOR") {
    return "Visual connector used to show how intermediate and final protected areas join around the alignment transition.";
  }
  if (kind === "MISSED_SURFACE") {
    return status === "ESTIMATED"
      ? "Missed-approach protected area built from estimated CA centerline data."
      : "Missed-approach protected area showing how protection continues after the MAPt.";
  }
  if (kind === "CA_COURSE_GUIDE") {
    return "Course-to-altitude missed-approach guide. It shows the published course direction but does not by itself define a terminating point.";
  }
  if (kind === "CA_CENTERLINE") {
    return "Estimated CA centerline built from course, altitude target, climb gradient, and start fix elevation.";
  }
  if (kind === "CA_ENDPOINT") {
    return "Estimated endpoint for a course-to-altitude missed leg. It is derived from the climb model and is not an explicit source fix.";
  }
  if (kind === "CA_MAHF_CONNECTOR") {
    return "Estimated continuity connector from a CA endpoint to the later missed approach holding fix. It is not source-coded leg geometry.";
  }
  if (kind === "TURNING_MISSED_DEBUG") {
    return "Debug-only turning missed approach primitive, used to inspect turn initiation, early/late baselines, nominal path, or wind spiral placeholders.";
  }
  if (kind === "TURN_FILL") {
    return "Visual fill patch between adjacent segment envelopes. It improves readability at turns but is not a compliant turn construction.";
  }
  return "Marker showing that an expected final protected surface was not constructed because required source data or implementation is missing.";
}

export function attachProcedureAnnotation<T extends object>(
  target: T,
  annotation: ProcedureEntityAnnotation,
): T {
  return Object.assign(target, { [PROCEDURE_ANNOTATION_ENTITY_FIELD]: annotation });
}

export function getProcedureAnnotation(target: unknown): ProcedureEntityAnnotation | null {
  if (!target || typeof target !== "object") return null;
  const candidate = (target as Record<string, unknown>)[PROCEDURE_ANNOTATION_ENTITY_FIELD];
  if (!candidate || typeof candidate !== "object") return null;
  return candidate as ProcedureEntityAnnotation;
}

export function resolvePickedProcedureAnnotation(picked: unknown): ProcedureEntityAnnotation | null {
  if (!picked || typeof picked !== "object") return null;
  const pickedRecord = picked as Record<string, unknown>;
  return (
    getProcedureAnnotation(pickedRecord.id) ??
    getProcedureAnnotation(pickedRecord.primitive) ??
    getProcedureAnnotation(pickedRecord.collection) ??
    getProcedureAnnotation(picked)
  );
}
