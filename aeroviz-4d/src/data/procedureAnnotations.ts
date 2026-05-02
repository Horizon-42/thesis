export type ProcedureAnnotationKind =
  | "FIX"
  | "SEGMENT_CENTERLINE"
  | "SEGMENT_ENVELOPE_PRIMARY"
  | "SEGMENT_ENVELOPE_SECONDARY"
  | "FINAL_OEA"
  | "LNAV_VNAV_OCS"
  | "PRECISION_SURFACE"
  | "ALIGNED_CONNECTOR"
  | "MISSED_SURFACE"
  | "CA_COURSE_GUIDE"
  | "CA_CENTERLINE"
  | "CA_ENDPOINT"
  | "TURNING_MISSED_DEBUG"
  | "TURN_FILL"
  | "MISSING_FINAL_SURFACE";

export type ProcedureAnnotationStatus =
  | "SOURCE_BACKED"
  | "ESTIMATED"
  | "DEBUG_ESTIMATE"
  | "VISUAL_FILL_ONLY"
  | "MISSING_SOURCE";

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

export function annotationStatusLabel(status: ProcedureAnnotationStatus): string {
  return status
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
