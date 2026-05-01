import type {
  BuildDiagnostic,
  ProcedureSegment,
} from "../data/procedurePackage";
import type {
  LateralEnvelopeGeometry,
  SegmentGeometryBundle,
} from "./procedureSegmentGeometry";

export type MissedSectionSurfaceType =
  | "MISSED_SECTION1_ENVELOPE"
  | "MISSED_SECTION2_STRAIGHT_ENVELOPE";

export interface MissedSectionSurfaceGeometry {
  segmentId: string;
  surfaceType: MissedSectionSurfaceType;
  primary: LateralEnvelopeGeometry;
  secondaryOuter: LateralEnvelopeGeometry | null;
}

function diagnostic(
  segment: ProcedureSegment,
  message: string,
): BuildDiagnostic {
  return {
    severity: "WARN",
    segmentId: segment.segmentId,
    code: "SOURCE_INCOMPLETE",
    message,
    sourceRefs: segment.sourceRefs,
  };
}

export function buildMissedSectionSurface(
  segment: ProcedureSegment,
  segmentGeometry: SegmentGeometryBundle,
): { geometry: MissedSectionSurfaceGeometry | null; diagnostics: BuildDiagnostic[] } {
  if (segment.segmentType !== "MISSED_S1" && segment.segmentType !== "MISSED_S2") {
    return { geometry: null, diagnostics: [] };
  }

  if (!segmentGeometry.primaryEnvelope) {
    return {
      geometry: null,
      diagnostics: [
        diagnostic(
          segment,
          `${segment.segmentId}: missed section surface requires a primary envelope.`,
        ),
      ],
    };
  }

  if (segment.segmentType === "MISSED_S2" && segmentGeometry.centerline.isArc) {
    return {
      geometry: null,
      diagnostics: [
        diagnostic(
          segment,
          `${segment.segmentId}: turning missed section 2 surface is not implemented yet.`,
        ),
      ],
    };
  }

  return {
    geometry: {
      segmentId: segment.segmentId,
      surfaceType:
        segment.segmentType === "MISSED_S1"
          ? "MISSED_SECTION1_ENVELOPE"
          : "MISSED_SECTION2_STRAIGHT_ENVELOPE",
      primary: segmentGeometry.primaryEnvelope,
      secondaryOuter: segmentGeometry.secondaryEnvelope ?? null,
    },
    diagnostics: [],
  };
}
