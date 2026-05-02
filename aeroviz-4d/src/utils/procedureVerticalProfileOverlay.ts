import type { ProcedureDetailDocument, ProcedureDetailVerticalProfile } from "../data/procedureDetails";
import type { ProcedureBranchPolyline, ProcedureChartPoint } from "./procedureDetailsGeometry";

const FEET_PER_METER = 3.280839895;

export interface FinalVerticalProfileOverlayPoint {
  fixRef: string;
  ident: string;
  role: string;
  stationM: number;
  altitudeFt: number;
  sourceLine: number | null;
}

export interface FinalVerticalProfileOverlayPathPoint {
  stationM: number;
  altitudeFt: number;
}

export interface FinalVerticalProfileOverlay {
  profileId: string;
  branchId: string;
  appliesToModes: string[];
  constraintPoints: FinalVerticalProfileOverlayPoint[];
  glidepathReference: {
    kind: "gpa_from_threshold";
    gpaDeg: number;
    thresholdFixRef: string | null;
    thresholdAltitudeFt: number;
    points: FinalVerticalProfileOverlayPathPoint[];
    estimated: boolean;
  } | null;
  warnings: string[];
}

function pointRole(point: ProcedureChartPoint): string {
  return point.role.toUpperCase();
}

function profileAnchorPoint(points: ProcedureChartPoint[]): ProcedureChartPoint | null {
  if (points.length === 0) return null;
  return (
    points.find((point) => pointRole(point) === "MAPT") ??
    points.reduce((best, point) =>
      Math.hypot(point.xM, point.yM) < Math.hypot(best.xM, best.yM) ? point : best,
    )
  );
}

function sampleAltitudeFt(
  sample: ProcedureDetailVerticalProfile["constraintSamples"][number],
): number | null {
  return sample.geometryAltitudeFt ?? sample.altitudeFt;
}

function thresholdAltitudeFt(
  document: ProcedureDetailDocument,
  profile: ProcedureDetailVerticalProfile,
): number | null {
  const thresholdFixRef = profile.toFixRef ?? document.runway.landingThresholdFixRef;
  const thresholdSample = profile.constraintSamples.find((sample) => sample.fixRef === thresholdFixRef);
  const sampleAltitude = thresholdSample ? sampleAltitudeFt(thresholdSample) : null;
  return sampleAltitude ?? document.runway.threshold?.elevationFt ?? null;
}

function buildGlidepathReference(
  document: ProcedureDetailDocument,
  profile: ProcedureDetailVerticalProfile,
  anchorDistanceM: number,
): FinalVerticalProfileOverlay["glidepathReference"] {
  const gpaDeg = profile.glidepathAngleDeg;
  const baseAltitudeFt = thresholdAltitudeFt(document, profile);
  if (
    typeof gpaDeg !== "number" ||
    !Number.isFinite(gpaDeg) ||
    gpaDeg <= 0 ||
    typeof baseAltitudeFt !== "number" ||
    !Number.isFinite(baseAltitudeFt)
  ) {
    return null;
  }

  const gpaRad = (gpaDeg * Math.PI) / 180;
  const orderedSamples = [...profile.constraintSamples].sort(
    (left, right) => left.distanceFromStartM - right.distanceFromStartM,
  );
  const points = orderedSamples.map((sample) => {
    const stationM = sample.distanceFromStartM - anchorDistanceM;
    const distanceBeforeThresholdM = Math.max(0, -stationM);
    return {
      stationM,
      altitudeFt: baseAltitudeFt + Math.tan(gpaRad) * distanceBeforeThresholdM * FEET_PER_METER,
    };
  });

  if (points.length < 2) return null;

  return {
    kind: "gpa_from_threshold",
    gpaDeg,
    thresholdFixRef: profile.toFixRef ?? document.runway.landingThresholdFixRef,
    thresholdAltitudeFt: baseAltitudeFt,
    points,
    estimated: profile.thresholdCrossingHeightFt === null,
  };
}

export function buildFinalVerticalProfileOverlays(
  document: ProcedureDetailDocument,
  polylines: ProcedureBranchPolyline[],
): FinalVerticalProfileOverlay[] {
  const branchById = new Map(polylines.map((branch) => [branch.branchId, branch]));

  return document.verticalProfiles.flatMap((profile): FinalVerticalProfileOverlay[] => {
    const branch = branchById.get(profile.branchId);
    const anchor = branch ? profileAnchorPoint(branch.points) : null;
    if (!anchor) return [];

    const anchorSample =
      profile.constraintSamples.find((sample) => sample.fixRef === anchor.fixId) ??
      profile.constraintSamples.find((sample) => sample.fixRef === profile.toFixRef) ??
      profile.constraintSamples[profile.constraintSamples.length - 1];
    if (!anchorSample) return [];

    const anchorDistanceM = anchorSample.distanceFromStartM;
    const constraintPoints = profile.constraintSamples.flatMap((sample): FinalVerticalProfileOverlayPoint[] => {
      const altitudeFt = sampleAltitudeFt(sample);
      if (typeof altitudeFt !== "number" || !Number.isFinite(altitudeFt)) return [];
      return [
        {
          fixRef: sample.fixRef,
          ident: sample.ident,
          role: sample.role,
          stationM: sample.distanceFromStartM - anchorDistanceM,
          altitudeFt,
          sourceLine: sample.sourceLine ?? null,
        },
      ];
    });

    if (constraintPoints.length === 0) return [];

    return [
      {
        profileId: profile.profileId,
        branchId: profile.branchId,
        appliesToModes: profile.appliesToModes,
        constraintPoints,
        glidepathReference: buildGlidepathReference(document, profile, anchorDistanceM),
        warnings: profile.warnings,
      },
    ];
  });
}
