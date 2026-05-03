import type {
  HorizontalPlateAssessmentSegment,
  HorizontalPlateRoute,
  RunwayFrame,
  RunwayProfilePoint,
} from "./runwayProfileGeometry";
import { projectPositionToRunwayFrame } from "./runwayProfileGeometry";
import {
  classifyPointAgainstProtectionSurfaces,
  type ProtectionVolumeAssessment,
} from "./procedureProtectionVolumeAssessment";
import type { GeoPoint } from "./procedureGeoMath";

export type HorizontalPlateContainment = "PRIMARY" | "SECONDARY" | "OUTSIDE";
export type SegmentAssessmentEventKind =
  | "LATERAL_CONTAINMENT"
  | "VERTICAL_DEVIATION"
  | "VERTICAL_OCS";

export interface SegmentAssessmentEvent {
  kind: SegmentAssessmentEventKind;
  label: string;
  valueM?: number;
}

export interface HorizontalPlateSegmentAssessment {
  routeId: string;
  branchId: string;
  activeSegmentId: string;
  segmentIndex: number;
  stationM: number;
  crossTrackErrorM: number;
  verticalErrorM: number | null;
  containment: HorizontalPlateContainment;
  closestPoint: RunwayProfilePoint;
  events: SegmentAssessmentEvent[];
  surfaceAssessment?: ProtectionVolumeAssessment;
}

interface CandidateAssessment extends HorizontalPlateSegmentAssessment {
  absoluteCrossTrackM: number;
  distanceToSegmentM: number;
}

function segmentIdFor(route: HorizontalPlateRoute, segmentIndex: number): string {
  return `${route.branchId}:profile-segment:${segmentIndex + 1}`;
}

const VERTICAL_DEVIATION_EVENT_THRESHOLD_M = 30.48;

function assessmentEvents(
  containment: HorizontalPlateContainment,
  verticalErrorM: number | null,
  usesOcsVerticalReference = false,
): SegmentAssessmentEvent[] {
  const events: SegmentAssessmentEvent[] = [
    {
      kind: "LATERAL_CONTAINMENT",
      label: containment,
    },
  ];

  if (
    verticalErrorM !== null &&
    Math.abs(verticalErrorM) > VERTICAL_DEVIATION_EVENT_THRESHOLD_M
  ) {
    events.push({
      kind: usesOcsVerticalReference ? "VERTICAL_OCS" : "VERTICAL_DEVIATION",
      label: usesOcsVerticalReference
        ? verticalErrorM > 0 ? "ABOVE_OCS" : "BELOW_OCS"
        : verticalErrorM > 0 ? "ABOVE_PROFILE" : "BELOW_PROFILE",
      valueM: verticalErrorM,
    });
  }

  return events;
}

function projectPointToSegment(
  point: Pick<RunwayProfilePoint, "xM" | "yM"> & Partial<Pick<RunwayProfilePoint, "zM">>,
  route: HorizontalPlateRoute,
  segmentId: string,
  segmentPoints: RunwayProfilePoint[],
  primaryHalfWidthM: number,
  secondaryHalfWidthM: number | null,
  segmentIndex: number,
  stationOffsetM: number,
  verticalReferencePoints?: RunwayProfilePoint[],
  usesOcsVerticalReference = false,
): CandidateAssessment | null {
  const start = segmentPoints[segmentIndex];
  const end = segmentPoints[segmentIndex + 1];
  if (!start || !end) return null;

  const dx = end.xM - start.xM;
  const dy = end.yM - start.yM;
  const lengthM = Math.hypot(dx, dy);
  if (lengthM === 0) return null;

  const alongRatio = Math.max(
    0,
    Math.min(
      1,
      ((point.xM - start.xM) * dx + (point.yM - start.yM) * dy) / (lengthM * lengthM),
    ),
  );
  const verticalStart = verticalReferencePoints?.[segmentIndex];
  const verticalEnd = verticalReferencePoints?.[segmentIndex + 1];
  const closestPoint = {
    xM: start.xM + alongRatio * dx,
    yM: start.yM + alongRatio * dy,
    zM:
      verticalStart && verticalEnd
        ? verticalStart.zM + alongRatio * (verticalEnd.zM - verticalStart.zM)
        : start.zM + alongRatio * (end.zM - start.zM),
  };
  const crossTrackErrorM =
    ((point.xM - start.xM) * dy - (point.yM - start.yM) * dx) / lengthM;
  const absoluteCrossTrackM = Math.abs(crossTrackErrorM);
  const distanceToSegmentM = Math.hypot(point.xM - closestPoint.xM, point.yM - closestPoint.yM);

  const containment =
    distanceToSegmentM <= primaryHalfWidthM
      ? "PRIMARY"
      : secondaryHalfWidthM !== null && distanceToSegmentM <= secondaryHalfWidthM
        ? "SECONDARY"
        : "OUTSIDE";
  const verticalErrorM = Number.isFinite(point.zM)
    ? (point.zM as number) - closestPoint.zM
    : null;

  return {
    routeId: route.routeId,
    branchId: route.branchId,
    activeSegmentId: segmentId,
    segmentIndex,
    stationM: stationOffsetM + alongRatio * lengthM,
    crossTrackErrorM,
    verticalErrorM,
    containment,
    closestPoint,
    events: assessmentEvents(containment, verticalErrorM, usesOcsVerticalReference),
    absoluteCrossTrackM,
    distanceToSegmentM,
  };
}

function projectPointToAssessmentSegment(
  point: Pick<RunwayProfilePoint, "xM" | "yM"> & Partial<Pick<RunwayProfilePoint, "zM">>,
  route: HorizontalPlateRoute,
  assessmentSegment: HorizontalPlateAssessmentSegment,
  segmentOffset: number,
  segmentIndexOffset: number,
): HorizontalPlateSegmentAssessment | null {
  let stationOffsetM = 0;
  let best: CandidateAssessment | null = null;
  const verticalReferencePoints =
    assessmentSegment.lnavVnavOcs?.points ?? assessmentSegment.finalVerticalReference?.points;
  const usesOcsVerticalReference = Boolean(assessmentSegment.lnavVnavOcs);

  for (
    let segmentIndex = 0;
    segmentIndex < assessmentSegment.points.length - 1;
    segmentIndex += 1
  ) {
    const candidate = projectPointToSegment(
      point,
      route,
      assessmentSegment.segmentId,
      assessmentSegment.points,
      assessmentSegment.primaryHalfWidthM,
      assessmentSegment.secondaryHalfWidthM,
      segmentIndex,
      stationOffsetM,
      verticalReferencePoints,
      usesOcsVerticalReference,
    );
    if (candidate && (!best || candidate.distanceToSegmentM < best.distanceToSegmentM)) {
      best = candidate;
    }

    const start = assessmentSegment.points[segmentIndex];
    const end = assessmentSegment.points[segmentIndex + 1];
    if (start && end) {
      stationOffsetM += Math.hypot(end.xM - start.xM, end.yM - start.yM);
    }
  }

  if (!best) return null;
  const {
    absoluteCrossTrackM: _absoluteCrossTrackM,
    distanceToSegmentM: _distanceToSegmentM,
    ...assessment
  } = best;
  return {
    ...assessment,
    segmentIndex: assessment.segmentIndex + segmentIndexOffset,
    stationM: assessment.stationM + segmentOffset,
  };
}

function fallbackAssessmentSegments(route: HorizontalPlateRoute): HorizontalPlateAssessmentSegment[] {
  return route.points.slice(0, -1).map((start, index) => ({
    segmentId: segmentIdFor(route, index),
    primaryHalfWidthM: route.halfWidthM,
    secondaryHalfWidthM: null,
    points: [start, route.points[index + 1]],
  }));
}

export function projectPointToHorizontalPlateRoute(
  point: Pick<RunwayProfilePoint, "xM" | "yM"> & Partial<Pick<RunwayProfilePoint, "zM">>,
  route: HorizontalPlateRoute,
): HorizontalPlateSegmentAssessment | null {
  const assessmentSegments = route.assessmentSegments ?? fallbackAssessmentSegments(route);
  let segmentOffset = 0;
  let best: CandidateAssessment | null = null;

  for (let assessmentSegmentIndex = 0; assessmentSegmentIndex < assessmentSegments.length; assessmentSegmentIndex += 1) {
    const assessmentSegment = assessmentSegments[assessmentSegmentIndex];
    const assessment = projectPointToAssessmentSegment(
      point,
      route,
      assessmentSegment,
      segmentOffset,
      route.assessmentSegments ? 0 : assessmentSegmentIndex,
    );
    if (assessment) {
      const distanceToSegmentM = Math.hypot(
        point.xM - assessment.closestPoint.xM,
        point.yM - assessment.closestPoint.yM,
      );
      const candidate = {
        ...assessment,
        absoluteCrossTrackM: Math.abs(assessment.crossTrackErrorM),
        distanceToSegmentM,
      };
      if (!best || candidate.distanceToSegmentM < best.distanceToSegmentM) {
        best = candidate;
      }
    }

    for (let index = 0; index < assessmentSegment.points.length - 1; index += 1) {
      const start = assessmentSegment.points[index];
      const end = assessmentSegment.points[index + 1];
      segmentOffset += Math.hypot(end.xM - start.xM, end.yM - start.yM);
    }
  }

  if (!best) return null;
  const {
    absoluteCrossTrackM: _absoluteCrossTrackM,
    distanceToSegmentM: _distanceToSegmentM,
    ...assessment
  } = best;
  return assessment;
}

export function classifyPointAgainstHorizontalPlateRoutes(
  point: Pick<RunwayProfilePoint, "xM" | "yM"> & Partial<Pick<RunwayProfilePoint, "zM">>,
  routes: HorizontalPlateRoute[],
): HorizontalPlateSegmentAssessment | null {
  const assessments = routes
    .map((route) => projectPointToHorizontalPlateRoute(point, route))
    .filter((assessment): assessment is HorizontalPlateSegmentAssessment => assessment !== null);

  const primaryAssessment = assessments
    .filter((assessment) => assessment.containment === "PRIMARY")
    .sort((left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM))[0];
  if (primaryAssessment) return primaryAssessment;

  const secondaryAssessment = assessments
    .filter((assessment) => assessment.containment === "SECONDARY")
    .sort((left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM))[0];
  if (secondaryAssessment) return secondaryAssessment;

  return assessments.sort(
    (left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM),
  )[0] ?? null;
}

export function classifyGeoPointAgainstHorizontalPlateRoutes(
  point: GeoPoint,
  routes: HorizontalPlateRoute[],
  frame: RunwayFrame,
): HorizontalPlateSegmentAssessment | null {
  const assessments = routes
    .map((route): HorizontalPlateSegmentAssessment | null => {
      const surfaceAssessment = classifyPointAgainstProtectionSurfaces(
        point,
        route.protectionSurfaces ?? [],
      );
      if (!surfaceAssessment) return null;
      const closestPoint = projectPositionToRunwayFrame(
        frame,
        surfaceAssessment.closestPoint.lonDeg,
        surfaceAssessment.closestPoint.latDeg,
        surfaceAssessment.closestPoint.altM,
      );
      const verticalErrorM =
        surfaceAssessment.verticalDeltaFt === null
          ? null
          : surfaceAssessment.verticalDeltaFt * 0.3048;
      return {
        routeId: route.routeId,
        branchId: route.branchId,
        activeSegmentId: surfaceAssessment.surfaceId,
        segmentIndex: surfaceAssessment.segmentIndex,
        stationM: surfaceAssessment.stationNm * 1852,
        crossTrackErrorM: surfaceAssessment.lateralOffsetNm * 1852,
        verticalErrorM,
        containment: surfaceAssessment.containment,
        closestPoint,
        events: assessmentEvents(
          surfaceAssessment.containment,
          verticalErrorM,
          surfaceAssessment.verticalKind === "OCS",
        ),
        surfaceAssessment,
      };
    })
    .filter((assessment): assessment is HorizontalPlateSegmentAssessment => assessment !== null);

  const primaryAssessment = assessments
    .filter((assessment) => assessment.containment === "PRIMARY")
    .sort((left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM))[0];
  if (primaryAssessment) return primaryAssessment;

  const secondaryAssessment = assessments
    .filter((assessment) => assessment.containment === "SECONDARY")
    .sort((left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM))[0];
  if (secondaryAssessment) return secondaryAssessment;

  return assessments.sort(
    (left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM),
  )[0] ?? null;
}
