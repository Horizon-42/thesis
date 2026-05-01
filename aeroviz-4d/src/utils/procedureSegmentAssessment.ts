import type {
  HorizontalPlateAssessmentSegment,
  HorizontalPlateRoute,
  RunwayProfilePoint,
} from "./runwayProfileGeometry";

export type HorizontalPlateContainment = "PRIMARY" | "SECONDARY" | "OUTSIDE";

export interface HorizontalPlateSegmentAssessment {
  routeId: string;
  branchId: string;
  activeSegmentId: string;
  segmentIndex: number;
  stationM: number;
  crossTrackErrorM: number;
  containment: HorizontalPlateContainment;
  closestPoint: RunwayProfilePoint;
}

interface CandidateAssessment extends HorizontalPlateSegmentAssessment {
  absoluteCrossTrackM: number;
  distanceToSegmentM: number;
}

function segmentIdFor(route: HorizontalPlateRoute, segmentIndex: number): string {
  return `${route.branchId}:profile-segment:${segmentIndex + 1}`;
}

function projectPointToSegment(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  route: HorizontalPlateRoute,
  segmentId: string,
  segmentPoints: RunwayProfilePoint[],
  primaryHalfWidthM: number,
  secondaryHalfWidthM: number | null,
  segmentIndex: number,
  stationOffsetM: number,
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
  const closestPoint = {
    xM: start.xM + alongRatio * dx,
    yM: start.yM + alongRatio * dy,
    zM: start.zM + alongRatio * (end.zM - start.zM),
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

  return {
    routeId: route.routeId,
    branchId: route.branchId,
    activeSegmentId: segmentId,
    segmentIndex,
    stationM: stationOffsetM + alongRatio * lengthM,
    crossTrackErrorM,
    containment,
    closestPoint,
    absoluteCrossTrackM,
    distanceToSegmentM,
  };
}

function projectPointToAssessmentSegment(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  route: HorizontalPlateRoute,
  assessmentSegment: HorizontalPlateAssessmentSegment,
  segmentOffset: number,
  segmentIndexOffset: number,
): HorizontalPlateSegmentAssessment | null {
  let stationOffsetM = 0;
  let best: CandidateAssessment | null = null;

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
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
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
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
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
