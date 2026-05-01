import type {
  HorizontalPlateRoute,
  RunwayProfilePoint,
} from "./runwayProfileGeometry";

export type HorizontalPlateContainment = "PRIMARY" | "OUTSIDE";

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
  segmentIndex: number,
  stationOffsetM: number,
): CandidateAssessment | null {
  const start = route.points[segmentIndex];
  const end = route.points[segmentIndex + 1];
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

  return {
    routeId: route.routeId,
    branchId: route.branchId,
    activeSegmentId: segmentIdFor(route, segmentIndex),
    segmentIndex,
    stationM: stationOffsetM + alongRatio * lengthM,
    crossTrackErrorM,
    containment: distanceToSegmentM <= route.halfWidthM ? "PRIMARY" : "OUTSIDE",
    closestPoint,
    absoluteCrossTrackM,
    distanceToSegmentM,
  };
}

export function projectPointToHorizontalPlateRoute(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  route: HorizontalPlateRoute,
): HorizontalPlateSegmentAssessment | null {
  let stationOffsetM = 0;
  let best: CandidateAssessment | null = null;

  for (let segmentIndex = 0; segmentIndex < route.points.length - 1; segmentIndex += 1) {
    const candidate = projectPointToSegment(point, route, segmentIndex, stationOffsetM);
    if (candidate && (!best || candidate.distanceToSegmentM < best.distanceToSegmentM)) {
      best = candidate;
    }

    const start = route.points[segmentIndex];
    const end = route.points[segmentIndex + 1];
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

  return assessments.sort(
    (left, right) => Math.abs(left.crossTrackErrorM) - Math.abs(right.crossTrackErrorM),
  )[0] ?? null;
}
