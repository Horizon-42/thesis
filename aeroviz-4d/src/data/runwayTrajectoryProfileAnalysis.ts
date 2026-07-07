import {
  classifyGeoPointAgainstHorizontalPlateRoutes,
  classifyPointAgainstHorizontalPlateRoutes,
  type HorizontalPlateContainment,
  type HorizontalPlateSegmentAssessment,
} from "../utils/procedureSegmentAssessment";
import type { GeoPoint } from "../utils/procedureGeoMath";
import type {
  HorizontalPlateRoute,
  RunwayFrame,
  RunwayProfilePoint,
} from "../utils/runwayProfileGeometry";

export interface ProfileAircraftSample extends RunwayProfilePoint {
  timeIso: string;
  segmentAssessment: HorizontalPlateSegmentAssessment;
}

export interface ProfileAircraftTrack {
  flightId: string;
  color: string;
  current: ProfileAircraftSample;
  trail: ProfileAircraftSample[];
  isSelected: boolean;
}

export interface SampledRunwayPoint extends RunwayProfilePoint {
  geoPosition: GeoPoint;
  timeIso: string;
}

export interface ProfileAircraftInput {
  flightId: string;
  /** The current-time sample. Non-null: the sampler drops entities not live now (returns
   *  null for them), so an input always has a current point. */
  current: SampledRunwayPoint;
  /** The whole time-ordered track INCLUDING the current sample (see sampleEntityTrack). */
  trail: SampledRunwayPoint[];
}

export function routeIsActive(
  route: HorizontalPlateRoute,
  procedureVisibility: Record<string, boolean>,
): boolean {
  return procedureVisibility[route.branchId] ?? route.defaultVisible;
}

export function activeHorizontalPlateRoutes(
  routes: HorizontalPlateRoute[],
  procedureVisibility: Record<string, boolean>,
): HorizontalPlateRoute[] {
  return routes.filter((route) => routeIsActive(route, procedureVisibility));
}

export function colorForFlightId(flightId: string): string {
  let hash = 0;
  for (let index = 0; index < flightId.length; index += 1) {
    hash = (hash * 31 + flightId.charCodeAt(index)) >>> 0;
  }
  return `hsl(${hash % 360} 72% 58%)`;
}

function classifyProfilePoint(
  point: SampledRunwayPoint,
  routes: HorizontalPlateRoute[],
  runwayFrame: RunwayFrame,
): HorizontalPlateSegmentAssessment | null {
  return (
    classifyGeoPointAgainstHorizontalPlateRoutes(
      point.geoPosition,
      routes,
      runwayFrame,
    ) ??
    classifyPointAgainstHorizontalPlateRoutes(
      point,
      routes,
    )
  );
}

function profileAircraftSample(
  point: SampledRunwayPoint,
  segmentAssessment: HorizontalPlateSegmentAssessment,
): ProfileAircraftSample {
  const { geoPosition: _geoPosition, ...profilePoint } = point;
  return {
    ...profilePoint,
    segmentAssessment,
  };
}

/** Classify one sampled point against the active procedure, or null if unclassifiable. */
export function classifyProfileSample(
  point: SampledRunwayPoint,
  activePlateRoutes: HorizontalPlateRoute[],
  runwayFrame: RunwayFrame,
): ProfileAircraftSample | null {
  const assessment = classifyProfilePoint(point, activePlateRoutes, runwayFrame);
  return assessment ? profileAircraftSample(point, assessment) : null;
}

/** Classify a whole track's samples (drops any unclassifiable). This is the expensive,
 *  currentTime-INDEPENDENT part — the caller caches it per entity so it runs once, not per
 *  clock tick (each point does a protection-surface + nearest-segment projection). */
export function classifyTrackSamples(
  points: SampledRunwayPoint[],
  activePlateRoutes: HorizontalPlateRoute[],
  runwayFrame: RunwayFrame,
): ProfileAircraftSample[] {
  return points
    .map((point) => classifyProfileSample(point, activePlateRoutes, runwayFrame))
    .filter((sample): sample is ProfileAircraftSample => sample !== null);
}

/**
 * Whether a track flies this procedure — some sample reaches PRIMARY containment. Used to keep
 * unrelated traffic out while still drawing the full approach (in AND out of the corridor) of
 * the aircraft that do fly it. A min-time solve that cuts the corner is out-of-corridor until
 * it joins the final approach; plotting only the in-corridor points left just that segment.
 */
export function trackEngagesProcedure(trail: ProfileAircraftSample[]): boolean {
  return trail.some((sample) => sample.segmentAssessment.containment === "PRIMARY");
}

/** Selected track first, then by flight id — the panel's stable draw/list order. */
export function sortProfileTracksBySelection(tracks: ProfileAircraftTrack[]): ProfileAircraftTrack[] {
  return [...tracks].sort((left, right) => {
    if (left.isSelected === right.isSelected) {
      return left.flightId.localeCompare(right.flightId);
    }
    return left.isSelected ? -1 : 1;
  });
}

/**
 * Convert sampled Cesium trajectory points into runway profile tracks (the batch form used in
 * tests; the hook assembles incrementally with a per-entity cache — see useRunwayTrajectoryProfile).
 */
export function buildProfileAircraftTracks(args: {
  aircraft: ProfileAircraftInput[];
  activePlateRoutes: HorizontalPlateRoute[];
  runwayFrame: RunwayFrame;
  selectedFlightId: string | null;
}): ProfileAircraftTrack[] {
  const { aircraft, activePlateRoutes, runwayFrame, selectedFlightId } = args;
  if (activePlateRoutes.length === 0) return [];

  const tracks = aircraft
    .map((input): ProfileAircraftTrack | null => {
      const current = classifyProfileSample(input.current, activePlateRoutes, runwayFrame);
      if (!current) return null;
      const trail = classifyTrackSamples(input.trail, activePlateRoutes, runwayFrame);
      // Engages if the trail OR the current point reaches the primary corridor (current is not
      // necessarily one of the trail's grid samples).
      if (!trackEngagesProcedure(trail) && current.segmentAssessment.containment !== "PRIMARY") {
        return null;
      }
      return {
        flightId: input.flightId,
        color: colorForFlightId(input.flightId),
        current,
        trail,
        isSelected: input.flightId === selectedFlightId,
      };
    })
    .filter((track): track is ProfileAircraftTrack => track !== null);
  return sortProfileTracksBySelection(tracks);
}

export interface ProfileTrackRun {
  /** The containment tier of this stretch. PRIMARY and SECONDARY are both CONTAINED (the
   *  panel draws them solid, primary brighter); only OUTSIDE is drawn dashed/dimmed — so a
   *  secondary-protection stretch never reads as an out-of-corridor breach. */
  containment: HorizontalPlateContainment;
  points: ProfileAircraftSample[];
}

/**
 * Split a track's ordered samples into contiguous runs of one containment tier, so the panel
 * can draw each stretch in its own style. Consecutive runs share their boundary point, so the
 * rendered polyline has no gap where the track crosses a containment edge.
 */
export function splitTrackByContainment(trail: ProfileAircraftSample[]): ProfileTrackRun[] {
  const runs: ProfileTrackRun[] = [];
  for (let index = 0; index < trail.length; index += 1) {
    const containment = trail[index].segmentAssessment.containment;
    const lastRun = runs[runs.length - 1];
    if (!lastRun || lastRun.containment !== containment) {
      const points: ProfileAircraftSample[] = [];
      if (lastRun) points.push(trail[index - 1]); // bridge the two runs — no visual gap
      runs.push({ containment, points });
    }
    runs[runs.length - 1].points.push(trail[index]);
  }
  return runs;
}
