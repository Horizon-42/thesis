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

/**
 * Converts sampled Cesium trajectory points into runway profile tracks.
 *
 * The sampling adapter provides time-stamped runway-frame points; this module owns the
 * domain decisions: which routes are active, each point's containment tier, and which
 * aircraft to plot.
 *
 * The WHOLE track is kept — points in AND out of the procedure corridor, each tagged with
 * its containment so the panel can style the out-of-corridor stretches (a min-time solve
 * that cuts the corner is out-of-corridor until it joins the final approach; showing only
 * the in-corridor points left just that final segment). An aircraft is plotted only if it
 * ENGAGES the procedure — some sample reaches PRIMARY containment — which keeps unrelated
 * traffic out while still drawing the full approach of the ones that fly it.
 */
export function buildProfileAircraftTracks(args: {
  aircraft: ProfileAircraftInput[];
  activePlateRoutes: HorizontalPlateRoute[];
  runwayFrame: RunwayFrame;
  selectedFlightId: string | null;
}): ProfileAircraftTrack[] {
  const { aircraft, activePlateRoutes, runwayFrame, selectedFlightId } = args;
  if (activePlateRoutes.length === 0) return [];

  return aircraft
    .map((input): ProfileAircraftTrack | null => {
      const currentAssessment = classifyProfilePoint(
        input.current,
        activePlateRoutes,
        runwayFrame,
      );
      if (!currentAssessment) return null;

      const trail = input.trail
        .map((point): ProfileAircraftSample | null => {
          const segmentAssessment = classifyProfilePoint(point, activePlateRoutes, runwayFrame);
          return segmentAssessment ? profileAircraftSample(point, segmentAssessment) : null;
        })
        .filter((sample): sample is ProfileAircraftSample => sample !== null);

      // Plot the aircraft only if it actually flies this procedure at some point along the
      // sampled track — not merely because it is nearby. `current` is part of `trail`
      // (sampleEntityTrack injects it), so scanning the trail already covers the current point.
      const engagesProcedure = trail.some(
        (sample) => sample.segmentAssessment.containment === "PRIMARY",
      );
      if (!engagesProcedure) return null;

      return {
        flightId: input.flightId,
        color: colorForFlightId(input.flightId),
        current: profileAircraftSample(input.current, currentAssessment),
        trail,
        isSelected: input.flightId === selectedFlightId,
      };
    })
    .filter((track): track is ProfileAircraftTrack => track !== null)
    .sort((left, right) => {
      if (left.isSelected === right.isSelected) {
        return left.flightId.localeCompare(right.flightId);
      }
      return left.isSelected ? -1 : 1;
    });
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
