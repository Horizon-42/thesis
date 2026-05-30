import {
  classifyGeoPointAgainstHorizontalPlateRoutes,
  classifyPointAgainstHorizontalPlateRoutes,
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
  current: SampledRunwayPoint | null;
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
 * The sampling adapter provides time-stamped runway-frame points; this module
 * owns the domain decision of which routes are active, whether a point is inside
 * primary containment, and how current/trail samples are shaped for the panel.
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
      if (!input.current) return null;
      const currentAssessment = classifyProfilePoint(
        input.current,
        activePlateRoutes,
        runwayFrame,
      );
      if (!currentAssessment || currentAssessment.containment !== "PRIMARY") {
        return null;
      }

      const trail = input.trail
        .map((point): ProfileAircraftSample | null => {
          const segmentAssessment = classifyProfilePoint(
            point,
            activePlateRoutes,
            runwayFrame,
          );
          if (!segmentAssessment || segmentAssessment.containment !== "PRIMARY") return null;
          return profileAircraftSample(point, segmentAssessment);
        })
        .filter((sample): sample is ProfileAircraftSample => sample !== null);
      trail.push(profileAircraftSample(input.current, currentAssessment));

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
