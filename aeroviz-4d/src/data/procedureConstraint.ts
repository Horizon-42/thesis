/**
 * Canonical, front↔back-common procedure constraint.
 *
 * This is the single, minimal representation of "an RNAV approach as a
 * constraint" that is meant to cross the frontend/backend boundary unchanged:
 * the frontend derives it from a {@link ProcedureDetailDocument} and ships it
 * in the optimizer request; the Python backend parses the SAME shape
 * (`aeroviz_backend/procedure_constraint.py`) so the optimizer and the
 * dynamics model can read the published path, altitude windows and speed
 * limits directly.
 *
 * It deliberately holds only what a path/altitude/speed constraint needs — an
 * ordered list of waypoints (position + altitude window + speed) plus the
 * final-approach course and glidepath — and it reuses the ONE canonical
 * altitude type ({@link AltitudeConstraint}) and the ONE canonical CIFP→
 * constraint conversion, so there is no second interpretation of the same
 * coded data. The richer visualization models (`ProcedureRouteViewModel`,
 * `ProcedurePackage`, the render bundle) are derived views for rendering; this
 * is the contract for computation.
 */
import type { ProcedureDetailDocument, ProcedureDetailLeg } from "./procedureDetails";
import type { AltitudeConstraint } from "./procedurePackage";
import { altitudeConstraintReferenceFt } from "./altitudeConstraints";
import {
  buildProcedureRoutes,
  type ProcedureRoutePoint,
  type ProcedureRouteViewModel,
} from "./procedureRoutes";
import {
  EARTH_RADIUS_M,
  FEET_TO_METERS,
  toDegrees,
  toRadians,
} from "../utils/procedureGeoMath";

export interface ProcedureConstraintWaypoint {
  /** Stable fix id (matches `ProcedureDetailFix.fixId`). */
  fixId: string;
  /** Published fix name, e.g. `WEPAS`. */
  ident: string;
  /** Role at this waypoint, e.g. `IF`, `FAF`, `MAPt`, `Route`. */
  role: string;
  /** ARINC 424 path terminator of the leg ending here, e.g. `IF`, `TF`. */
  legType: string;
  lonDeg: number;
  latDeg: number;
  /** Canonical altitude window (null when the leg carries no coded altitude). */
  altitude: AltitudeConstraint | null;
  /** Single reference altitude derived from {@link altitude} (optimizer convenience). */
  altitudeRefFt: number | null;
  /** Rendered/repaired altitude used for geometry (may be interpolated). */
  geometryAltFt: number | null;
  /** Maximum coded speed at the leg, KIAS (null when uncoded). */
  speedMaxKt: number | null;
  /** Cumulative along-track distance from the first waypoint, metres. */
  distanceFromStartM: number;
}

export interface ProcedureConstraint {
  procedureUid: string;
  airportIcao: string;
  runwayIdent: string | null;
  branchId: string;
  /** Final-approach course (deg true) derived from the last leg into the runway. */
  approachCourseDeg: number | null;
  /** Source-coded final glidepath, when published. */
  glidepath: { angleDeg: number; tchFt: number | null } | null;
  /** A procedure-level nominal speed hint (KIAS) for the model. */
  nominalSpeedKt: number;
  /** Ordered waypoints from the branch entry down to the runway threshold. */
  waypoints: ProcedureConstraintWaypoint[];
}

export interface BuildProcedureConstraintOptions {
  /** Which branch (entry transition / final) to anchor the constraint on. */
  branchId?: string;
}

function normalizeDegrees(value: number): number {
  return ((value % 360) + 360) % 360;
}

/** Bearing (deg true) from one geographic point to another, flat-earth at scale. */
function bearingDeg(
  from: { lonDeg: number; latDeg: number },
  to: { lonDeg: number; latDeg: number },
): number {
  const meanLat = toRadians((from.latDeg + to.latDeg) / 2);
  const east = toRadians(to.lonDeg - from.lonDeg) * EARTH_RADIUS_M * Math.cos(meanLat);
  const north = toRadians(to.latDeg - from.latDeg) * EARTH_RADIUS_M;
  return normalizeDegrees(toDegrees(Math.atan2(east, north)));
}

/**
 * Map each fix to the first coded speed limit on a leg that ends there. Speed
 * is uncoded in the current dataset (always null), so this is forward-looking;
 * it keeps the canonical waypoint honest the moment the parser starts emitting
 * `constraints.speedKt`.
 */
function speedByFix(document: ProcedureDetailDocument): Map<string, number> {
  const speeds = new Map<string, number>();
  const legs: ProcedureDetailLeg[] = document.branches.flatMap((branch) => branch.legs);
  for (const leg of legs) {
    const speed = leg.constraints.speedKt;
    if (typeof speed === "number" && Number.isFinite(speed) && !speeds.has(leg.path.endFixRef)) {
      speeds.set(leg.path.endFixRef, speed);
    }
  }
  return speeds;
}

function finalGlidepath(
  document: ProcedureDetailDocument,
): { angleDeg: number; tchFt: number | null } | null {
  const profile = document.verticalProfiles.find(
    (candidate) => typeof candidate.glidepathAngleDeg === "number",
  );
  if (!profile || typeof profile.glidepathAngleDeg !== "number") return null;
  return {
    angleDeg: profile.glidepathAngleDeg,
    tchFt: typeof profile.thresholdCrossingHeightFt === "number"
      ? profile.thresholdCrossingHeightFt
      : null,
  };
}

function pickRoute(
  routes: ProcedureRouteViewModel[],
  branchId: string | undefined,
): ProcedureRouteViewModel | null {
  if (routes.length === 0) return null;
  if (branchId) {
    const match = routes.find((route) => route.branchId === branchId);
    if (match) return match;
  }
  return routes.find((route) => route.defaultVisible) ?? routes[0];
}

function waypointFromRoutePoint(
  point: ProcedureRoutePoint,
  speeds: Map<string, number>,
): ProcedureConstraintWaypoint {
  return {
    fixId: point.fixId,
    ident: point.fixIdent,
    role: point.role,
    legType: point.legType,
    lonDeg: point.lon,
    latDeg: point.lat,
    altitude: point.altitudeConstraint,
    altitudeRefFt: altitudeConstraintReferenceFt(point.altitudeConstraint),
    geometryAltFt: point.geometryAltitudeFt,
    speedMaxKt: speeds.get(point.fixId) ?? null,
    distanceFromStartM: point.distanceFromStartM,
  };
}

/**
 * Build the canonical procedure constraint for one approach branch of a
 * detail document. Reuses the route builder (correct branch continuation +
 * the canonical CIFP→altitude conversion), then attaches the final-approach
 * course, the coded glidepath and per-fix speed.
 */
export function buildProcedureConstraint(
  document: ProcedureDetailDocument,
  options: BuildProcedureConstraintOptions = {},
): ProcedureConstraint | null {
  const route = pickRoute(buildProcedureRoutes([document]), options.branchId);
  if (!route || route.points.length < 2) return null;

  const speeds = speedByFix(document);
  const waypoints = route.points.map((point) => waypointFromRoutePoint(point, speeds));

  const penultimate = waypoints[waypoints.length - 2];
  const last = waypoints[waypoints.length - 1];

  return {
    procedureUid: document.procedureUid,
    airportIcao: document.airport.icao.toUpperCase(),
    runwayIdent: document.procedure.runwayIdent ?? document.runway.ident,
    branchId: route.branchId,
    approachCourseDeg: bearingDeg(penultimate, last),
    glidepath: finalGlidepath(document),
    nominalSpeedKt: document.displayHints.nominalSpeedKt,
    waypoints,
  };
}

/** Altitude window reference of each waypoint in metres (optimizer convenience). */
export function procedureConstraintAltitudesM(
  constraint: ProcedureConstraint,
): Array<number | null> {
  return constraint.waypoints.map((waypoint) =>
    waypoint.altitudeRefFt === null ? null : waypoint.altitudeRefFt * FEET_TO_METERS,
  );
}
