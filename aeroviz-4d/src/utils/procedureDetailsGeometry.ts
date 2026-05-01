import type {
  ProcedureDetailDocument,
  ProcedureDetailFix,
  ProcedureDetailBranch,
} from "../data/procedureDetails";
import {
  buildProcedureRoutes,
  procedureRouteBranchLookup,
  procedureRouteFixLookup,
} from "../data/procedureRoutes";

const EARTH_RADIUS_M = 6_378_137;

export interface ProcedureChartPoint {
  fixId: string;
  ident: string;
  role: string;
  branchId: string;
  branchIdent: string;
  branchRole: string;
  routeId: string;
  branchKey: string;
  transitionIdent: string | null;
  procedureIdent: string;
  procedureName: string;
  procedureFamily: string;
  lon: number;
  lat: number;
  altitudeFt: number | null;
  geometryAltitudeFt: number;
  altM: number;
  sequence: number;
  legType: string;
  sourceLine: number;
  timeSeconds: number;
  xM: number;
  yM: number;
  distanceM: number;
}

export interface ProcedureBranchPolyline {
  branchId: string;
  branchIdent: string;
  branchRole: string;
  routeId: string;
  branchKey: string;
  transitionIdent: string | null;
  procedureIdent: string;
  procedureName: string;
  procedureFamily: string;
  defaultVisible: boolean;
  warnings: string[];
  points: ProcedureChartPoint[];
}

export interface ProcedureRunwayMarker {
  centerX: number;
  centerY: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function pointToEastNorth(
  lon: number,
  lat: number,
  originLon: number,
  originLat: number,
): { east: number; north: number } {
  const dLon = toRadians(lon - originLon);
  const dLat = toRadians(lat - originLat);
  const meanLat = toRadians((lat + originLat) / 2);
  return {
    east: dLon * EARTH_RADIUS_M * Math.cos(meanLat),
    north: dLat * EARTH_RADIUS_M,
  };
}

export function fixLookup(document: ProcedureDetailDocument): Map<string, ProcedureDetailFix> {
  return procedureRouteFixLookup(document);
}

export function buildProcedureBranchPolylines(
  document: ProcedureDetailDocument,
): ProcedureBranchPolyline[] {
  const routes = buildProcedureRoutes([document]);
  const origin =
    document.runway.threshold ??
    document.fixes.find((fix) => fix.position)?.position ?? {
      lon: 0,
      lat: 0,
    };

  return routes
    .map((route) => {
      const points = route.points.map((point) => {
        const local = pointToEastNorth(point.lon, point.lat, origin.lon, origin.lat);

        return {
          fixId: point.fixId,
          ident: point.fixIdent,
          role: point.role,
          branchId: route.branchId,
          branchIdent: route.branchIdent,
          branchRole: route.branchType,
          routeId: route.routeId,
          branchKey: route.branchKey,
          transitionIdent: route.transitionIdent,
          procedureIdent: route.procedureIdent,
          procedureName: route.procedureName,
          procedureFamily: route.procedureFamily,
          lon: point.lon,
          lat: point.lat,
          altitudeFt: point.geometryAltitudeFt,
          geometryAltitudeFt: point.geometryAltitudeFt,
          altM: point.altM,
          sequence: point.sequence,
          legType: point.legType,
          sourceLine: point.sourceLine,
          timeSeconds: point.timeSeconds,
          xM: local.east,
          yM: local.north,
          distanceM: point.distanceFromStartM,
        };
      });

      return {
        branchId: route.branchId,
        branchIdent: route.branchIdent,
        branchRole: route.branchType,
        routeId: route.routeId,
        branchKey: route.branchKey,
        transitionIdent: route.transitionIdent,
        procedureIdent: route.procedureIdent,
        procedureName: route.procedureName,
        procedureFamily: route.procedureFamily,
        defaultVisible: route.defaultVisible,
        warnings: route.warnings,
        points,
      };
    })
    .sort((left, right) => {
      if (left.branchRole === right.branchRole) {
        return left.branchIdent.localeCompare(right.branchIdent);
      }
      return left.branchRole === "final" ? -1 : 1;
    });
}

export function buildRunwayMarker(
  document: ProcedureDetailDocument,
  polylines: ProcedureBranchPolyline[],
): ProcedureRunwayMarker | null {
  const thresholdFixId = document.runway.landingThresholdFixRef;
  if (!thresholdFixId) return null;

  const finalBranch =
    polylines.find((branch) => branch.branchRole === "final") ?? polylines[0];
  if (!finalBranch) return null;

  const thresholdIndex = finalBranch.points.findIndex((point) => point.fixId === thresholdFixId);
  if (thresholdIndex <= 0) return null;

  const thresholdPoint = finalBranch.points[thresholdIndex];
  const previousPoint = finalBranch.points[thresholdIndex - 1];
  const dx = thresholdPoint.xM - previousPoint.xM;
  const dy = thresholdPoint.yM - previousPoint.yM;
  const length = Math.hypot(dx, dy);
  if (length === 0) return null;

  const px = -dy / length;
  const py = dx / length;
  const halfWidthM = 350;
  return {
    centerX: thresholdPoint.xM,
    centerY: thresholdPoint.yM,
    x1: thresholdPoint.xM + px * halfWidthM,
    y1: thresholdPoint.yM + py * halfWidthM,
    x2: thresholdPoint.xM - px * halfWidthM,
    y2: thresholdPoint.yM - py * halfWidthM,
  };
}

export function findFix(
  document: ProcedureDetailDocument | null,
  fixId: string | null,
): ProcedureDetailFix | null {
  if (!document || !fixId) return null;
  return fixLookup(document).get(fixId) ?? null;
}

export function procedureBranchForFix(
  document: ProcedureDetailDocument | null,
  fixId: string | null,
): ProcedureDetailBranch[] {
  if (!document || !fixId) return [];
  const branchById = procedureRouteBranchLookup(document);
  return buildProcedureRoutes([document])
    .filter((route) => route.points.some((point) => point.fixId === fixId))
    .map((route) => branchById.get(route.branchId))
    .filter((branch): branch is ProcedureDetailBranch => branch !== undefined);
}

export function nmFromMeters(valueM: number): number {
  return valueM / 1852;
}
