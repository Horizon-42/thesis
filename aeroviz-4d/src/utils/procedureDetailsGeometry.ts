import type {
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailFix,
  ProcedureDetailLeg,
} from "../data/procedureDetails";

const EARTH_RADIUS_M = 6_378_137;

export interface ProcedureChartPoint {
  fixId: string;
  ident: string;
  role: string;
  branchId: string;
  branchIdent: string;
  branchRole: string;
  lon: number;
  lat: number;
  altitudeFt: number | null;
  xM: number;
  yM: number;
  distanceM: number;
}

export interface ProcedureBranchPolyline {
  branchId: string;
  branchIdent: string;
  branchRole: string;
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

interface RawBranchPoint {
  fixId: string;
  ident: string;
  role: string;
  lon: number;
  lat: number;
  altitudeFt: number | null;
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function fillMissingNumbers(values: Array<number | null>): number[] {
  const known = values
    .map((value, index) => ({ value, index }))
    .filter((entry): entry is { value: number; index: number } => entry.value !== null);

  if (known.length === 0) return values.map(() => 0);

  return values.map((value, index) => {
    if (value !== null) return value;

    const previous = [...known].reverse().find((entry) => entry.index < index);
    const next = known.find((entry) => entry.index > index);
    if (previous && next) {
      const ratio = (index - previous.index) / (next.index - previous.index);
      return previous.value + (next.value - previous.value) * ratio;
    }
    return previous?.value ?? next?.value ?? 0;
  });
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

function distance2d(
  left: { east: number; north: number },
  right: { east: number; north: number },
): number {
  return Math.hypot(right.east - left.east, right.north - left.north);
}

function dedupeSequential<T>(items: T[], keyFn: (item: T) => string): T[] {
  return items.filter((item, index) => {
    if (index === 0) return true;
    return keyFn(item) !== keyFn(items[index - 1]);
  });
}

export function fixLookup(document: ProcedureDetailDocument): Map<string, ProcedureDetailFix> {
  return new Map(document.fixes.map((fix) => [fix.fixId, fix]));
}

function altitudeForLeg(leg: ProcedureDetailLeg, fix: ProcedureDetailFix | undefined): number | null {
  return (
    leg.constraints.geometryAltitudeFt ??
    leg.constraints.altitude?.valueFt ??
    fix?.elevationFt ??
    null
  );
}

function positionedFixPoint(
  leg: ProcedureDetailLeg,
  fix: ProcedureDetailFix | undefined,
): RawBranchPoint | null {
  if (!fix?.position) return null;
  return {
    fixId: fix.fixId,
    ident: fix.ident,
    role: leg.roleAtEnd,
    lon: fix.position.lon,
    lat: fix.position.lat,
    altitudeFt: altitudeForLeg(leg, fix),
  };
}

function ownBranchPoints(
  branch: ProcedureDetailBranch,
  fixById: Map<string, ProcedureDetailFix>,
): RawBranchPoint[] {
  return dedupeSequential(
    branch.legs
      .map((leg) => positionedFixPoint(leg, fixById.get(leg.path.endFixRef)))
      .filter((point): point is RawBranchPoint => point !== null),
    (point) => point.fixId,
  );
}

function rawBranchPoints(
  branch: ProcedureDetailBranch,
  branchById: Map<string, ProcedureDetailBranch>,
  fixById: Map<string, ProcedureDetailFix>,
  cache: Map<string, RawBranchPoint[]>,
): RawBranchPoint[] {
  const cached = cache.get(branch.branchId);
  if (cached) return cached;

  const ownPoints = ownBranchPoints(branch, fixById);
  const continuationBranch =
    branch.continuesWithBranchId === null ? undefined : branchById.get(branch.continuesWithBranchId);

  let combined = ownPoints;
  if (continuationBranch) {
    const continuation = rawBranchPoints(continuationBranch, branchById, fixById, cache);
    let continuationStart = 0;
    if (branch.mergeFixRef) {
      const mergeIndex = continuation.findIndex((point) => point.fixId === branch.mergeFixRef);
      continuationStart = mergeIndex >= 0 ? mergeIndex : 0;
    }
    const continuationPoints = continuation.slice(continuationStart);
    if (
      combined.length > 0 &&
      continuationPoints.length > 0 &&
      combined[combined.length - 1].fixId === continuationPoints[0].fixId
    ) {
      combined = [...combined, ...continuationPoints.slice(1)];
    } else {
      combined = [...combined, ...continuationPoints];
    }
  }

  cache.set(branch.branchId, combined);
  return combined;
}

export function buildProcedureBranchPolylines(
  document: ProcedureDetailDocument,
): ProcedureBranchPolyline[] {
  const fixById = fixLookup(document);
  const branchById = new Map(document.branches.map((branch) => [branch.branchId, branch]));
  const origin =
    document.runway.threshold ??
    document.fixes.find((fix) => fix.position)?.position ?? {
      lon: 0,
      lat: 0,
    };
  const rawCache = new Map<string, RawBranchPoint[]>();

  return document.branches
    .map((branch) => {
      const rawPoints = rawBranchPoints(branch, branchById, fixById, rawCache);
      const repairedAltitudesFt = fillMissingNumbers(rawPoints.map((point) => point.altitudeFt));

      let cumulativeDistanceM = 0;
      let previousLocal: { east: number; north: number } | null = null;
      const points = rawPoints.map((point, index) => {
        const local = pointToEastNorth(point.lon, point.lat, origin.lon, origin.lat);
        if (previousLocal) {
          cumulativeDistanceM += distance2d(previousLocal, local);
        }
        previousLocal = local;

        return {
          fixId: point.fixId,
          ident: point.ident,
          role: point.role,
          branchId: branch.branchId,
          branchIdent: branch.branchIdent,
          branchRole: branch.branchRole,
          lon: point.lon,
          lat: point.lat,
          altitudeFt: repairedAltitudesFt[index] ?? 0,
          xM: local.east,
          yM: local.north,
          distanceM: cumulativeDistanceM,
        };
      });

      return {
        branchId: branch.branchId,
        branchIdent: branch.branchIdent,
        branchRole: branch.branchRole,
        defaultVisible: branch.defaultVisible,
        warnings: branch.warnings,
        points: dedupeSequential(points, (point) => point.fixId),
      };
    })
    .filter((branch) => branch.points.length >= 1)
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
  return document.branches.filter((branch) =>
    branch.legs.some((leg) => leg.path.endFixRef === fixId),
  );
}

export function nmFromMeters(valueM: number): number {
  return valueM / 1852;
}
