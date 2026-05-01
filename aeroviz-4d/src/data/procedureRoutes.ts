import {
  procedureDetailsDocumentUrl,
  procedureDetailsIndexUrl,
  type ProcedureDetailBranch,
  type ProcedureDetailDocument,
  type ProcedureDetailFix,
  type ProcedureDetailLeg,
  type ProcedureDetailsIndexManifest,
} from "./procedureDetails";
import { fetchJson } from "../utils/fetchJson";

const FEET_TO_METERS = 0.3048;
const NM_TO_METERS = 1852;
const EARTH_RADIUS_M = 6_378_137;

export interface ProcedureRoutePoint {
  fixId: string;
  fixIdent: string;
  sequence: number;
  legType: string;
  role: string;
  lon: number;
  lat: number;
  altitudeFt: number | null;
  geometryAltitudeFt: number;
  altM: number;
  distanceFromStartM: number;
  timeSeconds: number;
  sourceLine: number;
}

export interface ProcedureRouteViewModel {
  routeId: string;
  airport: string;
  procedureUid: string;
  procedureType: string;
  procedureIdent: string;
  procedureName: string;
  procedureFamily: string;
  procedureVariant: string | null;
  runwayIdent: string | null;
  branchId: string;
  branchKey: string;
  branchIdent: string;
  branchProcedureType: string | null;
  transitionIdent: string | null;
  branchType: string;
  defaultVisible: boolean;
  warnings: string[];
  nominalSpeedKt: number;
  tunnel: {
    lateralHalfWidthNm: number;
    verticalHalfHeightFt: number;
    sampleSpacingM: number;
    mode?: string;
  };
  points: ProcedureRoutePoint[];
}

export interface ProcedureRouteData {
  index: ProcedureDetailsIndexManifest;
  documents: ProcedureDetailDocument[];
  routes: ProcedureRouteViewModel[];
}

interface RawRoutePoint {
  fixId: string;
  fixIdent: string;
  sequence: number;
  legType: string;
  role: string;
  lon: number;
  lat: number;
  altitudeFt: number | null;
  geometryAltitudeFt: number | null;
  sourceLine: number;
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function distanceM(left: { lon: number; lat: number }, right: { lon: number; lat: number }): number {
  const dLon = toRadians(right.lon - left.lon);
  const dLat = toRadians(right.lat - left.lat);
  const meanLat = toRadians((left.lat + right.lat) / 2);
  const east = dLon * EARTH_RADIUS_M * Math.cos(meanLat);
  const north = dLat * EARTH_RADIUS_M;
  return Math.hypot(east, north);
}

export function procedureRouteFixLookup(document: ProcedureDetailDocument): Map<string, ProcedureDetailFix> {
  return new Map(document.fixes.map((fix) => [fix.fixId, fix]));
}

export function procedureRouteBranchLookup(
  document: ProcedureDetailDocument,
): Map<string, ProcedureDetailBranch> {
  return new Map(document.branches.map((branch) => [branch.branchId, branch]));
}

function branchKey(branch: ProcedureDetailBranch): string {
  return (branch.branchKey ?? branch.branchIdent).toUpperCase();
}

function routeIdFor(document: ProcedureDetailDocument, branch: ProcedureDetailBranch): string {
  return `${document.airport.icao.toUpperCase()}-${document.procedure.procedureIdent.toUpperCase()}-${branchKey(branch)}`;
}

function legIsRenderable(leg: ProcedureDetailLeg, fix: ProcedureDetailFix | undefined): boolean {
  return leg.quality.renderedInPlanView === true && fix?.position !== null && fix?.position !== undefined;
}

function altitudeConstraintFt(leg: ProcedureDetailLeg): number | null {
  return leg.constraints.altitude?.valueFt ?? null;
}

function geometryAltitudeFt(leg: ProcedureDetailLeg, fix: ProcedureDetailFix | undefined): number | null {
  return (
    leg.constraints.geometryAltitudeFt ??
    leg.constraints.altitude?.valueFt ??
    fix?.elevationFt ??
    null
  );
}

function ownBranchPoints(
  branch: ProcedureDetailBranch,
  fixById: Map<string, ProcedureDetailFix>,
): RawRoutePoint[] {
  const points = branch.legs
    .map((leg) => {
      const fix = fixById.get(leg.path.endFixRef);
      if (!legIsRenderable(leg, fix) || !fix?.position) return null;
      return {
        fixId: fix.fixId,
        fixIdent: fix.ident,
        sequence: leg.sequence,
        legType: leg.path.pathTerminator,
        role: leg.roleAtEnd,
        lon: Number(fix.position.lon.toFixed(8)),
        lat: Number(fix.position.lat.toFixed(8)),
        altitudeFt: altitudeConstraintFt(leg),
        geometryAltitudeFt: geometryAltitudeFt(leg, fix),
        sourceLine: leg.quality.sourceLine,
      };
    })
    .filter((point): point is RawRoutePoint => point !== null);

  return dedupeSequential(points, (point) => point.fixId);
}

function rawBranchPoints(
  branch: ProcedureDetailBranch,
  branchById: Map<string, ProcedureDetailBranch>,
  fixById: Map<string, ProcedureDetailFix>,
  cache: Map<string, RawRoutePoint[]>,
): RawRoutePoint[] {
  const cached = cache.get(branch.branchId);
  if (cached) return cached;

  let points = ownBranchPoints(branch, fixById);
  const continuationBranch =
    branch.continuesWithBranchId === null ? undefined : branchById.get(branch.continuesWithBranchId);

  if (continuationBranch) {
    const continuation = rawBranchPoints(continuationBranch, branchById, fixById, cache);
    let startIndex = 0;
    if (branch.mergeFixRef) {
      const mergeIndex = continuation.findIndex((point) => point.fixId === branch.mergeFixRef);
      startIndex = mergeIndex >= 0 ? mergeIndex : 0;
    }
    const continuationPoints = continuation.slice(startIndex);
    if (
      points.length > 0 &&
      continuationPoints.length > 0 &&
      points[points.length - 1].fixId === continuationPoints[0].fixId
    ) {
      points = [...points, ...continuationPoints.slice(1)];
    } else {
      points = [...points, ...continuationPoints];
    }
  }

  cache.set(branch.branchId, points);
  return points;
}

function dedupeSequential<T>(items: T[], keyFn: (item: T) => string): T[] {
  return items.filter((item, index) => index === 0 || keyFn(item) !== keyFn(items[index - 1]));
}

function fillMissingGeometryAltitudes(points: RawRoutePoint[]): {
  altitudesFt: number[];
  warnings: string[];
} {
  const known = points
    .map((point, index) => ({ altitudeFt: point.geometryAltitudeFt, index, point }))
    .filter((entry): entry is { altitudeFt: number; index: number; point: RawRoutePoint } => (
      entry.altitudeFt !== null && Number.isFinite(entry.altitudeFt)
    ));

  if (known.length === 0) {
    return {
      altitudesFt: points.map(() => 0),
      warnings: points.map((point) => (
        `${point.fixIdent}: altitude missing in procedure constraints; geometry altitude defaulted to 0 ft.`
      )),
    };
  }

  const warnings: string[] = [];
  const altitudesFt = points.map((point, index) => {
    if (point.geometryAltitudeFt !== null && Number.isFinite(point.geometryAltitudeFt)) {
      return point.geometryAltitudeFt;
    }

    const previous = [...known].reverse().find((entry) => entry.index < index);
    const next = known.find((entry) => entry.index > index);
    if (previous && next) {
      const ratio = (index - previous.index) / (next.index - previous.index);
      const interpolated = previous.altitudeFt + (next.altitudeFt - previous.altitudeFt) * ratio;
      warnings.push(
        `${point.fixIdent}: altitude missing in procedure constraints; geometry altitude interpolated from neighboring constraints.`,
      );
      return interpolated;
    }

    const nearest = previous ?? next;
    warnings.push(
      `${point.fixIdent}: altitude missing in procedure constraints; geometry altitude filled from nearest available constraint.`,
    );
    return nearest?.altitudeFt ?? 0;
  });

  return { altitudesFt, warnings };
}

export function buildProcedureRoutes(
  documents: ProcedureDetailDocument[],
): ProcedureRouteViewModel[] {
  return documents.flatMap((document) => {
    const fixById = procedureRouteFixLookup(document);
    const branchById = procedureRouteBranchLookup(document);
    const rawCache = new Map<string, RawRoutePoint[]>();
    const nominalSpeedKt = document.displayHints.nominalSpeedKt;
    const speedMps = (nominalSpeedKt * NM_TO_METERS) / 3600;

    return document.branches
      .map((branch) => {
        const rawPoints = rawBranchPoints(branch, branchById, fixById, rawCache);
        const altitudeRepair = fillMissingGeometryAltitudes(rawPoints);
        let cumulativeDistanceM = 0;
        let elapsedSeconds = 0;
        let previousPoint: RawRoutePoint | null = null;

        const points = rawPoints.map((point, index) => {
          if (previousPoint) {
            const legDistanceM = distanceM(previousPoint, point);
            cumulativeDistanceM += legDistanceM;
            elapsedSeconds += speedMps > 0 ? legDistanceM / speedMps : 0;
          }
          previousPoint = point;
          const repairedAltitudeFt = altitudeRepair.altitudesFt[index] ?? 0;

          return {
            fixId: point.fixId,
            fixIdent: point.fixIdent,
            sequence: point.sequence,
            legType: point.legType,
            role: point.role,
            lon: point.lon,
            lat: point.lat,
            altitudeFt: point.altitudeFt,
            geometryAltitudeFt: repairedAltitudeFt,
            altM: Number((repairedAltitudeFt * FEET_TO_METERS).toFixed(2)),
            distanceFromStartM: Number(cumulativeDistanceM.toFixed(1)),
            timeSeconds: Number(elapsedSeconds.toFixed(1)),
            sourceLine: point.sourceLine,
          };
        });

        return {
          routeId: routeIdFor(document, branch),
          airport: document.airport.icao.toUpperCase(),
          procedureUid: document.procedureUid,
          procedureType: document.procedure.procedureType,
          procedureIdent: document.procedure.procedureIdent,
          procedureName: document.procedure.chartName,
          procedureFamily: document.procedure.procedureFamily,
          procedureVariant: document.procedure.variant,
          runwayIdent: document.procedure.runwayIdent,
          branchId: branch.branchId,
          branchKey: branchKey(branch),
          branchIdent: branch.branchIdent,
          branchProcedureType: branch.procedureType ?? null,
          transitionIdent: branch.transitionIdent ?? null,
          branchType: branch.branchRole,
          defaultVisible: branch.defaultVisible,
          warnings: [...branch.warnings, ...altitudeRepair.warnings],
          nominalSpeedKt,
          tunnel: document.displayHints.tunnelDefaults,
          points,
        };
      })
      .filter((route) => route.points.length >= 2);
  });
}

export async function loadProcedureRouteData(airportCode: string): Promise<ProcedureRouteData> {
  const index = await fetchJson<ProcedureDetailsIndexManifest>(procedureDetailsIndexUrl(airportCode));
  const procedureUids = index.runways.flatMap((runway) =>
    runway.procedures.map((procedure) => procedure.procedureUid),
  );
  const uniqueProcedureUids = [...new Set(procedureUids)];
  const documents = await Promise.all(
    uniqueProcedureUids.map((procedureUid) =>
      fetchJson<ProcedureDetailDocument>(procedureDetailsDocumentUrl(airportCode, procedureUid)),
    ),
  );

  return {
    index,
    documents,
    routes: buildProcedureRoutes(documents),
  };
}
