import {
  procedureDetailsDocumentUrl,
  procedureDetailsIndexUrl,
  type ProcedureDetailDocument,
  type ProcedureDetailFix,
  type ProcedureDetailLeg,
} from "./procedureDetails";
import { fetchJson } from "../utils/fetchJson";

const FEET_TO_METRES = 0.3048;
const EARTH_RADIUS_M = 6_378_137;

export interface RnavInitialFixCandidate {
  key: string;
  runwayIdent: string;
  procedureUid: string;
  procedureIdent: string;
  chartName: string;
  branchId: string;
  branchIdent: string;
  fixId: string;
  fixIdent: string;
  nextFixId: string;
  nextFixIdent: string;
  lon: number;
  lat: number;
  altM: number;
  headingDeg: number;
}

export async function fetchRnavInitialFixCandidates(
  airportCode: string,
  runwayIdent: string,
): Promise<RnavInitialFixCandidate[]> {
  const normalizedRunwayIdent = normalizeRunwayIdent(runwayIdent);
  const index = await fetchJson<{
    runways: Array<{
      runwayIdent: string;
      procedures: Array<{ procedureUid: string }>;
    }>;
  }>(procedureDetailsIndexUrl(airportCode));
  const runway = index.runways.find(
    (candidate) => normalizeRunwayIdent(candidate.runwayIdent) === normalizedRunwayIdent,
  );
  if (!runway) return [];

  const procedureUids = [
    ...new Set(runway.procedures.map((procedure) => procedure.procedureUid)),
  ];
  const documents = await Promise.all(
    procedureUids.map((procedureUid) =>
      fetchJson<ProcedureDetailDocument>(
        procedureDetailsDocumentUrl(airportCode, procedureUid),
      )
    ),
  );

  return buildRnavInitialFixCandidates(documents, normalizedRunwayIdent);
}

export function buildRnavInitialFixCandidates(
  documents: ProcedureDetailDocument[],
  runwayIdent?: string,
): RnavInitialFixCandidate[] {
  const normalizedRunwayIdent = runwayIdent ? normalizeRunwayIdent(runwayIdent) : null;
  const candidates = documents.flatMap((document) => {
    if (!isRnavProcedure(document)) return [];
    if (
      normalizedRunwayIdent &&
      normalizeRunwayIdent(document.procedure.runwayIdent ?? document.runway.ident ?? "") !==
        normalizedRunwayIdent
    ) {
      return [];
    }
    return buildDocumentInitialFixCandidates(document);
  });

  return dedupeCandidates(candidates);
}

function buildDocumentInitialFixCandidates(
  document: ProcedureDetailDocument,
): RnavInitialFixCandidate[] {
  const fixLookup = new Map(document.fixes.map((fix) => [fix.fixId, fix]));

  return document.branches.flatMap((branch) => {
    const legs = [...branch.legs].sort((left, right) => left.sequence - right.sequence);
    return legs.flatMap((leg, index) => {
      const fix = fixLookup.get(leg.path.endFixRef);
      if (!fix || !isInitialFixLeg(leg, fix) || !fix.position) return [];

      const nextLeg = findNextLegFromFix(legs, index + 1, fix.fixId);
      if (!nextLeg) return [];
      const nextFix = fixLookup.get(nextLeg.path.endFixRef);
      if (!nextFix?.position) return [];

      // An IF without its own published crossing altitude (feeder/transition
      // IF such as KRDU R32 CONCA/SINNO) is given an altitude DERIVED from the
      // nearest fix(es) on the branch that do have one -- interpolated by
      // along-track distance when bracketed, else the single available
      // neighbour.  Only skip if no neighbour has a published altitude.
      // See docs/33-cifp-transition-altitude-misparse-postmortem.md.
      const altitudeFt = derivedInitialFixAltitudeFt(legs, index, fixLookup, fix.position);
      if (altitudeFt === null) return [];

      const candidateWithoutKey = {
        runwayIdent: normalizeRunwayIdent(
          document.procedure.runwayIdent ?? document.runway.ident ?? "",
        ),
        procedureUid: document.procedureUid,
        procedureIdent: document.procedure.procedureIdent,
        chartName: document.procedure.chartName,
        branchId: branch.branchId,
        branchIdent: branch.branchIdent,
        fixId: fix.fixId,
        fixIdent: fix.ident,
        nextFixId: nextFix.fixId,
        nextFixIdent: nextFix.ident,
        lon: fix.position.lon,
        lat: fix.position.lat,
        altM: altitudeFt * FEET_TO_METRES,
        headingDeg: simulatorPsiDeg(
          fix.position.lon,
          fix.position.lat,
          nextFix.position.lon,
          nextFix.position.lat,
        ),
      };
      return [{
        key: rnavInitialFixCandidateKey(candidateWithoutKey),
        ...candidateWithoutKey,
      }];
    });
  });
}

export function rnavInitialFixCandidateKey(
  candidate: Omit<RnavInitialFixCandidate, "key">,
): string {
  return [
    candidate.procedureUid,
    candidate.branchId,
    candidate.fixId,
    candidate.nextFixId,
    candidate.altM,
  ].join("|");
}

function isRnavProcedure(document: ProcedureDetailDocument): boolean {
  const family = document.procedure.procedureFamily.toUpperCase();
  const chartName = document.procedure.chartName.toUpperCase();
  return family.includes("RNAV") || chartName.includes("RNAV");
}

function isInitialFixLeg(
  leg: ProcedureDetailLeg,
  fix: ProcedureDetailFix,
): boolean {
  return normalizeRole(leg.roleAtEnd) === "IF" ||
    leg.path.pathTerminator.toUpperCase() === "IF" ||
    fix.roleHints.some((hint) => normalizeRole(hint) === "IF");
}

function findNextLegFromFix(
  legs: ProcedureDetailLeg[],
  startIndex: number,
  fixId: string,
): ProcedureDetailLeg | null {
  return legs.slice(startIndex).find((leg) => leg.path.startFixRef === fixId) ??
    legs[startIndex] ??
    null;
}

function altitudeFtForInitialFix(leg: ProcedureDetailLeg): number | null {
  // A leg's *own* published crossing altitude: a finite, positive value
  // from the Altitude 1/2 constraint.  We never use the transition altitude
  // (a procedure-wide constant) or the fix's terrain elevation here.
  // See docs/33-cifp-transition-altitude-misparse-postmortem.md.
  const geometryAltitudeFt = leg.constraints.geometryAltitudeFt;
  if (isFiniteNumber(geometryAltitudeFt) && geometryAltitudeFt > 0) {
    return geometryAltitudeFt;
  }

  const altitudeFt = leg.constraints.altitude?.valueFt;
  if (isFiniteNumber(altitudeFt) && altitudeFt > 0) {
    return altitudeFt;
  }

  return null;
}

interface AltitudeAnchor {
  altitudeFt: number;
  position: { lon: number; lat: number };
}

function nearestAnchor(
  legs: ProcedureDetailLeg[],
  fixLookup: Map<string, ProcedureDetailFix>,
  start: number,
  step: 1 | -1,
): AltitudeAnchor | null {
  for (let i = start; i >= 0 && i < legs.length; i += step) {
    const altitudeFt = altitudeFtForInitialFix(legs[i]);
    const fix = fixLookup.get(legs[i].path.endFixRef);
    if (altitudeFt !== null && fix?.position) {
      return { altitudeFt, position: fix.position };
    }
  }
  return null;
}

function derivedInitialFixAltitudeFt(
  legs: ProcedureDetailLeg[],
  index: number,
  fixLookup: Map<string, ProcedureDetailFix>,
  ifPosition: { lon: number; lat: number },
): number | null {
  // Prefer the IF's own published altitude.
  const own = altitudeFtForInitialFix(legs[index]);
  if (own !== null) return own;

  // Otherwise derive from the nearest published neighbours on the branch.
  const upstream = nearestAnchor(legs, fixLookup, index - 1, -1);
  const downstream = nearestAnchor(legs, fixLookup, index + 1, 1);

  if (upstream && downstream) {
    // Linear interpolation by along-track distance between the two anchors.
    const dUp = distanceM(ifPosition, upstream.position);
    const dDown = distanceM(ifPosition, downstream.position);
    const total = dUp + dDown;
    if (total <= 0) return upstream.altitudeFt;
    return upstream.altitudeFt +
      (downstream.altitudeFt - upstream.altitudeFt) * (dUp / total);
  }
  // A leading IF (the common case) only has a downstream anchor.
  return downstream?.altitudeFt ?? upstream?.altitudeFt ?? null;
}

function distanceM(
  a: { lon: number; lat: number },
  b: { lon: number; lat: number },
): number {
  const meanLat = toRadians((a.lat + b.lat) / 2);
  const east = toRadians(b.lon - a.lon) * EARTH_RADIUS_M * Math.cos(meanLat);
  const north = toRadians(b.lat - a.lat) * EARTH_RADIUS_M;
  return Math.hypot(east, north);
}

function dedupeCandidates(
  candidates: RnavInitialFixCandidate[],
): RnavInitialFixCandidate[] {
  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    if (seen.has(candidate.key)) return false;
    seen.add(candidate.key);
    return true;
  });
}

function normalizeRunwayIdent(ident: string): string {
  return ident.trim().toUpperCase();
}

function normalizeRole(role: string): string {
  return role.trim().toUpperCase();
}

function simulatorPsiDeg(
  fromLon: number,
  fromLat: number,
  toLon: number,
  toLat: number,
): number {
  const meanLat = toRadians((fromLat + toLat) / 2);
  const east = toRadians(toLon - fromLon) * EARTH_RADIUS_M * Math.cos(meanLat);
  const north = toRadians(toLat - fromLat) * EARTH_RADIUS_M;
  return normalizeDegrees(toDegrees(Math.atan2(north, east)));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeDegrees(value: number): number {
  return ((value % 360) + 360) % 360;
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function toDegrees(value: number): number {
  return (value * 180) / Math.PI;
}
