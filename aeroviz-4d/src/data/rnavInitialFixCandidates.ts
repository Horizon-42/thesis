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

      const altitudeFt = altitudeFtForInitialFix(leg, fix);
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

function altitudeFtForInitialFix(
  leg: ProcedureDetailLeg,
  fix: ProcedureDetailFix,
): number | null {
  const geometryAltitudeFt = leg.constraints.geometryAltitudeFt;
  if (isFiniteNumber(geometryAltitudeFt)) return geometryAltitudeFt;

  const altitudeFt = leg.constraints.altitude?.valueFt;
  if (isFiniteNumber(altitudeFt)) return altitudeFt;

  if (isFiniteNumber(fix.elevationFt)) return fix.elevationFt;
  return null;
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
