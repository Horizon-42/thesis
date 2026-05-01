import type {
  BuildDiagnostic,
  ProcedurePackage,
  ProcedurePackageBranch,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "./procedurePackage";
import {
  type ProcedureDetailDocument,
  type ProcedureDetailsIndexManifest,
  procedureDetailsDocumentUrl,
  procedureDetailsIndexUrl,
} from "./procedureDetails";
import { normalizeProcedurePackage } from "./procedurePackageAdapter";
import { fetchJson } from "../utils/fetchJson";
import {
  DEFAULT_GEOMETRY_BUILD_CONTEXT,
  buildSegmentGeometryBundle,
  type GeometryBuildContext,
  type SegmentGeometryBundle,
} from "../utils/procedureSegmentGeometry";
import {
  buildAlignedLnavConnector,
  type AlignedLnavConnectorGeometry,
} from "../utils/procedureConnectorGeometry";
import { buildLnavFinalOea, type LnavFinalOeaGeometry } from "../utils/procedureSurfaceGeometry";

export interface ProcedureRenderBundle {
  packageId: string;
  procedureId: string;
  procedureName: string;
  airportId: string;
  branchBundles: BranchGeometryBundle[];
  diagnostics: BuildDiagnostic[];
}

export interface ProcedureRenderBundleData {
  index: ProcedureDetailsIndexManifest;
  documents: ProcedureDetailDocument[];
  packages: ProcedurePackage[];
  renderBundles: ProcedureRenderBundle[];
}

export interface BranchGeometryBundle {
  branchId: string;
  routeId: string;
  branchName: string;
  branchRole: ProcedurePackageBranch["branchRole"];
  runwayId: string | null;
  segmentBundles: ProcedureSegmentRenderBundle[];
}

export interface ProcedureSegmentRenderBundle {
  segment: ProcedureSegment;
  legs: ProcedurePackageLeg[];
  segmentGeometry: SegmentGeometryBundle;
  finalOea: LnavFinalOeaGeometry | null;
  alignedConnector: AlignedLnavConnectorGeometry | null;
  diagnostics: BuildDiagnostic[];
}

function isLnavFinal(segment: ProcedureSegment): boolean {
  return segment.segmentType === "FINAL_LNAV" || segment.segmentType === "FINAL_LNAV_VNAV";
}

function shouldBuildAlignedConnector(segment: ProcedureSegment): boolean {
  return segment.transitionRule?.kind === "INTERMEDIATE_TO_FINAL_LNAV";
}

function routeIdFor(pkg: ProcedurePackage, branch: ProcedurePackageBranch): string {
  return `${pkg.airportId.toUpperCase()}-${pkg.procedureId.toUpperCase()}-${branch.legacy.branchKey.toUpperCase()}`;
}

export function buildProcedureRenderBundle(
  pkg: ProcedurePackage,
  ctx: GeometryBuildContext = DEFAULT_GEOMETRY_BUILD_CONTEXT,
): ProcedureRenderBundle {
  const diagnostics: BuildDiagnostic[] = [...pkg.diagnostics];
  const fixes = new Map(pkg.sharedFixes.map((fix) => [fix.fixId, fix]));
  const legsById = new Map(pkg.legs.map((leg) => [leg.legId, leg]));
  const segmentsById = new Map(pkg.segments.map((segment) => [segment.segmentId, segment]));

  const branchBundles = pkg.branches.map((branch): BranchGeometryBundle => {
    const segmentBundles = branch.segmentIds
      .map((segmentId): ProcedureSegmentRenderBundle | null => {
        const segment = segmentsById.get(segmentId);
        if (!segment) return null;

        const segmentLegs = segment.legIds
          .map((legId) => legsById.get(legId))
          .filter((leg): leg is ProcedurePackageLeg => leg !== undefined);
        const segmentGeometry = buildSegmentGeometryBundle(segment, segmentLegs, fixes, ctx);
        const segmentDiagnostics: BuildDiagnostic[] = [...segmentGeometry.diagnostics];

        const finalOeaResult =
          isLnavFinal(segment) && segmentGeometry.centerline.geoPositions.length >= 2
            ? buildLnavFinalOea(segment, segmentGeometry.centerline, {
                samplingStepNm: ctx.samplingStepNm,
              })
            : { geometry: null, diagnostics: [] };
        segmentDiagnostics.push(...finalOeaResult.diagnostics);

        const connectorResult =
          shouldBuildAlignedConnector(segment) && segmentGeometry.centerline.geoPositions.length >= 2
            ? buildAlignedLnavConnector(segment, segmentGeometry.centerline, {
                beforePfafNm: segment.transitionRule?.beforeNm,
                afterPfafNm: segment.transitionRule?.afterNm,
                samplingStepNm: ctx.samplingStepNm,
              })
            : { geometry: null, diagnostics: [] };
        segmentDiagnostics.push(...connectorResult.diagnostics);
        diagnostics.push(...segmentDiagnostics);

        return {
          segment,
          legs: segmentLegs,
          segmentGeometry,
          finalOea: finalOeaResult.geometry,
          alignedConnector: connectorResult.geometry,
          diagnostics: segmentDiagnostics,
        };
      })
      .filter((bundle): bundle is ProcedureSegmentRenderBundle => bundle !== null);

    return {
      branchId: branch.branchId,
      routeId: routeIdFor(pkg, branch),
      branchName: branch.branchName,
      branchRole: branch.branchRole,
      runwayId: branch.runwayId,
      segmentBundles,
    };
  });

  return {
    packageId: pkg.packageId,
    procedureId: pkg.procedureId,
    procedureName: pkg.procedureName,
    airportId: pkg.airportId,
    branchBundles,
    diagnostics,
  };
}

export async function loadProcedureRenderBundleData(
  airportCode: string,
  ctx: GeometryBuildContext = DEFAULT_GEOMETRY_BUILD_CONTEXT,
): Promise<ProcedureRenderBundleData> {
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
  const packages = documents.map(normalizeProcedurePackage);

  return {
    index,
    documents,
    packages,
    renderBundles: packages.map((pkg) => buildProcedureRenderBundle(pkg, ctx)),
  };
}
