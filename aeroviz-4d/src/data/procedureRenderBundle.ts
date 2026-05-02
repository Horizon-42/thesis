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
import {
  buildFinalApproachSurfaceStatus,
  buildLnavFinalOea,
  buildLnavVnavOcs,
  type FinalApproachSurfaceStatus,
  type LnavFinalOeaGeometry,
  type LnavVnavOcsGeometry,
} from "../utils/procedureSurfaceGeometry";
import {
  buildMissedCaCenterlines,
  buildMissedCaSegmentGeometry,
  buildMissedCaEndpoints,
  buildMissedCourseGuides,
  buildMissedSectionSurface,
  buildMissedTurnDebugPoint,
  type MissedCaCenterlineGeometry,
  type MissedCaEndpointGeometry,
  type MissedCourseGuideGeometry,
  type MissedSectionSurfaceGeometry,
  type MissedTurnDebugPointGeometry,
} from "../utils/procedureMissedGeometry";
import {
  buildInterSegmentTurnJunction,
  type InterSegmentTurnJunctionGeometry,
} from "../utils/procedureTurnGeometry";

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
  branchName: string;
  branchRole: ProcedurePackageBranch["branchRole"];
  runwayId: string | null;
  segmentBundles: ProcedureSegmentRenderBundle[];
  turnJunctions: InterSegmentTurnJunctionGeometry[];
}

export interface ProcedureSegmentRenderBundle {
  segment: ProcedureSegment;
  legs: ProcedurePackageLeg[];
  segmentGeometry: SegmentGeometryBundle;
  finalOea: LnavFinalOeaGeometry | null;
  lnavVnavOcs: LnavVnavOcsGeometry | null;
  finalSurfaceStatus: FinalApproachSurfaceStatus | null;
  alignedConnector: AlignedLnavConnectorGeometry | null;
  missedSectionSurface: MissedSectionSurfaceGeometry | null;
  missedCourseGuides: MissedCourseGuideGeometry[];
  missedCaEndpoints: MissedCaEndpointGeometry[];
  missedCaCenterlines: MissedCaCenterlineGeometry[];
  missedTurnDebugPoint: MissedTurnDebugPointGeometry | null;
  diagnostics: BuildDiagnostic[];
}

function isLnavFinal(segment: ProcedureSegment): boolean {
  return segment.segmentType === "FINAL_LNAV" || segment.segmentType === "FINAL_LNAV_VNAV";
}

function shouldBuildAlignedConnector(segment: ProcedureSegment): boolean {
  return segment.transitionRule?.kind === "INTERMEDIATE_TO_FINAL_LNAV";
}

function buildBranchTurnJunctions(
  branch: ProcedurePackageBranch,
  segmentBundles: ProcedureSegmentRenderBundle[],
): {
  turnJunctions: InterSegmentTurnJunctionGeometry[];
  diagnostics: BuildDiagnostic[];
} {
  const turnJunctions: InterSegmentTurnJunctionGeometry[] = [];
  const diagnostics: BuildDiagnostic[] = [];

  for (let index = 0; index < segmentBundles.length - 1; index += 1) {
    const fromBundle = segmentBundles[index];
    const toBundle = segmentBundles[index + 1];
    const junction = buildInterSegmentTurnJunction(
      branch.branchId,
      fromBundle.segment.segmentId,
      toBundle.segment.segmentId,
      fromBundle.segmentGeometry.centerline,
      toBundle.segmentGeometry.centerline,
      Math.max(fromBundle.segment.xttNm, toBundle.segment.xttNm) * 2,
      fromBundle.segment.secondaryEnabled || toBundle.segment.secondaryEnabled
        ? Math.max(fromBundle.segment.xttNm, toBundle.segment.xttNm) * 3
        : null,
    );

    if (!junction) continue;

    turnJunctions.push(junction);
    diagnostics.push({
      severity: "WARN",
      segmentId: fromBundle.segment.segmentId,
      code: "TURN_VISUAL_FILL_ONLY",
      message:
        `${branch.branchId}: visual fill was built between ${fromBundle.segment.segmentId} ` +
        `and ${toBundle.segment.segmentId}; this is not a compliant turn construction.`,
      sourceRefs: [
        ...fromBundle.segment.sourceRefs,
        ...toBundle.segment.sourceRefs,
      ],
    });
  }

  return { turnJunctions, diagnostics };
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
        const baseSegmentGeometry = buildSegmentGeometryBundle(segment, segmentLegs, fixes, ctx);

        const missedCourseGuideResult = buildMissedCourseGuides(segment, segmentLegs, fixes);
        const missedCaEndpointResult = buildMissedCaEndpoints(segment, segmentLegs, fixes);
        const missedCaCenterlines = buildMissedCaCenterlines(
          missedCaEndpointResult.geometries,
          { samplingStepNm: ctx.samplingStepNm },
        );
        const caSegmentGeometryResult = buildMissedCaSegmentGeometry(
          segment,
          segmentLegs,
          baseSegmentGeometry,
          missedCaCenterlines,
        );
        const segmentGeometry = caSegmentGeometryResult.geometry;
        const segmentDiagnostics: BuildDiagnostic[] = [
          ...segmentGeometry.diagnostics,
          ...missedCourseGuideResult.diagnostics,
          ...missedCaEndpointResult.diagnostics,
        ];

        const finalOeaResult =
          isLnavFinal(segment) && segmentGeometry.centerline.geoPositions.length >= 2
            ? buildLnavFinalOea(segment, segmentGeometry.centerline, {
                samplingStepNm: ctx.samplingStepNm,
              })
            : { geometry: null, diagnostics: [] };
        segmentDiagnostics.push(...finalOeaResult.diagnostics);
        const lnavVnavOcsResult = buildLnavVnavOcs(
          segment,
          segmentGeometry.centerline,
          finalOeaResult.geometry,
          { samplingStepNm: ctx.samplingStepNm },
        );
        segmentDiagnostics.push(...lnavVnavOcsResult.diagnostics);
        const finalSurfaceStatusResult = buildFinalApproachSurfaceStatus(
          segment,
          finalOeaResult.geometry,
          lnavVnavOcsResult.geometry,
        );
        segmentDiagnostics.push(...finalSurfaceStatusResult.diagnostics);

        const connectorResult =
          shouldBuildAlignedConnector(segment) && segmentGeometry.centerline.geoPositions.length >= 2
            ? buildAlignedLnavConnector(segment, segmentGeometry.centerline, {
                beforePfafNm: segment.transitionRule?.beforeNm,
                afterPfafNm: segment.transitionRule?.afterNm,
                samplingStepNm: ctx.samplingStepNm,
              })
            : { geometry: null, diagnostics: [] };
        segmentDiagnostics.push(...connectorResult.diagnostics);

        const missedSurfaceResult = buildMissedSectionSurface(segment, segmentGeometry);
        segmentDiagnostics.push(...missedSurfaceResult.diagnostics);
        const missedTurnDebugResult = buildMissedTurnDebugPoint(segment, segmentLegs, fixes);
        segmentDiagnostics.push(...missedTurnDebugResult.diagnostics);
        diagnostics.push(...segmentDiagnostics);

        return {
          segment,
          legs: segmentLegs,
          segmentGeometry,
          finalOea: finalOeaResult.geometry,
          lnavVnavOcs: lnavVnavOcsResult.geometry,
          finalSurfaceStatus: finalSurfaceStatusResult.status,
          alignedConnector: connectorResult.geometry,
          missedSectionSurface: missedSurfaceResult.geometry,
          missedCourseGuides: missedCourseGuideResult.geometries,
          missedCaEndpoints: missedCaEndpointResult.geometries,
          missedCaCenterlines,
          missedTurnDebugPoint: missedTurnDebugResult.geometry,
          diagnostics: segmentDiagnostics,
        };
      })
      .filter((bundle): bundle is ProcedureSegmentRenderBundle => bundle !== null);

    const branchTurnResult = buildBranchTurnJunctions(branch, segmentBundles);
    diagnostics.push(...branchTurnResult.diagnostics);

    return {
      branchId: branch.branchId,
      branchName: branch.branchName,
      branchRole: branch.branchRole,
      runwayId: branch.runwayId,
      segmentBundles,
      turnJunctions: branchTurnResult.turnJunctions,
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
