import {
  attachRenderBundleAssessmentSegments,
  buildHorizontalPlateRoutes,
  type HorizontalPlateRoute,
  type RunwayFrame,
} from "../utils/runwayProfileGeometry";
import { buildProcedureRoutes } from "./procedureRoutes";
import type { ProcedureRenderBundleData } from "./procedureRenderBundle";

export interface ProcedureProfileProjection {
  plateRoutes: HorizontalPlateRoute[];
  sourceCycle: string | null;
}

/**
 * Builds runway-profile procedure geometry from the canonical render data seam.
 *
 * The hook that samples aircraft should not need to know that procedure-detail
 * documents still supply route display points while render bundles supply
 * assessment segments and protection surfaces. Keeping the stitch here gives
 * profile callers one Interface for RNAV procedure context.
 */
export function buildProcedureProfileProjection(
  data: ProcedureRenderBundleData,
  runwayFrame: RunwayFrame,
  runwayIdent: string,
): ProcedureProfileProjection {
  const routeViewModels = buildProcedureRoutes(data.documents);
  const plateRoutes = attachRenderBundleAssessmentSegments(
    buildHorizontalPlateRoutes(routeViewModels, runwayFrame, runwayIdent),
    data.renderBundles,
    runwayFrame,
    runwayIdent,
  );

  return {
    plateRoutes,
    sourceCycle: data.index.sourceCycle ?? null,
  };
}
