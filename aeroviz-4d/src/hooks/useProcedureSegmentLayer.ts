import { useEffect, useRef } from "react";
import { useApp } from "../context/AppContext";
import { loadProcedureRenderBundleData } from "../data/procedureRenderBundle";
import { getProcedureAnnotation } from "../data/procedureAnnotations";
import {
  ProcedureSceneEntityRegistry,
  renderVisibleProcedureBranches,
  syncProcedureSceneVisibility,
  type ProcedureSceneFlags,
  type ProcedureSceneRenderData,
} from "../scene/procedureSceneRenderer";
import {
  isProcedureAnnotationLabelId,
  isProcedureMeasurementEntityId,
  procedureBranchVisible,
  procedureEntityShow,
  renderProcedureBranchEntities,
} from "../scene/procedureSceneEntityRenderer";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { runwayMatchesSelection } from "../utils/runwayIdent";

/**
 * Mask a procedure-visibility map so branches that don't belong to the selected
 * landing runway read as hidden. Keeps the rendered geometry in lock-step with the
 * runway-scoped procedure list (so a branch left visible under "All runways" can't
 * keep drawing — orphaned — after a runway is selected). `null` selection = no mask.
 */
export function scopeVisibilityToRunway(
  visibility: Record<string, boolean>,
  selectedRunway: string | null,
  branchRunwayById: Map<string, string>,
): Record<string, boolean> {
  if (selectedRunway === null) return visibility;
  const masked: Record<string, boolean> = { ...visibility };
  for (const branchId of Object.keys(masked)) {
    const runwayId = branchRunwayById.get(branchId);
    if (runwayId !== undefined && !runwayMatchesSelection(selectedRunway, runwayId)) {
      masked[branchId] = false;
    }
  }
  return masked;
}

export function useProcedureSegmentLayer({ enabled = true }: { enabled?: boolean } = {}): void {
  const {
    viewer,
    layers,
    procedureVisibility,
    selectedRunway,
    activeAirportCode,
    procedureAnnotationEnabled,
    procedureWidthMeasurementEnabled,
    procedureDisplayLevel,
  } = useApp();
  const visibleRef = useRef(layers.procedures);
  const annotationVisibleRef = useRef(procedureAnnotationEnabled);
  const widthMeasurementVisibleRef = useRef(procedureWidthMeasurementEnabled);
  const displayLevelRef = useRef(procedureDisplayLevel);
  const procedureVisibilityRef = useRef(procedureVisibility);
  const selectedRunwayRef = useRef(selectedRunway);
  // Maps each branchId → its runway ident ("RW05L"), so visibility can be runway-scoped.
  const branchRunwayByIdRef = useRef<Map<string, string>>(new Map());
  const registryRef = useRef(new ProcedureSceneEntityRegistry());
  const renderDataRef = useRef<ProcedureSceneRenderData | null>(null);
  const proceduresRequested = enabled && layers.procedures;

  // A branch renders only if it is visible AND belongs to the selected runway.
  const branchVisibleScoped = (
    pkg: Parameters<typeof procedureBranchVisible>[0],
    branchId: string,
    visibility: Record<string, boolean>,
  ): boolean => {
    if (!procedureBranchVisible(pkg, branchId, visibility)) return false;
    const runwayId = branchRunwayByIdRef.current.get(branchId);
    return runwayId === undefined || runwayMatchesSelection(selectedRunwayRef.current, runwayId);
  };

  useEffect(() => {
    visibleRef.current = layers.procedures;
    annotationVisibleRef.current = procedureAnnotationEnabled;
    widthMeasurementVisibleRef.current = procedureWidthMeasurementEnabled;
    displayLevelRef.current = procedureDisplayLevel;
    procedureVisibilityRef.current = procedureVisibility;
    selectedRunwayRef.current = selectedRunway;

    if (!enabled || !isCesiumViewerUsable(viewer)) return;
    const flags: ProcedureSceneFlags = {
      proceduresVisible: layers.procedures,
      procedureVisibility: scopeVisibilityToRunway(
        procedureVisibility,
        selectedRunway,
        branchRunwayByIdRef.current,
      ),
      annotationVisible: procedureAnnotationEnabled,
      widthMeasurementVisible: procedureWidthMeasurementEnabled,
      displayLevel: procedureDisplayLevel,
    };

    syncProcedureSceneVisibility({
      viewer,
      registry: registryRef.current,
      flags,
      getAnnotation: getProcedureAnnotation,
      entityShow: procedureEntityShow,
      isAnnotationLabelId: isProcedureAnnotationLabelId,
      isMeasurementEntityId: isProcedureMeasurementEntityId,
    });

    if (renderDataRef.current) {
      renderVisibleProcedureBranches({
        viewer,
        registry: registryRef.current,
        renderData: renderDataRef.current,
        flags,
        branchVisible: branchVisibleScoped,
        renderBranch: renderProcedureBranchEntities,
      });
    }
  }, [
    enabled,
    viewer,
    layers.procedures,
    procedureVisibility,
    selectedRunway,
    procedureAnnotationEnabled,
    procedureWidthMeasurementEnabled,
    procedureDisplayLevel,
  ]);

  useEffect(() => {
    if (!proceduresRequested) {
      if (isCesiumViewerUsable(viewer)) {
        registryRef.current.removeAll(viewer);
      }
      renderDataRef.current = null;
      return;
    }
    if (!viewer || !activeAirportCode) return;

    let cancelled = false;
    registryRef.current.clear();
    renderDataRef.current = null;

    loadProcedureRenderBundleData(activeAirportCode)
      .then(({ renderBundles, packages }) => {
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        const packageById = new Map(packages.map((pkg) => [pkg.packageId, pkg]));
        const branchRunwayById = new Map<string, string>();
        for (const bundle of renderBundles) {
          for (const branchBundle of bundle.branchBundles) {
            if (branchBundle.runwayId) {
              branchRunwayById.set(branchBundle.branchId, branchBundle.runwayId);
            }
          }
        }
        branchRunwayByIdRef.current = branchRunwayById;
        renderDataRef.current = { renderBundles, packageById };
        renderVisibleProcedureBranches({
          viewer,
          registry: registryRef.current,
          renderData: renderDataRef.current,
          flags: {
            proceduresVisible: visibleRef.current,
            procedureVisibility: scopeVisibilityToRunway(
              procedureVisibilityRef.current,
              selectedRunwayRef.current,
              branchRunwayById,
            ),
            annotationVisible: annotationVisibleRef.current,
            widthMeasurementVisible: widthMeasurementVisibleRef.current,
            displayLevel: displayLevelRef.current,
          },
          branchVisible: branchVisibleScoped,
          renderBranch: renderProcedureBranchEntities,
        });
      })
      .catch((error) => {
        if (isMissingJsonAsset(error)) {
          console.warn(
            `[useProcedureSegmentLayer] procedure-details data for ${activeAirportCode} not found. ` +
              "Run: python aeroviz-4d/python/preprocess_procedures.py --airport <ICAO>",
          );
        } else {
          console.error("[useProcedureSegmentLayer]", error);
        }
      });

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        registryRef.current.removeAll(viewer);
      }
      registryRef.current.clear();
      renderDataRef.current = null;
    };
  }, [proceduresRequested, viewer, activeAirportCode]);
}
