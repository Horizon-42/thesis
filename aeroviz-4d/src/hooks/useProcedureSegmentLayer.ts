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

export function useProcedureSegmentLayer({ enabled = true }: { enabled?: boolean } = {}): void {
  const {
    viewer,
    layers,
    procedureVisibility,
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
  const registryRef = useRef(new ProcedureSceneEntityRegistry());
  const renderDataRef = useRef<ProcedureSceneRenderData | null>(null);
  const proceduresRequested = enabled && layers.procedures;

  useEffect(() => {
    visibleRef.current = layers.procedures;
    annotationVisibleRef.current = procedureAnnotationEnabled;
    widthMeasurementVisibleRef.current = procedureWidthMeasurementEnabled;
    displayLevelRef.current = procedureDisplayLevel;
    procedureVisibilityRef.current = procedureVisibility;

    if (!enabled || !isCesiumViewerUsable(viewer)) return;
    const flags: ProcedureSceneFlags = {
      proceduresVisible: layers.procedures,
      procedureVisibility,
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
        branchVisible: procedureBranchVisible,
        renderBranch: renderProcedureBranchEntities,
      });
    }
  }, [
    enabled,
    viewer,
    layers.procedures,
    procedureVisibility,
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
        renderDataRef.current = { renderBundles, packageById };
        renderVisibleProcedureBranches({
          viewer,
          registry: registryRef.current,
          renderData: renderDataRef.current,
          flags: {
            proceduresVisible: visibleRef.current,
            procedureVisibility: procedureVisibilityRef.current,
            annotationVisible: annotationVisibleRef.current,
            widthMeasurementVisible: widthMeasurementVisibleRef.current,
            displayLevel: displayLevelRef.current,
          },
          branchVisible: procedureBranchVisible,
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
