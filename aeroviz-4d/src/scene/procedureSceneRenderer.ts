import type * as Cesium from "cesium";
import type { BranchGeometryBundle, ProcedureRenderBundle } from "../data/procedureRenderBundle";
import type { ProcedurePackage } from "../data/procedurePackage";
import type {
  ProcedureDisplayLevel,
  ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";

export interface ProcedureSceneRenderData {
  renderBundles: ProcedureRenderBundle[];
  packageById: Map<string, ProcedurePackage>;
}

export interface ProcedureSceneFlags {
  proceduresVisible: boolean;
  procedureVisibility: Record<string, boolean>;
  annotationVisible: boolean;
  widthMeasurementVisible: boolean;
  displayLevel: ProcedureDisplayLevel;
}

export interface ProcedureBranchRenderArgs {
  viewer: Cesium.Viewer;
  bundle: ProcedureRenderBundle;
  branchBundle: BranchGeometryBundle;
  branchIndex: number;
  pkg: ProcedurePackage | null;
  visible: boolean;
  annotationVisible: boolean;
  widthMeasurementVisible: boolean;
  displayLevel: ProcedureDisplayLevel;
}

export type ProcedureBranchRenderer = (args: ProcedureBranchRenderArgs) => string[];

export type ProcedureBranchVisible = (
  pkg: ProcedurePackage | null,
  branchId: string,
  procedureVisibility: Record<string, boolean>,
) => boolean;

export class ProcedureSceneEntityRegistry {
  private branchEntityIds: Record<string, string[]> = {};
  private allEntityIds: string[] = [];

  addBranchEntityIds(branchId: string, entityIds: string[]): void {
    this.allEntityIds.push(...entityIds);
    const existing = this.branchEntityIds[branchId] ?? [];
    this.branchEntityIds[branchId] = [...existing, ...entityIds];
  }

  idsByBranch(): Record<string, string[]> {
    return this.branchEntityIds;
  }

  hasBranch(branchId: string): boolean {
    return (this.branchEntityIds[branchId]?.length ?? 0) > 0;
  }

  removeAll(viewer: Cesium.Viewer): void {
    this.allEntityIds.forEach((id) => viewer.entities.removeById(id));
    this.clear();
  }

  clear(): void {
    this.branchEntityIds = {};
    this.allEntityIds = [];
  }
}

/**
 * Owns the scene-level bookkeeping for procedure entities.
 *
 * The renderer callback still creates the concrete Cesium entities. This module
 * owns the deeper scene concerns: branch entity registry, lazy branch rendering,
 * visibility refresh, and cleanup.
 */
export function renderVisibleProcedureBranches(args: {
  viewer: Cesium.Viewer;
  registry: ProcedureSceneEntityRegistry;
  renderData: ProcedureSceneRenderData;
  flags: ProcedureSceneFlags;
  branchVisible: ProcedureBranchVisible;
  renderBranch: ProcedureBranchRenderer;
}): void {
  const { viewer, registry, renderData, flags, branchVisible, renderBranch } = args;
  if (!flags.proceduresVisible) return;

  renderData.renderBundles.forEach((bundle) => {
    const pkg = renderData.packageById.get(bundle.packageId) ?? null;
    bundle.branchBundles.forEach((branchBundle, branchIndex) => {
      const visible = branchVisible(pkg, branchBundle.branchId, flags.procedureVisibility);
      if (!visible || registry.hasBranch(branchBundle.branchId)) return;
      registry.addBranchEntityIds(
        branchBundle.branchId,
        renderBranch({
          viewer,
          bundle,
          branchBundle,
          branchIndex,
          pkg,
          visible: true,
          annotationVisible: flags.annotationVisible,
          widthMeasurementVisible: flags.widthMeasurementVisible,
          displayLevel: flags.displayLevel,
        }),
      );
    });
  });
}

export function syncProcedureSceneVisibility(args: {
  viewer: Cesium.Viewer;
  registry: ProcedureSceneEntityRegistry;
  flags: ProcedureSceneFlags;
  getAnnotation: (entity: Cesium.Entity) => ProcedureEntityAnnotation | null;
  entityShow: (
    visible: boolean,
    annotation: ProcedureEntityAnnotation | null,
    displayLevel: ProcedureDisplayLevel,
    isAnnotationLabel?: boolean,
    annotationVisible?: boolean,
  ) => boolean;
  isAnnotationLabelId: (entityId: string) => boolean;
  isMeasurementEntityId: (entityId: string) => boolean;
}): void {
  const {
    viewer,
    registry,
    flags,
    getAnnotation,
    entityShow,
    isAnnotationLabelId,
    isMeasurementEntityId,
  } = args;

  Object.entries(registry.idsByBranch()).forEach(([branchId, entityIds]) => {
    const branchVisible = flags.procedureVisibility[branchId] ?? true;
    entityIds.forEach((entityId) => {
      const entity = viewer.entities.getById(entityId);
      if (!entity) return;

      const annotation = getAnnotation(entity);
      const baseVisible = flags.proceduresVisible && branchVisible;
      if (isMeasurementEntityId(entityId)) {
        entity.show =
          flags.widthMeasurementVisible &&
          entityShow(baseVisible, annotation, flags.displayLevel);
        return;
      }
      entity.show = entityShow(
        baseVisible,
        annotation,
        flags.displayLevel,
        isAnnotationLabelId(entityId),
        flags.annotationVisible,
      );
    });
  });
}
