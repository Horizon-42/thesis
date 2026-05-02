import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { resolvePickedProcedureAnnotation } from "../data/procedureAnnotations";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

const HIGHLIGHT_ENTITY_ID = "procedure-annotation-selected-highlight";

function removeHighlight(viewer: Cesium.Viewer): void {
  viewer.entities.removeById(HIGHLIGHT_ENTITY_ID);
}

function valueAtCurrentTime<T>(value: T | { getValue: (time?: unknown) => T }, viewer: Cesium.Viewer): T | null {
  if (!value) return null;
  if (typeof value === "object" && "getValue" in value && typeof value.getValue === "function") {
    return value.getValue(viewer.clock?.currentTime);
  }
  return value as T;
}

function entityRepresentativePosition(
  viewer: Cesium.Viewer,
  entityId: string,
): Cesium.Cartesian3 | null {
  const entity = viewer.entities.getById(entityId);
  if (!entity) return null;

  const position = valueAtCurrentTime<Cesium.Cartesian3>(
    entity.position as Cesium.Cartesian3 | { getValue: (time?: unknown) => Cesium.Cartesian3 },
    viewer,
  );
  if (position) return position;

  const positions = valueAtCurrentTime<Cesium.Cartesian3[]>(
    entity.polyline?.positions as Cesium.Cartesian3[] | { getValue: (time?: unknown) => Cesium.Cartesian3[] },
    viewer,
  );
  if (positions && positions.length > 0) return positions[Math.floor((positions.length - 1) / 2)];

  return null;
}

function addHighlight(
  viewer: Cesium.Viewer,
  entityId: string,
  clickPosition?: Cesium.Cartesian2,
): void {
  removeHighlight(viewer);

  const pickedPosition =
    clickPosition && viewer.scene.pickPositionSupported
      ? viewer.scene.pickPosition(clickPosition)
      : undefined;
  const position = pickedPosition ?? entityRepresentativePosition(viewer, entityId);
  if (!position) return;

  viewer.entities.add({
    id: HIGHLIGHT_ENTITY_ID,
    name: "Selected procedure annotation",
    position,
    point: {
      pixelSize: 18,
      color: Cesium.Color.WHITE.withAlpha(0.9),
      outlineColor: Cesium.Color.ORANGE.withAlpha(0.95),
      outlineWidth: 4,
    },
  });
}

export function useProcedureAnnotationPicking({ enabled = true }: { enabled?: boolean } = {}): void {
  const {
    viewer,
    layers,
    procedureAnnotationEnabled,
    setSelectedProcedureAnnotation,
  } = useApp();
  const selectedEntityIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled || !viewer || !procedureAnnotationEnabled || !layers.procedures) {
      if (viewer && isCesiumViewerUsable(viewer)) removeHighlight(viewer);
      selectedEntityIdRef.current = null;
      setSelectedProcedureAnnotation(null);
      return;
    }
    if (!isCesiumViewerUsable(viewer)) return;

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((event: { position: Cesium.Cartesian2 }) => {
      if (!isCesiumViewerUsable(viewer)) return;
      const picked = viewer.scene.pick(event.position);
      const annotation = resolvePickedProcedureAnnotation(picked);
      if (!annotation) return;

      selectedEntityIdRef.current = annotation.entityId;
      setSelectedProcedureAnnotation(annotation);
      addHighlight(viewer, annotation.entityId, event.position);
    }, Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    return () => {
      handler.destroy();
      removeHighlight(viewer);
      selectedEntityIdRef.current = null;
    };
  }, [
    enabled,
    viewer,
    layers.procedures,
    procedureAnnotationEnabled,
    setSelectedProcedureAnnotation,
  ]);
}
