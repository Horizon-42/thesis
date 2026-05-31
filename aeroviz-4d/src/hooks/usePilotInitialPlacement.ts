import { useEffect, useRef, type MutableRefObject } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import type { PilotResetState } from "../pilot/pilotClient";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

export interface PilotInitialPlacementPosition {
  lon: number;
  lat: number;
}

interface UsePilotInitialPlacementOptions {
  enabled: boolean;
  previewVisible: boolean;
  initialState: PilotResetState;
  onPositionChange: (position: PilotInitialPlacementPosition) => void;
  onFinish: () => void;
  onCancel: () => void;
}

const PLACEMENT_AIRCRAFT_ID = "pilot-initial-placement-aircraft";
const PLACEMENT_GROUND_POINT_ID = "pilot-initial-placement-ground-point";

export function usePilotInitialPlacement({
  enabled,
  previewVisible,
  initialState,
  onPositionChange,
  onFinish,
  onCancel,
}: UsePilotInitialPlacementOptions): void {
  const { viewer, setSelectedFlightId } = useApp();
  const aircraftRef = useRef<Cesium.Entity | null>(null);
  const groundPointRef = useRef<Cesium.Entity | null>(null);
  const isDraggingRef = useRef(false);
  const callbacksRef = useRef({ onPositionChange, onFinish, onCancel });

  useEffect(() => {
    callbacksRef.current = { onPositionChange, onFinish, onCancel };
  }, [onCancel, onFinish, onPositionChange]);

  useEffect(() => {
    if (!previewVisible || !isCesiumViewerUsable(viewer)) {
      removePlacementPreview(viewer, aircraftRef, groundPointRef);
      return;
    }

    updatePlacementPreview(viewer, initialState, aircraftRef, groundPointRef);
    viewer.scene.requestRender();
  }, [initialState, previewVisible, viewer]);

  useEffect(() => {
    if (!enabled || !isCesiumViewerUsable(viewer)) return;

    setSelectedFlightId(null);
    if (viewer.trackedEntity) {
      viewer.trackedEntity = undefined;
    }

    const { scene } = viewer;
    const canvas = scene.canvas;
    const controller = scene.screenSpaceCameraController;
    const previousControllerState = {
      enableLook: controller.enableLook,
      enableRotate: controller.enableRotate,
      enableTilt: controller.enableTilt,
      enableTranslate: controller.enableTranslate,
    };
    const previousCursor = canvas.style.cursor;

    controller.enableLook = false;
    controller.enableRotate = false;
    controller.enableTilt = false;
    controller.enableTranslate = false;
    canvas.style.cursor = "crosshair";

    const handler = new Cesium.ScreenSpaceEventHandler(canvas);

    const updateFromScreen = (position: Cesium.Cartesian2 | undefined): boolean => {
      if (!position || !isCesiumViewerUsable(viewer)) return false;

      const nextPosition = pickPilotPlacementPosition(viewer, position);
      if (!nextPosition) return false;

      callbacksRef.current.onPositionChange(nextPosition);
      return true;
    };

    handler.setInputAction((event: { position: Cesium.Cartesian2 }) => {
      isDraggingRef.current = true;
      updateFromScreen(event.position);
    }, Cesium.ScreenSpaceEventType.LEFT_DOWN);

    handler.setInputAction((movement: { endPosition: Cesium.Cartesian2 }) => {
      if (!isDraggingRef.current) return;
      updateFromScreen(movement.endPosition);
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    handler.setInputAction((event: { position: Cesium.Cartesian2 }) => {
      if (!isDraggingRef.current) return;
      updateFromScreen(event.position);
      isDraggingRef.current = false;
      callbacksRef.current.onFinish();
    }, Cesium.ScreenSpaceEventType.LEFT_UP);

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        callbacksRef.current.onCancel();
      } else if (event.key === "Enter") {
        event.preventDefault();
        callbacksRef.current.onFinish();
      }
    };

    window.addEventListener("keydown", onKeyDown);

    return () => {
      isDraggingRef.current = false;
      handler.destroy();
      window.removeEventListener("keydown", onKeyDown);

      if (!viewer.isDestroyed()) {
        controller.enableLook = previousControllerState.enableLook;
        controller.enableRotate = previousControllerState.enableRotate;
        controller.enableTilt = previousControllerState.enableTilt;
        controller.enableTranslate = previousControllerState.enableTranslate;
        canvas.style.cursor = previousCursor;
      }
    };
  }, [enabled, setSelectedFlightId, viewer]);

  useEffect(() => {
    return () => {
      removePlacementPreview(viewer, aircraftRef, groundPointRef);
    };
  }, [viewer]);
}

export function pickPilotPlacementPosition(
  viewer: Cesium.Viewer,
  screenPosition: Cesium.Cartesian2,
): PilotInitialPlacementPosition | null {
  const { scene } = viewer;
  let cartesian: Cesium.Cartesian3 | undefined;

  if (scene.pickPositionSupported) {
    try {
      cartesian = scene.pickPosition(screenPosition);
    } catch {
      cartesian = undefined;
    }
  }

  if (!cartesian) {
    const pickRay = viewer.camera.getPickRay(screenPosition);
    if (pickRay) {
      cartesian = scene.globe.pick(pickRay, scene);
    }
  }

  if (!cartesian) {
    cartesian = viewer.camera.pickEllipsoid(screenPosition, scene.globe.ellipsoid);
  }

  if (!cartesian) return null;

  const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
  return {
    lon: Cesium.Math.toDegrees(cartographic.longitude),
    lat: Cesium.Math.toDegrees(cartographic.latitude),
  };
}

function updatePlacementPreview(
  viewer: Cesium.Viewer,
  state: PilotResetState,
  aircraftRef: MutableRefObject<Cesium.Entity | null>,
  groundPointRef: MutableRefObject<Cesium.Entity | null>,
): void {
  const aircraftPosition = Cesium.Cartesian3.fromDegrees(state.lon, state.lat, state.altM);
  const orientation = Cesium.Transforms.headingPitchRollQuaternion(
    aircraftPosition,
    new Cesium.HeadingPitchRoll(
      Cesium.Math.toRadians(state.headingDeg),
      Cesium.Math.toRadians(state.flightPathDeg),
      0,
    ),
  );

  if (!aircraftRef.current) {
    aircraftRef.current = viewer.entities.add({
      id: PLACEMENT_AIRCRAFT_ID,
      name: "Pilot Initial Aircraft",
      position: new Cesium.ConstantPositionProperty(aircraftPosition),
      orientation: new Cesium.ConstantProperty(orientation),
      model: {
        uri: "/models/aircraft.glb",
        scale: 3.0,
        minimumPixelSize: 42,
        maximumScale: 20_000,
        color: Cesium.Color.fromCssColorString("#facc15").withAlpha(0.86),
        colorBlendMode: Cesium.ColorBlendMode.MIX,
        colorBlendAmount: 0.34,
        runAnimations: true,
      },
      label: {
        text: "START",
        font: "bold 12px sans-serif",
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        pixelOffset: new Cesium.Cartesian2(0, -36),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  } else {
    aircraftRef.current.position = new Cesium.ConstantPositionProperty(aircraftPosition);
    aircraftRef.current.orientation = new Cesium.ConstantProperty(orientation);
    aircraftRef.current.show = true;
  }

  const groundPosition = Cesium.Cartesian3.fromDegrees(state.lon, state.lat, 0);
  if (!groundPointRef.current) {
    groundPointRef.current = viewer.entities.add({
      id: PLACEMENT_GROUND_POINT_ID,
      name: "Pilot Initial Ground Point",
      position: groundPosition,
      point: {
        pixelSize: 11,
        color: Cesium.Color.fromCssColorString("#22d3ee"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  } else {
    groundPointRef.current.position = new Cesium.ConstantPositionProperty(groundPosition);
    groundPointRef.current.show = true;
  }
}

function removePlacementPreview(
  viewer: Cesium.Viewer | null,
  aircraftRef: MutableRefObject<Cesium.Entity | null>,
  groundPointRef: MutableRefObject<Cesium.Entity | null>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    aircraftRef.current = null;
    groundPointRef.current = null;
    return;
  }

  if (aircraftRef.current) {
    viewer.entities.remove(aircraftRef.current);
    aircraftRef.current = null;
  }
  if (groundPointRef.current) {
    viewer.entities.remove(groundPointRef.current);
    groundPointRef.current = null;
  }
  viewer.scene.requestRender();
}
