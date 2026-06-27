import { useEffect, useRef, type MutableRefObject } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import type { RunwayThresholdTarget } from "../data/runwayThresholdTargets";
import type { PilotAircraftConfig, PilotResetState } from "../pilot/pilotClient";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  EARTH_RADIUS_M,
  METERS_PER_NM as METRES_PER_NAUTICAL_MILE,
} from "../utils/procedureGeoMath";

export interface PilotInitialPlacementPosition {
  lon: number;
  lat: number;
  altM?: number;
  headingDeg?: number;
  flightPathDeg?: number;
}

interface PilotInitialPlacementGuidance {
  runway: RunwayThresholdTarget;
  aircraft: PilotAircraftConfig;
}

interface UsePilotInitialPlacementOptions {
  enabled: boolean;
  previewVisible: boolean;
  initialState: PilotResetState;
  placementGuidance: PilotInitialPlacementGuidance | null;
  onPositionChange: (position: PilotInitialPlacementPosition) => void;
  onFinish: () => void;
  onCancel: () => void;
}

const PLACEMENT_AIRCRAFT_ID = "pilot-initial-placement-aircraft";
const PLACEMENT_GROUND_POINT_ID = "pilot-initial-placement-ground-point";
const PLACEMENT_DROP_LINE_ID = "pilot-initial-placement-drop-line";
const PLACEMENT_GUIDANCE_ID = "pilot-initial-placement-guidance";

export function usePilotInitialPlacement({
  enabled,
  previewVisible,
  initialState,
  placementGuidance,
  onPositionChange,
  onFinish,
  onCancel,
}: UsePilotInitialPlacementOptions): void {
  const { viewer, setSelectedFlightId } = useApp();
  const aircraftRef = useRef<Cesium.Entity | null>(null);
  const groundPointRef = useRef<Cesium.Entity | null>(null);
  const dropLineRef = useRef<Cesium.Entity | null>(null);
  const guidanceRef = useRef<Cesium.Entity | null>(null);
  const isDraggingRef = useRef(false);
  const callbacksRef = useRef({ onPositionChange, onFinish, onCancel });

  useEffect(() => {
    callbacksRef.current = { onPositionChange, onFinish, onCancel };
  }, [onCancel, onFinish, onPositionChange]);

  useEffect(() => {
    if (!previewVisible || !isCesiumViewerUsable(viewer)) {
      removePlacementPreview(viewer, aircraftRef, groundPointRef, dropLineRef);
      return;
    }

    updatePlacementPreview(
      viewer,
      initialState,
      aircraftRef,
      groundPointRef,
      dropLineRef,
    );
    viewer.scene.requestRender();
  }, [initialState, previewVisible, viewer]);

  useEffect(() => {
    if (!enabled || !placementGuidance || !isCesiumViewerUsable(viewer)) {
      removePlacementGuidance(viewer, guidanceRef);
      return;
    }

    updatePlacementGuidance(viewer, placementGuidance, guidanceRef);
    viewer.scene.requestRender();
  }, [enabled, placementGuidance, viewer]);

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

      callbacksRef.current.onPositionChange(
        placementGuidance
          ? withSuggestedFinalApproachState(nextPosition, placementGuidance)
          : nextPosition,
      );
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
  }, [enabled, placementGuidance, setSelectedFlightId, viewer]);

  useEffect(() => {
    return () => {
      removePlacementPreview(viewer, aircraftRef, groundPointRef, dropLineRef);
      removePlacementGuidance(viewer, guidanceRef);
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
  dropLineRef: MutableRefObject<Cesium.Entity | null>,
): void {
  const aircraftPosition = Cesium.Cartesian3.fromDegrees(state.lon, state.lat, state.altM);
  const groundPosition = Cesium.Cartesian3.fromDegrees(state.lon, state.lat, 0);
  const orientation = Cesium.Transforms.headingPitchRollQuaternion(
    aircraftPosition,
    new Cesium.HeadingPitchRoll(
      Cesium.Math.toRadians(-state.headingDeg),
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
        minimumPixelSize: 54,
        maximumScale: 20_000,
        color: Cesium.Color.fromCssColorString("#38bdf8").withAlpha(0.92),
        colorBlendMode: Cesium.ColorBlendMode.MIX,
        colorBlendAmount: 0.5,
        silhouetteColor: Cesium.Color.fromCssColorString("#e0f2fe").withAlpha(0.96),
        silhouetteSize: 2.5,
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

  if (!groundPointRef.current) {
    groundPointRef.current = viewer.entities.add({
      id: PLACEMENT_GROUND_POINT_ID,
      name: "Pilot Initial Ground Point",
      position: groundPosition,
      point: {
        pixelSize: 17,
        color: Cesium.Color.fromCssColorString("#0ea5e9").withAlpha(0.96),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 4,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  } else {
    groundPointRef.current.position = new Cesium.ConstantPositionProperty(groundPosition);
    groundPointRef.current.show = true;
  }

  if (!dropLineRef.current) {
    dropLineRef.current = viewer.entities.add({
      id: PLACEMENT_DROP_LINE_ID,
      name: "Pilot Initial Aircraft Ground Link",
      polyline: {
        positions: [groundPosition, aircraftPosition],
        width: 2.5,
        material: new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.fromCssColorString("#7dd3fc").withAlpha(0.92),
          dashLength: 12,
        }),
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.fromCssColorString("#bae6fd").withAlpha(0.82),
          dashLength: 12,
        }),
      },
    });
  } else {
    dropLineRef.current.polyline = new Cesium.PolylineGraphics({
      positions: [groundPosition, aircraftPosition],
      width: 2.5,
      material: new Cesium.PolylineDashMaterialProperty({
        color: Cesium.Color.fromCssColorString("#7dd3fc").withAlpha(0.92),
        dashLength: 12,
      }),
      depthFailMaterial: new Cesium.PolylineDashMaterialProperty({
        color: Cesium.Color.fromCssColorString("#bae6fd").withAlpha(0.82),
        dashLength: 12,
      }),
    });
    dropLineRef.current.show = true;
  }
}

function removePlacementPreview(
  viewer: Cesium.Viewer | null,
  aircraftRef: MutableRefObject<Cesium.Entity | null>,
  groundPointRef: MutableRefObject<Cesium.Entity | null>,
  dropLineRef: MutableRefObject<Cesium.Entity | null>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    aircraftRef.current = null;
    groundPointRef.current = null;
    dropLineRef.current = null;
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
  if (dropLineRef.current) {
    viewer.entities.remove(dropLineRef.current);
    dropLineRef.current = null;
  }
  viewer.scene.requestRender();
}

function updatePlacementGuidance(
  viewer: Cesium.Viewer,
  guidance: PilotInitialPlacementGuidance,
  guidanceRef: MutableRefObject<Cesium.Entity | null>,
): void {
  const positions = buildFinalApproachBandPositions(guidance);

  if (!guidanceRef.current) {
    guidanceRef.current = viewer.entities.add({
      id: PLACEMENT_GUIDANCE_ID,
      name: "Pilot Initial Placement Guidance",
      polygon: makeGuidancePolygon(positions),
    });
    return;
  }

  guidanceRef.current.polygon = makeGuidancePolygon(positions);
  guidanceRef.current.show = true;
}

function makeGuidancePolygon(positions: Cesium.Cartesian3[]): Cesium.PolygonGraphics {
  return new Cesium.PolygonGraphics({
    hierarchy: new Cesium.PolygonHierarchy(positions),
    material: Cesium.Color.fromCssColorString("#22c55e").withAlpha(0.2),
    outline: true,
    outlineColor: Cesium.Color.fromCssColorString("#bbf7d0").withAlpha(0.92),
    heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
  });
}

function removePlacementGuidance(
  viewer: Cesium.Viewer | null,
  guidanceRef: MutableRefObject<Cesium.Entity | null>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    guidanceRef.current = null;
    return;
  }

  if (guidanceRef.current) {
    viewer.entities.remove(guidanceRef.current);
    guidanceRef.current = null;
    viewer.scene.requestRender();
  }
}

function buildFinalApproachBandPositions(
  guidance: PilotInitialPlacementGuidance,
): Cesium.Cartesian3[] {
  const runway = guidance.runway;
  const aircraft = guidance.aircraft;
  const innerM = aircraft.finalApproachMinNm * METRES_PER_NAUTICAL_MILE;
  const outerM = aircraft.finalApproachMaxNm * METRES_PER_NAUTICAL_MILE;
  const halfWidthM = aircraft.finalApproachLateralHalfWidthNm *
    METRES_PER_NAUTICAL_MILE;
  const direction = unitVectorFromSimulatorPsi(runway.psiDeg + 180);
  const lateral = { east: -direction.north, north: direction.east };

  const points = [
    offsetLonLat(runway.lon, runway.lat, direction, lateral, innerM, halfWidthM),
    offsetLonLat(runway.lon, runway.lat, direction, lateral, outerM, halfWidthM),
    offsetLonLat(runway.lon, runway.lat, direction, lateral, outerM, -halfWidthM),
    offsetLonLat(runway.lon, runway.lat, direction, lateral, innerM, -halfWidthM),
  ];

  return points.map((point) => Cesium.Cartesian3.fromDegrees(point.lon, point.lat, 0));
}

function withSuggestedFinalApproachState(
  position: PilotInitialPlacementPosition,
  guidance: PilotInitialPlacementGuidance,
): PilotInitialPlacementPosition {
  const distanceM = horizontalDistanceM(
    position.lon,
    position.lat,
    guidance.runway.lon,
    guidance.runway.lat,
  );
  const glideAngleRad = Cesium.Math.toRadians(
    guidance.aircraft.finalApproachGlideAngleDeg,
  );
  return {
    ...position,
    altM: guidance.runway.altM +
      guidance.aircraft.thresholdCrossingHeightM +
      Math.tan(glideAngleRad) * distanceM,
    headingDeg: guidance.runway.psiDeg,
    flightPathDeg: -guidance.aircraft.finalApproachGlideAngleDeg,
  };
}

function offsetLonLat(
  lon: number,
  lat: number,
  direction: { east: number; north: number },
  lateral: { east: number; north: number },
  alongM: number,
  lateralM: number,
): { lon: number; lat: number } {
  const eastM = direction.east * alongM + lateral.east * lateralM;
  const northM = direction.north * alongM + lateral.north * lateralM;
  const latRad = Cesium.Math.toRadians(lat);
  return {
    lon: lon + Cesium.Math.toDegrees(eastM / (EARTH_RADIUS_M * Math.cos(latRad))),
    lat: lat + Cesium.Math.toDegrees(northM / EARTH_RADIUS_M),
  };
}

function unitVectorFromSimulatorPsi(psiDeg: number): { east: number; north: number } {
  const psiRad = Cesium.Math.toRadians(psiDeg);
  return {
    east: Math.cos(psiRad),
    north: Math.sin(psiRad),
  };
}

function horizontalDistanceM(
  lonA: number,
  latA: number,
  lonB: number,
  latB: number,
): number {
  const meanLat = Cesium.Math.toRadians((latA + latB) * 0.5);
  const eastM = Cesium.Math.toRadians(lonA - lonB) *
    EARTH_RADIUS_M *
    Math.cos(meanLat);
  const northM = Cesium.Math.toRadians(latA - latB) * EARTH_RADIUS_M;
  return Math.hypot(eastM, northM);
}
