import { useEffect, useRef, type MutableRefObject } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

export interface PilotAircraftPose {
  lon: number;
  lat: number;
  altM: number;
  headingDeg: number;
  flightPathDeg: number;
  bankDeg: number;
  attackDeg: number;
}

interface UsePilotAircraftOptions {
  enabled: boolean;
  pose: PilotAircraftPose | null;
  trail: PilotAircraftPose[];
  follow: boolean;
}

const PILOT_AIRCRAFT_ID = "pilot-mode-aircraft";
const PILOT_TRAIL_ID = "pilot-mode-trail";

export function usePilotAircraft({
  enabled,
  pose,
  trail,
  follow,
}: UsePilotAircraftOptions): void {
  const { viewer, setSelectedFlightId } = useApp();
  const aircraftRef = useRef<Cesium.Entity | null>(null);
  const aircraftPositionPropertyRef =
    useRef<Cesium.ConstantPositionProperty | null>(null);
  const aircraftOrientationPropertyRef =
    useRef<Cesium.ConstantProperty | null>(null);
  const trailRef = useRef<Cesium.Entity | null>(null);
  const trailPositionsRef = useRef<Cesium.Cartesian3[]>([]);
  const trailPositionsPropertyRef = useRef<Cesium.CallbackProperty | null>(null);

  useEffect(() => {
    if (!viewer || !enabled || !pose) {
      removePilotEntities(viewer, aircraftRef, trailRef);
      resetPilotProperties(
        aircraftPositionPropertyRef,
        aircraftOrientationPropertyRef,
        trailPositionsRef,
        trailPositionsPropertyRef,
      );
      return;
    }

    const position = Cesium.Cartesian3.fromDegrees(pose.lon, pose.lat, pose.altM);
    const orientation = Cesium.Transforms.headingPitchRollQuaternion(
      position,
      new Cesium.HeadingPitchRoll(
        Cesium.Math.toRadians(-pose.headingDeg),
        Cesium.Math.toRadians(pose.flightPathDeg + pose.attackDeg),
        Cesium.Math.toRadians(-pose.bankDeg),
      ),
    );

    if (!aircraftRef.current) {
      aircraftPositionPropertyRef.current = new Cesium.ConstantPositionProperty(position);
      aircraftOrientationPropertyRef.current = new Cesium.ConstantProperty(orientation);
      aircraftRef.current = viewer.entities.add({
        id: PILOT_AIRCRAFT_ID,
        name: "Pilot Mode Aircraft",
        position: aircraftPositionPropertyRef.current,
        orientation: aircraftOrientationPropertyRef.current,
        model: {
          uri: "/models/aircraft.glb",
          scale: 3.0,
          minimumPixelSize: 36,
          maximumScale: 20_000,
          runAnimations: true,
        },
        label: {
          text: "PILOT",
          font: "bold 12px sans-serif",
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3,
          pixelOffset: new Cesium.Cartesian2(0, -34),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    } else {
      aircraftPositionPropertyRef.current?.setValue(position);
      aircraftOrientationPropertyRef.current?.setValue(orientation);
      aircraftRef.current.show = true;
    }

    trailPositionsRef.current = trail.map((point) =>
      Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.altM),
    );
    if (!trailRef.current) {
      trailPositionsPropertyRef.current = new Cesium.CallbackProperty(
        () => trailPositionsRef.current,
        false,
      );
      trailRef.current = viewer.entities.add({
        id: PILOT_TRAIL_ID,
        name: "Pilot Mode Trail",
        show: trailPositionsRef.current.length > 1,
        polyline: {
          positions: trailPositionsPropertyRef.current,
          width: 3,
          material: Cesium.Color.fromCssColorString("#67e8f9").withAlpha(0.78),
          clampToGround: false,
        },
      });
    } else if (trailRef.current.polyline) {
      trailRef.current.show = trailPositionsRef.current.length > 1;
    }

    if (follow) {
      if (viewer.trackedEntity !== aircraftRef.current) {
        viewer.trackedEntity = aircraftRef.current;
      }
      setSelectedFlightId(null);
    } else if (viewer.trackedEntity?.id === PILOT_AIRCRAFT_ID) {
      viewer.trackedEntity = undefined;
    }

    viewer.scene.requestRender();
  }, [enabled, follow, pose, setSelectedFlightId, trail, viewer]);

  useEffect(() => {
    return () => {
      removePilotEntities(viewer, aircraftRef, trailRef);
      resetPilotProperties(
        aircraftPositionPropertyRef,
        aircraftOrientationPropertyRef,
        trailPositionsRef,
        trailPositionsPropertyRef,
      );
    };
  }, [viewer]);
}

function resetPilotProperties(
  aircraftPositionPropertyRef: MutableRefObject<
    Cesium.ConstantPositionProperty | null
  >,
  aircraftOrientationPropertyRef: MutableRefObject<
    Cesium.ConstantProperty | null
  >,
  trailPositionsRef: MutableRefObject<Cesium.Cartesian3[]>,
  trailPositionsPropertyRef: MutableRefObject<Cesium.CallbackProperty | null>,
): void {
  aircraftPositionPropertyRef.current = null;
  aircraftOrientationPropertyRef.current = null;
  trailPositionsRef.current = [];
  trailPositionsPropertyRef.current = null;
}

function removePilotEntities(
  viewer: Cesium.Viewer | null,
  aircraftRef: MutableRefObject<Cesium.Entity | null>,
  trailRef: MutableRefObject<Cesium.Entity | null>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    aircraftRef.current = null;
    trailRef.current = null;
    return;
  }

  if (viewer.trackedEntity?.id === PILOT_AIRCRAFT_ID) {
    viewer.trackedEntity = undefined;
  }
  if (aircraftRef.current) {
    viewer.entities.remove(aircraftRef.current);
    aircraftRef.current = null;
  }
  if (trailRef.current) {
    viewer.entities.remove(trailRef.current);
    trailRef.current = null;
  }
  viewer.scene.requestRender();
}
