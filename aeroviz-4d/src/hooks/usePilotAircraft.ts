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
  segmentIndex?: number;
}

interface UsePilotAircraftOptions {
  enabled: boolean;
  pose: PilotAircraftPose | null;
  trail: PilotAircraftPose[];
  follow: boolean;
}

const PILOT_AIRCRAFT_ID = "pilot-mode-aircraft";
const PILOT_TRAIL_ID_PREFIX = "pilot-mode-trail-";
const PILOT_TRAIL_COLORS = [
  "#67e8f9",
  "#f97316",
  "#a3e635",
  "#f472b6",
  "#facc15",
  "#60a5fa",
];

interface PilotTrailSegment {
  segmentIndex: number;
  points: PilotAircraftPose[];
}

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
  const trailRefs = useRef<Cesium.Entity[]>([]);
  const trailPositionsRef = useRef<Cesium.Cartesian3[][]>([]);
  const trailPositionsPropertyRef = useRef<Cesium.CallbackProperty[]>([]);

  useEffect(() => {
    if (!viewer || !enabled || !pose) {
      removePilotEntities(viewer, aircraftRef, trailRefs);
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

    const trailSegments = buildTrailSegments(trail);
    trailPositionsRef.current = trailSegments.map((segment) =>
      segment.points.map((point) =>
        Cesium.Cartesian3.fromDegrees(point.lon, point.lat, point.altM),
      ),
    );
    syncTrailSegmentEntities(
      viewer,
      trailSegments,
      trailRefs,
      trailPositionsRef,
      trailPositionsPropertyRef,
    );

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
      removePilotEntities(viewer, aircraftRef, trailRefs);
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
  trailPositionsRef: MutableRefObject<Cesium.Cartesian3[][]>,
  trailPositionsPropertyRef: MutableRefObject<Cesium.CallbackProperty[]>,
): void {
  aircraftPositionPropertyRef.current = null;
  aircraftOrientationPropertyRef.current = null;
  trailPositionsRef.current = [];
  trailPositionsPropertyRef.current = [];
}

function removePilotEntities(
  viewer: Cesium.Viewer | null,
  aircraftRef: MutableRefObject<Cesium.Entity | null>,
  trailRefs: MutableRefObject<Cesium.Entity[]>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    aircraftRef.current = null;
    trailRefs.current = [];
    return;
  }

  if (viewer.trackedEntity?.id === PILOT_AIRCRAFT_ID) {
    viewer.trackedEntity = undefined;
  }
  if (aircraftRef.current) {
    viewer.entities.remove(aircraftRef.current);
    aircraftRef.current = null;
  }
  trailRefs.current.forEach((trailEntity) => viewer.entities.remove(trailEntity));
  trailRefs.current = [];
  viewer.scene.requestRender();
}

function buildTrailSegments(trail: PilotAircraftPose[]): PilotTrailSegment[] {
  const segments: PilotTrailSegment[] = [];
  for (let index = 1; index < trail.length; index += 1) {
    const previousPoint = trail[index - 1];
    const currentPoint = trail[index];
    const segmentIndex = currentPoint.segmentIndex ?? 0;
    const activeSegment = segments[segments.length - 1];

    if (!activeSegment || activeSegment.segmentIndex !== segmentIndex) {
      segments.push({
        segmentIndex,
        points: [previousPoint, currentPoint],
      });
    } else {
      activeSegment.points.push(currentPoint);
    }
  }
  return segments;
}

function syncTrailSegmentEntities(
  viewer: Cesium.Viewer,
  trailSegments: PilotTrailSegment[],
  trailRefs: MutableRefObject<Cesium.Entity[]>,
  trailPositionsRef: MutableRefObject<Cesium.Cartesian3[][]>,
  trailPositionsPropertyRef: MutableRefObject<Cesium.CallbackProperty[]>,
): void {
  while (trailRefs.current.length > trailSegments.length) {
    const entity = trailRefs.current.pop();
    trailPositionsPropertyRef.current.pop();
    if (entity) {
      viewer.entities.remove(entity);
    }
  }

  trailSegments.forEach((segment, index) => {
    const material = trailMaterialForSegment(segment.segmentIndex);
    let entity = trailRefs.current[index];
    if (!entity) {
      const positionsProperty = new Cesium.CallbackProperty(
        () => trailPositionsRef.current[index] ?? [],
        false,
      );
      trailPositionsPropertyRef.current[index] = positionsProperty;
      entity = viewer.entities.add({
        id: `${PILOT_TRAIL_ID_PREFIX}${index}`,
        name: `Pilot Mode Trail ${segment.segmentIndex + 1}`,
        show: trailPositionsRef.current[index]?.length > 1,
        polyline: {
          positions: positionsProperty,
          width: 3,
          material,
          clampToGround: false,
        },
      });
      trailRefs.current[index] = entity;
    } else {
      entity.show = trailPositionsRef.current[index]?.length > 1;
      entity.name = `Pilot Mode Trail ${segment.segmentIndex + 1}`;
      if (entity.polyline) {
        entity.polyline.material = material;
      }
    }
  });
}

function trailMaterialForSegment(segmentIndex: number): Cesium.ColorMaterialProperty {
  const color = PILOT_TRAIL_COLORS[segmentIndex % PILOT_TRAIL_COLORS.length];
  return new Cesium.ColorMaterialProperty(
    Cesium.Color.fromCssColorString(color).withAlpha(0.78),
  );
}
