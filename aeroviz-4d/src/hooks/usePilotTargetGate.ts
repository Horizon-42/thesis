import { useEffect, useRef, type MutableRefObject } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

export interface PilotTargetGateState {
  runwayThresholdId: string;
  runwayIdent: string;
  lon: number;
  lat: number;
  altM: number;
  headingDeg: number;
}

interface UsePilotTargetGateOptions {
  enabled: boolean;
  target: PilotTargetGateState | null;
}

const TARGET_GATE_ID = "trajectory-play-target-threshold-gate";
const TARGET_GATE_MARKER_ID = "trajectory-play-target-threshold-marker";
const TARGET_GATE_WIDTH_M = 90;
const TARGET_GATE_HEIGHT_M = 42;
const TARGET_GATE_BASE_HEIGHT_M = 4;
const METERS_PER_DEGREE_LAT = 111_320;

export function usePilotTargetGate({
  enabled,
  target,
}: UsePilotTargetGateOptions): void {
  const { viewer } = useApp();
  const gateRef = useRef<Cesium.Entity | null>(null);
  const markerRef = useRef<Cesium.Entity | null>(null);

  useEffect(() => {
    if (!enabled || !target || !target.runwayThresholdId || !isCesiumViewerUsable(viewer)) {
      removeTargetGate(viewer, gateRef, markerRef);
      return;
    }

    const gatePositions = buildTargetGatePositions(target);
    const markerPosition = Cesium.Cartesian3.fromDegrees(target.lon, target.lat, target.altM);
    const gateLabel = `TARGET ${target.runwayIdent}`;

    if (!gateRef.current) {
      gateRef.current = viewer.entities.add({
        id: TARGET_GATE_ID,
        name: "Trajectory Target Threshold Gate",
        polyline: {
          positions: gatePositions,
          width: 4,
          material: Cesium.Color.fromCssColorString("#22c55e").withAlpha(0.86),
          clampToGround: false,
        },
      });
    } else if (gateRef.current.polyline) {
      gateRef.current.polyline.positions = new Cesium.ConstantProperty(gatePositions);
      gateRef.current.show = true;
    }

    if (!markerRef.current) {
      markerRef.current = viewer.entities.add({
        id: TARGET_GATE_MARKER_ID,
        name: "Trajectory Target Threshold",
        position: markerPosition,
        point: {
          pixelSize: 12,
          color: Cesium.Color.fromCssColorString("#22c55e"),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: gateLabel,
          font: "bold 12px sans-serif",
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3,
          pixelOffset: new Cesium.Cartesian2(0, -30),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    } else {
      markerRef.current.position = new Cesium.ConstantPositionProperty(markerPosition);
      if (markerRef.current.label) {
        markerRef.current.label.text = new Cesium.ConstantProperty(gateLabel);
      }
      markerRef.current.show = true;
    }

    viewer.scene.requestRender();
  }, [enabled, target, viewer]);

  useEffect(() => {
    return () => {
      removeTargetGate(viewer, gateRef, markerRef);
    };
  }, [viewer]);
}

export function buildTargetGatePositions(
  target: Pick<PilotTargetGateState, "lon" | "lat" | "altM" | "headingDeg">,
): Cesium.Cartesian3[] {
  const headingRad = Cesium.Math.toRadians(target.headingDeg);
  const halfWidthM = TARGET_GATE_WIDTH_M / 2;
  const lateralEastM = -Math.sin(headingRad) * halfWidthM;
  const lateralNorthM = Math.cos(headingRad) * halfWidthM;
  const metersPerDegreeLon = Math.max(
    1,
    METERS_PER_DEGREE_LAT * Math.cos(Cesium.Math.toRadians(target.lat)),
  );
  const deltaLon = lateralEastM / metersPerDegreeLon;
  const deltaLat = lateralNorthM / METERS_PER_DEGREE_LAT;
  const baseAltM = target.altM + TARGET_GATE_BASE_HEIGHT_M;
  const topAltM = baseAltM + TARGET_GATE_HEIGHT_M;

  return [
    Cesium.Cartesian3.fromDegrees(target.lon - deltaLon, target.lat - deltaLat, baseAltM),
    Cesium.Cartesian3.fromDegrees(target.lon - deltaLon, target.lat - deltaLat, topAltM),
    Cesium.Cartesian3.fromDegrees(target.lon + deltaLon, target.lat + deltaLat, topAltM),
    Cesium.Cartesian3.fromDegrees(target.lon + deltaLon, target.lat + deltaLat, baseAltM),
    Cesium.Cartesian3.fromDegrees(target.lon - deltaLon, target.lat - deltaLat, baseAltM),
  ];
}

function removeTargetGate(
  viewer: Cesium.Viewer | null,
  gateRef: MutableRefObject<Cesium.Entity | null>,
  markerRef: MutableRefObject<Cesium.Entity | null>,
): void {
  if (!isCesiumViewerUsable(viewer)) {
    gateRef.current = null;
    markerRef.current = null;
    return;
  }

  if (gateRef.current) {
    viewer.entities.remove(gateRef.current);
    gateRef.current = null;
  }
  if (markerRef.current) {
    viewer.entities.remove(markerRef.current);
    markerRef.current = null;
  }
  viewer.scene.requestRender();
}
