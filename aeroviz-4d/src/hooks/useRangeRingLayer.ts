/**
 * useRangeRingLayer.ts
 * --------------------
 * Draws an airport-centred range ring: a circle of a user-adjustable radius
 * around the active airport's reference point, clamped to the terrain surface.
 *
 * Useful as a quick distance reference (e.g. "5 km from the airport") when
 * inspecting trajectories, obstacles, or procedure geometry.
 *
 * Two entities are rendered:
 *   • a ground-clamped polyline forming the circle outline
 *   • a label on the ring's north edge showing the current radius
 *
 * The circle is sampled in a local East-North-Up frame anchored at the airport,
 * which is accurate to sub-metre over the tens-of-km radii used here.
 *
 * Follows the dual-/triple-useEffect pattern used by the other layer hooks:
 *   1. create entities once per (viewer, airport)
 *   2. update geometry when the radius changes
 *   3. sync visibility when the layer toggle changes
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import type { AirportConfig } from "../data/airportData";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

const RANGE_RING_OUTLINE_ID = "range-ring-outline";
const RANGE_RING_LABEL_ID = "range-ring-label";

const RING_SEGMENTS = 180;
const RING_COLOR = Cesium.Color.fromCssColorString("#00e5ff");

/** Sample the circle boundary in the airport's local ENU frame. */
function computeRingPositions(
  airport: AirportConfig,
  radiusM: number,
): Cesium.Cartesian3[] {
  const center = Cesium.Cartesian3.fromDegrees(airport.lon, airport.lat, 0);
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(center);
  const positions: Cesium.Cartesian3[] = [];
  for (let i = 0; i <= RING_SEGMENTS; i += 1) {
    const theta = (i / RING_SEGMENTS) * Cesium.Math.TWO_PI;
    const offset = new Cesium.Cartesian3(
      radiusM * Math.cos(theta),
      radiusM * Math.sin(theta),
      0,
    );
    positions.push(
      Cesium.Matrix4.multiplyByPoint(enuToFixed, offset, new Cesium.Cartesian3()),
    );
  }
  return positions;
}

/** North point of the ring, used to anchor the radius label. */
function ringNorthPoint(airport: AirportConfig, radiusM: number): Cesium.Cartesian3 {
  const center = Cesium.Cartesian3.fromDegrees(airport.lon, airport.lat, 0);
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(center);
  return Cesium.Matrix4.multiplyByPoint(
    enuToFixed,
    new Cesium.Cartesian3(0, radiusM, 0),
    new Cesium.Cartesian3(),
  );
}

function formatRadius(radiusKm: number): string {
  return `${radiusKm.toFixed(1)} km`;
}

export function useRangeRingLayer(): void {
  const { viewer, airport, layers, rangeRingRadiusKm } = useApp();
  const outlineRef = useRef<Cesium.Entity | null>(null);
  const labelRef = useRef<Cesium.Entity | null>(null);

  // Effect 1 — create the ring entities once per (viewer, airport).
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || !airport) return;

    const radiusM = rangeRingRadiusKm * 1000;
    const visible = layers.rangeRing;

    const outline = viewer.entities.add({
      id: RANGE_RING_OUTLINE_ID,
      show: visible,
      polyline: {
        positions: computeRingPositions(airport, radiusM),
        width: 2,
        material: RING_COLOR,
        clampToGround: true,
      },
    });
    outlineRef.current = outline;

    const label = viewer.entities.add({
      id: RANGE_RING_LABEL_ID,
      show: visible,
      position: ringNorthPoint(airport, radiusM),
      label: {
        text: formatRadius(rangeRingRadiusKm),
        font: "bold 13px sans-serif",
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        fillColor: RING_COLOR,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -6),
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    labelRef.current = label;

    return () => {
      if (isCesiumViewerUsable(viewer)) {
        viewer.entities.removeById(RANGE_RING_OUTLINE_ID);
        viewer.entities.removeById(RANGE_RING_LABEL_ID);
      }
      outlineRef.current = null;
      labelRef.current = null;
    };
    // `layers.rangeRing` only seeds the initial visibility; effect 3 keeps it in
    // sync without rebuilding the geometry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewer, airport]);

  // Effect 2 — update geometry/label when the radius changes.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || !airport) return;

    const radiusM = rangeRingRadiusKm * 1000;
    if (outlineRef.current?.polyline) {
      outlineRef.current.polyline.positions = new Cesium.ConstantProperty(
        computeRingPositions(airport, radiusM),
      );
    }
    if (labelRef.current) {
      labelRef.current.position = new Cesium.ConstantPositionProperty(
        ringNorthPoint(airport, radiusM),
      );
      if (labelRef.current.label) {
        labelRef.current.label.text = new Cesium.ConstantProperty(
          formatRadius(rangeRingRadiusKm),
        );
      }
    }
  }, [viewer, airport, rangeRingRadiusKm]);

  // Effect 3 — sync visibility when the layer toggle changes.
  useEffect(() => {
    [outlineRef.current, labelRef.current].forEach((entity) => {
      if (entity) entity.show = layers.rangeRing;
    });
  }, [layers.rangeRing]);
}
