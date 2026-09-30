/**
 * The aircraft model's orientation (`aircraftOrientation`), the one convention Fly, Optimize and Training share: the
 * glTF's nose (+x) points along the compass heading, pitch lifts it, a right bank lowers the right wing.
 */
import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import { aircraftOrientation, compassFromPsiDeg } from "../aircraftOrientation";

const position = Cesium.Cartesian3.fromDegrees(-78.8, 35.9, 1000);
const enu = Cesium.Transforms.eastNorthUpToFixedFrame(position);
const toLocal = Cesium.Matrix4.inverse(enu, new Cesium.Matrix4());

/** The model's body axis ``axis`` (x nose, y left wing, z up) in the local east-north-up frame. */
function bodyAxis(orientation: Cesium.Quaternion, axis: Cesium.Cartesian3): Cesium.Cartesian3 {
  const rotation = Cesium.Matrix3.fromQuaternion(orientation);
  const fixed = Cesium.Matrix3.multiplyByVector(rotation, axis, new Cesium.Cartesian3());
  return Cesium.Matrix4.multiplyByPointAsVector(toLocal, fixed, new Cesium.Cartesian3());
}

describe("aircraftOrientation", () => {
  it("points the nose along the compass heading and lifts it with the pitch", () => {
    const north = bodyAxis(aircraftOrientation(position, 0, 0, 0), Cesium.Cartesian3.UNIT_X);
    [north.x, north.y, north.z].forEach((value, at) => expect(value).toBeCloseTo([0, 1, 0][at], 6));
    const east = bodyAxis(aircraftOrientation(position, 90, 0, 0), Cesium.Cartesian3.UNIT_X);
    [east.x, east.y].forEach((value, at) => expect(value).toBeCloseTo([1, 0][at], 6));
    const up = bodyAxis(aircraftOrientation(position, 0, 10, 0), Cesium.Cartesian3.UNIT_X);
    expect(up.z).toBeCloseTo(Math.sin(Cesium.Math.toRadians(10)), 6);
  });

  it("lowers the right wing for a right bank: the left wing (+y) rises", () => {
    const leftWing = bodyAxis(aircraftOrientation(position, 0, 0, 20), Cesium.Cartesian3.UNIT_Y);
    expect(leftWing.z).toBeCloseTo(Math.sin(Cesium.Math.toRadians(20)), 6);
  });

  it("reads a simulator heading (math-ENU: 0 east, counter-clockwise) as the compass", () => {
    expect([compassFromPsiDeg(0), compassFromPsiDeg(90)]).toEqual([90, 0]);
  });
});
