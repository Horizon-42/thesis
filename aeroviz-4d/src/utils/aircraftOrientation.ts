/**
 * aircraftOrientation.ts
 * ----------------------
 * How the aircraft model is turned to its attitude — the ONE convention every mode that draws it shares (Fly's pilot and
 * placement preview, Optimize's playback, Training). The glTF (`AIRCRAFT_MODEL_URI`) points its nose along +x, so at
 * Cesium heading 0 it faces east: a compass heading turns it by (heading − 90°); pitch up and a right bank are positive.
 */

import * as Cesium from "cesium";

export const AIRCRAFT_MODEL_URI = "/models/aircraft.glb";

/** A simulator heading ψ (math-ENU degrees: 0 east, counter-clockwise) as a compass heading. */
export function compassFromPsiDeg(psiDeg: number): number {
  return 90 - psiDeg;
}

/** The model's orientation at ``position``: ``compassDeg`` heading, ``pitchDeg`` nose up, ``bankRightDeg`` right wing down. */
export function aircraftOrientation(
  position: Cesium.Cartesian3, compassDeg: number, pitchDeg: number, bankRightDeg: number,
): Cesium.Quaternion {
  return Cesium.Transforms.headingPitchRollQuaternion(position, new Cesium.HeadingPitchRoll(
    Cesium.Math.toRadians(compassDeg - 90), Cesium.Math.toRadians(pitchDeg), Cesium.Math.toRadians(bankRightDeg)));
}
