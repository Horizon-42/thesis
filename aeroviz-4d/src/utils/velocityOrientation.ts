/**
 * velocityOrientation.ts
 * ----------------------
 * Shared orientation helper for time-dynamic entities whose CZML carries position
 * only (no orientation) — used by the dynamics-comparison playback and the
 * optimizer-comparison trajectory layer to point an aircraft model down its path.
 */

import * as Cesium from "cesium";

/**
 * A VelocityOrientationProperty that holds its last valid value. The position
 * series uses HOLD extrapolation, so once playback reaches the final sample the
 * velocity drops to zero and the underlying property returns `undefined` — which
 * would snap the model to a default attitude. Returning the last in-flight
 * orientation keeps the parked aircraft pointing along its final state instead.
 */
export function makeStableVelocityOrientation(
  position: Cesium.PositionProperty,
): Cesium.CallbackProperty {
  const velocity = new Cesium.VelocityOrientationProperty(position);
  let last: Cesium.Quaternion | undefined;
  return new Cesium.CallbackProperty((time, result) => {
    const value = time ? velocity.getValue(time, result) : undefined;
    if (value) {
      last = Cesium.Quaternion.clone(value, last);
      return value;
    }
    return last;
  }, false);
}
