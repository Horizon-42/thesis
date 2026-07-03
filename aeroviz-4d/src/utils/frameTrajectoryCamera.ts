/**
 * frameTrajectoryCamera.ts
 * ------------------------
 * One-time camera fly-to that frames an entire computed/compared trajectory in
 * view when its playback loads.
 *
 * Why: the viewer renders continuously (no requestRenderMode), so a freshly
 * loaded trajectory IS rendered — but the camera is left where `flyToAirport`
 * put it: locked ~1 km over the airport (a very tight view). Fly & Compare
 * trajectories can start far from that view (Compare begins ~30 km from the
 * target; an optimize run can start at a distant initial fix), so the aircraft
 * and its growing trail sit OFF-SCREEN until the user toggles "Follow camera".
 * Framing the whole path on load removes that manual step — compute → see it —
 * while leaving the camera free to orbit afterward (unlike continuous follow).
 */

import * as Cesium from "cesium";
import type { TrajectorySample } from "../pilot/trajectoryOptimizationClient";

/** Default oblique pitch (below horizontal) for the framed view. */
const FRAME_PITCH_DEG = -30;
/** Fly-to duration (seconds); snappy but not jarring. */
const FRAME_DURATION_S = 1.2;

/**
 * Fly the camera once so the entire trajectory (all `samples`) fits in view.
 *
 * Keeps the current heading (no disorienting spin) and looks down at a fixed
 * oblique pitch; range 0 lets Cesium compute a distance that fits the whole
 * bounding sphere for the current field of view.
 *
 * No-op when there is nothing to frame (empty samples / degenerate extent).
 */
export function frameTrajectoryCamera(
  viewer: Cesium.Viewer,
  samples: readonly TrajectorySample[],
  options: { duration?: number } = {},
): void {
  if (samples.length === 0) return;

  const points = samples.map((s) =>
    Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.altM),
  );
  const sphere = Cesium.BoundingSphere.fromPoints(points);
  if (!(sphere.radius > 0)) return; // single point / NaN — nothing to fit

  viewer.camera.flyToBoundingSphere(sphere, {
    duration: options.duration ?? FRAME_DURATION_S,
    offset: new Cesium.HeadingPitchRange(
      viewer.camera.heading,
      Cesium.Math.toRadians(FRAME_PITCH_DEG),
      0, // 0 ⇒ Cesium computes a range that fits the sphere
    ),
  });
}
