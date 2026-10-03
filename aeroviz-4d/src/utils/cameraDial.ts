/**
 * cameraDial.ts
 * -------------
 * Pure math behind the HUD's drag controller: a miniature ground disc seen from the camera.
 * Dragging it orbits the view (sideways = heading, up/down = pitch); a strip beside it zooms.
 * Angles are degrees, distances metres.
 */

import { clamp, toRadians } from "./procedureGeoMath";

/** Orbit pitch limits: straight down … just under the horizon (a camera below its focus sees the ground from beneath). */
export const ORBIT_PITCH_MIN_DEG = -89;
export const ORBIT_PITCH_MAX_DEG = -1;
export const ZOOM_MIN_RANGE_M = 10;

const ORBIT_DEG_PER_PX = 0.4;
const ZOOM_LOG_PER_PX = 0.012;
/** The disc never flattens past this, so its compass letters stay readable at a near-horizon pitch. */
const DIAL_FLATTEN_MIN = 0.28;

export function wrapDegrees360(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

/** Vertical squash of the ground disc: 1 looking straight down, → 0 at the horizon. */
export function dialFlatten(pitchDeg: number): number {
  return Math.max(DIAL_FLATTEN_MIN, Math.sin(toRadians(Math.abs(pitchDeg))));
}

/** Where a compass bearing sits on the disc (offset from its centre); `depth` is +1 on the far side, −1 on the near one. */
export function dialPoint(
  bearingDeg: number, headingDeg: number, radius: number, flatten: number,
): { x: number; y: number; depth: number } {
  const t = toRadians(bearingDeg - headingDeg);
  return { x: radius * Math.sin(t), y: -radius * Math.cos(t) * flatten, depth: Math.cos(t) };
}

/**
 * The disc turns with the finger: dragging right carries north to the right (heading falls), dragging down
 * tips the disc towards a top view (pitch falls) — the way the globe itself behaves under a mouse drag.
 */
export function orbitDrag(
  headingDeg: number, pitchDeg: number, dxPx: number, dyPx: number,
): { headingDeg: number; pitchDeg: number } {
  return {
    headingDeg: wrapDegrees360(headingDeg - dxPx * ORBIT_DEG_PER_PX),
    pitchDeg: clamp(pitchDeg - dyPx * ORBIT_DEG_PER_PX, ORBIT_PITCH_MIN_DEG, ORBIT_PITCH_MAX_DEG),
  };
}

/** Dragging up (negative dy) closes the distance to the focus; the change is proportional, so it feels the same at any range. */
export function zoomRange(rangeM: number, dyPx: number): number {
  return Math.max(ZOOM_MIN_RANGE_M, rangeM * Math.exp(dyPx * ZOOM_LOG_PER_PX));
}

/** What the HUD reads from the camera. */
export interface CameraReadout {
  heading: number;  // 0–360 degrees (compass bearing the camera faces)
  pitch: number;    // degrees, negative = looking down
  altitude: number; // metres above the ellipsoid
  lat: number;      // decimal degrees
  lon: number;      // decimal degrees
}

// The readout shows heading/pitch to 0.1°, altitude to 1 m, lat/lon to 1e-4°: a camera read finer than that
// is the same state, so it must not re-render the panel (the read runs after every frame).
export const quantiseDeg = (deg: number) => Math.round(deg * 10) / 10;

/** Heading to 0.1° on [0, 360): 359.95 rounds to 360.0, which is 0 (never "360"). */
export const quantiseHeading = (deg: number) => quantiseDeg(deg) % 360;

export function quantiseReadout(c: CameraReadout): CameraReadout {
  return {
    heading: quantiseHeading(c.heading),
    pitch: quantiseDeg(c.pitch) + 0, // + 0: a pitch of −0.04 rounds to −0, which must equal 0
    altitude: Math.round(c.altitude),
    lat: Math.round(c.lat * 1e4) / 1e4,
    lon: Math.round(c.lon * 1e4) / 1e4,
  };
}

export function sameReadout(a: CameraReadout, b: CameraReadout): boolean {
  return a.heading === b.heading && a.pitch === b.pitch && a.altitude === b.altitude
    && a.lat === b.lat && a.lon === b.lon;
}
