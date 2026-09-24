import {
  FEET_TO_METERS as METRES_PER_FOOT,
  knotsToMetresPerSecond,
} from "../utils/procedureGeoMath";

// Re-exported so existing importers keep working; the impl lives in procedureGeoMath.
export { knotsToMetresPerSecond };

export const TARGET_THRESHOLD_CROSSING_HEIGHT_FT = 50;
export const TARGET_THRESHOLD_CROSSING_HEIGHT_M =
  TARGET_THRESHOLD_CROSSING_HEIGHT_FT * METRES_PER_FOOT;

export const TARGET_RUNWAY_HEADING_TOLERANCE_DEG = 1;

interface TargetSpeedAircraftConfig {
  terminalSpeedKt: number;
  terminalSpeedMinKt: number;
  terminalSpeedMaxKt: number;
  thresholdCrossingHeightM: number;
}

export function targetAltitudeMForThreshold(
  thresholdElevationM: number,
  aircraft: TargetSpeedAircraftConfig | null = null,
): number {
  return thresholdElevationM +
    (aircraft?.thresholdCrossingHeightM ?? TARGET_THRESHOLD_CROSSING_HEIGHT_M);
}

// The backend catalog states each aircraft's target speed and the range it may be set in:
// the published approach speed at the landing mass, and the threshold speed gate's window at
// that mass (aeroviz_backend.simulation_backend._terminal_speeds_kt). There is no default
// aircraft: without a catalog entry there is no target speed.
export function defaultTargetSpeedMps(aircraft: TargetSpeedAircraftConfig): number {
  return knotsToMetresPerSecond(aircraft.terminalSpeedKt);
}

export function targetSpeedBoundsMps(
  aircraft: TargetSpeedAircraftConfig,
): { min: number; max: number } {
  return {
    min: knotsToMetresPerSecond(aircraft.terminalSpeedMinKt),
    max: knotsToMetresPerSecond(aircraft.terminalSpeedMaxKt),
  };
}

export function clampTargetSpeedMps(
  speedMps: number,
  aircraft: TargetSpeedAircraftConfig,
): number {
  const bounds = targetSpeedBoundsMps(aircraft);
  return clamp(
    speedMps,
    bounds.min,
    bounds.max,
  );
}

export function runwayAlignedHeadingDeg(runwayHeadingDeg: number): number {
  return normalizeDegrees(runwayHeadingDeg);
}

export function clampHeadingToRunwayTolerance(
  headingDeg: number,
  runwayHeadingDeg: number,
): number {
  const runwayHeading = normalizeDegrees(runwayHeadingDeg);
  const offset = signedAngularOffsetDeg(headingDeg, runwayHeading);
  return normalizeDegrees(
    runwayHeading + clamp(
      offset,
      -TARGET_RUNWAY_HEADING_TOLERANCE_DEG,
      TARGET_RUNWAY_HEADING_TOLERANCE_DEG,
    ),
  );
}

function signedAngularOffsetDeg(valueDeg: number, centerDeg: number): number {
  const offset = normalizeDegrees(valueDeg - centerDeg + 180) - 180;
  return offset === -180 ? 180 : offset;
}

function normalizeDegrees(value: number): number {
  return ((value % 360) + 360) % 360;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
