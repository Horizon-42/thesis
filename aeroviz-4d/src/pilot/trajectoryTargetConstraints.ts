const METRES_PER_FOOT = 0.3048;
const METRES_PER_NAUTICAL_MILE = 1852;
const SECONDS_PER_HOUR = 3600;

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

export function defaultTargetSpeedMps(
  aircraft: TargetSpeedAircraftConfig | null,
): number {
  return knotsToMetresPerSecond(aircraft?.terminalSpeedKt ?? 145);
}

export function targetSpeedBoundsMps(
  aircraft: TargetSpeedAircraftConfig | null,
): { min: number; max: number } {
  return {
    min: knotsToMetresPerSecond(aircraft?.terminalSpeedMinKt ?? 135),
    max: knotsToMetresPerSecond(aircraft?.terminalSpeedMaxKt ?? 155),
  };
}

export function clampTargetSpeedMps(
  speedMps: number,
  aircraft: TargetSpeedAircraftConfig | null,
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

export function knotsToMetresPerSecond(knots: number): number {
  return (knots * METRES_PER_NAUTICAL_MILE) / SECONDS_PER_HOUR;
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
