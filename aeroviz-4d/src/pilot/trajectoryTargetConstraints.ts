const METRES_PER_FOOT = 0.3048;
const METRES_PER_NAUTICAL_MILE = 1852;
const SECONDS_PER_HOUR = 3600;

export const TARGET_THRESHOLD_CROSSING_HEIGHT_FT = 6;
export const TARGET_THRESHOLD_CROSSING_HEIGHT_M =
  TARGET_THRESHOLD_CROSSING_HEIGHT_FT * METRES_PER_FOOT;

export const TARGET_APPROACH_SPEED_KT = 62;
export const TARGET_APPROACH_SPEED_TOLERANCE_KT = 5;
export const TARGET_APPROACH_SPEED_MPS = knotsToMetresPerSecond(
  TARGET_APPROACH_SPEED_KT,
);
export const TARGET_APPROACH_SPEED_MIN_MPS = knotsToMetresPerSecond(
  TARGET_APPROACH_SPEED_KT - TARGET_APPROACH_SPEED_TOLERANCE_KT,
);
export const TARGET_APPROACH_SPEED_MAX_MPS = knotsToMetresPerSecond(
  TARGET_APPROACH_SPEED_KT + TARGET_APPROACH_SPEED_TOLERANCE_KT,
);

export const TARGET_RUNWAY_HEADING_TOLERANCE_DEG = 1;

export function targetAltitudeMForThreshold(thresholdElevationM: number): number {
  return thresholdElevationM + TARGET_THRESHOLD_CROSSING_HEIGHT_M;
}

export function clampTargetSpeedMps(speedMps: number): number {
  return clamp(
    speedMps,
    TARGET_APPROACH_SPEED_MIN_MPS,
    TARGET_APPROACH_SPEED_MAX_MPS,
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

function knotsToMetresPerSecond(knots: number): number {
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
