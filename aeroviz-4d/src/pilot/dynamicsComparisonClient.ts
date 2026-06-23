import { AEROVIZ_BACKEND_URL, type PilotResetState } from "./pilotClient";
import type { TrajectorySample } from "./trajectoryOptimizationClient";
import {
  asFiniteNumber,
  isRecord,
  readNumber,
  readNumberArray,
  readOptionalNumber,
  readString,
} from "./responseValidators";

/**
 * dynamicsComparisonClient.ts
 * ---------------------------
 * Client for the backend /dynamics-comparison/run endpoint. The backend flies
 * one trajectory under one *constant* control four different ways (A fixed
 * tangent ENU, B re-anchored ENU = reference, C geodetic RHS +transport, D
 * geodetic RHS no-transport) and returns:
 *   - a multi-system CZML (one colored, hideable path per system), and
 *   - a chart series of each system's error vs the reference B (horizontal,
 *     altitude, heading, speed) over distance flown.
 */

/** Constant control held for the whole comparison flight. */
export interface DynamicsComparisonControl {
  thrustN: number;
  bankDeg: number;
  loadFactor: number;
}

export interface DynamicsComparisonRequest {
  initialState: PilotResetState;
  control: DynamicsComparisonControl;
  durationS: number;
  dtS: number;
}

/** One system's presentation metadata (drives the legend + hide toggles). */
export interface DynamicsComparisonSystem {
  key: string;
  label: string;
  colorRgba: [number, number, number, number];
  isReference: boolean;
}

/** Per-system error arrays vs the reference B (one value per sample). */
export interface DynamicsComparisonErrorSeries {
  horiz: number[];
  alt: number[];
  head: number[];
  speed: number[];
  /** Flight-path-angle (gamma) error, degrees (signed). */
  fpa: number[];
}

/** A single system's scalar error vs the reference B at one instant. */
export interface DynamicsComparisonDelta {
  horiz: number;
  alt: number;
  head: number;
  speed: number;
  /** Flight-path-angle (gamma) error, degrees (signed). */
  fpa: number;
}

/** Per-compared-system deltas (A/C/D) at one instant; B is the zero reference. */
export type DynamicsComparisonDeltas = Record<string, DynamicsComparisonDelta>;

export interface DynamicsComparisonChart {
  distanceKm: number[];
  timeS: number[];
  /** Keyed by compared system (A/C/D); B is the zero reference and is omitted. */
  series: Record<string, DynamicsComparisonErrorSeries>;
  final: Record<string, DynamicsComparisonDelta>;
}

export interface DynamicsComparisonPlayback {
  epochIso: string;
  multiplier: number;
  /** Opaque CZML packet array handed straight to Cesium's CzmlDataSource. */
  czml: unknown[];
  /** Dense state of the reference B, sampled at the clock time for the readout. */
  samples: TrajectorySample[];
}

export interface DynamicsComparisonResult {
  ok: true;
  durationS: number;
  /** The requested horizon; > durationS when the rollout truncated (hit ground). */
  requestedDurationS: number;
  dtS: number;
  aircraftType: string;
  /** Total stored runs after this one was persisted. */
  historyCount: number;
  systems: DynamicsComparisonSystem[];
  playback: DynamicsComparisonPlayback;
  chart: DynamicsComparisonChart;
}

/** Backend-averaged history: all averaging is done server-side. */
export interface DynamicsComparisonAverage {
  runCount: number;
  systems: DynamicsComparisonSystem[];
  chart: DynamicsComparisonChart;
}

interface DynamicsComparisonErrorResponse {
  ok: false;
  error: string;
}

export async function runDynamicsComparison(
  request: DynamicsComparisonRequest,
): Promise<DynamicsComparisonResult> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/dynamics-comparison/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  const data = (await response.json()) as unknown;

  if (!response.ok) {
    const message = readError(data) ?? `AeroViz backend returned ${response.status}`;
    throw new Error(message);
  }
  return parseResult(data);
}

function readError(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const response = value as Partial<DynamicsComparisonErrorResponse>;
  return response.ok === false && typeof response.error === "string"
    ? response.error
    : null;
}

function parseResult(value: unknown): DynamicsComparisonResult {
  if (!isRecord(value) || value.ok !== true) {
    throw new Error("AeroViz backend returned an invalid dynamics-comparison response");
  }
  if (!Array.isArray(value.systems)) {
    throw new Error("AeroViz backend dynamics-comparison response is missing systems");
  }
  const playback = parsePlayback(value.playback);
  return {
    ok: true,
    durationS: readNumber(value, "durationS"),
    requestedDurationS: readNumber(value, "requestedDurationS"),
    dtS: readNumber(value, "dtS"),
    aircraftType: readString(value, "aircraftType"),
    historyCount: readNumber(value, "historyCount"),
    systems: value.systems.map(parseSystem),
    playback,
    chart: parseChart(value.chart),
  };
}

/** Total stored comparison runs (GET; safe to call before any run this session). */
export async function fetchDynamicsComparisonHistoryCount(): Promise<number> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/dynamics-comparison/history`);
  const data = (await response.json()) as unknown;
  if (!response.ok || !isRecord(data)) {
    throw new Error(`AeroViz backend returned ${response.status}`);
  }
  return readNumber(data, "historyCount");
}

/** Average all stored runs on the backend; throws if there is no history. */
export async function averageDynamicsComparisonHistory(): Promise<DynamicsComparisonAverage> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/dynamics-comparison/history/average`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const data = (await response.json()) as unknown;
  if (!response.ok) {
    throw new Error(readError(data) ?? `AeroViz backend returned ${response.status}`);
  }
  if (!isRecord(data) || data.ok !== true) {
    throw new Error(readError(data) ?? "No comparison history to average yet.");
  }
  if (!Array.isArray(data.systems)) {
    throw new Error("AeroViz backend dynamics-comparison average is missing systems");
  }
  return {
    runCount: readNumber(data, "runCount"),
    systems: data.systems.map(parseSystem),
    chart: parseChart(data.chart),
  };
}

/** Delete all stored runs; returns the new (zero) count. */
export async function clearDynamicsComparisonHistory(): Promise<number> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/dynamics-comparison/history/clear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const data = (await response.json()) as unknown;
  if (!response.ok || !isRecord(data)) {
    throw new Error(`AeroViz backend returned ${response.status}`);
  }
  return readNumber(data, "historyCount");
}

function parseSystem(value: unknown): DynamicsComparisonSystem {
  if (!isRecord(value)) {
    throw new Error("AeroViz backend dynamics-comparison response has an invalid system");
  }
  const color = value.colorRgba;
  if (!Array.isArray(color) || color.length !== 4 || !color.every((c) => typeof c === "number")) {
    throw new Error("AeroViz backend dynamics-comparison system has an invalid colorRgba");
  }
  return {
    key: readString(value, "key"),
    label: readString(value, "label"),
    colorRgba: color as [number, number, number, number],
    isReference: value.isReference === true,
  };
}

function parsePlayback(value: unknown): DynamicsComparisonPlayback {
  if (!isRecord(value) || !Array.isArray(value.czml) || !Array.isArray(value.samples)) {
    throw new Error("AeroViz backend dynamics-comparison response has invalid playback");
  }
  return {
    epochIso: readString(value, "epochIso"),
    multiplier: readNumber(value, "multiplier"),
    czml: value.czml,
    samples: value.samples.map(parseSample),
  };
}

function parseSample(value: unknown): TrajectorySample {
  if (!isRecord(value)) {
    throw new Error("AeroViz backend dynamics-comparison playback has an invalid sample");
  }
  const sample: TrajectorySample = {
    t: readNumber(value, "t"),
    lon: readNumber(value, "lon"),
    lat: readNumber(value, "lat"),
    altM: readNumber(value, "altM"),
    speedMps: readNumber(value, "speedMps"),
    headingDeg: readNumber(value, "headingDeg"),
    flightPathDeg: readNumber(value, "flightPathDeg"),
    bankDeg: readNumber(value, "bankDeg"),
    thrustN: readNumber(value, "thrustN"),
    segmentIndex: readNumber(value, "segmentIndex"),
    liftCoefficient: readNumber(value, "liftCoefficient"),
    dragCoefficient: readNumber(value, "dragCoefficient"),
    actualLoadFactor: readNumber(value, "actualLoadFactor"),
  };
  const loadFactor = readOptionalNumber(value, "loadFactor");
  if (loadFactor !== null) sample.loadFactor = loadFactor;
  const attackDeg = readOptionalNumber(value, "attackDeg");
  if (attackDeg !== null) sample.attackDeg = attackDeg;
  return sample;
}

function parseChart(value: unknown): DynamicsComparisonChart {
  if (!isRecord(value) || !Array.isArray(value.distanceKm) || !Array.isArray(value.timeS)) {
    throw new Error("AeroViz backend dynamics-comparison response has invalid chart");
  }
  if (!isRecord(value.series) || !isRecord(value.final)) {
    throw new Error("AeroViz backend dynamics-comparison chart is missing series/final");
  }
  const series: Record<string, DynamicsComparisonErrorSeries> = {};
  for (const [key, raw] of Object.entries(value.series)) {
    series[key] = parseErrorSeries(raw);
  }
  const final: DynamicsComparisonDeltas = {};
  for (const [key, raw] of Object.entries(value.final)) {
    if (!isRecord(raw)) {
      throw new Error("AeroViz backend dynamics-comparison final has an invalid entry");
    }
    final[key] = {
      horiz: readNumber(raw, "horiz"),
      alt: readNumber(raw, "alt"),
      head: readNumber(raw, "head"),
      speed: readNumber(raw, "speed"),
      fpa: readNumber(raw, "fpa"),
    };
  }
  return {
    distanceKm: value.distanceKm.map(asFiniteNumber),
    timeS: value.timeS.map(asFiniteNumber),
    series,
    final,
  };
}

function parseErrorSeries(value: unknown): DynamicsComparisonErrorSeries {
  if (!isRecord(value)) {
    throw new Error("AeroViz backend dynamics-comparison series has an invalid entry");
  }
  return {
    horiz: readNumberArray(value, "horiz"),
    alt: readNumberArray(value, "alt"),
    head: readNumberArray(value, "head"),
    speed: readNumberArray(value, "speed"),
    fpa: readNumberArray(value, "fpa"),
  };
}

/**
 * Sample every compared system's error (vs the reference B) at ``elapsedS`` by
 * linearly interpolating the chart's per-sample series over its ascending time
 * grid. Used to drive the Live-State readout during a Compare playback, so each
 * state row can show B's value plus A/C/D's live deviation. Clamps at both ends;
 * returns null only when the chart has no samples.
 */
export function interpolateComparisonDeltas(
  chart: DynamicsComparisonChart,
  elapsedS: number,
): DynamicsComparisonDeltas | null {
  const times = chart.timeS;
  if (times.length === 0) return null;

  // Locate the bracketing interval [low, high] and the blend fraction f.
  let low: number;
  let high: number;
  let f: number;
  if (elapsedS <= times[0]) {
    low = high = 0;
    f = 0;
  } else if (elapsedS >= times[times.length - 1]) {
    low = high = times.length - 1;
    f = 0;
  } else {
    low = 0;
    high = times.length - 1;
    while (high - low > 1) {
      const mid = (low + high) >> 1;
      if (times[mid] <= elapsedS) low = mid;
      else high = mid;
    }
    const span = times[high] - times[low];
    f = span > 1e-9 ? (elapsedS - times[low]) / span : 0;
  }

  const lerp = (values: number[]) => values[low] + (values[high] - values[low]) * f;
  const deltas: DynamicsComparisonDeltas = {};
  for (const [key, series] of Object.entries(chart.series)) {
    deltas[key] = {
      horiz: lerp(series.horiz),
      alt: lerp(series.alt),
      head: lerp(series.head),
      speed: lerp(series.speed),
      fpa: lerp(series.fpa),
    };
  }
  return deltas;
}
