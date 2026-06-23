import { AEROVIZ_BACKEND_URL, type PilotResetState } from "./pilotClient";
import {
  asFiniteNumber,
  isRecord,
  readNumber,
  readNumberArray,
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
}

export interface DynamicsComparisonChart {
  distanceKm: number[];
  timeS: number[];
  /** Keyed by compared system (A/C/D); B is the zero reference and is omitted. */
  series: Record<string, DynamicsComparisonErrorSeries>;
  final: Record<string, { horiz: number; alt: number; head: number; speed: number }>;
}

export interface DynamicsComparisonPlayback {
  epochIso: string;
  multiplier: number;
  /** Opaque CZML packet array handed straight to Cesium's CzmlDataSource. */
  czml: unknown[];
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
  if (!isRecord(value) || !Array.isArray(value.czml)) {
    throw new Error("AeroViz backend dynamics-comparison response has invalid playback");
  }
  return {
    epochIso: readString(value, "epochIso"),
    multiplier: readNumber(value, "multiplier"),
    czml: value.czml,
  };
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
  const final: Record<string, { horiz: number; alt: number; head: number; speed: number }> = {};
  for (const [key, raw] of Object.entries(value.final)) {
    if (!isRecord(raw)) {
      throw new Error("AeroViz backend dynamics-comparison final has an invalid entry");
    }
    final[key] = {
      horiz: readNumber(raw, "horiz"),
      alt: readNumber(raw, "alt"),
      head: readNumber(raw, "head"),
      speed: readNumber(raw, "speed"),
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
  };
}
