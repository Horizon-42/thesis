import { AEROVIZ_BACKEND_URL, type PilotControls, type PilotResetState } from "./pilotClient";

export type TrajectoryOptimizer =
  // Direct-collocation variants, one per defect "fitting equation".
  // The bare name keeps the default (Hermite-Simpson).
  | "casadiDirectCollocation"
  | "casadiDirectCollocationTrapezoidal"
  | "casadiDirectCollocationHermiteSimpson"
  | "casadiDirectCollocationRk4"
  // Legacy optimizers: still served by the backend, no longer offered in
  // the UI (kept here so a response naming one still parses).
  | "casadiIpopt"
  | "transcription"
  | "leastSquaresTranscription"
  | "warmStartTranscription"
  | "variableTimeWarmStartTranscription"
  | "singleShooting";

export interface TrajectoryOptimizationRequest {
  optimizer: TrajectoryOptimizer;
  initialState: PilotResetState;
  targetState: PilotResetState;
  nSegments: number;
  arrivalTimeS: number;
  dtS: number;
  maxIterations: number;
}

/**
 * One dense rollout sample of the optimized trajectory. The backend rolls the
 * optimizer's controls forward once and returns these so the frontend can drive
 * the live readout by sampling at the current Cesium clock time.
 */
export interface TrajectorySample {
  t: number;
  lon: number;
  lat: number;
  altM: number;
  speedMps: number;
  headingDeg: number;
  flightPathDeg: number;
  bankDeg: number;
  thrustN: number;
  segmentIndex: number;
  liftCoefficient: number;
  dragCoefficient: number;
  actualLoadFactor: number;
  loadFactor?: number;
  attackDeg?: number;
}

/**
 * Playback bundle: a CZML document the frontend loads into a CzmlDataSource and
 * plays on Cesium's own clock (like a downloaded trajectory), plus the dense
 * sample series backing the live readout. `czml` is opaque to TypeScript — it
 * is handed straight to Cesium's CzmlDataSource.load().
 */
export interface TrajectoryPlayback {
  epochIso: string;
  multiplier: number;
  czml: unknown[];
  samples: TrajectorySample[];
}

export interface TrajectoryOptimizationResult {
  ok: true;
  optimizer: TrajectoryOptimizer;
  finalTimeS: number;
  nSegments: number;
  dtS: number;
  controls: PilotControls[];
  states: PilotResetState[];
  playback: TrajectoryPlayback | null;
}

interface TrajectoryOptimizationErrorResponse {
  ok: false;
  error: string;
}

export async function runTrajectoryOptimization(
  request: TrajectoryOptimizationRequest,
): Promise<TrajectoryOptimizationResult> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/optimization/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  const data = await response.json() as unknown;

  if (!response.ok) {
    const message = readOptimizationError(data) ??
      `AeroViz backend returned ${response.status}`;
    throw new Error(message);
  }
  return parseTrajectoryOptimizationResult(data);
}

function readOptimizationError(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const response = value as Partial<TrajectoryOptimizationErrorResponse>;
  return response.ok === false && typeof response.error === "string"
    ? response.error
    : null;
}

function parseTrajectoryOptimizationResult(
  value: unknown,
): TrajectoryOptimizationResult {
  if (!isRecord(value) || value.ok !== true) {
    throw new Error("AeroViz backend returned an invalid optimization response");
  }
  if (!Array.isArray(value.controls) || !Array.isArray(value.states)) {
    throw new Error("AeroViz backend optimization response is missing samples");
  }

  return {
    ok: true,
    optimizer: readOptimizer(value),
    finalTimeS: readNumber(value, "finalTimeS"),
    nSegments: readNumber(value, "nSegments"),
    dtS: readPositiveNumber(value, "dtS"),
    controls: value.controls.map(parseControl),
    states: value.states.map(parseState),
    playback: parsePlayback(value.playback),
  };
}

function parsePlayback(value: unknown): TrajectoryPlayback | null {
  if (value === undefined || value === null) return null;
  if (!isRecord(value)) {
    throw new Error("AeroViz backend optimization response has invalid playback");
  }
  if (!Array.isArray(value.czml) || !Array.isArray(value.samples)) {
    throw new Error("AeroViz backend optimization playback is missing czml/samples");
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
    throw new Error("AeroViz backend optimization playback has invalid sample");
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

function readOptimizer(value: Record<string, unknown>): TrajectoryOptimizer {
  const nested = value.optimizer;
  if (
    nested === "casadiDirectCollocation" ||
    nested === "casadiDirectCollocationTrapezoidal" ||
    nested === "casadiDirectCollocationHermiteSimpson" ||
    nested === "casadiDirectCollocationRk4" ||
    nested === "casadiIpopt" ||
    nested === "transcription" ||
    nested === "leastSquaresTranscription" ||
    nested === "warmStartTranscription" ||
    nested === "variableTimeWarmStartTranscription" ||
    nested === "singleShooting"
  ) {
    return nested;
  }
  throw new Error("AeroViz backend optimization response has invalid optimizer");
}

function parseControl(value: unknown): PilotControls {
  if (!isRecord(value)) {
    throw new Error("AeroViz backend optimization response has invalid control");
  }
  const attackDeg = readOptionalNumber(value, "attackDeg");
  const loadFactor = readOptionalNumber(value, "loadFactor");
  if (attackDeg === null && loadFactor === null) {
    throw new Error("AeroViz backend optimization response has invalid control");
  }
  return {
    thrustN: readNumber(value, "thrustN"),
    bankDeg: readNumber(value, "bankDeg"),
    attackDeg: attackDeg ?? 0,
    ...(loadFactor === null ? {} : { loadFactor }),
  };
}

function parseState(value: unknown): PilotResetState {
  if (!isRecord(value)) {
    throw new Error("AeroViz backend optimization response has invalid state");
  }
  return {
    lon: readNumber(value, "lon"),
    lat: readNumber(value, "lat"),
    altM: readNumber(value, "altM"),
    speedMps: readNumber(value, "speedMps"),
    headingDeg: readNumber(value, "headingDeg"),
    flightPathDeg: readNumber(value, "flightPathDeg"),
    massKg: readNumber(value, "massKg"),
    aircraftType: readString(value, "aircraftType"),
  };
}

function readNumber(value: Record<string, unknown>, key: string): number {
  const nested = value[key];
  if (typeof nested !== "number" || !Number.isFinite(nested)) {
    throw new Error(`AeroViz backend optimization response has invalid ${key}`);
  }
  return nested;
}

function readOptionalNumber(value: Record<string, unknown>, key: string): number | null {
  if (!(key in value)) return null;
  return readNumber(value, key);
}

function readPositiveNumber(value: Record<string, unknown>, key: string): number {
  const number = readNumber(value, key);
  if (number <= 0) {
    throw new Error(`AeroViz backend optimization response has invalid ${key}`);
  }
  return number;
}

function readString(value: Record<string, unknown>, key: string): string {
  const nested = value[key];
  if (typeof nested !== "string" || nested.length === 0) {
    throw new Error(`AeroViz backend optimization response has invalid ${key}`);
  }
  return nested;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
