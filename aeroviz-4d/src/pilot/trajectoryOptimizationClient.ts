import { AEROVIZ_BACKEND_URL, type PilotControls, type PilotResetState } from "./pilotClient";
import {
  isRecord,
  readNumber,
  readOptionalNumber,
  readPositiveNumber,
  readString,
} from "./responseValidators";

export type TrajectoryOptimizer =
  // Direct-collocation variants, one per defect "fitting equation".
  // The bare name keeps the default (Hermite-Simpson).
  | "casadiDirectCollocation"
  | "casadiDirectCollocationTrapezoidal"
  | "casadiDirectCollocationHermiteSimpson"
  | "casadiDirectCollocationRk4"
  | "casadiDirectCollocationReanchoredEnu"
  | "casadiDirectCollocationLocalEnu"
  | "casadiDirectCollocationLocalEnuTrapezoidal"
  | "casadiDirectCollocationLocalEnuHermiteSimpson"
  // Normalized geodetic: same geodetic RHS, but the decision STATE is metric
  // position offsets from the target, so the NLP is well-conditioned and the
  // solve is robust on loose arrival windows / finer meshes. The fitting is
  // selectable; the bare name is Hermite-Simpson.
  | "casadiDirectCollocationNormalized"
  | "casadiDirectCollocationNormalizedTrapezoidal"
  | "casadiDirectCollocationNormalizedRk4"
  // Legacy optimizers: still served by the backend, no longer offered in
  // the UI (kept here so a response naming one still parses).
  | "casadiIpopt"
  | "transcription"
  | "leastSquaresTranscription"
  | "warmStartTranscription"
  | "variableTimeWarmStartTranscription"
  | "singleShooting";

// ── Two-axis view of the direct-collocation family ─────────────────────────
// The backend still takes a single ``optimizer`` string, but the UI exposes it
// as two orthogonal choices that compose into that string:
//   dynamics  — what vector field / stepper drives the defect
//   fitting   — how the defect is formed (transcription)
// The polynomial fittings (trapezoidal / Hermite-Simpson) need a CONTINUOUS
// RHS, so they apply to the geodetic dynamics AND the fixed local-ENU dynamics
// (the flat point-mass RHS in a fixed tangent frame).  Only the *re-anchored*
// ENU dynamics is discrete (it re-anchors every step), so it is shooting-only.
// ``shooting`` is the RK4 integral defect of the chosen dynamics.
// ``geodeticNormalized`` is the geodetic RHS again, but the decision STATE is
// metric position offsets from the target (a pure change of variables, no
// physics change), so the NLP is well-conditioned and the solve is robust on
// loose arrival windows / finer meshes.  It is a continuous RHS, so it takes
// every fitting (Hermite-Simpson / trapezoidal / shooting) just like the plain
// geodetic dynamics; only the decision-state scaling differs.
export type OptimizerDynamics = "geodetic" | "reanchoredEnu" | "localEnu" | "geodeticNormalized";
export type OptimizerFitting = "hermiteSimpson" | "trapezoidal" | "shooting";

const COMBO_TO_OPTIMIZER: Record<string, TrajectoryOptimizer> = {
  "geodetic|hermiteSimpson": "casadiDirectCollocation",
  "geodetic|trapezoidal": "casadiDirectCollocationTrapezoidal",
  "geodetic|shooting": "casadiDirectCollocationRk4",
  "reanchoredEnu|shooting": "casadiDirectCollocationReanchoredEnu",
  "localEnu|hermiteSimpson": "casadiDirectCollocationLocalEnuHermiteSimpson",
  "localEnu|trapezoidal": "casadiDirectCollocationLocalEnuTrapezoidal",
  "localEnu|shooting": "casadiDirectCollocationLocalEnu",
  "geodeticNormalized|hermiteSimpson": "casadiDirectCollocationNormalized",
  "geodeticNormalized|trapezoidal": "casadiDirectCollocationNormalizedTrapezoidal",
  "geodeticNormalized|shooting": "casadiDirectCollocationNormalizedRk4",
};

const OPTIMIZER_TO_COMBO: Record<string, { dynamics: OptimizerDynamics; fitting: OptimizerFitting }> = {
  casadiDirectCollocation: { dynamics: "geodetic", fitting: "hermiteSimpson" },
  casadiDirectCollocationHermiteSimpson: { dynamics: "geodetic", fitting: "hermiteSimpson" },
  casadiDirectCollocationTrapezoidal: { dynamics: "geodetic", fitting: "trapezoidal" },
  casadiDirectCollocationRk4: { dynamics: "geodetic", fitting: "shooting" },
  casadiDirectCollocationReanchoredEnu: { dynamics: "reanchoredEnu", fitting: "shooting" },
  casadiDirectCollocationLocalEnu: { dynamics: "localEnu", fitting: "shooting" },
  casadiDirectCollocationLocalEnuHermiteSimpson: { dynamics: "localEnu", fitting: "hermiteSimpson" },
  casadiDirectCollocationLocalEnuTrapezoidal: { dynamics: "localEnu", fitting: "trapezoidal" },
  casadiDirectCollocationNormalized: { dynamics: "geodeticNormalized", fitting: "hermiteSimpson" },
  casadiDirectCollocationNormalizedTrapezoidal: { dynamics: "geodeticNormalized", fitting: "trapezoidal" },
  casadiDirectCollocationNormalizedRk4: { dynamics: "geodeticNormalized", fitting: "shooting" },
};

/** Fittings valid for a given dynamics: only the re-anchored ENU stepper is
 * discrete (shooting-only).  Geodetic, fixed local-ENU, and the normalized
 * geodetic dynamics all refine a continuous RHS, so they take every fitting. */
export function validFittingsForDynamics(dynamics: OptimizerDynamics): OptimizerFitting[] {
  if (dynamics === "reanchoredEnu") return ["shooting"];
  return ["hermiteSimpson", "trapezoidal", "shooting"];
}

export function optimizerToParts(
  optimizer: TrajectoryOptimizer,
): { dynamics: OptimizerDynamics; fitting: OptimizerFitting } {
  return OPTIMIZER_TO_COMBO[optimizer] ?? { dynamics: "geodetic", fitting: "hermiteSimpson" };
}

/** Compose a (dynamics, fitting) pair back into the optimizer name, snapping
 * the fitting to a valid one for the dynamics if needed. */
export function partsToOptimizer(
  dynamics: OptimizerDynamics,
  fitting: OptimizerFitting,
): TrajectoryOptimizer {
  const valid = validFittingsForDynamics(dynamics);
  const chosen = valid.includes(fitting) ? fitting : valid[0];
  return COMBO_TO_OPTIMIZER[`${dynamics}|${chosen}`];
}

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
    nested === "casadiDirectCollocationReanchoredEnu" ||
    nested === "casadiDirectCollocationLocalEnu" ||
    nested === "casadiDirectCollocationLocalEnuTrapezoidal" ||
    nested === "casadiDirectCollocationLocalEnuHermiteSimpson" ||
    nested === "casadiDirectCollocationNormalized" ||
    nested === "casadiDirectCollocationNormalizedTrapezoidal" ||
    nested === "casadiDirectCollocationNormalizedRk4" ||
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

