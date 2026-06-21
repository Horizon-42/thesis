export interface PilotControls {
  thrustN: number;
  bankDeg: number;
  attackDeg: number;
  loadFactor?: number;
}

export type PilotSimulationMode = "alpha" | "loadFactor" | "casadi";

type PilotControlPayload =
  | {
      thrustN: number;
      bankDeg: number;
      attackDeg: number;
    }
  | {
      thrustN: number;
      bankDeg: number;
      loadFactor: number;
    };

export type PilotAircraftType = string;

export interface PilotAircraftConfig {
  code: PilotAircraftType;
  name: string;
  category: string;
  massKg: number;
  wingAreaM2: number;
  maxThrustN: number;
  approachThrustGuessN: number;
  terminalSpeedKt: number;
  terminalSpeedMinKt: number;
  terminalSpeedMaxKt: number;
  finalApproachMinNm: number;
  finalApproachMaxNm: number;
  finalApproachLateralHalfWidthNm: number;
  finalApproachGlideAngleDeg: number;
  thresholdCrossingHeightM: number;
}

export interface PilotResetState {
  lon: number;
  lat: number;
  altM: number;
  speedMps: number;
  headingDeg: number;
  flightPathDeg: number;
  massKg: number;
  aircraftType: PilotAircraftType;
}

export interface PilotSnapshot {
  ok: true;
  elapsedS: number;
  simulationMode?: PilotSimulationMode;
  state: PilotResetState;
  control: PilotControls;
  aero: {
    liftCoefficient: number;
    dragCoefficient: number;
    actualLoadFactor: number;
  };
}

interface PilotErrorResponse {
  ok: false;
  error: string;
}

const DEFAULT_AEROVIZ_BACKEND_URL = "http://127.0.0.1:8765";

export const AEROVIZ_BACKEND_URL =
  import.meta.env.VITE_AEROVIZ_BACKEND_URL || DEFAULT_AEROVIZ_BACKEND_URL;

export async function resetPilotSimulation(
  state: PilotResetState,
  control: PilotControls,
  simulationMode: PilotSimulationMode = "alpha",
): Promise<PilotSnapshot> {
  return postPilot("/simulation/reset", {
    state,
    control: serializePilotControl(control, simulationMode),
    simulationMode,
  });
}

export async function stepPilotSimulation(
  control: PilotControls,
  dtS: number,
  simulationMode: PilotSimulationMode = "alpha",
): Promise<PilotSnapshot> {
  return postPilot("/simulation/step", {
    control: serializePilotControl(control, simulationMode),
    dtS,
    simulationMode,
  });
}

export async function fetchPilotAircraftConfigs(): Promise<PilotAircraftConfig[]> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}/simulation/aircraft`);
  const data = await response.json() as unknown;

  if (!response.ok) {
    const message = readPilotError(data) ?? `AeroViz backend returned ${response.status}`;
    throw new Error(message);
  }
  return parsePilotAircraftConfigs(data);
}

async function postPilot(path: string, payload: unknown): Promise<PilotSnapshot> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json() as unknown;

  if (!response.ok) {
    const message = readPilotError(data) ?? `AeroViz backend returned ${response.status}`;
    throw new Error(message);
  }
  return parsePilotSnapshot(data);
}

function readPilotError(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const response = value as Partial<PilotErrorResponse>;
  return response.ok === false && typeof response.error === "string"
    ? response.error
    : null;
}

function parsePilotSnapshot(value: unknown): PilotSnapshot {
  if (!isRecord(value) || value.ok !== true) {
    throw new Error("AeroViz backend returned an invalid response");
  }

  const state = readRecord(value, "state");
  const control = readRecord(value, "control");
  const aero = readRecord(value, "aero");
  const attackDeg = readOptionalNumber(control, "attackDeg");
  const loadFactor = readOptionalNumber(control, "loadFactor");
  if (attackDeg === null && loadFactor === null) {
    throw new Error("AeroViz backend response has invalid control");
  }

  const parsedControl: PilotControls = {
    thrustN: readNumber(control, "thrustN"),
    bankDeg: readNumber(control, "bankDeg"),
    attackDeg: attackDeg ?? 0,
  };
  if (loadFactor !== null) {
    parsedControl.loadFactor = loadFactor;
  }
  const simulationMode = readOptionalSimulationMode(value);

  const snapshot: PilotSnapshot = {
    ok: true,
    elapsedS: readNumber(value, "elapsedS"),
    state: {
      lon: readNumber(state, "lon"),
      lat: readNumber(state, "lat"),
      altM: readNumber(state, "altM"),
      speedMps: readNumber(state, "speedMps"),
      headingDeg: readNumber(state, "headingDeg"),
      flightPathDeg: readNumber(state, "flightPathDeg"),
      massKg: readNumber(state, "massKg"),
      aircraftType: readString(state, "aircraftType"),
    },
    control: parsedControl,
    aero: {
      liftCoefficient: readNumber(aero, "liftCoefficient"),
      dragCoefficient: readNumber(aero, "dragCoefficient"),
      actualLoadFactor: readNumber(aero, "actualLoadFactor"),
    },
  };
  if (simulationMode !== null) {
    snapshot.simulationMode = simulationMode;
  }
  return snapshot;
}

function serializePilotControl(
  control: PilotControls,
  simulationMode: PilotSimulationMode,
): PilotControlPayload {
  if (simulationMode === "loadFactor" || simulationMode === "casadi") {
    return {
      thrustN: control.thrustN,
      bankDeg: control.bankDeg,
      loadFactor: control.loadFactor ?? 1,
    };
  }

  return {
    thrustN: control.thrustN,
    bankDeg: control.bankDeg,
    attackDeg: control.attackDeg,
  };
}

function parsePilotAircraftConfigs(value: unknown): PilotAircraftConfig[] {
  if (!isRecord(value) || value.ok !== true || !Array.isArray(value.aircraft)) {
    throw new Error("AeroViz backend returned invalid aircraft config");
  }

  return value.aircraft.map((item) => {
    if (!isRecord(item)) {
      throw new Error("AeroViz backend returned invalid aircraft config");
    }
    return {
      code: readString(item, "code"),
      name: readString(item, "name"),
      category: readString(item, "category"),
      massKg: readNumber(item, "massKg"),
      wingAreaM2: readNumber(item, "wingAreaM2"),
      maxThrustN: readNumber(item, "maxThrustN"),
      approachThrustGuessN: readNumber(item, "approachThrustGuessN"),
      terminalSpeedKt: readNumber(item, "terminalSpeedKt"),
      terminalSpeedMinKt: readNumber(item, "terminalSpeedMinKt"),
      terminalSpeedMaxKt: readNumber(item, "terminalSpeedMaxKt"),
      finalApproachMinNm: readNumber(item, "finalApproachMinNm"),
      finalApproachMaxNm: readNumber(item, "finalApproachMaxNm"),
      finalApproachLateralHalfWidthNm: readNumber(
        item,
        "finalApproachLateralHalfWidthNm",
      ),
      finalApproachGlideAngleDeg: readNumber(
        item,
        "finalApproachGlideAngleDeg",
      ),
      thresholdCrossingHeightM: readNumber(item, "thresholdCrossingHeightM"),
    };
  });
}

function readRecord(value: Record<string, unknown>, key: string): Record<string, unknown> {
  const nested = value[key];
  if (!isRecord(nested)) {
    throw new Error(`AeroViz backend response is missing ${key}`);
  }
  return nested;
}

function readNumber(value: Record<string, unknown>, key: string): number {
  const nested = value[key];
  if (typeof nested !== "number" || !Number.isFinite(nested)) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

function readOptionalNumber(value: Record<string, unknown>, key: string): number | null {
  const nested = value[key];
  if (nested === undefined) return null;
  if (typeof nested !== "number" || !Number.isFinite(nested)) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

function readOptionalSimulationMode(
  value: Record<string, unknown>,
): PilotSimulationMode | null {
  const nested = value.simulationMode;
  if (nested === undefined) return null;
  if (nested === "alpha" || nested === "loadFactor" || nested === "casadi") {
    return nested;
  }
  throw new Error("AeroViz backend response has invalid simulationMode");
}

function readString(value: Record<string, unknown>, key: string): string {
  const nested = value[key];
  if (typeof nested !== "string" || nested.length === 0) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
