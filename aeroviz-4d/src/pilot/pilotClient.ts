export interface PilotControls {
  thrustN: number;
  bankDeg: number;
  attackDeg: number;
}

export type PilotAircraftType = string;

export interface PilotAircraftConfig {
  code: PilotAircraftType;
  name: string;
  category: string;
  massKg: number;
  wingAreaM2: number;
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
  state: PilotResetState;
  control: PilotControls;
  aero: {
    liftCoefficient: number;
    dragCoefficient: number;
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
): Promise<PilotSnapshot> {
  return postPilot("/simulation/reset", { state, control });
}

export async function stepPilotSimulation(
  control: PilotControls,
  dtS: number,
): Promise<PilotSnapshot> {
  return postPilot("/simulation/step", { control, dtS });
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

  return {
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
    control: {
      thrustN: readNumber(control, "thrustN"),
      bankDeg: readNumber(control, "bankDeg"),
      attackDeg: readNumber(control, "attackDeg"),
    },
    aero: {
      liftCoefficient: readNumber(aero, "liftCoefficient"),
      dragCoefficient: readNumber(aero, "dragCoefficient"),
    },
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
