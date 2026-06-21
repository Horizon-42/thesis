import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchPilotAircraftConfigs,
  resetPilotSimulation,
  stepPilotSimulation,
  type PilotControls,
  type PilotAircraftConfig,
  type PilotResetState,
} from "../pilotClient";

const state: PilotResetState = {
  lon: -114.0203,
  lat: 51.1139,
  altM: 1084,
  speedMps: 135,
  headingDeg: 12,
  flightPathDeg: -3,
  massKg: 351530,
  aircraftType: "B77W",
};

const control: PilotControls = {
  thrustN: 15000,
  bankDeg: 5,
  attackDeg: 4,
  loadFactor: 1.2,
};

const aircraftConfigs: PilotAircraftConfig[] = [
  {
    code: "A320",
    name: "Airbus A320-200",
    category: "narrow_body",
    massKg: 78000,
    wingAreaM2: 122.6,
    maxThrustN: 240000,
    approachThrustGuessN: 40000,
    terminalSpeedKt: 145,
    terminalSpeedMinKt: 135,
    terminalSpeedMaxKt: 155,
    finalApproachMinNm: 5,
    finalApproachMaxNm: 10,
    finalApproachLateralHalfWidthNm: 0.8,
    finalApproachGlideAngleDeg: 3,
    thresholdCrossingHeightM: 15,
  },
  {
    code: "B77W",
    name: "Boeing 777-300ER",
    category: "wide_body",
    massKg: 351530,
    wingAreaM2: 436.8,
    maxThrustN: 1026000,
    approachThrustGuessN: 140000,
    terminalSpeedKt: 155,
    terminalSpeedMinKt: 145,
    terminalSpeedMaxKt: 165,
    finalApproachMinNm: 6,
    finalApproachMaxNm: 12,
    finalApproachLateralHalfWidthNm: 1,
    finalApproachGlideAngleDeg: 3,
    thresholdCrossingHeightM: 15,
  },
];

const snapshot = {
  ok: true,
  elapsedS: 0.2,
  state,
  control,
  aero: {
    liftCoefficient: 0.41,
    dragCoefficient: 0.03,
  },
};

describe("pilotClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts reset payload using the simulation server contract", async () => {
    // This locks the frontend reset request shape to the Python /simulation/reset endpoint.
    const fetchMock = mockFetch(snapshot);

    const result = await resetPilotSimulation(state, control);

    expect(result).toEqual(snapshot);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/simulation/reset",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state,
          control: {
            thrustN: 15000,
            bankDeg: 5,
            attackDeg: 4,
          },
          simulationMode: "alpha",
        }),
      }),
    );
  });

  it("posts step payload using alpha control and dt by default", async () => {
    // This locks the frontend step request shape to the Python /simulation/step endpoint.
    const fetchMock = mockFetch(snapshot);

    await stepPilotSimulation(control, 0.2);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/simulation/step",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          control: {
            thrustN: 15000,
            bankDeg: 5,
            attackDeg: 4,
          },
          dtS: 0.2,
          simulationMode: "alpha",
        }),
      }),
    );
  });

  it("posts load-factor simulation mode with loadFactor control", async () => {
    const fetchMock = mockFetch(snapshot);

    await stepPilotSimulation(control, 0.2, "loadFactor");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/simulation/step",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          control: {
            thrustN: 15000,
            bankDeg: 5,
            loadFactor: 1.2,
          },
          dtS: 0.2,
          simulationMode: "loadFactor",
        }),
      }),
    );
  });

  it("posts casadi simulation mode with loadFactor control", async () => {
    const fetchMock = mockFetch({
      ...snapshot,
      simulationMode: "casadi",
      control: {
        thrustN: 15000,
        bankDeg: 5,
        loadFactor: 1.2,
      },
    });

    const result = await stepPilotSimulation(control, 0.2, "casadi");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/simulation/step",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          control: {
            thrustN: 15000,
            bankDeg: 5,
            loadFactor: 1.2,
          },
          dtS: 0.2,
          simulationMode: "casadi",
        }),
      }),
    );
    expect(result.simulationMode).toBe("casadi");
    expect(result.control.attackDeg).toBe(0);
    expect(result.control.loadFactor).toBe(1.2);
  });

  it("parses load-factor snapshots without attack angle", async () => {
    const fetchMock = mockFetch({
      ...snapshot,
      simulationMode: "loadFactor",
      control: {
        thrustN: 15000,
        bankDeg: 5,
        loadFactor: 1.2,
      },
    });

    const result = await stepPilotSimulation(control, 0.2, "loadFactor");

    expect(fetchMock).toHaveBeenCalled();
    expect(result.control.attackDeg).toBe(0);
    expect(result.control.loadFactor).toBe(1.2);
  });

  it("loads aircraft configs from the backend simulation namespace", async () => {
    const payload = { ok: true, aircraft: aircraftConfigs };
    const fetchMock = mockFetch(payload);

    const result = await fetchPilotAircraftConfigs();

    expect(result).toEqual(aircraftConfigs);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/simulation/aircraft",
    );
  });

  it("throws the server error message when the response is not ok", async () => {
    // This keeps Python validation errors visible in the Pilot Mode panel.
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: false, error: "dtS must be finite" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    await expect(stepPilotSimulation(control, Number.NaN)).rejects.toThrow(
      "dtS must be finite",
    );
  });
});

function mockFetch(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
