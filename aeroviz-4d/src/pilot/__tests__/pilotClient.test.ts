import { afterEach, describe, expect, it, vi } from "vitest";
import {
  resetPilotSimulation,
  stepPilotSimulation,
  type PilotControls,
  type PilotResetState,
} from "../pilotClient";

const state: PilotResetState = {
  lon: -114.0203,
  lat: 51.1139,
  altM: 1084,
  speedMps: 135,
  headingDeg: 12,
  flightPathDeg: -3,
  massKg: 12000,
};

const control: PilotControls = {
  thrustN: 15000,
  bankDeg: 5,
  attackDeg: 4,
};

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
    // This locks the frontend reset request shape to the Python /reset endpoint.
    const fetchMock = mockFetch(snapshot);

    const result = await resetPilotSimulation(state, control);

    expect(result).toEqual(snapshot);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/reset",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, control }),
      }),
    );
  });

  it("posts step payload using control and dt only", async () => {
    // This locks the frontend step request shape to the Python /step endpoint.
    const fetchMock = mockFetch(snapshot);

    await stepPilotSimulation(control, 0.2);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/step",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ control, dtS: 0.2 }),
      }),
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
