import { afterEach, describe, expect, it, vi } from "vitest";
import {
  runTrajectoryOptimization,
  type TrajectoryOptimizationRequest,
} from "../trajectoryOptimizationClient";

const request: TrajectoryOptimizationRequest = {
  optimizer: "singleShooting",
  nSegments: 4,
  arrivalTimeS: 95,
  dtS: 0.2,
  maxIterations: 25,
  initialState: {
    lon: -114.02,
    lat: 51.11,
    altM: 1000,
    speedMps: 120,
    headingDeg: 10,
    flightPathDeg: -3,
    massKg: 78000,
    aircraftType: "A320",
  },
  targetState: {
    lon: -114,
    lat: 51.1,
    altM: 900,
    speedMps: 70,
    headingDeg: 40,
    flightPathDeg: -3,
    massKg: 78000,
    aircraftType: "A320",
  },
  targetControl: {
    attackDeg: 4,
  },
};

describe("trajectoryOptimizationClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the optimization request to the backend", async () => {
    const responsePayload = {
      ok: true,
      optimizer: "leastSquaresTranscription",
      finalTimeS: 80,
      nSegments: 4,
      dtS: 0.2,
      controls: [
        { thrustN: 12000, bankDeg: 1, attackDeg: 4 },
      ],
      states: [
        request.targetState,
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responsePayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await runTrajectoryOptimization(request);

    expect(result).toEqual(responsePayload);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/optimization/run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }),
    );
  });

  it("throws the backend optimization error message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: false, error: "bad target" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    await expect(runTrajectoryOptimization(request)).rejects.toThrow("bad target");
  });
});
