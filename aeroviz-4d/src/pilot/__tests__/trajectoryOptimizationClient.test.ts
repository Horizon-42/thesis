import { afterEach, describe, expect, it, vi } from "vitest";
import {
  runTrajectoryOptimization,
  optimizerToParts,
  partsToOptimizer,
  validFittingsForDynamics,
  type TrajectoryOptimizationRequest,
} from "../trajectoryOptimizationClient";

describe("optimizer dynamics × fitting decomposition", () => {
  it("round-trips every direct-collocation optimizer through parts", () => {
    const optimizers = [
      "casadiDirectCollocation",
      "casadiDirectCollocationTrapezoidal",
      "casadiDirectCollocationRk4",
      "casadiDirectCollocationReanchoredEnu",
      "casadiDirectCollocationLocalEnu",
      "casadiDirectCollocationLocalEnuTrapezoidal",
      "casadiDirectCollocationLocalEnuHermiteSimpson",
      "casadiDirectCollocationNormalized",
      "casadiDirectCollocationNormalizedTrapezoidal",
      "casadiDirectCollocationNormalizedRk4",
    ] as const;
    for (const opt of optimizers) {
      const { dynamics, fitting } = optimizerToParts(opt);
      expect(partsToOptimizer(dynamics, fitting)).toBe(opt);
    }
  });

  it("maps geodetic + fitting to the right scheme", () => {
    expect(partsToOptimizer("geodetic", "hermiteSimpson")).toBe("casadiDirectCollocation");
    expect(partsToOptimizer("geodetic", "trapezoidal")).toBe("casadiDirectCollocationTrapezoidal");
    expect(partsToOptimizer("geodetic", "shooting")).toBe("casadiDirectCollocationRk4");
  });

  it("local ENU is continuous too: takes every fitting; only re-anchored ENU is shooting-only", () => {
    expect(validFittingsForDynamics("geodetic")).toEqual([
      "hermiteSimpson", "trapezoidal", "shooting",
    ]);
    expect(validFittingsForDynamics("localEnu")).toEqual([
      "hermiteSimpson", "trapezoidal", "shooting",
    ]);
    expect(validFittingsForDynamics("reanchoredEnu")).toEqual(["shooting"]);
    // localEnu composes with each fitting.
    expect(partsToOptimizer("localEnu", "hermiteSimpson")).toBe("casadiDirectCollocationLocalEnuHermiteSimpson");
    expect(partsToOptimizer("localEnu", "trapezoidal")).toBe("casadiDirectCollocationLocalEnuTrapezoidal");
    expect(partsToOptimizer("localEnu", "shooting")).toBe("casadiDirectCollocationLocalEnu");
    // The re-anchored stepper snaps any polynomial request to shooting.
    expect(partsToOptimizer("reanchoredEnu", "hermiteSimpson")).toBe("casadiDirectCollocationReanchoredEnu");
  });

  it("normalized geodetic is a continuous RHS: takes every fitting", () => {
    // Same geodetic RHS, just a metric-position change of decision variables, so
    // the normalized dynamics composes with each fitting.
    expect(validFittingsForDynamics("geodeticNormalized")).toEqual([
      "hermiteSimpson", "trapezoidal", "shooting",
    ]);
    expect(partsToOptimizer("geodeticNormalized", "hermiteSimpson")).toBe(
      "casadiDirectCollocationNormalized",
    );
    expect(partsToOptimizer("geodeticNormalized", "trapezoidal")).toBe(
      "casadiDirectCollocationNormalizedTrapezoidal",
    );
    expect(partsToOptimizer("geodeticNormalized", "shooting")).toBe(
      "casadiDirectCollocationNormalizedRk4",
    );
    expect(optimizerToParts("casadiDirectCollocationNormalizedRk4")).toEqual({
      dynamics: "geodeticNormalized",
      fitting: "shooting",
    });
  });
});

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
};

describe("trajectoryOptimizationClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the optimization request to the backend", async () => {
    const responsePayload = {
      ok: true,
      optimizer: "variableTimeWarmStartTranscription",
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

    // No `playback` in the response → parsed result carries `playback: null`.
    expect(result).toEqual({ ...responsePayload, playback: null });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/optimization/run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }),
    );
  });

  it("parses CasADi optimizer load-factor controls", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        ok: true,
        optimizer: "casadiIpopt",
        finalTimeS: 72,
        nSegments: 4,
        dtS: 0.2,
        controls: [
          { thrustN: 12000, bankDeg: 1, loadFactor: 1.2 },
        ],
        states: [
          request.targetState,
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    const result = await runTrajectoryOptimization({
      ...request,
      optimizer: "casadiIpopt",
    });

    expect(result.optimizer).toBe("casadiIpopt");
    expect(result.controls[0]).toEqual({
      thrustN: 12000,
      bankDeg: 1,
      attackDeg: 0,
      loadFactor: 1.2,
    });
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
