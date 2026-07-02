import { afterEach, describe, expect, it, vi } from "vitest";
import {
  runTrajectoryOptimization,
  decomposeOptimizer,
  composeOptimizer,
  validFittingsForFrame,
  type TrajectoryOptimizationRequest,
  type OptimizerParts,
} from "../trajectoryOptimizationClient";

describe("optimizer orthogonal-axis decomposition (the panel-facing API)", () => {
  const WIRE_OPTIMIZERS = [
    "casadiDirectCollocation",
    "casadiDirectCollocationTrapezoidal",
    "casadiDirectCollocationRk4",
    "casadiDirectCollocationReanchoredEnu",
    "casadiDirectCollocationLocalEnu",
    "casadiDirectCollocationLocalEnuHermiteSimpson",
    "casadiDirectCollocationFullTransport",
    "casadiDirectCollocationNormalized",
    "casadiDirectCollocationNormalizedFullTransport",
    "casadiMultiphaseNormalizedFullTransport",
    "casadiMultiphaseNormalizedFullTransportRk4",
  ] as const;

  it("round-trips every optimizer through the axes", () => {
    for (const opt of WIRE_OPTIMIZERS) {
      expect(composeOptimizer(decomposeOptimizer(opt))).toBe(opt);
    }
  });

  it("splits the conflated geodetic flags into independent transport + normalized axes", () => {
    expect(decomposeOptimizer("casadiDirectCollocation")).toEqual<OptimizerParts>({
      constrained: false, frame: "geodetic", transport: "approx", normalized: false, fitting: "hermiteSimpson",
    });
    expect(decomposeOptimizer("casadiDirectCollocationFullTransport")).toMatchObject({ transport: "full", normalized: false });
    expect(decomposeOptimizer("casadiDirectCollocationNormalized")).toMatchObject({ transport: "approx", normalized: true });
    expect(decomposeOptimizer("casadiDirectCollocationNormalizedFullTransport")).toMatchObject({ transport: "full", normalized: true });
  });

  it("treats procedure constraints as a MODE, not a dynamics (forces geodetic+full+normalized)", () => {
    const parts = decomposeOptimizer("casadiMultiphaseNormalizedFullTransport");
    expect(parts.constrained).toBe(true);
    expect(parts).toMatchObject({ frame: "geodetic", transport: "full", normalized: true });
    // constrained composes to the multiphase wire name regardless of the frame/transport/normalized axes.
    expect(composeOptimizer({ ...parts, frame: "localEnu", transport: "approx", normalized: false }))
      .toBe("casadiMultiphaseNormalizedFullTransport");
  });

  it("composes the axes into the right wire name (snapping fitting where a frame is shooting-only)", () => {
    expect(composeOptimizer({ constrained: false, frame: "geodetic", transport: "full", normalized: true, fitting: "trapezoidal" }))
      .toBe("casadiDirectCollocationNormalizedFullTransportTrapezoidal");
    expect(composeOptimizer({ constrained: true, frame: "geodetic", transport: "full", normalized: true, fitting: "shooting" }))
      .toBe("casadiMultiphaseNormalizedFullTransportRk4");
    expect(composeOptimizer({ constrained: false, frame: "reanchoredEnu", transport: "approx", normalized: false, fitting: "hermiteSimpson" }))
      .toBe("casadiDirectCollocationReanchoredEnu");
  });

  it("re-anchored ENU is shooting-only; other frames take every fitting", () => {
    expect(validFittingsForFrame("reanchoredEnu")).toEqual(["shooting"]);
    expect(validFittingsForFrame("geodetic")).toEqual(["hermiteSimpson", "trapezoidal", "shooting"]);
    expect(validFittingsForFrame("localEnu")).toEqual(["hermiteSimpson", "trapezoidal", "shooting"]);
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

    // No `playback`/`procedureConstraintSummary` in the response → both null.
    expect(result).toEqual({
      ...responsePayload,
      playback: null,
      procedureConstraintSummary: null,
    });
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

  it("accepts a multiphase optimizer name in the response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        ok: true,
        optimizer: "casadiMultiphaseNormalizedFullTransport",
        finalTimeS: 348,
        nSegments: 16,
        dtS: 0.2,
        controls: [{ thrustN: 12000, bankDeg: 1, loadFactor: 1.2 }],
        states: [request.targetState],
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    ));

    const result = await runTrajectoryOptimization({
      ...request,
      optimizer: "casadiMultiphaseNormalizedFullTransport",
    });

    expect(result.optimizer).toBe("casadiMultiphaseNormalizedFullTransport");
  });

  it("parses the procedure-constraint summary the backend echoes back", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        ok: true,
        optimizer: "casadiDirectCollocation",
        finalTimeS: 72,
        nSegments: 4,
        dtS: 0.2,
        controls: [{ thrustN: 12000, bankDeg: 1, loadFactor: 1.2 }],
        states: [request.targetState],
        procedureConstraintSummary: {
          waypointCount: 3,
          monotonicDescent: true,
          firstFixIdent: "SCHOO",
          lastFixIdent: "RW05L",
        },
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    const result = await runTrajectoryOptimization(request);

    expect(result.procedureConstraintSummary).toEqual({
      waypointCount: 3,
      monotonicDescent: true,
      firstFixIdent: "SCHOO",
      lastFixIdent: "RW05L",
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
