import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  knotsToMetresPerSecond,
  targetAltitudeMForThreshold,
} from "../../pilot/trajectoryTargetConstraints";
import { krduR05lyDocument } from "../../data/__tests__/krduR05lyDetailDocument.fixture";
import type {
  PilotControls,
  PilotResetState,
  PilotSimulationMode,
} from "../../pilot/pilotClient";
import PilotPanel from "../PilotPanel";

const a320Config = {
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
};

const mocks = vi.hoisted(() => ({
  airport: { code: "KRDU", lon: -78.7873, lat: 35.878659, height: 15000 },
  fetchPilotAircraftConfigs: vi.fn(),
  resetPilotSimulation: vi.fn(),
  stepPilotSimulation: vi.fn(),
  fetchRunwayThresholdTargets: vi.fn(),
  fetchRnavInitialFixCandidates: vi.fn(),
  fetchJson: vi.fn(),
  runTrajectoryOptimization: vi.fn(),
  usePilotAircraft: vi.fn(),
  usePilotInitialPlacement: vi.fn(),
  usePilotTargetGate: vi.fn(),
  useOptimizedTrajectoryPlayback: vi.fn(),
  useDynamicsComparisonPlayback: vi.fn(),
  runDynamicsComparison: vi.fn(),
  openWorkerSession: vi.fn(),
  closeWorkerSession: vi.fn(),
  beaconCloseWorkerSession: vi.fn(),
  setSelectedRunway: vi.fn(),
  setProceduresOpen: vi.fn(),
  toggleLayer: vi.fn(),
  layers: { procedures: false } as Record<string, boolean>,
  setPilotTransport: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    activeAirportCode: "KRDU",
    airport: mocks.airport,
    viewer: null,
    selectedRunway: null,
    setSelectedRunway: mocks.setSelectedRunway,
    proceduresOpen: false,
    setProceduresOpen: mocks.setProceduresOpen,
    layers: mocks.layers,
    toggleLayer: mocks.toggleLayer,
    setPilotTransport: mocks.setPilotTransport,
  }),
}));

vi.mock("../../hooks/usePilotAircraft", () => ({
  usePilotAircraft: mocks.usePilotAircraft,
}));

vi.mock("../../hooks/usePilotInitialPlacement", () => ({
  usePilotInitialPlacement: mocks.usePilotInitialPlacement,
}));

vi.mock("../../hooks/usePilotTargetGate", () => ({
  usePilotTargetGate: mocks.usePilotTargetGate,
}));

vi.mock("../../hooks/useOptimizedTrajectoryPlayback", () => ({
  useOptimizedTrajectoryPlayback: mocks.useOptimizedTrajectoryPlayback,
}));

vi.mock("../../hooks/useDynamicsComparisonPlayback", () => ({
  useDynamicsComparisonPlayback: mocks.useDynamicsComparisonPlayback,
}));

vi.mock("../../pilot/dynamicsComparisonClient", () => ({
  runDynamicsComparison: mocks.runDynamicsComparison,
}));

vi.mock("../../pilot/workerSessionClient", () => ({
  openWorkerSession: mocks.openWorkerSession,
  closeWorkerSession: mocks.closeWorkerSession,
  beaconCloseWorkerSession: mocks.beaconCloseWorkerSession,
}));

vi.mock("../../pilot/pilotClient", () => ({
  AEROVIZ_BACKEND_URL: "http://127.0.0.1:8765",
  fetchPilotAircraftConfigs: mocks.fetchPilotAircraftConfigs,
  resetPilotSimulation: mocks.resetPilotSimulation,
  stepPilotSimulation: mocks.stepPilotSimulation,
}));

vi.mock("../../data/runwayThresholdTargets", () => ({
  fetchRunwayThresholdTargets: mocks.fetchRunwayThresholdTargets,
}));

vi.mock("../../data/rnavInitialFixCandidates", () => ({
  fetchRnavInitialFixCandidates: mocks.fetchRnavInitialFixCandidates,
}));

// PilotPanel uses fetchJson only for the procedure detail document (the constrained optimizer).
vi.mock("../../utils/fetchJson", () => ({
  fetchJson: mocks.fetchJson,
}));

vi.mock("../../pilot/trajectoryOptimizationClient", async (importOriginal) => ({
  // Keep the real pure helpers (decomposeOptimizer / composeOptimizer /
  // validFittingsForFrame); only stub the network call.
  ...(await importOriginal<typeof import("../../pilot/trajectoryOptimizationClient")>()),
  runTrajectoryOptimization: mocks.runTrajectoryOptimization,
}));

describe("PilotPanel trajectory play mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.fetchPilotAircraftConfigs.mockResolvedValue([
      a320Config,
    ]);
    mocks.fetchRunwayThresholdTargets.mockResolvedValue([
      {
        id: "RW05L",
        runwayIdent: "RW05L",
        runwayPairIdent: "05L/23R",
        lon: -78.802,
        lat: 35.874,
        altM: 111.86,
        psiDeg: 45,
      },
    ]);
    mocks.fetchRnavInitialFixCandidates.mockResolvedValue([
      {
        key: "KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4",
        runwayIdent: "RW05L",
        procedureUid: "KRDU-R05LY-RW05L",
        procedureIdent: "R05LY",
        chartName: "RNAV(GPS) Y RWY 05L",
        branchId: "branch:R",
        branchIdent: "R",
        fixId: "fix:SCHOO",
        fixIdent: "SCHOO",
        nextFixId: "fix:WEPAS",
        nextFixIdent: "WEPAS",
        lon: -78.92647222,
        lat: 35.77341389,
        altM: 914.4,
        headingDeg: 39.1,
      },
    ]);
    mocks.fetchJson.mockResolvedValue(krduR05lyDocument);
    mocks.runTrajectoryOptimization.mockResolvedValue({
      ok: true,
      optimizer: "casadiIpopt",
      finalTimeS: 100,
      nSegments: 10,
      arrivalTimeS: 100,
      dtS: 0.2,
      controls: [
        { thrustN: 12000, bankDeg: 0, attackDeg: 0, loadFactor: 1.2 },
        { thrustN: 12000, bankDeg: 3, attackDeg: 0, loadFactor: 1.1 },
      ],
      states: [],
      playback: {
        epochIso: "2026-01-01T00:00:00Z",
        multiplier: 1,
        czml: [{ id: "document" }, { id: "optimized-trajectory-aircraft" }],
        samples: [
          {
            t: 0,
            lon: -78.79,
            lat: 35.87,
            altM: 900,
            speedMps: 115,
            headingDeg: 5,
            flightPathDeg: -1,
            bankDeg: 0,
            thrustN: 12000,
            segmentIndex: 0,
            liftCoefficient: 0.42,
            dragCoefficient: 0.041,
            actualLoadFactor: 1.2,
            loadFactor: 1.2,
          },
        ],
      },
    });
    // The real playback hook drives the live readout by sampling the rollout on
    // each Cesium clock tick. Here we stand in for that: when enabled, emit the
    // first sample once so the readout renders without a real Cesium clock.
    mocks.useOptimizedTrajectoryPlayback.mockImplementation((params: {
      enabled: boolean;
      samples: unknown[];
      onSample: (sample: unknown) => void;
    }) => {
      useEffect(() => {
        params.onSample(
          params.enabled && params.samples.length > 0 ? params.samples[0] : null,
        );
      }, [params.enabled, params.samples]);
      return { status: params.enabled ? "loaded" : "idle", error: null };
    });
    mocks.resetPilotSimulation.mockImplementation((
      _state: PilotResetState,
      control: PilotControls,
      simulationMode: PilotSimulationMode = "alpha",
    ) => Promise.resolve({
      ok: true,
      elapsedS: 0,
      simulationMode,
      state: {
        lon: -78.7873,
        lat: 35.878659,
        altM: 1000,
        speedMps: 120,
        headingDeg: 0,
        flightPathDeg: 0,
        massKg: 78000,
        aircraftType: "A320",
      },
      control,
      aero: {
        liftCoefficient: 0.4,
        dragCoefficient: 0.04,
        actualLoadFactor: control.loadFactor ?? 1.1,
      },
    }));
    mocks.stepPilotSimulation.mockImplementation((
      control: PilotControls,
      dtS: number,
      simulationMode: PilotSimulationMode = "alpha",
    ) => Promise.resolve({
      ok: true,
      elapsedS: dtS,
      simulationMode,
      state: {
        lon: -78.79,
        lat: 35.87,
        altM: 900,
        speedMps: 115,
        headingDeg: 5,
        flightPathDeg: -1,
        massKg: 78000,
        aircraftType: "A320",
      },
      control,
      aero: {
        liftCoefficient: 0.42,
        dragCoefficient: 0.041,
        actualLoadFactor: control.loadFactor ?? 1.12,
      },
    }));
  });

  it("keeps the casadi worker resident while the Optimize tab is open and releases it on close", async () => {
    mocks.openWorkerSession.mockResolvedValue(undefined);
    mocks.closeWorkerSession.mockResolvedValue(undefined);

    const optimize = render(<PilotPanel mode="trajectory" />);
    await waitFor(() => {
      expect(mocks.openWorkerSession).toHaveBeenCalledWith("optimizer");
    });
    // Leaving the tab (unmount) decommissions the worker.
    optimize.unmount();
    expect(mocks.closeWorkerSession).toHaveBeenCalledWith("optimizer");
  });

  it("hides backend URL and switches pilot controls between alpha and load factor", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    expect(screen.queryByText("http://127.0.0.1:8765")).toBeNull();

    const simulationSelect = screen.getByRole("combobox", {
      name: "Simulation",
    }) as HTMLSelectElement;
    expect(simulationSelect.value).toBe("alpha");
    const alphaInput = screen.getByLabelText("Alpha") as HTMLInputElement;
    expect(alphaInput).toBeTruthy();
    expect(screen.queryByLabelText("Load factor")).toBeNull();

    fireEvent.keyDown(simulationSelect, { key: "ArrowUp" });
    expect(simulationSelect.value).toBe("alpha");
    expect(alphaInput.value).toBe("6.283");

    fireEvent.change(simulationSelect, {
      target: { value: "loadFactor" },
    });

    expect(screen.queryByLabelText("Alpha")).toBeNull();
    const loadFactorInput = screen.getByLabelText("Load factor") as HTMLInputElement;
    expect(loadFactorInput.value).toBe("1.414214");

    fireEvent.keyDown(simulationSelect, { key: "ArrowUp" });
    expect(simulationSelect.value).toBe("loadFactor");
    expect(loadFactorInput.value).toBe("1.464214");

    fireEvent.change(loadFactorInput, { target: { value: "1.25" } });
    fireEvent.blur(loadFactorInput);
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));

    await waitFor(() => {
      expect(mocks.resetPilotSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          aircraftType: "A320",
          massKg: 78000,
        }),
        expect.objectContaining({
          loadFactor: 1.25,
        }),
        "loadFactor",
      );
    });
  });

  it("switches pilot controls to load factor for casadi simulation", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    const simulationSelect = screen.getByRole("combobox", {
      name: "Simulation",
    }) as HTMLSelectElement;

    fireEvent.change(simulationSelect, {
      target: { value: "casadi" },
    });

    expect(simulationSelect.value).toBe("casadi");
    expect(screen.queryByLabelText("Alpha")).toBeNull();
    const loadFactorInput = screen.getByLabelText("Load factor") as HTMLInputElement;
    expect(loadFactorInput.value).toBe("1.414214");

    fireEvent.change(loadFactorInput, { target: { value: "1.15" } });
    fireEvent.blur(loadFactorInput);
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));

    await waitFor(() => {
      expect(mocks.resetPilotSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          aircraftType: "A320",
          massKg: 78000,
        }),
        expect.objectContaining({
          loadFactor: 1.15,
        }),
        "casadi",
      );
    });
  });

  it("opens trajectory play from pilot mode and submits runway-target optimization", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    expect(screen.getByText("Trajectory Play")).toBeTruthy();
    expect(await screen.findByText("Target State")).toBeTruthy();
    expect(screen.getByText("RW05L")).toBeTruthy();

    // The optimizer is edited as orthogonal axes.  The DEFAULT is the procedure-constrained
    // mode; this test exercises a plain (unconstrained) optimized run, so switch Constraints
    // to "none" (direct initial -> target, no procedure).
    expect((screen.getByRole("combobox", { name: "Constraints" }) as HTMLSelectElement).value)
      .toBe("procedure");
    expect((screen.getByRole("combobox", { name: "Fitting" }) as HTMLSelectElement).value)
      .toBe("trapezoidal");
    expect((screen.getByLabelText("Max iter") as HTMLInputElement).value).toBe("300");
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "none" },
    });

    await waitFor(() => {
      expect(mocks.usePilotTargetGate).toHaveBeenLastCalledWith({
        enabled: true,
        target: expect.objectContaining({
          runwayThresholdId: "RW05L",
          runwayIdent: "RW05L",
          lon: -78.802,
          lat: 35.874,
          altM: targetAltitudeMForThreshold(111.86, a320Config),
          headingDeg: 45,
        }),
      });
    });
    await waitFor(() => {
      expect(mocks.fetchRnavInitialFixCandidates).toHaveBeenCalledWith(
        "KRDU",
        "RW05L",
      );
    });

    const initialSummary = screen.getByLabelText("Initial aircraft state summary");
    fireEvent.click(within(initialSummary).getByRole("button", { name: "Edit" }));
    const initialEditor = await screen.findByLabelText("Initial aircraft setup");
    const rnavIfSelect = within(initialEditor).getByRole("combobox", {
      name: "RNAV IF",
    });
    expect(rnavIfSelect.textContent).toContain(
      "SCHOO to WEPAS | R05LY | R | 914.4 m",
    );
    fireEvent.change(rnavIfSelect, {
      target: { value: "KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4" },
    });
    fireEvent.click(within(initialEditor).getByRole("button", { name: "Close" }));

    const targetSummary = screen.getByLabelText("Target aircraft state summary");
    fireEvent.click(within(targetSummary).getByRole("button", { name: "Edit" }));

    const targetEditor = await screen.findByLabelText("Target state setup");
    expect(within(targetEditor).getByText("Threshold Gate")).toBeTruthy();
    expect(within(targetEditor).getByRole("combobox").textContent).toContain("RW05L");

    const gammaInput = within(targetEditor).getByLabelText("Gamma deg");
    fireEvent.change(gammaInput, { target: { value: "-4" } });
    fireEvent.blur(gammaInput);

    const segmentInput = screen.getByLabelText("Control segments");
    fireEvent.change(segmentInput, { target: { value: "12" } });
    fireEvent.blur(segmentInput);
    const arrivalInput = screen.getByLabelText("Arrival time");
    fireEvent.change(arrivalInput, { target: { value: "96" } });
    fireEvent.blur(arrivalInput);
    const trajectoryDtInput = screen.getByLabelText("dt");
    expect((trajectoryDtInput as HTMLInputElement).value).toBe("0.5");
    fireEvent.change(trajectoryDtInput, { target: { value: "5" } });
    fireEvent.blur(trajectoryDtInput);
    fireEvent.click(within(targetEditor).getByRole("button", { name: "Close" }));

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    const calls = mocks.runTrajectoryOptimization.mock.calls;
    const request = calls[calls.length - 1]?.[0];
    expect(request).toMatchObject({
      initialState: expect.objectContaining({
        lon: -78.92647222,
        lat: 35.77341389,
        altM: 914.4,
        speedMps: knotsToMetresPerSecond(a320Config.terminalSpeedKt + 25),
        flightPathDeg: 0,
        massKg: 78000,
        aircraftType: "A320",
      }),
      targetState: expect.objectContaining({
        lon: -78.802,
        lat: 35.874,
        altM: targetAltitudeMForThreshold(111.86, a320Config),
        speedMps: knotsToMetresPerSecond(a320Config.terminalSpeedKt),
        headingDeg: 45,
        flightPathDeg: -4,
        aircraftType: "A320",
      }),
      nSegments: 12,
      arrivalTimeS: 96,
      dtS: 2,
      maxIterations: 300,
    });
    // Unconstrained run: the request carries nSegments (whole-trajectory control count)
    // and NO procedure — not the per-leg nSegPerPhase / procedureConstraint.
    expect(request.procedureConstraint).toBeUndefined();
    expect(request.nSegPerPhase).toBeUndefined();
    expect(request.initialState.headingDeg).toBeCloseTo(39.1);
  });

  it("chooses the optimizer via frame × fitting; ENU frame forces shooting", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    await waitFor(() => {
      expect(mocks.fetchRnavInitialFixCandidates).toHaveBeenCalledWith(
        "KRDU",
        "RW05L",
      );
    });

    const constraintsSelect = screen.getByRole("combobox", { name: "Constraints" }) as HTMLSelectElement;
    // The Frame select lives in the collapsed "Advanced dynamics" <details>; jsdom
    // renders its content, so it is queryable without opening the disclosure.
    const frameSelect = screen.getByRole("combobox", { name: "Frame" }) as HTMLSelectElement;
    const fittingSelect = screen.getByRole("combobox", { name: "Fitting" }) as HTMLSelectElement;
    const arrivalInput = screen.getByLabelText("Arrival time") as HTMLInputElement;

    // Default: procedure-constrained + Trapezoidal; all three fittings available,
    // but the advanced frame axis is LOCKED (the constrained mode pins the dynamics).
    expect(constraintsSelect.value).toBe("procedure");
    expect(fittingSelect.value).toBe("trapezoidal");
    expect(fittingSelect.querySelectorAll("option").length).toBe(3);
    expect(fittingSelect.disabled).toBe(false);
    expect(arrivalInput.disabled).toBe(false);
    expect(frameSelect.disabled).toBe(true);

    // Unconstrained (direct) mode unlocks the frame axis.
    fireEvent.change(constraintsSelect, { target: { value: "none" } });
    expect(frameSelect.disabled).toBe(false);
    expect(frameSelect.value).toBe("geodetic");

    // Geodetic supports all three fittings; arrival time stays editable.
    for (const fitting of ["trapezoidal", "shooting", "hermiteSimpson"]) {
      fireEvent.change(fittingSelect, { target: { value: fitting } });
      expect(fittingSelect.value).toBe(fitting);
      expect(arrivalInput.disabled).toBe(false);
    }

    // Re-anchored ENU is discrete (re-anchors each step) -> shooting-only.
    fireEvent.change(frameSelect, { target: { value: "reanchoredEnu" } });
    expect(fittingSelect.value).toBe("shooting");
    expect(fittingSelect.querySelectorAll("option").length).toBe(1);
    expect(fittingSelect.disabled).toBe(true);

    // Local ENU @ target is a continuous RHS too -> all three fittings, like
    // geodetic.  (Its dynamics differ; the transcription is free to vary.)
    fireEvent.change(frameSelect, { target: { value: "localEnu" } });
    expect(fittingSelect.querySelectorAll("option").length).toBe(3);
    expect(fittingSelect.disabled).toBe(false);
    fireEvent.change(fittingSelect, { target: { value: "hermiteSimpson" } });
    expect(fittingSelect.value).toBe("hermiteSimpson");
  });

  it("preserves the hidden Advanced axes across a Constraints toggle", async () => {
    // REGRESSION: the panel holds the optimizer AXES as state (not the composed wire string, which
    // can't encode frame/transport/normalized in constrained mode). So picking a non-default frame,
    // flipping Constraints to procedure and back, must return the SAME frame — not silently reset.
    render(<PilotPanel />);
    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    const constraintsSelect = screen.getByRole("combobox", { name: "Constraints" }) as HTMLSelectElement;
    const frameSelect = screen.getByRole("combobox", { name: "Frame" }) as HTMLSelectElement;

    fireEvent.change(constraintsSelect, { target: { value: "none" } });
    fireEvent.change(frameSelect, { target: { value: "localEnu" } });
    expect(frameSelect.value).toBe("localEnu");

    fireEvent.change(constraintsSelect, { target: { value: "procedure" } });   // frame locked/hidden
    fireEvent.change(constraintsSelect, { target: { value: "none" } });        // ...and back
    expect(frameSelect.value).toBe("localEnu");                                 // choice survived
  });

  it("drives the target runway's procedure display in Optimize+constrained, and restores on exit", async () => {
    render(<PilotPanel />);
    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    // Reactive (not a one-shot): entering Optimize with the DEFAULT constrained mode
    // drives the target runway's procedures on — no manual switch needed. Scopes to
    // the target runway (BARE form), opens the panel, enables the geometry layer.
    await waitFor(() => {
      expect(mocks.setProceduresOpen).toHaveBeenCalledWith(true);
    });
    expect(mocks.setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(mocks.toggleLayer).toHaveBeenCalledWith("procedures");

    mocks.setProceduresOpen.mockClear();
    mocks.setSelectedRunway.mockClear();

    // Turning constraints OFF restores the user's pre-force display (the mock's own
    // state: procedures closed / no runway) — so procedures are no longer forced.
    const constraintsSelect = screen.getByRole("combobox", { name: "Constraints" }) as HTMLSelectElement;
    fireEvent.change(constraintsSelect, { target: { value: "none" } });
    await waitFor(() => {
      expect(mocks.setProceduresOpen).toHaveBeenCalledWith(false);
    });
    expect(mocks.setSelectedRunway).toHaveBeenCalledWith(null);
  });

  it("clamps trajectory target speed and heading to threshold constraints", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    const targetSummary = await screen.findByLabelText("Target aircraft state summary");
    fireEvent.click(within(targetSummary).getByRole("button", { name: "Edit" }));

    const targetEditor = await screen.findByLabelText("Target state setup");
    const speedInput = within(targetEditor).getByLabelText("Vt m/s");
    fireEvent.change(speedInput, { target: { value: "100" } });
    fireEvent.blur(speedInput);
    const headingInput = within(targetEditor).getByLabelText("Psi deg");
    fireEvent.change(headingInput, { target: { value: "50" } });
    fireEvent.blur(headingInput);
    fireEvent.click(within(targetEditor).getByRole("button", { name: "Close" }));

    // Default optimizer is the procedure-constrained mode (needs an RNAV approach); this
    // test only checks target clamping, so switch to the unconstrained (direct) optimizer.
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "none" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    const calls = mocks.runTrajectoryOptimization.mock.calls;
    const request = calls[calls.length - 1]?.[0];
    expect(request.targetState.speedMps).toBeCloseTo(
      knotsToMetresPerSecond(a320Config.terminalSpeedMaxKt),
    );
    expect(request.targetState.headingDeg).toBe(46);
  });

  it("anchors the constrained optimizer target on the procedure's CIFP threshold", async () => {
    // REGRESSION: the multiphase target must be the procedure's OWN threshold (the backend
    // anchors the constraint (n, e) frame at the target and rejects a procedure ending
    // elsewhere) — NOT the runway.geojson pavement midpoint, which sits hundreds of metres off
    // on displaced-threshold runways. The mocked runway target here is deliberately offset from
    // the CIFP threshold so the assertion discriminates the two sources.
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    await waitFor(() => {
      expect(mocks.fetchRnavInitialFixCandidates).toHaveBeenCalled();
    });

    // Select the RNAV initial fix (identifies the procedure + branch).
    const initialSummary = screen.getByLabelText("Initial aircraft state summary");
    fireEvent.click(within(initialSummary).getByRole("button", { name: "Edit" }));
    const initialEditor = await screen.findByLabelText("Initial aircraft setup");
    fireEvent.change(within(initialEditor).getByLabelText("RNAV IF"), {
      target: { value: "KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4" },
    });
    fireEvent.click(within(initialEditor).getByRole("button", { name: "Close" }));

    // Default Constraints = procedure (the multiphase optimizer).
    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });

    const calls = mocks.runTrajectoryOptimization.mock.calls;
    const request = calls[calls.length - 1]?.[0];
    expect(request.procedureConstraint?.waypoints).toHaveLength(3);
    // CIFP threshold (fixture: RW05L), not the mocked runway.geojson target (-78.802, 35.874).
    expect(request.targetState.lat).toBeCloseTo(35.87445, 8);
    expect(request.targetState.lon).toBeCloseTo(-78.80196389, 8);
    // altitude = CIFP threshold elevation + the aircraft's threshold-crossing height
    expect(request.targetState.altM).toBeCloseTo(
      targetAltitudeMForThreshold(367 * 0.3048, a320Config),
      4,
    );
    // final course WEPAS -> RW05L in the simulator heading convention (0 = East, CCW)
    expect(request.targetState.headingDeg).toBeCloseTo(45.0, 0);
  });

  it("keeps the RNAV fix selection on a custom start so the optimizer flies TO the fix", async () => {
    // REGRESSION: editing the initial state used to CLEAR the RNAV IF selection, which made a
    // custom-start constrained optimize impossible (the panel demanded a selection, and
    // re-selecting snapped the start back onto the fix). The selector names the PROCEDURE; the
    // start is independent — the backend adds a transition phase + the fix-passage disc.
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    await waitFor(() => {
      expect(mocks.fetchRnavInitialFixCandidates).toHaveBeenCalled();
    });

    const initialSummary = screen.getByLabelText("Initial aircraft state summary");
    fireEvent.click(within(initialSummary).getByRole("button", { name: "Edit" }));
    const initialEditor = await screen.findByLabelText("Initial aircraft setup");
    const rnavSelect = within(initialEditor).getByLabelText("RNAV IF") as HTMLSelectElement;
    fireEvent.change(rnavSelect, {
      target: { value: "KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4" },
    });

    // Simulate a map placement ~8 km away from SCHOO — the selection must survive it.
    const placementCalls = mocks.usePilotInitialPlacement.mock.calls;
    const placement = placementCalls[placementCalls.length - 1]?.[0];
    act(() => {
      placement.onPositionChange({ lon: -78.95, lat: 35.84 });
    });
    expect(rnavSelect.value).toBe("KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4");
    // ...and a field edit must survive too.
    const altInput = within(initialEditor).getByLabelText("Alt m");
    fireEvent.change(altInput, { target: { value: "1500" } });
    fireEvent.blur(altInput);
    expect(rnavSelect.value).toBe("KRDU-R05LY-RW05L|branch:R|fix:SCHOO|fix:WEPAS|914.4");
    fireEvent.click(within(initialEditor).getByRole("button", { name: "Close" }));

    // Constrained optimize now runs from the CUSTOM start with the procedure attached.
    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    const calls = mocks.runTrajectoryOptimization.mock.calls;
    const request = calls[calls.length - 1]?.[0];
    expect(request.procedureConstraint?.waypoints).toHaveLength(3);
    expect(request.initialState.lat).toBeCloseTo(35.84, 6);
    expect(request.initialState.lon).toBeCloseTo(-78.95, 6);
    expect(request.initialState.altM).toBe(1500);
  });

  it("keeps the initial aircraft editor open after placing the aircraft", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    const initialSummary = screen.getByLabelText("Initial aircraft state summary");
    fireEvent.click(within(initialSummary).getByRole("button", { name: "Edit" }));

    const initialEditor = await screen.findByLabelText("Initial aircraft setup");
    fireEvent.click(within(initialEditor).getByRole("button", { name: "Place Aircraft" }));

    await waitFor(() => {
      expect(mocks.usePilotInitialPlacement).toHaveBeenLastCalledWith(
        expect.objectContaining({
          enabled: true,
          previewVisible: true,
          onFinish: expect.any(Function),
        }),
      );
    });

    const calls = mocks.usePilotInitialPlacement.mock.calls;
    const placementOptions = calls[calls.length - 1]?.[0];
    if (!placementOptions) {
      throw new Error("usePilotInitialPlacement was not called");
    }

    act(() => {
      placementOptions.onFinish();
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Initial aircraft setup")).toBeTruthy();
    });
    expect(
      within(screen.getByLabelText("Initial aircraft setup")).getByRole("button", {
        name: "Place Aircraft",
      }),
    ).toBeTruthy();
  });

  it("keeps the initial placement preview visible while replacing an existing snapshot", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));

    await waitFor(() => {
      expect(mocks.resetPilotSimulation).toHaveBeenCalled();
    });

    const initialSummary = screen.getByLabelText("Initial aircraft state summary");
    fireEvent.click(within(initialSummary).getByRole("button", { name: "Edit" }));
    const initialEditor = await screen.findByLabelText("Initial aircraft setup");
    fireEvent.click(within(initialEditor).getByRole("button", { name: "Place Aircraft" }));

    await waitFor(() => {
      expect(mocks.usePilotInitialPlacement).toHaveBeenLastCalledWith(
        expect.objectContaining({
          enabled: true,
          previewVisible: true,
        }),
      );
    });
  });

  it("plays the optimized trajectory on the Cesium clock and shows the sampled control", async () => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    // Default is the procedure-constrained mode; this test plays a plain (unconstrained) run.
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "none" },
    });
    fireEvent.click(await screen.findByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled)
        .toBe(false);
    });

    // Derived loading: the CZML is loaded (hook enabled) as soon as a result
    // exists — BEFORE pressing Play — so the shared bottom transport bar / native
    // clock dial are bound to it (the hook loads it paused; Play animates).
    expect(mocks.useOptimizedTrajectoryPlayback).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: true }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    // Playback is handed to Cesium's clock via the CZML — the frontend no longer
    // steps the dynamics segment-by-segment.
    expect(mocks.stepPilotSimulation).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(mocks.useOptimizedTrajectoryPlayback).toHaveBeenLastCalledWith(
        expect.objectContaining({
          enabled: true,
          czml: expect.any(Array),
          samples: expect.arrayContaining([
            expect.objectContaining({ segmentIndex: 0 }),
          ]),
          follow: expect.any(Boolean),
          onSample: expect.any(Function),
        }),
      );
    });

    const liveStatePanel = await screen.findByRole("complementary", {
      name: "Realtime aircraft state",
    });
    expect(within(liveStatePanel).getByText("Control")).toBeTruthy();
    expect(
      within(liveStatePanel).getByText("bank 0.0 deg | n 1.20 | thrust 12000 N"),
    ).toBeTruthy();
  });

  it("renders the live target-deviation as compare-style chips (not separate error rows)", async () => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
    mocks.runTrajectoryOptimization.mockResolvedValueOnce({
      ok: true,
      optimizer: "transcription",
      finalTimeS: 0.2,
      nSegments: 1,
      dtS: 0.02,
      controls: [{ thrustN: 12000, bankDeg: 0, attackDeg: 4 }],
      states: [],
      playback: {
        epochIso: "2026-01-01T00:00:00Z",
        multiplier: 1,
        czml: [{ id: "document" }],
        samples: [
          {
            t: 0,
            lon: -78.79,
            lat: 35.87,
            altM: 900,
            speedMps: 115,
            headingDeg: 5,
            flightPathDeg: -1,
            bankDeg: 0,
            thrustN: 12000,
            segmentIndex: 0,
            liftCoefficient: 0.42,
            dragCoefficient: 0.041,
            actualLoadFactor: 1.12,
            attackDeg: 4,
          },
        ],
      },
    });

    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    // Default is the procedure-constrained mode; this test exercises a plain optimized run.
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "none" },
    });
    fireEvent.click(await screen.findByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled)
        .toBe(false);
    });

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    const panel = await screen.findByRole("complementary", {
      name: "Realtime aircraft state",
    });
    // The deviation is shown as the compare-style chip strip + Horiz Err row,
    // not the old separate "Lat/Lon/Alt Error" rows.
    expect(within(panel).getByText("Horiz Err (vs target)")).toBeTruthy();
    expect(within(panel).queryByText("Lat Error")).toBeNull();
    expect(panel.querySelectorAll(".pilot-realtime-delta").length).toBeGreaterThan(0);
  });

  it("surfaces the per-leg constraint hint when Constraints = procedure", async () => {
    render(<PilotPanel />);
    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    expect(await screen.findByText("Target State")).toBeTruthy();

    const hint = "Per-leg constraints from the selected RNAV approach";
    // Procedure-constrained is the DEFAULT -> the per-leg hint is shown on open.
    expect(screen.getByText(hint)).toBeTruthy();

    // Switching Constraints to "none" (direct) hides it.
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "none" },
    });
    expect(screen.queryByText(hint)).toBeNull();

    // Switching back to "procedure" shows it again.
    fireEvent.change(screen.getByRole("combobox", { name: "Constraints" }), {
      target: { value: "procedure" },
    });
    expect(screen.getByText(hint)).toBeTruthy();
  });
});
