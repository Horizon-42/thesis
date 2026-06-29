import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  knotsToMetresPerSecond,
  targetAltitudeMForThreshold,
} from "../../pilot/trajectoryTargetConstraints";
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
  runTrajectoryOptimization: vi.fn(),
  usePilotAircraft: vi.fn(),
  usePilotInitialPlacement: vi.fn(),
  usePilotTargetGate: vi.fn(),
  useOptimizedTrajectoryPlayback: vi.fn(),
  useDynamicsComparisonPlayback: vi.fn(),
  runDynamicsComparison: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    activeAirportCode: "KRDU",
    airport: mocks.airport,
    viewer: null,
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

vi.mock("../../pilot/trajectoryOptimizationClient", async (importOriginal) => ({
  // Keep the real pure helpers (optimizerToParts / partsToOptimizer /
  // validFittingsForDynamics); only stub the network call.
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

    // Optimizer is chosen as two dropdowns: dynamics × fitting.  Default dynamics is the
    // multiphase mode; this test exercises the plain geodetic Rk4 flow, so switch to it.
    expect((screen.getByRole("combobox", { name: "Dynamics" }) as HTMLSelectElement).value)
      .toBe("geodeticMultiphase");
    expect((screen.getByRole("combobox", { name: "Fitting" }) as HTMLSelectElement).value)
      .toBe("hermiteSimpson");
    expect((screen.getByLabelText("Max iter") as HTMLInputElement).value).toBe("300");
    // geodetic + shooting => casadiDirectCollocationRk4.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodetic" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Fitting" }), {
      target: { value: "shooting" },
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

    const segmentInput = screen.getByLabelText("Segments");
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
      optimizer: "casadiDirectCollocationRk4",
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
    expect(request.initialState.headingDeg).toBeCloseTo(39.1);
  });

  it("chooses the optimizer as dynamics × fitting; ENU dynamics force shooting", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    await waitFor(() => {
      expect(mocks.fetchRnavInitialFixCandidates).toHaveBeenCalledWith(
        "KRDU",
        "RW05L",
      );
    });

    const dynamicsSelect = screen.getByRole("combobox", { name: "Dynamics" }) as HTMLSelectElement;
    const fittingSelect = screen.getByRole("combobox", { name: "Fitting" }) as HTMLSelectElement;
    const arrivalInput = screen.getByLabelText("Arrival time") as HTMLInputElement;

    // Default: multiphase + Hermite-Simpson; all three fittings available.
    expect(dynamicsSelect.value).toBe("geodeticMultiphase");
    expect(fittingSelect.value).toBe("hermiteSimpson");
    expect(fittingSelect.querySelectorAll("option").length).toBe(3);
    expect(fittingSelect.disabled).toBe(false);
    expect(arrivalInput.disabled).toBe(false);

    // geodetic supports all three fittings; arrival time stays editable.
    for (const fitting of ["trapezoidal", "shooting", "hermiteSimpson"]) {
      fireEvent.change(fittingSelect, { target: { value: fitting } });
      expect(fittingSelect.value).toBe(fitting);
      expect(arrivalInput.disabled).toBe(false);
    }

    // Re-anchored ENU is discrete (re-anchors each step) -> shooting-only.
    fireEvent.change(dynamicsSelect, { target: { value: "reanchoredEnu" } });
    expect(fittingSelect.value).toBe("shooting");
    expect(fittingSelect.querySelectorAll("option").length).toBe(1);
    expect(fittingSelect.disabled).toBe(true);

    // Local ENU @ target is a continuous RHS too -> all three fittings, like
    // geodetic.  (Its dynamics differ; the transcription is free to vary.)
    fireEvent.change(dynamicsSelect, { target: { value: "localEnu" } });
    expect(fittingSelect.querySelectorAll("option").length).toBe(3);
    expect(fittingSelect.disabled).toBe(false);
    fireEvent.change(fittingSelect, { target: { value: "hermiteSimpson" } });
    expect(fittingSelect.value).toBe("hermiteSimpson");
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

    // Default optimizer is the multiphase mode (needs an RNAV approach); this test only checks
    // target clamping, so use the plain geodetic optimizer.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodetic" },
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
    // Default is the multiphase mode; this test plays a plain (unconstrained) optimized run.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodetic" },
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
    // Default is the multiphase mode; this test exercises a plain optimized run.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodetic" },
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

  it("offers a multiphase dynamics that surfaces the per-leg constraint hint", async () => {
    render(<PilotPanel />);
    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    expect(await screen.findByText("Target State")).toBeTruthy();

    const hint = "Per-leg constraints from the selected RNAV approach";
    // Multiphase is the DEFAULT dynamics -> the per-leg hint is shown on open.
    expect(screen.getByText(hint)).toBeTruthy();

    // Switching to a non-multiphase dynamics hides it.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodetic" },
    });
    expect(screen.queryByText(hint)).toBeNull();

    // Switching back shows it again.
    fireEvent.change(screen.getByRole("combobox", { name: "Dynamics" }), {
      target: { value: "geodeticMultiphase" },
    });
    expect(screen.getByText(hint)).toBeTruthy();
  });
});
