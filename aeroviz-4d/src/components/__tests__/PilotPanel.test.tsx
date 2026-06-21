import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  runTrajectoryOptimization: vi.fn(),
  usePilotAircraft: vi.fn(),
  usePilotInitialPlacement: vi.fn(),
  usePilotTargetGate: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    activeAirportCode: "KRDU",
    airport: mocks.airport,
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

vi.mock("../../pilot/pilotClient", () => ({
  AEROVIZ_BACKEND_URL: "http://127.0.0.1:8765",
  fetchPilotAircraftConfigs: mocks.fetchPilotAircraftConfigs,
  resetPilotSimulation: mocks.resetPilotSimulation,
  stepPilotSimulation: mocks.stepPilotSimulation,
}));

vi.mock("../../data/runwayThresholdTargets", () => ({
  fetchRunwayThresholdTargets: mocks.fetchRunwayThresholdTargets,
}));

vi.mock("../../pilot/trajectoryOptimizationClient", () => ({
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

    expect((screen.getByRole("combobox", { name: "Optimizer" }) as HTMLSelectElement).value)
      .toBe("casadiIpopt");
    expect((screen.getByLabelText("Max iter") as HTMLInputElement).value).toBe("300");
    const optimizerSelect = screen.getByRole("combobox", { name: "Optimizer" });
    fireEvent.change(optimizerSelect, {
      target: { value: "warmStartTranscription" },
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
    fireEvent.change(trajectoryDtInput, { target: { value: "0.1" } });
    fireEvent.blur(trajectoryDtInput);
    fireEvent.click(within(targetEditor).getByRole("button", { name: "Close" }));

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalledWith({
        optimizer: "warmStartTranscription",
        initialState: expect.objectContaining({
          lon: -78.7873,
          lat: 35.878659,
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
        dtS: 0.1,
        maxIterations: 300,
      });
    });
  });

  it("disables arrival time input for variable-time warm start optimizer", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    const optimizerSelect = screen.getByRole("combobox", { name: "Optimizer" });
    const arrivalInput = screen.getByLabelText("Arrival time") as HTMLInputElement;

    expect(arrivalInput.disabled).toBe(false);
    fireEvent.change(optimizerSelect, {
      target: { value: "variableTimeWarmStartTranscription" },
    });
    expect(arrivalInput.disabled).toBe(true);

    fireEvent.change(optimizerSelect, {
      target: { value: "warmStartTranscription" },
    });
    expect(arrivalInput.disabled).toBe(false);
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

  it("shows the trajectory replay control row in the live state panel", async () => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    fireEvent.click(await screen.findByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled)
        .toBe(false);
    });

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    await waitFor(() => {
      expect(mocks.resetPilotSimulation).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(mocks.resetPilotSimulation).toHaveBeenCalledWith(
        expect.objectContaining({ aircraftType: "A320" }),
        { thrustN: 12000, bankDeg: 0, attackDeg: 0, loadFactor: 1.2 },
        "casadi",
      );
    });
    await waitFor(() => {
      expect(mocks.stepPilotSimulation).toHaveBeenCalledWith(
        { thrustN: 12000, bankDeg: 0, attackDeg: 0, loadFactor: 1.2 },
        0.2,
        "casadi",
        0.2,
      );
    });
    await waitFor(() => {
      expect(mocks.usePilotAircraft).toHaveBeenLastCalledWith(
        expect.objectContaining({
          trail: expect.arrayContaining([
            expect.objectContaining({ segmentIndex: 0 }),
          ]),
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

  it("keeps the final trajectory state panel visible after replay finishes", async () => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
    mocks.runTrajectoryOptimization.mockResolvedValueOnce({
      ok: true,
      optimizer: "transcription",
      finalTimeS: 0.2,
      nSegments: 1,
      dtS: 0.02,
      controls: [{ thrustN: 12000, bankDeg: 0, attackDeg: 4 }],
      states: [],
    });

    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));
    fireEvent.click(await screen.findByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Play" }) as HTMLButtonElement).disabled)
        .toBe(false);
    });

    fireEvent.click(screen.getByRole("button", { name: "Play" }));

    await waitFor(() => {
      expect(mocks.stepPilotSimulation).toHaveBeenCalledWith(
        { thrustN: 12000, bankDeg: 0, attackDeg: 4 },
        0.2,
        "alpha",
        0.02,
      );
    });
    await waitFor(() => {
      expect((screen.getByRole("button", { name: "Pause" }) as HTMLButtonElement).disabled)
        .toBe(true);
    });

    const finalStatePanel = screen.getByRole("complementary", {
      name: "Realtime aircraft state",
    });
    expect(within(finalStatePanel).getByText("Lat Error")).toBeTruthy();
    expect(within(finalStatePanel).getByText("-0.004000 deg")).toBeTruthy();
    expect(within(finalStatePanel).getByText("+0.012000 deg")).toBeTruthy();
    expect(within(finalStatePanel).getByText("+773.1 m")).toBeTruthy();
  });
});
