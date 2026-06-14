import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PilotPanel from "../PilotPanel";

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
      {
        code: "A320",
        name: "Airbus A320-200",
        category: "narrow_body",
        massKg: 78000,
        wingAreaM2: 122.6,
      },
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
      optimizer: "singleShooting",
      finalTimeS: 100,
      nSegments: 10,
      dtS: 0.2,
      controls: [
        { thrustN: 12000, bankDeg: 0, attackDeg: 4 },
        { thrustN: 12000, bankDeg: 3, attackDeg: 4 },
      ],
      states: [],
    });
    mocks.resetPilotSimulation.mockResolvedValue({
      ok: true,
      elapsedS: 0,
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
      control: { thrustN: 12000, bankDeg: 3, attackDeg: 4 },
      aero: { liftCoefficient: 0.4, dragCoefficient: 0.04 },
    });
    mocks.stepPilotSimulation.mockResolvedValue({
      ok: true,
      elapsedS: 10,
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
      control: { thrustN: 12000, bankDeg: 3, attackDeg: 4 },
      aero: { liftCoefficient: 0.42, dragCoefficient: 0.041 },
    });
  });

  it("opens trajectory play from pilot mode and submits runway-target optimization", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    expect(screen.getByText("Trajectory Play")).toBeTruthy();
    expect(await screen.findByText("Target State")).toBeTruthy();
    expect(screen.getByText("RW05L")).toBeTruthy();

    fireEvent.change(screen.getByRole("combobox", { name: "Optimizer" }), {
      target: { value: "singleShooting" },
    });

    await waitFor(() => {
      expect(mocks.usePilotTargetGate).toHaveBeenLastCalledWith({
        enabled: true,
        target: expect.objectContaining({
          runwayThresholdId: "RW05L",
          runwayIdent: "RW05L",
          lon: -78.802,
          lat: 35.874,
          altM: 111.86,
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
    const trajectoryDtInput = screen.getByLabelText("dt");
    fireEvent.change(trajectoryDtInput, { target: { value: "0.1" } });
    fireEvent.blur(trajectoryDtInput);
    const maxIterationsInput = screen.getByLabelText("Max iter");
    fireEvent.change(maxIterationsInput, { target: { value: "25" } });
    fireEvent.blur(maxIterationsInput);
    fireEvent.click(within(targetEditor).getByRole("button", { name: "Close" }));

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalledWith({
        optimizer: "singleShooting",
        initialState: expect.objectContaining({
          lon: -78.7873,
          lat: 35.878659,
          massKg: 78000,
          aircraftType: "A320",
        }),
        targetState: expect.objectContaining({
          lon: -78.802,
          lat: 35.874,
          altM: 111.86,
          speedMps: 70,
          headingDeg: 45,
          flightPathDeg: -4,
          aircraftType: "A320",
        }),
        targetControl: { attackDeg: 4 },
        nSegments: 12,
        dtS: 0.1,
        maxIterations: 25,
      });
    });
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
      expect(mocks.stepPilotSimulation).toHaveBeenCalledWith(
        { thrustN: 12000, bankDeg: 0, attackDeg: 4 },
        0.2,
      );
    });
    const liveStatePanel = await screen.findByRole("complementary", {
      name: "Realtime aircraft state",
    });
    expect(within(liveStatePanel).getByText("Control")).toBeTruthy();
    expect(
      within(liveStatePanel).getByText("bank 3.0 deg | alpha 4.00 deg | thrust 12000 N"),
    ).toBeTruthy();
  });

  it("keeps the final trajectory state panel visible after replay finishes", async () => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
    mocks.runTrajectoryOptimization.mockResolvedValueOnce({
      ok: true,
      optimizer: "transcription",
      finalTimeS: 0.2,
      nSegments: 1,
      dtS: 0.2,
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
    expect(within(finalStatePanel).getByText("+788.1 m")).toBeTruthy();
  });
});
