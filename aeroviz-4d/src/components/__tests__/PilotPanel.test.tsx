import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
      finalTimeS: 100,
      nSegments: 10,
      controls: [{ thrustN: 12000, bankDeg: 0, attackDeg: 4 }],
      states: [],
    });
  });

  it("opens trajectory play from pilot mode and submits runway-target optimization", async () => {
    render(<PilotPanel />);

    expect(await screen.findByText("A320")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Trajectory" }));

    expect(screen.getByText("Trajectory Play")).toBeTruthy();
    expect(await screen.findByText("Target Threshold")).toBeTruthy();
    expect(screen.getByRole("combobox").textContent).toContain("RW05L");

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));

    await waitFor(() => {
      expect(mocks.runTrajectoryOptimization).toHaveBeenCalledWith({
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
          flightPathDeg: -3,
          aircraftType: "A320",
        }),
        targetControl: { attackDeg: 4 },
        nSegments: 10,
      });
    });
  });
});
