import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const {
  appState,
  resetAppState,
  toggleLayer,
  setPlaybackSpeed,
  setActiveAirportCode,
  setSelectedRunway,
  setTrajectoryComparison,
  setTrajectoryComparisonCategory,
  setTrajectoryComparisonKind,
  setTrajectorySampleCount,
  landingsRef,
} = vi.hoisted(() => {
  const defaultLayers = {
    satelliteImagery: true,
    terrain: false,
    airportLocalTerrain: false,
    terrainHillshade: false,
    terrainHeightTint: false,
    runways: true,
    waypoints: false,
    ocsSurfaces: true,
    trajectories: true,
    obstacles: false,
    obstacleLabels: false,
    procedures: false,
  };
  const defaultAirportLocalTerrain = {
    status: "disabled",
    airportCode: "KRDU",
    sourceLabel: null,
    sourceKind: null,
    sourceName: null,
    horizontalResolutionM: null,
    sourceCrsCode: null,
    sourceCrsName: null,
    minimumHeightM: null,
    maximumHeightM: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  };
  const appState: any = {
    viewer: null,
    layers: { ...defaultLayers },
    airportLocalTerrain: { ...defaultAirportLocalTerrain },
    playbackSpeed: 60,
    airports: [
      { code: "KRDU", name: "Raleigh-Durham International Airport", lat: 35.878659, lon: -78.7873 },
      { code: "CYVR", name: "Vancouver International Airport", lat: 49.193901, lon: -123.183998 },
    ],
    activeAirportCode: "KRDU",
    trajectoryComparison: false,
    trajectoryComparisonCategory: null,
    trajectoryComparisonKinds: { reference: true, optimizer: true, simulator: true },
    trajectorySampleCount: 0,
  };

  return {
    appState,
    resetAppState: () => {
      appState.layers = { ...defaultLayers };
      appState.airportLocalTerrain = { ...defaultAirportLocalTerrain };
      appState.activeAirportCode = "KRDU";
    },
    toggleLayer: vi.fn(),
    setPlaybackSpeed: vi.fn(),
    setActiveAirportCode: vi.fn(),
    setSelectedRunway: vi.fn(),
    setTrajectoryComparison: vi.fn(),
    setTrajectoryComparisonCategory: vi.fn(),
    setTrajectoryComparisonKind: vi.fn(),
    setTrajectorySampleCount: vi.fn(),
    landingsRef: { current: { manifest: null as unknown, status: "empty" } },
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState,
    selectedRunway: appState.selectedRunway ?? null,
    toggleLayer,
    setPlaybackSpeed,
    setActiveAirportCode,
    setSelectedRunway,
    setTrajectoryComparison,
    setTrajectoryComparisonCategory,
    setTrajectoryComparisonKind,
    setTrajectorySampleCount,
  }),
}));

vi.mock("../../hooks/useLandingsManifest", () => ({
  useLandingsManifest: () => landingsRef.current,
}));

vi.mock("../../hooks/useComparisonCategories", () => ({
  useComparisonCategories: () => ({ categories: [], status: "empty" }),
}));

import ControlPanel from "../ControlPanel";

describe("ControlPanel", () => {
  beforeEach(() => {
    resetAppState();
    landingsRef.current = { manifest: null, status: "empty" };
    vi.clearAllMocks();
  });

  // Airport + landing-runway selection moved to the top bar (see WorkbenchTopBar.test);
  // ControlPanel is now the Observe-mode trajectory controls.
  it("toggles the trajectories layer", () => {
    appState.layers.trajectories = false;
    render(<ControlPanel />);

    const checkbox = screen.getByLabelText("Trajectories") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);

    expect(toggleLayer).toHaveBeenCalledWith("trajectories");
  });
});
