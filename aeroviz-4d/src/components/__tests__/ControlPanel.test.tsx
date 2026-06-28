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

  it("shows a landing-runway selector from the manifest and selects a runway", () => {
    landingsRef.current = {
      manifest: {
        airport: "KRDU",
        combined: "trajectories.czml",
        runways: [
          { runway: "23R", file: "landings/KRDU_23R.czml", count: 40 },
          { runway: "05L", file: "landings/KRDU_05L.czml", count: 12 },
        ],
      },
      status: "ready",
    };

    render(<ControlPanel />);

    const select = screen.getByLabelText("Landing Runway") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      "All runways",
      "23R (40)",
      "05L (12)",
    ]);

    fireEvent.change(select, { target: { value: "23R" } });
    expect(setSelectedRunway).toHaveBeenCalledWith("23R");

    fireEvent.change(select, { target: { value: "" } });
    expect(setSelectedRunway).toHaveBeenCalledWith(null);
  });

  it("hides the runway selector when there are no landings", () => {
    render(<ControlPanel />);
    expect(screen.queryByLabelText("Landing Runway")).toBeNull();
  });

  it("switches the active airport from the selector", () => {
    render(<ControlPanel />);

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "CYVR" },
    });

    expect(setActiveAirportCode).toHaveBeenCalledWith("CYVR");
  });

  it("toggles obstacle labels independently", () => {
    render(<ControlPanel />);

    const checkbox = screen.getByLabelText("Obstacle Labels") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);

    expect(toggleLayer).toHaveBeenCalledWith("obstacleLabels");
  });

  it("places satellite imagery before terrain in the layer toggles", () => {
    render(<ControlPanel />);

    const layerLabels = screen.getAllByRole("checkbox").map((checkbox) => {
      return checkbox.closest("label")?.textContent;
    });

    expect(layerLabels.indexOf("Satellite Imagery")).toBeLessThan(
      layerLabels.indexOf("Terrain"),
    );
  });

  it("shows local terrain metadata when airport local terrain is enabled", () => {
    appState.layers.airportLocalTerrain = true;
    appState.airportLocalTerrain = {
      status: "active",
      airportCode: "KRDU",
      sourceLabel: "Airport local heightmap terrain",
      sourceKind: "dsm",
      sourceName: "USGS TNM DSM",
      horizontalResolutionM: 2,
      sourceCrsCode: "EPSG:26917",
      sourceCrsName: "EPSG:26917 / UTM zone 17 projected metres",
      minimumHeightM: 89,
      maximumHeightM: 243,
      loadedTiles: 12,
      totalTiles: 12,
      error: null,
    };

    render(<ControlPanel />);

    expect(screen.getByText("Source spacing")).toBeTruthy();
    expect(screen.getByText("2 m spacing")).toBeTruthy();
    expect(screen.getByText("DSM (USGS TNM DSM)")).toBeTruthy();
    expect(screen.getByText("EPSG:26917")).toBeTruthy();
  });
});
