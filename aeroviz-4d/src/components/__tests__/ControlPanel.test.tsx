import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const {
  appState,
  resetAppState,
  toggleLayer,
  setPlaybackSpeed,
  setActiveAirportCode,
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
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState,
    toggleLayer,
    setPlaybackSpeed,
    setActiveAirportCode,
  }),
}));

import ControlPanel from "../ControlPanel";

describe("ControlPanel", () => {
  beforeEach(() => {
    resetAppState();
    vi.clearAllMocks();
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
