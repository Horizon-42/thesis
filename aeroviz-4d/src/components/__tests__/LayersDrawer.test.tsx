import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, toggleLayer, setLayersDrawerOpen } = vi.hoisted(() => {
  const defaultLayers = {
    satelliteImagery: true,
    terrain: false,
    airportLocalTerrain: false,
    terrainHillshade: false,
    terrainHeightTint: false,
    runways: true,
    waypoints: false,
    ocsSurfaces: false,
    trajectories: false,
    obstacles: false,
    obstacleLabels: false,
    procedures: false,
    rangeRing: false,
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
    layers: { ...defaultLayers },
    airportLocalTerrain: { ...defaultAirportLocalTerrain },
    rangeRingRadiusKm: 5,
    layersDrawerOpen: true,
  };
  return {
    appState,
    toggleLayer: vi.fn(),
    setLayersDrawerOpen: vi.fn(),
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState,
    toggleLayer,
    setRangeRingRadiusKm: vi.fn(),
    setLayersDrawerOpen,
  }),
}));

import LayersDrawer from "../LayersDrawer";

describe("LayersDrawer", () => {
  beforeEach(() => {
    appState.layers = {
      satelliteImagery: true,
      terrain: false,
      airportLocalTerrain: false,
      terrainHillshade: false,
      terrainHeightTint: false,
      runways: true,
      waypoints: false,
      ocsSurfaces: false,
      trajectories: false,
      obstacles: false,
      obstacleLabels: false,
      procedures: false,
      rangeRing: false,
    };
    appState.airportLocalTerrain = { ...appState.airportLocalTerrain, status: "disabled" };
    appState.layersDrawerOpen = true;
    vi.clearAllMocks();
  });

  it("renders nothing when the drawer is closed", () => {
    appState.layersDrawerOpen = false;
    const { container } = render(<LayersDrawer />);
    expect(container.firstChild).toBeNull();
  });

  it("does not include the RNAV procedures toggle (it lives in the Procedure panel)", () => {
    render(<LayersDrawer />);
    expect(screen.queryByLabelText("RNAV Procedures")).toBeNull();
  });

  it("toggles obstacle labels independently", () => {
    render(<LayersDrawer />);
    const checkbox = screen.getByLabelText("Obstacle Labels") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    fireEvent.click(checkbox);
    expect(toggleLayer).toHaveBeenCalledWith("obstacleLabels");
  });

  it("places satellite imagery before terrain in the layer toggles", () => {
    render(<LayersDrawer />);
    const layerLabels = screen
      .getAllByRole("checkbox")
      .map((checkbox) => checkbox.closest("label")?.textContent);
    expect(layerLabels.indexOf("Satellite Imagery")).toBeLessThan(layerLabels.indexOf("Terrain"));
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

    render(<LayersDrawer />);

    expect(screen.getByText("Source spacing")).toBeTruthy();
    expect(screen.getByText("2 m spacing")).toBeTruthy();
    expect(screen.getByText("DSM (USGS TNM DSM)")).toBeTruthy();
    expect(screen.getByText("EPSG:26917")).toBeTruthy();
  });

  it("closes from the header button", () => {
    render(<LayersDrawer />);
    fireEvent.click(screen.getByLabelText("Close layers"));
    expect(setLayersDrawerOpen).toHaveBeenCalledWith(false);
  });
});
