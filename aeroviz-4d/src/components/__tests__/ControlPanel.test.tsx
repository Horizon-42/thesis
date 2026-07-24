import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const {
  appState,
  resetAppState,
  toggleLayer,
  setPlaybackSpeed,
  setActiveAirportCode,
  setSelectedRunway,
  setProceduresOpen,
  setApproachViewOpen,
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
    selectedRunway: null as string | null,
    proceduresOpen: false,
    isApproachViewOpen: false,
    trajectoryComparison: false,
    trajectoryComparisonCategory: null as string | null,
    trajectoryComparisonKinds: { reference: true, optimizer: true, simulator: true },
    trajectorySampleCount: 0,
    comparisonCategories: [] as Array<Record<string, unknown>>,
  };

  return {
    appState,
    resetAppState: () => {
      appState.layers = { ...defaultLayers };
      appState.airportLocalTerrain = { ...defaultAirportLocalTerrain };
      appState.activeAirportCode = "KRDU";
      appState.selectedRunway = null;
      appState.proceduresOpen = false;
      appState.isApproachViewOpen = false;
      appState.trajectoryComparison = false;
      appState.trajectoryComparisonCategory = null;
      appState.comparisonCategories = [];
    },
    toggleLayer: vi.fn(),
    setPlaybackSpeed: vi.fn(),
    setActiveAirportCode: vi.fn(),
    setSelectedRunway: vi.fn(),
    setProceduresOpen: vi.fn(),
    setApproachViewOpen: vi.fn(),
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
    setProceduresOpen,
    setApproachViewOpen,
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
  useComparisonCategories: () => ({
    categories: appState.comparisonCategories,
    status: appState.comparisonCategories.length ? "ready" : "empty",
  }),
}));

/** A manifest category — `constrained` is the explicit field the panel keys off. */
const category = (dir: string, constrained: boolean, groups = 1) => ({
  key: dir,
  label: dir,
  dir,
  groups,
  constrained,
});

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

  // ── Constraint-scoped procedure display (Feature A) ──────────────────────────
  it("auto-opens the procedure display for a constrained category + runway, without touching the runway", async () => {
    appState.layers.trajectories = true;
    appState.layers.procedures = false;
    appState.trajectoryComparison = true;
    appState.comparisonCategories = [category("runway", false), category("runway_cons", true)];
    appState.trajectoryComparisonCategory = "runway_cons";
    appState.selectedRunway = "05L";

    render(<ControlPanel />);

    await waitFor(() => expect(setProceduresOpen).toHaveBeenCalledWith(true));
    expect(toggleLayer).toHaveBeenCalledWith("procedures");
    // The runway IS the user-owned global selection — the driver must never write it.
    expect(setSelectedRunway).not.toHaveBeenCalled();
  });

  it("does not force procedures for an UNconstrained category", async () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.comparisonCategories = [category("runway", false), category("runway_cons", true)];
    appState.trajectoryComparisonCategory = "runway";
    appState.selectedRunway = "05L";

    render(<ControlPanel />);
    // Let effects settle, then assert the panel was never forced open.
    await Promise.resolve();
    expect(setProceduresOpen).not.toHaveBeenCalledWith(true);
  });

  it("does not force procedures when the comparison overlay is off", async () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = false;
    appState.comparisonCategories = [category("runway", false), category("runway_cons", true)];
    appState.trajectoryComparisonCategory = "runway_cons";
    appState.selectedRunway = "05L";

    render(<ControlPanel />);
    await Promise.resolve();
    expect(setProceduresOpen).not.toHaveBeenCalledWith(true);
  });

  it("keeps report-only categories out of comparison and defaults to a drawable category", async () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.comparisonCategories = [
      category("observed", false, 0),
      category("fitted_adsb", false, 10),
      category("runway", false, 10),
    ];
    appState.trajectoryComparisonCategory = "observed";

    render(<ControlPanel />);

    const selector = screen.getByLabelText("Evaluation category") as HTMLSelectElement;
    expect([...selector.options].map((option) => option.value)).toEqual([
      "fitted_adsb",
      "runway",
    ]);
    await waitFor(() =>
      expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("fitted_adsb"),
    );
  });

  // ── Approach-approach-view toggle (Feature B) ──────────────────────────────────────
  it("disables the approach-view toggle and shows a hint when no landing runway is selected", () => {
    appState.selectedRunway = null;
    const { container } = render(<ControlPanel />);
    const button = screen.getByRole("button", { name: "View" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    // A VISIBLE hint (not just the hover tooltip) explains why clicking does nothing.
    expect(container.querySelector(".control-panel-approach-view-hint")?.textContent).toContain(
      "No landing runway selected",
    );
  });

  it("opens the approach view for the selected runway, with no hint", () => {
    appState.selectedRunway = "05L";
    const { container } = render(<ControlPanel />);
    const button = screen.getByRole("button", { name: "View" }) as HTMLButtonElement;
    expect(button.disabled).toBe(false);
    expect(container.querySelector(".control-panel-approach-view-hint")).toBeNull();

    fireEvent.click(button);

    expect(setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(setApproachViewOpen).toHaveBeenCalledWith(true);
  });
});
