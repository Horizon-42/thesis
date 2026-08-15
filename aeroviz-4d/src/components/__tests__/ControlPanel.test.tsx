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
  setObservedVerdictFilter,
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
    trajectoryComparisonKinds: {
      reference: true,
      optimizer: false,
      simulator: true,
      predicted: true,
      lookback: true,
    },
    trajectorySampleCount: 0,
    observedVerdictFilter: "all",
    comparisonCategories: [] as Array<Record<string, unknown>>,
    comparisonLegend: {
      kinds: [] as string[],
      statuses: [] as string[],
      status: "idle",
    },
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
      appState.comparisonLegend = { kinds: [], statuses: [], status: "idle" };
      appState.observedVerdictFilter = "all";
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
    setObservedVerdictFilter: vi.fn(),
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
    setObservedVerdictFilter,
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

vi.mock("../../hooks/useComparisonLegend", () => ({
  useComparisonLegend: () => appState.comparisonLegend,
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

    const selector = screen.getByLabelText("Prediction result") as HTMLSelectElement;
    expect([...selector.options].map((option) => option.value)).toEqual([
      "fitted_adsb",
      "runway",
    ]);
    await waitFor(() =>
      expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("fitted_adsb"),
    );
  });

  it("separates held-out test and in-sample training categories", () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "ts_model_test";
    appState.comparisonCategories = [
      {
        ...category("ts_model_train", false, 20),
        label: "Training split (in-sample) — Predicted model",
        datasetSplit: "train",
      },
      {
        ...category("ts_model_test", false, 5),
        label: "Test split (held-out) — Predicted model",
        datasetSplit: "test",
      },
    ];

    const { container } = render(<ControlPanel />);

    const groups = [...container.querySelectorAll("optgroup")];
    expect(groups.map((group) => group.label)).toEqual([
      "Held-out test results",
      "Training results (in-sample)",
    ]);
    expect([...groups[0].querySelectorAll("option")].map((option) => option.value)).toEqual([
      "ts_model_test",
    ]);
    expect([...groups[1].querySelectorAll("option")].map((option) => option.value)).toEqual([
      "ts_model_train",
    ]);
  });

  it("uses an exclusive result-source selector and keeps checkboxes for visibility only", () => {
    appState.layers.trajectories = true;
    appState.comparisonCategories = [
      category("ts_model_val", false, 5),
      {
        ...category("experiment_run_val", false, 4),
        datasetSplit: "val",
        resultSource: "experiment",
        experiment: {
          id: "campaign/stage/run",
          group: "campaign",
          checkpoint: "campaign/stage/run/checkpoint.pt",
          predictionOutput: "control",
          horizonMode: "normalized",
          seed: 1337,
        },
      },
    ];

    render(<ControlPanel />);

    const source = screen.getByLabelText("Result source") as HTMLSelectElement;
    expect(source.value).toBe("baseline");
    expect(screen.queryByLabelText("Prediction comparison")).toBeNull();
    fireEvent.change(source, { target: { value: "experiment" } });
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("experiment_run_val");
    expect(setTrajectoryComparison).toHaveBeenCalledWith(true);
  });

  it("filters verdicts within the baseline sample without changing sampling", () => {
    render(
      <ControlPanel
        observedVerdicts={{
          counts: { pass: 4, fail: 2, undecided: 3 },
          verdictsByFlightId: new Map(),
          matched: 9,
          total: 9,
          loading: false,
          missing: false,
        }}
      />,
    );

    const filter = screen.getByLabelText("Show baseline sample") as HTMLSelectElement;
    expect([...filter.options].map((option) => option.textContent)).toEqual([
      "All sampled trajectories",
      "Pass only",
      "Fail only",
      "Indeterminate only",
    ]);
    fireEvent.change(filter, { target: { value: "fail" } });

    expect(setObservedVerdictFilter).toHaveBeenCalledWith("fail");
    expect(setTrajectorySampleCount).not.toHaveBeenCalled();
    expect(setTrajectoryComparison).not.toHaveBeenCalled();
    expect(screen.getByText(/sampled trajectory set does not change/i)).toBeTruthy();
  });

  it("selects an experiment model and its train/validation publication independently", () => {
    const metadata = {
      id: "campaign/stage/run",
      group: "campaign",
      checkpoint: "campaign/stage/run/checkpoint.pt",
      model: "itransformer",
      predictionOutput: "control",
      horizonMode: "normalized" as const,
      seed: 1337,
    };
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "experiment_run_val";
    appState.comparisonCategories = [
      {
        ...category("experiment_run_train", false, 20),
        datasetSplit: "train",
        resultSource: "experiment",
        experiment: metadata,
      },
      {
        ...category("experiment_run_val", false, 5),
        datasetSplit: "val",
        resultSource: "experiment",
        experiment: metadata,
      },
    ];

    render(<ControlPanel />);

    expect((screen.getByLabelText("Result source") as HTMLSelectElement).value).toBe("experiment");
    expect((screen.getByLabelText("Experiment model") as HTMLSelectElement).value)
      .toBe("campaign/stage/run");
    expect(screen.getByLabelText("Experiment model").textContent).toContain("normalized time");
    const split = screen.getByLabelText("Dataset split") as HTMLSelectElement;
    expect([...split.options].map((option) => option.value)).toEqual([
      "experiment_run_train",
      "experiment_run_val",
    ]);
    fireEvent.change(split, { target: { value: "experiment_run_train" } });
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("experiment_run_train");
    expect(screen.getByText("campaign/stage/run/checkpoint.pt")).toBeTruthy();
  });

  it("shows only optimizer-category paths, explains verdict overrides, and omits Optimize states", () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "runway";
    appState.comparisonCategories = [category("runway", false)];
    appState.comparisonLegend = {
      kinds: ["reference", "simulator"],
      statuses: ["failedReference", "offTargetReference", "offTargetResult"],
      status: "ready",
    };

    render(<ControlPanel />);

    expect(screen.getByLabelText("Reference")).toBeTruthy();
    expect(screen.getByLabelText("Optimize results")).toBeTruthy();
    expect(screen.queryByLabelText("Optimize states")).toBeNull();
    expect(screen.queryByLabelText("Predicted")).toBeNull();
    expect(screen.getByText("Unsolved reference")).toBeTruthy();
    expect(screen.getByText("Off-target reference")).toBeTruthy();
    expect(screen.getByText("Off-target optimize result")).toBeTruthy();
  });

  it("switches the legend to the kinds in a prediction category", () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "ts_transformer";
    appState.comparisonCategories = [category("ts_transformer", false)];
    appState.comparisonLegend = {
      kinds: ["reference", "predicted", "lookback"],
      statuses: ["offTargetReference"],
      status: "ready",
    };

    render(<ControlPanel />);

    expect(screen.getByLabelText("Reference")).toBeTruthy();
    expect(screen.getByLabelText("Predicted")).toBeTruthy();
    expect(screen.getByLabelText("Predictor input")).toBeTruthy();
    expect(screen.queryByLabelText("Optimize results")).toBeNull();
    expect(screen.queryByLabelText("Optimize states")).toBeNull();
    expect(screen.queryByText("Off-target optimize result")).toBeNull();
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
