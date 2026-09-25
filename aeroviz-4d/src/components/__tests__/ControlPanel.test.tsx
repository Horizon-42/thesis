import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

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

  it("commits the sample count when the field is left or Enter is pressed, never on each keystroke", () => {
    appState.layers.trajectories = true;
    appState.trajectorySampleCount = 50;
    render(<ControlPanel />);
    const field = screen.getByLabelText("Sample count (0 = all)") as HTMLInputElement;
    expect(field.value).toBe("50");
    // typing 200: the loads it plans must not be planned for 2 and 20 on the way
    for (const typed of ["", "2", "20", "200"]) fireEvent.change(field, { target: { value: typed } });
    expect(setTrajectorySampleCount).not.toHaveBeenCalled();
    fireEvent.blur(field);
    expect(setTrajectorySampleCount.mock.calls).toEqual([[200]]);
    // cleared and Enter: all tracks (0); a negative count is all tracks too
    fireEvent.change(field, { target: { value: "" } });
    fireEvent.keyDown(field, { key: "Enter" });
    expect(setTrajectorySampleCount).toHaveBeenLastCalledWith(0);
    expect(field.value).toBe("0");
    fireEvent.change(field, { target: { value: "-5" } });
    fireEvent.blur(field);
    expect(setTrajectorySampleCount).toHaveBeenLastCalledWith(0);
    expect(setTrajectorySampleCount).toHaveBeenCalledTimes(3);
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

    appState.trajectoryComparisonCategory = "fitted_adsb";

    render(<ControlPanel />);

    const selector = screen.getByLabelText("Optimization result") as HTMLSelectElement;
    expect([...selector.options].map((option) => option.value)).toEqual([
      "fitted_adsb",
      "runway",
    ]);
  });

  it("defaults a report-only selection to the first drawable category", async () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.comparisonCategories = [
      category("observed", false, 0),
      category("fitted_adsb", false, 10),
      category("runway", false, 10),
    ];
    appState.trajectoryComparisonCategory = "observed";

    render(<ControlPanel />);

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

  it("ranks a split's prediction results by the chosen accuracy metric", () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "ts_weak_val";
    appState.comparisonCategories = [
      {
        ...category("ts_weak_val", false, 5),
        label: "Weak model",
        datasetSplit: "val",
        accuracy: { adeM: { mean: 1400, p95: 4100 }, fdeM: { mean: 900, p95: 5900 } },
      },
      {
        ...category("ts_strong_val", false, 5),
        label: "Strong model",
        datasetSplit: "val",
        accuracy: { adeM: { mean: 480, p95: 1500 }, fdeM: { mean: 1700, p95: 5000 } },
      },
    ];

    const { container } = render(<ControlPanel />);
    const optionValues = () =>
      [...container.querySelectorAll("optgroup option")].map((option) =>
        (option as HTMLOptionElement).value,
      );

    // Default: manifest order, no metric decoration.
    expect(optionValues()).toEqual(["ts_weak_val", "ts_strong_val"]);

    fireEvent.change(screen.getByLabelText("Sort results"), {
      target: { value: "adeMean" },
    });
    expect(optionValues()).toEqual(["ts_strong_val", "ts_weak_val"]);
    const first = container.querySelector("optgroup option") as HTMLOptionElement;
    expect(first.textContent).toContain("ADE mean 480 m");

    // FDE mean flips the ranking — the metrics are independent axes.
    fireEvent.change(screen.getByLabelText("Sort results"), {
      target: { value: "fdeMean" },
    });
    expect(optionValues()).toEqual(["ts_weak_val", "ts_strong_val"]);
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

  it("offers Optimization as its own result source and routes selection to its categories", () => {
    appState.layers.trajectories = true;
    appState.comparisonCategories = [
      category("runway", false, 10),
      category("ts_model_val", false, 5),
    ];

    render(<ControlPanel />);

    const source = screen.getByLabelText("Result source") as HTMLSelectElement;
    const optionValues = [...source.options].map((option) => option.value);
    expect(optionValues).toEqual(["baseline", "optimization", "prediction", "experiment"]);
    const optimizationOption = source.options[optionValues.indexOf("optimization")];
    expect(optimizationOption.disabled).toBe(false);

    fireEvent.change(source, { target: { value: "optimization" } });
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("runway");
    expect(setTrajectoryComparison).toHaveBeenCalledWith(true);
  });

  it("disables the Optimization source when no optimizer batch is published", () => {
    appState.layers.trajectories = true;
    appState.comparisonCategories = [category("ts_model_val", false, 5)];

    render(<ControlPanel />);

    const source = screen.getByLabelText("Result source") as HTMLSelectElement;
    const optimizationOption = [...source.options].find(
      (option) => option.value === "optimization",
    );
    expect(optimizationOption?.disabled).toBe(true);
  });

  it("explains that each baseline verdict is sampled independently", () => {
    render(
      <ControlPanel
        observedVerdicts={{
          counts: { pass: 4, fail: 2, undecided: 3 },
          matched: 9,
          total: 9,
        }}
      />,
    );

    const filter = screen.getByLabelText("Baseline verdict") as HTMLSelectElement;
    expect([...filter.options].map((option) => option.textContent)).toEqual([
      "All verdicts",
      "Pass only",
      "Fail only",
      "Indeterminate only",
    ]);
    fireEvent.change(filter, { target: { value: "fail" } });

    expect(setObservedVerdictFilter).toHaveBeenCalledWith("fail");
    expect(setTrajectorySampleCount).not.toHaveBeenCalled();
    expect(setTrajectoryComparison).not.toHaveBeenCalled();
    expect(screen.getByText(/sample limit applies within the selected verdict/i)).toBeTruthy();
    expect(screen.getByText(/if fewer tracks exist, all are shown/i)).toBeTruthy();
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
    const picker = screen.getByLabelText("Experiment model");
    expect(picker.getAttribute("aria-expanded")).toBe("false");
    // An unstamped publish: the flat label names the run, and the card says the intent is missing.
    expect(picker.textContent).toContain("normalized time");
    expect(screen.getByText(/No intent recorded/)).toBeTruthy();
    const split = screen.getByLabelText("Dataset split") as HTMLSelectElement;
    expect([...split.options].map((option) => option.value)).toEqual([
      "experiment_run_train",
      "experiment_run_val",
    ]);
    fireEvent.change(split, { target: { value: "experiment_run_train" } });
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("experiment_run_train");
    expect(screen.getByText("campaign/stage/run/checkpoint.pt")).toBeTruthy();
  });

  it("labels a day partition's held-out days as such, never as the held-out test", () => {
    const metadata = {
      id: "runway_intent_r2b_20260913/KRDU_day_a_expert",
      group: "runway_intent_r3_20260914",
      checkpoint: "runway_intent_r2b_20260913/KRDU_day_a_expert/checkpoint.pt",
      model: "itransformer",
      predictionOutput: "plan",
      horizonMode: "normalized" as const,
      seed: 1337,
    };
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "experiment_krdu_dayval";
    appState.comparisonCategories = [
      {
        ...category("experiment_krdu_dayval", false, 12),
        datasetSplit: "dayval",
        resultSource: "experiment",
        experiment: metadata,
      },
    ];

    render(<ControlPanel />);

    const split = screen.getByLabelText("Dataset split") as HTMLSelectElement;
    expect([...split.options].map((option) => option.textContent)).toEqual([
      "Held-out days (day partition validation) (12)",
    ]);
  });

  function stampedExperiment(run: string, group: string, dModel: string) {
    return {
      id: `${group}/${run}`,
      group,
      checkpoint: `${group}/${run}/checkpoint.pt`,
      label: `control · iTransformer · point-mass · simple-v3 · ${run}`,
      runName: run,
      variantLabel: null,
      intent: { groupTitle: `${group} title`, group: `${group} question`, run: `why ${run}` },
      parameters: [
        { section: "Model", name: "Output", value: "control", field: "prediction_output" },
        { section: "Architecture", name: "d_model", value: dModel },
      ],
      predictionOutput: "control" as const,
    };
  }

  function showStampedExperiments(): void {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "experiment_a_val";
    appState.comparisonCategories = [
      {
        ...category("experiment_a_val", false, 5),
        datasetSplit: "val",
        resultSource: "experiment",
        experiment: stampedExperiment("arm_a", "alpha", "512"),
      },
      {
        ...category("experiment_b_train", false, 5),
        datasetSplit: "train",
        resultSource: "experiment",
        experiment: stampedExperiment("arm_b", "beta", "256"),
      },
      {
        ...category("experiment_b_val", false, 5),
        datasetSplit: "val",
        resultSource: "experiment",
        experiment: stampedExperiment("arm_b", "beta", "256"),
      },
    ];
  }

  it("shows the selected run's intent and named Model rows in the panel card", () => {
    showStampedExperiments();

    render(<ControlPanel />);

    const trigger = screen.getByLabelText("Experiment model");
    expect(trigger.textContent).toContain("alpha title");
    expect(trigger.textContent).toContain("arm_a");
    expect(screen.getByText("why arm_a")).toBeTruthy();
    expect(screen.getByText("alpha question")).toBeTruthy();
    const model = screen.getByRole("region", { name: "Model" });
    expect(within(model).getByText("Output").nextSibling?.textContent).toBe("control");
    // The other sections fold behind a disclosure in the narrow panel.
    expect(screen.getByText("All parameters (1 more)")).toBeTruthy();
    expect(screen.getByText("alpha/arm_a/checkpoint.pt")).toBeTruthy();
  });

  it("opens a browser of campaigns with their question and previews a run's parameters", () => {
    showStampedExperiments();
    render(<ControlPanel />);

    fireEvent.click(screen.getByLabelText("Experiment model"));
    const browser = screen.getByRole("dialog", { name: "Experiment browser" });
    const list = within(browser).getByRole("navigation", { name: "Experiment campaigns" });
    // The active run's campaign is open with its question; the other starts collapsed.
    expect(within(list).getByText("alpha question")).toBeTruthy();
    expect(within(list).queryByText("beta question")).toBeNull();
    fireEvent.click(within(list).getByRole("button", { name: /beta title/ }));
    expect(within(list).getByText("beta question")).toBeTruthy();

    // Hover previews: the detail pane names every parameter.
    const armB = within(list).getByRole("button", { name: /arm_b/ });
    fireEvent.mouseEnter(armB);
    const detail = browser.querySelector(".experiment-browser-detail") as HTMLElement;
    expect(within(detail).getByRole("heading", { name: "arm_b" })).toBeTruthy();
    expect(within(detail).getByText("d_model").nextSibling?.textContent).toBe("256");

    // A click shows the run — its validation publication — and closes the browser.
    fireEvent.click(armB);
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("experiment_b_val");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("keeps the held-out days in view when switching to another run that has them", () => {
    appState.layers.trajectories = true;
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "experiment_a_dayval";
    appState.comparisonCategories = (["a", "b"] as const).flatMap((arm) =>
      (["val", "dayval"] as const).map((split) => ({
        ...category(`experiment_${arm}_${split}`, false, 5),
        datasetSplit: split,
        resultSource: "experiment" as const,
        experiment: stampedExperiment(`arm_${arm}`, "alpha", "512"),
      })),
    );
    render(<ControlPanel />);

    fireEvent.click(screen.getByLabelText("Experiment model"));
    const browser = screen.getByRole("dialog", { name: "Experiment browser" });
    const list = within(browser).getByRole("navigation", { name: "Experiment campaigns" });
    fireEvent.click(within(list).getByRole("button", { name: /arm_b/ }));
    expect(setTrajectoryComparisonCategory).toHaveBeenCalledWith("experiment_b_dayval");
  });

  it("filters the browser across intent and parameters, and closes on Escape", () => {
    showStampedExperiments();
    render(<ControlPanel />);

    fireEvent.click(screen.getByLabelText("Experiment model"));
    const browser = screen.getByRole("dialog", { name: "Experiment browser" });
    fireEvent.change(within(browser).getByLabelText("Filter experiments"), {
      target: { value: "d_model 256" },
    });
    // Filtering opens every campaign that still has a match.
    const list = within(browser).getByRole("navigation", { name: "Experiment campaigns" });
    expect(within(list).queryByRole("button", { name: /arm_a/ })).toBeNull();
    expect(within(list).getByRole("button", { name: /arm_b/ })).toBeTruthy();
    fireEvent.change(within(browser).getByLabelText("Filter experiments"), {
      target: { value: "no such run" },
    });
    expect(within(browser).getByText(/No run matches/)).toBeTruthy();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(setTrajectoryComparisonCategory).not.toHaveBeenCalled();
  });

  it("closes on a press outside it — pointerdown, which the Cesium canvas still delivers", () => {
    showStampedExperiments();
    render(<ControlPanel />);
    const trigger = screen.getByLabelText("Experiment model");
    // The trigger's accessible name carries the selected run, not only the field's label.
    expect(trigger.getAttribute("aria-labelledby")?.split(" ")).toHaveLength(2);

    fireEvent.click(trigger);
    const browser = screen.getByRole("dialog", { name: "Experiment browser" });
    fireEvent.pointerDown(within(browser).getByLabelText("Filter experiments"));
    expect(screen.getByRole("dialog")).toBeTruthy();

    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("walks the runs with the arrow keys and returns to the filter box", () => {
    showStampedExperiments();
    render(<ControlPanel />);
    fireEvent.click(screen.getByLabelText("Experiment model"));
    const browser = screen.getByRole("dialog", { name: "Experiment browser" });
    const filter = within(browser).getByLabelText("Filter experiments");
    expect(document.activeElement).toBe(filter);

    fireEvent.keyDown(filter, { key: "ArrowDown" });
    const first = document.activeElement as HTMLElement;
    expect(first.className).toContain("experiment-browser-run");
    expect(first.textContent).toContain("arm_a");
    fireEvent.keyDown(first, { key: "ArrowUp" });
    expect(document.activeElement).toBe(filter);
  });

  it("shows only optimizer-category paths, explains result overrides, and omits Optimize states", () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "runway";
    appState.comparisonCategories = [category("runway", false)];
    appState.comparisonLegend = {
      kinds: ["reference", "simulator"],
      statuses: ["offTargetResult"],
      status: "ready",
    };

    render(<ControlPanel />);

    expect(screen.getByLabelText("Reference")).toBeTruthy();
    expect(screen.getByLabelText("Optimize results")).toBeTruthy();
    expect(screen.queryByLabelText("Optimize states")).toBeNull();
    expect(screen.queryByLabelText("Predicted")).toBeNull();
    expect(screen.getByText("Off-target optimize result")).toBeTruthy();
  });

  it("switches the legend to the kinds in a prediction category", () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = "ts_transformer";
    appState.comparisonCategories = [category("ts_transformer", false)];
    appState.comparisonLegend = {
      kinds: ["reference", "predicted", "lookback"],
      statuses: ["predictionPass", "predictionFail", "predictionIndeterminate"],
      status: "ready",
    };

    render(<ControlPanel />);

    expect(screen.getByLabelText("Reference")).toBeTruthy();
    expect(screen.getByLabelText("Predicted")).toBeTruthy();
    expect(screen.getByLabelText("Predictor input")).toBeTruthy();
    expect(screen.queryByLabelText("Optimize results")).toBeNull();
    expect(screen.queryByLabelText("Optimize states")).toBeNull();
    expect(screen.queryByText("Off-target optimize result")).toBeNull();
    expect(screen.getByText("Prediction pass")).toBeTruthy();
    expect(screen.getByText("Prediction fail")).toBeTruthy();
    expect(screen.getByText("Prediction indeterminate")).toBeTruthy();
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
