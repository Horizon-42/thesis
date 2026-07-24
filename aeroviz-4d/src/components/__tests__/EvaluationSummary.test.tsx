import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ComparisonCategory, ComparisonIndex } from "../../data/airportData";
import type { EvaluationReport } from "../../data/evaluationReport";

const { appState, categoryState, fetchJsonMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KRDU" as string | null,
    trajectoryComparison: false,
    trajectoryComparisonCategory: "observed" as string | null,
  },
  categoryState: {
    categories: [] as ComparisonCategory[],
  },
  fetchJsonMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => appState }));
vi.mock("../../hooks/useComparisonCategories", () => ({
  useComparisonCategories: () => ({ categories: categoryState.categories, status: "ready" }),
}));
vi.mock("../../utils/fetchJson", () => ({
  fetchJson: fetchJsonMock,
  isMissingJsonAsset: (error: unknown) => String(error).includes("missing"),
}));

import EvaluationSummary from "../EvaluationSummary";

const OBSERVED: ComparisonCategory = {
  key: "observed",
  label: "Observed ADS-B",
  dir: "observed",
  groups: 0,
  constrained: false,
};
const FITTED: ComparisonCategory = {
  key: "fitted_adsb",
  label: "Fitted ADS-B crossing",
  dir: "fitted_adsb",
  groups: 10,
  constrained: false,
};
const RUNWAY_CONSTRAINED: ComparisonCategory = {
  key: "runway_cons",
  label: "Runway target (constrained)",
  dir: "runway_cons",
  groups: 10,
  constrained: true,
};
const PREDICTED: ComparisonCategory = {
  key: "ts_itr_full_test",
  label: "Predicted (iTransformer, full, test split)",
  dir: "ts_itr_full_test",
  groups: 10,
  constrained: false,
};

const REPORT: EvaluationReport = {
  thresholds: {
    lateral_max_m: 106.75,
    vertical_below_max_m: 3.05,
    vertical_above_max_m: 6.1,
  },
  total: 10,
  solved: 8,
  solve_rate: 0.8,
  successful: 6,
  success_rate: 0.6,
  success_rate_among_solved: 0.75,
  lateral_m: { mean: 12.4, p95: 30, max: 40 },
  vertical_m: { mean_signed: 1.2, mean_abs: 4.6, p95_abs: 7, max_abs: 9 },
  final_time_s: { mean: 305, min: 250, max: 360 },
  reference: null,
  trajectories: [],
};

const OBSERVED_REPORT: EvaluationReport = {
  ...REPORT,
  subject: "observed",
  measured: 8,
  solved: 10,
  solve_rate: 1,
  observed: {
    established: 8,
    not_established: 2,
    established_rate: 0.8,
    marginal: 3,
  },
};

const INDEX: ComparisonIndex = {
  schemaVersion: "comparison-v2-generation",
  generation: "batch123",
  epoch: "2026-04-01T08:00:00+00:00",
  startHidden: true,
  referenceSource: "canonicalObserved",
  groups: [],
  optimization: {
    total: 10,
    solved: 8,
    failed: 2,
    solveRate: 0.8,
    successful: 6,
    successRate: 0.6,
    avgStateErrorM: 12.4,
    avgTimeS: 305,
  },
  evaluationReport: "evaluation_report_batch123.json",
};

function metric(label: string): HTMLElement {
  return screen.getByText(label).closest(".evaluation-summary-row") as HTMLElement;
}

describe("EvaluationSummary", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KRDU";
    appState.trajectoryComparison = false;
    appState.trajectoryComparisonCategory = OBSERVED.dir;
    categoryState.categories = [OBSERVED, FITTED, RUNWAY_CONSTRAINED, PREDICTED];
    fetchJsonMock.mockReset();
  });

  it("renders the report-only observed category with baseline-specific metrics", async () => {
    fetchJsonMock.mockResolvedValue(OBSERVED_REPORT);
    render(<EvaluationSummary />);

    expect(
      screen.getByRole("region", { name: "Observed Baseline Evaluation" }),
    ).toBeTruthy();
    expect(await within(metric("Established approach rate")).findByText("80.0%")).toBeTruthy();
    expect(within(metric("Threshold-crossing pass rate")).getByText("75.0%")).toBeTruthy();
    expect(
      within(metric("Mean lateral deviation at threshold")).getByText("12 m"),
    ).toBeTruthy();
    expect(
      within(metric("Mean absolute vertical deviation at threshold")).getByText("5 m"),
    ).toBeTruthy();
    expect(screen.getByText(/No optimizer is involved/)).toBeTruthy();
    expect(fetchJsonMock).toHaveBeenCalledWith(
      expect.stringContaining("KRDU/comparison/observed/evaluation_report.json"),
    );
  });

  it("returns to the observed baseline when comparison is off", async () => {
    appState.trajectoryComparison = false;
    appState.trajectoryComparisonCategory = FITTED.dir;
    fetchJsonMock.mockResolvedValue(OBSERVED_REPORT);

    render(<EvaluationSummary />);

    expect(
      await screen.findByRole("region", { name: "Observed Baseline Evaluation" }),
    ).toBeTruthy();
    expect(fetchJsonMock).toHaveBeenCalledWith(
      expect.stringContaining("KRDU/comparison/observed/evaluation_report.json"),
    );
    expect(fetchJsonMock).not.toHaveBeenCalledWith(
      expect.stringContaining("fitted_adsb/comparison_index.json"),
    );
  });

  it("labels fitted ADS-B as an optimization target, not as an observed baseline", async () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = FITTED.dir;
    fetchJsonMock.mockResolvedValue(INDEX);
    render(<EvaluationSummary />);

    expect(await screen.findByText("Target: Fitted ADS-B crossing")).toBeTruthy();
    expect(screen.getByRole("region", { name: "Optimization Evaluation" })).toBeTruthy();
    expect(within(metric("Solve rate")).getByText("80.0%")).toBeTruthy();
    expect(within(metric("Fitted-target pass rate")).getByText("75.0%")).toBeTruthy();
    expect(
      within(metric("Mean lateral error to fitted target")).getByText("12 m"),
    ).toBeTruthy();
    expect(within(metric("Mean optimized flight time")).getByText("5:05")).toBeTruthy();
    expect(screen.getByText(/not the nominal runway threshold/)).toBeTruthy();
  });

  it("names the runway target and procedure constraint explicitly", async () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = RUNWAY_CONSTRAINED.dir;
    fetchJsonMock.mockResolvedValue(INDEX);
    render(<EvaluationSummary />);

    expect(
      await screen.findByText("Target: Runway threshold · Procedure constrained"),
    ).toBeTruthy();
    expect(within(metric("Runway-threshold pass rate")).getByText("75.0%")).toBeTruthy();
    expect(
      within(metric("Mean lateral error to runway threshold")).getByText("12 m"),
    ).toBeTruthy();
    expect(screen.getByText(/enforces the selected procedure as path constraints/)).toBeTruthy();
  });

  it("renders ADE and FDE instead of solve metrics for a data-driven model", async () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = PREDICTED.dir;
    fetchJsonMock.mockResolvedValue({
      ...INDEX,
      prediction: {
        flights: 10,
        flightsWithoutOverlap: 0,
        adeM: { mean: 1755.6, p95: 4656.2 },
        fdeM: { mean: 2082.4, p95: 6002.1 },
      },
    } satisfies ComparisonIndex);
    render(<EvaluationSummary />);

    expect(
      await screen.findByRole("region", { name: "Data-Driven Model Evaluation" }),
    ).toBeTruthy();
    expect(within(metric("Mean ADE")).getByText("1756 m")).toBeTruthy();
    expect(within(metric("95th-percentile ADE")).getByText("4656 m")).toBeTruthy();
    expect(within(metric("Mean FDE")).getByText("2082 m")).toBeTruthy();
    expect(within(metric("95th-percentile FDE")).getByText("6002 m")).toBeTruthy();
    expect(screen.queryByText("Solve rate")).toBeNull();
  });

  it("opens observed Details from its fixed report with a subject-aware title", async () => {
    fetchJsonMock.mockResolvedValue(OBSERVED_REPORT);
    render(<EvaluationSummary />);

    fireEvent.click(await screen.findByRole("button", { name: "Details" }));
    expect(
      await screen.findByRole("dialog", {
        name: "Observed Baseline Evaluation Report",
      }),
    ).toBeTruthy();
    expect(fetchJsonMock).toHaveBeenCalledTimes(1);
  });

  it("follows the immutable report named by a modelled category index", async () => {
    appState.trajectoryComparison = true;
    appState.trajectoryComparisonCategory = FITTED.dir;
    fetchJsonMock.mockImplementation((url: string) =>
      url.endsWith("comparison_index.json") ? Promise.resolve(INDEX) : Promise.resolve(REPORT),
    );
    render(<EvaluationSummary />);

    fireEvent.click(await screen.findByRole("button", { name: "Details" }));
    expect(fetchJsonMock).toHaveBeenCalledWith(
      expect.stringContaining("fitted_adsb/evaluation_report_batch123.json"),
    );
    expect(
      await screen.findByRole("dialog", { name: "Optimization Evaluation Report" }),
    ).toBeTruthy();
  });
});
