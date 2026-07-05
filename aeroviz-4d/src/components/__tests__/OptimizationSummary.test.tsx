import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { OptimizationStats } from "../../data/airportData";
import type { EvaluationReport } from "../../data/evaluationReport";

const { data, appState, fetchJsonMock } = vi.hoisted(() => ({
  data: {
    stats: null as OptimizationStats | null,
    categoryDir: null as string | null,
  },
  appState: { activeAirportCode: "KRDU" as string | null },
  fetchJsonMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => appState }));
vi.mock("../../hooks/useFlightOptimizerData", () => ({
  useFlightOptimizerData: () => data,
}));
vi.mock("../../utils/fetchJson", () => ({
  fetchJson: fetchJsonMock,
  isMissingJsonAsset: (error: unknown) => String(error).includes("missing"),
}));

import OptimizationSummary from "../OptimizationSummary";

const REPORT: EvaluationReport = {
  thresholds: { lateral_max_m: 106.75, vertical_below_max_m: 3.05, vertical_above_max_m: 6.1 },
  total: 2, solved: 1, solve_rate: 0.5, successful: 1, success_rate: 0.5,
  success_rate_among_solved: 1,
  lateral_m: { mean: 2.8, p95: 2.8, max: 2.8 },
  vertical_m: { mean_signed: 0.1, mean_abs: 0.1, p95_abs: 0.1, max_abs: 0.1 },
  final_time_s: { mean: 393.5, min: 393.5, max: 393.5 },
  reference: null,
  trajectories: [
    { id: "DAL1272", file: "a_eval.json", solved: true, success: true, violations: [],
      lateral_m: 2.8, vertical_m: 0.1, final_time_s: 393.5 },
    { id: "UPS1276", file: "b_eval.json", solved: false, success: false,
      violations: ["unsolved"], reason: "ValueError: infeasible" },
  ],
};

function row(label: string) {
  return screen.getByText(label).closest(".optimization-summary-row") as HTMLElement;
}

describe("OptimizationSummary", () => {
  beforeEach(() => {
    data.stats = null;
    data.categoryDir = null;
    appState.activeAirportCode = "KRDU";
    fetchJsonMock.mockReset();
  });

  it("shows the real solve rate and placeholders for the not-yet-computed metrics", () => {
    data.stats = { total: 1001, solved: 628, failed: 373, solveRate: 0.6267 };
    render(<OptimizationSummary />);

    // Solve rate is real (from the index) and not marked pending.
    const solve = row("Solve rate");
    expect(within(solve).getByText("62.7%")).toBeTruthy();
    expect(within(solve).queryByText("—")).toBeNull();

    // The evaluation-package metrics are placeholders (dash, marked pending).
    for (const label of ["Success rate", "Avg state error", "Avg flight time"]) {
      const dd = row(label).querySelector("dd")!;
      expect(dd.textContent).toBe("—");
      expect(dd.className).toContain("optimization-summary-pending");
    }
  });

  it("dashes the solve rate too when there are no stats", () => {
    render(<OptimizationSummary />);
    expect(row("Solve rate").querySelector("dd")!.textContent).toBe("—");
  });

  it("renders evaluation metrics once the index carries them", () => {
    data.stats = { solveRate: 0.9, successRate: 0.82, avgStateErrorM: 143.6, avgTimeS: 305 };
    render(<OptimizationSummary />);
    expect(within(row("Success rate")).getByText("82.0%")).toBeTruthy();
    expect(within(row("Avg state error")).getByText("144 m")).toBeTruthy();
    expect(within(row("Avg flight time")).getByText("5:05")).toBeTruthy();
  });

  it("disables Details without a category, opens the report window with one", async () => {
    render(<OptimizationSummary />);
    expect(
      (screen.getByRole("button", { name: "Details" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("Details fetches the published report and opens the evaluation window", async () => {
    data.stats = { solveRate: 0.5 };
    data.categoryDir = "runway_cons";
    fetchJsonMock.mockResolvedValue(REPORT);
    render(<OptimizationSummary />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(fetchJsonMock).toHaveBeenCalledWith(
      expect.stringContaining("KRDU/comparison/runway_cons/evaluation_report.json"),
    );
    // the window renders the backend report verbatim
    expect(await screen.findByRole("dialog", { name: "Evaluation report" })).toBeTruthy();
    expect(screen.getByText("solve rate 50.0%")).toBeTruthy();
    expect(screen.getByText("DAL1272")).toBeTruthy();
  });

  it("surfaces a helpful message when no report was published", async () => {
    data.categoryDir = "runway";
    fetchJsonMock.mockRejectedValue(new Error("missing json asset"));
    render(<OptimizationSummary />);
    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(await screen.findByText(/No evaluation report published/)).toBeTruthy();
    expect(screen.queryByRole("dialog", { name: "Evaluation report" })).toBeNull();
  });
});
