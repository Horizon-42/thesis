import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import EvaluationReportWindow from "../EvaluationReportWindow";
import type { EvaluationReport } from "../../data/evaluationReport";

const REPORT: EvaluationReport = {
  thresholds: { lateral_max_m: 106.75, vertical_below_max_m: 3.05, vertical_above_max_m: 6.1 },
  total: 3,
  solved: 2,
  solve_rate: 2 / 3,
  successful: 1,
  success_rate: 1 / 3,
  success_rate_among_solved: 0.5,
  lateral_m: { mean: 101.0, p95: 171.7, max: 179.5 },
  vertical_m: { mean_signed: -13.5, mean_abs: 13.5, p95_abs: 24.2, max_abs: 25.4 },
  final_time_s: { mean: 364.6, min: 329.1, max: 400.2 },
  reference: {
    compared: 2,
    flight_time_delta_s: { mean: -43.9, min: -206.9, max: 119.2 },
    path_lateral_m: { mean: 11142.1, max: 21541.1 },
    path_vertical_m: { mean_abs: 618.8, max_abs: 1859.3 },
  },
  trajectories: [
    {
      id: "FDX1738", file: "a_eval.json", solved: true, success: true, violations: [],
      lateral_m: 22.43, vertical_m: -1.68, final_time_s: 329.1,
      reference: { file: "r.json", flight_time_s: 536.0, flight_time_delta_s: -206.9 },
    },
    {
      id: "FDX1449", file: "b_eval.json", solved: true, success: false,
      violations: ["lateral 179.5 m > 106.75 m"],
      lateral_m: 179.53, vertical_m: -25.4, final_time_s: 400.2,
      reference: { file: "r2.json", flight_time_s: 281.0, flight_time_delta_s: 119.2 },
    },
    {
      id: "UPS1276", file: "c_eval.json", solved: false, success: false,
      violations: ["unsolved"], reason: "ValueError: Maximum_Iterations_Exceeded",
    },
  ],
};

describe("EvaluationReportWindow", () => {
  it("renders the report's cards, aggregates and verdict rows (backend numbers verbatim)", () => {
    render(<EvaluationReportWindow report={REPORT} subtitle="KRDU · runway_cons" onClose={vi.fn()} />);

    expect(screen.getByRole("dialog", { name: "Evaluation report" })).toBeTruthy();
    expect(screen.getByText("KRDU · runway_cons")).toBeTruthy();
    // cards
    expect(screen.getByText("solve rate 66.7%")).toBeTruthy();
    expect(screen.getByText("success rate 33.3%")).toBeTruthy();
    expect(screen.getByText("mean Δt vs observed (optimized − flown)")).toBeTruthy();
    // gates note carries the thresholds
    expect(screen.getByText(/Gates \(FAA Order 8260\.58D\)/).textContent).toContain("106.75 m");
    // aggregates straight from the report
    const aggregates = screen.getByRole("table", { name: /Aggregates/ });
    expect(aggregates.textContent).toContain("101.0");
    expect(aggregates.textContent).toContain("171.7");
    // verdict rows: gate-failed row flagged, unsolved row grayed with its reason
    const failRow = screen.getByText("FDX1449").closest("tr")!;
    expect(failRow.className).toContain("eval-row-fail");
    expect(failRow.textContent).toContain("lateral 179.5 m > 106.75 m");
    const unsolvedRow = screen.getByText("UPS1276").closest("tr")!;
    expect(unsolvedRow.className).toContain("eval-row-unsolved");
    expect(unsolvedRow.textContent).toContain("Maximum_Iterations_Exceeded");
  });

  it("closes via the Close button", () => {
    const onClose = vi.fn();
    render(<EvaluationReportWindow report={REPORT} subtitle="x" onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("renders without reference data and without solved flights", () => {
    const bare: EvaluationReport = {
      ...REPORT,
      solved: 0, solve_rate: 0, successful: 0, success_rate: 0,
      success_rate_among_solved: null,
      lateral_m: null, vertical_m: null, final_time_s: null, reference: null,
      trajectories: [REPORT.trajectories[2]],
    };
    render(<EvaluationReportWindow report={bare} subtitle="x" onClose={vi.fn()} />);
    expect(screen.getByText("No solved trajectories to chart.")).toBeTruthy();
    expect(screen.queryByText(/mean Δt vs observed/)).toBeNull();
  });
});
