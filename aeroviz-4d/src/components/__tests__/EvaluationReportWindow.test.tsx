import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
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
    render(
      <EvaluationReportWindow
        report={REPORT}
        title="Optimization Evaluation Report"
        subtitle="KRDU · runway_cons"
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole("dialog", { name: "Optimization Evaluation Report" })).toBeTruthy();
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
    // The combined 3D view exposes both deviations for every measured flight
    // and sits immediately before (left of, in the two-column grid) flight time.
    const deviation3D = screen.getByRole("img", { name: "3D trajectory deviation view" });
    expect(deviation3D.tagName).toBe("CANVAS");
    expect(deviation3D.getAttribute("data-renderer")).toBe("webgl");
    expect(screen.getByText(/drag to orbit · wheel to zoom/i)).toBeTruthy();
    const deviationFigure = deviation3D.closest("figure")!;
    const timeFigure = screen.getByRole("img", { name: "Flight time scatter" }).closest("figure")!;
    expect(timeFigure.previousElementSibling).toBe(deviationFigure);
    // The complete plot region is an interaction boundary: events over its
    // overlays or empty space must not reach and scroll/activate the report.
    const stage = deviation3D.parentElement!;
    const dialog = deviation3D.closest('[role="dialog"]')!;
    const leakedWheel = vi.fn();
    dialog.addEventListener("wheel", leakedWheel);
    fireEvent.wheel(stage, { deltaY: 100 });
    expect(leakedWheel).not.toHaveBeenCalled();
    // verdict rows: gate-failed row flagged, unsolved row grayed with its reason
    const failRow = screen.getByText("FDX1449").closest("tr")!;
    expect(failRow.className).toContain("eval-row-fail");
    expect(failRow.textContent).toContain("lateral 179.5 m > 106.75 m");
    const unsolvedRow = screen.getByText("UPS1276").closest("tr")!;
    expect(unsolvedRow.className).toContain("eval-row-unsolved");
    expect(unsolvedRow.textContent).toContain("Maximum_Iterations_Exceeded");
  });

  it("gives each statistic its own aggregates column (p95 and min never share one)", () => {
    render(
      <EvaluationReportWindow
        report={REPORT}
        title="Optimization Evaluation Report"
        subtitle="x"
        onClose={vi.fn()}
      />,
    );
    const aggregates = screen.getByRole("table", { name: /Aggregates/ });

    const headers = Array.from(aggregates.querySelectorAll("thead th")).map((h) => h.textContent);
    expect(headers).toEqual(["metric", "mean", "p95", "min", "max"]);

    // Read a row as {header: cell} so a column swap fails loudly rather than
    // passing on a substring match somewhere else in the table.
    const cellsOf = (rowLabel: string | RegExp) => {
      const row = within(aggregates).getByText(rowLabel).closest("tr")!;
      const values = Array.from(row.querySelectorAll("td")).map((td) => td.textContent);
      return Object.fromEntries(headers.slice(1).map((h, i) => [h!, values[i]]));
    };

    // Deviation rows report p95, never min.
    expect(cellsOf("final lateral deviation (m)")).toEqual(
      { mean: "101.0", p95: "171.7", min: "—", max: "179.5" },
    );
    expect(cellsOf("final vertical |deviation| (m)")).toEqual(
      { mean: "13.5", p95: "24.2", min: "—", max: "25.4" },
    );
    // The signed row is the only place a high/low bias is visible; the |…| row
    // above averages -13.5 to +13.5. Sign must survive to the cell.
    expect(cellsOf(/final vertical deviation, signed/)).toEqual(
      { mean: "-13.5", p95: "—", min: "—", max: "—" },
    );
    // Time rows report min, never p95 — these used to land in the same column.
    expect(cellsOf("flight time (s)")).toEqual(
      { mean: "364.6", p95: "—", min: "329.1", max: "400.2" },
    );
    expect(cellsOf(/Δt vs observed/)).toEqual(
      { mean: "-43.9", p95: "—", min: "-206.9", max: "119.2" },
    );
    // Path-shape carries only mean and max.
    expect(cellsOf("path-shape deviation, lateral (m)")).toEqual(
      { mean: "11142.1", p95: "—", min: "—", max: "21541.1" },
    );
  });

  it("closes via the Close button", () => {
    const onClose = vi.fn();
    render(
      <EvaluationReportWindow
        report={REPORT}
        title="Optimization Evaluation Report"
        subtitle="x"
        onClose={onClose}
      />,
    );
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
    render(
      <EvaluationReportWindow
        report={bare}
        title="Optimization Evaluation Report"
        subtitle="x"
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText("No solved trajectories to chart.")).toBeTruthy();
    expect(screen.queryByText(/mean Δt vs observed/)).toBeNull();
  });

  it("excludes missing and non-finite deviations from plots and identifies them", () => {
    const observed: EvaluationReport = {
      ...REPORT,
      subject: "observed",
      total: 3,
      measured: 1,
      solved: 3,
      solve_rate: 1,
      successful: 1,
      success_rate: 1 / 3,
      success_rate_among_solved: 1 / 3,
      observed: {
        established: 1,
        not_established: 2,
        established_rate: 1 / 3,
        marginal: 0,
      },
      reference: null,
      trajectories: [
        {
          id: "MEASURED", file: "measured.json", solved: true, success: true,
          subject: "observed", established: true, violations: [],
          lateral_m: 12.5, vertical_m: 1.25, final_time_s: 320,
        },
        {
          id: "NO_CROSSING", file: "missing.json", solved: true, success: false,
          subject: "observed", established: false, violations: ["not_established"],
          lateral_m: null, vertical_m: null,
          reason: "vertical RMS 8.0 m > 6.0 m",
        },
        {
          id: "NON_FINITE", file: "invalid.json", solved: true, success: false,
          subject: "observed", established: false, violations: ["not_established"],
          lateral_m: Number.POSITIVE_INFINITY, vertical_m: 2, final_time_s: 315,
          reason: "invalid derived deviation",
        },
      ],
    };

    render(
      <EvaluationReportWindow
        report={observed}
        title="Observed Baseline Evaluation Report"
        subtitle="KRDU"
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText(/2 solved flights excluded from deviation charts/i).textContent)
      .toContain("1 not measured; 1 invalid/non-finite");

    const lateralChart = screen.getByRole("img", {
      name: "Final lateral deviation, worst → best",
    });
    const verticalChart = screen.getByRole("img", {
      name: "Final vertical deviation, low → high",
    });
    expect(lateralChart.querySelectorAll("circle")).toHaveLength(1);
    expect(verticalChart.querySelectorAll("circle")).toHaveLength(1);
    expect(lateralChart.textContent).toContain("MEASURED: 12.50 m");
    expect(lateralChart.textContent).not.toContain("NO_CROSSING");
    expect(lateralChart.textContent).not.toContain("NON_FINITE");

    const missingRow = screen.getByText("NO_CROSSING").closest("tr")!;
    const invalidRow = screen.getByText("NON_FINITE").closest("tr")!;
    expect(within(missingRow).getByText("not established")).toBeTruthy();
    expect(within(invalidRow).getByText("invalid (non-finite)")).toBeTruthy();
  });
});
