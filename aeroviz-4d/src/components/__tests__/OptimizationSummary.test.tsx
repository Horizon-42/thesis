import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import type { OptimizationStats } from "../../data/airportData";

const { data } = vi.hoisted(() => ({
  data: { stats: null as OptimizationStats | null },
}));

vi.mock("../../hooks/useFlightOptimizerData", () => ({
  useFlightOptimizerData: () => data,
}));

import OptimizationSummary from "../OptimizationSummary";

function row(label: string) {
  return screen.getByText(label).closest(".optimization-summary-row") as HTMLElement;
}

describe("OptimizationSummary", () => {
  beforeEach(() => {
    data.stats = null;
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
});
