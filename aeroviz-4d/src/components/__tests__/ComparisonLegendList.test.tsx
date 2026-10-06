import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ComparisonLegendModel } from "../../utils/comparisonLegend";
import { COMPARISON_STATUS_STYLES } from "../../utils/trajectoryRenderModel";

const { app } = vi.hoisted(() => ({
  app: {
    trajectoryComparisonKinds: { reference: true, optimizer: false, simulator: true, predicted: true, lookback: true },
    setTrajectoryComparisonKind: vi.fn(),
  },
}));
vi.mock("../../context/AppContext", () => ({ useApp: () => app }));

import ComparisonLegendList from "../ComparisonLegendList";

const model = (over: Partial<ComparisonLegendModel> = {}): ComparisonLegendModel => ({
  kinds: ["reference", "simulator"], statuses: [], traffic: "windows", ...over });
const OFF_TARGET = "yellow: the optimized path ended off the runway-threshold target (failed the gates)";

beforeEach(() => app.setTrajectoryComparisonKind.mockReset());

describe("ComparisonLegendList of a traffic category", () => {
  it("explains the off-target yellow beside the optimized-path entry when the scene has such groups", () => {
    render(<ComparisonLegendList legend={model({ statuses: ["offTargetResult"] })} />);
    const kinds = screen.getAllByText(/Controlled aircraft/, { selector: "label" });
    expect(kinds.map((label) => label.textContent)).toEqual([
      "Controlled aircraft — its record", "Controlled aircraft — optimized path"]);
    const optimized = kinds[1].closest(".comparison-legend-kind")!;
    const explanation = within(optimized as HTMLElement).getByText(OFF_TARGET);
    expect(explanation).toBeTruthy();
    const swatch = explanation.querySelector("i") as HTMLElement;
    expect(swatch.style.background).toBe("rgb(255, 205, 40)");                   // the yellow the builder bakes
    expect(swatch.style.background).toBe(COMPARISON_STATUS_STYLES.offTargetResult.color);
    // said once: not listed again under "Outcome colours"
    expect(screen.queryByText("Outcome colours")).toBeNull();
    expect(screen.queryByText("Off-target optimize result")).toBeNull();
  });

  it("says nothing of yellow when no group of the scene is off target", () => {
    render(<ComparisonLegendList legend={model()} />);
    expect(screen.queryByText(OFF_TARGET)).toBeNull();
    expect(screen.queryByText("Outcome colours")).toBeNull();
  });

  it("names the recorded traffic, and in a scene the ones outside the scheduled set", () => {
    const { rerender } = render(<ComparisonLegendList legend={model()} />);
    expect(screen.getByText("Recorded traffic — not controlled")).toBeTruthy();
    rerender(<ComparisonLegendList legend={model({ traffic: "scene" })} />);
    expect(screen.getByText("Recorded traffic — not controlled, outside the scheduled set")).toBeTruthy();
  });

  it("switches the path kinds", () => {
    render(<ComparisonLegendList legend={model()} />);
    fireEvent.click(screen.getByLabelText("Controlled aircraft — its record"));
    expect(app.setTrajectoryComparisonKind).toHaveBeenCalledWith("reference", false);
  });
});

describe("ComparisonLegendList of any other category", () => {
  it("keeps the Outcome colours block, with the off-target row in it", () => {
    render(<ComparisonLegendList legend={model({ traffic: null, statuses: ["offTargetResult"] })} />);
    expect(screen.getByLabelText("Reference")).toBeTruthy();
    expect(screen.getByText("Outcome colours")).toBeTruthy();
    expect(screen.getByText("Off-target optimize result")).toBeTruthy();
    expect(screen.queryByText(OFF_TARGET)).toBeNull();
  });
});
