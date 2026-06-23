import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import DynamicsComparisonCharts from "../DynamicsComparisonCharts";
import type {
  DynamicsComparisonChart,
  DynamicsComparisonSystem,
} from "../../pilot/dynamicsComparisonClient";

const systems: DynamicsComparisonSystem[] = [
  { key: "A", label: "A · fixed tangent ENU", colorRgba: [244, 114, 22, 240], isReference: false },
  { key: "B", label: "B · re-anchored ENU", colorRgba: [226, 232, 240, 245], isReference: true },
  { key: "C", label: "C · geodetic +transport", colorRgba: [56, 189, 248, 240], isReference: false },
  { key: "D", label: "D · geodetic no transport", colorRgba: [250, 204, 21, 240], isReference: false },
];

const chart: DynamicsComparisonChart = {
  distanceKm: [0, 0.5, 1.0],
  timeS: [0, 3.8, 7.6],
  series: {
    A: { horiz: [0, 5, 12], alt: [0, -1, -3], head: [0, 0.01, 0.02], speed: [0, 0.1, 0.2] },
    C: { horiz: [0, 0.001, 0.002], alt: [0, 0, 0], head: [0, 0, 0], speed: [0, 0, 0] },
    D: { horiz: [0, 4, 9], alt: [0, -2, -5], head: [0, 0.05, 0.1], speed: [0, 0.1, 0.2] },
  },
  final: {
    A: { horiz: 12, alt: -3, head: 0.02, speed: 0.2 },
    C: { horiz: 0.002, alt: 0, head: 0, speed: 0 },
    D: { horiz: 9, alt: -5, head: 0.1, speed: 0.2 },
  },
};

function renderCharts(overrides: Partial<React.ComponentProps<typeof DynamicsComparisonCharts>> = {}) {
  const onToggleSystem = vi.fn();
  const onClose = vi.fn();
  render(
    <DynamicsComparisonCharts
      chart={chart}
      systems={systems}
      hiddenKeys={[]}
      onToggleSystem={onToggleSystem}
      onClose={onClose}
      {...overrides}
    />,
  );
  return { onToggleSystem, onClose };
}

describe("DynamicsComparisonCharts", () => {
  // The drag test overrides window.innerWidth/innerHeight; restore jsdom's live
  // getters afterwards so it can't pollute later tests in the file.
  afterEach(() => {
    delete (window as { innerWidth?: number }).innerWidth;
    delete (window as { innerHeight?: number }).innerHeight;
  });

  it("renders the four deviation charts and a final-value table", () => {
    renderCharts();
    expect(screen.getByText("Horizontal error")).toBeTruthy();
    expect(screen.getByText("Altitude error")).toBeTruthy();
    expect(screen.getByText("Heading error")).toBeTruthy();
    expect(screen.getByText("Speed error")).toBeTruthy();

    // final table has a row per compared system (A/C/D), not the reference B
    const table = screen.getByRole("table");
    const rowA = within(table).getByRole("row", { name: /^A/ });
    expect(within(table).queryByRole("row", { name: /^B/ })).toBeNull();
    // horiz is a magnitude (no sign); speed is signed (+0.20)
    expect(within(rowA).getByText("12.0")).toBeTruthy();
    expect(within(rowA).getByText("+0.20")).toBeTruthy();
  });

  it("toggles a system when its legend checkbox is clicked", async () => {
    const { onToggleSystem } = renderCharts();
    const checkbox = screen.getByRole("checkbox", { name: /fixed tangent ENU/ });
    await userEvent.click(checkbox);
    expect(onToggleSystem).toHaveBeenCalledWith("A");
  });

  it("omits a hidden system's line from the charts (one fewer path per chart)", () => {
    // The window portals into <body>, so count paths in baseElement (document.body)
    // and rerender in place rather than comparing two separate containers.
    const { baseElement, rerender } = render(
      <DynamicsComparisonCharts
        chart={chart}
        systems={systems}
        hiddenKeys={[]}
        onToggleSystem={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const fullPaths = baseElement.querySelectorAll(".dyncmp-chart svg path").length;

    rerender(
      <DynamicsComparisonCharts
        chart={chart}
        systems={systems}
        hiddenKeys={["A"]}
        onToggleSystem={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const hiddenPaths = baseElement.querySelectorAll(".dyncmp-chart svg path").length;

    // 4 charts × 1 line removed = 4 fewer paths
    expect(fullPaths - hiddenPaths).toBe(4);
  });

  it("calls onClose from the Close button", async () => {
    const { onClose } = renderCharts();
    await userEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("starts centered and becomes a positioned floating window after dragging the title bar", () => {
    Object.defineProperty(window, "innerWidth", { value: 1280, configurable: true });
    Object.defineProperty(window, "innerHeight", { value: 800, configurable: true });

    renderCharts();
    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("is-centered");

    // jsdom's PointerEvent drops clientX/Y; dispatch MouseEvents with the
    // pointer* types (React maps by type and reads clientX from the MouseEvent).
    const header = dialog.querySelector(".dyncmp-charts-header") as HTMLElement;
    fireEvent(header, new MouseEvent("pointerdown", { clientX: 120, clientY: 60, bubbles: true }));
    fireEvent(header, new MouseEvent("pointermove", { clientX: 320, clientY: 260, bubbles: true }));
    fireEvent(header, new MouseEvent("pointerup", { clientX: 320, clientY: 260, bubbles: true }));

    // No longer centered; positioned at (move - grab offset).
    expect(dialog.className).not.toContain("is-centered");
    expect(dialog.style.left).toBe("200px");
    expect(dialog.style.top).toBe("200px");
  });
});
