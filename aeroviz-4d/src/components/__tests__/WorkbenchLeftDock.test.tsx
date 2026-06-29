import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const { appState, setMode } = vi.hoisted(() => ({
  appState: { mode: "observe" as string },
  setMode: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setMode }),
}));

// Stub the heavy child panels so the dock's mode→panel mapping is what's under test.
vi.mock("../ControlPanel", () => ({ default: () => <div>CONTROL_PANEL</div> }));
vi.mock("../FlightTable", () => ({
  default: ({ flightIds }: { flightIds: string[] }) => <div>FLIGHTS:{flightIds.length}</div>,
}));
vi.mock("../PilotPanel", () => ({
  default: ({ mode }: { mode: string }) => <div>PILOT:{mode}</div>,
}));

import WorkbenchLeftDock from "../WorkbenchLeftDock";

function renderDock() {
  return render(
    <WorkbenchLeftDock flightIds={["a", "b", "c"]} procedures={<div>PROCEDURES</div>} />,
  );
}

describe("WorkbenchLeftDock", () => {
  beforeEach(() => {
    appState.mode = "observe";
    vi.clearAllMocks();
  });

  it("shows the trajectory controls + flight list in observe mode", () => {
    renderDock();
    expect(screen.getByText("CONTROL_PANEL")).toBeTruthy();
    expect(screen.getByText("FLIGHTS:3")).toBeTruthy();
    expect(screen.queryByText(/PILOT:/)).toBeNull();
  });

  it("shows the procedure tree in procedures mode", () => {
    appState.mode = "procedures";
    renderDock();
    expect(screen.getByText("PROCEDURES")).toBeTruthy();
    expect(screen.queryByText("CONTROL_PANEL")).toBeNull();
  });

  it("drives the PilotPanel sub-mode for fly / optimize / compare", () => {
    appState.mode = "fly";
    const { rerender } = renderDock();
    expect(screen.getByText("PILOT:pilot")).toBeTruthy();

    appState.mode = "optimize";
    rerender(<WorkbenchLeftDock flightIds={[]} procedures={<div>PROCEDURES</div>} />);
    expect(screen.getByText("PILOT:trajectory")).toBeTruthy();

    appState.mode = "compare";
    rerender(<WorkbenchLeftDock flightIds={[]} procedures={<div>PROCEDURES</div>} />);
    expect(screen.getByText("PILOT:comparison")).toBeTruthy();
  });
});
