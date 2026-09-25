import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const { appState, setMode, trainingMounts } = vi.hoisted(() => ({
  appState: { mode: "observe" as string },
  setMode: vi.fn(),
  trainingMounts: { count: 0 },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setMode }),
}));

// Stub the heavy child panels so the dock's mode→panel mapping is what's under test.
vi.mock("../ControlPanel", () => ({ default: () => <div>CONTROL_PANEL</div> }));
vi.mock("../FlightTable", () => ({
  default: ({ flightIds }: { flightIds: string[] }) => <div>FLIGHTS:{flightIds.length}</div>,
}));
vi.mock("../EvaluationSummary", () => ({ default: () => <div>EVAL_SUMMARY</div> }));
vi.mock("../PilotPanel", () => ({
  default: ({ mode }: { mode: string }) => <div>PILOT:{mode}</div>,
}));
vi.mock("../TrainingPanel", async () => {
  const { useEffect } = await import("react");
  return {
    default: ({ hidden }: { hidden: boolean }) => {
      useEffect(() => {
        trainingMounts.count += 1;
      }, []);
      return <div hidden={hidden}>TRAINING_PANEL</div>;
    },
  };
});

import WorkbenchLeftDock from "../WorkbenchLeftDock";

function renderDock() {
  return render(<WorkbenchLeftDock flightIds={["a", "b", "c"]} flightSummaries={{}} />);
}

describe("WorkbenchLeftDock", () => {
  beforeEach(() => {
    appState.mode = "observe";
    trainingMounts.count = 0;
    vi.clearAllMocks();
  });

  it("shows the trajectory controls + flight list in observe mode", () => {
    renderDock();
    expect(screen.getByText("CONTROL_PANEL")).toBeTruthy();
    expect(screen.getByText("FLIGHTS:3")).toBeTruthy();
    expect(screen.queryByText(/PILOT:/)).toBeNull();
  });

  // Training must NOT pull in Observe's panels: the observed CZML is loaded only in
  // Observe (it drives the shared Cesium clock), and Training reads its own sample
  // file instead. A dock that rendered ControlPanel here would reintroduce that load.
  it("shows only the TrainingPanel in training mode", () => {
    appState.mode = "training";
    renderDock();
    expect(screen.getByText("TRAINING_PANEL")).toBeTruthy();
    expect(screen.queryByText("CONTROL_PANEL")).toBeNull();
    expect(screen.queryByText(/FLIGHTS:/)).toBeNull();
    expect(screen.queryByText(/PILOT:/)).toBeNull();
  });

  it("mounts the TrainingPanel on its first visit and keeps it, hidden, in the other tasks", () => {
    const { rerender } = renderDock();
    // never opened: nothing of Training is mounted (nothing downloaded)
    expect(screen.queryByText("TRAINING_PANEL")).toBeNull();
    const show = (mode: string) => {
      appState.mode = mode;
      rerender(<WorkbenchLeftDock flightIds={["a", "b", "c"]} flightSummaries={{}} />);
    };
    show("training");
    expect(screen.getByText("TRAINING_PANEL").hidden).toBe(false);
    for (const mode of ["observe", "fly", "compare"]) {
      show(mode);
      expect(screen.getByText("TRAINING_PANEL").hidden).toBe(true);
    }
    expect(screen.getByText("PILOT:comparison")).toBeTruthy();
    show("training");
    expect(screen.getByText("TRAINING_PANEL").hidden).toBe(false);
    expect(screen.queryByText("CONTROL_PANEL")).toBeNull();
    // one session: mounted once, never again
    expect(trainingMounts.count).toBe(1);
  });

  it("drives the PilotPanel sub-mode for fly / optimize / compare", () => {
    appState.mode = "fly";
    const { rerender } = renderDock();
    expect(screen.getByText("PILOT:pilot")).toBeTruthy();

    appState.mode = "optimize";
    rerender(<WorkbenchLeftDock flightIds={[]} flightSummaries={{}} />);
    expect(screen.getByText("PILOT:trajectory")).toBeTruthy();

    appState.mode = "compare";
    rerender(<WorkbenchLeftDock flightIds={[]} flightSummaries={{}} />);
    expect(screen.getByText("PILOT:comparison")).toBeTruthy();
  });
});
