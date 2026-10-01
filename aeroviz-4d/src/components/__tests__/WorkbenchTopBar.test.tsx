import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const {
  appState,
  setActiveAirportCode,
  setSelectedRunway,
  setMode,
  setProceduresOpen,
  setLayersDrawerOpen,
  setPresentationMode,
  landingsRef,
} = vi.hoisted(() => {
  const appState: any = {
    airports: [
      { code: "KRDU", name: "Raleigh-Durham International Airport", lat: 35.878659, lon: -78.7873 },
      { code: "CYVR", name: "Vancouver International Airport", lat: 49.1939, lon: -123.184 },
    ],
    activeAirportCode: "KRDU",
    selectedRunway: null,
    mode: "evaluation",
    proceduresOpen: false,
    layersDrawerOpen: false,
    presentationMode: false,
  };
  return {
    appState,
    setActiveAirportCode: vi.fn(),
    setSelectedRunway: vi.fn(),
    setMode: vi.fn(),
    setProceduresOpen: vi.fn(),
    setLayersDrawerOpen: vi.fn(),
    setPresentationMode: vi.fn(),
    landingsRef: { current: { manifest: null as unknown, status: "empty" } },
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState,
    setActiveAirportCode,
    setSelectedRunway,
    setMode,
    setProceduresOpen,
    setLayersDrawerOpen,
    setPresentationMode,
  }),
}));

vi.mock("../../hooks/useLandingsManifest", () => ({
  useLandingsManifest: () => landingsRef.current,
}));

import WorkbenchTopBar from "../WorkbenchTopBar";

describe("WorkbenchTopBar", () => {
  beforeEach(() => {
    landingsRef.current = { manifest: null, status: "empty" };
    appState.mode = "evaluation";
    appState.proceduresOpen = false;
    appState.selectedRunway = null;
    appState.layersDrawerOpen = false;
    appState.presentationMode = false;
    vi.clearAllMocks();
  });

  it("renders the four exclusive task tabs and switches mode on click", () => {
    render(<WorkbenchTopBar />);

    for (const label of ["Evaluate", "Learning", "Fly", "Optimize"]) {
      expect(screen.getByRole("button", { name: label })).toBeTruthy();
    }

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
    expect(setMode).toHaveBeenCalledWith("optimize");
  });

  // The user's order (2026-10-01): Fly on its own, a gap, then Optimize, Learning, Evaluate;
  // the Procedures toggle after another gap.
  it("orders the tabs Fly | Optimize, Learning, Evaluate | Procedures and switches to Learning", () => {
    render(<WorkbenchTopBar />);

    const tabs = Array.from(document.querySelectorAll(".workbench-task-switcher .workbench-task-tab"));
    expect(tabs.map((node) => node.textContent)).toEqual([
      "Fly",
      "Optimize",
      "Learning",
      "Evaluate",
      "Procedures",
    ]);
    // a gap starts each group after the first
    expect(tabs.map((node) => node.classList.contains("workbench-task-group-start")))
      .toEqual([false, true, false, false, true]);

    fireEvent.click(screen.getByRole("button", { name: "Learning" }));
    expect(setMode).toHaveBeenCalledWith("training");
  });

  it("toggles the procedures panel independently of the active task", () => {
    appState.mode = "evaluation";
    render(<WorkbenchTopBar />);

    const procedures = screen.getByRole("button", { name: "Procedures" });
    expect(procedures.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(procedures);
    expect(setProceduresOpen).toHaveBeenCalledWith(true);
    // Toggling procedures does not change the active task.
    expect(setMode).not.toHaveBeenCalled();
  });

  it("marks the active task tab pressed", () => {
    appState.mode = "fly";
    render(<WorkbenchTopBar />);
    expect(screen.getByRole("button", { name: "Fly" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Evaluate" }).getAttribute("aria-pressed")).toBe("false");
  });

  it("switches the active airport from the selector", () => {
    render(<WorkbenchTopBar />);
    fireEvent.change(screen.getByLabelText("Active Airport"), { target: { value: "CYVR" } });
    expect(setActiveAirportCode).toHaveBeenCalledWith("CYVR");
  });

  it("shows the landing-runway selector from the manifest and selects a runway", () => {
    landingsRef.current = {
      manifest: {
        airport: "KRDU",
        combined: "trajectories.czml",
        runways: [
          { runway: "23R", file: "landings/KRDU_23R.czml", count: 40 },
          { runway: "05L", file: "landings/KRDU_05L.czml", count: 12 },
        ],
      },
      status: "ready",
    };
    render(<WorkbenchTopBar />);

    const select = screen.getByLabelText("Landing Runway") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      "All runways",
      "23R (40)",
      "05L (12)",
    ]);

    fireEvent.change(select, { target: { value: "23R" } });
    expect(setSelectedRunway).toHaveBeenCalledWith("23R");
    fireEvent.change(select, { target: { value: "" } });
    expect(setSelectedRunway).toHaveBeenCalledWith(null);
  });

  it("hides the runway selector when there are no landings", () => {
    render(<WorkbenchTopBar />);
    expect(screen.queryByLabelText("Landing Runway")).toBeNull();
  });

  it("toggles the layers drawer and presentation mode", () => {
    render(<WorkbenchTopBar />);
    fireEvent.click(screen.getByRole("button", { name: /Layers/ }));
    expect(setLayersDrawerOpen).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: /Present/ }));
    expect(setPresentationMode).toHaveBeenCalledWith(true);
  });
});
