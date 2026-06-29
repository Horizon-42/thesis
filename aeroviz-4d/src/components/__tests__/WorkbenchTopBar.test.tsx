import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const {
  appState,
  setActiveAirportCode,
  setSelectedRunway,
  setMode,
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
    mode: "observe",
    layersDrawerOpen: false,
    presentationMode: false,
  };
  return {
    appState,
    setActiveAirportCode: vi.fn(),
    setSelectedRunway: vi.fn(),
    setMode: vi.fn(),
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
    appState.mode = "observe";
    appState.selectedRunway = null;
    appState.layersDrawerOpen = false;
    appState.presentationMode = false;
    vi.clearAllMocks();
  });

  it("renders the five task tabs and switches mode on click", () => {
    render(<WorkbenchTopBar />);

    for (const label of ["Observe", "Procedures", "Fly", "Optimize", "Compare"]) {
      expect(screen.getByRole("button", { name: label })).toBeTruthy();
    }

    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
    expect(setMode).toHaveBeenCalledWith("optimize");
  });

  it("marks the active task tab pressed", () => {
    appState.mode = "compare";
    render(<WorkbenchTopBar />);
    expect(screen.getByRole("button", { name: "Compare" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Observe" }).getAttribute("aria-pressed")).toBe("false");
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
