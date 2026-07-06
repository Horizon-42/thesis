import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setSelectedRunway, setRunwayProfileOpen } = vi.hoisted(() => ({
  appState: {
    selectedRunway: null as string | null,
    isRunwayProfileOpen: false as boolean,
  },
  setSelectedRunway: vi.fn(),
  setRunwayProfileOpen: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    selectedRunway: appState.selectedRunway,
    setSelectedRunway,
    isRunwayProfileOpen: appState.isRunwayProfileOpen,
    setRunwayProfileOpen,
  }),
}));

import RunwayProfileToggle from "../RunwayProfileToggle";

describe("RunwayProfileToggle", () => {
  beforeEach(() => {
    appState.selectedRunway = null;
    appState.isRunwayProfileOpen = false;
    vi.clearAllMocks();
  });

  it("is disabled with no runway to govern", () => {
    render(<RunwayProfileToggle runwayIdent={null} />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.textContent).toBe("Profile");
  });

  it("opens the profile and focuses the global runway (bare) when closed", () => {
    render(<RunwayProfileToggle runwayIdent="RW05L" />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.disabled).toBe(false);
    expect(button.getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(button);

    expect(setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(true);
  });

  it("reflects the open state for the matching runway and closes on click", () => {
    appState.isRunwayProfileOpen = true;
    appState.selectedRunway = "05L";
    render(<RunwayProfileToggle runwayIdent="RW05L" />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.getAttribute("aria-pressed")).toBe("true");
    expect(button.textContent).toBe("Hide profile");

    fireEvent.click(button);

    expect(setRunwayProfileOpen).toHaveBeenCalledWith(false);
    expect(setSelectedRunway).not.toHaveBeenCalled();
  });

  it("does not read as open when the profile is open for a different runway", () => {
    appState.isRunwayProfileOpen = true;
    appState.selectedRunway = "23R";
    render(<RunwayProfileToggle runwayIdent="RW05L" />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.getAttribute("aria-pressed")).toBe("false");
    expect(button.textContent).toBe("Profile");
  });

  it("does not read as open with no global selection, and opens on click (null-selection regression)", () => {
    // runwayMatchesSelection(null, X) is match-ALL (filter semantics) — the toggle must
    // not claim an open profile it can't be showing (the page renders nothing without a
    // selection). Every toggle used to flip to "Hide profile" here and a click CLOSED
    // the invisible profile instead of opening this runway's.
    appState.isRunwayProfileOpen = true;
    appState.selectedRunway = null;
    render(<RunwayProfileToggle runwayIdent="RW05L" />);
    const button = screen.getByRole("button") as HTMLButtonElement;
    expect(button.getAttribute("aria-pressed")).toBe("false");
    expect(button.textContent).toBe("Profile");

    fireEvent.click(button);

    expect(setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(true);
    expect(setRunwayProfileOpen).not.toHaveBeenCalledWith(false);
  });

  // ── borrowSelection (unconstrained Optimize's target toggle) ─────────────────
  it("borrows the selection on open and returns it on close", () => {
    appState.selectedRunway = "23R";
    const { rerender } = render(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);

    fireEvent.click(screen.getByRole("button"));
    expect(setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(true);

    // Reflect the applied state, then close from the toggle: the borrow is returned
    // (prior runway + prior closed profile), not just a plain close.
    appState.selectedRunway = "05L";
    appState.isRunwayProfileOpen = true;
    vi.clearAllMocks();
    rerender(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);

    fireEvent.click(screen.getByRole("button"));
    expect(setSelectedRunway).toHaveBeenCalledWith("23R");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(false);
  });

  it("returns the borrowed selection on unmount (leaving the dock)", () => {
    appState.selectedRunway = "23R";
    const { rerender, unmount } = render(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);
    fireEvent.click(screen.getByRole("button"));

    appState.selectedRunway = "05L";
    appState.isRunwayProfileOpen = true;
    vi.clearAllMocks();
    rerender(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);

    unmount();

    expect(setSelectedRunway).toHaveBeenCalledWith("23R");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(false);
  });

  it("does not borrow when opening leaves the selection unchanged", () => {
    // Selection already focused on the governed runway (e.g. constrained Optimize, where
    // useForcedProcedureDisplay owns it): nothing is borrowed, so close is a plain close.
    appState.selectedRunway = "05L";
    const { rerender, unmount } = render(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);
    fireEvent.click(screen.getByRole("button"));
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(true);

    appState.isRunwayProfileOpen = true;
    vi.clearAllMocks();
    rerender(<RunwayProfileToggle runwayIdent="RW05L" borrowSelection />);

    fireEvent.click(screen.getByRole("button"));
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(false);
    expect(setSelectedRunway).not.toHaveBeenCalledWith("23R");

    vi.clearAllMocks();
    unmount();
    expect(setSelectedRunway).not.toHaveBeenCalled();
    expect(setRunwayProfileOpen).not.toHaveBeenCalled();
  });
});
