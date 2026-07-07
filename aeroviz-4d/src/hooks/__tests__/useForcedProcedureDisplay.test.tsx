import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";

const { appState, setSelectedRunway, setProceduresOpen, toggleLayer, setApproachViewOpen } = vi.hoisted(() => ({
  appState: {
    selectedRunway: null as string | null,
    proceduresOpen: false as boolean,
    layers: { procedures: false } as Record<string, boolean>,
    isApproachViewOpen: false as boolean,
  },
  setSelectedRunway: vi.fn(),
  setProceduresOpen: vi.fn(),
  toggleLayer: vi.fn(),
  setApproachViewOpen: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    selectedRunway: appState.selectedRunway,
    setSelectedRunway,
    proceduresOpen: appState.proceduresOpen,
    setProceduresOpen,
    layers: appState.layers,
    toggleLayer,
    isApproachViewOpen: appState.isApproachViewOpen,
    setApproachViewOpen,
  }),
}));

import { useForcedProcedureDisplay } from "../useForcedProcedureDisplay";

function Harness({ active, forceRunway }: { active: boolean; forceRunway: string | null }) {
  useForcedProcedureDisplay({ active, forceRunway });
  return null;
}

describe("useForcedProcedureDisplay", () => {
  beforeEach(() => {
    appState.selectedRunway = null;
    appState.proceduresOpen = false;
    appState.layers = { procedures: false };
    appState.isApproachViewOpen = false;
    vi.clearAllMocks();
  });

  it("owns selectedRunway when forceRunway is set: forces on entry, restores on exit", () => {
    const { rerender } = render(<Harness active forceRunway="05L" />);

    // Force: scope the runway, open the panel, enable the geometry layer.
    expect(setSelectedRunway).toHaveBeenCalledWith("05L");
    expect(setProceduresOpen).toHaveBeenCalledWith(true);
    expect(toggleLayer).toHaveBeenCalledWith("procedures");

    // Reflect the applied forced state — plus the user opening the target's 2D profile
    // mid-force (it keys on the owned selectedRunway) — then drop the force.
    appState.selectedRunway = "05L";
    appState.proceduresOpen = true;
    appState.layers = { procedures: true };
    appState.isApproachViewOpen = true;
    vi.clearAllMocks();

    rerender(<Harness active={false} forceRunway="05L" />);

    // Restore to the pre-force snapshot (runway null, panel closed, layer back off) —
    // including the approach view keyed on the owned runway (closed at save time), so it
    // can't silently retarget to (or blank on) the restored runway.
    expect(setSelectedRunway).toHaveBeenCalledWith(null);
    expect(setProceduresOpen).toHaveBeenCalledWith(false);
    expect(toggleLayer).toHaveBeenCalledWith("procedures");
    expect(setApproachViewOpen).toHaveBeenCalledWith(false);
  });

  it("restores on unmount", () => {
    const { unmount } = render(<Harness active forceRunway="05L" />);
    appState.selectedRunway = "05L";
    appState.proceduresOpen = true;
    appState.layers = { procedures: true };
    vi.clearAllMocks();

    unmount();

    expect(setProceduresOpen).toHaveBeenCalledWith(false);
    expect(setSelectedRunway).toHaveBeenCalledWith(null);
  });

  it("NEVER touches selectedRunway (nor the approach view) when forceRunway is null (Observe)", () => {
    appState.selectedRunway = "05L"; // the user-owned global selection
    appState.isApproachViewOpen = true; // a user-opened profile — also user-owned here
    const { rerender } = render(<Harness active forceRunway={null} />);

    // Only the panel + layer are driven; the runway and profile are left entirely alone.
    expect(setProceduresOpen).toHaveBeenCalledWith(true);
    expect(toggleLayer).toHaveBeenCalledWith("procedures");
    expect(setSelectedRunway).not.toHaveBeenCalled();
    expect(setApproachViewOpen).not.toHaveBeenCalled();

    appState.proceduresOpen = true;
    appState.layers = { procedures: true };
    vi.clearAllMocks();

    rerender(<Harness active={false} forceRunway={null} />);

    // Restore closes the panel / layer but STILL never writes the runway or profile — so
    // it can't revert the user's top-bar selection (or close their profile) on exit.
    expect(setProceduresOpen).toHaveBeenCalledWith(false);
    expect(setSelectedRunway).not.toHaveBeenCalled();
    expect(setApproachViewOpen).not.toHaveBeenCalled();
  });

  it("does not force anything while inactive", () => {
    render(<Harness active={false} forceRunway="05L" />);
    expect(setSelectedRunway).not.toHaveBeenCalled();
    expect(setProceduresOpen).not.toHaveBeenCalled();
    expect(toggleLayer).not.toHaveBeenCalled();
  });
});
