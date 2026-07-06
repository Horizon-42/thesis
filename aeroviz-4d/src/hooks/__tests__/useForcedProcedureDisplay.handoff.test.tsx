import { describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";

// AppProvider fetches the airport manifest on mount; stub it so the real provider mounts
// without hitting the network (its result is irrelevant to the procedure-display state).
vi.mock("../../utils/fetchJson", () => ({
  fetchJson: () =>
    Promise.resolve({ defaultAirport: "KRDU", airports: [{ code: "KRDU", name: "KRDU", lat: 0, lon: 0 }] }),
  isMissingJsonAsset: () => false,
}));

import { AppProvider, useApp } from "../../context/AppContext";
import { useForcedProcedureDisplay } from "../useForcedProcedureDisplay";

// This test uses the REAL AppProvider (real proceduresOpen / layers / selectedRunway state)
// so the outgoing-dock-cleanup → incoming-dock-mount ordering that the handoff bug depends on
// actually happens, rather than being faked by a mock.

let latest: ReturnType<typeof useApp> | null = null;
function Probe() {
  latest = useApp();
  return (
    <>
      <span data-testid="open">{String(latest.proceduresOpen)}</span>
      <span data-testid="layer">{String(latest.layers.procedures)}</span>
    </>
  );
}

// Optimize-style dock: owns the runway (forceRunway set).
function OptimizeDock() {
  useForcedProcedureDisplay({ active: true, forceRunway: "05L" });
  return <span>opt</span>;
}
// Observe-style dock: never touches the runway (forceRunway null).
function ObserveDock() {
  useForcedProcedureDisplay({ active: true, forceRunway: null });
  return <span>obs</span>;
}

function Harness({ dock }: { dock: "opt" | "obs" | "none" }) {
  return (
    <AppProvider>
      <Probe />
      {dock === "opt" ? <OptimizeDock /> : dock === "obs" ? <ObserveDock /> : null}
    </AppProvider>
  );
}

describe("useForcedProcedureDisplay — forcing→forcing dock handoff", () => {
  it("keeps procedures forced open across a switch between two forcing docks, and restores on exit", async () => {
    const { rerender, getByTestId } = render(<Harness dock="none" />);
    // Mirror the real scenario: a global runway is selected before either dock forces.
    await act(async () => {
      latest!.setSelectedRunway("05L");
    });
    expect(getByTestId("open").textContent).toBe("false");
    expect(getByTestId("layer").textContent).toBe("false");

    // Optimize dock forces the procedure display open.
    rerender(<Harness dock="opt" />);
    await act(async () => {});
    expect(getByTestId("open").textContent).toBe("true");
    expect(getByTestId("layer").textContent).toBe("true");

    // Hand off directly to the (also forcing) Observe dock: the outgoing restore and the
    // incoming force race across one flush — procedures must STAY open (the regression).
    rerender(<Harness dock="obs" />);
    await act(async () => {});
    expect(getByTestId("open").textContent).toBe("true");
    expect(getByTestId("layer").textContent).toBe("true");

    // Leaving the forcing dock restores the user's real (closed/off) display — no pollution.
    rerender(<Harness dock="none" />);
    await act(async () => {});
    expect(getByTestId("open").textContent).toBe("false");
    expect(getByTestId("layer").textContent).toBe("false");
  });
});
