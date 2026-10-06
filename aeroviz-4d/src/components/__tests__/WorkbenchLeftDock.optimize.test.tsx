import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The Optimize task's mode selector and the multi-aircraft job's life through the dock: the REAL panel and hook, the
 * backend client answering from a script.
 */

const { api, app, pilotMounts, pilot } = vi.hoisted(() => ({
  pilot: { fetchPilotAircraftConfigs: vi.fn() },
  api: {
    fetchTrafficScenarios: vi.fn(),
    startTrafficJob: vi.fn(),
    fetchTrafficJob: vi.fn(),
    cancelTrafficJob: vi.fn(),
    beaconCancelTrafficJob: vi.fn(),
  },
  app: {
    mode: "optimize" as string,
    activeAirportCode: "KRDU",
    viewer: null as unknown,
    selectedFlightId: null as string | null,
    setSelectedFlightId: vi.fn(),
    setTrafficScene: vi.fn(),
    setTrajectoryComparisonKind: vi.fn(),
    trajectoryComparisonKinds: { reference: true, optimizer: false, simulator: true, predicted: true, lookback: true },
  },
  pilotMounts: { count: 0 },
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => app }));
vi.mock("../../data/trafficJobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../data/trafficJobs")>()),
  ...api,
}));
vi.mock("../../pilot/pilotClient", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../pilot/pilotClient")>()),
  ...pilot,
}));
vi.mock("../ControlPanel", () => ({ default: () => <div>CONTROL_PANEL</div> }));
vi.mock("../FlightTable", () => ({ default: () => <div>FLIGHTS</div> }));
vi.mock("../EvaluationSummary", () => ({ default: () => <div>EVAL_SUMMARY</div> }));
vi.mock("../TrainingPanel", () => ({ default: () => <div>TRAINING_PANEL</div> }));
// A stand-in with a state of its own, as the real panel has (the aircraft set up in Fly, a computed trajectory in Optimize).
vi.mock("../PilotPanel", async () => {
  const { useEffect, useState } = await import("react");
  return {
    default: ({ mode, hidden = false }: { mode: string; hidden?: boolean }) => {
      const [setup, setSetup] = useState(0);
      useEffect(() => { pilotMounts.count += 1; }, []);
      return (
        <div hidden={hidden} data-testid="pilot-panel">
          PILOT:{mode}
          <button type="button" onClick={() => setSetup((n) => n + 1)}>set up</button>
          <output>setup {setup}</output>
        </div>
      );
    },
  };
});

import WorkbenchLeftDock from "../WorkbenchLeftDock";
import catalogMirror from "../../data/__tests__/fixtures/trafficScenarioCatalog.json";

const JOB = "20261006T120000123456Z-0123abcd";
const running = { state: "running", progress: { done: 0, total: 1, current: null, phase: "optimizing 1 of 1" }, error: null };

const dock = () => <WorkbenchLeftDock flightIds={[]} flightSummaries={{}} />;
const kindSelect = () => screen.getByLabelText("Mode") as HTMLSelectElement;
const show = (mode: string) => { app.mode = mode; };

beforeEach(() => {
  vi.useFakeTimers();
  window.localStorage.clear();
  for (const mock of [...Object.values(api), ...Object.values(pilot), app.setSelectedFlightId, app.setTrafficScene]) mock.mockReset();
  pilot.fetchPilotAircraftConfigs.mockResolvedValue([]);
  api.fetchTrafficScenarios.mockResolvedValue(catalogMirror);   // a MIRROR of the census's catalog (data/__tests__/trafficJobs.test.ts)
  api.startTrafficJob.mockResolvedValue(JOB);
  api.fetchTrafficJob.mockResolvedValue(running);
  api.cancelTrafficJob.mockResolvedValue({ ...running, state: "cancelled" });
  pilotMounts.count = 0;
  app.mode = "optimize";
  app.activeAirportCode = "KRDU";
});
afterEach(() => vi.useRealTimers());

const settle = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });

async function startM1Job() {
  fireEvent.change(kindSelect(), { target: { value: "m1" } });
  await settle();
  fireEvent.click(screen.getByText(/^SWA3131 · /));
  fireEvent.click(screen.getByRole("button", { name: "Start" }));
  await settle();
}

describe("the Optimize task's mode selector", () => {
  it("offers the single aircraft (the default, the PilotPanel) and the two multi-aircraft modes", () => {
    render(dock());
    expect(screen.getByText("PILOT:optimize", { exact: false })).toBeTruthy();
    expect(Array.from(kindSelect().options).map((o) => [o.value, o.textContent])).toEqual([
      ["single", "Single aircraft"],
      ["m1", "Multi-aircraft: one controlled"],
      ["m2", "Multi-aircraft: all controlled"],
    ]);
    expect(kindSelect().value).toBe("single");
  });

  it("shows the job panel of the mode chosen in place of the single-aircraft panel, which stays mounted, hidden", () => {
    render(dock());
    const pilot = () => screen.getByTestId("pilot-panel");
    expect(pilot().hidden).toBe(false);
    fireEvent.change(kindSelect(), { target: { value: "m1" } });
    expect(pilot().hidden).toBe(true);
    expect(screen.getByText("One controlled")).toBeTruthy();
    fireEvent.change(kindSelect(), { target: { value: "m2" } });
    expect(pilot().hidden).toBe(true);
    expect(screen.getByText("All controlled")).toBeTruthy();
    fireEvent.change(kindSelect(), { target: { value: "single" } });
    expect(pilot().hidden).toBe(false);
    expect(screen.queryByText("All controlled")).toBeNull();
    expect(pilotMounts.count).toBe(1);
  });

  it("never loses what was set up in the PilotPanel: single ↔ multi, and Fly → Optimize multi → Fly", () => {
    show("fly");
    const { rerender } = render(dock());
    fireEvent.click(screen.getByRole("button", { name: "set up" }));
    fireEvent.click(screen.getByRole("button", { name: "set up" }));
    expect(screen.getByText("setup 2")).toBeTruthy();                            // Fly's setup

    show("optimize");                                                            // Fly → Optimize (single)
    rerender(dock());
    fireEvent.change(kindSelect(), { target: { value: "m2" } });                 // → multi-aircraft
    expect(screen.getByTestId("pilot-panel").hidden).toBe(true);
    expect(screen.getByText("All controlled")).toBeTruthy();
    show("fly");                                                                 // → Fly
    rerender(dock());
    expect(screen.getByTestId("pilot-panel").hidden).toBe(false);
    expect(screen.getByText("PILOT:fly", { exact: false })).toBeTruthy();
    expect(screen.getByText("setup 2")).toBeTruthy();                            // kept
    expect(pilotMounts.count).toBe(1);

    show("optimize");                                                            // back: the mode chosen is multi still
    rerender(dock());
    expect(screen.getByTestId("pilot-panel").hidden).toBe(true);
    fireEvent.change(kindSelect(), { target: { value: "single" } });
    expect(screen.getByText("setup 2")).toBeTruthy();
    expect(pilotMounts.count).toBe(1);
  });

  it("is Optimize's alone: Fly has no selector and Evaluate no job panel", () => {
    show("fly");
    const { rerender } = render(dock());
    expect(screen.getByText("PILOT:fly", { exact: false })).toBeTruthy();
    expect(screen.queryByLabelText("Mode")).toBeNull();
    show("evaluation");
    rerender(dock());
    expect(screen.queryByLabelText("Mode")).toBeNull();
    expect(screen.getByText("CONTROL_PANEL")).toBeTruthy();
  });

  it("leaves Fly and Optimize (single) on one PilotPanel instance, as before the selector", () => {
    show("fly");
    const { rerender } = render(dock());
    show("optimize");
    rerender(dock());
    show("fly");
    rerender(dock());
    expect(pilotMounts.count).toBe(1);
  });
});

describe("a multi-aircraft job through the dock", () => {
  it("is cancelled, and its scene removed, by a change of mode", async () => {
    render(dock());
    await startM1Job();
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);
    app.setTrafficScene.mockClear();

    await act(async () => { fireEvent.change(kindSelect(), { target: { value: "single" } }); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(app.setTrafficScene).toHaveBeenCalledWith(null);
    expect(screen.getByTestId("pilot-panel").hidden).toBe(false);
  });

  it("is cancelled by the other multi-aircraft mode, and the next panel starts empty", async () => {
    render(dock());
    await startM1Job();
    await act(async () => { fireEvent.change(kindSelect(), { target: { value: "m2" } }); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(screen.getByText("Standby")).toBeTruthy();
    expect(screen.queryByText("Computing")).toBeNull();
  });

  it("is cancelled, and its scene removed, by leaving the Optimize task", async () => {
    const { rerender } = render(dock());
    await startM1Job();
    app.setTrafficScene.mockClear();

    show("evaluation");
    await act(async () => { rerender(dock()); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(app.setTrafficScene).toHaveBeenCalledWith(null);
    expect(screen.queryByText("Computing")).toBeNull();

    show("optimize");                                      // back in Optimize: the mode is kept, the job is not
    await act(async () => { rerender(dock()); });
    expect(kindSelect().value).toBe("m1");
    expect(screen.getByText("Standby")).toBeTruthy();
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);
  });

  it("is cancelled by another airport", async () => {
    const { rerender } = render(dock());
    await startM1Job();
    app.activeAirportCode = "KSMF";
    await act(async () => { rerender(dock()); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
  });
});
