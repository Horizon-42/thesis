import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import catalogMirror from "../../data/__tests__/fixtures/trafficScenarioCatalog.json";
import type { TrafficJobStatus, TrafficScenarioArrival, TrafficScenarioCatalog } from "../../data/trafficJobs";

/**
 * Starting a job must leave the scenario list as the user set it: its sort, its runway filter, the light switch, the page and
 * the pick. The REAL AppProvider, dock and panel (what a start does to the context — the scene removed, the job's files fed
 * back — is the real thing); only the backend's client and the index file are scripts.
 */

const { api, fetchIndex, pilot } = vi.hoisted(() => ({
  pilot: { fetchPilotAircraftConfigs: vi.fn() },
  api: { fetchTrafficScenarios: vi.fn(), startTrafficJob: vi.fn(), fetchTrafficJob: vi.fn(),
    cancelTrafficJob: vi.fn(), beaconCancelTrafficJob: vi.fn() },
  fetchIndex: vi.fn(),
}));
vi.mock("../../data/trafficJobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../data/trafficJobs")>()),
  ...api,
}));
vi.mock("../../pilot/pilotClient", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../pilot/pilotClient")>()),
  ...pilot,
}));
vi.mock("../../utils/fetchJson", async (importOriginal) => {
  const original = await importOriginal<typeof import("../../utils/fetchJson")>();
  // the job's files are scripted; the provider's own files (the airports) come through the stubbed network
  return { ...original, fetchJson: (url: string) => (url.includes("/traffic/jobs/") ? fetchIndex(url) : original.fetchJson(url)) };
});

import { AppProvider, useApp } from "../../context/AppContext";
import WorkbenchLeftDock from "../WorkbenchLeftDock";

const JOB = "20261006T120000123456Z-0123abcd";
const CATALOG = catalogMirror as TrafficScenarioCatalog;
const many: TrafficScenarioArrival[] = Array.from({ length: 250 }, (_, n) => ({
  ...CATALOG.m1[1], flightKey: `X${n}_05L_a_${n}`, callsign: `X${n}`, lossInstants: 250 - n, runway: n % 2 ? "05L" : "05R",
  landingUtc: new Date(Date.parse("2026-05-01T00:00:00Z") + n * 600_000).toISOString().replace(".000Z", "Z") }));
const BIG: TrafficScenarioCatalog = { ...CATALOG, m1: many, counts: { ...CATALOG.counts, withLoss: 250 } };
const change = { type: "B38M", firstSolveLosses: 9, finalLosses: 0, landingVsRecordS: -12, delayS: null, blockCheckLosses: null,
  optimizeS: 4.2, optimizeCpuS: 4.0, solves: 2, failedSolves: 0 };
const running = (phase: string): TrafficJobStatus => ({ state: "running", error: null,
  progress: { done: 0, total: 1, current: null, phase } });
const done = (key: string): TrafficJobStatus => ({ state: "done", error: null, summary: {},
  progress: { done: 1, total: 1, current: key, phase: "building the scene" }, perAircraft: { [key]: change },
  timing: { totalS: 20, phases: { "reading traffic": 5, optimizing: 15 } }, stayedRecords: {} });

function Probe() {
  const { setMode } = useApp();
  useEffect(() => { setMode("optimize"); }, [setMode]);
  return null;
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true, headers: { get: () => "application/json" },
    text: async () => JSON.stringify({ defaultAirport: "KRDU", airports: [
      { code: "KRDU", name: "Raleigh-Durham International Airport", lat: 35.878659, lon: -78.7873 }] }),
  })));
  for (const mock of [...Object.values(api), ...Object.values(pilot), fetchIndex]) mock.mockReset();
  pilot.fetchPilotAircraftConfigs.mockResolvedValue([]);
  api.fetchTrafficScenarios.mockResolvedValue(BIG);
  api.startTrafficJob.mockResolvedValue(JOB);
  api.cancelTrafficJob.mockResolvedValue({ ...running("optimizing 1 of 1"), state: "cancelled" });      // the panel's unmount cancels its job
});
afterEach(() => vi.unstubAllGlobals());

const list = () => within(screen.getByLabelText("Scenario list"));
const names = () => within(screen.getByLabelText("Arrivals with a loss")).getAllByRole("option")
  .map((li) => li.firstElementChild!.textContent!.split(" · ")[0]);

describe("starting a job", () => {
  it("leaves the list's sort, runway filter, light switch, page and pick as the user set them, through to the finished scene", async () => {
    render(<AppProvider><Probe /><WorkbenchLeftDock flightIds={[]} flightSummaries={{}} /></AppProvider>);
    await waitFor(() => expect(screen.getByLabelText("Mode")).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Mode"), { target: { value: "m1" } });
    await waitFor(() => expect(list().getByText("Showing 1–100 of 250", { exact: false })).toBeTruthy());

    fireEvent.change(list().getByLabelText("Sort by"), { target: { value: "time-desc" } });
    fireEvent.change(list().getByLabelText("Show runway"), { target: { value: "05L" } });
    fireEvent.click(list().getByLabelText("Hide light aircraft"));                  // off
    fireEvent.click(list().getByRole("button", { name: "Show 25 more" }));                  // 125 match: one page and 25
    const before = names();
    expect(before).toHaveLength(125);
    fireEvent.click(screen.getByText(/^X101 · /));
    const selection = screen.getByLabelText("Selection").textContent;
    expect(selection).toMatch(/^Selected: X101 — /);

    const kept = () => {
      expect((list().getByLabelText("Sort by") as HTMLSelectElement).value).toBe("time-desc");
      expect((list().getByLabelText("Show runway") as HTMLSelectElement).value).toBe("05L");
      expect((list().getByLabelText("Hide light aircraft") as HTMLInputElement).checked).toBe(false);
      expect(names()).toEqual(before);                                                       // the order and the page
      expect(screen.getByLabelText("Selection").textContent).toBe(selection);
      expect(list().getByText(/^X101 · /).closest("li")!.getAttribute("aria-selected")).toBe("true");
    };
    kept();

    api.fetchTrafficJob.mockResolvedValue(running("optimizing 1 of 1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByText("Computing")).toBeTruthy());
    kept();                                                                                   // while it runs

    const key = "X101_05L_a_101";
    api.fetchTrafficJob.mockResolvedValue(done(key));
    fetchIndex.mockResolvedValue({
      schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
      referenceSource: "canonicalObserved", evaluationReport: "r.json", groups: [{
        group: key, flightId: "X101", runway: "05L", airport: "KRDU", status: "solved", finalTimeS: 300, initialState: null,
        entities: [`ref-${key}`, `sim-${key}`], czml: "a.czml",
        traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } }] });
    await waitFor(() => expect(screen.getByText("Ready")).toBeTruthy(), { timeout: 5000 });
    kept();                                                                                   // and when the scene is up
  });

  it("does the same for the blocks of an M2 job: the length, the sort, the runway, the page and the pick stay", async () => {
    const blocks = Array.from({ length: 300 }, (_, n) => ({
      startUtc: new Date(Date.parse("2026-05-01T00:00:00Z") + n * 900_000).toISOString().replace(".000Z", "Z"),
      arrivals: 2, commandable: 2, lossInstants: n % 11, runways: n % 2 ? ["05L"] : ["05L", "05R"] }));
    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, m2: { ...CATALOG.m2, "900": blocks } });
    render(<AppProvider><Probe /><WorkbenchLeftDock flightIds={[]} flightSummaries={{}} /></AppProvider>);
    await waitFor(() => expect(screen.getByLabelText("Mode")).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Mode"), { target: { value: "m2" } });
    await waitFor(() => expect(list().getByLabelText("Length")).toBeTruthy());

    fireEvent.change(list().getByLabelText("Length"), { target: { value: "900" } });
    fireEvent.change(list().getByLabelText("Sort by"), { target: { value: "time-desc" } });
    fireEvent.change(list().getByLabelText("Show runway"), { target: { value: "05R" } });        // the 150 blocks that hold 05R
    fireEvent.click(list().getByRole("button", { name: "Show 50 more" }));
    const starts = () => within(screen.getByLabelText("Blocks")).getAllByRole("option").map((li) => li.firstElementChild!.textContent);
    const before = starts();
    expect(before).toHaveLength(150);
    fireEvent.click(list().getByText(before[7]!));
    const selection = screen.getByLabelText("Selection").textContent;

    api.fetchTrafficJob.mockResolvedValue(running("earliest arrival of each aircraft"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByText("Computing")).toBeTruthy());
    expect((list().getByLabelText("Length") as HTMLSelectElement).value).toBe("900");
    expect((list().getByLabelText("Sort by") as HTMLSelectElement).value).toBe("time-desc");
    expect((list().getByLabelText("Show runway") as HTMLSelectElement).value).toBe("05R");
    expect(starts()).toEqual(before);
    expect(screen.getByLabelText("Selection").textContent).toBe(selection);
  });
});
