import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TrafficArrival, TrafficJobStatus } from "../../data/trafficJobs";

const { api, app, fetchIndex } = vi.hoisted(() => ({
  api: {
    fetchTrafficArrivals: vi.fn(),
    startTrafficJob: vi.fn(),
    fetchTrafficJob: vi.fn(),
    cancelTrafficJob: vi.fn(),
    beaconCancelTrafficJob: vi.fn(),
  },
  app: {
    activeAirportCode: "KRDU",
    viewer: null as unknown,
    selectedFlightId: null as string | null,
    setSelectedFlightId: vi.fn(),
    selectedRunway: "05R" as string | null,
    setSelectedRunway: vi.fn(),
    setTrafficScene: vi.fn(),
    setTrajectoryComparisonKind: vi.fn(),
    trajectoryComparisonKinds: { reference: true, optimizer: false, simulator: true, predicted: true, lookback: true },
  },
  fetchIndex: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => app }));
vi.mock("../../data/trafficJobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../data/trafficJobs")>()),
  ...api,
}));
vi.mock("../../utils/fetchJson", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../utils/fetchJson")>()),
  fetchJson: (url: string) => fetchIndex(url),
}));

import TrafficJobPanel from "../TrafficJobPanel";

const JOB = "20261006T120000123456Z-0123abcd";
const DAY = "2026-05-21";
const arrival = (callsign: string, runway: string, landing: string, type: string | null): TrafficArrival => ({
  flightKey: `${callsign}_${runway}_a_${landing}`, callsign, runway, type,
  entryUtc: `${DAY}T${landing}Z`, landingUtc: `${DAY}T${landing}Z` });
const ARRIVALS = [
  arrival("AAL1", "05L", "17:00:10", "A320"), arrival("BAW2", "05R", "17:10:00", "B77W"),
  arrival("DAL3", "05L", "17:40:00", null),
];
const running = (done: number, total: number | null, current: string | null = null): TrafficJobStatus => ({
  state: "running", progress: { done, total, current }, error: null });

const indexOf = (groups: unknown[], scene?: unknown) => ({
  schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
  referenceSource: "canonicalObserved", evaluationReport: "r.json", groups, ...(scene ? { scene } : {}),
});
const group = (arrivalRow: TrafficArrival, extra: Record<string, unknown>) => ({
  group: arrivalRow.flightKey, flightId: arrivalRow.callsign, runway: arrivalRow.runway, airport: "KRDU",
  status: "solved", finalTimeS: 300, initialState: null, entities: [`ref-${arrivalRow.flightKey}`, `sim-${arrivalRow.flightKey}`],
  czml: "a.czml", ...extra });

beforeEach(() => {
  vi.useFakeTimers();
  window.localStorage.clear();
  for (const mock of [...Object.values(api), app.setSelectedFlightId, app.setSelectedRunway, app.setTrafficScene, fetchIndex]) mock.mockReset();
  api.fetchTrafficArrivals.mockResolvedValue({ airport: "KRDU", date: DAY, arrivals: ARRIVALS });
  api.startTrafficJob.mockResolvedValue(JOB);
  api.fetchTrafficJob.mockResolvedValue(running(0, 1));
  api.cancelTrafficJob.mockResolvedValue({ ...running(0, 1), state: "cancelled" });
  app.viewer = null;
  app.selectedFlightId = null;
});
afterEach(() => vi.useRealTimers());

const settle = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

async function pickDay() {
  fireEvent.change(screen.getByLabelText("UTC day"), { target: { value: DAY } });
  await settle();
}

describe("TrafficJobPanel, one controlled (M1)", () => {
  it("asks for a day, lists its arrivals, and starts the job on the arrival picked", async () => {
    render(<TrafficJobPanel kind="m1" />);
    expect(screen.getByText("Pick a UTC day to list the arrivals.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);

    await pickDay();
    expect(api.fetchTrafficArrivals).toHaveBeenCalledWith("KRDU", DAY);
    const rows = screen.getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("cell").map((c) => c.textContent));
    expect(rows).toEqual([["17:00:10", "AAL1", "05L", "A320"], ["17:10:00", "BAW2", "05R", "B77W"], ["17:40:00", "DAL3", "05L", "—"]]);
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);   // none picked yet

    fireEvent.click(screen.getByText("BAW2"));
    expect(screen.getByText("Selected: BAW2")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: ARRIVALS[1].flightKey });
  });

  it("lists a day with a flight that has no callsign, by the first field of its key, and starts it", async () => {
    const nameless = { ...arrival("N123AB", "05L", "17:20:00", "C172"), callsign: null };
    api.fetchTrafficArrivals.mockResolvedValue({ airport: "KRDU", date: DAY, arrivals: [...ARRIVALS, nameless] });
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    expect(screen.getByText("AAL1")).toBeTruthy();                           // the rest of the day is listed
    fireEvent.click(screen.getByText("N123AB"));
    expect(screen.getByText("Selected: N123AB")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: nameless.flightKey });
    api.fetchTrafficJob.mockResolvedValue({ state: "done", progress: { done: 1, total: 1, current: null },
      error: null, summary: {} });
    fetchIndex.mockResolvedValue(indexOf([group(nameless, { flightId: "N123AB",
      traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    expect(within(screen.getByLabelText("Result")).getByRole("status").textContent).toContain(`N123AB, ${DAY} — `);
  });

  it("remembers the day for the next visit", async () => {
    const first = render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    first.unmount();
    render(<TrafficJobPanel kind="m1" />);
    expect((screen.getByLabelText("UTC day") as HTMLInputElement).value).toBe(DAY);
    await settle();
    expect(screen.getByText("AAL1")).toBeTruthy();
  });

  it("shows why the arrivals did not load, and a day without any", async () => {
    api.fetchTrafficArrivals.mockRejectedValueOnce(new Error("HTTP 404 loading /traffic/arrivals"));
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    expect(screen.getByRole("alert").textContent).toContain("HTTP 404");

    api.fetchTrafficArrivals.mockResolvedValueOnce({ airport: "KRDU", date: "2026-05-22", arrivals: [] });
    fireEvent.change(screen.getByLabelText("UTC day"), { target: { value: "2026-05-22" } });
    await settle();
    expect(screen.getByText("No arrival of KRDU lands on 2026-05-22.")).toBeTruthy();
  });

  it("shows the progress while the job runs and cancels it on request", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    api.fetchTrafficJob.mockResolvedValue(running(2, 5, "BAW2_05R_b_2"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();

    expect(screen.getByText("Computing")).toBeTruthy();
    expect(screen.getByText("2 of 5 aircraft done · last: BAW2")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(false);

    api.fetchTrafficJob.mockResolvedValue(running(3, 5, "DAL3_05L_c_3"));
    await settle(2000);                                                      // every 2 s
    expect(screen.getByText("3 of 5 aircraft done · last: DAL3")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await settle();
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(screen.getByText("The job was cancelled.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables Start while a job runs, offers no Restart, and frees it when the job ends", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    const start = () => screen.getByRole("button", { name: "Start" }) as HTMLButtonElement;
    expect(start().disabled).toBe(false);
    fireEvent.click(start());
    await settle();

    expect(screen.queryByRole("button", { name: "Restart" })).toBeNull();
    expect(start().disabled).toBe(true);                                      // the user cancels first
    fireEvent.click(start());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);

    api.fetchTrafficJob.mockResolvedValue({ state: "failed", progress: { done: 0, total: 1, current: null }, error: "x" });
    await settle(2000);
    expect(start().disabled).toBe(false);
  });

  it("keeps Start disabled while the cancel is out", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    let release: (value: unknown) => void = () => undefined;
    api.cancelTrafficJob.mockReturnValue(new Promise((resolve) => { release = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await settle();
    expect(screen.getByText("Cancelling")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(true);
    await act(async () => { release({ ...running(0, 1), state: "cancelled" }); });
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("says the connection is lost, keeps the job and Cancel, and says no more when the backend answers again", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    api.fetchTrafficJob.mockRejectedValue(new Error("Failed to fetch"));
    await settle(2000);

    expect(screen.getByRole("alert").textContent).toContain("Connection lost, retrying");
    expect(screen.getByRole("alert").textContent).toContain("Failed to fetch");
    expect(screen.getByText("Computing")).toBeTruthy();                       // still the same running job
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(false);

    api.fetchTrafficJob.mockResolvedValue(running(1, 2, "AAL1_05L_a_1"));
    await settle(2000);
    expect(screen.queryByText(/Connection lost/)).toBeNull();
    expect(screen.getByText("1 of 2 aircraft done · last: AAL1")).toBeTruthy();
  });

  it("says it is still reading the traffic before the job knows its total", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    api.fetchTrafficJob.mockResolvedValue(running(0, null));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(screen.getByText("Reading the traffic…")).toBeTruthy();
  });

  it("shows a failed job's reason", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    api.fetchTrafficJob.mockResolvedValue({ state: "failed", progress: { done: 0, total: null, current: null },
      error: "NoAircraftDynamics: no aircraft dynamics for flight (type ZZZZ)" });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(screen.getByRole("alert").textContent).toContain("no aircraft dynamics");
    expect(screen.getByText("Failed")).toBeTruthy();
  });

  it("lists the result of a done job — plain outcome names, the raw string in the title — and selects a flight on a click", async () => {
    const trackedViewer = {
      dataSources: { length: 1, get: () => ({ entities: { getById: (id: string) => (id === ARRIVALS[0].flightKey ? { id } : undefined) } }) },
      trackedEntity: undefined as unknown,
    };
    app.viewer = trackedViewer;
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    api.fetchTrafficJob.mockResolvedValue({ state: "done", progress: { done: 1, total: 1, current: ARRIVALS[0].flightKey },
      error: null, summary: { windows: 1 } });
    fetchIndex.mockResolvedValue(indexOf([
      group(ARRIVALS[0], { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();

    expect(screen.getByText("Ready")).toBeTruthy();
    expect(app.setTrafficScene).toHaveBeenLastCalledWith(expect.objectContaining({ airportCode: "KRDU", jobId: JOB }));
    const result = within(screen.getByLabelText("Result"));
    expect(result.getByRole("status").textContent).toBe(
      `AAL1, ${DAY} — 1 aircraft controlled: 1 separated after re-solve`);              // names the flight and the day it was run for
    const row = result.getByText("separated after re-solve", { selector: "td" }).closest("tr")!;
    expect(row.getAttribute("title")).toBe(`${ARRIVALS[0].flightKey} — separated after re-solve (separated)`);
    expect(result.queryByText("Delay")).toBeNull();                                  // an M1 job has no slot

    fireEvent.click(row);
    expect(app.setSelectedFlightId).toHaveBeenCalledWith(ARRIVALS[0].flightKey);
    expect(trackedViewer.trackedEntity).toEqual({ id: ARRIVALS[0].flightKey });
    // the legend names the controlled aircraft and the recorded traffic
    expect(result.getByLabelText("Controlled aircraft — its record")).toBeTruthy();
    expect(result.getByLabelText("Controlled aircraft — optimized path")).toBeTruthy();
    expect(result.getByText("Recorded traffic — not controlled")).toBeTruthy();
  });

  it("never touches the top bar's runway selector, whatever the job does", async () => {
    render(<TrafficJobPanel kind="m1" />);
    await pickDay();
    fireEvent.click(screen.getByText("AAL1"));
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    api.fetchTrafficJob.mockResolvedValue({ state: "done", progress: { done: 1, total: 1, current: null },
      error: null, summary: {} });
    fetchIndex.mockResolvedValue(indexOf([group(ARRIVALS[0], { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    fireEvent.click(within(screen.getByLabelText("Result")).getByText("AAL1"));
    expect(app.setSelectedRunway).not.toHaveBeenCalled();
    expect(app.selectedRunway).toBe("05R");
  });

  it("shows the solver settings of the batch, read-only", async () => {
    render(<TrafficJobPanel kind="m1" />);
    const settings = screen.getByText("Solver settings (the batch's)").closest("details")!;
    expect(settings.textContent).toContain("max duration 2000 s");
    expect(settings.textContent).toContain("IPOPT cap 3000");
    expect(settings.querySelector("input, select")).toBeNull();
  });
});

describe("TrafficJobPanel, all controlled (M2)", () => {
  const hourSelect = () => screen.getByLabelText("Start hour (UTC)") as HTMLSelectElement;
  const minuteSelect = () => screen.getByLabelText("Minute") as HTMLSelectElement;
  const count = () => screen.getByLabelText("Arrivals in the block").textContent;

  it("starts the block at the hour of the day's first landing, rounded down, with 24-hour selects", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    expect(hourSelect().value).toBe("17");                                    // the first landing is 17:00:10
    expect(minuteSelect().value).toBe("00");
    expect(Array.from(hourSelect().options).map((o) => o.value)).toEqual(
      Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0")));      // 00 .. 23, not a locale time field
    expect(Array.from(minuteSelect().options).map((o) => o.value)).toEqual(["00", "15", "30", "45"]);
    expect(screen.queryByLabelText("Block start (UTC)")).toBeNull();
    expect((screen.getByLabelText("Length") as HTMLSelectElement).value).toBe("1800");
    expect(Array.from((screen.getByLabelText("Length") as HTMLSelectElement).options).map((o) => o.textContent))
      .toEqual(["15 min", "30 min", "60 min"]);
  });

  it("keeps a start the user set when the arrivals load again, and defaults again for a new day", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.change(hourSelect(), { target: { value: "06" } });
    fireEvent.change(minuteSelect(), { target: { value: "45" } });
    api.fetchTrafficArrivals.mockResolvedValue({ airport: "KRDU", date: "2026-05-22", arrivals: [
      arrival("BAW9", "05L", "09:30:00", "A320")] });
    fireEvent.change(screen.getByLabelText("UTC day"), { target: { value: "2026-05-22" } });
    await settle();
    expect(hourSelect().value).toBe("09");                                    // a new day, a new default
    expect(minuteSelect().value).toBe("00");
  });

  it("counts the arrivals that land in the block", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    expect(count()).toBe("2 arrivals land in this block");                    // 17:00:10 and 17:10:00 in 17:00–17:30
    fireEvent.change(screen.getByLabelText("Length"), { target: { value: "900" } });
    expect(count()).toBe("2 arrivals land in this block");                    // 17:00–17:15
    fireEvent.change(minuteSelect(), { target: { value: "15" } });
    expect(count()).toBe("0 arrivals land in this block");
    expect((screen.getByRole("button", { name: "Start" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Length"), { target: { value: "3600" } });
    expect(count()).toBe("1 arrival lands in this block");                    // 17:15–18:15 holds the 17:40
  });

  it("starts the block job with the UTC start and the length", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.change(minuteSelect(), { target: { value: "15" } });
    fireEvent.change(screen.getByLabelText("Length"), { target: { value: "900" } });
    fireEvent.change(hourSelect(), { target: { value: "17" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    // 17:15–17:30 holds no landing: nothing starts
    expect(api.startTrafficJob).not.toHaveBeenCalled();

    fireEvent.change(minuteSelect(), { target: { value: "00" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({
      mode: "m2", airport: "KRDU", blockStartUtc: `${DAY}T17:00:00Z`, blockS: 900 });
  });

  it("warns when the block runs past midnight", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.change(hourSelect(), { target: { value: "23" } });
    fireEvent.change(minuteSelect(), { target: { value: "45" } });
    expect(count()).toContain("runs past midnight");
  });

  const doneScene = (extra: Record<string, unknown> = {}) => {
    api.fetchTrafficJob.mockResolvedValue({
      state: "done", progress: { done: 2, total: 2, current: ARRIVALS[1].flightKey }, error: null,
      summary: { aircraft: 2, skipped_no_dynamics: 1, flown_aircraft_with_a_loss_left_after_the_block: {
        visual: { answered: 1, not_answered: 0 }, ifr: { answered: 2, not_answered: 2 } }, ...extra } });
    fetchIndex.mockResolvedValue(indexOf([
      group(ARRIVALS[0], { scene: { startOffsetS: 0, outcome: "separated_at_baseline", delayS: 0 } }),
      group(ARRIVALS[1], { runway: "05R", scene: { startOffsetS: 600, outcome: "solve_failed", delayS: 80.4 } }),
    ], { startUtc: "2026-05-21T17:00:10Z", background: { recorded: [], startOffsetsS: [] } }));
  };

  it("lists every controlled aircraft with its outcome and delay (s), and a summary line that names its block", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    doneScene();
    await settle(2000);

    const result = within(screen.getByLabelText("Result"));
    expect(result.getByRole("status").textContent).toBe(
      `Block ${DAY} 17:00–17:30 UTC — 2 aircraft controlled: 1 separated at the first solve (no re-solve), ` +
      "1 loss left (re-solve failed; last good solve shown)" +
      " · 1 more landed in the block and stayed records (no aircraft dynamics model)" +
      " · delay median 40 s, largest 80 s, 1 over 60 s" +
      " · after the block's final check: 1 aircraft with a loss of visual separation they answer for, 0 with one they do not");
    const rows = result.getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("cell").map((c) => c.textContent));
    expect(rows).toEqual([
      ["AAL1", "05L", "separated at the first solve (no re-solve)", "0"],
      ["BAW2", "05R", "loss left (re-solve failed; last good solve shown)", "80"],
    ]);
    expect(result.getByText("Delay (s)")).toBeTruthy();                       // the unit in the header, not "Delay"
    expect(result.queryByText("Delay")).toBeNull();
    expect(result.getByText("Recorded traffic — not controlled, outside the scheduled set")).toBeTruthy();
  });

  it("wraps the outcome text in the cell instead of scrolling the table sideways", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    doneScene();
    await settle(2000);
    const outcomeCell = within(screen.getByLabelText("Result")).getByText("separated at the first solve (no re-solve)", { selector: "td" });
    expect(outcomeCell.className).toBe("traffic-job-outcome");
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    expect(css).toMatch(/\.traffic-job-panel td\.traffic-job-outcome\s*\{[^}]*white-space:\s*normal/);
    expect(css).toMatch(/\.traffic-job-table-scroll\s*\{[^}]*overflow-x:\s*hidden/);
    expect(css).toMatch(/\.traffic-job-panel table\s*\{[^}]*table-layout:\s*fixed/);
  });

  it("gives the date field a row of its own so the native control is not clipped", () => {
    render(<TrafficJobPanel kind="m2" />);
    const date = screen.getByLabelText("UTC day");
    expect(date.closest(".traffic-job-date")).not.toBeNull();
    expect(date.closest(".traffic-job-block")).toBeNull();                    // not a column of the hour/minute/length row
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    expect(css).toMatch(/\.traffic-job-date input\s*\{[^}]*width:\s*100%/);
  });

  it("says how many of the block's arrivals were controlled and how many stayed records, once its job is done", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    expect(screen.getByLabelText("Arrivals in the block").textContent).toBe("2 arrivals land in this block");
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    doneScene({ aircraft: 4, skipped_no_dynamics: 1 });
    await settle(2000);
    expect(screen.getByLabelText("Arrivals in the block").textContent).toBe(
      "5 arrivals landed in this block: 4 controlled, 1 stayed records (no aircraft dynamics model)");
  });

  it("keeps the result's inputs when the panel's inputs change afterwards: a changed input never looks like a new result", async () => {
    render(<TrafficJobPanel kind="m2" />);
    await pickDay();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    doneScene({ aircraft: 4, skipped_no_dynamics: 1 });
    await settle(2000);
    const heading = () => within(screen.getByLabelText("Result")).getByRole("status").textContent!;
    expect(heading()).toContain(`Block ${DAY} 17:00–17:30 UTC`);

    fireEvent.change(hourSelect(), { target: { value: "18" } });
    fireEvent.change(screen.getByLabelText("Length"), { target: { value: "900" } });
    expect(heading()).toContain(`Block ${DAY} 17:00–17:30 UTC`);               // still the block the job was run for
    expect(screen.getByLabelText("Arrivals in the block").textContent).toBe("0 arrivals land in this block");
  });
});
