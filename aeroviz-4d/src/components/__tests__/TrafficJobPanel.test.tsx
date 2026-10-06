import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import catalogMirror from "../../data/__tests__/fixtures/trafficScenarioCatalog.json";
import { reportOf } from "../../data/__tests__/evaluationReport.fixture";
import type {
  TrafficAircraftChange, TrafficJobStatus, TrafficJobTiming, TrafficScenarioArrival, TrafficScenarioCatalog, TrafficStayedRecord,
} from "../../data/trafficJobs";

const { api, app, fetchIndex, calls, pilot } = vi.hoisted(() => ({
  // the aircraft catalog the type codes' plain names come from
  pilot: { fetchPilotAircraftConfigs: vi.fn() },
  // how often the list sorted its rows and built a row's title (the work a re-render must not repeat)
  calls: { listedRows: 0, arrivalDetail: 0, blockDetail: 0 },
  api: {
    fetchTrafficScenarios: vi.fn(),
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
vi.mock("../../pilot/pilotClient", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../pilot/pilotClient")>()),
  ...pilot,
}));
vi.mock("../../data/trafficJobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../data/trafficJobs")>()),
  ...api,
}));
vi.mock("../../utils/trafficScenarios", async (importOriginal) => {
  const original = await importOriginal<typeof import("../../utils/trafficScenarios")>();
  return {
    ...original,
    listedRows: ((...args: Parameters<typeof original.listedRows>) => {
      calls.listedRows += 1;
      return original.listedRows(...args);
    }) as typeof original.listedRows,
    arrivalDetail: (...args: Parameters<typeof original.arrivalDetail>) => {
      calls.arrivalDetail += 1;
      return original.arrivalDetail(...args);
    },
    blockDetail: (...args: Parameters<typeof original.blockDetail>) => {
      calls.blockDetail += 1;
      return original.blockDetail(...args);
    },
  };
});
vi.mock("../../utils/fetchJson", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../utils/fetchJson")>()),
  fetchJson: (url: string) => fetchIndex(url),
}));

import TrafficJobPanel from "../TrafficJobPanel";
import { CONTROLLABLE_TITLE, LIGHT_AIRCRAFT_TITLE, lossSecondsTitle } from "../../utils/trafficScenarios";

const JOB = "20261006T120000123456Z-0123abcd";
// the catalog is a MIRROR of what `traffic_scenarios.py` writes (see `data/__tests__/trafficJobs.test.ts`)
const CATALOG = catalogMirror as TrafficScenarioCatalog;
const [LIGHT, SWA, NAMELESS, DAL] = CATALOG.m1;             // by loss instants: 40 (a C172, CWT I), 28, 6, 6 (then 3); NAMELESS has no callsign, type or category
const MISSING_CATALOG = "no scenario catalog for KRDU at /x/catalog.json; " +
  "make it with `python 4dTrajectory/optimization/traffic_scenarios.py --airport KRDU`";
const running = (done: number, total: number | null, current: string | null = null,
  phase = "optimizing 1 of 1"): TrafficJobStatus => ({ state: "running", progress: { done, total, current, phase }, error: null });
const change = (over: Partial<TrafficAircraftChange> = {}): TrafficAircraftChange => ({
  type: "B38M", firstSolveLosses: 9, finalLosses: 0, landingVsRecordS: -400, delayS: null, blockCheckLosses: null,
  optimizeS: 12.34, optimizeCpuS: 11.8, solves: 3, failedSolves: 0, ...over });
const NOT_FLOWN = { firstSolveLosses: null, finalLosses: null, landingVsRecordS: null };
const TIMING = { totalS: 20, phases: { "reading traffic": 5, optimizing: 15 } };
/** A finished job's status: what changed for each aircraft the index lists, the readout's `summary` and how long it took. */
const doneStatus = (perAircraft: Record<string, TrafficAircraftChange>, summary: Record<string, unknown> = {},
  timing: TrafficJobTiming = TIMING, stayedRecords: Record<string, TrafficStayedRecord> = {}): TrafficJobStatus => ({
  state: "done", progress: { done: 1, total: 1, current: null, phase: "building the scene" }, error: null, summary, perAircraft, timing,
  stayedRecords });

const indexOf = (groups: unknown[], scene?: unknown) => ({
  schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
  referenceSource: "canonicalObserved", evaluationReport: "r.json", groups, ...(scene ? { scene } : {}),
});
const group = (arrivalRow: TrafficScenarioArrival, extra: Record<string, unknown>) => ({
  group: arrivalRow.flightKey, flightId: arrivalRow.callsign ?? "N123AB", runway: arrivalRow.runway, airport: "KRDU",
  status: "solved", finalTimeS: 300, initialState: null, entities: [`ref-${arrivalRow.flightKey}`, `sim-${arrivalRow.flightKey}`],
  czml: "a.czml", ...extra });

beforeEach(() => {
  vi.useFakeTimers();
  calls.listedRows = calls.arrivalDetail = calls.blockDetail = 0;
  window.localStorage.clear();
  for (const mock of [...Object.values(api), ...Object.values(pilot), app.setSelectedFlightId, app.setSelectedRunway, app.setTrafficScene, fetchIndex]) mock.mockReset();
  // the backend's own catalog (aircraft/aircraft_sets.py): three types have a plain name
  pilot.fetchPilotAircraftConfigs.mockResolvedValue([
    { code: "A320", name: "Airbus A320-200" }, { code: "B77W", name: "Boeing 777-300ER" }, { code: "C172", name: "Cessna 172" }]);
  api.fetchTrafficScenarios.mockResolvedValue(CATALOG);
  api.startTrafficJob.mockResolvedValue(JOB);
  api.fetchTrafficJob.mockResolvedValue(running(0, 1));
  api.cancelTrafficJob.mockResolvedValue({ ...running(0, 1), state: "cancelled" });
  app.viewer = null;
  app.selectedFlightId = null;
});
afterEach(() => vi.useRealTimers());

const settle = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

/** The panel with its scenario list loaded. */
async function open(kind: "m1" | "m2") {
  const view = render(<TrafficJobPanel kind={kind} />);
  await settle();
  return view;
}

/** The two lines of each card of a listbox. */
const cardLines = (label: string) => within(screen.getByLabelText(label)).getAllByRole("option")
  .map((li) => Array.from(li.children).map((line) => line.textContent));
const m1Cards = () => cardLines("Arrivals with a loss");
const blockCards = () => cardLines("Blocks");
const callsigns = () => m1Cards().map((lines) => lines[0]!.split(" · ")[0]);
const spans = () => blockCards().map((lines) => lines[0]!.slice(5, 16));          // `05-01 12:30`
const listed = () => screen.getByLabelText("Rows listed").textContent;
const startButton = () => screen.getByRole("button", { name: "Start" }) as HTMLButtonElement;
const sortBy = (value: string) => fireEvent.change(screen.getByLabelText("Sort by"), { target: { value } });
/** A card of the M1 list, by its callsign; of the M2 list, by the span of its first line. */
const arrivalCard = (callsign: string) => within(screen.getByLabelText("Arrivals with a loss")).getByText(new RegExp(`^${callsign} · `));
const blockCard = (span: string) => within(screen.getByLabelText("Blocks")).getByText(span);
const selection = () => screen.getByLabelText("Selection").textContent;

describe("TrafficJobPanel, one controlled (M1)", () => {
  it("has no date field: it lists the airport's scenario catalog and says what it holds", async () => {
    render(<TrafficJobPanel kind="m1" />);
    expect(screen.getByText("Loading the scenario list…")).toBeTruthy();
    await settle();
    expect(api.fetchTrafficScenarios).toHaveBeenCalledWith("KRDU");
    expect(screen.queryByLabelText("UTC day")).toBeNull();
    expect(document.querySelector("input[type=date]")).toBeNull();
    expect(screen.queryByLabelText("Start hour (UTC)")).toBeNull();
    expect(screen.queryByLabelText("Minute")).toBeNull();
    expect(screen.queryByLabelText("Arrivals of the day")).toBeNull();

    expect(screen.getByText(
      "5 of 6 arrivals have a loss of separation they answer for in their record (the records were checked every 1 s, 2026-10-06T09:11:42Z)"))
      .toBeTruthy();
    expect(screen.getByText(/recorded aircraft as flown.*dense traffic.*not a promise that the optimized flight has one/)).toBeTruthy();
    expect(screen.queryByText(/Partial census/)).toBeNull();                  // a whole roster says nothing of the sort
    expect(screen.queryByText(/blocks of/)).toBeNull();                       // the blocks' header is M2's
    expect(listed()).toBe("4 of 5 arrivals with a loss · 1 light aircraft hidden");
  });

  it("lists the arrivals as two-line cards — who, then when and how many losses — most losses first, in no table", async () => {
    await open("m1");
    expect(document.querySelector("table")).toBeNull();                       // no columns to fit a 242 px dock
    expect(m1Cards()).toEqual([
      ["SWA3131 · B38M · 05L", "2026-05-01 00:06 UTC · 28 seconds in loss"],
      ["N123AB · — · 05R", "2026-05-01 00:21 UTC · 6 seconds in loss"],
      ["DAL88 · A321 · 32", "2026-05-01 14:49 UTC · 6 seconds in loss"],
      ["AAL2634 · A321 · 05L", "2026-05-01 14:55 UTC · 3 seconds in loss"],
    ]);
  });

  it("sorts by a select above the list: losses or time, each way — the sort is the select's value", async () => {
    await open("m1");
    const select = screen.getByLabelText("Sort by") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => [o.value, o.textContent])).toEqual([
      ["loss-desc", "Most losses"], ["loss-asc", "Fewest losses"], ["time-asc", "Earliest"], ["time-desc", "Latest"]]);
    expect(select.value).toBe("loss-desc");
    expect(callsigns()).toEqual(["SWA3131", "N123AB", "DAL88", "AAL2634"]);
    sortBy("time-desc");
    expect(callsigns()).toEqual(["AAL2634", "DAL88", "N123AB", "SWA3131"]);
    sortBy("time-asc");
    expect(callsigns()).toEqual(["SWA3131", "N123AB", "DAL88", "AAL2634"]);
    sortBy("loss-asc");
    expect(callsigns()).toEqual(["AAL2634", "N123AB", "DAL88", "SWA3131"]);                     // equal ones in the catalog's order
    expect(select.value).toBe("loss-asc");
  });

  it("hides the light aircraft (CWT I) by default, says how many, and shows them when asked", async () => {
    await open("m1");
    const hide = screen.getByLabelText("Hide light aircraft") as HTMLInputElement;
    expect(hide.checked).toBe(true);
    expect(screen.queryByText(/^N2412P · /)).toBeNull();                      // a C172 tops the ranking with 40 instants
    expect(listed()).toBe("4 of 5 arrivals with a loss · 1 light aircraft hidden");

    fireEvent.click(hide);
    expect(hide.checked).toBe(false);
    expect(m1Cards()[0]).toEqual(["N2412P · C172 · 05L", "2026-05-01 12:30 UTC · 40 seconds in loss"]);   // first again, as the census made it
    expect(m1Cards()).toHaveLength(5);
    expect(listed()).toBe("5 of 5 arrivals with a loss");
    fireEvent.click(hide);
    expect(m1Cards()).toHaveLength(4);
  });

  it("counts the light aircraft it hides within the runway asked for", async () => {
    await open("m1");
    const runway = screen.getByLabelText("Show runway");
    fireEvent.change(runway, { target: { value: "05L" } });                                 // the C172 is on 05L: hidden from this list
    expect(listed()).toBe("2 of 5 arrivals with a loss · 1 light aircraft hidden");
    fireEvent.change(runway, { target: { value: "05R" } });                                 // none on 05R: nothing hidden from this list
    expect(listed()).toBe("1 of 5 arrivals with a loss");
    fireEvent.change(runway, { target: { value: "" } });
    expect(listed()).toBe("4 of 5 arrivals with a loss · 1 light aircraft hidden");
  });

  it("clears a pick that hiding the light aircraft takes off the list, and keeps any other", async () => {
    await open("m1");
    const hide = screen.getByLabelText("Hide light aircraft");
    fireEvent.click(hide);                                                    // show them
    fireEvent.click(arrivalCard("N2412P"));
    expect(startButton().disabled).toBe(false);
    fireEvent.click(hide);                                                    // hide them again: the pick goes with them
    expect(selection()).toBe("Pick the arrival to optimize in its recorded traffic.");
    expect(startButton().disabled).toBe(true);

    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(hide);
    fireEvent.click(hide);                                                    // a toggle that hides nothing picked keeps the pick
    expect(selection()).toMatch(/^Selected: SWA3131 — /);
  });

  it("says when hiding the light aircraft leaves nothing", async () => {
    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, m1: [LIGHT], counts: { ...CATALOG.counts, withLoss: 1 } });
    await open("m1");
    expect(screen.getByText("No row is left once the light aircraft are hidden.")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Hide light aircraft"));
    expect(m1Cards()).toHaveLength(1);
  });

  it("keeps the rest of a card — tightest, kinds, recorded aircraft — in its title and, once picked, in the Selected line", async () => {
    await open("m1");
    const card = arrivalCard("N123AB").closest("li")!;
    expect(card.getAttribute("title")).toBe(`${NAMELESS.flightKey} — 6 seconds in loss, tightest 86 % of the minimum (` +
      "too close in trail on one final; under the radar minimum and not vertically separated) · type unknown · runway 05R · " +
      "landing 2026-05-01 00:21:30 UTC · 9 recorded aircraft in its window. " + lossSecondsTitle(1));
    fireEvent.click(card);
    expect(selection()).toBe(
      "Selected: N123AB — 6 seconds in loss, tightest 86 % of the minimum (too close in trail on one final; under the radar " +
      "minimum and not vertically separated) · type unknown · runway 05R · landing 2026-05-01 00:21:30 UTC · " +
      "9 recorded aircraft in its window");
  });

  it("is a listbox of cards that wrap at spaces — never mid-word, never cut — in a vertical scroll area", () => {
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    const rule = (selector: string) => css.match(new RegExp(`${selector.replace(/[.]/g, "\\.")}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
    expect(rule(".traffic-job-card")).toMatch(/white-space:\s*normal/);
    expect(rule(".traffic-job-card")).toMatch(/overflow-wrap:\s*normal/);
    expect(rule(".traffic-job-card")).toMatch(/word-break:\s*normal/);
    expect(rule(".traffic-job-card")).not.toMatch(/(^|[\s;])overflow(-x|-y)?\s*:|ellipsis|nowrap/);       // nothing is clipped
    expect(rule(".traffic-job-catalog-scroll")).toMatch(/max-height:\s*34vh[^}]*overflow-x:\s*hidden[^}]*overflow-y:\s*auto/s);
    expect(css).toMatch(/--overlay-left-width:\s*clamp\(220px,\s*17vw,\s*280px\)/);
  });

  it("never clips the text of a select: two to a row, full width, labels short enough for half a 242 px dock", async () => {
    await open("m1");
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    const rule = (selector: string) => css.match(new RegExp(`${selector.replace(/[.]/g, "\\.")}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
    expect(rule(".traffic-job-catalog .pilot-optimization-row")).toMatch(/grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
    expect(rule(".traffic-job-catalog .pilot-optimization-row select")).toMatch(/width:\s*100%/);
    for (const name of ["Sort by", "Show runway"]) {
      const select = screen.getByLabelText(name) as HTMLSelectElement;
      expect(Math.max(...Array.from(select.options).map((o) => o.textContent!.length)), name).toBeLessThanOrEqual(13);
    }
  });

  it("says the jargon in plain words, each with a title: light aircraft, seconds in loss, how often the records were checked", async () => {
    await open("m1");
    const hide = screen.getByLabelText("Hide light aircraft");
    expect(hide.closest("label")!.getAttribute("title")).toBe(LIGHT_AIRCRAFT_TITLE);
    const header = screen.getByText(/arrivals have a loss of separation they answer for in their record/);
    expect(header.textContent).toContain("(the records were checked every 1 s,");
    expect(document.body.textContent).not.toContain("census checks");
    // spelled out once, in the header: every other place says "loss"
    expect(document.body.textContent!.match(/loss of separation/g)).toHaveLength(1);
    expect(header.getAttribute("title")).toBe(lossSecondsTitle(1));
    expect(arrivalCard("SWA3131").closest("li")!.getAttribute("title")).toContain(lossSecondsTitle(1));
    expect(screen.getByLabelText("Rows listed").getAttribute("title")).toBe(LIGHT_AIRCRAFT_TITLE);
    expect(document.body.textContent).not.toContain("loss instants");                  // the old word is gone from the page
  });

  it("gives an ICAO type code the plain name of the type as its title where the app has one, and leaves the others as they are", async () => {
    // the aircraft catalog names A320, B77W and C172; the fixture's arrivals are B38M, A321 and one untyped
    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, m1: CATALOG.m1.map((row) => (row.flightKey === SWA.flightKey ? { ...row, type: "A320" } : row)) });
    await open("m1");
    const swaCard = arrivalCard("SWA3131").closest("li")!;
    expect(within(swaCard).getByText("A320").getAttribute("title")).toBe("Airbus A320-200");
    expect(swaCard.firstElementChild!.textContent).toBe("SWA3131 · A320 · 05L");              // the line reads as it did
    expect(swaCard.getAttribute("title")).toContain(" · type A320 (Airbus A320-200) · runway 05L · ");
    const dalCard = arrivalCard("DAL88").closest("li")!;                                      // A321: the app has no name for it
    expect(dalCard.querySelector("[title='Airbus A321']")).toBeNull();
    expect(dalCard.querySelector(".traffic-job-type")).toBeNull();
    expect(dalCard.firstElementChild!.textContent).toBe("DAL88 · A321 · 32");
    expect(arrivalCard("N123AB").closest("li")!.firstElementChild!.textContent).toBe("N123AB · — · 05R");   // untyped

    fireEvent.click(swaCard);
    expect(selection()).toContain("type A320 (Airbus A320-200)");
    fireEvent.click(dalCard);
    expect(selection()).toContain("type A321 · runway");                                     // no name: the code alone
  });

  it("shows the type codes as they are when the aircraft catalog cannot be read, and says so on the console", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    pilot.fetchPilotAircraftConfigs.mockRejectedValue(new Error("backend down"));
    await open("m1");
    expect(m1Cards()[0]).toEqual(["SWA3131 · B38M · 05L", "2026-05-01 00:06 UTC · 28 seconds in loss"]);
    expect(document.querySelector(".traffic-job-type")).toBeNull();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("aircraft catalog could not be read"), expect.any(Error));
    warn.mockRestore();
  });

  it("labels its runway select 'Show runway': the top bar has a runway control of its own that this one is not", async () => {
    await open("m1");
    expect(screen.getByLabelText("Show runway")).toBeTruthy();
    expect(screen.queryByLabelText("Runway")).toBeNull();
    expect(screen.getByLabelText("Show runway").closest("label")!.textContent).toBe("Show runwayAll runways05L05R32");
  });

  it("cards can be picked from the keyboard as well", async () => {
    await open("m1");
    const card = arrivalCard("DAL88").closest("li")!;
    expect(card.getAttribute("role")).toBe("option");
    fireEvent.keyDown(card, { key: "Enter" });
    expect(selection()).toMatch(/^Selected: DAL88 — /);
    fireEvent.keyDown(arrivalCard("SWA3131").closest("li")!, { key: " " });
    expect(selection()).toMatch(/^Selected: SWA3131 — /);
    expect(screen.getByLabelText("Arrivals with a loss").querySelector("[aria-selected=true]")!.textContent).toContain("SWA3131");
  });

  it("starts the job on the arrival picked, and not before one is", async () => {
    await open("m1");
    expect(startButton().disabled).toBe(true);                                 // none picked yet
    expect(screen.getByText("Pick the arrival to optimize in its recorded traffic.")).toBeTruthy();
    fireEvent.click(arrivalCard("DAL88"));
    expect(selection()).toMatch(/^Selected: DAL88 — /);
    expect(startButton().disabled).toBe(false);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: DAL.flightKey });
  });

  it("lists a flight that has no callsign by the first field of its key, and starts it", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("N123AB"));
    expect(selection()).toMatch(/^Selected: N123AB — /);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: NAMELESS.flightKey });
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [NAMELESS.flightKey]: change({ type: null }) }));
    fetchIndex.mockResolvedValue(indexOf([group(NAMELESS, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    expect(within(screen.getByLabelText("Result")).getByRole("status").textContent).toContain("N123AB, 2026-05-01 — ");
  });

  it("filters by runway and says how many rows match", async () => {
    await open("m1");
    const runway = screen.getByLabelText("Show runway") as HTMLSelectElement;
    expect(Array.from(runway.options).map((o) => o.value)).toEqual(["", "05L", "05R", "32"]);
    expect(runway.options[0].textContent).toBe("All runways");

    fireEvent.change(runway, { target: { value: "05L" } });
    expect(callsigns()).toEqual(["SWA3131", "AAL2634"]);
    fireEvent.change(runway, { target: { value: "32" } });
    expect(callsigns()).toEqual(["DAL88"]);
    expect(listed()).toBe("1 of 5 arrivals with a loss");
    fireEvent.change(runway, { target: { value: "" } });
    expect(listed()).toBe("4 of 5 arrivals with a loss · 1 light aircraft hidden");
  });

  it("clears a pick the runway filter hides — Start waits for a visible row — and keeps one it shows", async () => {
    await open("m1");
    const runway = screen.getByLabelText("Show runway") as HTMLSelectElement;
    fireEvent.click(arrivalCard("N123AB"));                                       // runway 05R
    expect(startButton().disabled).toBe(false);
    fireEvent.change(runway, { target: { value: "05L" } });                       // hides it
    expect(selection()).toBe("Pick the arrival to optimize in its recorded traffic.");
    expect(startButton().disabled).toBe(true);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).not.toHaveBeenCalled();
    fireEvent.change(runway, { target: { value: "" } });                           // showing it again does not bring the pick back
    expect(selection()).toBe("Pick the arrival to optimize in its recorded traffic.");
    expect(startButton().disabled).toBe(true);

    fireEvent.click(arrivalCard("SWA3131"));                                       // runway 05L
    fireEvent.change(runway, { target: { value: "05L" } });                       // still visible: kept
    expect(selection()).toMatch(/^Selected: SWA3131 — /);
    expect(startButton().disabled).toBe(false);
    fireEvent.change(runway, { target: { value: "05R" } });
    expect(startButton().disabled).toBe(true);
  });

  const manyArrivals = (count: number): TrafficScenarioArrival[] => Array.from({ length: count }, (_, n) => ({
    ...SWA, flightKey: `X${n}_05L_a_${n}`, callsign: `X${n}`, lossInstants: count - n, runway: n % 2 ? "05L" : "05R",
    landingUtc: new Date(Date.parse("2026-05-01T00:00:00Z") + n * 600_000).toISOString().replace(".000Z", "Z") }));
  const withArrivals = (count: number) => api.fetchTrafficScenarios.mockResolvedValue({
    ...CATALOG, m1: manyArrivals(count), counts: { ...CATALOG.counts, withLoss: count } });
  const shown = () => screen.getByLabelText("Rows shown").textContent;
  const showMore = () => fireEvent.click(screen.getByRole("button", { name: /^Show \d+ more$/ }));

  it("shows 100 rows at a time — 'Showing 1–100 of N · Show 100 more' — and no row is dropped", async () => {
    withArrivals(250);
    await open("m1");
    expect(m1Cards()).toHaveLength(100);
    expect(shown()).toBe("Showing 1–100 of 250 · Show 100 more");
    expect(listed()).toBe("250 of 250 arrivals with a loss");
    showMore();
    expect(m1Cards()).toHaveLength(200);
    expect(shown()).toBe("Showing 1–200 of 250 · Show 50 more");
    showMore();
    expect(m1Cards()).toHaveLength(250);                                            // every row is there
    expect(shown()).toBe("Showing 1–250 of 250");
    expect(screen.queryByRole("button", { name: /Show \d+ more/ })).toBeNull();
    expect(callsigns()[249]).toBe("X249");
  });

  it("starts again at one page when the view changes — a filter, a sort, the light aircraft", async () => {
    withArrivals(250);
    await open("m1");
    showMore();
    expect(m1Cards()).toHaveLength(200);
    fireEvent.change(screen.getByLabelText("Show runway"), { target: { value: "05L" } });          // 125 rows match
    expect(m1Cards()).toHaveLength(100);
    expect(shown()).toBe("Showing 1–100 of 125 · Show 25 more");
    showMore();
    expect(shown()).toBe("Showing 1–125 of 125");
    sortBy("time-desc");
    expect(m1Cards()).toHaveLength(100);
    expect(shown()).toBe("Showing 1–100 of 125 · Show 25 more");
    showMore();
    fireEvent.click(screen.getByLabelText("Hide light aircraft"));
    expect(m1Cards()).toHaveLength(100);
  });

  it("keeps a pick across the pages: a row picked on the second page is still the pick on the first", async () => {
    withArrivals(250);
    await open("m1");
    showMore();
    fireEvent.click(arrivalCard("X150"));
    expect(selection()).toMatch(/^Selected: X150 — /);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: "X150_05L_a_150" });
  });

  it("keeps a pick the view no longer shows — a sort moved it off the page — picked and named in the Selected line", async () => {
    withArrivals(250);
    await open("m1");
    fireEvent.click(arrivalCard("X5"));                                              // on the first page
    sortBy("time-desc");                                                              // X5 is now among the last: off the drawn page
    expect(screen.queryByText(/^X5 · /)).toBeNull();
    expect(selection()).toMatch(/^Selected: X5 — /);                                   // still the pick, named
    expect(startButton().disabled).toBe(false);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({ mode: "m1", airport: "KRDU", flightKey: "X5_05L_a_5" });
  });

  it("scrolls inside a fixed height, and the line above says how many match", async () => {
    withArrivals(300);
    await open("m1");
    expect(screen.getByLabelText("Arrivals with a loss").closest(".traffic-job-catalog-scroll")).not.toBeNull();
    fireEvent.change(screen.getByLabelText("Show runway"), { target: { value: "05L" } });
    expect(listed()).toBe("150 of 300 arrivals with a loss");
  });

  it("says what a census of part of the roster is, and what an empty list is", async () => {
    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, config: { ...CATALOG.config, limit: 300 } });
    const view = await open("m1");
    expect(screen.getByText(/Partial census: only the first 300 arrivals by landing time were judged/)).toBeTruthy();
    view.unmount();

    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, m1: [], counts: { ...CATALOG.counts, withLoss: 0 } });
    await open("m1");
    expect(screen.getByText("No arrival of KRDU has a loss they answer for.")).toBeTruthy();
    expect(listed()).toBe("0 of 0 arrivals with a loss");
  });

  it("shows the backend's message, as it is, when the catalog is missing, and starts nothing", async () => {
    api.fetchTrafficScenarios.mockRejectedValue(new Error(MISSING_CATALOG));
    await open("m1");
    expect(screen.getByRole("alert").textContent).toBe(MISSING_CATALOG);
    expect(screen.queryByLabelText("Scenario list")).toBeNull();
    expect(startButton().disabled).toBe(true);
  });

  it("shows the progress while the job runs and cancels it on request", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    api.fetchTrafficJob.mockResolvedValue(running(2, 5, "BAW2_05R_b_2"));
    fireEvent.click(startButton());
    await settle();

    expect(screen.getByText("Computing")).toBeTruthy();
    expect(screen.getByText("2 of 5 aircraft done · last: BAW2")).toBeTruthy();
    expect(screen.getByLabelText("Phase").textContent).toBe("optimizing 1 of 1 · 00:00");
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

  it("does not fetch the catalog again, nor sort or rebuild its rows, at every poll of a running job", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    expect(calls.listedRows).toBeGreaterThan(0);                                   // the spy sees the list's sort
    api.fetchTrafficJob.mockResolvedValue(running(1, 5, "A_05L_a_1"));
    await settle(2000);                                                            // the first poll
    const sorted = calls.listedRows, titled = calls.arrivalDetail;
    await settle(2000);
    await settle(2000);                                                            // two more polls: the panel re-renders each time
    expect(screen.getByText("1 of 5 aircraft done · last: A")).toBeTruthy();
    expect(calls.listedRows).toBe(sorted);
    expect(calls.arrivalDetail).toBe(titled);
    expect(api.fetchTrafficScenarios).toHaveBeenCalledTimes(1);
    expect(m1Cards()).toHaveLength(4);
  });

  it("re-renders only the two cards whose selection changed when a row is picked, in a long list", async () => {
    withArrivals(300);
    await open("m1");
    expect(calls.arrivalDetail).toBe(100);                                         // one title per card shown, built once
    fireEvent.click(arrivalCard("X7"));
    expect(calls.arrivalDetail).toBe(101);                                         // the card picked
    fireEvent.click(arrivalCard("X50"));
    expect(calls.arrivalDetail).toBe(103);                                         // the card left and the card picked
    expect(arrivalCard("X50").closest("li")!.className).toContain("selected");
    expect(arrivalCard("X7").closest("li")!.className).not.toContain("selected");
  });

  it("disables Start while a job runs, offers no Restart, and frees it when the job ends", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    expect(startButton().disabled).toBe(false);
    fireEvent.click(startButton());
    await settle();

    expect(screen.queryByRole("button", { name: "Restart" })).toBeNull();
    expect(startButton().disabled).toBe(true);                                // the user cancels first
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);

    api.fetchTrafficJob.mockResolvedValue({ state: "failed", progress: { done: 0, total: 1, current: null, phase: "evaluation" }, error: "x" });
    await settle(2000);
    expect(startButton().disabled).toBe(false);
  });

  it("keeps Start disabled while the cancel is out", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    let release: (value: unknown) => void = () => undefined;
    api.cancelTrafficJob.mockReturnValue(new Promise((resolve) => { release = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await settle();
    expect(screen.getByText("Cancelling")).toBeTruthy();
    expect(startButton().disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(true);
    await act(async () => { release({ ...running(0, 1), state: "cancelled" }); });
    expect(startButton().disabled).toBe(false);
  });

  it("says the connection is lost, keeps the job and Cancel, and says no more when the backend answers again", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
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

  it("says which phase the job is in, and how long it has run, before it knows its total", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    api.fetchTrafficJob.mockResolvedValue(running(0, null, null, "reading traffic"));
    fireEvent.click(startButton());
    await settle();
    expect(screen.getByLabelText("Phase").textContent).toBe("reading traffic · 00:00");
    expect(screen.queryByText(/aircraft done/)).toBeNull();                  // no total yet: nothing is counted
    await settle(5000);
    expect(screen.getByLabelText("Phase").textContent).toBe("reading traffic · 00:05");
  });

  it("shows each phase the job writes as it is — in plain words — and the elapsed time as minutes and seconds", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    api.fetchTrafficJob.mockResolvedValue(running(0, 2, null, "earliest arrival of each aircraft"));
    fireEvent.click(startButton());
    await settle(65_000);                                                    // a minute and five seconds
    expect(screen.getByLabelText("Phase").textContent).toBe("earliest arrival of each aircraft · 01:05");
    api.fetchTrafficJob.mockResolvedValue(running(1, 2, SWA.flightKey, "building the scene"));
    await settle(2000);
    expect(screen.getByLabelText("Phase").textContent).toBe("building the scene · 01:07");
  });

  it("starts the clock again for the next job", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle(10_000);
    expect(screen.getByLabelText("Phase").textContent).toBe("optimizing 1 of 1 · 00:10");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await settle();
    fireEvent.click(startButton());
    await settle(3000);
    expect(screen.getByLabelText("Phase").textContent).toBe("optimizing 1 of 1 · 00:03");
  });

  it("shows a failed job's reason", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    api.fetchTrafficJob.mockResolvedValue({ state: "failed", progress: { done: 0, total: null, current: null, phase: "evaluation" },
      error: "NoAircraftDynamics: no aircraft dynamics for flight (type ZZZZ)" });
    fireEvent.click(startButton());
    await settle();
    expect(screen.getByRole("alert").textContent).toContain("no aircraft dynamics");
    expect(screen.getByText("Failed")).toBeTruthy();
  });

  const resultCards = () => cardLines("Controlled aircraft");

  it("lists the result of a done job as cards — plain outcome name, what changed, the time it took — and selects a flight on a click", async () => {
    const trackedViewer = {
      dataSources: { length: 1, get: () => ({ entities: { getById: (id: string) => (id === SWA.flightKey ? { id } : undefined) } }) },
      trackedEntity: undefined as unknown,
    };
    app.viewer = trackedViewer;
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [SWA.flightKey]: change({
      firstSolveLosses: 9, finalLosses: 0, landingVsRecordS: -400, optimizeS: 12.34, optimizeCpuS: 11.8, solves: 3 }) }, { windows: 1 }));
    fetchIndex.mockResolvedValue(indexOf([
      group(SWA, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    fireEvent.click(startButton());
    await settle();

    expect(screen.getByText("Ready")).toBeTruthy();
    expect(app.setTrafficScene).toHaveBeenLastCalledWith(expect.objectContaining({ airportCode: "KRDU", jobId: JOB }));
    const result = within(screen.getByLabelText("Result"));
    expect(result.getByRole("status").textContent).toBe(
      "SWA3131, 2026-05-01 — 1 aircraft controlled: 1 separated after re-solve");     // names the flight and the day it was run for
    expect(document.querySelector("table")).toBeNull();
    expect(resultCards()).toEqual([[
      "SWA3131 · B38M · 05L — separated after re-solve",
      "losses: record 28 → first solve 9 → final 0 · lands 6 min 40 s earlier than it really did " +
      "(shortest published route at minimum time; the real flight was vectored) · optimized in 12.3 s, CPU 11.8 s (3 solves, 0 failed)"]]);
    const card = result.getByText(/^SWA3131 · /).closest("li")!;
    expect(card.getAttribute("title")).toBe(`${SWA.flightKey} — separated after re-solve (separated)`);
    expect(result.queryByText(/delay/)).toBeNull();                                  // an M1 job has no slot

    fireEvent.click(card);
    expect(app.setSelectedFlightId).toHaveBeenCalledWith(SWA.flightKey);
    expect(trackedViewer.trackedEntity).toEqual({ id: SWA.flightKey });
    // the legend names the controlled aircraft and the recorded traffic
    expect(result.getByLabelText("Controlled aircraft — its record")).toBeTruthy();
    expect(result.getByLabelText("Controlled aircraft — optimized path")).toBeTruthy();
    expect(result.getByText("Recorded traffic — not controlled")).toBeTruthy();
  });

  async function finishWith(change_: TrafficAircraftChange) {
    api.fetchTrafficJob.mockResolvedValue(running(0, 1));                      // a new job: running until the test says it is done
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [SWA.flightKey]: change_ }));
    fetchIndex.mockResolvedValue(indexOf([group(SWA, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    return within(screen.getByLabelText("Result"));
  }

  it("says under the result how long the job took, by stage", async () => {
    const result = await finishWith(change());
    expect(result.getByLabelText("Timing").textContent).toBe("Total 0:20 — reading traffic 0:05 · optimizing 0:15");
  });

  it("says why a flight lands much earlier on its own line — and nowhere else: there is no footnote", async () => {
    const early = await finishWith(change({ landingVsRecordS: -400 }));
    expect(early.getByText(/lands 6 min 40 s earlier than it really did \(shortest published route at minimum time; the real flight was vectored\)/))
      .toBeTruthy();
    expect(early.queryByText(/^Optimized flights land earlier/)).toBeNull();           // the global footnote is gone
    cleanup();
    const a_little = await finishWith(change({ landingVsRecordS: -30 }));
    expect(a_little.getByText(/lands 30 s earlier than it really did · /)).toBeTruthy();
    expect(a_little.queryByText(/vectored/)).toBeNull();                               // not more than a minute earlier
    cleanup();
    const later = await finishWith(change({ landingVsRecordS: 30 }));
    expect(later.queryByText(/vectored|its slot/)).toBeNull();                         // later in an M1 job: no slot to blame
    cleanup();
    const notFlown = await finishWith(change({ ...NOT_FLOWN }));
    expect(notFlown.queryByText(/vectored/)).toBeNull();
  });

  it("gives the result the panel's height: no nested scroll box, no table", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [SWA.flightKey]: change() }));
    fetchIndex.mockResolvedValue(indexOf([group(SWA, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    const cards = screen.getByLabelText("Controlled aircraft");
    expect(cards.closest(".traffic-job-catalog-scroll")).toBeNull();
    expect(document.querySelector(".traffic-job-table-scroll")).toBeNull();
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    expect(css).not.toMatch(/\.traffic-job-results\s*\{[^}]*(max-height|overflow)/);
    expect(css).not.toMatch(/\.traffic-job-table-scroll/);
  });

  it("names an aircraft that was not flown in the result: its record's losses are not shown, only that it was not flown", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [SWA.flightKey]: change({ ...NOT_FLOWN }) }));
    fetchIndex.mockResolvedValue(indexOf([group(SWA, { status: "failed", traffic: undefined })]));
    await settle(2000);
    expect(resultCards()).toEqual([[
      "SWA3131 · B38M · 05L — not optimized", "not flown · tried for 12.3 s, CPU 11.8 s (3 solves, 0 failed)"]]);
  });

  it("refuses to show a job whose status says nothing of an aircraft its index lists", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    api.fetchTrafficJob.mockResolvedValue(doneStatus({}));
    fetchIndex.mockResolvedValue(indexOf([group(SWA, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    expect(screen.getByRole("alert").textContent).toContain(`lists ${SWA.flightKey} in its index and says nothing of what changed for it`);
    expect(screen.queryByLabelText("Result")).toBeNull();
  });

  it("never touches the top bar's runway selector, whatever the job does", async () => {
    await open("m1");
    fireEvent.click(arrivalCard("SWA3131"));
    fireEvent.click(startButton());
    await settle();
    api.fetchTrafficJob.mockResolvedValue(doneStatus({ [SWA.flightKey]: change() }));
    fetchIndex.mockResolvedValue(indexOf([group(SWA, { traffic: { outcome: "separated", recorded: [], startOffsetsS: [] } })]));
    await settle(2000);
    fireEvent.click(within(screen.getByLabelText("Result")).getByText(/^SWA3131 · /));
    fireEvent.change(screen.getByLabelText("Show runway"), { target: { value: "05L" } });  // the list's own filter, not the top bar's
    expect(app.setSelectedRunway).not.toHaveBeenCalled();
    expect(app.selectedRunway).toBe("05R");
  });

  it("says the solver settings of the batch in one plain sentence, the whole list in its title, read-only", async () => {
    await open("m1");
    const line = screen.getByText("Same settings as the batch experiments");
    expect(line.getAttribute("title")).toContain("max duration 2000 s");
    expect(line.getAttribute("title")).toContain("IPOPT cap 3000");
    expect(line.querySelector("input, select")).toBeNull();
    expect(document.querySelector("details")).toBeNull();                          // no fold to open
  });
});

describe("TrafficJobPanel, all controlled (M2)", () => {
  const lengthSelect = () => screen.getByLabelText("Length") as HTMLSelectElement;
  const manyBlocks = (count: number, stepS: number) => Array.from({ length: count }, (_, n) => ({
    startUtc: new Date(Date.parse("2026-05-01T00:00:00Z") + n * stepS * 1000).toISOString().replace(".000Z", "Z"),
    arrivals: 2, commandable: 2, lossInstants: n % 7, runways: ["05L"] }));
  const withBlocks = (length: string, blocks: unknown[]) => api.fetchTrafficScenarios.mockResolvedValue({
    ...CATALOG, m2: { ...CATALOG.m2, [length]: blocks } });

  it("has no date, hour or minute field: a block length and the catalog's blocks", async () => {
    await open("m2");
    expect(screen.queryByLabelText("UTC day")).toBeNull();
    expect(document.querySelector("input[type=date]")).toBeNull();
    expect(screen.queryByLabelText("Start hour (UTC)")).toBeNull();
    expect(screen.queryByLabelText("Minute")).toBeNull();
    expect(lengthSelect().value).toBe("1800");
    expect(Array.from(lengthSelect().options).map((o) => o.textContent)).toEqual(["15 min", "30 min", "60 min"]);
    expect(Array.from(lengthSelect().options).map((o) => o.value)).toEqual(["900", "1800", "3600"]);
    expect(screen.queryByLabelText("Hide light aircraft")).toBeNull();            // M1's: a block holds every arrival
  });

  it("joins a block's runways with commas and keeps them together on a card: `23L, 23R, 32` never wraps with one alone", async () => {
    withBlocks("1800", [{ startUtc: "2026-05-01T00:00:00Z", arrivals: 6, commandable: 5, lossInstants: 198, runways: ["23L", "23R", "32"] }]);
    await open("m2");
    expect(blockCards()).toEqual([["2026-05-01 00:00–00:30 UTC", "6 arrivals · 5 controllable · 198 seconds in loss · 23L, 23R, 32"]]);
    const keep = document.querySelector(".traffic-job-card .traffic-job-keep")!;
    expect(keep.textContent).toBe("23L, 23R, 32");                                    // the one piece the line may not break
    const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");
    expect(css).toMatch(/\.traffic-job-keep\s*\{[^}]*white-space:\s*nowrap/);
  });

  it("says controllable the same way on the card and in the Selected line, each with its definition in a title", async () => {
    await open("m2");
    expect(blockCard("2026-05-01 14:30–15:00 UTC").closest("li")!.getAttribute("title")).toContain(CONTROLLABLE_TITLE);
    expect(CONTROLLABLE_TITLE).toContain("has an aircraft dynamics model");
    fireEvent.click(blockCard("2026-05-01 14:30–15:00 UTC"));
    expect(selection()).toContain("2 controllable");
    expect(screen.getByLabelText("Selection").getAttribute("title")).toContain(CONTROLLABLE_TITLE);
    expect(selection()).not.toContain("with an aircraft dynamics model");              // not another word for it
  });

  it("heads the list with its blocks, not the arrivals' sentence, and follows the length", async () => {
    await open("m2");
    expect(screen.getByText(
      "4 blocks of 30 min with an arrival; sorted by the seconds their arrivals spend in loss of separation (the records were checked every 1 s)"))
      .toBeTruthy();
    expect(screen.getByText(/recorded aircraft as flown.*not a promise that the optimized flight has one/)).toBeTruthy();   // the same line
    expect(screen.queryByText(/arrivals have a loss they answer for/)).toBeNull();
    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    expect(screen.getByText(/^5 blocks of 15 min with an arrival; sorted by the seconds their arrivals spend in loss of separation/)).toBeTruthy();
    fireEvent.change(lengthSelect(), { target: { value: "3600" } });
    expect(screen.getByText(/^4 blocks of 60 min with an arrival/)).toBeTruthy();
  });

  it("lists the blocks of the length as two-line cards — its span, then arrivals, controllable, losses, runways — most losses first", async () => {
    await open("m2");
    expect(document.querySelector("table")).toBeNull();
    expect(blockCards()).toEqual([
      ["2026-05-01 12:30–13:00 UTC", "1 arrival · 1 controllable · 40 seconds in loss · 05L"],
      ["2026-05-01 00:00–00:30 UTC", "2 arrivals · 2 controllable · 34 seconds in loss · 05L, 05R"],
      ["2026-05-01 14:30–15:00 UTC", "2 arrivals · 2 controllable · 9 seconds in loss · 05L, 32"],
      ["2026-05-02 09:00–09:30 UTC", "1 arrival · 1 controllable · 0 seconds in loss · 05R"],
    ]);
    expect(listed()).toBe("4 of 4 blocks");
    // the commandable count is in the card itself; the title says the block in words
    expect(blockCard("2026-05-01 14:30–15:00 UTC").closest("li")!.getAttribute("title")).toBe(
      "Block 2026-05-01 14:30–15:00 UTC — 2 arrivals, 2 controllable, 9 seconds in loss, runways 05L, 32. " +
      `${CONTROLLABLE_TITLE}. ${lossSecondsTitle(1)}`);

    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    expect(blockCards().map((lines) => [lines[0], lines[1]!.split(" · ")[2]])).toEqual([
      ["2026-05-01 12:30–12:45 UTC", "40 seconds in loss"], ["2026-05-01 00:00–00:15 UTC", "28 seconds in loss"],
      ["2026-05-01 14:45–15:00 UTC", "9 seconds in loss"], ["2026-05-01 00:15–00:30 UTC", "6 seconds in loss"],
      ["2026-05-02 09:00–09:15 UTC", "0 seconds in loss"]]);
    fireEvent.change(lengthSelect(), { target: { value: "3600" } });
    expect(blockCards().map((lines) => lines[0])).toEqual([
      "2026-05-01 12:00–13:00 UTC", "2026-05-01 00:00–01:00 UTC", "2026-05-01 14:00–15:00 UTC", "2026-05-02 09:00–10:00 UTC"]);
  });

  it("sorts by the select, each way, and filters the blocks by a runway they hold", async () => {
    await open("m2");
    sortBy("time-asc");
    expect(spans()).toEqual(["05-01 00:00", "05-01 12:30", "05-01 14:30", "05-02 09:00"]);
    sortBy("time-desc");
    expect(spans()).toEqual(["05-02 09:00", "05-01 14:30", "05-01 12:30", "05-01 00:00"]);
    sortBy("loss-desc");
    expect(spans()).toEqual(["05-01 12:30", "05-01 00:00", "05-01 14:30", "05-02 09:00"]);
    sortBy("loss-asc");
    expect(spans()).toEqual(["05-02 09:00", "05-01 14:30", "05-01 00:00", "05-01 12:30"]);

    const runway = screen.getByLabelText("Show runway") as HTMLSelectElement;
    expect(Array.from(runway.options).map((o) => o.value)).toEqual(["", "05L", "05R", "32"]);
    fireEvent.change(runway, { target: { value: "32" } });
    expect(spans()).toEqual(["05-01 14:30"]);
    expect(listed()).toBe("1 of 4 blocks");
    fireEvent.change(runway, { target: { value: "05R" } });
    expect(spans()).toEqual(["05-02 09:00", "05-01 00:00"]);
  });

  it("starts the block job with the block's UTC start and the length, and not before a block is picked", async () => {
    await open("m2");
    expect(startButton().disabled).toBe(true);
    expect(screen.getByText("Pick the block to optimize.")).toBeTruthy();
    fireEvent.click(blockCard("2026-05-01 14:30–15:00 UTC"));
    expect(selection()).toBe(
      "Selected: Block 2026-05-01 14:30–15:00 UTC — 2 arrivals, 2 controllable, 9 seconds in loss, runways 05L, 32");
    // "controllable" is defined in a title, here as on the card
    expect(screen.getByLabelText("Selection").getAttribute("title")).toContain(CONTROLLABLE_TITLE);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({
      mode: "m2", airport: "KRDU", blockStartUtc: "2026-05-01T14:30:00Z", blockS: 1800 });
  });

  it("disables Start for a block with no arrival that has an aircraft dynamics model, and says why; another block frees it", async () => {
    withBlocks("1800", [...CATALOG.m2["1800"],
      { startUtc: "2026-05-03T03:00:00Z", arrivals: 2, commandable: 0, lossInstants: 0, runways: ["05R"] }]);
    await open("m2");
    expect(screen.queryByLabelText("Start disabled")).toBeNull();
    fireEvent.click(blockCard("2026-05-03 03:00–03:30 UTC"));
    expect(selection()).toBe(
      "Selected: Block 2026-05-03 03:00–03:30 UTC — 2 arrivals, 0 controllable, 0 seconds in loss, runways 05R");
    expect(screen.getByLabelText("Start disabled").textContent).toBe(
      "Start is disabled: no arrival in this block is controllable (has an aircraft dynamics model).");
    expect(startButton().disabled).toBe(true);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).not.toHaveBeenCalled();

    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));
    expect(screen.queryByLabelText("Start disabled")).toBeNull();
    expect(startButton().disabled).toBe(false);
  });

  it("clears a block the runway filter hides, and keeps one that holds the runway", async () => {
    await open("m2");
    const runway = screen.getByLabelText("Show runway") as HTMLSelectElement;
    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));                         // runways 05L, 05R
    fireEvent.change(runway, { target: { value: "05R" } });
    expect(selection()).toMatch(/^Selected: Block 2026-05-01 00:00–00:30 UTC/);
    expect(startButton().disabled).toBe(false);
    fireEvent.change(runway, { target: { value: "32" } });
    expect(selection()).toBe("Pick the block to optimize.");
    expect(startButton().disabled).toBe(true);
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).not.toHaveBeenCalled();
  });

  it("shows 100 blocks at a time, no block dropped, and starts again at one page when the view changes", async () => {
    withBlocks("1800", manyBlocks(250, 1800));
    await open("m2");
    expect(blockCards()).toHaveLength(100);
    expect(screen.getByLabelText("Rows shown").textContent).toBe("Showing 1–100 of 250 · Show 100 more");
    expect(listed()).toBe("250 of 250 blocks");
    fireEvent.click(screen.getByRole("button", { name: "Show 100 more" }));
    expect(blockCards()).toHaveLength(200);
    fireEvent.click(screen.getByRole("button", { name: "Show 50 more" }));
    expect(blockCards()).toHaveLength(250);
    expect(screen.getByLabelText("Rows shown").textContent).toBe("Showing 1–250 of 250");
    sortBy("time-asc");                                                                     // another view: one page again
    expect(blockCards()).toHaveLength(100);
  });

  it("switches the block length at once in a long list: only a page is drawn (6,448 blocks of 15 min at KRDU)", async () => {
    withBlocks("900", manyBlocks(6448, 900));
    await open("m2");
    const sortedBefore = calls.listedRows, titledBefore = calls.blockDetail;
    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    expect(blockCards()).toHaveLength(100);                                              // not 6,448 cards
    expect(screen.getByLabelText("Rows shown").textContent).toBe("Showing 1–100 of 6448 · Show 100 more");
    expect(screen.getByText(/^6448 blocks of 15 min with an arrival; sorted by the seconds their arrivals spend in loss of separation/)).toBeTruthy();
    expect(calls.blockDetail - titledBefore).toBe(100);                                  // one title per card drawn
    expect(calls.listedRows - sortedBefore).toBe(1);                                      // sorted once
    fireEvent.change(lengthSelect(), { target: { value: "1800" } });
    expect(blockCards()).toHaveLength(4);
  });

  it("re-renders only the two cards whose selection changed when a block is picked, in a long list", async () => {
    withBlocks("1800", manyBlocks(400, 1800));
    await open("m2");
    expect(calls.blockDetail).toBe(100);
    const [first, fifth] = [blockCards()[0][0]!, blockCards()[5][0]!];
    fireEvent.click(blockCard(first));
    expect(calls.blockDetail).toBe(101);
    fireEvent.click(blockCard(fifth));
    expect(calls.blockDetail).toBe(103);
    expect(blockCard(fifth).closest("li")!.className).toContain("selected");
    expect(blockCard(first).closest("li")!.className).not.toContain("selected");
  });

  it("takes the block length of the list it picked from, and drops the pick when the length changes", async () => {
    await open("m2");
    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    fireEvent.click(blockCard("2026-05-01 14:45–15:00 UTC"));
    fireEvent.click(startButton());
    await settle();
    expect(api.startTrafficJob).toHaveBeenCalledWith({
      mode: "m2", airport: "KRDU", blockStartUtc: "2026-05-01T14:45:00Z", blockS: 900 });

    fireEvent.change(lengthSelect(), { target: { value: "3600" } });           // a 15-minute block is not a 60-minute one
    expect(screen.getByText("Pick the block to optimize.")).toBeTruthy();
    expect(startButton().disabled).toBe(true);
  });

  it("shows an M2 job's phase, how many aircraft are settled and the elapsed time as an M1 job's, through every phase of a block", async () => {
    await open("m2");
    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));
    api.fetchTrafficJob.mockResolvedValue(running(0, null, null, "reading traffic"));
    fireEvent.click(startButton());
    await settle();
    expect(screen.getByText("Computing")).toBeTruthy();
    expect(screen.getByLabelText("Phase").textContent).toBe("reading traffic · 00:00");
    expect(screen.queryByText(/aircraft done/)).toBeNull();                          // the job has not chosen its aircraft yet
    const sequence: Array<[TrafficJobStatus, string, string | null]> = [
      [running(0, 12, null, "earliest arrival of each aircraft"), "earliest arrival of each aircraft · 00:12", "0 of 12 aircraft done"],
      [running(1, 12, "AAL1_05L_a_1", "earliest arrival of each aircraft"), "earliest arrival of each aircraft · 00:14", "1 of 12 aircraft done · last: AAL1"],
      [running(1, 12, "AAL1_05L_a_1", "schedule"), "schedule · 00:16", "1 of 12 aircraft done · last: AAL1"],
      [running(3, 12, "DAL2_05L_a_1", "optimizing 2 of 11"), "optimizing 2 of 11 · 00:18", "3 of 12 aircraft done · last: DAL2"],
    ];
    await settle(10_000);                                                              // ten seconds of reading traffic, then the first poll below
    for (const [status, phaseText, countText] of sequence) {
      api.fetchTrafficJob.mockResolvedValue(status);
      await settle(2000);
      expect(screen.getByLabelText("Phase").textContent).toBe(phaseText);
      expect(screen.getByLabelText("Progress").textContent).toContain(countText!);
    }
  });

  it("shows a running job's progress right under the header, above the list, so it is in view however long the list is", async () => {
    await open("m2");
    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));
    api.fetchTrafficJob.mockResolvedValue(running(2, 5, "AAL1_05L_a_1", "optimizing 3 of 5"));
    fireEvent.click(startButton());
    await settle();
    const progress = screen.getByLabelText("Progress");
    const list = screen.getByLabelText("Scenario list");
    const header = screen.getByText("Computing").closest("header")!;
    expect(header.compareDocumentPosition(progress) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();       // after the header ...
    expect(progress.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();         // ... and before the list
    expect(progress.textContent).toContain("optimizing 3 of 5 · 00:00");
    expect(progress.textContent).toContain("2 of 5 aircraft done · last: AAL1");
  });

  /**
   * A job done over three aircraft: SWA flown, DAL flown and off the landing gates, NAMELESS not flown (its slot solve failed);
   * and one more arrival of the block, a C172, that could not be controlled and flew its record.
   */
  const STAYED = { "N777_05L_a_9": { callsign: "N777", type: "C172" }, "NOCALL_05R_b_9": { callsign: null, type: null } };
  const doneScene = (extra: Record<string, unknown> = {}, stayed: Record<string, TrafficStayedRecord> = STAYED) => {
    api.fetchTrafficJob.mockResolvedValue(doneStatus({
      [SWA.flightKey]: change({ firstSolveLosses: 9, finalLosses: 0, landingVsRecordS: -400, delayS: 0, blockCheckLosses: 0 }),
      [DAL.flightKey]: change({ type: "A321", firstSolveLosses: 6, finalLosses: 2, landingVsRecordS: -3.5, delayS: 80.4,
        blockCheckLosses: 3, optimizeS: 20, solves: 1 }),
      [NAMELESS.flightKey]: change({ ...NOT_FLOWN, type: null, delayS: 2458.4 }),
    }, { aircraft: 2, flown_aircraft_with_a_loss_left_after_the_block: {
      visual: { answered: 1, not_answered: 2 }, ifr: { answered: 2, not_answered: 2 } }, ...extra },
    { totalS: 741, phases: { "reading traffic": 45, "earliest arrivals": 190, schedule: 1, optimizing: 482, evaluation: 20, "building the scene": 3 } },
    stayed));
    const index = indexOf([
      group(SWA, { scene: { startOffsetS: 0, outcome: "separated_at_baseline", delayS: 0 } }),
      group(DAL, { status: "offTarget", scene: { startOffsetS: 600, outcome: "solve_failed", delayS: 80.4 } }),
      group(NAMELESS, { status: "failed", scene: { startOffsetS: 300, outcome: "BaselineFailed: slot solve (delay 2458.4 s): x", delayS: 2458.4 } }),
    ], { startUtc: "2026-05-01T00:02:46.867Z", background: { recorded: [], startOffsetsS: [] } });
    // DAL88 is drawn yellow: the job's evaluation report (the file the index names) says which gates it failed
    const report = reportOf([{ flight_key: DAL.flightKey, violations: ["lateral", "speed"], lateral_m: 341, crossing_speed_ms: 63.4 }]);
    fetchIndex.mockImplementation(async (url: string) => (url.endsWith("r.json") ? report : index));
  };

  async function runFirstBlock() {
    await open("m2");
    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));
    fireEvent.click(startButton());
    doneScene();
    await settle(2000);
  }

  const resultCards = () => cardLines("Controlled aircraft");

  it("lists every controlled aircraft as a card of two lines, and a summary that names its block, those that missed the gates and those with a loss left", async () => {
    await runFirstBlock();
    const result = within(screen.getByLabelText("Result"));
    expect(result.getByRole("status").textContent).toBe(
      "Block 2026-05-01 00:00–00:30 UTC — 3 aircraft controlled: 1 separated at the first solve (no re-solve), " +
      "1 loss left — the best of its solves is shown (fewest losses), 1 not optimized: its slot could not be flown" +
      " · 1 missed the landing gates (yellow path): DAL88" +
      " · 2 more landed in the block and flew their records (not controllable, named below)" +
      " · slot delay median 40 s, largest 80 s, 1 over 60 s" +                       // the flown aircraft's delays: 0 and 80.4, not the 2458 s
      " · after the block's final check: 1 aircraft still has a loss they answer for (DAL88); " +
      "2 aircraft with a loss others caused");
    expect(resultCards()).toEqual([
      ["SWA3131 · B38M · 05L — separated at the first solve (no re-solve)",
        "when flown: record 28 → first solve 9 → final 0 · after the whole block: 0 · " +
        "lands 6 min 40 s earlier than it really did (shortest published route at minimum time; the real flight was vectored) · " +
        "slot delay 0 s (after its earliest arrival) · " +
        "optimized in 12.3 s, CPU 11.8 s (3 solves, 0 failed, the earliest-arrival solves included)"],
      ["DAL88 · A321 · 32 — loss left — the best of its solves is shown (fewest losses)",
        "when flown: record 6 → first solve 6 → final 2 · after the whole block: 3 (from aircraft flown after it) · " +
        "lands 4 s earlier than it really did · slot delay 80 s (after its earliest arrival) · " +
        "optimized in 20.0 s, CPU 11.8 s (1 solve, 0 failed, the earliest-arrival solves included) · missed the landing gates (yellow): " +
        "lateral 41 m too far (341 m, limit 300 m); speed 3.2 m/s too fast (63.4 m/s, window 55.2–60.2 m/s)"],
      ["N123AB · — · 05R — not optimized: its slot could not be flown",
        "not flown — the schedule gave it a slot 2458 s after its earliest arrival · " +
        "tried for 12.3 s, CPU 11.8 s (3 solves, 0 failed, the earliest-arrival solves included)"],
    ]);
    expect(result.queryByText("Delay (s)")).toBeNull();                              // no delay column: the delay is in the card
    expect(result.getByText("Recorded traffic — not controlled, outside the scheduled set")).toBeTruthy();
    // the yellow one's title carries the evaluation's own codes, so a code the card has no words for is still there
    expect(result.getByText(/^DAL88 · /).closest("li")!.getAttribute("title")).toBe(
      `${DAL.flightKey} — loss left — the best of its solves is shown (fewest losses) (solve_failed) — the evaluation's own words: lateral, speed`);
  });

  it("names each arrival of the block that could not be controlled: callsign, type with its plain name, and why", async () => {
    await runFirstBlock();
    const stayed = within(screen.getByLabelText("Not controllable"));
    expect(stayed.getAllByRole("listitem").map((li) => li.textContent)).toEqual([
      "N777 · C172: not controllable (no aircraft dynamics model), flew its record",
      "NOCALL · —: not controllable (no aircraft dynamics model), flew its record",     // no callsign: the key's first field; no type: a dash
    ]);
    expect(stayed.getByText("C172").getAttribute("title")).toBe("Cessna 172");           // the app has a name for this type
  });

  it("lists no stayed record when every arrival of the block was controlled, and says nothing of one in the summary", async () => {
    await open("m2");
    fireEvent.click(blockCard("2026-05-01 00:00–00:30 UTC"));
    fireEvent.click(startButton());
    doneScene({}, {});
    await settle(2000);
    expect(screen.queryByLabelText("Not controllable")).toBeNull();
    expect(within(screen.getByLabelText("Result")).getByRole("status").textContent).not.toContain("flew");
  });

  it("lists, under the block, which aircraft carry its recorded losses — an aircraft not flown included — so the block's number adds up", async () => {
    await runFirstBlock();
    const line = within(screen.getByLabelText("Result")).getByLabelText("Block losses");
    // the block's card says 34 seconds in loss; they are SWA3131's 28 and N123AB's 6, which the job did not fly
    expect(line.textContent).toBe("The block's 34 seconds in loss in the records: SWA3131 28 · N123AB 6 (not flown)");
    expect(line.getAttribute("title")).toBe(lossSecondsTitle(1));
    expect(blockCard("2026-05-01 00:00–00:30 UTC").closest("li")!.textContent).toContain("34 seconds in loss");   // the card the line reconciles
  });

  it("says the block's losses only for an M2 job, and for the block the job was started for", async () => {
    await runFirstBlock();
    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    fireEvent.click(blockCard("2026-05-01 14:45–15:00 UTC"));                           // another block picked afterwards
    expect(within(screen.getByLabelText("Result")).getByLabelText("Block losses").textContent).toMatch(/^The block's 34 seconds/);
  });

  it("explains the yellow of an aircraft in its line, and never reads the report of a job with none", async () => {
    await runFirstBlock();
    expect(fetchIndex).toHaveBeenCalledWith(expect.stringMatching(/\/files\/r\.json$/));
  });

  it("says under the summary how long each stage took, and in total", async () => {
    await runFirstBlock();
    expect(within(screen.getByLabelText("Result")).getByLabelText("Timing").textContent).toBe(
      "Total 12:21 — reading traffic 0:45 · earliest arrivals 3:10 · schedule 0:01 · optimizing 8:02 · evaluation 0:20 · " +
      "building the scene 0:03");
  });

  it("says how the record's losses were counted when the census did not use the job's check step", async () => {
    api.fetchTrafficScenarios.mockResolvedValue({ ...CATALOG, config: { ...CATALOG.config, stepS: 2 } });
    await runFirstBlock();
    expect(resultCards()[0][1]).toContain("when flown: record 28 at 2 s steps → first solve 9 → final 0");
    expect(within(screen.getByLabelText("Result")).getByLabelText("Block losses").textContent).toMatch(/^The block's 68 seconds/);   // 34 checks of 2 s
  });

  it("keeps the result's inputs when the panel's inputs change afterwards: a changed input never looks like a new result", async () => {
    await runFirstBlock();
    const heading = () => within(screen.getByLabelText("Result")).getByRole("status").textContent!;
    expect(heading()).toContain("Block 2026-05-01 00:00–00:30 UTC");

    fireEvent.change(lengthSelect(), { target: { value: "900" } });
    fireEvent.click(blockCard("2026-05-01 14:45–15:00 UTC"));
    expect(heading()).toContain("Block 2026-05-01 00:00–00:30 UTC");          // still the block the job was run for
  });
});
