import { afterEach, describe, expect, it, vi } from "vitest";
import {
  TRAFFIC_BLOCK_LENGTHS_S,
  TRAFFIC_CATALOG_SCHEMA,
  beaconCancelTrafficJob,
  cancelTrafficJob,
  fetchTrafficArrivals,
  fetchTrafficJob,
  fetchTrafficScenarios,
  isTrafficArrivals,
  isTrafficJobStatus,
  isTrafficScenarioCatalog,
  arrivalCallsign,
  startTrafficJob,
  trafficJobFilesUrl,
} from "../trafficJobs";
import catalogMirror from "./fixtures/trafficScenarioCatalog.json";

const JOB = "20261006T120000123456Z-0123abcd";
const status = (over: Record<string, unknown> = {}) => ({
  state: "running", progress: { done: 1, total: 3, current: "K", phase: "optimizing 1 of 3" }, error: null,
  // a done status carries what changed for each aircraft (the backend sends it with the summary)
  ...(over.state === "done" ? { perAircraft: { K: CHANGE }, timing: TIMING, stayedRecords: {} } : {}), ...over });
const CHANGE = { type: "A320", firstSolveLosses: 12, finalLosses: 0, landingVsRecordS: -3.5, delayS: null,
  blockCheckLosses: null, optimizeS: 12.3, optimizeCpuS: 11.8, solves: 3, failedSolves: 1 };
const TIMING = { totalS: 75.3, phases: { "reading traffic": 40, optimizing: 12.3, evaluation: 20, "building the scene": 3 } };
const arrival = { flightKey: "K", callsign: "AAL1", runway: "05L", type: "A320", entryUtc: "x", landingUtc: "y" };

function respond(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  const ok = init.ok ?? true;
  const fetchMock = vi.fn(async (_url: string, _options?: RequestInit) => ({
    ok, status: init.status ?? (ok ? 200 : 400),
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => body, text: async () => JSON.stringify(body),
  }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe("the shape guards", () => {
  it("accept what the backend sends", () => {
    expect(isTrafficArrivals({ airport: "KRDU", date: "2026-05-21", arrivals: [arrival, { ...arrival, type: null }] })).toBe(true);
    // a flight the roster has no callsign of must not refuse the whole day
    expect(isTrafficArrivals({ airport: "KRDU", date: "2026-05-29", arrivals: [arrival, { ...arrival, callsign: null }] })).toBe(true);
    expect(isTrafficJobStatus(status())).toBe(true);
    expect(isTrafficJobStatus(status({ progress: { done: 0, total: null, current: null, phase: "reading traffic" } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "failed", error: "ValueError: x" }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: {
      flown_aircraft_with_a_loss_left_after_the_block: {
        visual: { answered: 1, not_answered: 0 }, ifr: { answered: 2, not_answered: 3 } } } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: { windows: 1 } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: { aircraft: 4 } }))).toBe(true);
    // the arrivals of a block that stayed their records: named by flight key; the roster may have no callsign, the traffic no type
    expect(isTrafficJobStatus(status({ state: "done", stayedRecords: {
      A: { callsign: "N123", type: "C172" }, B: { callsign: null, type: null } } }))).toBe(true);
    // an aircraft whose failed sidecar lost its solves: the four are null together
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, firstSolveLosses: null, finalLosses: null,
      landingVsRecordS: null, optimizeS: null, optimizeCpuS: null, solves: null, failedSolves: null } } }))).toBe(true);
    // an aircraft that was not flown: nothing but its slot's delay and what its attempts cost; an untyped one has no type
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, firstSolveLosses: null,
      finalLosses: null, landingVsRecordS: null, delayS: 12.5, failedSolves: 3 } } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, type: null } } }))).toBe(true);
  });

  it("refuse what it does not", () => {
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, runway: 5 }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, callsign: 12 }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, callsign: undefined }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", arrivals: [] })).toBe(false);
    expect(isTrafficJobStatus(status({ state: "paused" }))).toBe(false);
    expect(isTrafficJobStatus(status({ progress: { done: "1", total: 3, current: null, phase: "x" } }))).toBe(false);
    expect(isTrafficJobStatus(status({ progress: { done: 1, total: 3, current: null } }))).toBe(false);          // no phase
    expect(isTrafficJobStatus(status({ progress: { done: 1, total: 3, current: null, phase: 3 } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: undefined }))).toBe(false);                 // a done job says what changed
    expect(isTrafficJobStatus(status({ state: "done", timing: undefined }))).toBe(false);                      // nor how long it took
    expect(isTrafficJobStatus(status({ state: "done", stayedRecords: undefined }))).toBe(false);               // nor who stayed their records
    expect(isTrafficJobStatus(status({ state: "done", stayedRecords: { A: { callsign: 5, type: null } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", stayedRecords: { A: { callsign: "N1" } } }))).toBe(false);    // a type is text or null
    expect(isTrafficJobStatus(status({ state: "done", timing: { totalS: "75", phases: {} } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", timing: { totalS: 75, phases: { optimizing: "12" } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, finalLosses: "0" } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, type: 7 } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, solves: "3" } } }))).toBe(false);
    // what the solver spent is all numbers or all null (the failure lost the solves): never some of them, never absent
    for (const field of ["optimizeS", "optimizeCpuS", "solves", "failedSolves"]) {
      expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, [field]: null } } })), field).toBe(false);
      expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { ...CHANGE, [field]: undefined } } })), field).toBe(false);
    }
    expect(isTrafficJobStatus(status({ state: "done", perAircraft: { K: { firstSolveLosses: 1 } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ error: undefined }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", summary: {
      flown_aircraft_with_a_loss_left_after_the_block: { visual: { answered: 1 }, ifr: {} } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", summary: { aircraft: "4" } }))).toBe(false);
    expect(isTrafficJobStatus(null)).toBe(false);
  });
});

describe("arrivalCallsign", () => {
  it("is the callsign, else the first field of the flight key", () => {
    expect(arrivalCallsign({ callsign: "AAL1", flightKey: "XYZ9_05L_a_1" })).toBe("AAL1");
    expect(arrivalCallsign({ callsign: null, flightKey: "N123AB_05L_a00001_20260529T000300Z" })).toBe("N123AB");
  });
});

describe("the client", () => {
  it("asks for a day's arrivals of an airport", async () => {
    const mock = respond({ airport: "KRDU", date: "2026-05-21", arrivals: [arrival] });
    const found = await fetchTrafficArrivals("KRDU", "2026-05-21");
    expect(String(mock.mock.calls[0][0])).toMatch(/\/traffic\/arrivals\?airport=KRDU&date=2026-05-21$/);
    expect(found.arrivals).toHaveLength(1);
  });

  it("starts a job with the request as it is and returns its id; a refusal rejects with the backend's reason", async () => {
    const mock = respond({ jobId: JOB });
    const request = { mode: "m2", airport: "KRDU", blockStartUtc: "2026-05-21T17:00:00Z", blockS: 900 } as const;
    expect(await startTrafficJob(request)).toBe(JOB);
    const [url, options] = mock.mock.calls[0];
    expect(String(url)).toMatch(/\/traffic\/jobs$/);
    expect(options?.method).toBe("POST");
    expect(JSON.parse(String(options?.body))).toEqual(request);

    respond({ ok: false, error: "a traffic job is running: cancel it or wait for it" }, { ok: false, status: 409 });
    await expect(startTrafficJob(request)).rejects.toThrow("a traffic job is running");
    respond({ ok: true }, {});
    await expect(startTrafficJob(request)).rejects.toThrow("without a jobId");
  });

  it("reads a job's status and cancels it", async () => {
    const mock = respond(status());
    expect((await fetchTrafficJob(JOB)).progress.done).toBe(1);
    expect(String(mock.mock.calls[0][0])).toMatch(new RegExp(`/traffic/jobs/${JOB}$`));

    const cancel = respond(status({ state: "cancelled" }));
    expect((await cancelTrafficJob(JOB)).state).toBe("cancelled");
    expect(String(cancel.mock.calls[0][0])).toMatch(new RegExp(`/traffic/jobs/${JOB}/cancel$`));
    expect(cancel.mock.calls[0][1]?.method).toBe("POST");

    respond({ state: "nonsense" });
    await expect(fetchTrafficJob(JOB)).rejects.toThrow("not a traffic job status");
  });

  it("cancels on page unload with a beacon", () => {
    const beacon = vi.fn((_url: string) => true);
    vi.stubGlobal("navigator", { sendBeacon: beacon });
    beaconCancelTrafficJob(JOB);
    expect(String(beacon.mock.calls[0][0])).toMatch(new RegExp(`/traffic/jobs/${JOB}/cancel$`));
  });

  it("names the base URL of a job's files with a trailing slash", () => {
    expect(trafficJobFilesUrl(JOB)).toMatch(new RegExp(`/traffic/jobs/${JOB}/files/$`));
  });

  it("takes the block lengths the backend takes", () => {
    expect([...TRAFFIC_BLOCK_LENGTHS_S]).toEqual([900, 1800, 3600]);
  });
});

// The fixture is a MIRROR of what `traffic_scenarios.build_catalog` and `main` write (`trafficScenarioCatalog.json`);
// `4dTrajectory/optimization/traffic/tests/test_scenario_catalog_mirror.py` pins its field names to the writer's.
type Json = Record<string, unknown>;
const cloneCatalog = (): Json => JSON.parse(JSON.stringify(catalogMirror)) as Json;

describe("the scenario catalog guard", () => {
  it("accepts the catalog the census writes, a null callsign, type and category included", () => {
    expect(isTrafficScenarioCatalog(catalogMirror)).toBe(true);
    expect((catalogMirror.m1 as Json[]).some((row) => row.category === null)).toBe(true);
    expect((catalogMirror.m1 as Json[]).some((row) => row.category === "I")).toBe(true);
    const nameless = (catalogMirror.m1 as Json[]).find((row) => row.callsign === null);
    expect(nameless).toBeDefined();
    expect((catalogMirror.m1 as Json[]).some((row) => row.type === null)).toBe(true);
    expect(isTrafficScenarioCatalog({ ...cloneCatalog(), config: { ...catalogMirror.config, limit: 300 } })).toBe(true);
  });

  it("names its schema as the backend's (a MIRROR, pinned) and refuses another one", () => {
    expect(catalogMirror.schema).toBe(TRAFFIC_CATALOG_SCHEMA);
    expect(isTrafficScenarioCatalog({ ...cloneCatalog(), schema: "traffic-scenario-catalog-v1" })).toBe(false);
  });

  it("reads exactly these fields: the catalog lacking any one of them is refused", () => {
    const M1_FIELDS = ["flightKey", "callsign", "runway", "type", "entryUtc", "landingUtc", "category", "lossInstants",
      "kinds", "tightest", "recordedAircraft"];
    const BLOCK_FIELDS = ["startUtc", "arrivals", "commandable", "lossInstants", "runways"];
    // the rows of the writer carry these fields and no others: nothing is read that is not there, nothing there is dropped
    expect(Object.keys(catalogMirror.m1[0]).sort()).toEqual([...M1_FIELDS].sort());
    for (const length of TRAFFIC_BLOCK_LENGTHS_S) {
      expect(Object.keys(catalogMirror.m2[String(length) as "900"][0]).sort()).toEqual([...BLOCK_FIELDS].sort());
    }
    const READ: Array<[string[], string]> = [
      [[], "schema"], [[], "airport"], [[], "writtenUtc"], [[], "m1"], [[], "m2"], [[], "config"], [[], "counts"],
      [["config"], "stepS"], [["config"], "limit"],
      [["counts"], "arrivals"], [["counts"], "judged"], [["counts"], "withLoss"],
      ...M1_FIELDS.map((field): [string[], string] => [["m1", "0"], field]),
      ...TRAFFIC_BLOCK_LENGTHS_S.flatMap((length) =>
        BLOCK_FIELDS.map((field): [string[], string] => [["m2", String(length), "0"], field])),
      ...TRAFFIC_BLOCK_LENGTHS_S.map((length): [string[], string] => [["m2"], String(length)]),
    ];
    for (const [path, field] of READ) {
      const catalog = cloneCatalog();
      const holder = path.reduce((node, key) => (node as Record<string, unknown>)[key], catalog as unknown) as Json;
      expect(Object.keys(holder), `${path.join(".")}.${field} is in the fixture`).toContain(field);
      delete holder[field];
      expect(isTrafficScenarioCatalog(catalog), `without ${[...path, field].join(".")}`).toBe(false);
    }
  });

  it("refuses a row of the wrong type", () => {
    const wrong = (edit: (catalog: Json) => void) => {
      const catalog = cloneCatalog();
      edit(catalog);
      return isTrafficScenarioCatalog(catalog);
    };
    expect(wrong((c) => { (c.m1 as Json[])[0].lossInstants = "28"; })).toBe(false);
    expect(wrong((c) => { (c.m1 as Json[])[0].kinds = [1]; })).toBe(false);
    expect(wrong((c) => { (c.m1 as Json[])[0].category = 4; })).toBe(false);
    expect(wrong((c) => { (c.m1 as Json[])[0].tightest = null; })).toBe(false);
    expect(wrong((c) => { ((c.m2 as Json)["900"] as Json[])[0].runways = "05L"; })).toBe(false);
    expect(isTrafficScenarioCatalog(null)).toBe(false);
  });
});

describe("fetchTrafficScenarios", () => {
  it("asks for an airport's scenario list and returns the catalog", async () => {
    const mock = respond(catalogMirror);
    const found = await fetchTrafficScenarios("KRDU");
    expect(String(mock.mock.calls[0][0])).toMatch(/\/traffic\/scenarios\?airport=KRDU$/);
    expect(found.counts.withLoss).toBe(5);
  });

  it("rejects with the backend's own message: the command a missing catalog is made with", async () => {
    const message = "no scenario catalog for KRDU at /x; make it with `python 4dTrajectory/optimization/traffic_scenarios.py --airport KRDU`";
    respond({ ok: false, error: message }, { ok: false, status: 404 });
    await expect(fetchTrafficScenarios("KRDU")).rejects.toThrow(message);
    respond({ ok: false, error: "catalog.json has schema 'x'" }, { ok: false, status: 500 });
    await expect(fetchTrafficScenarios("KRDU")).rejects.toThrow("has schema 'x'");
  });

  it("refuses an answer that is not a catalog", async () => {
    respond({ airport: "KRDU" });
    await expect(fetchTrafficScenarios("KRDU")).rejects.toThrow("not a traffic scenario catalog");
  });
});
