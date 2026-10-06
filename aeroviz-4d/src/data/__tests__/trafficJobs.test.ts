import { afterEach, describe, expect, it, vi } from "vitest";
import {
  TRAFFIC_BLOCK_LENGTHS_S,
  beaconCancelTrafficJob,
  cancelTrafficJob,
  fetchTrafficArrivals,
  fetchTrafficJob,
  isTrafficArrivals,
  isTrafficJobStatus,
  arrivalCallsign,
  startTrafficJob,
  trafficJobFilesUrl,
} from "../trafficJobs";

const JOB = "20261006T120000123456Z-0123abcd";
const status = (over: Record<string, unknown> = {}) => ({
  state: "running", progress: { done: 1, total: 3, current: "K" }, error: null, ...over });
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
    expect(isTrafficJobStatus(status({ progress: { done: 0, total: null, current: null } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "failed", error: "ValueError: x" }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: {
      flown_aircraft_with_a_loss_left_after_the_block: {
        visual: { answered: 1, not_answered: 0 }, ifr: { answered: 2, not_answered: 3 } } } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: { windows: 1 } }))).toBe(true);
    expect(isTrafficJobStatus(status({ state: "done", summary: { aircraft: 4, skipped_no_dynamics: 1 } }))).toBe(true);
  });

  it("refuse what it does not", () => {
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, runway: 5 }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, callsign: 12 }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", date: "d", arrivals: [{ ...arrival, callsign: undefined }] })).toBe(false);
    expect(isTrafficArrivals({ airport: "KRDU", arrivals: [] })).toBe(false);
    expect(isTrafficJobStatus(status({ state: "paused" }))).toBe(false);
    expect(isTrafficJobStatus(status({ progress: { done: "1", total: 3, current: null } }))).toBe(false);
    expect(isTrafficJobStatus(status({ error: undefined }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", summary: {
      flown_aircraft_with_a_loss_left_after_the_block: { visual: { answered: 1 }, ifr: {} } } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", summary: { aircraft: "4" } }))).toBe(false);
    expect(isTrafficJobStatus(status({ state: "done", summary: { skipped_no_dynamics: null } }))).toBe(false);
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
