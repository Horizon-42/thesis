import { afterEach, describe, expect, it, vi } from "vitest";
import {
  averageDynamicsComparisonHistory,
  clearDynamicsComparisonHistory,
  fetchDynamicsComparisonHistoryCount,
  runDynamicsComparison,
  type DynamicsComparisonRequest,
} from "../dynamicsComparisonClient";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const request: DynamicsComparisonRequest = {
  initialState: {
    lon: -78.73,
    lat: 35.41,
    altM: 2300,
    speedMps: 130,
    headingDeg: 45,
    flightPathDeg: -2,
    massKg: 78000,
    aircraftType: "A320",
  },
  control: { thrustN: 70000, bankDeg: 0, loadFactor: 1 },
  durationS: 240,
  dtS: 0.1,
};

function responsePayload() {
  return {
    ok: true,
    durationS: 240,
    requestedDurationS: 240,
    dtS: 0.1,
    aircraftType: "A320",
    historyCount: 1,
    systems: [
      { key: "A", label: "A · fixed tangent ENU", colorRgba: [244, 114, 22, 240], isReference: false },
      { key: "B", label: "B · re-anchored ENU", colorRgba: [226, 232, 240, 245], isReference: true },
      { key: "C", label: "C · geodetic +transport", colorRgba: [56, 189, 248, 240], isReference: false },
      { key: "D", label: "D · geodetic no transport", colorRgba: [250, 204, 21, 240], isReference: false },
    ],
    playback: {
      epochIso: "2026-01-01T00:00:00Z",
      multiplier: 6,
      czml: [{ id: "document" }, { id: "dyncmp-A" }],
    },
    chart: {
      distanceKm: [0, 0.5, 1.0],
      timeS: [0, 3.8, 7.6],
      series: {
        A: { horiz: [0, 5, 12], alt: [0, -1, -3], head: [0, 0.01, 0.02], speed: [0, 0.1, 0.2] },
        C: { horiz: [0, 0.001, 0.002], alt: [0, 0, 0], head: [0, 0, 0], speed: [0, 0, 0] },
        D: { horiz: [0, 4, 9], alt: [0, -2, -5], head: [0, 0.05, 0.1], speed: [0, 0.1, 0.2] },
      },
      final: {
        A: { horiz: 12, alt: -3, head: 0.02, speed: 0.2 },
        C: { horiz: 0.002, alt: 0, head: 0, speed: 0 },
        D: { horiz: 9, alt: -5, head: 0.1, speed: 0.2 },
      },
    },
  };
}

describe("dynamicsComparisonClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the comparison request to the backend and parses the result", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responsePayload()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await runDynamicsComparison(request);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/dynamics-comparison/run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }),
    );
    expect(result.systems.map((s) => s.key)).toEqual(["A", "B", "C", "D"]);
    expect(result.systems.find((s) => s.isReference)?.key).toBe("B");
    expect(result.playback.multiplier).toBe(6);
    expect(result.playback.czml).toHaveLength(2);
    expect(Object.keys(result.chart.series)).toEqual(["A", "C", "D"]);
    expect(result.chart.final.A.speed).toBe(0.2);
    expect(result.chart.distanceKm).toHaveLength(3);
  });

  it("throws the backend error message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: false, error: "bad start state" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    await expect(runDynamicsComparison(request)).rejects.toThrow("bad start state");
  });

  it("rejects a malformed response (missing chart)", async () => {
    const broken = responsePayload();
    delete (broken as Record<string, unknown>).chart;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(broken), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    await expect(runDynamicsComparison(request)).rejects.toThrow(/chart/);
  });

  it("rejects a system with a malformed colour", async () => {
    const broken = responsePayload();
    broken.systems[0].colorRgba = [1, 2, 3] as unknown as [number, number, number, number];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(broken), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ));

    await expect(runDynamicsComparison(request)).rejects.toThrow(/colorRgba/);
  });

  it("parses the run's historyCount", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(responsePayload())));
    const result = await runDynamicsComparison(request);
    expect(result.historyCount).toBe(1);
  });
});

describe("dynamics comparison history endpoints", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches the stored-run count", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true, historyCount: 4 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(fetchDynamicsComparisonHistoryCount()).resolves.toBe(4);
    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8765/dynamics-comparison/history");
  });

  it("parses a backend-averaged result", async () => {
    const payload = {
      ok: true,
      runCount: 3,
      systems: responsePayload().systems,
      chart: responsePayload().chart,
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(payload)));
    const averaged = await averageDynamicsComparisonHistory();
    expect(averaged.runCount).toBe(3);
    expect(Object.keys(averaged.chart.series)).toEqual(["A", "C", "D"]);
    expect(averaged.systems.map((s) => s.key)).toEqual(["A", "B", "C", "D"]);
  });

  it("throws the backend message when there is no history to average", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      jsonResponse({ ok: false, error: "No comparison history to average yet." }),
    ));
    await expect(averageDynamicsComparisonHistory()).rejects.toThrow(/No comparison history/);
  });

  it("clears history and returns the new count", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: true, historyCount: 0 })));
    await expect(clearDynamicsComparisonHistory()).resolves.toBe(0);
  });
});
