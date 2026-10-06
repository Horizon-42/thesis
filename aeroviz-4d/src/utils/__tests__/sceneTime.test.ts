import * as Cesium from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { fetchJson } = vi.hoisted(() => ({ fetchJson: vi.fn() }));
vi.mock("../fetchJson", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../fetchJson")>()), fetchJson }));

import { clearEntryCache, entryUtcOf, formatSceneTime, landingDay, sceneRealTimeMs } from "../sceneTime";

const EPOCH = "2026-04-01T08:00:00Z";
const clockAt = (secondsAfterEpoch: number) =>
  Cesium.JulianDate.addSeconds(Cesium.JulianDate.fromIso8601(EPOCH), secondsAfterEpoch, new Cesium.JulianDate());

describe("the real time of a scene's clock", () => {
  const scene = { startUtc: "2026-05-21T17:47:18.959Z", epoch: EPOCH };

  it("is the scene's start at the display epoch, and runs with the clock from there", () => {
    expect(sceneRealTimeMs(scene, clockAt(0))).toBe(Date.parse("2026-05-21T17:47:18.959Z"));
    expect(sceneRealTimeMs(scene, clockAt(125))).toBe(Date.parse("2026-05-21T17:49:23.959Z"));
    expect(sceneRealTimeMs(scene, clockAt(-10))).toBe(Date.parse("2026-05-21T17:47:08.959Z"));   // before the epoch too
  });

  it("is shown in UTC to the whole second, rounded down, 24-hour", () => {
    expect(formatSceneTime(Date.parse("2026-05-21T17:47:18.959Z"))).toBe("2026-05-21 17:47:18");
    expect(formatSceneTime(Date.parse("2026-05-22T00:05:00.000Z"))).toBe("2026-05-22 00:05:00");
    expect(formatSceneTime(Date.parse("2026-05-21T23:59:59.999Z"))).toBe("2026-05-21 23:59:59");
  });
});

describe("the entry time of a commanded flight", () => {
  const KEY = "AAL100_05L_a00001_20260501T000300Z";

  beforeEach(() => { clearEntryCache(); fetchJson.mockReset(); });
  afterEach(() => clearEntryCache());

  const day = (arrivals: Array<{ flightKey: string; entryUtc: string }>) => ({
    airport: "KRDU", date: "2026-05-01",
    arrivals: arrivals.map((a) => ({ callsign: "AAL100", runway: "05L", type: null, landingUtc: a.entryUtc, ...a })) });

  it("reads the day a flight lands on off its key", () => {
    expect(landingDay(KEY)).toBe("2026-05-01");
    expect(() => landingDay("AAL100")).toThrow(/carries no landing time/);
  });

  it("asks the roster of the landing day once per flight and returns the flight's entry", async () => {
    fetchJson.mockResolvedValue(day([{ flightKey: "OTHER_05L_b_20260501T000100Z", entryUtc: "2026-05-01T00:00:00Z" },
                                     { flightKey: KEY, entryUtc: "2026-05-01T00:00:00.901Z" }]));
    expect(await entryUtcOf("KRDU", KEY)).toBe("2026-05-01T00:00:00.901Z");
    expect(await entryUtcOf("KRDU", KEY)).toBe("2026-05-01T00:00:00.901Z");
    expect(fetchJson).toHaveBeenCalledTimes(1);
    expect(String(fetchJson.mock.calls[0][0])).toMatch(/\/traffic\/arrivals\?airport=KRDU&date=2026-05-01$/);
  });

  it("names a flight the roster lacks, and tries again after a failure", async () => {
    fetchJson.mockResolvedValueOnce(day([]));
    await expect(entryUtcOf("KRDU", KEY)).rejects.toThrow(`${KEY} is not in the KRDU arrivals roster`);
    fetchJson.mockResolvedValueOnce(day([{ flightKey: KEY, entryUtc: "2026-05-01T00:00:00.901Z" }]));
    expect(await entryUtcOf("KRDU", KEY)).toBe("2026-05-01T00:00:00.901Z");
    expect(fetchJson).toHaveBeenCalledTimes(2);
  });
});
