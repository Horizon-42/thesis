import { describe, expect, it } from "vitest";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";
import {
  comparisonCzmlUrl,
  comparisonIndexUrl,
  comparisonSourceKey,
  comparisonSourceOf,
  isWindowIndex,
  requestedWindow,
  type ComparisonSourceInputs,
} from "../comparisonSource";

const BASE = "http://backend:8765/traffic/jobs/20261006T120000123456Z-0123abcd/files/";
const inputs = (over: Partial<ComparisonSourceInputs> = {}): ComparisonSourceInputs => ({
  mode: "evaluation",
  trajectoryComparison: true,
  trajectoryComparisonCategory: "traffic_m1_runway",
  activeAirportCode: "KRDU",
  trafficScene: null,
  ...over,
});
const scene = { airportCode: "KRDU", jobId: "20261006T120000123456Z-0123abcd", baseUrl: BASE };

describe("comparisonSourceOf", () => {
  it("is the selected published category in the Evaluate task, while the comparison is on", () => {
    expect(comparisonSourceOf(inputs())).toEqual({ kind: "category", airportCode: "KRDU", categoryDir: "traffic_m1_runway" });
    expect(comparisonSourceOf(inputs({ trajectoryComparison: false }))).toBeNull();
    expect(comparisonSourceOf(inputs({ trajectoryComparisonCategory: null }))).toBeNull();
    expect(comparisonSourceOf(inputs({ trafficScene: scene }))?.kind).toBe("category");      // a job is for Optimize
  });

  it("is the traffic job's scene in the Optimize task, whatever Evaluate's switches say", () => {
    const optimize = inputs({ mode: "optimize", trajectoryComparison: false, trajectoryComparisonCategory: null, trafficScene: scene });
    expect(comparisonSourceOf(optimize)).toEqual({ kind: "job", airportCode: "KRDU", jobId: scene.jobId, baseUrl: BASE });
    expect(comparisonSourceOf({ ...optimize, trajectoryComparison: true, trajectoryComparisonCategory: "x" })?.kind).toBe("job");
  });

  it("is nothing in the Optimize task without a job, for another airport's job, or in another task", () => {
    expect(comparisonSourceOf(inputs({ mode: "optimize" }))).toBeNull();                      // the category is Evaluate's
    expect(comparisonSourceOf(inputs({ mode: "optimize", trafficScene: { ...scene, airportCode: "KSMF" } }))).toBeNull();
    expect(comparisonSourceOf(inputs({ mode: "fly", trafficScene: scene }))).toBeNull();
    expect(comparisonSourceOf(inputs({ mode: "training", trafficScene: scene }))).toBeNull();
    expect(comparisonSourceOf(inputs({ activeAirportCode: "" }))).toBeNull();
  });
});

describe("the URLs and the key of a source", () => {
  const category = { kind: "category", airportCode: "KRDU", categoryDir: "traffic_m1_runway" } as const;
  const job = { kind: "job", airportCode: "KRDU", jobId: scene.jobId, baseUrl: BASE } as const;

  it("read a published category from public/data and a job from the backend's file route", () => {
    expect(comparisonIndexUrl(category)).toBe("/data/airports/KRDU/comparison/traffic_m1_runway/comparison_index.json");
    expect(comparisonCzmlUrl(category, "a.czml")).toBe("/data/airports/KRDU/comparison/traffic_m1_runway/a.czml");
    expect(comparisonIndexUrl(job)).toBe(`${BASE}comparison_index.json`);
    expect(comparisonCzmlUrl(job, "a.czml")).toBe(`${BASE}a.czml`);
  });

  it("differ between a category and a job, and between two jobs", () => {
    expect(comparisonSourceKey(category)).toBe("traffic_m1_runway");
    expect(comparisonSourceKey(job)).toBe(`job:${scene.jobId}`);
    expect(comparisonSourceKey({ ...job, jobId: "other" })).not.toBe(comparisonSourceKey(job));
  });
});

function group(key: string, extra: Partial<ComparisonGroup> = {}): ComparisonGroup {
  return {
    group: key, flightId: key, runway: "05L", airport: "KRDU", status: "solved", finalTimeS: 1, initialState: null,
    entities: [`ref-${key}`], czml: "a.czml", ...extra,
  };
}
const indexOf = (groups: ComparisonGroup[], over: Partial<ComparisonIndex> = {}): ComparisonIndex => ({
  schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
  referenceSource: "canonicalObserved", evaluationReport: "r.json", groups, ...over,
});
const traffic = { outcome: "separated", recorded: ["ref-x"], startOffsetsS: [1] };

describe("isWindowIndex", () => {
  it("is an index of M1 windows: a traffic group, no scene", () => {
    expect(isWindowIndex(indexOf([group("A", { traffic })]))).toBe(true);
    expect(isWindowIndex(indexOf([group("A")]))).toBe(false);                                // a plain category
    const sceneIndex = indexOf([group("A", { scene: { startOffsetS: 0, outcome: "separated", delayS: 0 } })],
      { scene: { startUtc: "2026-05-21T17:47:18.959Z", background: { recorded: [], startOffsetsS: [] } } });
    expect(isWindowIndex(sceneIndex)).toBe(false);                                           // an M2 scene is drawn whole
  });
});

describe("requestedWindow", () => {
  it("is the selected flight's group, and nothing for a selection that is none (Reset view) or no group's", () => {
    expect(requestedWindow(["A", "B", "C"], "B")).toBe("B");
    expect(requestedWindow(["A", "B", "C"], null)).toBeNull();
    expect(requestedWindow(["A", "B", "C"], "Z")).toBeNull();
    expect(requestedWindow([], "A")).toBeNull();
  });
});
