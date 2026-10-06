import { describe, expect, it } from "vitest";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";
import {
  buildComparisonLegend,
  comparisonKindLabel,
  recordedTrafficLabel,
} from "../comparisonLegend";

function group(
  groupId: string,
  runway: string,
  status: ComparisonGroup["status"],
  entities: string[],
): ComparisonGroup {
  return {
    group: groupId,
    flightId: groupId,
    runway,
    airport: "KRDU",
    status,
    finalTimeS: status === "failed" ? null : 120,
    initialState: null,
    entities,
    czml: `${runway}.czml`,
  };
}

function index(groups: ComparisonGroup[]): ComparisonIndex {
  return {
    schemaVersion: "comparison-v2-generation",
    generation: "test",
    epoch: "2026-07-01T00:00:00Z",
    startHidden: true,
    referenceSource: "canonicalObserved",
    groups,
    evaluationReport: "evaluation_report.test.json",
  };
}

describe("buildComparisonLegend", () => {
  it("names a pass on another runway apart from the same-runway pass", () => {
    const result = buildComparisonLegend(index([
      group("pass", "05L", "solved", ["ref-pass", "look-pass", "pred-pass"]),
      group("other", "05L", "otherRunway", ["ref-other", "look-other", "pred-other"]),
    ]), null);
    expect(result.statuses).toEqual(["predictionPass", "predictionOtherRunway"]);
  });

  it("describes only the displayed optimizer-category paths and all verdict colours", () => {
    const result = buildComparisonLegend(index([
      group("solved", "05L", "solved", ["ref-solved", "opt-solved", "sim-solved"]),
      group("miss", "05L", "offTarget", ["ref-miss", "opt-miss", "sim-miss"]),
      group("failed", "05L", "failed", ["ref-failed"]),
    ]), null);

    expect(result.kinds).toEqual(["reference", "simulator"]);
    expect(result.statuses).toEqual(["offTargetResult"]);
    expect(result.kinds).not.toContain("optimizer");
  });

  it("uses ground-truth-style prediction outcome colours without recolouring references", () => {
    const result = buildComparisonLegend(index([
      group("pass", "05L", "solved", [
        "ref-pass",
        "look-pass",
        "pred-pass",
      ]),
      group("forecast", "05L", "offTarget", [
        "ref-forecast",
        "look-forecast",
        "pred-forecast",
      ]),
      group("unknown", "05L", "indeterminate", [
        "ref-unknown",
        "look-unknown",
        "pred-unknown",
      ]),
    ]), null);

    expect(result.kinds).toEqual(["reference", "predicted", "lookback"]);
    expect(result.statuses).toEqual([
      "predictionPass",
      "predictionFail",
      "predictionIndeterminate",
    ]);
  });

  it("limits the legend to the selected runway", () => {
    const result = buildComparisonLegend(index([
      group("optimized", "05L", "solved", ["ref-optimized", "opt-optimized", "sim-optimized"]),
      group("forecast", "23R", "solved", ["ref-forecast", "look-forecast", "pred-forecast"]),
    ]), "23R");

    expect(result.kinds).toEqual(["reference", "predicted", "lookback"]);
    expect(result.statuses).toEqual(["predictionPass"]);
  });
});

describe("the legend of a traffic category", () => {
  const traffic = { outcome: "separated", recorded: ["ref-x"], startOffsetsS: [1] };
  const window = (id: string, runway = "05L"): ComparisonGroup => ({
    ...group(id, runway, "solved", [`ref-${id}`, `sim-${id}`]), traffic,
  });

  it("is not a traffic legend for a category without traffic", () => {
    expect(buildComparisonLegend(index([group("a", "05L", "solved", ["ref-a", "sim-a"])]), null).traffic).toBeNull();
  });

  it("names the windows of an M1 category", () => {
    const legend = buildComparisonLegend(index([window("a"), window("b", "05R")]), null);
    expect(legend.traffic).toBe("windows");
    expect(legend.kinds).toEqual(["reference", "simulator"]);
    expect(legend.statuses).toEqual([]);
  });

  it("names the scene of an M2 category, and draws it whole whatever the runway selector says", () => {
    const sceneGroup = (id: string, runway: string): ComparisonGroup => ({
      ...group(id, runway, "solved", [`ref-${id}`, `sim-${id}`]),
      scene: { startOffsetS: 0, outcome: "separated", delayS: 0 },
    });
    const sceneIndex = {
      ...index([sceneGroup("a", "05L"), sceneGroup("b", "05R")]),
      scene: { startUtc: "2026-05-21T17:47:18.959Z", background: { recorded: [], startOffsetsS: [] } },
    };
    const legend = buildComparisonLegend(sceneIndex, "23R");              // no group lands on 23R: the scene shows all
    expect(legend.traffic).toBe("scene");
    expect(legend.kinds).toEqual(["reference", "simulator"]);
  });

  it("calls the reference the controlled aircraft's record and the result its optimized path", () => {
    expect(comparisonKindLabel("reference", "windows")).toBe("Controlled aircraft — its record");
    expect(comparisonKindLabel("simulator", "windows")).toBe("Controlled aircraft — optimized path");
    expect(comparisonKindLabel("reference", "scene")).toBe("Controlled aircraft — its record");
    expect(comparisonKindLabel("simulator", "scene")).toBe("Controlled aircraft — optimized path");
  });

  it("keeps the other categories' names", () => {
    expect(comparisonKindLabel("reference", null)).toBe("Reference");
    expect(comparisonKindLabel("simulator", null)).toBe("Optimize results");
    expect(comparisonKindLabel("predicted", null)).toBe("Predicted");
    expect(comparisonKindLabel("lookback", null)).toBe("Predictor input");
  });

  it("names the pink aircraft: recorded, not controlled, and in a scene the ones outside the scheduled set", () => {
    expect(recordedTrafficLabel("windows")).toBe("Recorded traffic — not controlled");
    expect(recordedTrafficLabel("scene")).toBe("Recorded traffic — not controlled, outside the scheduled set");
  });
});
