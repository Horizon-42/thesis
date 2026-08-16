import { describe, expect, it } from "vitest";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";
import { buildComparisonLegend } from "../comparisonLegend";

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

  it("uses baseline-style prediction outcome colours without recolouring references", () => {
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
