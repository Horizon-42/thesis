import { describe, expect, it } from "vitest";
import { sampleSubset, selectComparisonGroups } from "../sampleTrajectories";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";

/** Deterministic RNG (cycles through fixed fractions) so sampling is reproducible in tests. */
function seededRng(seed = 0.42): () => number {
  let x = seed;
  return () => {
    x = (x * 9301 + 49297) % 233280;
    return x / 233280;
  };
}

describe("sampleSubset", () => {
  it("returns all items (a copy) when count <= 0 or >= length", () => {
    const items = [1, 2, 3];
    expect(sampleSubset(items, 0)).toEqual([1, 2, 3]);
    expect(sampleSubset(items, -5)).toEqual([1, 2, 3]);
    expect(sampleSubset(items, 3)).toEqual([1, 2, 3]);
    expect(sampleSubset(items, 99)).toEqual([1, 2, 3]);
    expect(sampleSubset(items, 0)).not.toBe(items); // copy, not the same array
  });

  it("returns exactly `count` distinct items drawn from the input", () => {
    const items = ["a", "b", "c", "d", "e"];
    const picked = sampleSubset(items, 3, seededRng());
    expect(picked).toHaveLength(3);
    expect(new Set(picked).size).toBe(3); // no duplicates
    expect(picked.every((p) => items.includes(p))).toBe(true);
  });

  it("does not mutate the input array", () => {
    const items = [1, 2, 3, 4];
    sampleSubset(items, 2, seededRng());
    expect(items).toEqual([1, 2, 3, 4]);
  });
});

function group(over: Partial<ComparisonGroup>): ComparisonGroup {
  return {
    group: "X_05L",
    flightId: "X",
    runway: "05L",
    airport: "KRDU",
    status: "solved",
    finalTimeS: 100,
    initialState: null,
    entities: ["ref-X_05L", "opt-X_05L", "sim-X_05L"],
    czml: "comparison_KRDU_05L.czml",
    ...over,
  };
}

const INDEX: ComparisonIndex = {
  schemaVersion: "comparison-v2-generation",
  generation: "batch123",
  epoch: "2026-04-01T08:00:00+00:00",
  startHidden: true,
  referenceSource: "canonicalObserved",
  evaluationReport: "evaluation_report_batch123.json",
  groups: [
    group({ group: "A_05L", flightId: "A", runway: "05L", entities: ["ref-A_05L", "opt-A_05L", "sim-A_05L"] }),
    group({ group: "B_05L", flightId: "B", runway: "05L", entities: ["ref-B_05L", "opt-B_05L", "sim-B_05L"] }),
    group({ group: "C_23R", flightId: "C", runway: "23R", czml: "comparison_KRDU_23R.czml",
      entities: ["ref-C_23R", "opt-C_23R", "sim-C_23R"] }),
    group({ group: "D_23R", flightId: "D", runway: "23R", status: "failed", czml: "comparison_KRDU_23R.czml",
      entities: ["ref-D_23R"] }),
    group({ group: "E_05L", flightId: "E", runway: "05L", entities: [] }), // no entities -> ineligible
  ],
};

describe("selectComparisonGroups", () => {
  it("filters to a single runway and excludes empty groups", () => {
    const sel = selectComparisonGroups(INDEX, "05L", 0, seededRng());
    expect(sel.groups.map((g) => g.group).sort()).toEqual(["A_05L", "B_05L"]); // not E (empty), not 23R
    expect(sel.files).toEqual(["comparison_KRDU_05L.czml"]);
    expect(sel.shownEntityIds.has("opt-A_05L")).toBe(true);
    expect(sel.shownEntityIds.has("ref-D_23R")).toBe(false); // different runway
  });

  it("null runway spans all runways and includes failed (dark-red) groups", () => {
    const sel = selectComparisonGroups(INDEX, null, 0, seededRng());
    expect(sel.groups.map((g) => g.group).sort()).toEqual(["A_05L", "B_05L", "C_23R", "D_23R"]);
    expect(sel.files.sort()).toEqual(["comparison_KRDU_05L.czml", "comparison_KRDU_23R.czml"]);
    // the failed group's dark-red reference is revealed (no opt/sim for it)
    expect(sel.shownEntityIds.has("ref-D_23R")).toBe(true);
  });

  it("samples `count` groups and only lists the files those groups live in", () => {
    const sel = selectComparisonGroups(INDEX, null, 1, seededRng());
    expect(sel.groups).toHaveLength(1);
    expect(sel.files).toHaveLength(1);
    expect(sel.files[0]).toBe(sel.groups[0].czml);
    // every revealed id belongs to the sampled group
    expect([...sel.shownEntityIds].sort()).toEqual([...sel.groups[0].entities].sort());
  });
});

describe("selectComparisonGroups for a scene (an M2 run)", () => {
  const scene = { startUtc: "2026-05-21T17:47:18.959Z", background: { recorded: [], startOffsetsS: [] } };
  const groupScene = { startOffsetS: 0, outcome: "separated", delayS: 0 };
  const sceneIndex: ComparisonIndex = {
    ...INDEX,
    scene,
    groups: [
      group({ group: "A_05L", runway: "05L", scene: groupScene }),
      group({ group: "B_05R", runway: "05R", scene: groupScene, czml: "comparison_KRDU_05R.czml" }),
      group({ group: "C_05L", runway: "05L", scene: groupScene }),
      group({ group: "D_05R", runway: "05R", scene: groupScene, status: "failed", entities: ["ref-D_05R"],
              czml: "comparison_KRDU_05R.czml" }),
    ],
  };

  it("shows every group: the sample count and the runway selector cut nothing", () => {
    for (const [runway, count] of [[null, 1], ["05L", 1], ["05R", 2], [null, 0]] as const) {
      const selection = selectComparisonGroups(sceneIndex, runway, count, seededRng());
      expect(selection.groups.map((g) => g.group)).toEqual(["A_05L", "B_05R", "C_05L", "D_05R"]);
    }
    expect(selectComparisonGroups(sceneIndex, null, 1).files.sort())
      .toEqual(["comparison_KRDU_05L.czml", "comparison_KRDU_05R.czml"]);
  });

  it("still samples and filters an index without a scene", () => {
    const plain = { ...sceneIndex, scene: undefined };
    expect(selectComparisonGroups(plain, null, 1, seededRng()).groups).toHaveLength(1);
    expect(selectComparisonGroups(plain, "05R", 0).groups.map((g) => g.group)).toEqual(["B_05R", "D_05R"]);
  });
});
