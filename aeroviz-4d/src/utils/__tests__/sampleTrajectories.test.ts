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
  epoch: "2026-04-01T08:00:00+00:00",
  startHidden: true,
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
