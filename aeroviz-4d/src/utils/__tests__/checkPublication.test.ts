import { describe, expect, it } from "vitest";

import {
  COMPARISON_INDEX_SCHEMA_VERSION,
  EXPERIMENT_PREDICTION_OUTPUTS,
  type ComparisonCategory,
} from "../../data/airportData";
import {
  checkCategoriesManifest,
  checkComparisonIndex,
  explainCategoryRejection,
  indexCzmlFiles,
} from "../checkPublication";

const observed: ComparisonCategory = {
  key: "observed",
  label: "Observed ADS-B",
  dir: "observed",
  groups: 2,
  constrained: false,
};

function experimentCategory(overrides: Record<string, unknown> = {}) {
  return {
    key: "experiment_plan_val",
    label: "plan · validation",
    dir: "experiment_plan_val",
    groups: 3,
    constrained: false,
    datasetSplit: "val",
    resultSource: "experiment",
    experiment: {
      id: "campaign/plan_seed1337",
      group: "campaign",
      checkpoint: "campaign/plan_seed1337/checkpoint.pt",
      model: "itransformer",
      predictionOutput: EXPERIMENT_PREDICTION_OUTPUTS[EXPERIMENT_PREDICTION_OUTPUTS.length - 1],
      horizonMode: "normalized",
      seed: 1337,
    },
    ...overrides,
  };
}

function index(groups: Array<Partial<{ czml: string }>>) {
  return {
    schemaVersion: COMPARISON_INDEX_SCHEMA_VERSION,
    generation: "abc",
    epoch: "2026-09-12T00:00:00Z",
    startHidden: false,
    referenceSource: "canonicalObserved",
    datasetSplit: "val",
    evaluationReport: "evaluation_report_abc.json",
    groups: groups.map((group, position) => ({
      group: `f${position}_23L`,
      flightId: `f${position}`,
      runway: "23L",
      airport: "KRDU",
      status: "solved",
      finalTimeS: 300,
      initialState: null,
      czml: group.czml ?? "comparison_KRDU_23L_abc.czml",
      entities: ["a", "b"],
    })),
  };
}

describe("checkCategoriesManifest", () => {
  it("accepts a manifest the picker accepts", () => {
    expect(checkCategoriesManifest({ categories: [observed, experimentCategory()] })).toEqual([]);
  });

  it("names the category and the field when an unlisted prediction output would empty the picker", () => {
    const manifest = {
      categories: [
        observed,
        experimentCategory({
          experiment: { ...experimentCategory().experiment, predictionOutput: "not-an-output" },
        }),
      ],
    };
    const findings = checkCategoriesManifest(manifest);
    expect(findings).toHaveLength(1);
    expect(findings[0].level).toBe("error");
    expect(findings[0].category).toBe("experiment_plan_val");
    expect(findings[0].message).toContain("predictionOutput");
    expect(findings[0].message).toContain("not-an-output");
    expect(findings[0].message).toContain("NOTHING");
  });

  it("explains the field that fails, and nothing when the guard accepts", () => {
    const meta = experimentCategory().experiment;
    expect(explainCategoryRejection(experimentCategory({ experiment: { ...meta, horizonMode: "later" } })))
      .toContain("horizonMode");
    expect(explainCategoryRejection(experimentCategory({ experiment: undefined })))
      .toContain("`experiment` is absent");
    expect(explainCategoryRejection(experimentCategory({ groups: "3" }))).toContain("`groups`");
    expect(explainCategoryRejection([])).toContain("not an object");
    expect(explainCategoryRejection(experimentCategory({ experiment: [] }))).toContain("`experiment` must be an object");
    expect(explainCategoryRejection(experimentCategory())).toBeNull();
  });

  it("names a malformed intent, parameter row or run name", () => {
    const meta = experimentCategory().experiment;
    expect(explainCategoryRejection(experimentCategory({
      experiment: { ...meta, intent: { groupTitle: "t", group: "q" } },
    }))).toContain("`experiment.intent`");
    expect(explainCategoryRejection(experimentCategory({
      experiment: {
        ...meta,
        parameters: [{ section: "Model", name: "Output", value: "control" }, { section: "Model" }],
      },
    }))).toContain("`experiment.parameters[1]`");
    expect(explainCategoryRejection(experimentCategory({ experiment: { ...meta, runName: 3 } })))
      .toContain("`experiment.runName`");
  });

  it("reports a manifest whose categories are not an array", () => {
    const findings = checkCategoriesManifest({ categories: "nope" });
    expect(findings).toHaveLength(1);
    expect(findings[0].level).toBe("error");
  });
});

describe("checkComparisonIndex", () => {
  it("accepts an index that matches its manifest row", () => {
    const category = { ...observed, groups: 2, datasetSplit: "val" as const };
    expect(checkComparisonIndex(category, index([{}, {}]))).toEqual([]);
  });

  it("rejects an index the picker rejects", () => {
    const findings = checkComparisonIndex(observed, { schemaVersion: "comparison-v1" });
    expect(findings).toHaveLength(1);
    expect(findings[0].level).toBe("error");
    expect(findings[0].message).toContain("isComparisonIndex");
  });

  it("is an error when the manifest promises groups and the index holds none", () => {
    const findings = checkComparisonIndex({ ...observed, groups: 4 }, index([]));
    expect(findings).toHaveLength(1);
    expect(findings[0].level).toBe("error");
    expect(findings[0].message).toContain("nothing to draw");
  });

  it("warns when the manifest's group count or split disagrees with the index", () => {
    const category = { ...observed, groups: 5, datasetSplit: "train" as const };
    const messages = checkComparisonIndex(category, index([{}, {}])).map((finding) => finding.message);
    expect(messages.some((message) => message.includes("5 groups"))).toBe(true);
    expect(messages.some((message) => message.includes("split"))).toBe(true);
  });

  it("lists each CZML file once", () => {
    const value = index([{ czml: "a.czml" }, { czml: "a.czml" }, { czml: "b.czml" }]);
    expect(indexCzmlFiles(value as never)).toEqual(["a.czml", "b.czml"]);
  });
});
