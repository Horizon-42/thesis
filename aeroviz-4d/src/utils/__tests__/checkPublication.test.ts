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
  checkTrainingIndex,
  checkTrainingSet,
  checkTrainingSetAgrees,
} from "../checkPublication";
import { parseTrainingIndex, parseTrainingSample, type TrainingSetEntry } from "../../data/trainingSample";
import { SET_ID, stageAIndex, stageASampleFile } from "../../data/__tests__/stageA";

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

// ── the Training export ──────────────────────────────────────────────────────

function readable() {
  const index = parseTrainingIndex(stageAIndex());
  const sample = parseTrainingSample(stageASampleFile());
  if (!index.ok || !sample.ok) throw new Error("the fixture should parse");
  return { index: index.value, entry: index.value.sets.find((item) => item.id === SET_ID)!, sample: sample.value };
}

describe("the Training export's checks", () => {
  it("passes an index and a readable sample that agree", () => {
    const { entry, sample } = readable();
    expect(checkTrainingIndex(stageAIndex()).findings).toEqual([]);
    expect(checkTrainingSet(SET_ID, stageASampleFile()).findings).toEqual([]);
    expect(checkTrainingSetAgrees(entry, sample, "KXXX")).toEqual([]);
  });

  it("refuses an index of another schema whole, naming the one found and the one expected", () => {
    const { findings, value } = checkTrainingIndex({ ...stageAIndex(), schema: "aeroviz-training-index-v1" });
    expect(value).toBeNull();
    expect(findings).toEqual([expect.objectContaining({ level: "error" })]);
    expect(findings[0].message).toContain("training/index_v4.json is not an index");
    expect(findings[0].message).toContain('schema is "aeroviz-training-index-v1", expected "aeroviz-training-index-v2"');
  });

  it("names the entry and the field when the panel would reject an entry", () => {
    const index = stageAIndex();
    index.sets.push({ ...index.sets[0], id: "half_written", kind: "not-a-kind" });
    const { findings } = checkTrainingIndex(index);
    expect(findings).toHaveLength(1);
    expect(findings[0].category).toBe("half_written");
    expect(findings[0].message).toContain("not-a-kind");
  });

  it("names the field when a sample is wrong, and the schema when it is another one", () => {
    const sample = stageASampleFile();
    sample.flights[0].closedLoop["2"].flownFromRow += 1;
    const { findings, value } = checkTrainingSet(SET_ID, sample);
    expect(value).toBeNull();
    expect(findings[0].category).toBe(SET_ID);
    expect(findings[0].message).toContain("flownFromRow");
    const old = checkTrainingSet(SET_ID, { ...stageASampleFile(), schema: "aeroviz-training-sample-v8" });
    expect(old.findings[0].message).toContain('sample.schema is "aeroviz-training-sample-v8", not one of aeroviz-training-sample-v9');
  });

  // The two files come out of ONE run of the exporter; every field they must agree on is checked.
  it("catches each field the index and the sample must agree on", () => {
    const { entry, sample } = readable();
    for (const change of [
      { flights: 40 },
      { id: "another_set" },
      { source: { ...entry.source, specSha256: "0".repeat(64) } },
      { source: { ...entry.source, executorSpecSha256: "0000" } },
      { cohort: { ...entry.cohort, perStratum: 7 } },
      { cohort: { ...entry.cohort, seed: 7 } },
    ]) {
      const findings = checkTrainingSetAgrees({ ...entry, ...change } as TrainingSetEntry, sample, "KXXX");
      expect(findings, JSON.stringify(Object.keys(change))).toHaveLength(1);
    }
    // a sample filed under another airport's directory
    expect(checkTrainingSetAgrees(entry, sample, "KYYY")[0].message).toContain("airport: the index says KYYY");
  });
});
