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
  checkTrainingSample,
  checkTrainingSetAgrees,
} from "../checkPublication";
import { parseTrainingIndex, parseTrainingSample } from "../../data/trainingSample";
import { mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";

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

// ── the Training export (T8) ─────────────────────────────────────────────────

describe("the Training export's checks", () => {
  it("passes a manifest and a sample that agree", () => {
    expect(checkTrainingIndex(mockIndex())).toEqual([]);
    expect(checkTrainingSample("box_v3", mockSample(), "box-v2-wedge", "vocabulary-readback")).toEqual([]);

    const index = parseTrainingIndex(mockIndex());
    const sample = parseTrainingSample(mockSample(), "vocabulary-readback");
    if (!index.ok || !sample.ok) throw new Error("the fixture should parse");
    expect(checkTrainingSetAgrees(index.value.sets[0], sample.value)).toEqual([]);
  });

  // The acceptance test of T8: break one field, get the set id and the field back.
  it("names the set and the field when the panel would grey a set out", () => {
    const index = mockIndex() as any;
    index.sets.push({ ...index.sets[0], id: "half_written", kind: "not-a-kind" });

    const findings = checkTrainingIndex(index);
    expect(findings).toHaveLength(1);
    expect(findings[0].category).toBe("half_written");
    expect(findings[0].message).toContain("kind");
    expect(findings[0].message).toContain("not-a-kind");
  });

  it("names the field when a sample is wrong", () => {
    const sample = mockSample() as any;
    sample.flights[0].sentence.words[0] = [0, 0, 0, 0, 0, 0, 0];
    const findings = checkTrainingSample("box_v3", sample, "box-v2-wedge", "vocabulary-readback");
    expect(findings[0].category).toBe("box_v3");
    expect(findings[0].message).toContain("7 columns, expected 6");
  });

  // The commonest failure now: an export read under a retired rule. Its rows can
  // all still parse — six columns, every word in range — so the message has to
  // carry the rule that produced the file, or the reader looks broken rather than
  // the file stale.
  it("names the reading rule a refused sample was written under", () => {
    const stale = mockSample() as any;
    stale.vocabulary.readingRule = "segment-v14";
    const findings = checkTrainingSample("box_v3", stale, "segment-v14", "vocabulary-readback");
    expect(findings[0].message).toContain("read under segment-v14");
    expect(findings[0].message).toContain("box-v3");
  });

  // The two files come out of ONE run of the exporter. A disagreement means they did not,
  // and averaging over it would show a sample from a vocabulary nobody asked about.
  it("catches a manifest and a sample from different exports", () => {
    const index = parseTrainingIndex(mockIndex());
    const raw = mockSample() as any;
    raw.vocabulary.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    const sample = parseTrainingSample(raw, "vocabulary-readback");
    if (!index.ok || !sample.ok) throw new Error("the fixture should parse");

    const findings = checkTrainingSetAgrees(index.value.sets[0], sample.value);
    expect(findings).toHaveLength(1);
    expect(findings[0].category).toBe("box_v3");
    expect(findings[0].message).toContain("vocabularySha256");
  });

  // Every row of the agreement loop, not just the sha: deleting the runway-sha or
  // the reading-rule row passed the whole suite before this.
  it("catches each field the two files must agree on", () => {
    const index = parseTrainingIndex(mockIndex());
    const sample = parseTrainingSample(mockSample(), "vocabulary-readback");
    if (!index.ok || !sample.ok) throw new Error("the fixture should parse");

    for (const [field, value] of [
      ["runwaySha256", "0000"],
      ["readingRule", "plateau-v9"],
    ] as const) {
      const findings = checkTrainingSetAgrees(
        { ...index.value.sets[0], [field]: value },
        sample.value,
      );
      expect(findings).toHaveLength(1);
      expect(findings[0].message).toContain(field);
    }

    // and the sample calling itself by another name
    const renamed = checkTrainingSetAgrees(index.value.sets[0], {
      ...sample.value,
      setId: "another_set",
    });
    expect(renamed[0].message).toContain("another_set");
  });

  it("catches a manifest that promises more flights than the sample holds", () => {
    const index = parseTrainingIndex(mockIndex());
    const sample = parseTrainingSample(mockSample(), "vocabulary-readback");
    if (!index.ok || !sample.ok) throw new Error("the fixture should parse");

    const findings = checkTrainingSetAgrees({ ...index.value.sets[0], flights: 40 }, sample.value);
    expect(findings[0].message).toContain("lists 40 flights");
    expect(findings[0].message).toContain("holds 1");
  });
});
