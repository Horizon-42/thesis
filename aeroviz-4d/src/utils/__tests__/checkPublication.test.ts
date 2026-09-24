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
  checkTrainingOverlay,
  checkTrainingOverlays,
  checkTrainingSample,
  checkTrainingSetAgrees,
  checkTrainingSetRefusal,
} from "../checkPublication";
import { parseTrainingIndex, parseTrainingSample } from "../../data/trainingSample";
import { parseTrainingOverlays } from "../../data/trainingOverlays";
import { SET_ID, mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";
import {
  MOCK_SAMPLE_SHA, mockExecutorOverlay, mockOverlays, mockPriorOverlay,
} from "../../data/__tests__/trainingOverlays.fixture";

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
  const index = parseTrainingIndex(mockIndex());
  const sample = parseTrainingSample(mockSample());
  if (!index.ok || !sample.ok) throw new Error("the fixture should parse");
  const entry = index.value.sets.find((item) => item.id === SET_ID)!;
  return { index: index.value, entry, sample: sample.value };
}

describe("the Training export's checks", () => {
  it("passes a manifest and a readable sample that agree", () => {
    const { entry, sample } = readable();
    expect(checkTrainingIndex(mockIndex())).toEqual([]);
    expect(checkTrainingSample(SET_ID, mockSample())).toEqual([]);
    expect(checkTrainingSetRefusal(entry)).toEqual([]);
    expect(checkTrainingSetAgrees(entry, sample)).toEqual([]);
  });

  // A set of a superseded vocabulary is refused on purpose: a WARNING naming why, not an error.
  it("warns, by name, about the sets the panel refuses", () => {
    const { index } = readable();
    const findings = index.sets.flatMap(checkTrainingSetRefusal);
    expect(findings.map((finding) => [finding.level, finding.category])).toEqual([
      ["warn", "box_v3"], ["warn", "instruction_v1"], ["warn", "prior_s1337_val"],
    ]);
    expect(findings[0].message).toContain("read under box-v3, a superseded vocabulary");
  });

  it("names the entry and the field when the panel would grey an entry out", () => {
    const index = mockIndex() as any;
    index.sets.push({ ...index.sets[1], id: "half_written", kind: "not-a-kind" });
    const findings = checkTrainingIndex(index);
    expect(findings).toHaveLength(1);
    expect(findings[0].category).toBe("half_written");
    expect(findings[0].message).toContain("not-a-kind");
  });

  it("names the field when a readable sample is wrong", () => {
    const sample = mockSample() as any;
    sample.flights[0].envelopes.speed[0].check.bandInside = 7;
    const findings = checkTrainingSample(SET_ID, sample);
    expect(findings[0].category).toBe(SET_ID);
    expect(findings[0].message).toContain("says 7 band rows inside");
  });

  // The two files come out of ONE run of the exporter; every field they must agree on is checked.
  it("catches each field the manifest and the sample must agree on", () => {
    const { entry, sample } = readable();
    for (const [field, value] of [
      ["vocabularySha256", "0".repeat(64)],
      ["runwaySha256", "0000"],
      ["readingRule", "plateau-v9"],
      ["flights", 40],
      ["id", "another_set"],
    ] as const) {
      const findings = checkTrainingSetAgrees({ ...entry, [field]: value }, sample);
      expect(findings, field).toHaveLength(1);
    }
    const reseeded = checkTrainingSetAgrees({ ...entry, cohort: { ...entry.cohort, seed: 7 } }, sample);
    expect(reseeded[0].message).toContain("cohort.seed");
  });
});

describe("the Training overlays' checks", () => {
  function entries() {
    const parsed = parseTrainingOverlays(mockOverlays());
    if (!parsed.ok) throw new Error(parsed.problem);
    return parsed.value.overlays;
  }

  function sample() {
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    return parsed.value;
  }

  it("passes both overlays read against the sample whose sha they recorded", () => {
    const [executor, prior] = entries();
    expect(checkTrainingOverlays(mockOverlays())).toEqual([]);
    expect(checkTrainingOverlay(executor, mockExecutorOverlay(), sample(), MOCK_SAMPLE_SHA)).toEqual([]);
    expect(checkTrainingOverlay(prior, mockPriorOverlay(), sample(), MOCK_SAMPLE_SHA)).toEqual([]);
  });

  it("names an overlay the panel would reject", () => {
    const raw: any = mockOverlays();
    raw.overlays[1].flights = -1;
    expect(checkTrainingOverlays(raw)).toEqual([expect.objectContaining({ level: "error", category: "prior_test" })]);
  });

  it("is an error when the set's sample on disk is not the one the overlay was drawn over", () => {
    const [executor] = entries();
    const findings = checkTrainingOverlay(executor, mockExecutorOverlay(), sample(), "6".repeat(64));
    expect(findings).toEqual([expect.objectContaining({ level: "error", category: "executor_test" })]);
    expect(findings[0].message).toMatch(/the file on disk is 666666666666/);
  });

  it("names the field when an overlay does not read, and a flight count the manifest does not match", () => {
    const [executor, prior] = entries();
    const broken: any = mockPriorOverlay();
    broken.flights[0].columns[3].truthP[0] = -0.1;
    expect(checkTrainingOverlay(prior, broken, sample(), MOCK_SAMPLE_SHA)[0].message).toMatch(/truthP\[0\] is -0.1/);
    expect(checkTrainingOverlay({ ...executor, flights: 40 }, mockExecutorOverlay(), sample(), MOCK_SAMPLE_SHA)[0].message)
      .toMatch(/the manifest says 40 flights/);
  });
});
