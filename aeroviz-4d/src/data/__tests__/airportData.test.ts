import { describe, expect, it } from "vitest";

import {
  AIRPORTS_INDEX_URL,
  EXPERIMENT_PREDICTION_OUTPUTS,
  airportDataUrl,
  airportProcedureDetailUrl,
  airportProcedureDetailsIndexUrl,
  airportLocalTerrainUrl,
  airportChartsIndexUrl,
  isAirportsIndexManifest,
  isComparisonCategoriesManifest,
  isComparisonCategory,
  isDrawableComparisonCategory,
  isComparisonIndex,
  normalizeAirportCode,
  sortAirportCatalog,
} from "../airportData";

describe("airportData helpers", () => {
  it("builds airport-scoped data URLs", () => {
    expect(airportDataUrl("krdu", "airport.json")).toBe("/data/airports/KRDU/airport.json");
    expect(airportLocalTerrainUrl("cyvr", "metadata.json")).toBe(
      "/data/airports/CYVR/local-terrain/heightmap/metadata.json",
    );
    expect(airportProcedureDetailsIndexUrl("krdu")).toBe(
      "/data/airports/KRDU/procedure-details/index.json",
    );
    expect(airportProcedureDetailUrl("krdu", "KRDU-R05LY-RW05L")).toBe(
      "/data/airports/KRDU/procedure-details/KRDU-R05LY-RW05L.json",
    );
    expect(airportChartsIndexUrl("krdu")).toBe("/data/airports/KRDU/charts/index.json");
  });

  it("validates and sorts the airport manifest", () => {
    const manifest = {
      defaultAirport: "krdu",
      airports: [
        { code: "CYVR", name: "Vancouver", lat: 49.1, lon: -123.1 },
        { code: "KRDU", name: "Raleigh-Durham", lat: 35.8, lon: -78.7 },
      ],
    };

    expect(AIRPORTS_INDEX_URL).toBe("/data/airports/index.json");
    expect(isAirportsIndexManifest(manifest)).toBe(true);
    expect(normalizeAirportCode(manifest.defaultAirport)).toBe("KRDU");
    expect(sortAirportCatalog(manifest.airports).map((airport) => airport.code)).toEqual([
      "CYVR",
      "KRDU",
    ]);
  });

  it("requires the explicit constrained boolean on every comparison category", () => {
    const entry = { key: "runway_cons", label: "Runway (constrained)", dir: "runway_cons", groups: 3 };
    // Constrained-ness is a manifest FIELD, not a key/dir spelling — an entry
    // without the boolean is rejected so a stale manifest fails loudly.
    expect(isComparisonCategoriesManifest({ categories: [entry] })).toBe(false);
    expect(
      isComparisonCategoriesManifest({ categories: [{ ...entry, constrained: true }] }),
    ).toBe(true);
    expect(
      isComparisonCategoriesManifest({ categories: [{ ...entry, constrained: "yes" }] }),
    ).toBe(false);
  });

  it("accepts only explicit train/validation/test dataset split labels", () => {
    const entry = {
      key: "ts_model_train",
      label: "Training split",
      dir: "ts_model_train",
      groups: 3,
      constrained: false,
    };
    expect(isComparisonCategoriesManifest({ categories: [{ ...entry, datasetSplit: "train" }] }))
      .toBe(true);
    expect(isComparisonCategoriesManifest({ categories: [{ ...entry, datasetSplit: "training" }] }))
      .toBe(false);
    // a day partition's validation days (runway-intent R3's schedule), not the checkpoint's val
    expect(isComparisonCategoriesManifest({ categories: [{ ...entry, datasetSplit: "dayval" }] }))
      .toBe(true);
  });

  it("accepts experiment categories only with explicit checkpoint metadata", () => {
    const entry = {
      key: "experiment_run_val",
      label: "Experiment run · validation",
      dir: "experiment_run_val",
      groups: 3,
      constrained: false,
      datasetSplit: "val",
      resultSource: "experiment",
    };
    expect(isComparisonCategoriesManifest({ categories: [entry] })).toBe(false);
    expect(isComparisonCategoriesManifest({ categories: [{
      ...entry,
      experiment: {
        id: "campaign/stage/run",
        group: "campaign",
        checkpoint: "campaign/stage/run/checkpoint.pt",
        label: "control · iTransformer · point-mass · simple-v1 · run",
        model: "itransformer",
        predictionOutput: "control",
        horizonMode: "normalized",
        seed: 1337,
      },
    }] })).toBe(true);
    expect(isComparisonCategoriesManifest({ categories: [{
      ...entry,
      experiment: {
        id: "campaign/stage/run",
        group: "campaign",
        checkpoint: "campaign/stage/run/checkpoint.pt",
        label: 7,
      },
    }] })).toBe(false);
    expect(isComparisonCategoriesManifest({ categories: [{
      ...entry,
      experiment: { id: "run", group: "campaign" },
    }] })).toBe(false);
    expect(isComparisonCategoriesManifest({ categories: [{
      ...entry,
      experiment: {
        id: "campaign/stage/run",
        group: "campaign",
        checkpoint: "campaign/stage/run/checkpoint.pt",
        horizonMode: "unknown",
      },
    }] })).toBe(false);
  });

  // Every value of config.PREDICTION_OUTPUTS must survive here: one unlisted output rejects
  // its own category AND, through `.every`, empties the airport's entire manifest.
  it.each([...EXPERIMENT_PREDICTION_OUTPUTS])(
    "accepts %s experiment metadata without rejecting sibling categories",
    (predictionOutput) => {
      const groundTruth = {
        key: "observed",
        label: "Observed ADS-B",
        dir: "observed",
        groups: 0,
        constrained: false,
      };
      const arm = {
        key: `experiment_${predictionOutput}_val`,
        label: `${predictionOutput} · validation`,
        dir: `experiment_${predictionOutput}_val`,
        groups: 3,
        constrained: false,
        datasetSplit: "val",
        resultSource: "experiment",
        experiment: {
          id: `campaign/stage/${predictionOutput}_seed1337`,
          group: "campaign",
          checkpoint: `campaign/stage/${predictionOutput}_seed1337/checkpoint.pt`,
          model: "itransformer",
          predictionOutput,
          horizonMode: "normalized",
          seed: 1337,
        },
      };

      expect(isComparisonCategoriesManifest({ categories: [groundTruth, arm] })).toBe(true);
    },
  );

  it("distinguishes report-only evaluation categories from drawable comparisons", () => {
    expect(
      isDrawableComparisonCategory({
        key: "observed",
        label: "Observed ADS-B",
        dir: "observed",
        groups: 0,
        constrained: false,
      }),
    ).toBe(false);
    expect(
      isDrawableComparisonCategory({
        key: "runway",
        label: "Runway target",
        dir: "runway",
        groups: 12,
        constrained: false,
      }),
    ).toBe(true);
  });

  it("accepts only the current atomic comparison publication contract", () => {
    const current = {
      schemaVersion: "comparison-v2-generation",
      generation: "batch123",
      epoch: "2026-07-23T12:00:00Z",
      startHidden: true,
      referenceSource: "canonicalObserved",
      evaluationReport: "evaluation_report_batch123.json",
      groups: [],
    };

    expect(isComparisonIndex(current)).toBe(true);
    expect(isComparisonIndex({ ...current, schemaVersion: undefined })).toBe(false);
    expect(isComparisonIndex({ ...current, generation: undefined })).toBe(false);
    expect(isComparisonIndex({ ...current, referenceSource: undefined })).toBe(false);
    expect(isComparisonIndex({ ...current, evaluationReport: undefined })).toBe(false);
  });

  it("accepts a group that passed on another runway, naming the runway it landed on", () => {
    const group = { group: "X_05L", flightId: "X", runway: "05L", airport: "KRDU", czml: "c.czml", entities: ["pred-X_05L"],
                    status: "otherRunway", landedRunway: "05R", observedRunwayVerdict: "fail" };
    const index = { schemaVersion: "comparison-v2-generation", generation: "g", epoch: "2026-09-30T00:00:00Z",
                    startHidden: true, referenceSource: "canonicalObserved", evaluationReport: "r.json", groups: [group] };
    expect(isComparisonIndex(index)).toBe(true);
    expect(isComparisonIndex({ ...index, groups: [{ ...group, landedRunway: 5 }] })).toBe(false);
    expect(isComparisonIndex({ ...index, groups: [{ ...group, status: "elsewhere" }] })).toBe(false);
    const { landedRunway: _landed, ...unnamed } = group;
    expect(isComparisonIndex({ ...index, groups: [unnamed] })).toBe(false);          // which runway, then?
    expect(isComparisonIndex({ ...index, groups: [{ ...group, observedRunwayVerdict: "maybe" }] })).toBe(false);
  });
});

describe("a traffic window's group", () => {
  const traffic = {
    outcome: "separated",
    recorded: ["ref-DAL1312_05L_d4e5f6_20260501T000100Z", "ref-UPS22_05R_a7b8c9_20260501T000500Z"],
    startOffsetsS: [-9.349, 90.75],
  };
  const group = { group: "X_05L", flightId: "X", runway: "05L", airport: "KRDU", czml: "c.czml", status: "solved",
                  entities: ["ref-X_05L", "sim-X_05L"], traffic };
  const index = (groups: unknown[]) => ({
    schemaVersion: "comparison-v2-generation", generation: "g", epoch: "2026-10-05T00:00:00Z", startHidden: true,
    referenceSource: "canonicalObserved", evaluationReport: "r.json", groups,
  });

  it("is read with its traffic block, and a group without one is still a group", () => {
    expect(isComparisonIndex(index([group]))).toBe(true);
    const { traffic: _traffic, ...plain } = group;
    expect(isComparisonIndex(index([plain]))).toBe(true);
  });

  it("refuses a malformed traffic block rather than drawing a guess", () => {
    for (const broken of [
      { ...traffic, outcome: 3 },
      { ...traffic, recorded: ["UPS22_05R_a7b8c9_20260501T000500Z", traffic.recorded[1]] },   // not a ref- id
      { ...traffic, startOffsetsS: [90.75] },                                                // one offset short
      { ...traffic, startOffsetsS: [-9.349, "90.75"] },
      { outcome: "separated", recorded: traffic.recorded },                                  // no offsets
    ]) {
      expect(isComparisonIndex(index([{ ...group, traffic: broken }]))).toBe(false);
    }
  });
});

describe("experiment intent and parameter rows", () => {
  const base = {
    key: "experiment_run_val",
    label: "Experiment",
    dir: "experiment_run_val",
    groups: 3,
    constrained: false,
    datasetSplit: "val",
    resultSource: "experiment",
  };
  const metadata = {
    id: "campaign/run",
    group: "campaign",
    checkpoint: "campaign/run/checkpoint.pt",
    runName: "run",
    variantLabel: null,
    parameters: [{ section: "Model", name: "Output", value: "control", field: "prediction_output" }],
    intent: { groupTitle: "Campaign", group: "What it asks", run: "What it changes" },
  };

  it("accepts the stamped structure and its absence", () => {
    expect(isComparisonCategory({ ...base, experiment: metadata })).toBe(true);
    expect(isComparisonCategory({
      ...base,
      experiment: { ...metadata, intent: { ...metadata.intent, variant: "v", design: "doc.md" } },
    })).toBe(true);
    const { runName: _r, parameters: _p, intent: _i, ...legacy } = metadata;
    expect(isComparisonCategory({ ...base, experiment: legacy })).toBe(true);
  });

  it.each([
    ["an intent without its run", { intent: { groupTitle: "C", group: "q" } }],
    ["a non-string variant intent", { intent: { ...metadata.intent, variant: 3 } }],
    ["a parameter row without a value", { parameters: [{ section: "Model", name: "Output" }] }],
    ["parameters that are not a list", { parameters: { Output: "control" } }],
    ["a numeric run name", { runName: 7 }],
  ])("rejects %s", (_name, override) => {
    expect(isComparisonCategory({ ...base, experiment: { ...metadata, ...override } })).toBe(false);
  });
});
