import { describe, expect, it } from "vitest";

import {
  AIRPORTS_INDEX_URL,
  airportDataUrl,
  airportProcedureDetailUrl,
  airportProcedureDetailsIndexUrl,
  airportLocalTerrainUrl,
  airportChartsIndexUrl,
  isAirportsIndexManifest,
  isComparisonCategoriesManifest,
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
        model: "itransformer",
        predictionOutput: "control",
        horizonMode: "normalized",
        seed: 1337,
      },
    }] })).toBe(true);
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

  it("accepts control-mixture experiment metadata without rejecting sibling categories", () => {
    const baseline = {
      key: "observed",
      label: "Observed ADS-B",
      dir: "observed",
      groups: 0,
      constrained: false,
    };
    const mixture = {
      key: "experiment_mixture_val",
      label: "Control mixture · validation",
      dir: "experiment_mixture_val",
      groups: 3,
      constrained: false,
      datasetSplit: "val",
      resultSource: "experiment",
      experiment: {
        id: "campaign/stage/control_mixture_seed1337",
        group: "campaign",
        checkpoint: "campaign/stage/control_mixture_seed1337/checkpoint.pt",
        model: "itransformer",
        predictionOutput: "control-mixture",
        horizonMode: "normalized",
        seed: 1337,
      },
    };

    expect(isComparisonCategoriesManifest({ categories: [baseline, mixture] })).toBe(true);
  });

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
});
