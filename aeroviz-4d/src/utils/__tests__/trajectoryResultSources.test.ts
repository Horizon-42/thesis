import { describe, expect, it } from "vitest";
import type { ComparisonCategory } from "../../data/airportData";
import {
  activeTrajectoryResultSource,
  categoriesForResultSource,
  categoryAccuracyValue,
  categoryForExperimentSplit,
  categoryResultSource,
  experimentOptions,
  sortCategoriesByAccuracy,
} from "../trajectoryResultSources";

const prediction: ComparisonCategory = {
  key: "ts_model_val",
  label: "Prediction",
  dir: "ts_model_val",
  groups: 5,
  constrained: false,
  datasetSplit: "val",
};

/** An optimizer publish: no `resultSource` (the field postdates them), non-`ts_` key. */
const optimization: ComparisonCategory = {
  key: "runway_cons",
  label: "Runway target (constrained)",
  dir: "runway_cons",
  groups: 7,
  constrained: true,
};

function experiment(split: "train" | "val"): ComparisonCategory {
  return {
    key: `experiment_abc_${split}`,
    label: `Experiment ${split}`,
    dir: `experiment_abc_${split}`,
    groups: 5,
    constrained: false,
    datasetSplit: split,
    resultSource: "experiment",
    experiment: {
      id: "campaign/stage/run_seed1337",
      group: "campaign",
      checkpoint: "campaign/stage/run_seed1337/checkpoint.pt",
      model: "itransformer",
      predictionOutput: "control",
      horizonMode: "normalized",
      seed: 1337,
    },
  };
}

describe("trajectory result sources", () => {
  it("splits categories into Optimization, Prediction and Experiments", () => {
    const categories = [optimization, prediction, experiment("train"), experiment("val")];
    expect(categoriesForResultSource(categories, "optimization")).toEqual([optimization]);
    expect(categoriesForResultSource(categories, "prediction")).toEqual([prediction]);
    expect(categoriesForResultSource(categories, "experiment")).toHaveLength(2);
    expect(activeTrajectoryResultSource(false, experiment("val"))).toBe("baseline");
    expect(activeTrajectoryResultSource(true, optimization)).toBe("optimization");
    expect(activeTrajectoryResultSource(true, prediction)).toBe("prediction");
    expect(activeTrajectoryResultSource(true, experiment("val"))).toBe("experiment");
  });

  it("classifies legacy ts_-keyed publishes as Prediction, with or without resultSource", () => {
    expect(categoryResultSource(prediction)).toBe("prediction");
    expect(categoryResultSource({ ...prediction, resultSource: "prediction" }))
      .toBe("prediction");
    expect(categoryResultSource({ ...optimization, key: "fitted_adsb", dir: "fitted_adsb" }))
      .toBe("optimization");
  });

  it("deduplicates experiment models and selects the requested split", () => {
    const categories = [experiment("train"), experiment("val")];
    expect(experimentOptions(categories)).toEqual([{
      id: "campaign/stage/run_seed1337",
      group: "campaign",
      label: "run_seed1337 · control · normalized time · seed 1337",
      model: "itransformer",
      predictionOutput: "control",
      horizonMode: "normalized",
      seed: 1337,
      metricValue: null,
    }]);
    expect(categoryForExperimentSplit(categories, "campaign/stage/run_seed1337", "train")?.datasetSplit)
      .toBe("train");
  });

  it("ranks one split's categories best-first by the chosen metric, unknowns last", () => {
    const worse = { ...prediction, key: "b_val", dir: "b_val",
      accuracy: { adeM: { mean: 1400, p95: 4100 }, fdeM: { mean: 1700, p95: 5900 } } };
    const better = { ...prediction, key: "a_val", dir: "a_val",
      accuracy: { adeM: { mean: 480, p95: 1500 }, fdeM: { mean: 1750, p95: 5000 } } };
    const unranked = { ...prediction, key: "c_val", dir: "c_val" };

    expect(sortCategoriesByAccuracy([worse, better, unranked], null))
      .toEqual([worse, better, unranked]); // no metric → manifest order untouched
    expect(sortCategoriesByAccuracy([worse, better, unranked], "adeMean")
      .map((category) => category.key)).toEqual(["a_val", "b_val", "c_val"]);
    // FDE flips the ranking — the two metrics are independent axes.
    expect(sortCategoriesByAccuracy([worse, better, unranked], "fdeMean")
      .map((category) => category.key)).toEqual(["b_val", "a_val", "c_val"]);
    expect(categoryAccuracyValue(better, "adeP95")).toBe(1500);
    expect(categoryAccuracyValue(unranked, "adeMean")).toBeNull();
  });

  it("ranks experiments within a campaign by the preferred split's metric", () => {
    const strong = experiment("val");
    strong.experiment = { ...strong.experiment!, id: "campaign/strong" };
    strong.key = "experiment_strong_val";
    strong.dir = "experiment_strong_val";
    strong.accuracy = { adeM: { mean: 480, p95: 1500 } };
    const weak = experiment("val");
    weak.experiment = { ...weak.experiment!, id: "campaign/aaa_weak" };
    weak.key = "experiment_weak_val";
    weak.dir = "experiment_weak_val";
    weak.accuracy = { adeM: { mean: 1400, p95: 4100 } };

    // Default: label order (aaa_weak first). Sorted: the strong model leads.
    expect(experimentOptions([weak, strong]).map((option) => option.id))
      .toEqual(["campaign/aaa_weak", "campaign/strong"]);
    const ranked = experimentOptions([weak, strong], "adeMean", "val");
    expect(ranked.map((option) => option.id))
      .toEqual(["campaign/strong", "campaign/aaa_weak"]);
    expect(ranked[0]?.metricValue).toBe(480);
  });

  it("prefers the publisher's canonical run label when stamped", () => {
    const canonical = experiment("val");
    canonical.experiment = {
      ...canonical.experiment!,
      label: "control · iTransformer · point-mass · simple-v1 · run_seed1337",
    };

    expect(experimentOptions([canonical])[0]?.label).toBe(
      "control · iTransformer · point-mass · simple-v1 · run_seed1337",
    );
  });

  it("preserves the control-mixture output identity in experiment options", () => {
    const mixture = experiment("val");
    mixture.experiment = {
      ...mixture.experiment!,
      predictionOutput: "control-mixture",
    };

    expect(experimentOptions([mixture])[0]?.predictionOutput).toBe("control-mixture");
  });
});
