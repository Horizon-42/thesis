import { describe, expect, it } from "vitest";
import type { ComparisonCategory } from "../../data/airportData";
import {
  activeTrajectoryResultSource,
  categoriesForResultSource,
  categoryForExperimentSplit,
  categoryResultSource,
  experimentOptions,
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
      label: "run_seed1337",
      model: "itransformer",
      predictionOutput: "control",
      horizonMode: "normalized",
      seed: 1337,
    }]);
    expect(categoryForExperimentSplit(categories, "campaign/stage/run_seed1337", "train")?.datasetSplit)
      .toBe("train");
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
