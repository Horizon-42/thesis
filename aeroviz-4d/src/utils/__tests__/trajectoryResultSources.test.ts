import { describe, expect, it } from "vitest";
import type { ComparisonCategory } from "../../data/airportData";
import {
  activeTrajectoryResultSource,
  categoriesForResultSource,
  categoryForExperimentSplit,
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
  it("keeps legacy categories in Prediction and explicit sweeps in Experiments", () => {
    const categories = [prediction, experiment("train"), experiment("val")];
    expect(categoriesForResultSource(categories, "prediction")).toEqual([prediction]);
    expect(categoriesForResultSource(categories, "experiment")).toHaveLength(2);
    expect(activeTrajectoryResultSource(false, experiment("val"))).toBe("baseline");
    expect(activeTrajectoryResultSource(true, prediction)).toBe("prediction");
    expect(activeTrajectoryResultSource(true, experiment("val"))).toBe("experiment");
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
});
