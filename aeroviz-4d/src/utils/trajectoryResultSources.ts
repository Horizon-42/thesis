import type {
  ComparisonCategory,
  ComparisonResultSource,
  ExperimentPredictionOutput,
} from "../data/airportData";

export type TrajectoryResultSource = "baseline" | ComparisonResultSource;

export function isExperimentCategory(category: ComparisonCategory): boolean {
  return category.resultSource === "experiment" && category.experiment !== undefined;
}

export function categoryResultSource(
  category: ComparisonCategory,
): ComparisonResultSource {
  return isExperimentCategory(category) ? "experiment" : "prediction";
}

export function categoriesForResultSource(
  categories: ComparisonCategory[],
  source: ComparisonResultSource,
): ComparisonCategory[] {
  return categories.filter((category) => categoryResultSource(category) === source);
}

export function activeTrajectoryResultSource(
  comparisonEnabled: boolean,
  category: ComparisonCategory | null,
): TrajectoryResultSource {
  if (!comparisonEnabled) return "baseline";
  return category ? categoryResultSource(category) : "prediction";
}

export interface ExperimentOption {
  id: string;
  group: string;
  label: string;
  model?: string | null;
  predictionOutput?: ExperimentPredictionOutput | null;
  horizonMode?: "normalized" | "full" | "window" | null;
  seed?: number | null;
}

export function experimentOptions(categories: ComparisonCategory[]): ExperimentOption[] {
  const byId = new Map<string, ExperimentOption>();
  for (const category of categories) {
    const experiment = category.experiment;
    if (!isExperimentCategory(category) || !experiment || byId.has(experiment.id)) continue;
    const identityParts = experiment.id.split("/").filter(Boolean);
    const runName = identityParts[identityParts.length - 1] ?? experiment.id;
    byId.set(experiment.id, {
      id: experiment.id,
      group: experiment.group,
      label: runName,
      model: experiment.model,
      predictionOutput: experiment.predictionOutput,
      horizonMode: experiment.horizonMode,
      seed: experiment.seed,
    });
  }
  return [...byId.values()].sort(
    (left, right) => left.group.localeCompare(right.group) || left.label.localeCompare(right.label),
  );
}

export function categoryForExperimentSplit(
  categories: ComparisonCategory[],
  experimentId: string,
  preferredSplit: "train" | "val" | "test" = "val",
): ComparisonCategory | null {
  const matches = categories.filter((category) => category.experiment?.id === experimentId);
  return (
    matches.find((category) => category.datasetSplit === preferredSplit) ??
    matches.find((category) => category.datasetSplit === "val") ??
    matches.find((category) => category.datasetSplit === "train") ??
    matches[0] ??
    null
  );
}
