import type {
  ComparisonCategory,
  ComparisonResultSource,
  DatasetSplit,
  ExperimentIntent,
  ExperimentParameterRow,
  ExperimentPredictionOutput,
} from "../data/airportData";

/**
 * UI-level partition of the comparison categories. The manifest's `resultSource`
 * field only exists for data-driven publishes ("prediction" | "experiment");
 * optimizer publishes (`run_scenario_optimization.py` → fitted_adsb / runway /
 * runway_cons) predate the field and never stamp it. The `ts_` key prefix is the
 * legacy marker for data-driven categories published before `resultSource`
 * existed — the same rule EvaluationSummary keys its presentation off.
 */
export type ComparisonCategoryKind = ComparisonResultSource | "optimization";

export type TrajectoryResultSource = "baseline" | ComparisonCategoryKind;

export function isExperimentCategory(category: ComparisonCategory): boolean {
  return category.resultSource === "experiment" && category.experiment !== undefined;
}

export function categoryResultSource(
  category: ComparisonCategory,
): ComparisonCategoryKind {
  if (isExperimentCategory(category)) return "experiment";
  if (category.resultSource === "prediction" || category.key.startsWith("ts_")) {
    return "prediction";
  }
  return "optimization";
}

export function categoriesForResultSource(
  categories: ComparisonCategory[],
  source: ComparisonCategoryKind,
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

/**
 * Metric a split's results can be RANKED by — batch mean or p95 of ADE/FDE, read off
 * each category's `accuracy` block (stamped into categories.json at publish time).
 * `null` keeps the manifest's own (name) order.
 */
export type ResultAccuracySortKey = "adeMean" | "adeP95" | "fdeMean" | "fdeP95";

export const RESULT_ACCURACY_SORT_LABELS: Record<ResultAccuracySortKey, string> = {
  adeMean: "ADE mean",
  adeP95: "ADE p95",
  fdeMean: "FDE mean",
  fdeP95: "FDE p95",
};

export function categoryAccuracyValue(
  category: ComparisonCategory | null | undefined,
  sortBy: ResultAccuracySortKey,
): number | null {
  const metric = sortBy.startsWith("ade")
    ? category?.accuracy?.adeM
    : category?.accuracy?.fdeM;
  const value = sortBy.endsWith("Mean") ? metric?.mean : metric?.p95;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Rank categories best-first (smallest error) by the chosen metric; categories without
 * a value keep their relative order at the end. Callers pass ONE split's categories —
 * cross-split error comparisons are meaningless, and the pickers group by split.
 */
export function sortCategoriesByAccuracy(
  categories: ComparisonCategory[],
  sortBy: ResultAccuracySortKey | null,
): ComparisonCategory[] {
  if (!sortBy) return categories;
  return [...categories].sort((left, right) => {
    const a = categoryAccuracyValue(left, sortBy);
    const b = categoryAccuracyValue(right, sortBy);
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    return a - b;
  });
}

export interface ExperimentOption {
  id: string;
  group: string;
  /** The group's heading: the registry title, or the campaign id for unstamped publishes. */
  groupTitle: string;
  /** Ready-to-render display label — see {@link experimentOptionLabel}. */
  label: string;
  /** The run's short name (its id); unstamped publishes fall back to the full label. */
  runName: string;
  /** Which records these are (anytime bin, predict-time hook …); null for the plain run. */
  variantLabel: string | null;
  intent: ExperimentIntent | null;
  parameters: ExperimentParameterRow[];
  checkpoint: string;
  model?: string | null;
  predictionOutput?: ExperimentPredictionOutput | null;
  horizonMode?: "normalized" | "full" | "window" | null;
  seed?: number | null;
  /** The ranking metric's value for the preferred split; null without a sort/value. */
  metricValue?: number | null;
}

const HORIZON_SUFFIX = {
  normalized: " · normalized time",
  full: " · full horizon",
  window: " · recursive window",
} as const;

/**
 * The picker label for one experiment. The publisher stamps a canonical
 * self-describing `label` (run_naming grammar: output · backbone · dynamics · loss ·
 * meta) — use it verbatim. Publishes that predate it fall back to the run-directory
 * name decorated with the metadata fields.
 */
export function experimentOptionLabel(
  experiment: NonNullable<ComparisonCategory["experiment"]>,
): string {
  if (experiment.label) return experiment.label;
  const identityParts = experiment.id.split("/").filter(Boolean);
  const runName = identityParts[identityParts.length - 1] ?? experiment.id;
  const horizon = experiment.horizonMode ? HORIZON_SUFFIX[experiment.horizonMode] : "";
  const seed = experiment.seed == null ? "" : ` · seed ${experiment.seed}`;
  return `${runName} · ${experiment.predictionOutput ?? "state"}${horizon}${seed}`;
}

/**
 * Deduplicated experiment models, grouped by campaign. With `sortBy`, each option
 * carries the metric of its `preferredSplit` category and experiments are ranked
 * best-first WITHIN their campaign group (the picker renders one optgroup per
 * campaign); metric-less experiments keep label order at the group's end. An
 * experiment with no category of that split carries no metric: another split's
 * number (a held-out-days run's, an in-sample one) is never ranked against it.
 */
export function experimentOptions(
  categories: ComparisonCategory[],
  sortBy: ResultAccuracySortKey | null = null,
  preferredSplit: DatasetSplit = "val",
): ExperimentOption[] {
  const byId = new Map<string, ExperimentOption>();
  for (const category of categories) {
    const experiment = category.experiment;
    if (!isExperimentCategory(category) || !experiment || byId.has(experiment.id)) continue;
    const label = experimentOptionLabel(experiment);
    byId.set(experiment.id, {
      id: experiment.id,
      group: experiment.group,
      groupTitle: experiment.intent?.groupTitle ?? experiment.group,
      label,
      runName: experiment.runName ?? label,
      variantLabel: experiment.variantLabel ?? null,
      intent: experiment.intent ?? null,
      parameters: experiment.parameters ?? [],
      checkpoint: experiment.checkpoint,
      model: experiment.model,
      predictionOutput: experiment.predictionOutput,
      horizonMode: experiment.horizonMode,
      seed: experiment.seed,
      metricValue: sortBy
        ? categoryAccuracyValue(
            categories.find(
              (candidate) =>
                candidate.experiment?.id === experiment.id &&
                candidate.datasetSplit === preferredSplit,
            ),
            sortBy,
          )
        : null,
    });
  }
  return [...byId.values()].sort((left, right) => {
    const byGroup = left.group.localeCompare(right.group);
    if (byGroup !== 0) return byGroup;
    if (sortBy) {
      const a = left.metricValue ?? null;
      const b = right.metricValue ?? null;
      if (a != null || b != null) {
        if (a == null) return 1;
        if (b == null) return -1;
        if (a !== b) return a - b;
      }
    }
    // By what the browser shows — the run's name, then its variant — numerically, so a
    // campaign's "@ 6km" bin precedes its "@ 12km" one.
    return RUN_ORDER.compare(left.runName, right.runName) ||
      RUN_ORDER.compare(left.variantLabel ?? "", right.variantLabel ?? "") ||
      RUN_ORDER.compare(left.label, right.label);
  });
}

const RUN_ORDER = new Intl.Collator(undefined, { numeric: true });

/** One heading of the experiment browser: a campaign and its runs, in option order. */
export interface ExperimentGroup {
  id: string;
  title: string;
  /** The campaign's question — from its first stamped run; null when none is stamped. */
  intent: string | null;
  design: string | null;
  experiments: ExperimentOption[];
}

export function experimentGroups(experiments: ExperimentOption[]): ExperimentGroup[] {
  const groups = new Map<string, ExperimentGroup>();
  for (const experiment of experiments) {
    let group = groups.get(experiment.group);
    if (!group) {
      group = {
        id: experiment.group,
        title: experiment.group,
        intent: null,
        design: null,
        experiments: [],
      };
      groups.set(experiment.group, group);
    }
    // Heading and question from the same run — the first stamped one — so a campaign that is
    // only partly stamped never pairs its raw id with the registry's question.
    if (group.intent === null && experiment.intent) {
      group.title = experiment.intent.groupTitle;
      group.intent = experiment.intent.group;
      group.design = experiment.intent.design ?? null;
    }
    group.experiments.push(experiment);
  }
  return [...groups.values()];
}

/**
 * Case-insensitive match of every whitespace-separated term against what the browser shows
 * for a run: its names, its campaign, its intent and its parameter rows.
 */
export function experimentMatches(experiment: ExperimentOption, query: string): boolean {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;
  const haystack = [
    experiment.runName,
    experiment.variantLabel,
    experiment.label,
    experiment.group,
    experiment.groupTitle,
    experiment.intent?.group,
    experiment.intent?.run,
    experiment.intent?.variant,
    ...experiment.parameters.flatMap((row) => [row.name, row.value]),
  ].filter(Boolean).join("\n").toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

export function categoryForExperimentSplit(
  categories: ComparisonCategory[],
  experimentId: string,
  preferredSplit: DatasetSplit = "val",
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
