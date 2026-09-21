/**
 * checkPublication.ts
 * -------------------
 * What the comparison picker would do with a published airport, as pure functions over the
 * loaded JSON — the SAME guards the app runs (`isComparisonCategoriesManifest`,
 * `isComparisonIndex`), so a publication that passes here loads in the picker and one that
 * fails here names the category and the field. `scripts/check_publication.ts` is the CLI
 * around it (disk + the running dev server); the guards themselves stay in `airportData.ts`.
 *
 * Why this exists: the manifest validator is `.every(isComparisonCategory)`, so ONE category
 * carrying a value the frontend does not list empties the whole airport's picker, with a
 * one-line console error nobody reads after a publication (`closure` 2026-09-07, `plan`
 * 2026-09-12 — both times every file was served with HTTP 200).
 */

import {
  COMPARISON_RESULT_SOURCES,
  DATASET_SPLITS,
  EXPERIMENT_HORIZON_MODES,
  EXPERIMENT_PREDICTION_OUTPUTS,
  isComparisonCategoriesManifest,
  isComparisonCategory,
  isComparisonIndex,
  isExperimentHorizonMode,
  isExperimentIntent,
  isExperimentParameterRow,
  isExperimentPredictionOutput,
  type ComparisonCategory,
  type ComparisonIndex,
} from "../data/airportData";
import {
  parseTrainingIndex,
  parseTrainingSample,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";

export interface PublicationFinding {
  level: "error" | "warn";
  /** The category key the finding is about; absent for manifest-level findings. */
  category?: string;
  message: string;
}

function describe(value: unknown): string {
  return value === undefined ? "absent" : JSON.stringify(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Why `isComparisonCategory` rejects `value` — the required fields and the enumerated
 * vocabularies named one by one, so the reason is readable. The guard stays the authority: a
 * value it accepts is never explained, and one it rejects for a reason not listed here is
 * reported as such rather than passed.
 */
export function explainCategoryRejection(value: unknown): string | null {
  if (isComparisonCategory(value)) return null;
  if (!isPlainObject(value)) return `not an object: ${describe(value)}`;
  const candidate = value;
  for (const field of ["key", "label", "dir"] as const) {
    if (typeof candidate[field] !== "string") {
      return `\`${field}\` must be a string, got ${describe(candidate[field])}`;
    }
  }
  if (typeof candidate.constrained !== "boolean") {
    return `\`constrained\` must be a boolean, got ${describe(candidate.constrained)}`;
  }
  if (typeof candidate.groups !== "number") {
    return `\`groups\` must be a number, got ${describe(candidate.groups)}`;
  }
  if (
    candidate.datasetSplit !== undefined &&
    !(DATASET_SPLITS as readonly unknown[]).includes(candidate.datasetSplit)
  ) {
    return `\`datasetSplit\` ${describe(candidate.datasetSplit)} is not one of ${JSON.stringify(DATASET_SPLITS)}`;
  }
  if (
    candidate.resultSource !== undefined &&
    !(COMPARISON_RESULT_SOURCES as readonly unknown[]).includes(candidate.resultSource)
  ) {
    return `\`resultSource\` ${describe(candidate.resultSource)} is not one of ${JSON.stringify(COMPARISON_RESULT_SOURCES)}`;
  }
  if (
    candidate.accuracy !== undefined &&
    candidate.accuracy !== null &&
    typeof candidate.accuracy !== "object"
  ) {
    return `\`accuracy\` must be an object or null, got ${describe(candidate.accuracy)}`;
  }
  const experiment = candidate.experiment;
  if (candidate.resultSource === "experiment" && experiment === undefined) {
    return "`resultSource` is \"experiment\" but `experiment` is absent";
  }
  if (experiment !== undefined) {
    if (!isPlainObject(experiment)) {
      return `\`experiment\` must be an object, got ${describe(experiment)}`;
    }
    for (const field of ["id", "group", "checkpoint"] as const) {
      if (typeof experiment[field] !== "string") {
        return `\`experiment.${field}\` must be a string, got ${describe(experiment[field])}`;
      }
    }
    for (const field of ["label", "model", "runName", "variantLabel"] as const) {
      if (experiment[field] != null && typeof experiment[field] !== "string") {
        return `\`experiment.${field}\` must be a string or null, got ${describe(experiment[field])}`;
      }
    }
    if (experiment.predictionOutput != null && !isExperimentPredictionOutput(experiment.predictionOutput)) {
      return (
        `\`experiment.predictionOutput\` ${describe(experiment.predictionOutput)} is not one of ` +
        `${JSON.stringify(EXPERIMENT_PREDICTION_OUTPUTS)} — the frontend mirror of ` +
        "`config.PREDICTION_OUTPUTS` in `src/data/airportData.ts` has not learned it"
      );
    }
    if (experiment.horizonMode != null && !isExperimentHorizonMode(experiment.horizonMode)) {
      return (
        `\`experiment.horizonMode\` ${describe(experiment.horizonMode)} is not one of ` +
        `${JSON.stringify(EXPERIMENT_HORIZON_MODES)}`
      );
    }
    if (experiment.seed != null && typeof experiment.seed !== "number") {
      return `\`experiment.seed\` must be a number or null, got ${describe(experiment.seed)}`;
    }
    if (experiment.parameters != null) {
      if (!Array.isArray(experiment.parameters)) {
        return `\`experiment.parameters\` must be an array or null, got ${describe(experiment.parameters)}`;
      }
      const bad = experiment.parameters.findIndex((row) => !isExperimentParameterRow(row));
      if (bad >= 0) {
        return (
          `\`experiment.parameters[${bad}]\` must be {section, name, value: string, field?: string}, ` +
          `got ${describe(experiment.parameters[bad])}`
        );
      }
    }
    if (experiment.intent != null && !isExperimentIntent(experiment.intent)) {
      return (
        "`experiment.intent` must be {groupTitle, group, run: string, variant?, design?: string}, " +
        `got ${describe(experiment.intent)}`
      );
    }
  }
  return "rejected by `isComparisonCategory` for a reason this explainer does not name";
}

/** The picker's verdict on a `comparison/categories.json`, one finding per rejected category. */
export function checkCategoriesManifest(manifest: unknown): PublicationFinding[] {
  if (isComparisonCategoriesManifest(manifest)) {
    return manifest.categories.length === 0
      ? [{ level: "warn", message: "the manifest lists no categories" }]
      : [];
  }
  const categories = isPlainObject(manifest) ? manifest.categories : undefined;
  if (!Array.isArray(categories)) {
    return [{ level: "error", message: "`categories` is not an array — the picker shows nothing" }];
  }
  const findings: PublicationFinding[] = [];
  categories.forEach((category, position) => {
    const reason = explainCategoryRejection(category);
    if (reason === null) return;
    const key = isPlainObject(category) ? category.key : undefined;
    findings.push({
      level: "error",
      category: typeof key === "string" ? key : `#${position}`,
      message: `rejected by the picker (which then shows NOTHING for this airport): ${reason}`,
    });
  });
  return findings;
}

/** The distinct CZML file names a comparison index refers to. */
export function indexCzmlFiles(index: ComparisonIndex): string[] {
  return [...new Set(index.groups.map((group) => group.czml))];
}

/** The picker's verdict on one category's `comparison_index.json`, against its manifest row. */
export function checkComparisonIndex(
  category: ComparisonCategory,
  index: unknown,
): PublicationFinding[] {
  if (!isComparisonIndex(index)) {
    return [{
      level: "error",
      category: category.key,
      message: "`comparison_index.json` is rejected by `isComparisonIndex` — the category lists but cannot load",
    }];
  }
  if (index.groups.length === 0 && category.groups > 0) {
    return [{
      level: "error",
      category: category.key,
      message: `the manifest says ${category.groups} groups, the index holds none — nothing to draw`,
    }];
  }
  const findings: PublicationFinding[] = [];
  if (index.groups.length !== category.groups) {
    findings.push({
      level: "warn",
      category: category.key,
      message: `the manifest says ${category.groups} groups, the index holds ${index.groups.length}`,
    });
  }
  if (category.datasetSplit !== undefined && index.datasetSplit !== category.datasetSplit) {
    findings.push({
      level: "warn",
      category: category.key,
      message: `the manifest says split ${describe(category.datasetSplit)}, the index ${describe(index.datasetSplit)}`,
    });
  }
  return findings;
}

// ── the Training export (design §7, T8) ──────────────────────────────────────

/**
 * The Training manifest as the panel would read it, plus what the panel cannot see:
 * that every set it lists has a sample file on disk.
 *
 * Training is checked on the OPPOSITE rule to the comparison picker above. There a bad
 * category empties the airport, so the check exists to stop that; here a bad set is greyed
 * on its own by design (§4.5 ③), which means a half-written export is easy to publish and
 * never notice. The check is what notices — and it names the set and the field, because
 * "the manifest is invalid" is what wasted the two sessions this module's rule came from.
 */
export function checkTrainingIndex(manifest: unknown): PublicationFinding[] {
  const parsed = parseTrainingIndex(manifest);
  if (!parsed.ok) {
    return [{ level: "error", message: `training/index.json is not a manifest: ${parsed.problem}` }];
  }
  return parsed.value.rejected.map((item) => ({
    level: "error" as const,
    category: item.id,
    message: `the panel would grey this set out: ${item.problem}`,
  }));
}

/**
 * One set's sample file, through the panel's own reader.
 *
 * `readUnder` is the reading rule the MANIFEST claims for this set. It is
 * REQUIRED — its one caller always has it, and an optional one could only ever
 * fire on a caller that forgot, printing a message missing the half that
 * explains it. It is
 * printed with the failure because the commonest failure now is a superseded
 * export: the vocabulary's second word kind changed from a height to an angle on
 * 2026-09-21, so every sentence in an older file still parses as six columns and
 * the reader refuses it on `kinds` alone. "kinds is [heading,altitude,…]" says
 * what is wrong; the rule that produced it says WHY, and which command to rerun.
 */
export function checkTrainingSample(
  setId: string,
  sample: unknown,
  readUnder: string,
): PublicationFinding[] {
  const parsed = parseTrainingSample(sample);
  if (!parsed.ok) {
    return [{
      level: "error",
      category: setId,
      message: `sample.json (read under ${readUnder}): ${parsed.problem}`,
    }];
  }
  return [];
}

/**
 * What the manifest promises against what the sample holds. The two files are written by one
 * run of the exporter, so a disagreement means they came from different runs — which the
 * panel says on screen and the check says here, rather than either of them averaging over it.
 */
export function checkTrainingSetAgrees(
  entry: TrainingSetEntry,
  sample: TrainingSample,
): PublicationFinding[] {
  const findings: PublicationFinding[] = [];
  if (sample.setId !== entry.id) {
    findings.push({ level: "error", category: entry.id, message: `sample.json calls itself ${sample.setId}` });
  }
  if (sample.flights.length !== entry.flights) {
    findings.push({
      level: "error",
      category: entry.id,
      message: `the manifest lists ${entry.flights} flights, sample.json holds ${sample.flights.length}`,
    });
  }
  for (const [what, listed, held] of [
    ["vocabularySha256", entry.vocabularySha256, sample.vocabulary.sha256],
    ["runwaySha256", entry.runwaySha256, sample.vocabulary.runwaySha256],
    ["readingRule", entry.readingRule, sample.vocabulary.readingRule],
  ] as const) {
    if (listed !== held) {
      findings.push({
        level: "error",
        category: entry.id,
        message: `${what}: the manifest says ${listed}, sample.json says ${held}`,
      });
    }
  }
  return findings;
}
