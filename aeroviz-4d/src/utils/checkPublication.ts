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
  trainingSetRefusal,
  type TrainingIndex,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";
import {
  parseTrainingExecutorOverlay,
  parseTrainingGenerationOverlay,
  parseTrainingOverlays,
  parseTrainingPriorOverlay,
  type TrainingOverlayEntry,
  type TrainingOverlayKind,
  type TrainingOverlays,
} from "../data/trainingOverlays";
import type { Parsed } from "../data/trainingReader";

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

// ── the Training export ──────────────────────────────────────────────────────

/** A Training check's findings and what it read (null when it did not), so each file is parsed once. */
export interface TrainingChecked<T> {
  findings: PublicationFinding[];
  value: T | null;
}

/**
 * The Training manifest as the panel reads it.
 *
 * Training is checked on the OPPOSITE rule to the comparison picker above. There a bad category
 * empties the airport, so the check exists to stop that; here a bad entry is greyed out on its own
 * by design, which means a half-written export is easy to publish and never notice. The check is
 * what notices — naming the entry and the field.
 */
export function checkTrainingIndex(manifest: unknown): TrainingChecked<TrainingIndex> {
  const parsed = parseTrainingIndex(manifest);
  if (!parsed.ok) {
    return { findings: [{ level: "error", message: `training/index.json is not a manifest: ${parsed.problem}` }], value: null };
  }
  const findings = parsed.value.rejected.map((item) => ({
    level: "error" as const,
    category: item.id,
    message: `the panel would grey this entry out: ${item.problem}`,
  }));
  return { findings, value: parsed.value };
}

/**
 * A listed set the panel REFUSES by name — a superseded vocabulary, another spec, a prior set —
 * is a warning, not an error: it is refused on purpose, from the manifest alone, and deleting it
 * is a decision about data on disk, not about whether the panel loads.
 */
export function checkTrainingSetRefusal(entry: TrainingSetEntry): PublicationFinding[] {
  const refusal = trainingSetRefusal(entry);
  return refusal === null
    ? []
    : [{ level: "warn", category: entry.id, message: `listed, and refused by name: ${refusal}` }];
}

/** One readable set's sample file, through the panel's own reader. */
export function checkTrainingSample(setId: string, sample: unknown): TrainingChecked<TrainingSample> {
  const parsed = parseTrainingSample(sample);
  return parsed.ok
    ? { findings: [], value: parsed.value }
    : { findings: [{ level: "error", category: setId, message: `sample.json: ${parsed.problem}` }], value: null };
}

/**
 * What the manifest promises against what the sample holds. The two files are written by one run
 * of the exporter, so a disagreement means they came from different runs.
 */
export function checkTrainingSetAgrees(
  entry: TrainingSetEntry, sample: TrainingSample, airport: string,
): PublicationFinding[] {
  const findings: PublicationFinding[] = [];
  const pairs: Array<[string, string | number, string | number]> = [
    ["airport", airport, sample.airport],
    ["setId", entry.id, sample.setId],
    ["flights", entry.flights, sample.flights.length],
    ["vocabularySha256", entry.vocabularySha256, sample.vocabulary.specSha256],
    ["runwaySha256", entry.runwaySha256, sample.candidatesSha256],
    ["readingRule", entry.readingRule, sample.vocabulary.readingRule],
    ["cohort.split", entry.cohort.split, sample.cohort.split],
    ["cohort.perStratum", entry.cohort.perStratum, sample.cohort.perStratum],
    ["cohort.seed", entry.cohort.seed, sample.cohort.seed],
  ];
  for (const [what, listed, held] of pairs) {
    if (listed !== held) {
      findings.push({ level: "error", category: entry.id, message: `${what}: the manifest says ${listed}, sample.json says ${held}` });
    }
  }
  return findings;
}

// ── the Training overlays ────────────────────────────────────────────────────

/** The overlays manifest as the panel reads it: an entry it rejects is shown as a problem, so it is an error here. */
export function checkTrainingOverlays(manifest: unknown): TrainingChecked<TrainingOverlays> {
  const parsed = parseTrainingOverlays(manifest);
  if (!parsed.ok) {
    return { findings: [{ level: "error", message: `training/overlays.json is not a manifest: ${parsed.problem}` }], value: null };
  }
  const findings = parsed.value.rejected.map((item) => ({
    level: "error" as const, category: item.id, message: `the panel would reject this overlay: ${item.problem}`,
  }));
  return { findings, value: parsed.value };
}

/** The panel's reader for each kind of overlay: a kind added to the list needs its reader here, or this does not
 *  compile. */
const OVERLAY_READERS: Record<TrainingOverlayKind,
  (payload: unknown, entry: TrainingOverlayEntry, sample: TrainingSample) => Parsed<{ flights: unknown[] }>> = {
  "executor-replay": parseTrainingExecutorOverlay,
  "prior-prediction": parseTrainingPriorOverlay,
  "prior-generation": parseTrainingGenerationOverlay,
};

/**
 * One overlay's file through the panel's own reader, against the sample of the set it is drawn over — and that
 * sample's sha256 on disk against the one the overlay recorded: a set re-exported under its id since would otherwise
 * be matched by id alone.
 */
export function checkTrainingOverlay(
  entry: TrainingOverlayEntry, payload: unknown, sample: TrainingSample, sampleSha256: string,
): PublicationFinding[] {
  const findings: PublicationFinding[] = [];
  if (entry.baseSampleSha256 !== sampleSha256) {
    findings.push({
      level: "error", category: entry.id,
      message: `drawn over ${entry.base}'s sample ${entry.baseSampleSha256.slice(0, 12)}, the file on disk is ${sampleSha256.slice(0, 12)}`,
    });
  }
  const parsed = OVERLAY_READERS[entry.kind](payload, entry, sample);
  if (!parsed.ok) findings.push({ level: "error", category: entry.id, message: `${entry.file}: ${parsed.problem}` });
  else if (parsed.value.flights.length !== entry.flights) {
    findings.push({ level: "error", category: entry.id, message: `the manifest says ${entry.flights} flights, ${entry.file} holds ${parsed.value.flights.length}` });
  }
  return findings;
}
