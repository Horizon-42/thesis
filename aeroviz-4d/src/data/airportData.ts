export interface AirportConfig {
  code: string;
  lon: number;
  lat: number;
  /** Initial camera altitude/range in metres */
  height: number;
}

export interface AirportCatalogItem {
  code: string;
  name: string;
  lon: number;
  lat: number;
}

export interface AirportsIndexManifest {
  defaultAirport: string;
  airports: AirportCatalogItem[];
}

export const DATA_ROOT = "/data";
export const COMMON_DATA_ROOT = `${DATA_ROOT}/common`;
export const AIRPORTS_DATA_ROOT = `${DATA_ROOT}/airports`;
export const AIRPORTS_INDEX_URL = `${AIRPORTS_DATA_ROOT}/index.json`;

export function normalizeAirportCode(code: string): string {
  return code.trim().toUpperCase();
}

export function commonDataUrl(fileName: string): string {
  return `${COMMON_DATA_ROOT}/${fileName}`;
}

export function airportDataRootUrl(airportCode: string): string {
  return `${AIRPORTS_DATA_ROOT}/${normalizeAirportCode(airportCode)}`;
}

export function airportDataUrl(airportCode: string, fileName: string): string {
  return `${airportDataRootUrl(airportCode)}/${fileName}`;
}

export function airportProcedureDetailsRootUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/procedure-details`;
}

export function airportProcedureDetailsIndexUrl(airportCode: string): string {
  return `${airportProcedureDetailsRootUrl(airportCode)}/index.json`;
}

export function airportProcedureDetailUrl(airportCode: string, procedureUid: string): string {
  return `${airportProcedureDetailsRootUrl(airportCode)}/${procedureUid}.json`;
}

export function airportChartsRootUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/charts`;
}

export function airportChartsIndexUrl(airportCode: string): string {
  return `${airportChartsRootUrl(airportCode)}/index.json`;
}

export function airportLocalTerrainUrl(airportCode: string, fileName?: string): string {
  const root = `${airportDataRootUrl(airportCode)}/local-terrain/heightmap`;
  return fileName ? `${root}/${fileName}` : root;
}

export function airportLandingsIndexUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/landings/index.json`;
}

// ── Prediction-comparison trajectories (three-coloured: reference/optimizer/simulator) ───
//
// Produced by `aeroviz-4d/python/build_scenario_comparison_czml.py` into
// `<airport>/comparison/`: one result CZML per runway plus a single
// `comparison_index.json`. References are selected from the canonical observed datasource;
// the index lets the frontend sample groups without loading every result file.

export function airportComparisonRootUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/comparison`;
}

/** Manifest of available observed, optimization and data-driven evaluation categories. */
export function airportComparisonCategoriesUrl(airportCode: string): string {
  return `${airportComparisonRootUrl(airportCode)}/categories.json`;
}

/** The index for one category's comparison data (one record per flight group). */
export function airportComparisonIndexUrl(airportCode: string, categoryDir: string): string {
  return `${airportComparisonRootUrl(airportCode)}/${categoryDir}/comparison_index.json`;
}

/** The category's immutable evaluation report named by its committed comparison index. */
export function airportEvaluationReportUrl(
  airportCode: string,
  categoryDir: string,
  reportFile: string,
): string {
  return `${airportComparisonRootUrl(airportCode)}/${categoryDir}/${reportFile}`;
}

/** One runway's comparison CZML within a category, named as the index's `czml` field. */
export function airportComparisonCzmlUrl(
  airportCode: string,
  categoryDir: string,
  czmlFile: string,
): string {
  return `${airportComparisonRootUrl(airportCode)}/${categoryDir}/${czmlFile}`;
}

/**
 * The measured-baseline category (`trajectory_data_process/harvest/publish.py`).
 * Report-only — it ships no CZML, because the flown track it describes is already the
 * observed layer on screen. Named here so the frontend's baseline report readers and
 * the publisher cannot drift.
 */
export const OBSERVED_CATEGORY_KEY = "observed";
export const OBSERVED_EVALUATION_REPORT_FILE = "evaluation_report.json";

/**
 * Where a comparison category's trajectories came from. Exported (like the two `EXPERIMENT_*`
 * lists below) so the guards and `utils/checkPublication` read ONE list rather than each
 * spelling the union again.
 */
export const COMPARISON_RESULT_SOURCES = ["prediction", "experiment"] as const;
export type ComparisonResultSource = typeof COMPARISON_RESULT_SOURCES[number];

/**
 * Dataset partition of a learned-prediction category and of its comparison index. "dayval" is a day
 * partition's validation days, flown by a checkpoint trained on that partition's training days only
 * (`ts_transformer.run_naming.SPLIT_DAYVAL`) — not the checkpoint's own val, not the sealed test.
 * MIRRORED by `python/build_scenario_comparison_czml.py` `DATASET_SPLITS`.
 */
export const DATASET_SPLITS = ["train", "val", "test", "dayval"] as const;
export type DatasetSplit = typeof DATASET_SPLITS[number];

/** The comparison-index schema this app reads; an index of any other version is rejected. */
export const COMPARISON_INDEX_SCHEMA_VERSION = "comparison-v2-generation" as const;

/**
 * MUST match `4dTrajectory/ts_transformer/config.py::PREDICTION_OUTPUTS` — an unlisted value
 * fails `isComparisonCategory`, and one failing category empties the whole airport's manifest.
 * (`control-mixture` was listed here but no producer ever emitted it; `closure` was emitted
 * and not listed, which is what took the KRDU picker down; `plan` did the same on 2026-09-12.
 * Pinned by `ts_transformer/tests/test_frontend_mirrors.py`.)
 */
export const EXPERIMENT_PREDICTION_OUTPUTS = [
  "state",
  "control",
  "closure",
  "plan",
] as const;
export type ExperimentPredictionOutput = typeof EXPERIMENT_PREDICTION_OUTPUTS[number];

export function isExperimentPredictionOutput(
  value: unknown,
): value is ExperimentPredictionOutput {
  return typeof value === "string" &&
    (EXPERIMENT_PREDICTION_OUTPUTS as readonly string[]).includes(value);
}

/**
 * MUST match `4dTrajectory/ts_transformer/config.py::HORIZON_MODES` — the same validator, the
 * same `.every`, the same empty picker. Pinned by `ts_transformer/tests/test_frontend_mirrors.py`.
 */
export const EXPERIMENT_HORIZON_MODES = [
  "normalized",
  "full",
  "window",
] as const;
export type ExperimentHorizonMode = typeof EXPERIMENT_HORIZON_MODES[number];

export function isExperimentHorizonMode(value: unknown): value is ExperimentHorizonMode {
  return typeof value === "string" &&
    (EXPERIMENT_HORIZON_MODES as readonly string[]).includes(value);
}

/**
 * One named parameter of a published run — `run_naming.run_parameter_rows`, stamped by
 * `publish_ts_experiment_trajectories.py`. `section` groups the rows in the order they arrive
 * (`Model` first, then the loss edits, then the setting sections); a settings row's `name` IS
 * the config field, a `Model` row names the field it chiefly reads as `field`.
 */
export interface ExperimentParameterRow {
  section: string;
  name: string;
  value: string;
  field?: string;
}

/**
 * Why a published run exists — the publisher stamps it from the tracked registry
 * `4dTrajectory/ts_transformer/docs/experiments/intents.json` and refuses to publish without it.
 */
export interface ExperimentIntent {
  /** Short heading of the picker group (the campaign). */
  groupTitle: string;
  /** What the campaign asks, and against what comparator. */
  group: string;
  /** What this run changes relative to its comparator. */
  run: string;
  /** What this predict-time variant does differently, where the registry says. */
  variant?: string;
  /** The design/result document the campaign is defined in. */
  design?: string;
}

export function isExperimentParameterRow(value: unknown): value is ExperimentParameterRow {
  if (!value || typeof value !== "object") return false;
  const row = value as Record<string, unknown>;
  return typeof row.section === "string" && typeof row.name === "string" &&
    typeof row.value === "string" && (row.field === undefined || typeof row.field === "string");
}

export function isExperimentIntent(value: unknown): value is ExperimentIntent {
  if (!value || typeof value !== "object") return false;
  const intent = value as Record<string, unknown>;
  return typeof intent.groupTitle === "string" && typeof intent.group === "string" &&
    typeof intent.run === "string" &&
    (intent.variant === undefined || typeof intent.variant === "string") &&
    (intent.design === undefined || typeof intent.design === "string");
}

export interface ExperimentCategoryMetadata {
  /** Stable repository-relative run identity (without the checkpoint filename). */
  id: string;
  /** Campaign/collection used to group models in the experiment selector. */
  group: string;
  /** Repository-relative checkpoint path; presentation/provenance only. */
  checkpoint: string;
  /**
   * Canonical self-describing display name (the `run_naming` grammar:
   * output · backbone · dynamics · loss · meta). Publishes that predate it omit it and
   * the picker falls back to composing a label from the fields below.
   */
  label?: string | null;
  /**
   * The structured form (publishes from 2026-09-12 on; older ones carry only `label` until the
   * publisher's `--refresh-labels-only` restamps them): the run id, which records these are
   * (an anytime bin, a predict-time hook …), every parameter as a named row, and the intent.
   */
  runName?: string | null;
  variantLabel?: string | null;
  parameters?: ExperimentParameterRow[] | null;
  intent?: ExperimentIntent | null;
  model?: string | null;
  predictionOutput?: ExperimentPredictionOutput | null;
  horizonMode?: ExperimentHorizonMode | null;
  seed?: number | null;
}

/** mean/p95 of one accuracy metric, in metres. */
export interface CategoryAccuracyMetric {
  mean?: number | null;
  p95?: number | null;
}

/**
 * Compact per-category prediction accuracy (subset of the comparison index's
 * `prediction` block, stamped into `categories.json` so a split's results can be
 * RANKED without fetching every category's full index). Present only on
 * learned-prediction categories; older publishes were backfilled 2026-08-24.
 */
export interface CategoryAccuracy {
  adeM?: CategoryAccuracyMetric | null;
  fdeM?: CategoryAccuracyMetric | null;
}

/** One evaluation category, as listed in `comparison/categories.json`. */
export interface ComparisonCategory {
  /** Stable key, e.g. "asdb" / "runway" / "runway_cons". */
  key: string;
  /** Display label, e.g. "ADS-B target". */
  label: string;
  /** Subdirectory under `comparison/` holding this category's index + CZMLs. */
  dir: string;
  /** Number of flight groups in this category. */
  groups: number;
  /**
   * Whether this category's solves enforce the runway's RNAV(GPS) procedure as per-leg
   * NLP path constraints. An EXPLICIT manifest field (stamped by the pipeline's
   * `--constrained`), never derived from the key/dir spelling — a `_cons` suffix is a
   * naming convention, not a contract.
   */
  constrained: boolean;
  /** Dataset partition for learned prediction categories; absent for optimization/baselines. */
  datasetSplit?: DatasetSplit;
  /** Existing entries omit this and remain ordinary Prediction results. */
  resultSource?: ComparisonResultSource;
  /** Present only for categories published from the checkpoint experiment sweep. */
  experiment?: ExperimentCategoryMetadata;
  /** Batch mean/p95 ADE/FDE for ranking results; prediction categories only. */
  accuracy?: CategoryAccuracy | null;
}

/** Report-only categories have no comparison groups or CZML to draw. */
export function isDrawableComparisonCategory(
  category: ComparisonCategory,
): boolean {
  return category.groups > 0;
}

export interface ComparisonCategoriesManifest {
  categories: ComparisonCategory[];
}

export function isComparisonCategory(value: unknown): value is ComparisonCategory {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const experiment = candidate.experiment as Record<string, unknown> | undefined;
  const validExperiment =
    experiment === undefined ||
    (typeof experiment === "object" &&
      experiment !== null &&
      typeof experiment.id === "string" &&
      typeof experiment.group === "string" &&
      typeof experiment.checkpoint === "string" &&
      (experiment.label === undefined || experiment.label === null ||
        typeof experiment.label === "string") &&
      (experiment.runName === undefined || experiment.runName === null ||
        typeof experiment.runName === "string") &&
      (experiment.variantLabel === undefined || experiment.variantLabel === null ||
        typeof experiment.variantLabel === "string") &&
      (experiment.parameters === undefined || experiment.parameters === null ||
        (Array.isArray(experiment.parameters) &&
          experiment.parameters.every(isExperimentParameterRow))) &&
      (experiment.intent === undefined || experiment.intent === null ||
        isExperimentIntent(experiment.intent)) &&
      (experiment.model === undefined || experiment.model === null ||
        typeof experiment.model === "string") &&
      (experiment.predictionOutput === undefined || experiment.predictionOutput === null ||
        isExperimentPredictionOutput(experiment.predictionOutput)) &&
      (experiment.horizonMode === undefined || experiment.horizonMode === null ||
        isExperimentHorizonMode(experiment.horizonMode)) &&
      (experiment.seed === undefined || experiment.seed === null ||
        typeof experiment.seed === "number"));
  return (
    typeof candidate.key === "string" &&
    typeof candidate.label === "string" &&
    typeof candidate.dir === "string" &&
    typeof candidate.constrained === "boolean" &&
    typeof candidate.groups === "number" &&
    (candidate.datasetSplit === undefined ||
      (DATASET_SPLITS as readonly unknown[]).includes(candidate.datasetSplit)) &&
    (candidate.resultSource === undefined ||
      (COMPARISON_RESULT_SOURCES as readonly unknown[]).includes(candidate.resultSource)) &&
    (candidate.accuracy === undefined ||
      candidate.accuracy === null ||
      typeof candidate.accuracy === "object") &&
    validExperiment &&
    (candidate.resultSource !== "experiment" || experiment !== undefined)
  );
}

export function isComparisonCategoriesManifest(value: unknown): value is ComparisonCategoriesManifest {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return Array.isArray(candidate.categories) && candidate.categories.every(isComparisonCategory);
}

export interface ComparisonInitialState {
  lat: number;
  lon: number;
  alt: number;
  V: number;
  psi: number;
  gamma: number;
  /** Optimizer aircraft mass in kg (added 2026-07; absent in indexes generated before then). */
  m?: number;
}

/** One flight's comparison group: the entity ids of its (up to) three coloured paths. */
export interface ComparisonGroup {
  /** Unique group key, `${flightId}_${runway}`. */
  group: string;
  flightId: string;
  runway: string;
  airport: string;
  /**
   * `solved` = optimized (and, when the run was evaluated, inside the gates);
   * `offTarget` = optimized but the terminal verdict failed;
   * `indeterminate` = solved, but the evaluation cannot support pass or fail.
   * (yellow reference; added 2026-07 — absent in older indexes);
   * `failed` = no solution (dark-red reference only).
   */
  status: "solved" | "offTarget" | "indeterminate" | "failed";
  terminalVerdict?: "pass" | "fail" | "indeterminate" | null;
  finalTimeS: number | null;
  initialState: ComparisonInitialState | null;
  /**
   * Final-state deviations from the evaluation report (present when the run was
   * evaluated; added 2026-07). `lateralErrM` = horizontal miss distance;
   * `verticalErrM` = signed altitude miss (+ = high).
   */
  lateralErrM?: number | null;
  verticalErrM?: number | null;
  /**
   * Per-flight facts present on EVERY record (solved + failed), from the flight's scenario
   * initial state — so the flight list can show V + mass even for failed optimizations.
   * (Added 2026-07; absent in indexes generated before then.)
   */
  initialVMps?: number | null;
  massKg?: number | null;
  /** CZML entity ids belonging to this group (e.g. ref-/opt-/sim-`${group}`). */
  entities: string[];
  /** The CZML file (within `comparison/`) that holds this group's entities. */
  czml: string;
}

/**
 * Optimization-run stats for one category: solve counts from the run's summary.json,
 * plus — when the run was evaluated — the evaluation report's batch metrics
 * (`successRate` = inside-the-gates / total; `avgStateErrorM` = mean final lateral
 * deviation over solved flights; `avgTimeS` = mean optimized flight time).
 */
export interface OptimizationStats {
  total?: number | null;
  solved?: number | null;
  failed?: number | null;
  solveRate?: number | null;
  successful?: number | null;
  successRate?: number | null;
  avgStateErrorM?: number | null;
  avgTimeS?: number | null;
}

export interface PredictionErrorSpread {
  median?: number | null;
  mean?: number | null;
  p95?: number | null;
  max?: number | null;
}

export interface PredictionFinalTimeStats {
  mae?: number | null;
  p95Abs?: number | null;
  meanSigned?: number | null;
}

export interface PredictionRawKinematicStats {
  positionVelocityRmseMps?: PredictionErrorSpread | null;
  headingConsistencyP95Deg?: PredictionErrorSpread | null;
  turnRateP95DegS?: PredictionErrorSpread | null;
  accelerationP95Mps2?: PredictionErrorSpread | null;
  jerkP95Mps3?: PredictionErrorSpread | null;
}

export interface PredictionRawKinematics {
  predicted?: PredictionRawKinematicStats | null;
  observedBaseline?: PredictionRawKinematicStats | null;
  delta?: Record<string, number> | null;
}

/** ADE/FDE summary published from a ts_transformer's `summary.json.accuracy` block. */
export interface PredictionAccuracyStats {
  flights?: number | null;
  finalTimeS?: PredictionFinalTimeStats | null;
  adeM?: PredictionErrorSpread | null;
  fdeM?: PredictionErrorSpread | null;
  arrivalEndpointErrorM?: PredictionErrorSpread | null;
  crossTrackP95M?: PredictionErrorSpread | null;
  altitudeP95M?: PredictionErrorSpread | null;
  rawKinematics?: PredictionRawKinematics | null;
}

export interface EvaluationBatchStats {
  schemaVersion?: string | null;
  total?: number | null;
  measured?: number | null;
  solved?: number | null;
  solveRate?: number | null;
  successful?: number | null;
  successRate?: number | null;
  failed?: number | null;
  indeterminate?: number | null;
  verdictCounts?: Record<string, number> | null;
  lateralM?: Record<string, number> | null;
  verticalM?: Record<string, number> | null;
  finalTimeS?: Record<string, number> | null;
  observed?: Record<string, number> | null;
}

export interface ComparisonIndex {
  schemaVersion: typeof COMPARISON_INDEX_SCHEMA_VERSION;
  generation: string;
  epoch: string;
  startHidden: boolean;
  /** References reuse the airport's canonical observed datasource. */
  referenceSource: "canonicalObserved";
  /** Dataset partition for learned prediction categories. */
  datasetSplit?: DatasetSplit;
  groups: ComparisonGroup[];
  optimization?: OptimizationStats;
  /** Complete batch-level evaluation statistics; excludes only per-flight details. */
  evaluation?: EvaluationBatchStats;
  /** Present only for data-driven prediction categories. */
  prediction?: PredictionAccuracyStats;
  /** Immutable report artifact committed by this same index generation. */
  evaluationReport: string;
}

export function isComparisonGroup(value: unknown): value is ComparisonGroup {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.group === "string" &&
    typeof candidate.flightId === "string" &&
    typeof candidate.runway === "string" &&
    typeof candidate.czml === "string" &&
    (candidate.status === "solved" ||
      candidate.status === "offTarget" ||
      candidate.status === "indeterminate" ||
      candidate.status === "failed") &&
    Array.isArray(candidate.entities) &&
    candidate.entities.every((entity) => typeof entity === "string")
  );
}

export function isComparisonIndex(value: unknown): value is ComparisonIndex {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.schemaVersion === COMPARISON_INDEX_SCHEMA_VERSION &&
    typeof candidate.generation === "string" &&
    typeof candidate.epoch === "string" &&
    typeof candidate.startHidden === "boolean" &&
    candidate.referenceSource === "canonicalObserved" &&
    (candidate.datasetSplit === undefined ||
      (DATASET_SPLITS as readonly unknown[]).includes(candidate.datasetSplit)) &&
    typeof candidate.evaluationReport === "string" &&
    Array.isArray(candidate.groups) &&
    candidate.groups.every(isComparisonGroup)
  );
}

/** One runway selector entry in landings/index.json. */
export interface LandingRunwayEntry {
  runway: string;
  /** Path relative to the airport folder; v2 entries share "trajectories.czml". */
  file: string;
  count: number;
}

export interface LandingsManifest {
  schemaVersion: "observed-landings-v2-canonical";
  airport: string;
  /** Combined (all-runway) CZML file name, e.g. "trajectories.czml" */
  combined: string;
  runways: LandingRunwayEntry[];
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function isAirportConfig(value: unknown): value is AirportConfig {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.code === "string" &&
    isFiniteNumber(candidate.lon) &&
    isFiniteNumber(candidate.lat) &&
    isFiniteNumber(candidate.height)
  );
}

export function isAirportCatalogItem(value: unknown): value is AirportCatalogItem {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.code === "string" &&
    typeof candidate.name === "string" &&
    isFiniteNumber(candidate.lon) &&
    isFiniteNumber(candidate.lat)
  );
}

export function isAirportsIndexManifest(value: unknown): value is AirportsIndexManifest {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.defaultAirport === "string" &&
    Array.isArray(candidate.airports) &&
    candidate.airports.every(isAirportCatalogItem)
  );
}

export function sortAirportCatalog(airports: AirportCatalogItem[]): AirportCatalogItem[] {
  return [...airports].sort((left, right) => left.code.localeCompare(right.code));
}

function isLandingRunwayEntry(value: unknown): value is LandingRunwayEntry {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.runway === "string" &&
    typeof candidate.file === "string" &&
    isFiniteNumber(candidate.count)
  );
}

export function isLandingsManifest(value: unknown): value is LandingsManifest {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    candidate.schemaVersion === "observed-landings-v2-canonical" &&
    typeof candidate.airport === "string" &&
    typeof candidate.combined === "string" &&
    Array.isArray(candidate.runways) &&
    candidate.runways.every(
      (entry) =>
        isLandingRunwayEntry(entry) &&
        entry.file === candidate.combined,
    )
  );
}
