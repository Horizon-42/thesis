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

export type ComparisonResultSource = "prediction" | "experiment";

export const EXPERIMENT_PREDICTION_OUTPUTS = [
  "state",
  "control",
  "control-mixture",
] as const;
export type ExperimentPredictionOutput = typeof EXPERIMENT_PREDICTION_OUTPUTS[number];

export function isExperimentPredictionOutput(
  value: unknown,
): value is ExperimentPredictionOutput {
  return typeof value === "string" &&
    (EXPERIMENT_PREDICTION_OUTPUTS as readonly string[]).includes(value);
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
  model?: string | null;
  predictionOutput?: ExperimentPredictionOutput | null;
  horizonMode?: "normalized" | "full" | "window" | null;
  seed?: number | null;
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
  datasetSplit?: "train" | "val" | "test";
  /** Existing entries omit this and remain ordinary Prediction results. */
  resultSource?: ComparisonResultSource;
  /** Present only for categories published from the checkpoint experiment sweep. */
  experiment?: ExperimentCategoryMetadata;
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
      (experiment.model === undefined || experiment.model === null ||
        typeof experiment.model === "string") &&
      (experiment.predictionOutput === undefined || experiment.predictionOutput === null ||
        isExperimentPredictionOutput(experiment.predictionOutput)) &&
      (experiment.horizonMode === undefined || experiment.horizonMode === null ||
        experiment.horizonMode === "normalized" || experiment.horizonMode === "full" ||
        experiment.horizonMode === "window") &&
      (experiment.seed === undefined || experiment.seed === null ||
        typeof experiment.seed === "number"));
  return (
    typeof candidate.key === "string" &&
    typeof candidate.label === "string" &&
    typeof candidate.dir === "string" &&
    typeof candidate.constrained === "boolean" &&
    (candidate.datasetSplit === undefined ||
      candidate.datasetSplit === "train" ||
      candidate.datasetSplit === "val" ||
      candidate.datasetSplit === "test") &&
    (candidate.resultSource === undefined ||
      candidate.resultSource === "prediction" ||
      candidate.resultSource === "experiment") &&
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
  schemaVersion: "comparison-v2-generation";
  generation: string;
  epoch: string;
  startHidden: boolean;
  /** References reuse the airport's canonical observed datasource. */
  referenceSource: "canonicalObserved";
  /** Dataset partition for learned prediction categories. */
  datasetSplit?: "train" | "val" | "test";
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
    candidate.schemaVersion === "comparison-v2-generation" &&
    typeof candidate.generation === "string" &&
    typeof candidate.epoch === "string" &&
    typeof candidate.startHidden === "boolean" &&
    candidate.referenceSource === "canonicalObserved" &&
    (candidate.datasetSplit === undefined ||
      candidate.datasetSplit === "train" ||
      candidate.datasetSplit === "val" ||
      candidate.datasetSplit === "test") &&
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
