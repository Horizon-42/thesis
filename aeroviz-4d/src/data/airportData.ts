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

export function airportLandingsRunwayUrl(airportCode: string, runway: string): string {
  const code = normalizeAirportCode(airportCode);
  return airportDataUrl(code, `landings/${code}_${runway.toUpperCase()}.czml`);
}

// ── Optimizer-comparison trajectories (three-coloured: reference/optimizer/simulator) ─────
//
// Produced by `aeroviz-4d/python/build_scenario_comparison_czml.py` into
// `<airport>/comparison/`: one `comparison_<ICAO>_<RWY>.czml` per runway plus a single
// `comparison_index.json`. The index lets the frontend sample a subset of flight groups
// without loading every (large) per-runway CZML.

export function airportComparisonRootUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/comparison`;
}

/** Manifest of available optimization categories (ADS-B / runway target, ±constraints). */
export function airportComparisonCategoriesUrl(airportCode: string): string {
  return `${airportComparisonRootUrl(airportCode)}/categories.json`;
}

/** The index for one category's comparison data (one record per flight group). */
export function airportComparisonIndexUrl(airportCode: string, categoryDir: string): string {
  return `${airportComparisonRootUrl(airportCode)}/${categoryDir}/comparison_index.json`;
}

/** One runway's comparison CZML within a category, named as the index's `czml` field. */
export function airportComparisonCzmlUrl(
  airportCode: string,
  categoryDir: string,
  czmlFile: string,
): string {
  return `${airportComparisonRootUrl(airportCode)}/${categoryDir}/${czmlFile}`;
}

/** One optimization category, as listed in `comparison/categories.json`. */
export interface ComparisonCategory {
  /** Stable key, e.g. "asdb" / "runway" / "runwayConstrained". */
  key: string;
  /** Display label, e.g. "ADS-B target". */
  label: string;
  /** Subdirectory under `comparison/` holding this category's index + CZMLs. */
  dir: string;
  /** Number of flight groups in this category. */
  groups: number;
}

export interface ComparisonCategoriesManifest {
  categories: ComparisonCategory[];
}

export function isComparisonCategory(value: unknown): value is ComparisonCategory {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.key === "string" &&
    typeof candidate.label === "string" &&
    typeof candidate.dir === "string"
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
  status: "solved" | "failed";
  finalTimeS: number | null;
  initialState: ComparisonInitialState | null;
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
 * Optimization-run stats for one category, taken from its summary.json (not recomputed).
 * `solveRate` (converged / attempted) is known now; the rest are filled by the future
 * evaluation package.
 */
export interface OptimizationStats {
  total?: number | null;
  solved?: number | null;
  failed?: number | null;
  solveRate?: number | null;
  successRate?: number | null;
  avgStateErrorM?: number | null;
  avgTimeS?: number | null;
}

export interface ComparisonIndex {
  epoch: string;
  startHidden: boolean;
  groups: ComparisonGroup[];
  optimization?: OptimizationStats;
}

export function isComparisonGroup(value: unknown): value is ComparisonGroup {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.group === "string" &&
    typeof candidate.flightId === "string" &&
    typeof candidate.runway === "string" &&
    typeof candidate.czml === "string" &&
    (candidate.status === "solved" || candidate.status === "failed") &&
    Array.isArray(candidate.entities) &&
    candidate.entities.every((entity) => typeof entity === "string")
  );
}

export function isComparisonIndex(value: unknown): value is ComparisonIndex {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.epoch === "string" &&
    Array.isArray(candidate.groups) &&
    candidate.groups.every(isComparisonGroup)
  );
}

/** One runway's landing CZML, as listed in landings/index.json. */
export interface LandingRunwayEntry {
  runway: string;
  /** Path relative to the airport folder, e.g. "landings/KRDU_23R.czml" */
  file: string;
  count: number;
}

export interface LandingsManifest {
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
    typeof candidate.airport === "string" &&
    typeof candidate.combined === "string" &&
    Array.isArray(candidate.runways) &&
    candidate.runways.every(isLandingRunwayEntry)
  );
}
