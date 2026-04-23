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

export function airportDsmRootUrl(airportCode: string): string {
  return `${airportDataRootUrl(airportCode)}/dsm`;
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

export function airportDsmSourceUrl(airportCode: string, fileName?: string): string {
  const root = `${airportDsmRootUrl(airportCode)}/source`;
  return fileName ? `${root}/${fileName}` : root;
}

export function airportDsmHeightmapTerrainUrl(airportCode: string, fileName?: string): string {
  const root = `${airportDsmRootUrl(airportCode)}/heightmap-terrain`;
  return fileName ? `${root}/${fileName}` : root;
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
