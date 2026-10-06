/**
 * comparisonSource.ts
 * -------------------
 * Where the comparison layer reads its files from: a PUBLISHED CATEGORY (a directory of `public/data`, in the
 * Evaluate task) or a TRAFFIC JOB (the backend's file route, in the Optimize task). One renderer for both — a job's
 * directory has the same files as a published category (design §10.4, IM7): `comparison_index.json`, the CZML files
 * the index names, the evaluation report.
 */

import {
  airportComparisonCzmlUrl,
  airportComparisonIndexUrl,
  type ComparisonIndex,
} from "../data/airportData";
import type { WorkbenchMode } from "../context/AppContext";

/** The finished traffic job whose scene the Optimize task shows (`baseUrl` ends with `/`). */
export interface TrafficScene {
  airportCode: string;
  jobId: string;
  baseUrl: string;
}

export type ComparisonSource =
  | { kind: "category"; airportCode: string; categoryDir: string }
  | { kind: "job"; airportCode: string; jobId: string; baseUrl: string };

/** The index of a job: the one file a reader starts from (`traffic_job_files.INDEX_FILE`, the builder's own name). */
export const JOB_INDEX_FILE = "comparison_index.json";

export function comparisonIndexUrl(source: ComparisonSource): string {
  return source.kind === "category"
    ? airportComparisonIndexUrl(source.airportCode, source.categoryDir)
    : `${source.baseUrl}${JOB_INDEX_FILE}`;
}

export function comparisonCzmlUrl(source: ComparisonSource, file: string): string {
  return source.kind === "category"
    ? airportComparisonCzmlUrl(source.airportCode, source.categoryDir, file)
    : `${source.baseUrl}${file}`;
}

/** One string per source: a change of it reloads the layer; it also names the layer's Cesium data sources. */
export function comparisonSourceKey(source: ComparisonSource): string {
  return source.kind === "category" ? source.categoryDir : `job:${source.jobId}`;
}

export interface ComparisonSourceInputs {
  mode: WorkbenchMode;
  trajectoryComparison: boolean;
  trajectoryComparisonCategory: string | null;
  activeAirportCode: string;
  trafficScene: TrafficScene | null;
}

/**
 * The source the layer draws, or null when it draws nothing. The Evaluate task draws the selected published category
 * while the comparison is on; the Optimize task draws its traffic job's scene (of the active airport).
 */
export function comparisonSourceOf(inputs: ComparisonSourceInputs): ComparisonSource | null {
  const { mode, activeAirportCode, trafficScene } = inputs;
  if (!activeAirportCode) return null;
  if (mode === "evaluation" && inputs.trajectoryComparison && inputs.trajectoryComparisonCategory) {
    return { kind: "category", airportCode: activeAirportCode, categoryDir: inputs.trajectoryComparisonCategory };
  }
  if (mode === "optimize" && trafficScene && trafficScene.airportCode === activeAirportCode) {
    return { kind: "job", airportCode: trafficScene.airportCode, jobId: trafficScene.jobId, baseUrl: trafficScene.baseUrl };
  }
  return null;
}

/** M1 windows, drawn ONE at a time: an index of traffic windows (a `traffic` group), not a scene. */
export function isWindowIndex(index: ComparisonIndex): boolean {
  return index.scene === undefined && index.groups.some((group) => group.traffic !== undefined);
}

/**
 * The window a selection asks for: the selected flight's group, or null when the selection is none or none of the groups.
 * Only a group key changes the window drawn — a cleared selection (Reset view) leaves it as it is; the first group is the
 * window drawn until a selection asks for another.
 */
export function requestedWindow(groupKeys: readonly string[], selectedFlightId: string | null): string | null {
  return selectedFlightId !== null && groupKeys.includes(selectedFlightId) ? selectedFlightId : null;
}
