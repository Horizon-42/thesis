/**
 * trafficJobs.ts
 * --------------
 * The Optimize task's multi-aircraft jobs, as the backend serves them (`aeroviz_backend/traffic_jobs.py`; design
 * `4dTrajectory/docs/multi_aircraft_optimization/design.md` §10.3):
 *
 *   GET  /traffic/arrivals?airport=&date=        the arrivals that land on a UTC day (the roster only)
 *   POST /traffic/jobs                            {mode: "m1", airport, flightKey} | {mode: "m2", airport,
 *                                                  blockStartUtc, blockS} → {jobId}   (409 while a job runs)
 *   GET  /traffic/jobs/<id>                       {state, progress: {done, total, current}, error[, summary]}
 *   GET  /traffic/jobs/<id>/files/<name>          a file of the job's comparison index
 *   POST /traffic/jobs/<id>/cancel
 */

import { fetchJson } from "../utils/fetchJson";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";

export type TrafficJobMode = "m1" | "m2";

/** The block lengths a job takes, in seconds — a MIRROR of `traffic_job_files.BLOCK_LENGTHS_S` (pinned). */
export const TRAFFIC_BLOCK_LENGTHS_S = [900, 1800, 3600] as const;
export type TrafficBlockLengthS = typeof TRAFFIC_BLOCK_LENGTHS_S[number];
export const DEFAULT_TRAFFIC_BLOCK_S: TrafficBlockLengthS = 1800;

/** How often the panel asks a running job how far it is. */
export const TRAFFIC_JOB_POLL_MS = 2000;
/** The longest wait between two polls that failed (the wait doubles from `TRAFFIC_JOB_POLL_MS` up to this). */
export const TRAFFIC_JOB_RETRY_MAX_MS = 15_000;

/**
 * The solver settings of every job, shown read-only (design IM2): the batch defaults. A MIRROR of
 * `optimization_run_config.DEFAULT_MAX_DURATION_S`, `DEFAULT_ROLLOUT_DT_S`, `scenario_optimization.DEFAULT_MAX_ITERATIONS`
 * and the fields of `traffic.loop.LoopSettings`, pinned by `aeroviz-4d/python/tests/test_traffic_mirrors.py`.
 */
export const TRAFFIC_JOB_SETTINGS = {
  maxDurationS: 2000,
  rolloutDtS: 0.5,
  maxIterations: 3000,
  stepS: 1,
  rowWindowS: 60,
  margin: 0.01,
  maxRounds: 5,
} as const;

export interface TrafficArrival {
  flightKey: string;
  /** Null for a flight the roster has no callsign of: shown as the first field of its flight key (`arrivalCallsign`). */
  callsign: string | null;
  runway: string;
  /** The ICAO type, or null when the identity resolver cannot type the aircraft. */
  type: string | null;
  entryUtc: string;
  landingUtc: string;
}

/** The name a flight is shown by: its callsign, else the first field of its flight key (`id_runway_icao24_landingTime`). */
export function arrivalCallsign(arrival: Pick<TrafficArrival, "callsign" | "flightKey">): string {
  return arrival.callsign ?? arrival.flightKey.split("_")[0];
}

export interface TrafficArrivals {
  airport: string;
  date: string;
  arrivals: TrafficArrival[];
}

export type TrafficJobState = "running" | "done" | "failed" | "cancelled";

export interface TrafficJobProgress {
  done: number;
  /** Null until the job has chosen its aircraft (it first reads the traffic). */
  total: number | null;
  /** The `flight_key` of the aircraft last finished, null before the first. */
  current: string | null;
}

/** What an M2 job leaves after the block's final check, per reading: the flown aircraft with a loss in it. */
export interface TrafficFinalCheck {
  answered: number;
  not_answered: number;
}

/** The readout of a finished job (`traffic.readout`): the part the panel shows beyond the comparison index. */
export interface TrafficJobSummary {
  /** M2: the arrivals the block controls (those with a dynamics model) ... */
  aircraft?: number;
  /** ... and those that stay their records (no dynamics model). */
  skipped_no_dynamics?: number;
  flown_aircraft_with_a_loss_left_after_the_block?: { visual: TrafficFinalCheck; ifr: TrafficFinalCheck };
  [field: string]: unknown;
}

export interface TrafficJobStatus {
  state: TrafficJobState;
  progress: TrafficJobProgress;
  error: string | null;
  /** Once `done`. */
  summary?: TrafficJobSummary;
}

export type TrafficJobRequest =
  | { mode: "m1"; airport: string; flightKey: string }
  | { mode: "m2"; airport: string; blockStartUtc: string; blockS: TrafficBlockLengthS };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFinalCheck(value: unknown): value is TrafficFinalCheck {
  return isRecord(value) && typeof value.answered === "number" && typeof value.not_answered === "number";
}

export function isTrafficArrivals(value: unknown): value is TrafficArrivals {
  return (
    isRecord(value) &&
    typeof value.airport === "string" &&
    typeof value.date === "string" &&
    Array.isArray(value.arrivals) &&
    value.arrivals.every(
      (a) =>
        isRecord(a) &&
        typeof a.flightKey === "string" &&
        (a.callsign === null || typeof a.callsign === "string") &&
        typeof a.runway === "string" &&
        (a.type === null || typeof a.type === "string") &&
        typeof a.entryUtc === "string" &&
        typeof a.landingUtc === "string",
    )
  );
}

export function isTrafficJobStatus(value: unknown): value is TrafficJobStatus {
  if (!isRecord(value) || !isRecord(value.progress)) return false;
  const progress = value.progress;
  const left = isRecord(value.summary) ? value.summary.flown_aircraft_with_a_loss_left_after_the_block : undefined;
  return (
    (value.state === "running" || value.state === "done" || value.state === "failed" || value.state === "cancelled") &&
    typeof progress.done === "number" &&
    (progress.total === null || typeof progress.total === "number") &&
    (progress.current === null || typeof progress.current === "string") &&
    (value.error === null || typeof value.error === "string") &&
    (value.summary === undefined ||
      (isRecord(value.summary) &&
        (value.summary.aircraft === undefined || typeof value.summary.aircraft === "number") &&
        (value.summary.skipped_no_dynamics === undefined || typeof value.summary.skipped_no_dynamics === "number") &&
        (left === undefined || (isRecord(left) && isFinalCheck(left.visual) && isFinalCheck(left.ifr)))))
  );
}

function readError(value: unknown): string | null {
  return isRecord(value) && value.ok === false && typeof value.error === "string" ? value.error : null;
}

async function post(path: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${AEROVIZ_BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    // Let a cancel outlive a same-tick unmount (best effort).
    keepalive: true,
  });
  const data = (await response.json()) as unknown;
  if (!response.ok) throw new Error(readError(data) ?? `AeroViz backend returned ${response.status}`);
  return data;
}

/** The arrivals of `airport` that land on the UTC `date` (`YYYY-MM-DD`), by landing time. */
export async function fetchTrafficArrivals(airport: string, date: string): Promise<TrafficArrivals> {
  const url = `${AEROVIZ_BACKEND_URL}/traffic/arrivals?airport=${encodeURIComponent(airport)}&date=${encodeURIComponent(date)}`;
  const data = await fetchJson<unknown>(url);
  if (!isTrafficArrivals(data)) throw new Error(`${url} is not a traffic arrivals list`);
  return data;
}

/** Start a job; rejects with the backend's reason (409: another job runs; 400: the request cannot be served). */
export async function startTrafficJob(request: TrafficJobRequest): Promise<string> {
  const data = await post("/traffic/jobs", request);
  if (!isRecord(data) || typeof data.jobId !== "string") throw new Error("AeroViz backend started a job without a jobId");
  return data.jobId;
}

export async function fetchTrafficJob(jobId: string): Promise<TrafficJobStatus> {
  const url = `${AEROVIZ_BACKEND_URL}/traffic/jobs/${jobId}`;
  const data = await fetchJson<unknown>(url);
  if (!isTrafficJobStatus(data)) throw new Error(`${url} is not a traffic job status`);
  return data;
}

export async function cancelTrafficJob(jobId: string): Promise<TrafficJobStatus> {
  const data = await post(`/traffic/jobs/${jobId}/cancel`, {});
  if (!isTrafficJobStatus(data)) throw new Error("AeroViz backend answered a cancel with something else than a job status");
  return data;
}

/**
 * Cancel on page unload (`navigator.sendBeacon` is delivered as the page goes away; a closed tab would otherwise leave the
 * job running). No body: a "simple" POST that needs no CORS preflight.
 */
export function beaconCancelTrafficJob(jobId: string): void {
  if (typeof navigator === "undefined" || !navigator.sendBeacon) return;
  navigator.sendBeacon(`${AEROVIZ_BACKEND_URL}/traffic/jobs/${jobId}/cancel`);
}

/** Where the comparison layer reads a job's files (`ComparisonSource`, a trailing slash). */
export function trafficJobFilesUrl(jobId: string): string {
  return `${AEROVIZ_BACKEND_URL}/traffic/jobs/${jobId}/files/`;
}
