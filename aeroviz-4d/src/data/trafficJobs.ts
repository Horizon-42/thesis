/**
 * trafficJobs.ts
 * --------------
 * The Optimize task's multi-aircraft jobs, as the backend serves them (`aeroviz_backend/traffic_jobs.py`; design
 * `4dTrajectory/docs/multi_aircraft_optimization/design.md` §10.3):
 *
 *   GET  /traffic/arrivals?airport=&date=        the arrivals that land on a UTC day (the roster only; the Scene time
 *                                                  readout asks it for a flight's entry)
 *   GET  /traffic/scenarios?airport=             the airport's scenario list (design §10.6): the M1 arrivals and the M2
 *                                                  blocks of the census, written offline by `traffic_scenarios.py`
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

/** The schema of the scenario catalog — a MIRROR of `traffic_job_files.CATALOG_SCHEMA` (pinned). */
export const TRAFFIC_CATALOG_SCHEMA = "traffic-scenario-catalog-v2";

/** The phase shown until the job's first answer — a MIRROR of `traffic_job_files.PHASE_STARTING` (pinned): the backend says it too. */
export const TRAFFIC_PHASE_STARTING = "starting";

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

/**
 * One row of the M1 list: an arrival with a loss it answers for in its own record (`traffic_scenarios.build_catalog`'s
 * `m1`). `kinds` are the judge's raw kinds (`utils/trafficScenarios.ts` names them); `tightest` is the closest approach
 * as a share of the required minimum (1 = at the minimum), over the losses it answers for.
 */
export interface TrafficScenarioArrival extends TrafficArrival {
  /** The CWT wake category of its type (`I` is the lightest), null for a type without one. */
  category: string | null;
  lossInstants: number;
  kinds: string[];
  tightest: number;
  recordedAircraft: number;
}

/** One row of the M2 list: a block (aligned to its length) in which at least one arrival lands (`build_catalog`'s `m2`). */
export interface TrafficScenarioBlock {
  startUtc: string;
  arrivals: number;
  /** The arrivals with an aircraft dynamics model: the ones a job controls. */
  commandable: number;
  lossInstants: number;
  runways: string[];
}

/** An airport's scenario catalog, as `traffic_scenarios.py` writes it and the backend serves it. */
export interface TrafficScenarioCatalog {
  schema: typeof TRAFFIC_CATALOG_SCHEMA;
  airport: string;
  writtenUtc: string;
  config: {
    /** The census check step (s). */
    stepS: number;
    /** Set when only the first N arrivals by landing time were judged (a timing smoke); null for a whole roster. */
    limit: number | null;
  };
  counts: { arrivals: number; judged: number; withLoss: number };
  m1: TrafficScenarioArrival[];
  /** By block length in seconds (`TRAFFIC_BLOCK_LENGTHS_S`), as text. */
  m2: Record<string, TrafficScenarioBlock[]>;
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
  /** What the job is doing, in the writer's plain words (`traffic_job_files.PHASE_*`: "reading traffic", "earliest arrival of each aircraft", "schedule", "optimizing k of n", "evaluation", "building the scene"): shown as it is. */
  phase: string;
}

/**
 * What changed for one aircraft the job was to control (`state.json` `perAircraft`, `traffic_job_files`): its type, the loss
 * instants it answers for in its first solve and in the solve its loop kept as the record (the census's kind of count, as the
 * list's `lossInstants`), its record's landing against the recorded flight's (s, plus: later), its slot's delay (M2), its losses
 * in the block's final check (M2, a count of conflicts), and what the solver spent on it — wall and CPU seconds, solves, failed
 * solves — summed from the solves its sidecar (or, with no record, its failed sidecar) lists: every aircraft has them, an M2
 * aircraft's with its earliest-arrival solves — except one whose failed sidecar lost its solves (a failure outside the solve):
 * those four are null, "not recorded". Null: not flown (no record), or — delay, block check — not an M2 job's, or an untyped
 * aircraft.
 */
export interface TrafficAircraftChange {
  type: string | null;
  firstSolveLosses: number | null;
  finalLosses: number | null;
  landingVsRecordS: number | null;
  delayS: number | null;
  blockCheckLosses: number | null;
  optimizeS: number | null;
  optimizeCpuS: number | null;
  solves: number | null;
  failedSolves: number | null;
}

/**
 * An arrival of an M2 block that the job could not control — it has no aircraft dynamics model — and that flew its record
 * (`state.json` `stayedRecords`, by flight key). Null: the roster has no callsign / the recorded traffic no type.
 */
export interface TrafficStayedRecord {
  callsign: string | null;
  type: string | null;
}

/** The job's wall time (`state.json` `timing`): in total and by stage, in the order the stages happened. */
export interface TrafficJobTiming {
  totalS: number;
  phases: Record<string, number>;
}

/** What an M2 job leaves after the block's final check, per reading: the flown aircraft with a loss in it. */
export interface TrafficFinalCheck {
  answered: number;
  not_answered: number;
}

/** The readout of a finished job (`traffic.readout`): the part the panel shows beyond the comparison index. */
export interface TrafficJobSummary {
  /** M2: the arrivals the block controls (those with a dynamics model). */
  aircraft?: number;
  flown_aircraft_with_a_loss_left_after_the_block?: { visual: TrafficFinalCheck; ifr: TrafficFinalCheck };
  [field: string]: unknown;
}

export interface TrafficJobStatus {
  state: TrafficJobState;
  progress: TrafficJobProgress;
  error: string | null;
  /** Once `done`. */
  summary?: TrafficJobSummary;
  /** Once `done`, by flight key. */
  perAircraft?: Record<string, TrafficAircraftChange>;
  /** Once `done`. */
  timing?: TrafficJobTiming;
  /** Once `done`, by flight key: the arrivals of an M2 block that stayed their records ({} for an M1 job). */
  stayedRecords?: Record<string, TrafficStayedRecord>;
}

/**
 * A status whose state is `done`: `isTrafficJobStatus` refuses a done status without what changed for each aircraft, how long
 * the job took and which arrivals stayed their records, so a reader that has seen `state === "done"` may take them as there.
 */
export type TrafficJobDoneStatus = TrafficJobStatus & {
  state: "done";
  perAircraft: Record<string, TrafficAircraftChange>;
  timing: TrafficJobTiming;
  stayedRecords: Record<string, TrafficStayedRecord>;
};

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

function isScenarioArrival(row: unknown): row is TrafficScenarioArrival {
  return (
    isRecord(row) &&
    typeof row.flightKey === "string" &&
    (row.callsign === null || typeof row.callsign === "string") &&
    typeof row.runway === "string" &&
    (row.type === null || typeof row.type === "string") &&
    typeof row.entryUtc === "string" &&
    typeof row.landingUtc === "string" &&
    (row.category === null || typeof row.category === "string") &&
    typeof row.lossInstants === "number" &&
    Array.isArray(row.kinds) &&
    row.kinds.every((kind) => typeof kind === "string") &&
    typeof row.tightest === "number" &&
    typeof row.recordedAircraft === "number"
  );
}

function isScenarioBlock(row: unknown): row is TrafficScenarioBlock {
  return (
    isRecord(row) &&
    typeof row.startUtc === "string" &&
    typeof row.arrivals === "number" &&
    typeof row.commandable === "number" &&
    typeof row.lossInstants === "number" &&
    Array.isArray(row.runways) &&
    row.runways.every((runway) => typeof runway === "string")
  );
}

export function isTrafficScenarioCatalog(value: unknown): value is TrafficScenarioCatalog {
  if (!isRecord(value) || !isRecord(value.config) || !isRecord(value.counts) || !isRecord(value.m2)) return false;
  const m2 = value.m2;
  return (
    value.schema === TRAFFIC_CATALOG_SCHEMA &&
    typeof value.airport === "string" &&
    typeof value.writtenUtc === "string" &&
    typeof value.config.stepS === "number" &&
    (value.config.limit === null || typeof value.config.limit === "number") &&
    typeof value.counts.arrivals === "number" &&
    typeof value.counts.judged === "number" &&
    typeof value.counts.withLoss === "number" &&
    Array.isArray(value.m1) &&
    value.m1.every(isScenarioArrival) &&
    TRAFFIC_BLOCK_LENGTHS_S.every((length) => {
      const blocks = m2[String(length)];
      return Array.isArray(blocks) && blocks.every(isScenarioBlock);
    })
  );
}

/** The numeric fields of an aircraft's change that may be null (`type` is text or null). */
const AIRCRAFT_CHANGE_FIELDS = ["firstSolveLosses", "finalLosses", "landingVsRecordS", "delayS", "blockCheckLosses"] as const;
/** What the solver spent on the aircraft: numbers, or all four null when its failed sidecar lost its solves. */
const AIRCRAFT_SOLVER_FIELDS = ["optimizeS", "optimizeCpuS", "solves", "failedSolves"] as const;

function isPerAircraft(value: unknown): value is Record<string, TrafficAircraftChange> {
  return (
    isRecord(value) &&
    Object.values(value).every(
      (change) =>
        isRecord(change) &&
        (change.type === null || typeof change.type === "string") &&
        AIRCRAFT_CHANGE_FIELDS.every((field) => change[field] === null || typeof change[field] === "number") &&
        (AIRCRAFT_SOLVER_FIELDS.every((field) => typeof change[field] === "number") ||
          AIRCRAFT_SOLVER_FIELDS.every((field) => change[field] === null)),
    )
  );
}

function isStayedRecords(value: unknown): value is Record<string, TrafficStayedRecord> {
  return (
    isRecord(value) &&
    Object.values(value).every(
      (stayed) =>
        isRecord(stayed) &&
        (stayed.callsign === null || typeof stayed.callsign === "string") &&
        (stayed.type === null || typeof stayed.type === "string"),
    )
  );
}

function isTiming(value: unknown): value is TrafficJobTiming {
  return (
    isRecord(value) &&
    typeof value.totalS === "number" &&
    isRecord(value.phases) &&
    Object.values(value.phases).every((seconds) => typeof seconds === "number")
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
    typeof progress.phase === "string" &&
    (value.error === null || typeof value.error === "string") &&
    (value.state !== "done" || (isPerAircraft(value.perAircraft) && isTiming(value.timing) && isStayedRecords(value.stayedRecords))) &&
    (value.summary === undefined ||
      (isRecord(value.summary) &&
        (value.summary.aircraft === undefined || typeof value.summary.aircraft === "number") &&
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

/**
 * The scenario list of `airport` (design §10.6). Rejects with the backend's own message when it answers an error: a
 * missing catalog is a 404 that names the command that makes it, a catalog of another schema a 500 that names the schema.
 */
export async function fetchTrafficScenarios(airport: string): Promise<TrafficScenarioCatalog> {
  const url = `${AEROVIZ_BACKEND_URL}/traffic/scenarios?airport=${encodeURIComponent(airport)}`;
  const response = await fetch(url);
  const data = (await response.json()) as unknown;
  if (!response.ok) throw new Error(readError(data) ?? `AeroViz backend returned ${response.status} for ${url}`);
  if (!isTrafficScenarioCatalog(data)) throw new Error(`${url} is not a traffic scenario catalog`);
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
