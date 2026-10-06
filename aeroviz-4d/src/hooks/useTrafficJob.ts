/**
 * useTrafficJob.ts
 * ----------------
 * The life of the Optimize task's multi-aircraft job (design §10.1, §10.4): start it, ask it every 2 s how far it is,
 * hand its files to the comparison layer when it is done (`trafficScene`), cancel it.
 *
 * One job at a time, and Start waits for the last one to end: while a job is starting, running or being cancelled a call
 * to `start` does nothing (the panel disables the button), so a start never has a cancel to wait for. A change of mode, another
 * airport, leaving the Optimize task or a closed page cancels a running job and removes the scene — this hook's cleanup.
 *
 * EVERY async continuation (the start's answer, a poll, a cancel's answer) is gated on the epoch it began with: a start, a
 * cancel and the cleanup each take a new epoch, so whatever answers later is dropped and cannot touch the view or the scene
 * of what came after it.
 *
 * A poll that fails (the backend restarting, the network) never drops the job: its id is kept, the view says the connection
 * is lost, and the poll is repeated with a growing delay until it answers or the user cancels.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import { isComparisonIndex, type ComparisonIndex } from "../data/airportData";
import { isEvaluationReport, type EvaluationRow } from "../data/evaluationReport";
import {
  TRAFFIC_JOB_POLL_MS,
  TRAFFIC_JOB_RETRY_MAX_MS,
  TRAFFIC_PHASE_STARTING,
  beaconCancelTrafficJob,
  cancelTrafficJob,
  fetchTrafficJob,
  startTrafficJob,
  trafficJobFilesUrl,
  type TrafficAircraftChange,
  type TrafficJobDoneStatus,
  type TrafficJobRequest,
  type TrafficJobStatus,
  type TrafficJobTiming,
  type TrafficStayedRecord,
} from "../data/trafficJobs";
import { JOB_INDEX_FILE } from "../utils/comparisonSource";
import { fetchJson } from "../utils/fetchJson";

export type TrafficJobView =
  | { phase: "idle" }
  | { phase: "starting" }
  /** `connection`: why the last poll failed (the job is still there, being asked again); null when the poll answers. */
  | { phase: "running"; jobId: string; status: TrafficJobStatus; connection: string | null }
  | { phase: "cancelling" }
  /** `label`: the inputs the job was started with (`start`'s), so a panel that has since been edited still names them. */
  | {
    phase: "done"; jobId: string; status: TrafficJobStatus; index: ComparisonIndex; label: string;
    /** What changed for each aircraft the job controlled (every group of `index` has an entry). */
    perAircraft: Record<string, TrafficAircraftChange>;
    /** How long the job took, in total and by stage. */
    timing: TrafficJobTiming;
    /** The arrivals of an M2 block the job could not control (no aircraft dynamics model), by flight key: they flew their records. */
    stayedRecords: Record<string, TrafficStayedRecord>;
    /** The evaluation report's row of each aircraft whose index group is `offTarget` (a yellow path), by flight key: the gates it failed. */
    offTargetRows: Record<string, EvaluationRow>;
  }
  | { phase: "failed"; error: string }
  | { phase: "cancelled" };

const JUST_STARTED: TrafficJobStatus = { state: "running", progress: { done: 0, total: null, current: null, phase: TRAFFIC_PHASE_STARTING }, error: null };

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * The evaluation report's row of every aircraft the index draws off the landing gates (`offTarget`: a yellow path), by flight key.
 * The report is read only when there is one such aircraft; one that cannot be read, or that lacks a row of one, is refused by name.
 */
async function offTargetReportRows(jobId: string, baseUrl: string, index: ComparisonIndex): Promise<Record<string, EvaluationRow>> {
  const offTarget = index.groups.filter((group) => group.status === "offTarget");
  if (offTarget.length === 0) return {};
  const report = await fetchJson<unknown>(`${baseUrl}${index.evaluationReport}`);
  if (!isEvaluationReport(report)) throw new Error(`job ${jobId} wrote an evaluation report the viewer cannot read`);
  return Object.fromEntries(offTarget.map((group) => {
    const row = report.trajectories.find((candidate) => candidate.flight_key === group.group);
    if (row === undefined) {
      throw new Error(`job ${jobId}: its evaluation report has no row of ${group.group}, which its index draws off the landing gates`);
    }
    return [group.group, row];
  }));
}

/** The wait before poll number `failures + 1` after `failures` failures in a row: 2 s, 4 s, 8 s, … up to the cap. */
export function retryDelayMs(failures: number): number {
  return Math.min(TRAFFIC_JOB_POLL_MS * 2 ** failures, TRAFFIC_JOB_RETRY_MAX_MS);
}

export interface TrafficJob {
  view: TrafficJobView;
  /** Start a job (`label`: its inputs in words, kept with the result). Does nothing while a job is in flight. */
  start: (request: TrafficJobRequest, label: string) => Promise<void>;
  /** Cancel the job in flight. */
  cancel: () => void;
}

export function useTrafficJob(airportCode: string): TrafficJob {
  const { setTrafficScene } = useApp();
  const [view, setView] = useState<TrafficJobView>({ phase: "idle" });
  const epochRef = useRef(0);
  /** A start, a run or a cancel is in flight. */
  const busyRef = useRef(false);
  const cancellingRef = useRef(false);
  /** The job in flight, once the backend has answered the start. */
  const jobRef = useRef<string | null>(null);
  const labelRef = useRef("");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    timerRef.current = null;
  }, []);

  /** The end of a run: the slot is free again. */
  const settle = useCallback((next: TrafficJobView) => {
    busyRef.current = false;
    cancellingRef.current = false;
    jobRef.current = null;
    setView(next);
  }, []);

  /** The end of a cancel: only if no later epoch took over. */
  const settleCancel = useCallback((epoch: number, next: TrafficJobView) => {
    if (epoch === epochRef.current) settle(next);
  }, [settle]);

  const poll = useCallback(async (jobId: string, epoch: number, failures: number) => {
    let status: TrafficJobStatus;
    try {
      status = await fetchTrafficJob(jobId);
    } catch (error) {
      if (epoch !== epochRef.current) return;
      setView((current) => (current.phase === "running" ? { ...current, connection: message(error) } : current));
      timerRef.current = setTimeout(() => void poll(jobId, epoch, failures + 1), retryDelayMs(failures));
      return;
    }
    if (epoch !== epochRef.current) return;
    if (status.state === "running") {
      setView({ phase: "running", jobId, status, connection: null });
      timerRef.current = setTimeout(() => void poll(jobId, epoch, 0), TRAFFIC_JOB_POLL_MS);
      return;
    }
    if (status.state !== "done") {
      settle(status.state === "failed"
        ? { phase: "failed", error: status.error ?? "the job failed" }
        : { phase: "cancelled" });
      return;
    }
    const baseUrl = trafficJobFilesUrl(jobId);
    try {
      const index = await fetchJson<unknown>(`${baseUrl}${JOB_INDEX_FILE}`);
      if (epoch !== epochRef.current) return;
      if (!isComparisonIndex(index)) throw new Error(`job ${jobId} wrote a comparison index the viewer cannot read`);
      const { perAircraft, timing, stayedRecords } = status as TrafficJobDoneStatus;          // `isTrafficJobStatus` refused a done one without them
      const unsaid = index.groups.find((group) => !(group.group in perAircraft));
      if (unsaid !== undefined) throw new Error(`job ${jobId} lists ${unsaid.group} in its index and says nothing of what changed for it`);
      const offTargetRows = await offTargetReportRows(jobId, baseUrl, index);       // what the yellow paths failed
      if (epoch !== epochRef.current) return;
      settle({ phase: "done", jobId, status, index, perAircraft, timing, stayedRecords, offTargetRows, label: labelRef.current });
      setTrafficScene({ airportCode, jobId, baseUrl });
    } catch (error) {
      if (epoch === epochRef.current) settle({ phase: "failed", error: message(error) });
    }
  }, [airportCode, settle, setTrafficScene]);

  const start = useCallback(async (request: TrafficJobRequest, label: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    const epoch = ++epochRef.current;
    labelRef.current = label;
    setTrafficScene(null);                       // the last job's scene goes with it
    setView({ phase: "starting" });
    let jobId: string;
    try {
      jobId = await startTrafficJob(request);
    } catch (error) {
      if (epoch === epochRef.current) settle({ phase: "failed", error: message(error) });
      return;
    }
    if (epoch !== epochRef.current) {            // cancelled, or left, while the backend was starting it
      try {
        await cancelTrafficJob(jobId);
      } catch (error) {
        console.warn(`[traffic job] could not cancel ${jobId}`, error);
      }
      settleCancel(epochRef.current, { phase: "cancelled" });
      return;
    }
    jobRef.current = jobId;
    setView({ phase: "running", jobId, status: JUST_STARTED, connection: null });
    await poll(jobId, epoch, 0);
  }, [poll, settle, settleCancel, setTrafficScene]);

  const cancel = useCallback(() => {
    if (!busyRef.current || cancellingRef.current) return;
    cancellingRef.current = true;
    const jobId = jobRef.current;
    const epoch = ++epochRef.current;
    stopPolling();
    jobRef.current = null;
    setTrafficScene(null);
    setView({ phase: "cancelling" });
    if (jobId === null) return;                  // still starting: the start's answer finishes the cancel
    cancelTrafficJob(jobId).then(
      () => settleCancel(epoch, { phase: "cancelled" }),
      (error: unknown) => settleCancel(epoch, { phase: "failed", error: `could not cancel the job: ${message(error)}` }),
    );
  }, [settleCancel, stopPolling, setTrafficScene]);

  useEffect(() => {
    const releaseOnUnload = () => {
      if (jobRef.current !== null) beaconCancelTrafficJob(jobRef.current);
    };
    window.addEventListener("pagehide", releaseOnUnload);
    return () => {
      window.removeEventListener("pagehide", releaseOnUnload);
      // The panel is going (another mode, another airport, another task): its job goes with it, and its scene.
      epochRef.current += 1;
      stopPolling();
      const jobId = jobRef.current;
      jobRef.current = null;
      if (jobId !== null) {
        cancelTrafficJob(jobId).catch((error: unknown) => console.warn(`[traffic job] could not cancel ${jobId}`, error));
      }
      setTrafficScene(null);
    };
  }, [stopPolling, setTrafficScene]);

  return { view, start, cancel };
}
