/**
 * TrafficJobPanel.tsx
 * -------------------
 * The Optimize task's multi-aircraft panel (design §10.1). The user picks a UTC day; the panel lists the airport's
 * arrivals of that day (the backend's roster). M1: pick one arrival, which is optimized in its recorded traffic. M2: set a
 * block (a 24-hour UTC start — an hour and a quarter hour — and a length of 15, 30 or 60 minutes, with the count of arrivals
 * landing in it), and every arrival of the block is scheduled and optimized. Start runs the job on the backend
 * (`useTrafficJob`); the panel shows its progress every 2 s, and when it is done the comparison layer draws its scene and the
 * panel lists one row per controlled aircraft with a summary line that names the inputs the job was started with. A click on
 * a row selects that flight.
 *
 * Start is disabled while a job starts, runs or is being cancelled: the user cancels first. The panel is mounted while the
 * task is Optimize and its mode is a multi-aircraft one: leaving either cancels a running job and removes the scene
 * (`useTrafficJob`).
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import {
  DEFAULT_TRAFFIC_BLOCK_S,
  TRAFFIC_BLOCK_LENGTHS_S,
  TRAFFIC_JOB_SETTINGS,
  arrivalCallsign,
  fetchTrafficArrivals,
  type TrafficArrival,
  type TrafficBlockLengthS,
  type TrafficJobMode,
} from "../data/trafficJobs";
import { useTrafficJob } from "../hooks/useTrafficJob";
import { buildComparisonLegend } from "../utils/comparisonLegend";
import { trackEntityById } from "../utils/trackEntity";
import {
  BLOCK_START_HOURS,
  BLOCK_START_MINUTES,
  arrivalsInBlock,
  blockJobLabel,
  blockStartOf,
  defaultBlockHour,
  flightJobLabel,
  summarizeTrafficResult,
  trafficResultRows,
  trafficSummaryText,
} from "../utils/trafficJobResult";
import ComparisonLegendList from "./ComparisonLegendList";

const DATE_STORAGE_KEY = "aeroviz.trafficJob.date";

/** The last UTC day picked, kept for convenience; the panel works without storage. */
function rememberedDate(): string {
  try {
    return window.localStorage.getItem(DATE_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

function rememberDate(date: string): void {
  try {
    window.localStorage.setItem(DATE_STORAGE_KEY, date);
  } catch {
    // a per-viewer convenience only
  }
}

/** `HH:MM:SS` of a UTC stamp such as `2026-05-21T17:47:18Z`. */
function clockOf(utc: string): string {
  return utc.slice(11, 19);
}

type ArrivalsView =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; arrivals: TrafficArrival[] }
  | { status: "error"; error: string };

const STATUS_LABELS = {
  idle: "Standby",
  starting: "Starting",
  running: "Computing",
  cancelling: "Cancelling",
  done: "Ready",
  failed: "Failed",
  cancelled: "Cancelled",
} as const;

export default function TrafficJobPanel({ kind }: { kind: TrafficJobMode }) {
  const { activeAirportCode, viewer, selectedFlightId, setSelectedFlightId } = useApp();
  const job = useTrafficJob(activeAirportCode);
  const [date, setDate] = useState<string>(rememberedDate);
  const [arrivalsView, setArrivalsView] = useState<ArrivalsView>({ status: "idle" });
  const [pickedKey, setPickedKey] = useState<string | null>(null);
  const [hour, setHour] = useState<string>("00");
  const [minute, setMinute] = useState<string>("00");
  // the block start follows the day's first landing until the user sets it
  const [startSet, setStartSet] = useState(false);
  const [blockS, setBlockS] = useState<TrafficBlockLengthS>(DEFAULT_TRAFFIC_BLOCK_S);

  useEffect(() => {
    if (!activeAirportCode || !date) {
      setArrivalsView({ status: "idle" });
      return undefined;
    }
    let cancelled = false;
    setArrivalsView({ status: "loading" });
    fetchTrafficArrivals(activeAirportCode, date)
      .then((found) => {
        if (cancelled) return;
        setArrivalsView({ status: "ready", arrivals: found.arrivals });
        setPickedKey((current) => (found.arrivals.some((a) => a.flightKey === current) ? current : null));
      })
      .catch((error: unknown) => {
        if (!cancelled) setArrivalsView({ status: "error", error: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, date]);

  const arrivals = useMemo(() => (arrivalsView.status === "ready" ? arrivalsView.arrivals : []), [arrivalsView]);

  // the default start: the hour of the day's first landing, rounded down — until the user sets one
  useEffect(() => {
    if (startSet || arrivalsView.status !== "ready") return;
    setHour(defaultBlockHour(arrivals));
    setMinute("00");
  }, [startSet, arrivalsView.status, arrivals]);

  const blockStart = date ? blockStartOf(date, hour, minute) : "";
  const inBlock = useMemo(
    () => (kind === "m2" && blockStart ? arrivalsInBlock(arrivals, blockStart, blockS) : []),
    [arrivals, kind, blockStart, blockS],
  );
  const inBlockKeys = useMemo(() => new Set(inBlock.map((a) => a.flightKey)), [inBlock]);
  const crossesMidnight = kind === "m2" && blockStart !== "" &&
    Date.parse(blockStart) + blockS * 1000 > Date.parse(`${date}T00:00:00Z`) + 86_400_000;

  const phase = job.view.phase;
  const running = phase === "starting" || phase === "running";
  const busy = running || phase === "cancelling";
  const startDisabled = busy || (kind === "m1" ? pickedKey === null : inBlock.length === 0);
  const picked = arrivals.find((a) => a.flightKey === pickedKey);

  function startJob(): void {
    if (kind === "m1") {
      if (picked) {
        void job.start({ mode: "m1", airport: activeAirportCode, flightKey: picked.flightKey }, flightJobLabel(picked));
      }
    } else {
      void job.start(
        { mode: "m2", airport: activeAirportCode, blockStartUtc: blockStart, blockS },
        blockJobLabel(blockStart, blockS),
      );
    }
  }

  function selectFlight(flightKey: string): void {
    setSelectedFlightId(flightKey);
    if (viewer) trackEntityById(viewer, flightKey);
  }

  const result = useMemo(() => {
    if (job.view.phase !== "done") return null;
    const rows = trafficResultRows(job.view.index);
    const isScene = job.view.index.scene !== undefined;
    return {
      label: job.view.label,
      rows,
      isScene,
      summary: trafficSummaryText(summarizeTrafficResult(rows, job.view.status.summary, isScene)),
      legend: buildComparisonLegend(job.view.index, null),
      aircraft: job.view.status.summary?.aircraft,
      records: job.view.status.summary?.skipped_no_dynamics,
    };
  }, [job.view]);

  const progress = job.view.phase === "running" ? job.view.status.progress : null;
  const connectionLost = job.view.phase === "running" ? job.view.connection : null;
  // the count of arrivals in the block; once a job of THIS block is done, how many it controlled and how many stayed records
  const blockCount = `${inBlock.length} arrival${inBlock.length === 1 ? " lands" : "s land"} in this block`;
  const sameBlockDone = result !== null && result.isScene && result.label === blockJobLabel(blockStart, blockS) &&
    result.aircraft !== undefined && result.records !== undefined;

  return (
    <div className="pilot-panel traffic-job-panel">
      <header className="pilot-panel-header">
        <div className="pilot-panel-header-main">
          <div className="pilot-panel-title-block">
            <h3>{kind === "m1" ? "One controlled" : "All controlled"}</h3>
          </div>
          <span className={`pilot-status pilot-status-${STATUS_LABELS[phase].toLowerCase()}`}>
            {STATUS_LABELS[phase]}
          </span>
        </div>
      </header>

      <section className="traffic-job-when" aria-label="Day and block">
        <label className="traffic-job-date">
          <span>UTC day</span>
          <input
            type="date"
            className="pilot-select-input"
            value={date}
            onChange={(event) => {
              setDate(event.target.value);
              setStartSet(false);
              rememberDate(event.target.value);
            }}
          />
        </label>
        {kind === "m2" ? (
          <div className="traffic-job-block pilot-optimization-row">
            <label>
              <span>Start hour (UTC)</span>
              <select
                className="pilot-select-input"
                value={hour}
                onChange={(event) => { setHour(event.target.value); setStartSet(true); }}
              >
                {BLOCK_START_HOURS.map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
            </label>
            <label>
              <span>Minute</span>
              <select
                className="pilot-select-input"
                value={minute}
                onChange={(event) => { setMinute(event.target.value); setStartSet(true); }}
              >
                {BLOCK_START_MINUTES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <label>
              <span>Length</span>
              <select
                className="pilot-select-input"
                value={blockS}
                onChange={(event) => setBlockS(Number(event.target.value) as TrafficBlockLengthS)}
              >
                {TRAFFIC_BLOCK_LENGTHS_S.map((length) => (
                  <option key={length} value={length}>{length / 60} min</option>
                ))}
              </select>
            </label>
          </div>
        ) : null}
      </section>

      {kind === "m2" ? (
        <p className="traffic-job-note" role="status" aria-label="Arrivals in the block">
          {sameBlockDone
            ? `${result.aircraft! + result.records!} arrivals landed in this block: ${result.aircraft} controlled, ` +
              `${result.records} stayed records (no aircraft dynamics model)`
            : blockCount}
          {crossesMidnight ? " (the block runs past midnight: the next day's landings are flown but not counted here)" : ""}
        </p>
      ) : (
        <p className="traffic-job-note">
          {picked ? `Selected: ${arrivalCallsign(picked)}` : "Pick the arrival to optimize in its recorded traffic."}
        </p>
      )}

      <div className="traffic-job-arrivals traffic-job-table-scroll" aria-label="Arrivals of the day">
        {arrivalsView.status === "loading" ? <p className="traffic-job-note">Loading arrivals…</p> : null}
        {arrivalsView.status === "error" ? <p className="pilot-error" role="alert">{arrivalsView.error}</p> : null}
        {arrivalsView.status === "idle" ? <p className="traffic-job-note">Pick a UTC day to list the arrivals.</p> : null}
        {arrivalsView.status === "ready" && arrivals.length === 0 ? (
          <p className="traffic-job-note">No arrival of {activeAirportCode} lands on {date}.</p>
        ) : null}
        {arrivals.length > 0 ? (
          <table>
            <thead>
              <tr><th>Landing</th><th>Flight</th><th>Rwy</th><th>Type</th></tr>
            </thead>
            <tbody>
              {arrivals.map((arrival) => (
                <tr
                  key={arrival.flightKey}
                  className={
                    kind === "m1"
                      ? arrival.flightKey === pickedKey ? "selected" : ""
                      : inBlockKeys.has(arrival.flightKey) ? "in-block" : ""
                  }
                  title={arrival.flightKey}
                  onClick={kind === "m1" ? () => setPickedKey(arrival.flightKey) : undefined}
                  style={kind === "m1" ? { cursor: "pointer" } : undefined}
                >
                  <td>{clockOf(arrival.landingUtc)}</td>
                  <td>{arrivalCallsign(arrival)}</td>
                  <td>{arrival.runway}</td>
                  <td>{arrival.type ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>

      <details className="traffic-job-settings">
        <summary>Solver settings (the batch's)</summary>
        <p>
          max duration {TRAFFIC_JOB_SETTINGS.maxDurationS} s · rollout step {TRAFFIC_JOB_SETTINGS.rolloutDtS} s · IPOPT
          cap {TRAFFIC_JOB_SETTINGS.maxIterations} · check step {TRAFFIC_JOB_SETTINGS.stepS} s · row window{" "}
          {TRAFFIC_JOB_SETTINGS.rowWindowS} s · margin {TRAFFIC_JOB_SETTINGS.margin * 100} % · re-solve rounds{" "}
          {TRAFFIC_JOB_SETTINGS.maxRounds}
        </p>
      </details>

      <div className="traffic-job-actions">
        <button type="button" onClick={startJob} disabled={startDisabled}>Start</button>
        <button type="button" onClick={job.cancel} disabled={!running}>Cancel</button>
      </div>

      {phase === "starting" ? <p className="traffic-job-note">Starting the job…</p> : null}
      {progress ? (
        <div className="traffic-job-progress" role="status" aria-label="Progress">
          <progress max={progress.total ?? undefined} value={progress.total === null ? undefined : progress.done} />
          <span>
            {progress.total === null
              ? "Reading the traffic…"
              : `${progress.done} of ${progress.total} aircraft done`}
            {progress.current ? ` · last: ${progress.current.split("_")[0]}` : ""}
          </span>
        </div>
      ) : null}
      {connectionLost !== null ? (
        <p className="traffic-job-note traffic-job-connection" role="alert">
          Connection lost, retrying… the job is still on the backend ({connectionLost})
        </p>
      ) : null}
      {phase === "failed" && job.view.phase === "failed" ? <p className="pilot-error" role="alert">{job.view.error}</p> : null}
      {phase === "cancelling" ? <p className="traffic-job-note">Cancelling the job…</p> : null}
      {phase === "cancelled" ? <p className="traffic-job-note">The job was cancelled.</p> : null}

      {result ? (
        <section className="traffic-job-result" aria-label="Result">
          <p className="traffic-job-summary" role="status">
            <strong>{result.label}</strong> — {result.summary}
          </p>
          <div className="traffic-job-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Flight</th><th>Rwy</th><th>Outcome</th>
                  {result.isScene ? <th title="Delay of the aircraft's slot, in seconds">Delay (s)</th> : null}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr
                    key={row.flightKey}
                    className={row.flightKey === selectedFlightId ? "selected" : ""}
                    title={row.title}
                    onClick={() => selectFlight(row.flightKey)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>{row.callsign}</td>
                    <td>{row.runway}</td>
                    <td className="traffic-job-outcome">{row.outcomeName}</td>
                    {result.isScene ? <td>{row.delayS === null ? "—" : Math.round(row.delayS)}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ComparisonLegendList legend={result.legend} />
        </section>
      ) : null}
    </div>
  );
}
