/**
 * TrafficJobPanel.tsx
 * -------------------
 * The Optimize task's multi-aircraft panel (design §10.1, §10.6). The scenario is chosen from the airport's catalog, computed in
 * advance (`GET /traffic/scenarios`, `TrafficScenarioList`), never from a date. M1: pick one arrival that has a loss it answers
 * for in its record; it is optimized in its recorded traffic. M2: set a block length (15, 30 or 60 minutes) and pick one block;
 * every arrival of the block is scheduled and optimized. Start runs the job on the backend (`useTrafficJob`); the panel shows
 * its progress every 2 s, and when it is done the comparison layer draws its scene and the panel lists one row per controlled
 * aircraft with a summary line that names the inputs the job was started with. A click on a row selects that flight.
 *
 * While a job runs its progress (phase, how many aircraft are settled, the elapsed time) sits right under the header, above the
 * list: the same for M1 and M2, and in view whatever the list's length. The type codes of the lists and of the result carry their
 * plain names as titles where the app has them (`useAircraftTypeNames`).
 *
 * Start is disabled while a job starts, runs or is being cancelled: the user cancels first. The panel is mounted while the
 * task is Optimize and its mode is a multi-aircraft one: leaving either cancels a running job and removes the scene
 * (`useTrafficJob`).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import {
  DEFAULT_TRAFFIC_BLOCK_S,
  fetchTrafficScenarios,
  type TrafficBlockLengthS,
  type TrafficJobMode,
  type TrafficScenarioCatalog,
} from "../data/trafficJobs";
import { useAircraftTypeNames } from "../hooks/useAircraftTypeNames";
import { useTrafficJob } from "../hooks/useTrafficJob";
import { typeNameOf } from "../utils/aircraftTypeNames";
import { buildComparisonLegend } from "../utils/comparisonLegend";
import { trackEntityById } from "../utils/trackEntity";
import {
  SOLVER_SENTENCE,
  STAYED_RECORD_REASON,
  blockJobLabel,
  blockLossShares,
  blockLossesText,
  flightJobLabel,
  formatElapsed,
  recordLosses,
  resultLine1,
  resultLine2,
  solverTitle,
  stayedRecordLines,
  summarizeTrafficResult,
  timingText,
  trafficResultRows,
  trafficSummaryText,
} from "../utils/trafficJobResult";
import { CONTROLLABLE_TITLE, NO_COMMANDABLE_REASON, catalogBlocks, lossSecondsTitle, selectionText } from "../utils/trafficScenarios";
import ComparisonLegendList from "./ComparisonLegendList";
import TrafficCard from "./TrafficCard";
import TrafficScenarioList from "./TrafficScenarioList";
import TypeCode from "./TypeCode";

type CatalogView =
  | { status: "loading" }
  | { status: "ready"; catalog: TrafficScenarioCatalog }
  /** `error`: the backend's own message (a missing catalog names the command that makes it). */
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

/** Whole seconds since `active` became true (a job's wall time, while it runs). */
function useElapsedSeconds(active: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const started = Date.now();
    setElapsed(0);
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return elapsed;
}

export default function TrafficJobPanel({ kind }: { kind: TrafficJobMode }) {
  const { activeAirportCode, viewer, selectedFlightId, setSelectedFlightId } = useApp();
  const job = useTrafficJob(activeAirportCode);
  const typeNames = useAircraftTypeNames();
  const [catalogView, setCatalogView] = useState<CatalogView>({ status: "loading" });
  // M1: the flight key of the arrival picked; M2: the start (UTC) of the block picked, of the length `blockS`
  const [picked, setPicked] = useState<string | null>(null);
  const [blockS, setBlockS] = useState<TrafficBlockLengthS>(DEFAULT_TRAFFIC_BLOCK_S);
  // the block the M2 job in flight (or the last) was started for: its result reconciles the block's recorded losses
  const [startedBlock, setStartedBlock] = useState<{ startUtc: string; blockS: TrafficBlockLengthS } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCatalogView({ status: "loading" });
    fetchTrafficScenarios(activeAirportCode)
      .then((catalog) => {
        if (!cancelled) setCatalogView({ status: "ready", catalog });
      })
      .catch((error: unknown) => {
        if (!cancelled) setCatalogView({ status: "error", error: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [activeAirportCode]);

  // a block of another length is another scenario: the pick goes with it
  const chooseBlockLength = useCallback((length: TrafficBlockLengthS) => {
    setBlockS(length);
    setPicked(null);
  }, []);

  const catalog = catalogView.status === "ready" ? catalogView.catalog : null;
  const pickedArrival = kind === "m1" ? catalog?.m1.find((a) => a.flightKey === picked) : undefined;
  const pickedBlock = kind === "m2" && catalog ? catalogBlocks(catalog, blockS).find((b) => b.startUtc === picked) : undefined;

  const phase = job.view.phase;
  const running = phase === "starting" || phase === "running";
  const busy = running || phase === "cancelling";
  const elapsedS = useElapsedSeconds(running);
  // the job controls the arrivals that have an aircraft dynamics model: a block with none cannot be started (the reason is shown)
  const blockWithoutDynamics = pickedBlock !== undefined && pickedBlock.commandable === 0;
  const startDisabled = busy || (kind === "m1" ? pickedArrival === undefined : pickedBlock === undefined || blockWithoutDynamics);

  function startJob(): void {
    if (pickedArrival) {
      void job.start({ mode: "m1", airport: activeAirportCode, flightKey: pickedArrival.flightKey }, flightJobLabel(pickedArrival));
    } else if (pickedBlock) {
      setStartedBlock({ startUtc: pickedBlock.startUtc, blockS });
      void job.start(
        { mode: "m2", airport: activeAirportCode, blockStartUtc: pickedBlock.startUtc, blockS },
        blockJobLabel(pickedBlock.startUtc, blockS),
      );
    }
  }

  function selectFlight(flightKey: string): void {
    setSelectedFlightId(flightKey);
    if (viewer) trackEntityById(viewer, flightKey);
  }

  const result = useMemo(() => {
    if (job.view.phase !== "done" || catalog === null) return null;   // a job is started from a loaded list
    const rows = trafficResultRows(job.view.index, job.view.perAircraft, job.view.offTargetRows);
    const isScene = job.view.index.scene !== undefined;
    return {
      label: job.view.label,
      // per aircraft, two lines: who it is and how it was flown, then what changed (its record's loss instants come from the
      // list the job was started from)
      rows: rows.map((row) => ({
        ...row,
        typeName: typeNameOf(typeNames, row.change.type),
        line1: resultLine1(row),
        line2: resultLine2(row, recordLosses(catalog, row.flightKey), catalog.config.stepS, isScene),
      })),
      isScene,
      summary: trafficSummaryText(summarizeTrafficResult(rows, job.view.status.summary, isScene, job.view.stayedRecords)),
      // M2: the arrivals of the block that could not be controlled (no aircraft dynamics model), each named
      stayed: stayedRecordLines(job.view.stayedRecords),
      timing: timingText(job.view.timing),
      // M2: which aircraft carry the block's recorded losses (flown or not), so its seconds in loss add up to the lines below
      blockLosses: isScene && startedBlock !== null
        ? {
          text: blockLossesText(blockLossShares(catalog, startedBlock.startUtc, startedBlock.blockS, job.view.perAircraft)),
          title: lossSecondsTitle(catalog.config.stepS),
        }
        : null,
      legend: buildComparisonLegend(job.view.index, null),
    };
  }, [job.view, catalog, startedBlock, typeNames]);

  const progress = job.view.phase === "running" ? job.view.status.progress : null;
  const connectionLost = job.view.phase === "running" ? job.view.connection : null;

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

      {phase === "starting" ? <p className="traffic-job-note">Starting the job…</p> : null}
      {progress ? (
        <div className="traffic-job-progress" role="status" aria-label="Progress">
          <progress max={progress.total ?? undefined} value={progress.total === null ? undefined : progress.done} />
          <span aria-label="Phase">{progress.phase} · {formatElapsed(elapsedS)}</span>
          {progress.total === null ? null : (
            <span>
              {`${progress.done} of ${progress.total} aircraft done`}
              {progress.current ? ` · last: ${progress.current.split("_")[0]}` : ""}
            </span>
          )}
        </div>
      ) : null}
      {connectionLost !== null ? (
        <p className="traffic-job-note traffic-job-connection" role="alert">
          Connection lost, retrying… the job is still on the backend ({connectionLost})
        </p>
      ) : null}
      {catalogView.status === "loading" ? <p className="traffic-job-note">Loading the scenario list…</p> : null}
      {catalogView.status === "error" ? <p className="pilot-error" role="alert">{catalogView.error}</p> : null}
      {catalog ? (
        <TrafficScenarioList
          kind={kind}
          catalog={catalog}
          blockS={blockS}
          onBlockS={chooseBlockLength}
          picked={picked}
          onPick={setPicked}
          typeNames={typeNames}
        />
      ) : null}
      {catalog ? (
        <p
          className="traffic-job-note"
          role="status"
          aria-label="Selection"
          title={pickedBlock ? `${CONTROLLABLE_TITLE}. ${lossSecondsTitle(catalog.config.stepS)}` : undefined}
        >
          {selectionText(kind, pickedArrival, pickedBlock, blockS, catalog.config.stepS, typeNames)}
        </p>
      ) : null}
      {blockWithoutDynamics ? (
        <p className="traffic-job-note traffic-job-blocked" role="status" aria-label="Start disabled">
          Start is disabled: {NO_COMMANDABLE_REASON}.
        </p>
      ) : null}

      <p className="traffic-job-note traffic-job-settings" title={solverTitle()}>{SOLVER_SENTENCE}</p>

      <div className="traffic-job-actions">
        <button type="button" onClick={startJob} disabled={startDisabled}>Start</button>
        <button type="button" onClick={job.cancel} disabled={!running}>Cancel</button>
      </div>

      {phase === "failed" && job.view.phase === "failed" ? <p className="pilot-error" role="alert">{job.view.error}</p> : null}
      {phase === "cancelling" ? <p className="traffic-job-note">Cancelling the job…</p> : null}
      {phase === "cancelled" ? <p className="traffic-job-note">The job was cancelled.</p> : null}

      {result ? (
        <section className="traffic-job-result" aria-label="Result">
          <p className="traffic-job-summary" role="status">
            <strong>{result.label}</strong> — {result.summary}
          </p>
          <p className="traffic-job-note traffic-job-timing" aria-label="Timing">{result.timing}</p>
          {result.blockLosses === null ? null : (
            <p className="traffic-job-note traffic-job-block-losses" aria-label="Block losses" title={result.blockLosses.title}>
              {result.blockLosses.text}
            </p>
          )}
          <ul className="traffic-job-cards traffic-job-results" role="listbox" aria-label="Controlled aircraft">
            {result.rows.map((row) => (
              <TrafficCard
                key={row.flightKey}
                lines={[row.line1, row.line2]}
                title={row.title}
                selected={row.flightKey === selectedFlightId}
                pick={() => selectFlight(row.flightKey)}
                detailClass="traffic-job-detail"
                typeCode={row.change.type ?? undefined}
                typeName={row.typeName}
              />
            ))}
          </ul>
          {result.stayed.length === 0 ? null : (
            <ul className="traffic-job-stayed" aria-label="Not controllable">
              {result.stayed.map((stayed) => (
                <li key={stayed.flightKey} className="traffic-job-note" title={stayed.flightKey}>
                  {stayed.callsign} · <TypeCode code={stayed.type} name={typeNameOf(typeNames, stayed.type)} />: {STAYED_RECORD_REASON}
                </li>
              ))}
            </ul>
          )}
          <ComparisonLegendList legend={result.legend} />
        </section>
      ) : null}
    </div>
  );
}
