/**
 * TrafficScenarioList.tsx
 * -----------------------
 * The scenario list of the Optimize task's multi-aircraft panel (design §10.6): the airport's catalog, computed in advance by
 * `traffic_scenarios.py`, in place of a date. M1: the arrivals with a loss they answer for in their own record (practice traffic —
 * light aircraft, CWT I — hidden unless asked for: it tops the ranking). M2: the blocks (15, 30 or 60 min) in which an arrival
 * lands. Sorted by the "Sort by" select (losses or time), filtered by runway, 100 rows at a time ("Showing 1–100 of N · Show 100
 * more": no row is dropped, the count line says how many match) in a scroll area of fixed height.
 *
 * The dock is 242 px wide: a row is a two-line CARD, not a table row (no columns to fit; the text wraps at spaces, never
 * mid-word). The rest of a row is in its title and, once picked, in the panel's "Selected" line. A click on a card selects it
 * (the panel starts the job on it).
 *
 * What clears a pick, and what does not: a FILTER that hides the picked row (the runway, the light-aircraft switch) clears it,
 * so Start never runs on a row the filter has taken off the list. A pick the view merely no longer SHOWS — a sort moved it, or
 * it is on a page not drawn — stays picked, and stays named in the panel's "Selected" line (which is where to see it), so the
 * job still starts on what the line says. Cards are memoised components: a pick re-renders the two cards whose selection
 * changed, not the thousands of the list (KRDU: 6,448 blocks of 15 min).
 */

import { memo, useMemo, useState } from "react";
import {
  TRAFFIC_BLOCK_LENGTHS_S,
  type TrafficBlockLengthS,
  type TrafficJobMode,
  type TrafficScenarioArrival,
  type TrafficScenarioBlock,
  type TrafficScenarioCatalog,
} from "../data/trafficJobs";
import {
  CATALOG_MEANING,
  DEFAULT_TRAFFIC_SORT,
  LIGHT_AIRCRAFT_CATEGORY,
  LIST_PAGE_ROWS,
  TRAFFIC_SORT_CHOICES,
  CONTROLLABLE_TITLE,
  LIGHT_AIRCRAFT_TITLE,
  arrivalCardLines,
  arrivalDetail,
  blockCardLines,
  blockDetail,
  blockRunwaysText,
  blocksHeader,
  catalogBlocks,
  catalogHeader,
  catalogLimitNote,
  listedRows,
  lossSecondsTitle,
  matchCount,
  runwayChoices,
  sortChoiceValue,
  type TrafficSort,
} from "../utils/trafficScenarios";
import { typeNameOf, type AircraftTypeNames } from "../utils/aircraftTypeNames";
import TrafficCard from "./TrafficCard";

interface TrafficScenarioListProps {
  kind: TrafficJobMode;
  catalog: TrafficScenarioCatalog;
  /** M2: the block length. */
  blockS: TrafficBlockLengthS;
  onBlockS: (blockS: TrafficBlockLengthS) => void;
  /** The row picked: an M1 arrival's flight key, an M2 block's start (UTC); null when none. */
  picked: string | null;
  /** Pick a row; null clears the pick (a filter hid it). Must be stable: the cards are memoised on it. */
  onPick: (key: string | null) => void;
  /** The plain names of the ICAO type codes the app has: a type code of a card gets its name as its title. */
  typeNames: AircraftTypeNames;
}

interface CardProps<Row> {
  row: Row;
  selected: boolean;
  onPick: (key: string) => void;
  /** The census's check step (s): a card says its losses in seconds. */
  stepS: number;
}

/**
 * One M1 card. Memoised on (row, selected, onPick, stepS, typeName): its title is built when it renders, which only a change of
 * those causes (`typeName`: a string or undefined, so the names arriving re-render only the cards whose type got one).
 */
const ArrivalCard = memo(function ArrivalCard({ row, selected, onPick, stepS, typeName }: CardProps<TrafficScenarioArrival> & { typeName?: string }) {
  return (
    <TrafficCard
      lines={arrivalCardLines(row, stepS)}
      title={`${row.flightKey} — ${arrivalDetail(row, stepS, typeName)}. ${lossSecondsTitle(stepS)}`}
      selected={selected}
      pick={() => onPick(row.flightKey)}
      typeCode={row.type ?? undefined}
      typeName={typeName}
    />
  );
});

/** One M2 card; the block length is part of its title (the block's end), and does not change with a pick. */
const BlockCard = memo(function BlockCard({ row, blockS, selected, onPick, stepS }: CardProps<TrafficScenarioBlock> & { blockS: number }) {
  return (
    <TrafficCard
      lines={blockCardLines(row, blockS, stepS)}
      title={`${blockDetail(row, blockS, stepS)}. ${CONTROLLABLE_TITLE}. ${lossSecondsTitle(stepS)}`}
      selected={selected}
      pick={() => onPick(row.startUtc)}
      keep={blockRunwaysText(row)}
    />
  );
});

/**
 * A memo: the panel re-renders at every poll of a running job, and the lists hold thousands of rows (re-sorting and
 * re-diffing them every 2 s is the cost this skips).
 */
function TrafficScenarioList({ kind, catalog, blockS, onBlockS, picked, onPick, typeNames }: TrafficScenarioListProps) {
  const [sort, setSort] = useState<TrafficSort>(DEFAULT_TRAFFIC_SORT);
  const [runway, setRunway] = useState("");
  // M1: practice traffic (light aircraft, CWT I) tops the loss ranking; it is hidden until the user asks for it
  const [hideLight, setHideLight] = useState(true);
  // how many rows are shown, for the view they were asked in: any change of view (kind, length, filter, sort) starts again at one page
  const [paging, setPaging] = useState<{ view: string; rows: number }>({ view: "", rows: LIST_PAGE_ROWS });

  const isM1 = kind === "m1";
  const blocks = catalogBlocks(catalog, blockS);
  const total = isM1 ? catalog.m1.length : blocks.length;
  // the light aircraft the switch hides from THIS list: those on the runway asked for (the runway filter is applied beside it)
  const lightCount = useMemo(
    () => catalog.m1.filter((r) => r.category === LIGHT_AIRCRAFT_CATEGORY && (runway === "" || r.runway === runway)).length,
    [catalog, runway],
  );
  const hidingLight = isM1 && hideLight;
  const runways = useMemo(
    () => (isM1 ? runwayChoices(catalog.m1, (r) => [r.runway]) : runwayChoices(blocks, (r) => r.runways)),
    [isM1, catalog, blocks],
  );
  const m1Rows = useMemo(
    () => (isM1
      ? listedRows(hideLight ? catalog.m1.filter((r) => r.category !== LIGHT_AIRCRAFT_CATEGORY) : catalog.m1,
        (r) => [r.runway], (r) => r.landingUtc, runway, sort)
      : []),
    [isM1, catalog, hideLight, runway, sort],
  );
  const m2Rows = useMemo(
    () => (isM1 ? [] : listedRows(blocks, (r) => r.runways, (r) => r.startUtc, runway, sort)),
    [isM1, blocks, runway, sort],
  );
  const matching = isM1 ? m1Rows.length : m2Rows.length;
  const view = `${kind}|${blockS}|${runway}|${hideLight}|${sortChoiceValue(sort)}`;
  const shown = Math.min(matching, paging.view === view ? paging.rows : LIST_PAGE_ROWS);

  /** A filter that hides the row picked clears the pick: Start never runs on a row the user cannot see. */
  function chooseRunway(next: string): void {
    setRunway(next);
    if (picked === null || next === "") return;
    const row = isM1 ? catalog.m1.find((r) => r.flightKey === picked) : blocks.find((b) => b.startUtc === picked);
    const held = row === undefined ? [] : "runways" in row ? row.runways : [row.runway];
    if (!held.includes(next)) onPick(null);
  }

  function chooseHideLight(next: boolean): void {
    setHideLight(next);
    if (next && picked !== null && catalog.m1.find((r) => r.flightKey === picked)?.category === LIGHT_AIRCRAFT_CATEGORY) onPick(null);
  }

  const limitNote = catalogLimitNote(catalog);
  const lightNote = hidingLight && lightCount > 0 ? ` · ${lightCount} light aircraft hidden` : "";

  return (
    <section className="traffic-job-catalog" aria-label="Scenario list">
      <p className="traffic-job-note traffic-job-catalog-header" role="status" title={lossSecondsTitle(catalog.config.stepS)}>
        {isM1 ? catalogHeader(catalog) : blocksHeader(catalog, blockS)}
      </p>
      <p className="traffic-job-note">{CATALOG_MEANING}</p>
      {limitNote ? <p className="traffic-job-note traffic-job-partial">{limitNote}</p> : null}

      <div className="traffic-job-filters pilot-optimization-row">
        {isM1 ? null : (
          <label>
            <span>Length</span>
            <select
              className="pilot-select-input"
              value={blockS}
              onChange={(event) => onBlockS(Number(event.target.value) as TrafficBlockLengthS)}
            >
              {TRAFFIC_BLOCK_LENGTHS_S.map((length) => (
                <option key={length} value={length}>{length / 60} min</option>
              ))}
            </select>
          </label>
        )}
        <label>
          <span>Sort by</span>
          <select
            className="pilot-select-input"
            value={sortChoiceValue(sort)}
            onChange={(event) => setSort(TRAFFIC_SORT_CHOICES.find((choice) => choice.value === event.target.value)!.sort)}
          >
            {TRAFFIC_SORT_CHOICES.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
          </select>
        </label>
        <label>
          <span>Show runway</span>
          <select className="pilot-select-input" value={runway} onChange={(event) => chooseRunway(event.target.value)}>
            <option value="">All runways</option>
            {runways.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </label>
      </div>
      {isM1 ? (
        <label className="traffic-job-check" title={LIGHT_AIRCRAFT_TITLE}>
          <input type="checkbox" checked={hideLight} onChange={(event) => chooseHideLight(event.target.checked)} />
          <span>Hide light aircraft</span>
        </label>
      ) : null}

      <p className="traffic-job-note" role="status" aria-label="Rows listed" title={isM1 ? LIGHT_AIRCRAFT_TITLE : CONTROLLABLE_TITLE}>
        {matchCount(matching, total, isM1 ? "arrivals with a loss" : "blocks")}{lightNote}
      </p>

      <div className="traffic-job-catalog-scroll">
        {matching === 0 ? (
          <p className="traffic-job-note">
            {total === 0
              ? isM1 ? `No arrival of ${catalog.airport} has a loss they answer for.` : "No block lists an arrival."
              : hidingLight && runway === "" ? "No row is left once the light aircraft are hidden." : "No row matches."}
          </p>
        ) : (
          <ul className="traffic-job-cards" role="listbox" aria-label={isM1 ? "Arrivals with a loss" : "Blocks"}>
            {isM1
              ? m1Rows.slice(0, shown).map((row) => (
                <ArrivalCard key={row.flightKey} row={row} selected={row.flightKey === picked} onPick={onPick} stepS={catalog.config.stepS}
                             typeName={typeNameOf(typeNames, row.type)} />
              ))
              : m2Rows.slice(0, shown).map((row) => (
                <BlockCard key={row.startUtc} row={row} blockS={blockS} selected={row.startUtc === picked} onPick={onPick}
                           stepS={catalog.config.stepS} />
              ))}
          </ul>
        )}
      </div>
      {matching > 0 ? (
        <p className="traffic-job-note traffic-job-paging" role="status" aria-label="Rows shown">
          Showing 1–{shown} of {matching}
          {shown < matching ? (
            <>
              {" · "}
              <button type="button" onClick={() => setPaging({ view, rows: shown + LIST_PAGE_ROWS })}>
                Show {Math.min(LIST_PAGE_ROWS, matching - shown)} more
              </button>
            </>
          ) : null}
        </p>
      ) : null}
    </section>
  );
}

export default memo(TrafficScenarioList);
