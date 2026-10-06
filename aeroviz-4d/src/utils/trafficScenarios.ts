/**
 * trafficScenarios.ts
 * -------------------
 * What the Optimize task's scenario list shows (design `4dTrajectory/docs/multi_aircraft_optimization/design.md` §10.6):
 * the plain names of the losses, the header and the interpretation lines, the sort and the runway filter of the two
 * tables. Pure functions over the catalog's rows (`data/trafficJobs.ts`), so the panel only lays them out.
 */

import {
  arrivalCallsign,
  type TrafficBlockLengthS,
  type TrafficScenarioArrival,
  type TrafficScenarioBlock,
  type TrafficScenarioCatalog,
} from "../data/trafficJobs";
import { typeNameOf, type AircraftTypeNames } from "./aircraftTypeNames";
import { blockJobLabel, blockSpan } from "./trafficJobResult";

/**
 * The kinds of loss the judge reports, in the order `separation.py` declares them (`IN_TRAIL, DIAGONAL,
 * RADAR_OR_VERTICAL, AT_THRESHOLD`) — a MIRROR, pinned by `aeroviz-4d/python/tests/test_traffic_mirrors.py`.
 */
export const TRAFFIC_LOSS_KINDS = ["in_trail", "diagonal", "radar_or_vertical", "at_threshold"] as const;

type TrafficLossKind = typeof TRAFFIC_LOSS_KINDS[number];

const LOSS_KIND_NAMES: Record<TrafficLossKind, string> = {
  in_trail: "too close in trail on one final",
  diagonal: "too close between dependent parallel finals",
  radar_or_vertical: "under the radar minimum and not vertically separated",
  at_threshold: "wake gap behind the aircraft over the threshold",
};

function isLossKind(raw: string): raw is TrafficLossKind {
  return (TRAFFIC_LOSS_KINDS as readonly string[]).includes(raw);
}

/** The plain name of a loss kind; one this file does not know is shown as it is (a made-up name would be a guess). */
export function lossKindName(raw: string): string {
  return isLossKind(raw) ? LOSS_KIND_NAMES[raw] : raw;
}

export function lossKindsText(kinds: readonly string[]): string {
  return kinds.map(lossKindName).join("; ");
}

/**
 * The CWT category of the lightest aircraft: the last of `runway_schedule.CWT_CATEGORIES` ("ABCDEFGHI") — a MIRROR, pinned by
 * `aeroviz-4d/python/tests/test_traffic_mirrors.py`. Practice traffic (C172, SR22, …) is category I and tops the loss ranking.
 */
export const LIGHT_AIRCRAFT_CATEGORY = "I";

/** How many rows a list shows at a time (the rest is one click away; nothing is dropped). */
export const LIST_PAGE_ROWS = 100;

/** The tightest loss as a share of the required minimum: `0.6487` → `65 %` (100 % is exactly the minimum). */
export function percentOfMinimum(tightest: number): string {
  return `${Math.round(tightest * 100)} %`;
}

/** `2026-05-01T00:06:59Z` → `2026-05-01 00:06:59`. */
export function utcStamp(utc: string): string {
  return `${utc.slice(0, 10)} ${utc.slice(11, 19)}`;
}

/** `2026-05-01T00:06:59Z` → `2026-05-01 00:06`. */
export function utcMinute(utc: string): string {
  return `${utc.slice(0, 10)} ${utc.slice(11, 16)}`;
}

/**
 * The header of the M1 list: how many of the judged arrivals have a loss they answer for, how often the records were checked, and
 * when. It is where "loss of separation" is spelled out; every later place says "loss".
 */
export function catalogHeader(catalog: TrafficScenarioCatalog): string {
  const { withLoss, judged } = catalog.counts;
  return `${withLoss} of ${judged} arrivals have a loss of separation they answer for in their record ` +
    `(the records were checked every ${catalog.config.stepS} s, ${catalog.writtenUtc})`;
}

/** The header of the M2 list: the blocks of the length that hold an arrival, in the catalog's order (and "loss of separation", spelled out). */
export function blocksHeader(catalog: TrafficScenarioCatalog, blockS: TrafficBlockLengthS): string {
  return `${catalogBlocks(catalog, blockS).length} blocks of ${blockS / 60} min with an arrival; ` +
    `sorted by the seconds their arrivals spend in loss of separation (the records were checked every ${catalog.config.stepS} s)`;
}

/** What a loss in the list means (design §10.6 "Interpretation"). */
export const CATALOG_MEANING =
  "A loss here is a loss between recorded aircraft as flown. It marks dense traffic; " +
  "it is not a promise that the optimized flight has one.";

/** What "seconds in loss" is: the title of every place it is said. */
export function lossSecondsTitle(stepS: number): string {
  return `Seconds in loss: check instants, ${stepS} s apart, at which the aircraft is closer than its separation minimum ` +
    "and answers for it (a block adds up its arrivals)";
}

/** What "light aircraft" are. */
export const LIGHT_AIRCRAFT_TITLE = `CWT category ${LIGHT_AIRCRAFT_CATEGORY}: the lightest wake-turbulence category`;

/** What "controllable" is. */
export const CONTROLLABLE_TITLE =
  "controllable: the arrival has an aircraft dynamics model, so a job can fly it (the others stay their records)";

/** The seconds an aircraft spends in loss: its loss instants, each a check ``stepS`` apart. */
export function secondsInLoss(instants: number, stepS: number): number {
  return Math.round(instants * stepS * 1000) / 1000;
}

const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const secondsText = (instants: number, stepS: number) =>
  plural(secondsInLoss(instants, stepS), "second in loss", "seconds in loss");

/**
 * What an M1 card leaves to its title and the "Selected" line: the seconds in loss, the tightest loss, the kinds in plain
 * words, its type (with its plain name where the app has one: `typeName`) and runway, its landing and the recorded aircraft in
 * its window. `stepS`: the census's check step.
 */
export function arrivalDetail(row: TrafficScenarioArrival, stepS: number, typeName?: string): string {
  return `${secondsText(row.lossInstants, stepS)}, tightest ${percentOfMinimum(row.tightest)} of the minimum (` +
    `${lossKindsText(row.kinds)}) · type ${row.type ?? "unknown"}${typeName === undefined ? "" : ` (${typeName})`} · ` +
    `runway ${row.runway} · landing ${utcStamp(row.landingUtc)} UTC · ${row.recordedAircraft} recorded aircraft in its window`;
}

/** An M2 card beyond its lines: the block in words with its end, its arrivals, how many are controllable, the seconds in loss, the runways. */
export function blockDetail(row: TrafficScenarioBlock, blockS: number, stepS: number): string {
  return `${blockJobLabel(row.startUtc, blockS)} — ${plural(row.arrivals, "arrival")}, ${row.commandable} controllable, ` +
    `${secondsText(row.lossInstants, stepS)}, runways ${row.runways.join(", ")}`;
}

/** Why a block cannot be started: the job controls the arrivals that have an aircraft dynamics model. */
export const NO_COMMANDABLE_REASON = "no arrival in this block is controllable (has an aircraft dynamics model)";

/** The "Selected: …" line of the panel: the row picked, in full; or the instruction while none is. */
export function selectionText(
  kind: "m1" | "m2",
  arrival: TrafficScenarioArrival | undefined,
  block: TrafficScenarioBlock | undefined,
  blockS: number,
  stepS: number,
  /** The plain names of the ICAO type codes the app has (`useAircraftTypeNames`). */
  typeNames: AircraftTypeNames,
): string {
  if (arrival) return `Selected: ${arrivalCallsign(arrival)} — ${arrivalDetail(arrival, stepS, typeNameOf(typeNames, arrival.type))}`;
  if (block) return `Selected: ${blockDetail(block, blockS, stepS)}`;
  return kind === "m1" ? "Pick the arrival to optimize in its recorded traffic." : "Pick the block to optimize.";
}

/** An M1 card: `AAL1286 · B738 · 23L` over `2026-09-21 03:32 UTC · 124 seconds in loss` (callsign, type, runway; landing, seconds in loss). */
export function arrivalCardLines(row: TrafficScenarioArrival, stepS: number): [string, string] {
  return [
    `${arrivalCallsign(row)} · ${row.type ?? "—"} · ${row.runway}`,
    `${utcMinute(row.landingUtc)} UTC · ${secondsText(row.lossInstants, stepS)}`,
  ];
}

/** The runways of a block, as the card says them: `23L, 23R, 32` (the card keeps them on one line). */
export function blockRunwaysText(row: TrafficScenarioBlock): string {
  return row.runways.join(", ");
}

/**
 * An M2 card: `2026-05-15 17:30–17:45 UTC` over `6 arrivals · 5 controllable · 198 seconds in loss · 23L, 23R, 32` (the
 * arrivals that land in the block, those with an aircraft dynamics model — the ones a job controls —, seconds in loss, runways).
 */
export function blockCardLines(row: TrafficScenarioBlock, blockS: number, stepS: number): [string, string] {
  return [
    blockSpan(row.startUtc, blockS),
    `${plural(row.arrivals, "arrival")} · ${row.commandable} controllable · ${secondsText(row.lossInstants, stepS)} · ` +
      blockRunwaysText(row),
  ];
}

/** A census that judged only the first arrivals says so: a bounded coverage is never silent. */
export function catalogLimitNote(catalog: TrafficScenarioCatalog): string | null {
  const { limit } = catalog.config;
  return limit === null
    ? null
    : `Partial census: only the first ${limit} arrivals by landing time were judged (a timing smoke, not the whole roster).`;
}

export type TrafficSortKey = "loss" | "time";

export interface TrafficSort {
  key: TrafficSortKey;
  descending: boolean;
}

/** The catalog's own order: the most loss instants first. */
export const DEFAULT_TRAFFIC_SORT: TrafficSort = { key: "loss", descending: true };

/**
 * What the "Sort by" select offers, in its order: a column and a direction each. The labels are short on purpose: the select is
 * half the width of a 242 px dock and must never clip its text.
 */
export const TRAFFIC_SORT_CHOICES: ReadonlyArray<{ value: string; label: string; sort: TrafficSort }> = [
  { value: "loss-desc", label: "Most losses", sort: { key: "loss", descending: true } },
  { value: "loss-asc", label: "Fewest losses", sort: { key: "loss", descending: false } },
  { value: "time-asc", label: "Earliest", sort: { key: "time", descending: false } },
  { value: "time-desc", label: "Latest", sort: { key: "time", descending: true } },
];

/** The select's value of a sort. */
export function sortChoiceValue(sort: TrafficSort): string {
  return `${sort.key}-${sort.descending ? "desc" : "asc"}`;
}

/** `rows` filtered to one runway (`""`: all of them) and sorted; equal rows keep the catalog's order. */
export function listedRows<T extends { lossInstants: number }>(
  rows: readonly T[],
  runways: (row: T) => readonly string[],
  startUtc: (row: T) => string,
  runway: string,
  sort: TrafficSort,
): T[] {
  const sign = sort.descending ? -1 : 1;
  const value = (row: T) => (sort.key === "loss" ? row.lossInstants : Date.parse(startUtc(row)));
  return rows
    .filter((row) => runway === "" || runways(row).includes(runway))
    .sort((a, b) => sign * (value(a) - value(b)));
}

/** The runways of the rows, sorted. */
export function runwayChoices<T>(rows: readonly T[], runways: (row: T) => readonly string[]): string[] {
  return [...new Set(rows.flatMap(runways))].sort();
}

/** How many rows match, as a line: `12 of 340 blocks`. */
export function matchCount(shown: number, total: number, noun: string): string {
  return `${shown} of ${total} ${noun}`;
}

/** The blocks of one length, from the catalog. */
export function catalogBlocks(catalog: TrafficScenarioCatalog, blockS: TrafficBlockLengthS) {
  return catalog.m2[String(blockS)];
}
