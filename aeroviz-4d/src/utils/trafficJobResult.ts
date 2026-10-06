/**
 * trafficJobResult.ts
 * -------------------
 * What a finished traffic job shows: one row per controlled aircraft and one summary line, derived from the job's
 * comparison index (the groups: callsign, runway, outcome, delay) and, for an M2 job, the losses left after the block's
 * final check (the job's readout). Outcomes are named by `trafficOutcome`, the one mapping.
 */

import type { ComparisonIndex } from "../data/airportData";
import { arrivalCallsign, type TrafficArrival, type TrafficJobSummary } from "../data/trafficJobs";
import { trafficOutcomeName, trafficOutcomeTitle } from "./trafficOutcome";

export interface TrafficResultRow {
  flightKey: string;
  callsign: string;
  runway: string;
  /** The raw outcome string of the group; null when it has none (its record is a failure without a traffic result). */
  outcome: string | null;
  /** The plain name of the outcome. */
  outcomeName: string;
  /** Title text of the row: the flight, the plain name, then the raw string. */
  title: string;
  /** M2: the delay of its slot in seconds; null for an M1 job and for an aircraft without a slot. */
  delayS: number | null;
}

/** An aircraft whose record is a failure carries no traffic outcome: named by what it is. */
const NO_OUTCOME_NAME = "not optimized";

export function trafficResultRows(index: ComparisonIndex): TrafficResultRow[] {
  return index.groups.map((group) => {
    const outcome = group.traffic?.outcome ?? group.scene?.outcome ?? null;
    const outcomeName = outcome === null ? NO_OUTCOME_NAME : trafficOutcomeName(outcome);
    return {
      flightKey: group.group,
      callsign: group.flightId,
      runway: group.runway,
      outcome,
      outcomeName,
      title: `${group.group} — ${outcome === null ? NO_OUTCOME_NAME : trafficOutcomeTitle(outcome)}`,
      delayS: group.scene?.delayS ?? null,
    };
  });
}

export interface TrafficOutcomeCount {
  name: string;
  count: number;
}

export interface TrafficDelays {
  medianS: number;
  maxS: number;
  /** Aircraft delayed by more than a minute. */
  overMinute: number;
}

export interface TrafficResultSummary {
  aircraft: number;
  /** M2 only: the arrivals of the block that had no dynamics model and stayed their records (the summary's blocks). */
  records: number | null;
  /** By plain name, most frequent first (ties in the order of the rows). */
  outcomes: TrafficOutcomeCount[];
  /** M2 only, and only when some aircraft has a slot. */
  delays: TrafficDelays | null;
  /** M2 only: the flown aircraft with a VISUAL loss left after the block's final check — one they answer for, one they do not. */
  lossesLeft: { answered: number; notAnswered: number } | null;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarizeTrafficResult(
  rows: readonly TrafficResultRow[],
  summary: TrafficJobSummary | undefined,
  isScene: boolean,
): TrafficResultSummary {
  const counts = new Map<string, number>();
  for (const row of rows) counts.set(row.outcomeName, (counts.get(row.outcomeName) ?? 0) + 1);
  const outcomes = [...counts].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);
  const delays = rows.flatMap((row) => (row.delayS === null ? [] : [row.delayS]));
  const left = summary?.flown_aircraft_with_a_loss_left_after_the_block?.visual;
  return {
    aircraft: rows.length,
    records: isScene && summary?.skipped_no_dynamics !== undefined ? summary.skipped_no_dynamics : null,
    outcomes,
    delays: isScene && delays.length > 0
      ? { medianS: median(delays), maxS: Math.max(...delays), overMinute: delays.filter((d) => d > 60).length }
      : null,
    lossesLeft: isScene && left ? { answered: left.answered, notAnswered: left.not_answered } : null,
  };
}

/** The summary as one line of text. */
export function trafficSummaryText(summary: TrafficResultSummary): string {
  const parts = [
    `${summary.aircraft} aircraft controlled: ${summary.outcomes.map((o) => `${o.count} ${o.name}`).join(", ")}`,
  ];
  if (summary.records !== null && summary.records > 0) {
    parts.push(`${summary.records} more landed in the block and stayed records (no aircraft dynamics model)`);
  }
  if (summary.delays) {
    const { medianS, maxS, overMinute } = summary.delays;
    parts.push(`delay median ${Math.round(medianS)} s, largest ${Math.round(maxS)} s, ${overMinute} over 60 s`);
  }
  if (summary.lossesLeft) {
    parts.push(
      `after the block's final check: ${summary.lossesLeft.answered} aircraft with a loss of visual separation they ` +
        `answer for, ${summary.lossesLeft.notAnswered} with one they do not`,
    );
  }
  return parts.join(" · ");
}

/** The arrivals whose recorded landing is in the block `[start, start + blockS)`. */
export function arrivalsInBlock(arrivals: readonly TrafficArrival[], startUtc: string, blockS: number): TrafficArrival[] {
  const start = Date.parse(startUtc);
  if (Number.isNaN(start)) return [];
  return arrivals.filter((a) => {
    const landing = Date.parse(a.landingUtc);
    return landing >= start && landing < start + blockS * 1000;
  });
}

const pad2 = (n: number) => String(n).padStart(2, "0");
const hhmm = (ms: number) => `${pad2(new Date(ms).getUTCHours())}:${pad2(new Date(ms).getUTCMinutes())}`;
const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** An M2 job's inputs in words: `Block 2026-05-21 18:00–18:15 UTC` (a block past midnight names its second day). */
export function blockJobLabel(blockStartUtc: string, blockS: number): string {
  const start = Date.parse(blockStartUtc);
  const end = start + blockS * 1000;
  return day(end) === day(start)
    ? `Block ${day(start)} ${hhmm(start)}–${hhmm(end)} UTC`
    : `Block ${day(start)} ${hhmm(start)}–${day(end)} ${hhmm(end)} UTC`;
}

/** An M1 job's inputs in words: `FFL1206, 2026-07-16` (the callsign and the UTC day it lands). */
export function flightJobLabel(arrival: TrafficArrival): string {
  return `${arrivalCallsign(arrival)}, ${arrival.landingUtc.slice(0, 10)}`;
}

/** The block start (`HH` and `MM`) as an ISO UTC time on `date`. */
export function blockStartOf(date: string, hour: string, minute: string): string {
  return `${date}T${hour}:${minute}:00Z`;
}

/** The hour of the first landing of the day (`HH`, rounded down), or `00` when nothing lands. */
export function defaultBlockHour(arrivals: readonly TrafficArrival[]): string {
  return arrivals.length === 0 ? "00" : arrivals[0].landingUtc.slice(11, 13);
}

export const BLOCK_START_HOURS = Array.from({ length: 24 }, (_, hour) => pad2(hour));
export const BLOCK_START_MINUTES = ["00", "15", "30", "45"] as const;
