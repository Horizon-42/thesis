/**
 * trafficJobResult.ts
 * -------------------
 * What a finished traffic job shows: one row per controlled aircraft and one summary line, derived from the job's
 * comparison index (the groups: callsign, runway, outcome, delay) and, for an M2 job, the losses left after the block's
 * final check (the job's readout). Outcomes are named by `trafficOutcome`, the one mapping.
 */

import type { ComparisonIndex } from "../data/airportData";
import type { EvaluationRow } from "../data/evaluationReport";
import {
  TRAFFIC_JOB_SETTINGS,
  arrivalCallsign,
  type TrafficAircraftChange,
  type TrafficArrival,
  type TrafficJobSummary,
  type TrafficJobTiming,
  type TrafficScenarioCatalog,
  type TrafficStayedRecord,
} from "../data/trafficJobs";
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
  /** What changed for it (`perAircraft` of the job's status): the loss instants, the landing against its record, its time. */
  change: TrafficAircraftChange;
  /** Its path ended off the runway-threshold target (the index group's `offTarget` status: drawn yellow). */
  offTarget: boolean;
  /** The gates it failed and by how much, in words (`gateMisses`, from the job's evaluation report); empty unless `offTarget`. */
  gateMisses: GateMiss[];
}

/** An aircraft whose record is a failure carries no traffic outcome: named by what it is. */
const NO_OUTCOME_NAME = "not optimized";

export function trafficResultRows(
  index: ComparisonIndex,
  perAircraft: Record<string, TrafficAircraftChange>,
  /** The evaluation report's row of each aircraft drawn off the landing gates (by flight key): what it failed. */
  offTargetRows: Record<string, EvaluationRow>,
): TrafficResultRow[] {
  const isScene = index.scene !== undefined;
  return index.groups.map((group) => {
    // every group of a job's index is an aircraft the job was to control, and `perAircraft` says what changed for it
    const change = perAircraft[group.group];
    if (change === undefined) throw new Error(`the job's status says nothing of ${group.group}, which its index lists`);
    // an M2 aircraft that was flown has its slot's delay and its losses after the block (`traffic_job.per_aircraft`)
    if (isScene && isFlown(change) && (change.delayS === null || change.blockCheckLosses === null)) {
      throw new Error(`the job's status gives the flown ${group.group} of a block no slot delay or no loss count after the block`);
    }
    const outcome = group.traffic?.outcome ?? group.scene?.outcome ?? null;
    const outcomeName = outcome === null ? NO_OUTCOME_NAME : trafficOutcomeName(outcome);
    const offTarget = group.status === "offTarget";
    if (offTarget && offTargetRows[group.group] === undefined) {
      throw new Error(`the job's evaluation report has no row of ${group.group}, which its index draws off the landing gates`);
    }
    return {
      flightKey: group.group,
      callsign: group.flightId,
      runway: group.runway,
      outcome,
      outcomeName,
      title: `${group.group} — ${outcome === null ? NO_OUTCOME_NAME : trafficOutcomeTitle(outcome)}` +
        (offTarget ? ` — the evaluation's own words: ${offTargetRows[group.group].violations.join(", ")}` : ""),
      change,
      offTarget,
      gateMisses: offTarget ? gateMisses(offTargetRows[group.group]) : [],
    };
  });
}

/**
 * The loss instants an aircraft's record has in the census (its own record, as flown, in its recorded traffic): its row of
 * the catalog's M1 list; none there means none — unless the census judged only part of the roster (`config.limit`), which
 * leaves it unknown (null).
 */
export function recordLosses(catalog: TrafficScenarioCatalog, flightKey: string): number | null {
  const row = catalog.m1.find((r) => r.flightKey === flightKey);
  if (row !== undefined) return row.lossInstants;
  return catalog.config.limit === null ? 0 : null;
}

const plural = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;

/** Was the aircraft flown — has it a record (an aircraft whose baseline or slot solve failed has none, only a slot's delay)? */
export function isFlown(change: TrafficAircraftChange): boolean {
  return change.firstSolveLosses !== null;
}

/** Whole seconds in words: `40 s`, `6 min`, `6 min 40 s`. */
export function formatDuration(seconds: number): string {
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  if (minutes === 0) return `${rest} s`;
  return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
}

/**
 * How many whole seconds a record lands later (plus) or earlier (minus) than the recorded flight: the ONE rounding of every
 * sentence about it (`landsText`, the notes beside it), halves away from zero, so "earlier" and "1 s earlier" never disagree.
 */
export function landingShiftS(landingVsRecordS: number): number {
  return Math.sign(landingVsRecordS) * Math.round(Math.abs(landingVsRecordS));
}

/** When the optimized flight lands against the recorded one: `lands 6 min 40 s earlier than it really did`. */
export function landsText(landingVsRecordS: number): string {
  const shift = landingShiftS(landingVsRecordS);
  if (shift === 0) return "lands when it really did";
  return `lands ${formatDuration(Math.abs(shift))} ${shift < 0 ? "earlier" : "later"} than it really did`;
}

/** An aircraft that lands more than this much earlier than it really did says why (`VECTORED_NOTE`). */
export const EARLIER_NOTE_OVER_S = 60;
/** Why a flight lands much earlier than the real one: the optimizer flies the published route; the real flight was vectored. */
export const VECTORED_NOTE = "(shortest published route at minimum time; the real flight was vectored)";
/** Why an M2 flight lands later than the real one: the schedule gave it a slot. */
export const SLOT_NOTE = "(its slot)";

/** The first line of an aircraft of the result: `N850DP · H25B · 05R — separated at the first solve (no re-solve)`. */
export function resultLine1(row: TrafficResultRow): string {
  return `${row.callsign} · ${row.change.type ?? "—"} · ${row.runway} — ${row.outcomeName}`;
}

const whole = (value: number) => Math.round(value);
const tenth = (value: number) => Math.round(value * 10) / 10;

/** A number a failed gate needs, or the report is not the one this reads: refused by name. */
function needed(value: number | null | undefined, row: EvaluationRow, what: string): number {
  if (value === null || value === undefined) throw new Error(`the evaluation report's row of ${row.flight_key ?? row.id} fails ${what} without its number`);
  return value;
}

/** One gate a report row failed, in words, and the evaluation's own code for it (a code this file does not know: shown as it is). */
export interface GateMiss {
  code: string;
  text: string;
}

/**
 * The evaluation's codes for an event that has no measured miss (`event_status` of a computed record that never gave a threshold
 * crossing, `evaluation/arrival.py`) — a MIRROR, pinned by `aeroviz-4d/python/tests/test_traffic_mirrors.py`; and the sentence the
 * evaluation writes into the row's `reason` for the first of them, which holds the only amount there is.
 */
export const NOT_REACHED_CODE = "not_reached";
export const NOT_BRACKETED_CODE = "threshold_not_bracketed";
const NOT_REACHED_REASON = /^trajectory ended ([0-9.]+) m before the threshold plane$/;

/**
 * The gates one report row failed, and by how much, in words — from the row's own verdicts (`violations`): the three gates
 * `lateral 41 m too far (341 m, limit 300 m)`, `vertical 17 m too high (27 m, limits ±22 m)`, `speed 3.2 m/s too fast (63.4 m/s,
 * window 55.2–60.2 m/s)`, and the events that never crossed the threshold: `not_reached` — `did not reach the runway threshold
 * (stopped 87 m short)`, the 87 m the row's `reason` gives — and `threshold_not_bracketed` — `ended past the runway threshold
 * with no crossing to measure (its last segment does not cross the threshold plane)`. The numbers are the report's: the miss
 * distance and its runway-half-width bound, the signed altitude miss and its bounds, the crossing airspeed and its published
 * window. A code none of these is shown as the evaluation names it.
 */
export function gateMisses(row: EvaluationRow): GateMiss[] {
  return row.violations.map((code) => ({ code, text: gateMissText(code, row) }));
}

function gateMissText(violation: string, row: EvaluationRow): string {
  if (violation === "lateral") {
    const miss = needed(row.lateral_m, row, "lateral");
    const limit = row.bounds.lateral_m;
    return `lateral ${whole(miss - limit)} m too far (${whole(miss)} m, limit ${whole(limit)} m)`;
  }
  if (violation === "vertical") {
    const miss = needed(row.vertical_m, row, "vertical");
    const lower = needed(row.bounds.vertical_lower_m, row, "vertical"), upper = needed(row.bounds.vertical_upper_m, row, "vertical");
    const limits = lower === -upper ? `±${whole(upper)} m` : `${whole(lower)} to ${whole(upper)} m`;
    return miss > upper
      ? `vertical ${whole(miss - upper)} m too high (${whole(miss)} m, limits ${limits})`
      : `vertical ${whole(lower - miss)} m too low (${whole(miss)} m, limits ${limits})`;
  }
  if (violation === "speed") {
    const speed = needed(row.crossing_speed_ms, row, "speed");
    const lower = needed(row.bounds.speed_lower_ms, row, "speed"), upper = needed(row.bounds.speed_upper_ms, row, "speed");
    const window = `window ${tenth(lower)}–${tenth(upper)} m/s`;
    return speed > upper
      ? `speed ${tenth(speed - upper)} m/s too fast (${tenth(speed)} m/s, ${window})`
      : `speed ${tenth(lower - speed)} m/s too slow (${tenth(speed)} m/s, ${window})`;
  }
  if (violation === NOT_REACHED_CODE) {
    const short = NOT_REACHED_REASON.exec(row.reason ?? "");
    if (short === null) throw new Error(`the evaluation report's row of ${row.flight_key ?? row.id} says ${NOT_REACHED_CODE} without how far short`);
    return `did not reach the runway threshold (stopped ${whole(Number(short[1]))} m short)`;
  }
  if (violation === NOT_BRACKETED_CODE) {
    return "ended past the runway threshold with no crossing to measure (its last segment does not cross the threshold plane)";
  }
  return violation;
}

/**
 * The second line: what changed and where it lands, one number each, labelled so they never contradict —
 * M1: `losses: record 28 → first solve 9 → final 0`; M2: `when flown: record 28 → first solve 9 → final 0` and, after the
 * whole block has been checked, `after the whole block: 3 (from aircraft flown after it)`; where it lands against the
 * recorded flight (`lands 6 min 40 s earlier than it really did`, with the reason when it is more than a minute earlier, or an M2
 * flight's slot when later), an M2 job's `slot delay 80 s (after its earliest arrival)`, and the solver's time and solves; off
 * the landing gates, which gates and by how much. An aircraft that was not flown says so, what the schedule gave it and the time
 * its attempts took (or that the failure lost it). `record`: its loss instants in the census (`recordLosses`), counted at
 * `recordStepS`; when that is not the job's own check step the card says so — the two counts are of different instants.
 */
export function resultLine2(row: TrafficResultRow, record: number | null, recordStepS: number, isScene: boolean): string {
  const { change } = row;
  if (!isFlown(change)) {
    const why = !isScene ? "not flown"
      : change.delayS === null ? "not flown — it has no slot (its earliest-arrival solve failed)"
      : `not flown — the schedule gave it a slot ${Math.round(change.delayS)} s after its earliest arrival`;
    return `${why} · ${triedText(change, isScene)}`;
  }
  const steps = recordStepS === TRAFFIC_JOB_SETTINGS.stepS ? "" : ` at ${recordStepS} s steps`;
  const losses = `record ${record ?? "?"}${steps} → first solve ${change.firstSolveLosses} → final ${change.finalLosses}`;
  const parts = [isScene ? `when flown: ${losses}` : `losses: ${losses}`];
  if (isScene) {
    parts.push(`after the whole block: ${change.blockCheckLosses}${change.blockCheckLosses! > 0 ? " (from aircraft flown after it)" : ""}`);
  }
  parts.push(landsLine(change.landingVsRecordS!, isScene));
  if (isScene) parts.push(`slot delay ${Math.round(change.delayS!)} s (after its earliest arrival)`);
  parts.push(`optimized in ${solverSpent(change, isScene)!}`);
  if (row.offTarget) parts.push(`missed the landing gates (yellow): ${row.gateMisses.map((miss) => miss.text).join("; ")}`);
  return parts.join(" · ");
}

/** `lands … than it really did`, and why when it is far earlier (the real flight was vectored) or, in a block, later (its slot). */
function landsLine(landingVsRecordS: number, isScene: boolean): string {
  const shift = landingShiftS(landingVsRecordS);
  const text = landsText(landingVsRecordS);
  if (shift < -EARLIER_NOTE_OVER_S) return `${text} ${VECTORED_NOTE}`;
  return isScene && shift > 0 ? `${text} ${SLOT_NOTE}` : text;
}

/** What the solver spent on an aircraft: `12.3 s, CPU 11.8 s (3 solves, 0 failed)`; null when its failure lost the solves. */
function solverSpent(change: TrafficAircraftChange, isScene: boolean): string | null {
  if (change.optimizeS === null) return null;
  const solves = `${plural(change.solves!, "solve")}, ${change.failedSolves} failed${isScene ? ", the earliest-arrival solves included" : ""}`;
  return `${change.optimizeS.toFixed(1)} s, CPU ${change.optimizeCpuS!.toFixed(1)} s (${solves})`;
}

/** What an aircraft that was not flown spent: `tried for 12.3 s, …`, or that nothing of it was recorded. */
function triedText(change: TrafficAircraftChange, isScene: boolean): string {
  const spent = solverSpent(change, isScene);
  return spent === null ? "time not recorded (the failure lost it)" : `tried for ${spent}`;
}

/** `0:45`, `12:21`, `65:00` (whole seconds, the minutes not padded). */
export function formatMinutesSeconds(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** The job's wall time: `Total 12:21 — reading traffic 0:45 · earliest arrivals 3:10 · flying 8:02 · …`, the stages in order. */
export function timingText(timing: TrafficJobTiming): string {
  const stages = Object.entries(timing.phases).map(([name, seconds]) => `${name} ${formatMinutesSeconds(seconds)}`);
  return `Total ${formatMinutesSeconds(timing.totalS)} — ${stages.join(" · ")}`;
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
  /** M2 only: the arrivals of the block that had no dynamics model and stayed their records (the job's `stayedRecords`). */
  records: number | null;
  /** By plain name, most frequent first (ties in the order of the rows). */
  outcomes: TrafficOutcomeCount[];
  /** The aircraft whose path ended off the runway-threshold target (a yellow path), by callsign. */
  offTarget: string[];
  /** M2 only, and only when some aircraft has a slot. */
  delays: TrafficDelays | null;
  /**
   * M2 only: after the block's final check, the flown aircraft with a loss they answer for (who they are, by callsign, and so how
   * many: both from the job's `perAircraft`, the numbers on the cards), and the aircraft with a loss others caused (the readout's).
   */
  lossesLeft: { answered: number; notAnswered: number; names: string[] } | null;
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
  /** The arrivals of the block that stayed their records (`stayedRecords` of the job's status; {} for an M1 job). */
  stayed: Record<string, TrafficStayedRecord>,
): TrafficResultSummary {
  const counts = new Map<string, number>();
  for (const row of rows) counts.set(row.outcomeName, (counts.get(row.outcomeName) ?? 0) + 1);
  const outcomes = [...counts].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);
  // the delays of the aircraft that were flown, from the same `perAircraft` the cards read (one source); one that was not flown
  // has a slot's delay too, but it is the schedule's, not a flight's
  const delays = rows.flatMap((row) => (isFlown(row.change) && row.change.delayS !== null ? [row.change.delayS] : []));
  const left = summary?.flown_aircraft_with_a_loss_left_after_the_block?.visual;
  const names = rows.filter((row) => (row.change.blockCheckLosses ?? 0) > 0).map((row) => row.callsign);
  return {
    aircraft: rows.length,
    records: isScene ? Object.keys(stayed).length : null,
    outcomes,
    offTarget: rows.filter((row) => row.offTarget).map((row) => row.callsign),
    delays: isScene && delays.length > 0
      ? { medianS: median(delays), maxS: Math.max(...delays), overMinute: delays.filter((d) => d > 60).length }
      : null,
    lossesLeft: isScene && left ? { answered: names.length, notAnswered: left.not_answered, names } : null,
  };
}

/** The summary as one line of text. */
export function trafficSummaryText(summary: TrafficResultSummary): string {
  const parts = [
    `${summary.aircraft} aircraft controlled: ${summary.outcomes.map((o) => `${o.count} ${o.name}`).join(", ")}`,
  ];
  if (summary.offTarget.length > 0) {
    parts.push(`${summary.offTarget.length} missed the landing gates (yellow path): ${summary.offTarget.join(", ")}`);
  }
  if (summary.records !== null && summary.records > 0) {
    parts.push(`${summary.records} more landed in the block and flew ${summary.records === 1 ? "its record" : "their records"} ` +
      "(not controllable, named below)");
  }
  if (summary.delays) {
    const { medianS, maxS, overMinute } = summary.delays;
    parts.push(`slot delay median ${Math.round(medianS)} s, largest ${Math.round(maxS)} s, ${overMinute} over 60 s`);
  }
  if (summary.lossesLeft) {
    const { answered, notAnswered, names } = summary.lossesLeft;
    const left = answered === 0
      ? "no aircraft has a loss it answers for"
      : `${answered} aircraft still ${answered === 1 ? "has" : "have"} a loss they answer for (${names.join(", ")})`;
    parts.push(`after the block's final check: ${left}; ${notAnswered} aircraft with a loss others caused`);
  }
  return parts.join(" · ");
}

/** What a stayed record is, in one line after its name: not controllable, and so it flew its record. */
export const STAYED_RECORD_REASON = "not controllable (no aircraft dynamics model), flew its record";

/** The arrivals of a block that stayed their records, named: callsign and type, then why. Their order is the job's. */
export function stayedRecordLines(stayed: Record<string, TrafficStayedRecord>): Array<{ flightKey: string; callsign: string; type: string | null }> {
  return Object.entries(stayed).map(([flightKey, record]) => ({
    flightKey,
    callsign: arrivalCallsign({ callsign: record.callsign, flightKey }),
    type: record.type,
  }));
}

const pad2 = (n: number) => String(n).padStart(2, "0");
const hhmm = (ms: number) => `${pad2(new Date(ms).getUTCHours())}:${pad2(new Date(ms).getUTCMinutes())}`;
const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/** A block's span: `2026-05-21 18:00–18:15 UTC` (a block past midnight names its second day). */
export function blockSpan(blockStartUtc: string, blockS: number): string {
  const start = Date.parse(blockStartUtc);
  const end = start + blockS * 1000;
  return day(end) === day(start)
    ? `${day(start)} ${hhmm(start)}–${hhmm(end)} UTC`
    : `${day(start)} ${hhmm(start)}–${day(end)} ${hhmm(end)} UTC`;
}

/** An M2 job's inputs in words: `Block 2026-05-21 18:00–18:15 UTC`. */
export function blockJobLabel(blockStartUtc: string, blockS: number): string {
  return `Block ${blockSpan(blockStartUtc, blockS)}`;
}

/** An M1 job's inputs in words: `FFL1206, 2026-07-16` (the callsign and the UTC day it lands). */
export function flightJobLabel(arrival: TrafficArrival): string {
  return `${arrivalCallsign(arrival)}, ${arrival.landingUtc.slice(0, 10)}`;
}

/** `07:05` (minutes and seconds; the minutes run past 59 for a long job). */
export function formatElapsed(totalS: number): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.floor(totalS / 60))}:${pad(totalS % 60)}`;
}

/** What the solver settings are said to be in one sentence (the full list is its title: `solverTitle`). */
export const SOLVER_SENTENCE = "Same settings as the batch experiments";

/** The solver settings of every job, all of them (read-only: the batch's defaults, design IM2). */
export function solverTitle(): string {
  const s = TRAFFIC_JOB_SETTINGS;
  return `max duration ${s.maxDurationS} s · rollout step ${s.rolloutDtS} s · IPOPT cap ${s.maxIterations} · ` +
    `check step ${s.stepS} s · row window ${s.rowWindowS} s · margin ${s.margin * 100} % · re-solve rounds ${s.maxRounds}`;
}

/** An aircraft of a block that has a loss in its own record, and how many seconds. */
export interface BlockLossShare {
  flightKey: string;
  callsign: string;
  seconds: number;
  /** Was it flown by the job: true, false; null when the job says nothing of it. */
  flown: boolean | null;
}

/**
 * Which aircraft carry a block's recorded losses: the arrivals of the catalog's M1 list that land in `[start, start + blockS)`,
 * most seconds first (the sum is the block's own seconds in loss: the catalog adds up the same rows). Flown or not.
 */
export function blockLossShares(
  catalog: TrafficScenarioCatalog,
  startUtc: string,
  blockS: number,
  perAircraft: Record<string, TrafficAircraftChange>,
): BlockLossShare[] {
  const start = Date.parse(startUtc);
  return catalog.m1
    .filter((row) => {
      const landing = Date.parse(row.landingUtc);
      return landing >= start && landing < start + blockS * 1000;
    })
    .map((row) => ({
      flightKey: row.flightKey,
      callsign: arrivalCallsign(row),
      seconds: row.lossInstants * catalog.config.stepS,
      flown: perAircraft[row.flightKey] === undefined ? null : isFlown(perAircraft[row.flightKey]),
    }))
    .sort((a, b) => b.seconds - a.seconds);
}

/** `The block's 252 seconds in loss in the records: SWA3131 124 · DAL88 100 (not flown) · N123AB 28`. */
export function blockLossesText(shares: readonly BlockLossShare[]): string {
  if (shares.length === 0) return "No arrival of the block has a loss in its record.";
  const total = shares.reduce((sum, share) => sum + share.seconds, 0);
  const marks = (share: BlockLossShare) => (share.flown === false ? " (not flown)" : share.flown === null ? " (not in this job)" : "");
  return `The block's ${total} seconds in loss in the records: ` +
    shares.map((share) => `${share.callsign} ${share.seconds}${marks(share)}`).join(" · ");
}
