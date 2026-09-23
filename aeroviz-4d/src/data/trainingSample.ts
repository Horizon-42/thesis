/**
 * trainingSample.ts
 * -----------------
 * The Training module's data contract: the manifest of exported sets, and one set's flights
 * under the instruction vocabulary (`instruction-v2`, spec `103a6eae6b90`).
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`; the words:
 * `4dTrajectory/ts_transformer/docs/2026-09-23_instruction_vocabulary_design.zh.md`.
 *
 * A SENTENCE IS SIX COLUMNS PER 2 s STEP — runway pointer, approach, heading, altitude, angle,
 * speed, in that order — and every column has "unchanged" (-1). Step 0 carries all six; after it
 * the sentence is mostly silent. A WORD IS A TARGET PLUS THE ENVELOPE IT ALLOWS FROM ITS ISSUE
 * POINT: a heading word's turn region, where its turn may end and its hold funnel, the capture
 * corridor, an altitude word's tube, a speed word's transition and band.
 *
 * THIS FILE COMPUTES NO ENVELOPE. Every region, band and tube is geometry the exporter computed
 * in Python from the vocabulary's own functions (`instructions/display.py` over `envelope.py`
 * and `labeller/*`), and every verdict is the labeller's. The reader checks the file's
 * BOOKKEEPING — lengths, rows, that the per-row words are the events filled forward, that a
 * verdict's count is the count of its own per-row flags — and draws numbers. A second
 * implementation of the envelopes here would be a second answer on one screen.
 *
 * NO COMPATIBILITY. The schema, the reading rule and the spec sha are pinned below and a file
 * that carries anything else is refused by name; a set of a superseded vocabulary stays listed in
 * the manifest and is refused by name without being downloaded.
 *
 * VALIDATION IS PER SET: `parseTrainingIndex` keeps the good sets and returns the rejected ones
 * with the field that failed, instead of emptying the airport (the comparison picker's
 * `.every(...)` did that twice — AV6).
 *
 * SI units only: metres, m/s, degrees, seconds.
 */

import { fetchJson } from "../utils/fetchJson";

/** MIRROR of the exporter's `INDEX_SCHEMA`. The index keeps this shape across vocabularies:
 *  every set, current or superseded, is listed in it. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v1";
/** MIRROR of the exporter's `SAMPLE_SCHEMA` (`instruction_training_export.py`). The name changes
 *  with the file's shape, on both sides, in the same change: v4 is `instruction-v2`'s sample, and a
 *  file under any other name — `v3` included, whatever vocabulary it carries — is refused. */
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v4";
/** MIRROR of `instructions.spec.READING_RULE`: what a word MEANS, which no field can say. */
export const TRAINING_READING_RULE = "instruction-v2";
/** MIRROR of the spec's sha (`v2_20260924/spec.json`). A new vocabulary is a new sha, and this
 *  reader is bound to the one it was written for. */
export const TRAINING_SPEC_SHA256 = "103a6eae6b9083ad33d5a1b25daef4004a07458881786e90d4625ba04027204b";
/** MIRROR of the exporter's `KIND_READBACK`: the one kind of set this reader opens. */
export const TRAINING_READABLE_SET_KIND = "vocabulary-readback";
/** The set kinds the manifest may list. A `prior-generated` set has no contract under this
 *  vocabulary yet — no prior is trained — so it is listed and refused by name. */
export const TRAINING_SET_KINDS = ["vocabulary-readback", "prior-generated"] as const;
export type TrainingSetKind = (typeof TRAINING_SET_KINDS)[number];

/** MIRROR of `instructions.words.COLUMNS`. The columns are POSITIONAL: this order is the
 *  order of every `inForce` table and of the six rows the sentence bar draws. */
export const TRAINING_COLUMNS = ["runway", "approach", "heading", "altitude", "angle", "speed"] as const;
export type TrainingColumn = (typeof TRAINING_COLUMNS)[number];
/** MIRROR of `instructions.words.UNCHANGED`. */
export const TRAINING_UNCHANGED = -1;
/** MIRROR of `instructions.readout.STRATA`. */
export const TRAINING_STRATA = ["straight-in", "vectored"] as const;
export type TrainingStratum = (typeof TRAINING_STRATA)[number];

/** MIRROR of the exporter's `WORD_KINDS`: why the labeller issued a word. A kind outside it is
 *  refused by name — the views name every kind, and an unnamed one would be shown as a code. */
export const TRAINING_WORD_KINDS = [
  "initial", "turn", "turn-split", "intercept", "intercept-split", "clear", "target", "step", "angle", "unspecified",
] as const;
export type TrainingWordKind = (typeof TRAINING_WORD_KINDS)[number];

export const TRAINING_COLUMN_INDEX: Record<TrainingColumn, number> = {
  runway: 0, approach: 1, heading: 2, altitude: 3, angle: 4, speed: 5,
};

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingCohort {
  split: string;
  perStratum: number;
  seed: number;
  drawnFrom: string;
}

export interface TrainingSetEntry {
  id: string;
  kind: TrainingSetKind;
  title: string;
  /** Path of the sample file, relative to the airport's `training/` directory. */
  file: string;
  /** The vocabulary spec's sha. */
  vocabularySha256: string;
  /** The sha of the airport's geometry: its candidate runways (the runway pointer's choices) and
   *  every runway end the landing rule reads (the sample's `candidatesSha256`). */
  runwaySha256: string;
  readingRule: string;
  flights: number;
  cohort: TrainingCohort;
}

export interface TrainingIndex {
  airport: string;
  sets: TrainingSetEntry[];
  /** Entries that are not set entries at all, each with the field that failed. */
  rejected: Array<{ id: string; problem: string }>;
}

export interface TrainingAngleClass {
  value: number;
  name: string;
  nominalDeg: number;
  /** The class's range; a climb is negative, level is [0, 0]. */
  lowDeg: number;
  steepDeg: number;
}

/** What the words mean: every class as a table, so a label is a lookup, never a decode. */
export interface TrainingVocabulary {
  readingRule: string;
  specSha256: string;
  labellerSourceSha256: string;
  stepS: number;
  smoothingS: { track: number; altitude: number; speed: number };
  classCounts: Record<Exclude<TrainingColumn, "runway">, number>;
  approachClasses: string[];
  headingTargetsDeg: number[];
  headingToleranceDeg: number;
  headingMaxTurnDeg: number;
  /** A turn's rate: at most `turnRateMaxDegS` on every row (and at most `turnBankMaxDeg` of bank at
   *  the flown speed); at least `turnRateMinDegS` on average for a turn of `turnRateMinFromDeg` or
   *  more. It may begin up to `turnStartDelayMaxS` after the word. */
  turnRateMinDegS: number;
  turnRateMaxDegS: number;
  turnRateMinFromDeg: number;
  turnBankMaxDeg: number;
  turnStartDelayMaxS: number;
  interceptAngleDeg: number;
  corridorHalfWidthM: number;
  corridorWideningDeg: number;
  corridorCourseToleranceDeg: number;
  /** The landing: the threshold passed within this far of the centreline (capped per runway by a
   *  parallel runway — `TrainingCandidate.landingCrossLimitM`) and this high above the threshold. */
  landingCrossLimitM: number;
  landingMaxHeightM: number;
  /** Two runways within this of each other's course are parallel partners for the cap. */
  parallelCourseDeltaDeg: number;
  altitudeTargetsM: number[];
  /** The altitude value "descend to land": one past the last target. */
  altitudeLandValue: number;
  altitudeToleranceM: number;
  angleClasses: TrainingAngleClass[];
  /** The angle value "level": a target reached, not a slope. */
  angleLevelValue: number;
  speedTargetsMps: number[];
  /** The speed value "unspecified": one past the last target. */
  speedUnspecifiedValue: number;
  speedToleranceMps: number;
  speedAccelMaxMps2: number;
  speedRangeMps: [number, number];
}

/** Points in the airport frame (metres east / north of the reference point) and the same points
 *  on the globe — one computation at the exporter, two spellings. */
export interface TrainingPlanLine {
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
}

export interface TrainingCandidate {
  index: number;
  ident: string;
  thresholdEM: number;
  thresholdNM: number;
  courseDeg: number;
  elevationM: number;
  lengthM: number;
  /** The landing's limit off this runway's centreline: the vocabulary's, capped at half the spacing
   *  to a parallel runway. */
  landingCrossLimitM: number;
  /** From the threshold out along the approach side, `centrelineLengthM` long. */
  centreline: TrainingPlanLine;
  /** The runway: the threshold to its far end. */
  runway: TrainingPlanLine;
}

export interface TrainingWordEvent {
  row: number;
  column: number;
  value: number;
  /** Why the labeller issued it. */
  kind: TrainingWordKind;
}

/**
 * Where a turn may take the aircraft: between its fastest and its slowest turn, flown at the speeds
 * the flight flew, begun on time or as late as allowed.
 */
export interface TrainingTurnRegion {
  fromTrackDeg: number;
  /** The shorter way to the target, positive = right. */
  turnDeg: number;
  rateMinDegS: number;
  rateMaxDegS: number;
  bankMaxDeg: number;
  startDelayMaxS: number;
  /** The slowest turn reaches the target's band before the flight ends. When it does not, its path
   *  stops at the flight's end and the region is cut there. */
  slowFinished: boolean;
  /** A ring: the fastest turn begun on time, the two on-time ends, the slowest turn begun as late
   *  as allowed back to where it began, and back to the issue point. */
  region: TrainingPlanLine;
  fastPath: TrainingPlanLine;
  slowPath: TrainingPlanLine;
  /** Where the turn may end — four corners: the fastest turn's end, the slowest turn's end, then
   *  the same two moved along the issue track by the latest start. */
  end: TrainingPlanLine;
}

/** The labeller's check of a turn: monotone progress, turn rate inside the range. */
export interface TrainingTurnCheck {
  progressOk: boolean;
  rateOk: boolean;
  meanRateDegS: number;
  maxRateDegS: number;
  maxBankDeg: number;
  /** The lowest rate is judged only for turns of at least `turnRateMinFromDeg`. */
  rateMinApplies: boolean;
}

/** The labeller's check of a hold: its rows, from the hold's start to its end inclusive, against
 *  the drawn funnel. */
export interface TrainingHoldCheck {
  holdStartRow: number;
  holdEndRow: number;
  rows: number;
  inside: number;
  halfWidthEndM: number;
}

export interface TrainingHeadingEnvelope {
  row: number;
  value: number;
  kind: TrainingWordKind;
  targetDeg: number;
  split: { part: number; parts: number } | null;
  turnEndRow: number | null;
  holdStartRow: number | null;
  holdEndRow: number;
  fromTrackDeg: number;
  targetOnTrackDeg: number;
  turnBandDeg: [number, number] | null;
  holdBandDeg: [number, number] | null;
  turn: TrainingTurnRegion | null;
  funnel: {
    lengthM: number;
    startHalfWidthM: number;
    endHalfWidthM: number;
    axis: TrainingPlanLine;
    outline: TrainingPlanLine;
  } | null;
  /** Shared by every part of a split turn. */
  check: (TrainingTurnCheck & {
    kind: string; departureRow: number; arrivalRow: number; turnDeg: number; parts: number;
  }) | null;
  /** null for a word with no hold, and for a hold the labeller does not judge: after a turn under
   *  `turnRateMinFromDeg`, or when the slowest turn does not finish. Its funnel is still drawn. */
  holdCheck: TrainingHoldCheck | null;
}

export interface TrainingApproach {
  clearanceRow: number;
  captureRow: number;
  captureBeforeThresholdM: number;
  interceptInserted: boolean;
  captureTurn: {
    startRow: number;
    courseOnTrackDeg: number;
    bandDeg: [number, number];
    check: TrainingTurnCheck;
    turn: TrainingTurnRegion;
  } | null;
  /** The course ± its tolerance, on the heading chart's branch, from the capture on. */
  courseBandDeg: [number, number];
  corridor: {
    beforeThresholdM: number;
    halfWidthAtCaptureM: number;
    halfWidthAtThresholdM: number;
    rows: number;
    axis: TrainingPlanLine;
    outline: TrainingPlanLine;
  };
  landing: {
    cutAtCrossing: boolean;
    lastRowBeforeThresholdM: number;
    crossing: { row: number; eM: number; nM: number; lon: number; lat: number } | null;
  };
}

export interface TrainingAltitudeTube {
  row: number;
  /** One past the last row the tube covers. */
  endRow: number;
  value: number;
  kind: TrainingWordKind;
  /** null = "descend to land". */
  targetM: number | null;
  lowerM: number[];
  upperM: number[];
  lowerHaeM: number[];
  upperHaeM: number[];
  inside: boolean[];
  check: { rows: number; inside: number; contained: boolean; tubeWidthEndM: number };
}

export interface TrainingAngleWord {
  row: number;
  value: number;
  kind: TrainingWordKind;
  /** The straight piece's measured angle; null for level (a target reached, not a slope). */
  measuredDeg: number | null;
}

export interface TrainingSpeedSpan {
  row: number;
  endRow: number;
  value: number;
  kind: TrainingWordKind;
  /** null = "unspecified". */
  targetMps: number | null;
  arrivalRow: number | null;
  transitionLowerMps: number[] | null;
  transitionUpperMps: number[] | null;
  bandMps: [number, number] | null;
  bandInside: boolean[] | null;
  check: {
    arrivalRows: number; cutBeforeArrival: boolean; transitionOk: boolean; accelOk: boolean;
    bandRows: number; bandInside: number; contained: boolean;
  } | null;
  rangeMps: [number, number] | null;
}

export interface TrainingSignals {
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeHaeM: number[];
  raw: { trackDeg: number[]; altitudeM: number[]; groundSpeedMps: number[]; verticalRateMps: number[] };
  smoothed: { trackDeg: number[]; altitudeM: number[]; groundSpeedMps: number[]; distanceM: number[] };
  beforeThresholdM: number[];
  rightOfCourseM: number[];
}

export interface TrainingFlight {
  datasetId: string;
  flightKey: string;
  callsign: string;
  typecode: string;
  runway: string;
  runwayIndex: number;
  stratum: TrainingStratum;
  /** The sentence's steps = the signal rows it covers. */
  rows: number;
  captureRow: number;
  joinRow: number;
  unspecifiedRow: number;
  captureBeforeThresholdM: number;
  signals: TrainingSignals;
  words: { events: TrainingWordEvent[]; inForce: number[][] };
  envelopes: {
    heading: TrainingHeadingEnvelope[];
    approach: TrainingApproach;
    altitude: TrainingAltitudeTube[];
    angle: TrainingAngleWord[];
    speed: TrainingSpeedSpan[];
  };
}

export interface TrainingSample {
  setId: string;
  airport: string;
  cohort: TrainingCohort & { pool: number; read: number };
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidatesSha256: string;
  centrelineLengthM: number;
  candidates: TrainingCandidate[];
  flights: TrainingFlight[];
}

/** What the panel publishes for the sentence bar, the read-back window and the 3D layer. */
export interface TrainingSelection {
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  flight: TrainingFlight;
}

export type Parsed<T> = { ok: true; value: T } | { ok: false; problem: string };

// ── reading a word ───────────────────────────────────────────────────────────

/** The class count of a column; the runway pointer's is the airport's candidate count. */
export function trainingClassCount(
  vocabulary: TrainingVocabulary, candidates: TrainingCandidate[], column: TrainingColumn,
): number {
  return column === "runway" ? candidates.length : vocabulary.classCounts[column];
}

/** A word as a person reads it. SI units; the tables decide, nothing is decoded. */
export function trainingWordLabel(
  vocabulary: TrainingVocabulary, candidates: TrainingCandidate[], column: TrainingColumn, value: number,
): string {
  switch (column) {
    case "runway":
      return candidates[value].ident;
    case "approach":
      return vocabulary.approachClasses[value];
    case "heading":
      return `${vocabulary.headingTargetsDeg[value].toFixed(0).padStart(3, "0")}°`;
    case "altitude":
      return value === vocabulary.altitudeLandValue
        ? "descend to land"
        : `${vocabulary.altitudeTargetsM[value].toFixed(0)} m`;
    case "angle": {
      const angle = vocabulary.angleClasses[value];
      return value === vocabulary.angleLevelValue ? angle.name : `${angle.name} (${angle.nominalDeg.toFixed(2)}°)`;
    }
    case "speed":
      return value === vocabulary.speedUnspecifiedValue
        ? "unspecified"
        : `${vocabulary.speedTargetsMps[value].toFixed(0)} m/s`;
  }
}

/** Why a word was issued, in words — one for every kind the labeller writes. */
const KIND_LABEL: Record<TrainingWordKind, string> = {
  initial: "in force at entry (step 0)",
  turn: "a turn",
  "turn-split": "part of a split turn",
  intercept: "an inserted intercept heading",
  "intercept-split": "part of a split inserted intercept",
  clear: "cleared to join the final",
  target: "a new target",
  step: "a step between two holds",
  angle: "a new descent angle",
  unspecified: "speed left to the pilot",
};

export function trainingKindLabel(kind: TrainingWordKind, split?: { part: number; parts: number } | null): string {
  return split ? `${KIND_LABEL[kind]} ${split.part}/${split.parts}` : KIND_LABEL[kind];
}

/** The row in force at a time: the last row at or before it. */
export function rowAtTime(tS: number[], seconds: number): number {
  if (seconds <= tS[0]) return 0;
  let low = 0;
  let high = tS.length - 1;
  if (seconds >= tS[high]) return high;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (tS[mid] <= seconds) low = mid;
    else high = mid - 1;
  }
  return low;
}

/** A time from this artefact, as every Training view writes it. */
export function formatSeconds(seconds: number): string {
  return Number.isInteger(seconds) ? `${seconds}` : seconds.toFixed(1);
}

/** One word of a column and the rows it is in force: from its issue to the next word of its column. */
export interface TrainingWordRun {
  row: number;
  endRow: number;
  value: number;
  event: TrainingWordEvent;
}

/** The runs of one column: the value in force from each of its issue rows to the next. */
export function trainingColumnRuns(flight: TrainingFlight, column: TrainingColumn): TrainingWordRun[] {
  const index = TRAINING_COLUMN_INDEX[column];
  const events = flight.words.events.filter((event) => event.column === index);
  return events.map((event, position) => ({
    row: event.row,
    endRow: position + 1 < events.length ? events[position + 1].row : flight.rows,
    value: event.value,
    event,
  }));
}

/**
 * The word of ONE column in force at a row, and its place among that column's words — which is also
 * the place of its envelope: `envelopes.heading / altitude / angle / speed` hold one entry per word of
 * their column, in order (the parser checks it). Step 0 gives every column a word and the runs tile
 * the sentence, so there is always one.
 */
export function trainingWordAt(
  flight: TrainingFlight, column: TrainingColumn, row: number,
): TrainingWordRun & { index: number } {
  const runs = trainingColumnRuns(flight, column);
  const index = runs.findIndex((run) => run.row <= row && row < run.endRow);
  return { ...runs[index], index };
}

/** The flight's verdicts, counted from the labeller's own checks. */
export function trainingVerdicts(flight: TrainingFlight) {
  // The parts of a split turn share ONE check: count each turn once, by where it departed.
  const turns = new Map<string, NonNullable<TrainingHeadingEnvelope["check"]>>();
  for (const item of flight.envelopes.heading) {
    if (item.check !== null) turns.set(`${item.check.kind}@${item.check.departureRow}`, item.check);
  }
  const checks = [...turns.values()];
  const tubes = flight.envelopes.altitude;
  const speeds = flight.envelopes.speed.flatMap((span) => (span.check === null ? [] : [span.check]));
  const issued = new Set(flight.words.events.filter((event) => event.row > 0).map((event) => event.row));
  const holds = flight.envelopes.heading.flatMap((item) => (item.holdCheck === null ? [] : [item.holdCheck]));
  return {
    turns: checks.length,
    turnsProgressOk: checks.filter((check) => check.progressOk).length,
    turnsRateOk: checks.filter((check) => check.rateOk).length,
    holdsJudged: holds.length,
    holdsContained: holds.filter((hold) => hold.inside === hold.rows).length,
    holdsNotJudged: flight.envelopes.heading.filter((item) => item.funnel !== null && item.holdCheck === null).length,
    captureTurn: flight.envelopes.approach.captureTurn?.check ?? null,
    altitudeWords: tubes.length,
    altitudeContained: tubes.filter((tube) => tube.check.contained).length,
    speedWords: speeds.length,
    speedContained: speeds.filter((check) => check.contained).length,
    instructionsAfterStep0: flight.words.events.filter((event) => event.row > 0).length,
    silentSteps: flight.rows - 1 - issued.size,
  };
}

// ── small checkers ───────────────────────────────────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(source: Record<string, unknown>, key: string): string | null {
  const value = source[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function finite(source: Record<string, unknown>, key: string): number | null {
  const value = source[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function numberArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  for (const item of value) {
    if (typeof item !== "number" || !Number.isFinite(item)) return null;
  }
  return value as number[];
}

class Refusal extends Error {}

/** Reads one object, naming the path of every field it refuses. */
class Reader {
  constructor(private readonly source: Record<string, unknown>, readonly where: string) {}

  static of(value: unknown, where: string): Reader {
    if (!isRecord(value)) throw new Refusal(`${where} is not an object`);
    return new Reader(value, where);
  }

  fail(message: string): never {
    throw new Refusal(`${this.where}: ${message}`);
  }

  at(key: string): string {
    return `${this.where}.${key}`;
  }

  raw(key: string): unknown {
    return this.source[key];
  }

  child(key: string): Reader {
    return Reader.of(this.source[key], this.at(key));
  }

  nullableChild(key: string): Reader | null {
    return this.source[key] === null ? null : this.child(key);
  }

  string(key: string): string {
    const value = str(this.source, key);
    if (value === null) throw new Refusal(`${this.at(key)} is ${JSON.stringify(this.source[key])}, not a non-empty string`);
    return value;
  }

  number(key: string): number {
    const value = finite(this.source, key);
    if (value === null) throw new Refusal(`${this.at(key)} is ${JSON.stringify(this.source[key])}, not a number`);
    return value;
  }

  nullableNumber(key: string): number | null {
    return this.source[key] === null ? null : this.number(key);
  }

  integer(key: string, low: number, high: number): number {
    const value = this.number(key);
    if (!Number.isInteger(value) || value < low || value > high) {
      throw new Refusal(`${this.at(key)} is ${value}, not a whole number in ${low}…${high}`);
    }
    return value;
  }

  /** A whole number of at least ``low`` — a count, with no upper bound to pretend to. */
  count(key: string, low: number): number {
    const value = this.number(key);
    if (!Number.isInteger(value) || value < low) {
      throw new Refusal(`${this.at(key)} is ${value}, not a whole number of at least ${low}`);
    }
    return value;
  }

  nullableInteger(key: string, low: number, high: number): number | null {
    return this.source[key] === null ? null : this.integer(key, low, high);
  }

  boolean(key: string): boolean {
    const value = this.source[key];
    if (typeof value !== "boolean") throw new Refusal(`${this.at(key)} is ${JSON.stringify(value)}, not true/false`);
    return value;
  }

  numbers(key: string, length?: number): number[] {
    const values = numberArray(this.source[key]);
    if (values === null) throw new Refusal(`${this.at(key)} is missing or not a list of numbers`);
    if (length !== undefined && values.length !== length) {
      throw new Refusal(`${this.at(key)} has ${values.length} values, expected ${length}`);
    }
    return values;
  }

  flags(key: string, length: number): boolean[] {
    return this.numbers(key, length).map((value, index) => {
      if (value !== 0 && value !== 1) throw new Refusal(`${this.at(key)}[${index}] is ${value}, not 0/1`);
      return value === 1;
    });
  }

  /** An ascending [low, high] pair. */
  range(key: string): [number, number] {
    const values = this.numbers(key, 2);
    if (values[0] > values[1]) throw new Refusal(`${this.at(key)} is inverted: ${values[0]} above ${values[1]}`);
    return [values[0], values[1]];
  }

  nullableRange(key: string): [number, number] | null {
    return this.source[key] === null ? null : this.range(key);
  }

  strings(key: string): string[] {
    const value = this.source[key];
    if (!Array.isArray(value) || !value.every((item) => typeof item === "string" && item.length > 0)) {
      throw new Refusal(`${this.at(key)} is missing or not a list of names`);
    }
    return value as string[];
  }

  list(key: string): unknown[] {
    const value = this.source[key];
    if (!Array.isArray(value)) throw new Refusal(`${this.at(key)} is missing or not a list`);
    return value;
  }
}

function parsePlanLine(reader: Reader, key: string, minimum: number): TrainingPlanLine {
  const line = reader.child(key);
  const eM = line.numbers("eM");
  if (eM.length < minimum) line.fail(`holds ${eM.length} points, fewer than ${minimum}`);
  // ONE outline in two coordinate systems: a length that differs is two shapes drawn as one.
  return { eM, nM: line.numbers("nM", eM.length), lon: line.numbers("lon", eM.length), lat: line.numbers("lat", eM.length) };
}

// ── the index ────────────────────────────────────────────────────────────────

function parseSetEntry(raw: unknown, position: number): TrainingSetEntry {
  const id = isRecord(raw) && str(raw, "id") ? (raw.id as string) : `sets[${position}]`;
  const entry = Reader.of(raw, `set ${id}`);
  const kind = entry.string("kind");
  if (!(TRAINING_SET_KINDS as readonly string[]).includes(kind)) {
    entry.fail(`kind is ${JSON.stringify(kind)}, expected one of ${TRAINING_SET_KINDS.join(", ")}`);
  }
  const flights = entry.number("flights");
  if (!Number.isInteger(flights) || flights < 0) entry.fail(`flights is ${flights}, not a count`);
  const cohort = entry.child("cohort");
  const perStratum = cohort.number("perStratum");
  if (!Number.isInteger(perStratum) || perStratum <= 0) cohort.fail(`perStratum is ${perStratum}, not a positive count`);
  return {
    id: entry.string("id"),
    kind: kind as TrainingSetKind,
    title: entry.string("title"),
    file: entry.string("file"),
    vocabularySha256: entry.string("vocabularySha256"),
    runwaySha256: entry.string("runwaySha256"),
    readingRule: entry.string("readingRule"),
    flights,
    cohort: {
      split: cohort.string("split"),
      perStratum,
      seed: cohort.number("seed"),
      drawnFrom: cohort.string("drawnFrom"),
    },
  };
}

/** Parse the manifest. A bad entry is rejected on its own; only a manifest that is not a
 *  manifest at all fails the whole call. */
export function parseTrainingIndex(raw: unknown): Parsed<TrainingIndex> {
  if (!isRecord(raw)) return { ok: false, problem: "the manifest is not an object" };
  if (raw.schema !== TRAINING_INDEX_SCHEMA) {
    return { ok: false, problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_INDEX_SCHEMA)}` };
  }
  const airport = str(raw, "airport");
  if (airport === null) return { ok: false, problem: "airport is missing" };
  if (!Array.isArray(raw.sets)) return { ok: false, problem: "sets is not an array" };
  const sets: TrainingSetEntry[] = [];
  const rejected: Array<{ id: string; problem: string }> = [];
  raw.sets.forEach((item, position) => {
    try {
      sets.push(parseSetEntry(item, position));
    } catch (error) {
      if (!(error instanceof Refusal)) throw error;
      const id = isRecord(item) && str(item, "id") ? (item.id as string) : `sets[${position}]`;
      rejected.push({ id, problem: error.message });
    }
  });
  return { ok: true, value: { airport, sets, rejected } };
}

/**
 * Why this reader will not open a listed set — by name, from the manifest alone, so a set of a
 * superseded vocabulary is never downloaded — or null when it will.
 */
export function trainingSetRefusal(entry: TrainingSetEntry): string | null {
  if (entry.readingRule !== TRAINING_READING_RULE) {
    return (
      `read under ${entry.readingRule}, a superseded vocabulary: this reader reads ${TRAINING_READING_RULE} ` +
      `(spec ${TRAINING_SPEC_SHA256.slice(0, 12)}) and nothing else`
    );
  }
  if (entry.vocabularySha256 !== TRAINING_SPEC_SHA256) {
    return (
      `spec ${entry.vocabularySha256.slice(0, 12)} is not the ${TRAINING_READING_RULE} spec ` +
      `${TRAINING_SPEC_SHA256.slice(0, 12)} this reader is written for`
    );
  }
  if (entry.kind !== TRAINING_READABLE_SET_KIND) {
    return `a ${entry.kind} set: no prior is trained on ${TRAINING_READING_RULE} yet, so this reader has no contract for its words`;
  }
  return null;
}

// ── one sample set ───────────────────────────────────────────────────────────

function parseVocabulary(reader: Reader): TrainingVocabulary {
  const rule = reader.string("readingRule");
  if (rule !== TRAINING_READING_RULE) {
    reader.fail(`readingRule is ${rule}, and this reader is written for ${TRAINING_READING_RULE} — re-export under the current vocabulary`);
  }
  const sha = reader.string("specSha256");
  if (sha !== TRAINING_SPEC_SHA256) {
    reader.fail(`specSha256 is ${sha.slice(0, 12)}, and this reader is written for spec ${TRAINING_SPEC_SHA256.slice(0, 12)}`);
  }
  const columns = reader.strings("columns");
  if (columns.join(",") !== TRAINING_COLUMNS.join(",")) {
    reader.fail(`columns are [${columns.join(", ")}], expected [${TRAINING_COLUMNS.join(", ")}] in that order`);
  }
  if (reader.number("unchanged") !== TRAINING_UNCHANGED) {
    reader.fail(`unchanged is ${reader.raw("unchanged")}, expected ${TRAINING_UNCHANGED}`);
  }
  const counts = reader.child("classCounts");
  const classCounts = {
    approach: counts.count("approach", 1),
    heading: counts.count("heading", 1),
    altitude: counts.count("altitude", 1),
    angle: counts.count("angle", 1),
    speed: counts.count("speed", 1),
  };
  const approachClasses = reader.strings("approachClasses");
  const headingTargetsDeg = reader.numbers("headingTargetsDeg");
  const altitudeTargetsM = reader.numbers("altitudeTargetsM");
  const speedTargetsMps = reader.numbers("speedTargetsMps");
  const angleClasses = reader.list("angleClasses").map((item, index) => {
    const angle = Reader.of(item, reader.at(`angleClasses[${index}]`));
    if (angle.number("value") !== index) angle.fail(`value is ${angle.raw("value")}, expected ${index}: the table is in class order`);
    return {
      value: index, name: angle.string("name"), nominalDeg: angle.number("nominalDeg"),
      lowDeg: angle.number("lowDeg"), steepDeg: angle.number("steepDeg"),
    };
  });
  // Each table against the count the vocabulary states: a table one short shifts every word.
  const tables: Array<[string, number, number]> = [
    ["approachClasses", approachClasses.length, classCounts.approach],
    ["headingTargetsDeg", headingTargetsDeg.length, classCounts.heading],
    ["altitudeTargetsM (+ descend to land)", altitudeTargetsM.length + 1, classCounts.altitude],
    ["angleClasses", angleClasses.length, classCounts.angle],
    ["speedTargetsMps (+ unspecified)", speedTargetsMps.length + 1, classCounts.speed],
  ];
  for (const [name, held, stated] of tables) {
    if (held !== stated) reader.fail(`${name} holds ${held} classes, but classCounts says ${stated}`);
  }
  const altitudeLandValue = reader.number("altitudeLandValue");
  if (altitudeLandValue !== altitudeTargetsM.length) {
    reader.fail(`altitudeLandValue is ${altitudeLandValue}, expected ${altitudeTargetsM.length} (one past the last target)`);
  }
  const angleLevelValue = reader.integer("angleLevelValue", 0, angleClasses.length - 1);
  const speedUnspecifiedValue = reader.number("speedUnspecifiedValue");
  if (speedUnspecifiedValue !== speedTargetsMps.length) {
    reader.fail(`speedUnspecifiedValue is ${speedUnspecifiedValue}, expected ${speedTargetsMps.length} (one past the last target)`);
  }
  const smoothing = reader.child("smoothingS");
  return {
    readingRule: rule,
    specSha256: sha,
    labellerSourceSha256: reader.string("labellerSourceSha256"),
    stepS: reader.number("stepS"),
    smoothingS: { track: smoothing.number("track"), altitude: smoothing.number("altitude"), speed: smoothing.number("speed") },
    classCounts,
    approachClasses,
    headingTargetsDeg,
    headingToleranceDeg: reader.number("headingToleranceDeg"),
    headingMaxTurnDeg: reader.number("headingMaxTurnDeg"),
    turnRateMinDegS: reader.number("turnRateMinDegS"),
    turnRateMaxDegS: reader.number("turnRateMaxDegS"),
    turnRateMinFromDeg: reader.number("turnRateMinFromDeg"),
    turnBankMaxDeg: reader.number("turnBankMaxDeg"),
    turnStartDelayMaxS: reader.number("turnStartDelayMaxS"),
    interceptAngleDeg: reader.number("interceptAngleDeg"),
    corridorHalfWidthM: reader.number("corridorHalfWidthM"),
    corridorWideningDeg: reader.number("corridorWideningDeg"),
    corridorCourseToleranceDeg: reader.number("corridorCourseToleranceDeg"),
    landingCrossLimitM: reader.number("landingCrossLimitM"),
    landingMaxHeightM: reader.number("landingMaxHeightM"),
    parallelCourseDeltaDeg: reader.number("parallelCourseDeltaDeg"),
    altitudeTargetsM,
    altitudeLandValue,
    altitudeToleranceM: reader.number("altitudeToleranceM"),
    angleClasses,
    angleLevelValue,
    speedTargetsMps,
    speedUnspecifiedValue,
    speedToleranceMps: reader.number("speedToleranceMps"),
    speedAccelMaxMps2: reader.number("speedAccelMaxMps2"),
    speedRangeMps: reader.range("speedRangeMps"),
  };
}

function parseCandidates(reader: Reader, vocabulary: TrainingVocabulary): TrainingCandidate[] {
  const candidates = reader.list("candidates").map((item, index) => {
    const candidate = Reader.of(item, reader.at(`candidates[${index}]`));
    if (candidate.number("index") !== index) candidate.fail(`index is ${candidate.raw("index")}, expected ${index}`);
    // A parallel runway can only tighten the landing's limit, never loosen it.
    const landingCrossLimitM = candidate.number("landingCrossLimitM");
    if (!(landingCrossLimitM > 0 && landingCrossLimitM <= vocabulary.landingCrossLimitM)) {
      candidate.fail(`landingCrossLimitM is ${landingCrossLimitM}, not in (0, ${vocabulary.landingCrossLimitM}] m`);
    }
    return {
      index,
      ident: candidate.string("ident"),
      thresholdEM: candidate.number("thresholdEM"),
      thresholdNM: candidate.number("thresholdNM"),
      courseDeg: candidate.number("courseDeg"),
      elevationM: candidate.number("elevationM"),
      lengthM: candidate.number("lengthM"),
      landingCrossLimitM,
      centreline: parsePlanLine(candidate, "centreline", 2),
      runway: parsePlanLine(candidate, "runway", 2),
    };
  });
  if (candidates.length === 0) reader.fail("candidates is empty: the runway pointer has nothing to point at");
  return candidates;
}

function parseTurnRegion(reader: Reader, vocabulary: TrainingVocabulary): TrainingTurnRegion {
  // The region's limits are the vocabulary's own: a region drawn with other numbers is another turn.
  const limits: Array<[string, number]> = [
    ["rateMinDegS", vocabulary.turnRateMinDegS], ["rateMaxDegS", vocabulary.turnRateMaxDegS],
    ["bankMaxDeg", vocabulary.turnBankMaxDeg], ["startDelayMaxS", vocabulary.turnStartDelayMaxS],
  ];
  for (const [key, value] of limits) {
    if (reader.number(key) !== value) reader.fail(`${key} is ${reader.raw(key)}, but the vocabulary says ${value}`);
  }
  const end = parsePlanLine(reader, "end", 4);
  if (end.eM.length !== 4) reader.fail(`end has ${end.eM.length} corners; where a turn may end has four`);
  return {
    fromTrackDeg: reader.number("fromTrackDeg"),
    turnDeg: reader.number("turnDeg"),
    rateMinDegS: vocabulary.turnRateMinDegS,
    rateMaxDegS: vocabulary.turnRateMaxDegS,
    bankMaxDeg: vocabulary.turnBankMaxDeg,
    startDelayMaxS: vocabulary.turnStartDelayMaxS,
    slowFinished: reader.boolean("slowFinished"),
    region: parsePlanLine(reader, "region", 3),
    // A path is ONE point when the track is already within the hold band of the target at issue:
    // there is nothing left to turn (a small capture turn, mostly).
    fastPath: parsePlanLine(reader, "fastPath", 1),
    slowPath: parsePlanLine(reader, "slowPath", 1),
    end,
  };
}

function parseTurnCheck(reader: Reader): TrainingTurnCheck {
  return {
    progressOk: reader.boolean("progressOk"),
    rateOk: reader.boolean("rateOk"),
    meanRateDegS: reader.number("meanRateDegS"),
    maxRateDegS: reader.number("maxRateDegS"),
    maxBankDeg: reader.number("maxBankDeg"),
    rateMinApplies: reader.boolean("rateMinApplies"),
  };
}

function parseSignals(reader: Reader, rows: number, stepS: number): TrainingSignals {
  const tS = reader.numbers("tS", rows);
  if (tS[0] !== 0) reader.fail(`tS starts at ${tS[0]} s, not 0`);
  for (let row = 1; row < rows; row += 1) {
    if (Math.abs(tS[row] - tS[row - 1] - stepS) > 1e-3) {
      reader.fail(`tS steps ${tS[row - 1]} → ${tS[row]} at row ${row}, not the vocabulary's ${stepS} s`);
    }
  }
  const raw = reader.child("raw");
  const smoothed = reader.child("smoothed");
  const distanceM = smoothed.numbers("distanceM", rows);
  for (let row = 1; row < rows; row += 1) {
    if (distanceM[row] < distanceM[row - 1]) smoothed.fail(`distanceM falls at row ${row}: a path length only grows`);
  }
  return {
    tS,
    eM: reader.numbers("eM", rows),
    nM: reader.numbers("nM", rows),
    lon: reader.numbers("lon", rows),
    lat: reader.numbers("lat", rows),
    altitudeHaeM: reader.numbers("altitudeHaeM", rows),
    raw: {
      trackDeg: raw.numbers("trackDeg", rows),
      altitudeM: raw.numbers("altitudeM", rows),
      groundSpeedMps: raw.numbers("groundSpeedMps", rows),
      verticalRateMps: raw.numbers("verticalRateMps", rows),
    },
    smoothed: {
      trackDeg: smoothed.numbers("trackDeg", rows),
      altitudeM: smoothed.numbers("altitudeM", rows),
      groundSpeedMps: smoothed.numbers("groundSpeedMps", rows),
      distanceM,
    },
    beforeThresholdM: reader.numbers("beforeThresholdM", rows),
    rightOfCourseM: reader.numbers("rightOfCourseM", rows),
  };
}

function parseWords(
  reader: Reader, rows: number, counts: number[],
): { events: TrainingWordEvent[]; inForce: number[][] } {
  const events = reader.list("events").map((item, index) => {
    const event = Reader.of(item, reader.at(`events[${index}]`));
    const column = event.integer("column", 0, TRAINING_COLUMNS.length - 1);
    const kind = event.string("kind");
    if (!(TRAINING_WORD_KINDS as readonly string[]).includes(kind)) {
      event.fail(`kind is ${kind}, not one of the labeller's ${TRAINING_WORD_KINDS.join(", ")}`);
    }
    return {
      row: event.integer("row", 0, rows - 1),
      column,
      value: event.integer("value", 0, counts[column] - 1),
      kind: kind as TrainingWordKind,
    };
  });
  events.forEach((event, index) => {
    const previous = events[index - 1];
    if (previous && (previous.row > event.row || (previous.row === event.row && previous.column >= event.column))) {
      reader.fail(`events are not in (row, column) order at ${index}: one cell, one word`);
    }
  });
  // STEP 0 IS COMPLETE: the sentence opens with the six words already in force.
  const opening = events.filter((event) => event.row === 0).map((event) => event.column);
  if (opening.length !== TRAINING_COLUMNS.length) {
    const missing = TRAINING_COLUMNS.filter((_, column) => !opening.includes(column));
    reader.fail(`step 0 carries no word for ${missing.join(", ")} — step 0 gives all six`);
  }
  const inForce = reader.list("inForce");
  if (inForce.length !== TRAINING_COLUMNS.length) reader.fail(`inForce has ${inForce.length} columns, expected ${TRAINING_COLUMNS.length}`);
  const table = inForce.map((values, column) => {
    const parsed = numberArray(values);
    if (parsed === null || parsed.length !== rows) {
      reader.fail(`inForce.${TRAINING_COLUMNS[column]} is not ${rows} numbers, one per step`);
    }
    return parsed;
  });
  // The per-row table is the events filled forward — two spellings of one sentence.
  TRAINING_COLUMNS.forEach((name, column) => {
    let value = TRAINING_UNCHANGED;
    const issued = new Map(events.filter((event) => event.column === column).map((event) => [event.row, event.value]));
    for (let row = 0; row < rows; row += 1) {
      value = issued.get(row) ?? value;
      if (table[column][row] !== value) {
        reader.fail(`inForce.${name}[${row}] is ${table[column][row]}, but the events put ${value} in force there`);
      }
    }
  });
  return { events, inForce: table };
}

function eventsOf(events: TrainingWordEvent[], column: TrainingColumn): TrainingWordEvent[] {
  return events.filter((event) => event.column === TRAINING_COLUMN_INDEX[column]);
}

/** One envelope per word of its column, in the sentence's order: its row and its value. */
function matchWord(item: Reader, event: TrainingWordEvent | undefined, index: number, column: TrainingColumn): void {
  if (event === undefined) item.fail(`there is no ${column} word ${index} for it to belong to`);
  if (item.number("row") !== event.row || item.number("value") !== event.value || item.string("kind") !== event.kind) {
    item.fail(`is (row ${item.raw("row")}, value ${item.raw("value")}), but the sentence's ${column} word ${index} is ` +
              `(row ${event.row}, value ${event.value}, ${event.kind})`);
  }
}

function parseHeading(reader: Reader, events: TrainingWordEvent[], vocabulary: TrainingVocabulary, rows: number) {
  const words = eventsOf(events, "heading");
  const list = reader.list("heading");
  if (list.length !== words.length) reader.fail(`heading holds ${list.length} envelopes for ${words.length} heading words`);
  return list.map((raw, index): TrainingHeadingEnvelope => {
    const item = Reader.of(raw, reader.at(`heading[${index}]`));
    matchWord(item, words[index], index, "heading");
    const targetDeg = item.number("targetDeg");
    if (targetDeg !== vocabulary.headingTargetsDeg[words[index].value]) {
      item.fail(`targetDeg is ${targetDeg}, but word ${words[index].value} is ${vocabulary.headingTargetsDeg[words[index].value]}°`);
    }
    const word = words[index];
    const splitReader = item.nullableChild("split");
    const funnel = item.nullableChild("funnel");
    const check = item.nullableChild("check");
    const turn = item.nullableChild("turn");
    // WHAT COMES TOGETHER: a turn is a region, a chart band, an end row and the labeller's check of
    // it; a hold is a funnel, a band and a start row. Half of either is a half-written envelope.
    const turned = [turn, item.raw("turnBandDeg"), item.raw("turnEndRow"), check].map((part) => part !== null);
    if (new Set(turned).size !== 1) item.fail("turn, turnBandDeg, turnEndRow and check come together or not at all");
    const held = [funnel, item.raw("holdBandDeg"), item.raw("holdStartRow")].map((part) => part !== null);
    if (new Set(held).size !== 1) item.fail("funnel, holdBandDeg and holdStartRow come together or not at all");
    if ((word.kind === "initial") === turned[0]) item.fail(`a ${word.kind} word ${turned[0] ? "has" : "lacks"} a turn`);
    if ((splitReader !== null) !== word.kind.endsWith("-split")) item.fail(`split is given exactly for a split part, and this word is ${word.kind}`);
    const split = splitReader === null ? null : { part: splitReader.count("part", 1), parts: splitReader.count("parts", 2) };
    if (split !== null && split.part > split.parts) splitReader!.fail(`part ${split.part} of ${split.parts}`);
    const turnEndRow = item.nullableInteger("turnEndRow", word.row, rows);
    const holdEndRow = item.integer("holdEndRow", word.row, rows);
    const holdStartRow = item.nullableInteger("holdStartRow", word.row, holdEndRow - 1);
    if (turnEndRow !== null && turnEndRow > holdEndRow) item.fail(`the turn ends at row ${turnEndRow}, after the hold ends at ${holdEndRow}`);
    const verdict = check === null ? null : {
      ...parseTurnCheck(check),
      kind: check.string("kind"),
      departureRow: check.integer("departureRow", 0, word.row),
      arrivalRow: check.integer("arrivalRow", word.row, rows),
      turnDeg: check.number("turnDeg"),
      parts: check.count("parts", 1),
    };
    if (verdict !== null && verdict.kind !== word.kind.replace(/-split$/, "")) {
      check!.fail(`is the check of a ${verdict.kind}, but the word is ${word.kind}`);
    }
    // THE HOLD CHECK judges exactly the drawn funnel over the word's own hold rows, inclusive.
    const holdReader = item.nullableChild("holdCheck");
    if (holdReader !== null && funnel === null) item.fail("holdCheck is given for a word with no hold");
    const holdCheck = holdReader === null ? null : {
      holdStartRow: holdReader.integer("holdStartRow", holdStartRow!, holdStartRow!),
      holdEndRow: holdReader.integer("holdEndRow", holdEndRow, holdEndRow),
      rows: holdReader.integer("rows", holdEndRow - holdStartRow! + 1, holdEndRow - holdStartRow! + 1),
      inside: holdReader.integer("inside", 0, holdEndRow - holdStartRow! + 1),
      halfWidthEndM: holdReader.number("halfWidthEndM"),
    };
    return {
      row: word.row,
      value: word.value,
      kind: word.kind,
      targetDeg,
      split,
      turnEndRow,
      holdStartRow,
      holdEndRow,
      fromTrackDeg: item.number("fromTrackDeg"),
      targetOnTrackDeg: item.number("targetOnTrackDeg"),
      turnBandDeg: item.nullableRange("turnBandDeg"),
      holdBandDeg: item.nullableRange("holdBandDeg"),
      turn: turn === null ? null : parseTurnRegion(turn, vocabulary),
      funnel: funnel === null ? null : {
        lengthM: funnel.number("lengthM"),
        startHalfWidthM: funnel.number("startHalfWidthM"),
        endHalfWidthM: funnel.number("endHalfWidthM"),
        axis: parsePlanLine(funnel, "axis", 2),
        outline: parsePlanLine(funnel, "outline", 3),
      },
      check: verdict,
      holdCheck,
    };
  });
}

function parseApproach(
  reader: Reader, flight: { rows: number; captureRow: number; joinRow: number }, vocabulary: TrainingVocabulary,
): TrainingApproach {
  const approach = reader.child("approach");
  if (approach.number("clearanceRow") !== flight.joinRow) approach.fail(`clearanceRow is ${approach.raw("clearanceRow")}, the flight's joinRow ${flight.joinRow}`);
  if (approach.number("captureRow") !== flight.captureRow) approach.fail(`captureRow is ${approach.raw("captureRow")}, the flight's ${flight.captureRow}`);
  const capture = approach.nullableChild("captureTurn");
  const corridor = approach.child("corridor");
  const rows = corridor.integer("rows", 1, flight.rows);
  if (rows !== flight.rows - flight.captureRow) {
    corridor.fail(`rows is ${rows}, but the capture at row ${flight.captureRow} leaves ${flight.rows - flight.captureRow}`);
  }
  const landing = approach.child("landing");
  const crossing = landing.nullableChild("crossing");
  const cutAtCrossing = landing.boolean("cutAtCrossing");
  if (cutAtCrossing !== (crossing !== null)) landing.fail("the crossing row is given exactly when the sentence was cut at it");
  return {
    clearanceRow: flight.joinRow,
    captureRow: flight.captureRow,
    captureBeforeThresholdM: approach.number("captureBeforeThresholdM"),
    interceptInserted: approach.boolean("interceptInserted"),
    captureTurn: capture === null ? null : {
      startRow: capture.integer("startRow", 0, flight.captureRow),
      courseOnTrackDeg: capture.number("courseOnTrackDeg"),
      bandDeg: capture.range("bandDeg"),
      check: parseTurnCheck(capture.child("check")),
      turn: parseTurnRegion(capture.child("turn"), vocabulary),
    },
    courseBandDeg: approach.range("courseBandDeg"),
    corridor: {
      beforeThresholdM: corridor.number("beforeThresholdM"),
      halfWidthAtCaptureM: corridor.number("halfWidthAtCaptureM"),
      halfWidthAtThresholdM: corridor.number("halfWidthAtThresholdM"),
      rows,
      axis: parsePlanLine(corridor, "axis", 2),
      outline: parsePlanLine(corridor, "outline", 3),
    },
    landing: {
      cutAtCrossing,
      lastRowBeforeThresholdM: landing.number("lastRowBeforeThresholdM"),
      crossing: crossing === null ? null : {
        row: crossing.integer("row", flight.rows, flight.rows),
        eM: crossing.number("eM"), nM: crossing.number("nM"), lon: crossing.number("lon"), lat: crossing.number("lat"),
      },
    },
  };
}

function parseAltitude(reader: Reader, events: TrainingWordEvent[], vocabulary: TrainingVocabulary, rows: number) {
  const words = eventsOf(events, "altitude");
  const list = reader.list("altitude");
  if (list.length !== words.length) reader.fail(`altitude holds ${list.length} tubes for ${words.length} altitude words`);
  return list.map((raw, index): TrainingAltitudeTube => {
    const item = Reader.of(raw, reader.at(`altitude[${index}]`));
    matchWord(item, words[index], index, "altitude");
    const endRow = index + 1 < words.length ? words[index + 1].row : rows;
    if (item.number("endRow") !== endRow) item.fail(`endRow is ${item.raw("endRow")}, but the next altitude word opens at ${endRow}`);
    const value = words[index].value;
    const targetM = item.nullableNumber("targetM");
    const expected = value === vocabulary.altitudeLandValue ? null : vocabulary.altitudeTargetsM[value];
    if (targetM !== expected) item.fail(`targetM is ${targetM}, but word ${value} is ${expected === null ? "descend to land" : `${expected} m`}`);
    const length = endRow - words[index].row;
    const lowerM = item.numbers("lowerM", length);
    const upperM = item.numbers("upperM", length);
    lowerM.forEach((low, offset) => {
      if (low > upperM[offset]) item.fail(`is inverted at row ${words[index].row + offset}: ${low} above ${upperM[offset]}`);
    });
    const inside = item.flags("inside", length);
    const check = item.child("check");
    const tube = {
      rows: check.integer("rows", length, length),
      inside: check.integer("inside", 0, length),
      contained: check.boolean("contained"),
      tubeWidthEndM: check.number("tubeWidthEndM"),
    };
    const counted = inside.filter(Boolean).length;
    if (counted !== tube.inside) check.fail(`says ${tube.inside} rows inside, but the tube's own flags count ${counted}`);
    if (tube.contained !== (counted === length)) check.fail(`contained is ${tube.contained} with ${counted} of ${length} rows inside`);
    return {
      row: words[index].row, endRow, value, kind: words[index].kind, targetM, lowerM, upperM,
      lowerHaeM: item.numbers("lowerHaeM", length), upperHaeM: item.numbers("upperHaeM", length), inside, check: tube,
    };
  });
}

function parseAngle(reader: Reader, events: TrainingWordEvent[], vocabulary: TrainingVocabulary) {
  const words = eventsOf(events, "angle");
  const list = reader.list("angle");
  if (list.length !== words.length) reader.fail(`angle holds ${list.length} entries for ${words.length} angle words`);
  return list.map((raw, index): TrainingAngleWord => {
    const item = Reader.of(raw, reader.at(`angle[${index}]`));
    matchWord(item, words[index], index, "angle");
    const measuredDeg = item.nullableNumber("measuredDeg");
    if ((measuredDeg === null) !== (words[index].value === vocabulary.angleLevelValue)) {
      item.fail("a measured angle is given for every class but level, which is a target reached");
    }
    return { row: words[index].row, value: words[index].value, kind: words[index].kind, measuredDeg };
  });
}

function parseSpeed(reader: Reader, events: TrainingWordEvent[], vocabulary: TrainingVocabulary, rows: number) {
  const words = eventsOf(events, "speed");
  const list = reader.list("speed");
  if (list.length !== words.length) reader.fail(`speed holds ${list.length} spans for ${words.length} speed words`);
  return list.map((raw, index): TrainingSpeedSpan => {
    const item = Reader.of(raw, reader.at(`speed[${index}]`));
    matchWord(item, words[index], index, "speed");
    const row = words[index].row;
    const endRow = index + 1 < words.length ? words[index + 1].row : rows;
    if (item.number("endRow") !== endRow) item.fail(`endRow is ${item.raw("endRow")}, but the next speed word opens at ${endRow}`);
    const value = words[index].value;
    const targetMps = item.nullableNumber("targetMps");
    const unspecified = value === vocabulary.speedUnspecifiedValue;
    const expected = unspecified ? null : vocabulary.speedTargetsMps[value];
    if (targetMps !== expected) item.fail(`targetMps is ${targetMps}, but word ${value} is ${expected === null ? "unspecified" : `${expected} m/s`}`);
    if (unspecified) {
      for (const key of ["arrivalRow", "transitionLowerMps", "transitionUpperMps", "bandMps", "bandInside", "check"]) {
        if (item.raw(key) !== null) item.fail(`${key} is given for "unspecified", which has only the range`);
      }
      return {
        row, endRow, value, kind: words[index].kind, targetMps: null, arrivalRow: null, transitionLowerMps: null,
        transitionUpperMps: null, bandMps: null, bandInside: null, check: null, rangeMps: item.range("rangeMps"),
      };
    }
    if (item.raw("rangeMps") !== null) item.fail("rangeMps is given for a target, which has a band instead");
    const check = item.child("check");
    const cut = check.boolean("cutBeforeArrival");
    const arrivalRow = item.nullableInteger("arrivalRow", row, endRow - 1);
    if (cut !== (arrivalRow === null)) item.fail("arrivalRow is given exactly when the band is reached before the next word");
    const transitionRows = (arrivalRow === null ? endRow - 1 : arrivalRow) - row + 1;
    const bandRows = arrivalRow === null ? 0 : endRow - arrivalRow;
    const bandInside = arrivalRow === null ? null : item.flags("bandInside", bandRows);
    if (arrivalRow === null && item.raw("bandInside") !== null) item.fail("bandInside is given, but the band is never reached");
    const verdict = {
      arrivalRows: check.integer("arrivalRows", (arrivalRow ?? endRow) - row, (arrivalRow ?? endRow) - row),
      cutBeforeArrival: cut,
      transitionOk: check.boolean("transitionOk"),
      accelOk: check.boolean("accelOk"),
      bandRows: check.integer("bandRows", bandRows, bandRows),
      bandInside: check.integer("bandInside", 0, bandRows),
      contained: check.boolean("contained"),
    };
    const counted = bandInside === null ? 0 : bandInside.filter(Boolean).length;
    if (counted !== verdict.bandInside) check.fail(`says ${verdict.bandInside} band rows inside, but the band's own flags count ${counted}`);
    if (verdict.contained !== (verdict.transitionOk && counted === bandRows)) {
      check.fail(`contained is ${verdict.contained}, with the transition ${verdict.transitionOk ? "ok" : "failed"} and ${counted} of ${bandRows} band rows inside`);
    }
    return {
      row, endRow, value, kind: words[index].kind, targetMps, arrivalRow,
      transitionLowerMps: item.numbers("transitionLowerMps", transitionRows),
      transitionUpperMps: item.numbers("transitionUpperMps", transitionRows),
      bandMps: item.range("bandMps"), bandInside, check: verdict, rangeMps: null,
    };
  });
}

function parseFlight(
  raw: unknown, position: number, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[],
): TrainingFlight {
  const key = isRecord(raw) && str(raw, "flightKey") ? (raw.flightKey as string) : `flights[${position}]`;
  const flight = Reader.of(raw, `flight ${key}`);
  const rows = flight.count("rows", 2);
  const runwayIndex = flight.integer("runwayIndex", 0, candidates.length - 1);
  const runway = flight.string("runway");
  if (runway !== candidates[runwayIndex].ident) {
    flight.fail(`runway is ${runway}, but candidate ${runwayIndex} is ${candidates[runwayIndex].ident}`);
  }
  const stratum = flight.string("stratum");
  if (!(TRAINING_STRATA as readonly string[]).includes(stratum)) flight.fail(`stratum is ${stratum}, not one of ${TRAINING_STRATA.join(", ")}`);
  const counts = TRAINING_COLUMNS.map((column) => trainingClassCount(vocabulary, candidates, column));
  const words = parseWords(flight.child("words"), rows, counts);
  const pointer = words.events.find((event) => event.row === 0 && event.column === TRAINING_COLUMN_INDEX.runway)!;
  if (pointer.value !== runwayIndex) flight.fail(`step 0 points at runway ${pointer.value}, the flight lands on ${runwayIndex}`);
  const captureRow = flight.integer("captureRow", 0, rows - 1);
  const joinRow = flight.integer("joinRow", 0, rows - 1);
  const unspecifiedRow = flight.integer("unspecifiedRow", 0, rows - 1);
  // The markers the views draw are the words' own rows: the clearance, and the speed left to the pilot.
  const cleared = words.events.find((event) => event.kind === "clear");
  if (cleared === undefined || cleared.row !== joinRow) {
    flight.fail(`joinRow is ${joinRow}, but the clearance is issued at ${cleared === undefined ? "no row" : `row ${cleared.row}`}`);
  }
  const speeds = words.events.filter((event) => event.column === TRAINING_COLUMN_INDEX.speed);
  const unspecified = speeds[speeds.length - 1];
  if (unspecified.value !== vocabulary.speedUnspecifiedValue || unspecified.row !== unspecifiedRow) {
    flight.fail(`unspecifiedRow is ${unspecifiedRow}, but the last speed word is ${unspecified.value} at row ${unspecified.row}`);
  }
  const envelopes = flight.child("envelopes");
  return {
    datasetId: flight.string("datasetId"),
    flightKey: flight.string("flightKey"),
    callsign: flight.string("callsign"),
    typecode: flight.string("typecode"),
    runway,
    runwayIndex,
    stratum: stratum as TrainingStratum,
    rows,
    captureRow,
    joinRow,
    unspecifiedRow,
    captureBeforeThresholdM: flight.number("captureBeforeThresholdM"),
    signals: parseSignals(flight.child("signals"), rows, vocabulary.stepS),
    words,
    envelopes: {
      heading: parseHeading(envelopes, words.events, vocabulary, rows),
      approach: parseApproach(envelopes, { rows, captureRow, joinRow }, vocabulary),
      altitude: parseAltitude(envelopes, words.events, vocabulary, rows),
      angle: parseAngle(envelopes, words.events, vocabulary),
      speed: parseSpeed(envelopes, words.events, vocabulary, rows),
    },
  };
}

/** Parse one sample set — all or nothing: a flight the reader cannot trust would be drawn
 *  beside real ones with no way to tell. */
export function parseTrainingSample(raw: unknown): Parsed<TrainingSample> {
  if (!isRecord(raw)) return { ok: false, problem: "the sample is not an object" };
  if (raw.schema !== TRAINING_SAMPLE_SCHEMA) {
    return {
      ok: false,
      problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_SAMPLE_SCHEMA)} — a sample of another format is not read: re-export the set`,
    };
  }
  try {
    const sample = new Reader(raw, "sample");
    const vocabulary = parseVocabulary(sample.child("vocabulary"));
    const candidates = parseCandidates(sample, vocabulary);
    const cohort = sample.child("cohort");
    const frame = sample.child("airportFrame");
    const flights = sample.list("flights").map((item, position) => parseFlight(item, position, vocabulary, candidates));
    return {
      ok: true,
      value: {
        setId: sample.string("setId"),
        airport: sample.string("airport"),
        cohort: {
          split: cohort.string("split"), perStratum: cohort.number("perStratum"), seed: cohort.number("seed"),
          drawnFrom: cohort.string("drawnFrom"), pool: cohort.number("pool"), read: cohort.number("read"),
        },
        vocabulary,
        airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
        candidatesSha256: sample.string("candidatesSha256"),
        centrelineLengthM: sample.number("centrelineLengthM"),
        candidates,
        flights,
      },
    };
  } catch (error) {
    if (error instanceof Refusal) return { ok: false, problem: error.message };
    throw error;
  }
}

// ── where the files live ─────────────────────────────────────────────────────

/** The airport's Training directory, relative to the site root. */
export function trainingDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

/** The manifest the panel reads — shown in the empty state and used by the fetch below. */
export function trainingIndexPath(airportCode: string): string {
  return `${trainingDirectory(airportCode)}/index.json`;
}

/** A set's sample file. `file` is relative to the airport's Training directory. */
export function trainingSamplePath(airportCode: string, file: string): string {
  return `${trainingDirectory(airportCode)}/${file}`;
}

export async function fetchTrainingIndex(airportCode: string): Promise<Parsed<TrainingIndex>> {
  return parseTrainingIndex(await fetchJson<unknown>(trainingIndexPath(airportCode)));
}

export async function fetchTrainingSample(airportCode: string, file: string): Promise<Parsed<TrainingSample>> {
  return parseTrainingSample(await fetchJson<unknown>(trainingSamplePath(airportCode, file)));
}
