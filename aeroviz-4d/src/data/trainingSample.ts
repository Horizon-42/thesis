/**
 * trainingSample.ts
 * -----------------
 * The Training module's data contract: the manifest of exported sets, and one set's flights
 * under the instruction vocabulary (`instruction-v3`: the heading read step by step, vocabulary design §10.1; spec
 * `145d6911e75b`, measured on the day split's training days, `4dTrajectory/outputs/POOLED/instruction_language/v4_20260924/`).
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`; the words:
 * `4dTrajectory/ts_transformer/docs/2026-09-23_instruction_vocabulary_design.zh.md`.
 *
 * A SENTENCE IS SIX COLUMNS PER 2 s STEP — runway pointer, approach, heading, altitude, angle,
 * speed, in that order — and every column has "unchanged" (-1). Step 0 carries all six; after it
 * the sentence is mostly silent. A WORD IS A TARGET PLUS THE ENVELOPE IT ALLOWS: a heading word's
 * band — its target ± the heading tolerance over the rows it is judged on, from a lead after it is
 * said to the next heading word's — the capture turn and corridor, an altitude word's tube, a speed
 * word's transition and band.
 *
 * THIS FILE COMPUTES NO ENVELOPE. Every band, row verdict, region and tube is what the exporter
 * computed in Python from the vocabulary's own functions (`instructions/display.py` over
 * `envelope.py` and `labeller/*`), and every verdict is the labeller's. The reader checks the file's
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
import { attempt, parseManifest, Reader, type Parsed } from "./trainingReader";

/** MIRROR of the exporter's `INDEX_SCHEMA`. The index keeps this shape across vocabularies:
 *  every set, current or superseded, is listed in it. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v1";
/** MIRROR of the exporter's `SAMPLE_SCHEMA` (`instruction_training_export.py`). The name changes
 *  with the file's shape, on both sides, in the same change: v7 is `instruction-v3`'s sample — a
 *  heading word is its band over its judged rows with a verdict per row, the capture turn its rows
 *  and its check, and there are no turn regions, hold funnels, split parts or inserted intercepts;
 *  a file under any other name — `v6` (instruction-v2's turns and holds) and every earlier one
 *  included, whatever vocabulary it carries — is refused. */
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v7";
/** MIRROR of `instructions.spec.READING_RULE`: what a word MEANS, which no field can say. */
export const TRAINING_READING_RULE = "instruction-v3";
/** MIRROR of the spec's sha: a new vocabulary is a new sha, and this reader is bound to the one it was written for
 *  (`v4_20260924/spec.json`, instruction-v3 re-measured on the operating-day split's training days, 2026-09-25 — the
 *  artefact today's executor code flies; the flight-split `v3_20260924` spec `0b4ea75be36d` is refused by name). */
export const TRAINING_SPEC_SHA256 = "145d6911e75b02f61697cda79f7a5b9fe7de948e55c12199c5a1df081a2d5bd5";
/** MIRROR of the exporter's `KIND_READBACK`: the one kind of set this reader opens. */
export const TRAINING_READABLE_SET_KIND = "vocabulary-readback";
/** The set kinds the manifest may list. A `prior-generated` set (the segment prior's generated sentences) has no
 *  contract under this vocabulary — the instruction prior is drawn as an overlay over a read-back set
 *  (`trainingOverlays.ts`) — so it is listed and refused by name. */
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
  "initial", "per-step", "clear", "target", "step", "angle", "unspecified",
] as const;
export type TrainingWordKind = (typeof TRAINING_WORD_KINDS)[number];

export const TRAINING_COLUMN_INDEX = Object.fromEntries(
  TRAINING_COLUMNS.map((column, index) => [column, index]),
) as Record<TrainingColumn, number>;

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
  /** A heading word says where the track is this long after it is said, and is judged from then. */
  headingLeadS: number;
  /** The same lead in steps, as the exporter wrote it (`spec.rows_exact`); the reader checks the two agree. */
  headingLeadRows: number;
  /** A heading word's band: the track within this of its target on every row it is judged on. */
  headingToleranceDeg: number;
  /** The capture turn begins where the track turns toward the course faster than this. */
  turnOnsetRateDegS: number;
  /** The capture turn's rate: at most `turnRateMaxDegS` on every row (and at most `turnBankMaxDeg`
   *  of bank at the flown speed); at least `turnRateMinDegS` on average for a turn of
   *  `turnRateMinFromDeg` or more. */
  turnRateMinDegS: number;
  turnRateMaxDegS: number;
  turnRateMinFromDeg: number;
  turnBankMaxDeg: number;
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

/** The labeller's check of the capture turn: monotone progress toward the course, its rate inside the range. */
export interface TrainingTurnCheck {
  progressOk: boolean;
  rateOk: boolean;
  meanRateDegS: number;
  maxRateDegS: number;
  maxBankDeg: number;
  /** The lowest rate is judged only for turns of at least `turnRateMinFromDeg`. */
  rateMinApplies: boolean;
}

/**
 * A heading word's envelope (vocabulary design §10.1): its target ± the heading tolerance over the rows it is judged
 * on — from its row plus the lead to the next heading word's row plus the lead, never at or past the clearance — and
 * each row's verdict. None when the lead carries its rows to the clearance (`firstRow === stopRow`). The executor's
 * overlay writes its words' bands in this shape too, on its own flown rows.
 */
export interface TrainingHeadingBand {
  firstRow: number;
  /** One past the last row judged. */
  stopRow: number;
  /** The target on the heading chart's branch. */
  targetOnTrackDeg: number;
  bandDeg: [number, number];
  /** One per judged row: the track within the band. */
  inside: boolean[];
}

export interface TrainingHeadingEnvelope extends TrainingHeadingBand {
  row: number;
  value: number;
  kind: TrainingWordKind;
  targetDeg: number;
  /** The labeller's check: its judged rows and how many were inside. */
  check: { rows: number; inside: number };
}

export interface TrainingApproach {
  clearanceRow: number;
  captureRow: number;
  captureBeforeThresholdM: number;
  /** From the clearance to the capture, onto the course; null for a flight on the final at step 0. */
  captureTurn: {
    startRow: number;
    endRow: number;
    /** The course on the heading chart's branch at the start. */
    courseOnTrackDeg: number;
    check: TrainingTurnCheck;
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
  /** The flight's own ICAO type; null = its identity is unresolved (sample v6). */
  typecode: string | null;
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
  /** When the exporter wrote it: an overlay drawn over the set names the sample it was drawn over by it. */
  writtenUtc: string;
  cohort: TrainingCohort & { pool: number; read: number };
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidatesSha256: string;
  centrelineLengthM: number;
  candidates: TrainingCandidate[];
  flights: TrainingFlight[];
}

/** What the panel publishes for the sentence bar, the read-back window and the 3D layer: the flight on screen, with the
 *  set and airport it is read from — a flight key alone repeats across the sets of one airport. */
export interface TrainingSelection {
  airport: string;
  setId: string;
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  flight: TrainingFlight;
}

/** The flight on screen as one identity — the cursor and the live executor's pick belong to it and reset with it. */
export function trainingSelectionKey(selection: TrainingSelection | null): string | null {
  return selection === null ? null : `${selection.airport}/${selection.setId}/${selection.flight.flightKey}`;
}

/** The selection of one flight of a sample. */
export function trainingSelectionOf(sample: TrainingSample, flight: TrainingFlight): TrainingSelection {
  return { airport: sample.airport, setId: sample.setId, vocabulary: sample.vocabulary, candidates: sample.candidates, flight };
}

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
  "per-step": "the heading the track reaches a lead later",
  clear: "cleared to join the final",
  target: "a new target",
  step: "a step between two holds",
  angle: "a new descent angle",
  unspecified: "speed left to the pilot",
};

export function trainingKindLabel(kind: TrainingWordKind): string {
  return KIND_LABEL[kind];
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
  const heading = flight.envelopes.heading;
  const judged = heading.filter((item) => item.check.rows > 0);
  const tubes = flight.envelopes.altitude;
  const speeds = flight.envelopes.speed.flatMap((span) => (span.check === null ? [] : [span.check]));
  const issued = new Set(flight.words.events.filter((event) => event.row > 0).map((event) => event.row));
  return {
    headingJudged: judged.length,
    headingContained: judged.filter((item) => item.check.inside === item.check.rows).length,
    /** Heading words the lead carries to the clearance: no row of their own to judge. */
    headingNotJudged: heading.length - judged.length,
    captureTurn: flight.envelopes.approach.captureTurn?.check ?? null,
    altitudeWords: tubes.length,
    altitudeContained: tubes.filter((tube) => tube.check.contained).length,
    speedWords: speeds.length,
    speedContained: speeds.filter((check) => check.contained).length,
    instructionsAfterStep0: flight.words.events.filter((event) => event.row > 0).length,
    silentSteps: flight.rows - 1 - issued.size,
  };
}

function parsePlanLine(reader: Reader, key: string, minimum: number): TrainingPlanLine {
  const line = reader.child(key);
  const eM = line.numbers("eM");
  if (eM.length < minimum) line.fail(`holds ${eM.length} points, fewer than ${minimum}`);
  // ONE outline in two coordinate systems: a length that differs is two shapes drawn as one.
  return { eM, nM: line.numbers("nM", eM.length), lon: line.numbers("lon", eM.length), lat: line.numbers("lat", eM.length) };
}

// ── the index ────────────────────────────────────────────────────────────────

function parseCohort(cohort: Reader): TrainingCohort {
  return {
    split: cohort.string("split"), perStratum: cohort.count("perStratum", 1), seed: cohort.number("seed"),
    drawnFrom: cohort.string("drawnFrom"),
  };
}

function parseSetEntry(entry: Reader): TrainingSetEntry {
  return {
    id: entry.string("id"),
    kind: entry.oneOf("kind", TRAINING_SET_KINDS),
    title: entry.string("title"),
    file: entry.string("file"),
    vocabularySha256: entry.string("vocabularySha256"),
    runwaySha256: entry.string("runwaySha256"),
    readingRule: entry.string("readingRule"),
    flights: entry.count("flights"),
    cohort: parseCohort(entry.child("cohort")),
  };
}

/** Parse the manifest. A bad entry is rejected on its own; only a manifest that is not a
 *  manifest at all fails the whole call. */
export function parseTrainingIndex(raw: unknown): Parsed<TrainingIndex> {
  const manifest = parseManifest(raw, { name: "manifest", schema: TRAINING_INDEX_SCHEMA, listKey: "sets", entryName: "set" },
    parseSetEntry);
  if (!manifest.ok) return manifest;
  return { ok: true, value: { airport: manifest.value.airport, sets: manifest.value.entries, rejected: manifest.value.rejected } };
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
    return `a ${entry.kind} set: this reader opens only ${TRAINING_READABLE_SET_KIND} sets (the prior is drawn over one, ` +
      "as an overlay)";
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
  reader.sameNames("columns", TRAINING_COLUMNS);
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
  const angleClasses = reader.children("angleClasses").map((angle, index) => {
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
  const stepS = reader.number("stepS");
  // The lead in seconds and in steps are one number: a word's first judged row is its own row plus the steps.
  const headingLeadS = reader.number("headingLeadS");
  const headingLeadRows = reader.count("headingLeadRows", 0);
  if (Math.abs(headingLeadRows * stepS - headingLeadS) > 1e-9) {
    reader.fail(`headingLeadRows is ${headingLeadRows}, but ${headingLeadRows} steps of ${stepS} s are not headingLeadS ${headingLeadS} s`);
  }
  return {
    readingRule: rule,
    specSha256: sha,
    labellerSourceSha256: reader.string("labellerSourceSha256"),
    stepS,
    smoothingS: { track: smoothing.number("track"), altitude: smoothing.number("altitude"), speed: smoothing.number("speed") },
    classCounts,
    approachClasses,
    headingTargetsDeg,
    headingLeadS,
    headingLeadRows,
    headingToleranceDeg: reader.number("headingToleranceDeg"),
    turnOnsetRateDegS: reader.number("turnOnsetRateDegS"),
    turnRateMinDegS: reader.number("turnRateMinDegS"),
    turnRateMaxDegS: reader.number("turnRateMaxDegS"),
    turnRateMinFromDeg: reader.number("turnRateMinFromDeg"),
    turnBankMaxDeg: reader.number("turnBankMaxDeg"),
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
  const candidates = reader.children("candidates").map((candidate, index) => {
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
  const events = reader.children("events").map((event) => {
    const column = event.integer("column", 0, TRAINING_COLUMNS.length - 1);
    return {
      row: event.integer("row", 0, rows - 1),
      column,
      value: event.integer("value", 0, counts[column] - 1),
      kind: event.oneOf("kind", TRAINING_WORD_KINDS),
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
    if (!Array.isArray(values) || values.length !== rows || !values.every((value) => Number.isInteger(value))) {
      reader.fail(`inForce.${TRAINING_COLUMNS[column]} is not ${rows} whole numbers, one per step`);
    }
    return values as number[];
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

/**
 * Where a heading band's rows end: EXACTLY at ``stop`` — the sample's own words, whose stop the reader knows (the next
 * heading word's row plus the lead, never past the clearance: `envelope.heading_word_rows`) — or BY it: a band the
 * executor's judge drew on flown rows, which end where its own clearance, capture or track do.
 */
export type TrainingBandStop = { exactly: number } | { by: number };

/**
 * What is wrong with a heading word's band by the bookkeeping alone, or null: its first row is the word's own plus the
 * lead, its rows run forward and end at (or by) its stop — or it has none when the lead carries it there — one
 * verdict per row, and the band is the word's target ± the vocabulary's tolerance on some branch.
 */
function headingBandProblem(
  band: TrainingHeadingBand, wordRow: number, targetDeg: number, vocabulary: TrainingVocabulary, stop: TrainingBandStop,
): string | null {
  const first = wordRow + vocabulary.headingLeadRows;
  if (band.firstRow !== first) {
    return `firstRow is ${band.firstRow}, but a word told at step ${wordRow} is judged from ${first} (the ` +
      `${vocabulary.headingLeadS} s lead)`;
  }
  if ("exactly" in stop) {
    const expected = Math.max(first, stop.exactly);
    if (band.stopRow !== expected) return `stopRow is ${band.stopRow}, but this word's rows end at ${expected}`;
  } else {
    const stopMax = Math.max(first, stop.by);
    if (!(band.firstRow <= band.stopRow && band.stopRow <= stopMax)) return `stopRow is ${band.stopRow}, not in ${band.firstRow}…${stopMax}`;
  }
  if (band.inside.length !== band.stopRow - band.firstRow) {
    return `inside holds ${band.inside.length} verdicts for the ${band.stopRow - band.firstRow} rows judged`;
  }
  const turns = (band.targetOnTrackDeg - targetDeg) / 360;
  if (Math.abs(turns - Math.round(turns)) > 1e-5) {
    return `targetOnTrackDeg is ${band.targetOnTrackDeg}, not the word's ${targetDeg}° on any branch`;
  }
  const tolerance = vocabulary.headingToleranceDeg;
  if (Math.abs(band.bandDeg[0] - (band.targetOnTrackDeg - tolerance)) > 2e-3
      || Math.abs(band.bandDeg[1] - (band.targetOnTrackDeg + tolerance)) > 2e-3) {
    return `bandDeg is [${band.bandDeg.join(", ")}], not ${band.targetOnTrackDeg}° ± the vocabulary's ${tolerance}°`;
  }
  return null;
}

/**
 * The rows judged outside, as inclusive [first, last] row spans to DRAW on a line of rows `0..lastRow`: ``inside``
 * holds one verdict per row from ``firstRow``, and each run of rows outside is carried on to the next row, so that one
 * row outside is a segment (back to the row before it at the line's end) — never a single point, which draws nothing.
 * One rule for every verdict drawn red — a heading band's rows, a tube's, a speed band's — in the read-back window and
 * the 3D scene alike; the verdicts are the file's own.
 */
export function outsideSpans(inside: boolean[], firstRow: number, lastRow: number): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  let start: number | null = null;
  inside.forEach((ok, offset) => {
    if (!ok && start === null) start = offset;
    const closes = start !== null && (ok || offset === inside.length - 1);
    if (!closes) return;
    const first = firstRow + start!;
    const last = Math.min(firstRow + (ok ? offset - 1 : offset) + 1, lastRow);
    spans.push(last > first ? [first, last] : [Math.max(first - 1, 0), first]);
    start = null;
  });
  return spans.filter(([first, last]) => last > first);
}

/** A heading word's band as a file writes it, checked by `headingBandProblem`. */
export function readHeadingBand(
  item: Reader, wordRow: number, targetDeg: number, vocabulary: TrainingVocabulary, stop: TrainingBandStop,
): TrainingHeadingBand {
  const firstRow = item.integer("firstRow", 0, Number.MAX_SAFE_INTEGER);
  const stopRow = item.integer("stopRow", firstRow, Number.MAX_SAFE_INTEGER);
  const band = {
    firstRow, stopRow, targetOnTrackDeg: item.number("targetOnTrackDeg"), bandDeg: item.range("bandDeg"),
    inside: item.flags("inside", stopRow - firstRow),
  };
  const problem = headingBandProblem(band, wordRow, targetDeg, vocabulary, stop);
  if (problem !== null) item.fail(problem);
  return band;
}

function parseHeading(reader: Reader, events: TrainingWordEvent[], vocabulary: TrainingVocabulary, joinRow: number) {
  const words = eventsOf(events, "heading");
  const list = reader.children("heading");
  if (list.length !== words.length) reader.fail(`heading holds ${list.length} envelopes for ${words.length} heading words`);
  return list.map((item, index): TrainingHeadingEnvelope => {
    matchWord(item, words[index], index, "heading");
    const word = words[index];
    const targetDeg = item.number("targetDeg");
    if (targetDeg !== vocabulary.headingTargetsDeg[word.value]) {
      item.fail(`targetDeg is ${targetDeg}, but word ${word.value} is ${vocabulary.headingTargetsDeg[word.value]}°`);
    }
    // A word's rows end at the next word's first row, and never past the clearance (`envelope.heading_word_rows`).
    const next = words[index + 1];
    const stop = next === undefined ? joinRow : Math.min(next.row + vocabulary.headingLeadRows, joinRow);
    const band = readHeadingBand(item, word.row, targetDeg, vocabulary, { exactly: stop });
    // THE CHECK counts exactly the drawn rows and their verdicts.
    const check = item.child("check");
    const rows = check.integer("rows", band.stopRow - band.firstRow, band.stopRow - band.firstRow);
    const counted = band.inside.filter(Boolean).length;
    const insideCount = check.integer("inside", 0, rows);
    if (insideCount !== counted) check.fail(`says ${insideCount} rows inside, but the band's own flags count ${counted}`);
    return { row: word.row, value: word.value, kind: word.kind, targetDeg, ...band, check: { rows, inside: insideCount } };
  });
}

function parseApproach(reader: Reader, flight: { rows: number; captureRow: number; joinRow: number }): TrainingApproach {
  const approach = reader.child("approach");
  if (approach.number("clearanceRow") !== flight.joinRow) approach.fail(`clearanceRow is ${approach.raw("clearanceRow")}, the flight's joinRow ${flight.joinRow}`);
  if (approach.number("captureRow") !== flight.captureRow) approach.fail(`captureRow is ${approach.raw("captureRow")}, the flight's ${flight.captureRow}`);
  const capture = approach.nullableChild("captureTurn");
  const corridor = approach.child("corridor");
  const rows = corridor.integer("rows", 1, flight.rows);
  if (rows !== flight.rows - flight.captureRow) {
    corridor.fail(`rows is ${rows}, but the capture at row ${flight.captureRow} leaves ${flight.rows - flight.captureRow}`);
  }
  // The capture turn runs from the clearance to the capture; a flight on the final at step 0 has none.
  if ((capture === null) !== (flight.captureRow === 0)) {
    approach.fail(`captureTurn is ${capture === null ? "absent" : "given"} with the capture at step ${flight.captureRow}`);
  }
  const captureTurn = (turn: Reader) => ({
    startRow: turn.integer("startRow", flight.joinRow, flight.joinRow),
    endRow: turn.integer("endRow", flight.captureRow, flight.captureRow),
    courseOnTrackDeg: turn.number("courseOnTrackDeg"),
    check: parseTurnCheck(turn.child("check")),
  });
  const landing = approach.child("landing");
  const crossing = landing.nullableChild("crossing");
  const cutAtCrossing = landing.boolean("cutAtCrossing");
  if (cutAtCrossing !== (crossing !== null)) landing.fail("the crossing row is given exactly when the sentence was cut at it");
  return {
    clearanceRow: flight.joinRow,
    captureRow: flight.captureRow,
    captureBeforeThresholdM: approach.number("captureBeforeThresholdM"),
    captureTurn: capture === null ? null : captureTurn(capture),
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
  const list = reader.children("altitude");
  if (list.length !== words.length) reader.fail(`altitude holds ${list.length} tubes for ${words.length} altitude words`);
  return list.map((item, index): TrainingAltitudeTube => {
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
  const list = reader.children("angle");
  if (list.length !== words.length) reader.fail(`angle holds ${list.length} entries for ${words.length} angle words`);
  return list.map((item, index): TrainingAngleWord => {
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
  const list = reader.children("speed");
  if (list.length !== words.length) reader.fail(`speed holds ${list.length} spans for ${words.length} speed words`);
  return list.map((item, index): TrainingSpeedSpan => {
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
      const rangeMps = item.range("rangeMps");
      if (rangeMps.some((bound, end) => Math.abs(bound - vocabulary.speedRangeMps[end]) > 1e-3)) {
        item.fail(`rangeMps is [${rangeMps.join(", ")}], not the vocabulary's speed range [${vocabulary.speedRangeMps.join(", ")}]`);
      }
      return {
        row, endRow, value, kind: words[index].kind, targetMps: null, arrivalRow: null, transitionLowerMps: null,
        transitionUpperMps: null, bandMps: null, bandInside: null, check: null, rangeMps,
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

function parseFlight(flight: Reader, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[]): TrainingFlight {
  const rows = flight.count("rows", 2);
  const runwayIndex = flight.integer("runwayIndex", 0, candidates.length - 1);
  const runway = flight.string("runway");
  if (runway !== candidates[runwayIndex].ident) {
    flight.fail(`runway is ${runway}, but candidate ${runwayIndex} is ${candidates[runwayIndex].ident}`);
  }
  const stratum = flight.oneOf("stratum", TRAINING_STRATA);
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
  const signals = parseSignals(flight.child("signals"), rows, vocabulary.stepS);
  return {
    datasetId: flight.string("datasetId"),
    flightKey: flight.string("flightKey"),
    callsign: flight.string("callsign"),
    typecode: flight.nullableString("typecode"),
    runway,
    runwayIndex,
    stratum,
    rows,
    captureRow,
    joinRow,
    unspecifiedRow,
    captureBeforeThresholdM: flight.number("captureBeforeThresholdM"),
    signals,
    words,
    envelopes: {
      heading: parseHeading(envelopes, words.events, vocabulary, joinRow),
      approach: parseApproach(envelopes, { rows, captureRow, joinRow }),
      altitude: parseAltitude(envelopes, words.events, vocabulary, rows),
      angle: parseAngle(envelopes, words.events, vocabulary),
      speed: parseSpeed(envelopes, words.events, vocabulary, rows),
    },
  };
}

/** Parse one sample set — all or nothing: a flight the reader cannot trust would be drawn
 *  beside real ones with no way to tell. */
export function parseTrainingSample(raw: unknown): Parsed<TrainingSample> {
  return attempt(() => {
    const sample = Reader.of(raw, "sample");
    if (sample.raw("schema") !== TRAINING_SAMPLE_SCHEMA) {
      sample.fail(`schema is ${JSON.stringify(sample.raw("schema"))}, expected ${JSON.stringify(TRAINING_SAMPLE_SCHEMA)} — ` +
        "a sample of another format is not read: re-export the set");
    }
    const vocabulary = parseVocabulary(sample.child("vocabulary"));
    const candidates = parseCandidates(sample, vocabulary);
    const airport = sample.string("airport");
    const cohort = sample.child("cohort");
    const frame = sample.child("airportFrame");
    if (frame.string("code") !== airport) frame.fail(`code is ${frame.raw("code")}, but the sample is ${airport}'s`);
    const flights = sample.list("flights").map((item, position) => {
      const key = Reader.of(item, `sample.flights[${position}]`).string("flightKey");
      return parseFlight(Reader.of(item, `flight ${key}`), vocabulary, candidates);
    });
    // a flight is its key everywhere — the picker, the overlays, the live executor's request
    const seen = new Set<string>();
    for (const flight of flights) {
      if (seen.has(flight.flightKey)) sample.fail(`flight ${flight.flightKey} is listed twice`);
      seen.add(flight.flightKey);
    }
    return {
      setId: sample.string("setId"),
      airport,
      writtenUtc: sample.string("writtenUtc"),
      cohort: { ...parseCohort(cohort), pool: cohort.count("pool"), read: cohort.count("read") },
      vocabulary,
      airportFrame: { code: airport, lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidatesSha256: sample.string("candidatesSha256"),
      centrelineLengthM: sample.number("centrelineLengthM"),
      candidates,
      flights,
    };
  });
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

/** A file the manifests list — a set's sample, an overlay — relative to the airport's Training directory. */
export function trainingFilePath(airportCode: string, file: string): string {
  return `${trainingDirectory(airportCode)}/${file}`;
}

export async function fetchTrainingIndex(airportCode: string): Promise<Parsed<TrainingIndex>> {
  return parseTrainingIndex(await fetchJson<unknown>(trainingIndexPath(airportCode)));
}

export async function fetchTrainingSample(airportCode: string, file: string): Promise<Parsed<TrainingSample>> {
  return parseTrainingSample(await fetchJson<unknown>(trainingFilePath(airportCode, file)));
}
