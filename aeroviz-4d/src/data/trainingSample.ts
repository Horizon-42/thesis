/**
 * trainingSample.ts
 * -----------------
 * The Training view's data contract for STAGE A of the two-tier vocabulary (`instruction-v6`): the airport's index of
 * stage-A sets and one set's flights. Written by `ts_transformer/experiments/training_export.py` (the files:
 * `ts_transformer/instructions/training_files.py`); design: vocabulary §12.1 A23, outline §6.
 *
 * A SENTENCE IS FIVE COLUMNS — runway (with go-around), heading (relative to the course of the runway in force), altitude
 * (a level above the airport elevation E, or "no level-off"), angle, speed — and every column has "unchanged" (-1). A
 * flight carries two kinds of sentence:
 *
 *  • the OPEN-LOOP sentence: the labeller's reading of the observed track, on the observed 2 s rows from row 0, with the
 *    envelopes of its words on the observed track;
 *  • the CLOSED-LOOP sentence at each row interval Δ (`vocabulary.rowIntervalsS`): the words the closed-loop reading
 *    says to the executor from the first predicted step on, one Δ row apart, a word the reading ADDED marked
 *    (`correction`), with the states the executor flew — observed before the first predicted step, flown from it — and
 *    the judge's outcome, the threshold crossing and the decision-altitude (DA) check.
 *
 * ONE CLOCK: flight time, seconds from the observed track's row 0. The closed-loop states are on the 2 s rows from the
 * sentence's `firstRow` (state row k is observed row `firstRow + k`); the first predicted step is state row `flownFromRow`;
 * its Δ rows (the words) and the judge's envelopes count from it.
 *
 * THIS FILE DECODES NO WORD AND COMPUTES NO ENVELOPE: every word carries what it says (`says`, decoded by the
 * vocabulary's `Words` in Python) and every band, tube, span and verdict is the exporter's. The reader checks the file's
 * BOOKKEEPING — lengths, rows, that the events are the grid's words — and draws numbers.
 *
 * NO COMPATIBILITY. The index and sample schemas, the set kind and the reading rule are pinned below and a file that
 * carries anything else is refused by name (the schema found and the one expected). This reader never reads
 * `training/index.json`: its index is `index_v4.json`.
 *
 * SI units only: metres, m/s, degrees, seconds. Heights in the files are MSL; the 3D scene's are ellipsoid heights (the
 * flight's own `haeMinusMslM` added once, here).
 */

import { fetchJson } from "../utils/fetchJson";
import { attempt, parseManifest, Reader, type Parsed } from "./trainingReader";
import { readAttitude, type TrainingAttitude } from "./trainingAttitude";

/** MIRROR of the exporter's `INDEX_SCHEMA` (`ts_transformer/instructions/training_files.py`): the airport's index of
 *  stage-A sets. A name changes with its file's shape, on both sides, in the same change. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v2";
/** MIRROR of `INDEX_FILE`: a NEW index beside the old view's `index.json`, which this view never reads. */
export const TRAINING_INDEX_FILE = "index_v4.json";
/** MIRROR of `SAMPLE_SCHEMA`: a set's sample. */
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v9";
/** MIRROR of `SET_KIND`: a stage-A set — flights read back through the closed loop. */
export const TRAINING_SET_KIND = "closed-loop-readback";
/** MIRROR of `instructions.spec.READING_RULE`: what a word MEANS, which no field can say. */
export const TRAINING_READING_RULE = "instruction-v6";

/** MIRROR of `instructions.words.COLUMNS`. The columns are POSITIONAL: this order is that of every word grid and of the
 *  five rows the sentence bar draws. */
export const TRAINING_COLUMNS = ["runway", "heading", "altitude", "angle", "speed"] as const;
export type TrainingColumn = (typeof TRAINING_COLUMNS)[number];
export const TRAINING_COLUMN_INDEX = Object.fromEntries(
  TRAINING_COLUMNS.map((column, index) => [column, index]),
) as Record<TrainingColumn, number>;
/** MIRROR of `instructions.words.UNCHANGED`. */
export const TRAINING_UNCHANGED = -1;
/** MIRROR of `instructions.words.RUNWAY_GO_AROUND`: the runway column's word that sends the flight around. */
export const TRAINING_RUNWAY_GO_AROUND = -2;
/** MIRROR of `training_files.SPLITS`. */
export const TRAINING_SPLITS = ["train", "select"] as const;
/** MIRROR of `training_export.STRATA`. */
export const TRAINING_STRATA = ["straight-in", "vectored"] as const;
export type TrainingStratum = (typeof TRAINING_STRATA)[number];
/** MIRROR of `autopilot.judge.OUTCOMES`: how a flown flight ended. */
export const TRAINING_OUTCOMES = [
  "landed", "unstable_at_minimums", "crossed_too_high", "crossed_off_runway", "crossed_other_runway", "ground_contact",
  "timeout", "dynamics_failure",
] as const;
export type TrainingOutcome = (typeof TRAINING_OUTCOMES)[number];

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingCohort {
  splits: Record<string, number>;
  perStratum: number;
  strata: string[];
  seed: number;
  drawnFrom: string;
}

export interface TrainingSource {
  instructions: string;
  executor: string;
  specSha256: string;
  executorSpecSha256: string;
  git: { head: string; dirty: boolean };
}

/** A set as the index lists it. */
export interface TrainingSetEntry {
  id: string;
  title: string;
  /** The set's sample, relative to the airport's `training/` directory. */
  file: string;
  flights: number;
  formats: Record<string, string>;
  cohort: TrainingCohort;
  source: TrainingSource;
}

export interface TrainingIndex {
  airport: string;
  sets: TrainingSetEntry[];
  /** Entries that are not set entries at all, each with the field that failed. */
  rejected: Array<{ id: string; problem: string }>;
}

export interface TrainingAngleClass {
  index: number;
  name: string;
  nominalDeg: number;
  lowDeg: number;
  highDeg: number;
}

/** What the words mean, as numbers for the details page and the tolerances the charts name. */
export interface TrainingVocabulary {
  readingRule: string;
  /** The row step of the 2 s grid every track and envelope is on. */
  stepS: number;
  headingStepDeg: number;
  /** A heading word is judged from this long after it is said. */
  headingLeadS: number;
  headingToleranceDeg: number;
  altitudeLevelsM: number[];
  altitudeTolerancesM: number[];
  /** The altitude word that means "no level-off". */
  noLevelOff: number;
  angleClasses: TrainingAngleClass[];
  speed: { minMps: number; stepMps: number; levels: number; unspecified: number; toleranceMps: number };
  /** The closed-loop reading adds a word when the flown track is this far off the observed one. */
  closedLoopLateralM: number;
  closedLoopVerticalM: number;
  rowIntervalsS: number[];
}

export interface TrainingCandidate {
  index: number;
  ident: string;
  thresholdEM: number;
  thresholdNM: number;
  latDeg: number;
  lonDeg: number;
  courseDeg: number;
  elevationM: number;
  lengthM: number;
  haeMinusMslM: number;
  verticalPath: { crossingHeightM: number; glidepathDeg: number; decisionHeightM: number };
}

/** What a word says, as the exporter decoded it with the vocabulary (`training_export.said`). */
export type TrainingSays =
  | { column: "runway"; goAround: false; runway: string; runwayIndex: number }
  | { column: "runway"; goAround: true }
  | { column: "heading"; relativeDeg: number; trackDeg: number }
  | { column: "altitude"; noLevelOff: true }
  | { column: "altitude"; noLevelOff: false; levelM: number; mslM: number }
  | { column: "angle"; angleDeg: number; climb: boolean; level: boolean }
  /** null: speed left to the pilot. */
  | { column: "speed"; speedMps: number | null };

/** One word of a sentence: the row it is said at, its column (`TRAINING_COLUMNS` order), its value, whether the
 *  closed-loop reading added it, and what it says. */
export interface TrainingEvent {
  row: number;
  column: number;
  value: number;
  correction: boolean;
  says: TrainingSays;
}

/** A track on the flight clock: the observed one, or the one the executor flew. */
export interface TrainingTrack {
  /** Flight time (s from the observed row 0), one point per 2 s row. */
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeMslM: number[];
  /** The height Cesium draws in: MSL plus the flight's runway's HAE − MSL. */
  altitudeHaeM: number[];
  /** As the file has it: the observed track continues past 360°, a flown one is within [0, 360). */
  trackDeg: number[];
  /** `trackDeg` continuous, on the branch of the observed track at the same time: what a chart plots. */
  trackPlotDeg: number[];
  groundSpeedMps: number[];
  verticalRateMps: number[];
  attitude: TrainingAttitude;
}

/** A heading word's band: its target ± the tolerance over the rows it is judged on, and each judged row's verdict. Rows
 *  count from the envelopes' origin (the observed row 0, or the first predicted step). */
export interface TrainingHeadingBand {
  row: number;
  firstRow: number;
  /** One past the last row judged. */
  stopRow: number;
  /** The word's target track, compass degrees in [0, 360). */
  targetDeg: number;
  toleranceDeg: number;
  inside: boolean[];
}

export interface TrainingAltitudeTube {
  row: number;
  /** One past the last row the tube covers. */
  endRow: number;
  /** The level above E; null: "no level-off". */
  levelM: number | null;
  /** The tube's edges, MSL, one per row it covers. */
  lowMslM: number[];
  highMslM: number[];
  rows: number;
  inside: number;
  contained: boolean;
}

export interface TrainingSpeedSpan {
  row: number;
  endRow: number;
  targetMps: number;
  toleranceMps: number;
  /** The row the speed reached its band at. */
  arrivalRow: number;
  transitionOk: boolean;
  accelOk: boolean;
  contained: boolean;
}

export interface TrainingEnvelopes {
  heading: TrainingHeadingBand[];
  altitude: TrainingAltitudeTube[];
  speed: TrainingSpeedSpan[];
}

/** The labeller's kind of each word of the open-loop sentence. */
export interface TrainingWordKindEntry {
  row: number;
  column: number;
  kind: string;
}

export interface TrainingOpenLoop {
  rows: number;
  words: number[][];
  events: TrainingEvent[];
  kinds: TrainingWordKindEntry[];
  captureRow: number;
  unspecifiedRow: number;
  goAroundRows: number[];
  runwayAgainRows: number[];
  envelopes: TrainingEnvelopes;
}

/** The decision-altitude check at the crossing (D38): the flown point, its place against the glidepath and the cone. */
export interface TrainingDecision {
  /** The cycle of the set's executor (`TrainingSample.executor.cycleS`) from the first predicted step. */
  cycle: number;
  eM: number;
  nM: number;
  latDeg: number;
  lonDeg: number;
  heightMslM: number;
  rightM: number;
  coneHalfWidthM: number;
  aboveGlidepathM: number;
  lateralOk: boolean;
  verticalOk: boolean;
  passed: boolean;
}

export interface TrainingCrossing {
  /** Metres right of the centreline / above the threshold, at the crossing. */
  crossM: number;
  heightM: number;
  /** The executor's cycle (fractional) from the first predicted step. */
  atCycle: number;
  runwayIndex: number;
  /** null: no DA check (a crossing above the landing screen, or no DA point). */
  decision: TrainingDecision | null;
}

export interface TrainingReplay {
  outcome: TrainingOutcome;
  /** The executor's cycles from the first predicted step to the judge's outcome row. */
  endCycle: number;
  /** null: the flight did not cross the threshold. */
  crossing: TrainingCrossing | null;
  flewTheSentence: boolean;
  notReached: number;
  /** The judge's envelopes on the flown track, rows from the first predicted step; null: none were read. */
  envelopes: TrainingEnvelopes | null;
}

/** The closed-loop sentence at one Δ and what flying it gave. */
export interface TrainingClosedLoop {
  rowIntervalS: number;
  firstRow: number;
  startRow: number;
  flownFromRow: number;
  /** Flight time of the first predicted step: Δ row 0 of the words and cycle 0 of the replay. */
  startS: number;
  words: number[][];
  events: TrainingEvent[];
  /** Per Δ row, the flown track's distance from the observed one (null: the observed path has ended). */
  lateralM: Array<number | null>;
  verticalM: Array<number | null>;
  timedOut: boolean;
  /** The executor's control cycle (s): a replay cycle (`endCycle`, `atCycle`, a decision's `cycle`) is this long. */
  cycleS: number;
  /** The flight the executor flew, from the first predicted step to the judge's outcome (`replay.track`), on the 2 s rows. */
  flown: TrainingTrack;
  replay: TrainingReplay;
}

export interface TrainingFlight {
  datasetId: string;
  flightKey: string;
  /** The id part of the flight key (the callsign): a label only — a flight is its key. */
  callsign: string;
  split: (typeof TRAINING_SPLITS)[number];
  stratum: TrainingStratum;
  kind: string;
  /** null: the flight's identity is unresolved. */
  typecode: string | null;
  /** "own dynamics" or "stand-in dynamics". */
  group: string;
  runway: string;
  runwayIndex: number;
  entryTimeUtc: string;
  haeMinusMslM: number;
  observed: TrainingTrack;
  openLoop: TrainingOpenLoop;
  /** By row interval, as the set's `rowIntervalsS`. */
  closedLoop: Record<string, TrainingClosedLoop>;
}

export interface TrainingSample {
  setId: string;
  airport: string;
  /** The executor's control cycle (s), the unit of every replay cycle in the set. */
  executor: { cycleS: number };
  formats: Record<string, string>;
  source: TrainingSource;
  cohort: TrainingCohort;
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidatesSha256: string;
  candidates: TrainingCandidate[];
  flights: TrainingFlight[];
}

// ── the flight on screen ─────────────────────────────────────────────────────

/** What the panel publishes for the sentence bar, the read-back window and the 3D layer: the flight on screen with the set
 *  and airport it is read from (a flight key alone repeats across the sets of one airport). */
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

export function trainingSelectionOf(sample: TrainingSample, flight: TrainingFlight): TrainingSelection {
  return { airport: sample.airport, setId: sample.setId, vocabulary: sample.vocabulary, candidates: sample.candidates, flight };
}

// ── reading a sentence ───────────────────────────────────────────────────────

/**
 * The sentence the views read: the labelled (open-loop) one on the observed rows, or the closed-loop one at a Δ. Its
 * rows are ``rowS`` apart from ``originS`` (flight time); the envelopes are on 2 s rows from ``envelopeOriginS`` and
 * judge ``judged`` — the observed track (open loop) or the flown one (closed loop).
 */
export interface TrainingReading {
  loop: "open" | "closed";
  /** Δ for a closed-loop reading; null for the open loop. */
  intervalS: number | null;
  rows: number;
  rowS: number;
  originS: number;
  /** Flight time the sentence's last row ends at. */
  endS: number;
  events: TrainingEvent[];
  envelopeOriginS: number;
  envelopes: TrainingEnvelopes | null;
  /** The track the envelopes judge. */
  judged: TrainingTrack;
  /** The observed track beside it (the same track for the open loop). */
  observed: TrainingTrack;
  closed: TrainingClosedLoop | null;
  open: TrainingOpenLoop;
}

const readings = new WeakMap<TrainingFlight, Map<number | null, TrainingReading>>();

/** The reading of ``flight`` at row interval ``intervalS`` (null: the labelled, open-loop sentence). */
export function trainingReadingOf(flight: TrainingFlight, stepS: number, intervalS: number | null): TrainingReading {
  const cached = readings.get(flight)?.get(intervalS);
  if (cached !== undefined) return cached;
  const open = flight.openLoop;
  let reading: TrainingReading;
  if (intervalS === null) {
    reading = {
      loop: "open", intervalS: null, rows: open.rows, rowS: stepS, originS: flight.observed.tS[0],
      endS: flight.observed.tS[0] + open.rows * stepS, events: open.events, envelopeOriginS: flight.observed.tS[0],
      envelopes: open.envelopes, judged: flight.observed, observed: flight.observed, closed: null, open,
    };
  } else {
    const closed = flight.closedLoop[String(intervalS)];
    const rows = closed.words.length;
    reading = {
      loop: "closed", intervalS, rows, rowS: intervalS, originS: closed.startS, endS: closed.startS + rows * intervalS,
      events: closed.events, envelopeOriginS: closed.startS, envelopes: closed.replay.envelopes, judged: closed.flown,
      observed: flight.observed, closed, open,
    };
  }
  const byInterval = readings.get(flight) ?? new Map<number | null, TrainingReading>();
  byInterval.set(intervalS, reading);
  readings.set(flight, byInterval);
  return reading;
}

/** Flight time of sentence row ``row``. */
export function readingRowTimeS(reading: TrainingReading, row: number): number {
  return reading.originS + row * reading.rowS;
}

/** The sentence row in force at flight time ``atS``: null before the sentence opens, the last row at and after its end. */
export function readingRowAt(reading: TrainingReading, atS: number): number | null {
  if (atS < reading.originS - 1e-9) return null;
  return Math.min(Math.floor((atS - reading.originS) / reading.rowS + 1e-9), reading.rows - 1);
}

/** Flight time of a replay cycle (cycles of the set's executor, counted from the first predicted step). */
export function closedCycleTimeS(closed: TrainingClosedLoop, cycles: number): number {
  return closed.startS + cycles * closed.cycleS;
}

/** The cycle of the last state of a flight that ended at the judge's outcome row ``endCycle``: the row itself, but for a
 *  dynamics failure the failed state is left out — one before.
 *  MIRROR of `aeroviz_backend/autopilot_segment/fly.py` `last_state` and `training_export.replay_payload`. */
export function lastStateCycle(outcome: TrainingOutcome, endCycle: number): number {
  return outcome === "dynamics_failure" ? endCycle - 1 : endCycle;
}

/** A word said at Δ row ``row`` after the judge's outcome (its cycle is past the flight's last state): the flight the judge read
 *  never hears it, so it has no segment to fly (the backend refuses it). */
export function wordUnreached(closed: TrainingClosedLoop, row: number): boolean {
  return (row * closed.rowIntervalS) / closed.cycleS > lastStateCycle(closed.replay.outcome, closed.replay.endCycle);
}

/** One word of a column and the rows it is in force: from its row to the next word of its column (or the sentence's end). */
export interface TrainingWordRun {
  row: number;
  endRow: number;
  event: TrainingEvent;
  /** Its place among its column's words. */
  index: number;
}

/** The runs of one column of a sentence. */
export function sentenceColumnRuns(reading: TrainingReading, column: TrainingColumn): TrainingWordRun[] {
  const index = TRAINING_COLUMN_INDEX[column];
  const events = reading.events.filter((event) => event.column === index);
  return events.map((event, position) => ({
    row: event.row, endRow: position + 1 < events.length ? events[position + 1].row : reading.rows, event, index: position,
  }));
}

/** The word of one column in force at a sentence row, or null before the sentence opens. */
export function sentenceWordAt(reading: TrainingReading, column: TrainingColumn, row: number | null): TrainingWordRun | null {
  if (row === null) return null;
  return sentenceColumnRuns(reading, column).find((run) => run.row <= row && row < run.endRow) ?? null;
}

/** A word's row in the envelopes' rows (2 s, from the envelopes' origin): the judge hears a closed-loop word on the cycle
 *  that starts its Δ row, so a Δ of 4 s puts the word at envelope row 2 × its row. */
export function trainingEnvelopeRow(reading: TrainingReading, stepS: number, word: TrainingWordRun): number {
  return word.row * (reading.rowS / stepS);
}

/** Where a word's envelope sits in its column's list (`envelopes.heading` / `altitude` / `speed`), or null: a runway or
 *  angle word has none, nor a speed word left to the pilot, nor a word the flown flight never reached. */
export function trainingEnvelopeIndex(
  reading: TrainingReading, stepS: number, column: TrainingColumn, word: TrainingWordRun,
): number | null {
  if (reading.envelopes === null) return null;
  const row = trainingEnvelopeRow(reading, stepS, word);
  const list = column === "heading" ? reading.envelopes.heading : column === "altitude" ? reading.envelopes.altitude
    : column === "speed" ? reading.envelopes.speed : [];
  const index = list.findIndex((item) => item.row === row);
  return index < 0 ? null : index;
}

/** The closed-loop words the reading added, counted. */
export function correctionCount(events: TrainingEvent[]): number {
  return events.filter((event) => event.correction).length;
}

// ── a word as a person reads it (the exporter decoded it; this is its spelling) ──

function signedDegrees(value: number): string {
  if (value === 0) return "0°";
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(0)}°`;
}

/** A word as its band reads it, short. */
export function trainingBandLabel(says: TrainingSays): string {
  switch (says.column) {
    case "runway":
      return says.goAround ? "go-around" : says.runway;
    case "heading":
      return signedDegrees(says.relativeDeg);
    case "altitude":
      return says.noLevelOff ? "no level-off" : `${says.levelM.toFixed(0)} m`;
    case "angle":
      if (says.level) return "level";
      return says.climb ? `climb ${Math.abs(says.angleDeg).toFixed(1)}°` : `${says.angleDeg.toFixed(1)}°`;
    case "speed":
      return says.speedMps === null ? "unspecified" : `${says.speedMps.toFixed(0)} m/s`;
  }
}

/** A word in full: what it says in words, for tooltips and the live line. */
export function trainingWordLabel(says: TrainingSays): string {
  switch (says.column) {
    case "runway":
      return says.goAround ? "go-around" : `runway ${says.runway}`;
    case "heading":
      return `${signedDegrees(says.relativeDeg)} from the course (track ${says.trackDeg.toFixed(0).padStart(3, "0")}°)`;
    case "altitude":
      return says.noLevelOff ? "no level-off" : `${says.levelM.toFixed(0)} m above the airport (${says.mslM.toFixed(0)} m MSL)`;
    case "angle":
      if (says.level) return "level";
      return says.climb ? `climb ${Math.abs(says.angleDeg).toFixed(1)}°` : `descent ${says.angleDeg.toFixed(1)}°`;
    case "speed":
      return says.speedMps === null ? "speed unspecified (the pilot's own)" : `${says.speedMps.toFixed(0)} m/s ground speed`;
  }
}

// ── times, rows and the rows outside ─────────────────────────────────────────

/** The last index of ascending ``tS`` at or before ``seconds`` (0 before the first). */
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

/**
 * The rows judged outside, as inclusive [first, last] row spans to DRAW on a line of rows `0..lastRow`: ``inside`` holds
 * one verdict per row from ``firstRow``, and each run of rows outside is carried on to the next row, so that one row
 * outside is a segment (back to the row before it at the line's end) — never a single point, which draws nothing. One
 * rule for every verdict drawn red; the verdicts are the file's own.
 */
/** How many words left their envelope, as the judge counts them (`replay.word_results`): a heading word with a row of its
 *  band outside (an empty band was not judged), an altitude word whose tube did not contain it, a speed word whose span
 *  did not. */
export function wordsOutside(envelopes: TrainingEnvelopes): number {
  return envelopes.heading.filter((band) => band.inside.some((inside) => !inside)).length +
    envelopes.altitude.filter((tube) => !tube.contained).length + envelopes.speed.filter((span) => !span.contained).length;
}

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

/** Track degrees made continuous (no jump of more than 180° between neighbours), and — with a ``reference`` — shifted by
 *  whole turns so the first value is the nearest to it: the branch a chart draws a heading on. Drawing only: the
 *  verdicts are the file's. */
export function unwrapDegrees(values: number[], reference?: number): number[] {
  const out = [...values];
  for (let i = 1; i < out.length; i += 1) out[i] = out[i - 1] + (((values[i] - values[i - 1] + 540) % 360) - 180);
  if (reference === undefined || out.length === 0) return out;
  const turns = Math.round((reference - out[0]) / 360);
  return out.map((value) => value + 360 * turns);
}

/** ``target`` (compass degrees) on the branch nearest to ``reference``: a band is drawn on the branch of the track. */
export function nearestBranch(target: number, reference: number): number {
  return target + 360 * Math.round((reference - target) / 360);
}

// ── the index ────────────────────────────────────────────────────────────────

function parseCohort(cohort: Reader): TrainingCohort {
  return {
    splits: cohort.record("splits", (value, where) => {
      if (typeof value !== "number" || !Number.isInteger(value) || value < 0) throw new Error(`${where} is not a count`);
      return value;
    }),
    perStratum: cohort.count("perStratum", 1), strata: cohort.strings("strata"), seed: cohort.number("seed"),
    drawnFrom: cohort.string("drawnFrom"),
  };
}

export function parseFormats(reader: Reader): Record<string, string> {
  return reader.record("formats", (value, where) => {
    if (typeof value !== "string" || value.length === 0) throw new Error(`${where} is not a format name`);
    return value;
  });
}

function parseSource(reader: Reader): TrainingSource {
  const git = reader.child("git");
  return {
    instructions: reader.string("instructions"), executor: reader.string("executor"), specSha256: reader.string("specSha256"),
    executorSpecSha256: reader.string("executorSpecSha256"), git: { head: git.string("head"), dirty: git.boolean("dirty") },
  };
}

function parseSetEntry(entry: Reader): TrainingSetEntry {
  entry.oneOf("kind", [TRAINING_SET_KIND]);
  entry.oneOf("readingRule", [TRAINING_READING_RULE]);
  return {
    id: entry.string("id"), title: entry.string("title"), file: entry.string("file"), flights: entry.count("flights"),
    formats: parseFormats(entry), cohort: parseCohort(entry.child("cohort")), source: parseSource(entry.child("source")),
  };
}

/** Parse the index. A bad entry is rejected on its own; only an index that is not an index at all fails the call — and a
 *  schema other than `TRAINING_INDEX_SCHEMA` is refused whole, naming the one found and the one expected. */
export function parseTrainingIndex(raw: unknown): Parsed<TrainingIndex> {
  const manifest = parseManifest(raw, { name: "index", schema: TRAINING_INDEX_SCHEMA, listKey: "sets", entryName: "set" },
    parseSetEntry);
  if (!manifest.ok) return manifest;
  return { ok: true, value: { airport: manifest.value.airport, sets: manifest.value.entries, rejected: manifest.value.rejected } };
}

// ── one set ──────────────────────────────────────────────────────────────────

export function parseVocabulary(reader: Reader): TrainingVocabulary {
  reader.oneOf("readingRule", [TRAINING_READING_RULE]);
  reader.sameNames("columns", TRAINING_COLUMNS);
  const angleClasses = reader.children("angleClasses").map((angle, index) => {
    if (angle.number("index") !== index) angle.fail(`index is ${angle.raw("index")}, expected ${index}: the table is in class order`);
    return {
      index, name: angle.string("name"), nominalDeg: angle.number("nominalDeg"), lowDeg: angle.number("lowDeg"),
      highDeg: angle.number("highDeg"),
    };
  });
  const levels = reader.numbers("altitudeLevelsM");
  const speed = reader.child("speed");
  const rowIntervalsS = reader.numbers("rowIntervalsS");
  if (rowIntervalsS.length === 0) reader.fail("rowIntervalsS is empty: there is no closed-loop sentence to read");
  return {
    readingRule: TRAINING_READING_RULE, stepS: reader.number("stepS"), headingStepDeg: reader.number("headingStepDeg"),
    headingLeadS: reader.number("headingLeadS"), headingToleranceDeg: reader.number("headingToleranceDeg"),
    altitudeLevelsM: levels, altitudeTolerancesM: reader.numbers("altitudeTolerancesM", levels.length),
    noLevelOff: reader.integer("noLevelOff", 0, Number.MAX_SAFE_INTEGER), angleClasses,
    speed: {
      minMps: speed.number("minMps"), stepMps: speed.number("stepMps"), levels: speed.count("levels", 1),
      unspecified: speed.count("unspecified"), toleranceMps: speed.number("toleranceMps"),
    },
    closedLoopLateralM: reader.number("closedLoopLateralM"), closedLoopVerticalM: reader.number("closedLoopVerticalM"),
    rowIntervalsS,
  };
}

export function parseCandidates(reader: Reader): TrainingCandidate[] {
  const candidates = reader.children("candidates").map((candidate, index) => {
    if (candidate.number("index") !== index) candidate.fail(`index is ${candidate.raw("index")}, expected ${index}`);
    const path = candidate.child("verticalPath");
    return {
      index, ident: candidate.string("ident"), thresholdEM: candidate.number("thresholdEM"),
      thresholdNM: candidate.number("thresholdNM"), latDeg: candidate.number("latDeg"), lonDeg: candidate.number("lonDeg"),
      courseDeg: candidate.number("courseDeg"), elevationM: candidate.number("elevationM"),
      lengthM: candidate.number("lengthM"), haeMinusMslM: candidate.number("haeMinusMslM"),
      verticalPath: {
        crossingHeightM: path.number("crossingHeightM"), glidepathDeg: path.number("glidepathDeg"),
        decisionHeightM: path.number("decisionHeightM"),
      },
    };
  });
  if (candidates.length === 0) reader.fail("candidates is empty: the runway word has nothing to point at");
  return candidates;
}

/** The word grid (``[rows][5]``, "unchanged" = -1). */
export function parseGrid(reader: Reader, key: string, rows?: number): number[][] {
  const grid = reader.list(key).map((row, index) => {
    if (!Array.isArray(row) || row.length !== TRAINING_COLUMNS.length || !row.every((value) => Number.isInteger(value))) {
      reader.fail(`${key}[${index}] is not ${TRAINING_COLUMNS.length} whole numbers, one per column`);
    }
    return row as number[];
  });
  if (rows !== undefined && grid.length !== rows) reader.fail(`${key} holds ${grid.length} rows, expected ${rows}`);
  if (grid.length === 0) reader.fail(`${key} is empty`);
  return grid;
}

function parseSays(reader: Reader, column: number, value: number, candidates: TrainingCandidate[]): TrainingSays {
  switch (TRAINING_COLUMNS[column]) {
    case "runway": {
      if (reader.raw("goAround") !== undefined) {
        if (reader.boolean("goAround") !== true) reader.fail("goAround is false");
        return { column: "runway", goAround: true };
      }
      const runwayIndex = reader.integer("runwayIndex", 0, candidates.length - 1);
      const runway = reader.string("runway");
      if (candidates[runwayIndex].ident !== runway) reader.fail(`runway ${runway} is not candidate ${runwayIndex}'s (${candidates[runwayIndex].ident})`);
      return { column: "runway", goAround: false, runway, runwayIndex };
    }
    case "heading":
      return { column: "heading", relativeDeg: reader.number("relativeDeg"), trackDeg: reader.number("trackDeg") };
    case "altitude":
      return reader.boolean("noLevelOff")
        ? { column: "altitude", noLevelOff: true }
        : { column: "altitude", noLevelOff: false, levelM: reader.number("levelM"), mslM: reader.number("mslM") };
    case "angle":
      return { column: "angle", angleDeg: reader.number("angleDeg"), climb: reader.boolean("climb"), level: reader.boolean("level") };
    case "speed":
      return { column: "speed", speedMps: reader.nullableNumber("speedMps") };
  }
  return reader.fail(`column ${column} (value ${value}) is none of the five`);
}

/** A sentence's events, checked to be exactly the words of its grid, in (row, column) order. */
export function parseEvents(reader: Reader, grid: number[][], candidates: TrainingCandidate[], corrections: boolean): TrainingEvent[] {
  const events = reader.children("events").map((event) => {
    const column = event.integer("column", 0, TRAINING_COLUMNS.length - 1);
    const value = event.integer("value", TRAINING_RUNWAY_GO_AROUND, Number.MAX_SAFE_INTEGER);
    const correction = event.boolean("correction");
    if (correction && !corrections) event.fail("correction is true in the open-loop sentence, which no reading adds words to");
    const says = parseSays(event.child("says"), column, value, candidates);
    if ((says.column === "runway" && says.goAround) !== (value === TRAINING_RUNWAY_GO_AROUND)) {
      event.fail(`value ${value} and what it says disagree about the go-around`);
    }
    return { row: event.integer("row", 0, grid.length - 1), column, value, correction, says };
  });
  const words = grid.flatMap((row, r) => row.flatMap((value, column) => (value === TRAINING_UNCHANGED ? [] : [{ row: r, column, value }])));
  if (words.length !== events.length || words.some((word, i) => word.row !== events[i].row || word.column !== events[i].column
    || word.value !== events[i].value)) {
    reader.fail(`events are not the ${words.length} words of the grid, in (row, column) order`);
  }
  if (grid[0].some((value) => value === TRAINING_UNCHANGED)) reader.fail("row 0 does not say every column");
  return events;
}

export function parseAttitudeOf(reader: Reader, rows: number): TrainingAttitude {
  return readAttitude(reader.child("attitude"), rows);
}

/** The observed track on its 2 s rows, which must be the vocabulary's grid from flight time 0. */
function parseObserved(reader: Reader, stepS: number, haeMinusMslM: number): TrainingTrack {
  const rows = reader.count("rows", 2);
  const tS = reader.numbers("timeS", rows);
  for (let row = 0; row < rows; row += 1) {
    if (Math.abs(tS[row] - row * stepS) > 1e-3) reader.fail(`timeS[${row}] is ${tS[row]} s, not ${row * stepS}: the rows are the vocabulary's ${stepS} s grid from 0`);
  }
  const altitudeMslM = reader.numbers("altitudeMslM", rows);
  const trackDeg = reader.numbers("trackDeg", rows);
  return {
    tS, eM: reader.numbers("eM", rows), nM: reader.numbers("nM", rows), lat: reader.numbers("latDeg", rows),
    lon: reader.numbers("lonDeg", rows), altitudeMslM, altitudeHaeM: altitudeMslM.map((value) => value + haeMinusMslM),
    trackDeg, trackPlotDeg: unwrapDegrees(trackDeg), groundSpeedMps: reader.numbers("groundSpeedMps", rows),
    verticalRateMps: reader.numbers("verticalRateMps", rows), attitude: parseAttitudeOf(reader, rows),
  };
}

function headingBands(list: Reader[]): TrainingHeadingBand[] {
  return list.map((item) => {
    const firstRow = item.count("firstRow");
    const stopRow = item.count("stopRow");
    if (stopRow < firstRow) item.fail(`stopRow ${stopRow} is before firstRow ${firstRow}`);
    return {
      row: item.count("row"), firstRow, stopRow, targetDeg: item.number("targetDeg"), toleranceDeg: item.number("toleranceDeg"),
      inside: item.flags("inside", stopRow - firstRow),
    };
  });
}

function altitudeTubes(list: Reader[]): TrainingAltitudeTube[] {
  return list.map((item) => {
    const row = item.count("row");
    const endRow = item.count("endRow");
    if (endRow < row) item.fail(`endRow ${endRow} is before row ${row}`);
    const lowMslM = item.numbers("lowMslM", endRow - row);
    const rows = item.count("rows");
    const inside = item.count("inside");
    if (inside > rows) item.fail(`inside ${inside} is more than its ${rows} rows`);
    return {
      row, endRow, levelM: item.nullableNumber("levelM"), lowMslM, highMslM: item.numbers("highMslM", endRow - row), rows, inside,
      contained: item.boolean("contained"),
    };
  });
}

function speedSpans(list: Reader[]): TrainingSpeedSpan[] {
  return list.map((item) => {
    const row = item.count("row");
    const endRow = item.count("endRow");
    if (endRow < row) item.fail(`endRow ${endRow} is before row ${row}`);
    return {
      row, endRow, targetMps: item.number("targetMps"), toleranceMps: item.number("toleranceMps"),
      arrivalRow: item.count("arrivalRow"), transitionOk: item.boolean("transitionOk"), accelOk: item.boolean("accelOk"),
      contained: item.boolean("contained"),
    };
  });
}

function parseEnvelopes(reader: Reader): TrainingEnvelopes {
  return {
    heading: headingBands(reader.children("heading")), altitude: altitudeTubes(reader.children("altitude")),
    speed: speedSpans(reader.children("speed")),
  };
}

function parseOpenLoop(reader: Reader, observedRows: number, candidates: TrainingCandidate[]): TrainingOpenLoop {
  const rows = reader.count("rows", 1);
  if (rows !== observedRows) reader.fail(`rows is ${rows}, but the observed track has ${observedRows}`);
  const words = parseGrid(reader, "words", rows);
  const events = parseEvents(reader, words, candidates, false);
  const envelopes = parseEnvelopes(reader.child("envelopes"));
  requireWithin(reader.child("envelopes"), envelopes, observedRows, "the observed track");
  const kinds = reader.children("kinds").map((item) => ({
    row: item.integer("row", 0, rows - 1), column: item.integer("column", 0, TRAINING_COLUMNS.length - 1),
    // why the labeller said it (`labeller.records.Instruction.kind`): free text for readouts, many writers, no list
    kind: item.string("kind"),
  }));
  if (kinds.length !== events.length || kinds.some((kind, i) => kind.row !== events[i].row || kind.column !== events[i].column)) {
    reader.fail(`kinds are not the ${events.length} words of the sentence, in order`);
  }
  return {
    rows, words, events, kinds, captureRow: reader.integer("captureRow", 0, rows), unspecifiedRow: reader.integer("unspecifiedRow", 0, rows),
    goAroundRows: reader.numbers("goAroundRows"), runwayAgainRows: reader.numbers("runwayAgainRows"), envelopes,
  };
}

function parseDecision(reader: Reader): TrainingDecision {
  return {
    cycle: reader.count("cycle"), eM: reader.number("eM"), nM: reader.number("nM"), latDeg: reader.number("latDeg"),
    lonDeg: reader.number("lonDeg"), heightMslM: reader.number("heightMslM"), rightM: reader.number("rightM"),
    coneHalfWidthM: reader.number("coneHalfWidthM"), aboveGlidepathM: reader.number("aboveGlidepathM"),
    lateralOk: reader.boolean("lateralOk"), verticalOk: reader.boolean("verticalOk"), passed: reader.boolean("passed"),
  };
}

/** A threshold crossing and its decision-altitude check, as the export and the live answer write them. */
export function readCrossing(reader: Reader, candidateCount: number): TrainingCrossing {
  const decision = reader.nullableChild("decision");
  return {
    crossM: reader.number("crossM"), heightM: reader.number("heightM"), atCycle: reader.number("atCycle"),
    runwayIndex: reader.integer("runwayIndex", 0, candidateCount - 1), decision: decision === null ? null : parseDecision(decision),
  };
}

/** Every row an envelope ends at lies within the track it judged (``rows``, named ``track``; the export guarantees it) —
 *  but for a heading band whose word's lead runs past the track's end: it is empty (`firstRow == stopRow`, no verdicts),
 *  may lie anywhere, and is not drawn. So no drawing helper meets a row past its line. */
function requireWithin(reader: Reader, envelopes: TrainingEnvelopes, rows: number, track: string): void {
  const ends = [
    ...envelopes.heading.filter((band) => band.stopRow > band.firstRow).map((band) => ["heading stopRow", band.stopRow] as const),
    ...envelopes.altitude.map((tube) => ["altitude endRow", tube.endRow] as const),
    ...envelopes.speed.map((span) => ["speed endRow", span.endRow] as const),
  ];
  const past = ends.find(([, row]) => row > rows);
  if (past) reader.fail(`an envelope's ${past[0]} is ${past[1]}, past the ${rows} rows of ${track}`);
}

function parseReplay(
  reader: Reader, candidates: TrainingCandidate[], cycleS: number, stepS: number,
): { replay: TrainingReplay; track: Reader; rows: number } {
  const crossing = reader.nullableChild("crossing");
  const envelopes = reader.nullableChild("envelopes");
  const outcome = reader.oneOf("outcome", TRAINING_OUTCOMES);
  const endCycle = reader.count("endCycle");
  const track = reader.child("track");
  const rows = track.count("rows", 1);
  const lastCycle = track.count("lastCycle");
  if (lastCycle !== lastStateCycle(outcome, endCycle)) {
    track.fail(`lastCycle is ${lastCycle}, but a flight that ended at cycle ${endCycle} (${outcome}) has its last state at ${lastStateCycle(outcome, endCycle)}`);
  }
  if (rows !== Math.floor((lastCycle * cycleS) / stepS + 1e-9) + 1) {
    track.fail(`rows is ${rows}, but ${lastCycle} cycles of ${cycleS} s are ${Math.floor((lastCycle * cycleS) / stepS + 1e-9) + 1} rows of ${stepS} s`);
  }
  const judged = envelopes === null ? null : parseEnvelopes(envelopes);
  if (judged !== null) requireWithin(envelopes!, judged, rows, "the flown track (replay.track)");
  return {
    replay: {
      outcome, endCycle, crossing: crossing === null ? null : readCrossing(crossing, candidates.length),
      flewTheSentence: reader.boolean("flewTheSentence"), notReached: reader.count("notReached"), envelopes: judged,
    },
    track, rows,
  };
}

function parseClosedLoop(
  reader: Reader, key: string, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[], observed: TrainingTrack,
  haeMinusMslM: number, cycleS: number,
): TrainingClosedLoop {
  const rowIntervalS = reader.number("rowIntervalS");
  if (String(rowIntervalS) !== key) reader.fail(`rowIntervalS is ${rowIntervalS}, but it is listed under ${key}`);
  const stepS = vocabulary.stepS;
  const every = rowIntervalS / stepS;
  if (!Number.isInteger(every) || every < 1) reader.fail(`rowIntervalS ${rowIntervalS} is not a whole number of ${stepS} s rows`);
  const firstRow = reader.count("firstRow");
  const startRow = reader.count("startRow");
  const flownFromRow = reader.count("flownFromRow");
  if (flownFromRow !== startRow * every) reader.fail(`flownFromRow is ${flownFromRow}, but Δ row ${startRow} is state row ${startRow * every}`);
  const words = parseGrid(reader, "words");
  const events = parseEvents(reader, words, candidates, true);
  const states = reader.child("states");
  const stateRows = states.count("rows", flownFromRow + 2);
  if (stateRows !== (words.length + startRow - 1) * every + 1) {
    states.fail(`rows is ${stateRows}, but ${words.length} Δ rows from Δ row ${startRow} cover ${(words.length + startRow - 1) * every + 1}`);
  }
  const onInterval = states.flags("onInterval", stateRows);
  const marked = onInterval.flatMap((on, row) => (on ? [row] : []));
  if (marked[startRow] !== flownFromRow || marked.some((row, i) => row !== i * every)) {
    states.fail(`onInterval does not mark every ${every}th state row from row 0, or Δ row ${startRow} is not state row ${flownFromRow}`);
  }
  // the states before the first predicted step are the observed ones (observed row firstRow + k)
  const stateE = states.numbers("eM", stateRows);
  const stateN = states.numbers("nM", stateRows);
  const stateHeight = states.numbers("heightMslM", stateRows);
  for (let k = 0; k < flownFromRow; k += 1) {
    if (Math.abs(stateE[k] - observed.eM[firstRow + k]) > 0.2) {
      states.fail(`eM[${k}] is ${stateE[k]} m, but the observed row ${firstRow + k} is at ${observed.eM[firstRow + k]} m: the states before the first predicted step are observed`);
    }
  }
  const startIndex = firstRow + flownFromRow;
  const startS = startIndex * stepS;
  const { replay, track, rows } = parseReplay(reader.child("replay"), candidates, cycleS, stepS);
  // the flown flight is `replay.track`; the stored states are the same flight on the rows both have, to the written rounding
  const eM = track.numbers("eM", rows);
  const nM = track.numbers("nM", rows);
  const heightMslM = track.numbers("heightMslM", rows);
  for (let i = 0; i < Math.min(rows, stateRows - flownFromRow); i += 1) {
    const apart = Math.max(Math.abs(eM[i] - stateE[flownFromRow + i]), Math.abs(nM[i] - stateN[flownFromRow + i]),
      Math.abs(heightMslM[i] - stateHeight[flownFromRow + i]));
    if (apart > 0.11) track.fail(`row ${i} is ${apart.toFixed(2)} m from the stored state ${flownFromRow + i}: replay.track and states are one flight`);
  }
  const trackDeg = track.numbers("trackDeg", rows);
  const flown: TrainingTrack = {
    tS: Array.from({ length: rows }, (_, i) => (startIndex + i) * stepS),
    eM, nM, lat: track.numbers("latDeg", rows), lon: track.numbers("lonDeg", rows), altitudeMslM: heightMslM,
    altitudeHaeM: heightMslM.map((value) => value + haeMinusMslM), trackDeg,
    trackPlotDeg: unwrapDegrees(trackDeg, observed.trackPlotDeg[Math.min(startIndex, observed.tS.length - 1)]),
    groundSpeedMps: track.numbers("groundSpeedMps", rows), verticalRateMps: track.numbers("verticalRateMps", rows),
    // `replay.attitude` is on the rows of `replay.track`
    attitude: parseAttitudeOf(reader.child("replay"), rows),
  };
  return {
    rowIntervalS, firstRow, startRow, flownFromRow, startS, words, events,
    lateralM: reader.numbersOrNull("lateralM", words.length), verticalM: reader.numbersOrNull("verticalM", words.length),
    timedOut: reader.boolean("timedOut"), cycleS, flown, replay,
  };
}

/** ``closedIntervals``: the Δ the flight's `closedLoop` is listed at, in order — the set's `rowIntervalsS` (a set of stage B
 *  lists the one Δ its prior said at: `trainingPriorSample.ts`). */
export function parseFlight(
  reader: Reader, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[], cycleS: number,
  closedIntervals: readonly number[],
): TrainingFlight {
  const runwayIndex = reader.integer("runwayIndex", 0, candidates.length - 1);
  const runway = reader.string("runway");
  if (candidates[runwayIndex].ident !== runway) reader.fail(`runway ${runway} is not candidate ${runwayIndex}'s (${candidates[runwayIndex].ident})`);
  const haeMinusMslM = reader.number("haeMinusMslM");
  if (Math.abs(haeMinusMslM - candidates[runwayIndex].haeMinusMslM) > 0.02) {
    reader.fail(`haeMinusMslM is ${haeMinusMslM}, but runway ${runway}'s is ${candidates[runwayIndex].haeMinusMslM}`);
  }
  const flightKey = reader.string("flightKey");
  const observed = parseObserved(reader.child("observed"), vocabulary.stepS, haeMinusMslM);
  const closedReader = reader.child("closedLoop");
  const expected = closedIntervals.map(String);
  const found = Object.keys(reader.raw("closedLoop") as Record<string, unknown>);
  if (found.length !== expected.length || found.some((key, i) => key !== expected[i])) {
    reader.fail(`closedLoop is listed at [${found.join(", ")}] s, expected the set's [${expected.join(", ")}]`);
  }
  const closedLoop = Object.fromEntries(expected.map((key) => [
    key, parseClosedLoop(closedReader.child(key), key, vocabulary, candidates, observed, haeMinusMslM, cycleS)]));
  return {
    datasetId: reader.string("datasetId"), flightKey, callsign: flightKey.split("_")[0], split: reader.oneOf("split", TRAINING_SPLITS),
    stratum: reader.oneOf("stratum", TRAINING_STRATA), kind: reader.string("kind"), typecode: reader.nullableString("typecode"),
    group: reader.string("group"), runway, runwayIndex, entryTimeUtc: reader.string("entryTimeUtc"), haeMinusMslM, observed,
    openLoop: parseOpenLoop(reader.child("openLoop"), observed.tS.length, candidates), closedLoop,
  };
}

/** Parse a set's sample. A schema other than `TRAINING_SAMPLE_SCHEMA` is refused whole, naming the one found and the one
 *  expected. */
export function parseTrainingSample(raw: unknown): Parsed<TrainingSample> {
  return attempt(() => {
    const sample = Reader.of(raw, "sample");
    sample.oneOf("schema", [TRAINING_SAMPLE_SCHEMA]);
    sample.oneOf("readingRule", [TRAINING_READING_RULE]);
    const vocabulary = parseVocabulary(sample.child("vocabulary"));
    const candidates = parseCandidates(sample);
    const frame = sample.child("airportFrame");
    const cycleS = sample.child("executor").number("cycleS");
    if (!(cycleS > 0)) sample.fail(`executor.cycleS is ${cycleS}, not a positive length`);
    // a 2 s row is a whole number of cycles (the export's int(round(step / cycle)) must not round)
    if (!Number.isInteger(vocabulary.stepS / cycleS)) {
      sample.fail(`a ${vocabulary.stepS} s row is not a whole number of ${cycleS} s cycles`);
    }
    const flights = sample.children("flights").map((flight) => parseFlight(flight, vocabulary, candidates, cycleS, vocabulary.rowIntervalsS));
    const keys = new Set(flights.map((flight) => flight.flightKey));
    if (keys.size !== flights.length) sample.fail("two flights carry one flight key: a flight is its key");
    return {
      setId: sample.string("setId"), airport: sample.string("airport"), executor: { cycleS }, formats: parseFormats(sample),
      source: parseSource(sample.child("source")), cohort: parseCohort(sample.child("cohort")), vocabulary,
      airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidatesSha256: sample.string("candidatesSha256"), candidates, flights,
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

/** The airport's Training directory, relative to the site root. */
export function trainingDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

/** The index the panel reads — shown in the empty state and used by the fetch below. */
export function trainingIndexPath(airportCode: string): string {
  return `${trainingDirectory(airportCode)}/${TRAINING_INDEX_FILE}`;
}

/** A file the index lists — a set's sample — relative to the airport's Training directory. */
export function trainingFilePath(airportCode: string, file: string): string {
  return `${trainingDirectory(airportCode)}/${file}`;
}

/** The airport's index, refused unless it is the one asked for (a file copied under another airport's directory). */
export async function fetchTrainingIndex(airportCode: string): Promise<Parsed<TrainingIndex>> {
  const parsed = parseTrainingIndex(await fetchJson<unknown>(trainingIndexPath(airportCode)));
  if (parsed.ok && parsed.value.airport !== airportCode) {
    return { ok: false, problem: `${trainingIndexPath(airportCode)} is ${parsed.value.airport}'s index, not ${airportCode}'s` };
  }
  return parsed;
}

/** A set's sample, refused unless it is the set and airport asked for. */
export async function fetchTrainingSample(airportCode: string, file: string, setId: string): Promise<Parsed<TrainingSample>> {
  const parsed = parseTrainingSample(await fetchJson<unknown>(trainingFilePath(airportCode, file)));
  if (parsed.ok && (parsed.value.airport !== airportCode || parsed.value.setId !== setId)) {
    return { ok: false, problem: `${file} holds set ${parsed.value.setId} of ${parsed.value.airport}, not ${setId} of ${airportCode}` };
  }
  return parsed;
}
