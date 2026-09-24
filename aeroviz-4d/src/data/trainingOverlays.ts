/**
 * trainingOverlays.ts
 * -------------------
 * What another model makes of a Training set's own flights, drawn over the set: the EXECUTOR's replay (the truth
 * sentence flown from row 0 — its track, its outcome and every word's verdict) and the PRIOR's predictions (at every
 * step of the truth sentence, what it gives each column). Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`
 * §2.6 and §4.6; the writers: `experiments/executor_training_export.py`, `experiments/prior_training_export.py`.
 *
 * AN OVERLAY IS A FILE BESIDE ITS SET, NEVER INSIDE IT. `training/overlays.json` lists them, each naming the set it
 * is drawn over (`base`) and that set's sample by its sha256; the payload repeats the set id, the sample's time of
 * writing and the spec, and holds one entry per flight of the set, in the set's order. The reader binds the two by
 * those fields and by the flights themselves — the same keys, and for the executor the same words, for the prior the
 * same number of steps — and refuses the payload whole if any differs: a verdict drawn on the wrong word is worse
 * than none.
 *
 * NOTHING HERE IS COMPUTED. Every verdict is the executor's judge's, re-flown and checked against its formal replay
 * in Python — a heading word's with its band and a verdict per row on the flown track the judge read — and every
 * probability is the prior's own. The reader checks bookkeeping — lengths, ranges, the binding — and hands numbers to
 * the views.
 *
 * NO COMPATIBILITY: the three schemas are pinned below and anything else is refused by name.
 *
 * SI units only: metres, m/s, degrees, seconds.
 */

import { fetchJson } from "../utils/fetchJson";
import {
  headingBandProblem,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  TRAINING_SPEC_SHA256,
  TRAINING_UNCHANGED,
  trainingClassCount,
  trainingDirectory,
  type Parsed,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingSample,
  type TrainingVocabulary,
} from "./trainingSample";

/** MIRROR of the exporter's `OVERLAYS_SCHEMA` (`instruction_training_export.py`): the manifest of overlays. */
export const TRAINING_OVERLAYS_SCHEMA = "aeroviz-training-overlays-v1";
/** MIRROR of `OVERLAY_KINDS`: what an overlay can be. */
export const TRAINING_OVERLAY_KINDS = ["executor-replay", "prior-prediction"] as const;
export type TrainingOverlayKind = (typeof TRAINING_OVERLAY_KINDS)[number];
/** MIRROR of `executor_training_export.SCHEMA`: v2 (instruction-v3) gives each heading word judged its band and a
 *  verdict per row, and each flight judged its flown track as the judge read it; v1 (turns and holds) is refused. */
export const TRAINING_EXECUTOR_SCHEMA = "aeroviz-training-executor-v2";
/** MIRROR of `executor_training_export.STATUSES`: one per word. */
export const TRAINING_EXECUTOR_STATUSES = [
  "inside", "outside", "not judged", "not reached", "superseded", "no check",
] as const;
export type TrainingExecutorStatus = (typeof TRAINING_EXECUTOR_STATUSES)[number];
/** MIRROR of `prior_training_export.SCHEMA`. */
export const TRAINING_PRIOR_SCHEMA = "aeroviz-training-prior-v3";

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingOverlayEntry {
  id: string;
  kind: TrainingOverlayKind;
  /** The set it is drawn over. */
  base: string;
  /** The sha256 of that set's sample file when the overlay was written. */
  baseSampleSha256: string;
  title: string;
  /** Relative to the airport's `training/` directory. */
  file: string;
  flights: number;
}

export interface TrainingOverlays {
  airport: string;
  overlays: TrainingOverlayEntry[];
  rejected: Array<{ id: string; problem: string }>;
}

/** What an overlay says about the set it is drawn over. */
export interface TrainingOverlayBase {
  setId: string;
  sampleWrittenUtc: string;
  sampleSha256: string;
  specSha256: string;
}

export interface TrainingExecutorCheck {
  name: string;
  ok: boolean;
  /** For a check over rows: how many of them were inside. */
  inside: number | null;
  rows: number | null;
}

/** The executor's verdict on ONE word of the sentence (its row, column and value are the word's). */
export interface TrainingExecutorWord {
  row: number;
  column: number;
  value: number;
  status: TrainingExecutorStatus;
  /** The flown track's step the executor was told the word at; null when it never was. */
  flownRow: number | null;
  /** A heading word the judge judged: its band over the flown rows it was judged on (from `flownRow` plus the lead),
   *  a verdict per row, on the chart branch of `TrainingExecutorFlight.judgedTrackDeg`; null for every other word. */
  heading: TrainingHeadingBand | null;
  checks: TrainingExecutorCheck[];
  /** Why a word is not judged, not reached, superseded or has no check of its own. */
  reason: string | null;
}

/** The flown track every sentence step, from row 0 to its outcome's row, on the executor's own clock. */
export interface TrainingExecutorTrack {
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
  groundSpeedMps: number[];
  /** Unwrapped, on the observed smoothed track's branch at row 0. */
  trackDeg: number[];
  /** The horizontal distance the executor flew. */
  distanceM: number[];
}

export interface TrainingExecutorFlight {
  flightKey: string;
  datasetId: string;
  /** "own dynamics" / "stand-in dynamics", or why the replay does not fly it. */
  group: string;
  flown: boolean;
  outcome: string | null;
  flewTheSentence: boolean | null;
  endS: number | null;
  /** At the threshold: metres right of the centreline and above the threshold, and when. */
  crossing: { crossM: number; heightM: number; atS: number } | null;
  /** Why the labeller's gate refused the flown track (nothing is then judged). */
  refused: string | null;
  /** The formal replay's evaluation verdicts: the flown track's and the observed one's. */
  evaluation: { replay: string; observed: string } | null;
  alignment: { meanHorizontalDistanceM: number; meanVerticalDistanceM: number; landingTimeMinusObservedS: number | null } | null;
  limits: { cycles: number; bound: Record<string, number> } | null;
  counts: { wordsJudged: number; wordsInside: number; headingWordsNotJudged: number } | null;
  track: TrainingExecutorTrack | null;
  /** The flown track as the judge read it — the labeller's gate: smoothed, cut at the landing — every sentence step
   *  from row 0 (its step k is `track.tS[k]`, checked), on the same branch as `track.trackDeg`; null when the gate
   *  refused it, and for a dynamics failure (its judge read the failed state, which `track` leaves out: its words keep
   *  their statuses and checks, and draw no band). */
  judgedTrackDeg: number[] | null;
  /** One per word of the base flight's sentence, in its event order; empty when not flown. */
  words: TrainingExecutorWord[];
}

export interface TrainingGateCell {
  flights: number;
  landed: number | null;
  wordsJudged: number;
  wordsInside: number | null;
  observedPasses: number;
  evaluationPaired: number | null;
  /** Each gate's verdict — for the gated group only. */
  clears: { landed: boolean; words: boolean; evaluation: boolean } | null;
  /** Why a group is reported, not gated. */
  notGated: string | null;
  outcomes: Record<string, number>;
}

/** group → place (the airport, or "all") → stratum ("straight-in", "vectored", "all"). */
export type TrainingGateTable = Record<string, Record<string, Record<string, TrainingGateCell>>>;

export interface TrainingExecutorOverlay {
  overlayId: string;
  airport: string;
  base: TrainingOverlayBase;
  executor: { specSha256: string; wordClock: string; cycleS: number; params: Array<{ name: string; value: number | string }> };
  replay: { split: string; writtenUtc: string; gateShare: number; drawn: Record<string, unknown> };
  gate: TrainingGateTable;
  /** The base set's flights, in its order. */
  flights: TrainingExecutorFlight[];
}

/** The prior at one column of one flight, predicted step by predicted step (from `firstPredictedRow`). */
export interface TrainingPriorColumn {
  /** How many words are ranked at each step (fewer than asked when the column has fewer values). */
  k: number;
  /** The probability that a word is said at the step (1 at the first predicted step, which says every column). */
  changeP: number[];
  /** Row-major, `k` per step: the most likely words GIVEN that one is said, and their probabilities. */
  words: number[];
  wordsP: number[];
  /** The probability of what the truth sentence says at the step (a word, or unchanged; at the first predicted step
   *  the word in force there). */
  truthP: number[];
}

export interface TrainingPriorFlight {
  flightKey: string;
  datasetId: string;
  rows: number;
  /** The first step the prior speaks at; the rows before it are only observed and carry no prediction. */
  firstPredictedRow: number;
  /** Per predicted step. */
  nllPerStep: number;
  columnNllPerStep: number[];
  /** In `TRAINING_COLUMNS` order. */
  columns: TrainingPriorColumn[];
}

export interface TrainingPriorColumnReadout {
  nllPerStep: number;
  changeSteps: number;
  /** The first predicted step's word: how often the prior's most likely one is the truth's. */
  firstStepTop1: number;
  /** After the first predicted step, where the truth says a word; null for a column that never changes there. */
  changeProbabilityWhereChanged: number | null;
  top1GivenChange: number | null;
  top5GivenChange: number | null;
  falseChangeShareWhereKept: number;
}

/** The first predicted step's runway: right, in the right landing direction, and right within it. */
export interface TrainingPriorRunwayReadout {
  top1: number;
  direction: number;
  sideGivenDirection: number | null;
}

export interface TrainingPriorReadout {
  split: string;
  steps: number;
  bestEpoch: number;
  model: { nllPerStep: number; perplexityPerStep: number; perColumn: Record<TrainingColumn, TrainingPriorColumnReadout> };
  /** Negative log-likelihood per step per column, and `all`. */
  baselines: { repeat: Record<TrainingColumn | "all", number>; previousWord: Record<TrainingColumn | "all", number> };
  /** The prior's first-step runway beside each airport's own runway frequency and the causal rules (B0, B1, B3). */
  firstStepRunway: {
    model: TrainingPriorRunwayReadout;
    airportFrequency: TrainingPriorRunwayReadout;
    rules: Record<string, TrainingPriorRunwayReadout>;
  };
}

export interface TrainingPriorOverlay {
  overlayId: string;
  airport: string;
  base: TrainingOverlayBase;
  prior: { checkpointSha256: string; parameters: number; model: { dModel: number; layers: number; heads: number }; method: string };
  readout: TrainingPriorReadout;
  flights: TrainingPriorFlight[];
}

/** What the views draw for the selected flight: its overlay's own data, and the flight's entry. */
export interface TrainingExecutorView {
  overlay: TrainingExecutorOverlay;
  flight: TrainingExecutorFlight;
}

export interface TrainingPriorView {
  overlay: TrainingPriorOverlay;
  flight: TrainingPriorFlight;
}

// ── reading one step ─────────────────────────────────────────────────────────

/** The executor's verdict on the word at (row, column) of the sentence, if the flight was flown. */
export function executorWordAt(flight: TrainingExecutorFlight, row: number, column: TrainingColumn): TrainingExecutorWord | null {
  const index = TRAINING_COLUMNS.indexOf(column);
  return flight.words.find((word) => word.row === row && word.column === index) ?? null;
}

/** What the prior gives one column at one step: a word's probability, the ranked words, the truth's probability;
 *  null at a step before the first predicted one (observed only). */
export function priorStep(flight: TrainingPriorFlight, column: TrainingColumn, row: number) {
  if (row < flight.firstPredictedRow) return null;
  const at = row - flight.firstPredictedRow;
  const values = flight.columns[TRAINING_COLUMNS.indexOf(column)];
  const ranked = Array.from({ length: values.k }, (_, rank) => ({
    value: values.words[at * values.k + rank],
    p: values.wordsP[at * values.k + rank],
  }));
  return { changeP: values.changeP[at], truthP: values.truthP[at], ranked };
}

/** The flight's words the executor judged, and how many of them it flew inside their envelopes. */
export function executorWordCounts(flight: TrainingExecutorFlight) {
  const statuses = flight.words.map((word) => word.status);
  return {
    inside: statuses.filter((status) => status === "inside").length,
    outside: statuses.filter((status) => status === "outside").length,
    notJudged: statuses.filter((status) => status === "not judged").length,
    notReached: statuses.filter((status) => status === "not reached").length,
    superseded: statuses.filter((status) => status === "superseded").length,
  };
}

// ── small checkers ───────────────────────────────────────────────────────────

class Refusal extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

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
    const value = this.source[key];
    if (typeof value !== "string" || value.length === 0) this.fail(`${key} is ${JSON.stringify(value)}, not a non-empty string`);
    return value;
  }

  nullableString(key: string): string | null {
    return this.source[key] === null ? null : this.string(key);
  }

  number(key: string): number {
    const value = this.source[key];
    if (typeof value !== "number" || !Number.isFinite(value)) this.fail(`${key} is ${JSON.stringify(value)}, not a number`);
    return value;
  }

  nullableNumber(key: string): number | null {
    return this.source[key] === null ? null : this.number(key);
  }

  integer(key: string, low: number, high: number): number {
    const value = this.number(key);
    if (!Number.isInteger(value) || value < low || value > high) this.fail(`${key} is ${value}, not a whole number in ${low}…${high}`);
    return value;
  }

  nullableInteger(key: string, low: number, high: number): number | null {
    return this.source[key] === null ? null : this.integer(key, low, high);
  }

  boolean(key: string): boolean {
    const value = this.source[key];
    if (typeof value !== "boolean") this.fail(`${key} is ${JSON.stringify(value)}, not true/false`);
    return value;
  }

  nullableBoolean(key: string): boolean | null {
    return this.source[key] === null ? null : this.boolean(key);
  }

  numbers(key: string, length?: number): number[] {
    const value = this.source[key];
    if (!Array.isArray(value) || !value.every((item) => typeof item === "number" && Number.isFinite(item))) {
      this.fail(`${key} is missing or not a list of numbers`);
    }
    if (length !== undefined && value.length !== length) this.fail(`${key} has ${value.length} values, expected ${length}`);
    return value as number[];
  }

  nullableNumbers(key: string): number[] | null {
    return this.source[key] === null ? null : this.numbers(key);
  }

  flags(key: string, length: number): boolean[] {
    return this.numbers(key, length).map((value, index) => {
      if (value !== 0 && value !== 1) this.fail(`${key}[${index}] is ${value}, not 0/1`);
      return value === 1;
    });
  }

  /** Probabilities: numbers in [0, 1]. */
  probabilities(key: string, length: number): number[] {
    const values = this.numbers(key, length);
    const outside = values.findIndex((value) => value < 0 || value > 1);
    if (outside >= 0) this.fail(`${key}[${outside}] is ${values[outside]}, not a probability`);
    return values;
  }

  list(key: string): unknown[] {
    const value = this.source[key];
    if (!Array.isArray(value)) this.fail(`${key} is missing or not a list`);
    return value;
  }

  record<T>(key: string, read: (value: unknown, where: string) => T): Record<string, T> {
    return recordOf(this.source[key], this.at(key), read);
  }
}

/** An object whose every value is read by ``read``, keyed as it is. */
function recordOf<T>(value: unknown, where: string, read: (value: unknown, where: string) => T): Record<string, T> {
  if (!isRecord(value)) throw new Refusal(`${where} is not an object`);
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, read(item, `${where}.${key}`)]));
}

function asNumber(value: unknown, where: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Refusal(`${where} is ${JSON.stringify(value)}, not a number`);
  return value;
}

function attempt<T>(read: () => T): Parsed<T> {
  try {
    return { ok: true, value: read() };
  } catch (error) {
    if (error instanceof Refusal) return { ok: false, problem: error.message };
    throw error;
  }
}

// ── the manifest ─────────────────────────────────────────────────────────────

function parseEntry(raw: unknown, position: number): TrainingOverlayEntry {
  const id = isRecord(raw) && typeof raw.id === "string" && raw.id ? raw.id : `overlays[${position}]`;
  const entry = Reader.of(raw, `overlay ${id}`);
  const kind = entry.string("kind");
  if (!(TRAINING_OVERLAY_KINDS as readonly string[]).includes(kind)) {
    entry.fail(`kind is ${JSON.stringify(kind)}, expected one of ${TRAINING_OVERLAY_KINDS.join(", ")}`);
  }
  return {
    id: entry.string("id"),
    kind: kind as TrainingOverlayKind,
    base: entry.string("base"),
    baseSampleSha256: entry.string("baseSampleSha256"),
    title: entry.string("title"),
    file: entry.string("file"),
    flights: entry.integer("flights", 0, Number.MAX_SAFE_INTEGER),
  };
}

/** Parse the overlays manifest. A bad entry is rejected on its own; only a manifest that is not one fails the call. */
export function parseTrainingOverlays(raw: unknown): Parsed<TrainingOverlays> {
  if (!isRecord(raw)) return { ok: false, problem: "the overlays manifest is not an object" };
  if (raw.schema !== TRAINING_OVERLAYS_SCHEMA) {
    return { ok: false, problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_OVERLAYS_SCHEMA)}` };
  }
  if (typeof raw.airport !== "string" || !raw.airport) return { ok: false, problem: "airport is missing" };
  if (!Array.isArray(raw.overlays)) return { ok: false, problem: "overlays is not an array" };
  const overlays: TrainingOverlayEntry[] = [];
  const rejected: Array<{ id: string; problem: string }> = [];
  raw.overlays.forEach((item, position) => {
    try {
      overlays.push(parseEntry(item, position));
    } catch (error) {
      if (!(error instanceof Refusal)) throw error;
      const id = isRecord(item) && typeof item.id === "string" && item.id ? item.id : `overlays[${position}]`;
      rejected.push({ id, problem: error.message });
    }
  });
  return { ok: true, value: { airport: raw.airport, overlays, rejected } };
}

/** The overlays of one kind drawn over a set, in the manifest's order (the latest listed last). */
export function trainingOverlaysOf(overlays: TrainingOverlays, setId: string, kind: TrainingOverlayKind): TrainingOverlayEntry[] {
  return overlays.overlays.filter((entry) => entry.base === setId && entry.kind === kind);
}

// ── the binding ──────────────────────────────────────────────────────────────

/** The payload's own account of its set, against the sample the panel loaded. */
function parseBase(reader: Reader, sample: TrainingSample): TrainingOverlayBase {
  const base = reader.child("base");
  const found = {
    setId: base.string("setId"),
    sampleWrittenUtc: base.string("sampleWrittenUtc"),
    sampleSha256: base.string("sampleSha256"),
    specSha256: base.string("specSha256"),
  };
  if (found.specSha256 !== TRAINING_SPEC_SHA256) base.fail(`specSha256 is ${found.specSha256.slice(0, 12)}, not ${TRAINING_SPEC_SHA256.slice(0, 12)}`);
  if (found.setId !== sample.setId || found.sampleWrittenUtc !== sample.writtenUtc) {
    base.fail(
      `drawn over set ${found.setId} as written ${found.sampleWrittenUtc}, but the loaded sample is ${sample.setId} as ` +
      `written ${sample.writtenUtc}: the set was re-exported after the overlay`,
    );
  }
  return found;
}

/** The payload's flights, one per flight of the sample, in its order. */
function eachFlight<T>(reader: Reader, sample: TrainingSample, read: (item: Reader, flight: TrainingFlight) => T): T[] {
  const list = reader.list("flights");
  if (list.length !== sample.flights.length) reader.fail(`holds ${list.length} flights, the set ${sample.flights.length}`);
  return list.map((raw, index) => {
    const flight = sample.flights[index];
    const item = Reader.of(raw, `flight ${flight.flightKey}`);
    if (item.string("flightKey") !== flight.flightKey || item.string("datasetId") !== flight.datasetId) {
      item.fail(`is ${item.raw("flightKey")}, but the set's flight ${index} is ${flight.flightKey}`);
    }
    return read(item, flight);
  });
}

// ── the executor ─────────────────────────────────────────────────────────────

function parseTrack(reader: Reader): TrainingExecutorTrack {
  const tS = reader.numbers("tS");
  if (tS.length < 1 || tS[0] !== 0) reader.fail("does not start at 0 s");
  if (tS.some((value, index) => index > 0 && value <= tS[index - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  return {
    tS, eM: reader.numbers("eM", n), nM: reader.numbers("nM", n), lon: reader.numbers("lon", n), lat: reader.numbers("lat", n),
    altitudeM: reader.numbers("altitudeM", n), altitudeHaeM: reader.numbers("altitudeHaeM", n),
    groundSpeedMps: reader.numbers("groundSpeedMps", n), trackDeg: reader.numbers("trackDeg", n),
    distanceM: reader.numbers("distanceM", n),
  };
}

/** A heading word's band on the flown rows: its bookkeeping (`headingBandProblem`), within the rows the judge read. */
function parseExecutorBand(
  word: Reader, flownRow: number, targetDeg: number, vocabulary: TrainingVocabulary, judgedRows: number,
): TrainingHeadingBand {
  const band = word.child("heading");
  const firstRow = band.integer("firstRow", 0, Number.MAX_SAFE_INTEGER);
  const stopRow = band.integer("stopRow", firstRow, Number.MAX_SAFE_INTEGER);
  const bandDeg = band.numbers("bandDeg", 2);
  const parsed: TrainingHeadingBand = {
    firstRow, stopRow, targetOnTrackDeg: band.number("targetOnTrackDeg"), bandDeg: [bandDeg[0], bandDeg[1]],
    inside: band.flags("inside", stopRow - firstRow),
  };
  // the judge's rows end by the flown track it read, never past it
  const problem = headingBandProblem(parsed, flownRow, targetDeg, vocabulary, judgedRows);
  if (problem !== null) band.fail(problem);
  return parsed;
}

function parseExecutorWords(
  reader: Reader, flight: TrainingFlight, vocabulary: TrainingVocabulary, judgedRows: number | null,
): TrainingExecutorWord[] {
  const list = reader.list("words");
  const events = flight.words.events;
  if (list.length !== events.length) reader.fail(`judges ${list.length} words, the sentence says ${events.length}`);
  return list.map((raw, index) => {
    // typed, so that a `word.fail(...)` statement narrows what it guards
    const word: Reader = Reader.of(raw, reader.at(`words[${index}]`));
    const event = events[index];
    if (word.number("row") !== event.row || word.number("column") !== event.column || word.number("value") !== event.value) {
      word.fail(`is (${word.raw("row")}, ${word.raw("column")}, ${word.raw("value")}), but the sentence's word ${index} is ` +
        `(${event.row}, ${event.column}, ${event.value})`);
    }
    const status = word.string("status");
    if (!(TRAINING_EXECUTOR_STATUSES as readonly string[]).includes(status)) {
      word.fail(`status is ${status}, not one of ${TRAINING_EXECUTOR_STATUSES.join(", ")}`);
    }
    const checks = word.list("checks").map((check, position) => {
      const item = Reader.of(check, word.at(`checks[${position}]`));
      return {
        name: item.string("name"), ok: item.boolean("ok"),
        inside: item.nullableInteger("inside", 0, Number.MAX_SAFE_INTEGER), rows: item.nullableInteger("rows", 0, Number.MAX_SAFE_INTEGER),
      };
    });
    // a judged word carries its checks and they decide it; any other status says why instead
    const judged = status === "inside" || status === "outside";
    if (judged && (checks.length === 0 || (status === "inside") !== checks.every((check) => check.ok))) {
      word.fail(`is ${status}, but its checks say ${checks.map((check) => `${check.name} ${check.ok}`).join(", ") || "nothing"}`);
    }
    const reason = word.nullableString("reason");
    if (!judged && reason === null) word.fail(`is ${status} and says no reason`);
    const flownRow = word.nullableInteger("flownRow", 0, Number.MAX_SAFE_INTEGER);
    // A HEADING WORD THE JUDGE JUDGED — told on a step of the flown track it read — carries its band on the flown rows,
    // and no other word does: judged when it has rows (a check counting exactly its flags), not judged when it has none
    // (the lead carries them past the clearance, the capture or the track's end) — unless the executor left it to
    // intercept the final on its own, which fails it whatever its rows.
    const banded = event.column === TRAINING_COLUMN_INDEX.heading && flownRow !== null && judgedRows !== null
      && flownRow < judgedRows;
    if ((word.raw("heading") !== null) !== banded) {
      word.fail(banded ? "is a heading word the judge judged on the flown track, and carries no band"
        : "carries a heading band, but it is not a heading word the judge judged on a flown track");
    }
    let heading: TrainingHeadingBand | null = null;
    if (banded) {
      heading = parseExecutorBand(word, flownRow!, flight.envelopes.heading.find((item) => item.row === event.row)!.targetDeg,
        vocabulary, judgedRows!);
      const rows = heading.inside.length;
      const counted = heading.inside.filter(Boolean).length;
      if (rows > 0 && !checks.some((check) => check.rows === rows && check.inside === counted)) {
        word.fail(`its band counts ${counted} of ${rows} rows inside, and no check says so`);
      }
      if (rows === 0 && status !== "not judged" && status !== "outside") word.fail(`is ${status} with no row judged`);
      if (rows > 0 && !judged) word.fail(`is ${status}, but ${rows} of its rows were judged`);
    }
    return {
      row: event.row, column: event.column, value: event.value, status: status as TrainingExecutorStatus,
      flownRow, heading, checks, reason,
    };
  });
}

function parseExecutorFlight(item: Reader, flight: TrainingFlight, vocabulary: TrainingVocabulary): TrainingExecutorFlight {
  const flown = item.boolean("flown");
  const base = { flightKey: flight.flightKey, datasetId: flight.datasetId, group: item.string("group"), flown };
  if (!flown) {
    if (item.list("words").length !== 0 || item.raw("track") !== null || item.raw("outcome") !== null
        || item.raw("judgedTrackDeg") !== null) {
      item.fail("is not flown, yet carries a track, an outcome or words");
    }
    return {
      ...base, outcome: null, flewTheSentence: null, endS: null, crossing: null, refused: null, evaluation: null,
      alignment: null, limits: null, counts: null, track: null, judgedTrackDeg: null, words: [],
    };
  }
  const crossing = item.nullableChild("crossing");
  const evaluation = item.child("evaluation");
  const alignment = item.child("alignment");
  const limits = item.child("limits");
  const counts = item.child("counts");
  const track = parseTrack(item.child("track"));
  const refused = item.nullableString("refused");
  const outcome = item.string("outcome");
  // the judge's reading of the flown track is drawn exactly when the labeller's gate let it through and the dynamics
  // did not fail inside it; its step k is the exported track's point k
  const judgedTrackDeg = item.nullableNumbers("judgedTrackDeg");
  const drawn = refused === null && outcome !== "dynamics_failure";
  if ((judgedTrackDeg !== null) !== drawn) {
    item.fail(`judgedTrackDeg is ${judgedTrackDeg === null ? "absent" : "given"} for a flight ${refused !== null
      ? "whose flown track the gate refused" : outcome === "dynamics_failure" ? "whose dynamics failed" : "judged on its flown track"}`);
  }
  if (judgedTrackDeg !== null) {
    const misplaced = judgedTrackDeg.findIndex((_, step) => !(Math.abs(track.tS[step] - step * vocabulary.stepS) <= 1e-3));
    if (misplaced >= 0) {
      item.fail(`judgedTrackDeg's step ${misplaced} is not the flown track's point ${misplaced} (${track.tS[misplaced]} s)`);
    }
  }
  const words = parseExecutorWords(item, flight, vocabulary, judgedTrackDeg === null ? null : judgedTrackDeg.length);
  const judged = words.filter((word) => word.status === "inside" || word.status === "outside").length;
  // the judge counts the heading word left to intercept the final on its own once more when its band also had rows
  // judged: the one word with two checks, its band's and that one
  const twice = words.filter((word) => word.column === TRAINING_COLUMN_INDEX.heading && word.checks.length === 2).length;
  const wordsJudged = counts.integer("wordsJudged", 0, Number.MAX_SAFE_INTEGER);
  const wordsInside = counts.integer("wordsInside", 0, wordsJudged);
  if (wordsJudged !== judged + twice) {
    counts.fail(`says ${wordsJudged} words judged, but ${judged} words carry a verdict${twice ? ` (one counted twice)` : ""}`);
  }
  return {
    ...base,
    outcome,
    flewTheSentence: item.boolean("flewTheSentence"),
    endS: item.number("endS"),
    crossing: crossing === null ? null : { crossM: crossing.number("crossM"), heightM: crossing.number("heightM"), atS: crossing.number("atS") },
    refused,
    evaluation: { replay: evaluation.string("replay"), observed: evaluation.string("observed") },
    alignment: {
      meanHorizontalDistanceM: alignment.number("meanHorizontalDistanceM"),
      meanVerticalDistanceM: alignment.number("meanVerticalDistanceM"),
      landingTimeMinusObservedS: alignment.nullableNumber("landingTimeMinusObservedS"),
    },
    limits: { cycles: limits.integer("cycles", 0, Number.MAX_SAFE_INTEGER), bound: limits.record("bound", asNumber) },
    counts: { wordsJudged, wordsInside, headingWordsNotJudged: counts.integer("headingWordsNotJudged", 0, Number.MAX_SAFE_INTEGER) },
    track,
    judgedTrackDeg,
    words,
  };
}

function parseGateCell(value: unknown, where: string): TrainingGateCell {
  const cell = Reader.of(value, where);
  const clears = cell.nullableChild("clears");
  const notGated = cell.nullableString("notGated");
  if ((clears === null) === (notGated === null)) cell.fail("a cell is gated (clears) or says why not (notGated), one of the two");
  return {
    flights: cell.integer("flights", 0, Number.MAX_SAFE_INTEGER),
    landed: cell.nullableNumber("landed"),
    wordsJudged: cell.integer("wordsJudged", 0, Number.MAX_SAFE_INTEGER),
    wordsInside: cell.nullableNumber("wordsInside"),
    observedPasses: cell.integer("observedPasses", 0, Number.MAX_SAFE_INTEGER),
    evaluationPaired: cell.nullableNumber("evaluationPaired"),
    clears: clears === null ? null : { landed: clears.boolean("landed"), words: clears.boolean("words"), evaluation: clears.boolean("evaluation") },
    notGated,
    outcomes: cell.record("outcomes", asNumber),
  };
}

/** Parse an executor overlay against the sample it is drawn over — all or nothing. */
export function parseTrainingExecutorOverlay(raw: unknown, sample: TrainingSample): Parsed<TrainingExecutorOverlay> {
  if (!isRecord(raw)) return { ok: false, problem: "the executor overlay is not an object" };
  if (raw.schema !== TRAINING_EXECUTOR_SCHEMA) {
    return { ok: false, problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_EXECUTOR_SCHEMA)}` };
  }
  return attempt(() => {
    const overlay = new Reader(raw, "executor overlay");
    const executor = overlay.child("executor");
    const replay = overlay.child("replay");
    // group → place → stratum → cell
    const gate = overlay.record("gate", (group, where) =>
      recordOf(group, where, (place, placeWhere) => recordOf(place, placeWhere, parseGateCell)));
    return {
      overlayId: overlay.string("overlayId"),
      airport: overlay.string("airport"),
      base: parseBase(overlay, sample),
      executor: {
        specSha256: executor.string("specSha256"), wordClock: executor.string("wordClock"), cycleS: executor.number("cycleS"),
        params: executor.list("params").map((row, index) => {
          const param = Reader.of(row, executor.at(`params[${index}]`));
          const value = param.raw("value");
          if (typeof value !== "number" && typeof value !== "string") param.fail(`value is ${JSON.stringify(value)}`);
          return { name: param.string("name"), value: value as number | string };
        }),
      },
      replay: { split: replay.string("split"), writtenUtc: replay.string("writtenUtc"), gateShare: replay.number("gateShare"),
                drawn: replay.record("drawn", (value) => value) },
      gate,
      flights: eachFlight(overlay, sample, (item, flight) => parseExecutorFlight(item, flight, sample.vocabulary)),
    };
  });
}

// ── the prior ────────────────────────────────────────────────────────────────

function parsePriorFlight(item: Reader, flight: TrainingFlight, sample: TrainingSample): TrainingPriorFlight {
  const rows = item.integer("rows", flight.rows, flight.rows);
  const firstPredictedRow = item.integer("firstPredictedRow", 0, rows - 1);
  const steps = rows - firstPredictedRow;
  const list = item.list("columns");
  if (list.length !== TRAINING_COLUMNS.length) item.fail(`holds ${list.length} columns, expected ${TRAINING_COLUMNS.length}`);
  const columns = TRAINING_COLUMNS.map((name, index) => {
    const column = Reader.of(list[index], item.at(`columns[${index}] (${name})`));
    const values = trainingClassCount(sample.vocabulary, sample.candidates, name);
    const k = column.integer("k", 1, values);
    const words = column.numbers("words", steps * k);
    const wrong = words.findIndex((value) => !Number.isInteger(value) || value < 0 || value >= values);
    if (wrong >= 0) column.fail(`words[${wrong}] is ${words[wrong]}, not one of the column's ${values} values`);
    return {
      k, words, wordsP: column.probabilities("wordsP", steps * k), changeP: column.probabilities("changeP", steps),
      truthP: column.probabilities("truthP", steps),
    };
  });
  return {
    flightKey: flight.flightKey, datasetId: flight.datasetId, rows, firstPredictedRow,
    nllPerStep: item.number("nllPerStep"), columnNllPerStep: item.numbers("columnNllPerStep", TRAINING_COLUMNS.length), columns,
  };
}

function parseColumnScores<T>(reader: Reader, key: string, read: (value: unknown, where: string) => T, withAll: boolean) {
  const scores = reader.record(key, read);
  const expected = withAll ? [...TRAINING_COLUMNS, "all"] : [...TRAINING_COLUMNS];
  const missing = expected.filter((name) => !(name in scores));
  if (missing.length) reader.fail(`${key} has no ${missing.join(", ")}`);
  return scores;
}

function parseReadout(reader: Reader): TrainingPriorReadout {
  const model = reader.child("model");
  const baselines = reader.child("baselines");
  const perColumn = parseColumnScores(model, "perColumn", (value, where) => {
    const column = Reader.of(value, where);
    return {
      nllPerStep: column.number("nllPerStep"), changeSteps: column.integer("changeSteps", 0, Number.MAX_SAFE_INTEGER),
      firstStepTop1: column.number("firstStepTop1"),
      changeProbabilityWhereChanged: column.nullableNumber("changeProbabilityWhereChanged"),
      top1GivenChange: column.nullableNumber("top1GivenChange"), top5GivenChange: column.nullableNumber("top5GivenChange"),
      falseChangeShareWhereKept: column.number("falseChangeShareWhereKept"),
    };
  }, false) as Record<TrainingColumn, TrainingPriorColumnReadout>;
  const runway = reader.child("firstStepRunway");
  const readRunway = (value: unknown, where: string): TrainingPriorRunwayReadout => {
    const part = Reader.of(value, where);
    return { top1: part.number("top1"), direction: part.number("direction"), sideGivenDirection: part.nullableNumber("sideGivenDirection") };
  };
  return {
    split: reader.string("split"),
    steps: reader.integer("steps", 1, Number.MAX_SAFE_INTEGER),
    bestEpoch: reader.integer("bestEpoch", 1, Number.MAX_SAFE_INTEGER),
    model: { nllPerStep: model.number("nllPerStep"), perplexityPerStep: model.number("perplexityPerStep"), perColumn },
    baselines: {
      repeat: parseColumnScores(baselines, "repeat", asNumber, true) as Record<TrainingColumn | "all", number>,
      previousWord: parseColumnScores(baselines, "previousWord", asNumber, true) as Record<TrainingColumn | "all", number>,
    },
    firstStepRunway: {
      model: readRunway(runway.raw("model"), runway.at("model")),
      airportFrequency: readRunway(runway.raw("airportFrequency"), runway.at("airportFrequency")),
      rules: runway.record("rules", readRunway),
    },
  };
}

/** Parse a prior overlay against the sample it is drawn over — all or nothing. */
export function parseTrainingPriorOverlay(raw: unknown, sample: TrainingSample): Parsed<TrainingPriorOverlay> {
  if (!isRecord(raw)) return { ok: false, problem: "the prior overlay is not an object" };
  if (raw.schema !== TRAINING_PRIOR_SCHEMA) {
    return { ok: false, problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_PRIOR_SCHEMA)}` };
  }
  return attempt(() => {
    const overlay = new Reader(raw, "prior overlay");
    const columns = overlay.list("columns");
    if (columns.join(",") !== TRAINING_COLUMNS.join(",")) {
      overlay.fail(`columns are [${columns.join(", ")}], expected [${TRAINING_COLUMNS.join(", ")}] in that order`);
    }
    const prior = overlay.child("prior");
    const model = prior.child("model");
    return {
      overlayId: overlay.string("overlayId"),
      airport: overlay.string("airport"),
      base: parseBase(overlay, sample),
      prior: {
        checkpointSha256: prior.string("checkpointSha256"), parameters: prior.integer("parameters", 1, Number.MAX_SAFE_INTEGER),
        model: { dModel: model.number("dModel"), layers: model.number("layers"), heads: model.number("heads") },
        method: prior.string("method"),
      },
      readout: parseReadout(overlay.child("readout")),
      flights: eachFlight(overlay, sample, (item, flight) => parsePriorFlight(item, flight, sample)),
    };
  });
}

/** The value the truth sentence gives a column at a step: its word, or `TRAINING_UNCHANGED`. */
export function truthAt(flight: TrainingFlight, column: TrainingColumn, row: number): number {
  const index = TRAINING_COLUMNS.indexOf(column);
  return flight.words.events.find((event) => event.row === row && event.column === index)?.value ?? TRAINING_UNCHANGED;
}

// ── where the files live ─────────────────────────────────────────────────────

export function trainingOverlaysPath(airportCode: string): string {
  return `${trainingDirectory(airportCode)}/overlays.json`;
}

export function trainingOverlayPath(airportCode: string, file: string): string {
  return `${trainingDirectory(airportCode)}/${file}`;
}

export async function fetchTrainingOverlays(airportCode: string): Promise<Parsed<TrainingOverlays>> {
  return parseTrainingOverlays(await fetchJson<unknown>(trainingOverlaysPath(airportCode)));
}

export async function fetchTrainingExecutorOverlay(
  airportCode: string, entry: TrainingOverlayEntry, sample: TrainingSample,
): Promise<Parsed<TrainingExecutorOverlay>> {
  return parseTrainingExecutorOverlay(await fetchJson<unknown>(trainingOverlayPath(airportCode, entry.file)), sample);
}

export async function fetchTrainingPriorOverlay(
  airportCode: string, entry: TrainingOverlayEntry, sample: TrainingSample,
): Promise<Parsed<TrainingPriorOverlay>> {
  return parseTrainingPriorOverlay(await fetchJson<unknown>(trainingOverlayPath(airportCode, entry.file)), sample);
}
