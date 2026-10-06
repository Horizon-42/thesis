/**
 * trainingWindowSample.ts
 * -----------------------
 * The Training view's data contract for STAGE C of the two-tier model: the airport's index of WINDOW sets and one set's
 * windows of recorded traffic. Written by `ts_transformer/experiments/post_training_export.py` (the files:
 * `ts_transformer/post/training_files.py`); design: post-training §8 C11, outline §6.
 *
 * A set's `flights` are stage A's heads of the windows' commanded flights (read by stage A's reader: `trainingSample.ts`).
 * A window names its commanded flight (`datasetId`), its kind (real, A: one aircraft inserted, D: the aircraft ahead moved,
 * B: the commanded aircraft's start moved — `movedStart`, its observed rows to the first predicted step as the start moved
 * them), the other aircraft on their records over the window (`traffic`), and for each round of the post-training
 * campaign the sentence that round's model said for the commanded aircraft, flown — its words, flown track and attitude,
 * its end (the judge's outcome, or a loss of separation: `TRAINING_LOST_SEPARATION`) and the window's end (`end`: the
 * reward, the loss with its other aircraft and the minimum it broke).
 *
 * THIS FILE COMPUTES NO WORD, NO SEPARATION AND NO REWARD: every number is the exporter's. The reader checks the file's
 * BOOKKEEPING — lengths, rows, that the events are the grid's words, that a round's sentence starts where the flight's
 * closed-loop sentence does, that a loss goes with a loss's end — and places times on the flight's clock.
 *
 * NO COMPATIBILITY. The index and sample schemas and the set kind are pinned below; a file that carries anything else is
 * refused by name. This reader never reads stage A's or stage B's index.
 *
 * SI units only. Heights in the files are MSL; the 3D scene's are ellipsoid heights (each aircraft's own runway's
 * `haeMinusMslM` added once, here).
 */

import { fetchJson } from "../utils/fetchJson";
import { attempt, parseManifest, Reader, type Parsed } from "./trainingReader";
import {
  parseCandidates,
  parseEvents,
  parseFlight,
  parseFlownBlock,
  parseFormats,
  parseGrid,
  parseVocabulary,
  wordUnreached,
  TRAINING_LOST_SEPARATION,
  TRAINING_OUTCOMES,
  TRAINING_READING_RULE,
  TRAINING_SPLITS,
  type TrainingCandidate,
  type TrainingClosedLoop,
  type TrainingCrossing,
  type TrainingEnvelopes,
  type TrainingEvent,
  type TrainingFlight,
  type TrainingFlownEnd,
  type TrainingSelection,
  type TrainingTrack,
  type TrainingVocabulary,
} from "./trainingSample";

/** MIRROR of the exporter's `INDEX_SCHEMA` (`ts_transformer/post/training_files.py`): the airport's index of window sets.
 *  A name changes with its file's shape, on both sides, in the same change. */
export const TRAINING_WINDOW_INDEX_SCHEMA = "aeroviz-training-window-index-v2";
/** MIRROR of `training_files.INDEX_FILE`. */
export const TRAINING_WINDOW_INDEX_FILE = "index_post_v2.json";
/** MIRROR of `training_files.SAMPLE_SCHEMA` (v2: a round's flown track unrounded, prior D127 followed for windows). */
export const TRAINING_WINDOW_SAMPLE_SCHEMA = "aeroviz-training-window-sample-v3";
/** MIRROR of `training_files.SET_KIND`. */
export const TRAINING_WINDOW_SET_KIND = "post-training-windows";
/** MIRROR of `post_training_export.START`: the round that names the model at the start of the campaign. */
export const TRAINING_WINDOW_START = "start";
/** MIRROR of `post.scene`'s window kinds (`REAL`, `INSERTED`, `LEADER_MOVED`, `MOVED_START`). */
export const TRAINING_WINDOW_KINDS = ["real", "A", "D", "B"] as const;
export type TrainingWindowKind = (typeof TRAINING_WINDOW_KINDS)[number];
/** MIRROR of `post_training_export.traffic_payload`'s roles: an aircraft on its record, window A's inserted one, window D's
 *  moved one. */
export const TRAINING_WINDOW_ROLES = ["recorded", "inserted", "moved"] as const;
export type TrainingWindowRole = (typeof TRAINING_WINDOW_ROLES)[number];

/** A round of a campaign: the model at its start, or the model after a round. */
export type TrainingWindowRound = number | typeof TRAINING_WINDOW_START;

export interface TrainingWindowModel {
  campaign: string;
  rounds: TrainingWindowRound[];
  rowIntervalS: number;
  mostGoArounds: number;
  procedureMasks: string;
}

export interface TrainingWindowCohort {
  split: string;
  perAirport: number;
  kinds: TrainingWindowKind[];
  seed: number;
  windows: number;
  /** The airport's real windows of the split that do not open inside a loss (D113), and how many were drawn. */
  pool: number;
  real: number;
  /** By kind: the windows a drawn real window admitted none of, or that opened inside a loss. */
  leftOut: Record<string, number>;
  drawnFrom: string;
}

export interface TrainingWindowSource {
  campaign: string;
  instructions: string;
  executor: string;
  smoke: boolean;
  campaignSmoke: boolean;
  git: { head: string; dirty: boolean };
}

export interface TrainingWindowSetEntry {
  id: string;
  title: string;
  file: string;
  windows: number;
  flights: number;
  formats: Record<string, string>;
  model: TrainingWindowModel;
  cohort: TrainingWindowCohort;
  source: TrainingWindowSource;
}

export interface TrainingWindowIndex {
  airport: string;
  sets: TrainingWindowSetEntry[];
  rejected: Array<{ id: string; problem: string }>;
}

/** The loss of separation that ended a window: the loop's Δ row (from the window's row 0) and the flight time it was
 *  found at, the other aircraft (its key in the window's traffic), the kind of minimum and the distances. */
export interface TrainingWindowLoss {
  step: number;
  timeS: number;
  other: string;
  kind: string;
  relation: string;
  requiredM: number;
  distanceM: number;
  verticalM: number;
  wakeKnown: boolean;
}

export interface TrainingWindowEnd {
  reward: number;
  loss: TrainingWindowLoss | null;
  speedMaskRows: number;
  faultySteps: number;
  lossReadsFault: boolean;
}

/** One round's sentence for the commanded aircraft, flown. Rows are the sentence's Δ rows from the first predicted step. */
export interface TrainingWindowSentence {
  round: TrainingWindowRound;
  words: number[][];
  events: TrainingEvent[];
  firstRow: number;
  startRow: number;
  flownFromRow: number;
  /** The block of the flown sentence (stage A's reader, `parseFlownBlock`). */
  outcome: TrainingFlownEnd;
  endCycle: number;
  crossing: TrainingCrossing | null;
  flown: TrainingTrack;
  envelopes: TrainingEnvelopes | null;
  timedOut: boolean;
  goArounds: number;
  end: TrainingWindowEnd;
}

/** Another aircraft of a window on its record's 2 s rows over the window, on the commanded flight's clock. */
export interface TrainingWindowTraffic {
  key: string;
  role: TrainingWindowRole;
  /** The shift of an inserted or moved aircraft (s); null for one on its record. */
  shiftS: number | null;
  runway: string;
  category: string | null;
  landingS: number;
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeMslM: number[];
  altitudeHaeM: number[];
}

/** Window B's moved observed rows: positions and heights on the flight's clock. */
export interface TrainingWindowMovedStart {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeMslM: number[];
  altitudeHaeM: number[];
}

export interface TrainingWindow {
  /** Its place in the set's windows (the live executor's request names it). */
  index: number;
  datasetId: string;
  head: TrainingFlight;
  kind: TrainingWindowKind;
  /** The flight time of the window's row 0 and of its first predicted step, on the commanded flight's clock. */
  row0S: number;
  firstStepS: number;
  startMove: { turnDeg: number; heightM: number; speedScale: number };
  moved: Array<[string, number]>;
  /** Window B: its observed rows from the window's row 0 to the first predicted step as its start moved them (the view
   *  draws them beside the head's observed track, which stays the recorded flight's: its open-loop reading is read on
   *  it); null for any other window (its start is the head's). */
  movedStart: TrainingWindowMovedStart | null;
  traffic: TrainingWindowTraffic[];
  rounds: TrainingWindowSentence[];
}

export interface TrainingWindowSample {
  setId: string;
  airport: string;
  executor: { cycleS: number };
  formats: Record<string, string>;
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidates: TrainingCandidate[];
  model: TrainingWindowModel;
  cohort: TrainingWindowCohort;
  source: TrainingWindowSource;
  flights: TrainingFlight[];
  windows: TrainingWindow[];
}

// ── the index ────────────────────────────────────────────────────────────────

function parseRound(value: unknown, where: Reader): TrainingWindowRound {
  if (value === TRAINING_WINDOW_START || (Number.isInteger(value) && (value as number) >= 0)) return value as TrainingWindowRound;
  return where.fail(`round ${JSON.stringify(value)} is neither "${TRAINING_WINDOW_START}" nor a round's number`);
}

function parseModel(reader: Reader): TrainingWindowModel {
  const rounds = reader.list("rounds").map((value) => parseRound(value, reader));
  if (rounds.length === 0) reader.fail("rounds is empty");
  if (new Set(rounds).size !== rounds.length) reader.fail("a round is listed twice");
  return {
    campaign: reader.string("campaign"), rounds, rowIntervalS: reader.number("rowIntervalS"),
    mostGoArounds: reader.count("mostGoArounds"), procedureMasks: reader.string("procedureMasks"),
  };
}

function parseCohort(reader: Reader): TrainingWindowCohort {
  const kinds = reader.strings("kinds").map((kind) => {
    if (!(TRAINING_WINDOW_KINDS as readonly string[]).includes(kind)) reader.fail(`kind ${kind} is none of ${TRAINING_WINDOW_KINDS}`);
    return kind as TrainingWindowKind;
  });
  const split = reader.string("split");
  if (!(TRAINING_SPLITS as readonly string[]).includes(split)) reader.fail(`split ${split} is none of ${TRAINING_SPLITS}`);
  return {
    split, perAirport: reader.count("perAirport", 1), kinds, seed: reader.number("seed"), windows: reader.count("windows"),
    pool: reader.count("pool"), real: reader.count("real"),
    leftOut: reader.record("leftOut", (value, where) => {
      if (!Number.isInteger(value) || (value as number) < 0) throw new Error(`${where} is not a count`);
      return value as number;
    }),
    drawnFrom: reader.string("drawnFrom"),
  };
}

function parseSource(reader: Reader): TrainingWindowSource {
  const git = reader.child("git");
  return {
    campaign: reader.string("campaign"), instructions: reader.string("instructions"), executor: reader.string("executor"),
    smoke: reader.boolean("smoke"), campaignSmoke: reader.boolean("campaignSmoke"),
    git: { head: git.string("head"), dirty: git.boolean("dirty") },
  };
}

function parseSetEntry(entry: Reader): TrainingWindowSetEntry {
  entry.oneOf("kind", [TRAINING_WINDOW_SET_KIND]);
  entry.oneOf("readingRule", [TRAINING_READING_RULE]);
  return {
    id: entry.string("id"), title: entry.string("title"), file: entry.string("file"), windows: entry.count("windows"),
    flights: entry.count("flights"), formats: parseFormats(entry), model: parseModel(entry.child("model")),
    cohort: parseCohort(entry.child("cohort")), source: parseSource(entry.child("source")),
  };
}

/** Parse the index. A bad entry is rejected on its own; a schema other than `TRAINING_WINDOW_INDEX_SCHEMA` is refused
 *  whole, naming the one found and the one expected. */
export function parseTrainingWindowIndex(raw: unknown): Parsed<TrainingWindowIndex> {
  const manifest = parseManifest(raw, {
    name: "index", schema: TRAINING_WINDOW_INDEX_SCHEMA, listKey: "sets", entryName: "set",
  }, parseSetEntry);
  if (!manifest.ok) return manifest;
  return { ok: true, value: { airport: manifest.value.airport, sets: manifest.value.entries, rejected: manifest.value.rejected } };
}

// ── one set ──────────────────────────────────────────────────────────────────

/** A loss as the file writes it, its time (from the window's row 0) put on the flight's clock (``row0S``). */
function parseLoss(reader: Reader, row0S: number): TrainingWindowLoss {
  return {
    step: reader.count("step", 1), timeS: row0S + reader.number("timeS"), other: reader.string("other"),
    kind: reader.string("kind"), relation: reader.string("relation"), requiredM: reader.number("requiredM"),
    distanceM: reader.number("distanceM"), verticalM: reader.number("verticalM"), wakeKnown: reader.boolean("wakeKnown"),
  };
}

function parseSentence(
  reader: Reader, head: TrainingFlight, model: TrainingWindowModel, candidates: TrainingCandidate[], vocabulary: TrainingVocabulary,
  cycleS: number, round: TrainingWindowRound,
): TrainingWindowSentence {
  const stepS = vocabulary.stepS;
  const closed = head.closedLoop[String(model.rowIntervalS)];
  const firstRow = reader.count("firstRow");
  const startRow = reader.count("startRow");
  const flownFromRow = reader.count("flownFromRow");
  if (firstRow !== closed.firstRow || startRow !== closed.startRow || flownFromRow !== closed.flownFromRow) {
    reader.fail(`starts at row ${firstRow} + ${startRow} (state row ${flownFromRow}), but the flight's closed-loop sentence starts at ` +
      `${closed.firstRow} + ${closed.startRow} (${closed.flownFromRow}): a window's sentences start at its first predicted step`);
  }
  const words = parseGrid(reader, "words");
  if (reader.count("rows", 1) !== words.length) reader.fail(`rows is ${reader.raw("rows")}, but words holds ${words.length}`);
  const events = parseEvents(reader, words, candidates, false);
  const block = parseFlownBlock(reader, [...TRAINING_OUTCOMES, TRAINING_LOST_SEPARATION], candidates, cycleS, stepS,
    firstRow + flownFromRow, head.haeMinusMslM, head.observed);
  const { outcome, crossing } = block;
  const end = reader.child("end");
  const loss = end.nullableChild("loss");
  if ((loss !== null) !== (outcome === TRAINING_LOST_SEPARATION)) {
    end.fail(`the window ended ${outcome} but its loss is ${loss === null ? "absent" : "given"}: a loss of separation and its end go together`);
  }
  if (loss !== null && crossing !== null) reader.fail("a window ended at a loss of separation crossed no threshold");
  const row0S = firstRow * stepS;
  return {
    round, words, events, firstRow, startRow, flownFromRow, outcome, endCycle: block.endCycle, crossing, flown: block.flown,
    envelopes: block.envelopes, timedOut: reader.boolean("timedOut"), goArounds: reader.count("goArounds"),
    end: {
      reward: end.number("reward"),
      loss: loss === null ? null : parseLoss(loss, row0S),
      speedMaskRows: end.count("speedMaskRows"), faultySteps: end.count("faultySteps"), lossReadsFault: end.boolean("lossReadsFault"),
    },
  };
}

function parseTraffic(reader: Reader, row0S: number, candidates: TrainingCandidate[]): TrainingWindowTraffic {
  const role = reader.oneOf("role", TRAINING_WINDOW_ROLES);
  const shiftS = reader.nullableNumber("shiftS");
  if ((shiftS === null) !== (role === "recorded")) reader.fail(`an aircraft ${role} has ${shiftS === null ? "no" : "a"} shift`);
  const runway = reader.string("runway");
  if (!candidates.some((candidate) => candidate.ident === runway)) reader.fail(`runway ${runway} is none of the candidates`);
  const times = reader.numbers("tS");
  const rows = times.length;
  const hae = reader.number("haeMinusMslM");
  const heights = reader.numbers("heightMslM", rows);
  return {
    key: reader.string("key"), role, shiftS, runway, category: reader.nullableString("category"),
    landingS: row0S + reader.number("landingS"), tS: times.map((t) => row0S + t), lon: reader.numbers("lonDeg", rows),
    lat: reader.numbers("latDeg", rows), altitudeMslM: heights, altitudeHaeM: heights.map((value) => value + hae),
  };
}

/** Window B's observed rows from the window's row 0 to its first predicted step as its start moved them: the written
 *  positions and heights, on the flight's clock (`TrainingWindowMovedStart`). */
function parseMovedStart(reader: Reader, head: TrainingFlight, firstRow: number, stepS: number): TrainingWindowMovedStart {
  const rows = reader.count("rows", 1);
  const heights = reader.numbers("heightMslM", rows);
  return {
    tS: Array.from({ length: rows }, (_, i) => (firstRow + i) * stepS), lat: reader.numbers("latDeg", rows),
    lon: reader.numbers("lonDeg", rows), altitudeMslM: heights, altitudeHaeM: heights.map((value) => value + head.haeMinusMslM),
  };
}

function parseWindow(
  reader: Reader, index: number, heads: Map<string, TrainingFlight>, model: TrainingWindowModel, candidates: TrainingCandidate[],
  vocabulary: TrainingVocabulary, cycleS: number,
): TrainingWindow {
  const datasetId = reader.string("datasetId");
  const head = heads.get(datasetId);
  if (head === undefined) return reader.fail(`the set holds no flight ${datasetId}`);
  const kind = reader.oneOf("kind", TRAINING_WINDOW_KINDS);
  const closed = head.closedLoop[String(model.rowIntervalS)];
  const row0S = closed.firstRow * vocabulary.stepS;
  const move = reader.child("startMove");
  const startMove = { turnDeg: move.number("turnDeg"), heightM: move.number("heightM"), speedScale: move.number("speedScale") };
  const isMoved = startMove.turnDeg !== 0 || startMove.heightM !== 0 || startMove.speedScale !== 1;
  if (isMoved !== (kind === "B")) reader.fail(`a window ${kind} with ${isMoved ? "a" : "no"} start move: window B, and only B, moves its start`);
  const moved = reader.list("moved").map((pair) => {
    if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || typeof pair[1] !== "number") {
      return reader.fail("a moved aircraft is [key, shift]");
    }
    return [pair[0], pair[1]] as [string, number];
  });
  const start = reader.nullableChild("movedStart");
  if ((start !== null) !== (kind === "B")) reader.fail(`a window ${kind} with ${start === null ? "no" : "a"} moved start`);
  const rounds = reader.children("rounds").map((item) => {
    const round = parseRound(item.raw("round"), item);
    return parseSentence(item, head, model, candidates, vocabulary, cycleS, round);
  });
  if (rounds.map((item) => item.round).join() !== model.rounds.join()) {
    reader.fail(`rounds ${rounds.map((item) => item.round).join(", ")}, the set's are ${model.rounds.join(", ")}`);
  }
  return {
    index, datasetId, head, kind, row0S, firstStepS: row0S + reader.number("firstStepS"), startMove, moved,
    movedStart: start === null ? null : parseMovedStart(start, head, closed.firstRow, vocabulary.stepS),
    traffic: reader.children("traffic").map((item) => parseTraffic(item, row0S, candidates)), rounds,
  };
}

/** Parse a set's sample. A schema other than `TRAINING_WINDOW_SAMPLE_SCHEMA` is refused whole, naming the one found and
 *  the one expected. */
export function parseTrainingWindowSample(raw: unknown): Parsed<TrainingWindowSample> {
  return attempt(() => {
    const sample = Reader.of(raw, "sample");
    sample.oneOf("schema", [TRAINING_WINDOW_SAMPLE_SCHEMA]);
    sample.oneOf("readingRule", [TRAINING_READING_RULE]);
    const vocabulary = parseVocabulary(sample.child("vocabulary"));
    const candidates = parseCandidates(sample);
    const frame = sample.child("airportFrame");
    const cycleS = sample.child("executor").number("cycleS");
    if (!(cycleS > 0)) sample.fail(`executor.cycleS is ${cycleS}, not a positive length`);
    if (!Number.isInteger(vocabulary.stepS / cycleS)) sample.fail(`a ${vocabulary.stepS} s row is not a whole number of ${cycleS} s cycles`);
    const model = parseModel(sample.child("model"));
    if (!vocabulary.rowIntervalsS.includes(model.rowIntervalS)) {
      sample.fail(`the windows are flown at Δ ${model.rowIntervalS} s, which is none of the vocabulary's [${vocabulary.rowIntervalsS.join(", ")}]`);
    }
    const cohort = parseCohort(sample.child("cohort"));
    const flights = sample.children("flights").map((flight) => parseFlight(flight, vocabulary, candidates, cycleS, [model.rowIntervalS], TRAINING_SPLITS));
    const heads = new Map(flights.map((flight) => [flight.datasetId, flight]));
    if (heads.size !== flights.length) sample.fail("two flights carry one dataset id");
    const windows = sample.children("windows").map((window, index) => parseWindow(window, index, heads, model, candidates, vocabulary, cycleS));
    if (windows.length !== cohort.windows) sample.fail(`cohort.windows is ${cohort.windows}, but the set holds ${windows.length}`);
    return {
      setId: sample.string("setId"), airport: sample.string("airport"), executor: { cycleS }, formats: parseFormats(sample),
      vocabulary, airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidates, model, cohort, source: parseSource(sample.child("source")), flights, windows,
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

export function trainingWindowDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

export function trainingWindowIndexPath(airportCode: string): string {
  return `${trainingWindowDirectory(airportCode)}/${TRAINING_WINDOW_INDEX_FILE}`;
}

/** The airport's index, refused unless it is the one asked for. */
export async function fetchTrainingWindowIndex(airportCode: string): Promise<Parsed<TrainingWindowIndex>> {
  const parsed = parseTrainingWindowIndex(await fetchJson<unknown>(trainingWindowIndexPath(airportCode)));
  if (parsed.ok && parsed.value.airport !== airportCode) {
    return { ok: false, problem: `${trainingWindowIndexPath(airportCode)} is ${parsed.value.airport}'s index, not ${airportCode}'s` };
  }
  return parsed;
}

/** A set's sample, refused unless it is the set and airport asked for. */
export async function fetchTrainingWindowSample(airportCode: string, file: string, setId: string): Promise<Parsed<TrainingWindowSample>> {
  const parsed = parseTrainingWindowSample(await fetchJson<unknown>(`${trainingWindowDirectory(airportCode)}/${file}`));
  if (parsed.ok && (parsed.value.airport !== airportCode || parsed.value.setId !== setId)) {
    return { ok: false, problem: `${file} holds set ${parsed.value.setId} of ${parsed.value.airport}, not ${setId} of ${airportCode}` };
  }
  return parsed;
}

/** A shift in time as a window shows it (D129): in days from a day, in hours from an hour, else in seconds — window A
 *  inserts a flight of another time, often days away. */
export function windowShiftText(seconds: number): string {
  const sign = seconds < 0 ? "−" : "+";
  const size = Math.abs(seconds);
  if (size >= 86_400) return `${sign}${(size / 86_400).toFixed(1)} d`;
  if (size >= 3_600) return `${sign}${(size / 3_600).toFixed(1)} h`;
  return `${sign}${size.toFixed(0)} s`;
}

// ── a round's sentence as the stage-A views read it ──────────────────────────

/** What a derived flight stands for: the set's window and the round. */
export interface TrainingWindowOrigin {
  sample: TrainingWindowSample;
  window: TrainingWindow;
  round: TrainingWindowRound;
  sentence: TrainingWindowSentence;
}

const origins = new WeakMap<TrainingFlight, TrainingWindowOrigin>();
const views = new WeakMap<TrainingWindow, Map<TrainingWindowRound, TrainingFlight>>();

/** The set's window and round a flight on screen was derived from; undefined for any other flight. */
export function trainingWindowOriginOf(flight: TrainingFlight): TrainingWindowOrigin | undefined {
  return origins.get(flight);
}

/** A round's sentence as stage A's closed-loop sentence at Δ, so the sentence bar, the read-back window and the 3D layers
 *  read it unchanged (as stage B's: `trainingPriorSample.ts`): its envelopes the judge's on its flown track (D135); the
 *  reading added no word;
 *  `notReached` counts the words said after the window's end (`wordUnreached`). */
function closedLoopOf(model: TrainingWindowModel, flight: TrainingFlight, sentence: TrainingWindowSentence): TrainingClosedLoop {
  const closed = flight.closedLoop[String(model.rowIntervalS)];
  const result = {
    rowIntervalS: model.rowIntervalS, firstRow: sentence.firstRow, startRow: sentence.startRow, flownFromRow: sentence.flownFromRow,
    startS: closed.startS, words: sentence.words, events: sentence.events,
    lateralM: sentence.words.map(() => null), verticalM: sentence.words.map(() => null), timedOut: sentence.timedOut,
    cycleS: closed.cycleS, flown: sentence.flown,
    replay: {
      outcome: sentence.outcome, endCycle: sentence.endCycle, crossing: sentence.crossing, flewTheSentence: true, notReached: 0,
      envelopes: sentence.envelopes,
    },
  };
  const notReached = sentence.events.filter((event) => wordUnreached(result, event.row)).length;
  return { ...result, replay: { ...result.replay, flewTheSentence: notReached === 0, notReached } };
}

/** The window's commanded flight as the views draw it with ``round``'s sentence read at the set's Δ (its observed track
 *  and open-loop reading the head's: window B's moved start is drawn by the window's own layer). Another window or round
 *  is another flight (its key says which), so the cursor and the live pick reset with it. Built once. */
export function trainingWindowFlightView(sample: TrainingWindowSample, window: TrainingWindow, round: TrainingWindowRound): TrainingFlight {
  const kept = views.get(window)?.get(round);
  if (kept !== undefined) return kept;
  const sentence = window.rounds.find((item) => item.round === round);
  if (sentence === undefined) throw new Error(`window ${window.index} has no round ${round}`);
  const key = String(sample.model.rowIntervalS);
  const view: TrainingFlight = {
    ...window.head, flightKey: `${window.head.flightKey}~window-${window.index}-round-${round}`,
    closedLoop: { [key]: closedLoopOf(sample.model, window.head, sentence) },
  };
  origins.set(view, { sample, window, round, sentence });
  const byRound = views.get(window) ?? new Map<TrainingWindowRound, TrainingFlight>();
  byRound.set(round, view);
  views.set(window, byRound);
  return view;
}

/** What the sentence bar, the read-back window and the 3D layers read: the set's vocabulary with its one Δ, the candidates
 *  and the derived flight. */
export function trainingWindowSelectionOf(sample: TrainingWindowSample, flight: TrainingFlight): TrainingSelection {
  return {
    airport: sample.airport, setId: sample.setId, vocabulary: { ...sample.vocabulary, rowIntervalsS: [sample.model.rowIntervalS] },
    candidates: sample.candidates, flight,
  };
}
