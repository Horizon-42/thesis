/**
 * trainingWindowSample.ts
 * -----------------------
 * The Training view's data contract for STAGE C of the two-tier model: the airport's index of WINDOW sets and one set's
 * windows of recorded traffic. Written by `ts_transformer/experiments/post_training_export.py` (the files:
 * `ts_transformer/post/training_files.py`); design: post-training §8 C11, outline §6.
 *
 * ONE WINDOW FORMAT FOR STAGES C AND D (frontend D156, §5.7): a window holds a LIST of commanded aircraft — stage C's
 * windows one, stage D's several — in the order they join it. A set's `flights` are stage A's heads of the windows'
 * commanded flights (read by stage A's reader: `trainingSample.ts`). A window holds its kind (real, A: one aircraft
 * inserted, D: the aircraft ahead moved, B: the commanded aircraft's start moved) and c (a compressed window's; none in
 * stage C), its commanded aircraft — each its flight (`datasetId`), its join offset from the window's row 0, its move in
 * time, its first predicted step, its start move (window B: `movedStart`, its observed rows to the first predicted step as
 * the start moved them) and, for each round of the campaign, the sentence that round's model said for it, flown (its
 * words, flown track and attitude, its outcome — the judge's, or `TRAINING_LOST_SEPARATION` —, its reward and the row
 * from which it is silent) —, the other aircraft on their records over the window (`traffic`), and each round's end: its
 * losses of separation, each with its two aircraft, the ones that answer for it, the minimum it broke and whether it
 * costs W.
 *
 * TWO CLOCKS. The window's own times — the traffic, the losses — are seconds from the window's row 0. Each commanded
 * aircraft's are its flight's (from its observed track's row 0, as stage A's views read a flight): `clockS` is its flight
 * time at the window's row 0, so a window time t is the aircraft's t + `clockS` (`onAircraftClock`).
 *
 * THIS FILE COMPUTES NO WORD, NO SEPARATION AND NO REWARD: every number is the exporter's. The reader checks the file's
 * BOOKKEEPING — lengths, rows, that the events are the grid's words, that a round's sentence starts where the flight's
 * closed-loop sentence does, that a loss's aircraft are the window's and an aircraft that ended at a loss answers for one
 * — and places each aircraft's times on its flight's clock.
 *
 * NO COMPATIBILITY. The index and sample schemas and the set kind are pinned below; a file that carries anything else is
 * refused by name. This reader never reads stage A's or stage B's index.
 *
 * SI units only. Heights in the files are MSL; the 3D scene's are ellipsoid heights (each aircraft's own runway's
 * `haeMinusMslM` added once, here).
 */

import { fetchJson } from "../utils/fetchJson";
import { attempt, parseManifest, Reader, type Parsed } from "./trainingReader";
import { parseProcedure, type TrainingProcedure } from "./trainingProcedure";
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

/** MIRROR of the exporter's `INDEX_SCHEMA` (`ts_transformer/post/training_files.py`): the airport's index of window sets,
 *  one format for stages C and D (the stage is the index file's). A name changes with its file's shape, on both sides, in
 *  the same change. */
export const TRAINING_WINDOW_INDEX_SCHEMA = "aeroviz-training-window-index-v4";
/** The stages whose sets are windows (frontend §5.7): one format, the stage the index FILE's. */
export const TRAINING_WINDOW_STAGES = ["C", "D"] as const;
export type TrainingWindowStage = (typeof TRAINING_WINDOW_STAGES)[number];
/** Each stage's index: MIRROR of `post/training_files.INDEX_FILE` (stage C's) and `MULTI_INDEX_FILE` (stage D's). */
export const TRAINING_WINDOW_INDEX_FILES: Record<TrainingWindowStage, string> = {
  C: "index_post_v4.json",
  D: "index_multi_v2.json",
};
/** MIRROR of `training_files.SAMPLE_SCHEMA` (v4: the window format of stages C and D, frontend D156; v5: a cohort drawn
 *  or listed, post-training D176). */
export const TRAINING_WINDOW_SAMPLE_SCHEMA = "aeroviz-training-window-sample-v5";
/** MIRROR of `training_files.SET_KIND`. */
export const TRAINING_WINDOW_SET_KIND = "training-windows";
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

/** The round of another campaign that a campaign starts from (post-training D162): its directory (repository-relative),
 *  the round and the checkpoint's SHA-256. */
export interface TrainingWindowStart {
  campaign: string;
  round: number;
  checkpointSha256: string;
}

export interface TrainingWindowModel {
  campaign: string;
  rounds: TrainingWindowRound[];
  rowIntervalS: number;
  mostGoArounds: number;
  procedureMasks: string;
  /** What the campaign starts from (the set's `model.settings.start`): null for the base (D29), else another campaign's
   *  round (D162). */
  start: TrainingWindowStart | null;
}

/** A set's windows (post-training D176, frontend §5.7): drawn by the export, or a window list's. */
export type TrainingWindowCohort = TrainingWindowDrawn | TrainingWindowListed;

export interface TrainingWindowDrawn {
  form: "drawn";
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

export interface TrainingWindowListed {
  form: "listed";
  split: string;
  /** The window list (repository-relative), its sha256 and the sentence of what chose its windows. */
  list: string;
  sha256: string;
  chose: string;
  /** The list's windows, and this airport's of them (the set's). */
  listCount: number;
  windows: number;
  /** The select seed of the readout numbers each window flew with. */
  selectSeed: number;
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

/** A loss of separation of a round: the loop's step and its time (on the WINDOW's clock: s from its row 0), its two
 *  aircraft (a commanded one's dataset id, or a key of the window's traffic), the commanded ones that answer for it, the
 *  kind of minimum and the distances, whether the other aircraft read a faulty point near it (D114) and whether it
 *  costs W. */
export interface TrainingWindowLoss {
  step: number;
  timeS: number;
  aircraft: [string, string];
  answering: string[];
  kind: string;
  relation: string;
  requiredM: number;
  distanceM: number;
  verticalM: number;
  wakeKnown: boolean;
  readsFault: boolean;
  costsW: boolean;
}

/** A window's end in a round: its losses of separation and the steps at which a recorded aircraft read a faulty point. */
export interface TrainingWindowRoundEnd {
  round: TrainingWindowRound;
  losses: TrainingWindowLoss[];
  faultySteps: number;
}

/** One round's sentence for a commanded aircraft, flown. Rows are the sentence's Δ rows from the first predicted step. */
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
  reward: number;
  /** The Δ row from which the model no longer speaks to it (it answered for a loss and flies on), or null. */
  silentFromRow: number | null;
  speedMaskRows: number;
}

/** Another aircraft of a window on its record's 2 s rows over the window, on the WINDOW's clock (s from its row 0). */
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

/** Window B's moved observed rows: positions and heights on the aircraft's flight clock. */
export interface TrainingWindowMovedStart {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeMslM: number[];
  altitudeHaeM: number[];
}

/** A commanded aircraft of a window. */
export interface TrainingWindowAircraft {
  /** Its place in the window's commanded aircraft (the order they join it). */
  place: number;
  datasetId: string;
  head: TrainingFlight;
  /** When it joins: its row 0, s from the window's row 0. */
  joinS: number;
  /** Its move in time in a compressed window (s), or null. */
  shiftS: number | null;
  /** Its flight time at the window's row 0: a window time t is its flight time t + clockS. */
  clockS: number;
  /** Its first predicted step, on its flight's clock. */
  firstStepS: number;
  startMove: { turnDeg: number; heightM: number; speedScale: number };
  /** Window B: its observed rows from its row 0 to the first predicted step as its start moved them (the view draws them
   *  beside the head's observed track, which stays the recorded flight's: its open-loop reading is read on it); null
   *  otherwise (its start is the head's). */
  movedStart: TrainingWindowMovedStart | null;
  rounds: TrainingWindowSentence[];
}

export interface TrainingWindow {
  /** Its place in the set's windows (the live executor's request names it). */
  index: number;
  kind: TrainingWindowKind;
  /** A compressed window's c; null for a window that is not compressed (every window of stage C). */
  c: number | null;
  /** In the order they join; stage C's windows hold one. */
  commanded: TrainingWindowAircraft[];
  moved: Array<[string, number]>;
  traffic: TrainingWindowTraffic[];
  rounds: TrainingWindowRoundEnd[];
}

export interface TrainingWindowSample {
  /** The stage whose index lists the set (C or D): what the view adds for several commanded aircraft, and the colour of
   *  a round's sentences (§3 item 11). */
  stage: TrainingWindowStage;
  setId: string;
  airport: string;
  executor: { cycleS: number };
  formats: Record<string, string>;
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidates: TrainingCandidate[];
  /** The procedure's limits per candidate runway (`prior_training_export.procedure_block`), drawn in 3D. */
  procedure: TrainingProcedure[];
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

/** MIRROR of `post_train.Settings.start` as the campaign records it: null, or the round of another campaign (D162). Only
 *  this key of the campaign's settings is read. */
function parseStart(settings: Reader): TrainingWindowStart | null {
  const start = settings.nullableChild("start");
  if (start === null) return null;
  const checkpointSha256 = start.string("checkpoint_sha256");
  if (!/^[0-9a-f]{64}$/.test(checkpointSha256)) start.fail(`checkpoint_sha256 ${checkpointSha256} is not a SHA-256`);
  return { campaign: start.string("campaign"), round: start.count("round"), checkpointSha256 };
}

function parseModel(reader: Reader): TrainingWindowModel {
  const rounds = reader.list("rounds").map((value) => parseRound(value, reader));
  if (rounds.length === 0) reader.fail("rounds is empty");
  if (new Set(rounds).size !== rounds.length) reader.fail("a round is listed twice");
  return {
    campaign: reader.string("campaign"), rounds, rowIntervalS: reader.number("rowIntervalS"),
    mostGoArounds: reader.count("mostGoArounds"), procedureMasks: reader.string("procedureMasks"),
    start: parseStart(reader.child("settings")),
  };
}

function parseCohort(reader: Reader): TrainingWindowCohort {
  const form = reader.oneOf("form", ["drawn", "listed"] as const);
  const split = reader.string("split");
  if (!(TRAINING_SPLITS as readonly string[]).includes(split)) reader.fail(`split ${split} is none of ${TRAINING_SPLITS}`);
  if (form === "listed") {
    const listCount = reader.count("listCount", 1);
    const windows = reader.count("windows", 1);
    if (windows > listCount) reader.fail(`the set holds ${windows} of the list's ${listCount} windows`);
    return {
      form, split, list: reader.string("list"), sha256: reader.string("sha256"), chose: reader.string("chose"), listCount,
      windows, selectSeed: reader.number("selectSeed"),
    };
  }
  const kinds = reader.strings("kinds").map((kind) => {
    if (!(TRAINING_WINDOW_KINDS as readonly string[]).includes(kind)) reader.fail(`kind ${kind} is none of ${TRAINING_WINDOW_KINDS}`);
    return kind as TrainingWindowKind;
  });
  return {
    form, split, perAirport: reader.count("perAirport", 1), kinds, seed: reader.number("seed"), windows: reader.count("windows"),
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

/** A loss as the file writes it (its time on the window's clock). Its aircraft are the window's (``known``: the commanded
 *  aircraft's dataset ids and the traffic's keys); the ones that answer for it are commanded aircraft of it. */
function parseLoss(reader: Reader, commanded: Set<string>, known: Set<string>): TrainingWindowLoss {
  const aircraft = reader.strings("aircraft");
  if (aircraft.length !== 2 || aircraft[0] === aircraft[1]) reader.fail(`aircraft is [${aircraft.join(", ")}], not two aircraft`);
  for (const key of aircraft) if (!known.has(key)) reader.fail(`aircraft ${key} is none of the window's`);
  const answering = reader.strings("answering");
  for (const key of answering) {
    if (!aircraft.includes(key) || !commanded.has(key)) reader.fail(`${key} answers for the loss but is not a commanded aircraft of it`);
  }
  return {
    step: reader.count("step", 1), timeS: reader.number("timeS"), aircraft: [aircraft[0], aircraft[1]], answering,
    kind: reader.string("kind"), relation: reader.string("relation"), requiredM: reader.number("requiredM"),
    distanceM: reader.number("distanceM"), verticalM: reader.number("verticalM"), wakeKnown: reader.boolean("wakeKnown"),
    readsFault: reader.boolean("readsFault"), costsW: reader.boolean("costsW"),
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
  if (outcome === TRAINING_LOST_SEPARATION && crossing !== null) reader.fail("a flight ended at a loss of separation crossed no threshold");
  const silentFromRow = reader.nullableNumber("silentFromRow");
  if (silentFromRow !== null && !(Number.isInteger(silentFromRow) && silentFromRow >= 0 && silentFromRow < words.length)) {
    reader.fail(`silentFromRow ${silentFromRow} is not a row of the sentence`);
  }
  return {
    round, words, events, firstRow, startRow, flownFromRow, outcome, endCycle: block.endCycle, crossing, flown: block.flown,
    envelopes: block.envelopes, timedOut: reader.boolean("timedOut"), goArounds: reader.count("goArounds"),
    reward: reader.number("reward"), silentFromRow, speedMaskRows: reader.count("speedMaskRows"),
  };
}

function parseTraffic(reader: Reader, candidates: TrainingCandidate[]): TrainingWindowTraffic {
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
    landingS: reader.number("landingS"), tS: times, lon: reader.numbers("lonDeg", rows),
    lat: reader.numbers("latDeg", rows), altitudeMslM: heights, altitudeHaeM: heights.map((value) => value + hae),
  };
}

/** Window B's observed rows from its row 0 to its first predicted step as its start moved them: the written positions and
 *  heights, on the aircraft's flight clock (`TrainingWindowMovedStart`). */
function parseMovedStart(reader: Reader, head: TrainingFlight, firstRow: number, stepS: number): TrainingWindowMovedStart {
  const rows = reader.count("rows", 1);
  const heights = reader.numbers("heightMslM", rows);
  return {
    tS: Array.from({ length: rows }, (_, i) => (firstRow + i) * stepS), lat: reader.numbers("latDeg", rows),
    lon: reader.numbers("lonDeg", rows), altitudeMslM: heights, altitudeHaeM: heights.map((value) => value + head.haeMinusMslM),
  };
}

function parseAircraft(
  reader: Reader, place: number, kind: TrainingWindowKind, c: number | null, heads: Map<string, TrainingFlight>,
  model: TrainingWindowModel, candidates: TrainingCandidate[], vocabulary: TrainingVocabulary, cycleS: number,
): TrainingWindowAircraft {
  const datasetId = reader.string("datasetId");
  const head = heads.get(datasetId);
  if (head === undefined) return reader.fail(`the set holds no flight ${datasetId}`);
  const joinS = reader.number("joinS");
  if (!(joinS >= 0)) reader.fail(`joinS ${joinS} is before the window's row 0`);
  const shiftS = reader.nullableNumber("shiftS");
  if (c === null && shiftS !== null) reader.fail(`a window that is not compressed moves aircraft ${datasetId} by ${shiftS} s`);
  const closed = head.closedLoop[String(model.rowIntervalS)];
  const row0S = closed.firstRow * vocabulary.stepS;
  const clockS = row0S - joinS;
  const move = reader.child("startMove");
  const startMove = { turnDeg: move.number("turnDeg"), heightM: move.number("heightM"), speedScale: move.number("speedScale") };
  const isMoved = startMove.turnDeg !== 0 || startMove.heightM !== 0 || startMove.speedScale !== 1;
  if (isMoved !== (kind === "B")) reader.fail(`a window ${kind} with ${isMoved ? "a" : "no"} start move: window B, and only B, moves its start`);
  const start = reader.nullableChild("movedStart");
  if ((start !== null) !== (kind === "B")) reader.fail(`a window ${kind} with ${start === null ? "no" : "a"} moved start`);
  const rounds = reader.children("rounds").map((item) => {
    const round = parseRound(item.raw("round"), item);
    return parseSentence(item, head, model, candidates, vocabulary, cycleS, round);
  });
  if (rounds.map((item) => item.round).join() !== model.rounds.join()) {
    reader.fail(`rounds ${rounds.map((item) => item.round).join(", ")}, the set's are ${model.rounds.join(", ")}`);
  }
  // the file's first predicted step is on the window's clock; on the aircraft's it is its closed-loop sentence's
  const firstStepS = clockS + reader.number("firstStepS");
  if (Math.abs(firstStepS - closed.startS) > 1e-6) {
    reader.fail(`firstStepS puts the first predicted step at ${firstStepS} s of the flight, its closed-loop sentence at ${closed.startS} s`);
  }
  return {
    place, datasetId, head, joinS, shiftS, clockS, firstStepS, startMove,
    movedStart: start === null ? null : parseMovedStart(start, head, closed.firstRow, vocabulary.stepS), rounds,
  };
}

function parseWindow(
  reader: Reader, index: number, heads: Map<string, TrainingFlight>, model: TrainingWindowModel, candidates: TrainingCandidate[],
  vocabulary: TrainingVocabulary, cycleS: number,
): TrainingWindow {
  const kind = reader.oneOf("kind", TRAINING_WINDOW_KINDS);
  const c = reader.nullableNumber("c");
  const commanded = reader.children("commanded").map((item, place) =>
    parseAircraft(item, place, kind, c, heads, model, candidates, vocabulary, cycleS));
  if (commanded.length === 0) reader.fail("the window commands no aircraft");
  if (commanded[0].joinS !== 0) reader.fail(`its first commanded aircraft joins at ${commanded[0].joinS} s: the window's row 0 is its row 0`);
  if (commanded.some((aircraft, place) => place > 0 && aircraft.joinS < commanded[place - 1].joinS)) {
    reader.fail("the commanded aircraft are not in the order they join");
  }
  const ids = new Set(commanded.map((aircraft) => aircraft.datasetId));
  if (ids.size !== commanded.length) reader.fail("the window commands one flight twice");
  const moved = reader.list("moved").map((pair) => {
    if (!Array.isArray(pair) || pair.length !== 2 || typeof pair[0] !== "string" || typeof pair[1] !== "number") {
      return reader.fail("a moved aircraft is [key, shift]");
    }
    return [pair[0], pair[1]] as [string, number];
  });
  const traffic = reader.children("traffic").map((item) => parseTraffic(item, candidates));
  const known = new Set([...ids, ...traffic.map((aircraft) => aircraft.key)]);
  const rounds = reader.children("rounds").map((item) => ({
    round: parseRound(item.raw("round"), item),
    losses: item.children("losses").map((loss) => parseLoss(loss, ids, known)),
    faultySteps: item.count("faultySteps"),
  }));
  if (rounds.map((item) => item.round).join() !== model.rounds.join()) {
    reader.fail(`the window's rounds ${rounds.map((item) => item.round).join(", ")}, the set's are ${model.rounds.join(", ")}`);
  }
  // a loss and the end of the aircraft that answers for it go together: an aircraft answers for at most one loss of a
  // round (it is not spoken to after it, D144); one whose flight ended at a loss answers for one, and one that answers
  // ended there or is silent from a row on; one silent from a row on answers for one
  for (const aircraft of commanded) {
    aircraft.rounds.forEach((sentence, r) => {
      const answered = rounds[r].losses.filter((loss) => loss.answering.includes(aircraft.datasetId)).length;
      if (answered > 1) reader.fail(`${aircraft.datasetId} answers for ${answered} losses in round ${sentence.round}: one at most (D144)`);
      if (sentence.outcome === TRAINING_LOST_SEPARATION && answered === 0) {
        reader.fail(`${aircraft.datasetId} ended at a loss of separation in round ${sentence.round}, but answers for none: they go together`);
      }
      if (answered === 1 && sentence.outcome !== TRAINING_LOST_SEPARATION && sentence.silentFromRow === null) {
        reader.fail(`${aircraft.datasetId} answers for a loss in round ${sentence.round}, but flew on spoken to its ${sentence.outcome}: they go together`);
      }
      if (sentence.silentFromRow !== null && answered === 0) {
        reader.fail(`${aircraft.datasetId} is silent from row ${sentence.silentFromRow} in round ${sentence.round}, but answers for no loss: they go together`);
      }
    });
  }
  return { index, kind, c, commanded, moved, traffic, rounds };
}

/** Parse a set's sample. A schema other than `TRAINING_WINDOW_SAMPLE_SCHEMA` is refused whole, naming the one found and
 *  the one expected. */
export function parseTrainingWindowSample(raw: unknown, stage: TrainingWindowStage): Parsed<TrainingWindowSample> {
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
      stage, setId: sample.string("setId"), airport: sample.string("airport"), executor: { cycleS }, formats: parseFormats(sample),
      vocabulary, airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidates, procedure: parseProcedure(sample, candidates), model, cohort, source: parseSource(sample.child("source")),
      flights, windows,
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

export function trainingWindowDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

export function trainingWindowIndexPath(airportCode: string, stage: TrainingWindowStage): string {
  return `${trainingWindowDirectory(airportCode)}/${TRAINING_WINDOW_INDEX_FILES[stage]}`;
}

/** The airport's index of ``stage``'s window sets, refused unless it is the one asked for. */
export async function fetchTrainingWindowIndex(airportCode: string, stage: TrainingWindowStage): Promise<Parsed<TrainingWindowIndex>> {
  const path = trainingWindowIndexPath(airportCode, stage);
  const parsed = parseTrainingWindowIndex(await fetchJson<unknown>(path));
  if (parsed.ok && parsed.value.airport !== airportCode) {
    return { ok: false, problem: `${path} is ${parsed.value.airport}'s index, not ${airportCode}'s` };
  }
  return parsed;
}

/** A set's sample of ``stage``, refused unless it is the set and airport asked for. */
export async function fetchTrainingWindowSample(airportCode: string, file: string, setId: string, stage: TrainingWindowStage
): Promise<Parsed<TrainingWindowSample>> {
  const parsed = parseTrainingWindowSample(await fetchJson<unknown>(`${trainingWindowDirectory(airportCode)}/${file}`), stage);
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

/** The model a campaign starts from, by name (frontend §4.3): the base (D29), or another campaign's round by the
 *  campaign's directory name (D162). The one name the tabs, the statistics and the lines give it. */
export function startName(start: TrainingWindowStart | null): string {
  return start === null ? "base" : `${start.campaign.split("/").pop()} r${start.round}`;
}

/** A round as the view's lines and titles name it; the start by what it is (`startName`). */
export function roundLabel(round: TrainingWindowRound, start: TrainingWindowStart | null): string {
  return round === TRAINING_WINDOW_START ? `start (${startName(start)})` : `round ${round}`;
}

/** Where a commanded aircraft is silent from (D144), on its flight clock: the start of its sentence's row
 *  ``silentFromRow`` (rows of Δ ``rowIntervalS`` from its first predicted step); null when it is not silent. */
export function silentFromS(aircraft: TrainingWindowAircraft, sentence: TrainingWindowSentence, rowIntervalS: number): number | null {
  return sentence.silentFromRow === null ? null : aircraft.firstStepS + sentence.silentFromRow * rowIntervalS;
}

/** The window's reward W of a round (multi-aircraft control D141): the sum of its commanded aircraft's rewards. */
export function windowReward(window: TrainingWindow, place: number): number {
  return window.commanded.reduce((sum, aircraft) => sum + aircraft.rounds[place].reward, 0);
}

/** The most recorded aircraft in the air at one time over the window (its traffic's tracks, by their first and last row). */
export function mostTrafficAtOnce(window: TrainingWindow): number {
  const edges = window.traffic.flatMap((aircraft) => [[aircraft.tS[0], 1], [aircraft.tS[aircraft.tS.length - 1], -1]] as Array<[number, number]>);
  edges.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  let now = 0;
  let most = 0;
  for (const [, step] of edges) {
    now += step;
    most = Math.max(most, now);
  }
  return most;
}

/** Where aircraft ``key`` of the window is at window time ``windowS`` in round ``place``: a commanded aircraft from the
 *  row it joins — on its observed rows before its first predicted step (window B's: as its start moved them, which end
 *  one 2 s row before that step: null between), then on its round's flown track (its flight clock); another on its
 *  record; null outside its track. */
export function windowAircraftAt(window: TrainingWindow, place: number, key: string, windowS: number): [number, number, number] | null {
  const commanded = window.commanded.find((aircraft) => aircraft.datasetId === key);
  if (commanded !== undefined) {
    if (windowS < commanded.joinS) return null;
    const t = onAircraftClock(commanded, windowS);
    if (t >= commanded.firstStepS) return trackAt(commanded.rounds[place].flown, t);
    return trackAt(commanded.movedStart ?? commanded.head.observed, t);
  }
  const traffic = window.traffic.find((aircraft) => aircraft.key === key);
  return traffic === undefined ? null : trackAt(traffic, windowS);
}

/** Where a track is at time ``t`` of its own clock (on the straight line between the two rows around it — the only
 *  number computed here, between two of the exporter's points); null outside it. */
export function trackAt(track: { tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[] }, t: number): [number, number, number] | null {
  const times = track.tS;
  if (times.length === 0 || t < times[0] || t > times[times.length - 1]) return null;
  const hi = times.findIndex((value) => value >= t);
  if (times[hi] === t) return [track.lon[hi], track.lat[hi], track.altitudeHaeM[hi]];
  const lo = hi - 1;
  const f = (t - times[lo]) / (times[hi] - times[lo]);
  const mix = (values: number[]) => values[lo] + f * (values[hi] - values[lo]);
  return [mix(track.lon), mix(track.lat), mix(track.altitudeHaeM)];
}

/** A window time (s from its row 0) on ``aircraft``'s flight clock. */
export function onAircraftClock(aircraft: TrainingWindowAircraft, windowS: number): number {
  return windowS + aircraft.clockS;
}

/** A flight time of ``aircraft`` on the window's clock (s from its row 0): `onAircraftClock`'s inverse. */
export function onWindowClock(aircraft: TrainingWindowAircraft, flightS: number): number {
  return flightS - aircraft.clockS;
}

/** The losses of a round that ``datasetId`` is in. */
export function lossesOf(end: TrainingWindowRoundEnd, datasetId: string): TrainingWindowLoss[] {
  return end.losses.filter((loss) => loss.aircraft.includes(datasetId));
}

/** The other aircraft of a loss that ``datasetId`` is in. */
export function otherOf(loss: TrainingWindowLoss, datasetId: string): string {
  return loss.aircraft[0] === datasetId ? loss.aircraft[1] : loss.aircraft[0];
}

/** What a derived flight stands for: the set's window, the commanded aircraft, the round, its sentence and the window's
 *  end in that round. */
export interface TrainingWindowOrigin {
  sample: TrainingWindowSample;
  window: TrainingWindow;
  aircraft: TrainingWindowAircraft;
  round: TrainingWindowRound;
  sentence: TrainingWindowSentence;
  end: TrainingWindowRoundEnd;
}

const origins = new WeakMap<TrainingFlight, TrainingWindowOrigin>();
const views = new WeakMap<TrainingWindowAircraft, Map<TrainingWindowRound, TrainingFlight>>();

/** The set's window, aircraft and round a flight on screen was derived from; undefined for any other flight. */
export function trainingWindowOriginOf(flight: TrainingFlight): TrainingWindowOrigin | undefined {
  return origins.get(flight);
}

/** A round's sentence as stage A's closed-loop sentence at Δ, so the sentence bar, the read-back window and the 3D layers
 *  read it unchanged (as stage B's: `trainingPriorSample.ts`): its envelopes the judge's on its flown track (D135); the
 *  reading added no word;
 *  `notReached` counts the words said after the aircraft's end (`wordUnreached`). */
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

/** A commanded aircraft of a window as the views draw it with ``round``'s sentence read at the set's Δ (its observed
 *  track and open-loop reading its head's: window B's moved start is drawn by the window's own layer). Another window,
 *  aircraft or round is another flight (its key says which), so the cursor and the live pick reset with it. Built once. */
export function trainingWindowFlightView(
  sample: TrainingWindowSample, window: TrainingWindow, aircraft: TrainingWindowAircraft, round: TrainingWindowRound,
): TrainingFlight {
  const kept = views.get(aircraft)?.get(round);
  if (kept !== undefined) return kept;
  const place = window.rounds.findIndex((item) => item.round === round);
  if (place < 0) throw new Error(`window ${window.index} has no round ${round}`);
  const sentence = aircraft.rounds[place];
  const key = String(sample.model.rowIntervalS);
  const view: TrainingFlight = {
    ...aircraft.head, flightKey: `${aircraft.head.flightKey}~window-${window.index}-aircraft-${aircraft.place}-round-${round}`,
    closedLoop: { [key]: closedLoopOf(sample.model, aircraft.head, sentence) },
  };
  origins.set(view, { sample, window, aircraft, round, sentence, end: window.rounds[place] });
  const byRound = views.get(aircraft) ?? new Map<TrainingWindowRound, TrainingFlight>();
  byRound.set(round, view);
  views.set(aircraft, byRound);
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
