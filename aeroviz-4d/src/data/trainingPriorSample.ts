/**
 * trainingPriorSample.ts
 * ----------------------
 * The Training view's data contract for STAGE B of the two-tier model: the airport's index of PRIOR sets and one set's
 * flights, each with the sentences the prior said in free generation. Written by
 * `ts_transformer/experiments/prior_training_export.py` (the files: `ts_transformer/prior/training_files.py`); design:
 * prior design §12 B6, outline §6.
 *
 * A flight carries stage A's head — the observed track, the open-loop sentence, the closed-loop sentence at the prior's Δ
 * with the executor's flight (`closedLoop`, read by stage A's reader: `trainingSample.ts`) — and `prior`: every sentence
 * the prior said from the flight's first predicted step, flown again by the executor: its five columns (`words`, the
 * events decoded by the vocabulary in Python), its flown track and attitude, its outcome with the threshold crossing and
 * the decision-altitude (DA) check, the probability of "go-around" at each word row, and the words the procedure masks
 * BLOCKED at each row (by column). The set carries the procedure's limits per candidate runway (`procedure`): the region
 * outline, the glidepath lower edge, the DA point and the entry point at the FAF.
 *
 * THIS FILE COMPUTES NO WORD, MASK OR LIMIT: every number is the exporter's. The reader checks the file's BOOKKEEPING —
 * lengths, rows, that the events are the grid's words, that a sentence starts where the closed-loop sentence does — and
 * draws numbers.
 *
 * NO COMPATIBILITY. The index and sample schemas and the set kind are pinned below; a file that carries anything else is
 * refused by name (the schema found and the one expected). This reader never reads stage A's `index_v4.json`.
 *
 * SI units only. Heights in the files are MSL; the 3D scene's are ellipsoid heights (the flight's own, or the candidate
 * runway's, `haeMinusMslM` added once, here).
 */

import { fetchJson } from "../utils/fetchJson";
import { attempt, parseManifest, Reader, type Parsed } from "./trainingReader";
import {
  lastStateCycle,
  parseAttitudeOf,
  parseCandidates,
  parseEvents,
  parseFlight,
  parseFormats,
  parseGrid,
  parseVocabulary,
  unwrapDegrees,
  TRAINING_OUTCOMES,
  TRAINING_READING_RULE,
  readCrossing,
  type TrainingCandidate,
  type TrainingClosedLoop,
  type TrainingCrossing,
  type TrainingEvent,
  type TrainingFlight,
  type TrainingOutcome,
  type TrainingSelection,
  type TrainingTrack,
  type TrainingVocabulary,
  wordUnreached,
} from "./trainingSample";

/** MIRROR of the exporter's `INDEX_SCHEMA` (`ts_transformer/prior/training_files.py`): the airport's index of prior sets.
 *  A name changes with its file's shape, on both sides, in the same change. */
export const TRAINING_PRIOR_INDEX_SCHEMA = "aeroviz-training-prior-index-v1";
/** MIRROR of `INDEX_FILE`: an index of its own beside stage A's `index_v4.json`, which this reader never reads. */
export const TRAINING_PRIOR_INDEX_FILE = "index_prior_v1.json";
/** MIRROR of `SAMPLE_SCHEMA`: a prior set's sample. */
export const TRAINING_PRIOR_SAMPLE_SCHEMA = "aeroviz-training-prior-sample-v1";
/** MIRROR of `SET_KIND`: the flights of one free-generation readout. */
export const TRAINING_PRIOR_SET_KIND = "prior-free-generation";
/** MIRROR of the procedure masks' columns (`prior.procedure.ProcedureMasks.columns`, by `COLUMNS` name): the blocked words are
 *  listed under these names (the exporter's `blocked`); the reader refuses any other set of names, and
 *  `test_prior_segment.py` pins them there. */
export const TRAINING_PRIOR_BLOCKED_COLUMNS = ["altitude", "angle"] as const;
export type TrainingPriorBlockedColumn = (typeof TRAINING_PRIOR_BLOCKED_COLUMNS)[number];

export interface TrainingPriorModel {
  prior: string;
  selection: string;
  /** The row interval Δ the prior speaks at; the one Δ of the flights' closed-loop sentence. */
  rowIntervalS: number;
  temperature: number;
  mostGoArounds: number;
  procedureMasks: string;
  checkpointSha256: string;
}

export interface TrainingPriorCohort {
  split: string;
  flights: number;
  perAirport: number;
  seed: number;
  /** The sentences the prior said per flight. */
  samples: number;
  readoutSeed: number;
  readoutFlights: number;
  drawnFrom: string;
}

export interface TrainingPriorSource {
  readout: string;
  instructions: string;
  executor: string;
  smoke: boolean;
  git: { head: string; dirty: boolean };
}

/** A set as the index lists it. */
export interface TrainingPriorSetEntry {
  id: string;
  title: string;
  /** The set's sample, relative to the airport's `training/` directory. */
  file: string;
  flights: number;
  sentences: number;
  formats: Record<string, string>;
  model: TrainingPriorModel;
  cohort: TrainingPriorCohort;
  source: TrainingPriorSource;
}

export interface TrainingPriorIndex {
  airport: string;
  sets: TrainingPriorSetEntry[];
  /** Entries that are not set entries at all, each with the field that failed. */
  rejected: Array<{ id: string; problem: string }>;
}

/** A point of the procedure's limits: the airport frame, the globe, and its height MSL and as Cesium draws it. */
export interface TrainingProcedurePoint {
  eM: number;
  nM: number;
  lat: number;
  lon: number;
  heightMslM: number;
  heightHaeM: number;
  /** Metres before the threshold along the course. */
  beforeThresholdM: number;
}

/** One candidate runway's final as the procedure masks read it. */
export interface TrainingProcedure {
  index: number;
  ident: string;
  fafBeforeThresholdM: number;
  /** The glidepath lower edge lies this far below the glidepath. */
  glidepathBelowM: number;
  /** The closed outline of the region the masks rule (inside the FAF and the LPV cone), on the ground. */
  region: { eM: number[]; nM: number[]; lat: number[]; lon: number[] };
  /** The glidepath lower edge along the course: a height at each point. */
  glidepathLowerEdge: { lat: number[]; lon: number[]; heightMslM: number[]; heightHaeM: number[]; beforeThresholdM: number[] };
  /** Where the glidepath reaches the decision height. */
  decision: TrainingProcedurePoint;
  /** The entry point at the FAF, at the entry height. */
  entry: TrainingProcedurePoint;
}

/** One sentence the prior said, flown again by the executor. Rows are the sentence's Δ rows from the first predicted step. */
export interface TrainingPriorSentence {
  /** Which sample of the readout (the prior's seed order); the sentences of one flight differ by it. */
  sample: number;
  words: number[][];
  events: TrainingEvent[];
  firstRow: number;
  startRow: number;
  flownFromRow: number;
  outcome: TrainingOutcome;
  /** The executor's cycles from the first predicted step to the judge's outcome row. */
  endCycle: number;
  timedOut: boolean;
  goArounds: number;
  /** null: the flight did not cross the threshold. */
  crossing: TrainingCrossing | null;
  /** The flown flight on the 2 s rows from the first predicted step, on the flight's clock, with its attitude. */
  flown: TrainingTrack;
  /** Per word row: the probability the prior gave "go-around" there. */
  goAroundProbability: number[];
  /** Per word row: whether the procedure let a go-around be said there / whether the flight was on final. */
  goAroundPermitted: boolean[];
  onFinal: boolean[];
  /** Per masked column, per word row: the word values the procedure blocked. */
  blocked: Record<TrainingPriorBlockedColumn, number[][]>;
}

export interface TrainingPriorFlight {
  /** Stage A's flight: the observed track, the open-loop sentence, the closed-loop sentence at the prior's Δ. */
  head: TrainingFlight;
  sentences: TrainingPriorSentence[];
}

export interface TrainingPriorSample {
  setId: string;
  airport: string;
  executor: { cycleS: number };
  formats: Record<string, string>;
  vocabulary: TrainingVocabulary;
  airportFrame: { code: string; lat: number; lon: number; elevationM: number };
  candidates: TrainingCandidate[];
  model: TrainingPriorModel;
  cohort: TrainingPriorCohort;
  source: TrainingPriorSource;
  procedure: TrainingProcedure[];
  flights: TrainingPriorFlight[];
}

// ── the index ────────────────────────────────────────────────────────────────

function parseModel(reader: Reader): TrainingPriorModel {
  return {
    prior: reader.string("prior"), selection: reader.string("selection"), rowIntervalS: reader.number("rowIntervalS"),
    temperature: reader.number("temperature"), mostGoArounds: reader.count("mostGoArounds"),
    procedureMasks: reader.string("procedureMasks"), checkpointSha256: reader.string("checkpointSha256"),
  };
}

function parseCohort(reader: Reader): TrainingPriorCohort {
  return {
    split: reader.string("split"), flights: reader.count("flights"), perAirport: reader.count("perAirport", 1),
    seed: reader.number("seed"), samples: reader.count("samples", 1), readoutSeed: reader.number("readoutSeed"),
    readoutFlights: reader.count("readoutFlights"), drawnFrom: reader.string("drawnFrom"),
  };
}

function parseSource(reader: Reader): TrainingPriorSource {
  const git = reader.child("git");
  return {
    readout: reader.string("readout"), instructions: reader.string("instructions"), executor: reader.string("executor"),
    smoke: reader.boolean("smoke"), git: { head: git.string("head"), dirty: git.boolean("dirty") },
  };
}

function parseSetEntry(entry: Reader): TrainingPriorSetEntry {
  entry.oneOf("kind", [TRAINING_PRIOR_SET_KIND]);
  entry.oneOf("readingRule", [TRAINING_READING_RULE]);
  return {
    id: entry.string("id"), title: entry.string("title"), file: entry.string("file"), flights: entry.count("flights"),
    sentences: entry.count("sentences"), formats: parseFormats(entry), model: parseModel(entry.child("model")),
    cohort: parseCohort(entry.child("cohort")), source: parseSource(entry.child("source")),
  };
}

/** Parse the index. A bad entry is rejected on its own; only an index that is not an index at all fails the call — and a
 *  schema other than `TRAINING_PRIOR_INDEX_SCHEMA` is refused whole, naming the one found and the one expected. */
export function parseTrainingPriorIndex(raw: unknown): Parsed<TrainingPriorIndex> {
  const manifest = parseManifest(raw, {
    name: "index", schema: TRAINING_PRIOR_INDEX_SCHEMA, listKey: "sets", entryName: "set",
  }, parseSetEntry);
  if (!manifest.ok) return manifest;
  return { ok: true, value: { airport: manifest.value.airport, sets: manifest.value.entries, rejected: manifest.value.rejected } };
}

// ── one set ──────────────────────────────────────────────────────────────────

function procedurePoint(reader: Reader, haeMinusMslM: number): TrainingProcedurePoint {
  const heightMslM = reader.number("heightMslM");
  return {
    // the exporter writes a point's place as lists of one (`placed`)
    eM: reader.numbers("eM", 1)[0], nM: reader.numbers("nM", 1)[0], lat: reader.numbers("latDeg", 1)[0],
    lon: reader.numbers("lonDeg", 1)[0], heightMslM, heightHaeM: heightMslM + haeMinusMslM,
    beforeThresholdM: reader.number("beforeThresholdM"),
  };
}

function parseProcedure(reader: Reader, candidates: TrainingCandidate[]): TrainingProcedure[] {
  const procedure = reader.children("procedure").map((item, position) => {
    const index = item.integer("index", 0, candidates.length - 1);
    if (index !== position) item.fail(`index is ${index}, expected ${position}: the limits are in the candidates' order`);
    const ident = item.string("ident");
    if (candidates[index].ident !== ident) item.fail(`ident ${ident} is not candidate ${index}'s (${candidates[index].ident})`);
    const hae = candidates[index].haeMinusMslM;
    const region = item.child("region");
    const ring = region.numbers("eM");
    if (ring.length < 4) region.fail("the outline has fewer than four points");
    const ringN = region.numbers("nM", ring.length);
    if (ring[0] !== ring[ring.length - 1] || ringN[0] !== ringN[ringN.length - 1]) {
      region.fail("the outline is not closed: its last point is not its first");
    }
    const edge = item.child("glidepathLowerEdge");
    const points = edge.numbers("latDeg").length;
    if (points < 2) edge.fail("the glidepath lower edge has fewer than two points");
    const heightMslM = edge.numbers("heightMslM", points);
    return {
      index, ident, fafBeforeThresholdM: item.number("fafBeforeThresholdM"), glidepathBelowM: item.number("glidepathBelowM"),
      region: { eM: ring, nM: ringN, lat: region.numbers("latDeg", ring.length), lon: region.numbers("lonDeg", ring.length) },
      glidepathLowerEdge: {
        lat: edge.numbers("latDeg", points), lon: edge.numbers("lonDeg", points), heightMslM,
        heightHaeM: heightMslM.map((value) => value + hae), beforeThresholdM: edge.numbers("beforeThresholdM", points),
      },
      decision: procedurePoint(item.child("decision"), hae), entry: procedurePoint(item.child("entry"), hae),
    };
  });
  if (procedure.length !== candidates.length) reader.fail(`procedure lists ${procedure.length} runways, the set has ${candidates.length} candidates`);
  return procedure;
}

function parseFlags(reader: Reader, key: string, length: number): boolean[] {
  return reader.numbers(key, length).map((value) => {
    if (value !== 0 && value !== 1) reader.fail(`${key} holds ${value}, not 0 or 1`);
    return value === 1;
  });
}

function parseBlocked(reader: Reader, rows: number, vocabulary: TrainingVocabulary): Record<TrainingPriorBlockedColumn, number[][]> {
  const blocked = reader.child("blocked");
  const names = Object.keys(reader.raw("blocked") as Record<string, unknown>);
  if (names.length !== TRAINING_PRIOR_BLOCKED_COLUMNS.length || names.some((name, i) => name !== TRAINING_PRIOR_BLOCKED_COLUMNS[i])) {
    blocked.fail(`blocked is listed by [${names.join(", ")}], expected [${TRAINING_PRIOR_BLOCKED_COLUMNS.join(", ")}]`);
  }
  // the words of a column: an altitude is a level or "no level-off"; an angle is a class
  const words: Record<TrainingPriorBlockedColumn, number> = {
    altitude: vocabulary.noLevelOff + 1, angle: vocabulary.angleClasses.length,
  };
  const read = (column: TrainingPriorBlockedColumn): number[][] => {
    const list = blocked.list(column);
    if (list.length !== rows) blocked.fail(`${column} holds ${list.length} rows, expected one per word row (${rows})`);
    return list.map((row, index) => {
      if (!Array.isArray(row) || !row.every((value) => Number.isInteger(value) && value >= 0 && value < words[column])) {
        blocked.fail(`${column}[${index}] is not a list of ${column} words (0 to ${words[column] - 1})`);
      }
      return row as number[];
    });
  };
  return { altitude: read("altitude"), angle: read("angle") };
}

function parseSentence(
  reader: Reader, head: TrainingFlight, model: TrainingPriorModel, candidates: TrainingCandidate[], vocabulary: TrainingVocabulary,
  cycleS: number,
): TrainingPriorSentence {
  const stepS = vocabulary.stepS;
  const closed = head.closedLoop[String(model.rowIntervalS)];
  const sample = reader.count("sample");
  const firstRow = reader.count("firstRow");
  const startRow = reader.count("startRow");
  const flownFromRow = reader.count("flownFromRow");
  if (firstRow !== closed.firstRow || startRow !== closed.startRow || flownFromRow !== closed.flownFromRow) {
    reader.fail(`starts at row ${firstRow} + ${startRow} (state row ${flownFromRow}), but the flight's closed-loop sentence starts at ` +
      `${closed.firstRow} + ${closed.startRow} (${closed.flownFromRow}): the sentences of a flight start at one first predicted step`);
  }
  const words = parseGrid(reader, "words");
  if (reader.count("rows", 1) !== words.length) reader.fail(`rows is ${reader.raw("rows")}, but words holds ${words.length}`);
  // the closed-loop reading adds no word to what the prior said
  const events = parseEvents(reader, words, candidates, false);
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
  const crossing = reader.nullableChild("crossing");
  const startIndex = firstRow + flownFromRow;
  const heightMslM = track.numbers("heightMslM", rows);
  const trackDeg = track.numbers("trackDeg", rows);
  const hae = head.haeMinusMslM;
  const flown: TrainingTrack = {
    tS: Array.from({ length: rows }, (_, i) => (startIndex + i) * stepS),
    eM: track.numbers("eM", rows), nM: track.numbers("nM", rows), lat: track.numbers("latDeg", rows),
    lon: track.numbers("lonDeg", rows), altitudeMslM: heightMslM, altitudeHaeM: heightMslM.map((value) => value + hae), trackDeg,
    trackPlotDeg: unwrapDegrees(trackDeg, head.observed.trackPlotDeg[Math.min(startIndex, head.observed.tS.length - 1)]),
    groundSpeedMps: track.numbers("groundSpeedMps", rows), verticalRateMps: track.numbers("verticalRateMps", rows),
    attitude: parseAttitudeOf(reader, rows),
  };
  const rowsSaid = words.length;
  return {
    sample, words, events, firstRow, startRow, flownFromRow, outcome, endCycle, timedOut: reader.boolean("timedOut"),
    goArounds: reader.count("goArounds"), crossing: crossing === null ? null : readCrossing(crossing, candidates.length), flown,
    goAroundProbability: reader.numbers("goAroundProbability", rowsSaid), goAroundPermitted: parseFlags(reader, "goAroundPermitted", rowsSaid),
    onFinal: parseFlags(reader, "onFinal", rowsSaid), blocked: parseBlocked(reader, rowsSaid, vocabulary),
  };
}

/** Parse a set's sample. A schema other than `TRAINING_PRIOR_SAMPLE_SCHEMA` is refused whole, naming the one found and the
 *  one expected. */
export function parseTrainingPriorSample(raw: unknown): Parsed<TrainingPriorSample> {
  return attempt(() => {
    const sample = Reader.of(raw, "sample");
    sample.oneOf("schema", [TRAINING_PRIOR_SAMPLE_SCHEMA]);
    sample.oneOf("readingRule", [TRAINING_READING_RULE]);
    const vocabulary = parseVocabulary(sample.child("vocabulary"));
    const candidates = parseCandidates(sample);
    const frame = sample.child("airportFrame");
    const cycleS = sample.child("executor").number("cycleS");
    if (!(cycleS > 0)) sample.fail(`executor.cycleS is ${cycleS}, not a positive length`);
    if (!Number.isInteger(vocabulary.stepS / cycleS)) sample.fail(`a ${vocabulary.stepS} s row is not a whole number of ${cycleS} s cycles`);
    const model = parseModel(sample.child("model"));
    if (!vocabulary.rowIntervalsS.includes(model.rowIntervalS)) {
      sample.fail(`the prior speaks at Δ ${model.rowIntervalS} s, which is none of the vocabulary's [${vocabulary.rowIntervalsS.join(", ")}]`);
    }
    const flights = sample.children("flights").map((flight): TrainingPriorFlight => {
      // a prior set's flights carry the closed-loop sentence at the prior's Δ only
      const head = parseFlight(flight, vocabulary, candidates, cycleS, [model.rowIntervalS]);
      const sentences = flight.children("prior").map((item) => parseSentence(item, head, model, candidates, vocabulary, cycleS));
      if (sentences.length === 0) flight.fail("prior holds no sentence");
      return { head, sentences };
    });
    const keys = new Set(flights.map((flight) => flight.head.flightKey));
    if (keys.size !== flights.length) sample.fail("two flights carry one flight key: a flight is its key");
    return {
      setId: sample.string("setId"), airport: sample.string("airport"), executor: { cycleS }, formats: parseFormats(sample),
      vocabulary, airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidates, model, cohort: parseCohort(sample.child("cohort")), source: parseSource(sample.child("source")),
      procedure: parseProcedure(sample, candidates), flights,
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

export function trainingPriorDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

export function trainingPriorIndexPath(airportCode: string): string {
  return `${trainingPriorDirectory(airportCode)}/${TRAINING_PRIOR_INDEX_FILE}`;
}

/** The airport's index, refused unless it is the one asked for (a file copied under another airport's directory). */
export async function fetchTrainingPriorIndex(airportCode: string): Promise<Parsed<TrainingPriorIndex>> {
  const parsed = parseTrainingPriorIndex(await fetchJson<unknown>(trainingPriorIndexPath(airportCode)));
  if (parsed.ok && parsed.value.airport !== airportCode) {
    return { ok: false, problem: `${trainingPriorIndexPath(airportCode)} is ${parsed.value.airport}'s index, not ${airportCode}'s` };
  }
  return parsed;
}

/** A set's sample, refused unless it is the set and airport asked for. */
export async function fetchTrainingPriorSample(airportCode: string, file: string, setId: string): Promise<Parsed<TrainingPriorSample>> {
  const parsed = parseTrainingPriorSample(await fetchJson<unknown>(`${trainingPriorDirectory(airportCode)}/${file}`));
  if (parsed.ok && (parsed.value.airport !== airportCode || parsed.value.setId !== setId)) {
    return { ok: false, problem: `${file} holds set ${parsed.value.setId} of ${parsed.value.airport}, not ${setId} of ${airportCode}` };
  }
  return parsed;
}

// ── a sentence as the stage-A views read it ──────────────────────────────────

/** Which sentence of a flight is on screen: one the prior said (its sample), or the flight's closed-loop sentence. */
export type TrainingPriorWhich = number | "closedLoop";

/** What a derived flight stands for: the set's flight and the sentence. */
export interface TrainingPriorOrigin {
  /** The set's own flight key (the derived flight's key names its sentence too). */
  flightKey: string;
  which: TrainingPriorWhich;
  /** null: the flight's closed-loop sentence. */
  sentence: TrainingPriorSentence | null;
  sample: TrainingPriorSample;
  flight: TrainingPriorFlight;
}

const origins = new WeakMap<TrainingFlight, TrainingPriorOrigin>();
const views = new WeakMap<TrainingPriorFlight, Map<TrainingPriorWhich, TrainingFlight>>();

/** The set's flight and sentence a flight on screen was derived from; undefined for a flight of stage A. */
export function trainingPriorOriginOf(flight: TrainingFlight): TrainingPriorOrigin | undefined {
  return origins.get(flight);
}

/** A sentence the prior said as stage A's closed-loop sentence at Δ, so the sentence bar, the read-back window and the 3D
 *  layers read it unchanged. No envelope was judged on it (the judge's envelopes are the closed-loop sentence's own) and the
 *  reading added no word. The executor flies on to the sentence's last word, which can lie past the judge's outcome (a crossing
 *  of another runway): `notReached` counts the words said after it (`wordUnreached`). */
function closedLoopOf(model: TrainingPriorModel, flight: TrainingFlight, sentence: TrainingPriorSentence): TrainingClosedLoop {
  const closed = flight.closedLoop[String(model.rowIntervalS)];
  const result = {
    rowIntervalS: model.rowIntervalS, firstRow: sentence.firstRow, startRow: sentence.startRow, flownFromRow: sentence.flownFromRow,
    startS: closed.startS, words: sentence.words, events: sentence.events,
    lateralM: sentence.words.map(() => null), verticalM: sentence.words.map(() => null), timedOut: sentence.timedOut,
    cycleS: closed.cycleS, flown: sentence.flown,
    replay: {
      outcome: sentence.outcome, endCycle: sentence.endCycle, crossing: sentence.crossing, flewTheSentence: true, notReached: 0,
      envelopes: null,
    },
  };
  const notReached = sentence.events.filter((event) => wordUnreached(result, event.row)).length;
  return { ...result, replay: { ...result.replay, flewTheSentence: notReached === 0, notReached } };
}

/** The runway in force at Δ row ``row`` of a sentence: the last runway word (not a go-around) said at or before it, else the
 *  first one said; the masks act on this runway, whatever the observed flight landed on. */
export function runwayInForce(events: TrainingEvent[], row: number): number {
  const runways = events.filter((event) => event.column === 0 && event.says.column === "runway" && !event.says.goAround);
  const said = runways.filter((event) => event.row <= row);
  const event = said.length > 0 ? said[said.length - 1] : runways[0];
  return (event.says as { runwayIndex: number }).runwayIndex;
}

/** The flight as the views draw it with ``which`` of its sentences read at the prior's Δ. Another sentence is another flight
 *  (its key says which), so the cursor and the live pick, which belong to a flight, reset with it. Built once. */
export function trainingPriorFlightView(sample: TrainingPriorSample, flight: TrainingPriorFlight, which: TrainingPriorWhich): TrainingFlight {
  const kept = views.get(flight)?.get(which);
  if (kept !== undefined) return kept;
  const sentence = which === "closedLoop" ? null : flight.sentences.find((item) => item.sample === which) ?? null;
  if (which !== "closedLoop" && sentence === null) throw new Error(`flight ${flight.head.flightKey} has no prior sentence ${which}`);
  const key = String(sample.model.rowIntervalS);
  const view: TrainingFlight = {
    ...flight.head, flightKey: `${flight.head.flightKey}~${which === "closedLoop" ? "closed-loop" : `prior-${which}`}`,
    closedLoop: { [key]: sentence === null ? flight.head.closedLoop[key] : closedLoopOf(sample.model, flight.head, sentence) },
  };
  origins.set(view, { flightKey: flight.head.flightKey, which, sentence, sample, flight });
  const byWhich = views.get(flight) ?? new Map<TrainingPriorWhich, TrainingFlight>();
  byWhich.set(which, view);
  views.set(flight, byWhich);
  return view;
}

/** What the sentence bar, the read-back window and the 3D layers read: the set's vocabulary with its one closed-loop Δ, the
 *  candidates and the derived flight. */
export function trainingPriorSelectionOf(sample: TrainingPriorSample, flight: TrainingFlight): TrainingSelection {
  return {
    airport: sample.airport, setId: sample.setId, vocabulary: { ...sample.vocabulary, rowIntervalsS: [sample.model.rowIntervalS] },
    candidates: sample.candidates, flight,
  };
}
