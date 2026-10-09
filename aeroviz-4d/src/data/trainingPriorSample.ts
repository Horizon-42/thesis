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
 * refused by name (the schema found and the one expected). This reader never reads stage A's `index_v5.json`.
 *
 * SI units only. Heights in the files are MSL; the 3D scene's are ellipsoid heights (the flight's own, or the candidate
 * runway's, `haeMinusMslM` added once, here).
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
  TRAINING_OUTCOMES,
  TRAINING_READING_RULE,
  TRAINING_SPLITS,
  type TrainingCandidate,
  type TrainingClosedLoop,
  type TrainingCrossing,
  type TrainingEnvelopes,
  type TrainingEvent,
  type TrainingSplit,
  type TrainingFlight,
  type TrainingOutcome,
  type TrainingSelection,
  type TrainingTrack,
  type TrainingVocabulary,
  wordUnreached,
} from "./trainingSample";

/** MIRROR of the exporter's `INDEX_SCHEMA` (`ts_transformer/prior/training_files.py`): the airport's index of prior sets.
 *  A name changes with its file's shape, on both sides, in the same change. */
export const TRAINING_PRIOR_INDEX_SCHEMA = "aeroviz-training-prior-index-v3";
/** MIRROR of `INDEX_FILE`: an index of its own beside stage A's `index_v5.json`, which this reader never reads. */
export const TRAINING_PRIOR_INDEX_FILE = "index_prior_v3.json";
/** MIRROR of `SAMPLE_SCHEMA`: a prior set's sample (v3: each prior sentence's track written unrounded, D127). */
export const TRAINING_PRIOR_SAMPLE_SCHEMA = "aeroviz-training-prior-sample-v4";
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

/** The claim of the val read a set was exported under (outline D109): the base's one validation readout. */
export interface TrainingPriorValidationClaim {
  reader: string;
  prior: string;
  readout: string;
}

export interface TrainingPriorSource {
  readout: string;
  /** null for every set but the base's one validation readout, whose flights are of val (D109). */
  validationClaim: TrainingPriorValidationClaim | null;
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

/** One sentence the prior said, flown again by the executor. Rows are the sentence's Δ rows from the first predicted step. */
export interface TrainingPriorSentence {
  /** Which sample of the readout (the prior's seed order); the sentences of one flight differ by it. */
  sample: number;
  words: number[][];
  events: TrainingEvent[];
  firstRow: number;
  startRow: number;
  flownFromRow: number;
  /** The block of the flown sentence (stage A's reader, `parseFlownBlock`): outcome, end cycle, crossing, the flown
   *  flight and the envelopes of its words. */
  outcome: TrainingOutcome;
  endCycle: number;
  crossing: TrainingCrossing | null;
  flown: TrainingTrack;
  envelopes: TrainingEnvelopes | null;
  timedOut: boolean;
  goArounds: number;
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
  const claim = reader.nullableChild("validationClaim");
  return {
    readout: reader.string("readout"),
    validationClaim: claim === null ? null : { reader: claim.string("reader"), prior: claim.string("prior"), readout: claim.string("readout") }, instructions: reader.string("instructions"), executor: reader.string("executor"),
    smoke: reader.boolean("smoke"), git: { head: git.string("head"), dirty: git.boolean("dirty") },
  };
}

/** MIRROR of `instructions.artefact.SEALED_READINGS`: the splits whose readings are read once, in the stage's validation
 *  readout (outline D85) — the splits of a claimed set's flights, and only them (D109). */
export const TRAINING_SEALED_SPLITS = ["val"] as const;
/** MIRROR of `prior.training_files.CLAIM_READER` (the export writes it as ``validationClaim.reader``). */
export const TRAINING_PRIOR_CLAIM_READER = "prior_free_generation";

/** The splits a set's flights may be of (outline D109): the sealed readings alone for the set that holds a claim, stage A's
 *  for every other. A claimed set must be a sealed-split set and a sealed-split set must be claimed; a claim must say what
 *  the export writes (the reader, the set's own prior and readout); anything else is refused by name. */
export function trainingPriorSplits(
  cohort: TrainingPriorCohort, source: TrainingPriorSource, model: TrainingPriorModel, reader: Reader,
): readonly TrainingSplit[] {
  const claim = source.validationClaim;
  const sealed = (TRAINING_SEALED_SPLITS as readonly string[]).includes(cohort.split);
  if (claim !== null && !sealed) reader.fail(`the set holds a validation claim but its cohort.split is "${cohort.split}", not one of ${TRAINING_SEALED_SPLITS}`);
  if (claim === null && sealed) reader.fail(`the set is of split "${cohort.split}" but holds no validationClaim`);
  if (claim === null) return TRAINING_SPLITS;
  if (claim.reader !== TRAINING_PRIOR_CLAIM_READER) reader.fail(`validationClaim.reader is "${claim.reader}", expected "${TRAINING_PRIOR_CLAIM_READER}"`);
  if (claim.prior !== model.prior) reader.fail(`validationClaim.prior is "${claim.prior}", not the set's model.prior "${model.prior}"`);
  if (claim.readout !== source.readout) reader.fail(`validationClaim.readout is "${claim.readout}", not the set's source.readout "${source.readout}"`);
  return TRAINING_SEALED_SPLITS;
}

function parseSetEntry(entry: Reader): TrainingPriorSetEntry {
  entry.oneOf("kind", [TRAINING_PRIOR_SET_KIND]);
  entry.oneOf("readingRule", [TRAINING_READING_RULE]);
  const cohort = parseCohort(entry.child("cohort"));
  const source = parseSource(entry.child("source"));
  const model = parseModel(entry.child("model"));
  trainingPriorSplits(cohort, source, model, entry);
  return {
    id: entry.string("id"), title: entry.string("title"), file: entry.string("file"), flights: entry.count("flights"),
    sentences: entry.count("sentences"), formats: parseFormats(entry), model,
    cohort, source,
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
  const block = parseFlownBlock(reader, TRAINING_OUTCOMES, candidates, cycleS, stepS, firstRow + flownFromRow,
    head.haeMinusMslM, head.observed);
  const rowsSaid = words.length;
  return {
    sample, words, events, firstRow, startRow, flownFromRow, outcome: block.outcome as TrainingOutcome, endCycle: block.endCycle,
    crossing: block.crossing, flown: block.flown, envelopes: block.envelopes, timedOut: reader.boolean("timedOut"),
    goArounds: reader.count("goArounds"),
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
    const cohort = parseCohort(sample.child("cohort"));
    const source = parseSource(sample.child("source"));
    const splits = trainingPriorSplits(cohort, source, model, sample);
    const flights = sample.children("flights").map((flight): TrainingPriorFlight => {
      // a prior set's flights carry the closed-loop sentence at the prior's Δ only
      const head = parseFlight(flight, vocabulary, candidates, cycleS, [model.rowIntervalS], splits);
      const sentences = flight.children("prior").map((item) => parseSentence(item, head, model, candidates, vocabulary, cycleS));
      if (sentences.length === 0) flight.fail("prior holds no sentence");
      return { head, sentences };
    });
    const keys = new Set(flights.map((flight) => flight.head.flightKey));
    if (keys.size !== flights.length) sample.fail("two flights carry one flight key: a flight is its key");
    return {
      setId: sample.string("setId"), airport: sample.string("airport"), executor: { cycleS }, formats: parseFormats(sample),
      vocabulary, airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
      candidates, model, cohort, source,
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
 *  layers read it unchanged: its envelopes the judge's on its flown track (D135); the reading added no word. The executor flies on to the sentence's last word, which can lie past the judge's outcome (a crossing
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
      envelopes: sentence.envelopes,
    },
  };
  const notReached = sentence.events.filter((event) => wordUnreached(result, event.row)).length;
  return { ...result, replay: { ...result.replay, flewTheSentence: notReached === 0, notReached } };
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
