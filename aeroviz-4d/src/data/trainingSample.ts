/**
 * trainingSample.ts
 * -----------------
 * The Training task's data contract: the manifest of exported sample sets and one
 * set's flights (track + sentence + instructions, and later the geometric track).
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.
 *
 * THE SENTENCE IS AN EVENT SEQUENCE, NOT AN EVEN GRID (reading rule `plateau-v11`,
 * 2026-09-20). One row per moment something changed; the gaps between rows are
 * irregular. The even 10 s grid it replaced snapped every instruction forward by
 * 0–8 s, mean 4 s, always late. Two invariants follow and are checked here:
 * `eventTimesS` and `words` have the same length, and `eventTimesS` strictly
 * increases. Neither existed under the grid, and a reader that assumes a fixed
 * step will silently mis-time every instruction it draws.
 *
 * VALIDATION IS PER SET, AND A BAD SET IS REJECTED ALONE. This is a deliberate
 * divergence from the comparison manifest's `.every(isComparisonCategory)`, which
 * empties an entire airport's picker over one bad entry and has cost two debugging
 * sessions (AV6, `docs/35-viewer-reference.md`). Training is a development-time
 * view where a half-written export is normal, so `parseTrainingIndex` keeps the
 * good sets and returns the rejected ones WITH the field that failed.
 */

import { fetchJson } from "../utils/fetchJson";

/** MIRROR of the exporter's schema strings. A file that does not carry these is
 *  refused by name rather than read leniently. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v1";
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v1";

/**
 * MIRROR of `ts_transformer.manoeuvre.instructions.INSTRUCTION_KINDS` — the six
 * word kinds IN ORDER. The columns of `words` are positional, so this order is
 * load-bearing: reorder it and every word is read as another kind's.
 * (The intercept word was deleted on 2026-09-20, D73: it was the heading word's
 * shadow — all 4,897 were issued at the same instant as a heading word.)
 */
export const TRAINING_KINDS = [
  "heading",
  "altitude",
  "speed",
  "runway",
  "duration",
  "terminal",
] as const;

export type TrainingKind = (typeof TRAINING_KINDS)[number];

/** How many columns a `words` row carries. Derived, never typed as 6. */
export const TRAINING_WORD_COLUMNS = TRAINING_KINDS.length;

/** Column index per kind, so callers never count positions by hand. */
export const TRAINING_KIND_COLUMN: Record<TrainingKind, number> = {
  heading: 0,
  altitude: 1,
  speed: 2,
  runway: 3,
  duration: 4,
  terminal: 5,
};

/** MIRROR of `instructions.TERMINAL_CONTINUE / _LANDED / _GO_AROUND`. */
export const TERMINAL_CONTINUE = 0;
export const TERMINAL_LANDED = 1;
export const TERMINAL_GO_AROUND = 2;
/** MIRROR of `instructions.TERMINAL_WORDS`. */
export const TERMINAL_WORDS = 3;

/**
 * `go-around` is a CLASS THAT EXISTS AND IS NEVER OBSERVED in this data: the
 * vocabulary is read on the 25 km arrival slice, which keeps only the final
 * successful approach (results §13.5 — go-arounds are real in the fleet, 42 of
 * 44,622 tracks, but cannot appear here). A legend may list it; it must say the
 * class is never observed rather than implying the model declines to use it.
 */
export const TERMINAL_NEVER_OBSERVED: readonly number[] = [TERMINAL_GO_AROUND];

/** MIRROR of `instructions.ABSORBED_*`: why a manoeuvre was read but not worded. */
export const ABSORBED_REASONS = [
  "same word",
  "small change",
  "short tail",
] as const;
export type AbsorbedReason = (typeof ABSORBED_REASONS)[number];

/** The two kinds of sample set the exporter writes. */
export const TRAINING_SET_KINDS = ["vocabulary-readback", "prior-generated"] as const;
export type TrainingSetKind = (typeof TRAINING_SET_KINDS)[number];

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingSetEntry {
  id: string;
  kind: TrainingSetKind;
  title: string;
  /** Path of the sample file, relative to the airport's `training/` directory. */
  file: string;
  /** The vocabulary SPEC's sha. Not the runway classes' — see `runwaySha256`. */
  vocabularySha256: string;
  /**
   * The runway classes' OWN sha. The class set is per airport and is carried
   * BESIDE the spec (`runway_idents`), so it does not move `vocabularySha256`:
   * two artefacts with the same spec sha can carry different runway lists.
   * Comparing only one of the two and calling it "the same vocabulary" is wrong.
   */
  runwaySha256: string;
  readingRule: string;
  flights: number;
}

export interface TrainingIndex {
  airport: string;
  sets: TrainingSetEntry[];
  /** Sets that failed validation, each with the field that failed. Kept so the
   *  UI can grey one out by name instead of emptying the list (AV6). */
  rejected: Array<{ id: string; problem: string }>;
}

export interface TrainingVocabulary {
  sha256: string;
  runwaySha256: string;
  readingRule: string;
  headingBinDeg: number;
  altitudeBinM: number;
  altitudeMaxM: number;
  speedBinMps: number;
  speedMinMps: number;
  speedMaxMps: number;
  durationBinS: number;
  durationMaxS: number;
  /** The vocabulary's OWN runway classes. NEVER the airport's runway list: at
   *  KRDU the airport has six thresholds and the vocabulary four (05L 05R 23L
   *  23R), because the arrival manifest this line is built on carries only those
   *  four. Drawing six would say the model can name a runway it cannot. */
  runwayIdents: string[];
}

export interface TrainingSentence {
  /** [E] the moments something changed — strictly increasing, irregular gaps. */
  eventTimesS: number[];
  /** [E][6] the words in force at each event, in `TRAINING_KINDS` order. */
  words: number[][];
  /** How many gaps were clamped at the duration ceiling. Stated, never silent. */
  durationClamped: number;
}

export interface TrainingFlight {
  flightKey: string;
  callsign: string;
  runway: string;
  stratum: string;
  sentence: TrainingSentence;
}

export interface TrainingSample {
  setId: string;
  airport: string;
  vocabulary: TrainingVocabulary;
  flights: TrainingFlight[];
}

export type Parsed<T> = { ok: true; value: T } | { ok: false; problem: string };

// ── word counts ──────────────────────────────────────────────────────────────

/**
 * How many classes each kind has, under this file's own spec.
 *
 * MIRROR of `Vocabulary.heading_words / altitude_words / speed_words /
 * duration_words` and `word_counts()`: the same formulas, so a word index out of
 * range is caught here instead of indexing a legend off its end. Computing them
 * from the file's spec (rather than hardcoding 36/11/23/151) is what lets one
 * reader serve a re-binned vocabulary and every airport's runway list.
 *
 * The range check earns its keep on the POSITIONAL columns: swap two and a
 * duration word (0–150) lands in the runway column (0–3) and fails loudly,
 * instead of drawing the wrong runway for the whole flight.
 */
export function trainingWordCounts(
  vocabulary: TrainingVocabulary,
): Record<TrainingKind, number> {
  return {
    heading: Math.round(360 / vocabulary.headingBinDeg),
    altitude: Math.round(vocabulary.altitudeMaxM / vocabulary.altitudeBinM) + 1,
    speed:
      Math.round(
        (vocabulary.speedMaxMps - vocabulary.speedMinMps) / vocabulary.speedBinMps,
      ) + 1,
    runway: vocabulary.runwayIdents.length,
    duration: Math.round(vocabulary.durationMaxS / vocabulary.durationBinS) + 1,
    terminal: TERMINAL_WORDS,
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
  const out: number[] = [];
  for (const item of value) {
    if (typeof item !== "number" || !Number.isFinite(item)) return null;
    out.push(item);
  }
  return out;
}

// ── the index ────────────────────────────────────────────────────────────────

function parseSetEntry(raw: unknown, position: number): Parsed<TrainingSetEntry> {
  if (!isRecord(raw)) return { ok: false, problem: `sets[${position}] is not an object` };
  const id = str(raw, "id") ?? `sets[${position}]`;
  const where = (field: string) => `set ${id}: ${field}`;

  const kind = str(raw, "kind");
  if (kind === null || !(TRAINING_SET_KINDS as readonly string[]).includes(kind)) {
    return {
      ok: false,
      problem: `${where("kind")} is ${JSON.stringify(raw.kind)}, expected one of ${TRAINING_SET_KINDS.join(", ")}`,
    };
  }
  for (const field of ["id", "title", "file", "vocabularySha256", "runwaySha256", "readingRule"]) {
    if (str(raw, field) === null) {
      return { ok: false, problem: `${where(field)} is missing or not a non-empty string` };
    }
  }
  const flights = finite(raw, "flights");
  if (flights === null || flights < 0) {
    return { ok: false, problem: `${where("flights")} is missing or not a count` };
  }

  return {
    ok: true,
    value: {
      id: str(raw, "id") as string,
      kind: kind as TrainingSetKind,
      title: str(raw, "title") as string,
      file: str(raw, "file") as string,
      vocabularySha256: str(raw, "vocabularySha256") as string,
      runwaySha256: str(raw, "runwaySha256") as string,
      readingRule: str(raw, "readingRule") as string,
      flights,
    },
  };
}

/** Parse the manifest. A bad SET is rejected on its own; only a manifest that is
 *  not a manifest at all fails the whole call. */
export function parseTrainingIndex(raw: unknown): Parsed<TrainingIndex> {
  if (!isRecord(raw)) return { ok: false, problem: "the manifest is not an object" };
  if (raw.schema !== TRAINING_INDEX_SCHEMA) {
    return {
      ok: false,
      problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_INDEX_SCHEMA)}`,
    };
  }
  const airport = str(raw, "airport");
  if (airport === null) return { ok: false, problem: "airport is missing" };
  if (!Array.isArray(raw.sets)) return { ok: false, problem: "sets is not an array" };

  const sets: TrainingSetEntry[] = [];
  const rejected: Array<{ id: string; problem: string }> = [];
  raw.sets.forEach((entry, position) => {
    const parsed = parseSetEntry(entry, position);
    if (parsed.ok) sets.push(parsed.value);
    else {
      const id = isRecord(entry) && str(entry, "id") ? (str(entry, "id") as string) : `sets[${position}]`;
      rejected.push({ id, problem: parsed.problem });
    }
  });
  return { ok: true, value: { airport, sets, rejected } };
}

// ── one sample set ───────────────────────────────────────────────────────────

function parseVocabulary(raw: unknown): Parsed<TrainingVocabulary> {
  if (!isRecord(raw)) return { ok: false, problem: "vocabulary is not an object" };
  const numbers: Array<keyof TrainingVocabulary> = [
    "headingBinDeg", "altitudeBinM", "altitudeMaxM",
    "speedBinMps", "speedMinMps", "speedMaxMps",
    "durationBinS", "durationMaxS",
  ];
  const values: Record<string, number> = {};
  for (const field of numbers) {
    const value = finite(raw, field);
    if (value === null) return { ok: false, problem: `vocabulary.${field} is missing or not a number` };
    values[field] = value;
  }
  for (const field of ["headingBinDeg", "altitudeBinM", "speedBinMps", "durationBinS"]) {
    if (values[field] <= 0) return { ok: false, problem: `vocabulary.${field} must be positive` };
  }
  for (const field of ["sha256", "runwaySha256", "readingRule"]) {
    if (str(raw, field) === null) {
      return { ok: false, problem: `vocabulary.${field} is missing or not a non-empty string` };
    }
  }
  const idents = raw.runwayIdents;
  if (!Array.isArray(idents) || idents.length === 0 || !idents.every((i) => typeof i === "string" && i.length > 0)) {
    return { ok: false, problem: "vocabulary.runwayIdents is missing or not a non-empty list of runway names" };
  }

  return {
    ok: true,
    value: {
      sha256: str(raw, "sha256") as string,
      runwaySha256: str(raw, "runwaySha256") as string,
      readingRule: str(raw, "readingRule") as string,
      headingBinDeg: values.headingBinDeg,
      altitudeBinM: values.altitudeBinM,
      altitudeMaxM: values.altitudeMaxM,
      speedBinMps: values.speedBinMps,
      speedMinMps: values.speedMinMps,
      speedMaxMps: values.speedMaxMps,
      durationBinS: values.durationBinS,
      durationMaxS: values.durationMaxS,
      runwayIdents: idents as string[],
    },
  };
}

function parseSentence(
  raw: unknown,
  counts: Record<TrainingKind, number>,
  where: string,
): Parsed<TrainingSentence> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.sentence is not an object` };

  const eventTimesS = numberArray(raw.eventTimesS);
  if (eventTimesS === null) {
    return { ok: false, problem: `${where}.sentence.eventTimesS is missing or not a list of numbers` };
  }
  if (eventTimesS.length === 0) {
    return { ok: false, problem: `${where}.sentence.eventTimesS is empty: a flight with no event has no sentence` };
  }
  if (!Array.isArray(raw.words)) {
    return { ok: false, problem: `${where}.sentence.words is missing or not an array` };
  }
  if (raw.words.length !== eventTimesS.length) {
    return {
      ok: false,
      problem: `${where}.sentence has ${eventTimesS.length} event times but ${raw.words.length} word rows — one row per event`,
    };
  }
  // The event sequence's own invariant: each event is a later moment than the one
  // before it. Under the retired even grid this could not be violated; now it can,
  // and an unsorted sentence would draw bands backwards and mis-time every jump.
  for (let i = 1; i < eventTimesS.length; i += 1) {
    if (!(eventTimesS[i] > eventTimesS[i - 1])) {
      return {
        ok: false,
        problem:
          `${where}.sentence.eventTimesS is not strictly increasing at index ${i} ` +
          `(${eventTimesS[i - 1]} then ${eventTimesS[i]}) — the sentence is an event sequence`,
      };
    }
  }

  const words: number[][] = [];
  for (let row = 0; row < raw.words.length; row += 1) {
    const parsedRow = numberArray(raw.words[row]);
    if (parsedRow === null) {
      return { ok: false, problem: `${where}.sentence.words[${row}] is not a list of numbers` };
    }
    if (parsedRow.length !== TRAINING_WORD_COLUMNS) {
      return {
        ok: false,
        problem:
          `${where}.sentence.words[${row}] has ${parsedRow.length} columns, expected ${TRAINING_WORD_COLUMNS} ` +
          `(${TRAINING_KINDS.join(", ")})`,
      };
    }
    for (const kind of TRAINING_KINDS) {
      const value = parsedRow[TRAINING_KIND_COLUMN[kind]];
      if (!Number.isInteger(value) || value < 0 || value >= counts[kind]) {
        return {
          ok: false,
          problem:
            `${where}.sentence.words[${row}].${kind} is ${value}, outside this vocabulary's ` +
            `0…${counts[kind] - 1} — the columns are positional, so check their ORDER first`,
        };
      }
    }
    words.push(parsedRow);
  }

  const durationClamped = finite(raw, "durationClamped");
  if (durationClamped === null || durationClamped < 0) {
    return { ok: false, problem: `${where}.sentence.durationClamped is missing or not a count` };
  }

  return { ok: true, value: { eventTimesS, words, durationClamped } };
}

/** Parse one sample set. Unlike the manifest this is all-or-nothing: a flight the
 *  reader cannot trust would be drawn beside real ones with no way to tell. */
export function parseTrainingSample(raw: unknown): Parsed<TrainingSample> {
  if (!isRecord(raw)) return { ok: false, problem: "the sample is not an object" };
  if (raw.schema !== TRAINING_SAMPLE_SCHEMA) {
    return {
      ok: false,
      problem: `schema is ${JSON.stringify(raw.schema)}, expected ${JSON.stringify(TRAINING_SAMPLE_SCHEMA)}`,
    };
  }
  const setId = str(raw, "setId");
  const airport = str(raw, "airport");
  if (setId === null) return { ok: false, problem: "setId is missing" };
  if (airport === null) return { ok: false, problem: "airport is missing" };

  const vocabulary = parseVocabulary(raw.vocabulary);
  if (!vocabulary.ok) return vocabulary;

  // `kinds`, when present, is the exporter restating the column order. It must
  // agree with ours exactly — a disagreement means the columns moved.
  if (raw.kinds !== undefined) {
    const kinds = Array.isArray(raw.kinds) ? raw.kinds.join(",") : String(raw.kinds);
    if (kinds !== TRAINING_KINDS.join(",")) {
      return {
        ok: false,
        problem: `kinds is [${kinds}], expected [${TRAINING_KINDS.join(",")}] in that order`,
      };
    }
  }

  if (!Array.isArray(raw.flights)) return { ok: false, problem: "flights is not an array" };
  const counts = trainingWordCounts(vocabulary.value);
  const flights: TrainingFlight[] = [];
  for (let i = 0; i < raw.flights.length; i += 1) {
    const entry = raw.flights[i];
    if (!isRecord(entry)) return { ok: false, problem: `flights[${i}] is not an object` };
    const flightKey = str(entry, "flightKey");
    const where = flightKey ? `flight ${flightKey}` : `flights[${i}]`;
    if (flightKey === null) return { ok: false, problem: `${where}: flightKey is missing` };
    const runway = str(entry, "runway");
    if (runway === null) return { ok: false, problem: `${where}: runway is missing` };
    if (!vocabulary.value.runwayIdents.includes(runway)) {
      return {
        ok: false,
        problem:
          `${where}: runway ${runway} is not one of this vocabulary's classes ` +
          `(${vocabulary.value.runwayIdents.join(", ")})`,
      };
    }
    const sentence = parseSentence(entry.sentence, counts, where);
    if (!sentence.ok) return sentence;
    flights.push({
      flightKey,
      callsign: str(entry, "callsign") ?? flightKey,
      runway,
      stratum: str(entry, "stratum") ?? "unknown",
      sentence: sentence.value,
    });
  }

  return { ok: true, value: { setId, airport, vocabulary: vocabulary.value, flights } };
}

// ── where the files live ─────────────────────────────────────────────────────

/** The airport's Training directory, relative to the site root. */
export function trainingDirectory(airportCode: string): string {
  return `data/airports/${airportCode}/training`;
}

/** The manifest the panel reads. One definition — shown to the user in the empty
 *  state and used by the fetch below. */
export function trainingIndexPath(airportCode: string): string {
  return `${trainingDirectory(airportCode)}/index.json`;
}

/** A set's sample file. `file` is relative to the airport's Training directory. */
export function trainingSamplePath(airportCode: string, file: string): string {
  return `${trainingDirectory(airportCode)}/${file}`;
}

export async function fetchTrainingIndex(airportCode: string): Promise<Parsed<TrainingIndex>> {
  const raw = await fetchJson<unknown>(trainingIndexPath(airportCode));
  return parseTrainingIndex(raw);
}

export async function fetchTrainingSample(
  airportCode: string,
  file: string,
): Promise<Parsed<TrainingSample>> {
  const raw = await fetchJson<unknown>(trainingSamplePath(airportCode, file));
  return parseTrainingSample(raw);
}
