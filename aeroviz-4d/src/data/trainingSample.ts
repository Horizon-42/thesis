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
// The unit constants are the GENERATED mirror of geokit (the one exception the
// repo allows the frontend). A feet or knots factor typed here would be a second
// definition of a geodetic constant.
import { FEET_TO_METERS, metresPerSecondToKnots } from "../utils/procedureGeoMath";

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
  /** The class count the ARTEFACT states per kind (`word_counts` on the Python
   *  side). `trainingWordCounts` derives the same numbers from the bins, and
   *  `parseVocabulary` refuses a file where the two disagree — that is what makes
   *  the derivation a checked mirror rather than a second opinion. */
  words: Record<TrainingKind, number>;
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

/**
 * One instruction as the labeller issued it. Only the three geometric kinds and
 * the runway are ever issued: the duration and terminal words are read off the
 * EVENT SEQUENCE, not off this list, so they never appear here (that is why a
 * view must count words on `sentence`, never on `instructions`).
 */
export interface TrainingInstruction {
  kind: TrainingKind;
  word: number;
  /** The value the plateau settled at, unbinned, in the kind's own unit: degrees
   *  relative to the final approach course, metres above the threshold, m/s
   *  ground speed. The runway instruction has no target at all (its WORD is the
   *  answer) and the exporter writes `instructions.NO_TARGET`, 0.0, there — so
   *  nothing may print a target for a runway word. */
  target: number;
  issuedS: number;
  /** When the manoeuvre finished, or `null` when it never settled inside the
   *  track — the last altitude instruction usually runs to the threshold. Null is
   *  a real answer here, not a missing field. */
  settledS: number | null;
  /** Whether the target fell outside the vocabulary's range and was clamped. */
  clamped: boolean;
}

/**
 * MIRROR of `instructions.course_frame` — the observed track in the FINAL
 * APPROACH COURSE's frame, which is the frame the words were read in. Reading
 * the words against anything else (a geodetic track, say) would let the charts
 * and the words disagree about where the aircraft was.
 *
 * Every column has one value per row of `tS`, and the geodetic columns are
 * deliberately absent until T5/T6 puts the tracks in the 3D scene.
 */
export const TRAINING_OBSERVED_COLUMNS = [
  "toGoM",              // along the course, positive BEFORE the threshold
  "crossM",             // right of the course, positive
  "heightM",            // above the threshold
  "relCourseDeg",       // ground track against the course, wrapped
  "courseUnwrappedDeg", // the same, unwrapped along the rows
  "groundSpeedMps",
  "established",        // 0 / 1 per row
] as const;

export type TrainingObservedColumn = (typeof TRAINING_OBSERVED_COLUMNS)[number];

export type TrainingObserved = { tS: number[] } & Record<TrainingObservedColumn, number[]>;

/**
 * The columns of the track flown FROM THE WORDS (`manoeuvre/instruction_kinematics.py`),
 * in the same runway frame as the observed one. Geodetic columns are not here:
 * the plan view and the "how far apart now" readout are frame quantities, and
 * the 3D layer's input arrives with its own vertical datum (design V27).
 */
export const TRAINING_GEOMETRIC_COLUMNS = [
  "toGoM", "crossM", "heightM", "groundSpeedMps", "relCourseDeg",
] as const;

export type TrainingGeometricColumn = (typeof TRAINING_GEOMETRIC_COLUMNS)[number];

/** MIRROR of `instruction_kinematics.END_CROSSED / END_TIME_CAP`. */
export const TRAINING_END_REASONS = ["crossed-threshold", "time-cap"] as const;
export type TrainingEndReason = (typeof TRAINING_END_REASONS)[number];

export type TrainingGeometric = { tS: number[] } & Record<TrainingGeometricColumn, number[]> & {
  endReason: TrainingEndReason;
  /** The horizontal distance from the threshold where it stopped — `hypot(toGo,
   *  cross)`, so a track that crosses the plane two kilometres to the side
   *  reports two kilometres. The words are never extended to reach the runway. */
  finalGapM: number;
  /** The distance between the two tracks, over the time BOTH were flying. */
  meanGapM: number;
  gapP95M: number;
  /** How much of the approach that comparison covered. A mean gap read without
   *  it is a mean over an unstated window. */
  comparedS: number;
  comparedFraction: number;
};

/** A manoeuvre the labeller read but did not word, and why. Drawn on its kind's
 *  row so "the words miss this turn" is visible rather than argued about. */
export interface TrainingAbsorbed {
  kind: TrainingKind;
  startS: number;
  endS: number;
  word: number;
  change: number;
  reason: AbsorbedReason;
}

export interface TrainingFlight {
  flightKey: string;
  callsign: string;
  runway: string;
  stratum: string;
  /**
   * The TRACK's length — not the sentence's. The last event sits well before it
   * (median 145 s of a 326 s arrival in KRDU's export): the words in force at the
   * last event are held to the threshold. Every row's last band therefore runs to
   * `durationS`, and a bar that stopped at the last event would draw 44 % of the
   * approach as if nothing were being flown.
   */
  durationS: number;
  establishedFromStart: boolean;
  sentence: TrainingSentence;
  instructions: TrainingInstruction[];
  absorbed: TrainingAbsorbed[];
  observed: TrainingObserved;
  geometric: TrainingGeometric;
}

/** What the panel publishes for the full-width sentence bar to draw: one flight
 *  and the vocabulary its words are read under. */
export interface TrainingSelection {
  vocabulary: TrainingVocabulary;
  flight: TrainingFlight;
}

export interface TrainingSample {
  setId: string;
  airport: string;
  vocabulary: TrainingVocabulary;
  flights: TrainingFlight[];
}

/** The part of a vocabulary that decides what a word MEANS: the bins and the
 *  runway classes. Everything that reads words takes this, so `parseVocabulary`
 *  can derive the class counts while it is still building the vocabulary. */
export type TrainingWordSpec = Pick<
  TrainingVocabulary,
  | "headingBinDeg" | "altitudeBinM" | "altitudeMaxM"
  | "speedBinMps" | "speedMinMps" | "speedMaxMps"
  | "durationBinS" | "durationMaxS" | "runwayIdents"
>;

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
  vocabulary: TrainingWordSpec,
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

// ── what a word means ────────────────────────────────────────────────────────

/**
 * MIRROR of `ts_transformer.data.runway_context.wrap_deg` — the SAME half-open
 * range [-180, 180), so heading word 18 reads -180°, exactly as the labeller
 * wrote it. A wrap of the other convention would flip that one word's sign.
 */
export function wrapDeg(degrees: number): number {
  // The doubled modulo is not decoration: JS `%` truncates where Python's floors,
  // so the single-modulo spelling returns -190 for -190 and 0 stays 0 only by luck.
  // Today's one caller passes 0…350, but the signed relative course the artefact
  // carries is the obvious next caller.
  return (((degrees + 180) % 360) + 360) % 360 - 180;
}

/** MIRROR of `Vocabulary.heading_centre_deg`: degrees relative to the final
 *  approach course, 0 = on the course. */
export function headingCentreDeg(vocabulary: TrainingWordSpec, word: number): number {
  return wrapDeg(word * vocabulary.headingBinDeg);
}

/** MIRROR of `Vocabulary.altitude_centre_m`: metres ABOVE THE THRESHOLD. */
export function altitudeCentreM(vocabulary: TrainingWordSpec, word: number): number {
  return word * vocabulary.altitudeBinM;
}

/** MIRROR of `Vocabulary.speed_centre_mps`: ground speed. */
export function speedCentreMps(vocabulary: TrainingWordSpec, word: number): number {
  return vocabulary.speedMinMps + word * vocabulary.speedBinMps;
}

/** MIRROR of `Vocabulary.duration_centre_s`: the gap to the PREVIOUS event. */
export function durationCentreS(vocabulary: TrainingWordSpec, word: number): number {
  return word * vocabulary.durationBinS;
}

/**
 * A time from this artefact, as every Training view writes it. The rows are on a
 * 2 s grid so most times are whole seconds; a scrubbed cursor is not, and the
 * views must not disagree about whether the same moment is 201.9 s or 202 s.
 */
export function formatSeconds(seconds: number): string {
  return Number.isInteger(seconds) ? `${seconds}` : seconds.toFixed(1);
}

/** The terminal words, by index (`TERMINAL_CONTINUE / _LANDED / _GO_AROUND`). */
export const TERMINAL_LABELS = ["continue", "landed", "go-around"] as const;

/**
 * A word as a person reads it.
 *
 * The bins are DEFINED in feet and knots (1000 ft, 10 kt) and stored in SI, so
 * printing the stored metres would label every altitude 304.8 m and every speed
 * 5.14 m/s apart — arithmetic the reader would have to undo to recognise the
 * vocabulary. The conversion uses the generated geokit constants, never a factor
 * typed here.
 */
export function trainingWordLabel(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): string {
  switch (kind) {
    case "heading": {
      const degrees = Math.round(headingCentreDeg(vocabulary, word));
      return `${degrees > 0 ? "+" : ""}${degrees}\u00b0`;
    }
    case "altitude":
      return `${Math.round(altitudeCentreM(vocabulary, word) / FEET_TO_METERS)} ft`;
    case "speed":
      return `${Math.round(metresPerSecondToKnots(speedCentreMps(vocabulary, word)))} kt`;
    case "runway":
      return vocabulary.runwayIdents[word];
    case "duration":
      return `${durationCentreS(vocabulary, word)} s`;
    case "terminal":
      return TERMINAL_LABELS[word];
  }
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

  const stated = raw.words;
  if (!isRecord(stated)) {
    return { ok: false, problem: "vocabulary.words is missing: the artefact states its own class counts" };
  }
  const spec: TrainingWordSpec = {
    headingBinDeg: values.headingBinDeg,
    altitudeBinM: values.altitudeBinM,
    altitudeMaxM: values.altitudeMaxM,
    speedBinMps: values.speedBinMps,
    speedMinMps: values.speedMinMps,
    speedMaxMps: values.speedMaxMps,
    durationBinS: values.durationBinS,
    durationMaxS: values.durationMaxS,
    runwayIdents: idents as string[],
  };
  const words = {} as Record<TrainingKind, number>;
  const derived = trainingWordCounts(spec);
  for (const kind of TRAINING_KINDS) {
    const count = finite(stated, kind);
    if (count === null || !Number.isInteger(count) || count <= 0) {
      return { ok: false, problem: `vocabulary.words.${kind} is ${JSON.stringify(stated[kind])}, expected a class count` };
    }
    // The counts we derive from the bins and the counts the file states are two
    // spellings of `Vocabulary.words`. They agreeing is the whole value of the
    // derivation; disagreeing means one of the two mirrors has drifted, and
    // guessing which would put every word in the wrong legend.
    if (count !== derived[kind]) {
      return {
        ok: false,
        problem:
          `vocabulary.words.${kind} is ${count}, but this file's own bins give ${derived[kind]} — ` +
          `the stated counts and the spec disagree`,
      };
    }
    words[kind] = count;
  }

  return {
    ok: true,
    value: {
      sha256: str(raw, "sha256") as string,
      runwaySha256: str(raw, "runwaySha256") as string,
      readingRule: str(raw, "readingRule") as string,
      ...spec,
      words,
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

function parseObserved(raw: unknown, durationS: number, where: string): Parsed<TrainingObserved> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.observed is not an object` };
  const tS = numberArray(raw.tS);
  if (tS === null || tS.length < 2) {
    return { ok: false, problem: `${where}.observed.tS is missing or shorter than two rows` };
  }
  // The track's clock is the flight's clock: it starts at 0 and it ends where
  // `durationS` says the flight ends, because both come from the same rows. A
  // disagreement means the track and the sentence are not the same flight — the
  // same check the last event already gets.
  if (tS[0] !== 0) {
    return { ok: false, problem: `${where}.observed.tS starts at ${tS[0]} s, not 0` };
  }
  // Compared at the export's own precision: `tS` is written rounded to 0.1 s and
  // `durationS` is not, so an exact test would one day refuse a whole file over a
  // rounding difference while blaming it on the track and the sentence being
  // different flights.
  if (Math.abs(tS[tS.length - 1] - durationS) > 0.05) {
    return {
      ok: false,
      problem: `${where}.observed.tS ends at ${tS[tS.length - 1]} s but the flight is ${durationS} s long`,
    };
  }
  for (let i = 1; i < tS.length; i += 1) {
    if (!(tS[i] > tS[i - 1])) {
      return { ok: false, problem: `${where}.observed.tS is not increasing at row ${i}` };
    }
  }

  const columns: Record<string, number[]> = { tS };
  for (const column of TRAINING_OBSERVED_COLUMNS) {
    const values = numberArray(raw[column]);
    if (values === null) {
      return { ok: false, problem: `${where}.observed.${column} is missing or not a list of numbers` };
    }
    if (values.length !== tS.length) {
      return {
        ok: false,
        problem: `${where}.observed.${column} has ${values.length} rows, but tS has ${tS.length}`,
      };
    }
    columns[column] = values;
  }
  if (!columns.established.every((value) => value === 0 || value === 1)) {
    return { ok: false, problem: `${where}.observed.established is not 0/1 per row` };
  }

  return { ok: true, value: columns as TrainingObserved };
}

function parseGeometric(raw: unknown, where: string): Parsed<TrainingGeometric> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.geometric is not an object` };
  const tS = numberArray(raw.tS);
  if (tS === null || tS.length < 2) {
    return { ok: false, problem: `${where}.geometric.tS is missing or shorter than two steps` };
  }
  const columns: Record<string, number[]> = { tS };
  for (const column of TRAINING_GEOMETRIC_COLUMNS) {
    const values = numberArray(raw[column]);
    if (values === null || values.length !== tS.length) {
      return {
        ok: false,
        problem: `${where}.geometric.${column} is missing or does not have ${tS.length} rows`,
      };
    }
    columns[column] = values;
  }

  const endReason = str(raw, "endReason");
  if (endReason === null || !(TRAINING_END_REASONS as readonly string[]).includes(endReason)) {
    return {
      ok: false,
      problem: `${where}.geometric.endReason is ${JSON.stringify(raw.endReason)}, expected one of ${TRAINING_END_REASONS.join(", ")}`,
    };
  }
  const numbers: Record<string, number> = {};
  for (const field of ["finalGapM", "meanGapM", "gapP95M", "comparedS", "comparedFraction"]) {
    const value = finite(raw, field);
    if (value === null || value < 0) {
      return { ok: false, problem: `${where}.geometric.${field} is missing or not a distance` };
    }
    numbers[field] = value;
  }
  // The gap is a mean over the window both tracks were flying, and the fraction
  // is what says how much of the approach that was. A fraction over 1 would mean
  // the comparison outran the observation it is a fraction OF.
  if (numbers.comparedFraction > 1) {
    return { ok: false, problem: `${where}.geometric.comparedFraction is ${numbers.comparedFraction}, over the whole approach` };
  }

  return {
    ok: true,
    value: {
      ...(columns as { tS: number[] } & Record<TrainingGeometricColumn, number[]>),
      endReason: endReason as TrainingEndReason,
      finalGapM: numbers.finalGapM,
      meanGapM: numbers.meanGapM,
      gapP95M: numbers.gapP95M,
      comparedS: numbers.comparedS,
      comparedFraction: numbers.comparedFraction,
    },
  };
}

function parseKind(raw: Record<string, unknown>, where: string): Parsed<TrainingKind> {
  const kind = str(raw, "kind");
  if (kind === null || !(TRAINING_KINDS as readonly string[]).includes(kind)) {
    return { ok: false, problem: `${where}.kind is ${JSON.stringify(raw.kind)}, expected one of ${TRAINING_KINDS.join(", ")}` };
  }
  return { ok: true, value: kind as TrainingKind };
}

function parseInstruction(
  raw: unknown,
  counts: Record<TrainingKind, number>,
  where: string,
): Parsed<TrainingInstruction> {
  if (!isRecord(raw)) return { ok: false, problem: `${where} is not an object` };
  const kind = parseKind(raw, where);
  if (!kind.ok) return kind;

  const word = finite(raw, "word");
  if (word === null || !Number.isInteger(word) || word < 0 || word >= counts[kind.value]) {
    return { ok: false, problem: `${where}.word is ${JSON.stringify(raw.word)}, outside this vocabulary's 0…${counts[kind.value] - 1} for ${kind.value}` };
  }
  const target = finite(raw, "target");
  if (target === null) return { ok: false, problem: `${where}.target is missing or not a number` };
  const issuedS = finite(raw, "issuedS");
  if (issuedS === null) return { ok: false, problem: `${where}.issuedS is missing or not a number` };
  // `settledS` is `number | null` and the null is MEANINGFUL (it never settled
  // inside the track). A missing key is a different thing and is refused.
  if (!("settledS" in raw)) return { ok: false, problem: `${where}.settledS is missing (null means it never settled; absent means the field moved)` };
  const settled = raw.settledS;
  if (settled !== null && (typeof settled !== "number" || !Number.isFinite(settled))) {
    return { ok: false, problem: `${where}.settledS is ${JSON.stringify(settled)}, expected a number or null` };
  }
  if (typeof raw.clamped !== "boolean") {
    return { ok: false, problem: `${where}.clamped is ${JSON.stringify(raw.clamped)}, expected a boolean` };
  }

  return {
    ok: true,
    value: { kind: kind.value, word, target, issuedS, settledS: settled as number | null, clamped: raw.clamped },
  };
}

function parseAbsorbed(
  raw: unknown,
  counts: Record<TrainingKind, number>,
  where: string,
): Parsed<TrainingAbsorbed> {
  if (!isRecord(raw)) return { ok: false, problem: `${where} is not an object` };
  const kind = parseKind(raw, where);
  if (!kind.ok) return kind;

  const word = finite(raw, "word");
  if (word === null || !Number.isInteger(word) || word < 0 || word >= counts[kind.value]) {
    return { ok: false, problem: `${where}.word is ${JSON.stringify(raw.word)}, outside this vocabulary's 0…${counts[kind.value] - 1} for ${kind.value}` };
  }
  const startS = finite(raw, "startS");
  const endS = finite(raw, "endS");
  if (startS === null || endS === null) {
    return { ok: false, problem: `${where} needs numeric startS and endS` };
  }
  if (endS < startS) {
    return { ok: false, problem: `${where} ends at ${endS} s before it starts at ${startS} s` };
  }
  const change = finite(raw, "change");
  if (change === null) return { ok: false, problem: `${where}.change is missing or not a number` };
  const reason = str(raw, "reason");
  if (reason === null || !(ABSORBED_REASONS as readonly string[]).includes(reason)) {
    return { ok: false, problem: `${where}.reason is ${JSON.stringify(raw.reason)}, expected one of ${ABSORBED_REASONS.join(", ")}` };
  }

  return {
    ok: true,
    value: { kind: kind.value, word, startS, endS, change, reason: reason as AbsorbedReason },
  };
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

  // `kinds` is the exporter restating the column order, and it must agree with
  // ours exactly — a disagreement means the columns moved. Every export writes
  // it, so an absent one is a file from something else, not an older file to be
  // read leniently.
  const kinds = Array.isArray(raw.kinds) ? raw.kinds.join(",") : String(raw.kinds);
  if (kinds !== TRAINING_KINDS.join(",")) {
    return {
      ok: false,
      problem: `kinds is [${kinds}], expected [${TRAINING_KINDS.join(",")}] in that order`,
    };
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
    const callsign = str(entry, "callsign");
    if (callsign === null) return { ok: false, problem: `${where}: callsign is missing` };
    const stratum = str(entry, "stratum");
    if (stratum === null) return { ok: false, problem: `${where}: stratum is missing` };
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

    const durationS = finite(entry, "durationS");
    if (durationS === null || durationS <= 0) {
      return { ok: false, problem: `${where}: durationS is missing or not a positive length` };
    }
    // The bands run to the end of the TRACK, so a sentence whose last event is
    // past it would draw off the axis. This also catches a sample whose sentence
    // and track came from different flights.
    const lastEventS = sentence.value.eventTimesS[sentence.value.eventTimesS.length - 1];
    if (lastEventS > durationS) {
      return {
        ok: false,
        problem: `${where}: the last event is at ${lastEventS} s but the track is ${durationS} s long — the sentence and the track are not the same flight`,
      };
    }
    if (typeof entry.establishedFromStart !== "boolean") {
      return { ok: false, problem: `${where}: establishedFromStart is ${JSON.stringify(entry.establishedFromStart)}, expected a boolean` };
    }

    if (!Array.isArray(entry.instructions)) {
      return { ok: false, problem: `${where}: instructions is not an array` };
    }
    const instructions: TrainingInstruction[] = [];
    for (let k = 0; k < entry.instructions.length; k += 1) {
      const parsed = parseInstruction(entry.instructions[k], counts, `${where}: instructions[${k}]`);
      if (!parsed.ok) return parsed;
      // Same reason as the last event's check: a time outside the track is a
      // different flight's, and it draws off the plot rather than failing.
      const outside = [parsed.value.issuedS, parsed.value.settledS].find(
        (time) => time !== null && (time < 0 || time > durationS),
      );
      if (outside !== undefined) {
        return { ok: false, problem: `${where}: instructions[${k}] is at ${outside} s, outside the ${durationS} s track` };
      }
      instructions.push(parsed.value);
    }

    if (!Array.isArray(entry.absorbed)) {
      return { ok: false, problem: `${where}: absorbed is not an array` };
    }
    const absorbed: TrainingAbsorbed[] = [];
    for (let k = 0; k < entry.absorbed.length; k += 1) {
      const parsed = parseAbsorbed(entry.absorbed[k], counts, `${where}: absorbed[${k}]`);
      if (!parsed.ok) return parsed;
      if (parsed.value.startS < 0 || parsed.value.endS > durationS) {
        return {
          ok: false,
          problem:
            `${where}: absorbed[${k}] spans ${parsed.value.startS}–${parsed.value.endS} s, ` +
            `outside the ${durationS} s track`,
        };
      }
      absorbed.push(parsed.value);
    }

    const observed = parseObserved(entry.observed, durationS, where);
    if (!observed.ok) return observed;
    const geometric = parseGeometric(entry.geometric, where);
    if (!geometric.ok) return geometric;

    flights.push({
      flightKey,
      callsign,
      runway,
      stratum,
      durationS,
      establishedFromStart: entry.establishedFromStart,
      sentence: sentence.value,
      instructions,
      absorbed,
      observed: observed.value,
      geometric: geometric.value,
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
