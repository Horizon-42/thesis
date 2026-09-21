/**
 * trainingSample.ts
 * -----------------
 * The Training task's data contract: the manifest of exported sample sets, and
 * one set's flights — the track, the sentence, and the BOXES that sentence makes.
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.
 *
 * A WORD IS AN INTERVAL, AND A SENTENCE IS A CHAIN OF BOUNDING BOXES (reading
 * rule `box-v3`, 2026-09-21). The criterion is CONTAINMENT: every row of
 * the track has to lie inside the boxes in force at its moment. That replaces
 * the retired reading, where a word was a centre and the question was how far
 * the track sat from it — there is no centre here and no quantisation error, so
 * nothing in this file computes a distance to one.
 *
 * THREE KINDS ARE BOXES, THREE ARE NOT. Heading, altitude and speed each name an
 * interval; the runway word is the FRAME the other three are measured in, the
 * duration word is how long this box is held, and the terminal word is a label.
 * `trainingWordBox` returns null for those three, and a view that drew a band on
 * them would be inventing a number.
 *
 * THE ALTITUDE BOX IS NOT CONSTANT. A heading or speed word is the same interval
 * for as long as it is in force; an altitude word is a TARGET plus the wedge the
 * target is backward-reachable from, so its box narrows as the aircraft runs out
 * of path to the segment's end, closing onto ±5 % of the target. That is why the
 * envelope arrives as two columns per row rather than as one interval per word.
 *
 * THERE IS NO FLOWN TRACK IN THIS FILE, and its absence is a fact about the
 * vocabulary rather than a gap in the export: flying a box sentence needs a
 * height-tracking executor (the design's replay gate), which is not built. What
 * a box sentence can be checked by is containment, and that is what is here.
 *
 * THE SIGNALS THE BOXES JUDGE ARE SMOOTHED, and both versions are carried. The
 * boxes were read from a moving average (6 s on the course, 10 s on speed and
 * height), so that is what `readCourseDeg` / `readSpeedMps` / `readHeightM` are
 * and what every verdict here is computed on. The raw columns are beside them so
 * the smoothing is visible rather than hidden — a chart showing only the smoothed
 * line would be showing a signal nobody flew.
 *
 * VALIDATION IS PER SET, AND A BAD SET IS REJECTED ALONE. This is a deliberate
 * divergence from the comparison manifest's `.every(isComparisonCategory)`, which
 * empties an entire airport's picker over one bad entry and has cost two debugging
 * sessions (AV6, `docs/35-viewer-reference.md`). Training is a development-time
 * view where a half-written export is normal, so `parseTrainingIndex` keeps the
 * good sets and returns the rejected ones WITH the field that failed.
 */

import { fetchJson } from "../utils/fetchJson";
// No unit conversion is imported here: this vocabulary is DEFINED in SI (metres,
// m/s, degrees), so the labels print the stored numbers.

/** MIRROR of the exporter's schema strings. A file that does not carry these is
 *  refused by name rather than read leniently. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v1";
/** v2: the sample file is a different object — an envelope instead of a flown track. */
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v2";

/**
 * MIRROR of the artefact's `spec.reading_rule`, and REFUSED on mismatch.
 *
 * The file carries its own spec — the edge tables, the ladder, the wedge's two
 * angles — so most of what a word means is read from it. What is NOT in the file
 * is the rule's semantics: that a word is an interval rather than a centre, that
 * the second column is a target HEIGHT (the retired vertical word was an angle),
 * that the duration word describes the row it sits on rather than the gap behind
 * it. This reader hardcodes all three, so it is bound to the rule that produced
 * the file and says so by name.
 *
 * The cost of the pin is a one-line edit when the rule bumps; the cost of not
 * pinning it is a chart that looks right and means something else.
 */
export const TRAINING_READING_RULE = "box-v3";

/**
 * MIRROR of the artefact's `spec.kinds` — the six word kinds IN ORDER. The
 * columns of `words` are positional, so this order is load-bearing: reorder it
 * and every word is read as another kind's.
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

/** The three kinds that ARE boxes. The other three name the frame, the hold and
 *  the ending, and none of them bounds a signal. */
export const TRAINING_BOX_KINDS = ["heading", "altitude", "speed"] as const;
export type TrainingBoxKind = (typeof TRAINING_BOX_KINDS)[number];

export function isTrainingBoxKind(kind: TrainingKind): kind is TrainingBoxKind {
  return (TRAINING_BOX_KINDS as readonly string[]).includes(kind);
}

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

/**
 * MIRROR of `instruction_sample_export.INSIDE_EPSILON`: how close to an edge
 * still counts as inside, in each kind's own unit.
 *
 * It is not slack in the vocabulary. The columns are written at display
 * precision while the edge tables are full precision, and the labeller's greedy
 * reader extends a segment until the wedge binds — so rows sitting EXACTLY on an
 * edge are produced by construction (measured: one row in 13,922). Without this
 * they read as violations of a thousandth of a millimetre. The exporter computes
 * its verdict with the same number, and the two verdicts are compared.
 */
export const TRAINING_INSIDE_EPSILON = 1e-3;

/** MIRROR of the artefact's terminal classes. */
export const TERMINAL_CONTINUE = 0;
export const TERMINAL_LANDED = 1;
export const TERMINAL_GO_AROUND = 2;
export const TERMINAL_WORDS = 3;

/**
 * `go-around` is a CLASS THAT EXISTS AND IS NEVER OBSERVED in this data: the
 * vocabulary is read on the 25 km arrival slice, which keeps only the final
 * successful approach (results §13.5 — go-arounds are real in the fleet, 42 of
 * 44,622 tracks, but cannot appear here). A legend may list it; it must say the
 * class is never observed rather than implying the model declines to use it.
 */
export const TERMINAL_NEVER_OBSERVED: readonly number[] = [TERMINAL_GO_AROUND];

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
   * The runway classes' OWN sha. The class set is carried BESIDE the spec
   * (`runway_idents`), so it does not move `vocabularySha256`: two artefacts with
   * the same spec sha can carry different runway lists, and comparing only one of
   * the two and calling it "the same vocabulary" is wrong.
   */
  runwaySha256: string;
  readingRule: string;
  flights: number;
  /**
   * WHICH MODEL this set carries, when its kind says it carries one. It is in
   * the manifest as well as in the sample because the picker has to name the
   * model before anyone downloads several megabytes of it.
   */
  prior?: TrainingSetPrior;
  /**
   * WHICH flights this set holds and how they were chosen. It is required and it
   * is SHOWN, because "40 of 4,486" is not a statement until the rule that picked
   * the 40 is on screen beside it.
   */
  cohort: TrainingCohort;
}

export interface TrainingSetPrior {
  sha256: string;
  seed: number;
  method: string;
}

export interface TrainingCohort {
  split: string;
  perStratum: number;
  seed: number;
  drawnFrom: string;
}

export interface TrainingIndex {
  airport: string;
  sets: TrainingSetEntry[];
  /** Sets that failed validation, each with the field that failed. Kept so the
   *  UI can grey one out by name instead of emptying the list (AV6). */
  rejected: Array<{ id: string; problem: string }>;
}

/**
 * The vocabulary, as the artefact states it. Every number a box is built from is
 * here: the two edge TABLES tile their words, and the altitude table is a LADDER
 * OF TARGETS, one per word, not edges.
 */
export interface TrainingVocabulary {
  sha256: string;
  runwaySha256: string;
  readingRule: string;
  /** The one redundancy the whole vocabulary is built from: ±5 %. */
  redundancyFraction: number;
  /** [heading words + 1] the edges that tile -180…180. Word k is [k, k+1]. */
  headingEdgesDeg: number[];
  /** The absolute floor under the percentage, so the box at the course is 2°
   *  wide rather than nothing. */
  headingFloorDeg: number;
  /** [speed words + 1] the edges that tile the speed range. */
  speedEdgesMps: number[];
  /** [altitude words] the TARGETS, signed: the ladder runs below the threshold
   *  as well as above it, because a track that passes under the threshold's
   *  elevation is normal and was the commonest refusal before the ladder was
   *  signed. */
  altitudeTargetsM: number[];
  /** The offset inside the target's own tolerance, `redundancy × (T + h0)`, so a
   *  target of 0 m still has a box. */
  altitudeH0M: number;
  /** The wedge's two angles. DOWN is the wider side and opens ABOVE the target:
   *  it is the set the target is backward-reachable from, and losing height is
   *  the manoeuvre with the most room. */
  altitudeDownDeg: number;
  altitudeUpDeg: number;
  durationBinS: number;
  durationMaxS: number;
  /** The two smoothing windows the read signals were made with. Shown, because
   *  which signal a verdict was computed on decides the verdict. */
  courseSmoothingS: number;
  smoothingS: number;
  /** The vocabulary's OWN runway classes, never the airport's runway list. */
  runwayIdents: string[];
  /** The class count the ARTEFACT states per kind. `trainingWordCounts` derives
   *  the same numbers from the tables, and `parseVocabulary` refuses a file where
   *  the two disagree — that is what makes the derivation a checked mirror. */
  words: Record<TrainingKind, number>;
}

/**
 * How a track became the three signals the boxes judge. It is REQUIRED and it is
 * SHOWN: the same track against the same boxes is 100 % inside on the smoothed
 * signals and 93 % inside on the raw ones, so a verdict whose signal is not
 * stated is a number with no meaning.
 */
export interface TrainingReadingRule {
  rule: string;
  /** How the altitude box is SHAPED, and how the segments were cut. They sit
   *  here rather than on the vocabulary because `box-v3` stopped carrying them:
   *  they describe the exporter's RECONSTRUCTION, which is what they always
   *  described, and the block that says "produced by" is where that belongs. */
  altitudeForm: string;
  altitudeReading: string;
  courseSignal: string;
  speedSignal: string;
  heightSignal: string;
  pathSignal: string;
  remainingPathTo: string;
  windowRows: string;
  insideEpsilon: number;
  /** Which program wrote this file — and it says, in the file, that the
   *  artefact's own labeller is NOT in this repository. */
  producedBy: string;
  constantsFrom: string[];
}

export interface TrainingSentence {
  /** [E] the moments a box opens — strictly increasing, irregular gaps. */
  eventTimesS: number[];
  /** [E] how long each box is held. It is the duration WORD decoded, and the
   *  parser checks it against that word: the two are one answer (§2.5, the hold
   *  is written on the row it describes) and a file where they disagree has its
   *  duration column pointing at the wrong row. */
  holdS: number[];
  /** [E][6] the words in force at each event, in `TRAINING_KINDS` order. */
  words: number[][];
}

/**
 * The geodetic columns the 3D layer draws from.
 *
 * THE ALTITUDE IS HAE, and the name says so. A record is MSL; Cesium reads
 * `cartographicDegrees` altitude as metres above the WGS84 ellipsoid, so the
 * exporter converts on the way out exactly as the CZML path does. Since
 * h = H + N and N is negative here (-33.5 m at KRDU), a line handed the MSL
 * number renders |N| too HIGH, floating above its own terrain.
 *
 * `haeOffsetM` is that conversion as an OFFSET, per row: add it to any height
 * above the threshold to get HAE. The envelope has four more height columns over
 * the same ground track, and recomputing the geodesy per column would be the same
 * inverse five times.
 */
export const TRAINING_GEODETIC_COLUMNS = ["lon", "lat", "altHaeM", "haeOffsetM"] as const;
export type TrainingGeodeticColumn = (typeof TRAINING_GEODETIC_COLUMNS)[number];

/**
 * MIRROR of `instructions.course_frame` — the observed track in the FINAL
 * APPROACH COURSE's frame, which is the frame the words were read in. Reading
 * the words against anything else would let the charts and the words disagree
 * about where the aircraft was.
 *
 * Every column has one value per row of `tS`. The three `read*` columns are the
 * SMOOTHED signals the boxes are measured against; the raw ones beside them are
 * what the aircraft actually did.
 */
export const TRAINING_OBSERVED_COLUMNS = [
  "toGoM",              // along the course, positive BEFORE the threshold
  "crossM",             // right of the course, positive
  "heightM",            // above the threshold
  // Cumulative HORIZONTAL path from the first row — the axis the altitude wedge
  // is measured on (`r` is remaining PATH, never the projection on the course).
  "pathM",
  "relCourseDeg",       // ground track against the course, wrapped
  "groundSpeedMps",
  "established",        // 0 / 1 per row
  "readCourseDeg",      // the three the boxes judge, smoothed as the labeller did
  "readSpeedMps",
  "readHeightM",
] as const;

export type TrainingObservedColumn = (typeof TRAINING_OBSERVED_COLUMNS)[number];

export type TrainingObserved = { tS: number[] }
  & Record<TrainingObservedColumn, number[]>
  & Record<TrainingGeodeticColumn, number[]>;

/**
 * ONE WORD, AS THE REGION IT ALLOWS — the three intervals it names, and where
 * those intervals let the aircraft be while it stands.
 *
 * A WORD IS A BOX IN STATE SPACE, NOT IN POSITION SPACE. What the vocabulary
 * calls a bounding box is the product of three intervals: heading × speed ×
 * altitude. The set of PLACES that follows is not a box — it is a pie slice
 * fanning out from the aircraft, of radius `holdS × speedHiMps`, spanning the
 * heading interval. `lon` / `lat` are that sector's outline and `toGoM` /
 * `crossM` the same outline in the COURSE FRAME, which is the plan view's own
 * axes; **the first point is the apex**, the aircraft's own position when the
 * word opened. Both spellings come from one computation at the exporter, because
 * the frame's transform lives on that side of the wire and a second inverse here
 * could drift from it.
 *
 * The number of points VARIES — one per degree of the sector's own opening,
 * between 3 and 16 — so nothing here may assume four. An earlier version drew
 * the sector's axis-aligned bounding box instead, which is wrong in the way that
 * matters: it shows flyable-looking ground beside the apex that no heading inside
 * the box can reach.
 *
 * It is DERIVED, not the word itself: a word constrains the STATE at every
 * instant, and where that lets the aircraft go is this. Nothing bounds how fast
 * the heading may swing inside its box, because the word does not — a turn rate
 * belongs to an executor, and there is none here. The view has to say which of
 * the two it is drawing.
 *
 * `altLoM` / `altHiM` are the wedge at the instant the word OPENS, which is the
 * widest it gets while that word stands — so the prism's height is an outer bound
 * over the hold. They are not the target's ±5 %: that is what the wedge closes
 * onto at its segment's end.
 */
export interface TrainingEventBox {
  eventS: number;
  holdS: number;
  headingLoDeg: number;
  headingHiDeg: number;
  speedLoMps: number;
  speedHiMps: number;
  altitudeTargetM: number;
  altLoM: number;
  altHiM: number;
  altHaeLoM: number;
  altHaeHiM: number;
  lon: number[];
  lat: number[];
  toGoM: number[];
  crossM: number[];
}

/** How many rows one kind's boxes hold, and how many they do not. */
export interface TrainingInsideCount {
  rows: number;
  outside: number;
}

/**
 * THE ENVELOPE: the boxes a sentence makes over one track.
 *
 * The four `alt*` columns are per ROW, because the altitude box narrows along
 * its segment; the heading and speed boxes are constant while their word is in
 * force and live on `events`. `inside` is this vocabulary's own criterion, and
 * the reader recomputes it and refuses a file whose verdict differs.
 */
export interface TrainingEnvelope {
  altLoM: number[];
  altHiM: number[];
  altHaeLoM: number[];
  altHaeHiM: number[];
  events: TrainingEventBox[];
  inside: Record<TrainingBoxKind, TrainingInsideCount>;
}

/**
 * WHAT THE MODEL SAID, per flight — and it is not a sentence the model made up.
 *
 * At every event the prior saw the TRUTH's words and the TRUTH's state up to
 * that point and was asked what the next event would be; `words[k]` for k ≥ 1 is
 * that answer, and `words[0]` is the truth's opening event, which is given
 * (`givenEvents`). A free run is a different experiment (the closed loop), and
 * the file says which one this is in `TrainingPrior.method`. Nothing here may be
 * labelled "generated".
 *
 * The event TIMES and the HOLDS stay the truth's. The duration word is predicted
 * like every other kind and is carried, but it does not place the events: each
 * prediction was conditioned on the truth's state at the truth's instant, so
 * re-timing the sentence by the model's own gaps would put its words at moments
 * its conditioning never saw.
 *
 * Its `envelope` is the boxes ITS words make over the SAME track — so `inside`
 * answers the question the model is actually being asked: would the sentence it
 * said have contained the aircraft that flew.
 */
export interface TrainingFlightPrior {
  /** [E][6], the same shape and column order as the truth's sentence. */
  words: number[][];
  /** [E][6]: how much probability the model put on the word it said. The opening
   *  event is given, so its row is all 1. */
  confidence: number[][];
  givenEvents: number;
  /** When the model first said `landed`, or null if it never did — the model
   *  stopping early is a real answer and the bar marks it. The sentence is NOT
   *  truncated there: the model was asked at every event, so every answer is
   *  carried. */
  landedAtS: number | null;
  envelope: TrainingEnvelope;
}

/** MIRROR of `instruction_sample_export.PRIOR_METHOD`. */
export const TRAINING_PRIOR_METHOD = "teacher-forced-next-word";

/**
 * The model that said them, and what it scored. `readout` is the prior run's own
 * `readings.json` table, carried rather than recomputed: the NLL and top-1 the
 * view prints must be the ones the run reported.
 */
export interface TrainingPrior {
  sha256: string;
  method: string;
  seed: number;
  bestEpoch: number;
  /** How many of the DRAWN flights the model was fitted on. It is 0 for a val
   *  draw and it is shown, because a set drawn from `train` shows the model
   *  reciting what it was trained on and that is a different claim. */
  trainedOnTheseFlights: number;
  readout: Record<string, Record<string, number>>;
}

export interface TrainingFlight {
  flightKey: string;
  callsign: string;
  runway: string;
  stratum: string;
  /**
   * The TRACK's length. Under this rule the sentence covers all of it — the
   * events tile the track and the last box carries its own hold — so the parser
   * checks that the last event plus its hold lands on this number.
   */
  durationS: number;
  /** The track's own sampling step, and the two smoothing windows in ROWS that
   *  the read signals were made with. Shown, because the window is what decides
   *  how far the smoothed line can sit from the raw one. */
  dtS: number;
  courseWindowRows: number;
  signalWindowRows: number;
  sentence: TrainingSentence;
  observed: TrainingObserved;
  envelope: TrainingEnvelope;
  /** Present exactly when the set's kind is `prior-generated` — see
   *  `parseTrainingSample`, which keys it on the kind rather than on whether the
   *  field happens to be there. */
  prior?: TrainingFlightPrior;
}

/** What the panel publishes for the full-width sentence bar to draw: one flight
 *  and the vocabulary its words are read under. */
export interface TrainingSelection {
  vocabulary: TrainingVocabulary;
  flight: TrainingFlight;
  /** The model that said `flight.prior`, when there is one. */
  prior?: TrainingPrior;
  /** How the signals the boxes judge were made. It travels with the selection
   *  because the views that draw a verdict are the ones that have to say what
   *  the verdict was computed on. */
  reading: TrainingReadingRule;
}

export interface TrainingSample {
  setId: string;
  airport: string;
  vocabulary: TrainingVocabulary;
  reading: TrainingReadingRule;
  /** The model whose words every flight carries, when this is a prior set. */
  prior?: TrainingPrior;
  flights: TrainingFlight[];
}

/** The part of a vocabulary that decides what a word MEANS. Everything that
 *  reads words takes this, so `parseVocabulary` can derive the class counts while
 *  it is still building the vocabulary. */
export type TrainingWordSpec = Pick<
  TrainingVocabulary,
  | "redundancyFraction" | "headingEdgesDeg" | "headingFloorDeg" | "speedEdgesMps"
  | "altitudeTargetsM" | "altitudeH0M" | "altitudeDownDeg" | "altitudeUpDeg"
  | "durationBinS" | "durationMaxS" | "runwayIdents"
>;

export type Parsed<T> = { ok: true; value: T } | { ok: false; problem: string };

// ── word counts ──────────────────────────────────────────────────────────────

/**
 * How many classes each kind has, under this file's own spec.
 *
 * The two edge tables TILE their words, so they hold one more value than there
 * are words; the altitude ladder holds one target PER word. Getting that
 * backwards is the mistake this derivation exists to catch — it shifts every
 * altitude word by half a box and still plots.
 *
 * The range check earns its keep on the POSITIONAL columns: swap two and a
 * duration word (0–300) lands in the runway column (0–21) and fails loudly,
 * instead of drawing the wrong runway for the whole flight.
 */
export function trainingWordCounts(
  vocabulary: TrainingWordSpec,
): Record<TrainingKind, number> {
  return {
    heading: vocabulary.headingEdgesDeg.length - 1,
    altitude: vocabulary.altitudeTargetsM.length,
    speed: vocabulary.speedEdgesMps.length - 1,
    runway: vocabulary.runwayIdents.length,
    duration: Math.round(vocabulary.durationMaxS / vocabulary.durationBinS) + 1,
    terminal: TERMINAL_WORDS,
  };
}

// ── what a word means ────────────────────────────────────────────────────────

/** The heading word's interval, in degrees relative to the final approach
 *  course. The edges tile -180…180, so word k is [edge k, edge k+1]. */
export function headingBoxDeg(vocabulary: TrainingWordSpec, word: number): [number, number] {
  return [vocabulary.headingEdgesDeg[word], vocabulary.headingEdgesDeg[word + 1]];
}

/**
 * Which heading word holds a relative course — the edges tile, so it is the last
 * edge at or below the value. Used to name the box ON the course, which is the
 * one number that says how fine this vocabulary is where it matters (2° at the
 * course, 18° at the reciprocal).
 */
export function headingWordAt(vocabulary: TrainingWordSpec, degrees: number): number {
  const edges = vocabulary.headingEdgesDeg;
  for (let word = edges.length - 2; word > 0; word -= 1) {
    if (degrees >= edges[word]) return word;
  }
  return 0;
}

/** The speed word's interval, in m/s of ground speed. */
export function speedBoxMps(vocabulary: TrainingWordSpec, word: number): [number, number] {
  return [vocabulary.speedEdgesMps[word], vocabulary.speedEdgesMps[word + 1]];
}

/** The altitude word's TARGET — a height above the threshold, signed. */
export function altitudeTargetM(vocabulary: TrainingWordSpec, word: number): number {
  return vocabulary.altitudeTargetsM[word];
}

/**
 * The target's own tolerance, `redundancy × (T + h0)` — the half-width the wedge
 * closes onto at the END of its segment, and the spacing of the ladder itself.
 * The `+ h0` is what gives a target of 0 m a box at all.
 */
export function altitudeFloorM(vocabulary: TrainingWordSpec, word: number): number {
  return vocabulary.redundancyFraction * (altitudeTargetM(vocabulary, word) + vocabulary.altitudeH0M);
}

/**
 * The wedge at a row that has `remainingPathM` of track left before its
 * segment's end: `T - r·tan(up) - f ≤ h ≤ T + r·tan(down) + f`.
 *
 * MIRROR of `instruction_sample_export.altitude_envelope`, and of the artefact's
 * own `reading.altitudeForm`. The exporter is what writes the envelope columns; this is
 * here so a view can draw the wedge for a word the file carries no column for —
 * the MODEL's words at a row, say — without a second definition of the shape.
 */
export function altitudeWedgeM(
  vocabulary: TrainingWordSpec,
  word: number,
  remainingPathM: number,
): [number, number] {
  const target = altitudeTargetM(vocabulary, word);
  const floor = altitudeFloorM(vocabulary, word);
  const remaining = Math.max(remainingPathM, 0);
  return [
    target - remaining * Math.tan((vocabulary.altitudeUpDeg * Math.PI) / 180) - floor,
    target + remaining * Math.tan((vocabulary.altitudeDownDeg * Math.PI) / 180) + floor,
  ];
}

/**
 * The interval a word names, or `null` for the three kinds that name no interval.
 *
 * The null is the point of this function: the runway word is the FRAME, the
 * duration word is the hold and the terminal word is a label, so a view that drew
 * a band on any of them would be inventing a number.
 *
 * For an ALTITUDE word this is the box at the segment's END — the target's own
 * ±5 %, which is the tightest the wedge ever gets. The wedge at a particular row
 * is `altitudeWedgeM`, or the envelope's own columns.
 */
export function trainingWordBox(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): [number, number] | null {
  if (kind === "heading") return headingBoxDeg(vocabulary, word);
  if (kind === "speed") return speedBoxMps(vocabulary, word);
  if (kind === "altitude") {
    const target = altitudeTargetM(vocabulary, word);
    const floor = altitudeFloorM(vocabulary, word);
    return [target - floor, target + floor];
  }
  return null;
}

/** MIRROR of the duration word's decoding: the hold this box is kept for. */
export function durationHoldS(vocabulary: TrainingWordSpec, word: number): number {
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

/** The terminal words, by index. */
export const TERMINAL_LABELS = ["continue", "landed", "go-around"] as const;

/**
 * A word as a person reads it.
 *
 * THE UNITS ARE SI, because this vocabulary is defined in SI: the edges were
 * fitted in metres and m/s, so printing knots would label a box 130.0–143.5 kt —
 * arithmetic the reader would have to undo to recognise the vocabulary.
 *
 * The ALTITUDE word prints its target, not its box, because the target is what
 * the word names; the box depends on where in the segment you are, and the two
 * are shown together by `trainingWordBandLabel`.
 */
export function trainingWordLabel(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): string {
  switch (kind) {
    case "heading": {
      const [low, high] = headingBoxDeg(vocabulary, word);
      const sign = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(value === Math.round(value) ? 0 : 1)}`;
      return `${sign(low)}…${sign(high)}°`;
    }
    case "altitude":
      return `${Math.round(altitudeTargetM(vocabulary, word))} m`;
    case "speed": {
      const [low, high] = speedBoxMps(vocabulary, word);
      return `${low.toFixed(0)}–${high.toFixed(0)} m/s`;
    }
    case "runway":
      return vocabulary.runwayIdents[word];
    case "duration":
      return `${durationHoldS(vocabulary, word)} s`;
    case "terminal":
      return TERMINAL_LABELS[word];
  }
}

/**
 * The same word WITH the interval it names — what the sentence bar writes on a
 * box. For heading and speed the label already IS the interval, so this adds
 * nothing; for altitude it adds the target's own ±, which is the box the wedge
 * closes onto at the end of its segment.
 */
export function trainingWordBandLabel(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): string {
  const label = trainingWordLabel(vocabulary, kind, word);
  if (kind !== "altitude") return label;
  return `${label}±${altitudeFloorM(vocabulary, word).toFixed(1)}`;
}

// ── the verdict ──────────────────────────────────────────────────────────────

/** Which event is in force at each row: the last one that has opened. The events
 *  tile the track, so every row has exactly one. */
export function eventInForce(eventTimesS: number[], tS: number[]): number[] {
  const rows: number[] = [];
  let event = 0;
  for (const time of tS) {
    while (event + 1 < eventTimesS.length && eventTimesS[event + 1] <= time) event += 1;
    rows.push(event);
  }
  return rows;
}

/**
 * Row by row: is the signal inside the box in force. ONE VERDICT PER ROW, which
 * is what makes the counts add up — an earlier reading of this view walked spans
 * and counted the shared boundary row of two consecutive spans twice, so 132 rows
 * produced 134 judgements.
 *
 * It is recomputed here rather than read off the file, and `parseTrainingSample`
 * refuses a file whose own counts disagree. The exporter derives the boxes from
 * the artefact's spec and this side checks them against the columns beside them:
 * two implementations of the same comparison, which is the only check available
 * for a rule whose labeller is not in this repository.
 */
export function trainingContainment(
  flight: Pick<TrainingFlight, "observed" | "sentence">,
  envelope: TrainingEnvelope,
  vocabulary: TrainingWordSpec,
  words: number[][],
): Record<TrainingBoxKind, { rows: number; outside: number; inside: boolean[] }> {
  const { observed } = flight;
  const forced = eventInForce(flight.sentence.eventTimesS, observed.tS);
  const judge = (values: number[], low: (row: number) => number, high: (row: number) => number) => {
    const inside = values.map(
      (value, row) =>
        value >= low(row) - TRAINING_INSIDE_EPSILON && value <= high(row) + TRAINING_INSIDE_EPSILON,
    );
    return { rows: inside.length, outside: inside.filter((ok) => !ok).length, inside };
  };
  const headingBox = (row: number) =>
    headingBoxDeg(vocabulary, words[forced[row]][TRAINING_KIND_COLUMN.heading]);
  const speedBox = (row: number) =>
    speedBoxMps(vocabulary, words[forced[row]][TRAINING_KIND_COLUMN.speed]);
  return {
    heading: judge(observed.readCourseDeg, (row) => headingBox(row)[0], (row) => headingBox(row)[1]),
    altitude: judge(observed.readHeightM, (row) => envelope.altLoM[row], (row) => envelope.altHiM[row]),
    speed: judge(observed.readSpeedMps, (row) => speedBox(row)[0], (row) => speedBox(row)[1]),
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
  // Keyed on the KIND, like the sample's own prior block: a prior set that
  // cannot name its model is half-written, and a read-back set that names one
  // has the wrong kind.
  let prior: TrainingSetPrior | undefined;
  if (kind === "prior-generated") {
    const block = raw.prior;
    if (!isRecord(block)) {
      return { ok: false, problem: `${where("prior")} is missing: a prior set names its model in the manifest` };
    }
    const sha256 = str(block, "sha256");
    const method = str(block, "method");
    const seed = finite(block, "seed");
    if (sha256 === null || method === null || seed === null) {
      return { ok: false, problem: `${where("prior")} must carry the model's sha256, its seed and how it was asked` };
    }
    prior = { sha256, seed, method };
  } else if (raw.prior !== undefined) {
    return { ok: false, problem: `${where("prior")} names a model, but this set's kind is ${kind}` };
  }

  const cohort = raw.cohort;
  if (!isRecord(cohort)) {
    return { ok: false, problem: `${where("cohort")} is missing: a set that cannot say how its flights were drawn is a set nobody can reproduce` };
  }
  const split = str(cohort, "split");
  const drawnFrom = str(cohort, "drawnFrom");
  const perStratum = finite(cohort, "perStratum");
  const seed = finite(cohort, "seed");
  if (split === null || drawnFrom === null) {
    return { ok: false, problem: `${where("cohort")} must name its split and where the draw came from` };
  }
  if (perStratum === null || perStratum <= 0 || seed === null) {
    return { ok: false, problem: `${where("cohort")} must carry the per-stratum count and the seed that reproduces it` };
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
      ...(prior ? { prior } : {}),
      cohort: { split, perStratum, seed, drawnFrom },
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

/** A table that TILES its words: sorted, distinct, one more value than words. */
function parseTable(raw: unknown, field: string, minimum: number): Parsed<number[]> {
  const values = numberArray(raw);
  if (values === null || values.length < minimum) {
    return { ok: false, problem: `vocabulary.${field} is missing or holds fewer than ${minimum} values` };
  }
  for (let i = 1; i < values.length; i += 1) {
    if (!(values[i] > values[i - 1])) {
      return {
        ok: false,
        problem:
          `vocabulary.${field} is not stored sorted and distinct (index ${i}: ` +
          `${values[i - 1]} then ${values[i]}) — the boxes tile, so an unsorted table overlaps two words`,
      };
    }
  }
  return { ok: true, value: values };
}

function parseVocabulary(raw: unknown): Parsed<TrainingVocabulary> {
  if (!isRecord(raw)) return { ok: false, problem: "vocabulary is not an object" };
  const numbers: Array<keyof TrainingVocabulary> = [
    "redundancyFraction", "headingFloorDeg", "altitudeH0M", "altitudeDownDeg", "altitudeUpDeg",
    "durationBinS", "durationMaxS", "courseSmoothingS", "smoothingS",
  ];
  const values: Record<string, number> = {};
  for (const field of numbers) {
    const value = finite(raw, field);
    if (value === null) return { ok: false, problem: `vocabulary.${field} is missing or not a number` };
    values[field] = value;
  }
  // Every one of these is positive. A redundancy of zero is a box nothing can sit
  // in; a wedge angle of zero is a box that never opens, which turns the altitude
  // word back into the flat ladder this rule replaced.
  for (const field of numbers) {
    if (values[field] <= 0) return { ok: false, problem: `vocabulary.${field} must be positive` };
  }
  // The wedge is ASYMMETRIC and which way round decides which side is dangerous.
  // Swapping them draws a corridor of exactly the same width with the slack on
  // the wrong side of the target, which nothing else here would catch.
  if (!(values.altitudeDownDeg > values.altitudeUpDeg)) {
    return {
      ok: false,
      problem:
        `vocabulary.altitudeDownDeg is ${values.altitudeDownDeg}° and altitudeUpDeg is ` +
        `${values.altitudeUpDeg}° — the descent side is the wider one, so down > up`,
    };
  }

  const heading = parseTable(raw.headingEdgesDeg, "headingEdgesDeg", 3);
  if (!heading.ok) return heading;
  const speed = parseTable(raw.speedEdgesMps, "speedEdgesMps", 3);
  if (!speed.ok) return speed;
  const altitude = parseTable(raw.altitudeTargetsM, "altitudeTargetsM", 2);
  if (!altitude.ok) return altitude;
  // The boxes are two-ended and they tile: no ray, because a ray contains any
  // value and would turn a corrupt row into a legal word (§2.1).
  if (heading.value[0] > -180 || heading.value[heading.value.length - 1] < 180) {
    return {
      ok: false,
      problem:
        `vocabulary.headingEdgesDeg runs ${heading.value[0]}…${heading.value[heading.value.length - 1]}°, ` +
        `which does not cover -180…180 — a relative course outside the table has no word`,
    };
  }

  for (const field of ["sha256", "runwaySha256", "readingRule"]) {
    if (str(raw, field) === null) {
      return { ok: false, problem: `vocabulary.${field} is missing or not a non-empty string` };
    }
  }
  if (str(raw, "readingRule") !== TRAINING_READING_RULE) {
    return {
      ok: false,
      problem:
        `vocabulary.readingRule is ${JSON.stringify(raw.readingRule)}, and this reader is written for ` +
        `${TRAINING_READING_RULE} — the rule decides what a word IS (an interval, not a centre; the ` +
        `second column a target height, not an angle), which no field of the file can say. ` +
        `Re-export under the current rule`,
    };
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
    redundancyFraction: values.redundancyFraction,
    headingEdgesDeg: heading.value,
    headingFloorDeg: values.headingFloorDeg,
    speedEdgesMps: speed.value,
    altitudeTargetsM: altitude.value,
    altitudeH0M: values.altitudeH0M,
    altitudeDownDeg: values.altitudeDownDeg,
    altitudeUpDeg: values.altitudeUpDeg,
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
    // The counts we derive from the tables and the counts the file states are two
    // spellings of the same thing. Them agreeing is the whole value of the
    // derivation; disagreeing means one of the two mirrors has drifted, and
    // guessing which would put every word in the wrong legend.
    if (count !== derived[kind]) {
      return {
        ok: false,
        problem:
          `vocabulary.words.${kind} is ${count}, but this file's own tables give ${derived[kind]} — ` +
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
      courseSmoothingS: values.courseSmoothingS,
      smoothingS: values.smoothingS,
      ...spec,
      words,
    },
  };
}

function parseReadingRule(raw: unknown): Parsed<TrainingReadingRule> {
  if (!isRecord(raw)) {
    return { ok: false, problem: "reading is missing: a verdict whose signal is not stated is a number with no meaning" };
  }
  const strings: Record<string, string> = {};
  for (const field of ["rule", "altitudeForm", "altitudeReading", "courseSignal", "speedSignal",
                       "heightSignal", "pathSignal", "remainingPathTo", "windowRows", "producedBy"]) {
    const value = str(raw, field);
    if (value === null) return { ok: false, problem: `reading.${field} is missing or not a non-empty string` };
    strings[field] = value;
  }
  if (strings.rule !== TRAINING_READING_RULE) {
    return { ok: false, problem: `reading.rule is ${JSON.stringify(strings.rule)}, expected ${TRAINING_READING_RULE}` };
  }
  const epsilon = finite(raw, "insideEpsilon");
  if (epsilon === null || epsilon < 0) {
    return { ok: false, problem: "reading.insideEpsilon is missing or not a tolerance" };
  }
  // The two sides have to judge a row on the edge the same way, and the counts
  // are compared: a file written with a different epsilon would disagree with
  // this reader on exactly the rows that sit on a box's edge, which is where the
  // labeller's greedy reader leaves them.
  if (epsilon !== TRAINING_INSIDE_EPSILON) {
    return {
      ok: false,
      problem:
        `reading.insideEpsilon is ${epsilon}, and this reader judges with ${TRAINING_INSIDE_EPSILON} — ` +
        `the two verdicts are compared, so they have to be computed the same way`,
    };
  }
  const sources = raw.constantsFrom;
  if (!Array.isArray(sources) || !sources.every((item) => typeof item === "string")) {
    return { ok: false, problem: "reading.constantsFrom is missing or not a list of sources" };
  }
  return {
    ok: true,
    value: {
      rule: strings.rule,
      altitudeForm: strings.altitudeForm,
      altitudeReading: strings.altitudeReading,
      courseSignal: strings.courseSignal,
      speedSignal: strings.speedSignal,
      heightSignal: strings.heightSignal,
      pathSignal: strings.pathSignal,
      remainingPathTo: strings.remainingPathTo,
      windowRows: strings.windowRows,
      insideEpsilon: epsilon,
      producedBy: strings.producedBy,
      constantsFrom: sources as string[],
    },
  };
}

function parseWordRows(
  raw: unknown,
  events: number,
  counts: Record<TrainingKind, number>,
  where: string,
): Parsed<number[][]> {
  if (!Array.isArray(raw)) return { ok: false, problem: `${where} is missing or not an array` };
  if (raw.length !== events) {
    return { ok: false, problem: `${where} has ${raw.length} rows, but the sentence has ${events} events` };
  }
  const words: number[][] = [];
  for (let row = 0; row < raw.length; row += 1) {
    const parsedRow = numberArray(raw[row]);
    if (parsedRow === null) return { ok: false, problem: `${where}[${row}] is not a list of numbers` };
    if (parsedRow.length !== TRAINING_WORD_COLUMNS) {
      return {
        ok: false,
        problem:
          `${where}[${row}] has ${parsedRow.length} columns, expected ${TRAINING_WORD_COLUMNS} ` +
          `(${TRAINING_KINDS.join(", ")})`,
      };
    }
    for (const kind of TRAINING_KINDS) {
      const value = parsedRow[TRAINING_KIND_COLUMN[kind]];
      if (!Number.isInteger(value) || value < 0 || value >= counts[kind]) {
        return {
          ok: false,
          problem:
            `${where}[${row}].${kind} is ${value}, outside this vocabulary's ` +
            `0…${counts[kind] - 1} — the columns are positional, so check their ORDER first`,
        };
      }
    }
    words.push(parsedRow);
  }
  return { ok: true, value: words };
}

function parseSentence(
  raw: unknown,
  counts: Record<TrainingKind, number>,
  vocabulary: TrainingWordSpec,
  durationS: number,
  where: string,
): Parsed<TrainingSentence> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.sentence is not an object` };

  const eventTimesS = numberArray(raw.eventTimesS);
  if (eventTimesS === null || eventTimesS.length === 0) {
    return { ok: false, problem: `${where}.sentence.eventTimesS is missing, empty or not a list of numbers` };
  }
  if (eventTimesS[0] !== 0) {
    return {
      ok: false,
      problem: `${where}.sentence.eventTimesS opens at ${eventTimesS[0]} s, not 0 — the first box opens with the track`,
    };
  }
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
  const holdS = numberArray(raw.holdS);
  if (holdS === null || holdS.length !== eventTimesS.length) {
    return {
      ok: false,
      problem: `${where}.sentence.holdS is missing or does not have the ${eventTimesS.length} events' rows`,
    };
  }
  const words = parseWordRows(raw.words, eventTimesS.length, counts, `${where}.sentence.words`);
  if (!words.ok) return words;

  // THE BOXES TILE THE TRACK IN TIME. Each box is held for exactly as long as the
  // duration word says, and the next one opens where it ends — that is what makes
  // a sentence a chain rather than a list of moments, and it is what lets every
  // row have exactly one box in force. A gap would leave rows unjudged; an
  // overlap would judge a row twice.
  for (let i = 0; i < eventTimesS.length; i += 1) {
    const stated = durationHoldS(vocabulary, words.value[i][TRAINING_KIND_COLUMN.duration]);
    if (Math.abs(stated - holdS[i]) > 1e-6) {
      return {
        ok: false,
        problem:
          `${where}.sentence: event ${i} is held for ${holdS[i]} s but its duration word says ` +
          `${stated} s — the hold and the word are ONE answer (the word describes the row it sits on)`,
      };
    }
    const closes = eventTimesS[i] + holdS[i];
    const next = i + 1 < eventTimesS.length ? eventTimesS[i + 1] : durationS;
    if (Math.abs(closes - next) > 0.05) {
      return {
        ok: false,
        problem:
          `${where}.sentence: event ${i} closes at ${closes} s but the ${i + 1 < eventTimesS.length ? "next event opens" : "track ends"} ` +
          `at ${next} s — the boxes tile the track, so a gap leaves rows with no box in force`,
      };
    }
  }

  return { ok: true, value: { eventTimesS, holdS, words: words.value } };
}

function parseObserved(raw: unknown, durationS: number, where: string): Parsed<TrainingObserved> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.observed is not an object` };
  const tS = numberArray(raw.tS);
  if (tS === null || tS.length < 2) {
    return { ok: false, problem: `${where}.observed.tS is missing or shorter than two rows` };
  }
  // The track's clock is the flight's clock: it starts at 0 and it ends where
  // `durationS` says the flight ends, because both come from the same rows. A
  // disagreement means the track and the sentence are not the same flight.
  if (tS[0] !== 0) {
    return { ok: false, problem: `${where}.observed.tS starts at ${tS[0]} s, not 0` };
  }
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
  for (const column of [...TRAINING_OBSERVED_COLUMNS, ...TRAINING_GEODETIC_COLUMNS]) {
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
  // The path axis only counts up — it is a length, and the wedge reads a
  // DIFFERENCE of it as remaining distance. A column that ever fell would give a
  // negative remaining path, which clamps to zero and silently closes the box.
  for (let i = 1; i < columns.pathM.length; i += 1) {
    if (columns.pathM[i] < columns.pathM[i - 1]) {
      return { ok: false, problem: `${where}.observed.pathM falls at row ${i}: a path length only grows` };
    }
  }

  return { ok: true, value: columns as TrainingObserved };
}

function parseEventBox(raw: unknown, index: number, where: string): Parsed<TrainingEventBox> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.events[${index}] is not an object` };
  const numbers: Record<string, number> = {};
  for (const field of ["eventS", "holdS", "headingLoDeg", "headingHiDeg", "speedLoMps",
                       "speedHiMps", "altitudeTargetM", "altLoM", "altHiM", "altHaeLoM", "altHaeHiM"]) {
    const value = finite(raw, field);
    if (value === null) return { ok: false, problem: `${where}.events[${index}].${field} is missing or not a number` };
    numbers[field] = value;
  }
  // Every box is an interval, and an inverted one is not a narrower box — it is a
  // box nothing can be inside, which would read as "the model said something
  // impossible" rather than as a corrupt file.
  for (const [low, high] of [["headingLoDeg", "headingHiDeg"], ["speedLoMps", "speedHiMps"],
                             ["altLoM", "altHiM"], ["altHaeLoM", "altHaeHiM"]]) {
    if (numbers[low] > numbers[high]) {
      return {
        ok: false,
        problem: `${where}.events[${index}] is inverted: ${low} ${numbers[low]} is above ${high} ${numbers[high]}`,
      };
    }
  }
  const corners: Record<string, number[]> = {};
  for (const field of ["lon", "lat", "toGoM", "crossM"]) {
    const values = numberArray(raw[field]);
    if (values === null || values.length < 3) {
      return {
        ok: false,
        problem:
          `${where}.events[${index}].${field} is missing or shorter than three points — the ground ` +
          `footprint is a sector's outline, in geodetic and in course-frame coordinates`,
      };
    }
    corners[field] = values;
  }
  // The four arrays are ONE outline in two coordinate systems, so a length that
  // differs between them is two different shapes drawn as though they were one:
  // the plan view would draw a sector the 3D scene does not have.
  if (new Set(Object.values(corners).map((values) => values.length)).size !== 1) {
    return {
      ok: false,
      problem:
        `${where}.events[${index}]: lon/lat/toGoM/crossM have different lengths ` +
        `(${Object.entries(corners).map(([field, values]) => `${field} ${values.length}`).join(", ")}) — ` +
        `they are one outline in two coordinate systems`,
    };
  }
  return {
    ok: true,
    value: {
      eventS: numbers.eventS,
      holdS: numbers.holdS,
      headingLoDeg: numbers.headingLoDeg,
      headingHiDeg: numbers.headingHiDeg,
      speedLoMps: numbers.speedLoMps,
      speedHiMps: numbers.speedHiMps,
      altitudeTargetM: numbers.altitudeTargetM,
      altLoM: numbers.altLoM,
      altHiM: numbers.altHiM,
      altHaeLoM: numbers.altHaeLoM,
      altHaeHiM: numbers.altHaeHiM,
      lon: corners.lon,
      lat: corners.lat,
      toGoM: corners.toGoM,
      crossM: corners.crossM,
    },
  };
}

/**
 * The envelope, checked against the track it is drawn over AND against the
 * verdict the exporter wrote.
 *
 * The recomputation is the point. The artefact's labeller is not in this
 * repository, so the exporter's boxes are a reconstruction from the spec; this
 * reader measures them against the columns beside them and refuses a file where
 * the two answers differ. A file that passes has been judged twice.
 */
function parseEnvelope(
  raw: unknown,
  flight: { observed: TrainingObserved; sentence: TrainingSentence },
  words: number[][],
  vocabulary: TrainingWordSpec,
  where: string,
  what: string,
): Parsed<TrainingEnvelope> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.${what} is missing or not an object` };
  const rows = flight.observed.tS.length;
  const columns: Record<string, number[]> = {};
  for (const column of ["altLoM", "altHiM", "altHaeLoM", "altHaeHiM"]) {
    const values = numberArray(raw[column]);
    if (values === null || values.length !== rows) {
      return { ok: false, problem: `${where}.${what}.${column} is missing or does not have the track's ${rows} rows` };
    }
    columns[column] = values;
  }
  for (let row = 0; row < rows; row += 1) {
    if (columns.altLoM[row] > columns.altHiM[row]) {
      return {
        ok: false,
        problem:
          `${where}.${what} is inverted at row ${row}: altLoM ${columns.altLoM[row]} is above ` +
          `altHiM ${columns.altHiM[row]}`,
      };
    }
  }
  if (!Array.isArray(raw.events) || raw.events.length !== flight.sentence.eventTimesS.length) {
    return {
      ok: false,
      problem:
        `${where}.${what}.events has ${Array.isArray(raw.events) ? raw.events.length : "no"} boxes, but the ` +
        `sentence has ${flight.sentence.eventTimesS.length} events — one box per word`,
    };
  }
  const events: TrainingEventBox[] = [];
  for (let index = 0; index < raw.events.length; index += 1) {
    const parsed = parseEventBox(raw.events[index], index, `${where}.${what}`);
    if (!parsed.ok) return parsed;
    if (Math.abs(parsed.value.eventS - flight.sentence.eventTimesS[index]) > 0.05) {
      return {
        ok: false,
        problem:
          `${where}.${what}.events[${index}] opens at ${parsed.value.eventS} s but event ${index} is at ` +
          `${flight.sentence.eventTimesS[index]} s — the boxes are the sentence's own, in its order`,
      };
    }
    events.push(parsed.value);
  }

  const stated = raw.inside;
  if (!isRecord(stated)) {
    return { ok: false, problem: `${where}.${what}.inside is missing: containment is this vocabulary's criterion` };
  }
  const envelope: TrainingEnvelope = {
    altLoM: columns.altLoM,
    altHiM: columns.altHiM,
    altHaeLoM: columns.altHaeLoM,
    altHaeHiM: columns.altHaeHiM,
    events,
    inside: {} as Record<TrainingBoxKind, TrainingInsideCount>,
  };
  const measured = trainingContainment(flight, envelope, vocabulary, words);
  for (const kind of TRAINING_BOX_KINDS) {
    const block = stated[kind];
    if (!isRecord(block)) {
      return { ok: false, problem: `${where}.${what}.inside.${kind} is missing` };
    }
    const count = finite(block, "rows");
    const outside = finite(block, "outside");
    if (count === null || outside === null || outside < 0 || outside > count) {
      return { ok: false, problem: `${where}.${what}.inside.${kind} must carry a row count and how many are outside` };
    }
    if (count !== rows) {
      return {
        ok: false,
        problem: `${where}.${what}.inside.${kind} judges ${count} rows, but the track has ${rows}`,
      };
    }
    if (outside !== measured[kind].outside) {
      return {
        ok: false,
        problem:
          `${where}.${what}.inside.${kind} says ${outside} of ${count} rows are outside their box, and this ` +
          `reader measures ${measured[kind].outside} on the columns beside it — the exporter's boxes and ` +
          `this reader's reading of them disagree`,
      };
    }
    envelope.inside[kind] = { rows: count, outside };
  }
  return { ok: true, value: envelope };
}

/**
 * What the model said on one flight, checked against the sentence it answers.
 *
 * The row count is the check that matters: the prior was asked at every event of
 * THIS flight, so its sentence has exactly as many rows as the truth's. A shorter
 * one would line up silently — every row would still plot, just against the wrong
 * event — and the whole view is a comparison of the two row by row.
 */
function parseFlightPrior(
  raw: unknown,
  flight: { observed: TrainingObserved; sentence: TrainingSentence },
  counts: Record<TrainingKind, number>,
  vocabulary: TrainingWordSpec,
  durationS: number,
  where: string,
): Parsed<TrainingFlightPrior> {
  if (!isRecord(raw)) {
    return { ok: false, problem: `${where}.prior is missing: this set's kind says every flight carries what the model said` };
  }
  const events = flight.sentence.eventTimesS.length;
  const words = parseWordRows(raw.words, events, counts, `${where}.prior.words`);
  if (!words.ok) return words;
  if (!Array.isArray(raw.confidence) || raw.confidence.length !== events) {
    return {
      ok: false,
      problem: `${where}.prior.confidence has ${Array.isArray(raw.confidence) ? raw.confidence.length : "no"} rows, but the sentence has ${events} events`,
    };
  }
  const confidence: number[][] = [];
  for (let row = 0; row < events; row += 1) {
    const sure = numberArray((raw.confidence as unknown[])[row]);
    if (sure === null || sure.length !== TRAINING_WORD_COLUMNS) {
      return { ok: false, problem: `${where}.prior.confidence[${row}] is not ${TRAINING_WORD_COLUMNS} numbers` };
    }
    for (const probability of sure) {
      if (!(probability >= 0 && probability <= 1)) {
        return { ok: false, problem: `${where}.prior.confidence[${row}] holds ${probability}, not a probability` };
      }
    }
    confidence.push(sure);
  }
  const givenEvents = finite(raw, "givenEvents");
  if (givenEvents === null || !Number.isInteger(givenEvents) || givenEvents < 1 || givenEvents >= events) {
    return {
      ok: false,
      problem: `${where}.prior.givenEvents is ${JSON.stringify(raw.givenEvents)}: at least the opening event is given, and not the whole sentence`,
    };
  }
  const landed = raw.landedAtS;
  if (landed !== null && (typeof landed !== "number" || !Number.isFinite(landed) || landed < 0 || landed > durationS)) {
    return { ok: false, problem: `${where}.prior.landedAtS is ${JSON.stringify(landed)}, expected a time inside the track or null` };
  }
  const envelope = parseEnvelope(raw.envelope, flight, words.value, vocabulary, where, "prior.envelope");
  if (!envelope.ok) return envelope;
  return {
    ok: true,
    value: {
      words: words.value, confidence, givenEvents,
      landedAtS: landed as number | null, envelope: envelope.value,
    },
  };
}

/** The model itself: which one, how it was asked, and what the run scored. */
function parsePrior(raw: unknown): Parsed<TrainingPrior> {
  if (!isRecord(raw)) {
    return { ok: false, problem: "prior is missing: this set's kind says it carries a model's words" };
  }
  const sha256 = str(raw, "sha256");
  const method = str(raw, "method");
  if (sha256 === null || method === null) {
    return { ok: false, problem: "prior must name the model (sha256) and how it was asked (method)" };
  }
  // The method is REFUSED unless it is the one this view's wording describes.
  // Every label here says the model answered from the truth's history; a free run
  // is a different experiment, and reading one under this wording would publish
  // a claim nobody made.
  if (method !== TRAINING_PRIOR_METHOD) {
    return {
      ok: false,
      problem: `prior.method is ${JSON.stringify(method)}, and this view is written for ${TRAINING_PRIOR_METHOD} — a free run is a different experiment and would need its own wording`,
    };
  }
  const seed = finite(raw, "seed");
  const bestEpoch = finite(raw, "bestEpoch");
  const trainedOn = finite(raw, "trainedOnTheseFlights");
  if (seed === null || bestEpoch === null || trainedOn === null || trainedOn < 0) {
    return { ok: false, problem: "prior must carry its seed, its best epoch and how many of these flights it was trained on" };
  }
  if (!isRecord(raw.readout)) {
    return { ok: false, problem: "prior.readout is missing: the view prints the run's own numbers, never its own" };
  }
  const readout: Record<string, Record<string, number>> = {};
  for (const [split, table] of Object.entries(raw.readout)) {
    if (!isRecord(table)) return { ok: false, problem: `prior.readout.${split} is not a table` };
    const row: Record<string, number> = {};
    for (const [key, value] of Object.entries(table)) {
      if (typeof value === "number" && Number.isFinite(value)) row[key] = value;
    }
    readout[split] = row;
  }
  return {
    ok: true,
    value: { sha256, method, seed, bestEpoch, trainedOnTheseFlights: trainedOn, readout },
  };
}

/** Parse one sample set. Unlike the manifest this is all-or-nothing: a flight the
 *  reader cannot trust would be drawn beside real ones with no way to tell.
 *
 * ``kind`` comes from the MANIFEST ENTRY, and it decides whether the model's
 * words are required or forbidden — a `prior-generated` set without them is
 * half-written, and a `vocabulary-readback` set with them is a set whose kind is
 * wrong. Keying on the kind rather than on whether the field happens to be
 * present is what makes both of those loud.
 */
export function parseTrainingSample(raw: unknown, kind: TrainingSetKind): Parsed<TrainingSample> {
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
  const reading = parseReadingRule(raw.reading);
  if (!reading.ok) return reading;

  // `kinds` is the exporter restating the column order, and it must agree with
  // ours exactly — a disagreement means the columns moved.
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
    const durationS = finite(entry, "durationS");
    if (durationS === null || durationS <= 0) {
      return { ok: false, problem: `${where}: durationS is missing or not a positive length` };
    }
    const reading_numbers: Record<string, number> = {};
    for (const field of ["dtS", "courseWindowRows", "signalWindowRows"]) {
      const value = finite(entry, field);
      if (value === null || value <= 0) {
        return { ok: false, problem: `${where}: ${field} is missing or not positive — the smoothing is stated, never assumed` };
      }
      reading_numbers[field] = value;
    }

    const sentence = parseSentence(entry.sentence, counts, vocabulary.value, durationS, where);
    if (!sentence.ok) return sentence;
    const observed = parseObserved(entry.observed, durationS, where);
    if (!observed.ok) return observed;
    const partial = { observed: observed.value, sentence: sentence.value };
    const envelope = parseEnvelope(
      entry.envelope, partial, sentence.value.words, vocabulary.value, where, "envelope");
    if (!envelope.ok) return envelope;

    let prior: TrainingFlightPrior | undefined;
    if (kind === "prior-generated") {
      const parsed = parseFlightPrior(entry.prior, partial, counts, vocabulary.value, durationS, where);
      if (!parsed.ok) return parsed;
      prior = parsed.value;
    } else if (entry.prior !== undefined) {
      return {
        ok: false,
        problem: `${where} carries a prior block, but this set's kind is ${kind} — a model's words in a read-back set means the kind is wrong`,
      };
    }

    flights.push({
      flightKey,
      callsign,
      runway,
      stratum,
      durationS,
      dtS: reading_numbers.dtS,
      courseWindowRows: reading_numbers.courseWindowRows,
      signalWindowRows: reading_numbers.signalWindowRows,
      sentence: sentence.value,
      observed: observed.value,
      envelope: envelope.value,
      ...(prior ? { prior } : {}),
    });
  }

  let prior: TrainingPrior | undefined;
  if (kind === "prior-generated") {
    const parsed = parsePrior(raw.prior);
    if (!parsed.ok) return parsed;
    prior = parsed.value;
  } else if (raw.prior !== undefined) {
    return {
      ok: false,
      problem: `this sample carries a prior block, but its kind is ${kind} — one of the two is wrong`,
    };
  }

  return {
    ok: true,
    value: {
      setId, airport, vocabulary: vocabulary.value, reading: reading.value,
      ...(prior ? { prior } : {}),
      flights,
    },
  };
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
  kind: TrainingSetKind,
): Promise<Parsed<TrainingSample>> {
  const raw = await fetchJson<unknown>(trainingSamplePath(airportCode, file));
  return parseTrainingSample(raw, kind);
}
