/**
 * trainingSample.ts
 * -----------------
 * The Training task's data contract: the manifest of exported sample sets and one
 * set's flights (track + sentence + instructions, and later the geometric track).
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.
 *
 * A WORD IS A BAND, NOT A POINT (reading rule `segment-v12`, 2026-09-21). The
 * vertical and speed words each carry a tolerance: an executor that stays inside
 * it has obeyed the word. So a sentence does not name one track, it names a
 * family, and this module exposes the tolerances (`verticalToleranceDeg`,
 * `speedToleranceMps`) beside the centres. The kinds that have no tolerance
 * return `null` from `trainingWordTolerance` — a band drawn on them would be an
 * invented number (design §5.6, V36).
 *
 * THE SENTENCE IS AN EVENT SEQUENCE, NOT AN EVEN GRID (2026-09-20). One row per moment something changed; the gaps between rows are
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
// No unit conversion is imported here any more: this vocabulary is DEFINED in
// SI (metres, m/s, degrees), so the labels print the stored numbers. The feet
// and knots the retired bins were defined in were the reason for the conversion,
// and they went with them (V30).

/** MIRROR of the exporter's schema strings. A file that does not carry these is
 *  refused by name rather than read leniently. */
export const TRAINING_INDEX_SCHEMA = "aeroviz-training-index-v1";
export const TRAINING_SAMPLE_SCHEMA = "aeroviz-training-sample-v1";

/**
 * MIRROR of `instructions.READING_RULE`, and REFUSED on mismatch.
 *
 * The file carries its own spec — bins, centres, tolerances — so most of what a
 * word means is read from it. What is NOT in the file is the rule's semantics:
 * that the second column is an angle, that DESCENT IS POSITIVE, that a vertical
 * instruction's `settledS` is its segment's end rather than a plateau's. This
 * reader hardcodes all three. So it is bound to the rule that produced the file,
 * and says so by name rather than reading an older artefact into today's
 * meanings (the Python side refuses across rules for the same reason).
 *
 * The cost of the pin is a one-line edit when the rule bumps; the cost of not
 * pinning it is a chart that looks right and means something else.
 */
export const TRAINING_READING_RULE = "segment-v12";

/**
 * MIRROR of `ts_transformer.manoeuvre.instructions.INSTRUCTION_KINDS` — the six
 * word kinds IN ORDER. The columns of `words` are positional, so this order is
 * load-bearing: reorder it and every word is read as another kind's.
 * (The intercept word was deleted on 2026-09-20, D73: it was the heading word's
 * shadow — all 4,897 were issued at the same instant as a heading word. The
 * altitude word became the VERTICAL word on 2026-09-21: a flight path angle
 * instead of a height, read off a piecewise fit of the profile.)
 */
export const TRAINING_KINDS = [
  "heading",
  "vertical",
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
  vertical: 1,
  speed: 2,
  runway: 3,
  duration: 4,
  terminal: 5,
};

/**
 * MIRROR of `instructions.LEVEL_MODE_DEG`: the vertical mode that means level
 * flight. It is named rather than spotted, because it is the one mode whose
 * tolerance is absolute — a percentage of zero is no tolerance at all.
 *
 * DESCENT IS POSITIVE in this vocabulary, so the go-around mode is the NEGATIVE
 * one. Nothing may print a bare signed angle: a reader seeing "-3.0°" reads a
 * descent, which is exactly backwards (V37).
 */
export const TRAINING_LEVEL_MODE_DEG = 0;

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
  /**
   * The flight path angles the vertical word can name, DESCENT POSITIVE — a
   * TABLE, not a bin width. The descent modes were fitted to the data and
   * rounded to one decimal, so there is no formula to derive them from, and the
   * class count is the table's length.
   */
  verticalModesDeg: number[];
  /** How many straight segments one approach's profile is cut into. Not a word
   *  count — it is how the labeller read the profile, and the panel states it
   *  beside the vocabulary's other properties. */
  verticalSegments: number;
  /** Ground speed centres in m/s — again a fitted TABLE, not min + k × step.
   *  Control assigns indicated airspeed and the wind is inside this number, so
   *  ground speed carries no whole-knot structure to align to. */
  speedCentresMps: number[];
  /** The level mode's tolerance, in degrees and ABSOLUTE. */
  verticalLevelToleranceDeg: number;
  /** Every other vertical mode's tolerance, as a fraction of its own angle. */
  verticalToleranceFraction: number;
  /** The speed word's tolerance, as a fraction of its own centre. */
  speedToleranceFraction: number;
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
  /** The value this word was read from, unbinned, in the kind's own unit:
   *  degrees relative to the final approach course (heading), degrees of flight
   *  path angle with DESCENT POSITIVE (vertical — an angle, not a height), m/s
   *  ground speed. The runway instruction has no target at all (its WORD is the
   *  answer) and the exporter writes `instructions.NO_TARGET`, 0.0, there — so
   *  nothing may print a target for a runway word. */
  target: number;
  issuedS: number;
  /** When the manoeuvre finished, or `null` when it never settled inside the
   *  track — a deceleration still slowing at the threshold has none. Null is a
   *  real answer here, not a missing field.
   *
   *  For a VERTICAL instruction this is the segment's end rather than a
   *  plateau's: the segments tile the profile, so each one's `settledS` is the
   *  next one's `issuedS` and the last runs to the end of the record. */
  settledS: number | null;
  /** Whether the target fell outside the vocabulary's range and was clamped. */
  clamped: boolean;
}

/**
 * The geodetic columns BOTH tracks carry, for the 3D layer.
 *
 * THE ALTITUDE IS HAE, and the name says so. A record is MSL; Cesium reads
 * `cartographicDegrees` altitude as metres above the WGS84 ellipsoid, so the
 * exporter converts on the way out exactly as the CZML path does. Since
 * h = H + N and N is negative here (-33.5 m at KRDU), a line handed the MSL
 * number renders |N| too HIGH, floating above its own terrain.
 */
export const TRAINING_GEODETIC_COLUMNS = ["lon", "lat", "altHaeM"] as const;
export type TrainingGeodeticColumn = (typeof TRAINING_GEODETIC_COLUMNS)[number];

/**
 * MIRROR of `instructions.course_frame` — the observed track in the FINAL
 * APPROACH COURSE's frame, which is the frame the words were read in. Reading
 * the words against anything else (a geodetic track, say) would let the charts
 * and the words disagree about where the aircraft was.
 *
 * Every column has one value per row of `tS`. The geodetic columns are carried
 * beside these, as `TRAINING_GEODETIC_COLUMNS`.
 */
export const TRAINING_OBSERVED_COLUMNS = [
  "toGoM",              // along the course, positive BEFORE the threshold
  "crossM",             // right of the course, positive
  "heightM",            // above the threshold
  // Cumulative HORIZONTAL distance from the first row — the axis the vertical
  // word was read on (`instructions._profile`: the profile is fitted as height
  // against this, so the word IS this curve's slope). It is exported rather
  // than integrated here so the frontend does not need a second copy of
  // `MINIMUM_GROUND_SPEED_MPS`.
  "pathM",
  "relCourseDeg",       // ground track against the course, wrapped
  "courseUnwrappedDeg", // the same, unwrapped along the rows
  "groundSpeedMps",
  "established",        // 0 / 1 per row
] as const;

export type TrainingObservedColumn = (typeof TRAINING_OBSERVED_COLUMNS)[number];

export type TrainingObserved = { tS: number[] }
  & Record<TrainingObservedColumn, number[]>
  & Record<TrainingGeodeticColumn, number[]>;

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

/**
 * The vertical word's tolerance, flown (design §5.6). It is TWO HEIGHT COLUMNS,
 * not two tracks, and that is a property of the kinematics rather than a saving:
 * the commanded angle enters only the height step, so the horizontal columns and
 * the stopping time of the edges are identical to the nominal track's, row for
 * row. `altHaeLoM` / `altHaeHiM` are the same two heights as HAE, for the wall
 * the 3D layer draws between them.
 */
export const TRAINING_VERTICAL_BAND_COLUMNS = [
  "heightLoM", "heightHiM", "altHaeLoM", "altHaeHiM",
] as const;
export type TrainingVerticalBandColumn = (typeof TRAINING_VERTICAL_BAND_COLUMNS)[number];
export type TrainingVerticalBand = Record<TrainingVerticalBandColumn, number[]>;

/**
 * The speed word's tolerance, flown. This one IS two tracks: a speed change
 * moves the horizontal step, the turn radius and the moment of crossing. They
 * carry no geodetic columns — they run within ~100 m of the nominal line, so the
 * 3D layer does not draw them (design §5.6).
 */
export const TRAINING_SPEED_EDGE_COLUMNS = ["toGoM", "crossM"] as const;
export type TrainingSpeedEdgeColumn = (typeof TRAINING_SPEED_EDGE_COLUMNS)[number];

export type TrainingSpeedEdge = { tS: number[] }
  & Record<TrainingSpeedEdgeColumn, number[]> & {
  endReason: TrainingEndReason;
  /** When this edge stopped, on the track's own clock. It is what
   *  `arrivalWindowS` is made of, and it is checked against it. */
  endS: number;
};

export interface TrainingSpeedBand {
  /** Every speed word at the BOTTOM of its band, and at the top. Only the target
   *  moves: the start speed is the observation's first row for all three runs,
   *  so they share a first point by construction (design §5.4-1). */
  low: TrainingSpeedEdge;
  high: TrainingSpeedEdge;
  /**
   * `[the fast edge's crossing, the slow edge's crossing]` — the window the
   * speed tolerance alone opens on the arrival time, which is the one place that
   * tolerance is legible. `null` when either edge never reached the runway; then
   * the edge's own `endReason` says why, and nothing may print a window.
   */
  arrivalWindowS: [number, number] | null;
}

export type TrainingGeometric = { tS: number[] }
  & Record<TrainingGeometricColumn, number[]>
  & Record<TrainingGeodeticColumn, number[]> & {
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
  /** One kind's slack at a time — NOT the joint 2×2 envelope (V32). The
   *  `geometry` block states that in the file, as `bandsAreJoint: false`. */
  verticalBand: TrainingVerticalBand;
  speedBand: TrainingSpeedBand;
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
  /** What the flown tracks and their bands were drawn under. It travels with the
   *  selection because the views that draw the corridor are the ones that have
   *  to say where it came from — a band on screen whose rule is two components
   *  away is an approximation nobody can see stated (design §5.4). */
  geometry: TrainingGeometry;
}

/**
 * MIRROR of `instruction_kinematics.assumptions()`: what the flown sentences were
 * drawn under. It is REQUIRED and every field of it is SHOWN, in the panel's ⓘ
 * or in the read-back window's legend — an approximation nobody can see stated
 * is worse than none, and the block claims in its own docstring that the view
 * shows it.
 *
 * THREE OF THESE FIELDS HAVE NO PRODUCER YET: `verticalBandFrom`,
 * `speedBandFrom` and `bandsAreJoint` are written by the exporter step that also
 * writes the two bands (design §7, T10), which is not built. They are required
 * here rather than optional because the repo settles a schema BEFORE the
 * experiment that writes it runs — an export missing them is an export from
 * unfinished code, and it is refused by name rather than read half-way.
 */
export interface TrainingGeometry {
  method: string;
  dtS: number;
  bankDeg: number;
  gravityMps2: number;
  /** The vertical word IS the commanded angle, so there is no height error to
   *  close — which is why the height time constant is gone from this block. */
  verticalIsCommandedAngle: boolean;
  /** Where the executor levels off rather than flying through the runway. The
   *  vertical band closes onto this floor, so the widest part of the fan is
   *  BEFORE the threshold: that is the floor's doing, not the vocabulary's,
   *  and the legend has to say so (V34). */
  heightFloorM: number;
  descentMaxDeg: number;
  climbMaxDeg: number;
  accelMaxMps2: number;
  startsAt: string;
  stopRule: string;
  windModelled: boolean;
  aircraftTypeModelled: boolean;
  /** Which tolerance each band was flown from. A corridor on screen that cannot
   *  be traced back to the number that drew it is an approximation nobody can
   *  see stated. */
  verticalBandFrom: string;
  speedBandFrom: string;
  /** Whether the two bands are the joint envelope. It is `false` and it is
   *  stated, because the drawn corridor is one kind at a time (V32). */
  bandsAreJoint: boolean;
  constantsFrom: string[];
}

export interface TrainingSample {
  setId: string;
  airport: string;
  vocabulary: TrainingVocabulary;
  geometry: TrainingGeometry;
  flights: TrainingFlight[];
}

/** The part of a vocabulary that decides what a word MEANS: the bins and the
 *  runway classes. Everything that reads words takes this, so `parseVocabulary`
 *  can derive the class counts while it is still building the vocabulary. */
export type TrainingWordSpec = Pick<
  TrainingVocabulary,
  | "headingBinDeg" | "verticalModesDeg" | "verticalSegments" | "speedCentresMps"
  | "verticalLevelToleranceDeg" | "verticalToleranceFraction" | "speedToleranceFraction"
  | "durationBinS" | "durationMaxS" | "runwayIdents"
>;

export type Parsed<T> = { ok: true; value: T } | { ok: false; problem: string };

// ── word counts ──────────────────────────────────────────────────────────────

/**
 * How many classes each kind has, under this file's own spec.
 *
 * MIRROR of `Vocabulary.heading_words / vertical_words / speed_words /
 * duration_words` and `word_counts()`: the same formulas, so a word index out of
 * range is caught here instead of indexing a legend off its end. Computing them
 * from the file's spec (rather than hardcoding 72/6/16/151) is what lets one
 * reader serve a re-binned vocabulary and every airport's runway list.
 *
 * Two of them are TABLE LENGTHS now, not divisions: the vertical modes and the
 * speed centres are fitted values with no step to divide by.
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
    vertical: vocabulary.verticalModesDeg.length,
    speed: vocabulary.speedCentresMps.length,
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

/**
 * MIRROR of `Vocabulary.vertical_centre_deg`: the commanded flight path angle in
 * degrees, DESCENT POSITIVE. It is a rate, not a place — there is no height to
 * converge on, which is why the flown track integrates it directly.
 */
export function verticalCentreDeg(vocabulary: TrainingWordSpec, word: number): number {
  return vocabulary.verticalModesDeg[word];
}

/** MIRROR of `Vocabulary.speed_centre_mps`: ground speed, from the table. */
export function speedCentreMps(vocabulary: TrainingWordSpec, word: number): number {
  return vocabulary.speedCentresMps[word];
}

/**
 * MIRROR of `Vocabulary.vertical_tolerance_deg` — INCLUDING the level mode's
 * branch. The level mode gets an absolute tolerance because a fraction of zero
 * is no tolerance at all, and level flight is 12 % of the segments; every other
 * mode gets a fraction of its own angle.
 */
export function verticalToleranceDeg(vocabulary: TrainingWordSpec, word: number): number {
  const centre = verticalCentreDeg(vocabulary, word);
  if (centre === TRAINING_LEVEL_MODE_DEG) return vocabulary.verticalLevelToleranceDeg;
  return Math.abs(centre) * vocabulary.verticalToleranceFraction;
}

/** MIRROR of `Vocabulary.speed_tolerance`: a fraction of the commanded speed. */
export function speedToleranceMps(vocabulary: TrainingWordSpec, word: number): number {
  return speedCentreMps(vocabulary, word) * vocabulary.speedToleranceFraction;
}

/**
 * How far a word lets the executor sit from its centre, in the kind's own unit,
 * or `null` for the kinds that carry NO tolerance (design §5.6, V36).
 *
 * The null is the point of this function: the heading, runway, duration and
 * terminal words have no redundancy, so a view that drew a band on them would be
 * inventing a number. The plateau tolerances the LABELLER used
 * (`course_tolerance_deg`, `speed_tolerance_mps`) are a different quantity for a
 * different purpose and are not these.
 */
export function trainingWordTolerance(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): number | null {
  if (kind === "vertical") return verticalToleranceDeg(vocabulary, word);
  if (kind === "speed") return speedToleranceMps(vocabulary, word);
  return null;
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
 * THE UNITS ARE SI, because this vocabulary is defined in SI: the speed centres
 * were fitted in m/s and rounded to 1 m/s (44, 56, 63 …), so printing knots
 * would label them 85.6 kt and 108.8 kt — arithmetic the reader would have to
 * undo to recognise the vocabulary. That is the same argument the retired feet
 * and knots labels rested on, pointing the other way now that the bins moved
 * (V30).
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
    case "vertical": {
      const centre = verticalCentreDeg(vocabulary, word);
      if (centre === TRAINING_LEVEL_MODE_DEG) return "level";
      // The ARROW says which way, never the sign: descent is positive here, so
      // a bare "-3.0°" reads as a descent to everyone who has not read the
      // vocabulary (V37).
      return `${centre > 0 ? "\u2193" : "\u2191"}${Math.abs(centre).toFixed(1)}\u00b0`;
    }
    case "speed":
      return `${speedCentreMps(vocabulary, word)} m/s`;
    case "runway":
      return vocabulary.runwayIdents[word];
    case "duration":
      return `${durationCentreS(vocabulary, word)} s`;
    case "terminal":
      return TERMINAL_LABELS[word];
  }
}

/**
 * The same word WITH the band it allows — `↓3.1°±0.22`, `93 m/s±2.8` — which is
 * what the sentence bar writes on a band, because the tolerance is half of what
 * the word says (design §3.1). A kind without a tolerance reads exactly as
 * `trainingWordLabel`.
 */
export function trainingWordBandLabel(
  vocabulary: TrainingWordSpec,
  kind: TrainingKind,
  word: number,
): string {
  const label = trainingWordLabel(vocabulary, kind, word);
  const tolerance = trainingWordTolerance(vocabulary, kind, word);
  if (tolerance === null) return label;
  return `${label}\u00b1${tolerance.toFixed(kind === "vertical" ? 2 : 1)}`;
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
    "headingBinDeg", "verticalSegments",
    "verticalLevelToleranceDeg", "verticalToleranceFraction", "speedToleranceFraction",
    "durationBinS", "durationMaxS",
  ];
  const values: Record<string, number> = {};
  for (const field of numbers) {
    const value = finite(raw, field);
    if (value === null) return { ok: false, problem: `vocabulary.${field} is missing or not a number` };
    values[field] = value;
  }
  // Every one of these is positive in `Vocabulary.__post_init__` — including the
  // three tolerances, because a tolerance of zero is a band nothing can sit in
  // and would quietly turn every word into an unmeetable point target.
  for (const field of ["headingBinDeg", "verticalSegments", "durationBinS",
                       "verticalLevelToleranceDeg", "verticalToleranceFraction",
                       "speedToleranceFraction"]) {
    if (values[field] <= 0) return { ok: false, problem: `vocabulary.${field} must be positive` };
  }
  // Both mirror `Vocabulary.__post_init__`. The divisibility one is not
  // decoration: `trainingWordCounts` ROUNDS 360/bin, so a 7° bin would give 51
  // classes and read every heading word near the wrap into the wrong one.
  if (360 % values.headingBinDeg !== 0) {
    return { ok: false, problem: `vocabulary.headingBinDeg is ${values.headingBinDeg}, which does not divide 360°` };
  }
  if (!Number.isInteger(values.verticalSegments)) {
    return { ok: false, problem: `vocabulary.verticalSegments is ${values.verticalSegments}, not a count of segments` };
  }
  // The two tables. Sorted and distinct is `Vocabulary.__post_init__`'s check as
  // well: a word is the NEAREST centre, so two equal centres would make one of
  // them unreachable and an unsorted table would break the reading rule.
  const tables: Record<string, number[]> = {};
  for (const field of ["verticalModesDeg", "speedCentresMps"] as const) {
    const values_ = numberArray(raw[field]);
    if (values_ === null || values_.length < 2) {
      return { ok: false, problem: `vocabulary.${field} is missing or has fewer than two centres` };
    }
    for (let i = 1; i < values_.length; i += 1) {
      if (!(values_[i] > values_[i - 1])) {
        return {
          ok: false,
          problem: `vocabulary.${field} is not stored sorted and distinct (index ${i}: ${values_[i - 1]} then ${values_[i]})`,
        };
      }
    }
    tables[field] = values_;
  }
  // The level mode is the one the tolerance branches on, so a table without it
  // would send every vertical word down the fraction branch — silently, since
  // `verticalToleranceDeg` would still return a number.
  if (!tables.verticalModesDeg.includes(TRAINING_LEVEL_MODE_DEG)) {
    return {
      ok: false,
      problem: `vocabulary.verticalModesDeg has no level mode (${TRAINING_LEVEL_MODE_DEG}\u00b0): it is the one mode whose tolerance is absolute`,
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
        `${TRAINING_READING_RULE} — the rule decides what the columns MEAN (the second is an angle, ` +
        `descent positive), which no field of the file can say. Re-export under the current rule`,
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
    headingBinDeg: values.headingBinDeg,
    verticalModesDeg: tables.verticalModesDeg,
    verticalSegments: values.verticalSegments,
    speedCentresMps: tables.speedCentresMps,
    verticalLevelToleranceDeg: values.verticalLevelToleranceDeg,
    verticalToleranceFraction: values.verticalToleranceFraction,
    speedToleranceFraction: values.speedToleranceFraction,
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

  return { ok: true, value: columns as TrainingObserved };
}

function parseGeometric(raw: unknown, where: string): Parsed<TrainingGeometric> {
  if (!isRecord(raw)) return { ok: false, problem: `${where}.geometric is not an object` };
  const tS = numberArray(raw.tS);
  if (tS === null || tS.length < 2) {
    return { ok: false, problem: `${where}.geometric.tS is missing or shorter than two steps` };
  }
  // `rowAt`, the x axis and the gap readout all assume this clock runs forward
  // from 0, exactly as the observed one does.
  if (tS[0] !== 0) {
    return { ok: false, problem: `${where}.geometric.tS starts at ${tS[0]} s, not 0` };
  }
  for (let i = 1; i < tS.length; i += 1) {
    if (!(tS[i] > tS[i - 1])) {
      return { ok: false, problem: `${where}.geometric.tS is not increasing at step ${i}` };
    }
  }

  const columns: Record<string, number[]> = { tS };
  for (const column of [...TRAINING_GEOMETRIC_COLUMNS, ...TRAINING_GEODETIC_COLUMNS]) {
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

  const verticalBand = parseVerticalBand(raw.verticalBand, tS.length, where);
  if (!verticalBand.ok) return verticalBand;
  const speedBand = parseSpeedBand(raw.speedBand, where);
  if (!speedBand.ok) return speedBand;

  return {
    ok: true,
    value: {
      ...(columns as { tS: number[] }
        & Record<TrainingGeometricColumn, number[]>
        & Record<TrainingGeodeticColumn, number[]>),
      endReason: endReason as TrainingEndReason,
      finalGapM: numbers.finalGapM,
      meanGapM: numbers.meanGapM,
      gapP95M: numbers.gapP95M,
      comparedS: numbers.comparedS,
      comparedFraction: numbers.comparedFraction,
      verticalBand: verticalBand.value,
      speedBand: speedBand.value,
    },
  };
}

/**
 * The vertical tolerance flown, checked to be ALIGNED WITH THE NOMINAL TRACK.
 *
 * The row count is not a formality here: the edges share the nominal track's
 * `tS`, `toGoM` and `crossM` — that is what makes two height columns a legal
 * substitute for two tracks — so a band of a different length is not a band that
 * can be drawn at all, and reading it with the nominal x values would draw a
 * corridor that is simply somewhere else.
 */
function parseVerticalBand(raw: unknown, rows: number, where: string): Parsed<TrainingVerticalBand> {
  if (!isRecord(raw)) {
    return { ok: false, problem: `${where}.geometric.verticalBand is missing: the flown sentence carries the vertical word's tolerance` };
  }
  const columns: Record<string, number[]> = {};
  for (const column of TRAINING_VERTICAL_BAND_COLUMNS) {
    const values = numberArray(raw[column]);
    if (values === null || values.length !== rows) {
      return {
        ok: false,
        problem: `${where}.geometric.verticalBand.${column} is missing or does not have the nominal track's ${rows} rows`,
      };
    }
    columns[column] = values;
  }
  // Lo is the SHALLOWER edge and stays above: the band is ordered, and a file
  // where it is not has its two edges swapped, which would draw the fan inside
  // out without changing its width.
  for (let row = 0; row < rows; row += 1) {
    if (columns.heightLoM[row] < columns.heightHiM[row]) {
      return {
        ok: false,
        problem:
          `${where}.geometric.verticalBand is inverted at row ${row}: heightLoM ${columns.heightLoM[row]} ` +
          `is below heightHiM ${columns.heightHiM[row]} — lo is the SHALLOWER descent, so it stays above`,
      };
    }
  }
  return { ok: true, value: columns as TrainingVerticalBand };
}

/** One edge of the speed tolerance: a track of its own, plus where it stopped. */
function parseSpeedEdge(raw: unknown, where: string): Parsed<TrainingSpeedEdge> {
  if (!isRecord(raw)) return { ok: false, problem: `${where} is missing or not an object` };
  const tS = numberArray(raw.tS);
  if (tS === null || tS.length < 2) {
    return { ok: false, problem: `${where}.tS is missing or shorter than two steps` };
  }
  const columns: Record<string, number[]> = { tS };
  for (const column of TRAINING_SPEED_EDGE_COLUMNS) {
    const values = numberArray(raw[column]);
    if (values === null || values.length !== tS.length) {
      return { ok: false, problem: `${where}.${column} is missing or does not have ${tS.length} rows` };
    }
    columns[column] = values;
  }
  const endReason = str(raw, "endReason");
  if (endReason === null || !(TRAINING_END_REASONS as readonly string[]).includes(endReason)) {
    return {
      ok: false,
      problem: `${where}.endReason is ${JSON.stringify(raw.endReason)}, expected one of ${TRAINING_END_REASONS.join(", ")}`,
    };
  }
  const endS = finite(raw, "endS");
  if (endS === null || endS <= 0) return { ok: false, problem: `${where}.endS is missing or not a time` };
  if (Math.abs(endS - tS[tS.length - 1]) > 0.05) {
    return { ok: false, problem: `${where}.endS is ${endS} s but its own track ends at ${tS[tS.length - 1]} s` };
  }
  return {
    ok: true,
    value: {
      ...(columns as { tS: number[] } & Record<TrainingSpeedEdgeColumn, number[]>),
      endReason: endReason as TrainingEndReason,
      endS,
    },
  };
}

/**
 * The speed tolerance flown, and the arrival window it opens.
 *
 * The window is refused unless BOTH edges crossed the threshold: a window whose
 * far end is a time cap is not a window, it is the integration budget, and
 * printing it as "arrives between" would be a measurement of the stopping rule.
 */
function parseSpeedBand(raw: unknown, where: string): Parsed<TrainingSpeedBand> {
  if (!isRecord(raw)) {
    return { ok: false, problem: `${where}.geometric.speedBand is missing: the flown sentence carries the speed word's tolerance` };
  }
  const low = parseSpeedEdge(raw.low, `${where}.geometric.speedBand.low`);
  if (!low.ok) return low;
  const high = parseSpeedEdge(raw.high, `${where}.geometric.speedBand.high`);
  if (!high.ok) return high;

  // The window is DERIVED from the two edges, so it is checked against them in
  // both directions. One direction alone leaves the half a reader actually sees:
  // a null window beside two edges that both landed renders as "no window: an
  // edge ended on crossed-threshold", which names a reason that is not one.
  const bothCrossed =
    low.value.endReason === "crossed-threshold" && high.value.endReason === "crossed-threshold";
  const window = raw.arrivalWindowS;
  if (window === null) {
    if (bothCrossed) {
      return {
        ok: false,
        problem:
          `${where}.geometric.speedBand.arrivalWindowS is null, but both edges crossed the threshold ` +
          `(${high.value.endS} s and ${low.value.endS} s) — that is a window`,
      };
    }
    return { ok: true, value: { low: low.value, high: high.value, arrivalWindowS: null } };
  }
  const pair = numberArray(window);
  if (pair === null || pair.length !== 2) {
    return { ok: false, problem: `${where}.geometric.speedBand.arrivalWindowS is ${JSON.stringify(window)}, expected two times or null` };
  }
  if (!bothCrossed) {
    return {
      ok: false,
      problem:
        `${where}.geometric.speedBand.arrivalWindowS is a window, but an edge ended on ` +
        `${low.value.endReason === "crossed-threshold" ? high.value.endReason : low.value.endReason} — ` +
        `an edge that never reached the runway has no arrival time`,
    };
  }
  // It is the two edges' own crossings, in order: fast first. Checking the
  // VALUES rather than just their order is what makes the field a mirror of the
  // edges instead of a second opinion about them — and it is what catches a file
  // whose `low` and `high` are the wrong way round, which nothing else would.
  if (pair[0] !== high.value.endS || pair[1] !== low.value.endS) {
    return {
      ok: false,
      problem:
        `${where}.geometric.speedBand.arrivalWindowS is [${pair[0]}, ${pair[1]}], but its edges ` +
        `cross at ${high.value.endS} s (high) and ${low.value.endS} s (low) — the window is the two ` +
        `crossings, fast first`,
    };
  }
  if (!(pair[0] <= pair[1])) {
    return {
      ok: false,
      problem:
        `${where}.geometric.speedBand.arrivalWindowS is [${pair[0]}, ${pair[1]}]: the HIGH edge is ` +
        `the faster one and arrives first, so these two edges are swapped`,
    };
  }
  return { ok: true, value: { low: low.value, high: high.value, arrivalWindowS: [pair[0], pair[1]] } };
}

function parseGeometry(raw: unknown): Parsed<TrainingGeometry> {
  if (!isRecord(raw)) return { ok: false, problem: "geometry is missing: the flown tracks state what they were drawn under" };
  const numbers: Record<string, number> = {};
  for (const field of ["dtS", "bankDeg", "gravityMps2", "heightFloorM", "descentMaxDeg",
                       "climbMaxDeg", "accelMaxMps2"]) {
    const value = finite(raw, field);
    if (value === null) return { ok: false, problem: `geometry.${field} is missing or not a number` };
    numbers[field] = value;
  }
  const strings: Record<string, string> = {};
  for (const field of ["method", "startsAt", "stopRule", "verticalBandFrom", "speedBandFrom"]) {
    const value = str(raw, field);
    if (value === null) return { ok: false, problem: `geometry.${field} is missing or not a non-empty string` };
    strings[field] = value;
  }
  for (const field of ["windModelled", "aircraftTypeModelled", "verticalIsCommandedAngle", "bandsAreJoint"]) {
    if (typeof raw[field] !== "boolean") {
      return { ok: false, problem: `geometry.${field} is ${JSON.stringify(raw[field])}, expected a boolean` };
    }
  }
  const sources = raw.constantsFrom;
  if (!Array.isArray(sources) || !sources.every((item) => typeof item === "string")) {
    return { ok: false, problem: "geometry.constantsFrom is missing or not a list of files" };
  }

  return {
    ok: true,
    value: {
      method: strings.method,
      startsAt: strings.startsAt,
      stopRule: strings.stopRule,
      dtS: numbers.dtS,
      bankDeg: numbers.bankDeg,
      gravityMps2: numbers.gravityMps2,
      heightFloorM: numbers.heightFloorM,
      descentMaxDeg: numbers.descentMaxDeg,
      climbMaxDeg: numbers.climbMaxDeg,
      accelMaxMps2: numbers.accelMaxMps2,
      windModelled: raw.windModelled as boolean,
      aircraftTypeModelled: raw.aircraftTypeModelled as boolean,
      verticalIsCommandedAngle: raw.verticalIsCommandedAngle as boolean,
      verticalBandFrom: strings.verticalBandFrom,
      speedBandFrom: strings.speedBandFrom,
      bandsAreJoint: raw.bandsAreJoint as boolean,
      constantsFrom: sources as string[],
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
  // A manoeuvre cannot settle before it was issued. This is not a formality: a
  // vertical instruction's span IS `issuedS`…`settledS`, and a reversed pair
  // gives a segment with no rows in it — which does not draw, does not warn, and
  // leaves its rows out of the "inside the band" denominator, so a corrupt
  // export reads as a BETTER result than a good one.
  if (settled !== null && settled < issuedS) {
    return {
      ok: false,
      problem: `${where} settles at ${settled} s but was issued at ${issuedS} s — that span is empty`,
    };
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
  const geometry = parseGeometry(raw.geometry);
  if (!geometry.ok) return geometry;

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

  return {
    ok: true,
    value: { setId, airport, vocabulary: vocabulary.value, geometry: geometry.value, flights },
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
): Promise<Parsed<TrainingSample>> {
  const raw = await fetchJson<unknown>(trainingSamplePath(airportCode, file));
  return parseTrainingSample(raw);
}
