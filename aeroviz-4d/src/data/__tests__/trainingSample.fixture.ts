/**
 * trainingSample.fixture.ts
 * -------------------------
 * MOCK data for the Training views, so T4's components can be built and tested
 * before the exporter exists. It lives in the test tree and is never served: the
 * only thing under `public/data/.../training/` is a real export (design §1).
 *
 * The shape is the real one and the CONSTANTS ARE IMPORTED, never restated — a
 * fixture that types its own schema string or column order is a fixture that
 * cannot catch the contract moving (repo rule: a schema literal in a consumer is
 * a mirror; import it).
 *
 * The bin values below are this vocabulary's own spec, and the word counts they
 * imply (heading 36, altitude 11, speed 23, duration 151, terminal 3, runway 4)
 * are what `trainingWordCounts` derives — so a row here that is in range is in
 * range for the real artefact's defaults too.
 *
 * The flight is a plausible vectored arrival: downwind at +90°, two turns onto
 * the course, a step-down and two decelerations, landing at the last event.
 */

import {
  TERMINAL_CONTINUE,
  TERMINAL_LANDED,
  TRAINING_INDEX_SCHEMA,
  TRAINING_KINDS,
  TRAINING_SAMPLE_SCHEMA,
  trainingWordCounts,
} from "../trainingSample";

/** KRDU's four vocabulary classes — the ARRIVAL manifest's coverage, not the
 *  airport's six thresholds (14 and 32 are excluded upstream). */
export const MOCK_RUNWAY_IDENTS = ["05L", "05R", "23L", "23R"];

/** The bins this vocabulary is read under — the real artefact's own defaults. */
export const MOCK_BINS = {
  headingBinDeg: 10,
  altitudeBinM: 304.8,
  altitudeMaxM: 3048,
  speedBinMps: 5.144444444444445,
  speedMinMps: 51.44444444444444,
  speedMaxMps: 164.62222222222223,
  durationBinS: 2,
  durationMaxS: 300,
  runwayIdents: MOCK_RUNWAY_IDENTS,
};

export const MOCK_VOCABULARY = {
  sha256: "c7a4f4239f52000000000000000000000000000000000000000000000000mock",
  runwaySha256: "aa11bb22cc33000000000000000000000000000000000000000000000000mock",
  readingRule: "plateau-v11",
  ...MOCK_BINS,
  // The artefact states its counts beside its bins and the reader refuses a file
  // where the two disagree — so the fixture DERIVES them rather than typing
  // 36/11/23/4/151/3, which would be a third copy of `Vocabulary.words`.
  words: trainingWordCounts(MOCK_BINS),
};

/**
 * One flight's sentence. Columns are [heading, altitude, speed, runway, duration,
 * terminal] — `TRAINING_KINDS` order.
 *
 * The gaps are deliberately IRREGULAR (26, 44, 38, 22, 58, 34 s): under the
 * retired even grid every gap was 10 s, so a component that still assumes a fixed
 * step draws this flight visibly wrong instead of subtly wrong.
 * The duration word is the gap / 2 s, and the first event's is 0 (nothing before it).
 */
export const MOCK_EVENT_TIMES_S = [0, 26, 70, 108, 130, 188, 222];

export const MOCK_WORDS = [
  //  hdg  alt  spd  rwy  dur  term
  [9, 10, 21, 0, 0, TERMINAL_CONTINUE], // +90° downwind, 10 000 ft, 310 kt
  [9, 10, 17, 0, 13, TERMINAL_CONTINUE], // slowing
  [5, 7, 17, 0, 22, TERMINAL_CONTINUE], // +50°, descending
  [5, 7, 13, 0, 19, TERMINAL_CONTINUE], // slowing again
  [2, 4, 13, 0, 11, TERMINAL_CONTINUE], // +20°, lower
  [0, 4, 9, 0, 29, TERMINAL_CONTINUE], // on the course
  [0, 0, 9, 0, 17, TERMINAL_LANDED], // at the threshold
];

/**
 * The instructions behind the sentence above: one per word change, in issue order.
 * Only the three geometric kinds and the runway are issued — the duration and
 * terminal words are read off the EVENT SEQUENCE, never off this list.
 */
export const MOCK_INSTRUCTIONS = [
  { kind: "heading", word: 9, target: 88.4, issuedS: 0, settledS: 0, clamped: false },
  { kind: "altitude", word: 10, target: 3021.1, issuedS: 0, settledS: 0, clamped: false },
  { kind: "speed", word: 21, target: 159.1, issuedS: 0, settledS: 0, clamped: false },
  { kind: "runway", word: 0, target: 0, issuedS: 0, settledS: 0, clamped: false },
  { kind: "speed", word: 17, target: 138.7, issuedS: 26, settledS: 58, clamped: false },
  { kind: "heading", word: 5, target: 51.7, issuedS: 70, settledS: 94, clamped: false },
  { kind: "altitude", word: 7, target: 2119.6, issuedS: 70, settledS: 102, clamped: false },
  { kind: "speed", word: 13, target: 118.2, issuedS: 108, settledS: 126, clamped: false },
  { kind: "heading", word: 2, target: 21.3, issuedS: 130, settledS: 148, clamped: false },
  { kind: "altitude", word: 4, target: 1211.8, issuedS: 130, settledS: 162, clamped: false },
  { kind: "heading", word: 0, target: -1.2, issuedS: 188, settledS: 202, clamped: false },
  { kind: "speed", word: 9, target: 97.8, issuedS: 188, settledS: 214, clamped: false },
  // The last descent never settles inside the track: it runs to the threshold.
  { kind: "altitude", word: 0, target: 18.4, issuedS: 222, settledS: null, clamped: false },
];

/** Two manoeuvres the labeller read but did not word, one of each of two reasons. */
export const MOCK_ABSORBED = [
  { kind: "heading", startS: 40, endS: 48, word: 9, change: 4.2, reason: "short tail" },
  { kind: "speed", startS: 150, endS: 168, word: 13, change: -2.4, reason: "small change" },
];

/**
 * The observed track, on the artefact's own 2 s rows and in the course's frame.
 * It is GENERATED from a handful of anchors rather than typed out: 132 rows of
 * eight columns would bury the fixture, and every value here only has to be
 * plausible and consistent with the sentence above — a vectored arrival turning
 * from +90° onto the course, descending and slowing all the way in.
 *
 * `tS` starts at 0 and ends at `durationS`, because the reader requires it: both
 * come from the same rows in the export, so a disagreement means the track and
 * the sentence are not one flight.
 */
const ROW_STEP_S = 2;

function ramp(anchors: Array<[number, number]>, t: number): number {
  const last = anchors.length - 1;
  if (t <= anchors[0][0]) return anchors[0][1];
  if (t >= anchors[last][0]) return anchors[last][1];
  const index = anchors.findIndex(([at]) => at > t);
  const [t0, v0] = anchors[index - 1];
  const [t1, v1] = anchors[index];
  return v0 + ((v1 - v0) * (t - t0)) / (t1 - t0);
}

function wrap180(degrees: number): number {
  return (((degrees + 180) % 360) + 360) % 360 - 180;
}

export const MOCK_TRACK_S = 262;

/** KRDU's own numbers, so the mock lands where the real export does. */
const THRESHOLD_LON = -78.7875;
const THRESHOLD_LAT = 35.8776;
const THRESHOLD_HAE_M = 100.0;
const COURSE_RAD = (52 * Math.PI) / 180;

/**
 * The geodetic columns, derived from the same anchors as the frame columns rather
 * than typed out: a flat-earth inverse of `course_frame_rows` at KRDU's latitude,
 * which is all a fixture needs to be the right SHAPE and roughly the right place.
 */
function geodetic(
  tS: number[],
  toGo: Array<[number, number]>,
  cross: Array<[number, number]>,
  height: Array<[number, number]>,
) {
  const lon: number[] = [];
  const lat: number[] = [];
  const altHaeM: number[] = [];
  for (const t of tS) {
    const d = ramp(toGo, t);
    const x = ramp(cross, t);
    const east = -d * Math.cos(COURSE_RAD) + x * Math.sin(COURSE_RAD);
    const north = -d * Math.sin(COURSE_RAD) - x * Math.cos(COURSE_RAD);
    lat.push(THRESHOLD_LAT + north / 111320);
    lon.push(THRESHOLD_LON + east / (111320 * Math.cos((THRESHOLD_LAT * Math.PI) / 180)));
    altHaeM.push(THRESHOLD_HAE_M + ramp(height, t));
  }
  return { lon, lat, altHaeM };
}

export const MOCK_OBSERVED = (() => {
  const tS: number[] = [];
  for (let t = 0; t <= MOCK_TRACK_S; t += ROW_STEP_S) tS.push(t);
  const unwrapped = tS.map((t) => ramp([[0, 90], [70, 90], [94, 50], [130, 50], [148, 20], [188, 20], [202, 0]], t));
  return {
    tS,
    toGoM: tS.map((t) => ramp([[0, 25000], [MOCK_TRACK_S, 0]], t)),
    crossM: tS.map((t) => ramp([[0, 6400], [94, 4200], [148, 1500], [202, 0], [MOCK_TRACK_S, 0]], t)),
    heightM: tS.map((t) => ramp([[0, 3050], [102, 2120], [162, 1210], [MOCK_TRACK_S, 0]], t)),
    relCourseDeg: unwrapped.map(wrap180),
    courseUnwrappedDeg: unwrapped,
    groundSpeedMps: tS.map((t) => ramp([[0, 159], [58, 139], [126, 118], [214, 98], [MOCK_TRACK_S, 70]], t)),
    established: tS.map((t) => (t >= 202 ? 1 : 0)),
    ...geodetic(tS, [[0, 25000], [MOCK_TRACK_S, 0]], [[0, 6400], [94, 4200], [148, 1500], [202, 0], [MOCK_TRACK_S, 0]], [[0, 3050], [102, 2120], [162, 1210], [MOCK_TRACK_S, 0]]),
  };
})();

/**
 * The same sentence flown by `instruction_kinematics`, on its own 1 s steps. Like
 * the observed track it is GENERATED, and it is deliberately NOT the observed one:
 * it starts at the same point (the rules say so) and ends off the centreline,
 * which is what the real export does — a heading word says "fly this angle",
 * never "intercept the centreline".
 */
export const MOCK_GEOMETRIC = (() => {
  const tS: number[] = [];
  for (let t = 0; t <= 300; t += 1) tS.push(t);
  const course = tS.map((t) => ramp([[0, 90], [70, 90], [102, 50], [130, 50], [156, 20], [188, 20], [210, 0]], t));
  return {
    tS,
    toGoM: tS.map((t) => ramp([[0, 25000], [300, 400]], t)),
    crossM: tS.map((t) => ramp([[0, 6400], [102, 4600], [156, 2600], [210, 1500], [300, 1480]], t)),
    heightM: tS.map((t) => ramp([[0, 3050], [110, 2120], [180, 1210], [300, 0]], t)),
    groundSpeedMps: tS.map((t) => ramp([[0, 159], [60, 139], [130, 118], [220, 98], [300, 72]], t)),
    relCourseDeg: course,
    ...geodetic(tS, [[0, 25000], [300, 400]], [[0, 6400], [102, 4600], [156, 2600], [210, 1500], [300, 1480]], [[0, 3050], [110, 2120], [180, 1210], [300, 0]]),
    endReason: "crossed-threshold",
    finalGapM: 1480.5,
    meanGapM: 980.2,
    gapP95M: 1620.4,
    comparedS: 262,
    comparedFraction: 1,
  };
})();

export const MOCK_FLIGHT = {
  flightKey: "DAL123_05L_a1b2c3_1699999999",
  callsign: "DAL123",
  runway: "05L",
  stratum: "vectored",
  // The track outlives the sentence: the last event is at 222 s and the words in
  // force there are held to the threshold at 262 s. In KRDU's real export that
  // tail is a median 145 s of a 326 s arrival, so a fixture without one would let
  // a bar that stops at the last event pass.
  durationS: MOCK_TRACK_S,
  establishedFromStart: false,
  sentence: {
    eventTimesS: MOCK_EVENT_TIMES_S,
    words: MOCK_WORDS,
    durationClamped: 0,
  },
  instructions: MOCK_INSTRUCTIONS,
  absorbed: MOCK_ABSORBED,
  observed: MOCK_OBSERVED,
  geometric: MOCK_GEOMETRIC,
};

/** MIRROR of `instruction_kinematics.assumptions()` — the real defaults. */
export const MOCK_GEOMETRY = {
  method: "instruction-kinematics-v1",
  dtS: 1,
  bankDeg: 20,
  gravityMps2: 9.81,
  heightGainS: 12,
  descentMaxDeg: 6,
  climbMaxDeg: 2,
  accelMaxMps2: 1,
  startsAt: "observed-first-row",
  stopRule: "crossed-threshold or time-cap at the observed duration + 120 s",
  windModelled: false,
  aircraftTypeModelled: false,
  constantsFrom: ["outputs/guidance/route.py", "outputs/guidance/controller.py"],
};

export const MOCK_SAMPLE = {
  schema: TRAINING_SAMPLE_SCHEMA,
  setId: "vocabulary_tau10",
  airport: "KRDU",
  kinds: [...TRAINING_KINDS],
  vocabulary: MOCK_VOCABULARY,
  geometry: MOCK_GEOMETRY,
  flights: [MOCK_FLIGHT],
};

export const MOCK_INDEX = {
  schema: TRAINING_INDEX_SCHEMA,
  writtenUtc: "2026-09-20T12:00:00Z",
  airport: "KRDU",
  sets: [
    {
      id: "vocabulary_tau10",
      kind: "vocabulary-readback",
      title: "Instruction vocabulary (six kinds, event sequence)",
      file: "vocabulary_tau10/sample.json",
      vocabularySha256: MOCK_VOCABULARY.sha256,
      runwaySha256: MOCK_VOCABULARY.runwaySha256,
      readingRule: MOCK_VOCABULARY.readingRule,
      flights: 1,
    },
  ],
};

/** A deep copy, so a test that corrupts one field cannot leak into the next. */
export function mockSample(): Record<string, unknown> {
  return structuredClone(MOCK_SAMPLE);
}

export function mockIndex(): Record<string, unknown> {
  return structuredClone(MOCK_INDEX);
}
