/**
 * trainingSample.fixture.ts
 * -------------------------
 * MOCK data for the Training views, so the components can be built and tested
 * before the exporter has run. It lives in the test tree and is never served:
 * the only thing under `public/data/.../training/` is a real export (design §1).
 *
 * The shape is the real one and the CONSTANTS ARE IMPORTED, never restated — a
 * fixture that types its own schema string or column order is a fixture that
 * cannot catch the contract moving (repo rule: a schema literal in a consumer is
 * a mirror; import it).
 *
 * The spec below is this vocabulary's own (`segment-v12`, 2026-09-21), and the
 * word counts it implies (heading 72, vertical 6, speed 16, duration 151,
 * terminal 3, runway 4) are what `trainingWordCounts` derives — so a row here
 * that is in range is in range for the real artefact's defaults too.
 *
 * The flight is a plausible vectored arrival: downwind at +90°, three turns onto
 * the course, level then two descent segments, three decelerations, landing at
 * the last event. Its VERTICAL profile is built backwards from the words, so the
 * angles the instructions claim are the angles the height column actually has —
 * a fixture whose profile disagreed with its own words would let a chart that
 * draws the wrong one pass.
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

/**
 * The spec this vocabulary is read under — the real artefact's own defaults.
 *
 * The two tables are the point: the vertical modes and the speed centres were
 * FITTED (rounded to one decimal and to 1 m/s), so there is no bin width to
 * derive them from and the class count is the table's length.
 */
export const MOCK_SPEC = {
  headingBinDeg: 5,
  verticalModesDeg: [-3.0, 0.0, 1.4, 2.4, 3.1, 4.4],
  verticalSegments: 5,
  speedCentresMps: [44, 56, 63, 68, 74, 79, 86, 93, 99, 107, 114, 121, 129, 138, 147, 157],
  verticalLevelToleranceDeg: 0.1,
  verticalToleranceFraction: 0.07,
  speedToleranceFraction: 0.03,
  durationBinS: 2,
  durationMaxS: 300,
  runwayIdents: MOCK_RUNWAY_IDENTS,
};

/** The words this fixture's flight uses, by name, so a reader of the rows below
 *  does not have to count table positions. */
export const LEVEL = 1;
export const DESCEND_24 = 3;
export const DESCEND_31 = 4;

export const MOCK_VOCABULARY = {
  sha256: "c7a4f4239f52000000000000000000000000000000000000000000000000mock",
  runwaySha256: "aa11bb22cc33000000000000000000000000000000000000000000000000mock",
  readingRule: "segment-v12",
  ...MOCK_SPEC,
  // The artefact states its counts beside its spec and the reader refuses a file
  // where the two disagree — so the fixture DERIVES them rather than typing
  // 72/6/16/4/151/3, which would be a third copy of `Vocabulary.words`.
  words: trainingWordCounts(MOCK_SPEC),
};

/**
 * One flight's sentence. Columns are [heading, vertical, speed, runway, duration,
 * terminal] — `TRAINING_KINDS` order.
 *
 * The gaps are deliberately IRREGULAR (26, 44, 38, 22, 58, 34 s): under the
 * retired even grid every gap was 10 s, so a component that still assumes a fixed
 * step draws this flight visibly wrong instead of subtly wrong.
 * The duration word is the gap / 2 s, and the first event's is 0 (nothing before it).
 *
 * THE LAST EVENT CHANGES ONLY THE TERMINAL WORD. That is what a landing is in
 * this vocabulary — the geometric words in force are held to the threshold — and
 * it is also the one case where two rows differ in a single column.
 */
export const MOCK_EVENT_TIMES_S = [0, 26, 70, 108, 130, 188, 222];

export const MOCK_WORDS = [
  //  hdg  vert  spd  rwy  dur  term
  [18, LEVEL, 11, 0, 0, TERMINAL_CONTINUE], // +90° downwind, level, 121 m/s
  [18, LEVEL, 9, 0, 13, TERMINAL_CONTINUE], // slowing to 107
  [10, DESCEND_24, 9, 0, 22, TERMINAL_CONTINUE], // +50°, starting down at 2.4°
  [10, DESCEND_24, 7, 0, 19, TERMINAL_CONTINUE], // slowing to 93
  [4, DESCEND_31, 7, 0, 11, TERMINAL_CONTINUE], // +20°, steepening to 3.1°
  [0, DESCEND_31, 5, 0, 29, TERMINAL_CONTINUE], // on the course, slowing to 79
  [0, DESCEND_31, 5, 0, 17, TERMINAL_LANDED], // at the threshold
];

/**
 * The instructions behind the sentence above: one per word change, in issue order.
 * Only the three geometric kinds and the runway are issued — the duration and
 * terminal words are read off the EVENT SEQUENCE, never off this list.
 *
 * The VERTICAL ones tile the track (they are segments of one fitted profile, so
 * each ends where the next begins and the last ends at the record's end); the
 * plateau kinds do not, and one of them never settles at all.
 */
export const MOCK_INSTRUCTIONS = [
  { kind: "heading", word: 18, target: 88.4, issuedS: 0, settledS: 0, clamped: false },
  { kind: "vertical", word: LEVEL, target: 0.05, issuedS: 0, settledS: 70, clamped: false },
  { kind: "speed", word: 11, target: 120.4, issuedS: 0, settledS: 0, clamped: false },
  { kind: "runway", word: 0, target: 0, issuedS: 0, settledS: 0, clamped: false },
  { kind: "speed", word: 9, target: 106.2, issuedS: 26, settledS: 58, clamped: false },
  { kind: "heading", word: 10, target: 51.7, issuedS: 70, settledS: 94, clamped: false },
  { kind: "vertical", word: DESCEND_24, target: 2.31, issuedS: 70, settledS: 130, clamped: false },
  { kind: "speed", word: 7, target: 92.4, issuedS: 108, settledS: 126, clamped: false },
  { kind: "heading", word: 4, target: 21.3, issuedS: 130, settledS: 148, clamped: false },
  { kind: "vertical", word: DESCEND_31, target: 3.06, issuedS: 130, settledS: 262, clamped: false },
  { kind: "heading", word: 0, target: -1.2, issuedS: 188, settledS: 202, clamped: false },
  // The last deceleration never settles inside the track: it is still slowing at
  // the threshold, so `settledS` is null and that is a real answer.
  { kind: "speed", word: 5, target: 78.6, issuedS: 188, settledS: null, clamped: false },
];

/** Two manoeuvres the labeller read but did not word, one of each of two reasons. */
export const MOCK_ABSORBED = [
  { kind: "heading", startS: 40, endS: 48, word: 18, change: 4.2, reason: "short tail" },
  { kind: "speed", startS: 150, endS: 168, word: 7, change: -2.4, reason: "small change" },
];

/**
 * The observed track, on the artefact's own 2 s rows and in the course's frame.
 * It is GENERATED from a handful of anchors rather than typed out: 132 rows of
 * nine columns would bury the fixture, and every value here only has to be
 * plausible and consistent with the sentence above.
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

/**
 * Cumulative horizontal distance, the axis the vertical word is read on.
 * Each step is the ROW'S OWN speed times the gap before it, not a trapezoid,
 * which is how `instructions._profile` accumulates it. Its floor on the speed
 * (`MINIMUM_GROUND_SPEED_MPS`) is left out because this fixture never goes near
 * it — a fixture is the right SHAPE, and the real column comes from the
 * exporter, not from here.
 */
function cumulativePathM(tS: number[], speed: number[]): number[] {
  const out = [0];
  for (let i = 1; i < tS.length; i += 1) out.push(out[i - 1] + speed[i] * (tS[i] - tS[i - 1]));
  return out;
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

/**
 * The height anchors, built BACKWARDS from the vertical words: level at 929 m,
 * then 2.4° to 130 s, then 3.1° to the threshold over the ground the aircraft
 * covers at these speeds. So the profile's slopes are the words' angles, which
 * is what the read-back chart is there to check.
 */
const HEIGHT_ANCHORS: Array<[number, number]> = [[0, 929], [70, 929], [130, 665], [MOCK_TRACK_S, 0]];
const TO_GO_ANCHORS: Array<[number, number]> = [[0, 25000], [MOCK_TRACK_S, 0]];
const CROSS_ANCHORS: Array<[number, number]> =
  [[0, 6400], [94, 4200], [148, 1500], [202, 0], [MOCK_TRACK_S, 0]];

export const MOCK_OBSERVED = (() => {
  const tS: number[] = [];
  for (let t = 0; t <= MOCK_TRACK_S; t += ROW_STEP_S) tS.push(t);
  const unwrapped = tS.map((t) => ramp([[0, 90], [70, 90], [94, 50], [130, 50], [148, 20], [188, 20], [202, 0]], t));
  const groundSpeedMps = tS.map((t) => ramp([[0, 121], [58, 107], [126, 93], [214, 79], [MOCK_TRACK_S, 79]], t));
  return {
    tS,
    toGoM: tS.map((t) => ramp(TO_GO_ANCHORS, t)),
    crossM: tS.map((t) => ramp(CROSS_ANCHORS, t)),
    heightM: tS.map((t) => ramp(HEIGHT_ANCHORS, t)),
    pathM: cumulativePathM(tS, groundSpeedMps),
    relCourseDeg: unwrapped.map(wrap180),
    courseUnwrappedDeg: unwrapped,
    groundSpeedMps,
    established: tS.map((t) => (t >= 202 ? 1 : 0)),
    ...geodetic(tS, TO_GO_ANCHORS, CROSS_ANCHORS, HEIGHT_ANCHORS),
  };
})();

/**
 * The same sentence flown by `instruction_kinematics`, on its own 1 s steps. Like
 * the observed track it is GENERATED, and it is deliberately NOT the observed one:
 * it starts at the same point (the rules say so) and ends off the centreline,
 * which is what the real export does — a heading word says "fly this angle",
 * never "intercept the centreline".
 *
 * It carries BOTH bands (design §5.6). The vertical one is two height columns on
 * this track's own rows, because the commanded angle does not touch the
 * horizontal step; the speed one is two tracks of their own, because it does.
 */
const FLOWN_S = 300;
const FLOWN_TO_GO: Array<[number, number]> = [[0, 25000], [FLOWN_S, 400]];
const FLOWN_CROSS: Array<[number, number]> =
  [[0, 6400], [102, 4600], [156, 2600], [210, 1500], [FLOWN_S, 1480]];
const FLOWN_HEIGHT: Array<[number, number]> = [[0, 929], [76, 929], [140, 665], [FLOWN_S, 0]];

/** One edge of the speed band: the same sentence flown with every speed word at
 *  one end of its ±3 %, so it covers the same ground 3 % sooner or later. */
function speedEdge(scale: number) {
  const endS = Math.round(FLOWN_S / scale);
  const tS: number[] = [];
  for (let t = 0; t <= endS; t += 1) tS.push(t);
  return {
    tS,
    toGoM: tS.map((t) => ramp(FLOWN_TO_GO, t * scale)),
    crossM: tS.map((t) => ramp(FLOWN_CROSS, t * scale)),
    endReason: "crossed-threshold",
    endS,
  };
}

export const MOCK_GEOMETRIC = (() => {
  const tS: number[] = [];
  for (let t = 0; t <= FLOWN_S; t += 1) tS.push(t);
  const course = tS.map((t) => ramp([[0, 90], [70, 90], [102, 50], [130, 50], [156, 20], [188, 20], [210, 0]], t));
  const heightM = tS.map((t) => ramp(FLOWN_HEIGHT, t));
  const groundSpeedMps = tS.map((t) => ramp([[0, 121], [60, 107], [130, 93], [220, 79], [FLOWN_S, 79]], t));
  const path = cumulativePathM(tS, groundSpeedMps);
  // The fan opens with DISTANCE, and it closes on the height floor: both edges
  // level at 0 rather than flying through the runway, so the widest part of the
  // band is before the threshold, not at it (V34).
  const halfWidth = path.map((metres) => 0.0038 * metres);
  const fast = speedEdge(1.03);
  const slow = speedEdge(1 / 1.03);
  return {
    tS,
    toGoM: tS.map((t) => ramp(FLOWN_TO_GO, t)),
    crossM: tS.map((t) => ramp(FLOWN_CROSS, t)),
    heightM,
    groundSpeedMps,
    relCourseDeg: course,
    ...geodetic(tS, FLOWN_TO_GO, FLOWN_CROSS, FLOWN_HEIGHT),
    endReason: "crossed-threshold",
    finalGapM: 1480.5,
    meanGapM: 980.2,
    gapP95M: 1620.4,
    comparedS: 262,
    comparedFraction: 1,
    verticalBand: {
      heightLoM: heightM.map((metres, row) => metres + halfWidth[row]),
      heightHiM: heightM.map((metres, row) => Math.max(metres - halfWidth[row], 0)),
      altHaeLoM: heightM.map((metres, row) => THRESHOLD_HAE_M + metres + halfWidth[row]),
      altHaeHiM: heightM.map((metres, row) => THRESHOLD_HAE_M + Math.max(metres - halfWidth[row], 0)),
    },
    speedBand: {
      low: slow,
      high: fast,
      // The fast edge arrives first, so the window reads [fast, slow].
      arrivalWindowS: [fast.endS, slow.endS],
    },
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

/**
 * MIRROR of `instruction_kinematics.assumptions()` — the real defaults, value for
 * value (`ROUTE_BANK_RAD` 20°, `G` 9.81, `DESCENT_MAX_RAD` 6°, `CLIMB_MAX_RAD`
 * 4°, `ACCEL_MAX_MPS2` 1.0, and the three files those come from).
 *
 * THE CLIMB CAP IS 4°, NOT 2°: the limits have to cover every angle the
 * vocabulary can command, tolerance included, and the go-around mode's band
 * reaches 3.21°. `test_instruction_kinematics` pins that containment on the
 * Python side; a fixture written to the old 2° would make this reader's tests
 * pass against a sentence no executor could fly.
 *
 * The three band fields have NO producer yet (design §7, T10). They are here
 * because the schema is settled before the exporter that writes it runs — this
 * fixture is what the reader is tested against until then, and it is the one
 * place their shape is stated.
 */
export const MOCK_GEOMETRY = {
  method: "instruction-kinematics-v1",
  dtS: 1,
  bankDeg: 20,
  gravityMps2: 9.81,
  verticalIsCommandedAngle: true,
  heightFloorM: 0,
  descentMaxDeg: 6,
  // 4°, not 2°: the limits have to cover every angle the vocabulary can command,
  // tolerance included, and the go-around mode's band reaches 3.21°.
  climbMaxDeg: 4,
  accelMaxMps2: 1,
  startsAt: "observed-first-row",
  stopRule: "crossed-threshold or time-cap at the observed duration + 120 s",
  windModelled: false,
  aircraftTypeModelled: false,
  verticalBandFrom: "vocabulary.verticalTolerance",
  speedBandFrom: "vocabulary.speedTolerance",
  bandsAreJoint: false,
  constantsFrom: [
    "outputs/guidance/route.py", "outputs/guidance/controller.py", "geometry/flyability.py",
  ],
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
  writtenUtc: "2026-09-21T12:00:00Z",
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
