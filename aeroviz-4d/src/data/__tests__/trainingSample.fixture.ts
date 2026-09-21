/**
 * trainingSample.fixture.ts
 * -------------------------
 * MOCK data for the Training views. It lives in the test tree and is never
 * served: the only thing under `public/data/.../training/` is a real export
 * (design §1).
 *
 * The shape is the real one and the CONSTANTS ARE IMPORTED, never restated — a
 * fixture that types its own schema string or column order is a fixture that
 * cannot catch the contract moving (repo rule: a schema literal in a consumer is
 * a mirror; import it).
 *
 * THE FLIGHT IS BUILT FROM ITS WORDS, NOT BESIDE THEM. Under `box-v2-wedge` a
 * file is only valid if the track is inside the boxes its sentence makes, so a
 * fixture with hand-typed columns would be refused by its own reader. The three
 * signals here are therefore generated from the word sequence: at every event
 * boundary the value is the EDGE the two adjacent words share, and inside a span
 * it runs linearly between two values that are both in that span's box. That is
 * what a real crossing looks like, and it makes containment true by construction
 * rather than by luck.
 *
 * TWO THINGS ARE DELIBERATELY SIMPLER THAN THE ARTEFACT, and both are stated so
 * nothing is read off this file that only the real one can answer:
 *  • the descent is SHALLOW (about 1.3°, under the wedge's 1.5° opening), so one
 *    altitude word covers a long segment and the wedge is drawn three times
 *    rather than twenty-six. A real 3° approach cuts far shorter segments — that
 *    arithmetic is tested directly, against the real angles, rather than here.
 *  • the box FOOTPRINTS are a coarse sector — apex at the event's own position,
 *    three points on the arc. The exporter derives the real ones from the words'
 *    reachable set at one point per degree of opening; nothing in the reader
 *    recomputes them, so the fixture only has to be the right SHAPE (an outline
 *    whose first point is the apex, of the same length in all four arrays).
 *
 * The raw columns are the read ones plus a small wiggle, so a chart that plotted
 * the raw signal where it should plot the smoothed one is visibly wrong instead
 * of subtly wrong.
 */

import {
  TERMINAL_CONTINUE,
  TERMINAL_LANDED,
  TRAINING_INDEX_SCHEMA,
  TRAINING_KINDS,
  TRAINING_SAMPLE_SCHEMA,
  TRAINING_INSIDE_EPSILON,
  TRAINING_READING_RULE,
  altitudeWedgeM,
  eventInForce,
  headingBoxDeg,
  speedBoxMps,
  trainingWordCounts,
} from "../trainingSample";

/** KRDU's four vocabulary classes, with the airport prefix the runway word
 *  carries since the vocabulary went pooled. */
export const MOCK_RUNWAY_IDENTS = ["KRDU:05L", "KRDU:05R", "KRDU:23L", "KRDU:23R"];

/**
 * The spec this vocabulary is read under. The tables are COARSER than the
 * artefact's (11 heading boxes rather than 65, 6 speed boxes rather than 22, 8
 * altitude targets rather than 61) so a reader of the rows below can see which
 * box a value is in; every structural rule the parser checks — the heading edges
 * covering -180…180, the tables tiling, the wedge asymmetric — holds exactly as
 * it does in the real file.
 */
export const MOCK_SPEC = {
  redundancyFraction: 0.05,
  headingEdgesDeg: [-180, -90, -30, -10, -3, -1, 1, 3, 10, 30, 90, 180],
  headingFloorDeg: 1,
  speedEdgesMps: [60, 70, 80, 90, 100, 110, 125],
  altitudeTargetsM: [-20, 0, 30, 70, 110, 170, 250, 360],
  altitudeH0M: 50,
  altitudeDownDeg: 1.5,
  altitudeUpDeg: 1.0,
  durationBinS: 2,
  durationMaxS: 60,
  runwayIdents: MOCK_RUNWAY_IDENTS,
};

export const MOCK_VOCABULARY = {
  sha256: "8695be0c64e0000000000000000000000000000000000000000000000000mock",
  runwaySha256: "aa11bb22cc33000000000000000000000000000000000000000000000000mock",
  readingRule: TRAINING_READING_RULE,
  altitudeForm: "target + backward-reachable wedge, on remaining path length",
  altitudeReading: "greedy longest reach, read BACKWARDS (the target anchors the segment's end)",
  courseSmoothingS: 6,
  smoothingS: 10,
  ...MOCK_SPEC,
  // The artefact states its counts beside its spec and the reader refuses a file
  // where the two disagree — so the fixture DERIVES them rather than typing them,
  // which would be a third copy of the same numbers.
  words: trainingWordCounts(MOCK_SPEC),
};

export const MOCK_TRACK_S = 120;
const ROW_STEP_S = 2;

/**
 * The sentence. The boxes TILE the track: each hold is the duration word times
 * the bin, and the next event opens where the one before it closes — so the last
 * event plus its hold is exactly `durationS`, which the reader checks.
 *
 * The holds are deliberately IRREGULAR (10, 20, 16, 14, 24, 20, 16 s).
 */
export const MOCK_EVENT_TIMES_S = [0, 10, 30, 46, 60, 84, 104];
export const MOCK_HOLDS_S = [10, 20, 16, 14, 24, 20, 16];

/** The words this fixture uses, by name, so a reader of the rows below does not
 *  have to count table positions. Each kind's sequence steps between ADJACENT
 *  words, which is what lets the signals be continuous and still inside. */
const HEADING_WORDS = [9, 9, 8, 8, 7, 6, 5];
const ALTITUDE_WORDS = [5, 5, 4, 4, 1, 1, 1];
const SPEED_WORDS = [4, 4, 3, 3, 2, 2, 1];
const RUNWAY_WORD = 0;

export const MOCK_WORDS = MOCK_EVENT_TIMES_S.map((_, event) => [
  HEADING_WORDS[event],
  ALTITUDE_WORDS[event],
  SPEED_WORDS[event],
  RUNWAY_WORD,
  MOCK_HOLDS_S[event] / MOCK_SPEC.durationBinS,
  event === MOCK_EVENT_TIMES_S.length - 1 ? TERMINAL_LANDED : TERMINAL_CONTINUE,
]);

const TS: number[] = [];
for (let t = 0; t <= MOCK_TRACK_S; t += ROW_STEP_S) TS.push(t);

/** Linear between anchors, flat outside them. */
function ramp(anchors: Array<[number, number]>, t: number): number {
  const last = anchors.length - 1;
  if (t <= anchors[0][0]) return anchors[0][1];
  if (t >= anchors[last][0]) return anchors[last][1];
  const index = anchors.findIndex(([at]) => at > t);
  const [t0, v0] = anchors[index - 1];
  const [t1, v1] = anchors[index];
  return v0 + ((v1 - v0) * (t - t0)) / (t1 - t0);
}

/**
 * A signal that stays inside its own boxes: at every event boundary it sits on
 * the EDGE the two adjacent words share (or in the middle of the box when the
 * word does not change), and between boundaries it runs linearly between two
 * values that both lie in the box in force there.
 */
function signalFromWords(
  edges: number[],
  words: number[],
  eventTimesS: number[],
  endS: number,
): number[] {
  const centre = (word: number) => (edges[word] + edges[word + 1]) / 2;
  const anchors: Array<[number, number]> = eventTimesS.map((time, event) => {
    if (event === 0 || words[event] === words[event - 1]) return [time, centre(words[event])];
    return [time, edges[Math.max(words[event], words[event - 1])]];
  });
  anchors.push([endS, centre(words[words.length - 1])]);
  return TS.map((t) => ramp(anchors, t));
}

const READ_COURSE = signalFromWords(
  MOCK_SPEC.headingEdgesDeg, HEADING_WORDS, MOCK_EVENT_TIMES_S, MOCK_TRACK_S);
const READ_SPEED = signalFromWords(
  MOCK_SPEC.speedEdgesMps, SPEED_WORDS, MOCK_EVENT_TIMES_S, MOCK_TRACK_S);

/**
 * The path axis: the ground speed integrated. It is what the altitude wedge is
 * measured on — `r` is remaining PATH, never the projection on the course.
 */
const PATH_M = (() => {
  const out = [0];
  for (let i = 1; i < TS.length; i += 1) {
    out.push(out[i - 1] + READ_SPEED[i] * (TS[i] - TS[i - 1]));
  }
  return out;
})();

/**
 * The height, anchored on the ladder: each altitude segment ENDS on its own
 * target, and runs back up at about 1.3° — inside the wedge's 1.5° opening, so
 * the whole segment fits in one box.
 */
const READ_HEIGHT = TS.map((t) =>
  ramp(
    [
      [0, 214],
      [30, MOCK_SPEC.altitudeTargetsM[5]],
      [60, MOCK_SPEC.altitudeTargetsM[4]],
      [MOCK_TRACK_S, MOCK_SPEC.altitudeTargetsM[1]],
    ],
    t,
  ),
);

/** A repeatable wiggle, so the raw column is not the read one. */
function wiggle(row: number, amplitude: number): number {
  return amplitude * Math.sin(row * 1.7);
}

/**
 * The envelope one sentence makes over this track — the same derivation the
 * exporter does, so the fixture is a file the reader accepts rather than one it
 * refuses.
 *
 * The altitude segment's end is the NEXT ALTITUDE EVENT'S instant (the track's
 * last row for the final segment), and `r` is the path still to run to it.
 */
export function mockEnvelope(words: number[][]) {
  const forced = eventInForce(MOCK_EVENT_TIMES_S, TS);
  const altLoM = new Array<number>(TS.length).fill(0);
  const altHiM = new Array<number>(TS.length).fill(0);
  let row = 0;
  while (row < TS.length) {
    const word = words[forced[row]][1];
    let last = row;
    while (last + 1 < TS.length && words[forced[last + 1]][1] === word) last += 1;
    // the instant the word changes: the next row's time, or the end of the track
    const closesS = last + 1 < TS.length ? TS[last + 1] : TS[TS.length - 1];
    const endPath = last + 1 < TS.length ? PATH_M[last + 1] : PATH_M[TS.length - 1];
    void closesS;
    for (let index = row; index <= last; index += 1) {
      const [low, high] = altitudeWedgeM(MOCK_SPEC, word, endPath - PATH_M[index]);
      altLoM[index] = low;
      altHiM[index] = high;
    }
    row = last + 1;
  }
  const events = MOCK_EVENT_TIMES_S.map((eventS, event) => {
    const first = TS.findIndex((t) => t >= eventS);
    const closes = eventS + MOCK_HOLDS_S[event];
    const rows = TS.map((t, index) => (t >= eventS && t <= closes ? index : -1)).filter((i) => i >= 0);
    const [headingLoDeg, headingHiDeg] = headingBoxDeg(MOCK_SPEC, words[event][0]);
    const [speedLoMps, speedHiMps] = speedBoxMps(MOCK_SPEC, words[event][2]);
    const low = Math.min(...rows.map((index) => altLoM[index]));
    const high = Math.max(...rows.map((index) => altHiM[index]));
    // The footprint, the same SHAPE the exporter draws: apex at the event's own
    // position, then an arc at radius `hold × the speed box's upper edge` across
    // the heading box. The exporter puts one point per degree of opening; three
    // is its floor and all a fixture needs. A displacement at relative course ψ
    // over a distance d is `(-d·cos ψ, -d·sin ψ)` in this frame.
    const depth = speedHiMps * MOCK_HOLDS_S[event];
    const toGo = TO_GO_M[first];
    const cross = CROSS_M[first];
    const arc = [headingLoDeg, (headingLoDeg + headingHiDeg) / 2, headingHiDeg].map(
      (degrees) => (degrees * Math.PI) / 180,
    );
    return {
      eventS,
      holdS: MOCK_HOLDS_S[event],
      headingLoDeg,
      headingHiDeg,
      speedLoMps,
      speedHiMps,
      altitudeTargetM: MOCK_SPEC.altitudeTargetsM[words[event][1]],
      altLoM: low,
      altHiM: high,
      altHaeLoM: THRESHOLD_HAE_M + low,
      altHaeHiM: THRESHOLD_HAE_M + high,
      toGoM: [toGo, ...arc.map((angle) => toGo - depth * Math.cos(angle))],
      crossM: [cross, ...arc.map((angle) => cross - depth * Math.sin(angle))],
      // indicative geodetic corners: the reader only requires that the four
      // arrays are one outline, of the same length, with at least three points
      lon: [LON[first], ...arc.map((_, point) => LON[first] + 0.004 - 0.0002 * point)],
      lat: [LAT[first], ...arc.map((_, point) => LAT[first] + 0.003 + 0.0002 * point)],
    };
  });
  const inside = (values: number[], low: (row: number) => number, high: (row: number) => number) => {
    let outside = 0;
    values.forEach((value, index) => {
      if (value < low(index) - TRAINING_INSIDE_EPSILON || value > high(index) + TRAINING_INSIDE_EPSILON) {
        outside += 1;
      }
    });
    return { rows: values.length, outside };
  };
  const headingAt = (index: number) => headingBoxDeg(MOCK_SPEC, words[forced[index]][0]);
  const speedAt = (index: number) => speedBoxMps(MOCK_SPEC, words[forced[index]][2]);
  return {
    altLoM,
    altHiM,
    altHaeLoM: altLoM.map((metres) => THRESHOLD_HAE_M + metres),
    altHaeHiM: altHiM.map((metres) => THRESHOLD_HAE_M + metres),
    events,
    inside: {
      heading: inside(READ_COURSE, (i) => headingAt(i)[0], (i) => headingAt(i)[1]),
      altitude: inside(READ_HEIGHT, (i) => altLoM[i], (i) => altHiM[i]),
      speed: inside(READ_SPEED, (i) => speedAt(i)[0], (i) => speedAt(i)[1]),
    },
  };
}

/** KRDU's own numbers, so the mock lands where the real export does. */
const THRESHOLD_LON = -78.7875;
const THRESHOLD_LAT = 35.8776;
const THRESHOLD_HAE_M = 100.0;
const COURSE_RAD = (52 * Math.PI) / 180;

const TO_GO_M = TS.map((t) => ramp([[0, 9600], [MOCK_TRACK_S, 0]], t));
const CROSS_M = TS.map((t) => ramp([[0, 2400], [60, 900], [104, 60], [MOCK_TRACK_S, 0]], t));

/** A flat-earth inverse of `course_frame_rows` at KRDU's latitude — all a fixture
 *  needs to be the right SHAPE and roughly the right place. */
const LON: number[] = [];
const LAT: number[] = [];
TS.forEach((_, row) => {
  const east = -TO_GO_M[row] * Math.cos(COURSE_RAD) + CROSS_M[row] * Math.sin(COURSE_RAD);
  const north = -TO_GO_M[row] * Math.sin(COURSE_RAD) - CROSS_M[row] * Math.cos(COURSE_RAD);
  LAT.push(THRESHOLD_LAT + north / 111320);
  LON.push(THRESHOLD_LON + east / (111320 * Math.cos((THRESHOLD_LAT * Math.PI) / 180)));
});

export const MOCK_OBSERVED = {
  tS: TS,
  toGoM: TO_GO_M,
  crossM: CROSS_M,
  heightM: READ_HEIGHT.map((metres, row) => metres + wiggle(row, 3)),
  pathM: PATH_M,
  relCourseDeg: READ_COURSE.map((degrees, row) => degrees + wiggle(row, 1.5)),
  groundSpeedMps: READ_SPEED.map((mps, row) => mps + wiggle(row, 0.8)),
  established: TS.map((t) => (t >= 104 ? 1 : 0)),
  readCourseDeg: READ_COURSE,
  readSpeedMps: READ_SPEED,
  readHeightM: READ_HEIGHT,
  lon: LON,
  lat: LAT,
  altHaeM: READ_HEIGHT.map((metres) => THRESHOLD_HAE_M + metres),
  haeOffsetM: TS.map(() => THRESHOLD_HAE_M),
};

export const MOCK_FLIGHT = {
  flightKey: "DAL123_05L_a1b2c3_1699999999",
  callsign: "DAL123",
  runway: "KRDU:05L",
  stratum: "vectored",
  durationS: MOCK_TRACK_S,
  dtS: ROW_STEP_S,
  courseWindowRows: 4,
  signalWindowRows: 6,
  sentence: {
    eventTimesS: MOCK_EVENT_TIMES_S,
    holdS: MOCK_HOLDS_S,
    words: MOCK_WORDS,
  },
  observed: MOCK_OBSERVED,
  envelope: mockEnvelope(MOCK_WORDS),
};

/** MIRROR of `instruction_sample_export.reading_block()` — how a track became the
 *  signals the boxes judge. Its `insideEpsilon` is the reader's own constant,
 *  because the two verdicts are compared and a file written with another number
 *  would disagree on exactly the rows that sit on an edge. */
export const MOCK_READING = {
  rule: TRAINING_READING_RULE,
  courseSignal: "wrap(moving average of the wrapped relative ground track over courseSmoothingS)",
  speedSignal: "moving average of the ground speed over smoothingS",
  heightSignal: "moving average of the height above the threshold over smoothingS",
  pathSignal: "the smoothed ground speed integrated, floored at MINIMUM_GROUND_SPEED_MPS",
  remainingPathTo: "the next altitude event's instant; the track's last row for the final segment",
  windowRows: "round(seconds / median dt) + 1, centred, edges padded with the edge value",
  insideEpsilon: TRAINING_INSIDE_EPSILON,
  producedBy: "ts_transformer.experiments.instruction_sample_export (the artefact's own labeller is NOT in this repository)",
  constantsFrom: [
    "the artefact's spec block (redundancy, the wedge's angles, the ladder, the smoothing)",
    "ts_transformer/manoeuvre/instructions.py (course_frame, smooth, min_rows, wrap_deg)",
  ],
};

export const MOCK_SAMPLE = {
  schema: TRAINING_SAMPLE_SCHEMA,
  setId: "box_v3",
  airport: "KRDU",
  kinds: [...TRAINING_KINDS],
  vocabulary: MOCK_VOCABULARY,
  reading: MOCK_READING,
  flights: [MOCK_FLIGHT],
};

export const MOCK_INDEX = {
  schema: TRAINING_INDEX_SCHEMA,
  writtenUtc: "2026-09-21T12:00:00Z",
  airport: "KRDU",
  sets: [
    {
      id: "box_v3",
      kind: "vocabulary-readback",
      title: "Box vocabulary (a word is an interval)",
      file: "box_v3/sample.json",
      vocabularySha256: MOCK_VOCABULARY.sha256,
      runwaySha256: MOCK_VOCABULARY.runwaySha256,
      readingRule: MOCK_VOCABULARY.readingRule,
      flights: 1,
      cohort: {
        split: "val",
        perStratum: 20,
        seed: 1337,
        drawnFrom: "a seeded permutation of the val split at KRDU, stratified by approach_difficulty at the executor's anchor (pool 200)",
      },
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

/**
 * The same flight with WHAT THE MODEL SAID beside it — a `prior-generated` set.
 *
 * The said words are deliberately a MIXTURE: the opening event is the truth's
 * (it is given), two events match exactly, one differs in a single kind and two
 * in three — so a view that drew "differs" per row, per event or per kind can be
 * told apart. One of them names a different ALTITUDE target, so the model's own
 * wedge is a different shape and not one the track stays inside: that is the
 * answer the comparison exists to give, and a fixture whose model was always
 * right could not show it.
 */
export const MOCK_PRIOR_WORDS = [
  [9, 5, 4, RUNWAY_WORD, 5, TERMINAL_CONTINUE],   // given: the truth's opening event
  [9, 5, 4, RUNWAY_WORD, 3, TERMINAL_CONTINUE],   // the hold differs
  [8, 4, 3, RUNWAY_WORD, 8, TERMINAL_CONTINUE],   // exactly the truth
  [7, 5, 3, RUNWAY_WORD, 5, TERMINAL_CONTINUE],   // heading and ALTITUDE off, and the hold
  [7, 1, 2, RUNWAY_WORD, 12, TERMINAL_CONTINUE],  // exactly the truth
  [6, 1, 4, RUNWAY_WORD, 7, TERMINAL_LANDED],     // speed two boxes off, says landed early
  [5, 1, 1, RUNWAY_WORD, 8, TERMINAL_LANDED],     // the truth's last event, matched
];

const MOCK_PRIOR_CONFIDENCE = MOCK_PRIOR_WORDS.map((row, event) =>
  row.map((_, column) => (event === 0 ? 1 : column === 4 ? 0.06 : 0.84)),
);

export const MOCK_PRIOR = {
  sha256: "e7a9657afdc7000000000000000000000000000000000000000000000000mock",
  method: "teacher-forced-next-word",
  seed: 1337,
  bestEpoch: 35,
  trainedOnTheseFlights: 0,
  readout: {
    val: { heading: 0.4728, altitude: 0.6894, speed: 0.8554, runway: 0.0013, duration: 3.6703, terminal: 0.1105, next: 5.7996 },
  },
};

export function mockPriorSample(): Record<string, unknown> {
  const sample = structuredClone(MOCK_SAMPLE) as Record<string, unknown> & {
    setId: string;
    prior?: unknown;
    flights: Array<Record<string, unknown>>;
  };
  sample.setId = "prior_s1337_val";
  sample.prior = structuredClone(MOCK_PRIOR);
  sample.flights[0].prior = {
    words: structuredClone(MOCK_PRIOR_WORDS),
    confidence: structuredClone(MOCK_PRIOR_CONFIDENCE),
    givenEvents: 1,
    landedAtS: MOCK_EVENT_TIMES_S[5],
    envelope: mockEnvelope(MOCK_PRIOR_WORDS),
  };
  return sample;
}

/** The manifest entry that set would have: the kind is what makes the reader
 *  require the model's words. */
export function mockPriorIndex(): Record<string, unknown> {
  const index = structuredClone(MOCK_INDEX) as Record<string, unknown> & {
    sets: Array<Record<string, unknown>>;
  };
  index.sets[0] = {
    ...index.sets[0],
    id: "prior_s1337_val",
    kind: "prior-generated",
    file: "prior_s1337_val/sample.json",
    // the manifest names the model, so the picker can say which experiment a set
    // is without downloading it
    prior: { sha256: MOCK_PRIOR.sha256, seed: MOCK_PRIOR.seed, method: MOCK_PRIOR.method },
  };
  return index;
}
