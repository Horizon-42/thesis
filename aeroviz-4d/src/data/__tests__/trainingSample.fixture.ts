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
} from "../trainingSample";

/** KRDU's four vocabulary classes — the ARRIVAL manifest's coverage, not the
 *  airport's six thresholds (14 and 32 are excluded upstream). */
export const MOCK_RUNWAY_IDENTS = ["05L", "05R", "23L", "23R"];

export const MOCK_VOCABULARY = {
  sha256: "c7a4f4239f52000000000000000000000000000000000000000000000000mock",
  runwaySha256: "aa11bb22cc33000000000000000000000000000000000000000000000000mock",
  readingRule: "plateau-v11",
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

export const MOCK_FLIGHT = {
  flightKey: "DAL123_05L_a1b2c3_1699999999",
  callsign: "DAL123",
  runway: "05L",
  stratum: "vectored",
  sentence: {
    eventTimesS: MOCK_EVENT_TIMES_S,
    words: MOCK_WORDS,
    durationClamped: 0,
  },
};

export const MOCK_SAMPLE = {
  schema: TRAINING_SAMPLE_SCHEMA,
  setId: "vocabulary_tau10",
  airport: "KRDU",
  kinds: [...TRAINING_KINDS],
  vocabulary: MOCK_VOCABULARY,
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
