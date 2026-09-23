/**
 * A small, internally consistent Training export for the tests — the shape the exporter
 * (`instruction_training_export.py`) writes, built by hand. Every contract literal (schema,
 * reading rule, spec sha, column order, "unchanged") is IMPORTED from the reader, never restated:
 * a fixture that restated them could not catch them moving.
 *
 * Two flights on candidate "09" of two: a VECTORED one (a heading turn at step 10 with the
 * clearance, a capture turn from step 20, captured at step 25, one altitude row outside its tube)
 * and a STRAIGHT-IN one (on the final from step 0). 60 steps of 2 s each.
 */

import {
  TRAINING_COLUMNS,
  TRAINING_INDEX_SCHEMA,
  TRAINING_READABLE_SET_KIND,
  TRAINING_READING_RULE,
  TRAINING_SAMPLE_SCHEMA,
  TRAINING_SPEC_SHA256,
  TRAINING_UNCHANGED,
} from "../trainingSample";

/** The set the exporter names after the reading rule. */
export const SET_ID = TRAINING_READING_RULE.replace("-", "_");

export const MOCK_ROWS = 60;
export const MOCK_STEP_S = 2;
export const MOCK_CANDIDATES_SHA = "c".repeat(64);
export const MOCK_LABELLER_SHA = "d".repeat(64);

const range = (count: number, from = 0, step = 1) => Array.from({ length: count }, (_, i) => from + i * step);
const line = (eM: number[], nM: number[]) => ({
  eM, nM, lon: eM.map((e) => -78 + e / 90000), lat: nM.map((n) => 35 + n / 111000),
});

export const MOCK_VOCABULARY = {
  readingRule: TRAINING_READING_RULE,
  specSha256: TRAINING_SPEC_SHA256,
  labellerSourceSha256: MOCK_LABELLER_SHA,
  columns: [...TRAINING_COLUMNS],
  unchanged: TRAINING_UNCHANGED,
  stepS: MOCK_STEP_S,
  smoothingS: { track: 6, altitude: 10, speed: 10 },
  classCounts: { approach: 3, heading: 72, altitude: 182, angle: 6, speed: 48 },
  approachClasses: ["not cleared", "cleared", "go-around"],
  headingTargetsDeg: range(72, 0, 5),
  headingToleranceDeg: 4.5,
  headingMaxTurnDeg: 150,
  turnRateMinDegS: 0.5,
  turnRateMaxDegS: 4.7,
  turnRateMinFromDeg: 10,
  turnBankMaxDeg: 32,
  turnStartDelayMaxS: 10.5,
  interceptAngleDeg: 30,
  corridorHalfWidthM: 20,
  corridorWideningDeg: 0.45,
  corridorCourseToleranceDeg: 2,
  landingCrossLimitM: 1000,
  landingMaxHeightM: 100,
  parallelCourseDeltaDeg: 5,
  altitudeTargetsM: range(181, 0, 30),
  altitudeLandValue: 181,
  altitudeToleranceM: 25,
  angleClasses: [
    { value: 0, name: "level", nominalDeg: 0, lowDeg: 0, steepDeg: 0 },
    { value: 1, name: "descent 1", nominalDeg: 0.91, lowDeg: -0.5, steepDeg: 1.52 },
    { value: 2, name: "descent 2", nominalDeg: 2.12, lowDeg: 1.52, steepDeg: 2.59 },
    { value: 3, name: "descent 3", nominalDeg: 3.06, lowDeg: 2.59, steepDeg: 3.73 },
    { value: 4, name: "descent 4", nominalDeg: 4.41, lowDeg: 3.73, steepDeg: 10 },
    { value: 5, name: "climb", nominalDeg: -1.22, lowDeg: -15, steepDeg: -0.5 },
  ],
  angleLevelValue: 0,
  speedTargetsMps: range(47, 20, 5),
  speedUnspecifiedValue: 47,
  speedToleranceMps: 5,
  speedAccelMaxMps2: 1.7,
  speedRangeMps: [15, 255],
};

export const MOCK_CANDIDATES = [
  { index: 0, ident: "09", thresholdEM: 0, thresholdNM: 0, courseDeg: 90, elevationM: 100, lengthM: 3000,
    landingCrossLimitM: 1000, centreline: line([0, -30000], [0, 0]), runway: line([0, 3000], [0, 0]) },
  { index: 1, ident: "27", thresholdEM: 3000, thresholdNM: 0, courseDeg: 270, elevationM: 101, lengthM: 3000,
    landingCrossLimitM: 450, centreline: line([3000, 33000], [0, 0]), runway: line([3000, 0], [0, 0]) },
];

/** Word values used below, named. */
export const WORD = {
  runway09: 0, notCleared: 0, cleared: 1,
  heading270: 54, heading180: 36, heading090: 18,
  altitude1110: 37, land: 181, level: 0, descent3: 3,
  speed110: 18, speed90: 14, unspecified: 47,
} as const;

type Event = { row: number; column: number; value: number; kind: string };

function inForce(events: Event[]): number[][] {
  return TRAINING_COLUMNS.map((_, column) => {
    let value = TRAINING_UNCHANGED;
    return range(MOCK_ROWS).map((row) => {
      const issued = events.find((event) => event.row === row && event.column === column);
      if (issued) value = issued.value;
      return value;
    });
  });
}

/** A left turn of 90° from a westbound track: the fastest turn, the slowest, and both moved 1 km
 *  west by the latest start — the end is their four corners, the region the ring through them. */
const turnRegion = (e0: number, n0: number) => ({
  fromTrackDeg: 270, turnDeg: -90, rateMinDegS: 0.5, rateMaxDegS: 4.7, bankMaxDeg: 32, startDelayMaxS: 10.5,
  slowFinished: true,
  region: line([e0, e0 - 1000, e0 - 1300, e0 - 11500, e0 - 12500, e0 - 2000, e0 - 1000],
               [n0, n0 - 300, n0 - 1300, n0 - 11500, n0 - 11500, n0 - 1000, n0]),
  fastPath: line([e0, e0 - 1000, e0 - 1300], [n0, n0 - 300, n0 - 1300]),
  slowPath: line([e0, e0 - 9000, e0 - 11500], [n0, n0 - 4000, n0 - 11500]),
  end: line([e0 - 1300, e0 - 11500, e0 - 12500, e0 - 2300], [n0 - 1300, n0 - 11500, n0 - 11500, n0 - 1300]),
});
const turnCheck = {
  progressOk: true, rateOk: true, meanRateDegS: 2.4, maxRateDegS: 3.1, maxBankDeg: 24.9, rateMinApplies: true,
};

/** One flight. `vectored`: a turn at step 10 and a capture at 25; else on the final from step 0. */
export function mockFlight(key: string, vectored: boolean): Record<string, unknown> {
  const rows = MOCK_ROWS;
  const captureRow = vectored ? 25 : 0;
  const joinRow = vectored ? 10 : 0;
  const events: Event[] = [
    { row: 0, column: 0, value: WORD.runway09, kind: "initial" },
    { row: 0, column: 1, value: vectored ? WORD.notCleared : WORD.cleared, kind: vectored ? "initial" : "clear" },
    { row: 0, column: 2, value: vectored ? WORD.heading270 : WORD.heading090, kind: "initial" },
    { row: 0, column: 3, value: WORD.altitude1110, kind: "initial" },
    { row: 0, column: 4, value: WORD.level, kind: "initial" },
    { row: 0, column: 5, value: WORD.speed110, kind: "initial" },
    ...(vectored ? [
      { row: 10, column: 1, value: WORD.cleared, kind: "clear" },
      { row: 10, column: 2, value: WORD.heading180, kind: "turn" },
    ] : []),
    { row: 20, column: 3, value: WORD.land, kind: "target" },
    { row: 20, column: 4, value: WORD.descent3, kind: "angle" },
    { row: 30, column: 5, value: WORD.unspecified, kind: "unspecified" },
  ];
  const eM = range(rows, -20000, 300);
  const nM = range(rows).map((row) => (vectored && row < 25 ? 3000 - row * 100 : 0));
  const altitude = range(rows).map((row) => (row < 20 ? 1110 : 1110 - (row - 20) * 16));
  const track = range(rows).map((row) => (vectored ? (row < 10 ? 270 : row < 16 ? 270 - (row - 10) * 15 : row < 25 ? 180 : 90) : 90));
  const speed = range(rows).map((row) => (row < 30 ? 110 : 110 - (row - 30) * 0.8));
  const distance = range(rows, 0, 220);
  const heading = vectored
    ? [
        {
          row: 0, value: WORD.heading270, kind: "initial", targetDeg: 270, split: null, turnEndRow: null,
          holdStartRow: 0, holdEndRow: 10, fromTrackDeg: 270, targetOnTrackDeg: 270, turnBandDeg: null,
          holdBandDeg: [265.5, 274.5], turn: null,
          funnel: { lengthM: 2200, startHalfWidthM: 0, endHalfWidthM: 173.1, axis: line([-20000, -22200], [3000, 3000]),
                    outline: line([-20000, -22200, -22200, -20000], [3000, 2826.9, 3173.1, 3000]) },
          check: null,
          holdCheck: { holdStartRow: 0, holdEndRow: 10, rows: 11, inside: 11, halfWidthEndM: 173.1 },
        },
        {
          row: 10, value: WORD.heading180, kind: "turn", targetDeg: 180, split: null, turnEndRow: 16,
          holdStartRow: 16, holdEndRow: 20, fromTrackDeg: 270, targetOnTrackDeg: 180, turnBandDeg: [175.5, 274.5],
          holdBandDeg: [175.5, 184.5], turn: turnRegion(-17000, 2000),
          funnel: { lengthM: 880, startHalfWidthM: 6200, endHalfWidthM: 6269.3, axis: line([-24800, -24800], [-5800, -6680]),
                    outline: line([-31000, -31069, -18531, -18600], [-5800, -6680, -6680, -5800]) },
          check: { ...turnCheck, kind: "turn", departureRow: 10, arrivalRow: 16, turnDeg: -90, parts: 1 },
          // one of its five hold rows outside the funnel
          holdCheck: { holdStartRow: 16, holdEndRow: 20, rows: 5, inside: 4, halfWidthEndM: 6269.3 },
        },
      ]
    : [
        {
          row: 0, value: WORD.heading090, kind: "initial", targetDeg: 90, split: null, turnEndRow: null,
          holdStartRow: null, holdEndRow: 0, fromTrackDeg: 90, targetOnTrackDeg: 90, turnBandDeg: null,
          holdBandDeg: null, turn: null, funnel: null, check: null, holdCheck: null,
        },
      ];
  const tubeInside = range(40).map((offset) => (vectored && offset === 5 ? 0 : 1));
  return {
    datasetId: `KXXX:${key}`, flightKey: key, callsign: key.split("_")[0], typecode: "A320",
    runway: "09", runwayIndex: 0, stratum: vectored ? "vectored" : "straight-in", rows,
    captureRow, joinRow, unspecifiedRow: 30, captureBeforeThresholdM: 12500,
    signals: {
      tS: range(rows, 0, MOCK_STEP_S), eM, nM, lon: eM.map((e) => -78 + e / 90000), lat: nM.map((n) => 35 + n / 111000),
      altitudeHaeM: altitude.map((h) => h - 33),
      raw: { trackDeg: track.map((t) => t + 0.4), altitudeM: altitude.map((h) => h + 3), groundSpeedMps: speed.map((v) => v + 0.5),
             verticalRateMps: range(rows).map((row) => (row < 20 ? 0 : -4)) },
      smoothed: { trackDeg: track, altitudeM: altitude, groundSpeedMps: speed, distanceM: distance },
      beforeThresholdM: eM.map((e) => -e), rightOfCourseM: nM.map((n) => -n),
    },
    words: { events, inForce: inForce(events) },
    envelopes: {
      heading,
      approach: {
        clearanceRow: joinRow, captureRow, captureBeforeThresholdM: 12500, interceptInserted: false,
        captureTurn: vectored
          ? { startRow: 20, courseOnTrackDeg: 90, bandDeg: [85.5, 184.5], check: turnCheck, turn: turnRegion(-14000, 1000) }
          : null,
        courseBandDeg: [88, 92],
        corridor: {
          beforeThresholdM: 12500, halfWidthAtCaptureM: 118.2, halfWidthAtThresholdM: 20, rows: rows - captureRow,
          axis: line([-12500, 0], [0, 0]), outline: line([-12500, 0, 0, -12500], [-118.2, -20, 20, 118.2]),
        },
        landing: { cutAtCrossing: false, lastRowBeforeThresholdM: 2300, crossing: null },
      },
      altitude: [
        {
          row: 0, endRow: 20, value: WORD.altitude1110, kind: "initial", targetM: 1110,
          lowerM: range(20).map(() => 1085), upperM: range(20).map(() => 1135),
          lowerHaeM: range(20).map(() => 1052), upperHaeM: range(20).map(() => 1102),
          inside: range(20).map(() => 1), check: { rows: 20, inside: 20, contained: true, tubeWidthEndM: 50 },
        },
        {
          row: 20, endRow: 60, value: WORD.land, kind: "target", targetM: null,
          lowerM: range(40).map((i) => 1085 - i * 20), upperM: range(40).map((i) => 1135 - i * 10),
          lowerHaeM: range(40).map((i) => 1052 - i * 20), upperHaeM: range(40).map((i) => 1102 - i * 10),
          inside: tubeInside,
          check: { rows: 40, inside: tubeInside.filter(Boolean).length, contained: !vectored, tubeWidthEndM: 440 },
        },
      ],
      angle: [
        { row: 0, value: WORD.level, kind: "initial", measuredDeg: null },
        { row: 20, value: WORD.descent3, kind: "angle", measuredDeg: 3.05 },
      ],
      speed: [
        {
          row: 0, endRow: 30, value: WORD.speed110, kind: "initial", targetMps: 110, arrivalRow: 0,
          transitionLowerMps: [110], transitionUpperMps: [110], bandMps: [105, 115], bandInside: range(30).map(() => 1),
          check: { arrivalRows: 0, cutBeforeArrival: false, transitionOk: true, accelOk: true, bandRows: 30, bandInside: 30, contained: true },
          rangeMps: null,
        },
        {
          row: 30, endRow: 60, value: WORD.unspecified, kind: "unspecified", targetMps: null, arrivalRow: null,
          transitionLowerMps: null, transitionUpperMps: null, bandMps: null, bandInside: null, check: null, rangeMps: [15, 255],
        },
      ],
    },
  };
}

export const VECTORED_KEY = "TST1_09_abc123_20260101T000000Z";
export const STRAIGHT_KEY = "TST2_09_abc124_20260101T000100Z";

/** A readable sample: the vectored flight first, then the straight-in one. */
export function mockSample(): Record<string, unknown> {
  return {
    schema: TRAINING_SAMPLE_SCHEMA,
    setId: SET_ID,
    airport: "KXXX",
    writtenUtc: "2026-09-23T00:00:00+00:00",
    producedBy: { runner: "test", artefact: "test", git: { head: "test", dirty: false } },
    cohort: { split: "val", perStratum: 1, seed: 1337, drawnFrom: "a test draw", pool: 2, read: 2 },
    vocabulary: structuredClone(MOCK_VOCABULARY),
    airportFrame: { code: "KXXX", lat: 35, lon: -78, elevationM: 100 },
    candidatesSha256: MOCK_CANDIDATES_SHA,
    centrelineLengthM: 30000,
    candidates: structuredClone(MOCK_CANDIDATES),
    flights: [mockFlight(VECTORED_KEY, true), mockFlight(STRAIGHT_KEY, false)],
  };
}

/** The readable entry, a superseded one and a prior one — as a manifest of today looks. */
export function mockIndex(): Record<string, unknown> {
  const cohort = { split: "val", perStratum: 1, seed: 1337, drawnFrom: "a test draw" };
  return {
    schema: TRAINING_INDEX_SCHEMA,
    airport: "KXXX",
    sets: [
      { id: "box_v3", kind: "vocabulary-readback", title: "an old set", file: "box_v3/sample.json",
        vocabularySha256: "a".repeat(64), runwaySha256: "b".repeat(64), readingRule: "box-v3", flights: 40,
        cohort: { ...cohort, perStratum: 20 } },
      { id: "instruction_v1", kind: "vocabulary-readback", title: "the instruction vocabulary as first frozen",
        file: "instruction_v1/sample.json", vocabularySha256: "0".repeat(64), runwaySha256: "b".repeat(64),
        readingRule: "instruction-v1", flights: 40, cohort: { ...cohort, perStratum: 20 } },
      { id: SET_ID, kind: TRAINING_READABLE_SET_KIND, title: "Instruction vocabulary", file: `${SET_ID}/sample.json`,
        vocabularySha256: TRAINING_SPEC_SHA256, runwaySha256: MOCK_CANDIDATES_SHA, readingRule: TRAINING_READING_RULE,
        flights: 2, cohort, source: { any: "extra keys are the exporter's provenance" } },
      { id: "prior_s1337_val", kind: "prior-generated", title: "an old prior", file: "prior_s1337_val/sample.json",
        vocabularySha256: "e".repeat(64), runwaySha256: "b".repeat(64), readingRule: "segment-v13", flights: 40,
        cohort: { ...cohort, perStratum: 20 }, prior: { sha256: "f".repeat(64), seed: 1337, method: "teacher-forced-next-word" } },
    ],
  };
}
