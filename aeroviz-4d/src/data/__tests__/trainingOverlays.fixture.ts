/**
 * Overlays drawn over the Training fixture's set (`trainingSample.fixture.ts`), in the shapes the two exporters write
 * (`executor_training_export.py`, `prior_training_export.py`), built by hand. Every contract literal — the schemas,
 * the statuses, the columns — is IMPORTED from the reader, never restated.
 *
 * The executor flew the VECTORED flight (landed; its track reached 180° a row late, so that heading word has one row
 * outside its band; the clearance's corridor was left for five rows, so that word is outside too) and did not fly the
 * STRAIGHT-IN one (no identified type). The prior ranks, at every step, the truth's word
 * first where one is said — except the vectored flight's heading turn at step 10, where it ranks 185° above 180°.
 */

import { TRAINING_COLUMNS, TRAINING_SPEC_SHA256 } from "../trainingSample";
import {
  TRAINING_EXECUTOR_SCHEMA,
  TRAINING_OVERLAYS_SCHEMA,
  TRAINING_PRIOR_SCHEMA,
} from "../trainingOverlays";
import { MOCK_ROWS, SET_ID, STRAIGHT_KEY, VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";

export const EXECUTOR_ID = "executor_test";
export const PRIOR_ID = "prior_test";
export const MOCK_SAMPLE_SHA = "5".repeat(64);

/** What an overlay records of the fixture's set. */
export function mockBase(): Record<string, unknown> {
  const sample = mockSample() as { writtenUtc: string };
  return { setId: SET_ID, sampleWrittenUtc: sample.writtenUtc, sampleSha256: MOCK_SAMPLE_SHA, specSha256: TRAINING_SPEC_SHA256 };
}

export function mockOverlays(): Record<string, unknown> {
  const entry = (id: string, kind: string, file: string) => ({
    id, kind, base: SET_ID, baseSampleSha256: MOCK_SAMPLE_SHA, title: `the ${kind} test overlay`, file: `${id}/${file}`,
    flights: 2, source: { runner: "test" },
  });
  return {
    schema: TRAINING_OVERLAYS_SCHEMA,
    writtenUtc: "2026-09-24T00:00:00+00:00",
    airport: "KXXX",
    overlays: [entry(EXECUTOR_ID, "executor-replay", "executor.json"), entry(PRIOR_ID, "prior-prediction", "prior.json")],
  };
}

const check = (name: string, ok: boolean, inside: number | null = null, rows: number | null = null) => ({ name, ok, inside, rows });
const word = (row: number, column: number, value: number, status: string, checks: unknown[] = [], reason: string | null = null) =>
  ({ row, column, value, status, flownRow: row, heading: null, checks, reason });
/** A heading word the executor was told at its own step (the time clock): its band from two steps later (the 4 s lead)
 *  to `stop`, the rows' verdicts, and the judge's check of them. */
const headingWord = (row: number, value: number, targetDeg: number, stop: number, inside: number[]) => {
  const counted = inside.filter(Boolean).length;
  const ok = counted === inside.length;
  return {
    ...word(row, 2, value, ok ? "inside" : "outside",
      [check("track within ±4.5° of the word, 4 s after it was told, to the next word's", ok, counted, inside.length)]),
    heading: { firstRow: row + 2, stopRow: stop, targetOnTrackDeg: targetDeg, bandDeg: [targetDeg - 4.5, targetDeg + 4.5], inside },
  };
};

const cell = (flights: number, landed: number, gated: boolean) => ({
  flights, landed, wordsJudged: 6 * flights, wordsInside: 0.94, observedPasses: flights, evaluationPaired: 0.97,
  clears: gated ? { landed: landed >= 0.95, words: false, evaluation: true } : null,
  notGated: gated ? null : "not gated: a stand-in's dynamics", outcomes: { landed: flights },
});

export function mockExecutorOverlay(): Record<string, unknown> {
  const rows = 50;
  const range = (count: number, from = 0, step = 1) => Array.from({ length: count }, (_, i) => from + i * step);
  const eM = range(rows, -20000, 400);
  // it reaches 180° a row after the observed flight: at its step 12 it is still 15° short
  const flownTrack = range(rows).map((row) => (row < 10 ? 270 : row < 12 ? 225 : row === 12 ? 195 : row < 25 ? 180 : 90));
  return {
    schema: TRAINING_EXECUTOR_SCHEMA,
    overlayId: EXECUTOR_ID,
    airport: "KXXX",
    writtenUtc: "2026-09-24T00:00:00+00:00",
    producedBy: { runner: "test" },
    base: mockBase(),
    executor: { specSha256: "7".repeat(64), wordClock: "track", cycleS: 1, params: [{ name: "cycle_s", value: 1 }, { name: "word_clock", value: "track" }] },
    replay: { split: "val", writtenUtc: "2026-09-24T09:44:12+00:00", gateShare: 0.95, drawn: { flights: 3 }, git: { head: "test", dirty: false } },
    gate: {
      "own dynamics": {
        KXXX: { "straight-in": cell(1, 1, true), vectored: cell(1, 1, true), all: cell(2, 1, true) },
        all: { "straight-in": cell(2, 1, true), vectored: cell(2, 0.9, true), all: cell(4, 0.95, true) },
      },
      "stand-in dynamics": { KXXX: { all: cell(1, 1, false) }, all: { all: cell(3, 0.9, false) } },
    },
    flights: [
      {
        flightKey: VECTORED_KEY, datasetId: `KXXX:${VECTORED_KEY}`, group: "own dynamics", flown: true, outcome: "landed",
        flewTheSentence: false, endS: (rows - 1) * 2, crossing: { crossM: 1.5, heightM: 20.8, atS: (rows - 1) * 2 - 0.4 },
        refused: null, evaluation: { replay: "pass", observed: "pass" },
        alignment: { meanHorizontalDistanceM: 800, meanVerticalDistanceM: 30, landingTimeMinusObservedS: -4.2 },
        limits: { cycles: 98, bound: { bank_cap: 3, bank_rate: 10 } },
        counts: { wordsJudged: 7, wordsInside: 5, headingWordsNotJudged: 0 },
        track: {
          tS: range(rows, 0, 2), eM, nM: range(rows).map((row) => (row < 25 ? 2900 - row * 110 : 0)),
          lon: eM.map((e) => -78 + e / 90000), lat: range(rows).map(() => 35), altitudeM: range(rows).map((row) => 1110 - row * 10),
          altitudeHaeM: range(rows).map((row) => 1077 - row * 10), groundSpeedMps: range(rows).map(() => 100),
          trackDeg: flownTrack, distanceM: range(rows, 0, 200),
        },
        judgedTrackDeg: flownTrack.slice(0, rows - 1),
        words: [
          word(0, 0, WORD.runway09, "no check", [], "the runway pointer: judged by the landing, the flight's outcome"),
          word(0, 1, WORD.notCleared, "no check", [], "not cleared: no envelope of its own"),
          headingWord(0, WORD.heading270, 270, 10, range(8).map(() => 1)),
          word(0, 3, WORD.altitude1110, "inside", [check("in its tube", true, 20, 20)]),
          word(0, 4, WORD.level, "no check", [], "re-anchors its altitude word's tube: judged there"),
          word(0, 5, WORD.speed110, "inside", [check("transition monotone toward the target", true), check("band held", true, 30, 30)]),
          headingWord(8, WORD.heading225, 225, 12, [1, 1]),
          headingWord(10, WORD.heading180, 180, 20, [0, ...range(7).map(() => 1)]),
          word(20, 1, WORD.cleared, "outside", [check("capture turn monotone", true), check("corridor held to the landing", false, 30, 35)]),
          word(20, 3, WORD.land, "inside", [check("in its tube", true, 40, 40)]),
          word(20, 4, WORD.descent3, "no check", [], "re-anchors its altitude word's tube: judged there"),
          word(30, 5, WORD.unspecified, "no check", [], "the pilot's own speed: no band to hold"),
        ],
      },
      {
        flightKey: STRAIGHT_KEY, datasetId: `KXXX:${STRAIGHT_KEY}`, group: "no identified type", flown: false, outcome: null,
        flewTheSentence: null, endS: null, crossing: null, refused: null, evaluation: null, alignment: null, limits: null,
        counts: null, track: null, judgedTrackDeg: null, words: [],
      },
    ],
  };
}

/** The prior at one flight: a word said with 0.6 where the truth says one (1 at step 0), else 0.01; the truth's word
 *  ranked first — but for `wrong`, the steps whose second-ranked word is the truth. */
function priorFlight(key: string, events: Array<{ row: number; column: number; value: number }>, wrong: Set<string>) {
  const values = [2, 3, 72, 182, 6, 48];
  return {
    flightKey: key, datasetId: `KXXX:${key}`, rows: MOCK_ROWS, nllPerStep: 0.25, columnNllPerStep: [0.01, 0.02, 0.08, 0.05, 0.04, 0.05],
    columns: TRAINING_COLUMNS.map((_, column) => {
      const k = Math.min(3, values[column]);
      const changeP: number[] = [];
      const truthP: number[] = [];
      const words: number[] = [];
      const wordsP: number[] = [];
      for (let row = 0; row < MOCK_ROWS; row += 1) {
        const said = events.find((event) => event.row === row && event.column === column);
        const change = row === 0 ? 1 : said ? 0.6 : 0.01;
        changeP.push(change);
        const first = said ? said.value : 0;
        const ranked = wrong.has(`${row}:${column}`) ? [(first + 1) % values[column], first] : [first, (first + 1) % values[column]];
        if (k === 3) ranked.push((first + 2) % values[column]);
        words.push(...ranked);
        wordsP.push(...(k === 3 ? [0.5, 0.3, 0.1] : [0.7, 0.3]));
        truthP.push(said ? change * (wrong.has(`${row}:${column}`) ? 0.3 : 0.5) : 1 - change);
      }
      return { k, changeP, words, wordsP, truthP };
    }),
  };
}

export function mockPriorOverlay(): Record<string, unknown> {
  const sample = mockSample() as { flights: Array<{ flightKey: string; words: { events: Array<{ row: number; column: number; value: number }> } }> };
  const perColumn = Object.fromEntries(TRAINING_COLUMNS.map((column) => [column, {
    nllPerStep: 0.03, changeSteps: 100, changeProbabilityWhereChanged: 0.6, top1GivenChange: 0.7, top5GivenChange: 0.9,
    falseChangeShareWhereKept: 0.001,
  }]));
  const scores = (all: number) => ({ ...Object.fromEntries(TRAINING_COLUMNS.map((column) => [column, all / 6])), all });
  return {
    schema: TRAINING_PRIOR_SCHEMA,
    overlayId: PRIOR_ID,
    airport: "KXXX",
    writtenUtc: "2026-09-24T00:00:00+00:00",
    producedBy: { runner: "test" },
    base: mockBase(),
    prior: {
      checkpointSha256: "8".repeat(64), schema: "ts-prior-checkpoint-v1", specSha256: TRAINING_SPEC_SHA256, parameters: 2312581,
      model: { dModel: 192, layers: 4, heads: 6, feedforward: 768, dropout: 0.1, airports: ["KXXX"], classes: [3, 4, 73, 183, 7, 49] },
      train: { seed: 1337 }, method: "teacher-forced: each step sees the truth sentence's words before it",
    },
    readout: {
      split: "val", steps: 2053004, bestEpoch: 25,
      model: { nllPerStep: 0.1778, perplexityPerStep: 1.1946, perColumn },
      baselines: { repeat: scores(0.3444), previousWord: scores(0.326) },
    },
    columns: [...TRAINING_COLUMNS],
    flights: sample.flights.map((flight, index) =>
      priorFlight(flight.flightKey, flight.words.events, index === 0 ? new Set(["10:2"]) : new Set())),
  };
}
