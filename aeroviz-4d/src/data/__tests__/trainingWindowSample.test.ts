/**
 * The stage-C reader (`trainingWindowSample.ts`) on the files the Python code writes (`stageC.ts`): the index and the
 * sample are read; a file of another schema is refused by name; the bookkeeping is checked (a loss goes with a loss's end,
 * window B and only B moves its start, the rounds are the set's); a round's sentence is read as a closed-loop sentence by
 * stage A's views; window B's observed track is its moved start; the traffic is on the commanded flight's clock.
 */
import { describe, expect, it } from "vitest";
import {
  parseTrainingWindowIndex,
  parseTrainingWindowSample,
  trainingWindowFlightView,
  trainingWindowOriginOf,
  TRAINING_WINDOW_INDEX_SCHEMA,
  TRAINING_WINDOW_SAMPLE_SCHEMA,
} from "../trainingWindowSample";
import { TRAINING_LOST_SEPARATION } from "../trainingSample";
import { positionAt } from "../../hooks/useTrainingWindowLayer";
import { stageCIndex, stageCSample, stageCSampleFile, stageCSelection, WINDOW_SET_ID } from "./stageC";

describe("the index", () => {
  it("is read: one set with its model, cohort and source", () => {
    const parsed = parseTrainingWindowIndex(stageCIndex());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.rejected).toEqual([]);
    const [set] = parsed.value.sets;
    expect(set).toMatchObject({ id: WINDOW_SET_ID, file: `${WINDOW_SET_ID}/sample.json`, windows: 3, flights: 1 });
    expect(set.model.rounds).toEqual(["start"]);
    expect(set.model.rowIntervalS).toBe(4);
    expect(set.source.smoke).toBe(true);
  });

  it("refuses another schema by name, and rejects an entry of another kind on its own", () => {
    const parsed = parseTrainingWindowIndex({ ...stageCIndex(), schema: "aeroviz-training-prior-index-v2" });
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain(TRAINING_WINDOW_INDEX_SCHEMA);
    const index = stageCIndex();
    index.sets[0].kind = "prior-free-generation";
    const rejected = parseTrainingWindowIndex(index);
    if (!rejected.ok) throw new Error(rejected.problem);
    expect(rejected.value.sets).toEqual([]);
    expect(rejected.value.rejected[0].problem).toContain("kind");
  });
});

describe("the sample", () => {
  const sample = stageCSample();

  it("is read: the commanded flight's head and three windows, each with its round", () => {
    expect(sample.flights).toHaveLength(1);
    expect(sample.windows.map((window) => window.kind)).toEqual(["real", "A", "B"]);
    for (const window of sample.windows) {
      expect(window.head).toBe(sample.flights[0]);
      expect(window.rounds.map((sentence) => sentence.round)).toEqual(["start"]);
      expect(window.firstStepS).toBeGreaterThan(window.row0S);
    }
  });

  it("reads the lost window's end: a loss of separation with its other aircraft, on the flight's clock", () => {
    const lost = sample.windows[1];
    const [sentence] = lost.rounds;
    expect(sentence.outcome).toBe(TRAINING_LOST_SEPARATION);
    expect(sentence.crossing).toBeNull();
    expect(sentence.end.reward).toBe(0);
    const loss = sentence.end.loss!;
    expect(loss.other).toBe(lost.traffic[0].key);
    expect(lost.traffic[0]).toMatchObject({ role: "inserted", shiftS: -8 });
    // the loss is at the first step after the first predicted step, inside the flown track
    expect(loss.timeS).toBeCloseTo(lost.firstStepS + 4, 6);
    expect(loss.timeS).toBeLessThanOrEqual(sentence.flown.tS[sentence.flown.tS.length - 1]);
    // the traffic is placed on the commanded flight's clock: from the window's row 0
    expect(lost.traffic[0].tS[0]).toBe(lost.row0S);
  });

  it("reads window B's moved start, and no other window has one", () => {
    const [real, , moved] = sample.windows;
    expect(real.movedStart).toBeNull();
    const start = moved.movedStart!;
    expect(start.tS[0]).toBe(moved.row0S);
    expect(start.tS[start.tS.length - 1]).toBeLessThan(moved.firstStepS);
    expect(moved.startMove.turnDeg).not.toBe(0);
    // the start moved: not where the recorded flight was, and drawn at the flight's runway's ellipsoid height
    const row = moved.head.closedLoop["4"].firstRow;
    expect(start.lon[0]).not.toBeCloseTo(moved.head.observed.lon[row], 4);
    expect(start.altitudeHaeM[0]).toBeCloseTo(start.altitudeMslM[0] + moved.head.haeMinusMslM, 6);
  });

  it("refuses another schema by name", () => {
    const parsed = parseTrainingWindowSample({ ...stageCSampleFile(), schema: "aeroviz-training-prior-sample-v2" });
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain(TRAINING_WINDOW_SAMPLE_SCHEMA);
  });

  it("refuses what the bookkeeping does not hold", () => {
    const refused = (change: (raw: Record<string, any>) => void, says: string) => {
      const raw = stageCSampleFile();
      change(raw);
      const parsed = parseTrainingWindowSample(raw);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(says);
    };
    refused((raw) => { raw.windows[1].rounds[0].end.loss = null; }, "go together");
    refused((raw) => { raw.windows[0].rounds[0].end.loss = raw.windows[1].rounds[0].end.loss; }, "go together");
    refused((raw) => { raw.windows[0].startMove.turnDeg = 5; }, "and only B, moves its start");
    refused((raw) => { raw.windows[2].movedStart = null; }, "moved start");
    refused((raw) => { raw.windows[0].rounds[0].round = 3; }, "rounds");
    refused((raw) => { raw.windows[0].datasetId = "KXXX:nobody"; }, "no flight");
    refused((raw) => { raw.windows[1].traffic[0].shiftS = null; }, "shift");
    refused((raw) => { raw.windows[0].rounds[0].firstRow += 1; }, "first predicted step");
    refused((raw) => { raw.cohort.windows = 2; }, "cohort.windows");
  });
});

describe("a round's sentence as stage A's views read it", () => {
  const sample = stageCSample();

  it("is the closed-loop sentence at the set's Δ, with the window's end", () => {
    const selection = stageCSelection(sample, 1);
    const closed = selection.flight.closedLoop["4"];
    const sentence = sample.windows[1].rounds[0];
    expect(closed.words).toBe(sentence.words);
    expect(closed.replay.outcome).toBe(TRAINING_LOST_SEPARATION);
    expect(closed.flown).toBe(sentence.flown);
    expect(selection.vocabulary.rowIntervalsS).toEqual([4]);
  });

  it("keeps the head's observed track and open-loop reading, window B's too (its moved start is a line of its own)", () => {
    for (const place of [0, 2]) {
      const flight = stageCSelection(sample, place).flight;
      expect(flight.observed).toBe(sample.flights[0].observed);
      expect(flight.openLoop).toBe(sample.flights[0].openLoop);
    }
  });

  it("is built once, remembers where it came from, and is a flight of its own for the views (another key)", () => {
    const view = trainingWindowFlightView(sample, sample.windows[0], "start");
    expect(trainingWindowFlightView(sample, sample.windows[0], "start")).toBe(view);
    expect(trainingWindowOriginOf(view)).toMatchObject({ round: "start", window: sample.windows[0] });
    expect(view.flightKey).not.toBe(trainingWindowFlightView(sample, sample.windows[1], "start").flightKey);
    expect(() => trainingWindowFlightView(sample, sample.windows[0], 4)).toThrow("no round 4");
  });
});

describe("where an aircraft is at a time", () => {
  const track = { tS: [0, 2, 4], lon: [0, 1, 2], lat: [10, 10, 12], altitudeHaeM: [100, 200, 300] };

  it("is on the straight line between the two rows around it, and nowhere outside the track", () => {
    expect(positionAt(track, 2)).toEqual([1, 10, 200]);
    expect(positionAt(track, 3)).toEqual([1.5, 11, 250]);
    expect(positionAt(track, 0)).toEqual([0, 10, 100]);
    expect(positionAt(track, -1)).toBeNull();
    expect(positionAt(track, 4.5)).toBeNull();
  });
});
