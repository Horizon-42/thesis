/**
 * The stage-C reader (`trainingWindowSample.ts`) on the files the Python code writes (`stageC.ts`): the index and the
 * sample (v4, the window format of stages C and D: a list of commanded aircraft, stage C's one) are read; a file of another
 * schema is refused by name; the bookkeeping is checked (a loss's aircraft are the window's, an aircraft that ended at a
 * loss answers for one and one that answers ended there or is silent, window B and only B moves its start, the rounds are
 * the set's, the aircraft join in order); a round's sentence is read as a closed-loop sentence by stage A's views; window
 * B's observed track is its moved start; the traffic and the losses are on the window's clock, each aircraft has its own.
 */
import { describe, expect, it } from "vitest";
import {
  parseTrainingWindowIndex,
  parseTrainingWindowSample,
  lossesOf,
  onAircraftClock,
  onWindowClock,
  otherOf,
  roundLabel,
  startName,
  trainingWindowFlightView,
  trainingWindowOriginOf,
  TRAINING_WINDOW_INDEX_SCHEMA,
  TRAINING_WINDOW_SAMPLE_SCHEMA,
  trackAt,
  windowShiftText,
} from "../trainingWindowSample";
import { TRAINING_LOST_SEPARATION } from "../trainingSample";
import {
  stageCIndex, stageCSample, stageCSampleFile, stageCSampleFileFromRound, stageCSelection, WINDOW_SET_ID,
} from "./stageC";

describe("the index", () => {
  it("is read: one set with its model, cohort and source", () => {
    const parsed = parseTrainingWindowIndex(stageCIndex());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.rejected).toEqual([]);
    const [set] = parsed.value.sets;
    expect(set).toMatchObject({ id: WINDOW_SET_ID, file: `${WINDOW_SET_ID}/sample.json`, windows: 3, flights: 1 });
    expect(set.model.rounds).toEqual(["start"]);
    expect(set.model.rowIntervalS).toBe(4);
    expect(set.model.start).toBeNull();                                // the fixture's campaign starts from the base
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

  it("is read: the commanded flight's head and three windows, each commanding one aircraft at its row 0, with its round", () => {
    expect(sample.flights).toHaveLength(1);
    expect(sample.windows.map((window) => window.kind)).toEqual(["real", "A", "B"]);
    for (const window of sample.windows) {
      expect(window.c).toBeNull();
      const [aircraft, ...others] = window.commanded;
      expect(others).toEqual([]);
      expect(aircraft).toMatchObject({ place: 0, datasetId: sample.flights[0].datasetId, joinS: 0, shiftS: null });
      expect(aircraft.head).toBe(sample.flights[0]);
      expect(aircraft.rounds.map((sentence) => sentence.round)).toEqual(["start"]);
      expect(window.rounds.map((end) => end.round)).toEqual(["start"]);
      // its clock: the window's row 0 is its sentence's first row
      expect(aircraft.clockS).toBe(aircraft.head.closedLoop["4"].firstRow * sample.vocabulary.stepS);
      expect(aircraft.firstStepS).toBeGreaterThan(aircraft.clockS);
    }
  });

  it("reads the lost window's end: a loss with its two aircraft, the commanded one answering, on the window's clock", () => {
    const lost = sample.windows[1];
    const [aircraft] = lost.commanded;
    const [sentence] = aircraft.rounds;
    expect(sentence.outcome).toBe(TRAINING_LOST_SEPARATION);
    expect(sentence.crossing).toBeNull();
    expect(sentence.reward).toBe(0);
    expect(sentence.silentFromRow).toBeNull();
    const [loss] = lossesOf(lost.rounds[0], aircraft.datasetId);
    expect(loss.aircraft).toEqual([aircraft.datasetId, lost.traffic[0].key]);
    expect(loss.answering).toEqual([aircraft.datasetId]);
    expect(otherOf(loss, aircraft.datasetId)).toBe(lost.traffic[0].key);
    expect(loss.costsW).toBe(true);
    expect(lost.traffic[0]).toMatchObject({ role: "inserted", shiftS: -8 });
    // the loss is at the first step after the first predicted step, inside the flown track
    expect(onAircraftClock(aircraft, loss.timeS)).toBeCloseTo(aircraft.firstStepS + 4, 6);
    expect(onAircraftClock(aircraft, loss.timeS)).toBeLessThanOrEqual(sentence.flown.tS[sentence.flown.tS.length - 1]);
    // the traffic is on the window's clock: from its row 0
    expect(lost.traffic[0].tS[0]).toBe(0);
    expect(lossesOf(sample.windows[0].rounds[0], sample.windows[0].commanded[0].datasetId)).toEqual([]);
  });

  it("reads window B's moved start, and no other window has one", () => {
    const real = sample.windows[0].commanded[0];
    const moved = sample.windows[2].commanded[0];
    expect(real.movedStart).toBeNull();
    const start = moved.movedStart!;
    expect(start.tS[0]).toBe(moved.clockS);
    expect(start.tS[start.tS.length - 1]).toBeLessThan(moved.firstStepS);
    expect(moved.startMove.turnDeg).not.toBe(0);
    // the start moved: not where the recorded flight was, and drawn at the flight's runway's ellipsoid height
    const row = moved.head.closedLoop["4"].firstRow;
    expect(start.lon[0]).not.toBeCloseTo(moved.head.observed.lon[row], 4);
    expect(start.altitudeHaeM[0]).toBeCloseTo(start.altitudeMslM[0] + moved.head.haeMinusMslM, 6);
  });

  it("reads the campaign's start: the base, or another campaign's round (D162), named by one function", () => {
    expect(sample.model.start).toBeNull();
    expect(startName(sample.model.start)).toBe("base");
    expect(roundLabel("start", sample.model.start)).toBe("start (base)");
    const raw = stageCSampleFileFromRound();                           // post_train.start_of's setting (the fixture)
    const parsed = parseTrainingWindowSample(raw, "C");
    if (!parsed.ok) throw new Error(parsed.problem);
    const { start } = parsed.value.model;
    const written = raw.model.settings.start;
    expect(start).toEqual({ campaign: written.campaign, round: written.round, checkpointSha256: written.checkpoint_sha256 });
    expect(startName(start)).toBe(`${written.campaign.split("/").pop()} r${written.round}`);
    expect(startName(start)).toBe("post_source_fixture r0");
    expect(roundLabel("start", start)).toBe("start (post_source_fixture r0)");
    expect(roundLabel(5, start)).toBe("round 5");
  });

  it("refuses a start of another shape, and settings without one", () => {
    const refused = (change: (settings: Record<string, any>) => void, says: string) => {
      const raw = stageCSampleFileFromRound();
      change(raw.model.settings);
      const parsed = parseTrainingWindowSample(raw, "C");
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(says);
    };
    refused((settings) => { delete settings.start; }, "model.settings.start");
    refused((settings) => { settings.start = settings.start.campaign; }, "model.settings.start");
    refused((settings) => { settings.start.checkpoint_sha256 = "abc"; }, "not a SHA-256");
    refused((settings) => { settings.start.round = -1; }, "model.settings.start.round");
    refused((settings) => { delete settings.start.campaign; }, "model.settings.start.campaign");
  });

  it("refuses another schema by name", () => {
    const parsed = parseTrainingWindowSample({ ...stageCSampleFile(), schema: "aeroviz-training-prior-sample-v2" }, "C");
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain(TRAINING_WINDOW_SAMPLE_SCHEMA);
  });

  it("refuses what the bookkeeping does not hold", () => {
    const refused = (change: (raw: Record<string, any>) => void, says: string) => {
      const raw = stageCSampleFile();
      change(raw);
      const parsed = parseTrainingWindowSample(raw, "C");
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(says);
    };
    refused((raw) => { raw.windows[1].rounds[0].losses = []; }, "go together");
    refused((raw) => {                                                   // the real window: it landed, answering for a loss
      raw.windows[0].rounds[0].losses = raw.windows[1].rounds[0].losses;
      raw.windows[0].traffic = raw.windows[1].traffic;
    }, "go together");
    refused((raw) => { raw.windows[1].rounds[0].losses[0].aircraft[1] = "KXXX:nobody"; }, "none of the window's");
    refused((raw) => { raw.windows[1].rounds[0].losses[0].answering = [raw.windows[1].traffic[0].key]; }, "not a commanded aircraft");
    refused((raw) => { raw.windows[0].commanded = []; }, "commands no aircraft");
    refused((raw) => {                                                   // joining 4 s in: its first step 4 s later on the window's clock
      raw.windows[0].commanded[0].joinS = 4;
      raw.windows[0].commanded[0].firstStepS += 4;
    }, "the window's row 0 is its row 0");
    refused((raw) => { raw.windows[0].commanded.push(structuredClone(raw.windows[0].commanded[0])); }, "one flight twice");
    refused((raw) => { raw.windows[0].commanded[0].shiftS = 8; }, "not compressed");
    refused((raw) => { raw.windows[0].commanded[0].startMove.turnDeg = 5; }, "and only B, moves its start");
    refused((raw) => { raw.windows[2].commanded[0].movedStart = null; }, "moved start");
    refused((raw) => { raw.windows[0].commanded[0].rounds[0].round = 3; }, "rounds");
    refused((raw) => { raw.windows[0].rounds[0].round = 3; }, "the window's rounds");
    refused((raw) => { raw.windows[0].commanded[0].datasetId = "KXXX:nobody"; }, "no flight");
    refused((raw) => { raw.windows[1].traffic[0].shiftS = null; }, "shift");
    refused((raw) => { raw.windows[0].commanded[0].rounds[0].firstRow += 1; }, "first predicted step");
    refused((raw) => { raw.windows[0].commanded[0].rounds[0].silentFromRow = 10_000; }, "silentFromRow");
    refused((raw) => { raw.windows[0].commanded[0].rounds[0].silentFromRow = 1; }, "answers for no loss");
    refused((raw) => {                                                   // one aircraft answering two losses of a round
      const losses = raw.windows[1].rounds[0].losses;
      losses.push({ ...structuredClone(losses[0]), step: losses[0].step + 1 });
    }, "one at most");
    refused((raw) => { raw.windows[1].rounds[0].losses[0].aircraft.push("KXXX:a"); }, "not two aircraft");
    refused((raw) => { raw.windows[0].commanded[0].firstStepS += 4; }, "first predicted step at");
    refused((raw) => { raw.cohort.windows = 2; }, "cohort.windows");
  });
});

describe("the two clocks", () => {
  it("put a window time on an aircraft's flight clock and back (an aircraft whose window's row 0 is 300 s into its flight)", () => {
    const aircraft = { ...stageCSample().windows[0].commanded[0], clockS: 300 };
    expect(onAircraftClock(aircraft, 20)).toBe(320);
    expect(onWindowClock(aircraft, 320)).toBe(20);
    expect(onWindowClock(aircraft, onAircraftClock(aircraft, 7.5))).toBe(7.5);
  });
});

describe("a round's sentence as stage A's views read it", () => {
  const sample = stageCSample();

  it("is the closed-loop sentence at the set's Δ, with the window's end", () => {
    const selection = stageCSelection(sample, 1);
    const closed = selection.flight.closedLoop["4"];
    const sentence = sample.windows[1].commanded[0].rounds[0];
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
    const [first, second] = sample.windows;
    const view = trainingWindowFlightView(sample, first, first.commanded[0], "start");
    expect(trainingWindowFlightView(sample, first, first.commanded[0], "start")).toBe(view);
    expect(trainingWindowOriginOf(view)).toMatchObject({
      round: "start", window: first, aircraft: first.commanded[0], sentence: first.commanded[0].rounds[0], end: first.rounds[0],
    });
    expect(view.flightKey).not.toBe(trainingWindowFlightView(sample, second, second.commanded[0], "start").flightKey);
    expect(() => trainingWindowFlightView(sample, first, first.commanded[0], 4)).toThrow("no round 4");
  });
});

describe("where an aircraft is at a time", () => {
  const track = { tS: [0, 2, 4], lon: [0, 1, 2], lat: [10, 10, 12], altitudeHaeM: [100, 200, 300] };

  it("is on the straight line between the two rows around it, and nowhere outside the track", () => {
    expect(trackAt(track, 2)).toEqual([1, 10, 200]);
    expect(trackAt(track, 3)).toEqual([1.5, 11, 250]);
    expect(trackAt(track, 0)).toEqual([0, 10, 100]);
    expect(trackAt(track, -1)).toBeNull();
    expect(trackAt(track, 4.5)).toBeNull();
  });
});

describe("a window's shift in time (D129)", () => {
  it("reads in days from a day, in hours from an hour, else in seconds", () => {
    expect(windowShiftText(-4_409_144)).toBe("−51.0 d");
    expect(windowShiftText(7_200)).toBe("+2.0 h");
    expect(windowShiftText(-20)).toBe("−20 s");
    expect(windowShiftText(0)).toBe("+0 s");
  });
});
