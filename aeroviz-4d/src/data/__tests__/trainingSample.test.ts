/**
 * The stage-A reader (`trainingSample.ts`) on the files the Python code writes (`stageA.ts`): the index and the sample are
 * read; a file of another schema is refused by name with the schema found and the one expected; the bookkeeping is checked;
 * and the readings (the labelled sentence, the closed-loop sentence at Δ) put everything on the flight's clock.
 */
import { describe, expect, it } from "vitest";
import {
  nearestBranch,
  outsideSpans,
  parseTrainingIndex,
  parseTrainingSample,
  readingRowAt,
  readingRowTimeS,
  sentenceColumnRuns,
  sentenceWordAt,
  closedCycleTimeS,
  trainingBandLabel,
  trainingEnvelopeIndex,
  trainingReadingOf,
  trainingWordLabel,
  wordsOutside,
  unwrapDegrees,
  TRAINING_COLUMNS,
  TRAINING_INDEX_SCHEMA,
  TRAINING_READING_RULE,
  TRAINING_SAMPLE_SCHEMA,
  TRAINING_SET_KIND,
} from "../trainingSample";
import { trainingBandGround } from "../../scene/trainingEntities";
import { FLIGHT_KEY, stageAIndex, stageASample, stageASampleFile } from "./stageA";

describe("the index", () => {
  it("is read: one set, with its source and cohort", () => {
    const parsed = parseTrainingIndex(stageAIndex());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.rejected).toEqual([]);
    expect(parsed.value.sets.map((set) => set.id)).toEqual(["fixture_set"]);
    expect(parsed.value.sets[0].file).toBe("fixture_set/sample.json");
    expect(parsed.value.sets[0].cohort.strata).toEqual(["straight-in", "vectored"]);
  });

  it("refuses another schema by name: the one found and the one expected", () => {
    const old = { ...stageAIndex(), schema: "aeroviz-training-index-v1" };
    const parsed = parseTrainingIndex(old);
    expect(parsed).toEqual({
      ok: false, problem: `schema is "aeroviz-training-index-v1", expected "${TRAINING_INDEX_SCHEMA}"`,
    });
  });

  it("rejects an entry of another kind or reading rule on its own, naming the field", () => {
    const index = stageAIndex();
    index.sets.push({ ...index.sets[0], id: "old_kind", kind: "vocabulary-readback" });
    index.sets.push({ ...index.sets[0], id: "old_rule", readingRule: "instruction-v3" });
    const parsed = parseTrainingIndex(index);
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.sets.map((set) => set.id)).toEqual(["fixture_set"]);
    expect(parsed.value.rejected.map((item) => item.id)).toEqual(["old_kind", "old_rule"]);
    expect(parsed.value.rejected[0].problem).toContain(`not one of ${TRAINING_SET_KIND}`);
    expect(parsed.value.rejected[1].problem).toContain(`not one of ${TRAINING_READING_RULE}`);
  });
});

describe("the sample", () => {
  it("is read: five columns, the observed track, the labelled sentence and a closed-loop sentence per Δ", () => {
    const sample = stageASample();
    expect(TRAINING_COLUMNS).toEqual(["runway", "heading", "altitude", "angle", "speed"]);
    expect(sample.vocabulary.rowIntervalsS).toEqual([2, 4, 8]);
    const [flight] = sample.flights;
    expect(flight.flightKey).toBe(FLIGHT_KEY);
    expect(Object.keys(flight.closedLoop)).toEqual(["2", "4", "8"]);
    expect(flight.openLoop.events.every((event) => !event.correction)).toBe(true);
    expect(flight.observed.altitudeHaeM[0]).toBeCloseTo(flight.observed.altitudeMslM[0] + flight.haeMinusMslM, 6);
  });

  it("refuses an old sample schema by name", () => {
    const parsed = parseTrainingSample({ ...stageASampleFile(), schema: "aeroviz-training-sample-v8" });
    expect(parsed).toEqual({
      ok: false, problem: `sample.schema is "aeroviz-training-sample-v8", not one of ${TRAINING_SAMPLE_SCHEMA}`,
    });
  });

  it("refuses another reading rule", () => {
    const parsed = parseTrainingSample({ ...stageASampleFile(), readingRule: "instruction-v3" });
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain(`not one of ${TRAINING_READING_RULE}`);
  });

  it("refuses a vocabulary of six columns", () => {
    const file = stageASampleFile();
    file.vocabulary.columns = ["runway", "approach", "heading", "altitude", "angle", "speed"];
    const parsed = parseTrainingSample(file);
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("expected [runway, heading, altitude, angle, speed] in that order");
  });

  it("checks the bookkeeping: events are the grid's words; the closed loops are the set's Δ", () => {
    const events = stageASampleFile();
    events.flights[0].closedLoop["4"].events.pop();
    const a = parseTrainingSample(events);
    expect(a.ok).toBe(false);
    if (!a.ok) expect(a.problem).toContain("events are not the");

    const missing = stageASampleFile();
    delete missing.flights[0].closedLoop["8"];
    const b = parseTrainingSample(missing);
    expect(b.ok).toBe(false);
    if (!b.ok) expect(b.problem).toContain("closedLoop is listed at [2, 4] s, expected the set's [2, 4, 8]");

    const rows = stageASampleFile();
    rows.flights[0].closedLoop["2"].flownFromRow += 1;
    const c = parseTrainingSample(rows);
    expect(c.ok).toBe(false);
    if (!c.ok) expect(c.problem).toContain("flownFromRow");
  });

  it("refuses an unknown outcome and a rows grid that is not the vocabulary's 2 s from 0", () => {
    const outcome = stageASampleFile();
    outcome.flights[0].closedLoop["2"].replay.outcome = "went_around";
    const a = parseTrainingSample(outcome);
    expect(a.ok).toBe(false);

    const times = stageASampleFile();
    times.flights[0].observed.timeS[3] += 1;
    const b = parseTrainingSample(times);
    expect(b.ok).toBe(false);
    if (!b.ok) expect(b.problem).toContain("timeS[3]");
  });

  it("accepts a replay without envelopes and a crossing without a DA check", () => {
    const file = stageASampleFile();
    file.flights[0].closedLoop["2"].replay.envelopes = null;
    file.flights[0].closedLoop["4"].replay.crossing.decision = null;
    file.flights[0].closedLoop["8"].replay.crossing = null;
    const parsed = parseTrainingSample(file);
    if (!parsed.ok) throw new Error(parsed.problem);
    const [flight] = parsed.value.flights;
    expect(flight.closedLoop["2"].replay.envelopes).toBeNull();
    expect(flight.closedLoop["4"].replay.crossing!.decision).toBeNull();
    expect(flight.closedLoop["8"].replay.crossing).toBeNull();
  });
});

describe("the flown flight (replay.track)", () => {
  it("is the flight drawn: from the first predicted step to the judge's outcome, on the sample's executor cycle", () => {
    const sample = stageASample();
    expect(sample.executor.cycleS).toBe(1);
    const raw = stageASampleFile().flights[0].closedLoop;
    for (const key of ["2", "4", "8"]) {
      const closed = sample.flights[0].closedLoop[key];
      expect(closed.flown.tS).toHaveLength(raw[key].replay.track.rows);
      expect(closed.flown.eM).toEqual(raw[key].replay.track.eM);
      expect(closed.flown.attitude.headingDeg).toHaveLength(raw[key].replay.track.rows);
      expect(closed.cycleS).toBe(1);
      // the flown flight reaches the outcome: the stored states may end sooner (at the sentence's last said row)
      expect(closed.flown.tS[closed.flown.tS.length - 1]).toBe(closed.startS + (raw[key].replay.track.rows - 1) * 2);
      expect(closedCycleTimeS(closed, raw[key].replay.track.lastCycle) - closed.flown.tS[closed.flown.tS.length - 1]).toBeLessThan(2);
    }
  });

  it("turns a replay cycle into flight time by the executor's cycle", () => {
    const closed = stageASample().flights[0].closedLoop["2"];
    expect(closedCycleTimeS(closed, 10)).toBe(closed.startS + 10);
    expect(closedCycleTimeS({ ...closed, cycleS: 0.5 }, 10)).toBe(closed.startS + 5);
  });

  it("refuses an envelope that ends past the track, a track that is not the stored states, and a last cycle that is not the outcome's", () => {
    const past = stageASampleFile();
    past.flights[0].closedLoop["2"].replay.envelopes.speed[0].endRow = 300;
    const a = parseTrainingSample(past);
    expect(a.ok).toBe(false);
    if (!a.ok) expect(a.problem).toContain("an envelope's");

    const apart = stageASampleFile();
    apart.flights[0].closedLoop["4"].replay.track.eM[5] += 2;
    const b = parseTrainingSample(apart);
    expect(b.ok).toBe(false);
    if (!b.ok) expect(b.problem).toContain("replay.track and states are one flight");

    const last = stageASampleFile();
    last.flights[0].closedLoop["8"].replay.track.lastCycle -= 1;
    const c = parseTrainingSample(last);
    expect(c.ok).toBe(false);
    if (!c.ok) expect(c.problem).toContain("lastCycle is");

    const rows = stageASampleFile();
    rows.flights[0].closedLoop["2"].replay.track.rows -= 1;
    const r = parseTrainingSample(rows);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.problem).toMatch(/rows is \d+, but \d+ cycles/);

    const cycle = stageASampleFile();
    delete cycle.executor;
    expect(parseTrainingSample(cycle).ok).toBe(false);

    const uneven = stageASampleFile();
    uneven.executor.cycleS = 0.75;
    const u = parseTrainingSample(uneven);
    expect(u.ok).toBe(false);
    if (!u.ok) expect(u.problem).toContain("not a whole number of 0.75 s cycles");

    const open = stageASampleFile();
    const spans = open.flights[0].openLoop.envelopes.speed;
    spans[spans.length - 1].endRow = open.flights[0].observed.rows + 1;
    const o = parseTrainingSample(open);
    expect(o.ok).toBe(false);
    if (!o.ok) expect(o.problem).toContain("rows of the observed track");
  });

  it("accepts an empty heading band wherever it lies — its word's lead runs past the end of the flight — and draws nothing for it", () => {
    const file = stageASampleFile();
    const bands = file.flights[0].closedLoop["2"].replay.envelopes.heading;
    bands.push({ ...bands[0], row: 900, firstRow: 902, stopRow: 902, inside: [] });
    file.flights[0].openLoop.envelopes.heading.push({ ...file.flights[0].openLoop.envelopes.heading[0], row: 900, firstRow: 902, stopRow: 902, inside: [] });
    const parsed = parseTrainingSample(file);
    if (!parsed.ok) throw new Error(parsed.problem);
    const empty = parsed.value.flights[0].closedLoop["2"].replay.envelopes!.heading.slice(-1)[0];
    expect(empty.inside).toEqual([]);
    expect(trainingBandGround(parsed.value.flights[0].closedLoop["2"].flown, empty)).toEqual([]);
  });
});

describe("the readings", () => {
  const sample = stageASample();
  const [flight] = sample.flights;
  const stepS = sample.vocabulary.stepS;

  it("puts the labelled sentence on the observed rows from 0 and the closed-loop one on the first predicted step", () => {
    const open = trainingReadingOf(flight, stepS, null);
    expect(open.loop).toBe("open");
    expect(open.originS).toBe(0);
    expect(readingRowTimeS(open, 3)).toBe(6);
    expect(open.judged).toBe(flight.observed);
    const closed = trainingReadingOf(flight, stepS, 4);
    const raw = flight.closedLoop["4"];
    // state row k is observed row firstRow + k; the first predicted step is state row flownFromRow
    expect(closed.originS).toBe((raw.firstRow + raw.flownFromRow) * stepS);
    expect(readingRowTimeS(closed, 2)).toBe(closed.originS + 8);
    expect(closed.judged).toBe(raw.flown);
    expect(raw.flown.tS[0]).toBe(closed.originS);
    expect(raw.flown.tS[1] - raw.flown.tS[0]).toBe(stepS);
    expect(trainingReadingOf(flight, stepS, 4)).toBe(closed);
  });

  it("finds the sentence row at a time: none before the sentence opens, the last at and after its end", () => {
    const closed = trainingReadingOf(flight, stepS, 8);
    expect(readingRowAt(closed, closed.originS - 1)).toBeNull();
    expect(readingRowAt(closed, closed.originS)).toBe(0);
    expect(readingRowAt(closed, closed.originS + 15.9)).toBe(1);
    expect(readingRowAt(closed, closed.endS + 100)).toBe(closed.rows - 1);
  });

  it("runs a column's words to the next one's row and marks the corrections", () => {
    const closed = trainingReadingOf(flight, stepS, 2);
    const runs = sentenceColumnRuns(closed, "heading");
    expect(runs[0].row).toBe(0);
    expect(runs[runs.length - 1].endRow).toBe(closed.rows);
    runs.slice(1).forEach((run, i) => expect(runs[i].endRow).toBe(run.row));
    expect(runs.some((run) => run.event.correction)).toBe(true);
    expect(sentenceWordAt(closed, "heading", 52)!.event.value).toBe(35);
    expect(sentenceWordAt(closed, "heading", null)).toBeNull();
  });

  it("matches a word to its envelope: one to one in the labelled sentence, by the judge's row in a closed-loop one", () => {
    const open = trainingReadingOf(flight, stepS, null);
    sentenceColumnRuns(open, "heading").forEach((run, index) => expect(trainingEnvelopeIndex(open, stepS, "heading", run)).toBe(index));
    for (const interval of [2, 4, 8]) {
      const closed = trainingReadingOf(flight, stepS, interval);
      const bands = closed.envelopes!.heading;
      const found = sentenceColumnRuns(closed, "heading")
        .map((run) => trainingEnvelopeIndex(closed, stepS, "heading", run)).filter((index) => index !== null);
      expect(found).toEqual(bands.map((_, index) => index));
      // a column with no envelope gives none
      expect(trainingEnvelopeIndex(closed, stepS, "runway", sentenceColumnRuns(closed, "runway")[0])).toBeNull();
    }
  });

  it("spells a word from what it says, decoding nothing", () => {
    const closed = trainingReadingOf(flight, stepS, 2);
    const says = (column: (typeof TRAINING_COLUMNS)[number], row: number) => sentenceWordAt(closed, column, row)!.event.says;
    expect(trainingBandLabel(says("runway", 0))).toBe("09");
    expect(trainingWordLabel(says("runway", 0))).toBe("runway 09");
    expect(trainingBandLabel({ column: "heading", relativeDeg: -20, trackDeg: 205 })).toBe("−20°");
    expect(trainingBandLabel({ column: "heading", relativeDeg: 0, trackDeg: 225 })).toBe("0°");
    expect(trainingWordLabel({ column: "heading", relativeDeg: 15, trackDeg: 240.026 })).toBe("+15° from the course (track 240°)");
    expect(trainingBandLabel({ column: "altitude", noLevelOff: true })).toBe("no level-off");
    expect(trainingWordLabel({ column: "altitude", noLevelOff: false, levelM: 780, mslM: 912.6 })).toBe("780 m above the airport (913 m MSL)");
    expect(trainingBandLabel({ column: "angle", angleDeg: 0, climb: false, level: true })).toBe("level");
    expect(trainingBandLabel({ column: "angle", angleDeg: -1.5, climb: true, level: false })).toBe("climb 1.5°");
    expect(trainingBandLabel({ column: "angle", angleDeg: 3, climb: false, level: false })).toBe("3.0°");
    expect(trainingBandLabel({ column: "speed", speedMps: null })).toBe("unspecified");
    expect(trainingBandLabel({ column: "runway", goAround: true })).toBe("go-around");
  });
});

describe("words outside their envelopes", () => {
  it("counts a heading word with a row outside its band, a tube or span that did not hold, never an empty band", () => {
    const sample = stageASample();
    const envelopes = sample.flights[0].closedLoop["2"].replay.envelopes!;
    const expected = envelopes.heading.filter((band) => band.inside.includes(false)).length +
      envelopes.altitude.filter((tube) => !tube.contained).length + envelopes.speed.filter((span) => !span.contained).length;
    expect(wordsOutside(envelopes)).toBe(expected);
    const empty = { firstRow: 900, stopRow: 900, row: 898, targetDeg: 0, toleranceDeg: 4.5, inside: [] };
    const one = envelopes.altitude.length ? { ...envelopes.altitude[0], contained: false } : null;
    const more = { ...envelopes, heading: [...envelopes.heading, empty], altitude: one ? [one, ...envelopes.altitude.slice(1)] : [] };
    const before = envelopes.altitude.length && !envelopes.altitude[0].contained ? 0 : 1;
    expect(wordsOutside(more)).toBe(expected + (one ? before : 0));
  });
});

describe("drawing helpers", () => {
  it("carries a run of rows outside on to the next row, so one row is a segment", () => {
    expect(outsideSpans([true, false, false, true, false], 10, 20)).toEqual([[11, 13], [14, 15]]);
    expect(outsideSpans([true, true], 0, 5)).toEqual([]);
  });

  it("makes a track continuous and puts a target on the branch of the track", () => {
    expect(unwrapDegrees([350, 355, 2, 8])).toEqual([350, 355, 362, 368]);
    expect(unwrapDegrees([10, 5, 355], 370)).toEqual([370, 365, 355]);
    expect(nearestBranch(184, 544)).toBe(544);
    expect(nearestBranch(347, -13)).toBe(-13);
  });
});
