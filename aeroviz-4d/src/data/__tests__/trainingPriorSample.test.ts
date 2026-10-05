/**
 * The stage-B reader (`trainingPriorSample.ts`) on the files the Python code writes (`stageB.ts`): the index and the sample
 * are read; a file of another schema is refused by name with the schema found and the one expected; the bookkeeping is
 * checked; and a sentence the prior said is read as a closed-loop sentence by stage A's views.
 */
import { describe, expect, it } from "vitest";
import {
  parseTrainingPriorIndex,
  parseTrainingPriorSample,
  runwayInForce,
  trainingPriorFlightView,
  trainingPriorOriginOf,
  TRAINING_PRIOR_BLOCKED_COLUMNS,
  trainingPriorSelectionOf,
  TRAINING_PRIOR_INDEX_SCHEMA,
  TRAINING_PRIOR_SAMPLE_SCHEMA,
  TRAINING_PRIOR_SET_KIND,
} from "../trainingPriorSample";
import { wordUnreached } from "../trainingSample";
import { readingRowAt, sentenceColumnRuns, trainingReadingOf, TRAINING_READING_RULE, TRAINING_SAMPLE_SCHEMA } from "../trainingSample";
import { allowedWords, blockedWordName } from "../../components/training/TrainingPriorSession";
import { PRIOR_SET_ID, stageBIndex, stageBSample, stageBSampleFile } from "./stageB";

describe("the index", () => {
  it("is read: one set with its model, cohort and source", () => {
    const parsed = parseTrainingPriorIndex(stageBIndex());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.rejected).toEqual([]);
    const [set] = parsed.value.sets;
    expect(set).toMatchObject({ id: PRIOR_SET_ID, file: "fixture_set/sample.json", flights: 1, sentences: 2 });
    expect(set.model.rowIntervalS).toBe(4);
    expect(set.cohort.samples).toBe(2);
    expect(set.source.smoke).toBe(true);
  });

  it("refuses another schema by name: the one found and the one expected", () => {
    const parsed = parseTrainingPriorIndex({ ...stageBIndex(), schema: "aeroviz-training-index-v2" });
    expect(parsed).toEqual({ ok: false, problem: `schema is "aeroviz-training-index-v2", expected "${TRAINING_PRIOR_INDEX_SCHEMA}"` });
  });

  it("rejects an entry of another kind or reading rule on its own, naming the field", () => {
    const index = stageBIndex();
    index.sets.push({ ...index.sets[0], id: "stage_a_kind", kind: "closed-loop-readback" });
    index.sets.push({ ...index.sets[0], id: "old_rule", readingRule: "instruction-v3" });
    const parsed = parseTrainingPriorIndex(index);
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.sets.map((set) => set.id)).toEqual([PRIOR_SET_ID]);
    expect(parsed.value.rejected.map((item) => item.id)).toEqual(["stage_a_kind", "old_rule"]);
    expect(parsed.value.rejected[0].problem).toContain(`not one of ${TRAINING_PRIOR_SET_KIND}`);
    expect(parsed.value.rejected[1].problem).toContain(`not one of ${TRAINING_READING_RULE}`);
  });
});

describe("the sample", () => {
  it("is read: the flight's head, the prior's sentences, the procedure's limits", () => {
    const sample = stageBSample();
    expect(sample.flights).toHaveLength(1);
    const [flight] = sample.flights;
    expect(flight.sentences.map((s) => s.sample)).toEqual([0, 1]);
    expect(Object.keys(flight.head.closedLoop)).toEqual(["4"]);
    expect(sample.procedure.map((p) => p.ident)).toEqual(sample.candidates.map((c) => c.ident));
    const [one] = flight.sentences;
    expect(one.words.length).toBe(one.goAroundProbability.length);
    expect(one.blocked.altitude).toHaveLength(one.words.length);
    expect(one.events.every((event) => !event.correction)).toBe(true);
    // the flown track is on the flight clock from the first predicted step, in the flight's own HAE
    const closed = flight.head.closedLoop["4"];
    expect(one.flown.tS[0]).toBe(closed.startS);
    expect(one.flown.altitudeHaeM[0]).toBeCloseTo(one.flown.altitudeMslM[0] + flight.head.haeMinusMslM, 9);
  });

  it("puts each candidate's limits in HAE with that runway's offset, once", () => {
    const sample = stageBSample();
    const raw = stageBSampleFile();
    sample.procedure.forEach((limit, index) => {
      const hae = sample.candidates[index].haeMinusMslM;
      expect(limit.decision.heightHaeM).toBeCloseTo(raw.procedure[index].decision.heightMslM + hae, 9);
      expect(limit.entry.heightHaeM).toBeCloseTo(raw.procedure[index].entry.heightMslM + hae, 9);
      expect(limit.glidepathLowerEdge.heightHaeM[3]).toBeCloseTo(raw.procedure[index].glidepathLowerEdge.heightMslM[3] + hae, 9);
      expect(limit.region.eM[0]).toBe(limit.region.eM[limit.region.eM.length - 1]);
    });
  });

  it("refuses another schema by name — stage A's sample too", () => {
    expect(parseTrainingPriorSample({ ...stageBSampleFile(), schema: TRAINING_SAMPLE_SCHEMA })).toEqual({
      ok: false, problem: `sample.schema is "${TRAINING_SAMPLE_SCHEMA}", not one of ${TRAINING_PRIOR_SAMPLE_SCHEMA}`,
    });
  });

  it("refuses what the bookkeeping does not hold", () => {
    const refuses = (change: (file: Record<string, any>) => void, text: string) => {
      const file = stageBSampleFile();
      change(file);
      const parsed = parseTrainingPriorSample(file);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(text);
    };
    refuses((f) => { f.flights[0].prior[1].startRow += 1; }, "start at one first predicted step");
    refuses((f) => { f.flights[0].prior[0].goAroundProbability.pop(); }, "goAroundProbability");
    refuses((f) => { f.flights[0].prior[0].blocked.angle[0] = [99]; }, "is not a list of angle words");
    refuses((f) => { delete f.flights[0].prior[0].blocked.angle; f.flights[0].prior[0].blocked.speed = []; }, "blocked is listed by");
    refuses((f) => { f.flights[0].prior[0].blocked.altitude.pop(); }, "expected one per word row");
    refuses((f) => { f.flights[0].prior[0].events.pop(); }, "events are not the");
    refuses((f) => { f.flights[0].prior[0].track.lastCycle += 1; }, "has its last state at");
    refuses((f) => { f.flights[0].prior[0].goAroundPermitted[0] = 2; }, "not 0 or 1");
    refuses((f) => { f.procedure[0].region.eM.pop(); }, "region.nM has");
    refuses((f) => { f.procedure.pop(); }, "procedure lists 0 runways");
    refuses((f) => { f.model.rowIntervalS = 3; }, "none of the vocabulary's");
    refuses((f) => { f.flights[0].prior = []; }, "prior holds no sentence");
    refuses((f) => { f.flights[0].closedLoop["8"] = f.flights[0].closedLoop["4"]; }, "closedLoop is listed at");
  });
});

describe("a sentence as stage A's views read it", () => {
  const sample = stageBSample();
  const [flight] = sample.flights;

  it("is the closed-loop sentence at the prior's Δ, and the flight's own for the closed-loop reading", () => {
    const view = trainingPriorFlightView(sample, flight, 1);
    expect(view.flightKey).toBe(`${flight.head.flightKey}~prior-1`);
    const reading = trainingReadingOf(view, sample.vocabulary.stepS, 4);
    expect(reading.loop).toBe("closed");
    expect(reading.closed!.words).toBe(flight.sentences[1].words);
    expect(reading.closed!.replay.outcome).toBe(flight.sentences[1].outcome);
    expect(reading.closed!.replay.envelopes).toBeNull();
    expect(sentenceColumnRuns(reading, "heading").length).toBeGreaterThan(1);
    expect(readingRowAt(reading, reading.originS)).toBe(0);
    const closed = trainingPriorFlightView(sample, flight, "closedLoop");
    expect(closed.flightKey).toBe(`${flight.head.flightKey}~closed-loop`);
    expect(closed.closedLoop["4"]).toBe(flight.head.closedLoop["4"]);
  });

  it("is built once, remembers where it came from, and is a flight of its own for the views (another key)", () => {
    expect(trainingPriorFlightView(sample, flight, 0)).toBe(trainingPriorFlightView(sample, flight, 0));
    const view = trainingPriorFlightView(sample, flight, 0);
    expect(trainingPriorFlightView(sample, flight, 1).flightKey).not.toBe(view.flightKey);
    expect(trainingPriorOriginOf(view)).toMatchObject({ flightKey: flight.head.flightKey, which: 0, sentence: flight.sentences[0] });
    expect(trainingPriorOriginOf(flight.head)).toBeUndefined();
    expect(() => trainingPriorFlightView(sample, flight, 7)).toThrow("no prior sentence 7");
  });

  it("is selected with the one Δ the prior speaks at", () => {
    const selection = trainingPriorSelectionOf(sample, trainingPriorFlightView(sample, flight, 0));
    expect(selection.vocabulary.rowIntervalsS).toEqual([4]);
    expect(selection.setId).toBe(PRIOR_SET_ID);
  });
});

describe("the blocked words", () => {
  const { vocabulary } = stageBSample();

  it("are named as the vocabulary names them and listed as what is still allowed", () => {
    expect(blockedWordName("altitude", 1, vocabulary)).toBe("60 m");
    expect(blockedWordName("altitude", vocabulary.noLevelOff, vocabulary)).toBe("no level-off");
    expect(blockedWordName("angle", 0, vocabulary)).toBe("level");
    const everything = Array.from({ length: vocabulary.noLevelOff + 1 }, (_, value) => value);
    expect(allowedWords("altitude", everything.slice(1), vocabulary)).toEqual([0]);
    expect(allowedWords("angle", [], vocabulary)).toHaveLength(vocabulary.angleClasses.length);
  });
});

describe("the words said after the outcome", () => {
  it("are counted as not reached, by their cycle against the flight's last state", () => {
    const sample = stageBSample();
    const flight = sample.flights[0];
    const sentence = flight.sentences[0];
    const view = trainingPriorFlightView(sample, flight, 0);
    const closed = view.closedLoop["4"];
    const after = sentence.events.filter((event) => (event.row * 4) / closed.cycleS > sentence.endCycle).length;
    expect(closed.replay.notReached).toBe(after);
    expect(closed.replay.flewTheSentence).toBe(after === 0);
    // an outcome at cycle 40 leaves every word of a later row unreached; a dynamics failure's last state is one before its row
    const early = { ...closed, replay: { ...closed.replay, endCycle: 40 } };
    expect(wordUnreached(early, 10)).toBe(false);
    expect(wordUnreached(early, 11)).toBe(true);
    expect(wordUnreached({ ...early, replay: { ...early.replay, outcome: "dynamics_failure" } }, 10)).toBe(true);
  });
});

describe("the runway in force", () => {
  const sample = stageBSample();
  const events = sample.flights[0].sentences[0].events;

  it("is the last runway word at or before the row, the first said before any", () => {
    const runway = events.filter((event) => event.column === 0 && event.says.column === "runway" && !event.says.goAround);
    expect(runwayInForce(events, 0)).toBe((runway[0].says as { runwayIndex: number }).runwayIndex);
    const synthetic = [
      { row: 0, column: 0, value: 1, correction: false, says: { column: "runway", goAround: false, runway: "B", runwayIndex: 1 } },
      { row: 3, column: 0, value: -2, correction: false, says: { column: "runway", goAround: true } },
      { row: 5, column: 0, value: 0, correction: false, says: { column: "runway", goAround: false, runway: "A", runwayIndex: 0 } },
    ] as typeof events;
    expect([0, 3, 4, 5, 9].map((row) => runwayInForce(synthetic, row))).toEqual([1, 1, 1, 0, 0]);
  });
});

describe("the columns the procedure masks rule", () => {
  it("are the names the export writes its blocked words under", () => {
    expect(Object.keys(stageBSampleFile().flights[0].prior[0].blocked)).toEqual([...TRAINING_PRIOR_BLOCKED_COLUMNS]);
  });
});
