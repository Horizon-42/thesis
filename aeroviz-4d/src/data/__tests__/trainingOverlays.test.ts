/**
 * The Training overlays' reader: what it accepts over the fixture's set, and — the point of it — that an overlay that
 * does not bind to the set it names, or breaks its own bookkeeping, is refused whole and by name.
 */
import { describe, expect, it } from "vitest";

import { parseTrainingSample, TRAINING_UNCHANGED, type TrainingSample } from "../trainingSample";
import {
  executorWordAt,
  executorWordCounts,
  parseTrainingExecutorOverlay,
  parseTrainingOverlays,
  parseTrainingPriorOverlay,
  priorStep,
  trainingOverlaysOf,
  truthAt,
  TRAINING_EXECUTOR_SCHEMA,
  TRAINING_OVERLAYS_SCHEMA,
  TRAINING_PRIOR_SCHEMA,
} from "../trainingOverlays";
import { SET_ID, STRAIGHT_KEY, VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";
import { EXECUTOR_ID, PRIOR_ID, mockExecutorOverlay, mockOverlays, mockPriorOverlay } from "./trainingOverlays.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function executorRefusal(change: (raw: any) => void): string {
  const raw: any = mockExecutorOverlay();
  change(raw);
  const result = parseTrainingExecutorOverlay(raw, sample());
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

function priorRefusal(change: (raw: any) => void): string {
  const raw: any = mockPriorOverlay();
  change(raw);
  const result = parseTrainingPriorOverlay(raw, sample());
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

describe("parseTrainingOverlays", () => {
  it("lists the overlays and finds them by set and kind", () => {
    const parsed = parseTrainingOverlays(mockOverlays());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(trainingOverlaysOf(parsed.value, SET_ID, "executor-replay").map((entry) => entry.id)).toEqual([EXECUTOR_ID]);
    expect(trainingOverlaysOf(parsed.value, SET_ID, "prior-prediction").map((entry) => entry.id)).toEqual([PRIOR_ID]);
    expect(trainingOverlaysOf(parsed.value, "another_set", "executor-replay")).toEqual([]);
  });

  it("refuses another schema by name, and rejects a bad entry on its own", () => {
    const other: any = mockOverlays();
    other.schema = "aeroviz-training-overlays-v0";
    const refused = parseTrainingOverlays(other);
    expect(refused.ok).toBe(false);
    if (!refused.ok) expect(refused.problem).toContain(TRAINING_OVERLAYS_SCHEMA);
    const raw: any = mockOverlays();
    raw.overlays[0].kind = "free-generation";
    const parsed = parseTrainingOverlays(raw);
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.overlays.map((entry) => entry.id)).toEqual([PRIOR_ID]);
    expect(parsed.value.rejected[0]).toMatchObject({ id: EXECUTOR_ID });
    expect(parsed.value.rejected[0].problem).toMatch(/kind is "free-generation"/);
  });
});

describe("parseTrainingExecutorOverlay", () => {
  it("reads the flown flight word by word and the one not flown as such", () => {
    const parsed = parseTrainingExecutorOverlay(mockExecutorOverlay(), sample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const [vectored, straight] = parsed.value.flights;
    expect(vectored.flightKey).toBe(VECTORED_KEY);
    expect(executorWordCounts(vectored)).toEqual({ inside: 5, outside: 1, notJudged: 0, notReached: 0, superseded: 0 });
    expect(executorWordAt(vectored, 10, "approach")?.status).toBe("outside");
    expect(executorWordAt(vectored, 10, "heading")?.status).toBe("inside");
    expect(executorWordAt(vectored, 5, "heading")).toBeNull();          // no word is said there
    expect(straight).toMatchObject({ flightKey: STRAIGHT_KEY, flown: false, group: "no identified type", track: null, words: [] });
    expect(parsed.value.gate["own dynamics"].KXXX.all.clears).toEqual({ landed: true, words: false, evaluation: true });
    expect(parsed.value.gate["stand-in dynamics"].KXXX.all.notGated).toMatch(/stand-in/);
  });

  it("refuses another schema by name", () => {
    expect(executorRefusal((raw) => { raw.schema = "aeroviz-training-executor-v0"; })).toContain(TRAINING_EXECUTOR_SCHEMA);
  });

  it("refuses an overlay drawn over the set as it was before a re-export", () => {
    expect(executorRefusal((raw) => { raw.base.sampleWrittenUtc = "2026-09-01T00:00:00+00:00"; }))
      .toMatch(/the set was re-exported after the overlay/);
  });

  it("refuses flights out of the set's order", () => {
    expect(executorRefusal((raw) => raw.flights.reverse())).toMatch(/but the set's flight 0 is/);
  });

  it("refuses a verdict on another word than the sentence's", () => {
    expect(executorRefusal((raw) => { raw.flights[0].words[7].value = WORD.heading090; }))
      .toMatch(/but the sentence's word 7 is \(10, 2, 36\)/);
  });

  it("refuses a status it does not know, and a verdict its own checks contradict", () => {
    expect(executorRefusal((raw) => { raw.flights[0].words[2].status = "maybe"; })).toMatch(/status is maybe/);
    expect(executorRefusal((raw) => { raw.flights[0].words[6].status = "inside"; }))
      .toMatch(/is inside, but its checks say .*corridor held to the landing false/);
    expect(executorRefusal((raw) => { raw.flights[0].words[0].reason = null; })).toMatch(/is no check and says no reason/);
  });

  it("refuses a flight that is not flown yet carries a track", () => {
    expect(executorRefusal((raw) => { raw.flights[1].track = raw.flights[0].track; })).toMatch(/is not flown, yet carries a track/);
  });

  it("refuses a gate cell that is both gated and not", () => {
    expect(executorRefusal((raw) => { raw.gate["own dynamics"].KXXX.all.notGated = "why"; }))
      .toMatch(/gated \(clears\) or says why not/);
  });
});

describe("parseTrainingPriorOverlay", () => {
  it("reads each step's words and probabilities for every column", () => {
    const parsed = parseTrainingPriorOverlay(mockPriorOverlay(), sample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const flight = parsed.value.flights[0];
    const turn = priorStep(flight, "heading", 10);
    expect(turn.changeP).toBeCloseTo(0.6);
    expect(turn.ranked.map((item) => item.value)).toEqual([WORD.heading180 + 1, WORD.heading180, WORD.heading180 + 2]);
    // the runway column has two candidates, so two words are ranked
    expect(priorStep(flight, "runway", 0).ranked).toHaveLength(2);
    expect(parsed.value.readout.baselines.previousWord.all).toBeCloseTo(0.326);
  });

  it("reads the truth at a step off the sentence, unchanged where it says nothing", () => {
    const flight = sample().flights[0];
    expect(truthAt(flight, "heading", 10)).toBe(WORD.heading180);
    expect(truthAt(flight, "heading", 11)).toBe(TRAINING_UNCHANGED);
  });

  it("refuses another schema by name, and columns out of order", () => {
    expect(priorRefusal((raw) => { raw.schema = "aeroviz-training-prior-v0"; })).toContain(TRAINING_PRIOR_SCHEMA);
    expect(priorRefusal((raw) => raw.columns.reverse())).toMatch(/expected \[runway, approach, heading/);
  });

  it("refuses a flight of another length than the set's", () => {
    expect(priorRefusal((raw) => { raw.flights[0].rows = 59; })).toMatch(/rows is 59, not a whole number in 60…60/);
  });

  it("refuses more ranked words than the column has values, a value it does not have, and a probability above 1", () => {
    expect(priorRefusal((raw) => { raw.flights[0].columns[0].k = 3; })).toMatch(/k is 3, not a whole number in 1…2/);
    expect(priorRefusal((raw) => { raw.flights[0].columns[1].words[0] = 3; })).toMatch(/words\[0\] is 3, not one of the column's 3 values/);
    expect(priorRefusal((raw) => { raw.flights[0].columns[2].changeP[4] = 1.2; })).toMatch(/changeP\[4\] is 1.2, not a probability/);
  });

  it("refuses a readout missing a column's baseline", () => {
    expect(priorRefusal((raw) => { delete raw.readout.baselines.repeat.speed; })).toMatch(/repeat has no speed/);
  });
});
