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
import {
  EXECUTOR_ID, MOCK_FIRST_PREDICTED_ROW, PRIOR_ID, mockExecutorOverlay, mockOverlays, mockPriorOverlay,
} from "./trainingOverlays.fixture";

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
    expect(executorWordCounts(vectored)).toEqual({ inside: 5, outside: 2, notJudged: 0, notReached: 0, superseded: 0 });
    expect(executorWordAt(vectored, 20, "approach")?.status).toBe("outside");
    expect(executorWordAt(vectored, 8, "heading")?.status).toBe("inside");
    expect(executorWordAt(vectored, 5, "heading")).toBeNull();          // no word is said there
    // a heading word carries its band on the flown rows, from where it was told plus the lead; no other word does
    const turned = executorWordAt(vectored, 10, "heading")!;
    expect(turned).toMatchObject({ status: "outside", heading: { firstRow: 12, stopRow: 20, targetOnTrackDeg: 180 } });
    expect(turned.heading!.inside.filter((ok) => !ok)).toHaveLength(1);
    expect(executorWordAt(vectored, 20, "approach")?.heading).toBeNull();
    expect(vectored.judgedTrackDeg).toHaveLength(49);
    expect(straight).toMatchObject({ flightKey: STRAIGHT_KEY, flown: false, group: "no identified type", track: null,
      judgedTrackDeg: null, words: [] });
    expect(parsed.value.gate["own dynamics"].KXXX.all.clears).toEqual({ landed: true, words: false, evaluation: true });
    expect(parsed.value.gate["stand-in dynamics"].KXXX.all.notGated).toMatch(/stand-in/);
  });

  it("refuses another schema by name — v1's turns and holds included", () => {
    for (const schema of ["aeroviz-training-executor-v0", "aeroviz-training-executor-v1"]) {
      expect(executorRefusal((raw) => { raw.schema = schema; })).toContain(TRAINING_EXECUTOR_SCHEMA);
    }
  });

  it("refuses a heading band that is not a lead after the flown step the word was told, or runs past the judged track", () => {
    expect(executorRefusal((raw) => { raw.flights[0].words[7].heading.firstRow = 11; raw.flights[0].words[7].heading.inside.push(1); }))
      .toMatch(/words\[7\]\.heading: firstRow is 11, but a word told at step 10 is judged from 12/);
    expect(executorRefusal((raw) => { raw.flights[0].judgedTrackDeg = raw.flights[0].judgedTrackDeg.slice(0, 15); }))
      .toMatch(/words\[7\]\.heading: stopRow is 20, not in 12…15/);
  });

  it("refuses a heading band its word's check does not count, or on a word that is not a heading word", () => {
    expect(executorRefusal((raw) => { raw.flights[0].words[7].heading.inside[0] = 1; }))
      .toMatch(/its band counts 8 of 8 rows inside, and no check says so/);
    expect(executorRefusal((raw) => { raw.flights[0].words[3].heading = raw.flights[0].words[2].heading; }))
      .toMatch(/words\[3\]: carries a heading band, but it is not a heading word the judge judged/);
    // a word with no row of its own is not judged — unless it was left to intercept the final on its own
    expect(executorRefusal((raw) => {
      Object.assign(raw.flights[0].words[6].heading, { stopRow: 10, inside: [] });
      raw.flights[0].words[6].checks = [{ name: "a check of something else", ok: true, inside: null, rows: null }];
    })).toMatch(/words\[6\]: is inside with no row judged/);
    const raw: any = mockExecutorOverlay();
    Object.assign(raw.flights[0].words[6], { status: "not judged", checks: [], reason: "no row to judge" });
    Object.assign(raw.flights[0].words[6].heading, { stopRow: 10, inside: [] });
    raw.flights[0].counts.wordsJudged = 6;
    raw.flights[0].counts.wordsInside = 4;
    const parsed = parseTrainingExecutorOverlay(raw, sample());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.flights[0].words[6]).toMatchObject({ status: "not judged", heading: { firstRow: 10, stopRow: 10, inside: [] } });
  });

  it("refuses a judged track given with the gate refusing it or the dynamics failing, or missing otherwise", () => {
    expect(executorRefusal((raw) => { raw.flights[0].judgedTrackDeg = null; }))
      .toMatch(/judgedTrackDeg is absent for a flight judged on its flown track/);
    expect(executorRefusal((raw) => { raw.flights[0].refused = "too short"; }))
      .toMatch(/judgedTrackDeg is given for a flight whose flown track the gate refused/);
    expect(executorRefusal((raw) => { raw.flights[0].outcome = "dynamics_failure"; }))
      .toMatch(/judgedTrackDeg is given for a flight whose dynamics failed/);
    // a dynamics failure keeps its words' statuses and checks, and draws no band
    const raw: any = mockExecutorOverlay();
    raw.flights[0].outcome = "dynamics_failure";
    raw.flights[0].judgedTrackDeg = null;
    for (const word of raw.flights[0].words) word.heading = null;
    const parsed = parseTrainingExecutorOverlay(raw, sample());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.flights[0].words.filter((word) => word.status === "outside")).toHaveLength(2);
  });

  it("refuses a judged track whose steps are not the flown track's points", () => {
    expect(executorRefusal((raw) => { raw.flights[0].track.tS[3] = 7; }))
      .toMatch(/judgedTrackDeg's step 3 is not the flown track's point 3 \(7 s\)/);
  });

  it("refuses a heading word the judge judged that carries no band", () => {
    expect(executorRefusal((raw) => { raw.flights[0].words[6].heading = null; }))
      .toMatch(/words\[6\]: is a heading word the judge judged on the flown track, and carries no band/);
  });

  it("refuses a judged count that is not one per verdict, the word left to intercept on its own once more", () => {
    expect(executorRefusal((raw) => { raw.flights[0].counts.wordsJudged = 8; raw.flights[0].counts.wordsInside = 5; }))
      .toMatch(/says 8 words judged, but 7 words carry a verdict/);
    // left to intercept the final on its own: that word's two checks, counted twice
    const raw: any = mockExecutorOverlay();
    const word = raw.flights[0].words[6];
    word.status = "outside";
    word.checks.push({ name: "held until the capture — left for 3 cycles to intercept the final on its own", ok: false, inside: null, rows: null });
    raw.flights[0].counts.wordsJudged = 8;
    raw.flights[0].counts.wordsInside = 4;
    const parsed = parseTrainingExecutorOverlay(raw, sample());
    if (!parsed.ok) throw new Error(parsed.problem);
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
    expect(executorRefusal((raw) => { raw.flights[0].words[8].status = "inside"; }))
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
    const turn = priorStep(flight, "heading", 10)!;
    expect(turn.changeP).toBeCloseTo(0.6);
    expect(turn.ranked.map((item) => item.value)).toEqual([WORD.heading180 + 1, WORD.heading180, WORD.heading180 + 2]);
    // the first predicted step says every column; the runway column has two candidates, so two words are ranked
    const opening = priorStep(flight, "runway", MOCK_FIRST_PREDICTED_ROW)!;
    expect(opening.changeP).toBe(1);
    expect(opening.ranked).toHaveLength(2);
    // before it the prior only observes
    expect(priorStep(flight, "heading", MOCK_FIRST_PREDICTED_ROW - 1)).toBeNull();
    expect(parsed.value.readout.baselines.previousWord.all).toBeCloseTo(0.326);
    expect(parsed.value.readout.model.perColumn.runway.top1GivenChange).toBeNull();
    expect(parsed.value.readout.firstStepRunway.rules.B1_active_config.top1).toBeCloseTo(0.74);
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

  it("refuses per-step arrays that do not cover the predicted steps exactly", () => {
    expect(priorRefusal((raw) => { raw.flights[0].firstPredictedRow = 3; })).toMatch(/words has 112 values, expected 114/);
    expect(priorRefusal((raw) => { raw.flights[0].firstPredictedRow = 60; })).toMatch(/firstPredictedRow is 60/);
  });
});
