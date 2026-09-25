/**
 * The Training reader under the instruction vocabulary: what it accepts, and — the point of it — that
 * everything it refuses is refused BY NAME, with the field that failed.
 */
import { describe, expect, it } from "vitest";

import {
  parseTrainingIndex,
  parseTrainingSample,
  rowAtTime,
  outsideSpans,
  trainingColumnRuns,
  trainingWordAt,
  trainingSetRefusal,
  trainingVerdicts,
  trainingWordLabel,
  TRAINING_COLUMNS,
  TRAINING_INDEX_SCHEMA,
  TRAINING_READING_RULE,
  TRAINING_SAMPLE_SCHEMA,
  TRAINING_SPEC_SHA256,
  type TrainingSample,
} from "../trainingSample";
import { MOCK_ROWS, SET_ID, STRAIGHT_KEY, VECTORED_KEY, WORD, mockIndex, mockSample } from "./trainingSample.fixture";

function parsed(raw: unknown = mockSample()): TrainingSample {
  const result = parseTrainingSample(raw);
  if (!result.ok) throw new Error(result.problem);
  return result.value;
}

/** The sample with one change applied to its raw JSON, and the reader's answer. */
function refusal(change: (raw: any) => void): string {
  const raw: any = mockSample();
  change(raw);
  const result = parseTrainingSample(raw);
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

describe("parseTrainingSample", () => {
  it("reads both flights, the candidates and the vocabulary's tables", () => {
    const sample = parsed();
    expect(sample.flights.map((flight) => flight.flightKey)).toEqual([VECTORED_KEY, STRAIGHT_KEY]);
    expect(sample.candidates.map((candidate) => candidate.ident)).toEqual(["09", "27"]);
    expect(sample.vocabulary.specSha256).toBe(TRAINING_SPEC_SHA256);
    const flight = sample.flights[0];
    expect(flight.rows).toBe(MOCK_ROWS);
    expect(flight.words.inForce).toHaveLength(TRAINING_COLUMNS.length);
    expect(flight.envelopes.altitude[1].inside.filter((ok) => !ok)).toHaveLength(1);
    expect(flight.envelopes.approach.captureTurn).toMatchObject({ startRow: 20, endRow: 25 });
    // the lead in steps, from the vocabulary's seconds
    expect(sample.vocabulary.headingLeadRows).toBe(2);
    // each heading word's band: judged from two steps after it to the next word's, the last to the clearance
    expect(flight.envelopes.heading.map((item) => [item.row, item.firstRow, item.stopRow, item.check.inside, item.check.rows]))
      .toEqual([[0, 2, 10, 8, 8], [8, 10, 12, 2, 2], [10, 12, 20, 7, 8]]);
    // a flight on the final at step 0: its heading word has no row of its own, and there is no capture turn
    expect(sample.flights[1].envelopes.heading[0]).toMatchObject({ firstRow: 2, stopRow: 2, inside: [], check: { rows: 0, inside: 0 } });
    expect(sample.flights[1].envelopes.approach.captureTurn).toBeNull();
  });

  // ── refused by name ─────────────────────────────────────────────────────
  it("refuses a sample of an earlier format by its schema name, whatever vocabulary it carries", () => {
    // v2: the box vocabulary's sample; v3: instruction-v1's, and instruction-v2's before its change of
    // shape got a name of its own; v4: instruction-v2's before the heading wedge; v5: with the flown type
    // (an A320 for every type without dynamics); v6: instruction-v2's turns and holds. All are superseded
    // names, so they are written out here.
    for (const schema of ["aeroviz-training-sample-v2", "aeroviz-training-sample-v3", "aeroviz-training-sample-v4",
                          "aeroviz-training-sample-v5", "aeroviz-training-sample-v6"]) {
      expect(refusal((raw) => { raw.schema = schema; })).toContain(
        `schema is "${schema}", expected "${TRAINING_SAMPLE_SCHEMA}" — a sample of another format is not read`);
    }
  });

  it("reads a flight whose type is unresolved as null, and refuses an empty type", () => {
    const raw: any = mockSample();
    raw.flights[1].typecode = null;
    expect(parsed(raw).flights[1].typecode).toBeNull();
    expect(refusal((sample) => { sample.flights[0].typecode = ""; })).toContain("typecode");
  });

  it("refuses another reading rule and another spec by name", () => {
    for (const rule of ["box-v3", "instruction-v1", "instruction-v2"]) {
      expect(refusal((raw) => { raw.vocabulary.readingRule = rule; })).toContain(
        `readingRule is ${rule}, and this reader is written for ${TRAINING_READING_RULE}`);
    }
    expect(refusal((raw) => { raw.vocabulary.specSha256 = "0".repeat(64); })).toMatch(
      new RegExp(`specSha256 is 000000000000, and this reader is written for spec ${TRAINING_SPEC_SHA256.slice(0, 12)}`));
  });

  it("refuses the columns in another order — they are positional", () => {
    expect(refusal((raw) => { raw.vocabulary.columns = ["runway", "approach", "heading", "angle", "altitude", "speed"]; }))
      .toMatch(/columns are \[runway, approach, heading, angle, altitude, speed\], expected \[runway, approach, heading, altitude, angle, speed\]/);
  });

  it("refuses a class table that disagrees with the stated class count", () => {
    expect(refusal((raw) => { raw.vocabulary.headingTargetsDeg.pop(); }))
      .toMatch(/headingTargetsDeg holds 71 classes, but classCounts says 72/);
  });

  it("refuses a step 0 that does not carry all six words", () => {
    expect(refusal((raw) => {
      raw.flights[0].words.events = raw.flights[0].words.events.filter((e: any) => !(e.row === 0 && e.column === 4));
    })).toMatch(/step 0 carries no word for angle/);
  });

  it("refuses a per-row table that is not the events filled forward", () => {
    expect(refusal((raw) => { raw.flights[0].words.inForce[2][12] = WORD.heading270; }))
      .toMatch(/inForce\.heading\[12\] is 54, but the events put 36 in force there/);
  });

  it("refuses a word outside its column's classes", () => {
    expect(refusal((raw) => { raw.flights[0].words.events[0].value = 2; }))
      .toMatch(/events\[0\]\.value is 2, not a whole number in 0…1/);
  });

  it("refuses a word issued for a reason the labeller does not name — the holds reading's kinds included", () => {
    for (const kind of ["guess", "turn", "turn-split", "intercept"]) {
      expect(refusal((raw) => { raw.flights[0].words.events[6].kind = kind; }))
        .toMatch(new RegExp(`events\\[6\\]\\.kind is "${kind}", not one of initial, per-step`));
    }
  });

  it("refuses an envelope list that does not match its column's words", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading.pop(); }))
      .toMatch(/heading holds 2 envelopes for 3 heading words/);
    expect(refusal((raw) => { raw.flights[0].envelopes.altitude[1].row = 21; }))
      .toMatch(/altitude\[1\]: is \(row 21, value 181\), but the sentence's altitude word 1 is \(row 20, value 181, target\)/);
  });

  it("refuses a verdict whose count is not the count of its own flags", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.altitude[1].check.inside = 40; }))
      .toMatch(/says 40 rows inside, but the tube's own flags count 39/);
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[0].bandInside[3] = 0; }))
      .toMatch(/says 30 band rows inside, but the band's own flags count 29/);
  });

  it("refuses a tube of the wrong length and an inverted one", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.altitude[0].lowerM.pop(); }))
      .toMatch(/altitude\[0\]\.lowerM has 19 values, expected 20/);
    expect(refusal((raw) => { raw.flights[0].envelopes.altitude[0].lowerM[3] = 2000; }))
      .toMatch(/is inverted at row 3/);
  });

  it("refuses a runway that is not the candidate pointed at", () => {
    expect(refusal((raw) => { raw.flights[0].runway = "27"; }))
      .toMatch(/runway is 27, but candidate 0 is 09/);
  });

  it("refuses a missing field by its path rather than reading it as a default", () => {
    expect(refusal((raw) => { delete raw.flights[0].envelopes.approach.corridor; }))
      .toMatch(/envelopes\.approach\.corridor is not an object/);
    expect(refusal((raw) => { delete raw.flights[0].envelopes.approach.captureTurn; }))
      .toMatch(/envelopes\.approach\.captureTurn is not an object/);
  });

  it("refuses a heading band whose rows are not a lead after its word, or do not end at the next word's or the clearance", () => {
    expect(refusal((raw) => { Object.assign(raw.flights[0].envelopes.heading[1], { firstRow: 9, inside: [1, 1, 1] }); }))
      .toMatch(/heading\[1\]: firstRow is 9, but a word told at step 8 is judged from 10 \(the 4 s lead\)/);
    // past the next word's first row
    expect(refusal((raw) => {
      const item = raw.flights[0].envelopes.heading[1];
      item.stopRow = 13; item.inside = [1, 1, 1]; item.check = { rows: 3, inside: 3 };
    })).toMatch(/heading\[1\]: stopRow is 13, but this word's rows end at 12/);
    // short of it: a band cut early, its check counting the rows it kept
    expect(refusal((raw) => {
      const item = raw.flights[0].envelopes.heading[1];
      item.stopRow = 11; item.inside = [1]; item.check = { rows: 1, inside: 1 };
    })).toMatch(/heading\[1\]: stopRow is 11, but this word's rows end at 12/);
    // past the clearance
    expect(refusal((raw) => {
      const item = raw.flights[0].envelopes.heading[2];
      item.stopRow = 21; item.inside.push(1); item.check = { rows: 9, inside: 8 };
    })).toMatch(/heading\[2\]: stopRow is 21, but this word's rows end at 20/);
  });

  it("refuses a heading band's verdicts that are not one per row, or that its check does not count", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[2].inside.pop(); }))
      .toMatch(/heading\[2\]\.inside has 7 values, expected 8/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[2].check.inside = 8; }))
      .toMatch(/heading\[2\]\.check: says 8 rows inside, but the band's own flags count 7/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[2].check.rows = 7; }))
      .toMatch(/heading\[2\]\.check\.rows is 7, not a whole number in 8…8/);
  });

  it("refuses a band that is not its word's target ± the vocabulary's tolerance", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[2].targetOnTrackDeg = 185; }))
      .toMatch(/targetOnTrackDeg is 185, not the word's 180° on any branch/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[2].bandDeg = [175, 185]; }))
      .toMatch(/bandDeg is \[175, 185\], not 180° ± the vocabulary's 4\.5°/);
    // a whole turn away is the same heading on another branch of the unwrapped track
    const raw: any = mockSample();
    Object.assign(raw.flights[0].envelopes.heading[2], { targetOnTrackDeg: 540, bandDeg: [535.5, 544.5] });
    expect(parsed(raw).flights[0].envelopes.heading[2].targetOnTrackDeg).toBe(540);
  });

  it("refuses a lead in steps that is not the lead in seconds", () => {
    expect(refusal((raw) => { raw.vocabulary.headingLeadS = 3; }))
      .toMatch(/headingLeadRows is 2, but 2 steps of 2 s are not headingLeadS 3 s/);
    expect(refusal((raw) => { raw.vocabulary.headingLeadRows = 2.5; }))
      .toMatch(/headingLeadRows is 2\.5, not a whole number of at least 0/);
  });

  it("refuses the holds reading's fields by their absence, never reading them as defaults", () => {
    expect(refusal((raw) => { delete raw.flights[0].envelopes.heading[0].firstRow; raw.flights[0].envelopes.heading[0].turn = null; }))
      .toMatch(/heading\[0\]\.firstRow is undefined, not a number/);
    expect(refusal((raw) => { delete raw.vocabulary.headingLeadS; raw.vocabulary.headingMaxTurnDeg = 150; }))
      .toMatch(/headingLeadS is undefined, not a number/);
  });

  it("refuses markers that are not the words' own rows", () => {
    expect(refusal((raw) => { raw.flights[0].joinRow = 21; raw.flights[0].envelopes.approach.clearanceRow = 21; }))
      .toMatch(/joinRow is 21, but the clearance is issued at row 20/);
    expect(refusal((raw) => { raw.flights[0].unspecifiedRow = 31; }))
      .toMatch(/unspecifiedRow is 31, but the last speed word is 47 at row 30/);
  });

  it("refuses a speed verdict at odds with its own parts", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[0].check.arrivalRows = 2; }))
      .toMatch(/arrivalRows is 2, not a whole number in 0…0/);
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[0].check.transitionOk = false; }))
      .toMatch(/contained is true, with the transition failed/);
  });

  it("refuses a capture turn that is not the clearance's to the capture's, or one a flight on the final has", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.approach.captureTurn.startRow = 19; }))
      .toMatch(/captureTurn\.startRow is 19, not a whole number in 20…20/);
    expect(refusal((raw) => { raw.flights[0].envelopes.approach.captureTurn.endRow = 26; }))
      .toMatch(/captureTurn\.endRow is 26, not a whole number in 25…25/);
    expect(refusal((raw) => { raw.flights[0].envelopes.approach.captureTurn = null; }))
      .toMatch(/captureTurn is absent with the capture at step 25/);
    expect(refusal((raw) => { raw.flights[1].envelopes.approach.captureTurn = raw.flights[0].envelopes.approach.captureTurn; }))
      .toMatch(/captureTurn is given with the capture at step 0/);
  });

  it("refuses a runway's landing limit above the vocabulary's", () => {
    expect(refusal((raw) => { raw.candidates[1].landingCrossLimitM = 1200; }))
      .toMatch(/candidates\[1\]: landingCrossLimitM is 1200, not in \(0, 1000\] m/);
  });

  it("refuses a measured angle for the level class", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.angle[0].measuredDeg = 0.1; }))
      .toMatch(/angle\[0\]: a measured angle is given for every class but level/);
  });

  it("refuses a speed band for 'unspecified'", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[1].bandMps = [10, 20]; }))
      .toMatch(/bandMps is given for "unspecified"/);
  });
});

describe("the index", () => {
  it("keeps every entry and names what the reader refuses, from the manifest alone", () => {
    const index = parseTrainingIndex(mockIndex());
    if (!index.ok) throw new Error(index.problem);
    expect(index.value.sets.map((entry) => entry.id))
      .toEqual(["box_v3", "instruction_v1", "instruction_v2", SET_ID, "prior_s1337_val"]);
    const [old, first, second, current, prior] = index.value.sets;
    expect(trainingSetRefusal(current)).toBeNull();
    expect(trainingSetRefusal(old)).toMatch(/read under box-v3, a superseded vocabulary/);
    expect(trainingSetRefusal(first)).toMatch(/read under instruction-v1, a superseded vocabulary/);
    expect(trainingSetRefusal(second)).toMatch(/read under instruction-v2, a superseded vocabulary/);
    expect(trainingSetRefusal(prior)).toMatch(/read under segment-v13/);
    expect(trainingSetRefusal({ ...current, vocabularySha256: "9".repeat(64) }))
      .toContain(`spec 999999999999 is not the ${TRAINING_READING_RULE} spec`);
    expect(trainingSetRefusal({ ...current, kind: "prior-generated" }))
      .toContain("this reader opens only vocabulary-readback sets");
  });

  it("greys out one malformed entry without emptying the manifest", () => {
    const raw: any = mockIndex();
    delete raw.sets[0].cohort;
    const index = parseTrainingIndex(raw);
    if (!index.ok) throw new Error(index.problem);
    expect(index.value.sets.map((entry) => entry.id)).toEqual(["instruction_v1", "instruction_v2", SET_ID, "prior_s1337_val"]);
    expect(index.value.rejected).toEqual([{ id: "box_v3", problem: "set box_v3.cohort is not an object" }]);
  });

  it("refuses a manifest of another schema whole", () => {
    expect(parseTrainingIndex({ ...mockIndex(), schema: "aeroviz-training-index-v0" })).toEqual({
      ok: false, problem: `schema is "aeroviz-training-index-v0", expected "${TRAINING_INDEX_SCHEMA}"`,
    });
  });
});

describe("reading a sentence", () => {
  it("labels every column from the vocabulary's tables, in SI", () => {
    const { vocabulary, candidates } = parsed();
    const label = (column: (typeof TRAINING_COLUMNS)[number], value: number) =>
      trainingWordLabel(vocabulary, candidates, column, value);
    expect(label("runway", WORD.runway09)).toBe("09");
    expect(label("approach", WORD.cleared)).toBe("cleared");
    expect(label("heading", WORD.heading090)).toBe("090°");
    expect(label("altitude", WORD.altitude1110)).toBe("1110 m");
    expect(label("altitude", WORD.land)).toBe("descend to land");
    expect(label("angle", WORD.level)).toBe("level");
    expect(label("angle", WORD.descent3)).toBe("descent 3 (3.06°)");
    expect(label("speed", WORD.speed110)).toBe("110 m/s");
    expect(label("speed", WORD.unspecified)).toBe("unspecified");
  });

  it("runs a column from each issue to the next, the last to the end", () => {
    const flight = parsed().flights[0];
    expect(trainingColumnRuns(flight, "heading").map(({ row, endRow, value }) => [row, endRow, value]))
      .toEqual([[0, 8, WORD.heading270], [8, 10, WORD.heading225], [10, MOCK_ROWS, WORD.heading180]]);
    expect(trainingColumnRuns(flight, "runway")).toHaveLength(1);
  });

  it("finds one column's word at a row, and its place among that column's words (= its envelope's)", () => {
    const flight = parsed().flights[0];
    expect(trainingWordAt(flight, "heading", 7)).toMatchObject({ index: 0, row: 0, endRow: 8 });
    expect(trainingWordAt(flight, "heading", 9)).toMatchObject({ index: 1, row: 8, endRow: 10 });
    // after the capture the heading word in force is still the last one: the corridor is not a heading word's
    expect(trainingWordAt(flight, "heading", 25)).toMatchObject({ index: 2, row: 10, endRow: MOCK_ROWS });
    // heading changes at step 10 and altitude at step 20: at step 15 they are different runs
    expect(trainingWordAt(flight, "altitude", 15)).toMatchObject({ index: 0, row: 0, endRow: 20 });
    expect(trainingWordAt(flight, "altitude", 20).index).toBe(1);
    expect(trainingWordAt(flight, "speed", MOCK_ROWS - 1)).toMatchObject({ index: 1, row: 30, endRow: MOCK_ROWS });
    expect(rowAtTime(flight.signals.tS, 21)).toBe(10);
  });

  it("draws rows outside on to the next row, so one row outside is a segment", () => {
    const inside = [false, true, true, false, false];
    expect(outsideSpans(inside, 12, 59)).toEqual([[12, 13], [15, 17]]);
    // at the line's end, back to the row before
    expect(outsideSpans(inside, 12, 16)).toEqual([[12, 13], [15, 16]]);
    expect(outsideSpans([false], 16, 16)).toEqual([[15, 16]]);
    expect(outsideSpans(inside.map(() => true), 12, 59)).toEqual([]);
    // a tube outside on one row: a segment to the next row, never a point
    expect(outsideSpans([true, true, true, true, true, false, true], 0, 59)).toEqual([[5, 6]]);
  });

  it("counts the labeller's verdicts: heading words judged apart from those with no row of their own", () => {
    const [vectored, straight] = parsed().flights;
    const verdicts = trainingVerdicts(vectored);
    expect(verdicts).toMatchObject({
      headingJudged: 3, headingContained: 2, headingNotJudged: 0,
      altitudeWords: 2, altitudeContained: 1, speedWords: 1, speedContained: 1, instructionsAfterStep0: 6,
    });
    expect(verdicts.captureTurn).toMatchObject({ progressOk: true, rateOk: true });
    expect(verdicts.silentSteps).toBe(MOCK_ROWS - 1 - 4);   // steps 8, 10, 20 and 30 say something
    expect(trainingVerdicts(straight)).toMatchObject({ headingJudged: 0, headingContained: 0, headingNotJudged: 1, captureTurn: null });
  });
});
