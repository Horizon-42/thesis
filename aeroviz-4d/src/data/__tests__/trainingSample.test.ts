/**
 * The Training reader under the instruction vocabulary: what it accepts, and — the point of it — that
 * everything it refuses is refused BY NAME, with the field that failed.
 */
import { describe, expect, it } from "vitest";

import {
  parseTrainingIndex,
  parseTrainingSample,
  rowAtTime,
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
    expect(flight.envelopes.approach.captureTurn?.startRow).toBe(20);
  });

  // ── refused by name ─────────────────────────────────────────────────────
  it("refuses a sample of an earlier format by its schema name, whatever vocabulary it carries", () => {
    // v2: the box vocabulary's sample; v3: instruction-v1's, and instruction-v2's before its change of
    // shape got a name of its own. Both are superseded names, so they are written out here.
    for (const schema of ["aeroviz-training-sample-v2", "aeroviz-training-sample-v3"]) {
      expect(refusal((raw) => { raw.schema = schema; })).toContain(
        `schema is "${schema}", expected "${TRAINING_SAMPLE_SCHEMA}" — a sample of another format is not read`);
    }
  });

  it("refuses another reading rule and another spec by name", () => {
    for (const rule of ["box-v3", "instruction-v1"]) {
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

  it("refuses a word issued for a reason the labeller does not name", () => {
    expect(refusal((raw) => { raw.flights[0].words.events[6].kind = "guess"; }))
      .toMatch(/events\[6\]: kind is guess, not one of the labeller's initial, turn/);
  });

  it("refuses an envelope list that does not match its column's words", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading.pop(); }))
      .toMatch(/heading holds 1 envelopes for 2 heading words/);
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
    expect(refusal((raw) => { delete raw.flights[1].envelopes.approach.captureTurn; }))
      .toMatch(/envelopes\.approach\.captureTurn is not an object/);
  });

  it("refuses a heading envelope that is half a turn or half a hold", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].turnBandDeg = null; }))
      .toMatch(/heading\[1\]: turn, turnBandDeg, turnEndRow and check come together or not at all/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[0].funnel = null; }))
      .toMatch(/heading\[0\]: funnel, holdBandDeg and holdStartRow come together or not at all/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].turnEndRow = 30; }))
      .toMatch(/the turn ends at row 30, after the hold ends at 20/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].check.kind = "intercept"; }))
      .toMatch(/is the check of a intercept, but the word is turn/);
  });

  it("refuses markers that are not the words' own rows", () => {
    expect(refusal((raw) => { raw.flights[0].joinRow = 11; raw.flights[0].envelopes.approach.clearanceRow = 11; }))
      .toMatch(/joinRow is 11, but the clearance is issued at row 10/);
    expect(refusal((raw) => { raw.flights[0].unspecifiedRow = 31; }))
      .toMatch(/unspecifiedRow is 31, but the last speed word is 47 at row 30/);
  });

  it("refuses a speed verdict at odds with its own parts", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[0].check.arrivalRows = 2; }))
      .toMatch(/arrivalRows is 2, not a whole number in 0…0/);
    expect(refusal((raw) => { raw.flights[0].envelopes.speed[0].check.transitionOk = false; }))
      .toMatch(/contained is true, with the transition failed/);
  });

  it("refuses a turn drawn with limits other than the vocabulary's, or an end that is not four corners", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].turn.rateMinDegS = 0.4; }))
      .toMatch(/turn: rateMinDegS is 0\.4, but the vocabulary says 0\.5/);
    expect(refusal((raw) => {
      const end = raw.flights[0].envelopes.heading[1].turn.end;
      for (const key of ["eM", "nM", "lon", "lat"]) end[key] = end[key].slice(0, 2);
    })).toMatch(/turn\.end: holds 2 points, fewer than 4/);
    expect(refusal((raw) => { raw.flights[0].envelopes.approach.captureTurn.turn.startDelayMaxS = 9; }))
      .toMatch(/captureTurn\.turn: startDelayMaxS is 9, but the vocabulary says 10\.5/);
  });

  it("reads a turn already within the target's band at issue, whose paths are one point each", () => {
    const raw: any = mockSample();
    const turn = raw.flights[0].envelopes.approach.captureTurn.turn;
    const start = { eM: [turn.region.eM[0]], nM: [turn.region.nM[0]], lon: [turn.region.lon[0]], lat: [turn.region.lat[0]] };
    turn.fastPath = start;
    turn.slowPath = start;
    expect(parsed(raw).flights[0].envelopes.approach.captureTurn!.turn.fastPath.eM).toHaveLength(1);
  });

  it("refuses a hold check that is not the word's own hold rows", () => {
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].holdCheck.holdStartRow = 17; }))
      .toMatch(/holdCheck\.holdStartRow is 17, not a whole number in 16…16/);
    expect(refusal((raw) => { raw.flights[0].envelopes.heading[1].holdCheck.rows = 4; }))
      .toMatch(/holdCheck\.rows is 4, not a whole number in 5…5/);
    expect(refusal((raw) => {
      raw.flights[1].envelopes.heading[0].holdCheck = { holdStartRow: 0, holdEndRow: 0, rows: 1, inside: 1, halfWidthEndM: 0 };
    })).toMatch(/holdCheck is given for a word with no hold/);
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
    expect(index.value.sets.map((entry) => entry.id)).toEqual(["box_v3", "instruction_v1", SET_ID, "prior_s1337_val"]);
    const [old, first, current, prior] = index.value.sets;
    expect(trainingSetRefusal(current)).toBeNull();
    expect(trainingSetRefusal(old)).toMatch(/read under box-v3, a superseded vocabulary/);
    expect(trainingSetRefusal(first)).toMatch(/read under instruction-v1, a superseded vocabulary/);
    expect(trainingSetRefusal(prior)).toMatch(/read under segment-v13/);
    expect(trainingSetRefusal({ ...current, vocabularySha256: "9".repeat(64) }))
      .toContain(`spec 999999999999 is not the ${TRAINING_READING_RULE} spec`);
    expect(trainingSetRefusal({ ...current, kind: "prior-generated" }))
      .toContain(`no prior is trained on ${TRAINING_READING_RULE} yet`);
  });

  it("greys out one malformed entry without emptying the manifest", () => {
    const raw: any = mockIndex();
    delete raw.sets[0].cohort;
    const index = parseTrainingIndex(raw);
    if (!index.ok) throw new Error(index.problem);
    expect(index.value.sets.map((entry) => entry.id)).toEqual(["instruction_v1", SET_ID, "prior_s1337_val"]);
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
      .toEqual([[0, 10, WORD.heading270], [10, MOCK_ROWS, WORD.heading180]]);
    expect(trainingColumnRuns(flight, "runway")).toHaveLength(1);
  });

  it("finds one column's word at a row, and its place among that column's words (= its envelope's)", () => {
    const flight = parsed().flights[0];
    expect(trainingWordAt(flight, "heading", 9)).toMatchObject({ index: 0, row: 0, endRow: 10 });
    // after the capture the heading word in force is still the last one: the corridor is not a heading word's
    expect(trainingWordAt(flight, "heading", 25)).toMatchObject({ index: 1, row: 10, endRow: MOCK_ROWS });
    // heading changes at step 10 and altitude at step 20: at step 15 they are different runs
    expect(trainingWordAt(flight, "altitude", 15)).toMatchObject({ index: 0, row: 0, endRow: 20 });
    expect(trainingWordAt(flight, "altitude", 20).index).toBe(1);
    expect(trainingWordAt(flight, "speed", MOCK_ROWS - 1)).toMatchObject({ index: 1, row: 30, endRow: MOCK_ROWS });
    expect(rowAtTime(flight.signals.tS, 21)).toBe(10);
  });

  it("counts the labeller's verdicts, a split turn once", () => {
    const flight = parsed().flights[0];
    const verdicts = trainingVerdicts(flight);
    expect(verdicts).toMatchObject({
      turns: 1, turnsProgressOk: 1, turnsRateOk: 1, holdsJudged: 2, holdsContained: 1, holdsNotJudged: 0,
      altitudeWords: 2, altitudeContained: 1, speedWords: 1, speedContained: 1, instructionsAfterStep0: 5,
    });
    expect(verdicts.silentSteps).toBe(MOCK_ROWS - 1 - 3);   // steps 10, 20 and 30 say something
    const split = { ...flight.envelopes.heading[1].check!, parts: 2 };
    const twice = { ...flight, envelopes: { ...flight.envelopes, heading: [
      flight.envelopes.heading[0], { ...flight.envelopes.heading[1], check: split }, { ...flight.envelopes.heading[1], check: { ...split } },
    ] } };
    expect(trainingVerdicts(twice).turns).toBe(1);
  });
});
