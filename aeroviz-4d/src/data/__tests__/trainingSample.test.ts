/**
 * The Training contract, under `box-v2-wedge`.
 *
 * Three groups of test, and they are not the same question:
 *  • WHAT A WORD MEANS — the edge tables, the ladder, the wedge. These are the
 *    mirrors of the artefact's spec, and a drift in any of them silently moves
 *    every box.
 *  • WHAT THE READER REFUSES — a file of another rule, a table that does not
 *    tile, a sentence whose boxes leave a gap, a verdict that disagrees with the
 *    columns beside it. Each of these is a file that would otherwise draw
 *    something plausible and wrong.
 *  • THE VERDICT — containment, per row, computed here and checked against the
 *    file's own count.
 */

import { describe, expect, it } from "vitest";
import {
  TERMINAL_LANDED,
  TRAINING_BOX_KINDS,
  TRAINING_INSIDE_EPSILON,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  TRAINING_READING_RULE,
  TRAINING_SAMPLE_SCHEMA,
  altitudeFloorM,
  altitudeTargetM,
  altitudeWedgeM,
  eventInForce,
  headingBoxDeg,
  headingWordAt,
  isTrainingBoxKind,
  parseTrainingIndex,
  parseTrainingSample,
  speedBoxMps,
  trainingContainment,
  trainingWordBandLabel,
  trainingWordBox,
  trainingWordCounts,
  trainingWordLabel,
  type TrainingSample,
} from "../trainingSample";
import {
  MOCK_EVENT_TIMES_S,
  MOCK_HOLDS_S,
  MOCK_SPEC,
  MOCK_VOCABULARY,
  MOCK_WORDS,
  mockIndex,
  mockPriorIndex,
  mockPriorSample,
  mockSample,
} from "./trainingSample.fixture";

function parsed(raw: unknown = mockSample()): TrainingSample {
  const result = parseTrainingSample(raw, "vocabulary-readback");
  if (!result.ok) throw new Error(result.problem);
  return result.value;
}

function problem(raw: unknown, kind: "vocabulary-readback" | "prior-generated" = "vocabulary-readback"): string {
  const result = parseTrainingSample(raw, kind);
  expect(result.ok).toBe(false);
  return result.ok ? "" : result.problem;
}

describe("what a word means", () => {
  it("counts the classes from the tables, and knows which table tiles", () => {
    const counts = trainingWordCounts(MOCK_SPEC);
    // The edge tables hold one more value than they have words; the ladder holds
    // one target PER word. Getting that backwards shifts every altitude word by
    // half a box and still plots.
    expect(counts.heading).toBe(MOCK_SPEC.headingEdgesDeg.length - 1);
    expect(counts.speed).toBe(MOCK_SPEC.speedEdgesMps.length - 1);
    expect(counts.altitude).toBe(MOCK_SPEC.altitudeTargetsM.length);
    expect(counts.terminal).toBe(3);
    expect(counts.duration).toBe(MOCK_SPEC.durationMaxS / MOCK_SPEC.durationBinS + 1);
  });

  it("a heading or speed word is the interval between two edges", () => {
    expect(headingBoxDeg(MOCK_SPEC, 0)).toEqual([MOCK_SPEC.headingEdgesDeg[0], MOCK_SPEC.headingEdgesDeg[1]]);
    expect(speedBoxMps(MOCK_SPEC, 2)).toEqual([MOCK_SPEC.speedEdgesMps[2], MOCK_SPEC.speedEdgesMps[3]]);
    // they tile: one word's top edge is the next one's bottom
    expect(headingBoxDeg(MOCK_SPEC, 3)[1]).toBe(headingBoxDeg(MOCK_SPEC, 4)[0]);
  });

  it("the box on the course is the narrowest one", () => {
    const onCourse = headingWordAt(MOCK_SPEC, 0);
    const [low, high] = headingBoxDeg(MOCK_SPEC, onCourse);
    expect(low).toBeLessThanOrEqual(0);
    expect(high).toBeGreaterThanOrEqual(0);
    const widths = MOCK_SPEC.headingEdgesDeg.slice(1).map((edge, word) => edge - MOCK_SPEC.headingEdgesDeg[word]);
    expect(high - low).toBe(Math.min(...widths));
  });

  it("an altitude word is a target, and its own tolerance is a fraction of the target plus h0", () => {
    const word = 4;
    expect(altitudeTargetM(MOCK_SPEC, word)).toBe(MOCK_SPEC.altitudeTargetsM[word]);
    expect(altitudeFloorM(MOCK_SPEC, word)).toBeCloseTo(
      MOCK_SPEC.redundancyFraction * (MOCK_SPEC.altitudeTargetsM[word] + MOCK_SPEC.altitudeH0M), 9);
    // The `+ h0` is what gives a target of 0 m a box at all: without it the one
    // target on the ladder that means "the threshold" would be unmeetable.
    const atZero = MOCK_SPEC.altitudeTargetsM.indexOf(0);
    expect(altitudeFloorM(MOCK_SPEC, atZero)).toBeGreaterThan(0);
  });

  it("the wedge closes onto the target's own box, and opens ASYMMETRICALLY going back", () => {
    const word = 5;
    const target = altitudeTargetM(MOCK_SPEC, word);
    const floor = altitudeFloorM(MOCK_SPEC, word);
    expect(altitudeWedgeM(MOCK_SPEC, word, 0)).toEqual([target - floor, target + floor]);

    const [low, high] = altitudeWedgeM(MOCK_SPEC, word, 1000);
    // down is the WIDER side and it opens ABOVE the target: the wedge is the set
    // the target is backward-reachable from, and losing height has the most room.
    expect(high - target - floor).toBeCloseTo(1000 * Math.tan((1.5 * Math.PI) / 180), 6);
    expect(target - floor - low).toBeCloseTo(1000 * Math.tan((1.0 * Math.PI) / 180), 6);
    expect(high - target).toBeGreaterThan(target - low);
  });

  it("a negative remaining path cannot widen the wedge", () => {
    // The path column only grows, so this cannot arise from a good file — but a
    // clamp that was missing would turn rounding at a segment's end into a box
    // wider than the one the labeller used.
    expect(altitudeWedgeM(MOCK_SPEC, 3, -500)).toEqual(altitudeWedgeM(MOCK_SPEC, 3, 0));
  });

  it("only three kinds are boxes; the other three return null", () => {
    for (const kind of TRAINING_KINDS) {
      const box = trainingWordBox(MOCK_SPEC, kind, 0);
      expect(box === null).toBe(!isTrainingBoxKind(kind));
    }
  });

  it("labels the interval, in SI, and the altitude word's target with its own tolerance", () => {
    expect(trainingWordLabel(MOCK_SPEC, "speed", 2)).toBe("80–90 m/s");
    expect(trainingWordLabel(MOCK_SPEC, "runway", 0)).toBe(MOCK_SPEC.runwayIdents[0]);
    expect(trainingWordLabel(MOCK_SPEC, "duration", 5)).toBe(`${5 * MOCK_SPEC.durationBinS} s`);
    expect(trainingWordLabel(MOCK_SPEC, "terminal", TERMINAL_LANDED)).toBe("landed");
    // the altitude label names the TARGET; the band label adds the box it closes onto
    expect(trainingWordLabel(MOCK_SPEC, "altitude", 5)).toBe("170 m");
    expect(trainingWordBandLabel(MOCK_SPEC, "altitude", 5)).toBe("170 m±11.0");
    // a kind whose label is already the interval gains nothing from the band label
    expect(trainingWordBandLabel(MOCK_SPEC, "speed", 2)).toBe(trainingWordLabel(MOCK_SPEC, "speed", 2));
  });
});

describe("the sample the reader accepts", () => {
  it("reads the fixture, and every row of it is inside its boxes", () => {
    const sample = parsed();
    expect(sample.vocabulary.readingRule).toBe(TRAINING_READING_RULE);
    const flight = sample.flights[0];
    for (const kind of TRAINING_BOX_KINDS) {
      expect(flight.envelope.inside[kind].rows).toBe(flight.observed.tS.length);
      expect(flight.envelope.inside[kind].outside).toBe(0);
    }
  });

  it("the boxes tile the track, so every row has exactly one in force", () => {
    const flight = parsed().flights[0];
    const forced = eventInForce(flight.sentence.eventTimesS, flight.observed.tS);
    expect(forced.length).toBe(flight.observed.tS.length);
    expect(forced[0]).toBe(0);
    expect(forced[forced.length - 1]).toBe(flight.sentence.eventTimesS.length - 1);
    // monotone: a row never falls back to an earlier box
    for (let row = 1; row < forced.length; row += 1) {
      expect(forced[row]).toBeGreaterThanOrEqual(forced[row - 1]);
    }
    const last = flight.sentence.eventTimesS.length - 1;
    expect(flight.sentence.eventTimesS[last] + flight.sentence.holdS[last]).toBeCloseTo(flight.durationS, 6);
  });

  it("carries one box per event, in the sentence's own order", () => {
    const flight = parsed().flights[0];
    expect(flight.envelope.events.length).toBe(flight.sentence.eventTimesS.length);
    flight.envelope.events.forEach((box, event) => {
      expect(box.eventS).toBe(flight.sentence.eventTimesS[event]);
      expect(box.holdS).toBe(flight.sentence.holdS[event]);
      expect(box.lon).toHaveLength(4);
      expect(box.toGoM).toHaveLength(4);
    });
  });

  it("keeps the model's words and the boxes THEY make", () => {
    const result = parseTrainingSample(mockPriorSample(), "prior-generated");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const flight = result.value.flights[0];
    expect(flight.prior).toBeDefined();
    expect(flight.prior?.words).toHaveLength(flight.sentence.words.length);
    // The model's sentence is a different sentence, so its boxes are a different
    // envelope — and on this fixture it is one the track leaves. That is the
    // answer the comparison exists to give.
    const outside = TRAINING_BOX_KINDS.reduce(
      (total, kind) => total + (flight.prior?.envelope.inside[kind].outside ?? 0), 0);
    expect(outside).toBeGreaterThan(0);
  });
});

describe("what the reader refuses", () => {
  it("a file of another reading rule, by name", () => {
    const raw = mockSample() as any;
    raw.vocabulary.readingRule = "segment-v14";
    expect(problem(raw)).toMatch(/readingRule is "segment-v14".*written for box-v2-wedge/s);
  });

  it("a sample of the retired schema", () => {
    const raw = mockSample() as any;
    raw.schema = "aeroviz-training-sample-v1";
    expect(problem(raw)).toContain(TRAINING_SAMPLE_SCHEMA);
  });

  it("the kinds in another order — the columns are positional", () => {
    const raw = mockSample() as any;
    raw.kinds = ["altitude", "heading", "speed", "runway", "duration", "terminal"];
    expect(problem(raw)).toContain("expected [heading,altitude,speed,runway,duration,terminal]");
  });

  it("an edge table that does not tile -180…180", () => {
    const raw = mockSample() as any;
    raw.vocabulary.headingEdgesDeg = [-90, -30, 0, 30, 90];
    raw.vocabulary.words.heading = 4;
    expect(problem(raw)).toContain("does not cover -180");
  });

  it("a table stored unsorted", () => {
    const raw = mockSample() as any;
    raw.vocabulary.speedEdgesMps = [60, 80, 70, 90, 100, 110, 125];
    expect(problem(raw)).toContain("not stored sorted and distinct");
  });

  it("a wedge whose two angles are the wrong way round", () => {
    const raw = mockSample() as any;
    raw.vocabulary.altitudeDownDeg = 1.0;
    raw.vocabulary.altitudeUpDeg = 1.5;
    // Swapping them draws a corridor of exactly the same width with the slack on
    // the wrong side of the target, which nothing else here would catch.
    expect(problem(raw)).toContain("the descent side is the wider one");
  });

  it("class counts that disagree with the file's own tables", () => {
    const raw = mockSample() as any;
    raw.vocabulary.words.altitude = MOCK_SPEC.altitudeTargetsM.length + 1;
    expect(problem(raw)).toContain("the stated counts and the spec disagree");
  });

  it("a hold that is not what its duration word says", () => {
    const raw = mockSample() as any;
    raw.flights[0].sentence.holdS[2] += MOCK_SPEC.durationBinS;
    expect(problem(raw)).toMatch(/the hold and the word are ONE answer/);
  });

  it("boxes that leave a gap in the track", () => {
    const raw = mockSample() as any;
    // shorten one hold AND its word together, so the hold/word check passes and
    // only the tiling check can catch it
    raw.flights[0].sentence.holdS[1] -= MOCK_SPEC.durationBinS;
    raw.flights[0].sentence.words[1][TRAINING_KIND_COLUMN.duration] -= 1;
    expect(problem(raw)).toContain("the boxes tile the track");
  });

  it("a sentence that does not open with the track", () => {
    const raw = mockSample() as any;
    raw.flights[0].sentence.eventTimesS[0] = 2;
    expect(problem(raw)).toContain("the first box opens with the track");
  });

  it("an envelope inverted at one row", () => {
    const raw = mockSample() as any;
    const low = raw.flights[0].envelope.altLoM;
    const high = raw.flights[0].envelope.altHiM;
    [low[7], high[7]] = [high[7], low[7]];
    expect(problem(raw)).toContain("is inverted at row 7");
  });

  it("a verdict that disagrees with the columns beside it", () => {
    const raw = mockSample() as any;
    raw.flights[0].envelope.inside.speed.outside += 1;
    // This is the one check that stands in for a producer that is not in this
    // repository: the exporter derives the boxes from the artefact's spec, and
    // this reader measures them against the columns it shipped with them.
    expect(problem(raw)).toMatch(/the exporter's boxes and this reader's reading of them disagree/);
  });

  it("a footprint too short to be an outline", () => {
    const raw = mockSample() as any;
    raw.flights[0].envelope.events[0].toGoM = [0, 1];
    expect(problem(raw)).toContain("shorter than three points");
  });

  it("a footprint whose two coordinate systems are different lengths", () => {
    // The four arrays are ONE outline twice over. A length that differs between
    // them is two shapes drawn as though they were one — the plan view would show
    // a sector the 3D scene does not have.
    const raw = mockSample() as any;
    raw.flights[0].envelope.events[0].toGoM = [0, 1, 2, 3, 4];
    expect(problem(raw)).toContain("one outline in two coordinate systems");
  });

  it("the footprint is a SECTOR, and its first point is the aircraft", () => {
    const flight = parsed().flights[0];
    flight.envelope.events.forEach((box, event) => {
      // the apex sits on the track, at the row where this word opened
      const row = flight.observed.tS.findIndex((t) => t >= box.eventS);
      expect(box.toGoM[0]).toBeCloseTo(flight.observed.toGoM[row], 6);
      expect(box.crossM[0]).toBeCloseTo(flight.observed.crossM[row], 6);
      // and it reaches no further than the hold times the speed box's UPPER edge
      const depth = box.holdS * box.speedHiMps;
      for (let point = 1; point < box.toGoM.length; point += 1) {
        const reach = Math.hypot(box.toGoM[point] - box.toGoM[0], box.crossM[point] - box.crossM[0]);
        expect(reach).toBeLessThanOrEqual(depth + 1e-6);
      }
      void event;
    });
  });

  it("a path column that falls", () => {
    const raw = mockSample() as any;
    raw.flights[0].observed.pathM[5] = raw.flights[0].observed.pathM[4] - 10;
    expect(problem(raw)).toContain("a path length only grows");
  });

  it("a reading block written with another epsilon", () => {
    const raw = mockSample() as any;
    raw.reading.insideEpsilon = TRAINING_INSIDE_EPSILON * 10;
    expect(problem(raw)).toContain("the two verdicts are compared");
  });

  it("a word outside its kind's classes, naming the column order first", () => {
    const raw = mockSample() as any;
    raw.flights[0].sentence.words[0][TRAINING_KIND_COLUMN.runway] = 9;
    expect(problem(raw)).toContain("the columns are positional");
  });

  it("a read-back set that carries a model's words, and a prior set that does not", () => {
    expect(problem(mockPriorSample())).toContain("this set's kind is vocabulary-readback");
    const raw = mockSample() as any;
    expect(problem(raw, "prior-generated")).toContain("every flight carries what the model said");
  });

  it("a prior asked a way this view's wording does not describe", () => {
    const raw = mockPriorSample() as any;
    raw.prior.method = "free-generation";
    expect(problem(raw, "prior-generated")).toContain("a free run is a different experiment");
  });
});

describe("the verdict", () => {
  it("judges every row exactly once, against the box in force", () => {
    const flight = parsed().flights[0];
    const measured = trainingContainment(flight, flight.envelope, MOCK_VOCABULARY, flight.sentence.words);
    for (const kind of TRAINING_BOX_KINDS) {
      // One verdict per row: an earlier reading of this walked spans and
      // double-counted the boundary row two consecutive spans share.
      expect(measured[kind].inside).toHaveLength(flight.observed.tS.length);
      expect(measured[kind].rows).toBe(flight.observed.tS.length);
      expect(measured[kind].outside).toBe(flight.envelope.inside[kind].outside);
    }
  });

  it("marks the row that leaves its box, and only that row", () => {
    const raw = mockSample() as any;
    const flight = parsed().flights[0];
    void raw;
    const moved = {
      ...flight,
      observed: {
        ...flight.observed,
        readSpeedMps: flight.observed.readSpeedMps.map(
          (value, row) => (row === 12 ? value + 40 : value)),
      },
    };
    const measured = trainingContainment(moved, flight.envelope, MOCK_VOCABULARY, flight.sentence.words);
    expect(measured.speed.outside).toBe(1);
    expect(measured.speed.inside[12]).toBe(false);
    expect(measured.speed.inside[11]).toBe(true);
    // the other two kinds are untouched: the verdict is per kind, not per flight
    expect(measured.heading.outside).toBe(0);
    expect(measured.altitude.outside).toBe(0);
  });

  it("a row exactly on an edge is inside, and one a whisker past it is not", () => {
    const flight = parsed().flights[0];
    const forced = eventInForce(flight.sentence.eventTimesS, flight.observed.tS);
    const [, high] = speedBoxMps(MOCK_VOCABULARY, flight.sentence.words[forced[3]][TRAINING_KIND_COLUMN.speed]);
    const onEdge = {
      ...flight,
      observed: { ...flight.observed, readSpeedMps: flight.observed.readSpeedMps.map((v, row) => (row === 3 ? high : v)) },
    };
    expect(trainingContainment(onEdge, flight.envelope, MOCK_VOCABULARY, flight.sentence.words).speed.outside).toBe(0);
    const past = {
      ...flight,
      observed: {
        ...flight.observed,
        readSpeedMps: flight.observed.readSpeedMps.map(
          (v, row) => (row === 3 ? high + 10 * TRAINING_INSIDE_EPSILON : v)),
      },
    };
    expect(trainingContainment(past, flight.envelope, MOCK_VOCABULARY, flight.sentence.words).speed.outside).toBe(1);
  });
});

describe("the manifest", () => {
  it("keeps the good sets and reports a bad one by field", () => {
    const raw = mockIndex() as any;
    raw.sets.push({ ...raw.sets[0], id: "broken", cohort: undefined });
    const result = parseTrainingIndex(raw);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.sets.map((item) => item.id)).toEqual(["box_v3"]);
    expect(result.value.rejected[0].id).toBe("broken");
    expect(result.value.rejected[0].problem).toContain("cohort");
  });

  it("a prior set names its model in the manifest, before anyone downloads it", () => {
    const result = parseTrainingIndex(mockPriorIndex());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.sets[0].prior?.seed).toBe(1337);
  });

  it("lists a superseded set rather than dropping it — its rule is in the manifest", () => {
    const raw = mockIndex() as any;
    raw.sets[0].readingRule = "segment-v14";
    const result = parseTrainingIndex(raw);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    // The manifest does not pin the rule: the panel marks such a set and says why
    // when it is picked, which is what keeps a vocabulary bump from emptying the
    // picker (AV6 in reverse).
    expect(result.value.sets[0].readingRule).toBe("segment-v14");
    expect(result.value.rejected).toHaveLength(0);
  });
});

describe("the sentence's own invariants", () => {
  it("the fixture's events are strictly increasing and its holds are its duration words", () => {
    MOCK_EVENT_TIMES_S.forEach((time, event) => {
      if (event > 0) expect(time).toBeGreaterThan(MOCK_EVENT_TIMES_S[event - 1]);
      expect(MOCK_HOLDS_S[event]).toBe(
        MOCK_WORDS[event][TRAINING_KIND_COLUMN.duration] * MOCK_SPEC.durationBinS);
    });
  });

  it("only the last event says landed", () => {
    MOCK_WORDS.forEach((row, event) => {
      expect(row[TRAINING_KIND_COLUMN.terminal] === TERMINAL_LANDED).toBe(event === MOCK_WORDS.length - 1);
    });
  });
});
