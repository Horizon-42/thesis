import { describe, expect, it } from "vitest";

import {
  TERMINAL_GO_AROUND,
  TERMINAL_LANDED,
  TERMINAL_NEVER_OBSERVED,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  TRAINING_GEOMETRIC_COLUMNS,
  TRAINING_OBSERVED_COLUMNS,
  TRAINING_READING_RULE,
  TRAINING_SPEED_EDGE_COLUMNS,
  TRAINING_VERTICAL_BAND_COLUMNS,
  TRAINING_WORD_COLUMNS,
  formatSeconds,
  parseTrainingIndex,
  parseTrainingSample,
  trainingIndexPath,
  trainingSamplePath,
  trainingWordCounts,
  trainingWordBandLabel,
  trainingWordLabel,
  trainingWordTolerance,
  durationCentreS,
  headingCentreDeg,
  speedCentreMps,
  speedToleranceMps,
  verticalCentreDeg,
  verticalToleranceDeg,
  wrapDeg,
} from "../trainingSample";
import {
  DESCEND_31,
  LEVEL,
  MOCK_VOCABULARY,
  mockIndex,
  mockSample,
} from "./trainingSample.fixture";

function sampleWith(mutate: (sample: any) => void) {
  const sample = mockSample();
  mutate(sample);
  return parseTrainingSample(sample);
}

describe("the six word kinds", () => {
  // MIRROR of instructions.INSTRUCTION_KINDS. The columns are positional, so this
  // order is the contract; the intercept word was deleted on 2026-09-20 (D73).
  it("is heading, vertical, speed, runway, duration, terminal — in that order", () => {
    expect([...TRAINING_KINDS]).toEqual([
      "heading",
      "vertical",
      "speed",
      "runway",
      "duration",
      "terminal",
    ]);
    expect(TRAINING_WORD_COLUMNS).toBe(6);
    expect(TRAINING_KINDS).not.toContain("intercept");
    // The word the vertical one replaced. Its column is the same, which is
    // exactly why an artefact read under the old rule has to be refused rather
    // than read: every row would still parse.
    expect(TRAINING_KINDS).not.toContain("altitude");
  });

  it("derives each kind's class count from the file's own spec", () => {
    // The real artefact's defaults: 72 / 6 / 16 / 151 / 3, and the runway count
    // is the artefact's class list, not the airport's runway table. Two of them
    // are TABLE LENGTHS — there is no bin width to divide by.
    expect(trainingWordCounts(MOCK_VOCABULARY)).toEqual({
      heading: 72,
      vertical: 6,
      speed: 16,
      runway: 4,
      duration: 151,
      terminal: 3,
    });
  });

  it("counts the vertical and speed words off their tables, not off a step", () => {
    const narrower = {
      ...MOCK_VOCABULARY,
      verticalModesDeg: [0, 3.0],
      speedCentresMps: [60, 70, 80],
    };
    expect(trainingWordCounts(narrower).vertical).toBe(2);
    expect(trainingWordCounts(narrower).speed).toBe(3);
  });

  it("says go-around is a class that exists but is never observed here", () => {
    // The 25 km arrival slice keeps only the final successful approach, so the
    // class cannot fire. A legend must not read this as the model declining it.
    expect(TERMINAL_NEVER_OBSERVED).toContain(TERMINAL_GO_AROUND);
    expect(TERMINAL_NEVER_OBSERVED).not.toContain(TERMINAL_LANDED);
  });
});

describe("parseTrainingSample", () => {
  it("accepts the mock sample and keeps the event sequence intact", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const flight = parsed.value.flights[0];
    expect(flight.sentence.eventTimesS).toEqual([0, 26, 70, 108, 130, 188, 222]);
    expect(flight.sentence.words).toHaveLength(flight.sentence.eventTimesS.length);
    expect(parsed.value.vocabulary.runwayIdents).toEqual(["05L", "05R", "23L", "23R"]);
  });

  it("reads the last event as landed", () => {
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const words = parsed.value.flights[0].sentence.words;
    expect(words[words.length - 1][TRAINING_KIND_COLUMN.terminal]).toBe(TERMINAL_LANDED);
  });

  it("refuses a schema it does not know, by name", () => {
    const parsed = sampleWith((s) => {
      s.schema = "aeroviz-training-sample-v0";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("aeroviz-training-sample-v0");
    expect(parsed.problem).toContain("aeroviz-training-sample-v1");
  });

  // The event sequence's two invariants. Neither could be violated under the even
  // 10 s grid this replaced, so neither was checked before.
  it("refuses a sentence whose word rows and event times disagree in length", () => {
    const parsed = sampleWith((s) => {
      s.flights[0].sentence.words.pop();
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("7 event times but 6 word rows");
  });

  it("refuses event times that do not strictly increase", () => {
    const parsed = sampleWith((s) => {
      s.flights[0].sentence.eventTimesS[3] = s.flights[0].sentence.eventTimesS[2];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("strictly increasing");
    expect(parsed.problem).toContain("index 3");
  });

  it("refuses a word row that is not six columns, and says how many it found", () => {
    const parsed = sampleWith((s) => {
      s.flights[0].sentence.words[2] = [9, 10, 21, 0, 0]; // the retired five-column shape
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("has 5 columns, expected 6");
    expect(parsed.problem).toContain("heading, vertical, speed, runway, duration, terminal");
  });

  // The loudest symptom of a column-order mistake: a duration word (0–150) landing
  // in the runway column (0–3). This is why the range check exists.
  it("catches swapped columns as an out-of-range word and points at the order", () => {
    const parsed = sampleWith((s) => {
      const row = s.flights[0].sentence.words[1];
      [row[TRAINING_KIND_COLUMN.runway], row[TRAINING_KIND_COLUMN.duration]] = [
        row[TRAINING_KIND_COLUMN.duration],
        row[TRAINING_KIND_COLUMN.runway],
      ];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("runway is 13");
    expect(parsed.problem).toContain("0…3");
    expect(parsed.problem).toContain("ORDER");
  });

  it("refuses a flight whose runway is not one of the vocabulary's classes", () => {
    // KRDU's airport table has 32; this vocabulary does not. Reading the airport's
    // runway list instead of the artefact's is the mistake this guards.
    const parsed = sampleWith((s) => {
      s.flights[0].runway = "32";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("runway 32 is not one of this vocabulary's classes");
    expect(parsed.problem).toContain("05L, 05R, 23L, 23R");
  });

  it("refuses a vocabulary missing the runway sha, which is a field of its own", () => {
    const parsed = sampleWith((s) => {
      delete s.vocabulary.runwaySha256;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("vocabulary.runwaySha256");
  });

  it("refuses a restated kinds list that disagrees with the column order", () => {
    const parsed = sampleWith((s) => {
      // An artefact read under the retired rule: same six columns, second one
      // spelled `altitude`. This is the refusal that keeps a superseded export
      // off the screen (V38), and it is the FIRST thing a stale file trips.
      s.kinds = ["heading", "altitude", "speed", "runway", "duration", "terminal"];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("expected");
  });

  it("refuses a missing durationClamped rather than assuming zero", () => {
    // How many gaps hit the 300 s ceiling is stated, never silent.
    const parsed = sampleWith((s) => {
      delete s.flights[0].sentence.durationClamped;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("durationClamped");
  });
});

describe("parseTrainingIndex", () => {
  it("accepts the mock manifest", () => {
    const parsed = parseTrainingIndex(mockIndex());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.sets).toHaveLength(1);
    expect(parsed.value.rejected).toEqual([]);
    expect(parsed.value.sets[0].runwaySha256).not.toBe(parsed.value.sets[0].vocabularySha256);
  });

  // The deliberate divergence from AV6: one bad entry must not empty the list.
  it("rejects only the bad set and names the field, keeping the good ones", () => {
    const index = mockIndex() as any;
    index.sets.push({ ...index.sets[0], id: "broken_set", kind: "not-a-kind" });
    index.sets.push({ ...index.sets[0], id: "later_good_set" });

    const parsed = parseTrainingIndex(index);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    expect(parsed.value.sets.map((s) => s.id)).toEqual(["vocabulary_tau10", "later_good_set"]);
    expect(parsed.value.rejected).toHaveLength(1);
    expect(parsed.value.rejected[0].id).toBe("broken_set");
    expect(parsed.value.rejected[0].problem).toContain("kind");
    expect(parsed.value.rejected[0].problem).toContain("not-a-kind");
  });

  it("names the missing field when a set omits the runway sha", () => {
    const index = mockIndex() as any;
    delete index.sets[0].runwaySha256;
    const parsed = parseTrainingIndex(index);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.sets).toEqual([]);
    expect(parsed.value.rejected[0].problem).toContain("runwaySha256");
  });

  it("fails the whole call only when the manifest is not a manifest", () => {
    expect(parseTrainingIndex({ schema: "something-else", sets: [] }).ok).toBe(false);
    expect(parseTrainingIndex([]).ok).toBe(false);
    expect(parseTrainingIndex(null).ok).toBe(false);
  });
});

describe("how a time is written", () => {
  // The sentence bar and the read-back window show the SAME cursor. When each
  // had its own formatter they printed 201.9 s and 202 s for one moment, which
  // reads as two cursors.
  it("is one definition for every Training view", () => {
    expect(formatSeconds(70)).toBe("70");
    expect(formatSeconds(201.9)).toBe("201.9");
    expect(formatSeconds(201.94)).toBe("201.9");
    expect(formatSeconds(0)).toBe("0");
  });
});

describe("the observed track", () => {
  // The axis the vertical word was READ on: the word is the slope of height
  // against this, so a chart that plotted it against anything else would be
  // judging the word in coordinates it was never fitted in.
  it("carries the cumulative horizontal distance, non-decreasing from 0", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const observed = parsed.value.flights[0].observed;
    expect(observed.pathM).toHaveLength(observed.tS.length);
    expect(observed.pathM[0]).toBe(0);
    for (let row = 1; row < observed.pathM.length; row += 1) {
      expect(observed.pathM[row]).toBeGreaterThan(observed.pathM[row - 1]);
    }
  });

  it("reads every column the charts plot, one value per row", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const observed = parsed.value.flights[0].observed;
    for (const column of TRAINING_OBSERVED_COLUMNS) {
      expect(observed[column]).toHaveLength(observed.tS.length);
    }
    expect(observed.tS[0]).toBe(0);
    expect(observed.tS[observed.tS.length - 1]).toBe(parsed.value.flights[0].durationS);
  });

  // The track's clock and the flight's length come from the same rows in the
  // export, so a disagreement means they are not the same flight.
  it("refuses a track that does not run the length of the flight", () => {
    const short = sampleWith((sample) => {
      sample.flights[0].observed.tS = sample.flights[0].observed.tS.slice(0, -1);
    });
    expect(short.ok).toBe(false);
    if (short.ok) return;
    expect(short.problem).toContain("but the flight is 262 s long");
  });

  it("refuses a column that is short a row, by name", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].observed.heightM = sample.flights[0].observed.heightM.slice(0, -1);
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("observed.heightM");
  });

  it("refuses an established column that is not 0/1", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].observed.established[3] = 2;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("established");
  });
});

describe("the flown sentence", () => {
  it("reads its columns, its ending and how much of the approach was compared", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const flown = parsed.value.flights[0].geometric;
    for (const column of TRAINING_GEOMETRIC_COLUMNS) {
      expect(flown[column]).toHaveLength(flown.tS.length);
    }
    expect(flown.endReason).toBe("crossed-threshold");
    expect(flown.comparedFraction).toBeLessThanOrEqual(1);
  });

  // The flown track runs on its own clock and may outlast or fall short of the
  // observation, so it is NOT checked against `durationS` the way the observed
  // track is. Its own clock IS checked, because `rowAt`, the x axis and the gap
  // readout all assume it runs forward from 0.
  it("refuses a flown clock that does not run forward from 0", () => {
    const late = sampleWith((sample) => {
      sample.flights[0].geometric.tS[0] = 1;
    });
    expect(late.ok).toBe(false);
    if (late.ok) return;
    expect(late.problem).toContain("starts at 1 s, not 0");

    const backwards = sampleWith((sample) => {
      sample.flights[0].geometric.tS[5] = sample.flights[0].geometric.tS[4];
    });
    expect(backwards.ok).toBe(false);
    if (backwards.ok) return;
    expect(backwards.problem).toContain("not increasing");
  });

  // The assumptions the line was drawn under travel WITH it and are shown, so a
  // sample that does not state them is not a sample this view can draw.
  it("refuses a sample that does not say what its flown tracks assume", () => {
    const parsed = sampleWith((sample) => {
      delete sample.geometry;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("geometry is missing");

    const partial = sampleWith((sample) => {
      delete sample.geometry.windModelled;
    });
    expect(partial.ok).toBe(false);
    if (partial.ok) return;
    expect(partial.problem).toContain("geometry.windModelled");
  });

  it("refuses an ending it does not know", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.endReason = "ran-out-of-fuel";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("endReason");
    expect(parsed.problem).toContain("crossed-threshold");
  });

  it("refuses a column that does not match its own clock", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.crossM.pop();
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("geometric.crossM");
  });
});

describe("the corridor a sentence draws", () => {
  // The vertical band is two HEIGHT columns on the nominal track's own rows, and
  // that is only legal because the commanded angle never touches the horizontal
  // step. If it ever stops being aligned, the fan gets drawn against the wrong x
  // values — a corridor somewhere else entirely — so the length is checked.
  it("reads the vertical band on the nominal track's own rows", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const flown = parsed.value.flights[0].geometric;
    for (const column of TRAINING_VERTICAL_BAND_COLUMNS) {
      expect(flown.verticalBand[column]).toHaveLength(flown.tS.length);
    }
    // It opens with distance: wider at the far end than at the start.
    const last = flown.tS.length - 1;
    const openingWidth = flown.verticalBand.heightLoM[0] - flown.verticalBand.heightHiM[0];
    expect(flown.verticalBand.heightLoM[last] - flown.verticalBand.heightHiM[last])
      .toBeGreaterThan(openingWidth);
  });

  it("refuses a vertical band that is not aligned with the track it bands", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.verticalBand.heightHiM.pop();
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("verticalBand.heightHiM");
    expect(parsed.problem).toContain("rows");
  });

  // Swapping the edges keeps the band exactly as wide, so nothing downstream
  // would look wrong — it would just be drawn inside out.
  it("refuses a vertical band whose edges are the wrong way round", () => {
    const parsed = sampleWith((sample) => {
      const band = sample.flights[0].geometric.verticalBand;
      [band.heightLoM, band.heightHiM] = [band.heightHiM, band.heightLoM];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("inverted");
    expect(parsed.problem).toContain("SHALLOWER");
  });

  it("refuses a missing band rather than drawing a line and calling it the sentence", () => {
    for (const band of ["verticalBand", "speedBand"]) {
      const parsed = sampleWith((sample) => {
        delete sample.flights[0].geometric[band];
      });
      expect(parsed.ok).toBe(false);
      if (parsed.ok) return;
      expect(parsed.problem).toContain(band);
    }
  });

  // The speed edges are tracks of their own — a speed change moves the
  // horizontal step, the turn radius and the moment of crossing — so they have
  // their own clocks and their own endings.
  it("reads each speed edge as its own track, and the window it opens", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const band = parsed.value.flights[0].geometric.speedBand;
    for (const edge of [band.low, band.high]) {
      for (const column of TRAINING_SPEED_EDGE_COLUMNS) {
        expect(edge[column]).toHaveLength(edge.tS.length);
      }
      expect(edge.endS).toBeCloseTo(edge.tS[edge.tS.length - 1], 6);
    }
    // The fast edge covers the same ground sooner, so it arrives first.
    expect(band.high.endS).toBeLessThan(band.low.endS);
    expect(band.arrivalWindowS).toEqual([band.high.endS, band.low.endS]);
  });

  // A window whose far end is the integration budget is not an arrival window,
  // it is the stopping rule — printing it as "arrives between" would measure
  // this module instead of the vocabulary.
  it("refuses an arrival window when an edge never reached the runway", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.speedBand.low.endReason = "time-cap";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("arrivalWindowS");
    expect(parsed.problem).toContain("time-cap");
  });

  it("accepts a null window, which is how an edge that stopped short says so", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.speedBand.low.endReason = "time-cap";
      sample.flights[0].geometric.speedBand.arrivalWindowS = null;
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.flights[0].geometric.speedBand.arrivalWindowS).toBeNull();
    expect(parsed.value.flights[0].geometric.speedBand.low.endReason).toBe("time-cap");
  });

  it("refuses an edge whose endS disagrees with its own last step", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.speedBand.high.endS += 10;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("speedBand.high.endS");
  });

  // The `geometry` block is what lets a reader say WHERE the corridor came from.
  // `bandsAreJoint` is the one that stops it being read as the whole envelope.
  it("carries what each band was flown from, and that they are not joint", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.geometry.verticalBandFrom).toContain("vertical");
    expect(parsed.value.geometry.speedBandFrom).toContain("speed");
    expect(parsed.value.geometry.bandsAreJoint).toBe(false);
    // And the two assumptions that replaced the height time constant.
    expect(parsed.value.geometry.verticalIsCommandedAngle).toBe(true);
    expect(parsed.value.geometry.heightFloorM).toBe(0);
  });

  it("refuses a geometry block that does not say where a band came from", () => {
    for (const field of ["verticalBandFrom", "speedBandFrom", "bandsAreJoint", "heightFloorM"]) {
      const parsed = sampleWith((sample) => {
        delete sample.geometry[field];
      });
      expect(parsed.ok).toBe(false);
      if (parsed.ok) return;
      expect(parsed.problem).toContain(field);
    }
  });
});

describe("the vocabulary's two tables", () => {
  it("refuses centres that are not stored sorted and distinct", () => {
    const unsorted = sampleWith((sample) => {
      sample.vocabulary.speedCentresMps = [44, 63, 56, 68];
      sample.vocabulary.words.speed = 4;
    });
    expect(unsorted.ok).toBe(false);
    if (unsorted.ok) return;
    expect(unsorted.problem).toContain("sorted and distinct");
  });

  // The level mode is the branch `verticalToleranceDeg` turns on. Without it
  // every vertical word would quietly take the fraction branch and still return
  // a number, so nothing downstream would look wrong.
  it("refuses a vertical table with no level mode", () => {
    const parsed = sampleWith((sample) => {
      sample.vocabulary.verticalModesDeg = [-3.0, 1.4, 2.4, 3.1, 4.4];
      sample.vocabulary.words.vertical = 5;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("level mode");
  });

  // A tolerance of zero is a band nothing can sit inside: it turns every word
  // into a point target no executor can meet, and the "inside the band" readout
  // would report 0 % for a flight that flew the sentence exactly.
  it("refuses a tolerance of zero", () => {
    for (const field of ["verticalLevelToleranceDeg", "verticalToleranceFraction", "speedToleranceFraction"]) {
      const parsed = sampleWith((sample) => {
        sample.vocabulary[field] = 0;
      });
      expect(parsed.ok).toBe(false);
      if (parsed.ok) return;
      expect(parsed.problem).toContain(field);
    }
  });
});

describe("paths", () => {
  it("builds the manifest and sample paths from the airport code", () => {
    expect(trainingIndexPath("KRDU")).toBe("data/airports/KRDU/training/index.json");
    expect(trainingSamplePath("KRDU", "vocabulary_tau10/sample.json")).toBe(
      "data/airports/KRDU/training/vocabulary_tau10/sample.json",
    );
  });
});

// ── T4a: the fields the sentence bar draws beside the words ──────────────────

describe("the track length, the instructions and the absorbed manoeuvres", () => {
  it("reads all three off a good sample", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const flight = parsed.value.flights[0];
    expect(flight.durationS).toBe(262);
    expect(flight.establishedFromStart).toBe(false);
    expect(flight.instructions).toHaveLength(12);
    expect(flight.absorbed.map((item) => item.reason)).toEqual(["short tail", "small change"]);
  });

  // The track outlives the sentence by a median 145 s in the real export, so a
  // view that stopped at the last event would leave 44 % of the approach blank.
  it("keeps a track that runs on past the last event", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const flight = parsed.value.flights[0];
    const lastEventS = flight.sentence.eventTimesS[flight.sentence.eventTimesS.length - 1];
    expect(lastEventS).toBeLessThan(flight.durationS);
  });

  // The other way round is a different flight's sentence stapled to this track.
  it("refuses a sentence whose last event is past the end of the track", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].durationS = 100;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("not the same flight");
  });

  it("refuses a missing durationS by name", () => {
    const parsed = sampleWith((sample) => {
      delete sample.flights[0].durationS;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("durationS");
  });

  // null means "it never settled inside the track" — a real answer. An absent key
  // means the field moved, and reading that as null would invent the answer.
  it("keeps a null settledS but refuses an absent one", () => {
    const parsed = parseTrainingSample(mockSample());
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const instructions = parsed.value.flights[0].instructions;
    expect(instructions[instructions.length - 1].settledS).toBeNull();

    const absent = sampleWith((sample) => {
      delete sample.flights[0].instructions[0].settledS;
    });
    expect(absent.ok).toBe(false);
    if (absent.ok) return;
    expect(absent.problem).toContain("settledS");
  });

  // An empty span does not draw, does not warn, and leaves its own rows out of
  // the "inside the band" denominator — so a corrupt export reads as a BETTER
  // result than a good one. Measured on the fixture: 0.75 inside becomes 1.00.
  it("refuses an instruction that settles before it was issued", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].instructions[9].settledS = 100; // the vertical one issued at 130
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("settles at 100 s but was issued at 130 s");
  });

  it("refuses an instruction word outside its kind's range", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].instructions[0].word = 72; // heading has 72 classes: 0…35
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("heading");
  });

  it("refuses an absorbed manoeuvre with an unlisted reason", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].absorbed[0].reason = "because";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("reason");
    expect(parsed.problem).toContain("same word");
  });

  it("refuses an absorbed span that ends before it starts", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].absorbed[0].endS = 1;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("before it starts");
  });
});

describe("the fields a lenient reader would have defaulted", () => {
  // These two used to read `str(entry, "callsign") ?? flightKey`. The exporter
  // always writes both, so the default could only ever fire on a broken export —
  // and would have printed "unknown" beside the real strata as though the file
  // said so.
  it("refuses a flight with no callsign or no stratum, by name", () => {
    for (const field of ["callsign", "stratum", "runway"]) {
      const parsed = sampleWith((sample) => {
        delete sample.flights[0][field];
      });
      expect(parsed.ok).toBe(false);
      if (parsed.ok) return;
      expect(parsed.problem).toContain(field);
    }
  });

  // Every export writes `kinds`. A file without it is a file from something else.
  it("refuses a sample with no kinds at all", () => {
    const parsed = sampleWith((sample) => {
      delete sample.kinds;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("kinds");
  });

  // The artefact states its class counts; we derive them from the bins. The
  // check is what makes the derivation a mirror instead of a second opinion.
  it("refuses a vocabulary whose stated counts disagree with its own bins", () => {
    const parsed = sampleWith((sample) => {
      sample.vocabulary.words.speed = 15; // the table has 16 centres
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("words.speed is 15");
    expect(parsed.problem).toContain("16");
  });

  it("refuses a vocabulary that states no counts", () => {
    const parsed = sampleWith((sample) => {
      delete sample.vocabulary.words;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("vocabulary.words");
  });

  // Same reason as the last-event check: a time outside the track belongs to
  // another flight, and it would draw its mark off the plot instead of failing.
  it("refuses an instruction or an absorbed span outside the track", () => {
    const late = sampleWith((sample) => {
      // both, so this is about the TRACK's length and not about the pair's order
      sample.flights[0].instructions[0].issuedS = 400; // the track is 262 s
      sample.flights[0].instructions[0].settledS = 400;
    });
    expect(late.ok).toBe(false);
    if (late.ok) return;
    expect(late.problem).toContain("outside the 262 s track");

    const spill = sampleWith((sample) => {
      sample.flights[0].absorbed[1].endS = 400;
    });
    expect(spill.ok).toBe(false);
    if (spill.ok) return;
    expect(spill.problem).toContain("outside the 262 s track");
  });
});

describe("what a word means", () => {
  // MIRRORS of Vocabulary.*_centre_*: the same arithmetic, so a re-binned
  // vocabulary reads correctly without a second table anywhere.
  it("reads the bin centres the labeller wrote", () => {
    expect(headingCentreDeg(MOCK_VOCABULARY, 10)).toBeCloseTo(50, 9);
    // Descent is POSITIVE and the climb mode is the negative one. Pinned by
    // value, because a table read off by one would still return an angle.
    expect(verticalCentreDeg(MOCK_VOCABULARY, LEVEL)).toBe(0);
    expect(verticalCentreDeg(MOCK_VOCABULARY, DESCEND_31)).toBeCloseTo(3.1, 9);
    expect(verticalCentreDeg(MOCK_VOCABULARY, 0)).toBeCloseTo(-3.0, 9);
    expect(speedCentreMps(MOCK_VOCABULARY, 0)).toBeCloseTo(44, 9);
    expect(speedCentreMps(MOCK_VOCABULARY, 11)).toBeCloseTo(121, 9);
    expect(durationCentreS(MOCK_VOCABULARY, 13)).toBe(26);
  });

  // MIRROR of `Vocabulary.vertical_tolerance_deg` / `speed_tolerance`, the level
  // mode's branch included. That branch is the one worth pinning: a percentage
  // of zero is zero, so a reader that lost it would give level flight a band of
  // no width at all and report every level segment as disobeyed.
  it("gives the level mode an absolute tolerance and every other one a fraction", () => {
    expect(verticalToleranceDeg(MOCK_VOCABULARY, LEVEL)).toBeCloseTo(0.1, 9);
    expect(verticalToleranceDeg(MOCK_VOCABULARY, DESCEND_31)).toBeCloseTo(3.1 * 0.07, 9);
    // The climb mode's is a fraction of its MAGNITUDE, not a negative tolerance.
    expect(verticalToleranceDeg(MOCK_VOCABULARY, 0)).toBeCloseTo(3.0 * 0.07, 9);
    expect(speedToleranceMps(MOCK_VOCABULARY, 11)).toBeCloseTo(121 * 0.03, 9);
  });

  // The kinds with no redundancy return null rather than 0: a view that drew a
  // zero-width band would be drawing a tolerance the vocabulary does not have,
  // and one that drew any band at all would be inventing the number (V36).
  it("says which kinds have no tolerance at all", () => {
    expect(trainingWordTolerance(MOCK_VOCABULARY, "vertical", DESCEND_31)).toBeCloseTo(0.217, 6);
    expect(trainingWordTolerance(MOCK_VOCABULARY, "speed", 11)).toBeCloseTo(3.63, 6);
    for (const kind of ["heading", "runway", "duration", "terminal"] as const) {
      expect(trainingWordTolerance(MOCK_VOCABULARY, kind, 0)).toBeNull();
    }
  });

  // wrap_deg is half-open [-180, 180), so the one word at the wrap reads -180.
  // The NEGATIVE cases are the ones that catch the single-modulo spelling: JS `%`
  // truncates where Python's floors, so `((d+180) % 360) - 180` returns -190 for
  // -190. No word reaches it today; the artefact's signed relative course does.
  it("wraps the same way the labeller does, negatives included", () => {
    expect(headingCentreDeg(MOCK_VOCABULARY, 36)).toBe(-180);
    expect(headingCentreDeg(MOCK_VOCABULARY, 60)).toBeCloseTo(-60, 9);
    expect(wrapDeg(190)).toBe(-170);
    expect(wrapDeg(-190)).toBe(170);
    expect(wrapDeg(-360)).toBe(0);
    expect(wrapDeg(-180)).toBe(-180);
    expect(wrapDeg(540)).toBe(-180);
  });

  // The vocabulary is DEFINED in SI now — the speed centres were fitted in m/s
  // and rounded to 1 m/s — so printing knots would label them 85.6 and 108.8 and
  // hide the vocabulary's own grid (V30).
  it("labels speed in m/s and the vertical word as an angle with an arrow", () => {
    expect(trainingWordLabel(MOCK_VOCABULARY, "speed", 11)).toBe("121 m/s");
    expect(trainingWordLabel(MOCK_VOCABULARY, "heading", 10)).toBe("+50°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "heading", 0)).toBe("0°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "duration", 13)).toBe("26 s");
  });

  // DESCENT IS POSITIVE in this vocabulary, so a bare signed number reads
  // backwards to everyone who has not read the spec. The arrow carries the sign
  // and the printed number is always a magnitude (V37).
  it("never prints a signed vertical angle: the arrow says which way", () => {
    expect(trainingWordLabel(MOCK_VOCABULARY, "vertical", DESCEND_31)).toBe("↓3.1°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "vertical", LEVEL)).toBe("level");
    expect(trainingWordLabel(MOCK_VOCABULARY, "vertical", 0)).toBe("↑3.0°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "vertical", 0)).not.toContain("-");
  });

  // What the sentence bar writes on a band: the word is half the meaning and the
  // band it allows is the other half.
  it("writes the band beside the word, and nothing for the kinds without one", () => {
    expect(trainingWordBandLabel(MOCK_VOCABULARY, "vertical", DESCEND_31)).toBe("↓3.1°±0.22");
    expect(trainingWordBandLabel(MOCK_VOCABULARY, "vertical", LEVEL)).toBe("level±0.10");
    expect(trainingWordBandLabel(MOCK_VOCABULARY, "speed", 11)).toBe("121 m/s±3.6");
    expect(trainingWordBandLabel(MOCK_VOCABULARY, "heading", 10)).toBe("+50°");
  });

  // The runway classes come from the file, never from the airport's runway list.
  it("labels a runway word from the vocabulary's own classes", () => {
    expect(trainingWordLabel(MOCK_VOCABULARY, "runway", 0)).toBe("05L");
    expect(trainingWordLabel(MOCK_VOCABULARY, "runway", 3)).toBe("23R");
  });

  it("names the terminal classes, go-around included", () => {
    expect(trainingWordLabel(MOCK_VOCABULARY, "terminal", TERMINAL_LANDED)).toBe("landed");
    expect(trainingWordLabel(MOCK_VOCABULARY, "terminal", TERMINAL_GO_AROUND)).toBe("go-around");
  });
});

// ── the boundaries the review found unpinned ─────────────────────────────────

describe("the arrival window against its own edges", () => {
  // The window is DERIVED from the two edges. Checking only one direction left
  // the half a reader sees: a null window beside two landed edges rendered as
  // "no window: an edge ended on crossed-threshold" — a reason that is not one.
  it("refuses a null window when both edges crossed", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].geometric.speedBand.arrivalWindowS = null;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("both edges crossed");
  });

  // Checking the VALUES, not just their order, is what makes the field a mirror
  // of the edges — and it is the only thing that catches low/high swapped.
  it("refuses a window that is not the two edges' own crossings", () => {
    const parsed = sampleWith((sample) => {
      const band = sample.flights[0].geometric.speedBand;
      band.arrivalWindowS = [band.arrivalWindowS[0] - 1, band.arrivalWindowS[1]];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("the window is the two crossings, fast first");
  });

  it("refuses edges whose fast one is not the faster", () => {
    const parsed = sampleWith((sample) => {
      const band = sample.flights[0].geometric.speedBand;
      [band.low, band.high] = [band.high, band.low];
      band.arrivalWindowS = [band.high.endS, band.low.endS];
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("these two edges are swapped");
  });
});

describe("the refusals a lenient reader would not have made", () => {
  it("refuses a negative distance or a negative compared span", () => {
    for (const field of ["finalGapM", "meanGapM", "gapP95M", "comparedS", "comparedFraction"]) {
      const parsed = sampleWith((sample) => {
        sample.flights[0].geometric[field] = -1;
      });
      expect(parsed.ok).toBe(false);
      if (parsed.ok) return;
      expect(parsed.problem).toContain(field);
    }
  });

  // At the boundary, not past it: a word equal to the class count indexes one
  // off the end of every legend, and `runwayIdents[4]` renders `undefined`.
  it("refuses a sentence word exactly equal to its kind's class count", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].sentence.words[0][TRAINING_KIND_COLUMN.runway] = 4; // classes are 0…3
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("outside this vocabulary's 0…3");
  });

  // MIRROR of `Vocabulary.__post_init__`. `trainingWordCounts` ROUNDS 360/bin,
  // so a 7° bin would give 51 classes and read every heading near the wrap into
  // the wrong one.
  it("refuses a heading bin that does not divide the circle", () => {
    const parsed = sampleWith((sample) => {
      sample.vocabulary.headingBinDeg = 7;
      sample.vocabulary.words.heading = 51;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("does not divide 360");
  });

  it("refuses a fractional segment count", () => {
    const parsed = sampleWith((sample) => {
      sample.vocabulary.verticalSegments = 5.5;
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("not a count of segments");
  });
});

describe("the reading rule this reader is written for", () => {
  // The file states its own bins, centres and tolerances, so most of what a word
  // means is read from it. What is NOT in the file is that the second column is
  // an ANGLE and that descent is positive — this reader hardcodes both. So it is
  // bound to the rule, and refuses another one by name rather than reading an
  // older artefact into today's meanings.
  it("refuses an artefact read under another rule, naming both", () => {
    const parsed = sampleWith((sample) => {
      sample.vocabulary.readingRule = "segment-v13";
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("segment-v13");
    expect(parsed.problem).toContain(TRAINING_READING_RULE);
  });

  it("is the rule the fixture carries, so the mirror is checked both ways", () => {
    expect(TRAINING_READING_RULE).toBe(MOCK_VOCABULARY.readingRule);
  });
});
