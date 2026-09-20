import { describe, expect, it } from "vitest";

import {
  TERMINAL_GO_AROUND,
  TERMINAL_LANDED,
  TERMINAL_NEVER_OBSERVED,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  TRAINING_OBSERVED_COLUMNS,
  TRAINING_WORD_COLUMNS,
  formatSeconds,
  parseTrainingIndex,
  parseTrainingSample,
  trainingIndexPath,
  trainingSamplePath,
  trainingWordCounts,
  trainingWordLabel,
  altitudeCentreM,
  durationCentreS,
  headingCentreDeg,
  speedCentreMps,
  wrapDeg,
} from "../trainingSample";
import { MOCK_VOCABULARY, mockIndex, mockSample } from "./trainingSample.fixture";

function sampleWith(mutate: (sample: any) => void) {
  const sample = mockSample();
  mutate(sample);
  return parseTrainingSample(sample);
}

describe("the six word kinds", () => {
  // MIRROR of instructions.INSTRUCTION_KINDS. The columns are positional, so this
  // order is the contract; the intercept word was deleted on 2026-09-20 (D73).
  it("is heading, altitude, speed, runway, duration, terminal — in that order", () => {
    expect([...TRAINING_KINDS]).toEqual([
      "heading",
      "altitude",
      "speed",
      "runway",
      "duration",
      "terminal",
    ]);
    expect(TRAINING_WORD_COLUMNS).toBe(6);
    expect(TRAINING_KINDS).not.toContain("intercept");
  });

  it("derives each kind's class count from the file's own spec", () => {
    // The real artefact's defaults: 36 / 11 / 23 / 151 / 3, and the runway count
    // is the artefact's class list, not the airport's runway table.
    expect(trainingWordCounts(MOCK_VOCABULARY)).toEqual({
      heading: 36,
      altitude: 11,
      speed: 23,
      runway: 4,
      duration: 151,
      terminal: 3,
    });
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
    expect(parsed.problem).toContain("heading, altitude, speed, runway, duration, terminal");
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
      s.kinds = ["heading", "altitude", "speed", "intercept", "runway", "duration"];
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
    expect(flight.instructions).toHaveLength(13);
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

  it("refuses an instruction word outside its kind's range", () => {
    const parsed = sampleWith((sample) => {
      sample.flights[0].instructions[0].word = 36; // heading has 36 classes: 0…35
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
      sample.vocabulary.words.speed = 22; // the bins give 23
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.problem).toContain("words.speed is 22");
    expect(parsed.problem).toContain("23");
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
      sample.flights[0].instructions[0].issuedS = 400; // the track is 262 s
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
    expect(headingCentreDeg(MOCK_VOCABULARY, 5)).toBeCloseTo(50, 9);
    expect(altitudeCentreM(MOCK_VOCABULARY, 10)).toBeCloseTo(3048, 9);
    expect(speedCentreMps(MOCK_VOCABULARY, 0)).toBeCloseTo(MOCK_VOCABULARY.speedMinMps, 9);
    expect(durationCentreS(MOCK_VOCABULARY, 13)).toBe(26);
  });

  // wrap_deg is half-open [-180, 180), so the one word at the wrap reads -180.
  // The NEGATIVE cases are the ones that catch the single-modulo spelling: JS `%`
  // truncates where Python's floors, so `((d+180) % 360) - 180` returns -190 for
  // -190. No word reaches it today; the artefact's signed relative course does.
  it("wraps the same way the labeller does, negatives included", () => {
    expect(headingCentreDeg(MOCK_VOCABULARY, 18)).toBe(-180);
    expect(headingCentreDeg(MOCK_VOCABULARY, 30)).toBeCloseTo(-60, 9);
    expect(wrapDeg(190)).toBe(-170);
    expect(wrapDeg(-190)).toBe(170);
    expect(wrapDeg(-360)).toBe(0);
    expect(wrapDeg(-180)).toBe(-180);
    expect(wrapDeg(540)).toBe(-180);
  });

  // The bins are DEFINED in feet and knots; printing the stored SI would make a
  // 1000 ft step read as 304.8 m and hide the vocabulary's own grid.
  it("labels altitude in feet and speed in knots, off the generated constants", () => {
    expect(trainingWordLabel(MOCK_VOCABULARY, "altitude", 10)).toBe("10000 ft");
    expect(trainingWordLabel(MOCK_VOCABULARY, "altitude", 3)).toBe("3000 ft");
    expect(trainingWordLabel(MOCK_VOCABULARY, "speed", 21)).toBe("310 kt");
    expect(trainingWordLabel(MOCK_VOCABULARY, "heading", 5)).toBe("+50°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "heading", 0)).toBe("0°");
    expect(trainingWordLabel(MOCK_VOCABULARY, "duration", 13)).toBe("26 s");
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
