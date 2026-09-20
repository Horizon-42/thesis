import { describe, expect, it } from "vitest";

import {
  TERMINAL_GO_AROUND,
  TERMINAL_LANDED,
  TERMINAL_NEVER_OBSERVED,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  TRAINING_WORD_COLUMNS,
  parseTrainingIndex,
  parseTrainingSample,
  trainingIndexPath,
  trainingSamplePath,
  trainingWordCounts,
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

describe("paths", () => {
  it("builds the manifest and sample paths from the airport code", () => {
    expect(trainingIndexPath("KRDU")).toBe("data/airports/KRDU/training/index.json");
    expect(trainingSamplePath("KRDU", "vocabulary_tau10/sample.json")).toBe(
      "data/airports/KRDU/training/vocabulary_tau10/sample.json",
    );
  });
});
