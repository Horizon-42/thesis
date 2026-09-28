/**
 * The Training views' shared wording of the executor's replay: what went wrong, in the fewest words, and which words it
 * flew outside their envelopes — adding up to the gate's count, the one word it counts twice included.
 */
import { describe, expect, it } from "vitest";

import { parseTrainingSample, type TrainingSample } from "../trainingSample";
import { parseTrainingExecutorOverlay, type TrainingExecutorFlown } from "../trainingOverlays";
import { formatElapsed, replayIssueText, replayOutsideWords } from "../trainingText";
import { mockSample } from "./trainingSample.fixture";
import { EXECUTOR_ID, mockExecutorOverlay, mockOverlayEntry } from "./trainingOverlays.fixture";

function read(): { sample: TrainingSample; flown: TrainingExecutorFlown } {
  const sample = parseTrainingSample(mockSample());
  if (!sample.ok) throw new Error(sample.problem);
  const parsed = parseTrainingExecutorOverlay(mockExecutorOverlay(), mockOverlayEntry(EXECUTOR_ID), sample.value);
  if (!parsed.ok) throw new Error(parsed.problem);
  const flown = parsed.value.flights[0];
  if (!flown.flown) throw new Error("the fixture's first flight is not flown");
  return { sample: sample.value, flown };
}

describe("the replay's issue in words", () => {
  it("says the outcome when it did not land, a refused track, and the words out — nothing for a clean landing", () => {
    const { flown } = read();
    expect(replayIssueText(flown)).toBe("2 words out");
    expect(replayIssueText({ ...flown, counts: { ...flown.counts, wordsInside: 6 } })).toBe("1 word out");
    const clean = { ...flown, counts: { ...flown.counts, wordsInside: 7 } };
    expect(replayIssueText(clean)).toBeNull();
    expect(replayIssueText({ ...flown, outcome: "timeout" })).toBe("timed out · 2 words out");
    expect(replayIssueText({ ...clean, outcome: "crossed_too_high" })).toBe("too high");
    expect(replayIssueText({ ...flown, refused: "the flown track is too short to judge",
      counts: { ...flown.counts, wordsJudged: 0, wordsInside: 0 } })).toBe("track refused");
  });

  it("names the words it flew outside, the one the gate counts twice said to be so when both its checks failed", () => {
    const { sample, flown } = read();
    const name = (flight: TrainingExecutorFlown) => replayOutsideWords(flight, sample.vocabulary, sample.candidates);
    expect(name(flown)).toEqual(["heading 180° at step 10", "approach cleared at step 20"]);
    // the heading word left to intercept the final: its band's check and the intercept's
    const intercept = { name: "intercepts the final on its own", ok: false, inside: null, rows: null };
    const withIntercept = (bandOk: boolean) => ({ ...flown, words: flown.words.map((word) => (word.row === 10 && word.column === 2
      ? { ...word, checks: [{ ...word.checks[0], ok: bandOk }, intercept] } : word)) });
    expect(name(withIntercept(false))[0]).toBe("heading 180° at step 10 (counted twice: its band and the intercept)");
    expect(name(withIntercept(true))[0]).toBe("heading 180° at step 10");
  });
});

describe("the times written out", () => {
  it("chooses the unit after rounding", () => {
    expect(formatElapsed(0.0004)).toBe("0 ms");
    expect(formatElapsed(0.305)).toBe("305 ms");
    expect(formatElapsed(0.9996)).toBe("1.00 s");
    expect(formatElapsed(9.996)).toBe("10.0 s");
    expect(formatElapsed(64)).toBe("64.0 s");
  });
});
