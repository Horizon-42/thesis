/**
 * The Training views' shared wording of the judge's outcome, the crossing and the DA check, from the values the export
 * wrote (`stageA.ts`).
 */
import { describe, expect, it } from "vitest";
import { TRAINING_LOST_SEPARATION, TRAINING_OUTCOMES } from "../trainingSample";
import {
  checkMark, crossingText, decisionText, formatElapsed, landingText, replayText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT,
} from "../trainingText";
import { stageASample } from "./stageA";

describe("the outcome wording", () => {
  it("names every outcome the judge writes, in a tag and in words", () => {
    for (const outcome of TRAINING_OUTCOMES) {
      expect(TRAINING_OUTCOME_TAG[outcome].length).toBeGreaterThan(0);
      expect(TRAINING_OUTCOME_TEXT[outcome].length).toBeGreaterThan(0);
    }
    // and a window's end that is no judge's outcome (stage C: a loss of separation), in the same tags and words
    expect(Object.keys(TRAINING_OUTCOME_TAG).sort()).toEqual([...TRAINING_OUTCOMES, TRAINING_LOST_SEPARATION].sort());
    expect(Object.keys(TRAINING_OUTCOME_TEXT).sort()).toEqual([...TRAINING_OUTCOMES, TRAINING_LOST_SEPARATION].sort());
  });

  it("says where the threshold was crossed, the DA check's two values and the evaluation's three, with their marks", () => {
    const { replay } = stageASample().flights[0].closedLoop["2"];
    expect(crossingText(replay.crossing!)).toBe("19.3 m left of the centreline, 45.0 m above the threshold");
    expect(decisionText(replay.crossing!.decision!)).toBe(
      "DA check failed · 19.3 m left of the centreline (cone half width 115.9 m) ✓ · 30.0 m above the glidepath ✗");
    expect(landingText(replay.crossing!.evaluation!)).toBe("evaluation failed · 19.3 m off the centreline (at most 22.9 m) ✓ · "
      + "29.9 m from LTP + TCH (-22 to +22 m) ✗ · 73.5 m/s (67.8–78.1 m/s) ✓");
    expect(replayText(replay)).toContain("the evaluation failed the crossing");
    expect(replayText(replay)).toContain("evaluation failed · 19.3 m off the centreline");
    expect(replayText({ ...replay, crossing: null })).toContain("no threshold crossing");
    expect(replayText({ ...replay, crossing: { ...replay.crossing!, decision: null } })).toContain("no DA check");
    expect(replayText({ ...replay, crossing: { ...replay.crossing!, evaluation: null } })).not.toContain("evaluation failed ·");
  });

  it("writes an elapsed time in the unit it rounds to", () => {
    expect(formatElapsed(0.0894)).toBe("89 ms");
    expect(formatElapsed(0.9996)).toBe("1.00 s");
    expect(formatElapsed(48)).toBe("48.0 s");
    expect(checkMark(true) + checkMark(false)).toBe("✓✗");
  });
});
