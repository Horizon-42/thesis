/**
 * The Training views' shared wording of the judge's outcome, the crossing and the DA check, from the values the export
 * wrote (`stageA.ts`).
 */
import { describe, expect, it } from "vitest";
import { TRAINING_OUTCOMES } from "../trainingSample";
import {
  checkMark, crossingText, decisionText, formatElapsed, replayText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT,
} from "../trainingText";
import { stageASample } from "./stageA";

describe("the outcome wording", () => {
  it("names every outcome the judge writes, in a tag and in words", () => {
    for (const outcome of TRAINING_OUTCOMES) {
      expect(TRAINING_OUTCOME_TAG[outcome].length).toBeGreaterThan(0);
      expect(TRAINING_OUTCOME_TEXT[outcome].length).toBeGreaterThan(0);
    }
    expect(Object.keys(TRAINING_OUTCOME_TAG).sort()).toEqual([...TRAINING_OUTCOMES].sort());
  });

  it("says where the threshold was crossed and the DA check's two values with their marks", () => {
    const { replay } = stageASample().flights[0].closedLoop["2"];
    expect(crossingText(replay.crossing!)).toBe("15.5 m right of the centreline, 41.8 m above the threshold");
    expect(decisionText(replay.crossing!.decision!)).toBe(
      "DA check failed · 15.5 m right of the centreline (cone half width 116.3 m) ✓ · 26.8 m above the glidepath ✗");
    expect(replayText(replay)).toContain("unstable at minimums");
    expect(replayText({ ...replay, crossing: null })).toContain("no threshold crossing");
    expect(replayText({ ...replay, crossing: { ...replay.crossing!, decision: null } })).toContain("no DA check");
  });

  it("writes an elapsed time in the unit it rounds to", () => {
    expect(formatElapsed(0.0894)).toBe("89 ms");
    expect(formatElapsed(0.9996)).toBe("1.00 s");
    expect(formatElapsed(48)).toBe("48.0 s");
    expect(checkMark(true) + checkMark(false)).toBe("✓✗");
  });
});
